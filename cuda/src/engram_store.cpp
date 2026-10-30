// engram_store.cpp — see include/engram_store.h. io_uring is driven with raw syscalls (clean room:
// written from the kernel ABI in <linux/io_uring.h>, no liburing; none is installed here).
#include "engram_store.h"
#include "v41_config.h"
#include <cuda_runtime.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>

// ------------------------------------------------------------------ io_uring plumbing
static int sys_setup(unsigned e, io_uring_params* p) { return (int)syscall(__NR_io_uring_setup, e, p); }
static int sys_enter(int fd, unsigned sub, unsigned minc, unsigned flags) {
    return (int)syscall(__NR_io_uring_enter, fd, sub, minc, flags, nullptr, 0);
}
static int64_t now_ns() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1000000000LL + ts.tv_nsec; }

bool EngramStore::setup_ring(int entries, std::string* err) {
    io_uring_params p; memset(&p, 0, sizeof p);
    ring_fd_ = sys_setup((unsigned)entries, &p);
    if (ring_fd_ < 0) { *err = std::string("io_uring_setup: ") + strerror(errno); return false; }
    sq_sz_ = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    cq_sz_ = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    const bool single = p.features & IORING_FEAT_SINGLE_MMAP;
    if (single) { if (cq_sz_ > sq_sz_) sq_sz_ = cq_sz_; cq_sz_ = sq_sz_; }
    sq_ptr_ = mmap(nullptr, sq_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_SQ_RING);
    if (sq_ptr_ == MAP_FAILED) { *err = "mmap sq ring"; return false; }
    cq_ptr_ = single ? sq_ptr_ : mmap(nullptr, cq_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_CQ_RING);
    if (cq_ptr_ == MAP_FAILED) { *err = "mmap cq ring"; return false; }
    sqes_sz_ = p.sq_entries * sizeof(io_uring_sqe);
    sqes_ = mmap(nullptr, sqes_sz_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_SQES);
    if (sqes_ == MAP_FAILED) { *err = "mmap sqes"; return false; }
    char* sq = (char*)sq_ptr_; char* cq = (char*)cq_ptr_;
    sq_head_ = (unsigned*)(sq + p.sq_off.head); sq_tail_ = (unsigned*)(sq + p.sq_off.tail);
    sq_mask_ = (unsigned*)(sq + p.sq_off.ring_mask); sq_array_ = (unsigned*)(sq + p.sq_off.array);
    cq_head_ = (unsigned*)(cq + p.cq_off.head); cq_tail_ = (unsigned*)(cq + p.cq_off.tail);
    cq_mask_ = (unsigned*)(cq + p.cq_off.ring_mask); cqes_ = cq + p.cq_off.cqes;
    sq_entries_ = p.sq_entries;
    // the two row files as fixed files: saves an fget/fput per read
    if (syscall(__NR_io_uring_register, ring_fd_, IORING_REGISTER_FILES, fd_, 2) < 0) {
        *err = std::string("register files: ") + strerror(errno); return false;
    }
    return true;
}

// .v41rows header (llamacpp/ENGRAM_ROWFILE.md, the one on-disk format shared with llama.cpp and
// built by ops/s09_engram_rowfile.py). Every field the contract lists is checked; mismatch is fatal.
struct V41RowsHeader {
    char magic[8]; uint32_t version, layer_id; uint64_t n_rows; uint32_t key_len, n_scales, row_bytes, value_dtype,
    scale_dtype; uint64_t data_off;
} __attribute__((packed));

bool EngramStore::open(const EngramStoreConfig& cfg, std::string* err) {
    cfg_ = cfg;
    if (cfg.read_gran < 512 || (cfg.read_gran & (cfg.read_gran - 1))) { *err = "read_gran must be a power of two >= 512"; return false; }
    if (!tabs_.load(cfg.tokmap.empty() ? cfg.dir + "/engram_tokmap.bin" : cfg.tokmap, err)) return false;
    for (int l = 0; l < 2; ++l) {
        const std::string p = cfg.dir + "/engram_L" + std::to_string(v41::ENGRAM_LAYERS[l]) + ".v41rows";
        fd_[l] = ::open(p.c_str(), O_RDONLY | (cfg.no_odirect ? 0 : O_DIRECT));
        if (fd_[l] < 0 && !cfg.no_odirect && errno == EINVAL) fd_[l] = ::open(p.c_str(), O_RDONLY);   // contract: buffered fallback
        if (fd_[l] < 0) { *err = "open " + p + ": " + strerror(errno); return false; }
        void* hb = nullptr;
        if (posix_memalign(&hb, 4096, 4096)) { *err = "header alloc"; return false; }
        const ssize_t got = pread(fd_[l], hb, 4096, 0);
        V41RowsHeader h; memcpy(&h, hb, sizeof h); free(hb);
        if (got != 4096 || memcmp(h.magic, "V41ENGR1", 8) != 0 || h.version != 1) { *err = p + ": not a v1 .v41rows file"; return false; }
        if ((int)h.layer_id != v41::ENGRAM_LAYERS[l]) { *err = p + ": layer_id mismatch"; return false; }
        if (h.key_len != 256 || h.n_scales != 8 || h.scale_dtype != 1 || h.data_off % 4096) { *err = p + ": unexpected key_len/n_scales/scale_dtype/data_off"; return false; }
        int fmt;
        if (h.value_dtype == 1 && h.row_bytes == h.key_len + h.n_scales) fmt = 0;                   // FP8 E4M3FN, 264 B
        else if (h.value_dtype == 2 && h.row_bytes == h.key_len * 6 / 8 + h.n_scales) fmt = 1;      // MXFP6 E3M2 (CUDA-only, lossy)
        else { *err = p + ": unknown value_dtype/row_bytes"; return false; }
        if (l == 1 && fmt != fmt_) { *err = "the two Engram files have different formats"; return false; }
        fmt_ = fmt; row_bytes_ = (int)h.row_bytes; data_off_ = (int64_t)h.data_off;
        n_rows_[l] = (int64_t)h.n_rows;
        int64_t need_rows = 0;
        for (int c = 0; c < v41::ENGRAM_COLS; ++c) need_rows = std::max<int64_t>(need_rows, tabs_.offset[l][c] + tabs_.prime[l][c]);
        if (n_rows_[l] < need_rows) { *err = p + ": n_rows does not cover the largest hash bucket"; return false; }
        struct stat sb; fstat(fd_[l], &sb);
        if (sb.st_size < data_off_ + n_rows_[l] * row_bytes_) { *err = p + " is truncated"; return false; }
    }
    // arenas: slots of 1 KiB (a <= 1 KiB aligned read always covers a row), then uint16 offsets
    if (cfg.read_gran > 512) { *err = "read_gran > 512 does not fit a row window in a 1 KiB slot"; return false; }
    const size_t slots = (size_t)cfg.max_tokens * v41::ENGRAM_COLS;
    for (int l = 0; l < 2; ++l) need_[l].assign(slots, 0);
    arena_bytes_ = slots * 1024 + slots * 2;
    arena_bytes_ = (arena_bytes_ + 4095) & ~(size_t)4095;
    for (int l = 0; l < 2; ++l) {
        if (posix_memalign((void**)&arena_host_[l], 4096, arena_bytes_)) { *err = "arena alloc"; return false; }
        memset(arena_host_[l], 0, arena_bytes_);
        if (cfg.host_only) continue;
        if (cudaHostRegister(arena_host_[l], arena_bytes_, cudaHostRegisterMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**)&arena_dev_[l], arena_host_[l], 0) != cudaSuccess) {
            *err = "cudaHostRegister arena failed"; return false;
        }
    }
    if (cfg.host_only) { if (posix_memalign(&flag_alloc_, 256, 256)) { *err = "flag alloc"; return false; } }
    else if (cudaHostAlloc(&flag_alloc_, 256, cudaHostAllocMapped) != cudaSuccess) { *err = "flag alloc"; return false; }
    memset(flag_alloc_, 0, 256);
    for (int l = 0; l < 2; ++l) {
        flag_host_[l] = (volatile uint32_t*)((char*)flag_alloc_ + 64 * l);
        if (cfg.host_only) continue;
        void* d = nullptr; cudaHostGetDevicePointer(&d, (void*)flag_host_[l], 0); flag_dev_[l] = (uint32_t*)d;
    }
    return setup_ring(cfg.queue_depth, err);
}

EngramStore::~EngramStore() {
    for (int l = 0; l < 2; ++l) {
        if (arena_host_[l]) { if (!cfg_.host_only) cudaHostUnregister(arena_host_[l]); free(arena_host_[l]); }
        if (fd_[l] >= 0) close(fd_[l]);
    }
    if (flag_alloc_) { if (cfg_.host_only) free(flag_alloc_); else cudaFreeHost(flag_alloc_); }
    if (sqes_ && sqes_ != MAP_FAILED) munmap(sqes_, sqes_sz_);
    if (cq_ptr_ && cq_ptr_ != sq_ptr_ && cq_ptr_ != MAP_FAILED) munmap(cq_ptr_, cq_sz_);
    if (sq_ptr_ && sq_ptr_ != MAP_FAILED) munmap(sq_ptr_, sq_sz_);
    if (ring_fd_ >= 0) close(ring_fd_);
}

// at most sq_entries_ reads in flight (the CQ is 2x that); the rest wait in the backlog, in order
void EngramStore::queue_read(int layer, int slot, int64_t row) {
    if (bl_pos_ < backlog_.size() || pending_[0] + pending_[1] >= (int)sq_entries_ ||
        *sq_tail_ - __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE) >= sq_entries_) {
        backlog_.push_back({layer, slot}); backlog_ids_.push_back(row); ++bl_left_[layer]; return;
    }
    issue_read(layer, slot, row);
}

void EngramStore::refill() {
    while (bl_pos_ < backlog_.size() && pending_[0] + pending_[1] < (int)sq_entries_ &&
           *sq_tail_ - __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE) < sq_entries_) {
        const int layer = backlog_[bl_pos_].first;
        issue_read(layer, backlog_[bl_pos_].second, backlog_ids_[bl_pos_]);
        --bl_left_[layer]; ++bl_pos_;
    }
    if (bl_pos_ == backlog_.size() && bl_pos_) { backlog_.clear(); backlog_ids_.clear(); bl_pos_ = 0; }
    flush();
}

void EngramStore::issue_read(int layer, int slot, int64_t row) {
    const unsigned tail = *sq_tail_;
    // rows are packed (no page padding): a <= 264 B row crosses at most one 512 B boundary, so the
    // aligned window is <= 1 KiB at read_gran 512 -- one arena slot
    const int64_t off = data_off_ + row * row_bytes_;
    const int64_t g = cfg_.read_gran;
    const int64_t a = off / g * g, b = std::min<int64_t>((off + row_bytes_ + g - 1) / g * g, a + 1024);
    uint8_t* dst = arena_host_[layer] + (size_t)slot * 1024;
    reinterpret_cast<uint16_t*>(arena_host_[layer] + (size_t)cfg_.max_tokens * v41::ENGRAM_COLS * 1024)[slot] =
        (uint16_t)(off - a);
    need_[layer][slot] = (uint16_t)(off - a + row_bytes_);
    const unsigned idx = tail & *sq_mask_;
    io_uring_sqe* sqe = (io_uring_sqe*)sqes_ + idx;
    memset(sqe, 0, sizeof *sqe);
    sqe->opcode = IORING_OP_READ;
    sqe->flags = IOSQE_FIXED_FILE;
    sqe->fd = layer;                                          // index into the registered files
    sqe->off = (uint64_t)a;
    sqe->addr = (uint64_t)(uintptr_t)dst;
    sqe->len = (uint32_t)(b - a);
    sqe->user_data = ((uint64_t)layer << 32) | (uint32_t)slot;
    sq_array_[idx] = idx;
    __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);
    ++queued_; ++pending_[layer];
}

void EngramStore::flush() {
    if (!queued_) return;
    int r = sys_enter(ring_fd_, (unsigned)queued_, 0, 0);
    if (r < 0) { err_ = true; errmsg_ = std::string("io_uring_enter: ") + strerror(errno); }
    queued_ = 0;
}

void EngramStore::submit(const int64_t* ids, int n) {
    t0_ns_ = now_ns();
    err_ = false;
    for (int l = 0; l < 2; ++l) { *flag_host_[l] = 0; pending_[l] = 0; }
    // layer 1 first: it is needed ~13 layers before layer 14
    for (int l = 0; l < 2; ++l)
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < v41::ENGRAM_COLS; ++c)
                queue_read(l, i * v41::ENGRAM_COLS + c, ids[((size_t)i * 2 + l) * v41::ENGRAM_COLS + c]);
    flush();
}

bool EngramStore::complete(std::string* err) {
    while (pending_[0] + pending_[1] > 0 || bl_pos_ < backlog_.size()) {
        refill();                                 // keep sq_entries_ reads in flight while any wait
        if (pending_[0] + pending_[1] == 0) continue;
        int r = sys_enter(ring_fd_, 0, 1, IORING_ENTER_GETEVENTS);
        if (r < 0 && errno != EINTR) { if (err) *err = std::string("io_uring wait: ") + strerror(errno); return false; }
        unsigned head = *cq_head_;
        const unsigned tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
        while (head != tail) {
            const io_uring_cqe* cqe = (const io_uring_cqe*)cqes_ + (head & *cq_mask_);
            const int layer = (int)(cqe->user_data >> 32);
            const int slot = (int)(uint32_t)cqe->user_data;
            if (cqe->res < 0) { err_ = true; errmsg_ = std::string("engram read: ") + strerror(-cqe->res); }
            else if (cqe->res < need_[layer][slot]) { err_ = true; errmsg_ = "engram read: short read"; }
            --pending_[layer];
            if (pending_[layer] == 0 && bl_left_[layer] == 0 && !err_) {
                __atomic_store_n((uint32_t*)flag_host_[layer], 1u, __ATOMIC_RELEASE);
                layer_us_[layer] = (now_ns() - t0_ns_) / 1000.0;
            }
            ++head;
        }
        __atomic_store_n(cq_head_, head, __ATOMIC_RELEASE);
    }
    last_us_ = (now_ns() - t0_ns_) / 1000.0;
    if (err_ && err) *err = errmsg_;
    return !err_;
}
