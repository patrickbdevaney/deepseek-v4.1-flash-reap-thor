// engram_store.h — the Engram row store: n-gram hashing on the host at sampling time, and a
// clean-room io_uring O_DIRECT reader that pulls the 48 rows a token needs from the packed row
// files on NVMe into a cudaHostRegister'd arena the GPU reads directly.
//
// WHY NOT THE OBVIOUS ALTERNATIVES (PLAN.md "Speed is set by the dense path"; engram_off_device):
//   * resident tables: 188.8 GiB. Does not fit next to 101 GiB of experts on a 117 GiB envelope.
//   * mmap: every lookup is a random 4 KiB page fault on the request path, and the page cache it
//     fills is the same unified pool the weights live in -- the Thor double-copy OOM.
//   * pinned host copy: same pool, same size problem.
//   Rows have ~0% reuse (MiaAI runs with no cache at all), so the right shape is: compute the 48
//   row ids as soon as the token is sampled (they depend only on token ids), issue 48 small
//   O_DIRECT reads as one io_uring batch, and let the GPU start layer 0 while they land. Layer 1
//   needs rows ~1 layer later, layer 14 ~14 layers later. Measured on this NVMe: 1.13 ms p50 for
//   the 48-row batch (ops research note), against ~2.5 ms per layer at 10 tok/s.
//
// GRAPH CAPTURE. Decode is one CUDA graph. The graph contains, before each Engram layer's kernels,
// cuStreamWaitValue32(flag[l] == 1), and after them cuStreamWriteValue32(flag[l] = 0). The host
// sets flag[l] = 1 when layer l's reads have completed. The graph never changes shape, the host
// never blocks the launch, and a step can never read rows from the previous token: the flag is
// re-armed by the consumer, and the host only overwrites the arena after it has synchronised on
// the step's sampled token (it needs that token to compute the next hashes anyway).
//
// FILES:
//   engram_L1.v41rows / engram_L14.v41rows   llamacpp/ENGRAM_ROWFILE.md -- THE shared format (llama.cpp
//                              reads the same files), built by ops/s09_engram_rowfile.py: 4 KiB
//                              header, then rows packed back to back (264 B FP8 + E8M0). A row
//                              crosses at most one 512 B sector, so every read is one aligned 1 KiB.
//                              value_dtype 2 (MXFP6, 200 B) is a CUDA-only lossy variant written by
//                              tools/pack_engram_rows.py; llama.cpp refuses it, as it should.
//   engram_tokmap.bin          int32[129280] compressed token map; then pad id, multipliers,
//                              primes, offsets as int64 (see tools/engram_meta.py)
#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct EngramHashTables {
    std::vector<int32_t> tokmap;          // [vocab] compressed id
    int64_t pad = 0;                      // tokmap[engram_pad_id]
    int64_t mult[2][4];                   // [layer][lookback]
    int64_t prime[2][24];                 // [layer][(ngram-2)*8 + head]
    int64_t offset[2][24];
    bool load(const std::string& path, std::string* err);
};

// Host n-gram state: the compressed-id history (DEAD = -1 for image tokens), exactly
// engram.py NgramHashState. hash() appends tokens at [pos0, pos0+n) and emits row ids
// ids[n][2][24] (int64, absolute rows in each layer's table).
class EngramHasher {
public:
    explicit EngramHasher(const EngramHashTables& t) : t_(t) {}
    void truncate(int len) { if ((int)hist_.size() > len) hist_.resize(len); }
    void hash(const int* tokens, const uint8_t* is_image, int n, int pos0, int64_t* ids);
    int size() const { return (int)hist_.size(); }
private:
    const EngramHashTables& t_;
    std::vector<int64_t> hist_;
};

struct EngramStoreConfig {
    std::string dir;                      // directory holding engram_L1.v41rows / engram_L14.v41rows
    std::string tokmap;                   // engram_tokmap.bin (tools/engram_meta.py); "" = <dir>/engram_tokmap.bin
    int max_tokens = 128;                 // largest batch fetched at once (a prefill chunk)
    int queue_depth = 1024;               // io_uring SQ entries
    int read_gran = 512;                  // O_DIRECT granule (logical block size of the device); must be 512
    bool no_odirect = false;              // tests on tmpfs: O_DIRECT is unsupported there
    bool host_only = false;               // tests: plain host memory, no CUDA calls at all
};

class EngramStore {
public:
    ~EngramStore();
    bool open(const EngramStoreConfig& cfg, std::string* err);
    int  format() const { return fmt_; }  // 0 = FP8+E8M0 (264 B), 1 = MXFP6+E8M0 (200 B)
    int64_t n_rows(int l) const { return n_rows_[l]; }
    int  row_bytes() const { return row_bytes_; }
    int  capacity() const { return cfg_.max_tokens; }   // arena slots / 24: where the offset table starts
    const EngramHashTables& tables() const { return tabs_; }

    // Device pointers the kernels and the graph use. Arena layout per layer: max_tokens*24 slots of
    // 1 KiB, then a uint16 in-slot offset table (same order). flag: uint32 in mapped host memory.
    const uint8_t* arena_dev(int l) const { return arena_dev_[l]; }
    const uint8_t* arena_host(int l) const { return arena_host_[l]; }
    uint32_t* flag_dev(int l) const { return flag_dev_[l]; }
    volatile uint32_t* flag_host(int l) const { return flag_host_[l]; }

    // Submit the reads for n tokens' row ids (ids[n][2][24]) without waiting.
    void submit(const int64_t* ids, int n);
    // Reap completions; set flag[l] = 1 as each layer's reads finish. Returns false on an I/O error.
    bool complete(std::string* err);
    // submit + complete, for prefill and tests.
    bool fetch(const int64_t* ids, int n, std::string* err) { submit(ids, n); return complete(err); }

    // Latency telemetry (last fetch), microseconds.
    double last_us() const { return last_us_; }

private:
    bool setup_ring(int entries, std::string* err);
    EngramStoreConfig cfg_;
    EngramHashTables tabs_;
    int fmt_ = 0, row_bytes_ = 264;
    int64_t data_off_ = 4096;
    std::vector<uint16_t> need_[2];       // bytes each slot's read must return (short-read check)
    int64_t n_rows_[2] = {0, 0};
    int fd_[2] = {-1, -1};
    uint8_t* arena_host_[2] = {nullptr, nullptr};
    uint8_t* arena_dev_[2] = {nullptr, nullptr};
    size_t arena_bytes_ = 0;
    volatile uint32_t* flag_host_[2] = {nullptr, nullptr};
    uint32_t* flag_dev_[2] = {nullptr, nullptr};
    void* flag_alloc_ = nullptr;
    // io_uring (raw syscalls; no liburing on this box)
    int ring_fd_ = -1;
    void* sq_ptr_ = nullptr; size_t sq_sz_ = 0;
    void* cq_ptr_ = nullptr; size_t cq_sz_ = 0;
    void* sqes_ = nullptr; size_t sqes_sz_ = 0;
    unsigned *sq_head_ = nullptr, *sq_tail_ = nullptr, *sq_mask_ = nullptr, *sq_array_ = nullptr;
    unsigned *cq_head_ = nullptr, *cq_tail_ = nullptr, *cq_mask_ = nullptr;
    void* cqes_ = nullptr;
    unsigned sq_entries_ = 0;
    // in-flight bookkeeping
    int pending_[2] = {0, 0};
    int queued_ = 0;                      // SQEs written but not yet submitted
    std::vector<std::pair<int, int>> backlog_;   // (layer, slot) not yet queued (SQ full)
    std::vector<int64_t> backlog_ids_;
    int64_t t0_ns_ = 0;
    double last_us_ = 0;
    bool err_ = false;
    std::string errmsg_;
    void queue_read(int layer, int slot, int64_t row);
    void flush();
};
