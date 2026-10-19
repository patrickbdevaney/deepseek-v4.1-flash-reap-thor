// weight_store.h — load the full sharded checkpoint into GPU-accessible memory on integrated Jetson.
// Single-copy: pread each shard's data blob straight into a cudaHostAlloc(Mapped) buffer (never faults the
// mmap data pages), then every tensor's device pointer = shard_base_dev + offset_in_blob. No mmap+device
// doubling -> no OOM. mmaps stay lazy (headers only) for metadata. See ROADMAP Phase A.
#pragma once
#include "safetensors.h"
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <unistd.h>
#include <cstdlib>

namespace st {

struct DevTensor { const void* dev = nullptr; std::string dtype; std::vector<int64_t> shape; size_t nbytes = 0;
                   int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; } };

class WeightStore {
public:
    WeightStore(const std::string& dir, std::string (*key_map)(const std::string&) = nullptr,
                const char* only_prefix = nullptr) {
        ShardedSafeTensors S(dir, key_map, only_prefix);
        // 1. load each shard's data blob via pread (single copy, no mmap fault).
        //
        // THE ALLOCATOR IS A BANDWIDTH DECISION (LOOP_LOG Finding 44). This used cudaHostAlloc with
        // cudaHostAllocMapped unconditionally, and every byte-carrying kernel in the engine landed
        // in a 142-183 GB/s band against a 240 GB/s roofline — a uniform ~70% across kernels with
        // nothing else in common, which is never a kernel property. tools/alloc_probe measures the
        // same streaming kernel on the same buffer size out of four allocators on this box:
        //
        //   cudaMalloc (device)                233.8 GB/s stream / 235.0 strided
        //   cudaHostAlloc Mapped               180.2 / 194.3       <- what this was
        //   cudaMallocManaged                  181.8 / 234.8
        //   cudaMallocManaged + PreferredLoc   231.3 / 235.1       <- what this is now
        //
        // The kernels were never at 70% of achievable; they were at ~90% of a ceiling the weights
        // could not exceed, and ROOFLINE.md's "% of achievable" column was measured against a
        // buffer the model does not live in. Managed memory with the preferred location set to the
        // device and the range prefetched is device-resident on this integrated part, so it keeps
        // the single-copy property (no mmap+device doubling, no OOM) at device bandwidth.
        //
        // DSV4_WEIGHTS=mapped restores the old allocator for A/B; a failed managed allocation falls
        // back to it automatically rather than dying after a ten-minute load.
        const bool want_managed = [](){ const char* e=getenv("DSV4_WEIGHTS");
                                        return !(e && std::string(e)=="mapped"); }();
        int dev_id = 0; cudaGetDevice(&dev_id);
        std::unordered_map<const SafeTensors*, void*> base;   // shard -> device-accessible base
        for (auto& kv : S.shards()) {
            SafeTensors* sh = kv.second.get(); size_t nb = sh->dataBytes();
            void* buf = nullptr; bool managed = false;
            if (want_managed && cudaMallocManaged(&buf, nb) == cudaSuccess) managed = true;
            if (!managed) {
                buf = nullptr;
                cudaError_t e = cudaHostAlloc(&buf, nb, cudaHostAllocMapped);
                if (e != cudaSuccess) throw std::runtime_error(std::string("shard alloc failed: ") + cudaGetErrorString(e));
            }
            size_t got = 0; off_t off = (off_t)sh->dataFileOffset();
            while (got < nb) { ssize_t r = pread(sh->fd(), (char*)buf + got, nb - got, off + got);
                if (r <= 0) throw std::runtime_error("pread shard failed: " + sh->path()); got += (size_t)r; }
            // drop the file pages from the page cache — we've copied them to `buf`. Reclaims ~96 GiB of
            // otherwise-"used" reclaimable cache that Tegra cudaMalloc won't auto-evict (memory headroom).
            posix_fadvise(sh->fd(), off, nb, POSIX_FADV_DONTNEED);
            void* dev = buf;
            if (managed) {
                // pread faulted every page in on the HOST side; without the prefetch the first
                // device touch of each page migrates it one at a time, mid-decode.
                cudaMemLocation loc{}; loc.type = cudaMemLocationTypeDevice; loc.id = dev_id;
                cudaMemAdvise(buf, nb, cudaMemAdviseSetPreferredLocation, loc);
                cudaMemPrefetchAsync(buf, nb, loc, 0, 0);
                cudaStreamSynchronize(0);
                managed_.push_back(buf);
            } else {
                if (cudaHostGetDevicePointer(&dev, buf, 0) != cudaSuccess) throw std::runtime_error("getDevicePointer failed");
                pinned_.push_back(buf);
            }
            base[sh] = dev; host_base_[sh] = sh->dataStart(); dev_base_[sh] = dev;
            loaded_ += nb; managed_any_ = managed_any_ || managed;
        }
        // 2. resolve every tensor's device pointer = shard_dev_base + (t.data - shard_host_start)
        for (auto& kv : S.all()) {
            const Tensor& t = kv.second;
            const SafeTensors* owner = nullptr;
            for (auto& b : host_base_) { const uint8_t* hb = b.second;
                if (t.data >= hb && t.data < hb + b.first->dataBytes()) { owner = b.first; break; } }
            if (!owner) throw std::runtime_error("tensor not in any shard: " + kv.first);
            size_t offset = (size_t)(t.data - host_base_[owner]);
            DevTensor d; d.dev = (const uint8_t*)dev_base_[owner] + offset; d.dtype = t.dtype; d.shape = t.shape; d.nbytes = t.nbytes;
            t_.emplace(kv.first, std::move(d));
        }
    }
    ~WeightStore() { for (void* p : pinned_) cudaFreeHost(p); for (void* p : managed_) cudaFree(p); }
    bool managed() const { return managed_any_; }

    bool has(const std::string& n) const { return t_.count(n) > 0; }
    const DevTensor& get(const std::string& n) const {
        auto it = t_.find(n); if (it == t_.end()) throw std::runtime_error("weight not found: " + n); return it->second; }
    template<class T> const T* dev(const std::string& n) const { return (const T*)get(n).dev; }
    size_t count() const { return t_.size(); }
    double loadedGiB() const { return loaded_ / 1073741824.0; }

private:
    std::unordered_map<std::string, DevTensor> t_;
    std::unordered_map<const SafeTensors*, const uint8_t*> host_base_;
    std::unordered_map<const SafeTensors*, void*> dev_base_;
    std::vector<void*> pinned_;
    std::vector<void*> managed_;
    bool managed_any_ = false;
    size_t loaded_ = 0;
};

} // namespace st
