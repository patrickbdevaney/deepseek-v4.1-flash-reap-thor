// safetensors.h — minimal mmap reader for safetensors files (header-only).
// Format: [u64 LE header_len][JSON header][raw tensor bytes]. Offsets are relative to data start.
// Supports the dtypes present in the Gemma-4 NVFP4 checkpoint: U8 (packed FP4), F8_E4M3, BF16, F16, F32, I32, I64.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace st {

struct Tensor {
    std::string dtype;                 // "U8","F8_E4M3","BF16","F16","F32","I32",...
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;     // pointer into mmap
    size_t nbytes = 0;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};

// --- tiny JSON parser, just enough for safetensors headers ---
struct JsonP {
    const char* p; const char* end;
    JsonP(const char* s, size_t n) : p(s), end(s + n) {}
    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) ++p; }
    bool eat(char c) { ws(); if (p < end && *p == c) { ++p; return true; } return false; }
    std::string str() {
        ws(); if (*p != '"') throw std::runtime_error("json: expected string");
        ++p; std::string s;
        while (p < end && *p != '"') { if (*p == '\\') { ++p; } s.push_back(*p); ++p; }
        ++p; return s;
    }
    double num() {
        ws(); char* e; double v = strtod(p, &e); p = e; return v;
    }
    void skip_value();  // forward
};

inline void JsonP::skip_value() {
    ws();
    if (*p == '"') { str(); }
    else if (*p == '{') { ++p; ws(); if (eat('}')) return; do { str(); eat(':'); skip_value(); } while (eat(',')); eat('}'); }
    else if (*p == '[') { ++p; ws(); if (eat(']')) return; do { skip_value(); } while (eat(',')); eat(']'); }
    else { while (p < end && *p != ',' && *p != '}' && *p != ']') ++p; }
}

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path) : path_(path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("open failed: " + path);
        struct stat sb; fstat(fd_, &sb); filesize_ = sb.st_size;
        base_ = (const uint8_t*)mmap(nullptr, filesize_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
        uint64_t hlen; memcpy(&hlen, base_, 8);
        const char* hdr = (const char*)base_ + 8;
        const uint8_t* data_start = base_ + 8 + hlen;
        data_start_ = data_start; data_bytes_ = filesize_ - (8 + hlen); data_file_off_ = 8 + hlen;
        parse_header(hdr, hlen, data_start);
    }
    ~SafeTensors() { if (base_ && base_ != MAP_FAILED) munmap((void*)base_, filesize_); if (fd_ >= 0) close(fd_); }

    const uint8_t* dataStart() const { return data_start_; }
    size_t dataBytes() const { return data_bytes_; }
    const std::string& path() const { return path_; }
    size_t dataFileOffset() const { return data_file_off_; }   // byte offset of data blob within the file
    int fd() const { return fd_; }
    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const Tensor& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
        return it->second;
    }
    const std::unordered_map<std::string, Tensor>& all() const { return tensors_; }
    size_t count() const { return tensors_.size(); }

private:
    void parse_header(const char* hdr, size_t hlen, const uint8_t* data_start) {
        JsonP j(hdr, hlen);
        if (!j.eat('{')) throw std::runtime_error("header: expected object");
        j.ws(); if (j.eat('}')) return;
        do {
            std::string name = j.str(); j.eat(':');
            if (name == "__metadata__") { j.skip_value(); continue; }
            j.eat('{');
            Tensor t; int64_t off0 = 0, off1 = 0;
            do {
                std::string key = j.str(); j.eat(':');
                if (key == "dtype") t.dtype = j.str();
                else if (key == "shape") {
                    j.eat('['); j.ws();
                    if (!j.eat(']')) { do { t.shape.push_back((int64_t)j.num()); } while (j.eat(',')); j.eat(']'); }
                } else if (key == "data_offsets") {
                    j.eat('['); off0 = (int64_t)j.num(); j.eat(','); off1 = (int64_t)j.num(); j.eat(']');
                } else j.skip_value();
            } while (j.eat(','));
            j.eat('}');
            t.data = data_start + off0; t.nbytes = (size_t)(off1 - off0);
            tensors_.emplace(std::move(name), std::move(t));
        } while (j.eat(','));
    }
    std::string path_;
    int fd_ = -1; size_t filesize_ = 0; const uint8_t* base_ = nullptr;
    const uint8_t* data_start_ = nullptr; size_t data_bytes_ = 0, data_file_off_ = 0;
    std::unordered_map<std::string, Tensor> tensors_;
};

// --- read whole file into a std::string (for index.json) ---
inline std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("open failed: " + path);
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s(n, '\0');
    if (fread(&s[0], 1, n, f) != (size_t)n) { fclose(f); throw std::runtime_error("read failed: " + path); }
    fclose(f); return s;
}

// Multi-shard reader driven by model.safetensors.index.json (weight_map: name -> shard file).
// Lazily mmaps each shard on first touch; get(name) resolves across shards. Header-only parse is cheap
// (does not fault tensor bytes) so enumeration/validation runs at tiny RSS even for a 100GB checkpoint.
class ShardedSafeTensors {
public:
    // key_map: optional normalization applied to each checkpoint name -> internal name.
    // only_prefix: if set, load ONLY tensors whose raw name contains it (e.g. "mtp."); shards that don't
    //   exist on disk (partial download) are skipped rather than throwing. For extracting a head from a repo.
    explicit ShardedSafeTensors(const std::string& dir,
                                std::string (*key_map)(const std::string&) = nullptr,
                                const char* only_prefix = nullptr)
        : dir_(dir) {
        std::string idx = read_file(dir + "/model.safetensors.index.json");
        JsonP j(idx.data(), idx.size());
        if (!j.eat('{')) throw std::runtime_error("index: expected object");
        do {
            std::string k = j.str(); j.eat(':');
            if (k == "weight_map") {
                j.eat('{'); j.ws();
                if (!j.eat('}')) {
                    do { std::string name = j.str(); j.eat(':'); std::string file = j.str();
                         weight_map_.emplace(std::move(name), std::move(file)); } while (j.eat(','));
                    j.eat('}');
                }
            } else j.skip_value();
        } while (j.eat(','));
        auto keep = [&](const std::string& n){ return !only_prefix || n.find(only_prefix) != std::string::npos; };
        // Open shards referenced by kept tensors; skip shards that don't exist (partial download).
        for (auto& kv : weight_map_) {
            if (!keep(kv.first)) continue;
            const std::string& file = kv.second;
            if (!shards_.count(file)) { try { shards_[file].reset(new SafeTensors(dir_ + "/" + file)); }
                                        catch (...) { shards_.erase(file); } }
        }
        // Build the resolved name -> Tensor map (applying key_map).
        for (auto& kv : weight_map_) {
            if (!keep(kv.first) || !shards_.count(kv.second)) continue;
            const Tensor& t = shards_[kv.second]->get(kv.first);
            std::string internal = key_map ? key_map(kv.first) : kv.first;
            tensors_.emplace(std::move(internal), t);
        }
    }

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const Tensor& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
        return it->second;
    }
    const std::unordered_map<std::string, Tensor>& all() const { return tensors_; }
    size_t count() const { return tensors_.size(); }
    size_t shardCount() const { return shards_.size(); }

    // Mmap regions of each shard's tensor-data blob — for cudaHostRegister (zero-copy device access on
    // unified memory). Returns [(host_ptr, byte_len)] per shard.
    std::vector<std::pair<const void*, size_t>> shardRegions() const {
        std::vector<std::pair<const void*, size_t>> r;
        for (auto& kv : shards_) r.emplace_back(kv.second->dataStart(), kv.second->dataBytes());
        return r;
    }
    const std::unordered_map<std::string, std::unique_ptr<SafeTensors>>& shards() const { return shards_; }

private:
    std::string dir_;
    std::unordered_map<std::string, std::string> weight_map_;            // name -> shard file
    std::unordered_map<std::string, std::unique_ptr<SafeTensors>> shards_;
    std::unordered_map<std::string, Tensor> tensors_;                    // internal name -> Tensor
};

} // namespace st
