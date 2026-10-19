// dsv4_load.h — checkpoint -> device weight-struct helpers, shared by anything that builds the
// 43-layer model (currently src/engine.cu).
//
// These are lifted VERBATIM from src/decode.cu, which still carries its own private copies. That
// duplication is deliberate and temporary: decode.cu is the measurement harness every LOOP_LOG
// finding was taken with, and de-duplicating it now would mean re-gating the whole frozen suite --
// a ~10-minute checkpoint load per point -- to prove a copy-paste was faithful. Instead
// tests/gate_engine.cu proves the engine built from THIS header emits the same tokens as decode.cu
// on the same prompt, which is the property that actually matters. When the suite is next re-run
// from scratch, delete decode.cu's copies and include this instead.
#pragma once
#include "weight_store.h"
#include "deepseek_v4.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <string>
#include <vector>

namespace dsv4load {

#define DSV4L_CU(x) do{ cudaError_t e_=(x); if(e_){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)

// "model.layers.0.x" -> "layers.0.x": the store is keyed without the HF prefix.
inline std::string key_map(const std::string& in) {
    std::string s = in;
    if (s.rfind("model.", 0) == 0) s = s.substr(6);
    return s;
}

static __global__ void k_deq_e8m0(float* o, const uint8_t* in, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) o[i] = exp2f((float)in[i] - 127.f);
}
static __global__ void k_deq_bf16(float* o, const __nv_bfloat16* in, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) o[i] = __bfloat162float(in[i]);
}
static __global__ void k_deq_fp8_blk(float* o, const uint8_t* w, const uint8_t* sc, int rows, int cols, int blk) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * cols) return;
    int r = i / cols, c = i % cols;
    __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)w[i], __NV_E4M3);
    float wv = __half2float(*reinterpret_cast<__half*>(&hr));
    int scw = cols / blk;
    float sv = exp2f((float)sc[(size_t)(r / blk) * scw + c / blk] - 127.f);
    o[i] = wv * sv;
}
static __global__ void k_embed(float* h, const __nv_bfloat16* emb, const int* ids, int s, int dim) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)s * dim) return;
    int t = i / dim, j = i % dim;
    h[i] = __bfloat162float(emb[(size_t)ids[t] * dim + j]);
}
static __global__ void k_hc_expand(float* out, const float* h, int s, int hc, int dim) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)s * hc * dim) return;
    int t = i / (hc * dim), j = i % dim;
    out[i] = h[(size_t)t * dim + j];
}

// Dequantises on demand and owns every allocation it makes. `mark`/`release` bound a scope whose
// temporaries can be freed; the engine keeps everything, so it never releases.
struct Loader {
    st::WeightStore& W;
    std::vector<void*> allocs;
    explicit Loader(st::WeightStore& w) : W(w) {}
    ~Loader() { for (void* p : allocs) cudaFree(p); }
    size_t mark() { return allocs.size(); }
    void release(size_t m) { for (size_t i = m; i < allocs.size(); ++i) cudaFree(allocs[i]); allocs.resize(m); }
    const uint8_t* raw(const std::string& n) { return W.dev<uint8_t>(n); }
    const float*   f32(const std::string& n) { return W.dev<float>(n); }
    float* alloc(size_t nb) { void* p; DSV4L_CU(cudaMalloc(&p, nb)); allocs.push_back(p); return (float*)p; }
    const float* scale(const std::string& n) {
        auto& t = W.get(n); size_t ne = t.numel(); float* o = alloc(ne * 4);
        k_deq_e8m0<<<(ne + 255) / 256, 256>>>(o, (const uint8_t*)t.dev, ne); return o;
    }
    const float* bf16(const std::string& n) {
        auto& t = W.get(n); size_t ne = t.numel(); float* o = alloc(ne * 4);
        k_deq_bf16<<<(ne + 255) / 256, 256>>>(o, (const __nv_bfloat16*)t.dev, ne); return o;
    }
    const float* wo_a(const std::string& wn, const std::string& sn) {
        auto& t = W.get(wn); int rows = t.shape[0], cols = t.shape[1]; size_t ne = (size_t)rows * cols;
        float* o = alloc(ne * 4);
        k_deq_fp8_blk<<<(ne + 255) / 256, 256>>>(o, (const uint8_t*)t.dev, (const uint8_t*)W.get(sn).dev, rows, cols, 128);
        return o;
    }
};

inline const float* up_f(const std::vector<float>& v, std::vector<void*>& keep) {
    void* d; DSV4L_CU(cudaMalloc(&d, v.size() * 4));
    DSV4L_CU(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    keep.push_back(d);
    return (const float*)d;
}
inline std::vector<float> stride_rows(const std::vector<float>& in, int s, int half, int ratio) {
    int ng = s / ratio; if (ng < 1) ng = 1;
    std::vector<float> o((size_t)ng * half);
    for (int g = 0; g < ng; ++g) for (int j = 0; j < half; ++j) o[(size_t)g * half + j] = in[(size_t)(g * ratio) * half + j];
    return o;
}

} // namespace dsv4load
