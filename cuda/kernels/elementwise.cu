// elementwise.cu — RMSNorm / RoPE / gelu_tanh / residual / softcap (fp32).
#include "elementwise.h"
#include <cuda_bf16.h>

__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    unsigned int u = (unsigned int)h << 16; float f; memcpy(&f, &u, 4); return f;
}

// one block per row; blockDim threads cooperate on the reduction
__global__ void rmsnorm_kernel(float* out, const float* x, const uint16_t* w, int dim, float eps) {
    int row = blockIdx.x; const float* xr = x + (size_t)row * dim; float* orr = out + (size_t)row * dim;
    __shared__ float ssum[256];
    float local = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float v = xr[i]; local += v * v; }
    ssum[threadIdx.x] = local; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s]; __syncthreads(); }
    float inv = rsqrtf(ssum[0] / dim + eps);   // (mean+eps)^-0.5 ; pow(-0.5)==rsqrt
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xr[i] * inv;
        if (w) v *= bf16_to_f32(w[i]);
        orr[i] = v;
    }
}
void rmsnorm(float* out, const float* x, const uint16_t* w, int rows, int dim, float eps, cudaStream_t s) {
    rmsnorm_kernel<<<rows, 256, 0, s>>>(out, x, w, dim, eps);
}

// x[rows, n_heads, head_dim]; rotate first rot_dim dims. cos/sin: [rows, rot_dim/2]
__global__ void rope_kernel(float* x, const float* cosT, const float* sinT,
                            int n_heads, int head_dim, int rot_dim) {
    int row = blockIdx.y, h = blockIdx.x, half = rot_dim / 2;
    int i = threadIdx.x; if (i >= half) return;
    float* xh = x + (((size_t)row * n_heads + h) * head_dim);
    const float* c = cosT + (size_t)row * half; const float* sn = sinT + (size_t)row * half;
    float x0 = xh[i], x1 = xh[i + half], cc = c[i], ss = sn[i];
    xh[i]        = x0 * cc - x1 * ss;     // rotate_half convention
    xh[i + half] = x1 * cc + x0 * ss;
}
void rope_rotate_half(float* x, const float* cos, const float* sin,
                      int rows, int n_heads, int head_dim, int rot_dim, cudaStream_t s) {
    dim3 grid(n_heads, rows); rope_kernel<<<grid, rot_dim / 2, 0, s>>>(x, cos, sin, n_heads, head_dim, rot_dim);
}

__global__ void gelu_kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float v = x[i]; const float k = 0.7978845608028654f; // sqrt(2/pi)
    out[i] = 0.5f * v * (1.f + tanhf(k * (v + 0.044715f * v * v * v)));
}
void gelu_tanh(float* out, const float* x, int n, cudaStream_t s) {
    gelu_kernel<<<(n + 255) / 256, 256, 0, s>>>(out, x, n);
}

__global__ void add_kernel(float* acc, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) acc[i] += in[i];
}
void add_inplace(float* acc, const float* in, int n, cudaStream_t s) {
    add_kernel<<<(n + 255) / 256, 256, 0, s>>>(acc, in, n);
}

__global__ void softcap_kernel(float* x, float cap, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
void softcap(float* x, float cap, int n, cudaStream_t s) {
    softcap_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, cap, n);
}
