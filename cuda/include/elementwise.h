// elementwise.h — core fp32 elementwise kernels for the Gemma-4 forward.
// All activations flow in fp32; norm/embedding weights are bf16 in the checkpoint.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Gemma4RMSNorm: out = x*(mean(x^2)+eps)^-0.5 [* (bf16)weight if w!=nullptr]. Per-row over `dim`.
// fp32 compute. w==nullptr => with_scale=False (v_norm / no-weight norms).
void rmsnorm(float* out, const float* x, const uint16_t* w_bf16, int rows, int dim, float eps, cudaStream_t s);

// rotate-half RoPE on x[rows, n_heads, head_dim]; rotates the FIRST rot_dim dims using
// per-row cos/sin tables [rows, rot_dim/2]; dims >= rot_dim pass through unchanged.
void rope_rotate_half(float* x, const float* cos, const float* sin,
                      int rows, int n_heads, int head_dim, int rot_dim, cudaStream_t s);

// gelu_pytorch_tanh elementwise in-place (or out-of-place): y = 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
void gelu_tanh(float* out, const float* x, int n, cudaStream_t s);

// out[rows,dim] += in[rows,dim]
void add_inplace(float* acc, const float* in, int n, cudaStream_t s);

// logit softcap: y = cap*tanh(x/cap), in-place over n
void softcap(float* x, float cap, int n, cudaStream_t s);
