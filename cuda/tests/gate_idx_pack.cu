// gate_idx_pack.cu — the acceptance test for the MXFP4 index cache is memcmp, not a tolerance.
//   nvcc -O2 -gencode arch=compute_110a,code=sm_110a -I include tests/gate_idx_pack.cu -o build/gate_idx_pack
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "idx_pack.h"
using namespace dsv4idx;

// Verbatim from kernels/mla_attn.cu — the write path this packing must invert exactly.
__host__ __device__ __forceinline__ float round_e2m1(float v) {
    float a = fabsf(v), m;
    if (a < 0.25f) m = 0.f; else if (a < 0.75f) m = 0.5f; else if (a < 1.25f) m = 1.f;
    else if (a < 1.75f) m = 1.5f; else if (a < 2.5f) m = 2.f; else if (a < 3.5f) m = 3.f;
    else if (a < 5.f) m = 4.f; else m = 6.f;
    return (v < 0.f) ? -m : m;
}
static void act_quant_fp4sim_host(float* x, int n, int block) {
    for (int g = 0; g < n / block; ++g) {
        float* xr = x + g * block, amax = 6.f * 7.5231631e-37f;
        for (int i = 0; i < block; ++i) amax = fmaxf(amax, fabsf(xr[i]));
        float scale = exp2f(ceilf(log2f(amax * (1.f / 6.f))));
        for (int i = 0; i < block; ++i) {
            float q = fminf(fmaxf(xr[i] / scale, -6.f), 6.f);
            xr[i] = round_e2m1(q) * scale;
        }
    }
}
static void pack_row(const float* f, uint8_t* row) {
    for (int b = 0; b < PACK_NB; ++b) {
        float amax = 0.f;
        for (int i = 0; i < PACK_BLOCK; ++i) amax = fmaxf(amax, fabsf(f[b*PACK_BLOCK+i]));
        // The scale is recoverable from any nonzero element: value = grid * 2^k, and the grid
        // points are themselves powers of two or 1.5/3 times one, so take it from the max.
        float scale = 1.f;
        if (amax > 0.f) { int k = (int)floorf(log2f(amax / 6.f) + 0.5f);
                          // exact: amax is grid_max * 2^k with grid_max in {0.5..6}
                          scale = exp2f(ceilf(log2f(amax * (1.f/6.f)))); }
        row[IHD/2 + b] = ue8m0_from_pow2(scale);
        for (int i = 0; i < PACK_BLOCK; ++i) {
            int e = b*PACK_BLOCK + i;
            // Divide, do not multiply: f[e] == grid*scale exactly, so f[e]/scale recovers the grid
            // point exactly, and -0.0/positive stays -0.0 so the sign survives.
            int c = e2m1_code(f[e] / scale);
            if (e & 1) row[e>>1] = (uint8_t)((row[e>>1] & 0x0F) | (c << 4));
            else       row[e>>1] = (uint8_t)((row[e>>1] & 0xF0) | c);
        }
    }
}
int main(){
    const int ROWS = 4096;
    std::vector<float> f((size_t)ROWS*IHD);
    srand(7);
    for (size_t i = 0; i < f.size(); ++i) {
        int m = (i/IHD) % 4;
        float r = (float)rand()/RAND_MAX*2.f - 1.f;
        f[i] = m==0 ? r : m==1 ? r*1e-6f : m==2 ? r*1e5f : (i%17==0 ? 0.f : r*3.7f);
    }
    for (int r = 0; r < ROWS; ++r) act_quant_fp4sim_host(&f[(size_t)r*IHD], IHD, PACK_BLOCK);

    std::vector<uint8_t> packed((size_t)ROWS*PACK_BYTES, 0);
    for (int r = 0; r < ROWS; ++r) pack_row(&f[(size_t)r*IHD], &packed[(size_t)r*PACK_BYTES]);

    size_t bad = 0; float worst = 0.f;
    for (int r = 0; r < ROWS; ++r)
        for (int i = 0; i < IHD; ++i) {
            float a = f[(size_t)r*IHD+i];
            float b = idx_unpack(&packed[(size_t)r*PACK_BYTES], i);
            if (memcmp(&a,&b,4) != 0) { bad++; worst = fmaxf(worst, fabsf(a-b)); }
        }
    printf("rows %d  elements %zu\n", ROWS, (size_t)ROWS*IHD);
    printf("bytes/row  %d -> %d  (%.2fx)\n", IHD*4, PACK_BYTES, (double)(IHD*4)/PACK_BYTES);
    printf("bitwise mismatches: %zu   worst |delta| %g\n", bad, worst);
    printf("\nGATE: %s\n", bad==0 ? "PASS — unpack(pack(x)) == x bit-for-bit on every element"
                                  : "FAIL — packing is NOT lossless, do not ship");
    return bad==0?0:1;
}
