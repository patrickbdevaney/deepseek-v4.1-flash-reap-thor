// nvfp4_dense.cu — the M=1 NVFP4 GEMV and the overlay loader.
//
// LAYOUT, matching tools/requant_dense_nvfp4.py exactly:
//   packed[N, K/2]   E2M1 codes, two per byte, LOW NIBBLE FIRST (the order cvt.f16x2.e2m1x2 consumes)
//   scale [N, K/16]  e4m3, one per group of 16 along K
//   global           fp32;  real weight = e2m1(code) * (e4m3(scale) / global)
//
// The kernel is deliberately the same SHAPE as `fp8_gemv_m1_kernel`: one warp per output row,
// 4-byte loads, unrolled to keep several loads in flight, single accumulator in the original order.
// F125 measured that four accumulators are SLOWER here and that the accumulation order is worth
// keeping bit-stable, so there is no reason to deviate.
#include "nvfp4_dense.h"
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

static std::unordered_map<const uint8_t*, NvFp4Weight> g_map;
static bool g_on = false;

void nvfp4_register(const uint8_t* w, const NvFp4Weight& v){ g_map[w] = v; }
const NvFp4Weight* nvfp4_lookup(const uint8_t* w){
    if (!g_on) return nullptr;
    auto it = g_map.find(w);
    return it == g_map.end() ? nullptr : &it->second;
}
bool nvfp4_enabled(){ return g_on; }
// DSV4_NVFP4_STATS=1: how many M=1 dense calls actually reached the NVFP4 path, and how many bytes
// that saved. Without this it is impossible to tell "the kernel is slow" from "the kernel is barely
// being called" -- and the second turned out to be the case.
void nvfp4_note_hit(long long);
static long long g_hits = 0, g_miss = 0, g_bytes = 0;
void nvfp4_note_hit(long long bytes){ ++g_hits; g_bytes += bytes; }
void nvfp4_note_miss(){ ++g_miss; }
extern "C" void nvfp4_stats_dump(){
    if (getenv("DSV4_NVFP4_STATS") == nullptr) return;
    fprintf(stderr, "[nvfp4] M=1 dense calls: %lld served by NVFP4, %lld fell through to FP8; "
            "%.2f GB of weight served as NVFP4\n", g_hits, g_miss, g_bytes/1e9);
}
int  nvfp4_count(){ return (int)g_map.size(); }
void nvfp4_force_enable(bool on){ g_on = on; }

__device__ __forceinline__ float d_e4m3(unsigned char b){
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&r));
}
// two E2M1 codes in one byte -> two floats. Hardware instruction; identical to the MXFP4 path,
// because MXFP4 and NVFP4 share the E2M1 element and differ only in the scale.
__device__ __forceinline__ float2 d_e2m1x2(unsigned char b){
    __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1);
    __half2 h = *reinterpret_cast<__half2*>(&r);
    return make_float2(__low2float(h), __high2float(h));
}

// 8 activations (2 uint) x 8 packed E2M1 codes (1 uint) -> scalar dot
__device__ __forceinline__ float dot8(unsigned a0, unsigned a1, unsigned pv){
    float2 w0 = d_e2m1x2((pv      ) & 0xff), w1 = d_e2m1x2((pv >>  8) & 0xff);
    float2 w2 = d_e2m1x2((pv >> 16) & 0xff), w3 = d_e2m1x2((pv >> 24) & 0xff);
    float s = 0.f;
    s += d_e4m3((a0      ) & 0xff) * w0.x; s += d_e4m3((a0 >>  8) & 0xff) * w0.y;
    s += d_e4m3((a0 >> 16) & 0xff) * w1.x; s += d_e4m3((a0 >> 24) & 0xff) * w1.y;
    s += d_e4m3((a1      ) & 0xff) * w2.x; s += d_e4m3((a1 >>  8) & 0xff) * w2.y;
    s += d_e4m3((a1 >> 16) & 0xff) * w3.x; s += d_e4m3((a1 >> 24) & 0xff) * w3.y;
    return s;
}

// 8 float activations x 8 packed E2M1 codes.
// TWO float4 LOADS, not eight scalars. The first version read a[0]..a[7] individually and measured
// SLOWER than the FP8 ogroup kernel it was meant to replace (14.23 -> 13.26 tok/s): the activation
// here is fp32 and NOT shrunk by the requant, so at 4 bytes per activation against 0.5 per weight
// the loop is activation-bound, and issuing 8 loads where 2 suffice is the whole difference. The
// FP8 kernel this competes with has read `float4` since Finding 35.
__device__ __forceinline__ float dotf8(const float* a, unsigned pv){
    const float4 x0 = *(const float4*)(a), x1 = *(const float4*)(a+4);
    float2 w0=d_e2m1x2((pv)&0xff), w1=d_e2m1x2((pv>>8)&0xff);
    float2 w2=d_e2m1x2((pv>>16)&0xff), w3=d_e2m1x2((pv>>24)&0xff);
    float s=0.f;
    s=fmaf(x0.x,w0.x,s); s=fmaf(x0.y,w0.y,s); s=fmaf(x0.z,w1.x,s); s=fmaf(x0.w,w1.y,s);
    s=fmaf(x1.x,w2.x,s); s=fmaf(x1.y,w2.y,s); s=fmaf(x1.z,w3.x,s); s=fmaf(x1.w,w3.y,s);
    return s;
}

// one warp per output row n; each lane consumes 8 weights (4 packed bytes) per step
__global__ void nvfp4_gemv_m1_kernel(float* __restrict__ C, const uint8_t* __restrict__ A,
                                     const float* __restrict__ as, const uint8_t* __restrict__ P,
                                     const uint8_t* __restrict__ S, float gs, int N, int K){
    const int lane = threadIdx.x & 31;
    const int warp0 = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int stride = (gridDim.x*blockDim.x) >> 5;
    const float inv_gs = 1.0f / gs;
    for (int n = warp0; n < N; n += stride){
        const uint8_t* Prow = P + (size_t)n*(K/2);
        const uint8_t* Srow = S + (size_t)n*(K/16);
        float acc = 0.f;
        // MEMORY-LEVEL PARALLELISM, the same lesson the FP8 kernel records and the reason the
        // first version of this one was SLOWER than the path it replaces despite moving half the
        // bytes. Consuming each load immediately is ILP=1, which sustains 110-132 GB/s on this box
        // against 224-237 at ILP>=2 -- so half the bytes at half the bandwidth is no gain at all.
        // Measured here: 13.55 tok/s (worse than the 13.8 FP8 baseline) before this unroll.
        // Issue all loads for four steps before consuming any -> four independent misses in flight.
        // 1024 K-values per iteration: 4 steps x 32 lanes x 8. K is 1024/4096/8192, all divisible.
        const int K4 = K & ~1023;
        int base = 0;
        for (; base < K4; base += 1024){
            const int k0 = base + lane*8, k1 = k0+256, k2 = k0+512, k3 = k0+768;
            const unsigned p0 = *(const unsigned*)(Prow + (k0>>1)), p1 = *(const unsigned*)(Prow + (k1>>1)),
                           p2 = *(const unsigned*)(Prow + (k2>>1)), p3 = *(const unsigned*)(Prow + (k3>>1));
            const unsigned a0 = *(const unsigned*)(A+k0),   a1 = *(const unsigned*)(A+k0+4);
            const unsigned b0 = *(const unsigned*)(A+k1),   b1 = *(const unsigned*)(A+k1+4);
            const unsigned c0 = *(const unsigned*)(A+k2),   c1 = *(const unsigned*)(A+k2+4);
            const unsigned e0 = *(const unsigned*)(A+k3),   e1 = *(const unsigned*)(A+k3+4);
            const float s0 = d_e4m3(Srow[k0>>4])*inv_gs, s1 = d_e4m3(Srow[k1>>4])*inv_gs,
                        s2 = d_e4m3(Srow[k2>>4])*inv_gs, s3 = d_e4m3(Srow[k3>>4])*inv_gs;
            acc += dot8(a0,a1,p0) * s0 * as[k0>>7];
            acc += dot8(b0,b1,p1) * s1 * as[k1>>7];
            acc += dot8(c0,c1,p2) * s2 * as[k2>>7];
            acc += dot8(e0,e1,p3) * s3 * as[k3>>7];
        }
        for (; base < K; base += 256){
            const int k0 = base + lane*8;
            const unsigned pv = *(const unsigned*)(Prow + (k0>>1));
            const unsigned a0 = *(const unsigned*)(A+k0), a1 = *(const unsigned*)(A+k0+4);
            acc += dot8(a0,a1,pv) * d_e4m3(Srow[k0>>4]) * inv_gs * as[k0>>7];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) C[n] = acc;
    }
}

bool nvfp4_gemv_m1(float* C, const uint8_t* A_fp8, const float* a_s,
                   const NvFp4Weight& w, cudaStream_t stream){
    if (w.K % 256 || w.N <= 0) return false;
    int threads = 256, blocks = (w.N*32 + threads - 1)/threads;
    nvfp4_gemv_m1_kernel<<<blocks, threads, 0, stream>>>(C, A_fp8, a_s, w.packed, w.scale,
                                                         w.global, w.N, w.K);
    return true;
}

// ---- M=K variant: the verify path, so AR and verify compute the SAME function ------------------
// Routing only M=1 leaves the engine computing one layer two different ways -- NVFP4 during AR,
// FP8 during the spec verify. That is not a small numerical difference, it is two functions, and
// the LOSSLESS gate caught it at token 5 when wo_a made it large enough to see. It was equally
// present, and merely invisible, for the tiers that passed.
//
// Same shape as fp8_gemv_mkT_kernel: one warp per output row, the weight row read ONCE and dotted
// against all M activation rows, so weight bandwidth is amortised M times.
template<int MM>
__global__ void nvfp4_gemv_mk_kernel(float* __restrict__ C, const uint8_t* __restrict__ A,
                                     const float* __restrict__ as, const uint8_t* __restrict__ P,
                                     const uint8_t* __restrict__ S, float gs, int N, int K){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; if(warp>=N) return;
    int n=warp, lane=threadIdx.x&31, KB=K/128;
    const uint8_t* Prow=P+(size_t)n*(K/2);
    const uint8_t* Srow=S+(size_t)n*(K/16);
    const float inv_gs=1.0f/gs;
    float acc[MM];
    #pragma unroll
    for(int m=0;m<MM;++m) acc[m]=0.f;
    for (int base=0; base<K; base+=256){
        const int k0=base+lane*8;
        const unsigned pv=*(const unsigned*)(Prow+(k0>>1));
        const float sc=d_e4m3(Srow[k0>>4])*inv_gs;
        float2 w0=d_e2m1x2((pv)&0xff), w1=d_e2m1x2((pv>>8)&0xff);
        float2 w2=d_e2m1x2((pv>>16)&0xff), w3=d_e2m1x2((pv>>24)&0xff);
        const float b0=w0.x*sc,b1=w0.y*sc,b2=w1.x*sc,b3=w1.y*sc,
                    b4=w2.x*sc,b5=w2.y*sc,b6=w3.x*sc,b7=w3.y*sc;
        #pragma unroll
        for(int m=0;m<MM;++m){
            const unsigned a0=*(const unsigned*)(A+(size_t)m*K+k0);
            const unsigned a1=*(const unsigned*)(A+(size_t)m*K+k0+4);
            const float asc=as[(size_t)m*KB+(k0>>7)];
            float t=0.f;
            t=fmaf(d_e4m3((a0    )&0xff),b0,t); t=fmaf(d_e4m3((a0>> 8)&0xff),b1,t);
            t=fmaf(d_e4m3((a0>>16)&0xff),b2,t); t=fmaf(d_e4m3((a0>>24)&0xff),b3,t);
            t=fmaf(d_e4m3((a1    )&0xff),b4,t); t=fmaf(d_e4m3((a1>> 8)&0xff),b5,t);
            t=fmaf(d_e4m3((a1>>16)&0xff),b6,t); t=fmaf(d_e4m3((a1>>24)&0xff),b7,t);
            acc[m]+=t*asc;
        }
    }
    #pragma unroll
    for(int m=0;m<MM;++m){ float a=acc[m];
        #pragma unroll
        for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffff,a,o);
        if(lane==0) C[(size_t)m*N+n]=a; }
}

bool nvfp4_gemv_mk(float* C, const uint8_t* A, const float* as, const NvFp4Weight& w,
                   int M, cudaStream_t s){
    if (w.K % 256 || M < 2 || M > 8) return false;
    int threads=256, blocks=((size_t)w.N*32+threads-1)/threads;
    switch(M){
      case 2: nvfp4_gemv_mk_kernel<2><<<blocks,threads,0,s>>>(C,A,as,w.packed,w.scale,w.global,w.N,w.K); break;
      case 3: nvfp4_gemv_mk_kernel<3><<<blocks,threads,0,s>>>(C,A,as,w.packed,w.scale,w.global,w.N,w.K); break;
      case 4: nvfp4_gemv_mk_kernel<4><<<blocks,threads,0,s>>>(C,A,as,w.packed,w.scale,w.global,w.N,w.K); break;
      case 5: nvfp4_gemv_mk_kernel<5><<<blocks,threads,0,s>>>(C,A,as,w.packed,w.scale,w.global,w.N,w.K); break;
      case 6: nvfp4_gemv_mk_kernel<6><<<blocks,threads,0,s>>>(C,A,as,w.packed,w.scale,w.global,w.N,w.K); break;
      case 7: nvfp4_gemv_mk_kernel<7><<<blocks,threads,0,s>>>(C,A,as,w.packed,w.scale,w.global,w.N,w.K); break;
      case 8: nvfp4_gemv_mk_kernel<8><<<blocks,threads,0,s>>>(C,A,as,w.packed,w.scale,w.global,w.N,w.K); break;
      default: return false;
    }
    nvfp4_note_hit((long long)w.N*w.K/2);
    return true;
}

// ---- ogroup (wo_a) variant ---------------------------------------------------------------------
// wo_a does NOT go through fp8_block_gemm -- it has its own `ogroup_gemv_fp8_kernel`, which is why
// the first wiring registered wo_a (1.44 GB/token, the largest group in the overlay) and then never
// served a single call from it. Two differences from the fp8 path: the activation is FLOAT32, not
// fp8, and the rows are grouped (out[g*R + r] dots against o[g*Kd..]).
__global__ void nvfp4_ogroup_gemv_kernel(float* __restrict__ out, const float* __restrict__ o,
                                         const uint8_t* __restrict__ P, const uint8_t* __restrict__ S,
                                         float gs, int G, int R, int Kd){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int total=G*R; if(warp>=total) return;
    int gr=warp, g=gr/R, lane=threadIdx.x&31;
    const uint8_t* Prow = P + (size_t)gr*(Kd/2);
    const uint8_t* Srow = S + (size_t)gr*(Kd/16);
    const float*   og   = o + (size_t)g*Kd;
    const float inv_gs = 1.0f/gs;
    float a0=0.f,a1=0.f,a2=0.f,a3=0.f;      // ILP=4, the lesson this file already paid for once
    for (int base=0; base<Kd; base+=1024){
        const int k0=base+lane*8, k1=k0+256, k2=k0+512, k3=k0+768;
        const unsigned p0=*(const unsigned*)(Prow+(k0>>1)), p1=*(const unsigned*)(Prow+(k1>>1)),
                       p2=*(const unsigned*)(Prow+(k2>>1)), p3=*(const unsigned*)(Prow+(k3>>1));
        const float s0=d_e4m3(Srow[k0>>4])*inv_gs, s1=d_e4m3(Srow[k1>>4])*inv_gs,
                    s2=d_e4m3(Srow[k2>>4])*inv_gs, s3=d_e4m3(Srow[k3>>4])*inv_gs;
        a0 += dotf8(og+k0, p0)*s0;  a1 += dotf8(og+k1, p1)*s1;
        a2 += dotf8(og+k2, p2)*s2;  a3 += dotf8(og+k3, p3)*s3;
    }
    float acc=(a0+a1)+(a2+a3);
    #pragma unroll
    for(int o2=16;o2>0;o2>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o2);
    if(lane==0) out[gr]=acc;
}

bool nvfp4_ogroup_gemv(float* out, const float* o, const uint8_t* wo_fp8,
                       int G, int R, int Kd, cudaStream_t stream){
    const NvFp4Weight* w = nvfp4_lookup(wo_fp8);
    if (!w || w->N != G*R || w->K != Kd || (Kd % 1024)) return false;
    int threads=256;
    nvfp4_ogroup_gemv_kernel<<<((size_t)G*R*32+threads-1)/threads,threads,0,stream>>>(
        out,o,w->packed,w->scale,w->global,G,R,Kd);
    nvfp4_note_hit((long long)G*R*Kd/2);
    return true;
}

// ---- ogroup M=K: the verify side of wo_a --------------------------------------------------------
// Correctness first, tuning later. The FP8 ogroup M=K path is heavily tuned (templated on M AND an
// activation-reuse factor NR, smem variants, per-M lookup tables, register caps -- Findings 40/55/79).
// This is the plain NR=1 form: one warp per output row, weight row read once, dotted against all M
// activation rows. It exists so that enabling wo_a does not leave AR on NVFP4 and verify on FP8,
// which is the split that produced `diverges at token 5`. If it costs spec throughput it should be
// tuned, not switched off -- an engine computing a layer two ways is broken at any speed.
// NR = consecutive output rows served by one warp. They share the activation slice `og = o + g*Kd`,
// so loading it once and reusing it across NR rows cuts activation traffic by NR.
//
// WHY THIS MATTERS MORE FOR NVFP4 THAN FOR FP8. The FP8 kernel's note prices activation traffic at
// 4*M/NR times the weight traffic. NVFP4 halves the weight bytes and leaves the fp32 activation
// alone, so the ratio becomes 8*M/NR -- halving the weight DOUBLES the activation dominance. At
// NR=1, M=5 that is 40x, which is why the correct NR=1 kernel cost -24% on spec decode: it was
// never weight-bound at all.
//
// NR consecutive rows stay inside one group iff R % NR == 0 (R=1024 here, so 2/4/8 all hold).
// Per-row accumulation order is UNCHANGED -- four accumulators reduced (a0+a1)+(a2+a3), dot then
// scale -- so tests/gate_nvfp4_ogroup_mk.cu still holds at 0/40960.
template<int MM, int NR>
__global__ void nvfp4_ogroup_mk_kernel(float* __restrict__ out, const float* __restrict__ o,
                                       const uint8_t* __restrict__ P, const uint8_t* __restrict__ S,
                                       float gs, int G, int R, int Kd){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int total=G*R;
    int gr0=warp*NR; if(gr0>=total) return;
    int g=gr0/R, lane=threadIdx.x&31;
    const float inv_gs=1.0f/gs;
    float a0[NR][MM],a1[NR][MM],a2[NR][MM],a3[NR][MM];
    #pragma unroll
    for(int r=0;r<NR;++r)
        #pragma unroll
        for(int m=0;m<MM;++m){ a0[r][m]=a1[r][m]=a2[r][m]=a3[r][m]=0.f; }
    for(int base=0; base<Kd; base+=1024){
        const int k[4]={base+lane*8, base+256+lane*8, base+512+lane*8, base+768+lane*8};
        #pragma unroll
        for(int u=0;u<4;++u){
            // activation loaded ONCE per (u, m) and reused across all NR rows -- the whole point
            float4 x0[MM], x1[MM];
            #pragma unroll
            for(int m=0;m<MM;++m){
                const float* a=o+((size_t)m*G+g)*Kd+k[u];
                x0[m]=*(const float4*)(a); x1[m]=*(const float4*)(a+4);
            }
            #pragma unroll
            for(int r=0;r<NR;++r){
                const int gr=gr0+r; if(gr>=total) break;
                const uint8_t* Prow=P+(size_t)gr*(Kd/2);
                const uint8_t* Srow=S+(size_t)gr*(Kd/16);
                const unsigned pv=*(const unsigned*)(Prow+(k[u]>>1));
                const float sc=d_e4m3(Srow[k[u]>>4])*inv_gs;
                float2 w0=d_e2m1x2((pv)&0xff), w1=d_e2m1x2((pv>>8)&0xff);
                float2 w2=d_e2m1x2((pv>>16)&0xff), w3=d_e2m1x2((pv>>24)&0xff);
                #pragma unroll
                for(int m=0;m<MM;++m){
                    float t=0.f;
                    t=fmaf(x0[m].x,w0.x,t); t=fmaf(x0[m].y,w0.y,t);
                    t=fmaf(x0[m].z,w1.x,t); t=fmaf(x0[m].w,w1.y,t);
                    t=fmaf(x1[m].x,w2.x,t); t=fmaf(x1[m].y,w2.y,t);
                    t=fmaf(x1[m].z,w3.x,t); t=fmaf(x1[m].w,w3.y,t);
                    t*=sc;
                    if(u==0) a0[r][m]+=t; else if(u==1) a1[r][m]+=t;
                    else if(u==2) a2[r][m]+=t; else a3[r][m]+=t;
                }
            }
        }
    }
    #pragma unroll
    for(int r=0;r<NR;++r){ const int gr=gr0+r; if(gr>=total) break;
        #pragma unroll
        for(int m=0;m<MM;++m){ float a=(a0[r][m]+a1[r][m])+(a2[r][m]+a3[r][m]);
            #pragma unroll
            for(int o2=16;o2>0;o2>>=1) a+=__shfl_down_sync(0xffffffff,a,o2);
            if(lane==0) out[(size_t)m*total+gr]=a; }
    }
}

bool nvfp4_ogroup_mk(float* out, const float* o, const uint8_t* wo_fp8,
                     int M, int G, int R, int Kd, cudaStream_t stream){
    const NvFp4Weight* w = nvfp4_lookup(wo_fp8);
    if (!w || w->N != G*R || w->K != Kd || (Kd % 1024) || M < 2 || M > 8) return false;
    // NR: 4*NR*M accumulators, so it trades activation reuse against occupancy. NVFP4_NR overrides
    // for A/B. Default 2 -- at M=5 that is 40 accumulators, and the FP8 kernel's note records that
    // NR=8 spills against the 64-register cap and loses badly.
    int nr = 2; if(const char* e=getenv("NVFP4_NR")) nr=atoi(e);
    if(nr!=1 && nr!=2 && nr!=4) nr=2;
    if (R % nr) nr = 1;
    int threads=256; size_t warps=((size_t)G*R + nr - 1)/nr;
    size_t nb=(warps*32+threads-1)/threads;
    #define LAUNCH(MMv,NRv) nvfp4_ogroup_mk_kernel<MMv,NRv><<<nb,threads,0,stream>>>(out,o,w->packed,w->scale,w->global,G,R,Kd)
    #define BYM(NRv) switch(M){ case 2: LAUNCH(2,NRv); break; case 3: LAUNCH(3,NRv); break; \
        case 4: LAUNCH(4,NRv); break; case 5: LAUNCH(5,NRv); break; case 6: LAUNCH(6,NRv); break; \
        case 7: LAUNCH(7,NRv); break; case 8: LAUNCH(8,NRv); break; default: return false; }
    if(nr==1){ BYM(1) } else if(nr==2){ BYM(2) } else { BYM(4) }
    #undef BYM
    #undef LAUNCH
    nvfp4_note_hit((long long)G*R*Kd/2);
    return true;
}

// ---- overlay loading -----------------------------------------------------------------------------
// Minimal safetensors reader: the file is one header (u64 length + JSON) then a data blob. Only the
// three suffixes this overlay writes are understood; anything else is ignored rather than guessed at.
static bool json_find(const std::string& s, const std::string& key, size_t& b, size_t& e){
    size_t p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return false;
    b = s.find('{', p); if (b == std::string::npos) return false;
    int d = 0;
    for (e = b; e < s.size(); ++e){ if (s[e]=='{') ++d; else if (s[e]=='}' && --d == 0) { ++e; return true; } }
    return false;
}
static bool get_offsets(const std::string& obj, long& o0, long& o1){
    size_t p = obj.find("\"data_offsets\"");
    if (p == std::string::npos) return false;
    p = obj.find('[', p);
    return p != std::string::npos && sscanf(obj.c_str()+p, "[%ld,%ld]", &o0, &o1) == 2;
}

int nvfp4_load_overlay(const char* dir, const uint8_t* (*resolve)(const char*, void*), void* ctx){
    std::string path = std::string(dir) + "/dense_nvfp4.safetensors";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f){ fprintf(stderr, "[nvfp4] no overlay at %s\n", path.c_str()); return 0; }
    unsigned long long hn = 0;
    if (fread(&hn, 8, 1, f) != 1){ fclose(f); return 0; }
    std::string hdr(hn, '\0');
    if (fread(&hdr[0], 1, hn, f) != hn){ fclose(f); return 0; }
    const size_t base = 8 + hn;
    fseek(f, 0, SEEK_END); const size_t fsz = ftell(f);

    // one host read of the whole blob, then one device allocation per tensor
    std::vector<uint8_t> blob(fsz - base);
    fseek(f, base, SEEK_SET);
    if (fread(blob.data(), 1, blob.size(), f) != blob.size()){ fclose(f); return 0; }
    fclose(f);

    int n_reg = 0, n_missing = 0;
    size_t pos = 0;
    while (true){
        size_t q = hdr.find(".nvfp4\"", pos);
        if (q == std::string::npos) break;
        size_t st = hdr.rfind('"', q);
        std::string name = hdr.substr(st+1, q - st - 1 + 6);          // "<tensor>.nvfp4"
        pos = q + 6;
        std::string tname = name.substr(0, name.size() - 6);          // strip ".nvfp4"
        const uint8_t* fp8 = resolve(tname.c_str(), ctx);
        if (!fp8){ ++n_missing; continue; }
        size_t b0,e0,b1,e1,b2,e2; long o0,o1, s0,s1, g0,g1;
        if (!json_find(hdr, name, b0,e0) ||
            !json_find(hdr, name+"_scale", b1,e1) ||
            !json_find(hdr, name+"_global", b2,e2)) continue;
        if (!get_offsets(hdr.substr(b0,e0-b0), o0,o1) ||
            !get_offsets(hdr.substr(b1,e1-b1), s0,s1) ||
            !get_offsets(hdr.substr(b2,e2-b2), g0,g1)) continue;
        // shape from the packed entry: [N, K/2]
        std::string ob = hdr.substr(b0, e0-b0);
        size_t sp = ob.find("\"shape\""); long N=0, Kh=0;
        if (sp == std::string::npos) continue;
        sp = ob.find('[', sp);
        if (sscanf(ob.c_str()+sp, "[%ld,%ld]", &N, &Kh) != 2) continue;
        NvFp4Weight w{}; w.N = (int)N; w.K = (int)(Kh*2);
        uint8_t *dP=nullptr, *dS=nullptr;
        if (cudaMalloc(&dP, o1-o0) != cudaSuccess || cudaMalloc(&dS, s1-s0) != cudaSuccess) continue;
        cudaMemcpy(dP, blob.data()+o0, o1-o0, cudaMemcpyHostToDevice);
        cudaMemcpy(dS, blob.data()+s0, s1-s0, cudaMemcpyHostToDevice);
        memcpy(&w.global, blob.data()+g0, 4);
        w.packed = dP; w.scale = dS;
        nvfp4_register(fp8, w);
        ++n_reg;
    }
    g_on = (n_reg > 0) && (getenv("DSV4_NVFP4_OFF") == nullptr);
    fprintf(stderr, "[nvfp4] overlay %s: registered %d dense weights (%d not resolvable), active=%d\n",
            dir, n_reg, n_missing, (int)g_on);
    return n_reg;
}
