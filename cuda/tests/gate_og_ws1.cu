// gate_og_ws1.cu — is the WS1 scale-hoist in `ogroup_gemv_mk_kernel` BIT-identical to the kernel
// it replaces?
//
// The claim WS1 rests on is a VALUE-EQUALITY argument, not a numerical-tolerance one: `ws[r]` is
// indexed (rr/128, kb) with rr = gr0+r, gr0 a multiple of NR, and NR | 128, so all NR rows a warp
// owns fall in the same 128-row scale block and every `ws[r]` is the same float. WS1 loads it once
// instead of NR times. Every fma therefore sees the identical operand it saw before.
//
// A claim of that shape is checked with `memcmp`, not with a cosine. `tests/gate_ogroup_gemv.cu`
// next door is a cosine gate on the M=1 kernel and would pass a change that perturbed the last
// mantissa bit; LEVERS.md trap 6 (Finding 68) is the worked example of a numerics change that
// bought +28% tok/s by degrading the output past every tolerance gate in the project.
//
// Sweep, per trap 9 (a harness that cannot express the regime just confirms itself):
//   M   2..16, covering every `OG_MK_CASE` the verify can dispatch and both sides of the NR lookup
//   NR  1,2,4,8 — forced with OG_NR, because the shipped lookup only ever picks 2 and 4
//   Kd  4096 (real) and 512 (short row: 4 k-blocks, so the kb loop's prologue/epilogue dominate)
//   G   8 (real) and 4
// Outputs are poisoned to 0xEE before each launch, so a kernel that writes nothing cannot inherit
// a pass from the previous arm.
//
// NOT COVERED, and it is structural rather than an omission: the tail-warp clamp
// `rr = (gr0+r<total) ? (gr0+r) : gr0` is unreachable. A tail needs total % (8*NR) != 0, but the
// e8m0 scale plane requires total % 128 == 0 and 8*NR divides 128 for every NR the dispatch
// allows. No legal shape reaches it.
//
//   build: see scripts/build_gate.sh      run: ./build/gate_og_ws1
#include "dscratch.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CU(x) do{cudaError_t e=(x); if(e){printf("ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;}}while(0)

void ogroup_gemm_fp8(float* out, const float* o, const uint8_t* wo_fp8, const uint8_t* wo_sc,
                     int bs, int G, int R, int Kd, cudaStream_t stream);

int main(){
    // No arena_init: dscratch.h documents that dmalloc/dsync fall back to plain cudaMalloc and a
    // real sync when the arena was never initialised, and the M>=2 path here only uses dsync.
    const int Ms[]  = {2,3,4,5,6,7,8,13,16};
    const int NRs[] = {1,2,4,8};
    struct Shape { int G,R,Kd; } shapes[] = { {8,1024,4096}, {4,1024,512} };

    int checked=0, bad=0;
    for(const Shape& s : shapes){
        const int rows = s.G*s.R;
        std::vector<uint8_t> w((size_t)rows*s.Kd), sc((size_t)(rows/128)*(s.Kd/128));
        srand(7 + s.Kd);
        // 0x7f/0xff are e4m3 NaN. A NaN weight makes memcmp compare NaN payloads instead of the
        // arithmetic, so both arms would "differ" for a reason that is not the change.
        for(auto& b : w){ uint8_t v; do { v=(uint8_t)(rand()&0xff); } while(v==0x7f||v==0xff); b=v; }
        // Span the e8m0 exponent range the checkpoint actually uses, including 0 (2^-127, subnormal)
        // — both arms call the same exp2f on the same byte, so this is about coverage, not about
        // whether exp2f is exact.
        for(size_t i=0;i<sc.size();++i) sc[i] = (uint8_t)((i%3==0) ? 0 : (110 + rand()%30));

        uint8_t *dw,*dsc; CU(cudaMalloc(&dw,w.size())); CU(cudaMalloc(&dsc,sc.size()));
        CU(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dsc,sc.data(),sc.size(),cudaMemcpyHostToDevice));

        for(int M : Ms){
            std::vector<float> ho((size_t)M*s.G*s.Kd);
            for(auto& x : ho) x = (rand()%2000-1000)/997.f;
            float *dO,*Ca,*Cb;
            CU(cudaMalloc(&dO,ho.size()*4));
            CU(cudaMalloc(&Ca,(size_t)M*rows*4)); CU(cudaMalloc(&Cb,(size_t)M*rows*4));
            CU(cudaMemcpy(dO,ho.data(),ho.size()*4,cudaMemcpyHostToDevice));
            for(int nr : NRs){
                char buf[8]; snprintf(buf,sizeof buf,"%d",nr); setenv("OG_NR",buf,1);
                CU(cudaMemset(Ca,0xEE,(size_t)M*rows*4)); CU(cudaMemset(Cb,0xEE,(size_t)M*rows*4));
                setenv("OG_WS1","1",1);                                 // WS1 on  (opt-in)
                ogroup_gemm_fp8(Ca,dO,dw,dsc,M,s.G,s.R,s.Kd,0); CU(cudaDeviceSynchronize());
                unsetenv("OG_WS1");                                     // WS1 off (shipped default)
                ogroup_gemm_fp8(Cb,dO,dw,dsc,M,s.G,s.R,s.Kd,0); CU(cudaDeviceSynchronize());
                std::vector<float> a((size_t)M*rows), b((size_t)M*rows);
                CU(cudaMemcpy(a.data(),Ca,a.size()*4,cudaMemcpyDeviceToHost));
                CU(cudaMemcpy(b.data(),Cb,b.size()*4,cudaMemcpyDeviceToHost));
                // A pass must also mean the kernel RAN: 0xEEEEEEEE is -3.7e28, so a poisoned
                // output that survived both arms would compare equal and mean nothing.
                bool wrote=false; for(size_t i=0;i<a.size() && !wrote;++i) if(memcmp(&a[i],"\xEE\xEE\xEE\xEE",4)!=0) wrote=true;
                const bool eq = (memcmp(a.data(),b.data(),a.size()*4)==0);
                ++checked;
                if(!eq || !wrote){ ++bad;
                    printf("  MISMATCH G=%d R=%d Kd=%d M=%-3d NR=%d  %s\n",
                           s.G,s.R,s.Kd,M,nr, wrote?"values differ":"OUTPUT NEVER WRITTEN");
                    for(size_t i=0;i<a.size();++i) if(a[i]!=b[i]){
                        printf("    first at %zu: ws1=%.9g  ref=%.9g\n",i,(double)a[i],(double)b[i]); break; }
                }
            }
            unsetenv("OG_NR");
            CU(cudaFree(dO)); CU(cudaFree(Ca)); CU(cudaFree(Cb));
        }
        CU(cudaFree(dw)); CU(cudaFree(dsc));
    }
    printf("gate_og_ws1: %d/%d configurations bit-identical (OG_WS1=1 vs shipped default)\n", checked-bad, checked);
    printf(bad ? "GATE FAIL\n" : "GATE PASS\n");
    return bad ? 1 : 0;
}
