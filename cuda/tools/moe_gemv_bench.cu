// moe_gemv_bench.cu — what does the SHIPPING MoE GEMV actually achieve, and on which shape?
//
// THE QUESTION. `moe:w1w3` is the largest single mark in the engine and the wiki prices it at
// "155 GB/s = 67 % of roofline". That figure is F67's, it predates the BN=2 rewrite, and it was
// never re-taken. Meanwhile the same kernel's other call site, `moe:w2`, computes out of the same
// dprof logs at ~208 GB/s. Same kernel, same RB rule, same BN, same weights-per-expert — different
// shape. **A 24 % spread between two call sites of one kernel is a controlled measurement that
// nobody has run down**, and it is the only "headroom" claim in this repo that does not depend on
// comparing a microbenchmark against dprof (F126's caveat).
//
// So this bench runs `tc_fp4_grouped_gemv_e8m0` — the shipped entry point, not a rewrite, so the
// RB rule and the grid geometry are the engine's — at BOTH real shapes:
//
//     w1 / w3 :  N = moe_intermediate 2048,  K = hidden 4096
//     w2      :  N = hidden          4096,  K = moe_intermediate 2048
//
// and at BOTH real groupings, taken from `DSV4_MOEUNION=1` (LEVERS.md §5 reference table):
//
//     M=1 decode : 6 experts x 1 row              -> RB=1
//     K=5 verify : 12x1 + 3x2 + 1x3 + 1x4 + 1x5   -> RB=4, 18 experts / 30 rows
//
// L2 TRAP (F125). Thor's L2 is 33.6 MB. One M=1 w1w3 launch touches 6 x 4.19 = 25.2 MB, which FITS,
// so a loop over one expert set measures L2 and reports a number that has nothing to do with decode.
// Every rep here cycles a different set out of a pool sized well past L2.
//
// ALIGNMENT. F66 counted 43,470 of 44,436 expert tensors at `data_offset % 16 == 8`. Every weight
// pointer below is deliberately offset by 8, so the ALIGN8 template — the one that ships — is the
// one measured.
//
//   nvcc -O3 -arch=sm_110a -std=c++17 -I include -o build/moe_gemv_bench \
//        tools/moe_gemv_bench.cu kernels/tc_moe_gemm.cu
//   ./build/moe_gemv_bench            # add MOE_RB=<n> to override the shipped rule
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CU(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

void tc_fp4_grouped_gemv_e8m0(float*, const unsigned char*, const float*, const unsigned char* const*,
        const unsigned char* const*, const int*, const int*, const int*, const int*,
        int, int, int, cudaStream_t, int, int);
void tc_build_tiles(int*, int*, int*, const int*, int, cudaStream_t);
void tc_build_tiles_rg(int*, int*, int*, const int*, int, cudaStream_t, int);
int  tcm_rowg();
// The OTHER MoE kernel, already in the tree and reachable via MOE_MMA=1: one m16n8k16 tile per
// <=16 rows, so the weight is dequantised ONCE for all rows a tile owns. Needs mma-order weights
// (`tc_ensure_repacked`, in place) and fp16 activations (`tc_a_to_fp16`).
void tc_fp4_grouped_gemm_e8m0(float*, const __half*, const unsigned char* const*, const unsigned char* const*,
        const int*, const int*, const int*, const int*, int, int, int, cudaStream_t);
void tc_repack_weight_inplace(unsigned char*, int, int, unsigned char*, cudaStream_t);
void tc_a_to_fp16(__half*, const unsigned char*, const float*, int, int, cudaStream_t);

// A plain streaming read over the SAME pool, so the roofline is measured on this box at these
// clocks in this binary rather than quoted from HARDWARE.md.
__global__ void k_stream(const uint4* __restrict__ p, size_t n4, float* sink){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x*blockDim.x;
    unsigned acc = 0;
    for (; i < n4; i += st){ uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0xdeadbeefu) *sink = 1.f;
}

struct Case { const char* name; int N, K; };

// One "set" = the experts one launch touches, each with its own weight + scale buffer.
struct ExpertSet {
    std::vector<unsigned char*> w, s;          // device, offset by 8 (residue-8, as the loader gives)
    unsigned char** wptr_d; unsigned char** sptr_d;
};

static double bench(const Case& c, const std::vector<int>& rows_per_expert, int rows_hint,
                    int pool, int reps, const char* label, int maxtiles_override = 0,
                    bool mma = false){
    const int nr = (int)rows_per_expert.size();
    const size_t wbytes = (size_t)c.N * c.K / 2;
    const size_t sbytes = (size_t)c.N * c.K / 32;

    int total_rows = 0; for (int r : rows_per_expert) total_rows += r;

    std::vector<int> off(nr+1, 0);
    for (int e = 0; e < nr; ++e) off[e+1] = off[e] + rows_per_expert[e];

    // --- expert pool, sized past L2 -----------------------------------------------------------
    std::vector<ExpertSet> sets(pool);
    for (int p = 0; p < pool; ++p){
        sets[p].w.resize(nr); sets[p].s.resize(nr);
        std::vector<unsigned char*> hw(nr), hs(nr);
        for (int e = 0; e < nr; ++e){
            unsigned char* raw; CU(cudaMalloc(&raw, wbytes + 16));
            CU(cudaMemset(raw, 0x53, wbytes + 16));           // arbitrary nonzero fp4 nibbles
            sets[p].w[e] = raw + 8;                            // <-- residue 8, the shipped case
            unsigned char* rs; CU(cudaMalloc(&rs, sbytes + 16));
            CU(cudaMemset(rs, 127, sbytes + 16));              // e8m0 127 -> 2^0
            sets[p].s[e] = rs + 8;
            hw[e] = sets[p].w[e]; hs[e] = sets[p].s[e];
        }
        CU(cudaMalloc(&sets[p].wptr_d, nr*sizeof(void*)));
        CU(cudaMalloc(&sets[p].sptr_d, nr*sizeof(void*)));
        CU(cudaMemcpy(sets[p].wptr_d, hw.data(), nr*sizeof(void*), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(sets[p].sptr_d, hs.data(), nr*sizeof(void*), cudaMemcpyHostToDevice));
    }

    // --- activations, tiles, output -----------------------------------------------------------
    unsigned char* Xq; float* Xs; float* out;
    CU(cudaMalloc(&Xq, (size_t)total_rows*c.K));
    CU(cudaMemset(Xq, 0x38, (size_t)total_rows*c.K));         // e4m3 0x38 = 0.5
    CU(cudaMalloc(&Xs, (size_t)total_rows*(c.K/128)*sizeof(float)));
    { std::vector<float> h((size_t)total_rows*(c.K/128), 1.f);
      CU(cudaMemcpy(Xs, h.data(), h.size()*sizeof(float), cudaMemcpyHostToDevice)); }
    CU(cudaMalloc(&out, (size_t)total_rows*c.N*sizeof(float)));

    int* off_d; CU(cudaMalloc(&off_d, (nr+1)*sizeof(int)));
    CU(cudaMemcpy(off_d, off.data(), (nr+1)*sizeof(int), cudaMemcpyHostToDevice));
    int *tile_e, *tile_row0, *ntiles_d;
    CU(cudaMalloc(&tile_e, total_rows*sizeof(int)));
    CU(cudaMalloc(&tile_row0, total_rows*sizeof(int)));
    CU(cudaMalloc(&ntiles_d, sizeof(int)));
    // the mma path may pack RG row-groups into one tile; the GEMV path is always 16 rows.
    if(mma) tc_build_tiles_rg(tile_e, tile_row0, ntiles_d, off_d, nr, 0, tcm_rowg());
    else    tc_build_tiles(tile_e, tile_row0, ntiles_d, off_d, nr, 0);
    CU(cudaDeviceSynchronize());

    // --- bytes the launch MUST move -----------------------------------------------------------
    // DRAM traffic is ONE read per expert, always. RB only decides how many times the kernel
    // re-issues that read, and a re-read inside the same block hits L1/L2 rather than DRAM — which
    // is why scoring RB by `ceil(me/RB)` bytes reports 284 GB/s at RB=1, above the roofline. Time
    // is the metric; GB/s here is against the DRAM figure so it stays comparable across RB.
    const char* rbe = getenv("MOE_RB");
    const int RB = mma ? 16 : (rbe ? atoi(rbe) : (rows_hint <= 1 ? 1 : rows_hint <= 2 ? 2 : 4));
    const double wreads = (double)nr;
    const double bytes = wreads * (double)(wbytes + sbytes);

    // mma needs the weight in fragment order and the activation in fp16.
    __half* x16 = nullptr;
    if (mma){
        unsigned char* tmp; CU(cudaMalloc(&tmp, wbytes));
        for (int p = 0; p < pool; ++p)
            for (int e = 0; e < nr; ++e) tc_repack_weight_inplace(sets[p].w[e], c.N, c.K, tmp, 0);
        CU(cudaDeviceSynchronize()); cudaFree(tmp);
        CU(cudaMalloc(&x16, (size_t)total_rows*c.K*sizeof(__half)));
        tc_a_to_fp16(x16, Xq, Xs, total_rows, c.K, 0);
        CU(cudaDeviceSynchronize());
    }

    // --- run ----------------------------------------------------------------------------------
    // `maxtiles` is a HOST upper bound on the tile count; the engine passes total rows because the
    // real count is only known on the device. Overriding it isolates what the early-exit blocks cost.
    const int mt = maxtiles_override ? maxtiles_override : total_rows;
    // The mma arm pays `tc_a_to_fp16` inside the timed loop. In situ one convert serves two GEMMs
    // (w1 and w3), so charging it per launch is pessimistic for the mma arm, not generous.
    auto launch = [&](int i){
        if (mma){
            tc_a_to_fp16(x16, Xq, Xs, total_rows, c.K, 0);
            tc_fp4_grouped_gemm_e8m0(out, x16, (const unsigned char* const*)sets[i%pool].wptr_d,
                (const unsigned char* const*)sets[i%pool].sptr_d, off_d, tile_e, tile_row0,
                ntiles_d, mt, c.N, c.K, 0);
        } else {
            tc_fp4_grouped_gemv_e8m0(out, Xq, Xs, (const unsigned char* const*)sets[i%pool].wptr_d,
                (const unsigned char* const*)sets[i%pool].sptr_d, off_d, tile_e, tile_row0,
                ntiles_d, mt, c.N, c.K, 0, rows_hint, 1);
        }
    };
    for (int i = 0; i < 3; ++i) launch(i);   // warm
    CU(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; CU(cudaEventCreate(&t0)); CU(cudaEventCreate(&t1));
    CU(cudaEventRecord(t0));
    for (int i = 0; i < reps; ++i) launch(i);
    CU(cudaEventRecord(t1)); CU(cudaEventSynchronize(t1));
    float ms; CU(cudaEventElapsedTime(&ms, t0, t1));
    const double gbs = bytes * reps / (ms/1e3) / 1e9;

    // CORRECTNESS, not just speed. A kernel that does LESS work also runs faster, and the RG/NB
    // variants change how much work each warp owns -- so print a checksum of the whole output and
    // compare it ACROSS configs. Computed after the timed region; it cannot perturb the number.
    double chk = 0.0; long nz = 0;
    if(getenv("MOE_BENCH_SUM")){
        std::vector<float> h((size_t)total_rows*c.N);
        CU(cudaMemcpy(h.data(), out, h.size()*sizeof(float), cudaMemcpyDeviceToHost));
        for(size_t i=0;i<h.size();++i){ if(h[i]!=0.f){ chk += (double)h[i]*(double)((i%1021)+1); ++nz; } }
    }
    printf("  %-32s %-6s rows=%-3d experts=%-3d %-4s  %7.3f ms  %7.1f GB/s",
           label, c.name, total_rows, nr, mma ? "mma" : "gemv", ms/reps, gbs);
    if(getenv("MOE_BENCH_SUM")) printf("  chk=%.6e nz=%ld", chk, nz);
    printf("\n");
    if (x16) cudaFree(x16);

    for (int p = 0; p < pool; ++p){
        for (int e = 0; e < nr; ++e){ cudaFree(sets[p].w[e]-8); cudaFree(sets[p].s[e]-8); }
        cudaFree(sets[p].wptr_d); cudaFree(sets[p].sptr_d);
    }
    cudaFree(Xq); cudaFree(Xs); cudaFree(out);
    cudaFree(off_d); cudaFree(tile_e); cudaFree(tile_row0); cudaFree(ntiles_d);
    return gbs;
}

int main(int argc, char** argv){
    const int pool = argc > 1 ? atoi(argv[1]) : 8;
    const int reps = argc > 2 ? atoi(argv[2]) : 40;

    // Roofline on this box, this binary, these clocks — 512 MB, far past L2.
    {
        const size_t n4 = (512ull<<20)/16;
        uint4* p; CU(cudaMalloc(&p, n4*16)); CU(cudaMemset(p, 1, n4*16));
        float* sink; CU(cudaMalloc(&sink, 4));
        k_stream<<<320,256>>>(p, n4, sink); CU(cudaDeviceSynchronize());
        cudaEvent_t a,b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
        CU(cudaEventRecord(a));
        for (int i=0;i<20;++i) k_stream<<<320,256>>>(p, n4, sink);
        CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
        float ms; CU(cudaEventElapsedTime(&ms,a,b));
        printf("[roofline] streaming read 512 MB x20: %.1f GB/s\n\n", n4*16.0*20/(ms/1e3)/1e9);
        cudaFree(p); cudaFree(sink);
    }

    const Case w13{"w1/w3", 2048, 4096};
    const Case w2 {"w2",    4096, 2048};

    // M=1 decode: 6 routed experts, one row each (num_experts_per_tok = 6).
    std::vector<int> g_m1(6, 1);
    // K=5 verify: the MEASURED histogram, DSV4_MOEUNION=1 (LEVERS.md §5).
    std::vector<int> g_k5;
    for (int i=0;i<12;++i) g_k5.push_back(1);
    for (int i=0;i<3;++i)  g_k5.push_back(2);
    g_k5.push_back(3); g_k5.push_back(4); g_k5.push_back(5);

    // Controls that separate the two things the K=5 grouping changes at once: MORE TILES, and
    // MORE ROWS PER TILE. `18 x 1 row` has the same tile count and (near) the same weight reads as
    // the K=5 histogram but never enters the multi-row inner loop.
    std::vector<int> g_18(18, 1), g_30(30, 1);

    printf("[shipped path]\n");
    double m1  = bench(w13, g_m1, 1, pool, reps, "M=1 decode (6 x 1 row)");
                 bench(w2,  g_m1, 1, pool, reps, "M=1 decode (6 x 1 row)");
    double k5  = bench(w13, g_k5, 5, pool, reps, "K=5 verify (measured hist)");
                 bench(w2,  g_k5, 5, pool, reps, "K=5 verify (measured hist)");
    printf("  -> K=5 / M=1 on w1w3 = %.3fx\n\n", k5/m1);

    printf("[control: is it the TILE COUNT or the ROWS PER TILE?]\n");
    bench(w13, g_18, 1, pool, reps, "18 experts x 1 row, RB=1");
    bench(w13, g_30, 1, pool, reps, "30 experts x 1 row, RB=1");
    printf("\n[control: RB sweep on the K=5 histogram]\n");
    for (const char* rb : {"1","2","4","8"}){
        setenv("MOE_RB", rb, 1);
        char lab[64]; snprintf(lab, sizeof lab, "K=5 hist, MOE_RB=%s", rb);
        bench(w13, g_k5, 5, pool, reps, lab);
    }
    unsetenv("MOE_RB");
    printf("\n[control: what do the early-exit grid.y blocks cost?]\n");
    bench(w13, g_k5, 5, pool, reps, "K=5 hist, maxtiles=30 (shipped)", 30);
    bench(w13, g_k5, 5, pool, reps, "K=5 hist, maxtiles=18 (exact)",   18);

    // THE DECISIVE ONE. 18 experts, R rows each, RB >= R -> the weight is read ONCE per expert at
    // every R, so DRAM traffic is CONSTANT (18 x 4.456 MB) and the only thing varying is how much
    // arithmetic each weight byte feeds. Flat in R => memory-bound, nothing to win in the row loop.
    // Linear in R => the row loop is issue-bound, and the per-row re-dequant is the thing to attack.
    printf("\n[control: DRAM bytes FIXED (18 experts, one read each), rows per expert varying]\n");
    setenv("MOE_RB", "8", 1);
    for (int R = 1; R <= 5; ++R){
        std::vector<int> g(18, R);
        char lab[64]; snprintf(lab, sizeof lab, "18 experts x %d rows, RB=8", R);
        bench(w13, g, 8, pool, reps, lab);
    }
    unsetenv("MOE_RB");

    // LEVERS.md §9 lever #3, "unquantified" since it was written. The engine runs the GEMV at ALL
    // M (`g_moe_gemv` is set unconditionally); F85's -16 % decode verdict on MOE_MMA=1 was an
    // engine-level A/B dominated by the M=1 path, where mma loses badly. Nobody has put the two
    // kernels side by side at the grouping the VERIFY actually presents.
    printf("\n[GEMV vs mma, head to head, same weights, same grouping]\n");
    for (auto& g : {std::make_pair(g_m1, 1), std::make_pair(g_k5, 5)}){
        char l1[64], l2[64];
        snprintf(l1, sizeof l1, "%s grouping", g.second == 1 ? "M=1" : "K=5");
        snprintf(l2, sizeof l2, "%s grouping", g.second == 1 ? "M=1" : "K=5");
        bench(w13, g.first, g.second, pool, reps, l1, 0, false);
        bench(w13, g.first, g.second, pool, reps, l2, 0, true);
    }
    // How far does the mma advantage go? Rows per expert is what the verify width buys.
    printf("\n[GEMV vs mma as rows per expert rises (18 experts, DRAM bytes fixed)]\n");
    for (int R : {2, 4, 8}){
        std::vector<int> g(18, R);
        char lab[64]; snprintf(lab, sizeof lab, "18 experts x %d rows", R);
        setenv("MOE_RB", "8", 1); bench(w13, g, 8, pool, reps, lab, 0, false); unsetenv("MOE_RB");
        bench(w13, g, 8, pool, reps, lab, 0, true);
    }

    // ===================== PREFILL GROUPING (2026-08-26) =====================
    // Everything above is decode/verify: <= 8 rows per expert, a handful of experts. PREFILL is a
    // different regime and the engine's largest single mark. At PS=845 with 6 experts per token,
    // 845*6 = 5070 rows spread over ALL 160 experts = ~32 rows each -- 4x past the widest case
    // benched above, and every expert hit, so the ideal DRAM traffic is the whole expert set read
    // once. dprof says the MoE is 5044 ms with MOE_MMA=1 against a ~415 ms bandwidth floor; this is
    // where that 12x either shows up as redundancy or as plain inefficiency.
    printf("\n[PREFILL grouping: 160 experts, rows per expert varying, GEMV vs mma]\n");
    printf("  ideal DRAM = every expert read once; mma dequants once per 16-row tile\n");
    for (int R : {8, 16, 32, 64}){
        std::vector<int> g(160, R);
        char lab[64]; snprintf(lab, sizeof lab, "160 experts x %2d rows", R);
        setenv("MOE_RB", "8", 1); bench(w13, g, 8, pool, reps, lab, 0, false); unsetenv("MOE_RB");
        bench(w13, g, 8, pool, reps, lab, 0, true);
    }
    printf("\n[PREFILL, w2 shape]\n");
    {
        std::vector<int> g(160, 32);
        setenv("MOE_RB", "8", 1); bench(w2, g, 8, pool, reps, "160 x 32 rows", 0, false); unsetenv("MOE_RB");
        bench(w2, g, 8, pool, reps, "160 x 32 rows", 0, true);
    }
    return 0;
}
