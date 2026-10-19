// gate_hadamard_alias.cu — DECODE_LADDER 1.10. THE ALIASED HADAMARD.
//
// WHAT IS BEING TESTED. `hadamard_kernel` gives thread (r, j) the whole of row r:
//
//     for (int i = 0; i < D; ++i) acc += (__popc(i & j) & 1) ? -xr[i] : xr[i];
//     y[idx] = acc * scale;
//
// so every thread of a row READS all D elements of that row and WRITES one of them. That is correct
// only while `y != x`. Three call sites pass the SAME pointer twice:
//
//     kernels/compressor.cu:665      hadamard(out,  out,  groups, d)   <- PREFILL, groups = s/ratio
//     kernels/compressor.cu:713      hadamard(dst,  dst,  1,      d)   <- decode single-group emit
//     kernels/compressed_decode.cu:633 hadamard(cand, cand, 1,    d)   <- decode candidate emit
//
// and every other call site passes distinct buffers -- indexer.cu:417 even carries the comment
// `// out!=in`. Whether a reading thread sees the pre- or the post-transform value of an element
// its neighbours are overwriting is a scheduling outcome, so the result is nondeterministic.
//
// WHY THIS IS THE 1.10 CANDIDATE. `compressor_forward`'s `rotate` branch runs ONLY for the indexer's
// compressor, and `indexer_forward` runs ONLY on `compress_ratio == 4` layers. 1.9's own logs, read
// all-pairs by tools/lhash_pairs.py, put all 56 first-differing layers at ratio 4 and not one at
// ratio 128 or 0 -- ratio-128 layers run `compressor_forward` (non-overlap, rotate=false) and
// `sparse_attn` and never broke first.
//
// THE PREDICTION THIS GATE WAS WRITTEN TO TEST, AND HOW IT WAS WRONG. Threads of one row race only
// when they are not co-scheduled. blockDim is 256 and D is 128, so one block owns exactly two rows
// and `blocks = ceil(rows/2)`; the pre-registered claim was that while `blocks <= SM count` every
// block has an SM to itself, its warps issue in lockstep, the D-iteration read loop finishes
// everywhere before the single closing store lands anywhere, and the defect therefore has a SHARP
// boundary at rows = 40 on this 20-SM box (prefill s = 163/164 at ratio 4).
//
// MEASURED, 2026-08-20, 8 repeats per row count: rows 41, 42, 48 and 56 came back CLEAN and rows 44
// raced. It is not a step, it is a RATE that rises with the block count -- 0/8 to 20 blocks, 1/8 at
// 22, 2/8 at 32, 6/8 at 48, 8/8 with 8 distinct results from 64 blocks up. A step function and a
// rising rate are indistinguishable at ONE sample per length, which is exactly what 1.9's length
// ladder had, and it is why `--repeats` exists here. What survives of the prediction is the
// DIRECTION and the scale: nothing fires while the grid fits the machine once, everything fires
// well past it. Do not quote a threshold from this gate without quoting the repeat count with it.
//
// EXIT CODE IS THE RATCHET, AND BOTH ARMS RUN IN ONE PROCESS. `hadamard_set_stage(0)` is the
// pre-1.10 flat kernel and `(1)` is the shipped staged one, on the same binary and the same input,
// so "the fix is what changed the verdict" is a measurement and not a comparison of two builds --
// the same reason index_score has `index_score_impl` and gemm_fp32 has `gemm_fp32_set_tile`. A gate
// that passes both ways proves nothing, so this one REQUIRES the OFF arm to fail past the boundary:
// it returns non-zero if the ON arm ever differs from the reference, AND non-zero if the OFF arm
// never differs (the harness has stopped being able to see the bug it was written for).
// `--control` corrupts one reference element by one ulp: EVERY comparison on both arms must then
// differ, and the gate returns 0 only if every one of them did. It is a separate invocation with its
// own meaning -- "the memcmp can see a one-ulp change" -- not a second way to run the main gate.
//
//   nvcc -O2 -std=c++17 -gencode arch=compute_110a,code=sm_110a -I include \
//     tests/gate_hadamard_alias.cu kernels/indexer.cu kernels/compressor.cu kernels/mla_attn.cu \
//     kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/dscratch.cu kernels/nvfp4_dense.cu \
//     -o build/gate_hadamard_alias
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <set>
#include <string>
#include <cuda_runtime.h>
#include "indexer.h"
#include "deepseek_v4.h"

#define CK(x) do{ cudaError_t e=(x); if(e){ fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2);} }while(0)

static const int D = dsv4::INDEX_HEAD_DIM;      // 128 -- the only D any aliased call site uses
static int REPEATS = 8;    // --repeats N; the defect is a RATE, so the sample size is a knob

// Deterministic input, and DELIBERATELY not tiny: the race swaps a pre-transform element for a
// post-transform one, and post-transform values are ~sqrt(D) larger, so any input distribution
// makes the substitution visible. Fixed seed so two invocations of this gate are comparable.
static void fill(std::vector<float>& v, unsigned seed) {
    unsigned s = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = ((float)(s >> 8) / (float)(1u << 24)) * 2.f - 1.f;
    }
}

struct ArmResult { int bad = 0, first_bad = -1, negctl = 0; };

static ArmResult run_arm(const char* label, int stage, const std::vector<int>& sweep,
                         float* dx, float* dy, float* dz, int rows_boundary, int sms, bool control) {
    hadamard_set_stage(stage);
    printf("--- ARM %s (hadamard_set_stage(%d)) ---\n", label, stage);
    printf("%6s %7s  %-34s %-12s %s\n", "rows", "blocks", "aliased hadamard(y,y)", "non-aliased", "verdict");
    ArmResult R;
    for (int rows : sweep) {
        const size_t n = (size_t)rows * D;
        const int blocks = (int)((n + 255) / 256);
        std::vector<float> hx(n), href(n), hgot(n);
        fill(hx, (unsigned)rows);

        // Reference: distinct buffers, and taken from the SHIPPED kernel, because the claim this
        // gate makes about the non-aliased call sites is that they are unchanged.
        hadamard_set_stage(1);
        CK(cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice));
        hadamard(dy, dx, rows, D, 0); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(href.data(), dy, n * 4, cudaMemcpyDeviceToHost));
        hadamard_set_stage(stage);
        // --control: one ulp into the reference. Nothing downstream is allowed to still match it.
        if (control) { unsigned u; memcpy(&u, &href[n / 2], 4); u ^= 1u; memcpy(&href[n / 2], &u, 4); }

        // NEGATIVE CONTROL, and it is not decoration: if the non-aliased path were itself unstable
        // then "the aliased one differs" would be a statement about the harness. It also carries
        // the OTHER half of the bit-exactness claim -- this arm's non-aliased output must equal the
        // shipped kernel's, which is what says the fix changed nothing where nothing was wrong.
        int negctl_diff = 0;
        for (int r = 0; r < REPEATS; ++r) {
            CK(cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice));
            hadamard(dy, dx, rows, D, 0); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hgot.data(), dy, n * 4, cudaMemcpyDeviceToHost));
            if (memcmp(hgot.data(), href.data(), n * 4)) ++negctl_diff;
        }

        // The arm: y == x, exactly as kernels/compressor.cu:665 calls it.
        int alias_diff = 0; std::set<std::string> distinct;
        for (int r = 0; r < REPEATS; ++r) {
            CK(cudaMemcpy(dz, hx.data(), n * 4, cudaMemcpyHostToDevice));
            hadamard(dz, dz, rows, D, 0); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hgot.data(), dz, n * 4, cudaMemcpyDeviceToHost));
            if (memcmp(hgot.data(), href.data(), n * 4)) ++alias_diff;
            distinct.insert(std::string((const char*)hgot.data(), n * 4));
        }

        if (control) {
            // The ONLY question here is whether a one-ulp difference is visible at all.
            const bool live = (alias_diff == REPEATS) && (negctl_diff == REPEATS);
            if (!live) { ++R.bad; if (R.first_bad < 0) R.first_bad = rows; }
            printf("%6d %7d  %2d/%d differ, %2zu distinct result%s  %2d/%d differ  %-10s\n",
                   rows, blocks, alias_diff, REPEATS, distinct.size(), distinct.size() == 1 ? " " : "s",
                   negctl_diff, REPEATS, live ? "SEES THE ULP" : "MEMCMP BLIND");
            continue;
        }
        const bool ok = (alias_diff == 0);
        if (!ok) { ++R.bad; if (R.first_bad < 0) R.first_bad = rows; }
        if (negctl_diff) ++R.negctl;
        const bool predicted_ok = (rows <= rows_boundary);
        printf("%6d %7d  %2d/%d differ, %2zu distinct result%s  %2d/%d differ  %-10s%s\n",
               rows, blocks, alias_diff, REPEATS, distinct.size(), distinct.size() == 1 ? " " : "s",
               negctl_diff, REPEATS, ok ? "OK" : "ALIAS RACE",
               (stage == 0 && predicted_ok != ok) ? "   <- PREDICTION MISSED" : "");
    }
    printf("\n");
    (void)sms;
    return R;
}

// `--bench`: time both arms at the shapes the engine issues. The staged kernel replaces D global
// loads per thread with D shared loads per thread plus D/blockDim global loads per BLOCK, so it is
// not only the correct kernel, it moves the inner loop off L1. Whether that is worth anything at
// engine scale is a separate measurement (scripts/hadamard_ab_run.sh) -- this is the mechanism
// number, and it is the one that says whether to look for the win end to end at all.
static void bench(const std::vector<int>& rows_list, float* dx, float* dy) {
    const int ITERS = 200, WARM = 20;
    printf("--- BENCH: hadamard(y, x) with y != x, D=%d, %d iters after %d warm-up ---\n", D, ITERS, WARM);
    printf("%8s %10s %12s %12s %8s   %s\n", "rows", "blocks", "flat ms", "staged ms", "ratio", "where the engine issues it");
    const char* where[] = {"decode: both aliased emit call sites",
                           "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                           "", "", "", "", "", ""};
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (size_t k = 0; k < rows_list.size(); ++k) {
        const int rows = rows_list[k];
        float ms[2];
        for (int arm = 0; arm < 2; ++arm) {
            hadamard_set_stage(arm);
            for (int i = 0; i < WARM; ++i) hadamard(dy, dx, rows, D, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(a));
            for (int i = 0; i < ITERS; ++i) hadamard(dy, dx, rows, D, 0);
            CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms[arm], a, b)); ms[arm] /= ITERS;
        }
        printf("%8d %10d %12.5f %12.5f %7.2fx   %s\n", rows, (rows * D + 255) / 256,
               ms[0], ms[1], ms[1] > 0 ? ms[0] / ms[1] : 0.f, k < 1 ? where[0] : "");
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    printf("\n");
}

// `--concurrent`: the shape the SERVER issues, under the condition the server issues it in.
//
// The row sweep above finds rows = 1 and 2 clean at 200 repeats, and the two aliased call sites
// decode reaches are BOTH rows = 1 -- which said the decode path could not race and neatly explained
// why 1.5's 16 server legs were byte-identical. It is wrong, and the cross-run matrix in ladder 1.10
// is what caught it: 1.11's four server loads are 18/18 byte-identical, 1.12's four are 0/18 against
// EACH OTHER, and 1.12 changed `gemm_fp32` and nothing in this file.
//
// A single 128-thread block is four warps that issue in lockstep ONLY while they have the SM to
// themselves. `compressed_verify_step_indexer` forks the compressor emits onto `g_side` and runs
// them CONCURRENTLY with main-stream work, so those four warps share an SM with whatever else is
// resident -- and what else is resident is exactly what 1.12 changed. This mode reproduces that:
// a filler kernel occupying every SM on a second stream, then the aliased rows = 1 hadamard.
__global__ void filler_kernel(float* sink, int iters) {
    float a = (float)(blockIdx.x * blockDim.x + threadIdx.x) * 1e-6f;
    for (int i = 0; i < iters; ++i) a = a * 1.0000001f + 1e-7f;
    if (a == 12345.678f) sink[0] = a;                 // never true; keeps the loop alive
}

static int concurrent_probe(int reps) {
    int sms = 0; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    cudaStream_t side; CK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    float *dx, *dy, *dz, *dsink;
    const size_t n = (size_t)1 * D;
    CK(cudaMalloc(&dx, n * 4)); CK(cudaMalloc(&dy, n * 4)); CK(cudaMalloc(&dz, n * 4));
    CK(cudaMalloc(&dsink, 4));
    std::vector<float> hx(n), href(n), hgot(n);
    fill(hx, 11u);
    hadamard_set_stage(1);
    CK(cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice));
    hadamard(dy, dx, 1, D, 0); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(href.data(), dy, n * 4, cudaMemcpyDeviceToHost));

    printf("--- CONCURRENT: aliased hadamard(y,y) at rows=1, with a filler kernel on a second "
           "stream (%d SMs x 4 blocks) ---\n", sms);
    printf("%-34s %-16s %s\n", "arm", "differ", "distinct results");
    int off_bad = 0;
    for (int arm = 0; arm < 2; ++arm) {
        hadamard_set_stage(arm);
        int diff = 0; std::set<std::string> distinct;
        for (int r = 0; r < reps; ++r) {
            filler_kernel<<<sms * 4, 256, 0, side>>>(dsink, 200000);
            CK(cudaMemcpy(dz, hx.data(), n * 4, cudaMemcpyHostToDevice));
            hadamard(dz, dz, 1, D, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hgot.data(), dz, n * 4, cudaMemcpyDeviceToHost));
            if (memcmp(hgot.data(), href.data(), n * 4)) ++diff;
            distinct.insert(std::string((const char*)hgot.data(), n * 4));
        }
        if (arm == 0) off_bad = diff;
        printf("%-34s %3d/%-12d %zu\n", arm ? "ON  = shipped staged kernel" : "OFF = pre-1.10 flat kernel",
               diff, reps, distinct.size());
    }
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dsink); cudaStreamDestroy(side);
    printf("%s\n\n", off_bad ? "-> rows=1 DOES race once something else is resident. The row sweep's "
                                "\"decode cannot race\" reading was an artefact of measuring the kernel ALONE."
                              : "-> rows=1 did not race even under a co-resident filler.");
    return off_bad;
}

int main(int argc, char** argv) {
    bool control = false, dobench = false, doconc = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--control")) control = true;
        else if (!strcmp(argv[i], "--bench")) dobench = true;
        else if (!strcmp(argv[i], "--concurrent")) doconc = true;
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) REPEATS = atoi(argv[++i]);
    }
    if (REPEATS < 1) REPEATS = 1;

    int sms = 0; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int rows_boundary = sms * (256 / D);          // pre-1.10 mapping: 2 rows per 256-thread block
    printf("gate_hadamard_alias: D=%d  SMs=%d  pre-1.10 blockDim=256 -> %d rows/block\n", D, sms, 256 / D);
    printf("PREDICTION for the OFF arm: aliased == reference while blocks <= %d SMs, i.e. rows <= %d "
           "(prefill s <= %d at ratio 4); broken from rows %d (s >= %d).\n",
           sms, rows_boundary, rows_boundary * 4 + 3, rows_boundary + 1, (rows_boundary + 1) * 4);
    printf("The ON arm must be clean at EVERY row count.\n\n");
    if (control) printf("--control: one reference element is bumped by one ulp; EVERY row must FAIL.\n\n");

    // rows the engine actually issues (1 = both decode emit call sites), then a dense sweep across
    // the predicted boundary, then well past it.
    std::vector<int> sweep = {1, 2, 4, 8, 16, 24, 32, 36, 38, 39, 40, 41, 42, 44, 48, 56, 64, 96, 128, 256, 768};

    float *dx, *dy, *dz;
    const size_t maxel = (size_t)sweep.back() * D;
    CK(cudaMalloc(&dx, maxel * 4)); CK(cudaMalloc(&dy, maxel * 4)); CK(cudaMalloc(&dz, maxel * 4));

    if (doconc) concurrent_probe(REPEATS);

    if (dobench) {
        // The shapes the engine issues: 1 = the two aliased emit call sites; 64 = q-side per verify
        // step; 384 = q-side at K=6; 256/768 = the prefill compressor at s = 1,024 / 3,072.
        std::vector<int> hot = {1, 64, 256, 384, 768};
        std::vector<float> hx((size_t)sweep.back() * D); fill(hx, 7u);
        CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        bench(hot, dx, dy);
    }

    ArmResult off = run_arm("OFF = pre-1.10 flat kernel", 0, sweep, dx, dy, dz, rows_boundary, sms, control);
    ArmResult on  = run_arm("ON  = shipped staged kernel", 1, sweep, dx, dy, dz, rows_boundary, sms, control);

    cudaFree(dx); cudaFree(dy); cudaFree(dz);

    if (control) {
        const int blind = off.bad + on.bad;
        if (blind) {
            printf("CONTROL FAILED: %d row counts still MATCHED a reference corrupted by one ulp. "
                   "The memcmp in this gate is blind and every PASS it has ever printed is void.\n", blind);
            return 4;
        }
        printf("CONTROL PASS: with one ulp flipped in the reference, every comparison on both arms "
               "differed at all %zu row counts. The memcmp is live.\n", sweep.size());
        return 0;
    }

    if (off.negctl || on.negctl) {
        printf("NEGATIVE CONTROL FAILED (OFF %d, ON %d row counts): the NON-aliased hadamard did not "
               "reproduce the shipped kernel's output, so nothing here is attributable to aliasing.\n",
               off.negctl, on.negctl);
        return 3;
    }
    printf("negative control: hadamard(y,x) with y != x reproduced the shipped kernel bit for bit, "
           "%d/%d repeats, at every row count, on BOTH arms -- so the staging changed nothing where "
           "the buffers were already distinct, and every difference below is aliasing.\n\n", REPEATS, REPEATS);

    printf("OFF arm: %d of %zu row counts race, first at rows=%d (blocks=%d, SMs=%d).\n",
           off.bad, sweep.size(), off.first_bad, off.first_bad > 0 ? (off.first_bad * D + 255) / 256 : 0, sms);
    printf("ON  arm: %d of %zu row counts race.\n\n", on.bad, sweep.size());

    if (!off.bad) {
        printf("FAIL: the OFF arm did not reproduce the defect at ANY row count. This gate can no "
               "longer see the bug it was written for; a PASS from it would mean nothing.\n");
        return 2;
    }
    if (on.bad) {
        printf("FAIL: hadamard(y,y) still differs from hadamard(y,x) at %d row counts on the shipped "
               "kernel, first at rows=%d.\n", on.bad, on.first_bad);
        return 1;
    }
    printf("PASS: the pre-1.10 kernel races and the shipped one does not -- hadamard(y,y) == "
           "hadamard(y,x) at every row count, %d repeats each.\n", REPEATS);
    return 0;
}
