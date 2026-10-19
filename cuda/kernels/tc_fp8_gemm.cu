// tc_fp8_gemm.cu — native FP8 tensor-core GEMM (W8A8) via mma.sync.m16n8k32.e4m3 (2x fp16 on Thor sm_110a).
// C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T, per-128 act scale a_s[M,K/128], per-128x128 wt scale b_s[N/128,K/128].
// Drop-in for fp8_block_gemm; fp8 acts/weights feed the tensor core directly (no fp16 upconvert).
// *** UNGATED until it passes cosine vs fp8_block_gemm (tests: gate_units [fp8_block_gemm+TC]). ***
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// mma D[16,8] += A[16,32] @ B[8,32]^T ; A/B = e4m3 (4 fp8 per reg), C/D = f32 (4 per lane).
__device__ __forceinline__ void mma_m16n8k32_e4m3(float* c, const unsigned* a, const unsigned* b){
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]));
}

// grid = (ceil(N/8), ceil(M/16)); one warp -> one [16-row, 8-col] tile. Accumulate raw per 128-K-block, then
// scale (a_s per-row, b_s per-128-N-block) and add — matches fp8_block_gemm's per-block scale application.
__global__ void tc_fp8_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                              const uint8_t* __restrict__ B, const float* __restrict__ bs, int M, int N, int K){
    int lane=threadIdx.x&31, gid=lane>>2, tid4=lane&3;
    int nb=blockIdx.x, mt=blockIdx.y; int n0=nb*8, m0=mt*16; if(n0>=N) return;
    int r0=m0+gid, r1=m0+gid+8, nn=n0+gid; int KB=K/128;
    float acc[4]={0.f,0.f,0.f,0.f};
    for(int kblk=0; kblk<KB; ++kblk){
        float cb[4]={0.f,0.f,0.f,0.f};
        #pragma unroll
        for(int kt=0; kt<4; ++kt){ int k0=kblk*128+kt*32;
            unsigned a[4], b[2];
            a[0]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4)      : 0u;
            a[1]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4)      : 0u;
            a[2]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4+16)   : 0u;
            a[3]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4+16)   : 0u;
            b[0]=(nn<N)? *(const unsigned*)(B+(size_t)nn*K+k0+tid4*4)      : 0u;
            b[1]=(nn<N)? *(const unsigned*)(B+(size_t)nn*K+k0+tid4*4+16)   : 0u;
            mma_m16n8k32_e4m3(cb, a, b);
        }
        float bsc = bs[(size_t)(n0/128)*KB + kblk];
        float as0 = (r0<M)? as[(size_t)r0*KB + kblk] : 0.f;
        float as1 = (r1<M)? as[(size_t)r1*KB + kblk] : 0.f;
        acc[0]+=cb[0]*as0*bsc; acc[1]+=cb[1]*as0*bsc; acc[2]+=cb[2]*as1*bsc; acc[3]+=cb[3]*as1*bsc;
    }
    int cn=tid4*2;
    if(r0<M && n0+cn  <N) C[(size_t)r0*N + n0+cn  ]=acc[0];
    if(r0<M && n0+cn+1<N) C[(size_t)r0*N + n0+cn+1]=acc[1];
    if(r1<M && n0+cn  <N) C[(size_t)r1*N + n0+cn  ]=acc[2];
    if(r1<M && n0+cn+1<N) C[(size_t)r1*N + n0+cn+1]=acc[3];
}

// ---------------------------------------------------------------------------------------------
// SMEM-STAGED B (LOOP_LOG Finding 41). Measured on wq_b [32768,1024], 12 rotating copies so every
// call reads memory it has not just touched:
//
//            M=1     M=2     M=3     M=5     M=8
//   HOT     .145    .248    .353    .153    .203     <- m16 at M=5/8; compute is NOT the problem
//   COLD    .157    .499    .505    .488    .503     <- same kernel, 3.2x slower off DRAM
//
// A kernel that is 3.2x slower cold than hot is not compute-bound and not bandwidth-bound; it is
// fetching bytes it does not use. `tc_fp8_kernel` issues its B operand as `*(const unsigned*)(B +
// nn*K + k0 + tid4*4)` — four lanes covering 16 CONTIGUOUS bytes of one row, eight rows per warp,
// strided by K. Sixteen bytes is half a 32-byte sector, and the other half arrives on a separate
// instruction, so cold every sector is fetched for half its payload.
//
// The mma tiling is right; only the path from DRAM is wrong. Stage a [64 x 128] B tile through shared
// memory with eight consecutive lanes covering 128 contiguous bytes of one row — one full 128-byte
// line per eight lanes, nothing over-fetched — then read mma fragments back out of smem. The tile
// is not reused (each warp reads only its own 8 rows), so this buys exactly one thing: the global
// access pattern. That is the whole 3.2x.
//
// Row stride is padded 128 -> 144 bytes (36 words). The fragment read has lane group g = lane>>2
// touching row g at word (kt*8 + (lane&3)); at stride 36 words row g starts at bank (36g)%32 = 4g,
// so the eight groups land on banks 4g..4g+3 — a permutation of all 32, conflict-free. At stride 32
// every row would start at bank 0 and all eight groups would collide 8 ways.
// Tile height is adaptive. With W warps a block stages 8W B-rows, so the grid is ceil(N/8W) blocks
// — and at W=8 the small-N shapes do not fill the device: wkv [512,4096] gives 8 blocks for 20 SMs
// and measured 38 GB/s where wo_b [4096,4096] measured 199. Total warp count is N/8 either way, so
// shrinking W costs nothing and buys blocks. W is chosen so the grid is >= 64 blocks where N allows.
// Staging is 32 bytes/thread at every W (8W rows x 128 B over 32W threads), so the mapping below is
// independent of W and eight consecutive lanes always walk one row contiguously.
//
// ALIGNMENT (LOOP_LOG Finding 25, re-learned the hard way here). B is a mapped safetensors tensor.
// The old kernel loaded it with `*(const unsigned*)`, so 4-byte alignment was all it ever needed and
// nothing in the engine guaranteed more. A uint4 stage crashed prefill on the first real weight
// (`misaligned address`, mla_forward.cu:81 -> wo_b) and the unit gate did NOT catch it, because a
// gate that allocates B with cudaMalloc always gets a 256-byte-aligned pointer. So the staging loop
// is templated on alignment: 1 x uint4 when the base is 16-byte aligned, else 4 x unsigned covering
// the same 16 bytes. The four-instruction form touches the identical cache lines back to back, so
// the coalescing — the entire point of the staging — survives; only the instruction count grows.
// K-CHUNKED STAGING (LOOP_LOG Finding 74). The staging loop above used to run once per 128-K-block:
// two uint4 per thread, then __syncthreads, then the mma, then __syncthreads again. That is ONE
// round trip to DRAM per barrier pair with nothing else in flight, and the total warp count of this
// kernel is structurally N/8 — so the small-N shapes have neither enough warps NOR enough bytes per
// warp to cover the latency. Measured in situ (evidence/moescan.log, same weight bytes at both K):
//
//     mark     bytes/step   K=1 (GEMV path)   K=5 (this tile)
//     q:wq_a     0.172 GB    1.49 ms (115 GB/s)   6.71 ms ( 26 GB/s)   <- N=1024 -> 64 blk x 2 warps
//     q:wq_b     1.376 GB    7.04 ms (195 GB/s)  11.18 ms (123 GB/s)
//     o:wo_b     1.376 GB    7.43 ms (185 GB/s)   9.76 ms (141 GB/s)
//
// Same bytes, 1.3-4.5x the time. The M=1 GEMV proves 185-195 GB/s is reachable on these exact rows,
// so this is not a roofline; it is memory-level parallelism.
//
// Fix: stage KC consecutive K-blocks per barrier pair. Per-thread staging is 32*KC bytes and is
// INDEPENDENT of WARPS (8W rows x KC*128 B over 32W threads), so KC alone sets bytes-in-flight, and
// the barrier count drops by KC. Loads are issued into a register array FIRST and stored to smem
// after — a load->store->load loop would serialise at one outstanding miss no matter how large KC is.
//
// The accumulation is unchanged: `cb` is still built per 128-block from four mma, and `acc` still
// takes them in kblk order 0,1,2,..., so this is BIT-EXACT, not merely close (contrast Finding 68).
//
// KC is picked from the smem budget, which automatically gives the starved shapes the most: the
// staged tile is 8W*(KC*128+16) bytes, so W=8 -> KC=2 and W=2 -> KC=8. The starved shapes also have
// the registers to spare — wq_a runs 128 warps on the whole device, so 8*KC registers of staging
// costs nothing there.
//
// DOUBLE BUFFERING (LOOP_LOG Finding 78). KC raised the bytes in flight per round but left the round
// structure serial: sync -> issue NH loads -> WAIT for all of them -> store -> sync -> mma. The mma
// phase of round n has nothing in flight, and the load phase of round n+1 cannot start until every
// warp has finished round n's mma. So DRAM latency is exposed once per round no matter how large KC
// is; KC only amortises it over more bytes. `DB=true` issues round n+1's global loads into a SECOND
// register array immediately after the store barrier, i.e. BEFORE round n's mma, so the loads have
// the whole mma phase to land:
//
//     DB=0:  [sync] load..wait  store [sync] mma        [sync] load..wait  store [sync] mma
//     DB=1:  ...store [sync] issue(n+1) mma(n)  [sync] store(n+1) [sync] issue(n+2) mma(n+1) ...
//
// Cost is NH more live uint4 (8*KC registers) across the mma, which is why `ptxas -v` is the first
// instrument here and not the bench (trap 19): at <8,2> the shipped instantiation sits at 64
// registers, and 16 more crosses a 256-thread occupancy step. Arithmetic is untouched — the same
// bytes reach smem in the same order and `acc` still takes `cb` in kblk order — so this is BIT-EXACT
// and `gate_tc_fp8_kc` checks it as equality, not cosine.
//
// THE OCCUPANCY STEP IS THE WHOLE STORY, and it is worth only FOUR registers. Unbounded, DB takes
// <8,2> from 64 to 68 registers; at 256 threads that is 65536/(256*72) = 3 blocks/SM where 64 gives
// 4 — a 25 % occupancy loss for +4 registers. `evidence/db_bench.log` (COLD, m16+smem B+4, M=5):
// wo_b **+38.5 %**, sw2 +20.7 %, wq_b +13.5 %. With `__launch_bounds__` restoring 4 blocks/SM
// (64 registers, 8 bytes of spill) the same arms are `evidence/db_bench2.log`: wo_b +3.1 %, sw2
// −1.0 %, wq_b +2.4 %, wq_a −0.8 %, and the two shapes that were never at the occupancy cliff gain —
// sw1/3 (W=4) **−7.2 %** and wkv (W=1) −3.2 %.
#define TCB_PAD 16                                   // row-stride pad, in bytes
#ifndef TCB_DB_BPS                                   // blocks/SM the DB variant must still fit (see above)
#define TCB_DB_BPS 4
#endif
// UNROLL OF THE CP.ASYNC ISSUE LOOP, and it is the whole lever. `cp.async` is fire-and-forget, so a
// ROLLED loop still has all H copies in flight — but an UNROLLED one keeps H addresses live at once,
// which is trap 19 H times over. `ptxas -v` on the shipped shape `<8, NS=4, AL16=false>`:
//
//     UF     regs   smem     blocks/SM        (shipped smemB<8,2,false>: 64 regs, 17408 B, 4)
//      8      78    36864        3     <- FAILS the bar, the F78 failure mode by a different door
//      4      56    36864        4
//      2      48    36864        5
//      1      48    36864        5     <- 48 regs is 16 BELOW the kernel it replaces
//
// So the register array really is given back — but only if the loop is left rolled. Fully unrolled it
// is worse than the kernel it replaces. Default 1.
#ifndef TCB_CPA_UF
#define TCB_CPA_UF 1
#endif
// The body is a __device__ function so the DB=false ENTRY can stay attribute-free. An
// `__launch_bounds__(256,1)` on the shared template silently took the DB=false <8,2> instantiation
// from 64 to 84 registers — minBlocks=1 tells ptxas one resident block is enough — which would have
// changed the control arm of this very A/B. Only the DB entry carries the bound.
template<int WARPS, int KC, bool AL16, bool DB>
__device__ __forceinline__ void tc_fp8_smemB_body(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                    const uint8_t* __restrict__ B, const float* __restrict__ bs, int M, int N, int K){
    constexpr int LD = KC*128 + TCB_PAD;             // padded row stride in bytes
    constexpr int CPR = KC*8;                        // 16-byte chunks per staged row
    constexpr int NH  = 2*KC;                        // uint4 staged per thread per round
    __shared__ uint8_t sB[WARPS*8*LD];
    const int t=threadIdx.x, warp=t>>5, lane=t&31, gid=lane>>2, tid4=lane&3;
    const int n0blk=blockIdx.x*(WARPS*8), m0=blockIdx.y*16;
    const int n0=n0blk+warp*8;
    const int r0=m0+gid, r1=m0+gid+8, KB=K/128;
    float acc[4]={0.f,0.f,0.f,0.f};
    // stage B[n0blk .. n0blk+8W)[kb*128 .. +KC*128) into NH x uint4 per thread.
    // chunk c: row=c/CPR, byte offset=(c%CPR)*16 -> lanes 0..CPR-1 walk one row contiguously.
    auto issue = [&](uint4* dst, int kb){
        #pragma unroll
        for(int h=0; h<NH; ++h){
            const int c=t+h*(WARPS*32), row=c/CPR, off=(c%CPR)*16, gn=n0blk+row;
            dst[h]=make_uint4(0,0,0,0);
            if(gn<N){
                const uint8_t* src=B+(size_t)gn*K+(size_t)kb*128+off;
                if(AL16) dst[h]=*(const uint4*)src;
                else { dst[h].x=*(const unsigned*)(src);    dst[h].y=*(const unsigned*)(src+4);
                       dst[h].z=*(const unsigned*)(src+8);  dst[h].w=*(const unsigned*)(src+12); }
            }
        }
    };
    uint4 v[NH];
    if constexpr(DB) issue(v, 0);
    for(int kb0=0; kb0<KB; kb0+=KC){
        // At LD=(32*KC+4) words, row g starts at bank 4g, so the eight fragment groups are a
        // permutation of all 32 banks for every KC — the same conflict-free property as LD=144.
        __syncthreads();
        if constexpr(!DB) issue(v, kb0);
        #pragma unroll
        for(int h=0; h<NH; ++h){
            const int c=t+h*(WARPS*32), row=c/CPR, off=(c%CPR)*16;
            *(uint4*)&sB[row*LD+off]=v[h];
        }
        __syncthreads();
        // Round n+1's loads are issued HERE, before the mma below, so they overlap it. The guard is
        // block-uniform; on the last round nothing is issued and `v` is simply not read again.
        uint4 vn[DB ? NH : 1];
        if constexpr(DB) { if(kb0+KC<KB) issue(vn, kb0+KC); }
        #pragma unroll 1
        for(int kc=0; kc<KC; ++kc){
            const int kblk=kb0+kc;
            float cb[4]={0.f,0.f,0.f,0.f};
            #pragma unroll
            for(int kt=0; kt<4; ++kt){ const int k0=kblk*128+kt*32;
                unsigned a[4], b[2];
                a[0]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4)    : 0u;
                a[1]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4)    : 0u;
                a[2]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4+16) : 0u;
                a[3]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4+16) : 0u;
                const int so=(warp*8+gid)*LD + kc*128 + kt*32 + tid4*4;
                b[0]=*(const unsigned*)&sB[so];
                b[1]=*(const unsigned*)&sB[so+16];
                mma_m16n8k32_e4m3(cb, a, b);
            }
            const float bsc = bs[(size_t)(n0/128)*KB + kblk];
            const float as0 = (r0<M)? as[(size_t)r0*KB + kblk] : 0.f;
            const float as1 = (r1<M)? as[(size_t)r1*KB + kblk] : 0.f;
            acc[0]+=cb[0]*as0*bsc; acc[1]+=cb[1]*as0*bsc; acc[2]+=cb[2]*as1*bsc; acc[3]+=cb[3]*as1*bsc;
        }
        if constexpr(DB){
            #pragma unroll
            for(int h=0; h<NH; ++h) v[h]=vn[h];
        }
    }
    const int cn=tid4*2;
    if(r0<M && n0+cn  <N) C[(size_t)r0*N + n0+cn  ]=acc[0];
    if(r0<M && n0+cn+1<N) C[(size_t)r0*N + n0+cn+1]=acc[1];
    if(r1<M && n0+cn  <N) C[(size_t)r1*N + n0+cn  ]=acc[2];
    if(r1<M && n0+cn+1<N) C[(size_t)r1*N + n0+cn+1]=acc[3];
}
// DB=false entry: no attribute, so this is codegen-identical to the pre-Finding-78 kernel.
template<int WARPS, int KC, bool AL16>
__global__ void tc_fp8_smemB_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                    const uint8_t* __restrict__ B, const float* __restrict__ bs, int M, int N, int K){
    tc_fp8_smemB_body<WARPS,KC,AL16,false>(C,A,as,B,bs,M,N,K);
}
// DB=true entry. The bound reproduces the occupancy the DB=false kernel already achieves at each W
// (measured with `ptxas -v`: W=8 -> 64 regs = 4 blocks of 256 threads, W=2 -> 64 regs = 16 blocks of
// 64, W=1 -> 128 regs = 16 blocks of 32). Without it the extra live uint4 take <8,2> to 68 registers,
// which is 4 blocks/SM -> 3.
template<int WARPS, int KC, bool AL16>
__global__ void __launch_bounds__(32*WARPS, WARPS>=8 ? TCB_DB_BPS : (WARPS>=2 ? 32/WARPS : 16))
                tc_fp8_smemB_db_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                    const uint8_t* __restrict__ B, const float* __restrict__ bs, int M, int N, int K){
    tc_fp8_smemB_body<WARPS,KC,AL16,true>(C,A,as,B,bs,M,N,K);
}

// ---------------------------------------------------------------------------------------------
// CP.ASYNC MULTI-STAGE RING (LEVERS.md B8-cpasync). The lever the cycle-15 research phase promoted,
// and it is new EVIDENCE against Finding 78's specific measurement rather than a re-argument.
//
// F78 double-buffered in REGISTERS: the same warp issues round n+1's global loads into a second
// `uint4 v[NH]` array and then issues round n's mma. Two things follow and both were measured.
// (a) The buffer is only 2 deep and the issuing warp is the consuming warp, so one round of mma
//     (8 mma plus L1/L2-resident A loads, a few hundred cycles) is all the cover a ~600-cycle miss
//     gets — it hides part of the latency and no more. Measured +0.28 % in situ.
// (b) The second live array costs registers, and 4 of them cross a 256-thread occupancy step
//     (64 -> 68 -> 3 blocks/SM instead of 4, trap 21). `__launch_bounds__` bought the blocks back at
//     the price of 8 bytes of spill.
//
// `cp.async` fixes BOTH at once, and it is the only form of this that is implementable here:
//   - The copy is asynchronous IN HARDWARE. There is no register array at all — global bytes land in
//     shared memory without ever being named in the register file — so (b) inverts: the staging
//     registers are GIVEN BACK rather than doubled.
//   - Depth is a shared-memory question, not a register question, so the ring can be 4 stages deep.
//     smem is the resource this kernel has spare: at W=8 the shipped KC=2 tile is 17408 B and the
//     device has 233472 B/SM, so 4 resident blocks use 69632 of it — 30 %. A 4-stage ring of
//     one-K-block stages is 4 x 9216 = 36864 B, which is under the 49152 B static cap AND under
//     233472/4 = 58368, so blocks/SM stays register-limited at 4. That is falsification step (1) and
//     it is arithmetic, not a measurement — `ptxas -v` then has to agree that registers do not rise.
//
// WHY NOT `cp.async.bulk.tensor` / TMA, which is what the literature's producer-warp design uses.
// It assembles for sm_110a (probed: `cp.async.bulk.tensor.2d`, `cp.async.bulk`, `mbarrier.init`,
// `mbarrier.try_wait.parity` all compile at `-arch=sm_110a`), so the ISA is not the obstacle. The
// obstacle is the operand: TMA needs a host-built `CUtensorMap` whose `globalAddress` is 16-byte
// aligned, and Finding 66 counted this engine's fp8 weights — 43,470 of 44,436 tensors sit at
// `data_offset % 16 == 8` and 966 at 12, NONE at 0. That is the entire reason this kernel is
// templated on `AL16` in the first place. A descriptor would also have to be built per tensor on the
// host inside a GPU mark, which is exactly the 3.05 ms host stall trap 11 records from F72. So the
// implementable form of "cp.async staging global->shared" on THIS engine is the non-bulk
// `cp.async.ca.shared.global`, whose alignment requirement is cp-size, not 16 bytes.
//
// And non-bulk cp.async needs no producer warp: the asynchronous copy IS the decoupled producer.
// `commit_group`/`wait_group` give the >=4-stage software pipeline the sources ask for without
// splitting the block into producer and consumer roles, i.e. without spending warps on staging.
//
// STAGE GRANULARITY IS ONE K-BLOCK, not KC. Ring depth and chunk size trade against the same smem,
// and F74 already measured which one pays: KC=4 is WORSE than both 2 and 8, so bytes-per-round is
// non-monotone and past ~2 K-blocks it is not buying anything. Depth is the untested axis. At
// NS=4 x 1 K-block the bytes in flight are 3 x 9216 = 27648 B/block against the shipped KC=2's
// 4 uint4 x 256 threads = 16384 B, so this is MORE in flight on LESS register pressure.
//
// LD is back to 128+16 = 144 bytes, which is the stride F41 derived: row g starts at word
// (36g) % 32 = 4g, so the eight fragment groups are a permutation of all 32 banks. Conflict-free.
//
// COALESCING, and it is better than the register path rather than merely equal. cp-size must equal
// the natural alignment, so:
//   AL16  -> cp-size 16, 8 units/row, and 8 consecutive lanes walk one row's 128 bytes.
//   !AL16 -> cp-size 4 (NOT 8: the gate sweeps a B+4 offset and the engine's own weights include
//            `% 16 == 12`, so 4 is the only size that is always legal). 32 units/row, and now a
//            WHOLE WARP covers exactly one row: lane L takes byte 4L of row (warp + 8h). One full
//            128-byte line per instruction per warp, which the 16-byte register form never had.
//
// BIT-EXACTNESS. `cb` is still built from four mma per 128-K-block and `acc` still takes them in
// kblk order 0,1,2,..., over the same bytes. So this is BIT-EXACT to KC=1 — the pre-F74 kernel —
// exactly as KC and DB are, and `gate_tc_fp8_kc` checks it with memcmp rather than a cosine.
//
// DEFAULT OFF behind `TCB_CPA=<stages>` so the A/B is one env var (0/unset = the shipped kernel).
__device__ __forceinline__ void tcb_cpa16(void* dst, const void* src){
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void tcb_cpa4(void* dst, const void* src){
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void tcb_cpa_commit(){ asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void tcb_cpa_wait(){
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

template<int WARPS, int NS, bool AL16>
__global__ void tc_fp8_cpa_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                  const uint8_t* __restrict__ B, const float* __restrict__ bs, int M, int N, int K){
    constexpr int LD   = 128 + TCB_PAD;              // 144 B: row g -> bank 4g, conflict-free (F41)
    constexpr int ROWS = WARPS*8;                    // B rows staged per block
    constexpr int STG  = ROWS*LD;                    // bytes per ring stage
    constexpr int CPR  = AL16 ? 8 : 32;              // copy units per row
    constexpr int H    = CPR/4;                      // units per thread per stage (ROWS*CPR / 32W)
    constexpr int UF   = TCB_CPA_UF<H ? TCB_CPA_UF : H;   // issue-loop unroll (a #pragma arg is
                                                     // not macro-expanded, so it must be a name)
    __shared__ uint8_t sB[NS*STG];
    const int t=threadIdx.x, warp=t>>5, lane=t&31, gid=lane>>2, tid4=lane&3;
    const int n0blk=blockIdx.x*ROWS, m0=blockIdx.y*16;
    const int n0=n0blk+warp*8;
    const int r0=m0+gid, r1=m0+gid+8, KB=K/128;
    float acc[4]={0.f,0.f,0.f,0.f};
    // Issue K-block `kb` into ring slot `slot` and close the group. Rows past N are zero-filled
    // synchronously; they are different addresses from any in-flight copy, so the two do not race.
    // ONE 64-bit base, per-h 32-bit offsets. The naive form — `B + (size_t)gn*K + kb*128 + off`
    // inside the unrolled loop — makes H 64-bit addresses simultaneously live, which at H=8 (the
    // !AL16 path) cost 12 registers and one block/SM. That is trap 19 exactly: `n0blk*K + row*K +
    // off` never exceeds N*K = 33.5 M, so it fits in 32 bits and ptxas keeps offsets, not pointers.
    auto issue = [&](int slot, int kb){
        uint8_t* base = &sB[slot*STG];
        const uint8_t* gbase = B + (size_t)n0blk*K + (size_t)kb*128;
        #pragma unroll UF
        for(int h=0; h<H; ++h){
            const int c=t+h*(WARPS*32), row=c/CPR, off=(c%CPR)*(AL16?16:4), gn=n0blk+row;
            uint8_t* d = base + row*LD + off;
            const unsigned go = (unsigned)row*(unsigned)K + (unsigned)off;
            if(gn<N){ if constexpr(AL16) tcb_cpa16(d, gbase+go); else tcb_cpa4(d, gbase+go); }
            else    { if constexpr(AL16) *(uint4*)d=make_uint4(0,0,0,0); else *(unsigned*)d=0u; }
        }
        tcb_cpa_commit();
    };
    // Prologue fills NS-1 stages. Short K pads with EMPTY groups (legal, and they complete at once)
    // so the number of committed groups is always (NS-1)+kb and `wait_group` can stay a constant.
    #pragma unroll 1
    for(int s=0; s<NS-1; ++s){ if(s<KB) issue(s, s); else tcb_cpa_commit(); }
    for(int kb=0; kb<KB; ++kb){
        tcb_cpa_wait<NS-2>();                        // <= NS-2 pending  <=>  stage kb has landed
        __syncthreads();                             // publish it, and retire the mma of kb-1
        // Slot (kb+NS-1)%NS is slot (kb-1)%NS, whose last reader was iteration kb-1's mma — retired
        // by the barrier above. Issued BEFORE the mma so the copy has the whole mma phase to land.
        const int nxt=kb+NS-1;
        if(nxt<KB) issue(nxt%NS, nxt); else tcb_cpa_commit();
        const uint8_t* sbuf = &sB[(kb%NS)*STG];
        float cb[4]={0.f,0.f,0.f,0.f};
        #pragma unroll
        for(int kt=0; kt<4; ++kt){ const int k0=kb*128+kt*32;
            unsigned a[4], b[2];
            a[0]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4)    : 0u;
            a[1]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4)    : 0u;
            a[2]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4+16) : 0u;
            a[3]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4+16) : 0u;
            const int so=(warp*8+gid)*LD + kt*32 + tid4*4;
            b[0]=*(const unsigned*)&sbuf[so];
            b[1]=*(const unsigned*)&sbuf[so+16];
            mma_m16n8k32_e4m3(cb, a, b);
        }
        const float bsc = bs[(size_t)(n0/128)*KB + kb];
        const float as0 = (r0<M)? as[(size_t)r0*KB + kb] : 0.f;
        const float as1 = (r1<M)? as[(size_t)r1*KB + kb] : 0.f;
        acc[0]+=cb[0]*as0*bsc; acc[1]+=cb[1]*as0*bsc; acc[2]+=cb[2]*as1*bsc; acc[3]+=cb[3]*as1*bsc;
    }
    const int cn=tid4*2;
    if(r0<M && n0+cn  <N) C[(size_t)r0*N + n0+cn  ]=acc[0];
    if(r0<M && n0+cn+1<N) C[(size_t)r0*N + n0+cn+1]=acc[1];
    if(r1<M && n0+cn  <N) C[(size_t)r1*N + n0+cn  ]=acc[2];
    if(r1<M && n0+cn+1<N) C[(size_t)r1*N + n0+cn+1]=acc[3];
}

// -1 = read NO_TCSMEM from the environment on first use; the setter exists so gates and benches can
// A/B both paths in ONE process (a cached getenv cannot be re-read, which is how the stale
// "COLD+GEMV_MK" row in gemm_bench came to be measuring the default path for months).
int g_tc_smem = -1;
void tc_fp8_set_smem(int on){ g_tc_smem = on; }
// K-chunk depth for the staged tile. -1 = auto from the smem budget (see the kernel comment);
// TCB_KC=<n> or tc_fp8_set_kc(<n>) pins it, and KC=1 is exactly the pre-Finding-74 kernel, so the
// A/B is one env var. 0 also means auto.
static int g_tc_kc = -1;
void tc_fp8_set_kc(int kc){ g_tc_kc = kc; }
static int tcb_kc_env(){
    static int v = -2;
    if(v==-2){ const char* e=getenv("TCB_KC"); v = e ? atoi(e) : -1; }
    return v;
}
// KC=2 everywhere except W=1, where 8 wins. Measured COLD on the real verify shapes at M=5, the
// `m16+smem B+4` (4-byte-aligned weights = what the engine actually has) column, ms(GB/s)
// (evidence/kchunk_bench.log, clocks pinned):
//
//   shape                 KC=1        KC=2        KC=4        KC=8      W
//   wq_a  [1024,4096]  .0501( 84)  .0402(104)  .0521( 81)  .0438( 96)  2
//   wq_b  [4096,1024]  .0339(124)  .0309(136)  .0332(126)  .0328(128)  8
//   wkv   [ 512,4096]  .0434( 48)  .0329( 64)  .0361( 58)  .0266( 79)  1   <- the W=1 exception
//   wo_b  [4096,4096]  .1131(148)  .1054(159)  .1072(156)  .1074(156)  8
//   sw1/3 [2048,4096]  .0676(124)  .0588(143)  .0671(125)  .0606(139)  8
//   sw2   [4096,2048]  .0616(136)  .0571(147)  .0584(144)  .0597(140)  8
//
// KC=4 is consistently WORSE than both 2 and 8 — non-monotone, so this is a lookup, not a rule, and
// it is fitted to one bench (LEVERS.md trap 3: gemm_bench ranks, it does not predict end-to-end).
// TCB_KC=<n> overrides; TCB_KC=1 is exactly the pre-Finding-74 kernel.
//
// HARD cap at KC<=4 for W=8: 64 rows x 8 K-blocks would be a 66 KB static __shared__, which does not
// compile, so that instantiation must not exist even on a forced path.
// Double buffering of the staged chunk (Finding 78). 1 = prefetch round n+1's loads before round n's
// mma, 0 = the pre-Finding-78 kernel. -1 = read the environment on first use.
//
// DEFAULT OFF: MEASURED NULL IN SITU. `evidence/dbuf.log` vs `evidence/kchunk.log`, both
// `DSV4_DPROF=1 DSV4_KSWEEP=1`, clocks pinned — nine spec verifies pairing 1:1 at identical K and
// accept counts came to 1219.4 -> 1222.8 ms = **+0.28 %** (that instrument moved −5.9 % for F74),
// spec 21.68 -> 21.62 tok/s, ksweep K=5 121.11 -> 121.19 (+0.1 %), and the mark the lever was aimed
// at, `o:wo_b`, went the WRONG way at every K>=2 (+2.8 % at K=5) — the same sign and roughly the
// same size the bench predicted (+3.1 %). Kept behind `TCB_DB=1` because it is bit-identical
// (`gate_tc_fp8_kc`, 1134/1134) and because the occupancy result above is the reusable part.
static int g_tc_db = -1;
void tc_fp8_set_db(int on){ g_tc_db = on; }
static int tcb_db(){
    if(g_tc_db<0){ const char* e=getenv("TCB_DB"); g_tc_db = e ? (atoi(e)!=0) : 0; }
    return g_tc_db;
}
// Ring depth for the cp.async staging variant (B8-cpasync). 0/unset = the shipped KC kernel, so the
// A/B is one env var; >=2 selects that many stages. Clamped to the 49152 B static smem cap, which
// at W=8 (9216 B/stage) allows 5 and therefore 4 as the largest power of two. -1 = read the env.
static int g_tc_cpa = -1;
void tc_fp8_set_cpa(int ns){ g_tc_cpa = ns; }
static int tcb_cpa(){
    if(g_tc_cpa<0){ const char* e=getenv("TCB_CPA"); g_tc_cpa = e ? atoi(e) : 0; }
    return g_tc_cpa;
}
static int tcb_kc_max(int W){ return W>=8 ? 4 : 8; }
static int tcb_pick_kc(int W, int KB){
    const int hi = tcb_kc_max(W);
    const int forced = (g_tc_kc>0) ? g_tc_kc : tcb_kc_env();
    const int budget = forced>0 ? (40<<10) : 20480;      // forced may exceed the occupancy budget
    int want = forced>0 ? forced : (W==1 ? 8 : 2);
    if(want>hi) want=hi;
    int KC=1;
    while(KC<want && (KB%(KC*2))==0 && (size_t)W*8*((size_t)(KC*2)*128+16) <= (size_t)budget) KC*=2;
    return KC;
}
void tc_fp8_gemm(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp8, const float* b_s,
                 int M, int N, int K, cudaStream_t s){
    if(g_tc_smem<0) g_tc_smem = getenv("NO_TCSMEM")==nullptr;
    const int cpa = tcb_cpa();
    if(g_tc_smem && cpa>=2 && (K%128)==0){
        int W=8; while(W>1 && (N+8*W-1)/(8*W) < 64) W>>=1;      // same W rule as the KC kernel
        const int mt=(M+15)/16;
        const bool al16 = (((uintptr_t)B_fp8) & 15) == 0;
        dim3 g((N+8*W-1)/(8*W), mt);
        // Largest legal power-of-two depth: NS*W*8*144 must fit the 49152 B static cap.
        int NS=cpa; while(NS>2 && (size_t)NS*W*8*(128+TCB_PAD) > 49152u) NS>>=1;
        #define TCB_CPACASE(WW,SS) case SS: \
            if(al16) tc_fp8_cpa_kernel<WW,SS,true ><<<g, 32*(WW), 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K); \
            else     tc_fp8_cpa_kernel<WW,SS,false><<<g, 32*(WW), 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K); \
            break;
        #define TCB_CPASW(WW) switch(NS){ TCB_CPACASE(WW,2) TCB_CPACASE(WW,8) default: TCB_CPACASE(WW,4) }
        // W=8 caps at NS=4 (5 stages fit, 8 do not), so its NS=8 case must not be instantiated.
        switch(W){ case 8: switch(NS){ TCB_CPACASE(8,2) default: TCB_CPACASE(8,4) } break;
                   case 4: TCB_CPASW(4) break; case 2: TCB_CPASW(2) break; default: TCB_CPASW(1) break; }
        #undef TCB_CPASW
        #undef TCB_CPACASE
        return;
    }
    if(g_tc_smem && (K%128)==0){
        int W=8; while(W>1 && (N+8*W-1)/(8*W) < 64) W>>=1;      // >= 64 blocks where N allows
        const int mt=(M+15)/16;
        const int KC=tcb_pick_kc(W, K/128);
        // Row stride K is a multiple of 128 and the staging offset a multiple of 16, so the base
        // pointer alone decides whether every staged address is 16-byte aligned.
        const bool al16 = (((uintptr_t)B_fp8) & 15) == 0;
        const bool db = tcb_db()!=0;
        #define TCB_DBCASE(WW,KK,A) (db ? tc_fp8_smemB_db_kernel<WW,KK,A><<<g, 32*(WW), 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K) \
                                        : tc_fp8_smemB_kernel   <WW,KK,A><<<g, 32*(WW), 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K))
        #define TCB_KCASE(WW,KK) case KK: { dim3 g((N+8*(WW)-1)/(8*(WW)), mt); \
            if(al16) TCB_DBCASE(WW,KK,true); \
            else     TCB_DBCASE(WW,KK,false); } break;
        #define TCB_LAUNCH(WW) switch(KC){ TCB_KCASE(WW,2) TCB_KCASE(WW,4) TCB_KCASE(WW,8) \
                                           default: TCB_KCASE(WW,1) }
        #define TCB_LAUNCH8     switch(KC){ TCB_KCASE(8,2)  TCB_KCASE(8,4) \
                                           default: TCB_KCASE(8,1) }
        switch(W){ case 8: TCB_LAUNCH8 break; case 4: TCB_LAUNCH(4) break;
                   case 2: TCB_LAUNCH(2) break; default: TCB_LAUNCH(1) break; }
        #undef TCB_LAUNCH8
        #undef TCB_LAUNCH
        #undef TCB_DBCASE
        #undef TCB_KCASE
        return;
    }
    dim3 grid((N+7)/8, (M+15)/16);
    tc_fp8_kernel<<<grid, 32, 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K);
}
