// dscratch.cu — decode scratch arena globals. See dscratch.h.
#include "dscratch.h"
#include <chrono>
// See the draft-path instrument in dscratch.h. steady_clock, not the CUDA event timeline: what is
// being measured is HOST time inside the driver call, which is exactly the cost the arena removes.
double    g_raw_ms     = 0.0;
long long g_raw_n      = 0;
double    g_rawsync_ms = 0.0;
long long g_rawsync_n  = 0;
double dsv4_now_ms(){
    return std::chrono::duration<double,std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Lever B10 / Finding 83. Read ONCE at static-init so the seam costs one predicted branch per call
// and cannot be flipped mid-run. Default false = the draft path is on the arena.
bool g_draft_raw = getenv("DSV4_DRAFT_RAW") != nullptr;
void dksync_at(cudaStream_t s, const char* file, int line){
    if(g_draft_raw){
        cudaError_t e = rsync(s);
        if(e){ fprintf(stderr,"cuda %s:%d %s\n", file, line, cudaGetErrorString(e)); exit(1); }
    }
    dsync_at(s, file, line);   // no-op sync under the arena; keeps the Finding-53 last-error read
}
bool   g_arena_on  = false;
char*  g_arena     = nullptr;
size_t g_arena_off = 0;
size_t g_arena_hwm = 0;   // high-water mark, so the zeroing diagnostic need not clear all 512 MB
size_t g_arena_cap = 0;
cudaStream_t g_side       = nullptr;
cudaEvent_t  g_side_fork  = nullptr;
cudaEvent_t  g_side_join  = nullptr;
cudaStream_t g_side2      = nullptr;
cudaEvent_t  g_side2_fork = nullptr;
cudaEvent_t  g_side2_join = nullptr;
int           g_scratch_poison_idx = -2;
unsigned char g_scratch_poison_val = 0;
int           g_scratch_alloc_seq   = 0;
unsigned long long g_scratch_first_addr = 0;
void arena_init(size_t cap){
    if(g_arena){ cudaFree(g_arena); }
    cudaMalloc((void**)&g_arena, cap); g_arena_cap = cap; g_arena_off = 0; g_arena_hwm = 0; g_arena_on = true;
    // Created here, once, because arena_init runs long before any cudaStreamBeginCapture and
    // creating a stream or an event during capture is illegal. cudaEventDisableTiming is required:
    // a timing-enabled event used purely for ordering forces a host-visible timestamp and cannot be
    // captured into a graph.
    if(!g_side){
        if(cudaStreamCreateWithFlags(&g_side, cudaStreamNonBlocking) != cudaSuccess) g_side = nullptr;
        if(cudaEventCreateWithFlags(&g_side_fork, cudaEventDisableTiming) != cudaSuccess ||
           cudaEventCreateWithFlags(&g_side_join, cudaEventDisableTiming) != cudaSuccess){
            g_side = nullptr;                       // no events -> no safe fork/join; stay serial
        }
    }
    if(!g_side2){
        if(cudaStreamCreateWithFlags(&g_side2, cudaStreamNonBlocking) != cudaSuccess) g_side2 = nullptr;
        if(cudaEventCreateWithFlags(&g_side2_fork, cudaEventDisableTiming) != cudaSuccess ||
           cudaEventCreateWithFlags(&g_side2_join, cudaEventDisableTiming) != cudaSuccess){
            g_side2 = nullptr;
        }
    }
}
// DIAGNOSTIC (DSV4_ARENA_ZERO=1). The engine is nondeterministic across sweep points on identical
// input — 36 identical points give 19 distinct token sequences and 36 distinct first-verify margin
// vectors — and it is NOT the concurrency work (a splits-off control reproduces it exactly), NOT
// atomics (the only atomicAdd is in the unused non-batched MoE path), and NOT a stale-window read
// (dspark_attn clamps nwin to t+1). The remaining candidate is scratch that some kernel reads before
// writing: `dmalloc` hands out raw bump-allocated memory and never zeroes it, so any buffer that is
// accumulated into rather than assigned inherits whatever the previous layer or point left there —
// which differs run to run and point to point.
//
// Zeroing only the high-water mark keeps this cheap enough to actually run: the arena is 512 MB but
// a decode layer touches a small fraction of it. If determinism returns under this flag, the bug is
// an uninitialised read and the next step is to find WHICH buffer by bisecting; if it does not, the
// arena is exonerated and the search moves to the persistent (cudaMalloc) buffers.
void arena_reset(){
    static const bool zero = getenv("DSV4_ARENA_ZERO") != nullptr;
    if(zero && g_arena && g_arena_hwm) cudaMemsetAsync(g_arena, 0, g_arena_hwm, 0);
    g_arena_off = 0;
}

// See the comment on dsync() in dscratch.h. Drains the thread's last-error slot so a launch failure
// is named where it happened instead of surfacing at the next unrelated CU(). Reports every time,
// because a rate limit is how the first one gets lost; aborts only under DSV4_STRICT_LAUNCH=1.
// See the comment on dprobe() in dscratch.h. Reports the LAUNCH SITE, which is the thing three
// cycles of chasing indexer.cu:91/:96 never had.
void dprobe_at(cudaStream_t s, const char* file, int line){
    static const bool probe = getenv("DSV4_SYNCPROBE") != nullptr;
    if(!probe) return;
    cudaError_t e = cudaStreamSynchronize(s);
    cudaError_t l = cudaGetLastError();
    if(e == cudaSuccess) e = l;
    if(e == cudaSuccess) return;
    fprintf(stderr, "\n[syncprobe] FAULT ATTRIBUTED TO %s:%d -> %s (%d)\n", file, line, cudaGetErrorString(e), (int)e);
    fprintf(stderr, "[syncprobe] this is the launch that faulted, not the next sync after it.\n");
    fflush(stderr);
    exit(7);
}

void dsync_at(cudaStream_t s, const char* file, int line){
    if(!g_arena_on) cudaStreamSynchronize(s);
    cudaError_t e = cudaGetLastError();
    if(e == cudaSuccess) return;
    static const bool strict = getenv("DSV4_STRICT_LAUNCH") != nullptr;
    fprintf(stderr, "[launch] %s:%d pending CUDA error: %s (%d)\n", file, line, cudaGetErrorString(e), (int)e);
    if(strict) abort();
}
