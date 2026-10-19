// dprof.cu — see include/dprof.h.
#include "dprof.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

bool g_dprof_on = false;

namespace {
struct Mark { int id; cudaEvent_t a, b; };
std::vector<Mark> g_pool;
int  g_used = 0;
int  g_open[DP_N];              // slot currently open per id (-1 = none)
bool g_inited = false;

const char* kName[DP_N] = {
    "hc_pre  (attn)", "rmsnorm (attn)", "ATTENTION", "hc_post (attn)",
    "hc_pre  (ffn)",  "rmsnorm (ffn)",  "MoE",       "hc_post (ffn)", "kv xin copy",
    "embed+expand",   "hc_head+norm",   "lm_head",   "argmax+D2H",
    "  attn:q_proj",  "  attn:kv_write", "  attn:sparse", "  attn:ogroup", "  attn:misc",
    "  cattn:q_proj", "  cattn:compress", "  cattn:indexer", "  cattn:sparse", "  cattn:ogroup",
    "  moe:router", "  moe:group", "  moe:w1w3", "  moe:act", "  moe:w2", "  moe:combine", "  moe:shared",
    "    q:aq(x)", "    q:wq_a", "    q:rms+aq", "    q:wq_b", "    q:tail", "    q:kv_join",
    "    o:rope", "    o:wo_a", "    o:wo_b",
    "    i:qidx", "    i:iw", "    i:score", "    i:topk",
    "      mg:count", "      mg:prefix", "      mg:scatter", "      mg:tiles",
    "draft:main_kv", "draft:block", "draft:head",
};
} // namespace

void dprof_init(int max_marks){
    if (g_inited) return;
    g_dprof_on = getenv("DSV4_DPROF") != nullptr;
    if (!g_dprof_on) { g_inited = true; return; }
    g_pool.resize(max_marks);
    for (auto& m : g_pool) { cudaEventCreate(&m.a); cudaEventCreate(&m.b); m.id = -1; }
    for (int i = 0; i < DP_N; ++i) g_open[i] = -1;
    g_used = 0; g_inited = true;
    printf("[dprof] enabled, %d marks\n", max_marks);
}

void dprof_begin(int id, cudaStream_t s){
    if (!g_dprof_on || g_used >= (int)g_pool.size()) return;
    Mark& m = g_pool[g_used];
    m.id = id; cudaEventRecord(m.a, s);
    g_open[id] = g_used; ++g_used;
}

void dprof_end(int id, cudaStream_t s){
    if (!g_dprof_on) return;
    const int slot = g_open[id];
    if (slot < 0) return;
    cudaEventRecord(g_pool[slot].b, s);
    g_open[id] = -1;
}

void dprof_reset(){ g_used = 0; for (int i = 0; i < DP_N; ++i) g_open[i] = -1; }

void dprof_report(const char* tag){
    if (!g_dprof_on || !g_used) return;
    cudaDeviceSynchronize();
    double sum[DP_N] = {0}; int cnt[DP_N] = {0};
    for (int i = 0; i < g_used; ++i){
        float ms = 0.f;
        if (cudaEventElapsedTime(&ms, g_pool[i].a, g_pool[i].b) != cudaSuccess) continue;
        sum[g_pool[i].id] += ms; ++cnt[g_pool[i].id];
    }
    // TOP-LEVEL ids only. The DP_A_*/DP_C_*/DP_M_* marks are NESTED inside DP_ATTN and DP_MOE, so
    // summing everything double-counts them and every "%" column shrinks by however much detail
    // happens to be instrumented. Percentages are of the real step.
    double tot = 0; for (int i = 0; i <= DP_ARGMAX; ++i) tot += sum[i];
    printf("\n[dprof] %s — verify step by sub-op, summed over all layers (%d marks)\n", tag, g_used);
    printf("[dprof] %-16s %10s %8s %10s\n", "phase", "ms", "%", "calls");
    for (int i = 0; i < DP_N; ++i)
        if (cnt[i]) printf("[dprof] %-16s %10.2f %7.1f%% %10d\n", kName[i], sum[i], 100.0*sum[i]/tot, cnt[i]);
    printf("[dprof] %-16s %10.2f\n", "TOTAL", tot);
    // A nested phase can never outlast its parent. If one does, the marks are being recorded
    // outside the window being reported (e.g. a prefill pass sharing the same ids) and every
    // number below is meaningless — say so rather than letting it be read as a profile.
    double c_attn = 0, c_moe = 0;
    for (int i = DP_A_QPROJ; i <= DP_C_OGROUP;  ++i) c_attn += sum[i];
    for (int i = DP_M_ROUTER; i <= DP_M_SHARED; ++i) c_moe  += sum[i];
    if (c_attn > sum[DP_ATTN] * 1.02)
        printf("[dprof] *** INVALID: attn children %.2f ms > ATTENTION %.2f ms — marks recorded outside the reported window\n", c_attn, sum[DP_ATTN]);
    if (c_moe > sum[DP_MOE] * 1.02)
        printf("[dprof] *** INVALID: moe children %.2f ms > MoE %.2f ms — marks recorded outside the reported window\n", c_moe, sum[DP_MOE]);
    dprof_reset();
}
