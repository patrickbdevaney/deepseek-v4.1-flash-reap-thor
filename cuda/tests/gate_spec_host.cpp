// gate_spec_host.cpp — CPU gate for the bookkeeping V4.1 speculation rests on (include/v41_spec.h),
// against a host model of the state the GPU carries. No weights, no GPU.
//
// Every row a forward writes gets a unique id. A position's COMMITTED id is the one the generated
// sequence keeps (an accepted verify row, a prefill row, a plain step). The model replays what the
// kernels do with the carried state:
//   * window rings (main per layer and the drafter's, 256 slots, slot = pos % 256): a forward writes
//     all its rows, then each row reads positions pos-127..pos; the drafter at anchor p reads p-127..p
//   * the ratio-2 compressor's 2-slot state (k_pool2): row at pos writes slot pos & 1; an odd
//     position pools slots 0 and 1; pool history per row -> v41m::rollback_state
// and asserts that every read made on behalf of a committed row, and every draft, sees only
// committed ids of the right positions, over random accept patterns, stops and prefix reuses.
//
// It also checks v41spec::accept on hand cases and that the reuse rule is not vacuous (it allows
// reuse in the common case of extending the previous context).
#include "v41_spec.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

static constexpr int WINDOW = 128, RING = 256;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (fails++ < 20) { printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

struct Sim {
    std::vector<long> ring = std::vector<long>(RING, -1), dring = std::vector<long>(RING, -1);
    std::vector<long> ring_pos = std::vector<long>(RING, -1);
    long slot[2] = {-1, -1};                  // pool2 state: row ids
    std::map<int, long> committed;            // pos -> id
    std::map<long, int> pos_of;               // id -> pos
    long next_id = 0;
    int hi = -1, kv_len = 0, reuse_floor = 0;
    long pools = 0, reads = 0, restores = 0;

    // one forward over positions p0..p0+n-1; returns the row ids. `keep` = rows that will be
    // committed (only their reads are checked; rejected rows' outputs are discarded).
    std::vector<long> forward(int p0, int n, int keep) {
        std::vector<long> id(n);
        for (int i = 0; i < n; ++i) { id[i] = next_id++; pos_of[id[i]] = p0 + i; ring[(p0 + i) % RING] = id[i]; dring[(p0 + i) % RING] = id[i]; }
        for (int i = 0; i < keep; ++i) committed[p0 + i] = id[i];
        for (int i = 0; i < n; ++i) {
            const int pos = p0 + i;
            // window reads (rows in this call are already in the ring)
            if (i < keep)
                for (int q = std::max(0, pos - WINDOW + 1); q <= pos; ++q) {
                    ++reads;
                    const long got = ring[q % RING];
                    CHECK(committed.count(q) && got == committed[q], "window: row pos %d reads pos %d: id %ld (pos %d), committed %ld",
                          pos, q, got, got >= 0 ? pos_of[got] : -1, committed.count(q) ? committed[q] : -1);
                }
            // pool2 (k_pool2 order: write, then pool on odd)
            slot[pos & 1] = id[i];
            if (pos & 1) {
                ++pools;
                if (i < keep) {
                    CHECK(slot[0] >= 0 && pos_of[slot[0]] == pos - 1 && committed[pos - 1] == slot[0],
                          "pool2 at pos %d: slot0 holds id %ld (pos %d), committed %ld", pos, slot[0],
                          slot[0] >= 0 ? pos_of[slot[0]] : -1, committed[pos - 1]);
                    CHECK(slot[1] == id[i], "pool2 at pos %d: slot1 is not this row", pos);
                }
            }
        }
        hi = std::max(hi, p0 + n - 1);
        return id;
    }
    void rollback(const std::vector<long>& ids, int row, int pos) {
        if (!v41spec::pool2_needs_restore(pos)) return;
        ++restores;
        slot[0] = ids[row];                    // k_pool2_restore: slot 0 <- that row's (kv, score)
    }
    void draft_reads(int anchor) {
        for (int q = std::max(0, anchor - WINDOW + 1); q <= anchor; ++q) {
            ++reads;
            const long got = dring[q % RING];
            CHECK(committed.count(q) && got == committed[q], "draft at anchor %d reads pos %d: id %ld (pos %d)", anchor, q, got,
                  got >= 0 ? pos_of[got] : -1);
        }
    }
    void reset() {
        std::fill(ring.begin(), ring.end(), -1); std::fill(dring.begin(), dring.end(), -1);
        slot[0] = slot[1] = -1; committed.clear(); hi = -1; kv_len = 0;
    }
};

int main() {
    // accept()
    { int r[] = {7, 1, 2, 3}, t[] = {1, 2, 3, 9}; CHECK(v41spec::accept(r, t, 3) == 3, "all accepted"); }
    { int r[] = {7, 1, 2, 3}, t[] = {1, 5, 3, 9}; CHECK(v41spec::accept(r, t, 3) == 1, "stop at first mismatch"); }
    { int r[] = {7, 1, 2, 3}, t[] = {4, 2, 3, 9}; CHECK(v41spec::accept(r, t, 3) == 0, "none"); }
    { int r[] = {7, 1}, t[] = {1, 0}; CHECK(v41spec::accept(r, t, 1) == 1, "K=1"); }
    // the reuse rule is not vacuous: extending the previous context (k close to hi) is allowed
    CHECK(v41spec::window_reuse_ok(1000, 1003, WINDOW, RING), "extension must reuse");
    CHECK(!v41spec::window_reuse_ok(1000, 1129, WINDOW, RING), "a rewind past the ring must not");
    CHECK(v41spec::window_reuse_ok(1000, 1128, WINDOW, RING), "boundary: 1128 overwrote pos 872 only");

    std::mt19937_64 rng(1234);
    Sim S;
    std::vector<int> ctx;                      // committed tokens (values unimportant: positions are)
    int trials = 0, verifies = 0, reuses = 0, refused = 0;
    for (int req = 0; req < 400; ++req) {
        // new prompt: share a random prefix of the old context, then fresh tokens
        const int share = ctx.empty() ? 0 : (int)(rng() % (ctx.size() + 1));
        const int plen = std::max(1, share + (int)(rng() % 40));
        // Engine::Impl::reuse_len
        int k = std::min({share, S.kv_len, (int)ctx.size()});
        if (k >= plen) k = plen - 1;
        k = std::max(0, k & ~1);
        if (!v41spec::window_reuse_ok(k, S.hi, WINDOW, RING)) { k = 0; ++refused; }
        if (k == 0) S.reset(); else ++reuses;
        // committed positions >= k are about to be rewritten
        for (auto it = S.committed.lower_bound(k); it != S.committed.end();) it = S.committed.erase(it);
        // prefill rows k..plen-1 in chunks of up to 64
        for (int c0 = k; c0 < plen; c0 += 64) { const int M = std::min(64, plen - c0); S.forward(c0, M, M); }
        S.kv_len = plen;
        ctx.assign(plen, 0);
        // decode: tok at position plen (no KV yet)
        const int max_tokens = 1 + (int)(rng() % 200);
        int K0 = 1 + (int)(rng() % 5);
        std::vector<int> q{1};
        for (int n = 0; n < max_tokens; ++n) {
            q.erase(q.begin());
            const bool stop = rng() % 97 == 0;
            if (stop || n + 1 == max_tokens) { if (!stop) ctx.push_back(1); break; }
            const int pos = (int)ctx.size();
            ctx.push_back(1);
            if (!q.empty()) continue;
            const int K = std::min(K0, max_tokens - (n + 1) - 1);
            if (K > 0) {
                CHECK(S.kv_len == pos, "verify at pos %d with kv_len %d", pos, S.kv_len);
                S.draft_reads(pos - 1);
                const int a = (int)(rng() % (K + 1));             // accepted drafts
                const auto ids = S.forward(pos, K + 1, a + 1);
                S.rollback(ids, a, pos + a);
                S.kv_len = pos + 1 + a;
                for (int i = 0; i <= a; ++i) q.push_back(1);
                ++verifies;
            } else {
                S.forward(pos, 1, 1);
                S.kv_len = pos + 1;
                q.push_back(1);
            }
            ++trials;
        }
        S.kv_len = std::min(S.kv_len, (int)ctx.size());
        // positions past kv_len are not committed any more (the next request recomputes them)
        for (auto it = S.committed.lower_bound(S.kv_len); it != S.committed.end();) it = S.committed.erase(it);
    }
    printf("gate_spec_host: %d steps, %d verifies, %ld pools, %ld restores, %ld window reads, %d reuses, %d refused -> %s\n",
           trials, verifies, S.pools, S.restores, S.reads, reuses, refused, fails ? "FAIL" : "PASS");
    if (!verifies || !S.restores || !reuses) { printf("FAIL: the run did not exercise verify/restore/reuse\n"); return 1; }
    return fails ? 1 : 0;
}
