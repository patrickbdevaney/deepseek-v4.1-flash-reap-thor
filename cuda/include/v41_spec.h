// v41_spec.h — the host-side bookkeeping rules of V4.1 speculation, kept header-only so the CPU gate
// (tests/gate_spec_host.cpp) holds the engine to them without a GPU.
#pragma once

namespace v41spec {

// rows[0] = the committed token, rows[1..K] = drafts; tgt[i] = the target's token after row i.
// Accepted drafts = the longest prefix with rows[i+1] == tgt[i]. The target's own tgt[a] follows.
inline int accept(const int* rows, const int* tgt, int K) {
    int a = 0;
    while (a < K && rows[a + 1] == tgt[a]) ++a;
    return a;
}

// The ratio-2 compressor carries a 2-slot state (slot = pos & 1; an odd position pools slots 0, 1).
// After a verify whose last committed position is `pos`, slot 0 must hold that row again iff pos is
// even (a rejected even row overwrote it). An odd last position needs nothing: the next row (even)
// rewrites slot 0, the one after rewrites slot 1, before any pooling reads them.
inline bool pool2_needs_restore(int pos) { return (pos & 1) == 0; }

// Window rings of `ring` slots, slot = pos % ring. Restarting at k reads positions k-window+1..k-1;
// position q survives until q + ring is written. hi = the highest position written since the reset.
inline bool window_reuse_ok(int k, int hi, int window, int ring) { return k == 0 || hi < k - window + 1 + ring; }

} // namespace v41spec
