// suffix_draft.h — the LEVERS.md S6 candidate drafter (suffix-automaton / prompt-lookup), as a
// PURE function of the token sequence so it can be gated on the host without a checkpoint load.
//
// Rule: take the LONGEST suffix of S that occurred strictly earlier in S, break ties toward the
// MOST RECENT occurrence, and propose that occurrence's continuation verbatim. This is the
// suffix-automaton rule SuffixDecoding (arXiv 2411.04975) uses, and it strictly contains plain
// Prompt-Lookup, which fixes the match length instead of maximising it.
//
// Overlapping matches are deliberately legal (the occurrence may extend into the suffix itself):
// that is precisely how a periodic sequence gets extrapolated, and a degenerate repeating decode —
// which is what this model does on the gate prompt — is the case S6 is supposed to win.
#pragma once

// S[0..n-1] is the committed token sequence, S[n-1] being the token at the current position.
// Writes up to `blk` proposed continuation tokens into out[0..blk-1], padding with -1 (which can
// never equal a real token id, so a short proposal simply stops being accepted).
// Returns the match length, or 0 if not even the last token occurs earlier.
static inline int suffix_draft(const int* S, int n, int blk, int maxng, int* out) {
    for (int j = 0; j < blk; ++j) out[j] = -1;
    if (n < 2 || blk <= 0 || maxng < 1) return 0;
    const int Lmax = (n - 1 < maxng) ? (n - 1) : maxng;
    for (int L = Lmax; L >= 1; --L)
        for (int i = n - L - 1; i >= 0; --i) {          // i < n-L : a strictly earlier occurrence
            bool eq = true;
            for (int j = 0; j < L; ++j) if (S[i + j] != S[n - L + j]) { eq = false; break; }
            if (eq) {
                for (int j = 0; j < blk && i + L + j < n; ++j) out[j] = S[i + L + j];
                return L;
            }
        }
    return 0;
}
