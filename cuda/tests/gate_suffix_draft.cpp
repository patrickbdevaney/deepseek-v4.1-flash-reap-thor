// gate_suffix_draft — the S6 candidate drafter's matcher, gated on the host.
//
// The counterfactual probe in src/decode.cu prices LEVERS.md S6 from ONE full-model run, so its
// matcher has to be right BEFORE that run: a matcher that silently proposes nothing would report
// "S6 is worthless" and retire a lever on a bug. This costs a second instead of a checkpoint load.
//
// CPU-only, no CUDA. Run: ./build/gate_suffix_draft
#include "suffix_draft.h"
#include <cstdio>
#include <vector>

static int fails = 0, checks = 0;

static void expect(const char* name, const std::vector<int>& S, int blk, int maxng,
                   int want_len, const std::vector<int>& want_out) {
    std::vector<int> out(blk, -2);
    const int got = suffix_draft(S.data(), (int)S.size(), blk, maxng, out.data());
    bool ok = (got == want_len) && (out == want_out);
    ++checks;
    printf("  %-34s mlen=%d (want %d)  out=[", name, got, want_len);
    for (int i = 0; i < blk; ++i) printf("%s%d", i ? " " : "", out[i]);
    printf("] want=[");
    for (int i = 0; i < blk; ++i) printf("%s%d", i ? " " : "", want_out[i]);
    printf("]  %s\n", ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
}

int main() {
    printf("[gate_suffix_draft] S6 suffix/prompt-lookup matcher\n");

    // 1. Periodic sequence — the case a degenerate repeating decode presents, and the one S6 must
    //    win. Longest earlier-occurring suffix is {1,2,3,1,2,3} at i=0, continuation {1,2,3}.
    expect("periodic 123x3", {1,2,3,1,2,3,1,2,3}, 3, 32, 6, {1,2,3});

    // 2. Nothing repeats: no proposal at all, and every slot must be -1 so that it can never be
    //    counted as an accepted token.
    expect("no repeat", {5,6,7}, 3, 32, 0, {-1,-1,-1});

    // 3. The real shape: the gate prompt "0 671 6102 294 8760 344" followed by a decode that
    //    re-emits "6102 294". The suffix {6102,294} occurred at i=2, so the proposal is the prompt's
    //    own continuation {8760,344,11111}. This is classic prompt-lookup and must fire.
    expect("prompt-lookup on gate prompt",
           {0,671,6102,294,8760,344,11111,16,455,6102,294}, 3, 32, 2, {8760,344,11111});

    // 4. maxng caps the match length but must not suppress the match.
    expect("maxng=1 caps length", {1,2,3,1,2,3,1,2,3}, 3, 1, 1, {1,2,3});

    // 5. Continuation shorter than the block: pad with -1, never read past the end of S.
    expect("short continuation pads", {1,2,1}, 4, 32, 1, {2,1,-1,-1});

    // 6. Most-recent tie-break: {9} occurs at i=0 and i=3; the drafter must take i=3, so the
    //    proposal is {8}, not {7}.
    expect("most-recent occurrence wins", {9,7,4,9,8,4,9}, 1, 1, 1, {8});

    // 7. Degenerate inputs must not read out of bounds or invent a match.
    expect("n=1", {4}, 2, 32, 0, {-1,-1});
    expect("blk=0 is a no-op", {1,2,1,2}, 0, 32, 0, {});

    printf("[gate_suffix_draft] %d/%d checks -> %s\n", checks - fails, checks, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
