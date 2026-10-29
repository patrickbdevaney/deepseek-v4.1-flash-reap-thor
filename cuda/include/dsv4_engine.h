// dsv4_engine.h — the persistent serving engine: load once, serve many requests.
//
// WHY THIS EXISTS SEPARATELY FROM src/decode.cu. decode.cu is the measurement harness that every
// finding in LOOP_LOG rests on -- its sweeps, probes and env knobs are the instrument, and
// refactoring an instrument invalidates the readings taken with it. So the engine is a NEW
// translation unit over the SAME kernels, carrying only the shipping configuration (block 6,
// adaptK 1.5, the s3 head, VKPLUS on) and none of the sweep machinery. tests/gate_engine.cu holds
// it to decode.cu's own output: same prompt, same tokens.
//
// WHAT IT ADDS THAT THE HARNESS HAS NO NEED FOR:
//   * a KV state that survives a request, so an agentic turn can reuse its prefix (Engine::extend)
//   * sampling (temperature / top-p / seed) instead of a hardwired argmax
//   * a token callback, so the server can stream
//
// SAMPLING AND SPECULATION ARE COMPATIBLE HERE, exactly. The verify accepts draft[i] iff it equals
// the token the TARGET produces at that position. Replacing "argmax of the target logits" with "a
// sample from the target logits" keeps that property: every emitted token is still drawn from the
// target's own conditional, so the output distribution is the model's. Speculation only decides how
// many positions get evaluated per forward, never which token is emitted. What it does cost is
// acceptance -- matching a random draw is strictly harder than matching an argmax -- so tok/s falls
// as temperature rises. That is a real tradeoff, not a bug, and it is measured, not assumed.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace dsv4srv {

struct GenParams {
    float temperature = 0.0f;      // 0 => greedy (bit-identical to the measured decode path)
    float top_p       = 1.0f;
    int   max_tokens  = 512;
    uint64_t seed     = 0;
    bool  has_seed    = false;
    std::vector<int> stop_ids;     // in addition to EOS
};

// Per-request speculation telemetry. `tok_per_verify` is one number summarising a chain of
// DEPENDENT events, and accept_profile.py's whole argument is that it hides the thing training
// moves: position j is only reached if 0..j-1 were all accepted, so the quantity that matters is
// the conditional hazard h(j) = P(accept j | reached j). Two heads with identical tau can have
// opposite h and respond to more training, and to a different adaptK, in opposite directions.
//
// h(j) is not recoverable from the two MARGINAL histograms (widths, accept lengths) -- it needs
// the joint, because a verify only reaches position j if its width offered a proposal there. So
// the joint is what is recorded. It is 16x16 ints = 1 KiB per request, incremented once per
// verify with no sync and no device memory, which is why this can be always-on where
// DSV4_SPECPROF (documented at ~1 % in LEVERS.md §7) cannot.
struct SpecProfile {
    static constexpr int MAXW = 16;
    // accept_joint[w][a] = verifies that ran at realised width w and accepted a proposals.
    // Width varies per step under adaptK, which nothing previously recorded at all.
    int accept_joint[MAXW][MAXW] = {};
};

struct GenStats {
    int    prompt_tokens     = 0;
    int    cached_tokens     = 0;  // prompt tokens served from the prefix cache
    int    completion_tokens = 0;
    int    verifies          = 0;
    double prefill_ms        = 0;
    double decode_ms         = 0;
    double tok_per_s         = 0;
    double tok_per_verify    = 0;
    int    head2_fallbacks   = 0;  // V41_HEAD2 rows recomputed with the full head (too many candidates)
    SpecProfile spec;              // always-on; see SpecProfile
};

struct EngineConfig {
    std::string ckpt_dir;
    std::string engram_dir;        // V4.1: packed Engram row files (tools/pack_engram_rows.py); "" = <ckpt>/engram
    std::string keepset;           // V4.1: s04/keepset.json; its sha256 must match the thor build's
    bool  graph   = true;          // V4.1: replay the M=1 decode step as one CUDA graph
    bool  mxfp4_hw = true;         // V4.1: MXFP4 experts via the hardware e2m1x2 convert
    int   seqmax  = 8192;          // context ceiling; sizes the KV caches
    int   blk     = 5;             // DSpark draft block (LADDER 2.1: +3.91 +/- 1.65 % over 6 on 32
                                   // prompts at equal tau; F94's 6 was tuned when a verified
                                   // position was ~free. src/decode.cu carries the derivation.)
    float adaptK  = 1.5f;          // adaptive verify-width threshold (F49; 2.0 regressed, F-registry)
    int   npass   = 1;             // draft refinement passes
    int   ext_chunk = 64;          // positions per batched forward when extending a cached prefix
    bool  quiet   = false;
    // V4.1: DSpark speculation. spec_k = drafts verified per forward (0 = off, at most blk). The
    // drafter is the checkpoint's own mtp.* blocks; draft_dir = an s11 head dir that overrides them
    // ("" = the --ckpt's own, unpruned).
    int   spec_k  = 0;
    std::string draft_dir;
};

class Engine {
public:
    explicit Engine(const EngineConfig& cfg);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void load();                                     // ~10 min: weights, 43 layer structs, MTP head
    bool loaded() const;

    // Generate from a full prompt. Reuses whatever of `ids` the resident KV already covers.
    // `on_token(id)` is called for every committed token; return false from it to stop early.
    GenStats generate(const std::vector<int>& ids, const GenParams& p,
                      const std::function<bool(int)>& on_token);

    void  reset();                                   // drop the resident context entirely
    int   context_len() const;                       // committed positions currently in the KV
    const std::vector<int>& context_ids() const;

    int   eos_id() const;
    int   seqmax() const;

    // Exposed for gate_engine: force a full prefill rather than a prefix-cached extension, so the
    // two paths can be compared on the same prompt in one process.
    void set_prefix_cache(bool on);
    // Also for gate_engine. Varying the extension batch shape while holding everything else fixed
    // separates "my offset arithmetic is wrong" from "these kernels reassociate at different M".
    void set_ext_chunk(int n);
    // V4.1 speculation: drafts per verify for the next generate() calls, 0 = plain decode. At most
    // EngineConfig::spec_k (the drafter is loaded only when that is > 0). Returns the value set.
    int  set_spec_k(int k);
    std::string drafter_desc() const;                // "" when no drafter is loaded

    // Diagnostic: resolve `ids` exactly as generate() would (prefix cache included), then return
    // the target's logits for the next position, leaving the KV as it was found.
    //
    // This is the instrument the prefix cache actually needs. Comparing generated TOKENS between a
    // cached and an uncached route can only say "they diverged at step k", which is equally
    // consistent with a broken KV and with a near-tie flipped by the last bit of a float. Comparing
    // the logits says which: a wrong cache moves them grossly, reassociation moves them by ~1e-3.
    std::vector<float> debug_prefix_logits(const std::vector<int>& ids);
    // Teacher forcing: prefill ids, then step each forced token; logits before each forced token
    // (forced.size() rows of VOCAB). Compares kernels by KL instead of by where greedy text diverges.
    std::vector<float> debug_forced_logits(const std::vector<int>& ids, const std::vector<int>& forced);
    // Drafter taps (bf16 bits, [n - from, 3, dim]: the hc-mean entering each DSpark tap layer) of rows
    // [from, n) from one teacher-forced prefill of ids -- the rows the drafter reads while decoding them
    // (prefill == decode bitwise). Needs a loaded drafter (spec_k > 0 at load). Resets the context.
    std::vector<uint16_t> debug_taps(const std::vector<int>& ids, int from);

private:
    struct Impl;
    Impl* p_;
};

} // namespace dsv4srv
