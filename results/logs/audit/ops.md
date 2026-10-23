# Orchestration audit: chain.sh / chain2.sh / systemd / watchdog / memguard (2026-10-21 04:30 EDT)

Read-only audit. Nothing was edited, started, stopped or killed. The only network call was one HF `whoami` (a read).
Throwaway script: `/tmp/claude-2002/audit-ops/classify.py`. It runs `watchdog.classify()` over every STOP string.

## State at audit time (evidence)

| item | observed |
|---|---|
| chain.sh | pid 2610046, `bash ops/chain.sh`, **setsid (ppid 1), NOT under systemd**, started 23:46:23; fd 255 → inode 949402 = current `ops/chain.sh` (mtime 23:40, before start: OK) |
| s04 | `s04_select_arbitrate.py run` pid 2610085 (holds `/tmp/dsv41-chain.lock` via inherited fd 9). Unit `layer --layer 3` running, 12 arms at ~150 s per arm per layer, so about 30 min per layer × 37 layers left ≈ 19 h |
| chain2.sh | pid 2869017 under `dsv41-chain2.service`, started 03:48:12; fd 255 → inode 949440 = current file (mtime 03:47:53: OK). Idle in the `until grep … sleep 300` loop (only child: `sleep 300`) |
| memguard | `dsv41-memguard.service` pid 2841692. **Its fd 8 points to a deleted FIFO, not the lock file** (see O4). Two other guards are also running: `glm53-memguard.service` (glm-5.3-reap) and leafllm's `scripts/common/memguard.sh` (pid 67872) |
| watchdog | timer is active and enabled; last tick 04:09; `sd_state.json` tracks the s04 workers; no STOP, no NEEDS_OWNER, no OWNER_FOLLOWUP |
| dsv41-chain.service | inactive. The watchdog starts it every 15 min, and it exits 0 at `flock -n` because the setsid copy holds the lock (lock file mtime 04:09 confirms this) |
| lingering | `Linger=yes`; memguard and the timer are `enabled`; the chain units are `linked` (started by the watchdog) |
| GDM | `AutomaticLoginEnable=True`, user patrickd, so the desktop session (and with it the udisks automount of the Seagate) comes back after a reboot |
| sudo | `NOPASSWD: ALL`, so `sudo -n … drop_caches` works from the units |
| HF | `~/.cache/huggingface/token` is present; `whoami` → patrickbdevaney, **role write**. HF_HOME/HF_TOKEN are unset in every environment, so the token file is used. No `HF_XET_HIGH_PERFORMANCE` in the chain.sh, s04 or chain2 environments |
| NVMe `/` | Size 936, Used 791, **Avail 98 GiB**. Size − Used − Avail = **46 GiB of root-reserved blocks** |
| v41-work | s03 82 GiB (engram 60, taps 15, routing 4.5, bank 1.6, teacher 1.3); s04 105 GiB (replay: 12 arms × 7.6 GiB stream + 0.4-0.5 GiB state each, `_embed` 7.6); corpus 0.5; corpus.v1_defective 0.5 |
| source | 287 GiB. Shards 1-46 are real files: s04's `park.restore()` copied 3-42 back on 21 Oct 01:39-02:42. 47/48 are symlinks to `Seagate/v41_1017_engram` |
| Seagate | `/dev/sda2` fuseblk (ntfs-3g, udisks2), 1850 GiB free. It **still holds `v41_1017_parked/` (40 shards, 276 GiB)**, which duplicates shards 3-42 |
| memory | MemTotal 122.8 GiB. Every post-drop reclaim on record is 111-113 GiB. Largest RSS: apport-gtk 3.9 GiB, 2 × claude ~0.9 GiB, gnome-shell 0.4 |
| QA unit | `dsv41-qa-frontload` is **inactive (finished)**. It set `done.s09_build` (04:14), `done.s08_build` (04:16) and `done.s08_cpu_gates`; backend_ops_all was stopped after 8,088 OK and 0 FAIL |
| leafllm | `runner.stop` present since 19 Oct. `leafllm-runner.service` (Restart=always, 60 s) restarts every minute, sees runner.stop and exits; no leaf containers are running (`leaf-teacher-e4b_it` exited, restart=no) |

---

## 1. Disk timeline (`/`, GiB available as df reports it, i.e. after the 46 GiB root reserve)

Sizes are measured where they exist. The rest are computed from config (dim 5120, moe_inter 2304, 40 layers, MXFP4 = 4.25 bpw; one expert = 18.80 MB).
- **Thor build** (A192 at B's expert budget, or B128): 89.7 GiB experts + 6.3 GiB FP8 dense + 10.75 GiB globals/vision/mtp (shards 1, 2, 43, 44-46) = **≈ 107 GiB**.
- **Replay stream per arm** = T × 40,976 B:
  - s10 final (393k tokens): 15.1 GiB;
  - s04 and s10 heal (197k tokens): 7.6 GiB;
  - s07 thor (262k tokens): 10 GiB.
- **GGUF L2** ≈ 108 GiB: L0 is 152 GiB, of which 134.5 GiB is experts; L2 has 90 GiB of experts + 17.5 GiB fixed.

**Baseline**: the README chain2 table (ops/README.md:386-420) computes "free" as 936 − used. It ignores the 46 GiB root reserve, so every NVMe "free" figure in it is ~46 GiB too optimistic. That is the root cause of D1 and D2.

### As coded today (arm A wins)

| # | point | Δ | avail `/` | guard / outcome |
|---|---|---|---|---|
| 0 | now (s04 L3) | | 98 | |
| 1 | s04 later layers | state grows | ~85-90 min | no guard; fits |
| 2 | s04 end: `drop_streams` × 12 arms | +97 | **~195** | s04 keeps `replay/_embed` (7.6, dead) |
| 3 | s05 (shard → verify → upload → delete) | ±7 transient | 195 | staging_dir NVMe (195 − 16 > 40) |
| 4 | s06 (embed + heal 5.1 each, dropped incl. `_embed`) | −1 (bank/measure) | 194 | |
| 5 | s10 heal (embed + 2 arms 7.6 each; arms dropped) | −7.6 | 186 | **`_embed` kept (dead)** |
| 6 | s07 nvfp4 (per-shard stage, upload, delete) | 0 | 186 | |
| 7 | s07 thor: build + stream 10 + embed 10 + per-layer cache | −107 (peak −137) | 79 (peak ~49) | no guard; fits |
| 8 | s07 overlay | −3.5 | 76 | |
| 9 | s10 final: embed 15.1 + arms batch=1 (`max(1,(76−60)/15)`) | −15.1 (peak −30) | **~61** (peak ~46) | **`_embed` kept (dead)** |
| 10 | (arm **B** wins) s07 thor-b | −107 | **−46 → ENOSPC** | **D3 BLOCKER** |
| 11 | chain2 phase 0 stale_cache | 0 | 61 | `.cache` is 404 KiB; README expected −17 |
| 12 | trellis guard: KQ (107) + 60 = 167 | | 61 | soft-fail "disk: trellis needs 167 GiB", always (D4) |
| 13 | golden_ref/golden src + A192 (soft) | −10 | 51 | |
| 14 | move backbone 1-43 → Seagate | +279 | 330 | Seagate need 299 ✓ |
| 15 | s09 rowfile | −189 | **141** | need 200 ✓ (it had 330) |
| 16 | **s09 convert, `need / 170`** | | **141 < 170** | **STOP "disk: s09 source GGUF…" → class never → NEEDS_OWNER (D1)** |
| 16' | (if 16 were passed by freeing ~30) convert | −152 | ~19 | |
| 17 | quant L2 → Seagate | 0 | ~19 | Seagate need 320 ✓ |
| 18 | **stage L2 to NVMe, `need / 108+25`**: source GGUF is still on NVMe (dropped only at chain2.sh:231, after the gate) | | **~19 < 133** | **STOP "disk: stage L2…" → never (D2)** |

### With the proposed ordering (D1-D3 fixes; arm A wins | arm B wins)

| # | point | A | B |
|---|---|---|---|
| a | chain1 end, plus delete the dead replay `_embed`s (s04 7.6, s10/heal 7.6, s10/final 15.1) | 61 + 30 = **91** | |
| b | B only: before thor-b, delete the embeds, park thor/A192 on the Seagate (verified), then thor-b | | 61 + 30 + 107 − 107 = **91** |
| c | golden (soft) | 81 | 81 |
| d | move backbone | 360 | 360 |
| e | convert (need 170) | 208 | 208 |
| f | imatrix; quant **L2 and L3** → Seagate (CPU; no re-convert needed later) | 208 | 208 |
| g | **drop the source GGUF now** (it is regenerable) | 360 | 360 |
| h | rowfile (need 200) | 171 | 171 |
| i | stage L2 (need 133), then kld + smoke | 63 | 63 |
| j | unstage; (L3 staged only if L2 fails: need 113 ✓) | 171 | 171 |
| k | s11 (taps/heads ~30) | ~141 | ~141 |

**Seagate** (1850 free). Peak use ≈ backbone 279 + master 152 + L2 108 + L3 ~95 + a transient unsplit level ~108 (+ parked A192 107 if B wins) ≈ **740-850 GiB**, which leaves **≥ 1000 GiB**. The Seagate never limits the run.
- It could save 276 GiB and ~1-2 h of USB copy, because `v41_1017_parked/` already holds sha-verified copies of shards 3-42 (see N3).

**Disk verdict: as coded, the run cannot finish unattended.**
- chain2 STOPs at s09 convert: 141 GiB < 170 needed. That is ~4-5 days in, after the backbone move and the row files.
- If it got past that, it would STOP again when staging L2, because the 152 GiB source GGUF is dropped only after the gate.
- If arm B wins s10 (the pre-registered default whenever A's CI does not exclude 0), chain1 fills the NVMe in s07 thor-b.
- With the ordering above, every guard passes with ≥ 60 GiB to spare.

---

## 2. Findings

Severity: **BLOCKER** = will STOP or wedge the chain, or corrupt; **RISK**; **NIT**.

### Running-script edit rules (apply to every "procedure" line below)

- **chain2.sh**: it is idle in its wait loop until chain1 logs its end line (~3+ days away). Safe procedure:
  1. Write `ops/chain2.sh.new` and run `bash -n` on it.
  2. `mv -f ops/chain2.sh.new ops/chain2.sh`. The rename makes a new inode, and the running bash keeps reading its old inode, so nothing is corrupted.
  3. Then `systemctl --user restart dsv41-chain2`. That kills only bash + `sleep 300` and writes no STOP. The new code takes effect only after this restart.
  4. Before restarting, check that chain.log has no "chain reached the end" line and that `pstree -p 2869017` shows only `sleep`.
  - **Never edit in place.** That means no `cat > chain2.sh`, and no editor or tool that rewrites the same inode.
- **chain.sh**: it is running s04 and has **no STOP check between stages** (it checks only at line 11), so writing `chain.STOP` does not pause it at a boundary. Prefer fixes that live in files chain.sh invokes fresh: `ops/s07_quant.py`, `ops/s10_eval.py`, `ops/s12_publish.py`, `ops/sd/*`.
  - If chain.sh itself must change:
    1. Rename-replace it, as above.
    2. While a stage is running, `kill 2610046` (the bash only). The stage keeps running and holds the lock through its inherited fd 9.
    3. Once that stage exits, the watchdog's next `systemctl start dsv41-chain` gets the lock and re-enters the stage from its ledger.
  - For s04, that re-entry is cheap: select, every layer and the heads are ledgered; only decide re-runs.
- **Python stage files**: s04 units re-import these at **every layer**, so do not touch them until `done s04` appears in chain.log:
  - `ops/s04_select_arbitrate.py`, `ops/s03_saliency.py`, `ops/corpus_spec.py`;
  - **any `ops/lib/*.py`**;
  - `inference/model.py`.

  These are safe to rename-replace now, because nothing running imports them: `ops/s05_surgery.py`, `ops/s06_heal.py`, `ops/s07_quant.py`, `ops/s10_eval.py`, `ops/s12_publish.py`, `ops/s11_spec.py`, `ops/s09_*.py`, `llamacpp/*`, `cuda/*`.
  - **Exception**: s05, s06 and s07 call lib modules, so any lib/ change must wait for the end of s04.
- **`ops/sd/watchdog.py`, `ops/sd/should_run.sh`, `*/CHAIN_STEPS.sh`**: read fresh on every invocation. Rename-replace at any time. For the watchdog, do it while `systemctl --user is-active dsv41-watchdog.service` is `inactive`.
- **`ops/memguard.sh`**: rename-replace, then `systemctl --user restart dsv41-memguard`. The gap is about 1 s; do it while MemAvailable > 20 GiB, i.e. not during a unit's layer load.

---

### D1 BLOCKER: chain2 STOPs at the s09 convert df guard (141 GiB available < 170 needed)

- **Where**: chain2.sh:166 (rowfile placed before convert), chain2.sh:199 (`need / 170`).
- **Scenario**: the disk table above, rows 11-16.
  - At chain2 start there are about 61 GiB available.
  - The backbone move gives +279 and the rowfile takes −189.
  - So 141 GiB are available when convert's guard runs. The STOP "disk: s09 source GGUF (~152 GiB) needs 170 GiB…" classifies as **never** (selftest-confirmed), so it goes to NEEDS_OWNER.
  - Even without golden (CUDA failed), it is 151 < 170.
- **Root cause**: README's plan omits the 46 GiB root reserve, the s04 and s10 `_embed` streams that are never dropped (30 GiB), and the stale_cache it assumed was 17 GiB (it is 0).
- **Fix**: in chain2.sh:
  1. Add a phase-0 step `drop_dead_replay` after the `done.s10final` check: `rm -f $WORK/{s04/replay,s10/heal/replay,s10/final/replay}/_embed/{h,pre_mix}.npy` (and `rm -rf $WORK/corpus.v1_defective`).
  2. Move `step_done s09_rowfile_ …` (line 166) to **after** the GGUF quant(s) and the source-GGUF drop.
  3. Order: backbone → convert → imatrix → quant L2 → quant L3 → drop src GGUF → rowfile → `ladder` staging/kld/smoke. See D2 for the ladder split.
- **Procedure**: the chain2 rename-replace + restart procedure above, any time in the next ~3 days.

### D2 BLOCKER: the source GGUF is still on NVMe when L2 is staged

- **Where**: chain2.sh:216 (`need / sz+25`), chain2.sh:231-233 (`drop_src_gguf` only after `ladder L2` and the gate).
- **Scenario**: after convert there are ~19 GiB available (or less) and staging needs ~133. That gives a STOP "disk: stage L2 to NVMe needs …", class never.
  - The L3 fallback path has the same problem.
  - The README table row "phase C: after dropping the source GGUF, L0 staged" describes an order that the code does not implement.
- **Fix**: split `ladder` into `quantize LV` and `evaluate LV`.
  1. Quantize L2 **and** L3 to the Seagate. Both are CPU-only and read the source GGUF; Seagate space is ample.
  2. Then `rm -f $G/src/*-src.gguf; mark drop_src_gguf`.
  3. Then the rowfile.
  4. Then `evaluate L2`, and `evaluate L3` only if L2 fails the gate.

  Lazy L3 would instead require resetting `done.s09_convert`, `done.c2_s09_convert_` and `done.s09_imatrix` and re-converting ~152 GiB. Eager L3 costs only CPU time.
- **Procedure**: same as D1 (one chain2 edit).

### D3 BLOCKER (conditional: arm B wins s10): s07 thor-b fills the NVMe

- **Where**: chain.sh:131-134; s07_quant.py `cmd_thor_b` (about line 655) has no disk check.
- **Scenario**: after s10final there are ~61 GiB available, and thor-b writes ~107 GiB → ENOSPC at roughly the middle of layer 22.
  - The STOP "s07 thor-b failed" is class never.
  - The root filesystem is left at 0 user-available bytes. `watchdog.py` writes `sd_state.json` and `sd_health.log` non-atomically, so its ticks can start failing (see O9). Other users' processes on `/` fail too.
  - The pre-registered rule makes B the winner unless A's paired Δkl32 CI excludes 0 in A's favour, so this is a likely branch, not an edge case.
- **Fix** (in `s07_quant.py` `cmd_thor_b`, before the layer loop):
  1. Compute `need = 128 × 40 × 18.8 MB + dense + globals + 25 GiB` (≈ 132 GiB).
  2. If there is not enough free space: unlink the dead replay embeds (`s04/replay/_embed`, `s10/heal/replay/_embed`, `s10/final/replay/_embed`).
  3. If it is still short: park `thor/A<n>` on `Seagate/v41_staging/thor/A<n>`. Copy it, verify with `thor_format.Build(...).verify()` (as chain2's `park_build`), then remove it from NVMe and replace it with a symlink.
  4. chain2 already handles a parked A: `park_build` returns on `-L`, and `unpark_build` copies it back if it is served.
  5. Optionally add the `_embed` unlink at the end of `s10_eval.run_panel` too, which fixes D1's 30 GiB at the source.
- **Procedure**: `s07_quant.py` and `s10_eval.py` are not running and are not imported by s04, so rename-replace them now. Do not touch `ops/lib/` until s04 is done. No chain.sh edit is needed.

### D4 RISK: the trellis sub-arm is structurally infeasible on this NVMe, and a disk skip writes OWNER_FOLLOWUP, which keeps leafllm paused

- **Where**: chain2.sh:113-126.
- **Problem**:
  - The guard needs 107 + 60 = 167 GiB, and at most ~91 GiB is available even after D1's cleanup.
  - The build and the panel both replay the source, so they must run before the backbone move.
  - The panel needs thorA and thorA_kt (2 × 107) on NVMe at the same time.
  - So `soft trellis` always ends in `soft_fail.trellis` and an OWNER_FOLLOWUP line. The watchdog (watchdog.py:243) then never removes `runner.stop`.
- **Fix** (owner decision):
  - Either record "trellis skipped: disk (structural)" in chain.log **without** FOLLOW, or drop it from the unattended plan.
  - Or accept that leafllm resumes only by hand. That is already the expected outcome, because s11 e2e "BLOCKED" also writes FOLLOW.
- **Procedure**: the chain2 rename-replace + restart procedure.

---

### W1 RISK: STOP classification sends transient failures to NEEDS_OWNER

Run `/tmp/claude-2002/audit-ops/classify.py` to reproduce.

- **(a) "Seagate" matches the NEVER regex.** watchdog.py:51 has `gate`, so "chain2: the Seagate is not mounted at …" is classified **never**. README.md:448 promises 48 retries for it. Any reason that contains "Seagate" or `cpu_gates` (for example "memory did not return to baseline after soft s08_cpu_gates") is likewise never.
- **(b) chain2 wrappers drop the real reason.** `gpu`, `cpu` and `soft` STOP with just the command ("chain2: s09 convert", "s09 kld L2", "stage L2"…). A CHAIN_STEPS `die "another model-loading process is up"`, `die "memory did not return to baseline…"` or a curl or network error therefore becomes **never**. CHAIN_STEPS does log the real reason to chain.log as `s0N FAIL: …`.
- **(c) Python-side reclaim failures in chain1 become never.** In chain1 python stages, `env.reclaim(expect_gib=100)` raises `RuntimeError("reclaim left …")`. The STOP is "s04/s06/s10 … failed", class **never** (memguard class only within 15 min of a memguard kill).
- **(d) Network failures in "s07 nvfp4 failed", "s12 release re-render" and "s12 final re-render" are never.** All three upload to HF.
- **(e) A watchdog stall-kill is not recognised as transient.** After it, the following STOP is "sNN failed" and gets class never, so a hung s07 nvfp4 upload is never retried.

**Fix** (watchdog.py only; safe any time):
1. Test CONFLICT before NEVER for reasons that **start with** the fixed quiet/reclaim/mount phrases.
2. Change the NEVER regex to `\bgate\b|gate (failed|FAIL)`, or exclude `Seagate` explicitly.
3. Enrich `reason` with the last `FAIL:` line of chain.log written before `t_stop`, and the last 20 lines of the newest `ops/logs/s*.log`. Classify `reclaim left|memory did not return|another model-loading|HTTPError|ConnectionError|Timeout|5\d\d Server Error|xet` as conflict or transient.
4. Write a `stall_kill` timestamp to sd_state and treat the next STOP within 15 min like the memguard class.

### W2 RISK: retry budgets are cumulative over the whole 5-day run, and digit-normalisation merges stages

- **Where**: watchdog.py:213-214, 229.
- **Problem**: the retry key replaces every digit with N, so these collapse into single keys:
  - "memory did not return to baseline after s04" and "… after s06" → one key, "after sN";
  - "s05 surgery/upload failed" → one key;
  - "backbone copy model-00003…" through "model-00043…" → one key.

  The counter never resets after progress. So s05's ~152 GiB upload and s07's NVFP4 upload each get 3 retries in total, for the whole run.
- **Fix**: store `ndone = len(glob("ops/logs/done.*"))` in sd_state, and clear `st["retries"]` whenever it increases. Normalise only long digit runs (`\d{3,}`) so stage names survive.

### W3 RISK: should_run.sh skips silently when the Seagate is unmounted (no STOP, no NEEDS_OWNER)

- **Where**: should_run.sh:14-17; it applies to **chain 1** too.
- **Problem**:
  - After any retried STOP (or a reboot), if udisks does not remount the Seagate, both chains stay down indefinitely. The only trace is `seagate=UNMOUNTED` in sd_health.log.
  - chain1 (s04-s12) does not even need the Seagate, because `staging_dir` falls back gracefully.
  - From a user unit, `udisksctl mount` is subject to polkit `filesystem-mount` = "active: yes, any: auth_admin". Mounting therefore depends on GDM autologin (which is on), and fails headless.
- **Fix**:
  - Require the mount only for `chain2`.
  - In watchdog.py, write NEEDS_OWNER "Seagate unmounted for > 1 h while a chain is due".
- **Procedure**: rename-replace both files.

### W4 RISK: chain2's wait loop can wedge forever on a log line that is never written

- **Where**: chain2.sh:62-66; should_run.sh:8.
- **Problem**:
  - chain.sh marks `done.s12` (line 140) **before** logging the end line (line 142). If the process dies between the two (reboot or kill), should_run (line 8) never restarts chain1, so the line is never written.
  - chain2 then sleeps forever. It consists of bash + sleep, which the watchdog's stall detector deliberately ignores, and there is no STOP.
  - Separately, `while pgrep -f 'bash ops/chain.sh'` matches any process whose command line contains that text: a Claude Code `bash -c` wrapper, `grep "bash ops/chain.sh"`, or `less`. That only delays chain2.
- **Fix**:
  - In chain2: wait for `grep end-line || [ -f $L/done.s12 ]`.
  - In should_run `chain`: skip only if `done.s12` exists **and** the end line is present.
  - Use `pgrep -f '^(/bin/)?bash ops/chain\.sh$'`.

### W5 RISK: false positives in quiet(), and gaps in its coverage

- **Where**: chain2.sh:30-33; cuda/CHAIN_STEPS.sh:49-52; llamacpp/CHAIN_STEPS.sh:29-31; chain.sh:18-19.
- **False positives**: `pgrep -f` matches any command line. Evidence: during this audit, a Claude Code `bash -c "… pgrep -fa 'test-backend-ops|qa_frontload' …"` matched its own pattern. Any `less ops/s10_eval.py`, `vim ops/s11_spec.py`, `tail -f …s11_spec…` or Claude command that mentions `s10_eval`, `llama-server`, `v41_decode` or `ref_dump` while chain2 crosses a quiet() writes a conflict STOP. That STOP costs ≥ 20-30 min per retry and counts toward 48.
- **Gaps**:
  - **chain.sh quiet()** (before s04, s06, s10heal, s07thor, s10final) checks only `leaf-teacher` docker and the leafllm runner.
  - The python guard `env.assert_no_gpu_server` checks docker `leaf-teacher|vllm*` and `llama-server|vllm serve`. **s07_quant.py never calls it.**
  - Nothing in chain 1 detects: `v41_decode`, `v41_server`, `gate_v41_*`, `gemv_format_bench`, `test-backend-ops` (the QA unit ran it beside s04), `llama-perplexity`, `llama-cli`, other projects' GPU jobs (glm-5.3 `run_stage.py`, a dspark server), or non-docker vLLM.
  - chain2's pattern also misses `test-backend-ops`, `gemv_format_bench`, `s04_select_arbitrate|s06_heal|s07_quant`, `vllm serve` and llama tools.
- **Fix**:
  - Match on the process **executable or argv[0..1]** instead of the whole command line. For example, a small helper `ops/sd/gpu_busy.py` that scans `/proc/*/cmdline`, skips comm in {bash, sh, claude, less, vim, tail, grep}, and matches a single shared list.
  - Call the helper from all four quiet()s; it is a new file, so it is safe.
  - Add `env.assert_no_gpu_server` to `s07_quant.py` thor/thor-trellis. That edit is safe now.
  - chain.sh's own quiet() cannot change without a chain.sh restart.

### W6 RISK: the 100 GiB reclaim floor leaves 11-13 GiB of headroom

- **Where**: chain.sh:15-17; chain2.sh:27-29; CHAIN_STEPS reclaim; `env.reclaim` (default `--min-avail-gib 100`).
- **Evidence**: every reclaim on record is 111-113.1 GiB (s01-s03, 46 samples) with MemTotal 122.8. About 10 GiB is not reclaimable, of which:
  - apport-gtk (a crash dialog left open since 16 Oct) holds 3.9 GiB;
  - two Claude sessions hold ~0.9 GiB;
  - gnome-shell, gnome-software, update-manager, qdirstat and chromium hold ~1.2 GiB.

  Another open Claude or Chromium session, or a GNOME update pop-up, can push it under 100.
- **Consequences**:
  - A bash reclaim failure is class conflict: 48 cumulative retries, then the owner.
  - A python reclaim failure inside a stage is class **never** (W1c): an immediate owner stop.
  - Python's `env.reclaim` also lacks bash's `sleep 5` after `drop_caches`.
- **Fix**:
  - Owner: close apport-gtk and update-manager (about +4 GiB) and avoid extra sessions.
  - Code: pass `--min-avail-gib 95` from the not-yet-run stages' argparse defaults (s05, s06, s07, s10, s11 py files; safe now), and add `time.sleep(5)` in `env.reclaim` after s04.
  - Bash floors cannot change without editing chain.sh. chain2 and CHAIN_STEPS can change via their procedures.

### W7 RISK: hangs that the stall detector cannot see

- **Where**: cuda/CHAIN_STEPS.sh:163-168 (`curl` with no `--max-time`); llamacpp/CHAIN_STEPS.sh:131-136; watchdog.py:130-141.
- **Problem**:
  - Progress is the **sum** over all workers, so one spinning process masks a hung one. If `v41_server` busy-polls (as CUDA servers do) while a chat request never completes, `curl` blocks forever and the watchdog keeps seeing CPU progress.
  - The s08 server step is soft, but it blocks everything after it, including s11 and s12 final.
- **Fix**: `curl --max-time 600` on every request in both CHAIN_STEPS (safe now). Optionally, have the watchdog judge stalls **per pid** for curl and hf.

### W8 RISK: the s09 gate result is taken from a pipe without pipefail

- **Where**: chain2.sh:225 and 229: `PASS=$($PY ops/s09_gate.py … | tail -1) || stop`.
- **Problem**: chain2 has no `pipefail`, so a crash in `s09_gate.py` gives PASS="" and `|| stop` never fires. The crash is treated as "L2 failed its gate": L3 gets built, then "no GGUF level passed", nothing is published, and the chain continues.
- **Fix**: `PASS=$($PY ops/s09_gate.py --levels L2 2>>log) || stop "s09 gate crashed"; PASS=$(tail -1 <<<"$PASS")`.
- **Procedure**: chain2.

### W9 RISK: backend_ops can never pass after one FAIL

- **Where**: llamacpp/CHAIN_STEPS.sh:65-70.
- **Problem**: `s09_backend_ops_types.log` is appended (`>>`) and the verdict greps `\bFAIL\b` over **all** `s09_backend_ops_*.log`. One genuine or flaky FAIL makes every retry fail, so the step is a hard STOP until someone deletes the log.
- **Fix**: truncate both logs at the start of the step.
- **Procedure**: rename-replace now.

### W10 RISK: QA done markers can leave chain2 running stale binaries

- **Where**: `done.s08_build` (04:16), `done.s09_build` (04:14), `done.s08_cpu_gates`; CHAIN_STEPS `build` exits 0 when its marker exists.
- **Problem**: if `cuda/` or the llama.cpp fork changes during the next ~3 days, chain2 runs the 21-Oct binaries and cpu_gates. Only `rebuild()` in the spec steps recompiles `v41_decode`.
- **Interference with the chain**: none today. The QA unit is inactive, wrote only those markers plus `cuda/build/units/*` (deterministic refs that the chain also uses), and ran no GPU marker steps.
  - Do **not** re-launch it once chain2 is past phase A. `J=4 llamacpp/build.sh` would overwrite `libllama.so` and the tools under a running `llama-quantize`, `kld` or `llama-server`.
  - It also runs GPU binaries outside every memguard licence (O5) and beside whatever is loaded; its floor is 40 GiB.
- **Fix**: in chain2 phase A, `rm -f ops/logs/done.s08_build ops/logs/done.s09_build ops/logs/done.s08_cpu_gates` before the build steps (incremental builds take seconds to minutes).
- **Procedure**: chain2.

### O1 RISK: memguard's licence misses this project's heaviest GPU binaries

- **Where**: memguard.sh:146-148.
- **Problem**: the licence is `ops/sNN_*.py`, `llama-(completion|perplexity|quantize|imatrix|server)` and `hf upload`. It does **not** cover:
  - `v41_decode` / `v41_server`, which load the full ~107 GiB build;
  - `gate_v41_golden`, `gate_v41_units`, `gemv_format_bench`;
  - `v41-teacher-kld` (the GGUF kld runs the whole L2);
  - `test-backend-ops`;
  - `cuda/tests/golden/ref_dump.py`, `cuda/tests/v41_unit_ref.py`, `llamacpp/convert_v41_reap.py` (none of them under `ops/`).

  In a runaway during `s11 e2e` → `spec_bench` → `v41_decode`, the guard kills the newest **licensed** process. That is the `s11_spec.py` parent, which frees nothing. The 2026-09-26 incident class applies: the box hard-reboots.
- **Second problem**: the pattern also matches Claude Code `bash -c` wrappers that mention `ops/s10_eval.py`. Those are always the newest, so they are preferentially killed.
- **Fix**:
  - Add `v41_decode|v41_server|gate_v41_|gemv_format_bench|v41-teacher-kld|test-backend-ops|ref_dump\.py|v41_unit_ref\.py|convert_v41_reap\.py|llama-(quantize|imatrix|cli)`.
  - Skip candidates whose `/proc/$q/comm` is bash, sh, claude, node or less.
- **Procedure**: the memguard procedure above.

### O2 RISK: watchdog state and health writes are not crash-safe

- **Where**: watchdog.py:182, 249-253.
- **Problem**: `json.loads(STATE.read_text())` raises on an empty or truncated `sd_state.json`; the file is written by `write_text`, not tmp + rename. A truncated file (power loss, or ENOSPC as in D3) stops every later tick before the restart step, so chains are never restarted and no health line is written.
- **Fix**: `try/except` on the read (reset to `{}`), atomic tmp + `os.replace` on the write, and `try/except OSError` around `health()`.
- **Procedure**: rename-replace now.

### O3 RISK: no alerting

- **Problem**: NEEDS_OWNER and OWNER_FOLLOWUP are plain files. For an unattended run of 3-5 days, the first never-class STOP (D1 is guaranteed as coded) idles the box until someone looks.
- **Fix**: owner's choice, for example a desktop notify-send via the autologin session, or a mail hook. At minimum, re-check `ops/logs/NEEDS_OWNER` daily.

### O4 NIT: memguard's singleton lock is released right after start

- **Where**: memguard.sh:40 `exec 8>"$LOCK"` and memguard.sh:62 `exec 8<>"$_FIFO"`.
- **Problem**: line 62 reuses fd 8, which closes the lock description. Evidence: `/proc/2841692/fd/8 → /tmp/tmp.F6iHqE4RET (deleted)`, and no fd on `/tmp/dsv41-memguard.lock`. A second instance (from chain.sh:13 when `pgrep` races the systemd `RestartSec=5`, or a manual start) would run beside it.
- **Fix**: use fd 7 for the FIFO.
- **Procedure**: as O1.

### O5 NIT: chain.sh:13 spawns memguard with the chain lock still open

- **Problem**: the spawned memguard inherits fd 9 (`/tmp/dsv41-chain.lock`). If chain.sh ever launches it (when the systemd memguard is momentarily down), that memguard holds the chain lock **forever**. Every later chain.sh exits 0 at `flock -n` and chain 1 silently never runs again. This is the fd-inheritance trap noted in memory.
- **Status**: not happening now (pgrep found the systemd guard; memguard's fds show no lock).
- **Fix**: `… bash ops/memguard.sh 9>&- …`, or delete the line (systemd owns memguard). Requires the chain.sh procedure, so fold it into any other chain.sh change.
- The same inheritance applies to every child of both chains. Under systemd (`KillMode=control-group`), leftovers die with the unit. chain.sh itself is setsid today, so if its bash is killed, its s04 child keeps the lock until it exits. That is expected; see the procedure above.

### O6 NIT: leafllm never auto-resumes in practice

- **Where**: watchdog.py:242-246.
- **Problem**: the watchdog removes `runner.stop` only when there is no OWNER_FOLLOWUP. FOLLOW gets written by:
  - D4 (trellis disk skip): always;
  - s11 "BLOCKED" (chain2.sh:294-296) whenever cuda/ has no draft path;
  - any CUDA soft fail.

  This is consistent with the README, but in practice leafllm is resumed only by hand.
- **Side effects**:
  - `leafllm-runner.service` (Restart=always, 60 s) appends "runner stopped" to `~/leafllm/logs/runner.log` every minute, about 1,440 lines a day.
  - Removing `runner.stop` by hand while a chain step runs is a co-run: quiet() checks only at step start. Do it only when no chain is active.
- No leafllm files were touched.

### O7 NIT: two foreign memguards are running beside dsv41's

- **glm53-memguard** (glm-5.3-reap; licence includes `llama-(completion|perplexity|quantize|imatrix)` and `hf upload`, floor 250 MB): it can also kill chain2's `llama-quantize`. That is harmless, because the class is memguard and it is retried.
- **leafllm memguard** (floor 6,000 MB, slope 400 MB/s below 12,000 MB): it will trip during our legitimate loads. Its victims are the pgids in `~/leafllm/run/guard_victims` (96 lines, all dead). There is a small chance of pgid reuse: pid_max is 4,194,304, and our setsid stages get pgid = pid. Low.
- No action needed, but if `glm53-memguard` is no longer needed, the owner could stop it.

### N1 NIT: master download uses `HF_XET_HIGH_PERFORMANCE=1` onto the USB disk

- **Where**: chain2.sh:175.
- **Problem**: the target is NTFS over USB at ~130 MB/s, so HP mode buys nothing and only adds memory pressure on unified memory.
- **Fix**: drop `HF_XET_HIGH_PERFORMANCE=1`.
- **Upload environments are correct**:
  - s05, s07 and s12 pop HP and set `FIXED_UPLOAD_CONCURRENCY=8` in-process;
  - `ops/sd/env` has no HP;
  - the setsid chain.sh environment has no HP.

### N2 NIT: stale_cache step and README figures

- `$SRC/.cache` is already 404 KiB, so chain2 phase 0 frees nothing.
- README.md:386-420 should be corrected for the 46 GiB root reserve and the kept `_embed` files.

### N3 NIT: the backbone move re-copies 276 GiB that is already on the Seagate

- **Where**: chain2.sh:149-163.
- **Problem**: shards 3-42 exist, sha-verified, in `Seagate/v41_1017_parked/` (park ledger). chain2 copies them again to `v41_staging/backbone/`, which costs about 1-2 h of USB writes plus re-hashing, and 276 GiB of Seagate.
- **Fix**: if `v41_1017_parked/$s` exists, `sha_ok` it against the tree and symlink to it, instead of rsyncing.
- **Procedure**: chain2.

### N4 NIT: robustness after a reboot

- Units, lingering, autologin and the timer all come back.
- **NTFS**: after a hard reboot (the unified-memory failure mode) the NTFS volume may be marked dirty. ntfs-3g usually still mounts it, but if it mounts read-only, every copy gets a transient class STOP (3 retries).
- **Mount path**: if GNOME remounts it under `…/Seagate Portable Drive1` (stale mountpoint dir), the `$SRC` symlinks dangle and should_run skips silently (W3).

---

## 3. Answers to the specific checks

1. **Disk**: see section 1. As coded, chain2 is guaranteed to STOP at convert and then at L2 staging (D1, D2). chain1 fills the NVMe if arm B wins (D3). The ordering in section 1 fits everything with ≥ 60 GiB to spare. The Seagate is fine (≥ 1000 GiB free at peak).
2. **Stop and soft-fail paths**: the watchdog retries only the classes it recognises, but several transient failures are misclassified as never (W1), and the budgets are cumulative (W2). Retries are bounded (48 / 3 / 2), so there is no infinite loop.
   - Wedges with no STOP: W3 (unmounted Seagate), W4 (`done.s12` without the end line), W7 (hung curl beside a busy server), O5 (memguard holding the chain lock).
   - The flock-inheritance trap is not live now; under systemd, `KillMode=control-group` clears leftovers.
3. **reclaim ≥ 100 GiB**: it holds today (111-113) with an 11-13 GiB margin, of which apport-gtk alone is 3.9. A python-side shortfall stops for the owner immediately (W1c, W6).
4. **quiet() patterns**: chain1's quiet() detects only leafllm; the python guard misses s07 and every CUDA binary. chain2's is broad but substring-based, so it gives false positives on editors and Claude command lines (W5). memguard's licence misses the CUDA and kld binaries (O1).
5. **leafllm**: the pause is correct (`runner.stop` present; the runner exits every minute). Auto-resume will in practice never fire because of OWNER_FOLLOWUP (O6, D4).
6. **Reboot**: lingering is on, memguard and the timer are enabled, the chains are started by the watchdog, GDM autologin brings back the Seagate automount, and `/tmp` locks are irrelevant (flock does not persist). Gaps: W3 and N4.
7. **HF credentials**: the token file is present and has the write role; no `HF_TOKEN` or `HF_HOME` override. Uploads never use HP; chain2's download uses HP (N1).
8. **QA unit**: inactive. It does not interfere with current chain steps, but its build markers can freeze stale binaries for chain2 (W10). It must not be re-run once chain2 is past phase A.
