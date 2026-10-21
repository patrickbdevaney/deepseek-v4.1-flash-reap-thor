#!/usr/bin/env python3
"""Tests for ops/sd/watchdog.py: STOP classification on the real reason strings (W1), retry keys and the
progress reset (W2), the Seagate NEEDS_OWNER (W3), crash-safe state (O2), and full ticks of main() against a temp
log dir with systemctl, sudo and the process scan stubbed out. Starts, stops and kills nothing.

    python3 -I ops/tests/test_watchdog.py
"""
from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import tempfile
import time
import types
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WD_PATH = Path(os.environ.get("WATCHDOG", HERE.parent / "sd" / "watchdog.py"))


def load():
    m = types.ModuleType("wd")
    m.__file__ = str(WD_PATH)
    importlib.machinery.SourceFileLoader("wd", str(WD_PATH)).exec_module(m)
    return m


wd = load()

# every STOP string chain.sh / chain2.sh can write (audit's classify.py), with the class wanted when nothing else is
# known (no FAIL line, no log tail, no recent kill)
REAL = {
    "s04 failed (resumable; see ops/logs/s04.log)": "never",
    "s05 surgery/upload failed (resumable)": "transient",
    "s06 heal failed (resumable)": "never",
    "s10 heal panel failed": "never",
    "s07 nvfp4 failed (resumable)": "never",
    "s07 plan failed": "never",
    "s07 thor build failed (resumable)": "never",
    "s07 overlay failed": "never",
    "s10 final panel failed": "never",
    "s07 thor-b failed": "never",
    "s12 publish failed": "transient",
    "memory did not return to baseline after s04 (95 GiB < 100)": "conflict",
    "memory did not return to baseline after soft s08_cpu_gates (95 GiB < 100)": "conflict",
    "memory did not return to baseline after soft golden_ref_src (95 GiB < 100)": "conflict",
    "memory did not return to baseline after s09 backend_ops (95 GiB < 100)": "conflict",
    "chain2: memory did not return to baseline after s09 gate (95 GiB < 100)": "conflict",
    "a leafllm vLLM server is up": "conflict",
    "leafllm runner active": "conflict",
    "chain2: s09 build": "never",
    "chain2: s09 convert": "never",
    "chain2: s09 kld L2": "never",
    "chain2: stage L2": "transient",
    "chain2: unstage L2 results": "transient",
    "chain2: copy thor/B128": "transient",
    "chain2: copy back thor/A192": "transient",
    "chain2: thor/B128 Seagate copy does not verify": "never",
    "chain2: thor/B128 NVMe copy does not verify": "never",
    "chain2: backbone copy model-00003-of-00048.safetensors": "transient",
    "chain2: backbone: Seagate copy of model-00003-of-00048.safetensors does not match the pinned sha256": "never",
    "chain2: the Seagate is not mounted at /media/patrickd/Seagate Portable Drive": "conflict",
    "chain2: s10final has not passed; chain2 needs it": "never",
    "chain2: no s04 keep-set": "never",
    "chain2: disk: s09 source GGUF (~152 GiB) needs 170 GiB free on /, only 140 GiB": "never",
    "chain2: disk: stage L2 to NVMe needs 125 GiB free on /, only 28 GiB": "never",
    "chain2: master download user/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4": "transient",
    "chain2: master download does not verify against the s05 manifest": "never",
    "chain2: s09 gate": "never",
    "chain2: s09 manifest": "never",
    "chain2: s12 gguf publish": "transient",
    "chain2: s12 thor publish": "transient",
    "chain2: s12 release re-render": "never",
    "chain2: a vLLM server is up": "conflict",
    "chain2: another model-loading process is up": "conflict",
    "chain2: no s05 manifest": "never",
    "chain2: /home/patrickd/models/v41-work/s09/L2 is a real non-empty directory (expected the Seagate symlink)": "never",
    "s01 2k gate: rc=1 verdict=FAIL (see ops/logs/s01_*.log)": "never",
    "s00b sha mismatch on Seagate copy of model-00047-of-00048.safetensors": "never",
    "s11 end-to-end gate BLOCKED: cuda/ has no DSpark draft+verify path": "never",
    # owner pauses written by hand (chain.log 2026-10-20) must never be auto-retried
    "paused before s04 select: drift_guard rising criterion domain-confounded": "never",
    "paused in s04 select: protect ran on --rows all but check runs on --rows calib (row mismatch)": "never",
}

FAILS = 0


def check(cond: bool, what: str) -> None:
    global FAILS
    if not cond:
        FAILS += 1
        print(f"FAIL {what}")


def test_real_reasons() -> None:
    for r, want in REAL.items():
        got = wd.classify(r, False)
        check(got == want, f"classify({r!r}) = {got}, want {want}")
    # an explicit never is never overridden by a kill or by context
    for r in ("chain2: s09 gate", "chain2: disk: stage L2 to NVMe needs 125 GiB free on /, only 28 GiB"):
        check(wd.classify(r, True, "s09 FAIL: HTTPError 503", "Timeout", True) == "never", f"never sticks: {r}")
    # W1a: Seagate / cpu_gates are not 'gate'
    check(wd.classify("memory did not return to baseline after soft s08_cpu_gates (95 GiB < 100)", False) == "conflict",
          "cpu_gates")
    check(wd.classify("chain2: the Seagate is not mounted at /x", False) == "conflict", "Seagate")


def test_context() -> None:
    ts = "[2026-10-21T04:00:00-04:00]"
    # W1b: chain2 wrapper + CHAIN_STEPS FAIL line
    cases = [
        ("chain2: s09 convert", f"{ts} s09 FAIL: another model-loading process is up", "", "conflict"),
        ("chain2: s09 kld L2", f"{ts} s09 FAIL: memory did not return to baseline after kld L2 (90 GiB)", "", "conflict"),
        ("chain2: s09 convert", f"{ts} s09 FAIL: convert: curl: (28) Operation timed out", "", "transient"),
        ("chain2: s09 kld L2", f"{ts} s09 FAIL: kld L2 gate failed (top1 0.81)", "", "never"),
        ("chain2: s09 kld L2", f"{ts} s09 FAIL: kld L2 sha mismatch", "Timeout", "never"),
        # W1c: python reclaim failure inside a chain1 stage
        ("s06 heal failed (resumable)", "",
         "Traceback ...\nRuntimeError: reclaim left 95.2 GiB < required 100.0 GiB; refusing to continue", "conflict"),
        # W1d: network failures in s07 nvfp4 / s12 re-render
        ("s07 nvfp4 failed (resumable)", "",
         "huggingface_hub.errors.HfHubHTTPError: 502 Server Error: Bad Gateway for url: https://huggingface.co/api",
         "transient"),
        ("chain2: s12 release re-render", "", "requests.exceptions.ConnectionError: ('Connection aborted.')", "transient"),
        ("chain2: s12 final re-render", "", "RuntimeError: xet upload session failed", "transient"),
        ("s04 failed (resumable; see ops/logs/s04.log)", "", "SystemExit: s05: the Seagate is not mounted at /media",
         "conflict"),
        # an unrelated tail does not rescue a real failure
        ("s07 thor build failed (resumable)", "", "AssertionError: layer 3 cos 0.91 < 0.999", "never"),
    ]
    for reason, fail, tail, want in cases:
        got = wd.classify(reason, False, fail, tail)
        check(got == want, f"classify({reason!r}, fail={fail[-50:]!r}, tail={tail[-50:]!r}) = {got}, want {want}")
    # W1e: stall kill
    check(wd.classify("s07 nvfp4 failed (resumable)", False, stall_killed=True) == "stall", "stall class")
    check(wd.classify("s07 nvfp4 failed (resumable)", True, stall_killed=True) == "memguard", "memguard first")
    st = {"stall_kill": 1000.0}
    check(wd.stall_killed_near(st, 1000.0 + 120), "stall kill 2 min before STOP")
    check(not wd.stall_killed_near(st, 1000.0 + 3600), "stall kill 1 h before STOP is stale")
    check(not wd.stall_killed_near({}, 1000.0), "no stall kill")


def test_retry_key() -> None:
    k = wd.retry_key
    check(k("memory did not return to baseline after s04 (95 GiB < 100)")
          != k("memory did not return to baseline after s06 (95 GiB < 100)"), "s04/s06 keys stay separate")
    check(k("memory did not return to baseline after s04 (95 GiB < 100)")
          == k("memory did not return to baseline after s04 (93.5 GiB < 100)"), "values normalised")
    check(k("chain2: backbone copy model-00003-of-00048.safetensors")
          == k("chain2: backbone copy model-00041-of-00048.safetensors"), "shard numbers normalised")
    check(k("chain2: s09 kld L2") != k("chain2: s09 kld L3"), "levels kept")
    check(k("memory did not return to baseline after soft s08_cpu_gates (1 GiB)")
          != k("memory did not return to baseline after soft s09_cpu_gates (1 GiB)"), "stage ids kept")


def test_reset() -> None:
    st = {"retries": {"x": 3}}
    check(not wd.reset_retries_on_progress(st, 10) and st["retries"] == {"x": 3}, "first tick only records ndone")
    check(not wd.reset_retries_on_progress(st, 10) and st["retries"] == {"x": 3}, "no progress, no reset")
    check(wd.reset_retries_on_progress(st, 11) and st["retries"] == {} and st["ndone"] == 11, "new done.* resets")
    st["retries"] = {"y": 1}
    check(not wd.reset_retries_on_progress(st, 9) and st["retries"] == {"y": 1}, "fewer markers (rm) is no reset")
    with tempfile.TemporaryDirectory() as d:
        for n in ("done.s04", "done.c2_x", "chain.log"):
            Path(d, n).touch()
        check(wd.n_done(Path(d)) == 2, "n_done counts done.*")


def test_seagate() -> None:
    st: dict = {}
    check(wd.seagate_check(st, False, ["chain2"], 0.0) == "" and st["sea_unmounted_since"] == 0.0, "grace starts")
    check(wd.seagate_check(st, False, ["chain2"], 1800.0) == "", "inside grace")
    check("Seagate unmounted" in wd.seagate_check(st, False, ["chain2"], 3700.0), "after 1 h")
    check(wd.seagate_check(st, True, ["chain2"], 3800.0) == "" and "sea_unmounted_since" not in st, "mounted clears")
    check(wd.seagate_check(st, False, [], 9e9) == "" and "sea_unmounted_since" not in st, "nothing due")


def test_logs_and_state() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        t_stop = datetime.fromisoformat("2026-10-21T04:10:00-04:00").timestamp()
        (d / "chain.log").write_text(
            "[2026-10-21T01:00:00-04:00] s08 FAIL: old failure\n"
            "[2026-10-21T04:09:30-04:00] s09 FAIL: another model-loading process is up\n"
            "raw teed line FAIL: no timestamp\n"
            "[2026-10-21T04:09:31-04:00] chain2 STOP: s09 convert\n"
            "[2026-10-21T05:00:00-04:00] s09 FAIL: after the stop\n")
        f = wd.fail_line_before(t_stop, d / "chain.log")
        check("another model-loading" in f, f"fail_line_before -> {f!r}")
        check(wd.fail_line_before(t_stop - 7200, d / "chain.log") == "", "window excludes old FAIL lines")
        for name, age in (("s04.log", 60), ("s09_convert.log", 30), ("sd_health.log", 0), ("s01_old.log", 9 * 3600)):
            p = d / name
            p.write_text("\n".join(f"{name} line {i}" for i in range(50)))
            os.utime(p, (t_stop - age, t_stop - age))
        name, tail = wd.stage_log_tail(t_stop, d)
        check(name == "s09_convert.log" and len(tail.splitlines()) == 20, f"stage_log_tail -> {name}")
        # O2: corrupt state resets, writes are atomic
        old = wd.STATE
        try:
            wd.STATE = d / "sd_state.json"
            wd.STATE.write_text('{"retries": {"a"')
            check(wd.load_state() == {}, "truncated state -> {}")
            wd.STATE.write_text("")
            check(wd.load_state() == {}, "empty state -> {}")
            wd.save_state({"ndone": 3})
            check(json.loads(wd.STATE.read_text()) == {"ndone": 3}, "save_state round trip")
            check(not list(d.glob(".sd_state.json.tmp*")), "no tmp left behind")
        finally:
            wd.STATE = old


class Tick:
    """main() against a temp ops/logs with every side effect stubbed."""

    def __init__(self, d: Path):
        self.d, self.calls, self.runs = d, [], []
        self.saved = {k: getattr(wd, k) for k in ("L", "STOP", "STATE", "HEALTH", "OWNER", "FOLLOW", "HIST",
                                                   "MEMGUARD_LOG", "LEAF_STOP", "SEA", "systemctl", "chains",
                                                   "procs", "subprocess")}
        wd.L, wd.STOP, wd.STATE = d, d / "chain.STOP", d / "sd_state.json"
        wd.HEALTH, wd.OWNER, wd.FOLLOW, wd.HIST = d / "sd_health.log", d / "NEEDS_OWNER", d / "OWNER_FOLLOWUP", d / "hist"
        wd.MEMGUARD_LOG, wd.LEAF_STOP, wd.SEA = d / "memguard.log", d / "runner.stop", d / "no-seagate"
        wd.systemctl = lambda *a: (self.calls.append(a), "inactive")[1]
        wd.chains = lambda P: {}
        wd.procs = lambda: {}
        sp = types.SimpleNamespace(run=lambda *a, **k: self.runs.append(a))
        wd.subprocess = sp

    def restore(self):
        for k, v in self.saved.items():
            setattr(wd, k, v)


def test_main_ticks() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tk = Tick(d)
        try:
            (d / "chain.log").write_text("")
            (d / "done.s03").touch()
            # 1. a transient STOP older than the retry gap is retried; the budget is recorded under its key
            reason = "chain2: s12 gguf publish"
            wd.STOP.write_text(f"2026-10-21T00:00:00-04:00 {reason}\n")
            old = time.time() - 2 * wd.RETRY_GAP_S
            os.utime(wd.STOP, (old, old))
            wd.main(False)
            st = json.loads(wd.STATE.read_text())
            check(not wd.STOP.exists() and len(list(wd.HIST.iterdir())) == 1, "transient STOP moved to history")
            check(st["retries"] == {wd.retry_key(reason): 1}, f"retry recorded: {st.get('retries')}")
            check(("start", "--no-block", "dsv41-chain.service") in tk.calls, "chains restarted after the retry")
            # 2. W2: a new done.* resets the budget on the next tick
            (d / "done.s04").touch()
            wd.main(False)
            st = json.loads(wd.STATE.read_text())
            check(st["retries"] == {} and st["ndone"] == 2, f"budget reset on progress: {st}")
            # 3. a never STOP with a FAIL line: NEEDS_OWNER with context, no start
            tk.calls.clear()
            t_stop = int(time.time()) - 3600           # older than RETRY_GAP_S, newer than the FAIL-line window
            iso = lambda t: datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")
            wd.STOP.write_text(f"{iso(t_stop)} chain2: s09 convert\n")
            os.utime(wd.STOP, (t_stop, t_stop))
            with open(d / "chain.log", "a") as f:
                f.write(f"[{iso(t_stop - 1)}] s09 FAIL: convert_v41_reap.py: tensor count mismatch\n")
            wd.main(False)
            own = wd.OWNER.read_text()
            check(own.startswith("STOP (not retryable): chain2: s09 convert") and "tensor count mismatch" in own,
                  f"NEEDS_OWNER: {own!r}")
            check(not any(c[0] == "start" for c in tk.calls), "no start while a STOP is present")
            n_log = (d / "chain.log").read_text().count("needs the owner")
            wd.main(False)
            check((d / "chain.log").read_text().count("needs the owner") == n_log, "owner item logged once")
            # 4. the same wrapper with a conflict FAIL line is retried instead (W1b)
            wd.STOP.write_text(f"{iso(t_stop)} chain2: s09 kld L2\n")
            os.utime(wd.STOP, (t_stop, t_stop))
            with open(d / "chain.log", "a") as f:
                f.write(f"[{iso(t_stop)}] s09 FAIL: another model-loading process is up\n")
            wd.main(False)
            check(not wd.STOP.exists() and not wd.OWNER.exists(), "conflict FAIL line -> retried, NEEDS_OWNER cleared")
            # 5. W3: Seagate unmounted > 1 h while chains are due -> NEEDS_OWNER; mounted clears it
            st = json.loads(wd.STATE.read_text())
            st["sea_unmounted_since"] = time.time() - 2 * 3600
            wd.STATE.write_text(json.dumps(st))
            wd.main(False)
            check(wd.OWNER.exists() and "Seagate unmounted" in wd.OWNER.read_text(), "Seagate NEEDS_OWNER")
            wd.SEA = Path("/")              # a mount point
            wd.main(False)
            check(not wd.OWNER.exists(), "Seagate mounted -> NEEDS_OWNER removed")
            # 6. O2: a truncated state file does not stop the tick
            wd.STATE.write_text('{"sig": 12')
            tk.calls.clear()
            wd.main(False)
            check(("start", "--no-block", "dsv41-chain2.service") in tk.calls, "tick survives a corrupt state")
            # 7. O6: chain2 finished with follow-ups -> leafllm stays paused, owner told
            (d / "done.c2_s12_final").touch()
            with open(d / "chain.log", "a") as f:
                f.write("[2026-10-21T05:00:00-04:00] chain2 chain2 reached the end: x\n")
            wd.FOLLOW.write_text("a\nb\n")
            wd.LEAF_STOP.write_text("")
            wd.main(False)
            check(wd.LEAF_STOP.exists() and "2 OWNER_FOLLOWUP" in wd.OWNER.read_text(), "O6 owner note")
            wd.FOLLOW.unlink()
            wd.main(False)
            check(not wd.LEAF_STOP.exists() and not wd.OWNER.exists(), "leafllm resumes without follow-ups")
            # 8. dry run writes neither state nor NEEDS_OWNER
            wd.STATE.unlink()
            wd.STOP.write_text("2026-10-21T04:09:31-04:00 chain2: s09 gate\n")
            wd.main(True)
            check(not wd.STATE.exists() and not wd.OWNER.exists() and wd.STOP.exists(), "dry run is read-only")
        finally:
            tk.restore()


if __name__ == "__main__":
    for fn in (test_real_reasons, test_context, test_retry_key, test_reset, test_seagate, test_logs_and_state,
               test_main_ticks):
        fn()
        print(f"{fn.__name__}: done")
    print(f"{'FAILED ' + str(FAILS) if FAILS else 'all ok'} ({WD_PATH})")
    sys.exit(1 if FAILS else 0)
