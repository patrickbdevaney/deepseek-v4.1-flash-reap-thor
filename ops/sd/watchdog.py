#!/usr/bin/env python3
"""dsv41 watchdog (dsv41-watchdog.timer, every 15 min and 3 min after boot): the unattended replacement for the
interactive check-in, from s04 through the REAP and GGUF HF publishes and s11.

Each tick:
  1. writes one health line to ops/logs/sd_health.log (chains, current stage, memory, disks, STOP);
  2. stall check: if the chains' working processes (everything below chain.sh/chain2.sh except bash/sleep) made no
     progress for STALL_S -- no CPU time, no read/write bytes (sockets included, so a stuck upload counts) -- they
     are killed by exact PID and the kill time is recorded; the chain step then fails, writes chain.STOP, and
     step 3 treats that STOP like a memguard kill (class "stall");
  3. chain.STOP: retried only when the reason is a transient class (network/upload, a memguard or stall kill, a
     resource conflict), a bounded number of times, never sooner than RETRY_GAP_S after the STOP. Gates,
     sha/verify failures, missing artifacts, disk shortfalls and owner pauses are NEVER retried: they stay for the
     owner, flagged in ops/logs/NEEDS_OWNER. A retried STOP is moved to ops/logs/stop_history/.
     The STOP line is often only a wrapper ("chain2: s09 convert", "s06 heal failed"): when it does not decide
     the class, the last `FAIL:` line chain.log got just before the STOP and the tail of the newest stage log
     (ops/logs/sNN*.log) are read too (W1). Retry budgets reset whenever a new ops/logs/done.* appears (W2);
  4. starts dsv41-chain.service and dsv41-chain2.service (no-ops when active; should_run.sh skips finished chains,
     a STOP, or an unmounted Seagate; chain.sh/chain2.sh flocks reject a duplicate of a setsid-launched copy).
     A Seagate unmounted for more than SEA_GRACE_S while a chain is due (not finished, not running, no STOP)
     is written to NEEDS_OWNER (W3): should_run.sh would otherwise skip silently forever;
  5. leafllm: removes ~/leafllm/run/runner.stop once chain2 reached its end with no STOP and no OWNER_FOLLOWUP
     (soft-failed CUDA/s11 work is Thor-exclusive work still owed). Never earlier. If chain2 finished WITH
     follow-ups, NEEDS_OWNER says that leafllm stays paused until the owner reviews them (O6).

State (sd_state.json) and NEEDS_OWNER are written atomically; a corrupt state file is reset, not fatal (O2).

    python3 -I ops/sd/watchdog.py [--dry-run] [--selftest]
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

R = Path.home() / "deepseek-v4.1-flash-reap-thor"
L = R / "ops" / "logs"
STOP = L / "chain.STOP"
STATE = L / "sd_state.json"
HEALTH = L / "sd_health.log"
OWNER = L / "NEEDS_OWNER"
FOLLOW = L / "OWNER_FOLLOWUP"      # chain2 soft failures (CUDA server, s11): the release went out without them
HIST = L / "stop_history"
MEMGUARD_LOG = R / "logs" / "memguard.log"
LEAF_STOP = Path.home() / "leafllm" / "run" / "runner.stop"
SEA = Path("/media/patrickd/Seagate Portable Drive")
STALL_S = 2 * 3600
RETRY_GAP_S = 20 * 60
SEA_GRACE_S = 3600
KILL_WINDOW = (900, 60)            # a STOP written up to 15 min after (or 1 min before) a kill is that kill's
LIMIT = {"conflict": 48, "transient": 3, "memguard": 2, "stall": 2}
IDLE = {"bash", "sleep", "flock", "tail", "timeout"}
END1 = "chain reached the end of implemented stages"
END2 = "chain2 reached the end"

# 1. a reason that STARTS with one of the fixed quiet()/reclaim()/mount phrases is a conflict, whatever follows
#    ("memory did not return to baseline after soft s08_cpu_gates", "... after chain2: s09 gate")
CONFLICT_START = re.compile(r"^(memory did not return|another model-loading process|a (leafllm )?vLLM server is up|"
                            r"leafllm runner active|the Seagate is not mounted)")
# 2. NEVER: gates, sha/verify, missing artifacts, disk shortfalls, owner pauses. `\bgate\b` so that "Seagate",
#    "cpu_gates" and "gate_v41_units" do not match; `\bsha` so that "shard" does not.
NEVER = re.compile(r"\bgates?\b|gate (failed|FAIL)|verdict|\bsha(256)?\b|mismatch|verify|has not passed|BLOCKED|"
                   r"no K-quant|not readable|no s0\d|non-empty|disk:|needs .* GiB|no GGUF|\bpaus(e|ed)\b",
                   re.I)
CONFLICT = re.compile(r"vLLM server is up|leafllm runner active|another model-loading process|a model server is running|"
                      r"memory did not return|reclaim left|not mounted")
TRANSIENT = re.compile(r"upload|publish|download|copy|stage .* to NVMe|^(un)?stage\b|master download", re.I)
# 3. context (FAIL line, stage-log tail) of a STOP the line above left undecided
NET = re.compile(r"HTTPError|HfHubHTTPError|ConnectionError|Connection (reset|refused|aborted)|Timeout|timed out|"
                 r"\b5\d\d Server Error|\bxet\b|Temporary failure in name resolution|RemoteDisconnected|"
                 r"curl: \(\d+\)|Read timed out|IncompleteRead", re.I)


def _strip(reason: str) -> str:
    return re.sub(r"^chain2:\s*", "", reason.strip())


def classify(reason: str, memguard_killed: bool, fail_line: str = "", log_tail: str = "",
             stall_killed: bool = False) -> str:
    """never | conflict | transient | memguard | stall. The STOP line decides when it can; otherwise the FAIL line
    chain.log got just before it, then the newest stage log's tail, then a recent memguard/stall kill."""
    r = _strip(reason)
    if CONFLICT_START.search(r):
        return "conflict"
    if NEVER.search(r):
        return "never"
    if CONFLICT.search(r):
        return "conflict"
    if TRANSIENT.search(r):
        return "transient"
    f = re.sub(r"^.*?FAIL:\s*", "", fail_line.strip())
    if f:
        if CONFLICT_START.search(f) or CONFLICT.search(f):
            return "conflict"
        if NEVER.search(f):
            return "never"
        if NET.search(f) or TRANSIENT.search(f):
            return "transient"
    if log_tail:
        if CONFLICT.search(log_tail):
            return "conflict"
        if NET.search(log_tail):
            return "transient"
    if memguard_killed:         # an unclassified step failure right after memguard killed a stage: rerun it
        return "memguard"
    if stall_killed:            # ... or right after this watchdog killed a stalled step
        return "stall"
    return "never"


def retry_key(reason: str) -> str:
    """Digits that are values (sizes, rc, counts) or long runs (shard numbers) -> N; stage/level/arm names such as
    s04, L2, s08_cpu_gates keep theirs, so 'after s04' and 'after s06' stay separate budgets."""
    k = re.sub(r"\d{3,}", "N", reason)
    k = re.sub(r"(?<![A-Za-z_\d])\d+(\.\d+)?", "N", k)
    return k[:160]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write(p: Path, text: str) -> None:
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def load_state() -> dict:
    try:
        st = json.loads(STATE.read_text())
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    try:
        atomic_write(STATE, json.dumps(st, indent=1))
    except OSError as e:
        health(f"could not write {STATE.name}: {e}")


def health(msg: str) -> None:
    try:
        with open(HEALTH, "a") as f:
            f.write(f"[{now()}] {msg}\n")
    except OSError:
        pass


def chain_log(msg: str) -> None:
    try:
        with open(L / "chain.log", "a") as f:
            f.write(f"[{now()}] watchdog {msg}\n")
    except OSError:
        pass


def read_tail(p: Path, n: int, max_bytes: int = 1 << 16) -> list[str]:
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    return data.decode(errors="replace").splitlines()[-n:]


def _ts(line: str) -> float | None:
    m = re.match(r"\[([0-9T:+\-.]+)\]", line)
    if not m:
        return None
    try:
        return datetime.fromisoformat(m.group(1)).timestamp()
    except ValueError:
        return None


def fail_line_before(t_stop: float, chain_log_path: Path | None = None, window: float = 900) -> str:
    """The last timestamped `FAIL:` line of chain.log written in [t_stop - window, t_stop + 5]."""
    best = ""
    for ln in read_tail(chain_log_path or (L / "chain.log"), 3000, 1 << 20):
        if "FAIL:" not in ln:
            continue
        t = _ts(ln)
        if t is not None and t_stop - window <= t <= t_stop + 5:
            best = ln
    return best


def stage_log_tail(t_stop: float, logdir: Path | None = None, n: int = 20) -> tuple[str, str]:
    """(name, last n lines) of the newest ops/logs/sNN*.log written in [t_stop - 6 h, t_stop + 5 min]."""
    d = logdir or L
    best, bt = None, -1.0
    try:
        for p in d.iterdir():
            if not re.match(r"s\d\d.*\.log$", p.name):
                continue
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if t_stop - 6 * 3600 <= m <= t_stop + 300 and m > bt:
                best, bt = p, m
    except OSError:
        return "", ""
    if best is None:
        return "", ""
    return best.name, "\n".join(read_tail(best, n))


def procs() -> dict[int, dict]:
    out = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            st = (d / "stat").read_text()
            comm = st[st.index("(") + 1:st.rindex(")")]
            f = st[st.rindex(")") + 2:].split()
            argv = (d / "cmdline").read_bytes().split(b"\0")
            out[int(d.name)] = {"ppid": int(f[1]), "comm": comm, "cpu": int(f[11]) + int(f[12]),
                                "argv": [a.decode(errors="replace") for a in argv if a]}
        except (OSError, ValueError, IndexError):
            continue
    return out


def chains(P: dict) -> dict[str, int]:
    """{'chain': pid, 'chain2': pid} for chain processes running in this repo (systemd or setsid)."""
    found = {}
    for pid, p in P.items():
        a = p["argv"]
        if len(a) == 2 and a[0] in ("bash", "/bin/bash", "/usr/bin/bash") and a[1] in ("ops/chain.sh", "ops/chain2.sh"):
            try:
                if Path(f"/proc/{pid}/cwd").resolve() != R:
                    continue
            except OSError:
                continue
            found[a[1][4:-3]] = pid
    return found


def workers(P: dict, roots: list[int]) -> list[int]:
    kids: dict[int, list[int]] = {}
    for pid, p in P.items():
        kids.setdefault(p["ppid"], []).append(pid)
    out, todo = [], list(roots)
    while todo:
        q = todo.pop()
        for c in kids.get(q, []):
            todo.append(c)
            if P[c]["comm"] not in IDLE:
                out.append(c)
    return sorted(out)


def progress(P: dict, pids: list[int]) -> int:
    sig = 0
    for q in pids:
        sig += P[q]["cpu"]
        try:
            for line in Path(f"/proc/{q}/io").read_text().splitlines():
                k, v = line.split(":")
                if k in ("rchar", "wchar"):
                    sig += int(v)
        except OSError:
            pass
    return sig


def gib_free(p: Path) -> str:
    try:
        return f"{shutil.disk_usage(p).free / 2**30:.0f}G"
    except OSError:
        return "n/a"


def mem_avail() -> str:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return f"{int(line.split()[1]) / 2**20:.0f}G"
    except OSError:
        pass
    return "?"


def memguard_killed_near(t_stop: float) -> bool:
    lines = read_tail(MEMGUARD_LOG, 400)
    for ln in lines:
        if "killing stage" not in ln:
            continue
        m = re.match(r"\[memguard (\S+)\]", ln)
        try:
            t = datetime.fromisoformat(m.group(1)).timestamp()
        except (AttributeError, ValueError):
            continue
        if t_stop - KILL_WINDOW[0] <= t <= t_stop + KILL_WINDOW[1]:
            return True
    return False


def stall_killed_near(st: dict, t_stop: float) -> bool:
    t = st.get("stall_kill")
    return isinstance(t, (int, float)) and t_stop - KILL_WINDOW[0] <= t <= t_stop + KILL_WINDOW[1]


def n_done(logdir: Path | None = None) -> int:
    try:
        return sum(1 for p in (logdir or L).iterdir() if p.name.startswith("done."))
    except OSError:
        return 0


def reset_retries_on_progress(st: dict, nd: int) -> bool:
    """W2: a new done.* marker is progress, so every retry budget starts over. Returns True if it reset."""
    prev = st.get("ndone")
    st["ndone"] = nd
    if isinstance(prev, int) and nd > prev and st.get("retries"):
        st["retries"] = {}
        return True
    return False


def chain_finished(name: str, logdir: Path | None = None) -> bool:
    d = logdir or L
    marker, end = (("done.s12", END1) if name == "chain" else ("done.c2_s12_final", END2))
    if not (d / marker).exists():
        return False
    try:
        return end in (d / "chain.log").read_text(errors="replace")
    except OSError:
        return False


def seagate_check(st: dict, mounted: bool, due: list[str], t: float) -> str:
    """W3: '' or the NEEDS_OWNER message once the Seagate has been unmounted > SEA_GRACE_S while a chain is due."""
    if mounted or not due:
        st.pop("sea_unmounted_since", None)
        return ""
    since = st.setdefault("sea_unmounted_since", t)
    if t - since < SEA_GRACE_S:
        return ""
    return (f"Seagate unmounted for > {SEA_GRACE_S // 3600} h while {'+'.join(due)} is due: should_run.sh skips it "
            f"until {SEA} is mounted (udisksctl needs the desktop session)")


def systemctl(*args: str) -> str:
    r = subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def write_owner(msgs: list[str], dry: bool) -> None:
    """NEEDS_OWNER holds one line per open item plus a footer; rewritten (atomically) only when the set changes."""
    try:
        old = OWNER.read_text().splitlines() if OWNER.exists() else []
    except OSError:
        old = []
    if not msgs:
        if OWNER.exists() and not dry:
            try:
                OWNER.unlink()
            except OSError:
                pass
        return
    key = lambda m: m.split(" [", 1)[0]          # the bracketed context may drift between ticks; the item does not
    if [key(m) for m in old[:len(msgs)]] == [key(m) for m in msgs] and len(old) == len(msgs) + 1:
        return
    if dry:
        return
    for m in msgs:
        if key(m) not in {key(o) for o in old}:
            chain_log(f"needs the owner: {m}")
    try:
        atomic_write(OWNER, "\n".join(msgs) + f"\nsince {now()}; fix, then `rm {STOP}` if a STOP is listed -- "
                     "the watchdog restarts the chains\n")
    except OSError as e:
        health(f"could not write {OWNER.name}: {e}")


def main(dry: bool) -> None:
    st = load_state()
    P = procs()
    ch = chains(P)
    work = workers(P, list(ch.values()))
    stage = " | ".join(" ".join(P[q]["argv"][:4])[:90] for q in work[:3]) or "-"
    owner: list[str] = []
    if reset_retries_on_progress(st, n_done()):
        health("new done.* marker: retry budgets reset")

    # ---- 2. stall
    sig = progress(P, work) if work else None
    t = time.time()
    if work and sig == st.get("sig") and st.get("work") == work:
        idle = t - st.get("since", t)
        if idle >= STALL_S:
            health(f"STALL: no CPU/IO progress for {idle / 60:.0f} min in {work} ({stage}); killing them")
            chain_log(f"STALL: no CPU/IO progress for {idle / 60:.0f} min; killed {work} ({stage})")
            if not dry:
                st["stall_kill"] = t
                save_state(st)      # the kill time must survive a crash of this tick (W1e)
                for q in work:
                    try:
                        os.kill(q, signal.SIGTERM)
                    except OSError:
                        pass
                time.sleep(60)
                for q in work:
                    if Path(f"/proc/{q}").exists():
                        try:
                            os.kill(q, signal.SIGKILL)
                        except OSError:
                            pass
            st.pop("sig", None)
    else:
        st.update(sig=sig, work=work, since=t)

    # ---- 3. STOP
    note = ""
    if STOP.exists():
        try:
            raw = STOP.read_text().strip()
            t_stop = STOP.stat().st_mtime
        except OSError:
            raw, t_stop = "", t
        reason = re.sub(r"^\S+\s+", "", raw, count=1)   # drop the timestamp
        fail = fail_line_before(t_stop)
        lname, tail = stage_log_tail(t_stop)
        cls = classify(reason, memguard_killed_near(t_stop), fail, tail, stall_killed_near(st, t_stop))
        key = retry_key(reason)
        n = st.setdefault("retries", {}).get(key, 0)
        ctx = ""
        if fail:
            ctx += f" [FAIL line: {re.sub(r'^.*?FAIL:', '', fail).strip()[:200]}]"
        if tail and cls == "never":
            last = [x for x in tail.splitlines() if x.strip()][-3:]
            ctx += f" [{lname} tail: {' / '.join(x.strip()[:120] for x in last)}]"
        if cls == "never" or n >= LIMIT[cls]:
            why = "not retryable" if cls == "never" else f"{cls} retry limit {LIMIT[cls]} reached"
            note = f"STOP ({why}): {reason}"
            owner.append(note + ctx)
        elif t - t_stop < RETRY_GAP_S or ch:
            note = f"STOP ({cls}, retry {n + 1}/{LIMIT[cls]} pending): {reason}"
        else:
            note = f"STOP ({cls}) retried {n + 1}/{LIMIT[cls]}: {reason}"
            chain_log(f"retrying a {cls} STOP ({n + 1}/{LIMIT[cls]}): {reason}{ctx}")
            if not dry:
                HIST.mkdir(exist_ok=True)
                STOP.rename(HIST / f"{datetime.now():%Y%m%dT%H%M%S}.STOP")
                st["retries"][key] = n + 1
                st.pop("stall_kill", None)
                subprocess.run(["sync"])
                subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], capture_output=True)

    # ---- 4. (re)start
    if not STOP.exists() and not dry:
        for u in ("dsv41-chain.service", "dsv41-chain2.service"):
            if systemctl("is-active", u) not in ("active", "activating"):
                systemctl("start", "--no-block", u)

    mounted = os.path.ismount(SEA)
    due = [c for c in ("chain", "chain2") if c not in ch and not chain_finished(c)] if not STOP.exists() else []
    sea = seagate_check(st, mounted, due, t)
    if sea:
        owner.append(sea)

    # ---- 5. leafllm
    done2 = chain_finished("chain2")
    if done2 and not STOP.exists() and not FOLLOW.exists() and not ch and LEAF_STOP.exists():
        chain_log("REAP Thor-exclusive stages done: removing ~/leafllm/run/runner.stop (leafllm resumes)")
        if not dry:
            LEAF_STOP.unlink()
    elif done2 and not STOP.exists() and FOLLOW.exists() and LEAF_STOP.exists():
        try:
            nf = len(FOLLOW.read_text().splitlines())
        except OSError:
            nf = "?"
        owner.append(f"chain2 finished with {nf} OWNER_FOLLOWUP item(s): leafllm stays paused until the owner reviews "
                     f"{FOLLOW} and removes {LEAF_STOP} (no chain may be running then)")

    write_owner(owner, dry)
    try:
        follow = f" follow-ups={len(FOLLOW.read_text().splitlines())}" if FOLLOW.exists() else ""
    except OSError:
        follow = ""
    health(f"chains={ch or '-'} work={stage} mem={mem_avail()} nvme={gib_free(Path.home())} "
           f"seagate={gib_free(SEA) if mounted else 'UNMOUNTED'} {note or 'no STOP'}{follow}"
           + (" NEEDS_OWNER" if owner else ""))
    if not dry:
        save_state(st)


def selftest() -> None:
    cases = {
        "s05 surgery/upload failed (resumable)": "transient",
        "chain2: s12 gguf publish": "transient",
        "chain2: master download x/y": "transient",
        "s01 2k gate: rc=1 verdict=FAIL": "never",
        "chain2: s09 gate": "never",
        "chain2: backbone: Seagate copy of x does not match the pinned sha256": "never",
        "chain2: disk: quant L0 needs 320 GiB free on /, only 200 GiB": "never",
        "a leafllm vLLM server is up": "conflict",
        "memory did not return to baseline after s04 (90 GiB < 100)": "conflict",
        "memory did not return to baseline after soft s08_cpu_gates (95 GiB < 100)": "conflict",
        "chain2: the Seagate is not mounted at /media/patrickd/Seagate Portable Drive": "conflict",
        "s06 heal failed (resumable)": "never",
        "chain2: s08 decode A192": "never",
        "s11 end-to-end gate BLOCKED: cuda/ has no DSpark draft+verify path": "never",
    }
    for r, want in cases.items():
        got = classify(r, False)
        assert got == want, (r, got, want)
    assert classify("s06 heal failed (resumable)", True) == "memguard"
    # env.assert_no_gpu_server refusals land in the stage log only: wait for the server, never escalate
    assert classify("s06 heal failed (resumable)", False,
                    log_tail="[s06] REFUSING: a model server is running (863227 llama-server -m q.gguf).") == "conflict"
    assert classify("s07 nvfp4 failed (resumable)", False, stall_killed=True) == "stall"
    assert classify("chain2: s09 gate", True) == "never"
    print(f"selftest ok ({len(cases) + 4} cases)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main("--dry-run" in sys.argv)
