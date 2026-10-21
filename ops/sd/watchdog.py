#!/usr/bin/env python3
"""dsv41 watchdog (dsv41-watchdog.timer, every 15 min and 3 min after boot): the unattended replacement for the
interactive check-in, from s04 through the REAP and GGUF HF publishes and s11.

Each tick:
  1. writes one health line to ops/logs/sd_health.log (chains, current stage, memory, disks, STOP);
  2. stall check: if the chains' working processes (everything below chain.sh/chain2.sh except bash/sleep) made no
     progress for STALL_S -- no CPU time, no read/write bytes (sockets included, so a stuck upload counts) -- they
     are killed by exact PID; the chain step then fails, writes chain.STOP, and step 3 decides;
  3. chain.STOP: retried only when the reason is a transient class (network/upload, a memguard kill, a resource
     conflict), a bounded number of times, never sooner than RETRY_GAP_S after the STOP. Gates, sha/verify
     failures, missing artifacts and disk shortfalls are NEVER retried: they stay for the owner, flagged in
     ops/logs/NEEDS_OWNER. A retried STOP is moved to ops/logs/stop_history/;
  4. starts dsv41-chain.service and dsv41-chain2.service (no-ops when active; should_run.sh skips finished chains,
     a STOP, or an unmounted Seagate; chain.sh/chain2.sh flocks reject a duplicate of a setsid-launched copy);
  5. leafllm: removes ~/leafllm/run/runner.stop once chain2 reached its end with no STOP (the REAP's
     Thor-exclusive work is done). Never earlier.

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
HIST = L / "stop_history"
MEMGUARD_LOG = R / "logs" / "memguard.log"
LEAF_STOP = Path.home() / "leafllm" / "run" / "runner.stop"
SEA = Path("/media/patrickd/Seagate Portable Drive")
STALL_S = 2 * 3600
RETRY_GAP_S = 20 * 60
LIMIT = {"conflict": 48, "transient": 3, "memguard": 2}
IDLE = {"bash", "sleep", "flock", "tail", "timeout"}

# order matters: a NEVER match wins over everything (a failed upload of a gate-failed artifact is still a gate)
NEVER = re.compile(r"gate|verdict|sha|mismatch|verify|has not passed|BLOCKED|no K-quant|not readable|no s0\d|"
                   r"non-empty|disk:|needs .* GiB|no GGUF", re.I)
CONFLICT = re.compile(r"vLLM server is up|leafllm runner active|another model-loading process|"
                      r"memory did not return|not mounted")
TRANSIENT = re.compile(r"upload|publish|download|copy|stage .* to NVMe|unstage|master download", re.I)


def classify(reason: str, memguard_killed: bool) -> str:
    if NEVER.search(reason):
        return "never"
    if CONFLICT.search(reason):
        return "conflict"
    if TRANSIENT.search(reason):
        return "transient"
    if memguard_killed:         # an unclassified step failure right after memguard killed a stage: rerun it
        return "memguard"
    return "never"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def health(msg: str) -> None:
    with open(HEALTH, "a") as f:
        f.write(f"[{now()}] {msg}\n")


def chain_log(msg: str) -> None:
    with open(L / "chain.log", "a") as f:
        f.write(f"[{now()}] watchdog {msg}\n")


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
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return f"{int(line.split()[1]) / 2**20:.0f}G"
    return "?"


def memguard_killed_near(t_stop: float) -> bool:
    try:
        lines = MEMGUARD_LOG.read_text(errors="replace").splitlines()[-400:]
    except OSError:
        return False
    for ln in lines:
        if "killing stage" not in ln:
            continue
        m = re.match(r"\[memguard (\S+)\]", ln)
        try:
            t = datetime.fromisoformat(m.group(1)).timestamp()
        except (AttributeError, ValueError):
            continue
        if t_stop - 900 <= t <= t_stop + 60:
            return True
    return False


def systemctl(*args: str) -> str:
    r = subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def main(dry: bool) -> None:
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    P = procs()
    ch = chains(P)
    work = workers(P, list(ch.values()))
    stage = " | ".join(" ".join(P[q]["argv"][:4])[:90] for q in work[:3]) or "-"

    # ---- 2. stall
    sig = progress(P, work) if work else None
    t = time.time()
    if work and sig == st.get("sig") and st.get("work") == work:
        idle = t - st.get("since", t)
        if idle >= STALL_S:
            health(f"STALL: no CPU/IO progress for {idle / 60:.0f} min in {work} ({stage}); killing them")
            chain_log(f"STALL: no CPU/IO progress for {idle / 60:.0f} min; killed {work} ({stage})")
            if not dry:
                for q in work:
                    os.kill(q, signal.SIGTERM)
                time.sleep(60)
                for q in work:
                    if Path(f"/proc/{q}").exists():
                        os.kill(q, signal.SIGKILL)
            st.pop("sig", None)
    else:
        st.update(sig=sig, work=work, since=t)

    # ---- 3. STOP
    note = ""
    if STOP.exists():
        reason = re.sub(r"^\S+\s+", "", STOP.read_text().strip(), count=1)   # drop the timestamp
        t_stop = STOP.stat().st_mtime
        cls = classify(reason, memguard_killed_near(t_stop))
        key = re.sub(r"\d+(\.\d+)?", "N", reason)[:160]
        n = st.setdefault("retries", {}).get(key, 0)
        if cls == "never" or n >= LIMIT[cls]:
            why = "not retryable" if cls == "never" else f"{cls} retry limit {LIMIT[cls]} reached"
            note = f"STOP ({why}): {reason}"
            if not OWNER.exists() or OWNER.read_text().splitlines()[0] != note:
                OWNER.write_text(f"{note}\nsince {now()}; fix, then `rm {STOP}` -- the watchdog restarts the chains\n")
                chain_log(f"STOP needs the owner ({why}): {reason}")
        elif t - t_stop < RETRY_GAP_S or ch:
            note = f"STOP ({cls}, retry {n + 1}/{LIMIT[cls]} pending): {reason}"
        else:
            note = f"STOP ({cls}) retried {n + 1}/{LIMIT[cls]}: {reason}"
            chain_log(f"retrying a {cls} STOP ({n + 1}/{LIMIT[cls]}): {reason}")
            if not dry:
                HIST.mkdir(exist_ok=True)
                STOP.rename(HIST / f"{datetime.now():%Y%m%dT%H%M%S}.STOP")
                st["retries"][key] = n + 1
                subprocess.run(["sync"])
                subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], capture_output=True)
    elif OWNER.exists():
        OWNER.unlink()

    # ---- 4. (re)start
    if not STOP.exists() and not dry:
        for u in ("dsv41-chain.service", "dsv41-chain2.service"):
            if systemctl("is-active", u) not in ("active", "activating"):
                systemctl("start", "--no-block", u)

    # ---- 5. leafllm
    done2 = (L / "done.c2_s12_final").exists() and "chain2 reached the end" in (L / "chain.log").read_text(errors="replace")
    if done2 and not STOP.exists() and not ch and LEAF_STOP.exists():
        chain_log("REAP Thor-exclusive stages done: removing ~/leafllm/run/runner.stop (leafllm resumes)")
        if not dry:
            LEAF_STOP.unlink()

    mounted = os.path.ismount(SEA)
    health(f"chains={ch or '-'} work={stage} mem={mem_avail()} nvme={gib_free(Path.home())} "
           f"seagate={gib_free(SEA) if mounted else 'UNMOUNTED'} {note or 'no STOP'}")
    if not dry:
        STATE.write_text(json.dumps(st, indent=1))


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
        "s06 heal failed (resumable)": "never",
        "chain2: s08 decode A192": "never",
        "s11 end-to-end gate BLOCKED: cuda/ has no DSpark draft+verify path": "never",
    }
    for r, want in cases.items():
        got = classify(r, False)
        assert got == want, (r, got, want)
    assert classify("s06 heal failed (resumable)", True) == "memguard"
    assert classify("chain2: s09 gate", True) == "never"
    print(f"selftest ok ({len(cases) + 2} cases)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main("--dry-run" in sys.argv)
