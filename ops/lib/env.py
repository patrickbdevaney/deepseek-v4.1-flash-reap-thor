"""Paths, logging, ledgers and memory reclaim shared by every ops stage.

Nothing here imports torch, so the corpus stage and the tests can use it on a box where a GPU
job is running. Stages that need the reference model call `import_reference()`.

Conventions (from ~/glm-5.3-reap and the Thor memory notes):
  * every multi-hour stage runs under setsid/systemd, never as a child of an interactive shell;
  * one decoder layer per process: a process exit does NOT return the Tegra nvmap pool, only
    `drop_caches 3` does, so the orchestrator reclaims and verifies between layers;
  * every resumable unit is recorded in a JSON ledger written atomically (tmp + rename), and a
    unit's outputs are written before the ledger says it is done.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
REPO = OPS.parent

# The pinned teacher: deepseek-ai/DeepSeek-V4.1-Flash-1017 at this revision (s00_fetch.sh).
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SRC = Path(os.environ.get("V41_SRC", Path.home() / "models" / "DeepSeek-V4.1-Flash-1017"))
WORK = Path(os.environ.get("V41_WORK", Path.home() / "models" / "v41-work"))
INFERENCE = SRC / "inference"
ENCODING = SRC / "encoding"
LOGS = OPS / "logs"
for _d in (WORK, LOGS):
    _d.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str, stage: str = "-", level: str = "INFO") -> None:
    print(f"[{now()}] {level:5s} {stage:16s} {msg}", flush=True)


def atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Ledger:
    """A JSON dict of unit -> record, persisted after every change.

    A unit is done only when its record has "done": true. Records are written after the unit's
    outputs, so a crash between the two re-runs the unit rather than trusting half-written files.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = json.load(open(self.path)) if self.path.exists() else {}

    def done(self, unit: str) -> bool:
        return bool(self.data.get(unit, {}).get("done"))

    def get(self, unit: str, default=None):
        return self.data.get(unit, default)

    def mark(self, unit: str, **rec) -> None:
        self.data[unit] = {**self.data.get(unit, {}), **rec, "ts": now()}
        atomic_json(self.path, self.data)

    def finish(self, unit: str, **rec) -> None:
        self.mark(unit, done=True, **rec)


def mem_available_gib() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    raise RuntimeError("MemAvailable missing")


def reclaim(stage: str, *, expect_gib: float | None = None) -> float:
    """sync + drop_caches 3, then report MemAvailable. With `expect_gib`, a reclaim that leaves
    less than that available is a failure, not a warning: the next layer would start on a box
    whose driver pool did not come back, which is how the GLM pass wedged the board."""
    subprocess.run(["sync"], timeout=120, check=False)
    r = subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                       timeout=120, check=False, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"drop_caches failed rc={r.returncode}: {r.stderr.strip()}", stage, "WARN")
    a = mem_available_gib()
    log(f"reclaim: MemAvailable {a:.1f} GiB", stage)
    if expect_gib is not None and a < expect_gib:
        raise RuntimeError(f"reclaim left {a:.1f} GiB < required {expect_gib:.1f} GiB; refusing to continue")
    return a


def assert_no_gpu_server(stage: str) -> None:
    """Refuse to start a GPU stage beside a vLLM/llama server (Thor unified-memory rule)."""
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        out = ""
    busy = [n for n in out.split() if n.startswith(("leaf-teacher", "vllm"))]
    ps = subprocess.run(["pgrep", "-fa", "llama-server|vllm serve"], capture_output=True, text=True).stdout
    if busy or ps.strip():
        raise SystemExit(f"[{stage}] REFUSING: a model server is running ({busy or ps.strip()[:120]}). "
                         "Never co-run beside a vLLM/llama server on Thor.")


def import_reference(kernels: str = "tilelang"):
    """Import the checkpoint's own inference/model.py (there is no transformers implementation).

    Returns the `model` module. Globals that Transformer.__init__ would set are set here, because
    the stages build single Blocks and never construct a Transformer.

    model.py does `from kernel import ...` at import time, and kernel.py imports tilelang. With
    kernels="torch" (or when tilelang is missing) the module name `kernel` is pre-bound to
    ops/lib/torch_kernels.py, so the reference imports cleanly without TileLang."""
    for p in (str(INFERENCE), str(ENCODING)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch  # noqa: F401
    if kernels == "torch":
        from . import torch_kernels
        sys.modules["kernel"] = torch_kernels
    else:
        import kernel  # noqa: F401  (raises here, with the real error, if TileLang will not import)
    import model as M  # the reference module
    M.world_size, M.rank = 1, 0
    M.default_dtype = torch.float8_e4m3fn
    return M


def model_args(M, **overrides):
    cfg = json.load(open(INFERENCE / "config.json"))
    cfg.update(overrides)
    return M.ModelArgs(**cfg)

