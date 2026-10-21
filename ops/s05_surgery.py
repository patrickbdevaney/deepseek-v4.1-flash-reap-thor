#!/usr/bin/env python3
"""s05: bit-exact MXFP4 surgery, streamed shard by shard to a PRIVATE HF repo (numpy only, no GPU).

For every source shard (sha256-checked against the pinned HF tree):
  * decoder-layer shards (L+3): keep the winning keep-set's experts, renumbered 0..n-1 in
    ascending original id, with their packed E2M1 bytes and E8M0 scales copied byte for byte;
    slice ffn.gate.weight rows and gate.bias / gate.bias_vl entries to the same ids; every other
    tensor (attention FP8 + scales, shared expert, HC, norms, compressor, indexer) copied as is;
  * vision/aligner (1), embedding (2), head (43) and the three DSpark/MTP shards (44-46,
    128 experts each, NOT pruned) are copied verbatim;
  * Engram shards 47/48 are NOT rehosted: the index keeps their entries and engram_pointer.json
    gives repo, revision, file names and sha256 so users fetch them from the base repo.
Then: write -> re-open and verify every kept tensor against the source bytes (a mismatch is fatal)
-> upload (HF_XET_FIXED_UPLOAD_CONCURRENCY=8; HF_XET_HIGH_PERFORMANCE is removed from the
environment because it OOMed this box during uploads) -> delete the local copy.

Configs: n_routed_experts set to the keep size in config.json (text_config) and
inference/config.json; the index is regenerated; LICENSE, tokenizer, encoding/ (DSML chat
encoding: 1017 ships no chat_template.jinja) and inference/ are copied. The MXFP4 master ships
UNHEALED: an E8M0 scale can only move by powers of two, so the s06 gain cannot be applied
losslessly here; it is folded into the NVFP4 global scales and the Thor requant instead.

The repo is created private and must still be private when each file is uploaded; s12 makes it
public only after the s10 release gates pass.

    python3 ops/s05_surgery.py run [--keepset $V41_WORK/s04/keepset.json] [--repo user/name] [--no-upload]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.lib import env, keepset, orchestrate, st_io  # noqa: E402

STAGE = "s05"
OUT = env.WORK / "s05"
TREE = env.SRC.parent / "v41_1017_tree.json"
BASE_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
N_SHARDS = 48
ENGRAM_SHARDS = (47, 48)


def shard_name(i: int) -> str:
    return f"model-{i:05d}-of-{N_SHARDS:05d}.safetensors"


def tree_sha() -> dict[str, dict]:
    t = json.load(open(TREE))
    return {x["path"]: {"sha256": x["lfs"]["oid"], "size": x["lfs"]["size"]} for x in t if x.get("type") == "file" and "lfs" in x}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def layer_of(i: int) -> int | None:
    L = i - 3
    return L if 0 <= L < 40 else None


EXPERT = __import__("re").compile(r"^(layers\.(\d+)\.ffn\.experts\.)(\d+)(\.w[123]\.(weight|scale))$")


def prune_tensors(rd: st_io.Reader, L: int, keep: list[int]) -> tuple[list, dict]:
    """-> ([(name, dtype, array)], {new name: source name}) for one decoder-layer shard."""
    remap = keepset.remap(keep)
    out, src_of = [], {}
    names = sorted(rd.names(), key=rd.offset)
    for n in names:
        m = EXPERT.match(n)
        if m:
            if int(m.group(2)) != L:
                raise SystemExit(f"shard for layer {L} holds {n}")
            old = int(m.group(3))
            if old not in remap:
                continue
            nn = f"{m.group(1)}{remap[old]}{m.group(4)}"
            out.append((nn, rd.dtype(n), rd.get(n)))
            src_of[nn] = n
        elif n == f"layers.{L}.ffn.gate.weight":
            out.append((n, rd.dtype(n), np.ascontiguousarray(rd.get(n)[keep])))
            src_of[n] = n
        elif n in (f"layers.{L}.ffn.gate.bias", f"layers.{L}.ffn.gate.bias_vl"):
            out.append((n, rd.dtype(n), np.ascontiguousarray(rd.get(n)[keep])))
            src_of[n] = n
        else:
            out.append((n, rd.dtype(n), rd.get(n)))
            src_of[n] = n
    return out, src_of


def verify_layer(src: st_io.Reader, dst: st_io.Reader, L: int, keep: list[int], src_of: dict) -> dict:
    """Byte-compare every written tensor with its source (sliced for the router). Fatal on any miss."""
    n_exp = {int(EXPERT.match(n).group(3)) for n in dst.names() if EXPERT.match(n)}
    if n_exp != set(range(len(keep))):
        raise SystemExit(f"L{L}: written expert ids {sorted(n_exp)[:5]}.. != 0..{len(keep) - 1}")
    for n in dst.names():
        s = src.get(src_of[n])
        if n.endswith(("ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl")):
            s = s[keep]
        if dst.dtype(n) != src.dtype(src_of[n]) or not np.array_equal(np.asarray(dst.get(n)).view(np.uint8),
                                                                      np.ascontiguousarray(s).view(np.uint8)):
            raise SystemExit(f"L{L}: {n} differs from source {src_of[n]} - surgery is not bit-exact")
    return {"tensors": len(dst.names()), "experts": len(n_exp)}


class Uploader:
    def __init__(self, repo: str | None, enabled: bool):
        self.enabled = enabled
        self.repo = repo
        if not enabled:
            return
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
        os.environ["HF_XET_FIXED_UPLOAD_CONCURRENCY"] = "8"
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.api.create_repo(repo, private=True, exist_ok=True, repo_type="model")

    def put(self, local: Path, path_in_repo: str) -> None:
        if not self.enabled:
            return
        if not self.api.model_info(self.repo).private:
            raise SystemExit(f"{self.repo} is PUBLIC before the s10 gates: refusing to upload {path_in_repo}")
        retry(lambda: self.api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo, repo_id=self.repo,
                                           repo_type="model", commit_message=f"s05: {path_in_repo}"), what=path_in_repo)


def retry(fn, what: str, tries: int = 5, wait: float = 60.0):
    """Transient HF/network errors (timeouts, 5xx, connection resets) must not STOP an unattended chain: back off and
    retry. Client errors (4xx other than 408/429) are re-raised at once - they will not heal by waiting."""
    import time
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classified below, re-raised when not transient
            code = getattr(getattr(e, "response", None), "status_code", None)
            transient = code is None or code >= 500 or code in (408, 429)
            if not transient or k == tries - 1:
                raise
            env.log(f"{what}: transient upload error ({type(e).__name__} {code}); retry {k + 1}/{tries - 1} in {wait:.0f}s", STAGE, "WARN")
            time.sleep(wait * (k + 1))


def engram_pointer(weight_map: dict, shas: dict) -> dict:
    return {"why": "the two Engram tables (98 GB each) and their small projection tensors are unchanged by pruning and "
                   "are not rehosted; fetch these two files from the base repo at this revision into this directory",
            "base_repo": BASE_REPO, "revision": env.REVISION,
            "files": [{"name": shard_name(i), **shas[shard_name(i)],
                       "tensors": sorted(k for k, v in weight_map.items() if v == shard_name(i))} for i in ENGRAM_SHARDS]}


def local_master_dir(tag: str) -> Path:
    """The local master: on the Seagate (NVMe space belongs to s06/s07/the Engram rows). Never served from it: chain2's
    GGUF convert reads it once, sequentially; s12 uploads it."""
    if not orchestrate.SEAGATE.is_dir() or not os.path.ismount(orchestrate.SEAGATE):
        raise SystemExit(f"s05: the Seagate is not mounted at {orchestrate.SEAGATE}; the local master needs ~152 GiB there")
    need = 160 << 30
    if orchestrate.disk_free(orchestrate.SEAGATE) < need + (40 << 30):
        raise SystemExit("s05: not enough space on the Seagate for the local master (~152 GiB + 40 GiB margin)")
    return orchestrate.SEAGATE / "v41_staging" / f"master_{tag}"


def default_repo(n_keep: int) -> str:
    from huggingface_hub import HfApi
    return f"{HfApi().whoami()['name']}/DeepSeek-V4.1-Flash-1017-REAP{n_keep}-MXFP4"


def check_arbitration(fp: str) -> None:
    """s04 decide keeps only finite per-token kl32 and sorts a NaN mean first, so a numerically broken arm could win.
    Refuse to cut a master from a keep-set unless every arbitrated arm was scored on the same, complete token set
    with a finite mean, and the keep-set is arm A's winner."""
    import math
    arb = json.load(open(env.WORK / "s04" / "arbitration.json"))
    n = {k: v["overall"]["kl32"].get("n_tokens") for k, v in arb["summaries"].items()}
    bad = {k: v["overall"]["kl32"]["mean"] for k, v in arb["summaries"].items() if not math.isfinite(v["overall"]["kl32"]["mean"])}
    if bad:
        raise SystemExit(f"s04 arbitration has non-finite kl32 means {bad}: re-check the replay before cutting a master")
    if len(set(n.values())) != 1 or arb.get("tokens") not in set(n.values()):
        raise SystemExit(f"s04 arbitration arms were scored on different token counts {n} (total {arb.get('tokens')})")
    if arb["winners"]["A"]["sha256"] != fp and arb["winners"].get("B", {}).get("sha256") != fp:
        raise SystemExit("keep-set is not an s04 arbitration winner")


def run(a):
    keep_all, fp = keepset.load(Path(a.keepset))
    check_arbitration(fp)
    n_keep = len(keep_all["0"])
    tag = f"REAP{n_keep}"
    led = env.Ledger(env.LOGS / f"s05_{tag}_ledger.json")
    if led.get("keepset") and led.get("keepset")["sha256"] != fp:
        keepset.require(fp, led.get("keepset")["sha256"], where="s05 ledger (a different keep-set was already surgered)")
    led.mark("keepset", sha256=fp, path=str(a.keepset))
    # The free HF account has a 100 GB PRIVATE quota and the master is ~163 GB, so by default nothing is uploaded here:
    # the master is kept on the Seagate (write-once staging, never served from) and s12 uploads it straight to a PUBLIC
    # repo only after its pre-registered gate passed. --upload restores the old private-repo flow (needs a PRO account).
    repo = a.repo or default_repo(n_keep)
    up = Uploader(repo, a.upload)
    shas = tree_sha()
    idx = json.load(open(env.SRC / "model.safetensors.index.json"))["weight_map"]
    stage = orchestrate.staging_dir(16 << 30, f"s05_{tag}", prefer=env.WORK)
    out_dir = OUT / tag if up.enabled else local_master_dir(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_map = {}
    for i in range(1, N_SHARDS + 1):
        name = shard_name(i)
        if i in ENGRAM_SHARDS:
            new_map.update({k: v for k, v in idx.items() if v == name})
            continue
        u = f"shard/{name}"
        if led.done(u):
            new_map.update(led.get(u)["weight_map"])
            continue
        src = env.SRC / name
        if not src.exists() and a.fetch_shards:
            from huggingface_hub import hf_hub_download
            hf_hub_download(BASE_REPO, name, revision=env.REVISION, local_dir=env.SRC)
        if not led.get(f"srcsha/{name}"):
            got = sha256_file(src)
            if got != shas[name]["sha256"]:
                raise SystemExit(f"{name}: sha256 {got} != pinned {shas[name]['sha256']}")
            led.finish(f"srcsha/{name}", sha256=got)
        dst = stage / name
        L = layer_of(i)
        srd = st_io.Reader(src)
        if L is None:
            shutil.copyfile(src, dst)
            info = {"verbatim": True}
            wm = {n: name for n in srd.names()}
        else:
            tensors, src_of = prune_tensors(srd, L, keep_all[str(L)])
            st_io.write(dst, tensors, metadata={"format": "pt"}, align=1)
            info = verify_layer(srd, st_io.Reader(dst), L, keep_all[str(L)], src_of)
            wm = {n: name for n, _, _ in tensors}
        dsha = sha256_file(dst)
        if L is None and dsha != shas[name]["sha256"]:
            raise SystemExit(f"{name}: verbatim copy sha mismatch")
        up.put(dst, name)
        size = dst.stat().st_size
        if up.enabled:
            dst.unlink()
        else:
            shutil.move(str(dst), out_dir / name)
            if sha256_file(out_dir / name) != dsha:          # USB/NTFS write: verify what landed, not what was sent
                (out_dir / name).unlink()
                raise SystemExit(f"{name}: sha256 of the copy on {out_dir} differs from the staged file")
        led.finish(u, sha256=dsha, bytes=size, weight_map=wm, uploaded=up.enabled, **info)
        new_map.update(wm)
        env.log(f"{name}: {size / 2**30:.2f} GiB sha {dsha[:16]} {'uploaded' if up.enabled else 'kept local'}", STAGE)
    finish_repo(a, led, up, keep_all, fp, n_keep, new_map, shas, stage, out_dir)


def finish_repo(a, led, up, keep_all, fp, n_keep, new_map, shas, stage, out_dir):
    total = sum(led.get(f"shard/{shard_name(i)}")["bytes"] for i in range(1, N_SHARDS + 1) if i not in ENGRAM_SHARDS)
    small = stage / "_small"
    if small.exists():
        shutil.rmtree(small)
    shutil.copytree(env.SRC, small, ignore=shutil.ignore_patterns("*.safetensors", ".cache", "__pycache__", "*.pyc", "model.safetensors.index.json"))
    env.atomic_json(small / "model.safetensors.index.json", {"metadata": {"total_size_present": total}, "weight_map": new_map})
    cfg = json.load(open(small / "config.json"))
    cfg.setdefault("text_config", {})["n_routed_experts"] = n_keep
    env.atomic_json(small / "config.json", cfg)
    icfg = json.load(open(small / "inference" / "config.json"))
    icfg["n_routed_experts"] = n_keep
    env.atomic_json(small / "inference" / "config.json", icfg)
    env.atomic_json(small / "engram_pointer.json", engram_pointer(new_map, shas))
    keepset.save(small / "reap_keepset.json", keep_all, base_repo=BASE_REPO, revision=env.REVISION)
    if (small / "README.md").exists():
        (small / "README.md").unlink()      # the base card would misdescribe this model; s12 writes ours
    files = sorted(p for p in small.rglob("*") if p.is_file())
    for p in files:
        rel = str(p.relative_to(small))
        if not led.done(f"small/{rel}"):
            up.put(p, rel)
            led.finish(f"small/{rel}", sha256=sha256_file(p))
    if not up.enabled:
        shutil.copytree(small, out_dir, dirs_exist_ok=True)
    shutil.rmtree(small)
    man = {"keepset_sha256": fp, "n_routed_experts": n_keep, "repo": up.repo, "private": True, "uploaded": up.enabled,
           "revision_base": env.REVISION, "healed": False,
           "files": {k[6:]: {"sha256": v["sha256"], "bytes": v["bytes"]} for k, v in led.data.items() if k.startswith("shard/")},
           "local_dir": None if up.enabled else str(out_dir), "ts": env.now()}
    env.atomic_json(OUT / f"manifest_REAP{n_keep}.json", man)
    led.finish("done", repo=up.repo)
    env.log(f"master REAP{n_keep} complete: {len(man['files'])} shards, {total / 2**30:.1f} GiB, {'uploaded to ' + up.repo + ' (private)' if up.enabled else 'kept at ' + str(out_dir) + ' for s12'}", STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--keepset", default=str(env.WORK / "s04" / "keepset.json"))
    ap.add_argument("--repo")
    ap.add_argument("--upload", action="store_true", help="upload to a PRIVATE repo as it goes (needs >163 GB private quota)")
    ap.add_argument("--no-upload", action="store_true", help="default; kept for compatibility")
    ap.add_argument("--fetch-shards", action="store_true")
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
