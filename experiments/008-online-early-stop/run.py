"""Run the 18 independently timed training jobs and seal their checkpoint choices."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = json.loads((HERE / "protocol.json").read_bytes())
REFERENCES = {"rust": HERE.parent / "005-technical-bytes/prepared-reference.json",
              "vue": HERE.parent / "007-document-curves/prepared-reference.json"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def jobs():
    arms = SPEC["policies"]
    for i, seed in enumerate(SPEC["seeds"]):
        for j, corpus in enumerate(SPEC["corpora"]):
            for k in range(len(arms)):
                yield corpus, seed, arms[(i + j + k) % len(arms)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rust", required=True, type=Path)
    p.add_argument("--vue", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    data = {"rust": args.rust, "vue": args.vue}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for corpus, directory in data.items():
        if (directory / "manifest.json").read_bytes() != REFERENCES[corpus].read_bytes():
            raise ValueError(f"Unexpected train/validation split: {corpus}")
        for split, meta in json.loads(REFERENCES[corpus].read_bytes())["splits"].items():
            for arm in ("byte", "bpe"):
                if digest(directory / f"{split}-{arm}.bin") != meta[f"{arm}_sha256"]:
                    raise ValueError("Encoded data hash mismatch")
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    with (out / "environment.json").open("w") as stream:
        subprocess.run([sys.executable, str(ROOT / "scripts/check_gpu.py")], env=env, stdout=stream, check=True)
    seals = []
    for corpus, seed, arm in jobs():
        key = f"{corpus}-{seed}-{arm}"
        final = out / key
        if not final.exists():
            with tempfile.TemporaryDirectory(prefix=".pending-", dir=out) as tmp:
                staging = Path(tmp) / key
                staging.mkdir()
                write(out / "status.json", {"state": "training", "trial": key, "finished": len(seals)})
                cmd = [sys.executable, str(HERE / "worker.py"), "--data", str(data[corpus]), "--out", str(staging),
                       "--seed", str(seed), "--policy", arm]
                start = time.monotonic()
                subprocess.run(cmd, env=env, check=True)
                wall = time.monotonic() - start
                write(staging / "timing.json", {"single_gpu_wall_seconds": wall})
                staging.rename(final)
        seals.append({"trial": key, **{k: digest(final / f) for k, f in [
            ("checkpoint_sha256", "checkpoint.pt"), ("decision_sha256", "decision.json"),
            ("curve_sha256", "curve.json"), ("timing_sha256", "timing.json")]}})
    seal = {"protocol_sha256": digest(HERE / "protocol.json"), "trials": seals}
    if (out / "sealed.json").exists() and json.loads((out / "sealed.json").read_bytes()) != seal:
        raise ValueError("Completed training differs from existing seal")
    write(out / "sealed.json", seal)
    write(out / "status.json", {"state": "sealed", "finished": len(seals)})


if __name__ == "__main__":
    main()
