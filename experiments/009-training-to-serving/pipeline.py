"""Run the data-to-serving systems pipeline, or adopt an already evaluated 008 training run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from runtime import HERE

PREV = HERE.parent / "008-online-early-stop"
sys.path.insert(0, str(PREV))
from report import audit
from run import REFERENCES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    p.add_argument("--from-run", type=Path, help="reuse the weights and timings of a sealed and tested 008 run")
    p.add_argument("--data", type=Path, help="prepared 008 inputs, optional for a new training run")
    p.add_argument("--seven-b", action="store_true", help="also run the separate 7B optimizer-step probes")
    args = p.parse_args()
    if args.from_run and args.data:
        p.error("choose an evaluated run or training data")
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    def call(script, *parts):
        subprocess.run([sys.executable, str(script), *map(str, parts)], env=env, check=True)
    run = args.from_run
    if run is None:
        data = args.data or args.out / "inputs"
        if args.data is None:
            call(PREV / "prepare_data.py", "--out", data)
        run = args.out / "training"
        call(PREV / "run.py", "--rust", data/"trainval/rust", "--vue", data/"trainval/vue", "--out", run)
        call(PREV / "evaluate.py", "--run", run, "--test", data/"test")
    trials, _ = audit(run)
    perf = []
    for name, reference in REFERENCES.items():
        per_epoch = json.loads(reference.read_bytes())["splits"]["train"]["bpe_tokens"]
        for arm in ("early", "fixed8", "fixed24"):
            rows = [r for r in trials if r["corpus"] == name and r["policy"] == arm]
            tokens = sum(r["trained"] * per_epoch for r in rows)
            perf.append({"corpus": name, "policy": arm, "seeds": len(rows), "processed_bpe_tokens": tokens,
                         "update_tokens_s": tokens/sum(r["updates_s"] for r in rows),
                         "wall_tokens_s": tokens/sum(r["wall_s"] for r in rows)})
    (args.out/"training-perf.json").write_text(json.dumps(perf, indent=2) + "\n")
    bundle = args.out / "bundle"
    call(HERE/"export.py", "--run", run, "--out", bundle)
    call(HERE/"bench.py", "--bundle", bundle, "--out", args.out/"inference.json")
    call(HERE/"smoke_http.py", "--bundle", bundle, "--device", "cuda", "--out", args.out/"http.json")
    if args.seven_b:
        for mode in ("lora-adamw", "full-adafactor"):
            call(HERE/"seven_b.py", "--mode", mode, "--out", args.out/(mode+".json"))


if __name__ == "__main__":
    main()
