"""Verify a completed RTX PRO 6000 calibration run; emit a shareable summary."""
import argparse
import json
import math
from pathlib import Path

import torch

from wiki_zh import REFERENCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run = parser.parse_args().run_dir
    env = json.loads((run / "environment.json").read_text())
    if env["bf16_attention_backward"] != "passed":
        raise ValueError("missing BF16 preflight")
    manifest = json.loads((run / "data-manifest.json").read_text())
    if manifest != json.loads(REFERENCE.read_text()):
        raise ValueError("unexpected data manifest")
    for label, steps in (("smoke", 12), ("train", 100)):
        result = json.loads((run / label / "metrics.json").read_text())
        if not (result["gpu"] == env["gpu"] and result["device"] == "cuda" and result["steps"] == steps):
            raise ValueError(f"unexpected device or step count in {label}")
        values = [result[k] for k in ("val_bpb_initial", "val_bpb_final", "train_tokens_per_second", "peak_allocated_bytes")]
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError(f"invalid metrics in {label}")
    if result["train_tokens"] != 409600 or result["seed"] != 42:
        raise ValueError("unexpected training budget")
    for split in ("train", "val"):
        if result[f"{split}_sha256"] != manifest["files"][f"{split}.txt"]["sha256"]:
            raise ValueError(f"trainer read the wrong {split} corpus")
    checkpoint = torch.load(run / "train/model.pt", map_location="cpu", weights_only=True, mmap=True)
    if checkpoint["config"] != [768, 12, 12, 512] or checkpoint["model"]["tokens.weight"].shape != (256, 768):
        raise ValueError("final checkpoint does not match the experiment")
    summary = {key: result[key] for key in ("git_commit", "gpu", "torch", "parameters", "steps", "train_tokens",
                                             "train_wall_seconds", "train_tokens_per_second", "peak_allocated_bytes",
                                             "val_bpb_initial", "val_bpb_final", "train_sha256", "val_sha256")}
    summary["checkpoint_readable"] = True
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: verified GPU calibration; summary: {run / 'summary.json'}")


if __name__ == "__main__":
    main()
