"""Summarize the paired full-validation reports. Run on the GPU host or on the published JSON files."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

REFERENCE = Path(__file__).resolve().parents[1] / "002-first-train/wiki-zh-reference.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    args = parser.parse_args()
    ref = read(REFERENCE)
    val = ref["files"]["val.txt"]
    expected_targets = val["bytes"] - 3 * ref["val_articles"]  # two separators and one unscored first byte per article
    reference_counts = None
    rows = []
    for seed in (42, 43, 44):
        root = args.runs / f"seed-{seed}"
        train = read(root / "metrics.json")
        if (train["seed"], train["snapshot_step"], train["steps"], train["train_tokens"]) != (seed, 100, 1000, 4096000):
            raise ValueError("unexpected seed or training budget")
        if train["train_sha256"] != ref["files"]["train.txt"]["sha256"] or train["val_sha256"] != val["sha256"]:
            raise ValueError("unexpected training data")
        early, late = (read(root / f"eval-{s}.json") for s in (100, 1000))
        for ev in (early, late):
            counts = {a["id"]: a["targets"] for a in ev["articles"]}
            if reference_counts is None:
                reference_counts = counts
            if (ev["target_bytes"] != expected_targets or sum(counts.values()) != expected_targets
                    or len(counts) != ref["val_articles"] or len(ev["articles"]) != len(counts) or counts != reference_counts
                    or ev["val_sha256"] != val["sha256"] or (ev["context_bytes"], ev["stride"]) != (512, 256)):
                raise ValueError("inconsistent full-validation coverage")
            weighted = sum(a["targets"] * a["bits_per_byte"] for a in ev["articles"]) / expected_targets
            if not math.isclose(weighted, ev["bits_per_byte"], rel_tol=1e-12):
                raise ValueError("inconsistent full-validation score")
        early_by_id = {a["id"]: a["bits_per_byte"] for a in early["articles"]}
        rows.append({"seed": seed, "bpb_100": early["bits_per_byte"], "bpb_1000": late["bits_per_byte"],
                     "gain_bpb": early["bits_per_byte"] - late["bits_per_byte"],
                     "articles_improved": sum(a["bits_per_byte"] < early_by_id[a["id"]] for a in late["articles"]),
                     "first_100_seconds": train["snapshot_train_seconds"],
                     "next_900_seconds": train["train_wall_seconds"] - train["snapshot_train_seconds"]})
    print(json.dumps({"target_bytes": expected_targets, "articles": ref["val_articles"], "seeds": rows,
                      "mean_bpb_100": mean(r["bpb_100"] for r in rows), "mean_bpb_1000": mean(r["bpb_1000"] for r in rows),
                      "mean_gain_bpb": mean(r["gain_bpb"] for r in rows),
                      "mean_next_900_seconds": mean(r["next_900_seconds"] for r in rows)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
