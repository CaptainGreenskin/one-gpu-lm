"""Audit the Rust curves against experiment 005 and reproduce the 006 report."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "005-technical-bytes"
SEEDS = (42, 43, 44)
EPOCHS = 24


sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from learning_curves import audit_curves, plot, read, turning_points, write_report


def load_run(run):
    curves = audit_curves(run, PREVIOUS / "prepared-reference.json", SEEDS, EPOCHS, "cuda")
    anchors = []
    for seed, curve in curves.items():
        for epoch in (4, 24):
            row = curve[epoch - 1]
            old = read(PREVIOUS / f"results/2026-11-05/equal_bytes-{epoch}-{seed}-bpe-a1/metrics.json")
            if old["sample_order_sha256"] != row["sample_order_sha256"]:
                raise ValueError(f"Sampling order differs from 005 for seed {seed}, epoch {epoch}")
            anchors.append({"seed": seed, "epoch": epoch, "005_bpb": old["bpb"],
                            "006_bpb": row["val_bpb"], "difference": row["val_bpb"] - old["bpb"]})
    return curves, anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    curves, anchors = load_run(args.run)
    write_report(curves, args.out, extra={"comparison_with_005": anchors})


if __name__ == "__main__":
    main()
