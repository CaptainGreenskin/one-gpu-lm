"""Audit the three epoch curves and generate a summary table and learning-curve figure."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "005-technical-bytes"
SEEDS = (42, 43, 44)
EPOCHS = 24


def read(path):
    return json.loads(path.read_bytes())


def turning_points(curve):
    best = min(range(len(curve)), key=lambda i: curve[i]["val_bpb"])
    first_rise = None
    sustained = None
    streak = 0
    for i in range(best + 1, len(curve)):
        rises = curve[i]["val_bpb"] > curve[i - 1]["val_bpb"]
        if rises and first_rise is None:
            first_rise = curve[i]["epoch"]
        diverges = rises and curve[i]["train_bpb"] < curve[i - 1]["train_bpb"]
        streak = streak + 1 if diverges else 0
        if streak == 2:
            sustained = curve[i - 1]["epoch"]
            break
    return {"best_epoch": curve[best]["epoch"], "best_val_bpb": curve[best]["val_bpb"],
            "first_rise_epoch": first_rise, "two_diverging_epochs_start": sustained}


def load_run(run):
    manifest = PREVIOUS / "prepared-reference.json"
    reference = read(manifest)["splits"]
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    curves = {}
    anchors = []
    for seed in SEEDS:
        directory = run / f"seed-{seed}"
        metrics = read(directory / "metrics.json")
        curve = read(directory / "curve.json")
        if (metrics["device"] != "cuda" or metrics["arm"] != "bpe" or metrics["seed"] != seed
                or metrics["epochs"] != EPOCHS or metrics["manifest_sha256"] != manifest_hash
                or metrics["val_raw_bytes"] != reference["val"]["raw_bytes"] or len(curve) != EPOCHS):
            raise ValueError(f"Incomplete or mismatched run for seed {seed}")
        for epoch, row in enumerate(curve, 1):
            if (row["epoch"] != epoch or row["steps"] != epoch * math.ceil(reference["train"]["rows"] / 8)
                    or row["train_raw_bytes"] != epoch * reference["train"]["raw_bytes"]
                    or row["train_tokens"] != epoch * reference["train"]["bpe_tokens"]
                    or not all(math.isfinite(row[k]) for k in ("train_bpb", "val_bpb", "train_seconds"))):
                raise ValueError(f"Invalid observation for seed {seed}, epoch {epoch}")
            if epoch in (4, 24):
                old = read(PREVIOUS / f"results/2026-11-05/equal_bytes-{epoch}-{seed}-bpe-a1/metrics.json")
                if old["sample_order_sha256"] != row["sample_order_sha256"]:
                    raise ValueError(f"Sampling order differs from 005 for seed {seed}, epoch {epoch}")
                anchors.append({"seed": seed, "epoch": epoch, "005_bpb": old["bpb"],
                                "006_bpb": row["val_bpb"], "difference": row["val_bpb"] - old["bpb"]})
        if curve[-1]["val_bpb"] != metrics["bpb"] or curve[-1]["sample_order_sha256"] != metrics["sample_order_sha256"]:
            raise ValueError(f"Final epoch and completion record disagree for seed {seed}")
        curves[seed] = curve
    return curves, anchors


def plot(curves, averages, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in averages]
    best = turning_points(averages)["best_epoch"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    fig.suptitle("Chinese Rust Book  /  BPE learning curves", fontsize=16, fontweight="bold")
    for field, label, color in [("train_bpb", "Training chapters", "#106966"),
                                ("val_bpb", "Held-out chapters", "#b54b19")]:
        left.plot(epochs, [r[field] for r in averages], label=label, color=color, linewidth=2.6)
    left.set_title("Mean of three seeds", loc="left")
    for seed, color in zip(SEEDS, ("#35569c", "#976214", "#7b4a87")):
        series = [r["val_bpb"] for r in curves[seed]]
        best_i = series.index(min(series))
        right.plot(epochs, series, color=color, label=f"Seed {seed}", linewidth=1.6)
        right.scatter(epochs[best_i], series[best_i], color=color, s=27, zorder=3)
    right.set_title("Held-out chapters  /  dot = seed minimum", loc="left")
    left.scatter(best, averages[best-1]["val_bpb"], color="#b54b19", zorder=3)
    left.annotate(f"Best mean: epoch {best}", (best, averages[best-1]["val_bpb"]),
                  xytext=(10, -14), textcoords="offset points", fontsize=9, color="#944017")
    for ax in (left, right):
        ax.set(xlabel="Completed training epochs", ylabel="Bits per raw byte (lower is better)",
               xticks=[1, 4, 8, 12, 16, 20, 24])
        ax.grid(axis="y", alpha=0.23)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=165, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    curves, anchors = load_run(args.run)
    averages = [{"epoch": i + 1, **{k: mean(c[i][k] for c in curves.values())
                                     for k in ("train_bpb", "val_bpb", "train_seconds")}}
                for i in range(EPOCHS)]
    summary = {"mean_curve": averages, "mean_turning_points": turning_points(averages),
               "by_seed": {s: turning_points(c) for s, c in curves.items()}, "comparison_with_005": anchors}
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.out / "epochs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", *curves[SEEDS[0]][0]], lineterminator="\n")
        writer.writeheader()
        for seed, curve in curves.items():
            writer.writerows({"seed": seed, **r} for r in curve)
    plot(curves, averages, args.out / "learning-curve.png")


if __name__ == "__main__":
    main()
