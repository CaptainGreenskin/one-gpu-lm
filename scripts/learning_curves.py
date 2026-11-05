"""Audit epoch observations and render learning curves for one or more training seeds."""
import csv
import hashlib
import json
import math
from statistics import mean

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


def audit_curves(run, manifest, seeds, epochs, device):
    reference = read(manifest)["splits"]
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    curves = {}
    for seed in seeds:
        directory = run / f"seed-{seed}"
        metrics = read(directory / "metrics.json")
        curve = read(directory / "curve.json")
        if (metrics["device"] != device or metrics["arm"] != "bpe" or metrics["seed"] != seed
                or metrics["epochs"] != epochs or metrics["manifest_sha256"] != manifest_hash
                or metrics["val_raw_bytes"] != reference["val"]["raw_bytes"] or len(curve) != epochs):
            raise ValueError(f"Incomplete or mismatched run for seed {seed}")
        for epoch, row in enumerate(curve, 1):
            if (row["epoch"] != epoch or row["steps"] != epoch * math.ceil(reference["train"]["rows"] / 8)
                    or row["train_raw_bytes"] != epoch * reference["train"]["raw_bytes"]
                    or row["train_tokens"] != epoch * reference["train"]["bpe_tokens"]
                    or not all(math.isfinite(row[k]) for k in ("train_bpb", "val_bpb", "train_seconds"))):
                raise ValueError(f"Invalid observation for seed {seed}, epoch {epoch}")
        if curve[-1]["val_bpb"] != metrics["bpb"] or curve[-1]["sample_order_sha256"] != metrics["sample_order_sha256"]:
            raise ValueError(f"Final epoch and completion record disagree for seed {seed}")
        curves[seed] = curve
    return curves


def plot(curves, averages, path, title="Chinese Rust Book", unit="chapters"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in averages]
    best = turning_points(averages)["best_epoch"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    fig.suptitle(f"{title}  /  BPE learning curves", fontsize=16, fontweight="bold")
    for field, label, color in [("train_bpb", f"Training {unit}", "#106966"),
                                ("val_bpb", f"Held-out {unit}", "#b54b19")]:
        left.plot(epochs, [r[field] for r in averages], label=label, color=color, linewidth=2.6)
    subtitle = "Mean of three seeds" if len(curves) == 3 else f"Average across seeds (n={len(curves)})"
    left.set_title(subtitle, loc="left")
    colors = ("#35569c", "#976214", "#7b4a87", "#247c75", "#b43d5a")
    for i, seed in enumerate(curves):
        color = colors[i % len(colors)]
        series = [r["val_bpb"] for r in curves[seed]]
        best_i = series.index(min(series))
        right.plot(epochs, series, color=color, label=f"Seed {seed}", linewidth=1.6)
        right.scatter(epochs[best_i], series[best_i], color=color, s=27, zorder=3)
    right.set_title(f"Held-out {unit}  /  dot = seed minimum", loc="left")
    left.scatter(best, averages[best-1]["val_bpb"], color="#b54b19", zorder=3)
    left.annotate(f"Best mean: epoch {best}", (best, averages[best-1]["val_bpb"]),
                  xytext=(10, -14), textcoords="offset points", fontsize=9, color="#944017")
    tick_step = max(1, epochs[-1] // 6)
    ticks = sorted({1, epochs[-1], *range(tick_step, epochs[-1], tick_step)})
    for ax in (left, right):
        ax.set(xlabel="Completed training epochs", ylabel="Bits per raw byte (lower is better)",
               xticks=ticks)
        ax.grid(axis="y", alpha=0.23)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=165, facecolor="white")
    plt.close(fig)


def write_report(curves, out, *, title="Chinese Rust Book", unit="chapters", extra=None):
    first_curve = next(iter(curves.values()))
    averages = [{"epoch": i + 1, **{k: mean(c[i][k] for c in curves.values())
                                     for k in ("train_bpb", "val_bpb", "train_seconds")}}
                for i in range(len(first_curve))]
    summary = {"mean_curve": averages, "mean_turning_points": turning_points(averages),
               "by_seed": {s: turning_points(c) for s, c in curves.items()}, **(extra or {})}
    out.mkdir(exist_ok=True, parents=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (out / "epochs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", *first_curve[0]], lineterminator="\n")
        writer.writeheader()
        for seed, curve in curves.items():
            writer.writerows({"seed": seed, **r} for r in curve)
    plot(curves, averages, out / "learning-curve.png", title, unit)
