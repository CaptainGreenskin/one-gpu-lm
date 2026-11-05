"""Plot observed static-cache decode speedups from the published timing samples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    fig, ax = plt.subplots(figsize=(8.1, 4.8), layout="constrained")
    labels, speedups = [], []
    for source in ("inference", "extended"):
        runs = json.loads((args.results / f"{source}.json").read_bytes())["runs"]
        for plain, cached in zip(runs[::2], runs[1::2]):
            assert not plain["cached"] and cached["cached"]
            assert (plain["batch"], plain["prompt_tokens"]) == (cached["batch"], cached["prompt_tokens"])
            speedups.append(cached["median"]["decode_tokens_s"] / plain["median"]["decode_tokens_s"])
            labels.append(f"B{plain['batch']}  /  {plain['prompt_tokens']} prompt")
    y = list(range(len(labels)))
    ax.barh(y, speedups, color=["#18776b" if v >= 1 else "#7890a8" for v in speedups])
    ax.axvline(1, color="#232a31", lw=1, ls="--")
    ax.axhline(3.5, color="#9ba6ae", lw=1, ls=":")
    for i, v in enumerate(speedups):
        ax.text(v - .025, i, f"{v:.2f}×", va="center", ha="right", color="white", fontsize=9, fontweight="bold")
    ax.set(yticks=y, yticklabels=labels, xlim=(0, max(speedups)*1.17),
           xlabel="Cached decode throughput / full-prefix throughput  (above 1 = faster)")
    ax.invert_yaxis()
    ax.set_title("85.6M GPT  /  static KV cache on RTX PRO 6000", loc="left", fontweight="bold")
    ax.spines[["right","top"]].set_visible(False)
    fig.savefig(args.out, dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    main()
