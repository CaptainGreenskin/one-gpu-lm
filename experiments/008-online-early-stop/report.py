"""Audit the online choices and calculate seed-paired time savings and test differences."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from policy import EarlyStop
from run import HERE, SPEC, REFERENCES, digest, jobs, write


def read(p):
    return json.loads(p.read_bytes())


def audit(root):
    seal = read(root / "sealed.json")
    scores = read(root / "test-scores.json")
    expected = [f"{c}-{s}-{a}" for c, s, a in jobs()]
    if ([r["trial"] for r in seal["trials"]] != expected or set(scores["trials"]) != set(expected)
            or seal["protocol_sha256"] != digest(HERE / "protocol.json")
            or scores["seal_sha256"] != digest(root / "sealed.json")):
        raise ValueError("Incomplete run or inconsistent seal")
    runs, curves = [], {}
    for (corpus, seed, arm), record in zip(jobs(), seal["trials"]):
        key = record["trial"]
        directory = root / key
        for field in ("decision", "curve", "timing"):
            if digest(directory / f"{field}.json") != record[field + "_sha256"]:
                raise ValueError("Public training evidence does not match seal")
        choice = read(directory / "decision.json")
        curve = read(directory / "curve.json")
        timing = read(directory / "timing.json")["single_gpu_wall_seconds"]
        n = len(curve)
        if (choice["trained_epochs"] != n or choice["policy"] != arm or choice["seed"] != seed
                or not 0 < choice["update_seconds"] <= timing or choice["parameter_count"] != 85596576):
            raise ValueError(f"Invalid run: {key}")
        meta = read(REFERENCES[corpus])["splits"]["train"]
        rule = EarlyStop(SPEC["patience"])
        for i, r in enumerate(curve, 1):
            if (r["epoch"] != i or r["steps"] != i * math.ceil(meta["rows"] / 8)
                    or r["train_raw_bytes"] != i * meta["raw_bytes"]):
                raise ValueError("Invalid exposure")
            if arm == "early":
                _, stop = rule.observe(r["val_bpb"])
                if (r["best_epoch"] != rule.best_epoch or r["stale"] != rule.stale
                        or (stop or i == SPEC["max_epochs"]) != (i == n)):
                    raise ValueError("Decision cannot be reproduced using the observed prefix")
        selected = rule.best_epoch if arm == "early" else int(arm.removeprefix("fixed"))
        if selected != choice["selected_epoch"] or (arm != "early" and n != selected):
            raise ValueError("Wrong selected checkpoint")
        test = scores["trials"][key]
        docs = read(HERE / "test-references" / f"{corpus}.json")["documents"]
        observed = test["documents"]
        if ([(d["path"], d["raw_bytes"]) for d in observed] != [(d["path"], d["raw_bytes"]) for d in docs]
                or sum(d["raw_bytes"] for d in docs) != test["test_raw_bytes"]):
            raise ValueError("Test documents differ")
        recomputed = math.fsum(d["bpb"] * d["raw_bytes"] for d in observed) / test["test_raw_bytes"]
        if not math.isclose(recomputed, test["test_bpb"], rel_tol=1e-12):
            raise ValueError("Incorrect byte-weighted test score")
        runs.append({"corpus": corpus, "seed": seed, "policy": arm, "trained": n, "selected": selected,
                     "wall_s": timing, "updates_s": choice["update_seconds"], "test_bpb": test["test_bpb"]})
        curves[key] = curve
    for c in SPEC["corpora"]:
        for s in SPEC["seeds"]:
            control = curves[f"{c}-{s}-fixed24"]
            for arm in ("early", "fixed8"):
                short = curves[f"{c}-{s}-{arm}"]
                if [r["order_sha256"] for r in short] != [r["order_sha256"] for r in control[:len(short)]]:
                    raise ValueError("Paired runs used different sample orders")
    return runs, curves


def analyze(runs, curves):
    by_corpus = {}
    pairs = []
    for corpus in SPEC["corpora"]:
        group = [r for r in runs if r["corpus"] == corpus]
        result = {"policies": {}, "comparisons": {}}
        for arm in SPEC["policies"]:
            subset = [r for r in group if r["policy"] == arm]
            result["policies"][arm] = {k: mean(r[k] for r in subset) for k in ("wall_s", "updates_s", "test_bpb")}
        for base in ("fixed8", "fixed24"):
            subset = []
            for seed in SPEC["seeds"]:
                a, b = [next(r for r in group if r["seed"] == seed and r["policy"] == arm) for arm in ("early", base)]
                pair = {"corpus": corpus, "seed": seed, "baseline": base, "saved_s": b["wall_s"] - a["wall_s"],
                        "delta_bpb": a["test_bpb"] - b["test_bpb"]}
                subset.append(pair)
            pairs.extend(subset)
            deltas = [p["delta_bpb"] for p in subset]
            saved = mean(p["saved_s"] for p in subset)
            result["comparisons"][base] = {"mean_saved_s": saved, "saved_pct": 100 * saved / result["policies"][base]["wall_s"],
                "mean_delta_bpb": mean(deltas), "range_delta_bpb": [min(deltas), max(deltas)],
                "within_margin": sum(d <= SPEC["practical_margin_bpb"] for d in deltas), "seeds": len(deltas)}
        by_corpus[corpus] = result
    gaps = [abs(row["last_batch_loss"] - curves[f"{c}-{s}-fixed24"][i]["last_batch_loss"])
            for c in SPEC["corpora"] for s in SPEC["seeds"] for a in ("early","fixed8")
            for i, row in enumerate(curves[f"{c}-{s}-{a}"])]
    return {"corpora": by_corpus, "paired": pairs, "max_paired_prefix_last_batch_loss_gap": max(gaps),
            "actual_training_wall_seconds_all_18": sum(r["wall_s"] for r in runs)}


def table(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def plot(out, runs, curves, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), layout="constrained")
    colors = ["#2658a0", "#80538b", "#19806e"]
    for col, c in enumerate(SPEC["corpora"]):
        top, bottom = axes[:, col]
        for s, color in zip(SPEC["seeds"], colors):
            key = f"{c}-{s}-early"
            curve = curves[key]
            vals = [r["val_bpb"] for r in curve]
            best = curve[-1]["best_epoch"]
            top.plot(range(1, len(vals)+1), vals, color=color, label=f"Seed {s}")
            top.scatter(best, vals[best-1], marker="o", c=color, zorder=3)
            top.scatter(len(vals), vals[-1], marker="x", c=color, zorder=3)
        top.set(title=f"{c.title()}  /  validation seen by the policy", xlabel="Epoch", ylabel="Validation bits / byte")
        top.legend(fontsize=8, frameon=False)
        styles = {"fixed8": ("#737b85","s","Fixed 8"), "early": ("#13816c","o","Early stop"),
                  "fixed24": ("#bb602b","^","Fixed 24")}
        for arm, (color, marker, label) in styles.items():
            group = [r for r in runs if r["corpus"] == c and r["policy"] == arm]
            bottom.scatter([r["wall_s"] for r in group], [r["test_bpb"] for r in group], c=color, marker=marker, alpha=.45)
            avg = summary["corpora"][c]["policies"][arm]
            bottom.scatter(avg["wall_s"], avg["test_bpb"], c=color, marker=marker, s=88, edgecolor="black", linewidth=.65, label=label)
        bottom.set(title=f"{c.title()}  /  independent test", xlabel="Single-GPU wall time (s)", ylabel="Test bits / byte")
        bottom.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["right","top"]].set_visible(False)
    fig.suptitle("Online early stopping on Chinese technical documents\n"
                 "Top: circle = selected, cross = stopped     |     Bottom: outlined = three-seed mean", fontsize=13)
    fig.savefig(out / "early-stop.png", dpi=165)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    runs, curves = audit(args.run)
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    summary = analyze(runs, curves)
    write(out / "summary.json", summary)
    table(out / "runs.csv", runs)
    table(out / "paired.csv", summary["paired"])
    plot(out, runs, curves, summary)


if __name__ == "__main__":
    main()
