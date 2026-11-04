"""Summarize completed pairs and the observed operating window; no GPU dependencies."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import median, mean


def read(path):
    return json.loads(path.read_text())


def describe(series):
    if not series:
        return None
    return {"min": min(series), "median": median(series), "max": max(series)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    directory = p.parse_args().run
    campaign = read(directory / "campaign.json")
    state = read(directory / "status.json")
    ref = read(Path(__file__).with_name("prepared-reference.json"))
    metrics = {}
    for f in sorted(directory.glob("t*-s*-*-a*/metrics.json")):
        m = read(f)
        key = (int(m["requested_seconds"]), m["seed"], m["arm"])
        expected = ref["splits"]["val"]
        expected_tokens = expected["raw_bytes"] if m["arm"] == "byte" else expected["bpe_tokens"]
        if (key in metrics or m["arm"] not in ("byte", "bpe") or m["device"] != "cuda"
                or m["manifest_sha256"] != campaign["manifest_sha256"]
                or m["val_raw_bytes"] != expected["raw_bytes"] or m["val_tokens"] != expected_tokens
                or not math.isfinite(m["bpb"]) or m["train_seconds"] < m["requested_seconds"]):
            raise ValueError(f"invalid or duplicate metrics in {f}")
        metrics[key] = m
    pairs = []
    for budget,seed in sorted({(k[0],k[1]) for k in metrics}):
        if all((budget,seed,arm) in metrics for arm in ("byte","bpe")):
            a,b = (metrics[budget,seed,arm] for arm in ("byte","bpe"))
            pairs.append({"seconds":budget,"seed":seed,"byte_bpb":a["bpb"],"bpe_bpb":b["bpb"],
                          "bpe_gain":a["bpb"]-b["bpb"],"byte_raw_Bps":a["train_bytes_per_second"],
                          "bpe_raw_Bps":b["train_bytes_per_second"]})
    groups = []
    for budget in sorted({pair["seconds"] for pair in pairs}):
        rows=[pair for pair in pairs if pair["seconds"]==budget]
        groups.append({"seconds":budget,"pairs":len(rows),"byte_mean_bpb":mean(r["byte_bpb"] for r in rows),
                       "bpe_mean_bpb":mean(r["bpe_bpb"] for r in rows),
                       "mean_gain":mean(r["bpe_gain"] for r in rows), "gains":describe([r["bpe_gain"] for r in rows])})
    with (directory/"telemetry.csv").open() as file:
        telemetry = list(csv.DictReader(file))
    times = [datetime.fromisoformat(r["utc"]).timestamp() for r in telemetry]
    fields = ("temperature.gpu","power.draw","utilization.gpu","memory.used","clocks.current.sm")
    series = {k:[] for k in fields}
    for row in telemetry:
        for k in fields:
            try: val = float(row[k])
            except (TypeError,ValueError): continue
            if math.isfinite(val): series[k].append(val)
    events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines() if line]
    out = {"status":state["status"],"started_utc":campaign["started_utc"],"deadline_utc":campaign["deadline_utc"],
           "finished_utc":state.get("finished_utc"),"successful_trials":len(metrics),"complete_pairs":len(pairs),
           "unpaired_successes":len(metrics)-2*len(pairs),
           "failed_attempts":sum(e["status"]=="failed" for e in events),
           "deadline_interruptions":sum(e["status"]=="deadline" for e in events),
           "completed_training_seconds":sum(m["train_seconds"] for m in metrics.values()),
           "telemetry_samples":len(times),"observed_span_seconds":times[-1]-times[0] if times else 0,
           "max_sample_gap_seconds":max([b-a for a,b in zip(times,times[1:])],default=0),
           "telemetry":{k:describe(v) for k,v in series.items()},"by_budget":groups,"pairs":pairs}
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
