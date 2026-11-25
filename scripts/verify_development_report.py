"""Recompute a published paired development report on CPU."""

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/039-release-development"))

from report import call_curves, repair_scope, reproduce_analysis


def verify_table(path, rows):
    expected = [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(expected[0]) or list(reader) != expected:
            raise SystemExit(f"{path.name} differs from the recomputed table")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Published development result directory")
    directory = parser.parse_args().directory
    measurements = json.loads((directory / "measurements.json").read_text(encoding="utf-8"))
    if (measurements["passed"] is not True or measurements["final_test_opened"]
            or measurements["weights_public"] or measurements["optimizer_updates"]):
        raise SystemExit("Expected audited development measurements")
    design = measurements["design"]
    verdict = reproduce_analysis(measurements, design)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary != verdict:
        raise SystemExit("summary.json differs from the recomputed verdict")

    rows = sorted(measurements["rows"], key=lambda row: (row["task_number"], row["repetition"], row["arm"]))
    verify_table(directory / "matched-repeats.csv", rows)
    verify_table(directory / "call-curves.csv", call_curves(rows, design["evaluation_budget"]["max_model_calls"]))
    result = {
        "base_successes": verdict["successes"]["base"],
        "candidate_successes": verdict["successes"]["candidate"],
        "gain_pp": verdict["gain_pp"],
        "gain_interval_95_pp": verdict["gain_interval_95_pp"],
        "development_gate_passed": verdict["development_gate_passed"],
        "files_verified": ["summary.json", "matched-repeats.csv", "call-curves.csv"],
    }
    if any("multifile" in row for row in rows):
        result["repair_scope"] = repair_scope(rows)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
