"""Build auditable cards and a curve from an allowlisted, completed pilot receipt."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .records import sha256, write_json


def paired_summary(result, expected_tasks):
    if result["status"] != "completed":
        raise ValueError("Cannot make a completed model card from an incomplete run")
    base, adapted = result["baseline"], result["adapted"]
    if [r["id"] for r in base] != [r["id"] for r in adapted]:
        raise ValueError("Unpaired evaluation")
    if not base or len(base) != expected_tasks or len({r["id"] for r in base}) != len(base):
        raise ValueError("Incomplete or duplicate evaluation tasks")
    base_pass, adapted_pass = sum(r["passed"] for r in base), sum(r["passed"] for r in adapted)
    regressions = [a["id"] for b, a in zip(base, adapted) if b["passed"] and not a["passed"]]
    improvements = [a["id"] for b, a in zip(base, adapted) if not b["passed"] and a["passed"]]
    return {"schema_version": 1, "pairs": len(base), "baseline_passed": base_pass,
        "adapted_passed": adapted_pass, "regressions": regressions, "improvements": improvements,
        "quality_conclusion": "smoke_only_no_general_quality_claim", "regression_check_passed": not regressions,
        "daily_model_accepted": False,
        "rejection_reason": "heldout_regressions" if regressions else "smoke_scope_only"}


def render(run):
    read = lambda name: json.loads((run / name).read_text())
    result, profile, data = read("result.json"), read("profile.json"), read("data.json")
    summary = paired_summary(result, data["splits"]["test"]["records"])
    base, adapted = result["baseline"], result["adapted"]
    base_pass, adapted_pass = summary["baseline_passed"], summary["adapted_passed"]
    regressions, improvements = summary["regressions"], summary["improvements"]
    write_json(run / "summary.json", summary)
    rows = []
    for language in sorted({r["language"] for r in base}):
        b = [r for r in base if r["language"] == language]
        a = [r for r in adapted if r["language"] == language]
        rows.append(f"| {language} | {sum(r['passed'] for r in b)}/{len(b)} | {sum(r['passed'] for r in a)}/{len(a)} |")
    table = "| Language | Base | LoRA |\n| --- | --- | --- |\n" + "\n".join(rows)
    diagnostics = ""
    if (run / "control.json").exists():
        control = read("control.json")
        on = sum(t["adapter_enabled"]["passed"] for t in control["tasks"])
        off = sum(t["adapter_disabled"]["passed"] for t in control["tasks"])
        diagnostics = f"\nA separate saved-adapter on/off diagnostic rechecked {len(control['tasks'])} regressed tasks: enabled {on} passed; disabled {off} passed. It made no training updates. See [control.json](control.json) for exact output reproduction checks and its separate timing. This is a diagnosis of this run, not an independent final holdout.\n"
    report = f"""# Coding pilot result

This is a **pipeline smoke test**, not a coding benchmark or evidence of state-of-the-art ability.

- Base: `{profile['model_id']}` at `{profile['revision']}`.
- Actual UTC interval: {result['started_utc']} to {result['finished_utc']}; {result['wall_seconds']:.1f} s total.
- Parameters: {result['model_parameters']:,}; trainable: {result['trainable_parameters']:,}.
- Training: {result['training_steps']} updates, {result['training_seconds']:.1f} s, peak allocated {result['train_peak_allocated_bytes']/2**30:.2f} GiB.
- Sequence lengths actually used: {result['training_token_lengths']['min']}–{result['training_token_lengths']['max']}; profile maximum is only a limit, not tested context capacity.
- {data['splits']['train']['records']} private training tasks; {len(base)} held-out tasks; single seed.
- Same prompt, greedy decoding and generation budget before/after. A task passes only if all its cases pass in the offline sandbox.

{table}

Improvements: {len(improvements)}; regressions: {len(regressions)}. Small, simple fixtures can saturate and are not representative of repository work. This run does not establish a quality improvement and does not qualify the adapter as a daily-use model. Test results were not used for gradient updates, early stopping or choosing a checkpoint. Training uses the predeclared {profile['training']['epochs']} epochs.
{diagnostics}

Frozen parameter sample unchanged: `{result['frozen_sample_unchanged']}`; frozen gradients absent: `{result['frozen_gradients_absent']}`; adapter changed: `{result['adapter_changed']}`; save/reload greedy ID equality: `{result['adapter_reload_greedy_ids_equal']}`. The frozen sample check is not a full hash of all frozen parameters.

![Training loss](loss.png)

The curve mixes different training tasks and is not a validation/generalization curve. Cold compilation and startup costs are included where incurred; generation timings are end-to-end single requests, not separate TTFT/steady-state decode measurements. See `curve.json`, `baseline.json`, `adapted.json`, `environment.json`, `loading.json`, `source.json` and `telemetry.csv` for evidence. Input provenance and hashes are in [DATA_CARD.md](DATA_CARD.md); limitations and loading identity are in [MODEL_CARD.md](MODEL_CARD.md).

Rebuild these artifacts: `python -m training_lab.report --run <this-directory>`. Raw tasks, generated code and adapter weights remain private; hashes alone do not make this exact run independently reproducible. The public harness accepts the documented input contract so other users can run their own authorized tasks.
"""
    (run / "README.md").write_text(report)
    card = f"""# Model card draft — coding LoRA pipeline pilot

Status: **experimental, local weights only; not a published model release**. This document can accompany a later ModelScope release after its release checklist is satisfied. It is not a download page.

## Identity and intended use

Base: [{profile['model_id']}](https://huggingface.co/{profile['model_id']}/tree/{profile['revision']}); exact revision `{profile['revision']}`. Loader: `{profile['loader']}`; dtype `{profile['dtype']}`. Adapter: rank {profile['lora']['r']}, alpha {profile['lora']['lora_alpha']}; exact targets in `profile.json`. Base provenance and shard hashes are recorded in `download.json`; adapter content hashes in `adapter.json`.

Purpose: validate loading, frozen-backbone updates, isolated code evaluation and adapter round-trip on one GPU. It is not accepted for autonomous repository edits, long-running coding agents, production serving or general coding quality claims. Vision capability was not evaluated.

## Evaluation

Private held-out smoke tasks: base {base_pass}/{len(base)}, adapter {adapted_pass}/{len(base)}; {len(regressions)} regressions, {len(improvements)} improvements. Three languages, small self-authored function-level tasks, one seed. No SWE-bench, long-task, contamination, broader regression or robustness acceptance was completed. Training loss reduction alone is not evidence of better coding ability.

## Rights and redistribution

Original orchestration code is MIT. The base's pinned LICENSE is {profile['model_license']}; this does not audit upstream training data or warrant non-infringement. Adapter distribution requires the appropriate upstream license/notices and a separate review of data rights and actual packaging. No weight upload is authorized by this card. Private training/test text is not redistributed; see DATA_CARD.md and the run-specific audit for input and teacher provenance.

## Use and release requirements

The adapter must be paired with the exact base revision and tokenizer; it is not a standalone model. See the generic runner for `PeftModel.from_pretrained` loading. Before a public ModelScope version, record the repository owner, release version, exact base/adapter/config hashes, dependency lock, complete load command, license/notice files and a substantially stronger held-out evaluation. Publish a linked GitHub result for each released version. Do not merge weights or change quantization while retaining the same version identifier.
"""
    (run / "MODEL_CARD.md").write_text(card)
    dataset = ["# Data card — private pilot fixtures", "", "Purpose: pipeline validation, not representative coding training data. Raw records are not published.", ""]
    for name, split in data["splits"].items():
        dataset += [f"## {name}", "", f"- Records: {split['records']}; cases: {split['case_count']}; languages: `{json.dumps(split['languages'], sort_keys=True)}`.",
            f"- SHA-256 of exact JSONL bytes: `{split['sha256']}`.",
            f"- Provenance kinds: `{', '.join(split['provenance_kinds'])}`.",
            f"- Rights record: {'; '.join(split['rights_basis'])}.",
            f"- Redistribution allowed in the input manifest: `{split['redistribution_allowed']}`.", ""]
    dataset += ["Train/test IDs and exact prompts are disjoint. The generic validator does not prove semantic, translation, repository or task-family separation; any such claim requires a separate preparation audit. References were executed before evaluating the model. Existing model exposure to common algorithms cannot be ruled out. See the run-specific audit for authoring and teacher provenance. Synthetic material is not automatically a rights-cleared distillation dataset.", "", "The input contract requires a provenance kind, source description, affirmative training permission and rights basis. These fields record an operator's decision; software cannot determine legal ownership. Future datasets need per-source license/version/attribution records and repository/task-family splits, not just a dataset-level license label."]
    (run / "DATA_CARD.md").write_text("\n".join(dataset) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve = read("curve.json")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r["step"] for r in curve], [r["loss"] for r in curve], "o-", markersize=3)
    ax.set(xlabel="Optimizer step", ylabel="Assistant-token training loss", title=f"{profile['model_id'].split('/')[-1]} LoRA smoke: mixed-task loss")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(run / "loss.png", dpi=160, metadata={"Software": "one-gpu-lm training_lab"})
    plt.close(fig)
    # A publication allowlist, not a recursive directory upload. Never include private/.
    allowed = ["README.md", "MODEL_CARD.md", "DATA_CARD.md", "loss.png", "summary.json", "result.json",
               "profile.json", "source.json", "data.json", "references.json", "environment.json", "download.json",
               "loading.json", "baseline.json", "adapted.json", "curve.json", "adapter.json", "telemetry.csv", "events.jsonl", "status.json"]
    for name in ["runtimes.json", "code-identity.json", "audit.md", "hardware.json", "control.json", "validation.json"]:
        if (run / name).exists():
            allowed.append(name)
    for directory in sorted((run / "attempts").glob("*")):
        if not directory.is_dir():
            continue
        for name in ["result.json", "source.json", "status.json", "events.jsonl"]:
            relative = f"attempts/{directory.name}/{name}"
            if (run / relative).exists():
                allowed.append(relative)
    write_json(run / "artifacts.json", {"schema_version": 1, "weight_release": False,
        "report_generator_sha256": sha256(__file__),
        "files": {name: sha256(run / name) for name in allowed}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    render(parser.parse_args().run)
