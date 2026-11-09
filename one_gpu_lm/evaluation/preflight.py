"""Admit tasks only after clean, broken and restored controls, before model rollouts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from one_gpu_lm.common.receipts import sha256, write_json
from .environment import RepositoryEnvironment
from .tasks import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--clean-only", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    if args.out.exists():
        raise ValueError("Keep previous admission receipts and choose a new output")
    args.out.mkdir(parents=True)
    private = args.out / "private"
    private.mkdir()
    runtime = json.loads((root / "runtime.json").read_text())
    tasks = json.loads(args.tasks.read_text())["tasks"]
    value = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
             "task_spec_sha256": sha256(args.tasks), "clean": [], "tasks": [], "complete": False}

    def save(stage, **extra):
        write_json(args.out / "status.json", {"stage": stage, **extra})
        write_json(args.out / "admission.json", value)
        print(stage, json.dumps(extra), flush=True)

    def run(work, task, label):
        slug = task["repository"].replace("/", "--")
        deps = root / "node-deps" / slug / "node_modules"
        env = RepositoryEnvironment(work, python_site=root / ".venv-agent/lib/python3.10/site-packages",
            node_root=runtime["node_root"], node_modules=deps if deps.exists() else None, output_chars=100000)
        result = env.execute({"command": task["regression_command"]})
        (private / (label + ".log")).write_text(result["output"])
        return {"passed": result["returncode"] == 0, "returncode": result["returncode"],
                "seconds": result["seconds"], "reason": result["reason"]}

    grouped = {}
    for task in tasks:
        grouped.setdefault(task["repository"], []).append(task)
    for repository, group in grouped.items():
        first = group[0]
        slug = repository.replace("/", "--")
        snapshot = root / "sources" / slug
        clean = private / (slug + "-clean")
        prepare(snapshot, clean, first, broken=False)
        save("clean_control", repository=repository)
        control = run(clean, first, slug + "-clean")
        value["clean"].append({"repository": repository, "revision": first["revision"], **control})
        save("clean_control_finished", repository=repository, passed=control["passed"])
        if not control["passed"] or args.clean_only:
            continue
        for task in group:
            save("task_control", id=task["id"])
            work = private / task["id"]
            prepare(snapshot, work, task)
            before = {name: sha256(work / name) for name in task["editable_files"]}
            broken = run(work, task, task["id"] + "-broken")
            for name in task["editable_files"]:
                shutil.copyfile(snapshot / name, work / name)
            after = {name: sha256(work / name) for name in task["editable_files"]}
            restored = run(work, task, task["id"] + "-restored")
            # A timeout alone is not an informative negative test; require an actual nonzero test exit.
            killed = not broken["passed"] and broken["returncode"] not in {-1, 5}
            accepted = killed and restored["passed"] and all(before[k] != after[k] for k in before)
            value["tasks"].append({"id": task["id"], "repository": repository, "split": task["split"],
                "family": task["family"], "source_files": task["editable_files"], "broken_sha256": before,
                "reference_sha256": after, "broken": broken, "restored": restored, "admitted": accepted})
            save("task_control_finished", id=task["id"], admitted=accepted)
    value.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                 admitted=sum(t["admitted"] for t in value["tasks"]))
    save("completed", admitted=value["admitted"], clean_passed=sum(r["passed"] for r in value["clean"]))


if __name__ == "__main__":
    main()
