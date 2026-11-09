"""Score only submitted source files in a fresh, independently materialized task."""
from pathlib import Path
import shutil

from .tasks import file_manifest, prepare


def generated_files(repository):
    if repository == "jashkenas/underscore":
        return ("underscore-umd.js",)
    if repository == "epoberezkin/fast-deep-equal":
        return ("index.js", "react.js", "es6/index.js", "es6/react.js")
    return ()


def changed_protected(before, after, editable):
    return sorted(name for name, digest in before.items() if name not in editable and after.get(name) != digest)


def grade(task, workspace, snapshot, destination, original_manifest, make_env):
    try:
        after = file_manifest(workspace, exclude=generated_files(task["repository"]))
    except ValueError:
        return {"passed": False, "reason": "unsafe_workspace", "modified_source_files": []}
    forbidden = changed_protected(original_manifest, after, task["editable_files"])
    modified = [name for name in task["editable_files"] if after.get(name) != original_manifest.get(name)]
    if forbidden:
        return {"passed": False, "reason": "protected_file_tampering", "protected_files": forbidden,
                "modified_source_files": modified}
    if any(name not in after for name in task["editable_files"]):
        return {"passed": False, "reason": "missing_source", "modified_source_files": modified}
    fresh = prepare(snapshot, destination, task)
    for name in task["editable_files"]:
        shutil.copyfile(Path(workspace) / name, fresh / name)
    result = make_env(fresh).execute({"command": task["regression_command"]})
    (Path(destination).parent / (Path(destination).name + ".log")).write_text(result["output"])
    return {"passed": result["returncode"] == 0, "reason": result["reason"],
            "returncode": result["returncode"], "seconds": result["seconds"], "modified_source_files": modified,
            "source_sha256": {name: after[name] for name in task["editable_files"]},
            "ignored_new_files": sorted(set(after) - set(original_manifest))}
