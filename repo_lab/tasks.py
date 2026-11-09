"""Materialize private task specifications without exposing reference snapshots to the agent."""
import ast
import fnmatch
from pathlib import Path
import shutil

from training_lab.records import sha256


def apply_mutation(text, change):
    if change["kind"] == "replace":
        if text.count(change["before"]) != change.get("count", 1):
            raise ValueError("Mutation anchor is not unique at pinned source")
        return text.replace(change["before"], change["after"])
    if change["kind"] != "python_stub":
        raise ValueError("Unknown mutation kind")
    def find(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                if name == change["symbol"]:
                    return node
                found = find(node.body, name + ".")
                if found:
                    return found
        return None
    node = find(ast.parse(text).body)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("Missing target function")
    body = node.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        raise ValueError("Function has no implementation")
    lines = text.splitlines(keepends=True)
    lines[body[0].lineno - 1:body[-1].end_lineno] = [" " * body[0].col_offset + 'raise NotImplementedError("implementation missing")\n']
    return "".join(lines)


def copy_snapshot(snapshot, destination, repository):
    snapshot, destination = Path(snapshot), Path(destination)
    if destination.exists():
        raise ValueError("Workspace already exists")
    def ignore(directory, names):
        omit = {n for n in names if n in {".git", "__pycache__", "node_modules", ".source-identity.json"}}
        if repository == "jashkenas/underscore" and Path(directory) == snapshot:
            # Built distributions and source maps expose unmutated source answers.
            omit |= {n for n in names if n.startswith("underscore") or n in {"amd", "cjs", "docs", "index.html"}}
        return omit
    shutil.copytree(snapshot, destination, ignore=ignore)


def prepare(snapshot, destination, task, broken=True):
    copy_snapshot(snapshot, destination, task["repository"])
    root = Path(destination).resolve()
    for change in task["mutations"]:
        path = root / change["path"]
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError("Unsafe or missing source target")
        if broken:
            path.write_text(apply_mutation(path.read_text(), change))
    return root


def file_manifest(root, *, exclude=()):
    root = Path(root)
    result = {}
    for p in sorted(root.rglob("*")):
        relative = p.relative_to(root).as_posix()
        if any(part in {".git", "node_modules", "__pycache__", ".pytest_cache", ".hypothesis"} for part in p.relative_to(root).parts):
            continue
        if any(fnmatch.fnmatch(relative, pattern) for pattern in exclude):
            continue
        if p.is_symlink():
            raise ValueError("Workspace symlinks cannot be submitted")
        if p.is_file():
            result[relative] = sha256(p)
    return result
