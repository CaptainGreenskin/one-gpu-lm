"""Prepare pinned direct dependencies outside agent workspaces, with lifecycle scripts disabled."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from training_lab.records import sha256, write_json


DEPENDENCIES = {
    "jashkenas--underscore": {"qunit": "2.10.1", "rollup": "2.79.2"},
    "epoberezkin--fast-deep-equal": {"dot": "1.1.3", "mocha": "7.2.0", "sinon": "9.2.4",
                                   "react": "16.14.0", "react-test-renderer": "16.14.0"},
    "sindresorhus--p-limit": {"yocto-queue": "1.2.2", "ava": "6.4.1", "in-range": "3.0.0",
                            "random-int": "3.0.0", "time-span": "5.1.0"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ, PATH=str(args.node_root.resolve() / "bin") + os.pathsep + os.environ.get("PATH", ""))
    receipts = []
    for repository, dependencies in DEPENDENCIES.items():
        target = args.out / repository
        target.mkdir(parents=True, exist_ok=True)
        manifest = {"name": "repository-runtime-dependencies", "private": True, "version": "0.0.0", "dependencies": dependencies}
        lock_path = target / "package-lock.json"
        locked = json.loads(lock_path.read_text()) if lock_path.exists() else {}
        compatible = locked.get("packages", {}).get("", {}).get("dependencies") == dependencies
        write_json(target / "package.json", manifest)
        if compatible:
            command = [str(args.node_root.resolve() / "bin/npm"), "ci", "--ignore-scripts", "--no-audit", "--no-fund"]
        else:
            command = [str(args.node_root.resolve() / "bin/npm"), "install", "--ignore-scripts", "--no-audit", "--no-fund"]
        with (target / "install.log").open("w") as log:
            subprocess.run(command, cwd=target, env=env, stdout=log, stderr=log, check=True, timeout=300)
        lock = json.loads((target / "package-lock.json").read_text())
        receipts.append({"repository": repository, "direct": dependencies, "lock_sha256": sha256(target / "package-lock.json"),
                         "packages": {k: {f: v[f] for f in ["version", "integrity", "license"] if f in v}
                                      for k, v in lock.get("packages", {}).items() if k}})
        write_json(args.out / "receipts.json", {"schema_version": 1, "install_scripts": False, "repositories": receipts})
        print(repository, "dependencies ready", flush=True)


if __name__ == "__main__":
    main()
