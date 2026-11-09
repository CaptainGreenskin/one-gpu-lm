"""mini-swe-agent environment with per-command offline Linux namespaces."""
from pathlib import Path
import time

from one_gpu_lm.evaluation.sandbox import Sandbox


class RepositoryEnvironment:
    def __init__(self, workspace, *, python_site, node_root, node_modules=None, timeout=60, output_chars=2400):
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.output_chars = output_chars
        self.calls = 0
        self.seconds = 0.0
        self.config = {"timeout": timeout, "output_chars": output_chars, "network": False}
        mounts = [(python_site, "/opt/python-site"), (node_root, "/opt/node"),
                  (Path(__file__).resolve().parent, "/opt/repo-harness")]
        if node_modules is not None:
            (self.workspace / "node_modules").mkdir(exist_ok=True)
            mounts.append((node_modules, "/work/node_modules"))
        self.sandbox = Sandbox(readonly_mounts=mounts, virtual_memory_gib=16,
            environment={"PATH": "/opt/node/bin:/usr/bin:/bin", "PYTHONPATH": "/work:/work/src:/opt/python-site",
                         "PYTHONHASHSEED": "101", "PYTHONDONTWRITEBYTECODE": "1",
                         "NODE_OPTIONS": "--max-old-space-size=512", "CI": "1"})

    def execute(self, action, cwd=""):
        command = action.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("An action must contain a nonempty command")
        if cwd and cwd != "/work":
            raise ValueError("Use an explicit cd inside the isolated command")
        started = time.monotonic()
        result = self.sandbox.execute(self.workspace, ["/bin/bash", "-c", command], timeout=self.timeout)
        elapsed = time.monotonic() - started
        self.calls += 1
        self.seconds += elapsed
        output = result.get("stdout", "")
        if result.get("stderr"):
            output += "\nSTDERR:\n" + result["stderr"]
        if len(output) > self.output_chars:
            output = output[:self.output_chars] + "\n[tool output truncated; use smaller file ranges]"
        return {"output": output, "returncode": result.get("returncode", -1),
                "reason": result["reason"], "seconds": elapsed}

    def get_template_vars(self, **kwargs):
        return {"system": "Linux", "release": "isolated-runtime", "version": "pinned-dependencies",
                "machine": "x86_64", "cwd": "/work", **kwargs}

    def serialize(self):
        # Never include host paths in public-safe environment metadata.
        return {"environment": {"type": "bwrap-repository", "config": self.config,
                                "tool_calls": self.calls, "tool_seconds": self.seconds}}
