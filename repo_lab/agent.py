"""Local mini-swe-agent adapter with explicit budgets and completed-step checkpoints."""
import argparse
import json
import os
from pathlib import Path
import re
import time

from training_lab.records import sha256, write_json
from .backend import request
from .environment import RepositoryEnvironment
from .grading import generated_files, grade
from .tasks import file_manifest, prepare


SYSTEM = """You repair code in an offline repository at /work. Inspect source and tests, edit allowed source files, then run the declared regression command. Tests, package metadata and runtime dependencies must not be changed. No network is available. Use exactly one fenced bash block per response, optionally preceded by one brief sentence. Every shell command starts in /work. Keep output small: use targeted file ranges and tests. You have at most 12 model calls, 512 generated tokens per call, and 8192 input tokens; do not print whole large files. Finish with:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""
INSTANCE = """Task: {{ task }}
Allowed source files: {{ editable_files }}
Regression command: {{ regression_command }}
Only changes to allowed source files will be submitted to a fresh grader. Generated build outputs are rebuilt there. You can read existing tests and documentation. Preserve public APIs and passing behavior.
"""


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, value)
    with tmp.open("rb") as stream:
        os.fsync(stream.fileno())
    tmp.replace(path)


def parse_action(text):
    blocks = re.findall(r"```(?:bash|sh)\s*\n(.*?)```", text, re.S)
    # Some Qwen checkpoints emit their native tool syntax even when asked for a fence.
    # Accept only one explicitly named bash/command action; never execute prose.
    native = re.findall(r"<tool_call>\s*<function=bash>\s*<parameter=command>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>", text, re.S)
    actions = blocks + native
    if len(actions) != 1 or not actions[0].strip():
        raise ValueError("Expected exactly one nonempty bash action")
    if native and (text.count('<tool_call>') != 1 or text.count('<parameter=') != 1):
        raise ValueError('Ambiguous native tool call')
    return {"command": actions[0].strip()}


class LocalModel:
    def __init__(self, socket, budget):
        self.socket, self.config = socket, dict(budget)
        self.generated_tokens = 0
        self.responses = []
        self.timeout = 180

    def query(self, messages):
        from minisweagent.exceptions import FormatError, LimitsExceeded
        remaining = self.config["max_total_generated_tokens"] - self.generated_tokens
        if remaining <= 0:
            raise LimitsExceeded(self.format_message(role="exit", content="TokenLimit", extra={"exit_status": "TokenLimit"}))
        value = request(self.socket, {"operation": "generate", "messages": messages,
            "max_new_tokens": min(remaining, self.config["max_generated_tokens_per_call"]),
            "max_input_tokens": self.config["max_input_tokens"]}, timeout=self.timeout)
        self.generated_tokens += value["generated_tokens"]
        self.responses.append({k: v for k, v in value.items() if k not in {"text", "ids"}})
        message = self.format_message(role="assistant", content=value["text"], extra={"generation": self.responses[-1], "cost": 0.0})
        try:
            message["extra"]["actions"] = [parse_action(value["text"])]
        except ValueError:
            raise FormatError(message, self.format_message(role="user", content="Format error: respond with exactly one fenced bash block containing your next command."))
        return message

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        from minisweagent.exceptions import Submitted
        output = outputs[0]
        observation = self.format_message(role="user", content=f"Exit code: {output['returncode']}\n{output['output']}",
                                          extra={"tool": {k: v for k, v in output.items() if k != "output"}})
        if output["returncode"] == 0 and message["extra"]["actions"][0]["command"] == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            raise Submitted(observation, self.format_message(role="exit", content="Submitted", extra={"exit_status": "Submitted", "submission": "workspace"}))
        return [observation]

    def get_template_vars(self, **kwargs):
        return kwargs

    def serialize(self):
        return {"local_model": {"config": self.config, "generated_tokens": self.generated_tokens, "responses": self.responses}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "tasks", "protocol", "socket", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--interrupt-after", type=int, default=0)
    args = p.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(out / "private-framework-config")
    os.environ["MSWEA_SILENT_STARTUP"] = "1"
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.exceptions import TimeExceeded
    task = next(t for t in json.loads(args.tasks.read_text())["tasks"] if t["id"] == args.id)
    protocol = json.loads(args.protocol.read_text())
    budget = protocol["agent"]
    runtime = json.loads((root / "runtime.json").read_text())
    slug = task["repository"].replace("/", "--")
    deps = root / "node-deps" / slug / "node_modules"
    snapshot = root / "sources" / slug
    workspace = out / "workspace"

    def make_env(work, output_chars=budget["max_tool_output_chars"]):
        return RepositoryEnvironment(work, python_site=root / ".venv-agent/lib/python3.10/site-packages",
            node_root=runtime["node_root"], node_modules=deps if deps.exists() else None,
            timeout=budget["max_tool_seconds"], output_chars=output_chars)

    identity = {"id": args.id, "task_spec_sha256": sha256(args.tasks), "protocol_sha256": sha256(args.protocol),
                "backend": request(args.socket, {"operation": "health"})}
    checkpoint = out / "checkpoint.json"
    model = LocalModel(args.socket, budget)
    if args.resume:
        state = json.loads(checkpoint.read_text())
        if state["identity"] != identity or state["boundary"] != "completed_step":
            raise ValueError("Checkpoint identity or completed-step boundary mismatch")
        if state["workspace_manifest"] != file_manifest(workspace):
            raise ValueError("Workspace changed since checkpoint")
        original = json.loads((out / "original.json").read_text())
    else:
        if checkpoint.exists():
            raise ValueError("Existing task requires explicit resume")
        prepare(snapshot, workspace, task)
        original = file_manifest(workspace, exclude=generated_files(task["repository"]))
        write_json(out / "original.json", original)
        state = None

    class CheckpointAgent(DefaultAgent):
        def step(self):
            left = budget["wall_time_limit_seconds"] - (time.time() - self._start_time)
            if left <= 0:
                raise TimeExceeded({"role": "exit", "content": "TimeExceeded", "extra": {"exit_status": "TimeExceeded"}})
            self.model.timeout = max(1, min(180, left))
            atomic_json(checkpoint, {"identity": identity, "boundary": "in_flight"})
            return super().step()

        def execute_actions(self, message):
            left = budget["wall_time_limit_seconds"] - (time.time() - self._start_time)
            if left <= 0:
                raise TimeExceeded({"role": "exit", "content": "TimeExceeded", "extra": {"exit_status": "TimeExceeded"}})
            self.env.timeout = max(1, min(budget["max_tool_seconds"], int(left)))
            return super().execute_actions(message)

        def save(self, path, *extra):
            data = super().save(path, *extra)
            atomic_json(checkpoint, {"identity": identity, "boundary": "completed_step",
                "workspace_manifest": file_manifest(workspace), "started_epoch": self._start_time,
                "n_calls": self.n_calls, "cost": self.cost, "format_errors": self.n_consecutive_format_errors,
                "extra_template_vars": self.extra_template_vars, "trajectory": data,
                "recovered": bool(args.resume)})
            if args.interrupt_after and self.n_calls == args.interrupt_after and self.messages[-1]["role"] != "exit":
                raise SystemExit(75)
            return data

    agent = CheckpointAgent(model, make_env(workspace), system_template=SYSTEM, instance_template=INSTANCE,
        step_limit=budget["max_model_calls"], cost_limit=0, wall_time_limit_seconds=budget["wall_time_limit_seconds"],
        output_path=out / "trajectory.json")
    run_kwargs = dict(task=task["description"], editable_files=", ".join(task["editable_files"]), regression_command=task["regression_command"])
    if state:
        # DefaultAgent.run initializes messages. Restore immediately after that initialization;
        # then use the unchanged upstream loop, query budgets and exception handling.
        old_add = agent.add_messages
        def restore_once(*messages):
            agent.add_messages = old_add
            agent.messages = state["trajectory"]["messages"]
            agent.n_calls, agent.cost = state["n_calls"], state["cost"]
            agent.n_consecutive_format_errors = state["format_errors"]
            agent._start_time = state["started_epoch"]
            agent.extra_template_vars = state["extra_template_vars"]
            model.generated_tokens = state["trajectory"]["local_model"]["generated_tokens"]
            model.responses = state["trajectory"]["local_model"]["responses"]
            agent.env.calls = state["trajectory"]["environment"]["tool_calls"]
            agent.env.seconds = state["trajectory"]["environment"]["tool_seconds"]
            return agent.messages
        agent.add_messages = restore_once
    try:
        if not state or state["trajectory"]["messages"][-1]["role"] != "exit":
            agent.run(**run_kwargs)
        else:
            restore_once()
    except Exception:
        # Record task failure and grade the submitted source; never silently rerun the task.
        pass
    score = grade(task, workspace, snapshot, out / "grader", original, lambda work: make_env(work, 100000))
    write_json(out / "result.json", {"schema_version": 1, "id": args.id, "repository": task["repository"],
        "split": task["split"], "family": task["family"], **score, "calls": agent.n_calls,
        "generated_tokens": model.generated_tokens, "seconds": time.time() - agent._start_time,
        "tool_calls": agent.env.calls, "tool_seconds": agent.env.seconds,
        "exit_status": agent.messages[-1].get("extra", {}).get("exit_status", "Unknown") if agent.messages else "Unknown",
        "generations": model.responses, "recovered": bool(args.resume)})


if __name__ == "__main__":
    main()
