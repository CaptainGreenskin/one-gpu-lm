"""Shared action syntax and prompts used by repository supervision and evaluation."""
import re


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
