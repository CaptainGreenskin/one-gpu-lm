"""Explicit formatting and bounded-history interventions; no task-specific rules."""
from .actions import SYSTEM

LEGACY_POLICY = {"format": "legacy", "context": "full"}
BASH_TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Run one shell command in the offline repository at /work.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "description": "The complete bash command to execute."}},
        "required": ["command"], "additionalProperties": False}}}]
NATIVE_EXAMPLE = "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"
NATIVE_SYSTEM = SYSTEM.replace(
    "Use exactly one fenced bash block per response, optionally preceded by one brief sentence.",
    "Use exactly one bash tool call per response, with the required command parameter. "
    "Close all tool-call, function and parameter tags. Optional brief prose may precede the call."
).replace("```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
          NATIVE_EXAMPLE.replace("pwd", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"))


def validate_policy(policy=None):
    policy = dict(LEGACY_POLICY if policy is None else policy)
    if set(policy) != {"format", "context"} or policy["format"] not in {"legacy", "native_schema"} or policy["context"] not in {"full", "bounded_history_v1"}:
        raise ValueError("Unknown agent policy")
    return policy


def system_prompt(policy):
    return NATIVE_SYSTEM if validate_policy(policy)["format"] == "native_schema" else SYSTEM


def format_feedback(policy):
    if validate_policy(policy)["format"] == "legacy":
        return "Format error: respond with exactly one fenced bash block containing your next command."
    return ("Format error: provide exactly one complete bash tool call with one required command parameter. "
            "Do not execute or repeat an incomplete call. Example syntax (choose your own next command):\n" + NATIVE_EXAMPLE)


def prepare_context(messages, encode, limit, policy):
    """Keep the full audit history elsewhere; pack only the next inference request.

    At >7168 tokens, drop oldest complete assistant/observation pairs until <=6144,
    retaining the system, entire initial task, and newest complete pair. Never cut
    commands, output text, or the task mid-message. A non-fitting pinned core fails.
    """
    if policy not in {"full", "bounded_history_v1"}:
        raise ValueError("Unknown context policy")
    tokens = encode(messages)
    info = {"policy": policy, "raw_input_tokens": len(tokens), "packed_input_tokens": len(tokens),
            "dropped_rounds": 0, "original_messages": len(messages), "packed_messages": len(messages)}
    if policy == "full" or len(tokens) <= min(7168, limit):
        return tokens, info
    if len(messages) < 2 or [m["role"] for m in messages[:2]] != ["system", "user"]:
        raise ValueError("Bounded context requires pinned system and initial task")
    rounds = messages[2:]
    if len(rounds) % 2 or any(m["role"] != ("assistant" if i % 2 == 0 else "user") for i, m in enumerate(rounds)):
        raise ValueError("Only complete assistant/observation pairs may be compacted")
    kept = list(rounds)
    dropped = 0
    while len(tokens) > min(6144, limit) and len(kept) > 2:
        kept = kept[2:]
        dropped += 1
        marker = (f"\n\n[Context budget: {dropped} earlier interaction rounds omitted from this request. "
                  "Repository files persist on disk. Re-read relevant files if earlier details are needed.]")
        packed = [dict(messages[0]), dict(messages[1], content=messages[1]["content"] + marker)] + kept
        tokens = encode(packed)
        info.update(dropped_rounds=dropped, packed_input_tokens=len(tokens), packed_messages=len(packed))
    return tokens, info
