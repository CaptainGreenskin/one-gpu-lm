"""Common task prompts and narrowly scoped TAP timing normalization for 032."""
import hashlib
import re
from one_gpu_lm.common.agent_policy import system_prompt


def initial_messages(task, policy, budget):
    prompt = system_prompt(policy).replace('at most 12 model calls',
        f"at most {budget['max_model_calls']} model calls")
    prompt += ('\nRun the declared regression command as a separate command before submitting. '
               'A fresh protected workspace reruns the supplied public checks on your source changes.')
    instance = (f"Task: {task['description']}\nAllowed source files: {', '.join(task['editable_files'])}\n"
                f"Regression command: {task['regression_command']}\n"
                'Only allowed source changes are submitted to a fresh grader. Preserve public APIs and passing behavior.')
    return [{'role':'system','content':prompt},{'role':'user','content':instance}]


def tap_timings(text):
    """Change direct TAP diagnostic timing fields, never nested assertion data."""
    if not text.startswith('TAP version 13\n'):
        return text, 0
    lines = text.splitlines(keepends=True); count = 0; diagnostic_indent = None
    number = r'[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
    for i, line in enumerate(lines):
        match = re.fullmatch(r'( +)---\n', line)
        if match and i and re.match(r'\s*(?:not )?ok \d+\b', lines[i-1]):
            diagnostic_indent = len(match.group(1))
        if diagnostic_indent is not None:
            prefix = ' ' * diagnostic_indent
            if re.fullmatch(re.escape(prefix)+r'duration_ms: '+number+r'\n', line):
                lines[i] = prefix+'duration_ms: <TIME_MS>\n'; count += 1
            if line == prefix+'...\n':
                diagnostic_indent = None
        if re.fullmatch(r'# duration_ms '+number+r'\n?', line):
            # Node's aggregate metadata follows test-count summary comments.
            recent = ''.join(lines[max(0,i-9):i])
            if re.search(r'^# tests \d+$', recent, re.MULTILINE):
                lines[i] = '# duration_ms <TIME_MS>\n'; count += 1
    return ''.join(lines), count


def feedback(raw, normalizer, cap):
    prefix = raw if len(raw) <= cap else raw[:cap]+normalizer.MARKER
    normalized, old = normalizer.transform(prefix, 'extended', cap)
    delivered, count = tap_timings(normalized)
    if len(delivered) > cap:
        delivered = delivered[:cap]+normalizer.MARKER
    sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
    return delivered, {'policy':'024-extended-plus-direct-tap-timings-v1',
        'raw_output_sha256':sha(raw),'raw_output_chars':len(raw),'raw_prefix_clipped':len(raw)>cap,
        'legacy_receipt':old,'tap_timing_fields':count,'delivered_sha256':sha(delivered)}
