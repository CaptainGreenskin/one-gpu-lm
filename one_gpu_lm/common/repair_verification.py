"""Observe declared regression checks without running tools or grading repairs."""

PREFIX = (
    "[Repair verification] {used} of {limit} model calls used; {remaining} remain including this request. "
    "Source bytes have changed, which does not establish correctness. {status} "
)
UNCHECKED = (
    "Run the declared regression command as a standalone command (optionally preceded by cd /work &&): {command} . "
    "Do not pipe it to tail/grep or append a successful command: that can hide a failing exit status. "
    "Read the actual failures, not just whether the shell command completed."
)
FAILED = (
    "Use the smallest failing assertion to identify the violated behavior. Check the surrounding callers and existing helpers; "
    "preserve adjacent cases instead of replacing one failure with another. If output is truncated, use one focused test or a small source range. "
    "Make a minimal evidence-based correction, then rerun the declared regression command without a pipe. "
    "Reserve remaining calls for the correction and verification; do not spend them only rereading or repeating an unchanged failing suite."
)
PASSED = (
    "The declared regression command exited zero on these source bytes. This is tool evidence, not the independent final grader. "
    "If the task requirements are satisfied, submit; avoid unrelated changes. Any further edit requires a new check."
)


def is_declared_check(command, regression_command):
    """Deliberately narrow: compound edits, pipes and redirected status are unknown."""
    return command.strip() in {regression_command.strip(), 'cd /work && ' + regression_command.strip()}


def prepare_verification(messages, source_steps, state, used, limit, regression_command, enabled):
    if not 0 <= used < limit or not regression_command.strip():
        raise ValueError('Verification requires a declared command and a remaining call')
    steps = {s['call_index']: s for s in source_steps}
    if len(steps) != len(source_steps):
        raise ValueError('Duplicate source observation')
    call = 0
    last_check = None
    last_tool_returncode = None
    for i, message in enumerate(messages):
        if message.get('role') != 'assistant':
            continue
        call += 1
        if i + 1 == len(messages) or 'tool' not in messages[i + 1].get('extra', {}):
            continue
        actions = message.get('extra', {}).get('actions', [])
        if len(actions) != 1 or set(actions[0]) != {'command'} or call not in steps:
            raise ValueError('Tool verification lacks action/source observation')
        tool = messages[i + 1]['extra']['tool']
        last_tool_returncode = tool['returncode']
        if is_declared_check(actions[0]['command'], regression_command):
            before = message['extra']['generation']['progress']
            after = steps[call]
            stable = (before['editable_state_sha256'] == after['editable_state_sha256']
                      and not before['invalid_editable_files'] and not after['invalid_editable_files'])
            last_check = {'call': call, 'returncode': tool['returncode'],
                          'state_sha256': after['editable_state_sha256'], 'stable': stable}
    if last_check is None:
        status = 'not_run'
    elif not last_check['stable']:
        status = 'changed_during_check'
    elif last_check['state_sha256'] != state['editable_state_sha256'] or state['invalid_editable_files']:
        status = 'stale'
    else:
        status = 'passed' if last_check['returncode'] == 0 else 'failed'
    notice = ''
    if enabled and state['changed_editable_files'] and not state['invalid_editable_files']:
        descriptions = {
            'not_run': 'No standalone declared regression check has been observed.',
            'stale': 'The last declared regression check predates the current source bytes.',
            'changed_during_check': 'Source changed during the declared check; its exit status does not verify a stable edit.',
            'failed': 'The latest declared regression check failed on the current source bytes.',
            'passed': '',
        }
        notice = PREFIX.format(used=used, limit=limit, remaining=limit-used, status=descriptions[status])
        if status == 'passed':
            notice += PASSED
        elif status == 'failed':
            notice += FAILED
        else:
            notice += UNCHECKED.format(command=regression_command)
            if last_tool_returncode not in {None, 0}:
                notice += ' The most recent tool command failed. ' + FAILED
    receipt = {'enabled': bool(enabled), 'completed_calls': used, 'remaining_calls': limit-used,
               'current_source_sha256': state['editable_state_sha256'], 'check_status': status,
               'last_declared_check': last_check, 'last_tool_returncode': last_tool_returncode,
               'reminder': bool(notice), 'reminder_chars': len(notice)}
    if not notice:
        return messages, receipt
    if not messages or messages[0]['role'] != 'system':
        raise ValueError('Verification requires a pinned system message')
    return [dict(messages[0], content=messages[0]['content']+'\n\n'+notice), *messages[1:]], receipt
