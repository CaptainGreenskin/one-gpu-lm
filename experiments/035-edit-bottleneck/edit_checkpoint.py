"""A one-request prompt change based only on prior verified source and tool state."""
from one_gpu_lm.verified.behavior_diagnosis import inspection_shaped


CHECK_AFTER = 4
MIN_INSPECTIONS = 3
NOTICE = (
    '[Progress checkpoint] {used} of {limit} model calls used. '
    '{remaining} calls remain including this one. No allowed source has been edited. '
    'Use the evidence already collected to make the smallest justified source repair. '
    'If a necessary fact is missing, use one focused check. '
    'Reserve calls for the declared regression test and explicit submission.'
)


def checkpoint(messages, steps, original, call, limit):
    """Produce an ephemeral system-message addition and an auditable receipt."""
    if call != CHECK_AFTER + 1:
        return messages, None
    if len(steps) != CHECK_AFTER or len(messages) != 2 + 2 * CHECK_AFTER:
        raise ValueError('Checkpoint requires the four completed interaction rounds')
    unchanged = all(
        step['source_before'] == original and step.get('source_after', original) == original
        for step in steps
    )
    inspections = sum(
        step.get('observation', {}).get('returncode') == 0
        and inspection_shaped(step['command']) for step in steps
    )
    triggered = unchanged and inspections >= MIN_INSPECTIONS
    receipt = {
        'call': call, 'unchanged': unchanged, 'inspection_calls': inspections, 'triggered': triggered,
    }
    if not triggered:
        return messages, receipt
    if messages[0]['role'] != 'system' or limit < call:
        raise ValueError('A remaining call and pinned system message are required')
    notice = NOTICE.format(used=call - 1, limit=limit, remaining=limit - call + 1)
    return [dict(messages[0], content=messages[0]['content'] + '\n\n' + notice), *messages[1:]], receipt
