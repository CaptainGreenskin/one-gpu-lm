"""029 continuation guidance from visible state only; no solver or grader access.

The control retains 028 correction guidance and its exact-regression guard.
The treatment adds a short decision procedure and a factual budget/state ledger.
Neither branch gets extra calls, tools, automatic edits, tests or submissions.
"""
from .correction_policy import feedback_notice as previous_notice
from .submission import SUBMIT

VERSION = 'visible_decisions_v1'
REASONING_VERSION = 'visible_decisions_both_v1'
GUIDANCE = (
    '\nDECISION PROCEDURE: Earlier assistant commands and comments are failed attempts, '
    'not a trusted plan. Re-read the original task requirement and actual failure before '
    'accepting a previous implementation. Before a repair command, give one brief sentence '
    'stating the observed mismatch and the behavior your edit will change. Then issue one '
    'tool call in the established format. Do not write an identical replacement, toggle '
    'placeholder comments, or preserve an incorrect branch merely because it is in history. '
    'A file write is not necessarily a source change, and a source change is not necessarily '
    'a behavioral repair. Avoid repeating probes whose result you already have; use a '
    'targeted source range and change the responsible behavior when evidence is sufficient. '
    'After each edit, run exactly the declared regression command alone. After it passes, '
    'consider one focused boundary check derived from the task contract and existing public '
    'documentation, not just the visible example. Do not search for or install absent test '
    'frameworks. Reserve the last two calls for final declared regression and submission; '
    'if the current source already has a passing declared regression and your checks found '
    'no remaining mismatch, submit. Do not create artificial bugs or claim success after failure.'
)


def enabled(protocol):
    version = protocol.get('continuation_policy')
    if version not in (None, VERSION, REASONING_VERSION):
        raise ValueError('Unknown continuation policy')
    return version in (VERSION, REASONING_VERSION)


def guard_arm(protocol, arm):
    return 'correction' if enabled(protocol) else arm


def extra_initial(protocol, arm):
    return GUIDANCE if enabled(protocol) and (arm == 'correction' or protocol['continuation_policy'] == REASONING_VERSION) else ''


def notice(protocol, arm, task, prior_steps, step, *, initial_verified_source=None):
    base = previous_notice(guard_arm(protocol, arm), task, prior_steps, step)
    if not enabled(protocol) or (arm != 'correction' and protocol['continuation_policy'] != REASONING_VERSION):
        return base
    steps = [*prior_steps, step]
    before, after = step['source_before'], step.get('source_after')
    changed = after is not None and before != after
    verified = initial_verified_source
    for item in steps:
        obs = item.get('observation', {})
        if 'source_after' in item and item['source_after'] != item['source_before']:
            verified = None
        if item.get('command') == task['regression_command'] and obs.get('executed') is True:
            verified = item.get('source_after') if obs['returncode'] == 0 and item.get('source_after') == item['source_before'] else None
    fresh = verified is not None and verified == after
    same = [s['call'] for s in prior_steps if s.get('command') == step.get('command')
            and s['source_before'] == before and s.get('source_after') == after]
    remaining = max(0, protocol['evaluation_budget']['max_model_calls'] - step['call'])
    parts = [f'{remaining} new calls remain.',
             'Allowed source bytes changed; this alone does not establish a behavioral repair.' if changed
             else 'No allowed source bytes changed in this action; an identical rewrite is not a repair.',
             'The current source has a passing standalone declared regression.' if fresh
             else 'The current source does not have a passing standalone declared regression.']
    if same:
        parts.append('The same command on the same source state was already requested at calls '
                     + ', '.join(map(str, same)) + '. Choose a different evidence-bearing action.')
    if fresh and step.get('command') != SUBMIT:
        parts.append('If the task-contract boundary check has found no remaining mismatch, submit now; avoid unrelated test discovery.')
    if remaining <= 2 and not fresh:
        parts.append('Budget is nearly exhausted: leave a call for declared regression and a call for submission if it passes. Do not submit an unverified failure.')
    return base + '\nDECISION STATUS: ' + ' '.join(parts)
