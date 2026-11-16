"""Training-side feedback policy using only actor-visible state and test results."""
from pathlib import Path
import time

from .evaluation_audit import initial_messages as original_messages
from .submission import SubmissionGuard

ARMS = ('control', 'correction')
GUIDANCE = (
    '\nCollection procedure: once the visible evidence supports a repair hypothesis, make a minimal '
    'source edit early enough to leave time for testing and correction. After every source change, '
    'run the declared regression command by itself as your next tool command. If it fails, compare '
    'the actual failure with the changed source and revise the hypothesis; do not keep repeating '
    'unchanged probes. Keep commands concise and reserve calls for a final standalone test and '
    'submission. Stop when the task is fixed. Never introduce a deliberate bug or an unnecessary '
    'edit to produce a multi-edit example. Do not change tests or reference expected answers.'
)


def initial_messages(task, protocol, arm):
    if arm not in ARMS:
        raise ValueError('Unknown collection arm')
    messages = original_messages(task, protocol)
    if arm == 'correction':
        messages[0]['content'] += GUIDANCE
    return messages


def correction_rejection(regression):
    return ('Source changed. Collection policy requires the next tool command to be exactly the '
            'declared regression command, without a cd prefix or additional commands: '+regression+
            '\nUse the resulting feedback before making another edit. This rejected request was not executed.')


class CorrectionGuard(SubmissionGuard):
    def __init__(self, runner, actor_workspaces, arm):
        super().__init__(runner, actor_workspaces)
        if arm not in ARMS:
            raise ValueError('Unknown collection arm')
        self.arm = arm
        self.pending = set()

    def execute(self, workspace, command):
        started = time.monotonic(); key = str(Path(workspace).resolve())
        task = self.actors.get(key)
        if task is None:
            return super().execute(workspace, command)
        if self.arm == 'correction' and key in self.pending and command != task['regression_command']:
            return {'returncode': -2, 'reason': 'correction_requires_regression', 'executed': False,
                    'policy_event': 'rejected_non_regression_after_edit',
                    'output': correction_rejection(task['regression_command']),
                    'seconds': time.monotonic()-started}
        before = self.source_state(workspace, task)
        result = super().execute(workspace, command)
        after = self.source_state(workspace, task)
        if after != before:
            self.pending.add(key)
        elif command == task['regression_command'] and result.get('executed') is True:
            self.pending.discard(key)
        return result


def feedback_notice(arm, task, prior_steps, step):
    """Deterministic added text; never reads graders, reference code or task mutations."""
    if arm != 'correction':
        return ''
    observation = step.get('observation', {})
    if observation.get('executed') is not True:
        return ''
    before, after = step['source_before'], step.get('source_after')
    notice = []
    if after is not None and after != before:
        notice.append('Source changed. Next run the declared regression as a standalone command: '+task['regression_command'])
    elif step.get('command') == task['regression_command'] and observation['returncode'] != 0:
        notice.append('The visible regression failed. Use its concrete failure and the current source to revise your repair hypothesis. '
                      'A failing test is not a successful repair; repeating an unchanged probe does not fix it.')
    if step['call'] == 3 and not any(s.get('source_after') and s['source_after'] != s['source_before'] for s in [*prior_steps, step]):
        notice.append('Three calls used without a source change. If the evidence is sufficient, make a minimal justified edit and test it; leave budget for correction.')
    repeated = any(s.get('command') == step.get('command') and s['source_before'] == before
                   and s.get('source_after') == after for s in prior_steps)
    if repeated and step.get('command') != task['regression_command']:
        notice.append('This exact command has already been used on the same source state. Choose an action that can add new evidence or correct the source.')
    return '\nCOLLECTION FEEDBACK: '+' '.join(notice) if notice else ''


def correction_chain(trace, task):
    """Require an actual edited-source failure followed by another edit and success.

    An unmodified bug, malformed request, guard rejection, compound command,
    timeout, test tampering or a second edit without intervening failure cannot
    qualify. A source change alone is not a semantic proof of repair.
    """
    steps = trace['steps']; original = trace['original_source_sha256']
    edits = [s for s in steps if s.get('observation', {}).get('executed') is True
             and s.get('source_after') and s['source_after'] != s['source_before']]
    failures = []
    for step in steps:
        obs = step.get('observation', {}); current = step['source_before']
        if (step.get('command') == task['regression_command'] and obs.get('executed') is True
                and obs.get('returncode') != 0 and obs.get('reason') == 'runtime_error'
                and step.get('source_after') == current and current != original
                and all(v is not None for v in current.values())
                and any(e['call'] < step['call'] and e['source_after'] == current for e in edits)
                and any(e['call'] > step['call'] and e['source_after'] != current and e['source_after'] != original for e in edits)):
            failures.append(step['call'])
    protected = all(s.get('protected_before') is True and s.get('protected_after') is True
                    for s in steps if 'observation' in s)
    return {'edit_calls': [s['call'] for s in edits], 'qualifying_failure_calls': failures,
            'protected_files_intact_throughout': protected,
            'qualified_behavior': bool(trace.get('eligible_verified_trajectory') and len(edits) >= 2 and failures and protected),
            'first_failure_call': failures[0] if failures else None}
