"""Collect explicit verification before submission, without executing on behalf of a model.

The wrapper only receives actor paths, allowed source paths and the actor-visible
regression command. Policy rejections are recorded as unexecuted requests. Hidden
graders use different paths and pass through unchanged. This is a collection
intervention, not proof that an accepted patch passes independent grading.
"""
from pathlib import Path
import time
from .tasks import digest

SUBMIT = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'


class SubmissionGuard:
    def __init__(self, runner, actor_workspaces):
        self.runner = runner
        self.actors = {str(Path(k).resolve()): dict(v) for k, v in actor_workspaces.items()}
        if any(set(v) != {'editable_files', 'regression_command'} for v in self.actors.values()):
            raise ValueError('Guard must not receive hidden task specifications')
        self.verified = {}

    @property
    def timeout(self):
        return self.runner.timeout

    @timeout.setter
    def timeout(self, value):
        self.runner.timeout = value

    @staticmethod
    def source_state(workspace, task):
        result = {}
        for name in task['editable_files']:
            path = Path(workspace) / name
            result[name] = digest(path.read_bytes()) if path.is_file() and not path.is_symlink() else None
        return result

    def execute(self, workspace, command):
        started = time.monotonic()
        key = str(Path(workspace).resolve())
        task = self.actors.get(key)
        if task is None:
            return dict(self.runner.execute(workspace, command), executed=True)
        before = self.source_state(workspace, task)
        if self.verified.get(key) != before:
            self.verified.pop(key, None)
        if command == SUBMIT and (not all(v is not None for v in before.values()) or self.verified.get(key) != before):
            return {
                'returncode': -2, 'reason': 'submission_requires_verification', 'executed': False,
                'policy_event': 'rejected_submit_without_fresh_exact_regression',
                'output': 'Submission rejected by the collection policy. Run exactly the following regression command '
                          'as the entire next bash tool call, without a cd prefix or additional commands: '
                          + task['regression_command'] + '\nAfter it passes on the unchanged source, submit again.',
                'seconds': time.monotonic() - started,
            }
        result = dict(self.runner.execute(workspace, command), executed=True)
        after = self.source_state(workspace, task)
        if after != before or (command == task['regression_command'] and result['returncode'] != 0):
            self.verified.pop(key, None)
        if command == task['regression_command'] and result['returncode'] == 0 and before == after and all(v is not None for v in after.values()):
            self.verified[key] = after
        return result
