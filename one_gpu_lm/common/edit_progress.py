"""Observe editable files and construct a bounded, answer-free progress reminder."""
import hashlib
from pathlib import Path


REMINDER = (
    "[Execution progress] {used} of {limit} model calls used; {remaining} calls remain including this request. "
    "No allowed source file currently differs from its starting defective version. {repeat}"
    "Use the evidence already collected to make a minimal justified repair, then run the declared regression command and submit. "
    "If one detail is still missing, use one focused probe; reserve calls for editing and testing. "
    "Do not change unrelated code merely to satisfy this reminder."
)


def identical_command_streak(messages):
    commands = []
    for previous, current in zip(messages, messages[1:]):
        if 'tool' not in current.get('extra', {}):
            continue
        actions = previous.get('extra', {}).get('actions', [])
        if previous.get('role') != 'assistant' or len(actions) != 1 or set(actions[0]) != {'command'}:
            raise ValueError('Execution observation lacks one recorded action')
        commands.append(actions[0]['command'])
    streak = 0
    for command in reversed(commands):
        if command != commands[-1]:
            break
        streak += 1
    return streak


class EditableProgress:
    def __init__(self, workspace, original, editable):
        self.workspace = Path(workspace).resolve()
        self.original = {name: original[name] for name in editable}

    def snapshot(self):
        changed = invalid = 0
        states = []
        for name, original_hash in sorted(self.original.items()):
            path = self.workspace / name
            # Never follow a model-created symlink outside the allowed workspace.
            try:
                digest = None if path.is_symlink() or not path.resolve().is_relative_to(self.workspace) or not path.is_file() else hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, RuntimeError):
                digest = None
            invalid += digest is None
            changed += digest is not None and digest != original_hash
            states.append((name, digest))
        import json
        return {'changed_editable_files': changed, 'invalid_editable_files': invalid,
                'editable_state_sha256': hashlib.sha256(json.dumps(states, separators=(',', ':')).encode()).hexdigest()}

    def prepare_request(self, messages, used, limit, enabled):
        if used < 0 or used >= limit:
            raise ValueError('Progress observation requires a remaining model call')
        state = self.snapshot()
        streak = identical_command_streak(messages)
        trigger = state['changed_editable_files'] == 0 and (used >= 4 or streak >= 2)
        notice = ''
        if enabled and trigger:
            repeat = f'The last {streak} executed commands were identical. Do not repeat that same probe without a specific reason. ' if streak >= 2 else ''
            notice = REMINDER.format(used=used, limit=limit, remaining=limit-used, repeat=repeat)
        receipt = dict(state, enabled=bool(enabled), completed_calls=used, remaining_calls=limit-used,
                       identical_command_streak=streak, reminder=bool(notice), reminder_chars=len(notice))
        if not notice:
            return messages, receipt
        if not messages or messages[0]['role'] != 'system':
            raise ValueError('Progress reminder requires a pinned system message')
        return [dict(messages[0], content=messages[0]['content']+'\n\n'+notice), *messages[1:]], receipt
