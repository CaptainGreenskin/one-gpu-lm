"""Replay the same training repair after two executed source inspections."""

from shlex import quote

from one_gpu_lm.verified.tasks import relative


def with_read_history(case, actions, budget):
    task = case.task
    if task['split'] != 'training':
        raise ValueError('Use only approved training sources')
    if not actions or actions[0]['kind'] != 'focused_inspection':
        raise ValueError('Require the original focused inspection and repair plan')
    if len(actions) + 2 > budget['max_model_calls']:
        raise ValueError('read_history_exceeds_call_budget')
    paths = task['editable_files']
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Require the source files listed in the task opening')
    arguments = ' '.join(quote(str(relative(path))) for path in paths)
    return [
        {'kind': 'source_line_counts', 'command': f'wc -l -- {arguments}', 'expected': 'ok'},
        {'kind': 'source_headers', 'command': f'head -n 12 -- {arguments}', 'expected': 'ok'},
        *actions,
    ]
