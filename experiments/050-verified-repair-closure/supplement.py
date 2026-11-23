"""Replay oversized reference hunks as bounded, sequential line-range edits."""

import argparse
from difflib import SequenceMatcher
from pathlib import Path
import shlex

from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.verified.historic_plan import native
from one_gpu_lm.verified.submission import SUBMIT
from run import require, run


def command(path, edits):
    # The quoted delimiter and repr keep source text literal.
    return (
        "python3 - <<'REPAIR_EDIT'\n"
        "from pathlib import Path\n"
        f"p = Path({path!r})\n"
        "lines = p.read_bytes().decode('utf-8').splitlines(keepends=True)\n"
        f"edits = {edits!r}\n"
        "for start, stop, text in edits:\n"
        "    assert 0 <= start <= stop <= len(lines)\n"
        "    lines[start:stop] = text.splitlines(keepends=True)\n"
        "p.write_bytes(''.join(lines).encode('utf-8'))\n"
        "REPAIR_EDIT"
    )


def bounded_edits(path, parent, fixed, fits):
    """Apply windows bottom-up, inserting a split replacement tail-first."""
    old = parent.splitlines(keepends=True)
    new = fixed.splitlines(keepends=True)
    opcodes = SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes()
    operations = []

    def accepts(edits):
        return fits(command(path, edits))

    for kind, start, stop, left, right in reversed(opcodes):
        if kind == 'equal':
            continue
        if left == right:
            operations.append((start, stop, ''))
            continue
        while left < right:
            whole = (start, stop, ''.join(new[left:right]))
            if accepts([whole]):
                operations.append(whole)
                break
            lower, upper = 1, right - left - 1
            kept = 0
            while lower <= upper:
                count = (lower + upper) // 2
                attempt = (start, stop, ''.join(new[right - count:right]))
                if accepts([attempt]):
                    kept = count
                    lower = count + 1
                else:
                    upper = count - 1
            require(kept > 0, 'reference_line_over_budget')
            operations.append((start, stop, ''.join(new[right - kept:right])))
            right -= kept
            stop = start

    group = []
    for edit in operations:
        if group and not accepts(group + [edit]):
            yield command(path, group)
            group = []
        require(accepts([edit]), 'reference_line_over_budget')
        group.append(edit)
    if group:
        yield command(path, group)


def make_plan(case, messages, tokenizer, profile, budget):
    def labels(shell):
        record = {'messages': messages, 'target': native(shell)}
        return encode_training_example(record, tokenizer, profile)['supervised_tokens']

    def fits(shell):
        return labels(shell) <= budget['max_generated_tokens_per_call']

    task = case.task
    mutations = {row['path']: row for row in task['mutations']}
    necessary = sorted(case.admission['essential_files'])
    order = necessary + [name for name in task['editable_files'] if name not in necessary]
    require(set(order) == set(mutations), 'source_file_inventory')
    inspection = []
    actions = [{'kind': 'focused_inspection', 'expected': 'ok'}]
    for index, name in enumerate(order):
        mutation = mutations[name]
        old = mutation['after'].splitlines(keepends=True)
        fixed = mutation['before'].splitlines(keepends=True)
        first = next(item[1] for item in SequenceMatcher(a=old, b=fixed, autojunk=False).get_opcodes()
                     if item[0] != 'equal')
        begin = max(1, first - 2)
        end = max(begin, first + 3)
        inspection.append('sed -n ' + shlex.quote(f'{begin},{end}{{=;p;}}') + ' ' + shlex.quote(name))
        edits = bounded_edits(name, mutation['after'], mutation['before'], fits)
        actions.extend({'kind': 'reference_edit', 'expected': 'ok', 'source_file': name, 'command': edit}
                       for edit in edits)
        if case.admission['multifile'] and index == 0:
            actions.append({'kind': 'partial_repair_regression', 'command': task['regression_command'],
                            'expected': 'semantic_failure'})
    actions[0]['command'] = '\n'.join(inspection)
    actions.extend([{'kind': 'final_verification', 'expected': 'ok', 'command': task['regression_command']},
                    {'kind': 'explicit_submission', 'expected': 'ok', 'command': SUBMIT}])
    costs = [labels(action['command']) for action in actions]
    require(len(actions) <= budget['max_model_calls'], 'actor_call_budget')
    require(max(costs) <= budget['max_generated_tokens_per_call'], 'native_action_budget')
    require(sum(costs) <= budget['max_total_generated_tokens'], 'actor_token_budget')
    return actions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.protocol.resolve(), args.output.resolve(), make_plan=make_plan)
