"""Execute paired repair/stall and failure-recovery supervision on training tasks.

Both variants retain six targets per task: inspect, repair, verify, submit,
repair after a real read-only stall, and a matched late repair. Only the final
slot differs: read-only context versus an incorrect edit followed by failing
tests. Incorrect actions are context only, never supervised targets.
"""
import argparse
import ast
from collections import Counter
import copy
import difflib
import json
from pathlib import Path
import re
import shlex

from one_gpu_lm.common.actions import INSTANCE, parse_action
from one_gpu_lm.common.agent_policy import NATIVE_SYSTEM
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.data.reference_episodes import repair_commands
from one_gpu_lm.evaluation.environment import RepositoryEnvironment, repository_python_site
from one_gpu_lm.evaluation.grading import generated_files, grade
from one_gpu_lm.evaluation.tasks import apply_mutation, file_manifest, prepare


VARIANTS = ('repair', 'recovery')
SLOTS = ('inspect', 'repair', 'verify', 'submit', 'stall_repair', 'late_repair')


def replacement_edits(before, after):
    """Find literal, uniquely matched replacements without invented line numbers."""
    if set(before) != set(after):
        raise ValueError('A repair must preserve the declared editable file set')
    edits = []
    for name in sorted(before):
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or not re.fullmatch(r'[A-Za-z0-9_./-]+', name):
            raise ValueError('Unsafe patch filename')
        if before[name] == after[name]:
            continue
        a, b = before[name].splitlines(keepends=True), after[name].splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for context in range(0, 9):
            proposed = [(name, ''.join(a[g[0][1]:g[-1][2]]), ''.join(b[g[0][3]:g[-1][4]]))
                for g in matcher.get_grouped_opcodes(context)]
            trial = before[name]
            for _, old, new in proposed:
                if not old or trial.count(old) != 1:
                    break
                trial = trial.replace(old, new, 1)
            else:
                if trial == after[name]:
                    edits.extend(proposed); break
        else:
            raise ValueError('No bounded unique literal edit anchor')
    if not edits:
        raise ValueError('A supervised repair must make an actual source change')
    return edits


def replacement_command(before, after, *, fail_anchor=False):
    edits = replacement_edits(before, after)
    if fail_anchor:
        name, old, new = edits[0]
        edits[0] = (name, old+'# obsolete edit anchor\n', new)
        if edits[0][1] in before[name]:
            raise ValueError('Failure control accidentally found its stale anchor')
    # Validate every anchor before writing any file. Python literal replacement
    # preserves ampersands and backslashes; shell/sed replacement expansion is absent.
    return ("python3 - <<'PY'\nfrom pathlib import Path\nupdates={}\n"
        + 'for name,old,new in '+repr(edits)+':\n'
        + "    text=updates.get(name,Path(name).read_text())\n"
        + "    assert text.count(old)==1,(name,'edit anchor mismatch')\n"
        + "    updates[name]=text.replace(old,new,1)\n"
        + "for name,text in updates.items(): Path(name).write_text(text)\nPY")


def locate_command(task):
    """Read declared sources without using an oracle's line-number window."""
    commands = []
    for name in task['editable_files']:
        path = shlex.quote(name)
        commands.append('wc -l -- '+path)
        if name.endswith('.py'):
            commands.append("grep -n -B6 -A3 -F -- 'implementation missing' "+path)
        else:
            commands.append('cat -- '+path)
    return '\n'.join(commands)


def wrong_sources(task, broken, plan):
    """Materialize a declared incorrect attempt, never a development-derived fix."""
    if task['split'] != 'train' or plan['task_id'] != task['id'] or plan['split'] != 'train':
        raise ValueError('Wrong-edit plans must belong to the exact training task')
    if set(broken) != set(task['editable_files']):
        raise ValueError('Wrong-edit input differs from allowed sources')
    result = dict(broken)
    for change in plan['mutations']:
        if change['path'] not in result or change['kind'] != 'replace':
            raise ValueError('Incorrect attempts may only replace allowed source content')
        result[change['path']] = apply_mutation(result[change['path']], change)
    if result == broken:
        raise ValueError('Incorrect attempt must change actual source')
    for name, text in result.items():
        if name.endswith('.py'):
            ast.parse(text)
    return result


def check_train_inputs(tasks, admission, ledger, failure_plan, task_hash):
    train = [t for t in tasks if t['split'] == 'train']
    approved = {t['id'] for t in admission['tasks'] if t['admitted'] and t['split'] == 'train'}
    if not train or admission.get('complete') is not True or admission['task_spec_sha256'] != task_hash:
        raise ValueError('Require complete admission for actual training tasks')
    if failure_plan.get('parent_task_spec_sha256') != task_hash:
        raise ValueError('Incorrect-attempt plan belongs to different task inputs')
    if {t['id'] for t in train} != approved or len(train) != len(approved):
        raise ValueError('Do not drop or duplicate an admitted training task')
    allowed = {(s['repository'], s['revision']) for s in ledger['sources']
        if s['split'] == 'train' and s.get('license_sha256') and s.get('rights_review', {}).get('status')}
    if any((t['repository'], t['revision']) not in allowed for t in train):
        raise ValueError('Unreviewed training source')
    entries = failure_plan['tasks']
    if len(entries) != len(approved) or {p['task_id'] for p in entries} != approved or any(p['split'] != 'train' for p in entries):
        raise ValueError('Failure plans must cover training only, exactly once')
    return train, {p['task_id']: p for p in entries}


class Episode:
    def __init__(self, task, workspace, snapshot, private, original, make_env):
        from jinja2 import Template
        self.task, self.workspace, self.snapshot = task, workspace, snapshot
        self.private, self.original, self.make_env = private, original, make_env
        self.env = make_env(workspace)
        self.progress = EditableProgress(workspace, original, task['editable_files'])
        prompt = Template(INSTANCE).render(task=task['description'],
            editable_files=', '.join(task['editable_files']), regression_command=task['regression_command'])
        self.messages = [{'role': 'system', 'content': NATIVE_SYSTEM}, {'role': 'user', 'content': prompt}]
        self.steps, self.rows = [], []

    def sources(self):
        return {name: (self.workspace/name).read_text() for name in self.task['editable_files']}

    def action(self, kind, command, expected_exit, *, slot=None):
        index = len(self.steps)
        if index >= 12:
            raise ValueError('Reference episode exhausted the same call budget')
        if slot is not None and (expected_exit != 0 or kind == 'incorrect_attempt' or slot not in SLOTS):
            raise ValueError('Only successful declared reference targets may be supervised')
        target = '<tool_call>\n<function=bash>\n<parameter=command>\n'+command+'\n</parameter>\n</function>\n</tool_call>'
        action = parse_action(target)
        prefix, progress = self.progress.prepare_request(self.messages, index, 12, True)
        before = self.progress.snapshot()
        result = self.env.execute(action)
        after = self.progress.snapshot()
        if result['returncode'] != expected_exit:
            write_json(self.private/'failure.json', {'kind': kind, 'expected_exit': expected_exit, 'result': result})
            raise ValueError('Reference action did not meet its execution contract: '+self.task['id']+':'+kind)
        if slot is not None:
            self.rows.append({'id': self.task['id']+':curriculum-'+slot,
                'messages': copy.deepcopy(prefix), 'target': target,
                'provenance': {'repository': self.task['repository'], 'revision': self.task['revision'],
                    'split': 'train', 'task_id': self.task['id'], 'kind': 'verified-reference-repair',
                    'stage': slot, 'curriculum': '020-repair-curriculum',
                    'branch': self.private.name, 'assistant_message_index': len(self.messages),
                    'execution_returncode': result['returncode'], 'progress': progress}})
        self.steps.append({'kind': kind, 'supervised_slot': slot, 'call_index': index+1,
            'state_before': before, 'state_after': after, 'returncode': result['returncode'],
            'source_sha256_after': {name: sha256(self.workspace/name) for name in self.task['editable_files']}})
        self.messages.append({'role': 'assistant', 'content': target, 'extra': {'actions': [action]}})
        self.messages.append({'role': 'user', 'content': f"Exit code: {result['returncode']}\n{result['output']}",
            'extra': {'tool': {k:v for k,v in result.items() if k != 'output'}}})

    def finish(self, *, wrong_grade=None):
        result = grade(self.task, self.workspace, self.snapshot, self.private/'grade', self.original, self.make_env)
        expected = {name: sha256(self.snapshot/name) for name in self.task['editable_files']}
        if not result['passed'] or result['source_sha256'] != expected:
            raise ValueError('Reference does not restore and independently pass the exact task')
        trajectory = {'messages': self.messages, 'steps': self.steps, 'grade': result, 'wrong_grade': wrong_grade}
        proof = self.private/'trajectory.json'; write_json(proof, trajectory)
        for row in self.rows:
            row['provenance']['verified_trajectory_sha256'] = sha256(proof)
        return self.rows, {'id': self.task['id'], 'branch': self.private.name, 'passed': True,
            'trajectory_sha256': sha256(proof), 'retained_slots': [r['provenance']['stage'] for r in self.rows],
            'recovery_control_independently_failed': wrong_grade is not None and not wrong_grade['passed'],
            'recovery_mode': wrong_grade.get('recovery_mode') if wrong_grade else None,
            'recovery_control_changed_files': len(wrong_grade['modified_source_files']) if wrong_grade else None,
            'wrong_grade_returncode': wrong_grade.get('returncode') if wrong_grade else None}


def build(root, tasks_path, admission_path, licenses_path, failure_plan_path, out):
    root, out = Path(root).resolve(), Path(out)
    if out.exists():
        raise ValueError('Preserve existing dataset attempts')
    read = lambda p: json.loads(Path(p).read_text())
    tasks = read(tasks_path)['tasks']; admission = read(admission_path); ledger = read(licenses_path)
    train, plans = check_train_inputs(tasks, admission, ledger, read(failure_plan_path), sha256(tasks_path))
    runtime = read(root/'runtime.json')
    out.mkdir(parents=True, mode=0o700); (out/'private').mkdir(mode=0o700)
    datasets = {name: [] for name in VARIANTS}; controls = []
    for task in train:
        slug = task['repository'].replace('/', '--'); snapshot = root/'sources'/slug
        deps = root/'node-deps'/slug/'node_modules'
        def make_env(work):
            return RepositoryEnvironment(work, python_site=repository_python_site(root, runtime, task['repository']),
                node_root=runtime['node_root'], node_modules=deps if deps.exists() else None, output_chars=2400)
        common, contrast = [], {}
        for branch in ['ordinary', 'stalled', 'control', 'recovery']:
            private = out/'private'/task['id']/branch; private.mkdir(parents=True)
            workspace = prepare(snapshot, private/'workspace', task)
            original = file_manifest(workspace, exclude=generated_files(task['repository']))
            episode = Episode(task, workspace, snapshot, private, original, make_env)
            actions = repair_commands(task, workspace, snapshot)
            _, test, _, review, verify, submit = [r[1] for r in actions]
            inspect = locate_command(task)
            reference = {name: (snapshot/name).read_text() for name in task['editable_files']}
            repair = replacement_command(episode.sources(), reference)
            episode.action('inspect', inspect, 0, slot='inspect' if branch == 'ordinary' else None)
            episode.action('reproduce', test, 1)
            wrong_grade = None
            if branch == 'stalled':
                episode.action('repeated_inspection', inspect, 0)
                episode.action('repeated_test', test, 1)
            elif branch == 'control':
                count = 'wc -l -- '+' '.join(shlex.quote(n) for n in task['editable_files'])
                episode.action('source_size_probe', count, 0)
                episode.action('repeated_inspection', inspect, 0)
                episode.action('repeated_test', test, 1)
            elif branch == 'recovery':
                broken = episode.sources(); wrong = wrong_sources(task, broken, plans[task['id']])
                if wrong == reference:
                    raise ValueError('Incorrect attempt accidentally equals the reference')
                failed_anchor = plans[task['id']]['mode'] == 'failed_anchor'
                if plans[task['id']]['mode'] not in {'failed_anchor', 'incorrect_edit'}:
                    raise ValueError('Unregistered recovery mode')
                episode.action('incorrect_attempt', replacement_command(broken, wrong, fail_anchor=failed_anchor), 1 if failed_anchor else 0)
                # The marker disappears after the deliberately incorrect Python
                # edit; use the already observed range to inspect its current body.
                episode.action('review_incorrect_attempt', actions[0][1], 0)
                episode.action('failing_test_after_edit', test, 1)
                wrong_grade = grade(task, workspace, snapshot, private/'wrong-grade', original, make_env)
                if (wrong_grade.get('returncode') != 1 or wrong_grade['passed']
                    or bool(wrong_grade['modified_source_files']) == failed_anchor):
                    raise ValueError('Incorrect attempt lacks an independent failed-test control')
                wrong_grade['recovery_mode'] = plans[task['id']]['mode']
                repair = replacement_command(episode.sources(), reference)
            slot = 'repair' if branch == 'ordinary' else 'stall_repair' if branch == 'stalled' else 'late_repair'
            episode.action('correct_repair', repair, 0, slot=slot)
            if branch == 'ordinary':
                episode.action('review', review, 0)
            episode.action('verify', verify, 0, slot='verify' if branch == 'ordinary' else None)
            episode.action('submit', submit, 0, slot='submit' if branch == 'ordinary' else None)
            rows, control = episode.finish(wrong_grade=wrong_grade); controls.append(control)
            if branch in {'ordinary', 'stalled'}:
                common.extend(rows)
            else:
                contrast['repair' if branch == 'control' else 'recovery'] = rows
        for variant in VARIANTS:
            rows = common+contrast[variant]
            if [r['provenance']['stage'] for r in rows] != list(SLOTS):
                raise ValueError('Every task must retain all six paired slots')
            datasets[variant].extend(rows)
        write_json(out/'status.json', {'complete': False, 'tasks_done': len(controls)//4,
            'records_per_variant': {k:len(v) for k,v in datasets.items()}})
        print(task['id'], 'four branches independently verified', flush=True)
    receipts = {}
    for variant, rows in datasets.items():
        destination = out/variant; destination.mkdir(mode=0o700)
        data = destination/'train.jsonl'
        data.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        receipt = {'schema_version': 1, 'records': len(rows), 'sha256': sha256(data),
            'raw_data_public': False, 'task_spec_sha256': sha256(tasks_path),
            'license_record_sha256': sha256(licenses_path), 'admission_sha256': sha256(admission_path),
            'failure_plan_sha256': sha256(failure_plan_path), 'variant': variant,
            'kinds': dict(Counter(r['provenance']['kind'] for r in rows)),
            'stages': dict(Counter(r['provenance']['stage'] for r in rows)),
            'repositories': sorted({t['repository'] for t in train}), 'task_ids': sorted(t['id'] for t in train),
            'reference_controls': controls, 'teacher_api_used': False,
            'development_examples_used': False, 'incorrect_actions_supervised': False,
            'executed_tool_metadata_preserved': True}
        write_json(destination/'receipt.json', receipt); receipts[variant] = receipt
    write_json(out/'status.json', {'complete': True, 'tasks_done': len(train),
        'records_per_variant': {k:len(v) for k,v in datasets.items()}})
    return receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'tasks', 'admission', 'licenses', 'failure-plan', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.tasks, args.admission, args.licenses, args.failure_plan, args.out)
