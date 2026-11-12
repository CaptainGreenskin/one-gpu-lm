"""Execute early edit-and-test demonstrations on the training partition only.

Five exact 020 recovery records per task stay unchanged. No model generation,
evaluation trajectory, or evaluation answer is used in this CPU preparation.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.data.balanced import prepare as prepare_balanced
from one_gpu_lm.data.repair_curriculum import Episode, SLOTS, locate_command, replacement_command
from one_gpu_lm.evaluation.environment import RepositoryEnvironment, repository_python_site
from one_gpu_lm.evaluation.grading import generated_files, grade
from one_gpu_lm.evaluation.tasks import prepare, file_manifest

MODES = ('evidence_then_edit', 'missing_read_then_edit')
CHANGED_SLOT = 'stall_repair'


def assign_modes(tasks):
    if not tasks or any(t['split'] != 'train' for t in tasks):
        raise ValueError('Mode assignment accepts training tasks only')
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate training task')
    groups = defaultdict(list)
    for task in tasks: groups[task['repository']].append(task['id'])
    return {ident: MODES[index % 2] for repo in sorted(groups)
            for index, ident in enumerate(sorted(groups[repo]))}


def missing_read_command():
    # The path is a missing disposable preview, not an evaluation-derived path.
    return ("python3 - <<'PY'\nfrom pathlib import Path\n"
            "p=Path('.one_gpu_missing_source_preview')\n"
            "assert not p.exists()\np.read_text()\nPY")


def repair_and_test_command(before, after, regression_command):
    if not isinstance(regression_command, str) or not regression_command.strip():
        raise ValueError('A declared regression command is required')
    return 'set -e\n' + replacement_command(before, after) + '\n' + regression_command


def snapshot_example(episode, command):
    prefix, progress = episode.progress.prepare_request(episode.messages, len(episode.steps), 12, True)
    target = '<tool_call>\n<function=bash>\n<parameter=command>\n' + command + '\n</parameter>\n</function>\n</tool_call>'
    return {'messages': copy.deepcopy(prefix), 'target': target}, progress


def build(root, out, prepared):
    root, out, prepared = map(lambda p: Path(p).resolve(), (root, out, prepared))
    if out.exists() or prepared.exists(): raise ValueError('Preserve existing CPU preparation attempts')
    read = lambda p: json.loads(Path(p).read_text())
    exp = root/'experiments/022-action-repair'
    plan = read(exp/'preparation-plan.json'); profile = read(exp/'profile-action.json')
    task_file = root/'private/tasks.json'; tasks = read(task_file)['tasks']
    training = [t for t in tasks if t['split'] == 'train']; modes = assign_modes(training)
    admission_file = root/'runs/admission/admission.json'; admission = read(admission_file)
    approved = {t['id'] for t in admission['tasks'] if t['admitted'] and t['split'] == 'train'}
    if (len(training) != 55 or approved != set(modes) or not admission['complete']
            or admission['task_spec_sha256'] != sha256(task_file)):
        raise ValueError('Require the exact 55 admitted training tasks')
    parent = root/'runs/reference-data020/recovery/train.jsonl'
    parent_protocol = read(root/'experiments/021-observed-recovery/protocol.json')
    if sha256(parent) != parent_protocol['frozen_files'][str(parent.relative_to(root))]:
        raise ValueError('Fixed 020 parent training data changed')
    rows = [json.loads(line) for line in parent.read_text().splitlines()]
    if len(rows) != 330 or Counter(r['provenance']['task_id'] for r in rows) != Counter({k: 6 for k in modes}):
        raise ValueError('Require six exact parent slots per training task')
    for ident in modes:
        if [r['provenance']['stage'] for r in rows if r['provenance']['task_id'] == ident] != list(SLOTS):
            raise ValueError('Parent semantic slots changed')
    runtime = read(root/'runtime.json')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'], local_files_only=True, trust_remote_code=False)
    downloaded = read(Path(runtime['model'])/'download-receipt.json')
    if any(downloaded[k] != profile[k] for k in ['model_id', 'revision']):
        raise ValueError('Tokenizer identity differs')
    out.mkdir(parents=True, mode=0o700); (out/'private').mkdir(mode=0o700)
    replacements, checks, rejected, accepted_modes = {}, [], Counter(), Counter()
    for task in sorted(training, key=lambda t: t['id']):
        ident, mode = task['id'], modes[task['id']]
        slug = task['repository'].replace('/', '--'); snapshot = root/'sources'/slug
        private = out/'private'/ident; private.mkdir()
        workspace = prepare(snapshot, private/'workspace', task)
        original = file_manifest(workspace, exclude=generated_files(task['repository']))
        deps = root/'node-deps'/slug/'node_modules'
        def make_env(work):
            return RepositoryEnvironment(work, python_site=repository_python_site(root, runtime, task['repository']),
                node_root=runtime['node_root'], node_modules=deps if deps.exists() else None, output_chars=2400)
        episode = Episode(task, workspace, snapshot, private, original, make_env)
        failed = grade(task, workspace, snapshot, private/'before-grade', original, make_env)
        if failed['passed'] or failed.get('returncode') != 1:
            raise ValueError('The admitted original defect must still fail independent tests')
        if mode == 'missing_read_then_edit':
            episode.action('missing_preview_read', missing_read_command(), 1)
            if 'FileNotFoundError' not in episode.messages[-1]['content']:
                raise ValueError('The missing-file observation must be an actual filesystem error')
        episode.action('locate_allowed_sources', locate_command(task), 0)
        episode.action('declared_failing_test', task['regression_command'], 1)
        if any(s['state_after']['changed_editable_files'] or s['state_after']['invalid_editable_files'] for s in episode.steps):
            raise ValueError('Diagnostic prefix unexpectedly edited source')
        good = {name: (snapshot/name).read_text() for name in task['editable_files']}
        command = repair_and_test_command(episode.sources(), good, task['regression_command'])
        example, progress = snapshot_example(episode, command)
        encoded = encode_training_example(example, tokenizer, profile)
        counts = {'input_tokens': len(encoded['ids']), 'supervised_tokens': encoded['supervised_tokens']}
        reason = ('complete_context_limit' if counts['input_tokens'] > profile['training']['max_sequence_length'] else
                  'complete_target_limit' if counts['supervised_tokens'] > profile['generation']['max_new_tokens'] else None)
        if reason:
            rejected[reason] += 1
            write_json(private/'rejected-prefix.json', {'messages': episode.messages, 'steps': episode.steps,
                       'reason': reason, **counts})
            checks.append({'id': ident, 'mode': mode, 'fallback': True, 'reason': reason, **counts})
        else:
            repair_call = len(episode.steps)+1
            if repair_call != (3 if mode == MODES[0] else 4): raise ValueError('Early edit call changed')
            episode.action('early_repair_and_declared_test', command, 0, slot=CHANGED_SLOT)
            episode.action('submit', 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT', 0)
            produced, control = episode.finish(wrong_grade=failed)
            if len(produced) != 1 or any(produced[0][k] != example[k] for k in ['messages', 'target']):
                raise ValueError('Executed example differs from the complete checked prefix')
            row = produced[0]
            row['provenance'].update(curriculum='022-action-repair', mode=mode,
                repair_call=repair_call, regression_in_target=True, development_traces_used=False)
            replacements[row['id']] = row; accepted_modes[mode] += 1
            checks.append(dict(control, mode=mode, fallback=False, repair_call=repair_call,
                               prefix_no_source_edits=True, total_calls=len(episode.steps), **counts))
        write_json(out/'status.json', {'complete': False, 'tasks_done': len(checks), 'replaced_records': len(replacements)})
        print(ident, 'fallback' if reason else 'verified early edit', flush=True)
    result = [replacements.get(row['id'], row) for row in rows]
    common = [i for i, r in enumerate(rows) if r['provenance']['stage'] != CHANGED_SLOT]
    if len(common) != 275 or any(rows[i] != result[i] for i in common):
        raise ValueError('Five common parent slots must remain byte-equivalent as records')
    coverage = (len(replacements) >= plan['data']['minimum_replaced_records']
                and all(accepted_modes[m] >= plan['data']['minimum_each_mode'] for m in MODES))
    target = out/'action'; target.mkdir(mode=0o700); data = target/'train.jsonl'
    data.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in result))
    receipt = {'schema_version': 1, 'records': 330, 'sha256': sha256(data),
        'task_spec_sha256': sha256(task_file), 'admission_sha256': sha256(admission_file),
        'license_record_sha256': sha256(exp/'sources.json'), 'parent_data_sha256': sha256(parent),
        'raw_data_public': False, 'kinds': dict(Counter(r['provenance']['kind'] for r in result)),
        'task_ids': sorted(modes), 'replaced_records': len(replacements), 'replaced_tasks': len(replacements),
        'assigned_modes': dict(Counter(modes.values())), 'accepted_modes': dict(accepted_modes),
        'fallback_reasons': dict(rejected), 'coverage_passed': coverage, 'checks': checks,
        'evaluation_traces_used': False, 'teacher_api_used': False, 'incorrect_actions_supervised': False,
        'shared_records': 275, 'original_failure_recovery_records_preserved': 55}
    write_json(target/'receipt.json', receipt)
    selection = prepare_balanced({'data': data, 'data_receipt': target/'receipt.json',
        'admission': admission_file, 'licenses': exp/'sources.json'}, profile, tokenizer, 330, prepared)
    old_prepared = [json.loads(line) for line in (root/'runs/prepared-data020/recovery/train.jsonl').read_text().splitlines()]
    new_prepared = [json.loads(line) for line in (prepared/'train.jsonl').read_text().splitlines()]
    if [r['id'] for r in old_prepared] != [r['id'] for r in new_prepared] or selection['excluded']:
        raise ValueError('Preserve all 330 complete, unique records and semantic order')
    if selection['supervised_tokens'] < plan['data']['minimum_supervised_tokens']:
        raise ValueError('Insufficient supervision')
    public = {k:v for k,v in receipt.items() if k not in {'checks', 'task_ids'}}
    public.update(prepared_data_sha256=sha256(prepared/'train.jsonl'),
        prepared_receipt_sha256=sha256(prepared/'receipt.json'), selection_sha256=sha256(prepared/'selection.json'),
        input_tokens=selection['input_tokens'], supervised_tokens=selection['supervised_tokens'],
        semantic_order_sha256=hashlib.sha256(json.dumps([r['id'] for r in new_prepared],separators=(',', ':')).encode()).hexdigest())
    write_json(out/'derived-data.json', public)
    write_json(out/'status.json', {'complete': True, 'tasks_done': 55, 'replaced_records': len(replacements), 'coverage_passed': coverage})
    return public


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'out', 'prepared']: p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(build(a.root, a.out, a.prepared)))
