"""Validate executed paired curricula before any 020 weight updates or scores."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.training_data import read_training_data
from one_gpu_lm.data.repair_curriculum import SLOTS, locate_command, replacement_command, wrong_sources
from one_gpu_lm.evaluation.tasks import apply_mutation


def source_state(hashes, original):
    return {'changed_editable_files': sum(hashes[n] != original[n] for n in original),
        'invalid_editable_files': 0,
        'editable_state_sha256': hashlib.sha256(json.dumps(sorted(hashes.items()), separators=(',', ':')).encode()).hexdigest()}


def verify_trace(task, control, texts, plan, trajectory, branch):
    """Reconstruct permitted source transitions and target actions, not just flags."""
    broken, reference = texts
    hashed = lambda value: {n: hashlib.sha256(t.encode()).hexdigest() for n,t in value.items()}
    initial, repaired = hashed(broken), hashed(reference)
    if initial != control['broken_sha256'] or repaired != control['reference_sha256']:
        raise ValueError('Source snapshots differ from admitted source identities')
    expected_kinds = {
        'ordinary': ['inspect','reproduce','correct_repair','review','verify','submit'],
        'stalled': ['inspect','reproduce','repeated_inspection','repeated_test','correct_repair','verify','submit'],
        'control': ['inspect','reproduce','source_size_probe','repeated_inspection','repeated_test','correct_repair','verify','submit'],
        'recovery': ['inspect','reproduce','incorrect_attempt','review_incorrect_attempt','failing_test_after_edit','correct_repair','verify','submit'],
    }
    steps = trajectory['steps']; messages = trajectory['messages']
    if [s['kind'] for s in steps] != expected_kinds[branch] or len(messages) != 2+2*len(steps):
        raise ValueError('Reference branch differs from the declared execution plan')
    current = dict(broken); expected_rows = {}
    for index, step in enumerate(steps):
        assistant, observation = messages[2+index*2:4+index*2]
        if assistant['role'] != 'assistant' or observation['role'] != 'user':
            raise ValueError('Reference tool turn has wrong roles')
        command = parse_action(assistant['content'])['command']
        if assistant['extra']['actions'] != [{'command': command}] or step['call_index'] != index+1:
            raise ValueError('Reference action differs from its actual execution')
        before = source_state(hashed(current), initial)
        expected_exit = 1 if step['kind'] in {'reproduce','repeated_test','failing_test_after_edit'} else 0
        if step['kind'] == 'incorrect_attempt':
            wrong = wrong_sources(task, broken, plan)
            failed = plan['mode'] == 'failed_anchor'
            if command != replacement_command(broken, wrong, fail_anchor=failed):
                raise ValueError('Incorrect attempt differs from the training-only failure plan')
            if failed: expected_exit = 1
            else: current = wrong
        if step['kind'] == 'correct_repair':
            if command != replacement_command(current, reference):
                raise ValueError('Repair target is not grounded in the observed source state')
            current = dict(reference)
        if step['kind'] == 'inspect' and command != locate_command(task):
            raise ValueError('Initial source read uses an undeclared locator')
        if step['kind'] in {'reproduce','repeated_test','failing_test_after_edit','verify'} and command != task['regression_command']:
            raise ValueError('Reference changed the regression test command')
        if step['kind'] == 'submit' and command != 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT':
            raise ValueError('Reference submit action changed')
        if (step['state_before'] != before or step['state_after'] != source_state(hashed(current), initial)
            or step['source_sha256_after'] != hashed(current) or step['returncode'] != expected_exit
            or observation['extra']['tool']['returncode'] != expected_exit):
            raise ValueError('Executed source transition or tool outcome differs from the declared branch')
        slot = step['supervised_slot']
        if slot is not None:
            if expected_exit != 0 or step['kind'] == 'incorrect_attempt':
                raise ValueError('Incorrect action entered supervised targets')
            tracker = object.__new__(EditableProgress); tracker.snapshot = lambda state=before: dict(state)
            prefix, progress = tracker.prepare_request(messages[:2+index*2], index, 12, True)
            expected_rows[slot] = (assistant['content'], prefix, progress, 2+index*2)
    result = trajectory['grade']
    if result['passed'] is not True or result['returncode'] != 0 or result['source_sha256'] != repaired:
        raise ValueError('Independent final-source grade did not pass')
    if branch == 'recovery':
        wrong = trajectory['wrong_grade']; expected = initial if plan['mode'] == 'failed_anchor' else hashed(wrong_sources(task, broken, plan))
        if wrong['passed'] is not False or wrong['returncode'] != 1 or wrong['source_sha256'] != expected or wrong['recovery_mode'] != plan['mode']:
            raise ValueError('Independent failed-attempt control is missing')
    elif trajectory['wrong_grade'] is not None:
        raise ValueError('Read-only control unexpectedly contains failure-recovery grading')
    expected_slots = list(SLOTS[:4]) if branch == 'ordinary' else ['stall_repair'] if branch == 'stalled' else ['late_repair']
    if list(expected_rows) != expected_slots:
        raise ValueError('Supervision slot count or ordering changed')
    return expected_rows


def verify(root, out):
    root, out = Path(root).resolve(), Path(out)
    if out.exists(): raise ValueError('Preserve previous preparation receipts')
    exp = root/'experiments/020-repair-curriculum'
    read = lambda p: json.loads(Path(p).read_text())
    parent_spec = importlib.util.spec_from_file_location('curriculum_parent_preparation', root/'experiments/019-coding-release/preparation.py')
    parent = importlib.util.module_from_spec(parent_spec); parent_spec.loader.exec_module(parent)
    plan = read(exp/'preparation-plan.json'); parent_protocol = read(root/'experiments/019-coding-release/protocol.json')
    tasks_path = root/'private/tasks.json'; admission_path = root/'runs/admission/admission.json'
    if sha256(tasks_path) != parent_protocol['task_spec_sha256'] or sha256(admission_path) != parent_protocol['admission_sha256']:
        raise ValueError('020 must preserve all original task partitions and admission identities')
    tasks = read(tasks_path)['tasks']; admission = read(admission_path); ledger = read(exp/'sources.json')
    cohorts = parent.validate_cohorts(tasks, admission, plan)
    parent.validate_source_reviews(tasks, admission, ledger, exp)
    failure = read(root/'private/failure-plan.json'); failures = {v['task_id']:v for v in failure['tasks']}
    if failure['parent_task_spec_sha256'] != sha256(tasks_path) or set(failures) != set(cohorts['train']):
        raise ValueError('Incorrect-attempt plans must cover only the unchanged training partition')
    if dict(Counter(v['mode'] for v in failures.values())) != plan['data']['recovery_modes']:
        raise ValueError('Recovery-mode allocation changed')
    profiles = {n: read(exp/('profile-'+n+'.json')) for n in ['repair','recovery']}
    previous_profile = read(root/'experiments/019-coding-release/profile-low.json')
    if profiles['repair'] != profiles['recovery'] or profiles['repair'] != previous_profile:
        raise ValueError('Candidate profiles must both preserve the 019 low-learning-rate recipe')
    runtime = read(root/'runtime.json')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'], local_files_only=True, trust_remote_code=False)
    controls = {t['id']:t for t in admission['tasks']}; cache = {}; proofs = {}
    for task in tasks:
        if task['split'] != 'train': continue
        snapshot = root/'sources'/task['repository'].replace('/','--')
        reference = {n:(snapshot/n).read_text() for n in task['editable_files']}; broken = dict(reference)
        for m in task['mutations']: broken[m['path']] = apply_mutation(broken[m['path']], m)
        for branch in ['ordinary','stalled','control','recovery']:
            path = root/'runs/reference-data020/private'/task['id']/branch/'trajectory.json'
            cache[(task['id'],branch)] = verify_trace(task, controls[task['id']], (broken,reference), failures[task['id']], read(path), branch)
            proofs[(task['id'],branch)] = sha256(path)
    values = {}; selected = {}; ids = {}
    for variant in ['repair','recovery']:
        data_root = root/'runs/prepared-data020'/variant
        paths = {'data':data_root/'train.jsonl','data_receipt':data_root/'receipt.json',
            'admission':admission_path,'licenses':exp/'sources.json'}
        rows, approved = read_training_data({'paths':paths}); selection = read(data_root/'selection.json')
        if selection['excluded'] or len(rows) != 330:
            raise ValueError('Do not drop oversized or duplicate training records; fix preparation before freeze')
        if selection['data_sha256'] != sha256(paths['data']) or selection['receipt_sha256'] != sha256(paths['data_receipt']):
            raise ValueError('Selection receipts do not identify the actual candidate data')
        values[variant] = parent.validate_examples(rows, selection, approved, tokenizer, profiles[variant], plan)
        for row in rows:
            p = row['provenance']; key = (p['task_id'],p['branch']); expected = cache[key][p['stage']]
            if (row['target'],row['messages'],p['progress'],p['assistant_message_index']) != expected or p['verified_trajectory_sha256'] != proofs[key]:
                raise ValueError('Supervised target or prefix differs from its executed branch')
            if p['stage'] == 'late_repair' and p['branch'] != ('control' if variant == 'repair' else 'recovery'):
                raise ValueError('Counterfactual context assigned to the wrong candidate')
        selected[variant] = {r['id']:r for r in selection['selected']}; ids[variant] = [r['id'] for r in selection['selected']]
        config = read(exp/('train-'+variant+'.json'))
        if config['run']['max_steps'] != 330 or config['run']['max_examples'] != 330:
            raise ValueError('Both candidates must consume exactly 330 unique targets once')
    if ids['repair'] != ids['recovery']:
        raise ValueError('Task/slot sample order differs between candidates')
    shared = [name for name in ids['repair'] if not name.endswith(':curriculum-late_repair')]
    contrast = [name for name in ids['repair'] if name.endswith(':curriculum-late_repair')]
    if len(shared) != 275 or len(contrast) != 55 or any(selected['repair'][n]['record_sha256'] != selected['recovery'][n]['record_sha256'] for n in shared):
        raise ValueError('Common supervision differs outside the declared 55 contrast slots')
    if any(selected['repair'][n]['record_sha256'] == selected['recovery'][n]['record_sha256'] for n in contrast):
        raise ValueError('Recovery contrast contains an unchanged example')
    result = {'schema_version':1,'passed':True,'cohort_counts':{k:len(v) for k,v in cohorts.items()},
        'training':values,'shared_identical_records':275,'contrasting_records':55,'verified_execution_branches':len(cache),
        'supervised_incorrect_actions':0,'recovery_modes':dict(Counter(v['mode'] for v in failures.values())),
        'identities':{'tasks':sha256(tasks_path),'admission':sha256(admission_path),
            'failure_plan':sha256(root/'private/failure-plan.json'),'licenses':sha256(exp/'sources.json')},
        'semantic_order_sha256':hashlib.sha256(json.dumps(ids['repair'],separators=(',',':')).encode()).hexdigest(),
        'scope':'Actual executed source-state transitions, independent failed/successful controls and tokenizer eligibility. No model-quality claim; final model scores unopened.'}
    write_json(out,result); return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); value=verify(args.root,args.out); print(json.dumps(value['training']))
