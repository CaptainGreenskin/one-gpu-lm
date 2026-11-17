"""Admit historical training-only edited failures, without new model calls.

Compound-command errors are candidate signals only. Replay must preserve source
hashes and delivered feedback, then a standalone visible regression must fail.
No final grade or later trajectory success is used to select a state.
"""
import argparse
import copy
import json
from pathlib import Path
import time

from one_gpu_lm.common.receipts import sha256, write_json
from .correction_policy import CorrectionGuard
from .evaluation_rollout import normalization
from .recovery_state import canonical, copy_tree, inventory
from .sandbox import Sandbox
from .submission import SubmissionGuard
from .tasks import digest, files, prepare, relative

ORIGINS = [('025', 'teacher-collection-001'), ('026', 'recovery026-001'), ('027', 'correction027-001')]


def read(path):
    return json.loads(Path(path).read_text())


def candidates(trace, task):
    if trace['split'] != 'training' or task['split'] != 'training':
        raise ValueError('Training states only')
    if not trace['complete'] or trace.get('unknown_inflight_generation_cost') or len(trace['steps']) != trace['attempted_calls']:
        raise ValueError('Only complete accounted historical trajectories')
    original = trace['original_source_sha256']; states = {}; edited = {}
    for step in trace['steps']:
        obs = step.get('observation', {}); before = step['source_before']; after = step.get('source_after')
        if obs.get('executed') is True and after and after != before:
            edited[digest(canonical(after))] = step['call']
        if (obs.get('executed') is not True or obs.get('reason') != 'runtime_error'
                or obs.get('returncode', 0) == 0 or not after or after == original
                or set(after) != set(original) or any(v is None for v in after.values())):
            continue
        key = digest(canonical(after))
        if key not in edited:
            continue
        if key not in states:
            states[key] = {'call': step['call'], 'failed_edit_call': edited[key],
                           'source_state_sha256': key, 'recorded_standalone_failure_calls': []}
        if step.get('command') == task['regression_command'] and before == after:
            states[key]['recorded_standalone_failure_calls'].append(step['call'])
    return list(states.values())


def references(root, protocol):
    root = Path(root); result = []
    for origin, name in ORIGINS:
        directory = root/'runs'/name; path = directory/'audit.json'
        if sha256(path) != protocol['source_audits'][origin]:
            raise ValueError('Source audit changed')
        audit = read(path)
        if origin != '025' and not audit['passed']:
            raise ValueError('Prior audit failed')
        for row in audit['rows']:
            if origin == '025':
                if not row['input_replay_passed']:
                    raise ValueError('Prior input replay failed')
                branches = [('teacher', directory/'private/tasks'/row['id'], row['trajectory_sha256'])]
            elif origin == '026':
                if not row['audit_passed']:
                    raise ValueError('Prior pair audit failed')
                branches = [(arm, directory/'private/tasks'/row['id']/(arm if row['disposition'] == 'paired' else 'prefix'), row[arm]['trajectory_sha256'])
                            for arm in ['student', 'teacher'] if arm in row]
            else:
                if not row['audit_passed']:
                    raise ValueError('Prior correction audit failed')
                branches = [(row['arm'], directory/'private/tasks'/row['run_id'], row['result']['trajectory_sha256'])]
            for branch, location, expected in branches:
                path = location/'trajectory.json'
                if sha256(path) != expected:
                    raise ValueError('Historical trajectory changed')
                result.append({'origin': origin, 'branch': branch, 'directory': location,
                               'trace': read(path), 'trajectory_sha256': expected})
    return result


def source_state(workspace, task):
    value = files(workspace)
    return {name: value.get(name) for name in task['editable_files']}


def protected(workspace, before, task):
    current = files(workspace)
    return all(current.get(name) == value for name, value in before.items() if name not in task['editable_files'])


class BoundedRunner:
    def __init__(self, runner, deadline, maximum):
        self.runner = runner; self.deadline = deadline; self.maximum = maximum
        self.commands = 0; self.seconds = 0.; self.timeout = 60

    def execute(self, workspace, command):
        left = self.deadline-time.monotonic()
        if left < 1 or self.commands >= self.maximum:
            raise TimeoutError('CPU admission budget exhausted')
        self.runner.timeout = max(1, min(self.timeout, int(left)))
        self.commands += 1
        value = self.runner.execute(workspace, command)
        self.seconds += value['seconds']
        return value


def replay_to_failure(root, data, ref, item, task, source, directory, runner, norm, cap):
    """Replay only the prefix; do not inspect or execute subsequent actions."""
    directory = Path(directory); directory.mkdir()
    workspace = directory/'workspace'; trace = ref['trace']; historical = ref['directory']
    before = prepare(Path(data)/'snapshots'/task['repository'].replace('/', '--'), workspace, source, task)
    if {n: before[n] for n in task['editable_files']} != trace['original_source_sha256']:
        raise ValueError('Original source identity differs')
    actors = {str(workspace.resolve()): {k: task[k] for k in ['editable_files', 'regression_command']}}
    guard = CorrectionGuard(runner, actors, ref['branch']) if ref['origin'] == '027' else SubmissionGuard(runner, actors)
    receipts = []
    for step in trace['steps'][:item['call']]:
        call = step['call']
        if sha256(historical/f'request-{call:03d}.json') != step['request_sha256']:
            raise ValueError('Historical request changed')
        if source_state(workspace, task) != step['source_before'] or not protected(workspace, before, task):
            raise ValueError('Historical source prefix does not replay')
        if 'observation' not in step:
            if step['action_format_valid']:
                raise ValueError('Unexecuted in-flight prefix')
            continue
        original = read(historical/f'tool-{call:03d}.json')
        if {k: v for k, v in original.items() if k != 'output'} != step['observation']:
            raise ValueError('Historical tool receipt changed')
        raw = original['output']; delivered, recorded_norm = norm.transform(raw if len(raw) <= cap else raw[:cap]+norm.MARKER, 'extended', cap)
        if recorded_norm != step['normalization']:
            raise ValueError('Historical delivered feedback changed')
        expected_feedback = f"Exit code: {original['returncode']}\n{delivered}"+step.get('policy_notice', '')
        if trace['messages'][2*call+1]['content'] != expected_feedback:
            raise ValueError('Historical feedback history differs')
        guard.timeout = 60
        observed = guard.execute(workspace, step['command'])
        write_json(directory/f'replay-tool-{call:03d}.json', observed)
        actual = observed['output']; actual, _ = norm.transform(actual if len(actual) <= cap else actual[:cap]+norm.MARKER, 'extended', cap)
        if (any(observed.get(k) != original.get(k) for k in ['returncode', 'reason', 'executed', 'policy_event'])
                or actual != delivered or source_state(workspace, task) != step['source_after']
                or not protected(workspace, before, task)):
            raise ValueError('Source or delivered feedback differs on replay')
        receipts.append({'call': call, 'returncode': observed['returncode'], 'reason': observed['reason'],
                         'source_sha256': digest(canonical(source_state(workspace, task))),
                         'delivered_sha256': digest(actual.encode())})
    if digest(canonical(source_state(workspace, task))) != item['source_state_sha256']:
        raise ValueError('Candidate source changed during replay')
    # A newly recorded standalone visible check confirms the signal. It is not
    # represented as an original actor action or a hidden/reference check.
    checked_before = source_state(workspace, task)
    runner.timeout = 60; checked = runner.execute(workspace, task['regression_command'])
    write_json(directory/'standalone-check.json', checked)
    intact = protected(workspace, before, task) and source_state(workspace, task) == checked_before
    status = ('confirmed_failed_regression' if checked['reason'] == 'runtime_error' and checked['returncode'] != 0 and intact
              else 'standalone_passed' if checked['returncode'] == 0 and intact else 'invalid_confirmation')
    result = {'status': status, 'replayed_prefix': True, 'replayed_steps': len(receipts),
              'standalone_returncode': checked['returncode'], 'standalone_reason': checked['reason'],
              'protected_and_source_intact': intact}
    if status == 'confirmed_failed_regression':
        state = directory/'snapshot'; state.mkdir(); manifest = inventory(workspace)
        copy_tree(workspace, state/'tree', manifest)
        raw = checked['output']; delivered, _ = norm.transform(raw if len(raw) <= cap else raw[:cap]+norm.MARKER, 'extended', cap)
        prefix = {'messages': copy.deepcopy(trace['messages'][:2+2*item['call']]),
                  'standalone_feedback': f"Exit code: {checked['returncode']}\n{delivered}",
                  'failed_edit_call': item['failed_edit_call'], 'failure_call': item['call'],
                  'original_source_sha256': trace['original_source_sha256'],
                  'failed_source_sha256': checked_before,
                  'historical_calls': item['call'],
                  'historical_generated_tokens': sum(s['generation']['generated_tokens'] for s in trace['steps'][:item['call']]),
                  'historical_input_tokens': sum(s['generation']['input_tokens'] for s in trace['steps'][:item['call']])}
        for name, value in [('prefix.json', prefix), ('manifest.json', manifest), ('before.json', before), ('replay.json', receipts)]:
            (state/name).write_bytes(canonical(value))
        result['snapshot'] = {name: sha256(state/name) for name in ['prefix.json', 'manifest.json', 'before.json', 'replay.json']}
    return result


def select(rows, maximum):
    """First confirmed state per task; fixed source order, then hash round-robin."""
    by_task = {}
    for row in rows:
        if row['status'] == 'confirmed_failed_regression' and row['task_id'] not in by_task:
            by_task[row['task_id']] = row
    groups = {}
    for row in by_task.values():
        groups.setdefault(row['family'], []).append(row)
    for rows_in_group in groups.values():
        rows_in_group.sort(key=lambda r: digest(('028-failure-state-v1\0'+r['task_id']).encode()))
    chosen = []
    while groups and len(chosen) < maximum:
        for family in sorted(list(groups)):
            chosen.append(groups[family].pop(0))
            if not groups[family]: del groups[family]
            if len(chosen) == maximum: break
    return chosen


def run(root, protocol_path, out):
    root, protocol_path, out = Path(root), Path(protocol_path), Path(out)
    if out.exists(): raise ValueError('Preserve admission evidence; no automatic retry')
    p = read(protocol_path)
    if any(sha256(root/relative(n)) != value for n, value in p['frozen_files'].items()):
        raise ValueError('Frozen admission input changed')
    refs = references(root, p); data = root/relative(p['data_directory'])
    if len(refs) != p['expected_audited_trajectories']:
        raise ValueError('Historical trajectory inventory changed')
    tasks = {t['id']: t for t in read(data/'tasks.json')['tasks']}
    sources = {s['repository']: s for s in read(data/'sources.json')['sources']}
    plan = []
    for ref in refs:
        trace = ref['trace']; task = tasks[trace['id']]
        for item in candidates(trace, task): plan.append((ref, item, task))
    if len(plan) != p['expected_candidate_states'] or len(plan) > p['cpu_budget']['maximum_candidate_states']:
        raise ValueError('Inventory exceeds registered CPU scope')
    out.mkdir(parents=True); (out/'private').mkdir(); started = time.time()
    runner = BoundedRunner(Sandbox(read(root/'private/task-runtime.json')), time.monotonic()+p['cpu_budget']['max_seconds'], p['cpu_budget']['max_commands'])
    norm = normalization(root); rows = []; fatal = None
    for index, (ref, item, task) in enumerate(plan):
        name = f"state-{index+1:03d}"
        row = dict(item, state_id=name, origin=ref['origin'], branch=ref['branch'], task_id=task['id'], family=task['family'], trajectory_sha256=ref['trajectory_sha256'])
        if fatal:
            row.update(status='not_attempted_after_budget', error_type=fatal)
        else:
            try:
                row.update(replay_to_failure(root, data, ref, item, task, sources[task['repository']], out/'private'/name, runner, norm, p['feedback_chars']))
            except (ValueError, TimeoutError) as error:
                row.update(status='replay_rejected', error_type=type(error).__name__, reason=str(error))
                if isinstance(error, TimeoutError): fatal=type(error).__name__
        rows.append(row)
        write_json(out/'status.json', {'stage':'admitting','completed_states':len(rows),'expected_states':len(plan),'commands_executed':runner.commands})
    chosen = select(rows, p['maximum_selected_tasks']); families = len({r['family'] for r in chosen})
    gate = len(chosen) >= p['minimum_pool_tasks'] and families >= p['minimum_pool_families'] and not fatal
    unchanged = all(sha256(root/relative(n)) == value for n, value in p['frozen_files'].items())
    value = {'schema_version':1,'stage':'admitted','complete':True,'protocol_sha256':sha256(protocol_path),
             'audited_training_trajectories':len(refs),'candidate_states':len(plan),'rows':rows,
             'selected_state_ids':[r['state_id'] for r in chosen], 'selected_unique_tasks':len(chosen),
             'selected_families':families,'gpu_collection_admission_passed':gate and unchanged,
             'frozen_inputs_unchanged':unchanged,'new_model_calls':0,'optimizer_updates':0,
             'commands_executed':runner.commands,'command_seconds':runner.seconds,
             'started_epoch':started,'finished_epoch':time.time(),'final_test_opened':False,
             'hidden_checks_used_for_admission':False,'weights_public':False,
             'state_scope':'reconstructed historical prefix plus recorded standalone visible revalidation; not original inode/clock/process state'}
    write_json(out/'admission.json',value);write_json(out/'status.json',{'stage':'completed','complete':True,'gpu_collection_admission_passed':value['gpu_collection_admission_passed']})
    return value


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ['root','protocol','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();value=run(args.root,args.protocol,args.out)
    print(json.dumps({k:value[k] for k in ['complete','candidate_states','selected_unique_tasks','selected_families','gpu_collection_admission_passed','new_model_calls','commands_executed']}))
