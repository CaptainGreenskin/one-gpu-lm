"""One frozen training-only paired recovery pilot, with original-state restoration."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.verify import verify_model
from .evaluation import Backend
from .evaluation_rollout import atomic
from .recovery_rollout import initialize, advance, capture_prefix, fork_prefix, finish
from .recovery_state import validate_prefix
from .sandbox import Sandbox
from .tasks import relative, validate_source


def select_tasks(tasks, count=48):
    groups = defaultdict(list)
    for task in tasks:
        if task['split'] != 'training':
            raise ValueError('Only approved training tasks may enter the pilot')
        groups[task['repository']].append(task['id'])
    for ids in groups.values():
        ids.sort(key=lambda name: hashlib.sha256(('026-recovery-pilot-v1\0'+name).encode()).hexdigest())
    result = []; index = 0
    while len(result) < count:
        added = False
        for repository in sorted(groups):
            if index < len(groups[repository]) and len(result) < count:
                result.append(groups[repository][index]); added = True
        if not added:
            raise ValueError('Insufficient preselected training tasks')
        index += 1
    if len(set(result)) != len(result):
        raise ValueError('Duplicate task identity')
    return result


def branch_order(index):
    return ['teacher', 'student'] if index % 2 == 0 else ['student', 'teacher']


def actual_cost(row):
    """The physically executed prefix is charged once, despite two path reports."""
    if row['disposition'] != 'paired':
        return {k: row['student'][k] for k in ['attempted_calls', 'generated_tokens', 'input_tokens', 'actor_seconds']}
    return {k: row['student'][k]+row['teacher'][k]-row['prefix_cost'][k]
            for k in ['attempted_calls', 'generated_tokens', 'input_tokens', 'actor_seconds']}


def run(root, protocol_path):
    root, protocol_path = Path(root).resolve(), Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text()); out = root/relative(protocol['run_directory'])
    if out.exists():
        raise ValueError('Preserve every prior pilot attempt; no implicit retry')
    publication = json.loads((root/relative(protocol['publication_receipt'])).read_text())
    if (publication.get('anonymous_files_verified') is not True or publication.get('ci_conclusion') != 'success'
            or publication['protocol_sha256'] != sha256(protocol_path)):
        raise ValueError('Executable preregistration must be public and CI verified')
    for name, expected in protocol['frozen_files'].items():
        if sha256(root/relative(name)) != expected:
            raise ValueError('Frozen recovery input changed: '+name)
    design = json.loads((root/relative(protocol['design_path'])).read_text())
    data = root/relative(protocol['data_directory'])
    tasks = json.loads((data/'tasks.json').read_text())['tasks']
    sources = json.loads((data/'sources.json').read_text())['sources']
    ids = select_tasks(tasks)
    if (ids != design['task_selection']['task_ids'] or ids != protocol['task_ids']
            or protocol['maximum_pipeline_seconds'] != design['budget']['maximum_pipeline_seconds']):
        raise ValueError('Registered task selection or duration changed')
    for source in sources:
        validate_source(source, data/'snapshots'/source['repository'].replace('/', '--'), 'training')
    by_id = {t['id']: t for t in tasks}; by_repo = {s['repository']: s for s in sources}
    for task_id in ids:
        task = by_id[task_id]; source = by_repo[task['repository']]
        if task['revision'] != source['revision'] or task['family'] != source['family']:
            raise ValueError('Training task/source identity mismatch')
    runtime = json.loads((root/'private/parent-runtime.json').read_text())
    sandbox = Sandbox(json.loads((root/'private/task-runtime.json').read_text()), protocol['evaluation_budget']['max_tool_seconds'])
    out.mkdir(parents=True); (out/'private').mkdir()
    started = time.time(); deadline = started+protocol['maximum_pipeline_seconds']
    state = {'schema_version': 1, 'stage': 'verify_model', 'complete': False, 'started_epoch': started,
             'started_utc': datetime.now(timezone.utc).isoformat(), 'deadline_epoch': deadline, 'tasks': [],
             'protocol_sha256': sha256(protocol_path), 'publication_commit': publication['commit'],
             'backend_lifecycles': [], 'expected_tasks': len(ids), 'optimizer_updates': 0,
             'final_test_opened': False, 'weights_public': False}
    def save(): atomic(out/'controller.json', state)
    save(); telemetry = Telemetry(out/'telemetry.csv', 1); telemetry.start()
    backend = Backend(root, out, runtime, protocol, state['backend_lifecycles'], save)
    try:
        profile = json.loads((root/relative(protocol['evaluation_profile_path'])).read_text())
        write_json(out/'model-verification.json', verify_model(Path(runtime['model']), root/relative(protocol['model_manifest_path']), profile))
        for index, task_id in enumerate(ids):
            # Reserve the worst case for both paths, two graders, and all reloads.
            # No task is started if it cannot fit without extending the deadline.
            if time.time()+3120 > deadline:
                state['stage'] = 'pipeline_budget_exhausted'; break
            task = by_id[task_id]; source = by_repo[task['repository']]
            directory = out/'private/tasks'/task_id; prefix_out = directory/'prefix'
            state.update(stage='loading_student', current_task=task_id, current_index=index); save()
            socket = backend.ensure('student', deadline)
            prefix = initialize(data, task, source, protocol, prefix_out)
            state['stage'] = 'generating_prefix'; save()
            advance(root, task, protocol, prefix_out, prefix, socket, sandbox, 'student', 3)
            backend.check_exclusive()
            cost = {k: prefix[k] for k in ['attempted_calls', 'generated_tokens', 'actor_seconds']}
            cost['input_tokens'] = sum(s['generation']['input_tokens'] for s in prefix['steps'])
            row = {'id': task_id, 'repository': task['repository'], 'family': task['family'], 'index': index,
                   'prefix_cost': cost, 'branch_order': branch_order(index), 'disposition': None}
            try:
                validate_prefix(prefix)
            except ValueError:
                row['disposition'] = 'ineligible_prefix'
            if row['disposition'] is None:
                try:
                    receipt = capture_prefix(prefix_out, prefix, directory/'snapshot')
                except ValueError:
                    # Nonserializable actor state is retained as an unpaired cost/outcome.
                    row['disposition'] = 'nonserializable_prefix'
            if row['disposition'] is not None:
                prefix['branch'] = 'student'
                if not prefix['terminal_reason']:
                    advance(root, task, protocol, prefix_out, prefix, socket, sandbox, 'student')
                row['student'] = finish(data, task, source, protocol, prefix_out, prefix, sandbox)
            else:
                row.update(disposition='paired', snapshot_receipt=receipt)
                for branch in branch_order(index):
                    role = 'base' if branch == 'teacher' else 'student'
                    state.update(stage='loading_'+branch, current_branch=branch); save()
                    socket = backend.ensure(role, deadline)
                    branch_out = directory/branch
                    trace = fork_prefix(prefix_out, directory/'snapshot', receipt, branch_out, branch)
                    state['stage'] = 'continuing_'+branch; save()
                    advance(root, task, protocol, branch_out, trace, socket, sandbox, role)
                    backend.check_exclusive()
                    row[branch] = finish(data, task, source, protocol, branch_out, trace, sandbox)
                    atomic(directory/'partial-pair.json', row)
            row['actual_cost'] = actual_cost(row)
            state['tasks'].append(row); save()
            calls = sum(r['actual_cost']['attempted_calls'] for r in state['tasks'])
            tokens = sum(r['actual_cost']['generated_tokens'] for r in state['tasks'])
            if calls > 1008 or tokens > 516096:
                raise ValueError('Registered global generation cap exceeded')
            print(json.dumps({'task_id': task_id, 'disposition': row['disposition'], 'completed_tasks': len(state['tasks']),
                              'calls': calls, 'known_generated_tokens': tokens}), flush=True)
        if len(state['tasks']) == len(ids):
            state.update(stage='completed', complete=True)
    except BaseException as error:
        state.update(stage='failed', error_type=type(error).__name__)
        (out/'private/error.log').write_text(traceback.format_exc())
    finally:
        backend.stop(); state['telemetry'] = telemetry.stop()
        state.update(finished_epoch=time.time(), finished_utc=datetime.now(timezone.utc).isoformat(), backend_stopped=True,
                     frozen_files_unchanged=all((root/relative(n)).is_file() and sha256(root/relative(n)) == h for n, h in protocol['frozen_files'].items()))
        if not state['frozen_files_unchanged']:
            state.update(complete=False, stage='frozen_input_changed')
        save()
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root', 'protocol']: parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); run(args.root, args.protocol)
