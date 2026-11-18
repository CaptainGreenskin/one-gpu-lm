"""Prepare matched next-action inputs from the verified 032 development backup."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.receipt_client import payload_sha256
from .behavior_diagnosis import SUBMIT, inspection_shaped
from .historic_eval_design import VARIANTS


BASE, PREDECESSOR, CANDIDATE = VARIANTS


def read(path):
    return json.loads(Path(path).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ids_digest(ids):
    return digest(json.dumps(ids, separators=(',', ':')).encode())


def changed(step):
    return 'source_after' in step and step['source_after'] != step['source_before']


def first_edit(trace):
    return next((step['call'] for step in trace['steps'] if changed(step)), None)


def read_backup(archive, names):
    remaining = set(names)
    values = {}
    with tarfile.open(archive, 'r|gz') as stream:
        for entry in stream:
            if entry.name not in remaining:
                continue
            if not entry.isfile() or entry.islnk() or entry.size > 6 * 2**20:
                raise ValueError('Expected a regular development receipt within the size limit')
            values[entry.name] = stream.extractfile(entry).read()
            remaining.remove(entry.name)
            if not remaining:
                return values
    raise ValueError('The backup is missing registered development receipts')


def matched_groups(protocol, audit):
    if (not audit['passed'] or audit['final_test_opened']
            or not all(audit['checks'].values())):
        raise ValueError('The source development comparison must have passed its audit')
    expected = [(s['run_id'], s['task_id'], s['repetition'], s['variant']) for s in protocol['schedule']]
    actual = [(r['run_id'], r['id'], r['repetition'], r['variant']) for r in audit['rows']]
    if len(expected) != 324 or expected != actual or len(set(expected)) != len(expected):
        raise ValueError('The source has missing or reordered development runs')
    groups = defaultdict(dict)
    for row in audit['rows']:
        if not row['audit_passed'] or row['unknown_inflight_generation_cost'] or not row['request_replay']['passed']:
            raise ValueError('Every source run must be fully accounted')
        key = row['id'], row['repetition']
        if row['variant'] in groups[key]:
            raise ValueError('Duplicate task, repetition and model')
        groups[key][row['variant']] = row
    if len(groups) != 108 or any(set(group) != set(VARIANTS) for group in groups.values()):
        raise ValueError('Incomplete model pairing')
    return groups


def summarize_observed(rows, traces, tasks, groups):
    aggregates = {}
    first_actions = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row['variant'] == variant]
        summary = Counter()
        histogram = Counter()
        for row in selected:
            trace = traces[row['run_id']]
            first = first_edit(trace)
            regression = tasks[row['id']]['regression_command']
            steps = trace['steps']
            executed = [step for step in steps if 'observation' in step]
            checks = [step for step in executed if step.get('command') == regression]
            summary.update({
                'successful_and_submitted': row['complete_success_within_budget'],
                'never_changed_source': first is None,
                'first_edit_at_call_11_or_12': first is not None and first >= 11,
                'no_retained_source_change': trace['original_source_sha256'] == trace['final_source_sha256'],
                'exact_declared_regression_calls': len(checks),
                'runs_with_repeated_exact_regression': len(checks) > 1,
                'runs_with_repeated_executed_command': len({s['command'] for s in executed}) < len(executed),
                'context_compaction_calls': sum(s['generation']['context']['dropped_rounds'] > 0 for s in steps),
                'truncated_tool_observation_calls': sum(s.get('normalization', {}).get('raw_prefix_clipped', False) for s in steps),
                'generation_cap_calls': sum(s['generation']['hit_generation_limit'] for s in steps),
            })
            first_step = steps[0]
            if not first_step['action_format_valid']:
                category = 'invalid'
            elif first_step.get('command') == regression:
                category = 'declared_regression'
            elif inspection_shaped(first_step['command']):
                category = 'inspection_shaped'
            else:
                category = 'other_parseable'
            histogram[category] += 1
        aggregates[variant] = {'runs':len(selected), **summary,
                                'median_first_edit_when_edited':statistics.median(
                                    row['first_edit_call'] for row in selected if row['first_edit_call'] is not None)}
        first_actions[variant] = dict(histogram)

    comparisons = {}
    for reference in (BASE, PREDECESSOR):
        losses = []
        gains = []
        for (task_id, repetition), group in sorted(groups.items()):
            parent, candidate = group[reference], group[CANDIDATE]
            reference_passed = parent['complete_success_within_budget']
            candidate_passed = candidate['complete_success_within_budget']
            if reference_passed and not candidate_passed:
                p = traces[parent['run_id']]['steps']
                c = traces[candidate['run_id']]['steps']
                losses.append({'task_id':task_id, 'repetition':repetition,
                    'reference_first_edit':parent['first_edit_call'], 'candidate_first_edit':candidate['first_edit_call'],
                    'candidate_terminal':candidate['terminal_reason'], 'candidate_calls':len(c),
                    'first_input_equal':p[0]['generation']['input_tokens_sha256']==c[0]['generation']['input_tokens_sha256'],
                    'first_output_differs':p[0]['generation']['ids']!=c[0]['generation']['ids'],
                    'candidate_context_compaction_calls':sum(s['generation']['context']['dropped_rounds']>0 for s in c),
                    'candidate_generation_cap_calls':sum(s['generation']['hit_generation_limit'] for s in c)})
            elif candidate_passed and not reference_passed:
                gains.append({'task_id':task_id, 'repetition':repetition})
        comparisons[reference] = {'lost_runs':losses, 'gained_runs':gains}
    return {'by_model':aggregates, 'first_action_categories':first_actions, 'paired_changes':comparisons}


def select_anchors(groups, traces):
    tasks = sorted({task for task, _ in groups})
    chosen = []
    for task in tasks:
        rows = [groups[task, repeat] for repeat in range(3)]
        base_passes = [r[BASE] for r in rows if r[BASE]['complete_success_within_budget']]
        candidate_passes = sum(r[CANDIDATE]['complete_success_within_budget'] for r in rows)
        if len(base_passes) > candidate_passes:
            cohort = 'regression'
        elif len(base_passes) == candidate_passes == 3:
            cohort = 'stable_success'
        else:
            continue
        reference = base_passes[0]
        call = reference['first_edit_call']
        if call is None or not changed(traces[reference['run_id']]['steps'][call-1]):
            raise ValueError('Successful source run has no verified edit')
        paired = rows[reference['repetition']][CANDIDATE]
        if cohort == 'regression' and (paired['complete_success_within_budget']
                or (paired['first_edit_call'] is not None and paired['first_edit_call'] <= call)):
            raise ValueError('The frozen regression cohort requires both histories before an edit')
        sources = [reference] if cohort == 'stable_success' else [reference, paired]
        known_edits = set()
        for group in rows:
            for row in group.values():
                trace = traces[row['run_id']]
                first = first_edit(trace)
                if first is not None:
                    step = trace['steps'][first-1]
                    if step['source_before'] != trace['original_source_sha256'] or 'observation' not in step:
                        raise ValueError('Expected a recorded first edit on original source')
                    known_edits.add(step['command'])
        for source in sources:
            chosen.append((cohort, reference, source, call, sorted(known_edits)))
    return chosen


def prepare(archive, expected_archive, protocol_path, audit_path, private_out, public_out):
    if private_out.exists() or public_out.exists():
        raise ValueError('Preserve the prepared input archive and selection')
    if sha256(archive) != expected_archive:
        raise ValueError('Verified local archive identity changed')
    protocol = read(protocol_path)
    audit = read(audit_path)
    if audit['protocol_sha256'] != sha256(protocol_path) or protocol['final_test_opened']:
        raise ValueError('Only the audited 032 development episode set can be prepared')
    groups = matched_groups(protocol, audit)
    base = Path(protocol['run_directory']) / 'private/tasks'
    rows = audit['rows']
    tasks_path = str(Path(protocol['data_directory']) / 'tasks.json')
    trace_paths = {row['run_id']:str(base/row['run_id']/'trajectory.json') for row in rows}
    files = read_backup(archive, [tasks_path, *trace_paths.values()])
    if digest(files[tasks_path]) != protocol['frozen_files'][tasks_path]:
        raise ValueError('Development task inventory changed')
    tasks = {t['id']:t for t in json.loads(files.pop(tasks_path))['tasks']}
    if len(tasks) != 36 or any(t['split'] != 'development' for t in tasks.values()):
        raise ValueError('Only the 36 authorized development tasks are in scope')
    traces = {}
    for row in rows:
        raw = files.pop(trace_paths[row['run_id']])
        trace = json.loads(raw)
        if (digest(raw) != row['trajectory_sha256'] or trace['split'] != 'development' or not trace['complete']
                or trace['id'] != row['id'] or first_edit(trace) != row['first_edit_call']
                or trace['complete_success_within_budget'] != row['complete_success_within_budget']):
            raise ValueError('Development trace differs from the original audit')
        traces[row['run_id']] = trace
    selected = select_anchors(groups, traces)
    request_paths = {source['run_id']:str(base/source['run_id']/f'request-{call:03d}.json')
                     for _,_,source,call,_ in selected}
    captured = read_backup(archive, request_paths.values())
    states = []
    private = []
    for cohort,ref,source,call,known_edits in selected:
        trace = traces[source['run_id']]
        step = trace['steps'][call-1]
        if step['source_before'] != trace['original_source_sha256'] or any(changed(s) for s in trace['steps'][:call-1]):
            raise ValueError('The selected request already followed an edit')
        raw = captured.pop(request_paths[source['run_id']])
        payload = json.loads(raw)
        history = source['variant']
        state = {'id':f"{source['id']}--{history}", 'task_id':source['id'], 'cohort':cohort,
                 'history_model':history,'call':call,'source_run_id':source['run_id'],
                 'source_trajectory_sha256':source['trajectory_sha256'],'source_request_sha256':digest(raw),
                 'source_output_ids_sha256':ids_digest(step['generation']['ids']),
                 'input_tokens':step['generation']['input_tokens'],'input_tokens_sha256':step['generation']['input_tokens_sha256'],
                 'payload_sha256':payload_sha256(payload),'max_generated_tokens':payload['max_new_tokens'],
                 'known_first_edit_commands':len(known_edits)}
        if (state['source_request_sha256'] != step['request_sha256'] or payload['operation'] != 'generate'
                or payload['agent_policy'] != protocol['agent_policy'] or payload['max_input_tokens'] != 8192
                or not 1 <= payload['max_new_tokens'] <= 512):
            raise ValueError('The recovered request diverges from the original input or budget')
        reference_command = traces[ref['run_id']]['steps'][call-1]['command']
        states.append(state)
        private.append({'id':state['id'],'payload':payload,'base_first_edit_command':reference_command,
                        'observed_first_edit_commands':known_edits,
                        'declared_regression_command':tasks[source['id']]['regression_command']})
    private_out.mkdir(parents=True, mode=0o700)
    data = private_out/'states.json'
    write_json(data, {'states':private})
    data.chmod(0o600)
    result = {'schema_version':1,'source_protocol_sha256':sha256(protocol_path),'source_audit_sha256':sha256(audit_path),
              'archive_sha256':expected_archive,'private_states_sha256':sha256(data),
              'source_runs':324,'source_tasks':36,'prepared_states':states,
              'diagnosis':summarize_observed(rows,traces,tasks,groups),
              'new_model_generations':0,'new_tool_executions':0,'final_test_opened':False}
    public_out.parent.mkdir(parents=True,exist_ok=True)
    write_json(public_out,result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--archive-sha256',required=True)
    for name in ('protocol','audit','private-out','public-out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    result = prepare(args.archive,args.archive_sha256,args.protocol,args.audit,args.private_out,args.public_out)
    print(json.dumps({'source_runs':result['source_runs'],'prepared_states':len(result['prepared_states'])}))
