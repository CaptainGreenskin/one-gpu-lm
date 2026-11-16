"""Audit recorded paired training continuations; no new generation or test execution."""
import argparse
import json
from pathlib import Path
import traceback

from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.common.receipts import sha256, write_json
from .evaluation_audit import recorded_grade
from .evaluation_rollout import normalization
from .guard_audit import target_is_executed
from .recovery import actual_cost, branch_order
from .recovery_replay import replay
from .recovery_state import canonical, digest, inventory, validate_prefix
from .tasks import prepare, relative
from .submission import SubmissionGuard


def attempt_costs(run_dir, task_ids):
    """Include interrupted attempts; copied prefix receipts are not new requests."""
    attempts = []
    for task_id in task_ids:
        for branch in ['prefix', 'student', 'teacher']:
            path = Path(run_dir)/'private/tasks'/task_id/branch/'trajectory.json'
            if not path.exists(): continue
            trace = json.loads(path.read_text()); shared = 0 if branch == 'prefix' else 3
            steps = [s for s in trace['steps'] if s['call'] > shared]
            requests = [p for p in path.parent.glob('request-*.json') if int(p.stem.split('-')[1]) > shared]
            attempted = max(0, trace['attempted_calls']-shared, len(requests))
            attempts.append({'id': task_id, 'branch': branch, 'attempted_calls': attempted,
                'known_generated_tokens': sum(s['generation']['generated_tokens'] for s in steps),
                'known_input_tokens': sum(s['generation']['input_tokens'] for s in steps),
                'unknown_inflight_generation_cost': bool(trace.get('unknown_inflight_generation_cost'))
                    or attempted != len(steps) or len(requests) != attempted,
                'actor_seconds': trace['actor_seconds']-trace.get('shared_prefix_actor_seconds', 0),
                'trajectory_sha256': sha256(path)})
    return attempts


def encode_continuation(directory, trace, task, protocol, profile, tokenizer):
    records = []; excluded = []; errors = []; receipts = []
    for step in trace['steps'][3:]:
        if not target_is_executed(step):
            excluded.append({'call': step['call'], 'reason': 'malformed_or_unexecuted'}); continue
        try:
            payload = json.loads((Path(directory)/f"request-{step['call']:03d}.json").read_text()); selected = {}
            def encode(messages):
                selected['messages'] = messages
                return encode_prefix(tokenizer, messages, profile['chat_template_kwargs'], protocol['agent_policy']['format'])[0]
            prefix, context = prepare_context(payload['messages'], encode, payload['max_input_tokens'], protocol['agent_policy']['context'])
            if (digest(json.dumps(prefix, separators=(',', ':')).encode()) != step['generation']['input_tokens_sha256']
                    or context != step['generation']['context']):
                raise ValueError('Actual teacher input does not replay')
            kind = ('edit' if step.get('source_after') != step['source_before'] else
                    'verify' if step['command'] == task['regression_command'] else
                    'submit' if step['command'] == 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' else 'other')
            row = {'id': task['id']+':'+str(step['call']), 'messages': selected['messages'], 'target': step['generation']['text'],
                   'provenance': {'split': 'train', 'task_id': task['id'], 'repository': task['repository'],
                                  'kind': kind, 'call': step['call'], 'producer_kind': 'model', 'model_role': 'base',
                                  'request_sha256': step['request_sha256'], 'trajectory_sha256': sha256(Path(directory)/'trajectory.json'),
                                  'shared_prefix_sha256': trace['shared_prefix_receipt']['prefix_sha256'], 'executed': True}}
            encoded = encode_training_example(row, tokenizer, profile)
            if (encoded['ids'][:len(prefix)] != prefix or len(encoded['ids']) > protocol['encoding_screen_max_tokens']
                    or encoded['supervised_tokens'] <= 0):
                raise ValueError('Full teacher target does not fit registered encoding screen')
            receipt = {'call': step['call'], 'kind': kind, 'tokens': len(encoded['ids']), 'supervised_tokens': encoded['supervised_tokens']}
            row['encoding_receipt'] = receipt; records.append(row); receipts.append(receipt)
        except (ValueError, KeyError) as error:
            errors.append({'call': step['call'], 'error_type': type(error).__name__})
    eligible = (trace['eligible_verified_trajectory'] and not errors
                and {'edit', 'verify', 'submit'} <= {r['kind'] for r in receipts})
    return {'qualified': bool(eligible), 'encoded_actions': receipts, 'excluded_calls': excluded, 'encoding_errors': errors}, records


def check_shared_state(directory, trace, prefix, receipt):
    directory = Path(directory); snapshot = directory.parent/'snapshot'
    validate_prefix(prefix)
    checks = {
        'prefix_receipt': digest((snapshot/'prefix.json').read_bytes()) == receipt['prefix_sha256']
            and json.loads((snapshot/'prefix.json').read_text()) == prefix,
        'snapshot_manifest': digest((snapshot/'manifest.json').read_bytes()) == receipt['manifest_sha256']
            and inventory(snapshot/'tree') == json.loads((snapshot/'manifest.json').read_text()),
        'restored_state': json.loads((directory/'restored-state.json').read_text()) == {
            'manifest_sha256': receipt['manifest_sha256'], 'prefix_sha256': receipt['prefix_sha256'],
            'actor_seconds': prefix['actor_seconds'], 'attempted_calls': 3, 'generated_tokens': prefix['generated_tokens']},
        'shared_steps': trace['steps'][:3] == prefix['steps'],
        'shared_budget': trace['shared_prefix_calls'] == 3
            and trace['shared_prefix_generated_tokens'] == prefix['generated_tokens']
            and trace['shared_prefix_actor_seconds'] == prefix['actor_seconds']
            and trace['actor_seconds'] >= prefix['actor_seconds'],
        'same_prefix_receipts': all((directory/p.name).read_bytes() == p.read_bytes()
            for p in [*(directory.parent/'prefix').glob('request-*.json'), *(directory.parent/'prefix').glob('tool-*.json')])}
    return checks


def run(root, protocol_path, out):
    from transformers import AutoTokenizer
    root, protocol_path, out = Path(root), Path(protocol_path), Path(out)
    if out.exists(): raise ValueError('Preserve previous audit')
    protocol = json.loads(protocol_path.read_text()); run_dir = root/relative(protocol['run_directory'])
    controller = json.loads((run_dir/'controller.json').read_text())
    supervisor = json.loads((root/relative(protocol['supervisor_receipt'])).read_text())
    if not controller.get('backend_stopped') or not supervisor['complete']:
        raise ValueError('Wait until every owned backend stops')
    data = root/relative(protocol['data_directory'])
    tasks = {t['id']: t for t in json.loads((data/'tasks.json').read_text())['tasks']}
    sources = {s['repository']: s for s in json.loads((data/'sources.json').read_text())['sources']}
    profile = json.loads((root/relative(protocol['evaluation_profile_path'])).read_text())
    runtime = json.loads((root/'private/parent-runtime.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'], local_files_only=True, trust_remote_code=False)
    norm = normalization(root); reconstruction = run_dir/'private/audit-reconstruction'
    reconstruction.mkdir(); export = run_dir/'private/continuation-export'; export.mkdir()
    by_id = {r['id']: r for r in controller['tasks']}; rows = []; records = []
    for index, task_id in enumerate(protocol['task_ids']):
        task = tasks[task_id]; source = sources[task['repository']]; directory = run_dir/'private/tasks'/task_id
        row = {'id': task_id, 'repository': task['repository'], 'family': task['family'], 'index': index,
               'complete': False, 'audit_passed': False, 'qualified_teacher_continuation': False}
        if task_id not in by_id:
            row['disposition'] = 'incomplete_or_not_started'; rows.append(row); continue
        observed = by_id[task_id]; row['disposition'] = observed['disposition']
        try:
            before = prepare(data/'snapshots'/task['repository'].replace('/', '--'), reconstruction/'broken'/task_id, source, task)
            prefix = json.loads((directory/'prefix/trajectory.json').read_text())
            paired = observed['disposition'] == 'paired'; checks = {'branch_order': observed['branch_order'] == branch_order(index)}
            if paired:
                prefix_replay = replay(directory/'prefix', prefix, task, protocol, profile, tokenizer, norm)
                checks['prefix_replay'] = prefix_replay['passed']
                checks['prefix_cost'] = observed['prefix_cost'] == {**{k: prefix[k] for k in ['attempted_calls', 'generated_tokens', 'actor_seconds']},
                    'input_tokens': prefix_replay['known_input_tokens']}
            for branch in ['student', 'teacher'] if paired else ['student']:
                branch_dir = directory/branch if paired else directory/'prefix'
                trace = json.loads((branch_dir/'trajectory.json').read_text()); value = json.loads((branch_dir/'result.json').read_text())
                replayed = replay(branch_dir, trace, task, protocol, profile, tokenizer, norm)
                expected_grade = recorded_grade(root, data, branch_dir, task, source, before, reconstruction/'graders'/task_id/branch)
                budget = protocol['evaluation_budget']
                compliant = (trace['actor_seconds'] <= budget['max_task_seconds'] and trace['attempted_calls'] <= budget['max_model_calls']
                    and trace['generated_tokens'] <= budget['max_total_generated_tokens']
                    and all(s['generation']['generated_tokens'] <= budget['max_generated_tokens_per_call']
                            and s['generation']['input_tokens'] <= budget['max_input_tokens'] for s in trace['steps']))
                final_source = SubmissionGuard.source_state(branch_dir/'workspace', task)
                verified = (replayed['reconstructed_verified_source'] is not None
                    and replayed['reconstructed_verified_source'] == final_source)
                submitted = bool(trace['steps'] and trace['steps'][-1].get('command') == 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
                    and trace['steps'][-1].get('observation', {}).get('returncode') == 0
                    and trace['steps'][-1].get('observation', {}).get('executed') is True)
                branch_checks = {'request_replay': replayed['passed'], 'recorded_grader': expected_grade == trace['grade'],
                    'identity': trace['split'] == 'training' and trace['id'] == task_id and trace['branch'] == branch
                        and trace['protocol_sha256'] == digest(json.dumps(protocol, sort_keys=True).encode()),
                    'original_source': trace['original_source_sha256'] == {n: before[n] for n in task['editable_files']},
                    'final_source': final_source == trace['final_source_sha256'],
                    'score': compliant == trace['actor_budget_compliant'] == value['actor_budget_compliant']
                        and (expected_grade['passed'] and compliant) == trace['success_within_budget'] == value['success_within_budget'],
                    'verified_submission': verified == trace['agent_verified_final_source'] == value['agent_verified_final_source']
                        and submitted == trace['submitted'] == value['submitted']
                        and bool(expected_grade['passed'] and compliant and verified and submitted) == trace['eligible_verified_trajectory'] == value['eligible_verified_trajectory'],
                    'result_costs': all(value[k] == trace[k] for k in ['attempted_calls', 'generated_tokens', 'actor_seconds'])
                        and value['input_tokens'] == replayed['known_input_tokens'],
                    'recorded_result': value == observed[branch] and value['trajectory_sha256'] == sha256(branch_dir/'trajectory.json')}
                if paired:
                    branch_checks.update(check_shared_state(branch_dir, trace, prefix, observed['snapshot_receipt']))
                row[branch] = dict(value, audit_checks=branch_checks)
                checks[branch] = all(branch_checks.values())
                if branch == 'teacher':
                    encoded, raw_records = encode_continuation(branch_dir, trace, task, protocol, profile, tokenizer)
                    row['teacher_encoding'] = encoded
                    row['qualified_teacher_continuation'] = bool(encoded['qualified'] and checks[branch])
                    if row['qualified_teacher_continuation']: records.extend(raw_records)
            checks['actual_cost'] = actual_cost(observed) == observed['actual_cost']
            row.update(complete=True, audit_checks=checks, audit_passed=all(checks.values()), actual_cost=observed['actual_cost'])
            if not row['audit_passed']: row['qualified_teacher_continuation'] = False
        except Exception as error:
            row.update(error_type=type(error).__name__)
            (reconstruction/(task_id+'-error.log')).write_text(traceback.format_exc())
        rows.append(row)
    complete = all(r['complete'] and r['audit_passed'] for r in rows) and controller['complete']
    qualified_ids = {r['id'] for r in rows if r['qualified_teacher_continuation']}
    records = [r for r in records if r['provenance']['task_id'] in qualified_ids]
    (export/'data.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    manifest = {'full_episodes': 0, 'unique_teacher_continuations': len(qualified_ids), 'teacher_action_records': len(records),
                'supervised_tokens': sum(r['encoding_receipt']['supervised_tokens'] for r in records),
                'data_sha256': sha256(export/'data.jsonl'), 'candidate_training_authorized': False,
                'student_prefix_used_as_target': False, 'raw_data_public': False}
    write_json(export/'manifest.json', manifest)
    environments = [json.loads((run_dir/'private/backends'/f"{e['index']:04d}-{e['role']}-environment.json").read_text())
                    == json.loads((root/relative(protocol['environment_path'])).read_text()) for e in controller['backend_lifecycles']]
    attempts = attempt_costs(run_dir, protocol['task_ids'])
    checks = {'all_tasks_audited': complete, 'same_model_interpreter': bool(environments) and all(environments),
              'all_generation_cost_known': all(not a['unknown_inflight_generation_cost'] for a in attempts),
              'global_caps': sum(a['attempted_calls'] for a in attempts) <= 1008
                  and sum(a['known_generated_tokens'] for a in attempts) <= 516096,
              'total_cost_matches_physical_attempts': complete and
                  sum(a['attempted_calls'] for a in attempts) == sum(r['actual_cost']['attempted_calls'] for r in rows)
                  and sum(a['known_generated_tokens'] for a in attempts) == sum(r['actual_cost']['generated_tokens'] for r in rows),
              'frozen_inputs_unchanged': all(sha256(root/relative(n)) == h for n, h in protocol['frozen_files'].items()),
              'supervisor_complete': supervisor['complete'] and not supervisor['timed_out'] and supervisor['worker_returncode'] == 0,
              'controller_protocol_identity': controller['protocol_sha256'] == sha256(protocol_path)}
    result = {'schema_version': 1, 'protocol_sha256': sha256(protocol_path), 'checks': checks, 'passed': all(checks.values()),
              'rows': rows, 'attempts': attempts, 'private_export_summary': manifest, 'model_generations': 0, 'optimizer_updates': 0,
              'test_commands_reexecuted': 0, 'final_test_opened': False, 'weights_public': False}
    write_json(out, result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root', 'protocol', 'out']: parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); value = run(args.root, args.protocol, args.out)
    print(json.dumps({'passed': value['passed'], 'checks': value['checks']}))
