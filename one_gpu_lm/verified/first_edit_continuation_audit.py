"""Audit paired unchanged-source continuations and export verified action targets."""
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback

from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.reasoning import action_text, settings
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import atomic_json
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.serving.request_journal import RequestJournal
from .evaluation_audit import recorded_grade
from .failure_replay import replay_from_state
from .first_edit_continuation import load_cases, message_hash, read, require
from .guard_audit import target_is_executed
from .resilient_eval_audit import audit_task
from .rollout import normalization
from .submission import SUBMIT, SubmissionGuard
from .tasks import digest, relative


def edited_steps(trace):
    return [s for s in trace['steps'] if target_is_executed(s) and 'source_after' in s
            and s['source_after'] != s['source_before']]


def target_steps(trace, regression_command):
    """Use only the first successful edit, its final visible check and actual submission."""
    if not trace['eligible_verified_trajectory']:
        return [], 'repair_or_submission_failed'
    if any(not s['protected_before'] or not s['protected_after']
           for s in trace['steps'] if 'observation' in s):
        return [], 'protected_file_changed'
    edits = edited_steps(trace)
    if len(edits) != 1 or edits[0]['source_after'] != trace['final_source_sha256']:
        return [], 'first_edit_not_the_verified_final_patch'
    edit = edits[0]
    final = trace['final_source_sha256']
    checks = [s for s in trace['steps'] if s['call'] > edit['call'] and target_is_executed(s)
              and s['command'] == regression_command and s['observation']['returncode'] == 0
              and s['source_before'] == s.get('source_after') == final]
    submit = trace['steps'][-1]
    if (not checks or not target_is_executed(submit) or submit['command'] != SUBMIT
            or submit['observation']['returncode'] != 0 or submit['source_before'] != final):
        return [], 'final_verification_chain_missing'
    return [('edit', edit), ('verify', checks[-1]), ('submit', submit)], None


def encode_targets(directory, trace, case, reference, teacher, student, tokenizer, trajectory_sha):
    selected, reason = target_steps(trace, case['task']['regression_command'])
    if not selected:
        return [], reason
    policy = case['protocol']['agent_policy']
    rows = []
    for kind, step in selected:
        call = step['call']
        payload = read(directory / f'request-{call:03d}.json')
        captured = []
        def teacher_prefix(messages):
            captured[:] = messages
            return encode_prefix(tokenizer, messages, settings(teacher, payload)[0], policy['format'])[0]
        served, _ = prepare_context(payload['messages'], teacher_prefix, payload['max_input_tokens'],
                                    policy['context'])
        served_hash = digest(json.dumps(served, separators=(',', ':')).encode())
        require(served_hash == step['generation']['input_tokens_sha256'],
                'Positive-action history differs from the actual teacher request')
        if kind == 'edit' and not any(case['prefix']['standalone_visible_failure'] in msg['content']
                                     for msg in captured if msg['role'] == 'user'):
            return [], 'historical_visible_failure_dropped'
        student_prefix, _ = encode_prefix(tokenizer, captured, student['chat_template_kwargs'], policy['format'])
        record = {'id': '041:' + directory.name + ':' + str(call), 'episode_id': '041:' + directory.name,
                  'messages': captured, 'target': action_text(step['generation'], case['protocol']),
                  'provenance': {'split': 'train', 'task_id': trace['id'], 'repository': trace['repository'],
                                 'arm': trace['branch'], 'call': call, 'kind': kind, 'producer_kind': 'model',
                                 'model_role': 'base', 'executed': True, 'trajectory_sha256': trajectory_sha,
                                 'request_sha256': step['request_sha256'],
                                 'prefix_sha256': case['screen']['prefix_sha256'],
                                 'target_transform': 'verified_first_edit_action_only'}}
        try:
            encoded = encode_training_example(record, tokenizer, student)
        except ValueError:
            return [], 'student_encoding_rejected'
        if (encoded['ids'][:len(student_prefix)] != student_prefix
                or len(encoded['ids']) > reference['encoding_screen_max_tokens']
                or encoded['supervised_tokens'] <= 0):
            return [], 'untruncated_student_target_does_not_fit'
        record['encoding_receipt'] = {
            'tokens': len(encoded['ids']), 'supervised_tokens': encoded['supervised_tokens'],
            'teacher_input_sha256': served_hash,
            'student_input_sha256': digest(json.dumps(student_prefix, separators=(',', ':')).encode()),
            'student_profile_sha256': reference['reasoning_comparison']['student_profile_sha256']}
        rows.append(record)
    return rows, None


def check_arm(root, data, out, directory, case, item, result, controller_result, journal, reference,
              profiles, tokenizer, normalizer):
    trace_path = directory / 'trajectory.json'
    trace = read(trace_path)
    require(result == controller_result and result['trajectory_sha256'] == sha256(trace_path)
            and trace['complete'] and trace['id'] == item['task_id'] and trace['split'] == 'training'
            and trace['branch'] == result['arm'] == item['arm'],
            'Completed training outcome differs from its controller or trajectory')
    snapshot = case['snapshot']
    restored = read(directory / 'restored-state.json')
    require(restored == {'prefix_sha256': sha256(snapshot / 'prefix.json'),
                         'manifest_sha256': sha256(snapshot / 'manifest.json'),
                         'initial_messages_sha256': message_hash(case['messages'])}
            and sha256(directory / 'before.json') == sha256(snapshot / 'before.json'),
            'Actor restore differs from the registered CPU snapshot')
    task = case['task']
    budget = case['budget']
    teacher, student = profiles
    replay = replay_from_state(directory, trace, task, case['protocol'], teacher, tokenizer, normalizer,
                               case['messages'], case['prefix']['original_source_sha256'])
    with TemporaryDirectory(dir=out / 'private') as temp:
        regraded = recorded_grade(root, data, directory, task, case['source'], read(directory / 'before.json'),
                                  Path(temp) / 'grader')
    final = SubmissionGuard.source_state(directory / 'workspace', task)
    verified = replay['reconstructed_verified_source'] is not None and replay['reconstructed_verified_source'] == final
    last = trace['steps'][-1] if trace['steps'] else {}
    submitted = (last.get('command') == SUBMIT and last.get('observation', {}).get('returncode') == 0
                 and target_is_executed(last))
    compliant = (0 <= trace['actor_seconds'] <= budget['max_task_seconds']
                 and trace['attempted_calls'] <= budget['max_model_calls']
                 and trace['generated_tokens'] <= budget['max_total_generated_tokens'])
    proof = case['screen']['arms'][item['arm']]
    first = trace['steps'][0]['generation'] if trace['steps'] else {}
    require(replay['passed'] and replay['known_generated_tokens'] == result['generated_tokens'] == trace['generated_tokens']
            and replay['known_input_tokens'] == result['input_tokens']
            and sum(not s['action_format_valid'] for s in trace['steps']) == result['invalid_actions']
            and len(trace['steps']) == trace['attempted_calls'] == result['attempted_calls']
            and regraded == trace['grade'] and replay['reconstructed_final_source'] == trace['final_source_sha256'] == final
            and regraded['passed'] == result['grader_passed']
            and verified == trace['agent_verified_final_source'] == result['agent_verified_final_source']
            and submitted == trace['submitted'] == result['submitted']
            and compliant == trace['actor_budget_compliant'] == result['actor_budget_compliant']
            and (regraded['passed'] and compliant and verified and submitted)
                == trace['eligible_verified_trajectory'] == result['complete_success_within_budget']
            and first.get('input_tokens') == proof['input_tokens']
            and first.get('input_tokens_sha256') == proof['input_tokens_sha256'],
            'Replay, grader, budget or first request encoding differs')
    ids = audit_task(directory, trace, result, journal)
    records, reason = encode_targets(directory, trace, case, reference, teacher, student, tokenizer,
                                     result['trajectory_sha256'])
    edits = edited_steps(trace)
    cell = {'audited': True, 'first_edit_call': edits[0]['call'] if edits else None,
            'edit_actions': len(edits), 'correct_patch': regraded['passed'],
            'verified_and_submitted_within_budget': trace['eligible_verified_trajectory'],
            'qualified': bool(records), 'qualification_reason': reason,
            'attempted_calls': trace['attempted_calls'], 'generated_tokens': trace['generated_tokens'],
            'input_tokens': result['input_tokens'], 'seconds': trace['actor_seconds'],
            'timeout_recovered': result['timeout_call'] is not None,
            'invalid_actions': result['invalid_actions']}
    return cell, records, ids


def select_examples(pairs, examples, design, all_audited):
    families = set()
    selected = []
    arms = ('control', 'correction')
    for pair in pairs:
        index = pair['pair']
        for arm in arms:
            if (index, arm) in examples:
                selected.extend(examples[index, arm])
                families.add(pair['family'])
                break
    count = len({row['provenance']['task_id'] for row in selected})
    sufficient = count >= design['minimum_qualified_tasks'] and len(families) >= design['minimum_qualified_families']
    return selected if all_audited and sufficient else [], {'qualified_tasks': count, 'qualified_families': len(families),
                                                             'collection_gate_passed': all_audited and sufficient}


def audit(root, protocol_path):
    from transformers import AutoTokenizer

    root, protocol_path = Path(root).resolve(), Path(protocol_path).resolve()
    plan = read(protocol_path)
    out = root / relative(plan['run_directory'])
    controller = read(out / 'controller.json')
    require(controller.get('backend_stopped') and controller['protocol_sha256'] == sha256(protocol_path),
            'Audit only after this registered backend has stopped')
    require(not (out / 'audit.json').exists(), 'Preserve the original audit')
    reference, data, cases, design = load_cases(root, plan)
    runtime = read(root / 'private/parent-runtime.json')
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'], local_files_only=True, trust_remote_code=False)
    teacher = read(root / relative(reference['evaluation_profile_path']))
    student_path = root / relative(reference['reasoning_comparison']['student_profile_path'])
    require(sha256(student_path) == reference['reasoning_comparison']['student_profile_sha256'], 'Student template differs')
    profiles = teacher, read(student_path)
    normalizer = normalization(root)
    entries = controller['backend_lifecycles']
    journal_for = {e['index']: RequestJournal(out / 'private' / 'backends' / f"{e['index']:04d}-base" / 'requests')
                   for e in entries if e['role'] == 'base' and 'ready_epoch' in e}
    known = {index: set() for index in journal_for}
    saved = {r['run_id']: r for r in controller['runs']}
    pairs = {name: {'pair': n, 'family': case['task']['family'], 'language': case['task']['language'], 'arms': {}}
             for n, (name, case) in enumerate(cases.items(), 1)}
    exports = {}
    for item in plan['schedule']:
        name, arm = item['task_id'], item['arm']
        where = out / 'private' / 'tasks' / relative(item['run_id'])
        pair = pairs[name]
        cell = {'audited': False, 'reason': 'attempt_not_completed'}
        if item['run_id'] in saved:
            try:
                result = read(where / 'result.json')
                index = result['backend_load']
                cell, records, ids = check_arm(root, data, out, where, cases[name], item, result, saved[item['run_id']],
                                               journal_for[index], reference, profiles, tokenizer, normalizer)
                require(not known[index].intersection(ids), 'Generation receipt reused across arms')
                known[index].update(ids)
                if records:
                    exports[pair['pair'], arm] = records
            except (OSError, TypeError, ValueError, KeyError) as error:
                cell = {'audited': False, 'reason': 'audit_failed', 'error_type': type(error).__name__}
                (where / 'audit-error.private.log').write_text(traceback.format_exc(), encoding='utf-8')
        pair['arms'][arm] = cell
    counted = sum(len(v) for v in known.values())
    exact = all({p.name for p in journal.directory.iterdir()} == known[i] for i, journal in journal_for.items())
    environment = read(root / relative(reference['environment_path']))
    same_backend = (len(entries) == len(journal_for) == 1
                    and all(read(out / 'private' / 'backends' / f'{i:04d}-base-environment.json') == environment
                            for i in journal_for))
    all_audited = (controller['complete'] and controller['frozen_files_unchanged']
                   and len(saved) == len(plan['schedule']) and counted == sum(r['attempted_calls'] for r in saved.values())
                   and same_backend and exact
                   and all(a['audited'] for pair in pairs.values() for a in pair['arms'].values()))
    selected, gate = select_examples(list(pairs.values()), exports, design, all_audited)
    manifest = None
    if selected:
        directory = out / 'private' / 'first-edit-export'
        directory.mkdir()
        raw = directory / 'data.jsonl'
        raw.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8')
        manifest = {'data_sha256': sha256(raw), 'records': len(selected), 'raw_data_public': False,
                    'supervised_tokens': sum(row['encoding_receipt']['supervised_tokens'] for row in selected)}
        write_json(directory / 'manifest.json', manifest)
    report = {'schema_version': 1, 'passed': all_audited, 'protocol_sha256': sha256(protocol_path),
              'journal_requests_verified': counted, **gate, 'pairs': list(pairs.values()),
              'export': manifest, 'new_optimizer_updates': 0, 'final_test_opened': False}
    atomic_json(out / 'audit.json', report)
    return report
