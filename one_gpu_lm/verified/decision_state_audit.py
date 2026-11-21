"""Audit complete 043 continuations and export strictly matched action pairs privately."""
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback

from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.reasoning import action_text, settings
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.tool_history import render_history
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.serving.request_journal import RequestJournal
from .decision_state_collection import load_cases, read, require, restoration_receipt
from .evaluation_audit import recorded_grade
from .failure_replay import replay_from_state
from .first_edit_continuation import message_hash
from .guard_audit import target_is_executed
from .resilient_eval_audit import audit_task
from .rollout import normalization
from .submission import SUBMIT, SubmissionGuard
from .tasks import digest, relative


def packed_history(directory, step, teacher, tokenizer, policy):
    payload = read(directory / f"request-{step['call']:03d}.json")
    kwargs, _ = settings(teacher, payload)
    selected = []

    def encode(messages):
        selected[:] = messages
        return encode_prefix(tokenizer, messages, kwargs, policy['format'])[0]

    ids, _ = prepare_context(payload['messages'], encode, payload['max_input_tokens'], policy['context'])
    require(digest(json.dumps(ids, separators=(',', ':')).encode()) == step['generation']['input_tokens_sha256'],
            'Does not reproduce the served input token sequence')
    return selected


def check_arm(root, data, out, directory, case, item, result, journal, teacher, tokenizer, normalizer):
    trace_path = directory / 'trajectory.json'
    trace = read(trace_path)
    require(read(directory / 'result.json') == result and result['trajectory_sha256'] == sha256(trace_path)
            and trace['complete'] and trace['split'] == 'training' and trace['id'] == item['task_id']
            and trace['branch'] == result['arm'] == item['arm']
            and read(directory / 'restored-state.json') == restoration_receipt(case)
            and sha256(directory / 'before.json') == sha256(case.snapshot / 'before.json'),
            'Actor, restored source or controller receipt differs')
    task, budget = case.task, case.budget
    start_source = case.prefix['decision_source_sha256']
    verified_source = case.initial_verified_source
    replay = replay_from_state(directory, trace, task, case.protocol, teacher, tokenizer, normalizer,
                               case.messages, start_source, initial_verified_source=verified_source)
    require(trace['steps'], 'Registered decision had no accounted model response')
    first_step = trace['steps'][0]
    first = first_step['generation']
    proof = case.proof['arms'][item['arm']]
    packed = packed_history(directory, first_step, teacher, tokenizer, case.protocol['agent_policy'])
    with TemporaryDirectory(dir=out / 'private') as temp:
        grade = recorded_grade(root, data, directory, task, case.source, read(directory / 'before.json'),
                               Path(temp) / 'grader')
    final = SubmissionGuard.source_state(directory / 'workspace', task)
    verified = replay['reconstructed_verified_source'] is not None and replay['reconstructed_verified_source'] == final
    last = trace['steps'][-1]
    submitted = (target_is_executed(last) and last['command'] == SUBMIT and last['observation']['returncode'] == 0)
    compliant = (0 <= trace['actor_seconds'] <= budget['max_task_seconds']
                 and trace['attempted_calls'] <= budget['max_model_calls']
                 and trace['generated_tokens'] <= budget['max_total_generated_tokens'])
    successful = grade['passed'] and verified and submitted and compliant
    require(replay['passed'] and replay['reconstructed_final_source'] == trace['final_source_sha256'] == final
            and first_step['source_before'] == start_source and message_hash(packed) == case.proof['packed_messages_sha256']
            and first['input_tokens'] == proof['prompt_tokens'] and first['input_tokens_sha256'] == proof['prompt_ids_sha256']
            and grade == trace['grade'] and grade['passed'] == result['grader_passed']
            and verified == trace['agent_verified_final_source'] == result['agent_verified_final_source']
            and submitted == trace['submitted'] == result['submitted']
            and compliant == trace['actor_budget_compliant'] == result['actor_budget_compliant']
            and successful == trace['eligible_verified_trajectory'] == result['complete_success_within_budget']
            and len(trace['steps']) == trace['attempted_calls'] == result['attempted_calls']
            and replay['known_generated_tokens'] == trace['generated_tokens'] == result['generated_tokens']
            and replay['known_input_tokens'] == result['input_tokens']
            and sum(not step['action_format_valid'] for step in trace['steps']) == result['invalid_actions'],
            'Token history, accounting, restored verification, or independent grade differs')
    ids = audit_task(directory, trace, result, journal)
    first_edit = next((s['call'] for s in trace['steps'] if target_is_executed(s) and s.get('source_after')
                       and s['source_after'] != s['source_before']), None)
    cell = {'audited': True, 'first_edit_call': first_edit,
            'submit_call': last['call'] if submitted else None, 'independently_correct': grade['passed'],
            'verified_and_submitted': successful, 'attempted_calls': result['attempted_calls'],
            'input_tokens': result['input_tokens'], 'generated_tokens': result['generated_tokens'],
            'actor_seconds': result['actor_seconds'], 'grader_seconds': result['grader_seconds'],
            'invalid_actions': result['invalid_actions'], 'timeout_recovered': result['timeout_call'] is not None}
    return cell, trace, ids


def earliest_comparable_fork(traces, histories, budget):
    """Compare actual executed commands at the first common state and budget where they differ."""
    generated = [0, 0]
    for a, b in zip(*(trace['steps'] for trace in traces)):
        if (a['call'] == b['call'] and generated[0] == generated[1] < budget['max_total_generated_tokens']
                and target_is_executed(a) and target_is_executed(b) and a['command'] != b['command']
                and a['source_before'] == b['source_before'] and a['protected_before'] and b['protected_before']
                and a.get('protected_after') and b.get('protected_after')):
            first, second = histories[0](a), histories[1](b)
            if render_history(first, True) == render_history(second, True):
                return (a, b), first, budget['max_total_generated_tokens'] - generated[0]
        generated[0] += a['generation']['generated_tokens']
        generated[1] += b['generation']['generated_tokens']
    return None


def contrast(case, directories, traces, teacher, student, tokenizer):
    arms = ('control', 'correction')
    results = [traces[arm]['eligible_verified_trajectory'] for arm in arms]
    if results[0] == results[1]:
        return None, 'same_completion_outcome'
    policy = case.protocol['agent_policy']

    def history(arm):
        return lambda step: packed_history(directories[arm], step, teacher, tokenizer, policy)

    fork = earliest_comparable_fork([traces[arm] for arm in arms], [history(arm) for arm in arms], case.budget)
    if fork is None:
        return None, 'no_matched_executed_fork'
    steps, messages, remaining = fork
    winner = results.index(True)
    positive, negative = steps[winner], steps[1 - winner]
    chosen, rejected = arms[winner], arms[1 - winner]
    final = traces[chosen]['final_source_sha256']
    unchanged = case.prefix['decision_source_sha256']
    if case.kind == 'early_edit':
        if (positive['source_before'] != unchanged or positive['source_after'] != final
                or final == unchanged or positive['observation']['returncode'] != 0):
            return None, 'winning_action_not_the_verified_first_edit'
    elif (positive['command'] != SUBMIT or positive['observation']['returncode'] != 0
          or positive['source_before'] != positive['source_after'] or final != unchanged
          or positive['source_after'] != case.initial_verified_source):
        return None, 'winning_action_not_submission_of_restored_verification'
    if any(not step.get('protected_after') for step in traces[chosen]['steps'] if 'observation' in step):
        return None, 'winning_trajectory_modified_protected_files'

    targets = [action_text(step['generation'], case.protocol) for step in (positive, negative)]
    try:
        encodings = [encode_training_example({'messages': messages, 'target': value}, tokenizer, student)
                     for value in targets]
    except ValueError:
        return None, 'student_template_rejected'
    prefix = encodings[0]['prefix_tokens']
    if (any(len(row['ids']) > student['training']['max_sequence_length'] for row in encodings)
            or encodings[1]['prefix_tokens'] != prefix or encodings[0]['ids'][:prefix] != encodings[1]['ids'][:prefix]
            or encodings[0]['ids'] == encodings[1]['ids']):
        return None, 'student_pair_encoding_not_shared_or_too_long'
    call = positive['call']
    record = {'messages': messages, 'chosen': targets[0], 'rejected': targets[1], 'kind': case.kind,
              'provenance': {'task_id': case.task['id'], 'split': 'train', 'family': case.task['family'],
                             'prefix_sha256': case.proof['prefix_sha256'],
                             'chosen_arm': chosen, 'rejected_arm': rejected, 'fork_call': call,
                             'chosen_request_sha256': positive['request_sha256'],
                             'rejected_request_sha256': negative['request_sha256'],
                             'remaining_calls': case.budget['max_model_calls'] - call + 1,
                             'remaining_generated_tokens': remaining},
              'encoding_receipt': {'prefix_tokens': prefix, 'supervised_tokens':
                                   [row['supervised_tokens'] for row in encodings],
                                   'ids_sha256': [digest(json.dumps(row['ids'], separators=(',', ':')).encode())
                                                  for row in encodings]}}
    return record, 'matched_executed_contrast'


def audit(root, protocol_path):
    from transformers import AutoTokenizer

    plan = read(protocol_path)
    design, reference, data, cases = load_cases(root, plan)
    out = root / relative(plan['run_directory'])
    audit_path = out / 'audit.json'
    require(not audit_path.exists(), 'Preserve the original collection audit')
    controller_path = out / 'controller.json'
    controller = read(controller_path)
    require(controller['protocol_sha256'] == sha256(protocol_path) and controller['backend_stopped']
            and controller['finished_epoch'] and controller['final_test_opened'] is False
            and controller['optimizer_updates'] == 0, 'Audit only the stopped training-only collection')
    model = read(root / 'private/parent-runtime.json')['model']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    teacher = read(root / relative(reference['evaluation_profile_path']))
    student = read(root / relative(design['student_profile_path']))
    normalizer = normalization(root)
    entries = controller['backend_lifecycles']
    backends = out / 'private' / 'backends'
    journals = {e['index']: RequestJournal(backends / f"{e['index']:04d}-base" / 'requests')
                for e in entries if 'ready_epoch' in e and e['role'] == 'base'}
    seen = {n: set() for n in journals}
    runs = {r['run_id']: r for r in controller['runs']}
    pairs, records = [], []
    for number, (name, case) in enumerate(cases.items(), 1):
        cells, traces, directories = {}, {}, {}
        for arm in ('control', 'correction'):
            run_id = f'{name}--{arm}'
            item = {'task_id': name, 'arm': arm, 'run_id': run_id}
            where = out / 'private' / 'tasks' / relative(run_id)
            if run_id not in runs:
                cells[arm] = {'audited': False, 'reason': 'attempt_not_completed'}
                continue
            result = runs[run_id]
            try:
                index = result['backend_load']
                cell, trace, ids = check_arm(root, data, out, where, case, item, result, journals[index],
                                             teacher, tokenizer, normalizer)
                require(not ids.intersection(seen[index]), 'Journal response reused across arms')
                seen[index].update(ids)
                cells[arm], traces[arm], directories[arm] = cell, trace, where
            except (ValueError, OSError, TypeError, KeyError) as error:
                cells[arm] = {'audited': False, 'reason': 'audit_failed', 'error_type': type(error).__name__}
                (where / 'audit-error.private.log').write_text(traceback.format_exc(), encoding='utf-8')
        record, reason = None, 'missing_audited_pair'
        if len(traces) == 2:
            record, reason = contrast(case, directories, traces, teacher, student, tokenizer)
            if record is not None:
                records.append(record)
        pairs.append({'pair': number, 'kind': case.kind, 'family': case.task['family'],
                      'language': case.task['language'], 'available_calls': case.budget['max_model_calls'],
                      'arms': cells, 'contrast': reason})

    audited = sum(cell['audited'] for pair in pairs for cell in pair['arms'].values())
    received = sum(len(ids) for ids in seen.values())
    expected_env = read(root / relative(reference['environment_path']))
    passed = (controller['complete'] and controller['frozen_files_unchanged']
              and len(entries) == len(journals) == 1 and len(runs) == len(controller['runs']) == audited == len(plan['schedule'])
              and received == sum(r['attempted_calls'] for r in runs.values())
              and all({p.name for p in journal.directory.iterdir()} == seen[n]
                      and read(backends / f'{n:04d}-base-environment.json') == expected_env
                      for n, journal in journals.items()))
    if passed and records:
        export = out / 'private' / 'preference-pairs.json'
        atomic_json(export, records)
    else:
        export = None
    summary = {'schema_version': 1, 'passed': passed, 'audited_attempts': audited,
               'expected_attempts': len(plan['schedule']), 'journal_requests_verified': received,
               'protocol_sha256': sha256(protocol_path), 'controller_sha256': sha256(controller_path),
               'eligible_contrasts': dict(Counter(row['kind'] for row in records)),
               'training_export_sha256': sha256(export) if export else None, 'pairs': pairs,
               'training_only': True, 'optimizer_updates': 0, 'final_test_opened': False}
    atomic_json(audit_path, summary)
    return summary
