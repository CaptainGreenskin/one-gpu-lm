"""Audit the submission-guard calibration; exclude unexecuted requests from SFT targets.

This imports only the tokenizer, never model weights or development/final tasks.
It does not generate, repair, retry, or train. Raw trajectories stay private.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.common.receipts import sha256, write_json
from .tasks import digest


def target_is_executed(step):
    return step['action_format_valid'] and step.get('observation', {}).get('executed') is True


def decision(protocol, rows, controller):
    gate = protocol['feasibility_gate']
    eligible = [r for r in rows if r['eligible_for_sft']]
    counts = Counter(r['repository'] for r in eligible)
    checks = {
        'complete_single_attempt': controller['complete'] and len(rows) == len(protocol['task_ids']),
        'no_unaccounted_generation': all(r['input_replay_passed'] for r in rows),
        'minimum_full_episodes': len(eligible) >= gate['minimum_eligible_full_episodes'],
        'repository_coverage': all(counts[r] >= gate['minimum_per_repository'] for r in gate['repositories']),
        'language_coverage': set(gate['languages']) <= {r['language'] for r in eligible},
        'frozen_inputs_unchanged': controller['frozen_files_unchanged'],
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'eligible_full_episodes': len(eligible), 'eligible_distinct_tasks': len(eligible),
            'eligible_by_repository': dict(sorted(counts.items())),
            'model_quality_improvement_established': False, 'weight_publication_allowed': False}


def run(root, protocol_path, out):
    from transformers import AutoTokenizer
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError('Preserve previous audit')
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol['collection_policy'] != 'explicit_submission_guard_v1':
        raise ValueError('Unexpected collection policy')
    run_dir = root / 'runs/teacher-calibration-001'
    controller = json.loads((run_dir / 'controller.json').read_text())
    if not controller.get('finished_epoch') or not controller['backend_stopped']:
        raise ValueError('Audit only after the controller has stopped')
    runtime = json.loads((root / 'private/parent-runtime.json').read_text())
    profile = json.loads((root / 'experiments/025-verified-distillation/teacher-profile.json').read_text())
    # This is an encoding feasibility screen, not the final student training recipe.
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'], local_files_only=True, trust_remote_code=False)
    tasks = {t['id']: t for t in json.loads((root / 'private/pilot-data/tasks.json').read_text())['tasks']}
    sources = {s['repository']: s for s in json.loads((root / 'private/pilot-data/sources.json').read_text())['sources']}
    rows = []
    for task_id in protocol['task_ids']:
        task = tasks[task_id]; directory = run_dir / 'private/tasks' / task_id
        if not (directory / 'trajectory.json').exists():
            rows.append({'id': task_id, 'repository': task['repository'], 'language': sources[task['repository']]['language'],
                         'eligible_for_sft': False, 'input_replay_passed': False, 'reason': 'not_run'})
            continue
        trajectory = json.loads((directory / 'trajectory.json').read_text())
        replay = not trajectory.get('unknown_inflight_generation_cost', False)
        replay &= len(trajectory['steps']) == trajectory['attempted_calls']
        encoded_actions = []; errors = []; excluded_unexecuted = []
        for step in trajectory['steps']:
            request_path = directory / f"request-{step['call']:03d}.json"
            payload = json.loads(request_path.read_text()); generated = step['generation']; selected = {}
            def encode(messages):
                selected['messages'] = messages
                return encode_prefix(tokenizer, messages, profile['chat_template_kwargs'], protocol['agent_policy']['format'])[0]
            tokens, context = prepare_context(payload['messages'], encode, payload['max_input_tokens'], protocol['agent_policy']['context'])
            replay &= (sha256(request_path) == step['request_sha256'] and len(tokens) == generated['input_tokens']
                       and digest(json.dumps(tokens, separators=(',', ':')).encode()) == generated['input_tokens_sha256']
                       and context == generated['context']
                       and len(generated['ids']) == generated['generated_tokens']
                       and generated['generated_tokens'] <= payload['max_new_tokens']
                       and tokenizer.decode(generated['ids'], skip_special_tokens=True) == generated['text'])
            if not step['action_format_valid']:
                continue  # Preserve malformed output in history, never as a positive target.
            if not target_is_executed(step):
                excluded_unexecuted.append(step['call'])
                continue  # A collection-policy rejection is not a positive executed action.
            try:
                encoded = encode_training_example({'messages': selected['messages'], 'target': generated['text']}, tokenizer, profile)
                if encoded['ids'][:len(tokens)] != tokens:
                    raise ValueError('Serving/training prefix differs')
                if len(encoded['ids']) > protocol['encoding_screen_max_tokens']:
                    raise ValueError('Untruncated target exceeds encoding screen')
                command = step['command']
                kind = ('edit' if step.get('source_after') != step['source_before'] else
                        'verify' if command == task['regression_command'] else
                        'submit' if command == 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' else 'other')
                encoded_actions.append({'call': step['call'], 'kind': kind, 'tokens': len(encoded['ids']),
                                        'supervised_tokens': encoded['supervised_tokens']})
            except ValueError as error:
                errors.append({'call': step['call'], 'reason': str(error)})
        eligible = (trajectory.get('eligible_verified_trajectory', False) and replay and not errors
                    and {'edit', 'verify', 'submit'} <= {r['kind'] for r in encoded_actions})
        rows.append({'id': task_id, 'repository': task['repository'], 'language': sources[task['repository']]['language'],
                     'trajectory_sha256': sha256(directory / 'trajectory.json'),
                     'input_replay_passed': bool(replay), 'eligible_for_sft': bool(eligible),
                     'executed_verified_episode': trajectory.get('eligible_verified_trajectory', False),
                     'attempted_calls': trajectory['attempted_calls'], 'generated_tokens': trajectory['generated_tokens'],
                     'complete': trajectory['complete'], 'encoding_errors': errors, 'encoded_actions': encoded_actions,
                     'excluded_unexecuted_action_calls': excluded_unexecuted})
    result = {'schema_version': 1, 'protocol_sha256': sha256(protocol_path), 'rows': rows,
              'decision': decision(protocol, rows, controller), 'optimizer_updates': 0,
              'collection_policy': protocol['collection_policy'], 'unexecuted_requests_are_positive_targets': False,
              'final_test_opened': False, 'new_development_scores': 0}
    write_json(out, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(run(a.root, a.protocol, a.out)['decision']))
