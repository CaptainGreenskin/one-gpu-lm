"""Post-hoc paired calibration description; never changes a data or quality gate."""
import argparse
from collections import Counter
import json
from pathlib import Path
from one_gpu_lm.common.receipts import sha256, write_json


def shared_prefix(baseline, candidate):
    """A rejection after call n can affect call n+1, not the generation of n."""
    policy_call = next((s['call'] for s in candidate['steps']
                        if s.get('observation', {}).get('executed') is False), None)
    shared = 0
    difference = None
    for left, right in zip(baseline['steps'], candidate['steps']):
        if left['call'] != right['call']:
            raise ValueError('Nonaligned call indices')
        if policy_call is not None and right['call'] > policy_call:
            break
        a, b = left['generation'], right['generation']
        input_equal = a['input_tokens_sha256'] == b['input_tokens_sha256']
        output_equal = a['ids'] == b['ids']
        if not input_equal or not output_equal:
            difference = {'call': right['call'], 'input_equal': input_equal,
                          'output_equal': output_equal, 'before_policy_feedback': True}
            break
        shared += 1
    return {'first_rejected_call': policy_call,
            'identical_shared_calls_before_policy_feedback': shared,
            'difference_before_policy_feedback': difference,
            'classification': 'difference_before_policy_feedback' if difference else
                'shared_prefix_identical_through_first_rejection' if policy_call is not None and shared >= policy_call
                else 'shared_prefix_ends_before_rejection' if policy_call is not None
                else 'shared_prefix_identical_without_rejection',
            'trajectory_step_count_difference': len(candidate['steps']) - len(baseline['steps'])}


def pair_receipts(baseline, candidate, baseline_audit, candidate_audit):
    left = {r['id']: r for r in baseline['tasks']}
    right = {r['id']: r for r in candidate['tasks']}
    audits = [{r['id']: r for r in a['rows']} for a in [baseline_audit, candidate_audit]]
    ids = list(audits[0])
    if set(ids) != set(audits[1]):
        raise ValueError('Registered task sets differ')
    pairs, excluded = [], []
    for task_id in ids:
        a, b = left.get(task_id), right.get(task_id)
        if not a or not b or not a['complete'] or not b['complete']:
            excluded.append({'id': task_id, 'reason': 'missing_or_incomplete_task'})
            continue
        if not audits[0][task_id]['input_replay_passed'] or not audits[1][task_id]['input_replay_passed']:
            excluded.append({'id': task_id, 'reason': 'input_replay_not_verified'})
            continue
        if a['repository'] != b['repository'] or a['original_source_sha256'] != b['original_source_sha256']:
            raise ValueError('Paired source identity differs')
        row = {'id': task_id, 'repository': a['repository']}
        for prefix, result, audit in [('original', a, audits[0][task_id]), ('guard', b, audits[1][task_id])]:
            row.update({prefix + '_' + k: v for k, v in {
                'correct': result['grade']['passed'], 'submitted': result['submitted'],
                'qualified_episode': audit['eligible_for_sft'], 'calls': result['attempted_calls'],
                'generated_tokens': result['generated_tokens'], 'seconds': result['seconds']}.items()})
        pairs.append(row)
    return pairs, excluded


def attempt_cost(controller, audit):
    return {'all_observed_attempted_calls': sum(r.get('attempted_calls', 0) for r in audit['rows']),
            'all_observed_generated_tokens': sum(r.get('generated_tokens', 0) for r in audit['rows']),
            'pipeline_seconds': controller['finished_epoch'] - controller['started_epoch'],
            'unreplayed_attempts_with_uncertain_cost': [r['id'] for r in audit['rows']
                if r.get('reason') != 'not_run' and not r['input_replay_passed']],
            'not_run_tasks': [r['id'] for r in audit['rows'] if r.get('reason') == 'not_run'],
            'incomplete_attempt_costs_are_not_assumed_zero': True}


def run(root, out):
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError('Preserve previous paired analysis')
    dirs = [root / 'runs' / n for n in ['teacher-pilot-001', 'teacher-calibration-001']]
    controllers = [json.loads((d / 'controller.json').read_text()) for d in dirs]
    audits = [json.loads((d / 'audit.json').read_text()) for d in dirs]
    if not all(c.get('finished_epoch') and c['backend_stopped'] for c in controllers):
        raise ValueError('Wait for both terminal controllers and audits')
    pairs, excluded = pair_receipts(*controllers, *audits)
    prefixes = []
    for pair in pairs:
        traces = []
        for directory, controller in zip(dirs, controllers):
            p = directory / 'private/tasks' / pair['id'] / 'trajectory.json'
            receipt = next(r for r in controller['tasks'] if r['id'] == pair['id'])
            if sha256(p) != receipt['trajectory_sha256']:
                raise ValueError('Trajectory receipt changed')
            traces.append(json.loads(p.read_text()))
        prefixes.append({'id': pair['id'], **shared_prefix(*traces)})
    cost = [attempt_cost(c, a) for c, a in zip(controllers, audits)]
    value = {'schema_version': 1, 'analysis': 'post_hoc_training_only_calibration', 'gate_changed': False,
             'complete_pairs': len(pairs), 'excluded_pairs': excluded, 'pairs': pairs,
             'repair_gains': [r['id'] for r in pairs if not r['original_correct'] and r['guard_correct']],
             'repair_regressions': [r['id'] for r in pairs if r['original_correct'] and not r['guard_correct']],
             'all_attempt_costs_original_then_guard': cost, 'prefix_audit': prefixes,
             'prefix_categories': dict(Counter(r['classification'] for r in prefixes)),
             'independent_model_quality_improvement_established': False,
             'limitation': 'Same observed training tasks, sequential runs without randomized interleaving. '
                           'Ordinary failures remain paired. Incomplete or unaudited rows are explicitly excluded '
                           'from paired outcomes, never from total cost disclosure.'}
    write_json(out, value)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    value = run(a.root, a.out)
    print(json.dumps({'complete_pairs': value['complete_pairs'], 'prefix_categories': value['prefix_categories']}))
