"""Derive numeric, descriptive behavior counts from private audited 020 traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(case):
    actions = case['actions']
    executed = [a for a in actions if isinstance(a.get('tool'), dict)]
    commands = []
    for action in executed:
        assert len(action['actions']) == 1 and set(action['actions'][0]) == {'command'}
        commands.append(action['actions'][0]['command'])
    observed = [a for a in actions if a.get('source_state_after') is not None]
    edited = [a for a in observed if a['source_state_after']['changed_editable_files'] > 0]
    states = [a['source_state_after']['editable_state_sha256'] for a in observed]
    return {
        'tool_observations': len(executed),
        'nonzero_tool_returns': sum(a['tool']['returncode'] != 0 for a in executed),
        'tool_reasons': dict(sorted(Counter(a['tool']['reason'] for a in executed).items())),
        'tool_seconds': sum(a['tool']['seconds'] for a in executed),
        'exact_repeat_commands_beyond_first': len(commands) - len(set(commands)),
        'first_observed_allowed_edit_call': edited[0]['call'] if edited else None,
        'any_observed_allowed_edit': bool(edited),
        'invalid_editable_path_observations': sum(a['source_state_after']['invalid_editable_files'] > 0 for a in observed),
        'observed_source_state_transitions': sum(a != b for a, b in zip(states, states[1:])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    work = args.work
    evidence_path = work / 'private-postrun-behavior.json'
    evidence = json.loads(evidence_path.read_text())
    bundle = work / 'results-review'
    out = bundle / 'execution-behavior-summary.json'
    assert not out.exists(), 'Preserve existing descriptive evidence'
    audit = json.loads((bundle / 'execution-audit.json').read_text())
    selection = json.loads((bundle / 'selection.json').read_text())
    collection = json.loads((bundle / 'collection.json').read_text())
    assert digest(bundle / 'task-inventory.json') == collection['files']['task-inventory.json']
    inventory = {t['id']:t for t in json.loads((bundle / 'task-inventory.json').read_text())}
    assert evidence['posthoc_descriptive_only'] and evidence['raw_private_evidence']
    assert digest(bundle / 'execution-audit.json') == evidence['execution_audit_sha256']
    assert digest(bundle / 'selection.json') == evidence['selection_sha256']
    assert evidence['final_test_opened_by_original_controller'] == selection['final_test_opened']
    arms = ['dev-base', 'dev-repair', 'dev-recovery']
    if selection['selected'] is not None:
        arms += ['final-base', 'final-selected']
    assert set(evidence['arms']) == set(arms)
    rows = []
    aggregates = {}
    for arm in arms:
        campaign_path = bundle / 'campaigns' / (arm + '.json')
        campaign = json.loads(campaign_path.read_text())
        assert campaign['complete'] and digest(campaign_path) == audit['campaign_sha256'][arm]
        scores = {r['id']: r for r in campaign['results']}
        proofs = {r['id']: r for r in audit['variants'][arm]}
        cases = evidence['arms'][arm]
        assert len(cases) == len(scores) and {c['id'] for c in cases} == set(scores)
        current = []
        for case in cases:
            score, proof = scores[case['id']], proofs[case['id']]
            assert case['trajectory_sha256'] == proof['trajectory_sha256']
            assert all(case[k] == score[k] for k in ['passed', 'calls', 'generated_tokens', 'seconds', 'exit_status'])
            assert score['calls'] == len(score['generations']) + len(score['backend_errors'])
            assert [a['call'] for a in case['actions']] == list(range(1, len(score['generations']) + 1))
            row = {k:case[k] for k in ['id', 'passed', 'calls', 'generated_tokens', 'seconds', 'exit_status', 'trajectory_sha256']}
            task = inventory[case['id']]
            assert task['split'] == ('dev' if arm.startswith('dev-') else 'test')
            # Names inherited from older experiments may contain "fresh" or
            # "new"; only the frozen evaluation role defines the cohort.
            row['cohort'] = ('development' if task['split'] == 'dev' else
                            'historical' if task.get('evaluation_role') == 'historical_final_regression_not_selection' else 'fresh')
            row.update(arm=arm, assistant_generations=len(score['generations']),
                       backend_error_calls=len(score['backend_errors']), **describe(case))
            assert row['tool_observations'] == score['tool_calls']
            current.append(row)
        aggregates[arm] = {
            'tasks': len(current), 'passes': sum(r['passed'] for r in current),
            'failed_without_observed_allowed_edit': sum(not r['passed'] and not r['any_observed_allowed_edit'] for r in current),
            'failed_after_observed_allowed_edit': sum(not r['passed'] and r['any_observed_allowed_edit'] for r in current),
            'tasks_with_invalid_editable_paths': sum(r['invalid_editable_path_observations'] > 0 for r in current),
            **{k:sum(r[k] for r in current) for k in ['assistant_generations', 'backend_error_calls', 'tool_observations', 'nonzero_tool_returns', 'tool_seconds', 'exact_repeat_commands_beyond_first']},
        }
        rows.extend(current)
    result = {
        'schema_version': 1, 'posthoc_descriptive_only': True,
        'used_for_candidate_selection_or_release_gates': False,
        'execution_audit_sha256': digest(bundle / 'execution-audit.json'),
        'task_inventory_sha256': digest(bundle / 'task-inventory.json'),
        'private_evidence_sha256': digest(evidence_path),
        'analysis_script_sha256': digest(Path(__file__)),
        'definitions': {
            'repeated_commands': 'Exact equality of executed command strings, beyond the first occurrence within a task. Repeating a test may be justified; this is not automatically waste.',
            'nonzero_returns': 'Observed nonzero tool returns. Deliberately detecting a failing test can be useful; this does not classify an infrastructure failure.',
            'allowed_edit': 'At least one allowed regular source file differs from its starting defective bytes in an observed snapshot. Deleted or symlinked paths are reported separately as invalid paths.',
            'source_transitions': 'Different source-state digests between adjacent observed tool snapshots, excluding the unknown transition before the first observation.',
            'interpretation': 'Observed execution behavior only; not a causal explanation of weight changes or a significance test.',
        },
        'arms': aggregates, 'cases': rows,
        'raw_commands_or_source_patches_public': False,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'audited_cases':len(rows), 'arms':aggregates}))


if __name__ == '__main__':
    main()
