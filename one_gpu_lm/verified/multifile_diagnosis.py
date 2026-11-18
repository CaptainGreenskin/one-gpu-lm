"""Summarize observed development failures without exporting solutions as training data."""
import hashlib
import json


def diagnose(bundles, audit, max_calls=12, output_cap=2400):
    if not audit['passed']:
        raise ValueError('An audited development run is required')
    expected = {r['run_id']: r for r in audit['rows']}
    rows = []
    for run_id, bundle in sorted(bundles.items()):
        t = bundle['trace']; previous = expected[run_id]
        actual_hash = hashlib.sha256((json.dumps(t, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()).hexdigest()
        if (bundle['trajectory_sha256'] != previous['trajectory_sha256']
                or actual_hash != bundle['trajectory_sha256']
                or not previous['audit_passed'] or t['split'] != 'development'
                or t.get('training_export_allowed') is not False or not t['complete']):
            raise ValueError('Unverified or non-development trajectory')
        edits = [s['call'] for s in t['steps'] if s.get('source_after') is not None
                 and s['source_after'] != s['source_before']]
        clipped = []
        for s in t['steps']:
            if 'observation' not in s:
                continue
            tool = bundle['tools'][f"tool-{s['call']:03d}"]
            raw = tool['output']
            prefix = raw if len(raw)<=output_cap else raw[:output_cap]+'\n[tool output truncated; use smaller file ranges]'
            if hashlib.sha256(prefix.encode()).hexdigest() != s['normalization']['raw_sha256']:
                raise ValueError('Retained tool prefix disagrees with audited trace')
            # The old normalization receipt sees an already-capped prefix.
            # Count truncation from the original retained tool bytes instead.
            if len(tool['output']) > output_cap:
                clipped.append(s['call'])
        if (len(t['steps']) != previous['attempted_calls']
                or t['submitted'] != previous['submitted']
                or t['grade']['passed'] != previous['grader_passed']):
            raise ValueError('Trace disagrees with published audit')
        first = edits[0] if edits else None
        grade = bool(t['grade']['passed'])
        if grade and not t['submitted'] and first == max_calls:
            category = 'first_edit_on_final_call_no_completion_budget'
        elif not edits:
            category = 'no_edit_before_budget_exhaustion'
        elif grade and t['submitted']:
            category = 'correct_and_submitted'
        else:
            category = 'other'
        rows.append({
            'run_id': run_id, 'task_id': t['id'], 'variant': previous['variant'],
            'trajectory_sha256': bundle['trajectory_sha256'],
            'calls': t['attempted_calls'], 'first_edit_call': first, 'edit_calls': edits,
            'remaining_calls_after_first_edit': None if first is None else max_calls-first,
            'grader_passed': grade, 'submitted': t['submitted'],
            'standalone_verification_of_final_source': t['agent_verified_final_source'],
            'clipped_tool_output_calls': clipped, 'category': category,
            'input_tokens': previous['input_tokens'], 'generated_tokens': t['generated_tokens'],
        })
    return {'schema_version': 1, 'source_experiment': '031', 'rows': rows,
            'new_model_calls': 0, 'final_test_opened': False,
            'development_examples_added_to_training': 0,
            'causal_claim': 'Behavioral diagnosis only; these observations do not isolate a causal training intervention.',
            'limitations': ['The six cases were selected after seeing development results.',
                           'Detailed traces use repeat zero; the original audit checks all three repeats.',
                           'A clipped output is not proof that a specific missing line caused failure.',
                           'A correct patch at the call limit is distinct from explicit verified completion.']}
