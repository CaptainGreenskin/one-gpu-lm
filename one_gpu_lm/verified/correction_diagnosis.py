"""Read-only eight-case diagnosis; publish counts and hashes, never actor text."""
from collections import Counter
import argparse
import json
from pathlib import Path
from one_gpu_lm.common.receipts import sha256, write_json


def failed_ids(audit):
    if not audit['passed']:
        raise ValueError('Parent audit failed')
    return [r['id'] for r in audit['rows'] if r['disposition'] == 'paired'
            and not r['student']['success_within_budget'] and not r['teacher']['success_within_budget']]


def diagnose(trace, regression):
    edits = []; exact_failures = []; edit_errors = []; repeats = []; malformed = []; capped = []
    seen = set()
    for s in trace['steps']:
        call = s['call']; obs = s.get('observation', {})
        edited = s.get('source_after') and s['source_after'] != s['source_before']
        if edited:
            edits.append(call)
            if obs.get('executed') is True and obs.get('returncode') != 0:
                edit_errors.append(call)
        if (s.get('command') == regression and obs.get('executed') is True and obs.get('returncode') != 0
                and s['source_before'] != trace['original_source_sha256'] and not edited):
            exact_failures.append(call)
        if not s['action_format_valid']:
            malformed.append(call)
        if s['generation']['generated_tokens'] >= 512:
            capped.append(call)
        if s.get('command'):
            key = (s['command'], json.dumps(s['source_before'], sort_keys=True))
            if key in seen:
                repeats.append(call)
            seen.add(key)
    return {'attempted_calls': trace['attempted_calls'], 'edit_calls': edits,
            'standalone_failed_checks_after_edit': exact_failures,
            'edit_commands_with_nonzero_exit': edit_errors, 'repeated_exact_command_same_source_calls': repeats,
            'malformed_calls': malformed, 'generated_at_parent_512_token_cap_calls': capped,
            'terminal_reason': trace['terminal_reason'], 'success': trace['success_within_budget']}


def run(root, out):
    root = Path(root); parent = root/'runs/recovery026-001'
    audit_path = parent/'audit.json'; audit = json.loads(audit_path.read_text()); ids = failed_ids(audit)
    if len(ids) != 8:
        raise ValueError('Expected the eight completed both-failed training tasks')
    tasks = {t['id']: t for t in json.loads((root/'private/training-pool-data/tasks.json').read_text())['tasks']}
    rows = []
    for name in ids:
        task = tasks[name]
        if task['split'] != 'training': raise ValueError('Training only')
        r = {'id': name, 'family': task['family'], 'split': 'training'}
        for branch in ['student', 'teacher']:
            p = parent/'private/tasks'/name/branch/'trajectory.json'
            expected = next(x for x in audit['rows'] if x['id'] == name)[branch]['trajectory_sha256']
            if sha256(p) != expected: raise ValueError('Parent trace changed')
            r[branch] = dict(diagnose(json.loads(p.read_text()), task['regression_command']), trajectory_sha256=expected)
        rows.append(r)
    result = {'schema_version': 1, 'source_audit_sha256': sha256(audit_path), 'rows': rows,
              'training_tasks': len(rows), 'both_no_edit_tasks': sum(not r['student']['edit_calls'] and not r['teacher']['edit_calls'] for r in rows),
              'new_model_calls': 0, 'optimizer_updates': 0, 'test_commands_reexecuted': 0, 'final_test_opened': False}
    write_json(out, result); return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();v=run(a.root,a.out);print(json.dumps({'training_tasks':v['training_tasks'],'new_model_calls':0}))
