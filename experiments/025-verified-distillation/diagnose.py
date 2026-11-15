"""Post-hoc descriptive trace categories; never alters the frozen pilot gate."""
import argparse
from collections import Counter
import json
from pathlib import Path
from one_gpu_lm.common.receipts import sha256, write_json


def diagnose(root, out, run_name='teacher-pilot-001'):
    root, out = Path(root), Path(out)
    if out.exists(): raise ValueError('Preserve previous diagnosis')
    if run_name not in {'teacher-pilot-001', 'teacher-calibration-001'}:
        raise ValueError('Unknown registered training attempt')
    run = root / 'runs' / run_name
    controller = json.loads((run / 'controller.json').read_text())
    if not controller.get('finished_epoch'): raise ValueError('Wait for terminal controller')
    tasks = {t['id']: t for t in json.loads((root / 'private/pilot-data/tasks.json').read_text())['tasks']}
    rows = []
    for result in controller['tasks']:
        directory = run / 'private/tasks' / result['id']
        trajectory_path = directory / 'trajectory.json'
        if sha256(trajectory_path) != result['trajectory_sha256']: raise ValueError('Trajectory changed')
        t = json.loads(trajectory_path.read_text()); task = tasks[result['id']]
        previous = None; repeats = 0; mentions = 0; exact = 0; truncated = 0; reasons = Counter()
        for step in t['steps']:
            command = step.get('command')
            if command is None: previous = None; continue
            mentions += int(task['regression_command'] in command)
            exact += int(task['regression_command'] == command)
            key = (command, step.get('source_after'), step.get('normalization',{}).get('delivered_sha256'))
            if key == previous: repeats += 1
            previous = key
            tool_path = directory / f"tool-{step['call']:03d}.json"
            if tool_path.exists():
                tool = json.loads(tool_path.read_text())
                truncated += int(len(tool['output']) > 2400)
                reasons[tool['reason']] += 1
        category = ('executed_verified_episode_before_encoding_screen' if result['eligible_verified_trajectory'] else
                    'correct_repair_without_strict_verification' if result['grade']['passed'] and not result['agent_verified_final_source'] else
                    'correct_repair_without_submission' if result['grade']['passed'] else
                    'no_edit' if result['first_edit_call'] is None else 'edited_but_incorrect')
        rows.append({'id':result['id'],'repository':result['repository'],'category':category,
                     'independent_grader_passed':result['grade']['passed'],'strict_verified':result['agent_verified_final_source'],
                     'submitted':result['submitted'],'repeated_command_source_and_feedback':repeats,
                     'invalid_actions':sum(not s['action_format_valid'] for s in t['steps']),
                     'unexecuted_policy_rejections':sum(s.get('observation',{}).get('executed') is False for s in t['steps']),
                     'outputs_at_generation_cap':sum(s['generation']['generated_tokens'] == 512 for s in t['steps']),
                     'regression_command_text_mentions':mentions,'exact_regression_commands':exact,
                     'raw_tool_outputs_exceeding_2400_chars':truncated,'tool_reason_counts':dict(reasons),
                     'trajectory_sha256':result['trajectory_sha256']})
    value = {'schema_version':1,'analysis':'post_hoc_description_only','gate_changed':False,
             'regression_text_mention_proves_execution':False,
             'note':'String presence is not execution proof. Exact-command requirement was preregistered and remains unchanged; compound commands may need richer verification instrumentation in a future protocol.',
             'categories':dict(Counter(r['category'] for r in rows)),'rows':rows}
    write_json(out,value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run-name',default='teacher-pilot-001')
    a=p.parse_args();print(json.dumps(diagnose(a.root,a.out,a.run_name)['categories']))
