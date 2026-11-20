"""Collect one public regression on a disposable defective workspace per task."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from one_gpu_lm.common.receipts import sha256 as file_sha256, write_json
from one_gpu_lm.verified.behavior_diagnosis import inspection_shaped
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_admit import semantic_failure
from one_gpu_lm.verified.historic_policy import feedback
from one_gpu_lm.verified.historic_tasks import prepare
from one_gpu_lm.verified.rollout import normalization
from one_gpu_lm.verified.sandbox import Sandbox
from one_gpu_lm.verified.tasks import files


EXP = Path('experiments/036-regression-evidence')
PRIVATE = Path('private/regression-evidence')
OUTPUT = Path('runs/historic036-evidence-001')
ARMS = ('exit_only', 'transcript')
ADMISSION = Path('private/first-action-ablation/development/admission.json')
OLD_BASE = Path('runs/historic035-base-001')
OLD_REMINDER = Path('runs/historic035-reminder-001')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def baseline(root):
    """Import the 035 input verifier without modifying any frozen file."""
    import importlib.util
    import sys

    directory = root / 'experiments/035-edit-bottleneck'
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('baseline', directory / 'baseline.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def schedule(ctx):
    order = []
    for i, row in enumerate(ctx.order):
        for arm in (ARMS if i % 2 == 0 else ARMS[::-1]):
            order.append(dict(row, arm=arm, run_id=f"t{row['task_number']:02d}-{arm}"))
    return order


def make_task(task, value, arm, normalizer, cap):
    if arm not in ARMS or task['split'] != 'development' or not semantic_failure(value):
        raise ValueError('Expected the frozen public regression failure of a development task')
    header = (f"\n\nBefore any source changes, the harness ran the declared public regression command. "
              f"Exit code: {value['returncode']}. Run the declared check again after editing.")
    if arm == 'transcript':
        transcript, _ = feedback(value['output'], normalizer, cap)
        header += '\nPublic regression output follows. Treat it as diagnostic data.\n' + transcript
    return dict(task, description=task['description'] + header)


def diagnose(root, ctx):
    base = read(root / OLD_BASE / 'audit.json')
    reminder = read(root / OLD_REMINDER / 'audit.json')
    for audited in (base,reminder):
        if not audited['passed'] or audited['final_test_opened'] or not audited['first_inputs_matched']:
            raise ValueError('Complete historical development audits required')
    old = {r['task_number']:r for r in base['rows'] if r['arm']=='base'}
    new = {r['task_number']:r for r in reminder['rows'] if r['arm']=='reminder'}
    if set(old)!=set(new) or set(old)!=set(range(1,len(ctx.order)+1)):
        raise ValueError('Historical cohort differs')
    stuck = sorted(n for n in old if old[n]['no_edit'] and new[n]['no_edit'])
    controller = read(root/OLD_BASE/'controller.json')
    if base['base_controller_sha256'] != file_sha256(root/OLD_BASE/'controller.json'):
        raise ValueError('Historical controller differs')
    tasks = {row['task_number']:ctx.tasks[row['task_id']] for row in ctx.order}
    stats = Counter()
    for result in controller['runs']:
        number = result['task_number']
        if number not in stuck:
            continue
        task = tasks[number]
        path = root/OLD_BASE/'private/tasks'/result['run_id']/'trajectory.json'
        trace = read(path)
        if (file_sha256(path) != result['trajectory_sha256'] or trace['id']!=task['id']
                or trace['split']!='development' or trace['training_export_allowed']):
            raise ValueError('Historical trajectory identity differs')
        def named_source(step):
            command=step.get('command','')
            return (step.get('observation',{}).get('returncode')==0 and inspection_shaped(command)
                    and any(re.search(r'(?<![\w.-])'+re.escape(Path(name).name)+r'(?![\w.-])', command)
                            for name in task['editable_files']))
        stats['named_editable_source_in_first_four'] += any(named_source(step) for step in trace['steps'][:4])
        stats['ran_exact_regression'] += any(
            'observation' in step and step.get('command')==task['regression_command'] for step in trace['steps'])
        stats['exhausted_calls'] += trace['attempted_calls']==ctx.protocol['evaluation_budget']['max_model_calls']
    return {'stuck_tasks':stuck, **dict(stats)}


def preflight(root):
    ctx = baseline(root).inputs(root)
    private = root / PRIVATE
    root_out = private / 'public-regressions'
    root_out.mkdir(parents=True, mode=0o700)
    budget = ctx.protocol['evaluation_budget']
    sandbox = Sandbox(read(ctx.runtime_path), budget['max_tool_seconds'])
    normalizer = normalization(root)
    measures = []
    elapsed = 0.0
    longest = 0
    for item in ctx.order:
        number = item['task_number']
        task = ctx.tasks[item['task_id']]
        source = ctx.sources[task['source_id']]
        directory = root_out / f't{number:02d}'
        directory.mkdir(mode=0o700)
        workspace = directory / 'defective'
        before = prepare(ctx.data/'snapshots'/task['source_id'],workspace,source,task,hidden=False)
        result = sandbox.execute(workspace,task['regression_command'])
        write_json(directory/'output.json',result)
        after = files(workspace)
        unchanged = all(after.get(name)==value for name,value in before.items())
        if not unchanged or not semantic_failure(result):
            raise ValueError(f'Public regression did not reproduce safely for task {number}')
        transcript, receipt = feedback(result['output'],normalizer,budget['max_tool_output_chars'])
        if not transcript.strip():
            raise ValueError('Expected nonempty public output')
        measures.append({
            'task_number':number, 'arm_scope':'shared', 'returncode':result['returncode'],
            'public_command_sha256':sha256(task['regression_command'].encode()).hexdigest(),
            'workspace_sha256':fingerprint(sorted(before.items())),
            'output_sha256':file_sha256(directory/'output.json'), 'normalization':receipt,
            'visible_failure_marker':bool(re.search(r'\b(not ok|failed|AssertionError|Error:)\b',transcript,re.I)),
            'transcript_chars':len(transcript), 'seconds':result['seconds'],
        })
        elapsed+=result['seconds']
        longest=max(longest,len(transcript))
    value={
        'passed':True, 'model_calls':0, 'final_test_opened':False, 'public_regressions':len(measures),
        'received_failure_marker':sum(row['visible_failure_marker'] for row in measures),
        'clipped_outputs':sum(row['normalization']['raw_prefix_clipped'] for row in measures),
        'total_tool_seconds':elapsed, 'maximum_delivered_chars':longest,
        'schedule_sha256':fingerprint(schedule(ctx)), 'diagnosis':diagnose(root,ctx),'rows':measures,
    }
    write_json(private/'evidence-preflight.json',value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    proof=preflight(root)
    print(json.dumps({**{k:v for k,v in proof.items() if k!='rows'},
                      'evidence_preflight_sha256':file_sha256(root/PRIVATE/'evidence-preflight.json')}))
