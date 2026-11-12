"""Freeze exact CPU-derived data, inference budgets and unopened test partitions."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
from preparation import verify


def freeze(root,source_commit):
    root=Path(root).resolve();exp=root/'experiments/022-action-repair'
    if (exp/'protocol.json').exists():raise ValueError('Frozen protocols are immutable')
    if len(source_commit)!=40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Require an exact published source commit')
    if (root/'runs/release022').exists():raise ValueError('Freeze before starting any new model run')
    actual=verify(root);plan=json.loads((exp/'preparation-plan.json').read_text())
    if plan['status']!='rule_locked_after_cpu_admission_before_model_runs':raise ValueError('CPU preparation must be locked')
    parent=json.loads((root/'experiments/021-observed-recovery/protocol.json').read_text())
    paths=['cpu-preparation-timing.json','runtime.json','source-snapshots.json','dependency-snapshots.json','framework-verification.json',
        'private/tasks.json','private/fresh-tasks.json','private/inputs.json','private/legacy-functions.jsonl',
        'private/parent021-controller.json','private/parent021-integrity.json',
        'runs/admission/admission.json','runs/admission020/admission.json',
        'runs/admission-fresh-draft002/admission.json','runs/reference-screen021/screens.json',
        'experiments/011-coding-model-selection/weights-manifest.json','environments/pro6000-cu130.json',
        'experiments/020-repair-curriculum/protocol.json','experiments/021-observed-recovery/protocol.json',
        'experiments/021-observed-recovery/preparation.py','experiments/021-observed-recovery/preparation-plan.json',
        'experiments/021-observed-recovery/profile-observed.json','experiments/021-observed-recovery/sources.json',
        'training-data-audit.json','training-input-audit.json','runs/reference-data022/derived-data.json']
    for name,variant in [('reference-data020','recovery'),('prepared-data020','recovery'),
                         ('reference-data022','action'),('prepared-data022','action')]:
        paths += ['runs/'+name+'/'+variant+'/'+f for f in ['train.jsonl','receipt.json']]
    paths += ['runs/prepared-data020/recovery/selection.json','runs/prepared-data022/action/selection.json']
    paths += [str(p.relative_to(root)) for p in (root/'private/reference020-adapter').iterdir() if p.is_file()]
    paths += [str(p.relative_to(root)) for p in (root/'one_gpu_lm').rglob('*') if p.suffix in {'.py','.mjs'}]
    # Model-affecting sources are immutable. Read-only post-run report/audit code
    # is independently content-addressed in its execution receipts.
    paths += [str(exp.relative_to(root)/name) for name in ['README.md','preparation-plan.json','profile-action.json',
        'train-action.json','sources.json','diagnosis.json','build_data.py','audit_data.py','audit_inputs.py',
        'preparation.py','freeze.py','decision.py','run.py']]
    paths += [str(p.relative_to(root)) for p in (exp/'notices').glob('*.txt')]
    protocol={'schema_version':1,'experiment':'022-action-repair','status':'frozen_before_model_rollouts',
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_commit':source_commit,**plan['base'],
        'agent':parent['agent'],'variants':{'shared':plan['shared_recipe']['inference_policy']},
        'budget':plan['budget'],'candidates':plan['candidates'],'selection':plan['selection'],'release_gate':plan['release_gate'],
        'training_records':330,'reference_adapter':plan['reference_adapter'],'repository_count':20,
        'task_counts':{'train':55,'dev':15,'test':72},'task_spec_sha256':actual['task_spec_sha256'],
        'admission_sha256':actual['admission_sha256'],'frozen_files':{n:sha256(root/n) for n in sorted(set(paths))},
        'prepared_data_sha256':json.loads((root/'runs/reference-data022/derived-data.json').read_text())['prepared_data_sha256'],
        'exact_training_data_frozen_before_updates':True,'fresh_inherited_unopened_from':'021',
        'limitations':plan['limitations'],'publication':plan['publication'],'failure_policy':plan['failure_policy'],
        'new_model_scores_observed_before_freeze':False,'gpu_training_started_before_freeze':False}
    write_json(exp/'protocol.json',protocol);return protocol


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source-commit',required=True)
    a=p.parse_args();print(json.dumps({'frozen_utc':freeze(a.root,a.source_commit)['frozen_utc']}))
