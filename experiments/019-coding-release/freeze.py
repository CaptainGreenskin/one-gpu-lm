"""Freeze actual admitted inputs after preparation; does not start any GPU work."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.evaluation.tasks import file_manifest


def freeze(root,source_commit):
    root=Path(root).resolve();exp=root/'experiments/019-coding-release'
    if (exp/'protocol.json').exists():raise ValueError('Existing frozen protocols are immutable')
    if len(source_commit)!=40 or any(c not in '0123456789abcdef' for c in source_commit):raise ValueError('Require exact source commit')
    status=json.loads((root/'runs/preparation019/status.json').read_text())
    if status.get('stage')!='completed_ready_for_protocol_freeze' or status.get('complete') is not True:
        raise ValueError('Actual data preparation must finish before freezing')
    from preparation import verify
    actual=verify(root,root/'runs/preparation019/freeze-verified.json')
    plan=json.loads((exp/'preparation-plan.json').read_text())
    input_audit=json.loads((root/'training-input-audit.json').read_text())
    if input_audit.get('passed') is not True or input_audit['data_sha256']!=sha256(root/'runs/prepared-data/train.jsonl'):
        raise ValueError('Training input audit must cover this exact prepared data')
    framework=json.loads((root/'framework-verification.json').read_text())
    if (framework.get('all_package_python_files_match_pinned_archive') is not True
            or framework['revision']!='04d809ceab9df28f9adaed044884180159172930'):
        raise ValueError('Installed agent framework must match its pinned source')
    screens=json.loads((root/'runs/reference-screen-final/screens.json').read_text())
    if not screens['complete'] or len(screens['screens'])!=4 or any(s['screen_passed'] is not True for s in screens['screens']):
        raise ValueError('Final dependency reference screens must pass')
    tasks=json.loads((root/'private/tasks.json').read_text())['tasks']
    snapshots={name:file_manifest((root/'sources'/name.replace('/','--')).resolve()) for name in sorted({t['repository'] for t in tasks})}
    write_json(root/'source-snapshots.json',snapshots)
    runtime=json.loads((root/'runtime.json').read_text())
    sites={'historical-default':(root/'.venv-agent/lib/python3.10/site-packages').resolve()}
    sites.update({repo:root/path for repo,path in runtime.get('python_site_overrides',{}).items()})
    # Package contents are recorded separately from version labels. Rebuildable
    # __pycache__ files are omitted; standalone bytecode files remain included.
    dependency_files={}
    for label,site in sites.items():
        dependency_files[label]={str(p.relative_to(site)):sha256(p) for p in sorted(site.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    for directory in sorted((root/'node-deps').iterdir()):
        if directory.is_dir():
            dependency_files['node/'+directory.name]={str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    write_json(root/'dependency-snapshots.json',dependency_files)
    chosen=json.loads((root/'runs/prepared-data/selection.json').read_text())
    budget=plan['budget']
    paths=['runtime.json','source-snapshots.json','dependency-snapshots.json','framework-verification.json','training-input-audit.json','private/audit-training-inputs.py','private/tasks.json','private/inputs.json','private/legacy-functions.jsonl',
           'private/tokenizer-probe.json','private/prepare019.py','private/screen019-final.py','private/task-preparation-changes.json',
           'runs/preparation019/verified.json','runs/preparation019/freeze-verified.json','runs/preparation019/linux-checks.json','runs/preparation019/exclusions.json',
           'runs/reference-screen-final/screens.json','runs/admission/admission.json','runs/reference-data/train.jsonl','runs/reference-data/receipt.json',
           'runs/prepared-data/train.jsonl','runs/prepared-data/receipt.json','runs/prepared-data/selection.json',
           'experiments/011-coding-model-selection/weights-manifest.json','experiments/018-repair-verification/reference_screen.py','environments/pro6000-cu130.json']
    paths += [str(p.relative_to(root)) for p in (root/'one_gpu_lm').rglob('*') if p.suffix in {'.py','.mjs'}]
    paths += [str(p.relative_to(root)) for p in (root/'private').glob('prepare019-*.py')]
    paths += [str(p.relative_to(root)) for p in (root/'runs/preparation019').glob('failed-attempt*/recovery.json')]
    paths += [str(p.relative_to(root)) for p in exp.iterdir() if p.suffix in {'.py','.json'}]
    # Keep the actual preregistration addendum immutable; README status/results
    # may be updated without changing the experiment's executable identity.
    paths.append(str((exp/'preparation-notes.md').relative_to(root)))
    paths += [str(p.relative_to(root)) for p in (exp/'notices').iterdir() if p.is_file()]
    protocol={'schema_version':1,'experiment':'019-coding-release','status':'frozen_before_model_rollouts',
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_commit':source_commit,**plan['base'],
        'agent':{'framework':'SWE-agent/mini-swe-agent','revision':'04d809ceab9df28f9adaed044884180159172930',
            'max_model_calls':budget['max_model_calls'],'max_generated_tokens_per_call':budget['max_generated_tokens_per_call'],
            'max_total_generated_tokens':budget['max_total_generated_tokens'],'max_input_tokens':budget['max_input_tokens'],
            'max_tool_output_chars':2400,'max_tool_seconds':60,'wall_time_limit_seconds':budget['max_task_seconds'],
            'decoding':'greedy','network':'disabled inside tool workspace','shared_uid_process_thread_limit':512},
        'variants':{'shared':plan['shared_recipe']['inference_policy']},'budget':budget,'candidates':plan['candidates'],
        'selection':plan['selection'],'release_gate':plan['release_gate'],'repository_count':len(snapshots),
        'task_counts':dict(Counter(t['split'] for t in tasks)),'training_records':chosen['selected_records'],
        'supervised_tokens':chosen['supervised_tokens'],'prepared_data_sha256':sha256(root/'runs/prepared-data/train.jsonl'),
        'admission_sha256':sha256(root/'runs/admission/admission.json'),'task_spec_sha256':sha256(root/'private/tasks.json'),
        'frozen_files':{name:sha256(root/name) for name in sorted(set(paths))},
        'partition_note':'Exact cohort counts above. 14 original development defects plus one prefreeze no-op revision documented in preparation-notes.md; final historical24 and fresh24 never used in selection.',
        'publication':plan['publication'],'failure_policy':plan['failure_policy'],'limitations':plan['limitations'],
        'model_scores_observed_before_freeze':False,'gpu_training_started_before_freeze':False}
    write_json(exp/'protocol.json',protocol)
    return protocol


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--source-commit',required=True)
    a=p.parse_args();v=freeze(a.root,a.source_commit);print(json.dumps({k:v[k] for k in ['task_counts','training_records','supervised_tokens','frozen_utc']}))
