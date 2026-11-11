"""Freeze paired 020 execution inputs after actual preparation, never start a model."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.evaluation.tasks import file_manifest


def freeze(root, source_commit):
    root = Path(root).resolve(); exp = root/'experiments/020-repair-curriculum'
    if (exp/'protocol.json').exists(): raise ValueError('Frozen protocols are immutable')
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Require an exact published source commit')
    read = lambda p: json.loads(Path(p).read_text())
    status = read(root/'runs/preparation020/status.json')
    if status.get('complete') is not True or status['stage'] != 'completed_ready_for_protocol_freeze':
        raise ValueError('Actual Linux data preparation must finish first')
    plan = read(exp/'preparation-plan.json')
    if plan['status'] != 'rule_locked_after_data_admission_before_model_rollouts':
        raise ValueError('Resolve and lock the actual preparation addendum first')
    from preparation import verify
    actual = verify(root,root/'runs/preparation020/freeze-verified.json')
    audit = read(root/'training-input-audit.json')
    hashes = {n:sha256(root/'runs/prepared-data020'/n/'train.jsonl') for n in ['repair','recovery']}
    if audit['passed'] is not True or audit['data_sha256'] != hashes:
        raise ValueError('Input audit does not cover both exact candidate datasets')
    parent = read(root/'experiments/019-coding-release/protocol.json')
    framework = read(root/'framework-verification.json')
    if not framework['all_package_python_files_match_pinned_archive'] or framework['revision'] != parent['agent']['revision']:
        raise ValueError('Pinned agent framework identity changed')
    tasks = read(root/'private/tasks.json')['tasks']
    sources = {n:file_manifest((root/'sources'/n.replace('/','--')).resolve()) for n in sorted({t['repository'] for t in tasks})}
    if sources != read(root/'source-snapshots.json'):
        raise ValueError('Inherited source snapshot contents changed')
    runtime = read(root/'runtime.json')
    sites = {'historical-default':(root/'.venv-agent/lib/python3.10/site-packages').resolve()}
    sites.update({r:root/p for r,p in runtime.get('python_site_overrides',{}).items()})
    dependencies = {label:{str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts} for label,directory in sites.items()}
    for directory in sorted((root/'node-deps').iterdir()):
        if directory.is_dir(): dependencies['node/'+directory.name] = {str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    if dependencies != read(root/'dependency-snapshots.json'):
        raise ValueError('Inherited agent-visible dependencies changed')
    screen_path = root/'runs/reference-screen-final/screens.json'
    if sha256(screen_path) != parent['frozen_files']['runs/reference-screen-final/screens.json']:
        raise ValueError('Inherited reference-exposure screen identity changed')
    screens = read(screen_path)
    if not screens['complete'] or len(screens['screens']) != 4 or not all(s['screen_passed'] for s in screens['screens']):
        raise ValueError('Missing the four final-repository reference-exposure screens')
    paths = ['runtime.json','source-snapshots.json','dependency-snapshots.json','framework-verification.json','training-input-audit.json',
        'private/tasks.json','private/failure-plan.json','private/inputs.json','private/legacy-functions.jsonl',
        'private/prepare020.py','private/probe_targets020.py','private/audit-needles.json',
        'runs/preparation020/verified.json','runs/preparation020/freeze-verified.json','runs/preparation020/linux-checks.json',
        'runs/reference-screen-final/screens.json','runs/admission/admission.json','runs/static-targets020/lengths.json',
        'experiments/011-coding-model-selection/weights-manifest.json','environments/pro6000-cu130.json',
        'experiments/019-coding-release/protocol.json','experiments/019-coding-release/preparation.py','experiments/019-coding-release/profile-low.json']
    for variant in ['repair','recovery']:
        paths += ['runs/reference-data020/'+variant+'/'+name for name in ['train.jsonl','receipt.json']]
        paths += ['runs/prepared-data020/'+variant+'/'+name for name in ['train.jsonl','receipt.json','selection.json']]
    paths += [str(p.relative_to(root)) for p in (root/'one_gpu_lm').rglob('*') if p.suffix in {'.py','.mjs'}]
    paths += [str(p.relative_to(root)) for p in exp.iterdir() if p.suffix == '.json']
    # Post-run reporting/export inspection can be corrected independently. It is
    # content-addressed in final receipts and never changes the frozen evaluator.
    paths += [str((exp/n).relative_to(root)) for n in ['run.py','audit.py','audit_inputs.py','freeze.py','preparation.py','decision.py','preparation-notes.md']]
    paths += [str(p.relative_to(root)) for p in (exp/'notices').iterdir() if p.is_file()]
    chosen = {n:read(root/'runs/prepared-data020'/n/'selection.json') for n in ['repair','recovery']}
    if any(v['selected_records'] != 330 for v in chosen.values()): raise ValueError('Changed update budget')
    budget = plan['budget']
    protocol = {'schema_version':1,'experiment':'020-repair-curriculum','status':'frozen_before_model_rollouts',
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_commit':source_commit,**plan['base'],
        'agent':dict(parent['agent']), 'variants':{'shared':plan['shared_recipe']['inference_policy']},
        'budget':budget,'candidates':plan['candidates'],'selection':plan['selection'],'release_gate':plan['release_gate'],
        'repository_count':len(sources),'task_counts':dict(Counter(t['split'] for t in tasks)),
        'training_records':330,'supervised_tokens':{n:v['supervised_tokens'] for n,v in chosen.items()},
        'prepared_data_sha256':hashes,'shared_identical_records':actual['shared_identical_records'],
        'contrasting_records':actual['contrasting_records'],'semantic_order_sha256':actual['semantic_order_sha256'],
        'admission_sha256':sha256(root/'runs/admission/admission.json'),'task_spec_sha256':sha256(root/'private/tasks.json'),
        'frozen_files':{n:sha256(root/n) for n in sorted(set(paths))},
        'partition_note':'Unchanged 019 training/development/final partitions; no final model scores opened. Development was observed repeatedly and is not independent.',
        'parent_reference_screens_reused_with_identical_source_and_dependency_contents':True,
        'publication':plan['publication'],'failure_policy':plan['failure_policy'],'limitations':plan['limitations'],
        'model_scores_observed_before_freeze':False,'gpu_training_started_before_freeze':False}
    write_json(exp/'protocol.json',protocol); return protocol


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--source-commit',required=True)
    args=parser.parse_args(); value=freeze(args.root,args.source_commit); print(json.dumps({k:value[k] for k in ['frozen_utc','training_records','supervised_tokens']}))
