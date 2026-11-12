"""Freeze selection code, sources and all cohorts before any model collection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256, write_json
from preparation import verify


def freeze(root, source_commit):
    root = Path(root).resolve(); exp = root/'experiments/021-observed-recovery'
    if (exp/'protocol.json').exists(): raise ValueError('Frozen protocols are immutable')
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Require exact published source commit')
    actual = verify(root); plan = json.loads((exp/'preparation-plan.json').read_text())
    parent = json.loads((root/'experiments/020-repair-curriculum/protocol.json').read_text())
    if plan['status'] != 'rule_locked_after_cpu_admission_before_model_collection':
        raise ValueError('Lock CPU admission and rules before freezing')
    paths = ['runtime.json', 'source-snapshots.json', 'dependency-snapshots.json', 'framework-verification.json',
             'private/tasks.json', 'private/fresh-tasks.json', 'private/inputs.json', 'private/legacy-functions.jsonl',
             'runs/admission/admission.json', 'runs/admission020/admission.json',
             'runs/admission-fresh-draft002/admission.json', 'runs/reference-screen021/screens.json',
             'experiments/011-coding-model-selection/weights-manifest.json', 'environments/pro6000-cu130.json',
             'experiments/020-repair-curriculum/protocol.json']
    for name in ['reference-data020', 'prepared-data020']:
        paths += ['runs/'+name+'/recovery/'+f for f in ['train.jsonl', 'receipt.json']]
    paths += ['runs/prepared-data020/recovery/selection.json']
    paths += [str(p.relative_to(root)) for p in (root/'private/reference020-adapter').iterdir() if p.is_file()]
    paths += [str(p.relative_to(root)) for p in (root/'one_gpu_lm').rglob('*') if p.suffix in {'.py', '.mjs'}]
    paths += [str(p.relative_to(root)) for p in exp.rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.md', '.txt'}]
    protocol = dict(schema_version=1, experiment='021-observed-recovery', status='frozen_before_model_rollouts',
                    frozen_utc=datetime.now(timezone.utc).isoformat(), source_commit=source_commit,
                    **plan['base'], agent=parent['agent'], variants={'shared': plan['shared_recipe']['inference_policy']},
                    budget=plan['budget'], candidates=plan['candidates'], selection=plan['selection'],
                    release_gate=plan['release_gate'], collection=plan['collection'], training_records=330,
                    reference_adapter=plan['reference_adapter'], repository_count=actual['repositories'],
                    task_counts={'train':55, 'dev':15, 'test':72}, task_spec_sha256=actual['task_spec_sha256'],
                    admission_sha256=actual['admission_sha256'], frozen_files={n: sha256(root/n) for n in sorted(set(paths))},
                    derived_data_frozen_before_updates=False, derived_data_rule='Require public verified exact derived-data receipt before any optimizer step.',
                    limitations=plan['limitations'], publication=plan['publication'], failure_policy=plan['failure_policy'],
                    model_scores_observed_before_freeze=False, gpu_training_started_before_freeze=False)
    write_json(exp/'protocol.json', protocol); return protocol


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-commit', required=True); a = p.parse_args()
    print(json.dumps({'frozen_utc': freeze(a.root, a.source_commit)['frozen_utc']}))
