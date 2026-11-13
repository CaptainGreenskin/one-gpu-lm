"""Freeze all model-affecting024 files before the first repeated generation."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
from _feedback024 import module
schedule=module('schedule');blocks=schedule.blocks;task_schedule=schedule.task_schedule

def freeze(root,commit):
 root=Path(root).resolve();exp=root/'experiments/024-feedback-noise'
 if (exp/'protocol.json').exists() or (root/'runs/noise024').exists():raise ValueError('Freeze once before model runs')
 if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Exact public source commit required')
 plan=json.loads((exp/'plan.json').read_text());verified=json.loads((root/'preparation-verified.json').read_text())
 history=json.loads((root/'historical-review.json').read_text());assert history['passed'] and history['normalizer_sha256']==sha256(exp/'normalize.py')
 assert verified['passed'] and verified['audit_source_sha256']==sha256(exp/'preparation.py')
 tasks=json.loads((root/'private/tasks.json').read_text())['tasks'];ids=[t['id'] for t in tasks]
 assert verified['task_spec_sha256']==sha256(root/'private/tasks.json') and all(t['split']=='dev' for t in tasks)
 paths=['runtime.json','private/tasks.json','private/probes.json','private/inputs.json','parent022.json','preparation-verified.json',
 'runs/conformance/result.json','source-snapshots.json','dependency-snapshots.json','framework-verification.json','runs/admission/admission.json',
 'historical-review.json','parent023.json','experiments/023-evaluation-stability/normalize.py','experiments/011-coding-model-selection/weights-manifest.json','experiments/022-action-repair/profile-action.json']
 paths += [str(p.relative_to(root)) for p in (root/'one_gpu_lm').rglob('*') if p.suffix in {'.py','.mjs'}]
 for d in ['reference020-adapter','action022-adapter']:paths += [str(p.relative_to(root)) for p in (root/'private'/d).iterdir() if p.is_file()]
 paths += [str(exp.relative_to(root)/n) for n in ['_feedback024.py','historical_review.py','plan.json','profile.json','README.md','conformance.py','analysis.py','normalize.py','agent.py','schedule.py','serve.py','run.py','preparation.py','freeze.py','sources.json']]
 paths += [str(p.relative_to(root)) for p in (exp/'notices').glob('*') if p.is_file()]
 value={'schema_version':1,'experiment':'024-feedback-noise','status':'frozen_before_model_rollouts',
 'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_commit':commit,**plan['base'],'agent':plan['agent'],
 'variants':{c:plan['policy'] for c in plan['conditions']},'budget':plan['budget'],
 'task_counts':{'dev':15,'test':0,'train':0},'task_spec_sha256':verified['task_spec_sha256'],'admission_sha256':verified['admission_sha256'],
 'blocks':[{**b,'tasks':task_schedule(ids,b)} for b in blocks()], 'models':plan['models'],'probes':plan['probes'],
 'frozen_files':{n:sha256(root/n) for n in sorted(set(paths))},'training_updates':0,'final_test_opened':False,'weights_public':False,
 'new_model_scores_observed_before_freeze':False,'limitations':plan['limitations']}
 write_json(exp/'protocol.json',value);return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--source-commit',required=True);a=p.parse_args();v=freeze(a.root,a.source_commit);print(json.dumps({'frozen_files':len(v['frozen_files']),'blocks':len(v['blocks']),'final_test_opened':False}))
