"""Verify frozen development-only inputs and existing weights, without generations."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.common.runstate import identity_digest
from one_gpu_lm.evaluation.tasks import file_manifest
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.common.tokenization import encode_prefix

def verify(root):
 root=Path(root).resolve();read=lambda n:json.loads((root/n).read_text());exp=root/'experiments/023-evaluation-stability'
 tasks=read('private/tasks.json')['tasks'];parent=read('parent022.json');runtime=read('runtime.json');plan=json.loads((exp/'plan.json').read_text())
 assert len(tasks)==15 and all(t['split']=='dev' for t in tasks)
 assert identity_digest(tasks)==parent['development_subset_identity_sha256'] and parent['inherited_final_unopened'] and not parent['final_tasks_copied']
 assert not list((root/'private').glob('*final*')) and not list((root/'private').glob('*fresh*'))
 admission=read('runs/admission/admission.json')
 assert admission['complete'] and admission['task_spec_sha256']==sha256(root/'private/tasks.json')
 assert len(admission['tasks'])==15 and all(r['admitted'] for r in admission['tasks'])
 assert len(admission['clean'])==4 and all(r['passed'] for r in admission['clean'])
 for repo,expected in read('source-snapshots.json').items():assert file_manifest((root/'sources'/repo.replace('/','--')).resolve())==expected
 sites={'historical-default':(root/'.venv-agent/lib/python3.10/site-packages').resolve()}
 sites.update({k:root/v for k,v in runtime.get('python_site_overrides',{}).items()})
 deps={k:{str(p.relative_to(d)):sha256(p) for p in sorted(d.rglob('*')) if p.is_file() and '__pycache__' not in p.parts} for k,d in sites.items()}
 for d in sorted((root/'node-deps').iterdir()):
  if d.is_dir():deps['node/'+d.name]={str(p.relative_to(d)):sha256(p) for p in sorted(d.rglob('*')) if p.is_file()}
 assert deps==read('dependency-snapshots.json')
 installed=Path(subprocess.check_output([str(root/'.venv-agent/bin/python'),'-c',"import importlib.util;print(importlib.util.find_spec('minisweagent').submodule_search_locations[0])"],text=True).strip())
 assert {str(p.relative_to(installed)):sha256(p) for p in sorted(installed.rglob('*.py'))}==read('framework-verification.json')['python_package_files']
 profile=json.loads((exp/'profile.json').read_text())
 assert sha256(exp/'profile.json')==sha256(root/'experiments/022-action-repair/profile-action.json')
 weights=verify_model(Path(runtime['model']),root/'experiments/011-coding-model-selection/weights-manifest.json',profile)
 adapters={}
 for name,rel in [('reference020',runtime['reference_adapter']),('action022',runtime['action_adapter'])]:
  d=root/rel;digest=sha256(d/'adapter_model.safetensors')
  assert digest==plan['models'][name]['adapter_model_sha256']
  adapters[name]={p.name:sha256(p) for p in sorted(d.iterdir()) if p.is_file()}
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
 probes=[]
 for p in read('private/probes.json'):
  tokens,_=encode_prefix(tokenizer,p['messages'],profile['chat_template_kwargs'],plan['policy']['format'])
  import hashlib
  digest=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
  assert digest==p['expected_input_sha256'] and len(tokens)==p['expected_input_tokens']
  probes.append({k:v for k,v in p.items() if k!='messages'})
 for row in json.loads((exp/'sources.json').read_text())['sources']:
  assert sha256(exp/row['rights_review']['publication_notice'])==row['license_sha256']
 return {'schema_version':1,'passed':True,'audit_source_sha256':sha256(Path(__file__)),'task_spec_sha256':sha256(root/'private/tasks.json'),
 'admission_sha256':sha256(root/'runs/admission/admission.json'),'development_tasks':15,'repository_count':4,
 'source_dependency_framework_contents_verified':True,'profile_sha256':sha256(exp/'profile.json'),
 'model_verification':weights,'adapters':adapters,'probes':probes,'final_tasks_present':False,'model_generations_performed':0}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert not a.out.exists();write_json(a.out,verify(a.root));print('Development-only sources,weights,admission and exact probe inputs verified')
