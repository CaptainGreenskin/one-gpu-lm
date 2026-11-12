"""Audit frozen progress and real-tokenizer inputs; never execute a task command."""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import BASH_TOOLS
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.tool_history import render_history
from one_gpu_lm.common.receipts import sha256,write_json
import one_gpu_lm.common.actions as actions

def audit(root,run):
 c=json.loads((run/'controller.json').read_text());assert c['complete'] and c['telemetry']['thread_stopped']
 tasks={t['id']:t for t in json.loads((root/'private/tasks.json').read_text())['tasks']}
 admitted={t['id']:t for t in json.loads((root/'runs/admission/admission.json').read_text())['tasks']}
 runtime=json.loads((root/'runtime.json').read_text());profile=json.loads((root/'experiments/021-observed-recovery/profile-observed.json').read_text())
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
 def encode(ms):return tokenizer.apply_chat_template(render_history(ms,True),tokenize=True,return_dict=False,add_generation_prompt=True,tools=BASH_TOOLS,**profile['chat_template_kwargs'])
 value={'schema_version':1,'audit_source_sha256':sha256(Path(__file__)),'parser_sha256':sha256(Path(actions.__file__)),'template_sha256':sha256(Path(runtime['model'])/'chat_template.jinja'),'purpose':'Post-run replay of actual progress reminders and exact input-token hashes; no commands executed or scores changed','raw_traces_public':False,'variants':{},'campaign_sha256':{}}
 selection=json.loads((run/'selection.json').read_text())
 arms=['dev-base','dev-reference020','dev-observed']+(['final-base','final-selected'] if selection['selected'] is not None else [])
 for arm in arms:
  path=run/arm/'campaign.json';campaign=json.loads(path.read_text());assert campaign['complete'];value['campaign_sha256'][arm]=sha256(path);out=[]
  for r in campaign['results']:
   private=run/arm/'private'/r['id'];t=json.loads((private/'trajectory.json').read_text());original=json.loads((private/'original.json').read_text());editable=tasks[r['id']]['editable_files']
   assert all(original[name]==admitted[r['id']]['broken_sha256'][name] for name in editable),'Initial source differs from admitted defective input'
   hashes=sorted((name,original[name]) for name in editable)
   state={'changed_editable_files':0,'invalid_editable_files':0,'editable_state_sha256':hashlib.sha256(json.dumps(hashes,separators=(',',':')).encode()).hexdigest()}
   messages=[m for m in t['messages'] if m['role'] in ['system','user','assistant']];steps=r['source_progress'];assert steps==t['local_model']['source_progress']
   assert len(steps)==r['tool_calls'];used=step=0;commands=[];categories=Counter();native=0
   def check_request(prefix,g):
    tracker=object.__new__(EditableProgress);tracker.snapshot=lambda:dict(state)
    packed,receipt=tracker.prepare_request(prefix,used,12,True)
    assert receipt==g['progress'],'Progress receipt/trigger differs from observed state'
    tokens=encode(packed);assert len(tokens)==g['input_tokens'] and hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()==g['input_tokens_sha256'],'Actual tokenizer input differs from intended history/reminder'
    assert g['dialogue']=={'structured_tool_calls':native,'tool_results':native}
   for index,m in enumerate(messages):
    if m['role']=='assistant':
     g=r['generations'][used];assert m['extra']['generation']==g
     check_request(messages[:index],g)
     try:
      command=parse_action(m['content'])['command'];commands.append(command);valid=True
      assert m['extra']['actions']==[{'command':command}], 'Executed action differs from parsed output'
     except ValueError:
      valid=False;categories['invalid_at_output_limit' if g['hit_generation_limit'] else 'other_invalid_action']+=1
     assert valid==g['action_format_valid'];used+=1
    elif 'tool' in m.get('extra',{}):
     s=steps[step];assert s['call_index']==used;state={k:v for k,v in s.items() if k!='call_index'};step+=1;native+=1
   for error in r['backend_errors']:
    check_request(messages,error);used+=1
   assert used==r['calls'] and step==len(steps)
   assert used<=12 and r['generated_tokens']<=6144
   assert all(g['generated_tokens']<=512 and g['input_tokens']<=8192 for g in r['generations'])
   final=EditableProgress(private/'workspace',original,editable).snapshot();assert final==state
   if not final['invalid_editable_files']:assert final['changed_editable_files']==len(r['modified_source_files'])
   if r.get('source_sha256'):assert all(sha256(private/'workspace'/name)==digest for name,digest in r['source_sha256'].items()),'Graded source differs from final workspace'
   out.append({'id':r['id'],'split':r['split'],'passed':r['passed'],'input_token_hashes_verified':used,'progress_and_native_turns_verified':True,'final_source_verified':True,'format_categories':dict(categories),'consecutive_repeated_commands':sum(a==b for a,b in zip(commands,commands[1:])),'first_edit_call':next((s['call_index'] for s in steps if s['changed_editable_files']),None),'trajectory_sha256':sha256(private/'trajectory.json'),'grader_log_sha256':sha256(private/'grader.log') if (private/'grader.log').exists() else None})
   out[-1]['generation_output_sha256']=[hashlib.sha256(m['content'].encode()).hexdigest() for m in messages if m['role']=='assistant']
   out[-1]['generation_input_sha256']=[g['input_tokens_sha256'] for g in r['generations']]
  value['variants'][arm]=out
 from decision import select,paired
 plan=json.loads((root/'experiments/021-observed-recovery/preparation-plan.json').read_text())
 protocol=json.loads((root/'experiments/021-observed-recovery/protocol.json').read_text())
 for name,digest in protocol['frozen_files'].items():assert sha256(root/name)==digest,'Frozen input changed'
 def campaign(arm):return json.loads((run/arm/'campaign.json').read_text())
 dev=[t['id'] for t in tasks.values() if t['split']=='dev']
 recomputed=select(campaign('dev-base'),campaign('dev-reference020'),campaign('dev-observed'),dev,plan)
 assert {k:v for k,v in selection.items() if k!='final_test_opened'}=={k:v for k,v in recomputed.items() if k!='final_test_opened'}
 assert selection['final_test_opened']==(selection['selected'] is not None)
 for arm in arms:
  expected=dev if arm.startswith('dev-') else [t['id'] for t in tasks.values() if t['split']=='test']
  rows=campaign(arm)['results'];assert len(rows)==len(expected) and {r['id'] for r in rows}==set(expected)
 value['selection_recomputed_without_test_scores']=True
 value['original_wall_budget_respected']=c['finished_epoch']<=c['deadline_epoch']
 value['training']={}
 from one_gpu_lm.common.checkpoints import load_checkpoint
 for name in ['observed']:
  path=run/('train-'+name);result=json.loads((path/'result.json').read_text());metrics=json.loads((path/'metrics.json').read_text())
  assert result['status']=='completed' and result['steps']==protocol['training_records']==len(metrics)
  for key in ['frozen_sample_unchanged','frozen_gradients_absent','adapter_changed','saved_reload_greedy_ids_equal']:assert result[key] is True
  checkpoint,trainer,manifest=load_checkpoint(path,json.loads((path/'identity.json').read_text()))
  assert trainer['adapter_state_sha256']==result['adapter_state_sha256']
  assert result['supervised_tokens']==json.loads((root/'runs/reference-data021/derived-data.json').read_text())['supervised_tokens']==sum(r['supervised_tokens'] for r in metrics)
  value['training'][name]={'updates':result['steps'],'trainable_parameters':result['trainable_parameters'],'supervised_tokens':result['supervised_tokens'],'checkpoint_verified':True,'save_reload_greedy_match':True,'adapter_files':result['adapter_files']}
 value['release_audit_complete']=False
 value['outstanding_release_checks']=['rights_and_secret_scan','export_package_consistency','model_card_limitations_and_release_gate']
 return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--run',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert not a.out.exists();write_json(a.out,audit(a.root,a.run));print('All actual request hashes, progress states, source endpoints and parser decisions verified')
