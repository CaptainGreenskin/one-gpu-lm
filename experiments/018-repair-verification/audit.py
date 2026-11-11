"""Audit frozen progress and real-tokenizer inputs; never execute a task command."""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import BASH_TOOLS
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.repair_verification import prepare_verification, is_declared_check
from one_gpu_lm.common.tool_history import render_history
from one_gpu_lm.common.receipts import sha256,write_json
import one_gpu_lm.common.actions as actions

def audit(root,run):
 c=json.loads((run/'controller.json').read_text());assert c['complete'] and c['telemetry']['thread_stopped']
 tasks={t['id']:t for t in json.loads((root/'private/tasks.json').read_text())['tasks']}
 admitted={t['id']:t for t in json.loads((root/'runs/admission/admission.json').read_text())['tasks']}
 runtime=json.loads((root/'runtime.json').read_text());profile=json.loads((root/'experiments/018-repair-verification/model-profile.json').read_text())
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
 def encode(ms):return tokenizer.apply_chat_template(render_history(ms,True),tokenize=True,return_dict=False,add_generation_prompt=True,tools=BASH_TOOLS,**profile['chat_template_kwargs'])
 value={'schema_version':1,'audit_source_sha256':sha256(Path(__file__)),'parser_sha256':sha256(Path(actions.__file__)),'template_sha256':sha256(Path(runtime['model'])/'chat_template.jinja'),'purpose':'Post-run replay of actual progress and verification reminders, exact input-token hashes, tool exit statuses and source-state pairing; no commands executed or scores changed','raw_traces_public':False,'variants':{},'campaign_sha256':{}}
 for arm in ['control','verify']:
  path=run/(arm+'.json');campaign=json.loads(path.read_text());assert campaign['complete'];value['campaign_sha256'][arm]=sha256(path);out=[]
  for r in campaign['results']:
   private=run/'private'/arm/r['id'];t=json.loads((private/'trajectory.json').read_text());original=json.loads((private/'original.json').read_text());editable=tasks[r['id']]['editable_files']
   assert all(original[name]==admitted[r['id']]['broken_sha256'][name] for name in editable),'Initial source differs from admitted defective input'
   hashes=sorted((name,original[name]) for name in editable)
   state={'changed_editable_files':0,'invalid_editable_files':0,'editable_state_sha256':hashlib.sha256(json.dumps(hashes,separators=(',',':')).encode()).hexdigest()}
   messages=[m for m in t['messages'] if m['role'] in ['system','user','assistant']];steps=r['source_progress'];assert steps==t['local_model']['source_progress']
   assert len(steps)==r['tool_calls'];used=step=0;commands=[];categories=Counter();native=0;check_events=[];last_check=None;prompt_overhead=[]
   def check_request(prefix,g):
    tracker=object.__new__(EditableProgress);tracker.snapshot=lambda:dict(state)
    packed,receipt=tracker.prepare_request(prefix,used,12,True)
    assert receipt==g['progress'],'Progress receipt/trigger differs from observed state'
    before_verification=packed
    packed,vr=prepare_verification(packed,steps[:step],state,used,12,tasks[r['id']]['regression_command'],arm=='verify')
    assert vr==g['verification'],'Verification status/trigger differs from actual tool/source evidence'
    tokens=encode(packed);assert len(tokens)==g['input_tokens'] and hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()==g['input_tokens_sha256'],'Actual tokenizer input differs from intended history/reminder'
    assert g['dialogue']=={'structured_tool_calls':native,'tool_results':native}
    prompt_overhead.append(len(tokens)-len(encode(before_verification)) if vr['reminder'] else 0)
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
     s=steps[step];assert s['call_index']==used;before=state;state={k:v for k,v in s.items() if k!='call_index'};step+=1;native+=1
     if is_declared_check(messages[index-1]['extra']['actions'][0]['command'],tasks[r['id']]['regression_command']):
      last_check={'call':used,'returncode':m['extra']['tool']['returncode'],'state_sha256':state['editable_state_sha256'],'stable':before['editable_state_sha256']==state['editable_state_sha256'] and not before['invalid_editable_files'] and not state['invalid_editable_files']}
     status='not_run' if last_check is None else 'changed_during_check' if not last_check['stable'] else 'stale' if last_check['state_sha256']!=state['editable_state_sha256'] or state['invalid_editable_files'] else 'passed' if last_check['returncode']==0 else 'failed'
     check_events.append({'call':used,'changed_files':state['changed_editable_files'],'check_status':status,'last_declared_check':dict(last_check) if last_check else None})
   for error in r['backend_errors']:
    check_request(messages,error);used+=1
   assert used==r['calls'] and step==len(steps)
   final=EditableProgress(private/'workspace',original,editable).snapshot();assert final==state
   if not final['invalid_editable_files']:assert final['changed_editable_files']==len(r['modified_source_files'])
   if r.get('source_sha256'):assert all(sha256(private/'workspace'/name)==digest for name,digest in r['source_sha256'].items()),'Graded source differs from final workspace'
   out.append({'id':r['id'],'split':r['split'],'passed':r['passed'],'input_token_hashes_verified':used,'progress_and_native_turns_verified':True,'final_source_verified':True,'format_categories':dict(categories),'consecutive_repeated_commands':sum(a==b for a,b in zip(commands,commands[1:])),'verification_extra_input_tokens':prompt_overhead, 'verification_statuses':[g['verification']['check_status'] for g in r['generations']], 'verification_events':check_events, 'final_observed_check_status':check_events[-1]['check_status'] if check_events else 'not_run', 'first_edit_call':next((s['call_index'] for s in steps if s['changed_editable_files']),None),'trajectory_sha256':sha256(private/'trajectory.json'),'grader_log_sha256':sha256(private/'grader.log') if (private/'grader.log').exists() else None})
   out[-1]['generation_output_sha256']=[hashlib.sha256(m['content'].encode()).hexdigest() for m in messages if m['role']=='assistant']
   out[-1]['generation_input_sha256']=[g['input_tokens_sha256'] for g in r['generations']]
  value['variants'][arm]=out
 control={r['id']:r for r in value['variants']['control']}
 comparisons=[]
 for progress in value['variants']['verify']:
  prior=control[progress['id']]
  matched=[{'call':i+1,'different_output':ao!=bo} for i,(ai,bi,ao,bo) in enumerate(zip(prior['generation_input_sha256'],progress['generation_input_sha256'],prior['generation_output_sha256'],progress['generation_output_sha256'])) if ai==bi]
  comparisons.append({'id':progress['id'],'identical_input_calls':len(matched),'different_output_calls':[r['call'] for r in matched if r['different_output']]})
 value['same_input_output_comparison']=comparisons
 return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--run',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert not a.out.exists();write_json(a.out,audit(a.root,a.run));print('All actual request hashes, progress states, source endpoints and parser decisions verified')
