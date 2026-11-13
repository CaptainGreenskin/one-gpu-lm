"""Verify actual024 tokenizer requests,output-only edits and final-source endpoints."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.tokenization import encode_prefix
from _feedback024 import module
normalization=module('normalize');transform=normalization.transform;digest=normalization.digest

def audit(root,run):
 root=Path(root).resolve();run=Path(run).resolve();read=lambda p:json.loads(Path(p).read_text());exp=root/'experiments/024-feedback-noise'
 controller=read(run/'controller.json');assert controller['complete'] and controller['telemetry']['thread_stopped']
 protocol=read(exp/'protocol.json');profile=read(exp/'profile.json');runtime=read(root/'runtime.json')
 for n,h in protocol['frozen_files'].items():assert sha256(root/n)==h,'Changed frozen input'
 tasks={t['id']:t for t in read(root/'private/tasks.json')['tasks']};assert len(tasks)==15 and all(t['split']=='dev' for t in tasks.values())
 assert not list(run.glob('final-*')) and not list((root/'private').glob('*fresh*'))
 admitted={r['id']:r for r in read(root/'runs/admission/admission.json')['tasks']}
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
 def encode(messages):return encode_prefix(tokenizer,messages,profile['chat_template_kwargs'],'native_roundtrip')
 value={'schema_version':1,'audit_source_sha256':sha256(Path(__file__)),'normalizer_sha256':sha256(exp/'normalize.py'),
 'protocol_sha256':sha256(exp/'protocol.json'),'campaigns':{},'probe_count':0,'probes_verified':[],
 'final_test_opened':False,'training_updates':0,'raw_traces_public':False,'commands_reexecuted':0,'model_generations_performed':0}
 for block in protocol['blocks']:
  for condition in ['baseline','extended']:
   label=f'r{block["repeat"]:02d}-{block["model"]}-{condition}';folder=run/'campaigns'/label;p=folder/'campaign.json';campaign=read(p)
   assert campaign['complete'] and [r['id'] for r in campaign['results']]==list(tasks)
   rows=[]
   for r in campaign['results']:
    private=folder/'private'/r['id'];trajectory=read(private/'trajectory.json');original=read(private/'original.json');editable=tasks[r['id']]['editable_files']
    assert all(original[n]==admitted[r['id']]['broken_sha256'][n] for n in editable)
    state={'changed_editable_files':0,'invalid_editable_files':0,'editable_state_sha256':digest(json.dumps(sorted((n,original[n]) for n in editable),separators=(',',':')))}
    journal=[json.loads(s) for s in (private/'observations.jsonl').read_text().splitlines()] if (private/'observations.jsonl').exists() else []
    assert [v['receipt'] for v in journal]==r['observation_receipts']
    steps=r['source_progress'];assert steps==trajectory['local_model']['source_progress'] and len(journal)==len(steps)==r['tool_calls']
    messages=[m for m in trajectory['messages'] if m['role'] in ['system','user','assistant']];used=step=native=0;output_hashes=[];turns=[]
    for index,m in enumerate(messages):
     if m['role']=='assistant':
      g=r['generations'][used];assert m['extra']['generation']==g
      tracker=object.__new__(EditableProgress);tracker.snapshot=lambda:dict(state)
      packed,progress=tracker.prepare_request(messages[:index],used,12,True)
      assert progress==g['progress'];tokens,dialogue=encode(packed)
      assert len(tokens)==g['input_tokens'] and digest(json.dumps(tokens,separators=(',',':')))==g['input_tokens_sha256']
      assert dialogue==g['dialogue'] and dialogue=={'structured_tool_calls':native,'tool_results':native}
      try:
       command=parse_action(m['content'])['command'];valid=True
       assert m['extra']['actions']==[{'command':command}]
      except ValueError:valid=False
      assert valid==g['action_format_valid'];output_hashes.append(digest(m['content']));used+=1
     elif 'tool' in m.get('extra',{}):
      entry=journal[step];receipt=entry['receipt'];delivered,expected=transform(entry['raw_output'],condition,2400)
      assert all(receipt[k]==v for k,v in expected.items()) and entry['delivered_output']==delivered
      assert receipt['command_sha256']==digest(entry['command'])==digest(command)
      assert receipt['call_index']==step+1  # Environment calls exclude format failures.
      assert m['content']==f"Exit code: {receipt['returncode']}\n{delivered}"
      tool=m['extra']['tool'];assert tool['normalization']==receipt and tool['returncode']==receipt['returncode'] and tool['reason']==receipt['reason']
      assert steps[step]['call_index']==used
      turns.append(dict(receipt,model_call=used,prior_editable_state_sha256=state['editable_state_sha256']))
      state={k:v for k,v in steps[step].items() if k!='call_index'};step+=1;native+=1
    assert not r['backend_errors'] and used==r['calls'] and step==len(steps)
    assert used<=12 and sum(g['generated_tokens'] for g in r['generations'])==r['generated_tokens']<=6144
    assert all(0<g['generated_tokens']<=512 and g['input_tokens']<=8192 for g in r['generations'])
    final=EditableProgress(private/'workspace',original,editable).snapshot();assert final==state
    if not final['invalid_editable_files']:assert final['changed_editable_files']==len(r['modified_source_files'])
    if r.get('source_sha256'):assert all(sha256(private/'workspace'/n)==h for n,h in r['source_sha256'].items())
    rows.append({'id':r['id'],'passed':r['passed'],'calls':used,'input_token_hashes_verified':used,
     'generation_input_sha256':[g['input_tokens_sha256'] for g in r['generations']],
     'generation_output_sha256':output_hashes,'observation_receipts':turns,'source_progress':steps,
     'first_edit_call':next((s['call_index'] for s in steps if s['changed_editable_files']),None),
     'progress_native_history_and_output_transform_verified':True,'final_source_verified':True,
     'trajectory_sha256':sha256(private/'trajectory.json'),'journal_sha256':sha256(private/'observations.jsonl') if journal else None,
     'grader_log_sha256':sha256(private/'grader.log') if (private/'grader.log').exists() else None})
   value['campaigns'][label]={'sha256':sha256(p),'model':block['model'],'repeat':block['repeat'],'condition':condition,'tasks':rows}
 probes=read(root/'private/probes.json');probe_map={p['id']:p for p in probes}
 for p in sorted((run/'probes').glob('*.json')):
  row=read(p);source=probe_map[row['id']];response=read(run/'private'/('probe-'+p.stem+'.json'));tokens,_=encode(source['messages'])
  assert digest(json.dumps(tokens,separators=(',',':')))==row['input_tokens_sha256']==source['expected_input_sha256']
  assert row['output_sha256']==digest(response['text']) and row['ids_sha256']==digest(json.dumps(response['ids'],separators=(',',':')))
  assert all(row[k]==v for k,v in response.items() if k not in {'text','ids'}) and 0<row['generated_tokens']<=512
  value['probes_verified'].append({'name':p.name,'sha256':sha256(p),'private_response_sha256':sha256(run/'private'/('probe-'+p.stem+'.json'))})
 value['probe_count']=len(value['probes_verified']);assert value['probe_count']==36
 value['task_runs']=sum(len(v['tasks']) for v in value['campaigns'].values());assert value['task_runs']==270
 assert not list((run/'private').glob('probe-*.pending'))
 value['generation_calls_audited']=sum(r['calls'] for c in value['campaigns'].values() for r in c['tasks'])
 value['original_deadline_respected']=controller['finished_epoch']<=controller['deadline_epoch']
 return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();write_json(a.out,audit(a.root,a.run));print('All270development trajectories and36fixed probes verified without reruns')
