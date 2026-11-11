"""Replay real016 histories and synthetic no-edit state, without model generations."""
import argparse,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.agent_policy import BASH_TOOLS
from one_gpu_lm.common.tool_history import render_history
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.models.verify import verify_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);args=p.parse_args();root=args.root
 runtime=json.loads((root/'runtime.json').read_text());profile=json.loads((root/'experiments/017-edit-progress/model-profile.json').read_text());model=Path(runtime['model'])
 assert not (root/'template-progress-probe.json').exists()
 write_json(root/'model-verification.json',verify_model(model,root/'experiments/011-coding-model-selection/weights-manifest.json',profile))
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
 def encode(ms):return tokenizer.apply_chat_template(render_history(ms,True),tokenize=True,return_dict=False,add_generation_prompt=True,tools=BASH_TOOLS,**profile['chat_template_kwargs'])
 prior=args.previous_run/'roundtrip.json';campaign=json.loads(prior.read_text());assert campaign['complete'];rows=[]
 with tempfile.TemporaryDirectory() as tmp:
  f=Path(tmp)/'synthetic.py';f.write_text('unchanged');monitor=EditableProgress(tmp,{'synthetic.py':sha256(f)},['synthetic.py'])
  for row in campaign['results']:
   history=[m for m in json.loads((args.previous_run/'private/roundtrip'/row['id']/'trajectory.json').read_text())['messages'] if m['role'] in ['system','user','assistant']]
   used=0
   for index,message in enumerate(history):
    if message['role']!='assistant':continue
    ms=history[:index];before=json.dumps(ms,sort_keys=True);a=encode(ms)
    control,cr=monitor.prepare_request(ms,used,12,False);candidate,rr=monitor.prepare_request(ms,used,12,True)
    assert encode(control)==a and len(a)==message['extra']['generation']['input_tokens']
    b=encode(candidate);assert json.dumps(ms,sort_keys=True)==before
    assert candidate[1:]==ms[1:] and (rr['reminder'] or b==a)
    rows.append({'id':row['id'],'completed_calls':used,'control_tokens':len(a),'synthetic_no_edit_progress_tokens':len(b),'reminder':rr['reminder'],'would_exceed_limit':len(b)>8192});used+=1
 assert len(rows)==sum(len(r['generations']) for r in campaign['results']) and any(r['reminder'] for r in rows)
 write_json(root/'template-progress-probe.json',{'schema_version':1,'model_generation_performed':False,'prior_campaign_sha256':sha256(prior),'template_sha256':sha256(model/'chat_template.jinja'),'control_tokens_match016':True,'private_history_and_observations_unchanged':True,'candidate_state':'synthetic unchanged editable file over real histories, not actual past file state','prefix_count':len(rows),'prefixes':rows})
 print('Real-tokenizer control equivalence and synthetic reminder checks passed; no model scoring')
if __name__=='__main__':main()
