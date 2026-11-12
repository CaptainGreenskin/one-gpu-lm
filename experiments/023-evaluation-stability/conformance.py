"""CPU-only scripted tool-boundary check; no model generations or scored campaigns."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import write_json,sha256
import agent as boundary
from normalize import transform

def check(root,out):
 root=Path(root).resolve();out=Path(out).resolve();assert not out.exists();out.mkdir(parents=True)
 plan=json.loads((root/'experiments/023-evaluation-stability/plan.json').read_text())
 protocol=out/'scripted-protocol.json';write_json(protocol,{'agent':plan['agent'],'variants':{c:plan['policy'] for c in ['raw','canonical']}})
 commands=['ls -la',"printf '  9 passing (50ms)\\n'",'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT']
 receipts=[]
 for condition in ['raw','canonical']:
  count=0;requests=[]
  def scripted(sock,value,**kwargs):
   nonlocal count
   if value['operation']=='health':return {'cpu_scripted_fixture':True}
   requests.append(value);command=commands[count];count+=1
   return {'text':'<tool_call>\n<function=bash>\n<parameter=command>\n'+command+'\n</parameter>\n</function>\n</tool_call>',
    'generated_tokens':32,'seconds':0,'input_tokens':1,'input_tokens_sha256':'scripted-not-model-tokens',
    'hit_generation_limit':False,'dialogue':{'structured_tool_calls':count-1,'tool_results':count-1}}
  boundary.shared.request=scripted
  dest=out/condition
  sys.argv=['agent.py','--root',str(root),'--tasks',str(root/'private/tasks.json'),'--protocol',str(protocol),
    '--socket',str(out/'unused.sock'),'--out',str(dest),'--id','fast-deep-equal-nan','--variant',condition]
  boundary.main()
  rows=[json.loads(s) for s in (dest/'observations.jsonl').read_text().splitlines()]
  result=json.loads((dest/'result.json').read_text());trace=json.loads((dest/'trajectory.json').read_text())
  assert len(requests)==len(rows)==result['calls']==3 and not result['modified_source_files']
  assert [r['command'] for r in rows]==commands
  assert all(r['delivered_output']==transform(r['raw_output'],condition)[0] for r in rows)
  observations=[m for m in trace['messages'] if 'tool' in m.get('extra',{})]
  assert all(m['content']==f"Exit code: {r['receipt']['returncode']}\n{r['delivered_output']}" for m,r in zip(observations,rows))
  assert requests[1]['messages'][-1]['content']==observations[0]['content']
  assert requests[2]['messages'][-1]['content']==observations[1]['content']
  assert rows[1]['delivered_output']==('  9 passing (50ms)\n' if condition=='raw' else '  9 passing (0ms)\n')
  receipts.append({'condition':condition,'scripted_calls':3,'command_source_and_returncode_preserved':True,
    'actual_delivered_history_verified':True,'model_generations':0})
 return {'schema_version':1,'passed':True,'source_sha256':sha256(Path(__file__)),'cases':receipts,
   'purpose':'Scripted CPU boundary check,not model evaluation or a selection score','final_test_opened':False}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=check(a.root,a.out);write_json(a.out/'result.json',r);print('Both scripted output paths verified; no model loaded')
