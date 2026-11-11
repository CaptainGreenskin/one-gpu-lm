import copy,hashlib,importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
def module(name):
 spec=importlib.util.spec_from_file_location('progress_'+name,ROOT/'experiments/017-edit-progress'/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
REPORT=module('report')

def fixture():
 inventory=[{'id':str(i),'repository':'old/repo' if i<3 else 'new/repo','split':'dev' if i<3 else 'test','family':str(i)} for i in range(5)]
 p={'execution':{'expected_tasks_per_variant':5},'evaluation_task_counts':{'dev':3,'test':2},'agent':{'max_model_calls':12,'max_total_generated_tokens':6144,'max_input_tokens':8192,'max_generated_tokens_per_call':512},'variants':{'control':{'format':'native_roundtrip','context':'full'},'progress':{'format':'native_roundtrip','context':'full','progress':'edit_checkpoint_v1'}},'promotion':{'target_tasks':['0','1','2'],'minimum_net_fresh_gain':2,'maximum_regressed_fresh_tasks':0,'maximum_regressed_historical_tasks':0}}
 campaigns={}
 for arm in REPORT.ARMS:
  rows=[];policy=p['variants'][arm]
  for i,t in enumerate(inventory):
   passed=i<3 or arm=='progress';progress={'enabled':arm=='progress','completed_calls':0,'remaining_calls':12,'changed_editable_files':0,'identical_command_streak':0,'reminder':False,'reminder_chars':0}
   g={'progress':progress,'input_tokens_sha256':'a'*64,'context':{'policy':'full','dropped_rounds':0,'raw_input_tokens':100,'packed_input_tokens':100},'input_tokens':100,'generated_tokens':10,'seconds':1,'dialogue':{'structured_tool_calls':0,'tool_results':0},'adapter':False,'agent_policy':policy,'action_format_valid':True,'hit_generation_limit':False}
   rows.append(dict(t,variant=arm,agent_policy=policy,passed=passed,calls=1,generated_tokens=10,seconds=2,tool_seconds=1,tool_calls=1,exit_status='Submitted',modified_source_files=['code.py'] if passed else [],generations=[g],backend_errors=[],source_progress=[{'call_index':1,'changed_editable_files':int(passed)}]))
  campaigns[arm]={'complete':True,'identity':{'ids':[t['id'] for t in inventory],'variant':arm,'policy':policy},'results':rows}
 return campaigns,p,inventory

class ProgressReportContracts(unittest.TestCase):
 def test_dependency_review_indicators_are_private_and_do_not_claim_no_access(self):
  review=module('dependency_review')
  messages=[{'role':'assistant','extra':{'actions':[{'command':c}]}} for c in ["cat node_modules/p-map/index.js # PRIVATE", "node -e \"import('p-map')\"", 'cat index.js', 'node node_modules/ava/entrypoints/cli.mjs']]
  v=review.command_indicators(messages)
  self.assertEqual(v,{'accepted_commands':4,'explicit_dependency_path_commands':1,'bare_package_import_commands':1,'commands_mentioning_node_modules':2})
  self.assertNotIn('PRIVATE',json.dumps(v))

 def test_fresh_success_and_no_regressions_are_required_not_an_edit(self):
  c,p,i=fixture();s=REPORT.assess(c,p,i);self.assertTrue(s['candidate_for_replication']);self.assertFalse(s['deployment_allowed'])
  c['progress']['results'][3]['passed']=False
  s=REPORT.assess(c,p,i);self.assertFalse(s['candidate_for_replication']);self.assertEqual(s['totals']['progress']['passed']['fresh'],1)
  c,p,i=fixture();c['progress']['results'][0]['passed']=False
  s=REPORT.assess(c,p,i);self.assertFalse(s['gate_checks']['no_historical_regressions']);self.assertFalse(s['gate_checks']['all_three_targets_pass'])

 def test_exposure_budget_and_false_reminders_fail_closed(self):
  for mutate in [lambda c,p,i:i[3].update(repository='old/repo'),lambda c,p,i:c['control']['results'][0].update(calls=13),lambda c,p,i:c['progress']['results'][0]['generations'][0]['progress'].update(reminder=True,reminder_chars=1),lambda c,p,i:c['progress']['results'][0]['generations'][0]['progress'].update(remaining_calls=11)]:
   c,p,i=fixture();mutate(c,p,i)
   with self.assertRaises(ValueError):REPORT.assess(c,p,i)

 def test_actual_request_hash_audit_rejects_tampering_and_keeps_commands_private(self):
  from one_gpu_lm.common.receipts import write_json,sha256
  from one_gpu_lm.common.edit_progress import EditableProgress
  from one_gpu_lm.common.tool_history import render_history
  audit=module('audit')
  class Tokenizer:
   @classmethod
   def from_pretrained(cls,*args,**kwargs):return cls()
   def apply_chat_template(self,messages,**kwargs):return list(json.dumps(messages,sort_keys=True).encode())
  with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,{'transformers':types.SimpleNamespace(AutoTokenizer=Tokenizer)}):
   root=Path(tmp);run=root/'runs';run.mkdir();(root/'private').mkdir();(root/'experiments/017-edit-progress').mkdir(parents=True);model=root/'model';model.mkdir();(model/'chat_template.jinja').write_text('synthetic')
   write_json(root/'runtime.json',{'model':str(model)});write_json(root/'experiments/017-edit-progress/model-profile.json',{'chat_template_kwargs':{}})
   write_json(root/'private/tasks.json',{'tasks':[{'id':'sample','editable_files':['code.py']}]});write_json(run/'controller.json',{'complete':True,'telemetry':{'thread_stopped':True}})
   for arm in REPORT.ARMS:
    private=run/'private'/arm/'sample';workspace=private/'workspace';workspace.mkdir(parents=True);(workspace/'code.py').write_text('original')
    original={'code.py':sha256(workspace/'code.py')};write_json(private/'original.json',original);tracker=EditableProgress(workspace,original,['code.py'])
    ms=[{'role':'system','content':'rules'},{'role':'user','content':'task'}];packed,pr=tracker.prepare_request(ms,0,12,arm=='progress');tokens=Tokenizer().apply_chat_template(render_history(packed,True))
    g={'progress':pr,'input_tokens':len(tokens),'input_tokens_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),'dialogue':{'structured_tool_calls':0,'tool_results':0},'action_format_valid':True}
    ms += [{'role':'assistant','content':'```bash\necho PRIVATE_COMMAND\n```','extra':{'actions':[{'command':'echo PRIVATE_COMMAND'}],'generation':g}},{'role':'user','content':'PRIVATE_RESULT','extra':{'tool':{'returncode':0}}}]
    steps=[dict(tracker.snapshot(),call_index=1)];write_json(private/'trajectory.json',{'messages':ms,'local_model':{'source_progress':steps}})
    r={'id':'sample','split':'dev','passed':False,'calls':1,'tool_calls':1,'generations':[g],'backend_errors':[],'source_progress':steps,'modified_source_files':[]}
    write_json(run/(arm+'.json'),{'complete':True,'results':[r]})
   value=audit.audit(root,run);self.assertNotIn('PRIVATE_',json.dumps(value));self.assertEqual(value['variants']['progress'][0]['input_token_hashes_verified'],1)
   path=run/'progress.json';v=json.loads(path.read_text());v['results'][0]['generations'][0]['input_tokens_sha256']='0'*64;write_json(path,v)
   tp=run/'private/progress/sample/trajectory.json';t=json.loads(tp.read_text());t['messages'][2]['extra']['generation']['input_tokens_sha256']='0'*64;write_json(tp,t)
   with self.assertRaises(AssertionError):audit.audit(root,run)

if __name__=='__main__':unittest.main()
