"""Output-boundary,independent-grader and counterbalanced repeated-run contracts."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/023-evaluation-stability'
sys.path.insert(0,str(EXP))
def load(name):
 spec=importlib.util.spec_from_file_location('stability_'+name,EXP/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
norm=load('normalize');schedule=load('schedule');wrapper=load('agent')

class OutputBoundary(unittest.TestCase):
 def test_only_complete_metadata_fields_change(self):
  original='-rw-r--r-- 1 root root 120 Nov 12 09:35 index.js\n  47 passing (186ms)\n  3 failing\n'
  expected='-rw-r--r-- 1 root root 120 Jan 01 00:00 index.js\n  47 passing (0ms)\n  3 failing\n'
  actual,receipt=norm.transform(original,'canonical');self.assertEqual(actual,expected)
  self.assertEqual(receipt['applied_fields'],{'ls_timestamp':1,'js_summary_duration':1})
  self.assertEqual(norm.transform(original,'raw')[0],original)
 def test_task_dates_errors_and_unrecognized_times_survive(self):
  text='const date = "Nov 12 09:35";\nERROR: expected 2026-11-12,got 2026-11-13\nRan 3 tests in 0.22s\n123 failing (clock)\n'
  self.assertEqual(norm.transform(text,'canonical')[0],text)
 def test_file_metadata_and_names_are_preserved(self):
  text='lrwxrwxrwx 1 user group 8 Dec 31  2025 Nov 12 09:35 -> target\n'
  actual,_=norm.transform(text,'canonical')
  self.assertEqual(actual,'lrwxrwxrwx 1 user group 8 Jan 01 00:00 Nov 12 09:35 -> target\n')
 def test_no_hidden_tail_is_exposed(self):
  text=('x'*2380+'\n  2 passing (999ms)')[:2400]+norm.MARKER
  actual,receipt=norm.transform(text,'canonical')
  self.assertTrue(actual.endswith(norm.MARKER));self.assertTrue(receipt['raw_prefix_only'])
  self.assertLessEqual(len(actual.removesuffix(norm.MARKER)),2400)
  with self.assertRaises(ValueError):norm.transform('x'*2401,'canonical')
 def test_idempotent_known_rows(self):
  text='-rw-r--r-- 1 root root 120 Nov 12 09:35 x\n  9 passing (2.1s)\n'
  a,_=norm.transform(text,'canonical');self.assertEqual(norm.transform(a,'canonical')[0],a)
 def test_real_environment_boundary_records_raw_but_grader_stays_raw(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);original='  7 passing (200ms)\n';kind=wrapper.recorded_environment(out,'canonical')
   def env(work):
    e=kind(work,python_site=out,node_root=out)
    e.sandbox=SimpleNamespace(execute=lambda *a,**k:{'stdout':original,'stderr':'','returncode':7,'reason':'runtime_error'})
    return e
   value=env(out/'workspace').execute({'command':'test-command'})
   self.assertEqual(value['returncode'],7);self.assertEqual(value['reason'],'runtime_error');self.assertEqual(value['output'],'  7 passing (0ms)\n')
   row=json.loads((out/'observations.jsonl').read_text());self.assertEqual(row['raw_output'],original)
   grader=env(out/'grader').execute({'command':'independent-grade'})
   self.assertEqual(grader['output'],original);self.assertNotIn('normalization',grader)
   self.assertEqual(len((out/'observations.jsonl').read_text().splitlines()),1)

class FrozenSchedule(unittest.TestCase):
 def test_exact270_unique_jobs_and_balanced_model_positions(self):
  bs=schedule.blocks();ids=[str(i) for i in range(15)];jobs=[]
  self.assertEqual(len(bs),9)
  for b in bs:
   entries=schedule.task_schedule(ids,b);self.assertEqual(len(entries),30)
   for a,z in zip(entries[::2],entries[1::2]):
    self.assertEqual(a['id'],z['id']);self.assertEqual({a['condition'],z['condition']},{'raw','canonical'})
   jobs += [(b['repeat'],b['model'],v['id'],v['condition']) for v in entries]
  self.assertEqual(len(jobs),len(set(jobs)));self.assertEqual(len(jobs),270)
  for m in schedule.MODELS:self.assertEqual({b['index']%3 for b in bs if b['model']==m},{0,1,2})
 def test_development_only_and_budget_inherited(self):
  p=json.loads((EXP/'plan.json').read_text());old=json.loads((ROOT/'experiments/022-action-repair/protocol.json').read_text())
  self.assertEqual(p['agent'],old['agent']);self.assertFalse(p['final_test_opened']);self.assertEqual(p['training_updates'],0)
  self.assertEqual(p['probes']['planned_calls'],36);self.assertEqual(p['planned_task_runs'],270)
  with self.assertRaises(ValueError):schedule.task_schedule(['same']*15,schedule.blocks()[0])
 def test_published_frozen_sources_remain_exact(self):
  import hashlib
  path=EXP/'protocol.json'
  if not path.exists():self.skipTest('Protocol is not frozen yet')
  p=json.loads(path.read_text())
  self.assertEqual(p['task_counts'],{'dev':15,'test':0,'train':0})
  self.assertFalse(p['final_test_opened'])
  for name,digest in p['frozen_files'].items():
   local=ROOT/name
   if local.is_file():self.assertEqual(hashlib.sha256(local.read_bytes()).hexdigest(),digest,name)


class DivergenceAttribution(unittest.TestCase):
 def fixture(self):
  return {'calls':1,'passed':False,'generation_input_sha256':['i'],'generation_output_sha256':['o'],
   'source_progress':[{'call_index':1,'changed_editable_files':0}],
   'observation_receipts':[{'model_call':1,'command_sha256':'c','prior_editable_state_sha256':'s',
    'returncode':0,'reason':'ok','raw_sha256':'raw','canonical_sha256':'canon','delivered_sha256':'raw'}]}
 def test_identical_input_output_divergence_is_not_blame_on_tools(self):
  mod=load('analysis');a=self.fixture();b=self.fixture();b['generation_output_sha256']=['different']
  self.assertEqual(mod.first_divergence(a,b)['category'],'generation_at_identical_input')
 def test_metadata_vs_other_text_and_status_are_distinguished(self):
  mod=load('analysis');a=self.fixture();b=self.fixture();b['observation_receipts'][0]['delivered_sha256']='different'
  self.assertEqual(mod.first_divergence(a,b)['category'],'allowed_output_metadata')
  b['observation_receipts'][0]['canonical_sha256']='other'
  self.assertEqual(mod.first_divergence(a,b)['category'],'other_tool_text')
  b['observation_receipts'][0]['returncode']=1
  self.assertEqual(mod.first_divergence(a,b)['category'],'tool_status')
 def test_masked_raw_change_is_not_model_input_divergence(self):
  mod=load('analysis');a=self.fixture();b=self.fixture();b['observation_receipts'][0]['raw_sha256']='different'
  r=mod.first_divergence(a,b);self.assertEqual(r['category'],'identical_observed_trace');self.assertEqual(r['metadata_differences_masked_before_divergence'],1)
 def test_grade_disagreement_with_identical_trace_is_separate(self):
  mod=load('analysis');a=self.fixture();b=self.fixture();b['passed']=True
  self.assertEqual(mod.first_divergence(a,b)['category'],'grade_only')
 def test_partial_cell_matrix_is_rejected(self):
  with self.assertRaises(ValueError):load('analysis').summarize({}, {'campaigns':{}})

 def test_hand_counted_repeated_matrix_uses_exact_pair_denominators(self):
  mod=load('analysis');campaigns={};audit={'campaigns':{}}
  for model in mod.MODELS:
   for condition in mod.CONDITIONS:
    for repeat in [1,2,3]:
     name=f'r{repeat:02d}-{model}-{condition}';rows=[];traces=[]
     for index in range(15):
      passed=index==0 and (condition=='canonical' or repeat==2)
      trace=dict(self.fixture(),id=str(index),passed=passed);traces.append(trace)
      rows.append({'id':str(index),'passed':passed,'calls':1,'generated_tokens':3,'seconds':1,
       'source_progress':trace['source_progress'],'generations':[{'action_format_valid':True}]})
     campaigns[name]={'complete':True,'results':rows}
     audit['campaigns'][name]={'model':model,'condition':condition,'repeat':repeat,'tasks':traces}
  result=mod.summarize(campaigns,audit)
  self.assertEqual(len(result['pairs']),405)
  self.assertEqual(sum(p['outcome_disagreement'] for p in result['pairs'] if p['comparison']=='within_condition'),6)
  for row in result['stability']:
   self.assertEqual(row['pair_denominator'],45)
   self.assertEqual(row['pairwise_disagreements'],2 if row['condition']=='raw' else 0)
   self.assertEqual(row['mixed_outcome_tasks'],1 if row['condition']=='raw' else 0)

if __name__=='__main__':unittest.main()
