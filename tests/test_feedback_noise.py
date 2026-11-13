"""024 additive feedback rules preserve semantics, budgets and primary counts."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/024-feedback-noise'
sys.path.insert(0,str(EXP))
from _feedback024 import module
norm=module('normalize');agent=module('agent');analysis=module('analysis');schedule=module('schedule')

class AdditionalTestDurations(unittest.TestCase):
 def test_baseline_remains_exact_previous_behavior(self):
  value='-rw-r--r-- 1 u g 8 Nov 13 09:21 a\n  3 passing (77ms)\n1 passed in 0.21s\n  ✔ task (102ms)\n'
  self.assertEqual(norm.transform(value,'baseline')[0],norm.legacy.transform(value,'canonical')[0])
  self.assertIn('1 passed in 0.21s',norm.transform(value,'baseline')[0])
 def test_pytest_including_subtests_preserves_counts_status_and_order(self):
  for value in ['1 failed, 769 passed, 1 deselected, 21203 subtests passed in 31.80s',
                '=== 2 failed, 6 passed in 1.21s ===','2 warnings in 61.01s (0:01:01)']:
   actual,receipt=norm.transform(value,'extended')
   expected=value[:value.index(' in ')+4]+'0s'+(' ===' if value.endswith(' ===') else '')
   self.assertEqual(actual,expected);self.assertEqual(receipt['eligible_fields']['pytest_summary_duration'],1)
 def test_unittest_timing_is_not_test_count(self):
  actual,_=norm.transform('Ran 2 tests in 0.032s\nFAILED (errors=1)\n','extended')
  self.assertEqual(actual,'Ran 2 tests in 0s\nFAILED (errors=1)\n')
 def test_optional_ava_suffix_presence_converges(self):
  a='  ✔ task\n  ✔ other (2.3s)\n';b='  ✔ task (101ms)\n  ✔ other (2s)\n'
  self.assertEqual(norm.transform(a,'extended')[0],norm.transform(b,'extended')[0])
  self.assertEqual(norm.transform(a,'extended')[0],'  ✔ task\n  ✔ other\n')
 def test_failure_marker_test_title_and_order_remain(self):
  value='  ✘ slow task (101ms)\n  ✔ faster task (2s)\n'
  self.assertEqual(norm.transform(value,'extended')[0],'  ✘ slow task\n  ✔ faster task\n')
 def test_error_times_source_literals_unknown_formats_preserved(self):
  value='assert elapsed == 0.21\nERROR: expired in 0.21s\ntext = "1 passed in 0.21s"\nTask took 0.21s\nPID=5321\n0xab12\n  ✔ task (date 2026-11-13)\n'
  self.assertEqual(norm.transform(value,'extended')[0],value)
 def test_middle_of_test_name_is_not_removed(self):
  value='  ✔ timeout (200ms) should raise\n'
  self.assertEqual(norm.transform(value,'extended')[0],value)
 def test_clipped_last_line_unchanged_even_if_suffix_looks_complete(self):
  value='  ✔ potentially longer title (123ms)'+norm.MARKER
  self.assertEqual(norm.transform(value,'extended')[0],value)
 def test_cap_never_reveals_hidden_tail(self):
  value='x'*2370+'\n1 passed in 0.21s\n';value=value[:2400]+norm.MARKER
  actual,_=norm.transform(value,'extended')
  self.assertLessEqual(len(actual),len(value));self.assertTrue(actual.endswith(norm.MARKER))
  with self.assertRaises(ValueError):norm.transform('x'*2401,'extended')
 def test_line_endings_and_idempotence(self):
  value='1 passed in 0.21s\r\n  ✔ task (102ms)\r\n'
  actual,_=norm.transform(value,'extended');self.assertEqual(actual,'1 passed in 0s\r\n  ✔ task\r\n')
  self.assertEqual(norm.transform(actual,'extended')[0],actual)
 def test_real_wrapper_changes_agent_feedback_but_not_grader(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);kind=agent.recorded_environment(out,'extended')
   def env(path):
    v=kind(path,python_site=out,node_root=out)
    v.sandbox=SimpleNamespace(execute=lambda *a,**k:{'stdout':'1 failed in 0.21s\n','stderr':'','returncode':7,'reason':'runtime_error'})
    return v
   result=env(out/'workspace').execute({'command':'pytest'})
   self.assertEqual(result['output'],'1 failed in 0s\n');self.assertEqual(result['returncode'],7)
   self.assertEqual(env(out/'grader').execute({'command':'grade'})['output'],'1 failed in 0.21s\n')
   rows=(out/'observations.jsonl').read_text().splitlines();self.assertEqual(len(rows),1)
   self.assertEqual(json.loads(rows[0])['raw_output'],'1 failed in 0.21s\n')

class RepeatedDesign(unittest.TestCase):
 def test_budget_and_model_profiles_unchanged(self):
  now=json.loads((EXP/'plan.json').read_text());old=json.loads((EXP.parent/'023-evaluation-stability/plan.json').read_text())
  for key in ['agent','models','policy','budget','probes']:self.assertEqual(now[key],old[key])
  self.assertEqual(now['conditions'],['baseline','extended']);self.assertFalse(now['final_test_opened'])
  self.assertEqual((EXP/'profile.json').read_bytes(),(EXP.parent/'023-evaluation-stability/profile.json').read_bytes())
 def test_unique_counterbalanced270_jobs(self):
  jobs=[]
  for b in schedule.blocks():
   rows=schedule.task_schedule([str(i) for i in range(15)],b)
   for x,y in zip(rows[::2],rows[1::2]):self.assertEqual(x['id'],y['id']);self.assertEqual({x['condition'],y['condition']},{'baseline','extended'})
   jobs += [(b['model'],b['repeat'],r['id'],r['condition']) for r in rows]
  self.assertEqual(len(jobs),270);self.assertEqual(len(set(jobs)),270)
 def test_action_variation_can_exist_without_grade_variation(self):
  campaigns={};audit={'campaigns':{}}
  for m in schedule.MODELS:
   for c in schedule.CONDITIONS:
    for repeat in [1,2,3]:
     name=f'r{repeat:02d}-{m}-{c}';rows=[];traces=[]
     for i in range(15):
      command='different' if i==0 and repeat==2 and c=='baseline' else 'same'
      trace={'id':str(i),'calls':1,'passed':False,'generation_input_sha256':['i'],'generation_output_sha256':[command],
       'observation_receipts':[{'model_call':1,'command_sha256':command,'prior_editable_state_sha256':'s','returncode':0,'reason':'ok','raw_sha256':'r','canonical_sha256':'c','delivered_sha256':'r'}],
       'source_progress':[{'call_index':1,'changed_editable_files':0}]}
      traces.append(trace);rows.append({'id':str(i),'passed':False,'calls':1,'generated_tokens':1,'seconds':1,'source_progress':trace['source_progress'],'generations':[{'action_format_valid':True}]})
     campaigns[name]={'complete':True,'results':rows};audit['campaigns'][name]={'model':m,'condition':c,'repeat':repeat,'tasks':traces}
  value=analysis.summarize(campaigns,audit);self.assertEqual(len(value['pairs']),405)
  for row in value['stability']:
   self.assertEqual(row['pairwise_disagreements'],0)
   self.assertEqual(row['mixed_action_tasks'],1 if row['condition']=='baseline' else 0)
   self.assertEqual(row['action_pairwise_disagreements'],2 if row['condition']=='baseline' else 0)
 def test_frozen_sources_if_present(self):
  p=EXP/'protocol.json'
  if not p.exists():self.skipTest('CPU preparation precedes freeze')
  import hashlib
  for name,digest in json.loads(p.read_text())['frozen_files'].items():
   path=ROOT/name
   if path.is_file():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest,name)

if __name__=='__main__':unittest.main()
