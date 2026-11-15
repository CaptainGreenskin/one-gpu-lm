import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from one_gpu_lm.verified.submission import SubmissionGuard,SUBMIT
class GuardContract(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.p=self.root/'source.py';self.p.write_text('value=1\n');self.calls=[];self.rc=0
  def execute(root,command):
   self.calls.append(command);return {'returncode':self.rc,'reason':'ok' if self.rc==0 else 'runtime_error','output':'fixture','seconds':.001}
  self.runner=SimpleNamespace(timeout=3,execute=execute)
  self.guard=SubmissionGuard(self.runner,{str(self.root):{'editable_files':['source.py'],'regression_command':'python3 check.py'}})
 def test_rejected_submit_is_not_faked_as_execution(self):
  result=self.guard.execute(self.root,SUBMIT);self.assertFalse(result['executed']);self.assertEqual(self.calls,[]);self.assertEqual(result['reason'],'submission_requires_verification')
 def test_check_then_submit_executes_both(self):
  self.guard.execute(self.root,'python3 check.py');result=self.guard.execute(self.root,SUBMIT)
  self.assertTrue(result['executed']);self.assertEqual(self.calls,['python3 check.py',SUBMIT])
 def test_stale_source_and_failed_check_do_not_allow_submit(self):
  self.guard.execute(self.root,'python3 check.py');self.p.write_text('value=2\n');self.assertFalse(self.guard.execute(self.root,SUBMIT)['executed'])
  self.rc=1;self.guard.execute(self.root,'python3 check.py');self.assertFalse(self.guard.execute(self.root,SUBMIT)['executed'])
 def test_compound_command_does_not_claim_exact_check(self):
  self.guard.execute(self.root,'cd /work && python3 check.py');self.assertFalse(self.guard.execute(self.root,SUBMIT)['executed'])
 def test_failed_repeat_check_invalidates_old_success(self):
  self.guard.execute(self.root,'python3 check.py');self.rc=1;self.guard.execute(self.root,'python3 check.py')
  self.assertFalse(self.guard.execute(self.root,SUBMIT)['executed'])
 def test_reverting_source_does_not_restore_old_verification(self):
  self.guard.execute(self.root,'python3 check.py');self.p.write_text('value=2\n');self.guard.execute(self.root,'inspect')
  self.p.write_text('value=1\n');self.assertFalse(self.guard.execute(self.root,SUBMIT)['executed'])
 def test_independent_grader_not_intercepted_and_timeout_forwards(self):
  grader=self.root/'grader';grader.mkdir();self.guard.timeout=19;self.assertEqual(self.runner.timeout,19)
  self.assertTrue(self.guard.execute(grader,'python3 hidden.py')['executed'])
class GuardIntegration(unittest.TestCase):
 def test_complete_rollout_counts_rejection_and_requires_actual_check(self):
  import json
  from unittest.mock import patch
  from tests import test_verified_rollout as fixtures
  from one_gpu_lm.verified import rollout
  c=fixtures.RolloutBoundaries();c.setUp();self.addCleanup(c.doCleanups)
  c.actions=['edit-source',SUBMIT,'python3 check.py',SUBMIT];c.protocol['teacher_budget']['max_model_calls']=4
  runner=SimpleNamespace(execute=c.execute,timeout=5)
  guard=SubmissionGuard(runner,{str(c.root/'out/workspace'):{'editable_files':['logic.py'],'regression_command':'python3 check.py'}})
  with patch.object(rollout,'request',c.request),patch.object(rollout,'normalization',return_value=c.normalizer):
   value=rollout.run(c.root,c.data,c.task,c.source,'socket',guard,c.protocol,c.root/'out')
  self.assertTrue(value['eligible_verified_trajectory']);self.assertEqual(value['attempted_calls'],4);self.assertEqual(value['generated_tokens'],12)
  trace=json.loads((c.root/'out/trajectory.json').read_text())
  self.assertFalse(trace['steps'][1]['observation']['executed']);self.assertTrue(trace['steps'][2]['observation']['executed'])
  self.assertEqual(trace['steps'][2]['command'],'python3 check.py')
  self.assertIn('Submission rejected by the collection policy',c.requests[2]['messages'][-1]['content'])
 def test_hidden_specification_cannot_enter_guard(self):
  with self.assertRaises(ValueError):SubmissionGuard(None,{'/fixture':{'editable_files':['x.py'],'regression_command':'check','hidden_checks':{}}})
 def test_rejected_and_missing_execution_never_become_positive_targets(self):
  from one_gpu_lm.verified.guard_audit import target_is_executed
  self.assertFalse(target_is_executed({'action_format_valid':True,'observation':{'executed':False}}))
  self.assertFalse(target_is_executed({'action_format_valid':True}))
  self.assertFalse(target_is_executed({'action_format_valid':False,'observation':{'executed':True}}))
  self.assertTrue(target_is_executed({'action_format_valid':True,'observation':{'executed':True}}))

if __name__=='__main__':unittest.main()
