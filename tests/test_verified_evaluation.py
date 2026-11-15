import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tests import test_verified_rollout as fixtures
from one_gpu_lm.verified import evaluation_rollout as ev
from one_gpu_lm.verified.submission import SubmissionGuard, SUBMIT


class EvaluationIsolation(unittest.TestCase):
    def setUp(self):
        self.c=fixtures.RolloutBoundaries();self.c.setUp();self.addCleanup(self.c.doCleanups)
        c=self.c;c.task['split']='development';c.source['split']='development';c.source['approved_for']={'development':True}
        c.protocol['evaluation_budget']=c.protocol.pop('teacher_budget')
        c.protocol['evaluation_profile_sha256']=c.protocol.pop('teacher_profile_sha256')
        c.protocol['adapter_sha256']={'adapter.safetensors':'candidate-sha'}
    def evaluate(self,arm='base_current'):
        c=self.c;runner=SimpleNamespace(execute=c.execute,timeout=5)
        if arm.endswith('_guard'):
            runner=SubmissionGuard(runner,{str(c.root/'out/workspace'):{'editable_files':['logic.py'],'regression_command':'python3 check.py'}})
        with patch.object(ev,'request',c.request),patch.object(ev,'normalization',return_value=c.normalizer):
            return ev.run(c.root,c.data,c.task,c.source,'socket',runner,c.protocol,c.root/'out',arm)
    def test_current_counts_correct_patch_separately_from_verification(self):
        self.c.actions=['edit-source',SUBMIT]
        result=self.evaluate();self.assertTrue(result['success_within_budget'])
        self.assertFalse(result['eligible_verified_trajectory']);self.assertEqual(result['split'],'development')
        for request in self.c.requests:self.assertNotIn('HIDDEN_CASE_SENTINEL',json.dumps(request))
    def test_guard_rejection_consumes_same_budget_and_requires_model_verification(self):
        self.c.actions=['edit-source',SUBMIT,'python3 check.py']
        result=self.evaluate('base_guard')
        self.assertTrue(result['success_within_budget']);self.assertFalse(result['submitted'])
        self.assertEqual(result['attempted_calls'],3);self.assertEqual(result['generated_tokens'],9)
        trace=json.loads((self.c.root/'out/trajectory.json').read_text())
        self.assertFalse(trace['steps'][1]['observation']['executed'])
    def test_student_weights_and_base_weights_cannot_be_confused(self):
        self.c.actions=['edit-source',SUBMIT]
        with self.assertRaises(ValueError):self.evaluate('student_current')
        self.assertEqual(self.c.requests,[])
    def test_exact_student_artifact_is_accepted(self):
        self.c.health.update(adapter=True,adapter_sha256={'adapter.safetensors':'candidate-sha'})
        self.c.actions=['edit-source',SUBMIT]
        self.assertTrue(self.evaluate('student_current')['success_within_budget'])
    def test_training_or_final_tasks_cannot_enter_development_evaluator(self):
        self.c.task['split']='final'
        with self.assertRaises(ValueError):self.evaluate()
        self.assertEqual(self.c.requests,[])
    def test_patch_finishing_after_actor_cap_is_not_an_on_budget_success(self):
        clock={'now':0.0};execute=self.c.execute
        def slow(root,command):
            result=execute(root,command)
            if command=='edit-source':clock['now']=100.0
            return result
        self.c.execute=slow;self.c.actions=['edit-source']
        with patch.object(ev,'time',SimpleNamespace(monotonic=lambda:clock['now'],time=lambda:1000.0+clock['now'])):
            result=self.evaluate()
        self.assertTrue(result['grade']['passed'])
        self.assertFalse(result['actor_budget_compliant']);self.assertFalse(result['success_within_budget'])
        self.assertEqual(result['terminal_reason'],'time_limit')


if __name__=='__main__':unittest.main()
