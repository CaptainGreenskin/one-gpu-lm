"""Cost and provenance boundaries for model-produced rollout collection (CPU fixtures)."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from one_gpu_lm.verified import tasks,rollout
from one_gpu_lm.verified.sandbox import Sandbox


class RolloutBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.data=self.root/'data';self.snapshot=self.data/'snapshots/a--b';self.snapshot.mkdir(parents=True)
        (self.snapshot/'logic.py').write_text('answer = 7\n');(self.snapshot/'LICENSE').write_text('fixture notice\n')
        self.source={'repository':'a/b','revision':'a'*40,'family':'family','split':'training',
                     'approved_for':{'training':True},'notice_review_complete':True,'lineage_review_complete':True,
                     'notices':['LICENSE'],'files':tasks.files(self.snapshot)}
        self.task={'id':'fixture','repository':'a/b','revision':'a'*40,'family':'family','split':'training','description':'Repair the result',
                   'editable_files':['logic.py'],'mutations':[{'path':'logic.py','before':'7','after':'3'}],
                   'visible_checks':{'check.py':'visible check'},'hidden_checks':{'hidden.py':'HIDDEN_CASE_SENTINEL'},
                   'regression_command':'python3 check.py','grading_command':'python3 hidden.py'}
        self.protocol={'model_id':'fixture/base','revision':'a'*40,'teacher_profile_sha256':'profile-fixture',
                       'teacher_budget':{'max_model_calls':3,'max_task_seconds':60,'max_total_generated_tokens':30,
                       'max_input_tokens':8192,'max_generated_tokens_per_call':10,'max_tool_seconds':5,'max_tool_output_chars':2400},
                       'agent_policy':{'format':'native_roundtrip','context':'bounded_history_v1'}}
        self.requests=[];self.actions=[]
        self.health={'model_id':'fixture/base','revision':'a'*40,'profile_sha256':'profile-fixture','adapter':False}
        self.normalizer=SimpleNamespace(MARKER='[TRUNCATED]',transform=lambda output,*_: (output,{'fixture':True}))
    def request(self,socket,payload,**_):
        if payload['operation']=='health':return self.health
        self.requests.append(copy.deepcopy(payload))
        command=self.actions[len(self.requests)-1]
        return {'text':'```bash\n'+command+'\n```','ids':[1,2,3],'generated_tokens':3,'input_tokens':20,'input_tokens_sha256':'fixture'}
    def execute(self,root,command):
        if command=='edit-source':(root/'logic.py').write_text('answer = 7\n')
        if command=='tamper-test':(root/'check.py').write_text('tampered')
        if command=='python3 hidden.py':self.assertEqual((root/'hidden.py').read_text(),'HIDDEN_CASE_SENTINEL')
        return {'returncode':0,'reason':'ok','output':'PASS','seconds':.001}
    def collect(self):
        sandbox=SimpleNamespace(execute=self.execute,timeout=5)
        with patch.object(rollout,'request',self.request),patch.object(rollout,'normalization',return_value=self.normalizer):
            return rollout.run(self.root,self.data,self.task,self.source,'socket',sandbox,self.protocol,self.root/'out')
    def test_verified_episode_is_model_produced_and_no_answer_enters_requests(self):
        self.actions=['edit-source','python3 check.py','echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT']
        value=self.collect();self.assertTrue(value['eligible_verified_trajectory']);self.assertEqual(value['generated_tokens'],9)
        self.assertEqual(value['producer_kind'],'model');self.assertEqual(value['attempted_calls'],3)
        self.assertEqual(value['first_edit_call'],1)
        for r in self.requests:
            text=json.dumps(r);self.assertNotIn('HIDDEN_CASE_SENTINEL',text);self.assertNotIn('mutations',text)
        self.assertEqual(self.requests[1]['messages'][-2]['extra']['actions'],[{'command':'edit-source'}])
        self.assertIn('tool',self.requests[1]['messages'][-1]['extra'])
    def test_correct_patch_without_agent_check_remains_ineligible(self):
        self.actions=['edit-source','echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT']
        value=self.collect();self.assertTrue(value['grade']['passed']);self.assertFalse(value['eligible_verified_trajectory'])
        self.assertEqual(value['generated_tokens'],6)
    def test_test_tampering_cannot_become_a_positive_trajectory(self):
        self.actions=['tamper-test','python3 check.py','echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT']
        value=self.collect();self.assertFalse(value['eligible_verified_trajectory'])
        self.assertEqual(value['grade']['reason'],'protected_file_tampering')
    def test_backend_timeout_is_not_silently_retried_or_zero_cost_success(self):
        real=self.request
        def failure(socket,payload,**kwargs):
            if payload['operation']=='generate':raise TimeoutError()
            return real(socket,payload,**kwargs)
        self.request=failure
        with self.assertRaises(RuntimeError):self.collect()
        value=json.loads((self.root/'out/trajectory.json').read_text())
        self.assertFalse(value['complete']);self.assertTrue(value['unknown_inflight_generation_cost'])
        self.assertEqual(value['attempted_calls'],1);self.assertFalse((self.root/'out/result.json').exists())


class RuntimeInventory(unittest.TestCase):
    def test_grouping_cannot_expose_extra_package_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'python3').write_text('fixture')
            runtime={'mounts':[{'source':str(root/'python3'),'destination':'/usr/bin/python3','sha256':tasks.digest(b'fixture')}],
                     'mount_roots':[{'source':str(root),'destination':'/usr/bin'}],'environment':{}}
            Sandbox(runtime)
            (root/'unmodified_answer.py').write_text('answer')
            with self.assertRaises(ValueError):Sandbox(runtime)


if __name__=='__main__':unittest.main()
