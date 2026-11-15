import copy
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from one_gpu_lm.verified.evaluation import validate_scope,validate_candidate,Backend
from one_gpu_lm.verified.evaluation_watch import backend_matches
from one_gpu_lm.verified.factorial import ARMS,schedule


class ScopeAndCandidateGates(unittest.TestCase):
    def setUp(self):
        self.tasks=[{'id':f't{i}','repository':'new/repo','family':'new','revision':'a'*40,'split':'development'} for i in range(10)]
        self.sources=[{'repository':'new/repo','family':'new','revision':'a'*40,'split':'development','language':'python','approved_for':{'development':True}}]
        self.train=[{'repository':'old/repo','family':'old','revision':'b'*40,'split':'training'}]
        self.spec={'development':{'unique_tasks':10,'repository_families':1,'languages':1,'repetitions':3,'total_task_runs':120,
                    'arms':ARMS,'common_budget':{'max_model_calls':12},'tasks_sha256':'tasks','sources_sha256':'sources'}}
        self.protocol={'evaluation_budget':{'max_model_calls':12},'schedule':schedule([t['id'] for t in self.tasks])}
        self.admission={'complete':True,'passed':True,'tasks_sha256':'tasks','sources_sha256':'sources','runtime_sha256':'runtime',
                        'tasks':[{'id':t['id'],'admitted':True} for t in self.tasks]}
    def check(self):
        validate_scope(self.protocol,self.spec,self.tasks,self.sources,self.train,[],self.admission,'tasks','sources','runtime')
    def test_exact_admitted_isolated_schedule_accepted(self):self.check()
    def test_changed_budget_or_order_cannot_be_selected_after_scores(self):
        self.protocol['evaluation_budget']['max_model_calls']=13
        with self.assertRaises(ValueError):self.check()
        self.protocol['evaluation_budget']['max_model_calls']=12;self.protocol['schedule'].reverse()
        with self.assertRaises(ValueError):self.check()
    def test_family_leakage_and_final_tasks_rejected(self):
        self.sources[0]['related_families']=['old']
        with self.assertRaises(ValueError):self.check()
        self.sources[0].pop('related_families');self.tasks[0]['split']='final'
        with self.assertRaises(ValueError):self.check()
    def test_partial_candidate_cannot_be_evaluated(self):
        candidate=dict.fromkeys(['complete','frozen_files_unchanged','saved_reload_greedy_ids_equal','all_selected_records_consumed',
                                'frozen_sample_unchanged','frozen_gradients_absent','adapter_changed'],True)
        candidate.update(protocol_sha256='protocol',optimizer_steps=10,candidate_count=1)
        supervisor={'complete':True,'candidate_eligible':True,'timed_out':False,'worker_returncode':0}
        validate_candidate(candidate,supervisor,'protocol')
        candidate['all_selected_records_consumed']=False
        with self.assertRaises(ValueError):validate_candidate(candidate,supervisor,'protocol')
    def test_cleanup_matches_only_owned_socket_and_directory(self):
        args=['python','-m','one_gpu_lm.serving.local_backend','--socket','/fixture/run/backend.sock']
        self.assertTrue(backend_matches(args,'/fixture','/fixture',Path('/fixture/run/backend.sock')))
        self.assertFalse(backend_matches(args,'/other','/fixture',Path('/fixture/run/backend.sock')))
        self.assertFalse(backend_matches(args,'/fixture','/fixture',Path('/fixture/other/backend.sock')))
        args[2]='one_gpu_lm.verified.evaluation_backend'
        self.assertTrue(backend_matches(args,'/fixture','/fixture',Path('/fixture/run/backend.sock')))
    def test_foreign_gpu_compute_prevents_new_scores_without_killing_it(self):
        backend=Backend('/fixture','/fixture/run',{}, {},[])
        backend.process=SimpleNamespace(pid=123)
        with patch('one_gpu_lm.verified.evaluation.subprocess.check_output',return_value='123\n'):
            backend.check_exclusive()
        with patch('one_gpu_lm.verified.evaluation.subprocess.check_output',return_value='123\n456\n'):
            with self.assertRaises(RuntimeError):backend.check_exclusive()


if __name__=='__main__':unittest.main()
