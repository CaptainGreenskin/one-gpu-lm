import copy
import unittest
from one_gpu_lm.verified.collection import validate_inventory


class FrozenCollectionScope(unittest.TestCase):
    def setUp(self):
        self.tasks=[{'id':'a','repository':'example/repo','revision':'a'*40,'family':'example','split':'training'}]
        self.sources=[{'repository':'example/repo','revision':'a'*40,'family':'example','split':'training',
                       'approved_for':{'training':True}}]
        self.protocol={'task_ids':['a'],'attempts_per_task':1}
        self.admission={'complete':True,'passed':True,'tasks_sha256':'tasks','sources_sha256':'sources',
                        'runtime_sha256':'runtime','tasks':[{'id':'a','admitted':True}]}
    def check(self):
        validate_inventory(self.protocol,self.tasks,self.sources,self.admission,'tasks','sources','runtime')
    def test_exact_admitted_training_pool_accepted(self):
        self.check()
    def test_changed_tasks_sources_and_runtime_rejected(self):
        for field in ['tasks_sha256','sources_sha256','runtime_sha256']:
            original=copy.deepcopy(self.admission);self.admission[field]='changed'
            with self.subTest(field=field),self.assertRaises(ValueError):self.check()
            self.admission=original
    def test_development_task_and_retry_rejected(self):
        self.tasks[0]['split']='development'
        with self.assertRaises(ValueError):self.check()
        self.tasks[0]['split']='training';self.protocol['attempts_per_task']=2
        with self.assertRaises(ValueError):self.check()
    def test_summary_cannot_hide_missing_or_rejected_task(self):
        self.admission['tasks'][0]['admitted']=False
        with self.assertRaises(ValueError):self.check()
        self.admission['tasks']=[]
        with self.assertRaises(ValueError):self.check()
    def test_duplicate_task_and_revision_changes_rejected(self):
        self.tasks.append(copy.deepcopy(self.tasks[0]))
        with self.assertRaises(ValueError):self.check()
        self.tasks.pop();self.tasks[0]['revision']='b'*40
        with self.assertRaises(ValueError):self.check()
    def test_unapproved_source_rejected(self):
        self.sources[0]['approved_for']['training']=False
        with self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()
