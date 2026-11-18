import copy
from pathlib import Path
import tempfile
import unittest
from one_gpu_lm.verified import historic_tasks as history
from one_gpu_lm.verified.historic_admit import semantic_failure
from one_gpu_lm.verified.tasks import files


class HistoricalBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.snapshot=self.root/'snapshot';self.snapshot.mkdir()
        for name,text in {'one.py':'value = 1\n','two.py':'value = 2\n','LICENSE':'retained notice\n','test.py':'protected test\n'}.items():
            (self.snapshot/name).write_text(text)
        self.source={'source_id':'owner--project--'+'b'*40,'repository':'owner/project',
                     'family':'owner/project','revision':'b'*40,'upstream_parent':'a'*40,
                     'split':'training','approved_for':{'training':True},'notice_review_complete':True,
                     'lineage_review_complete':True,'notices':['LICENSE'],'files':files(self.snapshot)}
        self.task={**{k:self.source[k] for k in ['source_id','repository','family','revision','upstream_parent','split']},
                   'id':'two-files','origin':'upstream_history_parent_plus_test_patch',
                   'editable_files':['one.py','two.py'],
                   'mutations':[{'path':'one.py','before':'value = 1','after':'value = 0'},
                                {'path':'two.py','before':'value = 2','after':'value = 0'}],
                   'visible_checks':{},'hidden_checks':{},'grading_command':'python3 test.py'}

    def test_real_parent_bytes_and_fresh_protected_grading(self):
        actor=self.root/'actor';before=history.prepare(self.snapshot,actor,self.source,self.task)
        self.assertEqual((actor/'one.py').read_text(),'value = 0\n')
        (actor/'one.py').write_text('value = 1\n');(actor/'two.py').write_text('value = 2\n')
        def check(path,command):
            self.assertNotEqual(path,actor);self.assertEqual((path/'test.py').read_text(),'protected test\n')
            return {'returncode':0,'reason':'ok','output':'passed','seconds':0}
        self.assertTrue(history.grade(self.snapshot,self.root/'grader',self.source,self.task,actor,before,check)['passed'])
        (actor/'test.py').write_text('pass\n')
        self.assertEqual(history.grade(self.snapshot,self.root/'tampered',self.source,self.task,actor,before,check)['reason'],'protected_file_tampering')

    def test_parent_and_commit_identity_cannot_be_swapped(self):
        changed=copy.deepcopy(self.task);changed['upstream_parent']='c'*40
        with self.assertRaises(ValueError):history.prepare(self.snapshot,self.root/'bad',self.source,changed)
        changed=copy.deepcopy(self.source);changed['source_id']='owner--other--'+'b'*40
        with self.assertRaises(ValueError):history.adapt(changed,dict(self.task,source_id=changed['source_id']))

    def test_multiple_revisions_are_one_family_and_cannot_cross_split(self):
        second=copy.deepcopy(self.source);second.update(source_id='owner--project--'+'c'*40,revision='c'*40)
        self.assertEqual(history.validate_split([self.source,second])['training'],{'owner/project'})
        second.update(split='development',family='pretend-new-family')
        with self.assertRaises(ValueError):history.validate_split([self.source,second])
        second.update(repository='new/repo',source_id='new--repo--'+'c'*40,family='new/repo')
        with self.assertRaises(ValueError):history.validate_split([second],['NEW/REPO'])

    def test_environment_errors_are_not_bug_reproduction(self):
        for output,reason,code in [('1 failed: ModuleNotFoundError: missing','runtime_error',1),
                                  ('not ok 1: Cannot find module','runtime_error',1),
                                  ('1 failed','timeout',-1),('1 passed','ok',0)]:
            self.assertFalse(semantic_failure({'output':output,'reason':reason,'returncode':code}))
        self.assertTrue(semantic_failure({'output':'AssertionError\n1 failed, 3 passed','reason':'runtime_error','returncode':1}))


if __name__=='__main__':unittest.main()
