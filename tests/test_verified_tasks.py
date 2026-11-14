"""Admission and clean-grader boundaries for future verified distillation."""
import copy
from pathlib import Path
import tempfile
import unittest
from one_gpu_lm.verified import tasks


class VerifiedTasks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root/'snapshot'; self.snapshot.mkdir()
        (self.snapshot/'logic.py').write_text('def add(a,b): return a+b\n')
        (self.snapshot/'LICENSE').write_text('fixture notice\n')
        self.source = {'repository':'example/project','revision':'a'*40,'family':'example-family','split':'training',
                       'approved_for':{'training':True},'notice_review_complete':True,'lineage_review_complete':True,
                       'notices':['LICENSE'],'files':tasks.files(self.snapshot)}
        self.task = {'id':'example','repository':'example/project','revision':'a'*40,'family':'example-family','split':'training',
                     'editable_files':['logic.py'],'mutations':[{'path':'logic.py','before':'a+b','after':'a-b'}],
                     'visible_checks':{'check.py':'visible example\n'},'hidden_checks':{'holdout.py':'private case\n'},
                     'grading_command':'python3 holdout.py'}
    def test_only_explicit_purpose_approval_works(self):
        for value in [False,'approved',1,None]:
            source=copy.deepcopy(self.source);source['approved_for']['training']=value
            with self.subTest(value=value),self.assertRaises(ValueError):tasks.validate_source(source,self.snapshot,'training')
        with self.assertRaises(ValueError):tasks.validate_source(self.source,self.snapshot,'development')
    def test_family_aliases_and_previous_development_sources_rejected(self):
        train=copy.deepcopy(self.source);dev=copy.deepcopy(self.source)
        dev.update(repository='different-name/fork',split='development')
        with self.assertRaises(ValueError):tasks.validate_split([train,dev])
        dev['family']='new-project';dev['related_families']=['example-family']
        with self.assertRaises(ValueError):tasks.validate_split([train,dev])
        dev.pop('related_families');dev['family']='old-project'
        with self.assertRaises(ValueError):tasks.validate_split([train,dev],['old-project'])
    def test_added_distribution_copy_or_changed_notice_breaks_admission(self):
        (self.snapshot/'dist.py').write_text((self.snapshot/'logic.py').read_text())
        with self.assertRaises(ValueError):tasks.validate_source(self.source,self.snapshot,'training')
        (self.snapshot/'dist.py').unlink();(self.snapshot/'LICENSE').write_text('changed')
        with self.assertRaises(ValueError):tasks.validate_source(self.source,self.snapshot,'training')
    def test_agent_snapshot_has_fault_and_visible_checks_only(self):
        workspace=self.root/'agent';tasks.prepare(self.snapshot,workspace,self.source,self.task)
        self.assertIn('a-b',(workspace/'logic.py').read_text());self.assertFalse((workspace/'holdout.py').exists())
        self.assertEqual((self.snapshot/'logic.py').read_text(),'def add(a,b): return a+b\n')
        with self.assertRaises(ValueError):tasks.prepare(self.snapshot,workspace,self.source,self.task)
    def test_grader_copies_only_patch_into_independent_checks(self):
        workspace=self.root/'agent';before=tasks.prepare(self.snapshot,workspace,self.source,self.task)
        (workspace/'logic.py').write_text('def add(a,b): return sum((a,b))\n')
        (workspace/'holdout.py').write_text('attacker supplied check\n')
        def execute(root,command):
            self.assertEqual((root/'holdout.py').read_text(),'private case\n')
            self.assertEqual((root/'check.py').read_text(),'visible example\n')
            self.assertIn('sum(', (root/'logic.py').read_text())
            return {'returncode':0,'reason':'ok','output':'PASS','seconds':.01}
        result=tasks.grade(self.snapshot,self.root/'grader',self.source,self.task,workspace,before,execute)
        self.assertTrue(result['passed']);self.assertEqual(result['ignored_new_files'],['holdout.py'])
    def test_modifying_tests_or_no_edit_cannot_pass(self):
        workspace=self.root/'agent';before=tasks.prepare(self.snapshot,workspace,self.source,self.task)
        def execute(*_):self.fail('Invalid submission reached grader')
        result=tasks.grade(self.snapshot,self.root/'grader',self.source,self.task,workspace,before,execute)
        self.assertEqual(result['reason'],'no_source_change')
        (workspace/'check.py').write_text('tampered')
        result=tasks.grade(self.snapshot,self.root/'grader',self.source,self.task,workspace,before,execute)
        self.assertEqual(result['reason'],'protected_file_tampering')
    def test_unsafe_paths_and_symlinks_are_rejected(self):
        for name in ['../outside','/absolute','.git/config','node_modules/answer.js','a/../b','a//b']:
            with self.subTest(name=name),self.assertRaises(ValueError):tasks.relative(name)
        (self.snapshot/'link.py').symlink_to(self.snapshot/'logic.py')
        with self.assertRaises(ValueError):tasks.files(self.snapshot)


if __name__=='__main__':unittest.main()
