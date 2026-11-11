import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from one_gpu_lm.common.edit_progress import EditableProgress, identical_command_streak
from one_gpu_lm.common.agent_policy import validate_policy


class EditProgressContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'code.py'
        self.path.write_text('def f(): return 0\n')
        self.original = self.path.read_bytes()
        self.monitor = EditableProgress(self.root, {'code.py': hashlib.sha256(self.original).hexdigest()}, ['code.py'])

    def history(self, commands=()):
        result = [{'role':'system','content':'rules'}, {'role':'user','content':'task'}]
        for command in commands:
            result += [{'role':'assistant','content':'action','extra':{'actions':[{'command':command}]}},
                       {'role':'user','content':'actual output','extra':{'tool':{'returncode':0}}}]
        return result

    def test_control_exact_identity_and_candidate_threshold(self):
        ms=self.history();before=copy.deepcopy(ms)
        for used in range(12):
            packed,receipt=self.monitor.prepare_request(ms,used,12,False)
            self.assertIs(packed,ms);self.assertFalse(receipt['reminder'])
            packed,receipt=self.monitor.prepare_request(ms,used,12,True)
            self.assertEqual(receipt['reminder'],used>=4)
            self.assertEqual(receipt['remaining_calls'],12-used)
            self.assertEqual(packed[1:],ms[1:])
        self.assertEqual(ms,before)

    def test_real_edit_clears_notice_and_reverting_restores_it(self):
        ms=self.history()
        self.path.write_text('def f(): return 1\n')
        _,r=self.monitor.prepare_request(ms,8,12,True)
        self.assertEqual(r['changed_editable_files'],1);self.assertFalse(r['reminder'])
        self.path.write_bytes(self.original)
        _,r=self.monitor.prepare_request(ms,8,12,True)
        self.assertEqual(r['changed_editable_files'],0);self.assertTrue(r['reminder'])

    def test_identical_execution_triggers_early_but_format_errors_do_not(self):
        ms=self.history(['probe','probe']);packed,r=self.monitor.prepare_request(ms,2,12,True)
        self.assertEqual(r['identical_command_streak'],2);self.assertTrue(r['reminder'])
        self.assertIn('last 2 executed commands',packed[0]['content'])
        ms += [{'role':'assistant','content':'bad','extra':{}},{'role':'user','content':'format feedback'}]
        self.assertEqual(identical_command_streak(ms),2)
        self.assertEqual(identical_command_streak(self.history(['a','a','b'])),1)

    def test_deleted_or_escaping_source_is_not_progress(self):
        self.path.unlink()
        r=self.monitor.snapshot();self.assertEqual(r['invalid_editable_files'],1);self.assertEqual(r['changed_editable_files'],0)
        self.path.symlink_to('/nonexistent-private-target')
        r=self.monitor.snapshot();self.assertEqual(r['invalid_editable_files'],1);self.assertEqual(r['changed_editable_files'],0)

    def test_no_extra_budget_and_policy_is_explicit(self):
        for used in [-1,12]:
            with self.assertRaises(ValueError):self.monitor.prepare_request(self.history(),used,12,True)
        self.assertEqual(validate_policy(),{'format':'legacy','context':'full'})
        self.assertEqual(validate_policy({'format':'native_roundtrip','context':'full','progress':'edit_checkpoint_v1'})['progress'],'edit_checkpoint_v1')
        for value in ['anything',None]:
            with self.assertRaises(ValueError):validate_policy({'format':'native_roundtrip','context':'full','progress':value})

    def test_current_files_and_history_reconstruct_state_after_restart(self):
        ms=self.history(['a','a']);_,before=self.monitor.prepare_request(ms,4,12,True)
        restart=EditableProgress(self.root,self.monitor.original,list(self.monitor.original))
        _,after=restart.prepare_request(ms,4,12,True);self.assertEqual(before,after)


if __name__=='__main__':unittest.main()
