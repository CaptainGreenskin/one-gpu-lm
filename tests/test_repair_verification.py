import copy
import unittest

from one_gpu_lm.common.agent_policy import validate_policy
from one_gpu_lm.common.repair_verification import is_declared_check, prepare_verification


class RepairVerificationContracts(unittest.TestCase):
    command = 'python3 -m pytest -q tests'

    def state(self, digest='a', changed=1, invalid=0):
        return {'editable_state_sha256': digest*64, 'changed_editable_files': changed,
                'invalid_editable_files': invalid}

    def history(self, command=None, code=0, before=None, after=None):
        before = before or self.state()
        after = after or before
        messages = [{'role':'system','content':'rules'}, {'role':'user','content':'task'}]
        if command is None:
            return messages, []
        messages += [{'role':'assistant','content':'actual action','extra':{
            'actions':[{'command':command}], 'generation':{'progress':before}}},
            {'role':'user','content':'actual output','extra':{'tool':{'returncode':code}}}]
        return messages, [dict(after, call_index=1)]

    def check(self, command=None, code=0, state=None, enabled=True, before=None, after=None):
        ms, steps = self.history(command, code, before, after)
        original = copy.deepcopy(ms)
        packed, receipt = prepare_verification(ms, steps, state or self.state(), 1 if command else 0,
                                               12, self.command, enabled)
        self.assertEqual(ms, original)
        self.assertEqual(packed[1:], ms[1:])
        return packed, receipt

    def test_control_is_exact_identity_and_only_changed_candidate_gets_notice(self):
        ms, steps = self.history()
        packed, receipt = prepare_verification(ms, steps, self.state(), 0, 12, self.command, False)
        self.assertIs(packed, ms)
        self.assertFalse(receipt['reminder'])
        self.assertTrue(self.check()[1]['reminder'])
        self.assertFalse(self.check(state=self.state(changed=0))[1]['reminder'])
        self.assertFalse(self.check(state=self.state(invalid=1))[1]['reminder'])

    def test_pipeline_and_success_wrappers_cannot_verify_source(self):
        for command in [self.command+' | tail -5', self.command+'; true', self.command+' || true',
                        self.command+' 2>&1', 'echo '+self.command, 'python3 -m pytest -q tests/subset.py']:
            self.assertFalse(is_declared_check(command, self.command))
            self.assertEqual(self.check(command)[1]['check_status'], 'not_run')
        self.assertEqual(self.check(' cd /work && '+self.command+'\n')[1]['check_status'], 'passed')

    def test_nonzero_declared_check_is_not_a_successful_repair(self):
        packed, r = self.check(self.command, 1)
        self.assertEqual(r['check_status'], 'failed')
        self.assertIn('smallest failing assertion', packed[0]['content'])
        self.assertNotIn('passed', r)

    def test_edit_after_success_invalidates_verification(self):
        _, r = self.check(self.command, state=self.state('b'))
        self.assertEqual(r['check_status'], 'stale')
        self.assertEqual(r['last_declared_check']['state_sha256'], 'a'*64)

    def test_test_mutating_source_or_invalid_source_is_not_verified(self):
        _, r = self.check(self.command, before=self.state('b'), after=self.state('a'))
        self.assertEqual(r['check_status'], 'changed_during_check')
        _, r = self.check(self.command, state=self.state(invalid=1))
        self.assertEqual(r['check_status'], 'stale')

    def test_format_errors_do_not_create_checks_and_restart_reconstructs(self):
        ms, steps = self.history(self.command, 1)
        ms += [{'role':'assistant','content':'bad','extra':{}}, {'role':'user','content':'format error'}]
        a = prepare_verification(ms, steps, self.state(), 2, 12, self.command, True)
        b = prepare_verification(copy.deepcopy(ms), copy.deepcopy(steps), self.state(), 2, 12, self.command, True)
        self.assertEqual(a, b)
        self.assertEqual(a[1]['last_declared_check']['call'], 1)
        self.assertEqual(a[1]['remaining_calls'], 10)

    def test_policy_and_budget_are_explicit(self):
        base={'format':'native_roundtrip','context':'full','progress':'edit_checkpoint_v1'}
        self.assertEqual(validate_policy(base), base)
        self.assertEqual(validate_policy(dict(base,verification='repair_check_v1'))['verification'], 'repair_check_v1')
        for policy in [dict(base,verification='unknown'), {'format':'native_roundtrip','context':'full','verification':'repair_check_v1'}]:
            with self.assertRaises(ValueError): validate_policy(policy)
        for used in [-1,12]:
            with self.assertRaises(ValueError): prepare_verification(*self.history(),self.state(),used,12,self.command,True)


if __name__ == '__main__':
    unittest.main()
