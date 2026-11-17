"""Exercise the new policy through real fixture execution, replay and export."""
import copy
from tests import test_failure_continuation as continuation_tests
from one_gpu_lm.verified import decision_policy as policy
from one_gpu_lm.verified import failure_continuation as fc
from one_gpu_lm.verified.submission import SUBMIT


class RepairDecisions(continuation_tests.FailureContinuation):
    def setUp(self):
        super().setUp()
        self.f.protocol['continuation_policy'] = policy.VERSION

    def test_control_is_exactly_previous_correction_initial_context(self):
        old = copy.deepcopy(self.f.protocol);old.pop('continuation_policy')
        self.assertEqual(fc.initial_messages(self.prefix, old, 'correction'),
                         fc.initial_messages(self.prefix, self.f.protocol, 'control'))
        treatment = fc.initial_messages(self.prefix, self.f.protocol, 'correction')
        self.assertIn(policy.GUIDANCE, treatment[-1]['content'])

    def test_both_arms_reject_non_regression_after_edit_without_free_calls(self):
        for arm in ['control', 'correction']:
            out,t,_ = self.collect(arm, ['edit-source','cat logic.py','python3 check.py',SUBMIT])
            self.assertFalse(t['steps'][1]['observation']['executed'])
            self.assertEqual(t['attempted_calls'],4)
            self.assertTrue(self.replay(out,t)['passed'])

    def test_status_identifies_unchanged_source_repeat_and_fresh_verification(self):
        out,t,_ = self.collect('correction',['cat logic.py','cat logic.py','edit-source','python3 check.py',SUBMIT])
        notices=[s['policy_notice'] for s in t['steps']]
        self.assertIn('11 new calls remain',notices[0])
        self.assertIn('No allowed source bytes changed',notices[0])
        self.assertIn('already requested at calls 1',notices[1])
        self.assertIn('source bytes changed',notices[2])
        self.assertIn('current source has a passing standalone',notices[3])
        self.assertTrue(self.replay(out,t)['passed'])
        t['steps'][0]['policy_notice']='invented feedback'
        self.assertFalse(self.replay(out,t)['passed'])

    def test_unknown_version_rejected(self):
        self.f.protocol['continuation_policy']='unregistered'
        with self.assertRaises(ValueError):
            fc.initial_messages(self.prefix,self.f.protocol,'control')
