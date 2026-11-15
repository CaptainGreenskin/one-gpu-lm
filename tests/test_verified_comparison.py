import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'experiments/025-verified-distillation/compare.py'
spec = importlib.util.spec_from_file_location('verified_comparison', path)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ComparisonBoundaries(unittest.TestCase):
    @staticmethod
    def trace():
        return {'steps': [{'call': n, 'generation': {'input_tokens_sha256': str(n), 'ids': [n]},
                           'observation': {'executed': True}} for n in range(1, 5)]}

    def test_generation_of_rejected_call_is_still_before_policy_feedback(self):
        a, b = self.trace(), self.trace()
        b['steps'][1]['observation']['executed'] = False
        b['steps'][1]['generation']['ids'] = [999]
        row = compare.shared_prefix(a, b)
        self.assertEqual(row['difference_before_policy_feedback']['call'], 2)
        self.assertTrue(row['difference_before_policy_feedback']['input_equal'])

    def test_next_generation_may_change_after_recorded_rejection(self):
        a, b = self.trace(), self.trace()
        b['steps'][1]['observation']['executed'] = False
        b['steps'][2]['generation']['ids'] = [999]
        row = compare.shared_prefix(a, b)
        self.assertIsNone(row['difference_before_policy_feedback'])
        self.assertEqual(row['identical_shared_calls_before_policy_feedback'], 2)

    def test_ordinary_failures_remain_paired_but_missing_rows_are_explicit(self):
        row = {'id': 'a', 'repository': 'r', 'complete': True, 'original_source_sha256': {'f': 'same'},
               'grade': {'passed': False}, 'submitted': False, 'attempted_calls': 24,
               'generated_tokens': 100, 'seconds': 10}
        audit = {'rows': [{'id': 'a', 'input_replay_passed': True, 'eligible_for_sft': False},
                           {'id': 'missing', 'input_replay_passed': False, 'eligible_for_sft': False}]}
        a = {'tasks': [row]}; b = copy.deepcopy(a)
        pairs, excluded = compare.pair_receipts(a, b, audit, audit)
        self.assertEqual([p['id'] for p in pairs], ['a'])
        self.assertEqual(excluded, [{'id': 'missing', 'reason': 'missing_or_incomplete_task'}])
        b['tasks'][0]['original_source_sha256']['f'] = 'different'
        with self.assertRaises(ValueError):
            compare.pair_receipts(a, b, audit, audit)

    def test_partial_generation_cost_is_preserved_outside_paired_outcomes(self):
        audit = {'rows': [{'id': 'partial', 'input_replay_passed': False, 'attempted_calls': 3,
                           'generated_tokens': 17}, {'id': 'not_started', 'reason': 'not_run',
                           'input_replay_passed': False}]}
        cost = compare.attempt_cost({'started_epoch': 10, 'finished_epoch': 30}, audit)
        self.assertEqual(cost['all_observed_attempted_calls'], 3)
        self.assertEqual(cost['all_observed_generated_tokens'], 17)
        self.assertEqual(cost['unreplayed_attempts_with_uncertain_cost'], ['partial'])
        self.assertEqual(cost['not_run_tasks'], ['not_started'])


if __name__ == '__main__':
    unittest.main()
