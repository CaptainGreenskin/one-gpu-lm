import copy
import unittest

from one_gpu_lm.verified.first_action_loss import plan, target_receipt
from one_gpu_lm.verified.historic_plan import native


def record(record_id, kind, call=1):
    return {'id': record_id, 'episode_id': 'episode-1',
            'messages': [{'role': 'system', 'content': 'Use one action.'},
                         {'role': 'user', 'content': 'Regression command: python3 -m unittest'}],
            'target': native('python3 -m unittest'),
            'provenance': {'split': 'train', 'executed': True, 'call': call,
                           'producer_kind': 'upstream_reference_guided', 'kind': kind, 'model_generated': False},
            'encoding_receipt': {'supervised_tokens': 3}}


class FirstActionLossTest(unittest.TestCase):
    def setUp(self):
        self.rows = [record('opening', 'baseline_regression'), record('verification', 'final_verification', 5)]
        self.selected = [{'id': r['id'], 'episode_id': r['episode_id'], 'ids': [0, 1, 2, 3],
                          'labels': [-100, 1, 2, 3], 'supervised_tokens': 3} for r in self.rows]
        _, self.receipt = target_receipt(self.rows, self.selected)

    def test_only_opening_objective_is_masked_and_encoding_stays_same(self):
        before = copy.deepcopy(self.selected)
        baseline, a = plan(self.rows, self.selected, {'target_receipt': self.receipt, 'target_loss_weight': 1})
        masked, b = plan(self.rows, self.selected, {'target_receipt': self.receipt, 'target_loss_weight': 0})
        self.assertEqual(baseline, set())
        self.assertEqual(masked, {'opening'})
        self.assertEqual(a['encoding_sha256'], b['encoding_sha256'])
        self.assertEqual((a['effective_supervised_tokens'], b['effective_supervised_tokens']), (6, 3))
        self.assertEqual(self.selected, before)

    def test_broken_order_and_wrong_exact_target_fail_closed(self):
        spec = {'target_receipt': self.receipt, 'target_loss_weight': 0}
        with self.assertRaises(ValueError):
            plan(self.rows, self.selected[::-1], spec)
        self.rows[0]['target'] = native('echo incidental')
        with self.assertRaises(ValueError):
            plan(self.rows, self.selected, spec)

    def test_held_out_or_duplicate_records_fail_closed(self):
        with self.assertRaises(ValueError):
            target_receipt([self.rows[0]] * 2, self.selected)
        self.rows[0]['provenance']['split'] = 'development'
        with self.assertRaises(ValueError):
            target_receipt(self.rows, self.selected)
