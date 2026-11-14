import copy
import unittest
from one_gpu_lm.verified.audit import decision


class EpisodeGate(unittest.TestCase):
    def setUp(self):
        self.protocol = {'task_ids': list(range(12)), 'feasibility_gate': {
            'minimum_eligible_full_episodes': 6, 'minimum_per_repository': 1,
            'repositories': ['a', 'b', 'c'], 'languages': ['python', 'javascript']}}
        self.rows = [{'repository': 'abc'[i // 4], 'language': 'python' if i < 8 else 'javascript',
                      'eligible_for_sft': i % 2 == 0, 'input_replay_passed': True} for i in range(12)]
        self.controller = {'complete': True, 'frozen_files_unchanged': True}

    def test_full_episodes_not_action_rows(self):
        result = decision(self.protocol, self.rows, self.controller)
        self.assertTrue(result['passed']); self.assertEqual(result['eligible_full_episodes'], 6)
        self.assertFalse(result['model_quality_improvement_established'])
        self.assertFalse(result['weight_publication_allowed'])
        rows = copy.deepcopy(self.rows)
        for r in rows: r['encoded_actions'] = [{}] * 100
        rows[0]['eligible_for_sft'] = False
        self.assertFalse(decision(self.protocol, rows, self.controller)['passed'])

    def test_failure_cannot_be_hidden_by_successful_subset(self):
        rows = copy.deepcopy(self.rows); rows[1]['input_replay_passed'] = False
        self.assertFalse(decision(self.protocol, rows, self.controller)['passed'])
        self.controller['complete'] = False
        self.assertFalse(decision(self.protocol, self.rows, self.controller)['passed'])

    def test_repository_and_language_coverage(self):
        for row in self.rows:
            row['eligible_for_sft'] = row['repository'] != 'c'
        self.assertFalse(decision(self.protocol, self.rows, self.controller)['passed'])


if __name__ == '__main__': unittest.main()
