import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


EXPERIMENT = Path(__file__).resolve().parents[1] / 'experiments/045-first-action-forks'


def load(filename):
    spec = importlib.util.spec_from_file_location(filename, EXPERIMENT / f'{filename}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules, {'prepare': load('prepare')}):
    forks = load('run')


class FirstActionForksTest(unittest.TestCase):
    def setUp(self):
        ids = [2, 7, 5]
        self.ids = ids
        self.prompt_hash = forks.identity_digest(ids)
        source = {'implementation.py': 'a' * 64}
        self.case = SimpleNamespace(prefix={'decision_source_sha256': source},
                                    proof={'prefix_sha256':'b' * 64},
                                    protocol={'agent_policy':{}},
                                    budget={'max_model_calls':11,'max_total_generated_tokens':6000,
                                            'max_generated_tokens_per_call':512},
                                    task={'id':'fixture','family':'fixture'})
        self.arms = {}
        for arm in ('greedy','sampled'):
            step = {'generation':{'input_tokens_sha256':self.prompt_hash,'text':arm},
                    'request_sha256':'c' * 64,'source_before':source,'command':f'echo {arm}',
                    'action_format_valid':True,'observation':{'executed':True},
                    'protected_before':True,'protected_after':True}
            self.arms[arm] = {'steps':[step],'terminal_reason':'submitted' if arm=='sampled' else 'call_limit'}
        self.scores = {arm:{'verified_and_submitted':arm=='sampled'} for arm in self.arms}
        self.dirs = {arm:Path(arm) for arm in self.arms}
        self.profile = {'training':{'max_sequence_length':16}}

    def encode(self, example, *_):
        suffix = 8 if example['target']=='sampled' else 9
        return {'ids':self.ids+[suffix], 'prefix_tokens':len(self.ids), 'supervised_tokens':1}

    def compare(self):
        with patch.object(forks, 'packed_history', return_value=[]), patch.object(forks, 'encode_training_example', self.encode):
            return forks.compare_first(self.case,self.arms,self.scores,self.dirs,None,self.profile,self.profile)

    def test_accepts_executed_distinct_actions_with_a_verified_outcome_difference(self):
        record, reason = self.compare()
        self.assertEqual(reason,'matched_executed_contrast')
        self.assertEqual((record['chosen'],record['rejected']),('sampled','greedy'))
        self.assertEqual(record['provenance']['fork_call'],1)

    def test_refuses_a_pair_if_the_shared_snapshot_or_actor_input_differs(self):
        self.arms['greedy']['steps'][0]['generation']['input_tokens_sha256'] = 'e' * 64
        with self.assertRaisesRegex(ValueError,'identical inputs'):
            self.compare()

    def test_rejects_unexecuted_first_action_and_invalid_student_prefix(self):
        first = self.arms['greedy']['steps'][0]
        first['observation']['executed'] = False
        self.assertEqual(self.compare(),(None,'first_action_not_executed'))
        first['observation']['executed'] = True
        self.ids = [19, 20, 21]
        self.assertEqual(self.compare(),(None,'token_budget_or_shared_prefix'))


if __name__ == '__main__':
    unittest.main()
