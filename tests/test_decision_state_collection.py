from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.verified.decision_policy import REASONING_VERSION, notice
from one_gpu_lm.verified.decision_state_audit import earliest_comparable_fork
from one_gpu_lm.verified.decision_training_data import admitted_sources


class DecisionStates(unittest.TestCase):
    def test_training_input_must_match_winning_fork_and_exclude_development_families(self):
        repo = 'example/library'
        kind = 'verified_finish'
        with TemporaryDirectory() as directory:
            data = Path(directory)
            snapshot = data/'snapshots/example--library'
            snapshot.mkdir(parents=True)
            notice = snapshot/'LICENSE'
            notice.write_text('Testing notice')
            source = {'repository': repo, 'family': repo, 'split': 'training', 'revision': 'a'*40,
                      'license_spdx': 'MIT', 'notice_review_complete': True, 'lineage_review_complete': True,
                      'approved_for': {'training': True, 'weight_publication': False},
                      'notices': ['LICENSE'], 'files': {'LICENSE': sha256(notice)}}
            case = SimpleNamespace(source=source, kind=kind, task={'id': 't1', 'family': repo},
                                   proof={'prefix_sha256': 'a'*64},
                                   budget={'max_model_calls': 2, 'max_total_generated_tokens': 100})
            audit = {'pairs': [{'pair': 1, 'kind': kind, 'family': repo, 'contrast': 'matched_executed_contrast',
                                'arms': {'control': {'audited': True, 'verified_and_submitted': True, 'attempted_calls': 1},
                                         'correction': {'audited': True, 'verified_and_submitted': False, 'attempted_calls': 2}}}]}
            row = {'kind': kind, 'provenance': {'task_id': 't1', 'split': 'train', 'family': repo,
                                                'prefix_sha256': 'a'*64,
                                                'chosen_arm': 'control', 'rejected_arm': 'correction', 'fork_call': 1,
                                                'remaining_calls': 2, 'remaining_generated_tokens': 90}}
            review = admitted_sources(data, {'t1': case}, audit, [row], set())
            self.assertEqual(review[repo]['notices'], {'LICENSE': sha256(notice)})
            self.assertFalse(review[repo]['recorded_weight_publication_approval'])
            with self.assertRaisesRegex(ValueError, 'approved training scope'):
                admitted_sources(data, {'t1': case}, audit, [row], {repo})
            row['provenance'].update(chosen_arm='correction', rejected_arm='control')
            with self.assertRaisesRegex(ValueError, 'approved training scope'):
                admitted_sources(data, {'t1': case}, audit, [row], set())

    def test_restored_verification_survives_an_unchanged_probe(self):
        source = {'edit.py': 'verified'}
        protocol = {'continuation_policy': REASONING_VERSION, 'evaluation_budget': {'max_model_calls': 2}}
        step = {'call': 1, 'command': 'pwd', 'source_before': source, 'source_after': source,
                'observation': {'executed': True, 'returncode': 0}}
        task = {'regression_command': 'python check.py'}
        feedback = notice(protocol, 'control', task, [], step, initial_verified_source=source)
        self.assertIn('has a passing standalone declared regression', feedback)
        edit = dict(step, source_after={'edit.py': 'changed'})
        feedback = notice(protocol, 'correction', task, [], edit, initial_verified_source=source)
        self.assertIn('does not have a passing standalone declared regression', feedback)

    def test_fork_checks_remaining_tokens_and_rendered_history(self):
        source = {'edit.py': 'verified'}

        def step(call, command, tokens=10):
            return {'call': call, 'command': command, 'source_before': source, 'protected_before': True,
                    'protected_after': True, 'action_format_valid': True, 'observation': {'executed': True},
                    'generation': {'generated_tokens': tokens}}

        def history(seconds):
            return [{'role': 'system', 'content': 'Repair'},
                    {'role': 'user', 'content': 'Observed source', 'extra': {'timing': seconds}}]

        trace_a = {'steps': [step(1, 'pwd'), step(2, 'submit')]}
        trace_b = {'steps': [step(1, 'pwd'), step(2, 'ls')]}
        callbacks = [lambda _: history(1), lambda _: history(2)]
        fork = earliest_comparable_fork([trace_a, trace_b], callbacks, {'max_total_generated_tokens': 30})
        self.assertEqual(fork[0][0]['call'], 2)
        self.assertEqual(fork[2], 20)
        trace_b['steps'][0]['generation']['generated_tokens'] = 11
        self.assertIsNone(earliest_comparable_fork([trace_a, trace_b], callbacks,
                                                   {'max_total_generated_tokens': 30}))
        trace_b['steps'][0]['generation']['generated_tokens'] = 10
        different = lambda _: [{'role': 'user', 'content': 'Different source'}]
        self.assertIsNone(earliest_comparable_fork([trace_a, trace_b], [callbacks[0], different],
                                                   {'max_total_generated_tokens': 30}))
