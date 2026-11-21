import unittest

from one_gpu_lm.verified.decision_policy import REASONING_VERSION, notice
from one_gpu_lm.verified.decision_state_audit import earliest_comparable_fork


class DecisionStates(unittest.TestCase):
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
