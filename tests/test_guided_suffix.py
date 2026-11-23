from copy import deepcopy
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = runpy.run_path(str(ROOT / 'experiments/049-guided-repair-suffix/run.py'))


class GuidedSuffixTests(TestCase):
    def test_schedule_pairs_and_alternates_first_arm(self):
        schedule = list(TOOLS['schedule'](13))
        self.assertEqual(len({row[2] for row in schedule}), 26)
        for index in range(13):
            pair = schedule[2 * index:2 * index + 2]
            self.assertTrue(all(row[0] == index + 1 for row in pair))
            self.assertEqual([row[1] for row in pair],
                             ['control', 'guided'] if index % 2 == 0 else ['guided', 'control'])

    def test_supervision_conserves_context_and_rejects_partial_episodes(self):
        guide = 'Apply a supported edit and verify it.'
        case = SimpleNamespace(messages=[{'role': 'system', 'content': 'original'},
                                         {'role': 'user', 'content': 'task'}],
                               task={'id': 't', 'family': 'f', 'regression_command': 'test'},
                               protocol={'agent_policy': {}}, budget={})
        guided = TOOLS['with_guide'](case.messages, guide)
        original, final = {'f.py': 'before'}, {'f.py': 'after'}
        trace = {'original_source_sha256': original, 'final_source_sha256': final, 'steps': []}
        with TemporaryDirectory() as root:
            path = Path(root)
            history = deepcopy(guided)
            for call, command in enumerate(('edit', 'test', TOOLS['SUBMIT']), 1):
                step = {'call': call, 'command': command, 'action_format_valid': True,
                        'source_before': original if call == 1 else final, 'source_after': final,
                        'protected_before': True, 'protected_after': True,
                        'observation': {'returncode': 0, 'executed': True},
                        'generation': {'text': command}}
                trace['steps'].append(step)
                (path / f'request-{call:03d}.json').write_text(json.dumps({'messages': history}))
                history.extend([{'role': 'assistant', 'content': command},
                                {'role': 'user', 'content': 'verified tool response'}])
            (path / 'trajectory.json').write_text(json.dumps(trace))

            def encode(messages, target):
                self.assertEqual(messages[:2], case.messages)
                self.assertNotIn(guide, json.dumps(messages))
                return messages, {'target_tokens': 1}

            supervision = TOOLS['supervision']
            with patch.dict(supervision.__globals__, ReplayRuntime=lambda *args: SimpleNamespace(target=encode)):
                rows, status = supervision(case, guide, path, trace, {}, {}, None)
                self.assertEqual(status, 'qualified')
                self.assertEqual([row['kind'] for row in rows], ['edit', 'regression', 'submit'])
                self.assertIn('verified tool response', rows[-1]['messages'][-1]['content'])
                trace['steps'][1]['observation']['returncode'] = 1
                self.assertEqual(supervision(case, guide, path, trace, {}, {}, None),
                                 ([], 'no_verified_submission'))
