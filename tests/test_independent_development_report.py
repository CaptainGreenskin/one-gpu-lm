"""Keep the bootstrap and negative decision intact after removing repository names."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/039-release-development'
ANALYZE = runpy.run_path(str(EXP/'analysis.py'))['analyze']
PROJECT = runpy.run_path(str(ROOT/'experiments/042-first-edit-sft/development_report.py'))['project_rows']


class IndependentDevelopmentReport(unittest.TestCase):
    def test_real_039_decision_is_exact_after_anonymization(self):
        measured = json.loads((EXP/'results/2026-11-20/measurements.json').read_bytes())
        design = json.loads((EXP/'design.json').read_bytes())
        rows, aliases = PROJECT(measured['rows'],design['families'],design['evaluation_budget'])
        expected = deepcopy(measured['analysis'])
        for row in expected['pairs']:
            row['family'] = aliases[row['family']]
        expected['family_gain_pp'] = {aliases[name]:gain for name,gain in expected['family_gain_pp'].items()}
        anonymous = dict(design,families={alias:design['families'][name] for name,alias in aliases.items()})
        result = ANALYZE(rows,anonymous)
        for arm in ('base','candidate'):
            self.assertAlmostEqual(result['costs'][arm]['actor_seconds'],expected['costs'][arm]['actor_seconds'],places=8)
            result['costs'][arm]['actor_seconds'] = expected['costs'][arm]['actor_seconds']
        self.assertEqual(result,expected)
        self.assertEqual(len(rows),360)
        self.assertFalse(result['development_gate_passed'])
        self.assertTrue(all(name not in json.dumps(rows) for name in aliases))


if __name__=='__main__':
    unittest.main()
