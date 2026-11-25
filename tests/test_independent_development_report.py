"""Keep the bootstrap and negative decision intact after removing repository names."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/039-release-development'
ANALYZE = runpy.run_path(str(EXP/'analysis.py'))['analyze']
REPORT = runpy.run_path(str(ROOT/'experiments/042-first-edit-sft/development_report.py'))
PROJECT = REPORT['project_rows']
with patch.object(sys, 'path', [str(EXP), *sys.path]):
    REPAIR_SCOPE = runpy.run_path(str(EXP/'report.py'))['repair_scope']


class IndependentDevelopmentReport(unittest.TestCase):
    def test_repair_scope_uses_the_same_label_for_both_arms(self):
        rows = [
            {'task_number': 1, 'arm': 'base', 'multifile': False, 'completed_repair': False},
            {'task_number': 1, 'arm': 'candidate', 'multifile': False, 'completed_repair': True},
            {'task_number': 2, 'arm': 'base', 'multifile': True, 'completed_repair': True},
            {'task_number': 2, 'arm': 'candidate', 'multifile': True, 'completed_repair': False},
        ]
        self.assertEqual(REPAIR_SCOPE(rows), {
            'single_file': {'tasks': 1, 'paired_repeats': 1, 'successes': {'base': 0, 'candidate': 1}},
            'multi_file': {'tasks': 1, 'paired_repeats': 1, 'successes': {'base': 1, 'candidate': 0}},
        })
        rows[-1]['multifile'] = False
        with self.assertRaisesRegex(ValueError, 'same boolean multi-file label'):
            REPAIR_SCOPE(rows)

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

    def test_hardware_keeps_reboot_gap_and_unavailable_segment_out_of_averages(self):
        header = 'utc,attempt,temperature_c,power_w,utilization_percent,memory_used_mib\n'
        with TemporaryDirectory() as directory, patch.object(sys,'path',[str(EXP),*sys.path]):
            out = Path(directory)
            first = out/'telemetry.csv'
            first.write_text(header + '1970-01-01T00:00:01+00:00,1,50,150,20,1000\n'
                                     '1970-01-01T00:00:06+00:00,1,52,160,40,1100\n')
            second = out/'private/attempts/002/telemetry.csv'
            second.parent.mkdir(parents=True)
            second.write_text(header + '1970-01-01T00:10:01+00:00,2,51,180,90,1200\n'
                                       '1970-01-01T00:10:06+00:00,2,53,200,100,1300\n')
            state = {'started_epoch':0,'finished_epoch':610,'telemetry':{'samples':2,'errors':0},
                     'restarts':[{'boot_epoch':590,'resumed_epoch':600,
                                  'prior_telemetry_sha256':sha256(first.read_bytes()).hexdigest()}]}
            gpu = REPORT['gpu_statistics'](out,state)
            self.assertEqual(gpu['samples'],4)
            self.assertEqual(gpu['sample_span_seconds'],10)
            self.assertEqual(gpu['utilization_percent'],{'mean':62.5,'maximum':100.0})
            self.assertIsNone(gpu['segments'][0]['collector_errors'])

            first.unlink()
            gpu = REPORT['gpu_statistics'](out,state)
            self.assertEqual(gpu['unavailable_attempts'],[1])
            self.assertEqual(gpu['samples'],2)
            self.assertEqual(gpu['utilization_percent']['mean'],95)


if __name__=='__main__':
    unittest.main()
