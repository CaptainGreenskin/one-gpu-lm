"""Recompute actual 022 results and reject tampered schedules, scores and gates."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/022-action-repair'
RESULTS=EXP/'results/2026-11-12'
spec=importlib.util.spec_from_file_location('action_report',EXP/'report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

class PublishedActionReport(unittest.TestCase):
    def test_actual_evidence_recomputes_exact_decision(self):
        result=report.assess(RESULTS);expected=json.loads((RESULTS/'summary.json').read_text())
        key='sampled_gpu_energy_kwh'
        self.assertAlmostEqual(result['hardware'].pop(key),expected['hardware'].pop(key),delta=1e-12)
        self.assertEqual(result,expected)
        self.assertEqual(result['training']['steps'],330)
        self.assertEqual(len(result['diagnosed_development_tasks']),6)
        self.assertEqual(result['data']['replaced_records'],55)
        self.assertEqual(result['data']['shared_records'],275)
        self.assertEqual([result['evaluation_cost'][arm]['passed'] for arm in
                          ['dev-base','dev-reference020','dev-action']],[7,6,7])
        self.assertEqual(len(result['pairs']),30)
        self.assertIsNone(result['selection']['selected'])
        self.assertFalse(result['selection']['final_test_opened'])
        self.assertFalse(result['release_decision']['accepted'])
        self.assertFalse(result['weights_public'])

    def test_rehashed_tampering_fails_deeper_independent_checks(self):
        changes=[('training-action/metrics.json',lambda v:v[0].update(example_sha256='0'*64)),
                 ('campaigns/dev-action.json',lambda v:v['results'][0].update(passed=not v['results'][0]['passed'])),
                 ('selection.json',lambda v:v.update(final_test_opened=not v['final_test_opened'])),
                 ('training-data-audit.json',lambda v:v.update(common_identical_records=274)),
                 ('cpu-preparation-timing.json',lambda v:v.update(builder_seconds=1)),
                 ('rerun-variation-audit.json',lambda v:v.update(causal_effect_tested=True))]
        for name,change in changes:
            with self.subTest(file=name),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'evidence';shutil.copytree(RESULTS,root)
                value=json.loads((root/name).read_text());change(value);report.write_json(root/name,value)
                collected=json.loads((root/'collection.json').read_text());collected['files'][name]=report.sha256(root/name)
                report.write_json(root/'collection.json',collected)
                with self.assertRaises(ValueError):report.assess(root)

if __name__=='__main__':unittest.main()
