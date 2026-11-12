"""Recompute published 021 evidence and reject altered inputs; no model calls."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/021-observed-recovery'
RESULTS = EXP / 'results/2026-11-12'
spec = importlib.util.spec_from_file_location('observed_report', EXP / 'report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class PublishedReportContracts(unittest.TestCase):
    def test_complete_evidence_recomputes_rejection_without_final_scores(self):
        result = report.assess(RESULTS)
        self.assertEqual(result, json.loads((RESULTS / 'summary.json').read_text()))
        self.assertFalse(result['release_decision']['accepted'])
        self.assertFalse(result['release_decision']['final_test_opened'])
        self.assertEqual(result['comparisons'], {})
        self.assertEqual(len(result['pairs']), 30)
        self.assertEqual([result['evaluation_cost'][a]['passed'] for a in
                          ['dev-base', 'dev-reference020', 'dev-observed']], [6, 8, 6])

    def test_rehashed_changes_still_fail_independent_evidence_checks(self):
        changes = [
            ('training-observed/metrics.json',
             lambda v: v[0].update(example_sha256='0' * 64),
             'Actual optimizer examples differ'),
            ('campaigns/dev-observed.json',
             lambda v: v['results'][0].update(passed=not v['results'][0]['passed']),
             'A successful score requires a successful independent grader exit'),
            ('selection.json', lambda v: v.update(final_test_opened=True),
             'Unexpected final exposure'),
        ]
        for name, change, error in changes:
            with self.subTest(file=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / 'altered-evidence'
                shutil.copytree(RESULTS, root)
                target = root / name
                value = json.loads(target.read_text()); change(value)
                report.write_json(target, value)
                # Refresh the collection envelope so the deeper audit/selection
                # checks, rather than just the outer file hash, must reject it.
                collection = json.loads((root / 'collection.json').read_text())
                collection['files'][name] = report.sha256(target)
                report.write_json(root / 'collection.json', collection)
                with self.assertRaisesRegex(ValueError, error):
                    report.assess(root)


if __name__ == '__main__':
    unittest.main()
