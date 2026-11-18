import copy
import json
from pathlib import Path
import unittest
from one_gpu_lm.verified.historic_incident_report import corrected_summary


class IncidentReportTests(unittest.TestCase):
    def test_incomplete_coverage_unknown_cost_and_absent_repeats(self):
        root = Path(__file__).resolve().parents[1] / 'experiments/032-multifile-completion/results/2026-11-18-development'
        bundle = json.loads((root / 'analysis-input.json').read_text())
        incident = json.loads((root / 'incident.json').read_text())
        original = copy.deepcopy(bundle)
        result = corrected_summary(bundle, incident)
        self.assertEqual((result['completed_runs'], result['incomplete_runs'], result['not_started_runs']), (92, 1, 231))
        self.assertEqual(result['missing_runs'], 232)
        self.assertEqual(result['audited_completed_runs'], 92)
        self.assertEqual(result['cost']['generated_tokens_bounds'], [56645, 57157])
        self.assertEqual(result['cost']['attempted_calls'], 1033)
        self.assertEqual(result['paired_tasks'], 0)
        self.assertFalse(result['development_parameter_gate_passed'])
        self.assertFalse(result['weight_publication_allowed'])
        self.assertEqual(bundle, original)
        for value in result['variants'].values():
            self.assertIsNone(value['repeat_action_disagreement_tasks'])
            self.assertIsNone(value['repeat_outcome_disagreement_tasks'])
        self.assertEqual(result['variants']['multifile032']['recorded_runs'], 33)


if __name__ == '__main__':
    unittest.main()
