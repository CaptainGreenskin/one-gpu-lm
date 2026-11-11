import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('curriculum_diagnosis', Path(__file__).resolve().parents[1]/'experiments/020-repair-curriculum/diagnose.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class ObservedFailureTests(unittest.TestCase):
    def test_submission_and_valid_format_do_not_count_as_a_repair(self):
        row = {'passed': False, 'exit_status': 'Submitted',
            'source_progress': [{'changed_editable_files': 0}],
            'generations': [{'action_format_valid': True, 'progress': {'reminder': False}}]}
        result = module.observed_failure(row)
        self.assertEqual(result['failure_category'], 'without_observed_edit')
        self.assertTrue(result['submitted_but_failed_without_edit'])
        self.assertEqual(result['format_invalid_calls'], 0)

    def test_reverted_edit_is_not_misclassified_as_never_edited(self):
        row = {'passed': False, 'exit_status': 'LimitsExceeded',
            'source_progress': [{'changed_editable_files': 1}, {'changed_editable_files': 0}],
            'generations': [{'action_format_valid': False, 'progress': {'reminder': True}}]}
        result = module.observed_failure(row)
        self.assertEqual(result['failure_category'], 'after_edit')
        self.assertFalse(result['submitted_but_failed_without_edit'])
        self.assertEqual(result['format_invalid_calls'], 1)


if __name__ == '__main__':
    unittest.main()
