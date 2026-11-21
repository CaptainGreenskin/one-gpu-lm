"""A partial 041 audit preserves every disposition and anonymizes source metadata."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from one_gpu_lm.common.receipts import sha256, write_json


SPEC = importlib.util.spec_from_file_location(
    'first_edit_report', Path(__file__).resolve().parents[1] / 'experiments/041-first-edit-continuation/report.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def cell(success, calls=3):
    return {'audited': True, 'first_edit_call': 1 if success else None,
            'edit_actions': int(success), 'correct_patch': success,
            'verified_and_submitted_within_budget': success, 'qualified': success,
            'qualification_reason': None if success else 'repair_or_submission_failed',
            'attempted_calls': calls, 'generated_tokens': 34, 'input_tokens': 124,
            'seconds': 10.0, 'timeout_recovered': False, 'invalid_actions': 0}


class FirstEditReport(unittest.TestCase):
    def test_capture_projects_and_counts_audited_pairs_only(self):
        with tempfile.TemporaryDirectory() as area:
            root = Path(area)
            out = root / 'run'
            out.mkdir()
            design_path = Path(__file__).resolve().parents[1] / 'experiments/041-first-edit-continuation/design.json'
            design = REPORT.read(design_path)
            design['task_count'] = 2
            write_json(root / 'design.json', design)
            plan = root / 'plan.json'
            write_json(plan, {'run_directory': 'run', 'design_path': 'design.json'})
            identity = sha256(plan)
            clock = 1_700_000_000
            controller = {'complete': False, 'stage': 'failed', 'backend_stopped': True,
                          'frozen_files_unchanged': True, 'protocol_sha256': identity, 'final_test_opened': False,
                          'optimizer_updates': 0, 'started_epoch': clock, 'finished_epoch': clock + 30,
                          'telemetry': {'samples': 1, 'errors': 0, 'thread_stopped': True,
                                        'private_file': 'PRIVATE_TELEMETRY_PATH'}}
            write_json(out / 'controller.json', controller)
            (out / 'telemetry.csv').write_text(
                'utc,attempt,temperature_c,power_w,utilization_percent,memory_used_mib\n'
                '2023-11-14T22:13:20+00:00,1,51,476,90,59000\n')
            base = cell(False, 2)
            low = {**cell(True), 'raw_prompt': 'PRIVATE_TRAJECTORY'}
            audit = {'protocol_sha256': identity, 'passed': False, 'collection_gate_passed': False,
                     'qualified_tasks': 1, 'qualified_families': 1, 'journal_requests_verified': 7,
                     'final_test_opened': False, 'new_optimizer_updates': 0, 'export': None,
                     'pairs': [{'pair': 1, 'family': 'PRIVATE_FAMILY', 'language': 'python',
                                'task_id': 'PRIVATE_TASK', 'arms': {'control': base, 'correction': low}},
                               {'pair': 2, 'family': 'PRIVATE_OTHER_FAMILY', 'language': 'javascript',
                                'arms': {'control': base, 'correction': {'audited': False, 'reason': 'audit_failed',
                                         'raw_error': 'PRIVATE_ERROR'}}}]}
            write_json(out / 'audit.json', audit)
            result = REPORT.capture(root, plan)
            self.assertNotIn('PRIVATE_', json.dumps(result))
            summary = REPORT.summarize(result['pairs'], design, result)
            self.assertEqual(summary['complete_audited_pairs'], 1)
            self.assertEqual(summary['audited_arms'], 3)
            self.assertEqual(summary['verified_requests'], 7)
            self.assertEqual(summary['outcomes']['direct']['attempted_calls'], 2)
            self.assertEqual(summary['paired_low_minus_direct']['qualified'], {'gain': 1, 'loss': 0, 'net': 1})
            tampered = deepcopy(result)
            tampered['audit_passed'] = True
            with self.assertRaises(ValueError):
                REPORT.summarize(tampered['pairs'], design, tampered)
