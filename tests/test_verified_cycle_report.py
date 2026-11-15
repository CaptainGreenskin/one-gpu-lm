import json
from pathlib import Path
import tempfile
import unittest

from one_gpu_lm.verified.cycle_report import public_input, summarize, report, hardware
from one_gpu_lm.verified.factorial import ARMS


def fixture():
    tasks = [{'id': f't{i}', 'repository': f'r{i//2}', 'family': f'f{i//2}',
              'description': 'PRIVATE PROMPT'} for i in range(4)]
    spec = {'development': {'unique_tasks': 4, 'repository_families': 2, 'repetitions': 3, 'total_task_runs': 48},
            'analysis': {'minimum_development_parameter_gain_pp': 10,
                         'maximum_allowed_primary_drop_in_any_repository_pp': 10}}
    rows = []
    for task in tasks:
        for rep in range(3):
            for arm in ARMS:
                success = arm.startswith('student_')
                rows.append({'id': task['id'], 'repository': task['repository'], 'family': task['family'],
                    'repetition': rep, 'arm': arm, 'complete': True, 'audit_passed': True,
                    'success_within_budget': success, 'grader_passed': success,
                    'first_edit_call': 1 if success else None, 'grade_reason': 'exit' if success else 'no_source_change',
                    'attempted_calls': 3, 'generated_tokens': 8, 'input_tokens': 12, 'seconds': 10,
                    'submitted': success, 'messages': 'PRIVATE GENERATION',
                    'request_replay': {'action_fingerprint': str(rep if arm == 'base_current' else 0),
                                       'delivered_feedback_fingerprints': ['constant'], 'command': 'PRIVATE COMMAND'}})
    phase = {'started_epoch': 0, 'finished_epoch': 3600, 'complete': True, 'optimizer_steps': 1,
             'host': 'PRIVATE HOST', 'telemetry': {'samples': 0, 'errors': 2},
             'backend_lifecycles': [{'started_epoch': 0, 'ready_epoch': 20, 'pid': 999},
                                    {'started_epoch': 3000, 'pid': 1000}]}
    audit = {'passed': True, 'checks': {'frozen_inputs_unchanged': True}, 'rows': rows}
    curve = [{'record': 1, 'update': 1, 'tokens': 10, 'supervised_tokens': 2, 'loss': 1.2,
              'seconds': 1, 'id': 'PRIVATE EXAMPLE'}]
    return public_input(spec, tasks, audit, phase, phase, phase, curve,
                        [{'step': 1, 'records': 1, 'grad_norm': 1.5, 'learning_rate': 1e-5, 'seconds': 2}], {})


class CycleReportContracts(unittest.TestCase):
    def test_public_projection_drops_private_content_and_process_identifiers(self):
        value = fixture(); text = json.dumps(value)
        self.assertNotIn('PRIVATE', text); self.assertNotIn('pid', text)
        self.assertEqual(value['candidate_curve'][0]['loss'], 1.2)

    def test_three_repeats_and_system_effect_remain_separate_from_parameter_gain(self):
        value = fixture()
        for row in value['evaluation_audit']['rows']:
            row['success_within_budget'] = row['grader_passed'] = row['arm'].endswith('_guard')
        result = summarize(value)
        self.assertEqual(result['effects']['parameter'], 0)
        self.assertEqual(result['effects']['system'], 1)
        self.assertEqual(result['independent_task_units'], 4)
        self.assertEqual(result['outcome_curve'][-1]['paired_tasks'], 4)
        self.assertEqual(result['conclusion'], 'not_achieved')
        self.assertFalse(result['weight_publication_allowed'])

    def test_unverified_positive_is_excluded_but_keeps_all_cost(self):
        value = fixture(); row = value['evaluation_audit']['rows'][0]
        row.pop('audit_passed'); row['success_within_budget'] = True
        result = summarize(value)
        self.assertEqual(result['paired_tasks'], 3)
        self.assertEqual(result['cost']['generated_tokens'], 384)
        self.assertEqual(result['arms']['base_current']['categories']['unverified'], 1)
        self.assertEqual(result['conclusion'], 'not_achieved')

    def test_incomplete_loads_keep_pipeline_time_and_no_hardware_is_not_zero_usage(self):
        value = fixture(); value['candidate'].pop('backend_lifecycles')
        result = summarize(value)
        cost = result['phases']['evaluation']
        self.assertEqual(cost['reserved_single_gpu_hours'], 1)
        self.assertEqual(cost['completed_backend_load_seconds'], 20)
        self.assertEqual(cost['incomplete_backend_loads'], 1)
        self.assertIsNone(result['hardware']['evaluation']['utilization_percent']['mean'])
        self.assertEqual(result['hardware']['evaluation']['collection_errors'], 2)
        self.assertIsNone(result['phases']['candidate']['backend_loads'])
        self.assertIsNone(result['phases']['candidate']['completed_backend_load_seconds'])

    def test_action_instability_is_visible_with_stable_failure_scores(self):
        result = summarize(fixture())
        observed = result['instability']['base_current']
        self.assertEqual(observed['outcome_disagreement_tasks'], 0)
        self.assertEqual(observed['action_disagreement_tasks'], 4)
        self.assertEqual(observed['tasks'][0]['action_disagreeing_repeat_pairs'], 3)
        self.assertEqual(result['arms']['base_current']['categories'], {'no_final_source_edit': 12})

    def test_zero_pairs_still_writes_costs_and_failure_conclusion(self):
        value = fixture()
        for row in value['evaluation_audit']['rows']:
            row.update(complete=False, unknown_inflight_generation_cost=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'analysis-input.json').write_text(json.dumps(value))
            result = report(root, render=False)
            self.assertEqual(result['conclusion'], 'not_achieved')
            self.assertEqual(result['paired_tasks'], 0)
            self.assertEqual(result['cost']['generated_tokens'], 384)
            self.assertIn('未达成', (root/'README.md').read_text())
            self.assertTrue((root/'pairs.csv').is_file())

    def test_hardware_reports_boundary_coverage(self):
        sample = {'utc': '1970-01-01T00:00:02+00:00', 'temperature_c': 50, 'power_w': 300,
                  'utilization_percent': 20, 'memory_used_mib': 60000}
        second = {**sample, 'utc': '1970-01-01T00:00:08+00:00'}
        result = hardware([sample, second], {'started_epoch': 0, 'finished_epoch': 10})
        self.assertEqual(result['boundary_uncovered_seconds'], 4)
        self.assertEqual(result['maximum_sample_gap_seconds'], 6)
        with self.assertRaises(ValueError): hardware([second, sample], {})


if __name__ == '__main__':
    unittest.main()
