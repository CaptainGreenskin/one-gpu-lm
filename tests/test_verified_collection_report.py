import unittest

from one_gpu_lm.verified.collection_report import summarize
from one_gpu_lm.verified.guard_audit import decision


def fixture():
    protocol = {'task_ids': ['a', 'b'], 'max_pipeline_seconds': 100, 'supervisor_grace_seconds': 30,
        'feasibility_gate': {'minimum_eligible_full_episodes': 2, 'minimum_per_repository': 1,
                            'repositories': ['one', 'two'], 'languages': ['python', 'javascript']}}
    tasks = [{'id': name, 'repository': repo, 'language': language, 'complete': True,
              'input_replay_passed': True, 'eligible_for_sft': True,
              'encoded_actions': [{'call': 1, 'supervised_tokens': 2}, {'call': 2, 'supervised_tokens': 3}],
              'excluded_unexecuted_action_calls': [3]}
             for name, repo, language in [('a', 'one', 'python'), ('b', 'two', 'javascript')]]
    controller = {'complete': True, 'frozen_files_unchanged': True, 'started_epoch': 0, 'finished_epoch': 90,
        'deadline_epoch': 100, 'protocol_sha256': 'abc', 'backend_stopped': True, 'training_updates': 0,
        'development_tasks_scored': 0, 'final_test_opened': False, 'tasks': [
            {'id': name, 'complete': True, 'attempted_calls': 4, 'generated_tokens': 20,
             'grade': {'passed': True}, 'submitted': True, 'terminal_reason': 'submitted'} for name in ['a', 'b']]}
    audit = {'protocol_sha256': 'abc', 'rows': tasks, 'decision': decision(protocol, tasks, controller)}
    costs = [{'id': name, 'trace_present': True, 'complete': True, 'attempted_calls': 4,
              'completed_generation_receipts': 4, 'generated_tokens': 20, 'input_tokens': 40,
              'unknown_inflight_generation_cost': False} for name in ['a', 'b']]
    integrity = {'protocol_sha256': 'abc', 'passed': True, 'mismatches': []}
    supervisor = {'complete': True, 'timed_out': False, 'worker_returncode': 0}
    return [protocol, controller, audit, costs, integrity, supervisor, []]


class CollectionReportContracts(unittest.TestCase):
    def test_full_episode_count_is_not_action_count(self):
        value = summarize(*fixture())
        self.assertEqual(value['eligible_full_episodes'], 2)
        self.assertEqual(value['eligible_action_records'], 4)
        self.assertEqual(value['supervised_tokens'], 10)
        self.assertEqual(value['unexecuted_requests_retained_in_cost'], 2)
        self.assertEqual(value['known_cost']['attempted_calls'], 8)
        self.assertTrue(value['collection_verified_for_candidate_preparation'])
        self.assertFalse(value['model_quality_improvement_established'])

    def test_partial_run_cost_is_retained_and_blocks_data_gate(self):
        items = fixture(); protocol, controller, audit, costs = items[:4]
        controller.update(complete=False); controller['tasks'] = controller['tasks'][:1]
        audit['rows'][1].update(complete=False, input_replay_passed=False, eligible_for_sft=False)
        audit['decision'] = decision(protocol, audit['rows'], controller)
        costs[1].update(complete=False, attempted_calls=4, completed_generation_receipts=3,
                        unknown_inflight_generation_cost=True)
        value = summarize(*items)
        self.assertFalse(value['collection_verified_for_candidate_preparation'])
        self.assertEqual(value['known_cost']['generated_tokens'], 40)
        self.assertEqual(value['partial_tasks'], ['b'])
        self.assertEqual(value['unknown_inflight_tasks'], ['b'])

    def test_report_cannot_lower_the_registered_threshold_or_reorder_tasks(self):
        items = fixture(); items[0]['feasibility_gate']['minimum_eligible_full_episodes'] = 3
        with self.assertRaises(ValueError): summarize(*items)
        items[2]['decision'] = decision(items[0], items[2]['rows'], items[1])
        self.assertFalse(summarize(*items)['collection_verified_for_candidate_preparation'])
        items[3].reverse()
        with self.assertRaises(ValueError): summarize(*items)

    def test_report_checks_original_deadline_and_completed_cost_identity(self):
        items = fixture(); items[1]['finished_epoch'] = 131
        self.assertFalse(summarize(*items)['collection_verified_for_candidate_preparation'])
        items[3][0]['attempted_calls'] = 3
        with self.assertRaises(ValueError): summarize(*items)


if __name__ == '__main__': unittest.main()
