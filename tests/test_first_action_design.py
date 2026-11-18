import unittest

from one_gpu_lm.verified.first_action_design import analyze, schedule


def inventory():
    return [{'id':f'{family}-{i}', 'family':str(family), 'split':'development'}
            for family in range(4) for i in range(5)]


class FirstActionDesignTest(unittest.TestCase):
    def test_same_twenty_tasks_and_balanced_order(self):
        tasks = inventory()
        rows = schedule(tasks)
        self.assertEqual(rows, schedule(tasks[::-1]))
        self.assertEqual(len({r['run_id'] for r in rows}), 40)
        self.assertEqual([rows[i]['arm'] for i in range(0,40,10)], ['control','masked','control','masked'])
        for i in range(0,40,10):
            self.assertEqual([r['task_id'] for r in rows[i:i+5]], [r['task_id'] for r in rows[i+5:i+10]])
        tasks[0]['split'] = 'training'
        with self.assertRaises(ValueError):
            schedule(tasks)

    def test_paired_estimate_requires_matching_opening_and_counts_lost_cases(self):
        tasks = inventory()
        rows = []
        for item in schedule(tasks):
            n = item['task_number']
            arm = item['arm']
            rows.append({'task_number':n, 'arm':arm, 'audit_passed':True, 'family':item['task_id'][0],
                         'first_input_sha256':str(n), 'multifile':False,
                         'completed_repair':(arm=='masked' and n in (1,2,3)) or (arm=='control' and n==4),
                         'correct_without_submission':False, 'first_exact_regression':arm=='control',
                         'no_edit':False, 'attempted_calls':12, 'input_tokens':1220, 'generated_tokens':128,
                         'timeout_recovered':False})
        result = analyze(rows,tasks)
        self.assertEqual((result['masked_only_success'],result['control_only_success']), (3,1))
        self.assertEqual(result['paired_success_delta'], 0.1)
        self.assertTrue(result['mechanism_followup_gate_passed'])
        self.assertFalse(analyze(rows[:-1],tasks)['mechanism_followup_gate_passed'])
        rows[-1]['first_input_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            analyze(rows,tasks)
