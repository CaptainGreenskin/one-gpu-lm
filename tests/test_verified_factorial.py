import copy
import unittest
from one_gpu_lm.verified.factorial import ARMS,analyze,effects,schedule


class FactorialContracts(unittest.TestCase):
    def setUp(self):
        self.tasks=[{'id':f't{i}','repository':f'r{i//2}','family':f'f{i//2}'} for i in range(4)]
        self.spec={'development':{'unique_tasks':4,'repository_families':2,'repetitions':3,'total_task_runs':48},
                   'analysis':{'minimum_development_parameter_gain_pp':10,'maximum_allowed_primary_drop_in_any_repository_pp':10}}
        self.rows=[]
        for task in self.tasks:
            for rep in range(3):
                for arm in ARMS:
                    self.rows.append({'id':task['id'],'repository':task['repository'],'repetition':rep,'arm':arm,
                         'complete':True,'success_within_budget':arm.startswith('student_'),
                         'attempted_calls':2,'generated_tokens':5,'seconds':1})
    def test_schedule_covers_each_arm_once_per_task_repeat_with_balanced_rotation(self):
        plan=schedule(['a','b','c','d'])
        self.assertEqual(len(plan),48)
        self.assertEqual(len({(r['task_id'],r['repetition'],r['arm']) for r in plan}),48)
        self.assertEqual(plan,schedule(['d','c','b','a']))
        self.assertEqual({r['arm'] for r in plan[:16:4]},set(ARMS))
    def test_system_gain_cannot_be_called_parameter_gain(self):
        self.assertEqual(effects(dict(zip(ARMS,[.5,.8,.5,.8]))),{'parameter':0.,'system':.30000000000000004,'parameter_under_guard':0.,'interaction':0.})
    def test_repeated_wins_are_four_task_units_and_do_not_authorize_weights(self):
        value=analyze(self.rows,self.tasks,self.spec,True,True)
        self.assertTrue(value['development_parameter_gate_passed'])
        self.assertEqual(value['independent_task_units'],4)
        self.assertEqual(value['intervals_95']['parameter'],[1.,1.])
        self.assertFalse(value['weight_publication_allowed'])
    def test_partial_pair_fails_gate_without_discarding_cost(self):
        self.rows[0].update(complete=False,unknown_inflight_generation_cost=True)
        value=analyze(self.rows,self.tasks,self.spec,True,True)
        self.assertFalse(value['development_parameter_gate_passed'])
        self.assertEqual(value['paired_tasks'],3);self.assertEqual(value['cost']['generated_tokens'],240)
        self.assertEqual(value['cost']['unknown_inflight_runs'],1)
    def test_duplicate_run_is_not_averaged_as_an_extra_repeat(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaises(ValueError):analyze(self.rows,self.tasks,self.spec,True,True)
    def test_unverifiable_score_is_excluded_from_pairs_but_retains_cost(self):
        self.rows[0].update(audit_passed=False)
        self.rows[0].pop('success_within_budget')
        value=analyze(self.rows,self.tasks,self.spec,False,True)
        self.assertFalse(value['development_parameter_gate_passed'])
        self.assertEqual(value['paired_tasks'],3)
        self.assertEqual(value['cost']['generated_tokens'],240)
        self.assertEqual(value['cost']['unverified_score_runs'],1)


if __name__=='__main__':unittest.main()
