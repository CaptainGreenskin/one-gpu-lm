"""Training-only early-action contracts, with real disposable command checks."""
import importlib.util
from pathlib import Path
import json
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('action_data', ROOT/'experiments/022-action-repair/build_data.py')
data = importlib.util.module_from_spec(spec); spec.loader.exec_module(data)


class ActionRepairContracts(unittest.TestCase):
    def test_assignment_is_stable_per_repository_and_training_only(self):
        tasks=[{'id':f'{r}-{i}', 'repository':r, 'split':'train'} for r in ['b','a'] for i in range(5)]
        self.assertEqual(data.assign_modes(tasks), data.assign_modes(list(reversed(tasks))))
        self.assertEqual(data.assign_modes(tasks)['a-0'], data.MODES[0])
        self.assertEqual(data.assign_modes(tasks)['a-1'], data.MODES[1])
        for bad in [tasks+[tasks[0]], [dict(tasks[0], split='dev')]]:
            with self.assertRaises(ValueError): data.assign_modes(bad)

    def test_combined_action_really_edits_and_preserves_test_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'source.py';p.write_text('value = 1\n')
            command=data.repair_and_test_command({'source.py':p.read_text()},
                {'source.py':'value = 2\n'}, 'exit 7')
            run=subprocess.run(['/bin/bash','-c',command],cwd=root,capture_output=True)
            self.assertEqual(p.read_text(),'value = 2\n');self.assertEqual(run.returncode,7)

    def test_failed_anchor_stops_before_test_and_before_any_partial_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'a.py').write_text('a = 1\n');(root/'b.py').write_text('unexpected\n')
            command=data.repair_and_test_command({'a.py':'a = 1\n','b.py':'b = 1\n'},
                {'a.py':'a = 2\n','b.py':'b = 2\n'}, 'touch test-ran')
            run=subprocess.run(['/bin/bash','-c',command],cwd=root,capture_output=True)
            self.assertNotEqual(run.returncode,0)
            self.assertEqual((root/'a.py').read_text(),'a = 1\n');self.assertFalse((root/'test-ran').exists())

    def test_missing_read_is_real_without_creating_or_editing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            run=subprocess.run(['/bin/bash','-c',data.missing_read_command()],cwd=temp,capture_output=True,text=True)
            self.assertEqual(run.returncode,1);self.assertIn('FileNotFoundError',run.stderr)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_checked_prefix_does_not_alias_the_executed_episode(self):
        messages=[{'role':'user','content':'synthetic observed failure'}]
        progress=SimpleNamespace(prepare_request=lambda ms,*args:(ms,{'completed_calls':2}))
        episode=SimpleNamespace(messages=messages,steps=[{},{}],progress=progress)
        row,_=data.snapshot_example(episode,'true')
        messages.append({'role':'assistant','content':'later action'})
        self.assertEqual(len(row['messages']),1)
        self.assertNotIn('later action',str(row))


class ActionGates(unittest.TestCase):
    def setUp(self):
        path=ROOT/'experiments/022-action-repair/decision.py'
        spec=importlib.util.spec_from_file_location('action_gates',path)
        self.decision=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.decision)
        self.plan=json.loads(path.with_name('preparation-plan.json').read_text())

    def campaign(self, n, passed):
        return {'complete':True,'results':[{'id':str(i),'passed':i in passed,'generated_tokens':12,
            'generations':[{'action_format_valid':True}]} for i in range(n)]}

    def test_tie_with_reference_keeps_final_sealed(self):
        r=self.decision.select(self.campaign(15,{0}),self.campaign(15,{0,1}),
                               self.campaign(15,{0,1}),map(str,range(15)),self.plan)
        self.assertIsNone(r['selected']);self.assertFalse(r['final_test_opened'])
        r=self.decision.select(self.campaign(15,{0}),self.campaign(15,{0,1}),
                               self.campaign(15,{0,1,2}),map(str,range(15)),self.plan)
        self.assertEqual(r['selected'],'action');self.assertFalse(r['final_test_opened'])

    def test_more_passes_cannot_hide_two_development_regressions(self):
        r=self.decision.select(self.campaign(15,{0,1}),self.campaign(15,{0,1}),
                               self.campaign(15,{2,3,4,5}),map(str,range(15)),self.plan)
        self.assertIsNone(r['selected']);self.assertFalse(r['gates']['reference_regressions'])

    def test_final_gate_requires_fresh_and_historical_and_engineering_checks(self):
        pair=lambda n,a,b:self.decision.paired(self.campaign(n,a),self.campaign(n,b),map(str,range(n)))
        fresh=pair(24,set(),{0,1,2});history=pair(48,{0},{0})
        checks={'legacy_passes':9,'save_reload_greedy_match':True,'rights_and_secret_scan_pass':True,
                'all_execution_audits_pass':True,'single_training_seed_and_repository_cluster_limits_disclosed':True}
        self.assertTrue(self.decision.release_gate(fresh,history,checks,self.plan)['accepted'])
        self.assertFalse(self.decision.release_gate(fresh,pair(48,{0},set()),checks,self.plan)['accepted'])
        self.assertFalse(self.decision.release_gate(fresh,history,dict(checks,all_execution_audits_pass=False),self.plan)['accepted'])
        fresh['candidate_format']['rate']=0.1
        self.assertFalse(self.decision.release_gate(fresh,history,checks,self.plan)['accepted'])

    def test_missing_pair_cannot_change_effective_test_size(self):
        with self.assertRaises(ValueError):
            self.decision.paired(self.campaign(15,set()),self.campaign(14,set()),map(str,range(15)))


if __name__ == '__main__': unittest.main()
