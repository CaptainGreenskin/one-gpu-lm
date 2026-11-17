import copy
from collections import Counter
import unittest
from one_gpu_lm.verified import mixed_eval_design as design
from one_gpu_lm.verified.mixed_evaluation import validate_scope


class FrozenMixedDevelopment(unittest.TestCase):
    def setUp(self):
        self.tasks=[{'id':f't{i:02}','repository':f'repo/{i//10}','family':f'family{i//10}','revision':'a'*40,'split':'development'} for i in range(60)]
        self.sources=[{'repository':f'repo/{i}','family':f'family{i}','revision':'a'*40,'split':'development','language':'python' if i%2 else 'js','approved_for':{'development':True}} for i in range(6)]
        self.train=[{'repository':'other/repo','family':'other','split':'training'}]
        self.spec={'development':{'variants':design.VARIANTS,'unique_tasks':60,'repetitions':3,'block_size':10,'total_task_runs':540,'common_budget':{'max_model_calls':12},'tasks_sha256':'tasks','sources_sha256':'sources'},
          'analysis':{'minimum_development_parameter_gain_pp':10,'maximum_allowed_primary_drop_in_any_repository_pp':10}}
        self.p={'evaluation_budget':{'max_model_calls':12},'schedule':design.schedule([t['id'] for t in self.tasks])}
        self.admission={'complete':True,'passed':True,'tasks_sha256':'tasks','sources_sha256':'sources','runtime_sha256':'runtime','tasks':[{'id':t['id'],'admitted':True} for t in self.tasks]}
    def scope(self):validate_scope(self.p,self.spec,self.tasks,self.sources,self.train,self.admission,'tasks','sources','runtime')
    def rows(self):
        tasks={t['id']:t for t in self.tasks};rows=[]
        for item in self.p['schedule']:
            t=tasks[item['task_id']];v=item['variant'];score=v=='mixed031' or int(t['id'][1:])%2==0
            rows.append({'id':t['id'],'repository':t['repository'],'family':t['family'],'variant':v,'repetition':item['repetition'],
              'complete':True,'audit_passed':True,'success_within_budget':score,'request_replay':{'action_fingerprint':v+t['id']}})
        return rows
    def test_exact_scope_and_balanced_batched_schedule(self):
        self.scope();rows=self.p['schedule'];self.assertEqual(len(rows),540)
        self.assertEqual(Counter(r['variant'] for r in rows),{v:180 for v in design.VARIANTS})
        self.assertEqual(rows,design.schedule(list(reversed([t['id'] for t in self.tasks]))))
        self.assertEqual(1+sum(a['variant']!=b['variant'] for a,b in zip(rows,rows[1:])),52)
        for start in range(0,540,30):
            block=rows[start:start+30]
            self.assertEqual([r['task_id'] for r in block[:10]],[r['task_id'] for r in block[10:20]])
            self.assertEqual([r['task_id'] for r in block[:10]],[r['task_id'] for r in block[20:]])
    def test_final_task_related_family_and_changed_budget_rejected(self):
        self.tasks[0]['split']='final'
        with self.assertRaises(ValueError):self.scope()
        self.tasks[0]['split']='development';self.sources[0]['related_families']=['OTHER']
        with self.assertRaises(ValueError):self.scope()
        self.sources[0].pop('related_families');self.p['evaluation_budget']['max_model_calls']=13
        with self.assertRaises(ValueError):self.scope()
    def test_checkpoint_binding_is_explicit_and_independent(self):
        p={'variants':{'base':{},'original025':{'adapter_directory':'old','adapter_sha256':{'w':'old'},'all_adapter_files_sha256':{'w':'old'}},'mixed031':{'adapter_directory':'new','adapter_sha256':{'w':'new'},'all_adapter_files_sha256':{'w':'new'}}}}
        old=design.effective_protocol(p,'original025');new=design.effective_protocol(p,'mixed031')
        self.assertNotEqual(old['adapter_sha256'],new['adapter_sha256']);old['variants']['mixed031']['adapter_sha256']['w']='changed'
        self.assertEqual(p['variants']['mixed031']['adapter_sha256']['w'],'new')
        self.assertEqual(design.internal_arm('mixed031'),'student_current')
        p['variants']['base']['adapter_directory']='wrong'
        with self.assertRaises(ValueError):design.effective_protocol(p,'base')
    def test_repeats_not_independent_tasks_and_bootstrap_order_is_stable(self):
        rows=self.rows();v=design.analyze(rows,self.tasks,self.spec,True,True)
        self.assertEqual(v['paired_tasks'],60);self.assertAlmostEqual(v['primary_gain'],.5);self.assertTrue(v['development_parameter_gate_passed'])
        self.assertEqual(v,design.analyze(rows,list(reversed(self.tasks)),self.spec,True,True))
        self.assertFalse(design.analyze(rows[:-1],self.tasks,self.spec,True,True)['development_parameter_gate_passed'])
    def test_incomplete_audit_or_duplicate_run_cannot_pass(self):
        rows=self.rows();rows[0]['audit_passed']=False
        self.assertFalse(design.analyze(rows,self.tasks,self.spec,True,True)['development_parameter_gate_passed'])
        with self.assertRaises(ValueError):design.analyze(rows+[rows[0]],self.tasks,self.spec,True,True)


if __name__=='__main__':unittest.main()
