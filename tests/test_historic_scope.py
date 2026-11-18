import copy
import unittest
from one_gpu_lm.verified.historic_eval_design import schedule,VARIANTS
from one_gpu_lm.verified.historic_evaluation import validate_scope


class HistoricalEvaluationScope(unittest.TestCase):
    def setUp(self):
        self.tasks=[];self.sources=[]
        for i in range(36):
            repo=f'owner/repo{i//6}';revision=f'{i+1:040x}'
            source={'source_id':repo.replace('/','--')+'--'+revision,'repository':repo,'family':repo,
                'revision':revision,'upstream_parent':'f'*40,'split':'development',
                'approved_for':{'development':True},'language':'python' if i//6<4 else 'javascript'}
            self.sources.append(source);self.tasks.append(dict(source,id=f't{i}'))
        self.train=[{'source_id':'other--repo--'+'a'*40,'repository':'other/repo','family':'other/repo','split':'training'}]
        self.spec={'development':{'unique_tasks':36,'repository_families':6,'tasks_per_family':6,
            'variants':VARIANTS,'repetitions':3,'block_size':6,'total_task_runs':324,
            'minimum_multifile_tasks':8,'minimum_multifile_families':3,
            'excluded_prior_source_families':['prior/repo'],'tasks_sha256':'t','sources_sha256':'s',
            'common_budget':{'max_model_calls':12}}}
        self.protocol={'evaluation_budget':{'max_model_calls':12},'schedule':schedule([t['id'] for t in self.tasks],3,6)}
        self.admission={'complete':True,'passed':True,'tasks_sha256':'t','sources_sha256':'s','runtime_sha256':{'r':'h'},
            'rows':[{'id':t['id'],'repository':t['repository'],'admitted':True,'multifile':True} for t in self.tasks]}

    def check(self):
        validate_scope(self.protocol,self.spec,self.tasks,self.sources,self.train,self.admission,'t','s',{'r':'h'})

    def test_balanced_scope_and_admission_identity(self):
        self.check();self.admission['runtime_sha256']['r']='changed'
        with self.assertRaises(ValueError):self.check()

    def test_sealed_split_changed_budget_and_prior_alias_rejected(self):
        self.sources[0]['related_families']=['PRIOR/REPO']
        with self.assertRaises(ValueError):self.check()
        self.sources[0].pop('related_families');self.tasks[0]['split']='final'
        with self.assertRaises(ValueError):self.check()
        self.tasks[0]['split']='development';self.protocol['evaluation_budget']['max_model_calls']=13
        with self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()
