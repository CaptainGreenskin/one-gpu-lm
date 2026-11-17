import copy
import unittest
from one_gpu_lm.verified.mixed_training_audit import reconcile


class TrainingReconciliation(unittest.TestCase):
    def fixture(self):
        selected=[{'id':str(i),'episode_id':'e'+str(i),'ids':[1,2,3],'supervised_tokens':2,
                   'record_group_kind':'corrective_fragment' if i==8 else 'whole_episode'} for i in range(9)]
        metadata={'supervised_tokens':18};records=[{k:r[k] for k in ['id','episode_id','supervised_tokens','record_group_kind']}|{'tokens':3} for r in selected]
        selection={'records':records,'selected_records':9,'mixed_selection':metadata,'excluded_records':0,'target_truncation':False}
        curve=[dict(r,record=i+1,update=i//8+1,loss=.2,seconds=.3) for i,r in enumerate(records)]
        updates=[{'step':i+1,'records':n,'learning_rate':1e-5,'grad_norm':.5,'seconds':1.} for i,n in enumerate([8,1])]
        result={'mixed_selection':metadata,'action_records_consumed':9,'optimizer_steps':2,'input_tokens':27,'supervised_tokens':18,
                'development_scores':0,'final_test_opened':False,'weights_public':False,'started_epoch':0,'finished_epoch':10}
        recipe={'gradient_accumulation_records':8,'learning_rate':1e-5,'maximum_training_pipeline_seconds':100}
        return [selected,metadata,selection,curve,updates,result,recipe]

    def test_partial_final_accumulation_and_exact_epoch(self):
        v=reconcile(*self.fixture());self.assertTrue(v['passed']);self.assertEqual(v['last_accumulation_records'],1)
        self.assertEqual(v['record_groups'],{'whole_episode':8,'corrective_fragment':1})

    def test_duplicate_skipped_record_wrong_window_nonfinite_and_leakage_rejected(self):
        mutations=[lambda a:a[3].__setitem__(2,copy.deepcopy(a[3][1])),
          lambda a:a[4][1].update(records=8),lambda a:a[3][0].update(loss=float('nan')),
          lambda a:a[5].update(development_scores=1),lambda a:a[5].update(supervised_tokens=19),
          lambda a:a[2].update(target_truncation=True),lambda a:a[5].update(finished_epoch=101)]
        for mutate in mutations:
            a=self.fixture();mutate(a);self.assertFalse(reconcile(*a)['passed'])


if __name__=='__main__':unittest.main()
