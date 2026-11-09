import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('quality_report',Path(__file__).resolve().parents[1]/'experiments/014-coding-quality/report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


class QualityScreen(unittest.TestCase):
    def inputs(self):
        ids=['old','new1','new2'];base={'complete':True,'identity':{'ids':ids},'results':[]}
        for i,id_ in enumerate(ids):
            base['results'].append({'id':id_,'split':'dev' if i==0 else 'test','repository':'org/old' if i==0 else 'org/new',
                'family':id_,'passed':i==0,'seconds':1,'calls':1,'generated_tokens':1,'exit_status':'Submitted','modified_source_files':['a'],
                'generations':[{'adapter':False}]})
        adapted=copy.deepcopy(base)
        for r in adapted['results']:r['passed']=True;r['generations']=[{'adapter':True}]
        protocol={'evaluation_task_counts':{'dev':1,'test':2},'promotion':{'minimum_net_fresh_holdout_gain':2,'maximum_regressed_fresh_tasks':0,'maximum_regressed_historical_tasks':0,'legacy_function_passes_required':1},'execution':{'max_optimizer_updates':96}}
        legacy={'complete':True,'task_sha256':'fixture','results':[{'id':'function','passed':True}]}
        training={'status':'completed','steps':96,'saved_reload_greedy_ids_equal':True}
        return base,adapted,protocol,legacy,copy.deepcopy(legacy),training

    def test_gain_without_regression_can_only_advance_to_replication(self):
        result=report.assess(*self.inputs())
        self.assertTrue(result['advance_to_replication']);self.assertFalse(result['daily_model_accepted'])

    def test_historical_regression_blocks_despite_fresh_gain(self):
        values=self.inputs();values[1]['results'][0]['passed']=False
        result=report.assess(*values)
        self.assertFalse(result['advance_to_replication']);self.assertFalse(result['gate']['no_historical_regressions'])

    def test_unfinished_missing_duplicate_or_wrong_artifact_is_not_a_pair(self):
        for mutation in [lambda a:a.update(complete=False),lambda a:a['results'].pop(),
                         lambda a:a['results'].append(a['results'][0]),lambda a:a['results'][0].update(generations=[{'adapter':False}])]:
            values=self.inputs();mutation(values[1])
            with self.assertRaises(ValueError):report.assess(*values)

    def test_short_training_or_failed_reload_cannot_pass(self):
        values=self.inputs();values[-1]['steps']=95
        with self.assertRaises(ValueError):report.assess(*values)
        values=self.inputs();values[-1]['saved_reload_greedy_ids_equal']=False
        self.assertFalse(report.assess(*values)['advance_to_replication'])


if __name__=='__main__':unittest.main()
