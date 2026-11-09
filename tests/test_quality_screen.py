import copy
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
from one_gpu_lm.common.receipts import sha256, write_json
import one_gpu_lm.common.actions as actions

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
        legacy={'complete':True,'task_sha256':'fixture','results':[{'id':'function','passed':True,'adapter':False}]}
        training={'status':'completed','steps':96,'saved_reload_greedy_ids_equal':True}
        adapted_legacy=copy.deepcopy(legacy);adapted_legacy['results'][0]['adapter']=True
        return base,adapted,protocol,legacy,adapted_legacy,training

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

    def test_failure_audit_preserves_privacy_and_rejects_changed_parser(self):
        path=Path(__file__).resolve().parents[1]/'experiments/014-coding-quality/audit.py'
        spec=importlib.util.spec_from_file_location('quality_audit',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'controller.json',{'complete':True})
            for variant in ['baseline','adapted']:
                private=root/variant/'private/task';private.mkdir(parents=True)
                write_json(root/variant/'campaign.json',{'complete':True,'identity':{'source_files':{'one_gpu_lm/common/actions.py':sha256(Path(actions.__file__))}},'results':[{'id':'task','passed':False,'exit_status':'RuntimeError','reason':'runtime_error','returncode':1,'modified_source_files':[]}]})
                write_json(private/'trajectory.json',{'messages':[{'role':'assistant','content':'<tool_call><function=bash>PRIVATE_COMMAND</function></tool_call>'}]})
                (private/'controller.log').write_text('context_limit PRIVATE_CONTEXT')
                (private/'grader.log').write_text('FAILED PRIVATE_SOURCE')
            result=module.audit(root)
            self.assertNotIn('PRIVATE_',json.dumps(result))
            self.assertEqual(result['variants']['baseline'][0]['runtime_categories'],['context_limit'])
            campaign=json.loads((root/'baseline/campaign.json').read_text());campaign['identity']['source_files']['one_gpu_lm/common/actions.py']='changed';write_json(root/'baseline/campaign.json',campaign)
            with self.assertRaisesRegex(ValueError,'original parser'):module.audit(root)

    def test_short_training_or_failed_reload_cannot_pass(self):
        values=self.inputs();values[-1]['steps']=95
        with self.assertRaises(ValueError):report.assess(*values)
        values=self.inputs();values[-1]['saved_reload_greedy_ids_equal']=False
        self.assertFalse(report.assess(*values)['advance_to_replication'])
        values=self.inputs();values[4]['results'][0]['adapter']=False
        with self.assertRaisesRegex(ValueError,'wrong model variant'):report.assess(*values)


if __name__=='__main__':unittest.main()
