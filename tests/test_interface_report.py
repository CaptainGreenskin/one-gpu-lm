import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('interface_report',Path(__file__).resolve().parents[1]/'experiments/015-agent-interface/report.py')
REPORT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(REPORT)


def fixture():
    policies={a:{'format':'native_schema' if a in {'format','combined'} else 'legacy',
                 'context':'bounded_history_v1' if a in {'context','combined'} else 'full'} for a in REPORT.ARMS}
    protocol={'variants':policies,'execution':{'expected_tasks_per_variant':4},
        'agent':{'max_model_calls':12,'max_total_generated_tokens':6144,'max_input_tokens':8192},
        'evaluation_task_counts':{'dev':2,'test':2},
        'promotion':{'minimum_net_fresh_gain':2,'maximum_regressed_fresh_tasks':0,'maximum_regressed_historical_tasks':0}}
    inventory=[{'id':str(i),'repository':'fixture/'+str(i//2),'split':'dev' if i<2 else 'test','family':'test'} for i in range(4)]
    campaigns={}
    for arm in REPORT.ARMS:
        policy=policies[arm];rows=[]
        for t in inventory:
            g={'generated_tokens':10,'seconds':1.,'input_tokens':100,'adapter':False,'agent_policy':policy,
                'action_format_valid':True,'hit_generation_limit':False,
                'context':{'policy':policy['context'],'raw_input_tokens':100,'packed_input_tokens':100,'dropped_rounds':0}}
            rows.append(dict(t,variant=arm,agent_policy=policy,passed=t['split']=='dev',calls=1,generated_tokens=10,
                generations=[g],backend_errors=[],seconds=2.,tool_seconds=.1,tool_calls=1,exit_status='Submitted',modified_source_files=[],reason='ok'))
        campaigns[arm]={'complete':True,'identity':{'ids':[t['id'] for t in inventory],'variant':arm,'policy':policy},'results':rows}
    return campaigns,protocol,inventory


class InterfaceReportContracts(unittest.TestCase):
    def test_postrun_audit_exports_shapes_only_and_checks_original_parser_decisions(self):
        from one_gpu_lm.common.receipts import sha256,write_json
        import one_gpu_lm.common.actions as actions
        path=Path(__file__).resolve().parents[1]/'experiments/015-agent-interface/audit.py'
        spec=importlib.util.spec_from_file_location('interface_audit',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'controller.json',{'complete':True})
            for arm in REPORT.ARMS:
                private=root/'private'/arm/'synthetic';private.mkdir(parents=True)
                messages=[{'role':'assistant','content':'```bash\necho PRIVATE_MODEL_TEXT\n```'},
                    {'role':'assistant','content':'<tool_call><function=bash>PRIVATE_COMMAND'}]
                write_json(private/'trajectory.json',{'messages':messages})
                row={'id':'synthetic','split':'test','passed':False,'exit_status':'RepeatedFormatError',
                    'reason':'protected_file_tampering','modified_source_files':[],
                    'generations':[{'action_format_valid':True},{'action_format_valid':False}],'backend_errors':[]}
                write_json(root/(arm+'.json'),{'complete':True,'identity':{'source_files':{
                    'one_gpu_lm/common/actions.py':sha256(Path(actions.__file__))}},'results':[row]})
            output=module.audit(root)
            self.assertNotIn('PRIVATE_',json.dumps(output))
            self.assertEqual(output['variants']['baseline'][0]['format_categories'],{'unclosed_native_markup':1})
            self.assertIsNone(output['variants']['baseline'][0]['grader_log_sha256'])
            c=json.loads((root/'baseline.json').read_text());c['results'][0]['generations'][1]['action_format_valid']=True
            write_json(root/'baseline.json',c)
            with self.assertRaises(ValueError):module.audit(root)

    def test_no_gain_does_not_promote_and_gains_are_paired(self):
        c,p,i=fixture();s,calls=REPORT.assess(c,p,i)
        self.assertFalse(any(v['advance_to_replication'] for v in s['promotion'].values()))
        self.assertEqual(len(calls),16)
        for r in c['combined']['results']:r['passed']=True
        s,_=REPORT.assess(c,p,i)
        self.assertTrue(s['promotion']['combined']['advance_to_replication'])
        self.assertEqual(s['contrasts']['combined-vs-baseline']['fresh']['improved'],['2','3'])
        self.assertEqual(s['interaction_net_tasks']['fresh'],2)
        c['combined']['results'][0]['passed']=False
        s,_=REPORT.assess(c,p,i)
        self.assertFalse(s['promotion']['combined']['advance_to_replication'])

    def test_incomplete_missing_duplicate_or_wrong_policy_fail_closed(self):
        mutations=[lambda c:c['format'].update(complete=False),
                   lambda c:c['format']['results'].pop(),
                   lambda c:c['format']['results'].__setitem__(0,copy.deepcopy(c['format']['results'][1])),
                   lambda c:c['format']['results'][0].update(agent_policy={'format':'legacy','context':'full'})]
        for mutate in mutations:
            c,p,i=fixture();mutate(c)
            with self.assertRaises(ValueError):REPORT.assess(c,p,i)

    def test_silent_trimming_missing_request_or_wrong_model_rejected(self):
        mutations=[lambda r:r['generations'][0]['context'].update(dropped_rounds=1),
                   lambda r:r['generations'][0]['context'].update(raw_input_tokens=999),
                   lambda r:r.update(calls=2),lambda r:r['generations'][0].update(adapter=True),
                   lambda r:r.update(generated_tokens=20)]
        for mutate in mutations:
            c,p,i=fixture();mutate(c['baseline']['results'][0])
            with self.assertRaises(ValueError):REPORT.assess(c,p,i)

    def test_context_failure_remains_a_scored_failure_with_token_evidence(self):
        c,p,i=fixture();r=c['baseline']['results'][2]
        r.update(calls=2,backend_errors=[{'error':'context_limit','input_tokens':9000,
            'context':{'policy':'full','raw_input_tokens':9000,'packed_input_tokens':9000,'dropped_rounds':0}}])
        s,rows=REPORT.assess(c,p,i)
        self.assertEqual(s['totals']['baseline']['context_limit_requests'],1)
        failed=next(row for row in rows if row['backend_error'])
        self.assertIsNone(failed['generation_seconds'])
        self.assertEqual(failed['raw_tokens'],9000)


if __name__=='__main__':unittest.main()
