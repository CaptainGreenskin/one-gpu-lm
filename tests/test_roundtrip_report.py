import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from tests.test_interface_report import fixture

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location('roundtrip_'+name,ROOT/'experiments/016-tool-roundtrip'/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
REPORT=module('report')


def inputs():
    cs,p,inventory=fixture();out={}
    for t in inventory:t['split']='dev'
    for arm,old in [('schema','format'),('roundtrip','combined')]:
        c=cs[old];policy={'format':'native_schema' if arm=='schema' else 'native_roundtrip','context':'full'}
        c['identity'].update(variant=arm,policy=policy)
        for r in c['results']:
            r.update(split='dev',variant=arm,agent_policy=policy)
            for g in r['generations']:g.update(agent_policy=policy,dialogue={'structured_tool_calls':0,'tool_results':0});g['context']['policy']='full'
        out[arm]=c
    p['variants']={a:c['identity']['policy'] for a,c in out.items()};p['agent']['max_generated_tokens_per_call']=512
    p['promotion']={'next_fresh_study_candidate_requires':{'target_task':'2','target_passes':True,'minimum_net_historical_gain':1,'maximum_regressed_tasks_vs_concurrent_control':0}}
    return out,p,inventory,copy.deepcopy(out['schema'])


class RoundtripReportContracts(unittest.TestCase):
    def test_known_gain_is_diagnostic_only_and_regressions_fail_gate(self):
        c,p,i,prev=inputs();s=REPORT.assess(c,p,i,prev);self.assertFalse(s['candidate_for_new_holdout'])
        c['roundtrip']['results'][2]['passed']=True;s=REPORT.assess(c,p,i,prev)
        self.assertTrue(s['candidate_for_new_holdout']);self.assertFalse(s['adoption_allowed']);self.assertFalse(s['generalization_established']);self.assertEqual(s['fresh_tasks'],0)
        c['roundtrip']['results'][0]['passed']=False;s=REPORT.assess(c,p,i,prev);self.assertEqual(s['regressed'],['0']);self.assertFalse(s['candidate_for_new_holdout'])

    def test_wrong_role_pair_budget_or_native_history_fails_closed(self):
        for mutate in [lambda c,p,i:i[0].update(split='test'),lambda c,p,i:c['schema']['results'].pop(),lambda c,p,i:c['schema']['results'][0].update(calls=3),lambda c,p,i:c['schema']['results'][0]['generations'][0]['dialogue'].update(tool_results=1),lambda c,p,i:c['roundtrip']['results'][0]['generations'][0]['context'].update(dropped_rounds=1)]:
            c,p,i,prev=inputs();mutate(c,p,i)
            with self.assertRaises(ValueError):REPORT.assess(c,p,i,prev)

    def test_audit_validates_executed_turns_and_exports_no_commands(self):
        from one_gpu_lm.common.receipts import sha256,write_json
        import one_gpu_lm.common.actions as actions
        audit=module('audit')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'controller.json',{'complete':True})
            for arm in REPORT.ARMS:
                private=root/'private'/arm/'synthetic';private.mkdir(parents=True)
                gens=[{'action_format_valid':True,'dialogue':{'structured_tool_calls':0,'tool_results':0}}, {'action_format_valid':False,'dialogue':{'structured_tool_calls':int(arm=='roundtrip'),'tool_results':int(arm=='roundtrip')}}]
                messages=[{'role':'assistant','content':'```bash\necho PRIVATE_TEXT\n```','extra':{'generation':gens[0]}},{'role':'user','content':'PRIVATE_OBSERVATION','extra':{'tool':{'returncode':0}}},{'role':'assistant','content':'<tool_call>PRIVATE_COMMAND','extra':{'generation':gens[1]}}]
                write_json(private/'trajectory.json',{'messages':messages})
                row={'id':'synthetic','split':'dev','passed':False,'exit_status':'RepeatedFormatError','reason':'protected_file_tampering','modified_source_files':[],'generations':gens,'backend_errors':[]}
                write_json(root/(arm+'.json'),{'complete':True,'identity':{'source_files':{'one_gpu_lm/common/actions.py':sha256(Path(actions.__file__))}},'results':[row]})
            result=audit.audit(root);self.assertNotIn('PRIVATE_',json.dumps(result));self.assertTrue(result['variants']['roundtrip'][0]['native_turns_verified'])
            path=root/'private/roundtrip/synthetic/trajectory.json';t=json.loads(path.read_text());t['messages'][2]['extra']['generation']['dialogue']['tool_results']=0;write_json(path,t)
            with self.assertRaises(ValueError):audit.audit(root)


if __name__=='__main__':unittest.main()
