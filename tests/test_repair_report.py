import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tests.test_progress_report import fixture as previous_fixture

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location('repair_'+name,ROOT/'experiments/018-repair-verification'/(name+'.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


REPORT = module('report')


def fixture():
    campaigns, protocol, inventory = previous_fixture()
    campaigns['verify'] = campaigns.pop('progress')
    baseline = protocol['variants']['progress']
    protocol['variants'] = {'control':baseline,'verify':dict(baseline,verification='repair_check_v1')}
    protocol['promotion']['target_tasks'] = ['0','1']
    for arm,campaign in campaigns.items():
        policy = protocol['variants'][arm]
        campaign['identity'].update(variant=arm,policy=policy)
        for row in campaign['results']:
            row.update(variant=arm,agent_policy=policy)
            for g in row['generations']:
                g['agent_policy'] = policy
                g['progress'].update(enabled=True,invalid_editable_files=0,editable_state_sha256='b'*64)
                g['verification']={'enabled':arm=='verify','completed_calls':0,'remaining_calls':12,
                    'current_source_sha256':'b'*64,'reminder':False,'reminder_chars':0,'check_status':'not_run'}
    return campaigns,protocol,inventory


class RepairReportContracts(unittest.TestCase):
    def test_only_final_success_can_meet_paired_gate(self):
        c,p,i=fixture()
        self.assertTrue(REPORT.assess(c,p,i)['candidate_for_replication'])
        c['verify']['results'][3]['passed']=False
        # This task still edited source; a later successful tool check is not a final grade.
        s=REPORT.assess(c,p,i)
        self.assertFalse(s['candidate_for_replication'])
        self.assertEqual(s['totals']['verify']['failed_after_any_edit'],1)
        c,p,i=fixture();c['verify']['results'][0]['passed']=False
        s=REPORT.assess(c,p,i)
        self.assertFalse(s['gate_checks']['both_targets_pass'])
        self.assertFalse(s['gate_checks']['no_historical_regressions'])

    def test_baseline_reminders_extra_budget_and_wrong_policy_fail_closed(self):
        for mutate in [lambda c,p,i:c['control']['results'][0]['generations'][0]['verification'].update(reminder=True,reminder_chars=2),
                       lambda c,p,i:c['verify']['results'][0]['generations'][0]['verification'].update(remaining_calls=13),
                       lambda c,p,i:c['control']['results'][0]['generations'][0]['progress'].update(enabled=False),
                       lambda c,p,i:i[3].update(repository='old/repo')]:
            c,p,i=fixture();mutate(c,p,i)
            with self.assertRaises(ValueError): REPORT.assess(c,p,i)

    def test_audit_replays_post_edit_failure_and_rejects_fabricated_status(self):
        from one_gpu_lm.common.receipts import sha256,write_json
        from one_gpu_lm.common.edit_progress import EditableProgress
        from one_gpu_lm.common.repair_verification import prepare_verification
        from one_gpu_lm.common.tool_history import render_history
        audit=module('audit')
        class Tokenizer:
            @classmethod
            def from_pretrained(cls,*args,**kwargs): return cls()
            def apply_chat_template(self,messages,**kwargs): return list(json.dumps(messages,sort_keys=True).encode())
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules,{'transformers':types.SimpleNamespace(AutoTokenizer=Tokenizer)}):
            root=Path(temp);run=root/'runs/score';run.mkdir(parents=True)
            (root/'private').mkdir();exp=root/'experiments/018-repair-verification';exp.mkdir(parents=True)
            model=root/'model';model.mkdir();(model/'chat_template.jinja').write_text('synthetic')
            write_json(root/'runtime.json',{'model':str(model)});write_json(exp/'model-profile.json',{'chat_template_kwargs':{}})
            command='python3 -m pytest -q tests'
            write_json(root/'private/tasks.json',{'tasks':[{'id':'sample','editable_files':['code.py'],'regression_command':command}]})
            write_json(run/'controller.json',{'complete':True,'telemetry':{'thread_stopped':True}})
            for arm in REPORT.ARMS:
                private=run/'private'/arm/'sample';workspace=private/'workspace';workspace.mkdir(parents=True)
                (workspace/'code.py').write_text('original')
                original={'code.py':sha256(workspace/'code.py')};write_json(private/'original.json',original)
                (root/'runs/admission').mkdir(parents=True,exist_ok=True)
                write_json(root/'runs/admission/admission.json',{'tasks':[{'id':'sample','broken_sha256':original}]})
                tracker=EditableProgress(workspace,original,['code.py'])
                ms=[{'role':'system','content':'rules'},{'role':'user','content':'PRIVATE_TASK'}];steps=[];generations=[]
                for index,(action,code) in enumerate([('echo PRIVATE_EDIT',0),(command,1),('echo PRIVATE_SUBMIT',0)]):
                    packed,pr=tracker.prepare_request(ms,index,12,True)
                    packed,vr=prepare_verification(packed,steps,pr,index,12,command,arm=='verify')
                    tokens=Tokenizer().apply_chat_template(render_history(packed,True))
                    g={'progress':pr,'verification':vr,'input_tokens':len(tokens),
                       'input_tokens_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
                       'dialogue':{'structured_tool_calls':index,'tool_results':index},'action_format_valid':True}
                    generations.append(g)
                    ms += [{'role':'assistant','content':'```bash\n'+action+'\n```','extra':{'actions':[{'command':action}],'generation':g}},
                           {'role':'user','content':'PRIVATE_RESULT','extra':{'tool':{'returncode':code}}}]
                    (workspace/'code.py').write_text('edited but wrong')
                    steps.append(dict(tracker.snapshot(),call_index=index+1))
                write_json(private/'trajectory.json',{'messages':ms,'local_model':{'source_progress':steps}})
                row={'id':'sample','split':'dev','passed':False,'calls':3,'tool_calls':3,'generations':generations,
                     'backend_errors':[],'source_progress':steps,'modified_source_files':['code.py']}
                write_json(run/(arm+'.json'),{'complete':True,'results':[row]})
            value=audit.audit(root,run)
            self.assertNotIn('PRIVATE_',json.dumps(value))
            self.assertEqual(value['variants']['verify'][0]['final_observed_check_status'],'failed')
            self.assertEqual(value['variants']['verify'][0]['verification_events'][1]['last_declared_check']['call'],2)
            path=run/'verify.json';data=json.loads(path.read_text())
            data['results'][0]['generations'][2]['verification']['check_status']='passed';write_json(path,data)
            trajectory=run/'private/verify/sample/trajectory.json';data=json.loads(trajectory.read_text())
            data['messages'][6]['extra']['generation']['verification']['check_status']='passed';write_json(trajectory,data)
            with self.assertRaises(AssertionError):audit.audit(root,run)


if __name__=='__main__':
    unittest.main()
