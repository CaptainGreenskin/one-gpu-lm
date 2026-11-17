"""Synthetic checkpoint handoff, execution and replay; never reads real dev tasks."""
import contextlib
import copy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tests import test_verified_evaluation as fixtures
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.verified import mixed_evaluation as controller, mixed_eval_audit as audit, evaluation_rollout, evaluation_audit
from one_gpu_lm.verified.mixed_eval_design import schedule, effective_protocol
from one_gpu_lm.verified.submission import SUBMIT
from one_gpu_lm.verified.tasks import digest


class MixedControllerAudit(unittest.TestCase):
    def test_all_three_checkpoint_identities_survive_recorded_replay(self):
        f=fixtures.EvaluationIsolation();f.setUp();self.addCleanup(f.doCleanups);c=f.c;root=c.root;(root/'private').mkdir()
        c.source['language']='python';tasks=[dict(copy.deepcopy(c.task),id=f'fixture-{i}') for i in range(10)]
        write_json(c.data/'tasks.json',{'tasks':tasks});write_json(c.data/'sources.json',{'sources':[c.source]})
        write_json(root/'training-sources.json',{'sources':[{'repository':'other/train','family':'other','split':'training'}]})
        write_json(root/'private/task-runtime.json',{'fixture':True});write_json(root/'private/parent-runtime.json',{'model':'fixture','model_python':'fixture'})
        write_json(root/'admission.json',{'fixture':True});write_json(root/'environment.json',{'fixture':True});write_json(root/'integrity.json',{'passed':True})
        spec={'development':{'maximum_evaluation_pipeline_seconds':172800}};write_json(root/'spec.json',spec)
        profile={'model_id':c.protocol['model_id'],'revision':c.protocol['revision'],'chat_template_kwargs':{'enable_thinking':False}};write_json(root/'profile.json',profile)
        variants={'base':{}}
        for v in ['original025','mixed031']:
            plan={'profile_path':'profile.json','cycle_spec':'spec.json','frozen_files':{}};write_json(root/(v+'-protocol.json'),plan)
            result=dict.fromkeys(['complete','frozen_files_unchanged','saved_reload_greedy_ids_equal','all_selected_records_consumed','frozen_sample_unchanged','frozen_gradients_absent','adapter_changed'],True)
            result.update(protocol_sha256=sha256(root/(v+'-protocol.json')),profile_sha256=sha256(root/'profile.json'),optimizer_steps=1,candidate_count=1,expected_environment_sha256=sha256(root/'environment.json'))
            write_json(root/(v+'-result.json'),result);write_json(root/(v+'-supervisor.json'),{'complete':True,'candidate_eligible':True,'timed_out':False,'worker_returncode':0})
            variants[v]={'candidate_protocol':v+'-protocol.json','candidate_result':v+'-result.json','candidate_supervisor':v+'-supervisor.json','adapter_directory':v+'-adapter','adapter_sha256':{'weights':v},'all_adapter_files_sha256':{'weights':v}}
        p={'run_directory':'runs/mixed','publication_receipt':'publication.json','cycle_spec':'spec.json','data_directory':'data',
           'training_sources_path':'training-sources.json','admission_path':'admission.json','training_integrity_path':'integrity.json',
           'evaluation_profile_path':'profile.json','evaluation_profile_sha256':sha256(root/'profile.json'),
           'evaluation_budget':dict(c.protocol['evaluation_budget'],max_model_calls=12,max_total_generated_tokens=120,max_task_seconds=600),
           'model_id':profile['model_id'],'revision':profile['revision'],'agent_policy':c.protocol['agent_policy'],'variants':variants,
           'model_manifest_path':'unused.json','supervisor_receipt':'supervisor.json','environment_path':'environment.json',
           'schedule':schedule([t['id'] for t in tasks]),'frozen_files':{}}
        write_json(root/'protocol.json',p);write_json(root/'publication.json',{'anonymous_files_verified':True,'ci_conclusion':'success','protocol_sha256':sha256(root/'protocol.json'),'commit':'fixture'})
        state={'calls':0};decoded={};observed=[]
        def health():
            effective=state['effective'];return {'model_id':profile['model_id'],'revision':profile['revision'],'profile_sha256':sha256(root/'profile.json'),
              'adapter':state['role']=='student','adapter_sha256':effective['adapter_sha256']}
        class Backend:
            def __init__(self,root,out,runtime,protocol,record,changed):self.out=out;self.record=record;self.role=None;self.entry=None
            def ensure(self,role,deadline):
                if self.role==role:
                    if state['effective']['evaluation_variant']!=self.protocol['evaluation_variant']:raise AssertionError('Student weights reused across distinct variants')
                    return 'socket'
                self.role=role;state.update(role=role,effective=copy.deepcopy(self.protocol));index=len(self.record)+1
                self.entry={'index':index,'role':role};self.record.append(self.entry)
                d=self.out/'private/backends'/f'{index:04d}-{role}';d.mkdir(parents=True)
                write_json(d/'served-artifact.json',health());write_json(d.parent/f'{index:04d}-{role}-environment.json',{'fixture':True})
                return 'socket'
            def stop(self):
                if self.entry is not None:self.entry['stopped_epoch']=1
                self.role=None
            def check_exclusive(self):pass
        class Telemetry:
            def __init__(self,*a):pass
            def start(self):pass
            def stop(self):return {'fixture':True,'samples':0}
        def request(socket,payload,**kw):
            if payload['operation']=='health':return health()
            state['calls']+=1;observed.append(copy.deepcopy(payload));call=1+(len(payload['messages'])-2)//2
            command='edit-source' if call==1 else 'python3 check.py' if call==2 else SUBMIT
            text='```bash\n'+command+'\n```';ids=[state['calls'],999];decoded[tuple(ids)]=text
            tokens,context=prepare_context(payload['messages'],lambda _:list(range(20)),8192,'bounded_history_v1')
            return {'text':text,'ids':ids,'generated_tokens':2,'input_tokens':20,'input_tokens_sha256':digest(json.dumps(tokens,separators=(',',':')).encode()),'context':context,'adapter':state['role']=='student'}
        # Only fixture inventory size is bypassed; production60-task scope has separate tests.
        with patch.object(controller,'validate_scope'),patch.object(controller,'Sandbox',return_value=SimpleNamespace(execute=c.execute,timeout=60)),\
             patch.object(controller,'Backend',Backend),patch.object(controller,'Telemetry',Telemetry),patch.object(controller,'verify_model',return_value={'fixture':True}),\
             patch.object(evaluation_rollout,'request',request),patch.object(evaluation_rollout,'normalization',return_value=c.normalizer),contextlib.redirect_stdout(io.StringIO()):
            result=controller.run(root,root/'protocol.json')
        self.assertTrue(result['complete'],result.get('error_type'));self.assertEqual(len(result['runs']),90);self.assertEqual(state['calls'],270)
        self.assertEqual([r['run_id'] for r in result['runs']],[r['run_id'] for r in p['schedule']]);self.assertTrue(all(r['success_within_budget'] for r in result['runs']))
        self.assertNotIn('HIDDEN_CASE_SENTINEL',json.dumps(observed))
        write_json(root/'supervisor.json',{'complete':True,'timed_out':False,'worker_returncode':0,'protocol_sha256':sha256(root/'protocol.json')})
        tokenizer=SimpleNamespace(decode=lambda ids,**kw:decoded[tuple(ids)]);tm=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:tokenizer))
        with patch.dict('sys.modules',{'transformers':tm}),patch.object(evaluation_audit,'encode_prefix',return_value=(list(range(20)),{})),patch.object(audit,'normalization',return_value=c.normalizer):
            value=audit.run(root,root/'protocol.json',root/'audit.json')
        self.assertTrue(value['passed'],[r for r in value['rows'] if not r['audit_passed']][:1]);self.assertEqual(sum(r['attempted_calls'] for r in value['rows']),270)
        self.assertEqual(len(result['backend_lifecycles']),9)
        item=next(r for r in p['schedule'] if r['variant']=='original025');directory=root/p['run_directory']/'private/tasks'/item['run_id'];trace=json.loads((directory/'trajectory.json').read_text())
        with patch.object(evaluation_audit,'encode_prefix',return_value=(list(range(20)),{})):
            wrong=evaluation_audit.replay(directory,trace,tasks[0],effective_protocol(p,'mixed031'),profile,tokenizer,c.normalizer)
        self.assertFalse(wrong['identity_passed'])


if __name__=='__main__':unittest.main()
