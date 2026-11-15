"""CPU-only synthetic controller/audit integration; no real model or dev tasks."""
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_verified_evaluation as fixtures
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.verified import evaluation, evaluation_rollout, evaluation_audit
from one_gpu_lm.verified.factorial import ARMS, schedule
from one_gpu_lm.verified.submission import SUBMIT
from one_gpu_lm.verified.tasks import digest


class ControllerToRecordedAudit(unittest.TestCase):
    def test_registered_schedule_and_artifact_handoffs_replay_without_model_calls(self):
        fixture=fixtures.EvaluationIsolation();fixture.setUp();self.addCleanup(fixture.doCleanups)
        c=fixture.c;root=c.root;(root/'private').mkdir()
        c.source['language']='python'
        tasks=[dict(copy.deepcopy(c.task),id=f'fixture-{i}') for i in range(10)]
        write_json(c.data/'tasks.json',{'tasks':tasks});write_json(c.data/'sources.json',{'sources':[c.source]})
        write_json(root/'training-sources.json',{'sources':[{'repository':'other/train','family':'other','split':'training'}]})
        write_json(root/'previous.json',{'repository_names':['earlier/project']})
        write_json(root/'private/task-runtime.json',{'fixture':True})
        write_json(root/'private/parent-runtime.json',{'model':'fixture-model','model_python':'fixture-python'})
        write_json(root/'admission.json',{'complete':True,'passed':True,'tasks_sha256':sha256(c.data/'tasks.json'),
                   'sources_sha256':sha256(c.data/'sources.json'),'runtime_sha256':sha256(root/'private/task-runtime.json'),
                   'tasks':[{'id':t['id'],'admitted':True} for t in tasks]})
        budget=dict(c.protocol['evaluation_budget'],max_model_calls=12,max_total_generated_tokens=120,max_task_seconds=600)
        spec={'development':{'unique_tasks':10,'repository_families':1,'languages':1,'repetitions':3,'total_task_runs':120,
                  'arms':ARMS,'common_budget':budget,'tasks_sha256':sha256(c.data/'tasks.json'),
                  'sources_sha256':sha256(c.data/'sources.json'),'maximum_evaluation_pipeline_seconds':172800}}
        write_json(root/'spec.json',spec)
        profile={'model_id':c.protocol['model_id'],'revision':c.protocol['revision'],'chat_template_kwargs':{}}
        write_json(root/'profile.json',profile)
        write_json(root/'environment.json',{'fixture':True})
        write_json(root/'candidate-protocol.json',{'profile_path':'profile.json','cycle_spec':'spec.json','frozen_files':{}})
        candidate=dict.fromkeys(['complete','frozen_files_unchanged','saved_reload_greedy_ids_equal','all_selected_records_consumed',
                         'frozen_sample_unchanged','frozen_gradients_absent','adapter_changed'],True)
        candidate.update(protocol_sha256=sha256(root/'candidate-protocol.json'),profile_sha256=sha256(root/'profile.json'),
                         optimizer_steps=1,candidate_count=1,expected_environment_sha256=sha256(root/'environment.json'))
        write_json(root/'candidate-result.json',candidate)
        write_json(root/'candidate-supervisor.json',{'complete':True,'candidate_eligible':True,'timed_out':False,'worker_returncode':0})
        protocol={'run_directory':'runs/factorial','publication_receipt':'publication.json','cycle_spec':'spec.json',
          'data_directory':'data','training_sources_path':'training-sources.json','previous_families_path':'previous.json',
          'admission_path':'admission.json','candidate_result':'candidate-result.json','candidate_supervisor':'candidate-supervisor.json',
          'candidate_protocol':'candidate-protocol.json','evaluation_profile_path':'profile.json','evaluation_profile_sha256':sha256(root/'profile.json'),
          'evaluation_budget':budget,'model_id':profile['model_id'],'revision':profile['revision'],
          'agent_policy':c.protocol['agent_policy'],'adapter_sha256':{'adapter.safetensors':'fixture-candidate'},
          'model_manifest_path':'fixture-unused-manifest.json','supervisor_receipt':'supervisor.json',
          'environment_path':'environment.json',
          'schedule':schedule([t['id'] for t in tasks]),'frozen_files':{}}
        write_json(root/'protocol.json',protocol)
        write_json(root/'publication.json',{'anonymous_files_verified':True,'ci_conclusion':'success',
                   'protocol_sha256':sha256(root/'protocol.json'),'commit':'fixture-commit'})
        state={'role':None,'requests':0};decoded={};observed=[]
        class FixtureBackend:
            def __init__(self,root,out,runtime,protocol,record,changed):self.role=None;self.record=record;self.out=out
            def ensure(self,role,deadline):
                if self.role!=role:
                    index=len(self.record)+1;self.record.append({'index':index,'role':role})
                    path=self.out/'private/backends'/f'{index:04d}-{role}-environment.json'
                    path.parent.mkdir(parents=True,exist_ok=True);write_json(path,{'fixture':True})
                self.role=role;state['role']=role;return 'fixture-socket'
            def stop(self):self.role=None
        class FixtureTelemetry:
            def __init__(self,*args):pass
            def start(self):pass
            def stop(self):return {'samples':0,'fixture':True}
        def request(socket,payload,**kwargs):
            student=state['role']=='student'
            if payload['operation']=='health':
                return {'model_id':profile['model_id'],'revision':profile['revision'],'profile_sha256':sha256(root/'profile.json'),
                        'adapter':student,'adapter_sha256':protocol['adapter_sha256'] if student else {}}
            state['requests']+=1;observed.append(copy.deepcopy(payload))
            call=1+(len(payload['messages'])-2)//2
            command='edit-source' if call==1 else 'python3 check.py' if call==3 else SUBMIT
            text='```bash\n'+command+'\n```';ids=[state['requests'],999];decoded[tuple(ids)]=text
            tokens,context=prepare_context(payload['messages'],lambda _:list(range(20)),8192,'bounded_history_v1')
            return {'text':text,'ids':ids,'generated_tokens':2,'input_tokens':20,
                    'input_tokens_sha256':digest(json.dumps(tokens,separators=(',',':')).encode()),'context':context,'adapter':student}
        runner=SimpleNamespace(execute=c.execute,timeout=60)
        with patch.object(evaluation,'Sandbox',return_value=runner),patch.object(evaluation,'Backend',FixtureBackend),\
             patch.object(evaluation,'Telemetry',FixtureTelemetry),patch.object(evaluation,'verify_model',return_value={'fixture':True}),\
             patch.object(evaluation_rollout,'request',request),patch.object(evaluation_rollout,'normalization',return_value=c.normalizer),\
             contextlib.redirect_stdout(io.StringIO()):
            result=evaluation.run(root,root/'protocol.json')
        self.assertTrue(result['complete'],result.get('error_type'))
        self.assertEqual([r['run_id'] for r in result['runs']],[r['run_id'] for r in protocol['schedule']])
        self.assertEqual(len(result['runs']),120);self.assertEqual(state['requests'],360)
        self.assertTrue(all(r['success_within_budget'] for r in result['runs']))
        self.assertTrue(all('HIDDEN_CASE_SENTINEL' not in json.dumps(r) for r in observed))
        write_json(root/'supervisor.json',{'complete':True,'timed_out':False,'worker_returncode':0})
        tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:decoded[tuple(ids)])
        module=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args,**kwargs:tokenizer))
        with patch.dict('sys.modules',{'transformers':module}),\
             patch.object(evaluation_audit,'encode_prefix',return_value=(list(range(20)),{})),\
             patch.object(evaluation_audit,'normalization',return_value=c.normalizer):
            audited=evaluation_audit.run(root,root/'protocol.json',root/'audit.json')
        self.assertTrue(audited['passed'],[r for r in audited['rows'] if not r['audit_passed']][:1])
        self.assertEqual(sum(r['attempted_calls'] for r in audited['rows']),360)
        self.assertEqual(audited['model_generations'],0);self.assertEqual(audited['test_commands_reexecuted'],0)


if __name__=='__main__':unittest.main()
