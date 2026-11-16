"""Intervention, genuine failed-edit provenance and conditional export boundaries."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_verified_rollout as fixtures
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.verified import correction_rollout as rr, correction_replay as replay_module, correction_audit as audit
from one_gpu_lm.verified.correction_policy import correction_chain
from one_gpu_lm.verified.correction import select_tasks, schedule
from one_gpu_lm.verified.correction_report import summarize
from one_gpu_lm.verified.submission import SUBMIT
from one_gpu_lm.verified.tasks import digest


class CorrectiveCollection(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.RolloutBoundaries();self.f.setUp();self.addCleanup(self.f.doCleanups);f=self.f
        f.protocol['evaluation_budget']=f.protocol.pop('teacher_budget')
        f.protocol['evaluation_budget'].update(max_model_calls=18,max_task_seconds=900,max_generated_tokens_per_call=512,max_total_generated_tokens=9216)
        f.protocol['evaluation_profile_sha256']=f.protocol.pop('teacher_profile_sha256')
        f.protocol['adapter_sha256']={};f.protocol['encoding_screen_max_tokens']=8192
        self.profile={'chat_template_kwargs':{'enable_thinking':False}};self.decoded={};self.serial=0;self.requests=[];self.actions=[]
        self.runner=SimpleNamespace(execute=self.execute,timeout=5)
        for p in [patch.object(rr,'request',self.request),patch.object(rr,'normalization',return_value=f.normalizer),
                  patch.object(replay_module,'encode_prefix',return_value=(list(range(20)),{}))]:
            p.start();self.addCleanup(p.stop)
    def execute(self,root,command):
        if command=='bad-edit':(root/'logic.py').write_text('answer = 5\n')
        if command=='edit-source':(root/'logic.py').write_text('answer = 7\n')
        if command=='tamper-test':(root/'check.py').write_text('tampered')
        if command=='restore-test':(root/'check.py').write_text('visible check')
        if command=='python3 hidden.py':self.assertEqual((root/'hidden.py').read_text(),'HIDDEN_CASE_SENTINEL')
        failed=command in ['python3 check.py','python3 hidden.py'] and (root/'logic.py').read_text()!='answer = 7\n'
        return {'returncode':1 if failed else 0,'reason':'runtime_error' if failed else 'ok','output':'ACTUAL_FAILURE' if failed else 'PASS','seconds':.001}
    def request(self,socket,payload,**kwargs):
        if payload['operation']=='health':return dict(self.f.health,adapter_sha256={})
        self.requests.append(copy.deepcopy(payload));command=self.actions.pop(0)
        if command is None:raise TimeoutError()
        self.serial+=1;text='```bash\n'+command+'\n```';ids=[self.serial,777];self.decoded[tuple(ids)]=text
        tokens,context=prepare_context(payload['messages'],lambda _:list(range(20)),8192,'bounded_history_v1')
        return {'text':text,'ids':ids,'generated_tokens':2,'input_tokens':len(tokens),'input_tokens_sha256':digest(json.dumps(tokens,separators=(',',':')).encode()),'context':context,'adapter':False}
    def decode(self,ids,**kwargs):return self.decoded[tuple(ids)]
    def collect(self,arm,actions,directory=None):
        f=self.f;out=directory or f.root/arm;self.actions=actions[:]
        trace=rr.initialize(f.data,f.task,f.source,f.protocol,out,arm)
        rr.advance(f.root,f.task,f.protocol,out,trace,'socket',self.runner,arm)
        result=rr.finish(f.data,f.task,f.source,f.protocol,out,trace,self.runner)
        return out,trace,result
    def replay(self,out,trace):return replay_module.replay(out,trace,self.f.task,self.f.protocol,self.profile,self,self.f.normalizer)
    def test_enforces_check_without_executing_on_model_behalf_and_replays(self):
        out,t,r=self.collect('correction',['bad-edit','cat logic.py','python3 check.py','edit-source',SUBMIT,'python3 check.py',SUBMIT])
        self.assertTrue(r['success_within_budget']);self.assertTrue(correction_chain(t,self.f.task)['qualified_behavior'])
        self.assertFalse(t['steps'][1]['observation']['executed']);self.assertFalse(t['steps'][4]['observation']['executed'])
        self.assertEqual(t['attempted_calls'],7);self.assertTrue(self.replay(out,t)['passed'])
        self.assertIn('ACTUAL_FAILURE',self.requests[3]['messages'][-1]['content'])
        self.assertNotIn('HIDDEN_CASE_SENTINEL',json.dumps(self.requests))
        self.assertNotIn('mutations',json.dumps(self.requests))
    def test_success_on_first_edit_and_two_edits_without_test_failure_do_not_qualify(self):
        for arm,actions in [('correction',['edit-source','python3 check.py',SUBMIT]),('control',['bad-edit','edit-source','python3 check.py',SUBMIT])]:
            out,t,r=self.collect(arm,actions);self.assertTrue(r['success_within_budget'])
            self.assertFalse(correction_chain(t,self.f.task)['qualified_behavior']);self.assertTrue(self.replay(out,t)['passed'])
    def test_initial_fault_failure_timeout_and_test_tampering_are_not_corrections(self):
        out,t,r=self.collect('control',['python3 check.py','bad-edit','python3 check.py','edit-source','python3 check.py',SUBMIT])
        chain=correction_chain(t,self.f.task);self.assertEqual(chain['qualifying_failure_calls'],[3])
        changed=copy.deepcopy(t);changed['steps'][2]['observation'].update(returncode=-1,reason='timeout')
        self.assertFalse(correction_chain(changed,self.f.task)['qualified_behavior'])
        changed=copy.deepcopy(t);changed['steps'][1]['protected_after']=False
        self.assertFalse(correction_chain(changed,self.f.task)['qualified_behavior'])
    def test_source_evidence_and_policy_text_tampering_fail_replay(self):
        out,t,r=self.collect('correction',['bad-edit','python3 check.py','edit-source','python3 check.py',SUBMIT])
        changed=copy.deepcopy(t);changed['steps'][1]['policy_notice']='invented hint'
        self.assertFalse(self.replay(out,changed)['passed'])
        path=out/'source-states'/(t['steps'][0]['source_evidence_after']+'.json');path.write_text('{}')
        self.assertFalse(self.replay(out,t)['passed'])
    def test_unknown_generation_is_not_retried_and_counts_attempt(self):
        with self.assertRaises(RuntimeError):self.collect('control',[None])
        t=json.loads((self.f.root/'control/trajectory.json').read_text());self.assertEqual(t['attempted_calls'],1)
        self.assertTrue(t['unknown_inflight_generation_cost']);self.assertEqual(len(self.requests),1)
    def test_cropped_failure_context_or_truncated_target_cannot_be_exported(self):
        f=self.f;out,t,r=self.collect('correction',['bad-edit','python3 check.py','edit-source','python3 check.py',SUBMIT])
        with patch.object(audit,'encode_prefix',return_value=(list(range(20)),{})),patch.object(audit,'encode_training_example',return_value={'ids':list(range(20))+[888],'supervised_tokens':1}):
            def cropped(messages,encode,cap,policy):
                _,context=prepare_context(messages,lambda _:list(range(20)),cap,policy)
                return encode([m for m in messages if 'ACTUAL_FAILURE' not in m['content']]),context
            with patch.object(audit,'prepare_context',cropped):
                result,records=audit.encode_correction(out,t,f.task,f.protocol,self.profile,self)
            self.assertFalse(result['qualified']);self.assertTrue(result['encoding_errors'])
            with patch.object(audit,'encode_training_example',return_value={'ids':list(range(9000)),'supervised_tokens':1}):
                result,records=audit.encode_correction(out,t,f.task,f.protocol,self.profile,self)
            self.assertFalse(result['qualified']);self.assertTrue(result['encoding_errors'])
    def test_same_hard_budget_applies_to_both_arms(self):
        for arm in ['control','correction']:
            out,t,r=self.collect(arm,['cat logic.py']*18)
            self.assertEqual(t['attempted_calls'],18);self.assertEqual(t['terminal_reason'],'call_limit')
            self.assertTrue(self.replay(out,t)['passed']);self.assertFalse(r['success_within_budget'])
    def test_selection_excludes_parent_tasks_and_nontraining(self):
        parent={'passed':True,'rows':[{'id':'old','disposition':'paired','student':{'success_within_budget':False},'teacher':{'success_within_budget':False}}]}
        tasks=[{'id':n,'repository':'r'+str(i%3),'split':'training'} for i,n in enumerate(['old',*[str(x) for x in range(30)]])]
        a,b=select_tasks(tasks,parent,24);self.assertEqual(a,['old']);self.assertNotIn('old',b)
        self.assertEqual((a,b),select_tasks(list(reversed(tasks)),parent,24));self.assertEqual(len(schedule(a+b)),50)
        tasks[0]['split']='development'
        with self.assertRaises(ValueError):select_tasks(tasks,parent,24)
    def test_full_audit_export_excludes_bad_edit_targets_and_gate_remains_closed(self):
        f=self.f;p=f.protocol;run=f.root/'run'
        p.update(run_directory='run',data_directory='data',supervisor_receipt='supervisor.json',evaluation_profile_path='profile.json',
            environment_path='environment.json',frozen_files={},diagnostic_task_ids=['fixture'],supplemental_task_ids=[],schedule=schedule(['fixture']),
            global_caps={'model_calls':36,'generated_tokens':18432},feasibility_gate={'minimum_qualified_unique_tasks':6,'minimum_qualified_families':3,'minimum_qualified_supplemental_tasks':3,'minimum_net_qualified_gain':3})
        def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        write(f.root/'protocol.json',p);write(f.root/'profile.json',self.profile);write(f.root/'private/parent-runtime.json',{'model':'fixture'})
        write(f.root/'environment.json',{'fixture':True});write(f.data/'tasks.json',{'tasks':[f.task]});write(f.data/'sources.json',{'sources':[f.source]})
        results=[]
        for arm,actions in [('control',['edit-source','python3 check.py',SUBMIT]),('correction',['bad-edit','python3 check.py','edit-source','python3 check.py',SUBMIT])]:
            rid='fixture--'+arm;out,t,r=self.collect(arm,actions,run/'private/tasks'/rid)
            r.update(run_id=rid,arm=arm,cohort='diagnostic',correction_behavior=correction_chain(t,f.task));write(out/'result.json',r);results.append(r)
        write(run/'controller.json',{'complete':True,'backend_stopped':True,'runs':results,'protocol_sha256':digest((f.root/'protocol.json').read_bytes()),'backend_lifecycles':[{'index':1,'role':'base'}]})
        write(run/'private/backends/0001-base-environment.json',{'fixture':True});write(f.root/'supervisor.json',{'complete':True,'timed_out':False,'worker_returncode':0})
        tm=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args,**kw:self))
        with patch.dict(sys.modules,{'transformers':tm}),patch.object(audit,'normalization',return_value=f.normalizer),patch.object(audit,'encode_prefix',return_value=(list(range(20)),{})),patch.object(audit,'encode_training_example',return_value={'ids':list(range(20))+[888],'supervised_tokens':1}):
            v=audit.run(f.root,f.root/'protocol.json',run/'audit.json')
        self.assertTrue(v['passed'],v);records=[json.loads(line) for line in (run/'private/correction-export/data.jsonl').read_text().splitlines()]
        self.assertEqual([r['provenance']['call'] for r in records],[3,4,5])
        self.assertIn('ACTUAL_FAILURE',json.dumps(records[0]['messages']));self.assertNotIn('bad-edit',records[0]['target'])
        self.assertNotIn('HIDDEN_CASE_SENTINEL',json.dumps(records));self.assertEqual(sum(a['attempted_calls'] for a in v['attempts']),8)
        s=summarize(v,p);self.assertEqual(s['outcomes']['correction']['qualified'],1);self.assertFalse(s['data_feasibility_gate_passed'])


if __name__=='__main__':unittest.main()
