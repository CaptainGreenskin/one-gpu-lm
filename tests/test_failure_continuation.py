"""Equal-budget continuations, real historical failure context and audited export."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_correction_collection as fixtures
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.verified import failure_states as fs, failure_continuation as fc, failure_audit as audit, failure_replay as replay
from one_gpu_lm.verified import correction_rollout as rr
from one_gpu_lm.verified.correction import schedule
from one_gpu_lm.verified.submission import SUBMIT
from one_gpu_lm.verified.tasks import digest
from one_gpu_lm.verified.failure_report import summarize


class FailureContinuation(unittest.TestCase):
    def setUp(self):
        self.h=fixtures.CorrectiveCollection();self.h.setUp();self.addCleanup(self.h.doCleanups);self.f=self.h.f
        f=self.f;out,t,_=self.h.collect('correction',['bad-edit','python3 check.py','edit-source','python3 check.py',SUBMIT])
        ref={'origin':'027','branch':'correction','directory':out,'trace':t};item=fs.candidates(t,f.task)[0]
        value=fs.replay_to_failure(f.root,f.data,ref,item,f.task,f.source,f.root/'admitted',self.h.runner,f.normalizer,2400)
        self.selected=dict(item,state_id='state-001',task_id=f.task['id'],snapshot=value['snapshot'],snapshot_directory='admitted/snapshot')
        _,self.prefix,self.manifest=fc.load_snapshot(f.root,self.selected)
        f.protocol['evaluation_budget'].update(max_model_calls=12,max_total_generated_tokens=6144,max_task_seconds=600)
        self.patch=patch.object(replay,'encode_prefix',return_value=(list(range(20)),{}));self.patch.start();self.addCleanup(self.patch.stop)

    def collect(self,arm,actions,out=None):
        f=self.f;out=out or f.root/('new-'+arm);self.h.actions=actions[:]
        t=fc.initialize(f.root,f.task,f.protocol,out,arm,self.selected)
        rr.advance(f.root,f.task,f.protocol,out,t,'socket',self.h.runner,arm)
        r=rr.finish(f.data,f.task,f.source,f.protocol,out,t,self.h.runner)
        return out,t,r

    def replay(self,out,t):return replay.replay(out,t,self.f.task,self.f.protocol,self.h.profile,self.h,self.f.normalizer,self.prefix)

    def test_one_new_correct_edit_qualifies_after_real_historical_failure(self):
        out,t,r=self.collect('correction',['edit-source','python3 check.py',SUBMIT])
        self.assertEqual(t['attempted_calls'],3);self.assertEqual(t['historical_prefix_cost']['historical_calls'],2)
        self.assertTrue(fc.behavior(t,self.f.task)['qualified_behavior']);self.assertTrue(self.replay(out,t)['passed'])
        self.assertIn('ACTUAL_FAILURE',t['messages'][5]['content'])
        self.assertNotIn('HIDDEN_CASE_SENTINEL',json.dumps(t['messages']))

    def test_both_arms_restore_identical_files_and_have_identical_hard_caps(self):
        receipts=[]
        for arm in ['control','correction']:
            out,t,r=self.collect(arm,['cat logic.py']*12)
            receipts.append(json.loads((out/'restored-state.json').read_text()))
            self.assertEqual(t['attempted_calls'],12);self.assertEqual(t['terminal_reason'],'call_limit')
            self.assertTrue(self.replay(out,t)['passed']);self.assertFalse(fc.behavior(t,self.f.task)['qualified_behavior'])
        self.assertEqual(*receipts)

    def test_snapshot_tampering_stops_before_generation(self):
        (self.f.root/'admitted/snapshot/tree/logic.py').write_text('forged')
        with self.assertRaises(ValueError):fc.initialize(self.f.root,self.f.task,self.f.protocol,self.f.root/'never','control',self.selected)
        self.assertFalse((self.f.root/'never').exists())

    def test_replay_rejects_different_initial_failure_source(self):
        out,t,_=self.collect('control',['edit-source','python3 check.py',SUBMIT])
        prefix=copy.deepcopy(self.prefix);prefix['failed_source_sha256']={'logic.py':'0'*64}
        self.assertFalse(replay.replay(out,t,self.f.task,self.f.protocol,self.h.profile,self.h,self.f.normalizer,prefix)['passed'])

    def test_encoding_rejects_missing_failed_edit_or_confirmed_failure_and_truncation(self):
        out,t,_=self.collect('correction',['edit-source','python3 check.py',SUBMIT])
        with patch.object(audit,'encode_prefix',return_value=(list(range(20)),{})),patch.object(audit,'encode_training_example',return_value={'ids':list(range(20))+[888],'supervised_tokens':1}):
            for needle in ['bad-edit','CONTINUATION START']:
                def crop(messages,encode,cap,policy):
                    _,context=prepare_context(messages,lambda _:list(range(20)),cap,policy)
                    return encode([m for m in messages if needle not in m['content']]),context
                with patch.object(audit,'prepare_context',crop):
                    value,_=audit.encode_correction(out,t,self.f.task,self.f.protocol,self.h.profile,self.h,self.prefix,self.selected)
                self.assertFalse(value['qualified']);self.assertTrue(value['encoding_errors'])
            with patch.object(audit,'encode_training_example',return_value={'ids':list(range(9000)),'supervised_tokens':1}):
                value,_=audit.encode_correction(out,t,self.f.task,self.f.protocol,self.h.profile,self.h,self.prefix,self.selected)
            self.assertFalse(value['qualified'])

    def test_full_audit_counts_only_new_calls_and_exports_only_final_correction(self):
        f=self.f;p=f.protocol;run=f.root/'run'
        p.update(run_directory='run',data_directory='data',supervisor_receipt='supervisor.json',evaluation_profile_path='profile.json',
            environment_path='environment.json',frozen_files={},selected_states=[self.selected],known_corrective_task_ids=[],schedule=schedule(['fixture']),
            global_caps={'model_calls':24,'generated_tokens':12288},
            feasibility_gate={'minimum_qualified_unique_tasks':6,'minimum_qualified_families':3,'minimum_additional_corrective_tasks_beyond_027':3})
        def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        write(f.root/'protocol.json',p);write(f.root/'profile.json',self.h.profile);write(f.root/'private/parent-runtime.json',{'model':'fixture'})
        write(f.root/'environment.json',{'fixture':True});write(f.data/'tasks.json',{'tasks':[f.task]});write(f.data/'sources.json',{'sources':[f.source]})
        old=self.h.runner.execute
        def execute(root,command):
            if command=='second-wrong-edit':(root/'logic.py').write_text('answer = 6\n')
            return old(root,command)
        self.h.runner.execute=execute;results=[]
        for arm,actions in [('control',['edit-source','python3 check.py',SUBMIT]),('correction',['second-wrong-edit','python3 check.py','edit-source','python3 check.py',SUBMIT])]:
            rid='fixture--'+arm;out,t,r=self.collect(arm,actions,run/'private/tasks'/rid)
            r.update(run_id=rid,arm=arm,state_id='state-001',correction_behavior=fc.behavior(t,f.task));write(out/'result.json',r);results.append(r)
        write(run/'controller.json',{'complete':True,'backend_stopped':True,'runs':results,'protocol_sha256':digest((f.root/'protocol.json').read_bytes()),'backend_lifecycles':[{'index':1,'role':'base'}]})
        write(run/'private/backends/0001-base-environment.json',{'fixture':True});write(f.root/'supervisor.json',{'complete':True,'timed_out':False,'worker_returncode':0})
        tm=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args,**kw:self.h))
        with patch.dict(sys.modules,{'transformers':tm}),patch.object(audit,'normalization',return_value=f.normalizer),patch.object(audit,'encode_prefix',return_value=(list(range(20)),{})),patch.object(audit,'encode_training_example',return_value={'ids':list(range(20))+[888],'supervised_tokens':1}):
            value=audit.run(f.root,f.root/'protocol.json',run/'audit.json')
        self.assertTrue(value['passed'],value)
        records=[json.loads(line) for line in (run/'private/correction-export/data.jsonl').read_text().splitlines()]
        self.assertEqual([r['provenance']['call'] for r in records],[3,4,5]);self.assertEqual(sum(r['attempted_calls'] for r in value['attempts']),8)
        self.assertIn('ACTUAL_FAILURE',json.dumps(records[0]['messages']));self.assertNotIn('bad-edit',records[0]['target'])
        self.assertNotIn('HIDDEN_CASE_SENTINEL',json.dumps(records))
        summary=summarize(value,p);self.assertFalse(summary['data_feasibility_gate_passed'])
        self.assertEqual(summary['actual_new_cost']['attempted_calls'],8)
        self.assertEqual(summary['historical_prefix_cost_reused_once_per_task']['historical_calls'],2)


if __name__=='__main__':unittest.main()
