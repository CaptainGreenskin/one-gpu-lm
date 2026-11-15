import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tests import test_verified_evaluation as fixtures
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.verified import evaluation_audit as audit
from one_gpu_lm.verified.tasks import digest, prepare
from one_gpu_lm.verified.submission import SUBMIT


class RecordedInputAudit(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.EvaluationIsolation();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.c=self.fixture.c;self.decoded={};self.profile={'chat_template_kwargs':{'enable_thinking':False}}
        c=self.c;c.actions=['edit-source','python3 check.py',SUBMIT]
        def request(socket,payload,**kwargs):
            if payload['operation']=='health':return c.health
            c.requests.append(copy.deepcopy(payload));call=len(c.requests)
            text='```bash\n'+c.actions[call-1]+'\n```';ids=[call,777];self.decoded[tuple(ids)]=text
            tokens,context=prepare_context(payload['messages'],lambda _:list(range(20)),8192,'bounded_history_v1')
            return {'text':text,'ids':ids,'generated_tokens':2,'input_tokens':len(tokens),
                    'input_tokens_sha256':digest(json.dumps(tokens,separators=(',',':')).encode()),
                    'context':context,'adapter':c.health['adapter']}
        c.request=request
    def decode(self,ids,**kwargs):return self.decoded[tuple(ids)]
    def recorded(self,arm='base_current'):
        self.fixture.evaluate(arm)
        return json.loads((self.c.root/'out/trajectory.json').read_text())
    def check(self,trace):
        with patch.object(audit,'encode_prefix',return_value=(list(range(20)),{})):
            return audit.replay(self.c.root/'out',trace,self.c.task,self.c.protocol,self.profile,self,self.c.normalizer)
    def test_all_actual_requests_and_feedback_reconstruct(self):
        trace=self.recorded();value=self.check(trace)
        self.assertTrue(value['passed'],value)
        self.assertEqual(value['known_generated_tokens'],6);self.assertEqual(value['known_input_tokens'],60)
    def test_budget_tampering_detected_even_if_request_digest_is_rewritten(self):
        trace=self.recorded();path=self.c.root/'out/request-002.json';value=json.loads(path.read_text());value['max_new_tokens']+=1
        path.write_text(json.dumps(value)+'\n');trace['steps'][1]['request_sha256']=digest(path.read_bytes())
        self.assertFalse(self.check(trace)['passed'])
    def test_changed_raw_feedback_cannot_hide_behind_unchanged_token_ids(self):
        trace=self.recorded();path=self.c.root/'out/tool-001.json';value=json.loads(path.read_text());value['output']='Different evidence'
        path.write_text(json.dumps(value)+'\n')
        self.assertFalse(self.check(trace)['passed'])
    def test_guard_rejection_reconstructs_but_missing_executed_flag_fails(self):
        self.c.actions=['edit-source',SUBMIT,'python3 check.py'];trace=self.recorded('base_guard')
        self.assertTrue(self.check(trace)['passed'])
        path=self.c.root/'out/tool-003.json';value=json.loads(path.read_text());value.pop('executed');path.write_text(json.dumps(value)+'\n')
        trace['steps'][2]['observation'].pop('executed')
        self.assertFalse(self.check(trace)['passed'])
    def test_recorded_grader_input_checked_without_running_commands(self):
        trace=self.recorded();c=self.c
        before=prepare(c.snapshot,c.root/'audit-broken',c.source,c.task)
        value=audit.recorded_grade(c.root,c.data,c.root/'out',c.task,c.source,before,c.root/'audit-grader')
        self.assertEqual(value,trace['grade'])
        (c.root/'out/grader/hidden.py').write_text('tampered hidden check')
        with self.assertRaises(ValueError):
            audit.recorded_grade(c.root,c.data,c.root/'out',c.task,c.source,before,c.root/'audit-grader-2')


if __name__=='__main__':unittest.main()
