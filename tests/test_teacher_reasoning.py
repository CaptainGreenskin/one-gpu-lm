"""Thought boundaries must control execution, cost accounting and replay."""
import copy
import json
import unittest
from unittest.mock import patch
from one_gpu_lm.common.reasoning import settings, separate
from one_gpu_lm.verified import correction_rollout as rr, decision_policy, failure_audit as audit
from one_gpu_lm.verified.failure_continuation import initial_messages
from tests import test_failure_continuation as fixture
from one_gpu_lm.verified.submission import SUBMIT


class Tokenizer:
    def encode(self,text):
        result=[]
        while text:
            for marker,token in [('<think>',8),('</think>',9)]:
                if text.startswith(marker):result.append(token);text=text[len(marker):];break
            else:result.append(ord(text[0])+1000);text=text[1:]
        return result
    def decode(self,ids,**kwargs):return ''.join({8:'<think>',9:'</think>'}.get(i,chr(i-1000) if i>=1000 else '') for i in ids)


def profile():
    return {'chat_template_kwargs':{'enable_thinking':False},'generation':{'max_new_tokens':2048},
        'generation_modes':{'direct':{'enable_thinking':False},'low':{'enable_thinking':True,'reasoning_effort':'low'}},
        'reasoning_markers':{'opening_token_id':8,'closing_token_id':9}}


class TeacherReasoning(unittest.TestCase):
    def setUp(self):self.tok=Tokenizer();self.profile=profile()
    def split(self,text,mode='low'):return separate(self.tok,self.tok.encode(text),self.profile,{'generation_mode':mode})
    def test_only_final_action_survives_thought_commands(self):
        text='Maybe ```bash\nbad-edit\n``` or <tool_call>fake</tool_call> </think>```bash\npwd\n```'
        value=self.split(text)
        self.assertTrue(value['valid']);self.assertEqual(value['action_text'],'```bash\npwd\n```')
        self.assertEqual(value['thought_tokens']+value['action_tokens'],len(self.tok.encode(text)))
    def test_unfinished_or_ambiguous_thought_executes_nothing(self):
        for text in ['```bash\nbad-edit\n```','reason </think> x </think> y','<think>reason </think> action']:
            value=self.split(text);self.assertFalse(value['valid']);self.assertEqual(value['action_text'],'')
            self.assertEqual(value['thought_tokens'],len(self.tok.encode(text)))
    def test_direct_mode_cannot_smuggle_thoughts(self):
        self.assertFalse(self.split('<think>bad</think>```bash\npwd\n```','direct')['valid'])
        self.assertTrue(self.split('```bash\npwd\n```','direct')['valid'])
    def test_profile_ceiling_and_unknown_mode_fail_closed(self):
        self.assertEqual(settings(self.profile,{'generation_mode':'low'})[1],2048)
        with self.assertRaises(ValueError):settings(self.profile,{'generation_mode':'unknown'})
        with self.assertRaises(ValueError):settings(self.profile,{})
        self.assertEqual(settings({'chat_template_kwargs':{}},{})[1],512)
        with self.assertRaises(ValueError):settings({'chat_template_kwargs':{}},{'generation_mode':'low'})
    def fixture(self):
        f=fixture.FailureContinuation();f.setUp();self.addCleanup(f.doCleanups)
        f.f.protocol.update(continuation_policy=decision_policy.REASONING_VERSION,reasoning_comparison={'arms':{'control':'direct','correction':'low'}})
        f.h.profile=self.profile;f.h.decode=self.tok.decode
        return f
    def generator(self,f,thought):
        original=f.h.request
        def generate(sock,payload,**kwargs):
            g=original(sock,payload,**kwargs)
            if payload['operation']=='health':return g
            text=thought+g['text'];ids=self.tok.encode(text)
            g.update(text=text,ids=ids,generated_tokens=len(ids),separation=separate(self.tok,ids,self.profile,payload))
            return g
        return generate
    def test_both_arms_have_identical_visible_initial_guidance(self):
        f=self.fixture();self.assertEqual(initial_messages(f.prefix,f.f.protocol,'control'),initial_messages(f.prefix,f.f.protocol,'correction'))
    def test_real_execution_replay_and_all_thought_costs(self):
        f=self.fixture()
        with patch.object(rr,'request',self.generator(f,'Do not execute ```bash\nbad-edit\n``` </think>')):
            out,t,r=f.collect('correction',['edit-source','python3 check.py',SUBMIT])
        self.assertTrue(r['success_within_budget']);self.assertTrue(f.replay(out,t)['passed'])
        self.assertEqual(t['generated_tokens'],sum(len(s['generation']['ids']) for s in t['steps']))
        self.assertNotIn('Do not execute',json.dumps(t['messages']))
        t['steps'][0]['generation']['separation']['action_text']='```bash\nbad-edit\n```'
        self.assertFalse(f.replay(out,t)['passed'])
    def test_student_export_changes_template_and_never_labels_thoughts(self):
        f=self.fixture();f.f.protocol['reasoning_comparison']['student_profile_sha256']='student-hash'
        with patch.object(rr,'request',self.generator(f,'PRIVATE_THOUGHT </think>')):
            out,t,_=f.collect('correction',['edit-source','python3 check.py',SUBMIT])
        student={'chat_template_kwargs':{'enable_thinking':False}}
        def encode(tokenizer,messages,kwargs,fmt):return (list(range(20)) if kwargs.get('enable_thinking') else [90,91],{})
        with patch.object(audit,'encode_prefix',encode),patch.object(audit,'encode_training_example',return_value={'ids':[90,91,888],'supervised_tokens':1}):
            value,rows=audit.encode_correction(out,t,f.f.task,f.f.protocol,self.profile,self.tok,f.prefix,f.selected,student)
        self.assertTrue(value['qualified'],value)
        self.assertNotIn('PRIVATE_THOUGHT',json.dumps(rows))
        self.assertEqual(len(rows),3)
        self.assertNotEqual(rows[0]['encoding_receipt']['teacher_input_sha256'],rows[0]['encoding_receipt']['student_input_sha256'])

    def test_unfinished_thought_uses_call_and_tokens_without_source_change(self):
        f=self.fixture();original=self.generator(f,'Unfinished ')
        with patch.object(rr,'request',original):out,t,r=f.collect('correction',['bad-edit']*12)
        self.assertEqual(t['attempted_calls'],12);self.assertGreater(t['generated_tokens'],24)
        self.assertTrue(all('observation' not in s for s in t['steps']));self.assertTrue(f.replay(out,t)['passed'])
        self.assertEqual(t['final_source_sha256'],f.prefix['failed_source_sha256'])
