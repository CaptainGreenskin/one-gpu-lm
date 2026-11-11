import copy
import json
import unittest

from one_gpu_lm.common.agent_policy import NATIVE_EXAMPLE
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.training_encoding import encode_training_example


class RecordingTokenizer:
    eos_token = '!'

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        # A deterministic stand-in checks boundaries/masks, not real-template parity.
        text = json.dumps(kwargs.get('tools', []), sort_keys=True)
        for m in messages:
            text += '<' + m['role'] + '>' + json.dumps({k:v for k,v in m.items() if k!='role'}, sort_keys=True) + '</>'
        if add_generation_prompt:
            text += '<assistant>'
        return self.encode(text)


class TrainingEncodingContracts(unittest.TestCase):
    def setUp(self):
        self.tokenizer = RecordingTokenizer()
        self.row = {'messages':[
            {'role':'system','content':'rules'}, {'role':'user','content':'task'},
            {'role':'assistant','content':NATIVE_EXAMPLE,'extra':{'actions':[parse_action(NATIVE_EXAMPLE)]}},
            {'role':'user','content':'Exit code: 0\nPRIVATE_RESULT','extra':{'tool':{'returncode':0}}}],
            'target':NATIVE_EXAMPLE.replace('pwd', 'printf done')}
        self.profile = {'chat_template_kwargs':{'enable_thinking':False},'training':{}}

    def test_legacy_record_ids_and_labels_are_unchanged(self):
        encoded = encode_training_example(self.row,self.tokenizer,self.profile)
        prefix = self.tokenizer.apply_chat_template(
            [{k:m[k] for k in ['role','content']} for m in self.row['messages']],
            add_generation_prompt=True)
        suffix = self.tokenizer.encode(self.row['target'] + '!')
        self.assertEqual(encoded['ids'],prefix+suffix)
        self.assertEqual(encoded['labels'],[-100]*len(prefix)+suffix)

    def test_native_serving_prefix_exact_and_all_observations_masked(self):
        self.profile['training']['dialogue_format']='native_roundtrip'
        original=copy.deepcopy(self.row)
        encoded=encode_training_example(self.row,self.tokenizer,self.profile)
        prefix,dialogue=encode_prefix(self.tokenizer,self.row['messages'],self.profile['chat_template_kwargs'],'native_roundtrip')
        self.assertEqual(encoded['ids'][:len(prefix)],prefix)
        self.assertEqual(encoded['labels'][:len(prefix)],[-100]*len(prefix))
        suffix=bytes(encoded['labels'][len(prefix):]).decode()
        self.assertIn('tool_calls',suffix)
        self.assertIn('printf done',suffix)
        self.assertNotIn('PRIVATE_RESULT',suffix)
        self.assertEqual(dialogue,{'structured_tool_calls':1,'tool_results':1})
        self.assertEqual(self.row,original)

    def test_template_boundary_mismatch_fails_closed(self):
        self.profile['training']['dialogue_format']='native_roundtrip'
        tokenizer=self.tokenizer
        original=tokenizer.apply_chat_template
        tokenizer.apply_chat_template=lambda *a,**kw: ([99]+original(*a,**kw) if not kw['add_generation_prompt'] else original(*a,**kw))
        with self.assertRaisesRegex(ValueError,'inference prefix'):
            encode_training_example(self.row,tokenizer,self.profile)

    def test_unmatched_executor_metadata_and_invalid_targets_are_rejected(self):
        self.profile['training']['dialogue_format']='native_roundtrip'
        broken=copy.deepcopy(self.row)
        broken['messages'][2]['extra']['actions']=[{'command':'not executed'}]
        with self.assertRaises(ValueError): encode_training_example(broken,self.tokenizer,self.profile)
        broken=copy.deepcopy(self.row);broken['target']='unexecuted prose'
        with self.assertRaises(ValueError): encode_training_example(broken,self.tokenizer,self.profile)


if __name__=='__main__':unittest.main()
