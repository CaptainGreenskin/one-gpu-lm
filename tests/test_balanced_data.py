import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from one_gpu_lm.data.balanced import choose_balanced, prepare
from one_gpu_lm.common.receipts import sha256, write_json


class BalancedSelection(unittest.TestCase):
    def test_native_metadata_timings_cannot_disguise_duplicate_model_inputs(self):
        from tests.test_training_encoding import RecordingTokenizer, TrainingEncodingContracts
        import copy
        fixture=TrainingEncodingContracts();fixture.setUp()
        profile=fixture.profile
        profile['training'].update(dialogue_format='native_roundtrip',max_sequence_length=10000)
        profile['generation']={'max_new_tokens':512}
        first=copy.deepcopy(fixture.row);second=copy.deepcopy(fixture.row)
        first.update(id='one',provenance={'task_id':'a','kind':'verified-reference-repair'})
        second.update(id='two',provenance={'task_id':'a','kind':'verified-reference-repair'})
        first['messages'][-1]['extra']['tool']['seconds']=0.5
        second['messages'][-1]['extra']['tool']['seconds']=1.5
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths={'data':root/'raw','data_receipt':root/'receipt'}
            paths['data'].write_text('private');write_json(paths['data_receipt'],{'records':2})
            with patch('one_gpu_lm.data.balanced.read_training_data',return_value=([first,second],{'a'})):
                audit=prepare(paths,profile,RecordingTokenizer(),512,root/'selected',up_to=True)
            self.assertEqual(audit['selected_records'],1)
            self.assertEqual(audit['excluded'][0]['reason'],'duplicate_context_and_target')
            self.assertEqual(audit['deduplication'],'model-visible-token-ids')

    @staticmethod
    def records(counts):
        return [{'id':f'{task}:{i}','task_id':task,'source_index':i}
                for task,n in counts.items() for i in range(n)]

    def test_reference_only_tasks_are_not_crowded_out(self):
        rows=self.records({'easy-a':12,'easy-b':12,'hard-reference':1})
        chosen,quotas=choose_balanced(rows,9,set(r['task_id'] for r in rows))
        self.assertEqual(quotas,{'easy-a':4,'easy-b':4,'hard-reference':1})
        self.assertEqual(len(chosen),9)
        self.assertIn('hard-reference:0',[r['id'] for r in chosen])

    def test_quota_covers_start_middle_and_end_of_trajectory(self):
        rows=self.records({'task':11})
        chosen,_=choose_balanced(rows,3,{'task'})
        self.assertEqual([r['source_index'] for r in chosen],[0,5,10])

    def test_missing_tasks_and_insufficient_examples_fail_before_training(self):
        rows=self.records({'a':2,'b':1})
        for budget,required in [(1,{'a','b'}),(4,{'a','b'}),(3,{'a','b','missing'})]:
            with self.assertRaises(ValueError):choose_balanced(rows,budget,required)

    def test_deterministic_order_and_no_duplicate_slots(self):
        rows=self.records({'c':9,'a':3,'b':7})
        first,quotas=choose_balanced(rows,13,{'a','b','c'})
        second,_=choose_balanced(list(reversed(rows)),13,{'a','b','c'})
        self.assertEqual(first,second)
        self.assertEqual(len({r['id'] for r in first}),13)
        self.assertEqual(quotas,{'a':3,'b':5,'c':5})

    def test_preparation_filters_duplicates_and_unexecutable_targets_without_exporting_text(self):
        class Tokenizer:
            eos_token='!'
            def apply_chat_template(self,*args,**kwargs):return [1,2]
            def encode(self,text,**kwargs):return list(text)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths={'data':root/'raw.jsonl','data_receipt':root/'receipt.json'}
            paths['data'].write_text('private input');write_json(paths['data_receipt'],{'records':4})
            rows=[{'id':str(i),'messages':[{'role':'user','content':'PRIVATE_PROMPT'}],
                   'target':target,'provenance':{'task_id':'a','kind':'verified-reference-repair'}}
                  for i,target in enumerate(['ok','ok','too-long','yes'])]
            profile={'generation':{'max_new_tokens':4},'training':{'max_sequence_length':8},'chat_template_kwargs':{}}
            with patch('one_gpu_lm.data.balanced.read_training_data',return_value=(rows,{'a'})):
                audit=prepare(paths,profile,Tokenizer(),2,root/'prepared')
            self.assertEqual({r['reason'] for r in audit['excluded']},{'duplicate_context_and_target','context_limit'})
            self.assertEqual(audit['selected_records'],2)
            self.assertNotIn('PRIVATE_PROMPT',json.dumps(audit))
            self.assertEqual(audit['data_sha256'],sha256(root/'prepared/train.jsonl'))
            self.assertEqual([json.loads(s)['id'] for s in (root/'prepared/train.jsonl').read_text().splitlines()],['0','3'])


if __name__=='__main__':unittest.main()
