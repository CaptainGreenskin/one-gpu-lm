import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.serving.receipt_client import request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.training.decision_preference import EncodedDecision, token_digest
from one_gpu_lm.verified.decision_training_data import verify_first_choices


class FirstActionPreferenceInputs(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.out = Path(scratch.name)
        self.journal = RequestJournal(self.out/'private/backend/requests')
        self.design = {'sampling': {'seed_offset': 100, 'temperature': 0.7, 'top_p': 0.9}}
        source = {'implementation.py':'d'*64}
        payload = {'messages': [{'role':'user','content':'Repair the arithmetic regression.'}],
                   'max_input_tokens':8192,'max_new_tokens':512}
        prefix = (2,3,17)
        self.choice = EncodedDecision(prefix+(5,7), prefix+(12,), len(prefix), 'early_edit')
        actor = {'task_id':'fixture-task','ordinal':1,'prompt_tokens':len(prefix),'prompt_ids_sha256':token_digest(prefix),
                 'source_sha256':source,'messages':payload['messages'],
                 'budget':{'max_total_generated_tokens':1200,'max_input_tokens':8192,'max_generated_tokens_per_call':512}}
        self.plan = {'rows':[actor]}
        self.record = {'provenance':{'task_id':actor['task_id'],'fork_call':1,'remaining_generated_tokens':1200,
                                     'chosen_arm':'sampled','rejected_arm':'greedy'}}
        self.runs = []
        for label,arm in (('chosen','sampled'),('rejected','greedy')):
            name = f'01-{arm}'
            directory = self.out/'private/tasks'/name
            directory.mkdir(parents=True)
            request = directory/'request-001.json'
            request.write_text(json.dumps(payload))
            self.record[label] = f'echo {arm}'
            self.record['provenance'][label+'_request_sha256'] = sha256(request)
            generation = {'input_tokens_sha256':actor['prompt_ids_sha256'],'text':self.record[label]}
            if arm=='sampled':
                generation['first_action_sampling'] = {'seed':101,'temperature':0.7,'top_p':0.9}
            reply = self.journal.execute(request_identity(name,1),payload,lambda record: generation)
            self.assertEqual(reply,generation)
            step = {'call':1,'request_sha256':sha256(request),'source_before':source,'generation':generation,
                    'action_format_valid':True,'observation':{'executed':True}}
            trace = directory/'trajectory.json'
            trace.write_text(json.dumps({'steps':[step],'eligible_verified_trajectory':label=='chosen'}))
            self.runs.append({'run_id':name,'sampling_arm':arm,'ordinal':1,'trajectory_sha256':sha256(trace),
                              'complete_success_within_budget':label=='chosen'})

    def verify(self):
        return verify_first_choices(self.out,self.plan,self.design,self.runs,[self.record],[self.choice])

    def test_rechecks_both_physical_labels(self):
        self.assertEqual(self.verify(),2)

    def test_refuses_a_label_with_changed_action_or_prompt(self):
        original = self.record['chosen']
        self.record['chosen'] = 'echo different'
        with self.assertRaisesRegex(ValueError,'audited physical first action'):
            self.verify()
        self.record['chosen'] = original
        self.plan['rows'][0]['prompt_ids_sha256'] = 'a'*64
        with self.assertRaisesRegex(ValueError,'physical first-decision prefix'):
            self.verify()

    def test_refuses_later_action_or_changed_budget(self):
        self.record['provenance']['fork_call'] = 2
        with self.assertRaisesRegex(ValueError,'physical first-decision prefix'):
            self.verify()
        self.record['provenance'].update(fork_call=1,remaining_generated_tokens=1199)
        with self.assertRaisesRegex(ValueError,'physical first-decision prefix'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
