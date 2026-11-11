import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('release_report', Path(__file__).resolve().parents[1]/'experiments/019-coding-release/report.py')
report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)


class ReleaseReportTests(unittest.TestCase):
    def fixture(self):
        policy = {'format':'native_roundtrip','context':'full','progress':'edit_checkpoint_v1'}
        protocol = {'task_spec_sha256':'tasks','admission_sha256':'admission',
            'frozen_files':{'code.py':'code'},'variants':{'shared':policy},
            'agent':{'max_model_calls':12,'max_total_generated_tokens':6144,'max_input_tokens':8192,'max_generated_tokens_per_call':512}}
        generation = {'adapter':False,'agent_policy':policy,'input_tokens':200,'generated_tokens':20,
            'context':{'policy':'full','dropped_rounds':0},'action_format_valid':True,'seconds':1,'hit_generation_limit':False}
        row = {'id':'t','passed':False,'agent_policy':policy,'variant':'shared','generations':[generation],
            'backend_errors':[],'calls':1,'generated_tokens':20,'seconds':2,'tool_seconds':.5,
            'exit_status':'Submitted','source_progress':[{'changed_editable_files':1}]}
        campaign = {'complete':True,'identity':{'ids':['t'],'tasks_sha256':'tasks','admission_sha256':'admission',
            'protocol_sha256':'protocol','variant':'shared','source_files':{'code.py':'code'}},'results':[row]}
        return campaign, protocol

    def test_observed_edit_does_not_become_a_success(self):
        campaign, protocol = self.fixture()
        value = report.validate_campaign(campaign,['t'],protocol,'protocol',False)
        self.assertEqual(value['passed'],0)
        self.assertEqual(value['failed_after_edit'],1)

    def test_missing_pair_wrong_adapter_and_changed_code_fail_closed(self):
        campaign, protocol = self.fixture()
        for alter in [lambda c:c['results'].append(copy.deepcopy(c['results'][0])),
                      lambda c:c['results'][0]['generations'][0].update(adapter=True),
                      lambda c:c['identity']['source_files'].update({'code.py':'new-code'})]:
            changed = copy.deepcopy(campaign);alter(changed)
            with self.assertRaises(ValueError):report.validate_campaign(changed,['t'],protocol,'protocol',False)

    def test_budget_and_context_changes_are_not_hidden(self):
        campaign, protocol = self.fixture()
        for alter in [lambda c:c['results'][0].update(calls=13),
                      lambda c:c['results'][0].update(generated_tokens=21),
                      lambda c:c['results'][0]['generations'][0].update(input_tokens=8193),
                      lambda c:c['results'][0]['generations'][0]['context'].update(dropped_rounds=1)]:
            changed = copy.deepcopy(campaign);alter(changed)
            with self.assertRaises(ValueError):report.validate_campaign(changed,['t'],protocol,'protocol',False)


if __name__ == '__main__':unittest.main()
