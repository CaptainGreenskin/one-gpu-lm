import unittest
from one_gpu_lm.verified.historic_outcomes import executed_completion
from one_gpu_lm.verified.submission import SUBMIT


class CompletionReceipts(unittest.TestCase):
    def test_submit_requires_actual_successful_last_tool_action(self):
        trace={'terminal_reason':'submitted','steps':[{'command':SUBMIT,'observation':{'returncode':1}}]}
        self.assertFalse(executed_completion(trace,{'regression_command':'test'},{})['submitted'])
        trace['steps'][0]['observation']['returncode']=0
        self.assertTrue(executed_completion(trace,{'regression_command':'test'},{})['submitted'])
        trace['steps'].append({'command':'edit','observation':{'returncode':0}})
        self.assertFalse(executed_completion(trace,{'regression_command':'test'},{})['submitted'])

    def test_combined_test_or_post_test_change_is_not_final_verification(self):
        trace={'terminal_reason':'call_limit','steps':[{'command':'edit && test','observation':{'returncode':0},'source_after':{'a':'fixed'}}]}
        self.assertFalse(executed_completion(trace,{'regression_command':'test'},{'a':'fixed'})['agent_verified_final_source'])
        trace['steps'][0]['command']='test'
        self.assertTrue(executed_completion(trace,{'regression_command':'test'},{'a':'fixed'})['agent_verified_final_source'])
        self.assertFalse(executed_completion(trace,{'regression_command':'test'},{'a':'changed'})['agent_verified_final_source'])


if __name__=='__main__':unittest.main()
