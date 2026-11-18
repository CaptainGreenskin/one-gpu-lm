import copy
import hashlib
import json
import unittest
from one_gpu_lm.verified.multifile_diagnosis import diagnose


class DiagnosisTests(unittest.TestCase):
    def make_case(self):
        raw='x'*2500;prefix=raw[:2400]+'\n[tool output truncated; use smaller file ranges]'
        steps=[{'call':i,'source_before':{'a':'old'},'source_after':{'a':'new' if i==12 else 'old'},
                'observation':{'returncode':0},'normalization':{'raw_sha256':hashlib.sha256(prefix.encode()).hexdigest()}}
               for i in range(1,13)]
        trace={'id':'fixture','split':'development','training_export_allowed':False,'complete':True,
               'steps':steps,'attempted_calls':12,'submitted':False,'grade':{'passed':True},
               'agent_verified_final_source':False,'generated_tokens':20}
        h=hashlib.sha256((json.dumps(trace,indent=2,ensure_ascii=False)+'\n').encode()).hexdigest()
        bundles={'r0-fixture-mixed031':{'trace':trace,'trajectory_sha256':h,
                 'tools':{f'tool-{i:03d}':{'output':raw} for i in range(1,13)}}}
        audit={'passed':True,'rows':[{'run_id':'r0-fixture-mixed031','variant':'mixed031','trajectory_sha256':h,
               'audit_passed':True,'attempted_calls':12,'submitted':False,'grader_passed':True,'input_tokens':100}]}
        return bundles,audit

    def test_late_correct_patch_is_not_claimed_to_have_unused_budget(self):
        bundles,audit=self.make_case();result=diagnose(bundles,audit)
        row=result['rows'][0]
        self.assertEqual(row['category'],'first_edit_on_final_call_no_completion_budget')
        self.assertEqual(row['remaining_calls_after_first_edit'],0)
        self.assertEqual(row['clipped_tool_output_calls'],list(range(1,13)))
        self.assertEqual(result['development_examples_added_to_training'],0)

    def test_changed_trace_and_tool_prefix_are_rejected(self):
        for kind in ['trace','tool']:
            bundles,audit=self.make_case();b=next(iter(bundles.values()))
            if kind=='trace':b['trace']['submitted']=True
            else:b['tools']['tool-001']['output']='different'
            with self.subTest(kind=kind),self.assertRaises(ValueError):diagnose(bundles,audit)


if __name__=='__main__':unittest.main()
