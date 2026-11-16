import copy
import json
import unittest

from one_gpu_lm.verified.behavior_diagnosis import episode, inspection_shaped, verify_tool_receipts


class BehaviorDiagnosisTests(unittest.TestCase):
    def make_trace(self, commands):
        steps=[];tools={};source={'file.py':'a'}
        for call,(command,changed,rc) in enumerate(commands,1):
            after={'file.py':chr(ord('a')+call)} if changed else source.copy()
            steps.append({'call':call,'command':command,'action_format_valid':True,
                          'source_before':source.copy(),'source_after':after.copy(),
                          'generation':{'input_tokens':50,'generated_tokens':20,'hit_generation_limit':False,
                                        'context':{'dropped_rounds':0}},
                          'normalization':{'delivered_sha256':'same'}})
            tools[call]={'returncode':rc,'output':'PRIVATE_TOOL_TEXT'};source=after
        return {'complete':True,'attempted_calls':len(steps),'steps':steps},tools

    def test_initial_failing_check_is_not_repair_correction(self):
        t,tools=self.make_trace([('test',False,1),('write',True,0),('test',False,0)])
        r=episode(t,tools,'test',2400)
        self.assertEqual(r['counts']['edits_after_failed_exact_check'],1)
        self.assertNotIn('corrections_after_failed_exact_check_of_edited_source',r['counts'])

    def test_failed_edited_source_then_correction_and_privacy(self):
        t,tools=self.make_trace([('PRIVATE_PATCH_A',True,0),('test',False,1),('PRIVATE_PATCH_B',True,0),('test',False,0)])
        r=episode(t,tools,'test',2400)
        self.assertEqual(r['counts']['corrections_after_failed_exact_check_of_edited_source'],1)
        self.assertTrue(r['successful_exact_check_observed'])
        self.assertNotIn('PRIVATE_',json.dumps(r))

    def test_inspection_label_is_conservative(self):
        self.assertTrue(inspection_shaped('sed -n 10,20p src/file.py'))
        self.assertTrue(inspection_shaped('cat file.py; echo ---; grep -n foo file.py | head -20'))
        for command in ['sed -i s/a/b/ file.py','cat file.py > output','echo $(python mutate.py)',
                        'python repair.py','awk \'{system("touch marker")}\' file.py']:
            self.assertFalse(inspection_shaped(command),command)

    def test_repetition_requires_unchanged_source_and_feedback(self):
        t,tools=self.make_trace([('cat file.py',False,0),('cat file.py',False,0),('write',True,0),('cat file.py',False,0)])
        r=episode(t,tools,'test',2400)
        self.assertEqual(r['repeated_command_source_feedback_calls'],1)
        self.assertEqual(r['first_edit_call'],3)
        t['complete']=False
        with self.assertRaises(ValueError):episode(t,tools,'test',2400)

    def test_raw_tool_tamper_is_rejected(self):
        class Normalizer:
            MARKER='...'
            @staticmethod
            def transform(text,condition,cap):return text,{'raw':text}
        t,tools=self.make_trace([('cat file.py',False,0)])
        t['steps'][0]['observation']={'returncode':0}
        t['steps'][0]['normalization']={'raw':'PRIVATE_TOOL_TEXT'}
        verify_tool_receipts(t,tools,Normalizer,2400)
        changed=copy.deepcopy(tools);changed[1]['output']='ALTERED'
        with self.assertRaises(ValueError):verify_tool_receipts(t,changed,Normalizer,2400)


if __name__=='__main__':unittest.main()
