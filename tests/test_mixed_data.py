import copy
import hashlib
import json
import unittest
from unittest.mock import patch
from one_gpu_lm.verified import mixed_data as data


class MixedAdmission(unittest.TestCase):
    def setUp(self):
        self.profile={'training':{'max_sequence_length':8}}
        self.protocol={'reasoning_comparison':{'student_profile_sha256':'student-profile'}}
        self.sources=[{'repository':'fixture/repo','split':'training','approved_for':{'training':True}}]
        prefix_hash=hashlib.sha256(json.dumps([1,2],separators=(',',':')).encode()).hexdigest()
        receipts=[{'call':i,'kind':k,'tokens':4,'supervised_tokens':2,'student_input_sha256':prefix_hash,
                   'student_profile_sha256':'student-profile','export_mode':'action_only_cross_mode'} for i,k in enumerate(['edit','verify','submit'],3)]
        self.audit={'passed':True,'rows':[{'task_id':'task','arm':'correction','qualified':True,'state_id':'state',
            'result':{'repository':'fixture/repo','trajectory_sha256':'trace'},'encoding':{'encoded_actions':receipts,'target_start_call':3}}]}
        self.manifest={'action_records':3,'qualified_unique_tasks':1,'supervised_tokens':6}
        self.rows=[{'id':'new'+str(x['call']),'target':'```bash\npwd\n```','messages':[],
            'provenance':{'task_id':'task','call':x['call'],'kind':x['kind'],'split':'train','arm':'correction','producer_kind':'model',
            'model_role':'base','executed':True,'repository':'fixture/repo','trajectory_sha256':'trace','state_id':'state',
            'target_start_call':3,'target_transform':'verified_final_action_only_no_thought_labels'},'encoding_receipt':copy.deepcopy(x)} for x in receipts]
        p=patch.object(data,'summarize',return_value={'data_feasibility_gate_passed':True});p.start();self.addCleanup(p.stop)
    @staticmethod
    def encode(*args):return {'ids':[1,2,3,4],'labels':[-100,-100,3,4],'prefix_tokens':2,'supervised_tokens':2}
    def select(self):return data.select_fragments(self.rows,self.manifest,self.audit,self.protocol,self.sources,self.profile,None,self.encode)
    def test_all_actions_preserved_as_fragments(self):
        rows=self.select();self.assertEqual(len(rows),3);self.assertEqual({r['record_group_kind'] for r in rows},{'corrective_fragment'})
    def test_overlap_does_not_inflate_unique_tasks_or_whole_episodes(self):
        new=self.select();old=[dict(new[0],id='old',episode_id='whole-episode')]
        mixed,meta=data.combine(old,new)
        self.assertEqual(meta['unique_tasks'],1);self.assertEqual(meta['full_episodes'],1);self.assertEqual(meta['corrective_fragments'],1)
        self.assertEqual(meta['overlapping_task_ids'],['task']);self.assertEqual(len(mixed),4)
        self.assertEqual(data.combine(old,list(reversed(new)))[0],mixed)
    def test_missing_or_duplicate_action_is_not_silently_filtered(self):
        original=copy.deepcopy(self.rows);self.rows.pop()
        with self.assertRaises(ValueError):self.select()
        self.rows=original+[copy.deepcopy(original[0])];self.rows[-1]['id']='new-id'
        with self.assertRaises(ValueError):self.select()
    def test_wrong_lineage_unexecuted_and_heldout_targets_rejected(self):
        for key,value in [('trajectory_sha256','wrong'),('executed',False),('split','development'),('target_start_call',2)]:
            original=self.rows[0]['provenance'][key];self.rows[0]['provenance'][key]=value
            with self.assertRaises(ValueError):self.select()
            self.rows[0]['provenance'][key]=original
    def test_thoughts_or_encoding_changes_rejected(self):
        self.rows[0]['target']='<think>hidden</think>'+self.rows[0]['target']
        with self.assertRaises(ValueError):self.select()
        self.rows[0]['target']='```bash\npwd\n```';self.profile['training']['max_sequence_length']=3
        with self.assertRaises(ValueError):self.select()
        self.profile['training']['max_sequence_length']=8
        self.rows[0]['encoding_receipt']['student_input_sha256']='forged'
        with self.assertRaises(ValueError):self.select()
    def test_failed_gate_or_unapproved_source_stops_training_admission(self):
        with patch.object(data,'summarize',return_value={'data_feasibility_gate_passed':False}):
            with self.assertRaises(ValueError):self.select()
        self.sources[0]['approved_for']['training']=False
        with self.assertRaises(ValueError):self.select()
    def test_combining_duplicate_record_ids_is_rejected(self):
        new=self.select()
        with self.assertRaises(ValueError):data.combine(new,new)


if __name__=='__main__':unittest.main()
