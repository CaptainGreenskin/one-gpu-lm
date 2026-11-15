import copy
import unittest
from one_gpu_lm.verified.candidate_data import select, windows


class WholeEpisodeAdmission(unittest.TestCase):
    def setUp(self):
        self.profile={'training':{'max_sequence_length':8}}
        self.spec={'training_data_gate':{'minimum_complete_unique_verified_episodes':1,
                   'maximum_complete_unique_verified_episodes':2,'minimum_per_training_repository':1,'minimum_languages':1}}
        self.sources=[{'repository':'fixture/repo','split':'training','language':'python','approved_for':{'training':True}}]
        actions=[{'call':i,'kind':kind,'tokens':4,'supervised_tokens':2} for i,kind in enumerate(['edit','verify','submit'],1)]
        self.audit={'decision':{'passed':True},'rows':[{'id':'task','eligible_for_sft':True,'trajectory_sha256':'trace','encoded_actions':actions}]}
        self.manifest={'episodes':[{'episode_id':'ep','task_id':'task','repository':'fixture/repo','trajectory_sha256':'trace','positive_action_records':3}],
                       'full_episodes':1,'unique_tasks':1,'positive_action_records':3}
        self.rows=[{'id':'record'+str(a['call']),'episode_id':'ep',
                    'provenance':{'task_id':'task','repository':'fixture/repo','trajectory_sha256':'trace','split':'train','executed':True,
                                  'producer_kind':'model','call':a['call'],'kind':a['kind']},
                    'encoding_receipt':{'tokens':4,'supervised_tokens':2}} for a in actions]
    @staticmethod
    def encode(*args):
        return {'ids':[1,2,3,4],'labels':[-100,-100,3,4],'supervised_tokens':2}
    def check(self):
        return select(self.rows,self.manifest,self.audit,self.sources,self.profile,self.spec,None,self.encode)
    def test_complete_episode_preserved_with_stable_order(self):
        value=self.check();self.rows.reverse()
        self.assertEqual([r['id'] for r in value],[r['id'] for r in self.check()])
        self.assertEqual({r['provenance']['kind'] for r in value},{'edit','verify','submit'})
    def test_one_long_action_rejects_dataset_instead_of_partial_episode(self):
        self.profile['training']['max_sequence_length']=3
        with self.assertRaises(ValueError):self.check()
    def test_omitted_or_duplicated_actions_rejected(self):
        original=copy.deepcopy(self.rows);self.rows.pop()
        with self.assertRaises(ValueError):self.check()
        self.rows=original+[copy.deepcopy(original[0])];self.rows[-1]['id']='different-id'
        with self.assertRaises(ValueError):self.check()
    def test_unexecuted_and_non_model_targets_rejected(self):
        self.rows[0]['provenance']['executed']=False
        with self.assertRaises(ValueError):self.check()
        self.rows[0]['provenance']['executed']=True;self.rows[0]['provenance']['producer_kind']='reference_restoration'
        with self.assertRaises(ValueError):self.check()
    def test_actions_never_count_as_unique_episodes(self):
        self.spec['training_data_gate']['minimum_complete_unique_verified_episodes']=2
        with self.assertRaises(ValueError):self.check()
    def test_unapproved_source_and_failed_audit_rejected(self):
        self.sources[0]['approved_for']['training']=False
        with self.assertRaises(ValueError):self.check()
        self.sources[0]['approved_for']['training']=True;self.audit['decision']['passed']=False
        with self.assertRaises(ValueError):self.check()
    def test_short_final_window_is_retained(self):
        self.assertEqual(list(windows(list(range(10)),8)),[list(range(8)),[8,9]])


class RegisteredRecipe(unittest.TestCase):
    def test_actual_profile_matches_scientific_rules_and_rejects_changed_rate(self):
        import json
        from pathlib import Path
        from one_gpu_lm.verified.candidate import validate_recipe
        root=Path(__file__).resolve().parents[1]/'experiments/025-verified-distillation'
        profile=json.loads((root/'candidate-profile.json').read_text())
        spec=json.loads((root/'cycle-spec.json').read_text())
        validate_recipe(profile,spec)
        profile['training']['learning_rate']*=2
        with self.assertRaises(ValueError):validate_recipe(profile,spec)


if __name__=='__main__':unittest.main()
