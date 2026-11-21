from copy import deepcopy
import unittest

from one_gpu_lm.common.runstate import identity_digest
from one_gpu_lm.verified.reload_probe import PROBES, compare, probe_requests
from one_gpu_lm.verified.runtime_identity import PACKAGES


class ReloadProbeTests(unittest.TestCase):
    def setUp(self):
        self.before = {
            'schema_version':1,'capture_id':'a'*32,
            'configuration':{'probe_definition_sha256':identity_digest(probe_requests()),'model_id':'fixture/test',
                             'base_revision':'1'*40,'profile_sha256':'2'*64,
                             'base_manifest_sha256':'3'*64,'runtime_sha256':'4'*64,
                             'software':{'python':'3.12.0','packages':dict.fromkeys(PACKAGES,'1.0')},
                             'adapter_files_sha256':{'adapter_config.json':'a'*64,'adapter_model.safetensors':'b'*64}},
            'probes':[{'name':name,'input_tokens':6,'generated_tokens':16,
                       'input_tokens_sha256':'c'*64,'generated_ids_sha256':'d'*64} for name,_ in PROBES],
        }
        self.after = deepcopy(self.before)
        self.after['capture_id'] = 'b'*32

    def test_probe_generation_ids_must_match_for_both_inputs(self):
        value = compare(self.before,self.after)
        self.assertTrue(value['save_reload_greedy_match'])
        self.assertEqual(value['compared_probes'],2)
        self.after['probes'][1]['generated_ids_sha256'] = 'e'*64
        value = compare(self.before,self.after)
        self.assertFalse(value['save_reload_greedy_match'])
        self.assertEqual(value['mismatched_probes'],[PROBES[1][0]])

    def test_shared_capture_and_changed_adapter_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'independent reload captures'):
            compare(self.before,self.before)
        self.after['configuration']['adapter_files_sha256']['adapter_model.safetensors'] = '0'*64
        with self.assertRaisesRegex(ValueError,'configuration changed'):
            compare(self.before,self.after)

    def test_matching_but_incomplete_model_metadata_is_rejected(self):
        self.before['configuration'].pop('profile_sha256')
        self.after['configuration'].pop('profile_sha256')
        with self.assertRaisesRegex(ValueError,'complete verified model configuration'):
            compare(self.before,self.after)

    def test_missing_probe_and_out_of_budget_fingerprint_are_rejected(self):
        self.after['probes'].pop()
        with self.assertRaisesRegex(ValueError,'complete fixed-probe'):
            compare(self.before,self.after)
        self.after = deepcopy(self.before)
        self.after['capture_id'] = 'b'*32
        self.after['probes'][0]['generated_tokens'] = 17
        with self.assertRaisesRegex(ValueError,'complete token fingerprints'):
            compare(self.before,self.after)

    def test_separate_adapter_profile_is_bound_to_both_reloads(self):
        for receipt in (self.before,self.after):
            receipt['schema_version'] = 2
            receipt['configuration']['adapter_profile_sha256'] = '7'*64
        self.assertTrue(compare(self.before,self.after)['save_reload_greedy_match'])
        self.after['configuration']['adapter_profile_sha256'] = '8'*64
        with self.assertRaisesRegex(ValueError,'configuration changed'):
            compare(self.before,self.after)
