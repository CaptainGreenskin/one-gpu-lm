"""Post-hoc diagnostics must not confuse source changes with test metadata."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/023-evaluation-stability'
spec=importlib.util.spec_from_file_location('stability_residual_review',EXP/'residual_review.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
spec=importlib.util.spec_from_file_location('stability_report',EXP/'report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
RESULTS=EXP/'results/2026-11-13'

class ResidualFeedbackReview(unittest.TestCase):
    def test_complete_unittest_and_pytest_summaries(self):
        for a,b in [('Ran 4 tests in 0.021s','Ran 4 tests in 0.034s'),
                    ('===== 2 failed, 6 passed in 1.01s =====','===== 2 failed, 6 passed in 1.19s ====='),
                    ('1 failed in 0.02s','1 failed in 0.03s')]:
            self.assertEqual(review.classify(a,b)['candidate_class'],'test_duration_only')

    def test_changes_in_test_results_are_not_called_duration_only(self):
        a='===== 2 failed, 6 passed in 1.01s ====='
        b='===== 1 failed, 7 passed in 1.19s ====='
        self.assertEqual(review.classify(a,b)['candidate_class'],'unclassified_text_change')

    def test_dates_source_literals_and_task_timings_are_not_erased(self):
        for a,b in [('result = "in 0.021s"','result = "in 0.034s"'),
                    ('ERROR expired in 0.021s','ERROR expired in 0.034s'),
                    ('-rw-r--r-- 1 u g 4 Nov 12 09:35 x','-rw-r--r-- 1 u g 4 Nov 13 09:35 x')]:
            self.assertEqual(review.classify(a,b)['candidate_class'],'unclassified_text_change')

    def test_candidate_repr_addresses_and_combined_differences(self):
        a='<object at 0xab12>\nRan 1 test in 0.021s'
        self.assertEqual(review.classify(a,a.replace('ab12','ac23'))['candidate_class'],'object_repr_address_only')
        b=a.replace('ab12','ac23').replace('0.021','0.032')
        self.assertEqual(review.classify(a,b)['candidate_class'],'test_duration_and_object_repr_address')
        self.assertEqual(review.classify('0xab12','0xac23')['candidate_class'],'unclassified_text_change')

    def test_identical_feedback_is_not_a_residual_difference(self):
        with self.assertRaises(ValueError):review.classify('same','same')

@unittest.skipUnless(RESULTS.exists(),'023 is still running; no fabricated completed result fixture')
class PublishedStabilityEvidence(unittest.TestCase):
    def test_complete_actual_evidence_recomputes(self):
        value=report.assess(RESULTS);expected=json.loads((RESULTS/'summary.json').read_text())
        key='sampled_gpu_energy_kwh'
        self.assertAlmostEqual(value['hardware'].pop(key),expected['hardware'].pop(key),delta=1e-12)
        self.assertEqual(value,expected)
        self.assertEqual(len(value['cells']),18)
        self.assertEqual(sum(r['tasks'] for r in value['cells']),270)
        self.assertEqual((value['within_condition_pairs'],value['matched_condition_pairs']),(270,135))
        self.assertEqual(len(value['pairs']),405)
        self.assertEqual(len(value['task_stability']),90)
        self.assertEqual(len(value['probe_groups']),6)
        self.assertEqual(sum(r['calls'] for r in value['probe_groups']),36)
        self.assertFalse(value['final_test_opened']);self.assertEqual(value['training_updates'],0)
        self.assertFalse(value['weights_public']);self.assertTrue(value['time']['within_original_deadline'])
        self.assertEqual(len(value['outcome_flip_boundaries']),6)
        self.assertTrue(all(r['difference_is_only_ls_date_fields'] and r['same_command'] and r['same_prior_source'] and r['same_tool_status'] for r in value['outcome_flip_boundaries']))
        self.assertEqual(sum(r['repeated_input_groups'] for r in value['repeated_identical_input_summary']),402)
        self.assertEqual(sum(r['groups_with_different_output_text'] for r in value['repeated_identical_input_summary']),0)
        for r in value['stability']:
            self.assertEqual(r['pair_denominator'],45)
            self.assertEqual(r['pairwise_disagreements'],2*r['mixed_outcome_tasks'])
            self.assertEqual(r['mixed_outcome_tasks'],sum(t['mixed_outcome'] for t in value['task_stability'] if (t['model'],t['condition'])==(r['model'],r['condition'])))

    def test_rehashed_tampering_cannot_change_protocol_scores_or_scope(self):
        changes=[('controller.json',lambda v:v.update(final_test_opened=True)),
                 ('protocol.json',lambda v:v['agent'].update(max_model_calls=13)),
                 ('campaigns/r01-base-raw.json',lambda v:v['results'][0].update(passed=not v['results'][0]['passed'])),
                 ('artifacts/r01-base-attempt-1.json',lambda v:v.update(adapter_sha256={'adapter_model.safetensors':'0'*64})),
                 ('feedback-residual-review.json',lambda v:v.update(post_hoc=False)),
                 ('feedback-residual-review.json',lambda v:v.update(normalizer_changed=True)),
                 ('preregistration.json',lambda v:v.update(verified_utc='2099-01-01T00:00:00+00:00'))]
        for name,change in changes:
            with self.subTest(file=name),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'evidence';shutil.copytree(RESULTS,root)
                value=json.loads((root/name).read_text());change(value);report.write_json(root/name,value)
                collection=json.loads((root/'collection.json').read_text());collection['files'][name]=report.sha256(root/name)
                report.write_json(root/'collection.json',collection)
                with self.assertRaises(ValueError):report.assess(root)

if __name__=='__main__':unittest.main()
