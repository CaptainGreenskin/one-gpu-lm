"""Recompute024 public evidence and reject rehashed changes to scope and scores."""
import importlib.util,json,shutil,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/024-feedback-noise'
spec=importlib.util.spec_from_file_location('feedback024_report',EXP/'report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
RESULTS=EXP/'results/2026-11-14'
spec=importlib.util.spec_from_file_location('feedback024_residual',EXP/'residual_review.py')
residual=importlib.util.module_from_spec(spec);spec.loader.exec_module(residual)

class ResidualFeedbackClassification(unittest.TestCase):
 def test_prefix_diagnosis_requires_two_original_truncation_markers(self):
  marker=residual.norm.MARKER
  self.assertEqual(residual.classify('same tail'+marker,'same tail continued'+marker)['category'],'v2_truncated_prefix_length')
  for a,b in [('same tail','same tail continued'),('same tail'+marker,'same tail continued'),('left'+marker,'right'+marker)]:
   self.assertEqual(residual.classify(a,b)['category'],'v2_other_text')
 def test_only_status_rows_can_be_labeled_reordered(self):
  self.assertEqual(residual.classify('header\n  ✔ first\n  ✔ second\nend\n','header\n  ✔ second\n  ✔ first\nend\n')['category'],'v2_test_row_order')
  self.assertEqual(residual.classify('x = 1\ny = 2\n','y = 2\nx = 1\n')['category'],'v2_other_text')
  with self.assertRaises(ValueError):residual.classify('identical','identical')

@unittest.skipUnless(RESULTS.exists(),'024 complete evidence has not been collected; no invented results')
class ActualFeedbackEvidence(unittest.TestCase):
 def test_complete_evidence_recomputes_primary_and_secondary_counts(self):
  value=report.assess(RESULTS);expected=json.loads((RESULTS/'summary.json').read_text())
  key='sampled_gpu_energy_kwh';self.assertAlmostEqual(value['hardware'].pop(key),expected['hardware'].pop(key),delta=1e-12)
  self.assertEqual(value,expected);self.assertEqual(len(value['cells']),18)
  self.assertEqual(sum(r['tasks'] for r in value['cells']),270)
  self.assertEqual(len(value['pairs']),405);self.assertEqual(len(value['task_stability']),90)
  self.assertEqual((value['within_condition_pairs'],value['matched_condition_pairs']),(270,135))
  self.assertEqual(sum(r['calls'] for r in value['probe_groups']),36)
  self.assertFalse(value['final_test_opened']);self.assertEqual(value['training_updates'],0);self.assertFalse(value['weights_public'])
  self.assertTrue(value['time']['within_original_deadline'])
  for r in value['stability']:
   tasks=[t for t in value['task_stability'] if (t['model'],t['condition'])==(r['model'],r['condition'])]
   pairs=[p for p in value['pairs'] if p['comparison']=='within_condition' and (p['model'],p['condition'])==(r['model'],r['condition'])]
   self.assertEqual(r['pair_denominator'],45);self.assertEqual(len(pairs),45)
   self.assertEqual(r['pairwise_disagreements'],sum(p['outcome_disagreement'] for p in pairs))
   self.assertEqual(r['action_pairwise_disagreements'],sum(p['action_disagreement'] for p in pairs))
   self.assertEqual(r['mixed_outcome_tasks'],sum(t['mixed_outcome'] for t in tasks))
   self.assertEqual(r['mixed_action_tasks'],sum(t['mixed_action'] for t in tasks))
 def test_rehashed_tampering_cannot_change_scores_or_frozen_rules(self):
  changes=[('controller.json',lambda v:v.update(final_test_opened=True)),
   ('protocol.json',lambda v:v['agent'].update(max_model_calls=13)),
   ('campaigns/r01-base-baseline.json',lambda v:v['results'][0].update(passed=not v['results'][0]['passed'])),
   ('artifacts/r01-base-attempt-1.json',lambda v:v.update(adapter_sha256={'adapter_model.safetensors':'0'*64})),
   ('historical-review.json',lambda v:v.update(baseline_exactly_matches_023=False)),
   ('historical-review.json',lambda v:v.update(old_first_residual_pairs_now_equal=0)),
   ('preregistration.json',lambda v:v.update(verified_utc='2099-01-01T00:00:00+00:00'))]
  for name,change in changes:
   with self.subTest(file=name),tempfile.TemporaryDirectory() as temp:
    root=Path(temp)/'evidence';shutil.copytree(RESULTS,root)
    value=json.loads((root/name).read_text());change(value);report.write_json(root/name,value)
    collection=json.loads((root/'collection.json').read_text());collection['files'][name]=report.sha256(root/name)
    report.write_json(root/'collection.json',collection)
    with self.assertRaises(ValueError):report.assess(root)
 def test_rehashed_posthoc_diagnostic_cannot_change_scope_or_select_pairs(self):
  changes=[lambda v:v.update(normalizer_changed=True),lambda v:v.update(source_sha256='0'*64),
   lambda v:v['pairs'].pop(),lambda v:v['pairs'][0].update(v2_sha256=['0'*64,'1'*64])]
  for i,change in enumerate(changes):
   with self.subTest(change=i),tempfile.TemporaryDirectory() as temp:
    root=Path(temp)/'evidence';shutil.copytree(RESULTS,root)
    name='feedback-residual-review.json';value=json.loads((root/name).read_text());change(value);report.write_json(root/name,value)
    collection=json.loads((root/'posthoc-collection.json').read_text());collection['files'][name]=report.sha256(root/name)
    report.write_json(root/'posthoc-collection.json',collection)
    with self.assertRaises(ValueError):report.assess(root)

if __name__=='__main__':unittest.main()
