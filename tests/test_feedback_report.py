"""Recompute024 public evidence and reject rehashed changes to scope and scores."""
import importlib.util,json,shutil,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/024-feedback-noise'
spec=importlib.util.spec_from_file_location('feedback024_report',EXP/'report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
RESULTS=EXP/'results/2026-11-14'

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

if __name__=='__main__':unittest.main()
