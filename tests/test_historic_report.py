import json
from pathlib import Path
import tempfile
import unittest
from tests import test_historic_analysis as fixtures
from one_gpu_lm.verified import historic_report as report


class HistoricPublicReport(unittest.TestCase):
    def test_training_allowlist_discards_private_content_and_distinguishes_cost(self):
        sentinel='PRIVATE_PROMPT_SOURCE_OR_WEIGHT';r={'started_epoch':0,'finished_epoch':20,'training_seconds':15,
          'complete':True,'optimizer_steps':2,'trainable_parameters':8,'input_tokens':9,'supervised_tokens':3,'private':sentinel}
        integrity={'passed':True,'checks':{'ok':True,'private':sentinel},'mixture':{'full_episodes':1,'corrective_fragments':1,
          'selected_records':2,'unique_tasks':2,'private':sentinel},'private':sentinel}
        curve=[{'record':i+1,'update':i+1,'tokens':4,'supervised_tokens':2,'loss':.5,'seconds':1,'record_group_kind':g,'private':sentinel} for i,g in enumerate(['whole_episode','corrective_fragment'])]
        updates=[{'step':1,'records':2,'grad_norm':1,'learning_rate':1e-5,'seconds':1}]
        bundle=report.training_input(r,integrity,curve,updates,[],{'training_seconds':12,'optimizer_steps':1})
        self.assertNotIn(sentinel,json.dumps(bundle));v=report.training_summary(bundle);self.assertEqual(v['extra_training_seconds'],3)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'training-input.json').write_text(json.dumps(bundle));report.report_training(p,False)
            first=(p/'training-summary.json').read_bytes();report.report_training(p,False);self.assertEqual(first,(p/'training-summary.json').read_bytes())

    def test_public_development_pairs_costs_and_repeat_denominators(self):
        f=fixtures.HistoricPairedAnalysis();f.setUp();rows=f.rows();sentinel='PRIVATE_COMMAND'
        for row in rows:row.update(attempted_calls=2,generated_tokens=3,input_tokens=4,actor_seconds=5,seconds=6,private=sentinel)
        c={'complete':True,'started_epoch':0,'finished_epoch':100,'backend_lifecycles':[{'index':1,'role':'base','variant':'base','started_epoch':1,'ready_epoch':3,'stopped_epoch':99,'pid':sentinel}]}
        b=report.evaluation_input(f.spec,f.tasks,{'passed':True,'checks':{'frozen_inputs_unchanged':True},'rows':rows},c,[])
        self.assertNotIn(sentinel,json.dumps(b));r=report.summarize(b)
        self.assertEqual(r['paired_tasks'],60);self.assertEqual(len(r['accumulation_curve']),60)
        self.assertEqual(r['variants']['multifile032']['successes'],180);self.assertEqual(r['cost']['attempted_calls'],1080)
        self.assertEqual(r['phase']['completed_backend_load_seconds'],2)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'analysis-input.json').write_text(json.dumps(b));report.report_evaluation(p,False)
            first=(p/'summary.json').read_bytes();report.report_evaluation(p,False);self.assertEqual(first,(p/'summary.json').read_bytes())
        b['audit']['rows'][0]['unknown_inflight_generation_cost']=True
        self.assertFalse(report.summarize(b)['development_parameter_gate_passed'])


if __name__=='__main__':unittest.main()
