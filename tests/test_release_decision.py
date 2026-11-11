import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_decision',ROOT/'experiments/019-coding-release/decision.py')
decision=importlib.util.module_from_spec(spec);spec.loader.exec_module(decision)
PLAN=json.loads((ROOT/'experiments/019-coding-release/preparation-plan.json').read_text())


def campaign(passes, count=24, tokens=100):
    return {'complete':True,'results':[{'id':str(i),'passed':i in passes,'generated_tokens':tokens,
            'generations':[{'action_format_valid':True}]} for i in range(count)]}


class ReleaseDecisionTests(unittest.TestCase):
    def test_missing_or_duplicate_pairs_block_decision(self):
        a=campaign({0});b=campaign({0,1});b['results'][-1]=b['results'][0]
        with self.assertRaises(ValueError):decision.paired(a,b,map(str,range(24)))
        b=campaign({0,1});b['complete']=False
        with self.assertRaises(ValueError):decision.paired(a,b,map(str,range(24)))

    def test_dev_tie_uses_tokens_then_lower_learning_rate(self):
        a=campaign({0},15);b=campaign({0,1},15)
        ids=list(map(str,range(15)))
        result=decision.select(a,{'low':b,'high':b},ids,PLAN)
        self.assertEqual(result['selected'],'low');self.assertFalse(result['final_test_opened'])
        result=decision.select(a,{'low':b,'high':campaign({0,1},15,99)},ids,PLAN)
        self.assertEqual(result['selected'],'high')

    def test_no_dev_improvement_never_selects_candidate(self):
        a=campaign({0},15)
        self.assertIsNone(decision.select(a,{'low':a,'high':a},map(str,range(15)),PLAN)['selected'])

    def test_quality_alone_never_overrides_missing_rights_audit(self):
        a=campaign({0});fresh=decision.paired(a,campaign({0,1,2,3}),map(str,range(24)))
        history=decision.paired(a,a,map(str,range(24)))
        checks={'legacy_passes':9,'save_reload_greedy_match':True,'all_execution_audits_pass':True,
                'single_training_seed_and_repository_cluster_limits_disclosed':True}
        self.assertFalse(decision.release_gate(fresh,history,checks,PLAN)['accepted'])
        checks['rights_and_secret_scan_pass']=True
        self.assertTrue(decision.release_gate(fresh,history,checks,PLAN)['accepted'])
        history['regressed']=['0']
        self.assertFalse(decision.release_gate(fresh,history,checks,PLAN)['accepted'])


if __name__=='__main__':unittest.main()
