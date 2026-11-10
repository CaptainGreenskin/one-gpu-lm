import copy
import importlib.util
from pathlib import Path
import unittest

from one_gpu_lm.common.actions import SYSTEM, parse_action
from one_gpu_lm.common.agent_policy import (BASH_TOOLS, NATIVE_EXAMPLE, format_feedback,
    prepare_context, system_prompt, validate_policy)


class InterfaceContracts(unittest.TestCase):
    def test_four_arm_order_balances_positions_without_duplicate_tasks(self):
        path=Path(__file__).resolve().parents[1]/'experiments/015-agent-interface/run.py'
        spec=importlib.util.spec_from_file_location('interface_run',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        arms=['baseline','format','context','combined'];ids=[str(i) for i in range(24)]
        plan=module.schedule(ids,arms)
        self.assertEqual(len(plan),96)
        self.assertEqual(len(set(plan)),96)
        for position in range(4):
            self.assertEqual({a:sum(plan[i][1]==a for i in range(position,96,4)) for a in arms},dict.fromkeys(arms,6))

    def test_default_is_identical_and_no_unregistered_policy(self):
        self.assertEqual(system_prompt(validate_policy()), SYSTEM)
        self.assertIn('fenced bash', format_feedback(validate_policy()))
        for p in [{}, {'format':'anything','context':'full'}, {'format':'legacy','context':'lossy'}]:
            with self.assertRaises(ValueError): validate_policy(p)

    def test_native_contract_preserves_strict_action_parser(self):
        self.assertEqual(BASH_TOOLS[0]['function']['parameters']['required'], ['command'])
        self.assertEqual(parse_action(NATIVE_EXAMPLE), {'command':'pwd'})
        for text in [NATIVE_EXAMPLE.replace('</tool_call>',''),
                     NATIVE_EXAMPLE.replace('<parameter=command>',''), NATIVE_EXAMPLE*2]:
            with self.assertRaises(ValueError): parse_action(text)
        self.assertNotIn('fenced bash', system_prompt({'format':'native_schema','context':'full'}))

    @staticmethod
    def history():
        return [{'role':'system','content':'system rules'}, {'role':'user','content':'complete original task'}] + [
            {'role':role, 'content':str(i)+'x'*900} for i in range(6) for role in ['assistant','user']]

    def test_compaction_retains_task_recent_pair_and_never_mutates_audit(self):
        messages=self.history(); original=copy.deepcopy(messages); seen=[]
        def encode(ms):
            seen.append(copy.deepcopy(ms));return list(range(sum(len(m['content']) for m in ms)))
        tokens, info=prepare_context(messages,encode,8192,'bounded_history_v1')
        self.assertEqual(messages,original)
        self.assertLessEqual(len(tokens),6144)
        self.assertGreater(info['raw_input_tokens'],8192)
        self.assertGreater(info['dropped_rounds'],0)
        self.assertEqual(seen[-1][0],original[0])
        self.assertTrue(seen[-1][1]['content'].startswith(original[1]['content']))
        self.assertEqual(seen[-1][-2:],original[-2:])
        self.assertEqual(len(seen[-1])%2,0)

    def test_full_history_and_short_history_are_unchanged(self):
        for policy,ms in [('full',self.history()),('bounded_history_v1',self.history()[:4])]:
            observed=[]
            tokens,info=prepare_context(ms,lambda x: observed.append(copy.deepcopy(x)) or list(range(sum(len(m['content']) for m in x))),8192,policy)
            self.assertEqual(observed,[ms]);self.assertEqual(info['dropped_rounds'],0)

    def test_oversized_pinned_core_fails_without_truncating_task(self):
        ms=self.history()[:4];ms[1]['content']='task'*3000
        tokens,info=prepare_context(ms,lambda x:list(range(sum(len(m['content']) for m in x))),8192,'bounded_history_v1')
        self.assertGreater(len(tokens),8192);self.assertEqual(info['dropped_rounds'],0)

    def test_partial_round_is_not_discarded(self):
        with self.assertRaises(ValueError):
            prepare_context(self.history()[:-1],lambda x:list(range(9000)),8192,'bounded_history_v1')


if __name__=='__main__':unittest.main()
