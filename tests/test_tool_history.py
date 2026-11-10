import copy
import unittest

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import NATIVE_EXAMPLE, system_prompt, format_feedback, prepare_context
from one_gpu_lm.common.tool_history import render_history


class ToolHistoryContracts(unittest.TestCase):
    @staticmethod
    def history(content=None):
        text=content or NATIVE_EXAMPLE
        return [{'role':'system','content':'rules'}, {'role':'user','content':'original task'},
            {'role':'assistant','content':text,'extra':{'actions':[parse_action(text)]}},
            {'role':'user','content':'Exit code: 0\nresult <untrusted>','extra':{'tool':{'returncode':0}}}]

    def test_legacy_rendering_and_native_prompt_are_unchanged(self):
        history=self.history()
        self.assertEqual(render_history(history),[{k:m[k] for k in ['role','content']} for m in history])
        a={'format':'native_schema','context':'full'};b={'format':'native_roundtrip','context':'full'}
        self.assertEqual(system_prompt(a),system_prompt(b));self.assertEqual(format_feedback(a),format_feedback(b))

    def test_both_accepted_syntaxes_preserve_prose_command_and_observation(self):
        for text in ['Before\n'+NATIVE_EXAMPLE+'\nAfter', 'Before\n```bash\nprintf \'%s\\n\' "a & b"\n```\nAfter']:
            history=self.history(text);original=copy.deepcopy(history);rendered=render_history(history,True)
            self.assertEqual(history,original)
            self.assertEqual(rendered[:2],original[:2])
            self.assertEqual(rendered[2]['content'],'Before\n\nAfter')
            self.assertEqual(rendered[2]['tool_calls'][0]['function'],{'name':'bash','arguments':parse_action(text)})
            self.assertEqual(rendered[3],{'role':'tool','name':'bash','content':original[3]['content']})
            self.assertIsInstance(rendered[2]['tool_calls'][0]['function']['arguments'],dict)

    def test_format_feedback_and_user_content_are_never_fabricated_tool_turns(self):
        history=[{'role':'system','content':'rules'},{'role':'user','content':'task'},
            {'role':'assistant','content':'<tool_call>incomplete'},
            {'role':'user','content':'Exit code: 0\nFormat error: choose a valid command.'}]
        self.assertEqual(render_history(history,True),history)

    def test_unmatched_observations_cannot_invent_actions_or_repair_malformed_output(self):
        for mutate in [lambda h:h[2]['extra'].update(actions=[{'command':'different'}]),
                       lambda h:h[2].update(content='<tool_call>incomplete'),
                       lambda h:h[2].update(role='user'),lambda h:h[3].update(role='assistant')]:
            h=self.history();mutate(h)
            with self.assertRaises(ValueError):render_history(h,True)

    def test_compaction_keeps_metadata_for_complete_native_rounds_without_mutating_audit(self):
        h=self.history()[:2]
        for _ in range(10):h+=self.history()[2:]
        original=copy.deepcopy(h);seen=[]
        def encode(items):
            seen.append(render_history(items,True));return list(range(len(items)*500))
        _,info=prepare_context(h,encode,8192,'bounded_history_v1')
        self.assertGreater(info['dropped_rounds'],0);self.assertEqual(h,original)
        self.assertEqual(seen[-1][-1]['role'],'tool')
        self.assertIn('tool_calls',seen[-1][-2]);self.assertEqual(seen[-1][-1]['content'],h[-1]['content'])


if __name__=='__main__':unittest.main()
