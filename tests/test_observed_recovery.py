"""Contracts for private on-policy collection, including leakage and bad states."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from one_gpu_lm.common.receipts import sha256

path = Path(__file__).resolve().parents[1] / 'experiments/021-observed-recovery/capture.py'
spec = importlib.util.spec_from_file_location('capture021', path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)

def module(name):
    spec = importlib.util.spec_from_file_location(name+'021', path.with_name(name+'.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

builder = module('build_data')
decision = module('decision')


class CaptureContracts(unittest.TestCase):
    def test_evaluation_and_duplicate_tasks_rejected(self):
        for tasks in [[{'id': 'x', 'split': 'test'}], [{'id': 'x', 'split': 'train'}] * 2]:
            with self.assertRaises(ValueError):
                capture.training_task(tasks, 'x')

    def test_exact_sources_and_immutable_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'work'; root.mkdir(); (root / 'source.py').write_text('broken\n')
            task = {'id': 'x', 'split': 'train', 'editable_files': ['source.py']}
            original = {'source.py': sha256(root / 'source.py')}
            state = {'identity': {'id': 'x'}, 'boundary': 'completed_step', 'n_calls': 1,
                     'trajectory': {'local_model': {'responses': [{}]},
                                    'messages': [{'role': 'user', 'content': 'observed'}]}}
            out = root.parent / 'private.json'
            got = capture.record_state(task, state, root, original, out)
            self.assertEqual(got['sources'], {'source.py': 'broken\n'})
            self.assertEqual(got['source_state']['changed_editable_files'], 0)
            self.assertTrue(got['protected_files_unchanged'])
            self.assertEqual(got, capture.record_state(task, state, root, original, out))
            (root / 'source.py').write_text('changed\n')
            with self.assertRaises(ValueError):
                capture.record_state(task, state, root, original, out)
            self.assertEqual(json.loads(out.read_text()), got)

    def test_invalid_source_is_not_followed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'source.py').symlink_to('/does/not/exist')
            task = {'id': 'x', 'split': 'train', 'editable_files': ['source.py']}
            state = {'identity': {'id': 'x'}, 'boundary': 'completed_step', 'n_calls': 1,
                     'trajectory': {'local_model': {'responses': [{}]}, 'messages': []}}
            got = capture.record_state(task, state, root, {'source.py': '0' * 64}, root / 'private.json')
            self.assertEqual(got['sources'], {})
            self.assertEqual(got['source_state']['invalid_editable_files'], 1)

    def test_inflight_and_count_mismatch_rejected(self):
        task = {'id': 'x', 'split': 'train', 'editable_files': []}
        base = {'identity': {'id': 'x'}, 'boundary': 'completed_step', 'n_calls': 1,
                'trajectory': {'local_model': {'responses': []}, 'messages': []}}
        for change in [{'boundary': 'in_flight'}, {'n_calls': 9}, {}]:
            with self.assertRaises(ValueError):
                capture.record_state(task, dict(copy.deepcopy(base), **change), '.', {}, 'never-written.json')


class StateSelectionContracts(unittest.TestCase):
    def states(self):
        return [{'task_id':'train-x', 'split':'train', 'completed_calls':i, 'terminal':False,
                 'protected_files_unchanged':True, 'sources':{'source.py':'x = 1\n'},
                 'source_state':{'changed_editable_files':0, 'invalid_editable_files':0},
                 'trajectory':{'local_model':{'backend_errors':[]}, 'messages':[{'role':'user','content':'observed'}]}} for i in range(1,9)]

    def task(self): return {'id':'train-x','split':'train','editable_files':['source.py']}

    def test_stall_uses_newest_and_excludes_ever_edited_prefix(self):
        states = self.states()
        self.assertEqual([s['completed_calls'] for _,s in builder.ordered_states(states,self.task(),'stall_repair')],[8,7,6,5,4])
        states[4]['source_state']['changed_editable_files']=1
        self.assertEqual([s['completed_calls'] for _,s in builder.ordered_states(states,self.task(),'stall_repair')],[4])

    def test_wrong_edit_priority_then_tool_error_then_reads(self):
        states=self.states();states[4]['source_state']['changed_editable_files']=1
        states[5]['trajectory']['messages']=[{'role':'user','extra':{'tool':{'returncode':1}}}]
        got=builder.ordered_states(states,self.task(),'late_repair')
        self.assertEqual([(k,s['completed_calls']) for k,s in got],[('wrong_edit',5),('tool_error',6),('read_only',8),('read_only',7),('read_only',4)])

    def test_protected_tampering_invalid_and_backend_error_poison_prefix(self):
        for mutate in [lambda s:s.update(protected_files_unchanged=False),
                       lambda s:s['source_state'].update(invalid_editable_files=1),
                       lambda s:s['trajectory']['local_model'].update(backend_errors=[{}])]:
            states=self.states();mutate(states[2])
            self.assertEqual(builder.ordered_states(states,self.task(),'late_repair'),[])

    def test_gap_terminal_and_wrong_partition_fail_closed(self):
        states=self.states();states[-1]['terminal']=True
        self.assertEqual(builder.ordered_states(states,self.task(),'late_repair')[0][1]['completed_calls'],7)
        self.assertEqual(builder.ordered_states(states[1:],self.task(),'late_repair'),[])
        with self.assertRaises(ValueError):builder.ordered_states(states,dict(self.task(),split='test'),'late_repair')
        states[0]['task_id']='other'
        with self.assertRaises(ValueError):builder.ordered_states(states,self.task(),'late_repair')


class SelectionGates(unittest.TestCase):
    def campaign(self, passed):
        return {'complete':True,'results':[{'id':str(i),'passed':i in passed,'generated_tokens':12,
            'generations':[{'action_format_valid':True}]} for i in range(5)]}

    def test_must_beat_both_base_and_frozen_reference(self):
        plan={'candidates':[{'name':'observed'}], 'selection':{
            'must_improve_development_net_tasks':1,'maximum_development_regressions':1,
            'must_improve_reference_net_tasks':1,'maximum_reference_regressions':1}}
        args=(self.campaign({0}), self.campaign({0,1}), self.campaign({0,1}), map(str,range(5)), plan)
        result=decision.select(*args); self.assertIsNone(result['selected']);self.assertFalse(result['final_test_opened'])
        result=decision.select(self.campaign({0}), self.campaign({0,1}), self.campaign({0,1,2}),map(str,range(5)),plan)
        self.assertEqual(result['selected'],'observed')

    def test_missing_and_duplicate_pairs_rejected(self):
        value=self.campaign({0});value['results'][-1]=value['results'][0]
        with self.assertRaises(ValueError):decision.paired(self.campaign({0}),value,map(str,range(5)))


class CompleteEncodingContracts(unittest.TestCase):
    def test_failed_history_masked_and_limits_never_truncate(self):
        from tests.test_training_encoding import RecordingTokenizer
        from one_gpu_lm.common.agent_policy import NATIVE_EXAMPLE
        from one_gpu_lm.common.actions import parse_action
        from one_gpu_lm.common.training_encoding import encode_training_example
        tokenizer=RecordingTokenizer(); action=NATIVE_EXAMPLE.replace('pwd','false')
        row={'messages':[{'role':'system','content':'rules'},{'role':'user','content':'task'},
            {'role':'assistant','content':action,'extra':{'actions':[parse_action(action)]}},
            {'role':'user','content':'Exit code: 1\nobserved failure','extra':{'tool':{'returncode':1}}}],
            'target':NATIVE_EXAMPLE.replace('pwd','printf repair')}
        profile={'chat_template_kwargs':{'enable_thinking':False},'training':{'dialogue_format':'native_roundtrip'},'generation':{}}
        original=copy.deepcopy(row); encoded=encode_training_example(row,tokenizer,profile)
        profile['training']['max_sequence_length']=len(encoded['ids'])
        profile['generation']['max_new_tokens']=encoded['supervised_tokens']
        self.assertTrue(builder.within_limits(row,tokenizer,profile))
        profile['training']['max_sequence_length']-=1
        self.assertFalse(builder.within_limits(row,tokenizer,profile))
        profile['training']['max_sequence_length']+=1;profile['generation']['max_new_tokens']-=1
        self.assertFalse(builder.within_limits(row,tokenizer,profile)); self.assertEqual(row,original)
        self.assertTrue(all(n==-100 for n in encoded['labels'][:encoded['prefix_tokens']]))
        suffix=bytes(encoded['labels'][encoded['prefix_tokens']:]).decode()
        self.assertIn('printf repair',suffix);self.assertNotIn('false',suffix);self.assertNotIn('observed failure',suffix)
