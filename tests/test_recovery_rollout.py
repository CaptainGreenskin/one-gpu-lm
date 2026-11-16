"""CPU integration fixtures for shared states, fairness and failure accounting."""
import copy
import json
from types import SimpleNamespace
import unittest
import sys
from unittest.mock import patch

from tests import test_verified_rollout as fixtures
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.verified import recovery_rollout as rr, recovery_replay as audit
from one_gpu_lm.verified.recovery_audit import check_shared_state, attempt_costs
from one_gpu_lm.verified import recovery_audit
from one_gpu_lm.verified.recovery_report import summarize, render
from one_gpu_lm.verified.recovery import actual_cost, branch_order, select_tasks
from one_gpu_lm.verified.recovery_state import inventory
from one_gpu_lm.verified.submission import SUBMIT
from one_gpu_lm.verified.tasks import digest


class PairedRecovery(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.RolloutBoundaries(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.c = c = fixture; self.decoded = {}; self.serial = 0
        c.protocol['evaluation_budget'] = c.protocol.pop('teacher_budget')
        c.protocol['evaluation_budget'].update(max_model_calls=12, max_task_seconds=600,
            max_total_generated_tokens=6144, max_generated_tokens_per_call=512)
        c.protocol['evaluation_profile_sha256'] = c.protocol.pop('teacher_profile_sha256')
        c.protocol['adapter_sha256'] = {'adapter.safetensors': 'candidate-sha'}
        self.runner = SimpleNamespace(execute=c.execute, timeout=5)
        self.profile = {'chat_template_kwargs': {'enable_thinking': False}}
        self.actions = []; self.requests = []; self.role = 'student'
        self.patches = [patch.object(rr, 'request', self.request),
                        patch.object(rr, 'normalization', return_value=c.normalizer),
                        patch.object(audit, 'encode_prefix', return_value=(list(range(20)), {}))]
        for p in self.patches: p.start(); self.addCleanup(p.stop)
    def request(self, socket, payload, **kwargs):
        if payload['operation'] == 'health':
            return dict(self.c.health, adapter=self.role=='student',
                        adapter_sha256=self.c.protocol['adapter_sha256'] if self.role=='student' else {})
        self.requests.append(copy.deepcopy(payload)); self.serial += 1
        command = self.actions.pop(0)
        if command is None: raise TimeoutError()
        text = '```bash\n'+command+'\n```'; ids = [self.serial, 777]; self.decoded[tuple(ids)] = text
        tokens, context = prepare_context(payload['messages'], lambda _: list(range(20)), 8192, 'bounded_history_v1')
        return {'text': text, 'ids': ids, 'generated_tokens': 2, 'input_tokens': len(tokens),
                'input_tokens_sha256': digest(json.dumps(tokens, separators=(',', ':')).encode()),
                'context': context, 'adapter': self.role=='student'}
    def decode(self, ids, **kwargs): return self.decoded[tuple(ids)]
    def prefix(self):
        c = self.c; self.actions = ['cat logic.py']*3
        state = rr.initialize(c.data, c.task, c.source, c.protocol, c.root/'prefix')
        rr.advance(c.root, c.task, c.protocol, c.root/'prefix', state, 'socket', self.runner, 'student', 3)
        return state
    def replay(self, directory, trace):
        c = self.c
        return audit.replay(directory, trace, c.task, c.protocol, self.profile, self, c.normalizer)
    def test_identical_first_continuation_inputs_but_independent_workspaces_and_model_roles(self):
        c = self.c; prefix = self.prefix()
        receipt = rr.capture_prefix(c.root/'prefix', prefix, c.root/'snapshot')
        traces = {}; inputs = []; rows = {}
        for branch, role in [('teacher', 'base'), ('student', 'student')]:
            out = c.root/branch
            trace = rr.fork_prefix(c.root/'prefix', c.root/'snapshot', receipt, out, branch)
            self.assertEqual(inventory(out/'workspace'), inventory(c.root/'snapshot/tree'))
            self.assertEqual(trace['messages'], prefix['messages'])
            self.assertEqual(trace['actor_seconds'], prefix['actor_seconds'])
            self.role = role; self.actions = ['edit-source', 'python3 check.py', SUBMIT]
            request_start = len(self.requests)
            rr.advance(c.root, c.task, c.protocol, out, trace, 'socket', self.runner, role)
            inputs.append(self.requests[request_start])
            rows[branch] = rr.finish(c.data, c.task, c.source, c.protocol, out, trace, self.runner)
            self.assertTrue(rows[branch]['eligible_verified_trajectory'])
            self.assertTrue(self.replay(out, trace)['passed'])
            traces[branch] = trace
        self.assertEqual(inputs[0], inputs[1])
        self.assertEqual((c.root/'snapshot/tree/logic.py').read_text(), 'answer = 3\n')
        for req in self.requests:
            self.assertNotIn('HIDDEN_CASE_SENTINEL', json.dumps(req))
        row = dict(rows, disposition='paired', prefix_cost={'attempted_calls':3, 'generated_tokens':6,
                   'input_tokens':60, 'actor_seconds':prefix['actor_seconds']})
        self.assertEqual(actual_cost(row)['attempted_calls'], 9)
        self.assertEqual(actual_cost(row)['generated_tokens'], 18)
        altered = copy.deepcopy(traces['teacher']); altered['steps'][3]['model_role'] = 'student'
        self.assertFalse(self.replay(c.root/'teacher', altered)['passed'])
        self.assertTrue(all(check_shared_state(c.root/'teacher', traces['teacher'], prefix, receipt).values()))
        restored = c.root/'teacher/restored-state.json'; value = json.loads(restored.read_text()); value['actor_seconds'] = 0
        restored.write_text(json.dumps(value))
        self.assertFalse(all(check_shared_state(c.root/'teacher', traces['teacher'], prefix, receipt).values()))
    def test_teacher_cannot_get_twelve_additional_calls(self):
        c = self.c; prefix = self.prefix(); receipt = rr.capture_prefix(c.root/'prefix', prefix, c.root/'snapshot')
        trace = rr.fork_prefix(c.root/'prefix', c.root/'snapshot', receipt, c.root/'teacher', 'teacher')
        self.role = 'base'; self.actions = ['cat logic.py']*12
        rr.advance(c.root, c.task, c.protocol, c.root/'teacher', trace, 'socket', self.runner, 'base')
        self.assertEqual(trace['attempted_calls'], 12); self.assertEqual(len(self.actions), 3)
        self.assertEqual(trace['terminal_reason'], 'call_limit')
        self.assertTrue(self.replay(c.root/'teacher', trace)['passed'])
    def test_unknown_generation_stops_and_cannot_resume(self):
        c = self.c; prefix = self.prefix(); self.actions = [None]
        with self.assertRaises(RuntimeError):
            rr.advance(c.root, c.task, c.protocol, c.root/'prefix', prefix, 'socket', self.runner, 'student')
        stored = json.loads((c.root/'prefix/trajectory.json').read_text())
        self.assertEqual(stored['attempted_calls'], 4)
        self.assertTrue(stored['unknown_inflight_generation_cost'])
        with self.assertRaises(ValueError):
            rr.advance(c.root, c.task, c.protocol, c.root/'prefix', stored, 'socket', self.runner, 'student')
        self.assertFalse((c.root/'prefix/result.json').exists())
    def test_previous_edits_are_ineligible_and_wrong_artifact_is_rejected(self):
        c = self.c; prefix = self.prefix()
        prefix['steps'][1]['source_after']['logic.py'] = 'changed'
        with self.assertRaises(ValueError): rr.capture_prefix(c.root/'prefix', prefix, c.root/'snapshot')
        self.role = 'base'
        with self.assertRaises(ValueError):
            rr.advance(c.root, c.task, c.protocol, c.root/'prefix', prefix, 'socket', self.runner, 'student')
    def test_no_new_grader_information_is_exposed_and_tampering_fails(self):
        c = self.c; prefix = self.prefix(); self.actions = ['tamper-test', 'edit-source', 'python3 check.py', SUBMIT]
        rr.advance(c.root, c.task, c.protocol, c.root/'prefix', prefix, 'socket', self.runner, 'student')
        value = rr.finish(c.data, c.task, c.source, c.protocol, c.root/'prefix', prefix, self.runner)
        self.assertFalse(value['success_within_budget']); self.assertEqual(value['grade_reason'], 'protected_file_tampering')
    def test_selection_and_order_do_not_use_outcomes(self):
        tasks = [{'id':str(i), 'repository':'r'+str(i%3), 'split':'training'} for i in range(12)]
        self.assertEqual(select_tasks(tasks, 8), select_tasks(list(reversed(tasks)), 8))
        self.assertEqual(branch_order(0), ['teacher', 'student']); self.assertEqual(branch_order(1), ['student', 'teacher'])
        tasks[0]['split'] = 'development'
        with self.assertRaises(ValueError): select_tasks(tasks, 8)

    def test_full_paired_audit_and_private_continuation_export(self):
        c=self.c; run_dir=c.root/'run'; directory=run_dir/'private/tasks'/c.task['id']
        c.protocol.update(run_directory='run',data_directory='data',supervisor_receipt='supervisor.json',
            evaluation_profile_path='profile.json',environment_path='environment.json',frozen_files={},task_ids=[c.task['id']],encoding_screen_max_tokens=8192)
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        write(c.root/'protocol.json',c.protocol);write(c.root/'profile.json',self.profile)
        write(c.root/'private/parent-runtime.json',{'model':'fixture'});write(c.root/'environment.json',{'fixture':True})
        write(c.data/'tasks.json',{'tasks':[c.task]});write(c.data/'sources.json',{'sources':[c.source]})
        self.actions=['cat logic.py']*3
        prefix=rr.initialize(c.data,c.task,c.source,c.protocol,directory/'prefix')
        rr.advance(c.root,c.task,c.protocol,directory/'prefix',prefix,'socket',self.runner,'student',3)
        receipt=rr.capture_prefix(directory/'prefix',prefix,directory/'snapshot')
        row={'id':c.task['id'],'disposition':'paired','branch_order':branch_order(0),'snapshot_receipt':receipt,
             'prefix_cost':{**{k:prefix[k] for k in ['attempted_calls','generated_tokens','actor_seconds']},'input_tokens':60}}
        for branch,role in [('teacher','base'),('student','student')]:
            trace=rr.fork_prefix(directory/'prefix',directory/'snapshot',receipt,directory/branch,branch)
            self.role=role;self.actions=['edit-source','python3 check.py',SUBMIT]
            rr.advance(c.root,c.task,c.protocol,directory/branch,trace,'socket',self.runner,role)
            row[branch]=rr.finish(c.data,c.task,c.source,c.protocol,directory/branch,trace,self.runner)
        row['actual_cost']=actual_cost(row)
        write(run_dir/'controller.json',{'complete':True,'backend_stopped':True,'tasks':[row],
            'protocol_sha256':digest((c.root/'protocol.json').read_bytes()),'backend_lifecycles':[{'index':1,'role':'student'}]})
        write(run_dir/'private/backends/0001-student-environment.json',{'fixture':True})
        write(c.root/'supervisor.json',{'complete':True,'timed_out':False,'worker_returncode':0})
        tokenizer_module=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:self))
        with patch.dict(sys.modules,{'transformers':tokenizer_module}), \
             patch.object(recovery_audit,'normalization',return_value=c.normalizer), \
             patch.object(recovery_audit,'encode_prefix',return_value=(list(range(20)),{})), \
             patch.object(recovery_audit,'encode_training_example',return_value={'ids':list(range(20))+[888],'supervised_tokens':1}):
            value=recovery_audit.run(c.root,c.root/'protocol.json',run_dir/'audit.json')
        self.assertTrue(value['passed'],value)
        self.assertTrue(value['rows'][0]['qualified_teacher_continuation'])
        self.assertEqual(value['private_export_summary']['full_episodes'],0)
        self.assertEqual(value['private_export_summary']['unique_teacher_continuations'],1)
        records=[json.loads(line) for line in (run_dir/'private/continuation-export/data.jsonl').read_text().splitlines()]
        self.assertEqual([r['provenance']['call'] for r in records],[4,5,6])
        self.assertNotIn('HIDDEN_CASE_SENTINEL',json.dumps(records))
        self.assertEqual(sum(a['attempted_calls'] for a in value['attempts']),9)
        summary=summarize(value,{})
        self.assertEqual(summary['complete_audited_pairs'],1)
        self.assertFalse(summary['data_feasibility_gate_passed'])
        self.assertEqual(summary['net_teacher_gain'],0)
        self.assertEqual(summary['actual_cost_shared_prefix_once']['attempted_calls'],9)
        render(value,{},c.root/'report')
        self.assertNotIn('HIDDEN_CASE_SENTINEL',(c.root/'report/README.md').read_text())


if __name__ == '__main__': unittest.main()
