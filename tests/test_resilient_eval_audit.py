import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_verified_evaluation as fixtures
from one_gpu_lm.serving.receipt_client import request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified import historic_replay, historic_tasks, resilient_rollout
from one_gpu_lm.verified.resilient_eval_audit import audit_task
from one_gpu_lm.verified.submission import SUBMIT
from one_gpu_lm.verified.tasks import digest


class LateReceiptAuditTests(unittest.TestCase):
    def test_late_response_cannot_change_code_or_be_counted_as_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = RequestJournal(root/'journal')
            payload = {'operation':'generate','max_new_tokens':2}
            (root/'request-001.json').write_text(json.dumps(payload))
            identity = request_identity('trial', 1)
            started = time.time()
            generation = journal.execute(identity, payload, lambda _: {'generated_tokens':2,'ids':[1,2]})
            raw = journal.status(identity)
            recovery = dict(request_id=identity, accepted_epoch=raw['accepted_epoch'],
                            finished_epoch=raw['finished_epoch'], accounting_drain_seconds=.1)
            result = dict(run_id='trial', timeout_call=1, accounting_drain_seconds=.1,
                          complete_success_within_budget=False)
            trace = dict(started_epoch=started, finished_epoch=time.time(), submitted=False,
                         actor_seconds=.6, timeout_call=1, attempted_calls=1, accounting_drain_seconds=.1,
                         terminal_reason='time_limit_before_tool', final_source_sha256={'a':'initial'},
                         steps=[dict(call=1, generation=generation, source_before={'a':'initial'},
                                     transport_timeout_receipt=recovery)])
            self.assertEqual(audit_task(root, trace, result, journal), {identity})

            for field, value in [('final_source_sha256', {'a':'changed'}), ('submitted', True)]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    audit_task(root, dict(trace, **{field:value}), result, journal)
            with self.assertRaises(ValueError):
                audit_task(root, trace, dict(result, complete_success_within_budget=True), journal)
            altered = copy.deepcopy(trace)
            altered['steps'][0]['generation']['ids'][0] = 3
            with self.assertRaises(ValueError):
                audit_task(root, altered, result, journal)
            (root/'tool-001.json').write_text('{}')
            with self.assertRaises(ValueError):
                audit_task(root, trace, result, journal)


class GraderInputsAuditTests(unittest.TestCase):
    def test_reconstructs_inputs_even_when_grader_generates_cache(self):
        fixture = fixtures.EvaluationIsolation()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        case = fixture.c
        source_id = 'a--b--' + case.source['revision']
        snapshot = case.snapshot.rename(case.snapshot.with_name(source_id))
        case.source.update(source_id=source_id, upstream_parent='b'*40)
        case.task.update(source_id=source_id, upstream_parent='b'*40,
                         origin='upstream_history_parent_plus_test_patch')
        case.health['request_journal_version'] = 1
        case.actions = ['edit-source', SUBMIT]
        out = case.root/'out'
        receipt = out/'grader-input.json'

        def execute(workspace, command):
            result = case.execute(workspace, command)
            if command == case.task['grading_command']:
                self.assertTrue(receipt.is_file())
                self.assertIn('hidden.py', json.loads(receipt.read_bytes()))
                cache = workspace/'node_modules/.vitest-cache'
                cache.mkdir(parents=True)
                (cache/'_metadata.json').write_text('{}')
            return result

        def generate(socket, journal, run_id, call, payload, **kwargs):
            return case.request(socket, payload), None

        sandbox = SimpleNamespace(execute=execute, timeout=5)
        with patch.object(resilient_rollout, 'request', case.request), \
             patch.object(resilient_rollout, 'generate_once', generate), \
             patch.object(resilient_rollout, 'normalization', return_value=case.normalizer):
            result = resilient_rollout.run(case.root, case.data, case.task, case.source, 'socket',
                                           sandbox, case.protocol, out, 'base_current',
                                           journal_path=case.root/'journal', recovery_seconds=3)
        self.assertTrue(result['complete_success_within_budget'])
        trace_path = out/'trajectory.json'
        trace = json.loads(trace_path.read_bytes())
        self.assertEqual(trace['grader_input_sha256'], digest(receipt.read_bytes()))
        before = historic_tasks.prepare(snapshot, case.root/'before', case.source, case.task)

        def replay(name):
            return historic_replay.recorded_grade(case.root, case.data, out, case.task, case.source,
                                                  before, case.root/name)

        self.assertEqual(result['grade'], replay('reconstructed'))
        contents = json.loads(receipt.read_bytes())
        contents.pop('hidden.py')
        receipt.write_text(json.dumps(contents))
        with self.assertRaisesRegex(ValueError, 'trajectory checksum'):
            replay('altered-receipt')
        trace['grader_input_sha256'] = digest(receipt.read_bytes())
        trace_path.write_text(json.dumps(trace))
        with self.assertRaisesRegex(ValueError, 'differ from the submitted patch'):
            replay('altered-inventory')
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or unsafe'):
            replay('missing-receipt')
        trace.pop('grader_input_sha256')
        trace_path.write_text(json.dumps(trace))
        with self.assertRaisesRegex(ValueError, 'differ from the submitted patch'):
            replay('legacy-cache')


if __name__ == '__main__':
    unittest.main()
