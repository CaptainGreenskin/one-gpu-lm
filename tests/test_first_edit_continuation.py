"""A late reasoned response keeps its receipt and cannot modify the restored source."""
import json
import unittest
from unittest.mock import patch

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.receipt_client import request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified import correction_rollout as rollout
from one_gpu_lm.verified import first_edit_continuation as continuation
from one_gpu_lm.verified import first_edit_continuation_audit as audit
from one_gpu_lm.verified.recovery_state import inventory
from one_gpu_lm.verified.submission import SUBMIT
from one_gpu_lm.verified.tasks import digest, prepare
from tests import test_teacher_reasoning


class JournaledFirstEdit(unittest.TestCase):
    def setUp(self):
        helper = test_teacher_reasoning.TeacherReasoning()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        self.fixture = helper.fixture()
        fixture = self.fixture.f
        reference = fixture.protocol
        reference['reasoning_comparison']['student_profile_sha256'] = 'student-hash'
        self.reference = reference
        self.profile = helper.profile
        self.tokenizer = helper.tok
        self.student = {'chat_template_kwargs': {'enable_thinking': False}}
        self.root = fixture.root
        snapshot = self.root / 'new-shared-snapshot'
        before = prepare(fixture.snapshot, snapshot / 'tree', fixture.source, fixture.task)
        write_json(snapshot / 'before.json', before)
        write_json(snapshot / 'manifest.json', inventory(snapshot / 'tree'))
        prefix = {'messages': [{'role': 'system', 'content': 'Tool mode'}, {'role': 'user', 'content': 'Repair'}]
                              + [{'role': 'assistant', 'content': 'inspect'},
                                 {'role': 'user', 'content': 'seen source'}] * 6,
                  'standalone_visible_failure': 'ACTUAL_FAILURE', 'historical_calls': 6,
                  'historical_generated_tokens': 300, 'historical_input_tokens': 120,
                  'original_source_sha256': {'logic.py': before['logic.py']}}
        write_json(snapshot / 'prefix.json', prefix)
        self.messages, budget = continuation.initial_context(prefix, reference)
        self.case = {'task': fixture.task, 'source': fixture.source, 'prefix': prefix, 'snapshot': snapshot,
                     'messages': self.messages, 'budget': budget, 'protocol': {**reference, 'evaluation_budget': budget},
                     'screen': {'prefix_sha256': sha256(snapshot / 'prefix.json')}}
        tokens = list(range(20))
        proof = {'input_tokens': len(tokens), 'input_tokens_sha256': digest(json.dumps(tokens, separators=(',', ':')).encode())}
        self.case['screen']['arms'] = {arm: proof for arm in ('control', 'correction')}
        self.journal = RequestJournal(self.root / 'generation-receipts')
        self.helper = helper

    def collect(self, arm, actions, late):
        fixture = self.fixture
        fixture.h.actions = actions[:]
        out = self.root / 'actors'
        where = out / 'private' / (fixture.f.task['id'] + '--' + arm)
        trace = continuation.initialize(where, self.case, arm)
        thought = 'A bounded thought </think>' if arm == 'correction' else ''
        generate = self.helper.generator(fixture, thought)
        def health(sock, payload, **kwargs):
            return dict(generate(sock, payload, **kwargs), request_journal_version=1)
        def record(sock, path, label, call, payload, **kwargs):
            key = request_identity(label, call)
            result = self.journal.execute(key, payload, lambda req: generate(sock, req))
            receipt = self.journal.status(key)
            recovery = {k: receipt[k] for k in ('request_id', 'accepted_epoch', 'finished_epoch')}
            return result, {**recovery, 'accounting_drain_seconds': 0.0} if late else None
        with patch.object(rollout, 'request', health), patch.object(rollout, 'generate_once', record):
            rollout.advance(self.root, fixture.f.task, self.case['protocol'], where, trace, 'socket',
                            fixture.h.runner, arm, journal_path=self.journal.directory, recovery_seconds=120)
        result = rollout.finish(fixture.f.data, fixture.f.task, fixture.f.source, self.case['protocol'],
                                where, trace, fixture.h.runner)
        result.update(run_id=where.name, arm=arm, timeout_call=trace.get('timeout_call'),
                      accounting_drain_seconds=trace.get('accounting_drain_seconds', 0.0),
                      complete_success_within_budget=trace['eligible_verified_trajectory'])
        item = {'task_id': fixture.f.task['id'], 'run_id': where.name, 'arm': arm}
        with patch.object(audit, 'encode_prefix', return_value=(list(range(20)), {})), \
                patch.object(audit, 'encode_training_example', return_value={'ids': list(range(20)) + [888], 'supervised_tokens': 1}):
            return audit.check_arm(self.root, fixture.f.data, out, where, self.case, item, result, result,
                                   self.journal, self.reference, (self.profile, self.student),
                                   self.tokenizer, fixture.f.normalizer)

    def test_late_edit_is_counted_and_audited_but_the_other_arm_can_repair(self):
        control, _, control_ids = self.collect('control', ['edit-source'], late=True)
        self.assertTrue(control['timeout_recovered'])
        self.assertIsNone(control['first_edit_call'])
        self.assertFalse(control['correct_patch'])
        low, records, low_ids = self.collect('correction', ['edit-source', 'python3 check.py', SUBMIT], late=False)
        self.assertTrue(low['qualified'])
        self.assertEqual(low['first_edit_call'], 1)
        self.assertTrue(low_ids.isdisjoint(control_ids))
        self.assertEqual([r['provenance']['kind'] for r in records], ['edit', 'verify', 'submit'])
        self.assertIn('ACTUAL_FAILURE', json.dumps(records[0]['messages']))
        self.assertNotIn('A bounded thought', json.dumps(records))
        self.assertNotIn('HIDDEN_CASE_SENTINEL', json.dumps(records))
        trace = continuation.read(self.root / 'actors' / 'private' / 'fixture--correction' / 'trajectory.json')
        trace['steps'][1]['protected_after'] = False
        self.assertEqual(audit.target_steps(trace, 'python3 check.py'), ([], 'protected_file_changed'))
