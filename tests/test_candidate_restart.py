"""Check that a reboot preserves completed rollouts and accounts for partial generation."""
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.receipt_client import payload_sha256, request_identity
from one_gpu_lm.serving.request_journal import RequestJournal, persist_new


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                            'experiments/039-release-development/candidate_restart.py'))


class CandidateRestart(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.plan = {'pipeline_limit_seconds': 500, 'schedule_sha256': 'a' * 64,
                     'evaluation_budget': {'max_model_calls': 4}}
        self.order = [{'run_id': f't{n}', 'task_number': n, 'repetition': 0, 'block': 0, 'arm': 'base'}
                      for n in (1, 2)]
        self.backend = {'index': 1, 'attempt': 1, 'role': 'base', 'started_epoch': 101, 'ready_epoch': 105}
        folder = MODULE['backend_directory'](self.out, self.backend)
        self.journal = RequestJournal(folder/'requests')
        self.artifact = {'ready': True, 'adapter': False, 'request_journal_version': 1}
        write_json(folder/'served-artifact.json', self.artifact)
        self.state = {'stage': 'evaluating', 'complete': False, 'attempt': 1, 'boot_id': 'old-boot',
                      'started_epoch': 100, 'attempt_started_epoch': 100, 'deadline_epoch': 600,
                      'protocol_sha256': 'b' * 64, 'schedule_sha256': self.plan['schedule_sha256'],
                      'final_test_opened': False, 'weights_public': False, 'optimizer_updates': 0,
                      'backend_lifecycles': [self.backend], 'current_run': self.order[1],
                      'current_backend_index': 1, 'restarts': [], 'runs': [self.actor(self.order[0], 110)]}
        (self.out/'telemetry.csv').write_text('original telemetry receipt\n')

    def actor(self, item, timestamp, completed=True):
        directory = MODULE['actor_directory'](self.out, 1, item['run_id'])
        directory.mkdir(parents=True)
        payload = {'operation': 'generate', 'fixture': 1}
        write_json(directory/'request-001.json', payload)
        generated = {'input_tokens': 7, 'generated_tokens': 2, 'text': 'ordinary fixture'}
        identity = request_identity(item['run_id'], 1)
        with patch.object(MODULE['time'], 'time', return_value=timestamp):
            self.journal.execute(identity, payload, lambda _: generated)
        trace = {'started_epoch': timestamp-2, 'finished_epoch': timestamp+2, 'complete': completed,
                 'attempted_calls': 1 if completed else 3, 'backend': self.artifact,
                 'steps': [{'call': 1, 'generation': generated}]}
        trace_path = directory/'trajectory.json'
        write_json(trace_path, trace)
        if not completed:
            second = {'operation': 'generate', 'fixture': 2}
            write_json(directory/'request-002.json', second)
            interrupted = request_identity(item['run_id'], 2)
            claimed = self.journal.path(interrupted)
            claimed.mkdir()
            persist_new(claimed/'accepted.json', {'request_id': interrupted, 'state': 'accepted',
                        'payload_sha256': payload_sha256(second), 'accepted_epoch': timestamp+1})
            return None
        result = {key: item[key] for key in ('run_id', 'task_number', 'repetition', 'block')}
        result.update(contrast_arm=item['arm'], attempt=1, backend_index=1, attempted_calls=1,
                      timeout_call=None, accounting_drain_seconds=0, trajectory_sha256=sha256(trace_path))
        write_json(directory/'result.json', result)
        return result

    def restore(self, boot=('new-boot', 140), now=150):
        with patch.object(MODULE['time'], 'time', return_value=now):
            return MODULE['prepare_restart'](self.out, self.state, self.plan, 'b' * 64, self.order, boot)

    def test_reboot_keeps_prefix_and_identifies_partial_costs(self):
        self.actor(self.order[1], 120, completed=False)
        new = self.restore()
        self.assertEqual(new['runs'], self.state['runs'])
        self.assertEqual(new['deadline_epoch'], self.state['deadline_epoch'])
        self.assertEqual(new['attempt'], 2)
        self.assertEqual(new['backend_lifecycles'][0]['stopped_epoch'], 140)
        self.assertNotEqual(MODULE['actor_directory'](self.out, 1, 't2'),
                            MODULE['actor_directory'](self.out, 2, 't2'))
        request_ids, costs = MODULE['replay_interruptions'](self.out, new, self.order, 4)
        self.assertEqual(request_ids[1], {request_identity('t2', n) for n in (1, 2)})
        self.assertEqual(costs, {'reboots': 1, 'retained_prefixes': [1], 'attempted_calls': 3,
                                'journaled_requests': 2, 'finished_responses': 1, 'unknown_response_costs': 1,
                                'known_generated_tokens': 2, 'known_input_tokens': 7})
        (self.out/'telemetry.csv').write_text('different bytes\n')
        with self.assertRaisesRegex(ValueError, 'receipts differ'):
            MODULE['replay_interruptions'](self.out, new, self.order, 4)

    def test_committed_result_survives_before_controller_append(self):
        terminal = self.actor(self.order[1], 120)
        new = self.restore()
        self.assertEqual(new['runs'], self.state['runs'] + [terminal])
        self.assertTrue(new['restarts'][0]['recovered_committed_result'])
        receipts, costs = MODULE['replay_interruptions'](self.out, new, self.order, 4)
        self.assertFalse(receipts)
        self.assertEqual(costs['retained_prefixes'], [2])

    def test_rejects_same_boot_and_expired_deadline(self):
        with self.assertRaisesRegex(ValueError, 'original deadline'):
            self.restore(boot=('old-boot', 90))
        with self.assertRaisesRegex(ValueError, 'original deadline'):
            self.restore(now=601)

    def test_unaccounted_request_stops_recovery_before_the_next_task(self):
        (self.journal.directory/'unmatched-physical-request').mkdir()
        with self.assertRaisesRegex(ValueError, 'Unaccounted backend journal entries'):
            self.restore()


if __name__ == '__main__':
    unittest.main()
