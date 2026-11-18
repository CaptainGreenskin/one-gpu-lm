import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from one_gpu_lm.serving.client import request
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.serving.journaled_backend import make_server


class RequestJournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.journal = RequestJournal(self.root / 'receipts')
        self.ident = 'a' * 64

    def test_repeat_identity_never_regenerates_and_conflict_rejected(self):
        calls = []
        def generate(payload):
            calls.append(payload)
            return {'generated_tokens': 3, 'text': 'synthetic response'}
        expected = self.journal.execute(self.ident, {'limit': 3}, generate)
        self.assertEqual(self.journal.execute(self.ident, {'limit': 3}, generate), expected)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            self.journal.execute(self.ident, {'limit': 4}, generate)
        self.assertEqual(len(calls), 1)

    def test_interrupted_request_remains_unknown_and_cannot_retry(self):
        def interrupted(_):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.journal.execute(self.ident, {}, interrupted)
        receipt = self.journal.status(self.ident)
        self.assertTrue(receipt['unknown_generation_cost'])
        self.assertEqual(receipt['state'], 'in_flight_or_interrupted')
        with self.assertRaises(RuntimeError):
            self.journal.execute(self.ident, {}, lambda _: self.fail('must not regenerate'))

    def test_generation_exception_is_durable_and_never_retried(self):
        def failure(_):
            raise RuntimeError('private diagnostic must not be exposed')
        result = self.journal.execute(self.ident, {}, failure)
        self.assertEqual(result, {'error': 'RuntimeError', 'unknown_generation_cost': True})
        self.assertEqual(self.journal.execute(self.ident, {}, lambda _: self.fail()), result)
        self.assertNotIn('private diagnostic', json.dumps(self.journal.status(self.ident)))

    def test_identity_and_symlink_boundaries(self):
        for ident in ['../escape', '', 'A' * 64, None]:
            with self.assertRaises(ValueError):
                self.journal.status(ident)
        (self.journal.directory / self.ident).symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.journal.status(self.ident)

    def test_private_permissions_and_missing_receipt(self):
        self.assertEqual(self.journal.status(self.ident)['state'], 'not_received')
        self.journal.execute(self.ident, {}, lambda _: {'generated_tokens': 1})
        self.assertEqual(self.journal.directory.stat().st_mode & 0o777, 0o700)
        for p in (self.journal.directory / self.ident).glob('*.json'):
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)

    def test_socket_timeout_preserves_late_response_without_second_generation(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def slow(payload):
            calls.append(payload)
            entered.set()
            if not release.wait(3):
                raise TimeoutError('test release missing')
            return {'generated_tokens': 7, 'ids': list(range(7)), 'text': 'synthetic late output'}
        sock = self.root / 'backend.sock'
        server = make_server(sock, self.journal, slow, {'ready': True})
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        try:
            with self.assertRaises((TimeoutError, socket.timeout)):
                request(sock, {'operation': 'generate', 'request_id': self.ident, 'max_new_tokens': 7}, timeout=.05)
            self.assertTrue(entered.is_set())
            self.assertTrue(self.journal.status(self.ident)['unknown_generation_cost'])
            with self.assertRaises(RuntimeError):
                self.journal.execute(self.ident, {'operation': 'generate', 'max_new_tokens': 7}, slow)
            release.set()
            receipt = request(sock, {'operation': 'receipt', 'request_id': self.ident}, timeout=2)
            self.assertEqual(receipt['state'], 'completed')
            self.assertEqual(receipt['result']['generated_tokens'], 7)
            self.assertFalse(receipt['unknown_generation_cost'])
            result = request(sock, {'operation': 'generate', 'request_id': self.ident, 'max_new_tokens': 7})
            self.assertEqual(result, receipt['result'])
            self.assertEqual(len(calls), 1)
            self.assertTrue(request(sock, {'operation': 'health'})['ready'])
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(3)


if __name__ == '__main__':
    unittest.main()
