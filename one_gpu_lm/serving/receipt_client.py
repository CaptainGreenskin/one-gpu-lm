"""Recover a timed-out generation by receipt identity, without resending it."""
import hashlib
from pathlib import Path
import time

from .client import request
from .request_journal import RequestJournal, encoded


def request_identity(run_id, call):
    return hashlib.sha256(f'032-journal-v1\0{run_id}\0{call}'.encode()).hexdigest()


def payload_sha256(payload):
    return hashlib.sha256(encoded(payload)).hexdigest()


def generate_once(socket_path, journal_path, run_id, call, payload, timeout, recovery_seconds):
    request_id = request_identity(run_id, call)
    try:
        result = request(socket_path, dict(payload, request_id=request_id), timeout=timeout)
        return result, None
    except TimeoutError:
        stopped = time.monotonic()
        journal = RequestJournal(Path(journal_path))
        while True:
            receipt = journal.status(request_id)
            state = receipt['state']
            if state == 'completed':
                result = receipt['result']
                if (receipt['payload_sha256'] != payload_sha256(payload)
                        or receipt['unknown_generation_cost'] or result.get('error')):
                    raise RuntimeError('Invalid late generation receipt')
                return result, {'request_id': request_id, 'accounting_drain_seconds': time.monotonic()-stopped,
                                'accepted_epoch': receipt['accepted_epoch'], 'finished_epoch': receipt['finished_epoch']}
            if state == 'generation_error' or time.monotonic()-stopped >= recovery_seconds:
                raise RuntimeError('Timed-out generation has no complete cost receipt')
            time.sleep(.05)
