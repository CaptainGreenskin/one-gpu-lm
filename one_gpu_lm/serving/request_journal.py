"""Private, durable generation receipts; an interrupted request is never retried.

This is an opt-in backend component. It does not change any frozen experiment.
Completed responses are persisted before transport. A process killed during
generation still has unknown cost; a journal cannot reconstruct missing tokens.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


def persist_new(path, value):
    """Publish a fully fsynced file without replacing an existing receipt."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.unlink(temporary)


class RequestJournal:
    def __init__(self, directory):
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError('Journal must not be a symlink')
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory.chmod(0o700)

    def path(self, request_id):
        if not isinstance(request_id, str) or not re.fullmatch('[0-9a-f]{64}', request_id):
            raise ValueError('Expected a lowercase SHA256 request identity')
        path = self.directory / request_id
        if path.is_symlink():
            raise ValueError('Request directory must not be a symlink')
        return path

    def status(self, request_id):
        path = self.path(request_id)
        if not path.exists():
            return {'state': 'not_received', 'request_id': request_id}
        accepted = path / 'accepted.json'
        finished = path / 'finished.json'
        if not accepted.exists():
            return {'state': 'in_flight_or_interrupted', 'request_id': request_id,
                    'unknown_generation_cost': True}
        receipt = json.loads(accepted.read_text())
        if finished.exists():
            receipt.update(json.loads(finished.read_text()))
        else:
            receipt.update(state='in_flight_or_interrupted', unknown_generation_cost=True)
        return receipt

    def execute(self, request_id, payload, generate):
        path = self.path(request_id)
        payload_hash = hashlib.sha256(encoded(payload)).hexdigest()
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            receipt = self.status(request_id)
            if receipt.get('payload_sha256') != payload_hash:
                raise ValueError('Request identity conflict or unfinished acceptance')
            if receipt['state'] == 'completed':
                return receipt['result']
            if receipt['state'] == 'generation_error':
                return {'error': receipt['error_type'], 'unknown_generation_cost': True}
            raise RuntimeError('Request in flight or interrupted; no implicit retry')
        persist_new(path / 'accepted.json', {
            'schema_version': 1, 'request_id': request_id, 'payload_sha256': payload_hash,
            'accepted_epoch': time.time(), 'state': 'accepted',
        })
        try:
            result = generate(payload)
            encoded(result)
        except Exception as error:
            persist_new(path / 'finished.json', {
                'state': 'generation_error', 'error_type': type(error).__name__,
                'finished_epoch': time.time(), 'unknown_generation_cost': True,
            })
            return {'error': type(error).__name__, 'unknown_generation_cost': True}
        persist_new(path / 'finished.json', {
            'state': 'completed', 'result': result, 'finished_epoch': time.time(),
            'unknown_generation_cost': False,
        })
        return result
