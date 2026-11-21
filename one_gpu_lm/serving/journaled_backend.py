"""Opt-in local backend that persists generation results before socket delivery.

Actor deadlines remain the caller's responsibility. Recovering a late receipt
is accounting only: it must not extend the actor budget or execute a late tool.
"""
import argparse
import json
import os
from pathlib import Path
import socketserver
from .client import MAX_REQUEST_BYTES
from .request_journal import RequestJournal


def make_server(socket_path, journal, generate, identity):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
                if len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b'\n'):
                    raise ValueError('Invalid request size')
                value = json.loads(raw)
                operation = value.get('operation')
                if operation == 'health':
                    result = dict(identity, request_journal_version=1)
                elif operation == 'receipt':
                    result = journal.status(value['request_id'])
                elif operation == 'generate':
                    request_id = value.pop('request_id')
                    result = journal.execute(request_id, value, generate)
                else:
                    raise ValueError('Unregistered operation')
            except Exception as error:
                result = {'error': type(error).__name__}
            try:
                self.wfile.write(json.dumps(result, allow_nan=False).encode() + b'\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The completed response is already durable. Never regenerate.
                pass
    server = socketserver.UnixStreamServer(str(socket_path), Handler)
    Path(socket_path).chmod(0o600)
    return server


def main():
    from one_gpu_lm.common.receipts import sha256, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'profile', 'socket', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    args = parser.parse_args()
    if args.socket.exists() or args.out.exists():
        raise ValueError('Preserve existing backend and receipts')
    os.umask(0o077)
    args.out.mkdir(parents=True, mode=0o700)
    profile = json.loads(args.profile.read_text())
    if profile.get('generation_modes'):
        from one_gpu_lm.models.reasoning_runtime import Runtime
    else:
        from one_gpu_lm.models.hf_runtime import Runtime
    runtime = Runtime(args.model, profile, args.out, args.adapter)
    identity = {'ready': True, 'model_id': profile['model_id'], 'revision': profile['revision'],
                'profile_sha256': sha256(args.profile), 'adapter': bool(args.adapter),
                'adapter_sha256': {p.name: sha256(p) for p in sorted(args.adapter.glob('*.safetensors'))}
                if args.adapter else {}}
    journal = RequestJournal(args.out / 'requests')
    with make_server(args.socket, journal, runtime.generate, identity) as server:
        write_json(args.out / 'ready.json', dict(identity, request_journal_version=1))
        server.serve_forever(poll_interval=0.5)


if __name__ == '__main__':
    main()
