"""Offline Linux command runner with an explicit runtime file inventory.

No host /usr tree, home, package manager, model or reference snapshot is mounted.
This remains a single-user experiment harness, not a hostile multi-tenant service.
"""
from contextlib import nullcontext
import os
from pathlib import Path
import resource
import signal
import subprocess
import tempfile
import time
from .tasks import digest


class Sandbox:
    def __init__(self, runtime, timeout=60):
        self.runtime = runtime
        self.timeout = timeout
        self.mounts = runtime['mounts']
        self.mount_roots = runtime.get('mount_roots', [])
        destinations = set()
        for m in self.mounts:
            destination = Path(m['destination'])
            if not destination.is_absolute() or '..' in destination.parts or str(destination) in destinations:
                raise ValueError('Invalid or duplicated runtime destination')
            if not str(destination).startswith(('/usr/bin/', '/lib/', '/lib64/', '/opt/')):
                raise ValueError('Unexpected runtime destination')
            destinations.add(str(destination))
            source = Path(m['source'])
            if not source.is_file() or digest(source.read_bytes()) != m['sha256']:
                raise ValueError('Runtime file content changed')

        if self.mount_roots:
            allowed = {'/usr/bin','/lib','/lib64','/opt/python','/opt/node'}
            if len({r['destination'] for r in self.mount_roots}) != len(self.mount_roots):
                raise ValueError('Duplicate mount root')
            grouped = {}
            for r in self.mount_roots:
                if r['destination'] not in allowed:
                    raise ValueError('Unreviewed mount root')
                source = Path(r['source'])
                if not source.is_dir() or source.is_symlink():
                    raise ValueError('Invalid runtime directory')
                for p in source.rglob('*'):
                    if p.is_symlink():raise ValueError('Runtime directory symlink')
                    if p.is_file():
                        destination = r['destination']+'/'+str(p.relative_to(source))
                        grouped[destination] = (str(p.resolve()),digest(p.read_bytes()))
            expected = {m['destination']:(str(Path(m['source']).resolve()),m['sha256']) for m in self.mounts}
            if grouped != expected:
                raise ValueError('Grouped mounts expose a different file inventory')

    def execute(self, workspace, command):
        args = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
                '--clearenv', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/usr/bin',
                '--symlink', 'usr/bin', '/bin', '--bind', str(Path(workspace).resolve()), '/work', '--chdir', '/work']
        for mount in (self.mount_roots or self.mounts):
            args += ['--ro-bind', mount['source'], mount['destination']]
        for key, value in self.runtime['environment'].items():
            args += ['--setenv', key, str(value)]
        loopback_hosts = self.runtime.get('policy', {}).get('loopback_hosts') is True
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout + 1))
            resource.setrlimit(resource.RLIMIT_AS, (16 * 2**30,) * 2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 2**20,) * 2)
            resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
            resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        started = time.monotonic()
        with (tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr,
              (tempfile.TemporaryFile() if loopback_hosts else nullcontext()) as hosts):
            pass_fds = ()
            if hosts is not None:
                hosts.write(b'127.0.0.1 localhost\n::1 localhost\n')
                hosts.seek(0)
                pass_fds = (hosts.fileno(),)
                args += ['--ro-bind-data', str(hosts.fileno()), '/etc/hosts']
            args += ['--', '/bin/bash', '-c', command]
            p = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 preexec_fn=limits, start_new_session=True, pass_fds=pass_fds)
            try:
                p.wait(timeout=self.timeout + 2)
                reason = 'ok' if p.returncode == 0 else 'runtime_error'
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                reason = 'timeout'
            stdout.seek(0); stderr.seek(0)
            raw = stdout.read(2**20 + 1); error = stderr.read(2**20 + 1)
            if max(len(raw), len(error)) > 2**20:
                reason = 'output_limit'
            output = raw[:2**20].decode(errors='replace')
            if error:
                output += '\nSTDERR:\n' + error[:2**20].decode(errors='replace')
            return {'returncode': p.returncode if reason not in {'timeout', 'output_limit'} else -1,
                    'reason': reason, 'output': output, 'seconds': time.monotonic() - started}
