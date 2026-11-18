"""Own each journaled backend process and preserve the frozen load identity."""
import subprocess
import time
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.client import request
from .evaluation import Backend
from .tasks import relative


class JournalBackend(Backend):
    @property
    def journal_path(self):
        return self.out/'private/backends'/f'{self.number:04d}-{self.role}'/'requests'

    def ensure(self, role, deadline):
        if role not in {'base','student'}:raise ValueError('Unexpected weight role')
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError('Unexpected backend exit; do not silently reload and continue')
        if self.role==role and self.process is not None and self.process.poll() is None:
            self.check_exclusive()
            return self.socket
        self.stop()
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise ValueError('GPU occupied by another process; no competing load')
        self.number+=1;directory=self.out/'private/backends'/f'{self.number:04d}-{role}'
        directory.parent.mkdir(parents=True,exist_ok=True)
        args=[self.runtime['model_python'],'-m','one_gpu_lm.verified.resilient_backend',
              '--expected-environment',str(self.root/relative(self.protocol['environment_path'])),
              '--environment-receipt',str(directory.parent/f'{self.number:04d}-{role}-environment.json'),
              '--model',self.runtime['model'],
              '--profile',str(self.root/relative(self.protocol['evaluation_profile_path'])),
              '--socket',str(self.socket),'--out',str(directory)]
        if role=='student':
            adapter=self.root/relative(self.protocol['adapter_directory'])
            if {p.name:sha256(p) for p in adapter.iterdir() if p.is_file()}!=self.protocol['all_adapter_files_sha256']:
                raise ValueError('Candidate adapter changed before loading')
            args+=['--adapter',str(adapter)]
        self.entry={'index':self.number,'role':role,'started_epoch':time.time()}
        self.record.append(self.entry)
        self.process=subprocess.Popen(args,cwd=self.root,stdout=(directory.parent/f'{self.number:04d}-{role}.log').open('wb'),
                    stderr=subprocess.STDOUT,start_new_session=True)
        self.role=role;self.entry['pid']=self.process.pid
        self.changed()
        load_deadline=min(deadline,time.time()+600)
        while not (directory/'ready.json').exists():
            if self.process.poll() is not None:raise RuntimeError('Evaluation backend exited before readiness')
            if time.time()>=load_deadline:raise TimeoutError('Evaluation backend load cap exceeded')
            time.sleep(.5)
        self.entry['ready_epoch']=time.time()
        write_json(directory/'served-artifact.json',request(self.socket,{'operation':'health'}))
        self.check_exclusive()
        return self.socket

