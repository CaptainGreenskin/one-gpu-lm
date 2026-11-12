"""Use the unchanged agent; intercept only its already-capped tool output."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.evaluation import agent as shared
from one_gpu_lm.evaluation.environment import RepositoryEnvironment
from one_gpu_lm.common.receipts import write_json,sha256
from normalize import transform,digest

def recorded_environment(out,condition):
    out=Path(out).resolve();path=out/'observations.jsonl'
    class RecordedEnvironment(RepositoryEnvironment):
        def execute(self,action,cwd=''):
            value=super().execute(action,cwd)
            is_agent=self.workspace==out/'workspace'
            if not is_agent:return value  # Independent grader always sees raw output.
            original=value['output'];delivered,receipt=transform(original,condition,self.output_chars)
            receipt.update(call_index=self.calls,command_sha256=digest(action['command']),
                           returncode=value['returncode'],reason=value['reason'])
            with path.open('a') as stream:
                stream.write(json.dumps({'receipt':receipt,'command':action['command'],
                    'raw_output':original,'delivered_output':delivered})+'\n')
                stream.flush();os.fsync(stream.fileno())
            return dict(value,output=delivered,normalization=receipt)
    return RecordedEnvironment

def main():
    args=sys.argv[1:];out=Path(args[args.index('--out')+1]).resolve()
    condition=args[args.index('--variant')+1]
    if condition not in {'raw','canonical'}:raise ValueError('Unknown output condition')
    out.mkdir(parents=True,exist_ok=True);path=out/'observations.jsonl'
    shared.RepositoryEnvironment=recorded_environment(out,condition)
    shared.main()
    p=out/'result.json';result=json.loads(p.read_text())
    result['output_condition']=condition
    result['normalizer_sha256']=sha256(Path(__file__).with_name('normalize.py'))
    result['observation_receipts']=[json.loads(s)['receipt'] for s in path.read_text().splitlines()] if path.exists() else []
    if len(result['observation_receipts'])!=result['tool_calls']:raise ValueError('Incomplete observation journal')
    write_json(p,result)

if __name__=='__main__':main()
