"""Real Linux sandbox snapshot isolation and timeout checks, without model calls."""
import argparse
import json
from pathlib import Path
import shlex
from .recovery_state import capture, restore, inventory, digest
from .sandbox import Sandbox
from one_gpu_lm.common.receipts import write_json


def run(runtime, out):
    out = Path(out)
    if out.exists(): raise ValueError('Preserve prior CPU check')
    out.mkdir(parents=True); actor = out/'actor'; actor.mkdir()
    sandbox = Sandbox(runtime, 5)
    code = "from pathlib import Path; import os; p=Path('state'); p.write_bytes(b'fixture'); p.chmod(0o640); os.utime(p,ns=(1234567890000000000,1234567890000000000)); Path('empty').mkdir()"
    created = sandbox.execute(actor, 'python3 -c '+shlex.quote(code))
    if created['returncode'] != 0: raise RuntimeError('CPU fixture could not start in actual sandbox')
    source = {'state': digest(b'fixture')}
    prefix = {'attempted_calls':3, 'steps':[{'source_before':source, 'source_after':source, 'generation':{'generated_tokens':2}} for _ in range(3)],
        'generated_tokens':6, 'actor_seconds':1., 'boundary':'completed_step', 'submitted':False, 'terminal_reason':None,
        'messages':[{'role':r, 'content':'CPU fixture only'} for r in ['system','user','assistant','user','assistant','user','assistant','user']]}
    receipt = capture(actor, out/'snapshot', prefix)
    for name in ['student','teacher']: restore(out/'snapshot', out/name, receipt)
    identical = inventory(out/'student') == inventory(out/'teacher') == inventory(actor)
    changed = sandbox.execute(out/'student', "python3 -c \"from pathlib import Path; Path('state').write_text('changed')\"")
    independent = (out/'teacher/state').read_bytes() == (actor/'state').read_bytes() == b'fixture'
    probe = "from pathlib import Path; import socket; assert not Path('/home').exists(); assert not Path('/work/hidden.py').exists(); s=socket.socket(); assert s.connect_ex(('127.0.0.1',22)) != 0"
    isolated = sandbox.execute(out/'teacher', 'python3 -c '+shlex.quote(probe))
    failed = sandbox.execute(out/'teacher', "python3 -c 'raise SystemExit(7)'")
    sandbox.timeout = 1
    timed = sandbox.execute(out/'teacher', "python3 -c 'import time; time.sleep(5)'")
    checks = {'identical_restored_state':identical, 'independent_branch_writes':independent and changed['returncode']==0,
              'host_and_hidden_content_absent':isolated['returncode']==0, 'failed_command_retained':failed['returncode']==7,
              'tool_timeout_retained':timed['reason']=='timeout'}
    result = {'schema_version':1, 'checks':checks, 'passed':all(checks.values()), 'model_calls':0,
              'optimizer_updates':0, 'final_test_opened':False, 'fixture_not_training_data':True}
    write_json(out/'result.json', result); return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--runtime',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); print(json.dumps(run(json.loads(args.runtime.read_text()),args.out)))
