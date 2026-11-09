"""Run a frozen admitted task list once, with an explicit recovery demonstration."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from training_lab.records import sha256, write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'tasks', 'admission', 'protocol', 'socket', 'out']:
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--split', nargs='+', choices=['train','dev','test'], default=['train','dev','test'])
    p.add_argument('--recovery-task')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.out = args.out.resolve()
    receipt_path = args.out/'campaign.json'
    tasks = json.loads(args.tasks.read_text())['tasks']
    admission = json.loads(args.admission.read_text())
    if not admission['complete'] or admission['task_spec_sha256'] != sha256(args.tasks):
        raise ValueError('Admission is incomplete or does not match task content')
    approved = {t['id'] for t in admission['tasks'] if t['admitted']}
    selected = [t for t in tasks if t['id'] in approved and t['split'] in args.split]
    protocol = json.loads(args.protocol.read_text())
    if protocol.get('status') != 'frozen_before_model_rollouts':
        raise ValueError('Freeze protocol before model rollouts')
    if protocol['admission_sha256'] != sha256(args.admission):
        raise ValueError('Admission receipt differs from frozen protocol')
    code_root = Path(__file__).parent.parent
    relevant = ['repo_lab/'+name+'.py' for name in ['campaign','agent','backend','environment','grading','tasks']]
    relevant += ['training_lab/sandbox.py','training_lab/records.py']
    identity = {'tasks_sha256':sha256(args.tasks), 'admission_sha256':sha256(args.admission),
        'protocol_sha256':sha256(args.protocol), 'ids':[t['id'] for t in selected],
        'source_files':{name:sha256(code_root/name) for name in relevant}}
    if args.out.exists():
        if not args.resume:
            raise ValueError('Campaign exists; use explicit resume')
        receipt = json.loads(receipt_path.read_text())
        if receipt['identity'] != identity:
            raise ValueError('Cannot resume changed campaign')
    else:
        args.out.mkdir(parents=True)
        receipt = {'schema_version':1,'started_utc':utc(),'identity':identity,'results':[],'complete':False}
        write_json(receipt_path, receipt)
    for task in selected:
        out = args.out/'private'/task['id']
        result_path = out/'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            write_json(args.out/'status.json', {'stage':'task','id':task['id'],'completed':len(receipt['results']),'utc':utc()})
            command = [sys.executable,'-m','repo_lab.agent','--root',str(args.root.resolve()),
                '--tasks',str(args.tasks.resolve()),'--protocol',str(args.protocol.resolve()),
                '--socket',str(args.socket.resolve()),'--out',str(out),'--id',task['id']]
            out.mkdir(parents=True, exist_ok=True)
            if (out/'checkpoint.json').exists():
                command += ['--resume']
            elif task['id'] == args.recovery_task:
                command += ['--interrupt-after','2']
            with (out/'controller.log').open('a') as log:
                run = subprocess.run(command,stdout=log,stderr=log,timeout=1000)
                if run.returncode == 75:
                    command = command[:-2] + ['--resume']
                    run = subprocess.run(command,stdout=log,stderr=log,timeout=1000)
            if run.returncode != 0 or not result_path.exists():
                write_json(args.out/'status.json', {'stage':'controller_failure','id':task['id'],'returncode':run.returncode,'utc':utc()})
                raise RuntimeError('Task controller failed; inspect private logs, do not duplicate the task')
            result = json.loads(result_path.read_text())
        receipt['results'] = [r for r in receipt['results'] if r['id'] != result['id']] + [result]
        write_json(receipt_path, receipt)
        print(task['id'], 'passed' if result['passed'] else 'failed', 'calls',result['calls'], flush=True)
    receipt.update(complete=True,finished_utc=utc())
    write_json(receipt_path, receipt)
    write_json(args.out/'status.json', {'stage':'completed','completed':len(receipt['results']),
        'passed':sum(r['passed'] for r in receipt['results']),'utc':utc()})


if __name__ == '__main__':
    main()
