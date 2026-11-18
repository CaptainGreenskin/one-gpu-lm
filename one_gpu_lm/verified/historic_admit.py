"""CPU semantic admission for historical fixes with per-file necessity checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback
from one_gpu_lm.common.receipts import sha256, write_json
from .historic_tasks import prepare, grade, validate_split
from .sandbox import Sandbox
from .tasks import digest


def semantic_failure(value):
    text = value['output']
    bad = ['ModuleNotFoundError:', 'ImportError:', 'SyntaxError:', 'command not found',
           'ERR_MODULE_NOT_FOUND', 'Cannot find module', 'unrecognized arguments:']
    # An assertion or an exception from the regression can be a real failure;
    # collection/runtime failures are not evidence of the intended bug.
    return (value['reason'] == 'runtime_error' and value['returncode'] == 1
            and not any(s in text for s in bad)
            and ('failed' in text.lower() or 'not ok' in text.lower()))


def run(data, runtime_map, out):
    data, out = Path(data), Path(out)
    if out.exists():raise ValueError('Preserve every CPU admission attempt')
    out.mkdir(parents=True)
    sources = json.loads((data/'sources.json').read_text())['sources']
    tasks = json.loads((data/'tasks.json').read_text())['tasks']
    validate_split(sources)
    by_id = {s['source_id']:s for s in sources}
    runtime_paths = json.loads(Path(runtime_map).read_text())
    runners = {repo:Sandbox(json.loads(Path(p).read_text()),60) for repo,p in runtime_paths.items()}
    result = {'schema_version':1,'started_utc':datetime.now(timezone.utc).isoformat(),
              'complete':False,'model_calls':0,'final_test_opened':False,'rows':[],
              'tasks_sha256':sha256(data/'tasks.json'),'sources_sha256':sha256(data/'sources.json'),
              'runtime_sha256':{r:sha256(p) for r,p in runtime_paths.items()},
              'reference_replays_are_model_solutions':False,
              'grading_checks_are_unseen':False}
    def save():write_json(out/'admission.json',result)
    save()
    for task in tasks:
        row={'id':task['id'],'repository':task['repository'],'source_id':task['source_id'],
             'split':task['split'],'admitted':False,'multifile':False,'essential_files':[],
             'checks':{},'source_files':len(task['editable_files'])}
        d=out/task['id'];d.mkdir()
        source=by_id[task['source_id']];snapshot=data/'snapshots'/task['source_id']
        sandbox=runners[task['repository']]
        def execute(name,workspace):
            value=sandbox.execute(workspace,task['regression_command'])
            write_json(d/(name+'.json'),value)
            row['checks'][name]={k:v for k,v in value.items() if k!='output'}
            row['checks'][name]['output_sha256']=digest(value['output'].encode())
            return value
        try:
            prepare(snapshot,d/'fixed',source,task,broken=False)
            fixed=execute('fixed',d/'fixed')
            if fixed['returncode'] != 0:
                row['reason']='reference_suite_did_not_pass'
            else:
                before=prepare(snapshot,d/'parent',source,task)
                parent=execute('parent',d/'parent')
                if not semantic_failure(parent):
                    row['reason']='parent_did_not_reproduce_semantic_failure'
                else:
                    for index,mutation in enumerate(task['mutations']):
                        file=mutation['path'];dest=d/f'ablation-{index}'
                        prepare(snapshot,dest,source,task,broken=False)
                        target=dest/file
                        text=target.read_text()
                        if text.count(mutation['before']) != 1:raise ValueError('Ablation mismatch')
                        target.write_text(text.replace(mutation['before'],mutation['after']))
                        value=execute(f'ablation-{index}',dest)
                        if semantic_failure(value):row['essential_files'].append(file)
                    row['multifile']=len(row['essential_files'])>=2
                    # The same fresh protected submission boundary used by the actor.
                    for n in task['editable_files']:shutil.copyfile(snapshot/n,d/'parent'/n)
                    restored=grade(snapshot,d/'regrade',source,task,d/'parent',before,
                                   lambda path,command:execute('fresh_regrade',path))
                    row['regrade']=restored
                    row['admitted']=restored['passed']
                    row['reason']='ok' if row['admitted'] else 'fresh_regrade_failed'
        except Exception as error:
            row['reason']='admission_error';row['error_type']=type(error).__name__
            (d/'private-error.log').write_text(traceback.format_exc())
        result['rows'].append(row);save()
        print(json.dumps({k:row[k] for k in ['id','admitted','multifile','reason']}),flush=True)
    result.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),
                  admitted=sum(r['admitted'] for r in result['rows']),
                  multifile=sum(r['admitted'] and r['multifile'] for r in result['rows']))
    save();return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['data','runtime-map','out']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.data,a.runtime_map,a.out)
