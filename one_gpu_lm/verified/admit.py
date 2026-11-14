"""CPU-only semantic and isolation admission. Reference restoration is not teacher data."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import traceback
from .tasks import prepare,grade,validate_split,digest,files
from .sandbox import Sandbox
from one_gpu_lm.common.receipts import write_json


def run(data,runtime_path,out):
    data,out=Path(data),Path(out)
    if out.exists():raise ValueError('Preserve every admission attempt')
    out.mkdir(parents=True)
    source_rows=json.loads((data/'sources.json').read_text())['sources']
    task_rows=json.loads((data/'tasks.json').read_text())['tasks']
    validate_split(source_rows)
    if not task_rows or len({t['id'] for t in task_rows})!=len(task_rows):raise ValueError('Empty/duplicate task inventory')
    sources={s['repository']:s for s in source_rows}
    sandbox=Sandbox(json.loads(Path(runtime_path).read_text()))
    result={'schema_version':1,'started_utc':datetime.now(timezone.utc).isoformat(),'complete':False,'tasks':[],
            'model_generations':0,'reference_restoration_is_teacher_data':False,'final_test_opened':False,
            'tasks_sha256':digest((data/'tasks.json').read_bytes()),'sources_sha256':digest((data/'sources.json').read_bytes()),
            'runtime_sha256':digest(Path(runtime_path).read_bytes()),'implementation_sha256':files(Path(__file__).parent)}
    def save():write_json(out/'admission.json',result)
    save()
    for task in task_rows:
        source=sources[task['repository']];snapshot=data/'snapshots'/task['repository'].replace('/','--')
        destination=out/task['id'];destination.mkdir()
        row={'id':task['id'],'repository':task['repository'],'split':task['split'],'admitted':False}
        try:
            prepare(snapshot,destination/'clean',source,task,broken=False,hidden=True)
            clean=sandbox.execute(destination/'clean',task['grading_command'])
            before=prepare(snapshot,destination/'fault-visible',source,task)
            assert not any((destination/'fault-visible'/n).exists() for n in task['hidden_checks'])
            fault_visible=sandbox.execute(destination/'fault-visible',task['regression_command'])
            prepare(snapshot,destination/'fault-hidden',source,task,hidden=True)
            hidden_command=task['grading_command'].split(' && ')[-1]
            fault_hidden=sandbox.execute(destination/'fault-hidden',hidden_command)
            for name,value in [('clean',clean),('fault-visible',fault_visible),('fault-hidden',fault_hidden)]:
                (destination/(name+'.log')).write_text(value['output'])
                row[name]={'returncode':value['returncode'],'reason':value['reason'],'seconds':value['seconds'],
                           'output_sha256':digest(value['output'].encode())}
            for n in task['editable_files']:shutil.copyfile(snapshot/n,destination/'fault-visible'/n)
            restored=grade(snapshot,destination/'restored-grader',source,task,destination/'fault-visible',before,sandbox.execute)
            row['reference_restoration']=restored
            row['hidden_checks_absent_from_agent']=True
            row['admitted']=clean['returncode']==0 and fault_visible['returncode']!=0 and fault_hidden['returncode']!=0 and restored['passed']
            # Timeouts/missing runtimes are not evidence of a meaningful semantic fault.
            row['admitted'] &= fault_visible['reason']=='runtime_error' and fault_hidden['reason']=='runtime_error'
            if any(needle in (fault_visible['output']+fault_hidden['output']) for needle in ['SyntaxError:','ModuleNotFoundError:','command not found']):row['admitted']=False
        except Exception as error:
            row['error_type']=type(error).__name__
            (destination/'private-error.log').write_text(traceback.format_exc())
        result['tasks'].append(row);save()
        print(json.dumps({'task':task['id'],'admitted':row['admitted']}),flush=True)
    result['complete']=True;result['passed']=all(r['admitted'] for r in result['tasks'])
    result['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps({'complete':True,'admitted':sum(r['admitted'] for r in result['tasks']),'tasks':len(task_rows),'model_generations':0}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.data,a.runtime,a.out)
