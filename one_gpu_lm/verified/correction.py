"""One preregistered paired training-only collection; no retry or adaptive sampling."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.verify import verify_model
from .correction_diagnosis import failed_ids
from .correction_policy import correction_chain
from .correction_rollout import initialize, advance, finish
from .evaluation import Backend
from .evaluation_rollout import atomic
from .sandbox import Sandbox
from .tasks import relative, validate_source


def select_tasks(tasks, parent_audit, count=24):
    diagnostic = failed_ids(parent_audit); excluded = {r['id'] for r in parent_audit['rows']}
    groups = defaultdict(list)
    for task in tasks:
        if task['split'] != 'training': raise ValueError('Training only')
        if task['id'] not in excluded: groups[task['repository']].append(task['id'])
    for names in groups.values():
        names.sort(key=lambda n: hashlib.sha256(('027-correction-sampling-v1\0'+n).encode()).hexdigest())
    supplemental = []; i = 0
    while len(supplemental) < count:
        added = False
        for repo in sorted(groups):
            if i < len(groups[repo]) and len(supplemental) < count:
                supplemental.append(groups[repo][i]); added = True
        if not added: raise ValueError('Insufficient unselected training tasks')
        i += 1
    if not set(diagnostic) <= {t['id'] for t in tasks}: raise ValueError('Unknown diagnostic training task')
    return diagnostic, supplemental


def schedule(ids):
    return [{'task_id': name, 'arm': arm, 'run_id': name+'--'+arm}
            for i,name in enumerate(ids) for arm in (('control','correction') if i%2 == 0 else ('correction','control'))]


def run(root, protocol_path):
    root,protocol_path=Path(root).resolve(),Path(protocol_path).resolve()
    p=json.loads(protocol_path.read_text());out=root/relative(p['run_directory'])
    if out.exists(): raise ValueError('Preserve prior attempt; no implicit retry')
    public=json.loads((root/relative(p['publication_receipt'])).read_text())
    if not public.get('anonymous_files_verified') or public.get('ci_conclusion')!='success' or public['protocol_sha256']!=sha256(protocol_path):
        raise ValueError('Public executable protocol and CI verification required')
    for name,h in p['frozen_files'].items():
        if sha256(root/relative(name))!=h: raise ValueError('Frozen input changed: '+name)
    data=root/relative(p['data_directory']);tasks=json.loads((data/'tasks.json').read_text())['tasks']
    sources=json.loads((data/'sources.json').read_text())['sources']
    diagnostic,supplemental=select_tasks(tasks,json.loads((root/relative(p['parent_audit'])).read_text()))
    ids=diagnostic+supplemental
    if (diagnostic!=p['diagnostic_task_ids'] or supplemental!=p['supplemental_task_ids']
            or schedule(ids)!=p['schedule']): raise ValueError('Task selection or schedule changed')
    by_id={t['id']:t for t in tasks};by_repo={s['repository']:s for s in sources}
    for source in sources: validate_source(source,data/'snapshots'/source['repository'].replace('/','--'),'training')
    runtime=json.loads((root/'private/parent-runtime.json').read_text())
    runner=Sandbox(json.loads((root/'private/task-runtime.json').read_text()),p['evaluation_budget']['max_tool_seconds'])
    out.mkdir(parents=True);(out/'private').mkdir();started=time.time();deadline=started+p['maximum_pipeline_seconds']
    state={'schema_version':1,'stage':'verify_model','complete':False,'started_epoch':started,'deadline_epoch':deadline,
           'started_utc':datetime.now(timezone.utc).isoformat(),'runs':[],'backend_lifecycles':[],
           'protocol_sha256':sha256(protocol_path),'publication_commit':public['commit'],'expected_runs':len(p['schedule']),
           'optimizer_updates':0,'final_test_opened':False,'weights_public':False}
    def save(): atomic(out/'controller.json',state)
    save();telemetry=Telemetry(out/'telemetry.csv',1);telemetry.start()
    backend=Backend(root,out,runtime,p,state['backend_lifecycles'],save)
    try:
        profile=json.loads((root/relative(p['evaluation_profile_path'])).read_text())
        write_json(out/'model-verification.json',verify_model(Path(runtime['model']),root/relative(p['model_manifest_path']),profile))
        for index,item in enumerate(p['schedule']):
            reserve=(2 if index%2 == 0 else 1)*(p['evaluation_budget']['max_task_seconds']+120)+(600 if backend.role!='base' else 0)
            if time.time()+reserve>deadline:
                state['stage']='pipeline_budget_exhausted';break
            state.update(stage='loading_base',current_run=item);save();socket=backend.ensure('base',deadline)
            state['stage']='collecting';save();task=by_id[item['task_id']];directory=out/'private/tasks'/item['run_id']
            trace=initialize(data,task,by_repo[task['repository']],p,directory,item['arm'])
            advance(root,task,p,directory,trace,socket,runner,item['arm']);backend.check_exclusive()
            result=finish(data,task,by_repo[task['repository']],p,directory,trace,runner)
            result.update(run_id=item['run_id'],arm=item['arm'],cohort='diagnostic' if task['id'] in diagnostic else 'supplemental',
                          correction_behavior=correction_chain(trace,task))
            write_json(directory/'result.json',result);state['runs'].append(result);save()
            if (sum(r['attempted_calls'] for r in state['runs'])>p['global_caps']['model_calls']
                    or sum(r['generated_tokens'] for r in state['runs'])>p['global_caps']['generated_tokens']):
                raise ValueError('Global generation cap exceeded')
            print(json.dumps({'completed_runs':len(state['runs']),'run_id':item['run_id'],
                              'success':result['success_within_budget'],'correction_behavior':result['correction_behavior']}),flush=True)
        if len(state['runs'])==len(p['schedule']): state.update(stage='completed',complete=True)
    except BaseException as error:
        state.update(stage='failed',error_type=type(error).__name__);(out/'private/error.log').write_text(traceback.format_exc())
    finally:
        backend.stop();state['telemetry']=telemetry.stop()
        state.update(finished_epoch=time.time(),finished_utc=datetime.now(timezone.utc).isoformat(),backend_stopped=True,
                     frozen_files_unchanged=all(sha256(root/relative(n))==h for n,h in p['frozen_files'].items()))
        if not state['frozen_files_unchanged']: state.update(complete=False,stage='frozen_input_changed')
        save()
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.protocol)
