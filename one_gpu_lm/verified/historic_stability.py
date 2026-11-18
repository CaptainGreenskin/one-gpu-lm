"""CPU-only repeatability screen before any development model outcome exists."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from one_gpu_lm.common.receipts import sha256, write_json
from .historic_admit import semantic_failure
from .historic_tasks import prepare
from .sandbox import Sandbox


def run(data, runtime_map, out, per_family=6):
    data,out=Path(data),Path(out)
    if out.exists():raise ValueError('Preserve CPU stability attempts')
    out.mkdir(parents=True)
    tasks=json.loads((data/'tasks.json').read_text())['tasks']
    sources={s['source_id']:s for s in json.loads((data/'sources.json').read_text())['sources']}
    admitted={r['id']:r for r in json.loads((data/'admission.json').read_text())['rows']}
    runtimes=json.loads(Path(runtime_map).read_text())
    result={'complete':False,'model_calls':0,'final_test_opened':False,'started_epoch':time.time(),
            'repeat_count':2,'prior_semantic_checks':1,'rows':[],
            'tasks_sha256':sha256(data/'tasks.json'),'sources_sha256':sha256(data/'sources.json'),
            'runtime_sha256':{r:sha256(Path(path)) for r,path in runtimes.items()}}
    lock=threading.Lock()
    def family(repo):
        sandbox=Sandbox(json.loads(Path(runtimes[repo]).read_text()),60);chosen=0
        candidates=[t for t in tasks if t['repository']==repo]
        for task in candidates:
            if chosen>=per_family:break
            if task['split']!='development' or not admitted[task['id']]['admitted']:
                raise ValueError('Only CPU-admitted development candidates')
            row={'id':task['id'],'repository':repo,'stable':True,'checks':[]}
            for rep in range(2):
                for broken in [False,True]:
                    label=('parent' if broken else 'fixed')+'-'+str(rep)
                    directory=out/task['id']/label
                    prepare(data/'snapshots'/task['source_id'],directory,sources[task['source_id']],task,broken=broken)
                    value=sandbox.execute(directory,task['regression_command'])
                    receipt=out/task['id']/(label+'.json');write_json(receipt,value)
                    passed=semantic_failure(value) if broken else value['returncode']==0 and value['reason']=='ok'
                    row['checks'].append({'label':label,'expected_result_observed':passed,
                        'output_sha256':sha256(receipt),'seconds':value['seconds']})
                    row['stable']=row['stable'] and passed
            if row['stable']:chosen+=1
            row['selected']=row['stable']
            with lock:
                result['rows'].append(row);write_json(out/'status.json',result)
        return {'repository':repo,'selected':chosen,'required':per_family}
    write_json(out/'status.json',result)
    with ThreadPoolExecutor(max_workers=2) as pool:
        coverage=list(pool.map(family,sorted({t['repository'] for t in tasks})))
    result.update(complete=True,finished_epoch=time.time(),coverage=coverage,
                  passed=all(r['selected']==per_family for r in coverage))
    write_json(out/'status.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['data','runtime-map','out']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args.data,args.runtime_map,args.out)['coverage']))
