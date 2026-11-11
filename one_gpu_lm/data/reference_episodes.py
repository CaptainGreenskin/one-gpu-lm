"""Execute reference repair episodes; retain real tool turns and independent grades.

This is labeled reference supervision, not a model or hidden-reasoning trace.
No test/evaluation partition is accepted. Raw records remain private.
"""
import argparse
from collections import Counter
import copy
import difflib
import json
from pathlib import Path
import shlex

from one_gpu_lm.common.actions import INSTANCE, parse_action
from one_gpu_lm.common.agent_policy import NATIVE_SYSTEM
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.evaluation.environment import RepositoryEnvironment, repository_python_site
from one_gpu_lm.evaluation.grading import generated_files, grade
from one_gpu_lm.evaluation.tasks import file_manifest, prepare


def repair_commands(task, workspace, snapshot):
    patch, before, after = [], [], []
    for name in task['editable_files']:
        broken=(workspace/name).read_text().splitlines(keepends=True)
        good=(snapshot/name).read_text().splitlines(keepends=True)
        edits=[op for op in difflib.SequenceMatcher(a=broken,b=good,autojunk=False).get_opcodes() if op[0]!='equal']
        if not edits:raise ValueError('Reference has no repair')
        patch.extend(difflib.unified_diff(broken,good,fromfile='a/'+name,tofile='b/'+name,n=0))
        start=max(1,min(op[1] for op in edits)-5)
        before.append(f"sed -n '{start},{min(len(broken),max(op[2] for op in edits)+5)}p' "+shlex.quote(name))
        after.append(f"sed -n '{start},{min(len(good),max(op[4] for op in edits)+5)}p' "+shlex.quote(name))
    body=''.join(patch)
    marker='ONE_GPU_REFERENCE_PATCH'
    if marker in body:raise ValueError('Ambiguous patch delimiter')
    return [('inspect', '\n'.join(before),0),('reproduce',task['regression_command'],1),
            ('repair',"patch -p1 --batch <<'"+marker+"'\n"+body+marker,0),
            ('review','\n'.join(after),0),('verify',task['regression_command'],0),
            ('submit','echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT',0)]


def build(root, tasks_path, admission_path, licenses_path, out):
    root=Path(root).resolve();out=Path(out)
    if out.exists():raise ValueError('Preserve previous dataset attempts')
    all_tasks=json.loads(Path(tasks_path).read_text())['tasks']
    tasks=[t for t in all_tasks if t['split']=='train']
    admission=json.loads(Path(admission_path).read_text())
    licenses=json.loads(Path(licenses_path).read_text())
    if not admission['complete'] or admission['task_spec_sha256']!=sha256(tasks_path):
        raise ValueError('Training task list differs from complete admission')
    if not tasks or any(t['split']!='train' for t in tasks):
        raise ValueError('Only training partitions may enter reference supervision')
    approved={t['id'] for t in admission['tasks'] if t['admitted'] and t['split']=='train'}
    allowed={(s['repository'],s['revision']) for s in licenses['sources'] if s['split']=='train' and s.get('license_sha256') and s.get('rights_review',{}).get('status')}
    if any((t['repository'],t['revision']) not in allowed or t['id'] not in approved for t in tasks):
        raise ValueError('Unadmitted or unreviewed training source')
    from jinja2 import Template
    out.mkdir(parents=True,mode=0o700);(out/'private').mkdir(mode=0o700)
    runtime=json.loads((root/'runtime.json').read_text());rows=[];controls=[]
    for task in tasks:
        slug=task['repository'].replace('/','--');snapshot=root/'sources'/slug
        private=out/'private'/task['id'];private.mkdir()
        workspace=prepare(snapshot,private/'workspace',task)
        original=file_manifest(workspace,exclude=generated_files(task['repository']))
        deps=root/'node-deps'/slug/'node_modules'
        def make_env(work):
            return RepositoryEnvironment(work,python_site=repository_python_site(root,runtime,task['repository']),
                node_root=runtime['node_root'],node_modules=deps if deps.exists() else None,output_chars=2400)
        env=make_env(workspace);progress=EditableProgress(workspace,original,task['editable_files'])
        prompt=Template(INSTANCE).render(task=task['description'],editable_files=', '.join(task['editable_files']),regression_command=task['regression_command'])
        messages=[{'role':'system','content':NATIVE_SYSTEM},{'role':'user','content':prompt}]
        examples=[];commands=repair_commands(task,workspace,snapshot)
        for index,(kind,command,expected) in enumerate(commands):
            target='<tool_call>\n<function=bash>\n<parameter=command>\n'+command+'\n</parameter>\n</function>\n</tool_call>'
            action=parse_action(target)
            prefix,progress_receipt=progress.prepare_request(messages,index,12,True)
            result=env.execute(action)
            if result['returncode']!=expected:
                write_json(private/'failure.json',{'step':kind,'expected':expected,'result':result})
                raise ValueError('Reference episode command failed its declared execution contract: '+task['id']+':'+kind)
            examples.append({'id':task['id']+':reference-'+kind,'messages':copy.deepcopy(prefix),'target':target,
                'provenance':{'repository':task['repository'],'revision':task['revision'],'split':'train',
                    'task_id':task['id'],'kind':'verified-reference-repair','stage':kind,
                    'execution_returncode':result['returncode'],'progress':progress_receipt}})
            messages.append({'role':'assistant','content':target,'extra':{'actions':[action]}})
            messages.append({'role':'user','content':f"Exit code: {result['returncode']}\n{result['output']}",
                'extra':{'tool':{k:v for k,v in result.items() if k!='output'}}})
        score=grade(task,workspace,snapshot,private/'grade',original,make_env)
        write_json(private/'trajectory.json',{'messages':messages,'grade':score})
        if not score['passed']:raise ValueError('Reference failed independent fresh-workspace grading')
        proof=sha256(private/'trajectory.json')
        for row in examples:row['provenance']['verified_trajectory_sha256']=proof
        rows.extend(examples);controls.append({'id':task['id'],'passed':True,'trajectory_sha256':proof,'records':len(examples)})
        write_json(out/'status.json',{'complete':False,'tasks_done':len(controls),'records':len(rows)})
        print(task['id'],'verified',len(examples),flush=True)
    path=out/'train.jsonl';path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    receipt={'schema_version':1,'raw_data_public':False,'records':len(rows),'sha256':sha256(path),
        'task_spec_sha256':sha256(tasks_path),'license_record_sha256':sha256(licenses_path),'admission_sha256':sha256(admission_path),
        'kinds':dict(Counter(r['provenance']['kind'] for r in rows)),'repositories':sorted({t['repository'] for t in tasks}),
        'task_ids':sorted(approved),'reference_controls':controls,'teacher_api_used':False,'executed_tool_metadata_preserved':True}
    write_json(out/'receipt.json',receipt);write_json(out/'status.json',{'complete':True,'tasks_done':len(controls),'records':len(rows)})
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','tasks','admission','licenses','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();build(a.root,a.tasks,a.admission,a.licenses,a.out)
