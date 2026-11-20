"""Check original task inputs, both weight identities and every generation receipt."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import tempfile

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import prepare
from one_gpu_lm.verified.rollout import normalization

from analysis import analyze
from run import DATA, OUT, effective, inputs, load, read


def run(root):
    context=inputs(root)
    out=root/OUT
    target=out/'audit.json'
    if target.exists():
        raise ValueError('Preserve the original audit')
    controller=out/'controller.json'
    state=read(controller)
    if (not state['complete'] or not state['backend_stopped'] or not state['frozen_inputs_unchanged']
            or state['optimizer_updates'] or state['final_test_opened'] or state['weights_public']
            or state['protocol_sha256']!=sha256(context.plan_path) or state['schedule_sha256']!=context.plan['schedule_sha256']
            or [r['run_id'] for r in state['runs']]!=[r['run_id'] for r in context.order]):
        raise ValueError('Expected the complete original scheduled development run')
    common=context.common
    prior=root/'experiments/035-edit-bottleneck'
    sys.path.insert(0,str(prior))
    legacy=load('039_action_audit',prior/'audit.py')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(context.runtime['model'],local_files_only=True,trust_remote_code=False)
    profile=read(root/common['evaluation_profile_path'])
    norm=normalization(root)
    location=out/'private/backends'
    environment=read(root/common['environment_path'])
    backends={}
    journals={}
    served={}
    for backend in state['backend_lifecycles']:
        i,role=backend['index'],backend['role']
        arm='base' if role=='base' else 'candidate'
        directory=location/f'{i:04d}-{role}'
        artifact=read(directory/'served-artifact.json')
        selected=effective(common,arm)
        if (i in backends or role not in ('base','student') or not backend.get('stopped_epoch')
                or artifact['model_id']!=common['model_id'] or artifact['revision']!=common['revision']
                or artifact['adapter'] is not (role=='student') or artifact['adapter_sha256']!=selected['adapter_sha256']
                or artifact['profile_sha256']!=common['evaluation_profile_sha256']
                or read(directory.with_name(directory.name+'-environment.json'))!=environment):
            raise ValueError('Unexpected model load or runtime identity')
        journals[i]=RequestJournal(directory/'requests')
        served[i]=artifact
        backends[i]=backend
    seen=defaultdict(set)
    rows=[]
    originals={}
    with tempfile.TemporaryDirectory(prefix='039-audit-',dir=out/'private') as temporary:
        scratch=Path(temporary)
        for item,result in zip(context.order,state['runs']):
            n,arm,i=item['task_number'],item['arm'],result['backend_index']
            run_id=item['run_id']
            task=context.prompts[n]
            source=context.sources[task['source_id']]
            if (result['id']!=task['id'] or result['task_number']!=n or result['contrast_arm']!=arm
                    or result['block']!=item['block'] or result['repetition']!=item['repetition']
                    or backends[i]['role']!=('base' if arm=='base' else 'student')):
                raise ValueError('Rollout identity differs from schedule')
            directory=out/'private/tasks'/run_id
            trace=read(directory/'trajectory.json')
            if trace['backend']!=served[i]:
                raise ValueError('Trajectory differs from served weights')
            proof=context.proof['rows'][n-1]
            if n not in originals:
                originals[n]=prepare(root/DATA/'snapshots'/source['source_id'],scratch/f'original-{n:02d}',source,task)
                if fingerprint(sorted(originals[n].items()))!=proof['workspace_sha256']:
                    raise ValueError('Actor source or test inventory changed')
            row,first,requests=legacy.audited_row(root,root/DATA,directory,result,task,source,effective(common,arm),
                profile,tokenizer,norm,journals[i],scratch/run_id,originals[n],backends[i])
            if (first!=proof['input_sha256'] or trace['steps'][0]['generation']['input_tokens']!=proof['input_tokens']
                    or seen[i]&requests):
                raise ValueError('Opening input or physical generation count differs')
            seen[i].update(requests)
            row.update(repetition=item['repetition'],multifile=proof['multifile'],opening_input_tokens=proof['input_tokens'])
            rows.append(row)
    if set(seen)!=set(journals) or any(seen[i]!={path.name for path in journal.directory.iterdir() if path.is_dir()}
                                      for i,journal in journals.items()):
        raise ValueError('Unreconciled backend request journal')
    summary={'passed':True,'schema_version':1,'protocol_sha256':sha256(context.plan_path),
        'controller_sha256':sha256(controller),'verified_requests':sum(map(len,seen.values())),
        'started_epoch':state['started_epoch'],'finished_epoch':state['finished_epoch'],'backend_loads':len(backends),
        'telemetry':state['telemetry'],'rows':rows,'analysis':analyze(rows,context.plan),
        'optimizer_updates':0,'final_test_opened':False,'weights_public':False}
    write_json(target,summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    value=run(p.parse_args().root.resolve())
    print(json.dumps({'passed':value['passed'],'verified_requests':value['verified_requests'],
          'development_gate_passed':value['analysis']['development_gate_passed']}))
