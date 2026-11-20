"""Bind the score-blind cohort, prompts and exact selected adapter to the design."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256 as hash_bytes
import json
import os
from pathlib import Path
import shutil
import tempfile

from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_policy import initial_messages
from one_gpu_lm.verified.historic_tasks import prepare, validate_split
from one_gpu_lm.verified.tasks import relative

from analysis import schedule
from run import DATA, EXP, OUT, PRIVATE, REFERENCE, TEST_RUNTIME, prefix_prompt, read, reference_protocol


def select(root,design):
    private=root/PRIVATE
    lineage=read(private/'lineage-review.json')
    if (not lineage['passed'] or lineage['final_test_opened'] or lineage['overlap']
            or lineage['reference_experiment_commit']!=design['reference_experiment_commit']):
        raise ValueError('Complete development family lineage proof required')
    index=read(private/'admission-index.json')
    runtime=root/TEST_RUNTIME
    runtime_hash=sha256(runtime)
    mounted={m['sha256'] for m in read(runtime)['mounts']}
    selected=[]
    family_receipts={}
    admission_paths=[]
    for family,size in sorted(design['families'].items()):
        entry=index[family]
        corpus=private/relative(entry['candidate_directory'])
        stem=family.replace('/','--')
        output=private/relative(entry['admission_directory'])
        admission=output/'status.json'
        record=read(admission)
        sp=corpus/f'{stem}-sources.json'
        tp=corpus/f'{stem}-tasks.json'
        if (not record['complete'] or record['family']!=family or record['final_test_opened'] or record['model_calls']
                or record['sources_sha256']!=sha256(sp) or record['tasks_sha256']!=sha256(tp)
                or record['runtime_sha256']!=runtime_hash):
            raise ValueError('Candidate intake or CPU evidence differs')
        sources={s['source_id']:s for s in read(sp)['sources']}
        tasks={t['id']:t for t in read(tp)['tasks']}
        valid=sorted((r for r in record['rows'] if r['admitted']),key=lambda r:fingerprint(r['id']))
        if len(valid)<size:
            raise ValueError(f'Insufficient independently admitted tasks in {family}')
        family_receipts[family]={'checked':len(record['rows']),'admitted':len(valid),'selected':size,
                                 'admission_sha256':sha256(admission)}
        admission_paths.append(admission)
        for row in valid[:size]:
            task=tasks[row['id']]
            source=sources[task['source_id']]
            changed={source['files'][m['path']] for m in task['mutations']}
            changed.update(hash_bytes(m['after'].encode()).hexdigest() for m in task['mutations'])
            if mounted & changed:
                raise ValueError('Test runtime contains target implementation')
            case=output/row['id']
            checks={r['label']:r for r in row['checks']}
            for name in ('parent-0','parent-1','fixed-0','fixed-1','grader'):
                proof=checks[name]
                if not proof['passed'] or proof['sha256']!=sha256(case/(name+'.json')):
                    raise ValueError('CPU failure or passing reference proof changed')
            selected.append((task,source,corpus/'snapshots'/source['source_id'],case,row))
    validate_split([s for _,s,_,_,_ in selected],lineage['prior_families'])
    return selected,lineage,family_receipts,admission_paths


def freeze(root):
    path=root/EXP/'protocol.json'
    if path.exists() or (root/DATA).exists() or (root/OUT).exists():
        raise ValueError('Preserve first frozen cohort')
    design=read(root/EXP/'design.json')
    common=reference_protocol(root)
    private=root/PRIVATE
    candidate=read(private/'candidate-input-review.json')
    if not candidate['eligible_for_existing_scope_of_development'] or candidate['weights_public'] or candidate['final_test_opened']:
        raise ValueError('Verify the exact development candidate first')
    chosen,lineage,counts,admissions=select(root,design)
    if len(chosen)!=design['task_count'] or Counter(s['language'] for _,s,_,_,_ in chosen)!=design['language_counts']:
        raise ValueError('Original design coverage required')
    model=read(root/'private/parent-runtime.json')['model']
    profile=read(root/common['evaluation_profile_path'])
    policy=common['agent_policy']
    budget=common['evaluation_budget']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)

    def encode(messages):
        return encode_prefix(tokenizer,messages,profile['chat_template_kwargs'],policy['format'])[0]

    with tempfile.TemporaryDirectory(prefix='039-freeze-', dir=private) as temporary:
        tmp=Path(temporary)
        data=tmp/'development'
        (data/'snapshots').mkdir(parents=True)
        (data/'public-tests').mkdir()
        rows=[]
        for number,(task,source,origin,case,row) in enumerate(chosen,1):
            snapshot=data/'snapshots'/source['source_id']
            shutil.copytree(origin,snapshot,copy_function=os.link)
            before=prepare(snapshot,tmp/f'actor-{number:02d}',source,task)
            if any(sha256(case/'parent-0'/name)!=value for name,value in before.items()):
                raise ValueError('Recorded public failure differs from the actor starting workspace')
            transcript=case/'parent-0.json'
            shutil.copyfile(transcript,data/'public-tests'/f't{number:02d}.json')
            prompt=prefix_prompt(root,common,task,read(transcript))
            tokens,_=prepare_context(initial_messages(prompt,policy,budget),encode,budget['max_input_tokens'],policy['context'])
            if not tokens or len(tokens)>budget['max_input_tokens']:
                raise ValueError('Registered opening exceeds the per-call input cap')
            rows.append(dict(task_number=number,transcript_sha256=sha256(transcript),
                description_sha256=fingerprint(prompt['description']),
                input_sha256=hash_bytes(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
                input_tokens=len(tokens),workspace_sha256=fingerprint(sorted(before.items())),multifile=row['multifile']))
        write_json(data/'tasks.json',{'tasks':[t for t,_,_,_,_ in chosen]})
        write_json(data/'sources.json',{'sources':[s for _,s,_,_,_ in chosen]})
        write_json(data/'preflight.json',{'passed':True,'model_calls':0,'final_test_opened':False,
                                        'prior_families':lineage['prior_families'],'rows':rows})
        data.rename(root/DATA)
    dependencies=[EXP/name for name in ('design.json','freeze.py','run.py','analysis.py','audit.py')]
    dependencies += [REFERENCE/'protocol.json',Path('experiments/036-regression-evidence/evidence.py'),
                     Path('experiments/035-edit-bottleneck/audit.py'),
                     Path('experiments/032-multifile-completion/evaluation-protocol.json')]
    dependencies += [common[name] for name in ('environment_path','evaluation_profile_path','model_manifest_path')]
    dependencies += [TEST_RUNTIME]
    dependencies += [PRIVATE/name for name in ('runtime-proof-v2.json','lineage-review.json',
                                              'admission-index.json','candidate-input-review.json')]
    dependencies += [DATA/name for name in ('tasks.json','sources.json','preflight.json')]
    dependencies += [DATA/'public-tests'/f't{i:02d}.json' for i in range(1,len(chosen)+1)]
    dependencies += [p.relative_to(root) for p in admissions]
    dependencies += [p.relative_to(root) for p in (root/'one_gpu_lm').rglob('*.py')]
    original=read(root/'experiments/031-correction-mix/candidate-protocol.json')
    plan=dict(design,
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        schedule_sha256=fingerprint(schedule(list(range(1,len(chosen)+1)),design)),
        adapter_directory=original['run_directory']+'/private/adapter',
        all_adapter_files_sha256=candidate['checkpoint_files_sha256'],
        admission=counts, opening_tokens=sum(row['input_tokens'] for row in rows),
        multifile_tasks=sum(row['multifile'] for row in rows),
        frozen_files={n:sha256(root/relative(n)) for n in sorted({str(p) for p in dependencies})})
    write_json(path,plan)
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    value=freeze(a.root.resolve())
    print(json.dumps({key:value[key] for key in ('task_count','rollout_count','admission','multifile_tasks','opening_tokens','schedule_sha256')}))
