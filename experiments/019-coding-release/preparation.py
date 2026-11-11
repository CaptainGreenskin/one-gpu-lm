"""Verify actual preparation artifacts before freezing 019; never generate/model-score."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.common.training_data import read_training_data
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.actions import parse_action


def validate_cohorts(tasks,admission,plan):
    ids=[t['id'] for t in tasks]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicated task identities')
    groups={'train':[],'development':[],'historical':[],'fresh':[]}
    for task in tasks:
        if task['split']=='train':group='train'
        elif task['split']=='dev':group='development'
        elif task['split']=='test':
            group='historical' if task.get('evaluation_role')=='historical_final_regression_not_selection' else 'fresh'
        else:raise ValueError('Unknown task split')
        groups[group].append(task)
    if tuple(len(groups[k]) for k in ['development','historical','fresh'])!=(15,24,24):
        raise ValueError('Development/final cohort counts differ from the locked plan')
    if len(groups['train'])<plan['data']['minimum_admitted_train_tasks']:
        raise ValueError('Insufficient admitted training tasks')
    repositories={k:{t['repository'] for t in values} for k,values in groups.items()}
    if repositories['train']!=set(plan['data']['training_repositories']):
        raise ValueError('Training repository set differs from the locked sources')
    if len(repositories['fresh'])!=4 or set(Counter(t['repository'] for t in groups['fresh']).values())!={6}:
        raise ValueError('Fresh test must contain six tasks in each of four repositories')
    names=list(groups)
    if any(repositories[a]&repositories[b] for i,a in enumerate(names) for b in names[i+1:]):
        raise ValueError('Repository leakage across selection/training/final cohorts')
    if admission.get('complete') is not True:raise ValueError('Incomplete execution admission')
    controls=admission['clean'];seen=[r['repository'] for r in controls]
    if len(seen)!=len(set(seen)) or set(seen)!=set.union(*repositories.values()) or any(r['passed'] is not True or r['returncode']!=0 for r in controls):
        raise ValueError('Missing, duplicated or failed clean repository controls')
    records=admission['tasks']
    if len(records)!=len(tasks) or {r['id'] for r in records}!=set(ids):
        raise ValueError('Admission must cover each frozen task exactly once')
    specs={t['id']:t for t in tasks}
    for record in records:
        task=specs[record['id']]
        if any(record[k]!=task[k] for k in ['repository','split','family']):
            raise ValueError('Admission task metadata changed')
        if record['admitted'] is not True or record['broken']['returncode']!=1 or record['restored']['returncode']!=0 or record['restored']['passed'] is not True:
            raise ValueError('Require actual failing tests, not collection/environment failures')
        if set(record['broken_sha256'])!=set(task['editable_files']) or set(record['reference_sha256'])!=set(task['editable_files']):
            raise ValueError('Control source coverage differs from editable files')
        if any(record['broken_sha256'][name]==record['reference_sha256'][name] for name in task['editable_files']):
            raise ValueError('Defective and reference source must differ')
    return {k:[t['id'] for t in values] for k,values in groups.items()}


def validate_source_reviews(tasks,admission,ledger,experiment):
    """Bind admitted repair sources to the exact files whose notices were reviewed."""
    sources={}
    for source in ledger['sources']:
        key=(source['repository'],source['revision'])
        if key in sources:raise ValueError('Duplicated source review identity')
        sources[key]=source
    controls={r['id']:r for r in admission['tasks']}
    for task in tasks:
        if task['split']!='train':continue
        source=sources.get((task['repository'],task['revision']))
        if source is None or source['split']!='train' or source['rights_review'].get('selected_file_review_complete') is not True:
            raise ValueError('Training source lacks its exact revision/file-notice review')
        entries=source.get('selected_files',[])
        reviewed={r['path']:r for r in entries}
        if len(entries)!=len(reviewed):raise ValueError('Duplicated reviewed file identity')
        for name in task['editable_files']:
            if name not in reviewed or not reviewed[name].get('notice_review') or reviewed[name]['sha256']!=controls[task['id']]['reference_sha256'][name]:
                raise ValueError('Admitted reference source differs from the reviewed file')
        notice=Path(experiment)/'notices'/(task['repository'].replace('/','--')+'-LICENSE.txt')
        if sha256(notice)!=source['license_sha256']:
            raise ValueError('Published training-source notice differs from reviewed original')


def validate_examples(rows,selection,approved,tokenizer,profile,plan):
    if profile['training'].get('dialogue_format')!='native_roundtrip':
        raise ValueError('019 requires native tool roundtrip training')
    lower=plan['data'];maximum=plan['budget']
    if not lower['minimum_unique_records']<=len(rows)<=lower['maximum_records']:
        raise ValueError('Training record count violates the locked limits')
    selected={r['id']:r for r in selection['selected']}
    if len(selected)!=len(rows) or set(selected)!={r['id'] for r in rows}:
        raise ValueError('Selection receipt does not identify the actual training rows')
    seen=set();covered=set();repairs=set();supervised=inputs=0
    for row in rows:
        provenance=row['provenance'];task=provenance['task_id']
        if task not in approved or provenance['split']!='train':raise ValueError('Evaluation supervision is forbidden')
        encoded=encode_training_example(row,tokenizer,profile)
        fingerprint=hashlib.sha256(json.dumps(encoded['ids'],separators=(',',':')).encode()).hexdigest()
        if fingerprint in seen:raise ValueError('Duplicate model-visible training example')
        seen.add(fingerprint);covered.add(task)
        if provenance.get('stage')=='repair':repairs.add(task)
        if len(encoded['ids'])>maximum['maximum_sequence_tokens'] or encoded['supervised_tokens']>maximum['max_generated_tokens_per_call']:
            raise ValueError('A training record exceeds the complete context/action budget')
        observed=selected[row['id']]
        if (observed['record_sha256'],observed['tokens'],observed['supervised_tokens'])!=(fingerprint,len(encoded['ids']),encoded['supervised_tokens']):
            raise ValueError('Selection tokens differ from actual fixed-tokenizer inputs')
        supervised+=encoded['supervised_tokens'];inputs+=len(encoded['ids'])
    if covered!=set(approved) or repairs!=set(approved):
        raise ValueError('Every admitted training task must have a retained repair target')
    if supervised<lower['minimum_supervised_tokens']:raise ValueError('Insufficient actual supervised tokens')
    if selection['selected_records']!=len(rows) or selection['supervised_tokens']!=supervised or selection['input_tokens']!=inputs:
        raise ValueError('Selection aggregates differ from actual records')
    return {'records':len(rows),'covered_training_tasks':len(covered),'repair_targets_cover_all_tasks':True,
        'input_tokens':inputs,'supervised_tokens':supervised,'model_visible_duplicates':0}


def validate_reference_row(row,trajectory,control):
    stages=['inspect','reproduce','repair','review','verify','submit']
    provenance=row['provenance']
    if provenance['kind']!='verified-reference-repair' or provenance.get('stage') not in stages:
        raise ValueError('Unregistered 019 reference-episode stage')
    messages=trajectory['messages'];step=stages.index(provenance['stage']);index=2+2*step
    if len(messages)!=14 or trajectory['grade']['passed'] is not True or trajectory['grade']['source_sha256']!=control['reference_sha256']:
        raise ValueError('Reference execution lacks all six steps or the exact independently graded repair')
    target=messages[index];observation=messages[index+1]
    expected_exit=1 if step==1 else 0
    if (target['role']!='assistant' or target['content']!=row['target'] or target['extra']['actions']!=[parse_action(row['target'])]
            or observation['role']!='user' or observation['extra']['tool']['returncode']!=expected_exit
            or provenance['execution_returncode']!=expected_exit):
        raise ValueError('Supervised target/exit status differs from recorded execution')
    # Only the repair action mutates source in this fixed six-step recipe.
    hashes=control['broken_sha256'] if step<=2 else control['reference_sha256']
    state={'changed_editable_files':0 if step<=2 else len(hashes),'invalid_editable_files':0,
        'editable_state_sha256':hashlib.sha256(json.dumps(sorted(hashes.items()),separators=(',',':')).encode()).hexdigest()}
    tracker=object.__new__(EditableProgress);tracker.snapshot=lambda:dict(state)
    expected,receipt=tracker.prepare_request(messages[:index],step,12,True)
    if row['messages']!=expected or provenance['progress']!=receipt:
        raise ValueError('Training prefix contains unrecorded history or mismatched source progress')


def verify(root,out):
    root=Path(root).resolve();out=Path(out)
    if out.exists():raise ValueError('Preserve prior preparation receipts')
    exp=root/'experiments/019-coding-release';plan=json.loads((exp/'preparation-plan.json').read_text())
    paths={'data':root/'runs/prepared-data/train.jsonl','data_receipt':root/'runs/prepared-data/receipt.json',
        'admission':root/'runs/admission/admission.json','licenses':exp/'sources.json'}
    tasks_path=root/'private/tasks.json';tasks=json.loads(tasks_path.read_text())['tasks']
    admission=json.loads(paths['admission'].read_text())
    if admission['task_spec_sha256']!=sha256(tasks_path):raise ValueError('Admission/task content mismatch')
    cohorts=validate_cohorts(tasks,admission,plan)
    validate_source_reviews(tasks,admission,json.loads(paths['licenses'].read_text()),exp)
    profiles={name:json.loads((exp/f'profile-{name}.json').read_text()) for name in ['low','high']}
    copies=[]
    for candidate in plan['candidates']:
        profile=profiles[candidate['name']]
        if any(profile[k]!=plan['base'][k] for k in ['model_id','revision']):raise ValueError('Base model identity changed')
        if profile['training']['learning_rate']!=candidate['learning_rate']:raise ValueError('Unregistered candidate learning rate')
        recipe=plan['shared_recipe']
        actual=(profile['dtype'],profile['lora']['r'],profile['lora']['lora_alpha'],profile['lora']['lora_dropout'],
                profile['training']['seed'],profile['training']['epochs'],profile['training']['dialogue_format'],
                profile['training']['max_sequence_length'],profile['training']['gradient_checkpointing'])
        expected=(recipe['backbone_dtype'],recipe['rank'],recipe['alpha'],recipe['lora_dropout'],
                  recipe['seed'],recipe['epochs'],recipe['dialogue_format'],plan['budget']['maximum_sequence_tokens'],True)
        if actual!=expected:raise ValueError('Candidate recipe differs from the locked plan')
        copy=json.loads(json.dumps(profile));copy['training'].pop('learning_rate');copies.append(copy)
    if copies[0]!=copies[1]:raise ValueError('Candidates differ outside their declared learning rates')
    profile=profiles['low'];rows,approved=read_training_data({'paths':paths})
    if set(cohorts['train'])!=approved:raise ValueError('Training admission set mismatch')
    selection_path=root/'runs/prepared-data/selection.json';selection=json.loads(selection_path.read_text())
    if selection['data_sha256']!=sha256(paths['data']) or selection['receipt_sha256']!=sha256(paths['data_receipt']):
        raise ValueError('Prepared-data selection identities changed')
    runtime=json.loads((root/'runtime.json').read_text());model=Path(runtime['model'])
    receipt=json.loads((model/'download-receipt.json').read_text())
    if any(receipt[k]!=profile[k] for k in ['model_id','revision']):raise ValueError('Tokenizer identity differs')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    counts=validate_examples(rows,selection,approved,tokenizer,profile,plan)
    for name in ['low','high']:
        config=json.loads((exp/f'train-{name}.json').read_text())
        if config['profile']!=f'profile-{name}.json' or config['run']['max_steps']!=len(rows) or config['run']['max_examples']!=len(rows):
            raise ValueError('Training configuration must consume exactly one pass of the selected data')
        if config['run']['max_attempt_seconds']>plan['budget']['max_training_seconds_per_candidate']:
            raise ValueError('Candidate attempt budget exceeds the locked plan')
    references=root/'runs/reference-data/private'
    controls={r['id']:r for r in admission['tasks']}
    for row in rows:
        proof=references/row['provenance']['task_id']/'trajectory.json'
        if sha256(proof)!=row['provenance'].get('verified_trajectory_sha256'):
            raise ValueError('Training row lacks its recorded execution proof')
        trajectory=json.loads(proof.read_text())
        validate_reference_row(row,trajectory,controls[row['provenance']['task_id']])
    result={'schema_version':1,'passed':True,'cohort_counts':{k:len(v) for k,v in cohorts.items()},'training':counts,
        'identities':{'tasks':sha256(tasks_path),'admission':sha256(paths['admission']),'selection':sha256(selection_path),
                      'data':sha256(paths['data']),'data_receipt':sha256(paths['data_receipt']),'licenses':sha256(paths['licenses'])},
        'scope':'Actual task controls, retained repair supervision, fixed tokenizer counts and independent reference grades. Does not assert model quality or replace rights review/reference-exposure screens.'}
    write_json(out,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();result=verify(a.root,a.out);print(json.dumps(result['training']))
