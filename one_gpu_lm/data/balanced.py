"""Prepare a bounded task-balanced training partition before any candidate evaluation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.training_data import read_training_data
from one_gpu_lm.common.training_encoding import encode_training_example


def choose_balanced(records, budget, required_tasks):
    """Allocate slots round-robin, then spread each quota across its full trajectory."""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < len(required_tasks):
        raise ValueError('Budget cannot cover every admitted training task')
    groups=defaultdict(list)
    for row in records:
        groups[row['task_id']].append(row)
    if set(groups)!=set(required_tasks):
        raise ValueError('Every admitted training task must have eligible supervision')
    if len(records)<budget:
        raise ValueError('Not enough unique eligible examples for the frozen update budget')
    for values in groups.values():values.sort(key=lambda r:r['source_index'])
    quotas={task:0 for task in sorted(groups)}
    while sum(quotas.values())<budget:
        for task in quotas:
            if quotas[task]<len(groups[task]) and sum(quotas.values())<budget:
                quotas[task]+=1
    chosen={}
    for task,count in quotas.items():
        items=groups[task]
        positions=[0] if count==1 else [i*(len(items)-1)//(count-1) for i in range(count)]
        chosen[task]=[items[i] for i in positions]
    # Interleave tasks in the prepared file; the trainer applies its frozen seeded shuffle.
    ordered=[chosen[task][round_] for round_ in range(max(quotas.values()))
             for task in sorted(chosen) if round_<len(chosen[task])]
    assert len(ordered)==budget and len({r['id'] for r in ordered})==budget
    return ordered,quotas


def prepare(paths, profile, tokenizer, budget, out):
    out=Path(out)
    if out.exists():raise ValueError('Preserve existing prepared datasets')
    rows,approved=read_training_data({'paths':paths})
    eligible,excluded,seen=[],[],set()
    target_budget=profile['generation']['max_new_tokens']
    by_id={r['id']:r for r in rows}
    for index,row in enumerate(rows):
        fingerprint=hashlib.sha256(json.dumps({'messages':row['messages'],'target':row['target']},
            ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        reason=None
        if fingerprint in seen:reason='duplicate_context_and_target'
        seen.add(fingerprint)
        encoded=encode_training_example(row,tokenizer,profile)
        if len(encoded['ids'])>profile['training']['max_sequence_length']:reason='context_limit'
        elif encoded['supervised_tokens']>target_budget:reason='target_exceeds_action_generation_budget'
        record={'id':row['id'],'task_id':row['provenance']['task_id'],'source_index':index,
            'kind':row['provenance']['kind'],'tokens':len(encoded['ids']),'supervised_tokens':encoded['supervised_tokens'],
            'record_sha256':fingerprint}
        if reason:excluded.append({**record,'reason':reason})
        else:eligible.append(record)
    selected,quotas=choose_balanced(eligible,budget,approved)
    out.mkdir(parents=True,mode=0o700)
    data=out/'train.jsonl'
    data.write_text(''.join(json.dumps(by_id[r['id']],ensure_ascii=False)+'\n' for r in selected))
    source_receipt=json.loads(paths['data_receipt'].read_text())
    source_receipt.update(records=len(selected),sha256=sha256(data),
        parent_data_sha256=sha256(paths['data']),parent_receipt_sha256=sha256(paths['data_receipt']),
        selection_policy='task-round-robin-quota-uniform-trajectory-v1',
        kinds=dict(Counter(r['kind'] for r in selected)),task_ids=sorted(quotas))
    write_json(out/'receipt.json',source_receipt)
    audit={'schema_version':1,'policy':source_receipt['selection_policy'],'budget':budget,
        'input_records':len(rows),'eligible_records':len(eligible),'selected_records':len(selected),
        'required_tasks':len(approved),'covered_tasks':len(quotas),'per_task':quotas,
        'kinds':dict(Counter(r['kind'] for r in selected)),
        'input_tokens':sum(r['tokens'] for r in selected),
        'supervised_tokens':sum(r['supervised_tokens'] for r in selected),
        'max_input_tokens':max(r['tokens'] for r in selected),
        'max_target_tokens':max(r['supervised_tokens'] for r in selected),
        'selected':[{k:v for k,v in r.items() if k!='source_index'} for r in selected],
        'excluded':excluded,'raw_data_public':False,'evaluation_traces_used':False,
        'data_sha256':sha256(data),'receipt_sha256':sha256(out/'receipt.json')}
    write_json(out/'selection.json',audit)
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','data-receipt','admission','licenses','model','profile','out']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--budget',type=int,required=True)
    args=p.parse_args()
    from transformers import AutoTokenizer
    profile=json.loads(args.profile.read_text())
    downloaded=json.loads((args.model/'download-receipt.json').read_text())
    if any(downloaded[k]!=profile[k] for k in ['model_id','revision']):
        raise ValueError('Tokenizer model identity differs from profile')
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    paths={k:getattr(args,k) for k in ['data','data_receipt','admission','licenses']}
    audit=prepare(paths,profile,tokenizer,args.budget,args.out)
    print(json.dumps({k:audit[k] for k in ['selected_records','covered_tasks','kinds','supervised_tokens']}))


if __name__=='__main__':main()
