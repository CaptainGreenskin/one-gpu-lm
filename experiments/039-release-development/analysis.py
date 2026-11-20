"""Recompute the development gate with all three paired repeats kept per task."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from one_gpu_lm.verified.factorial import percentile


ARMS = ('base', 'candidate')


def schedule(tasks, design):
    if sorted(tasks)!=list(range(1,design['task_count']+1)) or design['task_count']%design['block_size']:
        raise ValueError('Require the fixed number of unique tasks and complete blocks')
    rows=[]
    for repetition in range(design['repetitions']):
        ordered=sorted(tasks,key=lambda number:hashlib.sha256(
            f'039-dev-v1\0{repetition}\0{number}'.encode()).hexdigest())
        size=design['block_size']
        for start in range(0,len(ordered),size):
            block=start//size
            arms=ARMS if (block+repetition)%2==0 else ARMS[::-1]
            for arm in arms:
                for number in ordered[start:start+size]:
                    rows.append({'task_number':number,'repetition':repetition,'block':block,'arm':arm,
                                 'run_id':f'r{repetition}-t{number:02d}-{arm}'})
    if len(rows)!=design['rollout_count']:
        raise ValueError('Schedule size differs from the registration')
    return rows


def analyze(rows, design):
    rules=design['gate']
    count, repeats=design['task_count'],design['repetitions']
    expected={(n,r,a) for n in range(1,count+1) for r in range(repeats) for a in ARMS}
    seen={}
    for row in rows:
        key=row['task_number'],row['repetition'],row['arm']
        if key not in expected or key in seen:
            raise ValueError('Duplicate or unregistered measurement')
        seen[key]=row
    complete=set(seen)==expected and all(row['audit_passed'] is True for row in rows)
    costs={arm:{key:sum(r[key] for r in rows if r['arm']==arm) for key in (
        'attempted_calls','input_tokens','generated_tokens','actor_seconds')} for arm in ARMS}
    output={'complete_accounting':complete,'scheduled_runs':len(expected),'received_runs':len(seen),
            'development_gate_passed':False,'costs':costs,'final_test_opened':False,
            'weights_public':False}
    if not complete:
        return output
    families=defaultdict(list)
    pairs=[]
    for number in range(1,count+1):
        groups={arm:[seen[number,r,arm] for r in range(repeats)] for arm in ARMS}
        family={row['family'] for group in groups.values() for row in group}
        if len(family)!=1:
            raise ValueError('Task family must match across all arms and repeats')
        name=family.pop()
        wins={a:sum(r['completed_repair'] for r in group) for a,group in groups.items()}
        delta=wins['candidate']-wins['base']
        families[name].append(delta)
        pairs.append({'task_number':number,'family':name,'passes':wins,'gain':delta/repeats,
             'score_disagreement':{a:0<w<repeats for a,w in wins.items()}})
    if {name:len(deltas) for name,deltas in families.items()}!=design['families']:
        raise ValueError('Family coverage differs from the frozen design')
    gains=[(p['passes']['candidate']-p['passes']['base']) for p in pairs]
    family_names=sorted(families)
    rng=random.Random(rules['bootstrap_seed'])
    draws=[]
    for _ in range(rules['bootstrap_draws']):
        selected=[families[rng.choice(family_names)] for _ in family_names]
        draws.append(sum(rng.choice(group) for group in selected for _ in group)
                     /(repeats*sum(map(len,selected))))
    bounds=[percentile(draws,.025),percentile(draws,.975)]
    checks={
        'complete_audit':complete,
        'minimum_gain':100*sum(gains)>=rules['minimum_gain_pp']*count*repeats,
        'positive_interval_lower_bound':bounds[0]>0,
        'maximum_family_drop':all(100*sum(g)>=-rules['maximum_family_drop_pp']*len(g)*repeats
                                  for g in families.values()),
    }
    successes={a:sum(p['passes'][a] for p in pairs) for a in ARMS}
    behaviors={}
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm]
        behaviors[arm]={metric:sum(r[metric] for r in group) for metric in (
            'no_edit','correct_without_submission','timeout_recovered')}
        behaviors[arm]['first_edit']=sum(r['first_edit_call'] is not None for r in group)
    output.update(paired_tasks=count,successes=successes,pairs=pairs,checks=checks,behaviors=behaviors,
        gain_pp=100*sum(gains)/(count*repeats),gain_interval_95_pp=[v*100 for v in bounds],
        family_gain_pp={name:100*sum(g)/(len(g)*repeats) for name,g in sorted(families.items())},
        development_gate_passed=all(checks.values()))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('measurements',type=Path)
    args=p.parse_args()
    design=json.loads(Path(__file__).with_name('design.json').read_text(encoding='utf-8'))
    rows=json.loads(args.measurements.read_text(encoding='utf-8'))['rows']
    print(json.dumps(analyze(rows,design),indent=2))
