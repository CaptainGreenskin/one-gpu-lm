"""Balanced matched development evaluation of the 034 training intervention."""
from collections import defaultdict
import hashlib
import random

from .factorial import percentile


ARMS = ('control', 'masked')


def task_order(task):
    return hashlib.sha256(('034-development-v1\0' + task['id']).encode()).hexdigest()


def schedule(tasks):
    families = defaultdict(list)
    for task in tasks:
        if task['split'] != 'development':
            raise ValueError('Frozen development tasks only')
        families[task['family']].append(task)
    if len(families) != 4 or {len(group) for group in families.values()} != {5} or len({t['id'] for t in tasks}) != 20:
        raise ValueError('Expected five unique tasks from each of four repository families')
    ordered = [sorted(families[family], key=task_order) for family in sorted(families)]
    interleaved = [t for layer in zip(*ordered) for t in layer]
    rows = []
    for block, start in enumerate(range(0, len(interleaved), 5)):
        for arm in (ARMS if block % 2 == 0 else ARMS[::-1]):
            for i, task in enumerate(interleaved[start:start+5], start+1):
                rows.append({'task_id':task['id'],'task_number':i,'block':block,'arm':arm,'run_id':f't{i:02d}-{arm}'})
    return rows


def analyze(rows, tasks):
    expected = {(p['task_number'],p['arm']) for p in schedule(tasks)}
    group = defaultdict(dict)
    for row in rows:
        key = row['task_number'], row['arm']
        if not row['audit_passed'] or key not in expected or row['arm'] in group[row['task_number']]:
            raise ValueError('Unregistered, unaudited or repeated development result')
        group[row['task_number']][row['arm']] = row
    complete = all(len(g) == 2 for g in group.values()) and len(group) == 20
    pairs = []
    for number in sorted(group):
        pair = group[number]
        if len(pair) != 2:
            continue
        a,b = (pair[arm] for arm in ARMS)
        if any(a[k]!=b[k] for k in ['family','first_input_sha256','multifile']):
            raise ValueError('Model arms did not receive the same opening input and task')
        pairs.append({'task_number':number,'family':a['family'],'multifile':a['multifile'],
                      'outcome':{arm:pair[arm]['completed_repair'] for arm in ARMS}})
    by_family = defaultdict(list)
    for p in pairs:
        by_family[p['family']].append(int(p['outcome']['masked'])-int(p['outcome']['control']))
    changes = [v for group_delta in by_family.values() for v in group_delta]
    metrics = ['completed_repair','correct_without_submission','first_exact_regression','no_edit','attempted_calls',
               'input_tokens','generated_tokens','timeout_recovered']
    totals = {arm:{k:sum(r[k] for r in rows if r['arm']==arm) for k in metrics} for arm in ARMS}
    passed = (complete and sum(changes)>=2 and sum(sum(v)>0 for v in by_family.values())>=2
              and totals['masked']['first_exact_regression']<totals['control']['first_exact_regression'])
    result={'complete':complete,'paired_tasks':len(pairs),'paired_outcomes':pairs,'arms':totals,
            'masked_only_success':changes.count(1),'control_only_success':changes.count(-1),
            'paired_delta_by_family':{family:sum(values) for family,values in sorted(by_family.items())},
            'mechanism_followup_gate_passed':passed,'final_test_opened':False,'weight_publication_allowed':False}
    if complete:
        rng=random.Random(34034);families=sorted(by_family);samples=[]
        for _ in range(10000):
            selected=rng.choices(families,k=len(families))
            draw=[v for name in selected for v in rng.choices(by_family[name],k=len(by_family[name]))]
            samples.append(sum(draw)/len(draw))
        result.update(paired_success_delta=sum(changes)/len(changes),
                      descriptive_cluster_bootstrap_95=[percentile(samples,.025),percentile(samples,.975)])
    return result
