"""Frozen 2x2 scheduling and paired analysis; repeats are not independent tasks."""
from collections import Counter, defaultdict
import hashlib
import random

ARMS=['base_current','base_guard','student_current','student_guard']


def schedule(task_ids, repetitions=3):
    if not task_ids or len(set(task_ids))!=len(task_ids) or repetitions!=3:
        raise ValueError('Expected unique tasks and exactly three repetitions')
    rows=[]
    for repetition in range(repetitions):
        ordered=sorted(task_ids,key=lambda name:hashlib.sha256(('025-dev-v1\0'+str(repetition)+'\0'+name).encode()).hexdigest())
        for index,task in enumerate(ordered):
            offset=(index+repetition)%4
            for arm in ARMS[offset:]+ARMS[:offset]:
                rows.append({'task_id':task,'repetition':repetition,'arm':arm,
                             'run_id':f'r{repetition}-{task}-{arm}'})
    return rows


def effects(scores):
    b,g,s,t=[scores[a] for a in ARMS]
    return {'parameter':s-b,'system':g-b,'parameter_under_guard':t-g,'interaction':(t-s)-(g-b)}


def percentile(values, fraction):
    ordered=sorted(values);position=(len(ordered)-1)*fraction
    left=int(position);right=min(left+1,len(ordered)-1)
    return ordered[left]+(ordered[right]-ordered[left])*(position-left)


def analyze(rows, tasks, spec, audit_passed, frozen_inputs_unchanged):
    """All ordinary failed runs count; incomplete pairs cannot pass the gate."""
    dev=spec['development']; rules=spec['analysis']
    task_map={t['id']:t for t in tasks}
    if len(task_map)!=len(tasks) or len(tasks)!=dev['unique_tasks']:
        raise ValueError('Unexpected development task inventory')
    expected={(p['task_id'],p['repetition'],p['arm']) for p in schedule(list(task_map),dev['repetitions'])}
    seen={}
    for row in rows:
        key=(row['id'],row['repetition'],row['arm'])
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicated evaluation run')
        if row['repository']!=task_map[row['id']]['repository']:
            raise ValueError('Task repository changed')
        seen[key]=row
    all_accounted=(set(seen)==expected and all(r.get('complete',False) and not r.get('unknown_inflight_generation_cost',False) for r in rows))
    cost={'attempted_calls':sum(r.get('attempted_calls',0) for r in rows),
          'generated_tokens':sum(r.get('generated_tokens',0) for r in rows),
          'input_tokens':sum(r.get('input_tokens',0) for r in rows),
          'task_seconds':sum(r.get('seconds',0) for r in rows),
          'partial_runs':sum(not r.get('complete',False) for r in rows),
          'unknown_inflight_runs':sum(bool(r.get('unknown_inflight_generation_cost',False)) for r in rows),
          'unverified_score_runs':sum(r.get('complete',False) and (r.get('audit_passed') is False or 'success_within_budget' not in r) for r in rows),
          'missing_or_partial_task_duration_runs':sum(not r.get('complete',False) or 'seconds' not in r for r in rows),
          'missing_runs':len(expected-set(seen))}
    paired=[]
    for task in tasks:
        group=[seen.get((task['id'],rep,arm)) for rep in range(dev['repetitions']) for arm in ARMS]
        if any(r is None or not r.get('complete',False) or r.get('unknown_inflight_generation_cost',False)
               or r.get('audit_passed') is False or 'success_within_budget' not in r for r in group):
            continue
        scores={arm:sum(bool(seen[task['id'],rep,arm]['success_within_budget']) for rep in range(dev['repetitions']))/dev['repetitions'] for arm in ARMS}
        paired.append({'id':task['id'],'repository':task['repository'],'family':task['family'],
                       'scores':scores,'effects':effects(scores),
                       'repeat_disagreement':{arm:len({bool(seen[task['id'],rep,arm]['success_within_budget']) for rep in range(dev['repetitions'])})>1 for arm in ARMS}})
    if not paired:
        return {'complete_accounting':all_accounted,'paired_tasks':0,'cost':cost,'development_parameter_gate_passed':False,
                'reason':'no_complete_pairs','final_test_opened':False,'weight_publication_allowed':False}
    means={arm:sum(r['scores'][arm] for r in paired)/len(paired) for arm in ARMS}
    groups=defaultdict(list)
    for row in paired:groups[row['family']].append(row)
    families=sorted(groups);rng=random.Random(25025)
    draws={key:[] for key in effects(means)}
    for _ in range(10000):
        sample=[]
        for _ in families:
            family=rng.choice(families);members=groups[family]
            sample.extend(rng.choice(members) for _ in members)
        for key in draws:
            draws[key].append(sum(r['effects'][key] for r in sample)/len(sample))
    intervals={key:[percentile(value,.025),percentile(value,.975)] for key,value in draws.items()}
    family_effects={f:sum(r['effects']['parameter'] for r in groups[f])/len(groups[f]) for f in families}
    primary=effects(means)['parameter']
    checks={'all_runs_accounted':all_accounted and len(rows)==dev['total_task_runs'],
            'all_tasks_paired':len(paired)==dev['unique_tasks'],
            'all_registered_families':len(groups)==dev['repository_families'],
            'input_and_generation_audit_passed':bool(audit_passed),
            'frozen_inputs_unchanged':bool(frozen_inputs_unchanged),
            'primary_gain_target':primary+1e-12>=rules['minimum_development_parameter_gain_pp']/100,
            'primary_interval_lower_above_zero':intervals['parameter'][0]>0,
            'no_excess_repository_regression':all(v+1e-12>=-rules['maximum_allowed_primary_drop_in_any_repository_pp']/100 for v in family_effects.values())}
    return {'complete_accounting':all_accounted,'paired_tasks':len(paired),'independent_task_units':len(paired),
            'repetitions_per_task':dev['repetitions'],'cost':cost,'scores':means,'effects':effects(means),
            'intervals_95':intervals,'primary_effect_by_family':family_effects,'tasks':paired,
            'repeat_disagreement_tasks_by_arm':{arm:sum(r['repeat_disagreement'][arm] for r in paired) for arm in ARMS},
            'gate_checks':checks,'development_parameter_gate_passed':all(checks.values()),
            'final_test_opened':False,'weight_publication_allowed':False}
