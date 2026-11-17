"""031 three-checkpoint scheduling, identity binding and paired development analysis."""
from collections import defaultdict
import copy
import hashlib
import random
from .factorial import percentile

VARIANTS=['base','original025','mixed031']


def internal_arm(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered checkpoint variant')
    return 'base_current' if variant=='base' else 'student_current'


def schedule(task_ids, repetitions=3, block_size=10):
    if not task_ids or len(set(task_ids))!=len(task_ids) or repetitions!=3 or block_size<1 or len(task_ids)%block_size:
        raise ValueError('Unique tasks, three repetitions and complete fixed blocks required')
    rows=[]
    for rep in range(repetitions):
        ordered=sorted(task_ids,key=lambda n:hashlib.sha256(('031-dev-v1\0'+str(rep)+'\0'+n).encode()).hexdigest())
        for start in range(0,len(ordered),block_size):
            block=start//block_size;offset=(block+rep)%len(VARIANTS)
            for variant in VARIANTS[offset:]+VARIANTS[:offset]:
                for name in ordered[start:start+block_size]:
                    rows.append({'task_id':name,'repetition':rep,'block':block,'variant':variant,
                                 'run_id':f'r{rep}-{name}-{variant}'})
    return rows


def effective_protocol(protocol, variant):
    internal_arm(variant)
    result=copy.deepcopy(protocol);model=protocol['variants'][variant]
    result.update(evaluation_variant=variant,adapter_sha256=model.get('adapter_sha256',{}),
                  all_adapter_files_sha256=model.get('all_adapter_files_sha256',{}),
                  adapter_directory=model.get('adapter_directory'))
    if variant=='base' and (result['adapter_sha256'] or result['adapter_directory']):
        raise ValueError('Base must not receive an adapter')
    if variant!='base' and (not result['adapter_sha256'] or not result['adapter_directory']):
        raise ValueError('Student checkpoint identity missing')
    return result


def analyze(rows,tasks,spec,audit_passed,unchanged):
    dev=spec['development'];rules=spec['analysis'];task_map={t['id']:t for t in tasks}
    if len(task_map)!=len(tasks) or len(tasks)!=dev['unique_tasks']:raise ValueError('Development inventory changed')
    expected={(r['task_id'],r['repetition'],r['variant']) for r in schedule(list(task_map),dev['repetitions'],dev['block_size'])}
    seen={}
    for row in rows:
        key=(row['id'],row['repetition'],row['variant'])
        if key not in expected or key in seen or row['family']!=task_map[row['id']]['family']:
            raise ValueError('Duplicate, foreign or mismatched development result')
        seen[key]=row
    complete=set(seen)==expected and all(r.get('complete') and r.get('audit_passed') and not r.get('unknown_inflight_generation_cost') for r in rows)
    paired=[]
    for task in sorted(tasks,key=lambda t:t['id']):
        group=[seen.get((task['id'],rep,v)) for rep in range(3) for v in VARIANTS]
        if any(not r or not r.get('complete') or not r.get('audit_passed') or r.get('unknown_inflight_generation_cost') for r in group):continue
        scores={v:sum(bool(seen[task['id'],rep,v]['success_within_budget']) for rep in range(3))/3 for v in VARIANTS}
        paired.append({'id':task['id'],'repository':task['repository'],'family':task['family'],'scores':scores,
          'gain_vs_base':scores['mixed031']-scores['base'],'gain_vs_original025':scores['mixed031']-scores['original025'],
          'repeat_disagreement':{v:len({seen[task['id'],rep,v]['success_within_budget'] for rep in range(3)})>1 for v in VARIANTS},
          'action_disagreement':{v:len({seen[task['id'],rep,v]['request_replay']['action_fingerprint'] for rep in range(3)})>1 for v in VARIANTS}})
    cost={k:sum(r.get(k,0) for r in rows) for k in ['attempted_calls','generated_tokens','input_tokens','actor_seconds']}
    result={'registered_runs':len(expected),'complete_accounting':complete,'paired_tasks':len(paired),'pairs':paired,'cost':cost,
            'missing_runs':len(expected-set(seen)),'unknown_generation_cost_runs':sum(bool(r.get('unknown_inflight_generation_cost')) for r in rows),
            'final_test_opened':False,'weight_publication_allowed':False,'development_parameter_gate_passed':False}
    if not paired:return result
    means={v:sum(r['scores'][v] for r in paired)/len(paired) for v in VARIANTS};groups=defaultdict(list)
    for row in paired:groups[row['family']].append(row)
    families=sorted(groups);draws=[];rng=random.Random(31031)
    for _ in range(10000):
        sample=[]
        for _ in families:
            members=groups[rng.choice(families)]
            sample.extend(rng.choice(members) for _ in members)
        draws.append(sum(r['gain_vs_base'] for r in sample)/len(sample))
    interval=[percentile(draws,.025),percentile(draws,.975)]
    repositories={n:{'tasks':len(g),'scores':{v:sum(r['scores'][v] for r in g)/len(g) for v in VARIANTS},
                     'gain_vs_base':sum(r['gain_vs_base'] for r in g)/len(g)} for n,g in groups.items()}
    checks={'complete_audit':bool(complete and audit_passed and unchanged),
      'minimum_gain':means['mixed031']-means['base']+1e-12>=rules['minimum_development_parameter_gain_pp']/100,
      'positive_interval_lower_bound':interval[0]>0,
      'no_excessive_repository_decline':all(r['gain_vs_base']+1e-12>=-rules['maximum_allowed_primary_drop_in_any_repository_pp']/100 for r in repositories.values()),
      'no_net_decline_vs_original025':means['mixed031']+1e-12>=means['original025']}
    result.update(scores=means,primary_gain=means['mixed031']-means['base'],secondary_gain=means['mixed031']-means['original025'],
      primary_interval_95=interval,repositories=repositories,checks=checks,development_parameter_gate_passed=all(checks.values()),
      gain_task_ids=[r['id'] for r in paired if r['gain_vs_base']>0],regression_task_ids=[r['id'] for r in paired if r['gain_vs_base']<0])
    return result
