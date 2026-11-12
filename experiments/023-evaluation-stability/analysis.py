"""Descriptive paired stability measurements,never model-selection statistics."""
from collections import Counter
from itertools import combinations
import statistics

MODELS=['base','reference020','action022'];CONDITIONS=['raw','canonical']

def first_divergence(a,b):
    """Locate the first observed boundary difference,without claiming sole cause."""
    ta={x['model_call']:x for x in a['observation_receipts']};tb={x['model_call']:x for x in b['observation_receipts']}
    sa={x['call_index']:x for x in a['source_progress']};sb={x['call_index']:x for x in b['source_progress']}
    masked=0
    def result(kind,call):return {'category':kind,'call':call,'metadata_differences_masked_before_divergence':masked}
    for i in range(max(a['calls'],b['calls'])):
        call=i+1
        if i>=a['calls'] or i>=b['calls']:return result('termination_or_length',call)
        if a['generation_input_sha256'][i]!=b['generation_input_sha256'][i]:return result('initial_input' if i==0 else 'unexplained_input',call)
        if a['generation_output_sha256'][i]!=b['generation_output_sha256'][i]:return result('generation_at_identical_input',call)
        x,y=ta.get(call),tb.get(call)
        if (x is None)!=(y is None):return result('tool_structure',call)
        if x is not None:
            if x['command_sha256']!=y['command_sha256']:return result('parsed_command',call)
            if x['prior_editable_state_sha256']!=y['prior_editable_state_sha256']:return result('prior_source_state',call)
            if (x['returncode'],x['reason'])!=(y['returncode'],y['reason']):return result('tool_status',call)
            if x['delivered_sha256']!=y['delivered_sha256']:
                return result('allowed_output_metadata' if x['canonical_sha256']==y['canonical_sha256'] else 'other_tool_text',call)
            masked+=int(x['raw_sha256']!=y['raw_sha256'] and x['canonical_sha256']==y['canonical_sha256'])
            if sa[call]!=sb[call]:return result('source_effect_with_same_feedback',call)
    return result('grade_only' if a['passed']!=b['passed'] else 'identical_observed_trace',None)

def summarize(campaigns,audit):
    expected={f'r{r:02d}-{m}-{c}' for r in [1,2,3] for m in MODELS for c in CONDITIONS}
    if set(campaigns)!=expected or set(audit['campaigns'])!=expected:raise ValueError('All18 cells required; do not select favorable repeats')
    inventory=None;results={};traces={};cells=[];stability=[];pairs=[]
    for name in sorted(expected):
        campaign=campaigns[name];info=audit['campaigns'][name]
        if not campaign['complete']:raise ValueError('Incomplete cell')
        rows=campaign['results'];ids=[r['id'] for r in rows]
        if len(ids)!=15 or len(set(ids))!=15 or (inventory is not None and ids!=inventory):raise ValueError('Missing,duplicate or reordered task')
        inventory=ids;results[name]={r['id']:r for r in rows};traces[name]={r['id']:r for r in info['tasks']}
        if set(traces[name])!=set(ids):raise ValueError('Incomplete trajectory audit')
        cells.append({'cell':name,'model':info['model'],'condition':info['condition'],'repeat':info['repeat'],
          'passed':sum(r['passed'] for r in rows),'tasks':15,'calls':sum(r['calls'] for r in rows),
          'generated_tokens':sum(r['generated_tokens'] for r in rows),'task_seconds':sum(r['seconds'] for r in rows),
          'failed_without_edit':sum(not r['passed'] and not any(s['changed_editable_files'] for s in r['source_progress']) for r in rows),
          'invalid_actions':sum(not g['action_format_valid'] for r in rows for g in r['generations'])})
    def compare(label,x,y,ident,model,condition):
        a,b=traces[x][ident],traces[y][ident]
        return {'comparison':label,'model':model,'condition':condition,'a':x,'b':y,'id':ident,
          'a_passed':a['passed'],'b_passed':b['passed'],'outcome_disagreement':a['passed']!=b['passed'],
          **first_divergence(a,b)}
    for m in MODELS:
        for c in CONDITIONS:
            names=[f'r{r:02d}-{m}-{c}' for r in [1,2,3]]
            subset=[compare('within_condition',a,b,i,m,c) for a,b in combinations(names,2) for i in inventory];pairs+=subset
            totals=[sum(v['passed'] for v in results[n].values()) for n in names]
            stability.append({'model':m,'condition':c,'repeat_passes':totals,'mean_passes':statistics.mean(totals),
              'score_range':max(totals)-min(totals),'mixed_outcome_tasks':sum(0<sum(results[n][i]['passed'] for n in names)<3 for i in inventory),
              'pairwise_disagreements':sum(v['outcome_disagreement'] for v in subset),'pair_denominator':45,
              'first_divergence_categories':dict(Counter(v['category'] for v in subset))})
        for r in [1,2,3]:
            a,b=f'r{r:02d}-{m}-raw',f'r{r:02d}-{m}-canonical'
            pairs += [compare('matched_condition',a,b,i,m,'raw_vs_canonical') for i in inventory]
    return {'cells':cells,'stability':stability,'pairs':pairs,'within_condition_pairs':270,'matched_condition_pairs':135,
      'final_test_opened':False,'training_updates':0,'weights_public':False,
      'interpretation':'First boundary divergence locates an observed difference,not the sole cause of eventual grades. Three repeats on reused tasks provide descriptive stability evidence only.'}
