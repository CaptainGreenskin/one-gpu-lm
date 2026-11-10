"""Paired historical analysis; never relabel observed tasks as fresh evidence."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json

ARMS=('schema','roundtrip')


def assess(campaigns,protocol,inventory,prior):
    ids=[t['id'] for t in inventory]
    if len(set(ids))!=len(ids) or len(ids)!=protocol['execution']['expected_tasks_per_variant'] or any(t['split']!='dev' for t in inventory):
        raise ValueError('Require a unique, entirely historical inventory')
    rows={};totals={}
    for arm in ARMS:
        c=campaigns[arm];policy=protocol['variants'][arm]
        if not c['complete'] or c['identity']['ids']!=ids or c['identity']['variant']!=arm or c['identity']['policy']!=policy or [r['id'] for r in c['results']]!=ids:
            raise ValueError('Missing or mismatched paired campaign')
        rows[arm]={r['id']:r for r in c['results']};rs=c['results'];gs=[]
        for r,t in zip(rs,inventory):
            if any(r[k]!=t[k] for k in ['repository','split','family']) or r['agent_policy']!=policy or r['variant']!=arm or type(r['passed']) is not bool:
                raise ValueError('Task identity or policy changed')
            if r['calls']>protocol['agent']['max_model_calls'] or r['generated_tokens']>protocol['agent']['max_total_generated_tokens'] or len(r['generations'])+len(r['backend_errors'])!=r['calls']:
                raise ValueError('Call/token budget or accounting mismatch')
            if sum(g['generated_tokens'] for g in r['generations'])!=r['generated_tokens']:
                raise ValueError('Missing generated token evidence')
            for g in r['generations']:
                if g['adapter'] or g['agent_policy']!=policy or g['input_tokens']>protocol['agent']['max_input_tokens'] or g['generated_tokens']>protocol['agent']['max_generated_tokens_per_call']:
                    raise ValueError('Serving identity or budget mismatch')
                if type(g['action_format_valid']) is not bool or g['context']['policy']!='full' or g['context']['dropped_rounds'] or g['context']['raw_input_tokens']!=g['input_tokens'] or g['context']['packed_input_tokens']!=g['input_tokens']:
                    raise ValueError('Silently modified full history or missing parser outcome')
                d=g['dialogue']
                if set(d)!={'structured_tool_calls','tool_results'} or d['structured_tool_calls']!=d['tool_results'] or (arm=='schema' and d['tool_results']!=0):
                    raise ValueError('Wrong native conversation receipt')
            gs+=r['generations']
        totals[arm]={'passed':sum(r['passed'] for r in rs),'tasks':len(rs),'calls':sum(r['calls'] for r in rs),'generation_calls':len(gs),'format_invalid_calls':sum(not g['action_format_valid'] for g in gs),
            'context_limit_requests':sum(e.get('error')=='context_limit' for r in rs for e in r['backend_errors']),
            'backend_errors':dict(Counter(e.get('error') for r in rs for e in r['backend_errors'])),
            'generation_limit_hits':sum(g['hit_generation_limit'] for g in gs),'generated_tokens':sum(g['generated_tokens'] for g in gs),'generation_seconds':sum(g['seconds'] for g in gs),
            'task_seconds':sum(r['seconds'] for r in rs),'tool_seconds':sum(r['tool_seconds'] for r in rs),'tool_calls':sum(r['tool_calls'] for r in rs),'median_task_seconds':statistics.median(r['seconds'] for r in rs),
            'calls_with_native_history':sum(g['dialogue']['tool_results']>0 for g in gs),'exit_status':dict(Counter(r['exit_status'] for r in rs)),
            'failed_without_source_change':sum(not r['passed'] and not r['modified_source_files'] for r in rs)}
    improved=[i for i in ids if not rows['schema'][i]['passed'] and rows['roundtrip'][i]['passed']]
    regressed=[i for i in ids if rows['schema'][i]['passed'] and not rows['roundtrip'][i]['passed']]
    gate=protocol['promotion']['next_fresh_study_candidate_requires'];target=gate['target_task']
    checks={'target_passes':rows['roundtrip'][target]['passed']==gate['target_passes'],'net_gain':len(improved)-len(regressed)>=gate['minimum_net_historical_gain'],'no_regressions':len(regressed)<=gate['maximum_regressed_tasks_vs_concurrent_control']}
    prev={r['id']:r for r in prior['results']}
    if not prior['complete'] or set(prev)!=set(ids):raise ValueError('Prior reference inventory mismatch')
    return {'schema_version':1,'historical_tasks':len(ids),'fresh_tasks':0,'totals':totals,'improved':improved,'regressed':regressed,'net_gain':len(improved)-len(regressed),
        'target':{'id':target,'passed':{a:rows[a][target]['passed'] for a in ARMS}},
        'per_repository':{repo:{'tasks':sum(t['repository']==repo for t in inventory),'passed':{a:sum(r['passed'] for r in rows[a].values() if r['repository']==repo) for a in ARMS}} for repo in sorted({t['repository'] for t in inventory})},
        'concurrent_control_vs_015_format':{'previous_passed':sum(r['passed'] for r in prev.values()),'improved':[i for i in ids if not prev[i]['passed'] and rows['schema'][i]['passed']],'regressed':[i for i in ids if prev[i]['passed'] and not rows['schema'][i]['passed']]},
        'diagnostic_checks':checks,'candidate_for_new_holdout':all(checks.values()),'adoption_allowed':False,'generalization_established':False}


def render(root):
    read=lambda name:json.loads((root/name).read_text())
    p=read('protocol.json');c=read('controller.json');inventory=read('task-inventory.json');launch=read('launch.json');pub=read('preregistration.json')
    if pub['commit']!=launch['source_commit'] or pub['protocol_sha256']!=sha256(root/'protocol.json') or not pub['anonymous'] or pub['ci']['conclusion']!='success':raise ValueError('Missing public pre-rollout receipt')
    if not c['complete'] or not c['telemetry']['thread_stopped'] or c['completed_runs']!=p['execution']['expected_model_task_runs'] or c['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('Incomplete frozen study')
    if sha256(root/'task-inventory.json')!=sha256(Path(__file__).parent/'preflight/task-inventory.json'):raise ValueError('Task inventory changed')
    for local,frozen in [('admission.json','runs/admission/admission.json'),('profile.json','experiments/016-tool-roundtrip/model-profile.json'),('sources.json','experiments/016-tool-roundtrip/sources.json'),('source-snapshots.json','source-snapshots.json'),('model-verification.json','model-verification.json'),('template-roundtrip-probe.json','template-roundtrip-probe.json')]:
        if sha256(root/local)!=p['frozen_files'][frozen]:raise ValueError('Frozen evidence changed: '+local)
    artifact=read('artifact.json')
    if any(artifact[k]!=p[k] for k in ['model_id','revision']) or artifact['adapter'] or artifact['adapter_sha256'] or artifact['profile_sha256']!=sha256(root/'profile.json'):raise ValueError('Model identity mismatch')
    if any(read('loading.json').get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('Model loading mismatch')
    controls=read('loaded-controls.json')
    if not controls['complete'] or controls['task_spec_sha256']!=p['task_spec_sha256'] or len(controls['clean'])!=p['repository_count'] or not all(r['passed'] for r in controls['clean']):raise ValueError('GPU-loaded clean controls failed')
    expected=[{'id':t['id'],'variant':a} for i,t in enumerate(inventory) for a in list(ARMS)[i%2:]+list(ARMS)[:i%2]]
    if c['schedule']!=expected or [{'id':s['id'],'variant':s['variant']} for s in c['stages'] if s['name']=='task']!=expected:raise ValueError('Order changed or task repeated')
    campaigns={a:read(a+'.json') for a in ARMS}
    for campaign in campaigns.values():
        identity=campaign['identity']
        if identity['tasks_sha256']!=p['task_spec_sha256'] or identity['protocol_sha256']!=sha256(root/'protocol.json') or identity['admission_sha256']!=p['admission_sha256']:raise ValueError('Campaign identity mismatch')
        if any(p['frozen_files'].get(name)!=digest for name,digest in identity['source_files'].items()):raise ValueError('Execution changed after freeze')
    prior_path=Path(__file__).resolve().parent.parent/'015-agent-interface/results/2026-11-10/format.json'
    prior=json.loads(prior_path.read_text())
    if sha256(prior_path)!=read('template-roundtrip-probe.json')['prior_campaign_sha256']['format']:raise ValueError('Wrong previous comparison')
    summary=assess(campaigns,p,inventory,prior);audit=read('failure-audit.json')
    if audit['parser_sha256']!=campaigns['schema']['identity']['source_files']['one_gpu_lm/common/actions.py']:raise ValueError('Audit parser changed')
    for arm in ARMS:
        rs=audit['variants'][arm]
        if audit['campaign_sha256'][arm]!=sha256(root/(arm+'.json')) or [r['id'] for r in rs]!=[r['id'] for r in inventory] or not all(r['native_turns_verified'] for r in rs):raise ValueError('Audit mismatch')
        categories=dict(sum((Counter(r['format_categories']) for r in rs),Counter()))
        if sum(categories.values())!=summary['totals'][arm]['format_invalid_calls']:raise ValueError('Audit/parser decisions mismatch')
        summary['totals'][arm].update(format_categories=categories,repeated_accepted_commands=sum(r['repeated_accepted_commands'] for r in rs),consecutive_repeated_commands=sum(r['consecutive_repeated_commands'] for r in rs),maximum_identical_command_streak=max(r['maximum_identical_command_streak'] for r in rs),task_controller_seconds=sum(s['seconds'] for s in c['stages'] if s['name']=='task' and s['variant']==arm))
    summary['task_time_ratio']=summary['totals']['roundtrip']['task_controller_seconds']/summary['totals']['schema']['task_controller_seconds']
    hw=list(csv.DictReader((root/'telemetry.csv').open()));ts=[datetime.fromisoformat(r['utc']).timestamp() for r in hw]
    if not hw or len(hw)!=c['telemetry']['samples'] or not all(b>a for a,b in zip(ts,ts[1:])):raise ValueError('Incomplete or unordered hardware evidence')
    hardware={'samples':len(hw),'collection_errors':c['telemetry']['errors'],'sampled_span_seconds':ts[-1]-ts[0],'max_gap_seconds':max(b-a for a,b in zip(ts,ts[1:]))}
    for key in ['temperature_c','power_w','utilization_percent','memory_used_mib']:
        values=[float(r[key]) for r in hw];hardware[key]={'min':min(values),'median':statistics.median(values),'max':max(values)}
    hardware['sampled_gpu_energy_kwh']=sum((b-a)*(float(x['power_w'])+float(y['power_w']))/2 for a,b,x,y in zip(ts,ts[1:],hw,hw[1:]))/3600000
    summary.update(time={k:c[k] for k in ['started_utc','finished_utc']},hardware=hardware)
    summary['time']['wall_seconds']=c['finished_epoch']-c['started_epoch']
    if summary['time']['wall_seconds']>p['execution']['max_pipeline_seconds']:raise ValueError('Original total time budget exceeded')
    calls=[]
    for arm,campaign in campaigns.items():
        for r in campaign['results']:
            for i,g in enumerate(r['generations']+r['backend_errors'],1):
                calls.append({'variant':arm,'id':r['id'],'call':i,'input_tokens':g.get('input_tokens'),'generated_tokens':g.get('generated_tokens'),'generation_seconds':g.get('seconds'),'format_valid':g.get('action_format_valid'),'native_tool_results':g.get('dialogue',{}).get('tool_results'),'backend_error':g.get('error','')})
    with (root/'call-metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(calls[0]));writer.writeheader();writer.writerows(calls)
    write_json(root/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt':'one-gpu-lm-016','font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained');colors=['#64748b','#0d9488']
    for ax,key,title,label in zip(axes,['passed','format_invalid_calls','consecutive_repeated_commands'],['Historical task success','Rejected action generations','Consecutive repeated commands'],['Passed / 24','Requests','Identical accepted command pairs']):
        values=[summary['totals'][a][key] for a in ARMS];ax.bar(ARMS,values,color=colors);ax.set(title=title,ylabel=label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(i,v+.1,str(v),ha='center')
        ax.set_ylim(0,max(1,max(values)*1.2))
    for ext in ['png','svg']:fig.savefig(root/('comparison.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for arm,color in zip(ARMS,colors):
        rs=[r for r in calls if r['variant']==arm];xs=sorted({r['call'] for r in rs})
        for ax,key,label in zip(axes,['input_tokens','generation_seconds'],['Mean input tokens','Mean generation seconds']):
            ys=[statistics.mean(r[key] for r in rs if r['call']==x and r[key] is not None) if any(r[key] is not None for r in rs if r['call']==x) else float('nan') for x in xs]
            ax.plot(xs,ys,marker='.',label=arm,color=color);ax.set(xlabel='Request index',ylabel=label);ax.grid(alpha=.2)
    axes[0].axhline(8192,color='#991b1b',ls=':',label='Input limit');axes[0].legend();axes[1].legend();fig.suptitle('Descriptive request curves; surviving task populations can differ')
    for ext in ['png','svg']:fig.savefig(root/('request-curves.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)
    write_json(root/'report-environment.json',{'python':platform.python_version(),'matplotlib':matplotlib.__version__,'report_sha256':sha256(Path(__file__))})
    table='\n'.join(f"| {a} | {t['passed']}/24 | {t['format_invalid_calls']}/{t['generation_calls']} | {t['context_limit_requests']} | {t['consecutive_repeated_commands']} | {t['task_controller_seconds']/60:.1f} |" for a,t in summary['totals'].items())
    repo_table='\n'.join(f"| {repo} | {v['tasks']} | {v['passed']['schema']} | {v['passed']['roundtrip']} |" for repo,v in summary['per_repository'].items())
    label=lambda ids:', '.join(ids) if ids else 'none'
    verdict='Meets the diagnostic gate; requires a new independent holdout.' if summary['candidate_for_new_holdout'] else 'Does not meet the frozen diagnostic gate.'
    (root/'README.md').write_text(f'''# 016 · Native tool round-trip results

**{verdict} No deployment or generalization claim.** All 24 tasks were already observed in 015. No weights were updated.

| Policy | Historical successes | Invalid / generated requests | Context-limit requests | Consecutive repeated commands | Task controller minutes |
| --- | --- | --- | --- | --- | --- |
{table}

![Historical outcomes](comparison.png)

Paired improvements: {label(summary['improved'])}. Regressions: {label(summary['regressed'])}. Net gain: {summary['net_gain']:+d}. Known target `{summary['target']['id']}`: {summary['target']['passed']}. Controller time ratio: {summary['task_time_ratio']:.3f}×. Every task is retained, including failures. Higher totals cannot erase individual regressions.

| Repository | Historical tasks | schema | roundtrip |
| --- | --- | --- | --- |
{repo_table}

The concurrent schema control compared with 015 format: improvements {label(summary['concurrent_control_vs_015_format']['improved'])}; regressions {label(summary['concurrent_control_vs_015_format']['regressed'])}. This is a repeat-variation check, not a new randomized replicate or a best-of selection. A single greedy run can still vary with numerical/runtime behavior.

## Intervention and evidence

Both policies use identical native tool definitions and system/error prompts, the unchanged strict action parser, the same BF16 base and the same budgets. Roundtrip converts executed commands to assistant `tool_calls` and actual observations to `role=tool`, using the pinned template. This bundles call and result serialization. No anti-loop hint, extra calls, semantic summary or automatic action repair. Default policy remains unchanged. [Protocol](protocol.json), [public preregistration](preregistration.json), [tokenizer replay](template-roundtrip-probe.json), [failure audit](failure-audit.json) and [raw artifact hashes](raw-artifacts.json) retain provenance.

Repetition counts measure exact accepted command strings, not semantic equivalence or proven waste; legitimate repeated tests can count. The audit validates every native-turn receipt against actual executed history, without exporting commands or rerunning code. Failed grader commands indicate repair failures, not GPU crashes. [Summary](summary.json) retains errors, output limits, no-source-change failures and all paired outcomes.

![Request curves](request-curves.png)

Curves are descriptive means at each request index; surviving tasks can differ. Failed generation timings are missing, not zero. [Per-request metrics](call-metrics.csv) retain denominators and backend-limit failures. Controller time sums include tools and grading; interleaved arms do not have separate contiguous elapsed spans.

Actual UTC: {c['started_utc']} to {c['finished_utc']}; {summary['time']['wall_seconds']/60:.2f} minutes. Hardware: {hardware['samples']} samples, {hardware['collection_errors']} collection errors, maximum gap {hardware['max_gap_seconds']:.2f}s; sampled peaks {hardware['temperature_c']['max']:.0f}C, {hardware['memory_used_mib']['max']:.0f}MiB and {hardware['power_w']['max']:.1f}W. Sampled GPU energy {hardware['sampled_gpu_energy_kwh']:.3f}kWh uses trapezoid interpolation and excludes CPU/system power; five-second sampling can miss peaks.

## Scope and reuse

These are constructed defects with visible upstream regression suites, not real issue resolution or SWE-bench. All tasks are historical, tests visible, base pretraining exposure unknown, and repository clusters small. A positive result only motivates a newly frozen independent study. [Reusable workflow](../../README.md). Original code is MIT; source/model rights remain separate. Raw tasks, source snapshots, commands, traces and model files remain private. Exact score reconstruction requires private inputs; the workflow supports others' authorized repositories. No model weights were published.
''')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);render(p.parse_args().run)
