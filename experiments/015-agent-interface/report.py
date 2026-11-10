"""Fail-closed paired analysis for the frozen four-arm interface experiment."""
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

ARMS=('baseline','format','context','combined')
CONTRASTS=(('baseline','format'),('baseline','context'),('baseline','combined'),('format','combined'),('context','combined'))


def assess(campaigns,protocol,inventory):
    if set(campaigns)!=set(ARMS) or set(protocol['variants'])!=set(ARMS):
        raise ValueError('Require all four registered variants')
    ids=[r['id'] for r in inventory]
    if len(ids)!=len(set(ids)) or len(ids)!=protocol['execution']['expected_tasks_per_variant']:
        raise ValueError('Task inventory membership is invalid')
    rows={}
    for arm in ARMS:
        c=campaigns[arm]
        if not c['complete'] or c['identity']['ids']!=ids or c['identity']['variant']!=arm or c['identity']['policy']!=protocol['variants'][arm]:
            raise ValueError('Incomplete or wrong campaign policy')
        if [r['id'] for r in c['results']]!=ids:
            raise ValueError('Unpaired or duplicated final tasks')
        rows[arm]={r['id']:r for r in c['results']}
        for r,t in zip(c['results'],inventory):
            if r['variant']!=arm or r['agent_policy']!=protocol['variants'][arm]:
                raise ValueError('Wrong task-level policy')
            if any(r[k]!=t[k] for k in ['repository','split','family']) or type(r['passed']) is not bool:
                raise ValueError('Task role or outcome changed')
            if r['calls']>protocol['agent']['max_model_calls'] or r['generated_tokens']>protocol['agent']['max_total_generated_tokens']:
                raise ValueError('Task exceeded frozen call/token budget')
            if len(r['generations'])+len(r['backend_errors'])!=r['calls']:
                raise ValueError('Missing generation or backend failure evidence')
            if sum(g['generated_tokens'] for g in r['generations'])!=r['generated_tokens']:
                raise ValueError('Generated token accounting mismatch')
            for g in r['generations']:
                if g['agent_policy']!=r['agent_policy'] or g['adapter'] or g['input_tokens']>protocol['agent']['max_input_tokens']:
                    raise ValueError('Wrong serving policy, adapter or input budget')
                if type(g['action_format_valid']) is not bool:
                    raise ValueError('Missing strict-parser outcome')
                ctx=g['context']
                if ctx['policy']!=r['agent_policy']['context'] or ctx['packed_input_tokens']!=g['input_tokens']:
                    raise ValueError('Context token receipt mismatch')
                if ctx['policy']=='full' and (ctx['raw_input_tokens']!=g['input_tokens'] or ctx['dropped_rounds']):
                    raise ValueError('Baseline history was silently changed')
    scopes={'historical':[t['id'] for t in inventory if t['split']=='dev'],
            'fresh':[t['id'] for t in inventory if t['split']=='test']}
    if {k:len(v) for k,v in scopes.items()}!={'historical':protocol['evaluation_task_counts']['dev'],'fresh':protocol['evaluation_task_counts']['test']}:
        raise ValueError('Wrong partition counts')
    contrasts={}
    for a,b in CONTRASTS:
        details={}
        for scope,selected in scopes.items():
            improved=[i for i in selected if not rows[a][i]['passed'] and rows[b][i]['passed']]
            regressed=[i for i in selected if rows[a][i]['passed'] and not rows[b][i]['passed']]
            details[scope]={'pairs':len(selected),'reference_passed':sum(rows[a][i]['passed'] for i in selected),
                'treatment_passed':sum(rows[b][i]['passed'] for i in selected),'improved':improved,'regressed':regressed,
                'net_gain':len(improved)-len(regressed)}
        contrasts[b+'-vs-'+a]=details
    gate=protocol['promotion'];promotion={}
    for arm in ARMS[1:]:
        d=contrasts[arm+'-vs-baseline']
        checks={'fresh_gain':d['fresh']['net_gain']>=gate['minimum_net_fresh_gain'],
            'no_fresh_regressions':len(d['fresh']['regressed'])<=gate['maximum_regressed_fresh_tasks'],
            'no_historical_regressions':len(d['historical']['regressed'])<=gate['maximum_regressed_historical_tasks']}
        promotion[arm]={'checks':checks,'advance_to_replication':all(checks.values())}
    per_repository={}
    for repo in sorted({t['repository'] for t in inventory}):
        selected=[t['id'] for t in inventory if t['repository']==repo]
        per_repository[repo]={'tasks':len(selected),'split':rows['baseline'][selected[0]]['split'],
            'passed':{arm:sum(rows[arm][i]['passed'] for i in selected) for arm in ARMS}}
    totals={};call_rows=[]
    for arm in ARMS:
        rs=list(rows[arm].values());gs=[g for r in rs for g in r['generations']]
        errors=[e for r in rs for e in r['backend_errors']]
        seconds=sum(g['seconds'] for g in gs)
        totals[arm]={'passed':{scope:sum(rows[arm][i]['passed'] for i in selected) for scope,selected in scopes.items()},
            'calls':sum(r['calls'] for r in rs),'successful_generation_calls':len(gs),'format_invalid_calls':sum(not g['action_format_valid'] for g in gs),
            'format_invalid_rate':sum(not g['action_format_valid'] for g in gs)/len(gs) if gs else None,
            'generation_limit_hits':sum(g['hit_generation_limit'] for g in gs),
            'context_limit_requests':sum(e.get('error')=='context_limit' for e in errors),
            'backend_error_categories':dict(Counter(e.get('error') for e in errors)),
            'compacted_generation_calls':sum(g['context']['dropped_rounds']>0 for g in gs),
            'generated_tokens':sum(g['generated_tokens'] for g in gs),'generation_seconds':seconds,
            'generation_tokens_per_second':sum(g['generated_tokens'] for g in gs)/seconds if seconds else None,
            'packing_seconds':sum(g.get('packing_seconds',0) for g in gs)+sum(e.get('packing_seconds',0) for e in errors),
            'task_wall_seconds':sum(r['seconds'] for r in rs),'median_task_seconds':statistics.median(r['seconds'] for r in rs),
            'tool_seconds':sum(r['tool_seconds'] for r in rs),'tool_calls':sum(r['tool_calls'] for r in rs),
            'exit_status':dict(Counter(r['exit_status'] for r in rs)),
            'failed_without_source_change':sum(not r['passed'] and not r['modified_source_files'] for r in rs),
            'grading_failures':dict(Counter('nonzero_regression_exit' if r.get('reason')=='runtime_error' and r.get('returncode',0)!=0 else r.get('reason') for r in rs if not r['passed']))}
        for r in rs:
            for index,g in enumerate(r['generations']+r['backend_errors'],1):
                ctx=g.get('context',{})
                call_rows.append({'variant':arm,'task_id':r['id'],'split':r['split'],'call':index,
                    'raw_tokens':ctx.get('raw_input_tokens'),'packed_tokens':ctx.get('packed_input_tokens'),
                    'dropped_rounds':ctx.get('dropped_rounds'),'generated_tokens':g.get('generated_tokens'),
                    'generation_seconds':g.get('seconds'),'packing_seconds':g.get('packing_seconds'),
                    'format_valid':g.get('action_format_valid'),'hit_generation_limit':g.get('hit_generation_limit'),
                    'backend_error':g.get('error','')})
    interaction={scope:sum(int(rows['combined'][i]['passed'])-int(rows['format'][i]['passed'])-int(rows['context'][i]['passed'])+int(rows['baseline'][i]['passed']) for i in selected) for scope,selected in scopes.items()}
    return {'schema_version':1,'pairs_per_variant':len(ids),'scope_sizes':{k:len(v) for k,v in scopes.items()},
        'totals':totals,'contrasts':contrasts,'interaction_net_tasks':interaction,'per_repository':per_repository,
        'promotion':promotion,'daily_use_certified':False},call_rows


def render(root):
    read=lambda name:json.loads((root/name).read_text())
    protocol=read('protocol.json');inventory=read('task-inventory.json');controller=read('controller.json')
    if not controller['complete'] or controller['protocol_sha256']!=sha256(root/'protocol.json') or controller['completed_runs']!=protocol['execution']['expected_model_task_runs']:
        raise ValueError('Controller did not complete frozen study')
    if sha256(root/'admission.json')!=protocol['admission_sha256']:
        raise ValueError('Admission changed')
    for local,remote in [('profile.json','experiments/015-agent-interface/model-profile.json'),('sources.json','experiments/015-agent-interface/sources.json'),('source-snapshots.json','source-snapshots.json'),('template-context-probe.json','template-context-probe.json'),('model-verification.json','model-verification.json')]:
        if sha256(root/local)!=protocol['frozen_files'][remote]:raise ValueError('Frozen evidence changed: '+local)
    artifact=read('artifact.json')
    if artifact['adapter'] or artifact['adapter_sha256'] or artifact['profile_sha256']!=sha256(root/'profile.json') or artifact['revision']!=protocol['revision']:
        raise ValueError('Model artifact mismatch')
    if any(read('loading.json').get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise ValueError('Model loading mismatch')
    campaigns={arm:read(arm+'.json') for arm in ARMS}
    for c in campaigns.values():
        ident=c['identity']
        if ident['tasks_sha256']!=protocol['task_spec_sha256'] or ident['protocol_sha256']!=sha256(root/'protocol.json') or ident['admission_sha256']!=protocol['admission_sha256']:
            raise ValueError('Campaign input identity mismatch')
        if any(protocol['frozen_files'].get(name)!=digest for name,digest in ident['source_files'].items()):
            raise ValueError('Execution code differs from frozen protocol')
    expected=[{'id':t['id'],'variant':arm} for i,t in enumerate(inventory) for arm in list(ARMS)[i%4:]+list(ARMS)[:i%4]]
    if controller['schedule']!=expected or [{'id':s['id'],'variant':s['variant']} for s in controller['stages'] if s['name']=='task']!=expected:
        raise ValueError('Execution order changed or runs duplicated')
    summary,call_rows=assess(campaigns,protocol,inventory)
    with (root/'call-metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(call_rows[0]));writer.writeheader();writer.writerows(call_rows)
    telemetry=list(csv.DictReader((root/'telemetry.csv').open()));ts=[datetime.fromisoformat(r['utc']).timestamp() for r in telemetry]
    if len(telemetry)!=controller['telemetry']['samples'] or not telemetry:
        raise ValueError('Telemetry receipt mismatch')
    hw={'samples':len(ts),'collection_errors':controller['telemetry']['errors'],
        'max_gap_seconds':max((b-a for a,b in zip(ts,ts[1:])),default=0),'sampled_span_seconds':ts[-1]-ts[0]}
    for column in ['temperature_c','power_w','utilization_percent','memory_used_mib']:
        values=[float(r[column]) for r in telemetry];hw[column]={'min':min(values),'median':statistics.median(values),'max':max(values)}
    hw['sampled_gpu_energy_kwh']=sum((b-a)*(float(x['power_w'])+float(y['power_w']))/2 for a,b,x,y in zip(ts,ts[1:],telemetry,telemetry[1:]))/3600000
    summary.update(hardware=hw,time={k:controller[k] for k in ['started_utc','finished_utc']})
    summary['time']['wall_seconds']=controller['finished_epoch']-controller['started_epoch']
    if summary['time']['wall_seconds']>protocol['execution']['max_pipeline_seconds']:
        raise ValueError('Study exceeded frozen total budget')
    for arm in ARMS:
        summary['totals'][arm]['task_controller_seconds']=sum(s['seconds'] for s in controller['stages'] if s['name']=='task' and s['variant']==arm)
    for a,b in CONTRASTS:
        x,y=summary['totals'][a],summary['totals'][b]
        summary['contrasts'][b+'-vs-'+a]['cost']={'task_controller_seconds_difference':y['task_controller_seconds']-x['task_controller_seconds'],
            'task_controller_time_ratio':y['task_controller_seconds']/x['task_controller_seconds'] if x['task_controller_seconds'] else None}
    write_json(root/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt':'one-gpu-lm-015','font.family':'DejaVu Sans'})
    colors=['#65748b','#2563eb','#0d9488','#d97706'];labels=list(ARMS)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for i,arm in enumerate(ARMS):
        axes[0].bar([i-.17,i+.17],[summary['totals'][arm]['passed']['historical'],summary['totals'][arm]['passed']['fresh']],width=.32,color=[colors[i],colors[i]],alpha=.8)
        axes[1].bar(i,100*(summary['totals'][arm]['format_invalid_rate'] or 0),color=colors[i])
        axes[2].bar(i,summary['totals'][arm]['task_controller_seconds']/60,color=colors[i])
    axes[0].set(title='Final successes: historical / fresh',ylabel='Tasks passed (each group n=12)',ylim=(0,13))
    for i,arm in enumerate(ARMS):
        for shift,scope in [(-.17,'historical'),(.17,'fresh')]:
            n=summary['totals'][arm]['passed'][scope];axes[0].text(i+shift,n+.15,str(n),ha='center',fontsize=9)
    axes[1].set(title='Rejected generated actions',ylabel='Invalid / generated requests (%)')
    axes[2].set(title='Repository task controller time',ylabel='Minutes; interleaved task sums')
    for ax in axes:ax.set_xticks(range(4),labels,rotation=15);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ext in ['png','svg']:fig.savefig(root/('comparison.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True,layout='constrained')
    for ax,arm,color in zip(axes.flat,ARMS,colors):
        selected=[r for r in call_rows if r['variant']==arm and r['raw_tokens'] is not None]
        xs=sorted({r['call'] for r in selected})
        for key,style,label in [('raw_tokens','--','Full recorded history'),('packed_tokens','-','Sent to model')]:
            means=[statistics.mean(r[key] for r in selected if r['call']==x) for x in xs]
            ax.plot(xs,means,style,color=color,label=label)
        ax.axhline(8192,color='#991b1b',alpha=.5,linestyle=':',label='Input limit')
        ax.set(title=arm,xlabel='Model request index',ylabel='Mean input tokens');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Context by request index; surviving task populations differ at later calls')
    for ext in ['png','svg']:fig.savefig(root/('context-curves.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)
    write_json(root/'report-environment.json',{'python':platform.python_version(),'matplotlib':matplotlib.__version__,'report_sha256':sha256(Path(__file__))})
    table='\n'.join(f"| {a} | {summary['totals'][a]['passed']['historical']}/12 | {summary['totals'][a]['passed']['fresh']}/12 | {summary['totals'][a]['format_invalid_calls']}/{summary['totals'][a]['successful_generation_calls']} | {summary['totals'][a]['context_limit_requests']} | {summary['totals'][a]['compacted_generation_calls']} | {summary['totals'][a]['task_controller_seconds']/60:.1f} |" for a in ARMS)
    promoted=[a for a,v in summary['promotion'].items() if v['advance_to_replication']]
    conclusion=('Eligible for independent replication: '+', '.join(promoted)+'.') if promoted else 'No policy passes the preregistered replication gate.'
    (root/'README.md').write_text(f'''# 015 · Tool format and bounded context

**{conclusion} No daily-use model certification is implied.**

The same unchanged Qwen27B base ran all four arms. No weights were updated. The protocol was public before any task rollout; [protocol.json](protocol.json), [preregistration.json](preregistration.json) and artifact hashes retain the identity.

| Policy | Historical | Fresh | Invalid / generated calls | Context-limit requests | Compacted generation calls | Task controller minutes |
| --- | --- | --- | --- | --- | --- | --- |
{table}

![Paired outcomes and cost](comparison.png)

Each scope has12 tasks. Historical means the already observed014 holdout. Fresh means6 ItsDangerous and6 qs constructed defects, with clean/broken/restored controls. Tests are visible; these are not historical issues or SWE-bench. Unknown pretraining exposure, two small repository clusters and one deterministic decode limit generalization. Every individual improved/regressed ID, five predeclared contrasts and the factorial interaction are retained in [summary.json](summary.json). All contrasts are reported; no post-hoc best-policy selection is hidden.

## What changed

Format supplies the pinned native chat template with one bash JSON tool schema and matching system/error instructions. The action parser and sandbox remain unchanged; no incomplete response is repaired into an executable command. This factor does not separate schema effects from prompt effects.

Context acts above7168 actual template tokens: remove oldest complete interaction pairs toward6144, retain system, full task and newest pair, and append an omission notice. The on-disk workspace and full private audit history persist. Same8192 hard input limit. Losing history can harm repair quality; fewer context errors alone is not success.

![Context curves](context-curves.png)

The curves average requests at each call index, including recorded backend-limit failures. Later calls contain different surviving task populations, so they are descriptive rather than matched causal token curves. [call-metrics.csv](call-metrics.csv) retains every request's token counts, discarded rounds, parser outcome and generation timing. Full raw history may exceed the input limit while a packed request fits; this is expected for bounded-context arms.

## Failure and cost interpretation

Format-invalid calls include rejected generations before an agent eventually stops; repeated-format exit counts are separate. A nonzero final regression command is a failed repair, not a GPU crash. Per-arm backend errors, no-source-change failures, output-limit hits and exit categories are in the summary. Tokens per second measures successful sequential generation requests only, not production serving throughput. Task controller times include tools and grading; interleaving means they are sums, not the elapsed span of an arm. Packing time is separately recorded. Missing generation times on failed requests are not treated as zero-cost inference.

Actual UTC: {controller['started_utc']} to {controller['finished_utc']}; total{summary['time']['wall_seconds']/3600:.3f} hours. Hardware sampling: {hw['samples']} records, {hw['collection_errors']} collection errors, maximum gap{hw['max_gap_seconds']:.2f}s. Sampled peaks: {hw['temperature_c']['max']:.0f}C, {hw['power_w']['max']:.1f}W, {hw['memory_used_mib']['max']:.0f}MiB. Sampled GPU energy approximately{hw['sampled_gpu_energy_kwh']:.3f}kWh, trapezoid interpolation excluding CPU/system power and time outside sampled coverage. Five-second sampling can miss peaks.

## Reuse and limitations

The initial environment packaging failures and their correction were recorded before model rollout in [preparation-recovery.json](preparation-recovery.json). No task or policy was selected using015 scores. These fresh tasks become historical regression if future tuning uses their outcomes.

Reusable policies, frozen orchestration, scoring, context instrumentation and this report are in the [experiment workflow](../../README.md). Original code is MIT; upstream BSD-3-Clause/MIT rights remain separate. Raw tasks, source snapshots, answers, trajectories and model files remain private. Exact score reconstruction requires private inputs; others can run the workflow with their own authorized repositories. No ModelScope weight release occurred.
''')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    render(p.parse_args().run)
