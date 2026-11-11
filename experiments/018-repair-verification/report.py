"""Analyze the fixed historical-to-fresh study without treating an edit as success."""
import argparse,csv,json,platform,statistics,sys
from collections import Counter
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
ARMS=('control','verify')
SPLITS={'dev':'historical','test':'fresh'}

def assess(campaigns,p,inventory):
 ids=[t['id'] for t in inventory]
 if len(ids)!=len(set(ids)) or len(ids)!=p['execution']['expected_tasks_per_variant']:raise ValueError('Incomplete inventory')
 if dict(Counter(t['split'] for t in inventory))!=p['evaluation_task_counts']:raise ValueError('Changed exposure split')
 if {t['repository'] for t in inventory if t['split']=='dev'} & {t['repository'] for t in inventory if t['split']=='test'}:raise ValueError('Holdout repositories overlap history')
 rows={};totals={};paired=[]
 for arm in ARMS:
  c=campaigns[arm];rs=c['results'];policy=p['variants'][arm]
  if not c['complete'] or c['identity']['ids']!=ids or c['identity']['variant']!=arm or c['identity']['policy']!=policy or [r['id'] for r in rs]!=ids:raise ValueError('Missing or changed paired campaign')
  rows[arm]={r['id']:r for r in rs}
  for r,t in zip(rs,inventory):
   if any(r[k]!=t[k] for k in ['repository','split','family']) or r['agent_policy']!=policy or r['variant']!=arm or type(r['passed']) is not bool:raise ValueError('Task identity changed')
   gs=r['generations'];requests=gs+r['backend_errors'];steps=r['source_progress']
   if r['calls']>p['agent']['max_model_calls'] or len(requests)!=r['calls'] or r['generated_tokens']!=sum(g['generated_tokens'] for g in gs) or r['generated_tokens']>p['agent']['max_total_generated_tokens']:raise ValueError('Call or token budget mismatch')
   if len(steps)!=r['tool_calls'] or any(a['call_index']>=b['call_index'] for a,b in zip(steps,steps[1:])) or any(s['call_index']<1 or s['call_index']>r['calls'] for s in steps):raise ValueError('Missing or reordered progress observations')
   for index,g in enumerate(requests):
    pr=g['progress']
    if pr['completed_calls']!=index or pr['remaining_calls']!=p['agent']['max_model_calls']-index or not pr['enabled']:raise ValueError('Progress policy/call accounting mismatch')
    expected=pr['changed_editable_files']==0 and (index>=4 or pr['identical_command_streak']>=2)
    if pr['reminder']!=expected or bool(pr['reminder_chars'])!=expected or len(g['input_tokens_sha256'])!=64:raise ValueError('Wrong reminder or missing actual input hash')
    vr=g['verification'];ve=arm=='verify' and pr['changed_editable_files']>0 and pr['invalid_editable_files']==0
    if vr['enabled']!=(arm=='verify') or vr['completed_calls']!=index or vr['remaining_calls']!=p['agent']['max_model_calls']-index or vr['current_source_sha256']!=pr['editable_state_sha256'] or vr['reminder']!=ve or bool(vr['reminder_chars'])!=ve or vr['check_status'] not in {'not_run','stale','changed_during_check','failed','passed'}:raise ValueError('Verification trigger or budget changed')
    if g['context']['policy']!='full' or g['context']['dropped_rounds'] or g['context']['raw_input_tokens']!=g['input_tokens'] or g['context']['packed_input_tokens']!=g['input_tokens']:raise ValueError('Hidden context change')
    if g['dialogue']['structured_tool_calls']!=g['dialogue']['tool_results']:raise ValueError('Native history mismatch')
   for g in gs:
    if g['adapter'] or g['agent_policy']!=policy or g['input_tokens']>p['agent']['max_input_tokens'] or g['generated_tokens']>p['agent']['max_generated_tokens_per_call'] or type(g['action_format_valid']) is not bool:raise ValueError('Serving or budget mismatch')
  gs=[g for r in rs for g in r['generations']];requests=[g for r in rs for g in r['generations']+r['backend_errors']]
  totals[arm]={'passed':{name:sum(r['passed'] for r in rs if r['split']==split) for split,name in SPLITS.items()},'tasks':{name:sum(r['split']==split for r in rs) for split,name in SPLITS.items()},'calls':sum(r['calls'] for r in rs),'generation_calls':len(gs),'generated_tokens':sum(r['generated_tokens'] for r in rs),'generation_seconds':sum(g['seconds'] for g in gs),'task_seconds':sum(r['seconds'] for r in rs),'tool_seconds':sum(r['tool_seconds'] for r in rs),'format_invalid_calls':sum(not g['action_format_valid'] for g in gs),'generation_limit_hits':sum(g['hit_generation_limit'] for g in gs),'backend_errors':dict(Counter(g.get('error') for r in rs for g in r['backend_errors'])),'context_limit_requests':sum(g.get('error')=='context_limit' for r in rs for g in r['backend_errors']),'reminder_requests':sum(g['progress']['reminder'] for g in requests),'failed_without_final_source_change':sum(not r['passed'] and not r['modified_source_files'] for r in rs),'failed_without_any_observed_edit':sum(not r['passed'] and not any(s['changed_editable_files'] for s in r['source_progress']) for r in rs),'exit_status':dict(Counter(r['exit_status'] for r in rs))}
  totals[arm]['processed_input_tokens']=sum(g['input_tokens'] for g in gs)
  totals[arm]['requested_input_tokens_including_errors']=sum(g['input_tokens'] for g in requests)
  totals[arm]['verification_reminder_requests']=sum(g['verification']['reminder'] for g in requests)
  totals[arm]['failed_after_any_edit']=sum(not r['passed'] and any(s['changed_editable_files'] for s in r['source_progress']) for r in rs)
  totals[arm]['call_limit_without_any_edit']=sum(r['calls']==p['agent']['max_model_calls'] and not any(s['changed_editable_files'] for s in r['source_progress']) for r in rs)
  totals[arm]['by_split']={label:{'calls':sum(r['calls'] for r in rs if r['split']==split),'failed_without_any_observed_edit':sum(not r['passed'] and not any(s['changed_editable_files'] for s in r['source_progress']) for r in rs if r['split']==split),'failed_after_any_edit':sum(not r['passed'] and any(s['changed_editable_files'] for s in r['source_progress']) for r in rs if r['split']==split),'call_limit_without_any_edit':sum(r['calls']==p['agent']['max_model_calls'] and not any(s['changed_editable_files'] for s in r['source_progress']) for r in rs if r['split']==split)} for split,label in SPLITS.items()}
 for t in inventory:
  a,b=(rows[arm][t['id']] for arm in ARMS);outcome='improved' if b['passed'] and not a['passed'] else 'regressed' if a['passed'] and not b['passed'] else 'both_passed' if a['passed'] else 'both_failed'
  item={k:t[k] for k in ['id','repository','split','family']};item['outcome']=outcome
  for arm,r in [('control',a),('verify',b)]:
   item.update({arm+'_'+k:r[k] for k in ['passed','calls','seconds','generated_tokens','exit_status']});item[arm+'_first_edit_call']=next((s['call_index'] for s in r['source_progress'] if s['changed_editable_files']),None);item[arm+'_final_changed_files']=len(r['modified_source_files'])
  paired.append(item)
 contrasts={name:{kind:[r['id'] for r in paired if r['split']==split and r['outcome']==kind] for kind in ['improved','regressed']} for split,name in SPLITS.items()}
 gate=p['promotion'];checks={'fresh_net_gain':len(contrasts['fresh']['improved'])-len(contrasts['fresh']['regressed'])>=gate['minimum_net_fresh_gain'],'no_fresh_regressions':len(contrasts['fresh']['regressed'])<=gate['maximum_regressed_fresh_tasks'],'no_historical_regressions':len(contrasts['historical']['regressed'])<=gate['maximum_regressed_historical_tasks'],'both_targets_pass':all(rows['verify'][i]['passed'] for i in gate['target_tasks'])} if 'target_tasks' in gate else {}
 return {'schema_version':1,'totals':totals,'contrasts':contrasts,'paired':paired,'targets':{i:{a:rows[a][i]['passed'] for a in ARMS} for i in gate.get('target_tasks',[])},'gate_checks':checks,'candidate_for_replication':bool(checks) and all(checks.values()),'deployment_allowed':False,'weight_updates':0,'per_repository':{repo:{'split':next(t['split'] for t in inventory if t['repository']==repo),'tasks':sum(t['repository']==repo for t in inventory),'passed':{a:sum(r['passed'] for r in rows[a].values() if r['repository']==repo) for a in ARMS}} for repo in sorted({t['repository'] for t in inventory})}}


def render(root):
 read=lambda name:json.loads((root/name).read_text())
 exp=Path(__file__).parent;p=read('protocol.json');controller=read('controller.json');inventory=read('task-inventory.json')
 pub=read('preregistration.json');launch=read('launch.json')
 if not pub['anonymous'] or pub['ci']['conclusion']!='success' or pub['commit']!=launch['source_commit'] or pub['protocol_sha256']!=sha256(root/'protocol.json') or datetime.fromisoformat(pub['verified_utc'])>=datetime.fromisoformat(controller['started_utc']):raise ValueError('Public preregistration must precede launch')
 if not controller['complete'] or not controller['telemetry']['thread_stopped'] or controller.get('error_type') or controller['completed_runs']!=p['execution']['expected_model_task_runs'] or controller['protocol_sha256']!=sha256(root/'protocol.json') or controller['finished_epoch']>controller['deadline_epoch']:raise ValueError('Incomplete run or deadline exceeded')
 if sha256(root/'task-inventory.json')!=sha256(exp/'preflight/task-inventory.json'):raise ValueError('Inventory changed')
 frozen={'admission.json':'runs/admission/admission.json','profile.json':'experiments/018-repair-verification/model-profile.json',
         **{n:'experiments/018-repair-verification/'+n for n in ['sources.json','intervention.json','diagnosis.json','preparation-plan.json']},
         **{n:n for n in ['source-snapshots.json','model-verification.json','template-verification-probe.json','reference-screen.json','reference-screen-broad.json','preparation-reuse.json','checks-linux.json']}}
 for name,relative in frozen.items():
  if sha256(root/name)!=p['frozen_files'][relative]:raise ValueError('Changed frozen input: '+name)
 artifact=read('artifact.json');controls=read('loaded-controls.json')
 if any(artifact[k]!=p[k] for k in ['model_id','revision']) or artifact['adapter'] or artifact['adapter_sha256'] or artifact['profile_sha256']!=sha256(root/'profile.json'):raise ValueError('Model identity changed')
 if any(read('loading.json').get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('Model load mismatch')
 if not controls['complete'] or controls['task_spec_sha256']!=p['task_spec_sha256'] or len(controls['clean'])!=p['repository_count'] or not all(r['passed'] for r in controls['clean']):raise ValueError('Loaded clean controls failed')
 expected=[{'id':t['id'],'variant':arm} for i,t in enumerate(inventory) for arm in list(ARMS)[i%2:]+list(ARMS)[:i%2]]
 if controller['schedule']!=expected or [{'id':s['id'],'variant':s['variant']} for s in controller['stages'] if s['name']=='task']!=expected:raise ValueError('Changed order or repeated score')
 fresh_repos={t['repository'] for t in inventory if t['split']=='test'}
 screens={name:read(name)['screens'] for name in ['reference-screen.json','reference-screen-after.json']}
 for name,rows in screens.items():
  if {s['repository'] for s in rows}!=fresh_repos or len(rows)!=len(fresh_repos) or not all(s['screen_passed'] and s['scanner_sha256']==p['frozen_files']['experiments/018-repair-verification/reference_screen.py'] for s in rows):raise ValueError('Incomplete or failed reference screen: '+name)
 campaigns={arm:read(arm+'.json') for arm in ARMS};audit=read('failure-audit.json')
 if audit['audit_source_sha256']!=sha256(exp/'audit.py'):raise ValueError('Different audit implementation')
 for arm,c in campaigns.items():
  identity=c['identity'];rows=audit['variants'][arm]
  if identity['tasks_sha256']!=p['task_spec_sha256'] or identity['protocol_sha256']!=sha256(root/'protocol.json') or identity['admission_sha256']!=p['admission_sha256'] or any(p['frozen_files'].get(k)!=v for k,v in identity['source_files'].items()):raise ValueError('Campaign input identity changed')
  if 'one_gpu_lm/common/repair_verification.py' not in identity['source_files']:raise ValueError('Verification source untracked')
  if audit['campaign_sha256'][arm]!=sha256(root/(arm+'.json')) or audit['parser_sha256']!=identity['source_files']['one_gpu_lm/common/actions.py'] or [r['id'] for r in rows]!=[t['id'] for t in inventory] or not all(r['progress_and_native_turns_verified'] and r['final_source_verified'] for r in rows) or sum(r['input_token_hashes_verified'] for r in rows)!=sum(r['calls'] for r in c['results']):raise ValueError('Missing input/source/verification audit')
 summary=assess(campaigns,p,inventory)
 summary['same_input_output_comparison']=audit['same_input_output_comparison']
 summary['reference_screening']={'before_and_after_passed':True,'screened_repositories':sorted(fresh_repos),'scope':'Bounded known Python package paths, exact selected source hashes and package paths in zip/wheel archives; not all renamed/embedded/semantic copies or pretraining.'}
 audited={arm:{r['id']:r for r in audit['variants'][arm]} for arm in ARMS}
 for row in summary['paired']:
  for arm in ARMS:
   evidence=audited[arm][row['id']]
   row[arm+'_first_observed_check_pass_call']=next((s['call'] for s in evidence['verification_events'] if s['changed_files'] and s['check_status']=='passed'),None)
   row[arm+'_final_observed_check_status']=evidence['final_observed_check_status']
 for arm in ARMS:
  total=summary['totals'][arm]
  total['controller_seconds']=sum(s['seconds'] for s in controller['stages'] if s['name']=='task' and s['variant']==arm)
  total['verification_extra_input_tokens']=sum(sum(r['verification_extra_input_tokens']) for r in audit['variants'][arm])
  total['consecutive_repeated_commands']=sum(r['consecutive_repeated_commands'] for r in audit['variants'][arm])
  total['final_observed_check_statuses']=dict(Counter(r['final_observed_check_status'] for r in audit['variants'][arm]))
  total['observed_check_pass_but_final_grade_failed']=sum(r['final_observed_check_status']=='passed' and not r['passed'] for r in audit['variants'][arm])
 prior_path=exp.parent/'017-edit-progress/results/2026-11-11/progress.json'
 if sha256(prior_path)!=read('template-verification-probe.json')['prior_campaign_sha256']:raise ValueError('Wrong017baseline reference')
 old={r['id']:r for r in json.loads(prior_path.read_text())['results']}
 historical=[r for r in campaigns['control']['results'] if r['split']=='dev']
 summary['control_vs017_progress']={'improved':[r['id'] for r in historical if r['passed'] and not old[r['id']]['passed']],'regressed':[r['id'] for r in historical if not r['passed'] and old[r['id']]['passed']]}
 summary['time']={k:controller[k] for k in ['started_utc','finished_utc']}
 summary['time']['wall_seconds']=controller['finished_epoch']-controller['started_epoch']
 summary['task_controller_time_ratio']=summary['totals']['verify']['controller_seconds']/summary['totals']['control']['controller_seconds']
 hw=list(csv.DictReader((root/'telemetry.csv').open()));ts=[datetime.fromisoformat(r['utc']).timestamp() for r in hw]
 if len(hw)!=controller['telemetry']['samples'] or len(hw)<2 or not all(b>a for a,b in zip(ts,ts[1:])):raise ValueError('Missing hardware samples')
 summary['hardware']={'samples':len(hw),'collection_errors':controller['telemetry']['errors'],'span_seconds':ts[-1]-ts[0],
  'max_gap_seconds':max(b-a for a,b in zip(ts,ts[1:])),
  'sampled_gpu_energy_kwh':sum((b-a)*(float(x['power_w'])+float(y['power_w']))/2 for a,b,x,y in zip(ts,ts[1:],hw,hw[1:]))/3600000}
 for key in ['temperature_c','memory_used_mib','power_w','utilization_percent']:
  values=[float(r[key]) for r in hw];summary['hardware'][key]={'min':min(values),'median':statistics.median(values),'max':max(values)}
 def csvout(name,rows):
  with (root/name).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 csvout('paired-outcomes.csv',summary['paired'])
 calls=[]
 for arm,campaign in campaigns.items():
  for row in campaign['results']:
   for index,g in enumerate(row['generations']+row['backend_errors'],1):
    calls.append({'variant':arm,'id':row['id'],'split':row['split'],'call':index,'input_tokens':g['input_tokens'],
     'verification_extra_input_tokens':audited[arm][row['id']]['verification_extra_input_tokens'][index-1],
     'generated_tokens':g.get('generated_tokens'),'generation_seconds':g.get('seconds'),'progress_reminder':g['progress']['reminder'],
     'verification_reminder':g['verification']['reminder'],'check_status_before_request':g['verification']['check_status'],
     'changed_files_before_request':g['progress']['changed_editable_files'],'backend_error':g.get('error','')})
 csvout('call-metrics.csv',calls)
 write_json(root/'summary.json',summary)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 matplotlib.rcParams.update({'svg.hashsalt':'one-gpu-lm-018','font.family':'DejaVu Sans'})
 colors=['#64748b','#b45309'];gate='met' if summary['candidate_for_replication'] else 'failed'
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');fig.suptitle('018 final-source test success; frozen gate '+gate)
 for ax,(split,label) in zip(axes,SPLITS.items()):
  count=p['evaluation_task_counts'][split];values=[summary['totals'][arm]['passed'][label] for arm in ARMS]
  ax.bar(ARMS,values,color=colors);ax.set(title=f'{label} ({count} task pairs)',ylabel='Passed tasks',ylim=(0,count));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  for x,value in enumerate(values):ax.text(x,value+.15,str(value),ha='center')
 for extension in ['png','svg']:fig.savefig(root/('comparison.'+extension),dpi=160,metadata={'Date':None} if extension=='svg' else {})
 plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');fig.suptitle('Source edits and observed test checks are not final repair success')
 curves=[]
 for column,(split,label) in enumerate(SPLITS.items()):
  selected=[r for r in summary['paired'] if r['split']==split]
  for row,metric,title in [(0,'first_edit_call','First observed edit'),(1,'first_observed_check_pass_call','First observed standalone test exit zero after edit')]:
   ax=axes[row,column]
   for arm,color in zip(ARMS,colors):
    values=[sum(t[arm+'_'+metric] is not None and t[arm+'_'+metric]<=call for t in selected)/len(selected) for call in range(1,13)]
    ax.step(range(1,13),values,where='post',color=color,label=arm)
    curves += [{'split':split,'variant':arm,'metric':metric,'call':call,'fraction':value,'tasks':len(selected)} for call,value in enumerate(values,1)]
   ax.set(title=label+': '+title,xlabel='Completed model calls',ylabel='Fraction of all tasks',ylim=(0,1));ax.grid(alpha=.2);ax.legend()
 for extension in ['png','svg']:fig.savefig(root/('verification-curves.'+extension),dpi=160,metadata={'Date':None} if extension=='svg' else {})
 plt.close(fig);csvout('verification-curves.csv',curves)
 write_json(root/'report-environment.json',{'python':platform.python_version(),'matplotlib':matplotlib.__version__,'report_sha256':sha256(Path(__file__))})
 totals=summary['totals'];paired_by_id={r['id']:r for r in summary['paired']}
 table='\n'.join(f"| {arm} | {v['passed']['historical']}/24 | {v['passed']['fresh']}/12 | {v['failed_after_any_edit']} | {v['verification_reminder_requests']} | {v['controller_seconds']/60:.1f} |" for arm,v in totals.items())
 targets='\n'.join(f"| `{name}` | {'pass' if v['control'] else 'fail'} | {'pass' if v['verify'] else 'fail'} | {paired_by_id[name]['verify_first_edit_call'] or 'none'} | {paired_by_id[name]['verify_final_observed_check_status']} |" for name,v in summary['targets'].items())
 contrasts='\n'.join(f"| {label} | {len(v['improved'])} | {len(v['regressed'])} | {len(v['improved'])-len(v['regressed'])} |" for label,v in summary['contrasts'].items())
 repositories='\n'.join(f"| {name} | {v['split']} | {v['tasks']} | {v['passed']['control']} | {v['passed']['verify']} |" for name,v in summary['per_repository'].items())
 gates='\n'.join(f"| {name.replace('_',' ')} | {'met' if value else 'failed'} |" for name,value in summary['gate_checks'].items())
 failures='\n'.join(f"| {arm} | {v['context_limit_requests']} | {v['format_invalid_calls']} | {v['generation_limit_hits']} | {v['calls']} | {v['generated_tokens']} | {v['processed_input_tokens']} |" for arm,v in totals.items())
 hardware=summary['hardware']
 (root/'README.md').write_text(f'''# 018 · Repair verification results

**Frozen numerical gate {gate}. No default policy change or deployment approval.** All 72 runs are retained: 24 historical pairs and 12 fresh pairs. Control is the explicit 017 progress policy; verify adds only the locked post-edit verification reminder. The unchanged 27B BF16 base received no weight updates.

| Policy | Historical passes | Fresh passes | Failed after an observed edit (all 36) | Verification reminders | Controller minutes |
| --- | --- | --- | --- | --- | --- |
{table}

Passing means the independent grader accepts the final source, not necessarily that the agent exits normally. Call/context limits and format errors are retained separately. The edited-but-failed count is descriptive: an intervention can alter which tasks get edited, so this conditional behavior is not itself a causal success metric.

![Final-source success](comparison.png)

| Split | Improved pairs | Regressed pairs | Net gain |
| --- | --- | --- | --- |
{contrasts}

| Original 017 regression | control | verify | Verify first edit call | Final observed check status |
| --- | --- | --- | --- | --- |
{targets}

| Frozen requirement | Result |
| --- | --- |
{gates}

| Repository | Exposure split | Tasks | control | verify |
| --- | --- | --- | --- | --- |
{repositories}

![Observed edit and verification curves](verification-curves.png)

Both curves retain every task in the denominator. An edit can later be reverted. A standalone test exit zero may not cover every requirement and can precede a later edit. These curves do not measure correctness at every call; only the final independent grade measures repair success. Compound edit/test commands, pipes and alternative spellings deliberately do not establish recognized verification. The final tool call is included even if no next model request occurs. Public [paired outcomes](paired-outcomes.csv), [call metrics](call-metrics.csv) and [curve data](verification-curves.csv) support the plots.

## Fixed-budget comparison

Both arms use 12 calls, 512 output tokens per call, 6,144 total output tokens, 8,192 input tokens, 2,400 tool-output characters and 900 seconds/task. Same parser, native tool round trips, full history, source observation, tasks and test dependencies. These are equal upper limits, not equal realized token usage. Actual input/output usage is reported; verification notices add {totals['verify']['verification_extra_input_tokens']} input tokens across requests relative to the same histories without those notices (including rejected context-limit requests). Both keep the 017 no-edit reminder. Only verify receives temporary post-edit reminders computed from actual tool exit codes and editable-file hashes. Reminder tokens count toward the input limit. No extra test execution, automatic patch, oracle feedback, teacher model or history truncation. [Protocol](protocol.json), [locked rule](intervention.json), [two-task diagnosis](diagnosis.json), [public preregistration](preregistration.json).

Task-controller time ratio verify/control: **{summary['task_controller_time_ratio']:.3f}×**. This includes tool execution and grading and is a descriptive interleaved shared-host measurement, not pure GPU generation speed. Compared with 017 progress, the repeated historical control had {len(summary['control_vs017_progress']['improved'])} improvements and {len(summary['control_vs017_progress']['regressed'])} regressions; all are kept, with no favorable run selection. Across arms, {sum(r['identical_input_calls'] for r in summary['same_input_output_comparison'])} matched requests had identical actual input-token hashes; {sum(len(r['different_output_calls']) for r in summary['same_input_output_comparison'])} produced different output text hashes. One greedy pair per task cannot separate all execution variation.

| Policy | Context-limit requests | Invalid formats | Output-limit hits | Calls | Output tokens | Processed input tokens |
| --- | --- | --- | --- | --- | --- | --- |
{failures}

## Input validity and screening

All 36 tasks passed original/broken/restored admission and all six loaded clean controls passed. A stricter pre-freeze review caught two draft Pydash stubs that accidentally damaged overload declarations, creating syntax errors; the constructor was fixed to select concrete implementations and reject invalid syntax. The [first admission](admission-before-overload-fix.json) is preserved. Final broken controls all return actual test failure (exit 1), not collection errors. The 24 historical defective byte hashes match 017 exactly. No model score was used for this recovery or task selection.

Ordered-set was rejected before scoring because its reference implementation is installed in setuptools. More-itertools was excluded because it was previously used in this project. Hyperlink and Pydash passed [before](reference-screen.json) and [after](reference-screen-after.json) namespace screens for known Python package/module/metadata names, exact selected source hashes and package paths in archives. Broad substring matches to office hyperlink icons and project logos are retained separately, not classified as Python reference copies; the [initial broad screen](reference-screen-broad.json) and [preparation plan](preparation-plan.json) retain the decision. Each scan records inaccessible paths by hash and error category. The post-run check reuses the pinned clean scan workspace with the same readonly dependency mounts; it does not inventory every model-created task-private file or instrument reads. This bounded check does not prove absence of renamed/embedded/semantic copies or pretraining exposure.

The [audit](failure-audit.json) independently replays every actual input-token hash, both reminder triggers, native tool pairing, parsed action identity, declared-check evidence and source endpoints without executing commands again. Intermediate source hashes are frozen executor observations; original defective and final graded source are checked independently. Test exit status remains evidence from the agent's working copy, not authority over the separate final grader.

## Hardware and scope

UTC {controller['started_utc']} to {controller['finished_utc']}; {summary['time']['wall_seconds']/60:.2f} minutes from controller start through scoring completion, within the original three-hour ceiling. Cleanup and CPU audit follow separately. {hardware['samples']} hardware samples, {hardware['collection_errors']} sampling errors, maximum gap {hardware['max_gap_seconds']:.2f}s. Sampled peaks: {hardware['temperature_c']['max']:.0f}C, {hardware['memory_used_mib']['max']:.0f}MiB, {hardware['power_w']['max']:.1f}W. Sampled GPU energy {hardware['sampled_gpu_energy_kwh']:.3f}kWh excludes CPU/system consumption and may miss peaks.

Fresh means repositories unused in this project's previous evaluations, not unseen pretraining or guaranteed independent authorship. Hyperlink acknowledges ancestry in adjacent helper code from Boltons, which appeared earlier in the project; selected targets and the mounted namespace are documented in the [source ledger](sources.json). Twelve related Python tasks in two repositories, visible tests and constructed missing-function defects are limited evidence for general coding ability. No model training, default change or advanced-coding claim follows automatically. [Reusable workflow](../../README.md). Original harness is MIT; upstream rights remain separate. Public artifacts omit task mutations, raw commands/trajectories, source snapshots, weights and connection details. Exact replay requires authorized private inputs.
''')
 return summary


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
 render(parser.parse_args().run)

