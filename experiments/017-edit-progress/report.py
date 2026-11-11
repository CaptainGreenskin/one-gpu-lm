"""Analyze the fixed historical-to-fresh study without treating an edit as success."""
import argparse,csv,json,platform,statistics,sys
from collections import Counter
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json
ARMS=('control','progress')
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
    if pr['completed_calls']!=index or pr['remaining_calls']!=p['agent']['max_model_calls']-index or pr['enabled']!=(arm=='progress'):raise ValueError('Progress policy/call accounting mismatch')
    expected=arm=='progress' and pr['changed_editable_files']==0 and (index>=4 or pr['identical_command_streak']>=2)
    if pr['reminder']!=expected or bool(pr['reminder_chars'])!=expected or len(g['input_tokens_sha256'])!=64:raise ValueError('Wrong reminder or missing actual input hash')
    if g['context']['policy']!='full' or g['context']['dropped_rounds'] or g['context']['raw_input_tokens']!=g['input_tokens'] or g['context']['packed_input_tokens']!=g['input_tokens']:raise ValueError('Hidden context change')
    if g['dialogue']['structured_tool_calls']!=g['dialogue']['tool_results']:raise ValueError('Native history mismatch')
   for g in gs:
    if g['adapter'] or g['agent_policy']!=policy or g['input_tokens']>p['agent']['max_input_tokens'] or g['generated_tokens']>p['agent']['max_generated_tokens_per_call'] or type(g['action_format_valid']) is not bool:raise ValueError('Serving or budget mismatch')
  gs=[g for r in rs for g in r['generations']];requests=[g for r in rs for g in r['generations']+r['backend_errors']]
  totals[arm]={'passed':{name:sum(r['passed'] for r in rs if r['split']==split) for split,name in SPLITS.items()},'tasks':{name:sum(r['split']==split for r in rs) for split,name in SPLITS.items()},'calls':sum(r['calls'] for r in rs),'generation_calls':len(gs),'generated_tokens':sum(r['generated_tokens'] for r in rs),'generation_seconds':sum(g['seconds'] for g in gs),'task_seconds':sum(r['seconds'] for r in rs),'tool_seconds':sum(r['tool_seconds'] for r in rs),'format_invalid_calls':sum(not g['action_format_valid'] for g in gs),'generation_limit_hits':sum(g['hit_generation_limit'] for g in gs),'backend_errors':dict(Counter(g.get('error') for r in rs for g in r['backend_errors'])),'context_limit_requests':sum(g.get('error')=='context_limit' for r in rs for g in r['backend_errors']),'reminder_requests':sum(g['progress']['reminder'] for g in requests),'failed_without_final_source_change':sum(not r['passed'] and not r['modified_source_files'] for r in rs),'failed_without_any_observed_edit':sum(not r['passed'] and not any(s['changed_editable_files'] for s in r['source_progress']) for r in rs),'exit_status':dict(Counter(r['exit_status'] for r in rs))}
  totals[arm]['call_limit_without_any_edit']=sum(r['calls']==p['agent']['max_model_calls'] and not any(s['changed_editable_files'] for s in r['source_progress']) for r in rs)
  totals[arm]['by_split']={label:{'calls':sum(r['calls'] for r in rs if r['split']==split),'failed_without_any_observed_edit':sum(not r['passed'] and not any(s['changed_editable_files'] for s in r['source_progress']) for r in rs if r['split']==split),'call_limit_without_any_edit':sum(r['calls']==p['agent']['max_model_calls'] and not any(s['changed_editable_files'] for s in r['source_progress']) for r in rs if r['split']==split)} for split,label in SPLITS.items()}
 for t in inventory:
  a,b=(rows[arm][t['id']] for arm in ARMS);outcome='improved' if b['passed'] and not a['passed'] else 'regressed' if a['passed'] and not b['passed'] else 'both_passed' if a['passed'] else 'both_failed'
  item={k:t[k] for k in ['id','repository','split','family']};item['outcome']=outcome
  for arm,r in [('control',a),('progress',b)]:
   item.update({arm+'_'+k:r[k] for k in ['passed','calls','seconds','generated_tokens','exit_status']});item[arm+'_first_edit_call']=next((s['call_index'] for s in r['source_progress'] if s['changed_editable_files']),None);item[arm+'_final_changed_files']=len(r['modified_source_files'])
  paired.append(item)
 contrasts={name:{kind:[r['id'] for r in paired if r['split']==split and r['outcome']==kind] for kind in ['improved','regressed']} for split,name in SPLITS.items()}
 gate=p['promotion'];checks={'fresh_net_gain':len(contrasts['fresh']['improved'])-len(contrasts['fresh']['regressed'])>=gate['minimum_net_fresh_gain'],'no_fresh_regressions':len(contrasts['fresh']['regressed'])<=gate['maximum_regressed_fresh_tasks'],'no_historical_regressions':len(contrasts['historical']['regressed'])<=gate['maximum_regressed_historical_tasks'],'all_three_targets_pass':all(rows['progress'][i]['passed'] for i in gate['target_tasks'])} if 'target_tasks' in gate else {}
 return {'schema_version':1,'totals':totals,'contrasts':contrasts,'paired':paired,'targets':{i:{a:rows[a][i]['passed'] for a in ARMS} for i in gate.get('target_tasks',[])},'gate_checks':checks,'candidate_for_replication':bool(checks) and all(checks.values()),'deployment_allowed':False,'weight_updates':0,'per_repository':{repo:{'split':next(t['split'] for t in inventory if t['repository']==repo),'tasks':sum(t['repository']==repo for t in inventory),'passed':{a:sum(r['passed'] for r in rows[a].values() if r['repository']==repo) for a in ARMS}} for repo in sorted({t['repository'] for t in inventory})}}

def render(root):
 read=lambda name:json.loads((root/name).read_text());exp=Path(__file__).parent;p=read('protocol.json');c=read('controller.json');inventory=read('task-inventory.json');pub=read('preregistration.json');launch=read('launch.json')
 if not pub['anonymous'] or pub['ci']['conclusion']!='success' or pub['commit']!=launch['source_commit'] or pub['protocol_sha256']!=sha256(root/'protocol.json') or datetime.fromisoformat(pub['verified_utc'])>=datetime.fromisoformat(c['started_utc']):raise ValueError('Missing verified public preregistration before launch')
 if not c['complete'] or not c['telemetry']['thread_stopped'] or c.get('error_type') or c['completed_runs']!=p['execution']['expected_model_task_runs'] or c['protocol_sha256']!=sha256(root/'protocol.json') or c['finished_epoch']>c['deadline_epoch']:raise ValueError('Incomplete run or original deadline exceeded')
 if sha256(root/'task-inventory.json')!=sha256(exp/'preflight/task-inventory.json'):raise ValueError('Inventory changed')
 for local,frozen in [('admission.json','runs/admission/admission.json'),('profile.json','experiments/017-edit-progress/model-profile.json'),('sources.json','experiments/017-edit-progress/sources.json'),('intervention.json','experiments/017-edit-progress/intervention.json'),('diagnosis.json','experiments/017-edit-progress/diagnosis.json'),('source-snapshots.json','source-snapshots.json'),('model-verification.json','model-verification.json'),('template-progress-probe.json','template-progress-probe.json')]:
  if sha256(root/local)!=p['frozen_files'][frozen]:raise ValueError('Frozen input changed: '+local)
 artifact=read('artifact.json');controls=read('loaded-controls.json')
 if any(artifact[k]!=p[k] for k in ['model_id','revision']) or artifact['adapter'] or artifact['adapter_sha256'] or artifact['profile_sha256']!=sha256(root/'profile.json'):raise ValueError('Model identity changed')
 if any(read('loading.json').get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('Model loading mismatch')
 if not controls['complete'] or controls['task_spec_sha256']!=p['task_spec_sha256'] or len(controls['clean'])!=p['repository_count'] or not all(r['passed'] for r in controls['clean']):raise ValueError('GPU-loaded clean controls failed')
 expected=[{'id':t['id'],'variant':arm} for index,t in enumerate(inventory) for arm in list(ARMS)[index%2:]+list(ARMS)[:index%2]]
 if c['schedule']!=expected or [{'id':s['id'],'variant':s['variant']} for s in c['stages'] if s['name']=='task']!=expected:raise ValueError('Order changed or scored task repeated')
 campaigns={arm:read(arm+'.json') for arm in ARMS};audit=read('failure-audit.json')
 for arm,campaign in campaigns.items():
  i=campaign['identity']
  if i['tasks_sha256']!=p['task_spec_sha256'] or i['protocol_sha256']!=sha256(root/'protocol.json') or i['admission_sha256']!=p['admission_sha256'] or any(p['frozen_files'].get(k)!=v for k,v in i['source_files'].items()):raise ValueError('Campaign execution identity changed')
  a=audit['variants'][arm]
  if audit['campaign_sha256'][arm]!=sha256(root/(arm+'.json')) or audit['parser_sha256']!=i['source_files']['one_gpu_lm/common/actions.py'] or [v['id'] for v in a]!=[t['id'] for t in inventory] or not all(v['progress_and_native_turns_verified'] and v['final_source_verified'] for v in a) or sum(v['input_token_hashes_verified'] for v in a)!=sum(v['calls'] for v in campaign['results']):raise ValueError('Missing independent input/progress audit')
 s=assess(campaigns,p,inventory)
 s['same_input_output_comparison']=audit['same_input_output_comparison']
 dependency=read('dependency-review.json')
 if dependency['review_source_sha256']!=sha256(exp/'dependency_review.py') or dependency['lock_sha256']!=sha256(exp/'preflight/node-package-lock.json') or dependency['scoring_or_inputs_changed'] or not dependency['accessible_to_both_arms']:raise ValueError('Missing dependency exposure review')
 for arm in ARMS:
  if [x['id'] for x in dependency['variants'][arm]]!=[r['id'] for r in campaigns[arm]['results'] if r['repository']==dependency['repository']]:raise ValueError('Incomplete dependency exposure review')
 s['reference_exposure']={'repository':dependency['repository'],'dependency_chain':dependency['dependency_chain'],'clean_reference_free_validation':False,'explicit_dependency_path_commands':{arm:sum(r['explicit_dependency_path_commands'] for r in dependency['variants'][arm]) for arm in ARMS},'bare_package_import_commands':{arm:sum(r['bare_package_import_commands'] for r in dependency['variants'][arm]) for arm in ARMS},'reference_read_access_instrumented':False}
 s['supports_reference_free_fresh_generalization']=False
 s['reference_exposure']['blinker_system_copies']=dependency['blinker_system_copies']
 s['reference_exposure']['blinker_commands_naming_system_path']={arm:sum(x['commands_naming_system_blinker_path'] for x in dependency['blinker_command_indicators'][arm]) for arm in ARMS}
 for arm in ARMS:
  s['totals'][arm]['controller_seconds']=sum(t['seconds'] for t in c['stages'] if t['name']=='task' and t['variant']==arm)
  s['totals'][arm]['consecutive_repeated_commands']=sum(t['consecutive_repeated_commands'] for t in audit['variants'][arm])
 prior_path=exp.parent/'016-tool-roundtrip/results/2026-11-11/roundtrip.json';prior=json.loads(prior_path.read_text());old={r['id']:r for r in prior['results']}
 if sha256(prior_path)!=read('template-progress-probe.json')['prior_campaign_sha256']:raise ValueError('Wrong016reference')
 hist=[r for r in campaigns['control']['results'] if r['split']=='dev'];s['control_vs016']={'improved':[r['id'] for r in hist if r['passed'] and not old[r['id']]['passed']],'regressed':[r['id'] for r in hist if not r['passed'] and old[r['id']]['passed']]}
 s['time']={k:c[k] for k in ['started_utc','finished_utc']};s['time']['wall_seconds']=c['finished_epoch']-c['started_epoch'];s['task_controller_time_ratio']=s['totals']['progress']['controller_seconds']/s['totals']['control']['controller_seconds']
 hw=list(csv.DictReader((root/'telemetry.csv').open()));ts=[datetime.fromisoformat(r['utc']).timestamp() for r in hw]
 if len(hw)!=c['telemetry']['samples'] or len(hw)<2 or not all(b>a for a,b in zip(ts,ts[1:])):raise ValueError('Missing or unordered hardware data')
 s['hardware']={'samples':len(hw),'collection_errors':c['telemetry']['errors'],'span_seconds':ts[-1]-ts[0],'max_gap_seconds':max(b-a for a,b in zip(ts,ts[1:])),'sampled_gpu_energy_kwh':sum((b-a)*(float(x['power_w'])+float(y['power_w']))/2 for a,b,x,y in zip(ts,ts[1:],hw,hw[1:]))/3600000}
 for key in ['temperature_c','memory_used_mib','power_w','utilization_percent']:
  values=[float(r[key]) for r in hw];s['hardware'][key]={'min':min(values),'median':statistics.median(values),'max':max(values)}
 def csvout(name,rows):
  with (root/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 csvout('paired-outcomes.csv',s['paired']);calls=[]
 for arm,campaign in campaigns.items():
  for r in campaign['results']:
   for index,g in enumerate(r['generations']+r['backend_errors'],1):calls.append({'variant':arm,'id':r['id'],'split':r['split'],'call':index,'input_tokens':g['input_tokens'],'generated_tokens':g.get('generated_tokens'),'generation_seconds':g.get('seconds'),'reminder':g['progress']['reminder'],'changed_files_before_request':g['progress']['changed_editable_files'],'backend_error':g.get('error','')})
 csvout('call-metrics.csv',calls);write_json(root/'summary.json',s)
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 matplotlib.rcParams.update({'svg.hashsalt':'one-gpu-lm-017','font.family':'DejaVu Sans'});colors=['#64748b','#0d9488'];gate='met' if s['candidate_for_replication'] else 'failed'
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');fig.suptitle('017 final task success; frozen gate '+gate)
 for ax,name,limit in zip(axes,['historical','fresh'],[p['evaluation_task_counts']['dev'],p['evaluation_task_counts']['test']]):
  vals=[s['totals'][a]['passed'][name] for a in ARMS];ax.bar(ARMS,vals,color=colors);ax.set(title=name,ylabel='Passed tasks',ylim=(0,limit));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  for x,v in enumerate(vals):ax.text(x,v+.1,str(v),ha='center')
 for ext in ['png','svg']:fig.savefig(root/('comparison.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
 plt.close(fig);fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');fig.suptitle('Observed first source change, not task success')
 progress_curves=[]
 for ax,(split,label) in zip(axes,SPLITS.items()):
  selected=[r for r in s['paired'] if r['split']==split]
  for arm,color in zip(ARMS,colors):
   ys=[sum(r[arm+'_first_edit_call'] is not None and r[arm+'_first_edit_call']<=n for r in selected)/len(selected) for n in range(1,13)]
   ax.step(range(1,13),ys,where='post',label=arm,color=color);progress_curves += [{'split':split,'variant':arm,'call':i,'ever_changed_fraction':y,'tasks':len(selected)} for i,y in enumerate(ys,1)]
  ax.set(title=label,xlabel='Completed model calls',ylabel='Fraction with observed source change',ylim=(0,1));ax.grid(alpha=.2);ax.legend()
 for ext in ['png','svg']:fig.savefig(root/('first-edit-curves.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
 plt.close(fig);csvout('first-edit-curves.csv',progress_curves)
 write_json(root/'report-environment.json',{'python':platform.python_version(),'matplotlib':matplotlib.__version__,'report_sha256':sha256(Path(__file__))})
 table='\n'.join(f"| {a} | {v['passed']['historical']}/24 | {v['passed']['fresh']}/12 | {v['failed_without_any_observed_edit']} | {v['reminder_requests']} | {v['controller_seconds']/60:.1f} |" for a,v in s['totals'].items())
 repos='\n'.join(f"| {k} | {v['split']} | {v['tasks']} | {v['passed']['control']} | {v['passed']['progress']} |" for k,v in s['per_repository'].items())
 (root/'README.md').write_text(f'''# 017 · Edit progress results

**Frozen numerical gate {gate}. No default policy change or deployment approval follows.** The same unchanged 27B BF16 base was used, with no weight updates. All 72 runs are retained: 24 historical and 12 fresh task pairs, with no tuning between stages. Both original fresh repositories have reference exposure, limiting the validation claim regardless of the gate result. See the [separate screened supplement](../../supplement/) for additional validation.

| Policy | Historical passes | Fresh passes | Failed without any observed edit | Reminder requests | Controller minutes |
| --- | --- | --- | --- | --- | --- |
{table}

![Final success](comparison.png)

Historical paired changes: {s['contrasts']['historical']}. Fresh paired changes: {s['contrasts']['fresh']}. Three prior regressions: {s['targets']}. Gate checks: {s['gate_checks']}. Overall task-controller time ratio {s['task_controller_time_ratio']:.3f}×; this interleaved total includes tools and grading, not just generation. Tasks reaching the 12-call limit without any observed edit: control {s['totals']['control']['call_limit_without_any_edit']}, progress {s['totals']['progress']['call_limit_without_any_edit']}.

| Repository | Exposure split | Tasks | control | progress |
| --- | --- | --- | --- | --- |
{repos}

![First source changes](first-edit-curves.png)

These curves show the fraction ever observed with a valid editable-file byte difference after a tool call, with all tasks kept in each denominator. Temporary edits later undone count here; deleted/unsafe files do not. Neither an edit nor earlier editing earns task success. Only the separate grader's tests do. See [paired outcomes](paired-outcomes.csv), [request metrics](call-metrics.csv) and [curve data](first-edit-curves.csv).

## What was controlled

Same task inputs, model, strict parser, tool round trips, full history, 12 calls, 512 output tokens/call, 6,144 total output tokens, 8,192 input tokens and 900 seconds/task. Only the progress arm receives a generic temporary system reminder using current editable state and completed-call history. Reminder tokens count toward the input limit; there is no extra call, truncation, automatic patch or answer hint. Source-state observation runs in both arms. The reminder rule was locked before inspecting the new source repositories. [Frozen protocol](protocol.json), [intervention](intervention.json), [initial diagnosis](diagnosis.json), [pre-rollout public verification](preregistration.json).

The unchanged control versus 016 differed as follows: {s['control_vs016']}. This repeat-variation check is descriptive; no better run was selected. Across arms there were {sum(len(r['different_output_calls']) for r in s['same_input_output_comparison'])} different outputs at matched call indices with identical actual input-token hashes. Greedy decoding is not a guarantee of identical outcomes; one pair per task cannot isolate all execution variation. [Audit](failure-audit.json) re-encodes every actual request with the pinned tokenizer, checks its token hash, reminder trigger, native tool pairing, strict parser decisions and final source state without rerunning commands or changing scores. Intermediate file states are recorded by the frozen executor; the final workspace and initial state are checked independently.

## Hardware and failures

UTC {c['started_utc']} to {c['finished_utc']}; {s['time']['wall_seconds']/60:.2f} minutes from controller start through scoring completion, within the original 3-hour ceiling. Backend cleanup and the subsequent CPU audit are separate. {s['hardware']['samples']} hardware samples, {s['hardware']['collection_errors']} collection errors, maximum gap {s['hardware']['max_gap_seconds']:.2f}s. Sampled peak {s['hardware']['temperature_c']['max']:.0f}C, {s['hardware']['memory_used_mib']['max']:.0f}MiB, {s['hardware']['power_w']['max']:.1f}W. Sampled GPU energy {s['hardware']['sampled_gpu_energy_kwh']:.3f}kWh uses trapezoid interpolation, excludes CPU/system power and can miss peaks. [Summary](summary.json) retains context/format failures and every outcome; raw hashes are in [raw-artifacts.json](raw-artifacts.json).

Before scoring, an added async-test plugin caused a historical clean test startup failure. [Preparation recovery](preparation-recovery.json) records the fix: preserve exact old test-package bytes as default and mount extra dependencies only for Blinker. No task, test, reminder or budget changed. All 36 original/broken/restored admissions and 6 GPU-loaded clean controls passed before task scoring.

## Reference exposure in p-map dependencies

After launch, dependency review found that the AVA test framework includes its own published p-map 7.1.1 package, mounted read-only but readable by both agents. The installed dependency's `index.js` is byte-identical to the unmutated target. This was found after launch but before fresh scoring; the [discovery receipt](../../dependency-exposure-discovery.json) records that boundary. [Dependency review](dependency-review.json) records both source hashes, equality, unmutated repair-fragment overlap and sanitized access indicators. There are {s['reference_exposure']['explicit_dependency_path_commands']} commands explicitly naming the dependency path and {s['reference_exposure']['bare_package_import_commands']} bare-package import commands. These are text indicators, not file-access instrumentation: zero indicators cannot prove that no reference was accessed. AVA's internal runtime use of its dependency is not itself evidence that reference source was shown to the model.

The mounted system directories also contain an older Blinker implementation, discovered before fresh scoring. See its [discovery receipt](../../system-reference-discovery.json) and the system-copy hashes in the dependency review. Commands explicitly naming a system Blinker path: {s['reference_exposure']['blinker_commands_naming_system_path']}. The same limitations of text indicators apply.

No running input, outcome, denominator or frozen gate was changed after these discoveries. All 12 original fresh pairs remain in the fixed aggregate, but neither repository establishes reference-free repair ability. A separately preregistered humanize supplement preserves the same rule, model and per-task budgets after reference screening; its results are not pooled into or substituted for this frozen gate. Preparation of that CPU-only supplement overlapped part of this run, so task time comparisons should also be treated as descriptive shared-host measurements.

## Limits and reuse

Fresh means two repositories unused in this project's earlier evaluation, not unseen by base pretraining or independent upstream authors. These are constructed defects with visible tests, small repository clusters and one greedy run per arm/task. No training, new default, model release or general coding certification follows automatically. [Reusable workflow](../../README.md) supports authorized private tasks; exact score reconstruction requires the private source/task inputs. Public artifacts omit commands, traces, source snapshots, weights and connection details. Original harness is MIT; upstream rights remain separate.
''')
 return s

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);render(p.parse_args().run)
