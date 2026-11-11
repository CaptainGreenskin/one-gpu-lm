"""Rebuild the separate screened replication; never substitute it for the primary gate."""
import argparse,csv,importlib.util,json,platform,statistics,sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from one_gpu_lm.common.receipts import sha256,write_json
spec=importlib.util.spec_from_file_location('primary_progress_report',Path(__file__).parents[1]/'report.py');primary=importlib.util.module_from_spec(spec);spec.loader.exec_module(primary)
ARMS=primary.ARMS

def render(root):
 read=lambda n:json.loads((root/n).read_text());exp=Path(__file__).parent
 p=read('protocol.json');c=read('controller.json');pub=read('preregistration.json');launch=read('launch.json');inventory=read('task-inventory.json');audit=read('failure-audit.json')
 base=json.loads((exp.parent/'protocol.json').read_text())
 if any(p[k]!=base[k] for k in ['model_id','revision','agent','variants']):raise ValueError('Supplement changed primary model/rule/budgets')
 if not pub['anonymous'] or pub['ci']['conclusion']!='success' or pub['commit']!=launch['source_commit'] or pub['protocol_sha256']!=sha256(root/'protocol.json') or datetime.fromisoformat(pub['verified_utc'])>=datetime.fromisoformat(c['started_utc']):raise ValueError('Missing public freeze before launch')
 if datetime.fromisoformat(launch['primary_finished_utc'])>=datetime.fromisoformat(c['started_utc']):raise ValueError('Supplement overlapped primary GPU run')
 if not c['complete'] or not c['telemetry']['thread_stopped'] or c.get('error_type') or c['completed_runs']!=12 or c['finished_epoch']>c['deadline_epoch'] or c['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('Incomplete or over-budget supplement')
 if sha256(root/'task-inventory.json')!=sha256(exp/'preflight/task-inventory.json'):raise ValueError('Changed inventory')
 for local,frozen in [('admission.json','runs/admission/admission.json'),('profile.json','experiments/017-edit-progress/model-profile.json'),('sources.json','experiments/017-edit-progress/sources.json'),('source-snapshots.json','source-snapshots.json'),('reference-screen.json','reference-screen.json'),('preparation-review.json','preparation-review.json')]:
  if sha256(root/local)!=p['frozen_files'][frozen]:raise ValueError('Frozen file mismatch: '+local)
 for name in ['reference-screen.json','reference-screen-after.json']:
  screen=read(name)
  if not screen['screen_passed'] or screen['blocking_scan_errors'] or screen['other_package_name_matches'] or screen['duplicate_selected_source_files'] or screen['archives_containing_package_name'] or screen['scanner_sha256']!=sha256(exp/'reference_screen.py'):raise ValueError('Reference screen did not pass')
 artifact=read('artifact.json');controls=read('loaded-controls.json')
 if any(artifact[k]!=p[k] for k in ['model_id','revision']) or artifact['adapter'] or artifact['adapter_sha256'] or artifact['profile_sha256']!=sha256(root/'profile.json'):raise ValueError('Model changed')
 if any(read('loading.json').get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('Model loading mismatch')
 if not controls['complete'] or controls['task_spec_sha256']!=p['task_spec_sha256'] or len(controls['clean'])!=1 or not controls['clean'][0]['passed']:raise ValueError('Loaded clean control failed')
 expected=[{'id':t['id'],'variant':a} for i,t in enumerate(inventory) for a in list(ARMS)[i%2:]+list(ARMS)[:i%2]]
 if c['schedule']!=expected or [{'id':t['id'],'variant':t['variant']} for t in c['stages'] if t['name']=='task']!=expected:raise ValueError('Task order changed or task repeated')
 campaigns={a:read(a+'.json') for a in ARMS}
 for arm,campaign in campaigns.items():
  identity=campaign['identity'];rows=audit['variants'][arm]
  if identity['tasks_sha256']!=p['task_spec_sha256'] or identity['protocol_sha256']!=sha256(root/'protocol.json') or identity['admission_sha256']!=p['admission_sha256'] or any(p['frozen_files'].get(k)!=v for k,v in identity['source_files'].items()):raise ValueError('Execution identity mismatch')
  if audit['campaign_sha256'][arm]!=sha256(root/(arm+'.json')) or [r['id'] for r in rows]!=[t['id'] for t in inventory] or not all(r['progress_and_native_turns_verified'] and r['final_source_verified'] for r in rows) or sum(r['input_token_hashes_verified'] for r in rows)!=sum(r['calls'] for r in campaign['results']):raise ValueError('Incomplete request/state audit')
 s=primary.assess(campaigns,p,inventory);assert not s['candidate_for_replication'] and not s['deployment_allowed']
 s['same_input_output_comparison']=audit['same_input_output_comparison']
 s['study']='Separate reference-screened humanize replication; descriptive only, no replacement of primary gate'
 for arm in ARMS:
  s['totals'][arm]['controller_seconds']=sum(t['seconds'] for t in c['stages'] if t['name']=='task' and t['variant']==arm)
  s['totals'][arm]['consecutive_repeated_commands']=sum(t['consecutive_repeated_commands'] for t in audit['variants'][arm])
 s['time']={k:c[k] for k in ['started_utc','finished_utc']};s['time']['wall_seconds']=c['finished_epoch']-c['started_epoch'];s['task_controller_time_ratio']=s['totals']['progress']['controller_seconds']/s['totals']['control']['controller_seconds']
 hw=list(csv.DictReader((root/'telemetry.csv').open()));ts=[datetime.fromisoformat(r['utc']).timestamp() for r in hw]
 if len(hw)!=c['telemetry']['samples'] or len(hw)<2 or not all(b>a for a,b in zip(ts,ts[1:])):raise ValueError('Missing hardware samples')
 s['hardware']={'samples':len(hw),'collection_errors':c['telemetry']['errors'],'max_gap_seconds':max(b-a for a,b in zip(ts,ts[1:])), 'sampled_gpu_energy_kwh':sum((b-a)*(float(x['power_w'])+float(y['power_w']))/2 for a,b,x,y in zip(ts,ts[1:],hw,hw[1:]))/3600000}
 for key in ['temperature_c','memory_used_mib','power_w','utilization_percent']:
  values=[float(r[key]) for r in hw];s['hardware'][key]={'min':min(values),'median':statistics.median(values),'max':max(values)}
 def csvout(name,rows):
  with (root/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 csvout('paired-outcomes.csv',s['paired']);curves=[]
 for arm in ARMS:
  for n in range(1,13):curves.append({'variant':arm,'call':n,'ever_changed_fraction':sum(r[arm+'_first_edit_call'] is not None and r[arm+'_first_edit_call']<=n for r in s['paired'])/6,'tasks':6})
 csvout('first-edit-curves.csv',curves);write_json(root/'summary.json',s)
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 matplotlib.rcParams.update({'svg.hashsalt':'one-gpu-lm-017-supplement','font.family':'DejaVu Sans'})
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');fig.suptitle('017 separate humanize replication; same rule and budget')
 values=[s['totals'][a]['passed']['fresh'] for a in ARMS];colors=['#64748b','#0d9488'];axes[0].bar(ARMS,values,color=colors);axes[0].set(ylabel='Passed tasks',ylim=(0,6),title='Final task success')
 for x,v in enumerate(values):axes[0].text(x,v+.07,str(v),ha='center')
 for arm,color in zip(ARMS,colors):axes[1].step(range(1,13),[r['ever_changed_fraction'] for r in curves if r['variant']==arm],where='post',label=arm,color=color)
 axes[1].set(xlabel='Completed calls',ylabel='Fraction with observed source change',ylim=(0,1),title='Source change is not success');axes[1].legend()
 for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 for ext in ['png','svg']:fig.savefig(root/('comparison.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
 plt.close(fig);write_json(root/'report-environment.json',{'python':platform.python_version(),'matplotlib':matplotlib.__version__,'report_sha256':sha256(Path(__file__)),'shared_assessment_sha256':sha256(exp.parent/'report.py')})
 table='\n'.join(f"| {a} | {v['passed']['fresh']}/6 | {v['call_limit_without_any_edit']} | {v['failed_without_any_observed_edit']} | {v['controller_seconds']/60:.1f} |" for a,v in s['totals'].items())
 (root/'README.md').write_text(f'''# 017 supplement results · humanize

**Separate descriptive replication. No default policy change, deployment approval or weight update.** All 12 runs completed after the primary run, using its identical model, rule and per-task budgets. These six new task pairs do not replace the original 72-run gate.

| Arm | Passed | 12 calls with no edit | Failed without any edit | Controller minutes |
| --- | --- | --- | --- | --- |
{table}

Passing means the independent grader accepts the final source, not necessarily that the agent exits normally. Context-limit requests: control {s['totals']['control']['context_limit_requests']}, progress {s['totals']['progress']['context_limit_requests']}; these are retained even when the final code passes. Invalid action formats: control {s['totals']['control']['format_invalid_calls']}, progress {s['totals']['progress']['format_invalid_calls']}.

![Success and first edit curves](comparison.png)

Paired outcomes: {s['contrasts']['fresh']}. Task-controller time ratio: {s['task_controller_time_ratio']:.3f}×. These interleaved times include tools and grading, not just generation. Curves retain all six tasks in each denominator and count temporary byte changes; they are secondary observations, never a substitute for passing the independent grader. [Paired data](paired-outcomes.csv), [curve data](first-edit-curves.csv), [full summary](summary.json).

The six constructed defects use the upstream number/filesize tests from pinned humanize 4.12.3. All original/broken/restored checks and the GPU-loaded clean control passed. The two arms each have 12 calls, 512 output tokens per call, 6,144 total output tokens, 8,192 input tokens and 900 seconds per task. The candidate receives only the generic progress reminder; the prompt tokens count toward the same input budget. No model scores were read during source/task selection, no rule was tuned and no scored task was repeated.

[Reference screening before](reference-screen.json) and [after](reference-screen-after.json) found no other package-name matches, duplicate selected source files or package names in zip/wheel archives across the same mounted namespace. Two entries were inaccessible to the sandbox UID, with error categories and path hashes retained. This bounded screen cannot rule out every renamed or embedded implementation, or exposure during base pretraining. One repository, six related tasks, visible tests and one greedy run per arm are limited evidence for general coding ability.

[Audit](failure-audit.json) independently re-encodes all actual requests, checks the input token hashes, reminder triggers, tool-history pairing and final source state. Intermediate source changes remain the frozen executor's observations. Across arms, {sum(len(r['different_output_calls']) for r in s['same_input_output_comparison'])} matched calls had identical input-token hashes but different output text hashes; one pair per task cannot isolate all execution variation. [Protocol](protocol.json) and [public preregistration](preregistration.json) were verified before this run. Raw commands, trajectories, tasks, source snapshots and weights are private.

Controller UTC {c['started_utc']} to {c['finished_utc']}, {s['time']['wall_seconds']/60:.2f} minutes within the separate one-hour ceiling. {s['hardware']['samples']} hardware samples, {s['hardware']['collection_errors']} sampling errors, maximum gap {s['hardware']['max_gap_seconds']:.2f}s. Sampled peak {s['hardware']['temperature_c']['max']:.0f}C, {s['hardware']['memory_used_mib']['max']:.0f}MiB, {s['hardware']['power_w']['max']:.1f}W. Sampled GPU energy {s['hardware']['sampled_gpu_energy_kwh']:.3f}kWh excludes the rest of the system and may miss peaks. The post-run CPU audit and backend cleanup are separate from the scoring interval.
''')
 return s

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);render(p.parse_args().run)
