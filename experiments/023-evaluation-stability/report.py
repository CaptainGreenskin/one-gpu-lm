"""Recompute descriptive023 paired stability from independently audited numbers."""
import argparse,csv,importlib.util,json,math
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parent;REPO=EXP.parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(EXP))
from one_gpu_lm.common.receipts import sha256,write_json
from analysis import summarize,MODELS,CONDITIONS

def require(ok,message):
 if not ok:raise ValueError(message)

def assess(root):
 root=Path(root);read=lambda n:json.loads((root/n).read_text())
 collection=read('collection.json');require(all(sha256(root/n)==h for n,h in collection['files'].items()),'Collected evidence changed')
 protocol=read('protocol.json');plan=read('plan.json');controller=read('controller.json');audit=read('execution-audit.json');post=read('postrun-status.json')
 require(controller['complete'] and controller['stage']=='completed' and not controller.get('error_type') and controller['telemetry']['thread_stopped'],'Incomplete pipeline')
 require(controller['protocol_sha256']==sha256(root/'protocol.json')==audit['protocol_sha256']==sha256(EXP/'protocol.json'),'Protocol identity mismatch')
 require(not controller['final_test_opened'] and controller['training_updates']==0 and not controller['weights_public'],'Unexpected training,test or publication')
 for name in ['plan.json','profile.json','sources.json']:
  require(sha256(root/name)==protocol['frozen_files']['experiments/023-evaluation-stability/'+name],'Collected input differs from preregistration: '+name)
 for n,h in protocol['frozen_files'].items():
  if (REPO/n).is_file():require(sha256(REPO/n)==h,'Frozen public implementation changed: '+n)
 require(audit['audit_source_sha256']==sha256(EXP/'audit.py') and audit['normalizer_sha256']==sha256(EXP/'normalize.py'),'Wrong output audit implementation')
 require(not audit['final_test_opened'] and audit['training_updates']==audit['commands_reexecuted']==audit['model_generations_performed']==0 and not audit['raw_traces_public'],'Audit must be read-only and development-only')
 require(post['complete'] and post['frozen_files_verified']==len(protocol['frozen_files']) and post['final_test_artifacts_absent'] and post['final_task_specs_absent'],'Post-run integrity incomplete')
 require(post['execution_audit_sha256']==sha256(root/'execution-audit.json') and post['content_integrity_sha256']==sha256(root/'content-integrity.json'),'Post-run receipt identity differs')
 for name in ['preparation-verified.json','content-integrity.json']:
  v=read(name);require(v['passed'] and v['development_tasks']==15 and v['repository_count']==4 and not v['final_tasks_present'] and v['model_generations_performed']==0,'Development-only content proof missing')
  require(v['audit_source_sha256']==sha256(EXP/'preparation.py') and v['model_verification']['content_verified'],'Wrong preparation source/model')
  require(v['task_spec_sha256']==protocol['task_spec_sha256'] and v['admission_sha256']==protocol['admission_sha256'],'Task/admission mismatch')
  for m in ['reference020','action022']:require(v['adapters'][m]['adapter_model.safetensors']==plan['models'][m]['adapter_model_sha256'],'Wrong fixed adapter')
 pub=read('preregistration.json');launch=read('launch.json')
 require(pub['anonymous_files_verified'] and pub['ci_conclusion']=='success' and pub['commit']==launch['source_commit'] and pub['protocol_sha256']==sha256(root/'protocol.json'),'Preregistration not verified')
 require(datetime.fromisoformat(pub['verified_utc'])<datetime.fromisoformat(controller['started_utc']),'Preregister before all023 generations')
 require(not launch['final_test_opened'] and launch['training_updates']==0 and launch['planned_task_runs']==270 and launch['planned_probe_calls']==36,'Changed run scope')
 names=[f'r{b["repeat"]:02d}-{b["model"]}' for b in protocol['blocks']]
 ids=list(dict.fromkeys(t['id'] for t in protocol['blocks'][0]['tasks']))
 require(len(ids)==15 and len(names)==9,'Invalid frozen task/block inventory')
 require(controller['completed_blocks']==names,'Incomplete or reordered blocks')
 campaigns={};runtime=[]
 for b,name in zip(protocol['blocks'],names):
  m=b['model'];artifacts=sorted(root.glob('artifacts/'+name+'-attempt-*.json'));require(bool(artifacts),'Missing served artifact')
  for artifact in artifacts:
   health=json.loads(artifact.read_text());expected={} if m=='base' else {'adapter_model.safetensors':plan['models'][m]['adapter_model_sha256']}
   require(health['adapter_sha256']==expected and health['profile_sha256']==sha256(root/'profile.json'),'Serving artifact/profile mismatch')
   require(health['ready'] and health['adapter']==(m!='base'),'Serving readiness/adapter mismatch')
   require(all(health[k]==protocol[k] for k in ['model_id','revision']),'Wrong served base')
   stem=artifact.stem;loading=read('loading/'+stem+'.json');require(not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),'Loading mismatch')
   controls=read('loaded-controls/'+stem+'.json');require(controls['complete'] and len(controls['clean'])==4 and all(v['passed'] for v in controls['clean']),'Loaded controls failed')
   settings=read('runtime/'+stem+'.json')
   require(settings['settings_changed_by_023'] is False and settings['do_sample'] is False and settings['seed']==101 and settings['attention']==read('profile.json')['attention'],'Generation settings changed')
   runtime.append({'block':name,'attempt':stem.rsplit('-',1)[1],**settings})
  for c in CONDITIONS:
   label=name+'-'+c;p=root/'campaigns'/(label+'.json');campaign=json.loads(p.read_text());campaigns[label]=campaign
   identity=campaign['identity'];check=audit['campaigns'][label]
   require(check['sha256']==sha256(p) and identity['protocol_sha256']==sha256(root/'protocol.json') and identity['tasks_sha256']==protocol['task_spec_sha256'],'Campaign not bound to audit/protocol')
   require(identity['model']==m and identity['repeat']==b['repeat'] and identity['variant']==c,'Changed cell identity')
   require(check['model']==m and check['repeat']==b['repeat'] and check['condition']==c,'Changed audited cell identity')
   require(identity['ids']==ids and identity['admission_sha256']==protocol['admission_sha256'],'Changed task inventory/admission')
   require([r['id'] for r in campaign['results']]==ids==[r['id'] for r in check['tasks']],'Missing,duplicate or reordered row/audit')
   require(identity['wrapper_sha256']==sha256(EXP/'agent.py') and identity['normalizer_sha256']==sha256(EXP/'normalize.py'),'Changed wrapper')
   require(all(protocol['frozen_files'].get(n)==h for n,h in identity['source_files'].items()),'Unfrozen shared model code')
   for r,t in zip(campaign['results'],check['tasks']):
    require(r['id']==t['id'] and r['passed']==t['passed'] and r['calls']==t['input_token_hashes_verified'],'Row/trajectory mismatch')
    require(r['calls']==t['calls']==len(r['generations'])==len(t['generation_input_sha256'])==len(t['generation_output_sha256']),'Generation audit length mismatch')
    require([g['input_tokens_sha256'] for g in r['generations']]==t['generation_input_sha256'],'Exact input receipts mismatch')
    require(r['source_progress']==t['source_progress'],'Source progress audit mismatch')
    require(r['tool_calls']==len(r['observation_receipts'])==len(t['observation_receipts'])==len(r['source_progress']),'Tool audit length mismatch')
    for observed,verified in zip(r['observation_receipts'],t['observation_receipts']):
     require(all(verified.get(k)==v for k,v in observed.items()),'Tool observation audit mismatch')
    require(t['progress_native_history_and_output_transform_verified'] and t['final_source_verified'],'Incomplete exact-input audit')
    require(r['split']=='dev' and r['output_condition']==c and not r['backend_errors'],'Unexpected task/backend')
    require(r['calls']<=12 and r['generated_tokens']==sum(g['generated_tokens'] for g in r['generations'])<=6144,'Task budget exceeded')
    require(all(g['input_tokens']<=8192 and g['generated_tokens']<=512 for g in r['generations']),'Call budget exceeded')
 value=summarize(campaigns,audit)
 residual=read('feedback-residual-review.json')
 require(residual['post_hoc'] and not residual['preregistered_metric'] and not residual['normalizer_changed'],'Residual review is post-hoc only')
 require(not residual['raw_feedback_public'] and not residual['final_test_opened'] and residual['model_generations_performed']==residual['commands_executed']==0,'Residual diagnostic changed execution')
 require(residual['source_sha256']==sha256(EXP/'residual_review.py') and residual['execution_audit_sha256']==sha256(root/'execution-audit.json') and residual['analysis_sha256']==sha256(EXP/'analysis.py'),'Residual diagnostic identity mismatch')
 keys=['comparison','model','condition','a','b','id','call','outcome_disagreement']
 expected_residual=[{k:p[k] for k in keys} for p in value['pairs'] if p['category']=='other_tool_text']
 require([{k:p[k] for k in keys} for p in residual['pairs']]==expected_residual,'Residual diagnostic selected or changed comparisons')
 require(residual['reviewed_first_other_feedback_differences']==len(expected_residual) and residual['candidate_classes']==dict(Counter(r['candidate_class'] for r in residual['pairs'])),'Residual diagnostic counts mismatch')
 traces={name:{r['id']:r for r in cell['tasks']} for name,cell in audit['campaigns'].items()}
 for p in residual['pairs']:
  require(p['delivered_sha256']==[next(o['delivered_sha256'] for o in traces[p[k]][p['id']]['observation_receipts'] if o['model_call']==p['call']) for k in ['a','b']],'Residual diagnostic feedback identity mismatch')
 repeated_inputs=defaultdict(list)
 for name,cell in audit['campaigns'].items():
  for row in cell['tasks']:
   for call,(input_hash,output_hash) in enumerate(zip(row['generation_input_sha256'],row['generation_output_sha256']),1):
    repeated_inputs[(cell['model'],input_hash)].append({'cell':name,'id':row['id'],'call':call,'output_sha256':output_hash})
 value['repeated_identical_input_groups']=[{'model':model,'input_tokens_sha256':input_hash,'calls':len(rows),
  'distinct_output_texts':len({r['output_sha256'] for r in rows}),'observations':rows}
  for (model,input_hash),rows in sorted(repeated_inputs.items()) if len(rows)>1]
 value['repeated_identical_input_summary']=[]
 for model in MODELS:
  rows=[r for r in value['repeated_identical_input_groups'] if r['model']==model]
  value['repeated_identical_input_summary'].append({'model':model,'repeated_input_groups':len(rows),
   'calls_in_repeated_groups':sum(r['calls'] for r in rows),
   'groups_with_different_output_text':sum(r['distinct_output_texts']>1 for r in rows),
   'scope':'Post-hoc grouping of exact input-token hashes within each fixed model. Same generation limits and policy; includes reused calls, not independent new requests. Decoded output text hashes here; token-output hashes separately in fixed probes.'})
 def first_difference(x,y):
  return next((i+1 for i in range(max(len(x),len(y))) if (x[i] if i<len(x) else None)!=(y[i] if i<len(y) else None)),None)
 for pair in value['pairs']:
  a,b=traces[pair['a']][pair['id']],traces[pair['b']][pair['id']]
  commands=lambda row:[(r['model_call'],r['command_sha256']) for r in row['observation_receipts']]
  pair.update(first_different_input_call=first_difference(a['generation_input_sha256'],b['generation_input_sha256']),
   first_different_output_call=first_difference(a['generation_output_sha256'],b['generation_output_sha256']),
   input_sequences_identical=a['generation_input_sha256']==b['generation_input_sha256'],
   assistant_outputs_identical=a['generation_output_sha256']==b['generation_output_sha256'],
   tool_actions_identical=commands(a)==commands(b),source_progress_identical=a['source_progress']==b['source_progress'])
 value['behavior_stability']=[]
 value['outcome_flip_boundaries']=[]
 for pair in value['pairs']:
  if not pair['outcome_disagreement']:continue
  observations=[]
  for key in ['a','b']:
   row=traces[pair[key]][pair['id']]
   observations.append(next((o for o in row['observation_receipts'] if o['model_call']==pair['call']),None))
  fields={'comparison':pair['comparison'],'model':pair['model'],'id':pair['id'],'a':pair['a'],'b':pair['b'],
          'first_divergence_category':pair['category'],'first_feedback_call':pair['call'],
          'first_different_input_call':pair['first_different_input_call'],'first_different_output_call':pair['first_different_output_call']}
  if all(observations):
   a,b=observations
   fields.update(same_command=a['command_sha256']==b['command_sha256'],same_prior_source=a['prior_editable_state_sha256']==b['prior_editable_state_sha256'],
    same_tool_status=(a['returncode'],a['reason'])==(b['returncode'],b['reason']),same_canonical_feedback=a['canonical_sha256']==b['canonical_sha256'],
    eligible_fields=[a['eligible_fields'],b['eligible_fields']],delivered_sha256=[a['delivered_sha256'],b['delivered_sha256']])
   fields['difference_is_only_ls_date_fields']=(pair['category']=='allowed_output_metadata' and fields['same_canonical_feedback'] and all(set(k for k,v in o['eligible_fields'].items() if v)=={'ls_timestamp'} for o in observations))
  value['outcome_flip_boundaries'].append(fields)
 for model in MODELS:
  for condition in CONDITIONS:
   rows=[p for p in value['pairs'] if p['comparison']=='within_condition' and p['model']==model and p['condition']==condition]
   value['behavior_stability'].append({'model':model,'condition':condition,'pairs':45,
    'different_input_sequences':sum(not p['input_sequences_identical'] for p in rows),
    'different_assistant_outputs':sum(not p['assistant_outputs_identical'] for p in rows),
    'different_tool_actions':sum(not p['tool_actions_identical'] for p in rows),
    'different_grades':sum(p['outcome_disagreement'] for p in rows)})
 value['matched_condition_outcomes']=[]
 for model in MODELS:
  rows=[p for p in value['pairs'] if p['comparison']=='matched_condition' and p['model']==model]
  value['matched_condition_outcomes'].append({'model':model,'pairs':45,
   'canonical_gained_runs':sum(not p['a_passed'] and p['b_passed'] for p in rows),
   'canonical_regressed_runs':sum(p['a_passed'] and not p['b_passed'] for p in rows),
   'raw_passes':sum(p['a_passed'] for p in rows),'canonical_passes':sum(p['b_passed'] for p in rows),
   'scope':'45 repeated runs on15 reused tasks,not45 independent new tasks'})
 value['task_stability']=[]
 for m in MODELS:
  for c in CONDITIONS:
   names=[f'r{repeat:02d}-{m}-{c}' for repeat in [1,2,3]]
   for ident in ids:
    rows=[traces[n][ident] for n in names];outcomes=[r['passed'] for r in rows]
    value['task_stability'].append({'model':m,'condition':c,'id':ident,'repeat_passed':outcomes,
     'mixed_outcome':0<sum(outcomes)<3,'repeat_calls':[r['calls'] for r in rows],
     'repeat_first_edit_call':[r['first_edit_call'] for r in rows]})
 require(audit['task_runs']==270 and audit['probe_count']==36,'Missing runs/probes')
 require(sum(v['calls'] for v in value['cells'])==audit['generation_calls_audited']<=3240,'Call audit incomplete')
 probes=[read('probes/'+v['name']) for v in audit['probes_verified']]
 require(len({(p['model'],p['repeat'],p['id'],p['within_load_repeat']) for p in probes})==36,'Duplicate probe')
 for v in audit['probes_verified']:require(sha256(root/'probes'/v['name'])==v['sha256'],'Probe evidence changed')
 groups=[]
 for m in MODELS:
  for ident in plan['probes']['task_ids']:
   rows=[p for p in probes if p['model']==m and p['id']==ident]
   require(len(rows)==6 and {p['repeat'] for p in rows}=={1,2,3} and {p['within_load_repeat'] for p in rows}=={1,2},'Unbalanced probe repetitions')
   require(len({p['input_tokens_sha256'] for p in rows})==1 and all(p['generated_tokens']<=512 for p in rows),'Probe input/budget differs')
   groups.append({'model':m,'id':ident,'calls':6,'distinct_output_texts':len({p['output_sha256'] for p in rows}),
    'distinct_output_token_sequences':len({p['ids_sha256'] for p in rows}),
    'within_load_disagreements':sum(len({p['output_sha256'] for p in rows if p['repeat']==r})>1 for r in [1,2,3])})
 coverage=[]
 for m in MODELS:
  for c in CONDITIONS:
   rows=[r for n,cell in campaigns.items() if cell['identity']['model']==m and cell['identity']['variant']==c for r in cell['results']]
   observations=[o for r in rows for o in r['observation_receipts']]
   coverage.append({'model':m,'condition':c,'observations':len(observations),'changed_observations':sum(o['changed'] for o in observations),
    'eligible_fields':dict(sum((Counter(o['eligible_fields']) for o in observations),Counter())),
    'applied_fields':dict(sum((Counter(o['applied_fields']) for o in observations),Counter())),
    'extra_truncations':sum(c=='canonical' and o['canonical_extra_truncation'] for o in observations)})
 spec=importlib.util.spec_from_file_location('stability_hardware',EXP.parent/'022-action-repair/hardware.py');hardware=importlib.util.module_from_spec(spec);spec.loader.exec_module(hardware)
 samples=[]
 for p in sorted(root.glob('telemetry-attempt-*.csv')):
  with p.open() as f:samples+=list(csv.DictReader(f))
 timing={'started_utc':controller['started_utc'],'finished_utc':controller['finished_utc'],
  'wall_seconds':controller['finished_epoch']-controller['started_epoch'],'within_original_deadline':controller['finished_epoch']<=controller['deadline_epoch']}
 for k in ['started','finished']:timing[k+'_beijing']=datetime.fromisoformat(controller[k+'_utc']).astimezone(timezone(timedelta(hours=8))).isoformat()
 require(timing['within_original_deadline'],'Original wall deadline exceeded')
 return {'schema_version':1,**value,'residual_feedback_review':residual,'probe_groups':groups,'probe_calls':36,'probe_generated_tokens':sum(p['generated_tokens'] for p in probes),
  'normalization_coverage':coverage,'runtime_settings':runtime,
  'runtime_settings_identical_across_loads':len({json.dumps({k:v for k,v in r.items() if k not in ['block','attempt']},sort_keys=True) for r in runtime})==1,
  'hardware':hardware.summarize(samples),
  'sampler_errors':controller['telemetry']['errors'],'time':timing,'report_sha256':sha256(Path(__file__)),
  'analysis_sha256':sha256(EXP/'analysis.py'),'hardware_summary_sha256':sha256(EXP.parent/'022-action-repair/hardware.py'),
  'limitations':plan['limitations']}

def render(root):
 root=Path(root);value=assess(root);write_json(root/'summary.json',value)
 for name,rows in [('paired-stability.csv',value['pairs']),('repeated-scores.csv',value['cells']),('task-stability.csv',value['task_stability'])]:
  with (root/name).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
   writer.writerows({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'one-gpu-lm-023'})
 colors={'raw':'#64748b','canonical':'#167c80'}
 def save(fig,name):
  fig.tight_layout()
  for suffix in ['png','svg']:fig.savefig(root/(name+'.'+suffix),dpi=170,metadata={'Date':None} if suffix=='svg' else None)
  plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,3.4),sharey=True)
 for ax,m in zip(axes,MODELS):
  for c in CONDITIONS:
   row=next(r for r in value['stability'] if r['model']==m and r['condition']==c);ax.plot([1,2,3],row['repeat_passes'],marker='o' if c=='raw' else 'x',linestyle='--' if c=='raw' else '-',label=c,color=colors[c])
  ax.set(title=m,xticks=[1,2,3],xlabel='Repeat',ylim=(-.5,15.5));ax.grid(alpha=.2)
 axes[0].set_ylabel('Development passes /15');axes[-1].legend();save(fig,'repeated-scores')
 fig,axes=plt.subplots(1,2,figsize=(10,3.8))
 for j,c in enumerate(CONDITIONS):
  rows=[next(r for r in value['stability'] if r['model']==m and r['condition']==c) for m in MODELS]
  for ax,key,label in [(axes[0],'mixed_outcome_tasks','Tasks with pass/fail flips /15'),(axes[1],'pairwise_disagreements','Disagreeing repeat pairs /45')]:
   positions=[i+(j-.5)*.34 for i in range(3)];counts=[r[key] for r in rows]
   ax.bar(positions,counts,width=.34,label=c,color=colors[c]);ax.set(xticks=range(3),xticklabels=MODELS,ylabel=label)
   for position,count in zip(positions,counts):ax.text(position,count+.04,str(count),ha='center',va='bottom',fontsize=10)
 from matplotlib.ticker import MaxNLocator
 for ax in axes:ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.set_ylim(0,ax.get_ylim()[1]+.2)
 axes[-1].legend();save(fig,'outcome-stability')
 fig,ax=plt.subplots(figsize=(11,4));rows=value['behavior_stability']
 for j,(key,label,color) in enumerate([('different_input_sequences','Input sequence','#94a3b8'),('different_tool_actions','Executed actions','#167c80'),('different_grades','Grade','#b45309')]):
  ax.bar([i+(j-1)*.24 for i in range(6)],[r[key] for r in rows],width=.24,label=label,color=color)
 ax.set(xticks=range(6),xticklabels=[r['model']+'\n'+r['condition'] for r in rows],ylabel='Disagreeing repeat pairs /45',ylim=(0,47),title='Feedback variation is separate from action and grade variation');ax.legend();save(fig,'behavior-stability')
 categories=sorted({r['category'] for r in value['pairs'] if r['comparison']=='within_condition'})
 fig,ax=plt.subplots(figsize=(11,5));labels=[r['model']+'\n'+r['condition'] for r in value['stability']];bottom=[0]*6
 for cat in categories:
  counts=[r['first_divergence_categories'].get(cat,0) for r in value['stability']];ax.bar(range(6),counts,bottom=bottom,label=cat.replace('_',' '));bottom=[a+b for a,b in zip(bottom,counts)]
 ax.set(xticks=range(6),xticklabels=labels,ylabel='Repeat comparisons /45',title='First observed trace difference (descriptive)');ax.legend(fontsize=8,bbox_to_anchor=(1.01,1),loc='upper left');save(fig,'trace-divergence')
 fig,ax=plt.subplots(figsize=(9,3.5));probe=value['probe_groups'];ax.bar(range(6),[r['distinct_output_token_sequences'] for r in probe],color='#b45309')
 ax.set(xticks=range(6),xticklabels=[r['model']+'\n'+r['id'].removeprefix('fast-deep-equal-') for r in probe],ylabel='Distinct token outputs /6 requests',ylim=(0,6.5),title='Fixed identical-input probes');save(fig,'identical-input-probes')
 samples=[]
 for p in sorted(root.glob('telemetry-attempt-*.csv')):
  with p.open() as f:samples+=list(csv.DictReader(f))
 # Use the common timestamp/gap logic; never connect unobserved intervals.
 if samples:
  spec=importlib.util.spec_from_file_location('stability_plot_hardware',EXP.parent/'022-action-repair/hardware.py');hw=importlib.util.module_from_spec(spec);spec.loader.exec_module(hw)
  fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True)
  for ax,k,label in [(axes[0],'temperature_c','GPU temperature (C)'),(axes[1],'memory_used_mib','GPU memory (MiB)'),(axes[2],'power_w','Board power (W)')]:
   minutes,y=hw.plot_series(samples,k);ax.plot([v/60 for v in minutes],y,linewidth=.8);ax.set_ylabel(label);ax.grid(alpha=.2)
  axes[-1].set_xlabel('Hours since first sample');save(fig,'hardware-timeline')
 table=['| Model | Output | Repeat passes /15 | Mixed outcomes /15 | Disagreeing pairs /45 |','| --- | --- | --- | --- | --- |']
 for r in value['stability']:table.append(f"| {r['model']} | {r['condition']} | {r['repeat_passes']} | {r['mixed_outcome_tasks']} | {r['pairwise_disagreements']} |")
 behavior=['| Model | Output | Input differences /45 | Action differences /45 | Grade differences /45 |','| --- | --- | --- | --- | --- |']
 for r in value['behavior_stability']:behavior.append(f"| {r['model']} | {r['condition']} | {r['different_input_sequences']} | {r['different_tool_actions']} | {r['different_grades']} |")
 quality=['| Model | Raw passes /45 runs | Canonical passes /45 runs | Canonical gains | Canonical regressions |','| --- | --- | --- | --- | --- |']
 for r in value['matched_condition_outcomes']:quality.append(f"| {r['model']} | {r['raw_passes']} | {r['canonical_passes']} | {r['canonical_gained_runs']} | {r['canonical_regressed_runs']} |")
 coverage=['| Model | Output | Observations | Changed observations | Eligible fields | Applied fields |','| --- | --- | --- | --- | --- | --- |']
 for r in value['normalization_coverage']:coverage.append(f"| {r['model']} | {r['condition']} | {r['observations']} | {r['changed_observations']} | {r['eligible_fields']} | {r['applied_fields']} |")
 identical=['| Model | Repeated exact-input groups | Calls in these groups | Groups with different output text |','| --- | --- | --- | --- |']
 for r in value['repeated_identical_input_summary']:identical.append(f"| {r['model']} | {r['repeated_input_groups']} | {r['calls_in_repeated_groups']} | {r['groups_with_different_output_text']} |")
 unstable=['| Model | Output | Task | Repeat pass/fail | First edit call by repeat |','| --- | --- | --- | --- | --- |']
 for r in value['task_stability']:
  if r['mixed_outcome']:unstable.append(f"| {r['model']} | {r['condition']} | {r['id']} | {r['repeat_passed']} | {r['repeat_first_edit_call']} |")
 if len(unstable)==2:unstable=['No task changed its pass/fail result within either condition in these three repeats.']
 lines=['# 023 · Repeated evaluation stability','', '[中文结论与行为复盘](behavior-review.md)','',
 'Development-only diagnostic. No training, model selection, final-test opening or weight publication occurred. All 270 task runs and 36 fixed-input probes were retained. Three repeats use the same 15 development tasks across four repositories; they are not independent new tasks.','',*table,'',
 '![Repeated scores](repeated-scores.png)','![Outcome stability](outcome-stability.png)','',*quality,'',
 'Stable behavior and higher task success are separate measurements. The table above counts 45 repeated runs per model and condition, not 45 independent tasks.','',*unstable,'',
 'The table lists every within-condition pass/fail flip. [All 90 model/condition/task records](task-stability.csv) also retain stable successes and stable failures.','',*behavior,'',
 '![Input, action and grade stability](behavior-stability.png)','![First trace divergence](trace-divergence.png)','![Identical-input probes](identical-input-probes.png)','',*identical,'',
 'The exact-input grouping is an additional post-hoc descriptive check over task calls, within each fixed model. It checks decoded output text; the 36 preregistered probes separately compare output token sequences. Neither establishes determinism for all future inputs.','',*coverage,'',
 f"Post-hoc residual-feedback review: {value['residual_feedback_review']['candidate_classes']}. This diagnostic was added after observing the first pair and did not change model inputs or the preregistered primary analysis. See [every reviewed comparison](feedback-residual-review.json).",'',
 'A stable score can still be wrong. A first differing tool response or generated action locates an observed boundary,not the sole cause of the final grade. The output intervention combines file-list dates and JS-summary durations; it does not isolate their individual effects or remove all possible variability.','',
 'The raw prefix is capped before normalization; file names, commands, source, return codes and independent grading remain unchanged. Raw/canonical conditions share the same upper bounds,not identical token counts or FLOPs.','',
 f"Actual Beijing time: {value['time']['started_beijing']} to {value['time']['finished_beijing']}; {value['time']['wall_seconds']/3600:.3f} hours within the original deadline. The whole-run time includes model loading, controls and probes; [per-cell costs](repeated-scores.csv) separately count task runtime and generated tokens.",'',
 f"Hardware samples: {value['hardware']['samples']}; maximum gap {value['hardware']['max_gap_seconds']:.2f}s. Sample-interval coverage: {value['hardware']['interval_coverage_fraction']:.2%}. Sampled board-power estimate: {value['hardware']['sampled_gpu_energy_kwh']:.3f} kWh. This covers sampled intervals only, not whole-system use. Sampler errors: {value['sampler_errors']}.",'',
 '![Hardware timeline](hardware-timeline.png)','',
 *[str(x) for x in value['limitations']],'',
 '[Complete evidence](summary.json) · [Every paired comparison](paired-stability.csv) · [Per-repeat costs](repeated-scores.csv) · [Frozen rules](protocol.json)']
 lines += ['', 'From the repository root, verify the published numeric evidence with Python 3.10+ (standard library only):', '',
  '```sh','python experiments/023-evaluation-stability/report.py --run experiments/023-evaluation-stability/results/2026-11-13 --check-only','```','',
  'Omit `--check-only` to regenerate tables and PNG/SVG curves; rendering also requires Matplotlib. No GPU, model weights or private task text is needed for this numeric recomputation. The private-input audits themselves require the original private journals and tokenizer.']
 (root/'README.md').write_text('\n'.join(lines)+'\n');return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args();v=assess(a.run) if a.check_only else render(a.run);print(json.dumps({'task_runs':270,'probe_calls':36,'final_test_opened':False}))
