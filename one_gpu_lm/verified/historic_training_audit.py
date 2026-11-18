"""CPU-only reconciliation of the entire 032 mixed-data training epoch and adapter."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from one_gpu_lm.common.receipts import sha256,write_json
from .candidate_data import windows
from .evaluation import validate_candidate
from .historic_inputs import select
from .tasks import relative


def reconcile(selected,metadata,selection,curve,updates,result,recipe):
    expected=[{'id':r['id'],'episode_id':r['episode_id'],'tokens':len(r['ids']),
               'supervised_tokens':r['supervised_tokens'],'record_group_kind':r['record_group_kind']} for r in selected]
    batches=windows(selected,recipe['gradient_accumulation_records'])
    # Materialize once: windows is an iterator in the training implementation.
    batches=list(batches);expected_updates=[];n=0
    for step,batch in enumerate(batches,1):
        expected_updates.append({'step':step,'records':len(batch),'learning_rate':recipe['learning_rate']})
        n+=len(batch)
    curve_identity=[{k:r[k] for k in expected[0]} for r in curve] if expected else []
    finite=all(math.isfinite(r[k]) and r[k]>=0 for r in curve for k in ['loss','seconds'])
    finite=finite and all(math.isfinite(r[k]) and r[k]>=0 for r in updates for k in ['grad_norm','seconds','learning_rate'])
    checks={
      'exact_selected_records':selection['records']==expected and selection['selected_records']==len(selected),
      'mixed_lineage_and_counts':selection['mixed_selection']==metadata==result['mixed_selection'],
      'complete_epoch_order':curve_identity==expected and result['action_records_consumed']==len(selected),
      'unique_records':len({r['id'] for r in curve})==len(selected),
      'record_and_update_numbers':[r['record'] for r in curve]==list(range(1,len(selected)+1))
          and [r['update'] for r in curve]==[i for i,b in enumerate(batches,1) for _ in b],
      'all_registered_optimizer_updates':[{k:r[k] for k in ['step','records','learning_rate']} for r in updates]==expected_updates
          and result['optimizer_steps']==len(batches) and n==len(selected),
      'all_losses_and_norms_finite':finite,
      'token_accounting':result['input_tokens']==sum(r['tokens'] for r in curve)
          and result['supervised_tokens']==sum(r['supervised_tokens'] for r in curve)==metadata['supervised_tokens'],
      'no_truncation_or_exclusion':selection['excluded_records']==0 and selection['target_truncation'] is False,
      'training_only':result['development_scores']==0 and result['final_test_opened'] is False and result['weights_public'] is False,
      'within_original_pipeline_cap':0<=result['finished_epoch']-result['started_epoch']<=recipe['maximum_training_pipeline_seconds'],
    }
    return {'checks':checks,'passed':all(checks.values()),'selected_records':len(selected),'optimizer_updates':len(batches),
      'last_accumulation_records':len(batches[-1]) if batches else 0,
      'record_groups':dict(Counter(r['record_group_kind'] for r in selected)),
      'input_tokens':sum(r['tokens'] for r in curve),'supervised_tokens':sum(r['supervised_tokens'] for r in curve)}


def run(root,protocol_path,out):
    from transformers import AutoTokenizer
    root,protocol_path,out=Path(root).resolve(),Path(protocol_path).resolve(),Path(out)
    if out.exists():raise ValueError('Preserve the original training integrity receipt')
    read=lambda n:json.loads((root/relative(n)).read_text())
    p=json.loads(protocol_path.read_text());directory=root/relative(p['run_directory'])
    result=json.loads((directory/'result.json').read_text());supervisor=read(p['supervisor_receipt'])
    validate_candidate(result,supervisor,sha256(protocol_path))
    profile=read(p['profile_path']);recipe=read(p['cycle_spec'])['candidate'];runtime=read('private/parent-runtime.json')
    tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
    rows=[json.loads(line) for line in (root/relative(p['data_path'])).read_text().splitlines() if line.strip()]
    selected,metadata=select(root,p,profile,tokenizer,rows,read(p['export_manifest']))
    selection=json.loads((directory/'selection.json').read_text());curve=json.loads((directory/'curve.json').read_text());updates=json.loads((directory/'updates.json').read_text())
    value=reconcile(selected,metadata,selection,curve,updates,result,recipe)
    adapter=json.loads((directory/'adapter.json').read_text());adapter_dir=directory/'private/adapter'
    files={f.name:sha256(f) for f in adapter_dir.iterdir() if f.is_file()}
    hashes={name:sha256(directory/name) for name in ['result.json','curve.json','updates.json','selection.json','adapter.json','environment.json']}
    flags={
      'candidate_and_supervisor_eligible':True,
      'all_frozen_inputs_unchanged':all((root/relative(n)).is_file() and sha256(root/relative(n))==h for n,h in p['frozen_files'].items()),
      'profile_identity':result['profile_sha256']==sha256(root/relative(p['profile_path'])),
      'environment_identity':json.loads((directory/'environment.json').read_text())==read(p['environment_path'])
          and result['expected_environment_sha256']==sha256(root/relative(p['environment_path'])),
      'data_identity':result['data_sha256']==sha256(root/relative(p['data_path'])),
      'trainable_parameter_count':result['trainable_parameters']==p['expected_trainable_parameters'],
      'saved_adapter_identity':files==adapter['files'] and any(n.endswith('.safetensors') for n in files)
          and adapter['base_model']==profile['model_id'] and adapter['revision']==profile['revision'] and not adapter['weights_public'],
      'parameter_checks_and_training_only_reload':all(result[k] is True for k in ['frozen_sample_unchanged','frozen_gradients_absent','adapter_changed','saved_reload_greedy_ids_equal']),
      'supervisor_protocol_identity':supervisor['protocol_sha256']==sha256(protocol_path),
    }
    value['checks'].update(flags);value.update(schema_version=1,passed=all(value['checks'].values()),protocol_sha256=sha256(protocol_path),
      receipt_sha256=hashes,adapter_files_sha256=files,mixture=metadata,actual_tokenizer=True,
      audit_model_generations=0,audit_optimizer_updates=0,final_test_opened=False,weights_public=False,
      limitations=['Frozen value check samples three tensors, eight values each; gradient absence checks all frozen parameters.',
                  'Two training-only greedy probes (at most16 generated tokens each) verify save/reload; their token IDs are hashed, not public.'])
    write_json(out,value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','protocol','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();v=run(a.root,a.protocol,a.out);print(json.dumps({'passed':v['passed'],'checks':v['checks']}))
