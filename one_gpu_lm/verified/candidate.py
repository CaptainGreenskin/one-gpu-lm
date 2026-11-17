"""One registered LoRA candidate from complete audited episodes; no held-out access."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.hf_runtime import Runtime
from one_gpu_lm.models.verify import verify_model
from .candidate_data import select, windows
from .rollout import atomic
from .tasks import relative
from .runtime_identity import verify as verify_environment


def validate_recipe(profile, spec):
    frozen=spec['candidate']; train=profile['training']; lora=profile['lora']
    direct=['seed','epochs','max_sequence_length','learning_rate','max_grad_norm','gradient_checkpointing']
    if any(train[k]!=frozen[k] for k in direct):
        raise ValueError('Candidate profile differs from registered recipe')
    if (lora['r']!=frozen['lora_r'] or lora['lora_alpha']!=frozen['lora_alpha']
            or lora['lora_dropout']!=frozen['lora_dropout']):
        raise ValueError('Candidate LoRA profile differs from registered recipe')
    if (frozen['count']!=1 or frozen['epochs']!=1 or frozen['micro_batch_size']!=1
            or frozen['optimizer']!='AdamW' or frozen['scheduler']!='constant'
            or profile['dtype']!=frozen['dtype'] or profile['attention']!=frozen['attention']
            or train['dialogue_format']!='native_roundtrip'):
        raise ValueError('Unsupported or changed registered recipe')


def run(root, protocol_path):
    root, protocol_path=Path(root).resolve(),Path(protocol_path).resolve()
    protocol=json.loads(protocol_path.read_text())
    out=root/relative(protocol['run_directory'])
    if out.exists():
        raise ValueError('A candidate attempt already exists; no overwrite or retry')
    publication=json.loads((root/relative(protocol['publication_receipt'])).read_text())
    if (publication['anonymous_files_verified'] is not True or publication['ci_conclusion']!='success'
            or publication['protocol_sha256']!=sha256(protocol_path)):
        raise ValueError('Executable candidate preregistration must be publicly verified')
    for name,expected in protocol['frozen_files'].items():
        if sha256(root/relative(name))!=expected:
            raise ValueError('Frozen candidate input changed: '+name)
    spec=json.loads((root/relative(protocol['cycle_spec'])).read_text())
    profile=json.loads((root/relative(protocol['profile_path'])).read_text());validate_recipe(profile,spec)
    capacity_profile=json.loads((root/relative(protocol['capacity_profile_path'])).read_text())
    if profile['lora']['target_modules']!=capacity_profile['lora']['target_modules']:
        raise ValueError('Target module expression differs from capacity check')
    capacity=json.loads((root/relative(protocol['capacity_result'])).read_text())
    if not capacity['capacity_gate_passed']:
        raise ValueError('Capacity gate not passed')
    controller=json.loads((root/relative(protocol['collection_controller'])).read_text())
    supervisor=json.loads((root/relative(protocol['collection_supervisor'])).read_text())
    if (not controller['complete'] or not controller['backend_stopped'] or not controller['frozen_files_unchanged']
            or not supervisor['complete'] or supervisor['timed_out'] or supervisor['worker_returncode']!=0):
        raise ValueError('Formal collection must have completed and stopped')
    audit=json.loads((root/relative(protocol['audit_path'])).read_text())
    manifest=json.loads((root/relative(protocol['export_manifest'])).read_text())
    data_path=root/relative(protocol['data_path'])
    if (manifest['data_sha256']!=sha256(data_path) or manifest['audit_sha256']!=sha256(root/relative(protocol['audit_path']))
            or manifest['protocol_sha256']!=controller['protocol_sha256']
            or audit['protocol_sha256']!=controller['protocol_sha256']):
        raise ValueError('Collection audit/export lineage mismatch')
    sources=json.loads((root/relative(protocol['sources_path'])).read_text())['sources']
    import subprocess
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise ValueError('GPU occupied; no competing candidate update')
    out.mkdir();(out/'private').mkdir()
    started=time.monotonic(); started_epoch=time.time()
    result={'schema_version':1,'stage':'starting','complete':False,'started_epoch':started_epoch,
            'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha256(protocol_path),
            'profile_sha256':sha256(root/relative(protocol['profile_path'])),'data_sha256':sha256(data_path),
            'optimizer_steps':0,'action_records_consumed':0,'development_scores':0,'final_test_opened':False,
            'candidate_count':1,'weights_public':False,'publication_commit':publication['commit']}
    curve=[]; updates=[]
    def save(stage):
        result['stage']=stage;atomic(out/'result.json',result)
    save('loading');telemetry=Telemetry(out/'telemetry.csv',1);telemetry.start()
    try:
        environment_path=root/relative(protocol['environment_path'])
        write_json(out/'environment.json',verify_environment(json.loads(environment_path.read_text())))
        result['expected_environment_sha256']=sha256(environment_path)
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        runtime_info=json.loads((root/'private/parent-runtime.json').read_text())
        model_path=Path(runtime_info['model'])
        verification=verify_model(model_path,root/relative(protocol['model_manifest_path']),profile)
        write_json(out/'model-verification.json',verification)
        runtime=Runtime(model_path,profile,out)
        rows=[json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
        mixed_selection=None
        if protocol.get('correction_mix'):
            from .mixed_inputs import select as select_mixed
            selected,mixed_selection=select_mixed(root,protocol,profile,runtime.tokenizer,rows,manifest)
        else:
            selected=select(rows,manifest,audit,sources,profile,spec,runtime.tokenizer)
        selection={'full_episodes':manifest['full_episodes'],'unique_tasks':manifest['unique_tasks'],
                   'selected_records':len(selected),'excluded_records':0,'target_truncation':False,
                   'records':[{'id':r['id'],'episode_id':r['episode_id'],'tokens':len(r['ids']),
                               'supervised_tokens':r['supervised_tokens']} for r in selected]}
        if mixed_selection is not None:
            selection['mixed_selection']=mixed_selection
            selection['unique_tasks']=mixed_selection['unique_tasks']
            result['mixed_selection']=mixed_selection
            for saved,row in zip(selection['records'],selected):saved['record_group_kind']=row['record_group_kind']
        write_json(out/'selection.json',selection)
        torch.manual_seed(spec['candidate']['seed'])
        model=get_peft_model(runtime.model,LoraConfig(task_type=TaskType.CAUSAL_LM,**profile['lora']))
        parameters=[p for p in model.parameters() if p.requires_grad]
        trainable=sum(p.numel() for p in parameters)
        if not parameters or trainable!=protocol['expected_trainable_parameters']:
            raise ValueError('Trainable parameter count differs from verified capacity shape')
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        recipe=spec['candidate']
        optimizer=torch.optim.AdamW(parameters,lr=recipe['learning_rate'],betas=tuple(recipe['adam_betas']),
                   eps=recipe['adam_epsilon'],weight_decay=recipe['weight_decay'],foreach=False)
        frozen_samples=[(p,p.detach().reshape(-1)[:8].clone()) for p in model.parameters() if not p.requires_grad][:3]
        adapter_before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        model.train();torch.cuda.reset_peak_memory_stats();training_started=time.monotonic()
        result.update(trainable_parameters=trainable,selected_records=len(selected),full_episodes=manifest['full_episodes'])
        for window in windows(selected,recipe['gradient_accumulation_records']):
            if time.monotonic()-started>=recipe['maximum_training_pipeline_seconds']:
                raise TimeoutError('Registered training cap exceeded; partial candidate is ineligible')
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.monotonic()
            for row in window:
                ids=torch.tensor([row['ids']],device='cuda');labels=torch.tensor([row['labels']],device='cuda')
                torch.cuda.synchronize();micro_tick=time.monotonic()
                output=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False)
                loss=output.loss
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss/len(window)).backward();torch.cuda.synchronize()
                curve.append({'record':len(curve)+1,'id':row['id'],'episode_id':row['episode_id'],
                              'update':result['optimizer_steps']+1,'tokens':len(row['ids']),
                              'supervised_tokens':row['supervised_tokens'],'loss':loss.item(),
                              'seconds':time.monotonic()-micro_tick})
                if mixed_selection is not None:curve[-1]['record_group_kind']=row['record_group_kind']
                result['action_records_consumed']=len(curve)
                atomic(out/'curve.json',curve);save('training')
                del loss,output,ids,labels
            norm=torch.nn.utils.clip_grad_norm_(parameters,recipe['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize();result['optimizer_steps']+=1
            updates.append({'step':result['optimizer_steps'],'records':len(window),'grad_norm':norm.item(),
                            'learning_rate':recipe['learning_rate'],'seconds':time.monotonic()-tick})
            atomic(out/'updates.json',updates);save('training')
            print(json.dumps({'step':result['optimizer_steps'],'records_consumed':len(curve),'total_records':len(selected)}),flush=True)
        result.update(training_seconds=time.monotonic()-training_started,
          input_tokens=sum(r['tokens'] for r in curve),supervised_tokens=sum(r['supervised_tokens'] for r in curve),
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
          frozen_sample_unchanged=all(torch.equal(p.detach().reshape(-1)[:8],v) for p,v in frozen_samples),
          frozen_gradients_absent=all(p.grad is None for p in model.parameters() if not p.requires_grad),
          adapter_changed=any(not torch.equal(p.detach(),adapter_before[n]) for n,p in model.named_parameters() if p.requires_grad),
          all_selected_records_consumed=[r['id'] for r in curve]==[r['id'] for r in selected])
        if not all(result[k] for k in ['frozen_sample_unchanged','frozen_gradients_absent','adapter_changed','all_selected_records_consumed']):
            raise ValueError('Parameter or complete-epoch integrity check failed')
        del optimizer,adapter_before
        for p in parameters:p.grad=None
        model.gradient_checkpointing_disable()
        model.peft_config['default'].base_model_name_or_path=profile['model_id']
        model.peft_config['default'].revision=profile['revision']
        adapter=out/'private/adapter';model.save_pretrained(adapter,safe_serialization=True)
        write_json(out/'adapter.json',{'base_model':profile['model_id'],'revision':profile['revision'],
                   'weights_public':False,'files':{p.name:sha256(p) for p in adapter.iterdir() if p.is_file()}})
        runtime.model=model.eval();probe=rows[0]['messages']
        request={'messages':probe,'max_new_tokens':16,'agent_policy':{'format':'native_roundtrip','context':'bounded_history_v1'}}
        before=runtime.generate(request)['ids'];base=model.unload()
        runtime.model=PeftModel.from_pretrained(base,adapter,is_trainable=False).eval()
        after=runtime.generate(request)['ids']
        result['saved_reload_greedy_ids_equal']=before==after
        result['reload_probe_ids_sha256']=hashlib.sha256(json.dumps(after).encode()).hexdigest()
        if before!=after:raise ValueError('Saved adapter reload changed the training-only probe')
        gc.collect();torch.cuda.empty_cache()
        result['complete']=True;save('completed')
    except BaseException as error:
        result.update(error_type=type(error).__name__,complete=False)
        (out/'private/error.log').write_text(traceback.format_exc());save('failed')
        raise
    finally:
        result.update(telemetry=telemetry.stop(),finished_epoch=time.time(),finished_utc=datetime.now(timezone.utc).isoformat(),
          frozen_files_unchanged=all((root/relative(n)).is_file() and sha256(root/relative(n))==h for n,h in protocol['frozen_files'].items()))
        if not result['frozen_files_unchanged']:
            result.update(complete=False,stage='frozen_input_changed')
        atomic(out/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args();run(a.root,a.protocol)
