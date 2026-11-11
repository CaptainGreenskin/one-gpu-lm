"""Bounded, resumable single-GPU LoRA with immutable input identities and durable checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
import sys
import time

from one_gpu_lm.common.receipts import write_json
from one_gpu_lm.common.runstate import atomic_json, event, gpu_lock_path, identity_digest, single_writer, utc
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.hf_runtime import Runtime
from one_gpu_lm.common.checkpoints import commit_checkpoint, load_checkpoint
from .config import load_config, read_training_data
from .doctor import inspect
from .repository_lora import select_examples
from .state import capture_rng, cpu_tree, restore_rng, state_digest


def example_id(value):
    return hashlib.sha256(value.encode()).hexdigest()


def execute(config, out, *, resume=False, stop_after_step=None):
    out=Path(out).resolve()
    if out.exists() and not resume:
        raise ValueError('Run exists; preserve it and use explicit --resume')
    if not out.exists() and resume:
        raise ValueError('Cannot resume a missing run')
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    (out/'private').mkdir(exist_ok=True,mode=0o700)
    if stop_after_step is not None and not 0<stop_after_step<config['run']['max_steps']:
        raise ValueError('Controlled interruption must precede the final update')
    with single_writer(out/'private/run.lock'),single_writer(gpu_lock_path()):
        return _locked(config,out,resume,stop_after_step)


def _locked(config,out,resume,stop_after_step):
    if (out/'result.json').exists() and json.loads((out/'result.json').read_text())['status']=='completed':
        raise ValueError('Completed runs are immutable')
    attempts=json.loads((out/'attempts.json').read_text()) if (out/'attempts.json').exists() else []
    attempt=len(attempts)+1
    attempts.append({'attempt':attempt,'started_utc':utc(),'status':'running','resumed':resume})
    atomic_json(out/'attempts.json',attempts)
    started=time.monotonic();stop_requested=[];monitor=None;history=[];identity=None
    handlers={s:signal.getsignal(s) for s in [signal.SIGINT,signal.SIGTERM]}
    for sig in handlers:signal.signal(sig,lambda signum,frame:stop_requested.append(signum))
    def status(stage,**extra):
        atomic_json(out/'status.json',{'stage':stage,'attempt':attempt,'utc':utc(),**extra})
        print(stage,json.dumps(extra),flush=True)
    event(out,'attempt_started',attempt=attempt,resumed=resume)
    try:
        status('preflight')
        check=inspect(config,out)
        identity=check['identity']
        if (out/'identity.json').exists():
            if json.loads((out/'identity.json').read_text())!=identity:
                raise ValueError('Resume rejected: code, data, profile, budget or environment changed')
        else:
            if (out/'latest.json').exists():raise ValueError('Checkpoint has no run identity')
            atomic_json(out/'identity.json',identity)
            (out/'profile.json').write_bytes(config['paths']['profile'].read_bytes())
            write_json(out/'private/input-locations.json',{k:str(v) for k,v in config['paths'].items()})
        (out/'preflight').mkdir(exist_ok=True)
        write_json(out/'preflight'/f'attempt-{attempt:04d}.json',check)
        rows,approved=read_training_data(config)
        checkpoint=None;trainer=None
        if (out/'latest.json').exists():
            if not resume:raise ValueError('Existing checkpoint requires resume')
            checkpoint,trainer,_=load_checkpoint(out,identity)
            history=trainer['history']
            if len(history)!=trainer['step'] or [r['step'] for r in history]!=list(range(1,len(history)+1)):
                raise ValueError('Checkpoint history is not a contiguous committed prefix')
            if (out/'metrics.json').exists():
                previous=json.loads((out/'metrics.json').read_text())
                if previous!=history:
                    write_json(out/'private'/f'orphan-metrics-attempt-{attempt:04d}.json',previous)
                    event(out,'uncommitted_metrics_discarded',attempt=attempt,previous_rows=len(previous),committed_rows=len(history))
            atomic_json(out/'metrics.json',history)
        if stop_requested:
            return _pause(out,history,'signal_during_preflight',status)
        monitor=Telemetry(out/'telemetry.csv',attempt,config['run']['telemetry_interval_seconds']);monitor.start()
        (out/'loading').mkdir(exist_ok=True)
        load_out=out/'private'/f'load-{attempt:04d}';load_out.mkdir()
        status('loading_model',committed_step=len(history))
        runtime=Runtime(config['paths']['model_dir'],config['profile'],load_out)
        (out/'loading'/f'attempt-{attempt:04d}.json').write_bytes((load_out/'loading.json').read_bytes())
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file
        settings=config['profile']['training']
        torch.manual_seed(settings['seed']);random.seed(settings['seed'])
        encoded,excluded=select_examples(rows,runtime.tokenizer,config['profile'],approved,config['run']['max_examples'])
        if config['run']['max_steps']>len(encoded):
            raise ValueError('Requested steps exceed one pass over selected examples')
        order=list(range(len(encoded)));random.Random(settings['seed']).shuffle(order)
        order=order[:config['run']['max_steps']]
        schedule=[{'example_sha256':example_id(encoded[i]['id']),'tokens':len(encoded[i]['ids']),
                   'supervised_tokens':encoded[i]['supervised_tokens']} for i in order]
        order_sha=identity_digest(schedule)
        selection={'selected_records':len(encoded),'excluded_records':len(excluded),'schedule':schedule,
                   'order_sha256':order_sha,'raw_examples_public':False}
        if trainer and trainer['order_sha256']!=order_sha:
            raise ValueError('Tokenization or sample order changed during resume')
        atomic_json(out/'selection.json',selection)
        model=get_peft_model(runtime.model,LoraConfig(task_type=TaskType.CAUSAL_LM,**config['profile']['lora']))
        model.peft_config['default'].base_model_name_or_path=config['profile']['model_id']
        model.peft_config['default'].revision=config['profile']['revision']
        parameters=[p for p in model.parameters() if p.requires_grad]
        parameter_spec=[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in model.named_parameters() if p.requires_grad]
        count=sum(p.numel() for p in parameters)
        if not 0<count<=100_000_000:raise ValueError('Unexpected trainable parameter count')
        spec_sha=identity_digest(parameter_spec)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=torch.optim.AdamW(parameters,lr=settings['learning_rate'],weight_decay=0.0)
        initial_adapter_digest=state_digest(get_peft_model_state_dict(model))
        if trainer:
            if trainer['parameter_spec_sha256']!=spec_sha:
                raise ValueError('Trainable parameter ordering or precision changed')
            saved=load_file(checkpoint/'adapter/adapter_model.safetensors',device='cpu')
            current=get_peft_model_state_dict(model)
            if set(saved)!=set(current) or any(saved[k].shape!=current[k].shape or saved[k].dtype!=current[k].dtype for k in saved):
                raise ValueError('Saved adapter keys/shapes/dtypes differ')
            outcome=set_peft_model_state_dict(model,saved)
            if outcome.unexpected_keys:raise ValueError('Unexpected adapter keys')
            saved_state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
            optimizer.load_state_dict(saved_state['optimizer'])
            restore_rng(saved_state['rng'])
            checks={'adapter':state_digest(get_peft_model_state_dict(model))==trainer['adapter_state_sha256'],
                    'optimizer':state_digest(optimizer.state_dict())==trainer['optimizer_state_sha256'],
                    'rng':state_digest(capture_rng())==trainer['rng_state_sha256']}
            if not all(checks.values()):raise ValueError('Restored training state differs from checkpoint')
            event(out,'state_restored',attempt=attempt,step=len(history),checks=checks)
            initial_adapter_digest=trainer['initial_adapter_sha256']
            del saved,saved_state,current
        frozen=[(p,p.detach().reshape(-1)[:8].clone()) for p in model.parameters() if not p.requires_grad][:3]
        model.train();torch.cuda.reset_peak_memory_stats()
        def save():
            status('checkpointing',step=len(history))
            state={'optimizer':cpu_tree(optimizer.state_dict()),'rng':capture_rng()}
            adapter_digest=state_digest(get_peft_model_state_dict(model))
            meta={'history':history,'order_sha256':order_sha,'parameter_spec_sha256':spec_sha,
                'initial_adapter_sha256':initial_adapter_digest,'adapter_state_sha256':adapter_digest,
                'optimizer_state_sha256':state_digest(state['optimizer']),'rng_state_sha256':state_digest(state['rng'])}
            def writer(directory):
                model.save_pretrained(directory/'adapter',safe_serialization=True)
                torch.save(state,directory/'state.pt')
            pointer,manifest=commit_checkpoint(out,len(history),identity,meta,writer)
            restore_rng(state['rng'])
            event(out,'checkpoint_committed',attempt=attempt,step=len(history),manifest_sha256=pointer['manifest_sha256'])
            return pointer,manifest
        if not trainer:
            save()
        reason=None
        for cursor in range(len(history),len(order)):
            if stop_requested or time.monotonic()-started>=config['run']['max_attempt_seconds']:
                reason='signal' if stop_requested else 'attempt_time_budget';break
            row=encoded[order[cursor]]
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.monotonic()
            ids=torch.tensor([row['ids']],device='cuda');labels=torch.tensor([row['labels']],device='cuda')
            output=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False)
            loss=output.loss
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,settings['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            record={'step':cursor+1,**schedule[cursor],'loss':loss.item(),'grad_norm':norm.item(),
                'seconds':time.monotonic()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            history.append(record);del output,loss,ids,labels
            optimizer.zero_grad(set_to_none=True)
            atomic_json(out/'metrics.json',history)
            status('training',step=len(history),total=len(order),loss=record['loss'])
            if len(history)%config['run']['checkpoint_every']==0:
                save()
            if stop_after_step and len(history)==stop_after_step:
                reason='controlled_process_interruption';break
        if len(history)<len(order):
            if json.loads((out/'latest.json').read_text())['step']!=len(history):save()
            return _pause(out,history,reason or 'signal',status)
        if json.loads((out/'latest.json').read_text())['step']!=len(history):save()
        checkpoint,trainer,manifest=load_checkpoint(out,identity)
        if not all(p.grad is None for p in model.parameters() if not p.requires_grad):
            raise ValueError('Frozen parameter received a gradient')
        frozen_ok=all(torch.equal(p.detach().reshape(-1)[:8],before) for p,before in frozen)
        changed=trainer['adapter_state_sha256']!=initial_adapter_digest
        if not frozen_ok or not changed:raise ValueError('Parameter update integrity check failed')
        status('saved_adapter_probe',step=len(history))
        optimizer.zero_grad(set_to_none=True);del optimizer
        model.gradient_checkpointing_disable();runtime.model=model.eval()
        probe={'messages':rows[0]['messages'],'max_new_tokens':16,
               'agent_policy':{'format':settings.get('dialogue_format','legacy'),'context':'full'}}
        before=runtime.generate(probe)['ids']
        backbone=model.unload()
        runtime.model=PeftModel.from_pretrained(backbone,checkpoint/'adapter',is_trainable=False).eval()
        after=runtime.generate(probe)['ids']
        if before!=after:raise ValueError('Saved adapter changed greedy probe output')
        adapter_files={name.removeprefix('adapter/'):digest for name,digest in manifest['files'].items() if name.startswith('adapter/') and name.endswith(('.safetensors','.json'))}
        result={'schema_version':1,'status':'completed','finished_utc':utc(),'steps':len(history),
            'identity_sha256':identity_digest(identity),'trainable_parameters':count,'order_sha256':order_sha,
            'training_update_seconds':sum(r['seconds'] for r in history),
            'input_tokens':sum(r['tokens'] for r in history),'supervised_tokens':sum(r['supervised_tokens'] for r in history),
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in history),
            'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in history),
            'frozen_sample_unchanged':frozen_ok,'frozen_gradients_absent':True,'adapter_changed':changed,
            'saved_reload_greedy_ids_equal':True,'probe_ids_sha256':identity_digest(after),
            'adapter_state_sha256':trainer['adapter_state_sha256'],
            'optimizer_state_sha256':trainer['optimizer_state_sha256'],'rng_state_sha256':trainer['rng_state_sha256'],
            'adapter_files':adapter_files,'weights_public':False,'model_quality_accepted':False}
        atomic_json(out/'result.json',result);status('completed',step=len(history))
        event(out,'training_completed',attempt=attempt,step=len(history));return 0
    except BaseException as error:
        status('failed',error_type=type(error).__name__,observed_steps=len(history))
        event(out,'attempt_failed',attempt=attempt,error_type=type(error).__name__)
        atomic_json(out/'result.json',{'status':'failed','error_type':type(error).__name__,'observed_steps':len(history)})
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        attempts[-1].update(finished_utc=utc(),wall_seconds=time.monotonic()-started,
            status=json.loads((out/'status.json').read_text())['stage'])
        if monitor:attempts[-1]['telemetry']=monitor.stop()
        atomic_json(out/'attempts.json',attempts)


def _pause(out,history,reason,status):
    atomic_json(out/'result.json',{'schema_version':1,'status':'paused','steps':len(history),'reason':reason})
    status('paused',step=len(history),reason=reason)
    event(out,'training_paused',step=len(history),reason=reason)
    return 75


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-step',type=int,help='Controlled recovery test; exits 75 at this committed update')
    args=p.parse_args()
    raise SystemExit(execute(load_config(args.config),args.out,resume=args.resume,stop_after_step=args.stop_after_step))


if __name__=='__main__':
    main()
