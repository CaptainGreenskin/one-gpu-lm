"""One preregistered assistant-only LoRA pass; raw trajectories and adapters stay private."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import time

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.models.hf_runtime import Runtime


def select_examples(rows, tokenizer, profile, approved_ids, max_examples=96):
    selected, excluded = [], []
    for row in rows:
        if row['provenance']['split'] != 'train' or row['provenance']['task_id'] not in approved_ids:
            raise ValueError('Non-training or unadmitted supervision')
        if not row['target'].strip() or not row['messages'] or row['messages'][-1]['role'] != 'user':
            raise ValueError('Invalid next-assistant training record')
        prefix = tokenizer.apply_chat_template(row['messages'], tokenize=True, return_dict=False,
            add_generation_prompt=True, **profile['chat_template_kwargs'])
        suffix = tokenizer.encode(row['target']+tokenizer.eos_token, add_special_tokens=False)
        if len(prefix)+len(suffix) > profile['training']['max_sequence_length']:
            excluded.append({'id':row['id'],'reason':'context_limit','tokens':len(prefix)+len(suffix)})
            continue
        if len(selected) >= max_examples:
            excluded.append({'id':row['id'],'reason':'example_budget','tokens':len(prefix)+len(suffix)})
            continue
        selected.append({'id':row['id'], 'ids':prefix+suffix, 'labels':[-100]*len(prefix)+suffix,
                         'provenance':row['provenance'], 'supervised_tokens':len(suffix)})
    if not selected:
        raise ValueError('No valid training examples')
    return selected, excluded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model','profile','protocol','admission','data','out']:
        p.add_argument('--'+name, required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Do not overwrite a candidate attempt')
    args.out.mkdir(parents=True)
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    profile = json.loads(args.profile.read_text())
    protocol = json.loads(args.protocol.read_text())
    training = profile['training']
    frozen = protocol['training_candidate']
    if training['epochs'] != 1 or training['learning_rate'] != frozen['learning_rate'] or training['max_sequence_length'] != frozen['max_training_sequence_tokens']:
        raise ValueError('Training profile differs from frozen protocol')
    torch.manual_seed(training['seed'])
    random.seed(training['seed'])
    result = {'schema_version':1,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'starting',
              'profile_sha256':sha256(args.profile),'protocol_sha256':sha256(args.protocol),
              'data_sha256':sha256(args.data),'training_scope':'assistant-only-constructed-repository-control'}
    def stage(name, **extra):
        write_json(args.out/'status.json', {'stage':name,**extra})
        write_json(args.out/'result.json', result)
        print(name,json.dumps(extra),flush=True)
    try:
        stage('loading_base')
        runtime = Runtime(args.model,profile,args.out)
        model, tokenizer = runtime.model, runtime.tokenizer
        admitted = json.loads(args.admission.read_text())
        if not admitted['complete'] or sha256(args.admission) != protocol['admission_sha256']:
            raise ValueError('Frozen admission mismatch')
        allowed = {r['id'] for r in admitted['tasks'] if r['admitted'] and r['split']=='train'}
        rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
        encoded, excluded = select_examples(rows,tokenizer,profile,allowed,frozen['maximum_training_examples'])
        write_json(args.out/'selection.json', {'selected':[{'id':r['id'],'tokens':len(r['ids']),
            'supervised_tokens':r['supervised_tokens'],'kind':r['provenance']['kind']} for r in encoded],
            'excluded':excluded, 'kinds':dict(Counter(r['provenance']['kind'] for r in encoded)),
            'repositories':sorted({r['provenance']['repository'] for r in encoded}), 'no_dev_test_supervision':True})
        model = get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,**profile['lora']))
        parameters = [p for p in model.parameters() if p.requires_grad]
        result['trainable_parameters'] = sum(p.numel() for p in parameters)
        if not parameters or result['trainable_parameters'] > 100_000_000:
            raise ValueError('Unexpected trainable parameter count')
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer = torch.optim.AdamW(parameters,lr=training['learning_rate'],weight_decay=0.0)
        frozen_samples = [(p,p.detach().reshape(-1)[:8].clone()) for p in model.parameters() if not p.requires_grad][:3]
        adapter_before = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        model.train()
        curve = []
        order = list(range(len(encoded)))
        random.Random(training['seed']).shuffle(order)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        for index in order:
            row = encoded[index]
            optimizer.zero_grad(set_to_none=True)
            ids = torch.tensor([row['ids']],device='cuda')
            labels = torch.tensor([row['labels']],device='cuda')
            torch.cuda.synchronize()
            tick = time.monotonic()
            output = model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False)
            loss = output.loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters,training['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            curve.append({'step':len(curve)+1,'id':row['id'],'tokens':len(row['ids']),
                'supervised_tokens':row['supervised_tokens'],'loss':loss.item(),'grad_norm':norm.item(),
                'seconds':time.monotonic()-tick})
            write_json(args.out/'curve.json',curve)
            stage('training',step=len(curve),total=len(encoded),loss=loss.item())
            del output, loss, ids, labels
        result.update(training_seconds=time.monotonic()-started,steps=len(curve),
            input_tokens=sum(r['tokens'] for r in curve),supervised_tokens=sum(r['supervised_tokens'] for r in curve),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            frozen_sample_unchanged=all(torch.equal(p.detach().reshape(-1)[:8],v) for p,v in frozen_samples),
            frozen_gradients_absent=all(p.grad is None for p in model.parameters() if not p.requires_grad),
            adapter_changed=any(not torch.equal(p.detach(),adapter_before[n]) for n,p in model.named_parameters() if p.requires_grad))
        if not all(result[k] for k in ['frozen_sample_unchanged','frozen_gradients_absent','adapter_changed']):
            raise ValueError('Parameter update check failed')
        del optimizer, adapter_before
        for p in parameters:
            p.grad = None
        model.gradient_checkpointing_disable()
        model.peft_config['default'].base_model_name_or_path = profile['model_id']
        model.peft_config['default'].revision = profile['revision']
        adapter = args.out/'private/adapter'
        model.save_pretrained(adapter,safe_serialization=True)
        write_json(args.out/'adapter.json',{'base_model':profile['model_id'],'revision':profile['revision'],
            'weights_public':False,'files':{p.name:sha256(p) for p in adapter.glob('*.safetensors')}})
        runtime.model = model.eval()
        # This is an integrity probe on a training context, not a heldout quality score.
        probe = rows[0]['messages']
        before = runtime.generate({'messages':probe,'max_new_tokens':16})['ids']
        base = model.unload()
        runtime.model = PeftModel.from_pretrained(base,adapter,is_trainable=False).eval()
        after = runtime.generate({'messages':probe,'max_new_tokens':16})['ids']
        result['saved_reload_greedy_ids_equal'] = before == after
        result['reload_probe_ids_sha256'] = hashlib.sha256(json.dumps(after).encode()).hexdigest()
        if before != after:
            raise ValueError('Saved adapter changed after reload')
        gc.collect()
        torch.cuda.empty_cache()
        result.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
        stage('completed',steps=len(curve))
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__)
        stage('failed',error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    main()
