"""Train one audited continuation candidate from the pinned base without opening held-out tasks."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
from tempfile import TemporaryDirectory
import time
import traceback

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, gpu_lock_path, identity_digest, single_writer, utc
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.adapter_bundle import FILES, verify_adapter
from one_gpu_lm.models.hf_runtime import Runtime
from one_gpu_lm.models.verify import verify_model
from .continuation_training_data import BASELINE, BASELINE_AUDIT, BASELINE_REVIEW, audited_continuations
from .continuation_training_data import load_baseline, mixture, read, record_shape, rows
from .runtime_identity import verify as verify_environment
from .tasks import digest, relative


KIND = 'first_edit_continuation_lora'


def json_bytes(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def materialize(root, continuations):
    from transformers import AutoTokenizer

    runtime = read(root/'private/parent-runtime.json')
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
    baseline = load_baseline(root,tokenizer)
    new_raw,new_encoded,evidence = audited_continuations(continuations,baseline,tokenizer)
    raw,batches,summary = mixture(baseline,new_raw,new_encoded,evidence)
    return runtime,baseline,raw,batches,summary


def freeze(root, continuations, inputs, run_directory, protocol_path, candidate):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',candidate):
        raise ValueError('Use a public experiment name without paths')
    inputs, protocol_path = inputs.resolve(),protocol_path.resolve()
    destination = relative(run_directory.as_posix())
    if destination.parts[0]!='runs' or (root/destination).exists() or protocol_path.exists():
        raise ValueError('Use a fresh training registration and run directory')
    if relative(inputs.relative_to(root).as_posix()).parts[0] != 'private':
        raise ValueError('Training examples belong in the private workspace')
    _,baseline,raw,_,summary = materialize(root,continuations)
    data = ''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in raw).encode()
    manifest = json_bytes(summary)
    origin = baseline.protocol
    paths = [root/relative(origin[name]) for name in ('profile_path','cycle_spec','sources_path','data_path',
                                                      'export_manifest','model_manifest_path','environment_path')]
    paths += [root/BASELINE,root/BASELINE_REVIEW,root/BASELINE_AUDIT,root/'private/parent-runtime.json']
    paths += [root/relative(name) for name in origin['correction_mix']['frozen_parent_files']]
    paths += list((root/'one_gpu_lm').rglob('*.py'))
    frozen = {str(path.relative_to(root)):sha256(path) for path in sorted(set(paths))}
    training_path = str((inputs/'data.jsonl').relative_to(root))
    manifest_path = str((inputs/'manifest.json').relative_to(root))
    frozen.update({training_path:digest(data),manifest_path:digest(manifest)})
    plan = {'schema_version':1,'candidate':candidate,'kind':KIND,'model_id':baseline.profile['model_id'],
            'revision':baseline.profile['revision'],'data_path':training_path,'data_manifest':manifest_path,
            'run_directory':str(destination),'profile_path':origin['profile_path'],'sources_path':origin['sources_path'],
            'model_manifest_path':origin['model_manifest_path'],'environment_path':origin['environment_path'],
            'expected_trainable_parameters':origin['expected_trainable_parameters'],
            'expected_optimizer_updates':summary['optimizer_updates'],'expected_records':len(raw),
            'first_edit_collection_sha256':summary['new_041']['protocol_sha256'],
            'data_mixture_sha256':identity_digest(summary),'frozen_files':frozen,
            'final_test_opened':False,'weights_public':False}
    if inputs.exists():
        if (inputs/'data.jsonl').read_bytes()!=data or (inputs/'manifest.json').read_bytes()!=manifest:
            raise FileExistsError('Preserve the first admitted training data')
    else:
        inputs.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with TemporaryDirectory(prefix='.train-data-',dir=inputs.parent) as tmp:
            stage = Path(tmp)/'candidate'
            stage.mkdir(mode=0o700)
            (stage/'data.jsonl').write_bytes(data)
            (stage/'manifest.json').write_bytes(manifest)
            if inputs.exists():
                raise FileExistsError('Training inputs appeared during admission')
            stage.rename(inputs)
    with protocol_path.open('xb') as stream:
        stream.write(json_bytes(plan))
    return {'protocol_sha256':sha256(protocol_path),'records':len(raw),'optimizer_updates':summary['optimizer_updates'],
            'new_041_records':summary['new_041']['records'],'final_test_opened':False}


def unchanged(root, files):
    return all((root/relative(name)).is_file() and sha256(root/relative(name))==checksum
               for name,checksum in files.items())


def registered(root, continuations, protocol_path, publication_path):
    plan = read(protocol_path)
    protocol_hash = sha256(protocol_path)
    publication = read(publication_path)
    if (plan['kind']!=KIND or plan['final_test_opened'] or plan['weights_public']
            or publication['protocol_sha256']!=protocol_hash or publication['anonymous_files_verified'] is not True
            or publication['ci_conclusion']!='success' or not re.fullmatch(r'[0-9a-f]{40}',publication['commit'])
            or not unchanged(root,plan['frozen_files'])):
        raise ValueError('A verified executable training preregistration is required')
    for key in ('profile_path','sources_path','model_manifest_path','environment_path','data_path','data_manifest'):
        if plan[key] not in plan['frozen_files']:
            raise ValueError('A registered training input is missing from the frozen inventory')
    runtime,baseline,raw,batches,summary = materialize(root,continuations)
    if (any(plan[key]!=baseline.protocol[key] for key in ('profile_path','sources_path','model_manifest_path',
                                                         'environment_path','expected_trainable_parameters'))
            or (plan['model_id'],plan['revision'])!=(baseline.profile['model_id'],baseline.profile['revision'])
            or raw!=rows(root/relative(plan['data_path'])) or summary!=read(root/relative(plan['data_manifest']))
            or plan['data_mixture_sha256']!=identity_digest(summary)
            or plan['first_edit_collection_sha256']!=summary['new_041']['protocol_sha256']
            or plan['expected_optimizer_updates']!=len(batches) or plan['expected_records']!=len(raw)
            or sha256(protocol_path)!=protocol_hash):
        raise ValueError('Matched training inputs differ from the approved archive')
    return plan,protocol_hash,runtime,baseline,raw,batches,summary


def train(root, plan, protocol_hash, runtime_info, baseline, raw, batches, summary):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    out = root/relative(plan['run_directory'])
    profile,recipe = baseline.profile,baseline.recipe
    with single_writer(gpu_lock_path()):
        if out.exists() or subprocess.check_output(
                ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip():
            raise RuntimeError('Require an idle GPU and the original unused candidate directory')
        (out/'private').mkdir(mode=0o700,parents=True)
        started = time.monotonic()
        result = {'schema_version':1,'stage':'loading','complete':False,'protocol_sha256':protocol_hash,
                  'profile_sha256':sha256(root/relative(plan['profile_path'])),'candidate_count':1,
                  'data_sha256':sha256(root/relative(plan['data_path'])),'started_epoch':time.time(),'started_utc':utc(),
                  'optimizer_steps':0,'action_records_consumed':0,'development_scores':0,'final_test_opened':False,
                  'weights_public':False}

        def save(stage):
            result['stage'] = stage
            atomic_json(out/'result.json',result)

        curve,updates = [],[]
        telemetry = Telemetry(out/'telemetry.csv',1)
        save('loading')
        telemetry.start()
        try:
            environment = root/relative(plan['environment_path'])
            atomic_json(out/'environment.json',verify_environment(read(environment)))
            model_path = Path(runtime_info['model'])
            atomic_json(out/'model-verification.json',verify_model(model_path,root/relative(plan['model_manifest_path']),profile))
            runtime = Runtime(model_path,profile,out)
            torch.manual_seed(recipe['seed'])
            model = get_peft_model(runtime.model,LoraConfig(task_type=TaskType.CAUSAL_LM,**profile['lora']))
            parameters = [p for p in model.parameters() if p.requires_grad]
            trainable = sum(p.numel() for p in parameters)
            if not parameters or trainable!=plan['expected_trainable_parameters']:
                raise ValueError('Unexpected retained LoRA capacity')
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            optimizer = torch.optim.AdamW(parameters,lr=recipe['learning_rate'],betas=tuple(recipe['adam_betas']),
                                          eps=recipe['adam_epsilon'],weight_decay=recipe['weight_decay'],foreach=False)
            frozen = [(p,p.detach().reshape(-1)[:8].clone()) for p in model.parameters() if not p.requires_grad][:3]
            original_adapter = [p.detach().clone() for p in parameters]
            atomic_json(out/'selection.json',{'mixture':summary,'records':[record_shape(r) for b in batches for r in b]})
            result['trainable_parameters'] = trainable
            model.train()
            torch.cuda.reset_peak_memory_stats()
            training_tick = time.monotonic()
            for step,batch in enumerate(batches,1):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                tick = time.monotonic()
                for row in batch:
                    if time.monotonic()-started >= recipe['maximum_training_pipeline_seconds']:
                        raise TimeoutError('Training exceeded its registered window')
                    micro = time.monotonic()
                    ids = torch.tensor([row['ids']],device='cuda')
                    labels = torch.tensor([row['labels']],device='cuda')
                    output = model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False)
                    loss = output.loss
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    (loss/len(batch)).backward()
                    torch.cuda.synchronize()
                    curve.append(dict(record_shape(row),record=len(curve)+1,update=step,
                                      loss=loss.item(),seconds=time.monotonic()-micro))
                    del loss,output,ids,labels
                norm = torch.nn.utils.clip_grad_norm_(parameters,recipe['max_grad_norm'],error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                updates.append({'step':step,'records':len(batch),'grad_norm':norm.item(),
                                'learning_rate':recipe['learning_rate'],'seconds':time.monotonic()-tick})
                result.update(optimizer_steps=step,action_records_consumed=len(curve))
                atomic_json(out/'curve.json',curve)
                atomic_json(out/'updates.json',updates)
                save('training')
                print(json.dumps({'step':step,'records_consumed':len(curve),'total_records':len(raw)}),flush=True)
            flags = {'frozen_sample_unchanged':all(torch.equal(p.detach().reshape(-1)[:8],v) for p,v in frozen),
                     'frozen_gradients_absent':all(p.grad is None for p in model.parameters() if not p.requires_grad),
                     'adapter_changed':any(not torch.equal(p.detach(),v) for p,v in zip(parameters,original_adapter)),
                     'all_selected_records_consumed':[r['id'] for r in curve]==[r['id'] for r in raw]}
            result.update(**flags,training_seconds=time.monotonic()-training_tick,
                          input_tokens=sum(r['tokens'] for r in curve),supervised_tokens=sum(r['supervised_tokens'] for r in curve),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if not all(flags.values()):
                raise ValueError('Frozen model or completed record integrity failed')
            del optimizer,original_adapter
            for parameter in parameters:
                parameter.grad = None
            model.gradient_checkpointing_disable()
            model.peft_config['default'].base_model_name_or_path = profile['model_id']
            model.peft_config['default'].revision = profile['revision']
            destination = out/'private/adapter'
            model.save_pretrained(destination,safe_serialization=True)
            atomic_json(out/'adapter.json',{'files':{p.name:sha256(p) for p in destination.iterdir() if p.is_file()},
                                           'final_test_opened':False,'weights_public':False})
            probe = {'messages':raw[0]['messages'],'max_new_tokens':16,
                     'agent_policy':{'format':'native_roundtrip','context':'bounded_history_v1'}}
            runtime.model = model.eval()
            before = runtime.generate(probe)['ids']
            runtime.model = PeftModel.from_pretrained(model.unload(),destination,is_trainable=False).eval()
            after = runtime.generate(probe)['ids']
            result.update(saved_reload_greedy_ids_equal=before==after,reload_probe_ids_sha256=identity_digest(after))
            if not before or before!=after:
                raise ValueError('Independent checkpoint reload changed the training-only probe')
            result['complete'] = True
            save('completed')
        except BaseException as error:
            result.update(complete=False,error_type=type(error).__name__)
            (out/'private/error.log').write_text(traceback.format_exc(),encoding='utf-8')
            atomic_json(out/'curve.json',curve)
            atomic_json(out/'updates.json',updates)
            save('failed')
            raise
        finally:
            verified = unchanged(root,plan['frozen_files'])
            result.update(telemetry=telemetry.stop(),finished_epoch=time.time(),finished_utc=utc(),frozen_files_unchanged=verified)
            if not verified:
                result.update(complete=False,stage='inputs_changed')
            save(result['stage'])
    return result


def replay_epoch(batches, summary, recipe, state, selection, curve, updates):
    expected = [record_shape(row) for batch in batches for row in batch]
    if (selection != {'mixture':summary,'records':expected} or not expected
            or [{key:row[key] for key in expected[0]} for row in curve]!=expected
            or [row['record'] for row in curve]!=list(range(1,len(expected)+1))
            or [row['update'] for row in curve]!=[step for step,batch in enumerate(batches,1) for _ in batch]
            or [row['step'] for row in updates]!=list(range(1,len(batches)+1))
            or [row['records'] for row in updates]!=list(map(len,batches))
            or any(row['learning_rate']!=recipe['learning_rate'] for row in updates)
            or not all(math.isfinite(row[name]) and row[name]>=0 for row in curve for name in ('loss','seconds'))
            or not all(math.isfinite(row[name]) and row[name]>=0 for row in updates for name in ('grad_norm','seconds'))
            or state['optimizer_steps']!=len(batches) or state['action_records_consumed']!=len(expected)
            or sum(r['supervised_tokens'] for r in curve)!=summary['supervised_tokens']
            or state['input_tokens']!=summary['input_tokens'] or state['supervised_tokens']!=summary['supervised_tokens']):
        raise ValueError('Recorded optimizer windows, targets or token totals differ from the frozen mix')


def audit(root,plan,protocol_hash,baseline,raw,batches,summary):
    out = root/relative(plan['run_directory'])
    if not (out/'result.json').is_file():
        raise ValueError('The registered training run has no completed records')
    with single_writer(out/'audit.lock'):
        return _audit_run(root,out,plan,protocol_hash,baseline,raw,batches,summary)


def _audit_run(root,out,plan,protocol_hash,baseline,raw,batches,summary):
    target = out/'integrity.json'
    state = read(out/'result.json')
    flags = ('complete','frozen_files_unchanged','all_selected_records_consumed','saved_reload_greedy_ids_equal',
             'frozen_sample_unchanged','frozen_gradients_absent','adapter_changed')
    if (state['stage']!='completed' or not all(state[key] is True for key in flags)
            or state['protocol_sha256']!=protocol_hash or state['final_test_opened'] or state['weights_public']
            or state['development_scores']!=0 or state['candidate_count']!=1
            or state['telemetry']['thread_stopped'] is not True or state['trainable_parameters']!=plan['expected_trainable_parameters']
            or state['profile_sha256']!=sha256(root/relative(plan['profile_path']))
            or state['data_sha256']!=sha256(root/relative(plan['data_path']))
            or not 0 < state['finished_epoch']-state['started_epoch'] <= baseline.recipe['maximum_training_pipeline_seconds']
            or read(out/'environment.json')!=read(root/relative(plan['environment_path']))
            or not unchanged(root,plan['frozen_files'])):
        raise ValueError('Only the complete training-only candidate can enter independent audit')
    replay_epoch(batches,summary,baseline.recipe,state,read(out/'selection.json'),read(out/'curve.json'),read(out/'updates.json'))
    adapter_record = read(out/'adapter.json')
    adapter = out/'private/adapter'
    paths = list(adapter.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError('The private candidate includes an unexpected adapter entry')
    files = {path.name:sha256(path) for path in paths}
    if (files!=adapter_record['files'] or adapter_record['final_test_opened'] or adapter_record['weights_public']):
        raise ValueError('Private candidate checkpoint differs from the completed training attempt')
    verified = verify_adapter(adapter,baseline.profile,{name:files[name] for name in FILES})
    if verified['parameters']!=plan['expected_trainable_parameters']:
        raise ValueError('Adapter CPU verification found an unexpected tensor count')
    existing = read(target) if target.is_file() else None
    evidence = {'schema_version':1,'passed':True,'protocol_sha256':protocol_hash,'adapter_files_sha256':files,
                'audited_utc':existing['audited_utc'] if existing else utc(),
                'optimizer_updates':len(batches),'mixture':summary,'adapter_cpu_verification':verified,
                'receipt_sha256':{name:sha256(out/name) for name in ('result.json','curve.json','updates.json',
                                                                  'selection.json','adapter.json','environment.json')},
                'audit_model_generations':0,'audit_optimizer_updates':0,'final_test_opened':False,'weights_public':False}
    counts = Counter(row['provenance']['repository'] for row in raw)
    added = Counter(row['provenance']['repository'] for row in raw
                    if row['provenance'].get('target_transform')=='verified_first_edit_action_only')
    if set(counts)!=set(baseline.review['sources']):
        raise ValueError('Actual training sources differ from the reviewed source inventory')
    by_source = {name:dict(value,used_records=counts[name],
                           verified_fragment_records=value['verified_fragment_records']+added[name])
                 for name,value in baseline.review['sources'].items()}
    if existing is not None and existing!=evidence:
        raise ValueError('The completed training audit changed on readback')
    if existing is None:
        atomic_json(target,evidence)
    review = dict(baseline.review,experiment=plan['candidate'],checked_utc=evidence['audited_utc'],sources=by_source,
                  actual_training_data_sha256=state['data_sha256'],training_integrity_sha256=sha256(target),
                  checkpoint_files_sha256=files,recorded_release_rights_complete=False)
    review_path = out/'candidate-input-review.json'
    if review_path.is_file():
        if read(review_path)!=review:
            raise ValueError('The reviewed input and exact checkpoint receipt changed')
    else:
        atomic_json(review_path,review)
    return evidence


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','continuations','protocol'):
        parser.add_argument('--'+key,type=Path,required=True)
    modes = parser.add_subparsers(dest='mode',required=True)
    prepare = modes.add_parser('freeze')
    prepare.add_argument('--inputs',type=Path,required=True)
    prepare.add_argument('--run-directory',type=Path,required=True)
    prepare.add_argument('--candidate',required=True)
    for mode in ('train','audit'):
        modes.add_parser(mode).add_argument('--publication',type=Path,required=True)
    a = parser.parse_args()
    root = a.root.resolve()
    if a.mode=='freeze':
        value = freeze(root,a.continuations.resolve(),a.inputs,a.run_directory,a.protocol,a.candidate)
    else:
        plan,plan_hash,runtime,baseline,raw,batches,summary = registered(root,a.continuations.resolve(),a.protocol,a.publication)
        if a.mode=='train':
            def interrupted(*_):
                raise KeyboardInterrupt('Candidate training interrupted')

            for kind in (signal.SIGTERM,signal.SIGINT):
                signal.signal(kind,interrupted)
            result = train(root,plan,plan_hash,runtime,baseline,raw,batches,summary)
            if result['complete'] is not True:
                raise ValueError('Training ended without eligible candidate evidence')
            value = {'complete':True,'optimizer_updates':result['optimizer_steps'],'final_test_opened':False}
        else:
            result = audit(root,plan,plan_hash,baseline,raw,batches,summary)
            value = {'passed':result['passed'],'optimizer_updates':result['optimizer_updates'],'final_test_opened':False}
    print(json.dumps(value))


if __name__=='__main__':
    main()
