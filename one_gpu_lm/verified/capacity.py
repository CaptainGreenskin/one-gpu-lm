"""One registered 8192-token LoRA backward capacity probe, with zero optimizer steps.

Synthetic tokens measure memory shape only. They are never teacher data or a
quality score. A launcher must also enforce the protocol's wall-clock deadline.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import traceback
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.hf_runtime import Runtime
from one_gpu_lm.models.verify import verify_model
from .tasks import digest, relative


def validate_protocol(protocol):
    expected = {'purpose': 'synthetic_backward_capacity_only', 'optimizer_steps': 0,
                'sequence_tokens': 8192, 'batch_size': 1, 'supervised_suffix_tokens': 512}
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected capacity probe scope')
    if not 60 <= protocol['maximum_seconds'] <= 1200:
        raise ValueError('Capacity wall-clock cap must be bounded')
    if protocol['minimum_reserved_headroom_bytes'] < 8 * 1024 ** 3:
        raise ValueError('Capacity must retain at least 8 GiB of reserved-memory headroom')
    relative(protocol['run_directory']); relative(protocol['profile_path'])


def run(root, protocol_path):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    for name, expected in protocol['frozen_files'].items():
        if sha256(root/relative(name)) != expected:
            raise ValueError('Capacity input differs: '+name)
    previous = json.loads((root/'runs/teacher-calibration-001/controller.json').read_text())
    audit_path = root/'runs/teacher-calibration-001/audit.json'
    audit = json.loads(audit_path.read_text())
    if not previous['complete'] or not previous['backend_stopped'] or not audit['decision']['passed']:
        raise ValueError('Finish and pass the data collection calibration first')
    if sha256(audit_path) != protocol['calibration_audit_sha256']:
        raise ValueError('Calibration audit differs')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise ValueError('GPU occupied')
    out = root/relative(protocol['run_directory'])
    if out.exists():
        raise ValueError('Preserve previous capacity attempt')
    publication = json.loads((root/'private/capacity-preregistration.json').read_text())
    if (publication['anonymous_files_verified'] is not True or publication['ci_conclusion'] != 'success'
            or publication['protocol_sha256'] != sha256(protocol_path)):
        raise ValueError('Capacity protocol must be public and verified before compute')
    out.mkdir(parents=True); (out/'private').mkdir()
    result = {'stage': 'starting', 'complete': False, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'started_epoch': time.time(), 'protocol_sha256': sha256(protocol_path),
              'optimizer_steps': 0, 'teacher_episodes_created': 0, 'new_development_scores': 0,
              'final_test_opened': False, 'weights_saved': False, 'synthetic_tokens_are_training_data': False}
    def save():
        write_json(out/'result.json', result)
    save(); telemetry = Telemetry(out/'telemetry.csv', 1); telemetry.start()
    try:
        import torch
        from importlib.metadata import version
        from peft import LoraConfig, TaskType, get_peft_model
        profile_path = root/relative(protocol['profile_path'])
        profile = json.loads(profile_path.read_text())
        runtime_paths = json.loads((root/'private/parent-runtime.json').read_text())
        result['stage'] = 'verifying_model'; save()
        verification = verify_model(Path(runtime_paths['model']), root/'experiments/011-coding-model-selection/weights-manifest.json', profile)
        write_json(out/'model-verification.json', verification)
        result['stage'] = 'loading_model'; save()
        runtime = Runtime(Path(runtime_paths['model']), profile, out/'private')
        torch.manual_seed(profile['training']['seed'])
        model = get_peft_model(runtime.model, LoraConfig(task_type=TaskType.CAUSAL_LM, **profile['lora']))
        parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        count = sum(p.numel() for _,p in parameters)
        if not parameters or count > 100_000_000:
            raise ValueError('Unexpected trainable parameter count')
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train()
        before = {n:p.detach().clone() for n,p in parameters}
        frozen = [(p,p.detach().reshape(-1)[:8].clone()) for p in model.parameters() if not p.requires_grad][:3]
        # AdamW initializes these lazily. Reserve both moment shapes without an
        # optimizer or an update, to avoid testing only an unrealistically empty state.
        moments = [torch.zeros_like(p) for _,p in parameters for _ in range(2)]
        unit = runtime.tokenizer.encode('Read the function, repair its behavior, and run the regression check.\n', add_special_tokens=False)
        if not unit:
            raise ValueError('Empty capacity fixture')
        size = protocol['sequence_tokens']
        tokens = (unit*((size+len(unit)-1)//len(unit)))[:size]
        ids = torch.tensor([tokens], device='cuda')
        labels = ids.clone(); labels[:, :-protocol['supervised_suffix_tokens']] = -100
        result.update(stage='backward_probe', trainable_parameters=count, sequence_tokens=size, batch_size=1,
                      runtime_versions={'torch':str(torch.__version__),'cuda':torch.version.cuda,
                                        'transformers':version('transformers'),'peft':version('peft')},
                      supervised_suffix_tokens=protocol['supervised_suffix_tokens'],
                      synthetic_input_sha256=digest(json.dumps(tokens,separators=(',',':')).encode()),
                      simulated_moment_bytes=sum(p.numel()*p.element_size() for p in moments))
        save(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); tick = time.monotonic()
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
        loss = output.loss
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite probe loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for _,p in parameters], profile['training']['max_grad_norm'], error_if_nonfinite=True)
        torch.cuda.synchronize()
        result.update(backward_seconds=time.monotonic()-tick, probe_loss=float(loss), gradient_norm=float(norm),
                      adapter_parameters_unchanged=all(torch.equal(p.detach(),before[n]) for n,p in parameters),
                      frozen_parameter_samples_unchanged=all(torch.equal(p.detach().reshape(-1)[:8],v) for p,v in frozen),
                      frozen_gradients_absent=all(p.grad is None for p in model.parameters() if not p.requires_grad),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      device_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        result['reserved_headroom_bytes'] = result['device_total_bytes'] - result['peak_reserved_bytes']
        result['capacity_gate_passed'] = (result['reserved_headroom_bytes'] >= protocol['minimum_reserved_headroom_bytes']
            and result['adapter_parameters_unchanged'] and result['frozen_parameter_samples_unchanged']
            and result['frozen_gradients_absent'] and result['gradient_norm'] > 0)
        result.update(stage='completed', complete=True,
                      limitation='One synthetic worst-length backward with simulated AdamW moments, not a full training run. '
                                 'Headroom uses PyTorch reserved memory; non-PyTorch allocation and future fragmentation remain outside this metric.')
    except BaseException as error:
        result.update(stage='failed', error_type=type(error).__name__, capacity_gate_passed=False)
        (out/'private/error.log').write_text(traceback.format_exc())
    finally:
        result['telemetry'] = telemetry.stop()
        result.update(finished_epoch=time.time(), finished_utc=datetime.now(timezone.utc).isoformat(),
                      frozen_files_unchanged=all(sha256(root/relative(n))==h for n,h in protocol['frozen_files'].items()))
        save()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    a = p.parse_args(); value = run(a.root, a.protocol)
    print(json.dumps({k:value[k] for k in ['stage','complete','capacity_gate_passed','optimizer_steps']}))
