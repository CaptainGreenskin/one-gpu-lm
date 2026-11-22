"""Train and resume one deterministic pass over audited action preferences."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

from one_gpu_lm.common.checkpoints import commit_checkpoint, load_checkpoint
from one_gpu_lm.common.runstate import atomic_json, event, identity_digest, single_writer
from .decision_preference import KINDS, preference_backward, reference_margin, token_digest
from .state import capture_rng, cpu_tree, restore_rng, state_digest


@dataclass(frozen=True)
class PreferenceRecipe:
    seed: int
    beta: float
    pairs_per_update: int
    learning_rate: float
    max_grad_norm: float

    def __post_init__(self):
        if (type(self.seed) is not int or self.seed < 0 or type(self.pairs_per_update) is not int
                or self.pairs_per_update < 1 or not all(math.isfinite(v) and v > 0 for v in (
                    self.beta, self.learning_rate, self.max_grad_norm))):
            raise ValueError('Need a deterministic seed and positive preference training settings')


def schedule(choices, recipe):
    """Record the exact pair order, including the two encoded labels of each choice."""
    if not choices:
        raise ValueError('No audited action preferences')
    order = list(range(len(choices)))
    random.Random(recipe.seed).shuffle(order)
    rows = []
    for ordinal, index in enumerate(order, 1):
        choice = choices[index]
        prefix = choice.prefix_tokens
        if (choice.kind not in KINDS or not 0 < prefix < min(len(choice.chosen), len(choice.rejected))
                or choice.chosen[:prefix] != choice.rejected[:prefix] or choice.chosen == choice.rejected):
            raise ValueError('Unexpected encoded preference pair')
        rows.append({'pair': ordinal, 'source_index': index, 'kind': choice.kind, 'prefix_tokens': prefix,
                     'chosen_tokens': len(choice.chosen) - prefix, 'rejected_tokens': len(choice.rejected) - prefix,
                     'chosen_sha256': token_digest(choice.chosen), 'rejected_sha256': token_digest(choice.rejected)})
    return [rows[i:i + recipe.pairs_per_update] for i in range(0, len(rows), recipe.pairs_per_update)]


def replay_history(batches, history):
    """Validate every committed optimizer window against the registered order."""
    if len(history) > len(batches):
        raise ValueError('More optimizer updates than registered windows')
    for step, (batch, update) in enumerate(zip(batches, history), 1):
        records = update['records']
        if (update['step'] != step or len(records) != len(batch)
                or not all(math.isfinite(update[key]) and update[key] >= 0 for key in ('grad_norm', 'seconds'))):
            raise ValueError('Invalid committed preference update')
        for expected, record in zip(batch, records):
            if (any(record[key] != value for key, value in expected.items())
                    or not all(math.isfinite(record[key]) for key in ('loss', 'policy_margin', 'reference_margin'))
                    or record['loss'] < 0):
                raise ValueError('A committed preference record differs from its encoded pair')


def restore_lora_checkpoint(model, optimizer, checkpoint, trainer):
    """Restore the adapter, optimizer and RNG and compare their recorded fingerprints."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    saved = load_file(checkpoint/'adapter/adapter_model.safetensors', device='cpu')
    current = get_peft_model_state_dict(model)
    if set(saved) != set(current) or any(
            saved[key].shape != current[key].shape or saved[key].dtype != current[key].dtype for key in saved):
        raise ValueError('Saved adapter keys, shapes or dtypes differ')
    outcome = set_peft_model_state_dict(model, saved)
    if outcome.unexpected_keys:
        raise ValueError('Unexpected adapter keys')
    state = torch.load(checkpoint/'state.pt', map_location='cpu', weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    checks = {'adapter': state_digest(get_peft_model_state_dict(model)) == trainer['adapter_state_sha256'],
              'optimizer': state_digest(optimizer.state_dict()) == trainer['optimizer_state_sha256'],
              'rng': state_digest(capture_rng()) == trainer['rng_state_sha256']}
    if not all(checks.values()):
        raise ValueError('Restored training state differs from checkpoint')
    return checks


def run_epoch(model, choices, recipe, out, identity, *, device, resume=False, interrupted=None,
              expected_trainable_parameters=None):
    """Update a prepared LoRA model and commit its optimizer state after each window.

    The caller supplies a freshly seeded adapter and holds the device lock when
    using CUDA. Checkpoint zero also makes interruption before the first update
    recoverable. The latest pointer is the source of truth on restart.
    """
    import torch
    from peft import get_peft_model_state_dict

    out = Path(out).resolve()
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    batches = schedule(choices, recipe)
    named = [(name, weight) for name, weight in model.named_parameters() if weight.requires_grad]
    parameters = [weight for _, weight in named]
    count = sum(weight.numel() for weight in parameters)
    if (set(model.peft_config) != {'default'} or not count
            or any(not any(f'.{part}.' in name for part in ('lora_A', 'lora_B')) for name, _ in named)
            or expected_trainable_parameters is not None and count != expected_trainable_parameters
            or any(config.lora_dropout != 0 or config.bias != 'none' for config in model.peft_config.values())):
        raise ValueError('Expected only the registered deterministic LoRA parameters')
    parameter_spec = identity_digest([(name, list(weight.shape), str(weight.dtype)) for name, weight in named])
    order_sha = identity_digest(batches)
    use_cuda = torch.device(device).type == 'cuda'
    bound_identity = {'inputs': identity, 'recipe': vars(recipe), 'ordered_pairs_sha256': order_sha,
                      'parameter_spec_sha256': parameter_spec, 'device_type': torch.device(device).type}
    identity_path = out/'preference-identity.json'
    selection_path = out/'selection.json'
    selection = {'pairs': len(choices), 'updates': len(batches), 'order_sha256': order_sha, 'batches': batches,
                 'final_test_opened': False}

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize(device)

    with single_writer(out/'private/preference-epoch.lock'):
        if identity_path.exists():
            if not resume or json.loads(identity_path.read_bytes()) != bound_identity:
                raise ValueError('Resume requires the same frozen preference training identity')
        elif resume and (out/'latest.json').exists():
            raise ValueError('Existing checkpoint lacks its original training identity')
        else:
            atomic_json(identity_path, bound_identity)
        if selection_path.exists():
            if json.loads(selection_path.read_bytes()) != selection:
                raise ValueError('Preference schedule changed')
        else:
            atomic_json(selection_path, selection)

        optimizer = torch.optim.AdamW(parameters, lr=recipe.learning_rate, weight_decay=0.0, foreach=False)
        initial = state_digest(get_peft_model_state_dict(model))
        frozen = [(weight, weight.detach().reshape(-1)[:8].clone())
                  for weight in [p for p in model.parameters() if not p.requires_grad][:3]]
        model.train()
        history = []
        restored = None
        if (out/'latest.json').exists():
            if not resume:
                raise ValueError('An existing preference checkpoint requires resume')
            checkpoint, trainer, _ = load_checkpoint(out, bound_identity)
            history = trainer['history']
            if trainer['step'] != len(history) or trainer['initial_adapter_sha256'] != initial:
                raise ValueError('Checkpoint step or fresh adapter initialization differs')
            replay_history(batches, history)
            restored = restore_lora_checkpoint(model, optimizer, checkpoint, trainer)
            event(out, 'preference_state_restored', step=len(history), checks=restored)

        def save():
            state = {'optimizer': cpu_tree(optimizer.state_dict()), 'rng': capture_rng()}
            metadata = {'history': history, 'initial_adapter_sha256': initial,
                        'adapter_state_sha256': state_digest(get_peft_model_state_dict(model)),
                        'optimizer_state_sha256': state_digest(state['optimizer']),
                        'rng_state_sha256': state_digest(state['rng'])}

            def writer(path):
                model.save_pretrained(path/'adapter', safe_serialization=True, save_embedding_layers=False)
                torch.save(state, path/'state.pt')

            pointer, _ = commit_checkpoint(out, len(history), bound_identity, metadata, writer)
            restore_rng(state['rng'])
            event(out, 'preference_update_committed', step=len(history), manifest_sha256=pointer['manifest_sha256'])
            atomic_json(out/'curve.json', history)

        if not (out/'latest.json').exists():
            save()
        else:
            atomic_json(out/'curve.json', history)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        for batch in batches[len(history):]:
            optimizer.zero_grad(set_to_none=True)
            synchronize()
            tick = time.monotonic()
            records = []
            for row in batch:
                if interrupted is not None and interrupted():
                    break
                choice = choices[row['source_index']]
                frozen_margin = reference_margin(model, choice, device)
                measured = preference_backward(model, choice, frozen_margin, recipe.beta, device,
                                               scale=1 / len(batch))
                records.append({**row, **measured})
            if len(records) != len(batch):
                optimizer.zero_grad(set_to_none=True)
                break
            norm = torch.nn.utils.clip_grad_norm_(parameters, recipe.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            synchronize()
            optimizer.zero_grad(set_to_none=True)
            history.append({'step': len(history) + 1, 'records': records,
                            'grad_norm': float(norm), 'seconds': time.monotonic() - tick,
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if use_cuda else 0,
                            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if use_cuda else 0})
            save()

        _, trainer, _ = load_checkpoint(out, bound_identity)
        if trainer['step'] != len(history) or trainer['history'] != history:
            raise ValueError('Committed preference history differs')
        complete = len(history) == len(batches)
        frozen_ok = (all(weight.grad is None for weight in model.parameters() if not weight.requires_grad)
                     and all(torch.equal(weight.detach().reshape(-1)[:8], before) for weight, before in frozen))
        adapter_changed = trainer['adapter_state_sha256'] != initial
        if not frozen_ok or complete and not adapter_changed:
            raise ValueError('Frozen backbone or trained adapter integrity check failed')
        return {'complete': complete, 'optimizer_updates': len(history), 'planned_updates': len(batches),
                'pairs_consumed': sum(len(row['records']) for row in history), 'planned_pairs': len(choices),
                'adapter_state_sha256': trainer['adapter_state_sha256'],
                'optimizer_state_sha256': trainer['optimizer_state_sha256'],
                'rng_state_sha256': trainer['rng_state_sha256'], 'order_sha256': order_sha,
                'restored_state': restored, 'trainable_parameters': count,
                'frozen_parameters_unchanged': frozen_ok, 'adapter_changed': adapter_changed,
                'final_test_opened': False}
