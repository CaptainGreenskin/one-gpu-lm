"""Validate audited native-action pairs and score them with one model copy."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256 as hash_bytes
import json
import math
from pathlib import Path

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.training_encoding import encode_training_example


KINDS = {'early_edit', 'verified_finish'}


@dataclass(frozen=True)
class EncodedDecision:
    chosen: tuple[int, ...]
    rejected: tuple[int, ...]
    prefix_tokens: int
    kind: str


def token_digest(ids):
    return hash_bytes(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def encode_decision(row, tokenizer, profile):
    """Reproduce the exact pair accepted by the collection audit."""
    if row['kind'] not in KINDS or row['provenance']['split'] != 'train':
        raise ValueError('Expected an audited training-action pair')
    targets = [row['chosen'], row['rejected']]
    if parse_action(targets[0])['command'] == parse_action(targets[1])['command']:
        raise ValueError('Preference targets must represent different commands')
    encoded = [encode_training_example({'messages': row['messages'], 'target': target}, tokenizer, profile)
               for target in targets]
    prefix = encoded[0]['prefix_tokens']
    ids = [tuple(item['ids']) for item in encoded]
    proof = {'prefix_tokens': prefix, 'supervised_tokens': [item['supervised_tokens'] for item in encoded],
             'ids_sha256': [token_digest(sequence) for sequence in ids]}
    if (prefix <= 0 or encoded[1]['prefix_tokens'] != prefix or ids[0][:prefix] != ids[1][:prefix]
            or not all(prefix < len(seq) <= profile['training']['max_sequence_length'] for seq in ids)
            or ids[0] == ids[1] or proof != row['encoding_receipt']):
        raise ValueError('Action pair differs from the audited shared prefix or labels')
    return EncodedDecision(ids[0], ids[1], prefix, row['kind'])


def audited_records(run, expected_audit_sha256):
    """Accept the private export only after a complete registered collection audit."""
    run = Path(run)
    audit_file = run / 'audit.json'
    controller_file = run / 'controller.json'
    audit_bytes = audit_file.read_bytes()
    if hash_bytes(audit_bytes).hexdigest() != expected_audit_sha256:
        raise ValueError('The stopped collection audit differs from the reviewed digest')
    controller_bytes = controller_file.read_bytes()
    audit, controller = (json.loads(payload) for payload in (audit_bytes, controller_bytes))
    if (audit['passed'] is not True or audit['training_only'] is not True or controller['complete'] is not True
            or controller['backend_stopped'] is not True
            or audit['controller_sha256'] != hash_bytes(controller_bytes).hexdigest()
            or audit['protocol_sha256'] != controller['protocol_sha256']
            or audit['audited_attempts'] != audit['expected_attempts']
            or audit['audited_attempts'] != len(controller['runs'])
            or audit['optimizer_updates'] != 0 or audit['final_test_opened'] is not False
            or controller['final_test_opened'] is not False):
        raise ValueError('Training requires complete audited paired rollouts')
    if not audit['eligible_contrasts']:
        if audit['training_export_sha256'] is not None:
            raise ValueError('Unexpected private export with zero qualifying contrasts')
        return audit, []
    export = run / 'private' / 'preference-pairs.json'
    export_bytes = export.read_bytes()
    if hash_bytes(export_bytes).hexdigest() != audit['training_export_sha256']:
        raise ValueError('Private preference data differs from the audited export')
    rows = json.loads(export_bytes)
    counts = Counter(row['kind'] for row in rows)
    tasks = {row['provenance']['task_id'] for row in rows}
    if counts != audit['eligible_contrasts'] or len(tasks) != len(rows):
        raise ValueError('Preference export coverage differs from the audited contrast counts')
    return audit, rows


def load_decisions(run, expected_audit_sha256, tokenizer, profile):
    _, rows = audited_records(run, expected_audit_sha256)
    return [encode_decision(row, tokenizer, profile) for row in rows]


def action_log_probability(model, ids, prefix_tokens, device):
    """Score the full native action using only its next-token logit positions."""
    import torch
    from torch.nn.functional import cross_entropy

    if not 0 < prefix_tokens < len(ids):
        raise ValueError('Need a nonempty shared prompt and action suffix')
    sequence = torch.tensor([ids], device=device, dtype=torch.long)
    positions = torch.arange(prefix_tokens - 1, len(ids) - 1, device=device)
    logits = model(input_ids=sequence, attention_mask=torch.ones_like(sequence),
                   logits_to_keep=positions, use_cache=False, return_dict=True).logits
    targets = sequence[0, prefix_tokens:]
    if logits.ndim != 3 or logits.shape[:2] != (1, targets.numel()):
        raise ValueError('The registered model must project only requested action positions')
    return -cross_entropy(logits[0].float(), targets, reduction='sum')


def reference_margin(model, choice, device):
    """Compute the frozen-base margin by disabling LoRA on the existing model."""
    import torch

    with torch.no_grad(), model.disable_adapter():
        chosen = action_log_probability(model, choice.chosen, choice.prefix_tokens, device)
        rejected = action_log_probability(model, choice.rejected, choice.prefix_tokens, device)
    return float(chosen - rejected)


def preference_backward(model, choice, frozen_margin, beta, device, *, scale=1.0):
    """Accumulate the DPO gradient with one action's activations resident at a time."""
    import torch
    from torch.nn.functional import logsigmoid

    if not all(math.isfinite(value) for value in (frozen_margin, beta, scale)) or min(beta, scale) <= 0:
        raise ValueError('Need finite reference margin and positive preference scale')
    sequences = choice.chosen, choice.rejected
    with torch.no_grad():
        scores = [action_log_probability(model, ids, choice.prefix_tokens, device) for ids in sequences]
        margin = scores[0] - scores[1]
        delta = beta * (margin - frozen_margin)
        loss = -logsigmoid(delta)
        coefficient = beta * torch.sigmoid(-delta) * scale
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite preference objective')
    # The detached DPO derivative weights the two log-probability gradients.
    # Recompute deterministically and free each graph before running the other branch.
    for ids, expected, sign in zip(sequences, scores, (-1, 1)):
        score = action_log_probability(model, ids, choice.prefix_tokens, device)
        if not torch.isclose(score.detach(), expected, rtol=1e-5, atol=2e-3):
            raise ValueError('Log probabilities changed between the scoring and gradient passes')
        (sign * coefficient * score).backward()
        del score
    return {'loss': float(loss), 'policy_margin': float(margin), 'reference_margin': frozen_margin}
