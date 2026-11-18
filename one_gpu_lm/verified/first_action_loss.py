"""Audit which 032 records can have their target loss zero-weighted in 034."""
import hashlib
import json
import re

from one_gpu_lm.common.actions import parse_action


def fingerprint(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def first_regression_targets(rows):
    """Return only the newly added, executed, exact first-regression targets."""
    found = {}
    seen = set()
    for row in rows:
        p = row['provenance']
        record_id = row['id']
        if record_id in seen or p['split'] != 'train' or p['executed'] is not True:
            raise ValueError('Duplicate or unverified training record')
        seen.add(record_id)
        if p['producer_kind'] != 'upstream_reference_guided' or p['kind'] != 'baseline_regression':
            continue
        messages = row['messages']
        if (p['call'] != 1 or [m['role'] for m in messages] != ['system', 'user']
                or p['model_generated'] is not False):
            raise ValueError('New baseline regression target is not the first action')
        declared = re.findall(r'^Regression command: (.+)$', messages[1]['content'], re.M)
        if len(declared) != 1 or parse_action(row['target'])['command'] != declared[0]:
            raise ValueError('New baseline regression target differs from the declared command')
        found[record_id] = row['encoding_receipt']['supervised_tokens']
    return found


def target_receipt(rows, ordered):
    targets = first_regression_targets(rows)
    if len(rows) != len(ordered) or {r['id'] for r in ordered} != {r['id'] for r in rows}:
        raise ValueError('Ablation changed the training records')
    counts = {r['id']: r['supervised_tokens'] for r in ordered if r['id'] in targets}
    if counts != targets or any(n < 1 for n in counts.values()):
        raise ValueError('Target token counts differ from the frozen export')
    return targets, {
        'records': len(ordered),
        'target_records': len(targets),
        'target_supervised_tokens': sum(targets.values()),
        'target_ids_sha256': fingerprint(sorted(targets)),
        'ordered_records_sha256': fingerprint(
            (r['id'], r['episode_id'], r['supervised_tokens']) for r in ordered),
        'supervised_tokens': sum(r['supervised_tokens'] for r in ordered),
    }


def plan(rows, selected, ablation):
    targets, receipt = target_receipt(rows, selected)
    if receipt != ablation['target_receipt']:
        raise ValueError('Ablation targets, counts or order differ from the registration')
    weight = ablation['target_loss_weight']
    if type(weight) is not int or weight not in (0, 1):
        raise ValueError('Unsupported loss weight')
    masked = frozenset(targets) if weight == 0 else frozenset()
    return masked, {
        **receipt,
        'encoding_sha256': fingerprint((r['id'], r['ids'], r['labels']) for r in selected),
        'masked_records': len(masked),
        'effective_supervised_tokens': receipt['supervised_tokens'] - (1 - weight) * receipt['target_supervised_tokens'],
        'target_loss_weight': weight,
    }


def initial_adapter_sha256(named_parameters, torch):
    digest = hashlib.sha256()
    for name, tensor in named_parameters:
        digest.update(name.encode() + b'\0')
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()
