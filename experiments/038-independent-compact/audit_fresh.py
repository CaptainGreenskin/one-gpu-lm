"""Reconstruct and audit every registered 038 rollout before exporting measurements."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_tasks import prepare
from one_gpu_lm.verified.rollout import normalization

from study import ARMS, OUT, inputs, load, read, variants


def analyze(rows, gate, size):
    pairs = defaultdict(dict)
    for row in rows:
        number, arm = row['task_number'], row['arm']
        if not row['audit_passed'] or arm not in ARMS or arm in pairs[number]:
            raise ValueError('Duplicate or unaudited result')
        pairs[number][arm] = row
    if set(pairs) != set(range(1, size + 1)) or any(set(pair) != set(ARMS) for pair in pairs.values()):
        raise ValueError('Every planned pair is required')
    stats = Counter()
    family_outcomes = defaultdict(Counter)
    subsets = ('old_prefix_clipped', 'recovered_failure_line', 'fallback')
    for pair in pairs.values():
        old, new = (pair[arm] for arm in ARMS)
        if any(old[k] != new[k] for k in ('family', 'multifile', *subsets)):
            raise ValueError('Task metadata differs between arms')
        delta = int(new['completed_repair']) - int(old['completed_repair'])
        family = family_outcomes[old['family']]
        for counter in (stats, family):
            counter['improvements'] += delta > 0
            counter['regressions'] += delta < 0
            counter['net_repairs'] += delta
        family['pairs'] += 1
        stats['new_edits'] += old['no_edit'] and not new['no_edit']
        stats['lost_edits'] += new['no_edit'] and not old['no_edit']
        for key in subsets:
            stats[f'{key}_pairs'] += old[key]
            for arm, row in pair.items():
                stats[f'{key}_repairs_{arm}'] += old[key] and row['completed_repair']
    passed = (stats['net_repairs'] >= gate['minimum_net_repairs']
              and sum(f['net_repairs'] > 0 for f in family_outcomes.values()) >= gate['minimum_positive_families']
              and stats['regressions'] <= gate['maximum_regressions'])
    return {'paired_tasks':size, **stats, 'by_family':{k:dict(v) for k, v in sorted(family_outcomes.items())},
            'followup_gate_passed':passed}


def run(root):
    _, context, proof, prior = inputs(root)
    out = root / OUT
    destination = out / 'audit.json'
    if destination.exists():
        raise ValueError('Preserve the completed audit')
    controller = out / 'controller.json'
    state = read(controller)
    if (not state['complete'] or not state['backend_stopped'] or not state['frozen_inputs_unchanged']
            or state['final_test_opened'] or state['optimizer_updates'] != 0
            or state['registration_sha256'] != sha256(context.plan_path)
            or state['schedule_sha256'] != context.plan['schedule_sha256']
            or [r['run_id'] for r in state['runs']] != [r['run_id'] for r in context.order]):
        raise ValueError('Complete registered rollout schedule required')
    backends = state['backend_lifecycles']
    if len(backends) != 1 or backends[0]['role'] != 'base' or backends[0]['index'] != 1:
        raise ValueError('One common verified model load required')
    backend = backends[0]
    location = out / 'private/backends/0001-base'
    served = read(location / 'served-artifact.json')
    if (served['adapter'] or served['adapter_sha256']
            or served['profile_sha256'] != context.protocol['evaluation_profile_sha256']
            or read(location.with_name('0001-base-environment.json')) != read(root / context.protocol['environment_path'])):
        raise ValueError('Served artifact or runtime differs')
    previous = load('historic035_audit', root / 'experiments/035-edit-bottleneck/audit.py')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(context.runtime['model'], local_files_only=True, trust_remote_code=False)
    profile = read(root / context.protocol['evaluation_profile_path'])
    norm = normalization(root)
    journal = RequestJournal(location / 'requests')
    prompts = variants(root, context, proof, prior)
    receipts = {r['task_number']:r for r in proof['rows']}
    admission = {r['id']:r for r in read(context.data / 'admission.json')['rows']}
    seen = set()
    rows = []
    starts = defaultdict(set)
    original = {}
    with tempfile.TemporaryDirectory(prefix='audit-', dir=out / 'private') as temp:
        reconstruction = Path(temp)
        for item, result in zip(context.order, state['runs']):
            number, arm, run_id = (item[k] for k in ('task_number', 'arm', 'run_id'))
            task = prompts[run_id]
            source = context.sources[task['source_id']]
            if (result['backend_index'] != 1 or result['id'] != task['id']
                    or result['task_number'] != number or result['contrast_arm'] != arm):
                raise ValueError('Scored result differs from schedule')
            directory = out / 'private/tasks' / run_id
            trace = read(directory / 'trajectory.json')
            if trace['backend'] != served:
                raise ValueError('Trajectory and backend differ')
            admitted = admission[task['id']]
            if number not in original:
                snapshot = context.data / 'snapshots' / task['source_id']
                original[number] = prepare(snapshot, reconstruction / f'original-{number:02d}', source, task)
                if fingerprint(sorted(original[number].items())) != admitted['workspace_sha256']:
                    raise ValueError('Actor input differs from the original public check workspace')
            row, first, requests = previous.audited_row(
                root, context.data, directory, result, task, source, context.protocol, profile, tokenizer, norm,
                journal, reconstruction / run_id, original[number], backend)
            receipt = receipts[number]
            if (first != receipt['opening_token_sha256'][arm]
                    or trace['steps'][0]['generation']['input_tokens'] != receipt['opening_tokens'][arm]
                    or seen & requests):
                raise ValueError('Opening prompt or physical generation receipts differ')
            seen.update(requests)
            starts[number].add(first)
            row.update(multifile=admitted['multifile'], old_prefix_clipped=receipt['raw_prefix_clipped'],
                       recovered_failure_line=receipt['selected_missing_failure_visible'] > 0,
                       fallback=receipt['fallback'], packet_chars=receipt[arm + '_chars'],
                       opening_input_tokens=receipt['opening_tokens'][arm])
            rows.append(row)
    if seen != {path.name for path in journal.directory.iterdir() if path.is_dir()}:
        raise ValueError('Physical generation inventory differs')
    changed = sum(len(tokens) == 2 for tokens in starts.values())
    if changed != len(context.tasks) - context.plan['preflight']['fallback']:
        raise ValueError('Each extracted pair must record a changed opening')
    summary = dict(schema_version=1, passed=True, optimizer_updates=0, final_test_opened=False,
                   registration_sha256=sha256(context.plan_path), controller_sha256=sha256(controller),
                   verified_requests=len(seen), opening_change_pairs=changed, backend_loads=len(backends),
                   started_epoch=state['started_epoch'], finished_epoch=state['finished_epoch'],
                   telemetry=state['telemetry'], preflight=context.plan['preflight'],
                   analysis=analyze(rows, context.plan['followup_gate'], len(context.tasks)),
                   arms=previous.totals(rows), rows=rows)
    write_json(destination, summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    result = run(parser.parse_args().root.resolve())
    print(json.dumps({'passed':result['passed'],'verified_requests':result['verified_requests'],
                      'opening_change_pairs':result['opening_change_pairs']}))
