"""Combine disjoint physical first-action collections after complete publication and audit."""

from collections import Counter
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import re
import runpy

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.training.decision_preference import audited_records, encode_decision
from .decision_training_data import (
    DecisionTrainingData, FORK_ARMS, FORK_PATH, admitted_sources, verify_first_choices, verify_private_labels,
)
from .tasks import relative


EXPANSION_PATH = Path('experiments/047-expanded-action-forks')
OUTPUT = Path('runs/q047-expanded-forks')


def read(path):
    return json.loads(path.read_bytes())


def expansion_run(root, protocol, published):
    audit_path = root / OUTPUT / 'audit.json'
    if (published['protocol_sha256'] != sha256(protocol)
            or published['audit_sha256'] != sha256(audit_path)
            or published['anonymous_files_verified'] is not True or published['ci_conclusion'] != 'success'
            or not re.fullmatch(r'[0-9a-f]{40}', published['commit'])):
        raise ValueError('Expected the exact anonymous result publication for the full stopped campaign')
    receipt = read(protocol.parent / 'publication.json')
    if receipt['commit'] != published['registration_commit']:
        raise ValueError('Results were produced under a different campaign registration')
    plan, terminal = read(protocol), read(audit_path)
    if (terminal['phase'] != 'audited' or terminal['active_block'] is not None
            or terminal['protocol_sha256'] != published['protocol_sha256'] or terminal['optimizer_updates'] != 0
            or terminal['final_test_opened'] or terminal['weights_public']
            or not terminal['audited_attempts'] == terminal['expected_attempts'] == published['rollouts']):
        raise ValueError('The expanded training collection is incomplete or outside training scope')
    program = protocol.parent / 'source' / EXPANSION_PATH / 'run.py'
    block_path = protocol.parent / 'blocks/block-01.json'
    if sha256(block_path) != plan['blocks']['blocks/block-01.json']:
        raise ValueError('The first block differs from the published registration')
    block = read(block_path)
    engine = program.parent.parent / FORK_PATH.name
    for path in (program, program.with_name('design.json'), engine / 'run.py', engine / 'prepare.py'):
        if sha256(path) != block['frozen_files'].get(str(path.relative_to(root))):
            raise ValueError('Served collection source differs from the originally frozen program')
    module = runpy.run_path(str(program))
    verified, design, scheduled = module['verified_campaign'](protocol, protocol.parent / 'publication.json')
    if (verified != plan or design['arms'] != list(FORK_ARMS) or design['subsequent_decoding'] != 'greedy'
            or terminal['audited_attempts'] != 2 * design['expected_training_states']
            or tuple(terminal['blocks']) != tuple(Path(where).name for _, where, _ in scheduled)
            or published['block_audit_sha256'] != {name: item['audit_sha256'] for name, item in terminal['blocks'].items()}):
        raise ValueError('The terminal expanded run differs from the scheduled training-only comparison')
    return module, terminal, scheduled


def merge_source_reviews(known, incoming):
    for repository, row in incoming.items():
        if repository not in known:
            known[repository] = dict(row)
            continue
        current = known[repository]
        totals = ('used_records', 'verified_fragment_records')
        if ({key: value for key, value in current.items() if key not in totals}
                != {key: value for key, value in row.items() if key not in totals}):
            raise ValueError('One repository has mismatched source rights, revision or notice bytes')
        for key in totals:
            current[key] += row[key]


def extend_review(base, root, base_protocol, protocol, published_path, cohort_path, tokenizer,
                  *, private_identifiers=()):
    """Add every verified expanded pair while retaining each original block receipt."""
    root, protocol = Path(root).resolve(), Path(protocol).resolve()
    if base.review['collection_experiment'] != FORK_PATH.name or not protocol.is_relative_to(root / 'private'):
        raise ValueError('The expansion must extend the verified physical 045 baseline')
    published = read(published_path)
    module, terminal, scheduled = expansion_run(root, protocol, published)
    source_result = base.paths['data'].read_bytes()
    if hash_bytes(source_result).hexdigest() != base.review['actual_training_data_sha256']:
        raise ValueError('The baseline collection export changed during candidate preparation')
    records, choices = json.loads(source_result), list(base.choices)
    previous_tasks = {row['task_id'] for row in read(base_protocol)['rows']}
    if len(records) != len(choices):
        raise ValueError('Baseline shapes differ from the audited input choices')
    prompts = {choice.chosen[:choice.prefix_tokens] for choice in choices}
    if len(prompts) != len(choices):
        raise ValueError('The baseline preference set contains repeated token prefixes')
    excluded = {name.split(':', 1)[0].casefold() for name in read(cohort_path)['families']}
    reviewed = {name: dict(row) for name, row in base.review['sources'].items()}
    first_count = len(records)
    verified_blocks = {}
    engine = module['load_engine']()
    for name, where, ordinals in scheduled:
        path = protocol.parent / relative(name)
        plan, design, data, cases = module['load_block'](root, path, protocol.parent / 'publication.json', engine)
        if previous_tasks.intersection(cases):
            raise ValueError('An expanded decision state repeats a task from another collection or block')
        previous_tasks.update(cases)
        if (sha256(data / 'sources.json') != base.review['source_manifest_sha256']
                or sha256(data / 'tasks.json') != base.review['task_manifest_sha256']
                or (design['model_id'], design['base_revision']) != (base.profile['model_id'], base.profile['revision'])):
            raise ValueError('Expanded preferences use a different model revision or source inventory')
        out = root / relative(where)
        summary = module['audited_block'](out, sha256(path), ordinals)
        if terminal['blocks'][out.name] != summary:
            raise ValueError('Block differs from the completed global audit')
        audit, raw = audited_records(out, summary['audit_sha256'])
        source_rights = admitted_sources(data, cases, audit, raw, excluded, arms=FORK_ARMS, first_ordinal=ordinals[0])
        encoded = [encode_decision(row, tokenizer, base.profile) for row in raw]
        for choice in encoded:
            prompt = choice.chosen[:choice.prefix_tokens]
            if prompt in prompts:
                raise ValueError('A preference label repeats the exact earlier model input')
            prompts.add(prompt)
        original_rows = [dict(cases[row['task_id']].proof['arms']['control'],
                              task_id=row['task_id'], ordinal=row['ordinal']) for row in plan['rows']]
        controller = read(out / 'controller.json')
        verified = verify_first_choices(out, {'rows': original_rows}, design, controller['runs'], raw, encoded)
        if verified != 2 * len(raw) or summary['preference_pairs'] != len(raw):
            raise ValueError('Expanded labels differ from their actual matched physical first responses')
        merge_source_reviews(reviewed, source_rights)
        records.extend(raw)
        choices.extend(encoded)
        verified_blocks[out.name] = {'protocol_sha256': sha256(path), 'audit_sha256': summary['audit_sha256'],
                                     'controller_sha256': audit['controller_sha256'],
                                     'export_sha256': audit['training_export_sha256'], 'pairs': len(raw),
                                     'physical_labels': verified}
    added = len(records) - first_count
    if not added or added != terminal['qualifying_preference_pairs'] or len(records) != len(choices):
        raise ValueError('Require the new verified labels from the complete registered physical collection')
    findings = verify_private_labels(records, private_identifiers)
    data_bytes = (json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    expanded = {'protocol_sha256': sha256(protocol), 'audit_sha256': published['audit_sha256'],
                'results_commit': published['commit'], 'result_publication_sha256': sha256(published_path),
                'blocks': verified_blocks, 'pairs': added, 'baseline_pairs': first_count}
    review = dict(base.review, collection_experiment=EXPANSION_PATH.name, expansion=expanded,
                  physical_first_actions_reverified=base.review['physical_first_actions_reverified'] + 2 * added,
                  actual_training_data_sha256=hash_bytes(data_bytes).hexdigest(), known_pattern_findings=findings,
                  sources=dict(sorted(reviewed.items())),
                  pairs=len(choices), decision_counts=dict(Counter(value.kind for value in choices)),
                  longest_sequence=max(max(len(choice.chosen), len(choice.rejected)) for choice in choices),
                  recorded_release_rights_complete=all(item['recorded_weight_publication_approval'] for item in reviewed.values()))
    return DecisionTrainingData(tuple(choices), base.profile, review, {'sources': base.paths['sources']}, data_bytes)
