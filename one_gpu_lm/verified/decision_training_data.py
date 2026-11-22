"""Bind action-preference training to complete collection and original source receipts."""
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256 as hash_bytes
from pathlib import Path
import json
import re

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.serving.receipt_client import payload_sha256, request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.training.decision_preference import EncodedDecision, audited_records, encode_decision
from .continuation_training_data import BASELINE
from .decision_state_collection import load_cases
from .first_edit_continuation import message_hash
from .guard_audit import target_is_executed
from .recovery_sft_projection import collection_inputs
from .tasks import relative, validate_source


SECRET_PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:ENCRYPTED |RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'aws_key_id': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'service_token': re.compile(r'\b(?:sk-proj-[A-Za-z0-9_-]{30,}|sk-ant-[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{30,})\b'),
}
FORK_OUTPUT = 'runs/q045-first-action-forks'
FORK_PATH = Path('experiments/045-first-action-forks')
FORK_ARMS = ('greedy', 'sampled')


@dataclass(frozen=True)
class DecisionTrainingData:
    choices: tuple[EncodedDecision, ...]
    profile: dict
    review: dict
    paths: dict


def admitted_sources(data, cases, audit, records, excluded, *, arms=('control', 'correction')):
    """Check exact eligible forks and the provenance of every selected training source."""
    expected = []
    if len(audit['pairs']) != len(cases):
        raise ValueError('The audit does not cover the registered decision states')
    for number, (case, pair) in enumerate(zip(cases.values(), audit['pairs']), 1):
        if (pair['pair'] != number or pair['kind'] != case.kind or pair['family'] != case.task['family']
                or set(pair['arms']) != set(arms)
                or any(cell['audited'] is not True for cell in pair['arms'].values())):
            raise ValueError('Decision pair differs from the complete executed audit')
        if pair['contrast'] == 'matched_executed_contrast':
            expected.append((case, pair))
    if [record['provenance']['task_id'] for record in records] != [case.task['id'] for case, _ in expected]:
        raise ValueError('Training export differs from the audited set of executed choices')

    counts = Counter()
    sources = {}
    for record, (case, pair) in zip(records, expected):
        evidence = record['provenance']
        names = evidence['chosen_arm'], evidence['rejected_arm']
        source = case.source
        family = source['family']
        aliases = {family, source['repository'], *source.get('related_families', [])}
        fork = evidence['fork_call']
        if (record['kind'] != case.kind or evidence['split'] != 'train' or evidence['family'] != family
                or evidence['prefix_sha256'] != case.proof['prefix_sha256']
                or set(names) != set(pair['arms'])
                or [pair['arms'][arm]['verified_and_submitted'] for arm in names] != [True, False]
                or not 1 <= fork <= min(pair['arms'][arm]['attempted_calls'] for arm in names)
                or evidence['remaining_calls'] != case.budget['max_model_calls'] - fork + 1
                or not 0 < evidence['remaining_generated_tokens'] <= case.budget['max_total_generated_tokens']
                or any(name.split(':', 1)[0].casefold() in excluded for name in aliases)):
            raise ValueError('A preference fork or repository family is outside the approved training scope')
        repo = source['repository']
        if sources.setdefault(repo, source) != source:
            raise ValueError('A repository is associated with more than one approved source identity')
        counts[repo] += 1

    review = {}
    for repo, source in sorted(sources.items()):
        snapshot = data/'snapshots'/relative(repo.replace('/', '--'))
        validate_source(source, snapshot, 'training')
        review[repo] = {'revision': source['revision'], 'license': source['license_spdx'],
                        'used_records': counts[repo], 'verified_fragment_records': counts[repo],
                        'notices': {name: sha256(snapshot/relative(name)) for name in source['notices']},
                        'recorded_weight_publication_approval': source['approved_for'].get('weight_publication') is True}
    return review


def fork_cases(root, protocol_path, plan):
    """Resolve only the training sources and runtime bound to the published fork."""
    folder = protocol_path.parent
    public = json.loads((folder/'publication.json').read_bytes())
    if (plan['output'] != FORK_OUTPUT or plan['training_only'] is not True or plan['final_test_opened'] is not False
            or public['protocol_sha256'] != sha256(protocol_path) or public['ci_conclusion'] != 'success'
            or public['anonymous_files_verified'] is not True or not re.fullmatch(r'[0-9a-f]{40}', public['commit'])
            or any(sha256(root/relative(name)) != expected for name, expected in plan['frozen_files'].items())
            or any(sha256(Path(name)) != expected for name, expected in plan['runtime_inputs'].items())):
        raise ValueError('Require the unchanged, published training-only fork collection')
    for name in ('design.json', 'prepare.py', 'run.py'):
        path = folder/'source'/name
        if plan['frozen_files'].get(str(path.relative_to(root))) != public['public_file_sha256'][str(FORK_PATH/name)]:
            raise ValueError('The served fork code differs from its public registration')
    design_path = folder/'source/design.json'
    design = json.loads(design_path.read_bytes())
    parent_path = root/'experiments/044-deployment-recovery-sft/design.json'
    if (sha256(design_path) != plan['design_sha256'] or design['experiment'] != FORK_PATH.name
            or design['arms'] != list(FORK_ARMS) or design['subsequent_decoding'] != 'greedy'
            or design['final_test_opened'] or design['optimizer_updates'] or design['weights_public']
            or sha256(parent_path) != design['parent_training_design_sha256']):
        raise ValueError('Require the registered first-action intervention on the pinned 27B base')
    parent = json.loads(parent_path.read_bytes())
    collection = Path(plan['collection'])
    prior, _, _, _ = collection_inputs(collection, parent)
    _, reference, data, old = load_cases(collection, prior)
    early = {key: case for key, case in old.items() if case.kind == 'early_edit'}
    if (list(early) != [row['task_id'] for row in plan['rows']] or len(early) != design['expected_training_states']
            or (reference['model_id'], reference['revision']) != (design['model_id'], design['base_revision'])):
        raise ValueError('The frozen first-action cohort differs from the audited early training states')
    cases = {}
    for ordinal, row in enumerate(plan['rows'], 1):
        case = early[row['task_id']]
        budget = dict(parent['actor_budget'])
        budget['max_model_calls'] -= case.prefix['historical_calls']
        budget['max_total_generated_tokens'] -= case.prefix['historical_generated_tokens']
        if (row['ordinal'] != ordinal or row['prefix_sha256'] != case.proof['prefix_sha256']
                or row['budget'] != budget or row['source_sha256'] != case.prefix['decision_source_sha256']
                or row['messages_sha256'] != message_hash(row['messages'])):
            raise ValueError('A restored training prefix or its remaining budget changed')
        cases[row['task_id']] = replace(case, protocol=dict(case.protocol, evaluation_budget=budget))
    profile_path = root/relative(json.loads((root/BASELINE).read_bytes())['profile_path'])
    return design, reference, data, cases, profile_path


def verify_first_choices(out, plan, design, rows, records, choices):
    """Bind approved labels to their two physical first-response receipts."""
    receipts = out/'private/backend/requests'
    if not receipts.is_dir():
        raise ValueError('Expected the original physical generation journal')
    journal = RequestJournal(receipts)
    actor_rows = {row['task_id']: row for row in plan['rows']}
    runs = {row['run_id']: row for row in rows}
    seen = set()
    for record, encoded in zip(records, choices, strict=True):
        provenance = record['provenance']
        actor = actor_rows[provenance['task_id']]
        budget = actor['budget']
        first_input_sha256 = actor['prompt_ids_sha256']
        if (provenance['fork_call'] != 1 or provenance['remaining_generated_tokens'] != budget['max_total_generated_tokens']
                or encoded.prefix_tokens != actor['prompt_tokens']
                or hash_bytes(json.dumps(encoded.chosen[:encoded.prefix_tokens], separators=(',', ':')).encode()).hexdigest()
                != first_input_sha256):
            raise ValueError('The preference label must reproduce the physical first-decision prefix and budget')
        for label in ('chosen', 'rejected'):
            arm = provenance[label+'_arm']
            name = f"{actor['ordinal']:02d}-{arm}"
            directory = out/'private/tasks'/name
            original = runs[name]
            request_path, trace_path = directory/'request-001.json', directory/'trajectory.json'
            trace = json.loads(trace_path.read_bytes())
            payload = json.loads(request_path.read_bytes())
            first = trace['steps'][0]
            generation = first['generation']
            request_id = request_identity(name,1)
            reply = journal.status(request_id)
            sample = design['sampling']
            expected_sample = {'seed':sample['seed_offset']+actor['ordinal'],'temperature':sample['temperature'],
                               'top_p':sample['top_p']} if arm == 'sampled' else None
            if (request_id in seen or sha256(trace_path) != original['trajectory_sha256']
                    or original['sampling_arm'] != arm or original['ordinal'] != actor['ordinal']
                    or trace['eligible_verified_trajectory'] is not (label=='chosen')
                    or original['complete_success_within_budget'] is not (label=='chosen')
                    or first['call'] != 1 or first['request_sha256'] != sha256(request_path)
                    or first['request_sha256'] != provenance[label+'_request_sha256']
                    or first['source_before'] != actor['source_sha256'] or not target_is_executed(first)
                    or payload['messages'] != actor['messages'] or payload['max_input_tokens'] != budget['max_input_tokens']
                    or payload['max_new_tokens'] != min(budget['max_generated_tokens_per_call'],
                                                       budget['max_total_generated_tokens'])
                    or generation['input_tokens_sha256'] != first_input_sha256 or generation['text'] != record[label]
                    or generation.get('first_action_sampling') != expected_sample or reply.get('state') != 'completed'
                    or reply['unknown_generation_cost'] or reply['result'] != generation
                    or reply['request_id'] != request_id or reply['payload_sha256'] != payload_sha256(payload)):
                raise ValueError('A preference label differs from the audited physical first action')
            seen.add(request_id)
    return len(seen)


def review_inputs(collection_root, protocol_path, protocol_sha256, audit_sha256, cohort_path, cohort_sha256,
                  tokenizer, *, private_identifiers=()):
    """Open only audited training data and the score-independent development manifest."""
    root, protocol_path, cohort_path = (Path(path).resolve() for path in (collection_root, protocol_path, cohort_path))
    protocol_bytes, cohort_bytes = protocol_path.read_bytes(), cohort_path.read_bytes()
    if (hash_bytes(protocol_bytes).hexdigest() != protocol_sha256
            or hash_bytes(cohort_bytes).hexdigest() != cohort_sha256):
        raise ValueError('The collection protocol or development cohort differs from its reviewed digest')
    plan, cohort = json.loads(protocol_bytes), json.loads(cohort_bytes)
    if cohort['final_test_opened'] is not False or cohort['model_generations'] != 0:
        raise ValueError('Require the independent pre-scoring development cohort')
    from_fork = plan.get('output') == FORK_OUTPUT
    out = root/relative(plan['output'] if from_fork else plan['run_directory'])
    audit, records = audited_records(out, audit_sha256)
    if audit['protocol_sha256'] != protocol_sha256 or not records:
        raise ValueError('Expected qualifying private pairs from the registered, completely audited collection')

    if from_fork:
        design, reference, data, cases, profile_path = fork_cases(root, protocol_path, plan)
        arms = FORK_ARMS
        expected = 2*design['expected_training_states']
    else:
        design, reference, data, cases = load_cases(root, plan)
        profile_path = root/relative(design['student_profile_path'])
        arms = ('control', 'correction')
        expected = design['expected_rollouts']
    excluded = {name.split(':', 1)[0].casefold() for name in cohort['families']}
    if len(audit['pairs'])*2 != audit['audited_attempts'] or audit['audited_attempts'] != expected:
        raise ValueError('The audited collection length differs from its registration')
    source_review = admitted_sources(data, cases, audit, records, excluded, arms=arms)

    findings = Counter()
    for record in records:
        text = json.dumps(record, ensure_ascii=False)
        findings.update(name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text))
        if any(value and value in text for value in private_identifiers):
            findings['known_private_identifier'] += 1
    if findings:
        raise ValueError('The preference export matches a private identifier or credential pattern')
    profile = json.loads(profile_path.read_bytes())
    if ((profile['model_id'], profile['revision']) != (reference['model_id'], reference['revision'])
            or profile['training']['dialogue_format'] != 'native_roundtrip'):
        raise ValueError('The approved student profile differs from the collected base or dialogue format')
    choices = tuple(encode_decision(record, tokenizer, profile) for record in records)
    physical_labels = 0
    if from_fork:
        controller = json.loads((out/'controller.json').read_bytes())
        physical_labels = verify_first_choices(out, plan, design, controller['runs'], records, choices)
    source_path = data/'sources.json'
    review = {'schema_version': 1, 'collection_experiment': FORK_PATH.name if from_fork else '043-decision-state-collection',
              'physical_first_actions_reverified': physical_labels,
              'collection_protocol_sha256': protocol_sha256, 'collection_audit_sha256': audit_sha256,
              'collection_controller_sha256': audit['controller_sha256'], 'development_cohort_sha256': cohort_sha256,
              'actual_training_data_sha256': audit['training_export_sha256'],
              'source_manifest_sha256': sha256(source_path), 'task_manifest_sha256': sha256(data/'tasks.json'),
              'source_snapshots_match': True, 'all_source_notices_and_lineage_reviewed_for_training': True,
              'known_pattern_findings': {}, 'sources': source_review, 'pairs': len(choices),
              'decision_counts': dict(Counter(value.kind for value in choices)),
              'longest_sequence': max(max(len(item.chosen), len(item.rejected)) for item in choices),
              'recorded_release_rights_complete': all(row['recorded_weight_publication_approval']
                                                      for row in source_review.values()),
              'final_test_opened': False, 'weights_public': False}
    paths = {'sources': source_path, 'data': out/'private/preference-pairs.json'}
    return DecisionTrainingData(choices, profile, review, paths)
