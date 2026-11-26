"""Run and audit the nine-task retention diagnostic with the original evaluator."""

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, identity_digest
from one_gpu_lm.models.release_lineage import audited_training
from one_gpu_lm.verified.tasks import relative


EXP = Path('experiments/055-retention-sft')
EVALUATOR = Path('experiments/039-release-development')
REFERENCE = Path('private/comparison-054')
REFERENCE_RUN = Path('runs/q054-dev')
OUTPUT = Path('runs/q055-keep')
STRATA = ('prior_regressions', 'prior_gains', 'prior_retained')


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(args):
    from analysis import schedule
    from candidate_comparison import adapter_files, registered_inputs, reviewed_adapter

    root, job = args.root, args.protocol.parent
    design_path, screen_path = root / EXP / 'design.json', job / 'screen.private.json'
    design = read(design_path)
    require(sha256(screen_path) == design['development_screen_sha256'], 'Use the pretraining diagnostic selection')
    screen = read(screen_path)
    require(screen['experiment'] == EXP.name and screen['development_audit_sha256'] == design['previous_development_audit_sha256']
            and screen['repetitions'] == [0, 1, 2] and screen['final_task_records_read'] is False
            and set(screen['task_strata']) == set(STRATA)
            and all(len(screen['task_strata'][name]) == 3 for name in STRATA),
            'Require the frozen development-only diagnostic inventory')
    numbers = [number for name in STRATA for number in screen['task_strata'][name]]
    require(len(set(numbers)) == 9 and all(type(n) is int and 1 <= n <= 60 for n in numbers),
            'Require nine distinct registered development tasks')

    reference_protocol, reference_publication = (root / REFERENCE / name for name in ('protocol.json', 'publication.json'))
    require(read(reference_protocol)['run_directory'] == str(REFERENCE_RUN), 'Use the original 054 development protocol')
    audit_path, control_path = (root / REFERENCE_RUN / name for name in ('audit.json', 'controller.json'))
    require(sha256(audit_path) == screen['development_audit_sha256'], 'Use the complete original development audit')
    original = read(audit_path)
    require(original['passed'] and original['protocol_sha256'] == sha256(reference_protocol)
            and original['controller_sha256'] == sha256(control_path)
            and len(original['rows']) == 360 and not original['final_test_opened'] and not original['weights_public'],
            'Require the actual complete 054 development comparison')
    expected = {(n, repeat, arm) for n in numbers for repeat in screen['repetitions'] for arm in ('base', 'candidate')}
    previous = {}
    for row in original['rows']:
        key = row['task_number'], row['repetition'], row['arm']
        if key in expected:
            require(row['audit_passed'] and key not in previous, 'Diagnostic controls must have unique audited rows')
            previous[key] = row['completed_repair']
    require(set(previous) == expected, 'The original development controls are incomplete')
    counts = {name: {arm: sum(previous[n, repeat, arm] for n in screen['task_strata'][name]
                              for repeat in screen['repetitions']) for arm in ('base', 'candidate')} for name in STRATA}
    require(counts == screen['old_arm_retained_pair_counts'], 'The diagnostic selection differs from the public history')

    reference_args = SimpleNamespace(root=root, protocol=reference_protocol, publication=reference_publication,
                                     openings=args.openings)
    parent, _, openings_hash, _, panel = registered_inputs(reference_args)
    require(parent['selected_opening'] == 'original', 'Use the same served opening and policy as 054')
    candidate = job / 'candidate'
    review, training_plan, profile, integrity = (candidate / name for name in (
        'candidate-input-review.json', 'training-protocol.json', 'training-profile.json', 'integrity.json'))
    evidence = audited_training(review, training_plan, profile, integrity)
    require(evidence.lineage['experiment'] == EXP.name and evidence.lineage['optimizer_updates'] == 50
            and evidence.protocol['recipe']['learning_rate'] == design['candidate_learning_rate']
            and evidence.lineage['training_data_sha256'] == design['input_sha256']['data_path'],
            'Load the independently audited lower-rate checkpoint')
    adapter = candidate / 'adapter'
    files = adapter_files(adapter)
    members = {number: panel.cases[old] for number, old in enumerate(numbers, 1)}
    families = dict(Counter(case['source']['family'] for case in members.values()))
    plan = dict(parent, comparison_kind='development_retention_diagnostic', candidate=EXP.name,
                task_count=9, block_size=9, rollout_count=54, families=families,
                language_counts=dict(Counter(case['source']['language'] for case in members.values())),
                run_directory=str(OUTPUT), adapter_directory=str(adapter.relative_to(root)),
                candidate_review_path=str(review.relative_to(root)), all_adapter_files_sha256=files,
                original_development_audit_sha256=sha256(audit_path),
                diagnostic_selection_sha256=sha256(screen_path), eligible_for_final_promotion=False)
    served = reviewed_adapter(evidence.review, files, plan)
    order = schedule(members, plan)
    references = (root / EXP / 'diagnostic.py', design_path, screen_path, reference_protocol,
                  reference_publication, audit_path, control_path, review, training_plan, profile, integrity)
    plan['frozen_files'] = parent['frozen_files'] | {str(path.relative_to(root)): sha256(path) for path in references}
    plan['schedule_sha256'] = identity_digest(order)
    common = dict(panel.common, adapter_directory=plan['adapter_directory'], all_adapter_files_sha256=files,
                  adapter_sha256=served)
    return plan, openings_hash, order, replace(panel, cases=members, common=common)


def run(args):
    from candidate_comparison import audit, evaluate

    plan, openings_hash, order, panel = prepare(args)
    if args.phase == 'freeze':
        require(not (args.root / OUTPUT).exists() and not args.protocol.is_symlink(), 'Preserve the original diagnostic attempt')
        if args.protocol.exists():
            require(read(args.protocol) == plan, 'The frozen diagnostic protocol differs from its original inputs')
        else:
            atomic_json(args.protocol, plan)
        return {'phase': 'frozen', 'protocol_sha256': sha256(args.protocol), 'task_count': 9,
                'rollout_count': 54, 'selection_sha256': plan['diagnostic_selection_sha256'],
                'eligible_for_final_promotion': False, 'final_test_opened': False}
    checksum = sha256(args.protocol)
    publication = read(args.publication)
    require(read(args.protocol) == plan and publication['protocol_sha256'] == checksum
            and publication['anonymous_files_verified'] is True and publication['ci_conclusion'] == 'success',
            'Use the exact verified development diagnostic registration')
    if args.phase == 'audit':
        result = audit(args.root, plan, checksum, openings_hash, order, panel)
        return {'passed': result['passed'], 'verified_requests': result['verified_requests'], 'final_test_opened': False}
    state = evaluate(args.root, plan, checksum, order, panel, resume=args.phase == 'resume')
    return {'complete': state['complete'], 'rollouts': len(state['runs']), 'final_test_opened': False}


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('phase', choices=('freeze', 'run', 'resume', 'audit'))
    for option in ('protocol', 'openings', 'publication'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.protocol = args.root / relative(str(args.protocol))
    args.openings = args.root / relative(str(args.openings))
    args.publication = args.root / relative(str(args.publication))
    sys.path.insert(0, str(args.root / EVALUATOR))
    print(json.dumps(run(args), allow_nan=False), flush=True)
