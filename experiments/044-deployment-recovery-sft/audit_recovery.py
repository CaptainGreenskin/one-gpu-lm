"""Replay 044 with independently verified grading build outputs."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json, identity_digest, single_writer
from one_gpu_lm.verified import historic_replay
from one_gpu_lm.verified.historic_tasks import grade
from one_gpu_lm.verified.sandbox import Sandbox
from one_gpu_lm.verified.tasks import digest, files, relative


def read(path):
    return json.loads(path.read_bytes())


def verify_build(workspace, grader, command, recorded, execute):
    original, physical = files(workspace), files(grader)
    if any(physical.get(path) != checksum for path, checksum in original.items()):
        raise ValueError('A frozen grader input is missing or changed')
    new_files = {path: checksum for path, checksum in physical.items() if path not in original}
    if not new_files:
        return None

    repeated = execute(workspace, command)
    if (files(workspace) != physical
            or any(repeated[key] != recorded[key] for key in ('returncode', 'reason'))):
        raise ValueError('Original offline grader did not reproduce the build files and exit outcome')
    return {
        'input_inventory_sha256': identity_digest(original),
        'post_run_inventory_sha256': identity_digest(physical),
        'reproduced_files_sha256': new_files,
        'repeated_output_sha256': digest(repeated['output'].encode()),
    }


def check_graders(out, plan, order, state, panel, scratch, actor_directory):
    counts, artifacts, sandboxes = Counter(), {}, {}
    for item, result in zip(order, state['runs'], strict=True):
        actor = actor_directory(out, result['attempt'], item['run_id'])
        grader, output = actor / 'grader', actor / 'grader-output.json'
        if not (grader.exists() or output.exists() or 'grader_output_sha256' in result['grade']):
            continue
        if not grader.is_dir() or grader.is_symlink() or not output.is_file() or output.is_symlink():
            raise ValueError('Executed grading must retain its workspace and output receipt')
        counts['executed_graders'] += 1
        case = panel.cases[item['task_number']]
        task = case['prompts'][plan['selected_opening']]
        snapshot = panel.data / 'snapshots' / relative(task['source_id'])
        recorded = read(output)
        proof = None

        def repeat(workspace, command):
            runtime = panel.runtimes.get(task['family'], panel.runtime_path)
            if runtime not in sandboxes:
                sandboxes[runtime] = Sandbox(read(runtime), plan['evaluation_budget']['max_tool_seconds'])
            return sandboxes[runtime].execute(workspace, command)

        def compare(workspace, command):
            nonlocal proof
            if command != task['grading_command']:
                raise ValueError('Reconstructed grading command differs from the frozen command')
            proof = verify_build(workspace, grader, command, recorded, repeat)
            return recorded

        receipt = grade(snapshot, scratch / item['run_id'], case['source'], task,
                        actor / 'workspace', case['before'], compare)
        if 'grader_output_sha256' not in receipt or receipt != result['grade']:
            raise ValueError('Reconstructed grading receipt differs from the original outcome')
        if proof is not None:
            counts['repeated_graders'] += 1
            counts['reproduced_files'] += len(proof['reproduced_files_sha256'])
            artifacts[grader] = dict(proof, grader_output_file_sha256=sha256(output))
    return counts, artifacts


def replay(args, comparison, plan, plan_hash, openings_hash, order, panel):
    out = args.root / relative(plan['run_directory'])
    controller, target = out / 'controller.json', out / 'audit.json'
    with single_writer(out / 'private/audit-recovery.lock'):
        state = read(controller)
        if (target.exists() or sha256(controller) != args.controller_sha256
                or sha256(args.failed_log) != args.failed_log_sha256
                or not state['complete'] or not state['backend_stopped'] or state['final_test_opened']
                or [run['run_id'] for run in state['runs']] != [item['run_id'] for item in order]):
            raise ValueError('Require the original complete controller and retained failure receipt')
        with TemporaryDirectory(prefix='grade-build-proof-', dir=out / 'private') as scratch:
            counts, artifacts = check_graders(out, plan, order, state, panel, Path(scratch),
                                              comparison.actor_directory)
        if not artifacts:
            raise ValueError('Use the registered audit directly if no build files need verification')

        consumed = set()

        def verified_files(path):
            path = Path(path)
            actual = files(path)
            evidence = artifacts.get(path)
            if evidence is None:
                return actual
            original = {name: value for name, value in actual.items()
                        if name not in evidence['reproduced_files_sha256']}
            if (path in consumed or identity_digest(actual) != evidence['post_run_inventory_sha256']
                    or identity_digest(original) != evidence['input_inventory_sha256']
                    or sha256(path.parent / 'grader-output.json') != evidence['grader_output_file_sha256']):
                raise ValueError('Proven grader artifacts or original output changed before replay')
            consumed.add(path)
            return original

        staged = None

        def stage_audit(path, value):
            nonlocal staged
            if path != target or staged is not None:
                raise ValueError('Unexpected audit output')
            staged = value

        with patch.object(historic_replay, 'files', verified_files), patch.object(comparison, 'atomic_json', stage_audit):
            evidence = comparison.audit(args.root, plan, plan_hash, openings_hash, order, panel)
        if (staged is not evidence or consumed != set(artifacts) or target.exists()
                or not evidence['passed'] or len(evidence['rows']) != len(order)
                or sha256(controller) != args.controller_sha256):
            raise ValueError('Require the complete original audit after checking every build file')

        manifest = {str(path.relative_to(out)): value for path, value in artifacts.items()}
        identity = identity_digest(manifest)
        proof_path = out / 'private' / f'grader-build-proof-{identity}.json'
        if proof_path.exists() and read(proof_path) != manifest:
            raise ValueError('A private evidence receipt changed')
        atomic_json(proof_path, manifest)
        summary = {'schema_version': 1, 'controller_sha256': args.controller_sha256,
                   'protocol_sha256': plan_hash, 'original_audit_failure_sha256': args.failed_log_sha256,
                   'recovery_code_sha256': sha256(Path(__file__)), 'private_build_proof_sha256': sha256(proof_path),
                   'audited_rollouts': len(order), 'grader_commands_repeated': counts['repeated_graders'],
                   'matched_grading_workspaces': counts['executed_graders'],
                   'reproduced_build_files': counts['reproduced_files'], 'model_generations': 0,
                   'final_test_opened': False}
        evidence['grading_build_recovery'] = summary
        atomic_json(target, evidence)
        return summary


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'publication', 'openings', 'failed-log'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    for name in ('controller-sha256', 'failed-log-sha256'):
        parser.add_argument(f'--{name}', required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    sys.path.insert(0, str(args.root / 'experiments/039-release-development'))
    import candidate_comparison as comparison

    print(json.dumps(replay(args, comparison, *comparison.registered_inputs(args)), allow_nan=False))


if __name__ == '__main__':
    main()
