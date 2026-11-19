"""Replay the 034 development schedule with the same 27B base and no adapter."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time
import traceback

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.resilient_backend_control import JournalBackend
from one_gpu_lm.verified.resilient_rollout import run as rollout
from one_gpu_lm.verified.rollout import atomic
from one_gpu_lm.verified.sandbox import Sandbox


EXPERIMENT = Path('experiments/035-edit-bottleneck')
PRIVATE = Path('private/baseline-diagnosis')
OUTPUT = Path('runs/historic035-base-001')
PREVIOUS_RUN = Path('runs/historic034-development-001')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def baseline_schedule(matched_order):
    return [
        dict(row, arm='base', run_id=f"t{row['task_number']:02d}-base")
        for row in matched_order if row['arm'] == 'control'
    ]


@dataclass
class Context:
    plan_path: Path
    plan: dict
    legacy: object
    data: Path
    tasks: dict
    sources: dict
    order: list
    protocol: dict
    runtime: dict
    runtime_path: Path
    legacy_proof: dict


def inputs(root):
    registration = root / EXPERIMENT / 'baseline-plan.json'
    plan = read(registration)
    receipt = read(root / PRIVATE / 'baseline-publication.json')
    if (receipt['anonymous_files_verified'] is not True or receipt['ci_conclusion'] != 'success'
            or receipt['plan_sha256'] != sha256(registration)):
        raise ValueError('The exact preregistration and its passing CI are required')
    spec = importlib.util.spec_from_file_location(
        'frozen_034_evaluation', root / 'experiments/034-first-action-ablation/evaluate.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    if not previous.frozen(root, plan['frozen_files']):
        raise ValueError('A 035 preregistered input changed')
    old_plan, parent, data, tasks, sources, matched_order = previous.inputs(root)
    if any(plan[k] != old_plan[k] for k in ('evaluation_budget', 'agent_policy', 'normalization')):
        raise ValueError('The baseline setting must match 034')
    schedule = baseline_schedule(matched_order)
    if (len(schedule) != plan['rollouts'] or fingerprint(schedule) != plan['schedule_sha256']
            or plan['final_test_opened'] or plan['optimizer_updates'] != 0):
        raise ValueError('Expected the registered development-only task order')
    previous_proof = read(root / previous.PRIVATE / 'development-integrity.json')
    old_controller = root / PREVIOUS_RUN / 'controller.json'
    if (not previous_proof['passed'] or previous_proof['final_test_opened']
            or previous_proof['controller_sha256'] != sha256(old_controller)
            or previous_proof['registration_sha256'] != sha256(root / previous.EXP / 'evaluation-plan.json')
            or len(previous_proof['rows']) != 2 * len(schedule)):
        raise ValueError('Complete audited 034 comparison required')
    protocol = {key: parent[key] for key in (
        'environment_path', 'evaluation_profile_path', 'evaluation_profile_sha256',
        'model_manifest_path', 'model_id', 'revision', 'evaluation_budget', 'agent_policy', 'normalization')}
    protocol.update(adapter_sha256={}, evaluation_variant='base', final_test_opened=False)
    runtime = read(root / 'private/parent-runtime.json')
    runtime_path = root / previous.PRIVATE / 'offline-runtime.json'
    return Context(registration, plan, previous, data, {task['id']: task for task in tasks},
                   sources, schedule, protocol, runtime, runtime_path, previous_proof)


def sweep(root, out, ctx, perform=rollout):
    if out.exists():
        raise ValueError('Preserve the original attempt')
    profile = read(root / ctx.protocol['evaluation_profile_path'])
    verify_model(Path(ctx.runtime['model']), root / ctx.protocol['model_manifest_path'], profile)
    budget = ctx.protocol['evaluation_budget']
    sandbox = Sandbox(read(ctx.runtime_path), budget['max_tool_seconds'])
    if len(str(out / 'backend.sock').encode()) >= 108:
        raise ValueError('Backend UNIX socket path exceeds its Linux limit')
    (out / 'private').mkdir(parents=True, mode=0o700)
    start = time.time()
    deadline = start + ctx.plan['pipeline_limit_seconds']
    recovery = ctx.plan['recovery_accounting_seconds']
    state = {
        'stage': 'loading', 'complete': False, 'started_epoch': start,
        'registration_sha256': sha256(ctx.plan_path), 'schedule_sha256': ctx.plan['schedule_sha256'],
        'runs': [], 'backend_lifecycles': [], 'optimizer_updates': 0, 'final_test_opened': False,
        'training_export_allowed': False, 'weights_public': False,
    }

    def save():
        atomic(out / 'controller.json', state)

    backend = JournalBackend(root, out, ctx.runtime, ctx.protocol, state['backend_lifecycles'], save)
    telemetry = Telemetry(out / 'telemetry.csv', 1)
    telemetry.start()
    save()
    try:
        socket = backend.ensure('base', deadline)
        state['stage'] = 'evaluating'
        for item in ctx.order:
            if time.time() + budget['max_task_seconds'] + budget['max_tool_seconds'] + recovery > deadline:
                raise TimeoutError('Campaign limit leaves insufficient time for another task')
            state['current_task_number'] = item['task_number']
            save()
            task = ctx.tasks[item['task_id']]
            directory = out / 'private/tasks' / item['run_id']
            result = perform(root, ctx.data, task, ctx.sources[task['source_id']], socket, sandbox,
                             ctx.protocol, directory, 'base_current',
                             journal_path=backend.journal_path, recovery_seconds=recovery)
            backend.check_exclusive()
            result.update(run_id=item['run_id'], contrast_arm=item['arm'], backend_index=backend.number,
                          task_number=item['task_number'], block=item['block'])
            write_json(directory / 'result.json', result)
            state['runs'].append(result)
            save()
            print(json.dumps({'completed_rollouts':len(state['runs']), 'planned_rollouts':len(ctx.order)}), flush=True)
        state.update(stage='completed', complete=True)
    except BaseException as error:
        state.update(stage='failed', error_type=type(error).__name__)
        (out / 'private/error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        backend.stop()
        frozen = ctx.legacy.frozen(root, ctx.plan['frozen_files'])
        state.update(telemetry=telemetry.stop(), backend_stopped=True, finished_epoch=time.time(),
                     finished_utc=datetime.now(timezone.utc).isoformat(), frozen_inputs_unchanged=frozen)
        if not frozen:
            state.update(stage='input_changed', complete=False)
        save()
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    context = inputs(root)
    if args.check:
        print(json.dumps({'scheduled_rollouts':len(context.order), 'final_test_opened':False,
                          'cpu_preflight_passed':True}))
    else:
        sweep(root, root / OUTPUT, context)
