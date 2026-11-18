"""Audit the scored schedule and disclose the costs of a discarded host-aborted actor."""
import argparse
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256, write_json
from .resilient_eval_audit import run as audit_journals
from .resume_prefix import read


def loads(schedule):
    return [item['variant'] for i,item in enumerate(schedule) if not i or item['variant'] != schedule[i-1]['variant']]


def run(root, protocol_path, out):
    if out.exists():
        raise ValueError('Preserve the previous audit')
    protocol = read(protocol_path)
    directory = root/protocol['run_directory']
    prior = read(root/protocol['interrupted_protocol_path'])
    archived = root/prior['run_directory']
    value = audit_journals(root,protocol_path,directory/'segmented-audit-intermediate.json')
    controller = read(directory/'controller.json')
    receipt = read(root/protocol['recovery_receipt_path'])
    split = receipt['completed_runs']
    old = read(archived/'controller.json')
    physical = [load['variant'] for load in controller['backend_lifecycles']]
    exact = value['checks'].pop('exact_registered_checkpoint_load_order')
    value['checks'].update(
        registered_segment_checkpoint_load_order=physical==loads(protocol['schedule'][:split])+loads(protocol['schedule'][split:]),
        retained_prefix_matches_archive=old['runs']==controller['runs'][:split]
            and sha256(archived/'controller.json')==receipt['interrupted_controller_sha256'],
        interrupted_actor_kept_out_of_scores=all(r['trajectory_sha256']!=receipt['interrupted_trajectory_sha256']
            for r in value['rows'] if r.get('trajectory_sha256')),
        recovery_receipt_identity=controller['recovery']==receipt and not receipt['final_test_opened'],
    )
    value['passed'] = all(value['checks'].values())
    value['host_recovery'] = dict(receipt, original_unsplit_backend_order_matched=exact,
        resumed_epoch=controller['resumed_epoch'], prior_completed_runs_retained=split,
        resumed_task_runs=len(controller['runs'])-split,
        discarded_actor_generation_cost_fully_known=receipt['aborted_unknown_response_count']==0)
    write_json(out,value)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','protocol','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    result = run(a.root.resolve(),a.protocol.resolve(),a.out.resolve())
    print(json.dumps({k:result[k] for k in ('passed','checks','journal','host_recovery')}))
