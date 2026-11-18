"""Verify protocol identity under the protocol that actually executed each run.

The first independent replay used the post-reboot protocol hash for every row.
The row's other replay, grading and journal checks already ran on the recorded
trace. Keep that audit intact and independently recheck the per-segment hash.
"""
import argparse
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256, write_json
from .historic_eval_design import effective_protocol
from .resilient_evaluation import validate_repeat
from .resume_prefix import read
from .tasks import digest, relative


def run(root, protocol_path):
    protocol = read(protocol_path)
    directory = root/relative(protocol['run_directory'])
    destination = directory/'audit-protocol-v2.json'
    post_destination = directory/'postprocess-protocol-v2.json'
    if destination.exists() or post_destination.exists():
        raise ValueError('Preserve the existing segment-identity audit')
    prior_path = root/relative(protocol['interrupted_protocol_path'])
    if sha256(prior_path) != protocol['frozen_files'][protocol['interrupted_protocol_path']]:
        raise ValueError('Archived first segment protocol differs from the frozen receipt')
    prior = read(prior_path)
    validate_repeat(root, prior)
    validate_repeat(root, protocol)
    if any(sha256(root/relative(n))!=h for n,h in protocol['frozen_files'].items()):
        raise ValueError('Frozen source or policy changed')
    audit_path = directory/'audit.json'
    audit = read(audit_path)
    prior_controller = root/relative(prior['run_directory'])/'controller.json'
    current = read(directory/'controller.json')
    archived = read(prior_controller)
    split = audit['host_recovery']['prior_completed_runs_retained']
    if (sha256(prior_controller) != audit['host_recovery']['interrupted_controller_sha256']
            or current['runs'][:split] != archived['runs']
            or len(audit['rows']) != len(protocol['schedule']) or not current['complete']
            or not all(audit['checks'].get(k) for k in (
                'retained_prefix_matches_archive','journal_inventory_exact','physical_costs_accounted',
                'all_task_journals_verified','registered_segment_checkpoint_load_order'))):
        raise ValueError('Incomplete evaluation or missing segment evidence')
    corrected = 0
    for index,(item,row) in enumerate(zip(protocol['schedule'],audit['rows'])):
        if item['run_id'] != row['run_id'] or item['variant'] != row['variant']:
            raise ValueError('Audit row does not match the schedule')
        segment = prior if index < split else protocol
        path = directory/'private/tasks'/item['run_id']/'trajectory.json'
        trace = read(path)
        expected = digest(json.dumps(effective_protocol(segment,item['variant']),sort_keys=True).encode())
        old_pass = row['checks']['protocol_identity']
        row['checks']['protocol_identity'] = (sha256(path)==row['trajectory_sha256']
                                             and trace['protocol_sha256']==expected)
        row['execution_protocol_sha256'] = sha256(prior_path if index<split else protocol_path)
        row['audit_passed'] = bool(row['journal_audit_passed'] and all(row['checks'].values()))
        corrected += int(not old_pass and row['checks']['protocol_identity'])
    audit['checks']['all_run_audits_passed'] = all(r['audit_passed'] for r in audit['rows'])
    audit['checks']['all_execution_protocols_verified_by_segment'] = all(r['checks']['protocol_identity'] for r in audit['rows'])
    audit['passed'] = all(audit['checks'].values())
    audit['protocol_identity_recheck'] = {
        'first_audit_sha256':sha256(audit_path),'updated_protocol_identity_rows':corrected,
        'first_segment_rows':split,'second_segment_rows':len(audit['rows'])-split,
        'no_model_generations_or_regrades':True,
    }
    write_json(destination,audit)
    post = read(directory/'postprocess.json')
    post.update(audit_passed=audit['passed'],audit_checks=audit['checks'],
                protocol_identity_recheck=audit['protocol_identity_recheck'])
    write_json(post_destination,post)
    return audit


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args()
    audit=run(args.root.resolve(),args.protocol.resolve())
    print(json.dumps({'passed':audit['passed'],'checks':audit['checks'],
                      'protocol_identity_recheck':audit['protocol_identity_recheck']}))
