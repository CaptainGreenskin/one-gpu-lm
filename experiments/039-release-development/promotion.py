"""Bind the development decision to a checkpoint and the existing final-test seal."""
import argparse
from hashlib import sha256 as hash_bytes
import importlib.util
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256
from analysis import analyze


REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO/'experiments/022-action-repair'
WEIGHT_FILES = ('adapter_config.json', 'adapter_model.safetensors')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def authorize(protocol_path, audit_path, seal_path, review_path):
    protocol = read(protocol_path)
    audit = read(audit_path)
    protocol_hash = sha256(protocol_path)
    budget = protocol['evaluation_budget']
    design = read(Path(__file__).with_name('design.json'))
    if any(protocol.get(key)!=value for key,value in design.items()):
        raise ValueError('Require the registered development design')
    if (audit['passed'] is not True or audit['final_test_opened'] is not False
            or audit['weights_public'] is not False or audit['optimizer_updates']!=0
            or audit['protocol_sha256']!=protocol_hash
            or audit['verified_requests']!=sum(row['attempted_calls'] for row in audit['rows'])
            or any(not 1<=row['attempted_calls']<=budget['max_model_calls']
                   or not 0<=row['generated_tokens']<=budget['max_total_generated_tokens']
                   or not 1<=row['opening_input_tokens']<=budget['max_input_tokens'] for row in audit['rows'])):
        raise ValueError('Require complete audited development results for the unchanged protocol')
    result = analyze(audit['rows'], protocol)
    if result!=audit['analysis'] or result['received_runs']!=protocol['rollout_count']:
        raise ValueError('Stored development decision differs from independent recomputation')
    amendment_path=Path(__file__).with_name('recovery-plan.json')
    amendment=read(amendment_path)
    recovery=audit['recovery']
    if (amendment['original_protocol_sha256']!=protocol_hash
            or recovery['plan_sha256']!=sha256(amendment_path)
            or any(recovery.get(key)!=value for key,value in amendment['incident'].items())
            or recovery['retained_rollouts']+recovery['resumed_rollouts']!=protocol['rollout_count']
            or not 0<audit['finished_epoch']-audit['started_epoch']<=protocol['pipeline_limit_seconds']):
        raise ValueError('Require the audited continuation inside the original deadline')
    if 'restart' in recovery:
        restart_path=Path(__file__).with_name('host-restart-plan.json')
        restart=read(restart_path)
        evidence=recovery['restart']
        if (restart['original_protocol_sha256']!=protocol_hash or restart['first_recovery_plan_sha256']!=sha256(amendment_path)
                or evidence['plan_sha256']!=sha256(restart_path)
                or any(evidence.get(key)!=value for key,value in restart['incident'].items())
                or evidence['replayed_rollouts']!=evidence['remaining_rollouts']
                or evidence['retained_rollouts']+evidence['replayed_rollouts']!=protocol['rollout_count']
                or not recovery['resumed_epoch']<evidence['last_pre_restart_sample_epoch']<evidence['boot_epoch']
                       <=evidence['resumed_epoch']<=audit['finished_epoch']):
            raise ValueError('Require the verified host restart continuation')
    if result['development_gate_passed'] is not True:
        return {'approved':False,'stage':'development_gate','checks':result.get('checks',{}),
                'development_protocol_sha256':protocol_hash,'final_test_opened':False,'weights_public':False}

    seal = read(seal_path)
    reference_path = REFERENCE/'protocol.json'
    reference = read(reference_path)
    gate = reference['release_gate']
    inventory_path = REFERENCE/'results/2026-11-12/task-inventory.json'
    if (seal['reference_release_protocol_sha256']!=sha256(reference_path)
            or seal['public_task_inventory_sha256']!=sha256(inventory_path)
            or seal['development_protocol_sha256']!=protocol_hash
            or seal['release_gate']!=gate or seal['fresh_family_overlap']
            or seal['final_campaign_directories']!=0 or seal['final_test_opened'] is not False
            or seal['private_task_schema_read'] is not False or seal['authorized_model_scoring'] is not False):
        raise ValueError('Require the verified unreleased final-task seal for this development protocol')
    inventory = read(inventory_path)
    cohorts = {'legacy':gate['legacy_passes_required']}
    for key,role in [('fresh','fresh_final_not_selection'),('historical','historical_final_regression_not_selection')]:
        ids = sorted(row['id'] for row in inventory if row.get('evaluation_role')==role)
        digest = hash_bytes(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        cohorts[key] = len(ids)
        if (cohorts[key]!=gate[key+'_tasks'] or len(set(ids))!=len(ids)
                or seal['cohorts'][key]!={'tasks':cohorts[key],'ids_sha256':digest}):
            raise ValueError('Final scope differs from the original preregistration')
    for name,digest in seal['frozen_file_digests'].items():
        if reference['frozen_files'].get(name)!=digest:
            raise ValueError('Sealed file digest is not registered')
    if not {'private/tasks.json','private/fresh-tasks.json','private/legacy-functions.jsonl',
            'source-snapshots.json','runs/admission/admission.json','runs/admission-fresh-draft002/admission.json'} <= seal['frozen_file_digests'].keys():
        raise ValueError('Incomplete sealed-file verification')

    review = read(review_path)
    if (review['audited_checkpoint_matches'] is not True or review['eligible_for_existing_scope_of_development'] is not True
            or review['model_id']!=protocol['pretrained_model'] or review['revision']!=protocol['base_revision']
            or review['model_id']!=reference['model_id'] or review['revision']!=reference['revision']
            or review['checkpoint_files_sha256']!=protocol['all_adapter_files_sha256']
            or review['weights_public'] or review['final_test_opened']):
        raise ValueError('The reviewed adapter is different from the measured candidate')
    return {'approved':True,'stage':'final_evaluation','model_id':review['model_id'],'base_revision':review['revision'],
            'checkpoint_sha256':{name:review['checkpoint_files_sha256'][name] for name in WEIGHT_FILES},
            'development_protocol_sha256':protocol_hash,'development_audit_sha256':sha256(audit_path),
            'seal_sha256':sha256(seal_path),'candidate_review_sha256':sha256(review_path),
            'reference_release_protocol_sha256':seal['reference_release_protocol_sha256'],
            'public_task_inventory_sha256':seal['public_task_inventory_sha256'],
            'frozen_final_files':seal['frozen_file_digests'],'cohorts':cohorts,
            'development_gain_pp':result['gain_pp'],'development_interval_95_pp':result['gain_interval_95_pp'],
            'final_test_opened':False,'weights_public':False}


def release_decision(authorization, fresh_base, fresh_candidate, historical_base, historical_candidate, checks):
    """Apply the 022 release gate to audited final campaign results and registered IDs."""
    if authorization['approved'] is not True or authorization['stage']!='final_evaluation':
        raise ValueError('An approved development decision is required')
    protocol=REFERENCE/'protocol.json'
    inventory_path=REFERENCE/'results/2026-11-12/task-inventory.json'
    if (authorization['reference_release_protocol_sha256']!=sha256(protocol)
            or authorization['public_task_inventory_sha256']!=sha256(inventory_path)):
        raise ValueError('The final decision must use the authorized sealed inventory')
    inventory = read(inventory_path)
    spec = importlib.util.spec_from_file_location('release022', REFERENCE/'decision.py')
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    roles = ('fresh_final_not_selection', 'historical_final_regression_not_selection')
    comparisons = []
    for role,base,candidate in zip(roles,(fresh_base,historical_base),(fresh_candidate,historical_candidate)):
        wanted = [r['id'] for r in inventory if r.get('evaluation_role')==role]
        comparisons.append(rules.paired(base,candidate,wanted))
    decision = rules.release_gate(*comparisons, checks, read(protocol))
    return {**decision,'model_id':authorization['model_id'],'base_revision':authorization['base_revision'],
            'checkpoint_sha256':authorization['checkpoint_sha256'],
            'development_audit_sha256':authorization['development_audit_sha256'],'weights_public':False}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','audit','seal','candidate-review','out'):
        p.add_argument('--'+name,required=True,type=Path)
    a = p.parse_args()
    value = authorize(a.protocol,a.audit,a.seal,a.candidate_review)
    with a.out.open('x',encoding='utf-8') as out:
        json.dump(value,out,indent=2)
        out.write('\n')
    print(json.dumps({'approved':value['approved'],'stage':value['stage'],'final_test_opened':False}))
