"""Bind the development decision to a checkpoint and the existing final-test seal."""
import argparse
from collections import Counter
from hashlib import sha256 as hash_bytes
import importlib.util
import json
from pathlib import Path
import re

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.verified.first_action_loss import fingerprint
from analysis import schedule
from report import reproduce_analysis


REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO/'experiments/022-action-repair'
WEIGHT_FILES = ('adapter_config.json', 'adapter_model.safetensors')
COMPARISON_RULES = ('pretrained_model','base_revision','task_count','language_counts','repetitions','rollout_count',
                    'block_size','evaluation_budget','pipeline_limit_seconds','recovery_accounting_seconds','gate',
                    'optimizer_updates_during_development','final_test_opened','weight_publication_allowed_by_development')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _read_digest(path):
    payload = path.read_bytes()
    return json.loads(payload),hash_bytes(payload).hexdigest()


def _development_result(protocol, audit, protocol_hash):
    budget = protocol['evaluation_budget']
    if (audit['passed'] is not True or audit['final_test_opened'] is not False
            or audit['weights_public'] is not False or audit['optimizer_updates']!=0
            or audit['protocol_sha256']!=protocol_hash
            or audit['verified_requests']!=sum(row['attempted_calls'] for row in audit['rows'])
            or any(not 1<=row['attempted_calls']<=budget['max_model_calls']
                   or not 0<=row['generated_tokens']<=budget['max_total_generated_tokens']
                   or not 1<=row['opening_input_tokens']<=budget['max_input_tokens'] for row in audit['rows'])):
        raise ValueError('Require complete audited development results for the unchanged protocol')
    result = reproduce_analysis(audit, protocol)
    if result['received_runs']!=protocol['rollout_count']:
        raise ValueError('Stored development decision differs from independent recomputation')
    return result


def authorize(protocol_path, audit_path, seal_path, review_path):
    """Authorize the original 039 attempt under its registered incident amendments."""
    protocol,protocol_hash = _read_digest(protocol_path)
    audit,audit_hash = _read_digest(audit_path)
    design = read(Path(__file__).with_name('design.json'))
    if any(protocol.get(key)!=value for key,value in design.items()):
        raise ValueError('Require the registered development design')
    result = _development_result(protocol,audit,protocol_hash)
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
    return _seal_permission(protocol,protocol_hash,audit_hash,result,seal_path,review_path)


def _family(name):
    return name.split(':',1)[0].casefold()


def _seal_permission(protocol, protocol_hash, audit_hash, result, seal_path, review_path, *, fresh_comparison=False):
    if result['development_gate_passed'] is not True:
        return {'approved':False,'stage':'development_gate','checks':result.get('checks',{}),
                'development_protocol_sha256':protocol_hash,'final_test_opened':False,'weights_public':False}

    reference_path = REFERENCE/'protocol.json'
    reference = read(reference_path)
    review,review_hash = _read_digest(review_path)
    if (review['audited_checkpoint_matches'] is not True or review['eligible_for_existing_scope_of_development'] is not True
            or review['model_id']!=protocol['pretrained_model'] or review['revision']!=protocol['base_revision']
            or review['model_id']!=reference['model_id'] or review['revision']!=reference['revision']
            or review['checkpoint_files_sha256']!=protocol['all_adapter_files_sha256']
            or review['weights_public'] or review['final_test_opened']):
        raise ValueError('The reviewed adapter is different from the measured candidate')

    gate = reference['release_gate']
    inventory_path = REFERENCE/'results/2026-11-12/task-inventory.json'
    inventory = read(inventory_path)
    if fresh_comparison:
        held_out = {_family(row['repository']) for row in inventory
                    if row.get('evaluation_role')=='fresh_final_not_selection'}
        development = {_family(name) for name in protocol['families']}
        trained = {_family(name) for name in review['sources']}
        if not trained or development & trained or held_out & (development | trained):
            raise ValueError('Training, independent development and fresh final families must stay separate')

    seal,seal_hash = _read_digest(seal_path)
    if (seal['reference_release_protocol_sha256']!=sha256(reference_path)
            or seal['public_task_inventory_sha256']!=sha256(inventory_path)
            or seal['development_protocol_sha256']!=protocol_hash
            or seal['release_gate']!=gate or seal['fresh_family_overlap']
            or seal['final_campaign_directories']!=0 or seal['final_test_opened'] is not False
            or seal['private_task_schema_read'] is not False or seal['authorized_model_scoring'] is not False):
        raise ValueError('Require the verified unreleased final-task seal for this development protocol')
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

    return {'approved':True,'stage':'final_evaluation','model_id':review['model_id'],'base_revision':review['revision'],
            'checkpoint_sha256':{name:review['checkpoint_files_sha256'][name] for name in WEIGHT_FILES},
            'development_protocol_sha256':protocol_hash,'development_audit_sha256':audit_hash,
            'seal_sha256':seal_hash,'candidate_review_sha256':review_hash,
            'reference_release_protocol_sha256':seal['reference_release_protocol_sha256'],
            'public_task_inventory_sha256':seal['public_task_inventory_sha256'],
            'frozen_final_files':seal['frozen_file_digests'],'cohorts':cohorts,
            'development_gain_pp':result['gain_pp'],'development_interval_95_pp':result['gain_interval_95_pp'],
            'final_test_opened':False,'weights_public':False}


def comparison_registration(protocol_path, publication_path, openings_path):
    """Check the frozen prompt, model, budget and order before starting the evaluation."""
    protocol,protocol_hash = _read_digest(protocol_path)
    publication = read(publication_path)
    if (publication['protocol_sha256']!=protocol_hash or publication['anonymous_files_verified'] is not True
            or publication['ci_conclusion']!='success' or not re.fullmatch(r'[0-9a-f]{40}',publication['commit'])):
        raise ValueError('Require this comparison’s public preregistration and passing CI')
    design = read(Path(__file__).with_name('design.json'))
    if (any(protocol.get(k)!=design[k] for k in COMPARISON_RULES)
            or protocol.get('comparison_kind')!='independent_adapter_comparison'
            or sorted(protocol['families'].values())!=sorted(design['families'].values())
            or {_family(name) for name in design['families']} & {_family(name) for name in protocol['families']}):
        raise ValueError('Require a new six-family panel, original budget and unchanged gain gate')

    openings,opening_hash = _read_digest(openings_path)
    selected = protocol['selected_opening']
    expected_openings = {row['task_number']:row for row in openings['rows']}
    if (selected not in ('original','editable_source') or protocol['opening_preflight_sha256']!=opening_hash
            or protocol['cohort_sha256']!=openings['cohort_sha256']
            or protocol['evaluation_profile_sha256']!=openings['profile_sha256']
            or openings['budget']!=protocol['evaluation_budget']
            or (openings['model_id'],openings['revision'])!=(protocol['pretrained_model'],protocol['base_revision'])
            or openings['model_generations']!=0 or openings['optimizer_updates']!=0 or openings['final_test_opened'] is not False
            or len(openings['rows'])!=protocol['task_count'] or len(expected_openings)!=protocol['task_count']
            or Counter(row['family'] for row in expected_openings.values())!=protocol['families']):
        raise ValueError('Expected the verified score-independent opening for the complete panel')
    order = schedule(list(expected_openings),protocol)
    if fingerprint(order)!=protocol['schedule_sha256']:
        raise ValueError('The paired trial schedule differs from the preregistration')
    return protocol,protocol_hash,openings,opening_hash,order


def verify_comparison(protocol_path, controller_path, audit_path, publication_path, openings_path):
    """Recompute the new development decision using development files alone."""
    protocol,protocol_hash,openings,opening_hash,order = comparison_registration(
        protocol_path,publication_path,openings_path)
    selected = protocol['selected_opening']
    expected_openings = {row['task_number']:row for row in openings['rows']}
    controller,controller_hash = _read_digest(controller_path)
    audit,audit_hash = _read_digest(audit_path)
    if (controller['stage']!='completed' or controller['complete'] is not True
            or controller['backend_stopped'] is not True or controller['frozen_inputs_unchanged'] is not True
            or controller['optimizer_updates']!=0 or controller['weights_public'] is not False
            or controller['final_test_opened'] is not False or controller['protocol_sha256']!=protocol_hash
            or controller['schedule_sha256']!=protocol['schedule_sha256']
            or controller['deadline_epoch']!=controller['started_epoch']+protocol['pipeline_limit_seconds']
            or not controller['started_epoch']<controller['finished_epoch']<=controller['deadline_epoch']
            or audit['controller_sha256']!=controller_hash or audit['opening_preflight_sha256']!=opening_hash
            or audit['started_epoch']!=controller['started_epoch'] or audit['finished_epoch']!=controller['finished_epoch']
            or len(controller['runs'])!=len(order) or len(audit['rows'])!=len(order)):
        raise ValueError('Require a complete development controller and its exact replay audit')

    served = audit['served_checkpoints']
    files = protocol['all_adapter_files_sha256']
    if (not set(WEIGHT_FILES)<=files.keys() or any(not re.fullmatch(r'[0-9a-f]{64}', h) for h in files.values())
            or served!={'base':{'adapter_files_sha256':{}, 'profile_sha256':openings['profile_sha256']},
                        'candidate':{'adapter_files_sha256':files, 'profile_sha256':openings['profile_sha256']}}):
        raise ValueError('The audited base, adapter or shared serving profile differs')
    for item, run, row in zip(order,controller['runs'],audit['rows']):
        cell = expected_openings[item['task_number']]
        first = cell['openings'][selected]
        order_keys = ('run_id','task_number','repetition','block')
        if (any(run[k]!=item[k] or row[k]!=item[k] for k in order_keys)
                or run['contrast_arm']!=item['arm'] or row['arm']!=item['arm'] or row['family']!=cell['family']
                or row['opening_input_sha256']!=first['input_sha256'] or row['opening_input_tokens']!=first['input_tokens']):
            raise ValueError('A paired arm used a different task, order or opening')
    return protocol,protocol_hash,audit_hash,_development_result(protocol,audit,protocol_hash)


def authorize_comparison(protocol_path, controller_path, audit_path, publication_path, openings_path, seal_path, review_path):
    """Authorize a fresh, registered adapter comparison backed by its own replay audit."""
    protocol,protocol_hash,audit_hash,result = verify_comparison(protocol_path,controller_path,audit_path,
                                                               publication_path,openings_path)
    return _seal_permission(protocol,protocol_hash,audit_hash,result,seal_path,review_path,fresh_comparison=True)


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
    for name in ('controller','publication','openings'):
        p.add_argument('--'+name,type=Path)
    a = p.parse_args()
    evidence = (a.controller,a.publication,a.openings)
    if any(evidence) and not all(evidence):
        p.error('A fresh comparison requires --controller, --publication and --openings together')
    value = (authorize_comparison(a.protocol,a.controller,a.audit,a.publication,a.openings,a.seal,a.candidate_review)
             if all(evidence) else authorize(a.protocol,a.audit,a.seal,a.candidate_review))
    with a.out.open('x',encoding='utf-8') as out:
        json.dump(value,out,indent=2)
        out.write('\n')
    print(json.dumps({'approved':value['approved'],'stage':value['stage'],'final_test_opened':False}))
