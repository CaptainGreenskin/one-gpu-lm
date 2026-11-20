"""Replay recorded 022 final execution and grading without new model calls."""
import argparse
from hashlib import sha256 as hash_bytes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from one_gpu_lm.common.actions import INSTANCE, parse_action
from one_gpu_lm.common.agent_policy import system_prompt
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.tool_history import render_history
from one_gpu_lm.data.function_records import load_split
from one_gpu_lm.evaluation.environment import RepositoryEnvironment, repository_python_site
from one_gpu_lm.evaluation.grading import generated_files, grade
from one_gpu_lm.evaluation.sandbox import Sandbox, extract_code
from one_gpu_lm.evaluation.tasks import file_manifest, prepare

from final_campaign import ARMS, FINAL, SEALED_EXPERIMENT, SNAPSHOT_CHECK, frozen_task_ids
from final_campaign import promotion_from_finished_run, verify_sealed_files
from promotion import REFERENCE, read, release_decision


REPLAY_MODULES = ('common.actions','common.agent_policy','common.edit_progress','common.tokenization',
                  'common.tool_history','data.function_records','evaluation.environment','evaluation.grading',
                  'evaluation.sandbox','evaluation.tasks')


def json_sha256(value):
    return hash_bytes(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def opening_messages(task, policy):
    from jinja2 import Template
    initial = Template(INSTANCE).render(task=task['description'],editable_files=', '.join(task['editable_files']),
                                         regression_command=task['regression_command'])
    return [('system',Template(system_prompt(policy)).render()),('user',initial)]


def replay_trace(directory, result, original_editable, identity, budget, profile, tokenizer, opening):
    """Compare one checkpoint, transcript, token receipt and final workspace."""
    trajectory = directory/'trajectory.json'
    trace = read(trajectory)
    checkpoint = read(directory/'checkpoint.json')
    original = read(directory/'original.json')
    policy = identity['agent_policy']
    workspace = directory/'workspace'
    editable = tuple(original_editable)
    model = trace['local_model']
    steps = result['source_progress']
    require(read(directory/'result.json') == result and checkpoint['trajectory'] == trace
            and checkpoint['boundary'] == 'completed_step' and checkpoint['identity'] == identity
            and checkpoint['n_calls'] == result['calls'] and result['agent_policy'] == model['policy'] == policy
            and model['config'] == budget
            and result['generations'] == model['responses'] and result['backend_errors'] == model['backend_errors']
            and result['generated_tokens'] == model['generated_tokens']
            and steps == model['source_progress'] and len(steps) == result['tool_calls'] == trace['environment']['tool_calls']
            and all(original[name] == digest for name,digest in original_editable.items()),
            'Campaign result differs from its completed actor checkpoint')
    require(checkpoint['workspace_manifest'] == file_manifest(workspace),'Recorded workspace manifest changed')
    limit = budget['max_model_calls']
    state = {'changed_editable_files':0, 'invalid_editable_files':0,
             'editable_state_sha256':json_sha256(sorted(original_editable.items()))}
    messages = [m for m in trace['messages'] if m['role'] in ('system','user','assistant')]
    require([(m['role'],m['content']) for m in messages[:2]] == opening,
            'Opening messages differ from the frozen task')
    render_history(messages,native_roundtrip=True)
    used = observed = 0

    def verify_request(prefix, receipt):
        tracker = object.__new__(EditableProgress)
        tracker.snapshot = lambda:dict(state)
        packed, progress = tracker.prepare_request(prefix,used,limit,True)
        tokens, dialogue = encode_prefix(tokenizer,packed,profile['chat_template_kwargs'],policy['format'])
        require(receipt['progress'] == progress and receipt['input_tokens'] == len(tokens)
                and receipt['input_tokens_sha256'] == json_sha256(tokens) and receipt['dialogue'] == dialogue
                and receipt['context'] == {'policy':'full','raw_input_tokens':len(tokens),'packed_input_tokens':len(tokens),
                                          'dropped_rounds':0,'original_messages':len(packed),'packed_messages':len(packed)},
                'Stored model input or progress differs from replay')

    for index,message in enumerate(messages):
        if message['role'] == 'assistant':
            receipt = result['generations'][used]
            require(message['extra']['generation'] == receipt and receipt['adapter'] is identity['backend']['adapter']
                    and receipt['agent_policy'] == policy and 0 < receipt['input_tokens'] <= budget['max_input_tokens']
                    and 0 <= receipt['generated_tokens'] <= budget['max_generated_tokens_per_call'],
                    'Stored model output has the wrong identity or token budget')
            verify_request(messages[:index],receipt)
            try:
                action = parse_action(message['content'])
            except ValueError:
                require(receipt['action_format_valid'] is False and not message['extra'].get('actions'),
                        'Invalid action was logged as executable')
            else:
                require(receipt['action_format_valid'] is True and [action] == message['extra']['actions'],
                        'Executed action differs from the validated parser output')
            used += 1
        elif 'tool' in message.get('extra',{}):
            step = steps[observed]
            require(step['call_index'] == used,'Recorded source observation has the wrong call index')
            state = {k:v for k,v in step.items() if k != 'call_index'}
            observed += 1
    for error in result['backend_errors']:
        verify_request(messages,error)
        require(error['error'] == 'context_limit' and error['input_tokens'] > budget['max_input_tokens'],
                'A backend failure cannot be accounted for within the frozen budget')
        used += 1
    require(used == result['calls'] <= limit and len(result['generations'])+len(result['backend_errors']) == used
            and observed == len(steps)
            and sum(g['generated_tokens'] for g in result['generations']) == result['generated_tokens']
            and result['generated_tokens'] <= budget['max_total_generated_tokens']
            and EditableProgress(workspace,original,editable).snapshot() == state
            and all(sha256(workspace/name) == digest for name,digest in result.get('source_sha256',{}).items()),
            'Stored execution counts, budgets or submitted source differ')
    return {'requests_replayed':used, 'trajectory_sha256':sha256(trajectory)}


def replay_grade(sealed, runtime, task, directory, result, limit, scratch):
    """Independently rerun the frozen grader on a fresh copy of submitted source."""
    repository = task['repository']
    slug = repository.replace('/','--')
    snapshot = sealed/'sources'/slug
    original = read(directory/'original.json')
    rebuilt = prepare(snapshot,scratch/'original',task)
    require(file_manifest(rebuilt,exclude=generated_files(repository)) == original,
            'The original workspace no longer matches the admitted repository and mutation')
    node_modules = sealed/'node-deps'/slug/'node_modules'
    site = repository_python_site(sealed,runtime,repository)

    def environment(work):
        return RepositoryEnvironment(work,python_site=site,node_root=runtime['node_root'],
                                     node_modules=node_modules if node_modules.exists() else None,
                                     timeout=limit,output_chars=100000)

    actual = grade(task,directory/'workspace',snapshot,scratch/'grader',original,environment)
    require(all(result.get(key) == value for key,value in actual.items() if key != 'seconds'),
            'Independent protected grading differs from the stored score')
    return {'passed':actual['passed'], 'grader_log_sha256':sha256(scratch/'grader.log')
            if (scratch/'grader.log').is_file() else None}


def audit(root, sealed, seal_path, review_path):
    out = sealed/FINAL
    target = out/'execution-audit.json'
    require(not target.exists(),'Preserve existing final execution evidence')
    permission = promotion_from_finished_run(root,seal_path,review_path)
    require(permission['approved'] is True and permission == read(out/'promotion.json'),
            'A completed development promotion must match the launch receipt')
    control = out/'controller.json'
    state = read(control)
    protocol = verify_sealed_files(sealed)
    replay_sources = {}
    for name in REPLAY_MODULES:
        module = sys.modules['one_gpu_lm.'+name]
        relative = 'one_gpu_lm/'+name.replace('.','/')+'.py'
        replay_sources[relative] = sha256(Path(module.__file__))
    harness = Path(sys.modules['one_gpu_lm.evaluation.environment'].__file__).parent
    for name in ('ava.config.mjs','underscore_tests.mjs'):
        replay_sources['one_gpu_lm/evaluation/'+name] = sha256(harness/name)
    require(all(protocol['frozen_files'].get(name) == digest for name,digest in replay_sources.items()),
            'Running replay modules differ from the frozen parser, tokenizer or grader')
    reference_sha = permission['reference_release_protocol_sha256']
    budget,policy = protocol['agent'],protocol['variants']['shared']
    expected_stages = [name for arm in ARMS for name in (arm,'legacy-'+arm)]
    require(state['stage'] == 'scored_audit_pending' and state['complete'] is True
            and state['final_test_opened'] is True and state['weights_public'] is False
            and state['optimizer_updates'] == 0 and state['reference_protocol_sha256'] == reference_sha
            and state['execution_snapshots_verified'] is True and state['campaign_variant'] == 'shared'
            and state['development_audit_sha256'] == permission['development_audit_sha256']
            and state['adapter_files_sha256'] == permission['checkpoint_sha256']
            and state['started_epoch'] < state['finished_epoch'] <= state['deadline_epoch']
            and state['deadline_epoch'] == state['started_epoch']+protocol['budget']['max_gpu_pipeline_wall_seconds']
            and state['stages'] == expected_stages and state['telemetry']['thread_stopped'],
            'Final campaign is incomplete or inconsistent with its frozen budget')
    profile_path = sealed/SEALED_EXPERIMENT/'profile-action.json'
    profile = read(profile_path)
    runtime = read(sealed/'runtime.json')
    cpu = str(sealed/'.venv-agent/bin/python')
    env = dict(os.environ,PYTHONPATH=str(sealed),CUDA_VISIBLE_DEVICES='')
    subprocess.run([cpu,'-c',SNAPSHOT_CHECK],cwd=sealed,env=env,capture_output=True,timeout=600,check=True)
    source_command = 'import json; from one_gpu_lm.common.source import campaign_sources; print(json.dumps(campaign_sources()))'
    sources = json.loads(subprocess.check_output([cpu,'-c',source_command],cwd=sealed,env=env,timeout=60))
    require(all(protocol['frozen_files'].get(name) == digest for name,digest in sources.items()),
            'Running campaign implementations were outside the frozen protocol')
    ids = frozen_task_ids(sealed)
    tasks = {t['id']:t for t in read(sealed/'private/tasks.json')['tasks'] if t['split']=='test'}
    admitted = {r['id']:r['broken_sha256'] for r in read(sealed/'runs/admission/admission.json')['tasks'] if r['id'] in tasks}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)

    campaigns,artifacts,replayed = {},{},{}
    for arm in ARMS:
        selected = arm=='final-selected'
        artifact = out/('artifact-'+arm+'.json')
        health = read(artifact)
        hashes = {'adapter_model.safetensors':permission['checkpoint_sha256']['adapter_model.safetensors']} if selected else {}
        require(health['ready'] is True and health['adapter'] is selected and health['adapter_sha256'] == hashes
                and health['model_id'] == permission['model_id'] and health['revision'] == permission['base_revision']
                and health['profile_sha256'] == sha256(profile_path),'Final backend differs from registered weights')
        campaign_path = out/arm/'campaign.json'
        campaign = read(campaign_path)
        require(campaign['complete'] is True and [r['id'] for r in campaign['results']] == ids
                and campaign['identity'] == {'ids':ids,'variant':'shared','protocol_sha256':reference_sha,
                                            'tasks_sha256':protocol['task_spec_sha256'],
                                            'admission_sha256':protocol['admission_sha256'],'source_files':sources},
                'Final campaign task or implementation inventory differs')
        rows = []
        for row in campaign['results']:
            task = tasks[row['id']]
            require(row['split']=='test' and row['variant']=='shared' and row['repository']==task['repository']
                    and row['family']==task['family'],'Final task identity differs from its frozen cohort')
            directory = out/arm/'private'/row['id']
            identity = {'id':row['id'],'variant':'shared','agent_policy':policy,'backend':health,
                        'task_spec_sha256':protocol['task_spec_sha256'],'protocol_sha256':reference_sha}
            editable = {name:admitted[row['id']][name] for name in task['editable_files']}
            evidence = replay_trace(directory,row,editable,identity,budget,profile,tokenizer,opening_messages(task,policy))
            with tempfile.TemporaryDirectory(prefix='final-audit-',dir=out/'private') as temp:
                score = replay_grade(sealed,runtime,task,directory,row,budget['max_tool_seconds'],Path(temp))
            rows.append({'id':row['id'],**evidence,**score})
        campaigns[arm] = campaign
        artifacts[arm] = {'campaign_sha256':sha256(campaign_path),'served_artifact_sha256':sha256(artifact)}
        replayed[arm] = rows

    legacy_check = REFERENCE/'postrun_integrity.py'
    spec = importlib.util.spec_from_file_location('final_legacy022',legacy_check)
    legacy_auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_auditor)
    legacy_tasks_file = sealed/'private/legacy-functions.jsonl'
    legacy_tasks = load_split(legacy_tasks_file,'test')
    legacy = legacy_auditor.check_legacy_inputs(out,legacy_tasks,sha256(legacy_tasks_file),
                                                tokenizer,profile['chat_template_kwargs'])
    for arm in ARMS:
        folder = out/('legacy-'+arm)
        sandbox = Sandbox(Path(runtime['node_root'])/'bin/node')
        for task,row in zip(legacy_tasks,read(folder/'result.json')['results']):
            output = (folder/'private'/(task['id']+'.txt')).read_text(encoding='utf-8')
            replayed_score = sandbox.score(task,extract_code(output))
            require(all(row[key]==value for key,value in replayed_score.items()),'Legacy function score failed replay')
        legacy[arm]['scores_replayed'] = len(legacy_tasks)

    def cohort(arm, label):
        return {'complete':True,'results':[r for r in campaigns[arm]['results']
                                          if tasks[r['id']]['evaluation_role']==label]}

    fresh,historical = ('fresh_final_not_selection','historical_final_regression_not_selection')
    fresh_base,fresh_candidate = (cohort(arm,fresh) for arm in ARMS)
    historical_base,historical_candidate = (cohort(arm,historical) for arm in ARMS)
    checks = {'legacy_passes':read(out/'legacy-final-selected/result.json')['passed'],'all_execution_audits_pass':True}
    decision = release_decision(permission,fresh_base,fresh_candidate,historical_base,historical_candidate,checks)
    value = {'schema_version':1,'execution_audit_passed':True,'model_generations':0,
             'final_test_opened':True,'weights_public':False,'controller_sha256':sha256(control),
             'promotion_sha256':sha256(out/'promotion.json'),'adapter_files_sha256':permission['checkpoint_sha256'],
             'replay_implementations':replay_sources,'audit_source_sha256':sha256(Path(__file__)),
             'legacy_audit_source_sha256':sha256(legacy_check),'artifacts':artifacts,'repository_replay':replayed,
             'legacy_replay':legacy,'checks':checks,'release_decision_with_execution_evidence':decision}
    atomic_json(target,value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('development-root','sealed-root','seal','candidate-review'):
        p.add_argument('--'+key,required=True,type=Path)
    a = p.parse_args()
    value = audit(a.development_root.resolve(),a.sealed_root.resolve(),a.seal,a.candidate_review)
    print(json.dumps({k:value[k] for k in ('execution_audit_passed','final_test_opened','weights_public')}))


if __name__ == '__main__':
    os.umask(0o077)
    main()
