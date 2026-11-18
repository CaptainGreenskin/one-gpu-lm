"""Execute reference-guided expert plans and export whole verified episodes.

This is supervised expert replay, not a teacher model rollout. Reference patch
knowledge and localization are explicitly recorded; every tool result is real.
"""
import argparse
import copy
import json
from pathlib import Path
import time
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.training_encoding import encode_training_example
from .evaluation_rollout import atomic, normalization
from .historic_admit import semantic_failure
from .historic_plan import plan, native
from .historic_policy import initial_messages, feedback
from .historic_tasks import prepare, grade, validate_split
from .sandbox import Sandbox
from .tasks import digest, files, relative


def execute_episode(root, data, task, source, admission, sandbox, tokenizer, profile, protocol, out):
    if out.exists():raise ValueError('Preserve each expert attempt')
    if task['split'] != 'training' or not source['approved_for']['training']:
        raise ValueError('Only approved training history may be exported')
    budget = protocol['budget']; policy = protocol['agent_policy']
    messages = initial_messages(task, policy, budget)
    def count(target):
        return encode_training_example({'messages':messages,'target':target},tokenizer,profile)['supervised_tokens']
    planned = plan(task,admission,count,budget)
    out.mkdir(parents=True)
    snapshot = data/'snapshots'/task['source_id']
    before = prepare(snapshot,out/'workspace',source,task)
    norm = normalization(root); steps = []; records = []; total_tokens = 0
    started = time.monotonic(); epoch = time.time(); episode_id = protocol['episode_namespace']+':'+task['id']
    trace = {'schema_version':1,'task_id':task['id'],'source_id':task['source_id'],
        'producer_kind':'upstream_reference_guided','oracle_localization':True,
        'reference_patch_used_for_targets':True,'model_calls':0,'split':'training',
        'complete':False,'plan':planned,'steps':steps,'started_epoch':epoch}
    atomic(out/'trajectory.json',trace)
    for call, action in enumerate(planned['commands'],1):
        if time.monotonic()-started >= budget['max_task_seconds']:
            raise ValueError('Expert episode exceeded actor wall time')
        selected = {}
        def encode(value):
            selected['messages'] = copy.deepcopy(value)
            return encode_prefix(tokenizer,value,profile['chat_template_kwargs'],policy['format'])[0]
        prefix, context = prepare_context(messages,encode,budget['max_input_tokens'],policy['context'])
        target = native(action['command'])
        record = {'id':episode_id+':'+str(call),'episode_id':episode_id,
            'messages':selected['messages'],'target':target}
        encoded = encode_training_example(record,tokenizer,profile)
        if (encoded['ids'][:len(prefix)] != prefix or len(encoded['ids']) > profile['training']['max_sequence_length']
                or encoded['supervised_tokens'] > budget['max_generated_tokens_per_call']):
            raise ValueError('Whole expert episode exceeds actual prefix/target encoding budget')
        total_tokens += encoded['supervised_tokens']
        if total_tokens > budget['max_total_generated_tokens']:
            raise ValueError('Expert label budget exceeded')
        request = {'messages':copy.deepcopy(messages),'selected_messages':selected['messages'],
                   'context':context,'target':target,'model_request':False}
        path = out/f'planned-action-{call:03d}.json';write_json(path,request)
        action_parsed = parse_action(target)
        if action_parsed['command'] != action['command']:raise ValueError('Planned tool format changed command')
        source_before = {n:files(out/'workspace').get(n) for n in task['editable_files']}
        sandbox.timeout = max(1,min(budget['max_tool_seconds'],int(budget['max_task_seconds']-(time.monotonic()-started))))
        result = sandbox.execute(out/'workspace',action['command']);write_json(out/f'tool-{call:03d}.json',result)
        valid = result['returncode']==0 and result['reason']=='ok'
        if action['expected']=='semantic_failure':valid=semantic_failure(result)
        delivered, receipt = feedback(result['output'],norm,budget['max_tool_output_chars'])
        source_after = {n:files(out/'workspace').get(n) for n in task['editable_files']}
        step = {'call':call,'kind':action['kind'],'command':action['command'],
                'target_tokens':encoded['supervised_tokens'],'input_tokens':len(prefix),
                'input_tokens_sha256':digest(json.dumps(prefix,separators=(',',':')).encode()),
                'planned_action_sha256':sha256(path),'tool_sha256':sha256(out/f'tool-{call:03d}.json'),
                'expected_result_observed':valid,'normalization':receipt,
                'source_before':source_before,'source_after':source_after}
        steps.append(step)
        messages.append({'role':'assistant','content':target,'extra':{'actions':[action_parsed]}})
        messages.append({'role':'user','content':f"Exit code: {result['returncode']}\n{delivered}",
                         'extra':{'tool':{k:v for k,v in result.items() if k!='output'}}})
        trace['messages']=messages;atomic(out/'trajectory.json',trace)
        if not valid:raise ValueError('Executed expert action did not produce its declared result')
        record['provenance'] = {'split':'train','task_id':task['id'],'repository':task['repository'],
            'source_id':task['source_id'],'kind':action['kind'],'producer_kind':'upstream_reference_guided',
            'reference_patch_used_for_targets':True,'oracle_localization':True,'model_generated':False,
            'executed':True,'call':call,'planned_action_sha256':step['planned_action_sha256'],
            'tool_sha256':step['tool_sha256'],'source_manifest_sha256':sha256(data/'sources.json')}
        record['encoding_receipt'] = {'tokens':len(encoded['ids']),'supervised_tokens':encoded['supervised_tokens'],
                                      'prefix_sha256':step['input_tokens_sha256']}
        records.append(record)
    seconds = time.monotonic()-started
    if seconds > budget['max_task_seconds']:raise ValueError('Expert wall-time budget exceeded')
    if steps[-2]['source_after'] != steps[-1]['source_after']:
        raise ValueError('Submission changed verified source')
    def independent(workspace, command):
        sandbox.timeout=budget['max_tool_seconds'];value=sandbox.execute(workspace,command)
        write_json(out/'grader-output.json',value);return value
    result = grade(snapshot,out/'grader',source,task,out/'workspace',before,independent)
    if not result['passed']:raise ValueError('Fresh protected expert regrade failed')
    trace.update(complete=True,grade=result,actor_seconds=seconds,finished_epoch=time.time(),
                 label_tokens=total_tokens,submitted=True,verified_final_source=True)
    atomic(out/'trajectory.json',trace)
    for record in records:record['provenance']['trajectory_sha256']=sha256(out/'trajectory.json')
    episode = {'episode_id':episode_id,'task_id':task['id'],'repository':task['repository'],
        'source_id':task['source_id'],'multifile':admission['multifile'],
        'essential_files':admission['essential_files'],'actions':len(records),
        'label_tokens':total_tokens,'actor_seconds':seconds,'trajectory_sha256':sha256(out/'trajectory.json'),
        'first_edit_call':planned['first_edit_call'],'verified_and_submitted':True,
        'producer_kind':'upstream_reference_guided','model_generations':0}
    return records, episode


def run(root, protocol_path):
    from transformers import AutoTokenizer
    root=Path(root);protocol=json.loads(Path(protocol_path).read_text())
    for name, expected in protocol['frozen_files'].items():
        if sha256(root/relative(name))!=expected:raise ValueError('Frozen expert input changed: '+name)
    out=root/relative(protocol['run_directory'])
    if out.exists():raise ValueError('Expert collection already exists')
    out.mkdir(parents=True);(out/'private/tasks').mkdir(parents=True)
    profile=json.loads((root/relative(protocol['profile_path'])).read_text())
    runtime=json.loads((root/'private/parent-runtime.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(runtime['model'],local_files_only=True,trust_remote_code=False)
    data=root/relative(protocol['data_directory'])
    sources=json.loads((data/'sources.json').read_text())['sources'];validate_split(sources)
    by_id={s['source_id']:s for s in sources}
    tasks=json.loads((data/'tasks.json').read_text())['tasks']
    admissions={r['id']:r for r in json.loads((data/'admission.json').read_text())['rows']}
    runtime_map=json.loads((root/relative(protocol['runtime_map'])).read_text())
    runners={name:Sandbox(json.loads(Path(path).read_text()),protocol['budget']['max_tool_seconds']) for name,path in runtime_map.items()}
    state={'complete':False,'model_calls':0,'optimizer_updates':0,'final_test_opened':False,
           'protocol_sha256':sha256(protocol_path),'started_epoch':time.time(),'rows':[]}
    records=[];episodes=[]
    for task in tasks:
        row={'id':task['id'],'repository':task['repository'],'admitted':False}
        try:
            values,episode=execute_episode(root,data,task,by_id[task['source_id']],admissions[task['id']],
                runners[task['repository']],tokenizer,profile,protocol,out/'private/tasks'/task['id'])
            records.extend(values);episodes.append(episode);row.update(admitted=True,**episode)
        except (ValueError,RuntimeError) as error:
            row.update(reason=str(error),error_type=type(error).__name__)
        state['rows'].append(row);atomic(out/'status.json',state)
    gate=protocol['data_gate'];multi=[e for e in episodes if e['multifile']]
    passed=(len(episodes)>=gate['minimum_episodes'] and len(multi)>=gate['minimum_multifile_episodes']
            and len({e['repository'] for e in multi})>=gate['minimum_multifile_families']
            and len({e['repository'] for e in episodes})>=gate['minimum_families'])
    export=out/'private/export';export.mkdir()
    path=export/'data.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    manifest={'data_sha256':sha256(path),'protocol_sha256':sha256(protocol_path),'passed':passed,
        'full_episodes':len(episodes),'unique_tasks':len(episodes),'positive_action_records':len(records),
        'multifile_episodes':len(multi),'episodes':episodes,'supervised_tokens':sum(e['label_tokens'] for e in episodes),
        'maximum_record_tokens':max((r['encoding_receipt']['tokens'] for r in records),default=0),
        'producer_kind':'upstream_reference_guided','autonomous_teacher_rollouts':0,'raw_data_public':False}
    write_json(export/'manifest.json',manifest)
    state.update(complete=True,finished_epoch=time.time(),data_gate_passed=passed,
        export_manifest_sha256=sha256(export/'manifest.json'),
        frozen_files_unchanged=all(sha256(root/relative(n))==v for n,v in protocol['frozen_files'].items()))
    atomic(out/'status.json',state)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();result=run(args.root,args.protocol)
    print(json.dumps({k:result[k] for k in ['complete','data_gate_passed','frozen_files_unchanged']}))
