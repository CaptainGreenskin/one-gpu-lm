"""Retain all 031 records once and admit executed reference-guided episodes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.training_encoding import encode_training_example
from .evaluation_rollout import normalization
from .historic_plan import native, plan
from .historic_policy import initial_messages, feedback
from .historic_admit import semantic_failure
from .mixed_inputs import admit as admit_parent
from .tasks import digest, relative


def audit_episode(root, data, task, admission, episode, records, protocol, tokenizer, profile, directory):
    """Replay every delivered observation and student token boundary, without inference."""
    path=directory/'trajectory.json'
    if sha256(path)!=episode['trajectory_sha256']:raise ValueError('Expert trajectory changed')
    trace=json.loads(path.read_text());budget=protocol['budget'];policy=protocol['agent_policy']
    messages=initial_messages(task,policy,budget)
    counter=lambda target:encode_training_example({'messages':messages,'target':target},tokenizer,profile)['supervised_tokens']
    expected=plan(task,admission,counter,budget)
    if (trace['plan']!=expected or not trace['complete'] or not trace['grade']['passed']
            or not trace['submitted'] or not trace['verified_final_source'] or trace['model_calls']!=0
            or trace['actor_seconds']>budget['max_task_seconds'] or len(trace['steps'])!=len(records)
            or len(records)!=expected['actions']):raise ValueError('Incomplete or changed expert episode')
    norm=normalization(root);selected=[]
    for number,(action,step,record) in enumerate(zip(expected['commands'],trace['steps'],records),1):
        request_path=directory/f'planned-action-{number:03d}.json';tool_path=directory/f'tool-{number:03d}.json'
        request=json.loads(request_path.read_text());tool=json.loads(tool_path.read_text());choice={}
        def encode(value):
            choice['messages']=value
            return encode_prefix(tokenizer,value,profile['chat_template_kwargs'],policy['format'])[0]
        prefix,context=prepare_context(messages,encode,budget['max_input_tokens'],policy['context'])
        target=native(action['command']);provenance=record['provenance']
        if (request!={'messages':messages,'selected_messages':choice['messages'],'context':context,'target':target,'model_request':False}
                or request['selected_messages']!=record['messages'] or record['target']!=target
                or step['call']!=number or step['kind']!=action['kind'] or step['command']!=action['command']
                or sha256(request_path)!=step['planned_action_sha256'] or sha256(tool_path)!=step['tool_sha256']
                or provenance['trajectory_sha256']!=sha256(path) or provenance['call']!=number
                or provenance['producer_kind']!='upstream_reference_guided' or provenance['model_generated'] is not False
                or provenance['split']!='train' or task['split']!='training' or provenance['executed'] is not True
                or provenance['source_id']!=task['source_id'] or provenance['repository']!=task['repository']
                or provenance['source_manifest_sha256']!=sha256(data/'sources.json')):
            raise ValueError('Expert action or provenance mismatch')
        valid=semantic_failure(tool) if action['expected']=='semantic_failure' else tool['returncode']==0 and tool['reason']=='ok'
        if not valid or not step['expected_result_observed']:raise ValueError('Expert feedback did not match real execution')
        encoded=encode_training_example(record,tokenizer,profile)
        receipt={'tokens':len(encoded['ids']),'supervised_tokens':encoded['supervised_tokens'],
                 'prefix_sha256':digest(json.dumps(prefix,separators=(',',':')).encode())}
        if (receipt!=record['encoding_receipt'] or encoded['ids'][:len(prefix)]!=prefix
                or len(encoded['ids'])>profile['training']['max_sequence_length']
                or encoded['supervised_tokens']>budget['max_generated_tokens_per_call']):
            raise ValueError('Expert student encoding changed')
        selected.append({'id':record['id'],'episode_id':record['episode_id'],'provenance':provenance,
                         'record_group_kind':'reference_guided_whole_episode',**encoded})
        delivered,normal=feedback(tool['output'],norm,budget['max_tool_output_chars'])
        if normal!=step['normalization']:raise ValueError('Delivered expert feedback changed')
        messages.append({'role':'assistant','content':target,'extra':{'actions':[parse_action(target)]}})
        messages.append({'role':'user','content':f"Exit code: {tool['returncode']}\n{delivered}",
                         'extra':{'tool':{k:v for k,v in tool.items() if k!='output'}}})
    if (messages!=trace['messages'] or sum(r['supervised_tokens'] for r in selected)!=episode['label_tokens']
            or sum(r['supervised_tokens'] for r in selected)>budget['max_total_generated_tokens']
            or trace['steps'][-2]['source_after']!=trace['steps'][-1]['source_after']):
        raise ValueError('Whole expert episode identity differs')
    return selected


def admit(root, protocol, profile, tokenizer):
    root=Path(root);mix=protocol['historic_mix']
    read=lambda name:json.loads((root/relative(name)).read_text())
    parent_protocol=read(mix['parent_protocol'])
    for name,value in parent_protocol['frozen_files'].items():
        if sha256(root/relative(name))!=value:raise ValueError('031 parent input changed')
    parent,parent_meta,parent_rows=admit_parent(root,parent_protocol,profile,tokenizer)
    expert=read(mix['expert_protocol']);run=root/relative(expert['run_directory']);status=read(expert['run_directory']+'/status.json')
    for name,value in expert['frozen_files'].items():
        if sha256(root/relative(name))!=value:raise ValueError('Frozen expert input changed')
    if (not status['complete'] or not status['data_gate_passed'] or not status['frozen_files_unchanged']
            or status['protocol_sha256']!=sha256(root/relative(mix['expert_protocol']))):
        raise ValueError('Expert collection gate or identity failed')
    manifest=read(expert['run_directory']+'/private/export/manifest.json');data_path=run/'private/export/data.jsonl'
    if (not manifest['passed'] or manifest['data_sha256']!=sha256(data_path)
            or status['export_manifest_sha256']!=sha256(run/'private/export/manifest.json')):
        raise ValueError('Expert export changed')
    data=root/relative(expert['data_directory']);tasks={t['id']:t for t in read(expert['data_directory']+'/tasks.json')['tasks']}
    admissions={r['id']:r for r in read(expert['data_directory']+'/admission.json')['rows']}
    rows=[json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    episodes=manifest['episodes'];new=[];seen=set()
    if {e['task_id'] for e in episodes}!={r['id'] for r in status['rows'] if r['admitted']}:
        raise ValueError('Eligible expert episode set changed')
    for episode in episodes:
        if episode['episode_id'] in seen:raise ValueError('Duplicate expert episode')
        seen.add(episode['episode_id']);task=tasks[episode['task_id']]
        records=[r for r in rows if r['episode_id']==episode['episode_id']]
        if len(records)!=episode['actions']:raise ValueError('Partial expert episode')
        new.extend(audit_episode(root,data,task,admissions[task['id']],episode,records,expert,tokenizer,profile,run/'private/tasks'/task['id']))
    if len(new)!=len(rows) or len(rows)!=manifest['positive_action_records']:raise ValueError('Orphan or missing expert record')
    combined=parent+new
    if len({r['id'] for r in combined})!=len(combined):raise ValueError('Duplicate mixed training action')
    combined.sort(key=lambda r:hashlib.sha256(('025-student-v1\0'+r['id']).encode()).hexdigest())
    metadata={'parent_031':parent_meta,'new_expert_episodes':len(episodes),'new_multifile_episodes':manifest['multifile_episodes'],
        'new_expert_action_records':len(new),'full_episodes':parent_meta['full_episodes']+len(episodes),
        'corrective_fragments':parent_meta['corrective_fragments'],'selected_records':len(combined),
        'unique_tasks':len({r['provenance']['task_id'] for r in combined}),
        'supervised_tokens':sum(r['supervised_tokens'] for r in combined),
        'record_groups':dict(Counter(r['record_group_kind'] for r in combined)),
        'oversampling':False,'target_truncation':False,'new_autonomous_teacher_rollouts':0,
        'all_031_records_retained_once':True,'development_answers_added_to_training':0}
    return combined,metadata,parent_rows+rows


def select(root,protocol,profile,tokenizer,rows,manifest):
    selected,metadata,expected=admit(root,protocol,profile,tokenizer)
    if rows!=expected or manifest['historic_selection']!=metadata:raise ValueError('Mixed export changed')
    return selected,metadata
