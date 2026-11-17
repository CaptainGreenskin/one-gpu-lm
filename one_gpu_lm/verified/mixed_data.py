"""Admit verified corrective fragments without relabeling them whole episodes."""
import hashlib
from one_gpu_lm.common.training_encoding import encode_training_example
from .decision_report import summarize
from .tasks import digest
import json


def select_fragments(rows, manifest, audit, collection_protocol, sources, profile, tokenizer,
                     encoder=encode_training_example):
    if not audit['passed'] or not summarize(audit, collection_protocol)['data_feasibility_gate_passed']:
        raise ValueError('Correction collection did not pass its frozen gate')
    eligible={r['task_id']:r for r in audit['rows'] if r['arm']=='correction' and r['qualified']}
    expected={(n,x['call']):x for n,r in eligible.items() for x in r['encoding']['encoded_actions']}
    approved={s['repository']:s for s in sources if s['split']=='training' and s['approved_for'].get('training') is True}
    if (len(rows)!=manifest['action_records'] or len(eligible)!=manifest['qualified_unique_tasks']
            or len({r['id'] for r in rows})!=len(rows)):
        raise ValueError('Fragment export counts or unique IDs differ')
    actual=set();selected=[]
    for row in rows:
        v=row['provenance'];key=(v['task_id'],v['call'])
        if key in actual or key not in expected:raise ValueError('Duplicate, extra or unaudited fragment action')
        parent=eligible[v['task_id']];receipt=expected[key]
        if (v['split']!='train' or v['arm']!='correction' or v['producer_kind']!='model'
                or v['model_role']!='base' or v['executed'] is not True
                or v['repository'] not in approved or v['repository']!=parent['result']['repository']
                or v['trajectory_sha256']!=parent['result']['trajectory_sha256']
                or v['kind']!=receipt['kind'] or v['state_id']!=parent['state_id']
                or v['target_start_call']!=parent['encoding']['target_start_call']
                or v['call']<v['target_start_call']
                or v.get('target_transform')!='verified_final_action_only_no_thought_labels'
                or row['encoding_receipt']!=receipt
                or receipt.get('export_mode')!='action_only_cross_mode'
                or receipt['student_profile_sha256']!=collection_protocol['reasoning_comparison']['student_profile_sha256']
                or '<think>' in row['target'] or '</think>' in row['target']):
            raise ValueError('Correction provenance or action-only boundary differs')
        encoded=encoder(row,tokenizer,profile)
        prefix=encoded['ids'][:encoded['prefix_tokens']]
        if (len(encoded['ids'])!=receipt['tokens'] or encoded['supervised_tokens']!=receipt['supervised_tokens']
                or len(encoded['ids'])>profile['training']['max_sequence_length']
                or encoded['supervised_tokens']<=0
                or digest(json.dumps(prefix,separators=(',',':')).encode())!=receipt['student_input_sha256']
                or len(encoded['ids'])!=len(encoded['labels'])):
            raise ValueError('Student encoding changed or does not fit without truncation')
        selected.append({'id':row['id'],'episode_id':'030-fragment:'+v['task_id'],
                         'record_group_kind':'corrective_fragment','provenance':v,**encoded})
        actual.add(key)
    if actual!=set(expected):raise ValueError('Missing eligible corrective actions')
    if sum(r['supervised_tokens'] for r in selected)!=manifest['supervised_tokens']:
        raise ValueError('Fragment supervision total differs')
    return selected


def combine(complete, fragments):
    rows=[dict(r,record_group_kind='whole_episode') for r in complete]+fragments
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Mixed action IDs collide')
    old={r['provenance']['task_id'] for r in complete};new={r['provenance']['task_id'] for r in fragments}
    metadata={'full_episodes':len({r['episode_id'] for r in complete}),
              'corrective_fragments':len({r['episode_id'] for r in fragments}),
              'original_action_records':len(complete),'corrective_action_records':len(fragments),
              'unique_tasks':len(old|new),'overlapping_task_ids':sorted(old&new),
              'selected_records':len(rows),'supervised_tokens':sum(r['supervised_tokens'] for r in rows),
              'target_truncation':False,'oversampling':False,'thought_labels':False}
    rows.sort(key=lambda r:hashlib.sha256(('025-student-v1\0'+r['id']).encode()).hexdigest())
    return rows,metadata
