"""Export audited, actually executed training episodes; never synthesize repairs.

The output is private. An inspection export can contain fewer than 100 episodes;
its manifest explicitly prohibits treating action rows as complete trajectories
or entering candidate training before the registered full-cycle data gate.
"""
import argparse
import json
from pathlib import Path
from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.training_encoding import encode_training_example
from one_gpu_lm.common.receipts import sha256, write_json
from .tasks import digest, relative
from .guard_audit import target_is_executed


def inventory(episodes, records):
    ids = [e['episode_id'] for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate full episode')
    if any(r['episode_id'] not in set(ids) for r in records):
        raise ValueError('Orphan action record')
    if any(not any(r['episode_id'] == e for r in records) for e in ids):
        raise ValueError('Empty episode')
    return {'full_episodes': len(episodes), 'unique_tasks': len({e['task_id'] for e in episodes}),
            'positive_action_records': len(records),
            'minimum_100_full_episodes_met': len(episodes) >= 100,
            'candidate_training_authorized_by_this_export': False}


def run(root, protocol_path, out):
    from transformers import AutoTokenizer
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError('Preserve previous export')
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get('collection_policy') != 'explicit_submission_guard_v1':
        raise ValueError('This exporter requires explicit execution receipts')
    for name, expected in protocol['frozen_files'].items():
        if sha256(root/relative(name)) != expected:
            raise ValueError('Collection input changed before export: '+name)
    run_dir = root / relative(protocol['run_directory'])
    audit_path = run_dir / 'audit.json'
    audit = json.loads(audit_path.read_text())
    controller = json.loads((run_dir / 'controller.json').read_text())
    if (not controller['complete'] or not controller['backend_stopped']
            or not controller['frozen_files_unchanged'] or not audit['decision']['passed']
            or audit['protocol_sha256'] != sha256(protocol_path)
            or controller['protocol_sha256'] != sha256(protocol_path)):
        raise ValueError('Complete, unchanged and audited collection is required')
    data = root / 'private/pilot-data'
    tasks = {t['id']: t for t in json.loads((data/'tasks.json').read_text())['tasks']}
    sources = {s['repository']: s for s in json.loads((data/'sources.json').read_text())['sources']}
    profile = json.loads((root/'experiments/025-verified-distillation/teacher-profile.json').read_text())
    runtime = json.loads((root/'private/parent-runtime.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'], local_files_only=True, trust_remote_code=False)
    records, episodes = [], []
    for row in audit['rows']:
        if not row['eligible_for_sft']:
            continue
        task = tasks[row['id']]; source = sources[task['repository']]
        if task['split'] != 'training' or source['approved_for'].get('training') is not True:
            raise ValueError('Non-training or unapproved source')
        directory = run_dir/'private/tasks'/row['id']; path = directory/'trajectory.json'
        if sha256(path) != row['trajectory_sha256']:
            raise ValueError('Audited trajectory changed')
        trace = json.loads(path.read_text())
        if not trace['complete'] or not trace['eligible_verified_trajectory']:
            raise ValueError('Incomplete or unverified full episode')
        episode_id = run_dir.name + ':' + row['id']
        encoded_calls = {r['call']: r for r in row['encoded_actions']}
        actual_calls = {s['call'] for s in trace['steps'] if target_is_executed(s)}
        if actual_calls != set(encoded_calls):
            raise ValueError('Whole-episode positive action selection differs from audit')
        episode_records = []
        for step in trace['steps']:
            if step['call'] not in encoded_calls:
                continue
            request_path = directory/f"request-{step['call']:03d}.json"
            if sha256(request_path) != step['request_sha256']:
                raise ValueError('Actual model request changed')
            request = json.loads(request_path.read_text()); selected = {}
            def encode(messages):
                selected['messages'] = messages
                return encode_prefix(tokenizer, messages, profile['chat_template_kwargs'], protocol['agent_policy']['format'])[0]
            prefix, context = prepare_context(request['messages'], encode, request['max_input_tokens'], protocol['agent_policy']['context'])
            if (digest(json.dumps(prefix, separators=(',', ':')).encode()) != step['generation']['input_tokens_sha256']
                    or context != step['generation']['context']):
                raise ValueError('Export prefix differs from the actual served request')
            record = {'id': episode_id + ':' + str(step['call']), 'episode_id': episode_id,
                      'messages': selected['messages'], 'target': step['generation']['text'],
                      'provenance': {'split': 'train', 'task_id': row['id'], 'repository': row['repository'],
                                     'kind': encoded_calls[step['call']]['kind'], 'producer_kind': 'model',
                                     'trajectory_sha256': row['trajectory_sha256'], 'request_sha256': step['request_sha256'],
                                     'source_manifest_sha256': sha256(data/'sources.json'),
                                     'teacher_profile_sha256': protocol['teacher_profile_sha256'],
                                     'call': step['call'], 'executed': True}}
            encoded = encode_training_example(record, tokenizer, profile)
            expected = encoded_calls[step['call']]
            if (encoded['ids'][:len(prefix)] != prefix or len(encoded['ids']) != expected['tokens']
                    or encoded['supervised_tokens'] != expected['supervised_tokens']
                    or len(encoded['ids']) > protocol['encoding_screen_max_tokens']):
                raise ValueError('Export target differs from audited complete encoding')
            record['encoding_receipt'] = {'tokens': len(encoded['ids']), 'supervised_tokens': encoded['supervised_tokens'],
                                          'prefix_sha256': step['generation']['input_tokens_sha256']}
            episode_records.append(record)
        records.extend(episode_records)
        episodes.append({'episode_id': episode_id, 'task_id': row['id'], 'repository': row['repository'],
                         'trajectory_sha256': row['trajectory_sha256'], 'positive_action_records': len(episode_records),
                         'excluded_unexecuted_action_calls': row['excluded_unexecuted_action_calls']})
    summary = inventory(episodes, records)
    out.mkdir(parents=True)
    data_path = out/'data.jsonl'
    data_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    summary.update(schema_version=1, purpose='training_data_inspection', audit_sha256=sha256(audit_path),
                   protocol_sha256=sha256(protocol_path), data_sha256=sha256(data_path),
                   episodes=episodes, maximum_record_tokens=max((r['encoding_receipt']['tokens'] for r in records), default=0),
                   supervised_tokens=sum(r['encoding_receipt']['supervised_tokens'] for r in records),
                   raw_data_public=False, optimizer_updates=0, final_test_opened=False, new_development_scores=0)
    write_json(out/'manifest.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'protocol', 'out']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); value = run(a.root, a.protocol, a.out)
    print(json.dumps({k: value[k] for k in ['full_episodes','unique_tasks','positive_action_records','maximum_record_tokens']}))
