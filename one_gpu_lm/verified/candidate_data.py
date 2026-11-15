"""Whole-episode training admission: reject inconsistencies, never filter actions."""
from collections import Counter
import hashlib
from one_gpu_lm.common.training_encoding import encode_training_example


def select(rows, manifest, audit, sources, profile, spec, tokenizer, encoder=encode_training_example):
    """Consume every audited eligible episode, preserving all its action targets."""
    gate = spec['training_data_gate']
    if not audit['decision']['passed']:
        raise ValueError('Collection did not pass its registered data gate')
    episodes = manifest['episodes']
    episode_ids = [e['episode_id'] for e in episodes]
    task_ids = [e['task_id'] for e in episodes]
    if len(set(episode_ids)) != len(episodes) or len(set(task_ids)) != len(episodes):
        raise ValueError('Duplicate complete episode or task')
    if not gate['minimum_complete_unique_verified_episodes'] <= len(episodes) <= gate['maximum_complete_unique_verified_episodes']:
        raise ValueError('Insufficient or excessive complete unique episodes')
    eligible = {r['id']:r for r in audit['rows'] if r['eligible_for_sft']}
    if set(task_ids) != set(eligible):
        raise ValueError('Export omitted or added eligible whole episodes')
    source_map = {s['repository']:s for s in sources}
    counts = Counter(e['repository'] for e in episodes)
    if set(counts) != set(source_map) or any(n < gate['minimum_per_training_repository'] for n in counts.values()):
        raise ValueError('Training repository coverage gate failed')
    if len({source_map[r]['language'] for r in counts}) < gate['minimum_languages']:
        raise ValueError('Training language coverage gate failed')
    for source in sources:
        if source['split'] != 'training' or source['approved_for'].get('training') is not True:
            raise ValueError('Unapproved or held-out source')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate action record')
    by_episode = {e['episode_id']:e for e in episodes}
    actual = Counter()
    selected = []
    seen_calls = set()
    for row in rows:
        if row['episode_id'] not in by_episode:
            raise ValueError('Orphan action record')
        episode = by_episode[row['episode_id']]
        provenance = row['provenance']; task = eligible[episode['task_id']]
        if (provenance['split'] != 'train' or provenance['executed'] is not True
                or provenance['producer_kind'] != 'model' or provenance['task_id'] != episode['task_id']
                or provenance['repository'] != episode['repository'] or provenance['trajectory_sha256'] != task['trajectory_sha256']
                or episode['trajectory_sha256'] != task['trajectory_sha256']):
            raise ValueError('Action provenance differs from verified training episode')
        expected = {r['call']:r for r in task['encoded_actions']}
        call = provenance['call']; key = (episode['task_id'], call)
        if call not in expected or key in seen_calls or provenance['kind'] != expected[call]['kind']:
            raise ValueError('Added, duplicate or changed audited action')
        seen_calls.add(key)
        encoded = encoder(row, tokenizer, profile)
        receipt = row['encoding_receipt']
        if (len(encoded['ids']) != receipt['tokens'] or encoded['supervised_tokens'] != receipt['supervised_tokens']
                or receipt['tokens'] != expected[call]['tokens']
                or receipt['supervised_tokens'] != expected[call]['supervised_tokens']
                or len(encoded['ids']) > profile['training']['max_sequence_length']):
            raise ValueError('Complete action encoding differs or exceeds cap; never silently truncate/filter')
        if encoded['supervised_tokens'] <= 0 or len(encoded['ids']) != len(encoded['labels']):
            raise ValueError('Invalid supervised target')
        selected.append({'id':row['id'],'episode_id':row['episode_id'],'provenance':provenance,**encoded})
        actual[row['episode_id']] += 1
    expected_calls = {(r['id'], a['call']) for r in eligible.values() for a in r['encoded_actions']}
    if seen_calls != expected_calls:
        raise ValueError('One or more actions missing from complete episodes')
    if any(actual[e['episode_id']] != e['positive_action_records'] for e in episodes):
        raise ValueError('Episode action count differs from manifest')
    if len(rows) != manifest['positive_action_records'] or len(episodes) != manifest['full_episodes']:
        raise ValueError('Export summary count differs from actual data')
    if manifest['unique_tasks'] != len(set(task_ids)):
        raise ValueError('Task count inflated by action records')
    return sorted(selected, key=lambda r: hashlib.sha256(('025-student-v1\0'+r['id']).encode()).hexdigest())


def windows(rows, size):
    if size < 1:
        raise ValueError('Invalid accumulation window')
    for start in range(0, len(rows), size):
        yield rows[start:start+size]
