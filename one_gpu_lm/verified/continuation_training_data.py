"""Reuse the audited 031 records and admit only the verified 041 continuations."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import identity_digest
from one_gpu_lm.common.training_encoding import encode_training_example
from .candidate import validate_recipe
from .candidate_data import windows
from .mixed_inputs import select
from .tasks import digest, relative


BASELINE = Path('experiments/031-correction-mix/candidate-protocol.json')
BASELINE_REVIEW = Path('private/modelscope-development/candidate-input-review.json')
BASELINE_AUDIT = BASELINE_REVIEW.with_name('baseline-031-integrity.json')
SECRET_SHAPES = (
    r'-----BEGIN (?:ENCRYPTED |RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
    r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
    r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    r'\b(?:sk-proj-[A-Za-z0-9_-]{30,}|sk-ant-[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{30,})\b',
)


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def record_shape(row):
    return {'id':row['id'],'episode_id':row['episode_id'],'tokens':len(row['ids']),
            'supervised_tokens':row['supervised_tokens'],'record_group_kind':row['record_group_kind']}


@dataclass
class TrainingData:
    protocol: dict
    profile: dict
    recipe: dict
    review: dict
    raw: list
    encoded: list
    baseline: dict


def load_baseline(root, tokenizer):
    path = root/BASELINE
    protocol = read(path)
    audit_path = root/BASELINE_AUDIT
    audit = read(audit_path)
    review = read(root/BASELINE_REVIEW)
    selection = read(root/relative(protocol['run_directory'])/'selection.json')
    runs = read(root/relative(protocol['run_directory'])/'result.json')
    profile = read(root/relative(protocol['profile_path']))
    spec = read(root/relative(protocol['cycle_spec']))
    validate_recipe(profile,spec)
    source = root/relative(protocol['sources_path'])
    data = root/relative(protocol['data_path'])
    manifest = read(root/relative(protocol['export_manifest']))
    if (audit['passed'] is not True or runs['complete'] is not True or runs['all_selected_records_consumed'] is not True
            or review['audited_checkpoint_matches'] is not True or review['source_snapshots_match'] is not True
            or review['all_source_notices_and_lineage_reviewed_for_training'] is not True
            or review['training_integrity_sha256'] != sha256(audit_path)
            or review['checkpoint_files_sha256'] != audit['adapter_files_sha256']
            or review['source_manifest_sha256'] != sha256(source)
            or (review['model_id'],review['revision']) != (profile['model_id'],profile['revision'])
            or any(x['final_test_opened'] or x['weights_public'] for x in (audit,review,runs))
            or not audit['protocol_sha256'] == runs['protocol_sha256'] == sha256(path)
            or not review['actual_training_data_sha256'] == runs['data_sha256'] == manifest['data_sha256'] == sha256(data)
            or runs['profile_sha256'] != sha256(root/relative(protocol['profile_path']))):
        raise ValueError('Require the exact source-reviewed and audited 031 training lineage')
    raw = rows(data)
    encoded, metadata = select(root,protocol,profile,tokenizer,raw,manifest)
    recipe = spec['candidate']
    input_tokens = sum(len(row['ids']) for row in encoded)
    if ([record_shape(row) for row in encoded] != selection['records']
            or len(encoded) != runs['action_records_consumed'] or len(raw) != len(encoded)
            or any(row['provenance']['split'] != 'train' for row in raw)
            or input_tokens != runs['input_tokens'] or input_tokens != audit['input_tokens']
            or sum(row['supervised_tokens'] for row in encoded) != audit['supervised_tokens']
            or len(list(windows(encoded,recipe['gradient_accumulation_records']))) != audit['optimizer_updates']):
        raise ValueError('The retained records do not reproduce the original tokenization and update count')
    summary = {'protocol_sha256':sha256(path),'audit_sha256':sha256(audit_path),'data_sha256':sha256(data),
               'profile_sha256':runs['profile_sha256'],'source_manifest_sha256':sha256(source),
               'selection_sha256':identity_digest(selection['records']),'records':len(encoded),
               'input_tokens':input_tokens,'supervised_tokens':audit['supervised_tokens'],
               'optimizer_updates':audit['optimizer_updates'],'full_episodes':metadata['full_episodes'],
               'retained_corrective_fragments':metadata['corrective_fragments'],'unique_tasks':metadata['unique_tasks'],
               'new_model_generations':0,'new_optimizer_updates':0,'final_test_opened':False,'weights_public':False}
    return TrainingData(protocol,profile,recipe,review,raw,encoded,summary)


def audited_continuations(root, baseline, tokenizer):
    from . import first_edit_continuation as capture
    from . import first_edit_continuation_audit as replay

    protocol_path = root/'private/protocol.json'
    protocol = read(protocol_path)
    protocol_hash = sha256(protocol_path)
    out = root/relative(protocol['run_directory'])
    audit_path,controller_path = out/'audit.json',out/'controller.json'
    handoff = read(root/'private/first-edit-job/state.json')
    if (handoff['phase'] != 'audited' or handoff['protocol_sha256'] != protocol_hash
            or handoff['audit_sha256'] != sha256(audit_path) or handoff['controller_sha256'] != sha256(controller_path)
            or handoff['final_test_opened']):
        raise ValueError('041 has no verified terminal handoff')
    audit,controller = read(audit_path),read(controller_path)
    # No training export is opened if the collection has failed.
    if audit['passed'] is not True or audit['collection_gate_passed'] is not True:
        raise ValueError('041 did not qualify for a continuation candidate')
    for module in (capture,replay):
        module_file = Path(module.__file__)
        key = 'one_gpu_lm/verified/'+module_file.name
        if protocol['frozen_files'][key] != sha256(module_file):
            raise ValueError('Continuation capture or audit implementation differs from the registered run')
    registration = read(root/relative(protocol['publication_receipt']))
    if (registration['protocol_sha256'] != protocol_hash or registration['anonymous_files_verified'] is not True
            or registration['ci_conclusion'] != 'success' or audit['protocol_sha256'] != protocol_hash
            or controller['protocol_sha256'] != protocol_hash or controller['backend_stopped'] is not True
            or controller['complete'] is not True or controller['frozen_files_unchanged'] is not True
            or audit['final_test_opened'] or controller['final_test_opened'] or audit['new_optimizer_updates'] != 0
            or audit['journal_requests_verified'] != sum(row['attempted_calls'] for row in controller['runs'])):
        raise ValueError('041 registration, audit or generation accounting differs')
    reference,data,cases,design = capture.load_cases(root,protocol)
    if (sha256(data/'sources.json') != baseline.baseline['source_manifest_sha256']
            or sha256(data/'tasks.json') != baseline.review['task_manifest_sha256']
            or (reference['model_id'],reference['revision']) != (baseline.profile['model_id'],baseline.profile['revision'])):
        raise ValueError('Continuation sources or base differ from the approved training lineage')
    teacher = read(root/relative(reference['evaluation_profile_path']))
    student_path = root/relative(reference['reasoning_comparison']['student_profile_path'])
    student = read(student_path)
    if (sha256(student_path) != reference['reasoning_comparison']['student_profile_sha256']
            or student['chat_template_kwargs'] != baseline.profile['chat_template_kwargs']):
        raise ValueError('041 was screened with a different student chat template')
    by_run = {row['run_id']:row for row in controller['runs']}
    schedule = {(row['task_id'],row['arm']):row['run_id'] for row in protocol['schedule']}
    if (len(by_run) != len(controller['runs']) or set(by_run) != set(schedule.values())
            or len(audit['pairs']) != len(cases)):
        raise ValueError('Incomplete or duplicate 041 arm receipts')
    selected = []
    qualified = {}
    for number,((task_id,case),pair) in enumerate(zip(cases.items(),audit['pairs']),1):
        if (pair['pair'] != number or pair['family'] != case['task']['family']
                or set(pair['arms']) != {'control','correction'}
                or not all(cell['audited'] is True for cell in pair['arms'].values())):
            raise ValueError('041 arm audit differs from the registered task')
        for arm in ('control','correction'):
            if not pair['arms'][arm]['qualified']:
                continue
            key = schedule[task_id,arm]
            directory = out/'private/tasks'/relative(key)
            trace_path = directory/'trajectory.json'
            trace = read(trace_path)
            result = read(directory/'result.json')
            restored = read(directory/'restored-state.json')
            if (result != by_run[key] or result['trajectory_sha256'] != sha256(trace_path)
                    or trace['id'] != task_id or trace['split'] != 'training' or trace['branch'] != arm
                    or restored['initial_messages_sha256'] != capture.message_hash(case['messages'])
                    or restored['prefix_sha256'] != sha256(case['snapshot']/'prefix.json')):
                raise ValueError('A qualified trajectory differs from its protected audit or starting state')
            targets, reason = replay.encode_targets(directory,trace,case,reference,teacher,student,tokenizer,
                                                  result['trajectory_sha256'])
            if reason or [row['provenance']['kind'] for row in targets] != ['edit','verify','submit']:
                raise ValueError('Qualified continuation failed exact target reconstruction')
            selected.extend(targets)
            qualified[task_id] = pair['family']
            break
    if (len(qualified) != audit['qualified_tasks'] or len(set(qualified.values())) != audit['qualified_families']
            or len(qualified) < design['minimum_qualified_tasks']
            or len(set(qualified.values())) < design['minimum_qualified_families']
            or set(qualified) & {row['provenance']['task_id'] for row in baseline.raw}):
        raise ValueError('Continuation families, tasks or registered training split differ')
    export = out/'private/first-edit-export'
    manifest,raw = read(export/'manifest.json'),rows(export/'data.jsonl')
    if (manifest != audit['export'] or manifest['raw_data_public'] is not False
            or manifest['data_sha256'] != sha256(export/'data.jsonl') or raw != selected
            or manifest['records'] != len(selected)):
        raise ValueError('The verified export omits, reorders or changes action targets')
    if any(re.search(pattern,json.dumps(row,ensure_ascii=False)) for row in raw for pattern in SECRET_SHAPES):
        raise ValueError('The new private training export needs review of credential-like text')
    encoded = []
    for row in selected:
        v, receipt = row['provenance'],row['encoding_receipt']
        if (v['split'] != 'train' or v['executed'] is not True or v['producer_kind'] != 'model'
                or v['repository'] not in baseline.review['sources'] or v['model_role'] != 'base'
                or v['target_transform'] != 'verified_first_edit_action_only'
                or '<think>' in row['target'] or '</think>' in row['target']):
            raise ValueError('Continuation contains unapproved, non-action or held-out supervision')
        value = encode_training_example(row,tokenizer,baseline.profile)
        prefix = value['ids'][:value['prefix_tokens']]
        if (not 0 < value['supervised_tokens'] == receipt['supervised_tokens']
                or not len(value['ids']) == receipt['tokens'] <= baseline.recipe['max_sequence_length']
                or digest(json.dumps(prefix,separators=(',', ':')).encode()) != receipt['student_input_sha256']
                or len(value['ids']) != len(value['labels'])):
            raise ValueError('Continuation encoding changed or exceeds the untruncated training cap')
        encoded.append({'id':row['id'],'episode_id':row['episode_id'],'record_group_kind':'first_edit_continuation',**value})
    if sum(row['supervised_tokens'] for row in encoded) != manifest['supervised_tokens']:
        raise ValueError('The verified continuation supervision totals differ')
    facts = {'protocol_sha256':protocol_hash,'controller_sha256':sha256(controller_path),'audit_sha256':sha256(audit_path),
             'export_sha256':manifest['data_sha256'],'qualifying_tasks':len(qualified),
             'qualifying_families':len(set(qualified.values())),'records':len(raw),
             'supervised_tokens':manifest['supervised_tokens'],'credential_like_records':0}
    return raw,encoded,facts


def matched_windows(retained, supplements, size):
    """Keep the original update count and distribute one new action to each selected window."""
    batches = list(windows(retained,size))
    all_rows = retained+supplements
    if not batches or len(supplements) > len(batches) or len({row['id'] for row in all_rows}) != len(all_rows):
        raise ValueError('Invalid or duplicate actions in the matched-update mix')
    ordered = sorted(supplements,key=lambda row:identity_digest(['041-continuation',row['id']]))
    for i,row in enumerate(ordered):
        step = (2*i+1)*len(batches)//(2*len(ordered))
        batches[step].append(row)
    return batches


def mixture(baseline, raw, extra, facts):
    batches = matched_windows(baseline.encoded,extra,baseline.recipe['gradient_accumulation_records'])
    indexed = {row['id']:row for row in baseline.raw+raw}
    ordered = [indexed[row['id']] for batch in batches for row in batch]
    if (len(ordered) != len(indexed) or len(indexed) != len(baseline.raw)+len(raw)
            or len(batches) != baseline.baseline['optimizer_updates']):
        raise ValueError('Mixture omitted retained targets or changed the update count')
    metadata = {'retained_031':baseline.baseline,'new_041':facts,'optimizer_updates':len(batches),
                'input_tokens':sum(len(row['ids']) for batch in batches for row in batch),
                'supervised_tokens':sum(row['supervised_tokens'] for batch in batches for row in batch),
                'record_counts':dict(Counter(row['record_group_kind'] for batch in batches for row in batch)),
                'window_sizes':list(map(len,batches)),
                'batch_identity_sha256':identity_digest([[record_shape(row) for row in batch] for batch in batches]),
                'target_truncation':False,'oversampling':False,'final_test_opened':False,'weights_public':False}
    return ordered,batches,metadata
