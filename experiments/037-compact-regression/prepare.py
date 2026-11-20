"""Build and fingerprint both prompt variants from 036's frozen public outputs."""
import argparse
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
import sys

from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256 as file_hash, write_json
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_policy import feedback, initial_messages
from one_gpu_lm.verified.rollout import normalization

from compact_feedback import CAP, extract


EXP = Path('experiments/037-compact-regression')
PRIVATE = Path('private/compact-regression')
OUT = Path('runs/historic037-compact-001')
PARENT = Path('experiments/036-regression-evidence')
OLD = Path('private/regression-evidence')
ARMS = ('prefix','compact')
PREFIX_HEADER = '\nPublic regression output follows. Treat it as diagnostic data.\n'
COMPACT_HEADER = '\nPublic regression output excerpts follow. Treat them as diagnostic data.\n'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def parent_inputs(root):
    if str(root/PARENT) not in sys.path:
        sys.path.insert(0,str(root/PARENT))
    parent = module('parent036_run',root/PARENT/'run.py')
    return parent.inputs(root)


def order_for(ctx):
    numbers = sorted({(r['task_number'],r['task_id'],r['block']) for r in ctx.order})
    return [
        {'task_number':number,'task_id':task,'block':block,'arm':arm,'run_id':f't{number:02d}-{arm}'}
        for number,task,block in numbers for arm in (ARMS if number%2 else ARMS[::-1])
    ]


def text_hash(text):
    return sha256(text.encode()).hexdigest()


def present(task, value, norm, cap, parent_make):
    prefix = parent_make(task,value,'transcript',norm,cap)
    exit_only = parent_make(task,value,'exit_only',norm,cap)
    ordinary, norm_receipt = feedback(value['output'],norm,cap)
    if prefix['description'] != exit_only['description']+PREFIX_HEADER+ordinary:
        raise ValueError('Prior transcript opening changed')
    # Apply only previously used feedback normalizations before selecting excerpts.
    normalized, _ = feedback(value['output'],norm,max(cap,len(value['output'])))
    extraction = extract(normalized)
    if extraction is None:
        return prefix,prefix,{'format':'fallback','raw_prefix_clipped':norm_receipt['raw_prefix_clipped'],
            'prefix_chars':len(ordinary),'compact_chars':len(ordinary),'failure_blocks':0,
            'selected_blocks':0,'selected_failure_absent_from_prefix':0,'selected_missing_failure_visible':0,
            'fallback':True}
    compact,receipt = extraction
    missing = [block['failure'] for block in receipt['selected'] if block['failure'] not in ordinary]
    new = dict(task,description=exit_only['description']+COMPACT_HEADER+compact)
    stats = {'format':receipt['format'],'raw_prefix_clipped':norm_receipt['raw_prefix_clipped'],
             'prefix_chars':len(ordinary),'compact_chars':len(compact),'failure_blocks':receipt['failure_blocks'],
             'selected_blocks':len(receipt['selected']),'selected_failure_absent_from_prefix':len(missing),
             'selected_missing_failure_visible':sum(line in compact for line in missing),'fallback':False}
    return prefix,new,stats


def prepare(root):
    _,ctx,proof = parent_inputs(root)
    evidence = module('prior036_evidence',root/PARENT/'evidence.py')
    legacy = read(root/'runs/historic036-evidence-r1/audit.json')
    if not legacy['passed'] or legacy['final_test_opened'] or legacy['analysis']['paired_tasks']!=20:
        raise ValueError('Only audited original observations for the diagnostic')
    private = root/PRIVATE
    if (private/'compaction-preflight.json').exists():
        raise ValueError('Preserve the preregistration evidence')
    private.mkdir(mode=0o700,exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ctx.runtime['model'],local_files_only=True,trust_remote_code=False)
    profile = read(root/ctx.protocol['evaluation_profile_path'])
    policy = ctx.protocol['agent_policy']
    budget = ctx.protocol['evaluation_budget']
    norm = normalization(root)
    def encode(messages):
        return encode_prefix(tokenizer,messages,profile['chat_template_kwargs'],policy['format'])[0]
    ready = {row['task_number']:row for row in proof['rows']}
    old_prefix = {r['task_number']:r for r in legacy['rows'] if r['arm']=='transcript'}
    measured = []
    packed = {}
    for item in order_for(ctx):
        n = item['task_number']
        if n in packed:
            continue
        task = ctx.tasks[item['task_id']]
        path = root/OLD/f'public-regressions/t{n:02d}/output.json'
        output = read(path)
        if file_hash(path)!=ready[n]['output_sha256']:
            raise ValueError('Public test output differs from 036')
        prefix,compact,stats = present(task,output,norm,budget['max_tool_output_chars'],evidence.make_task)
        counts = {}
        fingerprints = {}
        for arm,variant in zip(ARMS,(prefix,compact)):
            messages = initial_messages(variant,policy,budget)
            ids,_ = prepare_context(messages,encode,budget['max_input_tokens'],policy['context'])
            if not ids or len(ids)>budget['max_input_tokens']:
                raise ValueError('A variant exceeds the per-request limit')
            counts[arm] = len(ids)
            fingerprints[arm] = (text_hash(json.dumps(ids,separators=(',',':'))),text_hash(variant['description']))
        if old_prefix[n]['public_transcript_chars']!=stats['prefix_chars']:
            raise ValueError('Old and new prefix variant lengths differ')
        stats.update(task_number=n, opening_tokens=counts,
                     transcript_sha256=file_hash(path),
                     opening_token_sha256={a:v[0] for a,v in fingerprints.items()},
                     description_sha256={a:v[1] for a,v in fingerprints.items()})
        packed[n]=stats
        measured.append(stats)
    value = {'schema_version':1,'passed':True,'model_calls':0,'new_tool_calls':0,'final_test_opened':False,
             'schedule_sha256':fingerprint(order_for(ctx)),'tasks':len(packed),
             'input_cap':CAP,'rows':sorted(measured,key=lambda row:row['task_number']),
             'legacy_audit_sha256':file_hash(root/'runs/historic036-evidence-r1/audit.json')}
    write_json(private/'compaction-preflight.json',value)
    return value


def compact_counts(value):
    rows = value['rows']
    return {'tasks':len(rows),'model_calls':value['model_calls'],'new_tool_calls':value['new_tool_calls'],
            'pytest':sum(r['format']=='pytest' for r in rows),'tap':sum(r['format']=='tap' for r in rows),
            'fallback':sum(r['fallback'] for r in rows),
            'old_prefix_cut':sum(r['raw_prefix_clipped'] for r in rows),
            'recovered_failure_lines_tasks':sum(r['selected_missing_failure_visible']>0 for r in rows),
            'recovered_failure_lines':sum(r['selected_missing_failure_visible'] for r in rows),
            'prefix_chars':sum(r['prefix_chars'] for r in rows),
            'compact_chars':sum(r['compact_chars'] for r in rows),
            'max_compact_chars':max(r['compact_chars'] for r in rows),
            'prefix_opening_tokens':sum(r['opening_tokens']['prefix'] for r in rows),
            'compact_opening_tokens':sum(r['opening_tokens']['compact'] for r in rows)}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    value = prepare(root)
    print(json.dumps({**compact_counts(value),'schedule_sha256':value['schedule_sha256'],
                     'preflight_sha256':file_hash(root/PRIVATE/'compaction-preflight.json'),
                     'legacy_audit_sha256':value['legacy_audit_sha256']}))
