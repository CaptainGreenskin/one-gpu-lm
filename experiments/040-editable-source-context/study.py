"""Freeze and run the source-context contrast on the completed 039 development cohort."""
import argparse
from collections import Counter
from hashlib import sha256 as hash_bytes
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_policy import initial_messages
from one_gpu_lm.verified.historic_tasks import prepare
from one_gpu_lm.verified.sandbox import Sandbox
from one_gpu_lm.verified.tasks import relative

from source_context import with_source


EXP = Path('experiments/040-editable-source-context')
REFERENCE = Path('experiments/039-release-development')
PRIOR_CONTROLLER = Path('runs/q039-development-restart/controller.json')
PRIOR_AUDIT = PRIOR_CONTROLLER.with_name('audit.json')
PRIVATE = Path('private/source-context-040')
OUT = Path('runs/q040-source-context')
ARMS = ('base','candidate')
RUNNER = Path('experiments/035-edit-bottleneck/baseline.py')
REPLAYER = RUNNER.with_name('audit.py')


def read(path):
    return json.loads(path.read_bytes())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference(root, design):
    sys.path.insert(0,str(root/REFERENCE))
    runner = load('source_context_reference',root/REFERENCE/'run.py')
    context = runner.inputs(root,after_restart=True)
    audit = read(root/PRIOR_AUDIT)
    if (audit['passed'] is not True or audit['controller_sha256'] != sha256(root/PRIOR_CONTROLLER)
            or audit['protocol_sha256'] != sha256(context.plan_path) or audit['final_test_opened']
            or len(audit['rows']) != context.plan['rollout_count']
            or context.plan['task_count'] != design['task_count'] or design['repetitions'] != 1
            or design['evaluation_budget'] != context.common['evaluation_budget']
            or design['recovery_accounting_seconds'] != context.plan['recovery_accounting_seconds']
            or design['pipeline_limit_seconds'] != context.plan['pipeline_limit_seconds']
            or design['optimizer_updates'] or design['final_test_opened'] or design['weight_publication_allowed']):
        raise ValueError('Require the completed 039 development cohort and unchanged task budgets')
    return runner,context


def schedule(tasks):
    numbers = sorted(tasks,key=lambda n: fingerprint(['040-source-context',n]))
    return [dict(task_number=n,task_id=tasks[n]['id'],arm=arm,block=tasks[n]['family'],run_id=f't{n:02d}-{arm}')
            for i,n in enumerate(numbers) for arm in (ARMS if i%2==0 else ARMS[::-1])]


def input_sha256(tokens):
    return hash_bytes(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()


def build_prompts(root, runner, context, design):
    from transformers import AutoTokenizer

    profile = read(root/context.common['evaluation_profile_path'])
    policy = context.common['agent_policy']
    budget = context.common['evaluation_budget']
    tokenizer = AutoTokenizer.from_pretrained(context.runtime['model'],local_files_only=True,trust_remote_code=False)

    def encode(task):
        def tokens(messages):
            return encode_prefix(tokenizer,messages,profile['chat_template_kwargs'],policy['format'])[0]
        return prepare_context(initial_messages(task,policy,budget),tokens,budget['max_input_tokens'],policy['context'])[0]

    observed = []
    prompts = {}
    with tempfile.TemporaryDirectory(prefix='040-original-',dir=root/PRIVATE) as temp:
        scratch = Path(temp)
        for n,task in context.tasks.items():
            original = context.prompts[n]
            source = context.sources[task['source_id']]
            workspace = scratch/f't{n:02d}'
            before = prepare(root/runner.DATA/'snapshots'/source['source_id'],workspace,source,task)
            proof = context.proof['rows'][n-1]
            if fingerprint(sorted(before.items())) != proof['workspace_sha256']:
                raise ValueError('Actor workspace differs from the audited reference')
            files = {}
            for name in task['editable_files']:
                path = workspace/relative(name)
                if path.is_symlink() or not path.is_file() or sha256(path) != before[name]:
                    raise ValueError('Only original declared editable source is allowed')
                files[name] = path.read_text(encoding='utf-8')
            treatment,base_tokens,new_tokens,width = with_source(original,files,encode,design['source_context'])
            if len(base_tokens) != proof['input_tokens'] or input_sha256(base_tokens) != proof['input_sha256']:
                raise ValueError('The baseline opening no longer reproduces 039')
            prompts.update({f't{n:02d}-base':original,f't{n:02d}-candidate':treatment})
            observed.append({'task_number':n,'family':task['family'],'workspace_sha256':proof['workspace_sha256'],
                             'window_lines':width,'multifile':proof['multifile'],
                             'input_tokens':dict(zip(ARMS,map(len,(base_tokens,new_tokens)))),
                             'input_sha256':dict(zip(ARMS,map(input_sha256,(base_tokens,new_tokens)))),
                             'description_sha256':dict(zip(ARMS,[fingerprint(x['description']) for x in (original,treatment)]))})
    if len(observed) != design['task_count'] or not any(row['window_lines'] for row in observed):
        raise ValueError('Need the complete cohort and an actual source-context contrast')
    return prompts,observed


def inputs(root, freeze=False):
    design = read(root/EXP/'design.json')
    prior,context = reference(root,design)
    private = root/PRIVATE
    private.mkdir(mode=0o700,parents=True,exist_ok=True)
    prompts,rows = build_prompts(root,prior,context,design)
    order = schedule(context.tasks)
    if len(order) != design['rollout_count']:
        raise ValueError('Every task must appear in both conditions')
    proof_path = private/'preflight.json'
    proof = {'schema_version':1,'reference_inputs_sha256':fingerprint(context.files),
             'reference_audit_sha256':sha256(root/PRIOR_AUDIT),'rows':rows,'schedule_sha256':fingerprint(order),
             'model_generations':0,'optimizer_updates':0,'final_test_opened':False}
    if freeze:
        if proof_path.exists() and read(proof_path) != proof:
            raise ValueError('Preserve the original preflight')
        if not proof_path.exists():
            write_json(proof_path,proof)
    elif read(proof_path) != proof:
        raise ValueError('The frozen prompt preflight changed')
    extra = [EXP/'design.json',EXP/'study.py',EXP/'source_context.py',EXP/'audit.py',RUNNER,REPLAYER]
    plan = dict(design,pretrained_model=context.common['model_id'],base_revision=context.common['revision'],
                families=context.plan['families'],schedule_sha256=proof['schedule_sha256'],
                reference_inputs_sha256=proof['reference_inputs_sha256'],reference_audit_sha256=proof['reference_audit_sha256'],
                preflight_sha256=sha256(proof_path),frozen_files={str(path):sha256(root/path) for path in extra},
                preflight={'tasks':len(rows),'modified_openings':sum(bool(r['window_lines']) for r in rows),
                           'window_line_counts':dict(sorted(Counter(str(r['window_lines']) for r in rows).items())),
                           'max_candidate_input_tokens':max(r['input_tokens']['candidate'] for r in rows),
                           'max_added_input_tokens':max(r['input_tokens']['candidate']-r['input_tokens']['base'] for r in rows)})
    if freeze:
        return plan
    path = root/EXP/'protocol.json'
    publication = read(private/'publication.json')
    if (read(path) != plan or publication['plan_sha256'] != sha256(path)
            or publication['anonymous_files_verified'] is not True or publication['ci_conclusion'] != 'success'):
        raise ValueError('Require the published exact 040 preflight and protocol')
    runner = load('source_context_baseline_runner',root/RUNNER)
    protocol = prior.effective(context.common,'base')
    execution_plan = dict(plan,frozen_files={**context.files,**plan['frozen_files']})
    study = runner.Context(path,execution_plan,prior,root/prior.DATA,{t['id']:t for t in context.tasks.values()},
                           context.sources,order,protocol,context.runtime,root/prior.TEST_RUNTIME,proof)
    return runner,study,prompts


def run(root):
    runner,context,prompts = inputs(root)
    reference_run = context.legacy
    limit = context.protocol['evaluation_budget']['max_tool_seconds']
    date_sandbox = Sandbox(read(root/reference_run.DATE_RUNTIME),limit)

    def perform(root,data,task,source,backend,sandbox,protocol,directory,arm,**kwargs):
        selected = prompts[directory.name]
        if selected['id'] != task['id']:
            raise ValueError('Source context belongs to another scheduled task')
        active = date_sandbox if reference_run.test_runtime(task['family']) == reference_run.DATE_RUNTIME else sandbox
        return runner.rollout(root,data,selected,source,backend,active,protocol,directory,arm,**kwargs)

    runner.sweep(root,root/OUT,context,perform=perform)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--freeze',action='store_true')
    mode.add_argument('--check',action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    if a.freeze:
        print(json.dumps(inputs(root,freeze=True),indent=2))
    elif a.check:
        _,context,_ = inputs(root)
        print(json.dumps({'tasks':len(context.tasks),'rollouts':len(context.order),'final_test_opened':False}))
    else:
        run(root)
