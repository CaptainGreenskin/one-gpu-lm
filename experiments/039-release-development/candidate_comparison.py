"""Score and replay a registered base-versus-LoRA comparison on a fresh development panel."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
import traceback

from one_gpu_lm.common.agent_policy import prepare_context
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import atomic_json, gpu_lock_path, single_writer, utc
from one_gpu_lm.common.telemetry import Telemetry
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.historic_policy import initial_messages
from one_gpu_lm.verified.historic_tasks import prepare, validate_split
from one_gpu_lm.verified.resilient_backend_control import JournalBackend
from one_gpu_lm.verified.resilient_rollout import run as rollout
from one_gpu_lm.verified.rollout import normalization
from one_gpu_lm.verified.sandbox import Sandbox
from one_gpu_lm.verified.tasks import digest, relative

from analysis import analyze, schedule
from recovery import served_weight_hashes
from run import effective, frozen, load, prefix_prompt, read, reference_protocol


SOURCE_CONTEXT = Path('experiments/040-editable-source-context')
CONTEXT_RUN = Path('runs/q040-source-context')
EXPERIMENT = Path('experiments/039-release-development')
SELECTIONS = ('original', 'editable_source')


@dataclass
class Panel:
    data: Path
    runtime_path: Path
    model_runtime: dict
    common: dict
    profile: dict
    tokenizer: object
    proof: dict
    cases: dict


def build_panel(root, data, openings_path, runtime_path):
    from transformers import AutoTokenizer

    proof = read(openings_path)
    common = reference_protocol(root)
    budget, policy = common['evaluation_budget'], common['agent_policy']
    model_runtime = read(root / 'private/parent-runtime.json')
    profile_path = root / relative(common['evaluation_profile_path'])
    profile = read(profile_path)
    if (sha256(data / 'cohort.private.json') != proof['cohort_sha256']
            or any(sha256(data / (name + '.json')) != proof[name + '_sha256'] for name in ('tasks', 'sources'))
            or sha256(runtime_path) != proof['runtime_sha256'] or budget != proof['budget']
            or (common['model_id'], common['revision']) != (proof['model_id'], proof['revision'])
            or sha256(profile_path) != proof['profile_sha256'] or common['evaluation_profile_sha256'] != proof['profile_sha256']
            or sha256(root / relative(common['model_manifest_path'])) != proof['model_manifest_sha256']
            or proof['model_generations'] != 0 or proof['optimizer_updates'] != 0 or proof['final_test_opened'] is not False):
        raise ValueError('Frozen development data, source runtime or model identity changed')
    design = root / SOURCE_CONTEXT / 'design.json'
    context_code = design.with_name('source_context.py')
    if sha256(design) != proof['source_context_design_sha256'] or sha256(context_code) != proof['source_context_code_sha256']:
        raise ValueError('Source-context reconstruction code differs from the score-free preflight')
    with_source = load('candidate_source_context', context_code).with_source
    limits = read(design)['source_context']
    tokenizer = AutoTokenizer.from_pretrained(model_runtime['model'], local_files_only=True, trust_remote_code=False)

    def encode(prompt):
        def tokenize(messages):
            return encode_prefix(tokenizer, messages, profile['chat_template_kwargs'], policy['format'])[0]
        return prepare_context(initial_messages(prompt, policy, budget), tokenize,
                               budget['max_input_tokens'], policy['context'])[0]

    tasks, sources = read(data / 'tasks.json')['tasks'], read(data / 'sources.json')['sources']
    rows = proof['rows']
    if (len(tasks) != proof['actor_count'] or len(tasks) != len(sources) or len(tasks) != len(rows)
            or len({task['id'] for task in tasks}) != len(tasks)):
        raise ValueError('Expected the complete cohort with unique cases and opening proofs')
    validate_split(sources)
    cases = {}
    with TemporaryDirectory(prefix='compare-inputs-') as tmp:
        for number, (task, source, row) in enumerate(zip(tasks, sources, rows), 1):
            transcript = openings_path.parent / 'first-checks' / f't{number:02d}.json'
            if (row['task_number'] != number or row['family'] != task['family']
                    or sha256(transcript) != row['transcript_sha256']):
                raise ValueError('Initial regression record differs from the frozen case')
            workspace = Path(tmp) / str(number)
            before = prepare(data / 'snapshots' / relative(source['source_id']), workspace, source, task)
            if fingerprint(sorted(before.items())) != row['workspace_sha256']:
                raise ValueError('Initial actor workspace changed')
            original = prefix_prompt(root, common, task, read(transcript))
            editable = {name:(workspace / relative(name)).read_text(encoding='utf-8') for name in task['editable_files']}
            excerpts, base_ids, excerpt_ids, width = with_source(original, editable, encode, limits)
            options = dict(zip(SELECTIONS, (original, excerpts)))
            for kind, ids in zip(SELECTIONS, (base_ids, excerpt_ids)):
                encoded = {'input_tokens':len(ids), 'input_sha256':digest(json.dumps(ids,separators=(',', ':')).encode()),
                           'description_sha256':fingerprint(options[kind]['description'])}
                if (encoded != row['openings'][kind] or not 0 < len(ids) <= budget['max_input_tokens']
                        or width != row['window_lines']):
                    raise ValueError('Actual tokenizer does not reproduce the established opening')
            cases[number] = {'prompts':options,'source':source,'before':before,'opening':row}
    return Panel(data,runtime_path,model_runtime,common,profile,tokenizer,proof,cases)


def approved_opening(root, plan):
    """Select the treatment only if the completed 040 base-only contrast passed its gate."""
    from report import reproduce_analysis

    protocol = root / SOURCE_CONTEXT / 'protocol.json'
    source_plan = read(protocol)
    out = root / CONTEXT_RUN
    controller = out / 'controller.json'
    audit_path = out / 'audit.json'
    if (sha256(protocol) != plan['source_context_protocol_sha256']
            or sha256(audit_path) != plan['source_context_audit_sha256']):
        raise ValueError('The registered inference comparison changed')
    state, audit = read(controller),read(audit_path)
    if (not state['complete'] or not state['backend_stopped'] or not state['frozen_inputs_unchanged']
            or state['optimizer_updates'] or state['final_test_opened'] or state['weights_public']
            or not audit['passed'] or audit['final_test_opened'] or audit['optimizer_updates'] or audit['weights_public']
            or audit['controller_sha256'] != sha256(controller) or audit['protocol_sha256'] != sha256(protocol)
            or audit['started_epoch'] != state['started_epoch'] or audit['finished_epoch'] != state['finished_epoch']
            or len(audit['rows']) != len(state['runs']) or len(audit['rows']) != source_plan['rollout_count']
            or audit['verified_requests'] != sum(row['attempted_calls'] for row in audit['rows'])
            or source_plan['evaluation_budget'] != plan['evaluation_budget']
            or (source_plan['pretrained_model'],source_plan['base_revision']) != (plan['pretrained_model'],plan['base_revision'])):
        raise ValueError('Require the complete preceding inference audit and original budget')
    passed = reproduce_analysis(audit,source_plan)['development_gate_passed']
    return 'editable_source' if passed else 'original'


def adapter_files(adapter):
    entries = list(adapter.iterdir())
    if any(not p.is_file() or p.is_symlink() for p in entries):
        raise ValueError('Expected only registered regular files in the adapter bundle')
    return {p.name:sha256(p) for p in entries}


def reviewed_adapter(review, actual, plan):
    files = plan['all_adapter_files_sha256']
    if (not review['audited_checkpoint_matches'] or not review['eligible_for_existing_scope_of_development']
            or not review['all_source_notices_and_lineage_reviewed_for_training'] or not review['source_snapshots_match']
            or review['final_test_opened'] or review['weights_public'] or review['checkpoint_files_sha256'] != files
            or actual != files or (review['model_id'],review['revision']) != (plan['pretrained_model'],plan['base_revision'])):
        raise ValueError('Require the one independently reviewed candidate checkpoint')
    known = {name.split(':',1)[0].casefold() for name in review['sources']}
    if not known or known & {name.casefold() for name in plan['families']}:
        raise ValueError('Candidate training overlaps the independent development families')
    return served_weight_hashes(files)


def freeze_protocol(args):
    from promotion import COMPARISON_RULES

    root = args.root
    data, openings, runtime = (p.resolve() for p in (args.data,args.openings,args.runtime))
    checkpoint, review_path = (p.resolve() for p in (args.adapter,args.candidate_review))
    output = args.protocol
    run_directory = relative(str(args.run_directory))
    if run_directory.parts[0] != 'runs' or (root/run_directory).exists() or output.exists():
        raise ValueError('Use a fresh protocol and the registered development run directory')
    panel = build_panel(root,data,openings,runtime)
    design = read(root/EXPERIMENT/'design.json')
    review = read(review_path)
    files = adapter_files(checkpoint)
    plan = {name:design[name] for name in COMPARISON_RULES}
    plan.update(schema_version=1,comparison_kind='independent_adapter_comparison',
                candidate=review['experiment'],families=dict(Counter(case['source']['family'] for case in panel.cases.values())),
                cohort_sha256=panel.proof['cohort_sha256'],opening_preflight_sha256=sha256(openings),
                evaluation_profile_sha256=panel.common['evaluation_profile_sha256'],
                source_context_protocol_sha256=sha256(root/SOURCE_CONTEXT/'protocol.json'),
                source_context_audit_sha256=sha256(root/CONTEXT_RUN/'audit.json'),
                data_directory=str(data.relative_to(root)),offline_runtime_path=str(runtime.relative_to(root)),
                candidate_review_path=str(review_path.relative_to(root)),adapter_directory=str(checkpoint.relative_to(root)),
                all_adapter_files_sha256=files,run_directory=str(run_directory))
    if (Counter(c['source']['language'] for c in panel.cases.values()) != design['language_counts']
            or sorted(plan['families'].values()) != sorted(design['families'].values())
            or {name.casefold() for name in plan['families']} & {name.casefold() for name in design['families']}):
        raise ValueError('The new development panel differs from the registered coverage')
    reviewed_adapter(review,files,plan)
    plan['selected_opening'] = approved_opening(root,plan)
    plan['schedule_sha256'] = fingerprint(schedule(list(panel.cases),plan))
    evidence = [review_path,runtime,openings,data/'cohort.private.json',data/'sources.json',data/'tasks.json']
    evidence += [checkpoint/name for name in files]
    evidence += [openings.parent/'first-checks'/f't{n:02d}.json' for n in panel.cases]
    evidence += [root/'private/parent-runtime.json',root/CONTEXT_RUN/'controller.json',root/CONTEXT_RUN/'audit.json']
    evidence += [root/relative(panel.common[name]) for name in ('environment_path','evaluation_profile_path','model_manifest_path')]
    evidence += [root/'experiments/032-multifile-completion/evaluation-protocol.json',
                 root/'experiments/038-independent-compact/protocol.json',
                 root/'experiments/036-regression-evidence/evidence.py']
    for directory in (EXPERIMENT,SOURCE_CONTEXT,Path('experiments/023-evaluation-stability'),
                      Path('experiments/024-feedback-noise'),Path('experiments/035-edit-bottleneck'),Path('one_gpu_lm')):
        evidence.extend((root/directory).rglob('*.py'))
    evidence += [root/EXPERIMENT/'design.json',root/SOURCE_CONTEXT/'design.json',root/SOURCE_CONTEXT/'protocol.json']
    plan['frozen_files'] = {str(path.relative_to(root)):sha256(path) for path in sorted(set(evidence))}
    with output.open('x',encoding='utf-8') as stream:
        json.dump(plan,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return {'stage':'frozen','cases':len(panel.cases),'rollouts':plan['rollout_count'],
            'protocol_sha256':sha256(output),'opening':plan['selected_opening'],'final_test_opened':False}


def registered_inputs(args):
    from promotion import comparison_registration

    plan, plan_hash, openings, openings_hash, order = comparison_registration(args.protocol,args.publication,args.openings)
    root = args.root
    data = root / relative(plan['data_directory'])
    runtime_path = root / relative(plan['offline_runtime_path'])
    review_path = root / relative(plan['candidate_review_path'])
    if any(plan['frozen_files'].get(str(path.relative_to(root))) != sha256(path)
           for path in (review_path,runtime_path,data/'cohort.private.json',data/'tasks.json',data/'sources.json')):
        raise ValueError('Missing frozen development or audited candidate evidence')
    if not frozen(root,plan['frozen_files']) or approved_opening(root,plan) != plan['selected_opening']:
        raise ValueError('The registered inputs or inference choice changed')
    review = read(review_path)
    files = plan['all_adapter_files_sha256']
    served = reviewed_adapter(review,adapter_files(root / relative(plan['adapter_directory'])),plan)
    panel = build_panel(root,data,args.openings,runtime_path)
    if (panel.proof != openings or Counter(case['source']['language'] for case in panel.cases.values()) != plan['language_counts']
            or sha256(args.protocol) != plan_hash or sha256(args.openings) != openings_hash):
        raise ValueError('Loaded candidate inputs differ from the registered comparison')
    panel.common.update(adapter_directory=plan['adapter_directory'],all_adapter_files_sha256=files,
                        adapter_sha256=served)
    return plan,plan_hash,openings_hash,order,panel


def evaluate(root, plan, plan_hash, order, panel):
    out = root / relative(plan['run_directory'])
    if len(str(out/'backend.sock').encode()) >= 108:
        raise ValueError('The registered backend socket path is too long')
    chosen = plan['selected_opening']
    budget = plan['evaluation_budget']
    with single_writer(gpu_lock_path()):
        if out.exists():
            raise FileExistsError('Preserve the original development attempt')
        verify_model(panel.model_runtime['model'],root/relative(panel.common['model_manifest_path']),panel.profile)
        sandbox = Sandbox(read(panel.runtime_path),budget['max_tool_seconds'])
        (out/'private').mkdir(mode=0o700,parents=True)
        start = time.time()
        deadline = start + plan['pipeline_limit_seconds']
        state = {'schema_version':1,'stage':'loading','complete':False,'started_epoch':start,'started_utc':utc(),
                 'deadline_epoch':deadline,'protocol_sha256':plan_hash,'schedule_sha256':plan['schedule_sha256'],
                 'runs':[],'backend_lifecycles':[],'optimizer_updates':0,'final_test_opened':False,'weights_public':False}

        def save():
            atomic_json(out/'controller.json',state)

        backend = JournalBackend(root,out,panel.model_runtime,panel.common,state['backend_lifecycles'],save)
        telemetry = Telemetry(out/'telemetry.csv',1)
        save()
        telemetry.start()
        try:
            for item in order:
                arm = item['arm']
                role = 'base' if arm == 'base' else 'student'
                required = budget['max_task_seconds'] + budget['max_tool_seconds'] + plan['recovery_accounting_seconds']
                if time.time() + required + (600 if backend.role != role else 0) > deadline:
                    raise TimeoutError('Registered wall budget cannot fit the next task')
                case = panel.cases[item['task_number']]
                state.update(stage='evaluating',current_run=item)
                save()
                backend.protocol = effective(panel.common,arm)
                socket = backend.ensure(role,deadline)
                directory = out/'private/tasks'/item['run_id']
                result = rollout(root,panel.data,case['prompts'][chosen],case['source'],socket,sandbox,
                                 backend.protocol,directory,role+'_current',journal_path=backend.journal_path,
                                 recovery_seconds=plan['recovery_accounting_seconds'])
                backend.check_exclusive()
                steps = read(directory/'trajectory.json')['steps']
                opening = case['opening']['openings'][chosen]
                if (not steps or steps[0]['generation']['input_tokens_sha256'] != opening['input_sha256']
                        or steps[0]['generation']['input_tokens'] != opening['input_tokens']):
                    raise ValueError('Physical generation did not use the frozen paired opening')
                result.update(run_id=item['run_id'],task_number=item['task_number'],repetition=item['repetition'],
                              block=item['block'],contrast_arm=arm,backend_index=backend.number)
                write_json(directory/'result.json',result)
                state['runs'].append(result)
                save()
                print(json.dumps({'completed_rollouts':len(state['runs']),'planned_rollouts':len(order)}),flush=True)
            state.update(stage='completed',complete=True)
        except BaseException as error:
            state.update(stage='failed',error_type=type(error).__name__)
            (out/'private/error.log').write_text(traceback.format_exc(),encoding='utf-8')
            raise
        finally:
            try:
                backend.stop()
            finally:
                unchanged = frozen(root,plan['frozen_files'])
                state.update(backend_stopped=backend.process is None,telemetry=telemetry.stop(),
                             frozen_inputs_unchanged=unchanged,finished_epoch=time.time(),finished_utc=utc())
                if not unchanged:
                    state.update(stage='inputs_changed',complete=False)
                save()
    return state


def audit(root, plan, plan_hash, openings_hash, order, panel):
    out = root / relative(plan['run_directory'])
    controller = out/'controller.json'
    target = out/'audit.json'
    state = read(controller)
    if (target.exists() or not state['complete'] or not state['backend_stopped'] or not state['frozen_inputs_unchanged']
            or state['final_test_opened'] or state['weights_public'] or state['optimizer_updates']
            or state['protocol_sha256'] != plan_hash or state['schedule_sha256'] != plan['schedule_sha256']
            or not state['started_epoch'] < state['finished_epoch'] <= state['deadline_epoch']
            or [row['run_id'] for row in state['runs']] != [item['run_id'] for item in order]):
        raise ValueError('The complete registered controller is required for replay')
    prior = root/'experiments/035-edit-bottleneck'
    sys.path.insert(0,str(prior))
    try:
        replay = load('candidate_replay',prior/'audit.py')
    finally:
        sys.path.pop(0)
    norm = normalization(root)
    chosen = plan['selected_opening']
    environment = read(root/relative(panel.common['environment_path']))
    backends, journals, artifacts = {},{},{}
    for backend in state['backend_lifecycles']:
        index,role = backend['index'],backend['role']
        arm = {'base':'base','student':'candidate'}[role]
        directory = out/'private/backends'/f'{index:04d}-{role}'
        artifact = read(directory/'served-artifact.json')
        condition = effective(panel.common,arm)
        if (index in backends or not state['started_epoch'] <= backend['ready_epoch'] <= backend['stopped_epoch'] <= state['finished_epoch']
                or (artifact['model_id'],artifact['revision']) != (panel.common['model_id'],panel.common['revision'])
                or artifact['adapter'] is not (role=='student') or artifact['adapter_sha256'] != condition['adapter_sha256']
                or artifact['profile_sha256'] != panel.common['evaluation_profile_sha256']
                or read(directory.with_name(directory.name+'-environment.json')) != environment):
            raise ValueError('Audit detected a different checkpoint, decoding profile or environment')
        backends[index],artifacts[index],journals[index] = backend,artifact,RequestJournal(directory/'requests')
    seen,rows = defaultdict(set),[]
    with TemporaryDirectory(prefix='candidate-replay-',dir=out/'private') as directory:
        for item,result in zip(order,state['runs']):
            number,index,arm = item['task_number'],result['backend_index'],item['arm']
            case = panel.cases[number]
            task,opening = case['prompts'][chosen],case['opening']['openings'][chosen]
            run_dir = out/'private/tasks'/item['run_id']
            trace = read(run_dir/'trajectory.json')
            if (backends[index]['role'] != ('base' if arm=='base' else 'student') or trace['backend'] != artifacts[index]
                    or result['id'] != task['id'] or result['contrast_arm'] != arm
                    or any(result[key] != item[key] for key in ('task_number','repetition','block'))):
                raise ValueError('Physical run differs from the paired schedule')
            row,first,requests = replay.audited_row(
                root,panel.data,run_dir,result,task,case['source'],effective(panel.common,arm),panel.profile,panel.tokenizer,
                norm,journals[index],Path(directory)/item['run_id'],case['before'],backends[index])
            if (first != opening['input_sha256'] or trace['steps'][0]['generation']['input_tokens'] != opening['input_tokens']
                    or seen[index] & requests):
                raise ValueError('Opening identity or physical request receipts differ')
            seen[index].update(requests)
            row.update(run_id=item['run_id'],repetition=item['repetition'],block=item['block'],
                       multifile=case['opening']['multifile'],opening_input_tokens=opening['input_tokens'],
                       opening_input_sha256=first)
            rows.append(row)
    if (set(seen) != set(journals)
            or any(seen[i] != {p.name for p in journal.directory.iterdir()} for i,journal in journals.items())
            or sum(map(len,seen.values())) != sum(row['attempted_calls'] for row in rows)):
        raise ValueError('Unaccounted backend journal entries')
    profile_hash = panel.common['evaluation_profile_sha256']
    result = {'schema_version':1,'passed':True,'protocol_sha256':plan_hash,'controller_sha256':sha256(controller),
              'opening_preflight_sha256':openings_hash,'verified_requests':sum(map(len,seen.values())),
              'started_epoch':state['started_epoch'],'finished_epoch':state['finished_epoch'],'backend_loads':len(backends),
              'served_checkpoints':{'base':{'adapter_files_sha256':{},'profile_sha256':profile_hash},
                                    'candidate':{'adapter_files_sha256':plan['all_adapter_files_sha256'],
                                                 'profile_sha256':profile_hash}},
              'telemetry':state['telemetry'],'rows':rows,'analysis':analyze(rows,plan),
              'optimizer_updates':0,'final_test_opened':False,'weights_public':False}
    atomic_json(target,result)
    return result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    modes = parser.add_subparsers(dest='mode',required=True)
    check = modes.add_parser('check-panel',help='Rebuild both opening options with the real tokenizer on CPU')
    for name in ('data','openings','runtime'):
        check.add_argument('--'+name,type=Path,required=True)
    freeze = modes.add_parser('freeze',help='Freeze the ready candidate and development comparison before registration')
    for name in ('data','openings','runtime','adapter','candidate-review','run-directory','protocol'):
        freeze.add_argument('--'+name,type=Path,required=True)
    for mode in ('run','audit'):
        command = modes.add_parser(mode)
        for name in ('protocol','publication','openings'):
            command.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.mode == 'check-panel':
        panel = build_panel(args.root,args.data,args.openings,args.runtime)
        result = {'cases':len(panel.cases),'verified_openings':len(panel.cases)*len(SELECTIONS),
                  'opening_preflight_sha256':sha256(args.openings),'model_generations':0,'final_test_opened':False}
    elif args.mode == 'freeze':
        result = freeze_protocol(args)
    else:
        plan,plan_hash,openings_hash,order,panel = registered_inputs(args)
        if args.mode == 'run':
            def interrupted(*_):
                raise KeyboardInterrupt('Development comparison interrupted')

            for kind in (signal.SIGINT,signal.SIGTERM):
                signal.signal(kind,interrupted)
            state = evaluate(args.root,plan,plan_hash,order,panel)
            if state['complete'] is not True:
                raise ValueError('The registered comparison ended without complete evidence')
            result = {'stage':state['stage'],'rollouts':len(state['runs']),'final_test_opened':False}
        else:
            evidence = audit(args.root,plan,plan_hash,openings_hash,order,panel)
            result = {'audited_rollouts':len(evidence['rows']),'verified_requests':evidence['verified_requests'],
                      'development_gate_passed':evidence['analysis']['development_gate_passed'],'final_test_opened':False}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
