import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.runstate import identity_digest
from one_gpu_lm.verified.first_action_loss import fingerprint
from one_gpu_lm.verified.reload_probe import PROBES, probe_requests
from one_gpu_lm.verified.runtime_identity import PACKAGES


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/039-release-development'
entrypoints = ('analysis','candidate_comparison','candidate_release','candidate_restart','promotion','recovery','host_restart',
               'final_campaign','final_audit','release_package','report','run')
original_modules = {name:sys.modules.pop(name) for name in entrypoints if name in sys.modules}
sys.path.insert(0, str(EXP))
try:
    from analysis import analyze, schedule
    import candidate_comparison
    import candidate_release
    import final_audit
    import final_campaign
    import host_restart
    import release_package
    from promotion import REFERENCE, authorize, authorize_comparison, read, release_decision
    from recovery import served_weight_hashes, validate_amendment
    from report import reproduce_analysis
finally:
    sys.path.pop(0)
    for name in entrypoints:
        sys.modules.pop(name,None)
    sys.modules.update(original_modules)


class ModelScopePromotionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.plan_path = EXP/'protocol.json'
        self.plan = read(self.plan_path)
        self.amendment_path = EXP/'recovery-plan.json'
        self.amendment = read(self.amendment_path)

    def save(self, name, value):
        path = self.directory/name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def synthetic_audit(self, gain):
        families = [f for f,n in self.plan['families'].items() for _ in range(n)]
        rows = [
            dict(item, family=families[item['task_number']-1], audit_passed=True,
                 completed_repair=gain and item['arm']=='candidate', attempted_calls=1,
                 input_tokens=20, generated_tokens=10, actor_seconds=1, opening_input_tokens=20,
                 no_edit=False, correct_without_submission=False, timeout_recovered=False,
                 first_edit_call=1)
            for item in schedule(list(range(1,61)),self.plan)
        ]
        rows = [{k:v for k,v in r.items() if k!='run_id'} for r in rows]
        result = analyze(rows,self.plan)
        return dict(passed=True, final_test_opened=False, weights_public=False, optimizer_updates=0,
                    protocol_sha256=sha256(self.plan_path), rows=rows, analysis=result, verified_requests=len(rows),
                    started_epoch=1, finished_epoch=200,
                    recovery=dict(self.amendment['incident'],plan_sha256=sha256(self.amendment_path),resumed_rollouts=350))

    def test_recovery_requires_exact_weights_and_published_hashes(self):
        files = self.plan['all_adapter_files_sha256']
        self.assertEqual(served_weight_hashes(files),{'adapter_model.safetensors':files['adapter_model.safetensors']})
        with self.assertRaises(ValueError):
            served_weight_hashes({'adapter_model.safetensors':files['adapter_model.safetensors']})
        pub = dict(protocol_sha256=sha256(self.plan_path),recovery_plan_sha256=sha256(self.amendment_path),
                   anonymous_files_verified=True,ci_conclusion='success')
        _, frozen = validate_amendment(ROOT,self.plan_path,self.plan,self.save('publication',pub))
        for name,digest in self.amendment['frozen_replacements'].items():
            self.assertEqual(frozen[name],digest)
        restart = ROOT/host_restart.RESTART_AMENDMENT
        restart_inputs = self.directory/'registered'
        for source in (self.amendment_path,restart):
            location = restart_inputs/source.relative_to(ROOT)
            location.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,location)
        receipt = restart_inputs/host_restart.PRIVATE/'restart-publication.json'
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps(dict(pub,restart_plan_sha256=sha256(restart))))
        _, continued = host_restart.validate_amendment(restart_inputs,self.plan_path,self.plan,frozen)
        for name in ('run.py','audit.py','recovery.py','host_restart.py'):
            path = EXP/name
            self.assertEqual(continued[str(path.relative_to(ROOT))],sha256(path))
        receipt.write_text(json.dumps(dict(pub,restart_plan_sha256='bad')))
        with self.assertRaisesRegex(ValueError,'verified public restart amendment'):
            host_restart.validate_amendment(restart_inputs,self.plan_path,self.plan,frozen)
        with self.assertRaises(ValueError):
            validate_amendment(ROOT,self.plan_path,self.plan,self.save('publication',dict(pub,recovery_plan_sha256='bad')))

    def test_failed_or_incomplete_development_never_reads_the_seal(self):
        audit = self.synthetic_audit(False)
        absent = self.directory/'absent'
        result = authorize(self.plan_path,self.save('audit',audit),absent,absent)
        self.assertFalse(result['approved'])
        audit['rows'].pop()
        audit.update(analysis=analyze(audit['rows'],self.plan),verified_requests=359)
        with self.assertRaises(ValueError):
            authorize(self.plan_path,self.save('audit',audit),absent,absent)

    def comparison_fixture(self, gain):
        plan = {key:self.plan[key] for key in ('pretrained_model','base_revision','task_count','language_counts',
                'repetitions','rollout_count','block_size','evaluation_budget','pipeline_limit_seconds',
                'recovery_accounting_seconds','gate','optimizer_updates_during_development','final_test_opened',
                'weight_publication_allowed_by_development','all_adapter_files_sha256')}
        families = [f'example/development-{i}' for i in range(6)]
        plan.update(comparison_kind='independent_adapter_comparison', families=dict.fromkeys(families,10),
                    selected_opening='original',cohort_sha256='c'*64,evaluation_profile_sha256='d'*64)
        preflight = dict(cohort_sha256=plan['cohort_sha256'],profile_sha256=plan['evaluation_profile_sha256'],
                         budget=plan['evaluation_budget'],model_id=plan['pretrained_model'],revision=plan['base_revision'],
                         model_generations=0,optimizer_updates=0,final_test_opened=False,rows=[])
        for n in range(1,61):
            code = hashlib.sha256(str(n).encode()).hexdigest()
            preflight['rows'].append(dict(task_number=n,family=families[(n-1)//10],
                                          openings={'original':{'input_sha256':code,'input_tokens':20}}))
        proof = self.save('opening-preflight',preflight)
        order = schedule(list(range(1,61)),plan)
        plan.update(opening_preflight_sha256=sha256(proof),schedule_sha256=fingerprint(order))
        protocol = self.save('new-protocol',plan)
        registration = self.save('published-comparison',dict(protocol_sha256=sha256(protocol),
            anonymous_files_verified=True,ci_conclusion='success',commit='f'*40))
        controller = self.save('new-controller',dict(stage='completed',complete=True,backend_stopped=True,
            frozen_inputs_unchanged=True,final_test_opened=False,weights_public=False,optimizer_updates=0,
            protocol_sha256=sha256(protocol),schedule_sha256=plan['schedule_sha256'],started_epoch=1,finished_epoch=200,
            deadline_epoch=1+plan['pipeline_limit_seconds'],runs=[dict(item,contrast_arm=item['arm']) for item in order]))
        rows = [dict(item,family=preflight['rows'][item['task_number']-1]['family'],audit_passed=True,
            completed_repair=gain and item['arm']=='candidate',attempted_calls=1,input_tokens=20,
            generated_tokens=10,actor_seconds=1,opening_input_tokens=20,no_edit=False,
            correct_without_submission=False,timeout_recovered=False,first_edit_call=1,
            opening_input_sha256=preflight['rows'][item['task_number']-1]['openings']['original']['input_sha256'])
            for item in order]
        audit = self.save('new-audit',dict(passed=True,final_test_opened=False,weights_public=False,optimizer_updates=0,
            protocol_sha256=sha256(protocol),controller_sha256=sha256(controller),opening_preflight_sha256=sha256(proof),
            started_epoch=1,finished_epoch=200,rows=rows,analysis=analyze(rows,plan),verified_requests=len(rows),
            served_checkpoints={arm:dict(adapter_files_sha256=files,profile_sha256=plan['evaluation_profile_sha256'])
                                for arm,files in [('base',{}),('candidate',plan['all_adapter_files_sha256'])]}))
        return protocol,controller,audit,registration,proof

    def test_fresh_comparison_checks_both_openings_before_the_seal(self):
        evidence = self.comparison_fixture(False)
        absent = self.directory/'must-stay-unopened'
        refusal = authorize_comparison(*evidence,absent,absent)
        self.assertEqual(refusal['stage'],'development_gate')
        self.assertFalse(refusal['approved'])
        audit = read(evidence[2])
        audit['rows'][0]['opening_input_sha256'] = '0'*64
        self.save(evidence[2].name,audit)
        with self.assertRaisesRegex(ValueError,'different task, order or opening'):
            authorize_comparison(*evidence,absent,absent)

    def test_fresh_release_entry_does_not_touch_final_suite_when_candidate_fails(self):
        protocol,controller,audit,publication,openings = self.comparison_fixture(False)
        absent = self.directory/'must-stay-unopened'
        for action in ('check','score','audit','package'):
            with self.subTest(action=action):
                result = candidate_release.advance(SimpleNamespace(protocol=protocol,controller=controller,audit=audit,
                    publication=publication,openings=openings,seal=absent,input_review=absent,action=action))
                self.assertEqual(result,dict(approved=False,stage='development_gate',
                                             final_test_opened=False,weights_public=False))

    def test_fresh_release_scores_using_the_audited_training_profile(self):
        absent = self.directory/'unopened-fixture'
        args = SimpleNamespace(protocol=absent,controller=absent,audit=absent,publication=absent,openings=absent,
            seal=absent,input_review=absent,action='score',sealed=absent,adapter=absent,training_protocol=absent,
            training_profile=absent,training_integrity=absent)
        permission = {'approved':True,'stage':'final_evaluation'}
        profile = {'actual': 'reviewed training profile'}
        with patch.object(candidate_release,'authorize_comparison',return_value=permission), \
                patch.object(candidate_release,'reviewed_training',return_value=(profile,{})) as training, \
                patch.object(candidate_release,'run_approved',return_value={'stage':'fixture-scheduled'}) as run:
            self.assertEqual(candidate_release.advance(args),{'stage':'fixture-scheduled'})
        training.assert_called_once_with(permission,absent,absent,absent,absent)
        run.assert_called_once_with(absent.resolve(),absent,permission,adapter_profile=profile)

    def test_comparison_records_both_weight_arms_and_rejects_extra_requests(self):
        work = self.directory
        environment = {'fixture_runtime':True}
        families = ('example/development-0','example/development-1')
        family_paths = {family: self.save(f'offline-runtime-{i}',{'marker':i})
                        for i,family in enumerate(families)}
        family_hashes = {family:sha256(path) for family,path in family_paths.items()}
        runtime_path = self.save('runtime-map',{'schema_version':1,'final_test_opened':False,'runtimes':{
            family:{'path':path.name,'sha256':family_hashes[family]} for family,path in family_paths.items()}})
        self.assertEqual(candidate_comparison.runtime_files(work,runtime_path,families),
                         {family:path.resolve() for family,path in family_paths.items()})
        with self.assertRaisesRegex(ValueError,'full development panel'):
            candidate_comparison.runtime_files(work,runtime_path,families[:1])
        env_path = self.save('environment',environment)
        plan = dict(self.plan,task_count=2,repetitions=1,rollout_count=4,block_size=2,
                    families=dict.fromkeys(families,1),selected_opening='original',
                    run_directory='runs/fixture',frozen_files={})
        order = schedule([1,2],plan)
        plan['schedule_sha256'] = fingerprint(order)
        files = plan['all_adapter_files_sha256']
        common = dict(model_id=plan['pretrained_model'],revision=plan['base_revision'],
                      adapter_directory='private/adapter',all_adapter_files_sha256=files,
                      adapter_sha256=served_weight_hashes(files),evaluation_profile_sha256='f'*64,
                      environment_path=env_path.name,model_manifest_path='fixture-model')
        cases = {}
        for number,family in enumerate(families,1):
            task = {'id':f'case-{number}','family':family,'description':f'opening {number}'}
            cases[number] = {'prompts':{'original':task},'source':{},'before':{},'opening':{
                'multifile':False,'openings':{'original':{'input_sha256':fingerprint(task),'input_tokens':7}}}}
        proof = {'family_runtime_sha256':family_hashes}
        panel = candidate_comparison.Panel(work,runtime_path,{'model':'fixture'},common,{},None,proof,cases,family_paths)
        lock = work/'device.lock'
        used = []

        class JournalProvider:
            def __init__(self, root, out, runtime, protocol, record, changed):
                self.out,self.protocol,self.record = out,protocol,record
                self.number,self.role,self.process = 0,None,None

            def ensure(self, role, deadline):
                if self.role != role:
                    self.stop()
                    self.role,self.process = role,object()
                    self.number += 1
                    self.record.append(dict(index=self.number,role=role,ready_epoch=time.time()))
                    directory = self.out/'private/backends'/f'{self.number:04d}-{role}'
                    self.journal_path = directory/'requests'
                    self.journal_path.mkdir(parents=True)
                    write_json(directory.with_name(directory.name+'-environment.json'),environment)
                    self.artifact = dict(model_id=common['model_id'],revision=common['revision'],
                                         adapter=role=='student',adapter_sha256=self.protocol['adapter_sha256'],
                                         profile_sha256=common['evaluation_profile_sha256'])
                    write_json(directory/'served-artifact.json',self.artifact)
                return self.artifact

            def check_exclusive(self):
                pass

            def stop(self):
                if self.process is not None:
                    self.record[-1]['stopped_epoch'] = time.time()
                self.role,self.process = None,None

        def record(root,data,task,source,backend,sandbox,protocol,directory,arm,*,journal_path,**options):
            used.append((arm,task['description'],protocol['adapter_sha256'],sandbox['marker']))
            directory.mkdir(parents=True)
            write_json(directory/'trajectory.json',{
                'backend':backend,'steps':[{'generation':{'input_tokens':7,'input_tokens_sha256':fingerprint(task)}}]})
            (journal_path/fingerprint(directory.name)).touch()
            return {'id':task['id']}

        def replay(root,data,directory,result,task,*arguments):
            row = dict(task_number=result['task_number'],arm=result['contrast_arm'],family=task['family'],
                       audit_passed=True,completed_repair=False,attempted_calls=1,input_tokens=7,
                       generated_tokens=1,actor_seconds=1,no_edit=True,correct_without_submission=False,
                       timeout_recovered=False,first_edit_call=None)
            return row,fingerprint(task),{fingerprint(directory.name)}

        with patch.multiple(candidate_comparison,JournalBackend=JournalProvider,rollout=record,
                            gpu_lock_path=lambda:lock,Sandbox=lambda data,_:data,verify_model=lambda *_:None,
                            host_boot=lambda:('fixture-boot',1),
                            Telemetry=lambda *_:SimpleNamespace(start=lambda:None,stop=lambda:{'thread_stopped':True})):
            state = candidate_comparison.evaluate(work,plan,'c'*64,order,panel)
        self.assertEqual(state['stage'],'completed')
        self.assertEqual([item[1] for item in used[:2]],[item[1] for item in used[2:]])
        self.assertEqual([bool(item[2]) for item in used],[False,False,True,True])
        self.assertEqual([item[3] for item in used],[row['task_number']-1 for row in order])
        self.assertEqual([row['offline_runtime_sha256'] for row in state['runs']],
                         [family_hashes[families[row['task_number']-1]] for row in order])
        extra = next((work/'runs/fixture/private/backends').glob('*/requests'))/'extra'
        extra.touch()
        with patch.multiple(candidate_comparison,load=lambda *_:SimpleNamespace(audited_row=replay),
                            normalization=lambda *_:None):
            with self.assertRaisesRegex(ValueError,'Unaccounted backend journal entries'):
                candidate_comparison.audit(work,plan,'c'*64,'d'*64,order,panel)
            extra.unlink()
            result = candidate_comparison.audit(work,plan,'c'*64,'d'*64,order,panel)
        self.assertEqual(result['verified_requests'],4)
        self.assertFalse(result['analysis']['development_gate_passed'])
        self.assertFalse(result['final_test_opened'])

    def test_fresh_comparison_can_promote_with_its_own_audit_and_the_original_final_gate(self):
        evidence = self.comparison_fixture(True)
        protocol = read(evidence[0])
        reference_path = REFERENCE/'protocol.json'
        reference = read(reference_path)
        inventory_path = REFERENCE/'results/2026-11-12/task-inventory.json'
        inventory = read(inventory_path)
        cohorts = {}
        for key,role in [('fresh','fresh_final_not_selection'),('historical','historical_final_regression_not_selection')]:
            ids = sorted(row['id'] for row in inventory if row.get('evaluation_role')==role)
            cohorts[key] = dict(tasks=len(ids),ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
        seal = self.save('fixture-seal',dict(reference_release_protocol_sha256=sha256(reference_path),
            public_task_inventory_sha256=sha256(inventory_path),development_protocol_sha256=sha256(evidence[0]),
            release_gate=reference['release_gate'],fresh_family_overlap=[],final_campaign_directories=0,
            final_test_opened=False,private_task_schema_read=False,authorized_model_scoring=False,
            cohorts=cohorts,frozen_file_digests=reference['frozen_files']))
        review = self.save('fixture-review',dict(audited_checkpoint_matches=True,
            eligible_for_existing_scope_of_development=True,model_id=protocol['pretrained_model'],
            revision=protocol['base_revision'],checkpoint_files_sha256=protocol['all_adapter_files_sha256'],
            sources={'example/training':{}},weights_public=False,final_test_opened=False))
        permission = authorize_comparison(*evidence,seal,review)
        self.assertTrue(permission['approved'])
        self.assertEqual(permission['cohorts'],{'fresh':24,'historical':48,'legacy':9})
        self.assertEqual(permission['checkpoint_sha256'],{n:protocol['all_adapter_files_sha256'][n]
                                                         for n in release_package.FILES})
        self.assertFalse(permission['final_test_opened'])
        different = dict(read(review),checkpoint_files_sha256={name:'0'*64 for name in release_package.FILES})
        with self.assertRaisesRegex(ValueError,'different from the measured candidate'):
            authorize_comparison(*evidence,self.directory/'must-stay-unopened',self.save('swapped',different))
        changed = dict(read(review),sources={'EXAMPLE/Development-0:variant':{}})
        with self.assertRaisesRegex(ValueError,'families must stay separate'):
            authorize_comparison(*evidence,self.directory/'must-stay-unopened',self.save('overlap',changed))

    def test_report_permits_duration_roundoff_and_checks_completed_counts_exactly(self):
        audit = self.synthetic_audit(False)
        audit['analysis']['costs']['base']['actor_seconds'] += 2e-12
        self.assertEqual(reproduce_analysis(audit,self.plan),audit['analysis'])
        absent = self.directory/'sealed-absent'
        self.assertEqual(authorize(self.plan_path,self.save('audit',audit),absent,absent)['stage'],
                         'development_gate')
        audit['analysis']['successes']['base'] += 1
        with self.assertRaisesRegex(ValueError,'Exact audited repetitions'):
            reproduce_analysis(audit,self.plan)
        with self.assertRaisesRegex(ValueError,'Exact audited repetitions'):
            authorize(self.plan_path,self.save('audit',audit),absent,absent)

    def test_final_entry_waits_for_the_completed_audited_controller(self):
        root = self.directory/'working'
        absent = self.directory/'unopened'
        status = root/final_campaign.PRIVATE/'pipeline-resumed.json'
        status.parent.mkdir(parents=True)
        status.write_text(json.dumps({'stage':'executing','complete':False,'final_test_opened':False}))
        with patch.object(final_campaign,'authorize',side_effect=AssertionError('The seal is still closed')):
            result = final_campaign.promotion_from_finished_run(root,absent,absent)
        self.assertEqual(result,{'approved':False,'stage':'development_pending','final_test_opened':False})
        self.assertFalse(absent.exists())

        restart_launch = status.parent/host_restart.RESTART_LAUNCH
        restart_launch.write_text('{}')
        status.write_text(json.dumps({'stage':'complete','complete':True,'final_test_opened':False}))
        with patch.object(final_campaign,'authorize',side_effect=AssertionError('The seal is still closed')):
            result = final_campaign.promotion_from_finished_run(root,absent,absent)
        self.assertEqual(result['stage'],'development_pending')
        restart_launch.unlink()

        plan = root/final_campaign.EXP/'protocol.json'
        plan.parent.mkdir(parents=True)
        plan.write_bytes(self.plan_path.read_bytes())
        path = root/final_campaign.RESUMED
        path.mkdir(parents=True)
        status.write_text(json.dumps({'stage':'complete','complete':True,'final_test_opened':False}))
        controller = {'stage':'completed','complete':True,'backend_stopped':True,'frozen_inputs_unchanged':True,
                      'started_epoch':1,'finished_epoch':200,'final_test_opened':False,'runs':[0]*360}
        control = path/'controller.json'
        control.write_text(json.dumps(controller))
        audit = dict(self.synthetic_audit(False),controller_sha256=sha256(control))
        (path/'audit.json').write_text(json.dumps(audit))
        result = final_campaign.promotion_from_finished_run(root,absent,absent)
        self.assertFalse(result['approved'])
        self.assertEqual(result['stage'],'development_gate')
        control.write_text(json.dumps(dict(controller,finished_epoch=201)))
        with self.assertRaisesRegex(ValueError,'completed controller'):
            final_campaign.promotion_from_finished_run(root,absent,absent)

    def test_swapped_checkpoint_is_rejected_before_any_sealed_read(self):
        adapter = self.directory/'wrong-adapter'
        adapter.mkdir()
        for name in release_package.FILES:
            (adapter/name).write_text('unexpected bytes')
        promotion = {'approved':True,'stage':'final_evaluation',
                     'checkpoint_sha256':{name:'c'*64 for name in release_package.FILES}}
        with patch.object(final_campaign,'verify_sealed_files',side_effect=AssertionError('Seal should remain closed')):
            with self.assertRaisesRegex(ValueError,'promoted adapter changed'):
                final_campaign.run_approved(self.directory/'unused-seal',adapter,promotion)

    def test_final_seal_verifier_checks_bytes_without_reading_task_schema(self):
        root = self.directory/'archive'
        tasks = root/'private/tasks.json'
        tasks.parent.mkdir(parents=True)
        tasks.write_bytes(b'sealed fixture bytes')
        protocol = root/final_campaign.SEALED_EXPERIMENT/'protocol.json'
        protocol.parent.mkdir(parents=True)
        value = {'frozen_files':{'private/tasks.json':sha256(tasks)}}
        protocol.write_text(json.dumps(value))
        with patch.object(final_campaign,'REFERENCE',protocol.parent):
            self.assertEqual(final_campaign.verify_sealed_files(root),value)
            tasks.write_bytes(b'changed sealed fixture bytes')
            with self.assertRaisesRegex(ValueError,'frozen final environment'):
                final_campaign.verify_sealed_files(root)

    def test_final_replay_rejects_changed_tokens_opening_and_workspace(self):
        directory = self.directory
        workspace = directory/'workspace'
        workspace.mkdir()
        (workspace/'module.py').write_text('answer = 2\n')
        original = {'module.py':sha256(workspace/'module.py')}
        tracker = EditableProgress(workspace,original,original)
        state = tracker.snapshot()
        opening = [('system','Fixture system'),('user','Fixture repair')]
        messages = [{'role':role,'content':content} for role,content in opening]
        _,progress = tracker.prepare_request(messages,0,12,True)
        protocol = read(REFERENCE/'protocol.json')
        policy,budget = protocol['variants']['shared'],protocol['agent']
        identity = {'agent_policy':policy,'backend':{'adapter':True}}
        tokens = [1,2,3]
        tokenizer = SimpleNamespace(apply_chat_template=lambda *_args,**_kwargs:tokens)
        generation = {'progress':progress,'input_tokens':len(tokens),'input_tokens_sha256':final_audit.json_sha256(tokens),
                      'generated_tokens':8,'agent_policy':policy,'adapter':True,'action_format_valid':True,
                      'dialogue':{'structured_tool_calls':0,'tool_results':0},
                      'context':{'policy':'full','raw_input_tokens':3,'packed_input_tokens':3,
                                 'dropped_rounds':0,'original_messages':2,'packed_messages':2}}
        messages += [{'role':'assistant','content':'```bash\necho hello\n```',
                      'extra':{'generation':generation,'actions':[{'command':'echo hello'}]}},
                     {'role':'user','content':'Exit code: 0\nhello','extra':{'tool':{'returncode':0}}}]
        steps = [dict(state,call_index=1)]
        result = {'agent_policy':policy,'calls':1,'generated_tokens':8,'generations':[generation],
                  'tool_calls':1,'source_progress':steps,'backend_errors':[],'source_sha256':original}
        trace = {'messages':messages,'local_model':{'config':budget,'policy':policy,'responses':[generation],
                                                   'generated_tokens':8,'backend_errors':[],'source_progress':steps},
                 'environment':{'tool_calls':1}}
        checkpoint = {'trajectory':trace,'identity':identity,'n_calls':1,'boundary':'completed_step','workspace_manifest':original}
        self.save('original.json',original)

        def replay(prompt):
            for name,value in [('trajectory.json',trace),('result.json',result),('checkpoint.json',checkpoint)]:
                self.save(name,value)
            return final_audit.replay_trace(directory,result,original,identity,budget,
                                            {'chat_template_kwargs':{}},tokenizer,prompt)

        self.assertEqual(replay(opening)['requests_replayed'],1)
        with self.assertRaisesRegex(ValueError,'Opening messages'):
            replay(opening[:1]+[('user','Different repair')])
        generation['input_tokens_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'input or progress'):
            replay(opening)
        (workspace/'module.py').write_text('answer = 5\n')
        with self.assertRaisesRegex(ValueError,'workspace manifest changed'):
            replay(opening)

    def test_final_gate_reuses_the_full_public_inventory(self):
        reference = REFERENCE/'protocol.json'
        inventory = REFERENCE/'results/2026-11-12/task-inventory.json'
        plan = read(reference)
        frozen = ('private/tasks.json','private/fresh-tasks.json','private/legacy-functions.jsonl',
                  'source-snapshots.json','runs/admission/admission.json','runs/admission-fresh-draft002/admission.json')
        ids = {key:sorted(r['id'] for r in read(inventory) if r.get('evaluation_role')==role)
               for key,role in (('fresh','fresh_final_not_selection'),('historical','historical_final_regression_not_selection'))}
        seal = dict(reference_release_protocol_sha256=sha256(reference),public_task_inventory_sha256=sha256(inventory),
                    development_protocol_sha256=sha256(self.plan_path),release_gate=plan['release_gate'],
                    fresh_family_overlap=[],final_campaign_directories=0,final_test_opened=False,
                    private_task_schema_read=False,authorized_model_scoring=False,
                    frozen_file_digests={name:plan['frozen_files'][name] for name in frozen},
                    cohorts={k:{'tasks':len(v),'ids_sha256':hashlib.sha256(
                        json.dumps(v,separators=(',',':')).encode()).hexdigest()} for k,v in ids.items()})
        review = dict(audited_checkpoint_matches=True,eligible_for_existing_scope_of_development=True,
                      model_id=self.plan['pretrained_model'],revision=self.plan['base_revision'],
                      checkpoint_files_sha256=self.plan['all_adapter_files_sha256'],weights_public=False,final_test_opened=False)
        audit = self.save('audit',self.synthetic_audit(True))
        authorization = authorize(self.plan_path,audit,self.save('seal',seal),self.save('review',review))
        self.assertEqual(authorization['cohorts'],{'fresh':24,'historical':48,'legacy':9})
        with self.assertRaises(ValueError):
            authorize(self.plan_path,audit,self.directory/'seal',self.save('review',dict(review,checkpoint_files_sha256={})))

        def scores(key, passes):
            return dict(complete=True,results=[
                dict(id=n,passed=i<passes,generated_tokens=10,generations=[dict(action_format_valid=True)])
                for i,n in enumerate(ids[key])])

        checks = dict(legacy_passes=9, save_reload_greedy_match=True, rights_and_secret_scan_pass=True,
                      all_execution_audits_pass=True,single_training_seed_and_repository_cluster_limits_disclosed=True)
        base, candidate, historical = scores('fresh',2), scores('fresh',5), scores('historical',6)
        self.assertTrue(release_decision(authorization,base,candidate,historical,historical,checks)['accepted'])
        regressed = release_decision(authorization,base,candidate,historical,scores('historical',5),checks)
        self.assertIn('historical_regressions',regressed['failed_gates'])
        with self.assertRaises(ValueError):
            release_decision(authorization,base,candidate,historical,dict(complete=True,results=historical['results'][:24]),checks)

    def test_reused_notice_readback_must_cover_the_reviewed_source_revision(self):
        repo = 'owner/training-source'
        base = {'model_id':'fixture/base','base_revision':'c'*40}
        weights = {name:'e'*64 for name in release_package.FILES}
        review = self.save('source-review.json',{'known_pattern_findings':{'fixture':0},
            'sources':{repo:{'revision':'a'*40,'notices':{'LICENSE':'b'*64}}}})
        permission = dict(base,checkpoint_sha256=weights,candidate_review_sha256=sha256(review))
        readback_data = dict(base,source_revisions={repo:'f'*40},notices={
            release_package.BASE_LICENSE:{'source_sha256':'d'*64,'byte_equal':True},
            'training-source-notices/owner--training-source/LICENSE':{'source_sha256':'b'*64,'byte_equal':True}})
        readback = self.save('readback.json',readback_data)
        rights_data = dict(base,schema_version=1,approved_for_weight_publication=True,rights_and_secret_scan_pass=True,
            publication_basis='Fixture only',release_license='Apache-2.0',adapter_files_sha256=weights,
            input_review_sha256=sha256(review),notice_readback_sha256=sha256(readback))
        rights = self.save('rights-review.json',rights_data)
        with self.assertRaisesRegex(ValueError,'different training-source revisions'):
            release_package.approved_notices(permission,review,rights,readback)
        self.save('readback.json',dict(readback_data,source_revisions={repo:'a'*40}))
        self.save('rights-review.json',dict(rights_data,notice_readback_sha256=sha256(readback)))
        license_id, notices = release_package.approved_notices(permission,review,rights,readback)
        self.assertEqual(license_id,'Apache-2.0')
        self.assertEqual(len(notices),2)

    def test_release_packaging_checks_development_before_reading_final_state(self):
        destination = self.directory/'staged'
        args = SimpleNamespace(root=self.directory,seal=self.directory/'absent',
                               input_review=self.directory/'absent',out=destination)
        with self.assertRaisesRegex(ValueError,'development_pending'):
            release_package.assemble(args)
        self.assertFalse(destination.exists())

    def test_release_packaging_binds_audit_rights_and_payload(self):
        # Synthetic scores and adapter bytes exercise the assembly without the sealed task files.
        reference = REFERENCE/'protocol.json'
        inventory = REFERENCE/'results/2026-11-12/task-inventory.json'
        plan = read(reference)
        review = self.save('inputs.json',{'known_pattern_findings':{'fixture':0},'sources':{}})
        identity = {'model_id':self.plan['pretrained_model'],'base_revision':self.plan['base_revision']}
        training = self.directory/'training'
        training.mkdir()
        adapter = self.directory/'adapter'
        adapter.mkdir()
        for name in release_package.FILES:
            (adapter/name).write_text(name)
        (adapter/'private-training-log.txt').write_text('private fixture')
        adapter_hashes = {name:sha256(adapter/name) for name in release_package.FILES}
        (training/'candidate-profile.json').write_bytes((release_package.TRAINING/'candidate-profile.json').read_bytes())
        integrity = training/'results/2026-11-17-training/integrity.json'
        integrity.parent.mkdir(parents=True)
        integrity.write_text(json.dumps({'passed':True,'adapter_files_sha256':adapter_hashes}))
        permission = dict(identity,approved=True,stage='final_evaluation',development_gain_pp=12.0,
                          development_audit_sha256='a'*64,checkpoint_sha256=adapter_hashes,
                          candidate_review_sha256=sha256(review),cohorts={'fresh':24,'historical':48,'legacy':9},
                          reference_release_protocol_sha256=sha256(reference),public_task_inventory_sha256=sha256(inventory))
        out = self.directory/release_package.FINAL
        out.mkdir(parents=True)
        control = out/'controller.json'
        control.write_text(json.dumps({'stage':'scored_audit_pending','complete':True,'final_test_opened':True,
                                       'weights_public':False,'optimizer_updates':0}))
        promotion = out/'promotion.json'
        promotion.write_text(json.dumps(permission))
        artifacts, repository, legacy, pairs = {}, {}, {}, []
        for arm in release_package.ARMS:
            rows = []
            for role in release_package.COHORTS:
                ids = [r['id'] for r in read(inventory) if r.get('evaluation_role') == role]
                amount = 5 if arm == 'final-selected' and role == release_package.COHORTS[0] else 2
                scores = [dict(id=n,passed=i<amount,generated_tokens=1,generations=[{'action_format_valid':True}])
                          for i,n in enumerate(ids)]
                pairs.append({'complete':True,'results':scores})
                rows.extend(scores)
            campaign = out/arm/'campaign.json'
            campaign.parent.mkdir()
            campaign.write_text(json.dumps({'complete':True,'results':rows}))
            artifact = out/('artifact-'+arm+'.json')
            artifact.write_text('{}')
            functions = out/('legacy-'+arm)/'result.json'
            functions.parent.mkdir()
            functions.write_text(json.dumps({'complete':True,'passed':9,'results':[{'passed':True}]*9}))
            artifacts[arm] = {'campaign_sha256':sha256(campaign),'served_artifact_sha256':sha256(artifact)}
            repository[arm] = [{'id':row['id'],'passed':row['passed']} for row in rows]
            legacy[arm] = {'result_sha256':sha256(functions),'scores_replayed':9}
        checks = {'legacy_passes':9,'all_execution_audits_pass':True}
        early_decision = release_decision(permission,pairs[0],pairs[2],pairs[1],pairs[3],checks)
        (out/'execution-audit.json').write_text(json.dumps(dict(
            execution_audit_passed=True,model_generations=0,final_test_opened=True,weights_public=False,
            controller_sha256=sha256(control),promotion_sha256=sha256(promotion),adapter_files_sha256=adapter_hashes,
            audit_source_sha256=sha256(EXP/'final_audit.py'),artifacts=artifacts,repository_replay=repository,
            legacy_replay=legacy,checks=checks,release_decision_with_execution_evidence=early_decision)))
        notice = self.save(release_package.BASE_LICENSE,'public license fixture')
        notice_readback = self.save('notices.json',dict(identity,notices={notice.name:{
            'byte_equal':True,'source_sha256':sha256(notice)}}))
        rights = dict(identity,schema_version=1,approved_for_weight_publication=False,
                      rights_and_secret_scan_pass=True,publication_basis='Fixture only',release_license='Apache-2.0',
                      adapter_files_sha256=adapter_hashes,input_review_sha256=sha256(review),
                      notice_readback_sha256=sha256(notice_readback))
        rights_path = self.save('rights.json',rights)
        captures = dict(schema_version=1,configuration=dict(identity,
            adapter_files_sha256=adapter_hashes,profile_sha256=plan['frozen_files'][release_package.PROFILE_KEY],
            base_manifest_sha256=plan['frozen_files'][release_package.BASE_MANIFEST],runtime_sha256='d'*64,
            probe_definition_sha256=identity_digest(probe_requests()),
            software={'python':'3.12.0','packages':dict.fromkeys(PACKAGES,'1.0')}),
            probes=[dict(name=name,input_tokens=6,generated_tokens=10,input_tokens_sha256='e'*64,
                         generated_ids_sha256='f'*64) for name,_ in PROBES])
        args = SimpleNamespace(root=self.directory,sealed=self.directory,seal=self.directory/'absent',
                               input_review=review,rights=rights_path,notice_readback=notice_readback,
                               notices=self.directory,adapter=adapter,repo_id='CaptainGreenskin/fixture',
                               before=self.save('before.json',dict(captures,capture_id='a'*32)),
                               after=self.save('after.json',dict(captures,capture_id='b'*32)),
                               out=self.directory/'staged')

        def copy_fixture(source,destination,_profile,hashes):
            destination.mkdir()
            for name in hashes:
                shutil.copyfile(source/name,destination/name)
            return {'adapter_files_sha256':hashes}

        with patch.object(release_package,'promotion_from_finished_run',return_value=permission), \
                patch.object(release_package,'TRAINING',training), \
                patch.object(release_package,'restore_adapter',side_effect=copy_fixture) as restore:
            original_scores = campaign.read_bytes()
            campaign.write_bytes(original_scores+b'\n')
            with self.assertRaisesRegex(ValueError,'Scored artifacts differ'):
                release_package.assemble(args)
            campaign.write_bytes(original_scores)
            with self.assertRaisesRegex(ValueError,'distribution and sensitive-content review'):
                release_package.assemble(args)
            restore.assert_not_called()
            self.assertFalse(args.out.exists())
            self.save('rights.json',dict(rights,approved_for_weight_publication=True))
            notice.write_text('different notice')
            with self.assertRaisesRegex(ValueError,'Notice bytes differ'):
                release_package.assemble(args)
            self.assertFalse(args.out.exists())
            self.save(notice.name,'public license fixture')
            manifest = release_package.assemble(args)

        payload = args.out/'payload'
        self.assertEqual(set(manifest['files']),{'README.md',notice.name,*release_package.FILES})
        self.assertEqual({p.name for p in payload.iterdir()},set(manifest['files']))
        for name,entry in manifest['files'].items():
            self.assertEqual(entry,{'bytes':(payload/name).stat().st_size,'sha256':sha256(payload/name)})
        model_card = (payload/'README.md').read_text()
        self.assertIn('| 新留出 | 24 | 2 | 5 |',model_card)
        self.assertIn('一个训练种子 `101`',model_card)
        evidence = read(args.out/'release-evidence.json')
        self.assertTrue(evidence['decision']['accepted'])
        self.assertFalse(evidence['weights_public'])

        successor_profile = read(training/'candidate-profile.json')
        successor_profile['training']['seed'] = 152
        successor_profile['lora']['r'] = 16
        args.training_profile = self.save('successor-profile.json', successor_profile)
        sources = 'c'*64
        data = 'd'*64
        args.training_protocol = self.save('successor-protocol.json', {
            'profile_path':'training/profile.json','data_path':'private/training.jsonl','sources_path':'private/sources.json',
            'frozen_files':{'training/profile.json':sha256(args.training_profile),'private/training.jsonl':data,
                            'private/sources.json':sources},'final_test_opened':False,'weights_public':False})
        args.training_integrity = self.save('successor-integrity.json', {
            'passed':True,'protocol_sha256':sha256(args.training_protocol),'adapter_files_sha256':adapter_hashes,
            'audit_model_generations':0,'audit_optimizer_updates':0,'optimizer_updates':18,
            'final_test_opened':False,'weights_public':False})
        self.save('inputs.json',dict(read(review),experiment='candidate-fixture',actual_training_data_sha256=data,
            source_manifest_sha256=sources,training_integrity_sha256=sha256(args.training_integrity),
            eligible_for_existing_scope_of_development=True,audited_checkpoint_matches=True,
            source_snapshots_match=True,all_source_notices_and_lineage_reviewed_for_training=True,
            model_id=identity['model_id'],revision=identity['base_revision'],checkpoint_files_sha256=adapter_hashes,
            final_test_opened=False,weights_public=False))
        permission['candidate_review_sha256']=sha256(review)
        promotion.write_text(json.dumps(permission))
        audit_path=out/'execution-audit.json'
        audit_value=read(audit_path)
        audit_value['promotion_sha256']=sha256(promotion)
        audit_path.write_text(json.dumps(audit_value))
        self.save('rights.json',dict(rights,approved_for_weight_publication=True,input_review_sha256=sha256(review)))
        args.out=self.directory/'successor-release'
        with patch.object(release_package,'TRAINING',self.directory/'absent-031'), \
                patch.object(release_package,'restore_adapter',side_effect=copy_fixture):
            args.training_profile.write_text(json.dumps(dict(successor_profile,training={'seed':1})))
            with self.assertRaisesRegex(ValueError,'frozen training protocol'):
                release_package.assemble_approved(args,permission,out)
            self.assertFalse(args.out.exists())
            self.save('successor-profile.json',successor_profile)
            with self.assertRaisesRegex(ValueError,'audited adapter profile'):
                release_package.assemble_approved(args,permission,out)
            self.assertFalse(args.out.exists())
            successor_captures = dict(captures, schema_version=2,configuration=dict(
                captures['configuration'],adapter_profile_sha256=sha256(args.training_profile)))
            self.save('before.json',dict(successor_captures,capture_id='a'*32))
            self.save('after.json',dict(successor_captures,capture_id='b'*32))
            release_package.assemble_approved(args,permission,out)
        card=(args.out/'payload/README.md').read_text()
        self.assertIn('candidate-fixture 使用一个训练种子 `152`',card)
        self.assertNotIn('031-correction-mix',card)
        self.assertNotIn('039 在六个',card)
        self.assertIn('秩为 16',card)
        self.assertEqual(read(args.out/'release-evidence.json')['training_lineage']['training_data_sha256'],data)
