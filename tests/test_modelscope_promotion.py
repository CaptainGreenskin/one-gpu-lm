import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.runstate import identity_digest
from one_gpu_lm.verified.reload_probe import PROBES, probe_requests
from one_gpu_lm.verified.runtime_identity import PACKAGES


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/039-release-development'
entrypoints = ('analysis','promotion','recovery','host_restart','final_campaign','final_audit','release_package')
original_modules = {name:sys.modules.pop(name) for name in entrypoints if name in sys.modules}
sys.path.insert(0, str(EXP))
try:
    from analysis import analyze, schedule
    import final_audit
    import final_campaign
    import host_restart
    import release_package
    from promotion import REFERENCE, authorize, read, release_decision
    from recovery import served_weight_hashes, validate_amendment
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
