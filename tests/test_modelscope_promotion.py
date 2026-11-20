import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.edit_progress import EditableProgress


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/039-release-development'
entrypoints = ('analysis','promotion','recovery','final_campaign','final_audit')
original_modules = {name:sys.modules.pop(name) for name in entrypoints if name in sys.modules}
sys.path.insert(0, str(EXP))
try:
    from analysis import analyze, schedule
    import final_audit
    import final_campaign
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
        for name in ('run.py','audit.py','recovery.py'):
            path = EXP/name
            self.assertEqual(frozen[str(path.relative_to(ROOT))],sha256(path))
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
