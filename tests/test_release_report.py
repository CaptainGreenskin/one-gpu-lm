import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('release_report', Path(__file__).resolve().parents[1]/'experiments/019-coding-release/report.py')
report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)


class ReleaseReportTests(unittest.TestCase):
    def fixture(self):
        policy = {'format':'native_roundtrip','context':'full','progress':'edit_checkpoint_v1'}
        protocol = {'task_spec_sha256':'tasks','admission_sha256':'admission',
            'frozen_files':{'code.py':'code'},'variants':{'shared':policy},
            'agent':{'max_model_calls':12,'max_total_generated_tokens':6144,'max_input_tokens':8192,'max_generated_tokens_per_call':512}}
        generation = {'adapter':False,'agent_policy':policy,'input_tokens':200,'generated_tokens':20,
            'context':{'policy':'full','dropped_rounds':0},'action_format_valid':True,'seconds':1,'hit_generation_limit':False}
        row = {'id':'t','passed':False,'agent_policy':policy,'variant':'shared','generations':[generation],
            'backend_errors':[],'calls':1,'generated_tokens':20,'seconds':2,'tool_seconds':.5,
            'exit_status':'Submitted','source_progress':[{'changed_editable_files':1}]}
        campaign = {'complete':True,'identity':{'ids':['t'],'tasks_sha256':'tasks','admission_sha256':'admission',
            'protocol_sha256':'protocol','variant':'shared','source_files':{'code.py':'code'}},'results':[row]}
        return campaign, protocol

    def test_observed_edit_does_not_become_a_success(self):
        campaign, protocol = self.fixture()
        value = report.validate_campaign(campaign,['t'],protocol,'protocol',False)
        self.assertEqual(value['passed'],0)
        self.assertEqual(value['failed_after_edit'],1)

    def test_missing_pair_wrong_adapter_and_changed_code_fail_closed(self):
        campaign, protocol = self.fixture()
        for alter in [lambda c:c['results'].append(copy.deepcopy(c['results'][0])),
                      lambda c:c['results'][0]['generations'][0].update(adapter=True),
                      lambda c:c['identity']['source_files'].update({'code.py':'new-code'})]:
            changed = copy.deepcopy(campaign);alter(changed)
            with self.assertRaises(ValueError):report.validate_campaign(changed,['t'],protocol,'protocol',False)

    def test_budget_and_context_changes_are_not_hidden(self):
        campaign, protocol = self.fixture()
        for alter in [lambda c:c['results'][0].update(calls=13),
                      lambda c:c['results'][0].update(generated_tokens=21),
                      lambda c:c['results'][0]['generations'][0].update(input_tokens=8193),
                      lambda c:c['results'][0]['generations'][0]['context'].update(dropped_rounds=1)]:
            changed = copy.deepcopy(campaign);alter(changed)
            with self.assertRaises(ValueError):report.validate_campaign(changed,['t'],protocol,'protocol',False)


class CompleteEvidenceFixture:
    """Synthetic metadata only; never used as an experiment result or model score."""
    def __init__(self, root, eligible=True):
        self.root = Path(root)
        def put(name, value):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);report.write_json(path,value)
        self.put=put
        digest=lambda name:report.sha256(self.root/name)
        plan=json.loads((report.EXP/'preparation-plan.json').read_text());put('preparation-plan.json',plan)
        inventory=[]
        for label,count,split in [('train',55,'train'),('dev',15,'dev'),('historical',24,'test'),('fresh',24,'test')]:
            for i in range(count):
                inventory.append({'id':f'{label}-{i}','repository':f'{label}/repo-{i%4}','revision':'synthetic',
                    'family':f'{label}-{i}','split':split,
                    'evaluation_role':'historical_final_regression_not_selection' if label=='historical' else label})
        put('task-inventory.json',inventory);put('admission.json',{'tasks':inventory})
        put('data-receipt.json',{'synthetic_test_fixture':True})
        data={'selected':[{'id':str(i),'tokens':100,'supervised_tokens':66} for i in range(330)],
              'receipt_sha256':digest('data-receipt.json'),'data_sha256':'synthetic-data'}
        put('data-selection.json',data);put('training-input-audit.json',{'synthetic_test_fixture':True})
        for name in ['low','high']:put('profile-'+name+'.json',{'synthetic_candidate':name})
        prototype,protocol=ReleaseReportTests().fixture()
        protocol.update(model_id='synthetic/base',revision='synthetic-revision',prepared_data_sha256=data['data_sha256'],
            admission_sha256=digest('admission.json'),training_records=330,supervised_tokens=330*66,repository_count=16)
        frozen=protocol['frozen_files']
        frozen.update({'experiments/019-coding-release/preparation-plan.json':digest('preparation-plan.json'),
            'runs/prepared-data/selection.json':digest('data-selection.json'),
            'experiments/019-coding-release/audit.py':'synthetic-audit','training-input-audit.json':digest('training-input-audit.json'),
            'experiments/011-coding-model-selection/weights-manifest.json':'synthetic-model-manifest',
            'private/legacy-functions.jsonl':'synthetic-legacy'})
        for name in ['low','high']:frozen['experiments/019-coding-release/profile-'+name+'.json']=digest('profile-'+name+'.json')
        put('protocol.json',protocol);ph=digest('protocol.json')
        order=list(data['selected']);report.random.Random(101).shuffle(order)
        schedule=[{'example_sha256':report.hashlib.sha256(r['id'].encode()).hexdigest(),'tokens':r['tokens'],'supervised_tokens':r['supervised_tokens']} for r in order]
        audit={'audit_source_sha256':'synthetic-audit','selection_recomputed_without_test_scores':True,
            'original_wall_budget_respected':True,'training':{},'campaign_sha256':{},'variants':{}}
        integrity={'protocol_sha256':ph,'public_task_inventory_identity_sha256':report.identity_digest(inventory),
            'all_frozen_file_hashes_match':True,'source_snapshot_contents_unchanged':True,
            'dependency_contents_unchanged':True,'framework_contents_unchanged':True,
            'base_model_verification':{'content_verified':True,'manifest_sha256':'synthetic-model-manifest',
                'model_id':protocol['model_id'],'revision':protocol['revision']},
            'training_input_audit_sha256':digest('training-input-audit.json'),'adapter_tensor_audits':{},
            'rights_and_secret_scan_pass':True,'export':{'created':eligible,
                'adapter_files_match_training_and_evaluated_checkpoint':eligible,'candidate':'low' if eligible else None}}
        for name in ['low','high']:
            folder='training-'+name+'/'
            files={'adapter_model.safetensors':'synthetic-'+name}
            result={'status':'completed','steps':330,'supervised_tokens':330*66,'trainable_parameters':58,
                'order_sha256':'synthetic-same-order','adapter_files':files,
                **{k:True for k in ['frozen_sample_unchanged','frozen_gradients_absent','adapter_changed','saved_reload_greedy_ids_equal']}}
            put(folder+'result.json',result)
            put(folder+'identity.json',{'input_hashes':{'data':data['data_sha256'],'data_receipt':data['receipt_sha256'],
                'profile':digest('profile-'+name+'.json')},'source_files':{'code.py':'code'}})
            put(folder+'selection.json',{'schedule':schedule,'excluded_records':0,'selected_records':330})
            put(folder+'metrics.json',[{'step':i+1,'loss':1.,'grad_norm':1.,'supervised_tokens':66} for i in range(330)])
            audit['training'][name]={'adapter_files':files,'checkpoint_verified':True}
            integrity['adapter_tensor_audits'][name]={'parameters':58,'dtypes':['F32'],'file_sha256':files['adapter_model.safetensors']}
        campaigns={}
        arms=['dev-base','dev-low','dev-high']+(['final-base','final-selected'] if eligible else [])
        for arm in arms:
            selected=[t for t in inventory if t['split']==('dev' if arm.startswith('dev-') else 'test')]
            ids=[t['id'] for t in selected];candidate=not arm.endswith('-base')
            campaign=copy.deepcopy(prototype);campaign['identity'].update(ids=ids,
                tasks_sha256=protocol['task_spec_sha256'],admission_sha256=protocol['admission_sha256'],protocol_sha256=ph)
            campaign['results']=[]
            for t in selected:
                row=copy.deepcopy(prototype['results'][0]);row.update({k:t[k] for k in ['id','repository','family','split']})
                i=int(t['id'].rsplit('-',1)[1]);threshold=6
                if eligible and arm=='dev-low':threshold=7
                if t['evaluation_role']=='fresh' and arm=='final-selected':threshold=9
                row['passed']=i<threshold;row['generations'][0]['adapter']=candidate
                campaign['results'].append(row)
            campaigns[arm]=campaign;put('campaigns/'+arm+'.json',campaign)
            audit['campaign_sha256'][arm]=digest('campaigns/'+arm+'.json')
            audit['variants'][arm]=[{'id':t['id'],'progress_and_native_turns_verified':True,
                'final_source_verified':True,'input_token_hashes_verified':1} for t in selected]
            candidate_name='low' if arm=='final-selected' else arm.removeprefix('dev-')
            put('artifacts/'+arm+'.json',{'model_id':protocol['model_id'],'revision':protocol['revision'],
                'profile_sha256':digest('profile-low.json'),'adapter':candidate,
                'adapter_sha256':{'adapter_model.safetensors':'synthetic-'+candidate_name} if candidate else {}})
            put('loading/'+arm+'.json',{})
            if arm.startswith('final-'):
                put('legacy-'+arm+'.json',{'complete':True,'task_sha256':'synthetic-legacy',
                    'results':[{'id':str(i),'adapter':candidate,'passed':True} for i in range(9)]})
        selection=report.select(campaigns['dev-base'],{n:campaigns['dev-'+n] for n in ['low','high']},
            [t['id'] for t in inventory if t['split']=='dev'],plan)
        selection['final_test_opened']=eligible;put('selection.json',selection)
        put('execution-audit.json',audit);put('postrun-integrity.json',integrity)
        put('loaded-controls.json',{'complete':True,'task_spec_sha256':protocol['task_spec_sha256'],'clean':[{'passed':True} for _ in range(16)]})
        put('controller.json',{'complete':True,'stage':'completed','protocol_sha256':ph,
            'telemetry':{'thread_stopped':True,'errors':0},'started_utc':'2026-01-01T00:00:00+00:00',
            'finished_utc':'2026-01-01T00:00:10+00:00','started_epoch':0,'finished_epoch':10,'deadline_epoch':43200})
        put('preregistration.json',{'anonymous_files_verified':True,'ci_conclusion':'success',
            'protocol_sha256':ph,'commit':'synthetic-commit','verified_utc':'2025-12-31T23:59:00+00:00'})
        put('launch.json',{'source_commit':'synthetic-commit'})
        (self.root/'telemetry-attempt-1.csv').write_text('utc,temperature_c,memory_used_mib,power_w,utilization_percent\n2026-01-01T00:00:00+00:00,40,50000,200,50\n2026-01-01T00:00:05+00:00,41,50000,210,60\n')
        self.reseal_collection()

    def reseal_collection(self):
        self.put('collection.json',{'files':{str(p.relative_to(self.root)):report.sha256(p)
            for p in self.root.rglob('*') if p.is_file() and p.name!='collection.json'}})


class ReleaseReportIntegrationTests(unittest.TestCase):
    def test_complete_fixture_requires_consistent_export_and_variant_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CompleteEvidenceFixture(directory)
            self.assertTrue(report.assess(directory)['release_decision']['accepted'])
            path=Path(directory)/'postrun-integrity.json';v=json.loads(path.read_text());v['export']['created']=False
            fixture.put('postrun-integrity.json',v);fixture.reseal_collection()
            result=report.assess(directory)
            self.assertFalse(result['release_decision']['accepted'])
            self.assertIn('all_execution_audits_pass',result['release_decision']['failed_gates'])
            p=Path(directory)/'artifacts/dev-low.json';v=json.loads(p.read_text());v['adapter_sha256']={'adapter_model.safetensors':'synthetic-high'}
            fixture.put('artifacts/dev-low.json',v);fixture.reseal_collection()
            with self.assertRaisesRegex(ValueError,'served model identity'):report.assess(directory)

    def test_historical_fresh_relabeling_cannot_preserve_an_accepted_report(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CompleteEvidenceFixture(directory)
            p=Path(directory)/'task-inventory.json';v=json.loads(p.read_text())
            a=next(t for t in v if t['id']=='historical-0');b=next(t for t in v if t['id']=='fresh-0')
            a['evaluation_role'],b['evaluation_role']=b['evaluation_role'],a['evaluation_role']
            fixture.put('task-inventory.json',v);fixture.reseal_collection()
            with self.assertRaisesRegex(ValueError,'frozen private task manifest'):report.assess(directory)

    def test_no_development_winner_keeps_final_scores_unopened(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CompleteEvidenceFixture(directory,eligible=False)
            value=report.assess(directory)
            self.assertFalse(value['release_decision']['accepted']);self.assertIsNone(value['selection']['selected'])
            fixture.put('campaigns/final-base.json',{'complete':True});fixture.reseal_collection()
            with self.assertRaisesRegex(ValueError,'opened final scores'):report.assess(directory)


if __name__ == '__main__':unittest.main()
