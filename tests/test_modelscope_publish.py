import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from one_gpu_lm.common.receipts import sha256


EXP = Path(__file__).resolve().parents[1]/'experiments/039-release-development'
ENTRYPOINTS = ('analysis','promotion','recovery','host_restart','final_campaign','final_audit','release_package','publish_package')
original = {name:sys.modules.pop(name) for name in ENTRYPOINTS if name in sys.modules}
sys.path.insert(0,str(EXP))
try:
    import publish_package as release
finally:
    sys.path.pop(0)
    for name in ENTRYPOINTS:
        sys.modules.pop(name,None)
    sys.modules.update(original)


class ModelScopePublishTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.payload = self.root/'payload'
        self.payload.mkdir()
        for name in ('README.md',release.BASE_LICENSE,*release.FILES):
            (self.payload/name).write_text(name,encoding='utf-8')
        self.files = {path.name:{'bytes':path.stat().st_size,'sha256':sha256(path)} for path in self.payload.iterdir()}
        self.repo = release.OWNER+'/fixture'
        self.manifest = {'repo_id':self.repo,'endpoint':release.ENDPOINT,'files':self.files}
        hashes = {name:self.files[name]['sha256'] for name in release.FILES}
        model,revision = 'fixture/base','0'*40
        self.evidence = {'schema_version':1,'manifest':self.manifest,'release_license':'Apache-2.0',
                         'final_test_opened':True,'weights_public':False,
                         'adapter':{'adapter_files_sha256':hashes,'model_id':model,'revision':revision},
                         'execution_audit_sha256':'1'*64,'rights_review_sha256':'2'*64,
                         'reload_captures_sha256':['3'*64,'4'*64],
                         'decision':{'accepted':True,'failed_gates':[],'gates':dict.fromkeys(release.RELEASE_GATES,True),
                                     'checkpoint_sha256':hashes,'model_id':model,'base_revision':revision}}
        self.evidence_path = self.root/'release-evidence.json'
        self.evidence_path.write_text(json.dumps(self.evidence))

    def test_package_check_requires_accepted_gates_and_exact_allowlisted_files(self):
        self.assertEqual(release.approved_package(self.root),(self.manifest,'apache-2.0',sha256(self.evidence_path)))
        unexpected = self.payload/'private-log.json'
        unexpected.write_text('{}')
        with self.assertRaisesRegex(ValueError,'payload inventory'):
            release.approved_package(self.root)
        unexpected.unlink()
        name = release.FILES[0]
        (self.payload/name).write_text('x'*self.files[name]['bytes'])
        with self.assertRaisesRegex(ValueError,'release bytes'):
            release.approved_package(self.root)
        self.evidence['decision']['gates']['rights_and_secret_scan_pass'] = False
        self.evidence_path.write_text(json.dumps(self.evidence))
        with self.assertRaisesRegex(ValueError,'accepted 039 release'):
            release.approved_package(self.root)

    def test_upload_hook_requires_an_existing_private_model(self):
        class SDK:
            def __init__(self, *, endpoint):
                self.endpoint = endpoint

            def create_repo(self, *_args):
                raise AssertionError('An upload must never create a repository')

        with patch.dict(sys.modules, {'modelscope_hub':SimpleNamespace(HubApi=SDK)}):
            hub = release.publishing_hub()
        self.assertEqual(hub.endpoint,release.ENDPOINT)
        private = SimpleNamespace(visibility=release.PRIVATE)
        public = SimpleNamespace(visibility=release.PUBLIC)
        with patch.object(release,'repository',side_effect=[None,public,private]) as lookup:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError,'existing private ModelScope'):
                    hub._create_repo_exist_ok(self.repo,'model')
            hub._create_repo_exist_ok(self.repo,'model')
            self.assertEqual(lookup.call_count,3)

    def test_private_commit_and_public_readback_resume_after_remote_writes(self):
        state_path = self.root/'modelscope-release.json'
        state = {'schema_version':1,'release_evidence_sha256':sha256(self.evidence_path)}
        initial = {'.gitattributes':{'bytes':4,'sha256':'c'*64}}
        pinned = dict(self.manifest,revision='b'*40,files=initial|self.files)
        url = release.ENDPOINT+'/models/'+self.repo
        remote = SimpleNamespace(info=None,head='a'*40,files=initial)
        hub = SimpleNamespace(legacy=SimpleNamespace(token='fixture-auth'))
        downloaded = self.root/'downloaded'

        def create(repo,kind,**settings):
            self.assertEqual((repo,kind,settings),(self.repo,'model',{'visibility':'private','license':'apache-2.0'}))
            remote.info = SimpleNamespace(id='fixture-repo',owner=release.OWNER,name='fixture',
                                          license='apache-2.0',visibility=release.PRIVATE)

        def upload(repo,kind,folder,**settings):
            self.assertEqual(remote.info.visibility,release.PRIVATE)
            self.assertEqual((repo,kind),(self.repo,'model'))
            self.assertEqual(settings['allow_patterns'],sorted(self.files))
            self.assertFalse(settings['sync_remote_repo'])
            release.verify_tree(folder,self.files)
            remote.head,remote.files = pinned['revision'],pinned['files']
            raise TimeoutError('fixture disconnect after committing')

        def visibility(repo,kind,**settings):
            self.assertEqual((repo,kind,settings),(self.repo,'model',{'visibility':'public'}))
            self.assertEqual(remote.head,pinned['revision'])
            remote.info.visibility = release.PUBLIC

        def head(_repo,token=None,**_kwargs):
            if token is None:
                self.assertEqual(remote.info.visibility,release.PUBLIC)
            return remote.head

        with patch.object(release,'repository',side_effect=lambda *_:remote.info), \
                patch.object(release,'branch_head',side_effect=head), \
                patch.object(release,'inventory',side_effect=lambda *_:remote.files), \
                patch.object(release,'verify_private_download') as private_read, \
                patch.object(release,'readback',side_effect=[RuntimeError('fixture download interruption'),
                                                            {'model_url':url,'anonymous_public_readback_verified':True}]) as public_read, \
                patch.object(hub,'create_repo',side_effect=create,create=True) as created, \
                patch.object(hub,'upload_folder',side_effect=upload,create=True) as uploaded, \
                patch.object(hub,'update_repo_settings',side_effect=visibility,create=True) as published:
            with self.assertRaises(TimeoutError):
                release.stage(hub,self.root,state_path,state,self.manifest,'apache-2.0')
            state = json.loads(state_path.read_bytes())
            self.assertEqual(release.stage(hub,self.root,state_path,state,self.manifest,'apache-2.0'),pinned)
            self.assertEqual(created.call_count,1)
            self.assertEqual(uploaded.call_count,1)
            self.assertEqual(remote.info.visibility,release.PRIVATE)
            private_read.assert_called_once_with(hub,pinned)

            state = json.loads(state_path.read_bytes())
            remote.head = 'd'*40
            with self.assertRaisesRegex(ValueError,'exact staged commit'):
                release.publish(hub,state_path,state,self.manifest,'apache-2.0',downloaded)
            published.assert_not_called()
            remote.head = pinned['revision']
            with self.assertRaisesRegex(RuntimeError,'fixture download interruption'):
                release.publish(hub,state_path,state,self.manifest,'apache-2.0',downloaded)
            self.assertEqual(remote.info.visibility,release.PUBLIC)
            state = json.loads(state_path.read_bytes())
            receipt = release.publish(hub,state_path,state,self.manifest,'apache-2.0',downloaded)
            self.assertTrue(receipt['anonymous_public_readback_verified'])
            self.assertEqual(published.call_count,1)
            self.assertEqual(public_read.call_count,2)
            public_read.assert_called_with(pinned,downloaded)

    def test_public_readback_recovers_local_snapshot_and_receipt_independently(self):
        pinned = dict(self.manifest, revision='b'*40)
        state = {'repo_uid':'fixture-repo', 'initial_files':{}, 'staged_manifest':pinned}
        state_path = self.root / 'modelscope-release.json'
        state_path.write_text(json.dumps(state))
        info = SimpleNamespace(id='fixture-repo', owner=release.OWNER, name='fixture',
                               license='apache-2.0', visibility=release.PUBLIC)
        hub = SimpleNamespace(legacy=SimpleNamespace(token='fixture-auth'))
        downloaded = self.root / 'downloaded'
        shutil.copytree(self.payload, downloaded)
        expected = {'model_url':release.ENDPOINT+'/models/'+self.repo,
                    'anonymous_public_readback_verified':True}

        def download(manifest, directory):
            self.assertEqual(manifest, pinned)
            self.assertFalse(directory.exists())
            shutil.copytree(self.payload, directory)
            return expected

        with patch.object(release, 'repository', return_value=info), \
                patch.object(release, 'branch_head', return_value=pinned['revision']), \
                patch.object(release, 'inventory', return_value=pinned['files']), \
                patch.object(release, 'readback', side_effect=download) as readback, \
                patch.object(hub, 'update_repo_settings', create=True) as visibility:
            unrelated = downloaded / 'unrelated.txt'
            unrelated.write_text('separate file')
            with self.assertRaisesRegex(ValueError, 'payload inventory'):
                release.publish(hub, state_path, state, self.manifest, 'apache-2.0', downloaded)
            self.assertEqual(unrelated.read_text(), 'separate file')
            readback.assert_not_called()
            unrelated.unlink()

            receipt = release.publish(hub, state_path, state, self.manifest, 'apache-2.0', downloaded)
            self.assertEqual(receipt, expected)
            self.assertNotEqual(readback.call_args.args[1], downloaded)
            release.verify_tree(downloaded, pinned['files'])
            recorded = json.loads(state_path.read_bytes())
            self.assertEqual(recorded['public_readback'], expected)

            shutil.rmtree(downloaded)
            receipt = release.publish(hub, state_path, recorded, self.manifest, 'apache-2.0', downloaded)
            self.assertEqual(receipt, expected)
            self.assertEqual(readback.call_count, 2)
            readback.assert_called_with(pinned, downloaded)
            release.verify_tree(downloaded, pinned['files'])
            visibility.assert_not_called()
