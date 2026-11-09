import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from one_gpu_lm.common.checkpoints import commit_checkpoint, load_checkpoint
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import single_writer
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.training.collect import collect
from one_gpu_lm.training.config import read_training_data, resolve_path


class InfrastructureContracts(unittest.TestCase):
    def test_checkpoint_commits_and_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);identity={'data':'a','code':'b'}
            commit_checkpoint(run,3,identity,{'history':[1,2,3]},lambda p:(p/'weights.bin').write_bytes(b'weights'))
            _,state,_=load_checkpoint(run,identity);self.assertEqual(state['step'],3)
            with self.assertRaisesRegex(ValueError,'identity'):
                load_checkpoint(run,{'data':'changed','code':'b'})

    def test_interrupted_writer_keeps_last_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);identity={'recipe':1}
            commit_checkpoint(run,1,identity,{},lambda p:(p/'weights.bin').write_bytes(b'old'))
            pointer=(run/'latest.json').read_bytes()
            def crash(p):
                (p/'weights.bin').write_bytes(b'partial');raise RuntimeError('injected write failure')
            with self.assertRaises(RuntimeError):commit_checkpoint(run,2,identity,{},crash)
            self.assertEqual((run/'latest.json').read_bytes(),pointer)
            self.assertEqual(load_checkpoint(run,identity)[1]['step'],1)
            self.assertTrue(list((run/'private/checkpoints').glob('.pending-*')))

    def test_corrupted_state_and_escaping_pointer_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);identity={'recipe':1}
            commit_checkpoint(run,1,identity,{},lambda p:(p/'state.pt').write_bytes(b'valid'))
            root,_,_=load_checkpoint(run,identity);(root/'state.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):load_checkpoint(run,identity)
            write_json(run/'latest.json',{'schema_version':1,'checkpoint':'../elsewhere','step':1})
            with self.assertRaisesRegex(ValueError,'Unsafe'):load_checkpoint(run,identity)

    def test_symlink_checkpoint_and_metadata_override_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            with self.assertRaisesRegex(ValueError,'Reserved'):
                commit_checkpoint(run,1,{}, {'step':999},lambda p:None)
            with self.assertRaisesRegex(ValueError,'symbolic'):
                commit_checkpoint(run,1,{}, {},lambda p:(p/'escape').symlink_to('/etc/passwd'))
            self.assertFalse((run/'latest.json').exists())

    def test_duplicate_writer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lock'
            with single_writer(path):
                with self.assertRaisesRegex(RuntimeError,'Another'):
                    with single_writer(path):pass

    def test_weights_are_rehashed_not_only_receipt_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);profile={'model_id':'fixture','revision':'r1'}
            data=b'original';(root/'weight.safetensors').write_bytes(data)
            write_json(root/'download-receipt.json',profile)
            manifest={**profile,'files':[{'name':'weight.safetensors','size':len(data),'sha256':hashlib.sha256(data).hexdigest()}],'metadata_files':[]}
            write_json(root/'manifest.json',manifest)
            self.assertTrue(verify_model(root,root/'manifest.json',profile)['content_verified'])
            (root/'weight.safetensors').write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'content'):verify_model(root,root/'manifest.json',profile)

    def test_export_never_includes_private_files_or_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();(run/'private').mkdir()
            write_json(run/'result.json',{'status':'completed'})
            (run/'private/state.pt').write_bytes(b'private optimizer')
            (run/'secret.txt').write_text('not allowlisted')
            collect(run,root/'public')
            self.assertEqual({p.name for p in (root/'public').iterdir()},{'result.json','artifacts.json'})
            write_json(run/'identity.json',{'unexpected':'/home/private/data'})
            with self.assertRaisesRegex(ValueError,'private-looking'):collect(run,root/'blocked')
            self.assertFalse((root/'blocked').exists())

    def test_unresolved_environment_path_is_rejected(self):
        with self.assertRaises(ValueError):resolve_path('${UNSET_ONE_GPU_LM_TEST_PATH}',Path('.'))

    def test_evaluation_data_and_unreviewed_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source={'repository':'fixture/repo','revision':'v1','split':'train','license_sha256':'l','rights_review':{'status':'reviewed'}}
            task={'id':'t','repository':'fixture/repo','split':'train','admitted':True}
            row={'id':'record','messages':[{'role':'user','content':'task'}],'target':'answer',
                 'provenance':{'task_id':'t','repository':'fixture/repo','revision':'v1','split':'train','kind':'verified-reference-repair'}}
            write_json(root/'licenses.json',{'sources':[source]});write_json(root/'admission.json',{'complete':True,'task_spec_sha256':'t','tasks':[task]})
            def prepare(value):
                (root/'data.jsonl').write_text(json.dumps(value)+'\n')
                write_json(root/'receipt.json',{'records':1,'sha256':sha256(root/'data.jsonl'),
                    'license_record_sha256':sha256(root/'licenses.json'),'task_spec_sha256':'t'})
            paths={'data':root/'data.jsonl','data_receipt':root/'receipt.json','admission':root/'admission.json','licenses':root/'licenses.json'}
            prepare(row);self.assertEqual(len(read_training_data({'paths':paths})[0]),1)
            row['provenance']['split']='test';prepare(row)
            with self.assertRaisesRegex(ValueError,'Evaluation'):read_training_data({'paths':paths})
            row['provenance']['split']='train';row['provenance']['revision']='unknown';prepare(row)
            with self.assertRaisesRegex(ValueError,'Unregistered'):read_training_data({'paths':paths})

    def test_operational_cli_help_does_not_require_gpu(self):
        for command in ['doctor','train','status','collect','verify-resume','evaluate','serve']:
            result=subprocess.run([sys.executable,'-m','one_gpu_lm',command,'--help'],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,(command,result.stderr));self.assertIn('usage:',result.stdout)


if __name__=='__main__':unittest.main()
