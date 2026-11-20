import io
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from one_gpu_lm.models import modelscope_readback


class ModelScopeReadbackTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root/'downloaded'
        self.payload = {'adapter_model.safetensors':b'weights', 'README.md':b'model card',
                        'training-source-notices/fixture/LICENSE':b'license'}
        self.files = {n:{'bytes':len(b),'sha256':sha256(b).hexdigest()} for n,b in self.payload.items()}
        self.manifest = {'endpoint':'https://www.modelscope.ai','repo_id':'owner/model',
                         'revision':'a'*40,'files':self.files}
        self.requests = []
        self.info = {'Path':'owner','Name':'model','Visibility':5}
        self.redirect_site = None
        self.tree = [{'Type':'tree','Path':'training-source-notices'}]+[
            {'Type':'blob','Mode':'33188','Path':name,'Size':entry['bytes'],'Sha256':entry['sha256']}
            for name,entry in self.files.items()]

    def open(self, request, timeout):
        split = urlsplit(request.full_url)
        self.assertEqual(split.scheme,'https')
        self.assertEqual(split.netloc,'www.modelscope.ai')
        self.assertIsNone(request.get_header('Authorization'))
        self.assertIsNone(request.get_header('Cookie'))
        self.requests.append((split.path,split.query))
        if split.path.endswith('/repo/files'):
            self.assertEqual(parse_qs(split.query),{'Revision':['a'*40],'Recursive':['True']})
            data = {'Files':self.tree}
        elif split.path.endswith('/repo'):
            args = parse_qs(split.query)
            self.assertEqual(args['Revision'],['a'*40])
            return io.BytesIO(self.payload[args['FilePath'][0]])
        else:
            self.assertEqual(split.path,'/api/v1/models/owner/model')
            data = self.info
        response = io.BytesIO(json.dumps({'Success':True,'Data':data}).encode())
        response.geturl = lambda: request.full_url if not self.redirect_site else self.redirect_site+split.path
        return response

    def readback(self):
        with patch.object(modelscope_readback,'build_opener',return_value=self):
            return modelscope_readback.readback(self.manifest,self.out)

    def test_exact_public_commit_and_file_bytes(self):
        result = self.readback()
        self.assertTrue(result['anonymous_public_readback_verified'])
        self.assertEqual(result['verified_files'],3)
        self.assertEqual(result['revision'],'a'*40)
        self.assertEqual(result['model_url'],'https://www.modelscope.ai/models/owner/model')
        for name,value in self.payload.items():
            self.assertEqual((self.out/name).read_bytes(),value)
        with self.assertRaises(FileExistsError):
            self.readback()

    def test_unauthorized_visibility_or_extra_file_stops_before_download(self):
        self.info['Visibility'] = 1
        with self.assertRaisesRegex(ValueError,'intended public repository'):
            self.readback()
        self.assertEqual(len(self.requests),1)
        self.requests.clear()
        self.info['Visibility'] = 5
        self.tree.append(dict(self.tree[1],Path='unexpected.txt'))
        with self.assertRaisesRegex(ValueError,'differs from the release manifest'):
            self.readback()
        self.assertEqual(len(self.requests),2)
        self.assertFalse(self.out.exists())

    def test_payload_mismatch_leaves_no_completed_directory(self):
        self.payload['adapter_model.safetensors'] = b'altered'
        with self.assertRaisesRegex(ValueError,'Public file differs'):
            self.readback()
        self.assertEqual(list(self.root.iterdir()),[])

    def test_will_not_validate_metadata_from_another_modelscope_site(self):
        self.redirect_site = 'https://modelscope.cn'
        with self.assertRaisesRegex(ValueError,'different site'):
            self.readback()
        self.assertEqual(len(self.requests),1)
        self.assertFalse(self.out.exists())

    def test_unpinned_revision_and_outside_path_never_request(self):
        self.manifest['revision'] = 'master'
        with self.assertRaisesRegex(ValueError,'full ModelScope Git commit'):
            self.readback()
        self.manifest['revision'] = 'a'*40
        self.files['../escaped'] = self.files['README.md']
        with self.assertRaisesRegex(ValueError,'canonical path'):
            self.readback()
        self.assertEqual(self.requests,[])
