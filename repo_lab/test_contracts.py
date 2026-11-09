import tempfile
import copy
from pathlib import Path
import unittest

from .agent import parse_action
from .grading import changed_protected, grade
from .tasks import apply_mutation, file_manifest, prepare
from .train import select_examples
from .report import paired


class RepositoryContracts(unittest.TestCase):
    def test_ambiguous_or_empty_actions_fail(self):
        for text in ["no command", "```bash\n```", "```bash\nls\n```\n```bash\npwd\n```"]:
            with self.assertRaises(ValueError):
                parse_action(text)
        self.assertEqual(parse_action("Inspect.\n```bash\nsed -n '1,4p' file.py\n```"), {"command": "sed -n '1,4p' file.py"})

    def test_stub_preserves_notice_docstring_and_neighbor(self):
        source = '# Copyright original authors\ndef f(x):\n    """API documentation."""\n    y = x + 1\n    return y\n\ndef g():\n    return 2\n'
        broken = apply_mutation(source, {"kind": "python_stub", "symbol": "f"})
        self.assertIn('# Copyright original authors', broken)
        self.assertIn('"""API documentation."""', broken)
        self.assertIn('def g():\n    return 2', broken)
        self.assertNotIn('y = x + 1', broken)

    def test_native_bash_tool_requires_exactly_one_action(self):
        native = '<tool_call>\n<function=bash>\n<parameter=command>\ncat source.py\n</parameter>\n</function>\n</tool_call>'
        self.assertEqual(parse_action(native), {'command':'cat source.py'})
        for text in [native+native, native+'\n```bash\nls\n```', native.replace('function=bash','function=unknown')]:
            with self.assertRaises(ValueError):
                parse_action(text)

    def test_nonunique_mutation_fails_closed(self):
        with self.assertRaises(ValueError):
            apply_mutation('a a', {"kind": "replace", "before": "a", "after": "b"})

    def test_deleted_test_and_changed_metadata_rejected(self):
        self.assertEqual(changed_protected({'test.py':'a', 'package.json':'b', 'src.py':'c'},
            {'package.json':'c', 'src.py':'d'}, ['src.py']), ['package.json', 'test.py'])

    def test_fresh_grader_never_copies_injected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'source'
            src.mkdir()
            (src / 'answer.py').write_text('answer = 1\n')
            (src / 'test.py').write_text('immutable test\n')
            task = {'repository': 'fixture/repo', 'mutations': [{'kind':'replace','path':'answer.py','before':'1','after':'0'}],
                    'editable_files':['answer.py'], 'regression_command':'test-command'}
            work = prepare(src, root/'work', task)
            before = file_manifest(work)
            (work/'answer.py').write_text('answer = 1\n')
            (work/'sitecustomize.py').write_text('injected')
            class Env:
                def execute(self, action):
                    self_seen = root/'grade'
                    assert not (self_seen/'sitecustomize.py').exists()
                    assert (self_seen/'test.py').read_text() == 'immutable test\n'
                    return {'returncode':0, 'output':'ok', 'reason':'ok', 'seconds':0.0}
            result = grade(task, work, src, root/'grade', before, lambda _:Env())
            self.assertTrue(result['passed'])
            self.assertEqual(result['modified_source_files'], ['answer.py'])

    def test_symlink_submission_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'link').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):
                file_manifest(path)

    def test_training_masks_prefix_and_rejects_evaluation_tasks(self):
        class Tokenizer:
            eos_token = '!'
            def apply_chat_template(self, messages, **kwargs):
                return [1,2,3]
            def encode(self, text, **kwargs):
                return [4,5]
        row = {'id':'a:turn-1','messages':[{'role':'user','content':'task'}],'target':'answer',
               'provenance':{'split':'train','task_id':'a','kind':'fixture'}}
        profile = {'training':{'max_sequence_length':5},'chat_template_kwargs':{}}
        selected, excluded = select_examples([row],Tokenizer(),profile,{'a'})
        self.assertEqual(selected[0]['labels'],[-100,-100,-100,4,5])
        self.assertEqual(excluded,[])
        row['provenance']['split']='test'
        with self.assertRaises(ValueError):
            select_examples([row],Tokenizer(),profile,{'a'})

    def test_paired_report_rejects_incomplete_or_wrong_variant(self):
        rows=[]
        for split in ['train','dev','test']:
            rows.append({'id':split,'split':split,'repository':split+'/repo','family':split,
                'passed':True,'seconds':1.0,'calls':1,'generated_tokens':3,'exit_status':'Submitted',
                'modified_source_files':['a.py'],'generations':[{'adapter':False}]})
        base={'complete':True,'identity':{'ids':['train','dev','test']},'results':rows}
        adapted=copy.deepcopy(base)
        for row in adapted['results']:
            row['generations'][0]['adapter']=True
        counts={'train':1,'dev':1,'test':1}
        self.assertEqual(paired(base,adapted,counts)['splits']['test']['adapted_passed'],1)
        adapted['results'][0]['generations'][0]['adapter']=False
        with self.assertRaisesRegex(ValueError,'wrong model variant'):
            paired(base,adapted,counts)
        adapted['results'].pop()
        with self.assertRaisesRegex(ValueError,'Missing'):
            paired(base,adapted,counts)


if __name__ == '__main__':
    unittest.main()
