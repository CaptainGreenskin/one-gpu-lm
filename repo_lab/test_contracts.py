import tempfile
from pathlib import Path
import unittest

from .agent import parse_action
from .grading import changed_protected, grade
from .tasks import apply_mutation, file_manifest, prepare


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


if __name__ == '__main__':
    unittest.main()
