import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from one_gpu_lm.data.repair_curriculum import Episode, check_train_inputs, replacement_command, wrong_sources


class RepairCurriculumContracts(unittest.TestCase):
    def test_actual_wrong_edit_fails_and_corrective_patch_restores_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broken = {'a.py': 'def f(x):\n    raise NotImplementedError("implementation missing")\n'}
            task = {'id': 'synthetic', 'split': 'train', 'editable_files': ['a.py']}
            plan = {'task_id': 'synthetic', 'split': 'train', 'mutations': [{'path': 'a.py', 'kind': 'replace',
                'before': 'raise NotImplementedError("implementation missing")', 'after': 'return None'}]}
            wrong = wrong_sources(task, broken, plan)
            good = {'a.py': 'def f(x):\n    return x + 1\n'}
            (root/'a.py').write_text(broken['a.py'])
            for state, expected_exit in [(wrong, 1), (good, 0)]:
                current = {'a.py': (root/'a.py').read_text()}
                command = replacement_command(current, state)
                result = subprocess.run(['/bin/sh', '-c', command], cwd=root, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run([sys.executable, '-B', '-c', 'import a; assert a.f(2) == 3'], cwd=root, capture_output=True)
                self.assertEqual(result.returncode, expected_exit)
            self.assertEqual((root/'a.py').read_text(), good['a.py'])

    def test_partial_correction_leaves_already_correct_files_untouched(self):
        before = {'a.py': 'a=1\n', 'b.py': 'b=None\n'}
        after = {'a.py': 'a=1\n', 'b.py': 'b=2\n'}
        command = replacement_command(before, after)
        self.assertNotIn("('a.py',", command)
        self.assertIn("('b.py',", command)
        with self.assertRaises(ValueError):
            replacement_command(after, after)
        with self.assertRaises(ValueError):
            replacement_command({'../x.py': 'x=1\n'}, {'../x.py': 'x=2\n'})

    def test_ampersands_are_literal_and_failed_anchor_does_not_modify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); before = {'a.js': 'return a && b;\n'}; after = {'a.js': 'return a && b && c;\n'}
            (root/'a.js').write_text(before['a.js'])
            bad = subprocess.run(['/bin/sh', '-c', replacement_command(before, after, fail_anchor=True)], cwd=root, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertEqual((root/'a.js').read_text(), before['a.js'])
            good = subprocess.run(['/bin/sh', '-c', replacement_command(before, after)], cwd=root, capture_output=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual((root/'a.js').read_text(), after['a.js'])

    def test_development_plans_and_unrelated_file_edits_are_rejected(self):
        task = {'id': 'train', 'split': 'train', 'editable_files': ['a.py']}
        plan = {'task_id': 'train', 'split': 'train', 'mutations': [{'path': 'a.py', 'kind': 'replace', 'before': 'x=1', 'after': 'x=2'}]}
        bad = copy.deepcopy(plan); bad['split'] = 'dev'
        with self.assertRaises(ValueError): wrong_sources(task, {'a.py': 'x=1\n'}, bad)
        bad = copy.deepcopy(plan); bad['mutations'][0]['path'] = 'tests.py'
        with self.assertRaises(ValueError): wrong_sources(task, {'a.py': 'x=1\n'}, bad)
        admission = {'complete': True, 'task_spec_sha256': 'sha', 'tasks': [{'id': 'train', 'split': 'train', 'admitted': True}]}
        task.update(repository='synthetic/repo', revision='synthetic')
        ledger = {'sources': [{'repository': 'synthetic/repo', 'revision': 'synthetic', 'split': 'train', 'license_sha256': 'sha', 'rights_review': {'status': 'reviewed'}}]}
        with self.assertRaisesRegex(ValueError, 'training only'):
            check_train_inputs([task], admission, ledger, {'parent_task_spec_sha256': 'sha', 'tasks': [dict(plan, split='test')]}, 'sha')

    def test_incorrect_actions_cannot_be_supervised_or_executed_by_mistake(self):
        episode = object.__new__(Episode); episode.steps = []
        # No environment is attached: these must be rejected before execution.
        with self.assertRaisesRegex(ValueError, 'may be supervised'):
            episode.action('incorrect_attempt', 'touch forbidden', 0, slot='late_repair')
        with self.assertRaisesRegex(ValueError, 'may be supervised'):
            episode.action('reproduce', 'false', 1, slot='repair')


if __name__ == '__main__':
    unittest.main()
