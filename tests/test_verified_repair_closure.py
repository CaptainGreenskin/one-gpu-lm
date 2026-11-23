import importlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


EXPERIMENT = Path(__file__).resolve().parents[1] / 'experiments/050-verified-repair-closure'


class BoundedEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(EXPERIMENT))
        try:
            cls.replay = importlib.import_module('supplement')
        finally:
            sys.path.remove(str(EXPERIMENT))

    def test_large_replacement_keeps_literal_source_and_missing_trailing_newline(self):
        parent = 'old_header = 0\nkeep = 1\nold_body = 2'
        items = [f"source_{i} = '$(printf unsafe) `literal`'" for i in range(24)]
        fixed = 'new_header = 1\nkeep = 1\n' + '\n'.join(items)
        limit = 575
        edits = list(self.replay.bounded_edits('module.py', parent, fixed, lambda shell: len(shell) <= limit))
        self.assertGreater(len(edits), 1)
        with TemporaryDirectory() as directory:
            module = Path(directory) / 'module.py'
            module.write_bytes(parent.encode())
            for command in edits:
                self.assertLessEqual(len(command), limit)
                subprocess.run(['bash', '-c', command], cwd=directory, check=True, capture_output=True)
            self.assertEqual(module.read_bytes(), fixed.encode())

    def test_deletes_separate_windows_without_shifting_next_edit(self):
        parent = 'remove this\nkeep alpha\nchange this\nkeep beta\nremove tail\n'
        fixed = 'keep alpha\nnew value\nkeep beta\n'
        edits = list(self.replay.bounded_edits('values.txt', parent, fixed, lambda shell: len(shell) <= 600))
        with TemporaryDirectory() as directory:
            file = Path(directory) / 'values.txt'
            file.write_text(parent)
            for command in edits:
                subprocess.run(['bash', '-c', command], cwd=directory, check=True, capture_output=True)
            self.assertEqual(file.read_text(), fixed)

    def test_reports_oversized_line_before_executing_an_edit(self):
        with self.assertRaisesRegex(ValueError, 'reference_line_over_budget'):
            list(self.replay.bounded_edits('values.txt', 'old\n', 'x' * 600 + '\n',
                                           lambda shell: len(shell) <= 575))
