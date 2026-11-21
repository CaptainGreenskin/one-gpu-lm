import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1]/'experiments/040-editable-source-context/source_context.py'
SPEC = importlib.util.spec_from_file_location('editable_source_context_040',PATH)
CONTEXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTEXT)


class EditableSourceContextTests(unittest.TestCase):
    def setUp(self):
        self.task = {'description':'Fix counterLimit in the existing function.', 'editable_files':['lib/counter.py']}
        lines = ['unrelated method body']*80
        lines[60] = 'def counterLimit(): pass'
        self.files = {'lib/counter.py':'\n'.join(lines)}
        self.budget = {'window_lines':[64,32,16,8], 'max_opening_tokens':100, 'max_added_tokens':80}

    @staticmethod
    def encode(task):
        return task['description'].split()

    def test_shorter_window_fits_and_contains_the_visible_anchor(self):
        changed,original,new,width = CONTEXT.with_source(self.task,self.files,self.encode,self.budget)
        self.assertLess(width,64)
        self.assertGreater(width,0)
        self.assertLessEqual(len(new)-len(original),self.budget['max_added_tokens'])
        self.assertLessEqual(len(new),self.budget['max_opening_tokens'])
        self.assertIn('  61 | def counterLimit(): pass',changed['description'])
        self.assertTrue(changed['description'].startswith(self.task['description']))
        self.assertNotIn(' | ',self.task['description'])

    def test_fallback_preserves_original_opening_and_rejects_other_files(self):
        budget = dict(self.budget,max_added_tokens=1)
        changed,original,new,width = CONTEXT.with_source(self.task,self.files,self.encode,budget)
        self.assertEqual((changed,new,width),(self.task,original,0))
        with self.assertRaisesRegex(ValueError,'declared editable files'):
            CONTEXT.with_source(self.task,{'test/private.py':'example'},self.encode,self.budget)
