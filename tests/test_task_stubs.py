import ast
import unittest

from one_gpu_lm.evaluation.tasks import apply_mutation


class StubMutationContracts(unittest.TestCase):
    def test_preserves_overload_declarations_and_stubs_concrete_body(self):
        source = '''from typing import overload
import typing as t
@overload
def example(value: int) -> int: ...
@t.overload
def example(
    value: str,
) -> str: ...
def example(value):
    """Public contract."""
    return value
'''
        value = apply_mutation(source, {'kind':'python_stub','symbol':'example'})
        ast.parse(value)
        self.assertEqual(source.split('def example(value):')[0], value.split('def example(value):')[0])
        self.assertIn('"""Public contract."""', value)
        self.assertNotIn('return value', value)
        self.assertIn('raise NotImplementedError', value)

    def test_no_concrete_implementation_or_inline_body_fails_before_admission(self):
        for source in ['from typing import overload\n@overload\ndef example(value: int) -> int: ...\n',
                       'def example(value): return value\n']:
            with self.assertRaises(ValueError):
                apply_mutation(source, {'kind':'python_stub','symbol':'example'})


if __name__ == '__main__':
    unittest.main()
