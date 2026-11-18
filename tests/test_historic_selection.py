import unittest
from one_gpu_lm.verified.historic_selection import python_tests,regression_wrapper


class SelectionTests(unittest.TestCase):
    def test_changed_decorator_keeps_whole_parametrized_test(self):
        text='import pytest\n@pytest.mark.parametrize("x", [1,2])\ndef test_regression(x):\n    assert x > 0\n\ndef test_control():\n    assert True\n'
        selected=python_tests('tests/test_a.py',text,{2})
        self.assertEqual(selected['touched'],['tests/test_a.py::test_regression'])
        self.assertEqual(selected['controls'],['tests/test_a.py::test_control'])

    def test_unittest_methods_and_nested_helpers_not_inflated(self):
        text='class Regression:\n    def test_bad(self):\n        def test_helper():\n            return 1\n        assert test_helper() == 1\n    def test_other(self):\n        assert True\n'
        result=python_tests('t.py',text,{4},controls=0)
        self.assertEqual(result['selectors'],['t.py::Regression::test_bad'])

    def test_fixture_change_falls_back_without_guessing_test_dependencies(self):
        result=python_tests('t.py','X = [1]\ndef test_x():\n    assert X\n',{1})
        self.assertEqual(result['selectors'],['t.py'])
        self.assertIn('fallback',result['mode'])

    def test_wrapper_preserves_literal_arguments_and_expected_tests(self):
        text=regression_wrapper('javascript',['tests/a with spaces.js'],'./checks/globals.cjs')
        compile(text,'wrapper','exec')
        self.assertIn("'tests/a with spaces.js'",text)
        self.assertNotIn('shell=True',text)


if __name__=='__main__':unittest.main()
