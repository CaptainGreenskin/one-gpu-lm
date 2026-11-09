import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from one_gpu_lm.common.source import campaign_sources


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {
    'preflight': 'evaluation.preflight', 'campaign': 'evaluation.campaign',
    'agent': 'evaluation.agent', 'regression': 'evaluation.regression',
    'report': 'evaluation.report', 'audit': 'evaluation.audit',
    'install_node_deps': 'evaluation.install_node_deps',
    'dataset': 'data.repository', 'train': 'training.repository_lora',
    'backend': 'serving.local_backend',
}


class ModuleContracts(unittest.TestCase):
    def test_legacy_entry_points_are_the_same_implementation(self):
        for legacy, canonical in ENTRIES.items():
            with self.subTest(module=legacy):
                self.assertIs(importlib.import_module('repo_lab.'+legacy).main,
                              importlib.import_module('one_gpu_lm.'+canonical).main)
        from repo_lab.backend import Runtime
        from one_gpu_lm.models.hf_runtime import Runtime as CanonicalRuntime
        from training_lab.sandbox import Sandbox
        from one_gpu_lm.evaluation.sandbox import Sandbox as CanonicalSandbox
        self.assertIs(Runtime, CanonicalRuntime)
        self.assertIs(Sandbox, CanonicalSandbox)

    def test_both_cli_names_work_without_gpu_dependencies(self):
        for legacy, canonical in ENTRIES.items():
            for module in ['repo_lab.'+legacy, 'one_gpu_lm.'+canonical]:
                with self.subTest(module=module):
                    result = subprocess.run([sys.executable, '-m', module, '--help'],
                        cwd=ROOT, capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)

    def test_imports_do_not_load_a_model_or_depend_on_legacy_packages(self):
        modules = ['one_gpu_lm.'+'.'.join(p.relative_to(ROOT/'one_gpu_lm').with_suffix('').parts)
                   for p in (ROOT/'one_gpu_lm').rglob('*.py') if p.name != '__init__.py']
        program = ('import importlib,sys,json;'
                   '[importlib.import_module(m) for m in '+repr(modules)+'];'
                   'print(json.dumps(sorted(set(sys.modules)&'
                   '{"torch","transformers","peft","minisweagent"})))')
        result = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])
        for file in (ROOT/'one_gpu_lm').rglob('*.py'):
            for node in ast.walk(ast.parse(file.read_text())):
                imports = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                           else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                self.assertFalse(any(n.split('.')[0] in {'repo_lab','training_lab'} for n in imports), str(file))

    def test_model_training_and_serving_boundaries(self):
        allowed = {'common':{'common'}, 'models':{'models','common'},
                   'training':{'training','models','common'},
                   'serving':{'serving','models','common'}}
        for layer, dependencies in allowed.items():
            for file in (ROOT/'one_gpu_lm'/layer).glob('*.py'):
                for node in ast.walk(ast.parse(file.read_text())):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('one_gpu_lm.'):
                        self.assertIn(node.module.split('.')[1], dependencies, str(file))

    def test_execution_identity_hashes_canonical_code_and_runtime_assets(self):
        hashes = campaign_sources()
        for name in ['models/hf_runtime.py','serving/client.py','evaluation/agent.py',
                     'evaluation/sandbox.py','common/actions.py','common/source.py',
                     'evaluation/underscore_tests.mjs','evaluation/ava.config.mjs',
                     '__init__.py','evaluation/__init__.py']:
            self.assertIn('one_gpu_lm/'+name, hashes)
        self.assertTrue(all((ROOT/name).is_file() and len(digest)==64 for name,digest in hashes.items()))


if __name__ == '__main__':
    unittest.main()
