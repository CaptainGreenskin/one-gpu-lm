import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.data.reference_episodes import build, repair_commands


class ReferenceEpisodeContracts(unittest.TestCase):
    def test_patch_repairs_only_declared_sources_and_check_is_a_separate_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);broken=root/'broken';good=root/'good';broken.mkdir();good.mkdir()
            (broken/'a.py').write_text('def f(x):\n    raise NotImplementedError()\n')
            (good/'a.py').write_text('def f(x):\n    return x + 1\n')
            task={'editable_files':['a.py'],'regression_command':'python3 -c "import a; assert a.f(2)==3"'}
            commands=repair_commands(task,broken,good)
            self.assertEqual([r[0] for r in commands],['inspect','reproduce','repair','review','verify','submit'])
            self.assertEqual(commands[1][2],1)
            result=subprocess.run(['/bin/sh','-c',commands[2][1]],cwd=broken,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((broken/'a.py').read_bytes(),(good/'a.py').read_bytes())
            self.assertEqual(commands[4][1],task['regression_command'])

    def test_reference_without_actual_changes_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'a.py').write_text('x=1\n')
            with self.assertRaises(ValueError):repair_commands({'editable_files':['a.py']},p,p)

    def test_eval_only_partition_cannot_create_training_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tasks=root/'tasks.json';admission=root/'admission.json';licenses=root/'licenses.json'
            tasks.write_text(json.dumps({'tasks':[{'split':'test','id':'heldout'}]}))
            admission.write_text(json.dumps({'complete':True,'task_spec_sha256':sha256(tasks)}))
            licenses.write_text(json.dumps({'sources':[]}))
            # The builder does not need the template dependency to reject heldout data.
            with self.assertRaisesRegex(ValueError,'Only training'):
                build(root,tasks,admission,licenses,root/'out')
            self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
