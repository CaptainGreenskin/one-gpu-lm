import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

from one_gpu_lm.common.agent_policy import NATIVE_SYSTEM
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.data.repair_curriculum import Episode, locate_command, replacement_command, wrong_sources
from one_gpu_lm.evaluation.grading import grade
from one_gpu_lm.evaluation.tasks import file_manifest, prepare

spec = importlib.util.spec_from_file_location('curriculum_preparation', Path(__file__).resolve().parents[1]/'experiments/020-repair-curriculum/preparation.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class LocalTestEnvironment:
    """Real local CPU commands in disposable test fixtures; not experiment evidence."""
    def __init__(self, workspace): self.workspace = workspace
    def execute(self, action):
        start = time.monotonic()
        r = subprocess.run(['/bin/sh','-c',action['command']], cwd=self.workspace, capture_output=True, text=True, timeout=10)
        return {'returncode':r.returncode,'output':r.stdout+r.stderr,'seconds':time.monotonic()-start,
                'reason':'ok' if r.returncode == 0 else 'runtime_error'}


class CurriculumPreparationContracts(unittest.TestCase):
    def test_recovery_audit_requires_actual_failed_and_successful_source_states(self):
        for mode in ['failed_anchor','incorrect_edit']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); snapshot = root/'source'; snapshot.mkdir()
                (snapshot/'a.py').write_text('def f(x):\n    return x + 1\n')
                task = {'id':'synthetic','split':'train','repository':'synthetic/repo','revision':'test',
                    'editable_files':['a.py'],'mutations':[{'kind':'python_stub','path':'a.py','symbol':'f'}],
                    'regression_command': "python3 -B -c 'import a; assert a.f(2)==3'"}
                private = root/'recovery'; private.mkdir(); workspace = prepare(snapshot,private/'workspace',task)
                broken = {'a.py':(workspace/'a.py').read_text()}; reference = {'a.py':(snapshot/'a.py').read_text()}
                control = {'broken_sha256':{'a.py':sha256(workspace/'a.py')},'reference_sha256':{'a.py':sha256(snapshot/'a.py')}}
                plan = {'task_id':'synthetic','split':'train','mode':mode,'mutations':[{'kind':'replace','path':'a.py',
                    'before':'raise NotImplementedError("implementation missing")','after':'return None'}]}
                episode = object.__new__(Episode)
                episode.task,episode.workspace,episode.snapshot,episode.private = task,workspace,snapshot,private
                episode.original = file_manifest(workspace); episode.make_env = LocalTestEnvironment
                episode.env = LocalTestEnvironment(workspace)
                episode.progress = EditableProgress(workspace,episode.original,task['editable_files'])
                episode.messages = [{'role':'system','content':NATIVE_SYSTEM},{'role':'user','content':'Repair f.'}]
                episode.steps,episode.rows = [],[]
                episode.action('inspect',locate_command(task),0)
                episode.action('reproduce',task['regression_command'],1)
                wrong = wrong_sources(task,broken,plan); failed = mode == 'failed_anchor'
                episode.action('incorrect_attempt',replacement_command(broken,wrong,fail_anchor=failed),1 if failed else 0)
                episode.action('review_incorrect_attempt',"sed -n '1,5p' a.py",0)
                episode.action('failing_test_after_edit',task['regression_command'],1)
                wrong_grade = grade(task,workspace,snapshot,private/'wrong-grade',episode.original,LocalTestEnvironment)
                wrong_grade['recovery_mode'] = mode
                episode.action('correct_repair',replacement_command(episode.sources(),reference),0,slot='late_repair')
                episode.action('verify',task['regression_command'],0)
                episode.action('submit','echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT',0)
                rows,_ = episode.finish(wrong_grade=wrong_grade)
                trace = json.loads((private/'trajectory.json').read_text())
                actual = module.verify_trace(task,control,(broken,reference),plan,trace,'recovery')
                self.assertEqual(actual['late_repair'][0],rows[0]['target'])
                self.assertEqual(actual['late_repair'][2]['reminder'],failed)
                changed = copy.deepcopy(trace); changed['wrong_grade']['passed'] = True
                with self.assertRaisesRegex(ValueError,'failed-attempt control'):
                    module.verify_trace(task,control,(broken,reference),plan,changed,'recovery')
                changed = copy.deepcopy(trace); changed['steps'][2]['source_sha256_after'] = control['reference_sha256']
                with self.assertRaisesRegex(ValueError,'source transition'):
                    module.verify_trace(task,control,(broken,reference),plan,changed,'recovery')


if __name__ == '__main__': unittest.main()
