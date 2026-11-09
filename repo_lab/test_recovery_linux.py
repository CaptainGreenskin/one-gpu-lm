"""Exercise real controller process reconstruction with a deterministic non-model fixture."""
import json
from pathlib import Path
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest

from training_lab.records import write_json


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('bwrap'), 'Linux bwrap and pinned agent environment required')
class RecoveryContract(unittest.TestCase):
    def test_restart_preserves_history_budget_workspace_and_grade(self):
        with tempfile.TemporaryDirectory(prefix='repo-recovery-') as tmp:
            root = Path(tmp)
            snapshot = root/'sources/fixture--repo'
            snapshot.mkdir(parents=True)
            (snapshot/'answer.py').write_text('value = 1\n')
            (root/'.venv-agent').symlink_to(sys.prefix, target_is_directory=True)
            write_json(root/'runtime.json', {'node_root':'/usr'})
            write_json(root/'tasks.json', {'tasks':[{'id':'fixture-task','repository':'fixture/repo',
                'split':'train','family':'fixture','description':'Restore value to 1.',
                'editable_files':['answer.py'], 'regression_command':'python3 -c "import answer; assert answer.value == 1"',
                'mutations':[{'kind':'replace','path':'answer.py','before':'1','after':'0'}]}]})
            write_json(root/'protocol.json', {'agent':{'max_total_generated_tokens':6144,
                'max_generated_tokens_per_call':512, 'max_input_tokens':8192, 'max_tool_output_chars':2400,
                'max_tool_seconds':60, 'wall_time_limit_seconds':900, 'max_model_calls':12}})
            commands = ['cat answer.py', "printf 'value = 1\\n' > answer.py",
                        'python3 -c "import answer; assert answer.value == 1"', 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT']

            class Handler(socketserver.StreamRequestHandler):
                def handle(self):
                    value = json.loads(self.rfile.readline())
                    if value['operation'] == 'health':
                        answer = {'ready':True, 'model_id':'deterministic-controller-fixture', 'adapter':False}
                    else:
                        index = sum(m['role'] == 'assistant' for m in value['messages'])
                        answer = {'text':'```bash\n'+commands[index]+'\n```', 'ids':[1]*8,
                            'input_tokens':100, 'generated_tokens':8, 'seconds':0.01,
                            'hit_generation_limit':False,'peak_allocated_bytes':0,'adapter':False}
                    self.wfile.write(json.dumps(answer).encode()+b'\n')

            server = socketserver.UnixStreamServer(str(root/'backend.sock'), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                common = [sys.executable,'-m','repo_lab.agent','--root',str(root),'--tasks',str(root/'tasks.json'),
                    '--protocol',str(root/'protocol.json'),'--socket',str(root/'backend.sock'),'--id','fixture-task']
                def launch(label, *extra):
                    return subprocess.run(common+['--out',str(root/label),*extra], capture_output=True, text=True, timeout=30)
                first = launch('continuous')
                self.assertEqual(first.returncode, 0, first.stderr[-3000:])
                interrupted = launch('resumed', '--interrupt-after','2')
                self.assertEqual(interrupted.returncode, 75, interrupted.stderr[-3000:])
                frontier = json.loads((root/'resumed/checkpoint.json').read_text())
                self.assertEqual(frontier['n_calls'], 2)
                self.assertEqual(frontier['boundary'], 'completed_step')
                second = launch('resumed', '--resume')
                self.assertEqual(second.returncode, 0, second.stderr[-3000:])
                a, b = [json.loads((root/name/'result.json').read_text()) for name in ['continuous','resumed']]
                for key in ['passed','calls','generated_tokens','tool_calls','source_sha256','exit_status']:
                    self.assertEqual(a[key], b[key], key)
                self.assertTrue(a['passed'])
                self.assertEqual(b['calls'], 4)
                self.assertTrue(b['recovered'])
                histories = [json.loads((root/name/'trajectory.json').read_text())['messages'] for name in ['continuous','resumed']]
                self.assertEqual(*[[{k:m[k] for k in ['role','content']} for m in history] for history in histories])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
