import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from one_gpu_lm.verified.historic_expert import execute_episode
from one_gpu_lm.verified.historic_inputs import audit_episode
from one_gpu_lm.verified.tasks import files


def prefix(tokenizer,messages,kwargs,form):
    return list(json.dumps(messages,sort_keys=True).encode()), ''


def encoded(row,tokenizer,profile):
    start=prefix(None,row['messages'],{},'')[0];suffix=list(row['target'].encode())
    return {'ids':start+suffix,'labels':[-100]*len(start)+suffix,
            'supervised_tokens':len(suffix),'prefix_tokens':len(start)}


class FixtureShell:
    timeout=60
    def execute(self,workspace,command):
        p=subprocess.run(['bash','-c',command],cwd=workspace,capture_output=True,text=True,
                         env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=10)
        return {'returncode':p.returncode,'reason':'ok' if p.returncode==0 else 'runtime_error',
                'output':p.stdout+p.stderr,'seconds':0.01}


class ExpertExecutionTests(unittest.TestCase):
    def test_real_partial_failure_full_repair_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';identity='owner--repo--'+'b'*40
            snapshot=data/'snapshots'/identity;snapshot.mkdir(parents=True)
            for name,value in {'a.py':'value = 1\n','b.py':'value = 2\n','LICENSE':'fixture notice\n',
                'test.py':"import a,b\nimport sys\nok=(a.value,b.value)==(1,2)\nprint('1 passed' if ok else '1 failed')\nsys.exit(0 if ok else 1)\n"}.items():
                (snapshot/name).write_text(value)
            source={'source_id':identity,'repository':'owner/repo','family':'owner/repo',
                'revision':'b'*40,'upstream_parent':'a'*40,'split':'training','approved_for':{'training':True},
                'notice_review_complete':True,'lineage_review_complete':True,'notices':['LICENSE'],'files':files(snapshot)}
            (data/'sources.json').write_text(json.dumps({'sources':[source]}))
            task={**{k:source[k] for k in ['source_id','repository','family','revision','upstream_parent','split']},
                'id':'fixture','origin':'upstream_history_parent_plus_test_patch','description':'repair both values',
                'editable_files':['a.py','b.py'],'mutations':[{'path':n,'after':'value = 0\n','before':(snapshot/n).read_text()} for n in ['a.py','b.py']],
                'visible_checks':{},'hidden_checks':{},'regression_command':'python3 test.py','grading_command':'python3 test.py'}
            admitted={'admitted':True,'multifile':True,'essential_files':['a.py','b.py']}
            profile={'chat_template_kwargs':{},'training':{'max_sequence_length':8192}}
            protocol={'episode_namespace':'fixture','agent_policy':{'format':'native_roundtrip','context':'bounded_history_v1'},
                'budget':{'max_model_calls':12,'max_generated_tokens_per_call':512,'max_total_generated_tokens':6144,
                          'max_input_tokens':8192,'max_tool_seconds':60,'max_task_seconds':600,'max_tool_output_chars':2400}}
            repository=Path(__file__).resolve().parents[1];directory=root/'episode'
            with patch('one_gpu_lm.verified.historic_expert.encode_prefix',prefix), patch('one_gpu_lm.verified.historic_expert.encode_training_example',encoded), \
                 patch('one_gpu_lm.verified.historic_inputs.encode_prefix',prefix), patch('one_gpu_lm.verified.historic_inputs.encode_training_example',encoded):
                records,episode=execute_episode(repository,data,task,source,admitted,FixtureShell(),None,profile,protocol,directory)
                self.assertEqual(len(records),7);self.assertEqual(episode['first_edit_call'],3)
                self.assertTrue(episode['verified_and_submitted'])
                result=audit_episode(repository,data,task,admitted,episode,records,protocol,None,profile,directory)
                self.assertEqual(len(result),7)
                records[-1]['target']='pretend submit'
                with self.assertRaises(ValueError):
                    audit_episode(repository,data,task,admitted,episode,records,protocol,None,profile,directory)


if __name__=='__main__':unittest.main()
