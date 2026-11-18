from pathlib import Path
import subprocess,tempfile,unittest
from one_gpu_lm.verified.historic_plan import replacements,edit_command,plan


class ExpertPlanTests(unittest.TestCase):
    def test_actual_edits_preserve_literal_shell_syntax_and_context(self):
        parent="first = '$(do-not-execute)'\n"+'x = 1\n'*6+"last = '`literal`'\n"
        fixed=parent.replace('x = 1','x = 2').replace("'`literal`'","'end'")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'a.py').write_text(parent)
            result=subprocess.run(['bash','-c',edit_command('a.py',replacements(parent,fixed))],cwd=root,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual((root/'a.py').read_text(),fixed)

    def task(self):
        return {'id':'fixture','split':'training','editable_files':['a.py','b.py'],
                'mutations':[{'path':n,'after':'v = 0\n','before':'v = 1\n'} for n in ['a.py','b.py']],
                'regression_command':'python3 checks/regression.py'}

    def test_multifile_plan_reserves_verification_and_submission(self):
        result=plan(self.task(),{'admitted':True,'essential_files':['a.py','b.py'],'multifile':True},
                    lambda text:len(text)//4,{'max_model_calls':12,'max_generated_tokens_per_call':512,'max_total_generated_tokens':6144,'max_tool_output_chars':2400})
        self.assertEqual(result['first_edit_call'],3)
        self.assertEqual([c['kind'] for c in result['commands']][-2:],['final_verification','explicit_submission'])
        self.assertIn('partial_repair_regression',[c['kind'] for c in result['commands']])
        self.assertEqual(result['model_generations'],0)

    def test_over_budget_plan_and_development_targets_rejected_whole(self):
        admission={'admitted':True,'essential_files':['a.py','b.py'],'multifile':True}
        budget={'max_model_calls':5,'max_generated_tokens_per_call':512,'max_total_generated_tokens':6144,'max_tool_output_chars':2400}
        with self.assertRaises(ValueError):plan(self.task(),admission,lambda _:10,budget)
        task=self.task();task['split']='development';budget['max_model_calls']=12
        with self.assertRaises(ValueError):plan(task,admission,lambda _:10,budget)


if __name__=='__main__':unittest.main()
