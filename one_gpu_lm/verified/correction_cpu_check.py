"""Real Linux sandbox smoke check with scripted fixture actions, zero model calls."""
import argparse
import json
from pathlib import Path
import tempfile
from one_gpu_lm.common.receipts import write_json
from .correction_policy import CorrectionGuard
from .sandbox import Sandbox
from .submission import SUBMIT


def run(root,out):
    root=Path(root);runner=Sandbox(json.loads((root/'private/task-runtime.json').read_text()),5)
    if Path(out).exists():raise ValueError('Preserve previous CPU check')
    with tempfile.TemporaryDirectory(prefix='correction027-cpu-',dir=root/'private') as temp:
        workspace=Path(temp)/'actor';workspace.mkdir();(workspace/'logic.py').write_text('answer = 3\n')
        (workspace/'check.py').write_text('from logic import answer\nassert answer == 7\nprint("fixture passed")\n')
        guard=CorrectionGuard(runner,{str(workspace):{'editable_files':['logic.py'],'regression_command':'python3 check.py'}},'correction')
        bad=guard.execute(workspace,"python3 -c \"from pathlib import Path; Path('logic.py').write_text('answer = 5\\n')\"")
        rejected=guard.execute(workspace,"touch bypass")
        failure=guard.execute(workspace,'python3 check.py')
        good=guard.execute(workspace,"python3 -c \"from pathlib import Path; Path('logic.py').write_text('answer = 7\\n')\"")
        passed=guard.execute(workspace,'python3 check.py');submitted=guard.execute(workspace,SUBMIT)
        hidden=runner.execute(workspace,'test ! -e /home && test ! -e /root && test ! -e /work/hidden.py')
        checks={'bad_source_edit_executed':bad['returncode']==0,'non_regression_rejected_without_execution':rejected['executed'] is False and not (workspace/'bypass').exists(),
                'edited_source_test_failed':failure['executed'] is True and failure['returncode']!=0 and failure['reason']=='runtime_error',
                'corrective_edit_executed':good['returncode']==0,'standalone_regression_passed':passed['returncode']==0,
                'explicit_submission_accepted':submitted['returncode']==0 and submitted['executed'] is True,
                'host_and_hidden_files_absent':hidden['returncode']==0}
    result={'schema_version':1,'fixture_kind':'scripted CPU sandbox actions; not model data','passed':all(checks.values()),'checks':checks,
            'model_generations':0,'optimizer_updates':0,'final_test_opened':False}
    write_json(out,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.root,a.out)))
