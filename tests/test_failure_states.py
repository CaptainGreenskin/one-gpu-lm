"""Historical error admission must confirm source failures without future hints."""
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

from tests import test_correction_collection as fixtures
from one_gpu_lm.verified import failure_states as fs
from one_gpu_lm.verified.submission import SUBMIT


class HistoricalFailureStates(unittest.TestCase):
    def setUp(self):
        self.h=fixtures.CorrectiveCollection();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.f=self.h.f

    def collect(self, actions=None):
        out,trace,_=self.h.collect('correction',actions or ['bad-edit','python3 check.py','edit-source','python3 check.py',SUBMIT])
        return {'origin':'027','branch':'correction','directory':out,'trace':trace}

    def replay(self,ref,item,runner=None):
        return fs.replay_to_failure(self.f.root,self.f.data,ref,item,self.f.task,self.f.source,
            self.f.root/'replay',runner or self.h.runner,self.f.normalizer,2400)

    def test_reconstructs_real_failure_without_future_solution_or_hidden_checks(self):
        ref=self.collect();item=fs.candidates(ref['trace'],self.f.task)[0]
        result=self.replay(ref,item);self.assertEqual(result['status'],'confirmed_failed_regression')
        root=self.f.root/'replay/snapshot';prefix=json.loads((root/'prefix.json').read_text())
        self.assertEqual((root/'tree/logic.py').read_text(),'answer = 5\n')
        self.assertEqual(len(prefix['messages']),6)
        text=json.dumps(prefix);self.assertNotIn('edit-source',text);self.assertNotIn('HIDDEN_CASE_SENTINEL',text)
        self.assertIn('ACTUAL_FAILURE',prefix['standalone_feedback'])

    def test_untouched_fault_timeout_rejection_and_development_are_excluded(self):
        ref=self.collect();trace=ref['trace']
        for mutation in ['original','timeout','rejected']:
            value=copy.deepcopy(trace)
            if mutation=='original':value['steps'][1]['source_after']=value['original_source_sha256']
            elif mutation=='timeout':value['steps'][1]['observation']['reason']='timeout'
            else:value['steps'][1]['observation']['executed']=False
            self.assertEqual(fs.candidates(value,self.f.task),[])
        value=copy.deepcopy(trace);value['split']='development'
        with self.assertRaises(ValueError):fs.candidates(value,self.f.task)

    def test_nonzero_compound_command_does_not_prove_bad_patch(self):
        old=self.h.runner.execute
        def execute(root,command):
            if command=='correct-patch-then-error':
                old(root,'edit-source')
                return {'returncode':1,'reason':'runtime_error','output':'OTHER_COMMAND_FAILED','seconds':.001}
            return old(root,command)
        self.h.runner.execute=execute
        ref=self.collect(['correct-patch-then-error','python3 check.py',SUBMIT])
        item=fs.candidates(ref['trace'],self.f.task)[0]
        self.assertEqual(self.replay(ref,item)['status'],'standalone_passed')
        self.assertFalse((self.f.root/'replay/snapshot').exists())

    def test_changed_feedback_rejects_replay_even_if_source_matches(self):
        ref=self.collect();item=fs.candidates(ref['trace'],self.f.task)[0];old=self.h.runner.execute
        def changed(root,command):
            value=old(root,command);value['output']+=' DIFF';return value
        with self.assertRaisesRegex(ValueError,'feedback differs'):
            self.replay(ref,item,SimpleNamespace(execute=changed,timeout=5))

    def test_changed_protected_file_cannot_enter_snapshot(self):
        ref=self.collect();item=fs.candidates(ref['trace'],self.f.task)[0];old=self.h.runner.execute
        def changed(root,command):
            value=old(root,command)
            if command=='bad-edit':(root/'check.py').write_text('tampered')
            return value
        with self.assertRaises(ValueError):self.replay(ref,item,SimpleNamespace(execute=changed,timeout=5))

    def test_later_success_is_not_a_selection_feature(self):
        ref=self.collect();a=fs.candidates(ref['trace'],self.f.task)
        other=copy.deepcopy(ref['trace']);other['grade']={'passed':False};other['eligible_verified_trajectory']=False
        self.assertEqual(a,fs.candidates(other,self.f.task))
        ref['trace']['steps'][2]['command']='DO_NOT_EXECUTE_FUTURE_ACTION'
        self.assertEqual(self.replay(ref,a[0])['status'],'confirmed_failed_regression')

    def test_deduplicates_state_without_losing_later_standalone_evidence(self):
        old=self.h.runner.execute
        def execute(root,command):
            value=old(root,command)
            if command=='bad-edit':value.update(returncode=1,reason='runtime_error',output='FAILED_IN_COMPOUND')
            return value
        self.h.runner.execute=execute;ref=self.collect();items=fs.candidates(ref['trace'],self.f.task)
        self.assertEqual(len(items),1);self.assertEqual(items[0]['call'],1)
        self.assertEqual(items[0]['recorded_standalone_failure_calls'],[2])

    def test_deterministic_unique_task_selection_and_cpu_bounds(self):
        rows=[{'task_id':name,'family':family,'status':'confirmed_failed_regression','state_id':str(i)}
              for i,(name,family) in enumerate([('a','x'),('a','x'),('b','y'),('c','x')])]
        chosen=fs.select(rows,2);self.assertEqual(len(chosen),2);self.assertEqual(len({r['family'] for r in chosen}),2)
        self.assertNotIn('1',[r['state_id'] for r in chosen])
        runner=fs.BoundedRunner(SimpleNamespace(timeout=0,execute=lambda *args:{'seconds':.1}),time.monotonic()+10,1)
        runner.execute(Path('.'),'fixture')
        with self.assertRaises(TimeoutError):runner.execute(Path('.'),'fixture')
        self.assertEqual(runner.commands,1)


if __name__=='__main__':unittest.main()
