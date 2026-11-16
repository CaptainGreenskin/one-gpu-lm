import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from one_gpu_lm.verified.recovery_state import capture, inventory, restore, remaining_budget, validate_prefix


class RecoveryStateTests(unittest.TestCase):
    def prefix(self):
        messages=[{'role':'system','content':'policy'},{'role':'user','content':'visible task'}]
        steps=[]
        for call in range(1,4):
            messages += [{'role':'assistant','content':'read action'},{'role':'user','content':'actual feedback'}]
            steps.append({'call':call,'source_before':{'a.py':'hash'},'source_after':{'a.py':'hash'},'generation':{'generated_tokens':20}})
        return {'attempted_calls':3,'steps':steps,'generated_tokens':60,'messages':messages,
                'actor_seconds':45.,'boundary':'completed_step','terminal_reason':None,'submitted':False}

    def test_two_restores_identical_then_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);a=p/'actor';a.mkdir();(a/'a.py').write_text('original');(a/'empty').mkdir()
            (a/'a.py').chmod(0o640);os.utime(a/'a.py',ns=(1000000000,1000000000))
            before=inventory(a);receipt=capture(a,p/'snapshot',self.prefix())
            left=restore(p/'snapshot',p/'student',receipt);right=restore(p/'snapshot',p/'teacher',receipt)
            self.assertEqual(left,right);self.assertEqual(inventory(p/'student'),before);self.assertEqual(inventory(p/'teacher'),before)
            (p/'student/a.py').write_text('changed')
            self.assertEqual((p/'teacher/a.py').read_text(),'original');self.assertEqual(inventory(p/'snapshot/tree'),before)
            with self.assertRaises(ValueError):restore(p/'snapshot',p/'teacher',receipt)

    def test_mutated_snapshot_or_history_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);a=p/'actor';a.mkdir();(a/'a.py').write_text('original');receipt=capture(a,p/'snapshot',self.prefix())
            (p/'snapshot/tree/a.py').write_text('altered')
            with self.assertRaises(ValueError):restore(p/'snapshot',p/'teacher',receipt)
            (p/'snapshot/prefix.json').write_text('{}')
            with self.assertRaises(ValueError):restore(p/'snapshot',p/'student',receipt)

    def test_symlink_and_hardlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'a').write_text('x');(p/'link').symlink_to(p/'a')
            with self.assertRaises(ValueError):inventory(p)
            (p/'link').unlink();os.link(p/'a',p/'hard')
            with self.assertRaises(ValueError):inventory(p)

    def test_completed_boundaries_and_actual_accounting_required(self):
        for change in [{'boundary':'in_flight'},{'unknown_inflight_generation_cost':True},{'generated_tokens':59},
                       {'submitted':True},{'actor_seconds':600.},{'actor_seconds':float('nan')}]:
            prefix=self.prefix();prefix.update(change)
            with self.assertRaises(ValueError):validate_prefix(prefix)
        prefix=self.prefix();prefix['steps'][1]['source_after']={'a.py':'different'}
        with self.assertRaises(ValueError):validate_prefix(prefix)

    def test_remaining_budget_deducts_shared_prefix(self):
        b={'max_model_calls':12,'max_total_generated_tokens':6144,'max_task_seconds':600,'max_generated_tokens_per_call':512}
        self.assertEqual(remaining_budget(b,self.prefix()),{'calls':9,'generated_tokens':6084,'actor_seconds':555.,'next_generation_cap':512})
        exhausted=copy.deepcopy(self.prefix());exhausted.update(attempted_calls=12,generated_tokens=6144,actor_seconds=600.5)
        self.assertEqual(remaining_budget(b,exhausted),{'calls':0,'generated_tokens':0,'actor_seconds':0.,'next_generation_cap':0})


if __name__=='__main__':unittest.main()
