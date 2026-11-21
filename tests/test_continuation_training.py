import copy
import json
from pathlib import Path
import tempfile
import unittest

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.verified.continuation_training import replay_epoch
from one_gpu_lm.verified.continuation_training_data import audited_continuations, matched_windows, record_shape


def records(prefix, count):
    return [{'id':f'{prefix}-{i}', 'episode_id':prefix, 'ids':[1,2], 'supervised_tokens':1,
             'record_group_kind':prefix} for i in range(count)]


class ContinuationTrainingTests(unittest.TestCase):
    def test_same_update_count_while_preserving_every_retained_action(self):
        retained = records('retained',1006)
        new = records('supplement',27)
        batches = matched_windows(retained,new,8)
        self.assertEqual(len(batches),126)
        flattened = [row for batch in batches for row in batch]
        self.assertEqual([row for row in flattened if row['episode_id']=='retained'],retained)
        self.assertEqual({row['id'] for row in flattened if row['episode_id']=='supplement'},
                         {row['id'] for row in new})
        self.assertEqual(sum(len(b)==9 for b in batches),27)
        self.assertEqual(len(flattened),1033)
        with self.assertRaisesRegex(ValueError,'duplicate actions'):
            matched_windows(retained,new+[retained[0]],8)

    def test_epoch_audit_detects_changed_windows_and_nonfinite_loss(self):
        batches = matched_windows(records('retained',16),records('supplement',1),8)
        flat = [row for batch in batches for row in batch]
        summary = {'supervised_tokens':17,'input_tokens':34}
        selection = dict(mixture=summary,records=[record_shape(row) for row in flat])
        state = dict(summary,optimizer_steps=2,action_records_consumed=17)
        recipe = {'learning_rate':1e-5}
        curve = []
        updates = []
        for step,batch in enumerate(batches,1):
            updates.append(dict(step=step,records=len(batch),grad_norm=1.,seconds=.8,**recipe))
            for row in batch:
                curve.append(dict(record_shape(row),record=len(curve)+1,update=step,loss=1.,seconds=.1))
        replay_epoch(batches,summary,recipe,state,selection,curve,updates)
        changed = copy.deepcopy(curve)
        changed[-1]['update'] = 1
        with self.assertRaisesRegex(ValueError,'optimizer windows'):
            replay_epoch(batches,summary,recipe,state,selection,changed,updates)
        changed = copy.deepcopy(curve)
        changed[0]['loss'] = float('nan')
        with self.assertRaisesRegex(ValueError,'optimizer windows'):
            replay_epoch(batches,summary,recipe,state,selection,changed,updates)

    def test_a_failed_collection_cannot_read_or_admit_new_examples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def save(name,value):
                path = root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value),encoding='utf-8')
                return path
            protocol = save('private/protocol.json',{'run_directory':'runs/fixture'})
            controller = save('runs/fixture/controller.json',{})
            audit = save('runs/fixture/audit.json',{'passed':True,'collection_gate_passed':False})
            save('private/first-edit-job/state.json',{'phase':'audited','protocol_sha256':sha256(protocol),
                                                     'audit_sha256':sha256(audit),'controller_sha256':sha256(controller),
                                                     'final_test_opened':False})
            with self.assertRaisesRegex(ValueError,'did not qualify'):
                audited_continuations(root,None,None)


if __name__=='__main__':
    unittest.main()
