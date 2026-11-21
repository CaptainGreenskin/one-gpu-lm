"""Check that the training report binds audited bytes and excludes private rows."""
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

from one_gpu_lm.common.receipts import sha256, write_json


REPORT = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                            'experiments/042-first-edit-sft/report.py'))
CURVE = {'record': 1, 'update': 1, 'tokens': 100, 'supervised_tokens': 8,
         'loss': 0.25, 'private_trace': 'actor content'}
UPDATE = {'step': 1, 'records': 1, 'grad_norm': 1.0, 'seconds': 1.5}


class ContinuationTrainingReport(unittest.TestCase):
    def test_incomplete_window_is_excluded(self):
        tail = dict(CURVE, record=2, update=2)
        projected, leftover = REPORT['update_metrics']([CURVE, tail], [UPDATE])
        self.assertEqual(leftover, 1)
        self.assertEqual(projected, [{'step': 1, 'records': 1, 'input_tokens': 100,
                                     'supervised_tokens': 8, 'mean_loss': 0.25,
                                     'grad_norm': 1.0, 'seconds': 1.5}])
        with self.assertRaisesRegex(ValueError, 'lacks its recorded examples'):
            REPORT['update_metrics']([], [UPDATE])

    def test_changed_curve_fails_terminal_audit_check(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / 'protocol.json'
            write_json(protocol, {'candidate': '042-first-edit-sft', 'run_directory': 'runs/fixture',
                                  'expected_optimizer_updates': 1, 'expected_records': 1,
                                  'final_test_opened': False, 'weights_public': False})
            identity = sha256(protocol)
            publication = root / 'publication.json'
            write_json(publication, {'protocol_sha256': identity, 'commit': '1' * 40,
                                     'ci_conclusion': 'success', 'anonymous_files_verified': True})
            out = root / 'runs/fixture'
            out.mkdir(parents=True)
            state = {'stage': 'completed', 'protocol_sha256': identity, 'development_scores': 0,
                     'started_epoch': 100, 'finished_epoch': 102, 'optimizer_steps': 1,
                     'action_records_consumed': 1, 'input_tokens': 100, 'supervised_tokens': 8,
                     'final_test_opened': False, 'weights_public': False}
            state.update({check: True for check in REPORT['TRAINING_CHECKS']})
            write_json(out / 'result.json', state)
            write_json(out / 'curve.json', [CURVE])
            write_json(out / 'updates.json', [UPDATE])
            for name in ('selection.json', 'adapter.json', 'environment.json'):
                write_json(out / name, {})
            audit = out / 'integrity.json'
            write_json(audit, {'passed': True, 'protocol_sha256': identity, 'optimizer_updates': 1,
                               'final_test_opened': False, 'weights_public': False,
                               'audit_model_generations': 0, 'audit_optimizer_updates': 0,
                               'receipt_sha256': {name: sha256(out / name) for name in REPORT['RECEIPTS']}})
            job = root / 'job.json'
            write_json(job, {'phase': 'audited', 'protocol_sha256': identity,
                             'final_test_opened': False, 'adapter_integrity_sha256': sha256(audit)})
            receipt = REPORT['capture'](root, protocol, job, publication)
            self.assertTrue(receipt['audit_passed'])
            self.assertNotIn(CURVE['private_trace'], json.dumps(receipt))
            write_json(out / 'curve.json', [dict(CURVE, loss=0.5)])
            with self.assertRaisesRegex(ValueError, 'exact completed training audit'):
                REPORT['capture'](root, protocol, job, publication)


if __name__ == '__main__':
    unittest.main()
