"""Require the audited schedule and remove per-pair fields in the publication."""

import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import identity_digest


REPORT = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                            'experiments/048-expanded-action-preference/report.py'))


class PreferenceTrainingReport(unittest.TestCase):
    def test_exact_audit_and_anonymous_projection(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, public = root / 'protocol.json', root / 'publication.json'
            out = root / 'runs/fixture'
            out.mkdir(parents=True)
            record = {'pair': 1, 'chosen_sha256': 'a' * 64, 'rejected_sha256': 'b' * 64}
            batches = [[record]]
            order = identity_digest(batches)
            write_json(protocol, {'candidate': REPORT['CANDIDATE'], 'run_directory': 'runs/fixture',
                                  'expected_optimizer_updates': 1, 'planned_pairs': 1,
                                  'order_sha256': order, 'expected_trainable_parameters': 12,
                                  'final_test_opened': False, 'weights_public': False})
            identity = sha256(protocol)
            write_json(public, {'protocol_sha256': identity, 'commit': '1' * 40,
                                'anonymous_files_verified': True, 'ci_conclusion': 'success'})
            result = dict.fromkeys(REPORT['TRAINING_CHECKS'], True)
            result.update(stage='completed', protocol_sha256=identity, attempt=1,
                          started_utc='2026-11-22T11:30:00+00:00', finished_utc='2026-11-22T11:30:10+00:00',
                          order_sha256=order, optimizer_steps=1, pairs_consumed=1, trainable_parameters=12,
                          development_scores=0, final_test_opened=False, weights_public=False,
                          telemetry={'samples': 0, 'errors': 0, 'thread_stopped': True})
            write_json(out / 'result.json', result)
            write_json(out / 'selection.json', {'batches': batches, 'order_sha256': order,
                                                'pairs': 1, 'updates': 1, 'final_test_opened': False})
            update = {'step': 1, 'grad_norm': .2, 'seconds': 1, 'peak_allocated_bytes': 100,
                      'peak_reserved_bytes': 120}
            example = dict(record, loss=.3, policy_margin=3., reference_margin=2.,
                           private_action='training-only example')
            write_json(out / 'curve.json', [dict(update, records=[example])])
            write_json(out / 'updates.json', [dict(update, pairs=1)])
            for name in ('adapter.json', 'environment.json'):
                write_json(out / name, {})
            audit = out / 'integrity.json'
            write_json(audit, {'passed': True, 'protocol_sha256': identity, 'optimizer_updates': 1,
                               'final_test_opened': False, 'weights_public': False,
                               'audit_optimizer_updates': 0, 'audit_model_generations': 0,
                               'receipt_sha256': {name: sha256(out / name) for name in REPORT['TRAINING_RECEIPTS']}})
            write_json(out / 'candidate-input-review.json',
                       {'experiment': REPORT['CANDIDATE'], 'training_integrity_sha256': sha256(audit)})
            public_result = REPORT['capture'](root, protocol, public)
            self.assertEqual(public_result['updates'][0]['mean_margin_gain'], 1.)
            self.assertEqual(set(public_result['updates'][0]), set(REPORT['UPDATE_FIELDS']))
            self.assertNotIn(example['private_action'], json.dumps(public_result))
            self.assertNotIn(record['chosen_sha256'], json.dumps(public_result))

            write_json(out / 'curve.json', [dict(update, records=[dict(example, loss=.7)])])
            with self.assertRaisesRegex(ValueError, 'complete preference training audit'):
                REPORT['capture'](root, protocol, public)


if __name__ == '__main__':
    unittest.main()
