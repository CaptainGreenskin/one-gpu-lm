import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import identity_digest
from one_gpu_lm.verified import recovery_sft as training
from one_gpu_lm.verified import recovery_sft_projection as projection


class RecoveryProjectionTests(unittest.TestCase):
    def test_collection_rejection_never_runs_or_enters_the_prompt(self):
        runtime = SimpleNamespace(sandbox=Mock())
        state = {'lib.py': 'unchanged'}
        observation = {'executed': False, 'returncode': -2, 'reason': 'correction_requires_regression'}
        step = {'call': 1, 'action_format_valid': True, 'command': 'cat lib.py',
                'source_before': state, 'source_after': state, 'observation': observation}
        history = [{'role': 'user', 'content': 'Correct the sum.'}]
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path/'tool-001.json').write_text(json.dumps(dict(observation, output='Collection policy hint')))
            with patch.object(projection, 'source_state', return_value=state):
                self.assertEqual(projection.advance(runtime, path, {}, {}, path, step, '```bash\ncat lib.py\n```',
                                                    state, history), (state, 'collection_rejection_removed'))
                self.assertEqual(len(history), 1)
                self.assertFalse(runtime.sandbox.execute.called)
                step['source_after'] = {'lib.py': 'changed'}
                with self.assertRaisesRegex(ValueError, 'unverified_collection_rejection'):
                    projection.advance(runtime, path, {}, {}, path, step, '```bash\ncat lib.py\n```', state, history)
                self.assertFalse(runtime.sandbox.execute.called)


class RecoveryMixtureTests(unittest.TestCase):
    def setUp(self):
        self.baseline = SimpleNamespace(
            raw=[{'id': 'retained', 'provenance': {'task_id': 'old'}}],
            encoded=[{'id': 'retained', 'episode_id': 'old', 'ids': [1, 2], 'labels': [-100, 2],
                      'supervised_tokens': 1, 'record_group_kind': 'whole_episode'}],
            baseline={'optimizer_updates': 1, 'source_manifest_sha256': 'sources'},
            recipe={'gradient_accumulation_records': 8}, review={'sources': {'owner/repo': {}}},
            profile={'training': {'max_sequence_length': 8192}})
        self.design = {'collection': {'audit_sha256': 'audit'}, 'actor_budget': {'max_generated_tokens_per_call': 512}}

    def example(self, call, task, kind):
        run = 'audited-trajectory'
        episode = identity_digest([projection.TARGET_KIND, task, run])
        ids = [1, call+1]
        return {'id': identity_digest([episode,call]), 'episode_id': episode,
                'messages': [{'role': 'user', 'content': 'Add two integers.'}], 'target': 'tool call',
                'encoding_receipt': {'tokens': len(ids), 'target_tokens': 1, 'encoding_sha256': identity_digest(ids)},
                'provenance': {'split': 'train', 'producer_kind': 'model', 'executed': True, 'task_id': task,
                               'repository': 'owner/repo', 'family': 'owner/repo', 'collection_audit_sha256': 'audit',
                               'trajectory_sha256': run, 'source_manifest_sha256': 'sources', 'target_transform': projection.TARGET_KIND,
                               'call': call, 'kind': kind}}

    @staticmethod
    def encode(row, *_):
        ids = [1, row['provenance']['call'] + 1]
        return {'ids': ids, 'labels': [-100, ids[-1]], 'supervised_tokens': 1}

    def test_keeps_a_complete_sequence_and_rejects_a_retained_task(self):
        rows = [self.example(i, 'new', kind) for i,kind in enumerate(('edit','regression_pass','submit'),1)]
        with patch.object(training,'encode_training_example',self.encode):
            self.assertEqual(len(training.encode_projected(rows,self.baseline,None,self.design)), 3)
            with self.assertRaisesRegex(ValueError, 'Partial or reordered'):
                training.encode_projected(rows[:-1],self.baseline,None,self.design)
            with self.assertRaisesRegex(ValueError, 'reviewed training provenance'):
                training.encode_projected([self.example(1,'old','edit')],self.baseline,None,self.design)

    def test_additions_keep_the_retained_order_and_update_count(self):
        row = self.example(1,'new','edit')
        encoded = dict(self.encode(row), id=row['id'], episode_id=row['episode_id'], record_group_kind=projection.TARGET_KIND)
        receipt = {'outcomes':[{'selected': True, 'records': 1}]}
        ordered, windows, summary = training.mixture(self.baseline,[row],[encoded],receipt)
        self.assertEqual([item['id'] for item in ordered], ['retained',row['id']])
        self.assertEqual([len(window) for window in windows], [2])
        self.assertEqual(summary['optimizer_updates'], 1)
        self.assertEqual(summary['new_supervised_tokens'], 1)

    def test_scans_current_private_identifiers_and_credential_shapes(self):
        training.scan([{'target': 'a protected sample'}], ['known-private-host'])
        for value in ('known-private-host', 'hf_'+'a'*36):
            with self.assertRaisesRegex(ValueError, 'private identifier'):
                training.scan([{'target': value}], ['known-private-host'])

    def test_counts_cases_from_the_cohort_header_and_checks_family_aliases(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            retained = root/'retained.json'
            sources = root/'sources.json'
            cohort = root/'cohort.json'
            retained.write_text(json.dumps({'sources':[{'repository':'owner/core','family':'owner/core'}]}))
            source = {'repository':'owner/other','family':'owner/other','split':'development'}
            baseline = SimpleNamespace(protocol={'sources_path':'retained.json'})

            def inventory():
                sources.write_text(json.dumps({'sources':[source]}))
                cohort.write_text(json.dumps({'model_generations':0,'final_test_opened':False,'cases':1,
                                              'families':{'owner/other':1},'sources_sha256':sha256(sources)}))
                return {'development':{'cohort_sha256':sha256(cohort),'tasks':1,'families':1}}

            training.independent_development(root, baseline, inventory(), cohort)
            source['related_families'] = ['owner/core:shared']
            with self.assertRaisesRegex(ValueError, 'repository families overlap'):
                training.independent_development(root, baseline, inventory(), cohort)
