import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


PATH = Path(__file__).resolve().parents[1] / 'experiments/047-expanded-action-forks/run.py'
SPEC = importlib.util.spec_from_file_location('expanded_forks', PATH)
FORKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FORKS)


class ExpandedForksTest(unittest.TestCase):
    def test_blocks_cover_every_physical_pair_in_order(self):
        design = FORKS.read(FORKS.DESIGN)
        blocks = list(FORKS.registered_blocks(design['block_sizes']))
        ordinals = [ordinal for _, _, numbers in blocks for ordinal in numbers]
        self.assertEqual(ordinals, list(range(1, 130)))
        self.assertEqual([len(numbers) for _, _, numbers in blocks], [22, 22, 22, 22, 22, 19])
        engine = FORKS.load_engine()
        physical = list(engine.schedule([{'ordinal': ordinal} for ordinal in ordinals]))
        names = [name for _, _, name in physical]
        self.assertEqual(len(set(names)), len(physical))
        self.assertEqual(names[:4], ['01-greedy', '01-sampled', '02-sampled', '02-greedy'])

    def test_refuses_digest_changes_to_a_registered_block(self):
        with TemporaryDirectory() as area:
            stage = Path(area)
            protocol = stage / 'protocol.json'
            publication = stage / 'publication.json'
            design = FORKS.read(FORKS.DESIGN)
            plans = list(FORKS.registered_blocks(design['block_sizes']))
            sums = {filename: 'a' * 64 for filename, _, _ in plans}
            FORKS.atomic_json(protocol, {'design_sha256': FORKS.sha256(FORKS.DESIGN), 'blocks': sums,
                                         'training_only': True, 'final_test_opened': False, 'optimizer_updates': 0})
            FORKS.atomic_json(publication, {'protocol_sha256': FORKS.sha256(protocol),
                                            'anonymous_files_verified': True, 'ci_conclusion': 'success'})
            block = stage / plans[0][0]
            block.parent.mkdir()
            FORKS.atomic_json(block, {'rows': []})
            with self.assertRaisesRegex(ValueError, 'outside the frozen campaign'):
                FORKS.load_block(stage, block, publication, None)

    def test_refuses_to_count_a_partial_block_as_an_audited_block(self):
        with TemporaryDirectory() as area:
            run = Path(area)
            protocol = 'b' * 64
            control = run / 'controller.json'
            FORKS.atomic_json(control, {'backend_stopped': True, 'protocol_sha256': protocol,
                                       'runs': [{'run_id': '01-greedy'}]})
            FORKS.atomic_json(run / 'audit.json',
                              {'passed': True, 'training_only': True, 'protocol_sha256': protocol,
                               'weights_public': False, 'final_test_opened': False, 'optimizer_updates': 0,
                               'controller_sha256': FORKS.sha256(control), 'audited_attempts': 1,
                               'expected_attempts': 2, 'pairs': [], 'training_export_sha256': None})
            with self.assertRaisesRegex(ValueError, 'complete assigned schedule'):
                FORKS.audited_block(run, protocol, [1])


if __name__ == '__main__':
    unittest.main()
