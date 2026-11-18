import unittest
from one_gpu_lm.verified.historic_policy import tap_timings


class HistoricPolicyTests(unittest.TestCase):
    def test_direct_metadata_only(self):
        value = ('TAP version 13\nnot ok 1 - example\n  ---\n  duration_ms: 4.2\n'
                 '  actual:\n    duration_ms: 99\n  expected:\n    duration_ms: 100\n  ...\n'
                 '# tests 1\n# pass 0\n# fail 1\n# duration_ms 8.5\n')
        result, count = tap_timings(value)
        self.assertEqual(count, 2)
        self.assertIn('    duration_ms: 99', result)
        self.assertIn('    duration_ms: 100', result)
        self.assertNotIn('duration_ms: 4.2', result)

    def test_non_tap_and_arbitrary_source_preserved(self):
        value = '  duration_ms: 4.2\n# duration_ms 8.5\n'
        self.assertEqual(tap_timings(value), (value, 0))
        value = 'TAP version 13\n# duration_ms 8.5\n'
        self.assertEqual(tap_timings(value), (value, 0))


if __name__ == '__main__':
    unittest.main()
