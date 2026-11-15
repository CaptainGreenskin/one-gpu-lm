import unittest
from unittest.mock import patch
from one_gpu_lm.verified import runtime_identity as identity


class RuntimeIdentity(unittest.TestCase):
    def test_only_allowlisted_versions_recorded_and_drift_rejected(self):
        with patch.object(identity.platform,'python_version',return_value='3.12.fixture'),\
             patch.object(identity.metadata,'version',side_effect=lambda name:'1.0.'+name):
            expected=identity.capture()
            self.assertEqual(set(expected),{'python','packages'})
            self.assertEqual(set(expected['packages']),set(identity.PACKAGES))
            self.assertEqual(identity.verify(expected),expected)
            expected['packages']['transformers']='changed'
            with self.assertRaises(ValueError):identity.verify(expected)


if __name__=='__main__':unittest.main()
