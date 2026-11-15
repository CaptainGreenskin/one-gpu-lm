import unittest
from one_gpu_lm.verified.capacity import validate_protocol


class CapacityScope(unittest.TestCase):
    def test_probe_cannot_turn_into_training_or_shorter_sequence(self):
        p={'purpose':'synthetic_backward_capacity_only','optimizer_steps':0,'sequence_tokens':8192,
           'batch_size':1,'supervised_suffix_tokens':512,'maximum_seconds':1200,
           'minimum_reserved_headroom_bytes':8*1024**3,
           'run_directory':'runs/capacity-001','profile_path':'configs/capacity.json'}
        validate_protocol(p)
        for key, value in [('optimizer_steps',1),('sequence_tokens',4096),('maximum_seconds',3600),
                           ('minimum_reserved_headroom_bytes',0),('run_directory','../escape')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol({**p,key:value})


if __name__ == '__main__':
    unittest.main()
