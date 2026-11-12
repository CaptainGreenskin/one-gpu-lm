import importlib.util
import math
from pathlib import Path
import unittest

p=Path(__file__).resolve().parents[1]/'experiments/021-observed-recovery/hardware.py'
spec=importlib.util.spec_from_file_location('observed_hardware',p)
hardware=importlib.util.module_from_spec(spec);spec.loader.exec_module(hardware)


class HardwareGapTests(unittest.TestCase):
    def rows(self):
        return [dict(utc='2026-11-12T08:'+t+'+00:00',attempt=a,power_w=360,
                     temperature_c=40,memory_used_mib=60000,utilization_percent=90)
                for t,a in [('00:00','1'),('00:05','1'),('10:05','2'),('10:10','2')]]

    def test_outage_is_not_charged_as_observed_energy(self):
        value=hardware.summarize(self.rows())
        self.assertEqual(value['covered_interval_seconds'],10)
        self.assertEqual(value['unobserved_interval_seconds'],600)
        self.assertAlmostEqual(value['sampled_gpu_energy_kwh'],.001)
        self.assertEqual(len(value['unobserved_intervals']),1)

    def test_plot_does_not_join_across_outage(self):
        x,y=hardware.plot_series(self.rows(),'power_w')
        self.assertTrue(math.isnan(x[2]) and math.isnan(y[2]))
        self.assertEqual(len(x),5)

    def test_backwards_sample_is_rejected(self):
        with self.assertRaises(ValueError):hardware.summarize(self.rows()[::-1])
