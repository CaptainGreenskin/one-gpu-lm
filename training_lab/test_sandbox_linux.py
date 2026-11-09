import shutil
import sys
import tempfile
from pathlib import Path
import unittest

from .sandbox import Sandbox


@unittest.skipUnless(sys.platform == "linux" and shutil.which("bwrap"), "Linux bwrap required")
class SandboxChecks(unittest.TestCase):
    def setUp(self):
        self.sandbox = Sandbox()

    def task(self, expected):
        return {"language": "python", "cases": [{"input": None, "expected": expected}]}

    def test_wrong_code_fails(self):
        self.assertFalse(self.sandbox.score(self.task(1), "def solve(x): return 0")["passed"])

    def test_boolean_does_not_equal_integer(self):
        self.assertFalse(self.sandbox.score(self.task(1), "def solve(x): return True")["passed"])

    def test_host_home_not_mounted(self):
        self.assertTrue(self.sandbox.score(self.task(False), "import os\ndef solve(x): return os.path.exists('/home')")["passed"])

    def test_isolated_network_has_only_loopback(self):
        code = "import socket\ndef solve(x): return sorted(name for _,name in socket.if_nameindex())"
        self.assertTrue(self.sandbox.score(self.task(["lo"]), code)["passed"])

    def test_timeout_stops_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            score = self.sandbox.execute(Path(tmp), ["/usr/bin/python3", "-c", "while True: pass"], timeout=1)
        self.assertFalse(score["ok"])


if __name__ == "__main__":
    unittest.main()
