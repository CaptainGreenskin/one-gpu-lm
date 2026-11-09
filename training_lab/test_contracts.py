import copy
import json
from pathlib import Path
import tempfile
import unittest

from .records import data_receipt, load_split, write_json
from .sandbox import extract_code


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.row = {"id": "a", "language": "python", "prompt": "test contract", "target": "pass",
                    "cases": [{"input": None, "expected": None}],
                    "provenance": {"kind": "synthetic-fixture", "source": "unit test",
                                   "rights_basis": "original fixture", "training_allowed": True}}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows, name="train.jsonl"):
        p = self.root / name
        p.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return p

    def test_no_implicit_permission(self):
        del self.row["provenance"]["training_allowed"]
        with self.assertRaisesRegex(ValueError, "rights"):
            load_split(self.write([self.row]), "train")

    def test_duplicate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_split(self.write([self.row, self.row]), "train")

    def test_exact_prompt_leakage_fails_even_with_different_ids(self):
        test = copy.deepcopy(self.row)
        test["id"] = "b"
        with self.assertRaisesRegex(ValueError, "prompt overlap"):
            data_receipt(None, None, [self.row], [test])

    def test_private_default_and_content_identity(self):
        p = self.write([self.row])
        test = copy.deepcopy(self.row)
        test.update(id="b", prompt="different task", reference=test.pop("target"))
        q = self.write([test], "test.jsonl")
        value = data_receipt(p, q, load_split(p, "train"), load_split(q, "test"))
        self.assertFalse(value["raw_data_public"])
        self.assertFalse(value["splits"]["train"]["redistribution_allowed"])
        self.assertNotEqual(value["splits"]["train"]["sha256"], value["splits"]["test"]["sha256"])

    def test_fenced_and_bare_code(self):
        self.assertEqual(extract_code("```python\ndef solve(x): return x\n```"), "def solve(x): return x")
        self.assertEqual(extract_code("function solve(x) {return x;}"), "function solve(x) {return x;}")

    def test_library_loading_sets_have_stable_json(self):
        path = self.root / "load.json"
        write_json(path, {"missing_keys": set(), "unexpected_keys": {"b", "a"}})
        self.assertEqual(json.loads(path.read_text()), {"missing_keys": [], "unexpected_keys": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
