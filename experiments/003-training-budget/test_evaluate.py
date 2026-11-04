"""CPU regression for overlap, padding and article boundaries in full-byte scoring."""
import math
from pathlib import Path
import tempfile
import unittest

import torch

from evaluate import GPT, evaluate


class FullValidationTest(unittest.TestCase):
    def test_batched_scores_match_individual_predictions(self):
        torch.manual_seed(711)
        config = [16, 2, 1, 8]
        model = GPT(*config).eval()
        docs = [("long", bytes(range(12))), ("short", b"abc")]
        # For length 8 and stride 4 the final window of the first article starts at byte 3.
        context_starts = [[0] * 8 + [3] * 3, [0, 0]]
        with tempfile.TemporaryDirectory() as path:
            checkpoint = Path(path) / "m.pt"
            torch.save({"config": config, "model": model.state_dict()}, checkpoint)
            result = evaluate(checkpoint, iter(docs), "cpu")
        with torch.inference_mode():
            for item, (_, body), starts in zip(result["articles"], docs, context_starts):
                bits = 0
                for target, start in enumerate(starts, 1):
                    prefix = torch.tensor([list(body[start:target])])
                    log_probs = torch.log_softmax(model(prefix)[0, -1], dim=0)
                    bits -= log_probs[body[target]].item() / math.log(2)
                self.assertEqual(item["targets"], len(body) - 1)
                self.assertAlmostEqual(item["bits_per_byte"], bits / len(starts), places=5)
        self.assertEqual(result["target_bytes"], 13)


if __name__ == "__main__":
    unittest.main()
