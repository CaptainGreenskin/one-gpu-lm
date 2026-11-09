"""Small CPU tests of recomputation, including frozen-input LoRA gradients."""
import copy
import unittest

import torch
from torch.nn import functional as F

from bench import GPT, adapt, forward


class RecomputeTest(unittest.TestCase):
    def test_outputs_gradients_and_updates(self):
        torch.set_num_threads(1)
        for lora in (False, True):
            with self.subTest(lora=lora):
                torch.manual_seed(1729)
                plain = GPT(32, 4, 2, 16, 64, 96)
                if lora:
                    adapt(plain)
                cached = copy.deepcopy(plain)
                tokens = torch.randint(64, (2, 12))
                reference = plain(tokens)
                torch.testing.assert_close(forward(plain, tokens, False), reference)
                torch.testing.assert_close(forward(cached, tokens, True), reference)
                opts = [torch.optim.SGD((p for p in m.parameters() if p.requires_grad), lr=0.01)
                        for m in (plain, cached)]
                # Two steps ensure the initially zero LoRA B allows A gradients on step 2.
                for _ in range(2):
                    for model, opt, recompute in zip((plain, cached), opts, (False, True)):
                        opt.zero_grad(set_to_none=True)
                        logits = forward(model, tokens[:, :-1], recompute)
                        F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten()).backward()
                    for p, q in zip(plain.parameters(), cached.parameters()):
                        if p.requires_grad:
                            self.assertIsNotNone(p.grad)
                            self.assertIsNotNone(q.grad)
                            torch.testing.assert_close(p.grad, q.grad)
                    for opt in opts:
                        opt.step()
                    for p, q in zip(plain.parameters(), cached.parameters()):
                        torch.testing.assert_close(p, q)
                if lora:
                    self.assertGreater(plain.layers[0].qkv.a.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
