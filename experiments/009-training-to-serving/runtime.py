"""Load a verified local BPE bundle and run causal decoding with a per-request static KV cache."""
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "002-first-train"))
from train import GPT, digest


class KVCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.length = 0
        self.layers = []


class Runtime:
    def __init__(self, model, tokenizer, bos):
        self.model, self.tokenizer, self.bos = model.eval(), tokenizer, bos
        self.context = model.position.num_embeddings
        self.device = model.tokens.weight.device

    @classmethod
    def load(cls, directory, device):
        bundle = json.loads((directory / "bundle.json").read_bytes())
        for f in ("weights.pt", "tokenizer.json"):
            if digest(directory / f) != bundle["sha256"][f]:
                raise ValueError(f"Bundle hash mismatch: {f}")
        state = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True, mmap=True)
        if list(state["config"]) != bundle["config"]:
            raise ValueError("Wrong checkpoint configuration")
        tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        if tok.get_vocab_size() + 1 != state["config"][4]:
            raise ValueError("Tokenizer vocabulary does not match model")
        with torch.device("meta"):
            model = GPT(*state["config"])
        model.load_state_dict(state["model"], assign=True)
        model.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        return cls(model, tok, tok.get_vocab_size())

    @torch.inference_mode()
    def cached(self, ids, cache, last_only=False):
        start = cache.length
        batch, n = ids.shape
        end = start + n
        if n <= 0 or end > cache.capacity or cache.capacity > self.context:
            raise ValueError("KV cache capacity exceeded")
        positions = torch.arange(start, end, device=ids.device)
        h = self.model.tokens(ids) + self.model.position(positions)
        # A multi-token suffix needs a lower-right causal mask aligned to its prefix.
        mask = positions[:, None] >= torch.arange(end, device=ids.device) if start and n > 1 else None
        for i, block in enumerate(self.model.layers):
            width = h.shape[-1]
            heads = block.heads
            q, k, v = block.qkv(block.norm1(h)).view(batch, n, 3, heads, width // heads).permute(2, 0, 3, 1, 4)
            if not start:
                shape = (batch, heads, cache.capacity, width // heads)
                cache.layers.append((k.new_empty(shape), v.new_empty(shape)))
            keys, values = cache.layers[i]
            keys[:, :, start:end].copy_(k)
            values[:, :, start:end].copy_(v)
            attended = F.scaled_dot_product_attention(q, keys[:, :, :end], values[:, :, :end],
                                                       attn_mask=mask, is_causal=not start and n > 1)
            h = h + block.proj(attended.transpose(1, 2).reshape(batch, n, width))
            h = h + block.mlp(block.norm2(h))
        cache.length = end
        return F.linear(self.model.norm(h[:, -1:] if last_only else h), self.model.tokens.weight)

    @torch.inference_mode()
    def full_last(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.model.layers(self.model.tokens(ids) + self.model.position(pos))
        return F.linear(self.model.norm(h[:, -1:]), self.model.tokens.weight)

    @torch.inference_mode()
    def generate(self, prompt, new_tokens, use_cache=True):
        if type(new_tokens) is not int or not 1 <= new_tokens <= self.context:
            raise ValueError("max_new_tokens outside the context window")
        prompt_ids = [self.bos, *self.tokenizer.encode(prompt, add_special_tokens=False).ids]
        if len(prompt_ids) + new_tokens - 1 > self.context:
            raise ValueError("prompt and requested continuation exceed the context window")
        ids = torch.tensor([prompt_ids], device=self.device)
        cache = KVCache(len(prompt_ids) + new_tokens - 1) if use_cache else None
        generated = []
        for _ in range(new_tokens):
            logits = self.cached(ids, cache, last_only=True) if use_cache else self.full_last(ids)
            # BOS is input-only in the training objective.
            token = logits[:, -1, :self.bos].argmax(dim=-1, keepdim=True)
            generated.append(token)
            ids = token if use_cache else torch.cat((ids, token), dim=1)
        result = torch.cat(generated, dim=1)[0].tolist()
        return {"generated_text": self.tokenizer.decode(result, skip_special_tokens=False),
                "generated_ids": result, "prompt_tokens": len(prompt_ids), "generated_tokens": len(result)}
