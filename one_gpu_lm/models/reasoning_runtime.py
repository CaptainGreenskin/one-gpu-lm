"""Explicit reasoning variants; keep the published 023/024 runtime byte-stable."""
import json
import hashlib
import time
from .hf_runtime import Runtime as LegacyRuntime
from one_gpu_lm.common.agent_policy import prepare_context, validate_policy
from one_gpu_lm.common.tokenization import encode_prefix
from one_gpu_lm.common.reasoning import settings, separate


class Runtime(LegacyRuntime):
    def generate(self, value):
        torch = self.torch
        messages = value["messages"]
        if any(m["role"] not in {"system", "user", "assistant"} for m in messages):
            raise ValueError("Unexpected message role")
        policy = validate_policy(value.get("agent_policy"))
        dialogue = {}
        template_kwargs, generation_ceiling = settings(self.profile, value)
        def encode(items):
            tokens, receipt = encode_prefix(self.tokenizer, items,
                template_kwargs, policy['format'])
            dialogue.update(receipt)
            return tokens
        limit = min(int(value.get("max_input_tokens", 8192)), 8192)
        packing_started = time.monotonic()
        tokens, context = prepare_context(messages, encode, limit, policy["context"])
        packing_seconds = time.monotonic() - packing_started
        input_tokens_sha256 = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
        if len(tokens) > limit:
            return {"error": "context_limit", "input_tokens": len(tokens), "context": context,
                    "packing_seconds": packing_seconds, "dialogue": dialogue, "input_tokens_sha256": input_tokens_sha256}
        new = min(int(value.get("max_new_tokens", 512)), generation_ceiling)
        if new < 1:
            return {"error": "generation_budget_exhausted"}
        inputs = torch.tensor([tokens], device="cuda")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            ids = self.model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=new,
                do_sample=False, use_cache=True, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        suffix = ids[0, len(tokens):].tolist()
        result = {"text": self.tokenizer.decode(suffix, skip_special_tokens=True), "ids": suffix,
                "input_tokens": len(tokens), "input_tokens_sha256": input_tokens_sha256, "generated_tokens": len(suffix),
                "seconds": time.monotonic() - started, "hit_generation_limit": len(suffix) >= new,
                "agent_policy": policy, "context": context, "dialogue": dialogue, "packing_seconds": packing_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "adapter": self.adapter}
        if self.profile.get("generation_modes"):
            result["separation"] = separate(self.tokenizer, suffix, self.profile, value)
        return result
