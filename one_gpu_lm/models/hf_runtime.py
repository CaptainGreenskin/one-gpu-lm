"""Profile-controlled Hugging Face loading and local generation."""
import json
import hashlib
import time
from one_gpu_lm.common.receipts import write_json
from one_gpu_lm.common.agent_policy import prepare_context, validate_policy
from one_gpu_lm.common.tokenization import encode_prefix


class Runtime:
    def __init__(self, model_path, profile, out, adapter=None):
        import torch
        import transformers
        self.torch = torch
        self.profile = profile
        torch.set_num_threads(2)
        torch.manual_seed(101)
        receipt = json.loads((model_path / "download-receipt.json").read_text())
        if any(receipt[k] != profile[k] for k in ["model_id", "revision"]):
            raise ValueError("Verified model identity does not match profile")
        if profile["loader"] not in {"AutoModelForCausalLM", "AutoModelForImageTextToText"}:
            raise ValueError("Unregistered loader")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model, loading = getattr(transformers, profile["loader"]).from_pretrained(
            model_path, dtype=getattr(torch, profile["dtype"]), device_map={"": 0},
            local_files_only=True, trust_remote_code=False, attn_implementation=profile["attention"], output_loading_info=True)
        write_json(out / "loading.json", loading)
        if any(loading.get(k) for k in ["missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys"]):
            raise ValueError("Checkpoint loading inspection failed")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
        self.model.eval()
        self.adapter = bool(adapter)

    def generate(self, value):
        torch = self.torch
        messages = value["messages"]
        if any(m["role"] not in {"system", "user", "assistant"} for m in messages):
            raise ValueError("Unexpected message role")
        policy = validate_policy(value.get("agent_policy"))
        dialogue = {}
        def encode(items):
            tokens, receipt = encode_prefix(self.tokenizer, items,
                self.profile['chat_template_kwargs'], policy['format'])
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
        new = min(int(value.get("max_new_tokens", 512)), 512)
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
        return {"text": self.tokenizer.decode(suffix, skip_special_tokens=True), "ids": suffix,
                "input_tokens": len(tokens), "input_tokens_sha256": input_tokens_sha256, "generated_tokens": len(suffix),
                "seconds": time.monotonic() - started, "hit_generation_limit": len(suffix) >= new,
                "agent_policy": policy, "context": context, "dialogue": dialogue, "packing_seconds": packing_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "adapter": self.adapter}
