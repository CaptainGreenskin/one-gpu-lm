"""Profile-controlled Hugging Face loading and local generation."""
import json
import time
from one_gpu_lm.common.receipts import write_json
from one_gpu_lm.common.agent_policy import BASH_TOOLS, prepare_context, validate_policy
from one_gpu_lm.common.tool_history import render_history


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
        kwargs = dict(self.profile["chat_template_kwargs"])
        if policy["format"] != "legacy":
            kwargs["tools"] = BASH_TOOLS
        dialogue = {}
        def encode(items):
            rendered = render_history(items, policy["format"] == "native_roundtrip")
            dialogue.update(structured_tool_calls=sum(len(m.get('tool_calls', [])) for m in rendered),
                            tool_results=sum(m['role'] == 'tool' for m in rendered))
            return self.tokenizer.apply_chat_template(rendered, tokenize=True, return_dict=False,
                add_generation_prompt=True, **kwargs)
        limit = min(int(value.get("max_input_tokens", 8192)), 8192)
        packing_started = time.monotonic()
        tokens, context = prepare_context(messages, encode, limit, policy["context"])
        packing_seconds = time.monotonic() - packing_started
        if len(tokens) > limit:
            return {"error": "context_limit", "input_tokens": len(tokens), "context": context,
                    "packing_seconds": packing_seconds, "dialogue": dialogue}
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
                "input_tokens": len(tokens), "generated_tokens": len(suffix),
                "seconds": time.monotonic() - started, "hit_generation_limit": len(suffix) >= new,
                "agent_policy": policy, "context": context, "dialogue": dialogue, "packing_seconds": packing_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "adapter": self.adapter}
