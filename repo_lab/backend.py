"""Private Unix-socket Transformers backend, separate from untrusted task namespaces."""
import argparse
import json
import os
from pathlib import Path
import socket
import socketserver
import time

from training_lab.records import sha256, write_json


MAX_REQUEST_BYTES = 2**21


def request(path, value, timeout=180):
    payload = json.dumps(value).encode() + b"\n"
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Backend request exceeds byte limit")
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(payload)
        with client.makefile("rb") as stream:
            response = stream.readline(MAX_REQUEST_BYTES + 1)
    if len(response) > MAX_REQUEST_BYTES or not response.endswith(b"\n"):
        raise ValueError("Invalid backend response")
    result = json.loads(response)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result


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
        messages = [{"role": m["role"], "content": m["content"]} for m in value["messages"]]
        if any(m["role"] not in {"system", "user", "assistant"} for m in messages):
            raise ValueError("Unexpected message role")
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, **self.profile["chat_template_kwargs"])
        limit = min(int(value.get("max_input_tokens", 8192)), 8192)
        if len(tokens) > limit:
            return {"error": "context_limit", "input_tokens": len(tokens)}
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
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "adapter": self.adapter}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "profile", "socket", "out"]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--adapter", type=Path)
    args = p.parse_args()
    if args.socket.exists() or args.out.exists():
        raise ValueError("Preserve previous backend and receipts; choose new paths")
    args.out.mkdir(parents=True)
    profile = json.loads(args.profile.read_text())
    runtime = Runtime(args.model, profile, args.out, args.adapter)

    identity = {"ready": True, "model_id": profile["model_id"], "revision": profile["revision"],
                "profile_sha256": sha256(args.profile), "adapter": bool(args.adapter),
                "adapter_sha256": {p.name: sha256(p) for p in sorted(args.adapter.glob("*.safetensors"))} if args.adapter else {}}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
                if len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                    raise ValueError("Invalid request size")
                value = json.loads(raw)
                if value.get("operation") == "health":
                    result = identity
                elif value.get("operation") == "generate":
                    result = runtime.generate(value)
                else:
                    raise ValueError("Unregistered operation")
            except Exception as error:
                result = {"error": type(error).__name__}
            self.wfile.write(json.dumps(result).encode() + b"\n")

    os.umask(0o077)
    with socketserver.UnixStreamServer(str(args.socket), Handler) as server:
        write_json(args.out / "ready.json", {"ready": True, "profile_sha256": sha256(args.profile),
                    "model_id": profile["model_id"], "revision": profile["revision"], "adapter": bool(args.adapter)})
        server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
