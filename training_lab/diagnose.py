"""Recheck regressions with the saved adapter enabled and disabled; no training."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .records import load_split, sha256, write_json
from .sandbox import Sandbox, extract_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "run", "test", "node"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    output = args.run / "control.json"
    if output.exists():
        raise ValueError("Keep the previous control receipt")
    profile = json.loads((args.run / "profile.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    data = json.loads((args.run / "data.json").read_text())
    if sha256(args.test) != data["splits"]["test"]["sha256"]:
        raise ValueError("Held-out data changed")
    if result["status"] != "completed":
        raise ValueError("Main pilot must finish first")
    regressions = {a["id"] for b, a in zip(result["baseline"], result["adapted"])
                   if b["id"] == a["id"] and b["passed"] and not a["passed"]}
    tasks = [r for r in load_split(args.test, "test") if r["id"] in regressions]
    if not tasks:
        raise ValueError("No regressions to diagnose")
    started = time.monotonic()
    receipt = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": sha256(__file__), "purpose": "diagnostic_only_no_tuning", "tasks": []}
    import torch
    import transformers
    from peft import PeftModel
    torch.set_num_threads(2)
    torch.manual_seed(profile["training"]["seed"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if profile["loader"] not in {"AutoModelForCausalLM", "AutoModelForImageTextToText"}:
        raise ValueError("Unregistered loader")
    base = getattr(transformers, profile["loader"]).from_pretrained(args.model,
        dtype=getattr(torch, profile["dtype"]), device_map={"": 0}, local_files_only=True,
        trust_remote_code=False, attn_implementation=profile["attention"])
    model = PeftModel.from_pretrained(base, args.run / "private" / "adapter", is_trainable=False)
    model.eval()
    sandbox = Sandbox(args.node)
    for task in tasks:
        row = {"id": task["id"], "language": task["language"]}
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": task["prompt"]}],
            tokenize=True, return_dict=False, add_generation_prompt=True, **profile["chat_template_kwargs"])
        inputs = torch.tensor([inputs], device="cuda")
        for label, disabled in [("adapter_enabled", False), ("adapter_disabled", True)]:
            with (model.disable_adapter() if disabled else nullcontext()), torch.inference_mode():
                ids = model.generate(inputs, attention_mask=torch.ones_like(inputs), use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **profile["generation"])
            text = tokenizer.decode(ids[0, inputs.shape[1]:], skip_special_tokens=True)
            original = args.run / "private" / (("baseline" if disabled else "adapted") + "-" + task["id"] + ".txt")
            row[label] = {**sandbox.score(task, extract_code(text)),
                          "text_equal_to_original_stage": text == original.read_text()}
        receipt["tasks"].append(row)
        print(task["id"], json.dumps({k: row[k]["passed"] for k in ["adapter_enabled", "adapter_disabled"]}), flush=True)
    receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
    write_json(output, receipt)


if __name__ == "__main__":
    main()
