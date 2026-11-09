"""Run a bounded BF16 + LoRA pilot through a model profile and private JSONL inputs."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import time

from .records import data_receipt, load_split, sha256, write_json
from .sandbox import Sandbox, extract_code


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["profile", "model", "train", "test", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output exists; preserve prior attempts and choose a new run ID")
    args.out.mkdir(parents=True)
    profile = json.loads(args.profile.read_text())
    write_json(args.out / "profile.json", profile)
    write_json(args.out / "source.json", {"schema_version": 1, "profile_sha256": sha256(args.profile),
        "files": {"training_lab/" + p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}})
    started = time.monotonic()
    result = {"schema_version": 1, "started_utc": utc(), "status": "starting", "purpose": "pipeline-smoke-only"}
    telemetry = None

    def stage(name, **fields):
        write_json(args.out / "status.json", {"stage": name, "updated_utc": utc(), **fields})
        with (args.out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": name, "utc": utc(), **fields}) + "\n")
        print(name, json.dumps(fields), flush=True)

    try:
        stage("validate_data_and_sandbox")
        train = load_split(args.train, "train")
        test = load_split(args.test, "test")
        if len(train) < profile["acceptance"]["minimum_training_examples"]:
            raise ValueError("Too few training examples")
        receipt = data_receipt(args.train, args.test, train, test)
        write_json(args.out / "data.json", receipt)
        sandbox = Sandbox(args.node)
        references = []
        for split, rows in [("train", train), ("test", test)]:
            for row in rows:
                score = sandbox.score(row, row["target" if split == "train" else "reference"])
                references.append({"id": row["id"], "split": split, **score})
        write_json(args.out / "references.json", references)
        if not all(r["passed"] for r in references):
            raise ValueError("Reference solution failed: " + ",".join(r["id"] for r in references if not r["passed"]))
        if args.validate_only:
            result.update(status="validated", references=len(references))
            stage("validated", references=len(references))
            return

        import torch
        import transformers
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        torch.set_num_threads(2)
        torch.manual_seed(profile["training"]["seed"])
        random.seed(profile["training"]["seed"])
        if not torch.cuda.is_available():
            raise ValueError("CUDA is required; no implicit CPU fallback")
        packages = ["torch", "transformers", "peft", "accelerate", "huggingface-hub", "safetensors", "tokenizers", "flash-linear-attention", "fla-core"]
        environment = {"packages": {}, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                       "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                       "compute_capability": list(torch.cuda.get_device_capability(0)),
                       "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        for package in packages:
            try:
                environment["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                environment["packages"][package] = None
        write_json(args.out / "environment.json", environment)
        downloaded = json.loads((args.model / "download-receipt.json").read_text())
        if any(downloaded[k] != profile[k] for k in ["model_id", "revision"]):
            raise ValueError("Profile and verified checkpoint identity differ")
        write_json(args.out / "download.json", downloaded)
        with (args.out / "telemetry.csv").open("w") as log:
            telemetry = subprocess.Popen(["nvidia-smi", "-i", "0", "--query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,memory.used", "--format=csv,nounits", "-l", "2"], stdout=log, stderr=subprocess.DEVNULL)

        stage("load_model")
        load_start = time.monotonic()
        permitted = {"AutoModelForCausalLM", "AutoModelForImageTextToText"}
        if profile["loader"] not in permitted:
            raise ValueError("Unregistered model loader")
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        loader = getattr(transformers, profile["loader"])
        model, loading = loader.from_pretrained(args.model, dtype=getattr(torch, profile["dtype"]),
            device_map={"": 0}, local_files_only=True, trust_remote_code=False,
            attn_implementation=profile["attention"], output_loading_info=True)
        write_json(args.out / "loading.json", loading)
        if any(loading.get(k) for k in ["missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys"]):
            raise ValueError("Strict checkpoint loading inspection failed")
        result.update(load_seconds=time.monotonic() - load_start,
                      model_parameters=sum(p.numel() for p in model.parameters()),
                      load_allocated_bytes=torch.cuda.memory_allocated())

        def prompt_tokens(row):
            return tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True, **profile["chat_template_kwargs"])

        def generate(row, max_new_tokens=None):
            inputs = torch.tensor([prompt_tokens(row)], device="cuda")
            config = dict(profile["generation"])
            if max_new_tokens is not None:
                config["max_new_tokens"] = max_new_tokens
            torch.cuda.synchronize()
            begin = time.monotonic()
            with torch.inference_mode():
                ids = model.generate(inputs, attention_mask=torch.ones_like(inputs), use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **config)
            torch.cuda.synchronize()
            output = ids[0, inputs.shape[1]:].tolist()
            return tokenizer.decode(output, skip_special_tokens=True), output, time.monotonic() - begin, inputs.numel()

        def evaluate(label):
            stage(label)
            model.eval()
            scores = []
            private = args.out / "private"
            private.mkdir(exist_ok=True)
            for row in test:
                text, ids, seconds, input_tokens = generate(row)
                (private / (label + "-" + row["id"] + ".txt")).write_text(text)
                score = sandbox.score(row, extract_code(text))
                scores.append({"id": row["id"], "language": row["language"], **score,
                    "input_tokens": input_tokens, "generated_tokens": len(ids), "generation_seconds": seconds,
                    "hit_generation_limit": len(ids) >= profile["generation"]["max_new_tokens"]})
                write_json(args.out / (label + ".json"), scores)
                stage(label, completed=len(scores), passed=sum(r["passed"] for r in scores))
            return scores

        model.eval()
        generate(train[0], 8)  # warm-up outside timing/quality comparisons
        result["baseline"] = evaluate("baseline")
        stage("prepare_lora")
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, **profile["lora"]))
        parameters = [p for p in model.parameters() if p.requires_grad]
        result["trainable_parameters"] = sum(p.numel() for p in parameters)
        if not parameters or result["trainable_parameters"] >= result["model_parameters"] * 0.05:
            raise ValueError("Unexpected LoRA trainable parameter count")
        encoded = []
        for row in train:
            prefix = prompt_tokens(row)
            suffix = tokenizer.encode(row["target"] + tokenizer.eos_token, add_special_tokens=False)
            tokens = prefix + suffix
            if len(tokens) > profile["training"]["max_sequence_length"]:
                raise ValueError("Training record exceeds context budget; do not silently truncate")
            encoded.append((row["id"], tokens, [-100] * len(prefix) + suffix))
        result["training_token_lengths"] = {"min": min(len(r[1]) for r in encoded), "max": max(len(r[1]) for r in encoded)}
        model.enable_input_require_grads()
        if profile["training"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(parameters, lr=profile["training"]["learning_rate"], weight_decay=0.0)
        # Keep only small scalar snapshots for this smoke check; this is not a full base-weight hash.
        frozen = [(n, p, p.detach().reshape(-1)[:8].clone()) for n, p in model.named_parameters() if not p.requires_grad][:3]
        initial_adapters = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        train_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        curve, step, tokens_seen, supervised_tokens = [], 0, 0, 0
        for epoch in range(profile["training"]["epochs"]):
            order = list(range(len(encoded)))
            random.Random(profile["training"]["seed"] + epoch).shuffle(order)
            for index in order:
                task_id, ids, labels = encoded[index]
                step += 1
                optimizer.zero_grad(set_to_none=True)
                input_ids = torch.tensor([ids], device="cuda")
                labels = torch.tensor([labels], device="cuda")
                torch.cuda.synchronize()
                begin = time.monotonic()
                output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels, use_cache=False)
                loss = output.loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, profile["training"]["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                tokens_seen += len(ids)
                supervised_tokens += int((labels != -100).sum())
                curve.append({"step": step, "epoch": epoch + 1, "id": task_id, "loss": loss.item(),
                              "grad_norm": norm.item(), "seconds": time.monotonic() - begin, "tokens": len(ids)})
                write_json(args.out / "curve.json", curve)
                stage("training", step=step, loss=loss.item())
                del output, loss, labels, input_ids
        result.update(training_seconds=time.monotonic() - train_start, training_steps=step,
            training_input_tokens=tokens_seen, supervised_tokens=supervised_tokens,
            train_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            train_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            frozen_sample_unchanged=all(torch.equal(p.detach().reshape(-1)[:8], sample) for _, p, sample in frozen),
            frozen_gradients_absent=all(p.grad is None for p in model.parameters() if not p.requires_grad),
            adapter_changed=any(not torch.equal(p.detach(), initial_adapters[n]) for n, p in model.named_parameters() if p.requires_grad))
        if not all(result[k] for k in ["frozen_sample_unchanged", "frozen_gradients_absent", "adapter_changed"]):
            raise ValueError("Parameter update acceptance failed")
        del initial_adapters, optimizer
        for p in parameters:
            p.grad = None
        model.gradient_checkpointing_disable()
        adapter_path = args.out / "private" / "adapter"
        model.peft_config["default"].base_model_name_or_path = profile["model_id"]
        model.peft_config["default"].revision = profile["revision"]
        model.save_pretrained(adapter_path, safe_serialization=True)
        write_json(args.out / "adapter.json", {"base_model": profile["model_id"], "revision": profile["revision"],
            "public_weights": False, "files": {p.name: sha256(p) for p in adapter_path.glob("*.safetensors")}})
        model.eval()
        _, before, _, _ = generate(train[0], 16)
        base = model.unload()
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        model.eval()
        _, after, _, _ = generate(train[0], 16)
        result["adapter_reload_greedy_ids_equal"] = before == after
        if before != after:
            raise ValueError("Saved adapter reload changed greedy output")
        gc.collect()
        torch.cuda.empty_cache()
        result["adapted"] = evaluate("adapted")
        result["status"] = "completed"
        stage("completed", steps=step)
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        stage("failed", error_type=type(error).__name__)
        raise
    finally:
        if telemetry:
            telemetry.terminate()
            telemetry.wait(timeout=10)
        result.update(finished_utc=utc(), wall_seconds=time.monotonic() - started)
        write_json(args.out / "result.json", result)


if __name__ == "__main__":
    main()
