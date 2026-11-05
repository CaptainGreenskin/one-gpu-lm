"""Score sealed checkpoints on fully separate test files; fail closed on any changed choice."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from run import HERE, SPEC, digest, jobs, write

sys.path.insert(0, str(HERE.parent / "004-tokenizer-soak"))
from trial import GPT, read_rows, score_model


def validate_seal(run):
    seal = json.loads((run / "sealed.json").read_bytes())
    expected = [f"{corpus}-{seed}-{arm}" for corpus, seed, arm in jobs()]
    if seal["protocol_sha256"] != digest(HERE / "protocol.json") or [r["trial"] for r in seal["trials"]] != expected:
        raise ValueError("Wrong protocol or incomplete set of locked choices")
    for record in seal["trials"]:
        directory = run / record["trial"]
        for field, filename in (("checkpoint", "checkpoint.pt"), ("decision", "decision.json"),
                                ("curve", "curve.json"), ("timing", "timing.json")):
            if digest(directory / filename) != record[field + "_sha256"]:
                raise ValueError(f"Seal mismatch: {record['trial']} {filename}")
    return seal


def evaluate(checkpoint, data, manifest, epoch):
    state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    if state["epoch"] != epoch:
        raise ValueError("Checkpoint epoch differs from locked selection")
    with torch.device("meta"):
        model = GPT(*state["config"])
    model.load_state_dict(state["model"], assign=True)
    model.cuda()
    offset = 0
    documents = []
    for doc in manifest["documents"]:
        n = doc["rows"]
        score, _, _ = score_model(model, data[offset:offset+n], "cuda", doc, "bpe", manifest["bpe_vocab"])
        documents.append({"path": doc["path"], "raw_bytes": doc["raw_bytes"], "bpb": score})
        offset += n
    nbytes = sum(d["raw_bytes"] for d in documents)
    if offset != len(data) or nbytes != manifest["splits"]["test"]["raw_bytes"]:
        raise ValueError("Incomplete test coverage")
    score = math.fsum(d["raw_bytes"] * d["bpb"] for d in documents) / nbytes
    return {"test_bpb": score, "test_raw_bytes": nbytes, "documents": documents}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    args = parser.parse_args()
    if (args.run / "test-scores.json").exists():
        parser.error("test scores already exist")
    start = time.monotonic()
    seal = validate_seal(args.run)
    torch.use_deterministic_algorithms(True)
    results = {}
    for corpus in SPEC["corpora"]:
        directory = args.test / corpus
        ref = HERE / "test-references" / f"{corpus}.json"
        if (directory / "manifest.json").read_bytes() != ref.read_bytes():
            raise ValueError("Wrong test documents")
        manifest = json.loads(ref.read_bytes())
        if digest(directory / "test-bpe.bin") != manifest["splits"]["test"]["bpe_sha256"]:
            raise ValueError("Wrong test tokens")
        data = read_rows(directory, "test", "bpe", manifest)
        for record in seal["trials"]:
            key = record["trial"]
            if key.split("-")[0] != corpus:
                continue
            path = args.run / key
            choice = json.loads((path / "decision.json").read_bytes())["selected_epoch"]
            results[key] = evaluate(path / "checkpoint.pt", data, manifest, choice)
    write(args.run / "test-scores.json", {"seal_sha256": digest(args.run / "sealed.json"),
          "test_stage_seconds_including_integrity_checks": time.monotonic() - start, "trials": results})


if __name__ == "__main__":
    main()
