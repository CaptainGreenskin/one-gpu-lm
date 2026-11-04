"""Test 004 checkpoints on Rust prose, then train paired models at equal source-byte exposure."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent/"004-tokenizer-soak"
REFERENCE = HERE/"prepared-reference.json"
PUBLIC_RAW = "https://raw.githubusercontent.com/CaptainGreenskin/one-gpu-lm/main/experiments/004-tokenizer-soak/results/2026-11-04/summary.json"
sys.path.insert(0,str(PREVIOUS))
from trial import GPT, atomic_json, digest, read_rows, score_model


def wait_for_release(soak, out):
    campaign = json.loads((soak/"campaign.json").read_text())
    if campaign["code_commit"] != "010688e387b0d6e583c063ad66271420baa6a529" or campaign["duration_seconds"] != 28800:
        raise ValueError("unexpected upstream campaign")
    # Waiting does not initialize CUDA. Absolute deadline survives restarts.
    while True:
        state=json.loads((soak/"status.json").read_text())
        if state["status"] in ("failed","cancelled","paused"):
            raise RuntimeError(f"upstream campaign {state['status']}")
        if time.time() >= campaign["deadline_epoch"] and state["status"]=="completed":
            break
        atomic_json(out/"status.json",{"status":"waiting_for_soak","deadline_utc":campaign["deadline_utc"]})
        time.sleep(30)
    atomic_json(out/"status.json",{"status":"waiting_for_publication","url":PUBLIC_RAW})
    while True:
        # The workstation may have limited internet access. A trusted local publisher can supply
        # the exact publicly verified JSON and a receipt with its SHA-256 and immutable Git URL.
        local = out/"public-receipt.json"
        if local.exists():
            receipt=json.loads(local.read_text())
            blob=(out/"public-summary.json").read_bytes()
            if hashlib.sha256(blob).hexdigest()!=receipt["sha256"]:
                raise ValueError("invalid publication receipt")
            url=receipt["url"]
        else:
            url=PUBLIC_RAW
            try:
                with urlopen(Request(url+"?verify="+str(int(time.time())),headers={"User-Agent":"one-gpu-lm"}),timeout=12) as f:
                    blob=f.read()
            except (URLError, TimeoutError, OSError):
                time.sleep(60)
                continue
        summary=json.loads(blob)
        if (summary.get("status")!="completed" or summary.get("started_utc")!=campaign["started_utc"]
                or summary.get("deadline_utc")!=campaign["deadline_utc"] or not summary.get("complete_pairs")):
            raise ValueError("published result does not match completed upstream run")
        atomic_json(out/"release-gate.json",{"public_result":url,"public_summary_sha256":hashlib.sha256(blob).hexdigest(),
                                            "verified_epoch":time.time(),"prior_deadline":campaign["deadline_utc"]})
        return


def score_transfer(soak, data, out, ref, seconds, seed, arm):
    import torch
    key=f"t{seconds}-s{seed}-{arm}"
    candidates=[p for p in soak.glob(key+"-a*") if p.is_dir() and (p/"metrics.json").is_file()]
    if len(candidates)!=1:
        raise ValueError(f"expected one completed checkpoint for {key}")
    source=candidates[0]
    prev=json.loads((source/"metrics.json").read_text())
    if prev["arm"]!=arm or prev["seed"]!=seed or prev["requested_seconds"]!=seconds:
        raise ValueError("upstream checkpoint metadata mismatch")
    path=source/"model.pt"
    state=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    if state["arm"]!=arm or state["bos"]!=(256 if arm=="byte" else ref["bpe_vocab"]):
        raise ValueError("checkpoint vocabulary mismatch")
    with torch.device("meta"):
        model=GPT(*state["config"])
    model.load_state_dict(state["model"],assign=True)
    model.to("cuda")
    rows=read_rows(data,"val",arm,ref)
    score,count,elapsed=score_model(model,rows,"cuda",ref["splits"]["val"],arm,state["bos"])
    atomic_json(out/f"transfer-{key}.json",{"arm":arm,"seed":seed,"train_seconds_budget":seconds,
                "source_checkpoint_sha256":digest(path),"wiki_bpb":prev["bpb"],"technical_bpb":score,
                "technical_val_tokens":count,"technical_val_bytes":ref["splits"]["val"]["raw_bytes"],
                "validation_seconds":elapsed,"data_manifest_sha256":digest(data/"manifest.json")})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--soak",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--data",type=Path,default=ROOT/"data/rust-zh-pairs")
    args=p.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/".lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        ref=json.loads(REFERENCE.read_text())
        if ref!=json.loads((args.data/"manifest.json").read_text()):
            raise ValueError("wrong technical corpus")
        for split,meta in ref["splits"].items():
            for arm in ("byte","bpe"):
                if digest(args.data/f"{split}-{arm}.bin")!=meta[f"{arm}_sha256"]:
                    raise ValueError("encoded file mismatch")
        wait_for_release(args.soak,out)
        with (out/"environment.json").open("w") as env:
            subprocess.run([sys.executable,str(ROOT/"scripts/check_gpu.py")],stdout=env,check=True)
        pair=0
        for mode,budgets in [("transfer",(1200,2400)),("equal_bytes",(4,24))]:
            for budget in budgets:
                for seed in (42,43,44):
                    arms=("byte","bpe") if pair%2==0 else ("bpe","byte")
                    pair+=1
                    for arm in arms:
                        label=f"{mode}-{budget}-{seed}-{arm}"
                        atomic_json(out/"status.json",{"status":"running","trial":label,"epoch":time.time()})
                        if mode=="transfer":
                            if not (out/f"transfer-t{budget}-s{seed}-{arm}.json").exists():
                                score_transfer(args.soak,args.data,out,ref,budget,seed,arm)
                        else:
                            done=list(out.glob(f"{label}-a*/metrics.json"))
                            if len(done)>1: raise ValueError(f"duplicate results for {label}")
                            if done: continue
                            for attempt in range(1,4):
                                sub=out/f"{label}-a{attempt}"
                                if sub.exists(): continue
                                cmd=[sys.executable,str(PREVIOUS/"trial.py"),"--data",str(args.data),"--arm",arm,
                                     "--out",str(sub),"--seed",str(seed),"--epochs",str(budget)]
                                with (out/f"{sub.name}.log").open("a") as log:
                                    proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
                                if proc.returncode==0 and (sub/"metrics.json").exists(): break
                            else: raise RuntimeError(f"no successful attempt for {label}")
        with (out/"summary.json.tmp").open("w") as f:
            subprocess.run([sys.executable,str(HERE/"summarize.py"),str(out)],check=True,stdout=f)
        os.replace(out/"summary.json.tmp",out/"summary.json")
        atomic_json(out/"status.json",{"status":"completed","finished_epoch":time.time()})
    except BaseException as exc:
        atomic_json(out/"status.json",{"status":"failed","reason":str(exc),"epoch":time.time()})
        raise
    finally:
        lock.close()


if __name__=="__main__":
    main()
