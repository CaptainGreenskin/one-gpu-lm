"""Reject unpaired or unequal exposure; summarize the three seeds at each preregistered budget."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent


def paired(arm_metrics, score_field):
    a,b=arm_metrics
    if not all(math.isfinite(m[score_field]) for m in (a,b)):
        raise ValueError("nonfinite validation score")
    return {"seed":a["seed"],"byte_bpb":a[score_field],"bpe_bpb":b[score_field],
            "bpe_gain":a[score_field]-b[score_field]}


def group(rows):
    return {"pairs":rows,"mean_byte_bpb":mean(r["byte_bpb"] for r in rows),
            "mean_bpe_bpb":mean(r["bpe_bpb"] for r in rows),"mean_bpe_gain":mean(r["bpe_gain"] for r in rows)}


def summarize(out):
    ref_path=HERE/"prepared-reference.json"
    ref=json.loads(ref_path.read_text())
    manifest=hashlib.sha256(ref_path.read_bytes()).hexdigest()
    train,val=ref["splits"]["train"],ref["splits"]["val"]
    transfer={}
    for budget in (1200,2400):
        rows=[]
        for seed in (42,43,44):
            metrics=[json.loads((out/f"transfer-t{budget}-s{seed}-{arm}.json").read_text()) for arm in ("byte","bpe")]
            for arm,m in zip(("byte","bpe"),metrics):
                tokens=val["raw_bytes"] if arm=="byte" else val["bpe_tokens"]
                if ((m["arm"],m["seed"],m["train_seconds_budget"])!=(arm,seed,budget) or
                    m["technical_val_bytes"]!=val["raw_bytes"] or m["technical_val_tokens"]!=tokens or
                    m["data_manifest_sha256"]!=manifest):
                    raise ValueError("invalid transfer comparison")
            rows.append(paired(metrics,"technical_bpb"))
        transfer[str(budget)]=group(rows)
    equal={}
    for epochs in (4,24):
        rows=[]
        for seed in (42,43,44):
            metrics=[]
            for arm in ("byte","bpe"):
                paths=list(out.glob(f"equal_bytes-{epochs}-{seed}-{arm}-a*/metrics.json"))
                if len(paths)!=1: raise ValueError(f"missing/duplicate completed trial: {epochs}, {seed}, {arm}")
                m=json.loads(paths[0].read_text())
                exposure=train["raw_bytes"]*epochs
                tkn=(train["raw_bytes"] if arm=="byte" else train["bpe_tokens"])*epochs
                val_tokens=val["raw_bytes"] if arm=="byte" else val["bpe_tokens"]
                if ((m["arm"],m["seed"],m["epochs"])!=(arm,seed,epochs) or m["device"]!="cuda" or
                    m["train_raw_bytes"]!=exposure or m["train_tokens"]!=tkn or m["steps"]!=epochs*math.ceil(train["rows"]/8) or
                    m["val_raw_bytes"]!=val["raw_bytes"] or m["val_tokens"]!=val_tokens or m["manifest_sha256"]!=manifest):
                    raise ValueError("invalid byte-matched trial")
                metrics.append(m)
            a,b=metrics
            if a["sample_order_sha256"]!=b["sample_order_sha256"]:
                raise ValueError("matched byte count but different training chunks/order")
            rows.append({**paired(metrics,"bpb"),"raw_bytes_per_arm":exposure,"steps_per_arm":a["steps"],
                         "byte_seconds":a["train_seconds"],"bpe_seconds":b["train_seconds"],
                         "sample_order_sha256":a["sample_order_sha256"]})
        equal[str(epochs)]=group(rows)
    return {"source_revision":ref["source"]["revision"],"val_raw_bytes":val["raw_bytes"],
            "cross_domain_from_wiki":transfer,"matched_technical_training":equal}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    print(json.dumps(summarize(parser.parse_args().out),indent=2,allow_nan=False))
