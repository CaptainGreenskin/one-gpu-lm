"""Run alternating, paired trials until the absolute eight-hour deadline; restartable."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from trial import HERE, ROOT, atomic_json, digest


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def plans():
    pair = 0
    for seconds in (45, 300, 1200, 2400):
        for seed in (42, 43, 44):
            for arm in (("byte", "bpe") if pair % 2 == 0 else ("bpe", "byte")):
                yield seconds, seed, arm
            pair += 1
    seed = 45
    while True:
        for arm in (("byte", "bpe") if pair % 2 == 0 else ("bpe", "byte")):
            yield 300, seed, arm
        pair += 1
        seed += 1


def telemetry(out, stop):
    fields = "timestamp,index,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm"
    path = out / "telemetry.csv"
    with path.open("a", buffering=1) as file:
        if file.tell() == 0:
            file.write("utc,"+fields+"\n")
        while not stop.is_set():
            try:
                value = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu="+fields, "--format=csv,noheader,nounits"],
                                                text=True, timeout=10).strip()
                file.write(f"{utc(time.time())},{value}\n")
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"telemetry error: {exc}", flush=True)
            stop.wait(30)


def verify_data(data):
    ref = json.loads((HERE / "prepared-reference.json").read_text())
    if json.loads((data / "manifest.json").read_text()) != ref:
        raise ValueError("unexpected encoded corpus manifest")
    for split, meta in ref["splits"].items():
        for arm in ("byte", "bpe"):
            if digest(data / f"{split}-{arm}.bin") != meta[f"{arm}_sha256"]:
                raise ValueError("encoded data hash mismatch")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data/tokenizer-pairs")
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_data(args.data)
    file = out / "campaign.json"
    if file.exists():
        cfg = json.loads(file.read_text())
    else:
        subprocess.run([sys.executable, str(ROOT / "scripts/check_gpu.py")], check=True,
                       stdout=(out / "environment.json").open("w"))
        start = time.time()
        cfg = {"started_epoch": start, "started_utc": utc(start), "deadline_epoch": start + 28800,
               "deadline_utc": utc(start + 28800), "duration_seconds": 28800,
               "code_commit": subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"], text=True).strip(),
               "manifest_sha256": digest(args.data / "manifest.json")}
        atomic_json(file, cfg)
    deadline = cfg["deadline_epoch"]
    print(json.dumps(cfg), flush=True)
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(out, stop), daemon=True)
    monitor.start()
    events = (out / "events.jsonl").open("a", buffering=1)

    def event(status, **fields):
        events.write(json.dumps({"utc":utc(time.time()),"status":status,**fields})+"\n")

    try:
        for budget, seed, arm in plans():
            if time.time() >= deadline:
                break
            name = f"t{budget}-s{seed}-{arm}"
            attempts = sorted(d for d in out.glob(name+"-a*") if d.is_dir())
            if any((d/"metrics.json").is_file() for d in attempts):
                continue
            if len(attempts) >= 3:
                raise RuntimeError(f"three failed/incomplete attempts for {name}; inspect stderr before restarting")
            if shutil.disk_usage(out).free < 8*1024**3:
                raise RuntimeError("less than 8 GiB of free disk space")
            for attempt in range(len(attempts)+1, 4):
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                directory = out/f"{name}-a{attempt}"
                atomic_json(out/"status.json", {"status":"running","trial":directory.name,"deadline_utc":cfg["deadline_utc"],
                                                "updated_utc":utc(time.time())})
                event("start",trial=directory.name)
                with (out/(directory.name+".log")).open("a") as log:
                    cmd = [sys.executable,str(HERE/"trial.py"),"--arm",arm,"--out",str(directory),
                           "--data",str(args.data),"--seed",str(seed),"--seconds",str(budget)]
                    proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    try:
                        code = proc.wait(timeout=max(.01, deadline-time.time()))
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try: proc.wait(timeout=3)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                        event("deadline",trial=directory.name)
                        break
                if code == 0 and (directory/"metrics.json").is_file():
                    event("complete",trial=directory.name)
                    break
                event("failed",trial=directory.name,returncode=code)
                if attempt == 3:
                    raise RuntimeError(f"three unsuccessful attempts: {name}")
                time.sleep(max(0,min(15,deadline-time.time())))
        atomic_json(out/"status.json", {"status":"completed","deadline_utc":cfg["deadline_utc"],
                                        "finished_utc":utc(time.time())})
        event("finished")
    except BaseException as exc:
        atomic_json(out/"status.json",{"status":"failed","reason":str(exc),"updated_utc":utc(time.time())})
        event("error",reason=str(exc))
        raise
    finally:
        stop.set()
        monitor.join(timeout=12)
        events.close()
        lock.close()


if __name__ == "__main__":
    main()
