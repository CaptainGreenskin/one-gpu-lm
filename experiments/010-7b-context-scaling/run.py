"""Run 48 bounded, isolated trials; retain failures and hardware telemetry."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def matrix(protocol):
    for repeat in range(protocol["repeats"]):
        modes = protocol["modes"][::(-1 if repeat % 2 else 1)]
        for sequence in protocol["sequences"]:
            for mode in modes:
                for ac in protocol["checkpoint"][::(-1 if repeat % 2 else 1)]:
                    name = f"{mode}-s{sequence}-ac{int(ac)}-r{repeat}"
                    yield name, mode, sequence, ac, repeat


def sample(stop, path):
    fields = ["memory.used", "utilization.gpu", "temperature.gpu", "power.draw", "clocks.sm"]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["utc", "memory_used_mib", "utilization_percent", "temperature_c", "power_w", "sm_clock_mhz"])
        while not stop.is_set():
            try:
                gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                result = subprocess.run(["nvidia-smi", "-i", gpu_index, "--query-gpu=" + ",".join(fields),
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    writer.writerow([now(), *next(csv.reader(result.stdout.splitlines()))])
                    stream.flush()
            except (subprocess.TimeoutExpired, FileNotFoundError, StopIteration):
                pass
            stop.wait(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    lock = (args.out / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    (args.out / "protocol.json").write_bytes(protocol_bytes)
    sources = [*HERE.glob("*.py"), HERE / "protocol.json", HERE.parent / "002-first-train/train.py"]
    save(args.out / "sources.json", {str(p.relative_to(HERE.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(sources)})
    started = time.monotonic()
    campaign = dict(started_utc=now(), planned_trials=len(list(matrix(protocol))), records=[], state="running")
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=5)
    campaign["driver_version"] = driver.stdout.strip() if driver.returncode == 0 else "unavailable"
    save(args.out / "campaign.json", campaign)
    stop = threading.Event()
    telemetry = threading.Thread(target=sample, args=(stop, args.out / "telemetry.csv"), daemon=True)
    telemetry.start()
    env = dict(os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    try:
        for name, mode, sequence, ac, repeat in matrix(protocol):
            remaining = protocol["campaign_timeout_seconds"] - (time.monotonic() - started)
            if remaining < protocol["trial_timeout_seconds"]:
                campaign["state"] = "budget_exhausted"
                break
            result_path = args.out / (name + ".json")
            save(args.out / "status.json", dict(state="running", current=name, completed=len(campaign["records"])))
            command = [sys.executable, str(HERE / "bench.py"), "--mode", mode, "--sequence", str(sequence),
                       "--repeat", str(repeat), "--out", str(result_path)] + (["--checkpoint"] if ac else [])
            trial_start = time.monotonic()
            with (args.out / (name + ".log")).open("w") as log:
                try:
                    process = subprocess.run(command, env=env, stdout=log, stderr=log,
                                             timeout=protocol["trial_timeout_seconds"])
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    returncode = 124
            wall = time.monotonic() - trial_start
            if not result_path.exists():
                save(result_path, dict(status="timeout" if returncode == 124 else "worker_failed",
                                       mode=mode, sequence=sequence, checkpoint=ac, repeat=repeat, records=[]))
            record = json.loads(result_path.read_bytes())
            record.update(process_wall_seconds=wall, returncode=returncode)
            save(result_path, record)
            campaign["records"].append(dict(name=name, status=record["status"], process_wall_seconds=wall))
            save(args.out / "campaign.json", campaign)
            if returncode != 0:
                campaign["state"] = "worker_error"
                break
        else:
            campaign["state"] = "completed"
    finally:
        stop.set()
        telemetry.join(timeout=6)
        campaign.update(finished_utc=now(), wall_seconds=time.monotonic() - started)
        save(args.out / "campaign.json", campaign)
        save(args.out / "status.json", dict(state=campaign["state"], completed=len(campaign["records"])))
    print(json.dumps({k: v for k, v in campaign.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
