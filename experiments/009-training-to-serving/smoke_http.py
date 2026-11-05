"""Start a loopback server, send real requests, record the receipt and release the device."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("choose a new output path")
    cmd = [sys.executable, str(Path(__file__).with_name("serve.py")), "--bundle", str(args.bundle),
           "--port", "0", "--device", args.device]
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    try:
        url = "http://127.0.0.1:" + str(json.loads(proc.stdout.readline())["port"])
        with urlopen(url + "/health", timeout=10) as resp:
            health = json.load(resp)
        ready = time.monotonic() - start
        def generate(count):
            body = json.dumps({"prompt": "计算属性", "max_new_tokens": count}).encode()
            req = Request(url + "/generate", data=body, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=60) as resp:
                return resp.status, json.load(resp)
        start = time.monotonic()
        status, generation = generate(8)
        elapsed = time.monotonic()-start
        if status != 200 or generation["generated_tokens"] != len(generation["generated_ids"]) or len(generation["generated_ids"]) != 8:
            raise RuntimeError("Invalid completion")
        try:
            generate(health["context"]+1)
        except HTTPError as e:
            if e.code != 400:
                raise
        else:
            raise RuntimeError("Invalid context accepted")
        args.out.parent.mkdir(exist_ok=True, parents=True)
        receipt = {"health": health, "completion_http": status, "overlength_http": 400,
                   "startup_seconds": ready, "first_request_seconds": elapsed,
                   "generated_tokens": 8, "output_ids_sha256": hashlib.sha256(json.dumps(generation["generated_ids"]).encode()).hexdigest()}
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()
