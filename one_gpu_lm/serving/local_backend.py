"""A private Unix-socket service around the model adapter."""
import argparse
import json
import os
from pathlib import Path
import socketserver
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.models.hf_runtime import Runtime
from .client import MAX_REQUEST_BYTES


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
