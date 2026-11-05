"""Single-process loopback text-completion service for a local One GPU LM bundle."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path

from runtime import Runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    runtime = Runtime.load(args.bundle, args.device)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, obj):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ready", "device": args.device, "context": runtime.context})
            else:
                self.reply(404, {"error": "unknown route"})

        def do_POST(self):
            if self.path != "/generate":
                return self.reply(404, {"error": "unknown route"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 65536:
                    raise ValueError("invalid body size")
                body = json.loads(self.rfile.read(n))
                if not isinstance(body, dict) or not isinstance(body.get("prompt"), str):
                    raise ValueError("prompt must be a string")
                result = runtime.generate(body["prompt"], body.get("max_new_tokens", 32))
            except (ValueError, UnicodeError) as e:
                return self.reply(400, {"error": str(e)})
            self.reply(200, result)
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(json.dumps({"port": server.server_port}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
