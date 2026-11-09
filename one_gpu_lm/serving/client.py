"""Bounded request protocol for the private local model socket."""
import json
import socket


MAX_REQUEST_BYTES = 2**21

def request(path, value, timeout=180):
    payload = json.dumps(value).encode() + b"\n"
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Backend request exceeds byte limit")
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(payload)
        with client.makefile("rb") as stream:
            response = stream.readline(MAX_REQUEST_BYTES + 1)
    if len(response) > MAX_REQUEST_BYTES or not response.endswith(b"\n"):
        raise ValueError("Invalid backend response")
    result = json.loads(response)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result
