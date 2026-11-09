"""Fetch the pinned official checkpoint and verify every weight shard locally."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, entry):
    if not path.exists() or path.stat().st_size != entry["size"]:
        return False
    if entry.get("sha256"):
        return digest(path) == entry["sha256"]
    blob = path.read_bytes()
    return hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest() == entry["git_blob_sha1"]


def modelscope_download(model_id, entry, out):
    """Alternate official transport; pinned HF content hashes remain authoritative."""
    import httpx
    path = out / entry["name"]
    if verify(path, entry):
        return
    tmp = path.with_suffix(path.suffix + ".part")
    url = f"https://modelscope.cn/models/{model_id}/resolve/master/{entry['name']}"
    for attempt in range(3):
        offset = tmp.stat().st_size if tmp.exists() else 0
        if offset == entry["size"]:
            break
        if offset > entry["size"]:
            raise ValueError("Oversized partial download")
        try:
            with httpx.Client(trust_env=False, follow_redirects=True, timeout=60) as client:
                with client.stream("GET", url, headers={"Range": f"bytes={offset}-"} if offset else {}) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        raise ValueError("Source did not honor resume Range")
                    if offset and not response.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected resume range")
                    with tmp.open("ab" if offset else "wb") as stream:
                        for chunk in response.iter_bytes(1024 * 1024):
                            stream.write(chunk)
            break
        except httpx.TransportError:
            if attempt == 2:
                raise
            time.sleep(2)
    if not verify(tmp, entry):
        raise ValueError(f"Official mirror differs from pinned bytes: {entry['name']}")
    tmp.replace(path)
    print("Verified", entry["name"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    args = parser.parse_args()
    weights = json.loads((HERE / "weights-manifest.json").read_bytes())
    model = json.loads((HERE / "candidates.json").read_bytes())["models"][0]
    assert weights["model_id"] == model["id"] and weights["revision"] == model["revision"]
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.source == "huggingface":
        snapshot_download(model["id"], revision=model["revision"], local_dir=args.out,
                          allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "LICENSE", "README.md"],
                          max_workers=4)
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda entry: modelscope_download(model["id"], entry, args.out),
                          weights["metadata_files"] + weights["files"]))
    verified = []
    for f in weights["files"]:
        path = args.out / f["name"]
        if path.stat().st_size != f["size"] or digest(path) != f["sha256"]:
            raise ValueError(f"Weight integrity mismatch: {f['name']}")
        verified.append(f)
        print("Verified", f["name"], flush=True)
    for name, evidence in model["evidence"].items():
        if digest(args.out / name) != evidence["sha256"]:
            raise ValueError(f"Metadata integrity mismatch: {name}")
    for entry in weights["metadata_files"]:
        if not verify(args.out / entry["name"], entry):
            raise ValueError(f"Metadata integrity mismatch: {entry['name']}")
    receipt = dict(model_id=model["id"], revision=model["revision"], transport=args.source, weight_files=verified,
                   verified_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - started)
    (args.out / "download-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Pinned weights and metadata verified", flush=True)


if __name__ == "__main__":
    main()
