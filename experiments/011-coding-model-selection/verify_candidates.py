"""Verify pinned model metadata, without downloading weights or executing model code."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    manifest = json.loads(Path(__file__).with_name("candidates.json").read_bytes())
    for model in manifest["models"]:
        for name, evidence in model["evidence"].items():
            expected_url = f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/{name}"
            if evidence["url"] != expected_url or name not in ("LICENSE", "README.md", "config.json"):
                raise ValueError("Unexpected metadata location")
            with urlopen(Request(expected_url, headers={"User-Agent": "one-gpu-lm-metadata-review"}), timeout=30) as stream:
                data = stream.read(2_000_001)
            if len(data) > 2_000_000 or hashlib.sha256(data).hexdigest() != evidence["sha256"]:
                raise ValueError(f"Metadata changed or oversized: {model['id']} / {name}")
        print(f"Verified metadata: {model['id']} @ {model['revision']}")
    print("Metadata integrity verified; this is not a legal clearance or a model quality evaluation.")


if __name__ == "__main__":
    main()
