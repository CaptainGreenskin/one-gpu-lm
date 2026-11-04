"""Record three Chinese BPE learning curves on the chapter split from experiment 005."""
import argparse
from pathlib import Path
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRIAL = HERE.parent / "004-tokenizer-soak"
sys.path.insert(0, str(TRIAL))
from trial import digest

SEEDS = (42, 43, 44)
EPOCHS = 24


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/rust-zh-pairs")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/006-bpe-epoch-curve")
    args = parser.parse_args()
    reference = HERE.parent / "005-technical-bytes/prepared-reference.json"
    if (args.data / "manifest.json").read_bytes() != reference.read_bytes():
        raise ValueError("Expected the frozen experiment 005 chapter split")
    manifest = json.loads(reference.read_bytes())
    for split, meta in manifest["splits"].items():
        for arm in ("byte", "bpe"):
            if digest(args.data / f"{split}-{arm}.bin") != meta[f"{arm}_sha256"]:
                raise ValueError(f"Corrupt {split}-{arm} data")
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "environment.json").open("w") as stream:
        subprocess.run([sys.executable, str(ROOT / "scripts/check_gpu.py")], stdout=stream, check=True)
    for seed in SEEDS:
        subprocess.run([sys.executable, str(TRIAL / "trial.py"), "--arm", "bpe", "--seed", str(seed),
                        "--epochs", str(EPOCHS), "--epoch-curve", "--data", str(args.data),
                        "--out", str(args.out / f"seed-{seed}")], check=True)


if __name__ == "__main__":
    main()
