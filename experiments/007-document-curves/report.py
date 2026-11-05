"""Audit the completed seeds and render the document learning curves, without weights or a GPU."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from learning_curves import audit_curves, read, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory or a published results directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Document corpus", help="plot title")
    args = parser.parse_args()
    protocol = read(args.run / "protocol.json")
    curves = audit_curves(args.run, args.run / "manifest.json", protocol["seeds"],
                         protocol["epochs"], protocol["device"])
    write_report(curves, args.out, title=args.title, unit="documents")


if __name__ == "__main__":
    main()
