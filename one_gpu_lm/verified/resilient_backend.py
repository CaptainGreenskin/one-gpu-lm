"""Verify the frozen interpreter and serve generation with durable receipts."""
import argparse
import json
from pathlib import Path
import sys
from one_gpu_lm.common.receipts import write_json
from .runtime_identity import verify


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--expected-environment',type=Path,required=True)
    p.add_argument('--environment-receipt',type=Path,required=True)
    args,remaining=p.parse_known_args()
    if args.environment_receipt.exists():raise ValueError('Preserve previous backend environment receipt')
    actual=verify(json.loads(args.expected_environment.read_text()))
    write_json(args.environment_receipt,actual)
    from one_gpu_lm.serving.journaled_backend import main as serve
    sys.argv=[sys.argv[0],*remaining];serve()


if __name__=='__main__':main()
