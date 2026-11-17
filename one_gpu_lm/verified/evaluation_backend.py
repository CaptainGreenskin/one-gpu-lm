"""Check the model interpreter before every registered weight load."""
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
    options=argparse.ArgumentParser(add_help=False)
    options.add_argument('--profile',type=Path,required=True)
    variant,_=options.parse_known_args(remaining)
    profile=json.loads(variant.profile.read_text())
    if profile.get('generation_modes'):
        from one_gpu_lm.serving.reasoning_backend import main as serve
    else:
        from one_gpu_lm.serving.local_backend import main as serve
    sys.argv=[sys.argv[0],*remaining];serve()


if __name__=='__main__':main()
