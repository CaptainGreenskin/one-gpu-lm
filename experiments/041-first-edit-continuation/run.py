"""Inspect, run or audit the registered first-edit continuation."""
import argparse
import json
import os
from pathlib import Path

from one_gpu_lm.verified.first_edit_continuation import load_cases, read, run
from one_gpu_lm.verified.first_edit_continuation_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', 'audit'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.action == 'check':
        _, _, cases, _ = load_cases(args.root.resolve(), read(args.protocol))
        result = {'case_count': len(cases), 'paired_inputs': 2 * len(cases)}
    elif args.action == 'run':
        state = run(args.root, args.protocol)
        result = {'complete': state['complete'], 'attempts': len(state['runs'])}
    else:
        state = audit(args.root, args.protocol)
        result = {name: state[name] for name in ('passed', 'journal_requests_verified', 'collection_gate_passed')}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
