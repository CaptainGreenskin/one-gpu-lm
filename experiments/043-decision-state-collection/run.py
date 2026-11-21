"""Check, collect or audit the registered 043 training decision pairs."""
import argparse
import json
import os
from pathlib import Path

from one_gpu_lm.verified.decision_state_collection import load_cases, read, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', 'audit'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    root, protocol = args.root.resolve(), args.protocol.resolve()
    if args.action == 'check':
        _, _, _, cases = load_cases(root, read(protocol))
        result = {'cases': len(cases), 'paired_inputs': 2 * len(cases), 'final_test_opened': False}
    elif args.action == 'run':
        state = run(root, protocol)
        result = {'complete': state['complete'], 'attempts': len(state['runs'])}
    else:
        from one_gpu_lm.verified.decision_state_audit import audit

        receipt = audit(root, protocol)
        result = {key: receipt[key] for key in ('passed', 'audited_attempts', 'journal_requests_verified',
                                                'eligible_contrasts', 'final_test_opened')}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
