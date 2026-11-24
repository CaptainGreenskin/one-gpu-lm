"""Reproduce the post-edit recheck training report from its public measurements."""

import argparse
import json
from pathlib import Path
import runpy


EXPERIMENT = Path(__file__).resolve().parent
PARENT_REPORT = EXPERIMENT.parent / '052-grounded-first-edit/report.py'


def render(directory, registration=EXPERIMENT/'training-registration.json'):
    return runpy.run_path(str(PARENT_REPORT))['render'](
        directory, registration, title='053 · 失败回归与纠错微调结果')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--registration', type=Path, default=EXPERIMENT/'training-registration.json')
    args = parser.parse_args()
    print(json.dumps(render(args.directory, args.registration), allow_nan=False))
