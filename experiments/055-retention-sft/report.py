"""Render the independently checked 055 training measurements."""

import argparse
import json
from pathlib import Path
import runpy


EXPERIMENT = Path(__file__).resolve().parent
PARENT_REPORT = EXPERIMENT.parent / '052-grounded-first-edit/report.py'


def render(directory, registration=EXPERIMENT / 'training-registration.json'):
    expected = json.loads(registration.read_bytes())['learning_rate']
    updates = json.loads((directory / 'measurements.json').read_bytes())['updates']
    if not updates or any(update['learning_rate'] != expected for update in updates):
        raise ValueError('The executed updates must match the registered learning rate')
    return runpy.run_path(str(PARENT_REPORT))['render'](
        directory, registration, title='055 · 降低学习率微调结果')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--registration', type=Path, default=EXPERIMENT / 'training-registration.json')
    args = parser.parse_args()
    print(json.dumps(render(args.directory, args.registration), allow_nan=False))
