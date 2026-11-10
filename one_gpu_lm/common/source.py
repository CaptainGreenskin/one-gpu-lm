"""Hash executable implementations, including files behind compatibility entry points."""
from pathlib import Path

from .receipts import sha256


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_FILES = (
    'common/actions.py', 'common/agent_policy.py', 'common/tool_history.py', 'common/receipts.py', 'common/source.py',
    'models/hf_runtime.py', 'serving/client.py', 'serving/local_backend.py',
    'evaluation/campaign.py', 'evaluation/agent.py', 'evaluation/environment.py',
    'evaluation/grading.py', 'evaluation/tasks.py', 'evaluation/sandbox.py',
    'evaluation/ava.config.mjs', 'evaluation/underscore_tests.mjs',
)


def implementation_hashes(names):
    files = set(names)
    files.add('__init__.py')
    for name in names:
        for parent in Path(name).parents:
            if str(parent) != '.':
                files.add(str(parent / '__init__.py'))
    return {'one_gpu_lm/' + name: sha256(PACKAGE_ROOT / name) for name in sorted(files)}


def campaign_sources():
    return implementation_hashes(CAMPAIGN_FILES)


def historical_pilot_dependencies():
    return implementation_hashes(('common/receipts.py', 'common/source.py',
                                 'data/function_records.py', 'evaluation/sandbox.py'))
