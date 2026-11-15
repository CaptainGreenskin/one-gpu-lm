"""Allowlisted software versions only; no environment, account, path or token dump."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import platform

PACKAGES=('torch','transformers','peft','tokenizers','accelerate','safetensors')


def capture():
    return {'python':platform.python_version(),'packages':{name:metadata.version(name) for name in PACKAGES}}


def verify(expected):
    actual=capture()
    if actual!=expected:
        raise ValueError('Registered model-interpreter package versions changed')
    return actual


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expected',type=Path)
    a=p.parse_args();print(json.dumps(verify(json.loads(a.expected.read_text())) if a.expected else capture(),sort_keys=True))
