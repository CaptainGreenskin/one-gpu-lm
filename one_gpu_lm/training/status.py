"""Read public-safe progress without attaching to a process or loading a model."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    value={}
    for name in ['status','result','latest']:
        path=args.run/(name+'.json')
        if path.exists():value[name]=json.loads(path.read_text())
    if not value:raise ValueError('No run records found')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
