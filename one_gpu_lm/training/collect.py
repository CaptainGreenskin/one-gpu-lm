"""Collect an explicit public evidence bundle, excluding paths, weights, states and raw records."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from one_gpu_lm.common.receipts import sha256, write_json


ROOT_FILES={'identity.json','profile.json','selection.json','metrics.json','result.json','status.json',
            'attempts.json','events.jsonl','telemetry.csv','latest.json'}


def collect(run,out):
    run,out=Path(run),Path(out)
    if out.exists():raise ValueError('Public output already exists')
    result=json.loads((run/'result.json').read_text())
    if result['status'] not in {'completed','paused','failed'}:raise ValueError('Run has no final attempt receipt')
    files=[run/name for name in sorted(ROOT_FILES) if (run/name).is_file()]
    files += sorted((run/'preflight').glob('attempt-*.json'))+sorted((run/'loading').glob('attempt-*.json'))
    for file in files:
        if file.is_symlink():raise ValueError('Public artifact must be a regular owned file')
        content=file.read_text()
        if any(s in content for s in ['/home/','/Users/','Authorization:','api_key','password','BEGIN PRIVATE KEY']):
            raise ValueError('Review a private-looking value before publishing '+file.name)
    out.mkdir(parents=True)
    for file in files:
        destination=out/file.relative_to(run);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(file,destination)
    write_json(out/'artifacts.json',{'schema_version':1,'weights_public':False,'optimizer_states_public':False,
        'raw_data_public':False,'files':{str(p.relative_to(run)):sha256(p) for p in files}})
    return len(files)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();print('Collected public artifacts:',collect(args.run,args.out))


if __name__=='__main__':main()
