"""Re-run the prior simple coding fixtures through the same local model backend."""
import argparse
import json
from pathlib import Path

from one_gpu_lm.data.function_records import load_split
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.evaluation.sandbox import Sandbox, extract_code
from one_gpu_lm.serving.client import request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['tasks','socket','node','out']:
        p.add_argument('--'+name,required=True,type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Preserve previous regression result')
    args.out.mkdir(parents=True)
    private = args.out/'private'
    private.mkdir()
    rows = load_split(args.tasks,'test')
    sandbox = Sandbox(args.node)
    value = {'schema_version':1,'task_sha256':sha256(args.tasks),'complete':False,'results':[]}
    for task in rows:
        generated = request(args.socket,{'operation':'generate','messages':[{'role':'user','content':task['prompt']}],
            'max_new_tokens':512,'max_input_tokens':8192})
        (private/(task['id']+'.txt')).write_text(generated['text'])
        score = sandbox.score(task,extract_code(generated['text']))
        value['results'].append({'id':task['id'],'language':task['language'],**score,
            **{k:v for k,v in generated.items() if k not in {'text','ids'}}})
        write_json(args.out/'result.json',value)
        print(task['id'],score['passed'],flush=True)
    value['complete']=True
    value['passed']=sum(r['passed'] for r in value['results'])
    write_json(args.out/'result.json',value)


if __name__ == '__main__':
    main()
