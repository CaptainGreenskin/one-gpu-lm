"""Count exact regression targets in the audited 032 training inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import tarfile

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.receipts import sha256, write_json
from .first_edit_inputs import digest, read


INPUTS = {
    'private/mixed031-export/data.jsonl':'retained_031',
    'runs/historic032-expert-001/private/export/data.jsonl':'new_reference_guided',
}


def review(archive,expected_archive,training_protocol):
    if sha256(archive) != expected_archive:
        raise ValueError('Verified local archive identity changed')
    protocol = read(training_protocol)
    groups = {}
    all_ids = set()
    with tarfile.open(archive,'r|gz') as stream:
        for entry in stream:
            if entry.name not in INPUTS:
                continue
            name = INPUTS[entry.name]
            if name in groups or not entry.isfile() or entry.size > 32*2**20:
                raise ValueError('Unexpected training receipt artifact')
            raw = stream.extractfile(entry).read()
            checksum = digest(raw)
            if checksum != protocol['frozen_files'][entry.name]:
                raise ValueError('Input differs from the audited training export')
            counts = Counter()
            kinds = Counter()
            for line in raw.splitlines():
                row = json.loads(line)
                p = row['provenance']
                if p['split'] != 'train' or p['executed'] is not True or row['id'] in all_ids:
                    raise ValueError('Training scope or record identity differs')
                all_ids.add(row['id'])
                declared = re.findall(r'^Regression command: (.+)$',row['messages'][1]['content'],re.M)
                if len(declared) != 1:
                    raise ValueError('Cannot identify one declared visible check')
                exact = parse_action(row['target'])['command'] == declared[0]
                initial = p['call'] == 1 and [m['role'] for m in row['messages']] == ['system','user']
                counts.update(records=1,regression_target_records=exact,
                              initial_targets=initial,initial_exact_regression_targets=initial and exact,
                              supervised_tokens=row['encoding_receipt']['supervised_tokens'])
                kinds[p['kind']] += 1
            groups[name] = {'source_sha256':checksum,**counts,'registered_action_kinds':dict(kinds)}
    if set(groups) != set(INPUTS.values()) or len(all_ids) != 1238:
        raise ValueError('Expected all retained and all reference-guided records once')
    return {'schema_version':1,'source_training_protocol_sha256':sha256(training_protocol),
            'archive_sha256':expected_archive,'groups':groups,'records':len(all_ids),
            'initial_target_definition':'first physical call with only system and user messages',
            'compare_declared_visible_command_by_exact_text':True,'new_model_generations':0,
            'new_tool_executions':0,'new_optimizer_updates':0,'final_test_opened':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--archive-sha256',required=True)
    parser.add_argument('--training-protocol',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve the first training target review')
    result=review(args.archive,args.archive_sha256,args.training_protocol)
    write_json(args.out,result)
    print(json.dumps(result['groups']))
