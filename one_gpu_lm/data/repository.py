"""Build private assistant-only supervision from training repositories, never evaluation traces."""
import argparse
from collections import Counter
import difflib
import json
from pathlib import Path

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.actions import SYSTEM, parse_action
from one_gpu_lm.evaluation.environment import RepositoryEnvironment
from one_gpu_lm.evaluation.grading import generated_files, grade
from one_gpu_lm.evaluation.tasks import file_manifest, prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root','tasks','campaign','out']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Preserve dataset attempts')
    args.out.mkdir(parents=True)
    root = args.root.resolve()
    tasks = {t['id']:t for t in json.loads(args.tasks.read_text())['tasks']}
    campaign = json.loads((args.campaign/'campaign.json').read_text())
    if not campaign['complete']:
        raise ValueError('Baseline campaign must be complete')
    runtime = json.loads((root/'runtime.json').read_text())
    rows, reference_rows, references = [], [], []
    for result in sorted(campaign['results'], key=lambda r:r['id']):
        task = tasks[result['id']]
        if task['split'] != 'train':
            continue
        provenance = {'repository':task['repository'], 'revision':task['revision'], 'split':'train',
                      'task_id':task['id'], 'source_license_record':'experiments/012-repository-agent/sources.json'}
        if result['passed']:
            trajectory = json.loads((args.campaign/'private'/task['id']/'trajectory.json').read_text())['messages']
            for index, message in enumerate(trajectory):
                if message['role'] != 'assistant':
                    continue
                try:
                    parse_action(message['content'])
                except ValueError:
                    continue
                # Supervise only a command whose observed execution succeeded; prefix may show recovered errors.
                following = trajectory[index+1] if index+1 < len(trajectory) else {}
                if following.get('extra',{}).get('tool',{}).get('returncode') != 0:
                    continue
                prefix = [{k:m[k] for k in ['role','content']} for m in trajectory[:index] if m['role'] != 'exit']
                rows.append({'id':task['id']+f':turn-{index:03d}', 'messages':prefix, 'target':message['content'],
                             'provenance':{**provenance, 'kind':'verified-local-base-trajectory'}})
            continue
        slug = task['repository'].replace('/','--')
        snapshot = root/'sources'/slug
        workspace = prepare(snapshot, args.out/'private'/task['id'], task)
        original = file_manifest(workspace, exclude=generated_files(task['repository']))
        patch, context = [], []
        for name in task['editable_files']:
            broken = (workspace/name).read_text().splitlines(keepends=True)
            good = (snapshot/name).read_text().splitlines(keepends=True)
            patch.extend(difflib.unified_diff(broken,good,fromfile='a/'+name,tofile='b/'+name,n=3))
            excerpts = []
            for tag,_,_,j1,j2 in difflib.SequenceMatcher(a=good,b=broken).get_opcodes():
                if tag != 'equal':
                    excerpts.extend(broken[max(0,j1-5):min(len(broken),j2+5)])
            context.append(name+' (broken excerpt):\n'+''.join(excerpts))
        command = "patch -p1 --batch <<'END_REFERENCE_PATCH'\n"+''.join(patch)+'END_REFERENCE_PATCH'
        deps = root/'node-deps'/slug/'node_modules'
        def make_env(work):
            return RepositoryEnvironment(work,python_site=root/'.venv-agent/lib/python3.10/site-packages',
                node_root=runtime['node_root'],node_modules=deps if deps.exists() else None,output_chars=100000)
        applied = make_env(workspace).execute({'command':command})
        score = grade(task,workspace,snapshot,args.out/'private'/(task['id']+'-grade'),original,make_env)
        references.append({'id':task['id'],'patch_applied':applied['returncode']==0,'passed':score['passed']})
        if applied['returncode'] != 0 or not score['passed']:
            raise ValueError('Reference repair demonstration failed independent grading')
        prompt = ('Task: '+task['description']+'\nAllowed source files: '+', '.join(task['editable_files'])+
                  '\nRegression command: '+task['regression_command']+'\n'+('\n'.join(context))[:4000])
        reference_rows.append({'id':task['id']+':reference', 'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
                     'target':'```bash\n'+command+'\n```','provenance':{**provenance,'kind':'verified-reference-repair'}})
    rows.extend(reference_rows)
    if not rows or any(row['provenance']['split'] != 'train' for row in rows):
        raise ValueError('Invalid training split')
    path = args.out/'train.jsonl'
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    write_json(args.out/'receipt.json', {'schema_version':1,'raw_data_public':False,'records':len(rows),
        'sha256':sha256(path),'source_campaign_sha256':sha256(args.campaign/'campaign.json'),
        'task_spec_sha256':sha256(args.tasks),'kinds':dict(Counter(row['provenance']['kind'] for row in rows)),
        'repositories':sorted({row['provenance']['repository'] for row in rows}),
        'task_ids':sorted({row['provenance']['task_id'] for row in rows}), 'reference_controls':references,
        'license_record_sha256':sha256(root/'experiments/012-repository-agent/sources.json')})
    print('Verified private training records:',len(rows),flush=True)


if __name__ == '__main__':
    main()
