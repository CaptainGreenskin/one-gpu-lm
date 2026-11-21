"""Assemble a ModelScope payload after the registered development and final gates."""
import argparse
import json
from pathlib import Path
import re
import tempfile

from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import atomic_json
from one_gpu_lm.models.adapter_bundle import FILES, restore_adapter
from one_gpu_lm.models.modelscope_readback import file_path, model_endpoint, validate_files
from one_gpu_lm.models.release_lineage import reviewed_training
from one_gpu_lm.verified.reload_probe import compare

from final_audit import require
from final_campaign import ARMS, FINAL, promotion_from_finished_run
from promotion import REFERENCE, read, release_decision


ENDPOINT = 'https://www.modelscope.ai'
OWNER = 'CaptainGreenskin'
COHORTS = ('fresh_final_not_selection', 'historical_final_regression_not_selection')
PROFILE = REFERENCE/'profile-action.json'
PROFILE_KEY = 'experiments/022-action-repair/profile-action.json'
BASE_MANIFEST = 'experiments/011-coding-model-selection/weights-manifest.json'
TRAINING = REFERENCE.parent/'031-correction-mix'
BASE_LICENSE = 'BASE-MODEL-LICENSE.txt'


def audited_scores(out, permission):
    control = out/'controller.json'
    audit_path = out/'execution-audit.json'
    promotion_path = out/'promotion.json'
    state, audit = read(control), read(audit_path)
    require(read(promotion_path) == permission and state['complete'] is True
            and state['stage'] == 'scored_audit_pending' and state['final_test_opened'] is True
            and state['weights_public'] is False and state['optimizer_updates'] == 0
            and audit['execution_audit_passed'] is True and audit['model_generations'] == 0
            and audit['final_test_opened'] is True and audit['weights_public'] is False
            and audit['controller_sha256'] == sha256(control)
            and audit['promotion_sha256'] == sha256(promotion_path)
            and audit['adapter_files_sha256'] == permission['checkpoint_sha256']
            and audit['audit_source_sha256'] == sha256(Path(__file__).with_name('final_audit.py')),
            'Require completed final replay for the approved development checkpoint')

    inventory = read(REFERENCE/'results/2026-11-12/task-inventory.json')
    roles = {row['id']:row['evaluation_role'] for row in inventory if row.get('evaluation_role') in COHORTS}
    paired = {role:[] for role in COHORTS}
    legacy_passes = []
    for arm in ARMS:
        campaign = out/arm/'campaign.json'
        artifact = out/('artifact-'+arm+'.json')
        legacy = out/('legacy-'+arm)/'result.json'
        legacy_audit = audit['legacy_replay'][arm]
        require(audit['artifacts'][arm] == {'campaign_sha256':sha256(campaign),
                                           'served_artifact_sha256':sha256(artifact)}
                and legacy_audit['result_sha256'] == sha256(legacy)
                and legacy_audit['scores_replayed'] == permission['cohorts']['legacy'],
                'Scored artifacts differ from the independent replay')
        scores, functions = read(campaign), read(legacy)
        replay = audit['repository_replay'][arm]
        results = scores['results']
        require(scores['complete'] is True and len(results) == len(replay) == len(roles)
                and {row['id'] for row in results} == roles.keys()
                and all(a['id'] == b['id'] and a['passed'] is b['passed'] for a,b in zip(results,replay))
                and functions['complete'] is True and len(functions['results']) == permission['cohorts']['legacy'],
                'Require all registered repository and function pairs to be replayed')
        for role in COHORTS:
            paired[role].append({'complete':True,'results':[row for row in results if roles[row['id']] == role]})
        legacy_passes.append(functions['passed'])

    pairs = [scores for role in COHORTS for scores in paired[role]]
    checks = {'legacy_passes':legacy_passes[1],'all_execution_audits_pass':True}
    decision = release_decision(permission,*pairs,checks)
    require(audit['checks'] == checks and audit['release_decision_with_execution_evidence'] == decision,
            'Final release evidence differs from recomputation')
    outstanding = {'save_reload_greedy_match','rights_and_secret_scan_pass',
                   'single_training_seed_and_repository_cluster_limits_disclosed'}
    require(set(decision['failed_gates']) <= outstanding,
            'Final acceptance failed: '+', '.join(sorted(set(decision['failed_gates'])-outstanding)))
    table = []
    for label,role in zip(('新留出','历史回归'),COHORTS):
        base,candidate = (score['results'] for score in paired[role])
        table.append((label,len(base),sum(row['passed'] for row in base),sum(row['passed'] for row in candidate)))
    table.append(('函数回归',permission['cohorts']['legacy'],*legacy_passes))
    return pairs,checks,table


def approved_notices(permission, input_review_path, rights_path, readback_path):
    review, rights, readback = (read(path) for path in (input_review_path,rights_path,readback_path))
    identity = {'model_id':permission['model_id'],'base_revision':permission['base_revision']}
    require(rights['schema_version'] == 1 and rights['approved_for_weight_publication'] is True
            and rights['rights_and_secret_scan_pass'] is True and rights['publication_basis'].strip()
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+-]*',rights['release_license'])
            and all(rights[key] == readback[key] == value for key,value in identity.items())
            and rights['adapter_files_sha256'] == permission['checkpoint_sha256']
            and rights['input_review_sha256'] == sha256(input_review_path) == permission['candidate_review_sha256']
            and rights['notice_readback_sha256'] == sha256(readback_path)
            and review['known_pattern_findings'] and all(n == 0 for n in review['known_pattern_findings'].values()),
            'Require a distribution and sensitive-content review for the exact checkpoint and source notices')
    if 'source_revisions' in readback:
        require(readback['source_revisions'] == {repo: source['revision'] for repo,source in review['sources'].items()},
                'The notice readback covers different training-source revisions')
    expected = {BASE_LICENSE:readback['notices'][BASE_LICENSE]['source_sha256']}
    for repo,source in review['sources'].items():
        for name,digest in source['notices'].items():
            expected[f'training-source-notices/{repo.replace("/", "--")}/{name}'] = digest
    require(set(expected) == readback['notices'].keys()
            and all(readback['notices'][name]['byte_equal'] is True
                    and readback['notices'][name]['source_sha256'] == digest for name,digest in expected.items()),
            'Require exact official readback of the base license and every training-source notice')
    return rights['release_license'],expected


def model_card(repo_id, permission, evaluation, training, software, metrics, license_id, *, lineage=None):
    model, revision = permission['model_id'], permission['base_revision']
    packages = software['packages']
    require(all(re.fullmatch(r'[A-Za-z0-9.+!-]+',v) for v in [software['python'],*packages.values()]),
            'Expected package versions in the verified loading environment')
    rows = '\n'.join(f'| {label} | {count} | {base} | {selected} |' for label,count,base,selected in metrics)
    dependencies = ' '.join(f'`{name}=={value}`' for name,value in packages.items())
    adapter_hashes = '\n'.join(f'| `{name}` | `{permission["checkpoint_sha256"][name]}` |' for name in FILES)
    if lineage is None:
        experiment = '031'
        scope = '039 在六个仓库家族的 60 道开发题上复跑三轮。'
        source = (f'训练由本地底座生成、实际执行并审计的轨迹组成。适配器分发适用 `{license_id}`。'
                  '底座与训练来源的许可及署名原文随包保留。'
                  f'[训练来源核验](https://github.com/{OWNER}/one-gpu-lm/blob/main/experiments/031-correction-mix/release-provenance.md)')
    else:
        experiment = lineage['experiment']
        scope = ''
        source = (f'候选的训练记录与输入来源已核验，共执行 {lineage["optimizer_updates"]} 次优化器更新。'
                  f'适配器分发适用 `{license_id}`。底座与训练来源的许可及署名原文随包保留。')
    return f'''---
license: {license_id.lower()}
---

# {repo_id.split('/')[1]}

这是基于 `{model}` 的编码修复实验适配器。它以未合并的 PEFT LoRA 发布，秩为 {training['lora']['r']}，加载时需要固定底座 `{revision}`。

| 适配器文件 | SHA-256 |
| --- | --- |
{adapter_hashes}

## 验收结果

{scope}开发集采用相同的推理流程和预算，LoRA 的完成率提高 {permission['development_gain_pp']:+.2f} 个百分点。

| 最终测试 | 题数 | 底座通过 | LoRA 通过 |
| --- | ---: | ---: | ---: |
{rows}

同一题内，两组使用相同的工具流程和预算。{experiment} 使用一个训练种子 `{training['training']['seed']}`。仓库内任务共享代码和依赖，结果受家族分布影响。结论限于这些仓库和函数测试，长任务能力需单独验证。[实验与复现入口](https://github.com/{OWNER}/one-gpu-lm/blob/main/docs/modelscope-release.md)

## 加载

验证环境使用 Python `{software['python']}` 和 {dependencies}。先将发布的两个适配器文件下载到 `./adapter`，再加载固定版本底座。

```python
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import {evaluation['loader']}

model_path = snapshot_download({model!r}, revision={revision!r})
base = {evaluation['loader']}.from_pretrained(
    model_path,
    dtype=torch.{evaluation['dtype']},
    device_map={{"": 0}},
    attn_implementation={evaluation['attention']!r},
    local_files_only=True,
    trust_remote_code=False,
)
model = PeftModel.from_pretrained(base, "./adapter", is_trainable=False).eval()
```

{source}
'''


def training_for_release(args, permission):
    paths = [getattr(args, key, None) for key in ('training_protocol', 'training_profile', 'training_integrity')]
    if any(path is not None for path in paths):
        require(all(path is not None for path in paths), 'Provide the training protocol, profile and integrity receipt together')
        return reviewed_training(permission, args.input_review, *paths)
    training = read(TRAINING/'candidate-profile.json')
    integrity = read(TRAINING/'results/2026-11-17-training/integrity.json')
    require(integrity['passed'] is True
            and all(integrity['adapter_files_sha256'][name] == permission['checkpoint_sha256'][name] for name in FILES),
            'Expected the published 031 training lineage for this adapter')
    return training, None


def assemble_approved(args, permission, out):
    """Package the same final-suite evidence using an experiment's verified promotion."""
    require(permission['approved'] is True, 'Development release gate: '+permission['stage'])
    pairs,checks,table = audited_scores(out,permission)
    license_id,notices = approved_notices(permission,args.input_review,args.rights,args.notice_readback)
    training, lineage = training_for_release(args, permission)
    serving = read(PROFILE)

    reloaded = compare(read(args.before),read(args.after))
    config = reloaded['configuration']
    frozen = read(REFERENCE/'protocol.json')['frozen_files']
    require(reloaded['save_reload_greedy_match'] is True
            and config['model_id'] == permission['model_id'] and config['base_revision'] == permission['base_revision']
            and config['adapter_files_sha256'] == permission['checkpoint_sha256']
            and config['base_manifest_sha256'] == frozen[BASE_MANIFEST]
            and config['profile_sha256'] == frozen[PROFILE_KEY] == sha256(PROFILE),
            'Require independent matching reloads of the accepted adapter and final serving profile')
    if 'adapter_profile_sha256' in config:
        expected_profile = lineage['profile_sha256'] if lineage else sha256(TRAINING/'candidate-profile.json')
        require(config['adapter_profile_sha256'] == expected_profile,
                'The reload receipt used a different training profile')
    else:
        require(training['lora'] == serving['lora'],
                'Capture the reloads with the audited adapter profile when the LoRA recipe differs')

    model_endpoint(ENDPOINT,args.repo_id)
    require(args.repo_id.split('/')[0] == OWNER,'Use the authorized ModelScope owner')
    card = model_card(args.repo_id,permission,serving,training,config['software'],table,license_id,lineage=lineage)
    checks.update(save_reload_greedy_match=True,rights_and_secret_scan_pass=True,
                  single_training_seed_and_repository_cluster_limits_disclosed=True)
    decision = release_decision(permission,*pairs,checks)
    require(decision['accepted'] is True,'The registered release gate did not pass')
    require(not args.out.exists() and not args.out.is_symlink(),'Choose a fresh private release directory')
    with tempfile.TemporaryDirectory(prefix='.modelscope-release-',dir=args.out.parent) as temp:
        stage = Path(temp)/'release'
        stage.mkdir(mode=0o700)
        payload = stage/'payload'
        adapter = restore_adapter(args.adapter,payload,training,permission['checkpoint_sha256'])
        for name,digest in notices.items():
            entry = file_path(name)
            source = args.notices/entry
            require(source.is_file() and not source.is_symlink(),'Expected a reviewed local notice file')
            target = payload/entry
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes())
            require(sha256(target) == digest,'Notice bytes differ from the approved receipt')
        (payload/'README.md').write_text(card,encoding='utf-8')
        files = {name:{'bytes':(payload/name).stat().st_size,'sha256':sha256(payload/name)}
                 for name in sorted(['README.md',*FILES,*notices])}
        validate_files(files)
        manifest = {'endpoint':ENDPOINT,'repo_id':args.repo_id,'files':files}
        receipt = {'schema_version':1,'manifest':manifest,'release_license':license_id,
                   'adapter':adapter,'decision':decision,'final_test_opened':True,'weights_public':False,
                   'execution_audit_sha256':sha256(out/'execution-audit.json'),
                   'rights_review_sha256':sha256(args.rights),
                   'reload_captures_sha256':[sha256(args.before),sha256(args.after)]}
        if lineage is not None:
            receipt['training_lineage'] = lineage
        atomic_json(stage/'release-evidence.json',receipt)
        require(not args.out.exists() and not args.out.is_symlink(),'Release destination appeared during staging')
        stage.rename(args.out)
    return manifest


def assemble(args):
    permission = promotion_from_finished_run(args.root,args.seal,args.input_review)
    require(permission['approved'] is True, 'Development release gate: '+permission['stage'])
    return assemble_approved(args, permission, args.sealed/FINAL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','sealed','seal','input-review','rights','notice-readback','notices',
                 'before','after','adapter','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--repo-id',required=True)
    for name in ('training-protocol', 'training-profile', 'training-integrity'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    try:
        manifest = assemble(args)
    except FileNotFoundError:
        parser.exit(2,'A required release artifact or output parent is missing.\n')
    except ValueError as error:
        parser.exit(2,str(error)+'\n')
    print(json.dumps({'accepted':True,'staged_files':len(manifest['files']),'weights_public':False}))


if __name__ == '__main__':
    main()
