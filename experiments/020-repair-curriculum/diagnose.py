"""Describe 019 failures and supervision coverage without opening final tests.

This uses public numeric evidence only. Command-level causes require a separate
private trace review; an unchanged source snapshot does not prove no edit was
attempted. No development example becomes a training record.
"""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from one_gpu_lm.common.receipts import sha256, write_json


def observed_failure(row):
    edited = any(s['changed_editable_files'] > 0 for s in row['source_progress'])
    return {'passed': row['passed'], 'observed_allowed_source_edit': edited,
        'failure_category': None if row['passed'] else 'after_edit' if edited else 'without_observed_edit',
        'submitted_but_failed_without_edit': not row['passed'] and not edited and row['exit_status'] == 'Submitted',
        'format_invalid_calls': sum(g['action_format_valid'] is False for g in row['generations']),
        'progress_reminder_calls': sum(g['progress']['reminder'] for g in row['generations'])}


def analyze(run):
    run = Path(run)
    spec = importlib.util.spec_from_file_location('curriculum_parent_report', ROOT/'experiments/019-coding-release/report.py')
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    summary = parent.assess(run)
    if summary['selection']['selected'] is not None or summary['selection']['final_test_opened']:
        raise ValueError('020 expects the failed 019 screen with final scores still sealed')
    read = lambda name: json.loads((run/name).read_text())
    selection = read('data-selection.json')
    grouped = defaultdict(list)
    for row in selection['selected']:
        stage = row['id'].rsplit(':reference-', 1)
        if len(stage) != 2:
            raise ValueError('Unrecognized parent supervision identity')
        grouped[stage[1]].append(row)
    expected = {'inspect', 'reproduce', 'repair', 'review', 'verify', 'submit'}
    if set(grouped) != expected or any(len(rows) != 55 for rows in grouped.values()):
        raise ValueError('Parent supervision coverage differs from the executed six-step recipe')
    stages = {name: {'updates': len(rows), 'supervised_tokens': sum(r['supervised_tokens'] for r in rows),
                     'input_tokens': sum(r['tokens'] for r in rows)} for name, rows in grouped.items()}
    audits = read('execution-audit.json')
    arms = {}
    for arm in ['dev-base', 'dev-low', 'dev-high']:
        campaign = read('campaigns/'+arm+'.json')
        observations = [dict(id=r['id'], **observed_failure(r)) for r in campaign['results']]
        arms[arm] = {'tasks': len(observations), 'passed': sum(r['passed'] for r in observations),
            'failure_categories': dict(Counter(r['failure_category'] for r in observations if not r['passed'])),
            'submitted_but_failed_without_edit': sum(r['submitted_but_failed_without_edit'] for r in observations),
            'format_invalid_calls': sum(r['format_invalid_calls'] for r in observations),
            'progress_reminder_calls': sum(r['progress_reminder_calls'] for r in observations),
            'consecutive_repeated_commands': sum(r['consecutive_repeated_commands'] for r in audits['variants'][arm]),
            'observations': observations}
    regressions = [p for p in summary['pairs'] if p['outcome'] == 'regressed']
    return {'schema_version': 1, 'parent_commit': '2fbbf69d815cf7846410dc3246f15b19c3ccec23',
        'parent_protocol_sha256': sha256(run/'protocol.json'),
        'parent_collection_sha256': sha256(run/'collection.json'),
        'source': 'Validated public numeric evidence; no new model calls and no final-test reads',
        'supervision_by_stage': stages,
        'repair_update_fraction': stages['repair']['updates']/selection['selected_records'],
        'repair_supervised_token_fraction': stages['repair']['supervised_tokens']/selection['supervised_tokens'],
        'failure_recovery_target_stages': [],
        'recipe_coverage_observation': 'The verified 019 recipe repairs at call 3, before the call-5 no-edit reminder; later supervised actions already see repaired source. This is a recipe-level coverage deduction, not evidence that the gap caused the regressions.',
        'arms': arms, 'regressed_pairs': regressions,
        'high_regressions_with_format_errors': sum(p['candidate_invalid_format_calls'] > 0 for p in regressions if p['comparison'] == 'dev-high'),
        'private_command_level_diagnosis_complete': False,
        'causal_claims_supported': False, 'development_trajectories_used_as_training_data': False,
        'limitations': ['No edit observed can include an attempted edit that failed; raw command review is pending.',
            'Repair actions account for one sixth of updates but more than one sixth of label tokens.',
            'Single seed and single greedy rollout; learning-rate variants do not isolate data-coverage causes.',
            'The development set is repeatedly observed; only the unopened final cohorts can provide the later independent check.']}


def render(run, out):
    out = Path(out)
    if out.exists():
        raise ValueError('Preserve previous diagnosis receipts')
    result = analyze(run); out.mkdir(parents=True)
    write_json(out/'numeric-diagnosis.json', result)
    table = '\n'.join(f"| {name} | {v['passed']}/15 | {v['failure_categories'].get('without_observed_edit', 0)} | {v['failure_categories'].get('after_edit', 0)} | {v['submitted_but_failed_without_edit']} | {v['consecutive_repeated_commands']} |" for name,v in result['arms'].items())
    stages = '\n'.join(f"| {name} | {v['updates']} | {v['supervised_tokens']} |" for name,v in result['supervision_by_stage'].items())
    (out/'README.md').write_text(f'''# 020 · 019 的数值复盘

仅分析已公开的开发集与训练记录，没有新增模型调用，没有打开最终测试集。原始命令级复盘尚未完成，以下事实不构成退步机制的因果证明。

| 组别 | 通过 | 失败且未观察到源码修改 | 修改后仍失败 | 未修改且提前提交失败 | 连续重复命令次数 |
| --- | --- | --- | --- | --- | --- | --- |
{table}

低学习率的两道退步均修改过允许编辑的源码，但最终独立测试失败。高学习率的六道退步均未观察到源码修改，且这六题的工具格式错误数均为零。未观察到修改不等于没有尝试修改：失败的编辑命令也可能不改变文件，需用私有原始轨迹区分。

| 019 监督阶段 | 更新次数 | 监督 token |
| --- | --- | --- |
{stages}

修复占 {result['repair_update_fraction']:.1%} 的更新、{result['repair_supervised_token_fraction']:.1%} 的监督 token，不能简单说修复 token 被其他内容淹没。已验证的六步流程在第三次调用修复，因此没有覆盖第五次调用开始出现的未编辑进度提醒状态；也没有错误修改后测试仍失败的纠正目标。

020 待验证的假设：增加实际编辑目标和停滞状态覆盖可能有帮助；在相同训练更新与推理上限下，进一步加入执行过的错误修改与失败测试上下文，检验其是否比普通成功修复上下文更有效。开发题的源码、修复答案和轨迹不得进入训练。最终设计须在原始轨迹复盘和 Linux 数据准入完成后冻结。
''')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); value = render(args.parent_run, args.out)
    print(json.dumps({'repair_update_fraction': value['repair_update_fraction'],
        'repair_supervised_token_fraction': value['repair_supervised_token_fraction'],
        'high_regressions_with_format_errors': value['high_regressions_with_format_errors'],
        'private_command_level_diagnosis_complete': False}))
