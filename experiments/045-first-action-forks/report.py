"""Make an anonymous report from the completed, audited collection."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.verified.recovery_report import hardware


ARMS = ('greedy', 'sampled')
FLAGS = ('audited', 'independently_correct', 'verified_and_submitted', 'timeout_recovered')
COUNTS = ('attempted_calls', 'input_tokens', 'generated_tokens', 'invalid_actions')
LABELS = ('same_outcome', 'time_or_protected_source', 'first_action_not_executed', 'same_first_command',
          'template_incompatible', 'token_budget_or_shared_prefix', 'matched_executed_contrast')
BEIJING = timezone(timedelta(hours=8))


def read(path):
    return json.loads(path.read_bytes())


def anonymous_pairs(rows, aliases):
    matched = []
    for row in rows:
        if set(row['arms']) != set(ARMS):
            raise ValueError('A matched pair has an incomplete arm inventory')
        arms = {arm:{**{key:row['arms'][arm][key] for key in (*FLAGS,*COUNTS)},
                     'first_edit_call':row['arms'][arm]['first_edit_call'],
                     'completion_call':row['arms'][arm]['submit_call'] if row['arms'][arm]['verified_and_submitted'] else None,
                     'actor_seconds':row['arms'][arm]['actor_seconds']} for arm in ARMS}
        matched.append({'pair':row['pair'],'family':aliases[row['family']], 'available_calls':row['available_calls'],
                        'arms':arms,'contrast':row['contrast']})
    return matched


def capture(out, protocol, publication):
    controller, audit = out/'controller.json', out/'audit.json'
    state, replay, registered = read(controller), read(audit), read(publication)
    checksum = sha256(protocol)
    if (state['complete'] is not True or state['backend_stopped'] is not True
            or not state['frozen_inputs_unchanged'] or replay['passed'] is not True
            or not all(row['training_only'] is True and row['final_test_opened'] is False for row in (state,replay))
            or state['optimizer_updates'] or replay['optimizer_updates'] or replay['weights_public']
            or state['protocol_sha256'] != checksum or replay['protocol_sha256'] != checksum
            or registered['protocol_sha256'] != checksum or registered['ci_conclusion'] != 'success'
            or registered['anonymous_files_verified'] is not True
            or replay['controller_sha256'] != sha256(controller)
            or not replay['audited_attempts'] == replay['expected_attempts'] == state['expected_rollouts']
            or replay['audited_attempts'] != len(state['runs']) or 2*len(replay['pairs']) != len(state['runs'])):
        raise ValueError('Expected the completed first-action registration and replay')
    aliases = {value:f'F{i:02d}' for i,value in enumerate(sorted({row['family'] for row in replay['pairs']}),1)}
    matched = anonymous_pairs(replay['pairs'], aliases)
    telemetry = hardware(out/'telemetry.csv',state)
    keep = ('samples','first_to_last_seconds','maximum_gap_seconds','utilization_percent',
            'power_w','memory_used_mib','temperature_c')
    return {'schema_version':1,'protocol_sha256':checksum,'audit_sha256':sha256(audit),
            'controller_sha256':sha256(controller),'registration_commit':registered['commit'],
            'started_epoch':state['started_epoch'],'finished_epoch':state['finished_epoch'],
            'journal_requests_verified':replay['journal_requests_verified'],
            'qualified_pairs':sum(replay['eligible_contrasts'].values()),'pairs':matched,
            'hardware':{key:telemetry[key] for key in keep},'training_only':True,
            'optimizer_updates':0,'final_test_opened':False,'weights_public':False}


def summarize(value, registration):
    pairs = value['pairs']
    if (value['protocol_sha256'] != registration['protocol_sha256'] or value['training_only'] is not True
            or value['final_test_opened'] is not False or value['weights_public'] is not False or value['optimizer_updates']
            or not value['started_epoch'] < value['finished_epoch']
            or len(pairs) != registration['training_states']
            or len({pair['family'] for pair in pairs}) != registration['families']
            or [p['pair'] for p in pairs] != list(range(1,len(pairs)+1))):
        raise ValueError('The anonymous results differ from the registered collection')
    aliases = {f'F{i:02d}' for i in range(1,registration['families']+1)}
    outcome_by_winners = {(): 'neither', ('greedy',): 'greedy_only', ('sampled',): 'sampled_only', ARMS: 'both'}
    outcome_counts, contrast_counts = Counter(), Counter()
    requests = 0
    for pair in pairs:
        if pair['contrast'] not in LABELS or set(pair['arms'])!=set(ARMS) or pair['family'] not in aliases:
            raise ValueError('Unexpected collection result or family alias')
        contrast_counts[pair['contrast']] += 1
        for cell in pair['arms'].values():
            n = cell['attempted_calls']
            if (any(type(cell[key]) is not bool for key in FLAGS) or cell['audited'] is not True
                    or any(type(cell[key]) is not int or cell[key]<0 for key in COUNTS)
                    or not 0 < n <= pair['available_calls'] or not 0 < cell['input_tokens'] <= n*8192
                    or not 0 <= cell['generated_tokens'] <= n*512 or cell['invalid_actions'] > n
                    or not math.isfinite(cell['actor_seconds']) or cell['actor_seconds']<0
                    or any(call is not None and (type(call) is not int or not 1<=call<=n)
                           for call in (cell['first_edit_call'],cell['completion_call']))):
                raise ValueError('Unexpected anonymous task counts or timings')
            if (cell['completion_call'] is not None)!=cell['verified_and_submitted']:
                raise ValueError('A completion curve point requires a verified submission')
            requests += n
        winners = tuple(arm for arm in ARMS if pair['arms'][arm]['verified_and_submitted'])
        outcome_counts[outcome_by_winners[winners]] += 1
        if pair['contrast']=='matched_executed_contrast' and len(winners)!=1:
            raise ValueError('A qualified sample has no single successful branch')
    if (requests!=value['journal_requests_verified']
            or contrast_counts['matched_executed_contrast']!=value['qualified_pairs']):
        raise ValueError('Anonymous publication lost an audited branch or request')
    arm_counts = {arm:{'completed_repairs':sum(p['arms'][arm]['verified_and_submitted'] for p in pairs),
                       'never_edited':sum(p['arms'][arm]['first_edit_call'] is None for p in pairs),
                       'generated_tokens':sum(p['arms'][arm]['generated_tokens'] for p in pairs),
                       'attempted_calls':sum(p['arms'][arm]['attempted_calls'] for p in pairs)} for arm in ARMS}
    return {'paired_states':len(pairs),'rollouts':2*len(pairs),'verified_requests':requests,
            'outcomes':dict(outcome_counts),'arms':arm_counts,'contrast_counts':dict(contrast_counts),
            'qualified_pairs':value['qualified_pairs'],'wall_seconds':value['finished_epoch']-value['started_epoch'],
            'optimizer_updates':0,'final_test_opened':False}


def curves(pairs):
    for call in range(1,1+max(p['available_calls'] for p in pairs)):
        point = {'new_call':call}
        for arm in ARMS:
            for key,field in (('first_edit','first_edit_call'),('completed_repair','completion_call')):
                point[f'{key}_{arm}'] = sum(p['arms'][arm][field] is not None and p['arms'][arm][field]<=call for p in pairs)
        yield point


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_curves(directory, pairs, *, salt):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt'] = salt
    import matplotlib.pyplot as plt

    rows = [dict(pair=pair['pair'],family=pair['family'],available_calls=pair['available_calls'],
                 arm=arm,contrast=pair['contrast'],**pair['arms'][arm]) for pair in pairs for arm in ARMS]
    write_csv(directory/'matched-actions.csv',rows)
    points = list(curves(pairs))
    write_csv(directory/'call-curves.csv',points)
    fig, axes = plt.subplots(1,2,figsize=(9,3.6),layout='constrained',sharey=True)
    for axis,key,title in zip(axes,('first_edit','completed_repair'),('First source edit','Verified repair and submission')):
        for arm,color,label in zip(ARMS,('#2D6496','#B85034'),('Greedy','Sampled first response')):
            axis.step([point['new_call'] for point in points],[point[f'{key}_{arm}'] for point in points],
                      where='post',color=color,label=label,linewidth=2)
        axis.set(title=title,xlabel='New model calls after restoration',ylim=(0,len(pairs)+1))
        axis.grid(axis='y',alpha=.25)
    axes[0].set_ylabel('Training states')
    axes[0].legend(loc='best',frameon=False)
    graphic = directory/'call-curves.svg'
    fig.savefig(graphic,metadata={'Date':None})
    plt.close(fig)
    graphic.write_text('\n'.join(line.rstrip() for line in graphic.read_text().splitlines())+'\n',encoding='utf-8')


def render(directory, registration):
    value = read(directory/'measurements.json')
    result = summarize(value, registration)
    write_json(directory/'summary.json',result)
    write_curves(directory, value['pairs'], salt='045-first-action-forks')
    start,end = (datetime.fromtimestamp(value[key],BEIJING) for key in ('started_epoch','finished_epoch'))
    greedy, sampled = (result['arms'][arm]['completed_repairs'] for arm in ARMS)
    counts = result['outcomes']
    text = ['# 045 · 同一状态分叉的首次动作采集','',
            f'在 {result["paired_states"]} 个配对训练状态上，贪心组修对并提交 **{greedy}** 次，'
            f'首步采样组 **{sampled}** 次。共得到 **{result["qualified_pairs"]}** 组满足条件的偏好样本。','',
            f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}。'
            f'审计覆盖 {result["rollouts"]} 次执行和 {result["verified_requests"]} 条生成请求。', '',
            f'两组都完成 {counts.get("both",0)} 个状态，都未完成 {counts.get("neither",0)} 个。'
            f'贪心组独自完成 {counts.get("greedy_only",0)} 个，首步采样组独自完成 {counts.get("sampled_only",0)} 个。', '',
            '| 指标 | 贪心 | 首步采样 |','|---|---:|---:|']
    for label,key in (('全程未改码','never_edited'),('新调用数','attempted_calls'),('新生成 token','generated_tokens')):
        left,right = (result['arms'][arm][key] for arm in ARMS)
        text.append(f'| {label} | {left:,} | {right:,} |')
    telemetry = value['hardware']
    if telemetry['samples']:
        text += ['',f'记录 {telemetry["samples"]:,} 个 GPU 样本，首末采样跨度 '
                 f'{telemetry["first_to_last_seconds"]/60:.1f} 分钟。平均利用率 '
                 f'{telemetry["utilization_percent"]["mean"]:.1f}%，平均功耗 {telemetry["power_w"]["mean"]:.0f} W，'
                 f'峰值显存 {telemetry["memory_used_mib"]["maximum"]:.0f} MiB。']
    text += ['',
            '两组从相同的原始源码和可见反馈出发，首步之后共用贪心解码。计数反映训练样本采集，'
            '独立开发集用于检验后续训练后的模型。','',
            '![首次改码和修对提交曲线](call-curves.svg)','',
            '[匿名逐次数据](matched-actions.csv) · [配对统计](summary.json) · [冻结登记](../../registration.json)']
    (directory/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--registration',type=Path,default=Path(__file__).with_name('registration.json'))
    args = parser.parse_args()
    print(json.dumps(render(args.results,read(args.registration)),allow_nan=False))
