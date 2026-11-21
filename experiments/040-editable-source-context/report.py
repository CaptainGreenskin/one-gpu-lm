"""Make the paired source-context report from the completed anonymous 040 audit."""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys

from one_gpu_lm.common.receipts import sha256


HERE = Path(__file__).resolve().parent
BEIJING = timezone(timedelta(hours=8))


def shared_report():
    path = HERE.parent/'039-release-development/report.py'
    sys.path.insert(0,str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location('source_context_shared_report',path)
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        return report
    finally:
        sys.path.pop(0)


def run(directory, protocol_path):
    audit = json.loads((directory/'measurements.json').read_bytes())
    protocol = json.loads(protocol_path.read_bytes())
    if (audit['passed'] is not True or audit['final_test_opened'] or audit['weights_public'] or audit['optimizer_updates']
            or audit['protocol_sha256'] != sha256(protocol_path)
            or not audit['telemetry']['thread_stopped'] or audit['backend_loads'] != 1
            or audit['verified_requests'] != sum(r['attempted_calls'] for r in audit['rows'])):
        raise ValueError('Require the complete audited result for the registered context comparison')
    shared = shared_report()
    evidence = shared.reproduce_analysis(audit,protocol)
    rows = audit['rows']
    budget = protocol['evaluation_budget']
    points = shared.call_curves(rows,budget['max_model_calls'])
    shared.write_csv(directory/'call-curves.csv',points)
    shared.plot(directory/'call-curves.svg',points,labels=('Original opening','With source excerpts'))
    shared.write_csv(directory/'matched-tasks.csv',[
        {k:v for k,v in row.items() if k!='before_first_edit_actions'}
        for row in sorted(rows,key=lambda record:(record['task_number'],record['arm']))
    ])
    base,source = (evidence['successes'][name] for name in shared.ARMS)
    lower,upper = evidence['gain_interval_95_pp']
    status = '通过' if evidence['development_gate_passed'] else '未通过'
    start,end = (datetime.fromtimestamp(audit[k],BEIJING) for k in ('started_epoch','finished_epoch'))
    n = protocol['task_count']
    lines = ['# 040 · 开场源码与首次改码','',
             f'原开场修复并提交 **{base}/{n}**，附源码 **{source}/{n}**。净变化 '
             f'**{evidence["gain_pp"]:+.2f} 个百分点**，95% 配对 bootstrap 区间为 '
             f'[{lower:+.2f}, {upper:+.2f}]。独立复验门槛**{status}**。','',
             '| 指标 | 原开场 | 附源码 |','| --- | ---: | ---: |']
    for field,label,group in [('first_edit','有实际修改','behaviors'),('no_edit','全程未编辑','behaviors'),
                              ('correct_without_submission','修对未提交','behaviors'),('attempted_calls','调用数','costs'),
                              ('input_tokens','输入 token','costs'),('generated_tokens','生成 token','costs')]:
        counts = [evidence[group][arm][field] for arm in shared.ARMS]
        lines.append(f'| {label} | {counts[0]:,} | {counts[1]:,} |')
    duration = [evidence['costs'][arm]['actor_seconds'] for arm in shared.ARMS]
    lines += [f'| 任务执行累计秒数 | {duration[0]:,.2f} | {duration[1]:,.2f} |','',
              '![First edit and completed repair](call-curves.svg)','','| 家族 | 净变化（百分点） |','|---|---:|']
    lines.extend(f'| {name} | {gain:+.2f} |' for name,gain in evidence['family_gain_pp'].items())
    lines += ['',f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}。'
              f'从加载到退出用时 {(end-start).total_seconds()/3600:.2f} 小时，'
              f'{audit["verified_requests"]:,} 次生成请求通过审计。','',
              f'两组使用同一 27B 底座和 {n} 道既有开发题。此次结果用于选择下一轮待独立复验的推理流程。'
              '最终测试集封存。','',
              '[匿名逐题结果](matched-tasks.csv) · [曲线数值](call-curves.csv) · [审计结果](measurements.json)','']
    (directory/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args = p.parse_args()
    run(args.directory,HERE/'protocol.json')
