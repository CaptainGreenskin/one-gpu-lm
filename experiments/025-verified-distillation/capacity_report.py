"""Describe the one capacity probe; synthetic loss is never a quality score."""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path


def report(root):
    root=Path(root);v=json.loads((root/'result.json').read_text());s=json.loads((root/'supervisor.json').read_text())
    if not s['complete']:
        raise ValueError('Wait for the supervisor to finish')
    samples=list(csv.DictReader((root/'telemetry.csv').open())) if (root/'telemetry.csv').exists() else []
    stamps=[datetime.fromisoformat(r['utc']).timestamp() for r in samples]
    summary={'probe_complete':v.get('complete',False),'capacity_gate_passed':v.get('capacity_gate_passed',False),
             'supervisor_timed_out':s['timed_out'],'worker_returncode':s['worker_returncode'],
             'optimizer_steps':v.get('optimizer_steps'),'final_test_opened':False,'new_development_scores':0,
             'hardware_samples':len(samples),'maximum_sample_gap_seconds':max((b-a for a,b in zip(stamps,stamps[1:])),default=0),
             'collection_errors':v.get('telemetry',{}).get('errors'),'quality_improvement_established':False}
    if samples:
        summary['peak_device_used_gib_sampled']=max(float(r['memory_used_mib']) for r in samples)/1024
        if 'device_total_bytes' in v:
            summary['total_minus_sampled_peak_used_gib']=v['device_total_bytes']/1024**3-summary['peak_device_used_gib_sampled']
    if samples:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        minutes=[(t-v['started_epoch'])/60 for t in stamps]
        fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True)
        axes[0].plot(minutes,[float(r['memory_used_mib'])/1024 for r in samples],color='#167d8d')
        axes[0].set(ylabel='nvidia-smi memory (GiB)',title='One synthetic backward probe; includes model loading')
        axes[1].plot(minutes,[float(r['utilization_percent']) for r in samples],lw=.9)
        axes[1].set(xlabel='Minutes since worker start',ylabel='GPU utilization (%)',ylim=(0,105))
        fig.tight_layout();fig.savefig(root/'capacity-hardware.png',dpi=160);plt.close(fig)
    zone=timezone(timedelta(hours=8))
    def timestamp(key):
        return datetime.fromtimestamp(v[key],zone).strftime('%Y-%m-%d %H:%M:%S') if key in v else '未记录'
    lines=['# 025：8K 训练容量探测','',
        '**'+('通过' if summary['capacity_gate_passed'] else '未通过')+'预登记容量门槛**。这是合成输入的一次反向传播，没有优化器更新、学生权重或编码能力评分。','',
        f"北京时间 {timestamp('started_epoch')} 至 {timestamp('finished_epoch')}。监督进程超时：{s['timed_out']}；worker 退出码：{s['worker_returncode']}。",'',
        '| 项目 | 观测 |','| --- | --- |']
    for key,label in [('sequence_tokens','序列 token 数'),('supervised_suffix_tokens','监督后缀 token 数'),
                      ('trainable_parameters','LoRA 可训练参数'),('backward_seconds','前向、反向及裁剪秒数'),
                      ('optimizer_steps','优化器步骤'),('adapter_parameters_unchanged','adapter 参数全部未变'),
                      ('frozen_parameter_samples_unchanged','冻结参数抽查未变'),('frozen_gradients_absent','冻结参数无梯度'),
                      ('frozen_files_unchanged','协议输入文件未变')]:
        lines.append(f'| {label} | {v.get(key,"未记录")} |')
    for key,label in [('peak_allocated_bytes','PyTorch 峰值 allocated'),('peak_reserved_bytes','PyTorch 峰值 reserved'),
                      ('reserved_headroom_bytes','设备总量减峰值 reserved'),('device_total_bytes','设备总显存')]:
        if key in v:lines.append(f'| {label} | {v[key]/1024**3:.2f} GiB |')
    if 'peak_device_used_gib_sampled' in summary:
        lines.append(f'| nvidia-smi 采样峰值 used | {summary["peak_device_used_gib_sampled"]:.2f} GiB |')
    if 'total_minus_sampled_peak_used_gib' in summary:
        lines.append(f'| 设备总量减采样峰值 used | {summary["total_minus_sampled_peak_used_gib"]:.2f} GiB |')
    lines+=['','预分配了两份 AdamW moment 形状缓冲，但没有创建或执行优化器；8 GiB 的预留门槛未调整。完整软件版本、数值、错误状态见 [result.json](result.json)。合成 loss 与梯度只作数值/容量检查，不能解释为任务质量。','',
            f'硬件采样 {len(samples)} 次，最大相邻间隔 {summary["maximum_sample_gap_seconds"]:.2f} 秒，采样错误 {summary["collection_errors"]} 次。图中 nvidia-smi 显存与 PyTorch allocator 统计口径不同，不能互相替代。','',
            '![容量探测硬件记录](capacity-hardware.png)' if samples else '本次没有可绘制的硬件采样。','',
            '局限：只执行一次最长形状的合成 batch，冻结基模只做参数抽查和无梯度检查。总量减 PyTorch reserved 的余量不等于实际可用显存；nvidia-smi 为稀疏采样，也不保证捕获真实峰值。真实数据、完整优化器更新、持续训练碎片和非 PyTorch 分配仍需正式训练验证。此结果不授权开发集提前评分或参数发布。','',
            '[固定规则](../../capacity-protocol.json) · [方法](../../capacity.md) · [监督进程回执](supervisor.json)']
    (root/'README.md').write_text('\n'.join(lines)+'\n');(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();print(json.dumps(report(a.results)))
