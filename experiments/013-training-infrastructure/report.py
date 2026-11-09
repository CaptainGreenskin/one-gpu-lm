"""Rebuild the public recovery report from allowlisted JSON/CSV evidence."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def stamp(value):
    return datetime.fromisoformat(value.replace('Z','+00:00'))


def describe_run(root):
    result=read(root/'result.json')
    if result['status']!='completed':
        raise ValueError('Incomplete run excluded')
    artifacts=read(root/'artifacts.json')['files']
    import hashlib
    for name,digest in artifacts.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Public evidence checksum mismatch: '+name)
    attempts=read(root/'attempts.json')
    metrics=read(root/'metrics.json')
    if [m['step'] for m in metrics]!=list(range(1,result['steps']+1)):
        raise ValueError('Incomplete update history')
    samples=list(csv.DictReader((root/'telemetry.csv').open()))
    hardware=[]
    for a in attempts:
        group=[s for s in samples if int(s['attempt'])==a['attempt']]
        times=[stamp(s['utc']) for s in group]
        gaps=[(b-a).total_seconds() for a,b in zip(times,times[1:])]
        if len(group)!=a['telemetry']['samples']:
            raise ValueError('Hardware sample count differs from attempt receipt')
        hardware.append({'attempt':a['attempt'],'samples':len(group),
            'sampling_errors':a['telemetry']['errors'],'thread_stopped':a['telemetry']['thread_stopped'],
            'first_utc':group[0]['utc'] if group else None,'last_utc':group[-1]['utc'] if group else None,
            'sampled_span_seconds':(times[-1]-times[0]).total_seconds() if times else 0,
            'max_gap_seconds':max(gaps,default=0),
            'peak_temperature_c':max((float(s['temperature_c']) for s in group),default=None),
            'peak_power_w':max((float(s['power_w']) for s in group),default=None),
            'peak_memory_used_mib':max((float(s['memory_used_mib']) for s in group),default=None),
            'mean_utilization_percent':statistics.mean(float(s['utilization_percent']) for s in group) if group else None})
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    return {'result':result,'started_utc':attempts[0]['started_utc'],'finished_utc':attempts[-1]['finished_utc'],
        'attempt_wall_seconds_sum':sum(a['wall_seconds'] for a in attempts),
        'elapsed_including_restart_seconds':(stamp(attempts[-1]['finished_utc'])-stamp(attempts[0]['started_utc'])).total_seconds(),
        'hardware':hardware,'committed_steps':[e['step'] for e in events if e['event']=='checkpoint_committed'],
        'restores':[e for e in events if e['event']=='state_restored']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    args=p.parse_args();root=args.results
    comparison=read(root/'comparison.json')
    a,b=[describe_run(root/n) for n in ['continuous','resumed']]
    if read(root/'continuous/identity.json')!=read(root/'resumed/identity.json'):
        raise ValueError('Cannot pair different run identities')
    summary={'schema_version':1,'comparison':comparison,'continuous':a,'resumed':b,
        'regression':{n:read(root/'regression'/n/'result.json') for n in ['base','adapted']},
        'quality_improvement_claim':False,'checkpoint_storage':read(root/'checkpoint-storage.json'),
        'installation':read(root/'installation.json'),'checks_linux':read(root/'checks-linux.json'),
        'source':read(root/'launch.json'),'integration':read(root/'integration.json')}
    for n,r in summary['regression'].items():
        if not r['complete']:raise ValueError('Incomplete regression: '+n)
    summary['failed_pair']=read(root/'failed-pairs/recovery-v3/comparison.json')
    (root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt']='one-gpu-lm-013'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    for name,color,style in [('continuous','#1267a5','-'),('resumed','#d26a22','--')]:
        metrics=read(root/name/'metrics.json')
        axes[0].plot([r['step'] for r in metrics],[r['loss'] for r in metrics],style,marker='o' if name=='continuous' else 'x',color=color,label=name)
        rows=list(csv.DictReader((root/name/'telemetry.csv').open()))
        for attempt in sorted({int(r['attempt']) for r in rows}):
            group=[r for r in rows if int(r['attempt'])==attempt]
            start=stamp(group[0]['utc'])
            axes[1].plot([(stamp(r['utc'])-start).total_seconds() for r in group],
                [float(r['memory_used_mib'])/1024 for r in group],label=f'{name} / process {attempt}')
    axes[0].axvline(4,color='#777777',linestyle=':',label='resume boundary')
    axes[0].set(xlabel='Optimizer update',ylabel='Training loss',title='Same examples; not a quality benchmark')
    axes[1].set(xlabel='Seconds since first sample in each process',ylabel='GPU memory (GiB)',title='Load, updates, checkpointing and probe')
    for ax in axes:ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.savefig(root/'curves.png',dpi=160);fig.savefig(root/'curves.svg',metadata={'Date':None});plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for label,directory in [('Original settings (failed)',root/'failed-pairs/recovery-v3'),('Deterministic settings',root)]:
        left,right=[read(directory/n/'metrics.json') for n in ['continuous','resumed']]
        for ax,key in zip(axes,['loss','grad_norm']):
            ax.plot([r['step'] for r in left],[abs(a[key]-b[key]) for a,b in zip(left,right)],marker='o',label=label)
    for ax,key in zip(axes,['loss','gradient norm']):
        ax.axvline(4,color='#777777',linestyle=':');ax.set(xlabel='Optimizer update',ylabel='Absolute paired '+key+' difference')
        ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.savefig(root/'failure-diagnosis.png',dpi=160);plt.close(fig)
    tz=timezone(timedelta(hours=8))
    times=lambda v:stamp(v).astimezone(tz).strftime('%Y-%m-%d %H:%M:%S')
    rows=['# 013：27B 单卡训练基础设施验收','',
        f"恢复对照：**{'通过' if comparison['passed'] else '未通过'}**。连续 12 步与第 4 步退出、换进程续到 12 步配对。这里只验收训练状态与运维链路，不据此宣称编码能力提高。",'',
        '![训练与硬件曲线](curves.png)','',
        '| 项目 | 连续训练 | 换进程恢复 |','| --- | ---: | ---: |',
        f"| 更新数 | {a['result']['steps']} | {b['result']['steps']} |",
        f"| 更新用时（秒） | {a['result']['training_update_seconds']:.3f} | {b['result']['training_update_seconds']:.3f} |",
        f"| 各进程墙钟之和（秒） | {a['attempt_wall_seconds_sum']:.3f} | {b['attempt_wall_seconds_sum']:.3f} |",
        f"| 更新阶段峰值 allocated（GiB） | {a['result']['peak_allocated_bytes']/2**30:.3f} | {b['result']['peak_allocated_bytes']/2**30:.3f} |",
        f"| 更新阶段峰值 reserved（GiB） | {a['result']['peak_reserved_bytes']/2**30:.3f} | {b['result']['peak_reserved_bytes']/2**30:.3f} |",'',
        f"连续训练北京时间 {times(a['started_utc'])} 至 {times(a['finished_utc'])}；恢复组 {times(b['started_utc'])} 至 {times(b['finished_utc'])}。墙钟包含预检、模型加载、保存和探针，更新用时只统计前向、反向及优化器更新。",'',
        f"可训练参数 {a['result']['trainable_parameters']:,}，底座冻结。两组都使用同一 12 条样本顺序，共输入 {a['result']['input_tokens']:,} token、监督 {a['result']['supervised_tokens']:,} token。不同步骤的样本不同，loss 折线不能解释为同一验证集的学习曲线。",'',
        '## 恢复检查','',
        f"Adapter 最大绝对误差 {comparison['adapter_max_absolute_error']:.9g}；adapter 字节级状态一致：{comparison['adapter_state_bitwise_equal']}；优化器字节级状态一致：{comparison['optimizer_state_bitwise_equal']}；逐步 loss 完全相同：{comparison['per_step_losses_bitwise_equal']}。",'',
        '| 检查 | 结果 |','| --- | --- |']
    rows += [f"| `{name}` | {'通过' if value else '失败'} |" for name,value in comparison['checks'].items()]
    rows += ['',f"恢复位置：{comparison['restored_steps']}。容差预先固定：adapter/loss atol=1e-7、rtol=1e-6；优化器 atol=1e-10、rtol=1e-6。原始 [comparison.json](comparison.json) 保留具体判定。",'',
        '## 硬件采样','',
        '| 组 / 进程 | 样本 | 采样错误 | 最大间隔秒 | 温度峰值°C | 功率峰值W | 显存峰值GiB |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,run in [('continuous',a),('resumed',b)]:
        for h in run['hardware']:
            rows.append(f"| {name} / {h['attempt']} | {h['samples']} | {h['sampling_errors']} | {h['max_gap_seconds']:.2f} | {h['peak_temperature_c']:.0f} | {h['peak_power_w']:.2f} | {h['peak_memory_used_mib']/1024:.3f} |")
    rows += ['', '每 5 秒采样一次，包含加载、训练、保存和探针。预检和进程间停机不在采样区间；各进程首末时间、覆盖时长和均值见 [summary.json](summary.json)，原始记录见两组 `telemetry.csv`。它不是满载性能基准或长时间稳定性测试。','',
        '## 检查点与环境','',
        f"两个训练组各保留 {summary['checkpoint_storage']['continuous']['checkpoint_count']} 个完整检查点（含第 0 步）。连续组占用 {summary['checkpoint_storage']['continuous']['retained_bytes']/2**30:.3f} GiB，恢复组占用 {summary['checkpoint_storage']['resumed']['retained_bytes']/2**30:.3f} GiB；具体大小见 [checkpoint-storage.json](checkpoint-storage.json)。",'',
        '独立 Python 3.10 环境安装与 pip check 通过；61 包版本见 [environment.lock.txt](environment.lock.txt)。43 项 CPU/Linux 检查通过。完整版本约束根据这次实际安装生成，并做离线解析核对；未声称对加约束后的脚本再次执行了全新下载安装。','',
        '## 已有函数回归','']
    for name,r in summary['regression'].items():
        rows.append(f"- {name}: {r['passed']}/{len(r['results'])}。")
    rows += ['', '这些是已有的九道回归题，非新的独立测试，不能用于日用模型晋升。生成文本、任务原文与权重均未公开。','',
        '## 失败和复现范围','',
        '正式配对前两次预检拒绝：一次为依赖名称规范化问题，一次为旧 venv 中同名包的遮蔽识别问题；均未执行优化器更新。随后使用独立 venv，安装时的网络中断另留记录。失败、修复与后续尝试见 [integration.json](integration.json)。数值阈值、训练数据、步数预算未因失败而调整。','',
        '初次完整 GPU 配对也未通过：[失败比较](failed-pairs/recovery-v3/comparison.json)。初始 adapter、优化器、随机状态相同，第一步前向 loss 相同，但第一步梯度范数已经不同；第 2 步起 loss 分叉，发生在第 4 步中断之前。恢复加载的三个状态指纹仍全部匹配。因此这次失败首先揭示了非重复的训练计算，不能归为丢失优化器或样本顺序。','',
        '随后启用 PyTorch/cuDNN 确定性算法、固定 cuBLAS workspace、关闭 TF32，在新 run 中重做完整配对。数据、更新数、学习率和数值阈值均未改变；执行代码和数值设置身份改变并分别保留。未做单一开关消融，也没有确认具体上游内核缺陷，不能把修复归因于某一个开关。设置依据见 [PyTorch 确定性算法说明](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)。','',
        '![保留失败并复验数值设置](failure-diagnosis.png)','',
        '实际仅验证完整更新边界的进程退出和恢复；未宣称任意内核中断、整机断电、远程文件系统或多卡可恢复。CPU 另覆盖半写入、损坏/不兼容检查点和互斥锁。','',
        '从 [固定协议](../../protocol.json)、[运行手册](../../../../docs/27b-training.md) 和 [代码提交回执](launch.json) 复现。原数据为私有有来源的训练分区，外部读者需要按数据契约准备自己有权使用的记录，因此本公开包不支持逐样本完全重算。','',
        '```bash','python experiments/013-training-infrastructure/report.py \\',f'  --results experiments/013-training-infrastructure/results/{root.name}','```','',
        '报告生成需要 matplotlib；原始 JSON/CSV 的核对不依赖 GPU。公开包仅包含白名单证据，模型、adapter、优化器、数据和原始日志均保留在私有目录。','']
    (root/'README.md').write_text('\n'.join(rows))
    print('Recovery report rebuilt; passed =',comparison['passed'])


if __name__=='__main__':main()
