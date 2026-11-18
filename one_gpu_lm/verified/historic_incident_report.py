"""Explicit incomplete-run reporting, without changing the frozen 032 evaluator.

The original renderer counts audit placeholders as recorded runs and reports
zero disagreements when there are no repeated pairs. Those fields must not be
read as full coverage or demonstrated stability. This post-run presentation
revision preserves every input, score and preregistered acceptance gate.
"""
import argparse
import json
from pathlib import Path
from .historic_report import summarize, report_evaluation, VARIANTS


def corrected_summary(bundle, incident):
    result = summarize(bundle)
    rows = bundle['audit']['rows']
    completed = [r for r in rows if r.get('complete')]
    started = [r for r in rows if r.get('attempted_calls', 0) > 0]
    result.update(reporting_revision='032-interruption-v1', completed_runs=len(completed),
                  audited_completed_runs=sum(r.get('audit_passed', False) for r in completed),
                  incomplete_runs=len(started) - len(completed),
                  not_started_runs=result['registered_runs'] - len(started),
                  missing_runs=result['registered_runs'] - len(completed),
                  conclusion='incomplete_no_parameter_gain_conclusion',
                  development_parameter_gate_passed=False,
                  weight_publication_allowed=False)
    unknown = incident['unknown_response_requests']
    if unknown != 1 or incident['unknown_generated_tokens_bounds'] != [0, 512]:
        raise ValueError('This incident report requires its single bounded lost response')
    for cost in [result['cost'], *[x['cost'] for x in result['variants'].values()]]:
        for original, renamed in [('generated_tokens', 'known_generated_tokens'),
                                  ('input_tokens', 'known_input_tokens'),
                                  ('actor_seconds', 'completed_actor_seconds')]:
            cost[renamed] = cost.pop(original)
        cost['coverage'] = 'received generations and completed actor durations only'
    result['cost'].update(unknown_response_requests=unknown, exact_generation_cost_known=False,
        generated_tokens_bounds=[result['cost']['known_generated_tokens'],
                                result['cost']['known_generated_tokens'] + 512])
    for variant, value in result['variants'].items():
        group = [r for r in rows if r['variant'] == variant]
        value['scheduled_runs'] = value['recorded_runs']
        value['recorded_runs'] = sum(r.get('attempted_calls', 0) > 0 for r in group)
        value['not_started_runs'] = value['scheduled_runs'] - value['recorded_runs']
        value['incomplete_runs'] = sum(r.get('attempted_calls', 0) > 0 and not r.get('complete') for r in group)
        value['invalid_actions_from_complete_runs'] = value.pop('invalid_actions')
        value['repeat_coverage_tasks'] = result['paired_tasks']
        if result['paired_tasks'] == 0:
            value['repeat_outcome_disagreement_tasks'] = None
            value['repeat_action_disagreement_tasks'] = None
    return result


def report(directory, render=True):
    directory = Path(directory)
    bundle = json.loads((directory / 'analysis-input.json').read_text())
    incident = json.loads((directory / 'incident.json').read_text())
    # Reuse frozen score/CSV calculations, then explicitly correct only the
    # incomplete-run presentation. Neither audit nor original inputs is changed.
    report_evaluation(directory, render=False)
    result = corrected_summary(bundle, incident)
    (directory / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        labels = ['Completed / audited', 'Interrupted', 'Not started']
        counts = [result['completed_runs'], result['incomplete_runs'], result['not_started_runs']]
        axes[0].barh(labels, counts, color=['#167d8d', '#c46b26', '#cccccc'])
        for i, number in enumerate(counts):
            axes[0].text(number + 2, i, str(number), va='center')
        axes[0].set(xlim=(0, 260), xlabel='Scheduled task runs', title='Coverage of 324 registered runs')
        axes[1].axis('off')
        axes[1].text(.5, .5, '0 / 36 complete three-repeat pairs\n\nNo paired gain estimate\nRepeat stability: unavailable\nAcceptance gate: not passed',
                     ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        fig.suptitle('Interrupted evaluation: unequal coverage is not a model comparison')
        fig.tight_layout(); fig.savefig(directory / 'outcomes.png', dpi=160); plt.close(fig)
        steps = incident['known_response_steps']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].bar([s['call'] for s in steps], [s['generated_tokens'] for s in steps],
                    color=['#167d8d' if s['action_format_valid'] else '#c46b26' for s in steps])
        axes[0].axhline(512, color='#777777', ls='--', lw=.8)
        axes[0].set(xlabel='Model call', ylabel='Received generated tokens', xticks=list(range(1, 11)),
                    title='Calls 3–9: same invalid output, each at cap', xlim=(.3, 10.8))
        axes[0].annotate('Call 10:\nno receipt', xy=(10, 100), ha='center', fontsize=9)
        total = 0; elapsed = []
        for step in steps:
            total += step['generation_seconds']; elapsed.append(total)
        axes[1].plot([s['call'] for s in steps], elapsed, marker='o', color='#167d8d')
        axes[1].set(xlabel='Completed response', ylabel='Cumulative known generation seconds',
                    title='Model time only; tools and overhead excluded')
        fig.suptitle('Failure trace before the 600-second actor deadline; no extra inference')
        fig.tight_layout(); fig.savefig(directory / 'timeout.png', dpi=160); plt.close(fig)
    phase = result['phase']; cost = result['cost']
    lines = ['# 032 开发评测中断记录', '',
        '**本轮未完成验收，不支持参数收益或重复稳定性结论。** 计划 324 次，完成且逐条审计通过 92 次，另 1 次中断、231 次未启动；三种权重各三遍的完整配对为 0／36。最终测试继续封存，不发布权重。', '',
        '## 故障与停止原因', '',
        '032 在 `python-dotenv-0b2880591780` 的首次运行中，前两次调用分别执行测试和读取源码；第 3～9 次生成完全相同的非法工具输出，每次达到 512 token 上限。第 10 次请求在接近 600 秒任务截止时超时。已收到的 9 次响应共 3,656 token，最后一次响应没有持久回执，其实际生成量不可恢复。', '',
        '旧后端先完成生成再通过 socket 交付，没有先持久保存响应。客户端超时后，冻结的评测器按“未知生成成本即停止”规则终止整轮并释放 GPU。模型重复输出触发了任务超时；响应落盘和单任务故障隔离不足，使超时升级为整轮中断。记录未显示显卡故障，但这些记录不构成完整硬件诊断。', '',
        '原现场、92 条完整轨迹和 1 条部分轨迹均保留；没有重试、补造末次 token、延长该题预算或把未运行任务记成失败。来源与协议哈希均未变化。逐条结果见 [审计](evaluation-audit.json)；故障的数值证据与轨迹哈希见 [incident.json](incident.json)。', '',
        '![故障响应与已知生成时间](timeout.png)', '',
        '## 覆盖与描述性结果', '',
        '下表只描述已完成部分。覆盖范围不同，且每题尚无三次重复，不能横向估计参数收益；重复波动应为“不可评估”，不能写成“零波动”。', '',
        '| 权重 | 已完成且审计通过 | 修对并提交 | 修对未提交 | 中断 | 未启动 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for variant, value in result['variants'].items():
        lines.append(f"| {variant} | {value['verified_runs']} | {value['successes']} | {value['categories'].get('correct_patch_not_submitted', 0)} | {value['incomplete_runs']} | {value['not_started_runs']} |")
    lines += ['', '![实际覆盖](outcomes.png)', '', '## 成本与报告修正', '',
        f"流水线从 {phase['started_utc']} 到 {phase['finished_utc']}，共 {phase['reserved_single_gpu_hours']:.3f} 单卡小时，包含 16 次加载、CPU 工具与判分。不能等同于 GPU 计算活跃时间。",
        f"保留了 {cost['attempted_calls']:,} 次请求，已收到响应的生成 token 为 {cost['known_generated_tokens']:,}；最后一次响应实际为 0～512 token，全部生成量只能报告为 [{cost['generated_tokens_bounds'][0]:,}, {cost['generated_tokens_bounds'][1]:,}]，不能报告成精确总量。已完成 actor 的耗时合计不包含部分任务；总流水线时间包含中断及清理。", '',
        '![评测硬件采样](hardware.png)', '',
        '冻结的旧报告器把 324 个审计占位条目当作已记录条目，并在没有完整重复配对时输出零波动。本目录的事后报告修正仅调整覆盖、缺失、已知成本和“不可评估”的展示，不改判分、原始审计或冻结验收门槛。逐次数字见 [runs.csv](runs.csv)，修正后的汇总见 [summary.json](summary.json)。空的 pairs/families CSV 表示没有合格完整配对。', '',
        '## 已完成的基础设施修复及边界', '',
        '新增可选的 request journal 后端：先保存请求接收回执，生成完成后将完整响应 fsync 落盘，再交付 socket；相同请求身份不得再次生成。CPU 隔离测试覆盖客户端超时、迟到回执、重复请求、身份冲突、进程中断和生成异常。迟到响应仅供成本核算，不得执行其中的工具或延长 actor 预算。', '',
        '修复组件没有回写这次冻结运行，也尚未完成真实 GPU 超时集成验收。进程在生成中被杀且没有最终回执时，成本仍必须标未知；journal 不会凭空恢复已经丢失的 token。下一次正式对照须先验证超时隔离和审计闭环，再冻结新的运行规则。', '',
        '训练候选、输入、配置和本次原始审计都保留。独立仓库、可见测试与限定文件范围的任务，仍不足以证明生产级或通用长任务能力。训练曲线与数据来源见 [训练记录](../2026-11-18-training/)。', '',
        '复算本中断报告：`python -m one_gpu_lm.verified.historic_incident_report --results <本目录>`；可加 `--no-figures`。最终测试保持封存；ModelScope 发布门槛未满足。']
    (directory / 'README.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    value = report(args.results, not args.no_figures)
    print(json.dumps({k: value[k] for k in ['completed_runs', 'incomplete_runs', 'not_started_runs', 'paired_tasks', 'conclusion']}))
