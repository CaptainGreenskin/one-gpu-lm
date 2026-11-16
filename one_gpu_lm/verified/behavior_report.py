"""Recompute 026 descriptive tables and charts from public diagnosis receipts."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def summarize(data):
    rows = data['rows']
    training = [r for r in rows if r['phase'] == 'training']
    development = [r for r in rows if r['phase'] == 'development']
    if len(training) != 164 or len(development) != 720:
        raise ValueError('Expected complete 025 training and development inventories')
    if len({(r['id'], r['arm'], r['repetition']) for r in development}) != 720:
        raise ValueError('Duplicate development runs')
    arms = ['base_current', 'base_guard', 'student_current', 'student_guard']
    unique = {arm: {r['id']: r for r in development if r['arm'] == arm and r['repetition'] == 0} for arm in arms}
    if any(len(v) != 60 for v in unique.values()):
        raise ValueError('Incomplete first-repeat diagnostic comparison')
    eligible = [r for r in training if r['eligible_for_training']]
    if len(eligible) != data['training_allocation']['full_episodes']:
        raise ValueError('Training allocation disagrees with episodes')
    groups = {'teacher_all_164': training, 'teacher_selected_127': eligible,
              **{k: list(v.values()) for k, v in unique.items()}}
    summary = {}
    for name, members in groups.items():
        counts = Counter()
        for row in members:
            for step in row['steps']:
                counts[step['kind']] += 1
        summary[name] = {
            'unique_tasks': len(members), 'successes': sum(r['success'] for r in members),
            'no_edit_tasks': sum(r['first_edit_call'] is None for r in members),
            'multiple_edit_tasks': sum(r['source_edit_calls'] > 1 for r in members),
            'calls': sum(r['calls'] for r in members), 'action_counts': dict(sorted(counts.items())),
            'median_first_edit_among_edited': statistics.median([r['first_edit_call'] for r in members if r['first_edit_call'] is not None]),
            'compacted_calls': sum(r['context_compacted_calls'] for r in members),
            'generation_limit_calls': sum(r['generation_limit_calls'] for r in members),
            'feedback_truncated_calls': sum(r['feedback_truncated_calls'] for r in members),
            'exact_failed_edited_source_correction_episodes': sum(r['counts'].get('corrections_after_failed_exact_check_of_edited_source', 0) > 0 for r in members)}
    pairs = []
    for tid, base in unique['base_current'].items():
        student = unique['student_current'][tid]
        pairs.append({'id': tid, 'repository': base['repository'],
                      'base_success': base['success'], 'student_success': student['success'],
                      'base_calls': base['calls'], 'student_calls': student['calls'],
                      'base_first_edit': base['first_edit_call'], 'student_first_edit': student['first_edit_call'],
                      'student_context_compacted_calls': student['context_compacted_calls'],
                      'student_generation_limit_calls': student['generation_limit_calls'],
                      'student_truncated_feedback_calls': student['feedback_truncated_calls'],
                      'student_inspection_shaped_calls': student['counts'].get('inspection_shaped_no_source_change', 0),
                      'student_exact_repeat_calls': student['repeated_command_source_feedback_calls']})
    regressions = [r for r in pairs if r['base_success'] and not r['student_success']]
    gains = [r for r in pairs if not r['base_success'] and r['student_success']]
    return {'schema_version': 1, 'scope': 'post_hoc_diagnostic_only',
            'all_trajectories': len(rows), 'training_trajectories': len(training),
            'development_runs': len(development), 'development_repetitions_retained': 3,
            'descriptive_development_tables_use_repetition': 0,
            'training_allocation': data['training_allocation'], 'groups': summary,
            'pairs': pairs, 'regressions': regressions, 'gains': gains,
            'causal_training_mechanism_established': False,
            'new_model_calls': 0, 'optimizer_updates': 0, 'final_test_opened': False,
            'limitations': data['limitations']}


def render(summary, out):
    allocation = summary['training_allocation']; selected = summary['groups']['teacher_selected_127']
    lines = ['# 026：动作分配与未修改代码的诊断', '',
             '**这是 025 的事后诊断，不是新训练成绩，也没有修改原来的评分或验收门槛。**', '',
             '读取并校验全部 164 条采集轨迹及 720 次开发评测；轨迹哈希、工具元数据、原始反馈重算及 79 个冻结文件均核对。新增模型调用、优化器更新、任务命令执行均为 0。', '',
             '开发表只使用第 1 次重复的 60 道独立题；完整三次重复仍保留在数字回执中，不把 180 次运行算成 180 道题。', '',
             '## 五道退步题', '',
             '| 任务 ID | 底座首次修改 | 学生调用 | 学生首次修改 | 上下文裁剪次数 | 生成触顶次数 | 反馈截断次数 | 查阅形态动作 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in summary['regressions']:
        lines.append(f"| {r['id']} | {r['base_first_edit']} | {r['student_calls']} | {r['student_first_edit'] or '无'} | {r['student_context_compacted_calls']} | {r['student_generation_limit_calls']} | {r['student_truncated_feedback_calls']} | {r['student_inspection_shaped_calls']} |")
    lines += ['', '这五题没有触发上下文裁剪或单次生成上限，学生都在 12 次调用内没有修改源码。反馈截断在部分题上存在，但不能仅凭相关性断言它导致了退步。查阅命令不同于完全相同命令的循环；“持续阅读”和“机械重复”需要分开。', '',
              '## 实际训练目标分配', '',
              '| 审计动作类型 | 动作记录 | 监督 token | 训练时平均逐记录损失 |', '| --- | --- | --- | --- |']
    for kind, value in allocation['by_kind'].items():
        lines.append(f"| {kind} | {value['records']} | {value['supervised_tokens']} | {value['mean_record_loss']:.6f} |")
    lines += ['', f"127 条完整轨迹共 971 条动作记录。其中 580 条为原始审计的 other；本次按保守命令语法进一步识别出 {selected['action_counts'].get('inspection_shaped_no_source_change', 0)} 条查阅形态动作，其余不能直接叫作无效阅读。源码修改为 131 条，只有 {selected['multiple_edit_tasks']} 条完整轨迹包含多次源码修改。", '',
              f"教师首次修改的中位调用是第 {allocation['median_first_edit']:.0f} 次；{allocation['first_edit_after_8']} 条轨迹到第 8 次后才首次修改，{allocation['first_edit_after_12']} 条超过第 12 次。{allocation['episodes_over_12_calls']} 条完整教师轨迹使用了超过 12 次调用，而学生开发预算为 12 次。", '',
              '这些数字提示需要核查两件事：训练时是否过多奖励继续查阅，以及是否缺少学生失败状态下的修复示范。它们不能单独证明损失权重、学习率或过拟合是原因。平均损失来自不同样本，也不能作为各类动作的能力分数。', '',
              '严格“先修改 → 单独执行指定回归且失败 → 再修改”的计数为 0；这个规则不包含复合命令里的测试，也不包含返回码为 0 的自定义打印诊断，因此**不能解释为没有任何纠错示范**。多次修改也不自动等于纠错。', '',
              '![训练目标与首次修改分布](action-allocation.png)', '',
              '## 下一步实验设计', '',
              '优先准备训练分区内的同状态接手采集：先让 025 学生运行；在预先固定的第 3 次完整工具反馈后，若尚未修改源码，保存当时可见历史及工作目录。底座教师从同一状态继续，原学生也继续，以相同剩余调用和 token 上限比较。教师不得获得参考修复、隐藏判分反馈或开发题。', '',
              '这将验证能否产生有用的修复示范；它尚不是学生参数提升实验。所有失败、教师额外成本和任务重叠都要保留。只有执行与编码审计通过的训练侧接手轨迹，才可进入下一轮候选数据准备。开发轨迹仅用于诊断，不能转成训练目标。具体的训练侧样本清单及边界见 [接手试跑设计](../../recovery-pilot-design.json)。', '',
              '本阶段没有训练新适配器、调用外部教师服务或打开最终测试。下一次 GPU 采集尚未启动，执行器与启动协议仍需完成后冻结。', '',
              '复算：`python -m one_gpu_lm.verified.behavior_report --results <本目录>`；加 `--no-figures` 可只用标准库生成 [summary.json](summary.json)、[pairs.csv](pairs.csv) 与本报告。来源为 [diagnosis.json](diagnosis.json)；原始任务、命令、模型文本及权重不公开。']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')


def plot(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    groups = data['training_allocation']['by_kind']; keys = ['other', 'edit', 'verify', 'submit']
    values = [groups[k]['records'] for k in keys]
    axes[0].bar(keys, values, color=['#708395', '#167c80', '#d4a13c', '#7b68aa'])
    for i, v in enumerate(values):axes[0].text(i, v + 8, str(v), ha='center')
    axes[0].set(title='Actual training targets (971 records)', ylabel='Action records', ylim=(0, 660))
    categories = ['1-3', '4-6', '7-9', '10-12', 'No edit']
    for offset, arm, color in [(-.2, 'base_current', '#708395'), (.2, 'student_current', '#167c80')]:
        rows = [r for r in data['rows'] if r['phase'] == 'development' and r['arm'] == arm and r['repetition'] == 0]
        counts = [sum(r['first_edit_call'] is not None and lo <= r['first_edit_call'] <= hi for r in rows) for lo, hi in [(1,3),(4,6),(7,9),(10,12)]]
        counts.append(sum(r['first_edit_call'] is None for r in rows))
        axes[1].bar([i + offset for i in range(5)], counts, width=.4, label=arm, color=color)
    axes[1].set(xticks=range(5), xticklabels=categories, title='First source edit: 60 unique dev tasks', xlabel='Model call index', ylabel='Tasks')
    axes[1].legend();fig.savefig(out / 'action-allocation.png', dpi=160);plt.close(fig)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--no-figures', action='store_true');args=parser.parse_args()
    data=json.loads((args.results/'diagnosis.json').read_text());summary=summarize(data)
    (args.results/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (args.results/'pairs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary['pairs'][0]),lineterminator='\n');writer.writeheader();writer.writerows(summary['pairs'])
    render(summary,args.results)
    if not args.no_figures:plot(data,args.results)
    print(json.dumps({'diagnosed_trajectories':summary['all_trajectories'],'regressions':len(summary['regressions']),'gains':len(summary['gains']),'new_model_calls':0}))


if __name__=='__main__':main()
