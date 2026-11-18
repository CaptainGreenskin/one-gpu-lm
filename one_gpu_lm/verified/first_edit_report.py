"""Rebuild 033 next-action tables and figures from public numeric evidence."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from one_gpu_lm.common.receipts import write_json
from .cycle_report import hardware, phase_cost, write_csv
from .first_edit_inputs import BASE, CANDIDATE, PREDECESSOR, VARIANTS, read


ARMS = {BASE:'底座', PREDECESSOR:'031', CANDIDATE:'032'}
GROUPS = (('regression',BASE,'5 道退步题 · 底座历史'),
          ('regression',CANDIDATE,'5 道退步题 · 032 历史'),
          ('stable_success',BASE,'4 道稳定成功题 · 底座历史'))
CATEGORIES = {
    'exact_base_first_edit':('Matches historical base edit','#147c78'),
    'other_observed_first_edit':('Matches other historical edit','#6b57a6'),
    'declared_regression':('Declared regression','#db9f33'),
    'inspection_shaped':('Inspection shaped','#7c9bb6'),
    'other_parseable':('Other parseable action','#c7c7c7'),
    'submit':('Submit','#a47c39'),
    'invalid':('Invalid','#ba5047'),
}
KNOWN_EDITS = {'exact_base_first_edit','other_observed_first_edit'}


def summarize(data, preparation):
    audit = data['audit']
    states = preparation['prepared_states']
    rows = audit['rows']
    expected = [(variant,s['id']) for variant in VARIANTS for s in states]
    if (not audit['passed'] or audit['final_test_opened'] or not data['phase']['complete']
            or [(r['variant'],r['state_id']) for r in rows] != expected
            or audit['audited_generations'] != len(expected)):
        raise ValueError('A complete verified next-action grid is required')
    groups = []
    for cohort,history,label in GROUPS:
        cells = []
        for variant in VARIANTS:
            group = [r for r in rows if (r['cohort'],r['history_model'],r['variant']) == (cohort,history,variant)]
            if not group:
                raise ValueError('Missing the registered comparison history and model')
            counts = Counter(r['action_category'] for r in group)
            if counts.keys() - CATEGORIES.keys():
                raise ValueError('Unrecognized public action category')
            cells.append({'variant':variant,'states':len(group),'actions':dict(counts),
                          'observed_first_edit_commands':sum(counts[k] for k in KNOWN_EDITS),
                          'exact_base_commands':counts['exact_base_first_edit']})
        groups.append({'cohort':cohort,'history_model':history,'label':label,'cells':cells})

    lookup = {(r['task_id'],r['history_model'],r['variant']):r for r in rows}
    crossed = []
    for state in states:
        if (state['cohort'],state['history_model']) != ('regression',BASE):
            continue
        task = state['task_id']
        crossed.append({'task_id':task,'call':state['call'],
                        'categories':{history:{variant:lookup[task,history,variant]['action_category'] for variant in VARIANTS}
                                      for history in (BASE,CANDIDATE)}})
    original = [r for r in rows if r['variant']==r['history_model']]
    repeat = {variant:{'states':sum(r['variant']==variant for r in original),
                      'identical_output_ids':sum(r['same_as_source_output_ids'] for r in original if r['variant']==variant)}
              for variant in (BASE,CANDIDATE)}
    phase = phase_cost(data['phase'])
    phase['cost_definition'] = 'Wall time for checkpoint verification, loads, generation and cleanup'
    return {'scope':'observed_development_diagnostic_action_text_only','states':len(states),
            'audited_generations':len(rows),'groups':groups,'regression_crossed_pairs':crossed,
            'original_checkpoint_response_replay':repeat,'phase':phase,
            'hardware':hardware(data['telemetry'],data['phase']),
            'input_tokens':sum(r['input_tokens'] for r in rows),
            'generated_tokens':sum(r['generated_tokens'] for r in rows),
            'generation_seconds':sum(r['generation_seconds'] for r in rows),
            'recovered_socket_timeouts':sum(r['recovered_socket_timeout'] for r in rows),
            'new_tool_executions':audit['new_tool_executions'],'optimizer_updates':audit['optimizer_updates'],
            'final_test_opened':audit['final_test_opened'],'weights_public':audit['weights_public']}


def figures(directory,data,preparation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from .historic_report import hardware_figure

    rows = {(r['state_id'],r['variant']):r for r in data['audit']['rows']}
    states = preparation['prepared_states']
    keys = list(CATEGORIES)
    image = [[keys.index(rows[s['id'],v]['action_category']) for v in VARIANTS] for s in states]
    labels = [f"{s['task_id']} · {'base' if s['history_model']==BASE else '032'} history" for s in states]
    fig,ax = plt.subplots(figsize=(9,7.2))
    ax.imshow(image,aspect='auto',vmin=-.5,vmax=len(keys)-.5,cmap=ListedColormap([CATEGORIES[k][1] for k in keys]))
    ax.set_xticks(range(3),['Base','031','032'])
    ax.set_yticks(range(len(states)),labels,fontsize=9)
    for index,state in enumerate(states):
        for col,variant in enumerate(VARIANTS):
            row = rows[state['id'],variant]
            ax.text(col,index,'Edit match' if row['action_category'] in KNOWN_EDITS else '',
                    ha='center',va='center',fontsize=8,color='white')
    ax.set_title('One generation on each recorded pre-edit history')
    ax.set_xlabel('Checkpoint used for the next action')
    present = {r['action_category'] for r in rows.values()}
    ax.legend([Patch(facecolor=CATEGORIES[k][1]) for k in keys if k in present],
              [CATEGORIES[k][0] for k in keys if k in present],loc='upper center',
              bbox_to_anchor=(.5,-.07),ncol=2,frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/'next-action.png',dpi=160)
    plt.close(fig)
    hardware_figure(directory/'hardware.png',data['telemetry'],data['phase'])


def report(directory,preparation_path,render=True):
    data = read(directory/'analysis-input.json')
    preparation = read(preparation_path)
    value = summarize(data,preparation)
    write_json(directory/'summary.json',value)
    rows = data['audit']['rows']
    fields = ['task_id','cohort','history_model','source_call','state_id','variant','action_category',
              'input_tokens','generated_tokens','generation_seconds','recovered_socket_timeout','same_as_source_output_ids']
    write_csv(directory/'actions.csv',rows,fields)
    if render:
        figures(directory,data,preparation)

    phase = value['phase']
    shanghai = timezone(timedelta(hours=8))
    local = lambda stamp:datetime.fromisoformat(stamp).astimezone(shanghai).strftime('%Y-%m-%d %H:%M:%S')
    replay = value['original_checkpoint_response_replay']
    table = []
    for group in value['groups']:
        cells = ' | '.join(f"{arm['observed_first_edit_commands']}／{arm['states']}" for arm in group['cells'])
        table.append(f"| {group['label']} | {cells} |")
    lines = ['# 033 同历史下一步动作对照','',
             '14 段编辑前历史、42 次生成全部核验完成。每一段的三组输入 token 与源轨迹一致。下表计命令文本与该题已执行过的首次源码修改动作相同的次数。', '',
             '| 编辑前历史 | 底座 | 031 | 032 |','| --- | ---: | ---: | ---: |',*table,'',
             '![同历史逐步动作矩阵](next-action.png)','',
             f"使用轨迹自身的权重再次生成时，底座 {replay[BASE]['identical_output_ids']}／{replay[BASE]['states']}、032 {replay[CANDIDATE]['identical_output_ids']}／{replay[CANDIDATE]['states']} 与原输出 token 完全相同。不同命令可能是合法替代方案。记录动作文本类别，没有执行、评估补丁正确性或测完整任务成功率。",'',
             f"北京时间 {local(phase['started_utc'])} 至 {local(phase['finished_utc'])}，流水线 {phase['pipeline_seconds']/60:.2f} 分钟，加载 {phase['backend_loads']} 次。输入 {value['input_tokens']:,} token，生成 {value['generated_tokens']:,} token。GPU 采样 {value['hardware']['samples']} 条，采样峰值 {value['hardware']['temperature_c']['peak']:.0f}°C。",'',
             '![GPU 记录](hardware.png)','',
             '五道退步题是根据既有开发分数选出的诊断样本，四道控制题来自同一开发集。该结果只限已记录的状态和一次确定性生成，对新仓库或完整修复效果需另做对照。最终测试封存，没产生新的训练样本或模型参数。', '',
             '逐次结果见 [actions.csv](actions.csv)，每题对照和成本见 [summary.json](summary.json)，独立 token 与物理请求核验见 [evaluation-audit.json](evaluation-audit.json)。', '',
             '复算：`python -m one_gpu_lm.verified.first_edit_report --results <本目录> --preparation experiments/033-first-edit-probe/preparation.json`。可加 `--no-figures`。']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--no-figures',action='store_true')
    args = parser.parse_args()
    result = report(args.results,args.preparation,not args.no_figures)
    print(json.dumps({k:result[k] for k in ('states','audited_generations','groups','final_test_opened')},ensure_ascii=False))
