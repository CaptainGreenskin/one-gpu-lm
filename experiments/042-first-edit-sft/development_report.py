"""Recompute and publish the independent 042 comparison using anonymous rows."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys

from one_gpu_lm.common.receipts import sha256, write_json


BEIJING = timezone(timedelta(hours=8))
SHARED = Path('experiments/039-release-development')
FLAGS = ('audit_passed','completed_repair','no_edit','correct_without_submission','timeout_recovered')
COUNTS = ('task_number','repetition','attempted_calls','input_tokens','generated_tokens')
RESTART_COUNTS = ('reboots','attempted_calls','journaled_requests','finished_responses','unknown_response_costs',
                  'known_input_tokens','known_generated_tokens')


def read(path):
    return json.loads(path.read_bytes())


def project_rows(rows, families, budget):
    aliases = {family: f'F{index:02}' for index,family in enumerate(sorted(families), 1)}
    projected = []
    for row in rows:
        calls, first = row['attempted_calls'],row['first_edit_call']
        if (not all(type(row[key]) is bool for key in FLAGS) or row['audit_passed'] is not True
                or any(type(row[key]) is not int or row[key]<0 for key in COUNTS)
                or not 0<calls<=budget['max_model_calls'] or not 0<row['input_tokens']<=calls*budget['max_input_tokens']
                or row['generated_tokens']>budget['max_total_generated_tokens']
                or (first is not None and (type(first) is not int or not 1<=first<=calls))
                or not math.isfinite(row['actor_seconds']) or row['actor_seconds']<0
                or row['arm'] not in ('base','candidate')):
            raise ValueError('Unexpected audited task metrics')
        projected.append(dict({key:row[key] for key in (*FLAGS,*COUNTS,'arm','first_edit_call','actor_seconds')},
                              family=aliases[row['family']]))
    return projected, aliases


def capture(root, protocol, controller, audit_path, publication, openings):
    sys.path.insert(0, str(root/SHARED))
    from analysis import analyze
    from promotion import verify_comparison

    plan,protocol_hash,audit_hash,decision = verify_comparison(protocol,controller,audit_path,publication,openings)
    state,audit,published = read(controller),read(audit_path),read(publication)
    if (plan['candidate']!='042-first-edit-sft' or plan['task_count']!=60 or plan['rollout_count']!=360
            or audit['controller_sha256']!=sha256(controller) or sha256(audit_path)!=audit_hash):
        raise ValueError('Require the exact completed 042 comparison')
    design = {key:plan[key] for key in ('task_count','repetitions','gate','evaluation_budget')}
    rows,aliases = project_rows(audit['rows'],plan['families'],design['evaluation_budget'])
    design['families'] = {alias:plan['families'][name] for name,alias in aliases.items()}
    expected = deepcopy(decision)
    for pair in expected['pairs']:
        pair['family'] = aliases[pair['family']]
    expected['family_gain_pp'] = {aliases[name]:gain for name,gain in expected['family_gain_pp'].items()}
    public = analyze(rows,design)
    for arm in ('base','candidate'):
        old,new = expected['costs'][arm]['actor_seconds'],public['costs'][arm]['actor_seconds']
        if math.isclose(old,new,rel_tol=0,abs_tol=1e-9):
            public['costs'][arm]['actor_seconds'] = old
    if public!=expected:
        raise ValueError('Anonymization changed the measured development decision')
    interruptions = {key:audit['interruptions'][key] for key in (*RESTART_COUNTS,'retained_prefixes')}
    telemetry = {key:state['telemetry'][key] for key in ('samples','errors','thread_stopped')}
    return {'schema_version':1,'protocol_sha256':protocol_hash,'audit_sha256':audit_hash,
            'controller_sha256':sha256(controller),'registration_commit':published['commit'],
            'started_epoch':state['started_epoch'],'finished_epoch':state['finished_epoch'],
            'backend_loads':audit['backend_loads'],'verified_requests':audit['verified_requests'],
            'interruptions':interruptions,'last_telemetry_segment':telemetry,'design':design,
            'rows':rows,'analysis':public,'passed':True,'optimizer_updates':0,'final_test_opened':False,'weights_public':False}


def render(root, directory, registration):
    sys.path.insert(0, str(root/SHARED))
    from report import call_curves, plot, reproduce_analysis, write_csv

    value, plan = read(directory/'measurements.json'), read(registration)
    design = value['design']
    if (value['protocol_sha256']!=plan['private_protocol_sha256']
            or any(design[key]!=plan[key] for key in ('task_count','repetitions','gate','evaluation_budget'))
            or sorted(design['families'].values())!=plan['tasks_per_family'] or value['passed'] is not True
            or value['optimizer_updates'] or value['final_test_opened'] or value['weights_public']):
        raise ValueError('Public report does not match the registered 042 comparison')
    rows,_ = project_rows(value['rows'],design['families'],design['evaluation_budget'])
    if rows!=value['rows'] or value['verified_requests']!=sum(row['attempted_calls'] for row in rows):
        raise ValueError('Public rows contain unexpected fields or costs')
    result = reproduce_analysis(value,design)
    write_json(directory/'summary.json',result)
    points = call_curves(rows,design['evaluation_budget']['max_model_calls'])
    write_csv(directory/'call-curves.csv',points)
    write_csv(directory/'matched-repeats.csv',sorted(rows,key=lambda r:(r['task_number'],r['repetition'],r['arm'])))
    plot(directory/'call-curves.svg',points,labels=('Base weights','042 LoRA'))
    start,end = (datetime.fromtimestamp(value[key],BEIJING) for key in ('started_epoch','finished_epoch'))
    base,candidate = (result['successes'][arm] for arm in ('base','candidate'))
    total = design['task_count']*design['repetitions']
    low,high = result['gain_interval_95_pp']
    status = '通过' if result['development_gate_passed'] else '未通过'
    lines = ['# 042 · 独立开发对照','',
             f'底座完成 **{base}/{total}**，042 LoRA 完成 **{candidate}/{total}**。'
             f'净变化 **{result["gain_pp"]:+.2f} 个百分点**，配对 95% 区间为 [{low:+.2f}, {high:+.2f}]。开发门槛**{status}**。',
             '',f'六个独立仓库家族的 60 道题各执行三轮。每组每题最多 12 次调用、6,144 个生成 token，'
             '实际改码后通过独立评分并在预算内提交才计为成功。','',
             '| 指标 | 底座 | 042 LoRA |','|---|---:|---:|']
    for group,key,label in (('behaviors','no_edit','全程未改码'),
                           ('behaviors','correct_without_submission','修对未提交'),
                           ('costs','attempted_calls','调用数'),('costs','input_tokens','输入 token'),
                           ('costs','generated_tokens','生成 token')):
        lines.append(f'| {label} | {result[group]["base"][key]:,} | {result[group]["candidate"][key]:,} |')
    lines += ['','![累计首次改码和完成提交曲线](call-curves.svg)','',
              f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}。'
              f'流水线历时 {(end-start).total_seconds()/3600:.2f} 小时，完成配对中的 {value["verified_requests"]:,} 条生成请求已逐条核对。']
    overhead = value['interruptions']
    if overhead['reboots']:
        lines += ['',f'工作站重启 {overhead["reboots"]} 次。中断分支中的 {overhead["journaled_requests"]} 条请求单独记账，'
                  f'已知消耗 {overhead["known_input_tokens"]:,} 个输入 token 和 {overhead["known_generated_tokens"]:,} 个生成 token，'
                  f'另有 {overhead["unknown_response_costs"]} 条请求的消耗未知。']
    lines += ['','[匿名逐次结果](matched-repeats.csv) · [曲线数值](call-curves.csv) · [门槛和区间](summary.json) · [开发登记](../../development-registration.json)']
    (directory/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'base_successes':base,'candidate_successes':candidate,'gain_pp':result['gain_pp'],
            'development_gate_passed':result['development_gate_passed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    actions = parser.add_subparsers(dest='mode',required=True)
    capture_args = actions.add_parser('capture')
    for name in ('protocol','controller','audit','publication','openings'):
        capture_args.add_argument('--'+name,type=Path,required=True)
    render_args = actions.add_parser('render')
    for name in ('directory','registration'):
        render_args.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    output = (capture(args.root.resolve(),args.protocol,args.controller,args.audit,args.publication,args.openings)
              if args.mode=='capture' else render(args.root.resolve(),args.directory,args.registration))
    print(json.dumps(output,allow_nan=False))


if __name__=='__main__':
    main()
