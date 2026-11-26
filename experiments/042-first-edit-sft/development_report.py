"""Recompute and publish a registered independent comparison using anonymous rows."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256 as hash_bytes
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from one_gpu_lm.common.receipts import sha256, write_json


BEIJING = timezone(timedelta(hours=8))
SHARED = Path('experiments/039-release-development')
FLAGS = ('audit_passed','completed_repair','no_edit','correct_without_submission','timeout_recovered')
COUNTS = ('task_number','repetition','attempted_calls','input_tokens','generated_tokens')
RESTART_COUNTS = ('reboots','attempted_calls','journaled_requests','finished_responses','unknown_response_costs',
                  'known_input_tokens','known_generated_tokens')
GPU_FIELDS = ('utilization_percent','power_w','memory_used_mib','temperature_c')


def read(path):
    return json.loads(path.read_bytes())


def project_rows(rows, families, budget, *, include_multifile=False):
    aliases = {family: f'F{index:02}' for index,family in enumerate(sorted(families), 1)}
    fields = (*FLAGS, *COUNTS, 'arm', 'first_edit_call', 'actor_seconds')
    if include_multifile:
        fields += ('multifile',)
    projected = []
    for row in rows:
        calls, first = row['attempted_calls'],row['first_edit_call']
        if (not all(type(row[key]) is bool for key in FLAGS) or row['audit_passed'] is not True
                or any(type(row[key]) is not int or row[key]<0 for key in COUNTS)
                or not 0<calls<=budget['max_model_calls'] or not 0<row['input_tokens']<=calls*budget['max_input_tokens']
                or row['generated_tokens']>budget['max_total_generated_tokens']
                or (first is not None and (type(first) is not int or not 1<=first<=calls))
                or not math.isfinite(row['actor_seconds']) or row['actor_seconds']<0
                or (include_multifile and type(row.get('multifile')) is not bool)
                or row['arm'] not in ('base','candidate')):
            raise ValueError('Unexpected audited task metrics')
        projected.append(dict({key:row[key] for key in fields}, family=aliases[row['family']]))
    return projected, aliases


def gpu_statistics(out, state):
    from candidate_restart import attempt_directory
    from one_gpu_lm.verified.recovery_report import hardware

    restarts = state['restarts']
    starts = [state['started_epoch'], *(event['resumed_epoch'] for event in restarts)]
    ends = [*(event['boot_epoch'] for event in restarts), state['finished_epoch']]
    segments, unavailable = [], []
    for attempt, (start, end) in enumerate(zip(starts, ends), 1):
        path = attempt_directory(out, attempt)/'telemetry.csv'
        collector = state['telemetry'] if attempt == len(starts) else None
        dropped_bytes = 0
        try:
            expected = restarts[attempt - 1]['prior_telemetry_sha256'] if collector is None else None
            original = path.read_bytes()
            source_hash = hash_bytes(original).hexdigest()
            if collector is None and source_hash != expected:
                raise ValueError('Telemetry differs from the recorded restart')
            execution = {'started_epoch':start,'finished_epoch':end,'telemetry':collector}
            if b'\x00' in original:
                prefix = original.rstrip(b'\x00')
                if collector is not None or not prefix.endswith(b'\n') or b'\x00' in prefix:
                    raise ValueError('NUL bytes corrupt a telemetry row')
                # A reboot may leave a zero-filled tail after the last complete row.
                # Keep the original bytes untouched for the execution audit.
                dropped_bytes = len(original) - len(prefix)
                with TemporaryDirectory(prefix='telemetry-prefix-') as temporary:
                    clean = Path(temporary)/path.name
                    clean.write_bytes(prefix)
                    checked = hardware(clean,execution)
            else:
                checked = hardware(path,execution)
            if ((collector is not None and checked['samples'] != collector['samples'])
                    or any(not math.isfinite(v) for key in GPU_FIELDS if checked[key]
                           for v in checked[key].values())):
                raise ValueError('Hardware samples differ from their collector receipt')
        except (OSError, ValueError, TypeError, KeyError, csv.Error):
            unavailable.append(attempt)
            continue
        keep = ('samples','first_to_last_seconds','maximum_gap_seconds',*GPU_FIELDS)
        segment = {**{key:checked[key] for key in keep},'attempt':attempt,
                   'telemetry_sha256':source_hash,'collector_errors':collector['errors'] if collector else None}
        if dropped_bytes:
            segment['discarded_trailing_nul_bytes'] = dropped_bytes
        segments.append(segment)
    samples = sum(segment['samples'] for segment in segments)
    summary = {'samples':samples,'segments':segments,'unavailable_attempts':unavailable,
               'sample_span_seconds':sum(segment['first_to_last_seconds'] for segment in segments)}
    for key in GPU_FIELDS:
        valid = [segment for segment in segments if segment[key] is not None]
        summary[key] = {'mean':sum(s['samples']*s[key]['mean'] for s in valid)/samples,
                        'maximum':max(s[key]['maximum'] for s in valid)} if samples else None
    return summary


def evaluation_timing(out, state):
    from candidate_restart import actor_directory

    generation = 0.0
    for row in state['runs']:
        path = actor_directory(out,row.get('attempt',1),row['run_id'])/'trajectory.json'
        payload = path.read_bytes()
        if hash_bytes(payload).hexdigest() != row['trajectory_sha256']:
            raise ValueError('Timing must come from the audited execution trace')
        durations = [step['generation']['seconds'] for step in json.loads(payload)['steps']]
        if any(not math.isfinite(seconds) or seconds < 0 for seconds in durations):
            raise ValueError('Invalid generation duration')
        generation += sum(durations)
    return {'paired_generation_seconds':generation,
            'paired_actor_seconds':sum(row['actor_seconds'] for row in state['runs']),
            'paired_grading_and_close_seconds':sum(row['seconds']-row['actor_seconds'] for row in state['runs']),
            'ready_backend_loading_seconds':sum(entry['ready_epoch']-entry['started_epoch']
                                               for entry in state['backend_lifecycles'] if 'ready_epoch' in entry)}


def capture(root, protocol, controller, audit_path, publication, openings, *, candidate='042-first-edit-sft',
            include_multifile=False):
    sys.path.insert(0, str(root/SHARED))
    from analysis import analyze
    from promotion import verify_comparison

    plan,protocol_hash,audit_hash,decision = verify_comparison(protocol,controller,audit_path,publication,openings)
    state,audit,published = read(controller),read(audit_path),read(publication)
    if (plan['candidate']!=candidate or plan['task_count']!=60 or plan['rollout_count']!=360
            or audit['controller_sha256']!=sha256(controller) or sha256(audit_path)!=audit_hash):
        raise ValueError('Require the exact completed candidate comparison')
    design = {key:plan[key] for key in ('task_count','repetitions','gate','evaluation_budget')}
    rows,aliases = project_rows(audit['rows'],plan['families'],design['evaluation_budget'],
                               include_multifile=include_multifile)
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
    gpu = gpu_statistics(controller.parent,state)
    timing = evaluation_timing(controller.parent,state)
    return {'schema_version':1,'protocol_sha256':protocol_hash,'audit_sha256':audit_hash,
            'controller_sha256':sha256(controller),'registration_commit':published['commit'],
            'started_epoch':state['started_epoch'],'finished_epoch':state['finished_epoch'],
            'backend_loads':audit['backend_loads'],'verified_requests':audit['verified_requests'],
            'interruptions':interruptions,'last_telemetry_segment':telemetry,'gpu':gpu,'timing':timing,'design':design,
            'rows':rows,'analysis':public,'passed':True,'optimizer_updates':0,'final_test_opened':False,'weights_public':False}


def render(root, directory, registration, *, experiment='042'):
    sys.path.insert(0, str(root/SHARED))
    from report import call_curves, plot, repair_scope, reproduce_analysis, write_csv

    value, plan = read(directory/'measurements.json'), read(registration)
    design = value['design']
    if (value['protocol_sha256']!=plan['private_protocol_sha256']
            or any(design[key]!=plan[key] for key in ('task_count','repetitions','gate','evaluation_budget'))
            or sorted(design['families'].values())!=plan['tasks_per_family'] or value['passed'] is not True
            or value['optimizer_updates'] or value['final_test_opened'] or value['weights_public']):
        raise ValueError('Public report does not match the registered candidate comparison')
    include_multifile = any('multifile' in row for row in value['rows'])
    rows,_ = project_rows(value['rows'],design['families'],design['evaluation_budget'],
                         include_multifile=include_multifile)
    if rows!=value['rows'] or value['verified_requests']!=sum(row['attempted_calls'] for row in rows):
        raise ValueError('Public rows contain unexpected fields or costs')
    result = reproduce_analysis(value,design)
    scope = repair_scope(rows) if include_multifile else None
    if scope is not None and scope['multi_file']['tasks'] != plan['confirmed_multifile_tasks']:
        raise ValueError('Multi-file task count differs from the registered cohort')
    write_json(directory/'summary.json',result)
    points = call_curves(rows,design['evaluation_budget']['max_model_calls'])
    write_csv(directory/'call-curves.csv',points)
    write_csv(directory/'matched-repeats.csv',sorted(rows,key=lambda r:(r['task_number'],r['repetition'],r['arm'])))
    label = f'{experiment} LoRA'
    plot(directory/'call-curves.svg',points,labels=('Base weights',label))
    start,end = (datetime.fromtimestamp(value[key],BEIJING) for key in ('started_epoch','finished_epoch'))
    base,candidate = (result['successes'][arm] for arm in ('base','candidate'))
    total = design['task_count']*design['repetitions']
    low,high = result['gain_interval_95_pp']
    status = '通过' if result['development_gate_passed'] else '未通过'
    lines = [f'# {experiment} · 独立开发对照','',
             f'底座完成 **{base}/{total}**，{label} 完成 **{candidate}/{total}**。'
             f'净变化 **{result["gain_pp"]:+.2f} 个百分点**，配对 95% 区间为 [{low:+.2f}, {high:+.2f}]。开发门槛**{status}**。',
             '',f'六个独立仓库家族的 60 道题各执行三轮。每组每题最多 12 次调用、6,144 个生成 token，'
             '实际改码后通过独立评分并在预算内提交才计为成功。','',
             f'| 指标 | 底座 | {label} |','|---|---:|---:|']
    for group,key,label in (('behaviors','no_edit','全程未改码'),
                           ('behaviors','correct_without_submission','修对未提交'),
                           ('costs','attempted_calls','调用数'),('costs','input_tokens','输入 token'),
                           ('costs','generated_tokens','生成 token')):
        lines.append(f'| {label} | {result[group]["base"][key]:,} | {result[group]["candidate"][key]:,} |')
    if scope is not None:
        lines += ['', '### 按修复范围拆分', '',
                  f'| 修复范围 | 题数 | 底座修对并提交 | {experiment} LoRA 修对并提交 |',
                  '|---|---:|---:|---:|']
        for key, name in (('single_file', '单文件'), ('multi_file', '多文件')):
            segment = scope[key]
            paired = segment['paired_repeats']
            lines.append(f'| {name} | {segment["tasks"]} | {segment["successes"]["base"]}/{paired} | '
                         f'{segment["successes"]["candidate"]}/{paired} |')
    lines += ['','![累计首次改码和完成提交曲线](call-curves.svg)','',
              f'北京时间 {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}。'
              f'流水线历时 {(end-start).total_seconds()/3600:.2f} 小时，完成配对中的 {value["verified_requests"]:,} 条生成请求已逐条核对。']
    if timing := value.get('timing'):
        lines += ['',f'完整配对累计生成用时 {timing["paired_generation_seconds"]/3600:.2f} 小时，'
                  f'独立评分与收尾共 {timing["paired_grading_and_close_seconds"]/60:.1f} 分钟。'
                  f'已就绪后端累计加载 {timing["ready_backend_loading_seconds"]/60:.1f} 分钟。']
    if gpu := value.get('gpu'):
        if gpu['samples']:
            lines += ['',f'GPU 记录 {gpu["samples"]:,} 个样本，各次执行的首末采样跨度合计 '
                      f'{gpu["sample_span_seconds"]/3600:.2f} 小时。采样平均利用率 {gpu["utilization_percent"]["mean"]:.1f}%，'
                      f'平均功耗 {gpu["power_w"]["mean"]:.0f} W，峰值显存 {gpu["memory_used_mib"]["maximum"]:.0f} MiB，'
                      f'最高温度 {gpu["temperature_c"]["maximum"]:.0f} °C。']
        if gpu['unavailable_attempts']:
            attempts = ', '.join(str(n) for n in gpu['unavailable_attempts'])
            lines += ['',f'第 {attempts} 次执行的硬件记录无法核验。']
        for segment in gpu['segments']:
            if dropped := segment.get('discarded_trailing_nul_bytes'):
                lines += ['',f'第 {segment["attempt"]} 次执行的采样文件在重启时留下 {dropped} 字节的零填充尾部。'
                          '统计使用此前的完整采样行，已审计的原始文件保留。']
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
    capture_args.add_argument('--candidate',default='042-first-edit-sft')
    capture_args.add_argument('--include-multifile',action='store_true')
    render_args = actions.add_parser('render')
    for name in ('directory','registration'):
        render_args.add_argument('--'+name,type=Path,required=True)
    render_args.add_argument('--experiment',default='042')
    args = parser.parse_args()
    output = (capture(args.root.resolve(),args.protocol,args.controller,args.audit,args.publication,args.openings,
                      candidate=args.candidate,include_multifile=args.include_multifile) if args.mode=='capture'
              else render(args.root.resolve(),args.directory,args.registration,experiment=args.experiment))
    print(json.dumps(output,allow_nan=False))


if __name__=='__main__':
    main()
