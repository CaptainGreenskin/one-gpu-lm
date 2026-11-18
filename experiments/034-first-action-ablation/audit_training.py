"""Reconcile the completed pair against real tokenization and the registered loss plan."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.verified.first_action_loss import plan
from one_gpu_lm.verified.historic_inputs import select
from one_gpu_lm.verified.historic_training_audit import run as original_audit


ARMS = ('control', 'masked')


def read(path):
    return json.loads(path.read_text())


def run(root, out):
    if out.exists():
        raise ValueError('Preserve prior 034 training audit')
    private = root / 'private/first-action-ablation'
    reg = read(root / 'experiments/034-first-action-ablation/training-plan.json')
    controller = read(private / 'controller.json')
    if not controller['complete'] or controller['final_test_opened']:
        raise ValueError('Both registered training arms must be complete')
    runtime = read(root / 'private/parent-runtime.json')
    tokenizer = AutoTokenizer.from_pretrained(runtime['model'], local_files_only=True, trust_remote_code=False)
    evidence = {}
    for arm in ARMS:
        protocol_path = private / arm / 'protocol.json'
        protocol = read(protocol_path)
        if protocol['loss_ablation'] != {'target_loss_weight':reg['loss_weights'][arm], 'target_receipt':reg['target_receipt']}:
            raise ValueError('Arm loss treatment differs from registry')
        directory = root / protocol['run_directory']
        rec = read(directory / 'result.json')
        current = original_audit(root, protocol_path, directory / 'integrity.json')
        rows = [json.loads(s) for s in (root / protocol['data_path']).read_text().splitlines() if s]
        profile = read(root / protocol['profile_path'])
        selected, _ = select(root, protocol, profile, tokenizer, rows, read(root / protocol['export_manifest']))
        targets, receipt = plan(rows, selected, protocol['loss_ablation'])
        curve = read(directory / 'curve.json')
        actual = [r['loss_weight'] for r in curve]
        if (not current['passed'] or receipt != rec['loss_ablation']
                or actual != [int(r['id'] not in targets) for r in selected]
                or sum(r['supervised_tokens'] * w for r, w in zip(curve, actual)) != receipt['effective_supervised_tokens']):
            raise ValueError('Unexpected per-record loss weighting or training integrity failure')
        evidence[arm] = {'protocol_sha256':sha256(protocol_path),'weights_sha256':current['adapter_files_sha256'],
                         'result_sha256':sha256(directory / 'result.json'),'integrity_sha256':sha256(directory / 'integrity.json'),
                         'training':receipt, 'checks':current['checks'],'updates':rec['optimizer_steps'],
                         'input_tokens':rec['input_tokens'],'initial_adapter_sha256':rec['initial_adapter_sha256'],
                         'started_utc':rec['started_utc'],'finished_utc':rec['finished_utc'],
                         'pipeline_seconds':rec['finished_epoch'] - rec['started_epoch'],
                         'training_seconds':rec['training_seconds'],'peak_reserved_bytes':rec['peak_reserved_bytes']}
    a,b=(evidence[n] for n in ARMS)
    checks = {'same_token_ids_and_labels':a['training']['encoding_sha256']==b['training']['encoding_sha256'],
              'same_initial_adapter':a['initial_adapter_sha256']==b['initial_adapter_sha256'],
              'same_registered_steps':a['updates']==b['updates']==reg['optimizer_updates_per_arm'],
              'same_input_tokens':a['input_tokens']==b['input_tokens'],
              'different_trained_adapters':a['weights_sha256']['adapter_model.safetensors']!=b['weights_sha256']['adapter_model.safetensors']}
    result={'schema_version':1,'passed':all(checks.values()),'checks':checks,'arms':evidence,'final_test_opened':False}
    write_json(out,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(run(a.root.resolve(),a.out.resolve())))
