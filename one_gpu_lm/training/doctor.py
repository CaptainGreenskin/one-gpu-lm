"""Validate pinned weights, data lineage, package versions, and single-GPU headroom."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess

from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import gpu_lock_path, single_writer, utc
from one_gpu_lm.models.verify import verify_model
from .config import load_config, read_training_data


def source_files():
    root=Path(__file__).resolve().parents[1]
    return {'one_gpu_lm/'+str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in {'.py','.mjs'} and '__pycache__' not in p.parts}


def inspect(config, output_parent):
    if platform.system()!='Linux' or os.environ.get('CUDA_VISIBLE_DEVICES','0')!='0':
        raise ValueError('Validated v1 runtime requires Linux and physical CUDA GPU 0 only')
    expected=json.loads(config['paths']['environment_lock'].read_text())
    packages={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    if any(packages.get(k.lower())!=v for k,v in expected['packages'].items()):
        raise ValueError('Model environment differs from its pinned package versions')
    if platform.python_version_tuple()[:2]!=tuple(expected['python_major_minor']):
        raise ValueError('Python version differs from the validated environment')
    rows,_=read_training_data(config)
    usage=shutil.disk_usage(output_parent)
    estimated_checkpoints=math.ceil(config['run']['max_steps']/config['run']['checkpoint_every'])+3
    required_gib=max(config['run']['minimum_free_disk_gib'],estimated_checkpoints*1.4+2)
    if usage.free/2**30<required_gib:
        raise ValueError('Insufficient disk headroom for retained adapter/optimizer checkpoints')
    active=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(line.strip().isdigit() and int(line.strip())!=os.getpid() for line in active.splitlines()):
        raise ValueError('GPU has another compute process; leave its job untouched')
    raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],text=True)
    name,total,free,driver=[v.strip() for v in raw.strip().split(',')]
    if float(free)/1024<config['run']['minimum_free_gpu_gib']:
        raise ValueError('Insufficient free GPU memory for this recipe')
    verified=verify_model(config['paths']['model_dir'],config['paths']['weights_manifest'],config['profile'])
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise ValueError('One visible BF16 CUDA device is required')
    if torch.version.cuda!=expected['torch_cuda']:
        raise ValueError('Unexpected PyTorch CUDA build')
    probe=torch.ones((16,16),device='cuda',dtype=torch.bfloat16)
    if not bool(torch.isfinite(probe@probe).all()):
        raise ValueError('CUDA BF16 smoke failed')
    del probe
    torch.cuda.empty_cache()
    environment={'python':platform.python_version(),'packages':dict(sorted(packages.items())),
        'torch_cuda':torch.version.cuda,'gpu_name':name,'compute_capability':list(torch.cuda.get_device_capability()),
        'driver_version':driver,'gpu_total_mib':float(total),
        'numerics':{'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cudnn_benchmark':torch.backends.cudnn.benchmark}}
    identity={'schema_version':1,'profile':config['profile'],'run':config['run'],
        'input_hashes':config['input_hashes'],'environment':environment,'source_files':source_files()}
    return {'status':'passed','utc':utc(),'identity':identity,'model':verified,
        'data_records':len(rows),'free_gpu_mib_before_load':float(free),
        'free_disk_bytes':usage.free,'required_disk_gib':required_gib,
        'limitations':['Resource locks cover cooperating processes under one UID; they are not a cluster scheduler.',
                       'Source/license metadata records an operator review; software does not determine ownership.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();config=load_config(args.config)
    if args.out.exists():
        raise ValueError('Preserve the previous preflight receipt')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with single_writer(gpu_lock_path()):
        result=inspect(config,args.out.parent)
    write_json(args.out,result)
    print('Preflight passed: pinned content, data lineage, environment, disk and CUDA')


if __name__=='__main__':
    main()
