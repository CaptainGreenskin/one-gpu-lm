"""Verify and restore the two files of a pinned, unmerged PEFT LoRA checkpoint."""
import argparse
from collections import Counter
import json
from math import prod
from pathlib import Path
import re
import shutil
import tempfile

from one_gpu_lm.common.receipts import sha256


FILES = ('adapter_config.json', 'adapter_model.safetensors')


def verify_adapter(directory, profile, expected_hashes):
    """Check exact bytes, base/revision, and readable paired tensors on CPU."""
    import numpy as np
    from safetensors import safe_open

    directory = Path(directory)
    if set(expected_hashes) != set(FILES):
        raise ValueError('Expected digests for the two adapter files')
    for name in FILES:
        path = directory/name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected_hashes[name]:
            raise ValueError(f'Adapter file does not match the audited checkpoint: {name}')
    config = json.loads((directory/FILES[0]).read_text(encoding='utf-8'))
    expected_lora = profile['lora']
    if (config['base_model_name_or_path'] != profile['model_id'] or config['revision'] != profile['revision']
            or config['peft_type'] != 'LORA' or config['task_type'] != 'CAUSAL_LM'
            or any(config[key] != value for key,value in expected_lora.items())):
        raise ValueError('Adapter configuration differs from the evaluated base or LoRA profile')

    target_modules = expected_lora['target_modules']
    rank = expected_lora['r']
    pairs = {}
    dtypes = Counter()
    with safe_open(directory/FILES[1], framework='numpy', device='cpu') as weights:
        if weights.metadata() != {'format':'pt'}:
            raise ValueError('Expected a PEFT PyTorch safetensors checkpoint')
        for name in weights.keys():
            module, matrix, suffix = name.rsplit('.', 2)
            if suffix != 'weight' or matrix not in ('lora_A','lora_B'):
                raise ValueError('Unexpected tensor in the adapter checkpoint')
            target = (bool(re.fullmatch(target_modules,module)) if isinstance(target_modules,str)
                      else any(module == item or module.endswith('.'+item) for item in target_modules))
            if not target:
                raise ValueError('LoRA tensor targets a module outside the evaluated profile')
            tensor = weights.get_tensor(name)
            if tensor.ndim != 2 or not tensor.size or tensor.dtype.kind != 'f' or not np.isfinite(tensor).all():
                raise ValueError('Invalid or nonfinite LoRA matrix')
            pairs.setdefault(module,{})[matrix] = tensor.shape
            dtypes[str(tensor.dtype)] += tensor.size
    if not pairs or any(set(pair) != {'lora_A','lora_B'}
                        or pair['lora_A'][0] != rank or pair['lora_B'][1] != rank
                        for pair in pairs.values()):
        raise ValueError('Expected paired LoRA matrices with the registered rank')
    return {'model_id':profile['model_id'], 'revision':profile['revision'],
            'adapter_files_sha256':dict(expected_hashes), 'cpu_tensor_decoding_verified':True,
            'tensors':2*len(pairs), 'targeted_modules':len(pairs), 'rank':rank,
            'parameters':sum(prod(shape) for pair in pairs.values() for shape in pair.values()),
            'parameters_by_dtype':dict(dtypes)}


def restore_adapter(source, destination, profile, expected_hashes):
    """Copy the verified LoRA into a fresh directory and recheck the written bytes."""
    source,destination = Path(source),Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Choose a fresh adapter restore directory')
    verify_adapter(source,profile,expected_hashes)
    with tempfile.TemporaryDirectory(prefix='.adapter-restore-',dir=destination.parent) as temporary:
        staged = Path(temporary)/'adapter'
        staged.mkdir(mode=0o700)
        for name in FILES:
            shutil.copyfile(source/name,staged/name)
        receipt = verify_adapter(staged,profile,expected_hashes)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Choose a fresh adapter restore directory')
        staged.rename(destination)
    return {**receipt,'restored_to_fresh_directory':True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',required=True,type=Path)
    p.add_argument('--profile',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path,help='Audited JSON with adapter_files_sha256')
    p.add_argument('--restore-to',type=Path)
    a = p.parse_args()
    profile = json.loads(a.profile.read_text(encoding='utf-8'))
    published = json.loads(a.receipt.read_text(encoding='utf-8'))['adapter_files_sha256']
    hashes = {name:published[name] for name in FILES}
    result = (restore_adapter(a.adapter,a.restore_to,profile,hashes) if a.restore_to
              else verify_adapter(a.adapter,profile,hashes))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
