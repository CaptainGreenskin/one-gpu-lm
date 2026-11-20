"""Compare fixed greedy outputs from independently loaded copies of an adapter."""
import argparse
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import re
import subprocess
import tempfile
from uuid import uuid4

from one_gpu_lm.common.agent_policy import system_prompt
from one_gpu_lm.common.receipts import sha256
from one_gpu_lm.common.runstate import gpu_lock_path, identity_digest, single_writer
from one_gpu_lm.models import hf_runtime
from one_gpu_lm.models.adapter_bundle import FILES, verify_adapter
from one_gpu_lm.models.hf_runtime import Runtime
from one_gpu_lm.models.verify import verify_model
from one_gpu_lm.verified.runtime_identity import PACKAGES, capture as runtime_identity


POLICY = {'format':'native_roundtrip','context':'full','progress':'edit_checkpoint_v1'}
PROBES = (
    ('literal_output','Use the Bash tool to print the literal text reload-check.'),
    ('integer_sum','Use the Bash tool to print the sum of 17 and 25 using Python.'),
)
MAX_INPUT_TOKENS = 8192
MAX_NEW_TOKENS = 16


def probe_requests():
    return [
        {'name':name,'request':{'messages':[{'role':'system','content':system_prompt(POLICY)},
                                           {'role':'user','content':prompt}],
                                'agent_policy':dict(POLICY),'max_input_tokens':MAX_INPUT_TOKENS,
                                'max_new_tokens':MAX_NEW_TOKENS}}
        for name,prompt in PROBES
    ]


def require_idle_gpu():
    processes = subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    if processes.strip():
        raise RuntimeError('The CUDA evaluation device is occupied')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def capture(model, adapter, profile_path, manifest_path, adapter_receipt):
    """Load pinned local files and record token fingerprints from a new model instance."""
    profile_bytes = Path(profile_path).read_bytes()
    profile = json.loads(profile_bytes)
    files = load_json(adapter_receipt)['adapter_files_sha256']
    expected = {name:files[name] for name in FILES}
    requests = probe_requests()
    evidence = []
    with single_writer(gpu_lock_path()):
        require_idle_gpu()
        base = verify_model(model,manifest_path,profile)
        verified = verify_adapter(adapter,profile,expected)
        require_idle_gpu()
        environment = runtime_identity()
        with tempfile.TemporaryDirectory(prefix='one-gpu-lm-reload-') as temporary:
            model_runtime = Runtime(Path(model),profile,Path(temporary),Path(adapter))
            for probe in requests:
                result = model_runtime.generate(probe['request'])
                if result.get('error') or result.get('adapter') is not True:
                    raise ValueError('The reload probe did not generate with the verified adapter')
                tokens = result['ids']
                if (result['generated_tokens']!=len(tokens) or not 1<=len(tokens)<=MAX_NEW_TOKENS
                        or any(type(token) is not int or token<0 for token in tokens)
                        or not 1<=result['input_tokens']<=MAX_INPUT_TOKENS
                        or not re.fullmatch(r'[0-9a-f]{64}',result['input_tokens_sha256'])):
                    raise ValueError('Invalid token receipt for the fixed probe')
                evidence.append({'name':probe['name'],'input_tokens':result['input_tokens'],
                                 'input_tokens_sha256':result['input_tokens_sha256'],
                                 'generated_tokens':len(tokens),'generated_ids_sha256':identity_digest(tokens)})
            del model_runtime
    if any(sha256(Path(adapter)/name)!=expected[name] for name in FILES):
        raise ValueError('Adapter bytes changed during the reload probe')
    return {'schema_version':1,'capture_id':uuid4().hex,
            'configuration':{'model_id':base['model_id'],'base_revision':base['revision'],
                             'base_manifest_sha256':base['manifest_sha256'],'profile_sha256':hash_bytes(profile_bytes).hexdigest(),
                             'adapter_files_sha256':verified['adapter_files_sha256'],
                             'runtime_sha256':sha256(Path(hf_runtime.__file__)),
                             'probe_definition_sha256':identity_digest(requests),'software':environment},
            'probes':evidence}


def compare(before, after):
    required_hashes = ('base_manifest_sha256','profile_sha256','runtime_sha256','probe_definition_sha256')
    for receipt in (before,after):
        if (receipt['schema_version']!=1 or not re.fullmatch(r'[0-9a-f]{32}',receipt['capture_id'])
                or receipt['configuration']['probe_definition_sha256']!=identity_digest(probe_requests())
                or [p['name'] for p in receipt['probes']]!=[name for name,_ in PROBES]):
            raise ValueError('Expected a complete fixed-probe receipt')
        config = receipt['configuration']
        if (set(config)!={'model_id','base_revision','adapter_files_sha256','software',*required_hashes}
                or not config['model_id'] or not re.fullmatch(r'[0-9a-f]{40}',config['base_revision'])
                or set(config['adapter_files_sha256'])!=set(FILES)
                or set(config['software'])!={'python','packages'}
                or set(config['software']['packages'])!=set(PACKAGES)):
            raise ValueError('Expected the complete verified model configuration')
        hashes = [config[name] for name in required_hashes] + list(config['adapter_files_sha256'].values())
        if any(not re.fullmatch(r'[0-9a-f]{64}',digest) for digest in hashes):
            raise ValueError('Expected a full hash for each verified model artifact')
        for probe in receipt['probes']:
            if (not 1<=probe['input_tokens']<=MAX_INPUT_TOKENS or not 1<=probe['generated_tokens']<=MAX_NEW_TOKENS
                    or any(not re.fullmatch(r'[0-9a-f]{64}',probe[key])
                           for key in ('input_tokens_sha256','generated_ids_sha256'))):
                raise ValueError('Expected complete token fingerprints for every probe')
    if before['capture_id']==after['capture_id']:
        raise ValueError('Provide two independent reload captures')
    if before['configuration']!=after['configuration']:
        raise ValueError('The base, adapter, software or probe configuration changed')
    mismatches = [left['name'] for left,right in zip(before['probes'],after['probes']) if left!=right]
    return {'configuration':before['configuration'],'save_reload_greedy_match':not mismatches,
            'compared_probes':len(PROBES),'mismatched_probes':mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='operation',required=True)
    load = modes.add_parser('capture')
    for key in ('model','adapter','profile','base-manifest','adapter-receipt','out'):
        load.add_argument('--'+key,required=True,type=Path)
    match = modes.add_parser('compare')
    for key in ('before','after','out'):
        match.add_argument('--'+key,required=True,type=Path)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError('Preserve the existing probe receipt')
    if args.operation=='capture':
        result = capture(args.model,args.adapter,args.profile,args.base_manifest,args.adapter_receipt)
    else:
        result = compare(load_json(args.before),load_json(args.after))
        result.update(before_sha256=sha256(args.before),after_sha256=sha256(args.after))
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(result))
    if args.operation=='compare' and not result['save_reload_greedy_match']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
