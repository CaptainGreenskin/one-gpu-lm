"""Compare continuous training with a fresh-process resume under one fixed configuration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from one_gpu_lm.common.receipts import write_json
from one_gpu_lm.common.runstate import atomic_json
from one_gpu_lm.common.checkpoints import load_checkpoint
from .config import load_config
from .state import states_close


def compare(reference,resumed):
    a,b=[json.loads((p/'result.json').read_text()) for p in [reference,resumed]]
    if a['status']!='completed' or b['status']!='completed':raise ValueError('Incomplete comparison')
    identities=[json.loads((p/'identity.json').read_text()) for p in [reference,resumed]]
    if identities[0]!=identities[1]:raise ValueError('Comparison identities differ')
    ac,at,_=load_checkpoint(reference,identities[0]);bc,bt,_=load_checkpoint(resumed,identities[1])
    if at['order_sha256']!=bt['order_sha256']:raise ValueError('Sample orders differ')
    import torch
    torch.set_num_threads(2)
    from safetensors.torch import load_file
    left=load_file(ac/'adapter/adapter_model.safetensors',device='cpu')
    right=load_file(bc/'adapter/adapter_model.safetensors',device='cpu')
    keys_match=set(left)==set(right)
    atol,rtol=1e-7,1e-6
    max_error=max(float((left[k]-right[k]).abs().max()) for k in left) if keys_match else None
    close=keys_match and all(torch.allclose(left[k],right[k],atol=atol,rtol=rtol) for k in left)
    loss_equal=all(x['loss']==y['loss'] for x,y in zip(at['history'],bt['history']))
    numeric_losses=len(at['history'])==len(bt['history']) and all(abs(x['loss']-y['loss'])<=atol+rtol*abs(x['loss']) for x,y in zip(at['history'],bt['history']))
    events=[json.loads(line) for line in (resumed/'events.jsonl').read_text().splitlines()]
    restores=[e for e in events if e['event']=='state_restored']
    restored_ok=bool(restores) and all(all(e['checks'].values()) for e in restores)
    optimizer_states=[torch.load(path/'state.pt',map_location='cpu',weights_only=True)['optimizer'] for path in [ac,bc]]
    optimizer_close=states_close(*optimizer_states,atol=1e-10,rtol=1e-6)
    checks={'adapter_numerically_equal':close,'per_step_losses_equal_within_tolerance':numeric_losses,
        'sample_order_equal':at['order_sha256']==bt['order_sha256'],'steps_equal':a['steps']==b['steps'],
        'saved_reload_probes_pass':a['saved_reload_greedy_ids_equal'] and b['saved_reload_greedy_ids_equal'],
        'final_greedy_probe_equal':a['probe_ids_sha256']==b['probe_ids_sha256'],
        'loaded_optimizer_rng_and_adapter_match_checkpoint':restored_ok,
        'final_optimizer_state_equal_within_tolerance':optimizer_close,
        'final_rng_state_exact':a['rng_state_sha256']==b['rng_state_sha256']}
    return {'schema_version':1,'passed':all(checks.values()),'checks':checks,'steps':a['steps'],
        'tolerance':{'atol':atol,'rtol':rtol},'adapter_max_absolute_error':max_error,
        'adapter_state_bitwise_equal':a['adapter_state_sha256']==b['adapter_state_sha256'],
        'optimizer_state_bitwise_equal':a['optimizer_state_sha256']==b['optimizer_state_sha256'],
        'optimizer_tolerance':{'atol':1e-10,'rtol':1e-6},
        'per_step_losses_bitwise_equal':loss_equal,'restored_steps':[e['step'] for e in restores],
        'purpose':'training-recovery-integrity-not-coding-quality','raw_examples_public':False,'weights_public':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--interrupt-after',type=int,required=True)
    args=p.parse_args();args.out=args.out.resolve()
    config=load_config(args.config)
    if not 0<args.interrupt_after<config['run']['max_steps']:
        raise ValueError('Interruption must be before the final update')
    if args.out.exists():raise ValueError('Preserve the previous recovery experiment')
    args.out.mkdir(parents=True);(args.out/'private').mkdir(mode=0o700)
    commands=[('continuous',[]),('resumed',['--stop-after-step',str(args.interrupt_after)]),('resumed',['--resume'])]
    for index,(name,extra) in enumerate(commands):
        atomic_json(args.out/'status.json',{'stage':name,'phase':index+1})
        with (args.out/'private'/f'phase-{index+1}.log').open('w') as log:
            completed=subprocess.run([sys.executable,'-m','one_gpu_lm','train','--config',str(args.config.resolve()),
                '--out',str(args.out/name),*extra],stdout=log,stderr=log,
                timeout=config['run']['max_attempt_seconds']+1200)
        expected=75 if index==1 else 0
        if completed.returncode!=expected:
            atomic_json(args.out/'status.json',{'stage':'failed','phase':index+1,'returncode':completed.returncode})
            raise RuntimeError('Training phase failed; inspect its private log before resuming')
    result=compare(args.out/'continuous',args.out/'resumed')
    write_json(args.out/'comparison.json',result)
    atomic_json(args.out/'status.json',{'stage':'completed','passed':result['passed']})
    if not result['passed']:raise ValueError('Recovery integrity comparison failed')
    print('Continuous / fresh-process recovery comparison passed')


if __name__=='__main__':main()
