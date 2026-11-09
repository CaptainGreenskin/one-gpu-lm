"""Tensor state fingerprints and RNG restoration; imports torch only at execution time."""
import hashlib
import random


def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def state_digest(value):
    import torch
    digest=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            tensor=v.detach().cpu().contiguous()
            digest.update(('tensor:'+str(tensor.dtype)+':'+str(tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            digest.update(b'dict')
            for key in sorted(v,key=repr):visit(key);visit(v[key])
        elif isinstance(v,(list,tuple)):
            digest.update((type(v).__name__+str(len(v))).encode())
            for item in v:visit(item)
        else:digest.update((type(v).__name__+':'+repr(v)+';').encode())
    visit(value);return digest.hexdigest()


def capture_rng():
    import torch
    return {'python':random.getstate(),'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state['python']);torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])


def states_close(left, right, *, atol, rtol):
    import torch
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.shape!=right.shape or left.dtype!=right.dtype:return False
        return bool(torch.allclose(left,right,atol=atol,rtol=rtol)) if left.is_floating_point() else torch.equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(states_close(v,right[k],atol=atol,rtol=rtol) for k,v in left.items())
    if isinstance(left,(list,tuple)):
        return type(left) is type(right) and len(left)==len(right) and all(states_close(a,b,atol=atol,rtol=rtol) for a,b in zip(left,right))
    if isinstance(left,float):return isinstance(right,float) and abs(left-right)<=atol+rtol*abs(left)
    return type(left) is type(right) and left==right
