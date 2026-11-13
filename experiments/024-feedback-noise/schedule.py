"""Counterbalanced schedule fixed independently of all 024 outcomes."""
MODELS=['base','reference020','action022']
CONDITIONS=['baseline','extended']
REPEATS=3

def blocks():
    return [{'repeat':r+1,'model':MODELS[(r+j)%3], 'index':r*3+j}
            for r in range(REPEATS) for j in range(3)]

def task_schedule(ids,block):
    if len(ids)!=15 or len(set(ids))!=15:raise ValueError('Exactly 15 unique development tasks required')
    result=[]
    for i,ident in enumerate(ids):
        order=CONDITIONS if (i+block['repeat']+MODELS.index(block['model']))%2 else list(reversed(CONDITIONS))
        for condition in order:result.append({'id':ident,'condition':condition})
    return result
