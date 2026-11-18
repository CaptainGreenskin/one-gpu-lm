"""Derive completion and verification from executed actions, not summary flags."""
from .submission import SUBMIT


def executed_completion(trace, task, final_source):
    submitted_at=[];verified=None
    for index,step in enumerate(trace['steps']):
        observation=step.get('observation')
        if not observation or observation['returncode']!=0:continue
        if step.get('command')==task['regression_command'] and 'source_after' in step:
            verified=step['source_after']
        if step.get('command')==SUBMIT:submitted_at.append(index)
    submitted=(submitted_at==[len(trace['steps'])-1] and trace['terminal_reason']=='submitted')
    return {'submitted':submitted,'agent_verified_final_source':verified is not None and verified==final_source}
