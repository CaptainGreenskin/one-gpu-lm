"""Continue an admitted historical failure with equal new budgets in both arms."""
import copy
import json
from pathlib import Path
import time

from one_gpu_lm.common.receipts import sha256, write_json
from .correction_policy import GUIDANCE
from .recovery_state import canonical, copy_tree, inventory
from .tasks import digest, relative


def initial_messages(prefix, protocol, arm):
    if arm not in {'control','correction'}:raise ValueError('Unknown arm')
    messages=copy.deepcopy(prefix['messages']);budget=protocol['evaluation_budget']
    if not messages or messages[-1]['role']!='user':raise ValueError('Complete feedback boundary required')
    note=(f"\nCONTINUATION START: The workspace is restored to the edited state represented by the preceding history. "
          f"You now have at most {budget['max_model_calls']} NEW model calls, "
          f"{budget['max_total_generated_tokens']} generated tokens in total and {budget['max_task_seconds']} seconds. "
          "Earlier calls do not consume this new continuation budget. The same restored source was separately checked "
          "with the declared visible regression; its recorded current feedback follows:\n"+prefix['standalone_feedback'])
    messages[-1]['content']+=note+(GUIDANCE if arm=='correction' else '')
    return messages


def load_snapshot(root, selected):
    snapshot=Path(root)/relative(selected['snapshot_directory'])
    if any(sha256(snapshot/name)!=value for name,value in selected['snapshot'].items()):
        raise ValueError('Admitted prefix or manifest changed')
    prefix=json.loads((snapshot/'prefix.json').read_text());manifest=json.loads((snapshot/'manifest.json').read_text())
    if inventory(snapshot/'tree')!=manifest:raise ValueError('Admitted workspace changed')
    if digest(canonical(prefix['failed_source_sha256']))!=selected['source_state_sha256']:
        raise ValueError('Admitted source identity differs')
    return snapshot,prefix,manifest


def initialize(root, task, protocol, out, arm, selected):
    out=Path(out)
    if task['split']!='training' or out.exists():raise ValueError('Fresh training continuation required')
    snapshot,prefix,manifest=load_snapshot(root,selected);out.mkdir(parents=True)
    copy_tree(snapshot/'tree',out/'workspace',manifest)
    before=json.loads((snapshot/'before.json').read_text());write_json(out/'before.json',before)
    receipt={'state_id':selected['state_id'],'snapshot':selected['snapshot'],
             'manifest_sha256':digest(canonical(inventory(out/'workspace')))}
    write_json(out/'restored-state.json',receipt)
    state={'schema_version':1,'id':task['id'],'repository':task['repository'],'family':task['family'],
           'split':'training','branch':arm,'producer_kind':'model','state_id':selected['state_id'],
           'protocol_sha256':digest(json.dumps(protocol,sort_keys=True).encode()),
           'started_epoch':time.time(),'complete':False,'boundary':'before_generation',
           'messages':initial_messages(prefix,protocol,arm),'steps':[],'backend_identities':{},
           'attempted_calls':0,'generated_tokens':0,'actor_seconds':0.,'submitted':False,
           'terminal_reason':None,'verified_source':None,'original_source_sha256':prefix['original_source_sha256'],
           'failed_source_sha256':prefix['failed_source_sha256'],
           'historical_prefix_cost':{k:prefix[k] for k in ['historical_calls','historical_generated_tokens','historical_input_tokens']}}
    write_json(out/'trajectory.json',state)
    return state


def behavior(trace,task):
    edits=[s['call'] for s in trace['steps'] if s.get('observation',{}).get('executed') is True
           and s.get('source_after') and s['source_after']!=s['source_before']]
    protected=all(s.get('protected_before') is True and s.get('protected_after') is True
                  for s in trace['steps'] if 'observation' in s)
    final=trace.get('final_source_sha256')
    changed=bool(final and final!=trace['failed_source_sha256'] and final!=trace['original_source_sha256'])
    return {'edit_calls':edits,'protected_files_intact_throughout':protected,
            'changed_from_failed_and_original_source':changed,
            'qualified_behavior':bool(trace.get('eligible_verified_trajectory') and edits and protected and changed)}
