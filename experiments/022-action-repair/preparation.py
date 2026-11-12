"""Validate inherited, still-unopened heldout tasks and exact CPU-derived inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.receipts import sha256,write_json


def verify(root):
    root=Path(root).resolve();read=lambda p:json.loads(Path(p).read_text())
    exp=root/'experiments/022-action-repair';old_exp=root/'experiments/021-observed-recovery'
    spec=importlib.util.spec_from_file_location('inherited_preparation021',old_exp/'preparation.py')
    parent_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent_module)
    original=parent_module.verify(root)
    parent=Path(read(root/'private/parent021.json')['root'])
    for name in ['private/tasks.json','private/fresh-tasks.json','source-snapshots.json',
                 'dependency-snapshots.json','runs/admission/admission.json']:
        if sha256(root/name)!=sha256(parent/name):raise ValueError('Inherited 021 inputs changed: '+name)
    controller=read(parent/'runs/release021/controller.json');selection=read(parent/'runs/release021/selection.json')
    integrity=read(parent/'runs/release021/postrun-integrity.json')
    if not controller['complete'] or selection['selected'] is not None or integrity['unselected_final_test_artifacts_absent'] is not True:
        raise ValueError('021 fresh tasks were not preserved unopened')
    for pattern in ['final-*','legacy-final-*','artifact-final-*','private/backend-final-*']:
        if list((parent/'runs/release021').glob(pattern)):raise ValueError('Parent final-test model artifacts exist')
    for name in ['controller','integrity']:
        source=parent/'runs/release021'/('controller.json' if name=='controller' else 'postrun-integrity.json')
        if sha256(source)!=sha256(root/'private'/('parent021-'+name+'.json')):
            raise ValueError('Parent completion evidence changed')
    if sha256(exp/'profile-action.json')!=sha256(old_exp/'profile-observed.json'):
        raise ValueError('The shared training/inference profile must remain identical')
    plan=read(exp/'preparation-plan.json');prior=read(old_exp/'preparation-plan.json')
    for key in ['max_gpu_pipeline_wall_seconds','max_optimizer_updates_per_candidate','maximum_sequence_tokens',
                'max_model_calls','max_generated_tokens_per_call','max_total_generated_tokens','max_input_tokens','max_task_seconds']:
        if plan['budget'][key]!=prior['budget'][key]:raise ValueError('Shared budget changed')
    if plan['selection']!=prior['selection'] or plan['release_gate']!=prior['release_gate']:
        raise ValueError('Selection or release gate relaxed')
    if plan['candidates']!=[{'name':'action','learning_rate':2e-6,'data_variant':'action'}]:
        raise ValueError('One fixed new candidate only')
    data=read(root/'runs/reference-data022/derived-data.json');audit=read(root/'training-data-audit.json')
    screen=read(root/'training-input-audit.json')
    if not data['coverage_passed'] or not audit['passed'] or not screen['passed']:
        raise ValueError('CPU coverage, executed evidence or input screening failed')
    if (audit['audit_source_sha256']!=sha256(exp/'audit_data.py') or audit['builder_sha256']!=sha256(exp/'build_data.py')
            or audit['derived_data_sha256']!=sha256(root/'runs/reference-data022/derived-data.json')
            or screen['audit_source_sha256']!=sha256(exp/'audit_inputs.py')
            or screen['data_sha256']!={'action':data['prepared_data_sha256']}):
        raise ValueError('Actual data audit source or input identity differs')
    for leaf,key in [('train.jsonl','prepared_data_sha256'),('receipt.json','prepared_receipt_sha256'),('selection.json','selection_sha256')]:
        if sha256(root/'runs/prepared-data022/action'/leaf)!=data[key]:raise ValueError('Prepared inputs changed')
    return dict(original,experiment='022-action-repair',inherited_fresh_model_tasks_unopened=True,
        parent_protocol_sha256=sha256(old_exp/'protocol.json'),derived_data_sha256=sha256(root/'runs/reference-data022/derived-data.json'),
        data_audit_sha256=sha256(root/'training-data-audit.json'),input_audit_sha256=sha256(root/'training-input-audit.json'),
        actual_training_records=330,common_parent_records=275,replaced_records=data['replaced_records'],
        cpu_preparation_outside_model_pipeline_budget=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();value=verify(a.root);a.out.parent.mkdir(parents=True,exist_ok=True);write_json(a.out,value);print(json.dumps(value))
