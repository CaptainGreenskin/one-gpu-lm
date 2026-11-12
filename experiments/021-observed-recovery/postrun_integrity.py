"""Read-only post-run input checks and a byte-identical private adapter export.

This does not change any score or authorize publication. The export is private
until the independently recomputed release decision accepts it.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.checkpoints import load_checkpoint
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.common.runstate import identity_digest
from one_gpu_lm.evaluation.tasks import file_manifest
from one_gpu_lm.models.verify import verify_model


def check_legacy_inputs(run, tasks, task_sha256, tokenizer, template_kwargs):
    """Hash-check recorded function inputs/outputs without generating or grading."""
    from one_gpu_lm.common.tokenization import encode_prefix
    from one_gpu_lm.common.agent_policy import LEGACY_POLICY
    ids = [task['id'] for task in tasks]
    if len(ids) != 9 or len(set(ids)) != 9:
        raise ValueError('Legacy audit requires the exact nine unique tasks')
    fields = {'id', 'language', 'passed', 'passed_cases', 'cases', 'reason',
              'input_tokens', 'input_tokens_sha256', 'generated_tokens', 'seconds',
              'hit_generation_limit', 'agent_policy', 'context', 'dialogue',
              'packing_seconds', 'peak_allocated_bytes', 'adapter'}
    audits = {}
    for arm, adapter in [('final-base', False), ('final-selected', True)]:
        folder = Path(run)/('legacy-'+arm)
        result_file = folder/'result.json'
        value = json.loads(result_file.read_text())
        if (set(value) != {'schema_version', 'task_sha256', 'complete', 'results', 'passed'}
                or value['complete'] is not True or value['task_sha256'] != task_sha256
                or [r['id'] for r in value['results']] != ids
                or value['passed'] != sum(r['passed'] for r in value['results'])):
            raise ValueError('Legacy result identity or membership differs')
        rows = []
        for task, row in zip(tasks, value['results']):
            tokens, dialogue = encode_prefix(tokenizer,
                [{'role': 'user', 'content': task['prompt']}], template_kwargs, 'legacy')
            token_hash = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
            context = {'policy': 'full', 'raw_input_tokens': len(tokens),
                       'packed_input_tokens': len(tokens), 'dropped_rounds': 0,
                       'original_messages': 1, 'packed_messages': 1}
            if (set(row) != fields or row['adapter'] is not adapter
                    or row['language'] != task['language'] or row['cases'] != len(task['cases'])
                    or row['input_tokens'] != len(tokens) or row['input_tokens_sha256'] != token_hash
                    or row['agent_policy'] != LEGACY_POLICY or row['context'] != context
                    or row['dialogue'] != dialogue or not 0 < len(tokens) <= 8192
                    or not 0 < row['generated_tokens'] <= 512
                    or row['hit_generation_limit'] is not (row['generated_tokens'] >= 512)
                    or not 0 <= row['passed_cases'] <= row['cases']
                    or row['passed'] is not (row['passed_cases'] == row['cases'])):
                raise ValueError('Legacy recorded input, budget, metadata or score count differs')
            output = folder/'private'/(task['id']+'.txt')
            if output.is_symlink() or not output.is_file():
                raise ValueError('Missing regular private legacy output')
            rows.append({'id': task['id'], 'input_tokens_sha256': token_hash,
                         'generation_text_sha256': sha256(output), 'input_verified': True})
        audits[arm] = {'result_sha256': sha256(result_file), 'task_sha256': task_sha256,
                       'adapter': adapter, 'results': rows,
                       'scope': 'Stored inputs and output-file hashes only; no regeneration or score replay.'}
    return audits


def check_unselected_final_artifacts(run, selected):
    """Check remote disk, not only the subset later copied into a report."""
    if selected is not None:
        return None
    run = Path(run)
    paths = []
    for pattern in ['final-*', 'legacy-final-*', 'artifact-final-*',
                    'private/backend-final-*', 'private/final-*-attempt-*.log',
                    'private/legacy-final-*-attempt-*.log']:
        paths.extend(run.glob(pattern))
    if paths:
        raise ValueError('Unselected candidate has final-test execution artifacts')
    return True


def check(root, run, out):
    root, run, out = map(lambda p: Path(p).resolve(), (root, run, out))
    if out.exists():
        raise ValueError('Preserve existing integrity receipts')
    read = lambda p: json.loads(p.read_text())
    exp = root/'experiments/021-observed-recovery'
    protocol = read(exp/'protocol.json')
    controller = read(run/'controller.json')
    derived = read(root/'runs/reference-data021/derived-data.json')
    covered = derived['coverage_passed']
    selection = read(run/'selection.json') if covered else {'selected': None}
    if not covered:
        forbidden = [p for pattern in ['train-*','dev-*','artifact-dev-*','private/backend-dev-*'] for p in run.glob(pattern)]
        if forbidden: raise ValueError('Coverage rejection has training or evaluation artifacts')
    if not controller['complete'] or controller.get('error_type'):
        raise ValueError('Require the completed original campaign')
    unselected_final_absent = check_unselected_final_artifacts(run, selection['selected'])
    for name, digest in protocol['frozen_files'].items():
        if sha256(root/name) != digest:
            raise ValueError('Frozen input changed: '+name)
    snapshots = read(root/'source-snapshots.json')
    for repo, expected in snapshots.items():
        if file_manifest((root/'sources'/repo.replace('/', '--')).resolve()) != expected:
            raise ValueError('Original source snapshot changed: '+repo)
    runtime = read(root/'runtime.json')
    sites = {'historical-default': (root/'.venv-agent/lib/python3.10/site-packages').resolve()}
    sites.update({repo: root/path for repo, path in runtime.get('python_site_overrides', {}).items()})
    dependencies = {}
    for label, directory in sites.items():
        dependencies[label] = {str(p.relative_to(directory)): sha256(p)
                              for p in sorted(directory.rglob('*'))
                              if p.is_file() and '__pycache__' not in p.parts}
    for directory in sorted((root/'node-deps').iterdir()):
        if directory.is_dir():
            dependencies['node/'+directory.name] = {str(p.relative_to(directory)): sha256(p)
                for p in sorted(directory.rglob('*')) if p.is_file()}
    if dependencies != read(root/'dependency-snapshots.json'):
        raise ValueError('Agent-visible dependency contents changed')
    model = Path(runtime['model'])
    verification = verify_model(model, root/'experiments/011-coding-model-selection/weights-manifest.json',
                                read(exp/'profile-observed.json'))
    framework = read(root/'framework-verification.json')
    # Model environment may differ from the agent environment. Framework source
    # identity is checked using the recorded installed path via the agent Python.
    import subprocess
    framework_probe = "import importlib.util; print(importlib.util.find_spec('minisweagent').submodule_search_locations[0])"
    installed = Path(subprocess.check_output([str(root/'.venv-agent/bin/python'), '-c', framework_probe], text=True).strip())
    expected = framework['python_package_files']
    actual = {str(p.relative_to(installed)): sha256(p) for p in sorted(installed.rglob('*.py'))}
    if actual != expected:
        raise ValueError('Installed agent framework changed after preregistration')
    from audit_recovery import audit_recovery
    recovery = audit_recovery(root, run)
    training_audit = read(root/'training-input-audit.json')
    if not training_audit['passed'] or training_audit['data_sha256'] != {'observed':read(root/'runs/reference-data021/derived-data.json')['prepared_data_sha256']}:
        raise ValueError('Exact training data audit is missing')
    from safetensors import safe_open
    tensor_audits = {}
    for candidate_name in (['observed'] if covered else []):
        candidate = run/('train-'+candidate_name)
        checkpoint, _, _ = load_checkpoint(candidate, read(candidate/'identity.json'))
        with safe_open(checkpoint/'adapter/adapter_model.safetensors', framework='pt', device='cpu') as tensors:
            shapes = {key: list(tensors.get_slice(key).get_shape()) for key in tensors.keys()}
            dtypes = sorted({str(tensors.get_slice(key).get_dtype()) for key in tensors.keys()})
        parameters = sum(__import__('math').prod(s) for s in shapes.values())
        if dtypes != ['F32'] or parameters != read(candidate/'result.json')['trainable_parameters']:
            raise ValueError('Actual adapter tensor precision or parameter count changed')
        tensor_audits[candidate_name] = {'parameters':parameters, 'tensor_count':len(shapes), 'dtypes':dtypes,
            'file_sha256':sha256(checkpoint/'adapter/adapter_model.safetensors')}
    result = {'schema_version': 1, 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256(Path(__file__)), 'protocol_sha256': sha256(exp/'protocol.json'),
        'all_frozen_file_hashes_match': True, 'source_snapshot_contents_unchanged': True,
        'dependency_contents_unchanged': True, 'framework_contents_unchanged': True,
        'unselected_final_test_artifacts_absent': unselected_final_absent,
        'no_training_or_evaluation_artifacts': True if not covered else None,
        'base_model_verification': verification, 'training_input_audit_sha256': sha256(root/'training-input-audit.json'),
        'adapter_tensor_audits': tensor_audits, 'derivation_recovery': recovery,
        'public_task_inventory_identity_sha256': identity_digest([
            {k:t[k] for k in ['id','repository','revision','family','split','evaluation_role'] if k in t}
            for t in read(root/'private/tasks.json')['tasks']]),
        'rights_and_secret_scan_pass': True,
        'rights_scope': 'Exact reviewed source notices, fixed Apache-2.0 base LICENSE and known-pattern/private-identifier input screen; no teacher API. Not exhaustive secret detection or a legal warranty.',
        'export': {'required': selection['selected'] is not None, 'created': False},
        'weights_public': False, 'publication_authorized_by_this_receipt': False}
    if selection['selected'] is not None:
        from transformers import AutoTokenizer
        from one_gpu_lm.data.function_records import load_split
        result['legacy_input_audits'] = check_legacy_inputs(
            run, load_split(root/'private/legacy-functions.jsonl', 'test'),
            protocol['frozen_files']['private/legacy-functions.jsonl'],
            AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False),
            read(exp/'profile-observed.json')['chat_template_kwargs'])
        name = selection['selected']
        candidate = run/('train-'+name)
        checkpoint, _, manifest = load_checkpoint(candidate, read(candidate/'identity.json'))
        trained = read(candidate/'result.json')
        export = run/'private'/'adapter-export'
        if export.exists():
            raise ValueError('Preserve an existing private export')
        export.mkdir()
        files = ['adapter_config.json', 'adapter_model.safetensors']
        for name in files:
            source = checkpoint/'adapter'/name
            if sha256(source) != trained['adapter_files'][name] or sha256(source) != manifest['files']['adapter/'+name]:
                raise ValueError('Selected adapter file identity mismatch')
            shutil.copyfile(source, export/name)
            if sha256(export/name) != sha256(source):
                raise ValueError('Export changed adapter bytes')
        config = read(export/'adapter_config.json')
        if config['base_model_name_or_path'] != protocol['model_id'] or config['revision'] != protocol['revision']:
            raise ValueError('Export lacks the exact base identity')
        with safe_open(export/'adapter_model.safetensors', framework='pt', device='cpu') as tensors:
            shapes = {key: list(tensors.get_slice(key).get_shape()) for key in tensors.keys()}
        parameters = sum(__import__('math').prod(s) for s in shapes.values())
        if parameters != trained['trainable_parameters']:
            raise ValueError('Exported tensor count differs from trained adapter')
        shutil.copyfile(model/'LICENSE', export/'BASE_MODEL_LICENSE.txt')
        notice_dir=export/'training-source-notices';notice_dir.mkdir()
        for source in read(exp/'sources.json')['sources']:
            if source['split']!='train': continue
            notice=exp/source['rights_review']['publication_notice']
            if sha256(notice)!=source['license_sha256']:
                raise ValueError('Training-source notice differs from reviewed source')
            shutil.copyfile(notice,notice_dir/notice.name)
        export_files = {str(p.relative_to(export)): sha256(p) for p in sorted(export.rglob('*')) if p.is_file()}
        result['export'] = {'required': True, 'created': True, 'candidate': selection['selected'],
            'adapter_files_match_training_and_evaluated_checkpoint': True,
            'parameters': parameters, 'tensor_count': len(shapes), 'files': export_files,
            'scope': 'Byte-identical unmerged LoRA; no optimizer, raw data, base weights or generated README copied. Training already performed the exact save/reload greedy probe.'}
    write_json(out, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'run', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    check(args.root, args.run, args.out)
    print('Post-run content integrity verified; no weights published')
