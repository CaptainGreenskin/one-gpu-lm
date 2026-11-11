"""Read-only post-run input checks and a byte-identical private adapter export.

This does not change any score or authorize publication. The export is private
until the independently recomputed release decision accepts it.
"""
import argparse
from datetime import datetime, timezone
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
    exp = root/'experiments/020-repair-curriculum'
    protocol = read(exp/'protocol.json')
    controller = read(run/'controller.json')
    selection = read(run/'selection.json')
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
                                read(exp/'profile-repair.json'))
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
    training_audit = read(root/'training-input-audit.json')
    if not training_audit['passed'] or training_audit['data_sha256'] != protocol['prepared_data_sha256']:
        raise ValueError('Exact training data audit is missing')
    from safetensors import safe_open
    tensor_audits = {}
    for candidate_name in ['repair', 'recovery']:
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
        'base_model_verification': verification, 'training_input_audit_sha256': sha256(root/'training-input-audit.json'),
        'adapter_tensor_audits': tensor_audits,
        'public_task_inventory_identity_sha256': identity_digest([
            {k:t[k] for k in ['id','repository','revision','family','split','evaluation_role'] if k in t}
            for t in read(root/'private/tasks.json')['tasks']]),
        'rights_and_secret_scan_pass': True,
        'rights_scope': 'Exact reviewed source notices, fixed Apache-2.0 base LICENSE and known-pattern/private-identifier input screen; no teacher API. Not exhaustive secret detection or a legal warranty.',
        'export': {'required': selection['selected'] is not None, 'created': False},
        'weights_public': False, 'publication_authorized_by_this_receipt': False}
    if selection['selected'] is not None:
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
        shutil.copytree(exp/'notices', export/'training-source-notices')
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
