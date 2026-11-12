"""Verify the disclosed pre-update recovery without changing frozen inputs."""
import json
from pathlib import Path
from one_gpu_lm.common.receipts import sha256


def audit_recovery(root, run):
    root, run = Path(root), Path(run)
    read = lambda p: json.loads(Path(p).read_text())
    exp = root / 'experiments/021-observed-recovery'
    amendment = read(exp / 'derivation-recovery.json')
    receipt_path = run / 'derivation-recovery.json'
    receipt = read(receipt_path)
    assert receipt['status'] == 'completed'
    assert receipt['original_protocol_sha256'] == amendment['original_protocol_sha256'] == sha256(exp / 'protocol.json')
    assert receipt['builder_sha256'] == amendment['corrected_builder_sha256'] == sha256(exp / 'build_data_recovery.py')
    assert receipt['amendment_sha256'] == sha256(exp / 'derivation-recovery.json')
    assert sha256(exp / 'build_data.py') == amendment['original_builder_sha256']
    original = (exp / 'build_data.py').read_text()
    corrected = (exp / 'build_data_recovery.py').read_text()
    start = corrected.index('def snapshot_prefix(episode):')
    end = corrected.index('def build(root, collection, out):', start)
    helper = corrected[start:end]
    assert 'return copy.deepcopy(prefix), progress' in helper
    reverted = (corrected[:start] + corrected[end:]).replace(
        'prefix, progress = snapshot_prefix(episode)',
        'prefix, progress = episode.progress.prepare_request(episode.messages, len(episode.steps), 12, True)')
    assert reverted == original, 'Corrective builder changed more than checked-prefix ownership'
    protocol = read(exp / 'protocol.json')
    for name, digest in protocol['frozen_files'].items(): assert sha256(root / name) == digest
    assert receipt['collection_sha256'] == amendment['collection_sha256'] == sha256(run / 'collection/collection.json')
    assert receipt['derived_data_sha256'] == sha256(root / 'runs/reference-data021/derived-data.json')
    assert not receipt['model_collection_rerun'] and not receipt['deadline_extended']
    assert receipt['optimizer_updates_before_recovery'] == receipt['model_evaluations_before_recovery'] == 0
    failed = run / 'private/controller-failed-attempt-1.json'
    previous = read(failed); current = read(run / 'controller.json')
    assert receipt['original_controller_sha256'] == sha256(failed)
    assert previous['attempt'] == 1 and previous['stage'] == 'failed' and not previous['complete']
    assert previous['finished_utc'] == amendment['failed_utc']
    for key in ['started_epoch', 'deadline_epoch']:
        assert previous[key] == current[key] == receipt['original_' + key] == amendment['original_' + key]
    history = current['recovery_history']
    assert history[0]['failed_state_sha256'] == sha256(failed)
    assert history[0]['recovery_receipt_sha256'] == sha256(receipt_path)
    manifest_path = run / 'private/failed-derivation-manifest.json'
    manifest = read(manifest_path)
    assert receipt['failed_archive_manifest_sha256'] == sha256(manifest_path)
    assert receipt['failed_archive_files'] == len(manifest)
    archive = root / 'runs/reference-data021-failed-attempt-1'
    assert {str(p.relative_to(archive)) for p in archive.rglob('*') if p.is_file()} == set(manifest)
    for name, digest in manifest.items(): assert sha256(archive / name) == digest
    published = read(root / 'private/derivation-recovery-published.json')
    assert published['commit'] == receipt['publication_commit']
    assert published['anonymous_archive_matches_local_files']['experiments/021-observed-recovery/build_data_recovery.py'] == receipt['builder_sha256']
    assert published['ci'] and all(r['status'] == 'completed' and r['conclusion'] == 'success' for r in published['ci'])
    return {'passed': True, 'audit_source_sha256': sha256(Path(__file__)),
            'recovery_receipt_sha256': sha256(receipt_path), 'builder_sha256': receipt['builder_sha256'],
            'original_frozen_files_unchanged': True, 'only_checked_prefix_ownership_changed': True,
            'failed_archive_files_verified': len(manifest), 'collection_rerun': False,
            'original_start_and_deadline_preserved': True, 'publication_commit': published['commit']}
