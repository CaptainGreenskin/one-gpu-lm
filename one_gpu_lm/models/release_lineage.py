"""Bind the reviewed training inputs and recipe to the exact promoted LoRA."""
from dataclasses import dataclass
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import re

from one_gpu_lm.models.adapter_bundle import FILES


def read_digest(path):
    payload = Path(path).read_bytes()
    return json.loads(payload),hash_bytes(payload).hexdigest()


@dataclass(frozen=True)
class TrainingEvidence:
    review: dict
    review_sha256: str
    protocol: dict
    profile: dict
    integrity: dict
    lineage: dict


def audited_training(review_path, protocol_path, profile_path, integrity_path):
    """Validate completed training evidence independently of any final-test decision."""
    review,review_hash = read_digest(review_path)
    protocol,protocol_hash = read_digest(protocol_path)
    profile,profile_hash = read_digest(profile_path)
    integrity,integrity_hash = read_digest(integrity_path)
    if (review['training_integrity_sha256'] != integrity_hash
            or integrity['protocol_sha256'] != protocol_hash):
        raise ValueError('Training protocol or integrity receipt does not belong to the reviewed candidate')
    if (review['audited_checkpoint_matches'] is not True
            or review['eligible_for_existing_scope_of_development'] is not True
            or review['source_snapshots_match'] is not True
            or review['all_source_notices_and_lineage_reviewed_for_training'] is not True
            or any(document['final_test_opened'] is not False or document['weights_public'] is not False
                   for document in (review, protocol, integrity))):
        raise ValueError('Require audited training provenance from before access to the final test')

    if ((profile['model_id'],profile['revision']) != (review['model_id'],review['revision'])
            or integrity['adapter_files_sha256'] != review['checkpoint_files_sha256']
            or not set(FILES) <= review['checkpoint_files_sha256'].keys()):
        raise ValueError('Training profile or checkpoint differs from the audited adapter')

    frozen = protocol['frozen_files']
    expected = {protocol['profile_path']: profile_hash, protocol['data_path']: review['actual_training_data_sha256'],
                protocol['sources_path']: review['source_manifest_sha256']}
    if len(expected) != 3 or any(frozen.get(path) != digest or not re.fullmatch(r'[0-9a-f]{64}', digest)
                                 for path, digest in expected.items()):
        raise ValueError('Training profile, data and source manifest must match the frozen training protocol')
    updates = integrity['optimizer_updates']
    seed = profile['training']['seed']
    experiment = review['experiment']
    if (integrity['passed'] is not True or integrity['audit_model_generations'] != 0
            or integrity['audit_optimizer_updates'] != 0 or type(updates) is not int or updates < 1
            or type(seed) is not int or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', experiment)):
        raise ValueError('Require the recorded optimizer updates and seed of one audited training run')
    lineage = {
        'experiment': experiment, 'protocol_sha256': protocol_hash, 'profile_sha256': profile_hash,
        'integrity_sha256': integrity_hash, 'training_data_sha256': expected[protocol['data_path']],
        'source_manifest_sha256': expected[protocol['sources_path']], 'optimizer_updates': updates,
        'seed': seed,
    }
    return TrainingEvidence(review,review_hash,protocol,profile,integrity,lineage)


def reviewed_training(permission, review_path, protocol_path, profile_path, integrity_path):
    if permission['approved'] is not True or permission['stage'] != 'final_evaluation':
        raise ValueError('Require a verified final evaluation permission')
    evidence = audited_training(review_path,protocol_path,profile_path,integrity_path)
    if (evidence.review_sha256 != permission['candidate_review_sha256']
            or (evidence.profile['model_id'],evidence.profile['revision']) != (permission['model_id'],permission['base_revision'])
            or set(permission['checkpoint_sha256']) != set(FILES)
            or any(evidence.review['checkpoint_files_sha256'][name] != permission['checkpoint_sha256'][name]
                   for name in FILES)):
        raise ValueError('Training profile or audited checkpoint differs from the promoted adapter')
    return evidence.profile,evidence.lineage
