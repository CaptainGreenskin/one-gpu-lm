"""Read and admit both frozen data lineages for the single mixed candidate."""
import json
from pathlib import Path
from one_gpu_lm.common.receipts import sha256
from .candidate_data import select as select_complete
from .mixed_data import select_fragments, combine
from .tasks import relative


def admit(root, protocol, profile, tokenizer):
    root=Path(root);mix=protocol['correction_mix']
    def read(name):return json.loads((root/relative(name)).read_text())
    def rows(name):return [json.loads(line) for line in (root/relative(name)).read_text().splitlines() if line.strip()]
    for name,h in mix['frozen_parent_files'].items():
        if sha256(root/relative(name))!=h:raise ValueError('Mixed training parent changed: '+name)
    original=mix['original'];correction=mix['correction']
    sources=read(protocol['sources_path'])['sources']
    raw_complete=rows(original['data']);manifest=read(original['manifest']);audit=read(original['audit'])
    if manifest['data_sha256']!=sha256(root/relative(original['data'])) or manifest['audit_sha256']!=sha256(root/relative(original['audit'])):
        raise ValueError('Original data lineage mismatch')
    complete=select_complete(raw_complete,manifest,audit,sources,profile,read(original['cycle_spec']),tokenizer)
    raw_fragments=rows(correction['data']);export=read(correction['manifest']);audit=read(correction['audit']);p=read(correction['protocol'])
    controller=read(correction['controller']);supervisor=read(correction['supervisor'])
    if (not controller['complete'] or not controller['backend_stopped'] or not controller['frozen_files_unchanged']
            or not supervisor['complete'] or supervisor['timed_out'] or supervisor['worker_returncode']!=0
            or audit['protocol_sha256']!=controller['protocol_sha256']
            or controller['protocol_sha256']!=sha256(root/relative(correction['protocol']))
            or export!=audit['private_export_summary'] or export['data_sha256']!=sha256(root/relative(correction['data']))):
        raise ValueError('Correction data lineage or termination mismatch')
    fragments=select_fragments(raw_fragments,export,audit,p,sources,profile,tokenizer)
    selected,metadata=combine(complete,fragments)
    if metadata['full_episodes']!=manifest['full_episodes'] or metadata['corrective_fragments']!=export['qualified_unique_tasks']:
        raise ValueError('Complete episodes and fragments must remain distinct')
    return selected,metadata,raw_complete+raw_fragments


def select(root,protocol,profile,tokenizer,rows,manifest):
    selected,metadata,expected_rows=admit(root,protocol,profile,tokenizer)
    if rows!=expected_rows or manifest['mixed_selection']!=metadata:
        raise ValueError('Mixed export changed, omitted or reordered records')
    return selected,metadata
