"""Narrow output-only metadata intervention; never change executable inputs.

Apply after the historical raw-output cap. Never reveal text past that cap.
Only complete, anchored ls metadata rows and JS test aggregate timing rows
are changed. Source dates, error messages, test counts and return codes remain.
"""
import hashlib
import re

MARKER='\n[tool output truncated; use smaller file ranges]'
LS=re.compile(r'^(?P<prefix>[bcdlps-][rwxstST-]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+\d+\s+)(?P<time>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(?:\d{1,2}:\d{2}|\d{4}))(?P<suffix>\s+.+)$')
JS=re.compile(r'^(?P<prefix>\s*\d+ (?:passing|failing) \()(?P<time>\d+(?:\.\d+)?(?:ms|s))(?P<suffix>\)\s*)$')

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def canonicalize(output, limit=2400):
    truncated=output.endswith(MARKER)
    prefix=output[:-len(MARKER)] if truncated else output
    if len(prefix)>limit:raise ValueError('Normalize only the historical capped output')
    counts={'ls_timestamp':0,'js_summary_duration':0};lines=[]
    for line in prefix.splitlines(keepends=True):
        text=line.rstrip('\r\n');ending=line[len(text):]
        for pattern,key,replacement in [(LS,'ls_timestamp','Jan 01 00:00'),(JS,'js_summary_duration','0ms')]:
            match=pattern.fullmatch(text)
            if match:
                text=match['prefix']+replacement+match['suffix'];counts[key]+=1;break
        lines.append(text+ending)
    normalized=''.join(lines);extra_truncation=len(normalized)>limit
    normalized=normalized[:limit]+(MARKER if truncated or extra_truncation else '')
    return normalized,counts,extra_truncation

def transform(output,condition,limit=2400):
    if condition not in {'raw','canonical'}:raise ValueError('Unknown output condition')
    normalized,counts,extra=canonicalize(output,limit)
    delivered=output if condition=='raw' else normalized
    return delivered,{'condition':condition,'raw_sha256':digest(output),'canonical_sha256':digest(normalized),
        'delivered_sha256':digest(delivered),'raw_chars':len(output),'delivered_chars':len(delivered),
        'eligible_fields':counts,'applied_fields':counts if condition=='canonical' else dict.fromkeys(counts,0),
        'changed':output!=delivered,'canonical_extra_truncation':extra,'raw_prefix_only':True}
