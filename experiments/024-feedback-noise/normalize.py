"""Add recognized test durations to the unchanged023 metadata_v1 boundary."""
from pathlib import Path
import re
from _feedback024 import module

legacy=module('metadata023',Path(__file__).resolve().parent.parent/'023-evaluation-stability/normalize.py')
digest=legacy.digest;MARKER=legacy.MARKER
COUNT=r'\d+ (?:(?:subtests? )?(?:passed|failed|skipped|xfailed|xpassed|deselected)|errors?|warnings?)'
PYTEST=re.compile(r'^(?P<prefix>(?:=+ )?'+COUNT+r'(?:, '+COUNT+r')* in )(?P<time>\d+(?:\.\d+)?s(?: \(\d+:\d{2}:\d{2}\))?)(?P<suffix>(?: =+)?\r?)$')
UNITTEST=re.compile(r'^(?P<prefix>Ran \d+ tests? in )(?P<time>\d+(?:\.\d+)?s)(?P<suffix>\r?)$')
# A duration is optional in this reporter. Remove its suffix rather than add
# a synthetic duration to rows that never had one. Never reorder test rows.
AVA=re.compile(r'^(?P<prefix>[ \t]+[✔✓✘✖] [^\r\n]+?)(?P<time> \(\d+(?:\.\d+)?(?:ms|s)\))(?P<suffix>\r?)$')

def extended(output,limit=2400):
    baseline,old_fields,extra=legacy.canonicalize(output,limit)
    truncated=baseline.endswith(MARKER)
    prefix=baseline[:-len(MARKER)] if truncated else baseline
    counts={'pytest_summary_duration':0,'unittest_summary_duration':0,'ava_test_duration':0}
    lines=prefix.splitlines(keepends=True);changed=[]
    for i,line in enumerate(lines):
        # A cut-off suffix may accidentally resemble a complete timing field.
        if truncated and i==len(lines)-1 and not line.endswith(('\n','\r')):
            changed.append(line);continue
        text=line.rstrip('\n');ending=line[len(text):]
        for pattern,key,replacement in [(PYTEST,'pytest_summary_duration','0s'),(UNITTEST,'unittest_summary_duration','0s'),(AVA,'ava_test_duration','')]:
            match=pattern.fullmatch(text)
            if match:
                text=match['prefix']+replacement+match['suffix'];counts[key]+=1;break
        changed.append(text+ending)
    value=''.join(changed)+(MARKER if truncated else '')
    if len(value)>len(baseline):raise ValueError('Additional024 transforms must only shrink output')
    return baseline,value,old_fields,counts,extra

def transform(output,condition,limit=2400):
    if condition not in {'baseline','extended'}:raise ValueError('Unknown024 output condition')
    baseline,normalized,old_fields,new_fields,extra=extended(output,limit)
    delivered=baseline if condition=='baseline' else normalized
    eligible={**old_fields,**new_fields}
    applied={**old_fields,**(new_fields if condition=='extended' else dict.fromkeys(new_fields,0))}
    return delivered,{'condition':condition,'raw_sha256':digest(output),'baseline_sha256':digest(baseline),
      'canonical_sha256':digest(normalized),'delivered_sha256':digest(delivered),
      'raw_chars':len(output),'baseline_chars':len(baseline),'delivered_chars':len(delivered),
      'eligible_fields':eligible,'applied_fields':applied,'changed':output!=delivered,
      'additional_changed':baseline!=delivered,'canonical_extra_truncation':extra,'raw_prefix_only':True,
      'normalization_version':'metadata_v1' if condition=='baseline' else 'metadata_v2'}
