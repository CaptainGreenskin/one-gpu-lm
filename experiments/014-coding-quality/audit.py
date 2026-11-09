"""Classify completed-run failures without exporting text, executing commands or changing scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.receipts import sha256, write_json
import one_gpu_lm.common.actions as actions


def audit(run):
    run=Path(run)
    if not json.loads((run/'controller.json').read_text())['complete']:
        raise ValueError('Post-run audit requires a completed screen')
    parser_hash=sha256(Path(actions.__file__))
    result={'schema_version':1,'purpose':'post-run descriptive audit; scores and parser unchanged',
        'raw_outputs_public':False,'parser_sha256':parser_hash,'audit_source_sha256':sha256(Path(__file__)),
        'variants':{},'campaign_sha256':{}}
    for variant in ['baseline','adapted']:
        path=run/variant/'campaign.json';campaign=json.loads(path.read_text())
        if not campaign['complete'] or campaign['identity']['source_files']['one_gpu_lm/common/actions.py']!=parser_hash:
            raise ValueError('Use the original parser with a completed campaign')
        result['campaign_sha256'][variant]=sha256(path);rows=[]
        for row in campaign['results']:
            if row['passed']:continue
            private=run/variant/'private'/row['id'];trajectory=json.loads((private/'trajectory.json').read_text())
            formats=Counter();errors=Counter();structures=Counter();runtime=[]
            for message in trajectory.get('messages',[]):
                content=message.get('content','')
                if message.get('role')=='assistant':
                    fmt='qwen_tool_call' if '<tool_call>' in content else 'fenced_block' if '```' in content else 'other_text'
                    try:parse_action(content);formats[fmt+'_accepted']+=1
                    except ValueError as error:
                        formats[fmt+'_rejected']+=1
                        reason=str(error)
                        # The pinned parser exposes exactly these two static errors.
                        if reason not in {'Expected exactly one nonempty bash action','Ambiguous native tool call'}:
                            raise ValueError('Unreviewed parser error; do not publish arbitrary exception text')
                        errors[reason]+=1
                        shape={'function_bash':'<function=bash>' in content,'command_parameter':'<parameter=command>' in content,
                            'tool_call_open':content.count('<tool_call>'),'tool_call_close':content.count('</tool_call>'),
                            'fenced_bash':'```bash' in content}
                        structures[json.dumps(shape,sort_keys=True)]+=1
                if message.get('role')=='exit' and 'context_limit' in content:runtime.append('context_limit')
            log=(private/'controller.log').read_text()
            if 'context_limit' in log and 'context_limit' not in runtime:runtime.append('context_limit')
            if row['exit_status']=='RuntimeError' and not runtime:runtime.append('unclassified_runtime_error')
            grader=(private/'grader.log').read_text()
            rows.append({'id':row['id'],'exit_status':row['exit_status'],'grading_returncode':row.get('returncode'),
                'grading_reason':row['reason'],'modified_source_files':len(row['modified_source_files']),
                'upstream_test_failure_signal':bool(re.search(r'\b(?:FAILED|AssertionError|failing|not ok|failed)\b',grader,re.I)),
                'action_format_counts':dict(formats),'parser_error_counts':dict(errors),
                'rejected_shapes':[{'shape':json.loads(shape),'count':count} for shape,count in sorted(structures.items())],
                'runtime_categories':sorted(set(runtime)),
                'trajectory_sha256':sha256(private/'trajectory.json'),'grader_log_sha256':sha256(private/'grader.log')})
        result['variants'][variant]=rows
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('Preserve the previous audit receipt')
    write_json(args.out,audit(args.run));print('Exported failure categories and hashes only')


if __name__=='__main__':main()
