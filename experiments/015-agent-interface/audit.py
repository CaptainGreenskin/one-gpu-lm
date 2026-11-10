"""Classify all completed task responses without exporting text, executing commands or changing scores."""
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
    result={'schema_version':1,'purpose':'post-run structural observations on every task; no command execution, score changes or parser changes',
        'raw_outputs_public':False,'parser_sha256':parser_hash,'audit_source_sha256':sha256(Path(__file__)),
        'variants':{},'campaign_sha256':{}}
    for variant in ['baseline','format','context','combined']:
        path=run/(variant+'.json');campaign=json.loads(path.read_text())
        if not campaign['complete'] or campaign['identity']['source_files']['one_gpu_lm/common/actions.py']!=parser_hash:
            raise ValueError('Use the original parser with a completed campaign')
        result['campaign_sha256'][variant]=sha256(path);rows=[]
        for row in campaign['results']:
            private=run/'private'/variant/row['id'];trajectory=json.loads((private/'trajectory.json').read_text())
            formats=Counter();errors=Counter();structures=Counter();runtime=[];accepted=[];categories=Counter()
            for message in trajectory.get('messages',[]):
                content=message.get('content','')
                if message.get('role')=='assistant':
                    fmt='qwen_tool_call' if '<tool_call>' in content else 'fenced_block' if '```' in content else 'other_text'
                    try:
                        parse_action(content);formats[fmt+'_accepted']+=1;accepted.append(True)
                    except ValueError as error:
                        formats[fmt+'_rejected']+=1;accepted.append(False)
                        if message.get('extra',{}).get('generation',{}).get('hit_generation_limit'):
                            category='invalid_at_output_limit'
                        elif '<function=Bash>' in content:
                            category='unsupported_tool_name_case'
                        elif '<tool_call>' in content and '</tool_call>' not in content:
                            category='unclosed_native_markup'
                        elif '<tool_call>' in content and '<parameter=command>' not in content:
                            category='native_without_command_parameter'
                        elif '<tool_call>' in content:
                            category='other_rejected_native_shape'
                        else:
                            category='no_supported_action'
                        categories[category]+=1
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
            if accepted != [g['action_format_valid'] for g in row['generations']]:
                raise ValueError('Trajectory parser decisions differ from recorded generations')
            if any(e.get('error')=='context_limit' for e in row['backend_errors']) and 'context_limit' not in runtime:runtime.append('context_limit')
            if row['exit_status']=='RuntimeError' and not runtime:runtime.append('unclassified_runtime_error')
            grader_path=private/'grader.log'
            grader=grader_path.read_text() if grader_path.exists() else ''
            rows.append({'id':row['id'],'passed':row['passed'],'split':row['split'],'format_categories':dict(categories),'exit_status':row['exit_status'],'grading_returncode':row.get('returncode'),
                'grading_reason':row['reason'],'modified_source_files':len(row['modified_source_files']),
                'upstream_test_failure_signal':bool(re.search(r'\b(?:FAILED|AssertionError|failing|not ok|failed)\b',grader,re.I)) if grader_path.exists() else None,
                'action_format_counts':dict(formats),'parser_error_counts':dict(errors),
                'rejected_shapes':[{'shape':json.loads(shape),'count':count} for shape,count in sorted(structures.items())],
                'runtime_categories':sorted(set(runtime)),
                'trajectory_sha256':sha256(private/'trajectory.json'),'grader_log_sha256':sha256(grader_path) if grader_path.exists() else None})
        result['variants'][variant]=rows
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('Preserve the previous audit receipt')
    write_json(args.out,audit(args.run));print('Exported failure categories and hashes only')


if __name__=='__main__':main()
