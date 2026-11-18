"""Reference-guided expert action plans, explicitly distinct from model rollouts."""
import difflib
import shlex
from .submission import SUBMIT


def native(command):
    return '<tool_call>\n<function=bash>\n<parameter=command>\n'+command+'\n</parameter>\n</function>\n</tool_call>'


def replacements(parent, fixed):
    """Find executable unique anchors, recomputing after each bottom-up edit."""
    current=parent;out=[]
    while current!=fixed:
        a=current.splitlines(keepends=True);b=fixed.splitlines(keepends=True)
        groups=list(difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_grouped_opcodes(2))
        if not groups:raise ValueError('Non-line repair unsupported')
        group=groups[-1];i,j=group[0][1],group[-1][2];k,l=group[0][3],group[-1][4]
        while True:
            before=''.join(a[i:j]);after=''.join(b[k:l])
            if before and current.count(before)==1:break
            if i>0 and k>0:i-=1;k-=1
            elif j<len(a) and l<len(b):j+=1;l+=1
            else:raise ValueError('No unique source anchor')
        next_text=current.replace(before,after)
        if next_text==current:raise ValueError('Empty expert edit')
        out.append((before,after));current=next_text
        if len(out)>100:raise ValueError('Excessive reference hunks')
    return out


def edit_command(path, changes):
    # repr preserves quotes, newlines, shell substitutions and backticks as data
    # inside the quoted heredoc; do not serialize commands with JSON shell quoting.
    body=("from pathlib import Path\np = Path("+repr(path)+")\ns = p.read_bytes().decode('utf-8')\n"
          "changes = "+repr(changes)+"\n"
          "for before, after in changes:\n"
          "    assert s.count(before) == 1, 'source anchor changed'\n"
          "    s = s.replace(before, after)\np.write_bytes(s.encode('utf-8'))\n")
    delimiter='HISTORIC032_EDIT'
    while delimiter in body.splitlines():delimiter+='_' 
    return "python3 - <<'"+delimiter+"'\n"+body+delimiter


def plan(task, admission, target_tokens, budget):
    """No execution or scoring. Token counter uses the actual student template."""
    if not admission['admitted'] or task['split']!='training':raise ValueError('Admitted training tasks only')
    mutations={m['path']:m for m in task['mutations']}
    essential=sorted(admission['essential_files'])
    order=essential+[n for n in task['editable_files'] if n not in essential]
    if set(order)!=set(mutations):raise ValueError('Reference file inventory differs')
    commands=[{'kind':'baseline_regression','command':task['regression_command'],'expected':'semantic_failure'}]
    reads=[];expected_chars=0
    for name in order:
        m=mutations[name];a=m['after'].splitlines(keepends=True);b=m['before'].splitlines(keepends=True)
        changed=next((x for x in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes() if x[0]!='equal'),None)
        if changed is None:raise ValueError('Unchanged declared source')
        start=max(0,changed[1]-2);end=min(len(a),max(changed[2],changed[1]+1)+2)
        while sum(map(len,a[start:end]))>500 and end>start+1:end-=1
        expected_chars+=sum(map(len,a[start:end]))
        reads.append("sed -n "+shlex.quote(str(start+1)+','+str(end)+'p')+' '+shlex.quote(name))
    if expected_chars>budget['max_tool_output_chars']:raise ValueError('Focused inspection exceeds output cap')
    commands.append({'kind':'focused_inspection','command':'\n'.join(reads),'expected':'ok'})
    for index,name in enumerate(order):
        m=mutations[name];hunks=replacements(m['after'],m['before']);groups=[];current=[]
        for change in hunks:
            proposed=current+[change]
            if target_tokens(native(edit_command(name,proposed)))<=budget['max_generated_tokens_per_call']:
                current=proposed
            else:
                if not current:raise ValueError('Indivisible reference hunk exceeds action token cap')
                groups.append(current);current=[change]
                if target_tokens(native(edit_command(name,current)))>budget['max_generated_tokens_per_call']:
                    raise ValueError('Indivisible reference hunk exceeds action token cap')
        if current:groups.append(current)
        for group in groups:commands.append({'kind':'reference_edit','command':edit_command(name,group),'expected':'ok','source_file':name})
        if admission['multifile'] and index==0:
            commands.append({'kind':'partial_repair_regression','command':task['regression_command'],'expected':'semantic_failure'})
    commands += [{'kind':'final_verification','command':task['regression_command'],'expected':'ok'},
                 {'kind':'explicit_submission','command':SUBMIT,'expected':'ok'}]
    counts=[target_tokens(native(c['command'])) for c in commands]
    if (len(commands)>budget['max_model_calls'] or max(counts)>budget['max_generated_tokens_per_call']
            or sum(counts)>budget['max_total_generated_tokens']):raise ValueError('Complete expert plan exceeds common action/token limits')
    return {'producer_kind':'upstream_reference_guided','oracle_localization':True,
            'reference_patch_used_for_targets':True,'model_generations':0,'task_id':task['id'],
            'commands':commands,'action_target_tokens':counts,'actions':len(commands),
            'first_edit_call':next(i for i,c in enumerate(commands,1) if c['kind']=='reference_edit'),
            'last_two_actions':['final_verification','explicit_submission'],
            'partial_repair_is_reference_partitioned':admission['multifile']}
