"""Replay actual017 inputs to prove control identity before any018 generation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.agent_policy import BASH_TOOLS
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.repair_verification import prepare_verification
from one_gpu_lm.common.tool_history import render_history
from one_gpu_lm.common.receipts import sha256, write_json
from one_gpu_lm.models.verify import verify_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous-root', type=Path, required=True)
    args = p.parse_args()
    root, previous = args.root, args.previous_root
    runtime = json.loads((root/'runtime.json').read_text())
    profile = json.loads((root/'experiments/018-repair-verification/model-profile.json').read_text())
    model = Path(runtime['model'])
    output = root/'template-verification-probe.json'
    assert not output.exists()
    write_json(root/'model-verification.json', verify_model(model, root/'experiments/011-coding-model-selection/weights-manifest.json', profile))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    def encode(ms):
        return tokenizer.apply_chat_template(render_history(ms, True), tokenize=True, return_dict=False,
            add_generation_prompt=True, tools=BASH_TOOLS, **profile['chat_template_kwargs'])
    tasks = {t['id']:t for t in json.loads((previous/'private/tasks.json').read_text())['tasks']}
    run = previous/'runs/edit-progress'
    campaign = json.loads((run/'progress.json').read_text())
    assert campaign['complete']
    rows = []
    for result in campaign['results']:
        if result['split'] != 'dev':
            continue
        private = run/'private/progress'/result['id']
        history = [m for m in json.loads((private/'trajectory.json').read_text())['messages']
                   if m['role'] in {'system','user','assistant'}]
        original = json.loads((private/'original.json').read_text())
        initial = sorted((name,original[name]) for name in tasks[result['id']]['editable_files'])
        state = {'changed_editable_files':0,'invalid_editable_files':0,
                 'editable_state_sha256':hashlib.sha256(json.dumps(initial,separators=(',',':')).encode()).hexdigest()}
        used = 0
        steps = []
        for index, message in enumerate(history):
            if message['role'] == 'assistant':
                prefix = history[:index]
                observer = object.__new__(EditableProgress)
                observer.snapshot = lambda: dict(state)
                packed, receipt = observer.prepare_request(prefix, used, 12, True)
                assert receipt == result['generations'][used]['progress']
                control, cr = prepare_verification(packed, steps, state, used, 12, tasks[result['id']]['regression_command'], False)
                tokens = encode(control)
                expected = result['generations'][used]
                assert len(tokens) == expected['input_tokens']
                assert hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest() == expected['input_tokens_sha256']
                candidate, vr = prepare_verification(packed, steps, state, used, 12, tasks[result['id']]['regression_command'], True)
                assert candidate[1:] == prefix[1:] and (vr['reminder'] or encode(candidate) == tokens)
                rows.append({'id':result['id'],'completed_calls':used,'control_tokens':len(tokens),
                             'counterfactual_verify_tokens':len(encode(candidate)),
                             'verification_reminder':vr['reminder'],'check_status':vr['check_status'],
                             'would_exceed_limit':len(encode(candidate))>8192})
                used += 1
            elif 'tool' in message.get('extra',{}):
                observation = result['source_progress'][len(steps)]
                assert observation['call_index'] == used
                steps.append(observation)
                state = {k:v for k,v in observation.items() if k != 'call_index'}
    assert rows and any(r['verification_reminder'] for r in rows)
    write_json(output, {'schema_version':1,'model_generation_performed':False,
        'prior_campaign_sha256':sha256(run/'progress.json'),'template_sha256':sha256(model/'chat_template.jinja'),
        'control_actual_token_hashes_match017':True,'private_history_and_observations_unchanged':True,
        'candidate_state':'Counterfactual notices using actual017source observations; not018quality evidence',
        'prefix_count':len(rows),'prefixes':rows})
    print('Actual017control token hashes match; candidate reminders only; no model scoring')


if __name__ == '__main__':
    main()
