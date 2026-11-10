"""Replay existing private histories through the pinned tokenizer without model generation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from one_gpu_lm.common.agent_policy import BASH_TOOLS
from one_gpu_lm.common.tool_history import render_history
from one_gpu_lm.common.receipts import sha256,write_json
from one_gpu_lm.models.verify import verify_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);args=p.parse_args()
    root=args.root;runtime=json.loads((root/'runtime.json').read_text());profile=json.loads((root/'experiments/016-tool-roundtrip/model-profile.json').read_text());model=Path(runtime['model'])
    assert not (root/'template-roundtrip-probe.json').exists()
    write_json(root/'model-verification.json',verify_model(model,root/'experiments/011-coding-model-selection/weights-manifest.json',profile))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    prefixes=[];campaign_hashes={};changed=0
    def encode(messages,tools):
        return tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,**profile['chat_template_kwargs'],**({'tools':BASH_TOOLS} if tools else {}))
    for arm in ['baseline','format','context','combined']:
        path=args.previous_run/(arm+'.json');campaign=json.loads(path.read_text());assert campaign['complete'];campaign_hashes[arm]=sha256(path)
        for row in campaign['results']:
            path=args.previous_run/'private'/arm/row['id']/'trajectory.json'
            history=[m for m in json.loads(path.read_text())['messages'] if m['role'] in ['system','user','assistant']]
            assert len(history)%2==0
            for size in range(2,len(history)+1,2):
                original=history[:size];before=json.dumps(original,sort_keys=True)
                plain=[{k:m[k] for k in ['role','content']} for m in original]
                assert encode(plain,arm in ['format','combined'])==encode(render_history(original),arm in ['format','combined'])
                native=render_history(original,True)
                count=sum('tool' in m.get('extra',{}) for m in original)
                assert sum(m['role']=='tool' for m in native)==count
                assert sum(len(m.get('tool_calls',[])) for m in native)==count
                assert json.dumps(original,sort_keys=True)==before
                a=encode(plain,True);b=encode(native,True);changed+=a!=b
                prefixes.append({'id':row['id'],'previous_variant':arm,'messages':size,'schema_tokens':len(a),'roundtrip_tokens':len(b),'executed_tool_results':count})
    assert changed>0 and len(campaign_hashes)==4 and len(prefixes)>96
    write_json(root/'template-roundtrip-probe.json',{'schema_version':1,'model_generation_performed':False,'new_scores_used_to_select_policy':False,'template_sha256':sha256(model/'chat_template.jinja'),'prior_campaign_sha256':campaign_hashes,'exact_legacy_serialization_equivalence':True,'all_executed_pairs_linked':True,'private_histories_unchanged':True,'prefix_count':len(prefixes),'changed_native_prefixes':changed,'prefixes':prefixes})
    print('Pinned tokenizer replay passed; no model-generated scores')


if __name__=='__main__':main()
