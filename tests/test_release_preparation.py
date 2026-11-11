import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from one_gpu_lm.common.actions import parse_action
from one_gpu_lm.common.agent_policy import NATIVE_EXAMPLE
from one_gpu_lm.common.edit_progress import EditableProgress
from one_gpu_lm.common.training_encoding import encode_training_example
from tests.test_training_encoding import RecordingTokenizer

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_preparation',ROOT/'experiments/019-coding-release/preparation.py')
preparation=importlib.util.module_from_spec(spec);spec.loader.exec_module(preparation)
PLAN=json.loads((ROOT/'experiments/019-coding-release/preparation-plan.json').read_text())


def cohort_fixture():
    tasks=[]
    for group,repos,counts,split in [
        ('train',PLAN['data']['training_repositories'],[12]*4,'train'),
        ('development',['dev/a','dev/b','dev/c'],[5]*3,'dev'),
        ('historical',['old/a','old/b','old/c','old/d'],[6]*4,'test'),
        ('fresh',['new/a','new/b','new/c','new/d'],[6]*4,'test')]:
        for repo,count in zip(repos,counts):
            for i in range(count):
                tasks.append({'id':repo+':'+str(i),'repository':repo,'family':repo+':'+str(i),'split':split,
                    'editable_files':['f.py'],'evaluation_role':'historical_final_regression_not_selection' if group=='historical' else group})
    records=[{**{k:t[k] for k in ['id','repository','family','split']},'admitted':True,
        'broken':{'returncode':1},'restored':{'returncode':0,'passed':True},
        'broken_sha256':{'f.py':'a'*64},'reference_sha256':{'f.py':'b'*64}} for t in tasks]
    admission={'complete':True,'tasks':records,'clean':[{'repository':r,'passed':True,'returncode':0} for r in sorted({t['repository'] for t in tasks})]}
    return tasks,admission


class ReleasePreparationTests(unittest.TestCase):
    def test_actual_collection_failure_blocks_an_admitted_task(self):
        tasks,admission=cohort_fixture()
        self.assertEqual(len(preparation.validate_cohorts(tasks,admission,PLAN)['fresh']),24)
        admission['tasks'][0]['broken']['returncode']=2
        with self.assertRaisesRegex(ValueError,'collection/environment'):
            preparation.validate_cohorts(tasks,admission,PLAN)

    def test_repository_overlap_and_duplicate_admission_block_freeze(self):
        tasks,admission=cohort_fixture()
        for t in tasks:
            if t['repository']=='new/a':t['repository']=PLAN['data']['training_repositories'][0]
        with self.assertRaisesRegex(ValueError,'leakage'):
            preparation.validate_cohorts(tasks,admission,PLAN)
        tasks,admission=cohort_fixture();admission['tasks'][-1]=admission['tasks'][0]
        with self.assertRaisesRegex(ValueError,'exactly once'):
            preparation.validate_cohorts(tasks,admission,PLAN)

    def test_read_only_examples_cannot_satisfy_repair_coverage(self):
        plan=copy.deepcopy(PLAN);plan['data'].update(minimum_unique_records=1,minimum_supervised_tokens=1)
        row={'id':'row','messages':[{'role':'user','content':'task'}],'target':NATIVE_EXAMPLE,
             'provenance':{'task_id':'t','split':'train','stage':'inspect'}}
        profile={'training':{'dialogue_format':'native_roundtrip'},'chat_template_kwargs':{}}
        tokenizer=RecordingTokenizer();e=encode_training_example(row,tokenizer,profile)
        selected={'id':'row','record_sha256':hashlib.sha256(json.dumps(e['ids'],separators=(',',':')).encode()).hexdigest(),
                  'tokens':len(e['ids']),'supervised_tokens':e['supervised_tokens']}
        selection={'selected':[selected],'selected_records':1,'input_tokens':len(e['ids']),'supervised_tokens':e['supervised_tokens']}
        with self.assertRaisesRegex(ValueError,'repair target'):
            preparation.validate_examples([row],selection,{'t'},tokenizer,profile,plan)
        row['provenance']['stage']='repair'
        self.assertEqual(preparation.validate_examples([row],selection,{'t'},tokenizer,profile,plan)['records'],1)
        selection['supervised_tokens']+=1
        with self.assertRaisesRegex(ValueError,'aggregates'):
            preparation.validate_examples([row],selection,{'t'},tokenizer,profile,plan)

    def test_training_history_must_match_the_executed_reference(self):
        messages=[{'role':'system','content':'rules'},{'role':'user','content':'task'}]
        stages=['inspect','reproduce','repair','review','verify','submit']
        targets=[]
        for i,stage in enumerate(stages):
            target=NATIVE_EXAMPLE.replace('pwd','printf '+stage);targets.append(target)
            messages.extend([{'role':'assistant','content':target,'extra':{'actions':[parse_action(target)]}},
                {'role':'user','content':'recorded observation','extra':{'tool':{'returncode':1 if i==1 else 0}}}])
        control={'broken_sha256':{'f.py':'a'*64},'reference_sha256':{'f.py':'b'*64}}
        trajectory={'messages':messages,'grade':{'passed':True,'source_sha256':control['reference_sha256']}}
        state={'changed_editable_files':0,'invalid_editable_files':0,
            'editable_state_sha256':hashlib.sha256(json.dumps(sorted(control['broken_sha256'].items()),separators=(',',':')).encode()).hexdigest()}
        tracker=object.__new__(EditableProgress);tracker.snapshot=lambda:state
        prefix,receipt=tracker.prepare_request(messages[:6],2,12,True)
        row={'target':targets[2],'messages':copy.deepcopy(prefix),
             'provenance':{'kind':'verified-reference-repair','stage':'repair','execution_returncode':0,'progress':receipt}}
        preparation.validate_reference_row(row,trajectory,control)
        row['messages'][-1]['content']='invented successful output'
        with self.assertRaisesRegex(ValueError,'unrecorded history'):
            preparation.validate_reference_row(row,trajectory,control)


if __name__=='__main__':unittest.main()
