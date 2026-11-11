"""Synthetic accounting checks; fixtures are never experiment results."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('curriculum_report', ROOT/'experiments/020-repair-curriculum/report.py')
report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)


class CurriculumReportTests(unittest.TestCase):
    def fixture(self, root):
        protocol = {'frozen_files':{}, 'prepared_data_sha256':{}, 'supervised_tokens':{},
                    'training_records':330, 'shared_identical_records':275, 'contrasting_records':55}
        slots = ['inspect','repair','verify','submit','stall_repair','late_repair']
        for name in ['repair','recovery']:
            rows = []
            for task in range(55):
                for slot in slots:
                    ident = f'train-{task}:curriculum-{slot}'
                    contrast = name == 'recovery' and slot == 'late_repair'
                    rows.append({'id':ident, 'tokens':100+int(contrast)*50, 'supervised_tokens':20+int(contrast),
                                 'record_sha256':hashlib.sha256((ident+str(contrast)).encode()).hexdigest()})
            folder = root/('data-'+name); folder.mkdir()
            receipt = {'sha256':'synthetic-'+name, 'teacher_api_used':False}
            report.write_json(folder/'receipt.json', receipt)
            data = {'selected':rows, 'selected_records':330, 'excluded':[], 'evaluation_traces_used':False,
                    'receipt_sha256':report.sha256(folder/'receipt.json'), 'data_sha256':receipt['sha256'],
                    'input_tokens':sum(r['tokens'] for r in rows), 'supervised_tokens':sum(r['supervised_tokens'] for r in rows),
                    'max_input_tokens':max(r['tokens'] for r in rows), 'max_target_tokens':max(r['supervised_tokens'] for r in rows)}
            report.write_json(folder/'selection.json',data)
            protocol['frozen_files']['runs/prepared-data020/'+name+'/selection.json'] = report.sha256(folder/'selection.json')
            protocol['prepared_data_sha256'][name] = receipt['sha256']
            protocol['supervised_tokens'][name] = data['supervised_tokens']
        protocol['semantic_order_sha256'] = hashlib.sha256(json.dumps([r['id'] for r in rows],separators=(',',':')).encode()).hexdigest()
        return protocol

    def mutate_and_refreeze(self, root, protocol, transform):
        # Even a consistently rehashed receipt must satisfy the declared contrast.
        path=root/'data-recovery/selection.json'; value=json.loads(path.read_text()); transform(value)
        report.write_json(path,value)
        protocol['frozen_files']['runs/prepared-data020/recovery/selection.json']=report.sha256(path)

    def test_unequal_token_counts_keep_exact_common_and_contrast_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); protocol=self.fixture(root)
            values=report.validate_curricula(root,protocol)
            self.assertEqual(values['repair']['input_tokens'],33000)
            self.assertEqual(values['recovery']['input_tokens'],35750)
            self.assertNotEqual(protocol['supervised_tokens']['repair'],protocol['supervised_tokens']['recovery'])

    def test_common_input_mutation_and_reordered_ids_fail_closed(self):
        mutations=[lambda v:v['selected'][0].update(record_sha256='changed-common-input'),
                   lambda v:v['selected'].reverse(), lambda v:v.update(input_tokens=33000),
                   lambda v:v.update(evaluation_traces_used=True)]
        for transform in mutations:
            with self.subTest(transform=transform), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); protocol=self.fixture(root)
                self.mutate_and_refreeze(root,protocol,transform)
                with self.assertRaises(ValueError): report.validate_curricula(root,protocol)

    def test_declared_tie_order_does_not_depend_on_result_dictionary_order(self):
        def campaign(passed):
            return {'complete':True,'results':[{'id':'one','passed':passed,'generated_tokens':10,
                'generations':[{'action_format_valid':True}]}]}
        plan=json.loads((report.EXP/'preparation-plan.json').read_text())
        candidates={'recovery':campaign(True),'repair':campaign(True)}
        selected=report.select(campaign(False),candidates,['one'],plan)
        self.assertEqual(selected['selected'],'repair')
        self.assertFalse(selected['final_test_opened'])


if __name__ == '__main__': unittest.main()
