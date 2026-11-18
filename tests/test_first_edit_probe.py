import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from one_gpu_lm.verified.first_edit_inputs import read_backup, select_anchors
from one_gpu_lm.verified.first_edit_probe import classify


class FirstEditProbeTests(unittest.TestCase):
    def test_rejects_a_multi_action_proposal_instead_of_counting_its_edit(self):
        case = {'base_first_edit_command':'edit synthetic.py',
                'observed_first_edit_commands':['edit synthetic.py','change synthetic.py'],
                'declared_regression_command':'python3 run_checks.py'}
        action = '```bash\nedit synthetic.py\n```'
        self.assertEqual(classify(action,case),'exact_base_first_edit')
        self.assertEqual(classify(action+'\n```bash\ncat synthetic.py\n```',case),'invalid')
        self.assertEqual(classify('```bash\napply different edit\n```',case),'other_parseable')

    def test_does_not_follow_an_archive_link_for_a_registered_input(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/'input.tar.gz'
            with tarfile.open(archive,'w:gz') as raw:
                data = tarfile.TarInfo('development.json')
                data.size = 2
                raw.addfile(data,io.BytesIO(b'{}'))
                link = tarfile.TarInfo('alias.json')
                link.type = tarfile.LNKTYPE
                link.linkname = data.name
                raw.addfile(link)
            self.assertEqual(read_backup(archive,[data.name]),{data.name:b'{}'})
            with self.assertRaises(ValueError):
                read_backup(archive,[link.name])

    def test_refuses_a_comparison_history_that_already_edited_source(self):
        groups = {
            ('task',i): {
                'base':{'run_id':'base','repetition':i,'first_edit_call':1,'complete_success_within_budget':True},
                'multifile032':{'run_id':'candidate','repetition':i,'first_edit_call':1,'complete_success_within_budget':False},
            } for i in range(3)
        }
        traces = {'base':{'steps':[{'source_before':{'a':'old'},'source_after':{'a':'new'}}]}}
        with self.assertRaisesRegex(ValueError,'before an edit'):
            select_anchors(groups,traces)


if __name__ == '__main__':
    unittest.main()
