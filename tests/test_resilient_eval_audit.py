import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from one_gpu_lm.serving.receipt_client import request_identity
from one_gpu_lm.serving.request_journal import RequestJournal
from one_gpu_lm.verified.resilient_eval_audit import audit_task


class LateReceiptAuditTests(unittest.TestCase):
    def test_late_response_cannot_change_code_or_be_counted_as_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = RequestJournal(root/'journal')
            payload = {'operation':'generate','max_new_tokens':2}
            (root/'request-001.json').write_text(json.dumps(payload))
            identity = request_identity('trial', 1)
            started = time.time()
            generation = journal.execute(identity, payload, lambda _: {'generated_tokens':2,'ids':[1,2]})
            raw = journal.status(identity)
            recovery = dict(request_id=identity, accepted_epoch=raw['accepted_epoch'],
                            finished_epoch=raw['finished_epoch'], accounting_drain_seconds=.1)
            result = dict(run_id='trial', timeout_call=1, accounting_drain_seconds=.1,
                          complete_success_within_budget=False)
            trace = dict(started_epoch=started, finished_epoch=time.time(), submitted=False,
                         actor_seconds=.6, timeout_call=1, attempted_calls=1, accounting_drain_seconds=.1,
                         terminal_reason='time_limit_before_tool', final_source_sha256={'a':'initial'},
                         steps=[dict(call=1, generation=generation, source_before={'a':'initial'},
                                     transport_timeout_receipt=recovery)])
            self.assertEqual(audit_task(root, trace, result, journal), {identity})

            for field, value in [('final_source_sha256', {'a':'changed'}), ('submitted', True)]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    audit_task(root, dict(trace, **{field:value}), result, journal)
            with self.assertRaises(ValueError):
                audit_task(root, trace, dict(result, complete_success_within_budget=True), journal)
            altered = copy.deepcopy(trace)
            altered['steps'][0]['generation']['ids'][0] = 3
            with self.assertRaises(ValueError):
                audit_task(root, altered, result, journal)
            (root/'tool-001.json').write_text('{}')
            with self.assertRaises(ValueError):
                audit_task(root, trace, result, journal)


if __name__ == '__main__':
    unittest.main()
