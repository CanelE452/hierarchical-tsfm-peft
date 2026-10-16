import csv
import io
import json
import unittest

import audit_multik as audit


class NumericAuditTests(unittest.TestCase):
    def fixture(self, count=180):
        scope = dict(scope_id='synthetic_B1_shared', strategy='shared', dataset='synthetic',
                     batch=1, K_grid=[1, 2, 3], route_ids={'1': 's1', '2': 's2', '3': 's3'})
        rows = []
        schedule = [(phase, *entry) for phase, n in [('warmup', 10), ('timed', 20)] for entry in audit.edges([1, 2, 3], n)]
        for index, (phase, repeat, position, a, b) in enumerate(schedule[:count]):
            rows.append(dict(strategy='shared', phase=phase, repeat=repeat, order_position=position,
                from_K=a, to_K=b, batch_size=1, accuracy_run_id='s'+str(b),
                transition_plus_physical_seconds=0.12345678901234567+index*.001,
                physical_pipeline_seconds=0.012345678901234567+index*.0001,
                controller_guard_seconds_between_segments=.001,
                controller_guard_seconds_inside_switch=.0001,
                controller_guard_seconds_inside_whole=.001+.0001,
                derived_seconds_minus_observed_controller_guard=max(0.,0.12345678901234567+index*.001-.001-.0001),
                other_clock_overhead_seconds=-1e-10, F_streams=b, F_physical_streams=b,
                numeric_prediction_valid=True, resource_within_limit=True, resource_event_after_prediction=None))
        result = dict(strategy='shared', dataset='synthetic', batch=1, rows=rows,
            transition_ledger=audit.ledger([1, 2, 3], rows), timer_scope=audit.TIMER_SCOPE,
            status='COMPLETE_PROTOCOL' if count==180 else 'PARTIAL_RESOURCE_LIMIT',
            numeric_execution_status='COMPLETE_REQUESTED_TRIALS' if count==180 else 'PARTIAL',
            resource_status='WITHIN_OBSERVED_BOUNDARIES' if count==180 else 'RESOURCE_GUARD_TRIGGERED',
            post_measure_validation_status='PASS' if count==180 else 'NOT_RUN_RESOURCE_LIMIT',
            exact_minimum_protocol_complete=count==180)
        if count==180:
            result['summaries'] = self.saved_summaries(result)
        return scope, result

    def saved_summaries(self, result):
        rows=[]
        for entry in result['transition_ledger']:
            if entry['phase']=='timed':
                group=[r for r in result['rows'] if (r['phase'],r['from_K'],r['to_K'])==('timed',entry['from_K'],entry['to_K'])]
                import statistics
                rows.append(dict(entry, median_transition_plus_physical_seconds=statistics.median([r['transition_plus_physical_seconds'] for r in group]) if group else None,
                                 median_physical_pipeline_seconds=statistics.median([r['physical_pipeline_seconds'] for r in group]) if group else None))
        return rows

    def originals(self, result):
        originals={'shared/RESULT.json':audit.trial_bytes(result)}
        numeric=dict(result, status='NUMERIC_TRIALS_RECORDED_PENDING_FINAL_VALIDATION', exact_minimum_protocol_complete=False)
        originals['shared/NUMERIC_EXECUTION.json']=audit.trial_bytes(numeric)
        stream=io.StringIO(newline=''); writer=csv.DictWriter(stream, fieldnames=list(result['rows'][0]));writer.writeheader();writer.writerows(result['rows'])
        originals['shared/TRIALS.csv']=stream.getvalue().encode()
        for index,row in enumerate(result['rows']):
            originals[f'shared/actual_trials/trial_{index:06}.json']=audit.trial_bytes(row)
        return originals

    def test_full_protocol_and_exact_medians(self):
        scope,result=self.fixture()
        actual,ledger,medians=audit.audit_scope(scope,self.originals(result))
        self.assertTrue(actual['exact_minimum_protocol_complete'])
        self.assertEqual(sum(x['actual_complete'] for x in ledger),180)
        self.assertEqual(medians,result['summaries'])
        self.assertEqual(len(audit.summaries(scope,actual,ledger,'synthetic/RESULT.json','synthetic')),24)

    def test_partial_resource_row_is_preserved_and_separated(self):
        scope,result=self.fixture(166)
        result['rows'][-1].update(resource_within_limit=False,resource_event_after_prediction={'synthetic':True})
        result['transition_ledger']=audit.ledger(scope['K_grid'],result['rows'])
        actual,ledger,_=audit.audit_scope(scope,self.originals(result))
        self.assertEqual(sum(x['completed_within_resource'] for x in ledger),165)
        self.assertFalse(actual['exact_minimum_protocol_complete'])
        groups=audit.summaries(scope,actual,ledger,'synthetic/RESULT.json','synthetic')
        self.assertEqual(sum(x['actual_count'] for x in groups if x['validity_scope']=='all_recorded'),166)
        self.assertEqual(sum(x['actual_count'] for x in groups if x['validity_scope']=='resource_valid_only'),165)

    def test_changed_stored_median_rejected(self):
        scope,result=self.fixture(); result['summaries'][0]['median_physical_pipeline_seconds']+=1e-12
        with self.assertRaisesRegex(ValueError,'SAVED_MEDIAN'):
            audit.audit_scope(scope,self.originals(result))

    def test_consistently_changed_direction_order_rejected(self):
        scope,result=self.fixture(); result['rows'][0]['order_position']=1
        with self.assertRaisesRegex(ValueError,'DIRECTED_TRANSITION'):
            audit.audit_scope(scope,self.originals(result))

    def test_durable_bytes_and_csv_tampering_rejected(self):
        scope,result=self.fixture(); originals=self.originals(result)
        originals['shared/actual_trials/trial_000000.json']=originals['shared/actual_trials/trial_000000.json'].replace(b'\r\n',b'\n')
        with self.assertRaisesRegex(ValueError,'EXACT_BYTES'):
            audit.audit_scope(scope,originals)
        originals=self.originals(result);originals['shared/TRIALS.csv']=originals['shared/TRIALS.csv'].replace(b'0.12345678901234566',b'0.9',1)
        with self.assertRaisesRegex(ValueError,'CSV_ROW'):
            audit.audit_scope(scope,originals)

    def test_signed_residual_allowed_negative_duration_rejected(self):
        scope,result=self.fixture();audit.audit_scope(scope,self.originals(result))
        result['rows'][0]['physical_pipeline_seconds']=-1e-12
        with self.assertRaisesRegex(ValueError,'NEGATIVE_DURATION'):
            audit.audit_scope(scope,self.originals(result))

    def test_guard_fields_and_original_operation_order_checked(self):
        scope,result=self.fixture()
        result['rows'][0]['controller_guard_seconds_between_segments']=-1e-12
        with self.assertRaisesRegex(ValueError,'NEGATIVE_DURATION'):
            audit.audit_scope(scope,self.originals(result))
        scope,result=self.fixture()
        result['rows'][0]['derived_seconds_minus_observed_controller_guard']+=1e-12
        with self.assertRaisesRegex(ValueError,'TIMER_ARITHMETIC'):
            audit.audit_scope(scope,self.originals(result))


if __name__=='__main__':
    unittest.main(verbosity=2)
