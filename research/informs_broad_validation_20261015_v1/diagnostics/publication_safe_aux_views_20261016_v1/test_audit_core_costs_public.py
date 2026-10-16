import statistics
import unittest

import audit_core_costs as audit


class CoreCostAuditTests(unittest.TestCase):
    def fixture(self, fresh=False):
        ids = ['raw_cost'] if fresh else ['raw_cost', 'g_cost']
        bindings = {}; inputs = {}; reports = []; rows = []
        for condition in ids:
            identity=condition+'_accuracy'; family='RAW' if condition=='raw_cost' else 'DIRECT_G'
            accuracy=dict(run_id=identity, family=family, latent_k=3)
            spec=dict(accuracy,run_id=condition,accuracy_run_id=identity,row_microbatch=4 if family=='RAW' else 128)
            manifest='predictions/'+identity+'/V.json'; artifact='runs/'+identity+'/selected.pt'
            binding=dict(status='BOUND_EXACT_ROUTE',condition_id=condition,accuracy_run_id=identity,
                cost_route_spec=spec,cost_route_spec_sha256=audit.fingerprint(spec),accuracy_route_spec=accuracy,
                accuracy_route_spec_sha256=audit.fingerprint(accuracy),accuracy_manifest_public_relative_path=manifest,
                accuracy_manifest_sha256='manifest_'+identity,data_sha256='data',scaler_sha='scaler',
                artifact_bindings={'route':dict(cache_relative_path=artifact,sha256='artifact_'+identity)})
            bindings[condition]=binding
            inputs[('public',manifest)]=dict(sha256='manifest_'+identity)
            inputs[('private',artifact)]=dict(sha256='artifact_'+identity)
        for phase,count,instrumented in [('warmup',10,False),('timed',20,False),('stage_probe',20,True)]:
            for repeat in range(count):
                for position in range(len(ids)):
                    condition=ids[(repeat+position)%len(ids)]
                    streams,calls=audit.expected_F(bindings[condition]['cost_route_spec'],2,3)
                    rows.append(dict(run_id=condition,phase=phase,trial=str(repeat),order_position=str(position),
                        batch_size='2',whole_seconds=str(.01+repeat*.001+(0 if condition=='raw_cost' else .1)),
                        F_logical_streams=str(streams),F_physical_streams=str(streams),F_calls=str(calls),
                        row_microbatch=str(bindings[condition]['cost_route_spec']['row_microbatch']),
                        stage_instrumentation=str(instrumented),stage_CPU_preprocess_seconds='.001' if instrumented else ''))
        saved=[]
        for condition in ids:
            for phase in ['timed','stage_probe']:
                values=[float(row['whole_seconds']) for row in rows if row['run_id']==condition and row['phase']==phase]
                saved.append(dict(run_id=condition,condition_id=condition,accuracy_run_id=bindings[condition]['accuracy_run_id'],
                    phase=phase,batch_size=2,trials=20,median_seconds=statistics.median(values),mean_seconds=statistics.mean(values),
                    min_seconds=min(values),max_seconds=max(values),p90_seconds=values[17]+(values[18]-values[17])*.1,
                    stage_instrumentation=phase=='stage_probe',stage_sum_equals_whole_claimed=False))
            timed=[float(row['whole_seconds']) for row in rows if row['run_id']==condition and row['phase']=='timed']
            reports.append(dict(cost_run_id=condition,run_id=bindings[condition]['accuracy_run_id'],batch_size='2',
                scope='fresh_process_single_route' if fresh else 'general_shared_F_balanced',dataset='synthetic',timed_trials='20',
                accuracy_binding_status='BOUND_EXACT_ROUTE',status='COMPLETE_PROTOCOL',trials_sha256='synthetic_trial_sha',
                median_seconds=str(statistics.median(timed)),mean_seconds=str(statistics.mean(timed))))
        receipt=dict(run_id=ids[0],status='COMPLETE_PROTOCOL',scope=reports[0]['scope'],dataset='synthetic',batch_size=2,C=3,
            trials_sha256='synthetic_trial_sha',
            data_sha256='data',scaler_sha='scaler',accuracy_bindings=bindings,summaries=saved)
        if fresh:
            receipt.update(warmups=10,timed_trials=20,stage_trials=20)
        else:
            receipt.update(run_ids=ids,warmups_per_route=10,timed_trials_per_route=20,stage_trials_per_route=20)
        return receipt,rows,reports,inputs

    def test_full_balanced_and_single_protocols(self):
        for fresh in [False,True]:
            receipt,rows,reports,inputs=self.fixture(fresh)
            result,counts=audit.audit_receipt(receipt,rows,reports,inputs)
            self.assertTrue(counts['complete_protocol']);self.assertEqual(counts['actual_trials'],50*(1 if fresh else 2))
            self.assertEqual(len(result),2*(1 if fresh else 2))

    def test_stdlib_linear_p90_and_fixed_tolerance(self):
        self.assertAlmostEqual(audit.linear_p90(list(range(20))),17.1)
        self.assertEqual(audit.linear_p90([7]),7)
        self.assertEqual((audit.ABS_TOL,audit.REL_TOL),(1e-9,1e-7))

    def test_saved_p90_change_rejected(self):
        receipt,rows,reports,inputs=self.fixture();receipt['summaries'][0]['p90_seconds']+=.001
        with self.assertRaisesRegex(ValueError,'P90'):
            audit.audit_receipt(receipt,rows,reports,inputs)

    def test_rotation_stage_flag_and_uninstrumented_stage_value_rejected(self):
        for field,value in [('order_position','1'),('stage_instrumentation','True'),('stage_CPU_preprocess_seconds','.0001')]:
            receipt,rows,reports,inputs=self.fixture();rows[0][field]=value
            with self.assertRaises(ValueError):audit.audit_receipt(receipt,rows,reports,inputs)

    def test_nested_raw_F_calls_uses_leaf_microbatch(self):
        spec=dict(family='COMBINATION',row_microbatch=128,
                  g_route=dict(family='DIRECT_G'),b_route=dict(family='RAW',row_microbatch='full'))
        self.assertEqual(audit.expected_F(spec,8,170),(1360,1))
        spec['variant']='past_control';self.assertEqual(audit.expected_F(spec,8,170),(0,0))
        receipt,rows,reports,inputs=self.fixture();rows[0]['F_calls']='99'
        with self.assertRaisesRegex(ValueError,'F_STREAM_OR_CALL'):audit.audit_receipt(receipt,rows,reports,inputs)

    def test_metadata_binding_or_source_route_change_rejected(self):
        receipt,rows,reports,inputs=self.fixture();receipt['accuracy_bindings']['raw_cost']['cost_route_spec']['latent_k']=2
        with self.assertRaisesRegex(ValueError,'SPEC_BINDING'):audit.audit_receipt(receipt,rows,reports,inputs)
        receipt,rows,reports,inputs=self.fixture();inputs.pop(next(key for key in inputs if key[0]=='private'))
        with self.assertRaisesRegex(ValueError,'ARTIFACT_NOT_BOUND'):audit.audit_receipt(receipt,rows,reports,inputs)

    def test_negative_stage_and_malformed_relative_path_rejected(self):
        receipt,rows,reports,inputs=self.fixture()
        next(row for row in rows if row['phase']=='stage_probe')['stage_CPU_preprocess_seconds']='-1'
        with self.assertRaisesRegex(ValueError,'NEGATIVE_TIMER'):audit.audit_receipt(receipt,rows,reports,inputs)
        synthetic_drive = 'Q'
        synthetic_absolute = synthetic_drive + ':/' + 'publication_fixture.json'
        for path in ['../escape', synthetic_absolute, 'foo\\bar']:
            with self.assertRaises(ValueError):audit.inside('.',path)


if __name__=='__main__':
    unittest.main(verbosity=2)
