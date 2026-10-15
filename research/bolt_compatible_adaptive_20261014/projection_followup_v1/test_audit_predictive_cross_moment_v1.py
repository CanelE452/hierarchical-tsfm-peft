"""Synthetic independent direction/LS/oracle-score checks; no actual result or assets."""
from types import SimpleNamespace
from unittest.mock import patch
import copy
import subprocess
import sys
import unittest

import numpy as np

import audit_predictive_cross_moment_v1 as audit
from test_run_predictive_cross_moment_v1 import fixture,outputs_for
import predictive_cross_moment_v1 as direction
from causal_lag_decoder_v1 import fit_reconstruction
import run_predictive_cross_moment_v1 as producer

audit.initialize_numpy()


class IndependentCrossMomentAuditTests(unittest.TestCase):
    def test_independent_elementwise_moments_whitening_and_direction_match_both_rules(self):
        raw=fixture()
        for kind in audit.KINDS:
            for k in (1,2):
                with self.subTest(kind=kind,k=k):
                    saved=direction.fit_directions(raw['history'],raw['history_observed'],raw['target'],raw['observed'],k,kind=kind)
                    independent=audit.independent_direction(raw,kind,k)
                    np.testing.assert_allclose(saved['encoder_FP64'],independent['E'],atol=1e-9,rtol=1e-7)
                    np.testing.assert_allclose(saved['cross_moments_FP64'],independent['cross'],atol=1e-12,rtol=1e-10)
                    np.testing.assert_allclose(independent['latent_covariance'],np.eye(k),atol=1e-7)
                    self.assertEqual(independent['counts'].tolist(),saved['metadata']['horizon_complete_C_origins'])
                    self.assertGreater(independent['E'][np.argmax(np.abs(independent['E'][:,0])),0],0.)
        changed=copy.deepcopy(raw);changed['target'][~changed['observed']]=np.inf
        np.testing.assert_array_equal(audit.independent_direction(raw,'CROSS_MOMENT',2)['E'],audit.independent_direction(changed,'CROSS_MOMENT',2)['E'])

    def test_independent_centered_true_observed_D_noobs_oneobs_constant_response(self):
        rng=np.random.default_rng(92807);x=rng.normal(size=(2,512,4)).astype(np.float32);z=rng.normal(size=(2,512,2)).astype(np.float32)
        mask=np.ones_like(x,dtype=bool);mask[:,:,0]=False;mask[:,:,1]=False;mask[0,40,1]=True;x[:,:,2]=2.
        raw={'history':x,'history_observed':mask};saved=fit_reconstruction(x,mask,z,audit.LAGS);d,b,meta=audit.independent_decoder(raw,z)
        np.testing.assert_allclose(saved['decoder_FP64'],d,atol=1e-8,rtol=1e-7);np.testing.assert_allclose(saved['bias_FP64'],b,atol=1e-8,rtol=1e-7)
        self.assertEqual(meta['counts'],[0,1,960,960]);self.assertEqual(meta['rank'][0:2],[0,0]);self.assertIsNone(meta['residual'][0])
        np.testing.assert_array_equal(d[:,0],0.);np.testing.assert_array_equal(d[:,1],0.);np.testing.assert_array_equal(d[:,2],0.)
        self.assertEqual(b[1],float(x[0,40,1]));self.assertEqual(b[2],2.)
        self.assertEqual(meta['latent_binding'],audit.array_record(z))

    def test_saved_24cell_arithmetic_method_masks_and_diagnostics_without_F_replay(self):
        raw=fixture();_,_,state,_,_=producer.fit_state(raw,'CROSS_MOMENT',2,device='cpu')
        _,outputs,ey,eyp,flatp=outputs_for(raw,state);row,independent=producer.evaluate_arrays(outputs,raw,state,ey,eyp,flatp)
        saved={**outputs,'EY_future':ey,'EY_prediction':eyp,'FLAT_future':np.repeat(outputs['latent_history'][:,-1:],48,axis=1),'FLAT_prediction':flatp,'oracle_valid_horizon_mask':producer.oracle_dependency_mask(raw['observed']),'origins':raw['origins'].copy()}
        checked=audit.check_cell(row,saved,independent,raw,state)
        self.assertEqual(checked['score_fields_checked'],156);self.assertFalse(checked['actual_F_independently_reinferred'])
        self.assertLess(checked['oracle_valid_horizons'],checked['complete_C_horizons'])
        for method in checked['methods'].values():self.assertEqual(method['scores_oracle_common_mask']['normalized']['observed_counts'],[checked['oracle_valid_horizons']]*4)
        for change,label in ((lambda r,s,i:r['methods']['RAW_BOLT']['scores_oracle_common_mask']['normalized'].update(mse=99.),'SCORE'),(lambda r,s,i:s['EY_prediction'].__setitem__((0,4,0),s['EY_prediction'][0,4,0]+.1),'EY_prediction'),(lambda r,s,i:s['oracle_valid_horizon_mask'].__setitem__((0,3),True),'oracle_all')):
            a,b,c=copy.deepcopy(row),copy.deepcopy(saved),copy.deepcopy(independent);change(a,b,c)
            with self.subTest(label=label),self.assertRaises(ValueError):audit.check_cell(a,b,c,raw,state)

    def test_causal_oracle_independent_mask_excludes_later_target_with_missing_earlier_reference(self):
        observed=np.ones((1,48,4),bool);observed[0,2,1]=False
        mask=audit.valid_oracle_mask(observed);self.assertTrue(observed[0,3].all());self.assertFalse(mask[0,3]);self.assertFalse(mask[0,34])
        np.testing.assert_array_equal(mask,producer.oracle_dependency_mask(observed))
        self.assertTrue(mask[0,0]);self.assertTrue(mask[0,1])

    def test_timestamp_duplicated_occurrences_and_complete_HC_are_independently_checked(self):
        raw=fixture();recorded=producer.timestamp_metadata(raw);audit.check_timestamps(recorded,raw)
        self.assertEqual(recorded['complete_HC_origins'],10)
        recorded['history']['unique_timestamps']+=1
        with self.assertRaisesRegex(ValueError,'TIMESTAMP_OCCURRENCES'):audit.check_timestamps(recorded,raw)

    def test_legacy_parity_failure_is_nonrequired_while_new_forward_violation_fails(self):
        raw=fixture();_,_,state,_,_=producer.fit_state(raw,'PCA',1,device='cpu');_,outputs,ey,eyp,flatp=outputs_for(raw,state)
        row,independent=producer.evaluate_arrays(outputs,raw,state,ey,eyp,flatp)
        row['legacy_PCA_diagnostic']={'old_prediction_vs_refitted_new_prediction':{'pass':False},'parity_required_for_new_comparison_success':False}
        saved={**outputs,'EY_future':ey,'EY_prediction':eyp,'FLAT_future':np.repeat(outputs['latent_history'][:,-1:],48,axis=1),'FLAT_prediction':flatp,'oracle_valid_horizon_mask':producer.oracle_dependency_mask(raw['observed']),'origins':raw['origins'].copy()}
        self.assertTrue(audit.check_cell(row,saved,independent,raw,state)['legacy_PCA_parity_is_not_a_success_gate'])
        saved['prediction'][0,0,0]+=.1
        with self.assertRaisesRegex(ValueError,'actual_C'):audit.check_cell(row,saved,independent,raw,state)

    def test_closed_result_gate_precedes_IO_and_fresh_import_is_stdlib_only(self):
        with patch.object(audit,'RESULT_SHA',None),patch.object(audit.base,'file_record',side_effect=AssertionError('IO forbidden')):
            with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):audit.audit_result('missing',SimpleNamespace())
        code='import sys;sys.path.insert(0,'+repr(str(audit.HERE))+');import audit_predictive_cross_moment_v1;assert "numpy" not in sys.modules and "torch" not in sys.modules;print("STDLIB_AUDIT_IMPORT_PASS")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True);self.assertIn('STDLIB_AUDIT_IMPORT_PASS',result.stdout)


if __name__=='__main__':unittest.main()
