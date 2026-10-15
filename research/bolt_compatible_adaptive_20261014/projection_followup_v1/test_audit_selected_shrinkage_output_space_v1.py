"""Synthetic selected-D geometry/oracle/rounding boundaries, no actual assets."""
import json
import unittest

import numpy as np

import audit_selected_shrinkage_output_space_v1 as audit

audit.initialize_numpy()


def values_for(target,current=None,observed=None,scale=None):
    n,_,c=target.shape
    return {'target':target.astype(np.float32),'raw_last':np.zeros((n,c),np.float32) if current is None else current.astype(np.float32),
            'observed':np.ones_like(target,bool) if observed is None else observed,'std':np.ones(c,np.float32) if scale is None else np.asarray(scale,np.float32)}


class SelectedOutputSpaceFixtures(unittest.TestCase):
    def test_origin_specific_rank_one_can_flatten_to_rank_two(self):
        target=np.array([[[1.,0.]]*48,[[0.,1.]]*48],np.float32)
        result=audit.selected_real_space(values_for(target),np.array([[[1.,0.]],[[0.,1.]]],np.float32))
        self.assertEqual(result['rank_D_per_origin'],[1,1])
        self.assertLess(result['minimum_normalized_MSE_real_numeric_rowspace'],1e-28)
        self.assertEqual(np.linalg.matrix_rank(target.reshape(-1,2)),2)

    def test_EY_is_oblique_coordinate_choice_not_the_space_minimum(self):
        target=np.array([[[0.,1.]]*48],np.float32);values=values_for(target)
        d=np.array([[[1.,0.]]],np.float32)
        space=audit.selected_real_space(values,d)
        future=audit.encode_Y(target,np.zeros(2),np.array([[1.],[1.]]),'BOLT_NUMPY')
        last=np.zeros((1,1),np.float32)
        prediction=audit.local_audit.local_forecast(values['raw_last'],future,last,d)
        scored,_=audit.score_real_and_FP32(prediction,values['raw_last'],future,last,d,values,space,'E_Y_fixture')
        self.assertEqual(space['minimum_normalized_MSE_real_numeric_rowspace'],.5)
        self.assertEqual(scored['scores_complete_C']['normalized']['mse'],1.)

    def test_complete_C_mask_excludes_partially_missing_crosschannel_positions(self):
        target=np.ones((1,48,2),np.float32);target[0,0]=[np.nan,1000.]
        mask=np.ones_like(target,bool);mask[0,0,0]=False
        result=audit.selected_real_space(values_for(target,observed=mask),np.zeros((1,1,2),np.float32))
        self.assertEqual(result['scores']['normalized']['observed_counts'],[47,47])
        self.assertEqual(result['minimum_normalized_MSE_real_numeric_rowspace'],1.)

    def test_zero_rank_is_current_anchor_and_known_future_error(self):
        target=np.ones((2,48,3),np.float32)
        result=audit.selected_real_space(values_for(target),np.zeros((2,2,3),np.float32))
        self.assertEqual(result['rank_D_per_origin'],[0,0]);self.assertEqual(result['minimum_normalized_MSE_real_numeric_rowspace'],1.)
        np.testing.assert_array_equal(result['optimal_latent_delta_real64'],0.)

    def test_repeated_singular_values_do_not_require_vector_sign_identity(self):
        target=np.tile(np.array([1.,2.,3.],np.float32),(2,48,1))
        d=np.array([[[1.,0.,0.],[0.,1.,0.]],[[0.,1.,0.],[-1.,0.,0.]]],np.float32)
        result=audit.selected_real_space(values_for(target),d)
        self.assertEqual(result['rank_D_per_origin'],[2,2])
        np.testing.assert_allclose(result['projectors_real64'][0],result['projectors_real64'][1],atol=1e-14)
        self.assertEqual(result['minimum_normalized_MSE_real_numeric_rowspace'],3.)

    def test_numeric_rowspace_cutoff_is_registered_not_unlimited_exact_rank(self):
        target=np.array([[[0.,1.]]*48],np.float32)
        result=audit.selected_real_space(values_for(target),np.array([[[1.,0.],[0.,1e-16]]],np.float32))
        self.assertEqual(result['rank_D_per_origin'],[1]);self.assertEqual(result['minimum_normalized_MSE_real_numeric_rowspace'],.5)

    def test_FP32_rounding_can_beat_real_space_minimum_and_allowance_is_measured(self):
        current=np.array([[2**24,0.]],np.float32)
        target=np.broadcast_to(np.array([2**24,1.],np.float32),(1,48,2)).copy()
        values=values_for(target,current=current);d=np.ones((1,1,2),np.float32)
        future,last=np.ones((1,48,1),np.float32),np.zeros((1,1),np.float32)
        prediction=audit.local_audit.local_forecast(current,future,last,d)
        space=audit.selected_real_space(values,d)
        scored,_=audit.score_real_and_FP32(prediction,current,future,last,d,values,space,'rounding_counterexample')
        self.assertAlmostEqual(space['minimum_normalized_MSE_real_numeric_rowspace'],.25)
        self.assertEqual(scored['scores_complete_C']['normalized']['mse'],0.)
        self.assertEqual(scored['real_trace_scores_complete_C']['normalized']['mse'],.5)
        self.assertEqual(scored['measured_FP32_rounding_normalized_MSE_allowance'],.5)

    def test_MAE_of_MSE_solution_is_not_MAE_minimum(self):
        target=np.array([[[0.,0.,9.]]*48],np.float32);values=values_for(target)
        result=audit.selected_real_space(values,np.ones((1,1,3),np.float32))
        alternative=audit.prior.scores(np.zeros_like(target),target,values['observed'],values['std'])
        self.assertAlmostEqual(result['scores']['normalized']['mae'],4.)
        self.assertEqual(alternative['normalized']['mae'],3.)

    def test_physical_MSE_of_solution_is_not_weighted_physical_minimum(self):
        target=np.array([[[0.,3.]]*48],np.float32);values=values_for(target,scale=[10.,1.])
        result=audit.selected_real_space(values,np.ones((1,1,2),np.float32))
        alternative=audit.prior.scores(np.full_like(target,.1),target,values['observed'],values['std'])
        self.assertGreater(result['scores']['physical_per_channel_diagnostic']['mse'],alternative['physical_per_channel_diagnostic']['mse'])

    def test_Torch_CPU_chunk32_EY_and_NumPy_use_same_fixed_arrays(self):
        rng=np.random.default_rng(11)
        target=rng.normal(size=(35,48,3)).astype(np.float32)
        encoder=rng.normal(size=(3,2));center=rng.normal(size=3)
        native=audit.encode_Y(target,center,encoder,'NATIVE_TORCH32')
        numpy=audit.encode_Y(target,center,encoder,'BOLT_NUMPY')
        self.assertEqual(native.dtype,np.float32);np.testing.assert_allclose(native,numpy,atol=1e-5,rtol=2e-6)

    def test_empty_complete_mask_stops_diagnostic_instead_of_guessing(self):
        target=np.ones((1,48,2),np.float32);mask=np.ones_like(target,bool);mask[:,:,0]=False
        with self.assertRaisesRegex(ValueError,'NO_COMPLETE'):
            audit.selected_real_space(values_for(target,observed=mask),np.ones((1,1,2),np.float32))

    def test_changed_actualprediction_is_rejected_instead_of_canonicalizing_savedbytes(self):
        values=values_for(np.ones((1,48,2),np.float32));d=np.ones((1,1,2),np.float32)
        future,last=np.ones((1,48,1),np.float32),np.zeros((1,1),np.float32)
        actual=audit.local_audit.local_forecast(values['raw_last'],future,last,d);actual[0,0,0]+=.01
        with self.assertRaisesRegex(ValueError,'BYTE'):
            audit.score_real_and_FP32(actual,values['raw_last'],future,last,d,values,audit.selected_real_space(values,d),'modified_fixture')

    def test_geometry_metadata_JSON_is_finite_without_passing_arrays_to_JSON(self):
        result=audit.selected_real_space(values_for(np.ones((1,48,2),np.float32)),np.ones((1,1,2),np.float32))
        metadata={key:value for key,value in result.items() if not isinstance(value,np.ndarray)}
        json.loads(json.dumps(metadata,allow_nan=False))


if __name__=='__main__':unittest.main()
