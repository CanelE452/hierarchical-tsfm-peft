"""Small synthetic marginal-quantile fixtures only."""
import unittest
import numpy as np
import torch
import quantile_point_summary_v1 as summary


class QuantileSummaryTests(unittest.TestCase):
    def test_three_fixed_weights_and_median_preserves_signed_zero(self):
        q=np.arange(9,dtype=np.float32).reshape(1,1,1,9);q[...,4]=-0.
        self.assertEqual(tuple(summary.WEIGHTS),summary.SUMMARIES)
        self.assertEqual(summary.summarize_numpy(q,"MEDIAN").tobytes(),q[...,4].tobytes())
        for name in summary.SUMMARIES:
            with self.subTest(name=name):
                actual=summary.summarize_torch(torch.from_numpy(q),name).numpy();expected=summary.summarize_numpy(q,name)
                np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=2e-6)
        expected=np.float32(.15)*q[...,0]+sum(np.float32(.1)*q[...,i] for i in range(1,8))+np.float32(.15)*q[...,8]
        np.testing.assert_allclose(summary.summarize_numpy(q,"TAIL_CLAMPED_PROXY"),expected,atol=1e-6)

    def test_tail_clamped_integral_matches_piecewise_linear_and_constant_tail(self):
        q=np.array([-3,-2,0,1,2,4,5,9,15],dtype=np.float32).reshape(1,1,1,9)
        integral=.1*float(q[...,0].item())+.1*float(q[...,-1].item())
        integral+=sum(.1*(float(q[...,i].item())+float(q[...,i+1].item()))/2 for i in range(8))
        self.assertAlmostEqual(float(summary.summarize_numpy(q,"TAIL_CLAMPED_PROXY").item()),integral,places=5)
        self.assertNotEqual(float(summary.summarize_numpy(q,"NINE_MEAN_PROXY").item()),float(q[...,4].item()))

    def test_crossings_are_reported_and_never_sorted(self):
        q=np.broadcast_to(np.array([0,2,1,3,4,5,6,7,8],dtype=np.float32),(2,64,3,9)).copy();before=q.tobytes()
        info=summary.quantile_diagnostics(q);self.assertEqual(info['crossing_pairs'],2*64*3);self.assertEqual(info['maximum_adjacent_crossing'],1.)
        self.assertEqual(info['crossing_pairs_by_level_pair'][1],2*64*3);self.assertEqual(q.tobytes(),before)
        self.assertFalse(info['finite_nine_quantiles_identify_conditional_mean']);self.assertFalse(info['decoded_C_quantiles_or_CRPS_claimed'])

    def test_selection_uses_own_actual_TUNE_and_predeclared_exact_tie(self):
        rows=[{'summary':s,'status':'COMPLETE','TUNE_MSE':.2} for s in summary.SUMMARIES]
        self.assertEqual(summary.select_summary(rows)['summary'],'MEDIAN')
        rows[0]['status']='FAILED_PARITY_PRESERVED';self.assertEqual(summary.select_summary(rows)['summary'],'NINE_MEAN_PROXY')
        rows[2]['TUNE_MSE']=.1;self.assertEqual(summary.select_summary(rows)['summary'],'TAIL_CLAMPED_PROXY')
        for row in rows:row['status']='FAILED_PATH_PRESERVED'
        with self.assertRaisesRegex(ValueError,'NO_FINITE'):summary.select_summary(rows)

    def test_nonfinite_or_wrong_axis_rejected_without_implicit_repair(self):
        for q in (np.zeros((2,9,64,4),dtype=np.float32),np.zeros((2,64,4,9),dtype=np.float64),np.full((2,64,4,9),np.nan,dtype=np.float32)):
            with self.assertRaises(ValueError):summary.summarize_numpy(q,'MEDIAN')


if __name__=='__main__':unittest.main()
