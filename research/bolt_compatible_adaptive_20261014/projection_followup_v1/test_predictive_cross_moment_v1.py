import unittest

import numpy as np

import predictive_cross_moment_v1 as p


class PredictiveCrossMomentTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(92801)
        x = rng.normal(size=(12,512,3)).astype(np.float32)
        y = np.broadcast_to(x[:, -1:, [1,2,0]], (12,48,3)).copy()
        y += rng.normal(scale=.02,size=y.shape).astype(np.float32)
        return x, np.ones_like(x,bool), y, np.ones_like(y,bool)

    def test_whitening_and_independent_moment(self):
        x,xm,y,ym = self.data()
        state = p.fit_directions(x,xm,y,ym,2,kind="CROSS_MOMENT")
        z = (x.astype(float)-state["center_FP64"]) @ state["encoder_FP64"]
        self.assertTrue(np.allclose(z.reshape(-1,2).T@z.reshape(-1,2)/len(z.reshape(-1,2)),np.eye(2),atol=1e-7))
        a,b = x[:,-1].astype(float), y[:,7].astype(float)
        manual = sum(np.outer(a[i]-a.mean(0),b[i]-b.mean(0)) for i in range(len(a)))/len(a)
        np.testing.assert_allclose(state["cross_moments_FP64"][7],manual,atol=1e-14)
        self.assertEqual(p.encode(x,state).shape,(12,512,2))
        for j in range(2):
            self.assertGreater(state["E"][np.argmax(np.abs(state["E"][:,j])),j],0)

    def test_unobserved_placeholder_and_horizon_complete_mask(self):
        x,xm,y,ym = self.data()
        ym[0,3,1] = False
        y2=y.copy();y2[~ym]=np.nan
        a=p.fit_directions(x,xm,y,ym,2,kind="CROSS_MOMENT")
        b=p.fit_directions(x,xm,y2,ym,2,kind="CROSS_MOMENT")
        np.testing.assert_array_equal(a["E"],b["E"])
        self.assertEqual(a["metadata"]["horizon_complete_C_origins"][3],11)
        self.assertEqual(a["metadata"]["horizon_complete_C_origins"][2],12)

    def test_forward_only_needs_past_and_fixed_state(self):
        x,xm,y,ym=self.data()
        state=p.fit_directions(x,xm,y,ym,2,kind="CROSS_MOMENT")
        before=p.encode(x,state)
        y[:]=1e8;ym[:]=False
        np.testing.assert_array_equal(before,p.encode(x,{"center":state["center"],"E":state["E"]}))

    def test_pca_direction_nested_and_not_response_fitted(self):
        x,xm,y,ym=self.data()
        a=p.fit_directions(x,xm,y,ym,1,kind="PCA")
        b=p.fit_directions(x,xm,y*100,ym,2,kind="PCA")
        np.testing.assert_array_equal(a["E"],b["E"][:,:1])

    def test_invalid_mask_rank_and_k_preserved_as_failure(self):
        x,xm,y,ym=self.data()
        with self.assertRaisesRegex(ValueError,"BOOL_FIT"):
            p.fit_directions(x,xm,y,ym.astype(int),2,kind="CROSS_MOMENT")
        with self.assertRaisesRegex(ValueError,"TWO_COMPLETE"):
            p.fit_directions(x,xm,y,np.zeros_like(ym),2,kind="CROSS_MOMENT")
        with self.assertRaisesRegex(ValueError,"RANK_BELOW"):
            p.fit_directions(x,xm,np.zeros_like(y),ym,2,kind="CROSS_MOMENT")
        with self.assertRaisesRegex(ValueError,"INTEGER_K"):
            p.fit_directions(x,xm,y,ym,True,kind="PCA")


if __name__=="__main__":
    unittest.main()
