import unittest
import numpy as np
import masked_direct_ridge_v1 as ridge


class MaskedDirectRidgeTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(92901)
        x=rng.normal(size=(15,7)).astype(np.float32);x[:,2]=3
        y=rng.normal(size=(15,4));m=np.ones_like(y,bool);m[:3,1:3]=False
        return x,y,m

    def test_all_positive_alpha_match_independent_normal_equations_with_masks(self):
        x,y,m=self.data()
        for norm in ("GLOBAL_FIT","PER_RESPONSE_MASK"):
            states,meta=ridge.fit_path(x,y,m,normalization=norm)
            for state in states[1:]:
                for j in range(4):
                    a=x[m[:,j]].astype(float);b=y[m[:,j],j];mean=a.mean(0)
                    ref=x.astype(float) if norm=="GLOBAL_FIT" else a
                    scale=np.sqrt(np.mean((ref-ref.mean(0))**2,axis=0));scale[scale==0]=1
                    z=(a-mean)/scale
                    c=np.linalg.solve(z.T@z+state['alpha']*len(a)*np.eye(7),z.T@(b-b.mean()))/scale
                    np.testing.assert_allclose(state['coefficient_FP64'][:,j],c,rtol=1e-8,atol=1e-9)
                    self.assertAlmostEqual(state['intercept_FP64'][j],b.mean()-mean@c,places=9)
            self.assertEqual(len(meta['groups']),2)

    def test_alpha_zero_minimum_norm_and_fused_FP32_predict(self):
        x,y,m=self.data();x=np.concatenate([x,x[:,:4]],axis=1)
        states,_=ridge.fit_path(x,y,m,normalization="GLOBAL_FIT")
        for j in range(4):
            a=x[m[:,j]].astype(float);mean=a.mean(0);_,scale,_=ridge.feature_scale(x.astype(float))
            c=np.linalg.lstsq((a-mean)/scale,y[m[:,j],j]-y[m[:,j],j].mean(),rcond=1e-8)[0]/scale
            np.testing.assert_allclose(states[0]['coefficient_FP64'][:,j],c,atol=1e-9)
        p=ridge.predict(x,states[2]);self.assertEqual(p.dtype,np.float32)
        np.testing.assert_array_equal(p,x@states[2]['coefficient_FP32']+states[2]['intercept_FP32'])

    def test_unobserved_nan_and_inf_placeholders_are_irrelevant(self):
        x,y,m=self.data();other=y.copy();other[~m]=np.inf
        a,_=ridge.fit_path(x,y,m,normalization="GLOBAL_FIT")
        b,_=ridge.fit_path(x,other,m,normalization="GLOBAL_FIT")
        for aa,bb in zip(a,b):
            np.testing.assert_array_equal(aa['coefficient_FP32'],bb['coefficient_FP32'])
            np.testing.assert_array_equal(aa['intercept_FP32'],bb['intercept_FP32'])

    def test_constant_features_and_response_keep_intercept_unpenalized(self):
        x=np.ones((5,8),np.float32);y=np.full((5,2),7.,np.float64);m=np.ones_like(y,bool)
        states,_=ridge.fit_path(x,y,m,normalization="GLOBAL_FIT")
        for state in states:
            np.testing.assert_array_equal(state['coefficient_FP32'],np.zeros((8,2),np.float32))
            np.testing.assert_array_equal(ridge.predict(x,state),y.astype(np.float32))

    def test_invalid_observed_or_feature_data_fail_explicitly(self):
        x,y,m=self.data();m[:,0]=False
        with self.assertRaisesRegex(ValueError,"TWO_TRUE"):
            ridge.fit_path(x,y,m,normalization="GLOBAL_FIT")
        x[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"FINITE_FP32"):
            ridge.fit_path(x,y,m,normalization="GLOBAL_FIT")


if __name__=="__main__":unittest.main()
