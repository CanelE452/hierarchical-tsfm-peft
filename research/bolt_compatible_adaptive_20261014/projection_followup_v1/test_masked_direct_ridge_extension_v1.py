"""Small CPU synthetic checks for the explicit positive-alpha copy and zero endpoints."""
from pathlib import Path
import ast
import copy
import unittest

import numpy as np
import masked_direct_ridge_v1 as original
import masked_direct_ridge_extension_v1 as extension


class RidgeExtensionMathTests(unittest.TestCase):
    def test_source_algorithm_copy_changes_only_grid_and_solver_description(self):
        before=Path(original.__file__).read_text(encoding="utf-8")
        expected=before.replace("ALPHAS=(0.,1e-4,.001,.01,.1,1.,10.)","ALPHAS=(100.,1000.,10000.)").replace("reused for seven alpha","reused for three extension alpha")
        self.assertEqual(expected,Path(extension.__file__).read_text(encoding="utf-8"));self.assertEqual(extension.ALPHAS,(100.,1000.,10000.));self.assertEqual(original.ALPHAS,(0.,1e-4,.001,.01,.1,1.,10.))

    def test_three_positive_solutions_equal_independent_augmented_lstsq_with_masks(self):
        rng=np.random.default_rng(613);x=rng.normal(size=(9,4)).astype(np.float32);x[:,3]=4.;y=rng.normal(size=(9,3)).astype(np.float64)+np.array([12.,-8.,2.]);mask=np.ones_like(y,bool);mask[:3,0]=False;mask[::3,1]=False
        for normalization in ("GLOBAL_FIT","PER_RESPONSE_MASK"):
            with self.subTest(normalization=normalization):
                states,metadata=extension.fit_path(x,y,mask,normalization=normalization);global_mean,global_scale,global_constant=extension.feature_scale(x.astype(np.float64))
                self.assertEqual(metadata["alpha_grid"],list(extension.ALPHAS));self.assertFalse(metadata["intercept_penalized"])
                for state in states:
                    for j in range(y.shape[1]):
                        ids=mask[:,j];a=x[ids].astype(np.float64);mean=a.mean(0);local=np.sqrt(np.mean((a-mean)**2,axis=0));scale=global_scale if normalization=="GLOBAL_FIT" else np.where(local==0,1.,local);constant=global_constant if normalization=="GLOBAL_FIT" else local==0;design=(a-mean)/scale;design[:,constant]=0.;target=y[ids,j];augmented=np.vstack((design,np.sqrt(ids.sum()*state["alpha"])*np.eye(x.shape[1])));response=np.concatenate((target-target.mean(),np.zeros(x.shape[1])));beta=np.linalg.lstsq(augmented,response,rcond=None)[0]/scale;beta[constant]=0
                        np.testing.assert_allclose(state["coefficient_FP64"][:,j],beta,atol=2e-14,rtol=2e-12);self.assertAlmostEqual(state["intercept_FP64"][j],target.mean()-mean@beta,places=12)

    def test_masked_placeholder_invariance_and_unpenalized_intercept(self):
        x=np.array([[0.,7.],[1.,7.],[2.,7.],[3.,7.],[4.,7.]],np.float32);y=np.array([[12.,21.],[14.,23.],[16.,25.],[18.,27.],[20.,29.]],np.float64);mask=np.ones_like(y,bool);mask[1,0]=False;mask[3,1]=False;changed=y.copy();changed[~mask]=[np.nan,np.inf]
        a,_=extension.fit_path(x,y,mask,normalization="GLOBAL_FIT");b,_=extension.fit_path(x,changed,mask,normalization="GLOBAL_FIT")
        for first,second in zip(a,b):
            for key in ("coefficient_FP64","intercept_FP64","coefficient_FP32","intercept_FP32"):np.testing.assert_array_equal(first[key],second[key])
            np.testing.assert_array_equal(first["coefficient_FP64"][1],0.)
            for j in range(2):self.assertAlmostEqual(float((x[mask[:,j]].astype(np.float64)@first["coefficient_FP64"][:,j]+first["intercept_FP64"][j]).mean()),y[mask[:,j],j].mean(),places=12)

    def test_exact_zero_proportional_and_small_singular_directions_are_finite(self):
        s=np.linspace(-1,1,7,dtype=np.float32);x=np.stack((s,2*s,np.ones_like(s),s*1e-9),axis=1);y=(s[:,None].astype(np.float64)*3+5);mask=np.ones_like(y,bool)
        states,metadata=extension.fit_path(x,y,mask,normalization="GLOBAL_FIT")
        self.assertTrue(metadata["positive_alpha_uses_all_singular_directions"])
        self.assertTrue(all(np.isfinite(v).all() for state in states for key,v in state.items() if key!="alpha"));self.assertTrue(all(state["coefficient_FP64"][2,0]==0 for state in states))
        norms=[np.linalg.norm(state["coefficient_FP64"]*metadata["global_feature_scale_FP64"][:,None]) for state in states];self.assertEqual(norms,sorted(norms,reverse=True))
        constant=np.ones((5,2),np.float32);states,_=extension.fit_path(constant,np.arange(5,dtype=np.float64)[:,None],np.ones((5,1),bool),normalization="GLOBAL_FIT")
        for state in states:np.testing.assert_array_equal(state["coefficient_FP64"],0.);np.testing.assert_array_equal(state["intercept_FP64"],[2.])

    def test_fused_fp32_predict_matches_declared_arithmetic_and_original_function(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(8,3)).astype(np.float32);y=rng.normal(size=(8,2)).astype(np.float64);m=np.ones_like(y,bool);states,_=extension.fit_path(x,y,m,normalization="GLOBAL_FIT")
        for state in states:
            actual=extension.predict(x,state);np.testing.assert_array_equal(actual,x@state["coefficient_FP32"]+state["intercept_FP32"]);np.testing.assert_array_equal(actual,original.predict(x,state));self.assertEqual(actual.dtype,np.float32)


if __name__=="__main__":unittest.main()
