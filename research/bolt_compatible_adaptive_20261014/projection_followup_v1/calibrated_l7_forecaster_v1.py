"""Frozen actual E/F/q/L7-D deployment; identity computes fresh F and decodes it."""
import torch

from linear_e_adaptation_v1 import LinearLagAdapter,require
from fixed_l7_latent_calibration_v1 import LAGS,FAMILIES,apply_calibration


class CalibratedL7Forecaster(LinearLagAdapter):
    def __init__(self,center,encoder,decoder,a,b,*,family,identity=False):
        super().__init__(center,encoder,decoder,LAGS,learn_E=False)
        a,b=(torch.as_tensor(v,dtype=torch.float32,device=self.E.device).detach().clone() for v in (a,b))
        require(family in FAMILIES and type(identity) is bool and a.shape==b.shape==(self.K,) and torch.isfinite(a).all() and torch.isfinite(b).all(),"REGISTERED_FINITE_PER_K_CALIBRATION_CONFIG")
        require(family!="SCALE_ONLY" or bool((b==0).all()),"SCALE_ONLY_HAS_NO_BIAS")
        require(not identity or bool((a==0).all() and (b==0).all()),"IDENTITY_HAS_EXACT_ZERO_COEFFICIENTS")
        self.register_buffer("a",a); self.register_buffer("b",b)
        self.family,self.identity=family,identity
        self.eval().requires_grad_(False)

    def calibration_config(self):
        return {"family":self.family,"identity":self.identity,"lags":list(self.lags),"K":self.K,"cached_baseline_returned":False,"identity_actual_F_then_current_D_decode":True}

    def forward(self,x,bolt,*,return_components=False):
        zpast=self.encode(x); original_future=bolt(zpast)
        future=original_future if self.identity else apply_calibration(zpast,original_future,self.a,self.b)
        prediction=self.decode(x[:,-1],zpast,future)
        return (prediction,zpast,future) if return_components else prediction
