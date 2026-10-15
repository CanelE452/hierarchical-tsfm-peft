"""Pure orthogonal latent-coordinate intervention; no model/data loading or execution."""
import numpy as np
import torch
from torch import nn

from linear_e_adaptation_v1 import LinearLagAdapter,require

FP64_ATOL,FP64_RTOL=1e-10,1e-9
FP32_ATOL,FP32_RTOL=1e-5,2e-6


def orthogonal_from_theta(theta):
    require(isinstance(theta,torch.Tensor) and theta.dtype in (torch.float32,torch.float64)
        and theta.shape==(6,) and torch.isfinite(theta).all(),"FINITE_SIX_THETA_FOR_K4_REQUIRED")
    upper=torch.triu_indices(4,4,offset=1,device=theta.device)
    skew=torch.zeros((4,4),dtype=theta.dtype,device=theta.device)
    skew[upper[0],upper[1]]=theta
    return torch.matrix_exp(skew-skew.T)


def fixed_orthogonal_controls():
    matrices={"IDENTITY":np.eye(4),"PERMUTATION":np.eye(4)[:,[1,0,3,2]],
        "SIGN_FLIP":np.diag([-1.,1.,1.,1.])}
    controls={name:{"Q":value,"metadata":{"kind":name,"seed":None,
        "determinant_FP64":float(np.linalg.det(value)),"identity_bypass":name=="IDENTITY"}}
        for name,value in matrices.items()}
    for seed in (92711,92712):
        q,r=np.linalg.qr(np.random.default_rng(seed).normal(size=(4,4)))
        q=q*np.where(np.diag(r)<0,-1.,1.)[None,:]
        determinant=float(np.linalg.det(q)); flipped=determinant<0
        if flipped: q[:,-1]*=-1
        name="DENSE_SO_s"+str(seed)
        controls[name]={"Q":q,"metadata":{"kind":"DENSE_SO","seed":seed,
            "source":"NumPy default_rng Gaussian 4x4 QR; make diag(R) positive, then flip last Q column iff det<0",
            "canonical_QR_determinant_FP64":determinant,"SO_last_column_flipped":flipped,
            "determinant_FP64":float(np.linalg.det(q)),"identity_bypass":False}}
    return controls


def analytic_transport(future,Q):
    require(isinstance(future,torch.Tensor) and isinstance(Q,torch.Tensor)
        and future.dtype==Q.dtype and future.dtype in (torch.float32,torch.float64)
        and future.ndim==3 and future.shape[-1]==4 and Q.shape==(4,4)
        and future.device==Q.device and torch.isfinite(future).all() and torch.isfinite(Q).all(),"MATCHED_FINITE_NH4_AND_4X4_TRANSPORT_REQUIRED")
    return (future@Q)@Q.T


class OrthogonalLatentBasisAdapter(LinearLagAdapter):
    def __init__(self,center,encoder,decoder,lags,Q=None,*,learn_Q=False,identity=False):
        super().__init__(center,encoder,decoder,lags,learn_E=False)
        require(self.K==4 and type(learn_Q) is bool and type(identity) is bool,"K4_AND_EXPLICIT_BOOLEAN_Q_MODE_REQUIRED")
        q=torch.eye(4,dtype=self.E.dtype,device=self.E.device) if Q is None else torch.as_tensor(Q,dtype=self.E.dtype,device=self.E.device).detach().clone()
        require(q.shape==(4,4) and torch.isfinite(q).all()
            and torch.allclose(q.T@q,torch.eye(4,dtype=q.dtype,device=q.device),atol=FP32_ATOL,rtol=FP32_RTOL),"FINITE_ORTHOGONAL_FIXED_Q_REQUIRED")
        require(not identity or (not learn_Q and torch.equal(q,torch.eye(4,dtype=q.dtype,device=q.device))),"IDENTITY_BYPASS_ONLY_FIXED_EXACT_I")
        require(not learn_Q or torch.equal(q,torch.eye(4,dtype=q.dtype,device=q.device)),"LEARNED_Q_INITIALIZES_AT_EXACT_I")
        self.learn_Q,self.identity=learn_Q,identity
        self.E.requires_grad_(False); self.D.requires_grad_(False)
        self.register_buffer("fixed_Q",q)
        self.theta=nn.Parameter(torch.zeros(6,dtype=q.dtype,device=q.device)) if learn_Q else None

    def basis(self):
        return orthogonal_from_theta(self.theta) if self.learn_Q else self.fixed_Q

    def frozen_status(self):
        trainable=[p for p in self.parameters() if p.requires_grad]
        require(not self.E.requires_grad and not self.D.requires_grad and self.E.grad is None and self.D.grad is None
            and ((self.learn_Q and len(trainable)==1 and trainable[0] is self.theta and self.theta.numel()==6)
                 or (not self.learn_Q and not trainable)),"ONLY_SIX_THETA_TRAINABLE_E_D_FROZEN")
        return {"E_trainable":False,"D_trainable":False,"center_is_buffer":True,
            "theta_trainable":self.learn_Q,"trainable_parameters":6 if self.learn_Q else 0,
            "fixed_identity_arithmetic_bypass":self.identity,"learned_zero_theta_bypass":False,
            "F_freezing_checked_by_controller":True,"future_observed_mask_and_loss_owned_by_controller":True}

    def basis_diagnostics(self):
        q=self.basis().detach().double().cpu()
        return {"finite":bool(torch.isfinite(q).all()),"orthogonality_max_abs":float((q.T@q-torch.eye(4,dtype=q.dtype)).abs().max()),
            "determinant":float(torch.linalg.det(q)),"identity_bypass":self.identity,
            "theta_L2":float(torch.linalg.vector_norm(self.theta.detach().double()).cpu()) if self.theta is not None else None,
            "old_past_and_decoder_unchanged":True,"input_information_added":False}

    def forward(self,x,bolt,*,return_components=False):
        old_past=self.encode(x)
        if self.identity: future=bolt(old_past)
        else:
            q=self.basis()
            future=bolt(old_past@q)@q.T
        prediction=self.decode(x[:,-1],old_past,future)
        return (prediction,old_past,future) if return_components else prediction
