"""FIT-only identity ridge around cached latent F; no data/model loaders or import execution."""
import numpy as np
import torch

LAGS=(0,1,2,4,8,16,32)
LAMBDAS=(0.,1e-6,1e-4,1e-2,.1,1.,10.)
FAMILIES=("SCALE_ONLY","SCALE_BIAS")
ZERO_LAMBDA_RCOND=1e-12
DESIGN_ATOL,DESIGN_RTOL=1e-10,5e-12
DEPLOYMENT_ATOL,DEPLOYMENT_RTOL=1e-5,2e-6


def require(value,message):
    if not value: raise ValueError(message)


def design_columns(zpast,zfuture,decoder,lags=LAGS,family="SCALE_ONLY"):
    p,u,d=(np.asarray(v) for v in (zpast,zfuture,decoder)); lags=tuple(lags)
    require(lags==LAGS and family in FAMILIES,"REGISTERED_L7_AND_CALIBRATION_FAMILY_REQUIRED")
    require(p.dtype==u.dtype==d.dtype==np.float32 and p.ndim==u.ndim==3 and d.ndim==2 and p.shape[0]==u.shape[0]>0 and p.shape[2]==u.shape[2]>0 and p.shape[1]>32 and u.shape[1]>0 and d.shape[0]==7*p.shape[2] and d.shape[1]>0 and all(np.isfinite(v).all() for v in (p,u,d)),"FINITE_MATCHED_FP32_PAST_F_FUTURE_AND_L7_D_REQUIRED")
    n,h,k=u.shape; c=d.shape[1]; a=np.zeros((n,h,c,k),np.float64); b=np.zeros_like(a)
    innovation=u.astype(np.float64)-p[:,-1:,].astype(np.float64); d=d.astype(np.float64).reshape(7,k,c)
    for li,lag in enumerate(lags):
        if lag>=h: continue
        a[:,lag:]+=np.einsum("nhk,kc->nhck",innovation[:,:h-lag],d[li])
        b[:,lag:]+=d[li].T[None,None]
    result=a if family=="SCALE_ONLY" else np.concatenate((a,b),axis=3)
    require(np.isfinite(result).all(),"NONFINITE_FP64_DESIGN"); return result


def identity_candidate(k,family):
    require(type(k) is int and k>0 and family in FAMILIES,"IDENTITY_DIMENSION_AND_FAMILY")
    p=k if family=="SCALE_ONLY" else 2*k
    return {"id":"IDENTITY","kind":"IDENTITY","family":family,"lambda":None,"status":"COMPLETE","a_FP64":np.zeros(k,np.float64),"b_FP64":np.zeros(k,np.float64),"a_FP32":np.zeros(k,np.float32),"b_FP32":np.zeros(k,np.float32),"coefficients_FP64":np.zeros(p,np.float64),"coefficients_FP32":np.zeros(p,np.float32),"metadata":{"identity_baseline_byte_exact_separate_path":True}}


def fit_calibration(zpast,zfuture,decoder,baseline,target,observed,lags=LAGS,family="SCALE_ONLY"):
    phi=design_columns(zpast,zfuture,decoder,lags,family); baseline,target,observed=(np.asarray(v) for v in (baseline,target,observed))
    require(baseline.dtype==target.dtype==np.float32 and observed.dtype==np.bool_ and baseline.shape==target.shape==observed.shape==phi.shape[:3] and np.isfinite(baseline).all() and np.isfinite(target[observed]).all(),"ACTUAL_FP32_BASELINE_OBSERVED_FIT_TARGET_REQUIRED")
    counts=observed.sum(axis=(0,1)); active=counts>0; require(active.any(),"NO_OBSERVED_FIT_TARGETS")
    channel_weights=np.zeros(len(counts),np.float64); channel_weights[active]=1/(int(active.sum())*counts[active])
    w=np.broadcast_to(channel_weights,target.shape)[observed]; x=phi[observed]; r=target[observed].astype(np.float64)-baseline[observed].astype(np.float64)
    require(np.isclose(w.sum(),1.,atol=1e-13,rtol=0),"FIT_CHANNEL_MACRO_WEIGHTS_SUM_ONE")
    retained=np.any(x!=0,axis=0); scales=np.sqrt(np.sum(w[:,None]*x*x,axis=0))
    require(np.isfinite(scales).all() and np.all(scales[retained]>0),"NONZERO_FEATURE_RMS_UNDERFLOW_OR_NONFINITE")
    root_w=np.sqrt(w); response=root_w*r
    matrix=root_w[:,None]*(x[:,retained]/scales[retained])
    if retained.any(): left,singular,right=np.linalg.svd(matrix,full_matrices=False); projected=left.T@response
    else: singular=np.empty(0,np.float64); right=np.empty((0,0),np.float64); projected=np.empty(0,np.float64)
    rank_mask=singular>singular[0]*ZERO_LAMBDA_RCOND if len(singular) else np.zeros(0,bool)
    k=zfuture.shape[2]; candidates=[]
    for lam in LAMBDAS:
        factors=np.zeros_like(singular)
        if lam==0: factors[rank_mask]=1/singular[rank_mask]
        else: factors=singular/(singular*singular+lam)
        scaled=right.T@(factors*projected); coefficients=np.zeros(phi.shape[3],np.float64); coefficients[retained]=scaled/scales[retained]
        with np.errstate(over="ignore",invalid="ignore"): deploy=coefficients.astype(np.float32)
        a,b=coefficients[:k],coefficients[k:] if family=="SCALE_BIAS" else np.zeros(k,np.float64)
        a32,b32=deploy[:k],deploy[k:] if family=="SCALE_BIAS" else np.zeros(k,np.float32)
        finite=bool(np.isfinite(coefficients).all() and np.isfinite(deploy).all())
        residual=response-matrix@scaled
        scalar={"standardized_coefficient_L2":float(np.linalg.norm(scaled)),"coefficient_L2":float(np.linalg.norm(coefficients)),"coefficient_max_abs":float(np.max(np.abs(coefficients))),"surrogate_FIT_macro_MSE":float(residual@residual),"ridge_penalty":float(lam*(scaled@scaled))}
        nonfinite=[name for name,value in scalar.items() if not np.isfinite(value)]; scalar={name:value if np.isfinite(value) else None for name,value in scalar.items()}
        candidates.append({"id":"RIDGE_"+format(lam,"g"),"kind":"RIDGE","family":family,"lambda":lam,"status":"COMPLETE" if finite else "FAILED_NONFINITE_COEFFICIENTS_PRESERVED","a_FP64":a.copy(),"b_FP64":b.copy(),"a_FP32":a32.copy() if finite else None,"b_FP32":b32.copy() if finite else None,"coefficients_FP64":coefficients,"coefficients_FP32":deploy if finite else None,"metadata":{**scalar,"nonfinite_scalar_fields":nonfinite,"negative_total_scale_channels":np.flatnonzero(1+a<0).tolist(),"singular_filter":"relative_cutoff_1e-12" if lam==0 else "all_singular_values_sigma_over_sigma2_plus_lambda","coefficient_clipping_applied":False}})
    return {"family":family,"identity":identity_candidate(k,family),"candidates":candidates,"column_scales_FP64":scales,"active_columns":retained,"singularvalues_FP64":singular,"metadata":{"FIT_channel_observed_counts":counts.tolist(),"active_FIT_channels":int(active.sum()),"observed_FIT_rows":int(observed.sum()),"row_weight_sum":float(w.sum()),"exact_zero_FIT_columns":np.flatnonzero(~retained).tolist(),"FIT_column_RMS_only":True,"column_or_response_centered":False,"response_is_actual_cached_baseline_residual":True,"identity_centered_regularization_on_FIT_RMS_scaled_coefficients":True,"zero_lambda_relative_cutoff":ZERO_LAMBDA_RCOND,"zero_lambda_numeric_rank":int(rank_mask.sum()),"positive_lambda_discards_small_singular_values":False,"lambdas":list(LAMBDAS),"SVD_fits":int(retained.any()),"coefficient_candidates":len(candidates),"E_D_F_fits":0,"future_label_contract":"Caller must bind regression responses to FIT; API has no holdout-label arguments","coefficients_identified_unique_latent_causal_effects_claimed":False,"FP64_design_check_atol":DESIGN_ATOL,"FP64_design_check_rtol":DESIGN_RTOL,"independent_FP32_deployment_parity_atol":DEPLOYMENT_ATOL,"independent_FP32_deployment_parity_rtol":DEPLOYMENT_RTOL,"actual_FP32_equals_linear_surrogate_required":False}}


def apply_calibration(zpast,zfuture,a,b):
    require(isinstance(zpast,torch.Tensor) and isinstance(zfuture,torch.Tensor) and zpast.dtype==zfuture.dtype==torch.float32 and zpast.ndim==zfuture.ndim==3 and zpast.shape[0]==zfuture.shape[0] and zpast.shape[2]==zfuture.shape[2] and zpast.shape[1]>32 and zpast.device==zfuture.device and torch.isfinite(zpast).all() and torch.isfinite(zfuture).all(),"FINITE_MATCHED_FP32_TORCH_LATENT_PAST_AND_FUTURE")
    a,b=(torch.as_tensor(v,dtype=torch.float32,device=zfuture.device) for v in (a,b))
    require(a.shape==b.shape==(zfuture.shape[2],) and torch.isfinite(a).all() and torch.isfinite(b).all(),"FINITE_PER_K_FP32_CALIBRATION_PARAMETERS")
    result=zfuture+a*(zfuture-zpast[:,-1:])+b
    require(torch.isfinite(result).all(),"NONFINITE_FP32_CALIBRATED_LATENT_FUTURE"); return result


def predict_calibrated(adapter,currentC,zpast,zfuture,candidate,actual_baseline):
    require(tuple(adapter.lags)==LAGS and candidate["status"]=="COMPLETE" and candidate["kind"] in ("IDENTITY","RIDGE"),"REGISTERED_COMPLETE_CALIBRATION_CANDIDATE")
    if candidate["kind"]=="IDENTITY":
        require(isinstance(actual_baseline,torch.Tensor) and actual_baseline.dtype==torch.float32 and actual_baseline.shape==(len(zpast),zfuture.shape[1],adapter.C) and actual_baseline.device==zfuture.device and torch.isfinite(actual_baseline).all() and np.all(candidate["a_FP64"]==0) and np.all(candidate["b_FP64"]==0),"EXACT_IDENTITY_BASELINE_AND_ZERO_PARAMETERS")
        return actual_baseline,zfuture
    calibrated=apply_calibration(zpast,zfuture,candidate["a_FP32"],candidate["b_FP32"])
    return adapter.decode(currentC,zpast,calibrated),calibrated


def select_candidate(records):
    valid=[r for r in records if r.get("status")=="COMPLETE" and np.isfinite(r.get("TUNE_MSE",np.nan))]
    require(bool(valid),"NO_FINITE_COMPLETE_TUNE_CANDIDATE")
    require(all(r["kind"] in ("IDENTITY","RIDGE") and (r["kind"]=="IDENTITY" or r["lambda"] in LAMBDAS) for r in valid),"REGISTERED_TUNE_CANDIDATE_TYPES_AND_LAMBDAS")
    return min(valid,key=lambda r:(r["TUNE_MSE"],0 if r["kind"]=="IDENTITY" else 1,-r["lambda"] if r["kind"]=="RIDGE" else 0))


def surrogate_diagnostics(prediction,actual_baseline,phi,coefficients):
    prediction,baseline,phi,coefficients=(np.asarray(v) for v in (prediction,actual_baseline,phi,coefficients))
    require(prediction.shape==baseline.shape==phi.shape[:3] and coefficients.shape==(phi.shape[3],) and all(np.isfinite(v).all() for v in (prediction,baseline,phi,coefficients)),"FINITE_MATCHED_SURROGATE_DIAGNOSTIC_INPUTS")
    surrogate=baseline.astype(np.float64)+phi@coefficients.astype(np.float64); error=prediction.astype(np.float64)-surrogate
    return {"max_abs_actual_FP32_minus_linear_surrogate":float(np.max(np.abs(error))),"RMS_actual_FP32_minus_linear_surrogate":float(np.sqrt(np.mean(error*error))),"per_channel_max_abs":np.max(np.abs(error),axis=(0,1)).tolist(),"surrogate_zero_error_required":False,"linear_surrogate_is_actual_deployment_claimed":False}
