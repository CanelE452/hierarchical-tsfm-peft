"""Masked multioutput ridge paths with FIT-only scaling and fused FP32 readout."""
import numpy as np

ALPHAS=(0.,1e-4,.001,.01,.1,1.,10.)


def require(value,message):
    if not value:
        raise ValueError(message)


def feature_scale(x):
    center=x.mean(0)
    rms=np.sqrt(np.mean((x-center)**2,axis=0))
    constant=rms==0
    return center,np.where(constant,1.,rms),constant


def fit_path(features,response,observed,*,normalization):
    x,y,m=map(np.asarray,(features,response,observed))
    require(x.dtype==np.float32 and x.ndim==2 and len(x)>1 and x.shape[1]>0
            and np.isfinite(x).all(),"FINITE_FP32_NP_FIT_FEATURES")
    require(y.dtype==np.float64 and y.ndim==2 and y.shape[0]==len(x)
            and m.dtype==np.bool_ and m.shape==y.shape and np.isfinite(y[m]).all(),
            "MATCHED_FP64_OBSERVED_RESPONSE_AND_BOOL_MASK")
    require(normalization in ("GLOBAL_FIT","PER_RESPONSE_MASK"),"REGISTERED_FIT_NORMALIZATION")
    x64=x.astype(np.float64);global_mean,global_rms,global_constant=feature_scale(x64)
    p,d=x.shape[1],y.shape[1]
    states=[{"alpha":a,"coefficient_FP64":np.zeros((p,d)),"intercept_FP64":np.zeros(d)} for a in ALPHAS]
    groups={}
    for j in range(d):
        groups.setdefault(m[:,j].tobytes(),[]).append(j)
    diagnostics=[]
    for columns in groups.values():
        valid=m[:,columns[0]];n=int(valid.sum())
        require(n>=2,"TWO_TRUE_OBSERVED_FIT_RESPONSES_PER_COLUMN_REQUIRED")
        a,b=x64[valid],y[np.ix_(valid,columns)]
        local_mean,local_rms,local_constant=feature_scale(a)
        scale,constant=(global_rms,global_constant) if normalization=="GLOBAL_FIT" else (local_rms,local_constant)
        design=(a-local_mean)/scale
        design[:,constant]=0
        mean_y=b.mean(0);centered=b-mean_y
        u,s,vt=np.linalg.svd(design,full_matrices=False)
        keep=s>s[0]*1e-8 if len(s) and s[0]>0 else np.zeros_like(s,dtype=bool)
        projected=u.T@centered
        for state in states:
            alpha=state["alpha"];gain=np.zeros_like(s)
            if alpha==0:
                gain[keep]=1/s[keep]
            else:
                gain=s/(s*s+alpha*n)
            coefficient=((vt.T*gain)@projected)/scale[:,None]
            coefficient[constant]=0
            state["coefficient_FP64"][:,columns]=coefficient
            state["intercept_FP64"][columns]=mean_y-local_mean@coefficient
        diagnostics.append({"columns":columns,"observed_count":n,"observed_mask":valid.copy(),
                            "feature_mean_FP64":local_mean,"feature_scale_FP64":scale.copy(),
                            "constant_features":constant.copy(),"singularvalues":s,
                            "minimum_norm_retained_rank":int(keep.sum())})
    for state in states:
        state["coefficient_FP32"]=state["coefficient_FP64"].astype(np.float32)
        state["intercept_FP32"]=state["intercept_FP64"].astype(np.float32)
        require(all(np.isfinite(state[k]).all() for k in ("coefficient_FP64","intercept_FP64","coefficient_FP32","intercept_FP32")),"FINITE_FUSED_RIDGE_STATE")
    return states,{"groups":diagnostics,"global_feature_mean_FP64":global_mean,
                   "global_feature_scale_FP64":global_rms,"global_constant_features":global_constant,
                   "normalization":normalization,"alpha_grid":list(ALPHAS),"design_singular_rcond":1e-8,
                   "positive_alpha_uses_all_singular_directions":True,"intercept_penalized":False,
                   "objective":"observed response SSE/n + alpha*||standardized slope||^2",
                   "solver":"FP64 centered thin SVD per observed-mask group, reused for seven alpha",
                   "unobserved_response_placeholders_used":False,"coefficient_fused_to_original_feature_units":True,
                   "TUNE_DEV_used_for_fit":False}


def predict(features,state):
    x=np.asarray(features);c,b=state["coefficient_FP32"],state["intercept_FP32"]
    require(x.dtype==c.dtype==b.dtype==np.float32 and x.ndim==c.ndim==2
            and x.shape[1]==c.shape[0] and b.shape==(c.shape[1],)
            and np.isfinite(x).all(),"MATCHED_FINITE_FP32_DEPLOYMENT_FEATURES_AND_STATE")
    result=x@c+b
    require(np.isfinite(result).all(),"FINITE_ACTUAL_FP32_RIDGE_PREDICTION")
    return result
