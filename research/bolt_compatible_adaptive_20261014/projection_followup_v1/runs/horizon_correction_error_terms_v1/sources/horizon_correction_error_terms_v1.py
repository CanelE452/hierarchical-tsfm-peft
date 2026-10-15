"""Observed-channel decomposition of two actual FP32 forecast endpoints."""
import numpy as np

INTERVALS=(("1_16",0,16),("17_32",16,32),("33_48",32,48))
FIELDS=("baseline_MSE","selected_MSE","delta_MSE","interaction_term","correction_energy","direct_squared_error_change","closure_residual","direct_closure_residual","delta_minus_direct_change")
CLOSURE_ATOL,CLOSURE_RTOL=1e-12,1e-12


def require(value,message):
    if not value:raise ValueError(message)


def _values(baseline,selected,interaction,energy,direct):
    delta=selected-baseline;closure=delta-interaction-energy;direct_closure=direct-interaction-energy;difference=delta-direct
    tolerance=CLOSURE_ATOL+CLOSURE_RTOL*max(abs(baseline),abs(selected),abs(interaction),abs(energy),abs(direct))
    values=(baseline,selected,delta,interaction,energy,direct,closure,direct_closure,difference)
    return {**dict(zip(FIELDS,values)),"closure_tolerance":tolerance,"closure_pass":max(abs(closure),abs(direct_closure),abs(difference))<=tolerance}


def decompose(baseline,prediction,target,observed):
    require(all(isinstance(a,np.ndarray) for a in (baseline,prediction,target,observed)) and baseline.ndim==3 and baseline.shape==prediction.shape==target.shape==observed.shape and baseline.shape[2]>0 and baseline.dtype==prediction.dtype==target.dtype==np.float32 and observed.dtype==np.bool_ and np.isfinite(baseline).all() and np.isfinite(prediction).all() and np.isfinite(target[observed]).all(),"FINITE_FP32_ENDPOINTS_AND_OBSERVED_TARGETS_REQUIRED")
    counts=[];channels=[]
    for channel in range(target.shape[2]):
        valid=observed[:,:,channel];count=int(valid.sum());counts.append(count)
        if count:
            b=baseline[:,:,channel][valid].astype(np.float64);p=prediction[:,:,channel][valid].astype(np.float64);y=target[:,:,channel][valid].astype(np.float64);e0=b-y;c=p-b;e1=p-y
            values=_values(float(np.mean(e0*e0)),float(np.mean(e1*e1)),float(2*np.mean(e0*c)),float(np.mean(c*c)),float(np.mean(e1*e1-e0*e0)))
            require(values["closure_pass"],"FP64_ERROR_TERMS_CLOSURE");channels.append({"channel":channel,"observed_count":count,"status":"COMPLETE",**values})
        else:channels.append({"channel":channel,"observed_count":0,"status":"NO_OBSERVED_TARGETS",**{key:None for key in (*FIELDS,"closure_tolerance","closure_pass")}})
    active=[row for row in channels if row["observed_count"]]
    if active:
        values=_values(*(float(np.mean([row[key] for row in active])) for key in ("baseline_MSE","selected_MSE","interaction_term","correction_energy","direct_squared_error_change")))
        require(values["closure_pass"],"FP64_MACRO_ERROR_TERMS_CLOSURE")
    else:values={key:None for key in (*FIELDS,"closure_tolerance","closure_pass")}
    return {"status":"COMPLETE" if active else "NO_OBSERVED_TARGETS","observed_counts":counts,"observed_values":sum(counts),"active_channels":len(active),"per_channel":channels,**values}


def diagnose(baseline,prediction,target,observed,*,require_zero=False):
    full=decompose(baseline,prediction,target,observed)
    require(baseline.shape[1]==48,"REGISTERED_H48_REQUIRED")
    byte_equal=baseline.tobytes()==prediction.tobytes()
    if require_zero:require(byte_equal,"ZERO_ORIGINAL_BASELINE_BYTES")
    return {"full":full,"per_horizon":[decompose(baseline[:,h:h+1],prediction[:,h:h+1],target[:,h:h+1],observed[:,h:h+1]) for h in range(48)],"registered_intervals":{key:decompose(baseline[:,start:stop],prediction[:,start:stop],target[:,start:stop],observed[:,start:stop]) for key,start,stop in INTERVALS},"prediction_baseline_byte_equal":byte_equal,"actual_return_FP32_endpoints_interpreted_as_FP64":True,"horizon_macros_averaged_to_reconstruct_full":False,"unobserved_targets_used":False}
