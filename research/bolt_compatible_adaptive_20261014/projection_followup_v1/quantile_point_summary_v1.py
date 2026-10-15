"""Three fixed point summaries of nine stored marginal quantiles; no fitted mean claim."""
SUMMARIES=("MEDIAN","NINE_MEAN_PROXY","TAIL_CLAMPED_PROXY")
QUANTILES=tuple(i/10 for i in range(1,10))
WEIGHTS={"MEDIAN":(0.,0.,0.,0.,1.,0.,0.,0.,0.),"NINE_MEAN_PROXY":(1/9,)*9,
         "TAIL_CLAMPED_PROXY":(.15,.1,.1,.1,.1,.1,.1,.1,.15)}


def summarize_numpy(quantiles,summary):
    import numpy as np
    if summary not in SUMMARIES or quantiles.dtype!=np.float32 or quantiles.ndim!=4 or quantiles.shape[-1]!=9 or not np.isfinite(quantiles).all():raise ValueError("FINITE_N_H_K_NINE_FP32_QUANTILES_REQUIRED")
    if summary=="MEDIAN":return quantiles[...,4]
    return np.sum(quantiles*np.asarray(WEIGHTS[summary],dtype=np.float32),axis=-1,dtype=np.float32)


def summarize_torch(quantiles,summary):
    import torch
    if summary not in SUMMARIES or quantiles.dtype!=torch.float32 or quantiles.ndim!=4 or quantiles.shape[-1]!=9 or not torch.isfinite(quantiles).all():raise ValueError("FINITE_N_H_K_NINE_FP32_QUANTILES_REQUIRED")
    if summary=="MEDIAN":return quantiles[...,4]
    return (quantiles*quantiles.new_tensor(WEIGHTS[summary])).sum(dim=-1)


def quantile_diagnostics(quantiles):
    import numpy as np
    summarize_numpy(quantiles,"MEDIAN")
    difference=quantiles[...,1:].astype(np.float64)-quantiles[...,:-1].astype(np.float64); crossed=difference<0
    return {"shape":list(quantiles.shape),"dtype":str(quantiles.dtype),"quantile_levels":list(QUANTILES),
        "adjacent_pairs":int(difference.size),"crossing_pairs":int(crossed.sum()),"crossing_fraction":float(crossed.mean()),
        "maximum_adjacent_crossing":float(np.maximum(-difference,0).max(initial=0)),
        "crossing_pairs_by_level_pair":crossed.sum(axis=(0,1,2)).tolist(),"crossing_pairs_by_horizon":crossed.sum(axis=(0,2,3)).tolist(),
        "minimum":float(quantiles.min()),"maximum":float(quantiles.max()),"quantiles_sorted_after_inference":False,
        "finite_nine_quantiles_identify_conditional_mean":False,"decoded_C_quantiles_or_CRPS_claimed":False}


def select_summary(candidates):
    import math
    valid=[p for p in candidates if p["status"]=="COMPLETE" and math.isfinite(p.get("TUNE_MSE",float("nan")))]
    if not valid:raise ValueError("NO_FINITE_COMPLETE_TUNE_SUMMARY")
    return min(valid,key=lambda p:(p["TUNE_MSE"],SUMMARIES.index(p["summary"])))
