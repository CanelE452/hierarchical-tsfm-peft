"""Independently rescore bound historical CUDA PCA controls without refitting."""
from pathlib import Path

from .audit import verify_old_job
from .contracts import atomic_json,binding,resolve_binding,sha,utc


def execute_job(job,ctx):
    import numpy as np
    import torch
    from research.adapter_diagnostics_20261017_v2.existing_jobs import score_arrays,_array_fingerprint
    seed=int(job["seed"])
    previous,evidence=verify_old_job(f"C02_s{seed}",ctx)
    original=next(row for row in previous["rows"] if row["method"]=="B2_PCA")
    cache=Path(ctx["previous_cache"])
    bundle_path=cache/"runs"/f"init_PCA_s{seed}"/"selected_bundle.pt"
    prediction_path=cache/"initialization_predictions"/f"PCA_{seed}.npz"
    if sha(prediction_path)!=original["prediction_sha256"]:
        raise ValueError("Reused prediction archive differs from original row binding")
    bundle=torch.load(bundle_path,map_location="cpu",weights_only=False)
    spec=bundle["spec"]
    recipe=dict(family="B2",init_kind="PCA",latent_k=6,seed=seed,device="cuda",epochs=[.25,1.,3.],
                lr=.001,microbatch=4,effective_batch=16,F_row_microbatch=128,checkpoint_inputs=False)
    if any(spec.get(key)!=value for key,value in recipe.items()):
        raise ValueError("Reused control differs from registered CUDA training recipe")
    with np.load(prediction_path,allow_pickle=False) as archive:
        values={key:archive[key] for key in archive.files}
    metrics=score_arrays(values["pred_normalized"],values["target_normalized"],values["observed"])
    if any(abs(metrics[key]-original[key])>1e-12 for key in ("MSE","MAE")):
        raise ValueError("Reused saved predictions fail independent FP64 rescore")
    fingerprints={key:_array_fingerprint(value) for key,value in values.items()}
    row=dict(original,**metrics)
    row.update(run_id=job["job_id"],job_id=job["job_id"],original_run_id=spec["run_id"],
        status="REUSED_VERIFIED",stage="selected_refit",evaluation="V_select",fit="fresh_A60",
        prediction_binding=binding(prediction_path,ctx),checkpoint_binding=binding(bundle_path,ctx),
        array_fingerprints=fingerprints,independent_FP64="PASS",new_training_updates=0,new_F_calls=0,
        source_sha=bundle["contract"]["source_sha"],config_sha=bundle["contract"]["config_sha"],
        data_sha=bundle["contract"]["data_sha"],model_sha=bundle["contract"]["model_sha"],
        selection_split="A_stop",V_exposure="previously_reused_development",
        core_dtype="float32",prediction_dtype=str(values["pred_normalized"].dtype),
        metric_accumulation_dtype="float64",target_trained_params=35717,
        updates=original["refit_updates"],exposures=original["refit_exposures"],
        search_seconds=bundle["search_summary"]["seconds"],refit_seconds=bundle["refit_summary"]["seconds"])
    endpoints=[]
    for old in previous["exact_endpoint_curves"]:
        if old["initialization"]!="PCA":
            continue
        endpoints.append(dict(old,run_id=job["job_id"],method="B2_PCA",dataset="jena",seed=seed,
            stage="search",evaluation="A_stop",fit="A_fit",MSE=old["A_stop_MSE"],
            MAE=None,MAE_status="UNKNOWN_NOT_SAVED_AT_ORIGINAL_A_STOP_ENDPOINT",
            status="REUSED_VERIFIED",new_F_calls=0,
            source_binding=evidence["public_result"],
            score_scope="Original saved A_stop MSE bound to actual search checkpoint; no new endpoint predictions"))
    result=dict(status="REUSED_VERIFIED",checked_utc=utc(),rows=[row],common_endpoint_rows=endpoints,
        historical_evidence=evidence,bundle_binding=binding(bundle_path,ctx),
        private_artifacts=[binding(prediction_path,ctx),binding(bundle_path,ctx)],
        recipe=recipe,source_reuse="Original six scientific files unchanged; new initialization/factory is explicit intervention",
        comparison_scope="Matched CUDA PCA recipe; verify DCA same P/scaler/sampler/backend against original diagnostics",
        saved_array_independent_FP64="PASS",new_F_calls=0,new_training_updates=0)
    output=Path(ctx["report"])/"jobs"/(job["job_id"]+".json")
    atomic_json(output,result)
    return dict(status="REUSED_VERIFIED",reason="Bound existing PCA artifacts independently rescored; no refit or F",artifact_paths=[str(output)],result=result)
