"""Fresh G/LoRA AplusM controls: fixed A60 scale, V checkpoint selection."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

import data_contract
import fit_base
import training


def source_contract():
    engine_sha, engine_files = fit_base._source_contract()
    files = dict(engine_files, **{"fit_control.py": data_contract.sha256_file(Path(__file__))})
    return training.fingerprint(files), engine_sha, files


def fit_control(spec, task, cache_root, public_root, foundation, deadline):
    """Fit DIRECT_G or LORA freshly on every legal pre-V origin, never on T.

    Unlike calibration parents, these strong comparators use M labels directly.
    Their normalization and initialization statistics stay fixed at A60. No
    pretrained G/adapter parent weights or M predictions are accepted as input.
    """
    spec = copy.deepcopy(dict(spec))
    spec.setdefault("role", "AplusM_control")
    run_id = str(spec["run_id"])
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Safe run ID required")
    if spec.get("role") != "AplusM_control" or spec["family"] not in {"DIRECT_G", "LORA"}:
        raise ValueError("Only explicitly registered fresh DIRECT_G/LORA AplusM controls supported")
    epochs = list(spec.get("epochs", [.25, 1., 3.]))
    if not epochs or epochs != sorted(set(epochs)) or any(float(epoch) <= 0 for epoch in epochs):
        raise ValueError("Positive increasing epoch endpoints required")
    bounds = data_contract.split_ranges(task, spec.get("fold", "primary"))
    normalization_end, train_end = bounds["A_stop"][1], bounds["M_cal"][1]
    source_sha, engine_sha, files = source_contract()
    axes = dict(L=int(task["L"]), H=int(task["H"]), C=task["physical"].shape[1],
                columns=list(task["columns"]), fold=spec.get("fold", "primary"),
                normalization_end=normalization_end, train_end=train_end,
                evaluation_bounds=list(bounds["V_select"]), data_arrays=task.get("array_sha256", {}))
    run_contract = dict(source_sha=source_sha, engine_source_sha=engine_sha,
                        config_sha=training.fingerprint(dict(spec=spec, axes=axes)),
                        data_sha=task["data_sha256"], model_sha=fit_base._model_sha(foundation))
    private = Path(cache_root) / "runs" / run_id
    public = Path(public_root) / "runs" / run_id
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    contract_path = private / "CONTROL_RUN_CONTRACT.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != run_contract:
            raise ValueError("Control source/config/data/model changed; preserve old run and choose a new ID")
    else:
        fit_base._json_new(contract_path, run_contract)
        fit_base._json_new(public / "CONTROL_SOURCE_CONTRACT.json", dict(**run_contract, files=files, spec=spec))
    receipt_path = public / "CONTROL_FIT_RECEIPT.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        artifact = Path(cache_root) / receipt["artifact_cache_relative_path"]
        if not artifact.is_file() or data_contract.sha256_file(artifact) != receipt["artifact_sha256"]:
            raise ValueError("Complete control artifact absent or changed")
        return dict(receipt, reused_complete_artifact=True)
    # The shared stage engine checks its own science-source digest each CP.
    # Its checkpoint contract additionally pins the outer control source SHA.
    engine_contract = dict(run_contract, source_sha=engine_sha, control_source_sha=source_sha)
    model = None
    try:
        model, normalization = fit_base.create_model(spec, task, normalization_end, foundation)
        normalization["train_end"] = int(train_end)
        initial_sha = fit_base.networks.module_state_hash(model)
        stage = fit_base._stage_body(spec, task, train_end, bounds["V_select"], epochs,
                                     private / "control", public / "control", foundation,
                                     deadline, engine_contract, model, normalization)
    finally:
        if model is not None and callable(getattr(model, "close", None)):
            model.close()
        if foundation is not None and callable(getattr(foundation, "assert_frozen_unchanged", None)):
            foundation.assert_frozen_unchanged()
    if source_contract()[0] != source_sha:
        raise ValueError("Control scientific source changed during execution")
    summary = copy.deepcopy(stage["summary"])
    summary["selected_V_mse"] = summary.pop("selected_astop_mse")
    summary.update(checkpoint_selection_split="V_select", target_training_end=train_end,
                   normalization_fit_end=normalization_end, role="AplusM_control")
    if summary["status"] == "BUDGET_LIMIT":
        return dict(summary, run_id=run_id, training_status="UNDERTRAINED_OR_BUDGET_LIMIT")
    curves = [dict(epoch=row["epoch"], exposures=row["exposures"], updates=row["updates"],
                   V_mse=row["astop_mse"], segment_seconds=row["segment_seconds"], clipped_steps=row["clipped_steps"])
              for row in stage["curves"]]
    selected_epoch = summary["selected_epoch"]
    state_sha = training.fingerprint({key: data_contract.array_sha256(value.numpy()) for key, value in stage["model"].items()})
    selected_contract = dict(summary["contract"], source_sha=source_sha, engine_source_sha=engine_sha)
    artifact = private / "selected_bundle.pt"
    bundle = dict(spec=spec, model=stage["model"], normalization=stage["normalization"], contract=selected_contract,
                  selected_epoch=selected_epoch, checkpoint_selection_split="V_select", control_fit_end=train_end,
                  normalization_fit_end=normalization_end, control_summary=summary,
                  initial_model_state_sha=initial_sha, model_state_sha=state_sha)
    fit_base._torch_new(artifact, bundle)
    fit_base._json_new(public / "control" / "RECEIPT.json", summary)
    with (public / "control" / "CURVE.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "exposures", "updates", "V_mse", "segment_seconds", "clipped_steps"])
        writer.writeheader(); writer.writerows(curves)
    values = [row["V_mse"] for row in curves if row["V_mse"] is not None]
    training_status = "LIMITED_SCREEN" if max(epochs) < 3 else ("UNDERTRAINED_OR_BUDGET_LIMIT" if selected_epoch == max(epochs) else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED")
    receipt = dict(status="COMPLETE_VALID", run_id=run_id, family=spec["family"], role="AplusM_control",
                   training_status=training_status, epochs=epochs, selected_epoch=selected_epoch,
                   selected_V_mse=summary["selected_V_mse"], checkpoint_selection_split="V_select",
                   control_fit_end=train_end, normalization_fit_end=normalization_end,
                   parent_fit_end=train_end, fresh_initialization=True, initial_model_state_sha=initial_sha,
                   pretrained_G_or_adapter_parent_reused=False, canonical_scaler="A_fit_plus_A_stop",
                   optimizer="AdamW", weight_decay=0., updates=summary["updates"], exposures=summary["exposures"],
                   actual_epochs=summary["actual_epochs"], train_origins=summary["train_origins"],
                   unique_origins=summary["unique_origins"], unique_target_times=summary["unique_target_times"],
                   observed_target_exposures=summary["observed_target_exposures"],
                   F_logical_streams=summary["F_total_logical_streams"], clipping_fraction=summary["clipping_fraction"],
                   last_cp_best=bool(values and values[-1] <= min(values)),
                   last_scores_still_falling=bool(len(values) >= 2 and values[-1] < values[-2]),
                   artifact_cache_relative_path=artifact.relative_to(Path(cache_root)).as_posix(),
                   artifact_sha256=data_contract.sha256_file(artifact), model_state_sha=state_sha,
                   source_sha=source_sha, scaler_sha=selected_contract["scaler_sha"],
                   T_final_accessed=False, M_cal_prediction_parent_eligible=False, reused_complete_artifact=False)
    fit_base._json_new(receipt_path, receipt)
    return receipt
