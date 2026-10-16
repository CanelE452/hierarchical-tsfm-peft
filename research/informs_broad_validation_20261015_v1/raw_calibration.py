"""M-only per-channel bias correction of frozen Raw median forecasts."""
import json
from pathlib import Path
import time

import numpy as np

import combination
import data_contract
import evaluation
import training

PREDICTION_KEYS = ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")

def fit_bias(prediction, target, observed):
    p, y, mask = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(observed)
    if p.shape != y.shape or mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError("Matching observed forecast tensors required")
    count = mask.sum((0, 1))
    if np.any(count == 0):
        raise ValueError("Every calibrated channel requires an observed M target")
    if not np.isfinite(p[mask]).all() or not np.isfinite(y[mask]).all():
        raise ValueError("Nonfinite observed calibration value")
    bias = np.where(mask, y - p, 0).sum((0, 1), dtype=np.float64) / count
    return bias


def _contract(task, raw_m, mean, std):
    source = {name: data_contract.sha256_file(Path(__file__).with_name(name)) for name in
              ("raw_calibration.py", "combination.py", "data_contract.py", "evaluation.py", "training.py")}
    return dict(data_sha=task["data_sha256"], scaler_sha=combination.scaler_fingerprint(mean, std),
                    source_sha=training.fingerprint(source),
                    parent_model_sha=raw_m["metadata"]["model_sha"],
                    config_sha=training.fingerprint(dict(variant="per_channel_constant_bias", fit_split="M_cal", penalty=0)))


def validate_completed_receipt(receipt, run_id, task, raw_m, raw_v, mean, std, cache_root):
    combination.validate_parent_caches(task, raw_m, raw_v, raw_m, raw_v, mean, std)
    if receipt.get("run_id") != run_id or receipt.get("status") != "COMPLETE_VALID" or receipt.get("family") != "RAW_CALIBRATED":
        raise ValueError("Only this complete Raw calibration can be reused")
    if receipt.get("contract") != _contract(task, raw_m, mean, std):
        raise ValueError("Raw calibration source/config/data/scaler/parent model changed")
    parents = {"M_cal": combination.cache_fingerprint(raw_m), "V_select": combination.cache_fingerprint(raw_v)}
    if receipt.get("parent_cache_fingerprints") != parents:
        raise ValueError("Raw calibration parent prediction cache changed")
    private = Path(cache_root) / "raw_calibration" / run_id
    artifact = private / "bias.npz"
    if receipt.get("artifact_cache_relative_path") != artifact.relative_to(Path(cache_root)).as_posix() or data_contract.sha256_file(artifact) != receipt.get("artifact_sha256"):
        raise ValueError("Raw calibration bias artifact changed")
    with np.load(artifact, allow_pickle=False) as archive:
        if set(archive.files) != {"bias", "mean", "std", "contract_json"}:
            raise ValueError("Raw calibration bias artifact schema changed")
        bias = archive["bias"]
        if bias.shape != (len(task["columns"]),) or not np.isfinite(bias).all():
            raise ValueError("Raw calibration bias axes/values changed")
        if not np.array_equal(archive["mean"], mean) or not np.array_equal(archive["std"], std) or json.loads(str(archive["contract_json"].item())) != receipt["contract"]:
            raise ValueError("Raw calibration artifact contract changed")
    if set(receipt.get("prediction_artifacts", {})) != {"M_cal", "V_select"}:
        raise ValueError("Raw calibration prediction splits are incomplete")
    for split, parent in (("M_cal", raw_m), ("V_select", raw_v)):
        declared = receipt["prediction_artifacts"][split]
        path = private / f"{split}.npz"
        if declared.get("file") != path.name or data_contract.sha256_file(path) != declared.get("sha256"):
            raise ValueError("Raw calibration prediction output changed")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(PREDICTION_KEYS):
                raise ValueError("Raw calibration output array schema changed")
            expected = dict(parent, pred_normalized=(np.asarray(parent["pred_normalized"], np.float64) + bias).astype(np.float32))
            for key in PREDICTION_KEYS:
                if not np.array_equal(archive[key], expected[key]):
                    raise ValueError("Raw calibration output no longer matches frozen parent/bias: " + key)
    return receipt


def fit_raw_calibration(run_id, task, raw_m, raw_v, mean, std, cache_root, public_root):
    """Closed-form future-MSE correction; V and T never fit parameters."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Safe single-component calibration ID required")
    combination.validate_parent_caches(task, raw_m, raw_v, raw_m, raw_v, mean, std)
    private = Path(cache_root) / "raw_calibration" / run_id
    public = Path(public_root) / "raw_calibration" / run_id
    receipt_path = public / "receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        return validate_completed_receipt(receipt, run_id, task, raw_m, raw_v, mean, std, cache_root)
    started = time.time()
    private.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=False)
    contract = _contract(task, raw_m, mean, std)
    bias = fit_bias(raw_m["pred_normalized"], raw_m["target_normalized"], raw_m["observed"])
    artifact = private / "bias.npz"
    with artifact.open("xb") as handle:
        np.savez_compressed(handle, bias=bias, mean=mean, std=std,
                            contract_json=np.asarray(json.dumps(contract, sort_keys=True)))
    outputs, scores = {}, []
    for split, parent in (("M_cal", raw_m), ("V_select", raw_v)):
        pred = (np.asarray(parent["pred_normalized"], np.float64) + bias).astype(np.float32)
        outputs[split] = combination._private_forecast(private / f"{split}.npz", pred, parent)
        if split == "V_select":
            score = combination._public_stats(public / "PUBLIC_ERROR_STATS.csv.gz", pred, parent, run_id, split)
        else:
            score = dict(run_id=run_id, split=split, **combination.macro_metrics(pred, parent["target_normalized"], parent["observed"]))
        scores.append(score)
    receipt = dict(run_id=run_id, dataset=task.get("name"), status="COMPLETE_VALID", family="RAW_CALIBRATED",
                   training_status="CLOSED_FORM_M_CAL_ONLY", contract=contract, parameter_count=len(bias),
                   parent_cache_fingerprints={"M_cal": combination.cache_fingerprint(raw_m), "V_select": combination.cache_fingerprint(raw_v)},
                   started_unix=started, completed_unix=time.time(),
                   registered_before_V_analysis=True, parent_run_id=raw_m["metadata"]["run_id"],
                   fit_split="M_cal_dense_stride1", checkpoint_selection="none", T_final_accessed=False,
                   artifact_cache_relative_path=artifact.relative_to(Path(cache_root)).as_posix(),
                   artifact_sha256=data_contract.sha256_file(artifact), prediction_artifacts=outputs, scores=scores,
                   foundation_calls_during_bias_fit=0, deployment_requires_raw_foundation=True,
                   limitation="Only an M-fitted constant bias per channel; no claim of best possible Raw calibration")
    with (public / "receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
    return receipt
