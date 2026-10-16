"""Past-only frozen parent forecasts with separately sealed target contracts."""
import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

import data_contract
import evaluation
import fit_base
import training


KEYS = ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")
STAT_FIELDS = ("run_id", "split", "cohort", "origin", "channel", "horizon_start", "horizon_end", "sse", "sae", "count")


def _json_new_or_same(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Existing prediction contract differs; preserve it and use a new run ID")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _selection_gate(public_root, run_id, artifact_sha):
    path = Path(public_root) / "SELECTION_LOCK.json"
    if not path.is_file():
        raise ValueError("T_final is locked: SELECTION_LOCK.json absent")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("locked") is not True or run_id not in lock.get("allowed_run_ids", []):
        raise ValueError("T_final is locked for this run")
    if lock.get("allowed_artifact_sha", {}).get(run_id) != artifact_sha:
        raise ValueError("T_final artifact SHA differs from locked selection")


def _load_parent(run_id, task, cache_root, public_root, receipt, fold):
    if receipt is None:
        folder = Path(public_root) / "runs" / run_id
        receipt_path = next((folder / name for name in ("BASE_FIT_RECEIPT.json", "CONTROL_FIT_RECEIPT.json", "RIDGE_FIT_RECEIPT.json")
                             if (folder / name).is_file()), None)
        if receipt_path is None:
            raise ValueError("No complete parent fit receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != run_id:
        raise ValueError("Prediction requires this run's complete valid parent receipt")
    root = Path(cache_root).resolve()
    relative = Path(receipt["artifact_cache_relative_path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Parent artifact outside private root or absent")
    if data_contract.sha256_file(path) != receipt["artifact_sha256"]:
        raise ValueError("Parent artifact SHA mismatch")
    if receipt.get("family") == "RAW_RIDGE":
        saved = _load_ridge_parent(path, receipt, task, fold)
    else:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["contract"]["data_sha"] != task["data_sha256"] or saved["spec"].get("fold", "primary") != fold:
        raise ValueError("Parent data or time fold differs from requested cohort")
    if saved["spec"]["run_id"] != run_id:
        raise ValueError("Parent run ID mismatch")
    if fit_base.scaler_fingerprint(saved["normalization"]["mean"], saved["normalization"]["std"]) != saved["contract"]["scaler_sha"]:
        raise ValueError("Parent normalization SHA mismatch")
    return saved, receipt


def _load_ridge_parent(path, receipt, task, fold):
    import ridge_baseline
    artifact = ridge_baseline.load_artifact(path)
    contract = json.loads(str(artifact["contract_json"].item()))
    width, horizon = len(task["columns"]), int(task["H"])
    if receipt.get("fold", "primary") != fold or contract["data_sha"] != task["data_sha256"]:
        raise ValueError("Ridge data or fold contract differs")
    if contract.get("L") != int(task["L"]) or contract.get("H") != horizon:
        raise ValueError("Ridge input/forecast horizon differs")
    if np.asarray(artifact["coefficients"]).shape != (width, ridge_baseline.BLOCKS + 1, horizon):
        raise ValueError("Ridge coefficient channel/feature/horizon axes differ")
    if not np.array_equal(artifact["block_edges"], ridge_baseline.block_edges(task["L"])):
        raise ValueError("Ridge temporal feature coordinates differ")
    mean, std = np.asarray(artifact["mean"], np.float32), np.asarray(artifact["std"], np.float32)
    if mean.shape != (width,) or std.shape != (width,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid ridge normalization")
    if not np.isfinite(artifact["coefficients"]).all():
        raise ValueError("Nonfinite ridge coefficients")
    native_scaler = ridge_baseline.scaler_fingerprint(mean, std)
    if native_scaler != str(artifact["scaler_sha"].item()) or native_scaler != contract["scaler_sha"] or native_scaler != receipt["scaler_sha"]:
        raise ValueError("Ridge native normalization fingerprint mismatch")
    for key in ("data_sha", "source_sha", "feature_sha"):
        if str(artifact[key].item()) != str(contract[key]):
            raise ValueError("Ridge artifact identity mismatch: " + key)
    for key in ("data_sha", "source_sha", "config_sha"):
        if key in receipt and receipt[key] != contract[key]:
            raise ValueError("Ridge receipt identity mismatch: " + key)
    normalized_end = int(artifact["canonical_A60_end"].item())
    train_end = int(artifact["actual_fit_end"].item())
    if train_end != int(receipt["fit_end"]) or normalized_end != data_contract.split_ranges(task, fold)["A_stop"][1]:
        raise ValueError("Ridge train/normalization boundary mismatch")
    normalized_contract = dict(contract, scaler_sha=fit_base.scaler_fingerprint(mean, std), model_sha="NO_FOUNDATION")
    spec = dict(run_id=receipt["run_id"], family="RAW_RIDGE", fold=fold, role=receipt.get("role", "base_parent"),
                device="cpu", microbatch=4, ridge_config=receipt["config"])
    state_sha = training.fingerprint(dict(coefficients=evaluation.array_fingerprint(artifact["coefficients"]),
                                          feature_sha=contract["feature_sha"]))
    return dict(spec=spec, normalization=dict(mean=mean, std=std, fit_end=normalized_end, train_end=train_end),
                contract=normalized_contract, model_state_sha=state_sha, ridge_artifact=artifact,
                ridge_native_scaler_sha=native_scaler)


def _stats_file(path, rows):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=STAT_FIELDS)
                writer.writeheader(); writer.writerows(rows)


def predict_parent(run_id, task, split, cache_root, public_root, foundation, *, dense=False,
                   receipt=None, baseline=None, fold="primary", role="base_parent", deadline=None):
    """Return the five cache arrays, metadata, manifest/reference, score and row ledger.

    Dense M_cal returns no public score/statistics. T targets are not read before
    the selected run and exact artifact hash pass the public selection lock.
    """
    run_id = str(run_id)
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Safe single-component run ID required")
    if split not in {"M_cal", "V_select", "T_final"} or (dense and split != "M_cal"):
        raise ValueError("Registered split required; dense origins are M_cal-only")
    if role not in {"base_parent", "AplusM_control"} or (role == "AplusM_control" and split == "M_cal"):
        raise ValueError("AplusM controls are V/T-only; explicit registered role required")
    baseline_spec = dict(baseline) if isinstance(baseline, dict) else {"kind": baseline}
    kind = baseline_spec.get("kind")
    if kind is not None:
        kind = str(kind).upper()
        if kind not in {"RAW", "CURRENT", "SEASONAL", "PCA", "RAW_RIDGE"}:
            raise ValueError("Unknown fixed baseline")
    requested_ridge = kind == "RAW_RIDGE"
    if requested_ridge:
        kind = None
    bounds = data_contract.split_ranges(task, fold)
    canonical_end = bounds["A_stop"][1]
    canonical_mean, canonical_std, _ = data_contract.fit_statistics(task, canonical_end)
    scaler_sha = fit_base.scaler_fingerprint(canonical_mean, canonical_std)
    saved = None
    if kind is None:
        saved, receipt = _load_parent(run_id, task, cache_root, public_root, receipt, fold)
        spec = copy.deepcopy(saved["spec"])
        own = saved["normalization"]
        parent_fit_end = int(own.get("train_end", own["fit_end"]))
        family = spec["family"]
        if requested_ridge and family != "RAW_RIDGE":
            raise ValueError("RAW_RIDGE request requires a ridge artifact")
        artifact_sha = receipt["artifact_sha256"]
        foundation_sha = saved["contract"]["model_sha"]
        if family not in fit_base.DIRECT | {"RAW_RIDGE"} and fit_base._model_sha(foundation) != foundation_sha:
            raise ValueError("Parent requires its exact original frozen foundation")
        model_sha = training.fingerprint(dict(artifact_sha=artifact_sha, model_state_sha=saved["model_state_sha"],
                                             normalization=saved["contract"]["scaler_sha"]))
        training_source_sha = saved["contract"]["source_sha"]
    else:
        parent_fit_end = canonical_end
        own = dict(mean=canonical_mean, std=canonical_std)
        family = kind
        spec = dict(baseline_spec, kind=kind, fold=fold, L=task["L"], H=task["H"])
        foundation_sha = fit_base._model_sha(foundation) if kind in {"RAW", "PCA"} else "NO_FOUNDATION"
        if kind in {"RAW", "PCA"} and foundation is None:
            raise ValueError("RAW/PCA baseline requires the caller's frozen foundation")
        model_sha = training.fingerprint(dict(baseline=spec, foundation_sha=foundation_sha, scaler_sha=scaler_sha,
                                             data_sha=task["data_sha256"]))
        artifact_sha = model_sha
        training_source_sha = "NO_TRAINING_FIXED_BASELINE"
    if role == "AplusM_control" and (saved is None or saved["spec"].get("role") != role):
        raise ValueError("AplusM role requires an explicitly registered fitted control artifact")
    if role == "base_parent" and saved is not None and saved["spec"].get("role", "base_parent") != role:
        raise ValueError("AplusM artifact cannot be relabeled as a calibration parent")
    allowed_train_end = bounds["M_cal"][1] if role == "AplusM_control" else bounds["M_cal"][0]
    if parent_fit_end > allowed_train_end or parent_fit_end < 1:
        raise ValueError("Parent was fitted beyond its role's permitted training prefix")
    if split == "T_final":
        _selection_gate(public_root, run_id, artifact_sha)
    origins = data_contract.origins(bounds[split], task["L"], task["H"], 1 if dense else task["H"])
    if not len(origins):
        raise ValueError("No registered origins for prediction cohort")
    channel_ids = np.asarray(task["columns"], dtype=str)
    k = int(spec.get("latent_k", len(channel_ids)))
    if kind == "PCA":
        k = int(baseline_spec.get("latent_k", max(1, len(channel_ids) // 4)))
    used_f = kind in {"RAW", "PCA"} or (saved is not None and family not in fit_base.DIRECT | {"RAW_RIDGE"})
    streams_per_origin = (len(channel_ids) if family in {"RAW", "LORA"} else k) if used_f else 0
    expected_streams = len(origins) * streams_per_origin
    source_files = {name: data_contract.sha256_file(Path(__file__).with_name(name)) for name in
                    ("prediction_cache.py", "fit_base.py", "data_contract.py", "training.py", "evaluation.py", "networks.py", "gpu_runtime.py")}
    if family == "RAW_RIDGE":
        source_files["ridge_baseline.py"] = data_contract.sha256_file(Path(__file__).with_name("ridge_baseline.py"))
    source_sha = training.fingerprint(dict(deployment_sources=source_files, training_source_sha=training_source_sha))
    cohort = f"{task.get('name', task.get('dataset', 'task'))}:{fold}:{split}:stride{1 if dense else task['H']}"
    identity = dict(run_id=run_id, split=split, cohort=cohort, source_sha=source_sha,
                    config_sha=training.fingerprint(dict(parent_spec=spec, dense=bool(dense), fold=fold, role=role)),
                    data_sha=task["data_sha256"], model_sha=model_sha, scaler_sha=scaler_sha)
    # Build this independently from Task truth BEFORE inference or NPZ creation.
    targets, masks = zip(*(data_contract.target_window(task, int(origin), canonical_mean, canonical_std) for origin in origins))
    target = np.stack(targets).astype(np.float32)
    observed = np.stack(masks).astype(bool)
    reference = dict(identity, expected_F_streams=expected_streams, arrays={name: evaluation.array_fingerprint(value) for name, value in
                     dict(target_normalized=target, observed=observed, origins=origins, channel_ids=channel_ids).items()})
    suffix = split + ("_dense" if dense else "")
    private = Path(cache_root) / "predictions" / run_id
    public = Path(public_root) / "predictions" / run_id / suffix
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    cache_path = private / f"{suffix}.npz"
    reference_path, manifest_path = public / "REFERENCE_CONTRACT.json", public / "PREDICTION_MANIFEST.json"
    _json_new_or_same(reference_path, reference)
    if cache_path.exists():
        if not manifest_path.is_file():
            raise ValueError("Existing private cache has no completed public manifest; preserve it")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(name) != value for name, value in identity.items()) or data_contract.sha256_file(cache_path) != manifest["archive_sha256"]:
            raise ValueError("Existing prediction source/parent/cache contract changed")
        with np.load(cache_path, allow_pickle=False) as archive:
            if set(archive.files) != set(KEYS):
                raise ValueError("Prediction cache keys changed")
            arrays = {name: archive[name] for name in KEYS}
        for name, value in arrays.items():
            if evaluation.array_fingerprint(value) != manifest["arrays"][name]:
                raise ValueError("Cached array SHA differs")
            if name != "pred_normalized" and evaluation.array_fingerprint(value) != reference["arrays"][name]:
                raise ValueError("Cache differs from independently reconstructed cohort")
        metadata = dict(manifest["metadata"], reused_prediction_cache=True)
        return dict(arrays, metadata=metadata, manifest=manifest, reference_contract=reference,
                    score=manifest["score"], cost_record=manifest["cost_record"],
                    public_stats_relative_path=manifest.get("public_stats_relative_path"))
    device = spec.get("device", getattr(foundation, "device", "cpu")) if kind not in {"CURRENT", "SEASONAL"} else "cpu"
    prefix_end = int(origins[-1]) + task["H"]
    prefix = dict(task, physical=task["physical"][:prefix_end], observed=task["observed"][:prefix_end])
    windows = training.FastWindows(prefix, own["mean"], own["std"], device)
    model = None
    before = int(getattr(foundation, "streams", 0))
    predictions = []
    try:
        if kind == "PCA":
            pca = data_contract.initialize_pca(task, canonical_end, k, canonical_mean, canonical_std)
            model = fit_base._FixedPCA(pca["pca_center"], pca["E"]).to(device)
        elif saved is not None and family != "RAW_RIDGE":
            model, rebuilt = fit_base.create_model(spec, task, int(own["fit_end"]), foundation)
            model.load_state_dict(saved["model"], strict=True)
            if fit_base.scaler_fingerprint(rebuilt["mean"], rebuilt["std"]) != fit_base.scaler_fingerprint(own["mean"], own["std"]):
                raise ValueError("Loaded parent normalization differs from its fit prefix")
        if model is not None:
            model.eval()
        with torch.no_grad():
            for lo in range(0, len(origins), int(spec.get("microbatch", 4))):
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("Prediction deadline reached before forecast microbatch")
                x, _, _ = windows.get(origins[lo:lo + int(spec.get("microbatch", 4))])
                if kind == "CURRENT":
                    prediction = x[:, -1:].expand(-1, task["H"], -1)
                elif kind == "SEASONAL":
                    period = int(baseline_spec.get("period", task.get("period", 0)))
                    if not 1 <= period <= task["L"]:
                        raise ValueError("Seasonal baseline period must fit supplied history")
                    indices = task["L"] - period + torch.arange(task["H"], device=device) % period
                    prediction = x[:, indices]
                elif kind == "RAW":
                    prediction = foundation(x)
                elif kind == "PCA":
                    prediction = model(x, foundation)
                elif family == "RAW_RIDGE":
                    import ridge_baseline
                    prediction = torch.from_numpy(ridge_baseline.predict_normalized(saved["ridge_artifact"], x.cpu().numpy()))
                else:
                    prediction = fit_base.predict_model(model, x, foundation, family, None)
                if tuple(prediction.shape) != (len(x), task["H"], len(channel_ids)) or not torch.isfinite(prediction).all():
                    raise ValueError("Invalid original-channel prediction")
                values = prediction.cpu().numpy()
                if np.array_equal(own["mean"], canonical_mean) and np.array_equal(own["std"], canonical_std):
                    predictions.append(values.astype(np.float32))
                else:
                    physical = values.astype(np.float64) * np.asarray(own["std"], np.float64) + np.asarray(own["mean"], np.float64)
                    predictions.append(((physical - canonical_mean.astype(np.float64)) / canonical_std.astype(np.float64)).astype(np.float32))
    finally:
        if model is not None and callable(getattr(model, "close", None)):
            model.close()
        if foundation is not None and callable(getattr(foundation, "assert_frozen_unchanged", None)):
            foundation.assert_frozen_unchanged()
    actual_streams = int(getattr(foundation, "streams", 0)) - before
    if used_f and not hasattr(foundation, "streams"):
        raise ValueError("Foundation requires an actual logical-stream counter")
    if actual_streams != expected_streams:
        raise ValueError("Actual F streams differ from requested original/latent width")
    arrays = dict(pred_normalized=np.concatenate(predictions), target_normalized=target,
                  observed=observed, origins=origins, channel_ids=channel_ids)
    with cache_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    metadata = dict(identity, data_sha256=task["data_sha256"], parent_fit_end=parent_fit_end,
                    artifact_sha256=artifact_sha, family=family, fold=fold, dense=bool(dense), role=role,
                    canonical_scaler_fit_end=canonical_end, parent_normalization_fit_end=int(own.get("fit_end", canonical_end)),
                    reused_prediction_cache=False, source_files=source_files,
                    training_source_sha=training_source_sha, raw_target_publication=False)
    if family == "RAW_RIDGE":
        metadata["ridge_native_scaler_sha"] = saved["ridge_native_scaler_sha"]
    cost_record = dict(run_id=run_id, batch=len(origins), latent_k=streams_per_origin,
                       actual_F_streams=actual_streams, expected_F_streams=expected_streams, F_calls=1)
    score, stats_relative = None, None
    if not dense:
        score, statistics = evaluation.error_statistics(arrays["pred_normalized"], target, observed, origins,
                                                        run_id, split, cohort, channel_ids)
        stats_path = public / "PUBLIC_ERROR_STATS.csv.gz"
        _stats_file(stats_path, statistics)
        stats_relative = stats_path.relative_to(Path(public_root)).as_posix()
        with (public / "SCORES.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(score)); writer.writeheader(); writer.writerow(score)
    manifest = dict(identity, cache_relative_path=cache_path.relative_to(Path(cache_root)).as_posix(),
                    archive_sha256=data_contract.sha256_file(cache_path),
                    arrays={name: evaluation.array_fingerprint(value) for name, value in arrays.items()},
                    metadata=metadata, score=score, cost_record=cost_record, public_stats_relative_path=stats_relative)
    _json_new_or_same(manifest_path, manifest)
    return dict(arrays, metadata=metadata, manifest=manifest, reference_contract=reference,
                score=score, cost_record=cost_record, public_stats_relative_path=stats_relative)
