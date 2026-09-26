from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / "tsfm_peft_development_20260926"
OUT = HERE / "saved_validation_diagnostics.json"
DATASET = "electricity"
CONTEXT = 512
HORIZON = 48
RANK = 8
DECOMPOSITION_TOL = 1e-10
GLOBAL_CROSS_TOL = 1e-7


def digest(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pca_basis(train: np.ndarray, rank: int) -> np.ndarray:
    centered = train - train.mean(axis=0, keepdims=True)
    _, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    return vectors[:, -rank:][:, ::-1].copy().astype(np.float32)


def load_electricity() -> dict[str, Any]:
    contract = json.loads((HERE / "data_contract.json").read_text(encoding="utf-8"))
    path = ROOT / contract["datasets"]["electricity_first32"]["npz"]
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    data["_path"] = path
    data["finite"] = data["targetmask"]
    data["train_start"], data["train_end"] = data["split_bounds"][:2]
    return data


def source_receipt() -> dict[str, str]:
    return {
        p.name: digest(p)
        for p in [
            HERE / "diagnose_saved_validation.py",
            HERE / "model.py",
            HERE / "linear_fulltrain_baselines.json",
            HERE / "data_contract.json",
        ]
        if p.exists()
    }


def project_common(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (values @ basis) @ basis.T


def require_finite_array(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name} contains non-finite values")


def require_prediction_shape(name: str, pred: np.ndarray, target: np.ndarray) -> None:
    if pred.shape != target.shape:
        raise RuntimeError(f"{name} prediction shape {pred.shape} != target shape {target.shape}")
    if pred.ndim != 3 or pred.shape[1] != HORIZON:
        raise RuntimeError(f"{name} prediction must be [origins,{HORIZON},channels], got {pred.shape}")
    require_finite_array(f"{name} prediction", pred)
    require_finite_array(f"{name} target", target)


def decompose_score(pred: np.ndarray, target: np.ndarray, basis: np.ndarray) -> dict[str, Any]:
    require_prediction_shape("period", pred, target)
    error = pred.astype(np.float64) - target.astype(np.float64)
    common = project_common(error, basis.astype(np.float64))
    residual = error - common
    channel_total = (error**2).mean(axis=(0, 1))
    channel_common = (common**2).mean(axis=(0, 1))
    channel_residual = (residual**2).mean(axis=(0, 1))
    cross = 2 * (common * residual).mean(axis=(0, 1))
    return {
        "mse": float(channel_total.mean()),
        "common_mse": float(channel_common.mean()),
        "residual_mse": float(channel_residual.mean()),
        "cross_term": float(cross.mean()),
        "decomposition_abs_error": float(
            abs(channel_total.mean() - (channel_common.mean() + channel_residual.mean() + cross.mean()))
        ),
        "max_abs_channel_cross_term": float(np.max(np.abs(cross))),
        "channel_mse": channel_total.tolist(),
        "channel_common_mse": channel_common.tolist(),
        "channel_residual_mse": channel_residual.tolist(),
    }


def validate_decomposition(label: str, score: dict[str, Any]) -> None:
    scalar_keys = ["mse", "common_mse", "residual_mse", "cross_term", "decomposition_abs_error", "max_abs_channel_cross_term"]
    values = np.array([float(score[k]) for k in scalar_keys], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label} has non-finite scalar decomposition values")
    for key in ["channel_mse", "channel_common_mse", "channel_residual_mse"]:
        arr = np.asarray(score[key], dtype=np.float64)
        if arr.shape != (32,) or not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{label} {key} must be finite with shape [32], got {arr.shape}")
    if score["decomposition_abs_error"] >= DECOMPOSITION_TOL:
        raise RuntimeError(f"{label} decomposition error {score['decomposition_abs_error']} >= {DECOMPOSITION_TOL}")
    if abs(score["cross_term"]) >= GLOBAL_CROSS_TOL:
        raise RuntimeError(f"{label} global cross term {score['cross_term']} exceeds {GLOBAL_CROSS_TOL}")


def split_halves(origins: np.ndarray) -> dict[str, np.ndarray]:
    n = len(origins)
    return {"all": np.arange(n), "first_half": np.arange(0, n // 2), "second_half": np.arange(n // 2, n)}


def score_record(name: str, pred: np.ndarray, target: np.ndarray, basis: np.ndarray, origins: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    require_prediction_shape(name, pred, target)
    periods = {}
    for period, idx in split_halves(origins).items():
        periods[period] = decompose_score(pred[idx], target[idx], basis)
        validate_decomposition(f"{name}:{period}", periods[period])
    return {"name": name, **meta, "periods": periods}


def completed_fit_records(data: dict[str, Any], basis: np.ndarray, target: np.ndarray, val_origins: np.ndarray) -> list[dict[str, Any]]:
    records = []
    for result_path in sorted((HERE / "runs").glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        if result.get("dataset") != DATASET:
            continue
        fit_id = result_path.parent.name
        pred_path = CACHE / "runs" / fit_id / "best_val.npz"
        if not pred_path.exists():
            raise FileNotFoundError(f"complete {DATASET} result has no best_val.npz: {fit_id}")
        with np.load(pred_path, allow_pickle=False) as archive:
            pred = archive["prediction"]
            origins = archive["origins"]
        if not np.array_equal(origins, val_origins):
            raise RuntimeError(f"validation origins mismatch for {fit_id}")
        checkpoint = Path(result["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"complete {DATASET} result has no checkpoint: {fit_id}")
        checkpoint_digest = digest(checkpoint)
        recorded_checkpoint_digest = result.get("checkpoint_sha256")
        if recorded_checkpoint_digest and checkpoint_digest != recorded_checkpoint_digest:
            raise RuntimeError(f"checkpoint sha256 mismatch for {fit_id}")
        record = score_record(
            fit_id,
            pred,
            target,
            basis,
            val_origins,
            {
                "kind": "neural_fit",
                "dataset": result["dataset"],
                "arm": result["arm"],
                "seed": result["seed"],
                "lr": result["lr"],
                "selected_epoch": result.get("selected_epoch"),
                "selected_steps": result.get("selected_steps"),
                "best_val_mse": result.get("best_val_mse"),
                "prediction_file": str(pred_path.relative_to(ROOT)),
                "prediction_sha256": digest(pred_path),
                "result_file": str(result_path.relative_to(ROOT)),
                "result_sha256": digest(result_path),
                "checkpoint_file": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
                "checkpoint_sha256": checkpoint_digest,
            },
        )
        diff = abs(float(record["periods"]["all"]["mse"]) - float(result["best_val_mse"]))
        if diff > 1e-10:
            raise RuntimeError(f"{fit_id} best_val_mse mismatch: {diff} > 1e-10")
        record["best_val_mse_abs_diff"] = diff
        records.append(record)
    return records


def linear_selected_records(data: dict[str, Any], basis: np.ndarray, target: np.ndarray, val_origins: np.ndarray) -> list[dict[str, Any]]:
    summary_path = HERE / "linear_fulltrain_baselines.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_data_sha256 = digest(data["_path"])
    if summary.get("data_sha256") != expected_data_sha256:
        raise RuntimeError("linear_fulltrain_baselines.json data_sha256 does not match Electricity NPZ")
    records = []
    for arm, selected in summary["selected"].items():
        penalty = selected["ridge_penalty"]
        pred_path = CACHE / "baselines_fulltrain" / f"{arm}_{penalty:g}_val.npz"
        if not pred_path.exists():
            raise FileNotFoundError(f"selected fulltrain linear prediction is missing: {pred_path}")
        with np.load(pred_path, allow_pickle=False) as archive:
            pred = archive["prediction"]
            origins = archive["origins"]
        if not np.array_equal(origins, val_origins):
            raise RuntimeError(f"validation origins mismatch for {arm}")
        record = score_record(
            f"fulltrain_{arm}",
            pred,
            target,
            basis,
            val_origins,
            {
                "kind": "fulltrain_linear_selected",
                "dataset": DATASET,
                "arm": arm,
                "ridge_penalty": penalty,
                "selection": "minimum validation MSE from linear_fulltrain_baselines.json",
                "prediction_file": str(pred_path.relative_to(ROOT)),
                "prediction_sha256": digest(pred_path),
                "summary_file": str(summary_path.relative_to(ROOT)),
                "summary_sha256": digest(summary_path),
                "linear_data_sha256": summary["data_sha256"],
            },
        )
        selected_mse = float(selected["scores"]["val"]["mse"])
        diff = abs(float(record["periods"]["all"]["mse"]) - selected_mse)
        if diff > 1e-10:
            raise RuntimeError(f"{arm} selected val MSE mismatch: {diff} > 1e-10")
        record["selected_val_mse_abs_diff"] = diff
        records.append(record)
    return records


def adapter_diagnostics() -> list[dict[str, Any]]:
    rows = []
    for result_path in sorted((HERE / "runs").glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        if result.get("dataset") != DATASET:
            continue
        checkpoint = Path(result["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"complete {DATASET} result has no checkpoint: {result_path.parent.name}")
        checkpoint_digest = digest(checkpoint)
        recorded_checkpoint_digest = result.get("checkpoint_sha256")
        if recorded_checkpoint_digest and checkpoint_digest != recorded_checkpoint_digest:
            raise RuntimeError(f"checkpoint sha256 mismatch for {result_path.parent.name}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = saved.get("state", {})
        basis = np.asarray(saved.get("basis"), dtype=np.float64)
        arm = result["arm"]
        row: dict[str, Any] = {
            "fit": result_path.parent.name,
            "dataset": result["dataset"],
            "arm": arm,
            "seed": result["seed"],
            "lr": result["lr"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest,
            "result_file": str(result_path.relative_to(ROOT)),
            "result_sha256": digest(result_path),
        }
        if "encoder.weight" in state and "decoder.weight" in state:
            enc = state["encoder.weight"].detach().cpu().numpy().astype(np.float64)
            dec = state["decoder.weight"].detach().cpu().numpy().astype(np.float64)
            init_enc = basis.T
            init_dec = basis
            ed = dec @ enc
            init_ed = init_dec @ init_enc
            row.update(
                {
                    "encoder_init_max_abs_delta": float(np.max(np.abs(enc - init_enc))),
                    "decoder_init_max_abs_delta": float(np.max(np.abs(dec - init_dec))),
                    "ed_init_max_abs_delta": float(np.max(np.abs(ed - init_ed))),
                    "ed_singular_values": np.linalg.svd(ed, compute_uv=False).tolist(),
                    "ed_rank_gt_1e_6": int((np.linalg.svd(ed, compute_uv=False) > 1e-6).sum()),
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    started = time.perf_counter()
    data = load_electricity()
    train = data["x"][int(data["train_start"]) : int(data["train_end"])]
    basis = pca_basis(train, RANK).astype(np.float64)
    val_origins = data["val_origins"]
    target = np.stack([data["x"][o : o + HORIZON] for o in val_origins])
    if not data["finite"][int(data["split_bounds"][1]) : int(data["split_bounds"][2])].all():
        raise RuntimeError("This diagnostic assumes fully observed Electricity validation targets.")
    records = completed_fit_records(data, basis, target, val_origins)
    records.extend(linear_selected_records(data, basis, target, val_origins))
    payload = {
        "status": "complete",
        "scope": "Development diagnostic on saved Electricity validation predictions only; no new model predictions, no GPU, no Bull access.",
        "interpretation_boundary": "Common/residual error decomposition is descriptive. E/D drift and singular values are not causal evidence.",
        "data_sha256": digest(data["_path"]),
        "source": source_receipt(),
        "validation_origin_count": int(len(val_origins)),
        "basis_rank": RANK,
        "records": records,
        "adapter_diagnostics": adapter_diagnostics(),
        "elapsed_s": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(records),
                "neural_records": sum(r["kind"] == "neural_fit" for r in records),
                "linear_records": sum(r["kind"] == "fulltrain_linear_selected" for r in records),
                "elapsed_s": payload["elapsed_s"],
                "output": str(OUT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
