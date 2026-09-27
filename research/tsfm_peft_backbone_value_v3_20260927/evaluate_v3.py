"""Evaluate selected v3 linear candidate runs against reused prediction arrays.

This script deliberately reads existing baseline prediction files instead of
recreating F0, LoRA, SHARED, FACTOR, RAW, or TSFM predictions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import LinearResidual
from runtime import CACHE, HERE, ROOT, Job, digest, load_data, save_json, score_arrays, tensors


PERIODS = {"electricity": ("dev",), "bull": ("e1", "e2")}
SEEDS = (92601, 92602)
SAFE_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def resolve_existing_input(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return HERE / path


def origin_key(period: str) -> str:
    return f"eval_{period}_origins" if period in {"e1", "e2"} else f"{period}_origins"


def period_arrays(data: dict[str, Any], period: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins = np.asarray(data[origin_key(period)])
    target = np.stack([data["x"][o : o + 48] for o in origins])
    mask = np.stack([data["finite"][o : o + 48] for o in origins]).astype(bool)
    return origins, target, mask


def validate_selected_linear(selection: dict[str, Any]) -> None:
    if set(selection) != set(PERIODS):
        raise ValueError(f"selected_linear.json must contain exactly {sorted(PERIODS)}")
    for dataset, entry in selection.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{dataset} selection must be an object")
        if "lr" not in entry or not isinstance(entry.get("run_ids"), list) or len(entry["run_ids"]) != 2:
            raise ValueError(f"{dataset} selection must be {{'lr': number, 'run_ids': [id, id]}}")
        if not isinstance(entry["lr"], (int, float)):
            raise ValueError(f"{dataset} lr must be numeric")
        if len(set(entry["run_ids"])) != 2:
            raise ValueError(f"{dataset} selected run ids must be distinct")


def validate_name(name: str) -> None:
    if not name or any(char not in SAFE_NAME_CHARS for char in name):
        raise ValueError("name may contain only letters, digits, underscore, and hyphen")


def linear_role_for_arm(arm: str | None) -> str:
    if arm == "residual":
        return "linear_res"
    if arm == "raw_bypass":
        return "linear_raw"
    safe = "".join(char if char in SAFE_NAME_CHARS else "_" for char in str(arm or "unknown"))
    return f"linear_{safe}"


def recorded_metric(receipt: dict[str, Any], metric: str) -> Any:
    if metric in receipt:
        return receipt[metric]
    scores = receipt.get("scores")
    if isinstance(scores, dict):
        return scores.get(metric)
    return None


def load_prediction_receipt(
    receipt: dict[str, Any],
    origins: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    default_role: str | None = None,
) -> dict[str, Any]:
    path = resolve_path(receipt["prediction_path"])
    actual_hash = digest(path)
    if actual_hash != receipt["prediction_sha256"]:
        raise ValueError(f"prediction hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["origins"], origins):
            raise ValueError(f"origin mismatch: {path}")
        prediction = archive["prediction"].copy()
    scores = score_arrays(prediction, target, mask)
    for metric in ("mse", "mae"):
        recorded = recorded_metric(receipt, metric)
        if recorded is not None and not np.isclose(scores[metric], recorded, rtol=0, atol=1e-8):
            raise ValueError(f"recorded {metric} mismatch for {path}")
    role = receipt.get("role", default_role)
    if role is None:
        raise ValueError(f"prediction receipt has no role: {path}")
    return {
        "role": role,
        "arm": receipt["arm"],
        "key": receipt["key"],
        "fit": receipt.get("fit"),
        "seed": receipt.get("seed"),
        "ed_mode": receipt.get("ed_mode"),
        "latent": receipt.get("latent"),
        "prediction_path": str(path),
        "prediction_sha256": actual_hash,
        "scores": scores,
        "prediction": prediction,
    }


def load_existing_prediction(receipt: dict[str, Any], origins: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return load_prediction_receipt(receipt, origins, target, mask)


def load_reference_linear_res_predictions(
    reference: dict[str, Any],
    dataset: str,
    period: str,
    origins: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    if reference.get("status") != "complete":
        raise ValueError("linear_res reference output is not complete")
    rows = reference["datasets"][dataset]["periods"][period]["linear_predictions"]
    loaded = []
    for row in rows:
        if row.get("arm") != "residual":
            raise ValueError(f"linear_res reference contains non-residual arm for {dataset}/{period}: {row.get('arm')}")
        loaded.append(load_prediction_receipt(row, origins, target, mask, default_role="linear_res_reference"))
    return loaded


def restore_linear_model(checkpoint: dict[str, Any], initial_state: dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    spec = checkpoint["spec"]
    model = LinearResidual(spec, initial_state)
    model.restore_adapter(checkpoint["state"])
    if model.model_config() != checkpoint["model_config"]:
        raise ValueError("restored preprocessing/model configuration differs")
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def evaluate_linear_run(
    run_id: str,
    dataset: str,
    expected_lr: float,
    data: dict[str, Any],
    period: str,
    prediction_dir: Path,
    job: Job,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = CACHE / "runs" / run_id / "best.pt"
    initial_path = CACHE / "runs" / run_id / "initial.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not initial_path.exists():
        raise FileNotFoundError(initial_path)
    checkpoint_sha256 = digest(checkpoint_path)
    initial_sha256 = digest(initial_path)
    fit_result = read_json(HERE / "runs" / run_id / "result.json")
    if fit_result["status"] != "complete" or fit_result["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("selected checkpoint does not match completed fit receipt")
    if fit_result["initial_checkpoint_sha256"] != initial_sha256:
        raise ValueError("initial checkpoint does not match completed fit receipt")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    spec = saved["spec"]
    if spec.get("dataset") != dataset:
        raise ValueError(f"{run_id} dataset mismatch: {spec.get('dataset')} != {dataset}")
    if "t_lr" not in spec or not np.isclose(float(spec["t_lr"]), float(expected_lr), rtol=0, atol=1e-15):
        raise ValueError(f"{run_id} t_lr does not match selected_linear lr")
    if initial.get("spec", {}).get("dataset") != dataset:
        raise ValueError(f"{run_id} initial dataset mismatch")
    initial_state = {name: initial["state"][name] for name in LinearResidual.initial_names}
    model = restore_linear_model(saved, initial_state, device)
    origins, target, mask = period_arrays(data, period)
    predictions = []
    for start in range(0, len(origins), 4):
        job.check_limits()
        batch_origins = origins[start : start + 4]
        x, _, _ = tensors(data, batch_origins, device=device)
        predictions.append(model(x).detach().float().cpu().numpy())
        job.heartbeat(f"evaluate {run_id} {period} {start}/{len(origins)}")
    prediction = np.concatenate(predictions, axis=0)
    scores = score_arrays(prediction, target, mask)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{period}_{run_id}.npz"
    if prediction_path.exists():
        raise RuntimeError(f"preserve existing prediction file: {prediction_path}")
    np.savez_compressed(prediction_path, prediction=prediction, origins=origins)
    result_path = HERE / "runs" / run_id / "result.json"
    result_sha256 = digest(result_path) if result_path.exists() else None
    best_val_path = checkpoint_path.with_name("best_val.npz")
    return {
        "role": linear_role_for_arm(spec.get("arm", "residual")),
        "key": run_id,
        "fit": run_id,
        "seed": spec.get("seed"),
        "selection_lr": expected_lr,
        "t_lr": spec.get("t_lr"),
        "g_initial_lr": 0.001,
        "arm": spec.get("arm", "residual"),
        "dataset": spec.get("dataset"),
        "period": period,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": initial_sha256,
        "best_val_prediction_path": str(best_val_path) if best_val_path.exists() else None,
        "best_val_prediction_sha256": digest(best_val_path) if best_val_path.exists() else None,
        "result_path": str(result_path) if result_path.exists() else None,
        "result_sha256": result_sha256,
        "prediction_path": str(prediction_path),
        "prediction_sha256": digest(prediction_path),
        "scores": scores,
        "prediction": prediction,
    }


def without_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "prediction"}


def role_seed_mean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    return {
        "status": "available",
        "count": len(rows),
        "mse": float(np.mean([row["scores"]["mse"] for row in rows])),
        "mae": float(np.mean([row["scores"]["mae"] for row in rows])),
        "seeds": [row.get("seed") for row in rows],
        "keys": [row["key"] for row in rows],
        "scope": "arithmetic mean of per-seed losses; not a prediction ensemble",
    }


def predictions_by_seed(rows: list[dict[str, Any]], label: str) -> dict[int, np.ndarray]:
    by_seed = {row["seed"]: row["prediction"] for row in rows}
    if sorted(by_seed) != list(SEEDS):
        raise ValueError(f"seed mismatch for {label}")
    return by_seed


def single_role(rows: list[dict[str, Any]], label: str) -> str:
    roles = sorted({row["role"] for row in rows})
    if len(roles) != 1:
        raise ValueError(f"{label} must have one role, got {roles}")
    return roles[0]


def channel_macro_mse(error_sum: np.ndarray, count: np.ndarray) -> float:
    valid = count > 0
    if not np.any(valid):
        raise ValueError("no observed targets in selected block")
    return float(np.mean(error_sum[valid] / count[valid]))


def squared_error_terms(predictions: list[np.ndarray], target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    terms = []
    for prediction in predictions:
        error = np.where(mask, prediction.astype(np.float64) - target, 0.0)
        terms.append((error * error).sum(axis=1))
    return np.mean(terms, axis=0)


def paired_delta(
    linear_predictions: list[np.ndarray],
    reference_predictions: list[np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
    block: int = 7,
    draws: int = 2000,
) -> dict[str, Any]:
    if len(linear_predictions) != len(reference_predictions):
        raise ValueError("LINEAR and reference prediction lists must use the same seed count")
    linear_terms = squared_error_terms(linear_predictions, target, mask)
    reference_terms = squared_error_terms(reference_predictions, target, mask)
    counts = mask.sum(axis=1)

    def values(index: np.ndarray) -> dict[str, float]:
        count = counts[index].sum(axis=0)
        linear_mse = channel_macro_mse(linear_terms[index].sum(axis=0), count)
        reference_mse = channel_macro_mse(reference_terms[index].sum(axis=0), count)
        delta = linear_mse - reference_mse
        relative = float(100.0 * delta / reference_mse) if reference_mse > 0 else None
        return {
            "linear_mse": linear_mse,
            "reference_mse": reference_mse,
            "delta_mse_linear_minus_reference": delta,
            "relative_to_reference_pct": relative,
        }

    n = len(target)
    if not 1 <= block <= n:
        raise ValueError("block length must fit the evaluation period")
    rng = np.random.default_rng(9262026)
    sampled = []
    for _ in range(draws):
        starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
        index = np.concatenate([start + np.arange(block) for start in starts])[:n]
        sampled.append(values(index))
    full = values(np.arange(n))
    first = values(np.arange(n // 2))
    second = values(np.arange(n // 2, n))
    seed_deltas = []
    for linear, reference in zip(linear_predictions, reference_predictions):
        linear_score = score_arrays(linear, target, mask)["mse"]
        reference_score = score_arrays(reference, target, mask)["mse"]
        seed_deltas.append(
            {
                "linear_mse": linear_score,
                "reference_mse": reference_score,
                "delta_mse_linear_minus_reference": linear_score - reference_score,
            }
        )
    deltas = np.array([row["delta_mse_linear_minus_reference"] for row in sampled], dtype=np.float64)
    relatives = np.array([row["relative_to_reference_pct"] for row in sampled], dtype=np.float64)
    return {
        "status": "available",
        "full": full,
        "temporal_halves": [first, second],
        "paired_seed_deltas": seed_deltas,
        "conditional_block95_delta_mse": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "conditional_block95_relative_to_reference_pct": np.quantile(relatives, [0.025, 0.975]).tolist(),
        "block_origins": block,
        "draws": draws,
        "scope": "non-circular 7-origin time blocks, all channels move together, trained seeds fixed; exposed development interval, not independent confirmation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="final_development")
    parser.add_argument("--selected-linear", type=Path, default=HERE / "selected_linear.json")
    parser.add_argument("--reuse-manifest", type=Path, default=HERE / "reuse_manifest.json")
    parser.add_argument("--linear-res-reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--comparison-role",
        choices=("selected_tsfm_residual", "selected_raw", "same_variant_raw_control"),
        default="selected_tsfm_residual",
    )
    parser.add_argument("--reserve-seconds", type=float, default=900.0)
    args = parser.parse_args()

    validate_name(args.name)
    output_path = args.output if args.output is not None else HERE / f"{args.name}.json"
    prediction_dir = CACHE / f"{args.name}_predictions"
    if output_path.exists():
        raise RuntimeError(f"preserve existing output: {output_path}")
    manifest = read_json(args.reuse_manifest)
    selected_linear = read_json(args.selected_linear)
    validate_selected_linear(selected_linear)
    linear_res_reference_path = resolve_existing_input(args.linear_res_reference) if args.linear_res_reference is not None else None
    linear_res_reference = read_json(linear_res_reference_path) if linear_res_reference_path is not None else None

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")

    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "name": args.name,
        "started_utc": time.time(),
        "scope": "v3 exposed development evaluation; existing v2 arrays reused, new predictions only for selected linear T_phi candidate",
        "comparison_role": args.comparison_role,
        "reuse_manifest": {"path": str(args.reuse_manifest), "sha256": digest(args.reuse_manifest)},
        "selected_linear": {"path": str(args.selected_linear), "sha256": digest(args.selected_linear), "selection": selected_linear},
        "linear_res_reference": (
            {"path": str(linear_res_reference_path), "sha256": digest(linear_res_reference_path)}
            if linear_res_reference_path is not None
            else None
        ),
        "datasets": {},
    }

    with Job(f"{args.name}_evaluation", category="gpu", reserve_seconds=args.reserve_seconds) as job:
        data_by_dataset = {dataset: load_data(dataset) for dataset in PERIODS}
        linear_rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
        for dataset, entry in selected_linear.items():
            linear_rows_by_dataset[dataset] = []
            data = data_by_dataset[dataset]
            for run_id in entry["run_ids"]:
                for period in PERIODS[dataset]:
                    linear_rows_by_dataset[dataset].append(
                        evaluate_linear_run(run_id, dataset, float(entry["lr"]), data, period, prediction_dir, job, device)
                    )

        for dataset, periods in PERIODS.items():
            output["datasets"][dataset] = {"periods": {}}
            data = data_by_dataset[dataset]
            for period in periods:
                origins, target, mask = period_arrays(data, period)
                existing = [
                    load_existing_prediction(receipt, origins, target, mask)
                    for receipt in manifest["datasets"][dataset]["periods"][period]["reused_predictions"]
                ]
                linear = [row for row in linear_rows_by_dataset[dataset] if row["period"] == period]
                by_role: dict[str, list[dict[str, Any]]] = {}
                for row in existing:
                    by_role.setdefault(row["role"], []).append(row)
                by_role["linear_candidate"] = linear
                candidate_role = single_role(linear, f"{dataset}/{period} candidate")
                by_role[candidate_role] = linear
                seed_means = {role: role_seed_mean(rows) for role, rows in sorted(by_role.items())}
                linear_by_seed = predictions_by_seed(linear, f"{dataset}/{period} candidate")
                reference_by_seed = predictions_by_seed(by_role[args.comparison_role], f"{dataset}/{period} {args.comparison_role}")
                comparison = paired_delta(
                    [linear_by_seed[seed] for seed in SEEDS],
                    [reference_by_seed[seed] for seed in SEEDS],
                    target,
                    mask,
                )
                period_output = {
                    "origins": origins.tolist(),
                    "target_shape": list(target.shape),
                    "reused_predictions": [without_prediction(row) for row in existing],
                    "linear_predictions": [without_prediction(row) for row in linear],
                    "linear_candidate_role": candidate_role,
                    "seed_mean_scores": seed_means,
                    "comparison_linear_vs_reference": comparison,
                }
                if linear_res_reference is not None:
                    linear_res_rows = load_reference_linear_res_predictions(linear_res_reference, dataset, period, origins, target, mask)
                    linear_res_by_seed = predictions_by_seed(linear_res_rows, f"{dataset}/{period} linear_res_reference")
                    period_output["linear_res_reference_predictions"] = [without_prediction(row) for row in linear_res_rows]
                    period_output["comparison_linear_vs_linear_res"] = paired_delta(
                        [linear_by_seed[seed] for seed in SEEDS],
                        [linear_res_by_seed[seed] for seed in SEEDS],
                        target,
                        mask,
                    )
                output["datasets"][dataset]["periods"][period] = period_output
        output["status"] = "complete"
        output["ended_utc"] = time.time()
        save_json(output_path, output)


if __name__ == "__main__":
    main()
