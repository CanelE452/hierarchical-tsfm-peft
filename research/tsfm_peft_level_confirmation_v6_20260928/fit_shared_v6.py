from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from runtime_v6 import CACHE, HERE, ROOT, Job, digest, load_data, save_json, score_arrays


CONTEXT = 512
HORIZON = 48
CHANNELS = 17
PENALTIES = (0.001, 0.1)
MASKED_LINEAR_PATH = ROOT / "research/tsfm_peft_development_20260926/masked_linear.py"


def load_masked_linear():
    spec = importlib.util.spec_from_file_location("v6_v1_masked_linear", MASKED_LINEAR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {MASKED_LINEAR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["v6_v1_masked_linear"] = module
    spec.loader.exec_module(module)
    return module


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_protocol() -> dict[str, Any]:
    return json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


def assert_protocol(protocol: dict[str, Any]) -> None:
    if tuple(float(p) for p in protocol["ridge_penalties"]) != PENALTIES:
        raise RuntimeError("protocol ridge_penalties changed")
    if int(protocol["expected_origins"]["train"]) != 1945 or int(protocol["expected_origins"]["val"]) != 30:
        raise RuntimeError("protocol TRAIN/VAL origin counts changed")
    if int(protocol["channels"]) != CHANNELS or int(protocol["context"]) != CONTEXT or int(protocol["horizon"]) != HORIZON:
        raise RuntimeError("protocol shape contract changed")


def penalty_id(penalty: float) -> str:
    text = f"{penalty:g}".replace("-", "m").replace(".", "p")
    return f"shared_ridge_p{text}"


def save_npz_new(path: Path, **arrays: Any) -> str:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing cache file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)
    return digest(path)


def save_json_new(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing result file: {path}")
    save_json(path, payload)
    return digest(path)


def coerce_trainval(data: dict[str, Any]) -> dict[str, Any]:
    if "trainval" in data and isinstance(data["trainval"], dict):
        data = data["trainval"]
    required = {"x", "finite", "train_origins", "val_origins", "columns"}
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"load_data(include_test=False) missing required keys: {missing}")
    return data


def windows(data: dict[str, Any], origins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_all = np.asarray(data["x"], dtype=np.float32)
    finite = np.asarray(data["finite"], dtype=bool)
    origins = np.asarray(origins, dtype=np.int64)
    x = np.stack([x_all[o - CONTEXT : o] for o in origins]).astype(np.float32)
    y = np.stack([x_all[o : o + HORIZON] for o in origins]).astype(np.float32)
    mask = np.stack([finite[o : o + HORIZON] for o in origins]).astype(bool)
    return x, y, mask


class TorchSharedLinear:
    """Small FP32 torch restoration helper for cost/evaluation scripts."""

    def __init__(self, weight, bias):
        import torch

        self.weight = torch.as_tensor(weight, dtype=torch.float32)
        self.bias = torch.as_tensor(bias, dtype=torch.float32)

    def to(self, device):
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self

    def __call__(self, x):
        return (x.transpose(1, 2) @ self.weight + self.bias).transpose(1, 2)


def load_torch_shared(weights_path: str | Path):
    with np.load(weights_path, allow_pickle=False) as archive:
        return TorchSharedLinear(archive["weight"], archive["bias"])


def fit_one(penalty: float) -> dict[str, Any]:
    fit_id = penalty_id(penalty)
    weights_path = CACHE / "ridge" / f"{fit_id}.npz"
    val_predictions_path = CACHE / "ridge" / f"{fit_id}_val_predictions.npz"
    result_path = HERE / f"{fit_id}.json"
    if weights_path.exists() or val_predictions_path.exists() or result_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing shared ridge outputs for {fit_id}")

    masked_linear = load_masked_linear()
    data = coerce_trainval(load_data(include_test=False))
    train_origins = np.asarray(data["train_origins"], dtype=np.int64)
    val_origins = np.asarray(data["val_origins"], dtype=np.int64)
    columns = np.asarray(data["columns"]).astype(str)
    if len(train_origins) != 1945 or len(val_origins) != 30:
        raise RuntimeError(f"Unexpected origin counts: train={len(train_origins)} val={len(val_origins)}")
    if len(columns) != CHANNELS:
        raise RuntimeError(f"Expected {CHANNELS} channels, got {len(columns)}")

    train_x, train_y, train_mask = windows(data, train_origins)
    val_x, val_y, val_mask = windows(data, val_origins)
    if train_x.shape != (1945, CONTEXT, CHANNELS) or train_y.shape != (1945, HORIZON, CHANNELS):
        raise RuntimeError(f"Unexpected TRAIN window shapes: {train_x.shape}, {train_y.shape}")

    fit_started = time.perf_counter()
    with threadpool_limits(limits=4):
        fit = masked_linear.fit_masked_shared(train_x, train_y, train_mask, penalty=penalty)
    fit_seconds = time.perf_counter() - fit_started
    stored_fit = {
        "kind": fit["kind"],
        "weight": np.asarray(fit["weight"], dtype=np.float32),
        "bias": np.asarray(fit["bias"], dtype=np.float32),
        "target_counts": np.asarray(fit["target_counts"], dtype=np.int64),
        "penalty": float(penalty),
        "loss": fit["loss"],
    }
    val_prediction = masked_linear.predict_shared(val_x, stored_fit).astype(np.float32)
    val_score = score_arrays(val_prediction, val_y, val_mask)

    weights_sha = save_npz_new(
        weights_path,
        weight=stored_fit["weight"],
        bias=stored_fit["bias"],
        target_counts=stored_fit["target_counts"],
        penalty=np.array(float(penalty), dtype=np.float64),
        columns=columns.astype("U"),
    )
    val_predictions_sha = save_npz_new(
        val_predictions_path,
        prediction=val_prediction,
        origins=val_origins,
        penalty=np.array(float(penalty), dtype=np.float64),
    )

    trainval_path = CACHE / "data" / "trainval.npz"
    result = {
        "schema": "tsfm_peft_level_confirmation_v6_shared_ridge_fit_v1",
        "id": fit_id,
        "family": "shared_linear",
        "dataset": "bdg2_robin_office17",
        "penalty": float(penalty),
        "fit_scope": "TRAIN only, all 1945 origins and all 17 selected channels, masked targets",
        "selection_scope": "VAL only; TEST not loaded by this script",
        "solver": {
            "source": rel(MASKED_LINEAR_PATH),
            "source_sha256": digest(MASKED_LINEAR_PATH),
            "function": "fit_masked_shared",
            "ridge_objective": fit["loss"],
            "horizon_solve_count": HORIZON,
            "penalized_intercept": False,
            "threadpool_limits": 4,
            "fit_seconds": float(fit_seconds),
        },
        "data": {
            "trainval_path": rel(trainval_path),
            "trainval_sha256": digest(trainval_path) if trainval_path.exists() else None,
            "train_origin_count": int(len(train_origins)),
            "val_origin_count": int(len(val_origins)),
            "channels": columns.tolist(),
        },
        "weights": {
            "path": rel(weights_path),
            "sha256": weights_sha,
            "format": "npz with FP32 weight[512,48], FP32 bias[48], int64 target_counts[48]",
            "parameter_count": int(stored_fit["weight"].size + stored_fit["bias"].size),
            "torch_fp32_restore": "Use load_torch_shared(path), or torch: y=(x.transpose(1,2) @ weight + bias).transpose(1,2) for x[B,512,17].",
        },
        "validation_predictions": {
            "path": rel(val_predictions_path),
            "sha256": val_predictions_sha,
            "arrays": {"prediction": list(val_prediction.shape), "origins": list(val_origins.shape)},
        },
        "validation": val_score,
        "target_counts_by_horizon": stored_fit["target_counts"].astype(int).tolist(),
    }
    result_sha = save_json_new(result_path, result)
    result["result_file"] = {"path": rel(result_path), "sha256": result_sha}
    return result


def result_paths() -> list[Path]:
    return [HERE / f"{penalty_id(p)}.json" for p in PENALTIES]


def write_selection_if_complete() -> dict[str, Any] | None:
    paths = result_paths()
    if not all(path.exists() for path in paths):
        return None
    selection_path = HERE / "shared_ridge_selection.json"
    if selection_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing selection: {selection_path}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    order = {float(p): i for i, p in enumerate(PENALTIES)}
    selected = min(results, key=lambda row: (float(row["validation"]["mse"]), order[float(row["penalty"])]))
    payload = {
        "schema": "tsfm_peft_level_confirmation_v6_shared_ridge_selection_v1",
        "selection_metric": "validation.mse",
        "tie_break_penalty_order": [float(p) for p in PENALTIES],
        "test_used_for_selection": False,
        "selected_id": selected["id"],
        "selected_penalty": float(selected["penalty"]),
        "selected_weights": selected["weights"],
        "fits": [
            {
                "id": row["id"],
                "penalty": float(row["penalty"]),
                "val_mse": float(row["validation"]["mse"]),
                "val_mae": float(row["validation"]["mae"]),
                "result_file": {
                    "path": rel(HERE / f"{row['id']}.json"),
                    "sha256": digest(HERE / f"{row['id']}.json"),
                },
                "weights": row["weights"],
                "validation_predictions": row["validation_predictions"],
            }
            for row in results
        ],
    }
    save_json_new(selection_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit approved v6 masked shared ridge baselines on TRAIN only.")
    parser.add_argument(
        "--penalties",
        nargs="+",
        type=float,
        default=list(PENALTIES),
        help="Subset of approved penalties to fit. Default fits both approved penalties.",
    )
    parser.add_argument(
        "--name",
        default="v6_shared_ridge",
        help="Ledger label prefix. Use a new suffix for retries, for example v6_shared_ridge_retry01.",
    )
    parser.add_argument("--cpu-analysis-reserve-seconds", type=int, default=900)
    parser.add_argument("--reserve-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    protocol = read_protocol()
    assert_protocol(protocol)
    args = parse_args()
    penalties = tuple(float(p) for p in args.penalties)
    illegal = sorted(set(penalties) - set(PENALTIES))
    if illegal:
        raise RuntimeError(f"Unapproved shared ridge penalties requested: {illegal}")

    completed = []
    with Job(
        f"{args.name}_cpu_analysis",
        category="cpu_analysis",
        reserve_seconds=args.cpu_analysis_reserve_seconds,
        metadata={"family": "shared_linear", "penalties": [float(p) for p in penalties]},
    ) as analysis_job:
        analysis_job.heartbeat("starting_shared_ridge_fits")
        for penalty in penalties:
            fit_id = penalty_id(penalty)
            analysis_job.heartbeat(f"starting_{fit_id}")
            with Job(
                f"{args.name}_{fit_id}",
                category="real_fit",
                reserve_seconds=args.reserve_seconds,
                metadata={
                    "family": "shared_linear",
                    "penalty": float(penalty),
                    "dataset": "bdg2_robin_office17",
                    "horizon_solve_count": HORIZON,
                    "attempt_accounting": "one real_fit attempt for one penalty, solving 48 horizon-wise ridge systems internally",
                },
            ) as job:
                job.heartbeat("starting_fit")
                result = fit_one(penalty)
                job.heartbeat("fit_and_val_selection_score_complete")
                completed.append({"id": fit_id, "result": result["result_file"]})
            analysis_job.heartbeat(f"completed_{fit_id}")
        selection = write_selection_if_complete()
        analysis_job.heartbeat("complete")
    print(json.dumps({"completed": completed, "selection": selection}, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
