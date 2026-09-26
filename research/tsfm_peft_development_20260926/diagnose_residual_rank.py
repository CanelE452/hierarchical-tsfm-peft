from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / "tsfm_peft_development_20260926"
LOCAL = CACHE / "residual_rank_diagnostic"
OUT = HERE / "residual_rank_diagnostics.json"

CONTEXT = 512
HORIZON = 48
PCA_RANK = 8
CHUNK_ORIGINS = 128
PENALTY = 0.001
RANKS = (8, 16, 32, 48)
STORED_FLOAT32_SCORE_TOL = 1e-9


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(values: np.ndarray) -> str:
    a = np.ascontiguousarray(values.astype("<i8", copy=False))
    return hashlib.sha256(a.tobytes()).hexdigest()


def save_json_new(path: Path, data: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"Do not overwrite existing result: {path}")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_npz_new(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"Do not overwrite existing cache file: {path}")
    np.savez_compressed(path, **arrays)


def source_receipt() -> dict[str, str]:
    paths = [
        HERE / "diagnose_residual_rank.py",
        HERE / "linear_fulltrain.py",
        HERE / "data_contract.json",
    ]
    return {p.name: digest(p) for p in paths if p.exists()}


def pca_basis(train: np.ndarray, rank: int) -> np.ndarray:
    centered = train - train.mean(axis=0, keepdims=True)
    _, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    return vectors[:, -rank:][:, ::-1].copy().astype(np.float32)


def load_electricity() -> dict[str, Any]:
    contract = json.loads((HERE / "data_contract.json").read_text(encoding="utf-8"))
    path = ROOT / contract["datasets"]["electricity_first32"]["npz"]
    with np.load(path, allow_pickle=False) as archive:
        split_bounds = archive["split_bounds"]
        val_end = int(split_bounds[2])
        data = {
            "x": archive["x"][:val_end],
            "targetmask": archive["targetmask"][:val_end],
            "split_bounds": split_bounds,
            "train_origins": archive["train_origins"],
            "val_origins": archive["val_origins"],
        }
    data["_path"] = path
    data["finite"] = data["targetmask"]
    data["train_start"], data["train_end"] = data["split_bounds"][:2]
    return data


def windows(values: np.ndarray, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([values[o - CONTEXT : o] for o in origins])
    y = np.stack([values[o : o + HORIZON] for o in origins])
    return x, y


def as_examples(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        x.transpose(0, 2, 1).reshape(-1, x.shape[1]).astype(np.float64),
        y.transpose(0, 2, 1).reshape(-1, y.shape[1]).astype(np.float64),
    )


@dataclass
class SufficientStats:
    x_dim: int = CONTEXT
    y_dim: int = HORIZON

    def __post_init__(self) -> None:
        self.n = 0
        self.sum_x = np.zeros(self.x_dim, dtype=np.float64)
        self.sum_y = np.zeros(self.y_dim, dtype=np.float64)
        self.xx = np.zeros((self.x_dim, self.x_dim), dtype=np.float64)
        self.xy = np.zeros((self.x_dim, self.y_dim), dtype=np.float64)
        self.yy_trace = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.shape[1] != self.x_dim or y.shape[1] != self.y_dim:
            raise ValueError(f"shape mismatch: x {x.shape}, y {y.shape}")
        self.n += len(x)
        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.xx += x.T @ x
        self.xy += x.T @ y
        self.yy_trace += float(np.sum(y * y))

    def centered_terms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        if self.n <= 0:
            raise ValueError("empty sufficient statistics")
        xmean = self.sum_x / self.n
        ymean = self.sum_y / self.n
        covariance = self.xx / self.n - np.outer(xmean, xmean)
        cross = self.xy / self.n - np.outer(xmean, ymean)
        y_trace = self.yy_trace / self.n - float(ymean @ ymean)
        return xmean, ymean, covariance, cross, y_trace


def build_stats(data: dict[str, Any], basis: np.ndarray) -> tuple[SufficientStats, SufficientStats]:
    origins = data["train_origins"]
    common = SufficientStats()
    residual = SufficientStats()
    for start in range(0, len(origins), CHUNK_ORIGINS):
        chunk = origins[start : start + CHUNK_ORIGINS]
        x, y = windows(data["x"], chunk)
        zx, zy = x @ basis, y @ basis
        rx, ry = x - zx @ basis.T, y - zy @ basis.T
        common.update(*as_examples(zx, zy))
        residual.update(*as_examples(rx, ry))
    return common, residual


def ridge_from_terms(
    xmean: np.ndarray,
    ymean: np.ndarray,
    covariance: np.ndarray,
    cross: np.ndarray,
    penalty: float,
) -> dict[str, np.ndarray]:
    values, vectors = np.linalg.eigh(covariance)
    inv = 1 / np.maximum(values + penalty, 1e-12)
    weight = (vectors * inv) @ (vectors.T @ cross)
    return {"weight": weight, "bias": ymean - xmean @ weight}


def reduced_rank_ridge_from_terms(
    xmean: np.ndarray,
    ymean: np.ndarray,
    covariance: np.ndarray,
    cross: np.ndarray,
    penalty: float,
    rank: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    values, vectors = np.linalg.eigh(covariance)
    lam = np.maximum(values + penalty, 1e-12)
    full_weight = (vectors * (1 / lam)) @ (vectors.T @ cross)
    sqrt_left = (vectors * np.sqrt(lam)) @ vectors.T
    inv_sqrt_left = (vectors * (1 / np.sqrt(lam))) @ vectors.T
    whitened = sqrt_left @ full_weight
    u, s, vt = np.linalg.svd(whitened, full_matrices=False)
    keep = min(rank, len(s))
    whitened_k = (u[:, :keep] * s[:keep]) @ vt[:keep]
    weight = inv_sqrt_left @ whitened_k
    fit = {"weight": weight, "bias": ymean - xmean @ weight}
    energy = s * s
    total = float(energy.sum())
    diagnostics = {
        "requested_rank": int(rank),
        "kept_rank": int(keep),
        "weight_rank_gt_1e_8": int((np.linalg.svd(weight, compute_uv=False) > 1e-8).sum()),
        "whitened_rank_gt_1e_8": int((s > 1e-8).sum()),
        "whitened_top_singular_values": s[:12].tolist(),
        "whitened_topk_energy_fraction": float(energy[:keep].sum() / total) if total > 0 else 0.0,
        "whitened_tail_energy_fraction": float(energy[keep:].sum() / total) if total > 0 else 0.0,
    }
    return fit, diagnostics


def ridge_objective(
    weight: np.ndarray,
    covariance: np.ndarray,
    cross: np.ndarray,
    y_trace: float,
    penalty: float,
) -> float:
    return float(
        y_trace
        - 2 * np.trace(weight.T @ cross)
        + np.trace(weight.T @ covariance @ weight)
        + penalty * np.sum(weight * weight)
    )


def ridge_predict(x: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    return np.einsum("blc,lh->bhc", x, fit["weight"]) + fit["bias"][None, :, None]


def predict_factor(x: np.ndarray, basis: np.ndarray, common: dict[str, np.ndarray], residual: dict[str, np.ndarray]) -> np.ndarray:
    z = x @ basis
    r = x - z @ basis.T
    return ridge_predict(z, common) @ basis.T + ridge_predict(r, residual)


def score_arrays(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, target {target.shape}, mask {mask.shape}")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(target)):
        raise ValueError("non-finite prediction or target")
    n = mask.sum(axis=(0, 1))
    if not np.all(n > 0):
        raise ValueError("no evaluation target for a selected channel")
    squared = (pred.astype(np.float64) - target.astype(np.float64)) ** 2 * mask
    absolute = np.abs(pred.astype(np.float64) - target.astype(np.float64)) * mask
    return {
        "mse": float(np.mean(squared.sum(axis=(0, 1)) / n)),
        "mae": float(np.mean(absolute.sum(axis=(0, 1)) / n)),
        "channel_mse": (squared.sum(axis=(0, 1)) / n).tolist(),
        "target_counts": n.tolist(),
        "origins": int(len(pred)),
    }


def period_scores(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n = len(pred)
    idxs = {
        "all": np.arange(n),
        "first_half": np.arange(0, n // 2),
        "second_half": np.arange(n // 2, n),
    }
    return {name: score_arrays(pred[idx], target[idx], mask[idx]) for name, idx in idxs.items()}


def artificial_checks() -> dict[str, Any]:
    rng = np.random.default_rng(9262601)
    x = rng.normal(size=(37, 7))
    latent = rng.normal(size=(7, 3)) @ rng.normal(size=(3, 5))
    y = x @ latent + 0.05 * rng.normal(size=(37, 5))
    stats = SufficientStats(x_dim=7, y_dim=5)
    stats.update(x[:11], y[:11])
    stats.update(x[11:], y[11:])
    xmean, ymean, covariance, cross, y_trace = stats.centered_terms()
    full = ridge_from_terms(xmean, ymean, covariance, cross, 0.2)
    rank_full, full_diag = reduced_rank_ridge_from_terms(xmean, ymean, covariance, cross, 0.2, 5)
    rank2, rank2_diag = reduced_rank_ridge_from_terms(xmean, ymean, covariance, cross, 0.2, 2)
    full_obj = ridge_objective(full["weight"], covariance, cross, y_trace, 0.2)
    rank_full_obj = ridge_objective(rank_full["weight"], covariance, cross, y_trace, 0.2)
    rank2_obj = ridge_objective(rank2["weight"], covariance, cross, y_trace, 0.2)
    if np.max(np.abs(full["weight"] - rank_full["weight"])) > 1e-10:
        raise RuntimeError("artificial rank-full reduced-rank ridge does not match full ridge")
    if np.linalg.matrix_rank(rank2["weight"], tol=1e-8) > 2:
        raise RuntimeError("artificial rank-2 solution exceeds requested rank")
    if rank2_obj + 1e-12 < full_obj:
        raise RuntimeError("artificial rank-2 objective is below full ridge objective")
    return {
        "rank_full_weight_max_abs_diff": float(np.max(np.abs(full["weight"] - rank_full["weight"]))),
        "rank_full_bias_max_abs_diff": float(np.max(np.abs(full["bias"] - rank_full["bias"]))),
        "full_objective": float(full_obj),
        "rank_full_objective": float(rank_full_obj),
        "rank2_objective": float(rank2_obj),
        "rank2_minus_full_objective": float(rank2_obj - full_obj),
        "rank2_weight_rank": int(np.linalg.matrix_rank(rank2["weight"], tol=1e-8)),
        "rank2_whitened_tail_energy_fraction": rank2_diag["whitened_tail_energy_fraction"],
        "rank_full_whitened_tail_energy_fraction": full_diag["whitened_tail_energy_fraction"],
    }


def compute() -> None:
    if OUT.exists():
        raise RuntimeError(f"Do not overwrite existing result: {OUT}")
    if LOCAL.exists() and any(LOCAL.iterdir()):
        raise RuntimeError(f"Do not overwrite existing cache directory contents: {LOCAL}")
    LOCAL.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    checks = artificial_checks()
    data = load_electricity()
    if not data["finite"][: int(data["train_end"])].all():
        raise RuntimeError("Electricity TRAIN is expected to be fully observed for this closed-form diagnostic")
    if not data["finite"][int(data["split_bounds"][1]) : int(data["split_bounds"][2])].all():
        raise RuntimeError("Electricity VAL is expected to be fully observed for this diagnostic")

    basis = pca_basis(data["x"][int(data["train_start"]) : int(data["train_end"])], PCA_RANK)
    common_stats, residual_stats = build_stats(data, basis)
    common_terms = common_stats.centered_terms()
    residual_terms = residual_stats.centered_terms()
    common = ridge_from_terms(*common_terms[:4], PENALTY)
    xmean, ymean, covariance, cross, y_trace = residual_terms
    full_residual = ridge_from_terms(xmean, ymean, covariance, cross, PENALTY)

    weights_path = CACHE / "baselines_fulltrain" / f"linear_weights_{PENALTY:g}.npz"
    baseline_val_path = CACHE / "baselines_fulltrain" / f"factor_linear_{PENALTY:g}_val.npz"
    if not weights_path.exists() or not baseline_val_path.exists():
        raise FileNotFoundError("full-train factor-linear baseline weights or VAL predictions are missing")
    with np.load(weights_path, allow_pickle=False) as archive:
        saved_basis = archive["basis"]
        saved_common_weight = archive["common_weight"]
        saved_common_bias = archive["common_bias"]
        saved_residual_weight = archive["residual_weight"]
        saved_residual_bias = archive["residual_bias"]
    basis_max_diff = float(np.max(np.abs(saved_basis - basis)))
    common_weight_diff = float(np.max(np.abs(saved_common_weight - common["weight"])))
    common_bias_diff = float(np.max(np.abs(saved_common_bias - common["bias"])))
    residual_weight_diff = float(np.max(np.abs(saved_residual_weight - full_residual["weight"])))
    residual_bias_diff = float(np.max(np.abs(saved_residual_bias - full_residual["bias"])))
    if max(basis_max_diff, common_weight_diff, common_bias_diff, residual_weight_diff, residual_bias_diff) > 1e-9:
        raise RuntimeError(
            "recomputed full-train ridge terms do not match saved fulltrain weights: "
            f"basis={basis_max_diff}, common_w={common_weight_diff}, common_b={common_bias_diff}, "
            f"residual_w={residual_weight_diff}, residual_b={residual_bias_diff}"
        )

    val_origins = data["val_origins"]
    val_x, val_y = windows(data["x"], val_origins)
    val_mask = np.stack([data["finite"][o : o + HORIZON] for o in val_origins])
    with np.load(baseline_val_path, allow_pickle=False) as archive:
        baseline_pred = archive["prediction"].astype(np.float64)
        baseline_origins = archive["origins"]
    if not np.array_equal(baseline_origins, val_origins):
        raise RuntimeError("saved baseline VAL origins mismatch")
    baseline_scores = period_scores(baseline_pred, val_y, val_mask)

    records = []
    rank48_pred: np.ndarray | None = None
    for rank in RANKS:
        residual_fit, spectrum = reduced_rank_ridge_from_terms(xmean, ymean, covariance, cross, PENALTY, rank)
        objective = ridge_objective(residual_fit["weight"], covariance, cross, y_trace, PENALTY)
        pred = predict_factor(val_x, basis, common, residual_fit)
        scores = period_scores(pred, val_y, val_mask)
        pred_path = LOCAL / f"factor_linear_residual_rank{rank}_val.npz"
        save_npz_new(pred_path, prediction=pred.astype(np.float32), origins=val_origins)
        with np.load(pred_path, allow_pickle=False) as archive:
            replay = period_scores(archive["prediction"].astype(np.float64), val_y, val_mask)
        replay_diff = abs(replay["all"]["mse"] - scores["all"]["mse"])
        if replay_diff > STORED_FLOAT32_SCORE_TOL:
            raise RuntimeError(f"stored prediction replay mismatch for rank {rank}: {replay_diff}")
        if rank == 48:
            rank48_pred = pred
        records.append(
            {
                "rank": int(rank),
                "residual_objective": float(objective),
                "residual_objective_gap_vs_full": float(
                    objective - ridge_objective(full_residual["weight"], covariance, cross, y_trace, PENALTY)
                ),
                "scores": scores,
                "score_gap_vs_fullrank": None,
                "spectrum": spectrum,
                "prediction_file": str(pred_path.relative_to(ROOT)),
                "prediction_sha256": digest(pred_path),
                "stored_prediction_all_mse_abs_diff": float(replay_diff),
            }
        )

    if rank48_pred is None:
        raise RuntimeError("rank48 prediction was not computed")
    rank48_scores = next(r["scores"] for r in records if r["rank"] == 48)
    rank48_pred_max_diff = float(np.max(np.abs(rank48_pred - baseline_pred)))
    rank48_all_mse_diff = float(abs(rank48_scores["all"]["mse"] - baseline_scores["all"]["mse"]))
    if rank48_pred_max_diff > 1e-5 or rank48_all_mse_diff > 1e-10:
        raise RuntimeError(
            f"rank48 does not reproduce saved fullridge VAL baseline: pred_diff={rank48_pred_max_diff}, "
            f"mse_diff={rank48_all_mse_diff}"
        )
    for record in records:
        record["score_gap_vs_fullrank"] = {
            period: {
                "mse": float(record["scores"][period]["mse"] - rank48_scores[period]["mse"]),
                "mae": float(record["scores"][period]["mae"] - rank48_scores[period]["mae"]),
            }
            for period in ("all", "first_half", "second_half")
        }

    result = {
        "status": "complete",
        "dataset": "electricity",
        "scope": "CPU-only reduced-rank residual ridge diagnostic on Electricity VAL. No neural fit, no new TSFM prediction, no DEV, no Bull.",
        "decision_boundary": "This diagnostic tests whether factor-linear residual temporal correction needs more than rank 8 under the fixed full-TRAIN PCA8/common-map/ridge setup. It is not a neural architecture selection or protected evaluation.",
        "cache_subdirectory": "residual_rank_diagnostic",
        "source": source_receipt(),
        "data_sha256": digest(data["_path"]),
        "baseline_weights_file": str(weights_path.relative_to(ROOT)),
        "baseline_weights_sha256": digest(weights_path),
        "baseline_val_prediction_file": str(baseline_val_path.relative_to(ROOT)),
        "baseline_val_prediction_sha256": digest(baseline_val_path),
        "ridge_penalty": PENALTY,
        "pca_rank": PCA_RANK,
        "residual_ranks": list(RANKS),
        "chunk_origins": CHUNK_ORIGINS,
        "threadpool_limits": 2,
        "actual_training_origin_count": int(len(data["train_origins"])),
        "actual_training_origin_hash": array_digest(data["train_origins"]),
        "component_training_examples": {
            "common": int(common_stats.n),
            "residual": int(residual_stats.n),
        },
        "diagnostic_transform_count": len(RANKS),
        "cumulative_cpu_fit_accounting": {
            "previous_cpu_fits_from_existing_baselines": 18,
            "this_diagnostic_reduced_rank_transforms": len(RANKS),
            "total_cpu_fits_after_this_diagnostic": 18 + len(RANKS),
        },
        "saved_fulltrain_weight_replay": {
            "basis_max_abs_diff": basis_max_diff,
            "common_weight_max_abs_diff": common_weight_diff,
            "common_bias_max_abs_diff": common_bias_diff,
            "residual_weight_max_abs_diff": residual_weight_diff,
            "residual_bias_max_abs_diff": residual_bias_diff,
            "rank48_prediction_max_abs_diff_vs_saved_val_npz": rank48_pred_max_diff,
            "rank48_all_mse_abs_diff_vs_saved_val_npz": rank48_all_mse_diff,
        },
        "baseline_rank48_scores": rank48_scores,
        "records": records,
        "artificial_checks": checks,
        "elapsed_s": time.perf_counter() - started,
    }
    save_json_new(OUT, result)
    print(
        json.dumps(
            {
                "elapsed_s": result["elapsed_s"],
                "records": [
                    {
                        "rank": r["rank"],
                        "all_mse": r["scores"]["all"]["mse"],
                        "first_half_mse": r["scores"]["first_half"]["mse"],
                        "second_half_mse": r["scores"]["second_half"]["mse"],
                        "gap_vs_rank48": r["score_gap_vs_fullrank"]["all"]["mse"],
                        "tail_energy": r["spectrum"]["whitened_tail_energy_fraction"],
                    }
                    for r in records
                ],
                "rank48_replay": result["saved_fulltrain_weight_replay"],
                "output": str(OUT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        compute()
