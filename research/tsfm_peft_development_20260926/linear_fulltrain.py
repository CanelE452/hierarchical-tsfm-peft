from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / "tsfm_peft_development_20260926"
OUT = HERE / "linear_fulltrain_baselines.json"
LOCAL = CACHE / "baselines_fulltrain"
OLD_BASELINES = CACHE / "baselines"
PENALTIES = (0.001, 0.1, 10.0)
CONTEXT = 512
HORIZON = 48
CHUNK_ORIGINS = 128
RANK = 8


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(values: np.ndarray) -> str:
    a = np.ascontiguousarray(values.astype("<i8", copy=False))
    return hashlib.sha256(a.tobytes()).hexdigest()


def save_json_new(path: Path, data: dict) -> None:
    if path.exists():
        raise RuntimeError(f"Do not overwrite existing result: {path}")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_npz_new(path: Path, **arrays) -> None:
    if path.exists():
        raise RuntimeError(f"Do not overwrite existing cache file: {path}")
    np.savez_compressed(path, **arrays)


def source_receipt() -> dict[str, str]:
    paths = [
        HERE / "linear_fulltrain.py",
        HERE / "linear_baselines.py",
        HERE / "data_contract.json",
        HERE / "model.py",
    ]
    return {p.name: digest(p) for p in paths if p.exists()}


def pca_basis(train: np.ndarray, rank: int) -> np.ndarray:
    centered = train - train.mean(axis=0, keepdims=True)
    _, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    return vectors[:, -rank:][:, ::-1].copy().astype(np.float32)


def load_electricity() -> dict:
    contract = json.loads((HERE / "data_contract.json").read_text(encoding="utf-8"))
    path = ROOT / contract["datasets"]["electricity_first32"]["npz"]
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
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
        x.transpose(0, 2, 1).reshape(-1, CONTEXT).astype(np.float64),
        y.transpose(0, 2, 1).reshape(-1, HORIZON).astype(np.float64),
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

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        self.n += len(x)
        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.xx += x.T @ x
        self.xy += x.T @ y

    def fit(self, penalty: float) -> dict[str, np.ndarray]:
        if self.n <= 0:
            raise ValueError("empty sufficient statistics")
        xmean = self.sum_x / self.n
        ymean = self.sum_y / self.n
        covariance = self.xx / self.n - np.outer(xmean, xmean)
        cross = self.xy / self.n - np.outer(xmean, ymean)
        values, vectors = np.linalg.eigh(covariance)
        inv = 1 / np.maximum(values + penalty, 1e-9)
        weight = (vectors * inv) @ (vectors.T @ cross)
        return {"weight": weight, "bias": ymean - xmean @ weight}


def direct_ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float) -> dict[str, np.ndarray]:
    ex, ey = as_examples(x, y)
    xmean, ymean = ex.mean(0), ey.mean(0)
    xc, yc = ex - xmean, ey - ymean
    covariance = xc.T @ xc / len(ex)
    cross = xc.T @ yc / len(ex)
    values, vectors = np.linalg.eigh(covariance)
    inv = 1 / np.maximum(values + penalty, 1e-9)
    weight = (vectors * inv) @ (vectors.T @ cross)
    return {"weight": weight, "bias": ymean - xmean @ weight}


def streaming_checks() -> dict:
    rng = np.random.default_rng(9262026)
    x = rng.normal(size=(5, CONTEXT, 3)).astype(np.float64)
    y = rng.normal(size=(5, HORIZON, 3)).astype(np.float64)
    stats = SufficientStats()
    ex, ey = as_examples(x, y)
    stats.update(ex[:7], ey[:7])
    stats.update(ex[7:], ey[7:])
    direct = direct_ridge_fit(x, y, 0.1)
    streamed = stats.fit(0.1)
    return {
        "artificial_weight_max_abs_diff": float(np.max(np.abs(direct["weight"] - streamed["weight"]))),
        "artificial_bias_max_abs_diff": float(np.max(np.abs(direct["bias"] - streamed["bias"]))),
        "artificial_prediction_max_abs_diff": float(
            np.max(np.abs(ridge_predict(x, direct) - ridge_predict(x, streamed)))
        ),
    }


def ridge_predict(x: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    return np.einsum("blc,lh->bhc", x, fit["weight"]) + fit["bias"][None, :, None]


def score_arrays(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    n = mask.sum(axis=(0, 1))
    if not np.all(n > 0):
        raise ValueError("no evaluation target for a selected channel")
    squared = (pred.astype(np.float64) - target) ** 2 * mask
    absolute = np.abs(pred.astype(np.float64) - target) * mask
    return {
        "mse": float(np.mean(squared.sum(axis=(0, 1)) / n)),
        "mae": float(np.mean(absolute.sum(axis=(0, 1)) / n)),
        "channel_mse": (squared.sum(axis=(0, 1)) / n).tolist(),
        "target_counts": n.tolist(),
        "origins": len(pred),
    }


def score_prediction_file(path: Path, data: dict, split: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        pred = archive["prediction"]
        origins = archive["origins"]
    target = np.stack([data["x"][o : o + HORIZON] for o in origins])
    mask = np.stack([data["finite"][o : o + HORIZON] for o in origins])
    return score_arrays(pred, target, mask)


def build_stats(data: dict, basis: np.ndarray) -> tuple[SufficientStats, SufficientStats, SufficientStats]:
    origins = data["train_origins"]
    common = SufficientStats()
    residual = SufficientStats()
    direct = SufficientStats()
    for start in range(0, len(origins), CHUNK_ORIGINS):
        chunk = origins[start : start + CHUNK_ORIGINS]
        x, y = windows(data["x"], chunk)
        zx, zy = x @ basis, y @ basis
        rx, ry = x - zx @ basis.T, y - zy @ basis.T
        common.update(*as_examples(zx, zy))
        residual.update(*as_examples(rx, ry))
        direct.update(*as_examples(x, y))
    return common, residual, direct


def predict_factor(x: np.ndarray, basis: np.ndarray, common: dict, residual: dict) -> np.ndarray:
    z = x @ basis
    r = x - z @ basis.T
    return ridge_predict(z, common) @ basis.T + ridge_predict(r, residual)


def copy_or_compute_seasonal(data: dict, split: str, x: np.ndarray, y: np.ndarray) -> dict:
    dst = LOCAL / f"seasonal_{split}.npz"
    src = OLD_BASELINES / f"seasonal_{split}.npz"
    if src.exists():
        if dst.exists():
            raise RuntimeError(f"Do not overwrite existing cache file: {dst}")
        shutil.copy2(src, dst)
        return score_prediction_file(dst, data, split)
    pred = np.tile(x[:, -24:, :], (1, 2, 1))
    mask = np.stack([data["finite"][o : o + HORIZON] for o in data[f"{split}_origins"]])
    score = score_arrays(pred, y, mask)
    save_npz_new(dst, prediction=pred.astype(np.float32), origins=data[f"{split}_origins"])
    return score


def compute() -> None:
    if OUT.exists():
        raise RuntimeError(f"Do not overwrite existing result: {OUT}")
    if LOCAL.exists() and any(LOCAL.iterdir()):
        raise RuntimeError(f"Do not overwrite existing cache directory contents: {LOCAL}")
    LOCAL.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    data = load_electricity()
    if not data["finite"][: int(data["train_end"])].all():
        raise ValueError("Full-train closed-form baseline currently assumes fully observed Electricity TRAIN")
    train_origins = data["train_origins"].astype(np.int64)
    basis = pca_basis(data["x"][int(data["train_start"]) : int(data["train_end"])], RANK)
    checks = streaming_checks()
    common_stats, residual_stats, direct_stats = build_stats(data, basis)
    component_training_examples = {
        "common": int(common_stats.n),
        "residual": int(residual_stats.n),
        "direct": int(direct_stats.n),
    }

    splits = {key: windows(data["x"], data[f"{key}_origins"]) for key in ("val", "dev")}
    records: list[dict] = []
    prediction_verification: list[dict] = []
    for penalty in PENALTIES:
        common = common_stats.fit(penalty)
        residual = residual_stats.fit(penalty)
        direct = direct_stats.fit(penalty)
        for arm in ("factor_linear", "shared_linear"):
            scores = {}
            for split, (sx, sy) in splits.items():
                if arm == "factor_linear":
                    pred = predict_factor(sx, basis, common, residual)
                else:
                    pred = ridge_predict(sx, direct)
                mask = np.stack([data["finite"][o : o + HORIZON] for o in data[f"{split}_origins"]])
                scores[split] = score_arrays(pred, sy, mask)
                pred_path = LOCAL / f"{arm}_{penalty:g}_{split}.npz"
                save_npz_new(pred_path, prediction=pred.astype(np.float32), origins=data[f"{split}_origins"])
                replay = score_prediction_file(pred_path, data, split)
                prediction_verification.append(
                    {
                        "file": str(pred_path.relative_to(ROOT)),
                        "split": split,
                        "arm": arm,
                        "ridge_penalty": penalty,
                        "mse_abs_diff": abs(replay["mse"] - scores[split]["mse"]),
                    }
                )
            records.append({"arm": arm, "ridge_penalty": penalty, "scores": scores})
        save_npz_new(
            LOCAL / f"linear_weights_{penalty:g}.npz",
            basis=basis,
            common_weight=common["weight"],
            common_bias=common["bias"],
            residual_weight=residual["weight"],
            residual_bias=residual["bias"],
            direct_weight=direct["weight"],
            direct_bias=direct["bias"],
        )

    seasonal = {}
    for split, (sx, sy) in splits.items():
        seasonal[split] = copy_or_compute_seasonal(data, split, sx, sy)

    selected = {
        arm: min([r for r in records if r["arm"] == arm], key=lambda r: r["scores"]["val"]["mse"])
        for arm in ("factor_linear", "shared_linear")
    }
    result = {
        "status": "complete",
        "dataset": "electricity",
        "cache_subdirectory": "baselines_fulltrain",
        "source": source_receipt(),
        "data_sha256": digest(data["_path"]),
        "actual_training_origin_count": int(len(train_origins)),
        "actual_training_origin_hash": array_digest(train_origins),
        "training_origin_scope": "all Electricity TRAIN legal origins from data NPZ; no phase sampling",
        "chunk_origins": CHUNK_ORIGINS,
        "component_fit_count": 9,
        "prediction_records": 6,
        "ridge_penalties": list(PENALTIES),
        "component_training_examples": component_training_examples,
        "selection": "minimum validation MSE, never dev",
        "records": records,
        "selected": selected,
        "seasonal": seasonal,
        "streaming_checks": checks
        | {
            "prediction_score_max_mse_abs_diff": float(
                max(v["mse_abs_diff"] for v in prediction_verification)
            ),
            "prediction_files_checked": len(prediction_verification),
            "prediction_verification": prediction_verification,
        },
        "elapsed_s": time.perf_counter() - started,
        "scope": "All-TRAIN legal-origin stronger CPU linear controls. Fixed TRAIN PCA8, shared temporal map+bias, common/residual/direct closed-form ridge. Electricity development only; no Bull access.",
    }
    save_json_new(OUT, result)
    print(
        json.dumps(
            {
                "elapsed_s": result["elapsed_s"],
                "actual_training_origin_count": result["actual_training_origin_count"],
                "selected": {
                    a: {
                        "penalty": r["ridge_penalty"],
                        "val_mse": r["scores"]["val"]["mse"],
                        "dev_mse": r["scores"]["dev"]["mse"],
                    }
                    for a, r in selected.items()
                },
                "seasonal_dev": seasonal["dev"]["mse"],
                "streaming_checks": checks,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        compute()
