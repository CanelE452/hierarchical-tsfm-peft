"""TRAIN/VAL-only data contract for the LEVEL compressor fullscreen run.

This module deliberately opens only existing trainval archives through the
v18 runtime with include_test=False. It rebuilds fold-local normalization from
observed raw values recovered from the stored train-fitted mean/std.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME_V18 = ROOT / "research" / "level_matched_learned_ed_v18_20261006" / "runtime_v18.py"

CONTEXT = 512
HORIZON = 48
FIT_ORIGIN_COUNT = 512
CALIBRATION_ORIGIN_COUNT = 256
SAMPLING_SEED = 92603
PRIMARY_DATASETS = ("robin", "jena", "peacock_education")
FOLD_NAMES = ("canonical", "backtest_1", "backtest_2")
FOLD_ALIASES = {None: "canonical", "": "canonical", "canonical": "canonical", 0: "backtest_1", "0": "backtest_1", 1: "backtest_2", "1": "backtest_2", "backtest_1": "backtest_1", "backtest_2": "backtest_2"}
GRAPH_TOP_K = 3


def _import_file(path: Path, module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _runtime():
    return _import_file(RUNTIME_V18, "_fullscreen_runtime_v18")


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _public_path(path: Path | str) -> str:
    resolved = Path(path)
    text = str(resolved)
    root = str(ROOT)
    return text.replace(root + "\\", "").replace(ROOT.as_posix() + "/", "")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _public_path(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _legal_origins(bounds: tuple[int, int], stride: int = 1) -> np.ndarray:
    start, end = int(bounds[0]), int(bounds[1])
    first = max(start, CONTEXT)
    last = end - HORIZON
    if last < first:
        return np.empty((0,), dtype=np.int64)
    return np.arange(first, last + 1, int(stride), dtype=np.int64)


def _sample_sorted(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if len(values) <= int(count):
        return np.sort(values)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(values, size=int(count), replace=False).astype(np.int64))


def _split_three(bounds: tuple[int, int]) -> dict[str, tuple[int, int]]:
    start, end = int(bounds[0]), int(bounds[1])
    length = end - start
    fit_end = start + int(np.floor(length * 0.70))
    calibration_end = start + int(np.floor(length * 0.85))
    result = {
        "fit": (start, fit_end),
        "calibration": (fit_end, calibration_end),
        "selection": (calibration_end, end),
    }
    for label, segment in result.items():
        if len(_legal_origins(segment)) == 0:
            raise RuntimeError(f"{label} segment has no legal origins: {segment}")
    return result


def _canonical_fold_name(fold: str | int | None) -> str:
    if fold in FOLD_ALIASES:
        return FOLD_ALIASES[fold]
    fold_name = str(fold)
    if fold_name in FOLD_ALIASES:
        return FOLD_ALIASES[fold_name]
    raise ValueError(f"unknown fold {fold!r}; expected canonical/backtest_1/backtest_2 or integer aliases 0/1")


def _fold_plan(bounds: dict[str, list[int]], fold: str | int | None) -> dict[str, Any]:
    fold_name = _canonical_fold_name(fold)
    if fold_name not in FOLD_NAMES:
        raise ValueError(f"unknown fold {fold_name!r}; expected one of {FOLD_NAMES}")
    train = tuple(int(v) for v in bounds["train"])
    val = tuple(int(v) for v in bounds["val"])
    if fold_name == "canonical":
        prefix = train
        eval_bounds = val
        eval_stride = 24
        eval_source = "existing_outer_val"
    else:
        start, end = train
        length = end - start
        cut70 = start + int(np.floor(length * 0.70))
        cut85 = start + int(np.floor(length * 0.85))
        if fold_name == "backtest_1":
            prefix = (start, cut70)
            eval_bounds = (cut70, cut85)
        else:
            prefix = (start, cut85)
            eval_bounds = (cut85, end)
        eval_stride = 24
        eval_source = "train_internal_chronological_backtest"
    split = _split_three(prefix)
    eval_origins = _legal_origins(eval_bounds, stride=eval_stride)
    if len(eval_origins) == 0:
        raise RuntimeError(f"{fold_name} eval segment has no legal origins: {eval_bounds}")
    return {
        "fold": fold_name,
        "prefix_bounds": prefix,
        "fit_bounds": split["fit"],
        "calibration_bounds": split["calibration"],
        "selection_bounds": split["selection"],
        "eval_bounds": eval_bounds,
        "eval_stride": eval_stride,
        "eval_source": eval_source,
        "purge_policy": "Origins are generated only when the full target window is contained in its own segment; adjacent segments therefore have zero target timestamp overlap for H=48.",
    }


def _recover_raw(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x_old = np.asarray(data["x"], dtype=np.float32)
    finite = np.asarray(data["finite"], dtype=bool)
    old_mean = np.asarray(data["mean"], dtype=np.float32)
    old_std = np.asarray(data["std"], dtype=np.float32)
    if old_mean.shape != (x_old.shape[1],) or old_std.shape != (x_old.shape[1],):
        raise RuntimeError("Stored mean/std shape is incompatible with x")
    if not np.isfinite(old_mean).all() or not np.isfinite(old_std).all() or np.any(old_std <= 0):
        raise RuntimeError("Stored mean/std cannot recover raw observed values")
    raw = x_old.astype(np.float32) * old_std[None, :] + old_mean[None, :]
    roundtrip = (raw[finite] - np.broadcast_to(old_mean, x_old.shape)[finite]) / np.broadcast_to(old_std, x_old.shape)[finite]
    check = {
        "old_norm_recoverable_from_stored_mean_std": True,
        "observed_roundtrip_max_abs": float(np.max(np.abs(roundtrip - x_old[finite]))) if finite.any() else None,
    }
    raw = np.where(finite, raw, np.nan).astype(np.float32)
    return raw, old_mean, old_std, check


def _fit_normalizer(raw: np.ndarray, finite: np.ndarray, bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start, end = int(bounds[0]), int(bounds[1])
    mask = finite[start:end] & np.isfinite(raw[start:end])
    mean = np.zeros(raw.shape[1], dtype=np.float32)
    std = np.ones(raw.shape[1], dtype=np.float32)
    zero_variance = []
    observed_counts = mask.sum(axis=0).astype(np.int64)
    for channel in range(raw.shape[1]):
        values = raw[start:end, channel][mask[:, channel]].astype(np.float64)
        if len(values) == 0:
            raise RuntimeError(f"no observed fit values for channel {channel}")
        mean[channel] = np.float32(values.mean())
        value_std = float(values.std(ddof=0))
        if not np.isfinite(value_std) or value_std < 1e-6:
            std[channel] = np.float32(1.0)
            zero_variance.append(int(channel))
        else:
            std[channel] = np.float32(value_std)
    report = {
        "fit_bounds": [start, end],
        "observed_counts_by_channel": observed_counts.tolist(),
        "zero_variance_channels_std_forced_to_one": zero_variance,
    }
    return mean, std, report


def _normalize_ffill(raw: np.ndarray, finite: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (raw.astype(np.float32) - mean[None, :]) / std[None, :]
    output = np.zeros(raw.shape, dtype=np.float32)
    for channel in range(raw.shape[1]):
        last = np.float32(0.0)
        for row in range(raw.shape[0]):
            if finite[row, channel] and np.isfinite(z[row, channel]):
                last = np.float32(z[row, channel])
            output[row, channel] = last
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite value after fold-local normalization")
    return output


def _train_abs_pearson_graph(raw: np.ndarray, finite: np.ndarray, bounds: tuple[int, int], top_k: int = GRAPH_TOP_K) -> tuple[np.ndarray, dict[str, Any]]:
    start, end = int(bounds[0]), int(bounds[1])
    c = int(raw.shape[1])
    corr = np.zeros((c, c), dtype=np.float64)
    counts = np.zeros((c, c), dtype=np.int64)
    for i in range(c):
        for j in range(i + 1, c):
            mask = finite[start:end, i] & finite[start:end, j] & np.isfinite(raw[start:end, i]) & np.isfinite(raw[start:end, j])
            counts[i, j] = counts[j, i] = int(mask.sum())
            if int(mask.sum()) < 3:
                continue
            xi = raw[start:end, i][mask].astype(np.float64)
            xj = raw[start:end, j][mask].astype(np.float64)
            xi = xi - xi.mean()
            xj = xj - xj.mean()
            denom = float(np.sqrt(np.dot(xi, xi) * np.dot(xj, xj)))
            if denom > 0:
                value = abs(float(np.dot(xi, xj) / denom))
                corr[i, j] = corr[j, i] = value
    chosen = np.zeros((c, c), dtype=bool)
    k = max(0, min(int(top_k), c - 1))
    for i in range(c):
        if k == 0:
            continue
        order = np.argsort(-corr[i], kind="stable")
        kept = [int(j) for j in order if j != i and corr[i, j] > 0.0][:k]
        for j in kept:
            chosen[i, j] = True
            chosen[j, i] = True
    adjacency = np.where(chosen, corr, 0.0).astype(np.float64)
    np.fill_diagonal(adjacency, 0.0)
    adjacency = np.maximum(adjacency, adjacency.T)
    edge_count = int(np.count_nonzero(np.triu(adjacency > 0, 1)))
    if edge_count:
        eigvals = np.linalg.eigvalsh(adjacency)
        spectral_radius = float(np.max(np.abs(eigvals)))
    else:
        spectral_radius = 0.0
    if spectral_radius > 0:
        graph = (adjacency / spectral_radius).astype(np.float32)
        spectral_radius_after = 1.0
    else:
        graph = adjacency.astype(np.float32)
        spectral_radius_after = 0.0
    contract = {
        "rule": "TRAIN-fit-only symmetric abs(Pearson) graph; each node proposes top-3 neighbors, undirected union is kept, weights are abs correlations, adjacency is divided by spectral radius.",
        "source": "No physical graph metadata exists in the local Robin/Jena/Peacock trainval contracts or high-C inventory, so this graph is a fixed TRAIN-only estimated graph.",
        "warning": "This differs from earlier row-normalized graphs, which can be asymmetric and should not be described as an undirected spatial graph.",
        "bounds": [start, end],
        "top_k": int(top_k),
        "edge_count_undirected": edge_count,
        "spectral_radius_before": spectral_radius,
        "spectral_radius_after": spectral_radius_after,
        "pairwise_observed_count_min": int(counts[np.triu_indices(c, 1)].min()) if c > 1 else 0,
        "pairwise_observed_count_max": int(counts[np.triu_indices(c, 1)].max()) if c > 1 else 0,
        "graph_sha256": array_sha256(graph),
    }
    return graph, contract


def _contract_hash_material(
    *,
    name: str,
    plan: dict[str, Any],
    source_sha256: str,
    finite: np.ndarray,
    timestamps: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    train_fit_origins: np.ndarray,
    calibration_origins: np.ndarray,
    selection_origins: np.ndarray,
    val_origins: np.ndarray,
    outer_val_origins: np.ndarray,
    graph: np.ndarray,
    graph_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "level_compressor_fullscreen_contract_hash_v1",
        "dataset": name,
        "fold": plan["fold"],
        "source_sha256": source_sha256,
        "mask_sha256": array_sha256(finite.astype(np.uint8)),
        "timestamps": {
            "dtype": str(timestamps.dtype),
            "shape": list(timestamps.shape),
            "sha256": array_sha256(timestamps),
        },
        "normalization": {
            "mean_sha256": array_sha256(norm_mean),
            "std_sha256": array_sha256(norm_std),
            "fit_bounds": list(plan["fit_bounds"]),
            "dtype": str(norm_mean.dtype),
        },
        "origins": {
            "train_fit": _origin_receipt(train_fit_origins),
            "calibration": _origin_receipt(calibration_origins),
            "selection": _origin_receipt(selection_origins),
            "val": _origin_receipt(val_origins),
            "outer_val": _origin_receipt(outer_val_origins),
        },
        "fold_plan": plan,
        "precision": {
            "x": "float32",
            "raw": "float32",
            "mask": "bool",
            "origins": "int64",
            "graph": "float32",
            "normalization_fit": "float64 accumulation then float32 storage",
        },
        "graph": {
            "graph_sha256": array_sha256(graph),
            "rule": graph_contract["rule"],
            "bounds": graph_contract["bounds"],
            "top_k": graph_contract["top_k"],
        },
    }


def load_dataset(name: str, fold: str | int | None = None) -> dict[str, Any]:
    """Load one TRAIN/VAL-only dataset with fold-local normalization.

    For fold=None/canonical, val_origins are the existing outer VAL origins.
    For backtest folds, val_origins are the fold evaluation origins inside TRAIN;
    outer_val_origins keeps the existing development VAL untouched.
    """

    runtime = _runtime()
    if name not in PRIMARY_DATASETS:
        raise ValueError(f"unknown dataset {name!r}")
    contract = runtime.dataset_contract(name)
    data = runtime.load_data(name, include_test=False)
    raw, old_mean, old_std, recovery = _recover_raw(data)
    finite = np.asarray(data["finite"], dtype=bool)
    bounds = {key: [int(v) for v in contract["bounds"][key]] for key in ("train", "val", "test_a", "test_b")}
    plan = _fold_plan(bounds, fold)
    norm_mean, norm_std, norm_report = _fit_normalizer(raw, finite, plan["fit_bounds"])
    x = _normalize_ffill(raw, finite, norm_mean, norm_std)
    train_graph, graph_contract = _train_abs_pearson_graph(raw, finite, plan["fit_bounds"], GRAPH_TOP_K)

    fit_candidates = _legal_origins(plan["fit_bounds"], stride=1)
    calibration_candidates = _legal_origins(plan["calibration_bounds"], stride=1)
    selection_origins = _legal_origins(plan["selection_bounds"], stride=24)
    val_origins = np.asarray(data["val_origins"], dtype=np.int64) if plan["fold"] == "canonical" else _legal_origins(plan["eval_bounds"], stride=24)
    outer_val_origins = np.asarray(data["val_origins"], dtype=np.int64)
    canonical_train_origins = np.asarray(data["train_origins"], dtype=np.int64)
    train_fit_origins = _sample_sorted(fit_candidates, FIT_ORIGIN_COUNT, SAMPLING_SEED)
    calibration_origins = _sample_sorted(calibration_candidates, CALIBRATION_ORIGIN_COUNT, SAMPLING_SEED + 1)

    for label, origins, segment in (
        ("train_fit_origins", train_fit_origins, plan["fit_bounds"]),
        ("calibration_origins", calibration_origins, plan["calibration_bounds"]),
        ("selection_origins", selection_origins, plan["selection_bounds"]),
        ("val_origins", val_origins, plan["eval_bounds"]),
    ):
        if len(origins) == 0:
            raise RuntimeError(f"{label} is empty")
        if int(origins.min()) < max(int(segment[0]), CONTEXT) or int(origins.max()) + HORIZON > int(segment[1]):
            raise RuntimeError(f"{label} crosses its segment bounds")

    source_path = ROOT / contract["trainval"]["path"]
    actual_sha = _sha256_path(source_path)
    if actual_sha != contract["trainval"]["sha256"]:
        raise RuntimeError(f"trainval artifact hash changed for {name}")

    columns = [str(v) for v in np.asarray(data["columns"]).tolist()]
    timestamps = np.asarray(data["times"])
    hash_material = _contract_hash_material(
        name=name,
        plan=plan,
        source_sha256=actual_sha,
        finite=finite,
        timestamps=timestamps,
        norm_mean=norm_mean,
        norm_std=norm_std,
        train_fit_origins=train_fit_origins,
        calibration_origins=calibration_origins,
        selection_origins=selection_origins,
        val_origins=val_origins,
        outer_val_origins=outer_val_origins,
        graph=train_graph,
        graph_contract=graph_contract,
    )
    contract_hash = json_sha256(hash_material)
    return {
        "name": name,
        "contract_hash": contract_hash,
        "fold": plan["fold"],
        "x": x,
        "raw": raw,
        "finite": finite,
        "columns": columns,
        "timestamps": timestamps,
        "context": CONTEXT,
        "horizon": HORIZON,
        "train_fit_origins": train_fit_origins,
        "calibration_origins": calibration_origins,
        "selection_origins": selection_origins,
        "val_origins": val_origins,
        "eval_origins": val_origins,
        "outer_val_origins": outer_val_origins,
        "existing_val_origins": outer_val_origins,
        "canonical_train_origins": canonical_train_origins,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "train_graph": train_graph,
        "graph": train_graph,
        "old_norm_mean": old_mean,
        "old_norm_std": old_std,
        "fit_bounds": tuple(plan["fit_bounds"]),
        "calibration_bounds": tuple(plan["calibration_bounds"]),
        "selection_bounds": tuple(plan["selection_bounds"]),
        "eval_bounds": tuple(plan["eval_bounds"]),
        "original_bounds": bounds,
        "contract": {
            "dataset_contract": contract,
            "fold_plan": plan,
            "source_path": _public_path(source_path),
            "source_sha256": actual_sha,
            "normalization": norm_report,
            "graph": graph_contract,
            "contract_hash": contract_hash,
            "contract_hash_material": hash_material,
            "raw_recovery": recovery,
            "test_values_opened": False,
        },
    }


def _origin_receipt(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return {"count": 0, "first": None, "last": None, "sha256": array_sha256(values)}
    return {
        "count": int(len(values)),
        "first": int(values[0]),
        "last": int(values[-1]),
        "sha256": array_sha256(values),
    }


def _dataset_summary(name: str, fold: str) -> dict[str, Any]:
    bundle = load_dataset(name, None if fold == "canonical" else fold)
    finite = bundle["finite"]
    original_bounds = bundle["original_bounds"]
    def frac(bounds: tuple[int, int]) -> float:
        start, end = int(bounds[0]), int(bounds[1])
        return float(finite[start:end].mean())
    return {
        "dataset": name,
        "fold": fold,
        "shape": list(bundle["x"].shape),
        "columns": len(bundle["columns"]),
        "context": CONTEXT,
        "horizon": HORIZON,
        "source_path": bundle["contract"]["source_path"],
        "source_sha256": bundle["contract"]["source_sha256"],
        "bounds": {
            "fit": list(bundle["fit_bounds"]),
            "calibration": list(bundle["calibration_bounds"]),
            "selection": list(bundle["selection_bounds"]),
            "eval": list(bundle["eval_bounds"]),
            "outer_val": original_bounds["val"],
            "test_a": original_bounds["test_a"],
            "test_b": original_bounds["test_b"],
        },
        "origins": {
            "train_fit": _origin_receipt(bundle["train_fit_origins"]),
            "calibration": _origin_receipt(bundle["calibration_origins"]),
            "selection": _origin_receipt(bundle["selection_origins"]),
            "val": _origin_receipt(bundle["val_origins"]),
            "outer_val": _origin_receipt(bundle["outer_val_origins"]),
        },
        "finite_fraction": {
            "all_trainval": float(finite.mean()),
            "fit": frac(bundle["fit_bounds"]),
            "calibration": frac(bundle["calibration_bounds"]),
            "selection": frac(bundle["selection_bounds"]),
            "eval": frac(bundle["eval_bounds"]),
        },
        "normalization": {
            "fit_only": True,
            "mean_sha256": array_sha256(bundle["norm_mean"]),
            "std_sha256": array_sha256(bundle["norm_std"]),
            "zero_variance_channels": bundle["contract"]["normalization"]["zero_variance_channels_std_forced_to_one"],
        },
        "graph": bundle["contract"]["graph"],
        "contract_hash": bundle["contract"]["contract_hash"],
        "raw_recovery": bundle["contract"]["raw_recovery"],
        "test_values_opened": False,
    }


def _hierarchical_inventory() -> list[dict[str, Any]]:
    candidates = [
        ROOT / "data" / "hierarchical" / "Traffic" / "data.csv",
        ROOT / "data" / "hierarchical" / "TourismLarge" / "data.csv",
        ROOT / "data" / "hierarchical" / "Wiki2" / "data.csv",
        ROOT / "data" / "hierarchical" / "Labour" / "data.csv",
    ]
    rows = []
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            first = next(reader, None)
            row_count = 1 if first is not None else 0
            for _ in reader:
                row_count += 1
        channels = max(0, len(header) - 1)
        rows.append({
            "path": _public_path(path),
            "sha256": _sha256_path(path),
            "rows": int(row_count),
            "channels_excluding_time": int(channels),
            "first_timestamp": first[0] if first else None,
            "current_L512_H48_accuracy_status": "BLOCKED_SPLIT" if row_count < CONTEXT + HORIZON else "candidate_requires_contract",
            "reason": "raw high-C local source exists, but rows are shorter than L+H=560 or lack the current L512/H48 TRAIN/VAL/mask contract",
        })
    return rows


def _gift_eval_inventory() -> list[dict[str, Any]]:
    base = ROOT / "data" / "gift_eval"
    rows = []
    for info in sorted(base.glob("**/dataset_info.json")):
        try:
            data = json.loads(info.read_text(encoding="utf-8"))
        except Exception:
            continue
        split = data.get("splits", {}).get("train", {})
        target = data.get("features", {}).get("target", {})
        rows.append({
            "path": _public_path(info),
            "dataset_name": data.get("dataset_name"),
            "num_examples": split.get("num_examples"),
            "dataset_size": data.get("dataset_size"),
            "target_feature": target,
            "current_multivariate_contract_status": "BLOCKED_DATA",
            "reason": "local GIFT-Eval shards are univariate/example-oriented; no verified aligned CxT L512/H48 TRAIN/VAL/mask contract is present in this run",
        })
    return rows


def build_contract_summary() -> dict[str, Any]:
    dataset_folds = []
    for name in PRIMARY_DATASETS:
        for fold in FOLD_NAMES:
            dataset_folds.append(_dataset_summary(name, fold))
    return {
        "schema": "level_compressor_fullscreen_data_contract_v1",
        "scope": "TRAIN/VAL-only contract for A-H execution; protected TEST value archives are not opened.",
        "domain": {
            "type": "multivariate_time_series_forecasting",
            "downstream_decision": "Compare channel compressors and LEVEL residual paths under frozen univariate TSFM cost/accuracy constraints.",
            "primary_leakage_paths": [
                "normalization fitted beyond the fold fit prefix",
                "missing-value imputation using future observations",
                "target windows overlapping across fit/calibration/selection/evaluation roles",
                "opening protected TEST value arrays",
            ],
            "quality_bar": "publication-oriented exploratory experiment ledger; zero tolerance for TEST leakage",
        },
        "constants": {
            "context": CONTEXT,
            "horizon": HORIZON,
            "fit_origin_sample_count": FIT_ORIGIN_COUNT,
            "calibration_origin_sample_count": CALIBRATION_ORIGIN_COUNT,
            "sampling_seed": SAMPLING_SEED,
            "selection_stride": 24,
            "eval_stride": 24,
        },
        "primary_datasets": dataset_folds,
        "local_extra_data": {
            "peacock_education": "available as a primary TRAIN/VAL-only v18 dataset; not a new unseen dataset",
            "high_channel_hierarchical_sources": _hierarchical_inventory(),
            "gift_eval_sources": _gift_eval_inventory(),
        },
        "loader_contract": {
            "entrypoint": "research.level_compressor_fullscreen_20261011.data_contract.load_dataset(name, fold=None)",
            "folds": list(FOLD_NAMES),
            "fold_aliases": {"None": "canonical", "0": "backtest_1", "1": "backtest_2"},
            "returned_arrays": {
                "x": "float32 [T,C], fold-fit normalized, missing values filled only from previous observed normalized values or zero before first observation",
                "raw": "float32 [T,C], observed original-scale values with NaN where finite mask is false",
                "finite": "bool [T,C], original observation mask",
                "timestamps": "stored timestamp array from trainval archive",
                "train_graph": "float32 [C,C], symmetric TRAIN-fit-only abs(Pearson) top-3-union adjacency scaled to spectral radius <= 1",
            },
            "returned_origins": {
                "train_fit_origins": "sorted sample up to 512 legal target-contained origins from fold fit segment, seed 92603",
                "calibration_origins": "sorted sample up to 256 legal target-contained origins from fold calibration segment, seed 92604",
                "selection_origins": "stride-24 legal target-contained origins from fold selection segment",
                "val_origins": "canonical outer VAL origins for canonical fold; fold eval origins for backtest folds",
                "eval_origins": "alias of val_origins for the active fold",
                "outer_val_origins": "existing development VAL origins, preserved for all folds",
                "existing_val_origins": "alias of outer_val_origins",
                "canonical_train_origins": "existing TRAIN origins from the source trainval archive, preserved for reuse checks",
            },
            "hash_policy": "contract_hash hashes source artifact sha256, finite mask, fold-local normalization, origin receipts, fold plan, timestamp hash, precision contract, and graph rule/hash.",
            "graph_policy": "No physical graph metadata was found for the primary trainval contracts; the returned graph is an estimated TRAIN-only undirected graph. Earlier row-normalized graph utilities are asymmetric and should be reported separately.",
            "test_policy": "runtime_v18.load_data(..., include_test=False) only; TEST bounds may be recorded from JSON metadata, but TEST value archives are not opened.",
        },
    }


def write_summary(path: Path = HERE / "DATA_CONTRACT.json") -> dict[str, Any]:
    summary = build_contract_summary()
    path.write_text(json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--dataset", choices=PRIMARY_DATASETS)
    parser.add_argument("--fold", choices=list(FOLD_NAMES) + ["0", "1"], default="canonical")
    args = parser.parse_args()
    if args.write_summary:
        summary = write_summary()
        print(json.dumps({
            "path": _public_path(HERE / "DATA_CONTRACT.json"),
            "dataset_fold_rows": len(summary["primary_datasets"]),
            "schema": summary["schema"],
        }, indent=2))
        return
    if args.dataset:
        bundle = load_dataset(args.dataset, None if args.fold == "canonical" else args.fold)
        print(json.dumps(_to_jsonable({
            "dataset": args.dataset,
            "fold": bundle["fold"],
            "shape": list(bundle["x"].shape),
            "train_fit_origins": _origin_receipt(bundle["train_fit_origins"]),
            "calibration_origins": _origin_receipt(bundle["calibration_origins"]),
            "selection_origins": _origin_receipt(bundle["selection_origins"]),
            "val_origins": _origin_receipt(bundle["val_origins"]),
            "fit_bounds": bundle["fit_bounds"],
            "eval_bounds": bundle["eval_bounds"],
            "contract_hash": bundle["contract"]["contract_hash"],
            "graph_sha256": bundle["contract"]["graph"]["graph_sha256"],
            "graph_edges": bundle["contract"]["graph"]["edge_count_undirected"],
            "test_values_opened": bundle["contract"]["test_values_opened"],
        }), indent=2))


if __name__ == "__main__":
    main()
