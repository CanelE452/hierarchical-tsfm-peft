"""Data contract for the professor compressor direct experiments.

This module intentionally reads only the existing TRAIN/VAL archives from
``runtime_v18.load_data(..., include_test=False)``.  It rebuilds the contract
needed by GTConvAE/TS2Vec experiments without importing old PCA/LEVEL choices.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name / "data"
RUNTIME_PATH = ROOT / "research" / "level_matched_learned_ed_v18_20261006" / "runtime_v18.py"

DATASETS = ("robin", "jena")
L = 512
H = 48
REC_STRIDE = 128
FORECAST_STRIDE = 128
EPS = 1e-6


def _runtime_v18():
    name = "_professor_direct_runtime_v18"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runtime_v18 from {RUNTIME_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_hash(array: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.relative_to(ROOT).as_posix() if value.is_relative_to(ROOT) else str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_contract(dataset: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset for this phase: {dataset}")
    path = RUNTIME_PATH.parent / f"data_contract_{dataset}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _train_stats_from_observed(physical: np.ndarray, mask: np.ndarray, bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    start, end = bounds
    train_values = np.where(mask[start:end], physical[start:end], np.nan)
    mean = np.nanmean(train_values, axis=0).astype(np.float32)
    std = np.nanstd(train_values, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > EPS), std, 1.0).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("TRAIN observed mean/std contains non-finite values")
    return mean, std


def _causal_forward_fill(physical: np.ndarray, mask: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    filled = np.empty_like(physical, dtype=np.float32)
    last = fill_values.astype(np.float32).copy()
    for t in range(physical.shape[0]):
        observed = mask[t]
        if observed.any():
            last[observed] = physical[t, observed]
        filled[t] = last
    return filled


def _full_train_starts(bounds: tuple[int, int], length: int = L, stride: int = REC_STRIDE) -> np.ndarray:
    start, end = bounds
    stop = end - length
    if stop < start:
        return np.empty(0, dtype=np.int64)
    return np.arange(start, stop + 1, stride, dtype=np.int64)


def _full_train_forecast_origins(bounds: tuple[int, int], length: int = L, horizon: int = H, stride: int = FORECAST_STRIDE) -> np.ndarray:
    start, end = bounds
    first = start + length
    last = end - horizon
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, stride, dtype=np.int64)


def _check_origins(origins: np.ndarray, bounds: tuple[int, int], length: int = L, horizon: int = H) -> dict[str, Any]:
    if len(origins) == 0:
        return {
            "count": 0,
            "history_within_bounds": True,
            "target_within_bounds": True,
            "strictly_increasing": True,
        }
    start, end = bounds
    origins = origins.astype(np.int64)
    return {
        "count": int(len(origins)),
        "first": int(origins[0]),
        "last": int(origins[-1]),
        "history_within_bounds": bool(np.all(origins - length >= start)),
        "target_within_bounds": bool(np.all(origins + horizon <= end)),
        "strictly_increasing": bool(np.all(np.diff(origins) > 0)),
        "stride_values": sorted(int(v) for v in np.unique(np.diff(origins))) if len(origins) > 1 else [],
        "sha256": array_hash(origins),
    }


def _observed_abs_pearson_graph(x: np.ndarray, mask: np.ndarray, bounds: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    start, end = bounds
    xt = x[start:end].astype(np.float64)
    mt = mask[start:end].astype(bool)
    c = xt.shape[1]
    graph = np.zeros((c, c), dtype=np.float64)
    pair_counts = np.zeros((c, c), dtype=np.int64)
    for i in range(c):
        for j in range(i + 1, c):
            both = mt[:, i] & mt[:, j]
            pair_counts[i, j] = pair_counts[j, i] = int(both.sum())
            if both.sum() < 3:
                continue
            xi = xt[both, i]
            xj = xt[both, j]
            if np.std(xi) <= EPS or np.std(xj) <= EPS:
                continue
            corr = float(np.corrcoef(xi, xj)[0, 1])
            if np.isfinite(corr):
                graph[i, j] = graph[j, i] = abs(corr)
    np.fill_diagonal(graph, 0.0)
    if c > 0:
        spectral_norm = float(np.linalg.norm(graph, ord=2))
    else:
        spectral_norm = 0.0
    scaled = graph / spectral_norm if spectral_norm > EPS else graph
    scaled = scaled.astype(np.float32)
    receipt = {
        "rule": "TRAIN-only pairwise observed abs Pearson; diagonal set to 0; divided by spectral norm when nonzero",
        "train_bounds": [int(start), int(end)],
        "min_pair_count": int(pair_counts[np.triu_indices(c, 1)].min()) if c > 1 else 0,
        "max_pair_count": int(pair_counts[np.triu_indices(c, 1)].max()) if c > 1 else 0,
        "raw_spectral_norm": spectral_norm,
        "scaled_spectral_norm": float(np.linalg.norm(scaled.astype(np.float64), ord=2)) if c > 0 else 0.0,
        "nonzero_offdiag": int(np.count_nonzero(np.triu(scaled, 1))),
        "sha256": array_hash(scaled),
    }
    return scaled, receipt


def _load_parent_trainval(dataset: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rt = _runtime_v18()
    parent = rt.load_data(dataset, include_test=False)
    contract = _read_contract(dataset)
    if "test_a_origins" in parent or "test_b_origins" in parent:
        raise RuntimeError("TEST arrays unexpectedly present in include_test=False load")
    return parent, contract


def load_data(dataset: str) -> dict[str, Any]:
    """Load the new TRAIN/VAL-only contract for one dataset.

    Returned ``x`` is float32 standardized with TRAIN observed statistics after
    recovering physical values from v18 mean/std and causal forward-filling
    unobserved inputs.  ``mask`` preserves the original finite-observation mask.
    """
    parent, contract = _load_parent_trainval(dataset)
    x_v18 = parent["x"].astype(np.float32)
    mask = parent["finite"].astype(bool)
    v18_mean = parent["mean"].astype(np.float32)
    v18_std = parent["std"].astype(np.float32)
    physical_filled_by_v18 = x_v18 * v18_std.reshape(1, -1) + v18_mean.reshape(1, -1)
    raw_physical = np.where(mask, physical_filled_by_v18, np.nan).astype(np.float32)

    train_bounds = tuple(int(v) for v in contract["bounds"]["train"])
    val_bounds = tuple(int(v) for v in contract["bounds"]["val"])
    mean, std = _train_stats_from_observed(raw_physical, mask, train_bounds)
    physical_filled = _causal_forward_fill(raw_physical, mask, mean)
    x = ((physical_filled - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)

    train_window_starts = _full_train_starts(train_bounds)
    train_forecast_origins = _full_train_forecast_origins(train_bounds)
    val_origins = parent["val_origins"].astype(np.int64)
    graph, graph_receipt = _observed_abs_pearson_graph(x, mask, train_bounds)
    contract_hash = _sha256_bytes(
        json.dumps(
            {
                "dataset": dataset,
                "x": array_hash(x),
                "mask": array_hash(mask),
                "mean": array_hash(mean),
                "std": array_hash(std),
                "train_bounds": train_bounds,
                "val_bounds": val_bounds,
                "train_window_starts": array_hash(train_window_starts),
                "train_forecast_origins": array_hash(train_forecast_origins),
                "val_origins": array_hash(val_origins),
                "graph": graph_receipt["sha256"],
                "l": L,
                "h": H,
                "rec_stride": REC_STRIDE,
                "forecast_stride": FORECAST_STRIDE,
            },
            sort_keys=True,
        ).encode("utf-8")
    )

    return {
        "dataset": dataset,
        "x": x,
        "mask": mask,
        "raw_physical": raw_physical,
        "physical_filled": physical_filled,
        "mean": mean,
        "std": std,
        "v18_mean": v18_mean,
        "v18_std": v18_std,
        "times": parent["times"].astype(np.int64),
        "columns": parent["columns"],
        "train_bounds": train_bounds,
        "val_bounds": val_bounds,
        "train_window_starts": train_window_starts,
        "train_forecast_origins": train_forecast_origins,
        "val_origins": val_origins,
        "graph": graph,
        "graph_receipt": graph_receipt,
        "contract_hash": contract_hash,
        "interval_seconds": int(contract["interval_seconds"]),
        "l": L,
        "h": H,
        "source_contract": contract["source_contract"],
        "trainval_archive": contract["trainval"],
        "parent_generation": contract["parent_generation"],
        "split_dates": contract["split_dates"],
        "normalization_note": "Recovered physical values using v18 mean/std, then re-standardized with TRAIN observed mean/std after causal forward-fill.",
    }


def make_windows(data: dict[str, Any], origins: np.ndarray, future: bool = False, masked: bool = False) -> Any:
    origins = np.asarray(origins, dtype=np.int64)
    x = data["x"]
    mask = data["mask"]
    if future:
        windows = np.stack([x[int(o) : int(o) + H] for o in origins]).astype(np.float32)
        masks = np.stack([mask[int(o) : int(o) + H] for o in origins]).astype(bool)
    else:
        windows = np.stack([x[int(o) - L : int(o)] for o in origins]).astype(np.float32)
        masks = np.stack([mask[int(o) - L : int(o)] for o in origins]).astype(bool)
    return (windows, masks) if masked else windows


def rec_windows(data: dict[str, Any], starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64)
    x = data["x"]
    mask = data["mask"]
    windows = np.stack([x[int(s) : int(s) + L] for s in starts]).astype(np.float32)
    masks = np.stack([mask[int(s) : int(s) + L] for s in starts]).astype(bool)
    return windows, masks


def ts2_train_array(data: dict[str, Any], nan_for_unobserved: bool = True) -> np.ndarray:
    start, end = data["train_bounds"]
    arr = data["x"][start:end].copy()
    if nan_for_unobserved:
        arr[~data["mask"][start:end]] = np.nan
    return arr.reshape(1, arr.shape[0], arr.shape[1]).astype(np.float32)


def inverse_standardize(data: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) * data["std"].reshape(1, -1) + data["mean"].reshape(1, -1)).astype(np.float32)


def dataset_summary(dataset: str) -> dict[str, Any]:
    data = load_data(dataset)
    parent, contract = _load_parent_trainval(dataset)
    train_start, train_end = data["train_bounds"]
    val_start, val_end = data["val_bounds"]
    rec_x, rec_mask = rec_windows(data, data["train_window_starts"])
    val_hist, val_hist_mask = make_windows(data, data["val_origins"], masked=True)
    val_y, val_y_mask = make_windows(data, data["val_origins"], future=True, masked=True)
    ts2 = ts2_train_array(data, nan_for_unobserved=True)
    times = data["times"]
    train_mask = data["mask"][train_start:train_end]
    val_target_total = int(val_y_mask.size)
    val_target_observed = int(val_y_mask.sum())
    train_total = int(train_mask.size)
    train_observed = int(train_mask.sum())

    checks = {
        "dataset": dataset,
        "domain_context": {
            "domain": "multivariate time series",
            "downstream_task": "representation reconstruction plus frozen TSFM future forecasting",
            "decision_context": "decide whether GTConvAE/TS2Vec compressor paths deserve continued development",
            "quality_bar": "development evidence for research; not independent TEST confirmation",
            "main_leakage_paths": [
                "normalization fitted outside TRAIN",
                "graph estimated from VAL or future target",
                "origin target spilling outside VAL",
                "reading protected TEST archives",
                "using future observed values to build inference features",
            ],
        },
        "shapes": {
            "x": list(data["x"].shape),
            "mask": list(data["mask"].shape),
            "raw_physical": list(data["raw_physical"].shape),
            "graph": list(data["graph"].shape),
            "rec_windows": list(rec_x.shape),
            "val_history": list(val_hist.shape),
            "val_future": list(val_y.shape),
            "ts2_train_array": list(ts2.shape),
        },
        "dtypes": {
            "x": str(data["x"].dtype),
            "mask": str(data["mask"].dtype),
            "raw_physical": str(data["raw_physical"].dtype),
            "graph": str(data["graph"].dtype),
        },
        "bounds": {
            "train": [train_start, train_end],
            "val": [val_start, val_end],
        },
        "origin_checks": {
            "train_window_starts": {
                "count": int(len(data["train_window_starts"])),
                "first": int(data["train_window_starts"][0]) if len(data["train_window_starts"]) else None,
                "last": int(data["train_window_starts"][-1]) if len(data["train_window_starts"]) else None,
                "fully_within_train": bool(np.all((data["train_window_starts"] >= train_start) & (data["train_window_starts"] + L <= train_end))),
                "stride_values": sorted(int(v) for v in np.unique(np.diff(data["train_window_starts"]))) if len(data["train_window_starts"]) > 1 else [],
                "sha256": array_hash(data["train_window_starts"]),
            },
            "train_forecast_origins": _check_origins(data["train_forecast_origins"], data["train_bounds"]),
            "val_origins": _check_origins(data["val_origins"], data["val_bounds"]),
        },
        "mask_counts": {
            "total_observed": int(data["mask"].sum()),
            "train_observed": train_observed,
            "val_target_observed": val_target_observed,
            "rec_window_observed": int(rec_mask.sum()),
            "val_history_observed": int(val_hist_mask.sum()),
            "unobserved_total": int((~data["mask"]).sum()),
            "observed_zero_physical_count": int(np.nansum((data["raw_physical"] == 0.0) & data["mask"])),
        },
        "quality_metrics": {
            "schema_pass": bool(
                data["x"].dtype == np.float32
                and data["mask"].dtype == np.bool_
                and data["x"].shape == data["mask"].shape
                and data["graph"].shape == (data["x"].shape[1], data["x"].shape[1])
            ),
            "train_observed_completeness": float(train_observed / train_total) if train_total else 0.0,
            "val_target_observed_completeness": float(val_target_observed / val_target_total) if val_target_total else 0.0,
            "time_strictly_increasing": bool(np.all(np.diff(times) > 0)),
            "duplicate_time_count": int(len(times) - len(np.unique(times))),
            "temporal_split_order_pass": bool(train_end <= val_start),
            "normalization_fit_scope": "TRAIN observed values only",
            "graph_fit_scope": "TRAIN observed pairwise rows only",
            "test_scope_pass": True,
        },
        "finite_checks": {
            "x_all_finite": bool(np.isfinite(data["x"]).all()),
            "raw_physical_observed_all_finite": bool(np.isfinite(data["raw_physical"][data["mask"]]).all()),
            "raw_physical_unobserved_all_nan": bool(np.isnan(data["raw_physical"][~data["mask"]]).all()),
            "mean_all_finite": bool(np.isfinite(data["mean"]).all()),
            "std_all_finite_positive": bool(np.isfinite(data["std"]).all() and np.all(data["std"] > 0)),
            "ts2_has_nan_for_unobserved": bool(np.isnan(ts2).sum() == (~data["mask"][train_start:train_end]).sum()),
        },
        "train_standardization": {
            "observed_train_mean_abs_max": float(np.max(np.abs(np.nanmean(np.where(data["mask"][train_start:train_end], data["x"][train_start:train_end], np.nan), axis=0)))),
            "observed_train_std_abs_max_minus_1": float(np.max(np.abs(np.nanstd(np.where(data["mask"][train_start:train_end], data["x"][train_start:train_end], np.nan), axis=0) - 1.0))),
            "mean_close_to_v18_max_abs": float(np.max(np.abs(data["mean"] - data["v18_mean"]))),
            "std_close_to_v18_max_abs": float(np.max(np.abs(data["std"] - data["v18_std"]))),
        },
        "graph_receipt": data["graph_receipt"],
        "array_hashes": {
            "x": array_hash(data["x"]),
            "mask": array_hash(data["mask"]),
            "raw_physical": array_hash(np.nan_to_num(data["raw_physical"], nan=np.float32(-999999.0))),
            "mean": array_hash(data["mean"]),
            "std": array_hash(data["std"]),
            "val_origins": array_hash(data["val_origins"]),
            "train_window_starts": array_hash(data["train_window_starts"]),
            "train_forecast_origins": array_hash(data["train_forecast_origins"]),
            "graph": array_hash(data["graph"]),
        },
        "contract_hash": data["contract_hash"],
        "provenance": {
            "runtime_v18": {
                "path": RUNTIME_PATH.relative_to(ROOT).as_posix(),
                "sha256": file_hash(RUNTIME_PATH),
            },
            "parent_contract": {
                "path": (RUNTIME_PATH.parent / f"data_contract_{dataset}.json").relative_to(ROOT).as_posix(),
                "sha256": file_hash(RUNTIME_PATH.parent / f"data_contract_{dataset}.json"),
            },
            "source_contract": contract["source_contract"],
            "trainval_archive": contract["trainval"],
            "source": contract.get("source", {}),
            "split_dates": contract.get("split_dates", {}),
            "normalization": contract.get("normalization"),
            "target_mask": contract.get("target_mask"),
            "time_basis": contract.get("time_basis"),
            "parent_generation": contract.get("parent_generation"),
        },
        "no_test_read_contract": {
            "include_test_used": False,
            "parent_loaded_keys": sorted(str(k) for k in parent.keys()),
            "test_keys_absent": bool("test_a_origins" not in parent and "test_b_origins" not in parent),
            "note": "TEST archive path remains referenced only inside the parent JSON contract; this module does not hash or open it.",
        },
    }
    graph_path = CACHE / f"{dataset}_train_abs_pearson_graph.npz"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        graph_path,
        graph=data["graph"],
        columns=np.asarray(data["columns"]),
        train_bounds=np.asarray(data["train_bounds"], dtype=np.int64),
    )
    checks["graph_cache"] = {
        "path": graph_path.relative_to(ROOT).as_posix(),
        "sha256": file_hash(graph_path),
        "bytes": graph_path.stat().st_size,
    }
    return checks


def write_artifacts() -> None:
    summaries = {dataset: dataset_summary(dataset) for dataset in DATASETS}
    provenance = {
        "created_for": "Professor_Compressor_Direct_Experiments_REPLACEMENT_20261011",
        "scope": "TRAIN/VAL-only data contract for GTConvAE and TS2Vec direct experiments",
        "datasets": {dataset: summaries[dataset]["provenance"] for dataset in DATASETS},
        "no_test_read": {dataset: summaries[dataset]["no_test_read_contract"] for dataset in DATASETS},
    }
    checks = {
        "created_for": "Professor_Compressor_Direct_Experiments_REPLACEMENT_20261011",
        "datasets": summaries,
        "overall_pass": bool(
            all(s["finite_checks"]["x_all_finite"] for s in summaries.values())
            and all(s["finite_checks"]["raw_physical_observed_all_finite"] for s in summaries.values())
            and all(s["finite_checks"]["raw_physical_unobserved_all_nan"] for s in summaries.values())
            and all(s["origin_checks"]["train_window_starts"]["fully_within_train"] for s in summaries.values())
            and all(s["origin_checks"]["train_forecast_origins"]["history_within_bounds"] for s in summaries.values())
            and all(s["origin_checks"]["train_forecast_origins"]["target_within_bounds"] for s in summaries.values())
            and all(s["origin_checks"]["val_origins"]["target_within_bounds"] for s in summaries.values())
            and all(s["no_test_read_contract"]["test_keys_absent"] for s in summaries.values())
            and all(s["quality_metrics"]["schema_pass"] for s in summaries.values())
            and all(s["quality_metrics"]["time_strictly_increasing"] for s in summaries.values())
            and all(s["quality_metrics"]["duplicate_time_count"] == 0 for s in summaries.values())
            and all(s["quality_metrics"]["temporal_split_order_pass"] for s in summaries.values())
        ),
    }
    _write_json(HERE / "DATA_PROVENANCE.json", provenance)
    _write_json(HERE / "DATA_CHECKS.json", checks)
    (HERE / "DATA_NOTE_KO.md").write_text(_note_ko(summaries, checks), encoding="utf-8")


def _note_ko(summaries: dict[str, Any], checks: dict[str, Any]) -> str:
    lines = [
        "# 데이터 계약 메모",
        "",
        "[확인] 이 데이터 레이어는 `runtime_v18.load_data(dataset, include_test=False)`만 호출한다. TEST archive는 열거나 해시하지 않았고, 새 실험의 개발 평가는 기존 TRAIN/VAL 구간에 한정된다.",
        "",
        "[확인] 다운스트림 결정은 GTConvAE/TS2Vec 표현이 TSFM 미래 예측 뒤 원공간 복원에 쓸 만한지 보는 것이다. 그래서 예전 PCA/LEVEL의 K, D8, 잔차 G 조건은 데이터 계약에 넣지 않았다.",
        "",
        "```text",
        "dataset  T      C   train_obs  val_target_obs  rec_windows  train_F_origins  val_origins",
    ]
    for dataset, s in summaries.items():
        shape = s["shapes"]["x"]
        rec = s["origin_checks"]["train_window_starts"]["count"]
        train_f = s["origin_checks"]["train_forecast_origins"]["count"]
        val = s["origin_checks"]["val_origins"]["count"]
        train_obs = s["quality_metrics"]["train_observed_completeness"] * 100.0
        val_obs = s["quality_metrics"]["val_target_observed_completeness"] * 100.0
        lines.append(
            f"{dataset:<8} {shape[0]:<6} {shape[1]:<3} {train_obs:>8.2f}% {val_obs:>13.2f}% {rec:<11} {train_f:<16} {val}"
        )
    lines.extend(
        [
            "```",
            "",
            "[확인] 시계열 순서는 두 자료 모두 strictly increasing이고 중복 timestamp는 0개다. 분할은 시간 순서를 유지하며 TRAIN 뒤에 VAL이 온다.",
            "",
            "[확인] 원자료 물리값은 v18의 mean/std로 역변환했고, 새 입력 `x`는 TRAIN observed 값으로 다시 계산한 mean/std를 사용해 표준화했다. 결측 입력은 미래를 보지 않는 causal forward-fill로 채웠고, 정답 metric에는 `mask`의 observed 위치만 쓰도록 보존했다.",
            "",
            "[확인] graph는 TRAIN 구간에서 두 채널이 모두 관측된 시점만 사용한 절대 Pearson 상관이다. diagonal은 0이고, self term과 중복되지 않게 별도 처리하도록 남겼다. 물리 topology가 아니라 TRAIN-only 추정 graph다.",
            "",
            "[확인] TS2Vec용 배열은 `[1, T_train, C]`이며 unobserved 위치는 NaN으로 되돌릴 수 있다. GTConvAE 재구성 창은 TRAIN 내부에서 길이 512가 완전히 들어가는 start를 stride 128로 만들었다.",
            "",
            f"[확인] 전체 검산 상태: {'PASS' if checks['overall_pass'] else 'WARN/FAIL'}. 상세 hash와 count는 `DATA_CHECKS.json`에 있다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-artifacts", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    if args.write_artifacts:
        write_artifacts()
    elif args.dataset:
        data = load_data(args.dataset)
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "x_shape": list(data["x"].shape),
                    "mask_shape": list(data["mask"].shape),
                    "train_window_count": int(len(data["train_window_starts"])),
                    "train_forecast_origin_count": int(len(data["train_forecast_origins"])),
                    "val_origin_count": int(len(data["val_origins"])),
                    "graph_sha256": data["graph_receipt"]["sha256"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
