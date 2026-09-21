from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from hier_peft.audit import RESULTS, sha256, utc_now, write_json
from hier_peft.hierarchy import (
    HierarchyGraph,
    build_support_inclusion_graph,
    hierarchy_audit,
    validate_graph_contract,
)


OFFICIAL_SOURCE_URL = "https://nixtla-public.s3.amazonaws.com/hierarchical-data/datasets.zip"
REFERENCE_DATASETSFORECAST_COMMIT = "f85c26352bc5c2e37b70be729e3e2cfb89b590a2"
ALLOWED_GROUPS = ("Labour", "TourismLarge")
DEFAULT_CONTEXT = 48
DEFAULT_HORIZON = 12
SUMMING_TOLERANCE = 1e-8


class DataFailure(RuntimeError):
    def __init__(self, code: str, message: str, artifacts: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.artifacts = artifacts or {}


@dataclass(frozen=True)
class HierarchicalDataset:
    group: str
    Y_df: pd.DataFrame
    S_df: pd.DataFrame
    values: np.ndarray
    dates: pd.DatetimeIndex
    ids: tuple[str, ...]
    S: np.ndarray
    tags: dict[str, np.ndarray]
    tag_ids: dict[str, tuple[str, ...]]
    bottom_ids: tuple[str, ...]
    bottom_indices: np.ndarray
    graph: HierarchyGraph
    parents: dict[int, tuple[int, ...]]
    children: dict[int, tuple[int, ...]]
    receipt: dict[str, Any]
    data_audit: dict[str, Any]
    hierarchy_audit: dict[str, Any]
    split_manifest: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _loader_source_metadata() -> dict[str, Any]:
    try:
        import datasetsforecast.hierarchical as hierarchical
    except Exception as exc:  # pragma: no cover - exercised in environment preflight
        return {
            "datasetsforecast_version": _package_version("datasetsforecast"),
            "loader_import_status": "FAIL",
            "loader_import_error": repr(exc),
        }
    loader_path = Path(inspect.getfile(hierarchical)).resolve()
    return {
        "datasetsforecast_version": _package_version("datasetsforecast"),
        "loader_import_status": "PASS",
        "loader_file": str(loader_path),
        "loader_file_sha256": sha256(loader_path),
        "reference_commit_from_master": REFERENCE_DATASETSFORECAST_COMMIT,
        "official_source_url": OFFICIAL_SOURCE_URL,
    }


def _discovered_source_files(directory: Path, group: str) -> list[dict[str, Any]]:
    candidates = [
        directory / "hierarchical" / "datasets.zip",
        directory / "hierarchical" / f"{group}.p",
        directory / "hierarchical" / group / "data.csv",
        directory / "hierarchical" / group / "agg_mat.csv",
    ]
    receipts: list[dict[str, Any]] = []
    for path in candidates:
        if path.exists():
            receipts.append({
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return receipts


def load_official_frames(
    group: str,
    directory: str | Path = "data",
    *,
    cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if group not in ALLOWED_GROUPS:
        raise DataFailure(
            "DATASET_NOT_IN_MASTER_SCOPE",
            f"{group!r} is not one of {ALLOWED_GROUPS}",
        )
    try:
        from datasetsforecast.hierarchical import HierarchicalData
    except Exception as exc:
        raise DataFailure(
            "DATASETFORECAST_IMPORT_FAILED",
            "Could not import datasetsforecast.hierarchical.HierarchicalData",
            {"data_audit": _minimal_failure_audit(group, "DATASETFORECAST_IMPORT_FAILED", repr(exc))},
        ) from exc

    directory = Path(directory)
    try:
        Y_df, S_df, tags = HierarchicalData.load(
            directory=str(directory),
            group=group,
            cache=cache,
        )
    except Exception as exc:
        raise DataFailure(
            "OFFICIAL_DATA_LOAD_FAILED",
            f"Official HierarchicalData.load failed for {group}",
            {"data_audit": _minimal_failure_audit(group, "OFFICIAL_DATA_LOAD_FAILED", repr(exc))},
        ) from exc
    source = _loader_source_metadata()
    source["local_source_files"] = _discovered_source_files(directory, group)
    source["loader_cache_argument"] = cache
    return Y_df, S_df, tags, source


def _minimal_failure_audit(group: str, code: str, message: str) -> dict[str, Any]:
    return {
        "group": group,
        "status": "FAIL",
        "failure_type": "data failure",
        "blocker_code": code,
        "message": message,
        "created_at": utc_now(),
        "issues": [{"severity": "FAIL", "code": code, "message": message}],
    }


def _normalize_y(Y_df: pd.DataFrame, issues: list[dict[str, Any]]) -> pd.DataFrame:
    expected = {"unique_id", "ds", "y"}
    missing_columns = sorted(expected - set(Y_df.columns))
    if missing_columns:
        issues.append({
            "severity": "FAIL",
            "code": "Y_SCHEMA_MISSING_COLUMNS",
            "columns": missing_columns,
        })
        return Y_df.copy()

    ydf = Y_df.loc[:, ["unique_id", "ds", "y"]].copy()
    ydf["unique_id"] = ydf["unique_id"].astype(str)
    try:
        ydf["ds"] = pd.to_datetime(ydf["ds"], errors="raise")
    except Exception as exc:
        issues.append({"severity": "FAIL", "code": "Y_DS_PARSE_FAILED", "message": repr(exc)})
        return ydf
    if getattr(ydf["ds"].dt, "tz", None) is not None:
        issues.append({"severity": "FAIL", "code": "Y_DS_TIMEZONE_AWARE"})
    ydf["y"] = pd.to_numeric(ydf["y"], errors="coerce")
    return ydf


def _normalize_s(S_df: pd.DataFrame, issues: list[dict[str, Any]]) -> pd.DataFrame:
    if S_df.index.has_duplicates:
        issues.append({"severity": "FAIL", "code": "S_INDEX_DUPLICATED"})
    if S_df.columns.has_duplicates:
        issues.append({"severity": "FAIL", "code": "S_COLUMNS_DUPLICATED"})
    sdf = S_df.copy()
    sdf.index = [str(value) for value in sdf.index]
    sdf.columns = [str(value) for value in sdf.columns]
    if len(set(sdf.index)) != len(sdf.index):
        issues.append({"severity": "FAIL", "code": "S_INDEX_STRING_COLLISION"})
    if len(set(sdf.columns)) != len(sdf.columns):
        issues.append({"severity": "FAIL", "code": "S_COLUMNS_STRING_COLLISION"})
    for column in sdf.columns:
        sdf[column] = pd.to_numeric(sdf[column], errors="coerce")
    return sdf


def _normalize_tags(
    tags: dict[str, Any],
    ids: tuple[str, ...],
    issues: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    id_to_idx = {series_id: idx for idx, series_id in enumerate(ids)}
    normalized: dict[str, np.ndarray] = {}
    tag_ids: dict[str, tuple[str, ...]] = {}
    seen: list[int] = []

    for level, values in tags.items():
        level_name = str(level)
        arr = np.asarray(values)
        if arr.size == 0:
            issues.append({"severity": "FAIL", "code": "TAG_LEVEL_EMPTY", "level": level_name})
            normalized[level_name] = np.asarray([], dtype=np.int64)
            tag_ids[level_name] = tuple()
            continue

        if np.issubdtype(arr.dtype, np.integer) and np.all((arr >= 0) & (arr < len(ids))):
            indices = arr.astype(np.int64)
            names = tuple(ids[int(idx)] for idx in indices)
        else:
            names = tuple(str(value) for value in arr.tolist())
            unknown = [name for name in names if name not in id_to_idx]
            if unknown:
                issues.append({
                    "severity": "FAIL",
                    "code": "TAG_UNKNOWN_UNIQUE_ID",
                    "level": level_name,
                    "sample": unknown[:10],
                    "count": len(unknown),
                })
            indices = np.asarray(
                [id_to_idx[name] for name in names if name in id_to_idx],
                dtype=np.int64,
            )
        if len(np.unique(indices)) != len(indices):
            issues.append({"severity": "FAIL", "code": "TAG_LEVEL_DUPLICATES", "level": level_name})
        normalized[level_name] = indices
        tag_ids[level_name] = tuple(names)
        seen.extend(int(idx) for idx in indices)

    seen_counts = pd.Series(seen, dtype="int64").value_counts() if seen else pd.Series(dtype="int64")
    duplicate_nodes = [int(idx) for idx, count in seen_counts.items() if int(count) > 1]
    missing_nodes = [idx for idx in range(len(ids)) if idx not in set(seen)]
    if duplicate_nodes:
        issues.append({
            "severity": "FAIL",
            "code": "TAG_COVERAGE_DUPLICATED_NODES",
            "indices": duplicate_nodes[:20],
            "count": len(duplicate_nodes),
        })
    if missing_nodes:
        issues.append({
            "severity": "FAIL",
            "code": "TAG_COVERAGE_MISSING_NODES",
            "indices": missing_nodes[:20],
            "count": len(missing_nodes),
        })
    return normalized, tag_ids


def _issue_failed(issues: Iterable[dict[str, Any]]) -> bool:
    return any(issue.get("severity") == "FAIL" for issue in issues)


def _first_failure_code(issues: Iterable[dict[str, Any]], default: str) -> str:
    for issue in issues:
        if issue.get("severity") == "FAIL":
            return str(issue.get("code", default))
    return default


def build_data_receipt(
    group: str,
    Y_df: pd.DataFrame | None,
    S_df: pd.DataFrame | None,
    tags: dict[str, Any] | None,
    source: dict[str, Any] | None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "group": group,
        "created_at": utc_now(),
        "official_source_url": OFFICIAL_SOURCE_URL,
        "datasetsforecast_reference_commit_from_master": REFERENCE_DATASETSFORECAST_COMMIT,
        "source": source or _loader_source_metadata(),
    }
    if Y_df is not None:
        receipt["Y_shape"] = list(Y_df.shape)
        receipt["Y_columns"] = [str(column) for column in Y_df.columns]
        if {"unique_id", "ds", "y"}.issubset(Y_df.columns):
            ydf = Y_df.loc[:, ["unique_id", "ds", "y"]].copy()
            receipt["unique_series_count"] = int(ydf["unique_id"].nunique(dropna=False))
            try:
                dates = pd.to_datetime(ydf["ds"], errors="raise")
                receipt["date_min"] = pd.Timestamp(dates.min()).isoformat()
                receipt["date_max"] = pd.Timestamp(dates.max()).isoformat()
                receipt["unique_date_count"] = int(dates.nunique(dropna=False))
            except Exception as exc:
                receipt["date_parse_error"] = repr(exc)
            y_numeric = pd.to_numeric(ydf["y"], errors="coerce")
            receipt["Y_missing_cells"] = int(ydf.isna().sum().sum())
            receipt["Y_nonfinite_y"] = int((~np.isfinite(y_numeric.to_numpy(dtype=np.float64))).sum())
            receipt["duplicate_unique_id_ds"] = int(
                ydf.duplicated(subset=["unique_id", "ds"]).sum()
            )
    if S_df is not None:
        receipt["S_shape"] = list(S_df.shape)
        receipt["S_index_count"] = int(len(S_df.index))
        receipt["S_bottom_column_count"] = int(len(S_df.columns))
    if tags is not None:
        receipt["tags_levels"] = [str(level) for level in tags.keys()]
        receipt["tags_counts"] = {
            str(level): int(np.asarray(values).size) for level, values in tags.items()
        }
    return _json_ready(receipt)


def _assert_or_raise(
    group: str,
    data_audit: dict[str, Any],
    hierarchy_result: dict[str, Any] | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> None:
    issues = data_audit.get("issues", [])
    if _issue_failed(issues):
        code = _first_failure_code(issues, "DATA_AUDIT_FAILED")
        data_audit["status"] = "FAIL"
        data_audit["blocker_code"] = code
        raise DataFailure(
            code,
            f"{group} data audit failed: {code}",
            {
                "data_audit": data_audit,
                "hierarchy_audit": hierarchy_result,
                "split_manifest": split_manifest,
            },
        )


def materialize_hierarchical_dataset(
    group: str,
    Y_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict[str, Any],
    *,
    receipt: dict[str, Any] | None = None,
    context: int = DEFAULT_CONTEXT,
    horizon: int = DEFAULT_HORIZON,
    tolerance: float = SUMMING_TOLERANCE,
    fail_on_duplicate_support: bool = False,
) -> HierarchicalDataset:
    issues: list[dict[str, Any]] = []
    data_audit: dict[str, Any] = {
        "group": group,
        "status": "PENDING",
        "created_at": utc_now(),
        "context": context,
        "horizon": horizon,
        "summing_tolerance": tolerance,
        "issues": issues,
    }

    ydf = _normalize_y(Y_df, issues)
    sdf = _normalize_s(S_df, issues)
    receipt_value = receipt or build_data_receipt(group, Y_df, S_df, tags, None)

    _assert_or_raise(group, data_audit)

    ids = tuple(str(value) for value in sdf.index)
    bottom_ids = tuple(str(value) for value in sdf.columns)
    S_values = sdf.to_numpy(dtype=np.float64)

    observed_order = tuple(ydf["unique_id"].drop_duplicates().astype(str).tolist())
    if observed_order != ids:
        issues.append({
            "severity": "FAIL",
            "code": "Y_ORDER_DOES_NOT_MATCH_S_INDEX",
            "observed_first": list(observed_order[:10]),
            "expected_first": list(ids[:10]),
            "observed_count": len(observed_order),
            "expected_count": len(ids),
        })

    duplicate_count = int(ydf.duplicated(subset=["unique_id", "ds"]).sum())
    if duplicate_count:
        issues.append({
            "severity": "FAIL",
            "code": "Y_DUPLICATE_UNIQUE_ID_DS",
            "count": duplicate_count,
        })

    missing_y = int(ydf["y"].isna().sum())
    nonfinite_y = int((~np.isfinite(ydf["y"].to_numpy(dtype=np.float64))).sum())
    if missing_y or nonfinite_y:
        issues.append({
            "severity": "FAIL",
            "code": "Y_MISSING_OR_NONFINITE",
            "missing_y": missing_y,
            "nonfinite_y": nonfinite_y,
        })

    unique_dates = pd.DatetimeIndex(sorted(ydf["ds"].drop_duplicates()))
    if len(unique_dates) == 0:
        issues.append({"severity": "FAIL", "code": "Y_NO_TIMESTAMPS"})
        _assert_or_raise(group, data_audit)
    expected_dates = pd.date_range(unique_dates.min(), unique_dates.max(), freq="MS")
    if len(unique_dates) != len(expected_dates) or not unique_dates.equals(expected_dates):
        issues.append({
            "severity": "FAIL",
            "code": "Y_NOT_REGULAR_MONTH_START_GRID",
            "date_min": unique_dates.min().isoformat(),
            "date_max": unique_dates.max().isoformat(),
            "unique_date_count": len(unique_dates),
            "expected_month_count": len(expected_dates),
        })

    id_set = set(ids)
    y_id_set = set(ydf["unique_id"].unique().tolist())
    if y_id_set != id_set:
        issues.append({
            "severity": "FAIL",
            "code": "Y_IDS_DO_NOT_MATCH_S_INDEX",
            "missing_in_Y_sample": sorted(id_set - y_id_set)[:20],
            "extra_in_Y_sample": sorted(y_id_set - id_set)[:20],
            "missing_in_Y_count": len(id_set - y_id_set),
            "extra_in_Y_count": len(y_id_set - id_set),
        })

    _assert_or_raise(group, data_audit)

    pivot = ydf.pivot(index="unique_id", columns="ds", values="y").reindex(
        index=list(ids),
        columns=unique_dates,
    )
    if pivot.isna().any().any():
        missing_cells = int(pivot.isna().sum().sum())
        issues.append({
            "severity": "FAIL",
            "code": "Y_INCOMPLETE_COMMON_TIMELINE",
            "missing_cells": missing_cells,
        })

    distance_to_binary = np.minimum(np.abs(S_values), np.abs(S_values - 1.0))
    max_binary_distance = float(distance_to_binary.max(initial=0.0))
    if max_binary_distance > tolerance:
        issues.append({
            "severity": "FAIL",
            "code": "S_NOT_BINARY",
            "max_binary_distance": max_binary_distance,
        })

    index_pos = {series_id: idx for idx, series_id in enumerate(ids)}
    missing_bottom_rows = [bottom_id for bottom_id in bottom_ids if bottom_id not in index_pos]
    if missing_bottom_rows:
        issues.append({
            "severity": "FAIL",
            "code": "S_BOTTOM_COLUMNS_NOT_PRESENT_AS_ROWS",
            "sample": missing_bottom_rows[:20],
            "count": len(missing_bottom_rows),
        })

    _assert_or_raise(group, data_audit)

    bottom_indices = np.asarray([index_pos[bottom_id] for bottom_id in bottom_ids], dtype=np.int64)
    bottom_identity_max_abs = 0.0
    for column_index, row_index in enumerate(bottom_indices):
        expected = np.zeros(len(bottom_ids), dtype=np.float64)
        expected[column_index] = 1.0
        bottom_identity_max_abs = max(
            bottom_identity_max_abs,
            float(np.max(np.abs(S_values[row_index] - expected))),
        )
    if bottom_identity_max_abs > tolerance:
        issues.append({
            "severity": "FAIL",
            "code": "S_BOTTOM_ROWS_NOT_IDENTITY",
            "max_abs_diff": bottom_identity_max_abs,
        })

    normalized_tags, tag_ids = _normalize_tags(tags, ids, issues)
    values = pivot.to_numpy(dtype=np.float64)
    bottom_values = values[bottom_indices, :]
    reconstructed = S_values @ bottom_values
    abs_diff = np.abs(reconstructed - values)
    finite_abs_diff = abs_diff[np.isfinite(abs_diff)]
    max_abs_diff = float(finite_abs_diff.max(initial=0.0)) if finite_abs_diff.size else None
    mismatch_count = int((abs_diff > tolerance).sum())
    if mismatch_count:
        worst_position = np.unravel_index(int(abs_diff.argmax()), abs_diff.shape)
        issues.append({
            "severity": "FAIL",
            "code": "BLOCKED_HIERARCHY_INCONSISTENT",
            "max_abs_diff": max_abs_diff,
            "mismatch_count": mismatch_count,
            "worst_node_index": int(worst_position[0]),
            "worst_node_id": ids[int(worst_position[0])],
            "worst_date": unique_dates[int(worst_position[1])].isoformat(),
        })

    graph_audit_value: dict[str, Any] | None = None
    split_manifest: dict[str, Any] | None = None
    graph: HierarchyGraph | None = None
    if not _issue_failed(issues):
        try:
            graph = build_support_inclusion_graph(sdf, tolerance=tolerance)
            graph_audit_value = hierarchy_audit(graph)
            graph_issues = validate_graph_contract(
                graph,
                fail_on_duplicate_support=fail_on_duplicate_support,
            )
            issues.extend(graph_issues)
            graph_audit_value["issues"] = graph_issues
            graph_audit_value["status"] = "FAIL" if _issue_failed(graph_issues) else "PASS"
        except Exception as exc:
            issues.append({
                "severity": "FAIL",
                "code": "HIERARCHY_GRAPH_BUILD_FAILED",
                "message": repr(exc),
            })
    if not _issue_failed(issues):
        try:
            split_manifest = split_origins(
                len(unique_dates),
                dates=unique_dates,
                context=context,
                horizon=horizon,
            )
        except DataFailure as exc:
            issues.append({"severity": "FAIL", "code": exc.code, "message": str(exc)})
            split_manifest = exc.artifacts.get("split_manifest")

    data_audit.update({
        "Y_shape": list(ydf.shape),
        "S_shape": list(sdf.shape),
        "unique_series_count": len(ids),
        "bottom_series_count": len(bottom_ids),
        "time_axis": {
            "date_min": unique_dates.min().isoformat(),
            "date_max": unique_dates.max().isoformat(),
            "n_times": len(unique_dates),
            "freq": "MS",
        },
        "missing_y": missing_y,
        "nonfinite_y": nonfinite_y,
        "duplicate_unique_id_ds": duplicate_count,
        "summing_consistency": {
            "status": "PASS" if mismatch_count == 0 else "FAIL",
            "tolerance": tolerance,
            "max_abs_diff": max_abs_diff,
            "mismatch_count": mismatch_count,
            "method": "S_df @ bottom_values compared with all node values at every timestamp",
        },
        "levels": {
            name: {
                "count": int(len(indices)),
                "first_ids": [ids[int(idx)] for idx in indices[:5]],
            }
            for name, indices in normalized_tags.items()
        },
        "tag_coverage_node_count": int(sum(len(indices) for indices in normalized_tags.values())),
    })

    if _issue_failed(issues):
        data_audit["status"] = "FAIL"
        data_audit["blocker_code"] = _first_failure_code(issues, "DATA_AUDIT_FAILED")
        raise DataFailure(
            str(data_audit["blocker_code"]),
            f"{group} data audit failed: {data_audit['blocker_code']}",
            {
                "data_audit": data_audit,
                "hierarchy_audit": graph_audit_value,
                "split_manifest": split_manifest,
            },
        )

    assert graph is not None
    assert graph_audit_value is not None
    assert split_manifest is not None
    data_audit["status"] = "PASS"
    data_audit["failure_type"] = None

    return HierarchicalDataset(
        group=group,
        Y_df=ydf,
        S_df=sdf,
        values=values,
        dates=unique_dates,
        ids=ids,
        S=S_values,
        tags=normalized_tags,
        tag_ids=tag_ids,
        bottom_ids=bottom_ids,
        bottom_indices=bottom_indices,
        graph=graph,
        parents=graph.parents,
        children=graph.children,
        receipt=receipt_value,
        data_audit=data_audit,
        hierarchy_audit=graph_audit_value,
        split_manifest=split_manifest,
    )


def origin_windows(
    origin: int,
    *,
    context: int = DEFAULT_CONTEXT,
    horizon: int = DEFAULT_HORIZON,
    n_times: int | None = None,
) -> dict[str, list[int]]:
    if origin < context:
        raise DataFailure(
            "ORIGIN_CONTEXT_UNAVAILABLE",
            f"Origin {origin} has fewer than {context} past context points",
        )
    target_end = origin + horizon
    if n_times is not None and target_end > n_times:
        raise DataFailure(
            "ORIGIN_TARGET_OUT_OF_RANGE",
            f"Origin {origin} horizon {horizon} exceeds n_times {n_times}",
        )
    return {
        "context_indices": list(range(origin - context, origin)),
        "target_indices": list(range(origin, target_end)),
    }


def _role_origins(
    role_start: int,
    role_end: int,
    *,
    context: int,
    horizon: int,
) -> list[int]:
    first = max(context, role_start)
    last = role_end - horizon
    if last < first:
        return []
    return list(range(first, last + 1))


def split_origins(
    n_times: int,
    *,
    dates: pd.DatetimeIndex | None = None,
    context: int = DEFAULT_CONTEXT,
    horizon: int = DEFAULT_HORIZON,
    calibration_months: int = 24,
    validation_months: int = 24,
    test_months: int = 48,
) -> dict[str, Any]:
    min_required = context + calibration_months + validation_months + test_months + horizon
    if n_times < min_required:
        manifest = {
            "status": "FAIL",
            "blocker_code": "TIME_AXIS_TOO_SHORT_FOR_SCREEN_SPLIT",
            "n_times": n_times,
            "min_required": min_required,
            "context": context,
            "horizon": horizon,
        }
        raise DataFailure(
            "TIME_AXIS_TOO_SHORT_FOR_SCREEN_SPLIT",
            f"Need at least {min_required} timestamps, found {n_times}",
            {"split_manifest": manifest},
        )

    test_start = n_times - test_months
    test_end = n_times
    validation_start = test_start - validation_months
    validation_end = test_start
    calibration_start = validation_start - calibration_months
    calibration_end = validation_start
    train_start = 0
    train_end = calibration_start

    role_bounds = {
        "TRAIN": (train_start, train_end),
        "CALIBRATION": (calibration_start, calibration_end),
        "VALIDATION": (validation_start, validation_end),
        "TEST": (test_start, test_end),
    }
    origins = {
        role: _role_origins(start, end, context=context, horizon=horizon)
        for role, (start, end) in role_bounds.items()
    }
    empty_roles = [role for role, role_origins in origins.items() if not role_origins]
    if empty_roles:
        manifest = {
            "status": "FAIL",
            "blocker_code": "SPLIT_ROLE_HAS_NO_LEGAL_ORIGINS",
            "empty_roles": empty_roles,
            "n_times": n_times,
            "role_bounds": role_bounds,
        }
        raise DataFailure(
            "SPLIT_ROLE_HAS_NO_LEGAL_ORIGINS",
            f"Split roles without legal origins: {empty_roles}",
            {"split_manifest": manifest},
        )

    target_owner: dict[int, str] = {}
    for role, role_origins in origins.items():
        start, end = role_bounds[role]
        for origin in role_origins:
            window = origin_windows(origin, context=context, horizon=horizon, n_times=n_times)
            target = window["target_indices"]
            if min(target) < start or max(target) >= end:
                raise DataFailure(
                    "SPLIT_TARGET_CROSSES_ROLE_BOUNDARY",
                    f"{role} origin {origin} target crosses [{start}, {end})",
                )
            if max(window["context_indices"]) >= origin:
                raise DataFailure(
                    "SPLIT_CONTEXT_USES_TARGET_OR_FUTURE",
                    f"{role} origin {origin} context is not strictly before target",
                )
            for target_index in target:
                previous = target_owner.setdefault(target_index, role)
                if previous != role:
                    raise DataFailure(
                        "SPLIT_TARGET_ROLE_OVERLAP",
                        f"Time index {target_index} belongs to {previous} and {role}",
                    )

    def bound_dates(start: int, end: int) -> dict[str, str] | None:
        if dates is None:
            return None
        return {
            "start": dates[start].isoformat(),
            "end_inclusive": dates[end - 1].isoformat(),
        }

    manifest = {
        "status": "PASS",
        "context": context,
        "horizon": horizon,
        "n_times": n_times,
        "role_bounds": {
            role: {
                "start": start,
                "end_exclusive": end,
                "dates": bound_dates(start, end),
            }
            for role, (start, end) in role_bounds.items()
        },
        "origins": origins,
        "origin_counts": {role: len(role_origins) for role, role_origins in origins.items()},
        "rule": "target horizon must be fully contained in role interval; context indices are strictly before origin",
        "test_selection_guard": "TEST origins are excluded from model, LR, checkpoint, and reconciliation selection",
        "calibration_guard": "CALIBRATION origins are reserved for residual covariance and reconciliation fitting",
    }
    return _json_ready(manifest)


def _write_dataset_artifacts(
    group: str,
    results_dir: Path,
    receipt: dict[str, Any] | None,
    data_audit: dict[str, Any] | None,
    hierarchy_result: dict[str, Any] | None,
    split_manifest: dict[str, Any] | None,
) -> None:
    group_dir = results_dir / "data" / group
    if receipt is not None:
        write_json(group_dir / "DATA_RECEIPT.json", _json_ready(receipt))
    if data_audit is not None:
        write_json(group_dir / "DATA_AUDIT.json", _json_ready(data_audit))
    if hierarchy_result is not None:
        write_json(group_dir / "HIERARCHY_AUDIT.json", _json_ready(hierarchy_result))
    if split_manifest is not None:
        write_json(group_dir / "SPLIT_MANIFEST.json", _json_ready(split_manifest))


def _update_aggregate_artifact(
    path: Path,
    group: str,
    key: str,
    value: dict[str, Any] | None,
    *,
    base: dict[str, Any] | None = None,
) -> None:
    if value is None:
        return
    aggregate = _read_json_object(path)
    if base:
        aggregate.update(base)
    aggregate.setdefault("created_at", utc_now())
    aggregate["updated_at"] = utc_now()
    aggregate.setdefault("datasets", {})
    aggregate["datasets"].setdefault(group, {})
    aggregate["datasets"][group][key] = _json_ready(value)
    write_json(path, aggregate)


def _write_aggregate_artifacts(
    group: str,
    results_dir: Path,
    receipt: dict[str, Any] | None,
    hierarchy_result: dict[str, Any] | None,
    split_manifest: dict[str, Any] | None,
) -> None:
    _update_aggregate_artifact(
        results_dir / "DATA_SOURCE_MANIFEST.json",
        group,
        "receipt",
        receipt,
        base={
            "official_source_url": OFFICIAL_SOURCE_URL,
            "datasetsforecast_reference_commit_from_master": REFERENCE_DATASETSFORECAST_COMMIT,
        },
    )
    _update_aggregate_artifact(
        results_dir / "HIERARCHY_AUDIT.json",
        group,
        "hierarchy_audit",
        hierarchy_result,
    )
    _update_aggregate_artifact(
        results_dir / "SPLIT_MANIFEST.json",
        group,
        "split_manifest",
        split_manifest,
    )


def load_and_audit(
    group: str,
    directory: str | Path = "data",
    *,
    results_dir: str | Path = RESULTS,
    cache: bool = False,
    write_artifacts: bool = True,
    context: int = DEFAULT_CONTEXT,
    horizon: int = DEFAULT_HORIZON,
    tolerance: float = SUMMING_TOLERANCE,
) -> HierarchicalDataset:
    results_path = Path(results_dir)
    receipt: dict[str, Any] | None = None
    data_audit: dict[str, Any] | None = None
    hierarchy_result: dict[str, Any] | None = None
    split_manifest: dict[str, Any] | None = None
    try:
        Y_df, S_df, tags, source = load_official_frames(group, directory, cache=cache)
        receipt = build_data_receipt(group, Y_df, S_df, tags, source)
        dataset = materialize_hierarchical_dataset(
            group,
            Y_df,
            S_df,
            tags,
            receipt=receipt,
            context=context,
            horizon=horizon,
            tolerance=tolerance,
        )
        data_audit = dataset.data_audit
        hierarchy_result = dataset.hierarchy_audit
        split_manifest = dataset.split_manifest
        if write_artifacts:
            _write_dataset_artifacts(
                group,
                results_path,
                receipt,
                data_audit,
                hierarchy_result,
                split_manifest,
            )
            _write_aggregate_artifacts(group, results_path, receipt, hierarchy_result, split_manifest)
        return dataset
    except DataFailure as exc:
        artifacts = exc.artifacts
        data_audit = artifacts.get("data_audit", data_audit)
        hierarchy_result = artifacts.get("hierarchy_audit", hierarchy_result)
        split_manifest = artifacts.get("split_manifest", split_manifest)
        if data_audit is None:
            data_audit = _minimal_failure_audit(group, exc.code, str(exc))
        if receipt is None:
            receipt = build_data_receipt(group, None, None, None, None)
        if write_artifacts:
            _write_dataset_artifacts(
                group,
                results_path,
                receipt,
                data_audit,
                hierarchy_result,
                split_manifest,
            )
            _write_aggregate_artifacts(group, results_path, receipt, hierarchy_result, split_manifest)
        raise


def audit_official_datasets(
    groups: Iterable[str] = ALLOWED_GROUPS,
    *,
    directory: str | Path = "data",
    results_dir: str | Path = RESULTS,
    cache: bool = False,
) -> dict[str, HierarchicalDataset]:
    datasets: dict[str, HierarchicalDataset] = {}
    for group in groups:
        datasets[group] = load_and_audit(
            group,
            directory=directory,
            results_dir=results_dir,
            cache=cache,
        )
    return datasets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and audit official Nixtla hierarchy data.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--groups", nargs="+", default=list(ALLOWED_GROUPS), choices=list(ALLOWED_GROUPS))
    parser.add_argument("--cache", action="store_true", help="Allow datasetsforecast pickle cache.")
    args = parser.parse_args(argv)

    for group in args.groups:
        dataset = load_and_audit(
            group,
            directory=args.data_dir,
            results_dir=args.results_dir,
            cache=args.cache,
        )
        print(
            f"{group}: PASS nodes={len(dataset.ids)} bottom={len(dataset.bottom_ids)} "
            f"times={len(dataset.dates)}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
