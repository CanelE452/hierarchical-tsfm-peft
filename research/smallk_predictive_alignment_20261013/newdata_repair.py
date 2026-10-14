"""Repair wrapper for Beijing UCI source selection.

The frozen ``newdata.py`` implementation is left unchanged.  This wrapper only
patches Beijing parsing to select the Aotizhongxin PRSA member by per-member
schema/station provenance, then re-exports the same public API used by the
new-data runner.
"""
from __future__ import annotations

from contextlib import contextmanager
import io
from pathlib import Path
from typing import Any

import numpy as np

import newdata as base

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPAIR_CACHE = ROOT / ".cache" / HERE.name / "beijing_repair"
BEIJING = "beijing_aotizhongxin"


@contextmanager
def patched_beijing_parser():
    original = base._parse_beijing
    base._parse_beijing = _parse_beijing_selected_member
    try:
        yield
    finally:
        base._parse_beijing = original


def _parse_beijing_selected_member(source_path: Path, spec: base.DatasetSpec):
    pd = base._load_pandas()
    selected_frames = []
    member_audit = []
    for name, payload in base._read_archive_members(source_path):
        frame = pd.read_csv(io.BytesIO(payload))
        columns = set(frame.columns)
        has_schema = {"year", "month", "day", "hour", "station"}.issubset(columns)
        has_channels = all(column in columns for column in spec.channels)
        stations = sorted(frame["station"].astype(str).dropna().unique().tolist()) if "station" in columns else []
        selected = bool(has_schema and has_channels and spec.station in stations)
        member_audit.append(
            {
                "member": name,
                "rows": int(len(frame)),
                "has_station": "station" in columns,
                "has_year_month_day_hour": {"year", "month", "day", "hour"}.issubset(columns),
                "has_all_required_channels": has_channels,
                "stations": stations,
                "selected_for_aotizhongxin": selected,
            }
        )
        if selected:
            subset = frame[frame["station"].astype(str) == spec.station].copy()
            if len(subset):
                selected_frames.append((name, subset))
    if not selected_frames:
        raise ValueError("No PRSA member with Aotizhongxin station and required Beijing schema")
    frame = pd.concat([item[1] for item in selected_frames], ignore_index=True)
    timestamps = pd.to_datetime(
        {"year": frame["year"], "month": frame["month"], "day": frame["day"], "hour": frame["hour"]},
        errors="raise",
    )
    order = np.argsort(timestamps.to_numpy())
    frame = frame.iloc[order].reset_index(drop=True)
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    values = frame[list(spec.channels)].apply(pd.to_numeric, errors="coerce")
    physical, mask, columns, times, meta = base._frame_to_arrays(
        values,
        timestamps,
        spec,
        ",".join(name for name, _frame in selected_frames),
    )
    meta.update(
        {
            "repair_parser": "select_per_member_station_schema_and_timestamp",
            "selected_member_count": len(selected_frames),
            "selected_members": [name for name, _frame in selected_frames],
            "member_audit": member_audit,
            "skipped_top_level_stock_csv": [
                item["member"]
                for item in member_audit
                if not item["has_station"] and not item["has_year_month_day_hour"]
            ],
        }
    )
    return physical, mask, columns, times, meta


def prepare(
    dataset: str,
    method_lock_path: str | Path,
    *,
    source_path: str | Path | None = None,
    allow_download: bool = False,
    cache_root: str | Path = REPAIR_CACHE,
) -> dict[str, Any]:
    if dataset != BEIJING:
        raise ValueError("newdata_repair.py is scoped to beijing_aotizhongxin only")
    with patched_beijing_parser():
        contract = base.prepare(
            dataset,
            method_lock_path,
            source_path=source_path,
            allow_download=allow_download,
            cache_root=cache_root,
        )
        _ensure_eval_past_physical(contract)
        return contract


def load_training(
    dataset: str,
    method_lock_path: str | Path,
    *,
    source_path: str | Path | None = None,
    allow_download: bool = False,
    cache_root: str | Path = REPAIR_CACHE,
) -> dict[str, Any]:
    if dataset != BEIJING:
        raise ValueError("newdata_repair.py is scoped to beijing_aotizhongxin only")
    with patched_beijing_parser():
        data = base.load_training(
            dataset,
            method_lock_path,
            source_path=source_path,
            allow_download=allow_download,
            cache_root=cache_root,
        )
        contract = get_contract(dataset, method_lock_path, cache_root=cache_root)
        _ensure_eval_past_physical(contract)
        return data


def required_model_selection_fields(
    dataset: str,
    method_lock_path: str | Path,
    *,
    cache_root: str | Path = REPAIR_CACHE,
) -> dict[str, Any]:
    if dataset != BEIJING:
        raise ValueError("newdata_repair.py is scoped to beijing_aotizhongxin only")
    with patched_beijing_parser():
        return base.required_model_selection_fields(dataset, method_lock_path, cache_root=cache_root)


def open_evaluation(
    dataset: str,
    model_selection_lock_path: str | Path,
    *,
    method_lock_path: str | Path | None = None,
    cache_root: str | Path = REPAIR_CACHE,
) -> dict[str, Any]:
    if dataset != BEIJING:
        raise ValueError("newdata_repair.py is scoped to beijing_aotizhongxin only")
    with patched_beijing_parser():
        return base.open_evaluation(
            dataset,
            model_selection_lock_path,
            method_lock_path=method_lock_path,
            cache_root=cache_root,
        )


def get_contract(
    dataset: str,
    method_lock_path: str | Path,
    *,
    cache_root: str | Path = REPAIR_CACHE,
) -> dict[str, Any]:
    if dataset != BEIJING:
        raise ValueError("newdata_repair.py is scoped to beijing_aotizhongxin only")
    with patched_beijing_parser():
        return base._contract_from_registry(dataset, method_lock_path, cache_root, include_eval_target=False)


def _repair_manifest_path(contract: dict[str, Any]) -> Path:
    return Path(contract["payloads"]["eval_past_input"]["path"]).with_name("repair_eval_past_input_manifest.json")


def _repair_physical_payload_path(contract: dict[str, Any]) -> Path:
    return Path(contract["payloads"]["eval_past_input"]["path"]).with_name("repair_eval_past_physical_payload.npz")


def _ensure_eval_past_physical(contract: dict[str, Any]) -> dict[str, Any]:
    manifest_path = _repair_manifest_path(contract)
    physical_payload_path = _repair_physical_payload_path(contract)
    if manifest_path.exists():
        manifest = base.read_json(manifest_path)
        if physical_payload_path.exists() and base.sha256_path(physical_payload_path) == manifest["history_physical_sha256"]:
            return manifest
        raise ValueError("Repair eval past physical manifest/hash mismatch")

    with patched_beijing_parser():
        physical, mask, columns, _times, parse_meta = base._parse_source(BEIJING, Path(contract["source"]["path"]))
    input_path = Path(contract["payloads"]["eval_past_input"]["path"])
    with np.load(input_path, allow_pickle=False) as z:
        origins = z["origins"].astype(np.int64)
        history_mask = z["history_mask"].astype(bool)
    history_physical = np.stack([physical[o - base.L : o] for o in origins]).astype(np.float32)
    expected_mask = np.stack([mask[o - base.L : o] for o in origins]).astype(bool)
    if not np.array_equal(history_mask, expected_mask):
        raise ValueError("Repair physical history mask does not match eval past input mask")
    with physical_payload_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            history_physical=history_physical,
            history_mask=history_mask,
            origins=origins,
            columns=np.array(columns),
        )
    manifest = {
        "dataset": BEIJING,
        "payload": str(physical_payload_path.resolve()),
        "history_physical_sha256": base.sha256_path(physical_payload_path),
        "history_physical_array_sha256": base.sha256_array(history_physical),
        "origins_sha256": base.sha256_array(origins),
        "source_sha256": contract["source"]["sha256"],
        "source_member": parse_meta["source_member"],
        "no_eval_future_target_payload_opened": True,
    }
    base._write_json_once(manifest_path, manifest)
    return manifest


def load_evaluation_inputs(
    dataset: str,
    method_lock_path: str | Path,
    *,
    cache_root: str | Path = REPAIR_CACHE,
) -> dict[str, Any]:
    if dataset != BEIJING:
        raise ValueError("newdata_repair.py is scoped to beijing_aotizhongxin only")
    contract = get_contract(dataset, method_lock_path, cache_root=cache_root)
    manifest = _ensure_eval_past_physical(contract)
    input_path = Path(contract["payloads"]["eval_past_input"]["path"])
    physical_path = Path(manifest["payload"])
    with np.load(input_path, allow_pickle=False) as z, np.load(physical_path, allow_pickle=False) as p:
        origins = z["origins"].astype(np.int64)
        physical_origins = p["origins"].astype(np.int64)
        if not np.array_equal(origins, physical_origins):
            raise ValueError("Repair eval past input origins mismatch")
        return {
            "dataset": dataset,
            "history": z["history"].astype(np.float32),
            "history_observed": z["history_mask"].astype(bool),
            "history_physical": p["history_physical"].astype(np.float32),
            "origins": origins,
            "mean": z["mean"].astype(np.float32),
            "std": z["std"].astype(np.float32),
            "columns": [str(value) for value in z["columns"].tolist()],
            "contract_sha256": contract["contract_sha256"],
            "eval_input_payload_sha256": contract["payloads"]["eval_past_input"]["sha256"],
            "repair_history_physical_payload_sha256": manifest["history_physical_sha256"],
        }


def repair_metadata() -> dict[str, Any]:
    return {
        "dataset": BEIJING,
        "repair_kind": "select_per_member_aotizhongxin_prsa_schema",
        "repair_cache": str(REPAIR_CACHE),
        "public_api": [
            "load_training",
            "required_model_selection_fields",
            "open_evaluation",
            "get_contract",
            "load_evaluation_inputs",
        ],
        "frozen_newdata_modified": False,
    }

