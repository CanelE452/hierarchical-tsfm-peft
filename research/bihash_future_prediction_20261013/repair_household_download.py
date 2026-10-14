"""Repair the UCI 235 raw download after the METHOD_LOCK URL spelling mismatch.

This wrapper is intentionally separate from the frozen campaign sources.  It
does not change METHOD_LOCK, FRESH_SOURCE_PLAN, download_new.py, data.py, or any
fresh payload.  It only proves that the locked ``+`` URL and the downloader's
``%2B`` URL address the same UCI static path, then runs the existing downloader
with a process-local URL spelling override.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
import argparse
import json
import os
import time
import zipfile

import pandas as pd

import data
import download_new
import runtime as rt


DATASET = "uci235_household_hourly"
SML_DATASET = "uci274_sml2010"
HERE = Path(__file__).resolve().parent
REPAIR_DIR = HERE / "ingest_repairs_v1"
FROZEN_RELATIVE_PATHS = (
    "research/bihash_future_prediction_20261013/download_new.py",
    "research/bihash_future_prediction_20261013/data.py",
    "research/bihash_future_prediction_20261013/run.py",
    "research/bihash_future_prediction_20261013/runtime.py",
    "research/bihash_future_prediction_20261013/METHOD_LOCK.json",
    "research/bihash_future_prediction_20261013/FRESH_SOURCE_PLAN.json",
)


def _jsonable(value: Any) -> Any:
    return rt.jsonable(value)


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_relative_to(HERE):
        raise ValueError(f"Refusing to write outside repair study folder: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return {"path": path.relative_to(rt.ROOT).as_posix(), "sha256": rt.sha(path), "bytes": path.stat().st_size}


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(rt.ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _split_without_plus_decoding(url: str) -> dict[str, str]:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc or not parts.path:
        raise ValueError(f"URL is incomplete: {url}")
    if parts.query or parts.fragment:
        raise ValueError(f"URL must not carry query/fragment components: {url}")
    return {
        "scheme": parts.scheme.lower(),
        "netloc": parts.netloc.lower(),
        "raw_path": parts.path,
        "decoded_path": unquote(parts.path, errors="strict"),
    }


def prove_same_static_resource(expected_url: str, locked_url: str) -> dict[str, Any]:
    expected = _split_without_plus_decoding(expected_url)
    locked = _split_without_plus_decoding(locked_url)
    same = (
        expected["scheme"] == locked["scheme"]
        and expected["netloc"] == locked["netloc"]
        and expected["decoded_path"] == locked["decoded_path"]
    )
    if not same:
        raise ValueError(
            "Locked URL is not the same UCI static resource under path percent-decoding: "
            f"expected={expected_url!r}, locked={locked_url!r}"
        )
    if " " in locked["decoded_path"]:
        raise ValueError("Decoded static path contains a space; '+' must not be treated as a form-space")
    return {
        "status": "SAME_STATIC_RESOURCE",
        "normalization": "urllib.parse.unquote on URL path only; unquote_plus is intentionally not used",
        "expected": expected,
        "locked": locked,
    }


def _source_hashes() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for rel in FROZEN_RELATIVE_PATHS:
        path = rt.ROOT / rel
        result[rel] = {"sha256": rt.sha(path), "bytes": path.stat().st_size}
    return result


@contextmanager
def _process_local_download_url(dataset: str, locked_url: str):
    ds = download_new.canonical(dataset)
    original = download_new.DATASETS[ds]["download_url"]
    try:
        download_new.DATASETS[ds]["download_url"] = locked_url
        yield
    finally:
        download_new.DATASETS[ds]["download_url"] = original


def _method_lock_source(method_lock: Path, dataset: str) -> dict[str, Any]:
    lock = _read_json(method_lock)
    if not lock.get("locked"):
        raise ValueError("METHOD_LOCK must be locked before repairing raw ingest")
    sources = lock.get("newdata_sources") or lock.get("raw_sources") or lock.get("sources") or {}
    entry = sources.get(dataset)
    if not isinstance(entry, dict):
        raise ValueError(f"METHOD_LOCK source entry for {dataset} must be a dict")
    path = entry.get("path")
    url = entry.get("url") or entry.get("download_url")
    if not path or not url:
        raise ValueError(f"METHOD_LOCK source entry for {dataset} needs path and url")
    final = Path(path)
    if not final.is_absolute():
        final = rt.ROOT / final
    expected_parent = download_new.raw_dir(dataset).resolve()
    if final.resolve().parent != expected_parent:
        raise ValueError(f"Locked source path is outside raw/{dataset}: {final}")
    return {"path": final.resolve(), "url": str(url)}


def repair_household_download(method_lock: Path, *, timeout: float = 180.0) -> dict[str, Any]:
    method_lock = method_lock.resolve()
    before = _source_hashes()
    source = _method_lock_source(method_lock, DATASET)
    expected_url = download_new.DATASETS[DATASET]["download_url"]
    equivalence = prove_same_static_resource(expected_url, source["url"])
    with _process_local_download_url(DATASET, source["url"]):
        receipt = download_new.download_one(DATASET, method_lock_path=method_lock, timeout=timeout)
    after = _source_hashes()
    if before != after:
        raise RuntimeError("Frozen source files changed during repair download")
    artifact = receipt.get("artifact")
    if not artifact:
        raise RuntimeError(f"Repair did not produce a verified raw artifact: {receipt}")
    return {
        "status": "COMPLETE",
        "dataset": DATASET,
        "scope": "raw UCI 235 download repair only; no TEST access, no EVAL target open, no frozen-source edit",
        "method_lock": {"path": _display_path(method_lock), "sha256": rt.sha(method_lock)},
        "url_equivalence": equivalence,
        "download_receipt": receipt,
        "frozen_source_hashes_before": before,
        "frozen_source_hashes_after": after,
        "frozen_sources_unchanged": before == after,
    }


def _read_sml_raw_dates(source_zip: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(source_zip) as archive:
        for member in ("NEW-DATA-1.T15.txt", "NEW-DATA-2.T15.txt"):
            with archive.open(member) as handle:
                frame = pd.read_csv(handle, comment="#", header=None, sep=r"[;,\s]+", engine="python")
            frame = frame.dropna(axis=1, how="all")
            if frame.shape[1] != 24:
                raise ValueError(f"{member} expected 24 columns, got {frame.shape[1]}")
            raw = frame[0].astype(str) + " " + frame[1].astype(str)
            exact = pd.to_datetime(raw, format="%d/%m/%Y %H:%M", utc=True, errors="raise")
            inferred = pd.to_datetime(raw, utc=True, errors="raise")
            if not bool((exact == inferred).all()):
                mismatch = int((exact != inferred).to_numpy().nonzero()[0][0])
                raise ValueError(
                    f"SML inferred timestamp differs from exact dd/mm/yyyy parse in {member} at row {mismatch}: "
                    f"raw={raw.iloc[mismatch]!r}, exact={exact.iloc[mismatch]}, inferred={inferred.iloc[mismatch]}"
                )
            records.append({
                "member": member,
                "rows": int(len(frame)),
                "first_raw_timestamp": raw.iloc[0],
                "last_raw_timestamp": raw.iloc[-1],
                "exact_parse_first_utc": exact.iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "exact_parse_last_utc": exact.iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "inferred_matches_exact_dd_mm_yyyy": True,
            })
    return records


def audit_sml_timestamp_warning(method_lock: Path) -> dict[str, Any]:
    source = _method_lock_source(method_lock.resolve(), SML_DATASET)
    source_zip = source["path"]
    if not source_zip.is_file():
        raise FileNotFoundError(f"SML raw source missing: {source_zip}")
    return {
        "status": "COMPLETE",
        "dataset": SML_DATASET,
        "scope": "metadata/raw timestamp audit only; no payload mutation",
        "source": {
            "path": _display_path(source_zip),
            "sha256": rt.sha(source_zip),
            "bytes": source_zip.stat().st_size,
        },
        "timestamp_checks": _read_sml_raw_dates(source_zip),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-lock", default=str(HERE / "METHOD_LOCK.json"))
    parser.add_argument("--tag", default="household_download_v1")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    start = time.time()
    terminal_path = REPAIR_DIR / f"{args.tag}_TERMINAL.json"
    payload: dict[str, Any] = {
        "phase": "ingest_repair",
        "tag": args.tag,
        "pid": os.getpid(),
        "started_unix": start,
    }
    status = "FAILED_PRESERVED"
    try:
        method_lock = Path(args.method_lock)
        payload["household_download"] = repair_household_download(method_lock, timeout=args.timeout)
        payload["sml_timestamp_warning_audit"] = audit_sml_timestamp_warning(method_lock)
        status = "COMPLETE"
    except Exception as error:
        payload["failure_reason"] = repr(error)
        payload["failure_type"] = type(error).__name__
    payload["status"] = status
    payload["finished_unix"] = time.time()
    payload["wall_seconds"] = payload["finished_unix"] - start
    artifact = _write_exclusive_json(terminal_path, payload)
    print(rt.json_line({"terminal": artifact, "status": status, "wall_seconds": payload["wall_seconds"]}), flush=True)
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
