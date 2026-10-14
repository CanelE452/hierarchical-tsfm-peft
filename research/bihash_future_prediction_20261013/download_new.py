"""Download and materialize the two fresh UCI datasets after METHOD_LOCK.

This file is a CLI utility only.  Importing it does not download, prepare, or
open EVAL targets.  The intended sequence is:

1. root creates METHOD_LOCK and explicitly starts fresh ingest;
2. run `download` to stream official UCI zip files into `.cache/.../raw/<ds>`;
3. run `prepare` to call data.prepare_new for both datasets;
4. keep EVAL target closed until the joint ALL_SELECTED model lock.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import os
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

import data
import runtime as rt


DATASETS: dict[str, dict[str, Any]] = {
    "uci235_household_hourly": {
        "aliases": ("uci235", "household", "household_power"),
        "official_page": "https://archive.ics.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption",
        "download_url": "https://archive.ics.uci.edu/static/public/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip",
        "doi": "10.24432/C58K54",
        "license": "CC BY 4.0",
        "official_download_label": "Download (19.7 MB)",
        "filename": "uci235_individual_household_electric_power_consumption.zip",
        "required_members": ("household_power_consumption.txt",),
        "notes": [
            "Official page reports 2,075,259 one-minute rows, 9 variables and missing values.",
            "Parser later excludes Date/Time and aggregates the 7 numeric channels to hourly observed means.",
        ],
    },
    "uci274_sml2010": {
        "aliases": ("uci274", "sml2010", "sml"),
        "official_page": "https://archive.ics.uci.edu/dataset/274/sml2010",
        "download_url": "https://archive.ics.uci.edu/static/public/274/sml2010.zip",
        "doi": "10.24432/C5RS3S",
        "license": "CC BY 4.0",
        "official_download_label": "Download (202.1 KB)",
        "filename": "uci274_sml2010.zip",
        "required_members": ("NEW-DATA-1.T15.txt", "NEW-DATA-2.T15.txt"),
        "notes": [
            "Official page reports 4,137 instances, missing values possible, UTC time and 15-minute smoothed uploads.",
            "Parser later excludes Date, Time and Day_of_week, leaving C=21 channels.",
        ],
    },
}


def canonical(name: str) -> str:
    key = str(name).strip().lower()
    for dataset, spec in DATASETS.items():
        if key == dataset or key in spec["aliases"]:
            return dataset
    raise ValueError(f"Unknown fresh dataset: {name}")


def selected(names: list[str]) -> list[str]:
    if not names or names == ["all"]:
        return list(DATASETS)
    return [canonical(name) for name in names]


def raw_dir(dataset: str, cache_root: Path | None = None) -> Path:
    base = Path(cache_root or rt.CACHE).resolve()
    return base / "raw" / canonical(dataset)


def artifact_path(dataset: str, cache_root: Path | None = None) -> Path:
    return raw_dir(dataset, cache_root) / DATASETS[canonical(dataset)]["filename"]


def declared_source(dataset: str, method_lock_path: Path | None = None, cache_root: Path | None = None) -> tuple[Path, str]:
    ds = canonical(dataset)
    spec = DATASETS[ds]
    if method_lock_path is None:
        return artifact_path(ds, cache_root), spec["download_url"]
    lock = data.read_json(method_lock_path)
    sources = lock.get("newdata_sources") or lock.get("raw_sources") or lock.get("sources") or {}
    entry = sources.get(ds) or sources.get(dataset)
    if entry is None:
        raise FileNotFoundError(f"METHOD_LOCK missing newdata_sources entry for {ds}")
    if isinstance(entry, str):
        declared_path, declared_url = entry, spec["download_url"]
    elif isinstance(entry, dict):
        declared_path = entry.get("path") or entry.get("source_path") or entry.get("raw_path")
        declared_url = entry.get("url") or entry.get("download_url") or spec["download_url"]
    else:
        raise TypeError(f"Unsupported source declaration for {ds}: {type(entry)!r}")
    if not declared_path:
        raise ValueError(f"METHOD_LOCK source declaration missing path for {ds}")
    if declared_url != spec["download_url"]:
        raise ValueError(f"METHOD_LOCK URL differs from official UCI static URL for {ds}")
    path = Path(declared_path)
    if not path.is_absolute():
        path = rt.ROOT / path
    expected_parent = raw_dir(ds, cache_root).resolve()
    path = path.resolve()
    if path.parent != expected_parent:
        raise ValueError(f"Declared source path must be under raw/{ds}: {path}")
    return path, declared_url


def jsonable(value: Any) -> Any:
    return rt.jsonable(value)


def write_unique_json(folder: Path, prefix: str, payload: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for attempt in range(1000):
        path = folder / f"{prefix}_{time.time_ns()}_{os.getpid()}_{attempt}.json"
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(jsonable(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create unique receipt in {folder}")


def zip_members(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        return [
            {
                "name": info.filename,
                "file_size": int(info.file_size),
                "compress_size": int(info.compress_size),
                "crc": int(info.CRC),
            }
            for info in archive.infolist()
            if not info.is_dir()
        ]


def verify_zip(path: Path, dataset: str) -> dict[str, Any]:
    spec = DATASETS[canonical(dataset)]
    members = zip_members(path)
    names = [Path(m["name"]).name for m in members]
    missing = [name for name in spec["required_members"] if name not in names]
    if missing:
        raise ValueError(f"Downloaded zip missing required members for {dataset}: {missing}")
    return {
        "path": path.resolve().relative_to(rt.ROOT).as_posix(),
        "sha256": rt.sha(path),
        "bytes": path.stat().st_size,
        "zip_members": members,
        "required_members_present": list(spec["required_members"]),
    }


def reuse_receipt(dataset: str, path: Path, cache_root: Path | None = None) -> dict[str, Any]:
    spec = DATASETS[canonical(dataset)]
    artifact = verify_zip(path, dataset)
    return {
        "status": "RAW_ARTIFACT_REUSED_READONLY",
        "dataset": canonical(dataset),
        "created_unix": time.time(),
        "official": {
            "page": spec["official_page"],
            "download_url": spec["download_url"],
            "doi": spec["doi"],
            "license": spec["license"],
            "download_label": spec["official_download_label"],
            "primary_source_verified_by_web": True,
        },
        "artifact": artifact,
        "receipt_scope": "raw download receipt only; no TEST access and no EVAL target open",
        "protected_TEST": "not applicable to fresh UCI raw download; legacy protected TEST remains untouched",
        "exposed_EVAL_ingestion": "raw may contain future rows but EVAL target stays in separate payload and is not opened here",
        "notes": spec["notes"],
    }


def download_one(
    dataset: str,
    cache_root: Path | None = None,
    timeout: float = 120.0,
    method_lock_path: Path | None = None,
) -> dict[str, Any]:
    ds = canonical(dataset)
    spec = DATASETS[ds]
    folder = raw_dir(ds, cache_root)
    folder.mkdir(parents=True, exist_ok=True)
    final, url = declared_source(ds, method_lock_path, cache_root)
    if final.exists():
        receipt = reuse_receipt(ds, final, cache_root)
        receipt_path = write_unique_json(folder, "receipt_reuse", receipt)
        return {**receipt, "receipt_path": str(receipt_path.resolve().relative_to(rt.ROOT))}

    attempt = folder / f"{final.stem}.partial.{time.time_ns()}_{os.getpid()}.zip"
    started = time.time()
    receipt: dict[str, Any] = {
        "status": "RUNNING",
        "dataset": ds,
        "created_unix": started,
        "official": {
            "page": spec["official_page"],
            "download_url": url,
            "doi": spec["doi"],
            "license": spec["license"],
            "download_label": spec["official_download_label"],
            "primary_source_verified_by_web": True,
        },
        "attempt_path": attempt.resolve().relative_to(rt.ROOT).as_posix(),
        "final_path": final.resolve().relative_to(rt.ROOT).as_posix(),
        "receipt_scope": "raw download receipt only; no TEST access and no EVAL target open",
        "protected_TEST": "not applicable to fresh UCI raw download; legacy protected TEST remains untouched",
        "exposed_EVAL_ingestion": "raw may contain future rows but EVAL target stays in separate payload and is not opened here",
        "notes": spec["notes"],
    }
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "hierarchical-tsfm-peft/uci-fresh-ingest"})
        http_start = time.time()
        with urllib.request.urlopen(request, timeout=timeout) as response, attempt.open("xb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
            receipt["http"] = {
                "status": getattr(response, "status", None),
                "elapsed_seconds": time.time() - http_start,
                "headers": {
                    "content-length": response.headers.get("Content-Length"),
                    "content-type": response.headers.get("Content-Type"),
                    "last-modified": response.headers.get("Last-Modified"),
                    "etag": response.headers.get("ETag"),
                },
            }
        if final.exists():
            raise FileExistsError(f"Declared source appeared during download; refusing overwrite: {final}")
        attempt.rename(final)
        receipt["status"] = "DOWNLOADED"
        receipt["artifact"] = verify_zip(final, ds)
    except Exception as error:
        receipt["status"] = "FAILED_PRESERVED"
        receipt["failure_reason"] = repr(error)
        if attempt.exists():
            receipt["partial_artifact"] = {
                "path": attempt.resolve().relative_to(rt.ROOT).as_posix(),
                "sha256": rt.sha(attempt),
                "bytes": attempt.stat().st_size,
            }
        receipt_path = write_unique_json(folder, "receipt_failed", receipt)
        return {**receipt, "receipt_path": str(receipt_path.resolve().relative_to(rt.ROOT))}
    receipt["finished_unix"] = time.time()
    receipt["wall_seconds"] = receipt["finished_unix"] - started
    receipt_path = write_unique_json(folder, "receipt_download", receipt)
    return {**receipt, "receipt_path": str(receipt_path.resolve().relative_to(rt.ROOT))}


def raw_source_for(dataset: str, cache_root: Path | None = None, method_lock_path: Path | None = None) -> Path:
    path, _url = declared_source(dataset, method_lock_path, cache_root)
    if not path.exists():
        raise FileNotFoundError(f"Raw artifact missing for {canonical(dataset)}: {path}")
    verify_zip(path, dataset)
    return path


def summarize_contract(contract: dict[str, Any], ready: dict[str, Any]) -> dict[str, Any]:
    dataset = contract["dataset"]
    loaded, history, target, observed, history_observed = data.load_training_new(
        dataset,
        ready["method_lock_path"],
        cache_root=Path(ready["cache_root"]),
    )
    stats = contract["statistics"]
    return {
        "dataset": dataset,
        "status": "MATERIALIZED_READY_EVAL_TARGET_CLOSED",
        "contract_sha256": rt.sha(Path(ready["contract_path"])),
        "C": loaded["C"],
        "K": loaded["K"],
        "columns": loaded["columns"],
        "parser_metadata": contract["parser_metadata"],
        "bounds": contract["bounds"],
        "origin_counts": {split: int(len(origins)) for split, origins in loaded["origins"].items()},
        "history_shapes": {split: list(value.shape) for split, value in history.items()},
        "target_shapes": {split: list(value.shape) for split, value in target.items()},
        "future_mask_observed_cells": {split: int(value.sum()) for split, value in observed.items()},
        "history_mask_observed_cells": {split: int(value.sum()) for split, value in history_observed.items()},
        "statistics": {
            "scope": stats["scope"],
            "mean_sha256": stats["mean_sha256"],
            "std_sha256": stats["std_sha256"],
            "mean_finite": bool(np_all_finite(loaded["mean"])),
            "std_positive": bool((loaded["std"] > 0).all()),
        },
        "pca": {
            "basis_shape": list(loaded["basis"].shape),
            "basis_sha256": contract["pca_initialization"]["basis_sha256"],
            "pca_mean_sha256": contract["pca_initialization"]["pca_mean_sha256"],
            "source": contract["pca_initialization"]["note"],
        },
        "bank": {
            "helper_bank_entries": int(len(loaded["bank"]["bank_origins"])),
            "helper_bank_origin_sha256": data.array_hash(loaded["bank"]["bank_origins"]),
            "runner_note": "Core runner may rebuild bank128 directly from FIT256 origins; helper bank is deterministic uniform subset.",
        },
        "eval_payloads": {
            "past_input_sha256": contract["payloads"]["eval_past_input"]["sha256"],
            "future_target_sha256_declared_not_opened_by_ready": contract["payloads"]["eval_future_target"]["sha256"],
            "note": "Ready report does not call open_new_evaluation.",
        },
    }


def np_all_finite(value: Any) -> bool:
    return bool(np.isfinite(np.asarray(value)).all())


def prepare_one(dataset: str, method_lock_path: Path, cache_root: Path | None = None) -> dict[str, Any]:
    ds = canonical(dataset)
    source = raw_source_for(ds, cache_root, method_lock_path)
    contract = data.prepare_new(ds, method_lock_path, source_path=source, cache_root=cache_root)
    method_hash = rt.sha(method_lock_path)
    contract_path = Path(cache_root or rt.CACHE).resolve() / ds / method_hash / "contract.json"
    ready_seed = {
        "method_lock_path": str(method_lock_path.resolve()),
        "cache_root": str(Path(cache_root or rt.CACHE).resolve()),
        "contract_path": str(contract_path.resolve()),
    }
    summary = summarize_contract(contract, ready_seed)
    folder = raw_dir(ds, cache_root)
    ready_path = write_unique_json(folder, "materialized_ready", summary)
    return {**summary, "ready_report_path": str(ready_path.resolve().relative_to(rt.ROOT))}


def download_command(args: argparse.Namespace) -> int:
    limit = rt.Limit("fresh_raw_download", gpu=False)
    results = []
    try:
        method_lock = Path(args.method_lock).resolve() if args.method_lock else None
        for ds in selected(args.datasets):
            try:
                results.append(download_one(
                    ds,
                    cache_root=Path(args.cache_root) if args.cache_root else None,
                    timeout=args.timeout,
                    method_lock_path=method_lock,
                ))
            except Exception as error:
                folder = raw_dir(ds, Path(args.cache_root) if args.cache_root else None)
                payload = {"status": "FAILED_PRESERVED", "dataset": ds, "failure_reason": repr(error), "created_unix": time.time()}
                path = write_unique_json(folder, "receipt_exception", payload)
                results.append({**payload, "receipt_path": str(path.resolve().relative_to(rt.ROOT))})
            limit.check()
        print(rt.json_line({"download_results": results}), flush=True)
        limit.close("COMPLETE" if all(r["status"] in {"DOWNLOADED", "RAW_ARTIFACT_REUSED_READONLY"} for r in results) else "FAILED_PRESERVED")
        return 0 if all(r["status"] in {"DOWNLOADED", "RAW_ARTIFACT_REUSED_READONLY"} for r in results) else 1
    except Exception as error:
        limit.close("FAILED_PRESERVED", error)
        raise


def prepare_command(args: argparse.Namespace) -> int:
    limit = rt.Limit("fresh_prepare_ready", gpu=False)
    try:
        method_lock = Path(args.method_lock).resolve()
        results = []
        for ds in selected(args.datasets):
            try:
                results.append(prepare_one(ds, method_lock, cache_root=Path(args.cache_root) if args.cache_root else None))
            except Exception as error:
                results.append({"status": "FAILED_PRESERVED", "dataset": ds, "failure_reason": repr(error)})
            limit.check()
        print(rt.json_line({"prepare_results": results}), flush=True)
        limit.close("COMPLETE" if all(r["status"] == "MATERIALIZED_READY_EVAL_TARGET_CLOSED" for r in results) else "FAILED_PRESERVED")
        return 0 if all(r["status"] == "MATERIALIZED_READY_EVAL_TARGET_CLOSED" for r in results) else 1
    except Exception as error:
        limit.close("FAILED_PRESERVED", error)
        raise


def inspect_command(args: argparse.Namespace) -> int:
    payload = {}
    method_lock = Path(args.method_lock).resolve() if getattr(args, "method_lock", None) else None
    for ds in selected(args.datasets):
        path, _url = declared_source(ds, method_lock, Path(args.cache_root) if args.cache_root else None)
        payload[ds] = reuse_receipt(ds, path, Path(args.cache_root) if args.cache_root else None) if path.exists() else {
            "status": "RAW_ARTIFACT_MISSING",
            "dataset": ds,
            "expected_path": str(path.resolve()),
            "official": DATASETS[ds],
        }
    print(rt.json_line(payload), flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default=None, help="Override cache root; default is runtime.CACHE.")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("inspect", "download"):
        s = sub.add_parser(name)
        s.add_argument("datasets", nargs="*", default=["all"], help="all, uci235, uci274, household, sml2010")
        s.add_argument("--method-lock", default=None, help="Use METHOD_LOCK newdata_sources declared path/url.")
        if name == "download":
            s.add_argument("--timeout", type=float, default=120.0)
    s = sub.add_parser("prepare")
    s.add_argument("--method-lock", required=True)
    s.add_argument("datasets", nargs="*", default=["all"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "inspect":
        return inspect_command(args)
    if args.command == "download":
        return download_command(args)
    if args.command == "prepare":
        return prepare_command(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
