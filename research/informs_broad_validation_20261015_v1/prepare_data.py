"""Acquire pinned public research assets into the private campaign cache only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np

ROOT = Path(os.environ.get("INFORMS_ASSET_ROOT", str(Path(__file__).resolve().parents[2])))
CACHE = Path(os.environ.get("INFORMS_CACHE_ROOT", str(ROOT / ".cache/informs_broad_validation_20261015_v1")))
SOURCES = {
    "electricity": {
        "url": "https://zenodo.org/api/records/3889830/files/electricity_hourly_dataset.ts/content",
        "filename": "electricity_hourly_dataset.ts", "bytes": 35932338,
        "md5": "142c94e21026db3c15c2d5c5ca7e2af9", "license": "CC-BY-4.0",
    },
    "pems08": {
        "url": "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/9c2e19b98c4cedf1f35214d8789685b6381b3aad/data/PEMS08/PEMS08.npz",
        "filename": "PEMS08.npz", "bytes": 18525419,
        "git_blob_sha1": "f5f3e84a00417573aa8487fe1e9e8cd13c6026c8",
        "license": "AUTHOR_RESEARCH_DISTRIBUTION_FORMAL_LICENSE_UNSPECIFIED",
    },
}


def digest(path, algorithm="sha256", prefix=b""):
    value = hashlib.new(algorithm)
    value.update(prefix)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def acquire(name):
    spec = SOURCES[name]
    directory = CACHE / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec["filename"]
    started = time.time()
    downloaded = 0
    if not target.exists():
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "tsfm-research-reproducibility/1.0"})
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("xb") as output:
            while block := response.read(1024 * 1024):
                downloaded += len(block)
                if downloaded > spec["bytes"]:
                    raise ValueError("Source exceeded pinned size; retained partial for inspection")
                output.write(block)
        if temporary.stat().st_size != spec["bytes"]:
            raise ValueError("Source download is shorter than pinned size")
        temporary.rename(target)
    if target.stat().st_size != spec["bytes"]:
        raise ValueError("Pinned source size mismatch")
    actual = {"sha256": digest(target), "md5": digest(target, "md5")}
    if "md5" in spec and actual["md5"] != spec["md5"]:
        raise ValueError("Source MD5 mismatch")
    if "git_blob_sha1" in spec:
        actual["git_blob_sha1"] = digest(target, "sha1", f"blob {target.stat().st_size}\0".encode())
        if actual["git_blob_sha1"] != spec["git_blob_sha1"]:
            raise ValueError("Pinned Git blob SHA1 mismatch")
    receipt = {"name": name, **spec, **actual, "download_bytes_this_invocation": downloaded,
               "started_unix": started, "completed_unix": time.time(),
               "raw_publication_allowed": False, "cache_reference": f"sources/{target.name}"}
    receipt_path = directory / f"{name}_download_receipt.json"
    if not receipt_path.exists():
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return target


def _write_json_exclusive(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, ensure_ascii=False)


def fetch_source_evidence():
    sources = {
        "ASTGCN_README": "https://raw.githubusercontent.com/guoshnBJTU/ASTGCN-r-pytorch/2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296/README.md",
        "ASTGCN_prepareData": "https://raw.githubusercontent.com/guoshnBJTU/ASTGCN-r-pytorch/2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296/prepareData.py",
        "ASTGCN_PEMS08_config": "https://raw.githubusercontent.com/guoshnBJTU/ASTGCN-r-pytorch/2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296/configurations/PEMS08_astgcn.conf",
        "ELECTRICITY_ZENODO_METADATA": "https://zenodo.org/api/records/3889830",
    }
    receipts = {}
    for name, url in sources.items():
        path = CACHE / "sources" / f"{name}.txt"
        downloaded = 0
        if not path.exists():
            body = urllib.request.urlopen(url, timeout=60).read()
            downloaded = len(body)
            with path.open("xb") as stream:
                stream.write(body)
        receipts[name] = {"url": url, "sha256": digest(path), "bytes": path.stat().st_size,
                          "download_bytes_this_invocation": downloaded, "cache_reference": f"sources/{path.name}"}
    target = CACHE / "sources" / "source_evidence_receipt.json"
    if not target.exists():
        _write_json_exclusive(target, receipts)
    return receipts


def prepare(name):
    from data_contract import parse_electricity_prefix
    directory = CACHE / "data"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}_permitted.npz"
    receipt_path = directory / f"{name}_receipt.json"
    if target.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if digest(target) != receipt["cache_sha256"]:
            raise ValueError("Prepared cache changed")
        print(json.dumps({"prepared": name, "cache_reused_verified": True, "shape": receipt["shape"]}), flush=True)
        return receipt
    source = acquire(name)
    if name == "electricity":
        with source.open("r", encoding="utf-8") as stream:
            physical, times, metadata = parse_electricity_prefix(stream)
        observed = np.isfinite(physical)
        columns = metadata["columns"]
        interval = 3600
        metadata.update({"units": "Source calls hourly aggregated electricity consumption; raw UCI kW ancestry; exact aggregation and unit conversion not rederived",
                         "zero_policy": "All finite values including 0 observed; source declares no missing",
                         "preprocessing_ancestry": "Official 321 hourly 2012-2014 Lai et al. aggregated panel, not selected 321 of original 370 columns; upstream imputation/aggregation not reconstructed",
                         "license": "CC-BY-4.0", "source_missing_declaration": "false", "source_equallength_declaration": "true"})
    elif name == "pems08":
        with np.load(source, allow_pickle=False) as archive:
            raw = archive["data"]
            if archive.files != ["data"] or raw.shape != (17856, 170, 3):
                raise ValueError("Pinned PeMS08 schema differs")
            physical = raw[:, :, 0].astype(np.float32, copy=True)
        observed = np.isfinite(physical)
        times = np.arange(len(physical), dtype=np.int64) * 300 * 1_000_000_000
        columns = [f"sensor_index_{i:03d}" for i in range(170)]
        interval = 300
        metadata = {"source_key": "data", "source_shape": [17856, 170, 3], "source_dtype": "float64",
                    "selected_feature_axis": 2, "selected_feature_index": 0,
                    "target_semantics": "Traffic flow, feature0 as author ASTGCN preprocessing target and input selector",
                    "timestamp_timezone": "No timestamps in NPZ; synthetic elapsed grid from 1970-01-01T00:00:00; calendar dates not asserted",
                    "grid_evidence": "Pinned author configuration points_per_hour=12 => 5min; num_of_vertices=170",
                    "columns_are": "Preserved complete NPZ sensor axis order; original external sensor IDs not in NPZ",
                    "units": "Traffic flow as author data; exact physical counting-unit specification absent",
                    "zero_policy": "Finite mask including 0 observed; zero not treated as missing",
                    "upstream_imputation": "Author distributed NPZ preprocessing unavailable; cannot rule out original future-based interpolation. No new interpolation applied",
                    "license": "AUTHOR_RESEARCH_DISTRIBUTION_FORMAL_LICENSE_UNSPECIFIED",
                    "raw_publication_allowed": False, "task_deviation": "L512/H12 differs from TimeMixer paper L96/H12; not original benchmark-score reproduction"}
    else:
        raise ValueError("Only approved new source names supported")
    physical[~observed] = np.nan
    metadata.update({"source_url": SOURCES[name]["url"], "source_sha256": digest(source),
                     "source_bytes": source.stat().st_size, "source_reference": f"sources/{source.name}",
                     "new_to_campaign": True, "historically_unseen_to_researcher": "unknown",
                     "pretraining_overlap_known": "unknown", "raw_publication_allowed": False})
    with target.open("xb") as stream:
        np.savez_compressed(stream, physical=physical, observed=observed, times=times, columns=np.asarray(columns))
    receipt = {"name": name, "shape": list(physical.shape), "interval_seconds": interval,
               "cache_reference": f"data/{target.name}", "cache_sha256": digest(target),
               "cache_bytes": target.stat().st_size, "metadata": metadata,
               "created_unix": time.time(), "source_sha256": digest(source)}
    _write_json_exclusive(receipt_path, receipt)
    print(json.dumps({"prepared": name, "shape": list(physical.shape), "cache_sha256": receipt["cache_sha256"], "cache_reference": receipt["cache_reference"]}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--source-evidence", action="store_true")
    args = parser.parse_args()
    for dataset in SOURCES if args.download == "all" else [args.download]:
        prepare(dataset) if args.prepare else acquire(dataset)
    if args.source_evidence:
        fetch_source_evidence()
