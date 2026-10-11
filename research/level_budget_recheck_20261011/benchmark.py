"""Isolated full E-F-D benchmark for the 180-epoch budget recheck.

Root runs this after run.py creates result.json. The default --all mode launches
a fresh Python process for each dataset/adapter case and batch size.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch

import run as campaign


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
RUNTIME_V18 = ROOT / "research/level_matched_learned_ed_v18_20261006/runtime_v18.py"


def atomic_json(path: Path, value) -> None:
    campaign.atomic_json(path, value)


def load_result(path: Path) -> dict:
    result = campaign.read_json(path)
    if result.get("status") != "complete":
        raise RuntimeError(f"result is not complete: {result.get('status')}")
    return result


def case_rows(result: dict) -> list[dict]:
    rows = []
    protocol = result["protocol"]
    for dataset_index, dataset in enumerate(protocol["datasets"]):
        cases = list(result["datasets"][dataset]["benchmark_cases"])
        anchors = [case for case in cases if case in ("PCA_ANCHOR", "RAW_F0")]
        learned = [case for case in cases if case not in ("PCA_ANCHOR", "RAW_F0")]
        if learned:
            shift = dataset_index % len(learned)
            learned = learned[shift:] + learned[:shift]
        for adapter in anchors + learned:
            for batch in protocol["benchmark"]["batches"]:
                rows.append({"dataset": dataset, "adapter": adapter, "batch": int(batch), "case_order": len(rows)})
    return rows


def load_mean_std(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    runtime = campaign.import_file(RUNTIME_V18, "_budget_recheck_runtime_v18")
    data = runtime.load_data(dataset, include_test=False)
    if "mean" not in data or "std" not in data:
        raise RuntimeError(f"{dataset} archive has no mean/std; benchmark normalization contract cannot be executed")
    mean = np.asarray(data["mean"], dtype=np.float64)
    std = np.asarray(data["std"], dtype=np.float64)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise RuntimeError(f"{dataset} mean/std shape invalid")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise RuntimeError(f"{dataset} mean/std nonfinite or nonpositive")
    return mean, std


def raw_from_standardized(x_standardized: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x_standardized.astype(np.float64) * std[None, None, :] + mean[None, None, :]


def normalize_raw(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((raw.astype(np.float64) - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def make_timing_indices(total: int, batch: int) -> np.ndarray:
    if total <= 0:
        raise ValueError("total must be positive")
    return (np.arange(batch, dtype=np.int64) % total).astype(np.int64)


def load_encoder_for_case(result: dict, dataset: str, adapter: str, bundle, graph: np.ndarray):
    if adapter == "PCA_ANCHOR":
        return bundle.basis, result["datasets"][dataset]["decoder_contracts"][adapter]["encoder_state_sha256_before"]
    if adapter == "RAW_F0":
        return None, "RAW_F0"
    meta = result["datasets"][dataset]["decoder_contracts"][adapter]["meta"]
    fit = result["datasets"][dataset]["fits"][meta["fit_id"]]
    model, state_hash = campaign.load_milestone_encoder(bundle, fit, int(meta["milestone"]), graph)
    return model, state_hash


def load_decoder_coeff(result: dict, dataset: str, adapter: str) -> tuple[np.ndarray | None, dict | None]:
    if adapter == "RAW_F0":
        return None, result["datasets"][dataset]["raw_f0"]
    contract = result["protocol"]["benchmark"]["contract"]
    entry = result["datasets"][dataset]["decoder_contracts"][adapter]["contracts"][contract]
    npz_path = campaign.resolve_path(entry["prediction_npz"])
    if campaign.sha256_path(npz_path) != entry["prediction_sha256"]:
        raise RuntimeError(f"prediction NPZ changed: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        coeff = archive["decoder_coeff"].astype(np.float64)
    return coeff, entry


@torch.no_grad()
def encode_for_benchmark(context_standardized: np.ndarray, adapter_obj, device: str) -> np.ndarray:
    if isinstance(adapter_obj, torch.Tensor):
        x = torch.from_numpy(context_standardized.astype(np.float32)).to(device)
        return torch.einsum("btc,ck->btk", x, adapter_obj).detach().cpu().numpy().astype(np.float32)
    if isinstance(adapter_obj, np.ndarray):
        x = torch.from_numpy(context_standardized.astype(np.float32)).to(device)
        basis = torch.from_numpy(adapter_obj.astype(np.float32)).to(device)
        return torch.einsum("btc,ck->btk", x, basis).detach().cpu().numpy().astype(np.float32)
    x = torch.from_numpy(context_standardized.astype(np.float32)).to(device)
    return adapter_obj.encode(x).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_standardized(context_standardized: np.ndarray, adapter_obj, coeff: np.ndarray | None, backbone, protocol: dict, device: str) -> np.ndarray:
    pr = campaign.parent_runner()
    if adapter_obj is None:
        return pr.old_pilot().bolt_median_forecast(backbone, context_standardized.astype(np.float32), device, int(len(context_standardized))).astype(np.float32)
    z_history = encode_for_benchmark(context_standardized, adapter_obj, device)
    z_future = campaign.forecast_latents(backbone, z_history, {**protocol, "forecast_batch": int(len(context_standardized))}, device)
    features = pr.future_features(z_history, z_future, int(protocol["decoder_max_lag"]))
    return pr.apply_coeff(features, coeff).astype(np.float32)


def full_val_parity(result: dict, dataset: str, adapter: str, adapter_obj, coeff, backbone, protocol: dict, device: str) -> dict:
    pr = campaign.parent_runner()
    bundle = pr.load_dataset(dataset)
    context, _ = pr.old_pilot().windows(bundle, bundle.val_origins)
    chunks = []
    chunk_size = min(32, len(context))
    for start in range(0, len(context), chunk_size):
        chunks.append(predict_standardized(context[start:start + chunk_size], adapter_obj, coeff, backbone, protocol, device))
    pred = np.concatenate(chunks, axis=0)
    if adapter == "RAW_F0":
        npz_path = campaign.resolve_path(result["datasets"][dataset]["raw_f0"]["prediction_npz"])
        expected_sha = result["datasets"][dataset]["raw_f0"]["prediction_sha256"]
    else:
        entry = result["datasets"][dataset]["decoder_contracts"][adapter]["contracts"][protocol["benchmark"]["contract"]]
        npz_path = campaign.resolve_path(entry["prediction_npz"])
        expected_sha = entry["prediction_sha256"]
    if campaign.sha256_path(npz_path) != expected_sha:
        raise RuntimeError(f"stored prediction changed: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        expected = archive["prediction_tsfm"].astype(np.float32)
    diff = np.abs(pred.astype(np.float64) - expected.astype(np.float64))
    rel = diff / np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    atol = float(protocol["benchmark"]["parity_atol"])
    rtol = float(protocol["benchmark"]["parity_rtol"])
    close = np.isclose(pred, expected, atol=atol, rtol=rtol)
    return {
        "passed": bool(close.all()),
        "max_abs": float(diff.max(initial=0.0)),
        "max_relative": float(rel.max(initial=0.0)),
        "violating_elements": int((~close).sum()),
        "elements": int(close.size),
        "prediction_npz": npz_path,
        "prediction_sha256": expected_sha,
        "atol": atol,
        "rtol": rtol,
        "chunk_size": int(chunk_size),
    }


def prediction_entry_for(result: dict, dataset: str, adapter: str) -> tuple[Path, str]:
    if adapter == "RAW_F0":
        return campaign.resolve_path(result["datasets"][dataset]["raw_f0"]["prediction_npz"]), result["datasets"][dataset]["raw_f0"]["prediction_sha256"]
    entry = result["datasets"][dataset]["decoder_contracts"][adapter]["contracts"][result["protocol"]["benchmark"]["contract"]]
    return campaign.resolve_path(entry["prediction_npz"]), entry["prediction_sha256"]


def timed_input_parity(result: dict, dataset: str, adapter: str, indices: np.ndarray, raw_batch: np.ndarray, mean: np.ndarray, std: np.ndarray,
                       adapter_obj, coeff, backbone, protocol: dict, device: str) -> dict:
    normalized = normalize_raw(raw_batch, mean, std)
    pred = predict_standardized(normalized, adapter_obj, coeff, backbone, protocol, device)
    npz_path, expected_sha = prediction_entry_for(result, dataset, adapter)
    if campaign.sha256_path(npz_path) != expected_sha:
        raise RuntimeError(f"stored prediction changed: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        expected = archive["prediction_tsfm"].astype(np.float32)[indices]
    diff = np.abs(pred.astype(np.float64) - expected.astype(np.float64))
    rel = diff / np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    roundtrip = normalize_raw(raw_from_standardized(normalized, mean, std), mean, std)
    roundtrip_diff = np.abs(roundtrip.astype(np.float64) - normalized.astype(np.float64))
    atol = float(protocol["benchmark"]["parity_atol"])
    rtol = float(protocol["benchmark"]["parity_rtol"])
    close = np.isclose(pred, expected, atol=atol, rtol=rtol)
    return {
        "passed": bool(close.all()),
        "max_abs": float(diff.max(initial=0.0)),
        "max_relative": float(rel.max(initial=0.0)),
        "violating_elements": int((~close).sum()),
        "elements": int(close.size),
        "input_indices_sha256": campaign.array_sha256(indices),
        "raw_input_sha256": campaign.array_sha256(raw_batch.astype(np.float64)),
        "normalized_input_sha256": campaign.array_sha256(normalized.astype(np.float32)),
        "raw_dtype": str(raw_batch.dtype),
        "normalized_dtype": str(normalized.dtype),
        "roundtrip_max_abs": float(roundtrip_diff.max(initial=0.0)),
        "atol": atol,
        "rtol": rtol,
    }


class RssSampler:
    def __init__(self, interval: float):
        self.interval = float(interval)
        self.process = psutil.Process()
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), int(self.process.memory_info().rss)))
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1)

    def receipt(self) -> dict:
        values = [rss for _, rss in self.samples]
        return {
            "sample_count": len(values),
            "peak_rss_bytes": max(values) if values else int(self.process.memory_info().rss),
            "interval_seconds": self.interval,
            "scope": "observed host RSS while timed benchmark section ran",
        }


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def run_worker(args) -> dict:
    campaign.configure_runtime()
    protocol = campaign.read_json(HERE / "protocol.json")
    result = load_result(Path(args.result))
    campaign.ensure_source_hashes_unchanged(result["source_hashes"])
    pr = campaign.parent_runner()
    bundle = pr.load_dataset(args.dataset)
    graph = pr.build_train_graph(bundle.x, bundle.finite, bundle.train_bounds, top_k=3)
    adapter_obj, state_hash = load_encoder_for_case(result, args.dataset, args.adapter, bundle, graph)
    coeff, prediction_entry = load_decoder_coeff(result, args.dataset, args.adapter)
    if isinstance(adapter_obj, np.ndarray):
        adapter_obj = torch.from_numpy(adapter_obj.astype(np.float32)).to(args.device)
    if isinstance(adapter_obj, torch.nn.Module):
        adapter_obj = adapter_obj.to(args.device).eval()
        for parameter in adapter_obj.parameters():
            parameter.requires_grad_(False)
    backbone = pr.old_pilot().load_backbone(args.dataset, device=args.device)
    backbone_before = pr.old_pilot().backbone_state_hash(backbone)
    parity = full_val_parity(result, args.dataset, args.adapter, adapter_obj, coeff, backbone, protocol, args.device)
    if not parity["passed"]:
        raise RuntimeError(f"benchmark/evaluation prediction parity failed for {args.dataset}:{args.adapter}: {parity}")

    mean, std = load_mean_std(args.dataset)
    val_context, _ = pr.old_pilot().windows(bundle, bundle.val_origins)
    indices = make_timing_indices(len(val_context), int(args.batch))
    raw_batch = raw_from_standardized(val_context[indices], mean, std)
    timed_parity = timed_input_parity(result, args.dataset, args.adapter, indices, raw_batch, mean, std,
                                      adapter_obj, coeff, backbone, protocol, args.device)
    if not timed_parity["passed"]:
        raise RuntimeError(f"timed raw-normalize pipeline parity failed for {args.dataset}:{args.adapter}: {timed_parity}")
    del val_context
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    warmups = int(protocol["benchmark"]["warmups"])
    repetitions = int(protocol["benchmark"]["repetitions"])
    timings = []
    for _ in range(warmups):
        synchronize(args.device)
        normalized = normalize_raw(raw_batch, mean, std)
        _ = predict_standardized(normalized, adapter_obj, coeff, backbone, protocol, args.device)
        synchronize(args.device)
    cuda_baseline = {}
    if args.device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(args.device)
        synchronize(args.device)
        cuda_baseline = {
            "cuda_allocated_bytes_before_measured": int(torch.cuda.memory_allocated(dev)),
            "cuda_reserved_bytes_before_measured": int(torch.cuda.memory_reserved(dev)),
        }
        torch.cuda.reset_peak_memory_stats(dev)
    with RssSampler(float(protocol["benchmark"]["rss_sampling_interval_seconds"])) as rss:
        for _ in range(repetitions):
            synchronize(args.device)
            started = time.perf_counter()
            normalized = normalize_raw(raw_batch, mean, std)
            _ = predict_standardized(normalized, adapter_obj, coeff, backbone, protocol, args.device)
            synchronize(args.device)
            timings.append(time.perf_counter() - started)
    arr = np.asarray(timings, dtype=np.float64)
    cuda_receipt = {}
    if args.device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(args.device)
        cuda_receipt = {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
            "cuda_baseline_allocated_bytes_after": int(torch.cuda.memory_allocated(dev)),
            "cuda_baseline_reserved_bytes_after": int(torch.cuda.memory_reserved(dev)),
        }
    backbone_after = pr.old_pilot().backbone_state_hash(backbone)
    if backbone_before != backbone_after:
        raise RuntimeError("frozen backbone changed during benchmark")
    if not all(parameter.grad is None for parameter in backbone.parameters()):
        raise RuntimeError("frozen backbone accumulated gradients during benchmark")
    row = {
        "dataset": args.dataset,
        "adapter": args.adapter,
        "batch": int(args.batch),
        "encoder_state_sha256": state_hash,
        "prediction_entry": prediction_entry,
        "parity": parity,
        "timed_input_parity": timed_parity,
        "timed_input": {
            "origin_indices": indices.tolist(),
            "origin_indices_sha256": campaign.array_sha256(indices),
            "raw_input_sha256": campaign.array_sha256(raw_batch.astype(np.float64)),
            "raw_dtype": str(raw_batch.dtype),
        },
        "timings_seconds": timings,
        "median_seconds": float(np.median(arr)),
        "p95_seconds": float(np.percentile(arr, 95)),
        "throughput_origins_per_second": float(int(args.batch) / np.median(arr)),
        "warmups": warmups,
        "repetitions": repetitions,
        "normalization": {
            "mode": "raw-equivalent input reconstructed from TRAIN-standardized archive, then timed raw-to-standard normalize",
            "mean_sha256": campaign.array_sha256(mean),
            "std_sha256": campaign.array_sha256(std),
            "mean_dtype": str(mean.dtype),
            "std_dtype": str(std.dtype),
            "missing_input_contract": "standardized missing value 0 corresponds to raw TRAIN mean and normalizes back to 0",
        },
        "scope_included": protocol["benchmark"]["included"],
        "scope_excluded": protocol["benchmark"]["excluded"],
        "rss": rss.receipt(),
        "backbone_state_sha256_before": backbone_before,
        "backbone_state_sha256_after": backbone_after,
        **cuda_receipt,
        **cuda_baseline,
    }
    out = CACHE / "benchmarks" / f"{args.dataset}_{args.adapter}_B{int(args.batch)}.json"
    atomic_json(out, row)
    return row


def run_all(args) -> dict:
    result = load_result(Path(args.result))
    rows = case_rows(result)
    outputs = []
    for row in rows:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--result", str(Path(args.result)),
            "--dataset", row["dataset"],
            "--adapter", row["adapter"],
            "--batch", str(row["batch"]),
            "--device", args.device,
        ]
        completed = subprocess.run(cmd, cwd=str(ROOT), check=True, text=True, capture_output=True)
        outputs.append({"case": row, "stdout": completed.stdout, "stderr": completed.stderr})
    combined = {"schema": "budget_recheck_benchmark_v1", "rows": [], "subprocesses": outputs}
    for row in rows:
        path = CACHE / "benchmarks" / f"{row['dataset']}_{row['adapter']}_B{row['batch']}.json"
        combined["rows"].append(campaign.read_json(path))
    atomic_json(HERE / "benchmark.json", combined)
    return combined


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default=str(HERE / "result.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--adapter")
    parser.add_argument("--batch", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        row = run_worker(parsed)
        print(json.dumps({"dataset": row["dataset"], "adapter": row["adapter"], "batch": row["batch"], "median_seconds": row["median_seconds"]}))
    else:
        run_all(parsed)
