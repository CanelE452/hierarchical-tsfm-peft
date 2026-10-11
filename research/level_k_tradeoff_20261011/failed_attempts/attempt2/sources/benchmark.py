from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
REPLACE_RETRY_DELAYS_SECONDS = (0.010, 0.025, 0.050, 0.100, 0.200)


def import_file(path: Path, module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


campaign = import_file(HERE / "sweep.py", "_k_tradeoff_sweep_for_benchmark")


def atomic_json(path: Path, value) -> None:
    path = path.resolve()
    if not (path.is_relative_to(HERE.resolve()) or path.is_relative_to(CACHE.resolve())):
        raise PermissionError(f"refusing to write outside benchmark folders: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(campaign.to_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    replace_with_permission_retry(tmp, path)


def replace_with_permission_retry(tmp: Path, target: Path, delays: tuple[float, ...] = REPLACE_RETRY_DELAYS_SECONDS) -> None:
    for attempt in range(len(delays) + 1):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if attempt == len(delays):
                raise
            time.sleep(float(delays[attempt]))


def load_result(path: Path | str) -> dict:
    result = campaign.read_json(campaign.resolve_path(path))
    if result.get("status") != "complete":
        raise RuntimeError(f"result is not complete: {path}")
    return result


def adapter_k(dataset: str, adapter: str) -> int | None:
    if adapter == "RAW_F0" or adapter.endswith("_RAW_C_D8"):
        return campaign.CHANNELS[dataset]
    match = re.search(r"_K(\d+)(?:_|$)", adapter)
    return int(match.group(1)) if match else None


def case_rows(result: dict, method_result: dict | None = None) -> list[dict]:
    rows = []
    batches = [int(v) for v in result["protocol"]["benchmark"]["batches"]]
    for dataset_index, dataset in enumerate(result["protocol"]["datasets"]):
        local_cases = list(result["datasets"][dataset]["benchmark_cases"])
        raw = [case for case in local_cases if case == "RAW_F0"]
        learned = [case for case in local_cases if case != "RAW_F0"]
        if learned:
            shift = dataset_index % len(learned)
            learned = learned[shift:] + learned[:shift]
        combined = [("local", case) for case in raw + learned]
        if method_result is not None:
            combined.extend(("method", case) for case in method_result.get("datasets", {}).get(dataset, {}).get("benchmark_cases", []))
        for source, adapter in combined:
            for batch in batches:
                rows.append({"dataset": dataset, "adapter": adapter, "batch": int(batch), "source": source, "k": adapter_k(dataset, adapter), "case_order": len(rows)})
    return rows


def load_mean_std(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    pilot = campaign.parent_runner().old_pilot()
    runtime = campaign.import_file(pilot.RUNTIME_V18, "_k_tradeoff_runtime_v18")
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


def adapter_record(result: dict, method_result: dict | None, dataset: str, adapter: str) -> tuple[dict | None, str]:
    if adapter == "RAW_F0":
        return result["datasets"][dataset]["raw_f0"], "local"
    local = result["datasets"][dataset].get("decoder_contracts", {})
    if adapter in local:
        return local[adapter], "local"
    if method_result is not None:
        remote = method_result.get("datasets", {}).get(dataset, {}).get("decoder_contracts", {})
        if adapter in remote:
            return remote[adapter], "method"
    raise KeyError(f"unknown adapter {dataset}:{adapter}")


def load_method_adapter(dataset: str, adapter: str, record: dict, device: str):
    method_path = HERE / "method_run.py"
    if not method_path.exists():
        raise RuntimeError(f"method adapter requested before method_run.py exists: {adapter}")
    method = import_file(method_path, "_k_tradeoff_method_run")
    if not hasattr(method, "load_adapter"):
        raise RuntimeError("method_run.py does not expose load_adapter(dataset, adapter, record, device)")
    return method.load_adapter(dataset, adapter, record, device)


def load_adapter(dataset: str, adapter: str, record: dict | None, result: dict, device: str):
    if adapter == "RAW_F0":
        return None, "RAW_F0"
    meta = record.get("meta", {})
    family = str(meta.get("family", record.get("family", ""))).upper()
    k = int(meta.get("k", adapter_k(dataset, adapter)))
    bundle = campaign.load_bundle(dataset, k)
    if family.startswith("TS2VEC"):
        return load_method_adapter(dataset, adapter, record, device)
    if family == "RAW_C_D8":
        basis = torch.eye(bundle.channels, dtype=torch.float32, device=device)
        return basis, record["encoder_state_sha256_before"]
    if family == "PCA":
        basis_info = result["datasets"][dataset]["basis"][f"K{k}"]
        basis_path = campaign.resolve_path(basis_info["path"])
        if campaign.sha256_path(basis_path) != basis_info["sha256"]:
            raise RuntimeError(f"basis changed: {basis_path}")
        basis = torch.from_numpy(np.load(basis_path).astype(np.float32)).to(device)
        return basis, record["encoder_state_sha256_before"]
    if family == "MLP_POINT":
        fit = result["datasets"][dataset]["fits"][adapter]
        model, state_hash = campaign.load_mlp_encoder_from_fit(bundle, fit)
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, state_hash
    raise ValueError(f"unknown adapter family for benchmark: {family}")


def load_decoder_coeff(result: dict, dataset: str, adapter: str, method_result: dict | None = None) -> tuple[np.ndarray | None, dict | None]:
    record, _ = adapter_record(result, method_result, dataset, adapter)
    if adapter == "RAW_F0":
        return None, record
    entry = record["contracts"][campaign.CONTRACT]
    npz_path = campaign.resolve_path(entry["prediction_npz"])
    if campaign.sha256_path(npz_path) != entry["prediction_sha256"]:
        raise RuntimeError(f"prediction NPZ changed: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        coeff = archive["decoder_coeff"].astype(np.float64)
    return coeff, entry


@torch.no_grad()
def encode_for_benchmark(context_standardized: np.ndarray, adapter_obj, device: str) -> np.ndarray:
    x = torch.from_numpy(context_standardized.astype(np.float32)).to(device)
    if isinstance(adapter_obj, torch.Tensor):
        return torch.einsum("btc,ck->btk", x, adapter_obj).detach().cpu().numpy().astype(np.float32)
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


def prediction_entry_for(result: dict, method_result: dict | None, dataset: str, adapter: str) -> tuple[Path, str]:
    if adapter == "RAW_F0":
        entry = result["datasets"][dataset]["raw_f0"]
    else:
        record, _ = adapter_record(result, method_result, dataset, adapter)
        entry = record["contracts"][campaign.CONTRACT]
    return campaign.resolve_path(entry["prediction_npz"]), entry["prediction_sha256"]


def full_val_parity(result: dict, method_result: dict | None, dataset: str, adapter: str, adapter_obj, coeff, backbone, protocol: dict, device: str) -> dict:
    pr = campaign.parent_runner()
    bundle = campaign.load_original_bundle(dataset)
    context, _ = pr.old_pilot().windows(bundle, bundle.val_origins)
    chunks = []
    chunk_size = min(32, len(context))
    for start in range(0, len(context), chunk_size):
        chunks.append(predict_standardized(context[start:start + chunk_size], adapter_obj, coeff, backbone, protocol, device))
    pred = np.concatenate(chunks, axis=0)
    npz_path, expected_sha = prediction_entry_for(result, method_result, dataset, adapter)
    if campaign.sha256_path(npz_path) != expected_sha:
        raise RuntimeError(f"stored prediction changed: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        expected = archive["prediction_tsfm"].astype(np.float32)
    diff = np.abs(pred.astype(np.float64) - expected.astype(np.float64))
    rel = diff / np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    atol = float(protocol["benchmark"]["parity_atol"])
    rtol = float(protocol["benchmark"]["parity_rtol"])
    close = np.isclose(pred, expected, atol=atol, rtol=rtol)
    return {"passed": bool(close.all()), "max_abs": float(diff.max(initial=0.0)), "max_relative": float(rel.max(initial=0.0)),
            "violating_elements": int((~close).sum()), "elements": int(close.size), "prediction_npz": npz_path,
            "prediction_sha256": expected_sha, "atol": atol, "rtol": rtol, "chunk_size": int(chunk_size)}


def timed_input_parity(result: dict, method_result: dict | None, dataset: str, adapter: str, indices: np.ndarray, raw_batch: np.ndarray,
                       mean: np.ndarray, std: np.ndarray, adapter_obj, coeff, backbone, protocol: dict, device: str) -> dict:
    normalized = normalize_raw(raw_batch, mean, std)
    pred = predict_standardized(normalized, adapter_obj, coeff, backbone, protocol, device)
    npz_path, expected_sha = prediction_entry_for(result, method_result, dataset, adapter)
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
    return {"passed": bool(close.all()), "max_abs": float(diff.max(initial=0.0)), "max_relative": float(rel.max(initial=0.0)),
            "violating_elements": int((~close).sum()), "elements": int(close.size), "input_indices_sha256": campaign.array_sha256(indices),
            "raw_input_sha256": campaign.array_sha256(raw_batch.astype(np.float64)), "normalized_input_sha256": campaign.array_sha256(normalized.astype(np.float32)),
            "raw_dtype": str(raw_batch.dtype), "normalized_dtype": str(normalized.dtype), "roundtrip_max_abs": float(roundtrip_diff.max(initial=0.0)),
            "atol": atol, "rtol": rtol}


class RssSampler:
    def __init__(self, interval: float):
        self.interval = float(interval)
        self.process = psutil.Process() if psutil is not None else None
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.process is not None:
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
        current = int(self.process.memory_info().rss) if self.process is not None else None
        return {"sample_count": len(values), "peak_rss_bytes": max(values) if values else current,
                "interval_seconds": self.interval, "scope": "observed host RSS while timed benchmark section ran"}


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def run_worker(args) -> dict:
    campaign.configure_runtime()
    protocol = campaign.read_json(HERE / "protocol.json")
    result = load_result(args.result)
    method_result = load_result(args.method_result) if args.method_result else None
    campaign.ensure_source_hashes_unchanged(result["source_hashes"])
    record, source = adapter_record(result, method_result, args.dataset, args.adapter)
    adapter_obj, state_hash = load_adapter(args.dataset, args.adapter, record, result, args.device)
    coeff, prediction_entry = load_decoder_coeff(result, args.dataset, args.adapter, method_result=method_result)
    backbone = campaign.parent_runner().old_pilot().load_backbone(args.dataset, device=args.device)
    backbone_before = campaign.parent_runner().old_pilot().backbone_state_hash(backbone)
    parity = full_val_parity(result, method_result, args.dataset, args.adapter, adapter_obj, coeff, backbone, protocol, args.device)
    if not parity["passed"]:
        raise RuntimeError(f"benchmark/evaluation prediction parity failed: {parity}")
    mean, std = load_mean_std(args.dataset)
    bundle = campaign.load_original_bundle(args.dataset)
    val_context, _ = campaign.parent_runner().old_pilot().windows(bundle, bundle.val_origins)
    indices = make_timing_indices(len(val_context), int(args.batch))
    raw_batch = raw_from_standardized(val_context[indices], mean, std)
    timed_parity = timed_input_parity(result, method_result, args.dataset, args.adapter, indices, raw_batch, mean, std,
                                      adapter_obj, coeff, backbone, protocol, args.device)
    if not timed_parity["passed"]:
        raise RuntimeError(f"timed raw-normalize pipeline parity failed: {timed_parity}")
    del val_context
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    warmups = int(protocol["benchmark"]["warmups"])
    repetitions = int(protocol["benchmark"]["repetitions"])
    for _ in range(warmups):
        synchronize(args.device)
        _ = predict_standardized(normalize_raw(raw_batch, mean, std), adapter_obj, coeff, backbone, protocol, args.device)
        synchronize(args.device)
    cuda_baseline = {}
    if args.device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(args.device)
        synchronize(args.device)
        cuda_baseline = {"cuda_allocated_bytes_before_measured": int(torch.cuda.memory_allocated(dev)),
                         "cuda_reserved_bytes_before_measured": int(torch.cuda.memory_reserved(dev))}
        torch.cuda.reset_peak_memory_stats(dev)
    timings = []
    with RssSampler(float(protocol["benchmark"]["rss_sampling_interval_seconds"])) as rss:
        for _ in range(repetitions):
            synchronize(args.device)
            started = time.perf_counter()
            _ = predict_standardized(normalize_raw(raw_batch, mean, std), adapter_obj, coeff, backbone, protocol, args.device)
            synchronize(args.device)
            timings.append(time.perf_counter() - started)
    arr = np.asarray(timings, dtype=np.float64)
    cuda_receipt = {}
    if args.device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(args.device)
        cuda_receipt = {"cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
                        "cuda_baseline_allocated_bytes_after": int(torch.cuda.memory_allocated(dev)),
                        "cuda_baseline_reserved_bytes_after": int(torch.cuda.memory_reserved(dev))}
    backbone_after = campaign.parent_runner().old_pilot().backbone_state_hash(backbone)
    if backbone_before != backbone_after:
        raise RuntimeError("frozen backbone changed during benchmark")
    if not all(parameter.grad is None for parameter in backbone.parameters()):
        raise RuntimeError("frozen backbone accumulated gradients during benchmark")
    row = {"dataset": args.dataset, "adapter": args.adapter, "adapter_source": source, "batch": int(args.batch),
           "k": adapter_k(args.dataset, args.adapter), "encoder_state_sha256": state_hash, "prediction_entry": prediction_entry,
           "parity": parity, "timed_input_parity": timed_parity,
           "timed_input": {"origin_indices": indices.tolist(), "origin_indices_sha256": campaign.array_sha256(indices),
                           "raw_input_sha256": campaign.array_sha256(raw_batch.astype(np.float64)), "raw_dtype": str(raw_batch.dtype)},
           "timings_seconds": timings, "median_seconds": float(np.median(arr)), "p95_seconds": float(np.percentile(arr, 95)),
           "throughput_origins_per_second": float(int(args.batch) / np.median(arr)), "warmups": warmups, "repetitions": repetitions,
           "normalization": {"mode": "raw-equivalent input reconstructed from TRAIN-standardized archive, then timed raw-to-standard normalize",
                             "mean_sha256": campaign.array_sha256(mean), "std_sha256": campaign.array_sha256(std),
                             "mean_dtype": str(mean.dtype), "std_dtype": str(std.dtype),
                             "missing_input_contract": "standardized missing value 0 corresponds to raw TRAIN mean and normalizes back to 0"},
           "scope_included": protocol["benchmark"]["included"], "scope_excluded": protocol["benchmark"]["excluded"],
           "rss": rss.receipt(), "backbone_state_sha256_before": backbone_before, "backbone_state_sha256_after": backbone_after,
           **cuda_receipt, **cuda_baseline}
    out = CACHE / "benchmarks" / f"{args.dataset}_{args.adapter}_B{int(args.batch)}.json"
    atomic_json(out, row)
    return row


def run_all(args) -> dict:
    result = load_result(args.result)
    method_result = load_result(args.method_result) if args.method_result else None
    rows = case_rows(result, method_result=method_result)
    outputs = []
    for row in rows:
        cmd = [sys.executable, str(HERE / "benchmark.py"), "--worker", "--result", str(args.result), "--dataset", row["dataset"],
               "--adapter", row["adapter"], "--batch", str(row["batch"]), "--device", args.device]
        if args.method_result:
            cmd.extend(["--method-result", str(args.method_result)])
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        path = CACHE / "benchmarks" / f"{row['dataset']}_{row['adapter']}_B{row['batch']}.json"
        outputs.append(campaign.read_json(path))
    combined = {"schema": "k_tradeoff_benchmark_v1", "status": "complete", "result": args.result,
                "method_result": args.method_result, "rows": outputs, "case_count": len(outputs)}
    atomic_json(HERE / "benchmark.json", combined)
    return combined


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default=str(HERE / "result.json"))
    parser.add_argument("--method-result", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--adapter")
    parser.add_argument("--batch", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        run_worker(parsed)
    else:
        run_all(parsed)
