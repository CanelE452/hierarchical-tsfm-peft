from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
PARENT_RESULT = ROOT / "research/level_budget_recheck_20261011/result.json"
PARENT_RUN = ROOT / "research/level_budget_recheck_20261011/run.py"
PARENT_BENCH = ROOT / "research/level_budget_recheck_20261011/benchmark.py"
PARENT_RUNNER = ROOT / "research/level_encoder_contract_20261010/runner.py"
PARENT_ENCODERS = ROOT / "research/level_encoder_contract_20261010/encoders.py"

CHANNELS = {"robin": 17, "jena": 21}
CONTRACT = "FORECAST_RIDGE_D8"
MONITOR = None
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


def parent_runner():
    return import_file(PARENT_RUNNER, "_k_tradeoff_parent_runner")


def parent_encoders():
    return import_file(PARENT_ENCODERS, "_k_tradeoff_parent_encoders")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def public_path(path: Path | str) -> str:
    text = str(path)
    return text.replace(str(ROOT) + os.sep, "").replace(ROOT.as_posix() + "/", "")


def resolve_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def to_jsonable(value):
    if isinstance(value, Path):
        return public_path(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def atomic_json(path: Path, value) -> None:
    path = path.resolve()
    if not (path.is_relative_to(HERE.resolve()) or path.is_relative_to(CACHE.resolve())):
        raise PermissionError(f"refusing to write outside k-tradeoff folders: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def source_hashes() -> dict:
    files = {
        "PLAN": HERE / "PLAN.md",
        "protocol": HERE / "protocol.json",
        "sweep": HERE / "sweep.py",
        "benchmark": HERE / "benchmark.py",
        "test_sweep": HERE / "test_sweep.py",
        "seal": HERE / "seal.py",
    }
    return {k: sha256_path(v) for k, v in files.items() if v.exists()}


def dependency_hashes() -> dict:
    files = {
        "parent_result": PARENT_RESULT,
        "parent_run": PARENT_RUN,
        "parent_benchmark": PARENT_BENCH,
        "parent_runner": PARENT_RUNNER,
        "parent_encoders": PARENT_ENCODERS,
    }
    return {k: sha256_path(v) for k, v in files.items() if v.exists()}


def ensure_source_hashes_unchanged(reference: dict) -> None:
    now = source_hashes()
    if now != reference:
        raise RuntimeError({"source_hash_changed": {"before": reference, "after": now}})


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Monitor:
    def __init__(self, protocol: dict, device: str, progress_name: str = "progress.json"):
        self.protocol = protocol
        self.device = device
        self.progress_path = HERE / progress_name
        self.started = time.perf_counter()
        self.process = psutil.Process() if psutil is not None else None
        self.limits = protocol.get("limits", {})
        self.progress = {
            "status": "running",
            "pid": os.getpid(),
            "device": device,
            "stage": "init",
            "fit_starts": 0,
            "completed_fits": 0,
            "ridge_fits": 0,
        }

    def _owned_storage_bytes(self) -> int:
        return owned_storage_bytes_for((HERE, CACHE))

    def check(self, stage: str | None = None, **fields) -> None:
        if stage is not None:
            self.progress["stage"] = stage
        self.progress.update(fields)
        elapsed = time.perf_counter() - self.started
        self.progress["elapsed_seconds"] = float(elapsed)
        if self.process is not None:
            cpu = sum(self.process.cpu_times()[:2])
            rss = int(self.process.memory_info().rss)
            self.progress["process_cpu_seconds"] = float(cpu)
            self.progress["rss_bytes"] = rss
            if self.limits.get("process_cpu_seconds") and cpu > float(self.limits["process_cpu_seconds"]):
                raise RuntimeError("STOP_RESOURCE: process_cpu_seconds")
            if self.limits.get("ram_bytes") and rss > int(self.limits["ram_bytes"]):
                raise RuntimeError("STOP_RESOURCE: ram_bytes")
        if self.limits.get("wall_seconds") and elapsed > float(self.limits["wall_seconds"]):
            raise RuntimeError("STOP_RESOURCE: wall_seconds")
        if self.device.startswith("cuda") and torch.cuda.is_available():
            dev = torch.device(self.device)
            allocated = int(torch.cuda.max_memory_allocated(dev))
            reserved = int(torch.cuda.max_memory_reserved(dev))
            self.progress["cuda_max_allocated_bytes"] = allocated
            self.progress["cuda_max_reserved_bytes"] = reserved
            if self.limits.get("allocated_vram_bytes") and allocated > int(self.limits["allocated_vram_bytes"]):
                raise RuntimeError("STOP_RESOURCE: allocated_vram_bytes")
        if self.limits.get("incremental_storage_bytes"):
            storage = self._owned_storage_bytes()
            self.progress["owned_storage_bytes"] = storage
            if storage > int(self.limits["incremental_storage_bytes"]):
                raise RuntimeError("STOP_RESOURCE: incremental_storage_bytes")
        atomic_json(self.progress_path, self.progress)

    def receipt(self) -> dict:
        self.check()
        return copy.deepcopy(self.progress)


def monitor(stage: str | None = None, **fields) -> None:
    if MONITOR is not None:
        MONITOR.check(stage, **fields)


def owned_storage_bytes_for(folders: Iterable[Path]) -> int:
    total = 0

    def onerror(error: OSError) -> None:
        path = Path(getattr(error, "filename", "") or "")
        if isinstance(error, FileNotFoundError) and (path.name == "__pycache__" or path.name.endswith(".tmp")):
            return
        raise error

    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(folder, topdown=True, onerror=onerror):
            dirnames[:] = [name for name in dirnames if name != "__pycache__"]
            for filename in filenames:
                path = Path(dirpath) / filename
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    if path.name.endswith(".tmp") or "__pycache__" in path.parts:
                        continue
                    raise
    return int(total)


class _SharedMonitorAdapter:
    def check(self, stage: str | None = None, **fields) -> None:
        monitor(stage, **fields)


def install_monitor_hooks() -> None:
    adapter = _SharedMonitorAdapter()
    pr = parent_runner()
    pr.MONITOR = adapter
    pr.old_pilot().MONITOR = adapter


def compute_pca_basis(x: np.ndarray, train_bounds: tuple[int, int], rank: int) -> np.ndarray:
    start, end = int(train_bounds[0]), int(train_bounds[1])
    train = np.asarray(x[start:end], dtype=np.float64)
    if train.ndim != 2 or not 0 < int(rank) <= train.shape[1]:
        raise ValueError("invalid PCA train matrix or rank")
    centered = train - train.mean(axis=0, keepdims=True)
    _, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    return vectors[:, -int(rank):][:, ::-1].copy().astype(np.float32)


def adapter_id(dataset: str, family: str, k: int | None = None, seed: int | None = None) -> str:
    family = str(family).upper()
    if family == "RAW_F0":
        return "RAW_F0"
    if family == "RAW_C_D8":
        return f"{dataset}_RAW_C_D8"
    if family == "PCA":
        return f"{dataset}_PCA_K{int(k)}"
    if seed is None:
        raise ValueError("seed required for learned adapter id")
    return f"{dataset}_{family}_K{int(k)}_seed{int(seed)}"


def adapter_plan(protocol: dict, dataset: str) -> list[dict]:
    rows = []
    for k in protocol["k_grid"][dataset]:
        rows.append({"id": adapter_id(dataset, "PCA", k), "dataset": dataset, "family": "PCA", "k": int(k), "seed": None})
    for k in protocol["k_grid"][dataset]:
        for seed in protocol["seeds"]:
            rows.append({"id": adapter_id(dataset, "MLP_POINT", k, seed), "dataset": dataset, "family": "MLP_POINT", "k": int(k), "seed": int(seed)})
    rows.append({"id": adapter_id(dataset, "RAW_C_D8"), "dataset": dataset, "family": "RAW_C_D8", "k": CHANNELS[dataset], "seed": None})
    return rows


def all_adapter_plan(protocol: dict) -> list[dict]:
    rows = []
    for dataset in protocol["datasets"]:
        rows.extend(adapter_plan(protocol, dataset))
    return rows


future_features = parent_runner().future_features
apply_coeff = parent_runner().apply_coeff
coeff_to_taps = parent_runner().coeff_to_taps


def fit_scaled_ridge_grid(
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
    train_row_mask: np.ndarray,
    val_features: np.ndarray,
    val_target: np.ndarray,
    val_mask: np.ndarray,
    lambdas: Iterable[float],
    horizon_groups: list[tuple[int, int]] | None = None,
) -> dict:
    pr = parent_runner()
    if train_features.shape[:2] != train_target.shape[:2] or train_target.shape != train_mask.shape:
        raise ValueError("train feature/target/mask shapes differ")
    rows = train_row_mask.astype(bool).reshape(-1)
    flat = train_features.reshape(-1, train_features.shape[-1]).astype(np.float64)
    if not np.any(rows):
        raise ValueError("no TRAIN rows available for decoder feature scaling")
    rms = np.sqrt(np.mean(np.square(flat[rows]), axis=0))
    near_zero = rms <= 1e-12
    scale = np.maximum(rms, 1e-12)
    x_train = train_features.astype(np.float64) / scale
    dim = train_features.shape[-1]
    channels = train_target.shape[-1]
    xtx = np.zeros((channels, dim, dim), dtype=np.float64)
    xty = np.zeros((channels, dim), dtype=np.float64)
    counts = np.zeros(channels, dtype=np.int64)
    for channel in range(channels):
        monitor()
        obs = train_mask[:, :, channel].astype(bool) & train_row_mask.astype(bool)
        if not np.any(obs):
            continue
        x = x_train[obs]
        y = train_target[:, :, channel][obs].astype(np.float64)
        xtx[channel] = x.T @ x
        xty[channel] = x.T @ y
        counts[channel] = int(obs.sum())
    choices, coeffs, best = [], [], None
    for index, lam in enumerate(float(v) for v in lambdas):
        monitor()
        if MONITOR is not None:
            if MONITOR.progress["ridge_fits"] >= int(MONITOR.limits.get("ridge_fits", 10**18)):
                raise RuntimeError("STOP_RESOURCE: ridge_fits")
            MONITOR.progress["ridge_fits"] += 1
        coeff_scaled = np.zeros((dim, channels), dtype=np.float64)
        for channel in range(channels):
            if counts[channel] == 0:
                continue
            coeff_scaled[:, channel] = np.linalg.solve(xtx[channel] + lam * np.eye(dim, dtype=np.float64), xty[channel])
        coeff = coeff_scaled / scale[:, None]
        pred = apply_coeff(val_features, coeff)
        metrics = pr.horizon_metrics(pred, val_target, val_mask, horizon_groups) if horizon_groups else {
            "all": pr.masked_channel_macro_metrics_np(pred, val_target, val_mask)
        }
        choice = {"index": int(index), "lambda": float(lam), "metrics": metrics}
        choices.append(choice)
        coeffs.append(coeff.astype(np.float64))
        if best is None or metrics["all"]["mse"] < best["metrics"]["all"]["mse"]:
            best = {**choice, "coeff": coeff}
    return {
        "coeff": best["coeff"].astype(np.float64),
        "selected_index": int(best["index"]),
        "selected_lambda": float(best["lambda"]),
        "selection_metrics": best["metrics"],
        "choices": choices,
        "grid_lambdas": np.asarray([c["lambda"] for c in choices], dtype=np.float64),
        "grid_coefficients": np.stack(coeffs, axis=0).astype(np.float64),
        "feature_scale": scale.astype(np.float64),
        "near_zero_feature_count": int(near_zero.sum()),
        "scale_count": int(rows.sum()),
        "scale_row_mask_sha256": array_sha256(train_row_mask.astype(np.uint8)),
    }


def initial_grid_selection(fitted: dict, initial_lambdas: Iterable[float]) -> dict:
    allowed = [float(v) for v in initial_lambdas]
    best = None
    for choice in fitted["choices"]:
        if choice["lambda"] not in allowed:
            continue
        if best is None or choice["metrics"]["all"]["mse"] < best["metrics"]["all"]["mse"]:
            best = choice
    if best is None:
        raise RuntimeError("initial lambda grid has no overlap with final choices")
    return {"selected_index": int(best["index"]), "selected_lambda": float(best["lambda"]), "selection_metrics": best["metrics"]}


def load_parent(protocol: dict) -> dict:
    path = resolve_path(protocol.get("parent", PARENT_RESULT))
    result = read_json(path)
    if result.get("status") != "complete":
        raise RuntimeError(f"parent result is not complete: {path}")
    return result


def load_original_bundle(dataset: str):
    pr = parent_runner()
    bundle = pr.load_dataset(dataset)
    pr.old_pilot().assert_no_test_origins(bundle, bundle.train_origins)
    pr.old_pilot().assert_no_test_origins(bundle, bundle.val_origins)
    return bundle


def load_bundle(dataset: str, k: int, basis: np.ndarray | None = None):
    bundle = load_original_bundle(dataset)
    k = int(k)
    if basis is None:
        if k == bundle.latent:
            basis = bundle.basis
        elif k == bundle.channels:
            basis = np.eye(bundle.channels, dtype=np.float32)
        else:
            basis = compute_pca_basis(bundle.x, bundle.train_bounds, k)
    basis = np.asarray(basis, dtype=np.float32)
    if basis.shape != (bundle.channels, k):
        raise RuntimeError(f"basis shape {basis.shape} != {(bundle.channels, k)}")
    contract = copy.deepcopy(bundle.contract)
    contract["k"] = k
    return replace(bundle, basis=basis, contract=contract)


@torch.no_grad()
def encode_numpy(encoder_or_basis, context: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    if isinstance(encoder_or_basis, np.ndarray):
        return (context @ encoder_or_basis).astype(np.float32)
    if isinstance(encoder_or_basis, torch.Tensor):
        chunks = []
        basis = encoder_or_basis.to(device)
        for start in range(0, len(context), batch_size):
            monitor()
            x = torch.from_numpy(context[start:start + batch_size].astype(np.float32)).to(device)
            chunks.append(torch.einsum("btc,ck->btk", x, basis).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)
    chunks = []
    encoder_or_basis.eval()
    for start in range(0, len(context), batch_size):
        monitor()
        x = torch.from_numpy(context[start:start + batch_size].astype(np.float32)).to(device)
        chunks.append(encoder_or_basis.encode(x).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def eval_reconstruction(encoder, decoder, bundle, origins: np.ndarray, burn_in: int, device: str, batch_size: int) -> dict:
    pr = parent_runner()
    pilot = pr.old_pilot()
    preds, targets, masks = [], [], []
    encoder.eval()
    decoder.eval()
    for start in range(0, len(origins), batch_size):
        monitor()
        batch_origins = origins[start:start + batch_size]
        x_np, m_np = pilot.windows(bundle, batch_origins)
        x = torch.from_numpy(x_np.astype(np.float32)).to(device)
        preds.append(decoder.decode(encoder.encode(x)).detach().cpu().numpy().astype(np.float32))
        targets.append(x_np)
        masks.append(m_np)
    return pr.masked_channel_macro_metrics_np(
        np.concatenate(preds, axis=0)[:, burn_in:],
        np.concatenate(targets, axis=0)[:, burn_in:],
        np.concatenate(masks, axis=0)[:, burn_in:],
    )


def model_state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        arr = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def save_checkpoint(path: Path, payload: dict) -> None:
    path = path.resolve()
    if not path.is_relative_to(CACHE.resolve()):
        raise PermissionError(f"refusing checkpoint outside cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_mlp_new_k(bundle, seed: int, protocol: dict, device: str) -> dict:
    pe = parent_encoders()
    pr = parent_runner()
    if MONITOR is not None:
        if MONITOR.progress["fit_starts"] >= int(MONITOR.limits.get("neural_fit_starts", 10**18)):
            raise RuntimeError("STOP_RESOURCE: neural_fit_starts")
        MONITOR.progress["fit_starts"] += 1
    fit_id = adapter_id(bundle.name, "MLP_POINT", bundle.latent, seed)
    monitor(stage=f"fit:{fit_id}")
    encoder = pe.build_encoder("MLP_POINT", bundle.channels, bundle.latent, int(seed)).to(device)
    decoder = pe.build_decoder(bundle.channels, bundle.latent, int(seed), lag=int(protocol["decoder_max_lag"])).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(protocol["learning_rate"]), weight_decay=0.0)
    burn_in = int(protocol["burn_in"])
    batch_size = int(protocol["batch"])
    curve, epoch_origin_receipts = [], []
    optimizer_step_count = 0
    best_path = CACHE / "checkpoints" / f"{fit_id}_best180.pt"
    latest_path = CACHE / "checkpoints" / f"{fit_id}_latest.pt"
    started = time.perf_counter()

    def payload(epoch: int, val_metrics: dict) -> dict:
        return {
            "dataset": bundle.name,
            "family": "MLP_POINT",
            "k": bundle.latent,
            "seed": int(seed),
            "epoch": int(epoch),
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state_cpu": torch.get_rng_state(),
            "torch_rng_state_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") and torch.cuda.is_available() else None,
            "val_reconstruction": val_metrics,
            "optimizer_step_count": int(optimizer_step_count),
        }

    val_metrics = eval_reconstruction(encoder, decoder, bundle, bundle.val_origins, burn_in, device, batch_size)
    curve.append({"epoch": 0, "val_reconstruction": val_metrics, "train_loss_mean": None, "optimizer_steps_cumulative": 0})
    best = {"epoch": 0, "val_reconstruction": copy.deepcopy(val_metrics)}
    save_checkpoint(best_path, payload(0, val_metrics))
    save_checkpoint(latest_path, payload(0, val_metrics))
    for epoch in range(1, int(protocol["max_epochs"]) + 1):
        sample = pr.sample_epoch_origins(bundle.train_origins, int(seed), epoch - 1, int(protocol["epoch_origins"]))
        epoch_origin_receipts.append({"epoch": int(epoch), "sha256": array_sha256(sample), "count": int(len(sample))})
        encoder.train()
        decoder.train()
        losses, steps = [], 0
        for start in range(0, len(sample), batch_size):
            monitor()
            origins = sample[start:start + batch_size]
            x_np, m_np = pr.old_pilot().training_reconstruction_windows(bundle, origins)
            x = torch.from_numpy(x_np.astype(np.float32)).to(device)
            mask = torch.from_numpy(m_np[:, burn_in:]).to(device)
            if not bool(mask.any()):
                continue
            pred = decoder.decode(encoder.encode(x))[:, burn_in:]
            loss = pr.masked_channel_macro_mse_torch(pred, x[:, burn_in:], mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), float(protocol["clip_grad_norm"]))
            optimizer.step()
            optimizer_step_count += 1
            steps += 1
            losses.append(float(loss.detach().cpu()))
        val_metrics = eval_reconstruction(encoder, decoder, bundle, bundle.val_origins, burn_in, device, batch_size)
        row = {
            "epoch": int(epoch),
            "val_reconstruction": val_metrics,
            "train_loss_mean": float(np.mean(losses)) if losses else None,
            "optimizer_steps_this_epoch": int(steps),
            "optimizer_steps_cumulative": int(optimizer_step_count),
        }
        curve.append(row)
        if val_metrics["mse"] < best["val_reconstruction"]["mse"]:
            best = {"epoch": int(epoch), "val_reconstruction": copy.deepcopy(val_metrics)}
            save_checkpoint(best_path, payload(epoch, val_metrics))
        save_checkpoint(latest_path, payload(epoch, val_metrics))
        atomic_json(CACHE / "curves" / f"{fit_id}.json", {
            "id": fit_id,
            "dataset": bundle.name,
            "family": "MLP_POINT",
            "k": bundle.latent,
            "seed": int(seed),
            "curves": curve,
            "epoch_origin_receipts": epoch_origin_receipts,
            "optimizer_steps_completed": int(optimizer_step_count),
        })
    if MONITOR is not None:
        MONITOR.progress["completed_fits"] += 1
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model = pe.build_encoder("MLP_POINT", bundle.channels, bundle.latent, int(seed))
    best_model.load_state_dict(best_payload["encoder"])
    curve_path = CACHE / "curves" / f"{fit_id}.json"
    return {
        "id": fit_id,
        "dataset": bundle.name,
        "family": "MLP_POINT",
        "k": bundle.latent,
        "seed": int(seed),
        "max_epochs": int(protocol["max_epochs"]),
        "early_stopping": False,
        "optimizer_steps_completed": int(optimizer_step_count),
        "wall_seconds": float(time.perf_counter() - started),
        "wall_scope": "per-fit wall includes training, full VAL reconstruction, checkpoint saves, and curve writes",
        "curve": curve_path,
        "curve_sha256": sha256_path(curve_path),
        "checkpoint": best_path,
        "checkpoint_sha256": sha256_path(best_path),
        "latest_checkpoint": latest_path,
        "latest_checkpoint_sha256": sha256_path(latest_path),
        "best_epoch": int(best["epoch"]),
        "val_reconstruction": best["val_reconstruction"],
        "encoder_state_sha256": model_state_hash(best_model),
        "encoder_params": int(sum(p.numel() for p in best_model.parameters() if p.requires_grad)),
        "epoch_origin_receipt_count": len(epoch_origin_receipts),
        "epoch_origin_receipts_sha256": hashlib.sha256(json.dumps(epoch_origin_receipts, sort_keys=True).encode()).hexdigest(),
    }


def load_mlp_encoder_from_fit(bundle, fit_entry: dict):
    pe = parent_encoders()
    path = resolve_path(fit_entry["checkpoint"])
    if sha256_path(path) != fit_entry["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = pe.build_encoder("MLP_POINT", bundle.channels, bundle.latent, int(fit_entry["seed"]))
    model.load_state_dict(payload["encoder"])
    state_hash = model_state_hash(model)
    if state_hash != fit_entry["encoder_state_sha256"]:
        raise RuntimeError(f"encoder state hash mismatch: {path}")
    return model, state_hash


def load_parent_mlp_fit(parent: dict, dataset: str, k: int, seed: int) -> dict:
    parent_dataset = parent["datasets"][dataset]
    fit = parent_dataset["fits"][f"MLP_POINT_seed{int(seed)}"]
    milestone = fit["milestones"]["180"]
    return {
        "id": adapter_id(dataset, "MLP_POINT", k, seed),
        "dataset": dataset,
        "family": "MLP_POINT",
        "k": int(k),
        "seed": int(seed),
        "reused_from_parent": True,
        "parent_fit_id": f"MLP_POINT_seed{int(seed)}",
        "parent_adapter": f"MLP_POINT_seed{int(seed)}_m180",
        "checkpoint": milestone["checkpoint"],
        "checkpoint_sha256": milestone["checkpoint_sha256"],
        "best_epoch": int(milestone["best_epoch"]),
        "val_reconstruction": milestone["val_reconstruction"],
        "encoder_state_sha256": milestone["encoder_state_sha256"],
        "encoder_params": int(fit.get("encoder_params", 0)),
        "curve": fit["curve"],
        "curve_sha256": fit["curve_sha256"],
        "parent_prefix_parity": fit.get("parent_prefix_parity"),
    }


def forecast_latents(backbone, z_history: np.ndarray, protocol: dict, device: str) -> np.ndarray:
    return parent_runner().old_pilot().bolt_median_forecast(backbone, z_history.astype(np.float32), device, int(protocol["forecast_batch"]))


def trace_npz_parent_entry(parent: dict, dataset: str, parent_adapter: str) -> tuple[Path, dict]:
    entry = parent["datasets"][dataset]["decoder_contracts"][parent_adapter]["contracts"][CONTRACT]
    path = resolve_path(entry["prediction_npz"])
    if sha256_path(path) != entry["prediction_sha256"]:
        raise RuntimeError(f"parent trace NPZ changed: {path}")
    return path, entry


def load_reused_traces(parent: dict, dataset: str, parent_adapter: str, bundle, train_origins: np.ndarray, lag: int) -> dict:
    path, entry = trace_npz_parent_entry(parent, dataset, parent_adapter)
    with np.load(path, allow_pickle=False) as archive:
        needed = ["origins", "train_origins", "train_latent_history", "train_latent_forecast", "val_latent_history_tail", "latent_forecast"]
        missing = [key for key in needed if key not in archive.files]
        if missing:
            raise RuntimeError(f"parent trace lacks keys {missing}: {path}")
        if not np.array_equal(archive["origins"], bundle.val_origins):
            raise RuntimeError(f"parent trace VAL origins mismatch: {path}")
        if not np.array_equal(archive["train_origins"], train_origins):
            raise RuntimeError(f"parent trace TRAIN origins mismatch: {path}")
        return {
            "train_latent_history": archive["train_latent_history"].astype(np.float32),
            "train_latent_forecast": archive["train_latent_forecast"].astype(np.float32),
            "val_latent_history_tail": archive["val_latent_history_tail"].astype(np.float32),
            "latent_forecast": archive["latent_forecast"].astype(np.float32),
            "reused_trace": {"path": path, "sha256": entry["prediction_sha256"], "parent_adapter": parent_adapter, "required_lag": int(lag)},
        }


def compute_traces(bundle, encoder_or_basis, backbone, protocol: dict, device: str, train_origins: np.ndarray) -> dict:
    pr = parent_runner()
    lag = int(protocol["decoder_max_lag"])
    train_context, _ = pr.old_pilot().training_reconstruction_windows(bundle, train_origins)
    val_context, _ = pr.old_pilot().windows(bundle, bundle.val_origins)
    z_train = encode_numpy(encoder_or_basis, train_context, device)
    z_val = encode_numpy(encoder_or_basis, val_context, device)
    return {
        "train_latent_history": z_train[:, -lag:].astype(np.float32),
        "train_latent_forecast": forecast_latents(backbone, z_train, protocol, device).astype(np.float32),
        "val_latent_history_tail": z_val[:, -lag:].astype(np.float32),
        "latent_forecast": forecast_latents(backbone, z_val, protocol, device).astype(np.float32),
        "reused_trace": None,
    }


def raw_f0_from_parent(parent: dict, dataset: str) -> dict:
    entry = copy.deepcopy(parent["datasets"][dataset]["raw_f0"])
    path = resolve_path(entry["prediction_npz"])
    if sha256_path(path) != entry["prediction_sha256"]:
        raise RuntimeError(f"parent RAW_F0 changed: {path}")
    entry["reused_from_parent"] = True
    entry["path_verified"] = True
    return entry


def raw_c_traces(bundle, backbone, protocol: dict, device: str, train_origins: np.ndarray, raw_f0: dict) -> dict:
    pr = parent_runner()
    lag = int(protocol["decoder_max_lag"])
    train_context, _ = pr.old_pilot().training_reconstruction_windows(bundle, train_origins)
    val_context, _ = pr.old_pilot().windows(bundle, bundle.val_origins)
    raw_path = resolve_path(raw_f0["prediction_npz"])
    with np.load(raw_path, allow_pickle=False) as archive:
        val_forecast = archive["prediction_tsfm"].astype(np.float32)
    expected_shape = (len(bundle.val_origins), int(protocol["horizon"]), bundle.channels)
    if val_forecast.shape != expected_shape:
        raise RuntimeError(f"parent RAW_F0 shape {val_forecast.shape} != {expected_shape}")
    return {
        "train_latent_history": train_context[:, -lag:].astype(np.float32),
        "train_latent_forecast": forecast_latents(backbone, train_context.astype(np.float32), protocol, device).astype(np.float32),
        "val_latent_history_tail": val_context[:, -lag:].astype(np.float32),
        "latent_forecast": val_forecast,
        "reused_trace": {"path": raw_path, "sha256": raw_f0["prediction_sha256"], "parent_adapter": "RAW_F0", "field": "prediction_tsfm"},
    }


def fit_forecast_decoder_from_traces(bundle, adapter_name: str, traces: dict, protocol: dict, train_origins: np.ndarray) -> dict:
    pr = parent_runner()
    lag = int(protocol["decoder_max_lag"])
    groups = [tuple(v) for v in protocol["horizon_groups"]]
    z_train_tail = traces["train_latent_history"].astype(np.float32)
    z_train_forecast = traces["train_latent_forecast"].astype(np.float32)
    z_val_tail = traces["val_latent_history_tail"].astype(np.float32)
    z_val_forecast = traces["latent_forecast"].astype(np.float32)
    train_features = future_features(z_train_tail, z_train_forecast, lag)
    train_target, train_mask = pr.old_pilot().future_targets(bundle, train_origins)
    train_row_mask = train_mask.any(axis=2)
    val_features = future_features(z_val_tail, z_val_forecast, lag)
    val_future, val_future_mask = pr.old_pilot().future_targets(bundle, bundle.val_origins)
    fitted = fit_scaled_ridge_grid(
        train_features,
        train_target,
        train_mask,
        train_row_mask,
        val_features,
        val_future,
        val_future_mask,
        [float(v) for v in protocol["ridge_lambdas"]],
        horizon_groups=groups,
    )
    coeff = fitted["coeff"]
    z_val_last = np.repeat(z_val_tail[:, -1:, :], int(protocol["horizon"]), axis=1).astype(np.float32)
    pred_tsfm = apply_coeff(future_features(z_val_tail, z_val_forecast, lag), coeff)
    pred_last = apply_coeff(future_features(z_val_tail, z_val_last, lag), coeff)
    taps = coeff_to_taps(coeff, lag, bundle.latent)
    npz_path = CACHE / "predictions" / f"{adapter_name}_{CONTRACT}_val.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        origins=bundle.val_origins,
        train_origins=train_origins,
        train_latent_history=z_train_tail.astype(np.float32),
        train_latent_forecast=z_train_forecast.astype(np.float32),
        val_latent_history_tail=z_val_tail.astype(np.float32),
        latent_forecast=z_val_forecast.astype(np.float32),
        latent_last=z_val_last.astype(np.float32),
        prediction_tsfm=pred_tsfm.astype(np.float32),
        prediction_latent_last=pred_last.astype(np.float32),
        decoder_coeff=coeff.astype(np.float64),
        decoder_taps=taps.astype(np.float64),
        feature_scale=fitted["feature_scale"].astype(np.float64),
        grid_lambdas=fitted["grid_lambdas"].astype(np.float64),
        grid_coefficients=fitted["grid_coefficients"].astype(np.float64),
    )
    return {
        "contract": CONTRACT,
        "selected_lambda": fitted["selected_lambda"],
        "selected_index": fitted["selected_index"],
        "selection_metrics": fitted["selection_metrics"],
        "initial_grid_selection": initial_grid_selection(fitted, protocol["ridge_initial_lambdas"]),
        "choices": fitted["choices"],
        "future_tsfm": pr.horizon_metrics(pred_tsfm, val_future, val_future_mask, groups),
        "future_latent_last": pr.horizon_metrics(pred_last, val_future, val_future_mask, groups),
        "prediction_npz": npz_path,
        "prediction_sha256": sha256_path(npz_path),
        "decoder_taps_sha256": array_sha256(taps.astype(np.float64)),
        "decoder_coeff_sha256": array_sha256(coeff.astype(np.float64)),
        "feature_scale_sha256": array_sha256(fitted["feature_scale"].astype(np.float64)),
        "grid_coefficients_sha256": array_sha256(fitted["grid_coefficients"].astype(np.float64)),
        "near_zero_feature_count": fitted["near_zero_feature_count"],
        "scale_count": fitted["scale_count"],
        "scale_row_mask_sha256": fitted["scale_row_mask_sha256"],
        "trace_reuse": traces.get("reused_trace"),
        "precision": "ridge feature/stats/coeff/taps/scale FP64; final predictions FP32",
    }


def decoder_train_origins(bundle, protocol: dict) -> np.ndarray:
    return parent_runner().decoder_train_origins(bundle.train_origins, int(protocol["decoder_train_origin_count"]), int(protocol["decoder_sampling_seed"]))


def basis_receipt(bundle, kind: str) -> dict:
    path = CACHE / "basis" / f"{bundle.name}_K{bundle.latent}_{kind}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, bundle.basis.astype(np.float32))
    return {"path": path, "sha256": sha256_path(path), "array_sha256": array_sha256(bundle.basis.astype(np.float32)), "kind": kind, "shape": list(bundle.basis.shape)}


def build_dataset_bundles(dataset: str, protocol: dict) -> dict[int, object]:
    original = load_original_bundle(dataset)
    bundles = {}
    for k_value in protocol["k_grid"][dataset]:
        k = int(k_value)
        basis = original.basis if k == original.latent else compute_pca_basis(original.x, original.train_bounds, k)
        bundles[k] = load_bundle(dataset, k, basis=basis)
    bundles[original.channels] = load_bundle(dataset, original.channels, basis=np.eye(original.channels, dtype=np.float32))
    return bundles


def run_dataset(dataset: str, protocol: dict, parent: dict, device: str, partial: dict) -> dict:
    pr = parent_runner()
    bundles = build_dataset_bundles(dataset, protocol)
    original = load_original_bundle(dataset)
    dec_origins = decoder_train_origins(original, protocol)
    result = {
        "dataset": dataset,
        "shape": list(original.x.shape),
        "channels": original.channels,
        "k_grid": [int(v) for v in protocol["k_grid"][dataset]],
        "middle_k": int(protocol["middle_k"][dataset]),
        "source": {"path": original.source_path, "sha256": sha256_path(Path(original.source_path))},
        "basis": {},
        "fits": {},
        "decoder_train_origins": {"count": int(len(dec_origins)), "seed": int(protocol["decoder_sampling_seed"]), "sha256": array_sha256(dec_origins)},
        "decoder_contracts": {},
        "raw_f0": raw_f0_from_parent(parent, dataset),
        "benchmark_cases": [],
    }
    for k, bundle in bundles.items():
        kind = "identity_raw_c" if k == original.channels else ("parent_middle_pca" if k == original.latent else "new_train_centered_pca")
        result["basis"][f"K{k}"] = basis_receipt(bundle, kind)
    middle = int(protocol["middle_k"][dataset])
    for row in adapter_plan(protocol, dataset):
        if row["family"] == "MLP_POINT" and int(row["k"]) != middle:
            fit = train_mlp_new_k(bundles[int(row["k"])], int(row["seed"]), protocol, device)
            result["fits"][fit["id"]] = fit
            partial["datasets"][dataset] = to_jsonable(result)
            atomic_json(HERE / "result.partial.json", partial)
    for seed in protocol["seeds"]:
        fit = load_parent_mlp_fit(parent, dataset, middle, int(seed))
        result["fits"][fit["id"]] = fit

    backbone = pr.old_pilot().load_backbone(dataset, device=device)
    backbone_before = pr.old_pilot().backbone_state_hash(backbone)
    try:
        for row in adapter_plan(protocol, dataset):
            ensure_source_hashes_unchanged(partial["source_hashes"])
            name = row["id"]
            family = row["family"]
            bundle = bundles[int(row["k"])]
            monitor(stage=f"decoder:{name}", dataset=dataset, adapter=name)
            if family == "PCA":
                encoder_obj = bundle.basis
                encoder_state = array_sha256(bundle.basis.astype(np.float32))
                parent_adapter = "PCA_ANCHOR" if int(row["k"]) == middle else None
            elif family == "RAW_C_D8":
                encoder_obj = bundle.basis
                encoder_state = array_sha256(bundle.basis.astype(np.float32))
                parent_adapter = None
            elif family == "MLP_POINT":
                fit = result["fits"][name]
                encoder_obj, encoder_state = load_mlp_encoder_from_fit(bundle, fit)
                parent_adapter = fit.get("parent_adapter") if fit.get("reused_from_parent") else None
            else:
                raise ValueError(family)
            if parent_adapter is not None:
                traces = load_reused_traces(parent, dataset, parent_adapter, bundle, dec_origins, int(protocol["decoder_max_lag"]))
            elif family == "RAW_C_D8":
                traces = raw_c_traces(bundle, backbone, protocol, device, dec_origins, result["raw_f0"])
            else:
                if isinstance(encoder_obj, torch.nn.Module):
                    for parameter in encoder_obj.parameters():
                        parameter.requires_grad_(False)
                    encoder_obj = encoder_obj.to(device).eval()
                traces = compute_traces(bundle, encoder_obj, backbone, protocol, device, dec_origins)
            entry = {
                "meta": copy.deepcopy(row),
                "encoder_state_sha256_before": encoder_state,
                "contracts": {CONTRACT: fit_forecast_decoder_from_traces(bundle, name, traces, protocol, dec_origins)},
            }
            if isinstance(encoder_obj, torch.nn.Module):
                after = model_state_hash(encoder_obj.cpu())
                if after != encoder_state:
                    raise RuntimeError(f"encoder changed during decoder fitting: {name}")
            else:
                after = encoder_state
            entry["encoder_state_sha256_after"] = after
            result["decoder_contracts"][name] = entry
            result["benchmark_cases"] = ["RAW_F0"] + [planned["id"] for planned in adapter_plan(protocol, dataset)]
            partial["datasets"][dataset] = to_jsonable(result)
            atomic_json(HERE / "result.partial.json", partial)
        backbone_after = pr.old_pilot().backbone_state_hash(backbone)
        if backbone_before != backbone_after:
            raise RuntimeError(f"{dataset} frozen backbone changed")
        if not all(p.grad is None for p in backbone.parameters()):
            raise RuntimeError(f"{dataset} frozen backbone accumulated gradients")
        result["backbone"] = {"state_sha256_before": backbone_before, "state_sha256_after": backbone_after, "unchanged": True, "grads_none": True}
        return result
    finally:
        del backbone
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def expected_counts(protocol: dict) -> dict:
    adapters = all_adapter_plan(protocol)
    return {
        "decoder_adapters": len(adapters),
        "ridge_fits": len(adapters) * len(protocol["ridge_lambdas"]),
        "neural_fit_starts": sum(1 for row in adapters if row["family"] == "MLP_POINT" and int(row["k"]) != int(protocol["middle_k"][row["dataset"]])),
        "benchmark_cases": sum((len(adapter_plan(protocol, ds)) + 1) * len(protocol["benchmark"]["batches"]) for ds in protocol["datasets"]),
    }


def assert_fresh_outputs() -> None:
    existing = []
    for path in (HERE / "result.json", HERE / "result.partial.json"):
        if path.exists():
            existing.append(path)
    for folder_name in ("checkpoints", "predictions", "curves", "basis"):
        folder = CACHE / folder_name
        if folder.exists() and any(folder.rglob("*")):
            existing.append(folder)
    if existing:
        raise RuntimeError({"existing_k_tradeoff_outputs_refusing_overwrite": [public_path(p) for p in existing]})


def run(args) -> dict:
    global MONITOR
    configure_runtime()
    protocol = read_json(HERE / "protocol.json")
    assert_fresh_outputs()
    CACHE.mkdir(parents=True, exist_ok=True)
    MONITOR = Monitor(protocol, args.device)
    install_monitor_hooks()
    parent = load_parent(protocol)
    partial = {
        "schema": "k_tradeoff_result_v1",
        "status": "running",
        "protocol": protocol,
        "effective_counts": expected_counts(protocol),
        "parent": {"path": resolve_path(protocol["parent"]), "sha256": sha256_path(resolve_path(protocol["parent"])), "schema": parent.get("schema"), "status": parent.get("status")},
        "source_hashes": source_hashes(),
        "dependency_hashes": dependency_hashes(),
        "datasets": {},
    }
    atomic_json(HERE / "result.partial.json", partial)
    try:
        for dataset in protocol["datasets"]:
            ensure_source_hashes_unchanged(partial["source_hashes"])
            monitor(stage=f"dataset:{dataset}", dataset=dataset)
            partial["datasets"][dataset] = to_jsonable(run_dataset(dataset, protocol, parent, args.device, partial))
            atomic_json(HERE / "result.partial.json", partial)
        partial["status"] = "complete"
        partial["finished_utc"] = time.time()
        MONITOR.progress["status"] = "complete"
        MONITOR.progress["stage"] = "complete"
        partial["execution_counts"] = MONITOR.receipt()
        monitor(stage="complete", status="complete")
        atomic_json(HERE / "result.json", partial)
        return partial
    except Exception as exc:
        partial["status"] = "failed"
        partial["failed_utc"] = time.time()
        partial["error"] = {"type": type(exc).__name__, "message": str(exc)}
        partial["execution_counts"] = MONITOR.receipt() if MONITOR is not None else {}
        if MONITOR is not None:
            MONITOR.progress["status"] = "failed"
            MONITOR.progress["error"] = partial["error"]
            atomic_json(HERE / "progress.json", MONITOR.progress)
        atomic_json(HERE / "result.partial.json", partial)
        raise


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

