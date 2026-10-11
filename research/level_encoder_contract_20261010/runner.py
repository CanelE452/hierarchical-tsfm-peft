"""Runner for the same-decoder encoder contract study.

The script writes only under this study folder and .cache folder. Root runs the
scientific CUDA job; tests import the small CPU helpers below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import psutil
import torch

import encoders


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
OLD_PILOT = ROOT / "research/level_lag_compressor_pilot_20261010/pilot.py"
PARENT_RESULT = ROOT / "research/level_lag_compressor_pilot_20261010/result.json"
MONITOR = None


def import_file(path: Path, module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def old_pilot():
    module = import_file(OLD_PILOT, "_encoder_contract_old_pilot")
    if MONITOR is not None:
        module.MONITOR = MONITOR
    return module


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value).replace(str(ROOT) + os.sep, "").replace(ROOT.as_posix() + "/", "")
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
        raise PermissionError(f"refusing to write outside encoder-contract folders: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def module_state_hash(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        arr = value.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Monitor:
    def __init__(self, protocol: dict, datasets: list[str], device: str):
        self.limits = protocol["limits"]
        self.started_wall = time.perf_counter()
        self.started_cpu = time.process_time()
        self.process = psutil.Process()
        self.device = device
        self.base_bytes = self.storage_bytes()
        self.last_disk_check = 0.0
        self.progress = {
            "pid": os.getpid(),
            "datasets": datasets,
            "fit_starts": 0,
            "completed_fits": 0,
            "ridge_fits": 0,
            "started_utc": time.time(),
        }
        self.peak_rss = 0
        self.peak_allocated = 0
        self.peak_reserved = 0

    def storage_bytes(self) -> int:
        total = 0
        for folder in (HERE, CACHE):
            if folder.exists():
                total += sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        return int(total)

    def check(self, stage: str | None = None, **fields) -> None:
        elapsed = time.perf_counter() - self.started_wall
        cpu_seconds = time.process_time() - self.started_cpu
        rss = self.process.memory_info().rss
        allocated = torch.cuda.memory_allocated(torch.device(self.device)) if self.device.startswith("cuda") and torch.cuda.is_available() else 0
        reserved = torch.cuda.memory_reserved(torch.device(self.device)) if self.device.startswith("cuda") and torch.cuda.is_available() else 0
        self.peak_rss = max(self.peak_rss, int(rss))
        self.peak_allocated = max(self.peak_allocated, int(allocated))
        self.peak_reserved = max(self.peak_reserved, int(reserved))
        if elapsed > self.limits["wall_seconds"]:
            raise RuntimeError("STOP_RESOURCE: wall_seconds")
        if cpu_seconds > self.limits["process_cpu_seconds"]:
            raise RuntimeError("STOP_RESOURCE: process_cpu_seconds")
        if rss > self.limits["ram_bytes"] or allocated > self.limits["allocated_vram_bytes"]:
            raise RuntimeError("STOP_RESOURCE: memory")
        if elapsed - self.last_disk_check > 10.0:
            if self.storage_bytes() - self.base_bytes > self.limits["incremental_storage_bytes"]:
                raise RuntimeError("STOP_RESOURCE: incremental_storage_bytes")
            self.last_disk_check = elapsed
        if stage is not None:
            self.progress.update(stage=stage, elapsed_seconds=elapsed, cpu_seconds=cpu_seconds,
                                 ram_bytes=int(rss), allocated_vram_bytes=int(allocated), **fields)
            atomic_json(HERE / "progress.json", self.progress)

    def receipt(self) -> dict:
        return {
            "wall_seconds": float(time.perf_counter() - self.started_wall),
            "process_cpu_seconds": float(time.process_time() - self.started_cpu),
            "peak_ram_bytes": int(self.peak_rss),
            "peak_allocated_vram_bytes": int(self.peak_allocated),
            "peak_reserved_vram_bytes": int(self.peak_reserved),
            "fit_starts": int(self.progress["fit_starts"]),
            "completed_fits": int(self.progress["completed_fits"]),
            "ridge_fits": int(self.progress["ridge_fits"]),
            "incremental_storage_bytes": int(self.storage_bytes() - self.base_bytes),
        }


def monitor(stage: str | None = None, **fields) -> None:
    if MONITOR is not None:
        MONITOR.check(stage, **fields)


def build_train_graph(x: np.ndarray, finite: np.ndarray, train_bounds: tuple[int, int], top_k: int = 3) -> np.ndarray:
    start, end = train_bounds
    train_x = x[start:end].astype(np.float64, copy=False)
    train_m = finite[start:end].astype(bool, copy=False)
    channels = train_x.shape[1]
    corr = np.zeros((channels, channels), dtype=np.float64)
    for i in range(channels):
        for j in range(channels):
            if i == j:
                continue
            mask = train_m[:, i] & train_m[:, j]
            if int(mask.sum()) < 3:
                continue
            xi = train_x[mask, i] - train_x[mask, i].mean()
            xj = train_x[mask, j] - train_x[mask, j].mean()
            denom = float(np.sqrt(np.dot(xi, xi) * np.dot(xj, xj)))
            if denom > 0:
                corr[i, j] = abs(float(np.dot(xi, xj) / denom))
    graph = np.zeros_like(corr, dtype=np.float32)
    k = max(0, min(int(top_k), channels - 1))
    for i in range(channels):
        order = np.argsort(-corr[i], kind="stable")
        chosen = [j for j in order if j != i and corr[i, j] > 0][:k]
        if chosen:
            weights = corr[i, chosen]
            graph[i, chosen] = (weights / weights.sum()).astype(np.float32)
    np.fill_diagonal(graph, 0.0)
    return graph


def masked_channel_macro_metrics_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    if pred.shape != target.shape or pred.shape != mask.shape or pred.ndim != 3:
        raise ValueError("pred, target, mask must share [B,T,C]")
    observed = mask.astype(bool)
    if not np.isfinite(pred).all() or not np.isfinite(target[observed]).all():
        raise ValueError("nonfinite prediction or observed target")
    error = np.where(observed, pred.astype(np.float64) - target.astype(np.float64), 0.0)
    count = observed.sum(axis=(0, 1)).astype(np.float64)
    valid = count > 0
    if not np.any(valid):
        raise ValueError("no observed targets")
    sq = np.square(error).sum(axis=(0, 1))
    ab = np.abs(error).sum(axis=(0, 1))
    signed = error.sum(axis=(0, 1))
    return {
        "mse": float(np.mean(sq[valid] / count[valid])),
        "mae": float(np.mean(ab[valid] / count[valid])),
        "signed_mean_error": float(np.mean(signed[valid] / count[valid])),
        "channel_mse": (sq / np.maximum(count, 1)).tolist(),
        "channel_mae": (ab / np.maximum(count, 1)).tolist(),
        "target_counts": count.astype(np.int64).tolist(),
        "target_count": int(count.sum()),
        "origins": int(pred.shape[0]),
    }


def horizon_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, groups: Iterable[tuple[int, int]]) -> dict:
    out = {"all": masked_channel_macro_metrics_np(pred, target, mask)}
    for start, end in groups:
        out[f"h{start}_{end}"] = masked_channel_macro_metrics_np(pred[:, start - 1:end], target[:, start - 1:end], mask[:, start - 1:end])
    return out


def masked_channel_macro_mse_torch(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    observed = mask.bool()
    error = torch.where(observed, pred - target, torch.zeros_like(pred))
    count = observed.sum((0, 1)).to(pred.dtype)
    valid = count > 0
    if not bool(valid.any()):
        raise ValueError("no observed targets")
    return (error.square().sum((0, 1))[valid] / count[valid]).mean()


def sample_epoch_origins(train_origins: np.ndarray, seed: int, epoch: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    sampled = None
    for _ in range(int(epoch) + 1):
        sampled = rng.choice(train_origins, size=int(count), replace=True)
    return np.asarray(sampled, dtype=np.int64)


def decoder_train_origins(train_origins: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(train_origins) < count:
        raise ValueError("not enough TRAIN origins for without-replacement decoder sample")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(train_origins, size=int(count), replace=False)).astype(np.int64)


def reconstruction_features(z: np.ndarray, burn_in: int, lag: int) -> np.ndarray:
    parts = [z[:, burn_in - ell:z.shape[1] - ell, :] for ell in range(lag + 1)]
    return np.concatenate(parts, axis=2).astype(np.float64)


def future_features(z_history: np.ndarray, z_future: np.ndarray, lag: int) -> np.ndarray:
    prefix = z_history[:, -lag:] if lag else z_history[:, :0]
    if lag and prefix.shape[1] < lag:
        pad = np.zeros((z_history.shape[0], lag - prefix.shape[1], z_history.shape[2]), dtype=z_history.dtype)
        prefix = np.concatenate([pad, prefix], axis=1)
    full = np.concatenate([prefix, z_future], axis=1)
    parts = [full[:, lag - ell:lag - ell + z_future.shape[1], :] for ell in range(lag + 1)]
    return np.concatenate(parts, axis=2).astype(np.float64)


def apply_coeff(features: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return np.einsum("btd,dc->btc", features.astype(np.float64), coeff.astype(np.float64)).astype(np.float32)


def coeff_to_taps(coeff: np.ndarray, lag: int, latent: int) -> np.ndarray:
    return coeff.reshape(lag + 1, latent, coeff.shape[1]).astype(np.float64)


def fit_scaled_ridge(
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
    train_row_mask: np.ndarray,
    val_features: np.ndarray,
    val_target: np.ndarray,
    val_mask: np.ndarray,
    lambdas: list[float],
    horizon_groups: list[tuple[int, int]] | None = None,
) -> dict:
    if train_features.shape[:2] != train_target.shape[:2] or train_target.shape != train_mask.shape:
        raise ValueError("train feature/target/mask shapes differ")
    rows = train_row_mask.astype(bool).reshape(-1)
    flat_features = train_features.reshape(-1, train_features.shape[-1])
    if not np.any(rows):
        raise ValueError("no TRAIN rows available for decoder feature scaling")
    rms = np.sqrt(np.mean(np.square(flat_features[rows]), axis=0))
    near_zero = rms <= 1e-12
    scale = np.maximum(rms, 1e-12)
    x_train = train_features / scale
    dim = train_features.shape[-1]
    channels = train_target.shape[-1]
    xtx = np.zeros((channels, dim, dim), dtype=np.float64)
    xty = np.zeros((channels, dim), dtype=np.float64)
    channel_counts = np.zeros(channels, dtype=np.int64)
    for channel in range(channels):
        monitor()
        obs = train_mask[:, :, channel].astype(bool) & train_row_mask.astype(bool)
        if not np.any(obs):
            continue
        x = x_train[obs]
        y = train_target[:, :, channel][obs].astype(np.float64)
        xtx[channel] = x.T @ x
        xty[channel] = x.T @ y
        channel_counts[channel] = int(obs.sum())
    choices = []
    best = None
    for lam in lambdas:
        monitor()
        if MONITOR is not None:
            if MONITOR.progress["ridge_fits"] >= MONITOR.limits["ridge_fits"]:
                raise RuntimeError("STOP_RESOURCE: ridge_fits")
            MONITOR.progress["ridge_fits"] += 1
        coeff_scaled = np.zeros((dim, channels), dtype=np.float64)
        for channel in range(channels):
            monitor()
            if channel_counts[channel] == 0:
                continue
            lhs = xtx[channel] + float(lam) * np.eye(dim, dtype=np.float64)
            coeff_scaled[:, channel] = np.linalg.solve(lhs, xty[channel])
        coeff = coeff_scaled / scale[:, None]
        pred = apply_coeff(val_features, coeff)
        metrics = horizon_metrics(pred, val_target, val_mask, horizon_groups) if horizon_groups else {
            "all": masked_channel_macro_metrics_np(pred, val_target, val_mask)
        }
        choice = {"lambda": float(lam), "metrics": metrics, "coeff": coeff}
        choices.append(choice)
        if best is None or metrics["all"]["mse"] < best["metrics"]["all"]["mse"]:
            best = choice
    row_mask_hash = array_sha256(train_row_mask.astype(np.uint8))
    return {
        "coeff": best["coeff"],
        "selected_lambda": best["lambda"],
        "selection_metrics": best["metrics"],
        "choices": [{"lambda": c["lambda"], "metrics": c["metrics"]} for c in choices],
        "feature_scale": scale.astype(np.float64),
        "near_zero_feature_count": int(near_zero.sum()),
        "scale_count": int(rows.sum()),
        "scale_row_mask_sha256": row_mask_hash,
    }


def load_dataset(dataset: str):
    pilot = old_pilot()
    bundle = pilot.load_dataset(dataset)
    pilot.assert_no_test_origins(bundle, bundle.train_origins)
    pilot.assert_no_test_origins(bundle, bundle.val_origins)
    return bundle


@torch.no_grad()
def encode_numpy(encoder_or_basis, context: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    if isinstance(encoder_or_basis, np.ndarray):
        return (context @ encoder_or_basis).astype(np.float32)
    encoder_or_basis.eval()
    chunks = []
    for start in range(0, len(context), batch_size):
        monitor()
        x = torch.from_numpy(context[start:start + batch_size].astype(np.float32)).to(device)
        chunks.append(encoder_or_basis.encode(x).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def eval_reconstruction(encoder, decoder, bundle, origins: np.ndarray, burn_in: int, device: str, batch_size: int) -> dict:
    pilot = old_pilot()
    preds = []
    targets = []
    masks = []
    encoder.eval()
    decoder.eval()
    for start in range(0, len(origins), batch_size):
        monitor()
        batch_origins = origins[start:start + batch_size]
        x_np, m_np = pilot.windows(bundle, batch_origins)
        x = torch.from_numpy(x_np).to(device)
        pred = decoder.decode(encoder.encode(x)).detach().cpu().numpy().astype(np.float32)
        preds.append(pred)
        targets.append(x_np)
        masks.append(m_np)
    return {
        "metrics": masked_channel_macro_metrics_np(
            np.concatenate(preds, axis=0)[:, burn_in:],
            np.concatenate(targets, axis=0)[:, burn_in:],
            np.concatenate(masks, axis=0)[:, burn_in:],
        ),
    }


def fit_encoder_decoder(bundle, family: str, seed: int, graph: np.ndarray, protocol: dict, device: str) -> dict:
    pilot = old_pilot()
    monitor(stage=f"fit:{bundle.name}:{family}:{seed}", dataset=bundle.name, family=family, seed=seed)
    if MONITOR is not None:
        if MONITOR.progress["fit_starts"] >= MONITOR.limits["neural_fit_starts"]:
            raise RuntimeError("STOP_RESOURCE: neural_fit_starts")
        MONITOR.progress["fit_starts"] += 1
    encoder = encoders.build_encoder(family, bundle.channels, bundle.latent, seed, graph=graph).to(device)
    decoder = encoders.build_decoder(bundle.channels, bundle.latent, seed, lag=int(protocol["decoder_max_lag"])).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(protocol["learning_rate"]), weight_decay=0.0)
    burn_in = int(protocol["burn_in"])
    batch_size = int(protocol["batch"])
    curves = []
    checkpoint_path = CACHE / "checkpoints" / f"{bundle.name}_{family}_{seed}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(epoch: int) -> dict:
        rec = eval_reconstruction(encoder, decoder, bundle, bundle.val_origins, burn_in, device, batch_size)
        row = {"epoch": int(epoch), "val_reconstruction": rec["metrics"]}
        curves.append(row)
        return row

    best = evaluate(0)
    best_epoch = 0
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "family": family,
                "seed": int(seed), "epoch": 0, "val_reconstruction": best["val_reconstruction"]}, checkpoint_path)
    stale = 0
    for epoch in range(1, int(protocol["max_epochs"]) + 1):
        sample = sample_epoch_origins(bundle.train_origins, seed, epoch - 1, int(protocol["epoch_origins"]))
        order = np.arange(len(sample))
        encoder.train()
        decoder.train()
        train_losses = []
        for start in range(0, len(sample), batch_size):
            monitor()
            origins = sample[order[start:start + batch_size]]
            x_np, m_np = pilot.training_reconstruction_windows(bundle, origins)
            x = torch.from_numpy(x_np).to(device)
            mask = torch.from_numpy(m_np[:, burn_in:]).to(device)
            if not bool(mask.any()):
                continue
            pred = decoder.decode(encoder.encode(x))[:, burn_in:]
            target = x[:, burn_in:]
            loss = masked_channel_macro_mse_torch(pred, target, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), float(protocol["clip_grad_norm"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        row = evaluate(epoch)
        row["train_loss_mean"] = float(np.mean(train_losses)) if train_losses else None
        if row["val_reconstruction"]["mse"] < best["val_reconstruction"]["mse"]:
            best = row
            best_epoch = epoch
            stale = 0
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "family": family,
                        "seed": int(seed), "epoch": epoch, "val_reconstruction": best["val_reconstruction"]}, checkpoint_path)
        else:
            stale += 1
        curve_path = CACHE / "curves" / f"{bundle.name}_{family}_{seed}.json"
        atomic_json(curve_path, {"dataset": bundle.name, "family": family, "seed": int(seed), "curves": curves})
        if stale >= int(protocol["patience"]):
            break
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(payload["encoder"])
    decoder.load_state_dict(payload["decoder"])
    if MONITOR is not None:
        MONITOR.progress["completed_fits"] += 1
    train_hash = module_state_hash(encoder)
    return {
        "id": f"{family}_seed{seed}",
        "family": family,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "curve": CACHE / "curves" / f"{bundle.name}_{family}_{seed}.json",
        "val_reconstruction": best["val_reconstruction"],
        "encoder_params": encoders.count_parameters(encoder),
        "decoder_params": encoders.count_parameters(decoder),
        "encoder_state_sha256": train_hash,
        "encoder": encoder.cpu(),
    }


def latent_train_statistics(bundle, encoder_or_basis, device: str, burn_in: int, origins: np.ndarray, batch_size: int = 64) -> dict:
    pilot = old_pilot()
    total = None
    total_sq = None
    count = 0
    for start in range(0, len(origins), batch_size):
        monitor()
        batch_origins = origins[start:start + batch_size]
        context, _ = pilot.windows(bundle, batch_origins)
        z = encode_numpy(encoder_or_basis, context, device, batch_size=batch_size)
        timestamps = batch_origins[:, None] - 512 + np.arange(512)[None, :]
        selected = (timestamps >= bundle.train_bounds[0]) & (timestamps < bundle.train_bounds[1]) & (np.arange(512)[None, :] >= burn_in)
        rows = z[selected].astype(np.float64)
        if rows.size == 0:
            continue
        total = rows.sum(axis=0) if total is None else total + rows.sum(axis=0)
        total_sq = np.square(rows).sum(axis=0) if total_sq is None else total_sq + np.square(rows).sum(axis=0)
        count += int(rows.shape[0])
    if total is None:
        raise RuntimeError("no latent TRAIN rows")
    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), 0.0)
    return {
        "count": int(count),
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
        "condition_norm": float(np.linalg.cond(np.diag(np.sqrt(var) + 1e-12))),
        "scope": "decoder TRAIN 512 origins only; TRAIN timestamps after burn-in",
        "origins_sha256": array_sha256(origins),
    }


def forecast_latents(backbone, z_history: np.ndarray, protocol: dict, device: str) -> np.ndarray:
    pilot = old_pilot()
    return pilot.bolt_median_forecast(backbone, z_history.astype(np.float32), device, int(protocol["forecast_batch"]))


def fit_decoder_contract(
    bundle,
    adapter_id: str,
    encoder_or_basis,
    backbone,
    protocol: dict,
    device: str,
    origins: np.ndarray,
    contract: str,
) -> dict:
    pilot = old_pilot()
    burn_in = int(protocol["burn_in"])
    lag = int(protocol["decoder_max_lag"])
    groups = [tuple(v) for v in protocol["horizon_groups"]]
    lambdas = [float(v) for v in protocol["ridge_lambdas"]]
    train_context, train_context_mask = pilot.training_reconstruction_windows(bundle, origins)
    z_train = encode_numpy(encoder_or_basis, train_context, device)
    val_context, val_context_mask = pilot.windows(bundle, bundle.val_origins)
    z_val = encode_numpy(encoder_or_basis, val_context, device)
    z_val_forecast = forecast_latents(backbone, z_val, protocol, device)
    z_val_last = np.repeat(z_val[:, -1:, :], int(protocol["horizon"]), axis=1).astype(np.float32)
    val_future, val_future_mask = pilot.future_targets(bundle, bundle.val_origins)

    if contract == "RECON_RIDGE_D8":
        train_features = reconstruction_features(z_train, burn_in, lag)
        train_target = train_context[:, burn_in:].astype(np.float32)
        train_mask = train_context_mask[:, burn_in:].astype(bool)
        train_row_mask = train_mask.any(axis=2)
        val_features = reconstruction_features(z_val, burn_in, lag)
        val_target = val_context[:, burn_in:].astype(np.float32)
        val_mask = val_context_mask[:, burn_in:].astype(bool)
        fitted = fit_scaled_ridge(train_features, train_target, train_mask, train_row_mask,
                                  val_features, val_target, val_mask, lambdas)
    elif contract == "FORECAST_RIDGE_D8":
        z_train_forecast = forecast_latents(backbone, z_train, protocol, device)
        train_features = future_features(z_train, z_train_forecast, lag)
        train_target, train_mask = pilot.future_targets(bundle, origins)
        train_row_mask = train_mask.any(axis=2)
        val_features = future_features(z_val, z_val_forecast, lag)
        fitted = fit_scaled_ridge(train_features, train_target, train_mask, train_row_mask,
                                  val_features, val_future, val_future_mask, lambdas, horizon_groups=groups)
    else:
        raise ValueError(contract)
    coeff = fitted["coeff"]
    future_pred = apply_coeff(future_features(z_val, z_val_forecast, lag), coeff)
    last_pred = apply_coeff(future_features(z_val, z_val_last, lag), coeff)
    future_metrics = horizon_metrics(future_pred, val_future, val_future_mask, groups)
    last_metrics = horizon_metrics(last_pred, val_future, val_future_mask, groups)
    taps = coeff_to_taps(coeff, lag, bundle.latent)
    npz_path = CACHE / "predictions" / f"{bundle.name}_{adapter_id}_{contract}_val.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        origins=bundle.val_origins,
        train_origins=origins,
        latent_forecast=z_val_forecast,
        latent_last=z_val_last,
        prediction_tsfm=future_pred,
        prediction_latent_last=last_pred,
        decoder_taps=taps,
        decoder_coeff=coeff.astype(np.float64),
        feature_scale=fitted["feature_scale"].astype(np.float64),
    )
    return {
        "contract": contract,
        "selected_lambda": fitted["selected_lambda"],
        "selection_metrics": fitted["selection_metrics"],
        "choices": fitted["choices"],
        "future_tsfm": future_metrics,
        "future_latent_last": last_metrics,
        "prediction_npz": npz_path,
        "prediction_sha256": sha256_path(npz_path),
        "decoder_taps_sha256": array_sha256(taps),
        "feature_scale_sha256": array_sha256(fitted["feature_scale"].astype(np.float64)),
        "near_zero_feature_count": fitted["near_zero_feature_count"],
        "scale_count": fitted["scale_count"],
        "scale_row_mask_sha256": fitted["scale_row_mask_sha256"],
        "scale_scope": "TRAIN decoder feature rows only; rows require at least one observed target under this contract; no VAL/TEST scaling",
        "fitted_coefficients_are_rescaled_to_original_feature_coordinates": True,
    }


def run_dataset(bundle, protocol: dict, args, partial: dict) -> dict:
    configure_runtime()
    device = args.device
    graph = build_train_graph(bundle.x, bundle.finite, bundle.train_bounds, top_k=3)
    graph_path = CACHE / "graphs" / f"{bundle.name}_train_abspearson_top3.npy"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(graph_path, graph)
    result = {
        "dataset": bundle.name,
        "shape": list(bundle.x.shape),
        "channels": bundle.channels,
        "latent": bundle.latent,
        "source": {"path": bundle.source_path, "sha256": sha256_path(Path(bundle.source_path))},
        "origins": {
            "train_count": int(len(bundle.train_origins)),
            "val_count": int(len(bundle.val_origins)),
            "train_sha256": array_sha256(bundle.train_origins),
            "val_sha256": array_sha256(bundle.val_origins),
        },
        "graph": {"path": graph_path, "sha256": sha256_path(graph_path), "row_sums": graph.sum(axis=1).tolist()},
        "neural": {},
        "decoder_train_origins": {},
        "decoder_contracts": {},
    }
    dec_origins = decoder_train_origins(bundle.train_origins, int(protocol["decoder_train_origin_count"]), int(protocol["decoder_sampling_seed"]))
    result["decoder_train_origins"] = {
        "count": int(len(dec_origins)),
        "seed": int(protocol["decoder_sampling_seed"]),
        "sha256": array_sha256(dec_origins),
        "sampling": "sorted uniform without replacement from canonical TRAIN origins",
    }

    selected = []
    fit_limit = args.limitfits if args.limitfits is not None else protocol["limits"]["neural_fit_starts"]
    for family in protocol["families"]:
        for seed in protocol["seeds"]:
            if len(selected) >= fit_limit:
                break
            fit = fit_encoder_decoder(bundle, family, int(seed), graph, protocol, device)
            encoder = fit.pop("encoder")
            fit["latent_train_statistics"] = latent_train_statistics(bundle, encoder.to(device), device, int(protocol["burn_in"]), dec_origins)
            result["neural"][fit["id"]] = fit
            selected.append((fit["id"], encoder.cpu(), fit["encoder_state_sha256"]))
            partial["datasets"][bundle.name] = to_jsonable(result)
            atomic_json(HERE / "result.partial.json", partial)
        if len(selected) >= fit_limit:
            break

    ensure_source_hashes_unchanged(partial["source_hashes"])
    backbone = old_pilot().load_backbone(bundle.name, device=device)
    backbone_before = old_pilot().backbone_state_hash(backbone)
    val_context, _ = old_pilot().windows(bundle, bundle.val_origins)
    pca_val_history = val_context @ bundle.basis
    pca_val_forecast = forecast_latents(backbone, pca_val_history.astype(np.float32), protocol, device)
    pca_parity = old_pilot().pca_val_cache_parity(bundle, pca_val_forecast)
    if not pca_parity["passed"]:
        raise RuntimeError(f"{bundle.name} PCA VAL cache parity failed")
    result["pca_full_val_parity"] = pca_parity

    adapters = [("PCA_ANCHOR", bundle.basis, array_sha256(bundle.basis))]
    adapters.extend(selected)
    for adapter_id, adapter, state_hash in adapters:
        monitor(stage=f"decoder:{bundle.name}:{adapter_id}")
        result["decoder_contracts"][adapter_id] = {
            "encoder_state_sha256_before": state_hash,
            "latent_train_statistics": latent_train_statistics(bundle, adapter.to(device) if isinstance(adapter, torch.nn.Module) else adapter,
                                                               device, int(protocol["burn_in"]), dec_origins),
            "contracts": {},
        }
        if isinstance(adapter, torch.nn.Module):
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)
            adapter = adapter.to(device)
        for contract in protocol["decoder_contracts"]:
            result["decoder_contracts"][adapter_id]["contracts"][contract] = fit_decoder_contract(
                bundle, adapter_id, adapter, backbone, protocol, device, dec_origins, contract
            )
            partial["datasets"][bundle.name] = to_jsonable(result)
            atomic_json(HERE / "result.partial.json", partial)
        result["decoder_contracts"][adapter_id]["encoder_state_sha256_after"] = (
            module_state_hash(adapter.cpu()) if isinstance(adapter, torch.nn.Module) else state_hash
        )
    backbone_after = old_pilot().backbone_state_hash(backbone)
    backbone_grads_none = all(parameter.grad is None for parameter in backbone.parameters())
    if backbone_before != backbone_after:
        raise RuntimeError(f"{bundle.name} frozen backbone state changed")
    if not backbone_grads_none:
        raise RuntimeError(f"{bundle.name} frozen backbone has gradients")
    result["backbone"] = {
        "state_sha256_before": backbone_before,
        "state_sha256_after": backbone_after,
        "unchanged": backbone_before == backbone_after,
        "grads_none": backbone_grads_none,
    }
    del backbone
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def source_hashes() -> dict:
    paths = {
        "PLAN": HERE / "PLAN.md",
        "protocol": HERE / "protocol.json",
        "runner": HERE / "runner.py",
        "encoders": HERE / "encoders.py",
        "test_contract": HERE / "test_contract.py",
        "old_pilot": OLD_PILOT,
    }
    if PARENT_RESULT.exists():
        paths["parent_result"] = PARENT_RESULT
    return {name: {"path": path, "sha256": sha256_path(path)} for name, path in paths.items() if path.exists()}


def ensure_source_hashes_unchanged(reference: dict) -> None:
    current = source_hashes()
    for name, entry in reference.items():
        if name not in current or current[name]["sha256"] != entry["sha256"]:
            raise RuntimeError(f"source changed during run: {name}")


def run(args) -> dict:
    global MONITOR
    protocol = read_json(HERE / "protocol.json")
    CACHE.mkdir(parents=True, exist_ok=True)
    configure_runtime()
    MONITOR = Monitor(protocol, list(args.datasets), args.device)
    partial = {
        "schema": "same_decoder_encoder_contract_result_v1",
        "protocol": protocol,
        "source_hashes": source_hashes(),
        "datasets": {},
        "status": "running",
    }
    atomic_json(HERE / "result.partial.json", partial)
    try:
        for dataset in args.datasets:
            ensure_source_hashes_unchanged(partial["source_hashes"])
            monitor(stage=f"load:{dataset}")
            bundle = load_dataset(dataset)
            partial["datasets"][dataset] = to_jsonable(run_dataset(bundle, protocol, args, partial))
            atomic_json(HERE / "result.partial.json", partial)
        partial["status"] = "complete"
        partial["finished_utc"] = time.time()
        partial["execution_counts"] = MONITOR.receipt()
        monitor(stage="complete", status="complete")
        atomic_json(HERE / "result.json", partial)
        return partial
    except Exception as exc:
        partial["status"] = "failed"
        partial["failed_utc"] = time.time()
        partial["error"] = {"type": type(exc).__name__, "message": str(exc)}
        partial["execution_counts"] = MONITOR.receipt()
        if MONITOR is not None:
            MONITOR.progress["status"] = "failed"
            MONITOR.progress["error"] = partial["error"]
            MONITOR.progress["execution_counts"] = partial["execution_counts"]
            atomic_json(HERE / "progress.json", MONITOR.progress)
        atomic_json(HERE / "result.partial.json", partial)
        raise


def parse_args(argv: list[str] | None = None):
    protocol = read_json(HERE / "protocol.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=protocol["datasets"])
    parser.add_argument("--limitfits", type=int, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
