"""Small TRAIN/VAL lag-compressor pilot for LEVEL.

This script is intentionally direct. It reuses the sealed v18 train/val
archives, writes only under this research folder and its matching cache, and
does not open TEST.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import psutil
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
RUNTIME_V18 = ROOT / "research/level_matched_learned_ed_v18_20261006/runtime_v18.py"
READOUT_LOADER = ROOT / "research/tsfm_readout_adaptation_screen_20261007/model_readout.py"
V15_CACHE_MANIFEST = ROOT / "research/tsfm_peft_coordinate_decoder_v15_20261001/cache_manifest.json"
MONITOR = None


class PilotMonitor:
    def __init__(self, protocol, datasets, device):
        self.limits = protocol["limits"]
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.process = psutil.Process()
        self.device = device
        self.progress = {"pid": os.getpid(), "datasets": datasets, "completed_fits": 0,
                         "fit_starts": 0, "ridge_fits": 0, "started_utc": time.time()}
        self.base_bytes = self.storage_bytes()
        self.last_disk_check = 0.0

    def storage_bytes(self):
        return sum(p.stat().st_size for folder in (HERE, CACHE) if folder.exists()
                   for p in folder.rglob("*") if p.is_file())

    def check(self, stage=None, **fields):
        elapsed = time.perf_counter() - self.started
        cpu_seconds = time.process_time() - self.cpu_started
        rss = self.process.memory_info().rss
        allocated = torch.cuda.memory_allocated(self.device) if self.device.startswith("cuda") else 0
        if elapsed > self.limits["gpu_job_wall_seconds"]:
            raise RuntimeError("STOP_RESOURCE: pilot wall limit")
        if cpu_seconds > self.limits["cpu_other_seconds"]:
            raise RuntimeError("STOP_RESOURCE: process CPU-time limit")
        if rss > self.limits["ram_bytes"] or allocated > self.limits["allocated_vram_bytes"]:
            raise RuntimeError("STOP_RESOURCE: process RAM or allocated VRAM limit")
        if elapsed - self.last_disk_check > 10:
            if self.storage_bytes() - self.base_bytes > self.limits["incremental_storage_bytes"]:
                raise RuntimeError("STOP_RESOURCE: incremental storage limit")
            self.last_disk_check = elapsed
        if stage is not None:
            self.progress.update(stage=stage, elapsed_seconds=elapsed, cpu_seconds=cpu_seconds,
                                 ram_bytes=rss, allocated_vram_bytes=allocated, **fields)
            atomic_write_json(HERE / "progress.json", self.progress)


def monitor(stage=None, **fields):
    if MONITOR is not None:
        MONITOR.check(stage, **fields)


@dataclass
class DatasetBundle:
    name: str
    x: np.ndarray
    finite: np.ndarray
    basis: np.ndarray
    columns: list[str]
    train_origins: np.ndarray
    val_origins: np.ndarray
    train_bounds: tuple[int, int]
    val_bounds: tuple[int, int]
    contract: dict
    source_path: str

    @property
    def channels(self) -> int:
        return int(self.x.shape[1])

    @property
    def latent(self) -> int:
        return int(self.basis.shape[1])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def public_path(path: Path | str) -> str:
    text = str(path)
    root = str(ROOT)
    return text.replace(root + os.sep, "").replace(ROOT.as_posix() + "/", "")


def to_jsonable(value):
    if isinstance(value, Path):
        return public_path(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def atomic_write_json(path: Path, value) -> None:
    path = path.resolve()
    if not (path.is_relative_to(HERE.resolve()) or path.is_relative_to(CACHE.resolve())):
        raise PermissionError(f"Refusing to write outside pilot folders: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


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


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def reset_cuda_peak(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def resource_receipt(device: str, started: float) -> dict:
    receipt = {"wall_seconds": float(time.perf_counter() - started)}
    if device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device)
        receipt.update(
            cuda_max_allocated_bytes=int(torch.cuda.max_memory_allocated(dev)),
            cuda_max_reserved_bytes=int(torch.cuda.max_memory_reserved(dev)),
        )
    else:
        receipt.update(cuda_max_allocated_bytes=None, cuda_max_reserved_bytes=None)
    return receipt


def causal_fir_apply(x: torch.Tensor, taps: torch.Tensor, prefix: torch.Tensor | None = None) -> torch.Tensor:
    """Apply causal FIR taps to [B,T,I] input.

    taps[lag] maps input channels to output channels. Optional prefix provides
    the historical tail immediately before x, in chronological order.
    """
    if x.ndim != 3 or taps.ndim != 3:
        raise ValueError("x must be [B,T,I] and taps must be [lag,I,O]")
    lag_count, in_channels, _ = taps.shape
    if x.shape[2] != in_channels:
        raise ValueError("tap input channels do not match x")
    q = lag_count - 1
    if prefix is None:
        prefix = x.new_zeros((x.shape[0], q, in_channels))
    else:
        if prefix.ndim != 3 or prefix.shape[0] != x.shape[0] or prefix.shape[2] != in_channels:
            raise ValueError("prefix must be [B,T_prefix,I] with matching batch/channels")
        prefix = prefix[:, -q:] if q else prefix[:, :0]
        if prefix.shape[1] < q:
            pad = x.new_zeros((x.shape[0], q - prefix.shape[1], in_channels))
            prefix = torch.cat([pad, prefix], dim=1)
    full = torch.cat([prefix, x], dim=1)
    out = x.new_zeros((x.shape[0], x.shape[1], taps.shape[2]))
    for lag in range(lag_count):
        source = full[:, q - lag : q - lag + x.shape[1], :]
        out = out + torch.einsum("bti,io->bto", source, taps[lag])
    return out


def decode_future_with_fir(z_history: torch.Tensor, z_future: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    q = taps.shape[0] - 1
    return causal_fir_apply(z_future, taps, prefix=z_history[:, -q:] if q else z_history[:, :0])


def masked_channel_macro_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.shape != mask.shape or pred.ndim != 3:
        raise ValueError("pred, target, mask must share [B,T,C]")
    observed = mask.bool()
    error = torch.where(observed, pred - target, torch.zeros_like(pred))
    count = observed.sum((0, 1)).to(pred.dtype)
    valid = count > 0
    if not bool(valid.any()):
        raise ValueError("No observed entries in masked loss")
    per_channel = error.square().sum((0, 1))[valid] / count[valid]
    return per_channel.mean()


def masked_channel_macro_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    if pred.shape != target.shape or pred.shape != mask.shape or pred.ndim != 3:
        raise ValueError("pred, target, mask must share [B,T,C]")
    observed = mask.astype(bool)
    if not np.isfinite(pred).all() or not np.isfinite(target[observed]).all():
        raise ValueError("Nonfinite prediction or observed target")
    error = np.where(observed, pred.astype(np.float64) - target.astype(np.float64), 0.0)
    count = observed.sum(axis=(0, 1)).astype(np.float64)
    valid = count > 0
    if not np.any(valid):
        raise ValueError("No observed entries")
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
    result = {"all": masked_channel_macro_metrics(pred, target, mask)}
    for start, end in groups:
        label = f"h{start}_{end}"
        result[label] = masked_channel_macro_metrics(pred[:, start - 1 : end], target[:, start - 1 : end], mask[:, start - 1 : end])
    return result


class LinearLagED(nn.Module):
    def __init__(self, channels: int, latent: int, encoder_lag: int, decoder_lag: int, basis: torch.Tensor):
        super().__init__()
        if tuple(basis.shape) != (channels, latent):
            raise ValueError("basis must be [C,K]")
        self.channels = channels
        self.latent = latent
        self.encoder_lag = encoder_lag
        self.decoder_lag = decoder_lag
        self.encoder_taps = nn.Parameter(torch.zeros(encoder_lag + 1, channels, latent, dtype=basis.dtype, device=basis.device))
        self.decoder_taps = nn.Parameter(torch.zeros(decoder_lag + 1, latent, channels, dtype=basis.dtype, device=basis.device))
        with torch.no_grad():
            self.encoder_taps[0].copy_(basis)
            self.decoder_taps[0].copy_(basis.T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return causal_fir_apply(x, self.encoder_taps)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return causal_fir_apply(z, self.decoder_taps)

    def decode_future(self, z_history: torch.Tensor, z_future: torch.Tensor) -> torch.Tensor:
        return decode_future_with_fir(z_history, z_future, self.decoder_taps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class GraphTimeED(nn.Module):
    def __init__(self, channels: int, latent: int, encoder_lag: int, decoder_lag: int, basis: torch.Tensor, graph: torch.Tensor):
        super().__init__()
        if tuple(basis.shape) != (channels, latent) or tuple(graph.shape) != (channels, channels):
            raise ValueError("basis must be [C,K] and graph [C,C]")
        self.channels = channels
        self.latent = latent
        self.encoder_lag = encoder_lag
        self.decoder_lag = decoder_lag
        self.register_buffer("identity", torch.eye(channels, dtype=basis.dtype, device=basis.device))
        self.register_buffer("graph", graph.to(dtype=basis.dtype, device=basis.device))
        self.alpha = nn.Parameter(torch.zeros(encoder_lag + 1, dtype=basis.dtype, device=basis.device))
        self.beta = nn.Parameter(torch.zeros(encoder_lag + 1, dtype=basis.dtype, device=basis.device))
        self.encoder_basis = nn.Parameter(basis.clone())
        self.decoder_taps = nn.Parameter(torch.zeros(decoder_lag + 1, latent, channels, dtype=basis.dtype, device=basis.device))
        with torch.no_grad():
            self.alpha[0] = 1.0
            self.decoder_taps[0].copy_(basis.T)

    def encoder_taps(self) -> torch.Tensor:
        taps = []
        for lag in range(self.encoder_lag + 1):
            mixer = self.alpha[lag] * self.identity + self.beta[lag] * self.graph
            taps.append(mixer @ self.encoder_basis)
        return torch.stack(taps, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return causal_fir_apply(x, self.encoder_taps())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return causal_fir_apply(z, self.decoder_taps)

    def decode_future(self, z_history: torch.Tensor, z_future: torch.Tensor) -> torch.Tensor:
        return decode_future_with_fir(z_history, z_future, self.decoder_taps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def train_abs_pearson_graph(x: np.ndarray, finite: np.ndarray, train_bounds: tuple[int, int], top_k: int = 3) -> np.ndarray:
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
            xi = train_x[mask, i]
            xj = train_x[mask, j]
            xi = xi - xi.mean()
            xj = xj - xj.mean()
            denom = float(np.sqrt(np.dot(xi, xi) * np.dot(xj, xj)))
            if denom > 0:
                corr[i, j] = abs(float(np.dot(xi, xj) / denom))
    graph = np.zeros_like(corr, dtype=np.float32)
    k = max(0, min(top_k, channels - 1))
    for i in range(channels):
        if k == 0:
            continue
        order = np.argsort(-corr[i], kind="stable")
        chosen = [j for j in order if j != i and corr[i, j] > 0][:k]
        if chosen:
            weights = corr[i, chosen]
            graph[i, chosen] = (weights / weights.sum()).astype(np.float32)
    np.fill_diagonal(graph, 0.0)
    return graph


def load_dataset(dataset: str) -> DatasetBundle:
    runtime = import_file(RUNTIME_V18, "_lag_pilot_runtime_v18")
    data = runtime.load_data(dataset, include_test=False)
    contract = runtime.dataset_contract(dataset)
    train_bounds = tuple(int(v) for v in contract["bounds"]["train"])
    val_bounds = tuple(int(v) for v in contract["bounds"]["val"])
    x = np.asarray(data["x"], dtype=np.float32)
    finite = np.asarray(data["finite"], dtype=bool)
    basis = np.asarray(data["basis"], dtype=np.float32)
    train_origins = np.asarray(data["train_origins"], dtype=np.int64)
    val_origins = np.asarray(data["val_origins"], dtype=np.int64)
    if np.any(train_origins - 512 < 0) or np.any(train_origins < train_bounds[0]) or np.any(train_origins + 48 > train_bounds[1]):
        raise RuntimeError(f"{dataset} TRAIN origins cross the TRAIN bounds")
    if np.any(val_origins - 512 < 0) or np.any(val_origins < val_bounds[0]) or np.any(val_origins + 48 > val_bounds[1]):
        raise RuntimeError(f"{dataset} VAL origins cross the VAL bounds")
    expected = (contract["c"], contract["k"])
    if basis.shape != expected:
        raise RuntimeError(f"{dataset} basis shape changed: {basis.shape} != {expected}")
    if x.shape != tuple(contract["array_contract"]["trainval"]["x_shape"]):
        raise RuntimeError(f"{dataset} trainval shape changed")
    return DatasetBundle(
        name=dataset,
        x=x,
        finite=finite,
        basis=basis,
        columns=[str(v) for v in data["columns"].tolist()],
        train_origins=train_origins,
        val_origins=val_origins,
        train_bounds=train_bounds,
        val_bounds=val_bounds,
        contract=contract,
        source_path=str(data["_path"]),
    )


def windows(bundle: DatasetBundle, origins: np.ndarray, context: int = 512) -> tuple[np.ndarray, np.ndarray]:
    xs = np.stack([bundle.x[int(o) - context : int(o)] for o in origins]).astype(np.float32)
    ms = np.stack([bundle.finite[int(o) - context : int(o)] for o in origins]).astype(bool)
    return xs, ms


def training_reconstruction_windows(bundle: DatasetBundle, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs, ms = windows(bundle, origins)
    timestamps = origins[:, None] - 512 + np.arange(512)[None, :]
    in_train = (timestamps >= bundle.train_bounds[0]) & (timestamps < bundle.train_bounds[1])
    return xs, ms & in_train[:, :, None]


def future_targets(bundle: DatasetBundle, origins: np.ndarray, horizon: int = 48) -> tuple[np.ndarray, np.ndarray]:
    y = np.stack([bundle.x[int(o) : int(o) + horizon] for o in origins]).astype(np.float32)
    m = np.stack([bundle.finite[int(o) : int(o) + horizon] for o in origins]).astype(bool)
    return y, m


def assert_no_test_origins(bundle: DatasetBundle, origins: np.ndarray, horizon: int = 48) -> None:
    test_start = int(bundle.contract["bounds"]["test_a"][0])
    if np.any(origins + horizon > test_start):
        raise RuntimeError(f"{bundle.name} origin set would enter TEST")


def pca_reconstruct(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x @ basis @ basis.T).astype(np.float32)


def reconstruction_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, burn_in: int) -> dict:
    return masked_channel_macro_metrics(pred[:, burn_in:], target[:, burn_in:], mask[:, burn_in:])


def build_lag_features(z: np.ndarray, lag: int, burn_in: int) -> np.ndarray:
    parts = []
    for ell in range(lag + 1):
        parts.append(z[:, burn_in - ell : z.shape[1] - ell, :])
    return np.concatenate(parts, axis=2)


def reconstruction_timestamp_weights(bundle: DatasetBundle, burn_in: int) -> np.ndarray:
    changes = np.zeros(len(bundle.x) + 1, dtype=np.int64)
    starts = np.maximum(bundle.train_origins - 512 + burn_in, bundle.train_bounds[0])
    ends = np.minimum(bundle.train_origins, bundle.train_bounds[1])
    valid = starts < ends
    np.add.at(changes, starts[valid], 1)
    np.add.at(changes, ends[valid], -1)
    return np.cumsum(changes[:-1])


def fit_pca_lag_decoder(bundle: DatasetBundle, lambdas: list[float], decoder_lag: int, burn_in: int, batch_size: int = 64) -> dict:
    dim = (decoder_lag + 1) * bundle.latent
    xtx = np.zeros((bundle.channels, dim, dim), dtype=np.float64)
    xty = np.zeros((bundle.channels, dim), dtype=np.float64)
    counts = np.zeros(bundle.channels, dtype=np.int64)
    monitor(f"ridge:{bundle.name}")
    weights = reconstruction_timestamp_weights(bundle, burn_in)
    timestamps = np.flatnonzero(weights)
    z = bundle.x[:bundle.train_bounds[1]] @ bundle.basis
    features = np.concatenate([z[timestamps - ell] for ell in range(decoder_lag + 1)], axis=1).astype(np.float64)
    target = bundle.x[timestamps].astype(np.float64)
    mask = bundle.finite[timestamps]
    for c in range(bundle.channels):
        monitor()
        obs = mask[:, c]
        if not np.any(obs):
            continue
        row_scale = np.sqrt(weights[timestamps[obs]])
        fc = features[obs] * row_scale[:, None]
        yc = target[obs, c] * row_scale
        xtx[c] = fc.T @ fc
        xty[c] = fc.T @ yc
        counts[c] = int(weights[timestamps[obs]].sum())

    val_x, val_m = windows(bundle, bundle.val_origins)
    val_z = val_x @ bundle.basis
    choices = []
    best = None
    eye = np.eye(dim, dtype=np.float64)
    for lam in lambdas:
        if MONITOR is not None:
            MONITOR.progress["ridge_fits"] += 1
            if MONITOR.progress["ridge_fits"] > MONITOR.limits["ridge_fits"]:
                raise RuntimeError("STOP_RESOURCE: ridge fit count")
        monitor(f"ridge_fit:{bundle.name}:{lam}")
        coef = np.zeros((decoder_lag + 1, bundle.latent, bundle.channels), dtype=np.float32)
        flat = coef.reshape(dim, bundle.channels)
        for c in range(bundle.channels):
            if counts[c] == 0:
                continue
            flat[:, c] = np.linalg.solve(xtx[c] + float(lam) * eye, xty[c]).astype(np.float32)
        pred = apply_decoder_numpy(val_z, coef)
        metrics = reconstruction_metrics(pred, val_x, val_m, burn_in)
        row = {"lambda": float(lam), "val_reconstruction": metrics}
        choices.append(row)
        if best is None or metrics["mse"] < best["val_reconstruction"]["mse"]:
            best = {"lambda": float(lam), "decoder_taps": coef, "val_reconstruction": metrics}
    if best is None:
        raise RuntimeError("No ridge candidate was fit")
    return {"selected": best, "choices": choices, "train_counts": counts.tolist()}


def apply_decoder_numpy(z: np.ndarray, taps: np.ndarray, prefix: np.ndarray | None = None) -> np.ndarray:
    with torch.no_grad():
        zt = torch.from_numpy(z.astype(np.float32))
        tt = torch.from_numpy(taps.astype(np.float32))
        pt = None if prefix is None else torch.from_numpy(prefix.astype(np.float32))
        return causal_fir_apply(zt, tt, prefix=pt).cpu().numpy().astype(np.float32)


def model_for_family(family: str, bundle: DatasetBundle, graph: np.ndarray | None, device: str) -> nn.Module:
    basis = torch.tensor(bundle.basis, dtype=torch.float32, device=device)
    if family == "LINEAR_LAG_ED":
        return LinearLagED(bundle.channels, bundle.latent, 8, 8, basis)
    if family == "GRAPH_TIME_ED":
        if graph is None:
            raise ValueError("GRAPH_TIME_ED needs a graph")
        return GraphTimeED(bundle.channels, bundle.latent, 8, 8, basis, torch.tensor(graph, dtype=torch.float32, device=device))
    raise ValueError(family)


def evaluate_reconstruction_model(model: nn.Module, bundle: DatasetBundle, origins: np.ndarray, burn_in: int, device: str, batch_size: int = 64) -> dict:
    preds, targets, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(origins), batch_size):
            monitor()
            xw, mw = windows(bundle, origins[start : start + batch_size])
            xt = torch.from_numpy(xw).to(device)
            pred = model(xt).detach().cpu().numpy()
            preds.append(pred)
            targets.append(xw)
            masks.append(mw)
    return reconstruction_metrics(np.concatenate(preds), np.concatenate(targets), np.concatenate(masks), burn_in)


def train_neural(bundle: DatasetBundle, family: str, seed: int, graph: np.ndarray | None, protocol: dict, device: str) -> dict:
    fit_id = f"{bundle.name}_{family.lower()}_seed{seed}"
    if MONITOR is not None:
        MONITOR.progress["fit_starts"] += 1
        if MONITOR.progress["fit_starts"] > MONITOR.limits["new_neural_fit_starts"]:
            raise RuntimeError("STOP_RESOURCE: neural fit-start count")
    monitor(f"fit:{fit_id}", epoch=0)
    start_time = time.perf_counter()
    reset_cuda_peak(device)
    model = model_for_family(family, bundle, graph, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(protocol["learning_rate"]), weight_decay=float(protocol["weight_decay"]))
    rng = np.random.default_rng(seed)
    max_epochs = int(protocol["max_epochs"])
    patience = int(protocol["patience"])
    epoch_origins = int(protocol["epoch_origins"])
    batch_size = int(protocol["reconstruction_batch"])
    burn_in = int(protocol["reconstruction_burn_in"])
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_val = evaluate_reconstruction_model(model, bundle, bundle.val_origins, burn_in, device)["mse"]
    curve = [{"epoch": 0, "val_reconstruction_mse": float(best_val), "train_reconstruction_mse": None}]
    ckpt_path = CACHE / "checkpoints" / f"{fit_id}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    def save_best():
        torch.save({"fit_id": fit_id, "family": family, "dataset": bundle.name, "seed": seed,
                    "best_epoch": best_epoch, "state_dict": best_state}, ckpt_path)
        atomic_write_json(HERE / "curves" / f"{fit_id}.json", curve)
    save_best()
    stale = 0
    for epoch in range(1, max_epochs + 1):
        monitor(f"fit:{fit_id}", epoch=epoch, best_epoch=best_epoch, best_val_reconstruction_mse=float(best_val))
        model.train()
        sampled = rng.choice(bundle.train_origins, size=epoch_origins, replace=True)
        train_losses = []
        for start in range(0, len(sampled), batch_size):
            monitor()
            xw, mw = training_reconstruction_windows(bundle, sampled[start : start + batch_size])
            xt = torch.from_numpy(xw).to(device)
            mt = torch.from_numpy(mw).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xt)
            loss = masked_channel_macro_mse(pred[:, burn_in:], xt[:, burn_in:], mt[:, burn_in:])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite reconstruction loss: {fit_id}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(protocol["clip_grad_norm"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate_reconstruction_model(model, bundle, bundle.val_origins, burn_in, device)
        val = float(val_metrics["mse"])
        curve.append({"epoch": epoch, "val_reconstruction_mse": val, "train_reconstruction_mse": float(np.mean(train_losses))})
        if val < best_val:
            best_val = val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            save_best()
        else:
            stale += 1
        atomic_write_json(HERE / "curves" / f"{fit_id}.json", curve)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    full_val = evaluate_reconstruction_model(model, bundle, bundle.val_origins, burn_in, device)
    save_best()
    if MONITOR is not None:
        MONITOR.progress["completed_fits"] += 1
    monitor(f"fit_complete:{fit_id}", epoch=int(curve[-1]["epoch"]))
    return {
        "fit_id": fit_id,
        "family": family,
        "dataset": bundle.name,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "epochs_run": int(curve[-1]["epoch"]),
        "val_reconstruction": full_val,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "resource": resource_receipt(device, start_time),
        "checkpoint": {"path": ckpt_path, "sha256": sha256_path(ckpt_path), "bytes": ckpt_path.stat().st_size},
        "model": model,
    }


def load_backbone(dataset: str, device: str):
    loader = import_file(READOUT_LOADER, "_lag_pilot_model_readout")
    return loader.load_backbone(dataset, device=device)


def backbone_state_hash(backbone) -> str:
    h = hashlib.sha256()
    for name, value in sorted(backbone.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


@torch.no_grad()
def bolt_median_forecast(backbone, context: np.ndarray, device: str, batch_origins: int = 4) -> np.ndarray:
    chunks = []
    quantiles = list(backbone.chronos_config.quantiles)
    median = quantiles.index(0.5)
    native_horizon = int(backbone.chronos_config.prediction_length)
    num_quantiles = len(quantiles)
    if native_horizon < 48:
        raise RuntimeError("Bolt native horizon is shorter than 48")
    for start in range(0, context.shape[0], batch_origins):
        monitor()
        x = torch.from_numpy(context[start : start + batch_origins].astype(np.float32)).to(device)
        batch, length, channels = x.shape
        rows = x.transpose(1, 2).reshape(batch * channels, length)
        point = backbone(context=rows).quantile_preds[:, median, :48]
        chunks.append(point.reshape(batch, channels, 48).transpose(1, 2).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def adapter_encode(model_or_basis, context: np.ndarray, device: str) -> np.ndarray:
    if isinstance(model_or_basis, np.ndarray):
        return (context @ model_or_basis).astype(np.float32)
    model_or_basis.eval()
    with torch.no_grad():
        return model_or_basis.encode(torch.from_numpy(context.astype(np.float32)).to(device)).detach().cpu().numpy().astype(np.float32)


def latent_train_statistics(bundle: DatasetBundle, encoder, device: str, batch_size: int = 64) -> dict:
    total = None
    total_sq = None
    count = 0
    for start in range(0, len(bundle.train_origins), batch_size):
        monitor()
        origins = bundle.train_origins[start : start + batch_size]
        context, _ = windows(bundle, origins)
        z_all = adapter_encode(encoder, context, device)
        timestamps = origins[:, None] - 512 + np.arange(512)[None, :]
        selected = ((timestamps >= bundle.train_bounds[0]) & (timestamps < bundle.train_bounds[1])
                    & (np.arange(512)[None, :] >= 16))
        z = z_all[selected].astype(np.float64)
        if total is None:
            total = z.sum(axis=0)
            total_sq = np.square(z).sum(axis=0)
        else:
            total += z.sum(axis=0)
            total_sq += np.square(z).sum(axis=0)
        count += int(z.shape[0])
    if total is None or total_sq is None or count == 0:
        raise RuntimeError("No TRAIN latent values were available")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    return {
        "count": int(count),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "condition_norm": float(np.linalg.cond(np.diag(np.sqrt(variance) + 1e-12))),
        "scope": "TRAIN reconstruction timestamps after burn-in; repeated-window weights; condition_norm is diagonal std ratio",
    }


def adapter_decode_future(model_or_taps, z_history: np.ndarray, z_future: np.ndarray, device: str) -> np.ndarray:
    if isinstance(model_or_taps, np.ndarray):
        return apply_decoder_numpy(z_future, model_or_taps, prefix=z_history[:, -8:])
    model_or_taps.eval()
    with torch.no_grad():
        pred = model_or_taps.decode_future(
            torch.from_numpy(z_history.astype(np.float32)).to(device),
            torch.from_numpy(z_future.astype(np.float32)).to(device),
        )
        return pred.detach().cpu().numpy().astype(np.float32)


def pca_val_cache_parity(bundle: DatasetBundle, online_latent: np.ndarray) -> dict:
    manifest_path = V15_CACHE_MANIFEST
    manifest = read_json(manifest_path)
    entry = manifest["datasets"][bundle.name]["pca"]["val"]
    archive_path = Path(entry["path"])
    if sha256_path(archive_path) != entry["sha256"]:
        raise RuntimeError(f"v15 PCA VAL cache hash changed: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        origins = archive["origins"]
        cached = archive["prediction"].astype(np.float32)
    if not np.array_equal(origins, bundle.val_origins):
        raise RuntimeError(f"{bundle.name} PCA cache origins differ from v18 VAL origins")
    close = np.isclose(online_latent, cached, atol=1e-5, rtol=1e-4)
    diff = np.abs(online_latent.astype(np.float64) - cached.astype(np.float64))
    rel = diff / np.maximum(np.abs(cached.astype(np.float64)), 1e-12)
    return {
        "source": {"manifest": manifest_path, "archive": archive_path, "sha256": entry["sha256"], "bytes": archive_path.stat().st_size},
        "passed": bool(close.all()),
        "atol": 1e-5,
        "rtol": 1e-4,
        "max_absolute": float(diff.max(initial=0.0)),
        "max_relative": float(rel.max(initial=0.0)),
        "violating_elements": int((~close).sum()),
        "elements": int(close.size),
    }


def forecast_adapter(bundle: DatasetBundle, adapter_name: str, encoder, decoder, backbone, protocol: dict, device: str) -> dict:
    monitor(f"forecast:{bundle.name}:{adapter_name}")
    started = time.perf_counter()
    reset_cuda_peak(device)
    context, _ = windows(bundle, bundle.val_origins)
    target, target_mask = future_targets(bundle, bundle.val_origins)
    z_history = adapter_encode(encoder, context, device)
    latent_last = np.repeat(z_history[:, -1:, :], int(protocol["horizon"]), axis=1).astype(np.float32)
    raw_last = np.repeat(context[:, -1:, :], int(protocol["horizon"]), axis=1).astype(np.float32)
    z_forecast = bolt_median_forecast(backbone, z_history, device, int(protocol["forecast_batch_origins"]))
    pred_tsfm = adapter_decode_future(decoder, z_history, z_forecast, device)
    pred_last = adapter_decode_future(decoder, z_history, latent_last, device)
    forecast_resource = resource_receipt(device, started)
    forecast_resource["scope"] = "full VAL encode, TSFM, two decoder paths and array construction; excludes metrics, storage and TRAIN diagnostics; resident frozen backbone"
    npz_path = CACHE / "predictions" / f"{bundle.name}_{adapter_name}_val.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        origins=bundle.val_origins,
        latent_forecast=z_forecast,
        latent_last=latent_last,
        prediction_tsfm=pred_tsfm,
        prediction_latent_last=pred_last,
    )
    result = {
        "adapter": adapter_name,
        "forecast": {
            "tsfm": horizon_metrics(pred_tsfm, target, target_mask, [tuple(v) for v in protocol["horizon_groups"]]),
            "latent_last": horizon_metrics(pred_last, target, target_mask, [tuple(v) for v in protocol["horizon_groups"]]),
            "raw_last_reference": horizon_metrics(raw_last, target, target_mask, [tuple(v) for v in protocol["horizon_groups"]]),
        },
        "latent_train": latent_train_statistics(bundle, encoder, device),
        "resource": forecast_resource,
        "predictions": {"path": npz_path, "sha256": sha256_path(npz_path), "bytes": npz_path.stat().st_size},
    }
    if adapter_name == "PCA_STATIC":
        result["pca_val_cache_parity"] = pca_val_cache_parity(bundle, z_forecast)
        if not result["pca_val_cache_parity"]["passed"]:
            atomic_write_json(HERE / f"{bundle.name}_parity_failure.json", result["pca_val_cache_parity"])
            raise RuntimeError(f"{bundle.name}: online PCA full-VAL cache parity failed")
    return result


def forecast_raw_references(bundle: DatasetBundle, backbone, protocol: dict, device: str) -> dict:
    monitor(f"forecast:{bundle.name}:raw")
    started = time.perf_counter()
    reset_cuda_peak(device)
    context, _ = windows(bundle, bundle.val_origins)
    target, target_mask = future_targets(bundle, bundle.val_origins)
    raw_f0 = bolt_median_forecast(backbone, context, device, int(protocol["forecast_batch_origins"]))
    resource = resource_receipt(device, started)
    resource["scope"] = "full VAL raw TSFM forecast and input/target array construction; excludes metrics and storage; resident frozen backbone"
    last_started = time.perf_counter()
    raw_last = np.repeat(context[:, -1:, :], int(protocol["horizon"]), axis=1).astype(np.float32)
    last_resource = resource_receipt("cpu", last_started)
    last_resource["scope"] = "NumPy raw-last repeat only; resident process backbone not charged as raw-last compute"
    npz_path = CACHE / "predictions" / f"{bundle.name}_raw_references_val.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, origins=bundle.val_origins, raw_f0=raw_f0, raw_last=raw_last)
    return {
        "RAW_F0": {
            "forecast": horizon_metrics(raw_f0, target, target_mask, [tuple(v) for v in protocol["horizon_groups"]]),
            "resource": resource,
            "predictions": {"path": npz_path, "sha256": sha256_path(npz_path), "bytes": npz_path.stat().st_size, "key": "raw_f0"},
        },
        "RAW_LAST": {
            "forecast": horizon_metrics(raw_last, target, target_mask, [tuple(v) for v in protocol["horizon_groups"]]),
            "resource": last_resource,
            "predictions": {"path": npz_path, "sha256": sha256_path(npz_path), "bytes": npz_path.stat().st_size, "key": "raw_last"},
        },
    }


def run_dataset(bundle: DatasetBundle, protocol: dict, args, save_partial) -> dict:
    assert_no_test_origins(bundle, bundle.val_origins, int(protocol["horizon"]))
    device = args.device
    graph = train_abs_pearson_graph(bundle.x, bundle.finite, bundle.train_bounds, top_k=3)
    graph_path = CACHE / "graphs" / f"{bundle.name}_train_abs_pearson_top3.npy"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(graph_path, graph)
    val_x, val_m = windows(bundle, bundle.val_origins)
    pca_val = pca_reconstruct(val_x, bundle.basis)
    pca_static = reconstruction_metrics(pca_val, val_x, val_m, int(protocol["reconstruction_burn_in"]))
    ridge = fit_pca_lag_decoder(bundle, [float(v) for v in protocol["ridge_lambdas"]], int(protocol["decoder_max_lag"]), int(protocol["reconstruction_burn_in"]))
    backbone = load_backbone(bundle.name, device)
    before_state_hash = backbone_state_hash(backbone)
    dataset_result = {
        "shape": {"trainval": list(bundle.x.shape), "channels": bundle.channels, "latent": bundle.latent},
        "origins": {"train": int(len(bundle.train_origins)), "val": int(len(bundle.val_origins))},
        "bounds": {"train": bundle.train_bounds, "val": bundle.val_bounds},
        "input_hashes": {
            "x": array_sha256(bundle.x),
            "finite": array_sha256(bundle.finite),
            "basis": array_sha256(bundle.basis),
            "graph": sha256_path(graph_path),
        },
        "graph": {"path": graph_path, "row_sums": graph.sum(axis=1).tolist()},
        "reconstruction": {
            "PCA_STATIC": pca_static,
            "PCA_LAG_D": ridge["selected"]["val_reconstruction"],
            "PCA_LAG_D_choices": ridge["choices"],
        },
        "neural_fits": [],
        "forecast": {},
        "backbone": {"parent": backbone.readout_parent_receipt, "initial_state_sha256": before_state_hash,
                     "trainable_parameters": int(sum(p.numel() for p in backbone.parameters() if p.requires_grad))},
    }
    save_partial(dataset_result)
    dataset_result["forecast"].update(forecast_raw_references(bundle, backbone, protocol, device))
    dataset_result["forecast"]["PCA_STATIC"] = forecast_adapter(
        bundle,
        "PCA_STATIC",
        bundle.basis,
        np.concatenate([bundle.basis.T[None], np.zeros((8, bundle.latent, bundle.channels), dtype=np.float32)], axis=0),
        backbone,
        protocol,
        device,
    )
    dataset_result["forecast"]["PCA_LAG_D"] = forecast_adapter(bundle, "PCA_LAG_D", bundle.basis, ridge["selected"]["decoder_taps"], backbone, protocol, device)
    save_partial(dataset_result)
    for family in args.families:
        if family not in protocol["neural_families"]:
            raise ValueError(family)
        for seed in args.seeds:
            fit = train_neural(bundle, family, int(seed), graph, protocol, device)
            model = fit.pop("model")
            dataset_result["neural_fits"].append(fit)
            dataset_result["reconstruction"][fit["fit_id"]] = fit["val_reconstruction"]
            save_partial(dataset_result)
            dataset_result["forecast"][fit["fit_id"]] = forecast_adapter(bundle, fit["fit_id"], model, model, backbone, protocol, device)
            save_partial(dataset_result)
            del model
    after_state_hash = backbone_state_hash(backbone)
    if after_state_hash != before_state_hash or any(p.grad is not None for p in backbone.parameters()):
        raise RuntimeError("Frozen backbone state or gradient contract changed")
    dataset_result["backbone"].update(final_state_sha256=after_state_hash, unchanged=True, gradients_present=False)
    save_partial(dataset_result)
    return dataset_result


def run(args) -> dict:
    global MONITOR
    configure_runtime()
    protocol = read_json(HERE / "protocol.json")
    if len(args.datasets) != len(set(args.datasets)) or len(args.families) != len(set(args.families)):
        raise ValueError("Repeated dataset or family would duplicate fit starts")
    if len(args.seeds) != len(set(args.seeds)) or any(s not in protocol["seeds"] for s in args.seeds):
        raise ValueError("Only distinct protocol seeds are allowed")
    started = time.perf_counter()
    MONITOR = PilotMonitor(protocol, args.datasets, args.device)
    monitor("running")
    result = {
        "schema": "level_lag_compressor_pilot_result_v1",
        "protocol": protocol,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": args.device,
            "tf32": False,
        },
        "datasets": {},
        "source_hashes": {"pilot": sha256_path(HERE / "pilot.py"), "protocol": sha256_path(HERE / "protocol.json"),
                          "tests": sha256_path(HERE / "test_pilot.py")},
        "status": "running",
    }
    try:
        for dataset in args.datasets:
            monitor(f"dataset:{dataset}")
            bundle = load_dataset(dataset)
            def save_partial(dataset_result):
                result["datasets"][dataset] = dataset_result
                result["wall_seconds"] = float(time.perf_counter() - started)
                atomic_write_json(HERE / "result.json", result)
            result["datasets"][dataset] = run_dataset(bundle, protocol, args, save_partial)
            monitor(f"dataset_complete:{dataset}")
        result["wall_seconds"] = float(time.perf_counter() - started)
        result["status"] = "complete"
        result["execution_counts"] = {k: MONITOR.progress[k] for k in ("fit_starts", "completed_fits", "ridge_fits")}
        atomic_write_json(HERE / "result.json", result)
        monitor("complete")
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["wall_seconds"] = float(time.perf_counter() - started)
        result["execution_counts"] = {k: MONITOR.progress[k] for k in ("fit_starts", "completed_fits", "ridge_fits")}
        atomic_write_json(HERE / "result.json", result)
        MONITOR.progress.update(stage="failed", error=repr(exc), elapsed_seconds=result["wall_seconds"])
        atomic_write_json(HERE / "progress.json", MONITOR.progress)
        raise


def parse_args(argv: list[str] | None = None):
    protocol = read_json(HERE / "protocol.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=protocol["datasets"], choices=protocol["datasets"])
    parser.add_argument("--families", nargs="+", default=protocol["neural_families"], choices=protocol["neural_families"])
    parser.add_argument("--seeds", nargs="+", type=int, default=protocol["seeds"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
