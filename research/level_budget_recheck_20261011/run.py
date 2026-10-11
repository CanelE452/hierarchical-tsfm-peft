"""Fresh 180-epoch budget recheck for matched encoder/decoder contracts.

Root runs this script. It writes only to this research folder and the matching
.cache folder; parent experiments are read-only inputs.
"""

from __future__ import annotations

import argparse
import copy
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
PARENT_DIR = ROOT / "research/level_encoder_contract_20261010"
PARENT_RUNNER = PARENT_DIR / "runner.py"
PARENT_ENCODERS = PARENT_DIR / "encoders.py"
PILOT = ROOT / "research/level_lag_compressor_pilot_20261010/pilot.py"
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


def parent_runner():
    if str(PARENT_DIR) not in sys.path:
        sys.path.insert(0, str(PARENT_DIR))
    module = import_file(PARENT_RUNNER, "_budget_recheck_parent_runner")
    if MONITOR is not None:
        module.MONITOR = MONITOR
        module.old_pilot().MONITOR = MONITOR
    return module


def parent_encoders():
    if str(PARENT_DIR) not in sys.path:
        sys.path.insert(0, str(PARENT_DIR))
    if "encoders" in sys.modules:
        return sys.modules["encoders"]
    return import_file(PARENT_ENCODERS, "encoders")


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
        raise PermissionError(f"refusing to write outside budget-recheck folders: {path}")
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


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def source_hashes() -> dict:
    paths = {
        "PLAN": HERE / "PLAN.md",
        "protocol": HERE / "protocol.json",
        "run": HERE / "run.py",
        "benchmark": HERE / "benchmark.py",
        "test_recheck": HERE / "test_recheck.py",
        "parent_runner": PARENT_RUNNER,
        "parent_encoders": PARENT_ENCODERS,
        "pilot": PILOT,
    }
    return {name: {"path": path, "sha256": sha256_path(path)} for name, path in paths.items() if path.exists()}


def ensure_source_hashes_unchanged(reference: dict) -> None:
    current = source_hashes()
    for name, entry in reference.items():
        if name not in current or current[name]["sha256"] != entry["sha256"]:
            raise RuntimeError(f"source changed during run: {name}")


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Monitor:
    def __init__(self, protocol: dict, device: str):
        self.limits = protocol["limits"]
        self.device = device
        self.started_wall = time.perf_counter()
        self.started_cpu = time.process_time()
        self.process = psutil.Process()
        self.base_bytes = self.storage_bytes()
        self.last_disk_check = 0.0
        self.peak_rss = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.progress = {
            "pid": os.getpid(),
            "stage": "starting",
            "fit_starts": 0,
            "completed_fits": 0,
            "ridge_fits": 0,
            "started_utc": time.time(),
        }

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
        if elapsed - self.last_disk_check > 10:
            if self.storage_bytes() - self.base_bytes > self.limits["incremental_storage_bytes"]:
                raise RuntimeError("STOP_RESOURCE: incremental_storage_bytes")
            self.last_disk_check = elapsed
        if stage is not None:
            self.progress.update(stage=stage, elapsed_seconds=elapsed, process_cpu_seconds=cpu_seconds,
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


def adapter_id(family: str, seed: int, milestone: int) -> str:
    return f"{family}_seed{int(seed)}_m{int(milestone)}"


def learned_adapter_ids(protocol: dict) -> list[str]:
    return [adapter_id(f, int(s), int(m)) for f in protocol["families"] for s in protocol["seeds"] for m in protocol["milestones"]]


def decoder_adapter_ids(protocol: dict) -> list[str]:
    return ["PCA_ANCHOR", *learned_adapter_ids(protocol)]


def benchmark_case_ids(protocol: dict) -> list[str]:
    milestone = int(protocol["benchmark"]["milestone"])
    cases = ["PCA_ANCHOR", "RAW_F0"]
    cases.extend(adapter_id(f, int(s), milestone) for f in protocol["families"] for s in protocol["seeds"])
    return cases


def select_milestone_bests(curves: list[dict], milestones: Iterable[int]) -> dict:
    selected = {}
    for milestone in milestones:
        best = None
        for row in curves:
            epoch = int(row["epoch"])
            if epoch > milestone:
                continue
            value = float(row["val_reconstruction"]["mse"])
            if best is None or value < best["val_reconstruction"]["mse"]:
                best = {"epoch": epoch, "val_reconstruction": copy.deepcopy(row["val_reconstruction"])}
        if best is None:
            raise RuntimeError(f"no curve rows for milestone {milestone}")
        selected[int(milestone)] = best
    return selected


def model_state_hash(model: torch.nn.Module) -> str:
    return parent_runner().module_state_hash(model)


def save_checkpoint(path: Path, payload: dict) -> None:
    path = path.resolve()
    if not path.is_relative_to(CACHE.resolve()):
        raise PermissionError(f"checkpoint outside cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


@torch.no_grad()
def eval_reconstruction(encoder, decoder, bundle, origins: np.ndarray, burn_in: int, device: str, batch_size: int) -> dict:
    pr = parent_runner()
    preds, targets, masks = [], [], []
    encoder.eval()
    decoder.eval()
    for start in range(0, len(origins), batch_size):
        monitor()
        batch_origins = origins[start:start + batch_size]
        x_np, m_np = pr.old_pilot().windows(bundle, batch_origins)
        x = torch.from_numpy(x_np.astype(np.float32)).to(device)
        pred = decoder.decode(encoder.encode(x)).detach().cpu().numpy().astype(np.float32)
        preds.append(pred)
        targets.append(x_np)
        masks.append(m_np)
    return pr.masked_channel_macro_metrics_np(
        np.concatenate(preds, axis=0)[:, burn_in:],
        np.concatenate(targets, axis=0)[:, burn_in:],
        np.concatenate(masks, axis=0)[:, burn_in:],
    )


def compare_curve_prefix(new_curves: list[dict], parent_curve_path: Path, atol: float = 5e-5, rtol: float = 5e-5) -> dict:
    if not parent_curve_path.exists():
        return {"available": False, "passed": False, "reason": "parent curve missing"}
    parent_curve = read_json(parent_curve_path).get("curves", [])
    by_epoch = {int(row["epoch"]): row for row in new_curves}
    diffs = []
    for row in parent_curve:
        epoch = int(row["epoch"])
        if epoch not in by_epoch:
            break
        old = float(row["val_reconstruction"]["mse"])
        new = float(by_epoch[epoch]["val_reconstruction"]["mse"])
        diffs.append(abs(new - old))
    max_abs = max(diffs) if diffs else None
    passed = bool(diffs) and max_abs <= atol + rtol * max(abs(float(parent_curve[0]["val_reconstruction"]["mse"])), 1e-12)
    return {
        "available": True,
        "passed": passed,
        "compared_epochs": len(diffs),
        "max_abs_mse_diff": max_abs,
        "atol": atol,
        "rtol": rtol,
        "parent_curve": parent_curve_path,
        "parent_curve_sha256": sha256_path(parent_curve_path),
    }


def _state_diff(left: dict, right: dict) -> dict:
    max_abs = 0.0
    rms_num = 0.0
    count = 0
    for key, left_value in left.items():
        diff = left_value.detach().cpu().double() - right[key].detach().cpu().double()
        max_abs = max(max_abs, float(diff.abs().max().item()))
        rms_num += float((diff * diff).sum().item())
        count += diff.numel()
    return {"max_abs": max_abs, "rms": float(np.sqrt(rms_num / max(count, 1))), "count": int(count)}


def compare_parent_checkpoint_state(parent_entry: dict, local_checkpoint: Path, graph: np.ndarray, bundle) -> dict:
    if not local_checkpoint.exists():
        return {"available": False, "passed": False, "reason": "local snapshot missing"}
    pe = parent_encoders()
    parent_path = resolve_path(parent_entry["checkpoint"])
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    local_payload = torch.load(local_checkpoint, map_location="cpu", weights_only=False)
    parent_model = pe.build_encoder(parent_entry["family"], bundle.channels, bundle.latent, int(parent_entry["seed"]), graph=graph)
    local_model = pe.build_encoder(parent_entry["family"], bundle.channels, bundle.latent, int(parent_entry["seed"]), graph=graph)
    parent_decoder = pe.build_decoder(bundle.channels, bundle.latent, int(parent_entry["seed"]), lag=8)
    local_decoder = pe.build_decoder(bundle.channels, bundle.latent, int(parent_entry["seed"]), lag=8)
    parent_model.load_state_dict(parent_payload["encoder"])
    local_model.load_state_dict(local_payload["encoder"])
    parent_decoder.load_state_dict(parent_payload["decoder"])
    local_decoder.load_state_dict(local_payload["decoder"])
    encoder_diff = _state_diff(parent_model.state_dict(), local_model.state_dict())
    decoder_diff = _state_diff(parent_decoder.state_dict(), local_decoder.state_dict())
    max_abs = max(encoder_diff["max_abs"], decoder_diff["max_abs"])
    return {
        "available": True,
        "passed": max_abs <= 5e-5,
        "max_abs": max_abs,
        "encoder": encoder_diff,
        "decoder": decoder_diff,
        "parent_checkpoint": parent_path,
        "parent_checkpoint_sha256": sha256_path(parent_path),
        "local_checkpoint": local_checkpoint,
        "local_checkpoint_sha256": sha256_path(local_checkpoint),
        "tolerance": "max_abs<=5e-5; CUDA bitwise equality is not assumed",
    }


def train_fresh(bundle, family: str, seed: int, graph: np.ndarray, protocol: dict, parent_dataset: dict, device: str) -> dict:
    pr = parent_runner()
    pe = parent_encoders()
    if MONITOR is not None:
        if MONITOR.progress["fit_starts"] >= MONITOR.limits["neural_fit_starts"]:
            raise RuntimeError("STOP_RESOURCE: neural_fit_starts")
        MONITOR.progress["fit_starts"] += 1
    monitor(stage=f"fit:{bundle.name}:{family}:{seed}", dataset=bundle.name, family=family, seed=seed)
    encoder = pe.build_encoder(family, bundle.channels, bundle.latent, seed, graph=graph).to(device)
    decoder = pe.build_decoder(bundle.channels, bundle.latent, seed, lag=int(protocol["decoder_max_lag"])).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=float(protocol["learning_rate"]), weight_decay=0.0)
    burn_in = int(protocol["burn_in"])
    batch_size = int(protocol["batch"])
    curves = []
    paths = {
        "latest": CACHE / "checkpoints" / f"{bundle.name}_{family}_{seed}_latest.pt",
        "best60": CACHE / "checkpoints" / f"{bundle.name}_{family}_{seed}_best_m60.pt",
        "best180": CACHE / "checkpoints" / f"{bundle.name}_{family}_{seed}_best_m180.pt",
    }
    parent_id = f"{family}_seed{int(seed)}"
    parent_entry = parent_dataset.get("neural", {}).get(parent_id)
    parent_snapshot_path = None
    if parent_entry:
        parent_snapshot_path = CACHE / "checkpoints" / f"{bundle.name}_{family}_{seed}_epoch{int(parent_entry['best_epoch'])}_parent_epoch.pt"

    best_by_milestone = {60: None, 180: None}
    fit_started = time.perf_counter()
    optimizer_step_count = 0
    epoch_update_counts = []
    milestone_wall = {}

    def checkpoint_payload(epoch: int, val_metrics: dict) -> dict:
        return {
            "dataset": bundle.name,
            "family": family,
            "seed": int(seed),
            "epoch": int(epoch),
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state_cpu": torch.get_rng_state(),
            "torch_rng_state_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") and torch.cuda.is_available() else None,
            "numpy_rng_contract": "epoch origins are regenerated by np.default_rng(seed) and epoch index",
            "val_reconstruction": val_metrics,
            "optimizer_step_count": int(optimizer_step_count),
        }

    def maybe_update_bests(epoch: int, val_metrics: dict) -> None:
        for milestone, path_key in ((60, "best60"), (180, "best180")):
            if epoch <= milestone:
                current = best_by_milestone[milestone]
                if current is None or val_metrics["mse"] < current["val_reconstruction"]["mse"]:
                    best_by_milestone[milestone] = {"epoch": int(epoch), "val_reconstruction": copy.deepcopy(val_metrics)}
                    save_checkpoint(paths[path_key], checkpoint_payload(epoch, val_metrics))

    val_metrics = eval_reconstruction(encoder, decoder, bundle, bundle.val_origins, burn_in, device, batch_size)
    curves.append({"epoch": 0, "val_reconstruction": val_metrics, "train_loss_mean": None})
    maybe_update_bests(0, val_metrics)
    save_checkpoint(paths["latest"], checkpoint_payload(0, val_metrics))

    for epoch in range(1, int(protocol["max_epochs"]) + 1):
        sample = pr.sample_epoch_origins(bundle.train_origins, seed, epoch - 1, int(protocol["epoch_origins"]))
        encoder.train()
        decoder.train()
        losses = []
        epoch_steps = 0
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
            epoch_steps += 1
            losses.append(float(loss.detach().cpu()))
        val_metrics = eval_reconstruction(encoder, decoder, bundle, bundle.val_origins, burn_in, device, batch_size)
        epoch_update_counts.append(int(epoch_steps))
        row = {"epoch": int(epoch), "val_reconstruction": val_metrics, "train_loss_mean": float(np.mean(losses)) if losses else None,
               "optimizer_steps_this_epoch": int(epoch_steps), "optimizer_steps_cumulative": int(optimizer_step_count)}
        curves.append(row)
        maybe_update_bests(epoch, val_metrics)
        payload = checkpoint_payload(epoch, val_metrics)
        save_checkpoint(paths["latest"], payload)
        if parent_entry and epoch == int(parent_entry["best_epoch"]):
            save_checkpoint(parent_snapshot_path, payload)
        atomic_json(CACHE / "curves" / f"{bundle.name}_{family}_{seed}.json", {
            "dataset": bundle.name,
            "family": family,
            "seed": int(seed),
            "curves": curves,
            "optimizer_steps_completed": int(optimizer_step_count),
            "epoch_update_counts": epoch_update_counts,
        })
        if epoch in (60, 180):
            milestone_wall[int(epoch)] = float(time.perf_counter() - fit_started)

    if MONITOR is not None:
        MONITOR.progress["completed_fits"] += 1
    selected = select_milestone_bests(curves, protocol["milestones"])
    old_parity = {}
    if parent_entry:
        old_parity["curve_prefix"] = compare_curve_prefix(curves, resolve_path(parent_entry["curve"]))
        old_parity["checkpoint_state_at_parent_best_epoch"] = compare_parent_checkpoint_state(parent_entry, parent_snapshot_path, graph, bundle)
        old_parity["legacy_attribution_allowed"] = bool(
            old_parity["curve_prefix"].get("passed", False)
            and old_parity["checkpoint_state_at_parent_best_epoch"].get("passed", False)
        )
        old_parity["failure_interpretation"] = "If false, legacy-vs-extended attribution is blocked; 0..60 and 0..180 remain valid within this fresh trajectory."
    result = {
        "id": f"{family}_seed{int(seed)}",
        "family": family,
        "seed": int(seed),
        "max_epochs": int(protocol["max_epochs"]),
        "early_stopping": False,
        "expected_updates_per_epoch": int(protocol["updates_per_epoch"]),
        "optimizer_steps_completed": int(optimizer_step_count),
        "epoch_update_counts": epoch_update_counts,
        "wall_seconds_m60": milestone_wall.get(60),
        "wall_seconds_m180": milestone_wall.get(180),
        "wall_seconds_additional_60_to_180": (milestone_wall.get(180) - milestone_wall.get(60)) if 60 in milestone_wall and 180 in milestone_wall else None,
        "wall_scope": "per-fit wall includes training, full VAL reconstruction, checkpoint saves, and curve writes",
        "curve": CACHE / "curves" / f"{bundle.name}_{family}_{seed}.json",
        "curve_sha256": sha256_path(CACHE / "curves" / f"{bundle.name}_{family}_{seed}.json"),
        "latest_checkpoint": paths["latest"],
        "latest_checkpoint_sha256": sha256_path(paths["latest"]),
        "milestones": {},
        "parent_prefix_parity": old_parity,
        "encoder_params": int(sum(p.numel() for p in encoder.parameters() if p.requires_grad)),
    }
    for milestone, path_key in ((60, "best60"), (180, "best180")):
        path = paths[path_key]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = pe.build_encoder(family, bundle.channels, bundle.latent, seed, graph=graph)
        model.load_state_dict(payload["encoder"])
        result["milestones"][str(milestone)] = {
            "best_epoch": int(selected[milestone]["epoch"]),
            "val_reconstruction": selected[milestone]["val_reconstruction"],
            "checkpoint": path,
            "checkpoint_sha256": sha256_path(path),
            "encoder_state_sha256": model_state_hash(model),
        }
    return result


def load_milestone_encoder(bundle, fit_entry: dict, milestone: int, graph: np.ndarray):
    pe = parent_encoders()
    family = fit_entry["family"]
    seed = int(fit_entry["seed"])
    info = fit_entry["milestones"][str(milestone)]
    path = resolve_path(info["checkpoint"])
    if sha256_path(path) != info["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = pe.build_encoder(family, bundle.channels, bundle.latent, seed, graph=graph)
    model.load_state_dict(payload["encoder"])
    state_hash = model_state_hash(model)
    if state_hash != info["encoder_state_sha256"]:
        raise RuntimeError(f"encoder state hash mismatch: {path}")
    return model, state_hash


def decoder_train_origins(bundle, protocol: dict) -> np.ndarray:
    return parent_runner().decoder_train_origins(
        bundle.train_origins,
        int(protocol["decoder_train_origin_count"]),
        int(protocol["decoder_sampling_seed"]),
    )


def encode_numpy(encoder_or_basis, context: np.ndarray, device: str) -> np.ndarray:
    return parent_runner().encode_numpy(encoder_or_basis, context, device)


def forecast_latents(backbone, z_history: np.ndarray, protocol: dict, device: str) -> np.ndarray:
    return parent_runner().forecast_latents(backbone, z_history, protocol, device)


def fit_decoder_contract_new(bundle, adapter_name: str, encoder_or_basis, backbone, protocol: dict, device: str, origins: np.ndarray, contract: str) -> dict:
    pr = parent_runner()
    burn_in = int(protocol["burn_in"])
    lag = int(protocol["decoder_max_lag"])
    groups = [tuple(v) for v in protocol["horizon_groups"]]
    lambdas = [float(v) for v in protocol["ridge_lambdas"]]
    train_context, train_context_mask = pr.old_pilot().training_reconstruction_windows(bundle, origins)
    z_train = encode_numpy(encoder_or_basis, train_context, device)
    val_context, val_context_mask = pr.old_pilot().windows(bundle, bundle.val_origins)
    z_val = encode_numpy(encoder_or_basis, val_context, device)
    z_val_forecast = forecast_latents(backbone, z_val, protocol, device)
    z_val_last = np.repeat(z_val[:, -1:, :], int(protocol["horizon"]), axis=1).astype(np.float32)
    val_future, val_future_mask = pr.old_pilot().future_targets(bundle, bundle.val_origins)
    if contract == "RECON_RIDGE_D8":
        train_features = pr.reconstruction_features(z_train, burn_in, lag)
        train_target = train_context[:, burn_in:].astype(np.float32)
        train_mask = train_context_mask[:, burn_in:].astype(bool)
        train_row_mask = train_mask.any(axis=2)
        val_features = pr.reconstruction_features(z_val, burn_in, lag)
        val_target = val_context[:, burn_in:].astype(np.float32)
        val_mask = val_context_mask[:, burn_in:].astype(bool)
        fitted = pr.fit_scaled_ridge(train_features, train_target, train_mask, train_row_mask,
                                     val_features, val_target, val_mask, lambdas)
        train_latent_history = z_train.astype(np.float32)
        train_latent_forecast = None
    elif contract == "FORECAST_RIDGE_D8":
        z_train_forecast = forecast_latents(backbone, z_train, protocol, device)
        train_features = pr.future_features(z_train, z_train_forecast, lag)
        train_target, train_mask = pr.old_pilot().future_targets(bundle, origins)
        train_row_mask = train_mask.any(axis=2)
        val_features = pr.future_features(z_val, z_val_forecast, lag)
        fitted = pr.fit_scaled_ridge(train_features, train_target, train_mask, train_row_mask,
                                     val_features, val_future, val_future_mask, lambdas, horizon_groups=groups)
        train_latent_history = z_train[:, -lag:].astype(np.float32)
        train_latent_forecast = z_train_forecast.astype(np.float32)
    else:
        raise ValueError(contract)
    coeff = fitted["coeff"]
    pred_tsfm = pr.apply_coeff(pr.future_features(z_val, z_val_forecast, lag), coeff)
    pred_last = pr.apply_coeff(pr.future_features(z_val, z_val_last, lag), coeff)
    taps = pr.coeff_to_taps(coeff, lag, bundle.latent)
    metrics_tsfm = pr.horizon_metrics(pred_tsfm, val_future, val_future_mask, groups)
    metrics_last = pr.horizon_metrics(pred_last, val_future, val_future_mask, groups)
    npz_path = CACHE / "predictions" / f"{bundle.name}_{adapter_name}_{contract}_val.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        origins=bundle.val_origins,
        train_origins=origins,
        train_latent_history=train_latent_history,
        train_latent_forecast=train_latent_forecast if train_latent_forecast is not None else np.empty((0,), dtype=np.float32),
        val_latent_history_tail=z_val[:, -lag:].astype(np.float32),
        latent_forecast=z_val_forecast,
        latent_last=z_val_last,
        prediction_tsfm=pred_tsfm.astype(np.float32),
        prediction_latent_last=pred_last.astype(np.float32),
        decoder_coeff=coeff.astype(np.float64),
        decoder_taps=taps.astype(np.float64),
        feature_scale=fitted["feature_scale"].astype(np.float64),
    )
    return {
        "contract": contract,
        "selected_lambda": fitted["selected_lambda"],
        "selection_metrics": fitted["selection_metrics"],
        "choices": fitted["choices"],
        "future_tsfm": metrics_tsfm,
        "future_latent_last": metrics_last,
        "prediction_npz": npz_path,
        "prediction_sha256": sha256_path(npz_path),
        "decoder_taps_sha256": array_sha256(taps.astype(np.float64)),
        "feature_scale_sha256": array_sha256(fitted["feature_scale"].astype(np.float64)),
        "near_zero_feature_count": fitted["near_zero_feature_count"],
        "scale_count": fitted["scale_count"],
        "scale_row_mask_sha256": fitted["scale_row_mask_sha256"],
        "precision": "ridge coeff/taps/scale FP64; final predictions FP32",
    }


def evaluate_raw_f0(bundle, backbone, protocol: dict, device: str) -> dict:
    pr = parent_runner()
    context, _ = pr.old_pilot().windows(bundle, bundle.val_origins)
    target, mask = pr.old_pilot().future_targets(bundle, bundle.val_origins)
    pred = pr.old_pilot().bolt_median_forecast(backbone, context.astype(np.float32), device, int(protocol["forecast_batch"]))
    metrics = pr.horizon_metrics(pred, target, mask, [tuple(v) for v in protocol["horizon_groups"]])
    npz_path = CACHE / "predictions" / f"{bundle.name}_RAW_F0_val.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, origins=bundle.val_origins, prediction_tsfm=pred.astype(np.float32))
    return {"future_tsfm": metrics, "prediction_npz": npz_path, "prediction_sha256": sha256_path(npz_path)}


def run_dataset(dataset: str, protocol: dict, parent: dict, device: str, partial: dict) -> dict:
    pr = parent_runner()
    bundle = pr.load_dataset(dataset)
    graph = pr.build_train_graph(bundle.x, bundle.finite, bundle.train_bounds, top_k=3)
    graph_path = CACHE / "graphs" / f"{dataset}_train_abspearson_top3.npy"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(graph_path, graph)
    result = {
        "dataset": dataset,
        "shape": list(bundle.x.shape),
        "channels": bundle.channels,
        "latent": bundle.latent,
        "source": {"path": bundle.source_path, "sha256": sha256_path(Path(bundle.source_path))},
        "graph": {"path": graph_path, "sha256": sha256_path(graph_path)},
        "fits": {},
        "decoder_train_origins": {},
        "decoder_contracts": {},
        "raw_f0": {},
        "benchmark_cases": [],
    }
    parent_dataset = parent.get("datasets", {}).get(dataset, {})
    for family in protocol["families"]:
        for seed in protocol["seeds"]:
            fit = train_fresh(bundle, family, int(seed), graph, protocol, parent_dataset, device)
            result["fits"][fit["id"]] = fit
            partial["datasets"][dataset] = to_jsonable(result)
            atomic_json(HERE / "result.partial.json", partial)
    dec_origins = decoder_train_origins(bundle, protocol)
    result["decoder_train_origins"] = {
        "count": int(len(dec_origins)),
        "seed": int(protocol["decoder_sampling_seed"]),
        "sha256": array_sha256(dec_origins),
    }
    backbone = pr.old_pilot().load_backbone(dataset, device=device)
    backbone_before = pr.old_pilot().backbone_state_hash(backbone)
    val_context, _ = pr.old_pilot().windows(bundle, bundle.val_origins)
    pca_val_forecast = forecast_latents(backbone, (val_context @ bundle.basis).astype(np.float32), protocol, device)
    pca_parity = pr.old_pilot().pca_val_cache_parity(bundle, pca_val_forecast)
    if not pca_parity.get("passed", False):
        raise RuntimeError(f"{dataset} PCA parity failed")
    result["pca_full_val_parity"] = pca_parity
    adapters = [("PCA_ANCHOR", bundle.basis, array_sha256(bundle.basis), {"kind": "pca", "milestone": None})]
    for fit_id, fit in result["fits"].items():
        for milestone in protocol["milestones"]:
            model, state_hash = load_milestone_encoder(bundle, fit, int(milestone), graph)
            adapters.append((adapter_id(fit["family"], int(fit["seed"]), int(milestone)), model, state_hash,
                             {"kind": "learned", "fit_id": fit_id, "family": fit["family"], "seed": int(fit["seed"]), "milestone": int(milestone)}))
    for name, adapter, state_hash, meta in adapters:
        monitor(stage=f"decoder:{dataset}:{name}")
        result["decoder_contracts"][name] = {
            "meta": meta,
            "encoder_state_sha256_before": state_hash,
            "contracts": {},
        }
        if isinstance(adapter, torch.nn.Module):
            for p in adapter.parameters():
                p.requires_grad_(False)
            adapter = adapter.to(device)
        for contract in protocol["decoder_contracts"]:
            ensure_source_hashes_unchanged(partial["source_hashes"])
            result["decoder_contracts"][name]["contracts"][contract] = fit_decoder_contract_new(
                bundle, name, adapter, backbone, protocol, device, dec_origins, contract
            )
            partial["datasets"][dataset] = to_jsonable(result)
            atomic_json(HERE / "result.partial.json", partial)
        result["decoder_contracts"][name]["encoder_state_sha256_after"] = (
            model_state_hash(adapter.cpu()) if isinstance(adapter, torch.nn.Module) else state_hash
        )
        if result["decoder_contracts"][name]["encoder_state_sha256_after"] != state_hash:
            raise RuntimeError(f"encoder changed during decoder fitting: {name}")
    result["raw_f0"] = evaluate_raw_f0(bundle, backbone, protocol, device)
    backbone_after = pr.old_pilot().backbone_state_hash(backbone)
    if backbone_before != backbone_after:
        raise RuntimeError(f"{dataset} frozen backbone changed")
    if not all(p.grad is None for p in backbone.parameters()):
        raise RuntimeError(f"{dataset} frozen backbone accumulated gradients")
    result["backbone"] = {"state_sha256_before": backbone_before, "state_sha256_after": backbone_after, "unchanged": True, "grads_none": True}
    result["benchmark_cases"] = benchmark_case_ids(protocol)
    del backbone
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def load_parent(protocol: dict) -> dict:
    path = resolve_path(protocol["parent"])
    if not path.exists():
        raise FileNotFoundError(path)
    parent = read_json(path)
    if parent.get("status") != "complete":
        raise RuntimeError(f"parent result is not complete: {parent.get('status')}")
    return parent


def run(args) -> dict:
    global MONITOR
    protocol = read_json(HERE / "protocol.json")
    CACHE.mkdir(parents=True, exist_ok=True)
    configure_runtime()
    MONITOR = Monitor(protocol, args.device)
    parent = load_parent(protocol)
    partial = {
        "schema": "matched_additional_budget_full_cost_result_v1",
        "status": "running",
        "protocol": protocol,
        "parent": {"path": resolve_path(protocol["parent"]), "sha256": sha256_path(resolve_path(protocol["parent"])), "schema": parent.get("schema")},
        "source_hashes": source_hashes(),
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
            atomic_json(HERE / "progress.json", MONITOR.progress)
        atomic_json(HERE / "result.partial.json", partial)
        raise


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
