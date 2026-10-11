"""Native small-K TS2Vec stage. Root executes science after the source seal."""

from __future__ import annotations

import argparse
import gc
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import sweep
from ts2vec_adapter import NativeTs2VecAdapter, array_hash, state_hash, vendor_provenance, native_module


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name


class TracedTensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.fetched_indices = []

    def __getitem__(self, index):
        self.fetched_indices.append(int(index))
        return super().__getitem__(index)


def adapter_id(dataset: str, latent: int, seed: int) -> str:
    return sweep.adapter_id(dataset, "TS2VEC_CAUSAL_SMALL", latent, seed)


def eligible_training_origins(bundle, context: int = 512):
    canonical = np.asarray(bundle.train_origins, dtype=np.int64)
    start, end = (int(value) for value in bundle.train_bounds)
    eligible_mask = (canonical - context >= start) & (canonical <= end)
    eligible = canonical[eligible_mask]
    excluded = canonical[~eligible_mask]
    if not len(eligible):
        raise RuntimeError("No fully TRAIN-contained histories for native contrastive fitting")
    if np.any(eligible - context < start) or np.any(eligible > end):
        raise RuntimeError("Native TS2Vec fitting histories cross TRAIN bounds")
    return canonical, eligible, excluded


def make_training_pool(bundle, method_protocol: dict, monitor):
    context = int(method_protocol["context"])
    canonical, eligible, excluded = eligible_training_origins(bundle, context)
    required = int(len(eligible) * context * bundle.channels * np.dtype(np.float32).itemsize)
    cap = int(method_protocol["training_data"]["materialized_window_budget_bytes"])
    if required > cap:
        raise RuntimeError("STOP_RESOURCE: full eligible native window pool exceeds materialization cap")
    monitor.check("ts2vec_pool_preallocation", dataset=bundle.name, window_bytes=required)
    windows = np.empty((len(eligible), context, bundle.channels), dtype=np.float32)
    pilot = sweep.parent_runner().old_pilot()
    for begin in range(0, len(eligible), 64):
        chosen = eligible[begin:begin + 64]
        context_values, context_masks = pilot.training_reconstruction_windows(bundle, chosen)
        windows[begin:begin + len(chosen)] = context_values
        monitor.check("ts2vec_pool_materialization", dataset=bundle.name, materialized_origins=min(begin + 64, len(eligible)))
    if not np.isfinite(windows).all():
        raise RuntimeError("Native fitting pool did not preserve finite prior-imputed input contract")
    pool_path = CACHE / "ts2vec/training_pools" / f"{bundle.name}_eligible_origins.npz"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    if pool_path.exists():
        raise FileExistsError(pool_path)
    np.savez_compressed(pool_path, canonical_origins=canonical, eligible_origins=eligible, excluded_origins=excluded,
                        train_bounds=np.asarray(bundle.train_bounds, dtype=np.int64))
    receipt = {
        "pool": "all canonical TRAIN origins whose full past context is within TRAIN bounds",
        "canonical_origin_count": int(len(canonical)),
        "canonical_origin_sha256": array_hash(canonical),
        "eligible_origin_count": int(len(eligible)),
        "eligible_origin_sha256": array_hash(eligible),
        "excluded_origin_count": int(len(excluded)),
        "excluded_origin_sha256": array_hash(excluded),
        "train_bounds": list(bundle.train_bounds),
        "history_timestamp_bounds_inclusive": [int(eligible.min()) - context, int(eligible.max()) - 1],
        "window_shape": list(windows.shape),
        "window_bytes": int(windows.nbytes),
        "window_sha256": array_hash(windows),
        "materialized_window_budget_bytes": cap,
        "origin_npz": pool_path,
        "origin_npz_sha256": sweep.sha256_path(pool_path),
        "prior_imputed_inputs": True,
        "pre_train_values_used_for_native_fit": False,
        "mlp_pool_difference": "MLP can read known pre-TRAIN history but masks reconstruction targets; native pool excludes such histories",
        "native_instance_definition": "overlapping past512 histories; not the original forecasting single-series segmentation recipe",
        "native_data_order": "official shuffle/drop_last, differs from MLP's with-replacement stream",
    }
    return windows, receipt


@torch.no_grad()
def encode_origins(bundle, origins: np.ndarray, adapter, device: str, origin_batch: int):
    if origin_batch >= 256:
        raise ValueError("Native causal export must retain the internal buffer branch")
    pilot = sweep.parent_runner().old_pilot()
    chunks = []
    for begin in range(0, len(origins), origin_batch):
        sweep.monitor("ts2vec_export", dataset=bundle.name, encoded_origins=min(begin + origin_batch, len(origins)))
        values, _ = pilot.windows(bundle, origins[begin:begin + origin_batch])
        encoded = adapter.encode(torch.from_numpy(values.astype(np.float32)).to(device))
        chunks.append(encoded.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


@torch.no_grad()
def native_forecast_trace(bundle, adapter, backbone, common_protocol: dict, method_protocol: dict, device: str, train_origins: np.ndarray):
    parent = sweep.parent_runner()
    origin_batch = int(method_protocol["export"]["scientific_origin_batch"])
    train_latent = encode_origins(bundle, train_origins, adapter, device, origin_batch)
    val_latent = encode_origins(bundle, bundle.val_origins, adapter, device, origin_batch)
    train_forecast = sweep.forecast_latents(backbone, train_latent, common_protocol, device)
    val_forecast = sweep.forecast_latents(backbone, val_latent, common_protocol, device)
    lag = int(common_protocol["decoder_max_lag"])
    return {
        "train_origins": np.asarray(train_origins, dtype=np.int64),
        "origins": np.asarray(bundle.val_origins, dtype=np.int64),
        "train_latent_history": train_latent[:, -lag:].astype(np.float32),
        "train_latent_forecast": train_forecast.astype(np.float32),
        "val_latent_history_tail": val_latent[:, -lag:].astype(np.float32),
        "latent_forecast": val_forecast.astype(np.float32),
        "latent_last": np.repeat(val_latent[:, -1:], int(common_protocol["horizon"]), axis=1).astype(np.float32),
        "reused_trace": None,
    }


def sealed_method_sources() -> dict:
    path = HERE / "source_seal.method.json"
    if not path.is_file():
        raise FileNotFoundError("Root must seal all completed method sources before science")
    receipt = sweep.read_json(path)
    if receipt.get("stage") != "method" or receipt.get("phase") != "before_stage_science":
        raise RuntimeError("Native method source seal has wrong stage")
    hashes = {}
    for item in receipt["artifacts"]:
        actual = sweep.sha256_path(sweep.resolve_path(item["path"]))
        if actual != item["sha256"]:
            raise RuntimeError(f"Native method source differs from pre-science seal: {item['path']}")
        hashes[item["path"]] = actual
    return hashes


def ensure_method_sources(reference: dict) -> None:
    if sealed_method_sources() != reference:
        raise RuntimeError("Native method source seal changed during science")


def load_adapter(dataset: str, name: str, record: dict, device: str):
    sealed_method_sources()
    meta = record["meta"]
    if meta["family"] != "TS2VEC_CAUSAL_SMALL" or meta["dataset"] != dataset:
        raise RuntimeError("Native benchmark adapter metadata mismatch")
    if name != meta["id"] or name != adapter_id(dataset, int(meta["k"]), int(meta["seed"])):
        raise RuntimeError("Native benchmark adapter ID mismatch")
    checkpoint = sweep.resolve_path(meta["checkpoint"])
    if sweep.sha256_path(checkpoint) != meta["checkpoint_sha256"]:
        raise RuntimeError("Native benchmark checkpoint changed")
    adapter, payload = NativeTs2VecAdapter.load_checkpoint(checkpoint, device)
    if adapter.latent != int(meta["k"]) or adapter.seed != int(meta["seed"]):
        raise RuntimeError("Native benchmark checkpoint configuration mismatch")
    adapter.dataset = dataset
    actual = state_hash(adapter)
    if actual != record["encoder_state_sha256_before"] or actual != record["encoder_state_sha256_after"]:
        raise RuntimeError("Native benchmark checkpoint did not round-trip to frozen encoder state")
    del payload
    return adapter, actual


def train_native_fit(bundle, seed: int, windows: np.ndarray, pool: dict, method_protocol: dict, device: str, reference: dict):
    ensure_method_sources(reference)
    monitor = sweep.MONITOR
    name = adapter_id(bundle.name, bundle.latent, seed)
    if monitor.progress["fit_starts"] >= int(monitor.limits["neural_fit_starts"]):
        raise RuntimeError("STOP_RESOURCE: native neural_fit_starts")
    monitor.progress["fit_starts"] += 1
    monitor.check("ts2vec_fit_start", dataset=bundle.name, adapter=name, seed=int(seed))
    adapter = NativeTs2VecAdapter(bundle.channels, bundle.latent, seed, device, method_protocol["native_configuration"])
    adapter.dataset = bundle.name
    encoder_params_single = sum(parameter.numel() for parameter in adapter.training_net.parameters())
    resident_params_total = sum(parameter.numel() for parameter in adapter.parameters())
    initial_state = state_hash(adapter)
    created_datasets = []

    def traced_dataset_factory(*tensors):
        dataset = TracedTensorDataset(*tensors)
        created_datasets.append(dataset)
        return dataset

    with patch.object(native_module(), "TensorDataset", traced_dataset_factory):
        fitted = adapter.fit(windows, monitor)
    if len(created_datasets) != 1 or len(created_datasets[0]) != len(windows):
        raise RuntimeError("Native preprocessing changed the origin-to-window row mapping")
    fetched = np.asarray(created_datasets[0].fetched_indices, dtype=np.int64)
    batch = int(method_protocol["native_configuration"]["batch_size"])
    used_count = int(fitted["actual_optimizer_steps"]) * batch
    if len(fetched) < used_count or np.any(fetched < 0) or np.any(fetched >= len(windows)):
        raise RuntimeError("Native actual DataLoader sampling receipt is incomplete")
    pool_archive = np.load(sweep.resolve_path(pool["origin_npz"]), allow_pickle=False)
    eligible_origins = pool_archive["eligible_origins"].astype(np.int64)
    pool_archive.close()
    optimized_indices = fetched[:used_count].reshape(-1, batch)
    optimized_origins = eligible_origins[optimized_indices]
    order_path = CACHE / "ts2vec/sampling" / f"{name}_native_shuffle_stream.npz"
    order_path.parent.mkdir(parents=True, exist_ok=True)
    if order_path.exists():
        raise FileExistsError(order_path)
    np.savez_compressed(order_path, optimized_pool_indices=optimized_indices, optimized_origins=optimized_origins,
                        fetched_pool_indices=fetched, prefetched_after_budget_indices=fetched[used_count:])
    sampling_receipt = {
        "source": "native TensorDataset __getitem__ observer; returned tuple and RNG are unchanged",
        "data_order": "official DataLoader shuffle/drop_last, captured actual fetched indices",
        "optimized_updates": int(len(optimized_indices)),
        "optimized_windows": int(used_count),
        "optimized_origin_stream_sha256": array_hash(optimized_origins),
        "optimized_pool_index_stream_sha256": array_hash(optimized_indices),
        "fetched_windows": int(len(fetched)),
        "fetched_after_budget_windows": int(len(fetched) - used_count),
        "after_budget_note": "native loop fetches a batch before checking n_iters; fetched excess is not an optimizer update",
        "npz": order_path,
        "npz_sha256": sweep.sha256_path(order_path),
    }
    if array_hash(windows) != pool["window_sha256"]:
        raise RuntimeError("Native fitting mutated the common TRAIN window pool")
    ensure_method_sources(reference)
    checkpoint = CACHE / "ts2vec/checkpoints" / f"{name}_final_swa.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    adapter.save_checkpoint(checkpoint, pool)
    frozen_state = state_hash(adapter)
    fit = {
        "id": name, "dataset": bundle.name, "family": adapter.family,
        "k": int(bundle.latent), "seed": int(seed),
        "actual_optimizer_steps": int(fitted["actual_optimizer_steps"]),
        "native_iterations": int(fitted["native_iterations"]),
        "swa_n_averaged": int(fitted["swa_n_averaged"]),
        "encoder_selection": "final SWA; no VAL checkpoint selection",
        "early_stopping": False,
        "native_configuration": copy.deepcopy(method_protocol["native_configuration"]),
        "fit_receipt": fitted,
        "training_windows_shape": list(windows.shape),
        "training_provenance": copy.deepcopy(pool),
        "sampling_receipt": sampling_receipt,
        "algorithm": "native hierarchical contrastive/AdamW/SWA/crop/timestamp mask; official shuffle/drop_last data order",
        "wall_seconds": fitted["wall_seconds"],
        "process_cpu_seconds": fitted["process_cpu_seconds"],
        "encoder_params": int(encoder_params_single),
        "encoder_params_single": int(encoder_params_single),
        "resident_params_total": int(resident_params_total),
        "initial_encoder_state_sha256": initial_state,
        "encoder_state_sha256": frozen_state,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sweep.sha256_path(checkpoint),
        "selection": {"rule": "final SWA", "checkpoint": checkpoint, "checkpoint_sha256": sweep.sha256_path(checkpoint), "encoder_state_sha256": frozen_state},
    }
    monitor.progress["completed_fits"] += 1
    monitor.check("ts2vec_fit_complete", dataset=bundle.name, adapter=name)
    del adapter
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return fit


def run_dataset(dataset: str, common_protocol: dict, method_protocol: dict, device: str, partial: dict):
    latent = int(method_protocol["k"][dataset])
    bundle = sweep.load_bundle(dataset, latent)
    if latent != int(common_protocol["middle_k"][dataset]):
        raise RuntimeError("Native method must compare the same middle K")
    train_origins = sweep.decoder_train_origins(bundle, common_protocol)
    windows, pool = make_training_pool(bundle, method_protocol, sweep.MONITOR)
    result = {
        "dataset": dataset,
        "shape": list(bundle.x.shape),
        "channels": bundle.channels,
        "middle_k": latent,
        "k_grid": [latent],
        "source": {"path": bundle.source_path, "sha256": sweep.sha256_path(Path(bundle.source_path))},
        "training_pool": pool,
        "decoder_train_origins": {"count": int(len(train_origins)), "seed": int(common_protocol["decoder_sampling_seed"]), "sha256": array_hash(train_origins)},
        "fits": {}, "decoder_contracts": {}, "benchmark_cases": [],
    }
    partial["datasets"][dataset] = sweep.to_jsonable(result)
    sweep.atomic_json(HERE / "method_result.partial.json", partial)
    for seed in method_protocol["seeds"]:
        fit = train_native_fit(bundle, int(seed), windows, pool, method_protocol, device, partial["source_hashes"])
        result["fits"][fit["id"]] = fit
        partial["datasets"][dataset] = sweep.to_jsonable(result)
        sweep.atomic_json(HERE / "method_result.partial.json", partial)
    del windows
    gc.collect()
    parent = sweep.parent_runner()
    backbone = parent.old_pilot().load_backbone(dataset, device=device)
    before = parent.old_pilot().backbone_state_hash(backbone)
    try:
        for name, fit in result["fits"].items():
            ensure_method_sources(partial["source_hashes"])
            checkpoint = sweep.resolve_path(fit["checkpoint"])
            if sweep.sha256_path(checkpoint) != fit["checkpoint_sha256"]:
                raise RuntimeError("Native checkpoint changed before decoder fitting")
            adapter, checkpoint_payload = NativeTs2VecAdapter.load_checkpoint(checkpoint, device)
            encoder_before = state_hash(adapter)
            if encoder_before != fit["encoder_state_sha256"]:
                raise RuntimeError("Native checkpoint state did not round-trip")
            traces = native_forecast_trace(bundle, adapter, backbone, common_protocol, method_protocol, device, train_origins)
            contract = sweep.fit_forecast_decoder_from_traces(bundle, name, traces, common_protocol, train_origins)
            encoder_after = state_hash(adapter)
            if encoder_after != encoder_before or not all(parameter.grad is None for parameter in adapter.parameters()):
                raise RuntimeError("Frozen native encoder changed during forecast/readout fitting")
            result["decoder_contracts"][name] = {
                "meta": {"id": name, "dataset": dataset, "family": adapter.family, "kind": "ts2vec", "k": latent, "seed": int(fit["seed"]), "checkpoint": checkpoint, "checkpoint_sha256": fit["checkpoint_sha256"]},
                "encoder_state_sha256_before": encoder_before,
                "encoder_state_sha256_after": encoder_after,
                "encoder_grads_none": True,
                "contracts": {sweep.CONTRACT: contract},
                "export": copy.deepcopy(method_protocol["export"]),
                "encoder_params_single": fit["encoder_params_single"],
                "resident_params_total": fit["resident_params_total"],
            }
            result["benchmark_cases"].append(name)
            partial["datasets"][dataset] = sweep.to_jsonable(result)
            sweep.atomic_json(HERE / "method_result.partial.json", partial)
            del adapter, checkpoint_payload, traces
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        after = parent.old_pilot().backbone_state_hash(backbone)
        if before != after or not all(parameter.grad is None for parameter in backbone.parameters()):
            raise RuntimeError("Frozen TSFM changed during native method stage")
        result["backbone"] = {"state_sha256_before": before, "state_sha256_after": after, "unchanged": True, "grads_none": True}
    finally:
        del backbone
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return result


def run(args):
    if (HERE / "method_result.json").exists() or (HERE / "method_result.partial.json").exists():
        raise FileExistsError("Native method outputs already exist; inspect and record a continuation rather than overwrite")
    source_hashes = sealed_method_sources()
    sweep.configure_runtime()
    common_protocol = sweep.read_json(HERE / "protocol.json")
    method_protocol = sweep.read_json(HERE / "method_protocol.json")
    baseline = sweep.read_json(HERE / "result.json")
    if baseline.get("status") != "complete":
        raise RuntimeError("The K sweep must complete before native method comparison")
    if common_protocol["ridge_lambdas"] != method_protocol["decoder"]["ridge_lambdas"]:
        raise RuntimeError("Native method and K sweep must use the same precommitted ridge grid")
    fixture = sweep.read_json(HERE / "cpu_fixture_receipt.json")
    if fixture.get("status") != "PASS" or fixture.get("actual_smoke_optimizer_steps") != 1:
        raise RuntimeError("Native CPU compatibility fixtures must pass separately before science")
    CACHE.mkdir(parents=True, exist_ok=True)
    monitor_protocol = copy.deepcopy(common_protocol)
    monitor_protocol["limits"] = copy.deepcopy(method_protocol["limits"])
    sweep.MONITOR = sweep.Monitor(monitor_protocol, args.device, progress_name="method_progress.json")
    sweep.install_monitor_hooks()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    partial = {
        "schema": "native_ts2vec_k_tradeoff_result_v1",
        "status": "running",
        "protocol": method_protocol,
        "common_protocol": common_protocol,
        "effective_counts": {"neural_fit_starts": 4, "decoder_adapters": 4, "ridge_fits": 40, "benchmark_cases": 8},
        "parent": {"path": HERE / "result.json", "sha256": sweep.sha256_path(HERE / "result.json"), "schema": baseline["schema"], "status": baseline["status"]},
        "source_hashes": source_hashes,
        "dependency_hashes": sweep.dependency_hashes(),
        "vendor": vendor_provenance(),
        "cpu_fixture": {"path": HERE / "cpu_fixture_receipt.json", "sha256": sweep.sha256_path(HERE / "cpu_fixture_receipt.json"), "actual_smoke_optimizer_steps": 1, "included_in_scientific_fits": False},
        "datasets": {},
    }
    sweep.atomic_json(HERE / "method_result.partial.json", partial)
    try:
        for dataset in method_protocol["datasets"]:
            ensure_method_sources(source_hashes)
            partial["datasets"][dataset] = sweep.to_jsonable(run_dataset(dataset, common_protocol, method_protocol, args.device, partial))
            sweep.atomic_json(HERE / "method_result.partial.json", partial)
        ensure_method_sources(source_hashes)
        progress = sweep.MONITOR.progress
        if (progress["fit_starts"], progress["completed_fits"], progress["ridge_fits"]) != (4, 4, 40):
            raise RuntimeError("Native scientific execution count mismatch")
        if sweep.sha256_path(HERE / "result.json") != partial["parent"]["sha256"]:
            raise RuntimeError("K baseline result changed during native method stage")
        sweep.monitor("complete", status="complete")
        partial["status"] = "complete"
        partial["finished_utc"] = time.time()
        partial["execution_counts"] = sweep.MONITOR.receipt()
        sweep.atomic_json(HERE / "method_result.json", partial)
        return partial
    except Exception as exc:
        partial["status"] = "failed"
        partial["failed_utc"] = time.time()
        partial["error"] = {"type": type(exc).__name__, "message": str(exc)}
        partial["execution_counts"] = copy.deepcopy(sweep.MONITOR.progress)
        partial["execution_counts"]["status"] = "failed"
        sweep.atomic_json(HERE / "method_progress.json", partial["execution_counts"])
        sweep.atomic_json(HERE / "method_result.partial.json", partial)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "execution_counts": result["execution_counts"]}, ensure_ascii=False))
