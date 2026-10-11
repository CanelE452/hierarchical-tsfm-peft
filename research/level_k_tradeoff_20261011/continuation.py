from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import sweep

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
ATTEMPT1 = HERE / "failed_attempts" / "attempt1"
ATTEMPT2 = HERE / "failed_attempts" / "attempt2"
ATTEMPT2_PARTIAL = ATTEMPT2 / "result.partial.json"
PRESERVATION = HERE / "failed_attempt2_preservation.json"
CONTRACT = sweep.CONTRACT


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_map(manifest: dict) -> dict[str, dict]:
    return {str(row["path"]).replace("\\", "/"): row for row in manifest.get("archived_artifacts", [])}


def archived_artifact(manifest: dict, path: Path) -> dict:
    key = sweep.public_path(path).replace("\\", "/")
    mapping = artifact_map(manifest)
    if key not in mapping:
        raise RuntimeError(f"archive manifest lacks {key}")
    row = mapping[key]
    if sweep.sha256_path(path) != row["sha256"]:
        raise RuntimeError(f"archived artifact hash mismatch: {path}")
    return row


def active_cache_counterpart(archived_cache_path: Path) -> Path:
    parts = archived_cache_path.parts
    try:
        cache_index = parts.index("cache")
    except ValueError as exc:
        raise RuntimeError(f"not an archived cache path: {archived_cache_path}") from exc
    return CACHE.joinpath(*parts[cache_index + 1 :])


def validate_active_cache_reuse(manifest: dict) -> dict:
    mapping = artifact_map(manifest)
    checked = []
    for path in sorted(CACHE.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(CACHE)
        archived_key = sweep.public_path(ATTEMPT2 / "cache" / rel).replace("\\", "/")
        if archived_key not in mapping:
            raise RuntimeError(f"active cache file is not attempt2-preserved: {path}")
        actual = sweep.sha256_path(path)
        expected = mapping[archived_key]["sha256"]
        if actual != expected:
            raise RuntimeError(f"active cache hash mismatch: {path}")
        checked.append({"path": path, "sha256": actual, "bytes": path.stat().st_size})
    return {"checked_files": len(checked), "checked_bytes": int(sum(row["bytes"] for row in checked)), "files": checked}


def validate_attempt2_partial(manifest: dict) -> dict:
    row = archived_artifact(manifest, ATTEMPT2_PARTIAL)
    partial = read_json(ATTEMPT2_PARTIAL)
    if partial.get("status") != "failed":
        raise RuntimeError("attempt2 partial must be a preserved failed result")
    robin = partial.get("datasets", {}).get("robin", {})
    jena = partial.get("datasets", {}).get("jena", {})
    if len(robin.get("fits", {})) != 6 or len(robin.get("decoder_contracts", {})) != 10:
        raise RuntimeError("attempt2 partial does not contain completed Robin fits and 10 decoder adapters")
    if set(jena.get("fits", {})) != {"jena_MLP_POINT_K3_seed92601"}:
        raise RuntimeError("attempt2 partial Jena completed fit set is not the expected single fit")
    if jena.get("decoder_contracts"):
        raise RuntimeError("attempt2 partial must not contain Jena decoder contracts")
    return {"partial": partial, "partial_artifact": row}


def validate_completed_fit_paths(dataset_result: dict) -> dict:
    checked = []
    for fit_id, fit in dataset_result.get("fits", {}).items():
        if fit.get("reused_from_parent"):
            continue
        for key in ("checkpoint", "latest_checkpoint", "curve"):
            path = sweep.resolve_path(fit[key])
            expected = fit[f"{key}_sha256"] if key != "checkpoint" else fit["checkpoint_sha256"]
            actual = sweep.sha256_path(path)
            if actual != expected:
                raise RuntimeError(f"completed fit artifact changed: {fit_id}:{key}:{path}")
            checked.append({"fit": fit_id, "key": key, "path": path, "sha256": actual})
    return {"checked_fit_artifacts": len(checked), "artifacts": checked}


def validate_robin_decoder_paths(robin: dict) -> dict:
    checked = []
    for adapter, entry in robin.get("decoder_contracts", {}).items():
        contract = entry["contracts"][CONTRACT]
        path = sweep.resolve_path(contract["prediction_npz"])
        actual = sweep.sha256_path(path)
        if actual != contract["prediction_sha256"]:
            raise RuntimeError(f"Robin decoder prediction changed: {adapter}:{path}")
        checked.append({"adapter": adapter, "path": path, "sha256": actual})
    if len(checked) != 10:
        raise RuntimeError("Robin decoder reuse count must be 10")
    return {"checked_robin_decoders": len(checked), "artifacts": checked}


def continuation_source_hashes() -> dict:
    out = sweep.source_hashes()
    out["continuation"] = sweep.sha256_path(HERE / "continuation.py")
    return out


def assert_no_continuation_overwrite() -> None:
    if (HERE / "result.json").exists():
        raise RuntimeError("result.json already exists; refusing continuation overwrite")
    for name in (
        "jena_MLP_POINT_K3_seed92602",
        "jena_MLP_POINT_K12_seed92601",
        "jena_MLP_POINT_K12_seed92602",
    ):
        for suffix in ("_best180.pt", "_latest.pt"):
            path = CACHE / "checkpoints" / f"{name}{suffix}"
            if path.exists():
                raise RuntimeError(f"continuation target checkpoint already exists: {path}")
        curve = CACHE / "curves" / f"{name}.json"
        if curve.exists():
            raise RuntimeError(f"continuation target curve already exists: {curve}")
    for path in (CACHE / "predictions").glob("jena_*_FORECAST_RIDGE_D8_val.npz") if (CACHE / "predictions").exists() else []:
        raise RuntimeError(f"Jena decoder target already exists: {path}")


def fresh_fit_ids(protocol: dict) -> list[tuple[int, int]]:
    return [(3, 92602), (12, 92601), (12, 92602)]


def run_jena_continuation(protocol: dict, parent: dict, jena: dict, device: str, partial: dict) -> dict:
    bundles = sweep.build_dataset_bundles("jena", protocol)
    original = sweep.load_original_bundle("jena")
    dec_origins = sweep.decoder_train_origins(original, protocol)
    jena = copy.deepcopy(jena)
    jena["raw_f0"] = sweep.raw_f0_from_parent(parent, "jena")
    jena["decoder_train_origins"] = {"count": int(len(dec_origins)), "seed": int(protocol["decoder_sampling_seed"]), "sha256": sweep.array_sha256(dec_origins)}
    for k, bundle in bundles.items():
        key = f"K{k}"
        if key not in jena.get("basis", {}):
            kind = "identity_raw_c" if k == original.channels else ("parent_middle_pca" if k == original.latent else "new_train_centered_pca")
            jena.setdefault("basis", {})[key] = sweep.basis_receipt(bundle, kind)

    for k, seed in fresh_fit_ids(protocol):
        fit = sweep.train_mlp_new_k(bundles[k], seed, protocol, device)
        jena["fits"][fit["id"]] = fit
        partial["datasets"]["jena"] = sweep.to_jsonable(jena)
        sweep.atomic_json(HERE / "result.continuation.partial.json", partial)
    middle = int(protocol["middle_k"]["jena"])
    for seed in protocol["seeds"]:
        fit = sweep.load_parent_mlp_fit(parent, "jena", middle, int(seed))
        jena["fits"][fit["id"]] = fit

    pr = sweep.parent_runner()
    backbone = pr.old_pilot().load_backbone("jena", device=device)
    backbone_before = pr.old_pilot().backbone_state_hash(backbone)
    try:
        for row in sweep.adapter_plan(protocol, "jena"):
            sweep.ensure_source_hashes_unchanged(partial["source_hashes"])
            name = row["id"]
            family = row["family"]
            bundle = bundles[int(row["k"])]
            sweep.monitor(stage=f"continuation_decoder:{name}", dataset="jena", adapter=name)
            if family == "PCA":
                encoder_obj = bundle.basis
                encoder_state = sweep.array_sha256(bundle.basis.astype(np.float32))
                parent_adapter = "PCA_ANCHOR" if int(row["k"]) == middle else None
            elif family == "RAW_C_D8":
                encoder_obj = bundle.basis
                encoder_state = sweep.array_sha256(bundle.basis.astype(np.float32))
                parent_adapter = None
            elif family == "MLP_POINT":
                fit = jena["fits"][name]
                encoder_obj, encoder_state = sweep.load_mlp_encoder_from_fit(bundle, fit)
                parent_adapter = fit.get("parent_adapter") if fit.get("reused_from_parent") else None
            else:
                raise ValueError(family)
            if parent_adapter is not None:
                traces = sweep.load_reused_traces(parent, "jena", parent_adapter, bundle, dec_origins, int(protocol["decoder_max_lag"]))
            elif family == "RAW_C_D8":
                traces = sweep.raw_c_traces(bundle, backbone, protocol, device, dec_origins, jena["raw_f0"])
            else:
                if isinstance(encoder_obj, torch.nn.Module):
                    for parameter in encoder_obj.parameters():
                        parameter.requires_grad_(False)
                    encoder_obj = encoder_obj.to(device).eval()
                traces = sweep.compute_traces(bundle, encoder_obj, backbone, protocol, device, dec_origins)
            entry = {
                "meta": copy.deepcopy(row),
                "encoder_state_sha256_before": encoder_state,
                "contracts": {CONTRACT: sweep.fit_forecast_decoder_from_traces(bundle, name, traces, protocol, dec_origins)},
            }
            if isinstance(encoder_obj, torch.nn.Module):
                after = sweep.model_state_hash(encoder_obj.cpu())
                if after != encoder_state:
                    raise RuntimeError(f"encoder changed during decoder fitting: {name}")
            else:
                after = encoder_state
            entry["encoder_state_sha256_after"] = after
            jena.setdefault("decoder_contracts", {})[name] = entry
            jena["benchmark_cases"] = ["RAW_F0"] + [planned["id"] for planned in sweep.adapter_plan(protocol, "jena")]
            partial["datasets"]["jena"] = sweep.to_jsonable(jena)
            sweep.atomic_json(HERE / "result.continuation.partial.json", partial)
        backbone_after = pr.old_pilot().backbone_state_hash(backbone)
        if backbone_before != backbone_after:
            raise RuntimeError("Jena frozen backbone changed")
        if not all(parameter.grad is None for parameter in backbone.parameters()):
            raise RuntimeError("Jena frozen backbone accumulated gradients")
        jena["backbone"] = {"state_sha256_before": backbone_before, "state_sha256_after": backbone_after, "unchanged": True, "grads_none": True}
        return jena
    finally:
        del backbone
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def stage_receipt(path: Path) -> dict:
    if not path.exists():
        return {}
    return read_json(path)


def verify_continuation_seal() -> dict:
    path = HERE / "source_seal.continuation.json"
    if not path.exists():
        raise FileNotFoundError(path)
    seal = read_json(path)
    if seal.get("phase") != "before_stage_science" or seal.get("stage") != "continuation":
        raise RuntimeError(f"invalid continuation source seal phase/stage: {path}")
    artifacts = seal.get("artifacts", [])
    expected = {
        "research/level_k_tradeoff_20261011/PLAN.md",
        "research/level_k_tradeoff_20261011/protocol.json",
        "research/level_k_tradeoff_20261011/sweep.py",
        "research/level_k_tradeoff_20261011/benchmark.py",
        "research/level_k_tradeoff_20261011/test_sweep.py",
        "research/level_k_tradeoff_20261011/seal.py",
        "research/level_k_tradeoff_20261011/continuation.py",
    }
    seen = {str(row.get("path", "")).replace("\\", "/") for row in artifacts}
    if seen != expected:
        raise RuntimeError({"continuation_seal_artifacts_mismatch": {"expected": sorted(expected), "seen": sorted(seen)}})
    checked = []
    for row in artifacts:
        artifact_path = sweep.resolve_path(row["path"])
        actual_sha = sweep.sha256_path(artifact_path)
        if actual_sha != row["sha256"]:
            raise RuntimeError(f"continuation source seal SHA mismatch: {artifact_path}")
        actual_bytes = artifact_path.stat().st_size
        if "bytes" in row and int(row["bytes"]) != actual_bytes:
            raise RuntimeError(f"continuation source seal byte-size mismatch: {artifact_path}")
        checked.append({"path": row["path"], "sha256": actual_sha, "bytes": actual_bytes})
    return {"path": path, "sha256": sweep.sha256_path(path), "artifacts": checked}


def assert_continuation_counts(current: dict) -> dict:
    counts = {
        "fit_starts": int(current.get("fit_starts", 0)),
        "completed_fits": int(current.get("completed_fits", 0)),
        "ridge_fits": int(current.get("ridge_fits", 0)),
    }
    expected = {"fit_starts": 3, "completed_fits": 3, "ridge_fits": 100}
    if counts != expected:
        raise RuntimeError({"unexpected_continuation_counts": {"expected": expected, "actual": counts}})
    return counts


def clean_inherited_failure_fields(partial: dict) -> None:
    for key in ("error", "failed_utc"):
        partial.pop(key, None)


def aggregate_execution_counts(current: dict) -> dict:
    continuation_counts = assert_continuation_counts(current)
    attempt1 = read_json(ATTEMPT1 / "failure_ledger.json")
    attempt2 = read_json(ATTEMPT2 / "failure_ledger.json")
    current_counts = {
        **continuation_counts,
        "wall_seconds": float(current.get("elapsed_seconds", 0.0)),
        "process_cpu_seconds": float(current.get("process_cpu_seconds", 0.0)),
        "rss_bytes": int(current.get("rss_bytes", 0)),
        "cuda_max_allocated_bytes": int(current.get("cuda_max_allocated_bytes", 0)),
        "cuda_max_reserved_bytes": int(current.get("cuda_max_reserved_bytes", 0)),
    }
    attempt1_progress = attempt1.get("failure_progress", {})
    attempt2_progress = attempt2.get("failure_progress", {})
    wall_sum = float(attempt1["wall_seconds"]) + float(attempt2["wall_seconds"]) + current_counts["wall_seconds"]
    cpu_sum = float(attempt1["process_cpu_seconds"]) + float(attempt2["process_cpu_seconds"]) + current_counts["process_cpu_seconds"]
    peak_rss = max(int(attempt1_progress.get("rss_bytes", 0)), int(attempt2_progress.get("rss_bytes", 0)), current_counts["rss_bytes"])
    peak_alloc = max(int(attempt1_progress.get("cuda_max_allocated_bytes", 0)), int(attempt2_progress.get("cuda_max_allocated_bytes", 0)), current_counts["cuda_max_allocated_bytes"])
    peak_reserved = max(int(attempt1_progress.get("cuda_max_reserved_bytes", 0)), int(attempt2_progress.get("cuda_max_reserved_bytes", 0)), current_counts["cuda_max_reserved_bytes"])
    return {
        "fit_starts": int(attempt1["fit_starts"]) + int(attempt2["fit_starts"]) + current_counts["fit_starts"],
        "completed_fits": int(attempt1["completed_fits"]) + int(attempt2["completed_fits"]) + current_counts["completed_fits"],
        "ridge_fits": int(attempt1["ridge_fits"]) + int(attempt2["ridge_fits"]) + current_counts["ridge_fits"],
        "selected_fit_updates": 23040,
        "all_baseline_optimizer_steps_lower": 24384,
        "all_baseline_optimizer_steps_upper": 24415,
        "continuation_counts": continuation_counts,
        "stage_wall_seconds_sum": wall_sum,
        "stage_process_cpu_seconds_sum": cpu_sum,
        "stage_peak_rss_bytes_max": peak_rss,
        "stage_peak_cuda_allocated_bytes_max": peak_alloc,
        "stage_peak_cuda_reserved_bytes_max": peak_reserved,
        "attempts": {
            "attempt1": {"fit_starts": attempt1["fit_starts"], "completed_fits": attempt1["completed_fits"], "ridge_fits": attempt1["ridge_fits"], "wall_seconds": attempt1["wall_seconds"], "process_cpu_seconds": attempt1["process_cpu_seconds"]},
            "attempt2": {"fit_starts": attempt2["fit_starts"], "completed_fits": attempt2["completed_fits"], "ridge_fits": attempt2["ridge_fits"], "wall_seconds": attempt2["wall_seconds"], "process_cpu_seconds": attempt2["process_cpu_seconds"]},
            "continuation": current_counts,
        },
        "native4_expected_final_task_starts": 14,
        "native4_expected_completed_fits": 12,
        "native4_successful_updates_total": 34560,
        "native4_total_optimizer_steps_lower": 35904,
        "native4_total_optimizer_steps_upper": 35935,
    }


def run(args) -> dict:
    sweep.configure_runtime()
    assert_no_continuation_overwrite()
    protocol = sweep.read_json(HERE / "protocol.json")
    continuation_protocol = copy.deepcopy(protocol)
    continuation_protocol["limits"] = copy.deepcopy(protocol["limits"])
    continuation_protocol["limits"]["neural_fit_starts"] = 3
    continuation_protocol["limits"]["ridge_fits"] = 100
    manifest = read_json(PRESERVATION)
    seal_before = verify_continuation_seal()
    attempt2_validation = validate_attempt2_partial(manifest)
    active_reuse = validate_active_cache_reuse(manifest)
    partial = copy.deepcopy(attempt2_validation["partial"])
    clean_inherited_failure_fields(partial)
    parent = sweep.load_parent(protocol)
    sweep.MONITOR = sweep.Monitor(continuation_protocol, args.device, progress_name="continuation_progress.json")
    sweep.install_monitor_hooks()
    current_sources = sweep.source_hashes()
    partial.update(
        status="continuation_running",
        continuation={
            "schema": "k_tradeoff_continuation_v1",
            "started_utc": time.time(),
            "attempt2_partial": attempt2_validation["partial_artifact"],
            "active_cache_reuse": active_reuse,
            "source_hashes_with_continuation": continuation_source_hashes(),
            "source_seal_continuation": seal_before,
            "new_stage_limits": {"neural_fit_starts": 3, "ridge_fits": 100},
        },
        source_hashes=current_sources,
        dependency_hashes=sweep.dependency_hashes(),
    )
    partial["datasets"]["robin"]["reuse_validation"] = {
        "fits": validate_completed_fit_paths(partial["datasets"]["robin"]),
        "decoders": validate_robin_decoder_paths(partial["datasets"]["robin"]),
    }
    partial["datasets"]["jena"]["reuse_validation"] = {
        "fits": validate_completed_fit_paths(partial["datasets"]["jena"]),
    }
    sweep.atomic_json(HERE / "result.continuation.partial.json", partial)
    try:
        partial["datasets"]["jena"] = sweep.to_jsonable(
            run_jena_continuation(continuation_protocol, parent, partial["datasets"]["jena"], args.device, partial)
        )
        partial["status"] = "complete"
        partial["schema"] = "k_tradeoff_result_v1"
        partial["finished_utc"] = time.time()
        sweep.MONITOR.progress["status"] = "complete"
        sweep.MONITOR.progress["stage"] = "continuation_complete"
        current_counts = sweep.MONITOR.receipt()
        seal_after = verify_continuation_seal()
        partial["execution_counts"] = aggregate_execution_counts(current_counts)
        partial["continuation"]["finished_utc"] = partial["finished_utc"]
        partial["continuation"]["current_stage_counts"] = current_counts
        partial["continuation"]["source_seal_continuation_after"] = seal_after
        clean_inherited_failure_fields(partial)
        sweep.atomic_json(HERE / "result.json", partial)
        return partial
    except Exception as exc:
        partial["status"] = "continuation_failed"
        partial["failed_utc"] = time.time()
        partial["error"] = {"type": type(exc).__name__, "message": str(exc)}
        partial["continuation"]["current_stage_counts"] = sweep.MONITOR.receipt() if sweep.MONITOR is not None else {}
        sweep.atomic_json(HERE / "result.continuation.partial.json", partial)
        raise


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
