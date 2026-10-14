"""CPU-only independent audit for RF1/RF3 forecast-sequence runs.

Importing this file is source-only.  Actual producer/result/NPZ/checkpoint reads
only happen with ``--root-go`` after the resource extension gate is active.
The audit never constructs the old engine/F model, never calls ``.cuda()``, and
does not claim independent replay of past-latent forecasting; it reconstructs
saved decoder/head arithmetic from stored latents and CPU-loaded checkpoint
state dictionaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

DEFAULT_QUEUE = ROOT / ".cache" / STUDY.name / "forecast_sequence_v1" / "FORECAST_SEQUENCE_QUEUE_v1.json"
DEFAULT_OUT = HERE / "FORECAST_SEQUENCE_AUDIT_v1.json"

DATASETS = {"robin": {"C": 17, "K": 4}, "beijing_aotizhongxin": {"C": 11, "K": 2}}
SEEDS = (92601, 92602)
HEADS = ("STATIC_RF1", "TEMPORAL_RF3")
HEAD_TAG = {"STATIC_RF1": "static", "TEMPORAL_RF3": "temporal"}
CHECKPOINTS = (0, 100, 200, 400, 600)
FIT_TUNE_SPLITS = ("fit", "tune")
SPLITS_BY_DATASET = {"robin": ("fit", "tune", "val"), "beijing_aotizhongxin": ("fit", "tune", "eval")}
TRACE_FIELDS = ("prediction", "past_z_last_two", "absolute_future_z", "base_MAIN_style_prediction", "head_residual", "raw_last")
L = 512
H = 48
PARITY_ATOL = 1e-5
PARITY_RTOL = 2e-6
RUNNER_SOURCE = STUDY / "run_forecast_sequence_v1.py"
DECODER_SOURCE = STUDY / "forecast_sequence_decoder.py"
EXPECTED_RUNNER_SHA = "bace190480f46f11cd51ced5048d74e06d94559b2b0c8756d63e9ddf35ec76b7"
EXPECTED_DECODER_SHA = "f61bbeb1bdb104c420bbf46cd41d3b8b74c1d9931dc326739191b8303108853d"


class AuditError(RuntimeError):
    pass


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def rel(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def guard_no_test(path: str | Path, role: str) -> None:
    value = Path(path)
    if any(part.lower() == "test" for part in value.parts) or value.name.lower() in {"test.npz", "jena_test.npz"}:
        raise PermissionError(f"PROTECTED_TEST_PATH_DENIED:{role}:{path}")


def read_json_path(path: str | Path) -> dict[str, Any]:
    path = resolve(path)
    guard_no_test(path, "json")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def old_state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in state.items():
        if hasattr(value, "detach"):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def bi_state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if hasattr(value, "detach"):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def combined_bi_hash(codec_state: Mapping[str, Any], head_state: Mapping[str, Any]) -> str:
    combined = {"codec." + key: value for key, value in codec_state.items()}
    combined.update({"head." + key: value for key, value in head_state.items()})
    return bi_state_hash(combined)


def checked_npz(record: Mapping[str, Any], role: str, checks, artifacts: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    path = resolve(record["path"])
    guard_no_test(path, role)
    before = sha(path)
    checks(f"{role}:file_sha", before == record.get("sha256"), {"path": rel(path), "actual": before, "expected": record.get("sha256")})
    with np.load(path, allow_pickle=False) as opened:
        arrays = {name: opened[name].copy() for name in opened.files}
    declared = record.get("arrays")
    if declared is not None:
        checks(f"{role}:array_keys", set(arrays) == set(declared), {"actual": sorted(arrays), "expected": sorted(declared)})
        for name, value in arrays.items():
            meta = declared[name]
            checks(
                f"{role}:{name}:contract",
                list(value.shape) == meta["shape"] and str(value.dtype) == meta["dtype"] and array_sha(value) == meta["sha256"],
                {"array": name, "shape": list(value.shape), "dtype": str(value.dtype)},
            )
    after = sha(path)
    checks(f"{role}:unchanged", after == before, {"path": rel(path)})
    artifacts.append({"role": role, "path": rel(path), "sha256_before": before, "sha256_after": after})
    return arrays


def checked_checkpoint(record: Mapping[str, Any], role: str, checks, artifacts: list[dict[str, Any]], torch_module) -> dict[str, Any]:
    path = resolve(record["path"])
    guard_no_test(path, role)
    before = sha(path)
    checks(f"{role}:file_sha", before == record.get("sha256"), {"path": rel(path), "actual": before, "expected": record.get("sha256")})
    payload = torch_module.load(path, map_location="cpu", weights_only=True)
    after = sha(path)
    checks(f"{role}:unchanged", after == before, {"path": rel(path)})
    codec = payload.get("codec", {})
    head = payload.get("head", {})
    if "state_sha_old_algorithm" in record:
        checks(f"{role}:codec_old_hash", old_state_hash(codec) == record["state_sha_old_algorithm"])
    if "state_sha_Bi_algorithm" in record:
        checks(f"{role}:codec_Bi_hash", bi_state_hash(codec) == record["state_sha_Bi_algorithm"])
    if "head_state_sha_old_algorithm" in record:
        checks(f"{role}:head_old_hash", old_state_hash(head) == record["head_state_sha_old_algorithm"])
    if "head_state_sha_Bi_algorithm" in record:
        checks(f"{role}:head_Bi_hash", bi_state_hash(head) == record["head_state_sha_Bi_algorithm"])
    if "combined_state_sha_Bi_algorithm" in record:
        checks(f"{role}:combined_Bi_hash", combined_bi_hash(codec, head) == record["combined_state_sha_Bi_algorithm"])
    artifacts.append({"role": role, "path": rel(path), "sha256_before": before, "sha256_after": after})
    return payload


def snapshot_audit_sources(out_path: Path, checks) -> list[dict[str, Any]]:
    folder = out_path.with_suffix(".source_snapshot")
    checks("audit_source_snapshot_absent", not folder.exists(), {"path": rel(folder)})
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    records = []
    for source in (
        Path(__file__).resolve(),
        RUNNER_SOURCE,
        DECODER_SOURCE,
        STUDY / "continuation_runtime_v2.py",
        STUDY / "continuation_runtime_v1.py",
        STUDY / "fast_storage_scan_v1.py",
    ):
        before = sha(source)
        destination = folder / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        after = sha(source)
        copied = sha(destination)
        checks(f"audit_source_snapshot:{source.name}", before == after == copied, {"source": rel(source), "snapshot": rel(destination)})
        records.append({"path": rel(source), "sha256": before, "snapshot_path": rel(destination), "snapshot_sha256": copied})
    manifest = folder / "MANIFEST.json"
    manifest.write_text(json.dumps({"sources": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records.append({"path": rel(manifest), "sha256": sha(manifest), "snapshot_path": rel(manifest), "snapshot_sha256": sha(manifest)})
    return records


def require_exact(name: str, actual: np.ndarray, expected: np.ndarray, checks) -> None:
    ok = actual.dtype == expected.dtype and actual.shape == expected.shape and np.array_equal(actual, expected, equal_nan=True)
    details = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape), "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}
    if not ok and actual.shape == expected.shape:
        diff = actual.astype(np.float64) - expected.astype(np.float64)
        details["max_abs"] = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
    checks(name, ok, details)


def require_close(name: str, actual: np.ndarray, expected: np.ndarray, checks, *, atol: float = PARITY_ATOL, rtol: float = PARITY_RTOL) -> None:
    ok = actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    details = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape), "atol": atol, "rtol": rtol}
    if actual.shape == expected.shape:
        details["max_abs"] = float(np.nanmax(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0.0
    checks(name, bool(ok), details)


def splits_for(dataset: str) -> tuple[str, str, str]:
    return SPLITS_BY_DATASET[dataset]


def prefixed_trace(arrays: Mapping[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    return {name: arrays[f"{split}_{name}"] for name in TRACE_FIELDS}


def array_score(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray, std: np.ndarray | None = None) -> dict[str, Any]:
    err = prediction.astype(np.float64) - target.astype(np.float64)
    if std is not None:
        err = err * std
    counts = observed.sum((0, 1))
    mse, mae = [], []
    for channel in range(err.shape[-1]):
        mask = observed[..., channel].astype(bool)
        if mask.any():
            values = err[..., channel][mask]
            mse.append(float(np.mean(values ** 2)))
            mae.append(float(np.mean(np.abs(values))))
        else:
            mse.append(None)
            mae.append(None)
    active_mse = [value for value in mse if value is not None]
    active_mae = [value for value in mae if value is not None]
    return {
        "mse": float(np.mean(active_mse)) if active_mse else None,
        "mae": float(np.mean(active_mae)) if active_mae else None,
        "per_channel_mse": mse,
        "per_channel_mae": mae,
        "observed_counts": counts.tolist(),
        "active_channels": int((counts > 0).sum()),
    }


def score_all(prediction: np.ndarray, values: Mapping[str, np.ndarray], std: np.ndarray) -> dict[str, Any]:
    target, observed, raw = values["target"], values["observed"], values["raw_forecast"]
    return {
        "normalized_actual_Y": array_score(prediction, target, observed),
        "physical_actual_Y": array_score(prediction, target, observed, std),
        "normalized_RAW_teacher": array_score(prediction, raw, observed),
    }


def close_record_value(actual: Any, expected: Any, *, atol: float, rtol: float) -> bool:
    if expected is None or actual is None:
        return actual is None and expected is None
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            close_record_value(left, right, atol=atol, rtol=rtol) for left, right in zip(actual, expected)
        )
    if isinstance(expected, (int, np.integer)) and isinstance(actual, (int, np.integer)):
        return int(actual) == int(expected)
    return bool(np.isclose(float(actual), float(expected), atol=atol, rtol=rtol))


def compare_score_record(name: str, actual: Mapping[str, Any], declared: Mapping[str, Any], checks, *, atol=1e-11, rtol=1e-11) -> None:
    for group, values in actual.items():
        checks(f"{name}:{group}:present", group in declared, {"available": sorted(declared)})
        if group not in declared:
            continue
        checks(f"{name}:{group}:keys", set(values) == set(declared[group]), {"actual": sorted(values), "expected": sorted(declared[group])})
        for metric, expected in declared[group].items():
            value = values.get(metric)
            checks(f"{name}:{group}:{metric}", close_record_value(value, expected, atol=atol, rtol=rtol), {"actual": value, "expected": expected})


def torch_tensor(torch_module, value: Any):
    return torch_module.as_tensor(value, dtype=torch_module.float32, device="cpu")


def linear(torch_module, x, weight, bias):
    return torch_module.matmul(x, weight.t()) + bias


def codec_decode(torch_module, codec_state: Mapping[str, Any], z: np.ndarray):
    zt = torch_tensor(torch_module, z)
    if "codec.decode.weight" in codec_state:
        decoded = linear(torch_module, zt, codec_state["codec.decode.weight"], codec_state["codec.decode.bias"])
    else:
        decoded = linear(torch_module, zt, codec_state["decode.weight"], codec_state["decode.bias"])
    if "residual.0.weight" in codec_state:
        hidden = torch_module.nn.functional.gelu(
            linear(torch_module, zt, codec_state["residual.0.weight"], codec_state["residual.0.bias"]),
            approximate="none",
        )
        decoded = decoded + linear(torch_module, hidden, codec_state["residual.2.weight"], codec_state["residual.2.bias"])
    return decoded


def head_residual(torch_module, head_type: str, head_state: Mapping[str, Any], past_z_last_two: np.ndarray, future_z: np.ndarray):
    past = torch_tensor(torch_module, past_z_last_two)
    future = torch_tensor(torch_module, future_z)
    if head_type == "STATIC_RF1":
        last = past[:, -1:].expand(-1, future.shape[1], -1)

        def response(values):
            hidden = torch_module.nn.functional.gelu(linear(torch_module, values, head_state["hidden.weight"], head_state["hidden.bias"]))
            return linear(torch_module, hidden, head_state["output.weight"], head_state["output.bias"])

        return response(future) - response(last)
    if head_type == "TEMPORAL_RF3":
        repeated_last = past[:, -1:].expand(-1, future.shape[1], -1)

        def response(values):
            sequence = torch_module.cat((past, values), dim=1).transpose(1, 2)
            hidden = torch_module.nn.functional.gelu(torch_module.nn.functional.conv1d(sequence, head_state["hidden.weight"], head_state["hidden.bias"]))
            return torch_module.nn.functional.conv1d(hidden, head_state["output.weight"], head_state["output.bias"]).transpose(1, 2)

        return response(future) - response(repeated_last)
    raise ValueError(f"UNKNOWN_HEAD:{head_type}")


def reconstruct_saved_trace(torch_module, codec_state: Mapping[str, Any], head_state: Mapping[str, Any], head_type: str, arrays: Mapping[str, np.ndarray]):
    future_decoded = codec_decode(torch_module, codec_state, arrays["absolute_future_z"]).detach().cpu().numpy()
    last_decoded = codec_decode(torch_module, codec_state, arrays["past_z_last_two"][:, -1:]).detach().cpu().numpy()
    base = arrays["raw_last"] + future_decoded
    base = base - last_decoded
    residual = head_residual(torch_module, head_type, head_state, arrays["past_z_last_two"], arrays["absolute_future_z"]).detach().cpu().numpy()
    prediction = base + residual
    return base.astype(np.float32), residual.astype(np.float32), prediction.astype(np.float32)


def parameter_delta_groups(initial_codec, final_codec, initial_head, final_head) -> dict[str, Any]:
    groups = {"E": {"changed": 0, "parameters": []}, "D": {"changed": 0, "parameters": []}, "Head": {"changed": 0, "parameters": []}}
    for name, initial in initial_codec.items():
        if name == "fixed_basis":
            continue
        final = final_codec[name]
        group = "D" if name.startswith("codec.decode.") or name.startswith("decode.") or name.startswith("residual.") else "E"
        delta = final.detach().cpu().numpy().astype(np.float64) - initial.detach().cpu().numpy().astype(np.float64)
        changed = bool(np.any(delta != 0))
        groups[group]["changed"] += int(changed)
        groups[group]["parameters"].append({"name": name, "changed": changed, "max_abs": float(np.abs(delta).max()) if delta.size else 0.0, "l2": float(np.linalg.norm(delta))})
    for name, initial in initial_head.items():
        final = final_head[name]
        delta = final.detach().cpu().numpy().astype(np.float64) - initial.detach().cpu().numpy().astype(np.float64)
        changed = bool(np.any(delta != 0))
        groups["Head"]["changed"] += int(changed)
        groups["Head"]["parameters"].append({"name": name, "changed": changed, "max_abs": float(np.abs(delta).max()) if delta.size else 0.0, "l2": float(np.linalg.norm(delta))})
    return groups


def make_limit(tag: str, *, gpu: bool):
    from continuation_runtime_v2 import ContinuationLimit

    return ContinuationLimit(tag, gpu=gpu)


def producer_refs_from_inputs(paths: list[Path], checks) -> list[dict[str, Any]]:
    refs = []
    for index, path in enumerate(paths):
        resolved = resolve(path)
        guard_no_test(resolved, f"sequence_result_{index}")
        data = json.loads(resolved.read_text(encoding="utf-8-sig"))
        digest = sha(resolved)
        if "conditions" in data:
            checks(f"queue_{index}:status", data.get("status") == "COMPLETE", {"status": data.get("status")})
            for condition_index, item in enumerate(data["conditions"]):
                ref = dict(item["producer_result"])
                ref["_queue_condition"] = {key: item.get(key) for key in ("tag", "dataset", "seed", "head", "status", "selected_new_update")}
                ref["_queue_path"] = rel(resolved)
                ref["_queue_sha256"] = digest
                ref["_queue_index"] = condition_index
                refs.append(ref)
        else:
            refs.append({"path": str(resolved), "sha256": digest})
    checks("producer_ref_count", len(refs) == 8, {"count": len(refs), "paths": [ref.get("path") for ref in refs]})
    return refs


def load_producer(ref: Mapping[str, Any], index: int, checks) -> dict[str, Any]:
    path = resolve(ref["path"])
    actual = sha(path)
    checks(f"producer_{index}:sha", actual == ref.get("sha256"), {"path": rel(path), "actual": actual, "expected": ref.get("sha256")})
    producer = json.loads(path.read_text(encoding="utf-8-sig"))
    producer["_producer_result"] = {"path": ref["path"], "sha256": ref["sha256"]}
    if "_queue_condition" in ref:
        queue = ref["_queue_condition"]
        checks(
            f"producer_{index}:queue_identity",
            queue.get("dataset") == producer.get("dataset") and queue.get("seed") == producer.get("seed")
            and queue.get("head") == producer.get("head") and queue.get("status") == producer.get("status"),
            {"queue": queue, "producer": {key: producer.get(key) for key in ("dataset", "seed", "head", "status")}},
        )
    return producer


def load_bridge_splits(bridge_result: Mapping[str, Any], dataset: str, checks, artifacts: list[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    splits = {}
    for record in bridge_result["artifacts"]:
        if record.get("dataset") == dataset:
            splits[record["split"]] = checked_npz(record, f"bridge:{dataset}:{record['split']}", checks, artifacts)
    checks(f"bridge:{dataset}:all_splits", set(splits) == set(splits_for(dataset)), {"splits": sorted(splits)})
    return splits


def audit(sequence_results: list[Path], out_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "RUNNING",
        "schema": "smallk_forecast_sequence_independent_audit_v1",
        "checks": 0,
        "failed_count": 0,
        "failures": [],
        "conditions": [],
        "artifacts": [],
        "F_forward_replayed": False,
        "old_engine_or_F_constructed": False,
        "GPU_initialized": False,
        "protected_TEST_accessed": False,
        "past_latent_forecast_replayed": False,
        "residual_reconstruction_scope": "CPU reconstruction from stored latents, raw_last, codec decode state, and head state only",
        "runner_source_sha256_at_source_review": sha(RUNNER_SOURCE) if RUNNER_SOURCE.exists() else None,
        "decoder_source_sha256": sha(DECODER_SOURCE) if DECODER_SOURCE.exists() else None,
    }
    limit = None

    def check(name: str, passed: bool, details: Any = None) -> None:
        result["checks"] += 1
        if not bool(passed):
            result["failed_count"] += 1
            result["failures"].append({"check": name, "details": jsonable(details)})

    try:
        if out_path.exists():
            raise FileExistsError(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result["audit_source_snapshot"] = snapshot_audit_sources(out_path, check)
        check("runner_source_sha", sha(RUNNER_SOURCE) == EXPECTED_RUNNER_SHA if RUNNER_SOURCE.exists() else False)
        check("decoder_source_sha", sha(DECODER_SOURCE) == EXPECTED_DECODER_SHA if DECODER_SOURCE.exists() else False)
        limit = make_limit("forecast_sequence_independent_audit_v1", gpu=False)
        import torch

        torch.set_num_threads(2)
        refs = producer_refs_from_inputs(sequence_results, check)
        producers = [load_producer(ref, index, check) for index, ref in enumerate(refs)]
        expected = {(dataset, seed, head) for dataset in DATASETS for seed in SEEDS for head in HEADS}
        identities = {(row.get("dataset"), row.get("seed"), row.get("head")) for row in producers}
        check("all8_conditions_present", identities == expected, {"actual": sorted(map(str, identities))})
        grouped = {(dataset, seed): [] for dataset in DATASETS for seed in SEEDS}
        bridge_cache: dict[str, tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]] = {}
        for producer in producers:
            dataset, seed, head = producer["dataset"], int(producer["seed"]), producer["head"]
            tag = producer.get("tag", f"sequence_{dataset}_{HEAD_TAG.get(head, head).lower()}_s{seed}_v1")
            condition_summary = {
                "dataset": dataset,
                "seed": seed,
                "head": head,
                "producer_sha256": producer["_producer_result"]["sha256"],
                "producer_path": rel(resolve(producer["_producer_result"]["path"])),
                "tag": tag,
            }
            grouped[(dataset, seed)].append(producer)
            check(f"{tag}:status", producer.get("status") == "COMPLETE", {"status": producer.get("status"), "failure": producer.get("failure")})
            check(f"{tag}:identity", dataset in DATASETS and seed in SEEDS and head in HEADS and producer.get("objective") == "Y")
            if producer.get("status") != "COMPLETE":
                condition_summary.update(status=producer.get("status"), failure=producer.get("failure"))
                result["conditions"].append(condition_summary)
                continue
            check(f"{tag}:recipe", producer.get("recipe", {}).get("new_updates") == 600 and producer.get("recipe", {}).get("batch_size") == 4
                  and tuple(producer.get("recipe", {}).get("checkpoints", ())) == CHECKPOINTS)
            bridge_record = producer["bridge_provenance"]["input_bridge"]
            bridge_path = resolve(bridge_record["path"])
            bridge_digest = sha(bridge_path)
            check(f"{tag}:bridge_sha", bridge_digest == bridge_record["sha256"], bridge_record)
            if dataset not in bridge_cache:
                bridge_result = read_json_path(bridge_path)
                bridge_cache[dataset] = (bridge_result, load_bridge_splits(bridge_result, dataset, check, result["artifacts"]))
            bridge_result, splits = bridge_cache[dataset]
            std = splits["fit"]["std"]
            raw_source = checked_npz(producer["original_RAW_prediction_source"], f"{tag}:original_RAW", check, result["artifacts"])
            main_source = checked_npz(producer["original_MAIN_prediction_source"], f"{tag}:original_MAIN", check, result["artifacts"])
            for split, values in splits.items():
                for suffix, field in (("target", "target"), ("observed", "observed"), ("origins", "origins")):
                    require_exact(f"{tag}:RAW:{split}:{suffix}", raw_source[f"{split}_{suffix}"], values[field], check)
                    require_exact(f"{tag}:MAIN:{split}:{suffix}", main_source[f"{split}_{suffix}"], values[field], check)
                require_exact(f"{tag}:RAW:{split}:forecast", raw_source[f"{split}_prediction"], values["raw_forecast"], check)
                compare_score_record(f"{tag}:original_MAIN:{split}:scores", score_all(main_source[f"{split}_prediction"], values, std), producer["original_MAIN_scores"][split], check)
                compare_score_record(f"{tag}:original_RAW:{split}:scores", score_all(values["raw_forecast"], values, std), producer["original_RAW_scores"][split], check)
            original_payload = checked_checkpoint(producer["original_MAIN_checkpoint"]["checkpoint"], f"{tag}:original_MAIN_CP", check, result["artifacts"], torch)
            initial_codec = original_payload["codec"]
            check(f"{tag}:initial_codec_old_hash", old_state_hash(initial_codec) == producer["initial_state_hashes"]["state_sha_old_algorithm"])
            check(f"{tag}:initial_codec_Bi_hash", bi_state_hash(initial_codec) == producer["initial_state_hashes"]["state_sha_Bi_algorithm"])
            warm = checked_npz(producer["warmstart_all_splits"], f"{tag}:warmstart_all_splits", check, result["artifacts"])
            for split, values in splits.items():
                for name in ("prediction", "base_MAIN_style_prediction", "target", "observed", "origins", "original_MAIN"):
                    key = f"{split}_{name}"
                    if name == "prediction":
                        require_close(f"{tag}:warmstart:{split}:prediction_MAIN_parity", warm[key], main_source[f"{split}_prediction"], check)
                    elif name == "base_MAIN_style_prediction":
                        require_close(f"{tag}:warmstart:{split}:base_MAIN_parity", warm[key], main_source[f"{split}_prediction"], check)
                    elif name == "original_MAIN":
                        require_exact(f"{tag}:warmstart:{split}:original_MAIN", warm[key], main_source[f"{split}_prediction"], check)
                    else:
                        require_exact(f"{tag}:warmstart:{split}:{name}", warm[key], values[name], check)
                require_exact(f"{tag}:warmstart:{split}:mean", warm["mean"], values["mean"], check)
                require_exact(f"{tag}:warmstart:{split}:std", warm["std"], values["std"], check)
                checks_zero = int(np.count_nonzero(warm[f"{split}_head_residual"])) == 0
                check(f"{tag}:warmstart:{split}:zero_head_residual", checks_zero)
            parity = producer.get("initial_all_split_original_MAIN_parity", {})
            check(f"{tag}:declared_initial_all_split_parity", all(parity.get(split, {}).get("pass") is True for split in splits_for(dataset)), parity)
            check(f"{tag}:declared_zero_head", producer.get("initial_head_residual_exact_zero") is True)
            order_npz = checked_npz(producer["training_order"], f"{tag}:training_order", check, result["artifacts"])
            expected_order = np.random.default_rng(seed).integers(0, len(splits["fit"]["history"]), size=(600, 4))
            require_exact(f"{tag}:training_order", order_npz["indices"], expected_order, check)
            checkpoints = producer.get("checkpoints", [])
            check(f"{tag}:five_checkpoints", [row.get("new_update") for row in checkpoints] == list(CHECKPOINTS))
            checkpoint_rows = []
            tune_values = []
            states_by_update = {}
            for row in checkpoints:
                update = int(row["new_update"])
                payload = checked_checkpoint(row["checkpoint"], f"{tag}:checkpoint_u{update}", check, result["artifacts"], torch)
                check(f"{tag}:checkpoint_u{update}:identity", payload.get("dataset") == dataset and payload.get("seed") == seed
                      and payload.get("head_type") == head and payload.get("objective") == "Y" and payload.get("new_update") == update)
                if update == 0:
                    check(f"{tag}:checkpoint_u0_codec_old_hash", old_state_hash(payload["codec"]) == producer["initial_state_hashes"]["state_sha_old_algorithm"])
                    check(f"{tag}:checkpoint_u0_head_zero_hash", old_state_hash(payload["head"]) == producer["initial_state_hashes"]["head_state_sha_old_algorithm"])
                predictions = checked_npz(row["FIT_TUNE_predictions"], f"{tag}:FIT_TUNE_u{update}", check, result["artifacts"])
                scores_by_split = {}
                for split in FIT_TUNE_SPLITS:
                    values = splits[split]
                    for suffix, field in (("target", "target"), ("observed", "observed"), ("origins", "origins")):
                        require_exact(f"{tag}:u{update}:{split}:{suffix}", predictions[f"{split}_{suffix}"], values[field], check)
                    require_exact(f"{tag}:u{update}:{split}:mean", predictions["mean"], values["mean"], check)
                    require_exact(f"{tag}:u{update}:{split}:std", predictions["std"], values["std"], check)
                    if update == 0:
                        require_close(f"{tag}:u0:{split}:MAIN_parity", predictions[f"{split}_prediction"], main_source[f"{split}_prediction"], check)
                    trace = prefixed_trace(predictions, split)
                    rec_base, rec_residual, rec_prediction = reconstruct_saved_trace(torch, payload["codec"], payload["head"], head, trace)
                    require_close(f"{tag}:u{update}:{split}:base_reconstruct", rec_base, trace["base_MAIN_style_prediction"], check)
                    require_close(f"{tag}:u{update}:{split}:residual_reconstruct", rec_residual, trace["head_residual"], check)
                    require_close(f"{tag}:u{update}:{split}:prediction_reconstruct", rec_prediction, trace["prediction"], check)
                    scores = score_all(predictions[f"{split}_prediction"], values, std)
                    compare_score_record(f"{tag}:u{update}:{split}:scores", scores, row["scores"][split], check)
                    scores_by_split[split] = scores
                tune_mse = scores_by_split["tune"]["normalized_actual_Y"]["mse"]
                check(f"{tag}:u{update}:tune_mse", np.isclose(tune_mse, row["tune_actual_Y_mse"], atol=1e-11, rtol=1e-11))
                tune_values.append(tune_mse)
                states_by_update[update] = payload
                checkpoint_rows.append({"new_update": update, "scores": scores_by_split, "state_hashes": {key: row["checkpoint"].get(key) for key in ("state_sha_old_algorithm", "head_state_sha_old_algorithm", "combined_state_sha_Bi_algorithm")}})
                limit.check()
            selected_index = min(range(len(tune_values)), key=lambda index: tune_values[index]) if tune_values else None
            selected_update = CHECKPOINTS[selected_index] if selected_index is not None else None
            check(f"{tag}:earliest_tune_mse_selection", producer.get("selected_index") == selected_index and producer.get("selected_new_update") == selected_update)
            final_state = states_by_update[600]
            selected_state = states_by_update[selected_update]
            for label, declared, payload in (("final_u600", producer["final_u600_state_hashes"], final_state), ("selected", producer["selected_state_hashes"], selected_state)):
                check(f"{tag}:{label}:codec_old", old_state_hash(payload["codec"]) == declared["state_sha_old_algorithm"])
                check(f"{tag}:{label}:codec_Bi", bi_state_hash(payload["codec"]) == declared["state_sha_Bi_algorithm"])
                check(f"{tag}:{label}:head_old", old_state_hash(payload["head"]) == declared["head_state_sha_old_algorithm"])
                check(f"{tag}:{label}:head_Bi", bi_state_hash(payload["head"]) == declared["head_state_sha_Bi_algorithm"])
                check(f"{tag}:{label}:combined_Bi", combined_bi_hash(payload["codec"], payload["head"]) == declared["combined_state_sha_Bi_algorithm"])
            deltas = parameter_delta_groups(initial_codec, final_state["codec"], states_by_update[0]["head"], final_state["head"])
            for group in ("E", "D", "Head"):
                declared = producer["new_update_parameter_proof"][group]
                check(f"{tag}:delta:{group}:changed_count", deltas[group]["changed"] == declared["changed_parameter_tensors"] and deltas[group]["changed"] > 0)
            gradient_records = producer.get("gradient_diagnostics", [])
            check(f"{tag}:five_gradient_records", len(gradient_records) == 5, {"count": len(gradient_records)})
            for row in gradient_records:
                check(f"{tag}:gradient_record:{row.get('new_update')}", row.get("frozen_F_input_gradient_finite") is True
                      and row.get("frozen_F_input_gradient_norm", 0) > 0
                      and all(group.get("finite") for group in row.get("E_D_gradients", {}).values())
                      and row.get("Head_gradients", {}).get("finite") is True)
            curve = producer.get("curve", [])
            check(f"{tag}:curve600", len(curve) == 600 and [row.get("new_update") for row in curve] == list(range(1, 601)))
            for row in curve:
                check(f"{tag}:curve_finite:{row.get('new_update')}", all(np.isfinite(float(row[key])) for key in ("objective_loss", "actual_Y_loss", "gradient_norm_before_clip")))
                check(f"{tag}:curve_objective_is_actualY_MSE:{row.get('new_update')}", np.isclose(row["objective_loss"], row["actual_Y_loss"], atol=2e-6, rtol=2e-6))
            selected_scores = {}
            for split, record in producer.get("selected_inference", {}).items():
                arrays = checked_npz(record["artifact"], f"{tag}:selected:{split}", check, result["artifacts"])
                values = splits[split]
                for name in ("target", "observed", "origins", "mean", "std"):
                    require_exact(f"{tag}:selected:{split}:{name}", arrays[name], values[name], check)
                require_exact(f"{tag}:selected:{split}:RAW", arrays["original_RAW_forecast"], values["raw_forecast"], check)
                require_exact(f"{tag}:selected:{split}:MAIN", arrays["original_MAIN_forecast"], main_source[f"{split}_prediction"], check)
                rec_base, rec_residual, rec_prediction = reconstruct_saved_trace(torch, selected_state["codec"], selected_state["head"], head, arrays)
                require_close(f"{tag}:selected:{split}:base_reconstruct", rec_base, arrays["base_MAIN_style_prediction"], check)
                require_close(f"{tag}:selected:{split}:residual_reconstruct", rec_residual, arrays["head_residual"], check)
                require_close(f"{tag}:selected:{split}:prediction_reconstruct", rec_prediction, arrays["prediction"], check)
                scores = score_all(arrays["prediction"], values, std)
                compare_score_record(f"{tag}:selected:{split}:scores", scores, record["scores"], check)
                selected_scores[split] = scores
            check(f"{tag}:selected_all3", set(producer.get("selected_inference", {})) == set(splits_for(dataset)))
            condition_summary.update(selected_new_update=selected_update, checkpoint_scores=checkpoint_rows, selected_scores=selected_scores, final_delta_groups=deltas)
            result["conditions"].append(condition_summary)
        for key, group in grouped.items():
            complete = [row for row in group if row.get("status") == "COMPLETE"]
            check(f"{key}:same_cp_order_across_heads", len(complete) == 2 and {row.get("head") for row in complete} == set(HEADS)
                  and len({row["original_MAIN_checkpoint"]["checkpoint"]["sha256"] for row in complete}) == 1
                  and len({row["training_order_sha256"] for row in complete}) == 1)
        check("coverage_8_conditions_40_checkpoints_24_selected", len(result["conditions"]) == 8)
        result["GPU_initialized"] = bool(torch.cuda.is_initialized())
        check("no_CUDA_initialized", not result["GPU_initialized"])
        result["status"] = "PASS_WITHOUT_F_OR_PAST_LATENT_FORWARD_REPLAY" if result["failed_count"] == 0 else "FAIL_PRESERVED"
    except Exception as error:
        result.update(status="FAILED_PRESERVED", failure={"type": type(error).__name__, "message": str(error)})
    finally:
        result["wall_seconds_before_terminal"] = time.perf_counter() - started
        if limit is not None:
            try:
                receipt = limit.close("COMPLETE" if result["status"].startswith("PASS") else "FAILED_PRESERVED", result.get("failure"))
                result["shared_resource_receipt"] = receipt
                if receipt.get("terminal_status") != "COMPLETE":
                    result.update(status=receipt.get("terminal_status", result["status"]), terminal_error=receipt.get("terminal_error"))
            except Exception as close_error:
                result.update(status="FAILED_PRESERVED", resource_close_unavailable={
                    "type": type(close_error).__name__, "message": str(close_error)})
        result["audit_source_sha256"] = sha(__file__)
        result["sequence_results"] = [{"path": rel(resolve(path)), "sha256": sha(resolve(path)) if resolve(path).exists() else None} for path in sequence_results]
        with out_path.open("x", encoding="utf-8") as handle:
            json.dump(jsonable(result), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-result", type=Path, nargs="+", default=[DEFAULT_QUEUE], help="Queue RESULT or eight single-fit RESULT paths.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--root-go", action="store_true")
    if not parser.parse_known_args()[0].root_go:
        args = parser.parse_args()
        print(json.dumps({"status": "NOT_STARTED_ROOT_GO_REQUIRED", "default_queue": rel(args.sequence_result[0]), "default_out": rel(args.out)}, ensure_ascii=False, indent=2))
        return 2
    args = parser.parse_args()
    result = audit(args.sequence_result, args.out)
    print(json.dumps({"status": result["status"], "checks": result["checks"], "failed_count": result["failed_count"], "out": rel(args.out)}, ensure_ascii=False))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
