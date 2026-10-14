"""Independent CPU audit for Appliances loss warm-start runs.

This file is intentionally source-only until root passes ``--root-go`` after all
six Appliances loss producers are terminal.  The audit reads only already
materialized bridge/producer artifacts, never starts GPU training, never calls
``open_evaluation()``, and never replays F/per-update forward passes.
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

from followup_runtime import FollowupLimit
from research.smallk_failure_followup_20261014 import prepare_appliances_inputs as bridge_api

DEFAULT_BRIDGE_RESULT = ROOT / ".cache" / STUDY.name / "appliances_input_bridge" / "phase_v2" / "RESULT.json"
DEFAULT_LOSS_RESULT = ROOT / ".cache" / STUDY.name / "appliances_losses" / "APPLIANCES_QUEUE_v1.json"
DEFAULT_OUT = HERE / "APPLIANCES_LOSS_AUDIT_v2.json"

DATASET = "appliances"
SEEDS = (92601, 92602)
LOSSES = ("MSE", "MAE", "HUBER_DELTA1")
SPLITS = ("fit", "tune", "eval")
CHECKPOINTS = (0, 100, 200, 400, 600)
EXPECTED_COUNTS = {"fit": 256, "tune": 64, "eval": 82}
L = 512
H = 48
C = 26
K = 6
WARMSTART_ATOL = 1e-5
WARMSTART_RTOL = 2e-6

EXPECTED_PRODUCER_SCHEMA = {
    "top_level": {
        "status": "COMPLETE",
        "schema": "smallk_appliances_loss_queue_v1 or six individual single-fit RESULT files",
        "conditions": "queue conditions[*].producer_result {path, sha256}; individual RESULT paths may also be passed with --loss-result",
    },
    "condition": {
        "dataset": DATASET,
        "seed": "92601 or 92602",
        "objective": "MSE, MAE, or HUBER_DELTA1",
        "bridge_provenance.input_bridge": "{path, sha256}",
        "original_MAIN_checkpoint.checkpoint": "old selected LEARNED_REGIME checkpoint record from bridge checkpoint_cases[seed].source_checkpoint",
        "original_MAIN_checkpoint.q_state_hash_APP_shape_dtype_algorithm": "old q state hash; preserved but unused",
        "initial_codec_hash": "state_hash(codec) of original_checkpoint loaded on CPU",
        "q_aux_new_training_and_deployment": "false/false, with original_MAIN_checkpoint q flags false",
        "training_order": "NPZ record containing indices int array of shape [600, 4]",
        "recipe": "new_updates=600, batch_size=4, checkpoints=[0,100,200,400,600], selection_primary_metric TUNE MSE",
        "warmstart_all_splits.artifact": "single NPZ with <split>_prediction,target,observed,origins,original_LEARNED and mean/std",
        "checkpoints": "five records with new_update, checkpoint, FIT_TUNE_predictions, scores, tune_actual_Y_mse",
        "curve": "600 recorded rows; finite recorded loss arithmetic only",
        "gradient_diagnostics": "records proving finite E/D gradients and frozen-F input gradient diagnostics",
        "selected_inference": "dict fit/tune/eval -> artifact and scores",
    },
}


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


def contains_test_path(path: str | Path) -> bool:
    value = Path(path)
    return any(part.lower() == "test" for part in value.parts) or value.name.lower() in {"test.npz", "jena_test.npz"}


def guard_no_test(path: str | Path, role: str) -> None:
    if contains_test_path(path):
        raise PermissionError(f"PROTECTED_TEST_PATH_DENIED:{role}:{path}")


def read_json_path(path: str | Path) -> dict[str, Any]:
    path = resolve(path)
    guard_no_test(path, "json")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_json_record(record: Mapping[str, Any], role: str, checks) -> dict[str, Any]:
    path = resolve(record["path"])
    guard_no_test(path, role)
    actual = sha(path)
    checks(f"{role}:sha", actual == record.get("sha256"), {"path": rel(path), "actual": actual, "expected": record.get("sha256")})
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checked_npz(record: Mapping[str, Any], role: str, checks, artifacts: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    path = resolve(record["path"])
    guard_no_test(path, role)
    before = sha(path)
    checks(f"{role}:file_sha", before == record.get("sha256"), {"path": rel(path), "actual": before, "expected": record.get("sha256")})
    with np.load(path, allow_pickle=False) as opened:
        arrays = {name: opened[name].copy() for name in opened.files}
    if "arrays" in record:
        checks(f"{role}:array_keys", set(arrays) == set(record["arrays"]), {"path": rel(path), "actual": sorted(arrays), "expected": sorted(record["arrays"])})
        for name, value in arrays.items():
            meta = record["arrays"][name]
            checks(
                f"{role}:{name}:contract",
                list(value.shape) == meta["shape"] and str(value.dtype) == meta["dtype"] and array_sha(value) == meta["sha256"],
                {"path": rel(path), "array": name},
            )
    after = sha(path)
    checks(f"{role}:unchanged", after == before, {"path": rel(path), "before": before, "after": after})
    artifacts.append({"role": role, "path": rel(path), "sha256_before": before, "sha256_after": after})
    return arrays


def require_exact(name: str, actual: np.ndarray, expected: np.ndarray, checks) -> None:
    ok = actual.dtype == expected.dtype and actual.shape == expected.shape and np.array_equal(actual, expected, equal_nan=True)
    details = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape), "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}
    if not ok and actual.shape == expected.shape:
        diff = actual.astype(np.float64) - expected.astype(np.float64)
        details["max_abs"] = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
    checks(name, ok, details)


def require_close(name: str, actual: np.ndarray, expected: np.ndarray, checks, *, atol: float, rtol: float) -> None:
    ok = actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    details = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape), "atol": atol, "rtol": rtol}
    if actual.shape == expected.shape:
        details["max_abs"] = float(np.nanmax(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0.0
    checks(name, bool(ok), details)


def channel_macro_metric(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray, *, metric: str, std: np.ndarray | None = None) -> float:
    err = prediction.astype(np.float64) - target.astype(np.float64)
    if std is not None:
        err = err * std.reshape(1, 1, -1).astype(np.float64)
    if metric == "mse":
        values = err * err
    elif metric == "mae":
        values = np.abs(err)
    elif metric == "huber_delta1":
        abs_err = np.abs(err)
        values = np.where(abs_err <= 1.0, 0.5 * err * err, abs_err - 0.5)
    else:
        raise ValueError(metric)
    channel_values: list[float] = []
    for channel in range(values.shape[-1]):
        mask = observed[..., channel].astype(bool)
        if mask.any():
            channel_values.append(float(values[..., channel][mask].mean()))
    if not channel_values:
        return float("nan")
    return float(np.mean(channel_values))


def score_all(prediction: np.ndarray, target: np.ndarray, teacher: np.ndarray, observed: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    return {
        "normalized_actual_Y": {
            "mse": channel_macro_metric(prediction, target, observed, metric="mse"),
            "mae": channel_macro_metric(prediction, target, observed, metric="mae"),
            "huber_delta1": channel_macro_metric(prediction, target, observed, metric="huber_delta1"),
        },
        "physical_actual_Y": {
            "mse": channel_macro_metric(prediction, target, observed, metric="mse", std=std),
            "mae": channel_macro_metric(prediction, target, observed, metric="mae", std=std),
            "huber_delta1": channel_macro_metric(prediction, target, observed, metric="huber_delta1", std=std),
        },
        "normalized_RAW_teacher": {
            "mse": channel_macro_metric(prediction, teacher, observed, metric="mse"),
            "mae": channel_macro_metric(prediction, teacher, observed, metric="mae"),
            "huber_delta1": channel_macro_metric(prediction, teacher, observed, metric="huber_delta1"),
        },
    }


def compare_score_record(name: str, actual: Mapping[str, Any], declared: Mapping[str, Any], checks, *, atol: float = 1e-11, rtol: float = 1e-11) -> None:
    for group, values in actual.items():
        checks(f"{name}:{group}:present", group in declared, {"available": sorted(declared)})
        if group not in declared:
            continue
        for metric, value in values.items():
            expected = declared[group].get(metric)
            if expected is None:
                continue
            checks(f"{name}:{group}:{metric}", np.isclose(value, expected, atol=atol, rtol=rtol), {"actual": value, "expected": expected})


def state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if hasattr(value, "detach"):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def checked_checkpoint(record: Mapping[str, Any], role: str, checks, artifacts: list[dict[str, Any]], torch_module) -> dict[str, Any]:
    path = resolve(record["path"])
    guard_no_test(path, role)
    before = sha(path)
    checks(f"{role}:file_sha", before == record.get("sha256"), {"path": rel(path), "actual": before, "expected": record.get("sha256")})
    payload = torch_module.load(path, map_location="cpu", weights_only=True)
    after = sha(path)
    checks(f"{role}:unchanged", after == before, {"path": rel(path)})
    if "state_sha_APP_shape_dtype_algorithm" in record and "codec" in payload:
        checks(f"{role}:codec_state_sha_APP", state_hash(payload["codec"]) == record["state_sha_APP_shape_dtype_algorithm"], {"actual": state_hash(payload["codec"]), "expected": record["state_sha_APP_shape_dtype_algorithm"]})
    artifacts.append({"role": role, "path": rel(path), "sha256_before": before, "sha256_after": after})
    return payload


def producer_refs_from_inputs(loss_result_paths: list[Path], checks) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for index, path in enumerate(loss_result_paths):
        resolved = resolve(path)
        guard_no_test(resolved, f"loss_result_{index}")
        data = json.loads(resolved.read_text(encoding="utf-8-sig"))
        actual = sha(resolved)
        if data.get("schema") == "smallk_appliances_loss_queue_v1" or "conditions" in data and "producer_result" in data.get("conditions", [{}])[0]:
            checks(f"queue_{index}:status", data.get("status") == "COMPLETE", {"path": rel(resolved), "status": data.get("status")})
            for condition_index, item in enumerate(data.get("conditions", [])):
                ref = dict(item["producer_result"])
                ref["_queue_condition"] = {key: item.get(key) for key in ("tag", "dataset", "seed", "objective", "status", "selected_new_update")}
                ref["_queue_path"] = rel(resolved)
                ref["_queue_sha256"] = actual
                ref["_queue_index"] = condition_index
                refs.append(ref)
        else:
            refs.append({"path": str(resolved), "sha256": actual})
    checks("producer_ref_count", len(refs) == 6, {"count": len(refs), "paths": [ref.get("path") for ref in refs]})
    return refs


def load_condition_records(producer_refs: list[dict[str, Any]], checks) -> list[dict[str, Any]]:
    conditions = []
    for index, ref in enumerate(producer_refs):
        condition = read_json_record(ref, f"condition_{index}:producer_result", checks)
        condition["_producer_result"] = {"path": ref["path"], "sha256": ref["sha256"]}
        if "_queue_condition" in ref:
            condition["_queue_condition"] = ref["_queue_condition"]
            queue = ref["_queue_condition"]
            checks(
                f"condition_{index}:queue_identity_matches",
                queue.get("dataset") == condition.get("dataset")
                and queue.get("seed") == condition.get("seed")
                and queue.get("objective") == condition.get("objective")
                and queue.get("status") == condition.get("status"),
                {"queue": queue, "producer": {key: condition.get(key) for key in ("dataset", "seed", "objective", "status")}},
            )
        conditions.append(condition)
    return conditions


def artifact_from_split_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if "artifact" in record:
        return record["artifact"]
    return record


def metric_for_loss(scores: Mapping[str, Any], loss_name: str) -> float:
    normalized = scores["normalized_actual_Y"]
    return float(normalized["huber_delta1" if loss_name == "HUBER_DELTA1" else loss_name.lower()])


def expected_order(seed: int, count: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, count, size=(600, 4))


def audit(loss_result_paths: list[Path], bridge_result_path: Path, out_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "RUNNING",
        "schema": "smallk_appliances_loss_independent_audit_v2",
        "checks": 0,
        "failed_count": 0,
        "failures": [],
        "artifacts": [],
        "conditions": [],
        "producer_schema_contract": EXPECTED_PRODUCER_SCHEMA,
        "F_forward_replayed": False,
        "per_update_forward_replayed": False,
        "GPU_initialized": False,
        "protected_TEST_accessed": False,
        "old_campaign_resumed": False,
        "open_evaluation_called": False,
        "curve_claim_scope": "600 recorded updates/finite values/recorded loss arithmetic only; per-update forward replay is not claimed",
        "q_auxiliary_scope": "old q state is hash-verified and preserved as unused provenance; it is not used for new forward/deployment proof",
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
        for loss_result_path in loss_result_paths:
            guard_no_test(loss_result_path, "loss_result")
        guard_no_test(bridge_result_path, "bridge_result")
        limit = FollowupLimit("appliances_loss_independent_audit_v2", gpu=False)
        import torch

        torch.set_num_threads(2)
        bridge_result_path = bridge_result_path.resolve()
        bridge_before = sha(bridge_result_path)
        bridge_bundle = bridge_api.load_prepared_bundle(bridge_result_path)
        bridge_manifest = bridge_bundle["manifest"]
        check("bridge_status", bridge_manifest.get("status") == "COMPLETE", {"status": bridge_manifest.get("status")})
        check("bridge_sha_stable", sha(bridge_result_path) == bridge_before, {"path": rel(bridge_result_path), "sha": bridge_before})
        check("bridge_dataset_C_K", bridge_manifest.get("dataset") == DATASET and bridge_manifest.get("C") == C and bridge_manifest.get("K") == K)
        for split in SPLITS:
            values = bridge_bundle["splits"][split]
            check(f"bridge:{split}:count", int(values["history"].shape[0]) == EXPECTED_COUNTS[split], {"shape": list(values["history"].shape)})
            check(f"bridge:{split}:history_shape", list(values["history"].shape[1:]) == [L, C])
            check(f"bridge:{split}:target_shape", list(values["target"].shape[1:]) == [H, C])
            rebuilt = bridge_api.old_window_preprocess(values["history_physical"], values["history_observed"], values["mean"], values["std"])
            require_exact(f"bridge:{split}:physical_history_reconstructs_normalized_history", rebuilt, values["history"], check)
            require_exact(f"bridge:{split}:raw_last", values["history"][:, -1].astype(np.float32), values["raw_last"], check)
        raw_reference_records = bridge_manifest["raw_references"]
        reference_npz = {
            "RAW_BOLT": checked_npz(raw_reference_records["RAW_BOLT"], "original_RAW_BOLT", check, result["artifacts"]),
            "RAW_BOLT_RAW_R": checked_npz(raw_reference_records["RAW_BOLT_RAW_R"], "original_RAW_BOLT_RAW_R", check, result["artifacts"]),
        }
        for seed, record in raw_reference_records["LEARNED_BOLT"].items():
            reference_npz[f"LEARNED_{seed}"] = checked_npz(record, f"original_LEARNED_BOLT_{seed}", check, result["artifacts"])
        for split in SPLITS:
            values = bridge_bundle["splits"][split]
            for reference_name, arrays in reference_npz.items():
                require_exact(f"{reference_name}:{split}:target", arrays[f"{split}_target"], values["target"], check)
                require_exact(f"{reference_name}:{split}:observed", arrays[f"{split}_observed"], values["observed"], check)
                require_exact(f"{reference_name}:{split}:origins", arrays[f"{split}_origins"], values["origins"], check)
                require_exact(f"{reference_name}:{split}:mean", arrays["mean"], values["mean"], check)
                require_exact(f"{reference_name}:{split}:std", arrays["std"], values["std"], check)
            require_exact(f"RAW_BOLT:{split}:bridge_forecast", reference_npz["RAW_BOLT"][f"{split}_prediction"], values["raw_forecast"], check)
            require_exact(f"RAW_BOLT_RAW_R:{split}:bridge_forecast", reference_npz["RAW_BOLT_RAW_R"][f"{split}_prediction"], values["raw_plus_r_forecast"], check)
            for seed in map(str, SEEDS):
                require_exact(f"LEARNED_{seed}:{split}:bridge_forecast", reference_npz[f"LEARNED_{seed}"][f"{split}_prediction"], values[f"learned_{seed}_forecast"], check)
        cp_states: dict[str, dict[str, Any]] = {}
        for seed, case in bridge_manifest["checkpoint_cases"].items():
            cp_record = case["source_checkpoint"]
            payload = checked_checkpoint(cp_record, f"old_selected_checkpoint_{seed}", check, result["artifacts"], torch)
            check(f"old_cp_{seed}:payload_metadata", payload["seed"] == int(seed) and payload["update"] == case["selected_update"] and payload["method"] == "LEARNED_REGIME" and float(payload["weight"]) == 0.1)
            check(f"old_cp_{seed}:codec_hash", state_hash(payload["codec"]) == case["checkpoint_cpu_inspection"]["codec_state_sha256"])
            if payload.get("q") is not None:
                check(f"old_cp_{seed}:q_hash", state_hash(payload["q"]) == case["checkpoint_cpu_inspection"]["q_state_sha256"])
            check(f"old_cp_{seed}:q_preserved_unused", case["checkpoint_cpu_inspection"]["q_aux_deployment_use"] == "preserved_not_used")
            cp_states[seed] = {"payload": payload, "bridge_case": case}
        producer_refs = producer_refs_from_inputs(loss_result_paths, check)
        conditions = load_condition_records(producer_refs, check)
        identities = {(row.get("seed"), row.get("objective")) for row in conditions}
        check("six_conditions_complete", identities == {(seed, objective) for seed in SEEDS for objective in LOSSES}, {"actual": sorted(map(str, identities))})
        group_by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in SEEDS}
        raw_plus_r_scores = {}
        for split in SPLITS:
            values = bridge_bundle["splits"][split]
            raw_plus_r_scores[split] = score_all(values["raw_plus_r_forecast"], values["target"], values["raw_forecast"], values["observed"], values["std"])
        result["RAW_plus_R_shared_baseline_scores"] = raw_plus_r_scores
        result["RAW_baseline_scores"] = {
            split: score_all(values["raw_forecast"], values["target"], values["raw_forecast"], values["observed"], values["std"])
            for split, values in bridge_bundle["splits"].items()
        }
        for condition in conditions:
            seed = int(condition.get("seed"))
            objective = str(condition.get("objective"))
            tag = condition.get("tag", f"seed{seed}_{objective}")
            producer_sha256 = condition["_producer_result"]["sha256"]
            check(f"{tag}:identity", condition.get("dataset") == DATASET and seed in SEEDS and objective in LOSSES, {"condition": condition.get("tag")})
            producer_path_record = condition.get("_producer_result")
            condition_summary = {"tag": tag, "dataset": DATASET, "seed": seed, "objective": objective, "producer_sha256": producer_sha256, "producer_result": producer_path_record}
            group_by_seed[seed].append(condition)
            producer_bridge = condition.get("bridge_provenance", {}).get("input_bridge", {"path": rel(bridge_result_path), "sha256": bridge_before})
            check(f"{tag}:bridge_binding", sha(resolve(producer_bridge["path"])) == producer_bridge["sha256"] == bridge_before, producer_bridge)
            recipe = condition.get("recipe", {})
            check(
                f"{tag}:recipe_lock",
                recipe.get("new_updates") == 600
                and recipe.get("batch_size") == 4
                and tuple(recipe.get("checkpoints", ())) == CHECKPOINTS
                and recipe.get("checkpoint_selection") == "TUNE actual-Y normalized MSE; earliest exact tie"
                and recipe.get("RAW_forecast_reference_used_for_training") is False
                and recipe.get("future_alignment_q_auxiliary") == "disabled uniformly for all candidates",
                recipe,
            )
            q_scope = condition.get("q_aux_new_training_and_deployment", {})
            q_scope_text = q_scope if isinstance(q_scope, str) else ""
            original_main = condition.get("original_MAIN_checkpoint", {})
            check(
                f"{tag}:q_unused",
                (
                    ("disabled" in q_scope_text and "preserved" in q_scope_text)
                    or (
                        isinstance(q_scope, dict)
                        and q_scope.get("q_aux_used_in_new_training") is False
                        and q_scope.get("q_aux_used_in_new_deployment") is False
                    )
                )
                and original_main.get("q_aux_used_in_new_training") is False
                and original_main.get("q_aux_used_in_new_deployment") is False,
                {"q_scope": q_scope, "original_MAIN_checkpoint": original_main},
            )
            original_record = original_main.get("checkpoint")
            check(f"{tag}:original_MAIN_checkpoint_present", isinstance(original_record, dict), original_main)
            original_payload = checked_checkpoint(original_record, f"{tag}:original_MAIN_checkpoint", check, result["artifacts"], torch) if isinstance(original_record, dict) else cp_states[str(seed)]["payload"]
            initial_hash = state_hash(original_payload["codec"])
            check(f"{tag}:initial_cp_matches_bridge", original_record and original_record.get("sha256") == cp_states[str(seed)]["bridge_case"]["source_checkpoint"]["sha256"])
            check(
                f"{tag}:initial_codec_hash_binding",
                condition.get("initial_codec_hash") == original_main.get("selected_codec_hash_old_algorithm") == cp_states[str(seed)]["bridge_case"]["selected_codec_hash"],
                {
                    "producer_initial_codec_hash": condition.get("initial_codec_hash"),
                    "producer_original_MAIN_old_hash": original_main.get("selected_codec_hash_old_algorithm"),
                    "bridge_selected_codec_hash": cp_states[str(seed)]["bridge_case"]["selected_codec_hash"],
                    "audit_shape_dtype_hash": initial_hash,
                },
            )
            if original_payload.get("q") is not None:
                check(f"{tag}:initial_q_hash", original_main.get("q_state_hash_APP_shape_dtype_algorithm") == state_hash(original_payload["q"]), {"actual": state_hash(original_payload["q"]), "declared": original_main.get("q_state_hash_APP_shape_dtype_algorithm")})
            order_npz = checked_npz(condition["training_order"], f"{tag}:training_order", check, result["artifacts"])
            order = order_npz["indices"]
            require_exact(f"{tag}:same_seed_training_order", order, expected_order(seed, EXPECTED_COUNTS["fit"]), check)
            warmstart_arrays = checked_npz(condition["warmstart_all_splits"], f"{tag}:warmstart_all_splits", check, result["artifacts"])
            parity_record = condition.get("initial_all_split_original_LEARNED_parity", {})
            for split in SPLITS:
                values = bridge_bundle["splits"][split]
                for name in ("target", "observed", "origins"):
                    require_exact(f"{tag}:warmstart:{split}:{name}", warmstart_arrays[f"{split}_{name}"], values[name], check)
                require_exact(f"{tag}:warmstart:{split}:mean", warmstart_arrays["mean"], values["mean"], check)
                require_exact(f"{tag}:warmstart:{split}:std", warmstart_arrays["std"], values["std"], check)
                require_close(f"{tag}:warmstart:{split}:prediction_vs_original_LEARNED_field", warmstart_arrays[f"{split}_prediction"], warmstart_arrays[f"{split}_original_LEARNED"], check, atol=WARMSTART_ATOL, rtol=WARMSTART_RTOL)
                require_close(f"{tag}:warmstart:{split}:old_LEARNED_parity", warmstart_arrays[f"{split}_prediction"], values[f"learned_{seed}_forecast"], check, atol=WARMSTART_ATOL, rtol=WARMSTART_RTOL)
                check(f"{tag}:warmstart:{split}:declared_parity", parity_record.get(split, {}).get("pass") is True, parity_record.get(split))
            tune_selection_values: list[float] = []
            checkpoint_summaries = []
            checkpoints = condition.get("checkpoints", [])
            check(f"{tag}:five_checkpoints", [row.get("new_update") for row in checkpoints] == list(CHECKPOINTS), {"updates": [row.get("new_update") for row in checkpoints]})
            for row in checkpoints:
                update = int(row["new_update"])
                payload = checked_checkpoint(row["checkpoint"], f"{tag}:checkpoint_u{update}", check, result["artifacts"], torch)
                check(f"{tag}:checkpoint_u{update}:metadata", payload.get("new_update", payload.get("update")) == update and payload.get("seed") == seed and payload.get("objective", payload.get("loss")) == objective)
                if update == 0:
                    check(
                        f"{tag}:checkpoint_u0_equals_initial",
                        row["checkpoint"].get("state_sha_old_algorithm") == original_main.get("selected_codec_hash_old_algorithm")
                        and state_hash(payload["codec"]) == initial_hash,
                        {
                            "u0_old_algorithm": row["checkpoint"].get("state_sha_old_algorithm"),
                            "initial_old_algorithm": original_main.get("selected_codec_hash_old_algorithm"),
                            "audit_shape_dtype_hash": initial_hash,
                        },
                    )
                predictions = checked_npz(row["FIT_TUNE_predictions"], f"{tag}:FIT_TUNE_predictions_u{update}", check, result["artifacts"])
                split_scores = {}
                for split in ("fit", "tune"):
                    values = bridge_bundle["splits"][split]
                    for suffix, field in (("target", "target"), ("observed", "observed"), ("origins", "origins"), ("RAW_teacher", "raw_forecast")):
                        require_exact(f"{tag}:u{update}:{split}:{suffix}", predictions[f"{split}_{suffix}"], values[field], check)
                    require_exact(f"{tag}:u{update}:{split}:mean", predictions["mean"], values["mean"], check)
                    require_exact(f"{tag}:u{update}:{split}:std", predictions["std"], values["std"], check)
                    if update == 0:
                        require_close(f"{tag}:u0:{split}:old_LEARNED_parity", predictions[f"{split}_prediction"], values[f"learned_{seed}_forecast"], check, atol=WARMSTART_ATOL, rtol=WARMSTART_RTOL)
                    scores = score_all(predictions[f"{split}_prediction"], values["target"], values["raw_forecast"], values["observed"], values["std"])
                    compare_score_record(f"{tag}:u{update}:{split}:scores", scores, row["scores"][split], check)
                    split_scores[split] = scores
                tune_mse = split_scores["tune"]["normalized_actual_Y"]["mse"]
                check(f"{tag}:u{update}:selection_value", np.isclose(tune_mse, row["tune_actual_Y_mse"], atol=1e-11, rtol=1e-11), {"actual": tune_mse, "declared": row["tune_actual_Y_mse"]})
                tune_selection_values.append(tune_mse)
                checkpoint_summaries.append({"new_update": update, "scores": split_scores, "state_hash": state_hash(payload["codec"])})
            selected_index = min(range(len(tune_selection_values)), key=lambda idx: tune_selection_values[idx]) if tune_selection_values else None
            selected_update = CHECKPOINTS[selected_index] if selected_index is not None else None
            check(f"{tag}:TUNE_MSE_selection_including_u0", condition.get("selected_index") == selected_index and condition.get("selected_new_update") == selected_update)
            u600_record = next(row for row in checkpoints if int(row["new_update"]) == 600)
            final_payload = checked_checkpoint(u600_record["checkpoint"], f"{tag}:final_u600_checkpoint", check, result["artifacts"], torch)
            check(
                f"{tag}:final_u600_hash",
                u600_record["checkpoint"].get("state_sha_old_algorithm") == condition.get("final_u600_codec_hash"),
                {
                    "u600_old_algorithm": u600_record["checkpoint"].get("state_sha_old_algorithm"),
                    "declared_final_u600_codec_hash": condition.get("final_u600_codec_hash"),
                    "audit_shape_dtype_hash": state_hash(final_payload["codec"]),
                },
            )
            check(f"{tag}:F_frozen_record", condition.get("F_before") == condition.get("F_after") and condition.get("F_frozen_pass") and condition.get("F_has_no_parameter_gradients"))
            gradients = condition.get("gradient_diagnostics", [])
            check(f"{tag}:gradient_diagnostic_recorded", len(gradients) >= 5, {"count": len(gradients)})
            for diagnostic in gradients:
                check(f"{tag}:gradient_finite:{diagnostic.get('new_update')}", diagnostic.get("frozen_F_input_gradient_finite") is True and np.isfinite(diagnostic.get("frozen_F_input_gradient_norm", np.nan)))
            proof = condition.get("new_update_parameter_proof", {})
            for group in ("E", "D"):
                changed = proof.get(group, {}).get("changed_parameter_tensors", 0)
                check(f"{tag}:u600_{group}_changed", changed > 0, {"changed_parameter_tensors": changed})
            curve = condition.get("curve", [])
            check(f"{tag}:curve600coverage", len(curve) == 600 and [row.get("new_update") for row in curve] == list(range(1, 601)))
            for index, row in enumerate(curve):
                if index >= 600:
                    break
                recorded = [row.get(key) for key in ("objective_loss", "actual_Y_loss", "cached_RAW_teacher_loss", "gradient_norm_before_clip") if key in row]
                check(f"{tag}:curve_finite:{index+1}", all(np.isfinite(float(value)) for value in recorded), row)
                if objective == "MSE" and "objective_loss" in row and "actual_Y_loss" in row:
                    check(f"{tag}:curve_actualY_training_not_RAWtarget:{index+1}", np.isclose(row["objective_loss"], row["actual_Y_loss"], atol=2e-6, rtol=2e-6), row)
            selected_scores = {}
            selected_inference = condition.get("selected_inference", {})
            check(f"{tag}:selected_three_splits_present", set(selected_inference) == set(SPLITS), {"actual": sorted(selected_inference)})
            for split, record in selected_inference.items():
                arrays = checked_npz(artifact_from_split_record(record), f"{tag}:selected:{split}", check, result["artifacts"])
                values = bridge_bundle["splits"][split]
                for name in ("target", "observed", "origins", "mean", "std"):
                    require_exact(f"{tag}:selected:{split}:{name}", arrays[name], values[name], check)
                require_exact(f"{tag}:selected:{split}:RAW_teacher", arrays["original_RAW_forecast"], values["raw_forecast"], check)
                require_exact(f"{tag}:selected:{split}:original_LEARNED_reference", arrays["original_MAIN_forecast"], values[f"learned_{seed}_forecast"], check)
                scores = score_all(arrays["prediction"], values["target"], values["raw_forecast"], values["observed"], values["std"])
                compare_score_record(f"{tag}:selected:{split}:scores", scores, record["scores"], check)
                selected_scores[split] = scores
            for split, declared in condition.get("original_RAW_PLUS_R_scores", {}).items():
                compare_score_record(f"{tag}:original_RAW_PLUS_R:{split}:scores", raw_plus_r_scores[split], declared, check)
            condition_summary.update(
                selected_new_update=selected_update,
                checkpoint_scores=checkpoint_summaries,
                selected_scores=selected_scores,
                original_LEARNED_reference_scores={
                    split: score_all(bridge_bundle["splits"][split][f"learned_{seed}_forecast"], bridge_bundle["splits"][split]["target"], bridge_bundle["splits"][split]["raw_forecast"], bridge_bundle["splits"][split]["observed"], bridge_bundle["splits"][split]["std"])
                    for split in SPLITS
                },
            )
            result["conditions"].append(condition_summary)
            limit.check()
        for seed, group in group_by_seed.items():
            check(f"seed_{seed}:three_losses_same_initial_cp_order", len(group) == 3 and {row.get("objective") for row in group} == set(LOSSES) and len({row.get("initial_codec_hash") for row in group}) == 1)
            order_hashes = []
            for row in group:
                order_hashes.append(row.get("training_order_sha256") or row.get("training_order", {}).get("arrays", {}).get("indices", {}).get("sha256"))
            check(f"seed_{seed}:same_order_hash_all_losses", len(set(order_hashes)) == 1, {"order_hashes": order_hashes})
        check("coverage_6_conditions_30_checkpoints_18_selected", len(result["conditions"]) == 6)
        result["GPU_initialized"] = bool(torch.cuda.is_initialized())
        check("no_CUDA_initialized", not result["GPU_initialized"])
        result["status"] = "PASS_WITHOUT_GPU_F_OR_PER_UPDATE_FORWARD_REPLAY" if result["failed_count"] == 0 else "FAIL_PRESERVED"
    except Exception as error:
        result.update(status="FAILED_PRESERVED", failure={"type": type(error).__name__, "message": str(error)})
    finally:
        result["wall_seconds_before_terminal"] = time.perf_counter() - started
        if limit is not None:
            receipt = limit.close("COMPLETE" if result["status"].startswith("PASS") else "FAILED_PRESERVED", result.get("failure"))
            result["shared_resource_receipt"] = receipt
            if receipt.get("terminal_status", "COMPLETE") != "COMPLETE":
                result.update(status=receipt["terminal_status"], terminal_error=receipt.get("terminal_error"))
        result["audit_source_sha256"] = sha(__file__)
        result["loss_results"] = [{"path": rel(path), "sha256": sha(path) if resolve(path).exists() else None} for path in loss_result_paths]
        result["bridge_result"] = {"path": rel(bridge_result_path), "sha256": sha(bridge_result_path) if bridge_result_path.exists() else None}
        with out_path.open("x", encoding="utf-8") as handle:
            json.dump(jsonable(result), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-result", type=Path, nargs="+", default=[DEFAULT_LOSS_RESULT], help="Queue RESULT or six single-fit RESULT paths.")
    parser.add_argument("--bridge-result", type=Path, default=DEFAULT_BRIDGE_RESULT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--root-go", action="store_true", help="Required after root confirms all six Appliances loss producers are terminal.")
    parser.add_argument("--print-contract", action="store_true", help="Print expected producer schema without reading artifacts.")
    args = parser.parse_args()
    if args.print_contract:
        print(json.dumps(EXPECTED_PRODUCER_SCHEMA, ensure_ascii=False, indent=2))
        return 0
    if not args.root_go:
        print(json.dumps({"status": "NOT_STARTED_ROOT_GO_REQUIRED", "producer_schema_contract": EXPECTED_PRODUCER_SCHEMA}, ensure_ascii=False, indent=2))
        return 2
    result = audit(args.loss_result, args.bridge_result, args.out)
    print(json.dumps({"status": result["status"], "checks": result["checks"], "failed_count": result["failed_count"], "out": rel(args.out)}, ensure_ascii=False))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
