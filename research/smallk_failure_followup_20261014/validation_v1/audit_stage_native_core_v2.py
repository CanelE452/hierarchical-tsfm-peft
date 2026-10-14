"""Independent CPU audit for stage-separation native ridge producers.

Default import/describe is source-only.  Actual producer RESULT/NPZ reads occur
only when --execute is supplied after the fresh stage-separation runtime GO.
This auditor verifies saved arrays and metadata; it does not rerun SVD/lstsq
training, original E/F forward, GPU work, or old campaign jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
VALIDATION = HERE / "validation_v1"
DEFAULT_OUT = VALIDATION / "STAGE_NATIVE_AUDIT_v1.json"
ALPHAS = (0.0, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0)
LENGTH = 512
HORIZON = 48
ATOL = 1e-5
RTOL = 2e-6
DATASETS = {"robin": (17, 4, "val", 30), "beijing_aotizhongxin": (11, 2, "eval", 146)}
FAMILIES = ("O1", "N1", "N2", "N3")


class AuditFailure(Exception):
    pass


def sha_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path: Path | str, value: Any) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"path": str(path.resolve()), "sha256": sha_file(path), "bytes": path.stat().st_size}


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_sha(kind: str, *values: Any) -> str:
    digest = hashlib.sha256(kind.encode())
    for value in values:
        digest.update((array_sha(value) if isinstance(value, np.ndarray) else str(value)).encode())
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_npz_record(record: dict[str, Any], *, role: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = resolve(record["path"])
    if sha_file(path) != record.get("sha256"):
        raise AuditFailure(f"{role}_FILE_SHA_MISMATCH")
    with np.load(path, allow_pickle=False) as opened:
        arrays = {key: opened[key].copy() for key in opened.files}
    declared = record.get("arrays")
    if declared is not None:
        if set(arrays) != set(declared):
            raise AuditFailure(f"{role}_ARRAY_KEY_MISMATCH")
        for key, value in arrays.items():
            entry = declared[key]
            if list(value.shape) != entry.get("shape") or str(value.dtype) != entry.get("dtype") or array_sha(value) != entry.get("sha256"):
                raise AuditFailure(f"{role}_ARRAY_CONTENT_MISMATCH:{key}")
    return arrays, {"path": str(path), "sha256": record.get("sha256"), "role": role}


def approx_equal(left: Any, right: Any, *, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))


def assert_score_close(recorded: dict[str, Any], computed: dict[str, Any], *, label: str, failures: list[str]) -> None:
    for key in ("mse", "mae"):
        if not approx_equal(recorded.get(key), computed.get(key), atol=1e-8, rtol=1e-8):
            failures.append(f"SCORE_{key.upper()}_MISMATCH:{label}")
    for key in ("active_channels",):
        if recorded.get(key) != computed.get(key):
            failures.append(f"SCORE_{key.upper()}_MISMATCH:{label}")
    for key in ("observed_counts",):
        if recorded.get(key) != computed.get(key):
            failures.append(f"SCORE_{key.upper()}_MISMATCH:{label}")


def observed_score(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray, std: np.ndarray | None = None) -> dict[str, Any]:
    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    if std is not None:
        error = error * np.asarray(std, np.float64)
    observed = np.asarray(observed)
    error = np.where(observed, error, 0.0)
    count = observed.sum(axis=(0, 1))
    active = count > 0
    mse = np.divide((error ** 2).sum((0, 1)), count, out=np.zeros(len(count), np.float64), where=active)
    mae = np.divide(np.abs(error).sum((0, 1)), count, out=np.zeros(len(count), np.float64), where=active)
    return {
        "mse": float(mse[active].mean()) if active.any() else None,
        "mae": float(mae[active].mean()) if active.any() else None,
        "per_channel_mse": [float(value) if is_active else None for value, is_active in zip(mse, active)],
        "per_channel_mae": [float(value) if is_active else None for value, is_active in zip(mae, active)],
        "observed_counts": count.tolist(),
        "active_channels": int(active.sum()),
    }


def fixed_past_features(latent_past: np.ndarray) -> np.ndarray:
    value = np.asarray(latent_past)
    if value.ndim != 3 or value.shape[1] != LENGTH or not value.shape[0] or not value.shape[2]:
        raise AuditFailure("FINITE_NONEMPTY_N_512_K_PAST_REQUIRED")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise AuditFailure("PAST_LATENT_FLOAT_FINITE_REQUIRED")
    blocks = value.reshape(len(value), 32, 16, value.shape[2]).mean(axis=2)
    return np.concatenate((blocks, value[:, -1:]), axis=1).reshape(len(value), -1)


def transform_features(feature_state: dict[str, Any], latent_past: np.ndarray) -> np.ndarray:
    raw = fixed_past_features(latent_past)
    center, scale = feature_state["center"], feature_state["scale"]
    if raw.shape[1] != len(center):
        raise AuditFailure("FEATURE_K_MISMATCH")
    return (raw.astype(np.float64) - center) / scale


def feature_state_sha(center: np.ndarray, scale: np.ndarray, latent_channels: int, fit_rows: int, fit_raw_features_sha256: str) -> str:
    return state_sha("FIT_33K_FEATURE_STATE", center, scale, latent_channels, fit_rows, fit_raw_features_sha256)


def solution_sha(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    return state_sha(
        meta["kind"],
        float(meta["alpha"]),
        arrays["coefficients"],
        arrays["intercept"],
        arrays["observed_counts"],
        arrays["channel_counts"],
        meta["feature_state_sha256"],
    )


def native_model_sha(feature_sha: str, solution_hash: str) -> str:
    return state_sha("FIXED_NATIVE_MODEL", feature_sha, solution_hash)


def decoder_model_sha(meta: dict[str, Any], arrays: dict[str, np.ndarray], fixed_n1: str | None) -> str:
    return state_sha(
        meta["kind"],
        float(meta["alpha"]),
        arrays["effective_weight"],
        arrays["feature_rms"],
        arrays["hidden_weight"],
        arrays["hidden_bias"],
        fixed_n1,
    )


def predict_solution(design: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    coefficients = arrays["coefficients"]
    intercept = arrays["intercept"]
    flat = design @ coefficients.reshape(design.shape[1], -1) + intercept.reshape(-1)
    return flat.reshape((len(design),) + intercept.shape)


def gelu_torch(value: np.ndarray) -> np.ndarray:
    import torch
    from torch.nn import functional as F

    tensor = torch.from_numpy(np.array(value, dtype=np.float32, copy=True, order="C"))
    return F.gelu(tensor, approximate="none").numpy()


def decoder_features(z_future: np.ndarray, z_last: np.ndarray, decoder: dict[str, np.ndarray]) -> np.ndarray:
    z_future = np.asarray(z_future, np.float32)
    z_last = np.asarray(z_last, np.float32)
    weight = np.asarray(decoder["residual.0.weight"] if "residual.0.weight" in decoder else decoder["hidden_weight"], np.float32)
    bias = np.asarray(decoder["residual.0.bias"] if "residual.0.bias" in decoder else decoder["hidden_bias"], np.float32)
    hidden_future = gelu_torch(z_future @ weight.T + bias)
    hidden_last = gelu_torch(z_last @ weight.T + bias)
    return np.concatenate((z_future - z_last, hidden_future - hidden_last), axis=2)


def original_anchored_prediction(z_future: np.ndarray, z_last: np.ndarray, raw_last: np.ndarray, decoder: dict[str, np.ndarray]) -> np.ndarray:
    import torch
    from torch.nn import functional as F

    tensors = {key: torch.from_numpy(np.array(value, dtype=np.float32, copy=True, order="C")) for key, value in decoder.items()}
    zf = torch.from_numpy(np.array(z_future, dtype=np.float32, copy=True, order="C"))
    zl = torch.from_numpy(np.array(z_last, dtype=np.float32, copy=True, order="C"))
    last = torch.from_numpy(np.array(raw_last, dtype=np.float32, copy=True, order="C"))

    def decode(z):
        linear = F.linear(z, tensors["codec.decode.weight"], tensors["codec.decode.bias"])
        hidden = F.gelu(F.linear(z, tensors["residual.0.weight"], tensors["residual.0.bias"]), approximate="none")
        return linear + F.linear(hidden, tensors["residual.2.weight"], tensors["residual.2.bias"])

    return (last[:, None] + decode(zf) - decode(zl)).numpy()


def decoder_calibration_prediction(z_future: np.ndarray, z_last: np.ndarray, raw_last: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    features = decoder_features(z_future, z_last, arrays)
    return (np.asarray(raw_last, np.float32)[:, None] + features @ np.asarray(arrays["effective_weight"], np.float32)).astype(np.float32, copy=False)


def find_artifact_by_stem(producer: dict[str, Any], stem: str) -> dict[str, Any]:
    matches = [record for record in producer.get("artifacts", []) if Path(record["path"]).stem == stem]
    if len(matches) != 1:
        raise AuditFailure("ARTIFACT_STEM_LOOKUP_FAILED:" + stem)
    return matches[0]


def load_source_bundle(producer: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any]]:
    splits = {}
    for split in expected_splits(producer["dataset"]):
        splits[split], _ = load_npz_record(find_artifact_by_stem(producer, "INPUT_" + split), role="INPUT_" + split)
    decoder, _ = load_npz_record(producer["source_decoder_artifact"], role="SOURCE_DECODER")
    feature_arrays, _ = load_npz_record(producer["feature_state"]["artifact"], role="FEATURE_STATE")
    feature_record = producer["feature_state"]
    recomputed = feature_state_sha(
        feature_arrays["center"],
        feature_arrays["scale"],
        feature_record["latent_channels"],
        feature_record["fit_rows"],
        feature_record["fit_raw_features_sha256"],
    )
    if recomputed != feature_record["sha256"]:
        raise AuditFailure("FEATURE_STATE_SHA_MISMATCH")
    return splits, decoder, {**feature_record, **feature_arrays}


def expected_splits(dataset: str) -> tuple[str, str, str]:
    return ("fit", "tune", "val" if dataset == "robin" else "eval")


def check_source_records(records: list[dict[str, Any]], failures: list[str]) -> int:
    checked = 0
    for record in records:
        path = record.get("path") or record.get("snapshot_path")
        expected = record.get("sha256")
        if not path or not expected:
            failures.append("SOURCE_RECORD_SCHEMA")
            continue
        path = resolve(path)
        if not path.exists():
            failures.append("SOURCE_RECORD_MISSING:" + str(path))
            continue
        if sha_file(path) != expected:
            failures.append("SOURCE_RECORD_SHA_MISMATCH:" + str(path))
        checked += 1
    return checked


def audit_candidate_scores(producer: dict[str, Any], splits: dict[str, dict[str, np.ndarray]], failures: list[str]) -> int:
    count = 0
    for family, row in producer.get("families", {}).items():
        if family not in FAMILIES or row.get("status") != "COMPLETE":
            continue
        alpha_rows = row.get("alpha_grid", [])
        if tuple(entry.get("alpha") for entry in alpha_rows) != ALPHAS:
            failures.append("ALPHA_GRID_NOT_EXACT_SEVEN:" + family)
            continue
        selected = min(range(len(alpha_rows)), key=lambda index: alpha_rows[index]["tune_actual_Y_mse"])
        if selected != row.get("selected_index"):
            failures.append("SELECTION_INDEX_MISMATCH:" + family)
        if row.get("selected_alpha") != alpha_rows[selected]["alpha"]:
            failures.append("SELECTION_ALPHA_MISMATCH:" + family)
        for alpha_index, entry in enumerate(alpha_rows):
            count += 1
            for split in ("fit", "tune"):
                artifact, _ = load_npz_record(entry[split + "_artifact"], role=f"{family}_alpha{alpha_index}_{split}")
                computed = observed_score(artifact["prediction"], splits[split]["target"], splits[split]["observed"])
                assert_score_close(entry[split + "_scores"]["normalized"], computed, label=f"{family}:{alpha_index}:{split}", failures=failures)
            if not approx_equal(entry["tune_actual_Y_mse"], entry["tune_scores"]["normalized"]["mse"], atol=0.0, rtol=0.0):
                failures.append("TUNE_ACTUAL_Y_MSE_FIELD_MISMATCH:" + family + ":" + str(alpha_index))
    return count


def load_selected_arrays(row: dict[str, Any], *, role: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays, _ = load_npz_record(row["selected_coefficient_artifact"], role=role + "_SELECTED_COEFFICIENTS")
    return row["selected_model_metadata"], arrays


def audit_model_hashes(producer: dict[str, Any], feature_state: dict[str, Any], failures: list[str]) -> dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]]:
    restored = {}
    for family, row in producer.get("families", {}).items():
        if family not in FAMILIES or row.get("status") != "COMPLETE":
            continue
        meta, arrays = load_selected_arrays(row, role=family)
        sol_meta = meta["solution"]
        sol_hash = solution_sha(sol_meta, arrays)
        if sol_hash != sol_meta["solution_sha256"]:
            failures.append("SOLUTION_SHA_MISMATCH:" + family)
        if family in ("N1", "N2"):
            model_hash = native_model_sha(feature_state["sha256"], sol_hash)
        else:
            model_hash = decoder_model_sha(sol_meta, arrays, meta.get("fixed_n1_model_sha256"))
        if model_hash != row.get("selected_model_sha256"):
            failures.append("SELECTED_MODEL_SHA_MISMATCH:" + family)
        restored[family] = (meta, arrays)
    if "N3" in restored and "N1" in restored:
        if restored["N3"][0].get("fixed_n1_model_sha256") != producer["families"]["N1"].get("selected_model_sha256"):
            failures.append("N3_NOT_BOUND_TO_SELECTED_N1")
    return restored


def recompute_selected_predictions(
    producer: dict[str, Any],
    splits: dict[str, dict[str, np.ndarray]],
    decoder: dict[str, np.ndarray],
    feature_state: dict[str, Any],
    restored: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]],
    failures: list[str],
) -> int:
    checked = 0
    n1_latents = {}
    for family, (meta, arrays) in restored.items():
        row = producer["families"][family]
        for split, values in splits.items():
            design = transform_features(feature_state, values["latent_past"])
            latent = None
            if family == "N1":
                innovation = predict_solution(design, arrays).astype(np.float32)
                latent = (values["latent_past"][:, -1:] + innovation).astype(np.float32)
                prediction = original_anchored_prediction(latent, values["z_last"], values["raw_last"], decoder)
                n1_latents[split] = latent
            elif family == "N2":
                innovation = predict_solution(design, arrays).astype(np.float32)
                prediction = (values["raw_last"][:, None] + innovation).astype(np.float32)
            elif family == "N3" and "N1" in n1_latents:
                latent = n1_latents[split]
                prediction = decoder_calibration_prediction(latent, values["z_last"], values["raw_last"], arrays)
            elif family == "O1":
                oracle_record = producer.get("oracle_source")
                oracle_input_stem = "ORACLE_INPUT_DIAGNOSTIC_" + split
                oracle_arrays, _ = load_npz_record(find_artifact_by_stem(producer, oracle_input_stem), role=oracle_input_stem)
                prediction = decoder_calibration_prediction(oracle_arrays["oracle_z"], values["z_last"], values["raw_last"], arrays)
            else:
                continue
            artifact, _ = load_npz_record(row["selected_artifacts"][split], role=f"{family}_SELECTED_{split}")
            if not np.allclose(prediction, artifact["prediction"], atol=ATOL, rtol=RTOL):
                failures.append("SELECTED_PREDICTION_RECONSTRUCTION_MISMATCH:" + family + ":" + split)
            computed = observed_score(artifact["prediction"], artifact["target"], artifact["observed"])
            assert_score_close(row["splits"][split]["normalized"], computed, label=f"{family}:selected:{split}", failures=failures)
            if "predicted_latent_future" in artifact and latent is not None:
                if not np.allclose(latent, artifact["predicted_latent_future"], atol=ATOL, rtol=RTOL):
                    failures.append("SELECTED_LATENT_RECONSTRUCTION_MISMATCH:" + family + ":" + split)
            checked += 1
    if "N1" in producer.get("families", {}) and "N3" in producer.get("families", {}):
        n1 = producer["families"]["N1"]
        n3 = producer["families"]["N3"]
        if n1.get("status") == n3.get("status") == "COMPLETE":
            for split in splits:
                a, _ = load_npz_record(n1["selected_artifacts"][split], role="N1_SELECTED_LATENT_" + split)
                b, _ = load_npz_record(n3["selected_artifacts"][split], role="N3_SELECTED_LATENT_" + split)
                if not np.array_equal(a["predicted_latent_future"], b["predicted_latent_future"]):
                    failures.append("N3_LATENT_NOT_EXACT_SELECTED_N1:" + split)
    return checked


def audit_one(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    producer = read_json(path)
    producer_sha = sha_file(path)
    failures: list[str] = []
    if producer.get("status") != "COMPLETE":
        failures.append("PRODUCER_NOT_COMPLETE")
    dataset = producer.get("dataset")
    seed = producer.get("seed")
    if dataset not in DATASETS or seed not in (92601, 92602):
        failures.append("PRODUCER_CONDITION_IDENTITY")
    if producer.get("F_loaded") is not False or producer.get("F_forward_replayed") is not False or producer.get("E_forward_replayed") is not False:
        failures.append("PRODUCER_REPLAY_SCOPE_FLAGS")
    if producer.get("oracle_is_deployable") is not False or producer.get("physical_macro_is_primary") is not False:
        failures.append("PRODUCER_SCOPE_FLAGS")
    if producer.get("codec_before") != producer.get("codec_after") or producer.get("codec_before") != producer.get("codec_terminal_hash"):
        failures.append("CODEC_DUAL_HASH_TERMINAL_MISMATCH")
    if producer.get("old_E_D_F_updated") is not False:
        failures.append("OLD_E_D_F_UPDATED_FLAG")

    splits, decoder, feature_state = load_source_bundle(producer)
    if DATASETS.get(dataset, (None, None, None, None))[1] != producer.get("feature_state", {}).get("latent_channels"):
        failures.append("FEATURE_STATE_K_MISMATCH")
    source_record_count = check_source_records(producer.get("source_records", []), failures)
    source_record_count += check_source_records(producer.get("source_snapshots", []), failures)
    candidate_count = audit_candidate_scores(producer, splits, failures)
    restored = audit_model_hashes(producer, feature_state, failures)
    selected_prediction_count = recompute_selected_predictions(producer, splits, decoder, feature_state, restored, failures)

    teacher = producer.get("teacher_contract", {})
    if "N1" in producer.get("families", {}) and producer["families"]["N1"].get("status") == "COMPLETE":
        if teacher.get("mask_rule") != "all original C observed through entire future prefix; repeated K":
            failures.append("N1_TEACHER_MASK_RULE")
        if not teacher.get("fit_counts_by_horizon_K"):
            failures.append("N1_TEACHER_COUNTS_MISSING")
    n3_contract = producer.get("N3_fixed_N1_contract", {})
    if producer.get("families", {}).get("N3", {}).get("status") == "COMPLETE":
        if n3_contract.get("N1_coefficients_reselected_for_N3") is not False:
            failures.append("N3_RESELECTED_N1_FLAG")

    family_status = {family: producer.get("families", {}).get(family, {}).get("status", "MISSING") for family in FAMILIES}
    return {
        "dataset": dataset,
        "seed": seed,
        "producer": {"path": str(path.resolve()), "sha256": producer_sha},
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "family_status": family_status,
        "candidate_rows_checked": candidate_count,
        "selected_predictions_checked": selected_prediction_count,
        "source_records_checked": source_record_count,
        "scope": {
            "actual_E_or_F_forward_replayed": False,
            "solver_svd_lstsq_replayed": False,
            "audit_equations": "feature transforms, saved coefficients, CPU torch GELU decoder, score/selection/model-hash reconstruction",
        },
    }


def producer_paths_from_queue(path: Path | str) -> list[Path]:
    payload = read_json(path)
    candidates = payload.get("conditions") or payload.get("producers") or payload.get("rows")
    if not isinstance(candidates, list):
        raise AuditFailure("QUEUE_CONDITIONS_LIST_REQUIRED")
    paths = []
    for row in candidates:
        record = row.get("producer_result", row)
        if not isinstance(record, dict) or not record.get("path"):
            raise AuditFailure("QUEUE_PRODUCER_RESULT_PATH_REQUIRED")
        paths.append(resolve(record["path"]))
    return paths


def execute(paths: list[Path], *, out: Path) -> dict[str, Any]:
    if len(paths) != 4:
        raise AuditFailure("EXACT_FOUR_STAGE_NATIVE_PRODUCERS_REQUIRED")
    rows = [audit_one(path) for path in paths]
    conditions = [
        {
            "dataset": row["dataset"],
            "seed": row["seed"],
            "producer_path": row["producer"]["path"],
            "producer_sha256": row["producer"]["sha256"],
            "status": row["status"],
        }
        for row in rows
    ]
    failures = [failure for row in rows for failure in row["failures"]]
    result = {
        "schema": "smallk_stage_native_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "failed_count": len(failures),
        "conditions": conditions,
        "rows": rows,
        "candidate_rows_checked": sum(row["candidate_rows_checked"] for row in rows),
        "selected_predictions_checked": sum(row["selected_predictions_checked"] for row in rows),
        "scientific_scope": "independent CPU array audit of saved native ridge producer outputs; no E/F forward or solver replay",
    }
    result["artifact"] = write_new_json(out, result)
    return result


def describe() -> dict[str, Any]:
    return {
        "status": "SOURCE_ONLY_NOT_EXECUTED",
        "schema": "smallk_stage_native_audit_v1",
        "expected_conditions": "exactly 4 producer RESULT.json files; conditions rows expose dataset/seed/producer_sha256",
        "families": list(FAMILIES),
        "candidate_rows": "4 producers x 4 families x 7 alpha rows when every family is COMPLETE; partial family status is reported",
        "no_actual_reads_on_import_or_describe": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--producer-result", action="append", default=[])
    parser.add_argument("--queue")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0
    paths = [resolve(path) for path in args.producer_result]
    if args.queue:
        paths.extend(producer_paths_from_queue(args.queue))
    result = execute(paths, out=Path(args.out))
    print(json.dumps({"status": result["status"], "failed_count": result["failed_count"], "conditions": result["conditions"]}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
