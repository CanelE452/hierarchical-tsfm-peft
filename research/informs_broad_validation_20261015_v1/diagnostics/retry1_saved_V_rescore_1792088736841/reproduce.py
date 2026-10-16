"""Audit the single new CPU retry saved V; never resume/read the partial checkpoint."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public-root", "cache-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    public, cache, output = (Path(value).resolve() for value in (args.public_root, args.cache_root, args.output))
    if not output.is_relative_to(public / "diagnostics"):
        raise ValueError("New diagnostic output outside public diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    new(output / "RUN_CLAIM.json", {"started_unix": time.time(), "overwrite_allowed": False})
    started = time.time()
    inputs = {}

    def record(path):
        path = Path(path).resolve()
        root, scope = (public, "public") if path.is_relative_to(public) else (cache, "private")
        if not path.is_relative_to(root):
            raise ValueError("Escaped audit input")
        inputs[scope + ":" + path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return path

    def read(path):
        return json.loads(record(path).read_text(encoding="utf-8-sig"))

    identity = "primary_pems08_E1_DIRECT_B4_k0_s92601__cpu_fresh_e3_retry1"
    root13 = public / "diagnostics/new13_saved_V_rescore_1792088232537"
    expected_result = "178be42062dfe778278379116060a406c46408d7c26b7636291379f37e8e2f20"
    expected_closure = "009669386dd1764ee51ee7e58568e5660fb0e9c83073ddefbcd106a1ee38bcfa"
    expected_manifest = "c66916bf98467485477d4bfcc7e49cbb2633eb4df31739e5086675a0444342fa"
    execution = public / "diagnostics/native_cpu_retry1_execution_1792088105694155400"
    if sha(execution / "RESULT.json") != expected_result or sha(execution / "CLOSURE.json") != expected_closure:
        raise ValueError("Exact retry execution/closure bytes differ")
    executed, closure = read(execution / "RESULT.json"), read(execution / "CLOSURE.json")
    if closure["actual_runtime_exitcode"] != 0 or executed["run_id"] != identity or closure["run_id"] != identity or closure["original_partial_state_read_or_reused"]:
        raise ValueError("Retry execution terminal/fresh-state contract differs")
    base13 = read(root13 / "RESULT_PUBLIC.json")
    source_before = {name: sha(public / name) for name in base13["sources_after"]}
    if source_before != base13["sources_after"] or len(source_before) != 39:
        raise ValueError("Prior source39 contract differs")
    old_union = list(csv.DictReader(record(root13 / "UNION_BUNDLES.csv").open(encoding="utf-8", newline="")))
    old_keys = {(row["run_id"], row["split"], row["cohort"]) for row in old_union}
    old_paths = {row["archive_cache_relative_path"] for row in old_union}
    if len(old_union) != len(old_keys) or len(old_union) != len(old_paths) or len(old_union) != 408 or any(row["run_id"] == identity for row in old_union):
        raise ValueError("Prior408 bundle union identity differs")
    path = public / "predictions" / identity / "V_select/PREDICTION_MANIFEST.json"
    if sha(path) != expected_manifest:
        raise ValueError("Exact fresh retry V manifest differs")
    manifest, reference = read(path), read(path.with_name("REFERENCE_CONTRACT.json"))
    receipt = read(public / "runs" / identity / "BASE_FIT_RECEIPT.json")
    source = read(public / "runs" / identity / "SOURCE_CONTRACT.json")
    stats = record(public / manifest["public_stats_relative_path"])
    archive = record(cache / manifest["cache_relative_path"])
    artifact = record(cache / receipt["artifact_cache_relative_path"])
    score_path = record(path.with_name("SCORES.csv"))
    for private_path in (archive, artifact):
        if not private_path.is_relative_to(cache):
            raise ValueError("Prediction/artifact outside declared cache")
    if sha(archive) != manifest["archive_sha256"] or sha(artifact) != receipt["artifact_sha256"] or manifest["metadata"]["artifact_sha256"] != receipt["artifact_sha256"]:
        raise ValueError("Original prediction/selected artifact byte SHA differs")
    if source["model_sha"] != "NO_FOUNDATION" or receipt["status"] != "COMPLETE_VALID" or source["spec"]["device"] != "cpu" or source["spec"]["family"] != "DIRECT_B4":
        raise ValueError("Original NO_F CPU/direct completed fit contract differs")
    if source["spec"]["provenance"]["previous_CPU_weights_optimizer_RNG_or_state_read_or_reused"]:
        raise ValueError("This is not a separately fresh retry")
    if source["files"] != {name: source_before[name] for name in source["files"]} or source["source_sha"] != manifest["metadata"]["training_source_sha"]:
        raise ValueError("Original training source differs")
    expected_model = fingerprint(dict(artifact_sha=receipt["artifact_sha256"], model_state_sha=receipt["model_state_sha"], normalization=receipt["refit_scaler_sha"]))
    if manifest["model_sha"] != expected_model or manifest["scaler_sha"] != receipt["refit_scaler_sha"]:
        raise ValueError("Selected model/scaler identity differs")
    asset = read(public / "DATA_CONTRACT.json")["assets"]["pems08"]
    base = read(public / "predictions/primary_pems08_E1_DIRECT_G_k0_s92601/V_select/PREDICTION_MANIFEST.json")
    if manifest["run_id"] != identity or manifest["split"] != "V_select" or manifest["cohort"] != "pems08:primary:V_select:stride12" or manifest["data_sha"] != asset["data_sha256"]:
        raise ValueError("Original V dataset/fold/cohort differs")
    if manifest["metadata"]["fold"] != "primary" or source["spec"]["fold"] != "primary" or manifest["scaler_sha"] != base["scaler_sha"]:
        raise ValueError("Original primary A60 scaler/fold differs")
    if receipt["refit_fit_end"] != asset["primary_split_ranges"]["A_stop"][1] or any(manifest["arrays"][key] != base["arrays"][key] for key in ("target_normalized", "observed", "origins", "channel_ids")):
        raise ValueError("Original A60 or cohort target/mask/origin/channel fingerprint differs")
    registry_path = public / "EXPERIMENT_REGISTRY.csv"
    registry_bytes = registry_path.read_bytes()
    registry = list(csv.DictReader(io.StringIO(registry_bytes.decode("utf-8-sig"))))
    by_id = {row["experiment_id"]: row for row in registry}
    if len(by_id) != len(registry) or identity not in by_id or by_id[identity]["fold"] != "primary" or by_id[identity]["dataset"] != "pems08" or by_id[identity]["family"] != "DIRECT_B4":
        raise ValueError("Original registry identity differs")
    validation = [{"run_id": row["experiment_id"], "execution_status": "COMPLETE_VALID" if row["experiment_id"] == identity else row["execution_status"],
                   "reason": row.get("reason") or "No targeted completed artifact in this delta audit", "original_execution_status": row["execution_status"]} for row in registry]
    new(output / "VALIDATION_REGISTRY_PROJECTION.json", validation)
    new(output / "INPUTS_BEFORE.json", {"source39": source_before, "inputs": inputs, "registry_sha256": hashlib.sha256(registry_bytes).hexdigest()})
    forbidden = []

    def guard(event, values):
        if event == "import" and str(values[0]).split(".")[0] in {"torch", "chronos"}:
            forbidden.append("MODEL_RUNTIME_IMPORT");raise RuntimeError("No model runtime imports")
        if event == "pickle.find_class":
            forbidden.append("PICKLE_DESERIALIZATION");raise RuntimeError("No pickle deserialization")
        if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        opened = Path(os.fsdecode(values[0])).resolve()
        if "/t_final/" in opened.as_posix().casefold() or opened.name.casefold() in {"gpu_owner.lock", "active_gpu_lease.json"}:
            forbidden.append("T_OR_GPU_OWNER");raise RuntimeError("No T/GPU owner access")
        if opened.suffix.casefold() == ".npz" and opened != archive:
            forbidden.append("OTHER_NPZ");raise RuntimeError("Only retry1 saved prediction NPZ allowed")
        if opened.suffix.casefold() in {".pt", ".safetensors"} and opened != artifact:
            forbidden.append("OTHER_WEIGHT");raise RuntimeError("Only retry1 selected artifact bytes may be SHA-read")
    sys.addaudithook(guard)
    sys.path.insert(0, str(public))
    import audit_predictions as audit
    import numpy as np
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        with gzip.open(stats, "rt", encoding="utf-8-sig", newline="") as stream:
            result = audit.audit_prediction_bundle(manifest, reference, manifest["score"], csv.DictReader(stream), validation, list(by_id), [manifest["cost_record"]], cache)
        with np.load(archive, allow_pickle=False) as arrays:
            bounds = asset["primary_split_ranges"]["V_select"]
            origins = np.arange(max(bounds[0], asset["L"]), bounds[1] - asset["H"] + 1, asset["H"], dtype=np.int64)
            if not np.array_equal(arrays["origins"], origins) or not np.array_equal(arrays["channel_ids"], np.asarray(asset["columns"], dtype=str)):
                raise ValueError("Actual origin/channel coordinate arrays differ")
    score_rows = list(csv.DictReader(score_path.open(encoding="utf-8-sig", newline="")))
    if len(score_rows) != 1 or any(float(score_rows[0][key]) != float(result[key]) for key in ("mse", "mae", "n_observed", "n_origins", "num_channels", "horizon", "active_channels")):
        raise ValueError("Original score CSV differs from independent FP64 result")
    if manifest["cache_relative_path"] in old_paths or (identity, manifest["split"], manifest["cohort"]) in old_keys:
        raise ValueError("Retry1 overlaps prior408 bundle identity/archive path")
    inputs_after = {}
    for key, expected in inputs.items():
        scope, relative = key.split(":", 1)
        target = (public if scope == "public" else cache) / relative
        inputs_after[key] = {"sha256": sha(target), "bytes": target.stat().st_size}
    source_after = {name: sha(public / name) for name in source_before}
    if inputs != inputs_after or source_before != source_after or forbidden or "torch" in sys.modules:
        raise AssertionError("Input/source/isolation changed")
    registry_after = list(csv.DictReader(registry_path.open(encoding="utf-8-sig", newline="")))
    after_map = {row["experiment_id"]: row for row in registry_after}
    mutable = {"execution_status", "training_status", "validity_status", "scientific_status", "publication_status", "reason", "resume_command"}
    if any(identity0 not in after_map or any(after_map[identity0][key] != value for key, value in row.items() if key not in mutable) for identity0, row in by_id.items()):
        raise ValueError("Existing scientific registry identity changed")
    result.update(reported_mse=manifest["score"]["mse"], reported_mae=manifest["score"]["mae"], mse_difference=result["mse"] - manifest["score"]["mse"], mae_difference=result["mae"] - manifest["score"]["mae"],
        training_status=receipt["training_status"], selected_epoch=receipt["selected_epoch"], last_cp_best=receipt["last_cp_best"], last_scores_still_falling=receipt["last_scores_still_falling"],
        actual_F_streams=manifest["cost_record"]["actual_F_streams"], expected_F_streams=manifest["cost_record"]["expected_F_streams"], literal_F_calls_field=manifest["cost_record"]["F_calls"],
        literal_F_calls_is_not_foundation_invocation_count=True, manifest_sha256=sha(path), reference_sha256=sha(path.with_name("REFERENCE_CONTRACT.json")), archive_sha256=sha(archive), selected_artifact_sha256=sha(artifact),
        previous_unique_saved_bundles=408, new_actual_rescores=1, previous408_rerescored=False, union_unique_saved_bundles=409, union_unique_V=373, union_unique_dense_M=36, identity_and_archive_path_overlap=0,
        source39_before=source_before, source39_after=source_after, source39_unchanged=True, immutable_input_files=len(inputs), input_hashes_unchanged=True, registry_before_sha256=hashlib.sha256(registry_bytes).hexdigest(), registry_after_sha256=sha(registry_path),
        original_registry_execution_status=by_id[identity]["execution_status"], registry_written=False, raw_Task_or_new_scaler_loaded=False, selected_artifact_deserialized=False, old_partial_state_read_or_reused=False,
        torch_imported=False, T_accessed=False, GPU_lease_acquired=False, global_scores_checked=False, Git_calls=0, forbidden_io_attempts=forbidden, CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
        CPU_wall_seconds=time.time() - started, purpose="Only separate retry1 saved-V FP64 audit; no convergence/replay/generalization/campaign completion claim", reproducible_driver_sha256=sha(__file__))
    new(output / "INPUTS_AFTER.json", {"source39": source_after, "inputs": inputs_after})
    new(output / "RESULT.json", result)
    fields = ("run_id", "status", "reported_mse", "mse", "mse_difference", "reported_mae", "mae", "mae_difference", "n_observed", "n_origins", "num_channels", "horizon", "n_statistics", "training_status", "selected_epoch", "actual_F_streams", "archive_sha256", "selected_artifact_sha256")
    with (output / "SAVED_V_RESCORING.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();writer.writerow({key: result[key] for key in fields})
    print(json.dumps({"status": result["status"], "actual_new_rescores": 1, "mse": result["mse"], "mae": result["mae"], "union_bundles": 409, "seconds": result["CPU_wall_seconds"]}))


if __name__ == "__main__":
    main()
