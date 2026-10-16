"""Read-only saved-V FP64 delta audit; no Task/model/checkpoint deserialization."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

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
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--private-output", required=True)
    args = parser.parse_args()
    public, cache, output, private = (Path(value).resolve() for value in (args.public_root, args.cache_root, args.output, args.private_output))
    if not output.is_relative_to(public / "diagnostics") or not private.is_relative_to(cache):
        raise ValueError("New diagnostic/private output must remain in their declared roots")
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=False)
    new(output / "RUN_CLAIM.json", {"started_unix": time.time(), "overwrite_allowed": False})
    started = time.time()
    inputs = {}

    def record(path):
        path = Path(path).resolve()
        root, scope = (public, "public") if path.is_relative_to(public) else (cache, "private")
        if not path.is_relative_to(root):
            raise ValueError("Input outside declared roots")
        inputs[scope + ":" + path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return path

    def read(path):
        return json.loads(record(path).read_text(encoding="utf-8-sig"))

    def inside(root, relative):
        relative = Path(relative)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or path == root:
            raise ValueError("Escaped relative input")
        return path

    def sanitize(error):
        text = str(error)
        for root, token in ((cache, "PRIVATE_CACHE"), (public, "PUBLIC_CAMPAIGN")):
            text = text.replace(str(root), token).replace(root.as_posix(), token)
        return text

    summary = {"status": "NOT_COMPLETED", "checkpoint_replay": False, "training_rerun": False,
               "timer_reexecution": False, "T_accessed": False, "global_scores_checked": False,
               "registry_written": False, "model_weights_deserialized": False, "GPU_lease_acquired": False}
    errors, rows, results = [], [], []
    try:
        small = read(public / "execution_batches/remaining_breadth_1792085507193441200.json")
        handoff = read(public / "diagnostics/ROOT_Ada_native_retry_handoff_1792087803740204900/RESULT.json")
        native = read(public / "diagnostics/native_cpu_original9_seal_1792087586733966400/RESULT.json")
        native_rows = list(csv.DictReader(record(public / "diagnostics/native_cpu_execution_1792085507073860400/TRAINING_AND_V_SUMMARY.csv").open(encoding="utf-8-sig", newline="")))
        small_ids = [row["run_id"] for row in small["routes"] if row.get("status") == "COMPLETE_VALID"]
        native_ids = [row["run_id"] for row in native_rows if row["actual_status"] == "FIT_AND_V_COMPLETE_NO_F_CPU"]
        ada_id = handoff["Ada_actual_fit_and_V"]["run_id"]
        ids = small_ids + [ada_id] + native_ids
        if len(small_ids) != 4 or len(native_ids) != 8 or len(ids) != len(set(ids)) or len(ids) != 13:
            raise ValueError("Exact smallK4/Ada1/original native8 target contract differs")
        if native["fit_and_V_complete"] != 8 or native["partial_run_id"] in ids or "retry" in " ".join(ids):
            raise ValueError("Partial original/retry accidentally included")
        prior_specs = [("audits/audit_1792079299509044000/PREDICTION_AUDITS.json", None),
                       ("diagnostics/representative_e3_rescore_1792081904837511500/RESULT.json", "records"),
                       ("diagnostics/new12_saved_V_rescore_1792083486504811200/NEW12_AUDIT_RESULTS.json", "records"),
                       ("diagnostics/lag_e3_independent_rescore_1792085707119751700/RESULT.json", "records")]
        previous = []
        for relative, key in prior_specs:
            value = read(public / relative)
            previous.extend(value if key is None else value[key])
        prior_keys, prior_cache_paths = set(), set()
        for row in previous:
            if not str(row["status"]).startswith("PASS"):
                raise ValueError("Previous numerical audit contains a nonpassing record")
            key = (row["run_id"], row["split"], row.get("cohort"))
            relative = row.get("manifest_public_relative_path", f"predictions/{row['run_id']}/V_select/PREDICTION_MANIFEST.json")
            manifest_path = record(public / relative)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if sha(manifest_path) != row["manifest_sha256"] or manifest["run_id"] != row["run_id"] or manifest["split"] != row["split"]:
                raise ValueError("Previously audited manifest identity/bytes changed")
            actual_key = (manifest["run_id"], manifest["split"], manifest["cohort"])
            if actual_key in prior_keys or manifest["cache_relative_path"] in prior_cache_paths:
                raise ValueError("Previous bundles are not unique by identity/cache path")
            prior_keys.add(actual_key)
            prior_cache_paths.add(manifest["cache_relative_path"])
        if len(previous) != len(prior_keys) or len(previous) != 395:
            raise ValueError("Previous 395 union count differs")
        if any(identity in {key[0] for key in prior_keys} for identity in ids):
            raise ValueError("New targets overlap previous audited identities")
        sources_before = {name: sha(public / name) for name in native["source_pins"]}
        if sources_before != native["source_pins"] or len(sources_before) != 33:
            raise ValueError("Pinned source33 differs")
        for name in ("native_cpu_depth.py", "remaining_breadth.py", "measure_multik_transition.py", "measure_costs.py", "test_multik_transition.py", "MULTIK_TRANSITION_PROTOCOL_KO.md"):
            sources_before[name] = sha(public / name)
        assets = read(public / "DATA_CONTRACT.json")["assets"]
        registry_path = public / "EXPERIMENT_REGISTRY.csv"
        registry_bytes = registry_path.read_bytes()
        (private / "REGISTRY_AT_START.csv").write_bytes(registry_bytes)
        registry = list(csv.DictReader(io.StringIO(registry_bytes.decode("utf-8-sig"))))
        by_id = {row["experiment_id"]: row for row in registry}
        if len(by_id) != len(registry) or not set(ids).issubset(by_id):
            raise ValueError("Target registry IDs missing/duplicated")
        mutable = {"execution_status", "training_status", "validity_status", "scientific_status", "publication_status", "reason", "resume_command"}
        initial_path = record(public / "EXPERIMENT_REGISTRY_INITIAL.csv")
        declared = list(csv.DictReader(initial_path.open(encoding="utf-8-sig", newline="")))
        for path in sorted(public.glob("EXPERIMENT_REGISTRY_EXTENSION_*.csv")):
            extension = list(csv.DictReader(record(path).open(encoding="utf-8-sig", newline="")))
            receipt = read(path.with_suffix(".json"))
            if sha(path) != receipt["csv_sha256"] or not receipt["before_execution"] or not receipt["before_T_selection"]:
                raise ValueError("Original extension receipt/hash contract differs")
            if set(row["experiment_id"] for row in extension) != set(receipt["run_ids"]):
                raise ValueError("Extension IDs differ from original registration receipt")
            declared.extend(extension)
        declared_map = {row["experiment_id"]: row for row in declared}
        if len(declared_map) != len(declared) or set(declared_map) != set(by_id):
            raise ValueError("Initial+extension/current registry ID sets differ")
        for identity, actual in by_id.items():
            if any(actual[key] != value for key, value in declared_map[identity].items() if key not in mutable):
                raise ValueError("Registered scientific identity differs")
        validation = []
        fallback_ids = []
        for row in registry:
            identity, status = row["experiment_id"], row["execution_status"]
            reason = row.get("reason")
            if not reason:
                reason = "No targeted completed artifact in this delta audit"
                fallback_ids.append(identity)
            validation.append({"run_id": identity, "execution_status": "COMPLETE_VALID" if identity in ids else status,
                               "reason": reason, "original_execution_status": status,
                               "projection_scope": "Target saved V artifact only; not overall learning or experiment completion"})
        new(output / "VALIDATION_REGISTRY_PROJECTION.json", validation)
        headers, allowed_npz, allowed_weights, cohort_baselines = {}, set(), set(), {}
        for identity in ids:
            path = public / "predictions" / identity / "V_select/PREDICTION_MANIFEST.json"
            manifest, reference = read(path), read(path.with_name("REFERENCE_CONTRACT.json"))
            receipt = read(public / "runs" / identity / "BASE_FIT_RECEIPT.json")
            source = read(public / "runs" / identity / "SOURCE_CONTRACT.json")
            stats = record(inside(public, manifest["public_stats_relative_path"]))
            score_path = record(path.with_name("SCORES.csv"))
            archive = record(inside(cache, manifest["cache_relative_path"]))
            artifact = record(inside(cache, receipt["artifact_cache_relative_path"]))
            allowed_npz.add(str(archive).casefold())
            allowed_weights.add(str(artifact).casefold())
            dataset = by_id[identity]["dataset"]
            if dataset not in cohort_baselines:
                baseline = read(public / "predictions" / f"primary_{dataset}_E1_DIRECT_G_k0_s92601/V_select/PREDICTION_MANIFEST.json")
                cohort_baselines[dataset] = baseline
            headers[identity] = (path, manifest, reference, receipt, source, stats, score_path, archive, artifact)
        new(output / "FIXED_INPUT_CAPTURE.json", {"new_exact_ids": ids, "groups": {"smallK4": small_ids, "Ada1": [ada_id], "native8": native_ids},
            "previous_unique_bundles": len(prior_keys), "previous_V": sum(key[1] == "V_select" for key in prior_keys),
            "previous_dense_M": sum(key[1] == "M_cal" for key in prior_keys), "previous_new_overlap": [],
            "sources_before": sources_before, "inputs_sha256_before": inputs, "raw_private_arrays_published": False,
            "artifact_files_will_only_be_stream_hashed_not_deserialized": True})
        forbidden = []

        def io_guard(event, values):
            if event == "import" and str(values[0]).split(".")[0] in {"torch", "chronos"}:
                forbidden.append("MODEL_RUNTIME_IMPORT")
                raise RuntimeError("Model runtime imports are outside this audit")
            if event == "pickle.find_class":
                forbidden.append("PICKLE_DESERIALIZATION")
                raise RuntimeError("Pickle deserialization is outside this audit")
            if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(values[0])).resolve()
            low = path.as_posix().casefold()
            if "/t_final/" in low or path.name.casefold() in {"gpu_owner.lock", "active_gpu_lease.json"}:
                forbidden.append("T_OR_GPU_OWNER_OPEN")
                raise RuntimeError("T/GPU ownership is outside this audit")
            if path.suffix.casefold() == ".npz" and str(path).casefold() not in allowed_npz:
                forbidden.append("NONAPPROVED_NPZ_OPEN")
                raise RuntimeError("Only new13 exact prediction NPZs may be opened")
            if path.suffix.casefold() in {".pt", ".safetensors"} and str(path).casefold() not in allowed_weights:
                forbidden.append("NONAPPROVED_WEIGHT_OPEN")
                raise RuntimeError("Only bound selected artifact bytes may be SHA-read")
        sys.addaudithook(io_guard)
        sys.path.insert(0, str(public))
        import audit_predictions as audit
        import numpy as np
        import psutil
        from threadpoolctl import threadpool_limits, threadpool_info
        process = psutil.Process()
        max_rss = process.memory_info().rss
        new_keys, new_cache_paths = set(), set()
        with threadpool_limits(limits=2):
            for identity in ids:
                try:
                    if time.time() - started > 600 or process.memory_info().rss > 2 * 1024 ** 3:
                        raise RuntimeError("CPU audit wall600s or RSS2GiB boundary reached")
                    path, manifest, reference, receipt, source, stats, score_path, archive, artifact = headers[identity]
                    row = by_id[identity]
                    dataset, asset = row["dataset"], assets[row["dataset"]]
                    meta, spec = manifest["metadata"], source["spec"]
                    if receipt["status"] != "COMPLETE_VALID" or receipt["run_id"] != identity or spec["run_id"] != identity:
                        raise ValueError("Completed fit/source identity differs")
                    if source["files"] != {name: sources_before[name] for name in source["files"]}:
                        raise ValueError("Original scientific six-file contract differs")
                    if source["data_sha"] != asset["data_sha256"] or source["source_sha"] != receipt["source_sha"] or meta["training_source_sha"] != source["source_sha"]:
                        raise ValueError("Original data/training source identity differs")
                    if receipt["artifact_sha256"] != sha(artifact) or meta["artifact_sha256"] != receipt["artifact_sha256"] or manifest["archive_sha256"] != sha(archive):
                        raise ValueError("Selected artifact/archive byte SHA differs")
                    expected_model = fingerprint(dict(artifact_sha=receipt["artifact_sha256"], model_state_sha=receipt["model_state_sha"], normalization=receipt["refit_scaler_sha"]))
                    if manifest["model_sha"] != expected_model or manifest["scaler_sha"] != receipt["refit_scaler_sha"]:
                        raise ValueError("Prediction selected-model/scaler binding differs")
                    if row["fold"] != "primary" or spec.get("fold") != "primary" or meta["fold"] != "primary" or meta["dense"]:
                        raise ValueError("Primary sparse-V fold contract differs")
                    expected_cohort = f"{dataset}:primary:V_select:stride{asset['H']}"
                    if manifest["run_id"] != identity or manifest["split"] != "V_select" or manifest["cohort"] != expected_cohort or manifest["data_sha"] != asset["data_sha256"]:
                        raise ValueError("V cohort/data identity differs")
                    baseline = cohort_baselines[dataset]
                    if manifest["scaler_sha"] != baseline["scaler_sha"] or any(manifest["arrays"][name] != baseline["arrays"][name] for name in audit.REFERENCE_KEYS):
                        raise ValueError("New V targets/masks/origins/channels/scaler differ from original cohort")
                    if meta["parent_fit_end"] != asset["primary_split_ranges"]["A_stop"][1] or receipt["refit_fit_end"] != meta["parent_fit_end"]:
                        raise ValueError("Canonical A60 fit boundary differs")
                    if meta["family"].startswith("DIRECT_") and (source["model_sha"] != "NO_FOUNDATION" or manifest["cost_record"]["actual_F_streams"] != 0):
                        raise ValueError("Native F0 contract differs")
                    score_rows = list(csv.DictReader(score_path.open(encoding="utf-8-sig", newline="")))
                    if len(score_rows) != 1:
                        raise ValueError("Exactly one original saved score row required")
                    for key in ("mse", "mae", "n_observed", "n_origins", "num_channels", "horizon", "active_channels"):
                        if float(score_rows[0][key]) != float(manifest["score"][key]):
                            raise ValueError("Original score CSV differs from manifest")
                    with gzip.open(stats, "rt", encoding="utf-8-sig", newline="") as stream:
                        result = audit.audit_prediction_bundle(manifest, reference, manifest["score"], csv.DictReader(stream), validation, list(declared_map), [manifest["cost_record"]], cache)
                    with np.load(archive, allow_pickle=False) as arrays:
                        expected_channels = np.asarray(asset["columns"], dtype=str)
                        bounds = asset["primary_split_ranges"]["V_select"]
                        expected_origins = np.arange(max(bounds[0], asset["L"]), bounds[1] - asset["H"] + 1, asset["H"], dtype=np.int64)
                        if not np.array_equal(arrays["channel_ids"], expected_channels) or not np.array_equal(arrays["origins"], expected_origins):
                            raise ValueError("Array channel names/order or V origin coordinates differ from metadata contract")
                    key = (identity, manifest["split"], manifest["cohort"])
                    if key in prior_keys or key in new_keys or manifest["cache_relative_path"] in prior_cache_paths | new_cache_paths:
                        raise ValueError("Bundle identity/cache path overlap")
                    new_keys.add(key)
                    new_cache_paths.add(manifest["cache_relative_path"])
                    result.update(dataset=dataset, family=meta["family"], fold=meta["fold"], training_status=receipt["training_status"], selected_epoch=receipt["selected_epoch"],
                        reported_mse=manifest["score"]["mse"], reported_mae=manifest["score"]["mae"], mse_difference=result["mse"] - manifest["score"]["mse"], mae_difference=result["mae"] - manifest["score"]["mae"],
                        manifest_public_relative_path=path.relative_to(public).as_posix(), manifest_sha256=sha(path), reference_sha256=sha(path.with_name("REFERENCE_CONTRACT.json")),
                        archive_sha256=sha(archive), selected_artifact_sha256=sha(artifact), artifact_deserialized=False, actual_F_streams=manifest["cost_record"]["actual_F_streams"],
                        scaler_sha=manifest["scaler_sha"], data_sha=manifest["data_sha"], model_sha=manifest["model_sha"], original_registry_execution_status=row["execution_status"])
                    results.append(result)
                    rows.append({name: result[name] for name in ("run_id", "dataset", "family", "fold", "status", "training_status", "selected_epoch", "reported_mse", "mse", "mse_difference", "reported_mae", "mae", "mae_difference", "n_observed", "n_origins", "num_channels", "horizon", "active_channels", "n_statistics", "actual_F_streams", "archive_sha256", "selected_artifact_sha256")})
                    print(json.dumps({"run_id": identity, "status": "PASS_SAVED_V_FP64", "mse": result["mse"], "mae": result["mae"]}), flush=True)
                except Exception as error:
                    errors.append({"run_id": identity, "status": "FAILED", "error_type": type(error).__name__, "reason": sanitize(error)})
                    new(private / (identity + "_FAILURE.json"), {"error": str(error), "traceback": traceback.format_exc()})
                    print(json.dumps({"run_id": identity, "status": "FAILED", "error_type": type(error).__name__}), flush=True)
                max_rss = max(max_rss, process.memory_info().rss)
            pools = threadpool_info()
        after = {}
        for identity, expected in inputs.items():
            scope, relative = identity.split(":", 1)
            path = (public if scope == "public" else cache) / relative
            after[identity] = {"sha256": sha(path), "bytes": path.stat().st_size}
        changed_inputs = [key for key in inputs if inputs[key] != after[key]]
        sources_after = {name: sha(public / name) for name in sources_before}
        registry_after = list(csv.DictReader(registry_path.open(encoding="utf-8-sig", newline="")))
        registry_after_map = {row["experiment_id"]: row for row in registry_after}
        scientific_changes = [identity for identity, row in by_id.items() if identity not in registry_after_map or any(registry_after_map[identity][key] != value for key, value in row.items() if key not in mutable)]
        mutable_changes = [identity for identity, row in by_id.items() if identity in registry_after_map and any(registry_after_map[identity][key] != row[key] for key in mutable)]
        all_pass = len(results) == 13 and not errors and not changed_inputs and sources_before == sources_after and not scientific_changes and not forbidden and "torch" not in sys.modules
        summary.update(status="PASS_13_SAVED_V_FP64_BUNDLES" if all_pass else "PARTIAL_OR_FAILED_DELTA_AUDIT", actual_attempted=13, passed=len(results), failures=errors,
            previous_unique_saved_bundles=len(prior_keys), previous_unique_V=sum(key[1] == "V_select" for key in prior_keys), previous_unique_dense_M=sum(key[1] == "M_cal" for key in prior_keys),
            new_unique_saved_V=len(new_keys), previous_new_overlap=0, union_unique_saved_bundles=len(prior_keys | new_keys), union_unique_V=sum(key[1] == "V_select" for key in prior_keys | new_keys),
            unique_scope_definition="Distinct run/split/cohort and distinct saved archive path; not independent weights, statistical replicates, or generalization trials",
            source33_plus_dependencies_count=len(sources_before), sources_before=sources_before, sources_after=sources_after, sources_unchanged=sources_before == sources_after,
            immutable_input_files=len(inputs), input_files_changed=changed_inputs, prediction_NPZs_opened_read_only=13, selected_artifacts_stream_hashed_only=13,
            registry_scientific_changes=scientific_changes, registry_mutable_status_changes=mutable_changes, concurrently_added_registry_ids=sorted(set(registry_after_map) - set(by_id)),
            registry_before_sha256=hashlib.sha256(registry_bytes).hexdigest(), registry_after_sha256=sha(registry_path), validation_only_fallback_reason_count=len(fallback_ids),
            CPU_wall_seconds=time.time() - started, CPU_process_seconds=process.cpu_times().user + process.cpu_times().system,
            RSS_peak_bytes=getattr(process.memory_info(), "peak_wset", max_rss), RSS_peak_scope="Windows peak working set when available; otherwise largest observed process RSS at route boundaries", threadpools=pools,
            forbidden_io_attempts=forbidden, torch_imported="torch" in sys.modules, CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
            completion_scope="Only 13 saved V FP64 rescoring and source/metadata/archive byte checks; no replay, timing rerun, training convergence, T or whole campaign completion",
            reference_limit="Original pre-prediction reference contracts and original cohort fingerprints reused; raw Task/target/scaler regeneration was not performed")
        new(output / "INPUTS_SHA256_BEFORE.json", inputs)
        new(output / "INPUTS_SHA256_AFTER.json", after)
        new(output / "AUDIT_RESULTS.json", {"records": results, "failures": errors})
        with (output / "SAVED_V_RESCORING.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["run_id", "status"])
            writer.writeheader()
            writer.writerows(rows)
    except Exception as error:
        summary.update(status="DRIVER_FAILED_PRESERVED", error_type=type(error).__name__, reason=sanitize(error), failures=errors, actual_passes_before_driver_failure=len(results))
        new(private / "DRIVER_FAILURE.json", {"error": str(error), "traceback": traceback.format_exc()})
    summary.update(finished_unix=time.time(), reproducible_driver_sha256=sha(__file__))
    new(output / "RESULT.json", summary)
    print(json.dumps({"status": summary["status"], "passed": summary.get("passed", len(results)), "wall_seconds": time.time() - started}), flush=True)
    return 0 if summary["status"] == "PASS_13_SAVED_V_FP64_BUNDLES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
