"""Independent saved E2 V and new parent dense-M audit; terminal trigger required."""
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

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
PLAN_REL = "execution_batches/extended_parent_e2_plan_1792080455542007200.json"
PLAN_SHA = "82c5038eccb9164b641b12175de8eba05b917f0835e8a2709f354569ca005467"
PREVIOUS = "diagnostics/retry1_saved_V_rescore_1792088736841"
MUTABLE = {"execution_status", "training_status", "validity_status", "scientific_status", "publication_status", "reason", "resume_command"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_new(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public-root", "cache-root", "output", "execution-receipt", "execution-sha", "terminal-evidence"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    public, cache, output = (Path(value).resolve() for value in (args.public_root, args.cache_root, args.output))
    if not output.is_relative_to(public / "diagnostics"):
        raise ValueError("New audit output must be under public diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    new(output / "RUN_CLAIM.json", dict(started_unix=time.time(), overwrite_allowed=False, terminal_evidence=args.terminal_evidence))
    started = time.time()
    inputs, headers, parents, routes, errors, results, alias_results = {}, {}, {}, {}, [], [], []
    forbidden = []
    summary = dict(status="NOT_COMPLETED", checkpoint_replay=False, training_rerun=False, T_accessed=False, GPU_lease_acquired=False,
                   model_checkpoint_deserialized=False, registry_written=False, global_scores_checked=False, Git_calls=0)

    def inside(root, relative):
        relative = Path(relative)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or path == root:
            raise ValueError("Escaped declared input")
        return path

    def record(path):
        path = Path(path).resolve()
        root, scope = (public, "public") if path.is_relative_to(public) else (cache, "private")
        if not path.is_relative_to(root):
            raise ValueError("Input outside supplied roots")
        key = scope + ":" + path.relative_to(root).as_posix()
        value = dict(sha256=sha(path), bytes=path.stat().st_size)
        if key in inputs and inputs[key] != value:
            raise ValueError("Input changed within capture")
        inputs[key] = value
        return path

    def read(path):
        return json.loads(record(path).read_text(encoding="utf-8-sig"))

    def error_text(error):
        text = str(error)
        for path, label in ((cache, "PRIVATE_CACHE"), (public, "PUBLIC_CAMPAIGN")):
            text = text.replace(str(path), label).replace(path.as_posix(), label)
        return text

    try:
        if sha(public / PLAN_REL) != PLAN_SHA:
            raise ValueError("Sealed E2 plan changed")
        plan = read(public / PLAN_REL)
        execution_path = inside(public, args.execution_receipt)
        if sha(execution_path) != args.execution_sha:
            raise ValueError("Terminal execution receipt SHA changed")
        execution = read(execution_path)
        if not execution.get("finished_unix") or execution.get("status") == "EXECUTION_IN_PROGRESS":
            raise ValueError("E2 execution receipt is not terminal")
        previous_result = read(public / PREVIOUS / "RESULT.json")
        sources_before = {name: sha(public / name) for name in previous_result["source39_after"]}
        if len(sources_before) != 39 or sources_before != previous_result["source39_after"]:
            raise ValueError("Frozen source39 differs from prior independent audit")
        if any(sources_before[name] != digest for name, digest in plan["source_pins"].items()):
            raise ValueError("Sealed E2 source pins differ")
        old_union_path = record(public / PREVIOUS / "UNION_BUNDLES.csv")
        old_union = list(csv.DictReader(old_union_path.open(encoding="utf-8-sig", newline="")))
        old_keys = {(row["run_id"], row["split"], row["cohort"]) for row in old_union}
        old_paths = {row["archive_cache_relative_path"] for row in old_union}
        if len(old_union) != 409 or len(old_keys) != 409 or len(old_paths) != 409:
            raise ValueError("Prior409 bundle union identity differs")
        assets = read(public / "DATA_CONTRACT.json")["assets"]
        registry_path = public / "EXPERIMENT_REGISTRY.csv"
        registry_bytes = registry_path.read_bytes()
        registry = list(csv.DictReader(io.StringIO(registry_bytes.decode("utf-8-sig"))))
        by_id = {row["experiment_id"]: row for row in registry}
        if len(by_id) != len(registry):
            raise ValueError("Duplicate registry identity")
        proposed = {row["experiment_id"]: row for row in plan["proposed_registry_rows"]}
        if len(proposed) != 28 or plan["new_fit_count"] != 16 or plan["new_output_rows"] != 28:
            raise ValueError("Sealed16 fit/28 output contract differs")
        for identity, expected in proposed.items():
            if identity not in by_id or any(by_id[identity][key] != value for key, value in expected.items() if key not in MUTABLE):
                raise ValueError("E2 registered scientific identity differs")
        expected_ids = list(by_id)
        allowed_npz, allowed_weights = set(), set()

        def capture_manifest(path):
            manifest, reference = read(path), read(path.with_name("REFERENCE_CONTRACT.json"))
            archive = record(inside(cache, manifest["cache_relative_path"]))
            if sha(archive) != manifest["archive_sha256"]:
                raise ValueError("Prediction archive bytes differ")
            allowed_npz.add(archive)
            if manifest.get("public_stats_relative_path"):
                record(inside(public, manifest["public_stats_relative_path"]))
                record(path.with_name("SCORES.csv"))
            return dict(path=path, manifest=manifest, reference=reference, archive=archive)

        for item in plan["datasets"]:
            dataset = item["dataset"]
            for family, parent in item["parents"].items():
                identity = parent["run_id"]
                if identity in parents:
                    continue
                receipt_path = inside(public, parent["receipt_relative_path"])
                source_path = inside(public, parent["source_relative_path"])
                receipt, source = read(receipt_path), read(source_path)
                artifact = record(inside(cache, parent["artifact_cache_relative_path"]))
                allowed_weights.add(artifact)
                if sha(receipt_path) != parent["receipt_sha256"] or sha(source_path) != parent["source_contract_sha256"] or sha(artifact) != parent["artifact_sha256"]:
                    raise ValueError("Sealed parent artifact/source/receipt differs")
                if receipt["status"] != "COMPLETE_VALID" or source["files"] != plan["source_six"] or source["source_sha"] != fingerprint(plan["source_six"]):
                    raise ValueError("Parent fit/source contract differs")
                if receipt["refit_fit_end"] != assets[dataset]["primary_split_ranges"]["M_cal"][0] or source["data_sha"] != assets[dataset]["data_sha256"]:
                    raise ValueError("Parent fit boundary/data differs")
                pv = capture_manifest(public / "predictions" / identity / "V_select/PREDICTION_MANIFEST.json")
                pm = capture_manifest(public / "predictions" / identity / "M_cal_dense/PREDICTION_MANIFEST.json")
                for header in (pv, pm):
                    manifest = header["manifest"]
                    if manifest["model_sha"] != pv["manifest"]["model_sha"] or manifest["scaler_sha"] != parent["refit_scaler_sha"] or manifest["data_sha"] != parent["data_sha"] or manifest["metadata"]["artifact_sha256"] != parent["artifact_sha256"]:
                        raise ValueError("Parent M/V model/artifact/scaler identity differs")
                parents[identity] = dict(dataset=dataset, family=family, receipt=receipt, source=source, artifact=artifact, V_select=pv, M_cal=pm)
            for route in item["routes"]:
                identity = route["spec"]["run_id"]
                receipt = read(public / "combinations" / identity / "receipt.json")
                spec = read(cache / "combinations" / identity / "spec.json")
                if receipt["execution_status"] != "COMPLETE_VALID" or spec != route["spec"] or receipt["run_id"] != identity or receipt["variant"] != spec["variant"]:
                    raise ValueError("E2 completed exact-spec fit differs")
                if receipt["contract"]["source_sha"] != fingerprint({name: sources_before[name] for name in ("combination.py", "networks.py", "training.py", "evaluation.py", "data_contract.py")}):
                    raise ValueError("E2 fit source identity differs")
                if receipt["contract"]["config_sha"] != fingerprint({key: value for key, value in spec.items() if key not in {"run_id", "resume_checkpoint"}}):
                    raise ValueError("E2 scientific spec hash differs")
                if spec["variant"] == "alpha_family":
                    if receipt["updates"] != 0 or receipt["training_status"] != "NO_TRAINING":
                        raise ValueError("Analytic alpha family incorrectly counted as neural training")
                else:
                    artifact = record(cache / "combinations" / identity / "selected.pt")
                    allowed_weights.add(artifact)
                    if sha(artifact) != receipt["selected_checkpoint_sha256"] or receipt["updates"] <= 0:
                        raise ValueError("Selected neural head artifact/update contract differs")
                routes[identity] = dict(dataset=dataset, route=route, receipt=receipt, spec=spec)
                for row in route["registry_rows"]:
                    run = row["experiment_id"]
                    header = capture_manifest(public / "predictions" / run / "V_select/PREDICTION_MANIFEST.json")
                    record(inside(public, header["reference"]["pre_target_contract_public_relative_path"]))
                    header.update(dataset=dataset, family_run_id=identity)
                    headers[run] = header
        if len(headers) != 28 or len(routes) != 16 or len(parents) != 12:
            raise ValueError("Actual exact E2/parent manifest coverage differs")
        audit_ids = set(headers) | set(parents)
        validation = [dict(run_id=identity, execution_status="COMPLETE_VALID" if identity in audit_ids else row["execution_status"],
                           reason=row.get("reason") or "No targeted completed artifact in this delta audit", original_execution_status=row["execution_status"],
                           projection_scope="Saved artifact validation only; not campaign completion or convergence") for identity, row in by_id.items()]
        new(output / "VALIDATION_REGISTRY_PROJECTION.json", validation)
        new(output / "INPUTS_BEFORE.json", dict(source39=sources_before, inputs=inputs, registry_sha256=hashlib.sha256(registry_bytes).hexdigest()))

        def guard(event, values):
            if event == "import" and str(values[0]).split(".")[0] in {"torch", "chronos"}:
                forbidden.append("MODEL_IMPORT");raise RuntimeError("No model runtime import")
            if event == "pickle.find_class":
                forbidden.append("PICKLE_DESERIALIZATION");raise RuntimeError("No checkpoint deserialization")
            if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(values[0])).resolve()
            if "/t_final/" in path.as_posix().casefold() or path.name.casefold() in {"gpu_owner.lock", "active_gpu_lease.json"}:
                forbidden.append("T_OR_GPU_OWNER");raise RuntimeError("No T or GPU ownership access")
            if path.suffix.casefold() == ".npz" and path not in allowed_npz:
                forbidden.append("OTHER_NPZ");raise RuntimeError("Only E2 outputs and exact parent prediction NPZs allowed")
            if path.suffix.casefold() in {".pt", ".safetensors"} and path not in allowed_weights:
                forbidden.append("OTHER_CHECKPOINT");raise RuntimeError("Only selected bound artifact bytes may be SHA-read")
            flags = values[2] if len(values) > 2 and isinstance(values[2], int) else 0
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) and not path.is_relative_to(output):
                forbidden.append("WRITE_OUTSIDE_NEW_DIAGNOSTIC");raise RuntimeError("Audit writes confined to new output")
        sys.addaudithook(guard)
        sys.path.insert(0, str(public))
        import audit_predictions as audit
        import numpy as np
        import psutil
        from threadpoolctl import threadpool_limits
        process = psutil.Process()

        def arrays_for(header):
            arrays = audit._load_arrays(header["manifest"], header["reference"], cache)
            dataset = header.get("dataset") or header["manifest"]["cohort"].split(":")[0]
            asset = assets[dataset]
            split = header["manifest"]["split"]
            bounds = asset["primary_split_ranges"][split]
            stride = 1 if split == "M_cal" else asset["H"]
            expected_origins = np.arange(max(bounds[0], asset["L"]), bounds[1] - asset["H"] + 1, stride, dtype=np.int64)
            if not np.array_equal(arrays["origins"], expected_origins) or not np.array_equal(arrays["channel_ids"], np.asarray(asset["columns"], dtype=str)):
                raise ValueError("Actual original channel order/origin contract differs")
            return arrays

        def boundary():
            if time.time() - started > 600 or process.memory_info().rss > 2 * 1024**3:
                raise RuntimeError("CPU audit wall600s or RSS2GiB boundary")

        new_bundle_headers = []
        with threadpool_limits(limits=2):
            for identity, header in headers.items():
                boundary()
                try:
                    manifest, reference = header["manifest"], header["reference"]
                    family = routes[header["family_run_id"]]
                    receipt, spec = family["receipt"], family["spec"]
                    pre_path = inside(public, reference["pre_target_contract_public_relative_path"])
                    pre = read(pre_path)
                    if sha(pre_path) != reference["pre_target_contract_sha256"] or pre["arrays"] != reference["arrays"]:
                        raise ValueError("Pre-fit original truth seal changed")
                    for key in ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "scaler_sha"):
                        if pre[key] != reference[key]:
                            raise ValueError("Pre-fit identity differs")
                    if any(receipt["contract"].get(key) != value for key, value in pre["expected_receipt_contract"].items()):
                        raise ValueError("Realized fit contract differs from pre-fit seal")
                    realized = reference["realized_model_binding"]
                    if realized != manifest["metadata"]["realized_model_binding"] or fingerprint(realized) != manifest["model_sha"]:
                        raise ValueError("Realized model hash differs")
                    for role, parent_id in (("b", spec["parent_run_id"]), ("g", spec["g_run_id"])):
                        anchor, parent = pre["parent_anchors"][role], parents[parent_id]
                        pv = parent["V_select"]["manifest"]
                        if anchor["run_id"] != parent_id or anchor["model_sha"] != pv["model_sha"] or sha(public / anchor["manifest_public_relative_path"]) != anchor["manifest_sha256"]:
                            raise ValueError("Frozen deployment parent differs")
                        for suffix, split in (("m", "M_cal"), ("v", "V_select")):
                            key = ("parent_" if role == "b" else "g_") + suffix
                            cached = receipt["parent_models"][key]
                            pm = parent[split]["manifest"]
                            for field in ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "model_sha", "scaler_sha"):
                                if cached[field] != pm[field]:
                                    raise ValueError("Original M/V parent metadata differs")
                            actual_fp = fingerprint(dict(arrays=pm["arrays"], metadata={key0: value for key0, value in cached.items() if key0 != "reused_prediction_cache"}))
                            if receipt["parent_cache_fingerprints"][key] != actual_fp:
                                raise ValueError("Original parent cache fingerprint differs")
                    if fingerprint(receipt["parent_cache_fingerprints"]) != receipt["contract"]["parents_sha"]:
                        raise ValueError("Fit parent cache hash differs")
                    asset = assets[header["dataset"]]
                    if manifest["cohort"] != header["dataset"] + ":primary:V_select:stride" + str(asset["H"]) or manifest["data_sha"] != asset["data_sha256"] or manifest["scaler_sha"] != parents[spec["g_run_id"]]["V_select"]["manifest"]["scaler_sha"]:
                        raise ValueError("E2 dataset/scaler/cohort differs")
                    with gzip.open(public / manifest["public_stats_relative_path"], "rt", encoding="utf-8-sig", newline="") as stream:
                        result = audit.audit_prediction_bundle(manifest, reference, manifest["score"], csv.DictReader(stream), validation, expected_ids, [manifest["cost_record"]], cache)
                    arrays = arrays_for(header)
                    label = manifest["metadata"]["output_label"]
                    if spec["variant"] == "alpha_family":
                        g, b = arrays_for(parents[spec["g_run_id"]]["V_select"]), arrays_for(parents[spec["parent_run_id"]]["V_select"])
                        alpha = realized["alpha"]
                        expected = (g["pred_normalized"].astype(np.float64) + alpha * (b["pred_normalized"].astype(np.float64) - g["pred_normalized"])).astype(np.float32)
                        if not np.array_equal(arrays["pred_normalized"], expected):
                            raise ValueError("Actual alpha output differs from frozen-parent FP64 formula")
                        if label in {"alpha0", "alpha1"}:
                            role = "g" if label == "alpha0" else "b"
                            parent_arrays = g if role == "g" else b
                            same = all(np.array_equal(arrays[key], parent_arrays[key]) for key in audit.ARRAY_KEYS)
                            if not same or set(realized["parent_model_shas"]) != {role} or realized["selected_artifact_sha256"] is not None:
                                raise ValueError("Endpoint alias differs from actual active parent arrays/calculation path")
                            alias_results.append(dict(run_id=identity, parent_run_id=spec["g_run_id"] if role == "g" else spec["parent_run_id"], status="PASS_EXACT_ARRAYS_AND_ACTIVE_PARENT_IDENTITY", alpha=alpha,
                                original_model_sha=manifest["model_sha"], active_parent_model_sha=realized["parent_model_shas"][role], wrapper_model_sha_is_distinct_identity=True,
                                same_archive_bytes=manifest["archive_sha256"] == parents[spec["g_run_id"] if role == "g" else spec["parent_run_id"]]["V_select"]["manifest"]["archive_sha256"], independent_fit=False))
                    score_rows = list(csv.DictReader(header["path"].with_name("SCORES.csv").open(encoding="utf-8-sig", newline="")))
                    if len(score_rows) != 1 or any(float(score_rows[0][key]) != float(result[key]) for key in ("mse", "mae", "n_observed", "n_origins", "num_channels", "horizon", "active_channels")):
                        raise ValueError("Original score CSV differs from FP64 rescore")
                    result.update(reported_mse=manifest["score"]["mse"], reported_mae=manifest["score"]["mae"], mse_difference=result["mse"]-manifest["score"]["mse"], mae_difference=result["mae"]-manifest["score"]["mae"],
                        family_run_id=header["family_run_id"], variant=spec["variant"], output_label=label, updates=receipt["updates"], training_status=receipt["training_status"],
                        manifest_public_relative_path=header["path"].relative_to(public).as_posix(), manifest_sha256=sha(header["path"]), archive_cache_relative_path=manifest["cache_relative_path"], archive_sha256=manifest["archive_sha256"],
                        selected_head_artifact_sha256=realized["selected_artifact_sha256"], actual_F_streams=manifest["cost_record"]["actual_F_streams"], cached_parent_output_is_not_deployment_cost=True)
                    results.append(result)
                    new_bundle_headers.append(header)
                    del arrays
                    print(json.dumps(dict(run_id=identity, split="V_select", status=result["status"])), flush=True)
                except Exception as error:
                    errors.append(dict(run_id=identity, scope="standard_V", error_type=type(error).__name__, error=error_text(error)))
            for identity, parent in parents.items():
                boundary()
                header = parent["M_cal"]
                manifest = header["manifest"]
                key = (identity, manifest["split"], manifest["cohort"])
                if key in old_keys:
                    continue
                try:
                    reference = header["reference"]
                    if manifest["score"] is not None or manifest["public_stats_relative_path"] is not None or manifest["metadata"].get("dense") is not True:
                        raise ValueError("Dense parent cache contract differs")
                    if any(manifest[field] != reference[field] for field in audit.IDENTITY_KEYS):
                        raise ValueError("Dense original reference identity differs")
                    audit._check_registry(validation, expected_ids, identity)
                    audit._check_cost([manifest["cost_record"]], identity, reference)
                    arrays = arrays_for(header)
                    shape = arrays["pred_normalized"].shape
                    if len(shape) != 3 or arrays["target_normalized"].shape != shape or arrays["observed"].shape != shape or arrays["observed"].dtype != np.bool_:
                        raise ValueError("Dense prediction/target/boolean-mask dimensions differ")
                    mse, mae, count, active = [], [], 0, 0
                    for channel in range(arrays["pred_normalized"].shape[2]):
                        observed = arrays["observed"][:, :, channel]
                        n = int(observed.sum())
                        if n:
                            delta = arrays["pred_normalized"][:, :, channel][observed].astype(np.float64) - arrays["target_normalized"][:, :, channel][observed].astype(np.float64)
                            if not np.isfinite(delta).all():
                                raise ValueError("Dense observed nonfinite error")
                            mse.append(float(np.dot(delta, delta)) / n);mae.append(float(np.abs(delta).sum(dtype=np.float64)) / n)
                            count += n;active += 1
                    result = dict(status="PASS_DENSE_PRIVATE_RESCORING", run_id=identity, split="M_cal", cohort=manifest["cohort"], mse=float(np.mean(mse)) if mse else None, mae=float(np.mean(mae)) if mae else None,
                        n_observed=count, active_channels=active, n_origins=int(arrays["pred_normalized"].shape[0]), horizon=int(arrays["pred_normalized"].shape[1]), num_channels=int(arrays["pred_normalized"].shape[2]),
                        public_stats_reaggregation=False, prediction_rescoring=True, public_stats_missing_by_original_dense_contract=True, checkpoint_replay=False, training_rerun=False,
                        manifest_public_relative_path=header["path"].relative_to(public).as_posix(), manifest_sha256=sha(header["path"]), archive_cache_relative_path=manifest["cache_relative_path"], archive_sha256=manifest["archive_sha256"],
                        training_status=parent["receipt"]["training_status"], actual_F_streams=manifest["cost_record"]["actual_F_streams"])
                    results.append(result);new_bundle_headers.append(header);del arrays
                    print(json.dumps(dict(run_id=identity, split="M_cal", status=result["status"])), flush=True)
                except Exception as error:
                    errors.append(dict(run_id=identity, scope="parent_dense_M", error_type=type(error).__name__, error=error_text(error)))
        inputs_after = {}
        for key, before in inputs.items():
            scope, relative = key.split(":", 1)
            target = (public if scope == "public" else cache) / relative
            inputs_after[key] = dict(sha256=sha(target), bytes=target.stat().st_size)
        source_after = {name: sha(public / name) for name in sources_before}
        after_rows = list(csv.DictReader(registry_path.open(encoding="utf-8-sig", newline="")))
        after_map = {row["experiment_id"]: row for row in after_rows}
        scientific_changes = [identity for identity, row in by_id.items() if identity not in after_map or any(after_map[identity][key] != value for key, value in row.items() if key not in MUTABLE)]
        mutable_changes = [identity for identity, row in by_id.items() if identity in after_map and any(after_map[identity][key] != row[key] for key in MUTABLE)]
        new_union, duplicates = [], []
        content = {}
        for row in old_union:
            content.setdefault(row["archive_sha256_as_declared"], []).append(row["run_id"] + ":" + row["split"])
        for header in new_bundle_headers:
            manifest = header["manifest"]
            key = (manifest["run_id"], manifest["split"], manifest["cohort"])
            if key in old_keys or manifest["cache_relative_path"] in old_paths:
                raise ValueError("New scope overlaps a previously audited bundle")
            row = dict(run_id=manifest["run_id"], split=manifest["split"], cohort=manifest["cohort"], manifest_public_relative_path=header["path"].relative_to(public).as_posix(),
                manifest_sha256=sha(header["path"]), archive_cache_relative_path=manifest["cache_relative_path"], archive_sha256_as_declared=manifest["archive_sha256"], archive_exists=True, archive_bytes=header["archive"].stat().st_size)
            if manifest["archive_sha256"] in content:
                duplicates.append(dict(run_id=manifest["run_id"], split=manifest["split"], archive_sha256=manifest["archive_sha256"], same_byte_prior_identities=content[manifest["archive_sha256"]], distinct_identity_not_independent_replication=True))
            content.setdefault(manifest["archive_sha256"], []).append(manifest["run_id"] + ":" + manifest["split"])
            new_union.append(row)
        union = old_union + new_union
        if len({(row["run_id"], row["split"], row["cohort"]) for row in union}) != len(union) or len({row["archive_cache_relative_path"] for row in union}) != len(union):
            raise ValueError("Union identity/archive-path duplicate")
        sources_pass = sources_before == source_after
        inputs_pass = inputs == inputs_after
        passed_v = sum(row["split"] == "V_select" for row in results)
        passed_m = sum(row["split"] == "M_cal" for row in results)
        complete = passed_v == 28 and passed_m == 12 and len(alias_results) == 8 and not errors and sources_pass and inputs_pass and not scientific_changes and not forbidden and "torch" not in sys.modules
        new(output / "INPUTS_AFTER.json", dict(source39=source_after, inputs=inputs_after))
        new(output / "AUDIT_RESULTS.json", dict(records=results, errors=errors, aliases=alias_results, duplicate_archive_content=duplicates))
        fields = sorted(set().union(*(row.keys() for row in union)))
        csv_new(output / "UNION_BUNDLES.csv", union, fields)
        summary.update(status="PASS_SAVED_E2_AND_NEW_PARENT_M" if complete else "PARTIAL_OR_FAILED", previous_unique_saved_bundles=409, previous409_rerescored=False,
            actual_new_V_rescores=passed_v, actual_new_dense_M_rescores=passed_m, actual_new_rescores=len(results), union_unique_saved_bundles=len(union),
            union_unique_V=sum(row["split"] == "V_select" for row in union), union_unique_dense_M=sum(row["split"] == "M_cal" for row in union), duplicate_content_count=len(duplicates),
            exact_alias_checks=len(alias_results), neural_head_fits=12, analytic_alpha_family_fits=4, alpha_output_conditions=16, alpha_family_updates0=True,
            sources_before=sources_before, sources_after=source_after, source39_unchanged=sources_pass, immutable_input_files=len(inputs), input_hashes_unchanged=inputs_pass,
            registry_before_sha256=hashlib.sha256(registry_bytes).hexdigest(), registry_after_sha256=sha(registry_path), registry_scientific_changes=scientific_changes, registry_mutable_status_changes=mutable_changes,
            concurrently_added_registry_ids=sorted(set(after_map)-set(by_id)), torch_imported="torch" in sys.modules, forbidden_io_attempts=forbidden,
            CPU_wall_seconds=time.time()-started, CPU_process_seconds=process.cpu_times().user+process.cpu_times().system, RSS_peak_bytes=getattr(process.memory_info(), "peak_wset", process.memory_info().rss),
            CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"], numerical_scope="Saved arrays only; no checkpoint replay, training rerun, convergence or independent generalization claim", terminal_execution_receipt=args.execution_receipt, terminal_execution_sha256=args.execution_sha,
            errors=errors, actual_new_raw_conditioned_H_included=False)
        csv_new(output / "SAVED_BUNDLE_RESCORING.csv", results, ("run_id", "split", "cohort", "status", "reported_mse", "mse", "mse_difference", "reported_mae", "mae", "mae_difference", "n_observed", "n_origins", "num_channels", "horizon", "active_channels", "variant", "output_label", "training_status", "updates", "n_statistics", "actual_F_streams", "manifest_sha256", "archive_sha256", "selected_head_artifact_sha256"))
    except Exception as error:
        summary.update(status="FAILED_PRESERVED_DIAGNOSTIC", fatal_error_type=type(error).__name__, fatal_error=error_text(error), actual_partial_results=results, errors=errors, forbidden_io_attempts=forbidden)
    summary.update(finished_unix=time.time(), reproducible_driver_sha256=sha(__file__))
    new(output / "RESULT.json", summary)
    print(json.dumps(dict(status=summary["status"], V=summary.get("actual_new_V_rescores"), M=summary.get("actual_new_dense_M_rescores"), wall_seconds=time.time()-started)), flush=True)
    if summary["status"] != "PASS_SAVED_E2_AND_NEW_PARENT_M":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
