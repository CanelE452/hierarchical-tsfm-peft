"""Read-only FP64 audit of four Raw+deepG V/M outputs and matched-G comparisons."""
import argparse
import copy
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
OLD = "diagnostics/extended_e2_saved_rescore_v2_1792090317000"
READY = "diagnostics/raw_deep_g_control_ready_1792090190000000000_v2"
EXECUTION = "diagnostics/raw_deep_g_control_execution_1792090473924518100/RESULT.json"
MUTABLE = {"execution_status", "training_status", "validity_status", "scientific_status", "publication_status", "reason", "resume_command"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_new(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("public-root", "cache-root", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    public, cache, output = (Path(value).resolve() for value in (args.public_root, args.cache_root, args.output))
    if not output.is_relative_to(public / "diagnostics"):
        raise ValueError("Output outside new diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    new(output / "RUN_CLAIM.json", dict(started_unix=started, overwrite_allowed=False, actual_training_GPU_T0=True))
    inputs, results, errors, headers, comparisons, forbidden = {}, [], [], [], [], []
    summary = dict(status="NOT_COMPLETED", model_deserialized=False, checkpoint_replay=False, training_rerun=False, T_accessed=False, GPU_lease_acquired=False, registry_written=False, global_scores_written=False, Git_calls=0)

    def record(path):
        path = Path(path).resolve()
        root, scope = (public, "public") if path.is_relative_to(public) else (cache, "private")
        if not path.is_relative_to(root):
            raise ValueError("Input outside roots")
        key = scope + ":" + path.relative_to(root).as_posix()
        value = dict(sha256=sha(path), bytes=path.stat().st_size)
        if key in inputs and inputs[key] != value:
            raise ValueError("Input changed within capture")
        inputs[key] = value
        return path

    def read(path):
        return json.loads(record(path).read_text(encoding="utf-8-sig"))

    def pinned(path, digest):
        if sha(path) != digest:
            raise ValueError("Frozen input SHA differs: " + Path(path).name)
        return read(path)

    def relative(root, value):
        name = Path(value)
        path = (root / name).resolve()
        if name.is_absolute() or not path.is_relative_to(root) or path == root:
            raise ValueError("Escaped relative artifact")
        return path

    try:
        execution = pinned(public / EXECUTION, "96d83c069aec0d35e5c6511702c980c77547e67366388fcf92f8bf83022ef54f")
        plan = pinned(public / READY / "PLAN.json", "1082332ac3ad92ec13946bd1c0b472843c34e9a592debdDC54f6885c343be91f".lower())
        freeze = pinned(public / READY / "SOURCE_FREEZE.json", "6e516b44645fa87a380df9ec46b26f5bcddcc23a9eb87aad94d425ba8e69d4dc")
        prior = pinned(public / OLD / "RESULT.json", "28ed9b1abcfcf1cedf9d53e9b05b5bdc046d06b45c69c71f845a77d302e3ef23")
        prior_audit = pinned(public / OLD / "AUDIT_RESULTS_PUBLIC.json", "a11c418eca85bff3e34692a235efd5fb053a9149f2654cf24365226f5e414484")
        sources = {name: sha(public / name) for name in prior["sources_after"]}
        if len(sources) != 39 or sources != prior["sources_after"] or any(sources[name] != digest for name, digest in freeze["source_pins"].items()) or plan["source_pins"] != freeze["source_pins"]:
            raise ValueError("Frozen source33/39 contract differs")
        if execution["status"] != "COMPLETE_FOUR_FITS_AND_V" or len(execution["routes"]) != 4 or execution["plan_signature"] != plan["plan_signature"] or execution["central_lease"]["deadline_overrun_seconds"] != 0:
            raise ValueError("Actual terminal Raw4 execution contract differs")
        old_union = list(csv.DictReader(record(public / OLD / "UNION_BUNDLES.csv").open(encoding="utf-8", newline="")))
        if len(old_union) != 449 or sha(public / OLD / "UNION_BUNDLES.csv") != "3982dc4fbe7f3fc3383e2ab90b766770e9759689a5ad15b0ec7c4f8cd9327447":
            raise ValueError("Existing449 union changed")
        old_keys = {(row["run_id"], row["split"], row["cohort"]) for row in old_union}
        old_paths = {row["archive_cache_relative_path"] for row in old_union}
        old_manifest_shas = {row["manifest_public_relative_path"]: row["manifest_sha256"] for row in old_union}
        manifest449 = read(public / OLD / "FILE_MANIFEST.json")
        for item in manifest449["files"]:
            path = relative(public.parents[1], item["repository_relative_path"])
            if sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("Previous safe449 publication allowlist bytes changed")
        pinned(public / OLD / "PUBLICATION_SCAN.json", "b68eeadd38f093e000a905eb7feb813110fa574ae6f41acb54e315e4568aab2b")
        record(public / OLD / "PUBLICATION_ALLOWLIST.txt")
        if sha(public / OLD / "PUBLICATION_ALLOWLIST.txt") != "38622f44aade86d57193b86de8bbbd2f2e0b5afb3577709246e0768b4f4e56cc":
            raise ValueError("Previous23 safe allowlist changed")
        assets = read(public / "DATA_CONTRACT.json")["assets"]
        registry_path = public / "EXPERIMENT_REGISTRY.csv"
        registry_bytes = registry_path.read_bytes()
        registry = list(csv.DictReader(io.StringIO(registry_bytes.decode("utf-8-sig"))))
        by_id = {row["experiment_id"]: row for row in registry}
        if len(by_id) != len(registry):
            raise ValueError("Duplicate registry identity")
        allowed_npz, parent_hash_npz, allowed_weights, bundles, groups = set(), set(), set(), [], []

        def parent_header(binding):
            manifest = pinned(public / binding["manifest_relative_path"], binding["manifest_sha256"])
            reference = pinned(public / binding["reference_relative_path"], binding["reference_sha256"])
            path = record(relative(cache, manifest["cache_relative_path"]))
            parent_hash_npz.add(path)
            if sha(path) != manifest["archive_sha256"] or manifest["archive_sha256"] != binding["archive_sha256"]:
                raise ValueError("Frozen parent cache bytes differ")
            if binding["manifest_relative_path"] not in old_manifest_shas or old_manifest_shas[binding["manifest_relative_path"]] != binding["manifest_sha256"]:
                raise ValueError("Parent was not included in independently audited449")
            return dict(manifest=manifest, reference=reference, path=path)

        for route, executed in zip(plan["routes"], execution["routes"]):
            spec, dataset = route["spec"], route["dataset"]
            identity = spec["run_id"]
            if executed["run_id"] != identity or executed["status"] != "COMPLETE_VALID":
                raise ValueError("Actual route identity differs")
            expected_row = route["registry_row"]
            if any(by_id[identity][key] != value for key, value in expected_row.items() if key not in MUTABLE):
                raise ValueError("RawH registry science differs")
            receipt = pinned(public / executed["receipt_relative_path"], executed["receipt_sha256"])
            saved_spec = read(cache / "combinations" / identity / "spec.json")
            if saved_spec != spec or receipt["execution_status"] != "COMPLETE_VALID" or receipt["variant"] != "conditional" or receipt["updates"] <= 0:
                raise ValueError("RawH exact completed fit spec differs")
            selected = record(cache / "combinations" / identity / "selected.pt")
            allowed_weights.add(selected)
            if sha(selected) != receipt["selected_checkpoint_sha256"]:
                raise ValueError("Selected head byte SHA differs")
            if receipt["contract"]["source_sha"] != fingerprint({name: sources[name] for name in ("combination.py", "networks.py", "training.py", "evaluation.py", "data_contract.py")}):
                raise ValueError("H source identity differs")
            if receipt["contract"]["config_sha"] != fingerprint({key: value for key, value in spec.items() if key not in {"run_id", "resume_checkpoint"}}):
                raise ValueError("H spec fingerprint differs")
            parent = {"b": {split: parent_header(binding) for split, binding in route["raw_predictions"].items()}, "g": {split: parent_header(binding) for split, binding in route["g_predictions"].items()}}
            g_parent = route["item"]["parents"]["DIRECT_G"]
            g_receipt = pinned(public / g_parent["receipt_relative_path"], g_parent["receipt_sha256"])
            g_source = pinned(public / g_parent["source_relative_path"], g_parent["source_contract_sha256"])
            g_artifact = record(relative(cache, g_parent["artifact_cache_relative_path"]))
            allowed_weights.add(g_artifact)
            if sha(g_artifact) != g_parent["artifact_sha256"] or g_source["files"] != {name: sources[name] for name in g_source["files"]}:
                raise ValueError("DeepG parent artifact/source differs")
            if g_receipt["refit_fit_end"] != assets[dataset]["primary_split_ranges"]["M_cal"][0] or g_source["spec"]["run_id"] != spec["g_run_id"] or g_source["spec"]["family"] != "DIRECT_G":
                raise ValueError("DeepG is not the same A-only frozen parent")
            for role, expected_id in (("g", spec["g_run_id"]), ("b", spec["parent_run_id"])):
                for suffix, split in (("m", "M_cal"), ("v", "V_select")):
                    key = ("g_" if role == "g" else "parent_") + suffix
                    m, ref = parent[role][split]["manifest"], parent[role][split]["reference"]
                    metadata = receipt["parent_models"][key]
                    if m["run_id"] != expected_id or any(m[field] != ref[field] or metadata[field] != m[field] for field in ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "model_sha", "scaler_sha")):
                        raise ValueError("Raw/G M/V parent identity differs")
                    if metadata["parent_fit_end"] != g_receipt["refit_fit_end"] or metadata["fold"] != "primary" or m["scaler_sha"] != g_receipt["refit_scaler_sha"]:
                        raise ValueError("Raw/G A-only scaler or fit boundary differs")
                    expected_fp = fingerprint(dict(arrays=m["arrays"], metadata={key0: value for key0, value in metadata.items() if key0 != "reused_prediction_cache"}))
                    if expected_fp != receipt["parent_cache_fingerprints"][key]:
                        raise ValueError("Frozen M/V cache fingerprint differs")
            if fingerprint(receipt["parent_cache_fingerprints"]) != receipt["contract"]["parents_sha"]:
                raise ValueError("Combined frozen parent fingerprint differs")
            path = public / executed["V_manifest_relative_path"]
            manifest = pinned(path, executed["V_manifest_sha256"])
            reference = read(path.with_name("REFERENCE_CONTRACT.json"))
            pre_path = public / reference["pre_target_contract_public_relative_path"]
            pre = pinned(pre_path, reference["pre_target_contract_sha256"])
            for key in ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "scaler_sha"):
                if pre[key] != reference[key] or reference[key] != manifest[key]:
                    raise ValueError("V pre-fit truth seal identity differs")
            if pre["arrays"] != reference["arrays"] or any(receipt["contract"].get(key) != value for key, value in pre["expected_receipt_contract"].items()):
                raise ValueError("V pre-fit target seal/fit differs")
            realized = reference["realized_model_binding"]
            if realized != manifest["metadata"]["realized_model_binding"] or fingerprint(realized) != manifest["model_sha"] or realized["selected_artifact_sha256"] != receipt["selected_checkpoint_sha256"]:
                raise ValueError("Realized selected model differs")
            for role in ("g", "b"):
                if pre["parent_anchors"][role]["model_sha"] != parent[role]["V_select"]["manifest"]["model_sha"] or realized["parent_model_shas"][role] != parent[role]["V_select"]["manifest"]["model_sha"]:
                    raise ValueError("Deployed active parent identity differs")
            record(public / manifest["public_stats_relative_path"])
            original_scores = list(csv.DictReader(record(public / "combinations" / identity / "scores.csv").open(encoding="utf-8-sig", newline="")))
            for split in ("V_select", "M_cal"):
                declared = receipt["prediction_artifacts"]["head"][split]
                private_path = record(cache / "combinations" / identity / declared["file"])
                allowed_npz.add(private_path)
                if sha(private_path) != declared["sha256"]:
                    raise ValueError("Original V/M prediction artifact SHA differs")
                for key in ("target_normalized", "observed", "origins", "channel_ids"):
                    if declared["arrays"][key] != parent["g"][split]["reference"]["arrays"][key] or declared["arrays"][key] != parent["b"][split]["reference"]["arrays"][key]:
                        raise ValueError("Child and Raw/G original truth/mask/origin/channel differ")
                score = next(row for row in receipt["scores"] if row["split"] == split and row["run_id"] == identity)
                if split == "V_select":
                    m, ref, mp = manifest, reference, path
                else:
                    m = {key: manifest[key] for key in ("run_id", "source_sha", "config_sha", "data_sha", "scaler_sha", "model_sha")}
                    m.update(split=split, cohort=score["cohort"], cache_relative_path=private_path.relative_to(cache).as_posix(), archive_sha256=declared["sha256"], arrays=declared["arrays"], score=score,
                             public_stats_relative_path=None, cost_record=dict(run_id=identity, batch=score["n_origins"], latent_k=0, F_calls=0, actual_F_streams=0, expected_F_streams=0, execution_scope="cached_parent_calibration_output"),
                             metadata=dict(audit_only_derived_manifest=True, original_head_M_artifact=True, original_receipt_relative_path=executed["receipt_relative_path"], raw_target_publication=False))
                    ref = {key: m[key] for key in ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "scaler_sha", "model_sha")}
                    ref.update(arrays=parent["g"][split]["reference"]["arrays"], expected_F_streams=0, audit_only_derived_reference=True,
                               truth_anchor_relative_path=route["g_predictions"][split]["reference_relative_path"], truth_anchor_sha256=route["g_predictions"][split]["reference_sha256"],
                               scope="Post-fit audit mapping to preexisting frozen parent M truth; not a new original pre-fit child reference")
                    mp = output / "DERIVED_M_CONTRACTS" / identity / "PREDICTION_MANIFEST.json"
                    new(mp, m);new(mp.with_name("REFERENCE_CONTRACT.json"), ref)
                bundles.append(dict(manifest=m, reference=ref, manifest_path=mp, archive=private_path, score=score, dataset=dataset, receipt=receipt, original_scores=original_scores))
            groups.append(dict(dataset=dataset, spec=spec, route=route, receipt=receipt, g_artifact_sha256=sha(g_artifact), g_manifest=parent["g"]["V_select"]["manifest"], raw_manifest=parent["b"]["V_select"]["manifest"]))
            for comparison_route in route["item"]["routes"]:
                read(public / "combinations" / comparison_route["spec"]["run_id"] / "receipt.json")
                for comparison_row in comparison_route["registry_rows"]:
                    original = next(row for row in prior_audit["records"] if row["run_id"] == comparison_row["experiment_id"] and row["split"] == "V_select")
                    pinned(public / original["manifest_public_relative_path"], original["manifest_sha256"])
        target_ids = {group["spec"]["run_id"] for group in groups}
        validation = [dict(run_id=identity, execution_status="COMPLETE_VALID" if identity in target_ids else row["execution_status"], reason=row.get("reason") or "No targeted completed artifact in this delta audit", original_execution_status=row["execution_status"]) for identity, row in by_id.items()]
        new(output / "VALIDATION_REGISTRY_PROJECTION.json", validation)
        new(output / "INPUTS_BEFORE.json", dict(source39=sources, inputs=inputs, registry_sha256=hashlib.sha256(registry_bytes).hexdigest()))

        def guard(event, values):
            if event == "import" and str(values[0]).split(".")[0] in {"torch", "chronos"}:
                forbidden.append("MODEL_IMPORT");raise RuntimeError("No model imports")
            if event == "pickle.find_class":
                forbidden.append("PICKLE_DESERIALIZATION");raise RuntimeError("No model deserialization")
            if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(values[0])).resolve()
            if "/t_final/" in path.as_posix().casefold() or path.name.casefold() in {"gpu_owner.lock", "active_gpu_lease.json"}:
                forbidden.append("T_OR_GPU");raise RuntimeError("No T/GPU ownership")
            if path.suffix.casefold() == ".npz" and path not in allowed_npz and path not in parent_hash_npz:
                forbidden.append("OTHER_NPZ");raise RuntimeError("Only eight new RawH V/M NPZs may be deserialized")
            if path.suffix.casefold() in {".pt", ".safetensors"} and path not in allowed_weights:
                forbidden.append("OTHER_WEIGHT");raise RuntimeError("Only bound selected bytes may be hashed")
            flags = values[2] if len(values) > 2 and isinstance(values[2], int) else 0
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) and not path.is_relative_to(output):
                forbidden.append("OTHER_WRITE");raise RuntimeError("Writes confined to new diagnostics")
        sys.addaudithook(guard)
        sys.path.insert(0, str(public))
        import audit_predictions as audit
        import numpy as np
        import psutil
        from threadpoolctl import threadpool_limits
        process = psutil.Process()
        original_np_load = np.load
        def only_new_arrays(file, *args, **kwargs):
            if Path(file).resolve() not in allowed_npz:
                raise RuntimeError("Only eight new saved output arrays may be loaded")
            return original_np_load(file, *args, **kwargs)
        np.load = only_new_arrays
        with threadpool_limits(limits=2):
            for header in bundles:
                try:
                    if time.time()-started > 600 or process.memory_info().rss > 2*1024**3:
                        raise RuntimeError("CPU wall600/RSS2GiB audit boundary")
                    m, ref = header["manifest"], header["reference"]
                    if m["split"] == "V_select":
                        with gzip.open(public / m["public_stats_relative_path"], "rt", encoding="utf-8-sig", newline="") as stream:
                            result = audit.audit_prediction_bundle(m, ref, m["score"], csv.DictReader(stream), validation, list(by_id), [m["cost_record"]], cache)
                    else:
                        audit._check_registry(validation, list(by_id), m["run_id"]);audit._check_cost([m["cost_record"]], m["run_id"], ref)
                        arrays = audit._load_arrays(m, ref, cache)
                        mse, mae, count = [], [], 0
                        for channel in range(arrays["pred_normalized"].shape[2]):
                            mask = arrays["observed"][:, :, channel]
                            n = int(mask.sum())
                            if n:
                                delta = arrays["pred_normalized"][:, :, channel][mask].astype(np.float64)-arrays["target_normalized"][:, :, channel][mask].astype(np.float64)
                                if not np.isfinite(delta).all():raise ValueError("Observed dense error nonfinite")
                                mse.append(float(np.dot(delta, delta))/n);mae.append(float(np.abs(delta).sum(dtype=np.float64))/n);count += n
                        result = dict(status="PASS_PRIVATE_M_RECEIPT_RESCORING", run_id=m["run_id"], split=m["split"], cohort=m["cohort"], mse=float(np.mean(mse)), mae=float(np.mean(mae)), n_observed=count,
                            active_channels=len(mse), n_origins=arrays["pred_normalized"].shape[0], horizon=arrays["pred_normalized"].shape[1], num_channels=arrays["pred_normalized"].shape[2],
                            public_stats_reaggregation=False, prediction_rescoring=True, checkpoint_replay=False, training_rerun=False, M_reference_postfit_derived_from_existing_parent_truth=True)
                        del arrays
                    arrays = audit._load_arrays(m, ref, cache)
                    asset = assets[header["dataset"]];bounds = asset["primary_split_ranges"][m["split"]]
                    expected = np.arange(max(bounds[0], asset["L"]), bounds[1]-asset["H"]+1, 1 if m["split"] == "M_cal" else asset["H"], dtype=np.int64)
                    if not np.array_equal(arrays["origins"], expected) or not np.array_equal(arrays["channel_ids"], np.asarray(asset["columns"], dtype=str)):
                        raise ValueError("Actual registered origin/channel coordinates differ")
                    for key in ("mse", "mae"):
                        audit._same_number(header["score"][key], result[key], "Original head receipt " + key)
                    for key in ("n_observed", "n_origins", "horizon", "num_channels", "active_channels"):
                        if int(header["score"][key]) != result[key]:raise ValueError("Original head receipt count differs")
                    original = [row for row in header["original_scores"] if row["run_id"] == m["run_id"] and row["split"] == m["split"]]
                    if len(original) != 1:raise ValueError("Original aggregated score CSV identity differs")
                    for key in ("mse", "mae"):
                        audit._same_number(original[0][key], result[key], "Original combination score CSV " + key)
                    result.update(reported_mse=header["score"]["mse"], reported_mae=header["score"]["mae"], mse_difference=result["mse"]-header["score"]["mse"], mae_difference=result["mae"]-header["score"]["mae"],
                        manifest_public_relative_path=header["manifest_path"].relative_to(public).as_posix(), manifest_sha256=sha(header["manifest_path"]), archive_cache_relative_path=m["cache_relative_path"], archive_sha256=m["archive_sha256"],
                        selected_head_artifact_sha256=header["receipt"]["selected_checkpoint_sha256"], updates=header["receipt"]["updates"], selected_updates=header["receipt"]["selected_updates"], training_status=header["receipt"]["training_status"], actual_epochs=header["receipt"]["actual_epochs"])
                    results.append(result);headers.append(header);del arrays
                    print(json.dumps(dict(run_id=m["run_id"], split=m["split"], status=result["status"])), flush=True)
                except Exception as error:
                    message = str(error).replace(str(cache).replace("\\", "\\\\"), "PRIVATE_CACHE").replace(str(public).replace("\\", "\\\\"), "PUBLIC_CAMPAIGN").replace(str(cache), "PRIVATE_CACHE").replace(str(public), "PUBLIC_CAMPAIGN")
                    errors.append(dict(run_id=header["manifest"]["run_id"], split=header["manifest"]["split"], error_type=type(error).__name__, error=message))
        result_map = {(row["run_id"], row["split"]): row for row in results}
        old_records = {row["run_id"]: row for row in prior_audit["records"] if row["split"] == "V_select"}
        for group in groups:
            dataset, spec, route, receipt = (group[key] for key in ("dataset", "spec", "route", "receipt"))
            g_manifest, raw_manifest = group["g_manifest"], group["raw_manifest"]
            entries = [("RawH", spec["run_id"], result_map[(spec["run_id"], "V_select")], receipt, spec)]
            for previous_route in route["item"]["routes"]:
                previous_id = previous_route["spec"]["run_id"]
                if previous_route["spec"]["g_run_id"] != spec["g_run_id"]:raise ValueError("Comparison head uses different frozen G")
                for row in previous_route["registry_rows"]:
                    identity = row["experiment_id"]
                    old_result = old_records[identity]
                    manifest = pinned(public / old_result["manifest_public_relative_path"], old_result["manifest_sha256"])
                    old_receipt = read(public / "combinations" / previous_id / "receipt.json")
                    if old_receipt["parent_models"]["g_v"]["model_sha"] != g_manifest["model_sha"] or old_receipt["parent_models"]["g_v"]["artifact_sha256"] != group["g_artifact_sha256"]:
                        raise ValueError("Comparison head uses different actual G artifact/model")
                    for key in ("target_normalized", "observed", "origins", "channel_ids"):
                        if manifest["arrays"][key] != g_manifest["arrays"][key]:raise ValueError("Matched comparison target/mask/cohort differs")
                    family = row["family"]
                    label = ("H_B0" if previous_route["parent_family"] == "B0" else "H_B2") if family == "CONDITIONAL" else "Hpast" if family == "PAST_CONTROL" else family
                    if family in {"CONDITIONAL", "PAST_CONTROL"}:
                        for key in ("seed", "device", "learning_rate", "epoch_endpoints", "effective_batch", "microbatch", "fold"):
                            if previous_route["spec"][key] != spec[key]:raise ValueError("Matched H optimization condition differs")
                        if receipt["head_parameter_count"] != old_receipt["head_parameter_count"] or receipt["initial_state_sha"] != old_receipt["initial_state_sha"]:
                            raise ValueError("Matched H class/capacity/initialization differs")
                    entries.append((label, identity, old_result, old_receipt, previous_route["spec"]))
            entries += [("G", g_manifest["run_id"], g_manifest["score"], None, None), ("Raw", raw_manifest["run_id"], raw_manifest["score"], None, None)]
            for label, identity, score, fit, fit_spec in entries:
                comparisons.append(dict(dataset=dataset, method=label, run_id=identity, g_run_id=spec["g_run_id"], g_model_sha=g_manifest["model_sha"], g_artifact_sha256=group["g_artifact_sha256"],
                    mse=score["mse"], mae=score["mae"], mse_delta_vs_G=score["mse"]-g_manifest["score"]["mse"], mse_relative_pct_vs_G=(score["mse"]/g_manifest["score"]["mse"]-1)*100,
                    mse_delta_vs_RawH=score["mse"]-result_map[(spec["run_id"], "V_select")]["mse"], n_observed=score["n_observed"], n_origins=score["n_origins"], horizon=score["horizon"], num_channels=score["num_channels"],
                    cohort=g_manifest["cohort"], source_sha=receipt["contract"]["source_sha"] if fit else g_manifest["source_sha"] if label == "G" else raw_manifest["source_sha"],
                    updates=fit.get("updates", 0) if fit else 0, selected_updates=fit.get("selected_updates") if fit else None, training_status=fit["training_status"] if fit else "FROZEN_PARENT",
                    same_G_head_comparison=fit is not None, optimizer_fit_is_independent_replication=False, actual_cost_measured_here=False))
        after = {}
        for key in inputs:
            scope, relative_path = key.split(":", 1)
            path = (public if scope == "public" else cache) / relative_path
            after[key] = dict(sha256=sha(path), bytes=path.stat().st_size)
        sources_after = {name: sha(public / name) for name in sources}
        after_registry = {row["experiment_id"]: row for row in csv.DictReader(registry_path.open(encoding="utf-8-sig", newline=""))}
        scientific_changes = [identity for identity, row in by_id.items() if identity not in after_registry or any(after_registry[identity][key] != value for key, value in row.items() if key not in MUTABLE)]
        mutable_changes = [identity for identity, row in by_id.items() if identity in after_registry and any(after_registry[identity][key] != row[key] for key in MUTABLE)]
        new_rows = []
        for header in headers:
            m = header["manifest"]
            key = (m["run_id"], m["split"], m["cohort"])
            if key in old_keys or m["cache_relative_path"] in old_paths:raise ValueError("New RawH bundle overlaps previous449")
            new_rows.append(dict(run_id=m["run_id"], split=m["split"], cohort=m["cohort"], manifest_public_relative_path=header["manifest_path"].relative_to(public).as_posix(), manifest_sha256=sha(header["manifest_path"]),
                archive_cache_relative_path=m["cache_relative_path"], archive_sha256_as_declared=m["archive_sha256"], archive_exists=True, archive_bytes=header["archive"].stat().st_size,
                M_manifest_audit_only_derived=m["split"] == "M_cal"))
        union = old_union + new_rows
        if len({(row["run_id"], row["split"], row["cohort"]) for row in union}) != len(union) or len({row["archive_cache_relative_path"] for row in union}) != len(union):raise ValueError("Union duplicate identity/archive path")
        seen_content, duplicate_content = {}, []
        for index, row in enumerate(union):
            digest = row["archive_sha256_as_declared"]
            if index >= 449 and digest in seen_content:
                duplicate_content.append(dict(run_id=row["run_id"], split=row["split"], archive_sha256=digest, same_byte_prior_identities=list(seen_content[digest]), distinct_identity_not_independent_replication=True))
            seen_content.setdefault(digest, []).append(row["run_id"] + ":" + row["split"])
        complete = len(results) == 8 and len(groups) == 4 and not errors and sources == sources_after and inputs == after and not scientific_changes and not forbidden and "torch" not in sys.modules
        new(output / "INPUTS_AFTER.json", dict(source39=sources_after, inputs=after))
        new(output / "AUDIT_RESULTS.json", dict(records=results, errors=errors, new_duplicate_archive_content=duplicate_content))
        csv_new(output / "UNION_BUNDLES.csv", union, sorted(set().union(*(row.keys() for row in union))))
        csv_new(output / "SAVED_BUNDLE_RESCORING.csv", results, ("run_id", "split", "cohort", "status", "reported_mse", "mse", "mse_difference", "reported_mae", "mae", "mae_difference", "n_observed", "n_origins", "num_channels", "horizon", "active_channels", "updates", "selected_updates", "training_status", "actual_epochs", "manifest_sha256", "archive_sha256", "selected_head_artifact_sha256"))
        csv_new(output / "MATCHED_G_COMPARISON.csv", comparisons, ("dataset", "method", "run_id", "g_run_id", "g_model_sha", "g_artifact_sha256", "mse", "mae", "mse_delta_vs_G", "mse_relative_pct_vs_G", "mse_delta_vs_RawH", "n_observed", "n_origins", "horizon", "num_channels", "cohort", "source_sha", "updates", "selected_updates", "training_status", "same_G_head_comparison", "optimizer_fit_is_independent_replication", "actual_cost_measured_here"))
        summary.update(status="PASS_RAW_DEEP_G_SAVED_V_AND_M" if complete else "PARTIAL_OR_FAILED", actual_new_V_rescores=sum(row["split"] == "V_select" for row in results), actual_new_M_head_rescores=sum(row["split"] == "M_cal" for row in results),
            actual_new_rescores=len(results), previous449_rerescored=False, union_unique_saved_bundles=len(union), union_V=sum(row["split"] == "V_select" for row in union), union_M=sum(row["split"] == "M_cal" for row in union),
            distinct_new_bundle_ids_with_duplicate_archive_content=len(duplicate_content),
            new_M_reference_scope="Audit-only post-fit mapping to preexisting frozen parent truth; no original M child manifest existed", source39_before=sources, source39_after=sources_after, source39_unchanged=sources == sources_after,
            source33_unchanged=True, immutable_inputs=len(inputs), inputs_unchanged=inputs == after, previous449_public23_allowlist_preserved=True,
            registry_before_sha256=hashlib.sha256(registry_bytes).hexdigest(), registry_after_sha256=sha(registry_path), registry_scientific_changes=scientific_changes, registry_mutable_changes=mutable_changes,
            concurrently_added_registry_ids=sorted(set(after_registry)-set(by_id)), matched_G_comparison_rows=len(comparisons), matching_head_conditions_checked=True, all_parent_A_only_fit_boundaries_checked=True,
            forbidden_io_attempts=forbidden, torch_imported="torch" in sys.modules, CPU_wall_seconds=time.time()-started, CPU_process_seconds=process.cpu_times().user+process.cpu_times().system, RSS_peak_bytes=getattr(process.memory_info(), "peak_wset", process.memory_info().rss), errors=errors)
    except Exception as error:
        message = str(error)
        for path, label in ((cache, "PRIVATE_CACHE"), (public, "PUBLIC_CAMPAIGN")):
            message = message.replace(str(path).replace("\\", "\\\\"), label).replace(str(path), label).replace(path.as_posix(), label)
        summary.update(status="FAILED_PRESERVED_DIAGNOSTIC", fatal_error_type=type(error).__name__, fatal_error=message, partial_results=results, errors=errors, forbidden_io_attempts=forbidden)
    summary.update(finished_unix=time.time(), reproducible_driver_sha256=sha(__file__))
    new(output / "RESULT.json", summary)
    print(json.dumps(dict(status=summary["status"], new_rescores=len(results), union=summary.get("union_unique_saved_bundles"), seconds=time.time()-started)), flush=True)
    if summary["status"] != "PASS_RAW_DEEP_G_SAVED_V_AND_M":raise SystemExit(1)


if __name__ == "__main__":
    main()
