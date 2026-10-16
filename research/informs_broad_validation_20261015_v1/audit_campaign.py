"""CPU campaign audit and two-phase independent combination target sealing."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

import audit_predictions as independent
import data_contract as data
import evaluation
from training import fingerprint


def _json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    target = (root / relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(root) or target == root:
        raise ValueError("Artifact reference must stay inside its declared root")
    return target


def _source_contract(kind, spec, task, mean, std, parent):
    names = ("raw_calibration.py", "combination.py", "data_contract.py", "evaluation.py", "training.py") if kind == "raw_calibration" else (
        "combination.py", "networks.py", "training.py", "evaluation.py", "data_contract.py")
    source = fingerprint({name: data.sha256_file(Path(__file__).with_name(name)) for name in names})
    scaler = fingerprint({"mean": evaluation.array_fingerprint(np.asarray(mean, np.float32)), "std": evaluation.array_fingerprint(np.asarray(std, np.float32))})
    if kind == "raw_calibration":
        config = fingerprint(dict(variant="per_channel_constant_bias", fit_split="M_cal", penalty=0))
        return dict(source_sha=source, config_sha=config, data_sha=task["data_sha256"], scaler_sha=scaler,
                    parent_model_sha=parent["metadata"]["model_sha"])
    config = fingerprint({key: value for key, value in spec.items() if key not in {"run_id", "resume_checkpoint"}})
    cls = "LegacyResidualHead" if spec["variant"] == "legacy_residual" else "ConditionalHead"
    return dict(source_sha=source, config_sha=config, data_sha=task["data_sha256"], scaler_sha=scaler,
                model_sha=fingerprint({"class": cls, "L": task["L"], "H": task["H"], "hidden": 32}))


def _parent_anchor(parent, public_root):
    meta = parent["metadata"]
    path = Path(public_root) / "predictions" / meta["run_id"] / "V_select/PREDICTION_MANIFEST.json"
    if not path.is_file() or not isinstance(parent.get("manifest"), dict):
        raise ValueError("Combination references require actual completed parent prediction manifests")
    loaded = _load(path)
    if fingerprint(loaded) != fingerprint(parent["manifest"]):
        raise ValueError("Supplied parent manifest differs from its actual public file")
    if loaded["model_sha"] != meta["model_sha"]:
        raise ValueError("Parent forecast model SHA is inconsistent")
    return dict(run_id=meta["run_id"], model_sha=meta["model_sha"], manifest_public_relative_path=path.relative_to(public_root).as_posix(),
                manifest_sha256=data.sha256_file(path), archive_sha256=loaded["archive_sha256"])


def prepare_combination_reference(run_id, task, parent_v, g_v, mean, std, cache_root, public_root, *, spec, fold="primary", kind="combination"):
    """Freeze task-derived V truth BEFORE any combination fit/output exists.

    Selected weights cannot be known pre-fit. Their exact identity is attached
    later, without changing these frozen target/cohort/parent/spec fingerprints.
    """
    if not run_id or Path(run_id).name != run_id or kind not in {"combination", "raw_calibration"}:
        raise ValueError("Safe ID and registered reference kind required")
    bounds = data.split_ranges(task, fold)
    canonical_mean, canonical_std, _ = data.fit_statistics(task, bounds["A_stop"][1])
    if not np.array_equal(np.asarray(mean, np.float32), canonical_mean) or not np.array_equal(np.asarray(std, np.float32), canonical_std):
        raise ValueError("Independent reference requires actual fold A60 statistics")
    oo = data.origins(bounds["V_select"], task["L"], task["H"], task["H"])
    pairs = [data.target_window(task, int(origin), canonical_mean, canonical_std) for origin in oo]
    truth = dict(target_normalized=np.stack([p[0] for p in pairs]), observed=np.stack([p[1] for p in pairs]),
                 origins=oo, channel_ids=np.asarray(task["columns"], dtype=str))
    arrays = {key: evaluation.array_fingerprint(value) for key, value in truth.items()}
    anchors = {"b": _parent_anchor(parent_v, Path(public_root)), "g": _parent_anchor(g_v, Path(public_root))}
    for parent in (parent_v, g_v):
        for key in independent.REFERENCE_KEYS:
            if evaluation.array_fingerprint(parent[key]) != arrays[key]:
                raise ValueError("Supplied parent cohort differs from independent Task truth: " + key)
    labels = {"bias": run_id} if kind == "raw_calibration" else {"head": run_id}
    if spec.get("variant") == "alpha_family":
        if "_ALPHA_FIT_" not in run_id:
            raise ValueError("Alpha family needs the registered ALPHA_FIT family ID")
        labels = {"alpha0": run_id.replace("_ALPHA_FIT_", "_ALPHA0_"), "alpha05": run_id.replace("_ALPHA_FIT_", "_ALPHA05_"),
                  "alpha1": run_id.replace("_ALPHA_FIT_", "_ALPHA1_"), "fitted": run_id}
    expected = _source_contract(kind, spec, task, canonical_mean, canonical_std, parent_v)
    references = {}
    for label, alias in labels.items():
        name = f"{label}_V_select.npz" if spec.get("variant") == "alpha_family" else "V_select.npz"
        relative_cache = f"{'raw_calibration' if kind == 'raw_calibration' else 'combinations'}/{run_id}/{name}"
        path = Path(public_root) / "predictions" / alias / "V_select/PRE_TARGET_CONTRACT.json"
        reference = dict(run_id=alias, family_run_id=run_id, label=label, kind=kind, variant=spec.get("variant"), split="V_select",
                         cohort=parent_v["metadata"]["cohort"], source_sha=expected["source_sha"],
                         config_sha=fingerprint(dict(fit_config_sha=expected["config_sha"], output_label=label)), data_sha=task["data_sha256"],
                         scaler_sha=expected["scaler_sha"], arrays=arrays, expected_receipt_contract=expected,
                         expected_F_streams=0, parent_anchors=anchors, cache_relative_path=relative_cache,
                         target_cohort_sealed_before_fit=True, selected_model_identity_available_at_this_stage=False,
                         model_template_is_not_parameter_hash=True)
        if path.exists():
            previous = _load(path)
            if {key: value for key, value in previous.items() if key != "created_unix"} != reference:
                raise ValueError("Existing pre-target seal differs; preserve it and use a new ID")
            reference = previous
        else:
            if _inside(cache_root, relative_cache).exists():
                raise ValueError("Cannot retrospectively claim a pre-output target seal")
            reference["created_unix"] = time.time()
            _json_new(path, reference)
        references[label] = dict(reference, pre_reference_public_relative_path=path.relative_to(public_root).as_posix(),
                                 pre_reference_sha256=data.sha256_file(path))
    return references


def _write_stats(path, rows):
    with gzip.open(path, "xt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run_id", "split", "cohort", "origin", "channel", "horizon_start", "horizon_end", "sse", "sae", "count"))
        writer.writeheader(); writer.writerows(rows)


def wrap_combination_output(receipt, prepared_refs, cache_root, public_root, *, kind="combination"):
    """Attach exact realized parameter identities to the prior frozen truth seal."""
    outputs = []
    for label, prepared in prepared_refs.items():
        pre_path = _inside(public_root, prepared["pre_reference_public_relative_path"])
        if data.sha256_file(pre_path) != prepared["pre_reference_sha256"]:
            raise ValueError("Independent pre-target contract changed")
        pre = _load(pre_path)
        if receipt["run_id"] != pre["family_run_id"] or kind != pre["kind"]:
            raise ValueError("Combination receipt belongs to another registered family")
        for key, value in pre["expected_receipt_contract"].items():
            if receipt["contract"].get(key) != value:
                raise ValueError("Combination fit identity differs from pre-registration: " + key)
        for anchor in pre["parent_anchors"].values():
            if data.sha256_file(_inside(public_root, anchor["manifest_public_relative_path"])) != anchor["manifest_sha256"]:
                raise ValueError("Actual parent prediction manifest changed after target seal")
        cache_path = _inside(cache_root, pre["cache_relative_path"])
        declared = receipt["prediction_artifacts"]["V_select"] if kind == "raw_calibration" else receipt["prediction_artifacts"][label]["V_select"]
        if declared["file"] != cache_path.name or data.sha256_file(cache_path) != declared["sha256"]:
            raise ValueError("Combination output archive differs from actual fit receipt")
        if kind == "raw_calibration":
            artifact_relative = receipt["artifact_cache_relative_path"]
            artifact_sha = receipt["artifact_sha256"]
        elif pre["variant"] == "alpha_family":
            artifact_relative, artifact_sha = None, None
        else:
            artifact_relative = f"combinations/{receipt['run_id']}/selected.pt"
            artifact_sha = receipt["selected_checkpoint_sha256"]
        if artifact_relative is not None and data.sha256_file(_inside(cache_root, artifact_relative)) != artifact_sha:
            raise ValueError("Exact selected checkpoint/bias artifact SHA mismatch")
        alpha = ({"alpha0": 0., "alpha05": .5, "alpha1": 1.}.get(label, receipt.get("alpha_fit", {}).get("alpha_fit"))) if pre["variant"] == "alpha_family" else None
        active = ["b"] if kind == "raw_calibration" or pre["variant"] == "legacy_residual" or label == "alpha1" else (
            ["g"] if pre["variant"] == "past_control" or label == "alpha0" else ["b", "g"])
        realized = dict(selected_artifact_sha256=artifact_sha, selected_artifact_cache_relative_path=artifact_relative,
                        alpha=alpha, parent_model_shas={key: pre["parent_anchors"][key]["model_sha"] for key in active}, output_label=label)
        model_sha = fingerprint(realized)
        identity = {key: pre[key] for key in ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "scaler_sha")}
        identity["model_sha"] = model_sha
        reference = dict(identity, arrays=pre["arrays"], expected_F_streams=0, pre_target_contract_sha256=prepared["pre_reference_sha256"],
                         pre_target_contract_public_relative_path=prepared["pre_reference_public_relative_path"],
                         target_cohort_sealed_before_fit=True, selected_model_identity_bound_after_fit=True,
                         realized_model_binding=realized, model_binding_created_unix=time.time())
        with np.load(cache_path, allow_pickle=False) as archive:
            if set(archive.files) != independent.ARRAY_KEYS:
                raise ValueError("Combination output uses a different array schema")
            arrays = {key: archive[key] for key in archive.files}
        for key in independent.REFERENCE_KEYS:
            if evaluation.array_fingerprint(arrays[key]) != pre["arrays"][key]:
                raise ValueError("Combination output differs from the pre-fit Task truth: " + key)
        target = arrays["target_normalized"]
        score, statistics = evaluation.error_statistics(arrays["pred_normalized"], target, arrays["observed"], arrays["origins"],
                                                        identity["run_id"], "V_select", identity["cohort"], arrays["channel_ids"])
        directory = Path(public_root) / "predictions" / identity["run_id"] / "V_select"
        manifest_path = directory / "PREDICTION_MANIFEST.json"
        if manifest_path.exists():
            previous = _load(manifest_path)
            if previous["archive_sha256"] != data.sha256_file(cache_path) or previous["model_sha"] != model_sha:
                raise ValueError("Existing standardized output differs; do not overwrite")
            outputs.append(previous); continue
        _json_new(directory / "REFERENCE_CONTRACT.json", reference)
        stats = directory / "PUBLIC_ERROR_STATS.csv.gz"
        _write_stats(stats, statistics)
        cost = dict(run_id=identity["run_id"], batch=len(arrays["origins"]), latent_k=0, F_calls=0, actual_F_streams=0, expected_F_streams=0,
                    execution_scope="cached_parent_calibration_output", deployment_cost_included=False)
        metadata = dict(target_cohort_sealed_before_fit=True, selected_model_identity_bound_after_fit=True,
                        all_identity_fields_preinference_claimed=False, realized_model_binding=realized,
                        execution_scope="cached_parent_calibration_output", actual_parent_manifests=pre["parent_anchors"],
                        selected_model_is_parameter_template_hash=False, deployment_F_cost_measured_here=False,
                        active_prediction_parent_keys=active,
                        raw_target_publication=False, family_run_id=receipt["run_id"], output_label=label, reference_kind=kind)
        manifest = dict(identity, cache_relative_path=pre["cache_relative_path"], archive_sha256=data.sha256_file(cache_path),
                        arrays={key: evaluation.array_fingerprint(value) for key, value in arrays.items()}, score=score,
                        public_stats_relative_path=stats.relative_to(public_root).as_posix(), cost_record=cost, metadata=metadata)
        _json_new(manifest_path, manifest)
        outputs.append(manifest)
    return outputs


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _dense_rescore(arrays):
    mse, mae, count = [], [], 0
    p, y, mask = arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"]
    for channel in range(p.shape[2]):
        observed = mask[:, :, channel]
        n = int(observed.sum())
        if n:
            delta = p[:, :, channel][observed].astype(np.float64) - y[:, :, channel][observed].astype(np.float64)
            if not np.isfinite(delta).all(): raise ValueError("Dense cache nonfinite observed error")
            mse.append(float(np.dot(delta, delta)) / n); mae.append(float(np.abs(delta).sum(dtype=np.float64)) / n)
            count += n
    return dict(mse=float(np.mean(mse)) if mse else None, mae=float(np.mean(mae)) if mae else None, n_observed=count)


MUTABLE_REGISTRY_COLUMNS = {"execution_status", "training_status", "validity_status", "scientific_status", "publication_status", "reason", "resume_command"}


def _registry_contract(public, errors):
    initial = _rows(public / "EXPERIMENT_REGISTRY_INITIAL.csv")
    current = _rows(public / "EXPERIMENT_REGISTRY.csv")
    declared, extensions = list(initial), []
    for path in sorted(public.glob("EXPERIMENT_REGISTRY_EXTENSION_*.csv")):
        try:
            extension, receipt = _rows(path), _load(path.with_suffix(".json"))
            if not extension or set(extension[0]) != set(initial[0]):
                raise ValueError("Registry extension schema differs from immutable initial registry")
            if data.sha256_file(path) != receipt["csv_sha256"] or receipt.get("before_execution") is not True or receipt.get("before_T_selection") is not True:
                raise ValueError("Extension hash/pre-execution/pre-selection registration proof invalid")
            registered = float(receipt["registered_unix"])
            if not np.isfinite(registered) or not 0 < registered <= time.time():
                raise ValueError("Invalid extension registration timestamp")
            ids = [row["experiment_id"] for row in extension]
            if "run_ids" in receipt and set(ids) != set(receipt["run_ids"]):
                raise ValueError("Extension JSON IDs differ from CSV")
            if (public / "SELECTION_LOCK.json").is_file():
                lock = _load(public / "SELECTION_LOCK.json")
                locked = lock.get("locked_unix", lock.get("created_unix"))
                if locked is not None and registered > float(locked):
                    raise ValueError("Registry extension was registered after T selection lock")
            for run_id in ids:
                for candidate in (public / "combinations" / run_id / "receipt.json", public / "raw_calibration" / run_id / "receipt.json"):
                    if candidate.is_file():
                        start = _load(candidate).get("started_unix")
                        if start is not None and registered > float(start):
                            raise ValueError("Extension registered after actual route execution began")
            declared.extend(extension)
            extensions.append(dict(csv=path.name, csv_sha256=receipt["csv_sha256"], registered_unix=registered, rows=len(extension)))
        except Exception as error:
            errors.append(dict(scope="registry_extension", file=path.name, error=str(error), error_type=type(error).__name__))
    expected = [row["experiment_id"] for row in declared]
    actual = [row["experiment_id"] for row in current]
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual) or set(expected) != set(actual):
        errors.append(dict(scope="registry", error="Initial+extension/current registry ID sets differ or contain duplicates"))
    expected_rows = {row["experiment_id"]: row for row in declared}
    for row in current:
        original = expected_rows.get(row["experiment_id"])
        if original is None: continue
        for key, value in original.items():
            if key not in MUTABLE_REGISTRY_COLUMNS and row.get(key) != value:
                errors.append(dict(scope="registry_contract", run_id=row["experiment_id"], error="Previously registered scientific contract changed: " + key))
    return initial, declared, current, extensions


def _audit_legacy_manifest(path, manifest, public, private):
    import legacy_evaluation

    relative = path.relative_to(public)
    parts = relative.parts
    if (len(parts) != 5 or parts[:2] != ("legacy_e0", "evaluation")
            or re.fullmatch(r"cp\d+", parts[3]) is None):
        raise ValueError("Legacy B manifest has an unexpected evidence path")
    identity = manifest.get("identity", {})
    if (manifest.get("run_id") is not None or manifest.get("split") != legacy_evaluation.SPLIT
            or identity.get("legacy_id") != parts[2]
            or identity.get("checkpoint_update") != int(parts[3][2:])
            or manifest.get("historically_unseen") is not False
            or manifest.get("protected_TEST_accessed") is not False
            or manifest.get("C_scored") is not False):
        raise ValueError("Legacy B path/identity/development-only contract differs")
    result = legacy_evaluation.rescore_checkpoint(private / relative.parent, path.parent)
    saved = _load(path.with_name("INDEPENDENT_RESCORE.json"))
    if result != saved:
        raise ValueError("Legacy B saved independent receipt differs from actual FP64 rescore")
    if result["curve_comparison"]["passed"] is False:
        raise ValueError("Legacy B saved-weight replay differs from its recorded checkpoint curve")
    return dict(result, status="PASS_LEGACY_B_INDEPENDENT_RESCORING", legacy_id=parts[2],
                checkpoint_update=identity["checkpoint_update"], split=legacy_evaluation.SPLIT,
                manifest_public_relative_path=relative.as_posix(), manifest_sha256=data.sha256_file(path),
                rescorer_source_sha256=data.sha256_file(legacy_evaluation.__file__),
                independent_rescore_receipt_sha256=data.sha256_file(path.with_name("INDEPENDENT_RESCORE.json")),
                new_V_T_evidence=False, historically_unseen=False)


def audit_all(public_root, private_root):
    """Audit every saved manifest without changing registry or claiming training."""
    public, private = Path(public_root).resolve(), Path(private_root).resolve()
    batch = "audit_" + str(time.time_ns())
    directory = public / "audits" / batch
    directory.mkdir(parents=True, exist_ok=False)
    errors, records = [], []
    try:
        initial, declared, current, extensions = _registry_contract(public, errors)
    except Exception as error:
        initial, declared, current, extensions = [], [], [], []
        errors.append(dict(scope="registry_read", error_type=type(error).__name__, error=str(error).replace(str(public), "<public-root>")))
    expected_ids = [row["experiment_id"] for row in declared]
    manifests = sorted(public.rglob("PREDICTION_MANIFEST.json"))
    loaded, legacy_records = [], []
    for path in manifests:
        try:
            manifest = _load(path)
            if path.relative_to(public).parts[:2] == ("legacy_e0", "evaluation"):
                legacy_records.append(_audit_legacy_manifest(path, manifest, public, private))
                continue
            if not isinstance(manifest.get("run_id"), str): raise ValueError("Manifest lacks a string run ID")
            loaded.append((path, manifest))
        except Exception as error:
            errors.append(dict(scope="manifest_parse", manifest=path.relative_to(public).as_posix(), error_type=type(error).__name__, error=str(error)))
    manifested_ids = {manifest["run_id"] for _, manifest in loaded}
    current_map = {row["experiment_id"]: row for row in current}
    validation_registry = []
    for row in current:
        status = row["execution_status"]
        allowed_partial = status.startswith("PARTIAL") or status in {"BUDGET_LIMIT", "LIMITED_SCREEN"}
        has_prediction = row["experiment_id"] in manifested_ids
        projection = "COMPLETE_VALID" if has_prediction and (status == "COMPLETE_VALID" or allowed_partial) else status
        validation_registry.append(dict(run_id=row["experiment_id"], execution_status=projection,
                                        reason=row.get("reason") or "No completed prediction artifact at this audit",
                                        original_execution_status=status, projection_scope="Prediction artifact execution only; not full mandatory experiment completion"))
    global_scores = _rows(public / "SCORES.csv") if (public / "SCORES.csv").is_file() else None
    for path, manifest in loaded:
        try:
            reference = _load(path.with_name("REFERENCE_CONTRACT.json"))
            metadata = manifest.get("metadata", {})
            if metadata.get("execution_scope") == "cached_parent_calibration_output":
                pre_path = _inside(public, reference["pre_target_contract_public_relative_path"])
                if data.sha256_file(pre_path) != reference["pre_target_contract_sha256"]:
                    raise ValueError("Combination pre-target seal file changed")
                pre = _load(pre_path)
                if pre["arrays"] != reference["arrays"] or pre["run_id"] != reference["run_id"]:
                    raise ValueError("Post-fit reference changed pre-fit target cohort")
                for key in ("split", "cohort", "source_sha", "config_sha", "data_sha", "scaler_sha"):
                    if pre[key] != reference[key]: raise ValueError("Post-fit identity changed the pre-fit target seal: " + key)
                realized = reference["realized_model_binding"]
                if fingerprint(realized) != manifest["model_sha"]:
                    raise ValueError("Realized selected model identity mismatch")
                if realized["selected_artifact_cache_relative_path"] is not None and data.sha256_file(_inside(private, realized["selected_artifact_cache_relative_path"])) != realized["selected_artifact_sha256"]:
                    raise ValueError("Actual selected model artifact changed")
                for anchor in pre["parent_anchors"].values():
                    if data.sha256_file(_inside(public, anchor["manifest_public_relative_path"])) != anchor["manifest_sha256"]:
                        raise ValueError("Actual parent prediction proof changed")
            if manifest["score"] is None:
                for key in independent.IDENTITY_KEYS:
                    if manifest[key] != reference[key]: raise ValueError("Dense cache identity mismatch")
                independent._check_registry(validation_registry, expected_ids, manifest["run_id"])
                independent._check_cost([manifest["cost_record"]], manifest["run_id"], reference)
                arrays = independent._load_arrays(manifest, reference, private)
                result = dict(status="PASS_DENSE_PRIVATE_RESCORING", run_id=manifest["run_id"], split=manifest["split"],
                              prediction_rescoring=True, public_stats_reaggregation=False, **_dense_rescore(arrays))
            else:
                stats = _inside(public, manifest["public_stats_relative_path"])
                opener = gzip.open if stats.suffix == ".gz" else open
                with opener(stats, "rt", encoding="utf-8-sig", newline="") as stream:
                    result = independent.audit_prediction_bundle(manifest, reference, manifest["score"], csv.DictReader(stream),
                                                                 validation_registry, expected_ids, [manifest["cost_record"]], private)
                if global_scores is not None:
                    matching = [row for row in global_scores if all(str(row.get(key)) == str(manifest[key]) for key in ("run_id", "split", "cohort"))]
                    if len(matching) != 1: raise ValueError("Global SCORES lacks exactly one matching run/split/cohort")
                    for key in ("mse", "mae"): independent._same_number(matching[0][key], result[key], "Global score " + key)
                    for key in ("n_observed", "n_origins", "num_channels", "horizon"):
                        if independent._integer(matching[0][key], key) != result[key]: raise ValueError("Global score count/dimension differs: " + key)
            result.update(manifest_public_relative_path=path.relative_to(public).as_posix(),
                          manifest_sha256=data.sha256_file(path), registry_execution_status=current_map.get(manifest["run_id"], {}).get("execution_status"))
            records.append(result)
        except Exception as error:
            message = str(error).replace(str(private), "<private-cache>").replace(str(public), "<public-root>")
            errors.append(dict(scope="prediction", run_id=manifest.get("run_id"), manifest=path.relative_to(public).as_posix(), error_type=type(error).__name__, error=message))
    if global_scores is not None:
        identities = {(manifest["run_id"], manifest.get("split"), manifest.get("cohort")) for _, manifest in loaded if manifest.get("score") is not None}
        for row in global_scores:
            if row.get("split") in {"V_select", "T_final"} and tuple(row.get(key) for key in ("run_id", "split", "cohort")) not in identities:
                errors.append(dict(scope="global_scores", run_id=row.get("run_id"), error="Numeric V/T score has no auditable prediction manifest"))
    registry_valid = not any(error["scope"].startswith("registry") for error in errors)
    complete = registry_valid and all(row["execution_status"] == "COMPLETE_VALID" for row in current if row.get("requirement", "").startswith("MANDATORY"))
    report = dict(audit_batch=batch, status="FAIL" if errors else ("PASS_AVAILABLE_PREDICTIONS_COVERAGE_INCOMPLETE" if not complete else "PASS_AVAILABLE_PREDICTIONS"),
                  document_audit=True, prediction_rescoring_passed=len(records), public_stats_reaggregation_passed=sum(bool(r.get("public_stats_reaggregation")) for r in records),
                  dense_private_rescoring_passed=sum(r["status"] == "PASS_DENSE_PRIVATE_RESCORING" for r in records),
                  checkpoint_replay_count=0, training_rerun_count=0, timer_reexecution_count=0,
                  mandatory_coverage_complete=complete, global_scores_checked=global_scores is not None,
                  registry_rows=len(current), initial_registry_rows=len(initial), status_counts=dict(Counter(row["execution_status"] for row in current)),
                  registered_extension_rows=len(declared) - len(initial), registry_extensions=extensions,
                  manifests_found=len(manifests), errors=errors, result_count=len(records), created_unix=time.time(),
                  new_campaign_manifests_found=len(loaded), legacy_B_rescoring_passed=len(legacy_records),
                  legacy_B_counted_as_new_V_T=False,
                  scientific_success_inferred_from_audit=False, source_sha256=data.sha256_file(__file__),
                  full_registry_preserved=True, validation_registry_is_artifact_projection=True)
    report["coverage_scope"] = "Initial plus registered-extension experiment IDs; E0 legacy convergence, deployment costs and checkpoint replay are separate evidence"
    if not manifests and not errors: report["status"] = "NO_PREDICTIONS_TO_RESCORE"
    _json_new(directory / "AUDIT_REPORT.json", report)
    _json_new(directory / "PREDICTION_AUDITS.json", records)
    _json_new(directory / "LEGACY_B_AUDITS.json", legacy_records)
    _json_new(directory / "REGISTRY_COVERAGE.json", current)
    _json_new(directory / "VALIDATION_REGISTRY_PROJECTION.json", validation_registry)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", required=True); parser.add_argument("--private-root", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_all(args.public_root, args.private_root), indent=2, ensure_ascii=False))
