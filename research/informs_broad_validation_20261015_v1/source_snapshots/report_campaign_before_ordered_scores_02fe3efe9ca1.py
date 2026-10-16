"""Immutable Korean evidence snapshots; no inference, selection or registry writes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np

import audit_campaign
import audit_predictions as independent
import data_contract as data
from training import fingerprint


STATUS_FIELDS = ("execution_status", "training_status", "validity_status", "scientific_status", "publication_status")
SCORE_FIELDS = ("run_id", "split", "cohort", "dataset", "panel", "fold", "block", "family", "latent_k", "seed", "length", "lora_rank", "point_type", "foundation_point",
                "learning_rate", "learning_rate_source", "max_epoch", "selected_epoch", "training_status", "mse", "mae", "n_observed", "n_origins",
                "num_channels", "horizon", "active_channels", "registry_execution_status", "registry_numeric_status",
                "execution_status", "validity_status", "numeric_audit", "registry_reconciled", "cost_references", "manifest_relative_path", "archive_sha256",
                "source_sha", "config_sha", "model_sha", "data_sha", "scaler_sha", "truth_cohort_sha", "parent_signature",
                "cached_parent_output", "actual_F_streams", "checkpoint_replay")
FIT_PATTERNS = ("runs/*/BASE_FIT_RECEIPT.json", "runs/*/RIDGE_FIT_RECEIPT.json", "runs/*/CONTROL_FIT_RECEIPT.json",
                "combinations/*/receipt.json", "raw_calibration/*/receipt.json", "ordered/**/ORDERED_PILOT_RECEIPT.json")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or path == root or not path.is_relative_to(root):
        raise ValueError("Report input reference escapes its declared root")
    return path


def _link(directory, target, public):
    target, public = Path(target).resolve(), Path(public).resolve()
    if not target.is_relative_to(public) or not target.is_file():
        raise ValueError("Report links require existing public-root files")
    return Path(os.path.relpath(target, directory)).as_posix()


def _input(inputs, path, root, scope="public"):
    path, root = Path(path).resolve(), Path(root).resolve()
    relative = path.relative_to(root).as_posix()
    inputs[(scope, relative)] = dict(scope=scope, file=relative, sha256=data.sha256_file(path), bytes=path.stat().st_size)


def _reason(error, public, cache):
    text = str(error)
    for root, replacement in ((public, "<public-root>"), (cache, "<private-cache>")):
        text = text.replace(str(root), replacement).replace(root.as_posix(), replacement)
    return text


def _recorded_learning_rate(receipt, contract, contract_path):
    spec = contract.get("spec", {})
    candidates = []
    for key in ("lr", "learning_rate"):
        if spec.get(key) is not None:
            candidates.append((float(spec[key]), contract_path.name + ":spec." + key))
    if receipt.get("learning_rate") is not None:
        candidates.append((float(receipt["learning_rate"]), "fit_receipt:learning_rate"))
    if not candidates:
        return None, "NOT_RECORDED_OR_NOT_APPLICABLE"
    if any(not math.isfinite(rate) or rate <= 0 for rate, _ in candidates) or len({rate for rate, _ in candidates}) != 1:
        raise ValueError("Recorded training learning-rate evidence is invalid or inconsistent")
    return candidates[0][0], ";".join(source for _, source in candidates)


def _fits(public, cache, inputs, issues):
    fits, inventory = {}, []
    for path in sorted({path for pattern in FIT_PATTERNS for path in public.glob(pattern)}):
        try:
            receipt = _load(path); _input(inputs, path, public)
            run_id = receipt.get("run_id")
            status = receipt.get("status", receipt.get("execution_status", "UNKNOWN"))
            epochs = receipt.get("epochs", receipt.get("registered_epoch_endpoints", receipt.get("requested_epochs", [])))
            maximum = max(map(float, epochs)) if epochs else None
            info = dict(run_id=run_id or "", receipt_relative_path=path.relative_to(public).as_posix(),
                        status=status, training_status=receipt.get("training_status", "CLOSED_FORM_A_ONLY" if "RIDGE_FIT" in path.name else "UNKNOWN"),
                        max_epoch=maximum, selected_epoch=receipt.get("selected_epoch"),
                        last_cp_best=receipt.get("last_cp_best"), last_scores_still_falling=receipt.get("last_scores_still_falling"),
                        search_updates=receipt.get("search_updates"), refit_updates=receipt.get("refit_updates"),
                        updates=receipt.get("updates"), selected_updates=receipt.get("selected_updates"),
                        artifact_cache_relative_path=receipt.get("artifact_cache_relative_path"), artifact_sha256=receipt.get("artifact_sha256"))
            contract_path = path.with_name("CONTROL_SOURCE_CONTRACT.json" if path.name == "CONTROL_FIT_RECEIPT.json" else "SOURCE_CONTRACT.json")
            contract = _load(contract_path) if contract_path.is_file() else receipt.get("contract", {})
            if contract_path.is_file(): _input(inputs, contract_path, public)
            learning_rate, learning_rate_source = _recorded_learning_rate(receipt, contract, contract_path)
            info.update(source_sha=receipt.get("source_sha", contract.get("source_sha")),
                        config_sha=receipt.get("config_sha", contract.get("config_sha")),
                        data_sha=receipt.get("data_sha", contract.get("data_sha")), seed=receipt.get("seed", contract.get("spec", {}).get("seed")),
                        search_exposures=receipt.get("search_exposures"), refit_exposures=receipt.get("refit_exposures"),
                        exposures=receipt.get("exposures"), observed_target_exposures=receipt.get("observed_target_exposures"),
                        unique_origins=receipt.get("unique_origins"), unique_target_times=receipt.get("unique_target_times"),
                        F_logical_streams=receipt.get("F_logical_streams"), selected_checkpoint_sha=receipt.get("selected_checkpoint_sha256", receipt.get("artifact_sha256")),
                        lora_rank=contract.get("spec", {}).get("lora_rank") if receipt.get("family")=="LORA" else None,
                        learning_rate=learning_rate, learning_rate_source=learning_rate_source)
            inventory.append(info)
            if run_id and status == "COMPLETE_VALID":
                if info.get("artifact_cache_relative_path"):
                    artifact = _inside(cache, info["artifact_cache_relative_path"])
                    info["selected_artifact_present"] = artifact.is_file()
                    if not artifact.is_file() or data.sha256_file(artifact) != info["artifact_sha256"]:
                        raise ValueError("Completed fit selected artifact is absent or has changed")
                    info["selected_artifact_sha_verified"] = True
                    _input(inputs, artifact, cache, "private")
                if run_id in fits:
                    raise ValueError("Multiple completed fit receipts for one ID require explicit resolution")
                fits[run_id] = info
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues.append(dict(scope="fit_receipt", file=path.relative_to(public).as_posix(), reason=str(error)))
    return fits, inventory


def _fit_for(run_id, manifest, fits):
    meta = manifest.get("metadata", {})
    parent = meta.get("parent_run_id", meta.get("family_run_id", meta.get("route_spec", {}).get("source_parent_run_id")))
    return fits.get(run_id, fits.get(parent))


def _t_allowed(lock, manifest):
    if lock.get("locked") is not True or manifest["run_id"] not in lock.get("allowed_run_ids", []):
        return False
    meta, run_id = manifest.get("metadata", {}), manifest["run_id"]
    if "route_spec" in meta:
        return lock.get("allowed_route_sha", {}).get(run_id) == fingerprint(meta["route_spec"])
    artifact = meta.get("artifact_sha256", meta.get("realized_model_binding", {}).get("selected_artifact_sha256"))
    return artifact is not None and lock.get("allowed_artifact_sha", {}).get(run_id) == artifact


def _check_realized_binding(manifest, reference, public, cache, inputs):
    meta = manifest.get("metadata", {})
    if meta.get("execution_scope") != "cached_parent_calibration_output":
        return
    pre_path = _inside(public, reference["pre_target_contract_public_relative_path"])
    if data.sha256_file(pre_path) != reference["pre_target_contract_sha256"]:
        raise ValueError("Combination pre-fit target seal hash changed")
    pre = _load(pre_path)
    if pre.get("target_cohort_sealed_before_fit") is not True or pre["arrays"] != reference["arrays"]:
        raise ValueError("Combination target seal differs from the independently frozen cohort")
    for key in ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "scaler_sha"):
        if pre[key] != reference[key]:
            raise ValueError("Combination pre-fit identity changed: " + key)
    realized = reference["realized_model_binding"]
    if fingerprint(realized) != manifest["model_sha"]:
        raise ValueError("Combination actual selected-model identity differs")
    relative = realized.get("selected_artifact_cache_relative_path")
    if relative is not None:
        path = _inside(cache, relative)
        if data.sha256_file(path) != realized["selected_artifact_sha256"]:
            raise ValueError("Combination selected model artifact changed")
        _input(inputs, path, cache, "private")
    for parent in pre["parent_anchors"].values():
        path = _inside(public, parent["manifest_public_relative_path"])
        if data.sha256_file(path) != parent["manifest_sha256"]:
            raise ValueError("Combination parent manifest changed after pre-fit sealing")
        _input(inputs, path, public)
    _input(inputs, pre_path, public)


def _foundation_point(manifest, public):
    if int(manifest["cost_record"]["actual_F_streams"]) > 0:
        return "quantile_0.5_when_used"
    metadata = manifest.get("metadata", {})
    anchors = metadata.get("actual_parent_manifests", {})
    for key in metadata.get("active_prediction_parent_keys", []):
        if key in anchors:
            parent = _load(_inside(public, anchors[key]["manifest_public_relative_path"]))
            if int(parent["cost_record"]["actual_F_streams"]) > 0:
                return "quantile_0.5_in_cached_parent_when_used"
    return "not_used"


def _predictions(public, cache, registry, fits, inputs, issues, lock):
    results = []
    paths = sorted({path for name in ("predictions", "route_predictions") for path in (public / name).rglob("PREDICTION_MANIFEST.json")})
    for path in paths:
        try:
            manifest = _load(path)
            if manifest.get("split") not in {"V_select", "T_final"}:
                continue
            if manifest.get("score") is None:
                raise ValueError("V/T prediction lacks its saved score; no score inferred from a partial output")
            run_id, split = manifest["run_id"], manifest["split"]
            if run_id not in registry:
                raise ValueError("Prediction is not an immutable initial/extension registered ID")
            if split == "T_final" and not _t_allowed(lock, manifest):
                raise ValueError("T_final route/artifact is not covered by an existing locked selection; private T cache unopened")
            row = registry[run_id]
            cohort = f"{row['dataset']}:{row.get('fold', 'primary')}:{split}:stride{int(row['horizon'])}"
            if manifest["cohort"] != cohort:
                raise ValueError("Prediction cohort differs from registered dataset/fold/horizon")
            reference_path = path.with_name("REFERENCE_CONTRACT.json")
            reference = _load(reference_path)
            for key in independent.IDENTITY_KEYS:
                if not str(manifest.get(key, "")) or manifest.get(key) != reference.get(key):
                    raise ValueError("Independent reference identity mismatch: " + key)
            _check_realized_binding(manifest, reference, public, cache, inputs)
            arrays = independent._load_arrays(manifest, reference, cache)
            expected, score = independent._independent_statistics(arrays, manifest)
            stats_path = _inside(public, manifest["public_stats_relative_path"])
            with gzip.open(stats_path, "rt", encoding="utf-8", newline="") as stream:
                aggregated = independent._reaggregate(list(csv.DictReader(stream)), expected, manifest["score"])
            for key, value in score.items():
                if key in manifest["score"]:
                    independent._same_number(manifest["score"][key], value, "saved " + key)
            for key in ("mse", "mae"):
                independent._same_number(aggregated[key], score[key], "public " + key)
            if int(row["horizon"]) != score["horizon"] or int(row["num_channels"]) != score["num_channels"]:
                raise ValueError("Saved forecast dimensions differ from the registered task")
            independent._check_cost([manifest["cost_record"]], run_id, reference)
            fit = _fit_for(run_id, manifest, fits)
            if fit and fit.get("artifact_cache_relative_path"):
                artifact = _inside(cache, fit["artifact_cache_relative_path"])
                if data.sha256_file(artifact) != fit["artifact_sha256"]:
                    raise ValueError("Actual completed-fit selected artifact hash changed")
                _input(inputs, artifact, cache, "private")
            metadata = manifest.get("metadata", {})
            fixed_family = row["family"] in {"CURRENT", "SEASONAL_NAIVE", "RAW_BOLT_SMALL", "RAW_BOLT_MINI", "RAW_BOLT_TINY", "PCA_FIXED_BOLT", "NESTED_PCA"}
            training_status = fit["training_status"] if fit else ("NO_TRAINING_FIXED_RULE_OR_PRETRAINED" if fixed_family else "TRAINING_RECEIPT_NOT_VERIFIED")
            anchors = metadata.get("actual_parent_manifests", {})
            output = dict(run_id=run_id, split=split, cohort=manifest["cohort"], dataset=row["dataset"], panel=row.get("panel", ""), fold=row.get("fold", "primary"),
                block=row["block"], family=row["family"], latent_k=int(row.get("latent_k") or 0), seed=int(row.get("seed") or 0),
                length=int(row["length"]), lora_rank=fit.get("lora_rank") if fit else None,
                point_type="deterministic_point_forecast", foundation_point=_foundation_point(manifest, public),
                learning_rate=fit.get("learning_rate") if fit else None,
                learning_rate_source=fit.get("learning_rate_source", "NOT_RECORDED_OR_NOT_APPLICABLE") if fit else "NOT_RECORDED_OR_NOT_APPLICABLE",
                max_epoch=fit["max_epoch"] if fit else None, selected_epoch=fit["selected_epoch"] if fit else None,
                training_status=training_status, registry_execution_status=row.get("execution_status", "UNKNOWN"),
                registry_numeric_status=row.get("validity_status", "UNKNOWN"), execution_status=row.get("execution_status", "UNKNOWN"),
                validity_status=row.get("validity_status", "UNKNOWN"), numeric_audit="PASS_SAVED_PREDICTION_FP64_AND_PUBLIC_STATS",
                registry_reconciled=row.get("execution_status", "").startswith(("COMPLETE", "PARTIAL")),
                cost_references="", manifest_relative_path=path.relative_to(public).as_posix(), archive_sha256=manifest["archive_sha256"],
                truth_cohort_sha=fingerprint({key: manifest["arrays"][key] for key in sorted(independent.REFERENCE_KEYS)}),
                parent_signature=fingerprint({key: value["model_sha"] for key, value in anchors.items()}) if anchors else "",
                cached_parent_output=metadata.get("execution_scope") == "cached_parent_calibration_output",
                actual_F_streams=manifest["cost_record"]["actual_F_streams"], checkpoint_replay="NOT_CHECKED_HERE",
                **{key: manifest[key] for key in ("source_sha", "config_sha", "model_sha", "data_sha", "scaler_sha")}, **score)
            output["run_id"], output["split"], output["cohort"] = run_id, split, manifest["cohort"]
            for item in (path, reference_path, stats_path):
                _input(inputs, item, public)
            _input(inputs, _inside(cache, manifest["cache_relative_path"]), cache, "private")
            results.append(output)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(dict(scope="prediction", file=path.relative_to(public).as_posix(), reason=_reason(error, public, cache)))
    # Both saved normalized output and an actual physical-input route may exist.
    # Retain both proofs, never select the lower score or collapse distinct sources.
    return results


def _cost_binding(receipt, condition_id, public, cache, registry, inputs):
    binding = receipt.get("accuracy_bindings", {}).get(condition_id)
    if not binding or binding.get("status") != "BOUND_EXACT_ROUTE":
        return "", "UNBOUND_NO_DECLARED_ACCURACY"
    run_id = binding["accuracy_run_id"]
    if binding["condition_id"] != condition_id or run_id not in registry:
        raise ValueError("Cost accuracy binding has a different registered ID")
    spec = binding["accuracy_route_spec"]
    if fingerprint(spec) != binding["accuracy_route_spec_sha256"] or spec["run_id"] != run_id:
        raise ValueError("Cost accuracy route spec hash or ID differs")
    cost_spec = binding.get("cost_route_spec")
    if cost_spec is None:
        raise ValueError("Cost route spec is required for independent exact binding verification")
    if fingerprint(cost_spec) != binding["cost_route_spec_sha256"] or cost_spec["run_id"] != condition_id:
        raise ValueError("Actual cost route spec hash or condition ID differs")
    if cost_spec.get("accuracy_run_id") != run_id:
        raise ValueError("Actual cost route declares another accuracy run")
    allowed = {"run_id", "accuracy_run_id", "row_microbatch", "cost_origins"}
    if {key:value for key,value in cost_spec.items() if key not in allowed} != {key:value for key,value in spec.items() if key not in allowed}:
        raise ValueError("Cost-only changes alter the actual accuracy route")
    path = _inside(public, binding["accuracy_manifest_public_relative_path"])
    if data.sha256_file(path) != binding["accuracy_manifest_sha256"]:
        raise ValueError("Cost accuracy prediction manifest hash differs")
    manifest = _load(path)
    if manifest["run_id"] != run_id or manifest["split"] != "V_select":
        raise ValueError("Cost binding must reference the registered V prediction")
    if manifest["data_sha"] != binding["data_sha256"] or manifest["scaler_sha"] != binding["scaler_sha"]:
        raise ValueError("Cost data or normalization differs from the saved V forecast")
    if manifest["model_sha"] != binding["accuracy_prediction_model_sha"] or manifest["source_sha"] != binding["accuracy_prediction_source_sha"]:
        raise ValueError("Cost accuracy model/source identity differs")
    for artifact in binding["artifact_bindings"].values():
        artifact_path = _inside(cache, artifact["cache_relative_path"])
        if data.sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError("Actual cost selected artifact binding changed")
        _input(inputs, artifact_path, cache, "private")
    _input(inputs, path, public)
    return run_id, "BOUND_EXACT_ROUTE"


def _costs(public, cache, registry, inputs, issues):
    result = []
    for path in sorted((public / "costs").rglob("*.json")):
        try:
            receipt = _load(path)
            if "trials_file" not in receipt:
                continue
            trial_path = _inside(path.parent, receipt["trials_file"])
            if data.sha256_file(trial_path) != receipt["trials_sha256"]:
                raise ValueError("Saved cost trials hash differs from receipt")
            with gzip.open(trial_path, "rt", encoding="utf-8", newline="") as stream:
                trials = list(csv.DictReader(stream))
            for run_id in sorted({row["run_id"] for row in trials}):
                timings = [float(row["whole_seconds"]) for row in trials if row["run_id"] == run_id and row["phase"] == "timed"]
                if not timings or not all(math.isfinite(value) and value >= 0 for value in timings):
                    raise ValueError("Cost receipt has no finite nonnegative timed trials")
                accuracy_id, binding_status = _cost_binding(receipt, run_id, public, cache, registry, inputs)
                counts = Counter(row["phase"] for row in trials if row["run_id"] == run_id)
                complete = receipt["status"] == "COMPLETE_PROTOCOL" and counts["warmup"] >= 10 and counts["timed"] >= 20 and counts["stage_probe"] >= 20
                result.append(dict(run_id=accuracy_id, cost_run_id=run_id, dataset=receipt["dataset"], scope=receipt["scope"],
                    accuracy_binding_status=binding_status, batch_size=receipt["batch_size"], device=receipt["device"], dtype=receipt["dtype"], status="COMPLETE_PROTOCOL" if complete else "PARTIAL_COST_PROTOCOL",
                    timed_trials=len(timings), median_seconds=statistics.median(timings), mean_seconds=statistics.mean(timings),
                    receipt_relative_path=path.relative_to(public).as_posix(), trials_sha256=receipt["trials_sha256"],
                    training_peak_included=receipt.get("training_peak_included", False)))
            _input(inputs, path, public); _input(inputs, trial_path, public)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(dict(scope="cost", file=path.relative_to(public).as_posix(), reason=str(error)))
    return result


def _replays(public, cache, inputs, issues):
    result = []
    for path in sorted((public / "route_replays").glob("*/REPLAY_RECEIPT.json")):
        try:
            receipt = _load(path)
            standard_path = _inside(public, receipt["standard_manifest_public_relative_path"])
            archive = _inside(cache, receipt["cache_relative_path"])
            if data.sha256_file(standard_path) != receipt["standard_manifest_sha256"] or data.sha256_file(archive) != receipt["archive_sha256"]:
                raise ValueError("Replay cache or its original V manifest changed")
            manifest = _load(standard_path)
            if manifest["split"] != "V_select":
                raise ValueError("Replay receipt must reference V only")
            saved = independent._load_arrays(manifest, _load(standard_path.with_name("REFERENCE_CONTRACT.json")), cache)
            with np.load(archive, allow_pickle=False) as stream:
                actual = {key: stream[key] for key in stream.files}
            indices = [int(np.flatnonzero(saved["origins"] == origin)[0]) for origin in actual["origins"]]
            for key in ("target_normalized", "observed"):
                np.testing.assert_array_equal(actual[key], saved[key][indices])
            np.testing.assert_array_equal(actual["channel_ids"], saved["channel_ids"])
            passed = bool(np.allclose(actual["pred_normalized"], saved["pred_normalized"][indices], atol=receipt["atol"], rtol=receipt["rtol"]))
            if passed != (receipt["status"] == "PASS"):
                raise ValueError("Independent replay parity differs from saved receipt")
            result.append(dict(run_id=receipt["run_id"], status=receipt["status"], actual_F_streams=receipt["actual_F_streams"],
                               receipt_relative_path=path.relative_to(public).as_posix(), saved_replay_rescored=True, model_reexecuted_by_report=False))
            _input(inputs, path, public); _input(inputs, archive, cache, "private"); _input(inputs, standard_path, public)
        except (OSError, ValueError, KeyError, TypeError, IndexError, AssertionError) as error:
            reason = "Saved replay reference arrays differ; private array values omitted" if isinstance(error, AssertionError) else _reason(error, public, cache)
            issues.append(dict(scope="replay", file=path.relative_to(public).as_posix(), reason=reason))
    return result


def _legacy(public, cache, inputs, issues):
    ledger_path = public / "UNRESOLVED_RESOLUTION.csv"
    ledger = _rows(ledger_path) if ledger_path.is_file() else []
    if ledger_path.is_file(): _input(inputs, ledger_path, public)
    completed = defaultdict(list)
    for path in sorted((public / "legacy_e0/fits").glob("*.json")):
        try:
            receipt = _load(path); _input(inputs, path, public)
            if receipt.get("status") == "CONTINUED_ENDPOINT_COMPLETE":
                completed[receipt["id"]].append(dict(endpoint=int(receipt["endpoint"]), selected_update=receipt["selected_update"],
                    resolution_status=receipt["resolution_status"], continuation_fidelity=receipt["continuation_fidelity"],
                    receipt_relative_path=path.relative_to(public).as_posix()))
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues.append(dict(scope="legacy_fit", file=path.relative_to(public).as_posix(), reason=str(error)))
    result = []
    for row in ledger:
        available = sorted(completed.get(row["legacy_id"], []), key=lambda value: value["endpoint"])
        last = available[-1] if available else {}
        result.append(dict(legacy_id=row["legacy_id"], dataset=row["dataset"], family=row.get("kind"), latent_k=row["latent_k"],
            seed=row["seed"], original_updates=row["legacy_updates"], recorded_status=row["status"],
            completed_endpoints=";".join(str(item["endpoint"]) for item in available), actual_latest_endpoint=last.get("endpoint"),
            actual_resolution=last.get("resolution_status", "NO_NEW_COMPLETED_ENDPOINT"),
            continuation_fidelity=last.get("continuation_fidelity", row["continuation_fidelity"]),
            matched_control_ids=row.get("matched_control_ids", ""), receipt_references=";".join(item["receipt_relative_path"] for item in available),
            convergence_proven=False))
    scores = []
    for path in sorted((public / "legacy_e0/evaluation").rglob("PREDICTION_MANIFEST.json")):
        try:
            import legacy_evaluation
            relative = path.parent.relative_to(public)
            rescored = legacy_evaluation.rescore_checkpoint(cache / relative, path.parent)
            manifest = _load(path); identity = manifest["identity"]
            scores.append(dict(legacy_id=identity.get("legacy_id"), update=identity.get("checkpoint_update"), split=manifest["split"],
                cohort=manifest["cohort"], mse=rescored["score"]["mse"], mae=rescored["score"]["mae"],
                n_observed=rescored["score"]["n_observed"], curve_match=rescored["curve_comparison"]["passed"],
                manifest_relative_path=path.relative_to(public).as_posix(), independently_unseen=False))
            _input(inputs, path, public)
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues.append(dict(scope="legacy_evaluation", file=path.relative_to(public).as_posix(), reason=_reason(error, public, cache)))
    return result, scores


def _coverage(registry, fits, scores):
    groups = defaultdict(list)
    v, t = {row["run_id"] for row in scores if row["split"] == "V_select"}, {row["run_id"] for row in scores if row["split"] == "T_final"}
    for row in registry.values():
        groups[(row["block"], row["dataset"], row.get("fold", "primary"), row.get("latent_k", "0"))].append(row)
    result = []
    for (block, dataset, fold, k), rows in sorted(groups.items()):
        ids = [row["experiment_id"] for row in rows]
        result.append(dict(block=block, dataset=dataset, fold=fold, latent_k=k, planned=len(ids),
            completed_fit_receipts=sum(run_id in fits for run_id in ids), numeric_V_runs=sum(run_id in v for run_id in ids),
            numeric_T_runs=sum(run_id in t for run_id in ids), no_numeric_V_runs=sum(run_id not in v for run_id in ids),
            registered_complete=sum(row.get("execution_status") == "COMPLETE_VALID" for row in rows),
            registered_blocked=sum(row.get("execution_status", "").startswith("BLOCKED") for row in rows),
            run_ids=";".join(ids)))
    return result


def _matched(scores, issues=None):
    groups = defaultdict(list)
    for row in scores:
        key = tuple(row[key] for key in ("split", "dataset", "fold", "cohort", "latent_k", "seed", "max_epoch", "truth_cohort_sha", "data_sha", "scaler_sha"))
        groups[key].append(row)
    result, omitted = [], set()
    def known_rate(row):
        rate = row.get("learning_rate")
        if rate is not None and math.isfinite(float(rate)) and float(rate) > 0:
            return float(rate)
        if issues is not None and row["run_id"] not in omitted:
            issues.append(dict(scope="matched_comparison", run_id=row["run_id"],
                reason="Iterative model has no independently recorded finite positive learning rate; score retained, matched comparison omitted"))
            omitted.add(row["run_id"])
        return None
    for rows in groups.values():
        for reference, compared in (("B0", "B2"), ("B0", "B4"), ("LAG_CAUSAL", "LAG_BIDIR"), ("ALPHA05", "CONDITIONAL"), ("ALPHA_FIT", "CONDITIONAL")):
            left_all, right_all = [r for r in rows if r["family"] == reference], [r for r in rows if r["family"] == compared]
            if reference.startswith("ALPHA"):
                if len(left_all) != 1 or len(right_all) != 1:
                    continue
                a, b = left_all[0], right_all[0]
                head_rate = known_rate(b)
                if head_rate is None or not a["parent_signature"] or a["parent_signature"] != b["parent_signature"]:
                    continue
                pairs = [(a, b, head_rate, "closed_form_alpha_reference; known_iterative_head_LR; exact_frozen_parents")]
            else:
                left = [(r, known_rate(r)) for r in left_all]
                right = [(r, known_rate(r)) for r in right_all]
                rates = sorted({rate for _,rate in left+right if rate is not None})
                pairs = []
                for rate in rates:
                    arows, brows = [r for r,value in left if value == rate], [r for r,value in right if value == rate]
                    if len(arows) == 1 and len(brows) == 1:
                        pairs.append((arows[0], brows[0], rate, "equal_recorded_iterative_LR"))
            for a, b, rate, rate_contract in pairs:
                result.append(dict(reference_run_id=a["run_id"], compared_run_id=b["run_id"], split=a["split"], dataset=a["dataset"],
                    latent_k=a["latent_k"], seed=a["seed"], max_epoch=a["max_epoch"], learning_rate=rate,
                    reference_learning_rate=a.get("learning_rate"), compared_learning_rate=b.get("learning_rate"),
                    learning_rate_contract=rate_contract, comparison_contract_verified=True,
                    mse_reference=a["mse"], mse_compared=b["mse"], delta_mse=b["mse"] - a["mse"], delta_mae=b["mae"] - a["mae"],
                    interpretation="Same saved target/mask/order/scaler/seed/maximum epoch; iterative LR matched where applicable; no statistical significance or equal GPU-time claim"))
    return result


def _resource_snapshot(public, cache, inputs, issues):
    result = dict(closed_GPU_process_seconds=0., closed_lease_count=0, live_unclosed_seconds_excluded_from_closed_total=0.,
                  live_process_verified=False, active_lease_file_present=False, CUDA_or_GPU_utilization_measured=False)
    path = cache / "GPU_USAGE.jsonl"
    ids = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                lease_id = row.get("lease_id")
                if lease_id is not None and lease_id in ids:
                    raise ValueError("Duplicate closed GPU lease entry")
                if lease_id is not None: ids.add(lease_id)
                charge = float(row["charged_process_seconds"])
                if not math.isfinite(charge) or charge < 0:
                    raise ValueError("Invalid GPU process-wall charge")
                result["closed_GPU_process_seconds"] += charge
                result["closed_lease_count"] += 1
            except (ValueError, KeyError, TypeError) as error:
                issues.append(dict(scope="resource_journal", file=path.name, reason=str(error)))
        _input(inputs, path, cache, "private")
    path = cache / "ACTIVE_GPU_LEASE.json"
    if path.is_file():
        try:
            import psutil
            active = _load(path)
            alive = psutil.pid_exists(int(active["pid"])) and abs(psutil.Process(int(active["pid"])).create_time() - float(active["create_time"])) < 1.
            result.update(active_lease_file_present=True, live_process_verified=alive, live_purpose=active.get("purpose"),
                live_unclosed_seconds_excluded_from_closed_total=max(0., time.time() - float(active["started_unix"])) if alive and active.get("lease_id") not in ids else 0.)
            # The private lease includes an authentication token. Never copy it or its contents.
        except (OSError, ValueError, TypeError, KeyError, psutil.Error) as error:
            issues.append(dict(scope="active_resource", file=path.name, reason=type(error).__name__))
    path = cache / "DOWNLOAD_USAGE.jsonl"
    if path.is_file():
        charges = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result["reserved_download_bytes"] = sum(int(row["reserved_bytes"]) for row in charges)
        _input(inputs, path, cache, "private")
    config = _load(public / "CONFIG_LOCK.json") if (public / "CONFIG_LOCK.json").is_file() else {}
    result["limits"] = config.get("resource_limits", {})
    result["closed_plus_live_estimate_seconds"] = result["closed_GPU_process_seconds"] + result["live_unclosed_seconds_excluded_from_closed_total"]
    result["remaining_GPU_seconds_estimate"] = max(0., result["limits"].get("gpu_process_seconds", 0.) - result["closed_plus_live_estimate_seconds"])
    return result


def _figures(public, inputs, issues):
    candidates = []
    for path in (public / "figures").glob("*/FIGURE_MANIFEST.json"):
        try:
            receipt = _load(path)
            for output in receipt["outputs"]:
                image = _inside(path.parent, output["file"])
                if data.sha256_file(image) != output["sha256"]:
                    raise ValueError("Figure snapshot contains a missing or changed output")
            candidates.append((float(receipt["created_unix"]), path, receipt))
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(dict(scope="figure_snapshot", file=path.relative_to(public).as_posix(), reason=str(error)))
    if not candidates:
        return None
    _, path, receipt = max(candidates, key=lambda value: (value[0], value[1].as_posix()))
    _input(inputs, path, public)
    for output in receipt["outputs"]:
        _input(inputs, _inside(path.parent, output["file"]), public)
    return path, receipt


def _evidence_inventory(public, inputs, issues):
    records = []
    for pattern, kind in (("audits/*/AUDIT_REPORT.json", "independent_campaign_audit"),
                          ("audits/provided_tool_*/PROVIDED_TOOL_AUDIT.json", "supplied_tool_public_statistics_audit"),
                          ("execution_batches/**/COMMAND_*.json", "orchestration_command"),
                          ("diagnostics/**/DIAGNOSTIC_RECEIPT.json", "encoder_diagnostic"),
                          ("patterns/*/PATTERN_MANIFEST.json", "A_fit_channel_patterns")):
        for path in sorted(public.glob(pattern)):
            try:
                receipt = _load(path); _input(inputs, path, public)
                record = dict(kind=kind, file=path.relative_to(public).as_posix(), status=receipt.get("status", "NO_GLOBAL_STATUS"))
                for key in ("completed", "skipped", "failed", "prediction_rescoring_passed", "public_stats_reaggregation_passed",
                            "checkpoint_replay_count", "mandatory_coverage_complete", "returncode", "command_index"):
                    if key in receipt: record[key] = receipt[key]
                if "errors" in receipt: record["error_count"] = len(receipt["errors"])
                records.append(record)
            except (OSError, ValueError, TypeError) as error:
                issues.append(dict(scope=kind, file=path.relative_to(public).as_posix(), reason=str(error)))
    return records


def _run_index(registry, fits, scores):
    by_id = defaultdict(list)
    for row in scores: by_id[row["run_id"]].append(row)
    result = []
    for run_id, row in sorted(registry.items()):
        actual = fits.get(run_id, {})
        predictions = by_id[run_id]
        entry = dict(run_id=run_id, dataset=row["dataset"], block=row["block"], family=row["family"],
            latent_k=row["latent_k"], seed=row["seed"], command=row.get("resume_command", ""),
            command_scope="Registered command; actual execution only when a receipt is linked",
            registry_execution_status=row.get("execution_status"), fit_receipt_present=bool(actual),
            numeric_prediction_manifests=";".join(r["manifest_relative_path"] for r in predictions))
        entry.update(actual)
        entry["run_id"] = run_id
        entry["seed"] = actual.get("seed") if actual.get("seed") is not None else row["seed"]
        if predictions and not actual:
            for key in ("source_sha", "config_sha", "data_sha", "scaler_sha", "model_sha"):
                entry[key] = predictions[0][key]
        result.append(entry)
    return result


def _prediction_index(public, scores, inputs, issues):
    verified = {row["manifest_relative_path"]:row for row in scores}
    bad = {row.get("file") for row in issues if row.get("scope")=="prediction"}
    result = []
    for root in (public/"predictions", public/"route_predictions"):
        for path in sorted(root.rglob("PREDICTION_MANIFEST.json")):
            relative = path.relative_to(public).as_posix()
            try:
                item = _load(path)
                result.append(dict(run_id=item.get("run_id"), split=item.get("split"), cohort=item.get("cohort"),
                    manifest_relative_path=relative, manifest_sha256=data.sha256_file(path),
                    cache_relative_path=item.get("cache_relative_path"), archive_sha256=item.get("archive_sha256"),
                    numeric_status="PASS" if relative in verified else ("EXCLUDED" if relative in bad else "NOT_MAIN_V_OR_T_SCORE"),
                    public_stats_relative_path=item.get("public_stats_relative_path"),
                    source_sha=item.get("source_sha"), config_sha=item.get("config_sha"), data_sha=item.get("data_sha"),
                    scaler_sha=item.get("scaler_sha"), model_sha=item.get("model_sha")))
                _input(inputs, path, public)
            except (OSError, ValueError, TypeError) as error:
                result.append(dict(manifest_relative_path=relative, numeric_status="UNREADABLE", reason=type(error).__name__))
    return result


def _claims(scores, matched, costs, replays, fits, legacy):
    v = [row for row in scores if row["split"] == "V_select"]
    comparisons = {row["compared_run_id"] for row in matched if row.get("comparison_contract_verified") is True}
    recipes = (
        ("future_information_leakage", "실제 미래 타깃/미래 입력 누출이 있었는가", False,
         "데이터 계약과 과거 창 내부 prefix 검사는 독립 타깃 누출 개입 검사를 대신하지 않음", "타깃/미래 개입 및 전처리 학습 범위를 별도 검증"),
        ("predictive_compression", "예측 목적 압축이 같은 K의 단순 인코더보다 유용한가", any(r["family"] in {"B2","B4"} and r["run_id"] in comparisons for r in v),
         "동일 budget 비교와 추가 epoch·seed 재현 필요", "학습량을 늘린 공정 비교와 전체 비용 확인"),
        ("delayed_information", "다른 채널의 지연 정보가 실제로 도움이 되는가", any(r["family"]=="LAG_BIDIR" and r["run_id"] in comparisons for r in v),
         "과거 lag 상관은 추가 예측력이나 인과성의 증거가 아님", "동일 K/학습량의 causal·bidir·단순 대조 비교"),
        ("conditional_vs_simple", "조건부 결합이 고정/학습 α보다 필요한가", any(r["family"]=="CONDITIONAL" and r["run_id"] in comparisons for r in v),
         "같은 frozen 부모 SHA와 M/V 조건의 비교만 허용", "Raw 보정·G·A+M 통제와 전체 경로 비용 비교"),
        ("ordered_multik", "하나의 ordered checkpoint가 여러 K를 효율적으로 지원하는가", any(r["block"]=="E4" for r in v),
         "K별 노출·초기화·max-only prefix 차이가 남을 수 있음", "동일 예산 대조와 실제 K-row 추론 비용 확인"),
        ("convergence", "나쁜 결과가 수렴한 방법의 한계인가", False,
         f"마지막 checkpoint 최적 receipt {sum(r.get('last_cp_best') is True for r in fits.values())}개; 짧은 screen은 수렴 증거가 아님", "1/3 epoch, 학습 곡선·checkpoint 선택·E0 연장으로 구분"),
        ("contrastive_dynamic_graph", "대조학습과 동적 그래프가 필요한가", False,
         "이 보고서는 실행하지 않은 학습 목표·그래프를 가정하지 않음", "단순 모델의 한계와 실패 단계가 확인될 때 별도 가설로 판단"),
        ("bolt_retention", "Chronos-Bolt를 유지할 근거가 있는가", False,
         "같은 표현의 native readout 및 G, Raw, 전체 비용과 최종 검증이 필요", "정확도·비용·단순 대안 대비 가치로 판단"),
        ("whole_pipeline_cost", "실제 전체 경로 비용이 줄었는가", bool(costs),
         "K/C 비율을 측정 속도 향상으로 대신하지 않음", "fresh/Balanced B1/B8 실제 trial·메모리·parity 확인"),
        ("checkpoint_replay", "저장 checkpoint 실제 추론이 저장 숫자와 같은가", bool(replays),
         "이 보고서는 저장된 실제 replay 결과만 다시 검산하며 모델을 실행하지 않음", "누락 route의 실제 V subset replay 수행"),
        ("E0_resolution", "기존 미해결 학습을 수렴 관점에서 해결했는가", False,
         f"새 endpoint가 있는 legacy 대상 {sum(bool(r['completed_endpoints']) for r in legacy)}/{len(legacy)}; 과거 RNG 부재", "개발 B replay와 학습 곡선을 확인하고 일반화와 분리"),
        ("main_publication", "main 게시 및 이미지 링크 확인이 완료됐는가", False,
         "compiler는 commit/push/원격 SHA 조회를 수행하지 않음", "안전 파일 audit 후 게시 및 원격 SHA·이미지 확인"))
    return [dict(claim_id=key, question=question, status="PARTIAL_EVIDENCE_UNRESOLVED" if partial else "UNRESOLVED",
                 uncertainty=uncertainty, next_decision=next_decision,
                 evidence="SCORES.csv;MATCHED_COMPARISONS.csv;FIT_RECEIPTS.csv;COST.csv;REPLAYS.csv;E0_LEDGER.csv")
            for key, question, partial, uncertainty, next_decision in recipes]


def _resume_queue(registry, fits, scores, cache, lock=None):
    lock = lock or {}
    selection_locked = lock.get("locked") is True
    v = {row["run_id"] for row in scores if row["split"]=="V_select"}
    t = {row["run_id"] for row in scores if row["split"]=="T_final"}
    supported = {"B0","B2","B4","DIRECT_G","DIRECT_B2","DIRECT_B4","LORA_R8","ADAPTS_BOLT_PORT","PCA_FIXED_BOLT","LAG_CAUSAL","LAG_BIDIR"}
    result = []
    for run_id, row in sorted(registry.items()):
        fit = fits.get(run_id)
        status, command, checkpoint, reason = "PENDING", "", fit.get("artifact_cache_relative_path") if fit else None, row.get("reason", "")
        ds, family = row["dataset"], row["family"]
        if row.get("execution_status", "").startswith("BLOCKED"):
            status, reason = "BLOCKED_ASSET_OR_ACCESS", reason or "실제 데이터 자산·접근 계약이 필요함"
        elif row.get("execution_status")=="RUNNING":
            status, reason = "RUNNING_PRESERVE_EXISTING_PROCESS", "실행 중 작업을 중단하거나 중복 실행하지 않음"
        elif family in supported and not fit:
            status, command = "PENDING_FIT", f"& $py -X utf8 -u campaign.py --run-id {run_id} --epochs .25"
        elif family in supported and fit.get("max_epoch") is not None and fit["max_epoch"] < 3 and family!="PCA_FIXED_BOLT":
            block = row["block"] if row["block"] in {"E1","E3","E4"} else "E1"
            status, command = "PENDING_TRAINING_EXPOSURE", f"& $py -X utf8 -u deepen.py --parent-id {run_id} --block {block} --max-epoch 3"
        elif run_id not in v:
            if row["block"] == "E2":
                status, command = "PENDING_COMBINATION_OR_CONTROL", f"& $py -X utf8 -u postfit.py --phase {'controls' if family.endswith('REFIT_APLUSM') else 'combine'} --dataset {ds} --epochs .25 1 3"
            elif row["block"] == "E4":
                status, command = "PENDING_ORDERED_PARENTS_OR_INFERENCE", f"& $py -X utf8 -u postfit.py --phase ordered --dataset {ds} --epochs .25 1 3"
            elif family=="RAW_RIDGE" and not fit:
                status, command = "PENDING_CLOSED_FORM_FIT", f"& $py -X utf8 ridge_baseline.py --cache $env:INFORMS_CACHE_ROOT --public . --datasets {ds}"
            elif family in supported or family in {"RAW_BOLT_SMALL","RAW_BOLT_MINI","RAW_BOLT_TINY","CURRENT","SEASONAL_NAIVE","RAW_RIDGE"}:
                status, command = "PENDING_V_INFERENCE", f"& $py -X utf8 -u postfit.py --phase predict --dataset {ds}"
            else:
                status, reason = "IMPLEMENTATION_OR_CONTRACT_PENDING", "현재 지원 runner에 이 등록 방법의 정확한 실행 계약이 없음; 임의 대체 금지"
        elif run_id not in t:
            status, command = "PENDING_FROZEN_SELECTION_AND_T", "RESUME_COMMANDS.ps1의 locked T_final 명령; 기존 선택 lock 필요"
        else:
            continue
        pre_selection_command = command
        resume_scope = "PRE_SELECTION_ONLY"
        if selection_locked and status == "PENDING_FROZEN_SELECTION_AND_T":
            if run_id in lock.get("allowed_run_ids", []):
                status, resume_scope = "PENDING_LOCKED_T_EXECUTION", "EXISTING_FROZEN_SELECTION_ONLY"
            else:
                status, command, reason = "NOT_SELECTED_FOR_LOCKED_T", "", "기존 선택에 포함되지 않은 ID이며 T 결과를 보고 추가하지 않음"
        elif selection_locked and status not in {"BLOCKED_ASSET_OR_ACCESS", "RUNNING_PRESERVE_EXISTING_PROCESS"}:
            status, command, reason = "NEW_CAMPAIGN_OR_POST_T_DEVELOPMENT_REQUIRED", "", "현재 selection/T 보존; 새 campaign/별도 ID와 독립 검증 계약이 필요하며 기존 T를 학습·선택에 재사용하지 않음"
        result.append(dict(run_id=run_id, dataset=ds, block=row["block"], family=family, status=status,
            recorded_execution_status=row.get("execution_status"), checkpoint_cache_relative_path=checkpoint,
            command=command, pre_selection_command=pre_selection_command, resume_scope=resume_scope,
            executable_training_under_existing_selection=not selection_locked, reason=reason, GPU_or_wall_cost_estimate="UNMEASURED_NO_COMPLETION_ETA",
            prerequisites="중앙 lease·잔여 예산·정확한 source/data/scaler 계약; TEST 접근과 자동 선택 금지"))
    return dict(status="PARTIAL_PENDING_QUEUE", automatically_executed=False, run_count=len(result), runs=result,
                selection_locked=selection_locked, training_commands_scope="PRE_SELECTION_ONLY",
                post_selection_training_policy="NEW_CAMPAIGN_OR_SEPARATE_POST_T_DEVELOPMENT_WITH_NEW_IDS; never tune on existing T",
                global_resume_commands="RESUME_COMMANDS.ps1", T_selection_performed=False)


def _supporting_docs(directory, summary, claims, queue):
    brief = ["# 교수님 상담용 실험 진행 요약", "", "**PARTIAL — 현재 진행 snapshot이며 방법론 완성 선언이 아니다.**", "",
        "연구 질문은 과거만 읽는 저차원 표현을 frozen 시계열 foundation model에 연결했을 때, 단순 직접 예측·원 채널 모델·선형 보정 대비 "
        "어떤 조건에서 예측 오차와 전체 비용의 이득이 생기는지다. 낮은 재구성 오차를 미래 예측 적합성의 근거로 대체하지 않는다.", "",
        f"등록 {summary['registry_rows']}개; 완료 fit receipt {summary['complete_fit_receipts']}개; "
        f"독립 숫자 검산 V {summary['numeric_V_run_count']} route / T {summary['numeric_T_run_count']} route. "
        f"실제 비용 조건 {summary['verified_cost_condition_count']}개, 저장 checkpoint replay 검산 {summary['verified_replay_count']}개.", "",
        "현재 제시 가능한 답과 미확정 원인:", ""]
    for claim in claims:
        brief.append(f"- {claim['question']}: **{claim['status']}**. {claim['uncertainty']} 다음 판단: {claim['next_decision']}.")
    brief += ["", "표현·잠재 미래 예측·디코더 문제는 같은 E/D와 학습량의 대조 및 실제 oracle 진단이 있을 때 분리할 수 있다. "
        "현재 안 나온 결과를 원인으로 확정하지 않는다. 현 단계의 우선 확인은 부족한 학습 노출량, 실제 V 추론, 단순 α/G/Raw 보정, E3/E4와 전체 비용이다.", "",
        "[상세 보고서](REPORT_KO.md) · [원수치](SCORES.csv) · [주장과 증거](CLAIM_EVIDENCE.csv) · [남은 조건](RESUME_QUEUE.json)", ""]
    (directory/"PROFESSOR_BRIEF.md").write_text("\n".join(brief), encoding="utf-8")
    reproduce = ["# 재현과 재개", "", "이 snapshot의 report 생성은 CPU 저장 결과 재채점이며 학습/모델 추론을 실행하지 않는다. "
        "입력 SHA는 INPUT_MANIFEST.json, 산출물 SHA와 compiler source는 REPORT_MANIFEST.json에 기록한다. 원 데이터와 weights는 로컬 private cache에 남긴다.", "",
        "1. 저장소에서 CONFIG_LOCK.json, DATA_CONTRACT.json, 초기/확장 registry, 공식 모델 pin을 확인한다.",
        "2. INFORMS_ASSET_ROOT와 INFORMS_CACHE_ROOT를 기존 실제 자산 위치로 설정한다. 이 컴퓨터의 정확한 예시는 RESUME_COMMANDS.ps1에 있다.",
        "3. 중앙 lease가 실행 중이면 기다린다. 기존 프로세스 종료나 중복 GPU 실행을 하지 않는다.",
        "4. RESUME_QUEUE.json의 상태별 선행 조건을 만족한 route만 명령으로 실행한다. 구현/접근 계약이 없는 항목은 명령을 지어내지 않았다.",
        "5. 재개 carry가 명시된 BUDGET_LIMIT 작업은 해당 receipt의 exact resume 명령을 사용한다. 이전 2400 구간의 RNG 복구를 가정하지 않는다.",
        "6. 모든 저장 V 예측을 독립 감사하고 실제 checkpoint replay·비용을 측정한 뒤, 별도 판단으로 선택을 동결해야 T를 실행할 수 있다.",
        "7. 모든 available V/T seed와 조건을 유지한다. T를 보고 seed나 방법을 다시 고르지 않는다.",
        "8. 새 그림/report snapshot을 생성한다. 보고서는 원 입력·기존 결과·registry를 덮어쓰지 않는다.", "",
        "학습 명령은 PRE_SELECTION_ONLY다. 선택이 locked된 뒤 새 학습은 현재 캠페인에서 재개하지 않는다. "
        "새 campaign/별도 ID로 POST_T_DEVELOPMENT를 분리하고 기존 T를 학습·hyperparameter 선택에 재사용하지 않는다. "
        "E0의 과거 B 연장은 별도 개발 증거이며 고정된 E1–E4/T 선택으로 되먹이지 않는다.", "",
        f"현재 queue 항목 {queue['run_count']}개. 추가 GPU/CPU/전체 소요시간은 미측정 조건이 있어 합계 ETA를 계산하지 않았다. "
        "각 runner의 기존 자원 상한과 실제 부과 ledger가 우선한다. M5는 공식 자산/접근이 생겨야 재개 가능하며 권한 우회나 자동 동의는 하지 않는다.", "",
        "```powershell", "& $py -X utf8 -m unittest test_report_campaign -v",
        "& $py -X utf8 report_campaign.py --public-root . --private-root $env:INFORMS_CACHE_ROOT", "```", "",
        "이 문서에 commit/push 명령은 없다. 게시 여부는 실제 원격 SHA와 파일/링크 확인 후 별도 기록해야 한다.", ""]
    (directory/"REPRODUCE.md").write_text("\n".join(reproduce), encoding="utf-8")


def _resume_commands(registry, scores):
    datasets = sorted({r["dataset"] for r in registry.values() if not r["dataset"].startswith("m5_")})
    lines = ["# 중앙 GPU lease가 비어 있고 잔여 예산이 있는 경우 실행. 이 문서는 명령을 자동 실행하지 않는다.",
             "$repo = (git rev-parse --show-toplevel).Trim()",
             "if (-not $env:INFORMS_ASSET_ROOT) { throw 'INFORMS_ASSET_ROOT를 실제 기존 자산 저장소 위치로 먼저 설정하세요.' }",
             "$assetRoot = (Resolve-Path -LiteralPath $env:INFORMS_ASSET_ROOT).Path",
             "if (-not $env:INFORMS_CACHE_ROOT) { $env:INFORMS_CACHE_ROOT = Join-Path $assetRoot '.cache/informs_broad_validation_20261015_v1' }",
             "$py = if ($env:INFORMS_PYTHON) { $env:INFORMS_PYTHON } elseif (Test-Path (Join-Path $assetRoot '.venv/Scripts/python.exe')) { Join-Path $assetRoot '.venv/Scripts/python.exe' } else { Join-Path $assetRoot '.venv/bin/python' }",
             "Set-Location (Join-Path $repo 'research/informs_broad_validation_20261015_v1')", "",
             "$selectionLocked = (Test-Path -LiteralPath 'SELECTION_LOCK.json') -and ((Get-Content -LiteralPath 'SELECTION_LOCK.json' -Raw | ConvertFrom-Json).locked -eq $true)",
             "# PRE_SELECTION_ONLY: final 선택 후 새 학습은 NEW_CAMPAIGN / 별도 POST_T_DEVELOPMENT와 새 ID가 필요.",
             "if ($selectionLocked) { Write-Warning '선택이 동결되어 아래 E1-E4 학습 명령을 건너뜁니다. 기존 T로 학습 조건을 고르지 마세요.' } else {",
             "# E1/E3 최소 .25 screen의 남은 fit. 완료 receipt는 엄격한 계약이 맞을 때만 재사용."]
    for ds in datasets:
        minimum = "" if ds == "robin" else " --minimum-only"
        lines.append(f"& $py -X utf8 -u campaign.py --dataset {ds}{minimum} --epochs .25")
    lines += ["", "# A50 search 1/3 epoch 확장과 새 A60 refit. 기존 .25 A60 weights를 이어 학습하지 않는다."]
    for ds in datasets:
        for block in ("E1", "E3", "E4"):
            if any(r["dataset"] == ds and r["block"] == block for r in registry.values()):
                lines.append(f"& $py -X utf8 -u deepen.py --dataset {ds} --block {block} --max-epoch 3")
    lines += ["", "# E1/E3 V 추론, E2 M-only 결합, E4 공통 prefix. 각 단계는 필요 부모가 있어야 실행 가능."]
    for ds in datasets:
        lines.append(f"& $py -X utf8 -u postfit.py --phase predict --dataset {ds}")
        if ds != "robin":
            for phase in ("combine", "controls", "ordered"):
                lines.append(f"& $py -X utf8 -u postfit.py --phase {phase} --dataset {ds} --epochs .25 1 3")
    lines += ["}", "", "# E0: 과거 B의 별도 개발 증거. 고정 E1-E4/T 선택에 되먹이지 않는다.",
              "# 이전 RNG/sampler가 없으므로 과거 구간 전체의 exact resume라고 부르지 않는다.",
              "& $py -X utf8 -u legacy_queue.py run --asset-root $assetRoot --cache-root $env:INFORMS_CACHE_ROOT --include-controls --endpoint 4800 --endpoint 9600 --endpoint 19200 --execute-training",
              "# BUDGET_LIMIT carry가 있으면 해당 budgets receipt의 --resume-budget-checkpoint 명령을 그대로 사용.",
              "", "# 실제 저장된 V 숫자 전체 감사. 이전 실패 snapshot을 덮지 않는다.",
              "& $py -X utf8 audit_campaign.py --public-root . --private-root $env:INFORMS_CACHE_ROOT", "",
              "# 현재 검산된 V route의 실제 비용. 이미 생성된 T 결과로 방법을 골라 비용 후보를 바꾸지 않는다."]
    for ds in datasets:
        ids = sorted({r["run_id"] for r in scores if r["dataset"] == ds and r["split"] == "V_select"})
        if ids:
            lines.append(f"& $py -X utf8 -u cost_campaign.py --dataset {ds} " + " ".join("--run-id " + run_id for run_id in ids))
    lines += ["", "# T: 기존의 유효한 SELECTION_LOCK이 있을 때만 실행. 여기서 T 후보를 새로 선택하지 않는다.",
              "@'", "from pathlib import Path", "import os", "import final_evaluation as f", "from resources import GPUReservation",
              "p = Path.cwd(); c = Path(os.environ['INFORMS_CACHE_ROOT'])", "assert (p/'SELECTION_LOCK.json').is_file(), '먼저 V/비용 근거로 별도 동결된 선택이 필요함'",
              "with GPUReservation(c, total_gpu_seconds=43200, reserve_seconds=0, wall_deadline=1792050350+172800, purpose='audit', kind='locked_T_final') as lease:",
              "    print(f.execute_T(c, p, lease, lease.deadline-5))", "'@ | & $py -X utf8 -", "",
              "# 독립 검산 후 그림/보고서를 새 snapshot으로 갱신.",
              "& $py -X utf8 report_figures.py --public-root . --private-root $env:INFORMS_CACHE_ROOT --split V_select",
              "& $py -X utf8 report_campaign.py --public-root . --private-root $env:INFORMS_CACHE_ROOT"]
    return "\n".join(lines) + "\n"


def _render(directory, public, registry, summary, scores, coverage, fits, legacy, matched, evidence, figure, issues):
    now = datetime.fromtimestamp(summary["created_unix"], timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = ["# INFORMS 전체 실험 중간 검산 보고서", "", f"작성 시점: {now}. 상태: **PARTIAL**. 이 snapshot 생성은 학습·실험 완료 선언이 아니다.", "",
        "## 먼저 답할 수 있는 것과 아직 답할 수 없는 것", "",
        f"[확인] 등록 {len(registry)}개 중 저장된 예측을 독립 FP64로 검산한 V route는 {summary['numeric_V_run_count']}개, T route는 {summary['numeric_T_run_count']}개다. "
        "실행 상태, 학습 노출량, 숫자 유효성, 연구적 가치, main 게시 상태를 별도로 기록했다.", "",
        "과거 입력만으로 만든 표현이 실제 미래 타깃을 더 잘 예측하는지는 동일 데이터·K·seed·학습량 비교로 판단한다. "
        "재구성 성공, 누출 검사 통과, 학습 receipt 생성만으로 예측 개선이나 독립 일반화를 주장하지 않는다. "
        "현재 결과가 나쁘더라도 짧은 screen·마지막 checkpoint 최적·추가 하강 징후가 있으면 방법의 수렴 후 한계로 결론내릴 수 없다.", "",
        "## 데이터와 비교 계약", "",
        "입력은 과거 512 step만 사용한다. 새 캠페인의 허용 prefix를 시간순 A_fit 50% / A_stop 10% / M_cal 10% / V_select 10% / T_final 20%로 나눈다. "
        "하이퍼파라미터/epoch는 A_stop, 결합 파라미터는 M_cal, 방법 선택은 V_select에서 정한다. A50 search 뒤 선택 epoch를 새 A60 scaler와 초기화로 refit한다. "
        "M의 dense origin과 V/T의 horizon-stride origin을 섞지 않는다. 주지표는 관측 mask를 적용한 A60 정규화 원 채널 macro MSE이며 MAE도 함께 기록한다.", "",
        "G나 잔차 anchor는 원 C 채널을 읽을 수 있으므로 F에 K행만 넣었다고 전체 파이프라인이 K개의 정보만 사용한다고 부르지 않는다. "
        "공정성은 동일 목표·분할·mask·scaler·seed·학습 상한과 전체 비용을 함께 확인해야 한다. 고정 모델의 추론 시간과 cache 읽기 시간은 구분한다.", "",
        "새 T는 과거 보호 TEST와 별개이며, 기존 자료를 허용 prefix 내부에서 다시 분할한 것이다. Beijing/Jena/Robin 등에는 연구자가 이전에 본 데이터가 포함돼 독립적으로 처음 본 일반화 검증이라 주장할 수 없다. "
        "Jena/Robin의 일부 원 단위 값은 기존 FP32 역정규화 계보를 가진다. Electricity는 표준 321채널 hourly source를 쓰며 보호 cutoff 뒤 타깃을 열지 않는다. "
        "PeMS08에는 실제 달력 timestamp가 없어 경과 step/초 축을 사용한다. M5 공식 접근은 401과 로컬 자산 부재로 BLOCKED이며 비공식 우회·동의는 수행하지 않는다.", "",
        "과거 정보 사용 계약의 단위 검사와 foundation model의 사전학습 데이터 계보는 별개다. 이 캠페인은 사전학습 corpus와 평가 자료의 중복 여부를 독립 감사하지 않았으며, "
        "중복이 있다고도 없다고도 주장하지 않는다. F 내부에서는 quantile 0.5 출력을 사용하지만 decoder·결합을 거친 원 C 채널 출력이 조건부 중앙값이라는 보장은 없다. "
        "따라서 최종 출력은 deterministic point forecast로 표시하고 F의 point 선택을 별도 필드에 기록한다.", "",
        "## 상태를 서로 구분한 집계", "", "```json", json.dumps(summary["status_counts"], ensure_ascii=False, indent=2), "```", "",
        "위 집계는 registry에 기록된 상태이며 보고서가 쓰거나 승격하지 않았다. 독립 숫자 재검산은 별도 집계다. "
        f"독립 검산 score 행 {len(scores)}개, 검산 배제 입력 {summary['excluded_prediction_count']}개. "
        "registry PENDING과 숫자 PASS가 동시에 있으면 숫자는 보존하되 전체 캠페인 감사·완료로 바꾸지 않는다.", "",
        "## E0: 기존 미해결 학습의 연장", "",
        f"대상 ledger {len(legacy)}개. 실제 새 endpoint receipt가 있는 대상은 {sum(bool(row['completed_endpoints']) for row in legacy)}개다. "
        "과거 2400 update의 weights와 Adam state를 재사용하더라도 당시 RNG/sampler가 없어 최초 연장을 RESTARTED_NOT_EXACT_RESUME로 표시한다. "
        "새 연장 구간의 full carry checkpoint와 exact resume는 별도 계약이다. B는 개발 데이터이며 새 V/T 일반화 수치와 합치지 않는다. "
        "4800/9600/19200 완료, 마지막 최적 여부, 학습 곡선, 실제 저장 checkpoint B replay를 모두 확인해야 수렴 논의를 좁힐 수 있다.", "",
        "[E0 전체 ledger](E0_LEDGER.csv) · [E0 별도 재검산 수치](E0_EVALUATION_SCORES.csv)", ""]
    for block, purpose in (("E1", "독립 K 압축·시간 인코더·직접 예측·LoRA·고정 PCA의 실제 예측 비교"),
                           ("E2", "동일 frozen 부모에서 α, 조건부 결합, 과거 대조, Raw 보정과 A+M 재학습을 분리"),
                           ("E3", "과거 범위의 인과 지연과 양방향 시간 혼합의 유용성·표현 일관성 비교"),
                           ("E4", "한 checkpoint의 ordered prefix와 독립 K·max-only·nested PCA의 효율 비교")):
        ids = {run_id for run_id, row in registry.items() if row["block"] == block}
        available = [row for row in scores if row["run_id"] in ids]
        lines += [f"## {block}: {purpose}", "", f"등록 {len(ids)}개, 완료 fit receipt {sum(run_id in fits for run_id in ids)}개, "
                  f"독립 검산 V ID {len({r['run_id'] for r in available if r['split']=='V_select'})}개, "
                  f"T ID {len({r['run_id'] for r in available if r['split']=='T_final'})}개.", ""]
        if block == "E2":
            lines += ["M에서 캐시 부모 예측으로 H를 학습한 비용의 F=0은 배포 F=0을 뜻하지 않는다. 동일 부모 SHA·M/V·loss·checkpoint 선택을 맞춘 비교만 차이로 해석한다. "
                      "Raw 보정, G 단독, A+M 재학습, α=0/0.5/1은 각기 다른 가설을 검사한다. 제안 head가 더 좋아도 전체 경로 비용과 단순 대안 차이가 남아 있다.", ""]
        if block == "E3":
            lines += ["과거 창 내부의 bidirectional 접근이나 prefix 변화는 미래 타깃 누출과 다르다. 시점별 과거 전용 표현 일관성, 예측 오차, 지연 패턴의 추가 효용을 분리한다. "
                      "다른 채널 lag 상관은 A_fit에서 측정한 연관성이며 인과효과나 추가 예측력의 증거가 아니다. 대조학습·동적 그래프 필요성은 단순 대조와의 실제 차이가 있어야 주장할 수 있다.", ""]
        if block == "E4":
            lines += ["공유 max-K 학습과 독립 K 학습은 각 K의 노출량 및 PCA 초기화가 다를 수 있다. nested PCA와 prefix decoder의 실제 K-row F 추론을 별도로 확인한다. "
                      "검색 update가 같아도 GPU 시간이나 K별 노출이 같다고 부르지 않는다. 사후 잘라낸 max-only prefix와 학습 중 모든 K를 본 ordered model을 구분한다.", ""]
        if not available:
            lines += ["이 snapshot에는 이 블록의 검산된 V/T 숫자가 없다. 계획·학습 시작·코드 테스트를 실제 예측 결과로 대체하지 않았다.", ""]
        else:
            for row in sorted(available, key=lambda r: (r["dataset"], r["split"], r["latent_k"], r["family"], r["seed"], r["manifest_relative_path"])):
                lines.append(f"- `{row['run_id']}` · {row['split']} · K={row['latent_k']} · lr={row['learning_rate']} · max epoch={row['max_epoch']} / selected={row['selected_epoch']} · "
                             f"MSE={row['mse']:.9g}, MAE={row['mae']:.9g}, 관측={row['n_observed']}, origin={row['n_origins']} · {row['training_status']}")
            lines.append("")
    lines += ["## 같은 조건에서 확인한 차이", "", f"엄격한 조건 키로 묶인 비교 {len(matched)}개. 음수 ΔMSE는 compared 방법의 MSE 감소를 뜻한다. "
              "불확실성: 현재 seed·origin 표본의 범위이며 유의성, 다중 비교 보정, 독립 unseen 데이터의 재현을 주장하지 않는다. "
              "같은 조건 짝이 없으면 데이터 간 평균이나 가장 좋은 seed로 대신하지 않았다.", "", "[조건과 원수치 전체](MATCHED_COMPARISONS.csv)", ""]
    lines += ["학습률은 실제 SOURCE/CONTROL_SOURCE spec 또는 fit receipt에서 읽는다. 학습 모델 간 비교는 같은 기록 LR만 묶으며 "
              "LR가 확인되지 않은 학습 모델은 원수치를 유지하고 matched 비교에서 제외해 INPUT_ISSUES에 표시한다. "
              "닫힌 형식 α는 optimizer LR가 적용되지 않는 기준선이므로 동일 frozen 부모를 요구하고 비교 head의 실제 LR를 별도 기록한다.", ""]
    for row in matched:
        lines.append(f"- `{row['compared_run_id']}` − `{row['reference_run_id']}`: ΔMSE={row['delta_mse']:+.9g}, ΔMAE={row['delta_mae']:+.9g} ({row['split']}, K={row['latent_k']}, seed={row['seed']}, max epoch={row['max_epoch']}, lr={row['learning_rate']}, {row['learning_rate_contract']}).")
    resource = summary["resource_usage"]
    lines += ["", "## 비용과 불확실성", "", f"실제 trial을 다시 집계한 비용 조건 {summary['verified_cost_condition_count']}개, "
              f"저장 checkpoint V replay 검산 {summary['verified_replay_count']}개. 비용이 없으면 정확도–비용 우위를 주장하지 않는다. "
              "fresh process B=1/8 전체 물리 입력→전처리→P/E/F/D→역변환→CPU 출력과 shared-F balanced 비용을 분리한다. "
              "최소 10 warmup·20 timed·20 stage trial, Raw row chunk 및 LoRA merge/unmerge parity가 필요하다. RSS/VRAM은 공유와 독점 범위가 다르며 학습 peak와 추론 peak도 다르다.", "",
              f"종료 lease의 GPU process-wall 누계 {resource['closed_GPU_process_seconds']:.3f}초. "
              f"닫히지 않은 live 구간 추정 {resource['live_unclosed_seconds_excluded_from_closed_total']:.3f}초는 종료 누계에 포함하지 않았다. "
              "process-wall은 실제 GPU 연산 시간/사용률과 다르다. 잔여 추정은 저장 시점 값이며 다음 작업의 독립 승인이나 완수 보장이 아니다.", "",
              "## 실패와 남은 확인", "",
              f"이번 compiler 배제/검사 이슈 {len(issues)}개. 원인별 파일은 [INPUT_ISSUES.csv](INPUT_ISSUES.csv)에 남겼다. "
              "실패한 실행과 이전 실패 감사는 지우지 않고 [EVIDENCE_INVENTORY.csv](EVIDENCE_INVENTORY.csv)에 보존했다. "
              "숫자 부재는 실패한 연구 가설과 구분한다. 지연·다중 K·결합을 앞 성능이 낮다는 이유로 생략한 것으로 표시하지 않으며, 실행되지 않은 경로는 그대로 미실행이다.", "",
              "현 단계에서 필요한 순서는 남은 최소 fit/추론과 1/3 epoch 학습량 확인 → 동일 부모/학습량의 E2/E3/E4 비교 → 실제 비용·checkpoint replay → "
              "독립 수치·registry 감사 → V와 비용 근거를 동결한 뒤 허용 route의 T 실행이다. 대조학습·동적 그래프·Chronos-Bolt 유지 근거는 "
              "단순 과거 예측기, 고정/학습 압축, G 및 보정 대조의 실제 정확도·전체 비용을 넘어서야 한다. 아직 없는 결과를 가정해 필요성·불필요성을 결론내리지 않는다.", "",
              "## 검증된 시각 자료", ""]
    if figure:
        path, receipt = figure
        lines += [f"모든 output 파일 SHA가 맞는 최신 완결 저장 snapshot을 연결했다. 그림 snapshot 자체의 split은 `{receipt['split']}`이며 "
                  "현재 보고서와 생성 시점이 다를 수 있다. 완결 저장은 전체 실험 완료를 뜻하지 않는다.", "",
                  f"[그림 설명]({_link(directory, path.with_name('README_KO.md'), public)})", ""]
        for group in receipt["groups"]:
            png = next((name for name in group["figures"] if name.endswith(".png")), None)
            if png:
                lines += [f"{group['group']} · {group['status']}", "", f"![{group['group']}]({_link(directory, path.parent / png, public)})", ""]
            else:
                lines += [f"- {group['group']}: {group['status']} — 저장된 검증 그림 없음."]
    else:
        lines += ["아직 파일 SHA를 모두 확인한 그림 snapshot이 없어 이미지를 첨부하지 않았다.", ""]
    lines += ["", "## 출처·재현 자료", "", "[전 조건 원수치](SCORES.csv) · [데이터·K별 coverage](COVERAGE.csv) · "
              "[학습 receipt 목록](FIT_RECEIPTS.csv) · [비용 재집계](COSTS.csv) · [replay 검산](REPLAYS.csv) · [재개 명령](RESUME_COMMANDS.ps1)", ""]
    pins_path = public / "LITERATURE_MODEL_PINS.json"
    if pins_path.is_file():
        for source in _load(pins_path).get("sources", []):
            lines.append(f"- [{source['name']} 공식 원본]({source['repository']}/tree/{source['revision']}) · pin `{source['revision']}`. 포팅의 원리 출처이며 공식 수치 재현 완료 주장은 아니다.")
    for filename, label in (("DATA_CONTRACT.json", "실제 데이터 계약"), ("CONFIG_LOCK.json", "고정 실험 설정"), ("MODEL_ASSETS.json", "검증 모델 자산")):
        if (public / filename).is_file():
            lines.append(f"- [{label}]({_link(directory, public / filename, public)})")
    lines += ["", "원본 데이터·타깃 배열·모델 weights·인증 토큰·개인 상담 원문은 이 보고서에 복사하지 않았다. "
              "main 게시 상태는 registry에서 읽은 사실만 제시하며 보고서 생성이 push나 원격 SHA 검증을 뜻하지 않는다.", ""]
    (directory / "REPORT_KO.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "README_KO.md").write_text("# INFORMS 실험 검산 snapshot\n\n**PARTIAL** — 실험은 진행 중이며 전체 완료 보고가 아니다.\n\n"
        f"- 독립 검산 V {summary['numeric_V_run_count']} route / T {summary['numeric_T_run_count']} route\n"
        f"- 등록 {len(registry)}개 / 완료 fit receipt {len(fits)}개 / 배제 예측 {summary['excluded_prediction_count']}개\n"
        "- [상세 한국어 보고서](REPORT_KO.md)\n- [모든 원수치](SCORES.csv)\n- [분리된 상태와 자원 집계](SUMMARY.json)\n"
        "- [누락과 재개 방법](RESUME_COMMANDS.ps1)\n\n숫자 검산 통과는 수렴·연구 성공·독립 일반화·main 게시 완료와 다르다.\n", encoding="utf-8")


def generate(public_root, cache_root):
    public, cache = Path(public_root).resolve(), Path(cache_root).resolve()
    started = time.time()
    directory = public / "reports" / ("snapshot_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + str(time.time_ns()))
    directory.mkdir(parents=True, exist_ok=False)
    inputs, issues = {}, []
    initial, declared, current, extensions = audit_campaign._registry_contract(public, issues)
    config_path = public / "CONFIG_LOCK.json"
    config = _load(config_path) if config_path.is_file() else {}
    if config.get("registry_rows", len(initial)) != len(initial):
        issues.append(dict(scope="initial_registry", reason="Initial registry count differs from frozen CONFIG_LOCK"))
    invalid_ids = {issue.get("run_id") for issue in issues if issue.get("scope")=="registry_contract"}
    registry = {row["experiment_id"]: row for row in current
                if row["experiment_id"] in {d["experiment_id"] for d in declared} and row["experiment_id"] not in invalid_ids}
    for path in (public / "EXPERIMENT_REGISTRY.csv", public / "EXPERIMENT_REGISTRY_INITIAL.csv", config_path,
                 public / "DATA_CONTRACT.json", public / "LITERATURE_MODEL_PINS.json", public / "MODEL_ASSETS.json"):
        if path.is_file(): _input(inputs, path, public)
    for extension in extensions:
        path = public / extension["csv"]
        _input(inputs, path, public); _input(inputs, path.with_suffix(".json"), public)
    lock_path = public / "SELECTION_LOCK.json"
    lock = _load(lock_path) if lock_path.is_file() else {}
    if lock_path.is_file(): _input(inputs, lock_path, public)
    fits, fit_inventory = _fits(public, cache, inputs, issues)
    fits = {run_id:receipt for run_id,receipt in fits.items() if run_id in registry}
    scores = _predictions(public, cache, registry, fits, inputs, issues, lock)
    costs, replays = _costs(public, cache, registry, inputs, issues), _replays(public, cache, inputs, issues)
    legacy, legacy_scores = _legacy(public, cache, inputs, issues)
    evidence = _evidence_inventory(public, inputs, issues)
    figure = _figures(public, inputs, issues)
    resources = _resource_snapshot(public, cache, inputs, issues)
    for row in scores:
        row["cost_references"] = ";".join(sorted({c["receipt_relative_path"] for c in costs if c["run_id"] == row["run_id"]}))
        statuses = [r["status"] for r in replays if r["run_id"] == row["run_id"]]
        row["checkpoint_replay"] = ";".join(statuses) if statuses else "NOT_AVAILABLE"
    coverage, matched = _coverage(registry, fits, scores), _matched(scores, issues)
    run_index = _run_index(registry, fits, scores)
    prediction_index = _prediction_index(public, scores, inputs, issues)
    claims = _claims(scores, matched, costs, replays, fits, legacy)
    queue = _resume_queue(registry, fits, scores, cache, lock)
    queue["legacy_targets"] = [dict(row, next_endpoint=next((n for n in (4800,9600,19200) if str(n) not in row["completed_endpoints"].split(";")), None),
        command=f"& $py -X utf8 -u legacy_queue.py run --asset-root $assetRoot --cache-root $env:INFORMS_CACHE_ROOT --id {row['legacy_id']} --endpoint 4800 --endpoint 9600 --endpoint 19200 --execute-training")
        for row in legacy if "19200" not in row["completed_endpoints"].split(";")]
    queue["budget_receipts"] = []
    for path in sorted((public/"legacy_e0/budgets").glob("*.json")):
        receipt = _load(path)
        queue["budget_receipts"].append({key:receipt.get(key) for key in ("id","endpoint","status","resume_command","budget_checkpoint_cache_relative_path")})
        _input(inputs,path,public)
    status_counts = {key: dict(sorted(Counter(row.get(key, "UNRECORDED") for row in current).items())) for key in STATUS_FIELDS}
    summary = dict(status="PARTIAL_SNAPSHOT", created_unix=time.time(), snapshot=directory.name,
        registry_rows=len(current), initial_registry_rows=len(initial), registered_extension_rows=len(declared)-len(initial),
        frozen_expected_initial_rows=config.get("registry_rows"), initial_registry_expected_391=len(initial)==391,
        status_counts=status_counts, complete_fit_receipts=len(fits), fit_receipts_seen=len(fit_inventory),
        observed_fit_training_status_counts=dict(Counter(row["training_status"] for row in fits.values())),
        numeric_V_run_count=len({r["run_id"] for r in scores if r["split"] == "V_select"}),
        numeric_T_run_count=len({r["run_id"] for r in scores if r["split"] == "T_final"}),
        numeric_score_rows=len(scores), excluded_prediction_count=sum(row.get("scope")=="prediction" for row in issues),
        registry_unreconciled_numeric_rows=sum(not row["registry_reconciled"] for row in scores),
        verified_cost_condition_count=sum(row["accuracy_binding_status"]=="BOUND_EXACT_ROUTE" for row in costs), measured_cost_condition_count=len(costs),
        verified_replay_count=len(replays), E0_ledger_rows=len(legacy),
        E0_actual_completed_targets=sum(bool(row["completed_endpoints"]) for row in legacy), E0_rescored_B_rows=len(legacy_scores),
        resource_usage=resources, issue_count=len(issues), latest_verified_figure_snapshot=figure[0].parent.name if figure else None,
        campaign_complete=False, converged_model_claimed=False, scientific_success_claimed=False,
        independent_generalization_claimed=False, publication_verified_here=False,
        registry_mutated=False, training_executed=False, inference_executed=False, T_selected=False,
        existing_selection_locked=lock.get("locked") is True, training_resume_scope="PRE_SELECTION_ONLY",
        T_private_cache_opened=any(row["split"]=="T_final" for row in scores),
        scoring="Independent saved-prediction FP64 channel macro + saved independent truth seal + public sufficient-statistics reaggregation",
        snapshot_atomicity="Captured over a read interval; active writers may create later files; immutable artifacts selected by exact hashes",
        started_unix=started)
    _write_csv(directory / "SCORES.csv", scores, SCORE_FIELDS)
    for filename, rows in (("COVERAGE.csv", coverage), ("FIT_RECEIPTS.csv", fit_inventory), ("MATCHED_COMPARISONS.csv", matched),
                           ("COSTS.csv", costs), ("REPLAYS.csv", replays), ("E0_LEDGER.csv", legacy),
                           ("E0_EVALUATION_SCORES.csv", legacy_scores), ("EVIDENCE_INVENTORY.csv", evidence), ("INPUT_ISSUES.csv", issues),
                           ("RUN_INDEX.csv", run_index), ("PREDICTION_MANIFEST.csv", prediction_index), ("COST.csv", costs), ("CLAIM_EVIDENCE.csv", claims)):
        _write_csv(directory / filename, rows)
    _write_json(directory / "SUMMARY.json", summary)
    _write_json(directory / "RESUME_QUEUE.json", queue)
    (directory / "RESUME_COMMANDS.ps1").write_text(_resume_commands(registry, scores), encoding="utf-8")
    _render(directory, public, registry, summary, scores, coverage, fits, legacy, matched, evidence, figure, issues)
    _supporting_docs(directory, summary, claims, queue)
    _write_json(directory / "INPUT_MANIFEST.json", dict(inputs=list(inputs.values()), raw_arrays_published=False,
        private_authentication_tokens_copied=False, absolute_private_paths_published=False))
    outputs = [dict(file=path.name, sha256=data.sha256_file(path), bytes=path.stat().st_size) for path in sorted(directory.iterdir()) if path.is_file()]
    _write_json(directory / "REPORT_MANIFEST.json", dict(status="PARTIAL_SNAPSHOT_WRITTEN", source_sha256=data.sha256_file(__file__),
        dependency_sha256={name: data.sha256_file(Path(__file__).with_name(name)) for name in ("audit_campaign.py", "audit_predictions.py", "data_contract.py", "training.py")},
        outputs=outputs, started_unix=started, completed_unix=time.time(), registry_mutated=False, GPU_execution=False,
        independent_prediction_rescoring=True, campaign_completion_claimed=False, main_publication_claimed=False))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--private-root", required=True)
    args = parser.parse_args()
    print(generate(args.public_root, args.private_root))


if __name__ == "__main__":
    main()
