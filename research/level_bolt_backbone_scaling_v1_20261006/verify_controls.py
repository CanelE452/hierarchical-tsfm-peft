"""CPU artifact audit; no model loading, inference, fitting, scoring, or bootstrap."""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
import traceback

import numpy as np

from runtime import (CACHE, DATASETS, HERE, LIMITS, ROOT, Job, artifact,
                     budget_snapshot, check_seal, digest, public_paths, read_json, save_json, storage_bytes)


ARMS = ("BOLT_SMALL_LEVEL", "BOLT_SMALL_F0", "BOLT_MINI_F0", "BOLT_TINY_F0")
NEW_MODELS = {"BOLT_MINI_F0": "MINI", "BOLT_TINY_F0": "TINY"}
COUNTS = dict(zip(ARMS, (2, 1, 1, 1)))
PERIODS = ("test_a", "test_b", "combined")
METRICS = ("mse", "mae", "signed_mean_error")
PAIRS = ((ARMS[0], ARMS[2]), (ARMS[0], ARMS[3]), (ARMS[0], ARMS[1]),
         (ARMS[1], ARMS[2]), (ARMS[1], ARMS[3]), (ARMS[2], ARMS[3]))
TOLERANCE = 1e-11
SAVED_POOLING_RTOL = 64 * np.finfo(np.float64).eps


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and abs(float(actual) - float(expected)) <= TOLERANCE, label + " differs")


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def check_artifact(receipt):
    value = artifact(path_of(receipt["path"]), receipt["sha256"])
    if "bytes" in receipt:
        require(value["bytes"] == receipt["bytes"], "Artifact byte count changed: " + receipt["path"])
    return value


def artifact_objects(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for item in value.values():
            yield from artifact_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from artifact_objects(item)


def table(name):
    with (HERE / name).open(encoding="utf-8", newline="") as stream:
        result = []
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                elif key in {"dataset", "method", "id", "period", "original_method", "channel", "source_kind",
                             "model_key", "device", "execution", "key", "aggregation", "row_type", "status", "sentinel"}:
                    row[key] = value
                elif value in ("True", "False"):
                    row[key] = value == "True"
                else:
                    try:
                        row[key] = json.loads(value)
                    except (ValueError, TypeError):
                        row[key] = value
            result.append(row)
    return result


def one(rows, **keys):
    matches = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    require(len(matches) == 1, "Missing/duplicate row: " + repr(keys))
    return matches[0]


def canonical_origins(unit):
    return np.asarray(unit["origins"]["test_a"]["values"] + unit["origins"]["test_b"]["values"], dtype=np.int64)


def load_archive(receipt):
    check_artifact(receipt)
    with np.load(path_of(receipt["path"]), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def verify_protection(job):
    selected = read_json(HERE / "protected_before.json")
    require(selected["new_fits"] == 0, "Original artifact protection records a new fit")
    for index, receipt in enumerate(selected["files"]):
        check_artifact(receipt)
        if index % 20 == 0:
            job.heartbeat({"phase": "selected_old_artifact_protection", "checked": index})
    guard_path = HERE / "protected_workspace_before.json"
    guard = read_json(guard_path)
    require(guard["stage"] == "After implementation and preflight; before new TEST and cost, not an initial-session snapshot",
            "Workspace guard stage is unclear")
    allowed = set(guard["permitted_tracked_edit"])
    require(allowed == {"README.md"}, "Workspace protection exceptions changed")
    changed_allowed, failures, checked = [], [], 0
    for name, before in guard["files"].items():
        source = path_of(name)
        if not source.is_file():
            failures.append({"path": name, "reason": "missing"})
        else:
            after = digest(source)
            if after != before["sha256"]:
                if name in allowed:
                    changed_allowed.append({"path": name, "before_sha256": before["sha256"],
                                            "after_sha256": after,
                                            "scope": "Authorized related README publication edit; excluded from immutable claim"})
                else:
                    failures.append({"path": name, "reason": "hash_changed"})
            checked += 1
        if checked % 250 == 0:
            job.heartbeat({"phase": "pre_test_workspace_guard", "checked": checked,
                           "total": len(guard["files"])})
    require(not failures, "Protected workspace files changed: " + repr(failures[:10]))
    require(all(name in guard["files"] and name not in allowed for name in guard["presentation_files"]),
            "An original presentation file is outside the declared protected set")
    return {"selected_old_artifacts_checked": len(selected["files"]),
            "selected_snapshot": artifact(HERE / "protected_before.json"),
            "workspace_snapshot_sha256": digest(guard_path), "workspace_snapshot_local_only": True,
            "workspace_protected_files": len(guard["files"]), "workspace_actual_files_checked": checked,
            "original_presentation_files_checked": len(guard["presentation_files"]),
            "allowed_related_changes": changed_allowed,
            "excluded_paths_count": len(guard["excluded"]), "scope": guard["scope"], "stage": guard["stage"],
            "full_initial_session_or_unknown_external_file_protection_claimed": False,
            "external_raw_sources_reread": False}


def verify_seal_and_models(job):
    seal = check_seal()
    required = {"input_contract.json", "reuse_manifest.json", "model_manifest.json", "preflight_checks.json",
                "runtime.py", "baselines.py", "bolt_models.py", "preflight.py", "evaluate_controls.py", "cost_controls.py"}
    require(required.issubset(seal["files"]), "TEST seal omits a scientific implementation/data/model binding")
    request = read_json(HERE / "request_binding.json")
    require(request["user_adopted_contract"] is True and request["new_fitting_authorization"] == 0,
            "User contract/zero-fitting authorization changed")
    check_artifact(request["repository_copy"])
    documents = [read_json(HERE / name) for name in ("input_contract.json", "reuse_manifest.json", "model_manifest.json")]
    external, unique = [], {}
    for document in documents:
        for receipt in artifact_objects(document):
            if receipt["path"].startswith(("<external-source>", "<external-local-path>")):
                external.append(receipt["path"])
            else:
                unique[(receipt["path"], receipt["sha256"])] = receipt
    for index, receipt in enumerate(unique.values()):
        check_artifact(receipt)
        if index % 20 == 0:
            job.heartbeat({"phase": "sealed_model_data_source_hashes", "checked": index})
    models = documents[2]["models"]
    require(set(models) == {"SMALL", "MINI", "TINY"}, "Official three-model family changed")
    preflight = read_json(HERE / "preflight_checks.json")
    for key, info in models.items():
        require(info["id"] == "amazon/chronos-bolt-" + key.lower()
                and re.fullmatch("[0-9a-f]{40}", info["revision"]), "Official immutable model pin is missing")
        directory = path_of(info["checkpoint_dir"])
        require(directory.is_dir(), "Pinned local model directory is missing")
        for name, receipt in info["files"].items():
            require(Path(name).name == name, "Model filename is not a local basename")
            actual = check_artifact(receipt)
            artifact(directory / name, receipt["sha256"])
            if name == "config.json":
                content = (directory / name).read_bytes()
                official_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                require(official_blob == receipt["official_git_blob"], "Official config Git blob differs")
            else:
                require(receipt["official_lfs_sha256"] == actual["sha256"], "Official model LFS SHA differs")
        config = read_json(directory / "config.json")
        chronos = config["chronos_config"]
        require(config["architectures"] == ["ChronosBoltModelForForecasting"]
                and chronos["context_length"] >= 512 and chronos["prediction_length"] >= 48
                and chronos["quantiles"].count(0.5) == 1, "Official native L/H/q contract failed")
        receipt = one(preflight["bolt_models"], model=key)["receipt"]
        require(info["loaded_model_receipt"] == receipt, "Manifest does not preserve actual loaded model receipt")
        require(receipt["parameter_count"] > 0 and receipt["parameter_bytes"] == 4 * receipt["parameter_count"]
                and receipt["buffer_bytes"] >= 0
                and receipt["total_tensor_bytes"] == receipt["parameter_bytes"] + receipt["buffer_bytes"]
                and receipt["trainable_parameter_count"] == 0 and receipt["eval_mode"] is True,
                "Actual loaded FP32 parameter/buffer/frozen accounting is invalid")
        require(receipt["architecture"] == {name: config[name] for name in
                ("d_model", "d_ff", "num_layers", "num_decoder_layers", "num_heads")},
                "Loaded backbone architecture differs from official config")
        require(receipt["quantiles"] == chronos["quantiles"]
                and receipt["quantiles"][receipt["median_index"]] == 0.5,
                "Actual native median/config binding changed")
    bridge = read_json(HERE / "preflight_model_binding_update.json")
    original = check_artifact(bridge["preserved_preflight_input_manifest"])
    check_artifact(bridge["model_manifest_after"])
    check_artifact(bridge["native_receipts_source"])
    require(original["sha256"] == bridge["preflight_model_manifest_before"]["sha256"]
            == preflight["bindings"]["model_manifest.json"]["sha256"] and bridge["fit_count"] == 0,
            "Preflight-to-seal actual-model-receipt update lacks its preserved pre-update manifest")
    old_models = read_json(path_of(bridge["preserved_preflight_input_manifest"]["path"]))["models"]
    require(all(all(models[key][field] == old_models[key][field] for field in
                    ("id", "revision", "files", "config", "chronos_config", "checkpoint_dir")) for key in models),
            "Preflight receipt augmentation changed a scientific model pin/config/file")
    reuse_math = read_json(HERE / "scoring_math_reuse.json")
    check_artifact(reuse_math["original_module"])
    check_artifact(reuse_math["new_module"])
    source_trees = [ast.parse(path_of(reuse_math[name]["path"]).read_text(encoding="utf-8"))
                    for name in ("original_module", "new_module")]
    functions = [{node.name: ast.dump(node, include_attributes=False) for node in tree.body
                  if isinstance(node, ast.FunctionDef)} for tree in source_trees]
    require({row["function"] for row in reuse_math["functions"]}
            == {"origin_contract", "error_terms", "score_terms", "mean_seed_terms"}
            and all(row["AST_identical"] and functions[0][row["function"]] == functions[1][row["function"]]
                    for row in reuse_math["functions"]), "Original score arithmetic AST reuse review failed")
    require(reuse_math["paired_bootstrap"]["weights_and_sampled_statistics_AST_identical"] is True
            and reuse_math["paired_bootstrap"]["draws"] == 2000 and reuse_math["fit_count"] == 0
            and reuse_math["new_forward"] is False, "Paired bootstrap reuse review failed")
    return {"seal": artifact(HERE / "evaluation_seal.json"), "sealed_files": len(seal["files"]),
            "bound_artifact_hashes_checked": len(unique), "external_declarations_not_reread": len(external),
            "official_models": {key: {"id": info["id"], "revision": info["revision"],
                                      "actual_parameter_count": info["loaded_model_receipt"]["parameter_count"]}
                                for key, info in models.items()},
            "preflight_manifest_receipt_update_verified": True,
            "score_math_review": artifact(HERE / "scoring_math_reuse.json"),
            "no_model_loading_or_forward": True}


def verify_preflight(job):
    report = read_json(HERE / "preflight_checks.json")
    require(report["status"] == "pass" and report["small_anchor_all_datasets_passed"] is True,
            "SMALL anchor or preflight did not pass")
    require(report["fits"] == report["optimizer_updates"] == 0 and report["test_scoring_performed"] is False,
            "Preflight violated zero-fit/VAL-only scope")
    require(report["tolerances"] == {"same_path": {"atol": 1e-6, "rtol": 0.0},
                                     "batch_merge": {"atol": 1e-5, "rtol": 1e-4}}, "Parity tolerance changed")
    require(report["baseline_instances_checked"] == len(report["baselines"]) == 9
            and report["mini_tiny_dataset_units_checked"] == 6, "Preflight coverage changed")
    unit_contracts = read_json(HERE / "input_contract.json")["units"]
    unavailable = []
    for row in report["baselines"]:
        require(row["state_sha256_before"] == row["state_sha256_after"]
                and row["checks"] and all(check["status"] == "pass" for check in row["checks"]),
                "Original selected baseline frozen/parity check failed")
        parity = row["saved_val_parity"]
        if parity["status"] == "unavailable":
            require(row["dataset"] == "peacock_education" and row["method"] == "F0"
                    and parity["substitution_used"] is False, "Stored VAL exception was extended")
            unavailable.append({"dataset": row["dataset"], "method": row["method"], "reason": parity["reason"]})
        else:
            require(parity["status"] == "pass", "Stored selected VAL prediction parity failed")
    require(set(row["model"] for row in report["bolt_models"]) == {"SMALL", "MINI", "TINY"},
            "Preflight lacks an official model")
    for model in report["bolt_models"]:
        require(model["status"] == "pass" and model["state_sha256_before"] == model["state_sha256_after"],
                "Preflight model state changed")
        require(len(model["units"]) == 3 and {unit["dataset"] for unit in model["units"]} == set(DATASETS),
                "Preflight dataset set changed")
        for unit in model["units"]:
            require(unit["origins"] == unit_contracts[unit["dataset"]]["origins"]["val"]["values"][:4]
                    and unit["status"] == "pass" and all(check["status"] == "pass" for check in unit["checks"]),
                    "First-four VAL parity/isolation/order/replay failed")
            native = unit["native_quantile"]
            require(native["status"] == "pass" and native["selection_parity"]["status"] == "pass"
                    and len(native["records"]) == 1, "Actual native quantile/row restoration is unverified")
            actual = native["records"][0]
            require(actual["actual_quantile"] == 0.5 and actual["row_order_bit_equal"] is True
                    and actual["actual_argument_names"] == ["context"]
                    and actual["target_provided"] is actual["future_covariates_provided"] is False,
                    "Native model saw an unapproved quantile/order/future argument")
            if model["model"] != "SMALL":
                require(unit["future_barrier"]["past_input_bit_identical"] is True
                        and unit["future_barrier"]["future_inputs_provided"] is False
                        and len(unit["checks"]) == 7, "Mini/Tiny future or isolation checks are missing")
        if model["model"] != "SMALL":
            require(len(model["reload_checks"]) == 3 and all(check["status"] == "pass" for check in model["reload_checks"])
                    and model["reload_state_sha256_before"] == model["reload_state_sha256_after"] == model["state_sha256_before"],
                    "Pinned reload replay/frozen state failed")
    return {"preflight": artifact(HERE / "preflight_checks.json"), "small_anchor_dataset_units": 3,
            "mini_tiny_dataset_units": 6, "selected_baseline_instances": 9,
            "allowed_stored_val_unavailability": unavailable,
            "native_quantile_channel_origin_future_and_frozen_checks": True}


def verify_data_and_predictions(job):
    contracts = read_json(HERE / "input_contract.json")["units"]
    reuse = read_json(HERE / "reuse_manifest.json")["units"]
    models = read_json(HERE / "model_manifest.json")["models"]
    manifest = read_json(HERE / "prediction_manifest.json")
    seal_hash = digest(HERE / "evaluation_seal.json")
    require(manifest["status"] == "complete" and manifest["evaluation_units"] == 6
            and manifest["additional_fit_count"] == 0 and manifest["prediction_batch_origins"] == 4
            and manifest["seal_sha256"] == seal_hash, "Fixed six-unit prediction phase is incomplete/unsealed")
    check_artifact(manifest["preflight"])
    require(set(manifest["units"]) == set(DATASETS) and set(manifest["state_preservation"]) == set(NEW_MODELS),
            "Canonical prediction model/dataset coverage changed")
    checked, old_checked, counts, mask_hashes = [], [], {}, {}
    for dataset in DATASETS:
        unit = contracts[dataset]
        require((unit["c"], unit["k"]) == {"robin": (17, 5), "peacock_education": (13, 4), "jena": (21, 6)}[dataset]
                and (unit["l"], unit["h"]) == (512, 48), "Dataset dimensions changed")
        archives = {}
        for kind in ("trainval", "test"):
            data = load_archive(unit[kind])
            archives[kind] = data
            contract = unit["array_contract"][kind]
            require(set(data) == set(contract["keys"]) and list(data["x"].shape) == contract["x_shape"]
                    and str(data["x"].dtype) == contract["x_dtype"]
                    and data["finite"].shape == data["x"].shape
                    and str(data["finite"].dtype) == contract["mask_dtype"]
                    and np.isfinite(data["x"]).all() and data["columns"].tolist() == unit["columns"],
                    "Existing standardized data/column/mask schema changed")
            require(data["basis"].shape == (unit["c"], unit["k"]), "Existing fixed PCA shape changed")
            for name, expected in contract["value_hashes"].items():
                require(array_hash(data[name]) == expected, "Existing TRAIN statistics/PCA/mask value hash changed")
        for name in ("columns", "mean", "std", "basis"):
            require(np.array_equal(archives["trainval"][name], archives["test"][name]),
                    "TRAIN/VAL and TEST use different frozen statistics/PCA/columns")
        for period, declared in unit["origins"].items():
            origins = archives[declared["array_source"]][declared["key"]]
            require(origins.dtype == np.int64 and len(origins) == declared["count"]
                    and np.array_equal(origins, np.asarray(declared["values"], dtype=np.int64))
                    and array_hash(origins) == declared["values_sha256"] and np.all(np.diff(origins) > 0),
                    "Existing origin order/count/hash changed")
        origins = canonical_origins(unit)
        require(np.all(np.diff(origins) > 0), "Canonical A/B origin concatenation is unordered")
        mask = np.stack([archives["test"]["finite"][o:o + 48] for o in origins])
        mask_hashes[dataset] = array_hash(mask)
        counts[dataset] = {"test_a": unit["origins"]["test_a"]["count"],
                           "test_b": unit["origins"]["test_b"]["count"], "combined": len(origins)}
        shape = (len(origins), 48, unit["c"])
        require(len(reuse[dataset]["rows"]) == 3 and Counter(row["method"] for row in reuse[dataset]["rows"])
                == {"LEVEL": 2, "F0": 1}, "Original LEVEL/F0 instance binding changed")
        require(set(manifest["units"][dataset]) == set(NEW_MODELS), "A Mini/Tiny prediction unit is missing")
        for old in reuse[dataset]["rows"]:
            receipt = old["test_prediction"]
            archive = load_archive(receipt)
            value, order = archive[receipt["key"]], archive[receipt["origins_key"]]
            require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
                    and np.array_equal(order, origins), "Original saved prediction binding changed")
            old_checked.append({"dataset": dataset, "id": old["id"], "shape": list(shape), "sha256": receipt["sha256"]})
        for method, model_key in NEW_MODELS.items():
            record = manifest["units"][dataset][method]
            require(read_json(HERE / "prediction_receipts" / f"{dataset}_{method}.json") == record,
                    "Individual prediction receipt differs from manifest")
            require(record["status"] == "complete" and record["dataset"] == dataset and record["method"] == method
                    and record["seed"] is None and record["additional_fit_count"] == 0
                    and record["model_key"] == model_key and record["model_id"] == models[model_key]["id"]
                    and record["revision"] == models[model_key]["revision"] and record["seal_sha256"] == seal_hash,
                    "Deterministic Mini/Tiny identity/pin/zero-fit binding changed")
            for name in ("model_manifest", "preflight", "data"):
                check_artifact(record[name])
            require(record["data"]["sha256"] == unit["test"]["sha256"], "Prediction uses another data artifact")
            receipt = record["prediction"]
            archive = load_archive(receipt)
            value, order = archive[receipt["key"]], archive[receipt["origins_key"]]
            require(value.shape == shape and receipt["shape"] == list(shape) and value.dtype == np.float32
                    and receipt["dtype"] == "float32" and np.isfinite(value).all() and np.array_equal(order, origins)
                    and record["origin_values_sha256"] == array_hash(origins), "New prediction shape/FP32/origin/hash failed")
            require(record["origin_shape_fp32_finite"] == {"origin_count": len(origins), "shape": list(shape),
                                                           "dtype": "float32", "finite": True}, "Prediction value receipt is inconsistent")
            model_receipt = record["model_receipt"]
            require(model_receipt["eval_mode"] is True and model_receipt["trainable_parameter_count"] == 0
                    and model_receipt["quantiles"][model_receipt["median_index"]] == 0.5
                    and model_receipt["load_arguments"]["local_files_only"] is True, "Prediction model was not frozen/native/pinned")
            for name in ("parameter_count", "parameter_bytes", "buffer_bytes", "total_tensor_bytes", "architecture",
                         "quantiles", "median_index", "model_context_length", "model_prediction_length"):
                require(model_receipt[name] == models[model_key]["loaded_model_receipt"][name],
                        "Prediction deployment receipt differs from preflight actual model")
            for checkpoint_receipt in model_receipt["checkpoint_files"]:
                check_artifact(checkpoint_receipt)
            state = read_json(path_of(manifest["state_preservation"][method]["path"]))
            check_artifact(manifest["state_preservation"][method])
            require(state["method"] == method and state["seal_sha256"] == seal_hash and state["unchanged"] is True
                    and state["before"]["state_sha256"] == state["after"]["state_sha256"] == model_receipt["state_sha256"],
                    "Prediction inference altered model state")
            checked.append({"dataset": dataset, "method": method, "shape": list(shape), "prediction": check_artifact(receipt)})
        job.heartbeat({"phase": "data_and_prediction_arrays", "dataset": dataset})
    total = sum(row["shape"][0] for row in checked)
    require(len(checked) == 6 and len(old_checked) == 9 and total == manifest["full_channel_prediction_origins"],
            "Prediction-unit/origin accounting changed")
    return {"new_prediction_units": 6, "new_full_channel_origin_forecasts": total, "old_prediction_instances": 9,
            "new_predictions": checked, "reused_predictions": old_checked, "period_origin_counts": counts,
            "actual_target_mask_hashes": mask_hashes, "new_score_computation": False,
            "model_loads": 0, "model_forwards": 0, "new_fits": 0}


def check_saved_score(score, channels):
    count = np.asarray(score["target_counts"], dtype=np.int64)
    require(count.shape == (channels,) and np.all(count > 0) and int(count.sum()) == score["target_count"],
            "Saved score channel denominators are invalid")
    for metric, channel_key, sum_key in (
            ("mse", "channel_mse", "squared_error_sums"),
            ("mae", "channel_mae", "absolute_error_sums"),
            ("signed_mean_error", "channel_signed_mean_error", "signed_error_sums")):
        require(len(score[channel_key]) == len(score[sum_key]) == channels, "Saved score channel coverage changed")
        ratios = np.asarray(score[sum_key]) / count
        for actual, expected in zip(score[channel_key], ratios):
            close(actual, expected, "Saved channel sum/count")
        close(score[metric], np.mean(score[channel_key]), "Saved channel-macro aggregation")


def verify_accuracy(job):
    result = read_json(HERE / "evaluation_summary.json")
    contracts = read_json(HERE / "input_contract.json")["units"]
    require(result["status"] == "complete" and result["arms"] == list(ARMS), "Four-arm accuracy phase is incomplete")
    for receipt in result["tables"].values():
        check_artifact(receipt)
    for name in ("prediction_manifest", "seal", "preflight", "old_score_replay"):
        check_artifact(result[name])
    verification = read_json(HERE / "accuracy_verification.json")
    require(verification["pass"] is True and verification["zero_fit"] is True
            and verification["model_inference"] is False and verification["independent_reproduction"] is False
            and len(verification["checks"]) == 486 and all(row["pass"] for row in verification["checks"]),
            "Saved-array score verification receipt failed or lost coverage")
    check_artifact(verification["evaluation"])
    replay = read_json(HERE / "old_score_replay.json")
    require(replay["pass"] is True and len(replay["checks"]) == 54 and all(row["pass"]
            and row["absolute_difference"] <= row["tolerance"] for row in replay["checks"]), "Original saved score replay failed")
    scalars, channels, origins, paired = [table(name) for name in
        ("accuracy_comparison.csv", "accuracy_channels.csv", "accuracy_origins.csv", "paired_comparisons.csv")]
    expected_scalar = sum((sum(COUNTS.values()) + len(ARMS)) * len(PERIODS) for _ in DATASETS)
    expected_channel = sum((sum(COUNTS.values()) + len(ARMS)) * len(PERIODS) * contracts[d]["c"] for d in DATASETS)
    expected_origin = sum(sum(COUNTS.values()) * len(canonical_origins(contracts[d])) for d in DATASETS)
    require((len(scalars), len(channels), len(origins), len(paired)) ==
            (expected_scalar, expected_channel, expected_origin, len(DATASETS) * len(PAIRS) * len(PERIODS) * 2),
            "Accuracy/channel/origin/paired CSV row coverage changed")
    require(len({(r["dataset"], r["id"], r["seed"], r["period"], r["channel_index"]) for r in channels}) == len(channels)
            and len({(r["dataset"], r["id"], r["origin"]) for r in origins}) == len(origins),
            "Channel/origin diagnostic rows were duplicated")
    pooling_roundoff = []
    for dataset in DATASETS:
        unit, contract = result["units"][dataset], contracts[dataset]
        require(len(unit["models"]) == 5 and set(unit["methods"]) == set(ARMS), "Selected accuracy instances/arms changed")
        data = load_archive(contract["test"])
        canonical = canonical_origins(contract)
        masks = np.stack([data["finite"][o:o + 48] for o in canonical])
        require(array_hash(masks) == unit["target_mask_values_sha256"]
                and array_hash(canonical) == unit["origin_values_sha256"], "Accuracy target mask/origin binding changed")
        references = {row["id"]: row for row in unit["models"]}
        for model in unit["models"]:
            for period in PERIODS:
                score = model["periods"][period]
                check_saved_score(score, contract["c"])
                require(score["origins"] == unit["origin_counts"][period], "Saved score origin count changed")
            combined, a, b = (model["periods"][period] for period in ("combined", "test_a", "test_b"))
            require(np.array_equal(np.asarray(combined["target_counts"]),
                                   np.asarray(a["target_counts"]) + np.asarray(b["target_counts"])),
                    "Combined pools saved A/B target counts differ")
            for key in ("squared_error_sums", "absolute_error_sums", "signed_error_sums"):
                expected = np.asarray(a[key]) + np.asarray(b[key])
                for channel, (actual, value) in enumerate(zip(combined[key], expected)):
                    difference = abs(float(actual) - float(value))
                    bound = TOLERANCE + SAVED_POOLING_RTOL * max(abs(float(actual)), abs(float(value)))
                    require(math.isfinite(float(actual)) and math.isfinite(float(value)) and difference <= bound,
                            f"Combined pools saved A/B FP64 sums differ: {dataset}/{model['id']}/{key}/{channel}")
                    if difference > TOLERANCE:
                        pooling_roundoff.append({"dataset": dataset, "id": model["id"], "sum": key,
                                                 "channel_index": channel, "absolute_difference": difference,
                                                 "allowed_roundoff": bound})
            for origin in canonical:
                record = one(origins, dataset=dataset, id=model["id"], origin=int(origin))
                index = int(np.searchsorted(canonical, origin))
                require(record["period"] == ("test_a" if index < unit["origin_counts"]["test_a"] else "test_b")
                        and record["target_count"] == int(masks[index].sum())
                        and record["observed_channels"] == int((masks[index].sum(axis=0) > 0).sum()),
                        "Origin diagnostic period/count/observed-channel binding changed")
        for arm, group in unit["methods"].items():
            members = [model for model in unit["models"] if model["method"] == arm]
            require(len(members) == COUNTS[arm] and group["ids"] == [row["id"] for row in members]
                    and group["seeds"] == [row["seed"] for row in members], "Selected seed grouping changed")
            if arm != ARMS[0]:
                require(len(group["ids"]) == 1, "Deterministic F0 was duplicated as seeds")
            for period in PERIODS:
                score = group["periods"][period]
                check_saved_score(score, contract["c"])
                require(score["target_counts"] == members[0]["periods"][period]["target_counts"],
                        "Seed loss averaging multiplied target denominators")
                for metric in METRICS:
                    close(score[metric], np.mean([model["periods"][period][metric] for model in members]),
                          "Individual seed loss mean")
        local = [row for row in scalars if row["dataset"] == dataset]
        for record in local:
            reference = (references[record["id"]]["periods"][record["period"]]
                         if record["aggregation"] == "individual" else
                         unit["methods"][record["method"]]["periods"][record["period"]])
            for metric in METRICS:
                close(record[metric], reference[metric], "Accuracy CSV " + metric)
            require(record["target_counts"] == reference["target_counts"]
                    and record["target_count"] == reference["target_count"]
                    and record["origins"] == reference["origins"], "Accuracy CSV denominator/origin metadata differs")
        expected_scalar_keys = {(model["id"], period) for model in unit["models"] for period in PERIODS}
        require({(r["id"], r["period"]) for r in local if r["aggregation"] == "individual"} == expected_scalar_keys
                and len([r for r in local if r["aggregation"] == "individual"]) == len(expected_scalar_keys),
                "Individual accuracy CSV key coverage differs")
        require(len([r for r in local if r["aggregation"] != "individual"]) == 12
                and {(r["method"], r["period"]) for r in local if r["aggregation"] != "individual"}
                == {(arm, period) for arm in ARMS for period in PERIODS}, "Group accuracy CSV coverage differs")
        for record in [row for row in channels if row["dataset"] == dataset]:
            index = record["channel_index"]
            require(0 <= index < contract["c"] and record["channel"] == contract["columns"][index],
                    "Channel diagnostic original channel order differs")
            reference = (references[record["id"]]["periods"][record["period"]]
                         if record["seed"] not in ("mean_selected_seeds", "single_model") else
                         unit["methods"][record["method"]]["periods"][record["period"]])
            for field, source in (("mse", "channel_mse"), ("mae", "channel_mae"),
                                  ("signed_mean_error", "channel_signed_mean_error"),
                                  ("squared_error_sum", "squared_error_sums"),
                                  ("absolute_error_sum", "absolute_error_sums"),
                                  ("signed_error_sum", "signed_error_sums"), ("target_count", "target_counts")):
                close(record[field], reference[source][index], "Channel CSV " + field)
        require(len(unit["paired_comparisons"]) == 36, "Paired comparison set changed")
        for record in unit["paired_comparisons"]:
            require((record["left"], record["right"]) in PAIRS and record["period"] in PERIODS
                    and record["metric"] in ("mse", "mae"), "An unauthorized paired comparison was added")
            config = contract["bootstrap"]
            require(record["bootstrap_draws"] == config["draws"] and record["block_origins"] == config["block_origins"]
                    and record["rng_seed"] == config["seed"] and record["rng_salt"] == config["seed_sequence_salt"]
                    and record["period_stratified"] is record["fixed_seeds_conditional"] is record["not_equivalence_or_noninferiority"] is True,
                    "Existing period-stratified paired bootstrap contract changed")
            current = one(paired, dataset=dataset, left=record["left"], right=record["right"],
                          period=record["period"], metric=record["metric"])
            require(all(current[name] == value for name, value in record.items()), "Paired CSV differs from saved comparison")
            close(record["left_value"], unit["methods"][record["left"]]["periods"][record["period"]][record["metric"]], "Paired left point")
            close(record["right_value"], unit["methods"][record["right"]]["periods"][record["period"]][record["metric"]], "Paired right point")
            close(record["absolute_difference"], record["left_value"] - record["right_value"], "Paired difference")
            close(record["denominator"], record["right_value"], "Paired denominator")
            for field in ("absolute_ci95", "relative_ci95_percent"):
                interval = record[field]
                require(interval is None or (len(interval) == 2 and all(math.isfinite(v) for v in interval)
                        and interval[0] <= interval[1]), "Paired interval is malformed")
            if record["denominator"]:
                close(record["relative_percent"], 100 * (record["left_value"] / record["right_value"] - 1), "Paired relative percent")
        job.heartbeat({"phase": "saved_aggregate_csv_consistency", "dataset": dataset})
    return {"accuracy_csv_rows": len(scalars), "accuracy_channels_csv_rows": len(channels),
            "accuracy_origins_csv_rows": len(origins), "paired_csv_rows": len(paired),
            "saved_array_verification_checks": len(verification["checks"]), "old_score_replay_checks": len(replay["checks"]),
            "saved_score_tolerance": TOLERANCE, "new_errors_or_bootstrap_computed": False,
            "saved_A_B_pooling": {"target_counts_exact": True, "sum_absolute_tolerance": TOLERANCE,
                                  "sum_relative_roundoff": SAVED_POOLING_RTOL,
                                  "scope": "Saved FP64 extensive sums only; metric/forward tolerances unchanged",
                                  "sum_differences_above_absolute_tolerance": pooling_roundoff},
            "verification_receipt": artifact(HERE / "accuracy_verification.json")}


def rotate(values, block):
    values = list(values)
    return values[1:] + values[:1] if block == 1 else list(reversed(values)) if block == 2 else values


def expected_cost_grid():
    reused = read_json(HERE / "reuse_manifest.json")["units"]
    instances, grid = {}, []
    for dataset in DATASETS:
        members = [{"id": row["id"], "dataset": dataset, "method": row["arm"],
                    "original_method": row["method"], "seed": row["seed"],
                    "source_kind": "baseline", "model_key": None} for row in reused[dataset]["rows"]]
        members.extend({"id": dataset + "_" + arm, "dataset": dataset, "method": arm,
                        "original_method": "F0", "seed": None, "source_kind": "bolt", "model_key": key}
                       for arm, key in NEW_MODELS.items())
        require(Counter(row["method"] for row in members) == COUNTS, "Cost instance mapping differs")
        instances[dataset] = members
    for block in range(3):
        for dataset in rotate(DATASETS, block):
            f0 = next(row for row in instances[dataset] if row["method"] == ARMS[1])
            ordered = [(f0, "pre", (1,))] + [(row, None, (1, 4)) for row in rotate(instances[dataset], block)] + [(f0, "post", (1,))]
            for item, sentinel, batches in ordered:
                for batch in batches:
                    row = {**item, "device": "cuda", "block": block, "sentinel": sentinel,
                           "batch": batch, "execution": "full", "chunk_rows": None}
                    row["key"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]
                    row["planned_sequence"] = len(grid)
                    grid.append(row)
    return grid


def verify_cost(job):
    bound = read_json(HERE / "cost_binding.json")
    for name, expected in bound["files"].items():
        artifact(HERE / name, expected)
    require(bound["precision"] == "float32" and bound["tf32"] is False and bound["cpu_threads"] == 4
            and bound["passes"] == bound["warmup_calls"] == 10 and bound["blocks"] == 3
            and bound["test_used_for_cost_selection"] is False and bound["additional_fitting"] == 0,
            "Fixed cost precision/thread/pass/VAL-only contract changed")
    require(bound["timing_scope"] == "Standardized CPU input through full online inference to all original-channel H48 contiguous CPU output"
            and set(bound["excluded_from_latency"]) == {"model_load", "disk", "hashing", "scoring", "ledger", "telemetry"},
            "Matched full online CPU-input-to-CPU-output timing scope changed")
    contracts = read_json(HERE / "input_contract.json")["units"]
    for dataset in DATASETS:
        unit, entry = contracts[dataset], bound["inputs"][dataset]
        data = load_archive(unit["trainval"])
        origins = np.asarray(unit["origins"]["val"]["values"][:24], dtype=np.int64)
        cpu_input = np.stack([data["x"][o - 512:o] for o in origins]).astype(np.float32)
        require(len(origins) == 24 and np.all(np.diff(origins) > 0)
                and cpu_input.shape == (24, 512, unit["c"]) and np.isfinite(cpu_input).all(), "Cost VAL24 input construction differs")
        require(entry["origins"] == origins.tolist() and entry["origin_sha256"] == array_hash(origins)
                and entry["input_sha256"] == array_hash(cpu_input) and entry["C"] == unit["c"] and entry["K"] == unit["k"]
                and entry["trainval"]["sha256"] == unit["trainval"]["sha256"], "Cost origin/input/PCA/data binding differs")
    output, grid, summary = [read_json(HERE / name) for name in
                           ("cost_cuda_rows.json", "cost_cuda_grid.json", "cost_cuda_summary.json")]
    require(output["status"] == summary["status"] == "complete", "Cost campaign coverage is incomplete")
    require(grid["binding_sha256"] == output["binding_sha256"] == digest(HERE / "cost_binding.json"), "Cost source seal changed")
    planned = expected_cost_grid()
    require(len(planned) == len(grid["rows"]) == len(output["rows"]) == output["expected_rows"] == 108
            and grid["rows"] == planned, "Cost 90-primary/18-sentinel sequence/rotation grid differs")
    counts = output["counts"]
    require(counts == grid["counts"] and counts["measurement_rows"] == 108 and counts["sentinel_rows"] == 18
            and counts["total_wrapper_calls"] == sum(10 + 10 * (24 // row["batch"]) for row in planned)
            and counts["timed_origin_units"] == 108 * 10 * 24
            and counts["warmup_origin_units"] == sum(10 * row["batch"] for row in planned), "Planned cost call/origin accounting differs")
    for row, counted in zip(planned, counts["rows"]):
        require(counted["key"] == row["key"] and counted["warmup_wrapper_calls"] == 10
                and counted["timed_wrapper_calls"] == 10 * (24 // row["batch"])
                and counted["total_wrapper_calls"] == 10 + counted["timed_wrapper_calls"], "Cost row call plan differs")
    environment = output["environment"]
    require(environment["dtype"] == "float32" and environment["tf32"] is False
            and environment["cpu_threads"] == 4 and "RTX 4070" in environment["device"], "Matched RTX4070 numerical environment differs")
    preflight = read_json(HERE / "preflight_checks.json")
    model_receipts = {row["model"]: row for row in preflight["bolt_models"]}
    baseline_receipts = {row["id"]: row for row in preflight["baselines"]}
    oom_rows, complete_rows = [], []
    for index, (identity, row) in enumerate(zip(planned, output["rows"])):
        require(all(row.get(name) == value for name, value in identity.items()) and row["binding_sha256"] == grid["binding_sha256"],
                "Raw cost row identity/source/input scope differs")
        require(read_json(HERE / "cost_rows" / "cuda" / (row["key"] + ".json")) == row,
                "Individual cost record differs from aggregate")
        require(row["status"] in ("complete", "oom") and row["passes"] == row["warmup_wrapper_calls"] == 10,
                "Failed/partial cost row is hidden or measurement plan changed")
        if row["status"] == "oom":
            require(row["oom_is_result"] is True and bool(row["error"]) and bool(row["traceback"])
                    and row["seconds_per_24_origins"] == [] and row["milliseconds_per_origin"] is None,
                    "OOM was erased or substituted with measured performance")
            oom_rows.append({name: row[name] for name in ("dataset", "method", "id", "block", "batch", "sentinel", "key")})
            continue
        complete_rows.append(row)
        times = np.asarray(row["seconds_per_24_origins"], dtype=np.float64)
        require(times.shape == (10,) and np.isfinite(times).all() and np.all(times > 0)
                and row["timed_wrapper_calls"] == 10 * (24 // row["batch"]), "Ten complete VAL24 raw passes are missing")
        require(row["raw_milliseconds_per_origin"] == (times * 1000 / 24).tolist()
                and row["raw_origins_per_second"] == (24 / times).tolist(), "Raw ms/origin or throughput arithmetic differs")
        for field, values in (("milliseconds_per_origin", row["raw_milliseconds_per_origin"]),
                              ("origins_per_second", row["raw_origins_per_second"])):
            expected = {"median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
                        "q75": float(np.quantile(values, .75)), "minimum": float(min(values)), "maximum": float(max(values))}
            require(row[field] == expected, "Raw timing median/quantile/range summary differs")
        model = row["model"]
        reference = (model_receipts[row["model_key"]]["receipt"] if row["source_kind"] == "bolt"
                     else baseline_receipts[row["id"]]["receipt"])
        expected_state = (model_receipts[row["model_key"]]["state_sha256_before"] if row["source_kind"] == "bolt"
                          else baseline_receipts[row["id"]]["state_sha256_before"])
        require(model["deployed_parameters"] == reference["parameter_count"] and model["parameter_bytes"] == reference["parameter_bytes"]
                and model["buffer_bytes"] == reference["buffer_bytes"]
                and model["deployment_tensor_bytes"] == model["parameter_bytes"] + model["buffer_bytes"]
                and model["inference_trainable_parameters"] == model["inference_trainable_parameter_count"] == 0,
                "Cost deployment actual params/bytes/frozen receipt differs from preflight")
        preserved = row["state_preservation"]
        require(preserved["before_sha256"] == preserved["after_sha256"] == expected_state
                and preserved["unchanged"] is preserved["eval_mode"] is True
                and preserved["trainable_parameters"] == 0 and preserved["optimizer_present"] is False
                and preserved["hash_scope"] == "parameters_and_buffers", "Cost inference altered model parameters/buffers")
        require(model["loaded_state"]["sha256"] == expected_state and model["loaded_state"]["eval_mode"] is True
                and model["loaded_state"]["trainable_parameters"] == 0, "Loaded cost state differs")
        require(row["peak_reserved_bytes"] >= row["peak_allocated_bytes"] > 0
                and row["resident_after_model_load_and_cleanup"]["allocated_bytes"] >= model["parameter_bytes"]
                and row["resident_after_model_load_and_cleanup"]["reserved_bytes"]
                >= row["resident_after_model_load_and_cleanup"]["allocated_bytes"], "Measured peak/resident GPU bytes are invalid")
        require("process_cpu_memory" in row and "telemetry_before" in row and "telemetry_after" in row,
                "Raw CPU memory/telemetry evidence was dropped")
        if index % 18 == 0:
            job.heartbeat({"phase": "raw_cost_artifact_and_frozen_checks", "checked": index})
    require(output["oom_rows"] == len(oom_rows), "OOM count differs from retained raw records")
    require(len(summary["instances"]) == 30 and len(summary["groups"]) == 24 and len(summary["sentinels"]) == 9,
            "Cost summary selected-instance/arm/sentinel coverage differs")
    for instance in summary["instances"]:
        raw = [row for row in output["rows"] if row["sentinel"] is None and all(row[name] == instance[name]
               for name in ("dataset", "id", "method", "seed", "batch"))]
        require(len(raw) == 3 and {row["block"] for row in raw} == {0, 1, 2}, "An instance lacks all three blocks")
        raw.sort(key=lambda row: row["block"])
        complete = [row for row in raw if row["status"] == "complete"]
        require(instance["complete_blocks"] == len(complete) and instance["oom_blocks"] == 3 - len(complete)
                and instance["status"] == ("complete" if len(complete) == 3 else "oom"), "Instance OOM/completeness was hidden")
        for destination, source in (("block_ms", "milliseconds_per_origin"), ("block_origins_s", "origins_per_second")):
            expected = [row[source]["median"] if row["status"] == "complete" else None for row in raw]
            require(instance[destination] == expected, "Instance raw block values were dropped")
        for destination, source in (("median_block_median_ms", "block_ms"), ("median_block_median_origins_s", "block_origins_s")):
            values = [value for value in instance[source] if value is not None]
            require(instance[destination] == (float(np.median(values)) if values else None), "Instance median-of-block aggregation differs")
        for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
            require(instance[field] == [row[field] if row["status"] == "complete" else None for row in raw], "Instance memory blocks differ")
    evaluation = read_json(HERE / "evaluation_summary.json")
    for group in summary["groups"]:
        instances = [row for row in summary["instances"] if all(row[name] == group[name] for name in ("dataset", "method", "batch"))]
        require(len(instances) == COUNTS[group["method"]] and group["ids"] == [row["id"] for row in instances]
                and set(group["ids"]) == set(evaluation["units"][group["dataset"]]["methods"][group["method"]]["ids"]),
                "Cost uses a different selected seed/model set than accuracy")
        complete = [row for row in instances if row["status"] == "complete"]
        require(group["status"] == ("complete" if len(complete) == len(instances) else "oom"), "Group cost completeness was hidden")
        for field, source in (("mean_seed_median_ms", "median_block_median_ms"),
                              ("mean_seed_median_origins_s", "median_block_median_origins_s")):
            expected = float(np.mean([row[source] for row in complete])) if complete else None
            require(group[field] == expected, "Selected-seed mean of instance block medians differs")
        for field, source in (("mean_seed_median_peak_allocated_bytes", "peak_allocated_bytes"),
                              ("mean_seed_median_peak_reserved_bytes", "peak_reserved_bytes")):
            expected = float(np.mean([np.median([v for v in row[source] if v is not None]) for row in complete])) if complete else None
            require(group[field] == expected, "Selected-seed memory aggregation differs")
        for field in ("deployed_parameters", "parameter_bytes", "buffer_bytes", "deployment_tensor_bytes"):
            values = [row["model"][field] for row in instances if row.get("model")]
            expected = values[0] if values and len(set(values)) == 1 else None
            require(group[field] == expected, "Summary deployment parameters/bytes differ")
    for sentinel in summary["sentinels"]:
        pair = {row["sentinel"]: row for row in output["rows"] if row["dataset"] == sentinel["dataset"]
                and row["block"] == sentinel["block"] and row["sentinel"] is not None}
        require(set(pair) == {"pre", "post"}, "A pre/post F0 sentinel was lost")
        for side in ("pre", "post"):
            expected = pair[side]["milliseconds_per_origin"]["median"] if pair[side]["status"] == "complete" else None
            require(sentinel[side + "_status"] == pair[side]["status"] and sentinel[side + "_row_key"] == pair[side]["key"]
                    and sentinel[side + "_ms"] == expected, "Sentinel drift input was changed")
        expected = sentinel["post_ms"] / sentinel["pre_ms"] - 1 if sentinel["pre_ms"] is not None and sentinel["post_ms"] is not None else None
        require(sentinel["post_relative_change"] == expected, "Raw sentinel drift arithmetic differs")
    csv = table("cost_comparison.csv")
    require(len(csv) == 132 and Counter(row["row_type"] for row in csv) == {"summary": 24, "block": 108},
            "Cost CSV dropped a raw block/sentinel or summary")
    for group in summary["groups"]:
        current = one(csv, row_type="summary", dataset=group["dataset"], method=group["method"], batch=group["batch"])
        require(all(public_paths(current[name]) == value for name, value in group.items()),
                "Cost CSV summary differs from raw summary JSON after runtime path serialization")
    for row in output["rows"]:
        current = one(csv, row_type="block", key=row["key"])
        excluded = {"seconds_per_24_origins", "raw_milliseconds_per_origin", "raw_origins_per_second", "telemetry_before", "telemetry_after"}
        require(all(public_paths(current[name]) == value for name, value in row.items() if name not in excluded),
                "Cost CSV block/OOM/model/state record differs from raw JSON after runtime path serialization")
        if row["status"] == "complete":
            require(current["block_median_ms"] == row["milliseconds_per_origin"]["median"]
                    and current["block_median_origins_s"] == row["origins_per_second"]["median"], "Flattened cost median differs")
    check_artifact(summary["cost_comparison_csv"])
    return {"raw_cost_rows": 108, "primary_measurement_rows": 90, "sentinel_measurement_rows": 18,
            "cost_summary_instances": 30, "cost_summary_groups": 24, "sentinel_pairs": 9, "cost_csv_rows": 132,
            "contract_grid_coverage_complete": True, "all_measurements_completed_without_oom": not oom_rows,
            "complete_measurement_rows": len(complete_rows), "oom_result_rows": oom_rows,
            "planned_wrapper_calls": counts["total_wrapper_calls"],
            "completed_row_wrapper_calls": sum(row["warmup_wrapper_calls"] + row["timed_wrapper_calls"] for row in complete_rows),
            "oom_partial_calls_not_claimed_counted": True,
            "cost_binding": artifact(HERE / "cost_binding.json"), "raw_cost": artifact(HERE / "cost_cuda_rows.json"),
            "CSV_JSON_path_comparison": "Exact runtime.public_paths serialization reused; numeric/results values unchanged",
            "same_VAL24_hash_all_arms": True, "same_full_online_timing_scope": True,
            "actual_params_and_state_preservation_checked": True, "new_measurement_performed": False}


def verify_reports_and_figures(job):
    required = ("PURPOSE.md", "PLAN.md", "STATUS.md", "REQUEST.txt", "FAIRNESS_TABLE.md", "FINAL_REPORT_KO.md",
                "PRESENTATION_PATCH.md", "README.md", "model_manifest.json", "input_contract.json", "preflight_checks.json",
                "prediction_manifest.json", "accuracy_comparison.csv", "accuracy_channels.csv", "accuracy_origins.csv",
                "paired_comparisons.csv", "cost_binding.json", "cost_cuda_rows.json", "cost_cuda_summary.json",
                "cost_comparison.csv", "ledger.json")
    files = [artifact(HERE / name) for name in required]
    fairness = (HERE / "FAIRNESS_TABLE.md").read_text(encoding="utf-8")
    attributes = ("Family", "Pinned model revision", "Actual backbone params", "Total deployed params",
                  "Previously fitted target-data params", "Original input channels C", "TSFM rows per origin", "TRAIN PCA",
                  "Residual G", "Target-data fitting history", "Context / horizon observations", "Precision / TF32",
                  "Same preprocessing", "Same origins / target mask", "Point forecast", "TEST exposure")
    require(fairness.count("| Attribute | Small + LEVEL | Small F0 | Mini F0 | Tiny F0 |") == 3
            and all(fairness.count("| " + name + " |") == 3 for name in attributes),
            "Three complete comparison-fairness tables are missing")
    models = read_json(HERE / "model_manifest.json")["models"]
    for model in models.values():
        require(model["revision"] in fairness and str(model["loaded_model_receipt"]["parameter_count"]) in fairness,
                "Fairness table omits immutable revisions or actual backbone parameters")
    report = (HERE / "FINAL_REPORT_KO.md").read_text(encoding="utf-8")
    checks = {
        "one_sentence_question": ("질문:",),
        "same_family_comparison": ("same-family practical allocation comparison", "method ablation"),
        "fairness_before_accuracy": ("## 비교 공정성", "## 정확도: TEST-A / TEST-B / combined"),
        "three_dataset_MSE_MAE": ("Robin", "Peacock", "Jena", "MSE", "MAE"),
        "A_B_combined": ("test_a", "test_b", "combined"),
        "relative_and_conditional_intervals": ("대응 상대 차이와 조건부 구간", "95%", "차이 불확실"),
        "B1_B4_allocated_reserved": ("B1", "B4", "allocated", "reserved"),
        "actual_deployment_parameters_bytes": ("Deploy params", "Parameter bytes", "Buffer bytes"),
        "Pareto_interpretation": ("Pareto", "Phase 0"),
        "four_strongest_objections": ("**A —", "**B —", "**C —", "**D —"),
        "retain_reduce_defer_claims": ("**유지:**", "**축소:**", "**보류:**"),
        "need_for_next_experiment": ("**다음 실험:**", "자동 실행하지 않습니다"),
    }
    require(all(all(token in report for token in tokens) for tokens in checks.values()), "Final report omits a required contract item")
    require(report.index("## 비교 공정성") < report.index("## 정확도: TEST-A / TEST-B / combined"),
            "Accuracy conclusions precede fairness binding")
    require("이미 TEST가 노출" in report and "독립 재현 또는 논문 신규성 인증이 아닙니다" in report,
            "Exposure/self-verification limits were omitted")
    patch = (HERE / "PRESENTATION_PATCH.md").read_text(encoding="utf-8")
    require("Chronos-2" in patch and "별도" in patch and "원본 PPT/PDF" in patch and "제안" in patch,
            "Presentation patch mixes families or claims an original deck edit")
    manifest = read_json(HERE / "figures" / "manifest.json")
    require(manifest["status"] == "generated_from_complete_tables"
            and manifest["no_inference"] is manifest["no_resampling"] is manifest["zero_axes"] is True,
            "Figure manifest lacks saved-table-only provenance")
    check_artifact(manifest["code"])
    sealed_report = HERE / "report_controls.py"
    sealed_report_hash = read_json(HERE / "evaluation_seal.json")["files"]["report_controls.py"]
    require(digest(sealed_report) == sealed_report_hash, "Original sealed figure/report implementation changed")
    renderer = path_of(manifest["code"]["path"]).resolve()
    renderer_detail = {"actual_renderer": manifest["code"], "original_sealed_report_sha256": sealed_report_hash,
                       "presentation_axes_correction": False}
    if renderer == sealed_report.resolve():
        require(manifest["code"]["sha256"] == sealed_report_hash, "Sealed figure renderer receipt differs")
    else:
        require(renderer == (HERE / "report_layout_fix.py").resolve(), "Unapproved figure renderer")
        correction_path = HERE / "figures" / "layout_correction.json"
        correction = read_json(correction_path)
        for name in ("before_manifest", "after_manifest", "original_report_code", "renderer"):
            check_artifact(correction[name])
        require(correction["renderer"] == manifest["code"]
                and correction["after_manifest"]["sha256"] == digest(HERE / "figures" / "manifest.json")
                and correction["original_report_code"] == manifest["original_report_code"]
                and correction["original_report_code"]["sha256"] == sealed_report_hash
                and path_of(correction["original_report_code"]["path"]).resolve() == sealed_report.resolve(),
                "Layout correction does not bind the actual renderer and original sealed report")
        require(correction["only_presentation_axes"] is correction["plotted_values_identical"] is True
                and correction["scientific_seal_changed"] is correction["test_settings_changed"] is False
                and correction["fit_count"] == 0 and correction["model_inference"] is correction["new_resampling"] is False,
                "Layout correction exceeded its presentation-only scope")
        before_path = path_of(correction["before_manifest"]["path"]).resolve()
        require(before_path.parent == (CACHE / "figures_before_layout_fix").resolve(),
                "Original figure manifest is not preserved in the declared local cache")
        before = read_json(before_path)
        require(before["code"] == correction["original_report_code"]
                and before["sources"] == manifest["sources"] == correction["identical_CSV_sources"]
                and before["decisions"] == manifest["decisions"]
                and before["point_frontiers"] == manifest["point_frontiers"]
                and [(row["dataset"], row["plotted_rows"]) for row in before["figure_rows"]]
                == [(row["dataset"], row["plotted_rows"]) for row in manifest["figure_rows"]],
                "Layout correction changed CSV inputs, plotted data or decisions")
        preserved_exports = 0
        for figure in before["figure_rows"]:
            for receipt in figure["outputs"]:
                preserved = before_path.parent / Path(receipt["path"]).name
                actual = artifact(preserved, receipt["sha256"])
                require(actual["bytes"] == receipt["bytes"], "Preserved original figure bytes differ")
                preserved_exports += 1
        require(preserved_exports == 6, "Original PNG/SVG exports were not all preserved")
        for receipt in correction["preserved_old_figures"]:
            check_artifact(receipt)
        original_tree = ast.parse(sealed_report.read_text(encoding="utf-8"))
        corrected_tree = ast.parse(renderer.read_text(encoding="utf-8"))
        original_function = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_figures")
        corrected_function = next(node for node in corrected_tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_figures")
        top_keywords = [keyword for node in ast.walk(corrected_function) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "set_ylim"
                        for keyword in node.keywords if keyword.arg == "top"]
        expected_top = ast.parse('max(score(data, dataset, a)["mse"] for a in ARMS) * 1.12', mode="eval").body
        require(len(top_keywords) == 1 and ast.dump(top_keywords[0].value, include_attributes=False)
                == ast.dump(expected_top, include_attributes=False), "Correction is not the declared 12-percent y headroom")
        for node in ast.walk(corrected_function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set_ylim":
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != "top"]
        require(ast.dump(corrected_function, include_attributes=False) == ast.dump(original_function, include_attributes=False),
                "Figure function differs beyond the single presentation-axis top keyword")
        expected_limits = {figure["dataset"]: [0, max(row["mse"] for row in figure["plotted_rows"]) * 1.12]
                           for figure in manifest["figure_rows"]}
        require(correction["expected_y_limits"] == manifest["presentation_y_limits"] == expected_limits,
                "Presentation y-axis limits differ from the identical plotted data")
        renderer_detail.update(presentation_axes_correction=True, correction=artifact(correction_path),
                               single_top_keyword_only_AST_verified=True, expected_y_limits=expected_limits,
                               identical_CSV_and_plotted_values=True, preserved_original_exports=preserved_exports,
                               original_figures_preservation_local_only=True)
    require(set(manifest["sources"]) == {"accuracy_comparison.csv", "paired_comparisons.csv", "cost_comparison.csv"},
            "Figure source CSV set differs")
    sources = {}
    for name, receipt in manifest["sources"].items():
        check_artifact(receipt)
        require(receipt["sha256"] == digest(HERE / name), "Figure source hash differs")
        sources[name] = table(name)
    require(len(manifest["figure_rows"]) == 3 and {row["dataset"] for row in manifest["figure_rows"]} == set(DATASETS),
            "Dataset figure coverage differs")
    checked_figures = []
    for figure in manifest["figure_rows"]:
        require(len(figure["outputs"]) == 2 and {Path(row["path"]).suffix for row in figure["outputs"]} == {".png", ".svg"},
                "PNG/SVG exports are incomplete")
        for receipt in figure["outputs"]:
            check_artifact(receipt)
            path = path_of(receipt["path"])
            require(path.stem == "allocation_" + figure["dataset"] and path.stat().st_size > 0, "Figure output identity differs")
            if path.suffix == ".png":
                require(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "PNG file signature is invalid")
            else:
                require("<svg" in path.read_text(encoding="utf-8"), "SVG export is invalid")
            checked_figures.append(receipt)
        require(len(figure["plotted_rows"]) == 4 and {row["arm"] for row in figure["plotted_rows"]} == set(ARMS),
                "Figure omits a primary allocation arm")
        for row in figure["plotted_rows"]:
            require(row["dataset"] == figure["dataset"] and row["period"] == "combined" and row["batch"] == 4,
                    "Figure uses a different dataset/period/batch")
            accuracy_index, cost_index = row["accuracy_csv_record"], row["cost_csv_record"]
            require(row["accuracy_csv_line"] == accuracy_index + 2 and row["cost_csv_line"] == cost_index + 2,
                    "Figure row/line mapping differs")
            a, c = sources["accuracy_comparison.csv"][accuracy_index], sources["cost_comparison.csv"][cost_index]
            require(a["dataset"] == c["dataset"] == row["dataset"] and a["method"] == c["method"] == row["arm"]
                    and a["aggregation"] != "individual" and a["period"] == "combined"
                    and c["row_type"] == "summary" and c["batch"] == 4
                    and row["mse"] == a["mse"] and row["origins_s"] == c["mean_seed_median_origins_s"]
                    and row["peak_allocated_bytes"] == c["mean_seed_median_peak_allocated_bytes"],
                    "Plotted point source/value differs from exact CSV rows")
    require(len(manifest["decisions"]) == 6, "A LEVEL-versus-Mini/Tiny dataset decision is missing")
    for decision in manifest["decisions"]:
        require(decision["token"] in report, "Report omits a recorded Phase 0 decision token")
    job.heartbeat({"phase": "reports_and_figures_audited", "figure_exports": len(checked_figures)})
    return {"required_files": files, "fairness_datasets": 3, "fairness_required_attributes": len(attributes),
            "final_report_contract_items": list(checks), "figure_exports": checked_figures,
            "figure_manifest": artifact(HERE / "figures" / "manifest.json"), "source_CSV_hash_and_exact_rows_verified": True,
            "figure_renderer": renderer_detail,
            "original_deck_written_by_campaign": False, "visual_review_claimed": False,
            "independent_reproduction_or_novelty_certification": False}


def verify_ledger_and_failures(job):
    ledger = read_json(HERE / "ledger.json")
    require(ledger["fits"] == ledger["optimizer_updates"] == 0 and ledger["limits"] == LIMITS,
            "Zero-fit or operational resource caps changed")
    require([row["id"] for row in ledger["jobs"]] == list(range(len(ledger["jobs"]))), "Ledger job IDs are discontinuous")
    failed_jobs = []
    seal = read_json(HERE / "evaluation_seal.json")
    for row in ledger["jobs"]:
        require(row["category"] in ("gpu", "cpu") and row["fit_count"] == 0,
                "Unauthorized fitting/job category is recorded")
        metadata = row.get("metadata", {})
        require(all(metadata.get(key, 0) == 0 for key in ("fit_count", "additional_fit_count", "optimizer_updates")),
                "A target-data fit/update was recorded")
        if row["id"] != job.id:
            require(row["status"] in ("complete", "failed") and "ended_utc" in row
                    and row["elapsed_s"] >= 0, "A previous ledger job remains nonterminal")
        if row["status"] == "failed":
            require(bool(row.get("error") or row.get("heartbeat_error")), "A failed job lost its exception")
            failed_jobs.append({"id": row["id"], "label": row["label"], "category": row["category"]})
        if row["label"] in ("fixed_predict", "fixed_score", "fixed_verify", "same_campaign_cuda_cost"):
            require(row["started_utc"] >= seal["sealed_utc"], "TEST/scoring/cost job began before the scientific seal")
    attempts, failure_files, partial_files = [], [], []
    for path in HERE.rglob("*.json"):
        relative = path.relative_to(HERE).as_posix()
        if path.name.startswith("final_checks") or path.name == "ledger.json" or path.name.startswith("protected_workspace"):
            continue
        if (path.parent.name == "evaluation_attempts" or path.name.startswith("preflight_attempt")
                or ".failure_" in path.name or "failure" in path.name or "partial" in path.name):
            record = read_json(path)
            status = record.get("status")
            attempts.append({"path": relative, "status": status, "sha256": digest(path)})
            if status == "failed":
                require(bool(record.get("traceback") or record.get("error")), "Failure artifact lost diagnostic evidence")
                failure_files.append({"path": relative, "sha256": digest(path)})
                for partial in record.get("preserved_partial_records", []):
                    check_artifact(partial["snapshot"])
            elif status in ("partial", "predicting", "scoring"):
                partial_files.append({"path": relative, "status": status, "sha256": digest(path)})
    for phase in ("predict", "score", "verify"):
        phase_attempts = sorted((HERE / "evaluation_attempts").glob(phase + "_[0-9][0-9].json"))
        require(phase_attempts and read_json(phase_attempts[-1])["status"] == "complete", "A completed evaluation phase attempt is missing")
        for index, path in enumerate(phase_attempts):
            record = read_json(path)
            require(record["additional_fit_count"] == 0, "An evaluation phase fitted a model")
            if index and read_json(phase_attempts[index - 1])["status"] == "failed":
                require(bool(record["retry_reason"]), "A corrected technical retry lacks its recorded reason")
    forbidden = {"backward", "Adam", "AdamW", "SGD", "RMSprop", "Optimizer", "fit", "fit_transform", "partial_fit", "PCA"}
    findings, code = [], {}
    for path in HERE.glob("*.py"):
        code[path.name] = digest(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
                if name in forbidden:
                    findings.append({"file": path.name, "line": node.lineno, "call": name})
    require(not findings, "New campaign code contains an unauthorized fitting/update call: " + repr(findings))
    budget = budget_snapshot()
    size = storage_bytes()
    require(budget["gpu_seconds"] <= LIMITS["gpu_seconds"] and budget["cpu_seconds"] <= LIMITS["cpu_seconds"]
            and ledger["download_bytes"] <= LIMITS["download_bytes"] and size <= LIMITS["storage_bytes"], "Operational hard cap was exceeded")
    require(all(row["id"] == job.id for row in budget["active"]), "Another campaign job remains active")
    require(sum(row["bytes_charged"] for row in ledger["downloads"]) == ledger["download_bytes"], "Download materialized-byte accounting differs")
    predictions = read_json(HERE / "prediction_manifest.json")
    return {"new_fit_count": 0, "optimizer_updates": 0, "new_source_no_fit_calls": code,
            "failed_ledger_jobs_preserved": failed_jobs, "failure_artifacts_preserved": failure_files,
            "partial_artifacts_preserved": partial_files, "attempt_records": attempts,
            "partial_is_not_pass_or_completed_measurement": True,
            "new_canonical_prediction_units": predictions["evaluation_units"],
            "new_full_channel_origin_forecasts": predictions["full_channel_prediction_origins"],
            "budget_at_check": {key: value for key, value in budget.items() if key != "active"},
            "own_running_audit_job_is_the_only_active_job": True,
            "new_storage_bytes": size,
            "limitations": "Ledger/source/artifact audit; unknown external work, pretraining and independent replication are not certified"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publication-metadata", type=Path,
                        help="Optional existing publication receipt; Git/publication are outside this local scientific audit")
    args = parser.parse_args()
    checks, error = [], None
    try:
        with Job("cpu", "final_verify", reserve_s=0,
                 metadata={"fit_count": 0, "model_inference": False, "new_performance_scoring": False,
                           "bootstrap_draws": 0}) as job:
            for name, function in (("protected_old_files_and_workspace", verify_protection),
                                   ("sealed_official_models_and_provenance", verify_seal_and_models),
                                   ("VAL_preflight_and_native_isolation", verify_preflight),
                                   ("canonical_data_and_prediction_receipts", verify_data_and_predictions),
                                   ("saved_accuracy_arithmetic_and_CSV_consistency", verify_accuracy),
                                   ("matched_cost_coverage_raw_values_and_frozen_states", verify_cost),
                                   ("required_reports_fairness_and_figures", verify_reports_and_figures),
                                   ("zero_fit_failure_preservation_and_resource_caps", verify_ledger_and_failures)):
                job.check_limits()
                try:
                    detail = function(job)
                    checks.append({"name": name, "pass": True, "detail": detail})
                except Exception as exc:
                    checks.append({"name": name, "pass": False, "error": repr(exc), "traceback": traceback.format_exc()})
            if not all(row["pass"] for row in checks):
                raise RuntimeError("Local artifact audit failed: " + ", ".join(row["name"] for row in checks if not row["pass"]))
    except Exception as exc:
        error = {"error": repr(exc), "traceback": traceback.format_exc()}
    passed = len(checks) == 8 and all(row["pass"] for row in checks) and error is None
    cost_check = next((row["detail"] for row in checks if row["name"] == "matched_cost_coverage_raw_values_and_frozen_states" and row["pass"]), None)
    all_measured = cost_check is not None and cost_check["all_measurements_completed_without_oom"]
    final_ledger = read_json(HERE / "ledger.json")
    terminal = all(row["status"] in ("complete", "failed") and "ended_utc" in row for row in final_ledger["jobs"])
    final_budget = budget_snapshot()
    caps = (final_budget["gpu_seconds"] <= LIMITS["gpu_seconds"] and final_budget["cpu_seconds"] <= LIMITS["cpu_seconds"]
            and final_ledger["download_bytes"] <= LIMITS["download_bytes"] and storage_bytes() <= LIMITS["storage_bytes"])
    passed = passed and terminal and caps
    publication = {"scope": "Optional external phase; no Git query/commit/push is performed here",
                   "root_README_and_publication_required_for_local_scientific_PASS": False,
                   "verified": False, "metadata_supplied": args.publication_metadata is not None}
    if args.publication_metadata is not None:
        try:
            publication["metadata"] = artifact(path_of(args.publication_metadata))
            publication["record"] = read_json(path_of(args.publication_metadata))
            publication["scope"] = "Metadata file integrity only; remote state is not independently queried"
        except Exception as exc:
            publication["metadata_error"] = repr(exc)
    status = "complete" if passed and all_measured else "complete_with_resource_results" if passed else "incomplete_or_failed"
    output = {"schema": "level_bolt_backbone_scaling_final_checks_v1", "status": status, "pass": passed,
              "artifact_contract_consistency_pass": passed,
              "complete_campaign_verified": passed and all_measured,
              "all_cost_conditions_measured_without_oom": bool(all_measured),
              "checks": checks, "terminal_ledger_verified": terminal, "final_resource_caps_verified": caps,
              "final_budget": {key: value for key, value in final_budget.items() if key != "active"},
              "ledger_at_audit_completion": artifact(HERE / "ledger.json"),
              "generated_utc": time.time(), "code": artifact(__file__), "fit_count": 0,
              "model_loads": 0, "model_inference": False, "new_performance_scores": False, "bootstrap_draws": 0,
              "independent_reproduction": False, "novelty_certification": False,
              "publication": publication, "scope": "Execution-team CPU consistency audit of saved sealed artifacts",
              "error": error}
    attempts = sorted(HERE.glob("final_checks_attempt[0-9][0-9].json"))
    attempt_path = HERE / f"final_checks_attempt{len(attempts) + 1:02d}.json"
    require(not attempt_path.exists(), "Preserve prior final audit attempt")
    save_json(attempt_path, output)
    save_json(HERE / "final_checks.json", output)
    print(json.dumps({"status": status, "pass": passed, "complete_campaign_verified": output["complete_campaign_verified"],
                      "failed_checks": [row["name"] for row in checks if not row["pass"]]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
