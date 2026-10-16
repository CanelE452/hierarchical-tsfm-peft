"""Audit only explicitly pinned new CPU cost receipts, never run a cost worker.

Default: stdlib metadata, no receipt/trial reads or writes.  --execute imports
the unchanged original summary/binding APIs (which import NumPy and Torch),
but never calls a Task loader, torch.load, np.load, Route, fit, or forward.
Resource, quiet-window, and fresh CPU prediction parity remain separate proof.
"""
import argparse
import csv
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time


NAMES = {key: "postT_V_only_cpu_" + suffix + "_20261016_v1" for key, suffix in
         (("lag", "lag"), ("fixed_pca", "fixed_pca"), ("multik", "multik"))}
PLAN_SHAS = dict(lag="fedf00ebb1807528514c0685a6b226fd6a466149531adc7a5791ffa07d92af35",
    fixed_pca="b4207edec63331cf95b24909c24c692091251dd8e29316f8867b9f66c65a0930",
    multik="2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec")
REPORT_SHA = "e9a4a05abae5d696bd54d9e82b0c14d7b7d3add794fb5ac196e7f950da9f4bf6"
REFERENCES = {
    "primary_jena_E1_RAW_BOLT_SMALL_k0_s92601": ("RAW", "8423a4fbc3b1338ad04baa2108aee3001178b8791eaa5ee5708dc21b6b37b451"),
    "primary_jena_E1_RAW_RIDGE_k0_s92601": ("RAW_RIDGE", "12012ac13367942f8cfdc81b14732039934d5aa3ed6c0dee8c24dc0728a53c42"),
    "primary_jena_E1_DIRECT_G_k0_s92601__cpu_fresh_e3": ("DIRECT_G", "8046b4b90e40ce907fefdd0f07857990db7947540288b295981e9cf318fc7f49")}
PREDICTION_SOURCES = ("prediction_cache.py", "fit_base.py", "data_contract.py", "training.py", "evaluation.py", "networks.py", "gpu_runtime.py")
ORDERED_SOURCES = ("ordered_pilot.py", "prediction_cache.py", "fit_base.py", "training.py", "networks.py", "data_contract.py", "evaluation.py", "gpu_runtime.py")
IDENTITY = ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "model_sha", "scaler_sha")
COUNTS = {"warmup": 10, "timed": 20, "stage_probe": 20}
INTEGER_FIELDS = ("trial", "order_position", "batch_size", "F_logical_streams", "F_physical_streams", "F_calls", "row_microbatch")
REQUIRED_FIELDS = {"run_id", "phase", "whole_seconds", "prediction_sha256", "stage_instrumentation", *INTEGER_FIELDS}
SUMMARY_NUMBERS = ("mean_seconds", "median_seconds", "min_seconds", "max_seconds", "p90_seconds")
ABS_TOL, REL_TOL = 1e-12, 1e-12
MAX_TRIAL_BYTES = 20 * 1024 ** 2


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inside(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or PurePosixPath(relative).is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("UNSAFE_ROOT_RELATIVE_PATH")
    result = (Path(root).resolve() / relative).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("PATH_ESCAPES_DECLARED_ROOT")
    return result


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def registry(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["experiment_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("DUPLICATE_REGISTRY_ID")
    return result


class InputPins:
    def __init__(self):
        self.records = {}

    def add(self, root, relative, scope, expected=None):
        path = inside(root, relative)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError("INPUT_SHA_CHANGED: " + scope + "/" + relative)
        key = (scope, relative)
        value = dict(root=Path(root).resolve(), relative_path=relative, sha256=actual, bytes=path.stat().st_size)
        if key in self.records and self.records[key] != value:
            raise ValueError("INPUT_CHANGED_BETWEEN_READS")
        self.records[key] = value
        return path

    def unchanged(self):
        try:
            return all(sha(inside(item["root"], item["relative_path"])) == item["sha256"]
                       and inside(item["root"], item["relative_path"]).stat().st_size == item["bytes"]
                       for item in self.records.values())
        except (OSError, ValueError):
            return False

    def public_records(self):
        return [dict(scope=scope, relative_path=relative, sha256=value["sha256"], bytes=value["bytes"])
                for (scope, relative), value in sorted(self.records.items())]


def metadata(public, cache, namespace, plan_sha, pins):
    public, cache = Path(public).resolve(), Path(cache).resolve()
    name = NAMES[namespace]
    if plan_sha != PLAN_SHAS[namespace]:
        raise ValueError("EXACT_REGISTERED_NAMESPACE_PLAN_SHA_REQUIRED")
    path = pins.add(public, name + "/PLAN.json", "original_public", plan_sha)
    plan = read_json(path)
    if (plan.get("stage") != "POST_T_V_ONLY_DEVELOPMENT" or len(plan.get("source39", {})) != 39
            or len(plan.get("immutable_public", {})) != 9 or plan.get("output_public_relative_root") != name
            or plan.get("output_cache_relative_root") != name):
        raise ValueError("ORIGINAL_39_9_OR_NAMESPACE_CONTRACT_DIFFERS")
    for collection in (plan["source39"], plan["immutable_public"]):
        for relative, expected in collection.items():
            pins.add(public, relative, "original_public", expected)
    if read_json(public / "SELECTION_LOCK.json").get("locked") is not True:
        raise ValueError("ORIGINAL_SELECTION_MUST_REMAIN_LOCKED")
    pins.add(public, "report_campaign.py", "original_public", REPORT_SHA)
    post, private = public / name, cache / name
    registered = pins.add(post, plan["postT_registry_relative_path"], "namespace_public", plan["postT_registry_sha256"])
    alias = pins.add(post, "EXPERIMENT_REGISTRY.csv", "namespace_public", plan["postT_registry_sha256"])
    if registered.read_bytes() != alias.read_bytes():
        raise ValueError("REGISTRY_ALIAS_MUST_ALREADY_BE_EXACT_BYTE_COPY")
    runner = {"lag": "cpu_lag_runner.py", "fixed_pca": "fixed_pca_runner.py", "multik": "multik_runner.py"}[namespace]
    pins.add(post, runner, "namespace_public", plan["runner_sha256"])
    for relative_key, sha_key in (("shared_utility_relative_path", "shared_utility_sha256"), ("pca_runner_relative_path", "pca_runner_sha256")):
        if relative_key in plan:
            pins.add(public, plan[relative_key], "original_public", plan[sha_key])
    for key in ("family_lock", "task_contract_reference"):
        if key in plan:
            pins.add(public, plan[key]["relative_path"], "original_public", plan[key]["sha256"])
    norm = plan["instance_norm_source_relative_path"]
    pins.add(cache.parents[1], norm, "asset_source", plan["instance_norm_source_sha256"])
    pins.add(public.parent, "temporal_frontend_campaign_20261015_v1/runtime.py", "worktree_research_source", plan["instance_norm_source_sha256"])
    return dict(plan=plan, post=post, private=private, registry=registry(alias), original_registry=registry(public / "EXPERIMENT_REGISTRY.csv"))


def validate_descriptor(value, namespace, plan_sha):
    if (set(value) != {"schema", "namespace", "plan_sha256", "receipts"} or value["schema"] != 1
            or value["namespace"] != namespace or value["plan_sha256"] != plan_sha
            or not isinstance(value["receipts"], list) or not value["receipts"]):
        raise ValueError("EXPLICIT_RECEIPT_DESCRIPTOR_OR_PLAN_DIFFERS")
    found = set()
    for item in value["receipts"]:
        if set(item) != {"receipt_relative_path", "receipt_sha256", "binding_scope"}:
            raise ValueError("UNSUPPORTED_DESCRIPTOR_FIELDS_OR_K_TRANSITION")
        relative = item["receipt_relative_path"]
        inside(Path.cwd(), relative)
        if (not relative.startswith(NAMES[namespace] + "/") or not relative.endswith(".json")
                or relative in found or not valid_sha(item["receipt_sha256"])
                or item["binding_scope"] not in {"postT_namespace", "original"}
                or item["binding_scope"] == "original" and namespace != "lag"):
            raise ValueError("DUPLICATE_UNSAFE_OR_UNSUPPORTED_RECEIPT_DESCRIPTOR")
        found.add(relative)
    return value["receipts"]


def integer(value):
    if isinstance(value, bool) or not re.fullmatch(r"0|[1-9][0-9]*", str(value)):
        raise ValueError("NONNEGATIVE_INTEGER_FIELD_REQUIRED")
    return int(value)


def finite(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("NONFINITE_OR_NEGATIVE_TIMER")
    return result


def typed_trials(rows):
    result = []
    for raw in rows:
        if not REQUIRED_FIELDS.issubset(raw) or None in raw:
            raise ValueError("TRIAL_CSV_SCHEMA_DIFFERS")
        row = dict(raw)
        for key in INTEGER_FIELDS:
            row[key] = integer(raw[key])
        row["whole_seconds"] = finite(raw["whole_seconds"])
        if raw["stage_instrumentation"] not in {"True", "False"}:
            raise ValueError("NONBOOLEAN_STAGE_FLAG")
        row["stage_instrumentation"] = raw["stage_instrumentation"] == "True"
        for key, value in raw.items():
            if key.startswith("stage_") and key.endswith("_seconds"):
                row[key] = finite(value) if value != "" else ""
        if not valid_sha(raw["prediction_sha256"]):
            raise ValueError("INVALID_PREDICTION_HASH")
        result.append(row)
    return result


def validate_trials(receipt, raw_rows, expected_F_streams, summarize):
    if receipt.get("status") != "COMPLETE_PROTOCOL":
        raise ValueError("NOT_COMPLETE_RECEIPT_PRESERVED_WITHOUT_PASS")
    if any(type(receipt.get(key)) is not int or receipt[key] != count for key, count in
           (("warmups", 10), ("timed_trials", 20), ("stage_trials", 20))):
        raise ValueError("EXACT_10_20_20_REQUIRED")
    rows = typed_trials(raw_rows)
    condition = receipt["run_id"]
    binding = receipt["accuracy_bindings"][condition]
    spec = binding["cost_route_spec"]
    batch, channels = integer(receipt["batch_size"]), integer(receipt["C"])
    if batch not in (1, 8) or integer(receipt["row_microbatch"]) != 128 or spec.get("row_microbatch") != 128:
        raise ValueError("CPU_B1_B8_ROW128_CONTRACT_REQUIRED")
    streams = expected_F_streams(spec, batch, channels)
    calls = math.ceil(streams / 128)
    schedule = [(phase, index, 0) for phase, count in COUNTS.items() for index in range(count)]
    if [(row["phase"], row["trial"], row["order_position"]) for row in rows] != schedule:
        raise ValueError("PHASE_TRIAL_INDEX_DUPLICATE_MISSING_OR_ORDER_DIFFERS")
    for row in rows:
        stage = row["phase"] == "stage_probe"
        if (row["run_id"] != condition or row["batch_size"] != batch or row["row_microbatch"] != 128
                or row["stage_instrumentation"] is not stage or row["F_logical_streams"] != streams
                or row["F_physical_streams"] != streams or row["F_calls"] != calls):
            raise ValueError("ACTUAL_F_COUNTER_BATCH_OR_STAGE_DIFFERS")
        stages = [value for key, value in row.items() if key.startswith("stage_") and key.endswith("_seconds") and value != ""]
        if bool(stages) != stage:
            raise ValueError("INSTRUMENTED_STAGE_COLUMNS_DIFFER")
    computed = summarize(rows, receipt["accuracy_bindings"])
    expected_keys = {(condition, phase) for phase in ("timed", "stage_probe")}
    saved = {(item["run_id"], item["phase"]): item for item in receipt["summaries"]}
    if len(saved) != len(receipt["summaries"]) or set(saved) != expected_keys or len(computed) != 2:
        raise ValueError("SAVED_SUMMARY_KEY_SET_DIFFERS")
    for actual in computed:
        original = saved[(actual["run_id"], actual["phase"])]
        for key in ("run_id", "phase", "batch_size", "trials", "stage_instrumentation", "stage_sum_equals_whole_claimed", "condition_id", "accuracy_run_id"):
            if type(original.get(key)) is not type(actual[key]) or original[key] != actual[key]:
                raise ValueError("SUMMARY_IDENTITY_COUNTS_OR_FLAGS_DIFFER")
        if any(not math.isclose(finite(original[key]), finite(actual[key]), rel_tol=REL_TOL, abs_tol=ABS_TOL) for key in SUMMARY_NUMBERS):
            raise ValueError("SAVED_MEAN_MEDIAN_MIN_MAX_P90_DIFFERS")
    return dict(actual_rows=len(rows), phase_counts=dict(COUNTS), expected_F_streams_per_trial=streams,
                expected_F_calls_per_trial=calls, recomputed_summaries=computed,
                float_comparison_absolute_tolerance=ABS_TOL, float_comparison_relative_tolerance=REL_TOL)


class OriginalAPIs:
    def __init__(self, public):
        public = Path(public).resolve()
        names = ("report_campaign", "measure_costs", "route_catalog", "audit_campaign", "audit_predictions", "data_contract", "fit_base", "training", "evaluation", "networks", "gpu_runtime", "resources", "deployment", "combination")
        for name in names:
            present = sys.modules.get(name)
            if present is not None and Path(present.__file__).resolve() != public / (name + ".py"):
                raise ValueError("ORIGINAL_API_MODULE_NAME_ALREADY_BOUND_ELSEWHERE")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(public))
        report = importlib.import_module("report_campaign")
        costs = importlib.import_module("measure_costs")
        catalog = importlib.import_module("route_catalog")
        import torch
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise ValueError("CPU_AUDIT_CANNOT_HAVE_INITIALIZED_CUDA")
        self.cost_binding = report._cost_binding
        self.summarize = costs._summaries_with_accuracy
        self.expected_F_streams = catalog.expected_F_streams
        self.artifact_bindings = catalog.artifact_bindings


def task_contract(state, namespace, dataset):
    plan = state["plan"]
    if namespace == "fixed_pca":
        return plan["task_contracts"][dataset]
    if dataset != ("jena" if namespace == "lag" else "beijing"):
        raise ValueError("DATASET_NOT_IN_NAMESPACE_PLAN")
    return plan["task_contract"]


def binding_context(binding_scope, run_id, state, namespace, public, cache):
    if binding_scope == "original":
        if namespace != "lag" or run_id not in REFERENCES or run_id not in state["original_registry"]:
            raise ValueError("ONLY_THREE_PINNED_ORIGINAL_JENA_REFERENCES_SUPPORTED")
        return dict(public=public, cache=cache, registry=state["original_registry"], scope="original")
    if binding_scope != "postT_namespace" or run_id not in state["registry"]:
        raise ValueError("POST_T_BINDING_REQUIRES_THIS_NAMESPACE_REGISTERED_ID")
    return dict(public=state["post"], cache=state["private"], registry=state["registry"], scope="namespace")


def validate_binding(receipt, item, state, namespace, public, cache, pins, api):
    plan = state["plan"]
    condition = receipt["run_id"]
    if set(receipt.get("accuracy_bindings", {})) != {condition}:
        raise ValueError("ONLY_ONE_EXPLICIT_SINGLE_ROUTE_BINDING_SUPPORTED")
    binding = receipt["accuracy_bindings"][condition]
    run_id = binding["accuracy_run_id"]
    context = binding_context(item["binding_scope"], run_id, state, namespace, public, cache)
    bound_public, bound_cache, registered, scope = (context[key] for key in ("public", "cache", "registry", "scope"))
    original = scope == "original"
    definitions = plan.get("outputs", plan.get("conditions", []))
    declared = next((row for row in definitions if row["run_id"] == run_id), None)
    if not original and declared is None:
        raise ValueError("ACCURACY_RUN_NOT_IN_SELECTED_NAMESPACE_PLAN")
    inputs = {}
    accuracy_id, status = api.cost_binding(receipt, condition, bound_public, bound_cache, registered, inputs)
    if status != "BOUND_EXACT_ROUTE" or accuracy_id != run_id:
        raise ValueError("ORIGINAL_COST_BINDING_DID_NOT_PASS")
    for (kind, relative), value in inputs.items():
        pins.add(bound_cache if kind == "private" else bound_public, relative, scope + "_" + kind, value["sha256"])
    spec, measured = binding["accuracy_route_spec"], binding["cost_route_spec"]
    if api.artifact_bindings(spec) != binding["artifact_bindings"]:
        raise ValueError("EMBEDDED_CANONICAL_ARTIFACT_BINDINGS_DIFFER")
    relative = binding["accuracy_manifest_public_relative_path"]
    manifest_path = pins.add(bound_public, relative, scope + "_public", binding["accuracy_manifest_sha256"])
    manifest = read_json(manifest_path)
    reference_path = manifest_path.with_name("REFERENCE_CONTRACT.json")
    pins.add(bound_public, reference_path.relative_to(bound_public).as_posix(), scope + "_public")
    reference = read_json(reference_path)
    if any(manifest.get(key) != reference.get(key) for key in IDENTITY):
        raise ValueError("V_REFERENCE_IDENTITY_DIFFERS")
    if original and binding["accuracy_manifest_sha256"] != REFERENCES[run_id][1]:
        raise ValueError("ORIGINAL_REFERENCE_MANIFEST_SHA_DIFFERS")
    task = task_contract(state, namespace, receipt["dataset"])
    channels, horizon, length = len(task["columns"]), task["H"], task["L"]
    ranges = task["primary_split_ranges"]
    end = ranges["A_stop"][1]
    n = task.get("expected_V_origins", task.get("legal_origin_counts_primary", {}).get("V_select", 43))
    expected_k = 21 if original and REFERENCES[run_id][0] == "RAW" else 0 if original else declared.get("latent_k", declared.get("spec", {}).get("latent_k"))
    expected_family = REFERENCES[run_id][0] if original else ("NESTED_PCA" if declared.get("kind") == "nested" else "B0" if namespace == "multik" else declared["family"])
    expected = dict(dataset=receipt["dataset"], family=expected_family, fold="primary", canonical_fit_end=end, latent_k=expected_k,
                    model_size=None if expected_family in {"DIRECT_G", "RAW_RIDGE"} else "small")
    execution_family = "PCA_FIXED_BOLT" if expected_family == "NESTED_PCA" else expected_family
    if (any(spec.get(key) != value for key, value in expected.items()) or receipt.get("family") != execution_family
            or any(receipt.get(key) != value for key, value in (("C", channels), ("H", horizon), ("L", length)))
            or manifest.get("cohort") != f"{receipt['dataset']}:primary:V_select:stride{horizon}"
            or manifest.get("data_sha") != task["data_sha256"] or receipt.get("data_sha256") != task["data_sha256"]
            or manifest.get("scaler_sha") != task["refit_scaler_sha256"] or receipt.get("scaler_sha") != manifest["scaler_sha"]):
        raise ValueError("PLAN_ROUTE_AXES_DATA_OR_CANONICAL_A60_SCALER_DIFFERS")
    candidates = list(range(max(ranges["V_select"][0], length), ranges["V_select"][1] - horizon + 1, horizon))
    if (receipt.get("origins") != candidates[:receipt["batch_size"]]
            or receipt.get("repeated_histories_for_batch_cost_only") is not False
            or "cost_origins" in measured
            or not all(valid_sha(receipt.get(key)) for key in ("physical_input_sha256", "observation_mask_sha256"))):
        raise ValueError("DECLARED_FIRST_V_PAST_HISTORY_COHORT_DIFFERS")
    shapes = dict(pred_normalized=[n, horizon, channels], target_normalized=[n, horizon, channels], observed=[n, horizon, channels], origins=[n], channel_ids=[channels])
    if set(manifest["arrays"]) != set(shapes) or any(manifest["arrays"][key]["shape"] != value for key, value in shapes.items()):
        raise ValueError("STANDARD_FULL_V_ARRAY_METADATA_SHAPES_DIFFER")
    if reference.get("arrays") != {key: value for key, value in manifest["arrays"].items() if key != "pred_normalized"}:
        raise ValueError("STANDARD_V_TRUTH_REFERENCE_METADATA_DIFFERS")
    uses_F = api.expected_F_streams(measured, receipt["batch_size"], channels) > 0
    foundation = plan["foundation_tensor_sha256"] if uses_F else "NO_FOUNDATION_USED"
    if binding.get("foundation_state_sha256") != foundation or spec.get("foundation_sha") != (foundation if uses_F else "NO_FOUNDATION"):
        raise ValueError("ACTUAL_FOUNDATION_BINDING_DIFFERS")
    if (receipt.get("dtype") != ("float64" if expected_family == "RAW_RIDGE" else "float32")
            or receipt.get("physical_output_dtype") != "float64" or receipt.get("inverse_transform_dtype") != "float64"):
        raise ValueError("ORIGINAL_CPU_PIPELINE_DTYPE_CONTRACT_DIFFERS")
    if receipt.get("artifact_sha256") != spec.get("artifact_sha256"):
        raise ValueError("COST_SELECTED_ARTIFACT_SHA_DIFFERS")
    foundation_fields = {key: spec[key] for key in ("foundation_sha", "foundation_asset", "foundation_revision", "model_size") if key in spec}
    if binding.get("foundation_bindings") != ({"route": foundation_fields} if foundation_fields else {}):
        raise ValueError("CANONICAL_FOUNDATION_METADATA_BINDINGS_DIFFER")
    meta = manifest["metadata"]
    if meta.get("family") != expected_family:
        raise ValueError("SAVED_V_CANONICAL_FAMILY_DIFFERS")
    ordered = not original and namespace == "multik" and declared["kind"] != "independent"
    parent = None
    if expected_family == "NESTED_PCA":
        if spec.get("artifact_cache_relative_path") != "ordered/primary_beijing/nested_pca.npz":
            raise ValueError("CANONICAL_NESTED_ARTIFACT_PATH_DIFFERS")
        contract_relative = "ordered/primary_beijing/nested_pca_contract.json"
        contract = read_json(pins.add(bound_cache, contract_relative, scope + "_private"))
        expected_contract = dict(artifact_sha256=spec["artifact_sha256"], data_sha=task["data_sha256"], fit_end=end,
            source_sha=fingerprint({key: plan["source39"][key] for key in ORDERED_SOURCES}))
        if contract != expected_contract or meta.get("family") != "NESTED_PCA":
            raise ValueError("FIXED_NESTED_FOUR_FIELD_CONTRACT_OR_FAMILY_DIFFERS")
        parent = contract
    elif "fit_receipt_public_relative_path" in spec:
        parent = read_json(pins.add(bound_public, spec["fit_receipt_public_relative_path"], scope + "_public", spec["fit_receipt_sha256"]))
        parent_id = declared["parent_run_id"] if ordered else run_id
        if (parent.get("status") != "COMPLETE_VALID" or parent.get("artifact_sha256") != spec["artifact_sha256"]
                or parent.get("run_id") != parent_id or parent.get("artifact_cache_relative_path") != spec["artifact_cache_relative_path"]):
            raise ValueError("SELECTED_PARENT_FIT_RECEIPT_DIFFERS")
        scaler_key, end_key = ("scaler_sha", "fit_end") if expected_family == "RAW_RIDGE" else ("refit_scaler_sha", "refit_fit_end")
        if parent.get(scaler_key) != task["refit_scaler_sha256"] or parent.get(end_key) != end:
            raise ValueError("PARENT_A60_NORMALIZATION_DIFFERS")
        source_relative = "runs/" + parent["run_id"] + "/SOURCE_CONTRACT.json"
        if expected_family != "RAW_RIDGE":
            source = read_json(pins.add(bound_public, source_relative, scope + "_public"))
            if (source.get("source_sha") != parent.get("source_sha") or source.get("data_sha") != task["data_sha256"]
                    or not source.get("files") or any(plan["source39"].get(key) != value for key, value in source["files"].items())
                    or fingerprint(source["files"]) != source["source_sha"]):
                raise ValueError("PARENT_SOURCE_CONTRACT_IDENTITY_DIFFERS")
            planned_parent = next((row for row in plan.get("fits", plan.get("conditions", [])) if row["run_id"] == parent["run_id"]), None)
            if not original and (planned_parent is None or source.get("spec") != planned_parent["spec"]):
                raise ValueError("PARENT_RECIPE_DIFFERS_FROM_FROZEN_NAMESPACE_PLAN")
        elif parent.get("source_files") != {key: plan["source39"][key] for key in ("data_contract.py", "ridge_baseline.py")}:
            raise ValueError("ORIGINAL_RIDGE_FITTING_SOURCES_DIFFER")
        canonical_model = parent["model_sha"] if expected_family == "RAW_RIDGE" else fingerprint(dict(
            artifact_sha=parent["artifact_sha256"], model_state_sha=parent["model_state_sha"], normalization=parent["refit_scaler_sha"]))
        if spec.get("expected_prediction_model_sha") != canonical_model:
            raise ValueError("CANONICAL_MODEL_OR_ORDERED_PARENT_NORMALIZATION_HASH_DIFFERS")
    elif expected_family != "RAW":
        raise ValueError("SELECTED_FIT_RECEIPT_REQUIRED_FOR_NON_RAW_ROUTE")
    if ordered:
        if (spec.get("ordered_manifest_public_relative_path") != relative or spec.get("ordered_manifest_sha256") != binding["accuracy_manifest_sha256"]
                or meta.get("parent_artifact_sha256") != spec["artifact_sha256"] or meta.get("inference_latent_k") != expected_k
                or meta.get("variant") != declared["variant"] or meta.get("T_final_accessed") is not False
                or meta.get("parent_run_id") != ("nested_pca" if expected_family == "NESTED_PCA" else declared["parent_run_id"])
                or expected_family != "NESTED_PCA" and spec.get("source_parent_run_id") != declared["parent_run_id"]):
            raise ValueError("ORDERED_OR_NESTED_PREFIX_BINDING_DIFFERS")
        training_source = parent["source_sha"]
        expected_source = fingerprint(dict(sources={key: plan["source39"][key] for key in ORDERED_SOURCES}, training_source_sha=training_source))
        if expected_family != "NESTED_PCA" and manifest["model_sha"] != fingerprint(dict(artifact_sha=parent["artifact_sha256"], state_sha=parent["model_state_sha"])):
            raise ValueError("ORDERED_V_MODEL_SHA_IS_NOT_ITS_SELECTED_PARENT_STATE")
    else:
        files = {key: plan["source39"][key] for key in PREDICTION_SOURCES}
        if expected_family == "RAW_RIDGE":
            files["ridge_baseline.py"] = plan["source39"]["ridge_baseline.py"]
        if meta.get("source_files") != files:
            raise ValueError("STANDARD_V_DEPLOYMENT_SOURCE_FILES_DIFFER")
        expected_source = fingerprint(dict(deployment_sources=files, training_source_sha=meta["training_source_sha"]))
        standard = dict(model_sha=manifest["model_sha"], archive_sha256=manifest["archive_sha256"],
            manifest_public_relative_path=relative, manifest_sha256=binding["accuracy_manifest_sha256"])
        if spec.get("standard_V_binding") != standard or meta.get("canonical_scaler_fit_end") != end:
            raise ValueError("STANDARD_V_OR_CANONICAL_SCALER_BINDING_DIFFERS")
        if expected_family != "RAW_RIDGE" and spec.get("expected_prediction_model_sha") != manifest["model_sha"]:
            raise ValueError("STANDARD_V_MODEL_DIFFERS_FROM_CANONICAL_MODEL")
        if parent is not None and meta.get("training_source_sha") != parent.get("source_sha"):
            raise ValueError("STANDARD_V_TRAINING_SOURCE_DIFFERS")
    if manifest.get("source_sha") != expected_source:
        raise ValueError("V_PREDICTION_SOURCE_FINGERPRINT_DIFFERS")
    return dict(accuracy_run_id=run_id, binding_scope=item["binding_scope"], canonical_family=expected_family,
        cost_execution_family=execution_family,
        dataset=receipt["dataset"], batch_size=receipt["batch_size"], origins=receipt["origins"],
        data_sha256=task["data_sha256"], scaler_sha=manifest["scaler_sha"], artifact_bindings=binding["artifact_bindings"],
        actual_foundation_state_sha256=binding["foundation_state_sha256"], V_manifest_sha256=binding["accuracy_manifest_sha256"],
        canonical_route_sha256=binding["accuracy_route_spec_sha256"], V_archive_payload_read=False,
        selected_artifacts_checked_as_opaque_bytes=bool(binding["artifact_bindings"]), canonical_route_rebuilt_from_Task_or_weights=False,
        actual_nested_archive_scaler_values_rechecked=False,
        physical_input_sha256=receipt["physical_input_sha256"], observation_mask_sha256=receipt["observation_mask_sha256"])


def load_trials(path):
    if path.stat().st_size > MAX_TRIAL_BYTES:
        raise ValueError("TRIAL_FILE_EXCEEDS_ORIGINAL_20_MIB_CONTRACT")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        content = stream.read(MAX_TRIAL_BYTES + 1)
    if len(content.encode("utf-8")) > MAX_TRIAL_BYTES:
        raise ValueError("INFLATED_TRIAL_FILE_TOO_LARGE")
    import io
    reader = csv.DictReader(io.StringIO(content, newline=""))
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("MISSING_OR_DUPLICATE_CSV_COLUMNS")
    return list(reader)


def safe_reason(error, roots):
    reason = type(error).__name__ + ": " + str(error)
    for root in sorted({str(Path(root).resolve()) for root in roots}, key=len, reverse=True):
        reason = reason.replace(root, "<ROOT>").replace(root.replace("\\", "/"), "<ROOT>")
    return re.sub(r"[A-Za-z]:[\\/][^\s\"']+", "<PRIVATE_PATH>", reason)


def write_result(output, value):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "AUDIT_RESULT.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def execute(args):
    public, cache = Path(args.public_root).resolve(), Path(args.cache_root).resolve()
    output, input_path = Path(args.output).resolve(), Path(args.input_list).resolve()
    post = public / NAMES[args.namespace]
    if output.exists() or not output.is_relative_to(post / "diagnostics") or output == post / "diagnostics":
        raise FileExistsError("FRESH_OUTPUT_DIRECTORY_UNDER_NAMESPACE_DIAGNOSTICS_REQUIRED")
    pins = InputPins()
    state = metadata(public, cache, args.namespace, args.plan_sha, pins)
    pins.add(Path(__file__).parent, Path(__file__).name, "audit_source")
    pins.add(Path(__file__).parent, "test_audit_new_cpu_costs.py", "audit_source")
    pins.add(input_path.parent, input_path.name, "explicit_descriptor", args.input_sha)
    descriptors = validate_descriptor(read_json(input_path), args.namespace, args.plan_sha)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("EXPLICIT_CPU_ONLY_ENVIRONMENT_CUDA_VISIBLE_DEVICES_MINUS_ONE_REQUIRED")
    api = OriginalAPIs(public)
    records = []
    for item in descriptors:
        record = dict(receipt_relative_path=item["receipt_relative_path"], receipt_sha256=item["receipt_sha256"],
            status="FAILED_VALIDATION", original_status=None, original_row_count=None, trials_sha256=None)
        try:
            path = pins.add(public, item["receipt_relative_path"], "original_public", item["receipt_sha256"])
            receipt = read_json(path)
            record["original_status"] = receipt.get("status")
            original_reason = receipt.get("reason", receipt.get("failure", receipt.get("failure_class")))
            record["original_failure_reason"] = safe_reason(ValueError(str(original_reason)), (public, cache, input_path.parent)) if original_reason is not None else None
            if "trials_file" in receipt:
                trials = inside(path.parent, receipt["trials_file"])
                trial_relative = trials.relative_to(public).as_posix()
                record["trials_sha256"] = receipt.get("trials_sha256")
                pins.add(public, trial_relative, "original_public", receipt["trials_sha256"])
                rows = load_trials(trials)
                record["original_row_count"] = len(rows)
            else:
                rows = None
            if receipt.get("status") != "COMPLETE_PROTOCOL":
                record.update(status="INCOMPLETE_OR_FAILED_ORIGINAL_RECEIPT", reason="Original status preserved; no PASS and no zero imputation")
                records.append(record)
                continue
            if (receipt.get("scope") != "fresh_process_single_route" or receipt.get("device") != "cpu" or "run_ids" in receipt
                    or receipt.get("cached_forecast_reads") != 0 or receipt.get("training_peak_included") is not False
                    or receipt.get("stage_probe_is_intrusive_separate_scope") is not True
                    or receipt.get("source_sha256") != {key: state["plan"]["source39"][key] for key in ("deployment.py", "measure_costs.py")}):
                raise ValueError("ONLY_ORIGINAL_FRESH_SINGLE_ROUTE_CPU_COST_PROTOCOL_SUPPORTED")
            if rows is None:
                raise ValueError("COMPLETE_RECEIPT_HAS_NO_ACTUAL_TRIAL_FILE")
            record["binding"] = validate_binding(receipt, item, state, args.namespace, public, cache, pins, api)
            record.update(validate_trials(receipt, rows, api.expected_F_streams, api.summarize))
            record.update(status="PASS_COST_NUMBERS_AND_METADATA_ONLY", actual_saved_trial_numbers_recomputed=True)
        except Exception as error:
            record["reason"] = safe_reason(error, (public, cache, input_path.parent, output))
        records.append(record)
    cohorts = {}
    for record in records:
        if record["status"] == "PASS_COST_NUMBERS_AND_METADATA_ONLY":
            binding = record["binding"]
            key = (binding["dataset"], binding["data_sha256"], binding["batch_size"])
            cohorts.setdefault(key, []).append(record)
    for group in cohorts.values():
        identities = {(tuple(row["binding"]["origins"]), row["binding"]["physical_input_sha256"], row["binding"]["observation_mask_sha256"]) for row in group}
        if len(identities) != 1:
            for record in group:
                record.update(status="FAILED_VALIDATION", reason="Provided same-task/batch cost histories have different recorded input hashes")
    preserved = pins.unchanged()
    passed = preserved and all(row["status"] == "PASS_COST_NUMBERS_AND_METADATA_ONLY" for row in records)
    result = dict(schema=1, status="PASS_COST_NUMBERS_AND_METADATA_ONLY" if passed else "FAILED_OR_INCOMPLETE",
        namespace=args.namespace, plan_sha256=args.plan_sha, descriptor_sha256=args.input_sha, records=records,
        explicit_receipts_only=True, automatic_discovery=False, k_transition_supported=False,
        original_source39_immutable9_namespace_registration_and_captured_inputs_preserved=preserved,
        input_pins=pins.public_records(), source_sha256=sha(__file__), completed_unix=time.time(),
        science_api_imported=True, numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules,
        Task_loader_called=False, model_or_Route_constructed=False, weights_deserialized=False, NPZ_deserialized=False,
        training_or_forward_executed=False, T_target_requested=False, cost_workers_started=0,
        resource_guard_verified=False, quiet_window_verified=False, fresh_CPU_prediction_parity_verified=False,
        external_proofs_required=["CPU tree guard/closure and resource receipts", "quiet cost window", "fresh CPU replay parity"],
        scope="Only saved cost trial arithmetic and metadata/opaque selected-artifact bindings; no new inference or existing audit promotion")
    write_result(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--public-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--namespace", choices=tuple(NAMES))
    parser.add_argument("--plan-sha")
    parser.add_argument("--input-list", help="Private JSON of explicit original-R-relative receipt descriptors")
    parser.add_argument("--input-sha")
    parser.add_argument("--output", help="Fresh directory under the selected public namespace diagnostics")
    args = parser.parse_args(argv)
    if not args.execute:
        checked = False
        if any((args.public_root, args.cache_root, args.namespace, args.plan_sha)):
            if not all((args.public_root, args.cache_root, args.namespace, args.plan_sha)):
                parser.error("Metadata pin checking requires public/cache roots, namespace and plan SHA together")
            metadata(args.public_root, args.cache_root, args.namespace, args.plan_sha, InputPins())
            checked = True
        print(json.dumps(dict(status="METADATA_ONLY_NO_RECEIPTS_READ", supported_namespaces=list(NAMES),
            metadata_pins_checked=checked, science_api_imported=False, files_written=False,
            k_transition_supported=False, source_sha256=sha(__file__))))
        return 0
    if not all((args.public_root, args.cache_root, args.namespace, args.plan_sha, args.input_list, args.input_sha, args.output)):
        parser.error("Execute requires roots, namespace, exact plan SHA, input list/SHA and fresh output")
    if not valid_sha(args.input_sha):
        parser.error("Input descriptor SHA must be lowercase SHA256")
    result = execute(args)
    print(json.dumps(dict(status=result["status"], receipts=len(result["records"]),
                         source_sha256=result["source_sha256"], files_written=True)))
    return 0 if result["status"] == "PASS_COST_NUMBERS_AND_METADATA_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
