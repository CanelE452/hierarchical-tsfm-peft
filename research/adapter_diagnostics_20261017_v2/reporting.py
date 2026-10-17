"""Public evidence aggregation and figures; never trains or loads a model."""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import csv
import io
import json
import math
import pickle
import pickletools
import hashlib
import gzip
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

from .core import SUCCESS, atomic_json, read, sha, utc, validate_plan
from .data_audit import atomic

NAME = "adapter_diagnostics_20261017_v2"
OLD = "informs_b2_why_where_20261017"
BROAD = "informs_broad_validation_20261015_v1"
FIGURES = ("accuracy_cost", "learning_curves", "horizon_delta", "channel_delta", "backbone_comparison", "qualitative_forecasts")
RUN_FIELDS = ["run_id", "cohort", "method", "source_family", "backbone", "backbone_revision", "routing", "seed", "C", "K", "L", "H", "fit", "selection", "refit", "input_info_scope", "supervision_scope", "device", "backend", "dtype", "point_statistic", "trainable_params", "deployed_trainable_params", "fit_stage_params", "MSE", "MAE", "epochs", "updates", "exposures", "split", "metric_scale", "status", "origin", "source_file", "source_sha256", "scaler_sha256", "target_sha256", "mask_sha256", "origins_sha256", "channel_sha256", "checkpoint_sha256", "scientific_identity", "continuation_kind", "historical_bitwise_equivalence_claimed", "channel_order_validation", "G6_channel_coverage", "G6_horizon_coverage", "independent_private_score_sha256"]
RUN_FIELDS += ["evaluation_contract_derivation", "prediction_binding_scope"]
FIT_FIELDS = ["search_selection_stage_seconds", "refit_stage_seconds", "fit_stage_total_seconds", "fit_time_scope", "fit_device", "fit_source_refs"]
RUN_FIELDS += FIT_FIELDS
RUN_FIELDS += ["fit_MSE", "fit_MAE", "fit_score_scope", "fit_score_source", "fit_score_source_sha256"]
ARRAY_KEYS = {"target_sha256": "target_normalized", "mask_sha256": "observed", "origins_sha256": "origins", "channel_sha256": "channel_ids"}


def report_helper(ctx, name, bindings=None):
    path = Path(ctx["repo"])/"reports"/NAME/"code"/(name+".py")
    module_name = "adapter_report_"+name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if bindings is not None:
        _source(path, ctx, bindings)
    return module


def join_trajectory_evidence(data, value):
    data["bindings"].update(value["public_sources"])
    by_run = {row["run_id"]: row for row in data["runs"]}
    for record in value["b03_fit_rows"]:
        if record["run_id"] in by_run:
            by_run[record["run_id"]].update(fit_MSE=record.get("fit_MSE"), fit_MAE=record.get("fit_MAE"),
                fit_score_scope=record.get("score_scope", "Producer recorded masked fit score; stage="+str(record.get("stage"))),
                fit_score_source="TRAJECTORY_DIAGNOSTICS.json", fit_score_source_sha256=None)
    endpoints = {row["run_id"]: row for row in data["endpoints"]}
    for pair in value["a03_sampler_pairs"]:
        for record in pair["records"]:
            if record["job_id"] in endpoints:
                endpoints[record["job_id"]].update(sampler_sha256=record["sampler_sha256"],
                    order_sha256=record["order_sha256"], sampler_comparison_status=pair["status"],
                    sampler_comparison_scope=pair["scope"], historical_consumed_order_trace_verified=False)
    data["trajectory"] = value


def canonical_method(value):
    return {"Bolt_Raw": "Raw", "Bolt_B0": "B0", "Bolt_B2": "B2", "B2_CAP_POINTWISE": "CAP",
            "RAW": "Raw", "RAW_RIDGE": "Ridge", "RIDGE": "Ridge", "LORA": "LoRA", "Bolt_LoRA_R8": "LoRA"}.get(value, value)


def f_routing(run):
    method = canonical_method(run["method"])
    if run.get("routing") == "native_multivariate_per_origin" or method == "Chronos2_nativeRaw":
        return "Native multivariate: C joint channels per origin group; B independent origin groups"
    if run.get("backbone") == "NONE" or method == "Ridge" or method.endswith("FULL_HISTORY_PROBE"):
        return "No F calls; source-bound deterministic regression"
    if method in ("Raw", "Moirai_Raw", "LoRA"):
        return "C independent univariate streams per origin"
    return "K latent independent univariate streams per origin"


def cost_accuracy_match(cost, runs):
    if str(cost.get("accuracy_source", "")).startswith("NA:"):
        return None
    choices = [run for run in runs if run["cohort"] == cost["cohort"] and canonical_method(run["method"]) == canonical_method(cost["method"])
               and all(number(run.get(key)) == number(cost.get(key)) for key in ("C", "L", "H"))]
    if cost.get("accuracy_run_id"):
        choices = [run for run in choices if run["run_id"] == cost["accuracy_run_id"]]
    seed = cost.get("supervision_seed")
    if seed is not None:
        choices = [run for run in choices if number(run.get("seed")) == number(seed)]
    elif len(choices) > 1:
        reference_seed = cost.get("evaluation_reference_seed")
        choices = [run for run in choices if number(run.get("seed")) in (None, number(reference_seed))]
    for run in choices:
        if number(cost.get("accuracy_MSE")) is not None and number(cost.get("accuracy_MAE")) is not None and max(abs(float(cost["accuracy_MSE"])-float(run["MSE"])), abs(float(cost["accuracy_MAE"])-float(run["MAE"]))) > 1e-10:
            continue
        expected_archive = cost.get("accuracy_archive_sha256")
        actual_archive = run.get("independent_private_score_sha256")
        if expected_archive and expected_archive != actual_archive:
            continue
        return run
    return None


def contract_metadata(source):
    arrays = source.get("array_fingerprints", source.get("arrays", {}))
    output = {"dtype": arrays["pred_normalized"]["dtype"]} if isinstance(arrays.get("pred_normalized"), dict) and arrays["pred_normalized"].get("dtype") else {}
    for field, key in ARRAY_KEYS.items():
        value = arrays.get(key, {})
        direct = source.get(field)
        if isinstance(value, dict) and value.get("sha256"):
            output[field] = value["sha256"]
        elif direct is not None:
            output[field] = direct
    for key in ("scaler_sha256", "point_statistic", "dtype", "backend", "device", "fit", "selection", "refit"):
        if source.get(key) is not None:
            output[key] = source[key]
    return output


def canonical_cohort(run, repo):
    """Preserve separate C32 and C170 identities and verify registered channel order."""
    import numpy as np
    name = str(run.get("cohort", run.get("dataset", "")))
    c, length, horizon = (number(run.get(key)) for key in ("C", "L", "H"))
    if name.startswith("jena"):
        if (c, length, horizon) != (21, 512, 48):
            raise ValueError("Jena report dimensions differ from registered cohort")
        canonical, dataset = "jena", "jena"
    elif name.startswith("pems"):
        if c not in (32, 170) or (length, horizon) != (512, 12):
            raise ValueError("PeMS report dimensions differ from registered cohort")
        if ("C32" in name or name == "pems32") and c != 32:
            raise ValueError("C32 cohort label conflicts with actual channel count")
        canonical, dataset = ("pems32" if c == 32 else "pems08"), "pems08"
    else:
        return dict(run, channel_order_validation="NOT_APPLICABLE")
    contracts = read(Path(repo)/"research"/BROAD/"DATA_CONTRACT.json")["assets"]
    columns = contracts[dataset]["columns"]
    if canonical == "pems32":
        plan = read(Path(repo)/"research"/BROAD/"priority_pems_subset_20261016_v1/PLAN.json")
        columns = plan["column_selection"]["selected_ids"]
    expected = hashlib.sha256(np.asarray(columns, dtype=str).tobytes()).hexdigest()
    supplied = run.get("channel_sha256")
    if supplied and re.fullmatch("[a-f0-9]{64}", str(supplied)) and supplied != expected:
        raise ValueError("Report channel order conflicts with registered cohort")
    validation = "EXACT_REGISTERED_ORDER" if supplied == expected else "UNVERIFIED_MISSING_CANONICAL_HASH"
    if supplied == expected and str(run.get("channel_order_validation", "")).startswith("DERIVED_"):
        validation = run["channel_order_validation"]
    return dict(run, cohort=canonical, channel_order_validation=validation)


def score_prediction_archive(path, expected_sha, expected_arrays, run):
    """Independent FP64 sufficient statistics from one hash-bound saved V archive."""
    import numpy as np
    path = Path(path)
    if path.suffix != ".npz" or not path.is_file() or sha(path) != expected_sha:
        raise ValueError("Saved V prediction archive SHA mismatch")
    with np.load(path, allow_pickle=False) as archive:
        required = {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}
        if set(archive.files) != required:
            raise ValueError("Only registered five-member saved V prediction archives supported")
        values = {key: archive[key] for key in required}
    return score_prediction_values(values, expected_arrays, run)


def score_prediction_values(values, expected_arrays, run):
    import numpy as np
    target, mask, prediction = values["target_normalized"], values["observed"], values["pred_normalized"]
    if (target.ndim != 3 or target.shape != prediction.shape or target.shape != mask.shape or mask.dtype != np.dtype(bool)
            or target.shape[0] > 148 or target.shape[1:] != (int(run["H"]), int(run["C"]))):
        raise ValueError("Saved V dimensions/mask violate bounded cohort")
    t_end = 16772 if run["cohort"] == "jena" else 14284
    v_start = 14675 if run["cohort"] == "jena" else 12499
    origins = values["origins"]
    if np.any(origins < v_start) or np.any(origins+int(run["H"]) > t_end):
        raise ValueError("Saved prediction outside registered development V")
    fingerprints = {}
    for key, value in values.items():
        actual = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        if expected_arrays and actual != expected_arrays[key]:
            raise ValueError("Saved V array fingerprint mismatch")
        fingerprints[key] = actual
    error = np.where(mask, prediction.astype(np.float64)-target.astype(np.float64), 0.)
    if not np.isfinite(error).all():
        raise ValueError("Saved V has nonfinite observed error")
    sse = np.sum(error*error, axis=0, dtype=np.float64)
    sae = np.sum(np.abs(error), axis=0, dtype=np.float64)
    counts = mask.sum(axis=0)
    total = counts.sum(axis=0)
    if not np.all(total > 0):
        raise ValueError("Saved V has no observations for registered channel")
    mse = float(np.mean(sse.sum(axis=0)/total)); mae = float(np.mean(sae.sum(axis=0)/total))
    if max(abs(mse-float(run["MSE"])), abs(mae-float(run["MAE"]))) > 1e-10:
        raise ValueError("Saved V FP64 score disagrees with published selected run")
    return dict(MSE=mse, MAE=mae, horizon_sse=sse.tolist(), horizon_sae=sae.tolist(), horizon_counts=counts.tolist(),
                channel_sse=sse.sum(axis=0).tolist(), channel_sae=sae.sum(axis=0).tolist(), channel_counts=total.tolist(),
                array_fingerprints=fingerprints, dtype=str(prediction.dtype), independent_FP64_saved_rescore=True)


def number(value):
    if value in (None, "", "NA", "N/A", "null", "None"):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, values, fields=None):
    fields = fields or list(dict.fromkeys(key for row in values for key in row)) or ["status", "reason"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in values:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                         for key, value in row.items()})
    atomic(path, stream.getvalue())


def _source(path, ctx, bindings):
    path = Path(path).resolve()
    repo = Path(ctx["repo"]).resolve()
    if not path.is_relative_to(repo):
        raise ValueError("Public quantitative source outside repository")
    relative = path.relative_to(repo).as_posix()
    bindings[relative] = sha(path)
    return relative, bindings[relative]


def _legacy(ctx, bindings):
    source = Path(ctx["repo"]) / "reports/informs_ppt_ready_20261017/PPT_MAIN_TABLES.csv"
    relative, digest = _source(source, ctx, bindings)
    results, seen = [], set()
    for row in rows(source):
        if number(row.get("MSE")) is None or row.get("dataset") not in ("jena", "pems08", "pems08_C32"):
            continue
        identity = (row["run_id"], row["seed"])
        if identity in seen:
            continue
        seen.add(identity)
        results.append(dict(run_id=row["run_id"], cohort="pems32" if row["dataset"] == "pems08_C32" else row["dataset"], method=row["method"],
            backbone="NONE" if row["method"] in ("Ridge", "Direct G", "Fixed no-Bolt", "Joint no-Bolt") else "Chronos-Bolt-small",
            backbone_revision="772f3d25d38aec6d914c8949dab4462e2d46f5d8",
            seed=number(row["seed"]), C=number(row["C"]), K=number(row["K"]), L=number(row["L"]), H=number(row["H"]),
            fit="historical_selected", selection="A_stop historical registered recipe", refit="fresh_A60",
            input_info_scope="Original source recipe; Raw own-channel; B2 all cohort past",
            supervision_scope="A_fit/A_stop; fresh A60", device=row["accuracy_device"], dtype="source_bound",
            point_statistic="source_bound_point_forecast", trainable_params=number(row["trainable_parameters"]),
            MSE=number(row["MSE"]), MAE=number(row["MAE"]), epochs=number(row["selected_epoch"]),
            updates=number(row["refit_updates"]), split="V_select", metric_scale="canonical_A60",
            status="REUSED_VERIFIED_PUBLIC_SOURCE", origin="EXISTING_RESULT_REUSED", source_file=relative, source_sha256=digest,
            **{key: row.get(key) for key in ("scaler_sha256", "target_sha256", "mask_sha256", "origins_sha256", "channel_sha256", "checkpoint_sha256")}))
    stopped_path = Path(ctx["repo"]) / "reports" / OLD / "COMPARISONS_AT_STOP.csv"
    stopped_relative, stopped_sha = _source(stopped_path, ctx, bindings)
    for row in rows(stopped_path):
        if row["comparison_id"] != "SELECTED_A60" or row["model"] == "B2_CAP_POINTWISE":
            continue
        if (row.get("checkpoint_sha256"), row["seed"]) in [(r.get("checkpoint_sha256"), str(int(r["seed"]))) for r in results if r.get("seed") is not None]:
            continue
        results.append(dict(run_id="historical_"+row["model"]+"_s"+row["seed"], cohort="jena", method=row["model"],
            backbone="Chronos-Bolt-small", seed=number(row["seed"]), C=21, K=6, L=512, H=48,
            MSE=number(row["MSE"]), MAE=number(row["MAE"]), epochs=number(row["epoch"]), updates=number(row["updates"]),
            exposures=number(row["exposures"]), trainable_params=number(row["parameters"]), split="V_select",
            metric_scale="canonical_A60", fit="historical_selected", selection=row["selection"], refit="fresh_A60",
            status="REUSED_VERIFIED_PUBLIC_SOURCE", origin="EXISTING_RESULT_REUSED", source_file=stopped_relative, source_sha256=stopped_sha,
            **{key: row.get(key) for key in ("scaler_sha256", "target_sha256", "mask_sha256", "origins_sha256", "checkpoint_sha256")}))
    for run in results:
        if "_DIRECT_G_" in run["run_id"].upper():
            run.update(backbone="NONE", source_family="DIRECT_G", routing="no_foundation",
                       input_info_scope="Own-channel history; channel-shared L-hidden-H innovation plus current value")
            for filename in ("networks.py", "fit_base.py"):
                _source(Path(ctx["repo"])/"research"/BROAD/filename, ctx, bindings)
        run["point_statistic"] = "deterministic_ridge" if run["method"] == "Ridge" else "deterministic_point_regression" if run["backbone"] == "NONE" else "quantile_0.5"
        run["backend"] = "numpy_blas" if run["method"] == "Ridge" else "torch_eager"
        if run["method"] in ("B0", "B2"):
            run["dtype"] = "float32"
            run.setdefault("device", "cpu")
        run["cohort"] = str(run["cohort"])
        if run["backbone"] == "NONE":
            run["backbone_revision"] = None
    return results


def legacy_statistics(ctx, data):
    """Follow a public hash to the existing private manifest inventory, then only V NPZs."""
    if "asset" not in ctx:
        return []
    repo, asset = Path(ctx["repo"]), Path(ctx["asset"])
    binding_path = repo/"reports/informs_ppt_ready_20261017/SOURCE_BINDINGS.json"
    _source(binding_path, ctx, data["bindings"])
    evidence_path = asset/".cache/informs_ppt_ready_20261017/LEGACY_TABLE_EVIDENCE.json"
    if not evidence_path.is_file():
        return [dict(status="BLOCKED_MISSING_ASSET", reason="Previously pinned legacy evidence inventory unavailable")]
    binding = read(binding_path)
    if sha(evidence_path) != binding["evidence_sha256"]:
        raise ValueError("Legacy evidence inventory no longer matches public binding")
    evidence = read(evidence_path)
    records = {entry["run_id"]: entry for entry in evidence["rows"] if entry.get("split") == "V_select"}
    covered = {row["run_id"] for row in data["channel"]}
    replay = []
    for run in data["runs"]:
        if run.get("origin") != "EXISTING_RESULT_REUSED" or run["run_id"] not in records:
            continue
        record = records[run["run_id"]]
        descriptor = record.get("manifest_source")
        if not descriptor or not descriptor.get("absolute_private"):
            continue
        manifest_path = Path(descriptor["absolute_private"])
        if not manifest_path.is_file():
            replay.append(dict(run_id=run["run_id"], status="BLOCKED_MISSING_ASSET", reason="Bound original V manifest unavailable"))
            continue
        if sha(manifest_path) != descriptor["sha256"]:
            raise ValueError("Bound original V manifest changed")
        manifest = read(manifest_path)
        if manifest["split"] != "V_select" or manifest["run_id"] != run["run_id"] or manifest["scaler_sha"] != run["scaler_sha256"]:
            raise ValueError("Legacy V manifest/scaler differs from selected table")
        cache = (asset/".cache"/BROAD).resolve()
        cache_relative = manifest.get("cache_relative_path")
        if cache_relative is None:
            manifest_relative = Path(descriptor["path"]).relative_to(Path("research")/BROAD)
            if manifest_relative.name != "PREDICTION_MANIFEST.json" or manifest_relative.parent.name != "V_select":
                raise ValueError("Legacy registered prediction layout differs")
            cache_relative = manifest_relative.parent.with_suffix(".npz")
        archive_path = (cache/cache_relative).resolve()
        if not archive_path.is_relative_to(cache):
            raise ValueError("Legacy V cache reference escapes pinned root")
        if not archive_path.is_file():
            replay.append(dict(run_id=run["run_id"], status="BLOCKED_MISSING_ASSET", reason="Hash-bound saved V archive unavailable"))
            continue
        score = score_prediction_archive(archive_path, manifest["archive_sha256"], manifest["arrays"], run)
        run.update(contract_metadata(score), independent_private_score_sha256=manifest["archive_sha256"],
                   private_manifest_sha256=descriptor["sha256"], G6_channel_coverage="INDEPENDENT_PRIVATE_SAVED_V_FP64",
                   G6_horizon_coverage="INDEPENDENT_PRIVATE_SAVED_V_FP64")
        run["_archive_path"] = archive_path
        run.update(canonical_cohort(run, repo))
        if run["run_id"] in covered:
            data["horizon"][:] = [row for row in data["horizon"] if row["run_id"] != run["run_id"]]
            data["channel"][:] = [row for row in data["channel"] if row["run_id"] != run["run_id"]]
        _stats_from_result(score, run, data["horizon"], data["channel"])
        covered.add(run["run_id"])
        replay.append(dict(run_id=run["run_id"], status="PASS_INDEPENDENT_STORED_FP64", source_manifest_reference=descriptor["path"],
            source_manifest_sha256=descriptor["sha256"], private_archive_sha256=manifest["archive_sha256"],
            private_inventory_sha256=binding["evidence_sha256"], MSE=score["MSE"], MAE=score["MAE"],
            C=run["C"], H=run["H"], V_origins=score["array_fingerprints"]["origins"]["shape"][0],
            public_channel_statistics=True, public_horizon_channel_statistics=True, model_calls=0, protected_T_access=False))
    return replay


def complete_new_metadata(ctx, results, data):
    if "cache" not in ctx:
        return
    import numpy as np
    replay = data.setdefault("new_stats_replay", [])
    for identifier, (job, result, receipt_path) in results.items():
        if job["question_id"] not in ("B02", "B03", "C01", "C02", "D02", "D03"):
            continue
        candidates = []
        bindings = list(result.get("private_artifacts", []))
        if result.get("private_prediction_ref") and result.get("private_prediction_archive_sha256"):
            bindings.append(dict(reference=result["private_prediction_ref"], sha256=result["private_prediction_archive_sha256"]))
        if job["question_id"] == "C01":
            receipt = read(receipt_path)
            handler_relative = "jobs/"+identifier+"_handler.json"
            if handler_relative in receipt.get("artifact_hashes", {}):
                from .existing_jobs import _identity
                handler_path = Path(ctx["report"])/handler_relative
                if sha(handler_path) != receipt["artifact_hashes"][handler_relative]:
                    raise ValueError("C01 handler source binding changed")
                handler = read(handler_path)
                if handler.get("job_id") != identifier or handler.get("scientific_identity_sha256") != _identity(job):
                    raise ValueError("C01 handler scientific identity differs")
                _source(handler_path, ctx, data["bindings"])
                bindings.extend(dict(reference=value["path"], sha256=value["sha256"]) for value in handler.get("artifact_hashes", []))
        for binding in bindings:
            reference = binding.get("reference", "")
            if ("://" not in reference or not reference.endswith(".npz") or "coefficient" in reference.lower()
                    or ("prediction" not in reference.lower() and binding.get("sha256") != result.get("private_prediction_archive_sha256"))):
                continue
            scheme, relative = reference.split("://", 1)
            if scheme not in ctx:
                continue
            root = Path(ctx[scheme]).resolve(); path = (root/relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha(path) != binding["sha256"]:
                raise ValueError("New saved V metadata binding rejected")
            candidates.append((binding, path))
        relevant = [r for r in data["runs"] if r["run_id"] == identifier or r["run_id"].startswith(identifier+"_")]
        for run in relevant:
            choice = next(((binding, path) for binding, path in candidates if binding["sha256"] == run.get("_prediction_sha256")), None)
            if job["question_id"] == "B02" and candidates:
                choice = candidates[0]
            if choice is None:
                continue
            binding, path = choice
            if job["question_id"] == "B02":
                contract = read(Path(ctx["repo"])/"research"/BROAD/"DATA_CONTRACT.json")["assets"]
                columns = contract["jena"]["columns"] if job["dataset_cohort"] == "jena" else read(Path(ctx["repo"])/"research"/BROAD/"priority_pems_subset_20261016_v1/PLAN.json")["column_selection"]["selected_ids"]
                with np.load(path, allow_pickle=False) as archive:
                    key = "own_prediction" if run["method"].startswith("OWN") else "cross_prediction"
                    values = dict(pred_normalized=archive[key], target_normalized=archive["target"], observed=archive["observed"],
                                  origins=archive["origins"], channel_ids=np.asarray(columns, dtype=str))
                scored = score_prediction_values(values, None, run)
            else:
                scored = score_prediction_archive(path, binding["sha256"], None, run)
            run.update(contract_metadata(scored), independent_private_score_sha256=binding["sha256"])
            run["evaluation_contract_derivation"] = "Independent bound saved prediction: canonical raw array-byte fingerprints; original producer's encoded fingerprints remain in its immutable source"
            if job["question_id"] == "B02":
                run["channel_order_validation"] = "DERIVED_REGISTERED_ORDER_FROM_BOUND_B02_WRITER"
                _source(Path(ctx["repo"])/"research"/BROAD/"DATA_CONTRACT.json", ctx, data["bindings"])
                _source(Path(ctx["repo"])/"research"/NAME/"ridge_jobs.py", ctx, data["bindings"])
                if job["dataset_cohort"] == "pems32":
                    _source(Path(ctx["repo"])/"research"/BROAD/"priority_pems_subset_20261016_v1/PLAN.json", ctx, data["bindings"])
            run["_archive_path"] = path
            data["horizon"][:] = [row for row in data["horizon"] if row["run_id"] != run["run_id"]]
            data["channel"][:] = [row for row in data["channel"] if row["run_id"] != run["run_id"]]
            _stats_from_result(scored, run, data["horizon"], data["channel"])
            replay.append(dict(run_id=run["run_id"], status="PASS_INDEPENDENT_STORED_FP64", private_archive_sha256=binding["sha256"],
                               MSE=scored["MSE"], MAE=scored["MAE"], arrays=scored["array_fingerprints"], model_calls=0))


class _FitMetadataValue:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _FitMetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) in {("_codecs", "encode"), ("numpy", "dtype"), ("numpy", "ndarray"),
                              ("numpy._core.multiarray", "_reconstruct"), ("torch", "FloatStorage"),
                              ("torch._utils", "_rebuild_tensor_v2")}:
            return _FitMetadataValue
        raise ValueError("Unsupported fit metadata global: "+module+"."+name)

    def persistent_load(self, identifier):
        if not isinstance(identifier, tuple) or len(identifier) != 5 or identifier[0] != "storage":
            raise ValueError("Unsupported fit metadata persistent identifier")
        return _FitMetadataValue()


def fit_bundle_metadata(path):
    """Decode only the observed pickle metadata schema, replacing all tensor/NumPy globals."""
    with zipfile.ZipFile(path) as archive:
        entries = [item for item in archive.infolist() if item.filename.endswith("/data.pkl")]
        if len(entries) != 1 or entries[0].file_size > 1024*1024:
            raise ValueError("Unsupported fit metadata archive layout or size")
        metadata = archive.read(entries[0])
        if any(op.name in ("EXT1", "EXT2", "EXT4") for op, _, _ in pickletools.genops(metadata)):
            raise ValueError("Unsupported fit metadata extension opcode")
        value = _FitMetadataReader(io.BytesIO(metadata)).load()
    if not isinstance(value, dict) or any(not isinstance(value.get(key), dict) for key in ("spec", "search_summary", "refit_summary")):
        raise ValueError("Unsupported fit metadata stage schema")
    return {key: value[key] for key in ("spec", "search_summary", "refit_summary")}


def complete_fit_metadata(ctx, results, data):
    if "cache" not in ctx:
        return
    cache, report = Path(ctx["cache"]).resolve(), Path(ctx["report"])
    for identifier, (job, result, receipt_path) in results.items():
        if job["question_id"] not in ("C01", "C02", "D02"):
            continue
        candidates = list(result.get("private_artifacts", []))
        public_sources = [receipt_path]
        if job["question_id"] == "C01":
            from .existing_jobs import _identity
            relative = "jobs/"+identifier+"_handler.json"
            expected = read(receipt_path).get("artifact_hashes", {}).get(relative)
            if expected is None:
                continue
            handler_path = report/relative
            if sha(handler_path) != expected:
                raise ValueError("C01 fit handler SHA binding rejected")
            handler = read(handler_path)
            if handler.get("job_id") != identifier or handler.get("scientific_identity_sha256") != _identity(job):
                raise ValueError("C01 fit handler scientific identity differs")
            candidates += [dict(reference=item["path"], sha256=item["sha256"]) for item in handler.get("artifact_hashes", [])]
            public_sources.append(handler_path)
        by_reference = {item.get("reference"): item for item in candidates}
        for run in data["runs"]:
            if not (run["run_id"] == identifier or run["run_id"].startswith(identifier+"_")):
                continue
            method, seed = run["method"], job["seed"]
            if job["question_id"] == "C02" and method in ("B2_PCA", "B2_QR"):
                source_run = "init_"+method.removeprefix("B2_")+"_s"+str(seed)
            elif job["question_id"] == "D02" and method in ("Moirai_B0", "Moirai_B2"):
                source_run = "moirai_jena_"+method.removeprefix("Moirai_")+"_k6_s"+str(seed)
            elif job["question_id"] == "C01" and method in ("B0", "B2"):
                source_run = "pems_"+method
            else:
                if method == "Moirai_Raw":
                    run["fit_time_scope"] = "No target parameter training; scaler/preparation time UNKNOWN; overall fit time is not zero"
                continue
            reference = "cache://runs/"+source_run+"/selected_bundle.pt"
            binding = by_reference.get(reference)
            if binding is None:
                run["fit_time_scope"] = "UNKNOWN: no previously bound selected bundle metadata"
                continue
            path = (cache/reference.removeprefix("cache://")).resolve()
            if not path.is_relative_to(cache) or not path.is_file() or sha(path) != binding["sha256"]:
                raise ValueError("Fit bundle SHA binding rejected")
            try:
                payload = fit_bundle_metadata(path)
            except (ValueError, pickle.UnpicklingError, EOFError, zipfile.BadZipFile) as error:
                run["fit_time_scope"] = "UNKNOWN: "+str(error)
                continue
            search, refit, spec = payload["search_summary"], payload["refit_summary"], payload["spec"]
            sources = list(public_sources)
            expected_source_run = source_run
            if job["question_id"] == "C01":
                relative = "runs/"+source_run+"/DEVICE_BRANCH_LINEAGE.json"
                expected = read(receipt_path).get("artifact_hashes", {}).get(relative)
                if expected is None:
                    run["fit_time_scope"] = "UNKNOWN: no bound original-run lineage for resumed stage timers"
                    continue
                lineage_path = report/relative
                if sha(lineage_path) != expected:
                    raise ValueError("C01 fit lineage SHA binding rejected")
                original = read(lineage_path).get("original_checkpoint_ref", "")
                expected_source_run = Path(original.split("://", 1)[-1]).parent.parent.name
                sources.append(lineage_path)
            if (spec.get("run_id") != expected_source_run or spec.get("seed") != seed or spec.get("device") != run.get("device")
                    or search.get("selected_epoch") != run.get("epochs") or refit.get("updates") != run.get("updates")
                    or refit.get("exposures") != run.get("exposures") or refit.get("contract", {}).get("scaler_sha") != run.get("scaler_sha256")):
                raise ValueError("Fit bundle differs from selected run identity/progress/scaler")
            for stage, summary in (("search", search), ("refit", refit)):
                relative = stage.upper()+"_SUMMARY.json" if job["question_id"] == "C01" else stage+"/RECEIPT.json"
                public = report/"runs"/source_run/relative
                if not public.is_file() or read(public) != summary:
                    raise ValueError("Fit stage payload differs from previously bound bundle")
                if summary.get("status") != "SEGMENT_COMPLETE" or number(summary.get("seconds")) is None or float(summary["seconds"]) < 0:
                    raise ValueError("Fit stage timing is incomplete or invalid")
                sources.append(public)
            refs = [binding]
            for public in sources:
                name, digest = _source(public, ctx, data["bindings"])
                refs.append(dict(reference="repo://"+name, sha256=digest))
            scope = ("GPU remainder search after historical CPU checkpoint plus fresh GPU A60 refit; historical CPU time excluded. "
                     if job["question_id"] == "C01" and run["device"] == "cuda" else
                     "Original-device remaining search plus fresh A60 refit; preceding checkpoint sessions excluded. "
                     if job["question_id"] == "C01" else "Fresh complete search plus selected fresh A60 refit. ")
            scope += ("Stage wall timers include SGD, A_stop endpoint selection evaluation, checkpoint/event I/O and source checks; "
                      "model/scaler/PCA/F load initialization, final V evaluation and selected-bundle serialization excluded; "
                      "pure SGD versus selection-evaluation time UNKNOWN separately.")
            run.update(search_selection_stage_seconds=search["seconds"], refit_stage_seconds=refit["seconds"],
                       fit_stage_total_seconds=search["seconds"]+refit["seconds"], fit_time_scope=scope,
                       fit_device=spec["device"], fit_source_refs=refs)
            data.setdefault("new_stats_replay", []).append(dict(run_id=run["run_id"], status="PASS_PREVIOUSLY_BOUND_FIT_STAGE_METADATA",
                private_artifacts=[binding], public_verification_sources=refs[1:],
                search_selection_stage_seconds=search["seconds"], refit_stage_seconds=refit["seconds"],
                tensor_storage_loaded=False, model_calls=0))
    for cost in data["costs"]:
        run = cost_accuracy_match(cost, data["runs"])
        if run is not None and run.get("fit_stage_total_seconds") is not None:
            cost.update({key: run[key] for key in FIT_FIELDS})
            cost.update(fit_seconds=run["fit_stage_total_seconds"], fit_source_run_id=run["run_id"],
                        fit_source_run_sha256=run.get("source_sha256"))


def complete_cost_metadata(ctx, results, data):
    for cost in data["costs"]:
        job, result, receipt_path = results[cost["job_id"]]
        if (job["question_id"] == "E02" and cost["method"] in ("B0", "B2") and "asset" in ctx
                and cost_accuracy_match(cost, data["runs"]) is None):
            route = result.get("bindings", {}).get(cost["source_method"], {})
            expected = cost.get("accuracy_archive_sha256")
            candidates = [run for run in data["runs"] if route.get("artifact_sha256")
                          and run.get("checkpoint_sha256") == route["artifact_sha256"]]
            candidate_cost = dict(cost, accuracy_archive_sha256=None)
            run = cost_accuracy_match(candidate_cost, candidates)
            binding = next((item for item in result.get("private_artifacts", [])
                            if expected and item.get("sha256") == expected and item.get("reference", "").endswith("/V_select.npz")), None)
            if run is not None and binding is not None:
                scheme, relative = binding["reference"].split("://", 1)
                if scheme not in ctx:
                    continue
                root = Path(ctx[scheme]).resolve(); path = (root/relative).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("E02 saved V reference escapes bound root")
                score = score_prediction_archive(path, expected, None, run)
                actual = contract_metadata(score)
                if any(not run.get(key) or run[key] != actual.get(key)
                       for key in ("target_sha256", "mask_sha256", "origins_sha256", "channel_sha256")):
                    raise ValueError("E02 saved V canonical arrays disagree with selected run")
                run.update(independent_private_score_sha256=expected)
                run["_archive_path"] = path
                if "horizon" in data and "channel" in data:
                    for key in ("horizon", "channel"):
                        data[key][:] = [row for row in data[key] if row["run_id"] != run["run_id"]]
                    _stats_from_result(score, run, data["horizon"], data["channel"])
                name, digest = _source(receipt_path, ctx, data["bindings"])
                data.setdefault("new_stats_replay", []).append(dict(run_id=run["run_id"], status="PASS_E02_BOUND_STORED_FP64_JOIN",
                    MSE=score["MSE"], MAE=score["MAE"], arrays=score["array_fingerprints"],
                    checkpoint_sha256=route["artifact_sha256"], private_artifacts=[binding],
                    public_verification_sources=[dict(reference="repo://"+name, sha256=digest)], model_calls=0))
        if job["question_id"] != "E01":
            continue
        for relative, expected in read(receipt_path).get("artifact_hashes", {}).items():
            if not relative.endswith("COST_TRIALS.csv.gz"):
                continue
            path = (Path(ctx["report"])/relative).resolve()
            if not path.is_relative_to(Path(ctx["report"]).resolve()) or sha(path) != expected or (result.get("trials_sha256") and expected != result["trials_sha256"]):
                raise ValueError("Cost trial geometry source binding changed")
            with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
                trials = list(csv.DictReader(stream))
            if not trials:
                raise ValueError("Measured cost trial source is empty")
            geometry = {tuple(int(row[key]) for key in ("F_logical_streams", "F_physical_streams", "F_calls")) for row in trials}
            if len(geometry) != 1 or (result.get("status") == "COMPLETE_PROTOCOL" and len(trials) != 50):
                raise ValueError("Measured F trial geometry is not constant or protocol count differs")
            logical, physical, calls = next(iter(geometry))
            source_file, source_sha = _source(path, ctx, data["bindings"])
            cost.update(logical_F_series_rows=logical, physical_F_series_rows=physical, F_calls=calls,
                        trial_geometry_source=source_file, trial_geometry_source_sha256=source_sha, measured_trial_rows=len(trials))
        run = cost_accuracy_match(cost, data["runs"])
        if run is not None and cost.get("trainable_params") is None:
            cost.update(trainable_params=run.get("trainable_params"), parameter_source_run_id=run["run_id"],
                        parameter_source_sha256=run["source_sha256"], parameter_source_archive_sha256=run.get("independent_private_score_sha256"),
                        parameter_derivation="Trainable count from exact archive-bound C01 accuracy source; resident count remains unmeasured")


def complete_device_replay_metadata(ctx, data):
    if "cache" not in ctx:
        return
    for run in data["runs"]:
        if run.get("origin") != "EXISTING_SELECTED_ROUTE_DEVICE_PARITY_REPLAY":
            continue
        path = Path(run["_archive_path"])
        digest = run["independent_private_score_sha256"]
        score = score_prediction_archive(path, digest, None, run)
        actual = contract_metadata(score)
        if any(run.get(field) and run[field] != actual.get(field) for field in ARRAY_KEYS):
            raise ValueError("Device replay canonical score arrays differ")
        run.update(actual, G6_channel_coverage="INDEPENDENT_DEVICE_REPLAY_SAVED_FP64",
                   G6_horizon_coverage="INDEPENDENT_DEVICE_REPLAY_SAVED_FP64")
        for key in ("horizon", "channel"):
            data[key][:] = [row for row in data[key] if row["run_id"] != run["run_id"]]
        _stats_from_result(score, run, data["horizon"], data["channel"])
        root = Path(ctx["cache"]).resolve()
        if not path.resolve().is_relative_to(root):
            raise ValueError("Device replay archive escapes campaign cache")
        data.setdefault("new_stats_replay", []).append(dict(run_id=run["run_id"],
            status="PASS_INDEPENDENT_DEVICE_REPLAY_FP64", MSE=score["MSE"], MAE=score["MAE"], arrays=score["array_fingerprints"],
            private_artifacts=[dict(reference="cache://"+path.resolve().relative_to(root).as_posix(), sha256=digest)],
            public_verification_sources=[dict(reference="repo://"+run["source_file"], sha256=run["source_sha256"])], model_calls=0))


def complete_probe_metadata(ctx, results, data):
    if "cache" not in ctx:
        return
    import numpy as np
    from .existing_jobs import _identity, _reference, _array_fingerprint
    previous_path = Path(ctx["report"])/"NEW_STATS_REPLAY.json"
    previous = {row["run_id"]: row for row in read(previous_path).get("rows", []) if row.get("binding_captured_at_verification")} if previous_path.is_file() else {}
    for identifier, (job, result, receipt_path) in results.items():
        if job["question_id"] != "B01":
            continue
        run = next(row for row in data["runs"] if row["run_id"] == identifier)
        archive = Path(ctx["cache"])/"probes"/identifier/"V_PREDICTIONS_FP64.npz"
        prior = previous.get(identifier, {})
        if not archive.is_file():
            if prior:
                raise ValueError("Previously late-bound probe archive is missing")
            data.setdefault("new_stats_replay", []).append(dict(run_id=identifier, status="BLOCKED_MISSING_SAVED_PROBE", historical_prediction_hash_binding=False))
            continue
        digest = sha(archive)
        if prior.get("scientific_identity") == job["scientific_identity"] and prior.get("private_archive_sha256") != digest:
            raise ValueError("Previously late-bound probe archive changed")
        receipt = read(receipt_path)
        source_paths = [Path(ctx["report"])/"probes"/identifier/"RESULT.json", Path(ctx["report"])/"jobs"/(identifier+"_handler.json")]
        for path in source_paths:
            expected = receipt.get("artifact_hashes", {}).get(path.relative_to(ctx["report"]).as_posix())
            if expected is None or sha(path) != expected:
                raise ValueError("Probe public RESULT/handler source binding differs")
            _source(path, ctx, data["bindings"])
        result_path, handler_path = source_paths
        handler = read(handler_path)
        if read(result_path) != result or handler.get("job_id") != identifier or handler.get("scientific_identity_sha256") != _identity(job):
            raise ValueError("Probe RESULT/handler scientific identity differs")
        coefficient = Path(ctx["cache"])/"probes"/identifier/"RIDGE_COEFFICIENTS_FP64.npz"
        coefficient_reference = _reference(coefficient, ctx)
        coefficient_expected = next((entry["sha256"] for entry in handler.get("artifact_hashes", []) if entry["path"] == coefficient_reference), None)
        if coefficient_expected is None or not coefficient.is_file() or sha(coefficient) != coefficient_expected:
            raise ValueError("Probe coefficient binding differs")
        reference = next((row for row in data["runs"] if row.get("origin") == "EXISTING_RESULT_REUSED" and row["cohort"] == "jena"
                          and row.get("_archive_path") and row.get("scaler_sha256") == run.get("scaler_sha256")), None)
        if reference is None:
            data.setdefault("new_stats_replay", []).append(dict(run_id=identifier, status="BLOCKED_MISSING_CANONICAL_REFERENCE", historical_prediction_hash_binding=False))
            continue
        reference_path = Path(reference["_archive_path"])
        if sha(reference_path) != reference["independent_private_score_sha256"]:
            raise ValueError("Probe canonical Jena V reference archive changed")
        public_reference = Path(ctx["repo"])/reference["source_file"]
        if sha(public_reference) != reference["source_sha256"]:
            raise ValueError("Probe canonical reference public source changed")
        _source(public_reference, ctx, data["bindings"])
        with np.load(reference_path, allow_pickle=False) as source:
            canonical = {key: source[key] for key in ("target_normalized", "observed", "origins", "channel_ids")}
        with np.load(archive, allow_pickle=False) as source:
            if set(source.files) != {"prediction", "y", "mask", "origins"}:
                raise ValueError("Unexpected saved probe prediction schema")
            saved = {key: source[key] for key in source.files}
        if saved["prediction"].dtype != np.float64 or saved["y"].dtype != np.float64:
            raise ValueError("Saved probe forecast/target must be FP64")
        if not np.array_equal(saved["y"], canonical["target_normalized"].astype(np.float64), equal_nan=True):
            raise ValueError("Saved probe target differs from canonical FP32-valued V target")
        if not np.array_equal(saved["mask"], canonical["observed"]) or not np.array_equal(saved["origins"], canonical["origins"]):
            raise ValueError("Saved probe mask/origins differ from canonical V")
        contract_path = Path(ctx["repo"])/"research"/BROAD/"DATA_CONTRACT.json"
        columns = np.asarray(read(contract_path)["assets"]["jena"]["columns"], dtype=str)
        if not np.array_equal(columns, canonical["channel_ids"]):
            raise ValueError("Saved probe reference channel order differs from registry")
        _source(contract_path, ctx, data["bindings"])
        writer_path = Path(ctx["repo"])/"reports"/OLD/"code/linear_probe.py"
        _source(writer_path, ctx, data["bindings"])
        score = score_prediction_values(dict(pred_normalized=saved["prediction"], target_normalized=saved["y"], observed=saved["mask"],
            origins=saved["origins"], channel_ids=columns), None, run)
        observed = np.asarray(result["channel_observed"], np.int64)
        if not np.array_equal(observed, score["channel_counts"]):
            raise ValueError("Saved probe observed counts differ from public RESULT")
        differences = [np.abs(np.asarray(score[key])-np.asarray(result[source], np.float64))
                       for key, source in (("channel_sse", "channel_SSE"), ("channel_sae", "channel_SAE"))]
        mean_difference = max(float((value/observed).max()) for value in differences)
        if mean_difference > 1e-10:
            raise ValueError("Saved probe channel statistics differ from public RESULT")
        canonical_fingerprints = {key: _array_fingerprint(value) for key, value in canonical.items()}
        run.update(contract_metadata(score), **{field: canonical_fingerprints[key]["sha256"] for field, key in ARRAY_KEYS.items()},
            independent_private_score_sha256=digest, channel_order_validation="DERIVED_REGISTERED_ORDER_WITH_EXACT_REFERENCE_TARGETS",
            evaluation_contract_derivation="Saved FP64 y equals canonical reference FP32 target values exactly; target SHA denotes reference bytes. Channel IDs derive from registered order plus exact reference-column equality.",
            prediction_binding_scope="LATE_VERIFICATION_CAPTURE; historical prediction immutability unproven",
            G6_channel_coverage="INDEPENDENT_LATE_BOUND_SAVED_PROBE_FP64", G6_horizon_coverage="INDEPENDENT_LATE_BOUND_SAVED_PROBE_FP64")
        data["horizon"][:] = [row for row in data["horizon"] if row["run_id"] != identifier]
        data["channel"][:] = [row for row in data["channel"] if row["run_id"] != identifier]
        _stats_from_result(score, run, data["horizon"], data["channel"])
        private_bindings = [dict(reference=_reference(path, ctx), sha256=sha(path)) for path in (archive, coefficient, reference_path)]
        data.setdefault("new_stats_replay", []).append(dict(run_id=identifier, scientific_identity=job["scientific_identity"],
            status="PASS_LATE_BOUND_SAVED_FP64", binding_captured_at_verification=True,
            binding_capture_utc=prior.get("binding_capture_utc", utc()), historical_prediction_hash_binding=False,
            private_archive_sha256=digest, private_artifacts=private_bindings,
            public_verification_sources=[dict(reference="repo://"+path.relative_to(ctx["repo"]).as_posix(), sha256=sha(path))
                                         for path in (receipt_path, result_path, handler_path, public_reference, writer_path, contract_path)],
            archive_arrays={key: _array_fingerprint(value) for key, value in saved.items()}, canonical_evaluation_arrays=canonical_fingerprints,
            exact_target_numeric_equality=True, exact_mask_and_origins_equality=True, stored_channel_ids=False,
            derived_channel_order_source="DATA_CONTRACT assets.jena.columns; writer original target columns; exact canonical target-column equality",
            MSE=score["MSE"], MAE=score["MAE"], maximum_channel_statistic_raw_difference=max(float(value.max()) for value in differences),
            maximum_channel_statistic_observed_mean_difference=mean_difference, model_calls=0, protected_T_access=False,
            limitation="Prediction archive had no campaign-time SHA binding. Current saved state is captured after independent metric, channel-marginal and reference-array checks; historical immutability before this capture is not proved."))


def replay_binding_validation(ctx):
    from .core import bound_references_valid
    path = Path(ctx["report"])/"NEW_STATS_REPLAY.json"
    value = read(path) if path.is_file() else {"rows": []}
    roots = {key: ctx[key] for key in ("repo", "report", "asset", "cache") if key in ctx}
    return dict(source="NEW_STATS_REPLAY.json", sha256=sha(path) if path.is_file() else None,
                late_bound_rows=sum(bool(row.get("binding_captured_at_verification")) for row in value.get("rows", [])),
                all_bound_artifacts_verified=bound_references_valid(value, roots))


def _stats_from_result(result, run, horizon, channel):
    sse, sae, counts = result.get("horizon_sse"), result.get("horizon_sae"), result.get("horizon_counts")
    if sse is not None and sae is not None and counts is not None:
        for h in range(len(sse)):
            for c in range(len(sse[h])):
                horizon.append(dict(run_id=run["run_id"], cohort=run["cohort"], method=run["method"], seed=run.get("seed"),
                    split=run.get("split"), metric_scale=run.get("metric_scale"), horizon=h+1, channel_id=str(c),
                    SSE=sse[h][c], SAE=sae[h][c], count=counts[h][c], origins_sha256=run.get("origins_sha256"),
                    source_file=run["source_file"], source_sha256=run["source_sha256"],
                    private_archive_sha256=run.get("independent_private_score_sha256")))
    sse = result.get("channel_sse", result.get("channel_SSE"))
    sae = result.get("channel_sae", result.get("channel_SAE"))
    counts = result.get("channel_counts", result.get("channel_observed"))
    if sse is not None and sae is not None and counts is not None:
        for c in range(len(sse)):
            channel.append(dict(run_id=run["run_id"], cohort=run["cohort"], method=run["method"], seed=run.get("seed"),
                split=run.get("split"), metric_scale=run.get("metric_scale"), channel_id=str(c),
                SSE=sse[c], SAE=sae[c], count=counts[c], origins_sha256=run.get("origins_sha256"),
                source_file=run["source_file"], source_sha256=run["source_sha256"],
                private_archive_sha256=run.get("independent_private_score_sha256")))


def _legacy_stats(ctx, bindings, runs, horizon, channel):
    # Existing channel marginals retain the exact observed count per channel.
    source = Path(ctx["repo"]) / "reports" / OLD / "ERROR_DECOMPOSITION_CHANNEL.csv"
    relative, digest = _source(source, ctx, bindings)
    for row in rows(source):
        for method in ("B0", "B2"):
            run = next((r for r in runs if r["cohort"] == "jena" and r["method"] == method and r.get("seed") == number(row["seed"])), None)
            if run is None:
                continue
            count = int(row["full_channel_observed_count"])
            channel.append(dict(run_id=run["run_id"], cohort="jena", method=method, seed=run["seed"], split="V_select",
                metric_scale="canonical_A60", channel_id=str(row["channel_index"]), channel_label=row["label"],
                SSE=float(row[method+"_subset_macro_mse"])*count, SAE=float(row[method+"_subset_macro_mae"])*count,
                count=count, origins_sha256=run.get("origins_sha256"), source_file=relative, source_sha256=digest,
                lineage="Derived from historical per-channel FP64 score and explicit observed count"))
    horizon_source = Path(ctx["repo"]) / "reports" / OLD / "ERROR_DECOMPOSITION_HORIZON.csv"
    horizon_rel, horizon_sha = _source(horizon_source, ctx, bindings)
    for row in rows(horizon_source):
        if row["partition"] != "horizon_step":
            continue
        if int(row["n_observed"]) != 43*21 or int(row["active_channels"]) != 21:
            raise ValueError("Horizon marginal requires equal channel observed counts")
        for method in ("B0", "B2"):
            run = next((r for r in runs if r["cohort"] == "jena" and r["method"] == method and r.get("seed") == number(row["seed"])), None)
            if run:
                horizon.append(dict(run_id=run["run_id"], cohort="jena", method=method, seed=run["seed"], split="V_select",
                    metric_scale="canonical_A60", horizon=int(row["label"]), channel_id="ALL_CHANNELS_EQUAL_COUNTS",
                    SSE=float(row[method+"_subset_macro_mse"])*903, SAE=float(row[method+"_subset_macro_mae"])*903,
                    count=903, origins_sha256=run.get("origins_sha256"), source_file=horizon_rel, source_sha256=horizon_sha,
                    lineage="Historical horizon marginal;21channels equal43count; channel marginal separate"))


def collect(ctx):
    repo, report = Path(ctx["repo"]), Path(ctx["report"])
    plan_path = report / "EXPERIMENT_PLAN.json"
    plan = read(plan_path)
    validate_plan(plan, repo)
    bindings = {}
    _source(plan_path, ctx, bindings)
    runs = _legacy(ctx, bindings)
    job_rows, results = [], {}
    for job in plan["jobs"]:
        receipt_path = report / "receipts" / (job["job_id"]+".json")
        receipt = read(receipt_path) if receipt_path.is_file() else dict(status="PLANNED", reason="Not dispatched in current snapshot")
        if receipt_path.is_file():
            _source(receipt_path, ctx, bindings)
            for relative, expected in receipt.get("artifact_hashes", {}).items():
                artifact = (report/relative).resolve()
                if not artifact.is_relative_to(report.resolve()) or not artifact.is_file() or sha(artifact) != expected:
                    raise ValueError("Completed public receipt artifact changed: "+job["job_id"])
                _source(artifact, ctx, bindings)
        job_rows.append({**{key: job.get(key) for key in ("job_id", "question_id", "family", "dataset_cohort", "seed", "K", "backbone_revision", "scientific_identity", "device", "expected_cost", "dependency_ids")},
                         **{key: receipt.get(key) for key in ("status", "reason", "started_at", "finished_at", "consumed_resource", "artifact_hashes")}})
        if receipt.get("status") in SUCCESS and isinstance(receipt.get("result"), dict):
            results[job["job_id"]] = (job, receipt["result"], receipt_path)
    endpoints, probes, initializations, costs, horizon, channel, init_curves = [], [], [], [], [], [], []
    _legacy_stats(ctx, bindings, runs, horizon, channel)
    for identifier, (job, result, source) in results.items():
        relative, digest = _source(source, ctx, bindings)
        common = dict(run_id=identifier, cohort=job["dataset_cohort"], method=job["family"], seed=job.get("seed"),
            C=21 if job["dataset_cohort"] == "jena" else 32, K=job.get("K"), L=512, H=48 if job["dataset_cohort"] == "jena" else 12,
            backbone="Chronos-Bolt-small", backbone_revision=job.get("backbone_revision"), device=job["device"], backend="torch_eager", dtype="source_bound",
            status="COMPLETE", origin="NEW_CAMPAIGN_EXECUTION", source_file=relative, source_sha256=digest,
            scientific_identity=job["scientific_identity"], split="V_select", metric_scale="canonical_A60",
            fit="fresh_A60", refit="fresh_A60", selection="source_bound", input_info_scope="past observations only",
            supervision_scope="A_fit/A_stop/A60 only", point_statistic="quantile_0.5")
        question = job["question_id"]
        if question == "A03":
            endpoint = dict({**common, **contract_metadata(result)}, method=result.get("model", job["family"]), MSE=result.get("MSE"), MAE=result.get("MAE"),
                epoch=result.get("epoch"), updates=result.get("updates"), exposures=result.get("exposure", result.get("exposures")),
                split=result.get("eval"), metric_scale=result.get("metric_scale"), checkpoint_sha256=result.get("checkpoint_sha256"),
                fit=result.get("fit", "A_fit_search"), refit="NOT_APPLICABLE_SEARCH_ENDPOINT",
                selection=result.get("selection", "NOT_SELECTED_COMMON_ENDPOINT"),
                origin="NEW_CAMPAIGN_EXECUTION" if result.get("new_forward") else "EXISTING_RESULT_REUSED",
                new_forward=result.get("new_forward"), trainable_params=result.get("params"),
                parity=result.get("parity"), foundation_sha256=result.get("foundation_sha256"),
                source_contract_sha256=result.get("source_contract_sha256"), source_code_sha256=result.get("source_code_sha256"))
            endpoints.append(endpoint)
            _stats_from_result(result, endpoint, horizon, channel)
            continue
        if question == "A02":
            run = dict({**common, **contract_metadata(result)}, method="CAP", MSE=result["MSE"], MAE=result["MAE"], C=21, K=6, H=48,
                       trainable_params=35708, epochs=1, updates=752, exposures=12020, selection="A_stop CP0/.25/1/3",
                       origin="EXISTING_PREDICTION_NEW_INDEPENDENT_RESCORE", scaler_sha256=result.get("scaler_sha256"),
                       dtype=result.get("stored_precision", "float32"), device="cpu",
                       independent_private_score_sha256=result.get("source_archive_sha256"))
            runs.append(run)
            _stats_from_result(result, run, horizon, channel)
        elif question == "B01":
            metadata = result.get("metadata", {})
            run = dict(common, method=job["family"]+"_FULL_HISTORY_PROBE", MSE=result.get("MSE"), MAE=result.get("MAE"),
                trainable_params=result.get("parameter_count_with_intercept"), selection="lambda=1 fixed before V", dtype="float64", backend="numpy_blas", point_statistic="deterministic_ridge",
                input_info_scope="flattened full [512,6] fixed P/E latent history; no F/D", scaler_sha256=metadata.get("scaler_sha256"))
            runs.append(run)
            probes.append(dict(run, lambda_value=1, feature_dimension=result.get("input_features"), coefficient_count=result.get("coefficient_count"),
                PE_sha256=result.get("adapter_state_sha256"), F_called=result.get("foundation_called"), wall_seconds=result.get("wall_seconds")))
            _stats_from_result(result, run, horizon, channel)
        elif question == "B02":
            for method, metrics in result.get("methods", {}).items():
                run = dict({**common, **{key: value for key, value in metrics.items() if key in RUN_FIELDS}},
                    run_id=identifier+"_"+method, method=method, backbone="NONE", K=None, MSE=metrics["MSE"], MAE=metrics["MAE"],
                    scaler_sha256=result.get("scaler_sha256"), origins_sha256=result.get("origins_sha256"),
                    channel_sha256=None, selection="A_stop alpha grid .01/.1/1/10/100", dtype="float64", backend="numpy_blas", point_statistic="deterministic_ridge")
                runs.append(run)
            stats_path = report / "B02" / job["dataset_cohort"] / "ERROR_BY_HORIZON_CHANNEL.csv"
            if stats_path.is_file():
                stat_rel, stat_sha = _source(stats_path, ctx, bindings)
                for entry in rows(stats_path):
                    horizon.append(dict(run_id=identifier+"_"+entry["method"], cohort=job["dataset_cohort"], method=entry["method"], seed=None,
                        split="V_select", metric_scale="canonical_A60", horizon=int(entry["horizon"]), channel_id=entry["channel_id"],
                        SSE=float(entry["SSE"]), SAE=float(entry["SAE"]), count=int(entry["count"]),
                        origins_sha256=result.get("origins_sha256"), source_file=stat_rel, source_sha256=stat_sha))
        elif question in ("B03", "C02", "D02", "D03"):
            for index, row in enumerate(result.get("rows", [])):
                if row.get("evaluation", row.get("split", "V_select")) not in ("V_select", "repeated_development_V43"):
                    continue
                run = dict({**common, **{key: value for key, value in row.items() if key in RUN_FIELDS}, **contract_metadata(row)},
                    run_id=identifier+"_"+str(index), method=row.get("method", row.get("family", "UNKNOWN")),
                    MSE=row.get("MSE"), MAE=row.get("MAE"), epochs=row.get("selected_epoch", result.get("original_selected_epoch")),
                    updates=row.get("refit_updates"), exposures=row.get("refit_exposures"),
                    selection=result.get("selection", "NO_TARGET_SELECTION" if question == "D03" else "A_stop CP0/.25/1/3"),
                    mask_sha256=contract_metadata(row).get("mask_sha256", row.get("observed_sha256")),
                    channel_sha256=contract_metadata(row).get("channel_sha256", row.get("channel_order_sha256")))
                if question == "B03":
                    run["deployed_trainable_params"] = row.get("total_target_trained_params")
                    run["fit_stage_params"] = row.get("decoder_refit_trainable_params", row.get("trainable_params"))
                if row.get("fit") == "NO_TARGET_PARAMETER_LEARNING" or run["method"] in ("Moirai_Raw", "Chronos2_nativeRaw"):
                    run.update(selection=row.get("selection", "NO_TARGET_SELECTION"), refit="NO_TARGET_PARAMETER_REFIT",
                               epochs=None, updates=0, exposures=0, trainable_params=0,
                               supervision_scope="A60 scaler statistics only; no target parameter learning or selection")
                run["_prediction_sha256"] = row.get("prediction_sha256")
                run["_private_prediction_reference"] = row.get("private_prediction_reference")
                runs.append(run)
                _stats_from_result(row, run, horizon, channel)
            initializations += [dict(job_id=identifier, source_file=relative, source_sha256=digest, **entry) for entry in result.get("diagnostics", [])]
            if question == "C02":
                for entry in result.get("exact_endpoint_curves", []):
                    init_curves.append(dict(job_id=identifier, source_file=relative, source_sha256=digest,
                        curve_type="A_fit_exact_endpoint", **entry))
                    endpoint = dict(common, run_id=identifier+"_train_endpoint_"+entry["initialization"]+"_"+str(entry["epoch"]),
                        method="B2_"+entry["initialization"], epoch=entry["epoch"], updates=entry["updates"], exposures=entry["exposures"],
                        MSE=entry["A_fit_MSE"], MAE=entry["A_fit_MAE"], split="A_fit", metric_scale="search_A_fit",
                        fit="A_fit_search", refit="NOT_APPLICABLE_SEARCH_ENDPOINT", selection="NOT_SELECTED_COMMON_ENDPOINT",
                        dtype="float32", train_eval_origin_stride=entry["train_eval_origin_stride"],
                        FP64_CPU_GPU_aggregate_difference=entry.get("FP64_CPU_GPU_aggregate_difference"))
                    endpoints.append(endpoint)
                    _stats_from_result(entry, endpoint, horizon, channel)
                for entry in result.get("training_curves", []):
                    init_curves.append(dict(job_id=identifier, seed=job.get("seed"), source_file=relative,
                        source_sha256=digest, curve_type="stochastic_segment_mean", **entry))
                for kind in ("PCA", "QR"):
                    curve_path = report/"runs"/("init_"+kind+"_s"+str(job["seed"]))/"search/CURVE.csv"
                    if curve_path.is_file():
                        curve_rel, curve_sha = _source(curve_path, ctx, bindings)
                        for entry in rows(curve_path):
                            init_curves.append(dict(job_id=identifier, seed=job.get("seed"), initialization=kind,
                                curve_type="A_stop_exact_endpoint", updates=number(entry["updates"]), exposures=number(entry["exposures"]),
                                MSE=number(entry["astop_mse"]), source_file=curve_rel, source_sha256=curve_sha))
        elif question in ("C01", "D03") and result.get("MSE") is not None:
            run = dict({**common, **{key: value for key, value in result.items() if key in RUN_FIELDS}, **contract_metadata(result)},
                       run_id=identifier, MSE=result["MSE"], MAE=result.get("MAE"), epochs=result.get("selected_epoch"),
                       exposures=result.get("exposure", result.get("exposures")), trainable_params=result.get("params", result.get("trainable_params")),
                       selection="A_stop CP0/.25/1/3" if question == "C01" else common["selection"],
                       checkpoint_sha256=result.get("artifact_sha256"))
            run["_prediction_sha256"] = result.get("private_prediction_archive_sha256")
            runs.append(run)
            _stats_from_result(result, run, horizon, channel)
        elif question in ("E01", "E02", "E03", "E04"):
            for method, replay in result.get("full_v_replay", {}).items():
                if replay.get("MSE") is None or replay.get("MAE") is None:
                    continue
                run = {**common, **{key: value for key, value in replay.items() if key in RUN_FIELDS}, **contract_metadata(replay)}
                run.update(run_id=replay["run_id"], method=canonical_method(method),
                    origin="EXISTING_SELECTED_ROUTE_DEVICE_PARITY_REPLAY", fit="EXISTING_SELECTED_ROUTE_NO_NEW_TRAINING",
                    refit="NO_NEW_REFIT", selection="EXISTING_SELECTED_ARTIFACT", MSE=replay["MSE"], MAE=replay["MAE"],
                    independent_private_score_sha256=replay.get("prediction_sha256"),
                    checkpoint_sha256=replay.get("source_checkpoint_sha256"),
                    prediction_binding_scope="Full development V replay of the already-selected route on the measured device; original saved reference retained")
                original = next((row for row in runs if row["run_id"] == replay.get("original_accuracy_run_id")), None)
                if original is None:
                    original = next((row for row in runs if row["cohort"] == run["cohort"] and row["method"] == run["method"]
                                     and row.get("seed") in (run.get("seed"), None)), None)
                if original is not None:
                    for field in ("backbone", "backbone_revision", "routing", "input_info_scope", "supervision_scope"):
                        if field in original:
                            run[field] = original[field]
                if run["method"] == "Ridge":
                    run.update(backbone="NONE", backbone_revision=None, routing="no_foundation")
                if run["method"] in ("Raw", "Chronos2_nativeRaw"):
                    run["supervision_scope"] = "A60 scaler statistics only; no target parameter learning or selection"
                if run["method"] == "Chronos2_nativeRaw":
                    run["routing"] = "native_multivariate_per_origin"
                reference = replay.get("private_prediction_ref")
                if reference and "://" in reference:
                    scheme, name = reference.split("://", 1)
                    if scheme in ctx:
                        root = Path(ctx[scheme]).resolve(); path = (root/name).resolve()
                        if not path.is_relative_to(root):
                            raise ValueError("Device replay prediction escapes bound root")
                        run["_archive_path"] = path
                runs.append(run)
                _stats_from_result(replay, run, horizon, channel)
            cost_rows = result.get("rows", result.get("measurements", [result]))
            if isinstance(cost_rows, dict):
                cost_rows = list(cost_rows.values())
            for entry in cost_rows:
                batch = number(entry.get("B", entry.get("batch", entry.get("batch_size"))))
                median = number(entry.get("median_seconds", entry.get("whole_median_seconds")))
                source_method = entry.get("method", entry.get("family", job["family"]))
                method = canonical_method(source_method)
                trained_seed = entry.get("seed", job.get("seed"))
                if method in ("B0", "B2", "CAP", "Moirai_B0", "Moirai_B2") and trained_seed is None:
                    trained_seed = 92601
                deterministic = method in ("Raw", "Ridge", "Moirai_Raw", "Chronos2_nativeRaw")
                cohort = entry.get("cohort", job["dataset_cohort"])
                source_parity = entry.get("parity_state", entry.get("parity", entry.get("forward_parity")))
                parity = source_parity.get("status") if isinstance(source_parity, dict) else source_parity
                costs.append(dict(job_id=identifier, cohort=cohort, source_method=source_method,
                    method=method, seed=None if deterministic else trained_seed,
                    supervision_seed=None if deterministic else trained_seed, evaluation_reference_seed=92601 if deterministic else trained_seed,
                    seed_intent="No random target-supervision seed; fixed saved evaluation reference" if deterministic else "Selected supervised adapter seed",
                    C=entry.get("C", 21 if cohort == "jena" else 32), L=entry.get("L", 512), H=entry.get("H", 48 if cohort == "jena" else 12),
                    B=batch, median_ms_per_batch=median*1000 if median is not None else entry.get("ms_per_batch", entry.get("median_ms_batch")),
                    median_ms_per_origin=median*1000/batch if median is not None and batch else entry.get("ms_per_origin", entry.get("median_ms_origin")),
                    IQR_ms=number(entry.get("IQR_seconds"))*1000 if number(entry.get("IQR_seconds")) is not None else entry.get("IQR_ms_batch"),
                    scope=entry.get("scope", result.get("scope")), device=entry.get("device", job["device"]), threads=entry.get("threads", 2),
                    dtype=entry.get("dtype", "source_bound"), hardware=entry.get("hardware", result.get("hardware")),
                    warmups=entry.get("warmups", 10), repetitions=entry.get("repetitions", entry.get("timed_trials", 20)),
                    accuracy_run_id=entry.get("accuracy_run_id"),
                    fit_seconds=entry.get("fit_seconds"), cold_load_seconds=entry.get("cold_load_seconds"),
                    cold_shared_load_seconds=entry.get("cold_shared_load_seconds"), cold_load_scope=entry.get("cold_load_scope", result.get("cold_load_scope")),
                    trainable_params=entry.get("trainable_params"),
                    total_resident_model_params=entry.get("total_resident_parameters", entry.get("backbone_resident_parameters")),
                    core_dtype=entry.get("core_dtype"), resident_parameter_scope=entry.get("resident_parameter_scope"),
                    shared_process_resident_parameters=entry.get("shared_process_resident_parameters"),
                    shared_process_parameter_scope=entry.get("shared_process_parameter_scope"),
                    logical_F_series_rows=entry.get("logical_F_rows"),
                    process_RSS_peak_bytes=entry.get("process_RSS_sampled_peak_bytes", entry.get("process_RSS_peak_bytes", entry.get("process_RSS_peak_sampled_bytes", entry.get("process_RSS_bytes")))),
                    process_RSS_snapshot_bytes=entry.get("process_RSS_bytes"), RSS_sample_count=entry.get("RSS_sample_count"),
                    RSS_sampling_interval_seconds=entry.get("RSS_sampling_interval_seconds"), RSS_peak_is_sampled_not_continuous=entry.get("RSS_peak_is_sampled_not_continuous"),
                    RSS_scope=entry.get("RSS_scope", "Source-bound sampling/end scope; no isolated activation-memory claim"),
                    peak_allocated_bytes=entry.get("peak_allocated_bytes"), peak_reserved_bytes=entry.get("peak_reserved_bytes"),
                    physical_input_sha256=entry.get("physical_input_sha256"), observation_mask_sha256=entry.get("observation_mask_sha256", entry.get("mask_sha256")),
                    parity_state=parity, source_forward_parity=source_parity, status=entry.get("status", result.get("status")),
                    individual_timed_seconds=entry.get("individual_timed_seconds"),
                    exclusive=entry.get("exclusive_queue_owned_by_caller", result.get("exclusive_queue_owned_by_caller",
                        result.get("measurements_exclusive") is True or (result.get("cpu_heavy_workers_during_cost") == 1 and result.get("other_campaign_workers_observed") == []))),
                    accuracy_MSE=entry.get("accuracy_MSE", entry.get("MSE")), accuracy_MAE=entry.get("accuracy_MAE", entry.get("MAE")), accuracy_source=entry.get("accuracy_source"),
                    accuracy_archive_sha256=entry.get("accuracy_archive_sha256", entry.get("prediction_archive_sha256", result.get("bindings", {}).get(source_method, {}).get("prediction_sha256"))),
                    measured_batch_parity=entry.get("measured_batch_parity"),
                    physical_F_series_rows=entry.get("physical_F_rows"), F_calls=entry.get("F_calls"),
                    IQR_ms_per_origin=entry.get("IQR_ms_origin"),
                    accuracy_reference_run_id=entry.get("accuracy_reference_run_id"),
                    accuracy_reference_archive_sha256=entry.get("accuracy_reference_archive_sha256"),
                    accuracy_reference_MSE=entry.get("accuracy_reference_MSE"), accuracy_reference_MAE=entry.get("accuracy_reference_MAE"),
                    source_file=relative, source_sha256=digest))
    # Complete channel marginals for every public per-horizon/channel statistic.
    grouped = defaultdict(lambda: dict(SSE=0., SAE=0., count=0))
    for row in horizon:
        if row["channel_id"] == "ALL_CHANNELS_EQUAL_COUNTS":
            continue
        key = (row["run_id"], row["channel_id"])
        grouped[key]["SSE"] += float(row["SSE"])
        grouped[key]["SAE"] += float(row["SAE"])
        grouped[key]["count"] += int(row["count"])
    existing = {(row["run_id"], row["channel_id"]) for row in channel}
    for (identifier, c), statistics in grouped.items():
        if (identifier, c) not in existing:
            sample = next(row for row in horizon if row["run_id"] == identifier)
            channel.append({**{key: sample.get(key) for key in ("run_id", "cohort", "method", "seed", "split", "metric_scale", "origins_sha256", "source_file", "source_sha256")}, "channel_id": c, **statistics})
    collected = dict(plan=plan, jobs=job_rows, runs=runs, endpoints=endpoints, probes=probes,
                initializations=initializations, initialization_curves=init_curves,
                costs=costs, horizon=horizon, channel=channel, bindings=bindings)
    collected["legacy_stats_replay"] = legacy_statistics(ctx, collected)
    complete_new_metadata(ctx, results, collected)
    complete_probe_metadata(ctx, results, collected)
    for index, run in enumerate(collected["runs"]):
        collected["runs"][index] = canonical_cohort(run, repo)
    for index, endpoint in enumerate(collected["endpoints"]):
        collected["endpoints"][index] = canonical_cohort(endpoint, repo)
    for index, cost in enumerate(collected["costs"]):
        collected["costs"][index] = canonical_cohort(cost, repo)
    complete_cost_metadata(ctx, results, collected)
    if "asset" in ctx and "cache" in ctx and any(job["question_id"] == "E01" for job, _, _ in results.values()):
        helper = report_helper(ctx, "pems32_cost_metadata", collected["bindings"])
        collected["pems32_cost_metadata"] = helper.complete_cost_metadata(ctx, results, collected)
    complete_device_replay_metadata(ctx, collected)
    complete_fit_metadata(ctx, results, collected)
    if "asset" in ctx and "cache" in ctx and any(job["question_id"] == "B03" for job in plan["jobs"]):
        helper = report_helper(ctx, "trajectory_diagnostics", collected["bindings"])
        join_trajectory_evidence(collected, helper.build_trajectory_diagnostics(ctx, collected))
    if "cache" in ctx and (report/"M5_LOCAL_ASSET_DISCOVERY.json").is_file():
        helper = report_helper(ctx, "m5_local_assets", collected["bindings"])
        collected["m5_local_assets"] = helper.verify_local_m5_assets(ctx)
        for name in ("M5_LOCAL_ASSET_DISCOVERY.json", "M5_LOCAL_ASSETS_KO.md"):
            _source(report/name, ctx, collected["bindings"])
    cohort_by_run = {run["run_id"]: run["cohort"] for run in collected["runs"]+collected["endpoints"]}
    for row in collected["channel"]+collected["horizon"]:
        row["cohort"] = cohort_by_run[row["run_id"]]
    channel_covered = {row["run_id"] for row in collected["channel"]}
    horizon_covered = {row["run_id"] for row in collected["horizon"]}
    for run in collected["runs"]+collected["endpoints"]:
        run.setdefault("G6_channel_coverage", "PUBLIC_SUFFICIENT_STATS" if run["run_id"] in channel_covered else "PENDING_MISSING_SUFFICIENT_STATS")
        run.setdefault("G6_horizon_coverage", "PUBLIC_SUFFICIENT_STATS" if run["run_id"] in horizon_covered else "PENDING_MISSING_SUFFICIENT_STATS")
    by_run = {run["run_id"]: run for run in collected["runs"]}
    for probe in collected["probes"]:
        probe.update({field: by_run[probe["run_id"]].get(field) for field in RUN_FIELDS})
    return collected


def pair_contract(reference, method):
    reasons = []
    for field in ("C", "L", "H"):
        if number(reference.get(field)) is None or number(reference.get(field)) != number(method.get(field)):
            reasons.append("MISSING_OR_DIFFERENT_"+field)
    for field in ("cohort", "split", "metric_scale", "point_statistic", "device", "backend", "dtype"):
        a, b = reference.get(field), method.get(field)
        if a in (None, "", "NA", "source_bound", "source_bound_point_forecast") or a != b:
            reasons.append("MISSING_OR_DIFFERENT_"+field)
    for field in ("scaler_sha256", "target_sha256", "mask_sha256", "origins_sha256", "channel_sha256"):
        a, b = reference.get(field), method.get(field)
        if not isinstance(a, str) or not re.fullmatch("[a-f0-9]{64}", a) or a != b:
            reasons.append("MISSING_OR_DIFFERENT_"+field)
    return reasons


def comparisons(data, contextual=False):
    output = []
    groups = defaultdict(list)
    for row in data["runs"]:
        if number(row.get("MSE")) is not None:
            block = next((prefix for prefix in ("B01", "B02", "B03", "C02", "D02") if row["run_id"].startswith(prefix)), "SELECTED_BASELINES")
            groups[(row["cohort"], row.get("metric_scale"), block)].append(row)
    registered = {
        "SELECTED_BASELINES": (("B0", "B2"), ("CAP", "B2"), ("Raw", "B2"), ("Ridge", "B2"),
                               ("Raw", "Chronos2_nativeRaw"), ("B2", "Chronos2_nativeRaw"),
                               ("Ridge", "Chronos2_nativeRaw"), ("LoRA", "Chronos2_nativeRaw")),
        "B01": (("B0_FULL_HISTORY_PROBE", "B2_FULL_HISTORY_PROBE"), ("B2_CAP_POINTWISE_FULL_HISTORY_PROBE", "B2_FULL_HISTORY_PROBE")),
        "B02": (("OWN_CHANNEL_32BLOCK_RIDGE", "ALL_CHANNEL_32BLOCK_RIDGE"),),
        "B03": (("B2_ORIGINAL_D", "B2_DECODER_SHARED"), ("B2_ORIGINAL_D", "B2_DECODER_HORIZON"), ("B2_DECODER_SHARED", "B2_DECODER_HORIZON")),
        "C02": (("B2_PCA", "B2_QR"),),
        "D02": (("Moirai_B0", "Moirai_B2"), ("Moirai_Raw", "Moirai_B2")),
    }
    for key, group in groups.items():
        for reference_name, method_name in registered.get(key[2], ()):
            for method in [r for r in group if r["method"] == method_name]:
                deterministic = reference_name in ("Raw", "Ridge", "Moirai_Raw")
                candidates = [r for r in group if r["method"] == reference_name and r.get("seed") == method.get("seed")]
                pairing_rule = "Same source seed; device/backend/dtype-matching reference preferred when duplicate routes exist"
                if not candidates and method_name == "Chronos2_nativeRaw" and method.get("seed") is None:
                    candidates = [r for r in group if r["method"] == reference_name and r.get("seed") == 92601]
                    pairing_rule = "Native route has no target-training seed; compare primary already-selected supervised seed92601; reference seed and device differences retained"
                if not candidates and deterministic:
                    candidates = [r for r in group if r["method"] == reference_name and r.get("seed") in (None, 92601)]
                    pairing_rule = "Deterministic baseline fixed saved evaluation reference; original source seed retained, target-supervision seed not applicable"
                if not candidates:
                    continue
                reference = max(candidates, key=lambda r: sum(r.get(field) == method.get(field) for field in ("device", "backend", "dtype", "origin")))
                reasons = pair_contract(reference, method)
                if bool(reasons) != contextual:
                    continue
                output.append(dict(comparison_scope="SELECTED_REFIT_OR_FIXED_DIAGNOSTIC", cohort=key[0], seed=method.get("seed"), reference_run_id=reference["run_id"], method_run_id=method["run_id"],
                reference=reference["method"], method=method["method"], reference_seed=reference.get("seed"), method_seed=method.get("seed"),
                reference_supervision_seed=None if deterministic else reference.get("seed"), pairing_rule=pairing_rule,
                metric_scale=key[1], comparison_family=key[2], reference_MSE=reference["MSE"], method_MSE=method["MSE"],
                reference_MAE=reference["MAE"], method_MAE=method["MAE"],
                reference_trainable_params=reference.get("trainable_params"), method_trainable_params=method.get("trainable_params"),
                reference_fit_stage_params=reference.get("fit_stage_params"), method_fit_stage_params=method.get("fit_stage_params"),
                reference_deployed_trainable_params=reference.get("deployed_trainable_params"), method_deployed_trainable_params=method.get("deployed_trainable_params"),
                parameter_count_note="B03 trainable_params is the D coefficient block, not whole deployed model; fit_stage_params counts coefficients fitted in this diagnostic (original D=0); retained trained PE is included in deployed totals 35717/77171" if key[2] == "B03" else None,
                delta_MSE=method["MSE"]-reference["MSE"], delta_MAE=method["MAE"]-reference["MAE"],
                delta_MSE_pct=100*(method["MSE"]-reference["MSE"])/reference["MSE"],
                delta_MAE_pct=100*(method["MAE"]-reference["MAE"])/reference["MAE"],
                scaler_sha256=reference.get("scaler_sha256"), origins_sha256=reference.get("origins_sha256"),
                common_contract_hash_status="UNVERIFIED_CONTEXTUAL" if reasons else "ALL_FIVE_EXACT",
                match_status="UNVERIFIED_CONTEXTUAL" if reasons else "EXACT_MATCHED",
                unresolved_contract_fields=reasons,
                changed_factors={"B01": "Fixed P/E representation; same fixed full-history reader lambda1", "B02": "Own versus all-channel block features and coefficient count", "B03": "Original jointly trained D versus refit shared/horizon-specific decoder; effective penalties matched between refit arms", "C02": "PCA versus Gaussian QR E initialization", "D02": "Adapter front, capacity, target supervision and selected learning amount under same Moirai F"}.get(key[2], "Architecture/capacity/target supervision/selected learning amount per source recipe; nativeRaw also changes backbone and grouping"),
                remaining_confound="Unverified contract/backend/point definition; numerical contextual difference only" if reasons else "Selected-refit learning amount/capacity may differ; repeated development V; not all comparisons isolate one factor"))
    return output


def same_update_comparisons(data):
    groups = defaultdict(list)
    for row in data.get("endpoints", []):
        if canonical_method(row["method"]) not in ("B0", "B2", "CAP") or number(row.get("MSE")) is None or number(row.get("MAE")) is None:
            continue
        key = tuple(row.get(field) for field in ("cohort", "seed", "split", "metric_scale"))+tuple(number(row.get(field)) for field in ("epoch", "updates", "exposures"))
        groups[key].append(row)
    output = []
    for key, group in groups.items():
        for reference_name, method_name in (("B0", "B2"), ("CAP", "B2"), ("CAP", "B0")):
            for reference in [row for row in group if canonical_method(row["method"]) == reference_name]:
                for method in [row for row in group if canonical_method(row["method"]) == method_name]:
                    reasons = pair_contract(reference, method)
                    for field, value in zip(("epoch", "updates", "exposures"), key[4:]):
                        if value is None:
                            reasons.append("MISSING_COMMON_ENDPOINT_"+field)
                    if reference.get("fit") != "A_fit_search" or method.get("fit") != "A_fit_search":
                        reasons.append("NOT_COMMON_A_FIT_SEARCH_WEIGHTS")
                    for label, row in (("reference", reference), ("method", method)):
                        if not row.get("source_file") or not re.fullmatch("[a-f0-9]{64}", str(row.get("source_sha256", ""))):
                            reasons.append("MISSING_"+label.upper()+"_SOURCE_BINDING")
                    rmse, rmae, mse, mae = (float(reference["MSE"]), float(reference["MAE"]), float(method["MSE"]), float(method["MAE"]))
                    output.append(dict(comparison_scope="COMMON_SEARCH_ENDPOINT", comparison_family="SAME_UPDATE", cohort=key[0], seed=key[1],
                        split=key[2], metric_scale=key[3], epoch=key[4], updates=key[5], exposures=key[6],
                        C=method.get("C"), L=method.get("L"), H=method.get("H"), reference=reference_name, method=method_name,
                        reference_run_id=reference["run_id"], method_run_id=method["run_id"],
                        reference_MSE=rmse, method_MSE=mse, reference_MAE=rmae, method_MAE=mae,
                        delta_MSE=mse-rmse, delta_MAE=mae-rmae,
                        delta_MSE_pct=100*(mse-rmse)/rmse if rmse else None, delta_MAE_pct=100*(mae-rmae)/rmae if rmae else None,
                        match_status="UNVERIFIED_CONTEXTUAL" if reasons else "EXACT_MATCHED",
                        common_contract_hash_status="UNVERIFIED_CONTEXTUAL" if reasons else "ALL_FIVE_EXACT",
                        unresolved_contract_fields=reasons, scaler_sha256=method.get("scaler_sha256"), origins_sha256=method.get("origins_sha256"),
                        reference_source_file=reference["source_file"], reference_source_sha256=reference["source_sha256"],
                        method_source_file=method["source_file"], method_source_sha256=method["source_sha256"],
                        fit="A_fit_search", selection="NOT_SELECTED_COMMON_ENDPOINT", refit="NOT_APPLICABLE_SEARCH_ENDPOINT",
                        changed_factors="Architecture at exactly the same recorded search epoch/updates/exposures; B2/CAP capacity is close but not identical",
                        remaining_confound="Repeated development evaluation and two seeds; common search weights are distinct from selected fresh-A60 refits"))
    return output


def common_endpoint_interpretation(data, compared):
    selected = [row for row in comparisons(data)+comparisons(data, contextual=True) if row["reference"] == "CAP" and row["method"] == "B2"]
    common = [row for row in compared if row["split"] == "V_select" and row["reference"] == "CAP" and row["method"] == "B2"]
    if not common:
        return "[미검증] B2−CAP의 같은 학습량 개발 V 대조가 아직 없다."
    lines = ["[확인] 공통 search endpoint의 개발 V 대조(B2−CAP)는 SAME_UPDATE_COMPARISONS.csv에 별도 기록한다. A_fit search 가중치의 평가이며 A_fit/A_stop 평가 행, A_stop 선택 및 selected fresh-A60 재적합 결과와 구분한다. 음수 Δ는 오차 개선이다."]
    for cohort, seed in sorted({(row["cohort"], row["seed"]) for row in common}, key=str):
        group = sorted([row for row in common if (row["cohort"], row["seed"]) == (cohort, seed)], key=lambda row: row["epoch"])
        for row in group:
            lines.append(cohort+" seed="+str(seed)+" search epoch="+str(row["epoch"])+" updates="+str(row["updates"])+" exposures="+str(row["exposures"])+
                ": ΔMSE="+repr(row["delta_MSE_pct"])+"%, ΔMAE="+repr(row["delta_MAE_pct"])+"%; "+row["match_status"])
        learned = [row for row in group if row["epoch"] > 0]
        if learned and all(row["match_status"] == "EXACT_MATCHED" for row in learned):
            lines.append(cohort+" seed="+str(seed)+(": 완료한 비영 endpoint 모두 MSE/MAE가 개선됐다." if all(row["delta_MSE"] < 0 and row["delta_MAE"] < 0 for row in learned) else ": 공통 학습량에서 두 오차의 일관된 우위가 아니다. endpoint와 MSE/MAE에 따라 방향이 달라진다."))
    if selected:
        lines.append("선택 후 fresh-A60에서는 "+("완료한 B2−CAP 대조 모두 MSE/MAE가 낮다." if all(row["delta_MSE"] < 0 and row["delta_MAE"] < 0 for row in selected) else "B2−CAP의 두 오차가 모든 완료 대조에서 동시에 개선되지는 않는다.")+" 이 selected-refit 관측을 공통 학습량의 일관된 시간·채널 혼합 효과로 바꾸어 읽지 않는다.")
    lines.append("B2−B0와 B0−CAP의 같은 endpoint 대조 및 CP0의 실제 Δ도 같은 전용 CSV에 보존했다.")
    return "\n\n".join(lines)


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": NAME,
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"])})
    return plt


def _save(fig, name, report, metadata):
    import matplotlib.pyplot as plt
    folder = Path(report) / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(folder/(name+"."+extension), dpi=300, bbox_inches="tight",
                    metadata={"Date": None} if extension == "svg" else {"Software": NAME})
    metadata[name] = dict(source_code="research/"+NAME+"/reporting.py", plot_type=name,
        png_sha256=sha(folder/(name+".png")), svg_sha256=sha(folder/(name+".svg")))
    plt.close(fig)


def error_profiles(data, dimension):
    statistics = defaultdict(lambda: defaultdict(list))
    for row in data["horizon" if dimension == "horizon" else "channel"]:
        position = int(row["horizon"]) if dimension == "horizon" else int(row["channel_id"])
        if int(row["count"]) > 0:
            statistics[row["run_id"]][position].append((float(row["SSE"])/int(row["count"]), float(row["SAE"])/int(row["count"])))
    output = []
    for pair in comparisons(data)+comparisons(data, contextual=True):
        if pair["method"] not in ("B2", "CAP", "ALL_CHANNEL_32BLOCK_RIDGE", "B2_DECODER_HORIZON", "B2_ORIGINAL_D", "B2_QR", "Moirai_B2", "B2_FULL_HISTORY_PROBE"):
            continue
        left, right = statistics.get(pair["reference_run_id"]), statistics.get(pair["method_run_id"])
        if not left or not right or set(left) != set(right):
            continue
        positions = sorted(left)
        output.append(dict(cohort=pair["cohort"], reference=pair["reference"], method=pair["method"], seed=pair["seed"],
            match_status=pair["match_status"], unresolved_contract_fields=pair["unresolved_contract_fields"], positions=positions,
            delta_MSE=[math.fsum(v[0] for v in right[p])/len(right[p])-math.fsum(v[0] for v in left[p])/len(left[p]) for p in positions],
            delta_MAE=[math.fsum(v[1] for v in right[p])/len(right[p])-math.fsum(v[1] for v in left[p])/len(left[p]) for p in positions]))
    return output


def initialization_curve_label(kind, seed, end):
    return kind+" "+str(int(seed))+(" end="+str(int(end)) if number(end) is not None else "")


def error_profile_figure(plt, profiles, dimension):
    cohorts = list(dict.fromkeys(profile["cohort"] for profile in profiles))
    heights = []
    for cohort in cohorts or [None]:
        count = sum(profile["cohort"] == cohort for profile in profiles)
        heights.extend((4.2, max(1., .25*math.ceil(count/2))))
    fig = plt.figure(figsize=(14, sum(heights)), constrained_layout=True)
    grid = fig.add_gridspec(len(heights), 2, height_ratios=heights)
    for index, cohort in enumerate(cohorts or [None]):
        pair = [fig.add_subplot(grid[2*index, column]) for column in range(2)]
        legend_axis = fig.add_subplot(grid[2*index+1, :]); legend_axis.axis("off")
        cohort_profiles = [p for p in profiles if p["cohort"] == cohort]
        positions = sorted({position for profile in cohort_profiles for position in profile["positions"]})
        channel_index = {position: index for index, position in enumerate(positions)}
        width = .8/max(1, len(cohort_profiles))
        for profile_index, profile in enumerate(cohort_profiles):
            label = profile["method"]+" - "+profile["reference"]+" s"+str(profile["seed"])
            exact = profile["match_status"] == "EXACT_MATCHED"
            for axis, metric in zip(pair, ("MSE", "MAE")):
                if dimension == "channel":
                    ordered = sorted(zip(profile["positions"], profile["delta_"+metric]))
                    offset = (profile_index-(len(cohort_profiles)-1)/2)*width
                    axis.bar([channel_index[position]+offset for position, _ in ordered], [value for _, value in ordered],
                             width=width, hatch=None if exact else "///", edgecolor="none" if exact else "black",
                             linewidth=0 if exact else .35, label=label+("" if exact else " [context]"))
                else:
                    axis.plot(profile["positions"], profile["delta_"+metric], lw=1.1,
                              linestyle="-" if exact else ":", label=label+("" if exact else " [context]"))
        for axis, metric in zip(pair, ("MSE", "MAE")):
            axis.axhline(0, color="black", lw=.8); axis.set_xlabel("Forecast horizon step" if dimension == "horizon" else "Original stored channel index")
            axis.set_ylabel("Method - reference macro "+metric); axis.set_title(str(cohort)+" | repeated development V")
            axis.grid(alpha=.2)
            if dimension == "channel":
                axis.set_xticks(range(len(positions)), [str(position) for position in positions], rotation=90,
                                fontsize=6 if len(positions) <= 32 else 4)
        handles, labels = pair[0].get_legend_handles_labels()
        if handles:
            legend_axis.legend(handles, labels, loc="center", ncol=2, fontsize=7, frameon=False,
                               mode="expand", borderaxespad=0)
        else:
            pair[0].text(.5, .5, "No complete paired sufficient statistics", transform=pair[0].transAxes, ha="center")
    style = "grouped bars, fixed original channel order; plain: verified; hatched: contextual" if dimension == "channel" else "solid: contract verified; dotted: contextual"
    fig.suptitle("All stored "+dimension+" positions | "+style+"; negative = improvement")
    return fig


def figures(ctx, data, private_qualitative=True):
    import numpy as np
    plt = _plot_setup()
    report, repo = Path(ctx["report"]), Path(ctx["repo"])
    metadata = {}
    points, cost_groups = [], defaultdict(list)
    for cost in data["costs"]:
        latency = number(cost.get("median_ms_per_origin"))
        if (latency is None or latency <= 0 or cost.get("exclusive") is not True or cost.get("parity_state") != "PASS"
                or any(not re.fullmatch("[a-f0-9]{64}", str(cost.get(key, ""))) for key in ("physical_input_sha256", "observation_mask_sha256"))):
            continue
        method = canonical_method(cost["method"])
        accuracy = cost_accuracy_match(cost, data["runs"])
        if accuracy is None:
            continue
        point = dict(method=method, latency_ms_per_origin=latency, MSE=accuracy["MSE"], MAE=accuracy["MAE"],
                     cost_source=cost["source_sha256"], accuracy_run_id=accuracy["run_id"], accuracy_source=accuracy["source_sha256"],
                     accuracy_device=accuracy.get("device"), accuracy_scope=cost.get("accuracy_source", "Saved full-V accuracy with measured-route parity; not fresh full-V timing inference"))
        points.append(point)
        key = (cost["device"], cost["cohort"], cost.get("B"), cost.get("dtype"), cost.get("physical_input_sha256"),
               cost.get("observation_mask_sha256"), cost.get("threads"), cost.get("scope"))
        cost_groups[key].append(point)
    fig, axes = plt.subplots(max(1, len(cost_groups)), 2, figsize=(12, 4*max(1, len(cost_groups))), squeeze=False)
    for pair, (key, group) in zip(axes, cost_groups.items()):
        for axis, metric in zip(pair, ("MSE", "MAE")):
            for point in group:
                axis.scatter(point["latency_ms_per_origin"], point[metric], s=45)
                axis.annotate(point["method"], (point["latency_ms_per_origin"], point[metric]), fontsize=8, xytext=(3, 3), textcoords="offset points")
            axis.set_title(str(key[0])+" | "+str(key[1])+" B"+str(key[2])+" | same input, dtype/scope")
    for pair in axes:
        for axis, metric in zip(pair, ("MSE", "MAE")):
            axis.set_xscale("log"); axis.set_xlabel("Exclusive matched E2E warm ms/origin (log)")
            axis.set_ylabel("Normalized macro "+metric); axis.grid(alpha=.2)
            if not points:
                axis.text(.5, .5, "No valid exclusive matched latency yet\nUnmeasured values remain NA", transform=axis.transAxes, ha="center")
    fig.suptitle("Saved full-V accuracy and first-B E2E parity/cost | separate device/cohort/batch/dtype/input facets", y=1.001)
    fig.tight_layout()
    _save(fig, "accuracy_cost", report, metadata)
    metadata["accuracy_cost"]["plotted_points"] = points
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    for axis, split in zip(axes.flat[:3], ("A_fit", "A_stop", "V_select")):
        groups = defaultdict(list)
        for row in data["endpoints"]:
            if row["split"] == split:
                groups[(row["method"], row["seed"])].append(row)
        for (method, seed), group in groups.items():
            group.sort(key=lambda v: number(v.get("updates")) or 0)
            axis.plot([r["updates"] for r in group], [r["MSE"] for r in group], marker="o", markersize=3,
                      linestyle="-" if seed == 92601 else "--", label=method+" "+str(int(seed)))
        axis.set_title(split+" | "+("A60 scale" if split == "V_select" else "A50 search scale"))
        axis.set_xlabel("Search optimizer updates"); axis.set_ylabel("Normalized macro MSE"); axis.grid(alpha=.2)
        if groups:
            axis.legend(fontsize=7)
    selected = [r for r in data["runs"] if r["cohort"] == "jena" and r["method"] in ("B0", "B2", "CAP", "B2_PCA", "B2_QR")]
    for row in selected:
        if number(row.get("updates")) is not None:
            axes.flat[3].scatter(row["updates"], row["MSE"], label=row["method"]+" "+str(int(row["seed"])), marker="*")
    axes.flat[3].set_title("Selected fresh A60 refits (separate from search)")
    axes.flat[3].set_xlabel("Refit updates"); axes.flat[3].set_ylabel("V normalized macro MSE")
    if selected:
        axes.flat[3].legend(fontsize=7)
    init_groups = defaultdict(list)
    for entry in data.get("initialization_curves", []):
        init_groups[(entry["initialization"], entry["seed"], entry["curve_type"], entry.get("fit_end"))].append(entry)
    for (kind, seed, curve_type, end), group in init_groups.items():
        group.sort(key=lambda row: number(row["updates"]) or 0)
        if curve_type == "A_fit_exact_endpoint":
            for axis, key in ((axes.flat[4], "A_fit_MSE"), (axes.flat[5], "A_fit_MAE")):
                axis.plot([number(row["updates"]) for row in group], [number(row[key]) for row in group],
                          marker="o", ms=3, label=kind+" "+str(int(seed)))
            continue
        if curve_type == "A_stop_exact_endpoint":
            axis, ys = axes.flat[7], [row["MSE"] for row in group]
        else:
            axis, ys = axes.flat[6], [number(row.get("mean_observed_training_batch_loss")) for row in group]
        usable = [(number(row["updates"]), y) for row, y in zip(group, ys) if y is not None]
        if usable:
            axis.plot([v[0] for v in usable], [v[1] for v in usable], marker="o", ms=3, label=initialization_curve_label(kind, seed, end))
    for axis, metric in ((axes.flat[4], "MSE"), (axes.flat[5], "MAE")):
        axis.set_title("PCA/QR exact A_fit endpoint | stride1, A50 search scale")
        axis.set_xlabel("Search updates"); axis.set_ylabel("A_fit macro "+metric)
    axes.flat[6].set_title("PCA/QR mean stochastic segment loss (not endpoint loss)")
    axes.flat[6].set_xlabel("Updates recorded by segment receipt"); axes.flat[6].set_ylabel("Mean observed training-batch loss")
    axes.flat[7].set_title("PCA/QR exact A_stop endpoints, A50 search scale")
    axes.flat[7].set_xlabel("Search updates"); axes.flat[7].set_ylabel("A_stop macro MSE")
    for axis in axes.flat[4:]:
        if axis.lines:
            axis.legend(fontsize=7)
        else:
            axis.text(.5, .5, "PCA/QR curves not yet available", transform=axis.transAxes, ha="center", fontsize=8)
    fig.suptitle("Learning amount | common endpoints and selected refits are separate; completion is not convergence")
    fig.tight_layout(); _save(fig, "learning_curves", report, metadata)
    for dimension, figure_name in (("horizon", "horizon_delta"), ("channel", "channel_delta")):
        profiles = error_profiles(data, dimension)
        fig = error_profile_figure(plt, profiles, dimension)
        _save(fig, figure_name, report, metadata)
        metadata[figure_name]["paired_profiles"] = profiles
    fig, axes = plt.subplots(3, 4, figsize=(22, 12), gridspec_kw={"width_ratios": [3, 3, 1.6, 1.6]})
    backbone_sources = []
    for panels, cohort in zip(axes, ("jena", "pems08", "pems32")):
        representatives = {}
        for run in data["runs"]:
            if run["cohort"] != cohort or run.get("seed") not in (92601, None):
                continue
            if run["method"] in ("Raw", "B0", "B2", "CAP", "Ridge", "LoRA", "Chronos2_nativeRaw") or run["method"].startswith("Moirai_"):
                representatives.setdefault(run["method"], run)
        backbone_sources.extend(dict(cohort=cohort, method=method, run_id=run["run_id"],
            source_sha256=run["source_sha256"], device=run.get("device"), dtype=run.get("dtype"))
            for method, run in representatives.items())
        for practical, pair in ((False, panels[:2]), (True, panels[2:])):
            chosen = [run for method, run in representatives.items() if (method == "Chronos2_nativeRaw") == practical]
            for axis, metric in zip(pair, ("MSE", "MAE")):
                labels = [run["method"]+" | "+str(run.get("device"))+" "+str(run.get("dtype")) for run in chosen]
                axis.barh(range(len(chosen)), [run[metric] for run in chosen],
                    color=["#E69F00" if practical else "#009E73" if run["method"].startswith("Moirai_") else "#0072B2" for run in chosen])
                axis.set_yticks(range(len(chosen)), labels, fontsize=7)
                axis.set_xlabel("Normalized macro "+metric)
                axis.set_title(cohort+(" | Practical alternative" if practical else " | Univariate F / regression"), fontsize=9)
                axis.invert_yaxis()
                if practical:
                    axis.set_facecolor("#fff6e8")
                    axis.set_xlim(panels[0 if metric == "MSE" else 1].get_xlim())
                    axis.set_ylim(-1, 1)
                if not chosen:
                    axis.text(.5, .5, "NOT MEASURED", transform=axis.transAxes, ha="center", fontsize=8)
    pending = [j for j in data["jobs"] if j["question_id"] in ("D02", "D03")]
    coverage = "; ".join(j["job_id"]+"="+j["status"] for j in pending)
    fig.suptitle("Backbones by cohort/device/provenance | native multivariate = practical alternative; see contract limits for mechanisms")
    fig.text(.5, -.02, coverage, ha="center", fontsize=7, wrap=True)
    fig.tight_layout(); _save(fig, "backbone_comparison", report, metadata)
    metadata["backbone_comparison"].update(selected_runs=backbone_sources,
        selection="First source-bound selected run per method/cohort, seed92601 or deterministic; no averaging. Device replicas remain in RUNS and F1.",
        native_layout="Separate right-hand practical-alternative panels; no single-factor backbone claim")
    if private_qualitative:
        qualitative(ctx, data, metadata)
    else:
        # Source-only quantitative rerender leaves the private-dependent F6 untouched.
        if not (report/"figures/qualitative_forecasts.png").exists():
            fig, axis = plt.subplots(figsize=(10, 4)); axis.axis("off")
            axis.text(.5, .5, "Qualitative figure requires approved private prediction assets", ha="center")
            _save(fig, "qualitative_forecasts", report, metadata)
    return metadata


def qualitative(ctx, data, metadata):
    import numpy as np
    import matplotlib.pyplot as plt
    from .existing_jobs import _run_root
    values, sources = {}, []
    def source_descriptor(method, digest, entry, run_id=None):
        return dict(method=method, run_id=run_id, private_source_sha256=digest,
                    shared_evaluation_array_sha256={key: hashlib.sha256(np.ascontiguousarray(entry[key]).tobytes()).hexdigest()
                        for key in ("target_normalized", "observed", "origins", "channel_ids")})
    for run in data["runs"]:
        archive, expected = run.get("_archive_path"), run.get("independent_private_score_sha256")
        if run["cohort"] != "jena" or run.get("seed") not in (92601, None) or archive is None or expected is None:
            continue
        if sha(archive) != expected:
            raise ValueError("Qualitative independently scored archive changed")
        with np.load(archive, allow_pickle=False) as source:
            if set(source.files) != {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}:
                continue
            entry = {key: source[key] for key in source.files}
        if entry["target_normalized"].shape != (43, 48, 21) or np.any(entry["origins"] < 14675) or np.any(entry["origins"]+48 > 16772):
            raise ValueError("Qualitative archive outside registered development V")
        if values:
            original = next(iter(values.values()))
            for key in ("target_normalized", "observed", "origins", "channel_ids"):
                np.testing.assert_array_equal(original[key], entry[key])
        values[run["method"]] = entry
        sources.append(source_descriptor(run["method"], expected, entry, run["run_id"]))
    for family in ("B0", "B2", "B2_CAP_POINTWISE"):
        if ("CAP" if family == "B2_CAP_POINTWISE" else family) in values:
            continue
        if family == "B2_CAP_POINTWISE":
            archive = _run_root(ctx, 92601, family)/"selected_V.npz"
            native = _run_root(ctx, 92601, family, True)/"SELECTED_V_RESULT.json"
            expected = read(native)["private_prediction_archive_sha256"] if native.is_file() else None
        else:
            run = _run_root(ctx, 92601, family)
            cache_root = Path(ctx["asset"])/".cache"/BROAD
            archive = run.parent.parent/"predictions"/run.name/"V_select.npz"
            manifest = Path(ctx["repo"])/"research"/BROAD/run.parent.parent.relative_to(cache_root)/"predictions"/run.name/"V_select"/"PREDICTION_MANIFEST.json"
            expected = read(manifest)["archive_sha256"] if manifest.is_file() else None
        if not archive.is_file() or expected is None:
            continue
        if sha(archive) != expected:
            raise ValueError("Qualitative source archive identity changed")
        with np.load(archive, allow_pickle=False) as source:
            entry = {key: source[key] for key in ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")}
        if entry["target_normalized"].shape != (43, 48, 21) or np.any(entry["origins"]+48 > 16772):
            raise ValueError("Only original Jena development V43 allowed for qualitative figure")
        if values:
            reference = next(iter(values.values()))
            for key in ("target_normalized", "observed", "origins", "channel_ids"):
                np.testing.assert_array_equal(reference[key], entry[key])
        values["CAP" if family == "B2_CAP_POINTWISE" else family] = entry
        sources.append(source_descriptor(family, expected, entry))
    # Follow only registered prediction bindings; never open a raw/task feature cache.
    for job in data["jobs"]:
        if job["status"] not in SUCCESS or job.get("seed") not in (92601, None) or job["dataset_cohort"] != "jena":
            continue
        receipt_path = Path(ctx["report"])/"receipts"/(job["job_id"]+".json")
        result = read(receipt_path).get("result", {})
        for index, binding in enumerate(result.get("private_artifacts", [])):
            reference = binding.get("reference", "")
            if not reference.startswith("cache://") or not reference.endswith(".npz") or "prediction" not in reference.lower() or "coefficient" in reference.lower():
                continue
            if any(source["private_source_sha256"] == binding["sha256"] for source in sources):
                continue
            path = (Path(ctx["cache"])/reference.split("://", 1)[1]).resolve()
            if not path.is_relative_to(Path(ctx["cache"]).resolve()) or not path.is_file() or sha(path) != binding["sha256"]:
                raise ValueError("Qualitative new prediction binding changed")
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                if {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}.issubset(keys):
                    additions = [(job["job_id"]+"_"+path.stem, {key: archive[key] for key in ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")})]
                elif {"own_prediction", "cross_prediction", "target", "observed", "origins"}.issubset(keys) and values:
                    channels = next(iter(values.values()))["channel_ids"]
                    additions = [(job["job_id"]+"_"+method, dict(pred_normalized=archive[method+"_prediction"], target_normalized=archive["target"],
                        observed=archive["observed"], origins=archive["origins"], channel_ids=channels)) for method in ("own", "cross")]
                else:
                    continue
            for label, entry in additions:
                if entry["target_normalized"].shape != (43, 48, 21) or np.any(entry["origins"]+48 > 16772):
                    continue
                if values:
                    original = next(iter(values.values()))
                    for key in ("target_normalized", "observed", "origins", "channel_ids"):
                        np.testing.assert_array_equal(original[key], entry[key])
                values[label] = entry
                sources.append(source_descriptor(label, binding["sha256"], entry))
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    selection = dict(origin_indices=[0, 21, 42], channel_indices=[0, 10, 20], seed=92601,
                     rule="First/middle/last origin and original stored channel; fixed before viewing errors", T_access=False)
    if values:
        reference = next(iter(values.values()))
        private_curves = {}
        from matplotlib.colors import to_hex
        styles = {method: dict(color=to_hex(plt.get_cmap("tab20")(index % 20)), linestyle=("-", "--", "-.", ":")[(index % 4+index//20) % 4])
                  for index, method in enumerate(values)}
        selection["method_styles"] = styles
        for oi, origin_index in enumerate(selection["origin_indices"]):
            for ci, channel in enumerate(selection["channel_indices"]):
                axis = axes[oi, ci]
                observed = reference["observed"][origin_index, :, channel]
                target = np.where(observed, reference["target_normalized"][origin_index, :, channel], np.nan)
                axis.plot(range(1, 49), target, color="black", lw=1.7, label="Observed target")
                private_curves[f"target_{oi}_{ci}"] = target
                for method, value in values.items():
                    curve = value["pred_normalized"][origin_index, :, channel]
                    axis.plot(range(1, 49), curve, label=method, lw=1.1, **styles[method])
                    private_curves[f"prediction_{method}_{oi}_{ci}"] = curve
                axis.set_title("o="+str(int(reference["origins"][origin_index]))+" | "+str(reference["channel_ids"][channel]), fontsize=9)
                axis.set_xlabel("Future horizon step"); axis.set_ylabel("A60 normalized value"); axis.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.12), ncol=3, fontsize=7)
        from .ridge_jobs import save_npz
        private_path = Path(ctx["cache"])/"report_qualitative_curves.npz"
        save_npz(private_path, **private_curves)
        selection["private_selected_curves_sha256"] = sha(private_path)
        selection["origins"] = [int(reference["origins"][i]) for i in selection["origin_indices"]]
        selection["channels"] = [str(reference["channel_ids"][i]) for i in selection["channel_indices"]]
    else:
        for axis in axes.flat:
            axis.axis("off")
        axes[1, 1].text(.5, .5, "Missing approved private arrays; no curves fabricated", ha="center")
    fig.suptitle("Jena repeated V | fixed first/middle/last selection, full H48; normalized units")
    fig.tight_layout(); _save(fig, "qualitative_forecasts", ctx["report"], metadata)
    metadata["qualitative_forecasts"].update(selection=selection, private_sources=sources,
        exact_reproduction_requires_approved_private_assets=True, raw_curve_values_published=False)


def next_steps_text(ctx, data):
    report = Path(ctx["report"])
    contrasted = comparisons(data)+comparisons(data, contextual=True)
    paragraphs = ["# 결과 확인 후의 후속안"]
    for family, reference, method, interpretation in (
        ("B01", "B0_FULL_HISTORY_PROBE", "B2_FULL_HISTORY_PROBE", "λ1 고정 full-history reader의 조건부 접근성 결과이며 표현 정보 전체의 부재나 우위로 확대하지 않는다."),
        ("B03", "B2_DECODER_SHARED", "B2_DECODER_HORIZON", "Dh와 shared의 오차 방향·용량·선택 regularization을 함께 읽고 단일 원인으로 단정하지 않는다.")):
        observations = [row for row in contrasted if row["comparison_family"] == family and row["reference"] == reference and row["method"] == method]
        if observations:
            details = "; ".join("seed="+str(row["seed"])+" ΔMSE="+repr(row["delta_MSE_pct"])+"%, ΔMAE="+repr(row["delta_MAE_pct"])+"% ("+row["match_status"]+")" for row in observations)
            direction = ("모든 완료 seed에서 B2 probe의 두 오차가 B0보다 높다. " if family == "B01" and all(row["delta_MSE"] > 0 and row["delta_MAE"] > 0 for row in observations)
                else "모든 완료 seed에서 Dh는 shared보다 MSE가 높고 MAE가 낮다. " if family == "B03" and all(row["delta_MSE"] > 0 and row["delta_MAE"] < 0 for row in observations) else "seed·오차별 관측 방향을 개별적으로 유지한다. ")
            paragraphs.append("[확인] "+family+" 실제 관측: "+details+". "+direction+interpretation)
        else:
            paragraphs.append("[미검증] "+family+"의 완료 대조가 없어 해당 설명은 보류한다. "+interpretation)

    def profile_estimate(relative, key):
        path = report/relative
        if not path.is_file():
            return None, "UNKNOWN: 해당 조건의 실제 training profile 없음"
        recorded = {(job.get("artifact_hashes") or {}).get(relative) for job in data["jobs"]}
        if sha(path) not in recorded:
            return None, "UNKNOWN: 현재 plan/job 영수증의 SHA에 결속된 실제 training profile 없음"
        seconds = number(read(path).get(key))
        if seconds is None:
            return None, "UNKNOWN: 실제 profile에 "+key+" 없음"
        _source(path, ctx, data["bindings"])
        return seconds, "[추정] "+relative+" (SHA256="+sha(path)+")의 "+key+"="+repr(seconds)+"초, 약 "+format(seconds/3600, ".2f")+"시간. 실제 disposable update profile의 보수적 외삽이며 전체 완주 실측이 아니다"

    pending_moirai = []
    for question, names, label, unit in (
        ("C02", {"B2_PCA", "B2_QR"}, "초기화", "같은 seed의 PCA/QR fresh-fit pair"),
        ("D02", {"Moirai_Raw", "Moirai_B0", "Moirai_B2"}, "두 번째 단변량 backbone", "같은 seed의 Moirai Raw/B0/B2 trio"),
        ("D03", {"Chronos2_nativeRaw"}, "native multivariate 대안", "해당 cohort의 nativeRaw V 평가와 동일 조건 E2E 비용 확인")):
        jobs = [job for job in data["jobs"] if job["question_id"] == question]
        completed, pending = [], []
        for job in jobs:
            methods = {row["method"] for row in data["runs"] if row["run_id"].startswith(job["job_id"]+"_")}
            (completed if job["status"] in SUCCESS and names.issubset(methods) else pending).append(job)
        details = "; ".join(job["job_id"]+"="+job["status"]+("; "+str(job.get("reason")) if job.get("reason") else "") for job in pending)
        if pending:
            paragraphs.append(question+" "+label+": 실제 rows와 job 상태로 완료 "+str(len(completed))+"/"+str(len(jobs))+"개를 확인했다. 남은 항목: "+details+". 추가 실행의 최소 단위는 "+unit+"이며 한 arm만으로 결론내리지 않는다.")
            if question in ("C02", "D02"):
                key = "conservative_pair_seconds" if question == "C02" else "conservative_full_trio_seconds"
                for job in pending:
                    _, estimate = profile_estimate("profiles/"+job["job_id"]+".json", key)
                    paragraphs.append(job["job_id"]+" 비용: "+estimate+".")
                    if question == "D02":
                        pending_moirai.append(job["job_id"])
        elif jobs:
            paragraphs.append(question+" "+label+": 등록된 "+str(len(completed))+"개 평가가 rows와 완료 영수증에 있다. 같은 pair/trio의 재실행을 자동 추가하지 않으며, 남은 질문은 용량·선택 학습량·반복 V의 한계를 분리할 수 있는지다.")
        else:
            paragraphs.append(question+" "+label+": 등록된 job/완료 rows가 없어 상태와 추가비용은 UNKNOWN이다. 필요시 새 승인 뒤 "+unit+"만 검토한다.")

    cpu_jobs = [job for job in data["jobs"] if job["question_id"] == "C01" and job["device"] == "cpu"]
    gpu_jobs = [job for job in data["jobs"] if job["question_id"] == "C01" and job["device"] == "cuda"]
    available = {row["run_id"] for row in data["runs"]}
    cpu_complete = len(cpu_jobs) == 2 and all(job["status"] in SUCCESS and job["job_id"] in available for job in cpu_jobs)
    gpu_complete = len(gpu_jobs) == 2 and all(job["status"] in SUCCESS and job["job_id"] in available for job in gpu_jobs)
    if cpu_complete:
        paragraphs.append("[확인] C01 원 CPU 동일 device pair의 완전한 continuation/선택·재적합/V 평가가 완료 rows와 영수증에 있다. 같은 실행을 후속 큐로 자동 재등록하지 않는다.")
    else:
        seconds, estimate = profile_estimate("profiles/C00_pems_pair_cpu.json", "conservative_complete_pair_seconds")
        paragraphs.append("C01 원 CPU exact continuation은 남은 장치 재현 질문이다. "+("[확인] CUDA numerical branch pair는 이미 완료했으며 역사적 CPU bitwise continuation과 구분한다. " if gpu_complete else "CUDA pair의 완전한 완료도 실제 rows/영수증 기준으로 보류한다. ")+
            "기존 CPU 단일 next-update restore 검사와 원 CPU search/refit 전체 완료는 별개다. 최소 추가 단위는 원 CPU 동일 device B0/B2 pair이며, "+estimate+
            (". 이는 현재 6시간 전체 캠페인 예산 밖이므로 별도 장시간 예산 승인 전 실행 대상으로 삼지 않는다." if seconds is not None and seconds > 21600 else ". 원 device의 새 실제 profile과 별도 승인 예산을 확인하기 전 추가 실행하지 않는다."))
    paragraphs.append("이번 고정 계획을 자동 확대하지 않는다. 남은 승인 큐의 status/reason/dependency와 실제 잔여 예산을 먼저 확인하고, 예산 종료 뒤 추가 실행은 새 승인 예산에서 위 최소 pair/trio/cohort 단위만 검토한다. 기존 deadline/사용량을 초기화하거나 자동 대형큐를 재개하지 않는다.")
    paragraphs.append("D01 합성 smoke/profile 시간을 Moirai 또는 Chronos2의 실제 전체 학습·평가 시간으로 외삽하지 않는다. "+("미완료 Moirai 항목은 위에 실제 training profile 값이 있을 때만 인용했다. " if pending_moirai else "Moirai의 후속 비용은 해당 실제 training profile/측정 범위를 따른다. ")+"Ridge의 정확도·비용 우위가 확인된 cohort는 실용적 약점으로 유지한다. 새 독립 test/generalization 평가의 자료 확보·학습·전체 비용은 UNKNOWN이며, 개발 V의 개선을 독립 일반화로 바꾸어 주장하지 않는다.")
    return "\n\n".join(paragraphs)+"\n"


def practical_cost_observations(data):
    groups = defaultdict(dict)
    for cost in data["costs"]:
        if (cost.get("cohort") != "jena" or cost.get("device") not in ("cpu", "cuda") or cost.get("exclusive") is not True or cost.get("parity_state") != "PASS"
                or number(cost.get("median_ms_per_origin")) is None or float(cost["median_ms_per_origin"]) <= 0
                or any(not re.fullmatch("[a-f0-9]{64}", str(cost.get(key, ""))) for key in ("physical_input_sha256", "observation_mask_sha256"))):
            continue
        numeric = tuple(number(cost.get(key)) for key in ("B", "C", "L", "H", "threads"))
        if any(value is None for value in numeric) or not cost.get("scope"):
            continue
        run = cost_accuracy_match(cost, data["runs"])
        if run is None or any(number(run.get(key)) is None for key in ("MSE", "MAE")):
            continue
        key = (cost.get("device"), *numeric, cost["physical_input_sha256"], cost["observation_mask_sha256"], cost["scope"])
        groups[key][canonical_method(cost["method"])] = (cost, run)
    output = []
    for key, group in sorted(groups.items()):
        device, batch = key[:2]
        reference = "Ridge" if device == "cpu" else "Raw" if device == "cuda" else None
        if reference not in group or "B2" not in group:
            continue
        left, a = group[reference]; right, b = group["B2"]
        if any(a.get(field) in (None, "", "NA") or a.get(field) != b.get(field)
               for field in ("split", "metric_scale", "scaler_sha256", "target_sha256", "mask_sha256", "origins_sha256", "channel_sha256")):
            continue
        x, y = float(left["median_ms_per_origin"]), float(right["median_ms_per_origin"])
        facet = "같은 device/B/cohort/physical input/mask/scope/threads facet"
        if device == "cpu":
            p, q = left.get("core_dtype"), right.get("core_dtype")
            if p not in ("float32", "float64") or q not in ("float32", "float64"):
                continue
            dominance = float(a["MSE"]) < float(b["MSE"]) and float(a["MAE"]) < float(b["MAE"]) and x < y
            output.append("[확인] Jena CPU B"+str(int(batch))+" 실용 대조: Ridge core="+p+" MSE="+repr(a["MSE"])+", MAE="+repr(a["MAE"])+
                ", warm ms/origin="+repr(x)+"; B2 core="+q+" MSE="+repr(b["MSE"])+", MAE="+repr(b["MAE"])+", warm ms/origin="+repr(y)+
                ". "+facet+". "+("Ridge가 두 오차와 지연 모두 낮다. 이 조건에서 B2의 실용적 약점이다. " if dominance else "두 오차·지연의 동시 우위 조건은 성립하지 않는다. ")+
                ("core 정밀도가 다르며 동일 정밀도 단일요인 비교가 아니다." if p != q else "다른 모델·용량의 실용 대조이며 단일요인 비교가 아니다."))
        elif left.get("dtype") not in (None, "", "source_bound") and left.get("dtype") == right.get("dtype"):
            ratio = x/y
            direction = "빠름" if ratio > 1 else "느림" if ratio < 1 else "같음"
            output.append("[확인] Jena GPU Bolt Raw/B2 B"+str(int(batch))+", dtype="+str(left["dtype"])+": Raw="+repr(x)+
                ", B2="+repr(y)+" warm ms/origin; Raw/B2 지연 비="+repr(ratio)+"x, B2가 "+direction+". "+facet+
                ". 배치별 실측이며 C→K 변환이 항상 빠르다고 일반화하지 않는다.")
    return output


def _documents(ctx, data, matched):
    report = Path(ctx["report"])
    presentation = report_helper(ctx, "report_presentation", data["bindings"])
    counts = Counter(j["status"] for j in data["jobs"])
    closed = all(j["status"] not in ("PLANNED", "RUNNING") for j in data["jobs"])
    scientific = "SCIENTIFIC_COMPLETE" if all(j["status"] in SUCCESS for j in data["jobs"]) else "SCIENTIFIC_PARTIAL"
    audit = read(report/"DATA_AUDIT.json")
    measured = [r for r in data["runs"] if r.get("cohort") == "jena" and number(r.get("MSE")) is not None]
    best = min(measured, key=lambda r: r["MSE"]) if measured else None
    common_compared = same_update_comparisons(data)
    questions = [
        ("Q1", "데이터·평가", "[확인] pre-T 경계·mask·zero·scaler 지문과 독립 FP64/KKT 검사를 수행했다. 상류 보간·pretraining overlap UNKNOWN, 반복 V·과거 T46노출을 유지한다."),
        ("Q2", "시간·표현·디코더·최적화", "[확인] Jena B2는 B0보다 MSE가 낮지만 일부 단기 MAE가 악화된다. CAP same-update·고정 PE probe·shared/Dh 완료 범위만 인과 후보를 좁힌다. 미완료 job은 JOB_STATUS에 남긴다."),
        ("Q3", "PCA 초기화", "[미검증] PCA/QR 쌍이 모두 닫힌 seed에서만 비교한다. 미완료·예산중단은 초기화의 우열 증거가 아니다."),
        ("Q4", "단변량 backbone 교체", "[미검증] D01 smoke/parity와 실제 matched Raw/B0/B2 trio는 다르다. D02 미완료는 adapter 일반성 미확인을 뜻한다."),
        ("Q5", "native multivariate 대안", "[미검증] Chronos2 D01과 실제 nativeRaw 개발 V·공정 비용은 구분한다. D03이 없으면 실용 우열을 결론내리지 않는다."),
        ("Q6", "정확도·전체 비용", "[확인] 새 독점 E2E 측정만 주 비교에 쓴다. 미측정 Raw latency/speedup은 NA다. Ridge 우위 cohort를 유지하며 보편적 TSFM 우열은 주장하지 않는다.")]
    if data.get("m5_local_assets"):
        questions[0] = ("Q1", questions[0][1], questions[0][2]+"\n\n[확인] 사용자 요청 뒤 넓힌 로컬 검색에서 M5 sales validation/evaluation·calendar·sell_prices와 원 ZIP을 찾았다. [로컬 자산 검증](M5_LOCAL_ASSET_DISCOVERY.json)은 파일 SHA·크기·header만 대사한다. 원 A01의 401은 과거 외부 접근과 제한된 inventory 기록이며, 현재 로컬 접근 차단을 뜻하지 않는다. M5 수치 행 파싱·학습·평가는 수행하지 않았고, M5 입력·split·평가 계약은 NOT_REGISTERED다.")
    def paired_observation(reference, method, prefix=None):
        run_map = {r["run_id"]: r for r in data["runs"]}
        relevant = [pair for pair in comparisons(data)+comparisons(data, contextual=True)
                    if pair["reference"] == reference and pair["method"] == method
                    and (prefix is None or pair["method_run_id"].startswith(prefix))]
        pairs, descriptions = defaultdict(list), []
        for pair in relevant:
            ref, val = run_map[pair["reference_run_id"]], run_map[pair["method_run_id"]]
            cohort, seed = pair["cohort"], pair["method_seed"]
            dmse, dmae, reasons = pair["delta_MSE_pct"], pair["delta_MAE_pct"], pair["unresolved_contract_fields"]
            pairs[(cohort, bool(reasons), ref.get("device"), val.get("device"))].append((dmse, dmae))
            descriptions.append(("[맥락 비교: "+", ".join(reasons)+"] " if reasons else "[계약 동등성 확인] ")+cohort+" seed"+str(int(seed) if seed is not None else "none")+
                " (reference seed="+str(pair["reference_seed"])+", "+str(ref.get("device"))+"→"+str(val.get("device"))+"): "+method+
                " MSE="+repr(val["MSE"])+", MAE="+repr(val["MAE"])+"; 대 "+reference+
                " ΔMSE="+repr(dmse)+"%, ΔMAE="+repr(dmae)+"%")
        for (cohort, contextual, ref_device, method_device), values in pairs.items():
            if len(values) >= 2:
                descriptions.append(cohort+" "+str(ref_device)+"→"+str(method_device)+(" 맥락 비교 관측치의" if contextual else " 계약 동등성이 확인된")+" 완료 seed별 ΔMSE 평균="+repr(sum(v[0] for v in values)/len(values))+"%, 범위=["+
                    repr(min(v[0] for v in values))+","+repr(max(v[0] for v in values))+"]; ΔMAE 평균="+
                    repr(sum(v[1] for v in values)/len(values))+"%, 범위=["+repr(min(v[1] for v in values))+","+repr(max(v[1] for v in values))+"]%. CI가 아니다.")
        return "\n\n".join(descriptions)
    b2 = paired_observation("B0", "B2")
    cap = paired_observation("CAP", "B2")
    probe = paired_observation("B0_FULL_HISTORY_PROBE", "B2_FULL_HISTORY_PROBE", "B01")
    probe_cap = paired_observation("B2_CAP_POINTWISE_FULL_HISTORY_PROBE", "B2_FULL_HISTORY_PROBE", "B01")
    b02 = paired_observation("OWN_CHANNEL_32BLOCK_RIDGE", "ALL_CHANNEL_32BLOCK_RIDGE", "B02")
    decoder = paired_observation("B2_DECODER_SHARED", "B2_DECODER_HORIZON", "B03")
    original_decoder = paired_observation("B2_ORIGINAL_D", "B2_DECODER_SHARED", "B03")
    probe_pairs = [r for r in comparisons(data)+comparisons(data, contextual=True) if r["reference"] == "B0_FULL_HISTORY_PROBE" and r["method"] == "B2_FULL_HISTORY_PROBE"]
    probe_interpretation = ("[확인] 완료한 모든 seed에서 B2 probe의 MSE와 MAE가 B0보다 높다. 따라서 이 관측은 B2 표현의 고정 선형 접근성 개선을 뒷받침하지 않는다. λ1 고정·3,097,584 parameter의 full-history reader 조건에 한정된 반례이며, 표현의 모든 비선형 정보나 일반성을 판정하지 않는다."
        if probe_pairs and all(r["delta_MSE"] > 0 and r["delta_MAE"] > 0 for r in probe_pairs) else "고정 λ1·대규모 full-history reader의 조건부 결과이며, 표현 우위로 일반화하지 않는다.")
    questions[1] = ("Q2", questions[1][1], "[확인] 기존 selected-refit B2−B0 관측:\n\n"+b2+
        ("\n\n동일 근접용량 CAP 대조:\n\n"+cap if cap else "")+
        ("\n\n고정 PE full-history reader:\n\n"+probe if probe else "\n\n[미검증] B01 완료 probe 쌍이 없다.")+
        ("\n\nCAP 고정 PE reader 대조:\n\n"+probe_cap+"\n\n"+probe_interpretation if probe_cap else "")+
        ("\n\n채널 정보 own/cross Ridge 대조; cohort별 방향을 따로 읽는다:\n\n"+b02 if b02 else "")+
        ("\n\n원 공동학습 D 대 재적합 shared D 비교:\n\n"+original_decoder if original_decoder else "")+
        ("\n\n재적합 shared 대 Dh:\n\n"+decoder if decoder else "\n\n[미검증] B03 shared/Dh 완료 쌍이 없다.")+
        "\n\n"+common_endpoint_interpretation(data, common_compared)+
        "\n\nA_stop-selected refit의 학습량·용량 차이가 남는다. SAME_UPDATE_RESULTS의 동일 endpoint를 별도로 읽는다. 기존 H1~16 MAE 악화와 일부 채널 손해를 숨기지 않는다.")
    questions[1] = ("Q2", questions[1][1], questions[1][2]+
        "\n\n[TRAJECTORY_DIAGNOSTICS.json](TRAJECTORY_DIAGNOSTICS.json)은 실제 저장 sampler의 permutation·cursor·epoch·RNG를 비교한다. 중간의 전체 과거 소비 순서는 별도 로그가 없어 입증하지 않는다. 같은 파일과 RUNS의 fit_MSE/fit_MAE는 원 D·shared·Dh의 fit/V 차이를 보존한다. 원 D의 FP64 core 재계산과 원 FP32 core 점수는 수치 공차와 함께 구분한다.\n\n[H1~16/17~32/33~48](HORIZON_BAND_SUMMARY.csv), [paired-origin coverage](PAIRED_ORIGIN_COVERAGE.csv)에서 MSE와 MAE를 함께 확인한다. origin 차이 원수치는 private에 보존하며 공개 SHA로 결속한다.")
    initialization = paired_observation("B2_PCA", "B2_QR", "C02")
    questions[2] = ("Q3", questions[2][1], ("[확인] 실제 완료한 초기화 쌍:\n\n"+initialization+
        "\n\n두 arm의 common endpoint sampler/order 지문과 P/center 초기 지문은 원 job receipt에 결속했다. 해당 K·F·고정 학습 예산에서의 관측이며 PCA 일반적 필요성·설명분산과 미래정보 동일성을 입증하지 않는다.")
        if initialization else "[미검증] 실제 PCA/QR fresh-fit 쌍이 아직 닫히지 않았다. 예산/접근/프로토콜 실패를 초기화의 우열로 해석하지 않는다.")
    moirai = paired_observation("Moirai_B0", "Moirai_B2", "D02")
    moirai_raw = paired_observation("Moirai_Raw", "Moirai_B2", "D02")
    questions[3] = ("Q4", questions[3][1], ("[확인] 완료한 두번째 단변량 backbone의 비교; 행별 계약 검증 상태를 따른다:\n\n"+moirai+"\n\n"+moirai_raw+
        "\n\n선택 epoch·용량 차이와 반복 V 제한을 함께 유지한다. 두 백본 관측을 모든 TSFM 일반성으로 확대하지 않는다.")
        if moirai else "[미검증] 실제 Moirai Raw/B0/B2 matched trio가 닫히지 않았다. D01 공식 parity·입력 gradient smoke는 D02 정확도 검증과 구분한다.")
    native = []
    for run in data["runs"]:
        if run["method"] != "Chronos2_nativeRaw":
            continue
        reference = next((r for r in data["runs"] if r["cohort"] == run["cohort"] and r["method"] == "Raw"), None)
        text = run["cohort"]+": MSE="+repr(run["MSE"])+", MAE="+repr(run["MAE"])+", device="+str(run.get("device"))
        if reference:
            text += "; 기존 Bolt Raw 대비 ΔMSE="+repr(100*(run["MSE"]-reference["MSE"])/reference["MSE"])+"%, ΔMAE="+repr(100*(run["MAE"]-reference["MAE"])/reference["MAE"])+"%"
            reasons = pair_contract(reference, run)
            text += "; 맥락 비교 제한="+", ".join(reasons) if reasons else "; 평가 계약 동등성 확인"
        native.append(text)
    questions[4] = ("Q5", questions[4][1], ("[확인] 실제 native multivariate 개발 V 평가:\n\n"+"\n\n".join(native)+
        "\n\nbackbone과 native channel grouping이 함께 바뀐 실용 비교다. B2 P/E의 단일요인 메커니즘 증거가 아니며, 정확도와 공정 E2E 비용을 함께 봐야 한다.")
        if native else "[미검증] D03 실제 nativeRaw cohort 평가가 아직 없다. D01 smoke/profile만으로 실용 대안의 우열을 판단하지 않는다.")
    questions[4] = ("Q5", questions[4][1], questions[4][2]+"\n\n"+
        "\n\n".join(paired_observation(method, "Chronos2_nativeRaw") for method in ("B2", "Ridge", "LoRA"))+
        "\n\nE04는 이미 선택된 모델을 측정 장치에서 다시 예측한 수치 경로다. 원 CPU 결과를 덮지 않으며 run ID·device·dtype·prediction SHA를 비용에 직접 연결한다.")
    cost_text = [str(row["cohort"])+" "+str(row["method"])+" "+str(row["device"])+" B"+str(row["B"])+
                 ": warm ms/origin="+str(row.get("median_ms_per_origin"))+", status="+str(row.get("status")) for row in data["costs"]]
    questions[5] = ("Q6", questions[5][1], ("[확인] 실제 새 비용 영수증:\n\n"+"\n\n".join(cost_text) if cost_text else "[미검증] 새 공정 E2E 비용 완료 영수증이 없다.")+
        "\n\n"+"\n\n".join(practical_cost_observations(data))+
        "\n\n속도 비교는 장치·batch·dtype·physical input 지문이 같은 facet을 따른다. 위 CPU Ridge 실용 대조는 별도로 core 정밀도 차이를 명시한다. 미측정 Raw latency/speedup은 NA다. resident model params·trainable params·logical F rows·RSS sampling은 서로 다른 지표다.")
    summary = "상태 수: "+", ".join(f"{key}={value}" for key, value in sorted(counts.items()))+"."
    text = ["# Adapter 진단 결과", "", "[확인] "+scientific+". ENGINEERING_COMPLETE는 실제 전체 검증·게시·원격 SHA 확인 이후에만 판정한다.",
        "모든 정확도 결과는 반복 사용한 개발 V다. 기존 결과 재사용과 이번 신규 실행/재채점은 RUNS_FULL_PRECISION의 origin 열로 구분한다.", summary, "",
        ("현재 수집된 Jena 최소 MSE는 "+best["method"]+" `"+repr(best["MSE"])+"`, MAE `"+repr(best["MAE"])+"`이다. 서로 다른 기능·용량·학습량의 결과를 포함하므로 순수 구조 효과로 읽지 않는다.") if best else "완료 정확도 없음.",
        "정확도 허용폭·실용 속도 목표는 UNREGISTERED다. 2seed는 각각 보이며 범위 이상의 일반화·수렴·계층적 모든 K-prefix를 입증하지 않는다.", "",
        "[전체 job 상태](JOB_STATUS.csv) · [정확도](RUNS_FULL_PRECISION.csv) · [공통 update](SAME_UPDATE_RESULTS.csv) · [같은 endpoint 대조](SAME_UPDATE_COMPARISONS.csv) · [계약 확인 비교](MATCHED_COMPARISONS.csv) · [제한 있는 맥락 비교](CONTEXTUAL_COMPARISONS.csv) · [비용](COST_FULL_PRECISION.csv) · [질문별 판단](QUESTIONS_AND_DECISIONS_KO.md) · [감사](VALIDATION.json)", ""]
    text += [presentation.readme_supplement(data),
             "[자료 감사 보완](DATA_AUDIT_SUPPLEMENT_KO.md) · [자료 출처·단위](DATA_SOURCE_DETAILS.json) · [horizon 구간 요약](HORIZON_BAND_SUMMARY.csv) · [paired-origin 결속](PAIRED_ORIGIN_BINDINGS.json)", ""]
    for figure in FIGURES:
        text += ["!["+figure+"](figures/"+figure+".png)", ""]
    atomic(report/"README_KO.md", "\n".join(text))
    atomic(report/"QUESTIONS_AND_DECISIONS_KO.md", "# 질문별 판단\n\n"+"\n\n".join(q+" "+title+"\n\n"+body for q, title, body in questions)+"\n\n결과에 따른 다음 변경은 NEXT_STEPS_KO.md에 적었다.\n")
    atomic(report/"METHOD_CONTRACT.md", "# 고정 방법·변경 요인\n\nB2: Z=E(P(X)); U=frozen_F(Z); Yhat=X_last+D(phi_lag(Z,U)). P는 temporal/channel residual MLP, E는 PCA prefix 초기화 학습행렬, D는 lag-major [7,K,C], lags=(0,1,2,4,8,16,32)다. D[0]=pinv(E), 나머지는0이다. h별 phi가 다르고 D는 공유한다. 원채널 미래 관측 mask를 반영한 정규화 macro MSE로 P/E/D를 공동 학습하며 F frozen과 input autograd를 함께 유지한다.\n\nA_fit/A_stop/M_cal/V/T=50/10/10/10/20. 학습 stride1, 평가 strideH. A_stop 선택 뒤 A60 fresh refit하고 scaler/PCA도 새로 구한다. M_cal/T 선택 사용 금지. CP0/.25/1/3 endpoint 유지. SAME_UPDATE_COMPARISONS는 A_fit search 가중치의 같은 epoch/updates/exposures/split/scale 대조이며 A_stop 선택·selected fresh-A60 refit과 구분한다.\n\nB01은 선택 A60 PE 고정 full [512,6] history, lambda1 고정, F/D 호출0. B02는 own32block 대 all-channel32block, 같은 alpha grid(.01,.1,1,10,100), SSE_c/N_c+alpha L2, intercept 비규제다. B03은 stage별 PE/cache·common RMS를 공유하고 shared penalty lambda/C, horizon penalty lambda/(C×H)를 맞춘다. A60 PE로 A_stop 선택 금지. C02는 PCA/QR만 변경한 matched fresh fit 쌍이다.\n\nB03 parameter 각주: trainable_params는 D coefficient block 수(shared/원 D=882, horizon-specific=42336)다. 이번 진단에서 실제 적합한 fit_stage_params는 원 D=0, 재적합 shared=882, horizon-specific=42336이다. 앞서 target-trained된 retained P/E 34835개를 포함한 deployed_trainable_params는 원/shared=35717, horizon-specific=77171이다. 따라서 882개짜리 전체 모델로 해석하지 않는다. frozen backbone resident parameter 수는 다시 별도다.\n\nPoint output은 source-bound이며 Bolt/새 quantile 모델의 median0.5와 Ridge deterministic regression을 구분한다. Raw/C→K stream rows, 파라미터, 메모리, E2E latency는 별도 지표다.\n")
    atomic(report/"NEXT_STEPS_KO.md", next_steps_text(ctx, data))
    a03_note = "\n\n공개 stdlib 검산 helper의 실제 인수는 --repo와 --cache-root다.\n\n```text\npython -B reports/adapter_diagnostics_20261017_v2/code/verify_a03_snapshot.py --repo . --cache-root PRIVATE_CACHE\n```\n\n[검산 코드](code/verify_a03_snapshot.py)는 공개 plan/score/receipt JSON으로 cohort·진행량·지문·FP64 집계를 확인하고 PRIVATE_CACHE에 검증 JSON을 쓴다. 원자료·예측 NPZ·checkpoint를 로드하지 않으며, private 실물의 지문 재검증은 G7의 별도 결속 경로다."
    a03_path = report/"A03_INDEPENDENT_COHORT_CHECK.json"
    if a03_path.is_file():
        a03 = read(a03_path)
        _source(a03_path, ctx, data["bindings"])
        helper = Path(ctx["repo"])/"reports"/NAME/"code/verify_a03_snapshot.py"
        if helper.is_file():
            _source(helper, ctx, data["bindings"])
        a03_note += " [기록된 A03 검산 영수증](A03_INDEPENDENT_COHORT_CHECK.json): status="+str(a03.get("status"))+", exit_code="+str(a03.get("verification_exit_code"))+", endpoint="+str(a03.get("checked_jobs"))+", public artifact SHA="+str(a03.get("checked_public_artifact_hashes"))+", CP0 group="+str(a03.get("CP0_family_groups"))+", parity PASS="+str(a03.get("parity_pass_jobs"))+". 원 private 예측값은 공개하지 않는다."
    atomic(report/"REPRODUCE_KO.md", "# 검증한 재현 경로\n\n명령의 repository working directory는 clone root다. 기본 실행은 no-op이다.\n\n```text\npython -m research.adapter_diagnostics_20261017_v2.run\npython -m research.adapter_diagnostics_20261017_v2.run --mode audit --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE\npython -m research.adapter_diagnostics_20261017_v2.run --mode plan --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE\npython -m research.adapter_diagnostics_20261017_v2.run --mode execute --plan REPORT_ROOT/EXPERIMENT_PLAN.json --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE\npython -m research.adapter_diagnostics_20261017_v2.run --mode report --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE\npython -m research.adapter_diagnostics_20261017_v2.run --mode verify --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE\npython -m research.adapter_diagnostics_20261017_v2.reporting --reaggregate REPORT_ROOT\n```"
        +"\n실제 실행 및 종료코드는 VALIDATION.json과 root 실행 영수증을 따른다. plan/execute는 봉인 source/data identity가 맞는 승인 private 자산을 필요로 한다. SOURCE_BINDINGS.json은 수치 source SHA를, requirements-lock.txt와 START_STATE는 환경·출발 commit을 남긴다.\n\n마지막 --reaggregate는 공개 ERROR_BY_CHANNEL의 SSE/SAE/count로 macro 점수를 독립 재집계하며 원자료·모델·네트워크를 쓰지 않는다. --figures REPORT_ROOT는 공개 집계 CSV만으로 F1~F5를 재생성한다. F6 curve 원수치는 private에만 있으며 승인된 해당 hash의 private prediction이 있어야 정확히 재생성할 수 있다. main clone만으로 원자료까지 완전 재현한다고 주장하지 않는다."+a03_note+"\n")
    with (report/"REPRODUCE_KO.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n\nE04 보완 실행은 원 106-job 실행기와 별도로 수행했다. 공개 worker는 [lora_cost_runner.py](code/lora_cost_runner.py)와 [lora_cost_runner_v2.py](code/lora_cost_runner_v2.py)의 execute_job(job, ctx)이며, 승인 private 자산·원 누적 CAMPAIGN 장부·인증 GPU parent lease가 필요하다. 두 private controller의 SHA는 확장 계획과 receipt의 execution_control에 결속되어 있다. 실제 실행 명령은 아래와 같다.\n\n```text\npython PRIVATE_CACHE/e04_orchestrate.py --execute\npython PRIVATE_CACHE/e04_parent_v2_orchestrate.py --execute\n```\n\n첫 controller는 Jena parent convention 실패와 PeMS 성공을 모두 보존했다. v2는 Jena만 교정했으며 PeMS를 다시 측정하지 않았다. 초기 manifest의 artifact/state 2-field label과 deployment의 scaler 포함 3-field parent를 각각 exact 검증한다. 기존 106개 job과 원예산은 변하지 않았다. 위 generic run --mode execute는 최초106개 계획을 실행한 역사적 명령이며, 확장 E04 재실행은 별도 worker/parent lease 경로를 따른다.\n\n실제로 검증한 모델 로드 없는 자산 검사:\n\n```text\npython reports/adapter_diagnostics_20261017_v2/code/lora_cost_runner.py --inspect --cohort jena --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE\npython reports/adapter_diagnostics_20261017_v2/code/lora_cost_runner.py --inspect --cohort pems08 --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE\npython reports/adapter_diagnostics_20261017_v2/code/lora_cost_runner_v2.py --inspect --cohort jena --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE\n```\n\n보조 paired-origin/band/trajectory gate는 reporting.verify가 code/verify_report_supplements.py를 호출하여 실제 공개 집계와 private SHA를 대조한다. CODE와 report/code의 최종 테스트 대상 SHA는 TEST_VALIDATION.json에 있다.\n")
    ppt_accuracy = []
    for run in data["runs"]:
        matching = [cost for cost in data["costs"] if cost_accuracy_match(cost, [run]) is not None]
        for cost in matching or [{}]:
            ppt_accuracy.append(dict(cohort=run["cohort"], method=run["method"], seed=run.get("seed"), C=run.get("C"), K=run.get("K"),
                F_routing=f_routing(run),
                L=run.get("L"), H=run.get("H"), split=run.get("split"), MSE=run.get("MSE"), MAE=run.get("MAE"),
                warm_ms_per_origin=cost.get("median_ms_per_origin"), B=cost.get("B"), trainable_params=run.get("trainable_params"),
                cold_load_seconds=cost.get("cold_load_seconds"), cold_shared_load_seconds=cost.get("cold_shared_load_seconds"),
                cold_load_scope=cost.get("cold_load_scope"),
                **{key: run.get(key) for key in FIT_FIELDS},
                deployed_trainable_params=run.get("deployed_trainable_params"), fit_stage_params=run.get("fit_stage_params"),
                parameter_count_note="B03 D block: original/shared882 or horizon42336; this fit original0/shared882/horizon42336; deployed includes retained PE34835 giving35717/77171" if run["run_id"].startswith("B03") else None,
                device=cost.get("device"), dtype=cost.get("dtype"), threads=cost.get("threads"), scope=cost.get("scope"),
                accuracy_device=run.get("device"), accuracy_dtype=run.get("dtype"),
                cost_join_strategy="Canonical method/cohort/C/L/H; supervised seed or deterministic evaluation reference; bound accuracy/archive when supplied" if cost else "NO_MEASURED_COST",
                accuracy_cost_scope=cost.get("accuracy_source", "Saved full V accuracy; timed first B with source parity") if cost else None,
                accuracy_status=run["status"], cost_status=cost.get("status", "NOT_MEASURED"), run_id=run["run_id"],
                source_sha256=run["source_sha256"], cost_source_sha256=cost.get("source_sha256")))
    hardware_path = report/"HARDWARE.json"
    hardware = read(hardware_path) if hardware_path.is_file() else {}
    if hardware_path.is_file():
        _source(hardware_path, ctx, data["bindings"])
    csv_write(report/"PPT_T1_ACCURACY_COST.csv", presentation.enrich_accuracy_cost_rows(ppt_accuracy, data, hardware))
    csv_write(report/"PPT_T2_MECHANISM_ABLATION.csv", presentation.mechanism_rows(matched+comparisons(data, contextual=True)+common_compared, data))
    csv_write(report/"PPT_T3_COVERAGE_CLAIMS.csv", presentation.coverage_rows(questions, data, audit))
    csv_write(report/"PPT_T4_DATASET_PROTOCOL.csv", presentation.dataset_protocol_rows(audit, data, ctx["repo"]))
    details = presentation.data_source_details(audit, data, ctx["repo"])
    for source in details["source_bindings"]:
        name = source.get("path", source.get("reference", "")).removeprefix("repo://")
        if name:
            data["bindings"][name] = source["sha256"]
    atomic_json(report/"DATA_SOURCE_DETAILS.json", details)
    atomic(report/"DATA_AUDIT_SUPPLEMENT_KO.md", "# 원 A01 감사의 출처·단위 보완\n\n원 DATA_AUDIT_KO.md와 DATA_AUDIT.json은 검증 당시의 receipt-bound 원본을 보존한다. 아래는 같은 자료의 추가 공개 메타데이터다.\n\n"+
           presentation.data_audit_supplement(audit, data, ctx["repo"]))
    return dict(scientific_status=scientific, all_job_statuses_terminal=closed, job_counts=dict(counts))


def public_reaggregate(report):
    grouped = defaultdict(lambda: defaultdict(lambda: [0., 0., 0]))
    for row in rows(Path(report)/"ERROR_BY_CHANNEL.csv"):
        cell = grouped[row["run_id"]][row["channel_id"]]
        cell[0] += float(row["SSE"]); cell[1] += float(row["SAE"]); cell[2] += int(row["count"])
    scores = {}
    for identifier, channels in grouped.items():
        active = [v for v in channels.values() if v[2] > 0]
        scores[identifier] = dict(MSE=math.fsum(v[0]/v[2] for v in active)/len(active),
                                  MAE=math.fsum(v[1]/v[2] for v in active)/len(active), channels=len(active))
    return scores


def horizon_bindings(report, statistics):
    by_run = {}
    fields = ("cohort", "method", "seed", "split", "metric_scale", "origins_sha256", "source_file", "source_sha256", "private_archive_sha256", "lineage")
    for row in statistics:
        descriptor = {key: row.get(key) for key in fields}
        if row["run_id"] in by_run and by_run[row["run_id"]] != descriptor:
            raise ValueError("Horizon sufficient-statistic provenance changes within run")
        by_run[row["run_id"]] = descriptor
    atomic_json(Path(report)/"HORIZON_STATS_BINDINGS.json", dict(
        statistic_file="ERROR_BY_HORIZON.csv", statistic_sha256=sha(Path(report)/"ERROR_BY_HORIZON.csv"),
        join_key="run_id", scope="All h/channel SSE, SAE and observed counts; repeated source provenance normalized by run_id", runs=by_run))


def public_horizon_consistency(report):
    report = Path(report)
    channel_rows = rows(report/"ERROR_BY_CHANNEL.csv")
    channels = {(r["run_id"], r["channel_id"]): r for r in channel_rows}
    if len(channels) != len(channel_rows):
        raise ValueError("Duplicate public channel sufficient statistics")
    grouped, positions, seen, marginal = defaultdict(list), defaultdict(set), set(), set()
    for row in rows(report/"ERROR_BY_HORIZON.csv"):
        if row["channel_id"] == "ALL_CHANNELS_EQUAL_COUNTS":
            marginal.add(row["run_id"])
            continue
        key = (row["run_id"], row["channel_id"])
        identity = (*key, int(row["horizon"]))
        if identity in seen:
            raise ValueError("Duplicate public horizon/channel sufficient statistic")
        seen.add(identity); grouped[key].append(row); positions[key].add(int(row["horizon"]))
    expected_h = {}
    for name in ("RUNS_FULL_PRECISION.csv", "SAME_UPDATE_RESULTS.csv"):
        if (report/name).is_file():
            expected_h.update({r["run_id"]: int(float(r["H"])) for r in rows(report/name) if number(r.get("H")) is not None})
    maximum, maximum_mean, discrepancies = 0., 0., []
    complete = {key[0] for key in grouped}
    for run in complete:
        if {key for key in channels if key[0] == run} != {key for key in grouped if key[0] == run}:
            raise ValueError("Public horizon/channel coverage differs")
    for key, group in grouped.items():
        if key not in channels:
            raise ValueError("Public horizon statistic has no channel marginal")
        expected = set(range(1, expected_h.get(key[0], max(positions[key]))+1))
        if positions[key] != expected:
            raise ValueError("Missing public horizon sufficient statistic")
        reference = channels[key]
        if sum(int(r["count"]) for r in group) != int(reference["count"]):
            raise ValueError("Public horizon/channel observed counts disagree")
        count = int(reference["count"])
        for field in ("SSE", "SAE"):
            total = math.fsum(float(r[field]) for r in group)
            discrepancy = abs(total-float(reference[field]))
            mean_difference = discrepancy/max(count, 1)
            maximum = max(maximum, discrepancy); maximum_mean = max(maximum_mean, mean_difference)
            discrepancies.append(dict(run_id=key[0], channel_id=key[1], statistic=field,
                horizon_sum=total, channel_marginal=float(reference[field]), raw_sum_absolute_difference=discrepancy,
                observed_count=count, channel_mean_absolute_difference=mean_difference))
            if mean_difference > 1e-10 or (count == 0 and discrepancy != 0):
                raise ValueError("Public horizon/channel SSE or SAE disagrees: "+key[0]+" channel="+key[1]+" "+field+
                                 " raw_difference="+repr(discrepancy)+" observed_mean_difference="+repr(mean_difference))
    return dict(checked_runs=sorted(complete), checked_channels=len(grouped), maximum_statistic_absolute_difference=maximum,
        maximum_channel_mean_absolute_difference=maximum_mean, channel_mean_tolerance=1e-10,
        tolerance_units="Per-channel observed mean SSE/count or SAE/count; exact integer counts and all H positions remain required. FP64 reduction order can change large unnormalized sums.",
        largest_statistic_discrepancies=sorted(discrepancies, key=lambda value: value["raw_sum_absolute_difference"], reverse=True)[:10],
        horizon_missing_runs=sorted({key[0] for key in channels}-complete-marginal), historical_equal_count_marginal_only=sorted(marginal))


def resume_evidence(ctx, plan):
    evidence = []
    for job in plan["jobs"]:
        if job["question_id"] not in ("C00", "C01"):
            continue
        path = Path(ctx["report"])/"receipts"/(job["job_id"]+".json")
        if not path.is_file():
            continue
        receipt = read(path)
        if receipt.get("status") not in SUCCESS:
            continue
        result = receipt.get("result", {})
        item = dict(job_id=job["job_id"], source=path.relative_to(ctx["report"]).as_posix(), sha256=sha(path), status=receipt["status"])
        if job["question_id"] == "C00":
            profiles = result.get("profiles", {})
            if isinstance(profiles, list):
                profiles = {value.get("family", str(index)): value for index, value in enumerate(profiles)}
            item["disposable_next_update_profiles"] = {family: {key: value.get(key) for key in
                ("next_update_branch_audit", "source_bundle_unchanged_after_both_profiles", "source_bundle_fingerprint_before_and_after", "global_CPU_and_CUDA_RNG_restored", "disposable_profile_only")}
                for family, value in profiles.items()}
        else:
            item.update({key: result.get(key) for key in ("exact_CPU_continuation", "continuation_kind", "historical_bitwise_equivalence_claimed", "original_partial_sha256", "original_resume_audit_sha256")})
            runtime = []
            for relative, expected in receipt.get("artifact_hashes", {}).items():
                if not relative.endswith("ACTUAL_BRANCH_RESTORE_CHECKS.json"):
                    continue
                restore_path = Path(ctx["report"])/relative
                if sha(restore_path) != expected:
                    raise ValueError("Continuation runtime restore evidence changed")
                runtime.append(dict(source=relative, sha256=expected, evidence=read(restore_path)))
            item["saved_branch_runtime_restore"] = runtime
        evidence.append(item)
    return evidence


def method_evidence(ctx, plan):
    report = Path(ctx["report"])
    evidence = []
    producer = "research/"+NAME+"/backbone_jobs.py"
    for job in plan["jobs"]:
        if job["question_id"] != "D01":
            continue
        receipt_path = report/"receipts"/(job["job_id"]+".json")
        if not receipt_path.is_file():
            evidence.append(dict(job_id=job["job_id"], status="NOT_RUN"))
            continue
        receipt = read(receipt_path)
        if receipt.get("status") not in SUCCESS:
            evidence.append(dict(job_id=job["job_id"], status=receipt.get("status"), reason=receipt.get("reason")))
            continue
        relative = "jobs/"+job["job_id"]+".json"
        path, expected = report/relative, receipt.get("artifact_hashes", {}).get(relative)
        if (not job.get("scientific_identity") or receipt.get("scientific_identity") != job["scientific_identity"]
                or expected is None or not path.is_file() or sha(path) != expected):
            evidence.append(dict(job_id=job["job_id"], status="REJECTED_PUBLIC_SOURCE_BINDING", source=relative))
            continue
        value = read(path)
        if value != receipt.get("result"):
            evidence.append(dict(job_id=job["job_id"], status="REJECTED_PUBLIC_SOURCE_BINDING", source=relative))
            continue
        native = value.get("routing") == "native_multivariate_per_origin"
        gradient = value.get("adapter_gradient", {})
        required_gradient = "NOT_APPLICABLE_NATIVE_RAW" if native else "PASS"
        passed = (value.get("status") == "PASS" and value.get("frozen_weights_unchanged") is True
                  and value.get("save_load_prediction_parity") is True and value.get("trainable_backbone_parameters") == 0
                  and value.get("routing_sentinel", {}).get("passed") is True and gradient.get("status") == required_gradient)
        item = {key: value.get(key) for key in ("backbone", "device", "dtype", "shape", "point_statistic", "median_index", "routing",
            "official_parity_max_abs", "mask_parity_max_abs", "tolerance", "routing_sentinel", "adapter_gradient", "backbone_weight_hash",
            "frozen_weights_unchanged", "trainable_backbone_parameters", "save_load_prediction_parity", "original_scaler_forward_parity", "cost_scope")}
        item.update(job_id=job["job_id"], status="PASS_APPLICABLE_RECORDED_SCOPE" if passed else "FAIL_RECORDED_D01_INVARIANT",
            receipt_source=receipt_path.relative_to(report).as_posix(), receipt_sha256=sha(receipt_path), source=relative, sha256=expected,
            scientific_identity=job["scientific_identity"], producer_source=producer, producer_source_sha256=job.get("scientific_source_hashes", {}).get(producer),
            private_binding_verification_gate="G7_CACHE", private_binding_count=len(value.get("private_artifacts", [])),
            binding_chain="Plan scientific identity -> complete receipt -> bound public producer JSON -> private_artifacts checked by G7; no model loaded here",
            optimizer_scope=dict(status="NOT_APPLICABLE_NATIVE_RAW" if native else "SOURCE_ASSERTION_NO_SEPARATE_RUNTIME_FIELD",
                detail="Native Raw has no adapter optimizer/gradient test; channel-isolation NA is retained" if native else
                "Producer asserts optimizer excludes frozen F before synthetic adapter steps; result has gradient steps but no separate optimizer-exclusion boolean"))
        evidence.append(item)
    if any(row["status"] == "REJECTED_PUBLIC_SOURCE_BINDING" for row in evidence):
        status = "FAIL_D01_PUBLIC_BINDING"
    elif any(row["status"] == "FAIL_RECORDED_D01_INVARIANT" for row in evidence):
        status = "FAIL_D01_RECORDED_INVARIANT"
    elif evidence and all(row["status"] == "PASS_APPLICABLE_RECORDED_SCOPE" for row in evidence):
        status = "PASS_RECORDED_D01_APPLICABLE_SCOPE"
    else:
        status = "PARTIAL_RECORDED_D01_SCOPE" if any(row["status"] == "PASS_APPLICABLE_RECORDED_SCOPE" for row in evidence) else "NOT_RUN_IN_REPORT_VERIFIER"
    return dict(gate="G2_METHOD", status=status, backbone_smoke=evidence, raw_or_model_loads=0,
        scope="Actual D01 producer records only: official/masked parity, routing, frozen weights and save/load. Moirai synthetic adapter gradient is applicable; native Chronos2 gradient/optimizer/channel-isolation NA stay NA. No Jena full-training/generalization or E2E-cost conclusion; private/source hashes follow G7.")


def verify(ctx, write=False):
    report, repo = Path(ctx["report"]), Path(ctx["repo"])
    problems, checks = [], []
    required = ["README_KO.md", "QUESTIONS_AND_DECISIONS_KO.md", "DATA_AUDIT_KO.md", "DATA_AUDIT.json", "METHOD_CONTRACT.md", "EXPERIMENT_PLAN.json", "JOB_STATUS.csv", "RUNS_FULL_PRECISION.csv", "SAME_UPDATE_RESULTS.csv", "SAME_UPDATE_COMPARISONS.csv", "MATCHED_COMPARISONS.csv", "CONTEXTUAL_COMPARISONS.csv", "COST_FULL_PRECISION.csv", "ERROR_BY_HORIZON.csv", "HORIZON_STATS_BINDINGS.json", "ERROR_BY_CHANNEL.csv", "REPRESENTATION_PROBES.csv", "INITIALIZATION_DIAGNOSTICS.csv", "SOURCE_BINDINGS.json", "NEW_STATS_REPLAY.json", "RESOURCE_LEDGER.json", "REPRODUCE_KO.md", "NEXT_STEPS_KO.md"]
    required += ["TRAJECTORY_DIAGNOSTICS.json", "HORIZON_BAND_SUMMARY.csv", "PAIRED_ORIGIN_COVERAGE.csv", "PAIRED_ORIGIN_BINDINGS.json",
                 "DATA_SOURCE_DETAILS.json", "DATA_AUDIT_SUPPLEMENT_KO.md", "PPT_T1_ACCURACY_COST.csv", "PPT_T2_MECHANISM_ABLATION.csv",
                 "PPT_T3_COVERAGE_CLAIMS.csv", "PPT_T4_DATASET_PROTOCOL.csv"]
    required += ["figures/"+figure+"."+extension for figure in FIGURES for extension in ("png", "svg")]
    for name in required:
        if not (report/name).is_file():
            problems.append("Missing required output: "+name)
    if problems:
        raise ValueError("; ".join(problems))
    plan = read(report/"EXPERIMENT_PLAN.json")
    validate_plan(plan, repo)
    status = rows(report/"JOB_STATUS.csv")
    if {j["job_id"] for j in plan["jobs"]} != {j["job_id"] for j in status} or len(status) != len(plan["jobs"]):
        problems.append("Job coverage differs from sealed plan")
    audit = read(report/"DATA_AUDIT.json")
    pre_t_ok = audit.get("status") == "PASS_WITH_M5_BLOCK"
    if not pre_t_ok:
        problems.append("Required pre-T data audit failed: "+str(audit.get("status")))
    checks.append(dict(gate="G1_DATA", status="PASS_PRE_T_SCOPE" if pre_t_ok else "FAIL_REQUIRED_PRE_T_AUDIT", source_status=audit.get("status"),
                       source_sha256=sha(report/"DATA_AUDIT.json"), scope="Actual pinned pre-T prefix boundaries/mask/zero/scaler plus synthetic fill; upstream ancestry/overlap remain UNKNOWN"))
    if (report/"M5_LOCAL_ASSET_DISCOVERY.json").is_file():
        try:
            local_m5 = report_helper(ctx, "m5_local_assets").verify_local_m5_assets(ctx)
            checks.append(dict(gate="G1_M5_LOCAL_ASSETS", status=local_m5["status"],
                source_sha256=sha(report/"M5_LOCAL_ASSET_DISCOVERY.json"),
                scope="Current local file bytes/header schema only; historical external HTTP401 retained; no M5 row values parsed, fitting or evaluation"))
        except (ValueError, FileNotFoundError) as error:
            problems.append("M5 local asset binding rejected: "+str(error))
            checks.append(dict(gate="G1_M5_LOCAL_ASSETS", status="FAIL", reason=str(error)))
    tests_path = report/"TEST_VALIDATION.json"
    tests = read(tests_path) if tests_path.is_file() else None
    method = method_evidence(ctx, plan)
    method["test_receipt"] = tests
    if method["status"].startswith("FAIL_"):
        problems.append("D01 method evidence binding/invariant rejected")
    checks.append(method)
    comparisons_rows = rows(report/"MATCHED_COMPARISONS.csv")
    contextual_rows = rows(report/"CONTEXTUAL_COMPARISONS.csv")
    common_rows = rows(report/"SAME_UPDATE_COMPARISONS.csv")
    expected_common = same_update_comparisons(dict(endpoints=rows(report/"SAME_UPDATE_RESULTS.csv")))
    expected_by_pair = {(row["reference_run_id"], row["method_run_id"]): row for row in expected_common}
    if len(common_rows) != len(expected_common) or {(row["reference_run_id"], row["method_run_id"]) for row in common_rows} != set(expected_by_pair):
        problems.append("Common endpoint registered comparison coverage differs")
    for row in common_rows:
        expected = expected_by_pair.get((row["reference_run_id"], row["method_run_id"]))
        if expected is None:
            continue
        if row["comparison_scope"] != "COMMON_SEARCH_ENDPOINT" or row["match_status"] != expected["match_status"]:
            problems.append("Common endpoint contract/scope status differs")
        for field in ("epoch", "updates", "exposures", "delta_MSE", "delta_MAE", "delta_MSE_pct", "delta_MAE_pct"):
            actual, intended = number(row[field]), number(expected[field])
            if (actual is None) != (intended is None) or (actual is not None and abs(actual-intended) > 1e-12):
                problems.append("Common endpoint comparison arithmetic differs")
    exact = sum(r.get("common_contract_hash_status") == "ALL_FIVE_EXACT" for r in comparisons_rows)
    checks.append(dict(gate="G3_PAIR", status="PASS_EXACT_SUBSET_WITH_CONTEXTUAL_LIMITS" if exact else "NO_EXACT_PAIRS_VERIFIED", exact_hash_comparisons=exact,
        contextual_comparisons=len(contextual_rows), common_search_endpoint_rows=len(common_rows),
        common_search_endpoint_exact=sum(row["match_status"] == "EXACT_MATCHED" for row in common_rows),
        common_search_endpoint_contextual=sum(row["match_status"] != "EXACT_MATCHED" for row in common_rows),
        common_progress_identity="Same cohort/seed/split/metric_scale/epoch/updates/exposures; all canonical contract fields checked separately; source-derived arithmetic rechecked",
        scope="Exact subset requires C/L/H, cohort/split/scale/point/device/backend/dtype plus identical canonical scaler/target/mask/origin/channel hashes. B01 canonical target hashes identify reference FP32 bytes after exact saved-FP64-value equality; raw saved dtype/hash and derived channel order are explicit in NEW_STATS_REPLAY. Common search endpoints are separate from selected-refit/fixed diagnostic pairs. Contextual differences do not pass this gate; sampler order/select-refit evidence remains in corresponding jobs"))
    checks.append(dict(gate="G4_SOLVER", status="PASS_REGISTERED_SYNTHETIC_CASE" if audit.get("synthetic_G1_G4", {}).get("status") == "PASS" else "NOT_VERIFIED",
        evidence=audit.get("synthetic_G1_G4"), scope="Independent masked augmented-lstsq, KKT, intercept/save-load and identical shared/replicated Dh objective; real decoder evidence is separate"))
    restore = repo/"reports"/OLD/"diagnostics/PEMS_RESTORE_AUDIT_RESULT.json"
    runtime_resume = resume_evidence(ctx, plan)
    checks.append(dict(gate="G5_RESUME", status="CURRENT_RUNTIME_AND_HISTORICAL_EVIDENCE" if runtime_resume else "EXISTING_RESTORE_EVIDENCE" if restore.is_file() else "NOT_RUN",
        source_sha256=sha(restore) if restore.is_file() else None,
        current_runtime=runtime_resume,
        scope="Historical exact CPU restore, disposable common-state CPU/CUDA next-update numerical audit, and completed C01 CPU or CUDA branch continuation are separate evidence; CUDA branch does not claim historical CPU bitwise equality"))
    checks.append(dict(gate="G8_JOB_COVERAGE", checked=len(status), status="PASS" if not problems else "FAIL"))
    bindings = read(report/"SOURCE_BINDINGS.json")["public_inputs"]
    for name, expected in bindings.items():
        path = (repo/name).resolve()
        if not path.is_relative_to(repo.resolve()) or not path.is_file() or sha(path) != expected:
            problems.append("Public source binding changed: "+name)
    scores = public_reaggregate(report)
    h_bindings = read(report/"HORIZON_STATS_BINDINGS.json")
    h_stats = rows(report/"ERROR_BY_HORIZON.csv")
    if h_bindings["statistic_sha256"] != sha(report/"ERROR_BY_HORIZON.csv") or {row["run_id"] for row in h_stats} != set(h_bindings["runs"]):
        problems.append("Compact horizon CSV binding coverage/SHA mismatch")
    for descriptor in h_bindings["runs"].values():
        if bindings.get(descriptor["source_file"]) != descriptor["source_sha256"]:
            problems.append("Compact horizon provenance absent from public source graph")
    try:
        horizon_check = public_horizon_consistency(report)
    except ValueError as error:
        problems.append(str(error))
        horizon_check = dict(status="FAIL", reason=str(error))
    matched, uncovered = [], []
    score_rows = rows(report/"RUNS_FULL_PRECISION.csv")+rows(report/"SAME_UPDATE_RESULTS.csv")
    for run in score_rows:
        if number(run.get("MSE")) is None or number(run.get("MAE")) is None:
            continue
        if run["run_id"] not in scores:
            uncovered.append(dict(run_id=run["run_id"], scope="COMMON_UPDATE_ENDPOINT" if run.get("refit") == "NOT_APPLICABLE_SEARCH_ENDPOINT" else "SELECTED_OR_FIXED_RUN",
                                  reason="No public per-channel SSE/SAE/count"))
            continue
        independent = scores[run["run_id"]]
        discrepancy = max(abs(independent[key]-float(run[key])) for key in ("MSE", "MAE"))
        if discrepancy > 1e-10:
            problems.append("Independent public aggregate mismatch: "+run["run_id"])
        matched.append(dict(run_id=run["run_id"], maximum_absolute_difference=discrepancy))
    checks.append(dict(gate="G6_PUBLIC_FP64_REAGGREGATION", status="FAIL" if horizon_check.get("status") == "FAIL" or any("aggregate mismatch" in p for p in problems) else "PASS_ALL_SCORED_ROWS" if matched and not uncovered else "PARTIAL_COVERAGE" if matched else "NOT_VERIFIED", rows=matched,
                       uncovered=uncovered, checked_scored_rows=len(matched)+len(uncovered),
                       horizon_channel_consistency=horizon_check,
                       scope="Only runs with public observed per-channel sufficient statistics; absence is not a score verification"))
    from .core import reusable
    private_roots = {key: ctx[key] for key in ("repo", "report", "asset", "cache") if key in ctx}
    cache_checks = []
    for job in plan["jobs"]:
        receipt = report/"receipts"/(job["job_id"]+".json")
        if receipt.is_file() and read(receipt).get("status") in SUCCESS:
            okay = reusable(receipt, job["scientific_identity"], report, private_roots=private_roots)
            cache_checks.append(dict(job_id=job["job_id"], all_bound_artifacts_verified=okay))
            if not okay:
                problems.append("Completed job source/private artifact binding rejected: "+job["job_id"])
    replay_check = replay_binding_validation(ctx)
    if not replay_check["all_bound_artifacts_verified"]:
        problems.append("Late-bound saved prediction/source artifact graph rejected")
    checks.append(dict(gate="G7_CACHE", status="FAIL_BOUND_ARTIFACTS" if not replay_check["all_bound_artifacts_verified"] or any(not row["all_bound_artifacts_verified"] for row in cache_checks) else "PASS_CURRENT_BOUND_ARTIFACTS" if cache_checks else "NO_COMPLETED_BOUND_ARTIFACTS",
                       checked=cache_checks, saved_replay_graph=replay_check,
                       scope="Actual complete receipt identity and bound public/private hashes plus verification-time B01 late bindings. Late capture does not prove historical prediction immutability; subsequent mutation is rejected."))
    try:
        supplement_check = report_helper(ctx, "verify_report_supplements").verify_supplements(ctx)
        checks.append(supplement_check)
    except ValueError as error:
        problems.append("Independent supplement verification failed: "+str(error))
        checks.append(dict(gate="G8_INDEPENDENT_REPORT_SUPPLEMENTS", status="FAIL", reason=str(error)))
    for row in rows(report/"COST_FULL_PRECISION.csv"):
        if row.get("median_ms_per_origin") and float(row["median_ms_per_origin"]) <= 0:
            problems.append("Nonpositive measured latency")
    forbidden = {".npz", ".npy", ".pt", ".safetensors", ".pkl", ".zip", ".log"}
    payload = [p for p in report.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    if any(p.suffix.lower() in forbidden for p in payload):
        problems.append("Forbidden raw/private/binary payload type")
    if any(p.stat().st_size > 20*1024**2 for p in payload) or sum(p.stat().st_size for p in payload) > 250*1024**2:
        problems.append("Publication size limit exceeded")
    checks.append(dict(gate="G9_TYPES_AND_SIZE", status="PASS" if not any("payload" in p or "size limit" in p for p in problems) else "FAIL", public_files=len(payload), bytes=sum(p.stat().st_size for p in payload)))
    if not write and (report/"CHECKSUMS.sha256").is_file():
        for line in (report/"CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = (repo/relative).resolve()
            if not path.is_relative_to(repo.resolve()) or not path.is_file() or sha(path) != expected:
                problems.append("Final package checksum mismatch: "+relative)
    result = dict(status="PASS" if not problems else "FAIL", checked_utc=utc(), validator_command="python -m research.adapter_diagnostics_20261017_v2.run --mode verify --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE",
                  exit_code=0 if not problems else 2, checked_plan_sha256=plan["plan_sha256"], checked_source_sha256=sha(__file__),
                  checks=checks, problems=problems, raw_or_model_loads=0, independent_public_reaggregation_runs=len(matched),
                  publication_note="This validation is not a remote main push receipt or official staged payload checker")
    if write:
        atomic_json(report/"VALIDATION.json", result)
    print(json.dumps(dict(status=result["status"], public_reaggregation_runs=len(matched), jobs=len(status))))
    if problems:
        raise ValueError("; ".join(problems))
    return result


def manifest(ctx):
    report, repo = Path(ctx["report"]), Path(ctx["repo"])
    excluded = {"PUBLICATION_MANIFEST.json", "CHECKSUMS.sha256"}
    paths = sorted(p for p in report.rglob("*") if p.is_file() and p.name not in excluded and "__pycache__" not in p.parts)
    payload = [dict(path=p.relative_to(repo).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
    atomic_json(report/"PUBLICATION_MANIFEST.json", dict(schema="PUBLIC_ADAPTER_PACKAGE_v1", files=payload,
        self_reference="Manifest does not hash itself or CHECKSUMS; CHECKSUMS hashes manifest and all other payload, excluding itself",
        allowed_raw_arrays=False, qualitative_numeric_source_public=False))
    paths.append(report/"PUBLICATION_MANIFEST.json")
    atomic(report/"CHECKSUMS.sha256", "".join(sha(p)+"  "+p.relative_to(repo).as_posix()+"\n" for p in sorted(paths)))


def build_report(ctx):
    report, repo = Path(ctx["report"]), Path(ctx["repo"])
    data = collect(ctx)
    if data.get("trajectory"):
        atomic_json(report/"TRAJECTORY_DIAGNOSTICS.json", data["trajectory"])
        for run in data["runs"]:
            if run.get("fit_score_source") == "TRAJECTORY_DIAGNOSTICS.json":
                run["fit_score_source_sha256"] = sha(report/"TRAJECTORY_DIAGNOSTICS.json")
        _source(report/"TRAJECTORY_DIAGNOSTICS.json", ctx, data["bindings"])
    matched = comparisons(data)
    csv_write(report/"JOB_STATUS.csv", data["jobs"])
    csv_write(report/"RUNS_FULL_PRECISION.csv", data["runs"], RUN_FIELDS)
    csv_write(report/"RUNS_ALL.csv", data["runs"], RUN_FIELDS)
    csv_write(report/"SAME_UPDATE_RESULTS.csv", data["endpoints"])
    csv_write(report/"SAME_UPDATE_COMPARISONS.csv", same_update_comparisons(data))
    csv_write(report/"MATCHED_COMPARISONS.csv", matched)
    csv_write(report/"CONTEXTUAL_COMPARISONS.csv", comparisons(data, contextual=True))
    csv_write(report/"COST_FULL_PRECISION.csv", data["costs"])
    csv_write(report/"ERROR_BY_HORIZON.csv", data["horizon"], ["run_id", "horizon", "channel_id", "SSE", "SAE", "count"])
    horizon_bindings(report, data["horizon"])
    csv_write(report/"ERROR_BY_CHANNEL.csv", data["channel"])
    csv_write(report/"REPRESENTATION_PROBES.csv", data["probes"])
    csv_write(report/"INITIALIZATION_DIAGNOSTICS.csv", data["initializations"])
    csv_write(report/"INITIALIZATION_CURVES.csv", data["initialization_curves"])
    atomic_json(report/"LEGACY_STATS_REPLAY.json", dict(rows=data["legacy_stats_replay"], model_calls=0, protected_T_access=False))
    helper = report_helper(ctx, "report_supplements", data["bindings"])
    supplement = helper.build_supplements(ctx, data, matched+comparisons(data, contextual=True)+same_update_comparisons(data))
    for path in supplement["public_files"]:
        _source(path, ctx, data["bindings"])
    atomic_json(report/"NEW_STATS_REPLAY.json", dict(rows=data.get("new_stats_replay", []),
        trajectory=data.get("trajectory"), report_supplements=supplement["binding"],
        source_derived_cost_metadata=data.get("pems32_cost_metadata"), m5_local_assets=data.get("m5_local_assets"),
        model_calls=0, protected_T_access=False))
    state = _documents(ctx, data, matched)
    for name in ("TRAJECTORY_ENVIRONMENT_CHECK.json", "TRAJECTORY_ENVIRONMENT_KO.md"):
        if (report/name).is_file():
            _source(report/name, ctx, data["bindings"])
    report_helper(ctx, "verify_report_supplements", data["bindings"])
    figure_metadata = figures(ctx, data)
    atomic_json(report/"FIGURE_METADATA.json", figure_metadata)
    ledger_path = Path(ctx["cache"])/"CAMPAIGN.json"
    ledger = read(ledger_path) if ledger_path.is_file() else {}
    ledger["snapshot_utc"] = utc()
    ledger["elapsed_wall_seconds"] = time.time()-ledger["started_unix"] if "started_unix" in ledger else None
    atomic_json(report/"RESOURCE_LEDGER.json", ledger)
    atomic_json(report/"SOURCE_BINDINGS.json", dict(schema="ADAPTER_REPORT_BINDINGS_v1", public_inputs=data["bindings"],
        report_source_sha256=sha(__file__), state=state,
        qualitative_source_scope="Private archive hashes and fixed selection only in FIGURE_METADATA; no curve values in public metadata"))
    verify(ctx, write=True)
    manifest(ctx)
    return state


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaggregate", type=Path)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args(argv)
    if args.reaggregate:
        print(json.dumps(public_reaggregate(args.reaggregate), indent=2, allow_nan=False))
    elif args.figures:
        report = args.figures.resolve()
        repo = Path(__file__).resolve().parents[2]
        data = {"bindings": {}}
        for key, filename in (("runs", "RUNS_FULL_PRECISION.csv"), ("endpoints", "SAME_UPDATE_RESULTS.csv"),
                              ("costs", "COST_FULL_PRECISION.csv"), ("jobs", "JOB_STATUS.csv"),
                              ("initialization_curves", "INITIALIZATION_CURVES.csv"),
                              ("horizon", "ERROR_BY_HORIZON.csv"), ("channel", "ERROR_BY_CHANNEL.csv")):
            data[key] = rows(report/filename)
            for entry in data[key]:
                for name in ("seed", "MSE", "MAE", "C", "K", "L", "H", "updates", "exposures", "B", "median_ms_per_origin", "supervision_seed", "evaluation_reference_seed", "accuracy_MSE", "accuracy_MAE", "horizon", "SSE", "SAE", "count"):
                    if name in entry:
                        entry[name] = number(entry[name])
                if "exclusive" in entry:
                    entry["exclusive"] = entry["exclusive"] == "True"
        metadata = figures(dict(repo=repo, report=report), data, private_qualitative=False)
        print(json.dumps(dict(status="QUANTITATIVE_FIGURES_RECREATED", figures=list(metadata), private_arrays_read=False)))
    else:
        print("SOURCE_ONLY: use the campaign --mode report interface; no inputs opened")


if __name__ == "__main__":
    main()
