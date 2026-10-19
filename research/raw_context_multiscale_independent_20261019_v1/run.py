"""Independent terminal-gated saved-array and cost-arithmetic audit.

Only stdlib/NumPy; no model imports, weights, inference, training or reselection.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np

CAMPAIGN = "raw_context_multiscale_20261018_v1"
SEEDS = (92601, 92602)
METRIC_ATOL, STATS_ATOL, STATS_RTOL = 1e-12, 1e-12, 1e-13
DEPLOY_ATOL, DEPLOY_RTOL = 1e-5, 1e-4
F_SHA = "c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
ARRAY_KEYS = ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")
EXPECTED = {
    "jena": dict(C=21, L=512, H=48, K=12, A_fit_end=10482,
                 A_stop=(10482, 12579), V_select=(14675, 16772),
                 V_sha256="6d6bc4124e0d770ceae0819bbd59b36ca3750ff610aed7d26dbfe5c1a9e775ce"),
    "pems32": dict(C=32, L=512, H=12, K=8, A_fit_end=8928,
                   A_stop=(8928, 10713), V_select=(12499, 14284),
                   V_sha256="9cf012ce9e2f496dd3e10056955bc2f79623ee65e7a1d1be52a98f1e455fbf4f"),
}


class AuditFailure(Exception):
    def __init__(self, code, **details):
        super().__init__(code)
        self.code, self.details = code, details


def require(condition, code, **details):
    if not condition:
        raise AuditFailure(code, **details)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def ref(repo, path):
    return dict(reference="repo://" + Path(path).resolve().relative_to(Path(repo).resolve()).as_posix(),
                sha256=sha(path))


def resolve(repo, binding):
    roots = dict(repo=Path(repo), report=Path(repo)/"reports"/CAMPAIGN,
                 cache=Path(repo)/".cache"/CAMPAIGN)
    kind, relative = binding["reference"].split("://", 1)
    require(kind in roots, "UNSUPPORTED_BINDING_ROOT", root=kind)
    root = roots[kind].resolve()
    path = (root/relative).resolve()
    require(path.is_relative_to(root), "BINDING_ESCAPES_ROOT")
    require(path.is_file() and sha(path) == binding["sha256"], "BOUND_SHA_MISMATCH",
            reference=binding["reference"])
    return path


def close(actual, expected, code, *, atol=METRIC_ATOL, rtol=0., **scope):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol), code, **scope)


def array_equal(actual, expected, code, **scope):
    require(np.asarray(actual).shape == np.asarray(expected).shape
            and np.array_equal(actual, expected), code, **scope)


def origins(dataset, split):
    task = EXPECTED[dataset]
    lo, hi = task[split]
    return np.asarray(list(range(max(lo, task["L"]), hi-task["H"]+1, task["H"])), dtype=np.int64)


def numerical_gate(repo):
    """This precedes SCORES, prediction NPZs, and cost-trial numerical access."""
    report, cache = Path(repo)/"reports"/CAMPAIGN, Path(repo)/".cache"/CAMPAIGN
    terminal_path = cache/"TERMINAL_CORE_002.json"
    try:
        terminal = read_json(terminal_path)
    except FileNotFoundError:
        raise AuditFailure("GATE_MAIN_TERMINAL_MISSING") from None
    require(terminal.get("status") == "COMPLETE" and terminal.get("exit_code") == 0,
            "GATE_MAIN_TERMINAL_NOT_COMPLETE")
    validation_path = report/"VALIDATION_REPORT.json"
    try:
        validation = read_json(validation_path)
    except FileNotFoundError:
        raise AuditFailure("GATE_PRIMARY_VALIDATION_MISSING") from None
    require(validation.get("status") == "PASS", "GATE_PRIMARY_VALIDATION_NOT_PASS")
    bindings = validation.get("manifest_bindings", {})
    for name in ("SCORES.json", "COST_INDEX.json", "MODELS_SEALED_BEFORE_V.json"):
        require(name in bindings, "GATE_PRIMARY_BINDING_MISSING", manifest=name)
        require(resolve(repo, bindings[name]) == (report/name).resolve(), "GATE_PRIMARY_PATH_MISMATCH",
                manifest=name)
    plan_path = resolve(repo, bindings["registered_plan"])
    plan = read_json(plan_path)
    require(len(plan.get("source_files", {})) == 13, "GATE_PRIMARY_SOURCE_COUNT")
    for relative, digest in plan["source_files"].items():
        source = (Path(repo)/relative).resolve()
        require(source.is_relative_to(Path(repo).resolve()) and sha(source) == digest,
                "GATE_PRIMARY_SOURCE_CHANGED", source=relative)
    return dict(terminal=ref(repo, terminal_path), primary_validation=ref(repo, validation_path),
                registered_plan=ref(repo, plan_path), validated_models=validation.get("models"),
                validated_cost_facets=validation.get("cost_facets"))


def statistics(prediction, target, observed):
    """Independent math.fsum FP64 by horizon/channel, then equal-channel macro."""
    p, y, mask = map(np.asarray, (prediction, target, observed))
    require(p.ndim == 3 and p.shape == y.shape == mask.shape, "ARRAY_SHAPE")
    require(p.dtype == y.dtype == np.dtype("float32"), "FP32_PREDICTION_TARGET_REQUIRED")
    require(mask.dtype == np.bool_, "BOOL_MASK_REQUIRED")
    require(np.isfinite(p).all() and np.isfinite(y[mask]).all(), "NONFINITE_PREDICTION_OR_OBSERVED_TARGET")
    h, c = p.shape[1:]
    sse, sae, counts = [], [], []
    for horizon in range(h):
        squares, absolutes, count = [], [], []
        for channel in range(c):
            observed = mask[:, horizon, channel]
            error = p[observed, horizon, channel].astype(np.float64)-y[observed, horizon, channel].astype(np.float64)
            squares.append(math.fsum(float(x)*float(x) for x in error))
            absolutes.append(math.fsum(abs(float(x)) for x in error))
            count.append(int(observed.sum()))
        sse.append(squares); sae.append(absolutes); counts.append(count)
    cc = [sum(row[k] for row in counts) for k in range(c)]
    cs = [math.fsum(row[k] for row in sse) for k in range(c)]
    ca = [math.fsum(row[k] for row in sae) for k in range(c)]
    active = [k for k in range(c) if cc[k]]
    require(bool(active), "NO_OBSERVED_TARGET")
    return dict(mse=math.fsum(cs[k]/cc[k] for k in active)/len(active),
                mae=math.fsum(ca[k]/cc[k] for k in active)/len(active),
                n_origins=len(p), n_observed=sum(cc), horizon=h, channels=c,
                channel_counts=cc, channel_sse=cs, channel_sae=ca,
                horizon_channel_sse=sse, horizon_channel_sae=sae, horizon_channel_count=counts)


def compare_stats(actual, saved, run_id):
    for name in ("mse", "mae", "n_origins", "n_observed", "horizon", "channels"):
        if name in saved:
            close(actual[name], saved[name], "SAVED_SCORE_MISMATCH", run_id=run_id, field=name)
    for name in ("channel_counts", "horizon_channel_count"):
        if name in saved:
            array_equal(actual[name], saved[name], "SAVED_COUNT_MISMATCH", run_id=run_id, field=name)
    for name in ("channel_sse", "channel_sae", "horizon_channel_sse", "horizon_channel_sae"):
        if name in saved:
            a, b = np.asarray(actual[name]), np.asarray(saved[name])
            require(a.shape == b.shape and np.allclose(a, b, atol=STATS_ATOL, rtol=STATS_RTOL),
                    "SAVED_SUFFICIENT_STATISTIC_MISMATCH", run_id=run_id, field=name)


def archive_values(path_or_stream):
    with np.load(path_or_stream, allow_pickle=False) as archive:
        require(set(ARRAY_KEYS).issubset(archive.files) and "metadata_json" in archive.files, "NPZ_REQUIRED_KEYS")
        values = {name:archive[name] for name in ARRAY_KEYS}
        metadata = json.loads(str(archive["metadata_json"].item()))
    return values, metadata


def check_geometry(values, dataset, split, columns, run_id):
    task = EXPECTED[dataset]
    require(values["origins"].dtype == np.dtype("int64"), "ORIGIN_INT64_REQUIRED", run_id=run_id)
    array_equal(values["origins"], origins(dataset, split), "CANONICAL_INTEGER_ORIGIN_MISMATCH", run_id=run_id)
    require(values["pred_normalized"].shape == (len(origins(dataset, split)), task["H"], task["C"]),
            "ORIGINAL_H_C_GEOMETRY_MISMATCH", run_id=run_id)
    array_equal(values["channel_ids"], np.asarray(columns), "CHANNEL_ORDER_MISMATCH", run_id=run_id)
    if split == "V_select":
        require(identity(values["origins"])["sha256"] == task["V_sha256"],
                "LEGACY_ORIGIN_IDENTITY_MISMATCH", run_id=run_id)


def compare_cohort(shared, values, metadata, dataset, split, run_id):
    key, fields = (dataset, split), ("target_normalized", "observed", "origins", "channel_ids")
    if key not in shared:
        shared[key] = ({name:values[name].copy() for name in fields}, metadata.get("scaler_sha256"))
    else:
        reference, scaler = shared[key]
        for name in fields:
            require(values[name].dtype == reference[name].dtype, "COHORT_DTYPE_MISMATCH", run_id=run_id, field=name)
            array_equal(values[name], reference[name], "DATASET_TARGET_MASK_ORIGIN_CHANNEL_MISMATCH", run_id=run_id, field=name)
        require(metadata.get("scaler_sha256") == scaler, "DATASET_SCALER_IDENTITY_MISMATCH", run_id=run_id)


def legacy_origins(repo):
    result = {}
    proof_path = Path(repo)/"reports/b2_revin_20261018_v1/PREDICTION_PROOF.json"
    proof = read_json(proof_path)
    validation = read_json(Path(repo)/"reports/b2_revin_20261018_v1/VALIDATION_REPORT.json")
    require(proof.get("status") == "PASS" and sha(proof_path) == validation["prediction_proof"]["sha256"],
            "LEGACY_PUBLIC_PROOF_BINDING_MISMATCH")
    for dataset in EXPECTED:
        expected, sources = identity(origins(dataset, "V_select")), []
        for seed in SEEDS:
            if dataset == "jena":
                source = Path(repo)/("reports/memory_budget_20261018_v1/receipts/"
                                     f"B2_jena_K12_OFF_s{seed}_cap3_B1_R12_confirm.receipt.json")
                receipt = read_json(source)
                require(receipt.get("status") == "COMPLETE" and receipt["parity"]["status"] == "PASS",
                        "LEGACY_JENA_RECEIPT_INCOMPLETE", seed=seed)
                actual = receipt["parity"]["input_fingerprints"]["origins"]
            else:
                source = proof_path
                rid = f"B2_pems32_REVIN_OFF_s{seed}_cap6"
                rows = [row for row in proof["rows"] if row["run_id"] == rid]
                require(len(rows) == 1 and rows[0]["exact_actual_target_mask_origin_channel"] is True,
                        "LEGACY_PEMS_CANONICAL_ORIGIN_PROOF_MISSING", seed=seed)
                actual = rows[0]["arrays"]["origins"]
            require(actual == expected, "LEGACY_TWO_SEED_ORIGIN_IDENTITY_MISMATCH", dataset=dataset, seed=seed)
            sources.append(dict(seed=seed, binding=ref(repo, source), origin_identity=actual))
        result[dataset] = dict(origin_identity=expected, first=int(origins(dataset,"V_select")[0]),
                               last=int(origins(dataset,"V_select")[-1]), stride=EXPECTED[dataset]["H"],
                               two_seed_public_evidence=sources)
    return result


def validate_coverage(arms, selected_mode):
    require(selected_mode in ("A", "B"), "TRANSFER_FAMILY_INVALID")
    require(len(arms) == len({a["run_id"] for a in arms}) == 56, "EXACT_56_UNIQUE_ARMS_REQUIRED")
    actual = {(a["dataset"],a["mode"],int(a["seed"])) for a in arms if a["mode"] in ("S","A","B")}
    expected = {(d,m,s) for d,modes in (("jena",("S","A","B")),("pems32",("S",selected_mode))) for m in modes for s in SEEDS}
    require(actual == expected, "EXACT_TEN_BODY_CONFIGURATIONS_REQUIRED")
    actual = {(a["dataset"],int(a["seed"]),a["base_role"],a["alpha_label"]) for a in arms if a["mode"] == "alpha"}
    expected = {(d,s,r,l) for d in EXPECTED for s in SEEDS for r in ("Raw","G") for l in ("alpha0","alpha05","alpha1","fitted")}
    require(actual == expected and sum(a["mode"] == "alpha" for a in arms) == 32, "EXACT_32_ALPHA_ROUTES_REQUIRED")
    actual = {(a["dataset"],a["mode"],int(a["seed"])) for a in arms if a["mode"] in ("conditional","past_control")}
    expected = {(d,m,s) for d in EXPECTED for m in ("conditional","past_control") for s in SEEDS}
    require(actual == expected and sum(a["mode"] in ("conditional","past_control") for a in arms) == 8,
            "EXACT_EIGHT_HEAD_ROUTES_REQUIRED")
    actual = {(a["dataset"],a["mode"]) for a in arms if a["mode"] not in ("S","A","B","alpha","conditional","past_control")}
    require(actual == {("jena","Raw"),("jena","G"),("jena","LoRA"),("jena","Ridge"),("pems32","Raw"),("pems32","G")},
            "EXACT_SIX_BASELINES_REQUIRED")


def audit_selection(arms,transfer):
    selected={}
    for arm in arms:
        if arm["mode"] not in ("S","A","B"):
            continue
        values=[(float(p["astop_mse"]),float(p["epoch"])) for p in arm["curves"]]
        require(values and any(epoch == 0 for _,epoch in values)
                and len(values) == len({epoch for _,epoch in values})
                and all(math.isfinite(score) and math.isfinite(epoch) for score,epoch in values),
                "A_STOP_FINITE_UNIQUE_CURVE_INCLUDING_CP0_REQUIRED",run_id=arm["run_id"])
        score,epoch=min(values)
        close(score,arm["search_summary"]["selected_astop_mse"],"A_STOP_EARLIEST_MIN_SCORE_MISMATCH",run_id=arm["run_id"])
        close(epoch,arm["search_summary"]["selected_epoch"],"A_STOP_EARLIEST_MIN_EPOCH_MISMATCH",run_id=arm["run_id"])
        selected[arm["run_id"]]=dict(selected_epoch=epoch,selected_A_stop_mse=score)
    means={mode:math.fsum(selected[a["run_id"]]["selected_A_stop_mse"] for a in arms
                         if a["dataset"] == "jena" and a["mode"] == mode)/2 for mode in ("A","B")}
    require(means == transfer["mean_astop_mse"] and transfer["selected_mode"] == ("A" if means["A"] <= means["B"] else "B"),
            "A_STOP_TWO_SEED_TRANSFER_EXACT_TIE_A_MISMATCH")
    return dict(status="PASS",source_split="A_stop",exact_tie="A",selected_mode=transfer["selected_mode"],
                two_seed_mean_A_stop_mse=means,selected_body_checkpoints=selected)


def quantile_linear(values, q):
    ordered = sorted(float(x) for x in values)
    require(bool(ordered) and 0 <= q <= 1, "INVALID_QUANTILE_INPUT")
    position = (len(ordered)-1)*q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)


def expected_invocations(spec, arm, batch, schedule="native", nested=False):
    task, family = EXPECTED[arm["dataset"]], spec["family"]
    if family == "COMBINED":
        operation, alpha = spec["operation"], float(spec.get("alpha", .5))
        require(operation in ("alpha","conditional","past_control"), "UNKNOWN_COMBINATION_OPERATION")
        require(operation != "alpha" or 0 <= alpha <= 1, "INVALID_GLOBAL_ALPHA")
        parents = []
        if operation != "alpha" or alpha != 1:
            parents.append(spec["g_route"])
        if operation == "conditional" or (operation == "alpha" and alpha != 0):
            parents.append(spec["b_route"])
        return [inv for parent in parents for inv in expected_invocations(parent,arm,batch,nested=True)]
    if family in ("RAW","LORA"):
        widths, lengths, horizons = [task["C"]], [task["L"]], [task["H"]]
    elif family == "B2":
        mode, k = ("S" if nested else arm["mode"]), int(spec["latent_k"])
        separate = mode == "B" or (mode == "A" and schedule == "two_call")
        scale = 2 if mode == "B" else 1
        widths = [k//2,k//2] if separate else [k]
        lengths = [task["L"],task["L"]//scale] if separate else [task["L"]]
        horizons = [task["H"],task["H"]//scale] if separate else [task["H"]]
    else:
        require(family in ("DIRECT_G","RAW_RIDGE"), "UNKNOWN_DEPLOYMENT_FAMILY", family=family)
        return []
    return [dict(input_shape=[batch,length,width], requested_context=length, actual_context=length,
                 used_horizon=horizon, wrapper_horizon=task["H"], F_logical_streams=batch*width,
                 F_physical_streams=batch*width, F_calls=math.ceil(batch*width/128))
            for width,length,horizon in zip(widths,lengths,horizons)]


def check_work(work, expected, cost_id):
    require(len(work["F_invocations"]) == len(expected), "F_INVOCATION_COUNT_MISMATCH", cost_id=cost_id)
    for measured, registered in zip(work["F_invocations"],expected):
        for key,value in registered.items():
            require(measured.get(key) == value, "F_GEOMETRY_OR_CALL_MISMATCH", cost_id=cost_id, field=key)
    for key in ("F_logical_streams","F_physical_streams","F_calls"):
        require(work[key] == sum(x[key] for x in expected), "F_WORK_TOTAL_MISMATCH", cost_id=cost_id, field=key)


def compare_trial_csv(text, trials, cost_id):
    raw = list(csv.DictReader(io.StringIO(text)))
    require(len(raw) == len(trials), "RAW_TRIAL_CSV_COUNT_MISMATCH", cost_id=cost_id)
    for actual,saved in zip(raw,trials):
        require(set(actual) == set(saved), "RAW_TRIAL_CSV_SCHEMA_MISMATCH", cost_id=cost_id)
        for key,value in saved.items():
            expected = json.dumps(value,separators=(",",":")) if isinstance(value,(list,dict)) else "" if value is None else str(value)
            require(actual[key] == expected, "RAW_TRIAL_CSV_VALUE_MISMATCH", cost_id=cost_id, field=key)


def audit_cost(receipt, arm, facet, *, raw_csv_text, prediction_identity, used_pids):
    row,trials,parity = receipt["row"],receipt["trials"],receipt["parity"]
    cost_id,batch,schedule = facet["cost_id"],int(facet["batch"]),facet["schedule"]
    require(receipt.get("status") == row.get("status") == "COMPLETE" and row.get("cost_valid") is True
            and row.get("fresh_process_cli") is True, "COST_NOT_VALID_FRESH_COMPLETE", cost_id=cost_id)
    require(row["cost_id"] == cost_id and row["accuracy_run_id"] == arm["run_id"]
            and row["dataset"] == arm["dataset"] and row["batch_size"] == batch and row["schedule"] == schedule,
            "COST_FACET_IDENTITY_MISMATCH", cost_id=cost_id)
    require(row["row_microbatch"] == 128 and row["cached_forecast_reads_in_timed_path"] == 0
            and row["stage_probe_trials"] == 0, "COST_WHOLE_PATH_PROTOCOL_MISMATCH", cost_id=cost_id)
    require(row["warmups"] == row["protocol"]["warmups"] == 10
            and row["timed_trials"] == row["protocol"]["timed_trials"] == 20
            and row["protocol"]["F_row_microbatch"] == 128
            and row["protocol"]["atol"] == DEPLOY_ATOL and row["protocol"]["rtol"] == DEPLOY_RTOL,
            "COST_REGISTERED_TRIAL_OR_PARITY_PROTOCOL_MISMATCH",cost_id=cost_id)
    pid = (int(row["process_pid"]),float(row["process_create_time"]))
    require(pid[0] > 0 and math.isfinite(pid[1]) and pid not in used_pids, "COST_PROCESS_IDENTITY_DUPLICATE", cost_id=cost_id)
    used_pids.add(pid)
    require([(v["phase"],v["trial"]) for v in trials] == [("warmup",i) for i in range(10)]+[("timed",i) for i in range(20)],
            "EXACT_WARMUP10_TIMED20_SEQUENCE", cost_id=cost_id)
    compare_trial_csv(raw_csv_text,trials,cost_id)
    expected = expected_invocations(arm["route_spec"],arm,batch,schedule)
    first = origins(arm["dataset"],"V_select")[:8].tolist()
    for trial in trials:
        bundle = first if batch == 8 else [first[trial["trial"]%8]]
        require(trial["origins"] == bundle and trial["batch_size"] == batch, "FIXED_TIMING_ORIGIN_BUNDLE_MISMATCH", cost_id=cost_id)
        require(math.isfinite(float(trial["whole_seconds"])) and trial["whole_seconds"] > 0, "COST_DURATION_INVALID", cost_id=cost_id)
        check_work(trial,expected,cost_id)
    timed = [v for v in trials if v["phase"] == "timed"]
    ms = [float(v["whole_seconds"])*1000 for v in timed]
    q25,p50,q75,p95 = [quantile_linear(ms,q) for q in (.25,.5,.75,.95)]
    computed = dict(median_batch_ms=p50,p50_batch_ms=p50,IQR_batch_ms=q75-q25,p95_batch_ms=p95,
                    median_ms_per_origin=p50/batch,p50_ms_per_origin=p50/batch,
                    IQR_ms_per_origin=(q75-q25)/batch,p95_ms_per_origin=p95/batch)
    for key,value in computed.items():
        close(row[key],value,"COST_QUANTILE_MISMATCH",cost_id=cost_id,field=key)
    for key in ("peak_allocated_MiB","peak_reserved_MiB"):
        values = [float(v[key]) for v in timed if v[key] is not None]
        if values:
            require(all(math.isfinite(x) and x >= 0 for x in values),"ALLOCATOR_MEMORY_INVALID",cost_id=cost_id)
            close(row[key],max(values),"COST_ALLOCATOR_PEAK_MISMATCH",cost_id=cost_id,field=key)
        else:
            require(row[key] is None,"UNAVAILABLE_ALLOCATOR_MEMORY_NOT_NULL",cost_id=cost_id,field=key)
    for key in ("CPU_RSS_sampled_peak_MiB","process_GPU_sampled_peak_MiB","process_GPU_sampling_status"):
        require(row[key] == receipt["sampling"][key],"SAMPLER_SUMMARY_MISMATCH",cost_id=cost_id,field=key)
    samples=receipt["sampling"]["samples"]
    require(len(samples) == receipt["sampling"]["sample_count"] and len(samples) >= 2,
            "RAW_MEMORY_SAMPLE_COUNT_MISMATCH",cost_id=cost_id)
    require(all(s["phase"] in ("warmup","timed") and math.isfinite(float(s["CPU_RSS_MiB"]))
                and s["CPU_RSS_MiB"] >= 0 for s in samples),
            "RAW_MEMORY_SAMPLE_INVALID",cost_id=cost_id)
    require(all(samples[i]["elapsed_seconds"] <= samples[i+1]["elapsed_seconds"]
                for i in range(len(samples)-1)),"RAW_MEMORY_SAMPLE_TIME_ORDER",cost_id=cost_id)
    close(row["CPU_RSS_sampled_peak_MiB"],max(s["CPU_RSS_MiB"] for s in samples),
          "RSS_RAW_SAMPLER_PEAK_MISMATCH",cost_id=cost_id)
    gpu_samples=[s["process_GPU_MiB"] for s in samples if s["process_GPU_MiB"] is not None]
    if gpu_samples:
        require(all(math.isfinite(float(x)) and x >= 0 for x in gpu_samples),"RAW_PID_GPU_SAMPLE_INVALID",cost_id=cost_id)
        close(row["process_GPU_sampled_peak_MiB"],max(gpu_samples),"GPU_RAW_SAMPLER_PEAK_MISMATCH",cost_id=cost_id)
    else:
        require(row["process_GPU_sampled_peak_MiB"] is None,"GPU_RAW_SAMPLER_PEAK_NOT_NULL",cost_id=cost_id)
    if row["process_GPU_sampling_status"] != "AVAILABLE":
        require(row["process_GPU_sampled_peak_MiB"] is None,"UNAVAILABLE_PER_PID_MEMORY_NOT_NULL",cost_id=cost_id)
    if arm["mode"] == "Ridge":
        require(row["device"] == "cpu" and row["comparison_scope"] == "CPU_RIDGE_CONTEXT" and row["memory_layer"] == "cpu_RSS",
                "RIDGE_CPU_SCOPE_MISMATCH",cost_id=cost_id)
    else:
        require(row["device"].startswith("cuda") and row["memory_layer"] == "pytorch_allocated_tensor","GPU_MEMORY_LAYER_MISMATCH",cost_id=cost_id)
    require(parity.get("status") == "PASS" and parity.get("saved_reference") == "PASS"
            and parity.get("atol") == DEPLOY_ATOL and parity.get("rtol") == DEPLOY_RTOL,
            "FIXED_WHOLE_V_PARITY_RECEIPT_REQUIRED",cost_id=cost_id)
    loaded_f = row["loaded_F_state_sha256"]
    if arm["mode"] == "LoRA":
        # The comparator has a registered merged weight update; do not mistake
        # its deployment identity for the unchanged main-path foundation.
        require(isinstance(loaded_f,str) and len(loaded_f) == 64
                and all(x in "0123456789abcdef" for x in loaded_f),
                "COST_LORA_FOUNDATION_IDENTITY_MISSING",cost_id=cost_id)
    else:
        require(loaded_f == (F_SHA if expected else "NO_FOUNDATION_USED"),
                "COST_FOUNDATION_STATE_IDENTITY_MISMATCH",cost_id=cost_id)
    for source_name,parity_name in (("origins","origins"),("target_normalized","target"),("observed","observed")):
        require(parity[parity_name] == prediction_identity[source_name],"COST_PARITY_COHORT_IDENTITY_MISMATCH",cost_id=cost_id,field=parity_name)
    require(parity["n_origins"] == arm["score"]["n_origins"],"COST_PARITY_FULL_V_COUNT_MISMATCH",cost_id=cost_id)
    for metric in ("MSE","MAE"):
        close(row[metric],parity["independent_metrics"][metric.lower()],"COST_PARITY_METRIC_SUMMARY_MISMATCH",cost_id=cost_id,metric=metric)
        close(row[metric],arm["score"][metric.lower()],"COST_V_METRIC_FIXED_TOLERANCE_MISMATCH",
              atol=DEPLOY_ATOL,cost_id=cost_id,metric=metric)
    remaining = len(origins(arm["dataset"],"V_select"))
    require(len(parity["F_work"]) == math.ceil(remaining/batch),"FULL_V_WORK_BATCH_COUNT_MISMATCH",cost_id=cost_id)
    for work in parity["F_work"]:
        count = min(batch,remaining)
        check_work(work,expected_invocations(arm["route_spec"],arm,count,schedule),cost_id)
        remaining -= count
    require(remaining == 0 and row["F_invocations"] == timed[0]["F_invocations"],"COST_WORK_SUMMARY_MISMATCH",cost_id=cost_id)
    for key in ("F_calls","F_logical_streams","F_physical_streams"):
        require(row[key] == timed[0][key],"COST_WORK_TOTAL_SUMMARY_MISMATCH",cost_id=cost_id,field=key)
    return dict(cost_id=cost_id,run_id=arm["run_id"],dataset=arm["dataset"],batch_size=batch,schedule=schedule,
                device=row["device"],process_pid=pid[0],process_create_time=pid[1],**computed,
                peak_allocated_MiB=row["peak_allocated_MiB"],peak_reserved_MiB=row["peak_reserved_MiB"],
                process_GPU_sampling_status=row["process_GPU_sampling_status"],trial_count=len(trials),
                timed_trial_count=20,whole_path_parity_receipt="PASS",independent_trial_arithmetic="PASS")


def audit_cp0(repo, validation, models, contract, shared):
    require(validation.get("status") == "PASS" and validation.get("coverage") == validation.get("expected_coverage") == 10
            and validation.get("training_updates") == 0 and validation.get("V_access") is False
            and validation.get("T_access") is False,"CP0_VALIDATION_SCOPE_OR_COVERAGE")
    terminal_path = resolve(repo,validation["terminal_binding"])
    require(terminal_path == (Path(repo)/".cache"/CAMPAIGN/"TERMINAL_CORE_002.json").resolve(),
            "CP0_TERMINAL_BINDING_MISMATCH")
    supplement_plan = read_json(resolve(repo,validation["plan_binding"]))
    require(supplement_plan.get("status") == "REGISTERED_BEFORE_NEW_V"
            and supplement_plan.get("scientific_contract_changed") is False
            and supplement_plan.get("selection_changes") is False
            and supplement_plan.get("new_body_training") is False,"CP0_SUPPLEMENT_PLAN_SCOPE")
    primary_path = resolve(repo,validation["primary_plan_binding"])
    primary_plan = read_json(primary_path)
    require(resolve(repo,supplement_plan["primary_plan_binding"]) == primary_path
            and supplement_plan["primary_source_files"] == primary_plan["source_files"],
            "CP0_PRIMARY_PLAN_BINDING_MISMATCH")
    for relative,digest in supplement_plan["source_files"].items():
        source=(Path(repo)/relative).resolve()
        require(source.is_relative_to(Path(repo).resolve()) and sha(source) == digest,
                "CP0_SUPPLEMENT_SOURCE_CHANGED",source=relative)
    start=read_json(resolve(repo,validation["original_execution_start_binding"]))
    close(validation["original_deadline_unix"],start["started_unix"]+start["limits"]["wall_seconds"],
          "CP0_ORIGINAL_DEADLINE_MISMATCH")
    close(validation["GPU_seconds_added"],validation["GPU_seconds_after"]-validation["GPU_seconds_before"],
          "CP0_GPU_ARITHMETIC_MISMATCH")
    require(validation["GPU_seconds_added"] >= 0,"CP0_NEGATIVE_GPU_ACCOUNTING")
    if validation["device"] == "cpu":
        close(validation["GPU_seconds_added"],0.,"CP0_CPU_DIAGNOSTIC_GPU_ACCOUNTING")
    csv_path = resolve(repo,validation["CSV_binding"])
    with csv_path.open(newline="",encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    evidence = validation["evidence"]
    bodies = {key:arm for key,arm in models.items() if arm["mode"] in ("S","A","B")}
    require(len(rows) == len(evidence) == 10 and {x["run_id"] for x in rows} == {x["run_id"] for x in evidence} == set(bodies),
            "CP0_EXACT_TEN_UNIQUE_BODY_COVERAGE")
    by_id,result = {row["run_id"]:row for row in rows},[]
    for record in evidence:
        run_id = record["run_id"]
        arm,row = bodies[run_id],by_id[run_id]
        dataset = arm["dataset"]
        values,metadata = archive_values(resolve(repo,record["prediction_binding"]))
        check_geometry(values,dataset,"A_stop",contract["tasks"][dataset]["columns"],run_id)
        require(metadata == record["metadata"] and metadata.get("run_id") == run_id
                and metadata.get("dataset") == dataset and metadata.get("mode") == arm["mode"]
                and metadata.get("seed") == arm["seed"] and metadata.get("epoch") == 0
                and metadata.get("training_updates") == 0 and metadata.get("split") == "A_stop"
                and metadata.get("A_fit_end") == EXPECTED[dataset]["A_fit_end"]
                and metadata.get("V_access") is False and metadata.get("T_access") is False,
                "CP0_METADATA_SCOPE_IDENTITY",run_id=run_id)
        require(metadata["foundation_sha256"] == F_SHA
                and metadata["checkpoint_contract"]["data_sha"] == contract["tasks"][dataset]["data_sha256"]
                and metadata["checkpoint_contract"]["model_sha"] == F_SHA
                and metadata["checkpoint_contract"]["scaler_sha"] == metadata["scaler_sha256"]
                and metadata["checkpoint_contract"]["initialization_sha"] == metadata["initialization_sha256"]
                and metadata["source_files"] == primary_plan["source_files"],
                "CP0_FOUNDATION_DATA_SCALER_SOURCE_METADATA_MISMATCH",run_id=run_id)
        identities = {name:identity(value) for name,value in values.items()}
        require(identities == record["arrays"],"CP0_ARRAY_IDENTITY_MISMATCH",run_id=run_id)
        compare_cohort(shared,values,metadata,dataset,"A_stop",run_id)
        stats = statistics(values["pred_normalized"],values["target_normalized"],values["observed"])
        compare_stats(stats,record["statistics"],run_id)
        require(row["dataset"] == dataset and row["mode"] == arm["mode"] and int(row["seed"]) == arm["seed"]
                and row["split"] == "A_stop" and int(row["epoch"]) == int(row["updates"]) == int(row["exposures"]) == 0
                and int(row["fit_end"]) == EXPECTED[dataset]["A_fit_end"] and row["MSE_parity"] == "PASS"
                and row["origins_sha256"] == identities["origins"]["sha256"]
                and int(row["n_origins"]) == stats["n_origins"] and int(row["n_observed"]) == stats["n_observed"],
                "CP0_CSV_IDENTITY_SCOPE_MISMATCH",run_id=run_id)
        for metric in ("mse","mae"):
            close(stats[metric],row[metric.upper()],"CP0_CSV_METRIC_MISMATCH",run_id=run_id,metric=metric)
            close(stats[metric],row["A_stop_"+metric],"CP0_CSV_A_STOP_METRIC_MISMATCH",run_id=run_id,metric=metric)
        originals = [p for p in arm["curves"] if float(p["epoch"]) == 0]
        require(len(originals) == 1,"ORIGINAL_CP0_UNIQUE_POINT_REQUIRED",run_id=run_id)
        close(stats["mse"],originals[0]["astop_mse"],"CP0_ORIGINAL_MSE_PARITY_MISMATCH",
              atol=DEPLOY_ATOL,rtol=DEPLOY_RTOL,run_id=run_id)
        close(row["original_CP0_MSE"],originals[0]["astop_mse"],"CP0_CSV_ORIGINAL_MSE_MISMATCH",run_id=run_id)
        require(row["prediction_sha256"] == record["prediction_binding"]["sha256"],"CP0_CSV_PREDICTION_BINDING_MISMATCH",run_id=run_id)
        result.append(dict(run_id=run_id,dataset=dataset,mode=arm["mode"],seed=arm["seed"],
                           prediction_binding=record["prediction_binding"],array_identities=identities,
                           statistics=stats,original_mse_parity="PASS",status="PASS"))
    return result


def main_audit(repo, require_cp0=False):
    started,repo = time.time(),Path(repo).resolve()
    gate = numerical_gate(repo)  # FIRST: no numerical archive/table before this gate.
    report,cache = repo/"reports"/CAMPAIGN,repo/".cache"/CAMPAIGN
    contract = read_json(repo/"reports/b2_revin_20261018_v1/DATA_CONTRACT.json")
    for dataset,task in EXPECTED.items():
        saved = contract["tasks"][dataset]
        require(all(saved[key] == task[key] for key in ("C","L","H"))
                and tuple(saved["split_ranges"]["A_stop"]) == task["A_stop"]
                and tuple(saved["split_ranges"]["V_select"]) == task["V_select"]
                and saved["split_ranges"]["A_fit"][1] == task["A_fit_end"],"SEALED_DATA_RANGE_OR_GEOMETRY_MISMATCH",dataset=dataset)
    legacy = legacy_origins(repo)
    transfer = read_json(report/"TRANSFER_SELECTION.json")
    require(transfer.get("status") == "SEALED" and transfer.get("V_used") is False
            and transfer.get("selection_split") == "A_stop","TRANSFER_NOT_A_STOP_SEALED")
    scores = read_json(report/"SCORES.json")
    require(scores.get("status") == "COMPLETE" and scores.get("V_independent_test") is False,"SCORES_NOT_COMPLETE_DEVELOPMENT_V")
    arms = scores["arms"]
    validate_coverage(arms,transfer["selected_mode"])
    selection = audit_selection(arms,transfer)
    require(gate["validated_models"] == len(arms),"PRIMARY_VALIDATION_MODEL_COVERAGE_MISMATCH")
    shared,score_rows,array_identities = {},[],{}
    for arm in arms:
        run_id,dataset = arm["run_id"],arm["dataset"]
        values,metadata = archive_values(resolve(repo,arm["prediction_binding"]))
        check_geometry(values,dataset,"V_select",contract["tasks"][dataset]["columns"],run_id)
        require(metadata.get("run_id") == run_id and metadata.get("dataset") == dataset
                and metadata.get("split") == "V_select" and metadata.get("T_access") is False
                and metadata.get("route_spec") == arm["route_spec"]
                and metadata.get("data_sha256") == contract["tasks"][dataset]["data_sha256"]
                and metadata.get("parent_fit_end") == EXPECTED[dataset]["A_stop"][1],
                "PREDICTION_METADATA_MODEL_DATA_SCOPE_MISMATCH",run_id=run_id)
        identities = {name:identity(value) for name,value in values.items()}
        require(identities == arm["array_identities"],"SCORED_ARM_ARRAY_IDENTITY_MISMATCH",run_id=run_id)
        compare_cohort(shared,values,metadata,dataset,"V_select",run_id)
        stats = statistics(values["pred_normalized"],values["target_normalized"],values["observed"])
        compare_stats(stats,arm["score"],run_id)
        score_rows.append(dict(run_id=run_id,dataset=dataset,mode=arm["mode"],seed=arm.get("seed"),
                               prediction_binding=arm["prediction_binding"],array_identities=identities,
                               statistics=stats,status="PASS"))
        array_identities[run_id] = identities
    index = read_json(report/"COST_INDEX.json")
    require(index.get("status") == "COMPLETE","COST_INDEX_NOT_COMPLETE")
    facets = index["facets"]
    expected = {(a["run_id"],schedule,batch) for a in arms
                for schedule in (("native","two_call") if a["mode"] == "A" else ("native",)) for batch in (1,8)}
    actual = [(f["run_id"],f["schedule"],int(f["batch"])) for f in facets]
    require(len(actual) == len(set(actual)) and set(actual) == expected,"EXACT_NATIVE_A_TWO_CALL_B1_B8_COVERAGE")
    require(gate["validated_cost_facets"] == len(facets),"PRIMARY_VALIDATION_COST_COVERAGE_MISMATCH")
    models,used_pids,cost_rows = {a["run_id"]:a for a in arms},set(),[]
    for facet in facets:
        path = resolve(repo,facet["output"])
        receipt = read_json(path);arm = models[facet["run_id"]]
        request_path = resolve(repo,receipt["request_binding"]);request = read_json(request_path)
        spec = dict(arm["route_spec"],schedule="two_call") if facet["schedule"] == "two_call" else arm["route_spec"]
        require(receipt["route_spec"] == request["route_spec"] == spec
                and receipt["request_contract_sha256"] == fingerprint({k:v for k,v in request.items() if not k.startswith("_")})
                and request["accuracy_run_id"] == arm["run_id"] and request["cost_id"] == facet["cost_id"]
                and receipt["prediction_reference_sha256"] == request["prediction_sha256"] == arm["prediction_binding"]["sha256"],
                "COST_REQUEST_MODEL_PREDICTION_BINDING_MISMATCH",cost_id=facet["cost_id"])
        require(bool(receipt.get("producer_environment")) and bool(receipt.get("producer_sources")),
                "COST_PRODUCER_PROVENANCE_MISSING",cost_id=facet["cost_id"])
        for relative,digest in receipt["producer_sources"].items():
            source = (repo/relative).resolve()
            require(source.is_relative_to(repo) and sha(source) == digest,"COST_PRODUCER_SOURCE_CHANGED",source=relative)
        trial_path = path.with_suffix(".trials.csv")
        require(sha(trial_path) == receipt["trials_sha256"],"RAW_TRIAL_CSV_SHA_MISMATCH",cost_id=facet["cost_id"])
        row = audit_cost(receipt,arm,facet,raw_csv_text=trial_path.read_text(encoding="utf8"),
                         prediction_identity=array_identities[arm["run_id"]],used_pids=used_pids)
        row.update(receipt_binding=facet["output"],raw_trials_binding=ref(repo,trial_path))
        cost_rows.append(row)
    cp0_rows,cp0_path = [],report/"CP0_A_STOP_VALIDATION.json"
    if cp0_path.exists():
        cp0_rows = audit_cp0(repo,read_json(cp0_path),models,contract,shared)
    else:
        require(not require_cp0,"CP0_VALIDATION_REQUIRED_NOT_PRESENT")
    return dict(status="PASS" if cp0_rows else "PASS_MAIN_CP0_NOT_YET_PRESENT",created_unix=time.time(),
        elapsed_CPU_wall_seconds=time.time()-started,gate=gate,legacy_origin_identity=legacy,A_stop_selection_arithmetic=selection,
        coverage=dict(V_models=len(score_rows),whole_path_cost_facets=len(cost_rows),CP0_models=len(cp0_rows),expected_CP0_models=10),
        scope="Independent saved FP32 array FP64 arithmetic and integer cohorts; bound raw-trial cost arithmetic; no model loading/inference",
        tolerances=dict(saved_metric_absolute=METRIC_ATOL,sufficient_stats_absolute=STATS_ATOL,sufficient_stats_relative=STATS_RTOL,
                        deployment_receipt_metric_absolute=DEPLOY_ATOL,deployment_prediction_atol=DEPLOY_ATOL,deployment_prediction_rtol=DEPLOY_RTOL),
        tolerance_basis="Frozen primary report._close/np.allclose and costs.PROTOCOL, fixed before new numerical access and never widened",
        V_development_evaluation=True,T_access=False,new_model_inference=False,model_weights_opened=False,
        training_updates=0,selection_changes=False,V_scores=score_rows,cost_facets=cost_rows,CP0_A_stop=cp0_rows,
        limitations=["Deployment forward parity is bound runtime evidence; this CPU audit does not rerun models.",
                     "Targets/masks/scalers are compared across every saved dataset cohort; primary PASS binds raw-source canonical verification.",
                     "Overlapping origins/channels and two seeds are not treated as independent replication."])


def serialize_trials(trials):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text,fieldnames=list(trials[0]))
    writer.writeheader()
    for row in trials:
        writer.writerow({k:json.dumps(v,separators=(",",":")) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    return text.getvalue()


def synthetic_cost():
    arm = dict(dataset="jena",mode="B",run_id="B_jena_s92601",route_spec=dict(family="B2",latent_k=12),
               score=dict(mse=1.,mae=.7,n_origins=43))
    facet = dict(cost_id="synthetic_B8",run_id=arm["run_id"],schedule="native",batch=8)
    trials = []
    work = expected_invocations(arm["route_spec"],arm,8)
    totals = {key:sum(v[key] for v in work) for key in ("F_calls","F_logical_streams","F_physical_streams")}
    for phase,n in (("warmup",10),("timed",20)):
        for i in range(n):
            trials.append(dict(phase=phase,trial=i,whole_seconds=.010+i*.0001,batch_size=8,
                origins=origins("jena","V_select")[:8].tolist(),F_invocations=copy.deepcopy(work),**totals,
                peak_allocated_MiB=200.+i if phase == "timed" else None,
                peak_reserved_MiB=300.+i if phase == "timed" else None,CPU_RSS_MiB=128.,process_GPU_MiB=None,
                prediction_sha256="synthetic_only"))
    ms=[v["whole_seconds"]*1000 for v in trials if v["phase"] == "timed"]
    q25,p50,q75,p95 = [float(np.quantile(ms,q)) for q in (.25,.5,.75,.95)]
    sampling = dict(CPU_RSS_sampled_peak_MiB=128.,process_GPU_sampled_peak_MiB=None,
                    process_GPU_sampling_status="UNAVAILABLE_SYNTHETIC",sample_count=2,
                    samples=[dict(phase="warmup",elapsed_seconds=0.,CPU_RSS_MiB=120.,process_GPU_MiB=None),
                             dict(phase="timed",elapsed_seconds=1.,CPU_RSS_MiB=128.,process_GPU_MiB=None)])
    row = dict(status="COMPLETE",cost_valid=True,fresh_process_cli=True,cost_id=facet["cost_id"],
        accuracy_run_id=arm["run_id"],dataset="jena",batch_size=8,schedule="native",row_microbatch=128,
        cached_forecast_reads_in_timed_path=0,stage_probe_trials=0,process_pid=123,process_create_time=456.,
        warmups=10,timed_trials=20,loaded_F_state_sha256=F_SHA,
        protocol=dict(warmups=10,timed_trials=20,F_row_microbatch=128,atol=DEPLOY_ATOL,rtol=DEPLOY_RTOL),
        median_batch_ms=p50,p50_batch_ms=p50,IQR_batch_ms=q75-q25,p95_batch_ms=p95,
        median_ms_per_origin=p50/8,p50_ms_per_origin=p50/8,IQR_ms_per_origin=(q75-q25)/8,p95_ms_per_origin=p95/8,
        peak_allocated_MiB=219.,peak_reserved_MiB=319.,device="cuda",memory_layer="pytorch_allocated_tensor",
        MSE=1.,MAE=.7,F_invocations=copy.deepcopy(work),**totals,
        **{k:v for k,v in sampling.items() if k not in ("samples","sample_count")})
    ident = dict(origins=identity(origins("jena","V_select")),
                 target_normalized=dict(shape=[43,48,21],dtype="float32",sha256="synthetic_target"),
                 observed=dict(shape=[43,48,21],dtype="bool",sha256="synthetic_mask"))
    allwork=[]
    for count in (8,8,8,8,8,3):
        inv = expected_invocations(arm["route_spec"],arm,count)
        allwork.append(dict(F_invocations=inv,**{key:sum(v[key] for v in inv) for key in totals}))
    parity=dict(status="PASS",saved_reference="PASS",atol=DEPLOY_ATOL,rtol=DEPLOY_RTOL,
                origins=ident["origins"],target=ident["target_normalized"],observed=ident["observed"],
                n_origins=43,independent_metrics=dict(mse=1.,mae=.7),F_work=allwork)
    return dict(status="COMPLETE",row=row,trials=trials,parity=parity,sampling=sampling),arm,facet,ident


def self_test():
    """All arrays/NPZ/receipts below are generated in memory, without campaign reads."""
    passed=[]
    def fails(callback,code):
        try:
            callback()
        except AuditFailure as error:
            require(error.code == code,"SELF_TEST_WRONG_FAILURE",expected=code,actual=error.code)
        else:
            raise AuditFailure("SELF_TEST_FAILURE_EXPECTED",expected=code)
    p=np.ones((2,2,3),np.float32);p[:,:,1]=10;p[:,:,2]=100
    y=np.zeros_like(p);mask=np.zeros_like(p,dtype=bool);mask[:,:,0]=True;mask[0,0,1]=True;y[~mask]=np.nan
    stats=statistics(p,y,mask)
    require(stats["mse"] == 50.5 and stats["mae"] == 5.5 and stats["n_observed"] == 5,"SELF_TEST_MACRO")
    passed.append("FP64_equal_channel_macro_with_unequal_missing_counts")
    fails(lambda:statistics(p,y,mask.astype(np.uint8)),"BOOL_MASK_REQUIRED")
    bad=p.copy();bad[0,0,0]=np.inf
    fails(lambda:statistics(bad,y,mask),"NONFINITE_PREDICTION_OR_OBSERVED_TARGET")
    passed.append("original_bool_mask_and_nonfinite_fail_closed")
    for dataset in EXPECTED:
        a=origins(dataset,"V_select");b=a.copy();b[0]+=1
        require(identity(a)["sha256"] == EXPECTED[dataset]["V_sha256"] and identity(a) != identity(b),"SELF_TEST_ORIGINS")
    passed.append("exact_integer_origin_arrays_and_legacy_payload_hashes")
    buffer=io.BytesIO()
    np.savez_compressed(buffer,pred_normalized=p,target_normalized=y,observed=mask,origins=np.array([1,2],np.int64),
                        channel_ids=np.array(["a","b","c"]),metadata_json=np.asarray("{}"))
    buffer.seek(0);values,meta=archive_values(buffer)
    require(meta == {} and np.array_equal(values["pred_normalized"],p),"SELF_TEST_NPZ")
    passed.append("synthetic_in_memory_NPZ_only_no_pickle")
    with patch(__name__+".read_json",return_value=dict(status="RUNNING",exit_code=None)), patch.object(np,"load",side_effect=AssertionError("numerical access before terminal")):
        fails(lambda:main_audit(Path(__file__).resolve().parents[2]),"GATE_MAIN_TERMINAL_NOT_COMPLETE")
    passed.append("live_terminal_blocks_every_numerical_read")
    with patch(__name__+".read_json",side_effect=[dict(status="COMPLETE",exit_code=0),dict(status="FAIL")]), patch.object(np,"load",side_effect=AssertionError("numerical access before primary PASS")):
        fails(lambda:main_audit(Path(__file__).resolve().parents[2]),"GATE_PRIMARY_VALIDATION_NOT_PASS")
    passed.append("failed_primary_validation_blocks_every_numerical_read")
    numbers=[.3,.01,7.,2.,5.,9.,1.,8.,4.,6.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.]
    for q in (.25,.5,.75,.95):
        close(quantile_linear(numbers,q),float(np.quantile(numbers,q)),"SELF_TEST_QUANTILE")
    passed.append("independent_linear_order_statistic_quantiles")
    selection_arms=[dict(run_id=f"{mode}_jena_s{seed}",dataset="jena",mode=mode,seed=seed,
                         curves=[dict(epoch=0.,astop_mse=2.),dict(epoch=.25,astop_mse=1.),dict(epoch=1.,astop_mse=1.)],
                         search_summary=dict(selected_epoch=.25,selected_astop_mse=1.))
                    for mode in ("A","B") for seed in SEEDS]
    transfer=dict(selected_mode="A",mean_astop_mse=dict(A=1.,B=1.))
    require(audit_selection(selection_arms,transfer)["selected_mode"] == "A","SELF_TEST_A_STOP_TIE")
    bad=copy.deepcopy(selection_arms);bad[0]["search_summary"]["selected_epoch"]=1.
    fails(lambda:audit_selection(bad,transfer),"A_STOP_EARLIEST_MIN_EPOCH_MISMATCH")
    passed.append("A_stop_earliest_checkpoint_and_exact_two_seed_tie_A")
    receipt,arm,facet,ident=synthetic_cost()
    kwargs=dict(raw_csv_text=serialize_trials(receipt["trials"]),prediction_identity=ident,used_pids=set())
    result=audit_cost(receipt,arm,facet,**kwargs)
    require(result["timed_trial_count"] == 20 and result["peak_allocated_MiB"] == 219.,"SELF_TEST_COST")
    passed.append("whole_B8_synthetic_receipt_CSV_quantile_peaks_full_V_partial_batch")
    bad=copy.deepcopy(receipt);bad["row"]["p95_batch_ms"]+=1
    fails(lambda:audit_cost(bad,arm,facet,**dict(kwargs,used_pids=set())),"COST_QUANTILE_MISMATCH")
    bad=copy.deepcopy(receipt);bad["row"]["peak_allocated_MiB"]+=1
    fails(lambda:audit_cost(bad,arm,facet,**dict(kwargs,used_pids=set())),"COST_ALLOCATOR_PEAK_MISMATCH")
    passed.append("latency_and_allocator_peak_tampering_rejected")
    bad=copy.deepcopy(receipt);bad["trials"][0]["F_invocations"][1]["actual_context"]=512
    fails(lambda:audit_cost(bad,arm,facet,**dict(kwargs,raw_csv_text=serialize_trials(bad["trials"]),used_pids=set())),"F_GEOMETRY_OR_CALL_MISMATCH")
    fails(lambda:audit_cost(receipt,arm,facet,**dict(kwargs,used_pids={(123,456.)})),"COST_PROCESS_IDENTITY_DUPLICATE")
    passed.append("padded_coarse_context_and_duplicate_process_rejected")
    control=dict(family="COMBINED",operation="past_control",g_route=dict(family="DIRECT_G"),
                 b_route=dict(family="B2",latent_k=12))
    require(expected_invocations(control,arm,8) == [],"SELF_TEST_CONTROL")
    control.update(operation="alpha",alpha=1.)
    require(len(expected_invocations(control,arm,8)) == 1,"SELF_TEST_ALPHA")
    passed.append("past_control_and_alpha_endpoint_unused_parent_pruning")
    fails(lambda:compare_trial_csv(kwargs["raw_csv_text"].replace("0.01,","0.02,",1),receipt["trials"],"synthetic"),
          "RAW_TRIAL_CSV_VALUE_MISMATCH")
    compare_stats(stats,copy.deepcopy(stats),"synthetic")
    bad=copy.deepcopy(stats);bad["mae"]+=.01
    fails(lambda:compare_stats(stats,bad,"synthetic"),"SAVED_SCORE_MISMATCH")
    passed.append("raw_CSV_every_field_and_metric_tampering_rejected")
    require("torch" not in sys.modules,"SELF_TEST_TORCH_IMPORT")
    return dict(status="PASS_SYNTHETIC_ONLY",checks=passed,count=len(passed),actual_campaign_scores_opened=False,
                actual_prediction_arrays_opened=False,model_weights_opened=False,GPU_execution=False)


def write_new_json(path,value):
    path=Path(path)
    require(not path.exists(),"PRESERVE_EXISTING_AUDIT_OUTPUT")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("self-test","audit"))
    parser.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output",type=Path)
    parser.add_argument("--require-cp0",action="store_true")
    args=parser.parse_args(argv)
    if args.stage == "self-test":
        print(json.dumps(self_test(),ensure_ascii=False));return 0
    repo=args.repo_root.resolve()
    output=args.output.resolve() if args.output else repo/"reports"/CAMPAIGN/"INDEPENDENT_FINAL_AUDIT_001.json"
    require(output.is_relative_to(repo/"reports"/CAMPAIGN) or output.is_relative_to(repo/".cache"/CAMPAIGN),"AUDIT_OUTPUT_ROOT")
    require(not output.exists(),"PRESERVE_EXISTING_AUDIT_OUTPUT")
    try:
        value=main_audit(repo,require_cp0=args.require_cp0)
    except AuditFailure as error:
        print(json.dumps(dict(status="FAIL",code=error.code,details=error.details),ensure_ascii=False));return 1
    value["audit_source"]=dict(reference="repo://"+Path(__file__).resolve().relative_to(repo).as_posix(),sha256=sha(__file__))
    write_new_json(output,value)
    print(json.dumps(dict(status=value["status"],coverage=value["coverage"],output=ref(repo,output)),ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
