"""Fresh-worker whole physical-path costs, with independent whole-V parity."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import psutil
import torch

from research.memory_budget_20261018_v1.worker import ProcessSampler, _memory, _sync
from . import common
from .deployment import make_route, resident_state


PROTOCOL = dict(warmups=10, timed_trials=20, F_row_microbatch=128,
    atol=common.ATOL, rtol=common.RTOL, stage_probe_trials=0,
    whole_path="physical+bool mask -> causal fill -> fixed scaler -> actual model/F/repeat/lag -> inverse scaler -> CPU numpy",
    cached_forecast_reads_in_timed_path=0,
    timing_origins="first eight fixed V origins; B1 cycles, B8 repeats their ordered bundle")


def producer_sources(ctx):
    sources = common.model_sources(ctx['repo'])
    relatives = ['research/'+common.NAME+'/'+name for name in ('costs.py','deployment.py')]
    relatives += ['research/informs_broad_validation_20261015_v1/'+name for name in
                  ('deployment.py','fit_base.py','training.py','data_contract.py','evaluation.py','networks.py','gpu_runtime.py','combination.py')]
    relatives += ['research/memory_budget_20261018_v1/worker.py',
                  'research/temporal_frontend_campaign_20261015_v1/runtime.py']
    sources.update({name:common.sha(ctx['repo']/name) for name in relatives})
    return sources


def request_identity(request):
    return common.fingerprint({key:value for key,value in request.items() if not key.startswith('_')})


def validate_receipt(value,request,ctx,request_path=None):
    if (value.get('status')!='COMPLETE' or value.get('request_contract_sha256')!=request_identity(request)
            or value.get('route_spec')!=request['route_spec']
            or value.get('producer_sources')!=producer_sources(ctx)
            or value.get('prediction_reference_sha256')!=request.get('prediction_sha256')):
        raise ValueError('Cost receipt request/route/source/prediction binding changed')
    if request_path is not None and value.get('request_binding')!=common.binding(request_path,ctx):
        raise ValueError('Cost receipt is not bound to this exact sealed request file')
    return value


def _deadline(request):
    if time.time()>=float(request.get('_deadline_unix',1e12)):
        raise TimeoutError('Registered wall/GPU budget reached during cost measurement')


def validate_request(request):
    value = dict(request)
    for key in ("accuracy_run_id", "cost_id"):
        if not isinstance(value.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", value[key]):
            raise ValueError("Safe registered cost/accuracy identity required: " + key)
    if value.get("dataset") not in common.EXPECTED or type(value.get("batch_size")) is not int or value["batch_size"] not in (1, 8):
        raise ValueError("Registered dataset and B1/B8 required")
    if value.get("device", "cuda") not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Only the registered CPU or first CUDA device is supported")
    if not isinstance(value.get("route_spec"), dict):
        raise ValueError("An exact serialized deployment route is required")
    if value["route_spec"].get("family") == "RAW_RIDGE" and value.get("device", "cuda") != "cpu":
        raise ValueError("Ridge must retain a separate CPU cost scope")
    if value.get("row_microbatch", 128) != 128:
        raise ValueError("This campaign fixes the common F row microbatch at 128")
    if value.get("prediction_path") and not value.get("prediction_sha256"):
        raise ValueError("Saved whole-V prediction requires its exact SHA256")
    value.setdefault("device", "cuda")
    return value


def fixed_origins(task):
    lo, hi = task["split_ranges"]["V_select"]
    values = np.arange(max(int(lo), task["L"]), int(hi)-task["H"]+1, task["H"], dtype=np.int64)
    if not len(values) or hi > len(task["physical"]):
        raise ValueError("Whole registered V must stay inside the admitted prefix")
    return values


def timing_origin_bundles(origins, batch_size):
    origins = np.asarray(origins)
    if origins.ndim != 1 or origins.dtype.kind not in "iu" or not len(origins) or batch_size not in (1, 8):
        raise ValueError("Ordered integer origins and B1/B8 required")
    selected = origins[np.arange(8) % len(origins)]
    return [selected[index:index+1] for index in range(8)] if batch_size == 1 else [selected]


def windows(task, origins):
    return (np.stack([task["physical"][int(o)-task["L"]:int(o)] for o in origins]).astype(np.float32),
            np.stack([task["observed"][int(o)-task["L"]:int(o)] for o in origins]))


def _row128(route):
    route.spec.update(row_microbatch=128, F_row_microbatch=128, checkpoint_inputs=False)
    foundation = route.pool.foundation
    if foundation is not None:
        foundation.checkpoint_inputs = False
    for parent in (route.g_route, route.b_route):
        if parent is not None:
            _row128(parent)


def expected_work(route, batch_size):
    """Physical F row chunks and true time lengths for this loaded route."""
    if route.family == "COMBINED":
        result = []
        for parent in (route.g_route, route.b_route):
            if parent is not None:
                result.extend(expected_work(parent, batch_size))
        return result
    family, model = route.family, route.model
    if family in ("RAW", "LORA"):
        widths, lengths, horizons = [len(route.task["columns"])], [route.task["L"]], [route.task["H"]]
    elif family == "B2":
        mode = getattr(model, "raw_context_mode", "S")
        separate = mode == "B" or (mode == "A" and model.raw_context_schedule == "two_call")
        scale = getattr(model, "raw_context_scale", 1)
        widths = [route.latent_k//2]*2 if separate else [route.latent_k]
        lengths = [route.task["L"], route.task["L"]//scale] if separate else [route.task["L"]]
        horizons = [route.task["H"], route.task["H"]//scale] if separate else [route.task["H"]]
    else:
        return []
    return [dict(input_shape=[batch_size, length, width], requested_context=length,
                 used_horizon=horizon, F_logical_streams=batch_size*width,
                 F_physical_streams=batch_size*width, F_calls=math.ceil(batch_size*width/128))
            for width, length, horizon in zip(widths, lengths, horizons)]


def check_work(route, batch_size):
    expected = expected_work(route, batch_size)
    receipt = route.last_receipt
    for key in ("F_logical_streams", "F_physical_streams", "F_calls"):
        if receipt.get(key) != sum(row[key] for row in expected):
            raise ValueError("Actual F physical work differs: " + key)
    actual = receipt.get("F_invocations", [])
    if len(actual) != len(expected):
        raise ValueError("Actual F invocation count differs")
    for measured, registered in zip(actual, expected):
        for key, value in registered.items():
            if measured.get(key) != value:
                raise ValueError("Actual F branch length/width/horizon differs: " + key)
        if measured.get("actual_context") != registered["requested_context"]:
            raise ValueError("F truncated the registered context")
    return dict(F_logical_streams=receipt["F_logical_streams"],
                F_physical_streams=receipt["F_physical_streams"], F_calls=receipt["F_calls"],
                F_invocations=actual)


def latency_statistics(trials, batch_size):
    timed = [float(row["whole_seconds"])*1000 for row in trials if row["phase"] == "timed"]
    if not timed or not np.isfinite(timed).all() or min(timed) <= 0:
        raise ValueError("Finite positive timed whole-path trials required")
    q25, p50, q75, p95 = np.quantile(np.asarray(timed, np.float64), [.25, .5, .75, .95])
    return dict(p50_batch_ms=float(p50), median_batch_ms=float(p50),
                IQR_batch_ms=float(q75-q25), p95_batch_ms=float(p95),
                p50_ms_per_origin=float(p50/batch_size), median_ms_per_origin=float(p50/batch_size),
                IQR_ms_per_origin=float((q75-q25)/batch_size), p95_ms_per_origin=float(p95/batch_size))


def _reference(request, ctx):
    if not request.get("prediction_path"):
        return None
    root = Path(ctx["cache"]).resolve()
    path = (root / request["prediction_path"]).resolve()
    if not path.is_relative_to(root) or common.sha(path) != request["prediction_sha256"]:
        raise ValueError("Saved prediction cache path or SHA256 changed")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")}


def whole_v_parity(route, task, origins, inputs, reference, batch_size, *, deadline_request=None):
    predictions = []
    receipts = []
    for index in range(0, len(origins), batch_size):
        _deadline(deadline_request or {})
        physical, observed = inputs[0][index:index+batch_size], inputs[1][index:index+batch_size]
        physical_prediction = route.predict_physical(physical, observed)
        predictions.append(((physical_prediction-route.mean)/route.std).astype(np.float32))
        receipts.append(check_work(route, len(physical)))
    _deadline(deadline_request or {})
    prediction = np.concatenate(predictions)
    rows = origins[:, None] + np.arange(task["H"])[None]
    observed = task["observed"][rows]
    target = np.where(observed, (task["physical"][rows]-route.mean)/route.std, 0.).astype(np.float32)
    if reference is not None:
        if reference["observed"].dtype != np.bool_:
            raise ValueError("Saved target observation mask must retain bool dtype")
        for key, expected in (("origins", origins), ("channel_ids", np.asarray(task["columns"])), ("observed", observed)):
            if not np.array_equal(reference[key], expected):
                raise ValueError("Saved full-V cohort differs: " + key)
        if (reference["target_normalized"].shape != target.shape
                or not np.array_equal(reference["target_normalized"][observed], target[observed])):
            raise ValueError("Saved original normalized targets changed")
        expected = reference["pred_normalized"]
        if expected.shape != prediction.shape or not np.allclose(prediction, expected, atol=common.ATOL, rtol=common.RTOL):
            raise ValueError("Whole-V deployment prediction parity failed")
        maximum = float(np.abs(prediction-expected).max())
    else:
        maximum = None
    return dict(status="PASS", saved_reference="PASS" if reference is not None else "NOT_PROVIDED",
        max_abs=maximum, atol=common.ATOL, rtol=common.RTOL, n_origins=len(origins),
        prediction=common.array_identity(prediction), origins=common.array_identity(origins),
        target=common.array_identity(target), observed=common.array_identity(observed),
        independent_metrics=common.statistics(prediction, target, observed), F_work=receipts)


def evaluate_inputs(request, ctx, task, loader, *, reference=None, sampler_factory=ProcessSampler, trial_sink=None):
    """The same measurement core accepts synthetic CPU routes for unit gates."""
    request = validate_request(request)
    _deadline(request)
    device, batch = request["device"], request["batch_size"]
    extraction_started = time.perf_counter()
    origins = fixed_origins(task)
    inputs = windows(task, origins)
    bundles = timing_origin_bundles(origins, batch)
    timing_inputs = [windows(task, selected) for selected in bundles]
    extraction_seconds = time.perf_counter()-extraction_started
    _deadline(request)
    started = time.perf_counter()
    route = loader()
    try:
        _deadline(request)
        if expected_work(route, batch):
            route.pool.get(task["H"])
        _deadline(request)
        _row128(route)
        _sync(device)
        _deadline(request)
        load_seconds = time.perf_counter()-started
        lora_started = time.perf_counter()
        merge = dict(status="NOT_APPLICABLE")
        if route.family == "LORA":
            merge = route.check_lora_parity(*timing_inputs[0], atol=common.ATOL, rtol=common.RTOL)
            if merge.get("status") != "PASS":
                raise ValueError("One-time LoRA merge parity failed")
        merge_seconds = time.perf_counter()-lora_started if route.family == "LORA" else 0.
        route.activate()
        _sync(device)
        _deadline(request)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            _sync(device)
        baseline = _memory(device)
        loaded_resident = resident_state(route)
        loaded_rss = psutil.Process().memory_info().rss/1024**2
        parity_started = time.perf_counter()
        parity = whole_v_parity(route, task, origins, inputs, reference, batch, deadline_request=request)
        parity_seconds = time.perf_counter()-parity_started
        sampler = sampler_factory(device, interval_seconds=.05)
        trials = []
        sampler.start()
        try:
            for phase, count in (("warmup", 10), ("timed", 20)):
                sampler.set_phase(phase)
                for index in range(count):
                    _deadline(request)
                    selected = index % len(timing_inputs)
                    physical, observed = timing_inputs[selected]
                    _sync(device)
                    if phase == "timed" and device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    trial_started = time.perf_counter()
                    prediction = route.predict_physical(physical, observed)
                    _sync(device)
                    elapsed = time.perf_counter()-trial_started
                    work = check_work(route, batch)
                    if prediction.shape != (batch, task["H"], len(task["columns"])) or not np.isfinite(prediction).all():
                        raise ValueError("Timed physical output is invalid")
                    memory = _memory(device)
                    trials.append(dict(phase=phase, trial=index, whole_seconds=elapsed,
                        origins=bundles[selected].tolist(), batch_size=batch,
                        prediction_sha256=common.array_identity(prediction)["sha256"],
                        peak_allocated_MiB=memory["peak_allocated_MiB"] if phase == "timed" else None,
                        peak_reserved_MiB=memory["peak_reserved_MiB"] if phase == "timed" else None,
                        CPU_RSS_MiB=psutil.Process().memory_info().rss/1024**2,
                        process_GPU_MiB=sampler.latest_gpu_mib, **work))
                    if trial_sink is not None:
                        trial_sink(trials[-1])
        finally:
            sampling = sampler.stop()
        timed = [row for row in trials if row["phase"] == "timed"]
        row = dict(status="COMPLETE", cost_valid=True, cost_id=request["cost_id"],
            dataset=request["dataset"], accuracy_run_id=request["accuracy_run_id"],
            family=route.family, mode=getattr(route.model, "raw_context_mode", None),
            schedule=getattr(route.model, "raw_context_schedule", "native"),
            operation=getattr(route, "operation", None), device=device,
            batch_size=batch, B=batch, C=len(task["columns"]), K=route.latent_k,
            L=task["L"], H=task["H"], warmups=10, timed_trials=20,
            row_microbatch=128, parity_status=parity["status"],
            MSE=parity["independent_metrics"]["mse"], MAE=parity["independent_metrics"]["mae"],
            **latency_statistics(trials, batch), **loaded_resident,
            baseline_allocated_MiB=baseline["allocated_MiB"], baseline_reserved_MiB=baseline["reserved_MiB"],
            model_loaded_CPU_RSS_MiB=loaded_rss,
            peak_allocated_MiB=max((v["peak_allocated_MiB"] for v in timed if v["peak_allocated_MiB"] is not None), default=None),
            peak_reserved_MiB=max((v["peak_reserved_MiB"] for v in timed if v["peak_reserved_MiB"] is not None), default=None),
            F_calls=timed[0]["F_calls"], F_logical_streams=timed[0]["F_logical_streams"],
            F_physical_streams=timed[0]["F_physical_streams"], F_invocations=timed[0]["F_invocations"],
            loaded_F_state_sha256=getattr(route.pool.foundation, "initial_hash", "NO_FOUNDATION_USED"),
            model_load_seconds=load_seconds, foundation_load_seconds=route.pool.load_seconds,
            input_extraction_seconds=extraction_seconds, full_V_parity_seconds=parity_seconds,
            lora_merge_parity_seconds=merge_seconds, lora_merge_parity=merge,
            process_pid=os.getpid(), process_create_time=psutil.Process().create_time(),
            fresh_process_cli=request.get("_fresh_process_cli", False),
            process_scope="fresh CLI worker, one route load" if request.get("_fresh_process_cli") else "in-process measurement, one route load",
            CPU_RSS_sampled_peak_MiB=sampling["CPU_RSS_sampled_peak_MiB"],
            process_GPU_sampled_peak_MiB=sampling["process_GPU_sampled_peak_MiB"],
            process_GPU_sampling_status=sampling["process_GPU_sampling_status"],
            process_GPU_sampling_reason=sampling["process_GPU_sampling_reason"],
            memory_layer="pytorch_allocated_tensor" if device.startswith("cuda") else "cpu_RSS",
            comparison_scope="CPU_RIDGE_CONTEXT" if route.family == "RAW_RIDGE" else "GPU" if device.startswith("cuda") else "CPU",
            p95_scope="20-trial descriptive quantile; no service tail guarantee",
            baseline_scope="after one-time LoRA merge, before whole-V parity and warmups",
            peak_reset_scope="after warmup and before each timed trial; summary maximum across timed trials",
            timing_scope=PROTOCOL["whole_path"], cached_forecast_reads_in_timed_path=0,
            stage_probe_trials=0, protocol=PROTOCOL)
        return dict(status="COMPLETE", row=row, summary=row, trials=trials, parity=parity,
                    sampling=sampling, timing_origin_bundles=[v.tolist() for v in bundles])
    finally:
        route.close()


def evaluate(request, *, trial_sink=None):
    request = validate_request(request)
    ctx = common.context(request["asset_root"], device=request["device"])
    _deadline(request)
    ctx["scientific_source_hashes"] = common.model_sources(ctx["repo"])
    sources = producer_sources(ctx)
    task = common.load_task(ctx, request["dataset"])
    reference = _reference(request, ctx)
    result = evaluate_inputs(request, ctx, task,
        lambda: make_route(ctx, task, request["route_spec"], device=request["device"]),
        reference=reference, trial_sink=trial_sink)
    if sources!=producer_sources(ctx):
        raise ValueError('Cost producer source changed during its fresh process')
    result.update(request_contract_sha256=request_identity(request),route_spec=request['route_spec'],
                  prediction_reference_sha256=request.get('prediction_sha256'),producer_sources=sources,
                  producer_environment=dict(python=sys.version.split()[0],numpy=np.__version__,torch=torch.__version__,
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
    if request.get('_request_path'):
        result['request_binding'] = common.binding(request['_request_path'],ctx)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deadline-unix", type=float,default=1e12)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_suffix(".trials.csv").exists():
        raise FileExistsError("Preserve the existing cost attempt; use a fresh output identity")
    request = common.read(args.request)
    request["_fresh_process_cli"] = True
    request['_request_path'] = str(args.request.resolve())
    request['_deadline_unix'] = args.deadline_unix
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trial_path = args.output.with_suffix(".trials.csv")
    with trial_path.open("x", encoding="utf8", newline="") as stream:
        writer = None
        def preserve_trial(row):
            nonlocal writer
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
            stream.flush()
            os.fsync(stream.fileno())
        try:
            result = evaluate(request, trial_sink=preserve_trial)
        except Exception as error:
            stream.flush()
            common.sealed_json(args.output, dict(status="INVALID", cost_id=request.get("cost_id"),
                error_type=type(error).__name__, error=str(error), process_pid=os.getpid(),
                process_create_time=psutil.Process().create_time(), trials_sha256=common.sha(trial_path)))
            raise
    result["trials_sha256"] = common.sha(trial_path)
    common.sealed_json(args.output, result)
    print(json.dumps(dict(status=result["status"], cost_id=request["cost_id"])))


if __name__ == "__main__":
    main()
