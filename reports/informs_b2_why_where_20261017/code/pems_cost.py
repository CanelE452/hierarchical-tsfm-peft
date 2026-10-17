"""Caller-controlled PeMS C32 selected-route cost measurement.

Import/default invocation reads no inputs and creates no model. Run each of the
four B0/B2 x B1/B8 conditions serially in a fresh CPU2 process after training.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


NAMESPACE = "informs_b2_why_where_20261017"
ORIGINAL = "priority_pems_subset_20261016_v1"
PROTOCOL = dict(device="cpu", CPU_threads=2, warmups=10, timed_trials=20,
                stage_trials=20, actual_trial_rows=50, row_microbatch=128,
                physical_output_and_inverse_dtype="float64",
                training_peak_included=False, stage_probe_separate_intrusive_scope=True)
ATOL, RTOL = 1e-5, 1e-4


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def pin_protocol(repo):
    """Reconcile original same-C32 Raw/G receipts without rerunning either."""
    campaign = Path(repo).resolve() / "research/informs_broad_validation_20261015_v1"
    receipts = []
    for family in ("raw", "direct_g"):
        for batch in (1, 8):
            name = f"pems_c32_{family}_cost_b{batch}_root_v1"
            relative = Path(ORIGINAL) / "diagnostics" / name / "COST_RECEIPT.json"
            row = read(campaign / relative)
            if row["status"] != "COMPLETE_PROTOCOL" or any(row.get(k) != v for k, v in PROTOCOL.items()):
                raise ValueError("Original same-C32 cost protocol differs: " + name)
            origins = [12499 + 12*i for i in range(batch)]
            if row["origins"] != origins or row["batch_size"] != batch:
                raise ValueError("Original fixed cost histories differ")
            if row["data_sha256"] != "6e0125acbbc93d4009e45185767d674929593de832c1da5d9ef6d3c043c602ee":
                raise ValueError("Original C32 identity differs")
            expected_modes = {"rescore", "replay"} | ({"raw_parity"} if family == "raw" else set())
            if set(row["validation_result_sha256"]) != expected_modes or not row["quiet_proof_sha256"]:
                raise ValueError("Original completed-route prerequisite bindings missing")
            if any(item["trials"] != 20 for item in row["summaries"]) or len(row["summaries"]) != 2:
                raise ValueError("Original timed/stage trial count differs")
            receipts.append(dict(family=family, batch=batch, relative_path=relative.as_posix(),
                  sha256=sha(campaign / relative), origins=origins, data_sha256=row["data_sha256"],
                  physical_input_sha256=row["physical_input_sha256"],
                  observation_mask_sha256=row["observation_mask_sha256"],
                  original_trials_sha256=row["trials_sha256"],
                  original_validation_result_sha256=row["validation_result_sha256"],
                  raw_G_values_remeasured=False))
    for batch in (1, 8):
        pair = [r for r in receipts if r["batch"] == batch]
        for key in ("origins", "data_sha256", "physical_input_sha256", "observation_mask_sha256"):
            if pair[0][key] != pair[1][key]:
                raise ValueError("Original Raw/G histories or C32 identity do not match")
    sources = {name: sha(campaign / name) for name in ("deployment.py", "measure_costs.py", "fit_base.py",
                                                                    "data_contract.py", "training.py", "gpu_runtime.py")}
    return dict(schema="PEMS_NEW_FINITE_COST_PROTOCOL_v1", status="PINNED_METADATA_ONLY",
                conditions=[dict(family=f, batch=b) for f in ("B0", "B2") for b in (1, 8)],
                protocol=PROTOCOL, original_receipts=receipts, original_scientific_sources=sources,
                parity=dict(atol=ATOL, rtol=RTOL, scope="first fixed 1 or 8 selected V origins; actual physical route normalized back against saved selected V"),
                serial_fresh_process_required=True, max_model_workers_during_cost=1,
                resource_limits=dict(CPU_threads=2, process_RSS_bytes=3*1024**3, new_GPU_seconds=0),
                actual_measurements_executed=0, original_deadlines_or_admission_modified=False,
                full_pipeline="physical input/mask -> causal forward-fill -> normalization -> P/E -> frozen F row128 -> lag D/current anchor -> FP64 inverse -> CPU return",
                load_time_separate_from_steady_state=True, raw_G_remeasured=False,
                original_C170_cost_reused=False, protected_T_numeric_parse_count=0)


def _quiet(psutil, asset):
    """Supplement the caller's phase guard with observed same-campaign workers."""
    names = {"mechanism_runner.py", "linear_probe.py", "pems_cost.py",
             "CAP_run_initialization_audit.py", "PEMS_run_restore_audit.py"}
    current = psutil.Process()
    own_arguments = current.cmdline()[1:]
    expected_shim = (Path(asset).resolve() / ".venv/Scripts/python.exe").resolve()
    wrappers = {}
    for ancestor in current.parents():
        if (Path(ancestor.exe()).resolve() == expected_shim
                and ancestor.cmdline()[1:] == own_arguments
                and any(Path(arg).name == "pems_cost.py" for arg in own_arguments)):
            wrappers[ancestor.pid] = ancestor.create_time()
    competing = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        if process.info["pid"] == os.getpid() or "python" not in (process.info["name"] or "").lower():
            continue
        argv = process.info["cmdline"] or []
        if (process.info["pid"] in wrappers
                and wrappers[process.info["pid"]] == process.info["create_time"]):
            continue
        if any(Path(arg).name in names for arg in argv):
            competing.append(dict(pid=process.info["pid"], create_time=process.info["create_time"],
                                  script=next(Path(arg).name for arg in argv if Path(arg).name in names)))
    if competing:
        raise RuntimeError("Same-campaign model worker overlaps exclusive cost phase")
    return dict(checked_unix=time.time(), current_pid=os.getpid(), observed_competing_model_workers=competing,
                excluded_own_venv_shim_ancestors=[dict(pid=pid, create_time=created,
                    binary_role="asset_relative_venv_python_shim", command_arguments_match_current=True)
                    for pid, created in wrappers.items()],
                scope="recognized same-campaign scientific worker scripts; caller also owns global phase guard")


def run_cost(family, batch, repo, asset, deadline):
    """Measure one explicit new selected B0/B2 route; return FAILED with NA times on failure."""
    if family not in ("B0", "B2") or batch not in (1, 8):
        raise ValueError("Only original B0/B2 and B1/B8 are authorized")
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or time.time() >= deadline:
        raise ValueError("Live finite new-campaign deadline required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "2":
            raise ValueError("Set all four CPU thread environment limits to 2 before Python starts")
    repo, asset = Path(repo).resolve(), Path(asset).resolve()
    campaign = repo / "research/informs_broad_validation_20261015_v1"
    public = repo / "reports" / NAMESPACE / "costs" / f"pems_{family}_b{batch}"
    private = asset / ".cache" / NAMESPACE
    private_cost = private / "PEMS_cost" / f"{family}_b{batch}"
    public.mkdir(parents=True, exist_ok=False)
    private_cost.mkdir(parents=True, exist_ok=False)
    started = time.time()
    result = dict(status="FAILED_COST_PROTOCOL", family=family, batch_size=batch,
                  median_seconds=None, p90_seconds=None, timing_status="NA", summaries=[],
                  actual_training_updates=0, protected_T_numeric_parse_count=0)
    pool = route = None
    try:
        sys.path.insert(0, str(campaign))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import torch
        import numpy as np
        import psutil
        import deployment
        import measure_costs
        import fit_base as fit
        import data_contract as data
        import training
        import pems_resume
        torch.set_num_threads(2)
        quiet_before = _quiet(psutil, asset)
        protocol = pin_protocol(repo)
        _new(public / "PROTOCOL.json", protocol)
        source_bindings = {"pems_cost.py": sha(__file__), "pems_resume.py": sha(Path(pems_resume.__file__))}
        source_bindings.update(protocol["original_scientific_sources"])
        artifact = private / "runs" / f"pems_{family}" / "selected_bundle.pt"
        prediction = artifact.parent / "selected_V.npz"
        selected_metrics = repo / "reports" / NAMESPACE / "runs" / f"pems_{family}" / "SELECTED_V_RESULT.json"
        bindings = dict(artifact_sha256=sha(artifact), prediction_archive_sha256=sha(prediction),
                        selected_metrics_sha256=sha(selected_metrics))
        metrics = read(selected_metrics)
        if metrics.get("status") != "COMPLETE_SELECTED_V" or metrics.get("C") != 32 or metrics.get("K") != 8:
            raise ValueError("Complete original C32 selected V metrics required")
        bundle = torch.load(artifact, map_location="cpu", weights_only=False)
        for key in ("model", "normalization", "spec", "contract", "model_state_sha"):
            if key not in bundle:
                raise ValueError("Selected bundle missing required deployment field: " + key)
        if (bundle["spec"]["family"] != family or bundle["spec"]["seed"] != 92601
                or bundle["normalization"]["fit_end"] != 10713):
            raise ValueError("Original family/seed/fresh A60 fit boundary differs")
        actual_state_sha = training.fingerprint({k: data.array_sha256(v.numpy()) for k, v in bundle["model"].items()})
        if actual_state_sha != bundle["model_state_sha"]:
            raise ValueError("Selected adapter tensor hash differs")
        task, access, old_plan = pems_resume.load_task(campaign, asset / ".cache/informs_broad_validation_20261015_v1")
        _new(private_cost / "PREFIX_ACCESS.json", access)
        origins = data.origins(pems_resume.BOUNDS["V_select"], 512, 12, 12)
        if len(origins) != 148 or origins[:batch].tolist() != [12499+12*i for i in range(batch)]:
            raise ValueError("Original selected V148/fixed cost histories differ")
        with np.load(prediction, allow_pickle=False) as archive:
            if set(archive.files) != {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}:
                raise ValueError("Selected V archive schema differs")
            arrays = {key: archive[key] for key in archive.files}
        if (not np.array_equal(arrays["origins"], origins) or not np.array_equal(arrays["channel_ids"], np.asarray(task["columns"], str))
                or arrays["pred_normalized"].shape != (148, 12, 32)):
            raise ValueError("Selected V origin/channel/prediction axes differ")
        physical = np.stack([task["physical"][int(o)-512:int(o)] for o in origins[:batch]])
        observed = np.stack([task["observed"][int(o)-512:int(o)] for o in origins[:batch]])
        input_sha, mask_sha = sha_bytes(physical), sha_bytes(observed)
        original_batch = next(row for row in protocol["original_receipts"] if row["batch"] == batch)
        if input_sha != original_batch["physical_input_sha256"] or mask_sha != original_batch["observation_mask_sha256"]:
            raise ValueError("New cost input/mask do not equal original same-C32 Raw/G cost histories")
        process = psutil.Process()
        class Lease:
            def check(self):
                if time.time() >= deadline:
                    raise TimeoutError("New cost deadline reached")
                if process.memory_info().rss > 3*1024**3:
                    raise MemoryError("Cost worker RSS exceeds3GiB")
                if torch.get_num_threads() != 2 or torch.cuda.is_initialized():
                    raise RuntimeError("CPU2/GPU0 cost policy differs")
                _quiet(psutil, asset)
        lease = Lease()
        lease.check()
        pool = deployment.FoundationPool(asset / old_plan["model_asset_relative_path"], "cpu")
        route_spec = dict(run_id=f"mechanism_pems_{family}_selected_cost_b{batch}", family=family, latent_k=8,
                          canonical_fit_end=10713, device="cpu", row_microbatch=128,
                          artifact_cache_relative_path=artifact.relative_to(private).as_posix(),
                          artifact_sha256=bindings["artifact_sha256"])
        route = deployment.Route(route_spec, task, private, pool.model_path, "cpu", pool)
        if fit.scaler_fingerprint(route.mean, route.std) != bundle["contract"]["scaler_sha"]:
            raise ValueError("Route scaler differs from selected artifact")
        value = route.predict_physical(physical, observed)
        normalized = ((value-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32)
        expected = arrays["pred_normalized"][:batch]
        passed = bool(np.allclose(normalized, expected, atol=ATOL, rtol=RTOL))
        parity = dict(status="PASS" if passed else "FAILED", atol=ATOL, rtol=RTOL,
                      max_abs=float(np.max(np.abs(normalized.astype(np.float64)-expected.astype(np.float64)))),
                      origins=origins[:batch].tolist(), actual_physical_forward=True,
                      F_counters={k: route.last_receipt[k] for k in ("F_logical_streams", "F_physical_streams", "F_calls")},
                      saved_V_pred_used_only_for_parity=True)
        _new(public / "FORWARD_PARITY.json", parity)
        if not passed:
            result.update(status="FAILED_COST_PARITY", forward_parity=parity)
            raise ValueError("Physical deployment route does not reproduce saved selected V; timing remains NA")
        rows = measure_costs.measure_balanced([route], physical, observed, warmups=10, timed_trials=20,
                                             stage_trials=20, lease=lease)
        expected_counters = (batch*8, batch*8, math.ceil(batch*8/128))
        if len(rows) != 50 or any((r["F_logical_streams"], r["F_physical_streams"], r["F_calls"]) != expected_counters for r in rows):
            raise ValueError("Actual cost trials/F geometry differs")
        memory = route.memory_state()
        summaries = measure_costs.summarize(rows)
        trials = public / "COST_TRIALS.csv.gz"
        measure_costs._write_trials(trials, rows)
        if any(sha(path) != bindings[key] for path, key in ((artifact, "artifact_sha256"), (prediction, "prediction_archive_sha256"), (selected_metrics, "selected_metrics_sha256"))):
            raise ValueError("Accuracy evidence changed during cost measurement")
        for name, digest in protocol["original_scientific_sources"].items():
            if sha(campaign / name) != digest:
                raise ValueError("Original scientific source changed during cost measurement")
        pool.foundation.assert_frozen_unchanged()
        timed = next(item for item in summaries if item["phase"] == "timed")
        result.update(status="COMPLETE_PROTOCOL", timing_status="MEASURED", median_seconds=timed["median_seconds"],
                      p90_seconds=timed["p90_seconds"], summaries=summaries, **PROTOCOL,
                      **bindings, model_state_sha256=actual_state_sha, route_spec=route_spec,
                      data_sha256=task["data_sha256"], scaler_sha256=bundle["contract"]["scaler_sha"],
                      physical_input_sha256=input_sha, observation_mask_sha256=mask_sha,
                      origins=origins[:batch].tolist(), forward_parity=parity, memory=memory,
                      model_load_seconds=route.load_seconds,
                      foundation_loading_seconds_included_in_route_load=pool.load_seconds,
                      model_load_scope="Route constructor includes its FoundationPool.get loading; no added double count",
                      cold_load_in_whole_trial=False, foundation_instances_created=pool.instances_created,
                      F_counters_per_trial=dict(streams=batch*8, physical_streams=batch*8, physical_calls=expected_counters[2]),
                      trials_sha256=sha(trials), source_sha256=source_bindings, quiet_before=quiet_before,
                      quiet_after=_quiet(psutil, asset), process_RSS_bytes=process.memory_info().rss,
                      GPU_initialized=torch.cuda.is_initialized(), actual_raw_G_remeasurements=0,
                      cached_forecast_reads_for_timing=0)
    except Exception as error:
        result.update(error_class=type(error).__name__, error_message=str(error))
        _new(private_cost / "FAILURE.json", dict(error_class=type(error).__name__, error_message=str(error), traceback=traceback.format_exc()))
    finally:
        try:
            if route is not None:
                route.close()
            if pool is not None:
                pool.close()
        except Exception as error:
            result.update(status="FAILED_COST_CLOSE", timing_status="NA", median_seconds=None, p90_seconds=None,
                          summaries=[], close_error_class=type(error).__name__)
        result["elapsed_wall_seconds"] = time.time()-started
        _new(public / "RESULT.json", result)
    return result


def sha_bytes(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--family", choices=("B0", "B2"))
    parser.add_argument("--batch", type=int, choices=(1, 8))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="CALLER_CONTROLLED_COST_API", input_reads=0, writes=0,
                              torch_imported="torch" in sys.modules, measurements_executed=0)))
        return 0
    if any(getattr(args, key) is None for key in ("family", "batch", "repo", "asset", "deadline")):
        parser.error("One family/batch, roots and finite new deadline required")
    result = run_cost(args.family, args.batch, args.repo, args.asset, args.deadline)
    print(json.dumps({k: result[k] for k in ("status", "family", "batch_size", "timing_status")}))
    return 0 if result["status"] == "COMPLETE_PROTOCOL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
