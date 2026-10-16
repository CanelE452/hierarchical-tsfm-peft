"""Explicit CPU replay/cost for two exact legacy baselines; default reads nothing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

NAME = "priority_b2_validation_20261016_v1"
FOUNDATION = "c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
SCALER = "5f0c7d6aeddcfe9577e381375d36cadc47d2572469118da73e6ad837031fa2b9"
PREFLIGHT_REL = "diagnostics/portable_route_cpu_preflight_20261016T0139/ROUTE_RESULTS.json"
PREFLIGHT_SHA = "f94b45f45c01da2374388b74cabfb757cd00356be4e4d39227182c9d2f3eaefb"
ORIGINS6 = [14675, 14723, 14915, 15587, 15683, 16691]
PINS = {
    "PCA": dict(run_id="primary_jena_E1_PCA_FIXED_BOLT_k6_s92601", family="PCA_FIXED_BOLT", latent_k=6,
        artifact_sha256="86b6373002acf4a65f965a3ee289d5603d0ed01e18e8a07053932852cbaa859c",
        model_state_sha="e5717391dd5fcc24844b68a5c96dbe074b74b7c11b6ccb81ef5cb342a7964f21",
        fit_receipt_sha256="650fbfa32d3632aceab386a918746a288c77611620db1e054952d4367b91f562",
        V_manifest_sha256="80b70fb2af4832f05093223d54654423a27b154cff4dd2f3beefd51369641f34",
        V_archive_sha256="7b7a7c16a5e6a7b580be553546ff16bdc515e6460b3022a1b3e548cd28c25dfd",
        prediction_model_sha="78854f37a34b7c8ca3b669f905a58329c4af4bc1394d76697231bacfc39ec946",
        canonical_fingerprint="ae48aaf5da6315e0dc481f9caf91982d18dd1f0f2970f957a19240f7988d7179"),
    "LORA": dict(run_id="primary_jena_E1_LORA_R8_k0_s92601", family="LORA", latent_k=21,
        artifact_sha256="aa4e70a154fe1f9906fdac50d1c6db4e62fd04d2d77283835bcc55a69e62cb2b",
        model_state_sha="e48d9f9695752414ef9bb3ccbfc5be34f25081574bf7d0a09b4abff122364b79",
        fit_receipt_sha256="f1817562ba11d0d09346502620a0314faba1bccc271adf72edb1d6201938228e",
        V_manifest_sha256="661dd5912cd18e3b0558ec31042fe47995bfc5ca260a14338f6cacce681b557d",
        V_archive_sha256="3f3e32aba3e0d735da91fcb19a76541f83a701f3e16409c3b972df3483014074",
        prediction_model_sha="0888a1a03b8710242da3f04a99e5ac35d54d09d72062153671da2d0c4b5e3f0a",
        canonical_fingerprint="a8a8a1f383e61141d80f14d0d1ffc0098cd04ef14d4ac7ecacbafb50fa0f2919")}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def spec_draft():
    return dict(schema="LEGACY_BASELINE_CPU_ADDITIVE_SPEC_v1", status="DRAFT_NOT_EXECUTABLE",
        namespace=NAME, priority_plan_sha256=None, entrypoint_sha256=None,
        foundation_tensor_sha256=FOUNDATION, refit_scaler_sha256=SCALER,
        preflight=dict(public_relative_path=PREFLIGHT_REL, sha256=PREFLIGHT_SHA),
        routes={key: dict(value) for key, value in PINS.items()},
        permitted_modes=["replay", "cost"], no_training=True, new_T_numeric_values_parsed=0,
        CPU_threads=2, F_row_microbatch=128, checkpoint_inputs=False,
        replay_origins=ORIGINS6, replay_microbatches=[4, 2], atol=1e-5, rtol=1e-4,
        cost_batches=[1, 8], warmups=10, timed_trials=20, stage_trials=20,
        original_PREREG_PLAN_must_not_be_changed=True,
        ROOT_external_deadline_RSS_guard_and_quiet_cost_window_required=True)


def validate_additive_spec(spec, plan_sha, entry_sha):
    expected = spec_draft()
    expected.update(status="SEALED_BY_ROOT", priority_plan_sha256=plan_sha, entrypoint_sha256=entry_sha)
    if spec != expected:
        raise ValueError("An exact separately ROOT-sealed legacy baseline specification is required")


def canonical_spec(baseline, receipt, source, manifest):
    pin = PINS[baseline]
    run_id = pin["run_id"]
    if (receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != run_id
            or receipt.get("family") != pin["family"] or receipt.get("artifact_sha256") != pin["artifact_sha256"]
            or receipt.get("model_state_sha") != pin["model_state_sha"] or receipt.get("refit_fit_end") != 12579
            or receipt.get("refit_scaler_sha") != SCALER or receipt.get("search_fit_end") != 10482
            or receipt.get("epochs") != [.25] or receipt.get("training_status") != "LIMITED_SCREEN"):
        raise ValueError("Exact original completed limited-screen fit receipt required")
    spec = source["spec"]
    if (tuple(spec.get(key) for key in ("run_id", "dataset", "family", "latent_k", "seed", "fold"))
            != (run_id, "jena", pin["family"], pin["latent_k"], 92601, "primary")
            or source.get("model_sha") != FOUNDATION or source.get("source_sha") != receipt.get("source_sha")):
        raise ValueError("Original baseline recipe/foundation/source differs")
    if (manifest.get("run_id") != run_id or manifest.get("split") != "V_select"
            or manifest.get("cohort") != "jena:primary:V_select:stride48"
            or manifest.get("model_sha") != pin["prediction_model_sha"] or manifest.get("scaler_sha") != SCALER
            or manifest.get("data_sha") != source.get("data_sha") or manifest.get("archive_sha256") != pin["V_archive_sha256"]
            or manifest.get("metadata", {}).get("artifact_sha256") != pin["artifact_sha256"]):
        raise ValueError("Original complete V accuracy binding differs")
    relative = "runs/" + run_id + "/selected_bundle.pt"
    if receipt.get("artifact_cache_relative_path") != relative:
        raise ValueError("Original selected bundle path differs")
    result = dict(run_id=run_id, dataset="jena", family=pin["family"], fold="primary",
        artifact_cache_relative_path=relative, artifact_sha256=pin["artifact_sha256"], latent_k=pin["latent_k"],
        canonical_fit_end=12579, foundation_sha=FOUNDATION, expected_prediction_model_sha=pin["prediction_model_sha"],
        fit_receipt_public_relative_path="runs/" + run_id + "/BASE_FIT_RECEIPT.json",
        fit_receipt_sha256=pin["fit_receipt_sha256"],
        standard_V_binding=dict(model_sha=pin["prediction_model_sha"], archive_sha256=pin["V_archive_sha256"],
            manifest_public_relative_path="predictions/" + run_id + "/V_select/PREDICTION_MANIFEST.json",
            manifest_sha256=pin["V_manifest_sha256"]), model_size="small")
    if fingerprint(result) != pin["canonical_fingerprint"]:
        raise ValueError("Canonical route differs from its actual original CPU preflight")
    return result


def check_trial_counters(rows, baseline, batch):
    streams = PINS[baseline]["latent_k"] * batch
    calls = (streams + 127) // 128
    expected = dict(warmup=10, timed=20, stage_probe=20)
    for phase, count in expected.items():
        group = [row for row in rows if row["phase"] == phase]
        if len(group) != count:
            raise AssertionError("Incomplete original cost phase: " + phase)
    if len(rows) != 50:
        raise AssertionError("Expected exactly 50 original cost trial rows")
    for row in rows:
        if (row["batch_size"] != batch or row["F_logical_streams"] != streams
                or row["F_physical_streams"] != streams or row["F_calls"] != calls or row["row_microbatch"] != 128):
            raise AssertionError("Actual row128 baseline stream/call contract differs")


def close_and_verify(route, foundation, state_hash):
    route.close()
    actual = state_hash(foundation.model)
    if actual != FOUNDATION or any(p.grad is not None or p.requires_grad for p in foundation.model.parameters()):
        raise AssertionError("Original pinned frozen F was not exactly restored at route close")
    return dict(status="PASS_EXACT_HASH", foundation_state_sha256=actual,
                F_parameter_grad_none=True, F_requires_grad_false=True)


def load_runner(public, plan_sha, plan_path):
    path = Path(public) / NAME / "PREREG_PLAN.json"
    if plan_path is not None and Path(plan_path).resolve() != path.resolve():
        raise ValueError("Use the existing priority PREREG_PLAN")
    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(path) != plan_sha:
        raise ValueError("Priority plan SHA differs")
    plan = json.loads(path.read_text(encoding="utf-8"))
    source = Path(public) / NAME / "priority_runner.py"
    if digest(source) != plan["runner_sha256"]:
        raise ValueError("Priority runner must match its existing sealed pin before import")
    spec = importlib.util.spec_from_file_location("legacy_priority_guard", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def metadata(public, cache, plan_sha, spec_path, spec_sha, baseline, process_api, plan_path=None):
    runner = load_runner(public, plan_sha, plan_path)
    plan = runner.metadata(public, cache, plan_sha, process_api, plan_path)
    if not spec_sha or runner.sha(spec_path) != spec_sha:
        raise ValueError("Explicit additive baseline spec SHA required")
    spec = runner.read(spec_path)
    validate_additive_spec(spec, plan_sha, runner.sha(__file__))
    if plan["foundation_tensor_sha256"] != FOUNDATION or plan["task_contract"]["refit_scaler_sha256"] != SCALER:
        raise ValueError("Existing priority foundation/A60 scaler differs")
    preflight_path = runner.inside(public, PREFLIGHT_REL)
    if runner.sha(preflight_path) != PREFLIGHT_SHA:
        raise ValueError("Original actual CPU route-construction proof changed")
    pin, run_id = PINS[baseline], PINS[baseline]["run_id"]
    records = runner.read(preflight_path)
    old = next(row for row in records if row["run_id"] == run_id)
    if old["status"] != "ACTUAL_PORTABLE_ROUTE_CPU_VALID" or old["route_fingerprint"] != pin["canonical_fingerprint"]:
        raise ValueError("Original baseline has no exact preflight route fingerprint")
    folder = Path(public) / "runs" / run_id
    prediction = Path(public) / "predictions" / run_id / "V_select"
    paths = dict(receipt=folder / "BASE_FIT_RECEIPT.json", source=folder / "SOURCE_CONTRACT.json",
                 manifest=prediction / "PREDICTION_MANIFEST.json", reference=prediction / "REFERENCE_CONTRACT.json")
    for key, expected in (("receipt", pin["fit_receipt_sha256"]), ("manifest", pin["V_manifest_sha256"])):
        if runner.sha(paths[key]) != expected:
            raise ValueError("Original exact baseline metadata changed: " + key)
    values = {key: runner.read(path) for key, path in paths.items()}
    canonical = canonical_spec(baseline, values["receipt"], values["source"], values["manifest"])
    if values["source"]["files"] != {name: plan["source39"][name] for name in values["source"]["files"]}:
        raise ValueError("Baseline scientific source files differ from pinned source39")
    r4 = runner.bound_R4(plan, public, cache)
    for name in ("target_normalized", "observed", "origins", "channel_ids"):
        if values["manifest"]["arrays"][name] != r4["arrays"][name] or values["reference"]["arrays"][name] != r4["arrays"][name]:
            raise ValueError("Baseline V truth/mask/origin/channel differs from current priority cohort")
    artifact = runner.inside(cache, canonical["artifact_cache_relative_path"])
    archive = runner.inside(cache, values["manifest"]["cache_relative_path"])
    if runner.sha(artifact) != pin["artifact_sha256"] or runner.sha(archive) != pin["V_archive_sha256"]:
        raise ValueError("Original selected adapter or full V archive bytes changed")
    pins = {str(path): runner.sha(path) for path in (*paths.values(), artifact, archive, preflight_path, Path(spec_path), Path(__file__))}
    return runner, plan, canonical, values, pins


def load_saved_V(archive_path, manifest, reference, context):
    np = context["np"]
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}:
            raise ValueError("Only the original five V prediction archive members")
        arrays = {key: archive[key] for key in archive.files}
    for name, value in arrays.items():
        if context["evaluation"].array_fingerprint(value) != manifest["arrays"][name]:
            raise ValueError("Original full V saved array fingerprint differs")
    for name in ("target_normalized", "observed", "origins", "channel_ids"):
        if manifest["arrays"][name] != reference["arrays"][name]:
            raise ValueError("Original full V cohort differs")
    return arrays


def actual_route(canonical, task, cache, asset, plan, context):
    import deployment
    import gpu_runtime
    route = deployment.Route(dict(canonical, row_microbatch=128), task, cache,
        Path(asset) / plan["model_asset_relative_path"], "cpu")
    try:
        foundation = route.pool.foundation
        bundle = route.bundle
        pin = next(value for value in PINS.values() if value["run_id"] == canonical["run_id"])
        if (route.device != "cpu" or route.pool.instances_created != 1 or foundation.row_microbatch != 128
                or foundation.checkpoint_inputs is not False or foundation.initial_hash != FOUNDATION
                or foundation.instance_norm_source_sha256 != plan["instance_norm_source_sha256"]
                or gpu_runtime.state_hash(foundation.model) != FOUNDATION
                or context["fit"].scaler_fingerprint(route.mean, route.std) != SCALER
                or bundle["model_state_sha"] != pin["model_state_sha"]
                or context["fit"]._source_contract()[0] != bundle["contract"]["source_sha"]
                or bundle["normalization"]["train_end"] != 12579):
            raise ValueError("Actual loaded baseline/foundation/scaler differs")
        if pin["family"] == "LORA" and (bundle["spec"]["lora_rank"] != 8 or bundle["spec"]["lora_alpha"] != 16 or bundle["selected_search_epoch"] != .25):
            raise ValueError("This baseline must remain the exact .25epoch rank8/alpha16 LoRA")
    except Exception:
        route.close()
        raise
    return route, foundation, gpu_runtime


def run_scientific(mode, baseline, batch, canonical, values, task, bounds, context, public, cache, asset, plan, runner, output, deadline):
    import deployment
    import measure_costs
    import psutil
    np, torch = context["np"], context["torch"]
    arrays = None
    if mode == "replay":
        arrays = load_saved_V(runner.inside(cache, values["manifest"]["cache_relative_path"]), values["manifest"], values["reference"], context)
        selected = arrays["origins"][runner.replay_indices(len(arrays["origins"]), np)]
        if selected.tolist() != ORIGINS6:
            raise ValueError("Exact original six replay origins required")
        origins = selected[:4]
    else:
        origins = context["data"].origins(bounds["V_select"], 512, 48, 48)[:batch]
        if len(origins) != batch:
            raise ValueError("Exact first V histories required for batch1/8 cost")
    physical = np.stack([task["physical"][int(o)-512:int(o)] for o in origins])
    observed = np.stack([task["observed"][int(o)-512:int(o)] for o in origins])
    route, foundation, runtime = actual_route(canonical, task, cache, asset, plan, context)
    parity = dict(status="NOT_LORA")
    class DeadlineCheck:
        def check(self): runner.check_deadline(deadline)
    try:
        runner.check_deadline(deadline)
        parity_before = route.pool.counters()
        if baseline == "LORA":
            parity = route.check_lora_parity(physical, observed, atol=1e-5, rtol=1e-4)
            route.deactivate()
        parity_after = route.pool.counters()
        parity_counters = {key: parity_after[key] - parity_before[key] for key in parity_before}
        if mode == "replay":
            expected = arrays["pred_normalized"][runner.replay_indices(len(arrays["origins"]), np)]
            predictions, telemetry = [], []
            for offset in (0, 4):
                runner.check_deadline(deadline)
                oo = selected[offset:offset+4]
                physical = np.stack([task["physical"][int(o)-512:int(o)] for o in oo])
                observed = np.stack([task["observed"][int(o)-512:int(o)] for o in oo])
                raw = route.predict_physical(physical, observed)
                normalized = ((raw-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32)
                predictions.append(normalized)
                telemetry.append(dict(route.last_receipt))
            actual = np.concatenate(predictions)
            passed = bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-4))
            if sum(row["F_logical_streams"] for row in telemetry) != 6*PINS[baseline]["latent_k"] or sum(row["F_calls"] for row in telemetry) != 2:
                raise ValueError("Actual CPU6 replay F counts differ")
            result = dict(status="PASS_FRESH_CPU6_REPLAY" if passed else "FAIL_REPLAY_PARITY",
                checkpoint_forward_executed=True, origins=selected.tolist(), microbatches=[4,2],
                max_abs=float(np.abs(actual.astype(np.float64)-expected.astype(np.float64)).max()),
                atol=1e-5, rtol=1e-4, telemetry=telemetry, saved_V_array_hashes_checked=True,
                actual_F_streams=sum(row["F_logical_streams"] for row in telemetry),
                actual_F_calls=sum(row["F_calls"] for row in telemetry),
                independent_full_saved_V_rescore_executed=False,
                whole_V_scope="Existing V43 archive reused; byte and array hashes verified here; only six origins reinferred; no new whole-V accuracy calculation")
        else:
            rows = measure_costs.measure_balanced([route], physical, observed, warmups=10, timed_trials=20, stage_trials=20, lease=DeadlineCheck())
            check_trial_counters(rows, baseline, batch)
            memory = route.memory_state()
            current = psutil.Process().memory_info()
            memory.update(process_RSS_bytes=current.rss, OS_peak_working_set_bytes=getattr(current,"peak_wset",None))
            trials = output / "COST_TRIALS.csv.gz"
            measure_costs._write_trials(trials, rows)
            result = dict(status="COMPLETE_PROTOCOL", batch_size=batch, origins=list(map(int,origins)),
                warmups=10, timed_trials=20, stage_trials=20, summaries=measure_costs.summarize(rows), memory=memory,
                physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
                observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
                trials_sha256=runner.sha(trials), actual_trial_rows=len(rows), physical_output_dtype="float64",
                inverse_transform_dtype="float64", cached_forecast_or_Z_reads=0,
                stage_probe_is_intrusive_separate_scope=True, training_peak_included=False,
                fresh_process_single_route=True, original_measure_balanced_reused=True,
                independent_full_saved_V_rescore_executed=False,
                whole_V_scope="Original V archive SHA/manifest binding reused; V prediction arrays are not consumed by cost measurement")
        result.update(run_id=canonical["run_id"], artifact_sha256=PINS[baseline]["artifact_sha256"],
            accuracy_manifest_sha256=PINS[baseline]["V_manifest_sha256"], scaler_sha256=SCALER,
            scope="fresh_process_single_route", LoRA_actual_merged_unmerged_parity=parity, auxiliary_parity_F_counters=parity_counters,
            accuracy_binding=dict(run_id=canonical["run_id"], canonical_route=canonical,
                canonical_route_fingerprint=PINS[baseline]["canonical_fingerprint"],
                artifact_sha256=PINS[baseline]["artifact_sha256"], V_manifest_sha256=PINS[baseline]["V_manifest_sha256"],
                V_archive_sha256=PINS[baseline]["V_archive_sha256"], scaler_sha256=SCALER,
                canonical_route_reconstruction="Matches original actual CPU preflight fingerprint; original route_catalog not called on truncated Task"),
            foundation_instances_created=route.pool.instances_created, row_microbatch=128, checkpoint_inputs=False,
            family=PINS[baseline]["family"], effective_F_input_width=PINS[baseline]["latent_k"],
            original_training_status=values["receipt"]["training_status"],
            original_search_updates=values["receipt"]["search_updates"], original_refit_updates=values["receipt"]["refit_updates"],
            new_training=False, actual_new_optimizer_steps=0, device="cpu", CPU_threads=2,
            protected_T_target_operations=0, new_T_numeric_values_parsed=0,
            independent_generalization_claimed=False)
    finally:
        restoration = close_and_verify(route, foundation, runtime.state_hash)
    result["backbone_close"] = restoration
    result["RSS_bytes_after_close"] = psutil.Process().memory_info().rss
    runner.new(output / ("REPLAY_RECEIPT.json" if mode == "replay" else "COST_RECEIPT.json"), result)
    if not result["status"].startswith(("PASS", "COMPLETE")):
        raise AssertionError("Actual selected baseline CPU checkpoint parity failed")
    return result


def execute(args):
    import psutil
    public, cache, asset = (Path(getattr(args,key)).resolve() for key in ("public_root","cache_root","asset_root"))
    spec_path, output = Path(args.spec).resolve(), Path(args.output).resolve()
    runner, plan, canonical, values, pins = metadata(public,cache,args.plan_sha,spec_path,args.spec_sha,args.baseline,psutil,args.plan)
    runner.require_environment()
    if not args.deadline or not time.time() < args.deadline <= plan["deadline_unix"]:
        raise ValueError("Explicit live deadline within original priority plan required")
    if args.mode == "cost" and args.batch not in (1,8) or args.mode == "replay" and args.batch is not None:
        raise ValueError("Only cost accepts explicit batch1/8")
    root = public / NAME / "diagnostics"
    if output.exists() or output == root or not output.is_relative_to(root):
        raise ValueError("A fresh priority diagnostics output is required")
    runner.inside(public,output.relative_to(public))
    output.mkdir(parents=True,exist_ok=False)
    started = time.time()
    process = psutil.Process()
    cpu_before = process.cpu_times()
    context = result = None
    failed = None
    try:
        context = runner.load_context(public,plan,args.plan_sha,True)
        task,bounds,access = runner.load_task_prefix(asset,plan,context,public)
        runner.new(output/"PREFIX_ACCESS_RECEIPT.json",access)
        result = run_scientific(args.mode,args.baseline,args.batch,canonical,values,task,bounds,context,
            public,cache,asset,plan,runner,output,args.deadline)
        runner.check_deadline(args.deadline)
    except Exception as error:
        failed = type(error).__name__
        raise
    finally:
        metadata(public,cache,args.plan_sha,spec_path,args.spec_sha,args.baseline,psutil,args.plan)
        if any(runner.sha(path) != digest for path,digest in pins.items()):
            raise ValueError("Baseline inputs changed during CPU execution")
        if context is not None and context["torch"].cuda.is_initialized():
            raise RuntimeError("Legacy CPU baseline initialized CUDA")
        cpu_after = process.cpu_times()
        memory = process.memory_info()
        runner.new(output/"RESULT.json",dict(status="FAILED_PRESERVED" if failed else (result or {}).get("status","INCOMPLETE"),
            baseline=args.baseline,run_id=canonical["run_id"],mode=args.mode,scientific_result=result,error_class=failed,
            priority_plan_sha256=args.plan_sha,additive_spec_sha256=args.spec_sha,entrypoint_sha256=runner.sha(__file__),
            original_source39_and_immutable9_preserved=True,baseline_input_SHA_prepost_equal=True,
            elapsed_process_wall_seconds=time.time()-started,
            actual_CPU_process_seconds=cpu_after.user+cpu_after.system-cpu_before.user-cpu_before.system,
            RSS_bytes=memory.rss,OS_peak_working_set_bytes=getattr(memory,"peak_wset",None),
            actual_new_optimizer_steps=0,new_training_conditions=0,new_T_numeric_values_parsed=0,
            raw_values_published=False,independent_generalization_claimed=False,
            ROOT_external_deadline_RSS_guard_and_cost_quiet_window_required=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--print-spec-draft",action="store_true")
    for key in ("public-root","cache-root","asset-root","plan","plan-sha","spec","spec-sha","output"):
        parser.add_argument("--"+key)
    parser.add_argument("--baseline",choices=tuple(PINS))
    parser.add_argument("--mode",choices=("replay","cost"))
    parser.add_argument("--batch",type=int,choices=(1,8))
    parser.add_argument("--deadline",type=float)
    args = parser.parse_args(argv)
    if args.print_spec_draft:
        if args.execute: parser.error("Draft output cannot execute")
        print(json.dumps(spec_draft(),indent=2))
        return 0
    if not args.execute:
        print(json.dumps(dict(status="NOT_EXECUTED",input_reads=0,writes=0,torch_imported="torch" in sys.modules,
            numpy_imported="numpy" in sys.modules,actual_model_or_array_or_cost_execution=0)))
        return 0
    if not all((args.public_root,args.cache_root,args.asset_root,args.plan_sha,args.spec,args.spec_sha,
                args.output,args.baseline,args.mode,args.deadline)):
        parser.error("Execution requires explicit roots, priority plan SHA, additive spec/SHA, baseline, mode, deadline and fresh output")
    result = execute(args)
    print(json.dumps(dict(status=result["status"],baseline=args.baseline,mode=args.mode)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
