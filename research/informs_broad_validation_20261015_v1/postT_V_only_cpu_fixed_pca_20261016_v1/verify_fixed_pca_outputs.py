"""Fixed-PCA variable-task saved-V rescore or selected-checkpoint V subset replay.

Default execution reads metadata only. Scientific APIs are imported unchanged.
No command accepts T evaluation, training, tolerance selection, or registry edits.
"""
import argparse
import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent


def load_runner():
    spec = importlib.util.spec_from_file_location("postT_fixed_pca_frozen_runner", HERE / "fixed_pca_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_utilities(public, plan):
    runner = load_runner()
    path = runner.relative(public, plan["shared_utility_relative_path"])
    if sha(path) != plan["shared_utility_sha256"]:
        raise ValueError("Original stdlib diagnostic utility identity changed")
    spec = importlib.util.spec_from_file_location("fixed_pca_verification_stdlib", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return load_runner().sha(path)


def read(path):
    return load_runner().read(path)


def new(path, value):
    return load_runner().new(path, value)


def choose(plan, run_id, mode):
    if mode not in {"rescore", "replay"}:
        raise ValueError("Only saved V rescore or actual V checkpoint replay is supported")
    rows = [row for row in plan["conditions"] if row["run_id"] == run_id]
    if len(rows) != 1:
        raise ValueError("One explicit registered post-T CPU run ID is required")
    return rows[0]


def require_new_output(output, public):
    if not output.is_relative_to(public) or output == public:
        raise ValueError("New output must stay under the separate post-T public root")
    if output.exists():
        raise FileExistsError("Existing verification output must be preserved")


def registry_projection(plan, public):
    rows = []
    for condition in plan["conditions"]:
        path = public / "runs" / condition["run_id"] / "BASE_FIT_RECEIPT.json"
        if path.is_file():
            try:
                receipt = read(path)
                if receipt.get("run_id") != condition["run_id"] or receipt.get("family") != condition["family"]:
                    status = "INVALID_BASE_RECEIPT_IDENTITY"
                    reason = "Actual receipt belongs to another scientific identity; this route is not complete"
                else:
                    status = receipt["status"]
                    reason = "Actual BASE_FIT_RECEIPT status; fixed transformation and learning remain separate"
            except (OSError, ValueError, KeyError) as error:
                status = "UNREADABLE_BASE_FIT_RECEIPT"
                reason = "Receipt incomplete or unreadable at this capture: " + type(error).__name__
        else:
            status = "NO_COMPLETE_BASE_FIT_RECEIPT"
            reason = "No immutable BASE fit receipt at this capture; may be pending, running or partial"
        rows.append(dict(run_id=condition["run_id"], execution_status=status, reason=reason))
    return rows


def ensure_registry_alias(public, plan):
    source = public / plan["postT_registry_relative_path"]
    expected = plan["postT_registry_sha256"]
    if sha(source) != expected:
        raise ValueError("Separate immutable scientific registry changed")
    target = public / "EXPERIMENT_REGISTRY.csv"
    created = not target.exists()
    if created:
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
    if sha(target) != expected or sha(source) != expected:
        raise ValueError("Portable-route metadata alias differs; no existing file overwritten")
    return dict(file=target.name, sha256=expected, created_now=created, source_file=source.name,
                scientific_rows_or_execution_status_changed=False)


def rescore_call(api, manifest, reference, score, statistics, registry, expected_ids, private):
    return api(manifest, reference, score, statistics, registry, expected_ids, [manifest["cost_record"]], private)


def replay_call(api, run_id, task, private, public, model_path, deadline, k):
    result = api(run_id, task, private, public, model_path, device="cpu", deadline=deadline,
                 lease=None, batch_size=4, atol=1e-5, rtol=1e-4)
    if (result.get("run_id") != run_id or len(result.get("origins", [])) != 6
            or result.get("actual_F_streams") != 6*k or result.get("expected_F_streams") != 6*k
            or result.get("atol") != 1e-5 or result.get("rtol") != 1e-4
            or not isinstance(result.get("executed_now"), bool) or result.get("T_final_accessed") is not False):
        raise ValueError("Original six-origin CPU replay scope, actual F rows or fixed tolerance differs")
    return result


def validate_fixed_receipts(base, stages):
    if any(base.get(key) != 0 for key in ("search_updates", "refit_updates", "search_exposures", "refit_exposures")):
        raise ValueError("Fixed PCA must have zero actual optimizer updates and training exposures")
    if set(stages) != {"search", "refit"}:
        raise ValueError("Both original stage receipts are required")
    for stage in stages.values():
        if (stage.get("optimizer") != "NONE_FIXED_TRANSFORM"
                or any(stage.get(key) != 0 for key in ("updates", "exposures", "actual_epochs", "trainable_parameters", "F_training_logical_streams"))):
            raise ValueError("Actual fixed-transform stage records show training or a different optimizer contract")
    # BASE's original literal optimizer='AdamW' is not evidence of learning.


def validate_saved_v(manifest, receipt, condition, contract):
    dataset = condition["spec"]["dataset"]
    count, horizon, channels = contract["legal_origin_counts_primary"]["V_select"], contract["H"], contract["shape"][1]
    if (manifest["run_id"] != condition["run_id"] or manifest["split"] != "V_select"
            or manifest["cohort"] != f"{dataset}:primary:V_select:stride{horizon}"
            or manifest["data_sha"] != contract["data_sha256"] or manifest["scaler_sha"] != contract["refit_scaler_sha256"]
            or manifest["metadata"]["artifact_sha256"] != receipt["artifact_sha256"]
            or manifest["metadata"]["canonical_scaler_fit_end"] != contract["primary_split_ranges"]["A_stop"][1]
            or manifest["metadata"]["family"] != "PCA_FIXED_BOLT" or manifest["metadata"]["fold"] != "primary"):
        raise ValueError("Saved V task, horizon, cohort, data, scaler, artifact or family differs")
    shapes = dict(pred_normalized=[count, horizon, channels], target_normalized=[count, horizon, channels],
                  observed=[count, horizon, channels], origins=[count], channel_ids=[channels])
    if any(manifest["arrays"][key]["shape"] != shape for key, shape in shapes.items()):
        raise ValueError("Original full V dimensions differ, including dataset-specific horizon")
    cost, expected_streams = manifest["cost_record"], count*condition["spec"]["latent_k"]
    if cost.get("actual_F_streams") != expected_streams or cost.get("expected_F_streams") != expected_streams:
        raise ValueError("Full V F-stream ledger differs from the declared dataset cohort times K")


def prepare_inputs(runner, plan, old_public, old_cache, run_id, condition):
    public, private = old_public / runner.NAME, old_cache / runner.NAME
    folder = public / "predictions" / run_id / "V_select"
    paths = dict(fit_receipt=public / "runs" / run_id / "BASE_FIT_RECEIPT.json",
                 source_contract=public / "runs" / run_id / "SOURCE_CONTRACT.json",
                 manifest=folder / "PREDICTION_MANIFEST.json", reference=folder / "REFERENCE_CONTRACT.json", score=folder / "SCORES.csv",
                 search=public / "runs" / run_id / "search/RECEIPT.json", refit=public / "runs" / run_id / "refit/RECEIPT.json")
    values = {key: read(path) for key, path in paths.items() if key != "score"}
    receipt, manifest, source = values["fit_receipt"], values["manifest"], values["source_contract"]
    contract = plan["task_contracts"][condition["spec"]["dataset"]]
    if (receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != run_id or receipt.get("family") != condition["family"]
            or receipt.get("reused_complete_artifact") or source["spec"] != condition["spec"]
            or source["data_sha"] != contract["data_sha256"] or source["model_sha"] != plan["foundation_tensor_sha256"]):
        raise ValueError("A fresh complete fit with this exact registered recipe is required")
    validate_fixed_receipts(receipt, {key: values[key] for key in ("search", "refit")})
    validate_saved_v(manifest, receipt, condition, contract)
    with paths["score"].open(encoding="utf-8-sig", newline="") as stream:
        scores = list(csv.DictReader(stream))
    if len(scores) != 1:
        raise ValueError("Exactly one actual per-run V score is required")
    score = scores[0]
    for key, value in manifest["score"].items():
        if (isinstance(value, (int, float)) and float(score[key]) != value) or (not isinstance(value, (int, float)) and score[key] != str(value)):
            raise ValueError("Stored score CSV differs from the V manifest")
    paths["statistics"] = runner.relative(public, manifest["public_stats_relative_path"])
    for key, digest in source["files"].items():
        if plan["source39"].get(key) != digest:
            raise ValueError("Fit scientific source differs from the original frozen source")
    return values, score, paths, {path.relative_to(old_public).as_posix(): sha(path) for path in paths.values()}


def execute(old_public, old_cache, asset, plan_sha, mode, run_id, deadline, output):
    runner = load_runner()
    if not plan_sha:
        raise ValueError("Explicit plan SHA is required for any real array/model operation")
    plan = runner.metadata(old_public, old_cache, asset, plan_sha)
    condition = choose(plan, run_id, mode)
    dataset, k = condition["spec"]["dataset"], condition["spec"]["latent_k"]
    expected_origins = plan["task_contracts"][dataset]["legal_origin_counts_primary"]["V_select"]
    if deadline is None or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("A live deadline within the fixed campaign limit is required")
    public, private = old_public / runner.NAME, old_cache / runner.NAME
    runner.require_environment(old_public, old_cache, asset)
    utilities = load_utilities(old_public, plan)
    require_new_output(output, public)
    output.mkdir(parents=True, exist_ok=False)
    source_sha = sha(__file__)
    new(output / "STARTED.json", dict(mode=mode, run_id=run_id, plan_sha256=plan_sha, verifier_sha256=source_sha,
        actual_pid=os.getpid(), started_unix=time.time(), new_training=False, new_T_scoring=False))
    sys.path.insert(0, str(old_public))
    import psutil
    import torch
    process = psutil.Process(os.getpid())
    torch.set_num_threads(2)
    before = utilities.rss_record(process)
    stop = threading.Event()
    def guard():
        while not stop.wait(.25):
            rss = process.memory_info().rss
            if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
                new(output / "RESOURCE_STOP.json", dict(status="RESOURCE_STOP", rss_bytes=rss, deadline=deadline))
                os._exit(98)
    watchdog = threading.Thread(target=guard, daemon=True)
    watchdog.start()
    error_class, postflight_error_class, value, alias = None, None, None, None
    failures, bindings = [], {}
    source_post, fixed_activity_verified = False, False
    replay_attempted = False
    started = time.perf_counter()
    try:
        values, score, paths, bindings = prepare_inputs(runner, plan, old_public, old_cache, run_id, condition)
        fixed_activity_verified = True
        projection = registry_projection(plan, public)
        new(output / "INPUT_BINDINGS.json", dict(files_sha256=bindings, registry_projection=projection,
            expected_run_ids=[row["run_id"] for row in plan["conditions"]], selected_artifact_sha256=values["fit_receipt"]["artifact_sha256"],
            saved_V_origins=expected_origins, horizon=plan["task_contracts"][dataset]["H"], latent_k=k,
            stage="POST_T_V_ONLY_DEVELOPMENT", fixed_transform=True, actual_optimizer_updates=0, actual_training_exposures=0))
        if torch.cuda.is_initialized():
            raise RuntimeError("Unexpected CUDA initialization")
        if mode == "rescore":
            import audit_predictions
            with gzip.open(paths["statistics"], "rt", encoding="utf-8", newline="") as stream:
                statistics = list(csv.DictReader(stream))
            value = rescore_call(audit_predictions.audit_prediction_bundle, values["manifest"], values["reference"], score,
                statistics, projection, [row["run_id"] for row in plan["conditions"]], private)
            if value["status"] != "PASS" or value["n_origins"] != expected_origins or value.get("checkpoint_replay"):
                raise ValueError("Independent saved full-V rescore scope differs")
        else:
            alias = ensure_registry_alias(public, plan)
            bindings[(public / "EXPERIMENT_REGISTRY.csv").relative_to(old_public).as_posix()] = alias["sha256"]
            import data_contract as data
            import fit_base
            import route_predictions
            task = data.load_task(dataset, cache_dir=old_cache)
            runner.validate_task(task, plan, dataset, data, fit_base)
            replay_attempted = True
            value = replay_call(route_predictions.replay_route, run_id, task, private, public,
                                asset / plan["model_asset_relative_path"], deadline, k)
            if sha(public / "EXPERIMENT_REGISTRY.csv") != plan["postT_registry_sha256"]:
                raise ValueError("Portable route metadata alias changed during replay")
    except Exception as error:
        error_class = type(error).__name__
        failures.append(utilities.save_private_failure(private, old_cache, mode, run_id, "verification", error))
    finally:
        try:
            runner.metadata(old_public, old_cache, asset, plan_sha)
            if sha(__file__) != source_sha or any(sha(old_public / name) != digest for name, digest in bindings.items()):
                raise ValueError("Verifier or captured immutable fit/V evidence changed")
            source_post = True
        except Exception as error:
            postflight_error_class = type(error).__name__
            failures.append(utilities.save_private_failure(private, old_cache, mode, run_id, "postflight", error))
        stop.set()
        watchdog.join(timeout=1)
        passed = error_class is None and postflight_error_class is None and source_post and value is not None and value.get("status") == "PASS"
        actual_checkpoint_forward = False
        if replay_attempted:
            actual_checkpoint_forward = None if value is None else value["executed_now"]
        new(output / "RESULT.json", dict(status="PASS" if passed else "FAILED_OR_NUMERICAL_MISMATCH_PRESERVED", run_id=run_id, mode=mode,
            scientific_result=value, actual_checkpoint_forward_in_this_command=actual_checkpoint_forward,
            checkpoint_replay_call_attempted=replay_attempted, checkpoint_replay_receipt_returned=replay_attempted and value is not None,
            saved_full_V_rescoring=mode == "rescore", subset_checkpoint_replay=mode == "replay", training_rerun=False,
            full_V_origin_count=expected_origins, replay_origin_count=6 if mode == "replay" else 0,
            dataset=dataset, latent_k=k, horizon=plan["task_contracts"][dataset]["H"], fixed_transform=True,
            original_BASE_optimizer_literal_is_not_actual_training=True, fixed_zero_training_activity_verified=fixed_activity_verified,
            actual_optimizer_updates=0 if fixed_activity_verified else None, actual_training_exposures=0 if fixed_activity_verified else None,
            actual_pid=process.pid, actual_create_time=process.create_time(),
            error_class=error_class, postflight_error_class=postflight_error_class, private_failures=failures, metadata_alias=alias,
            original_source_registry_selection_T_and_captured_inputs_preserved=source_post, cuda_initialized=torch.cuda.is_initialized(),
            new_T_target_or_scoring=False, independent_generalization_claimed=False, original_GPU_minimum_gap_completed=False,
            archive_deserialization_scope="Original approved task loader may physically deserialize T bytes; no T target/evaluation API is requested.",
            elapsed_process_wall_seconds=time.perf_counter()-started, resource_before=before, resource_after=utilities.rss_record(process),
            plan_sha256=plan_sha, verifier_sha256=source_sha, finished_unix=time.time()))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ("public", "cache", "asset"):
        parser.add_argument("--"+root+"-root", default=os.environ.get("INFORMS_"+root.upper()+"_ROOT"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("rescore", "replay"))
    parser.add_argument("--run-id")
    parser.add_argument("--plan-sha")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not all((args.public_root, args.cache_root, args.asset_root)):
        parser.error("Explicit original public/cache/asset roots required")
    public, cache, asset = (Path(value).resolve() for value in (args.public_root, args.cache_root, args.asset_root))
    if args.execute:
        if not all((args.mode, args.run_id, args.plan_sha, args.deadline, args.output)):
            parser.error("Execute requires explicit mode/run/plan/deadline/new output")
        raise SystemExit(execute(public, cache, asset, args.plan_sha, args.mode, args.run_id, args.deadline, Path(args.output).resolve()))
    runner = load_runner()
    plan = runner.metadata(public, cache, asset, args.plan_sha)
    print(json.dumps(dict(status="METADATA_ONLY_VERIFIER", registered_conditions=len(plan["conditions"]),
        plan_sha256=sha(public / runner.NAME / "PLAN.json"), verifier_sha256=sha(__file__),
        numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules,
        actual_model_or_array_access=0, training_or_replay_or_T_executed=False)), flush=True)


if __name__ == "__main__":
    main()
