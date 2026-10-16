"""Explicit post-T multi-K saved-V rescore or six-origin checkpoint replay.

The default command is stdlib metadata only. Original scientific APIs and
tolerances are preserved; each completed output can be checked independently.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
PLAN_SHA = "2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec"
RUNNER_SHA = "44cd5e3fed5f4ad8a2110c54c9118063c6838b68d9059dea46f6b33bc32be10a"
FIT_SOURCES = ("fit_base.py", "training.py", "networks.py", "data_contract.py", "gpu_runtime.py", "evaluation.py")
ORDERED_SOURCES = ("ordered_pilot.py", "prediction_cache.py", "fit_base.py", "training.py", "networks.py", "data_contract.py", "evaluation.py", "gpu_runtime.py")
PREDICTION_SOURCES = ("prediction_cache.py", "fit_base.py", "data_contract.py", "training.py", "evaluation.py", "networks.py", "gpu_runtime.py")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_runner():
    path = HERE / "multik_runner.py"
    if sha(path) != RUNNER_SHA:
        raise ValueError("Frozen multi-K runner changed")
    spec = importlib.util.spec_from_file_location("multik_verification_frozen_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(runner, public, cache, asset, plan_sha):
    if plan_sha != PLAN_SHA:
        raise ValueError("This verifier requires the exact ROOT-sealed multi-K PLAN")
    return runner.metadata(public, cache, asset, plan_sha)


def choose(plan, run_id, mode):
    if mode not in {"rescore", "replay"}:
        raise ValueError("Only saved full V rescore or selected-checkpoint V replay is supported")
    matches = [row for row in plan["outputs"] if row["run_id"] == run_id]
    if len(matches) != 1 or matches[0].get("split") != "V_select":
        raise ValueError("One explicit preregistered V output is required")
    return matches[0]


def folder_for(public, condition):
    run_id = condition["run_id"]
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ValueError("Safe output identity required")
    if condition["kind"] == "independent":
        return public / "predictions" / run_id / "V_select"
    if condition["kind"] not in {"ordered", "nested"}:
        raise ValueError("Unknown original output layout")
    return public / "ordered_predictions" / run_id


def require_new_output(runner, output, public):
    if not output.is_relative_to(public) or output == public:
        raise ValueError("New verifier output must stay in the separate post-T namespace")
    runner.relative(public, output.relative_to(public))
    if output.exists():
        raise FileExistsError("Existing verification evidence must be preserved")


def registry_projection(plan, public):
    rows = []
    for condition in plan["outputs"]:
        manifest_path = folder_for(public, condition) / "PREDICTION_MANIFEST.json"
        status, reason = "NO_COMPLETE_V_MANIFEST", "No completed immutable V manifest at this capture; pending/running/partial remain distinct from success"
        try:
            parent = condition["parent_run_id"]
            base_path = public / "runs" / parent / "BASE_FIT_RECEIPT.json" if parent else None
            base = read(base_path) if base_path is not None and base_path.is_file() else None
            if base is not None and base.get("status") != "COMPLETE_VALID":
                status, reason = str(base.get("status") or "INVALID_PARENT_STATUS"), "Actual parent receipt is not complete; saved V is not accepted"
            elif manifest_path.is_file():
                manifest = read(manifest_path)
                meta = manifest["metadata"]
                expected_parent = "nested_pca" if condition["kind"] == "nested" else parent
                artifact = meta.get("artifact_sha256") if condition["kind"] == "independent" else meta.get("parent_artifact_sha256")
                if manifest.get("run_id") != condition["run_id"] or manifest.get("split") != "V_select":
                    raise ValueError("Saved manifest identity differs")
                if condition["kind"] != "independent" and (meta.get("parent_run_id") != expected_parent or meta.get("variant") != condition["variant"]):
                    raise ValueError("Saved ordered parent/variant differs")
                if parent is not None and (base is None or base.get("run_id") != parent or base.get("family") != "B0" or artifact != base.get("artifact_sha256")):
                    raise ValueError("Saved V has no matching complete selected parent")
                status, reason = "COMPLETE_VALID", "Completed saved-V metadata; independent numeric/weight verification is recorded separately per explicit output"
        except (OSError, ValueError, KeyError, TypeError) as error:
            status, reason = "INVALID_OR_UNREADABLE_V_METADATA", "Local immutable metadata invalid/incomplete: " + type(error).__name__
        rows.append(dict(run_id=condition["run_id"], execution_status=status, reason=reason,
                         registered_role=condition["role"], output_kind=condition["kind"]))
    return rows


def ensure_registry_alias(runner, public, plan):
    source = runner.relative(public, plan["postT_registry_relative_path"])
    expected = plan["postT_registry_sha256"]
    if sha(source) != expected:
        raise ValueError("Separate immutable twelve-output registry changed")
    target = runner.relative(public, "EXPERIMENT_REGISTRY.csv")
    created = not target.exists()
    if created:
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
    if sha(target) != expected or sha(source) != expected:
        raise ValueError("Portable registry alias must be an exact byte copy without overwrite")
    return dict(file=target.name, sha256=expected, created_now=created, source_file=source.name,
                scientific_rows_or_execution_status_changed=False)


def validate_parent(runner, plan, condition, base, source, search, refit):
    matches = [row for row in plan["fits"] if row["run_id"] == condition["parent_run_id"]]
    if len(matches) != 1:
        raise ValueError("Exactly one preregistered selected fit parent required")
    fit_row, contract = matches[0], plan["task_contract"]
    runner.validate_complete_fit(base, search, refit, fit_row)
    files = {name: plan["source39"][name] for name in FIT_SOURCES}
    source_sha = runner.fingerprint(files)
    axes = dict(L=512, H=48, C=11, columns=contract["columns"], fold="primary", source_array_fingerprints=contract["array_sha256"])
    config_sha = runner.fingerprint(dict(spec=fit_row["spec"], axes=axes))
    shared = dict(source_sha=source_sha, config_sha=config_sha, data_sha=contract["data_sha256"], model_sha=plan["foundation_tensor_sha256"])
    if (source.get("spec") != fit_row["spec"] or source.get("files") != files
            or any(source.get(key) != value for key, value in shared.items())
            or base.get("source_sha") != source_sha or base.get("epochs") != [.25, 1., 3.]
            or base.get("seed") != 92601 or base.get("search_updates") != search.get("updates")
            or search.get("train_origins") != contract["primary_split_ranges"]["A_fit"][1]-512-48+1
            or refit.get("train_origins") != contract["primary_split_ranges"]["A_stop"][1]-512-48+1
            or base.get("search_train_origins") != search.get("train_origins")
            or base.get("search_fit_end") != contract["primary_split_ranges"]["A_fit"][1]
            or base.get("refit_fit_end") != contract["primary_split_ranges"]["A_stop"][1]
            or base.get("normalization_fit_end") != contract["primary_split_ranges"]["A_stop"][1]
            or base.get("refit_scaler_sha") != contract["refit_scaler_sha256"]):
        raise ValueError("Exact parent recipe/axes/source/model/normalization or stage updates differ")
    for stage, scaler in ((search, "search_scaler_sha256"), (refit, "refit_scaler_sha256")):
        if any(stage["contract"].get(key) != value for key, value in shared.items()) or stage["contract"].get("scaler_sha") != contract[scaler]:
            raise ValueError("Stage source/config/model/data/scaler contract differs")
    expected_artifact = f"runs/{fit_row['run_id']}/selected_bundle.pt"
    if base.get("artifact_cache_relative_path") != expected_artifact or not base.get("artifact_sha256") or not base.get("model_state_sha"):
        raise ValueError("Selected parent artifact location or identity differs")


def validate_nested_contract(runner, plan, contract):
    expected = dict(source_sha=runner.fingerprint({name: plan["source39"][name] for name in ORDERED_SOURCES}),
                    data_sha=plan["task_contract"]["data_sha256"], fit_end=plan["task_contract"]["primary_split_ranges"]["A_stop"][1])
    if any(contract.get(key) != value for key, value in expected.items()) or not contract.get("artifact_sha256"):
        raise ValueError("Original fixed nested-PCA source, data, A60 fit end or basis archive differs")


def validate_saved_v(runner, plan, condition, manifest, parent):
    contract, meta, k = plan["task_contract"], manifest["metadata"], condition["latent_k"]
    end = contract["primary_split_ranges"]["A_stop"][1]
    if (manifest.get("run_id") != condition["run_id"] or manifest.get("split") != "V_select"
            or manifest.get("cohort") != "beijing:primary:V_select:stride48" or manifest.get("data_sha") != contract["data_sha256"]
            or manifest.get("scaler_sha") != contract["refit_scaler_sha256"] or meta.get("data_sha256") != contract["data_sha256"]
            or meta.get("fold") != "primary" or meta.get("parent_fit_end") != end):
        raise ValueError("Saved full V data/scaler/A60/cohort binding differs")
    shapes = dict(pred_normalized=[43,48,11], target_normalized=[43,48,11], observed=[43,48,11], origins=[43], channel_ids=[11])
    if set(manifest.get("arrays", {})) != set(shapes) or any(manifest["arrays"][key]["shape"] != value for key,value in shapes.items()):
        raise ValueError("Exact V43/H48/C11 array axes required")
    cost = manifest["cost_record"]
    expected_cost = dict(run_id=condition["run_id"], batch=43, latent_k=k, actual_F_streams=43*k, expected_F_streams=43*k, F_calls=1)
    if any(cost.get(key) != value for key,value in expected_cost.items()):
        raise ValueError("Saved full-V per-K foundation stream ledger differs")
    if condition["kind"] == "independent":
        fit_row = next(row for row in plan["fits"] if row["run_id"] == condition["parent_run_id"])
        expected_sources = {name: plan["source39"][name] for name in PREDICTION_SOURCES}
        expected_source = runner.fingerprint(dict(deployment_sources=expected_sources, training_source_sha=parent["source_sha"]))
        model_sha = runner.fingerprint(dict(artifact_sha=parent["artifact_sha256"], model_state_sha=parent["model_state_sha"], normalization=parent["refit_scaler_sha"]))
        if (meta.get("family") != "B0" or meta.get("artifact_sha256") != parent["artifact_sha256"]
                or meta.get("canonical_scaler_fit_end") != end or meta.get("parent_normalization_fit_end") != end
                or meta.get("source_files") != expected_sources or meta.get("training_source_sha") != parent["source_sha"]
                or meta.get("dense") is not False or meta.get("role") != "base_parent"
                or manifest.get("config_sha") != runner.fingerprint(dict(parent_spec=fit_row["spec"], dense=False, fold="primary", role="base_parent"))):
            raise ValueError("Independent B0 selected artifact/source/normalization differs")
    else:
        training_source = parent["source_sha"]
        expected_source = runner.fingerprint(dict(sources={name: plan["source39"][name] for name in ORDERED_SOURCES}, training_source_sha=training_source))
        nested = condition["kind"] == "nested"
        expected_parent = "nested_pca" if nested else condition["parent_run_id"]
        if (meta.get("parent_run_id") != expected_parent or meta.get("parent_artifact_sha256") != parent["artifact_sha256"]
                or meta.get("variant") != condition["variant"] or meta.get("family") != ("NESTED_PCA" if nested else "B0")
                or meta.get("inference_latent_k") != k or meta.get("stored_latent_k") != 11
                or meta.get("foundation_sha") != plan["foundation_tensor_sha256"] or meta.get("actual_F_streams") != 43*k
                or meta.get("T_final_accessed") is not False
                or meta.get("posthoc_prefix_was_trained") != (condition["variant"] == "shared" or condition["variant"] in {"posthoc","max_only"} and k == 11)
                or manifest.get("config_sha") != runner.fingerprint(dict(parent_run_id=expected_parent, variant=condition["variant"], latent_k=k, fold="primary"))):
            raise ValueError("Ordered/nested prefix role, parent, foundation or fixed basis binding differs")
        model_sha = manifest["model_sha"] if nested else runner.fingerprint(dict(artifact_sha=parent["artifact_sha256"], state_sha=parent["model_state_sha"]))
    if manifest.get("source_sha") != expected_source or manifest.get("model_sha") != model_sha:
        raise ValueError("Prediction source or selected model fingerprint differs")


def validate_nested_arrays(runner, plan, manifest, contract, artifact, np, evaluation, fit):
    with np.load(artifact, allow_pickle=False) as archive:
        expected_keys = {"basis","center","mean","std","fit_end","data_sha","source_sha"}
        if set(archive.files) != expected_keys:
            raise ValueError("Original nested-PCA archive schema differs")
        value = {key: archive[key] for key in expected_keys}
    if (value["basis"].shape != (11,11) or value["center"].shape != (11,)
            or value["mean"].shape != (11,) or value["std"].shape != (11,)
            or any(not np.isfinite(value[key]).all() for key in ("basis","center","mean","std"))
            or not (value["std"] > 0).all() or int(value["fit_end"].item()) != contract["fit_end"]
            or str(value["data_sha"].item()) != contract["data_sha"] or str(value["source_sha"].item()) != contract["source_sha"]
            or fit.scaler_fingerprint(value["mean"],value["std"]) != plan["task_contract"]["refit_scaler_sha256"]):
        raise ValueError("Nested full basis/center/scaler or A60/source/data identity differs")
    basis = evaluation.array_fingerprint(value["basis"])
    center = evaluation.array_fingerprint(value["center"])
    model_sha = runner.fingerprint(dict(full_basis=basis, center=center, artifact_sha=contract["artifact_sha256"], foundation_sha=plan["foundation_tensor_sha256"]))
    if manifest["model_sha"] != model_sha:
        raise ValueError("Nested model fingerprint differs from actual fixed basis and foundation")
    return dict(basis=basis, center=center, artifact_sha256=contract["artifact_sha256"], fit_end=contract["fit_end"],
                actual_optimizer_updates=0, actual_training_exposures=0, no_BASE_fit_receipt_required=True,
                zero_activity_evidence="Frozen original nested artifact path performs initialize_pca; it creates no optimizer or train_segment")


def prepare_inputs(runner, plan, old_public, old_cache, run_id, condition):
    public, private = old_public/runner.NAME, old_cache/runner.NAME
    folder = folder_for(public, condition)
    paths = dict(manifest=folder/"PREDICTION_MANIFEST.json", reference=folder/"REFERENCE_CONTRACT.json", score=folder/"SCORES.csv")
    values = {key:read(path) for key,path in paths.items() if key != "score"}
    if condition["kind"] == "nested":
        paths["nested_contract"] = runner.relative(private, "ordered/primary_beijing/nested_pca_contract.json")
        values["nested_contract"] = read(paths["nested_contract"])
        validate_nested_contract(runner, plan, values["nested_contract"])
        parent = values["nested_contract"]
        artifact_relative = "ordered/primary_beijing/nested_pca.npz"
    else:
        root = public/"runs"/condition["parent_run_id"]
        paths.update(fit_receipt=root/"BASE_FIT_RECEIPT.json", source_contract=root/"SOURCE_CONTRACT.json", search=root/"search/RECEIPT.json", refit=root/"refit/RECEIPT.json")
        values.update({key:read(paths[key]) for key in ("fit_receipt","source_contract","search","refit")})
        validate_parent(runner, plan, condition, values["fit_receipt"], values["source_contract"], values["search"], values["refit"])
        parent = values["fit_receipt"]
        artifact_relative = parent["artifact_cache_relative_path"]
    manifest = values["manifest"]
    validate_saved_v(runner, plan, condition, manifest, parent)
    paths["artifact"] = runner.relative(private, artifact_relative)
    expected_cache = f"predictions/{run_id}/V_select.npz" if condition["kind"] == "independent" else f"ordered_predictions/{run_id}/V_select.npz"
    expected_statistics = (folder/"PUBLIC_ERROR_STATS.csv.gz").relative_to(public).as_posix()
    if manifest.get("cache_relative_path") != expected_cache or manifest.get("public_stats_relative_path") != expected_statistics:
        raise ValueError("Original prediction/statistic layout differs from registered output kind")
    paths["prediction_archive"] = runner.relative(private, expected_cache)
    paths["statistics"] = runner.relative(public, expected_statistics)
    if sha(paths["artifact"]) != parent["artifact_sha256"] or sha(paths["prediction_archive"]) != manifest["archive_sha256"]:
        raise ValueError("Selected basis/weights or saved-V archive changed")
    for key in ("run_id","split","cohort","source_sha","config_sha","data_sha","model_sha","scaler_sha"):
        if manifest.get(key) != values["reference"].get(key):
            raise ValueError("Independent frozen V reference identity differs")
    scores = runner.rows(paths["score"])
    if len(scores) != 1:
        raise ValueError("Exactly one saved full-V score row required")
    score = scores[0]
    for key,value in manifest["score"].items():
        if (isinstance(value,(int,float)) and float(score[key]) != value) or (not isinstance(value,(int,float)) and score[key] != str(value)):
            raise ValueError("Saved V CSV differs from immutable score manifest")
    bindings = []
    for path in paths.values():
        root_name, root = ("public",old_public) if path.is_relative_to(old_public) else ("cache",old_cache)
        bindings.append(dict(root=root_name, relative_path=path.relative_to(root).as_posix(), sha256=sha(path)))
    parent_receipts = []
    for row in plan["fits"]:
        path = public/"runs"/row["run_id"]/"BASE_FIT_RECEIPT.json"
        entry = dict(run_id=row["run_id"], receipt_present=path.is_file())
        if entry["receipt_present"]:
            digest = sha(path)
            entry.update(public_relative_path=path.relative_to(old_public).as_posix(), sha256=digest)
            bindings.append(dict(root="public", relative_path=entry["public_relative_path"], sha256=digest))
        parent_receipts.append(entry)
    values["parent_receipts_at_capture"] = parent_receipts
    return values, score, paths, bindings


def rescore_call(api, manifest, reference, score, statistics, registry, expected_ids, private):
    return api(manifest, reference, score, statistics, registry, expected_ids, [manifest["cost_record"]], private)


def replay_call(api, run_id, task, private, public, model_path, deadline, k):
    result = api(run_id,task,private,public,model_path,device="cpu",deadline=deadline,lease=None,batch_size=4,atol=1e-5,rtol=1e-4)
    if (result.get("run_id") != run_id or len(result.get("origins",[])) != 6 or len(set(result["origins"])) != 6
            or result.get("actual_F_streams") != 6*k or result.get("expected_F_streams") != 6*k
            or result.get("F_physical_streams") != 6*k or result.get("atol") != 1e-5 or result.get("rtol") != 1e-4
            or not isinstance(result.get("executed_now"),bool) or result.get("T_final_accessed") is not False):
        raise ValueError("Original six-origin CPU replay, actual logical/physical 6K streams or fixed tolerance differs")
    return result


def forward_evidence(attempted, result):
    returned = attempted and result is not None
    return dict(actual_checkpoint_forward_in_this_command=(result["executed_now"] if returned else None) if attempted else False,
                checkpoint_replay_call_attempted=attempted, checkpoint_replay_receipt_returned=returned)


def execute(old_public, old_cache, asset, plan_sha, mode, run_id, deadline, output):
    runner = load_runner()
    plan = metadata(runner,old_public,old_cache,asset,plan_sha)
    condition = choose(plan,run_id,mode)
    if deadline is None or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("Live deadline inside the original fixed campaign wall required")
    runner.environment(old_cache,asset)
    public, private = old_public/runner.NAME, old_cache/runner.NAME
    require_new_output(runner,output,public)
    used = sum(path.stat().st_size for root in (old_public,old_cache) for path in root.rglob("*") if path.is_file())
    if used+256*1024**2 > plan["storage_limit_bytes"]:
        raise ValueError("Combined campaign storage limit reached")
    utility = runner.load_dependency(old_public,runner.UTILITY_PATH,"multik_verifier_stdlib_utility")
    pca = runner.load_dependency(old_public,runner.PCA_RUNNER_PATH,"multik_verifier_task_contract")
    output.mkdir(parents=True,exist_ok=False)
    source_sha = sha(__file__)
    new(output/"STARTED.json",dict(mode=mode,run_id=run_id,plan_sha256=plan_sha,verifier_sha256=source_sha,
        actual_pid=os.getpid(),started_unix=time.time(),new_training=False,new_T_scoring=False))
    sys.path.insert(0,str(old_public))
    import psutil
    import torch
    process, stop = psutil.Process(os.getpid()), threading.Event()
    torch.set_num_threads(2)
    before, started = utility.rss_record(process), time.perf_counter()
    def guard():
        while not stop.wait(.25):
            rss = process.memory_info().rss
            if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
                new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss,deadline_unix=deadline))
                os._exit(98)
    watcher = threading.Thread(target=guard,daemon=True)
    watcher.start()
    value, alias, nested_evidence = None, None, None
    error_class, postflight_error, failures, bindings = None, None, [], []
    source_post, replay_attempted, phase = False, False, "saved_input_binding"
    try:
        values, score, paths, bindings = prepare_inputs(runner,plan,old_public,old_cache,run_id,condition)
        projection = registry_projection(plan,public)
        new(output/"INPUT_BINDINGS.json",dict(files=bindings,registry_projection=projection,
            expected_run_ids=[row["run_id"] for row in plan["outputs"]],parent_receipts=values["parent_receipts_at_capture"],
            selected_artifact_sha256=(values.get("fit_receipt") or values["nested_contract"])["artifact_sha256"],
            dataset="beijing",saved_V_origins=43,horizon=48,channels=11,latent_k=condition["latent_k"],stage=plan["stage"]))
        if torch.cuda.is_initialized() or process.memory_info().rss > plan["max_RSS_bytes"]:
            raise RuntimeError("CPU/RSS contract violated before checkpoint or array execution")
        if condition["kind"] == "nested":
            phase = "fixed_nested_basis_identity"
            import numpy as np
            import evaluation
            import fit_base
            nested_evidence = validate_nested_arrays(runner,plan,values["manifest"],values["nested_contract"],paths["artifact"],np,evaluation,fit_base)
            new(output/"NESTED_FIXED_BASIS_BINDING.json",nested_evidence)
        if mode == "rescore":
            phase = "independent_full_V_rescore"
            import audit_predictions
            with gzip.open(paths["statistics"],"rt",encoding="utf-8",newline="") as stream:
                statistics = list(csv.DictReader(stream))
            value = rescore_call(audit_predictions.audit_prediction_bundle,values["manifest"],values["reference"],score,
                                 statistics,projection,[row["run_id"] for row in plan["outputs"]],private)
            if value.get("status") != "PASS" or value.get("n_origins") != 43 or value.get("num_channels") != 11 or value.get("horizon") != 48 or value.get("checkpoint_replay") is not False:
                raise ValueError("Independent full V43/H48/C11 saved-array rescore scope differs")
        else:
            phase = "task_binding_before_replay"
            alias = ensure_registry_alias(runner,public,plan)
            bindings.append(dict(root="public",relative_path=(public/"EXPERIMENT_REGISTRY.csv").relative_to(old_public).as_posix(),sha256=alias["sha256"]))
            import data_contract as data
            import fit_base
            import route_predictions
            task = data.load_task("beijing",cache_dir=old_cache)
            task_binding = pca.validate_task(task,dict(task_contracts=dict(beijing=plan["task_contract"])),"beijing",data,fit_base)
            new(output/"TASK_BINDING.json",task_binding)
            phase = "original_checkpoint_subset_replay"
            replay_attempted = True
            value = replay_call(route_predictions.replay_route,run_id,task,private,public,asset/plan["model_asset_relative_path"],deadline,condition["latent_k"])
            receipt_path = public/"route_replays"/run_id/"REPLAY_RECEIPT.json"
            reference_path = public/"route_replays"/run_id/"PRE_INFERENCE_REFERENCE.json"
            replay_reference = read(reference_path)
            if (value.get("standard_manifest_sha256") != sha(paths["manifest"])
                    or value.get("standard_manifest_public_relative_path") != paths["manifest"].relative_to(public).as_posix()
                    or replay_reference.get("standard_archive_sha256") != values["manifest"]["archive_sha256"]
                    or replay_reference.get("expected_F_streams") != 6*condition["latent_k"]
                    or replay_reference.get("created_before_inference") is not True
                    or value.get("selected_artifact_bindings") != replay_reference.get("artifact_bindings")
                    or value.get("selected_artifact_bindings",{}).get("route",{}).get("sha256") != (values.get("fit_receipt") or values["nested_contract"])["artifact_sha256"]):
                raise ValueError("Replay standard archive/manifest or selected artifact binding differs")
            bindings.extend(dict(root="public",relative_path=path.relative_to(old_public).as_posix(),sha256=sha(path)) for path in (receipt_path,reference_path))
    except Exception as error:
        error_class = type(error).__name__
        failures.append(utility.save_private_failure(private,old_cache,mode,run_id,phase,error))
    finally:
        try:
            metadata(runner,old_public,old_cache,asset,plan_sha)
            roots = dict(public=old_public,cache=old_cache)
            if sha(__file__) != source_sha or any(sha(runner.relative(roots[row["root"]],row["relative_path"])) != row["sha256"] for row in bindings):
                raise ValueError("Verifier or captured original/selected evidence bytes changed")
            if torch.cuda.is_initialized():
                raise RuntimeError("Unexpected CUDA initialization")
            source_post = True
        except Exception as error:
            postflight_error = type(error).__name__
            failures.append(utility.save_private_failure(private,old_cache,mode,run_id,"postflight",error))
        stop.set(); watcher.join(timeout=1)
        passed = error_class is None and postflight_error is None and source_post and value is not None and value.get("status") == "PASS"
        new(output/"RESULT.json",dict(status="PASS" if passed else "FAILED_OR_NUMERICAL_MISMATCH_PRESERVED",run_id=run_id,mode=mode,
            scientific_result=value,**forward_evidence(replay_attempted,value),saved_full_V_rescoring=mode=="rescore",subset_checkpoint_replay=mode=="replay",
            training_rerun=False,full_V_origin_count=43,replay_origin_count=6 if mode=="replay" else 0,dataset="beijing",horizon=48,channels=11,
            latent_k=condition["latent_k"],registered_role=condition["role"],output_kind=condition["kind"],nested_fixed_basis_evidence=nested_evidence,
            actual_pid=process.pid,actual_create_time=process.create_time(),error_class=error_class,postflight_error_class=postflight_error,private_failures=failures,
            metadata_alias=alias,original_source_registry_selection_T_and_captured_inputs_preserved=source_post,cuda_initialized=torch.cuda.is_initialized(),
            new_T_target_or_scoring=False,independent_generalization_claimed=False,original_GPU_minimum_gap_completed=False,
            archive_deserialization_scope="Replay Task loader may physically deserialize original T bytes; only original A scalers and V replay are requested.",
            elapsed_process_wall_seconds=time.perf_counter()-started,resource_before=before,resource_after=utility.rss_record(process),
            plan_sha256=plan_sha,verifier_sha256=source_sha,finished_unix=time.time(),
            parent_must_reconcile_forced_exit_and_known_process_tree=True))
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ("public","cache","asset"):
        parser.add_argument("--"+root+"-root")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--mode",choices=("rescore","replay"))
    parser.add_argument("--run-id")
    parser.add_argument("--plan-sha")
    parser.add_argument("--deadline",type=float)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    roots = (args.public_root,args.cache_root,args.asset_root)
    if args.execute:
        if not all(roots+(args.mode,args.run_id,args.plan_sha,args.deadline,args.output)):
            parser.error("Execution requires roots, explicit mode/ID, sealed PLAN SHA, deadline and fresh output")
        public,cache,asset = (Path(value).resolve() for value in roots)
        return execute(public,cache,asset,args.plan_sha,args.mode,args.run_id,args.deadline,Path(args.output).resolve())
    runner = load_runner()
    if any(roots):
        if not all(roots) or not args.plan_sha:
            parser.error("Metadata validation requires all original roots and the exact PLAN SHA")
        metadata(runner,*(Path(value).resolve() for value in roots),args.plan_sha)
    print(json.dumps(dict(status="METADATA_ONLY_VERIFIER_PREPARATION",registered_outputs=12,fit_parents=5,sealed_plan_sha256=PLAN_SHA,
        numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,actual_model_or_array_access=0,
        training_or_replay_or_T_executed=False,verifier_sha256=sha(__file__))),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
