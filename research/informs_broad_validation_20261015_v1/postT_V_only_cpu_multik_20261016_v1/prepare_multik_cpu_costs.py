"""Prepare one sealed post-T multi-K route's B1/B8 CPU cost jobs.

The default is stdlib metadata only. Explicit preparation reuses the original
Task/catalog and selected adapter or PCA basis; it constructs no F or forecasts.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
PLAN_SHA = "2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec"
DEPENDENCIES = {
    "verify_multik_outputs.py": "0455361c307b7142cc15d5f820fef9cb827271b1df5669e1724f7ab5a9666238",
    "multik_runner.py": "44cd5e3fed5f4ad8a2110c54c9118063c6838b68d9059dea46f6b33bc32be10a",
    "test_multik_runner.py": "f873e5e75c68a7e6d1d339983940c24e58783493ee334ca108a53562ca13f5f7",
}
REPLAY_SOURCES = ("route_catalog.py", "route_predictions.py", "deployment.py", "data_contract.py", "evaluation.py",
                  "training.py", "networks.py", "gpu_runtime.py", "ridge_baseline.py", "combination.py", "fit_base.py")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): value.update(block)
    return value.hexdigest()


def load_verifier():
    if any(sha(HERE/name) != digest for name, digest in DEPENDENCIES.items()):
        raise ValueError("Frozen multi-K verifier, runner or reviewed runner test changed")
    spec = importlib.util.spec_from_file_location("multik_cost_frozen_verifier", HERE/"verify_multik_outputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def requested_paths(verifier, runner, plan, public, private, condition):
    folder = verifier.folder_for(public, condition)
    run_id = condition["run_id"]
    paths = dict(alias=runner.relative(public,"EXPERIMENT_REGISTRY.csv"), manifest=folder/"PREDICTION_MANIFEST.json",
        reference=folder/"REFERENCE_CONTRACT.json", score=folder/"SCORES.csv", statistics=folder/"PUBLIC_ERROR_STATS.csv.gz",
        prediction=runner.relative(private,f"predictions/{run_id}/V_select.npz" if condition["kind"]=="independent" else f"ordered_predictions/{run_id}/V_select.npz"),
        replay_reference=runner.relative(public,f"route_replays/{run_id}/PRE_INFERENCE_REFERENCE.json"),
        replay_receipt=runner.relative(public,f"route_replays/{run_id}/REPLAY_RECEIPT.json"),
        replay_archive=runner.relative(private,f"route_replays/{run_id}/V_subset.npz"))
    if condition["kind"] == "nested":
        paths.update(nested_contract=runner.relative(private,"ordered/primary_beijing/nested_pca_contract.json"),
                     artifact=runner.relative(private,"ordered/primary_beijing/nested_pca.npz"))
    else:
        parent = condition["parent_run_id"]
        for key,name in (("fit_receipt","BASE_FIT_RECEIPT.json"),("source","SOURCE_CONTRACT.json"),
                         ("search","search/RECEIPT.json"),("refit","refit/RECEIPT.json")):
            paths[key] = runner.relative(public,f"runs/{parent}/{name}")
        paths["artifact"] = runner.relative(private,f"runs/{parent}/selected_bundle.pt")
    return paths


def readiness(verifier, runner, plan, public, private, condition):
    paths = requested_paths(verifier,runner,plan,public,private,condition)
    alias = paths["alias"]
    if alias.is_file() and sha(alias) != plan["postT_registry_sha256"]:
        raise ValueError("ROOT's preexisting registry alias must equal the sealed twelve-row registry")
    missing = [key for key,path in paths.items() if not path.is_file()]
    return dict(ready=not missing,missing=missing,requested_run_id=condition["run_id"],
        other_eleven_outputs_are_not_a_gate=True,scope="File presence and exact preexisting registry alias; numeric/selected input validation occurs only in explicit preparation")


def metadata(public, cache, asset, run_id=None, plan_sha=None):
    if plan_sha is not None and plan_sha != PLAN_SHA:
        raise ValueError("Only the sealed post-T multi-K PLAN is supported")
    verifier = load_verifier()
    runner = verifier.load_runner()
    plan = verifier.metadata(runner,public,cache,asset,PLAN_SHA)
    condition = verifier.choose(plan,run_id,"replay") if run_id is not None else None
    state = readiness(verifier,runner,plan,public/runner.NAME,cache/runner.NAME,condition) if condition else None
    return verifier,runner,plan,condition,state


def cost_spec(canonical, size, binding, plan, fingerprint):
    condition, parent, manifest = binding["condition"],binding["parent"],binding["manifest"]
    nested = condition["kind"] == "nested"
    expected = dict(run_id=condition["run_id"],dataset="beijing",family="NESTED_PCA" if nested else "B0",fold="primary",
        artifact_cache_relative_path=binding["artifact_relative"],artifact_sha256=parent["artifact_sha256"],latent_k=condition["latent_k"],
        canonical_fit_end=plan["task_contract"]["primary_split_ranges"]["A_stop"][1],foundation_sha=plan["foundation_tensor_sha256"],model_size="small")
    if not nested:
        expected.update(expected_prediction_model_sha=fingerprint(dict(artifact_sha=parent["artifact_sha256"],
            model_state_sha=parent["model_state_sha"],normalization=parent["refit_scaler_sha"])),
            fit_receipt_public_relative_path=binding["receipt_relative"],fit_receipt_sha256=binding["receipt_sha256"])
    if condition["kind"] == "independent":
        expected["standard_V_binding"] = dict(model_sha=manifest["model_sha"],archive_sha256=manifest["archive_sha256"],
            manifest_public_relative_path=binding["manifest_relative"],manifest_sha256=binding["manifest_sha256"])
    else:
        expected.update(ordered_manifest_public_relative_path=binding["manifest_relative"],ordered_manifest_sha256=binding["manifest_sha256"])
        if not nested:
            expected.update(source_parent_run_id=condition["parent_run_id"],prefix_variant=condition["variant"])
    if size != "small" or canonical != expected:
        raise ValueError("Original canonical multi-K route differs; ordered parent and saved-V model fingerprints must remain distinct")
    return dict(canonical,accuracy_run_id=condition["run_id"],row_microbatch=128)


def validate_replay(runner, plan, binding, canonical, reference, receipt):
    condition, manifest = binding["condition"],binding["manifest"]
    run_id, streams = condition["run_id"],6*condition["latent_k"]
    artifacts = dict(route=dict(cache_relative_path=binding["artifact_relative"],sha256=binding["parent"]["artifact_sha256"]))
    origins = reference.get("origins",[])
    source_sha = runner.fingerprint({name:plan["source39"][name] for name in REPLAY_SOURCES})
    expected_reference = dict(run_id=run_id,split="V_select",route_sha=runner.fingerprint(canonical),
        standard_manifest_sha256=binding["manifest_sha256"],standard_archive_sha256=manifest["archive_sha256"],
        source_sha=source_sha,artifact_bindings=artifacts,expected_F_streams=streams,created_before_inference=True)
    expected_receipt = dict(status="PASS",run_id=run_id,origins=origins,route_sha=expected_reference["route_sha"],model_size="small",
        reference_sha=runner.fingerprint(reference),selected_artifact_bindings=artifacts,
        standard_manifest_public_relative_path=binding["manifest_relative"],standard_manifest_sha256=binding["manifest_sha256"],
        atol=1e-5,rtol=1e-4,actual_F_streams=streams,expected_F_streams=streams,F_physical_streams=streams,
        independent_training_rerun=False,T_final_accessed=False)
    if (len(origins) != 6 or len(set(origins)) != 6 or origins != sorted(origins)
            or not math.isfinite(reference.get("created_unix",float("nan")))
            or any(reference.get(key) != value for key,value in expected_reference.items())
            or any(receipt.get(key) != value for key,value in expected_receipt.items())):
        raise ValueError("This output requires its original six-origin PASS replay, fixed tolerance/source/route/artifact/saved-V binding")


def worker_arguments(spec_relative, output_relative, batch, model_relative):
    if batch not in (1,8): raise ValueError("Only original B1/B8 cost conditions are supported")
    return ["{PYTHON}","-X","utf8","-u","-B","{ORIGINAL_PUBLIC_ROOT}/measure_costs.py","--worker",
        "--route-spec","{POST_T_CACHE_ROOT}/"+spec_relative,"--public-root","{POST_T_PUBLIC_ROOT}","--cache-root","{POST_T_CACHE_ROOT}",
        "--model","{ASSET_ROOT}/"+model_relative,"--device","cpu","--batch",str(batch),"--warmups","10","--timed-trials","20","--stage-trials","20",
        "--output","{POST_T_PUBLIC_ROOT}/"+output_relative]


class ScienceRuntime:
    def __init__(self, public):
        sys.path.insert(0,str(public))
        import torch
        import numpy as np
        import data_contract
        import evaluation
        import fit_base
        import route_catalog
        for module in (data_contract,evaluation,fit_base,route_catalog):
            if Path(module.__file__).resolve() != public/(module.__name__+".py"):
                raise ValueError("Original scientific modules must resolve to the pinned source root")
        torch.set_num_threads(2)
        if torch.cuda.is_initialized(): raise ValueError("No CUDA initialization is permitted")
        self.torch,self.np,self.data,self.evaluation,self.fit,self.catalog = torch,np,data_contract,evaluation,fit_base,route_catalog

    def task(self, runner, plan, cache, public):
        pca = runner.load_dependency(public,runner.PCA_RUNNER_PATH,"multik_cost_task_validator")
        task = self.data.load_task("beijing",cache_dir=cache)
        evidence = pca.validate_task(task,dict(task_contracts=dict(beijing=plan["task_contract"])),"beijing",self.data,self.fit)
        return task,evidence

    def nested(self, verifier, runner, plan, values, paths):
        return verifier.validate_nested_arrays(runner,plan,values["manifest"],values["nested_contract"],paths["artifact"],self.np,self.evaluation,self.fit)

    def route(self, run_id, task, private, public):
        return self.catalog.build_route(run_id,task,private,public)


def execute(public, cache, asset, plan_sha, run_id, output, *, deadline=None, _runtime=None):
    if plan_sha != PLAN_SHA: raise ValueError("Explicit sealed multi-K PLAN SHA required")
    verifier,runner,plan,condition,state = metadata(public,cache,asset,run_id,plan_sha)
    post,private = public/runner.NAME,cache/runner.NAME
    runner.environment(cache,asset)
    if not state["ready"]: raise ValueError("Only the requested output's V/artifact/replay and existing registry alias must be ready")
    verifier.require_new_output(runner,output,post)
    if deadline is None or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("Explicit live deadline inside the fixed campaign wall required")
    used = sum(path.stat().st_size for root in (public,cache) for path in root.rglob("*") if path.is_file())
    if used+256*1024**2 > plan["storage_limit_bytes"]: raise ValueError("Original combined storage cap including 256MiB reserve reached")
    utility = runner.load_dependency(public,runner.UTILITY_PATH,"multik_cost_private_errors")
    source_sha = sha(__file__)
    claims = private/"cpu_cost_preparation_claims"
    claims.mkdir(parents=True,exist_ok=True)
    runner.new(claims/(run_id+".json"),dict(run_id=run_id,helper_sha256=source_sha,plan_sha256=plan_sha,claimed_unix=time.time()))
    output.mkdir(parents=True,exist_ok=False)
    runner.new(output/"STARTED.json",dict(run_id=run_id,helper_sha256=source_sha,plan_sha256=plan_sha,actual_pid=os.getpid(),workers_started=0))
    import psutil
    process,stop = psutil.Process(),threading.Event()
    def guard():
        while not stop.wait(.25):
            rss = process.memory_info().rss
            if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
                runner.new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss,deadline_unix=deadline)); os._exit(98)
    watch = threading.Thread(target=guard,daemon=True)
    watch.start()
    jobs,bindings,failures = [],[],[]
    error_class,post_error,prepared,preserved = None,None,False,False
    task_attempted,route_attempted,nested_evidence,phase = False,False,None,"selected_input_binding"
    token = "multik_"+str(time.time_ns())+"_"+uuid.uuid4().hex
    try:
        values,score,paths,bindings = verifier.prepare_inputs(runner,plan,public,cache,run_id,condition)
        alias = runner.relative(post,"EXPERIMENT_REGISTRY.csv")
        bindings.append(dict(root="public",relative_path=alias.relative_to(public).as_posix(),sha256=sha(alias)))
        parent = values.get("fit_receipt") or values["nested_contract"]
        manifest = values["manifest"]
        binding = dict(condition=condition,parent=parent,manifest=manifest,
            artifact_relative=paths["artifact"].relative_to(private).as_posix(),manifest_relative=paths["manifest"].relative_to(post).as_posix(),
            manifest_sha256=sha(paths["manifest"]),receipt_relative=paths["fit_receipt"].relative_to(post).as_posix() if "fit_receipt" in paths else None,
            receipt_sha256=sha(paths["fit_receipt"]) if "fit_receipt" in paths else None)
        replay_paths = requested_paths(verifier,runner,plan,post,private,condition)
        for key in ("replay_reference","replay_receipt","replay_archive"):
            path = replay_paths[key]
            root_name,root = ("public",public) if path.is_relative_to(public) else ("cache",cache)
            bindings.append(dict(root=root_name,relative_path=path.relative_to(root).as_posix(),sha256=sha(path)))
        reference,receipt = verifier.read(replay_paths["replay_reference"]),verifier.read(replay_paths["replay_receipt"])
        if receipt.get("cache_relative_path") != f"route_replays/{run_id}/V_subset.npz" or sha(replay_paths["replay_archive"]) != receipt.get("archive_sha256"):
            raise ValueError("Original successful replay archive byte identity differs")
        runtime = _runtime if _runtime is not None else ScienceRuntime(public)
        phase,task_attempted = "original_safeFIT_task",True
        task,task_binding = runtime.task(runner,plan,cache,public)
        runner.new(output/"TASK_BINDING.json",task_binding)
        if condition["kind"] == "nested":
            phase = "fixed_nested_full_basis_binding"
            nested_evidence = runtime.nested(verifier,runner,plan,values,paths)
        phase,route_attempted = "original_canonical_catalog_no_forward",True
        canonical,size = runtime.route(run_id,task,private,post)
        specification = cost_spec(canonical,size,binding,plan,runner.fingerprint)
        validate_replay(runner,plan,binding,canonical,reference,receipt)
        runner.new(output/"INPUT_BINDINGS.json",dict(files=bindings,condition=condition,selected_artifact_sha256=parent["artifact_sha256"],
            canonical_route=canonical,replay_receipt_sha256=sha(replay_paths["replay_receipt"]),nested_fixed_basis_evidence=nested_evidence,
            parent_receipts_at_capture=values["parent_receipts_at_capture"],other_eleven_outputs_are_not_a_gate=True))
        specifications = private/"cpu_cost_specs"/token
        specifications.mkdir(parents=True,exist_ok=False)
        phase = "write_two_cost_jobs_no_worker"
        for batch in (1,8):
            label = "B"+str(batch)
            path = specifications/(label+".json")
            runner.new(path,specification)
            relative = path.relative_to(private).as_posix()
            output_relative = "costs/"+token+"/"+label+".json"
            jobs.append(dict(job_id=label,run_id=run_id,accuracy_run_id=run_id,dataset="beijing",family=canonical["family"],registered_role=condition["role"],
                batch=batch,device="cpu",model_size="small",L=512,H=48,C=11,K=condition["latent_k"],V_origins=43,
                row_microbatch=128,checkpoint_inputs=False,warmups=10,timed_trials=20,stage_trials=20,expected_F_streams_per_trial=batch*condition["latent_k"],
                required_receipt_status="COMPLETE_PROTOCOL",spec_cache_relative_path=relative,spec_sha256=sha(path),
                output_public_relative_path=output_relative,original_selected_artifact_sha256=parent["artifact_sha256"],
                accuracy_manifest_public_relative_path=binding["manifest_relative"],accuracy_manifest_sha256=binding["manifest_sha256"],
                replay_receipt_sha256=sha(replay_paths["replay_receipt"]),checkpoint_subset_parity="BOUND_EXISTING_PASS_SIX_ORIGINS_FIXED_TOLERANCE",
                original_replay_receipt_records_device=False,ROOT_must_bind_actual_CPU_verifier_RESULT_before_cost=True,
                arguments=worker_arguments(relative,output_relative,batch,plan["model_asset_relative_path"])))
        prepared = True
    except Exception as error:
        error_class = type(error).__name__
        failures.append(utility.save_private_failure(private,cache,"multik_cost_preparation",run_id,phase,error))
    finally:
        try:
            metadata(public,cache,asset,run_id,plan_sha)
            roots = dict(public=public,cache=cache)
            if sha(__file__) != source_sha or any(sha(runner.relative(roots[row["root"]],row["relative_path"])) != row["sha256"] for row in bindings):
                raise ValueError("Original source/control or captured selected/replay input bytes changed")
            if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized(): raise ValueError("Unexpected CUDA initialization")
            preserved = True
        except Exception as error:
            post_error = type(error).__name__
            failures.append(utility.save_private_failure(private,cache,"multik_cost_preparation",run_id,"postflight",error))
        stop.set(); watch.join(timeout=2)
        success = prepared and error_class is None and post_error is None and preserved
        result = dict(status="PREPARED_NOT_EXECUTED_CPU_COST_WORKERS" if success else "FAILED_PRESERVED",run_id=run_id,
            post_T_plan_sha256=plan_sha,preparation_helper_sha256=source_sha,dependency_sha256=DEPENDENCIES,worker_count=len(jobs),jobs=jobs,
            original_source39_registry_scores_cost_T_and_selected_inputs_preserved=preserved,other_eleven_outputs_are_not_a_gate=True,
            error_class=error_class,postflight_error_class=post_error,private_failures=failures,source_bindings=dict(source39=plan["source39"],immutable_public=plan["immutable_public"]),
            original_safeFIT_task_load_call_attempted=task_attempted,original_route_builder_call_attempted=route_attempted,nested_fixed_basis_evidence=nested_evidence,
            actual_subprocesses_started=0,actual_cost_workers_started=0,actual_cost_trials=0,actual_training=0,actual_forecasting=0,actual_foundation_models_constructed=0,
            new_T_target_or_scoring=False,CPU_threads=2,max_RSS_bytes=plan["max_RSS_bytes"],deadline_unix=deadline,maximum_concurrent_cost_workers=1,
            external_process_tree_supervision_required=True,original_cost_worker_has_no_builtin_CPU_RSS_or_deadline_guard=True,
            ROOT_must_verify_all_training_inference_and_other_cost_workers_terminated=True,original_paths_preserved=True,registry_alias_created=False,
            selected_artifact_or_prediction_copies_created=False,worker_paths_scope="Original measure_costs source; explicit public=postT multi-K and cache=postT multi-K; new specs and cost outputs only",
            preparation_scope="Original safeFIT Task, hash-bound selected adapters or full fixed PCA basis; no Route/F construction, forecast, new fit, worker or numerical rescore",
            replay_scope="Existing hash-bound six-origin original replay PASS; receipt has no device field. ROOT must additionally bind actual frozen CPU verifier RESULT; preparation performs no forward or full V numerical rescore",
            archive_deserialization_scope="Original Task may physically deserialize/hash campaign T bytes; no T target or T evaluation API is requested",
            memory_scope="Original worker resident weights and after-close RSS; external sampled peak RSS required; training peak excluded",
            independent_generalization_claimed=False,original_GPU_minimum_gap_completed=False,
            actual_pid=process.pid,actual_create_time=process.create_time(),finished_unix=time.time())
        runner.new(output/"COST_EXECUTION_PLAN.JSON",result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ("public","cache","asset"): parser.add_argument("--"+root+"-root")
    parser.add_argument("--run-id"); parser.add_argument("--plan-sha"); parser.add_argument("--deadline",type=float)
    parser.add_argument("--output"); parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    roots = (args.public_root,args.cache_root,args.asset_root)
    if args.execute:
        if not all(roots+(args.run_id,args.plan_sha,args.deadline,args.output)):
            parser.error("Preparation requires original roots, one sealed ID, PLAN SHA, deadline and fresh output")
        public,cache,asset = (Path(value).resolve() for value in roots)
        result = execute(public,cache,asset,args.plan_sha,args.run_id,Path(args.output).resolve(),deadline=args.deadline)
        print(json.dumps(dict(status=result["status"],worker_count=result["worker_count"],actual_cost_trials=0)))
        return 0 if result["status"] == "PREPARED_NOT_EXECUTED_CPU_COST_WORKERS" else 1
    state = None
    if any(roots) or args.run_id:
        if not all(roots) or not args.run_id or not args.plan_sha:
            parser.error("Metadata validation requires all original roots, one sealed output and PLAN SHA")
        state = metadata(*(Path(value).resolve() for value in roots),args.run_id,args.plan_sha)[-1]
    print(json.dumps(dict(status="METADATA_ONLY_MULTIK_CPU_COST_PREPARATION",registered_outputs=12,requested_route_state=state,
        numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,psutil_imported="psutil" in sys.modules,
        actual_model_array_worker_or_cost_execution=0,files_written=0,helper_sha256=sha(__file__))),flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
