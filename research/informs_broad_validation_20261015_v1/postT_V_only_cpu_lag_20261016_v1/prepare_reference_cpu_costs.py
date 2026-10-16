"""Prepare one original Jena reference's two CPU cost jobs; start no worker.

Original public/cache roots stay authoritative for artifacts and accuracy.
Preparation loads no foundation, forecasts, trains, or certifies CPU V parity.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
PLAN_SHA = "fedf00ebb1807528514c0685a6b226fd6a466149531adc7a5791ffa07d92af35"
RUNNER_SHA = "36c75b38cd2c072bac4466129249cc5213c2df679ad80232d4ac511ab6d7e7f1"
TRUTH_SHA = "674debb60054e0005fc817f2efd5623d483de6b79915a86f05cd9d782511c2cc"
PROBE_SHA = "37691b6337f95ae61f02a7ec32728dd88d61a3f66ef05145e0457786a326473c"
REFERENCES = {
    "primary_jena_E1_RAW_BOLT_SMALL_k0_s92601": dict(family="RAW", registry_family="RAW_BOLT_SMALL", F_width=21, model_size="small",
        manifest_sha256="8423a4fbc3b1338ad04baa2108aee3001178b8791eaa5ee5708dc21b6b37b451"),
    "primary_jena_E1_RAW_RIDGE_k0_s92601": dict(family="RAW_RIDGE", registry_family="RAW_RIDGE", F_width=0, model_size=None,
        manifest_sha256="12012ac13367942f8cfdc81b14732039934d5aa3ed6c0dee8c24dc0728a53c42",
        receipt_file="RIDGE_FIT_RECEIPT.json", receipt_sha256="b0e851155e0e2f1d9dc5f3fb13cbdbe2393390de319f275f9e4a71993070541c"),
    "primary_jena_E1_DIRECT_G_k0_s92601__cpu_fresh_e3": dict(family="DIRECT_G", registry_family="DIRECT_G", F_width=0, model_size=None,
        manifest_sha256="8046b4b90e40ce907fefdd0f07857990db7947540288b295981e9cf318fc7f49",
        receipt_file="BASE_FIT_RECEIPT.json", receipt_sha256="3ee3f25b064fb62348a75a83ee6478942fdffd085a638753c98991d43bd20d90",
        source_sha256="f56c1cb86543cd82bcf36883b6ceb270f05df641d11f6c2b635b072f8279dc50")}
IDENTITY = ("run_id","split","cohort","source_sha","config_sha","data_sha","model_sha","scaler_sha")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""): result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def load_runner():
    path = HERE/"cpu_lag_runner.py"
    if sha(path) != RUNNER_SHA: raise ValueError("Frozen CPU stdlib utility changed")
    spec = importlib.util.spec_from_file_location("reference_cost_frozen_runner",path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def choose(run_id):
    if run_id not in REFERENCES: raise ValueError("One of the three explicitly approved original Jena reference IDs is required")
    return REFERENCES[run_id]


def reference_binding(runner, plan, public, cache, run_id):
    selected, task = choose(run_id),plan["task_contract"]
    folder = public/"predictions"/run_id/"V_select"
    paths = dict(manifest=folder/"PREDICTION_MANIFEST.json",reference=folder/"REFERENCE_CONTRACT.json",score=folder/"SCORES.csv")
    if selected["family"] == "RAW": paths["probe"] = public/"PROBE_ACTUAL_GPU.json"
    else: paths["receipt"] = public/"runs"/run_id/selected["receipt_file"]
    if selected["family"] == "DIRECT_G": paths["source"] = public/"runs"/run_id/"SOURCE_CONTRACT.json"
    missing = [path.relative_to(public).as_posix() for path in paths.values() if not path.is_file()]
    if missing: return None,missing
    expected_pins = dict(manifest=selected["manifest_sha256"])
    if "receipt" in paths: expected_pins["receipt"] = selected["receipt_sha256"]
    if "source" in paths: expected_pins["source"] = selected["source_sha256"]
    if "probe" in paths: expected_pins["probe"] = PROBE_SHA
    if any(sha(paths[key]) != digest for key,digest in expected_pins.items()): raise ValueError("Approved original reference metadata identity changed")
    values = {key:runner.read(path) for key,path in paths.items() if key != "score"}
    manifest, reference = values["manifest"],values["reference"]
    meta,score,cost = manifest["metadata"],manifest["score"],manifest["cost_record"]
    shapes = dict(pred_normalized=[43,48,21],target_normalized=[43,48,21],observed=[43,48,21],origins=[43],channel_ids=[21])
    if (manifest.get("run_id") != run_id or manifest.get("split") != "V_select" or manifest.get("cohort") != "jena:primary:V_select:stride48"
            or manifest.get("data_sha") != task["data_sha256"] or manifest.get("scaler_sha") != task["refit_scaler_sha256"]
            or meta.get("family") != selected["family"] or meta.get("fold") != "primary" or meta.get("dense") is not False
            or meta.get("role") != "base_parent" or meta.get("parent_fit_end") != 12579 or meta.get("canonical_scaler_fit_end") != 12579
            or set(manifest["arrays"]) != set(shapes) or any(manifest["arrays"][key]["shape"] != shape for key,shape in shapes.items())
            or any(reference.get(key) != manifest.get(key) for key in IDENTITY)
            or reference.get("arrays") != {key:manifest["arrays"][key] for key in shapes if key != "pred_normalized"}
            or fingerprint(reference["arrays"]) != TRUTH_SHA):
        raise ValueError("Original V43/H48/C21 task, truth, reference or scaler differs")
    expected_cost = dict(run_id=run_id,batch=43,latent_k=selected["F_width"],actual_F_streams=43*selected["F_width"],expected_F_streams=43*selected["F_width"],F_calls=1)
    if any(cost.get(key) != value for key,value in expected_cost.items()) or reference.get("expected_F_streams") != expected_cost["expected_F_streams"]:
        raise ValueError("Reference F ledger must distinguish Raw C channels from zero-F G/ridge")
    sources = {key:plan["source39"][key] for key in ("prediction_cache.py","fit_base.py","data_contract.py","training.py","evaluation.py","networks.py","gpu_runtime.py")}
    if selected["family"] == "RAW_RIDGE": sources["ridge_baseline.py"] = plan["source39"]["ridge_baseline.py"]
    if meta.get("source_files") != sources or manifest["source_sha"] != fingerprint(dict(deployment_sources=sources,training_source_sha=meta["training_source_sha"])):
        raise ValueError("Original prediction scientific source binding differs")
    registry = [row for row in runner.csv_rows(public/"EXPERIMENT_REGISTRY.csv") if row["experiment_id"] == run_id]
    registered = dict(dataset="jena",family=selected["registry_family"],fold="primary",seed="92601",latent_k="0",length="512",horizon="48",execution_status="COMPLETE_VALID")
    if len(registry) != 1 or any(registry[0].get(key) != value for key,value in registered.items()):
        raise ValueError("Original registered reference identity is absent or incomplete")
    global_scores = [row for row in runner.csv_rows(public/"SCORES.csv") if row["run_id"] == run_id and row["split"] == "V_select"]
    if len(global_scores) != 1: raise ValueError("Exactly one original approved V accuracy row required")
    row = global_scores[0]
    if (row.get("truth_cohort_sha") != TRUTH_SHA or row.get("manifest_relative_path") != paths["manifest"].relative_to(public).as_posix()
            or any(row.get(key) != str(manifest[key]) for key in ("archive_sha256","source_sha","config_sha","model_sha","data_sha","scaler_sha"))):
        raise ValueError("Frozen global score/reference identity differs")
    local_scores = runner.csv_rows(paths["score"])
    if len(local_scores) != 1: raise ValueError("Exactly one original per-run V score row required")
    for key,value in score.items():
        for actual in (local_scores[0],row):
            if actual is row and key == "metric_scale" and key not in row:
                if value != "A_normalized_original_channels": raise ValueError("Unexpected original score units")
                continue
            if (isinstance(value,(int,float)) and float(actual[key]) != value) or (not isinstance(value,(int,float)) and actual[key] != str(value)):
                raise ValueError("Original CSV score differs from approved saved V")
    receipt = values.get("receipt")
    if receipt is not None:
        scaler_key, end_key = ("scaler_sha","fit_end") if selected["family"] == "RAW_RIDGE" else ("refit_scaler_sha","refit_fit_end")
        if (receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != run_id or receipt.get("family") != selected["family"]
                or receipt.get(scaler_key) != task["refit_scaler_sha256"] or receipt.get(end_key) != 12579
                or meta.get("artifact_sha256") != receipt.get("artifact_sha256") or meta.get("training_source_sha") != receipt.get("source_sha")):
            raise ValueError("Original selected trained artifact/source/scaler differs")
        if selected["family"] == "DIRECT_G":
            source = values["source"]
            if source.get("model_sha") != "NO_FOUNDATION" or source["spec"].get("device") != "cpu" or source["spec"].get("family") != "DIRECT_G":
                raise ValueError("Direct G must retain its exact CPU-trained zero-foundation recipe")
        elif receipt.get("source_files") != {key:plan["source39"][key] for key in ("data_contract.py","ridge_baseline.py")}:
            raise ValueError("Native ridge fitting source differs")
        if not runner.relative(cache,receipt["artifact_cache_relative_path"]).is_file(): missing.append(receipt["artifact_cache_relative_path"])
    elif (values["probe"].get("model_state_sha256") != plan["foundation_tensor_sha256"] or values["probe"].get("model_revision") != plan["model_revision"]):
        raise ValueError("Raw foundation probe identity differs")
    if manifest["cache_relative_path"] != f"predictions/{run_id}/V_select.npz": raise ValueError("Original standard-V layout required")
    if not runner.relative(cache,manifest["cache_relative_path"]).is_file(): missing.append(manifest["cache_relative_path"])
    if missing: return None,missing
    return dict(run_id=run_id,family=selected["family"],F_width=selected["F_width"],model_size=selected["model_size"],receipt=receipt,
        manifest=manifest,metadata_pins={path.relative_to(public).as_posix():sha(path) for path in paths.values()},
        manifest_relative=paths["manifest"].relative_to(public).as_posix(),receipt_relative=paths["receipt"].relative_to(public).as_posix() if receipt else None,
        original_training_status=row["training_status"],original_checkpoint_replay_status=row.get("checkpoint_replay"),
        original_fit_device="cpu" if selected["family"] in {"RAW_RIDGE","DIRECT_G"} else "PRETRAINED_EXTERNAL",
        original_V_device_evidence="DIRECT_G predict_parent consumes immutable saved spec device=cpu; ridge uses native CPU FP64; Raw V manifest records no inference device",
        cpu_vs_original_V_numeric_parity="NOT_CHECKED_BY_CANONICAL_COST_PREPARATION_OR_ORIGINAL_COST_WORKER"), []


def metadata(public, cache, asset, run_id, plan_sha=None):
    choose(run_id)
    if plan_sha is not None and plan_sha != PLAN_SHA: raise ValueError("Only the frozen original Jena post-T PLAN is supported")
    runner = load_runner()
    plan = runner.metadata(public,cache,asset,PLAN_SHA)
    try:
        binding,missing = reference_binding(runner,plan,Path(public),Path(cache),run_id)
        error_class = None
    except (OSError,ValueError,KeyError,TypeError) as error:
        binding,missing,error_class = None,[],type(error).__name__
    return dict(plan=plan,binding=binding,missing=missing,route_error_class=error_class,ready=binding is not None,
                only_requested_run_id_checked=run_id,other_two_references_are_not_a_gate=True)


def cost_spec(canonical, size, binding, plan):
    run_id,manifest,receipt = binding["run_id"],binding["manifest"],binding["receipt"]
    expected = dict(run_id=run_id,dataset="jena",family=binding["family"],fold="primary",latent_k=binding["F_width"],canonical_fit_end=12579,model_size=binding["model_size"])
    if receipt is None:
        expected.update(foundation_sha=plan["foundation_tensor_sha256"],foundation_revision=plan["model_revision"],expected_prediction_model_sha=manifest["model_sha"])
        if "artifact_cache_relative_path" in canonical: raise ValueError("Raw must use its original pretrained route without invented adapter artifacts")
    else:
        expected.update(artifact_cache_relative_path=receipt["artifact_cache_relative_path"],artifact_sha256=receipt["artifact_sha256"],
            fit_receipt_public_relative_path=binding["receipt_relative"],fit_receipt_sha256=binding["metadata_pins"][binding["receipt_relative"]],
            foundation_sha="NO_FOUNDATION",expected_prediction_model_sha=receipt["model_sha"] if binding["family"] == "RAW_RIDGE" else manifest["model_sha"])
    standard = dict(model_sha=manifest["model_sha"],archive_sha256=manifest["archive_sha256"],manifest_public_relative_path=binding["manifest_relative"],manifest_sha256=binding["metadata_pins"][binding["manifest_relative"]])
    if size != binding["model_size"] or any(canonical.get(key) != value for key,value in expected.items()) or canonical.get("standard_V_binding") != standard or "accuracy_run_id" in canonical or "row_microbatch" in canonical:
        raise ValueError("Original canonical accuracy route differs; G/ridge model_size=None and ridge's native/model-V identities must remain distinct")
    return dict(canonical,accuracy_run_id=run_id,row_microbatch=128)


def worker_arguments(spec_relative, output_relative, batch, model_relative):
    return ["{PYTHON}","-X","utf8","-u","-B","{ORIGINAL_PUBLIC_ROOT}/measure_costs.py","--worker",
        "--route-spec","{POST_T_CACHE_ROOT}/"+spec_relative,"--public-root","{ORIGINAL_PUBLIC_ROOT}","--cache-root","{ORIGINAL_CACHE_ROOT}",
        "--model","{ASSET_ROOT}/"+model_relative,"--device","cpu","--batch",str(batch),"--warmups","10","--timed-trials","20","--stage-trials","20",
        "--output","{POST_T_PUBLIC_ROOT}/"+output_relative]


class ScienceRuntime:
    def __init__(self, public):
        sys.path.insert(0,str(public))
        import torch
        import data_contract
        import fit_base
        import route_catalog
        torch.set_num_threads(2)
        if torch.cuda.is_initialized(): raise ValueError("CPU-only route preparation required")
        self.data,self.fit,self.catalog = data_contract,fit_base,route_catalog

    def task(self, runner, plan, cache):
        task = self.data.load_task("jena",cache_dir=cache)
        return task,runner.validate_task(task,plan,self.data,self.fit)

    def route(self, run_id, task, cache, public):
        return self.catalog.build_route(run_id,task,cache,public)


def bound_archives(runner, cache, binding):
    expected = {binding["manifest"]["cache_relative_path"]:binding["manifest"]["archive_sha256"]}
    if binding["receipt"] is not None: expected[binding["receipt"]["artifact_cache_relative_path"]] = binding["receipt"]["artifact_sha256"]
    if any(sha(runner.relative(cache,name)) != digest for name,digest in expected.items()): raise ValueError("Original selected artifact or standard V bytes changed")
    return expected


def execute(public, cache, asset, plan_sha, run_id, output, *, deadline=None, _runtime=None):
    if plan_sha != PLAN_SHA: raise ValueError("Explicit frozen original Jena PLAN SHA required")
    state = metadata(public,cache,asset,run_id,plan_sha)
    runner,plan = load_runner(),state["plan"]
    post,private = public/runner.NAME,cache/runner.NAME
    if not state["ready"]: raise ValueError("Only the requested original reference must be ready")
    if output == post or not output.is_relative_to(post) or output.exists(): raise FileExistsError("A fresh preparation output in the separate post-T public root is required")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any(key.startswith("INFORMS_GPU_") for key in os.environ)
            or any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"))
            or Path(os.environ.get("INFORMS_ASSET_ROOT","")).resolve() != asset
            or Path(os.environ.get("INFORMS_CACHE_ROOT","")).resolve() != private): raise ValueError("Separate-cache CPU2/offline/no-GPU environment required")
    if deadline is None or not time.time() < deadline <= plan["deadline_unix"]: raise ValueError("Explicit live deadline inside original campaign wall required")
    used = sum(path.stat().st_size for root in (public,cache) for path in root.rglob("*") if path.is_file())
    if used+64*1024**2 > plan["storage_limit_bytes"]: raise ValueError("Original combined 30GiB storage cap reached")
    claims = private/"reference_cost_preparation_claims"; claims.mkdir(parents=True,exist_ok=True)
    source_sha = sha(__file__)
    runner.new(claims/(run_id+".json"),dict(run_id=run_id,plan_sha256=plan_sha,helper_sha256=source_sha,claimed_unix=time.time()))
    output.mkdir(parents=True,exist_ok=False)
    runner.new(output/"STARTED.json",dict(run_id=run_id,helper_sha256=source_sha,plan_sha256=plan_sha,actual_pid=os.getpid(),actual_workers_started=0))
    import psutil
    process,stop = psutil.Process(),threading.Event()
    def guard():
        while not stop.wait(.25):
            rss = process.memory_info().rss
            if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
                runner.new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss,deadline_unix=deadline)); os._exit(98)
    watch = threading.Thread(target=guard,daemon=True); watch.start()
    token = "reference_"+str(time.time_ns())+"_"+uuid.uuid4().hex
    specifications = private/"reference_cost_specs"/token
    jobs,failures,binding = [],[],state["binding"]
    failed,post_error,preserved,prepared,task_attempted,route_attempted = None,None,False,False,False,False
    try:
        archives = bound_archives(runner,cache,binding)
        runtime = _runtime if _runtime is not None else ScienceRuntime(public)
        task_attempted = True
        task,task_binding = runtime.task(runner,plan,cache)
        runner.new(output/"TASK_BINDING.json",task_binding)
        route_attempted = True
        canonical,size = runtime.route(run_id,task,cache,public)
        specification = cost_spec(canonical,size,binding,plan)
        after = metadata(public,cache,asset,run_id,plan_sha)
        if not after["ready"] or after["binding"] != binding or sha(__file__) != source_sha: raise ValueError("Original reference metadata changed during preparation")
        specifications.mkdir(parents=True,exist_ok=False)
        for batch in (1,8):
            label = binding["family"]+"_B"+str(batch)
            path = specifications/(label+".json"); runner.new(path,specification)
            spec_relative = path.relative_to(private).as_posix()
            output_relative = "costs/"+token+"/"+label+".json"
            jobs.append(dict(job_id=label,run_id=run_id,accuracy_run_id=run_id,dataset="jena",family=binding["family"],model_size=size,
                batch=batch,device="cpu",L=512,H=48,C=21,K=binding["F_width"],registered_latent_k=0,V_origins=43,
                row_microbatch=128,checkpoint_inputs=False,warmups=10,timed_trials=20,stage_trials=20,
                expected_F_streams_per_trial=batch*binding["F_width"],required_receipt_status="COMPLETE_PROTOCOL",
                spec_cache_relative_path=spec_relative,spec_sha256=sha(path),original_artifact_binding=canonical.get("artifact_cache_relative_path"),
                original_selected_artifact_sha256=canonical.get("artifact_sha256"),standard_V_binding=specification["standard_V_binding"],
                original_metadata_pins=binding["metadata_pins"],original_archive_pins=archives,output_public_relative_path=output_relative,
                arguments=worker_arguments(spec_relative,output_relative,batch,plan["model_asset_relative_path"]),
                original_public_and_cache_roots_must_be_used=True,cpu_V_parity_gate_required_before_cost=True,
                cpu_V_parity_status="NOT_EXECUTED_BY_THIS_HELPER_OR_ORIGINAL_COST_WORKER",fixed_parity_atol=1e-5,fixed_parity_rtol=1e-4))
        bound_archives(runner,cache,binding)
        prepared = True
    except Exception as error:
        failed = type(error).__name__
        failures.append(runner.save_private_failure(private,cache,"reference_cost_preparation",run_id,"preparation",error))
    finally:
        stop.set(); watch.join(timeout=2)
        try:
            after = metadata(public,cache,asset,run_id,plan_sha)
            preserved = after["ready"] and after["binding"] == binding and sha(__file__) == source_sha
            bound_archives(runner,cache,binding)
            if not preserved: raise ValueError("Original selected reference source/metadata changed")
        except Exception as error:
            preserved = False; post_error = type(error).__name__
            failures.append(runner.save_private_failure(private,cache,"reference_cost_preparation",run_id,"postflight",error))
        success = prepared and failed is None and post_error is None and preserved
        result = dict(status="PREPARED_NOT_EXECUTED_CPU_COST_WORKERS" if success else "FAILED_PRESERVED",run_id=run_id,
            post_T_plan_sha256=plan_sha,preparation_helper_sha256=source_sha,runner_sha256=RUNNER_SHA,worker_count=len(jobs),jobs=jobs,
            original_source39_registry_scores_cost_T_and_selected_inputs_preserved=preserved,other_two_references_are_not_a_gate=True,
            error_class=failed,postflight_error_class=post_error,private_failures=failures,source_bindings=dict(source39=plan["source39"],immutable_public=plan["immutable_public"],selected_metadata=binding["metadata_pins"]),
            original_training_status=binding["original_training_status"],original_fit_device=binding["original_fit_device"],original_V_device_evidence=binding["original_V_device_evidence"],
            original_checkpoint_replay_status=binding["original_checkpoint_replay_status"],cpu_V_numeric_parity=binding["cpu_vs_original_V_numeric_parity"],
            actual_subprocesses_started=0,actual_cost_workers_started=0,actual_cost_trials=0,actual_training=0,actual_forecasting=0,actual_foundation_models_constructed=0,
            original_safeFIT_task_load_call_attempted=task_attempted,original_route_builder_call_attempted=route_attempted,new_T_target_or_scoring=False,
            CPU_threads=2,max_RSS_bytes=plan["max_RSS_bytes"],deadline_unix=deadline,maximum_concurrent_cost_workers=1,
            original_cost_worker_has_no_builtin_CPU_RSS_or_deadline_guard=True,external_process_tree_supervision_required=True,
            ROOT_must_verify_all_training_and_other_cost_workers_terminated=True,
            cost_dtype_scope="Native ridge CPU FP64, G/Raw FP32; physical output and inverse transform FP64; no matched-compute-dtype claim",
            F_scope="Only Raw uses C=21 foundation streams per history; G/ridge load no foundation despite a pinned --model argument",
            preparation_scope="Original safeFIT task plus original hash-bound adapter/ridge artifacts; no Route construction, F construction, forecast or numerical parity",
            archive_deserialization_scope="Original Jena trainval members decompress before FIT slice, including old TUNE/VAL; protected old TEST file is not opened. Campaign T bytes may be deserialized/hashed but no T targets/evaluation requested.",
            original_paths_preserved=True,receipt_or_artifact_copies_created=False,worker_paths_scope="--public-root ORIGINAL_R --cache-root ORIGINAL_C; only route spec and new cost output use separate post-T roots",
            memory_scope="Original worker resident weights and after-close RSS; external sampled peak RSS required; no GPU-memory-as-CPU-memory claim",
            actual_pid=process.pid,actual_create_time=process.create_time(),finished_unix=time.time())
        runner.new(output/"COST_EXECUTION_PLAN.JSON",result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ("public","cache","asset"): parser.add_argument("--"+root+"-root")
    parser.add_argument("--run-id",choices=tuple(REFERENCES)); parser.add_argument("--plan-sha")
    parser.add_argument("--deadline",type=float); parser.add_argument("--output"); parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    roots = (args.public_root,args.cache_root,args.asset_root)
    if args.execute:
        if not all(roots+(args.run_id,args.plan_sha,args.deadline,args.output)): parser.error("Execution requires original roots, one ID, frozen PLAN SHA, deadline and new output")
        public,cache,asset = (Path(value).resolve() for value in roots)
        result = execute(public,cache,asset,args.plan_sha,args.run_id,Path(args.output).resolve(),deadline=args.deadline)
        print(json.dumps(dict(status=result["status"],worker_count=result["worker_count"],actual_cost_trials=0)))
        return 0 if result["status"] == "PREPARED_NOT_EXECUTED_CPU_COST_WORKERS" else 1
    state = None
    if any(roots) or args.run_id:
        if not all(roots) or not args.run_id: parser.error("Metadata validation requires all original roots and exactly one approved ID")
        state = metadata(*(Path(value).resolve() for value in roots),args.run_id,args.plan_sha)
    print(json.dumps(dict(status="METADATA_ONLY_REFERENCE_CPU_COST_PREPARATION",approved_references=3,requested_route_ready=state["ready"] if state else None,
        missing_metadata_files=state["missing"] if state else [],route_error_class=state["route_error_class"] if state else None,
        numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,actual_model_array_worker_or_cost_execution=0,files_written=0,
        helper_sha256=sha(__file__),CPU_V_parity_certified=False)),flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
