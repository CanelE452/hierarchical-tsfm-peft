"""Prepare original CPU K-transition workers for one sealed post-T strategy.

Default: stdlib metadata only. Explicit preparation creates plans, never a child,
fit, foundation, forecast or measurement. Actual workers require external guards.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
PLAN_SHA = "2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec"
COST_HELPER_SHA = "c5a4495a12e41a607c8172f10e1f701b69373ca047059d262f3d9102d07f88b0"
COST_TEST_SHA = "6d0e4323cb00adecb24125598aac3cf8523224844ab4f03fd16592ee162b72fb"
STRATEGIES = ("shared","posthoc","nested","independent_hot","independent_cold")
GRID = [3,6,11]
PROTOCOL = dict(directed_transitions=6,warmup_per_transition=10,timed_per_transition=20,
                total_trials_per_worker=180,timed_trials_per_worker=120,worker_scopes=10)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""): value.update(block)
    return value.hexdigest()


def load_cost_preparer():
    path = HERE/"prepare_multik_cpu_costs.py"
    if sha(path) != COST_HELPER_SHA or sha(HERE/"test_prepare_multik_cpu_costs.py") != COST_TEST_SHA:
        raise ValueError("Frozen canonical CPU cost preparer or test changed")
    spec = importlib.util.spec_from_file_location("transition_frozen_cpu_cost_preparer",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strategy_map(plan):
    rows = plan["outputs"]
    if len(rows) != 12 or len({row["run_id"] for row in rows}) != 12:
        raise ValueError("Exactly twelve unique sealed output identities required")
    result = {}
    for strategy in STRATEGIES:
        selected = []
        for row in rows:
            if ((strategy == "shared" and row["kind"] == "ordered" and row["variant"] == "shared")
                    or (strategy == "posthoc" and row["kind"] == "ordered" and row["variant"] in {"posthoc","max_only"})
                    or (strategy == "nested" and row["kind"] == "nested" and row["variant"] == "nested_pca")
                    or (strategy.startswith("independent_") and row["kind"] == "independent")):
                selected.append(row)
        if len(selected) != 3 or sorted(row["latent_k"] for row in selected) != GRID or any(row["split"] != "V_select" for row in selected):
            raise ValueError("Each original strategy requires exactly K3/K6/K11 V outputs")
        result[strategy] = {str(row["latent_k"]):row["run_id"] for row in sorted(selected,key=lambda row:row["latent_k"])}
    if len({value for group in result.values() for value in group.values()}) != 12:
        raise ValueError("The five maps must bind all twelve outputs; hot/cold share only the independent controls")
    return result


def selected_conditions(plan, strategy):
    if strategy not in STRATEGIES: raise ValueError("One explicit original strategy is required; all is not a fresh-process scope")
    mapping = strategy_map(plan)[strategy]
    by_id = {row["run_id"]:row for row in plan["outputs"]}
    conditions = [by_id[mapping[str(k)]] for k in GRID]
    if strategy == "nested":
        if any(row["parent_run_id"] is not None for row in conditions):
            raise ValueError("Nested PCA uses its original common fixed basis, without an invented fit parent")
        return conditions
    parents = {row["run_id"]:row for row in plan["fits"]}
    for condition in conditions:
        parent = parents[condition["parent_run_id"]]
        spec = parent["spec"]
        if (spec.get("family") != "B0" or spec.get("device") != "cpu" or spec.get("epochs") != [.25,1.,3.]
                or spec.get("F_row_microbatch") != 128 or spec.get("checkpoint_inputs") is not False):
            raise ValueError("Original selected B0 recipe/device/checkpoint-input conditions changed")
        if strategy == "shared" and (parent["role"] != "ORDERED_MULTI_K" or spec.get("k_grid") != GRID or spec.get("latent_k") != 11):
            raise ValueError("Shared strategy requires the genuine jointly trained three-K parent")
        if strategy == "posthoc" and (parent["role"] != "MAXK_PCA_TRAINED" or spec.get("k_grid") != [11] or spec.get("latent_k") != 11):
            raise ValueError("Posthoc strategy requires maximum-only training grid [11]")
        if strategy.startswith("independent_") and (parent["role"] != "INDEPENDENT_B0" or "k_grid" in spec or spec.get("latent_k") != condition["latent_k"]):
            raise ValueError("Independent controls must retain each original single-K B0 recipe")
    if not strategy.startswith("independent_") and len({row["parent_run_id"] for row in conditions}) != 1:
        raise ValueError("A shared/posthoc strategy must use exactly one selected parent")
    return conditions


def strategy_readiness(cost, verifier, runner, plan, public, private, strategy):
    states = [cost.readiness(verifier,runner,plan,public,private,row) for row in selected_conditions(plan,strategy)]
    return dict(ready=all(row["ready"] for row in states),strategy=strategy,outputs=states,other_strategies_are_not_a_gate=True,
        original_worker_minimum_scope="All three K within this strategy; no valid two-K original worker mode",
        CPU_verifier_proof_and_selected_input_validation_pending=True)


def metadata(public, cache, asset, strategy, plan_sha=None):
    if plan_sha is not None and plan_sha != PLAN_SHA: raise ValueError("Exact sealed multi-K PLAN required")
    cost = load_cost_preparer()
    verifier,runner,plan,_,_ = cost.metadata(public,cache,asset,plan_sha=PLAN_SHA)
    state = strategy_readiness(cost,verifier,runner,plan,public/runner.NAME,cache/runner.NAME,strategy)
    return cost,verifier,runner,plan,state


def validate_artifact_sharing(strategy, bindings):
    if strategy not in STRATEGIES or len(bindings) != 3: raise ValueError("One strategy and all three K bindings required")
    identities = {(row["artifact_relative"],row["parent"]["artifact_sha256"]) for row in bindings}
    paths = {row["artifact_relative"] for row in bindings}
    expected = 3 if strategy.startswith("independent_") else 1
    if len(identities) != expected or len(paths) != expected:
        raise ValueError("Original strategy's one shared/basis artifact or three independent artifact identities differ")
    return dict(unique_artifact_paths=len(paths),unique_path_sha256_identities=len(identities),
                artifact_files_copied=False,common_foundation_storage_excluded=True)


def validate_cpu_replay_result(cost, condition, receipt, value):
    expected = dict(status="PASS",run_id=condition["run_id"],mode="replay",plan_sha256=PLAN_SHA,
        verifier_sha256=cost.DEPENDENCIES["verify_multik_outputs.py"],actual_checkpoint_forward_in_this_command=True,
        checkpoint_replay_call_attempted=True,checkpoint_replay_receipt_returned=True,
        original_source_registry_selection_T_and_captured_inputs_preserved=True,new_T_target_or_scoring=False,
        saved_full_V_rescoring=False,subset_checkpoint_replay=True,training_rerun=False,cuda_initialized=False,
        full_V_origin_count=43,replay_origin_count=6,dataset="beijing",horizon=48,channels=11,
        latent_k=condition["latent_k"],registered_role=condition["role"],output_kind=condition["kind"])
    if (any(value.get(key) != expected_value for key,expected_value in expected.items())
            or value.get("scientific_result") != dict(receipt,executed_now=True)):
        raise ValueError("A hash-bound actual fresh CPU verifier PASS must match this original six-origin receipt exactly; cache-hit proof is insufficient")


def load_proof_index(runner, plan, public, cache, strategy, path, expected_sha):
    path = Path(path).absolute()
    if not path.is_relative_to(cache): raise ValueError("Explicit proof index must stay in the private campaign cache")
    path = runner.relative(cache,path.relative_to(cache))
    if not re.fullmatch("[0-9a-f]{64}",expected_sha or "") or sha(path) != expected_sha or path.stat().st_size > 8*1024**2:
        raise ValueError("CPU proof index byte identity or small-metadata bound differs")
    index = runner.read(path)
    if index.get("schema") != "MULTIK_CPU_REPLAY_PROOFS_v1" or not isinstance(index.get("outputs"),list) or not 1 <= len(index["outputs"]) <= 12:
        raise ValueError("Explicit CPU replay proof index schema differs")
    known = {row["run_id"] for row in plan["outputs"]}
    rows = {}
    for row in index["outputs"]:
        if set(row) != {"run_id","public_relative_path","sha256"} or row["run_id"] not in known or row["run_id"] in rows:
            raise ValueError("Proof identities must be distinct sealed outputs")
        if not re.fullmatch("[0-9a-f]{64}",row["sha256"] or ""): raise ValueError("Explicit proof SHA required")
        rows[row["run_id"]] = row
    result = {}
    bindings = [dict(root="cache",relative_path=path.relative_to(cache).as_posix(),sha256=expected_sha)]
    for condition in selected_conditions(plan,strategy):
        if condition["run_id"] not in rows: raise ValueError("All three selected K need explicit CPU verifier proofs")
        row = rows[condition["run_id"]]
        target = runner.relative(public,row["public_relative_path"])
        if not target.is_relative_to(public/runner.NAME) or target.name != "RESULT.json" or target.stat().st_size > 8*1024**2 or sha(target) != row["sha256"]:
            raise ValueError("CPU verifier proof must be exact metadata in the separate post-T namespace")
        result[row["run_id"]] = runner.read(target)
        bindings.append(dict(root="public",relative_path=target.relative_to(public).as_posix(),sha256=row["sha256"]))
    return result,bindings


def worker_arguments(plan_relative, output_relative, batch, strategy, model_relative):
    if batch not in (1,8) or strategy not in STRATEGIES: raise ValueError("One original strategy per fresh B1/B8 worker required")
    return ["{PYTHON}","-X","utf8","-u","-B","{ORIGINAL_PUBLIC_ROOT}/measure_multik_transition.py","--worker",
        "--plan-json","{POST_T_CACHE_ROOT}/"+plan_relative,"--public-root","{POST_T_PUBLIC_ROOT}","--cache-root","{POST_T_CACHE_ROOT}",
        "--model","{ASSET_ROOT}/"+model_relative,"--device","cpu","--batch",str(batch),"--strategy",strategy,
        "--warmups","10","--timed-trials","20","--output","{POST_T_PUBLIC_ROOT}/"+output_relative]


def execute(public, cache, asset, plan_sha, strategy, output, *, deadline=None,
            cpu_replay_proofs=None, cpu_replay_proofs_sha=None, _runtime=None):
    if plan_sha != PLAN_SHA: raise ValueError("Explicit sealed multi-K PLAN SHA required")
    cost,verifier,runner,plan,state = metadata(public,cache,asset,strategy,plan_sha)
    post,private = public/runner.NAME,cache/runner.NAME
    runner.environment(cache,asset)
    verifier.require_new_output(runner,output,post)
    if not state["ready"]: raise ValueError("This strategy's three V/artifact/replay outputs and existing registry alias must be ready")
    if deadline is None or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("Live deadline inside the fixed original campaign wall required")
    if cpu_replay_proofs is None or cpu_replay_proofs_sha is None: raise ValueError("Explicit private index/SHA of actual CPU verifier RESULTS required")
    proofs,bindings = load_proof_index(runner,plan,public,cache,strategy,cpu_replay_proofs,cpu_replay_proofs_sha)
    used = sum(path.stat().st_size for root in (public,cache) for path in root.rglob("*") if path.is_file())
    if used+256*1024**2 > plan["storage_limit_bytes"]: raise ValueError("Original combined storage cap including 256MiB reserve reached")
    utility = runner.load_dependency(public,runner.UTILITY_PATH,"transition_cost_private_errors")
    source_sha = sha(__file__)
    claims = private/"transition_cost_preparation_claims"
    claims.mkdir(parents=True,exist_ok=True)
    runner.new(claims/(strategy+".json"),dict(strategy=strategy,helper_sha256=source_sha,plan_sha256=plan_sha,claimed_unix=time.time()))
    output.mkdir(parents=True,exist_ok=False)
    runner.new(output/"STARTED.json",dict(strategy=strategy,helper_sha256=source_sha,plan_sha256=plan_sha,actual_pid=os.getpid(),workers_started=0))
    import psutil
    process,stop = psutil.Process(),threading.Event()
    def guard():
        while not stop.wait(.25):
            rss = process.memory_info().rss
            if rss > plan["max_RSS_bytes"] or time.time() >= deadline:
                runner.new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=rss,deadline_unix=deadline)); os._exit(98)
    watcher = threading.Thread(target=guard,daemon=True)
    watcher.start()
    jobs,selected,nested_evidence,failures = [],[],[],[]
    error_class,post_error,prepared,preserved = None,None,False,False
    task_attempted,route_attempted,phase = False,False,"selected_inputs"
    token = "transition_"+str(time.time_ns())+"_"+uuid.uuid4().hex
    try:
        runtime = _runtime if _runtime is not None else cost.ScienceRuntime(public)
        phase,task_attempted = "original_safeFIT_Task",True
        task,task_binding = runtime.task(runner,plan,cache,public)
        runner.new(output/"TASK_BINDING.json",task_binding)
        alias = runner.relative(post,"EXPERIMENT_REGISTRY.csv")
        bindings.append(dict(root="public",relative_path=alias.relative_to(public).as_posix(),sha256=sha(alias)))
        for condition in selected_conditions(plan,strategy):
            run_id = condition["run_id"]
            phase = "selected_input_binding_"+run_id
            values,score,paths,captured = verifier.prepare_inputs(runner,plan,public,cache,run_id,condition)
            bindings.extend(captured)
            parent = values.get("fit_receipt") or values["nested_contract"]
            binding = dict(condition=condition,parent=parent,manifest=values["manifest"],
                artifact_relative=paths["artifact"].relative_to(private).as_posix(),manifest_relative=paths["manifest"].relative_to(post).as_posix(),
                manifest_sha256=sha(paths["manifest"]),receipt_relative=paths["fit_receipt"].relative_to(post).as_posix() if "fit_receipt" in paths else None,
                receipt_sha256=sha(paths["fit_receipt"]) if "fit_receipt" in paths else None)
            if condition["kind"] == "nested":
                phase = "actual_fixed_full_basis_"+run_id
                nested_evidence.append(dict(run_id=run_id,evidence=runtime.nested(verifier,runner,plan,values,paths)))
            phase,route_attempted = "original_catalog_no_forward_"+run_id,True
            canonical,size = runtime.route(run_id,task,private,post)
            cost.cost_spec(canonical,size,binding,plan,runner.fingerprint)
            rp = cost.requested_paths(verifier,runner,plan,post,private,condition)
            reference,receipt = verifier.read(rp["replay_reference"]),verifier.read(rp["replay_receipt"])
            if receipt.get("cache_relative_path") != f"route_replays/{run_id}/V_subset.npz" or sha(rp["replay_archive"]) != receipt.get("archive_sha256"):
                raise ValueError("Original replay archive byte binding differs")
            cost.validate_replay(runner,plan,binding,canonical,reference,receipt)
            validate_cpu_replay_result(cost,condition,receipt,proofs[run_id])
            for key in ("replay_reference","replay_receipt","replay_archive"):
                path = rp[key]
                root_name,root = ("public",public) if path.is_relative_to(public) else ("cache",cache)
                bindings.append(dict(root=root_name,relative_path=path.relative_to(root).as_posix(),sha256=sha(path)))
            selected.append(dict(**binding,canonical_route=canonical))
        sharing = validate_artifact_sharing(strategy,selected)
        runner.new(output/"INPUT_BINDINGS.json",dict(files=bindings,selected_routes=selected,nested_full_basis_evidence=nested_evidence,
            artifact_sharing=sharing,actual_CPU_verifier_proofs_bound=True,other_strategies_are_not_a_gate=True))
        folder = private/"transition_cost_specs"/token
        folder.mkdir(parents=True,exist_ok=False)
        transition_plan = dict(dataset="beijing",K_grid=GRID,strategies=strategy_map(plan),selection_on_T=False,
            status="POST_T_V_ONLY_CPU_TRANSITION_PLAN_NOT_EXECUTED",postT_plan_sha256=PLAN_SHA,
            source39=plan["source39"],registered_outputs=12,prepared_strategy=strategy,
            original_make_plan_not_used=True,original_scientific_measurement_source_unchanged=True)
        plan_path = folder/"PLAN.json"
        runner.new(plan_path,transition_plan)
        plan_relative = plan_path.relative_to(private).as_posix()
        for batch in (1,8):
            label = strategy+"_B"+str(batch)
            output_relative = "costs/"+token+"/"+label
            jobs.append(dict(job_id=label,strategy=strategy,batch=batch,device="cpu",K_grid=GRID,CPU_threads=2,
                plan_cache_relative_path=plan_relative,plan_sha256=sha(plan_path),output_public_relative_path=output_relative,
                whole_protocol=PROTOCOL,warmups_per_directed_transition=10,timed_trials_per_directed_transition=20,
                actual_trial_count=0,required_final_receipt="BATCH_RECEIPT.json and strategy/RESULT.json, exact_minimum_protocol_complete=true",
                selected_run_ids=[row["condition"]["run_id"] for row in selected],fresh_process_single_strategy=True,
                other_nine_worker_scopes_are_not_a_gate=True,canonical_parity_still_required_inside_actual_worker=True,
                arguments=worker_arguments(plan_relative,output_relative,batch,strategy,plan["model_asset_relative_path"])))
        prepared = True
    except Exception as error:
        error_class = type(error).__name__
        failures.append(utility.save_private_failure(private,cache,"transition_cost_preparation",strategy,phase,error))
    finally:
        try:
            metadata(public,cache,asset,strategy,plan_sha)
            roots = dict(public=public,cache=cache)
            if sha(__file__) != source_sha or any(sha(runner.relative(roots[row["root"]],row["relative_path"])) != row["sha256"] for row in bindings):
                raise ValueError("Original source/control, selected inputs or actual CPU proof bytes changed")
            if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized(): raise ValueError("Unexpected CUDA initialization")
            preserved = True
        except Exception as error:
            post_error = type(error).__name__
            failures.append(utility.save_private_failure(private,cache,"transition_cost_preparation",strategy,"postflight",error))
        stop.set(); watcher.join(timeout=2)
        success = prepared and error_class is None and post_error is None and preserved
        result = dict(status="PREPARED_NOT_EXECUTED_TRANSITION_COST_WORKERS" if success else "FAILED_PRESERVED",strategy=strategy,
            plan_sha256=PLAN_SHA,preparation_helper_sha256=source_sha,reused_cost_helper_sha256=COST_HELPER_SHA,
            worker_count=len(jobs),jobs=jobs,whole_protocol=PROTOCOL,actual_training_forward_measurement_or_child_operations=0,
            actual_cost_workers_started=0,actual_trial_count=0,actual_foundation_models_constructed=0,actual_training_or_forecasting=0,
            Task_load_call_attempted=task_attempted,original_catalog_call_attempted=route_attempted,
            original_source39_registry_selection_T_and_captured_inputs_preserved=preserved,
            original_source39=plan["source39"],original_immutable_public=plan["immutable_public"],
            error_class=error_class,postflight_error_class=post_error,private_failures=failures,
            artifact_or_result_copies_created=False,registry_alias_created=False,other_strategies_are_not_a_gate=True,
            original_worker_unit="One strategy, all three K, one B; two prepared fresh workers here, ten across all five strategies",
            limitation="One missing/invalid K prevents only its three-K strategy; original worker has no two-K or individual-transition preparation mode",
            runtime_cost_scope="Original guarded K switch plus physical-input forecast; distinct from steady-state warmed prediction costs",
            canonical_fresh_parity_required_at_runtime=True,CPU_threads=2,max_RSS_bytes=plan["max_RSS_bytes"],deadline_unix=deadline,
            storage_limit_bytes=plan["storage_limit_bytes"],admission_reserve_bytes=256*1024**2,external_process_tree_guard_required=True,
            CPU_worker_has_no_internal_lease_deadline_or_RSS_enforcement=True,ROOT_must_confirm_all_training_inference_and_other_cost_work_closed=True,
            post_T_V_only_development=True,original_GPU_gap_completed=False,independent_generalization_claimed=False,new_T_target_or_scoring=False,
            preparation_scope="Original safeFIT Task and hash-bound selected adapters/full PCA basis; no Route/F construction, forward, fit, measurement or worker launch",
            actual_CPU_proof_scope="Explicit byte-pinned frozen verifier RESULT with actual forward true, matching original six-origin PASS receipt; no cached-only proof",
            archive_deserialization_scope="Original Task may physically deserialize/hash campaign T bytes; no T target or T evaluation API is requested",
            observed_process_pid=process.pid,observed_process_create_time=process.create_time(),finished_unix=time.time())
        runner.new(output/"TRANSITION_COST_EXECUTION_PLAN.json",result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ("public","cache","asset"): parser.add_argument("--"+root+"-root")
    parser.add_argument("--strategy",choices=STRATEGIES); parser.add_argument("--plan-sha")
    parser.add_argument("--cpu-replay-proofs"); parser.add_argument("--cpu-replay-proofs-sha")
    parser.add_argument("--deadline",type=float); parser.add_argument("--output"); parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    roots = (args.public_root,args.cache_root,args.asset_root)
    if args.execute:
        if not all(roots+(args.strategy,args.plan_sha,args.cpu_replay_proofs,args.cpu_replay_proofs_sha,args.deadline,args.output)):
            parser.error("Preparation requires roots, one strategy, PLAN SHA, actual CPU proof index/SHA, deadline and fresh output")
        public,cache,asset = (Path(value).resolve() for value in roots)
        result = execute(public,cache,asset,args.plan_sha,args.strategy,Path(args.output).absolute(),deadline=args.deadline,
            cpu_replay_proofs=Path(args.cpu_replay_proofs).absolute(),cpu_replay_proofs_sha=args.cpu_replay_proofs_sha)
        print(json.dumps(dict(status=result["status"],worker_count=result["worker_count"],actual_trial_count=0)))
        return 0 if result["status"] == "PREPARED_NOT_EXECUTED_TRANSITION_COST_WORKERS" else 1
    state = None
    if any(roots):
        if not all(roots) or not args.strategy or not args.plan_sha:
            parser.error("Metadata validation requires all original roots, one strategy and exact PLAN SHA")
        state = metadata(*(Path(value).resolve() for value in roots),args.strategy,args.plan_sha)[-1]
    print(json.dumps(dict(status="METADATA_ONLY_MULTIK_CPU_TRANSITION_PREPARATION",whole_protocol_worker_scopes=10,
        prepared_workers_per_strategy=2,requested_strategy_state=state,numpy_imported="numpy" in sys.modules,
        torch_imported="torch" in sys.modules,psutil_imported="psutil" in sys.modules,files_written=0,
        actual_scientific_or_child_operations=0,helper_sha256=sha(__file__))),flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
