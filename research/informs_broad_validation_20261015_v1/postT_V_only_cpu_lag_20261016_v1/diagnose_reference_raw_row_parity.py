"""Capture original Jena Raw CPU parity; never repair its cost gate.

The original four-variant function runs unchanged. A separate six-forward pass
records detailed differences for the same four row sizes and deployed128 twice.
Only aggregates are public. ROOT must run this in a closed, quiet CPU guard.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
NAME="postT_V_only_cpu_lag_20261016_v1"
RUN_ID="primary_jena_E1_RAW_BOLT_SMALL_k0_s92601"
PLAN_SHA="fedf00ebb1807528514c0685a6b226fd6a466149531adc7a5791ffa07d92af35"
REFERENCE_HELPER_SHA="62a89fde89402847e762f2a54fdcbd148158ddeb2a89679cc8e7555196828884"
REPLAY_PROOF_REL=NAME+"/diagnostics/root_reference_CPU_phase3_v1_00_replay/CPU_REPLAY_RESULT.json"
REPLAY_PROOF_SHA="7480bf99e2bf0a7bf41fe79b5f9398dee17c1d7242f20d60db52560cbad70f5f"
ATOL,RTOL=1e-5,1e-4
ORIGINAL_ROWS=("full",32,8,4)
DETAIL_ROWS=("full",32,8,4,128,128)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def safe(path):
    path=Path(path).absolute()
    if any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in [path]+list(path.parents)):
        raise ValueError("Nonsymlink and nonjunction paths required before resolution")
    return path.resolve()


def inside(root,name):
    root,name=safe(root),Path(name)
    if name.is_absolute() or ".." in name.parts:raise ValueError("Explicit root-relative evidence required")
    path=safe(root/name)
    if path==root or not path.is_relative_to(root):raise ValueError("Evidence leaves its declared root")
    return path


def read(path):
    path=safe(path)
    if path.stat().st_size>8*1024**2:raise ValueError("Small JSON metadata limit exceeded")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")


def load_reference():
    path=HERE/"prepare_reference_cpu_costs.py"
    if sha(path)!=REFERENCE_HELPER_SHA:raise ValueError("Frozen reference metadata helper changed")
    spec=importlib.util.spec_from_file_location("raw_diagnostic_reference_metadata",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_closed(value,batch,spec,public,cache,asset):
    if (value.get("schema")!="CPU_GROUP_SUPERVISOR_RESULT_v1" or value.get("helper_unchanged") is not True
            or value.get("remaining_owned_processes")!=[] or value.get("supervisor_failure_class") is not None
            or value.get("GPU_owner_or_token_created") is not False or value.get("CUDA_VISIBLE_DEVICES")!="-1"):
        raise ValueError("Original single-worker supervisor must be terminal and closed")
    processes=value.get("processes",[]);effective=value.get("effective_spec",{});workers=effective.get("processes",[])
    if (len(processes)!=1 or len(workers)!=1 or processes[0].get("returncode") is None
            or processes[0].get("observed_tree_join_confirmed") is not True or processes[0].get("remaining_owned_processes")!=[]
            or processes[0].get("process_tree_errors")!=[]):raise ValueError("Original observed owned tree closure is required")
    command=workers[0].get("command",[])
    def argument(flag):
        if command.count(flag)!=1:raise ValueError("Exactly one original worker argument required")
        i=command.index(flag)
        if i+1>=len(command):raise ValueError("Missing original worker argument")
        return command[i+1]
    scripts=[token for token in command if isinstance(token,str) and token.replace("\\","/").endswith("/measure_costs.py")]
    if (command.count("--worker")!=1 or "--balanced" in command or len(scripts)!=1 or safe(scripts[0])!=public/"measure_costs.py"
            or argument("--device")!="cpu" or argument("--batch")!=str(batch)
            or any(argument(flag)!=value for flag,value in (("--warmups","10"),("--timed-trials","20"),("--stage-trials","20")))
            or safe(argument("--route-spec"))!=safe(spec) or safe(argument("--public-root"))!=public
            or safe(argument("--cache-root"))!=cache or not safe(argument("--model")).is_relative_to(asset)
            or not safe(argument("--output")).is_relative_to(public/NAME/"costs")
            or effective.get("max_each_RSS")!=3*1024**3 or effective.get("max_group_RSS")!=3*1024**3):
        raise ValueError("Original Raw cost worker, batch, source roots or CPU protocol differs")
    return dict(status=value["status"],returncode=processes[0]["returncode"],observed_tree_join_confirmed=True,
        remaining_owned_processes=[],owned_bindings=[{key:row[key] for key in ("pid","create_time")} for row in processes[0].get("owned_processes",[])],
        unobserved_reparented_descendants_absence_proven=value.get("unobserved_reparented_descendants_absence_proven"))


def metadata(public,cache,asset,plan_sha,batch,spec_path,spec_sha,closed_path,closed_sha,log_path,log_sha):
    if plan_sha!=PLAN_SHA or batch not in (1,8):raise ValueError("Exact original Jena PLAN and one B1/B8 scope required")
    helper=load_reference();state=helper.metadata(public,cache,asset,RUN_ID,plan_sha)
    if not state["ready"]:raise ValueError("Original Raw reference is not ready")
    paths={"cost_spec":safe(spec_path),"closed_supervisor":safe(closed_path),"original_private_log":safe(log_path)}
    pins=dict(zip(paths,(spec_sha,closed_sha,log_sha)))
    for key,path in paths.items():
        if (not path.is_relative_to(cache/NAME) or not re.fullmatch("[0-9a-f]{64}",pins[key] or "")
                or sha(path)!=pins[key]):raise ValueError("Explicit original cost evidence bytes or cache scope differ")
    if paths["original_private_log"].stat().st_size>8*1024**2:raise ValueError("Bounded original worker log required")
    spec=read(paths["cost_spec"])
    canonical={key:value for key,value in spec.items() if key not in {"accuracy_run_id","row_microbatch"}}
    if helper.cost_spec(canonical,"small",state["binding"],state["plan"])!=spec or spec.get("cost_origins") is not None:
        raise ValueError("Original Raw canonical route and default cost histories required")
    closed=validate_closed(read(paths["closed_supervisor"]),batch,paths["cost_spec"],public,cache,asset)
    closed_value=read(paths["closed_supervisor"])
    expected_log=paths["closed_supervisor"].parent/(closed_value["processes"][0]["id"]+".log")
    if paths["original_private_log"]!=expected_log or paths["closed_supervisor"].name!="RESULT.json":
        raise ValueError("The original log must belong to this closed single worker")
    actual_model=closed_value["effective_spec"]["processes"][0]["command"]
    if safe(actual_model[actual_model.index("--model")+1])!=inside(asset,state["plan"]["model_asset_relative_path"]):
        raise ValueError("Original worker's exact model snapshot differs")
    proof_path=inside(public,REPLAY_PROOF_REL)
    if sha(proof_path)!=REPLAY_PROOF_SHA:raise ValueError("Original fresh CPU six-origin proof changed")
    proof=read(proof_path);actual=proof.get("scientific_result") or {}
    if (proof.get("status")!="PASS_FRESH_CPU_SIX_ORIGIN_PARITY" or proof.get("run_id")!=RUN_ID or proof.get("device")!="cpu"
            or proof.get("actual_checkpoint_forward_in_this_command") is not True or proof.get("CPU_threads")!=2
            or proof.get("plan_sha256")!=plan_sha or proof.get("original_source_registry_scores_cost_T_and_selected_inputs_preserved") is not True
            or actual.get("status")!="PASS" or actual.get("executed_now") is not True or len(actual.get("origins",[]))!=6
            or actual.get("actual_F_streams")!=126 or actual.get("F_physical_streams")!=126
            or actual.get("atol")!=ATOL or actual.get("rtol")!=RTOL or actual.get("T_final_accessed") is not False
            or actual.get("standard_manifest_sha256")!=state["binding"]["metadata_pins"][state["binding"]["manifest_relative"]]):
        raise ValueError("Bound CPU six-origin checkpoint parity evidence differs")
    evidence={key:dict(cache_relative_path=path.relative_to(cache).as_posix(),sha256=pins[key]) for key,path in paths.items()}
    evidence["prior_CPU_checkpoint_replay"]=dict(public_relative_path=REPLAY_PROOF_REL,sha256=REPLAY_PROOF_SHA)
    return dict(plan=state["plan"],binding=state["binding"],spec=spec,evidence=evidence,closed=closed,
        prior_checkpoint_scope="Six selected V origins, CPU batch4 row128; not a row-size parity certification",batch=batch,
        original_log_contains_gate_message="Actual Raw row variants disagree or all failed" in paths["original_private_log"].read_text(encoding="utf-8",errors="replace"))


def validate_original(value):
    if (not isinstance(value,dict) or value.get("status")!="CHECKED" or value.get("atol")!=ATOL or value.get("rtol")!=RTOL
            or [row.get("row_microbatch") for row in value.get("variants",[])]!=list(ORIGINAL_ROWS)
            or any(row.get("status") not in {"PASS","FAIL","BLOCKED_CUDA_OOM"} for row in value["variants"])
            or not isinstance(value.get("all_passed"),bool) or not isinstance(value.get("successful_variants_parity"),bool)):
        raise ValueError("Original default four candidates and fixed tolerance required")
    successes=[row for row in value["variants"] if row["status"]!="BLOCKED_CUDA_OOM"]
    if value.get("reference_row_microbatch")!=(successes[0]["row_microbatch"] if successes else None):
        raise ValueError("Original first successful variant must remain its reference")
    if (value["all_passed"]!=all(row["status"]=="PASS" for row in value["variants"])
            or value["successful_variants_parity"]!=bool(successes and not any(row["status"]=="FAIL" for row in value["variants"]))):
        raise ValueError("Original parity gate values differ from its preserved variants")


def capture_original(function,route,physical,observed,path):
    value=function(route,physical,observed,lease=None)
    new(path,value)
    validate_original(value)
    return value


def counter_expectation(batch,row):
    if batch not in (1,8) or row not in ("full",32,8,4,128):raise ValueError("Declared B1/B8 and original/deployed row widths only")
    return dict(F_logical_streams=batch*21,F_physical_streams=batch*21,F_calls=1 if row=="full" else math.ceil(batch*21/row))


def comparison_summary(reference,predicted,std):
    if not isinstance(std,list) or not std or any(isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) or s<=0 for s in std):raise ValueError("Finite positive A60 channel scales required")
    c=len(std);shape=[len(reference),len(reference[0]) if reference else 0,c]
    if not shape[0] or not shape[1] or len(predicted)!=shape[0]:raise ValueError("Nonempty same-shape forecast lists required")
    per=[dict(channel_index=i,cells=0,failed_cells=0,max_abs=0.,max_A_normalized_abs=0.,near_zero_reference_cells=0,
        near_zero_failed_cells=0,near_zero_rtol_operand_cells=0,near_zero_rtol_operand_failed_cells=0,
        failed_sign_crossings=0,exact_zero_reference_failed_cells=0,nonfinite_cells=0) for i in range(c)]
    finite=True;max_abs=max_normalized=0.;failed=nearzero=nearfailed=rtolzero=rtolfailed=signfailed=zerofailed=0
    for arow,brow in zip(reference,predicted):
        if len(arow)!=shape[1] or len(brow)!=shape[1]:raise ValueError("Forecast horizon dimensions differ")
        for av,bv in zip(arow,brow):
            if len(av)!=c or len(bv)!=c:raise ValueError("Forecast channel dimensions differ")
            for i,(a,b) in enumerate(zip(av,bv)):
                if isinstance(a,bool) or isinstance(b,bool) or not isinstance(a,(int,float)) or not isinstance(b,(int,float)):
                    raise ValueError("Numeric forecast aggregates required")
                ok=math.isfinite(a) and math.isfinite(b);finite=finite and ok
                cell=per[i];cell["cells"]+=1
                if not ok:
                    cell["failed_cells"]+=1;cell["nonfinite_cells"]+=1;failed+=1
                    continue
                diff=abs(a-b);norm=diff/std[i];bad=diff>ATOL+RTOL*abs(b);nz=abs(a)<=ATOL/RTOL;rz=abs(b)<=ATOL/RTOL
                cross=bad and a*b<0;zero=bad and a==0
                cell["failed_cells"]+=int(bad);cell["near_zero_reference_cells"]+=int(nz);cell["near_zero_failed_cells"]+=int(nz and bad)
                cell["near_zero_rtol_operand_cells"]+=int(rz);cell["near_zero_rtol_operand_failed_cells"]+=int(rz and bad)
                cell["failed_sign_crossings"]+=int(cross);cell["exact_zero_reference_failed_cells"]+=int(zero)
                cell["max_abs"]=max(cell["max_abs"],diff);cell["max_A_normalized_abs"]=max(cell["max_A_normalized_abs"],norm)
                max_abs=max(max_abs,diff);max_normalized=max(max_normalized,norm)
                failed+=int(bad);nearzero+=int(nz);nearfailed+=int(nz and bad);rtolzero+=int(rz);rtolfailed+=int(rz and bad);signfailed+=int(cross);zerofailed+=int(zero)
    for cell in per:
        if cell["nonfinite_cells"]:cell["max_abs"]=cell["max_A_normalized_abs"]=None
    return dict(shape=shape,finite=finite,allclose=finite and failed==0,failed_cells=failed,total_cells=shape[0]*shape[1]*c,
        max_abs=max_abs if finite else None,max_A_normalized_abs=max_normalized if finite else None,per_channel=per,
        near_zero_reference_abs_threshold=ATOL/RTOL,near_zero_reference_cells=nearzero,near_zero_failed_cells=nearfailed,
        near_zero_rtol_operand_cells=rtolzero,near_zero_rtol_operand_failed_cells=rtolfailed,
        failed_sign_crossings=signfailed,exact_zero_reference_failed_cells=zerofailed,atol=ATOL,rtol=RTOL,
        original_gate_units="physical output; np.allclose(reference,predicted): abs(pred-reference) <= atol + rtol*abs(predicted)",
        A_normalized_difference_scope="abs physical difference divided by original A60 std; diagnostic only, not a changed parity gate")


class Runtime:
    def __init__(self,public):
        sys.path.insert(0,str(public))
        import numpy as np
        import torch
        import measure_costs
        import data_contract
        import fit_base
        for module in (measure_costs,data_contract,fit_base):
            if Path(module.__file__).resolve()!=public/(module.__name__+".py"):raise ValueError("Original pinned scientific API source root required")
        for name in ("deployment","gpu_runtime","training","route_catalog","networks","evaluation","combination"):
            module=sys.modules.get(name)
            if module is not None and Path(module.__file__).resolve()!=public/(name+".py"):
                raise ValueError("Transitive scientific API must retain its original source root")
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():raise RuntimeError("CPU-only diagnostic required")
        self.np,self.torch,self.api,self.data,self.fit=np,torch,measure_costs,data_contract,fit_base

    def histories(self,runner,plan,cache,batch):
        np=self.np;task=self.data.load_task("jena",cache_dir=cache)
        binding=runner.validate_task(task,plan,self.data,self.fit)
        origins=self.api.origins(task["split_ranges"]["V_select"],task["L"],task["H"],task["H"])
        if not len(origins):raise ValueError("Original V cost histories absent")
        chosen=np.resize(origins[:batch],batch).tolist()
        physical=np.stack([task["physical"][o-task["L"]:o] for o in chosen])
        observed=np.stack([task["observed"][o-task["L"]:o] for o in chosen])
        return task,physical,observed,list(map(int,chosen)),binding

    def route(self,spec,task,cache,model):
        return self.api.Route(spec,task,cache,model,"cpu")


def detail_pass(route,physical,observed,std,original,check,save_failure):
    rows=[];reference=deployed=None;previous=route.spec.get("row_microbatch","full")
    try:
        for index,row in enumerate(DETAIL_ROWS):
            check();route.spec["row_microbatch"]=row;predicted=None;receipt=None
            record=dict(row_microbatch=row,scope="same_four_candidates_independent_detail_pass" if index<4 else "extra_deployed128" if index==4 else "extra_deployed128_repeat",attempted=True)
            try:
                predicted=route.predict_physical(physical,observed);receipt=dict(route.last_receipt)
                if reference is None:reference=predicted.copy()
                summary=comparison_summary(reference.tolist(),predicted.tolist(),std)
                counters={key:receipt[key] for key in ("F_logical_streams","F_physical_streams","F_calls")}
                record.update(status="PASS" if summary["allclose"] else "FAIL",prediction_sha256=hashlib.sha256(predicted.tobytes()).hexdigest(),comparison_to_first_detail_success=summary,
                    actual_F_counters=counters,expected_F_counters=counter_expectation(len(physical),row),F_counters_match=counters==counter_expectation(len(physical),row))
                if not record["F_counters_match"]:raise ValueError("Actual detailed F counters differ from physical row partition")
                if index<4 and original is not None:
                    old=original["variants"][index]
                    record["matches_original_call_prediction_sha256"]=record["prediction_sha256"]==old.get("prediction_sha256")
                    record["original_call_variant_status"]=old["status"]
                if index==4:deployed=predicted.copy()
                if index==5 and deployed is not None:record["comparison_to_deployed128_first"]=comparison_summary(deployed.tolist(),predicted.tolist(),std)
            except Exception as error:
                record.update(status="ERROR_PRESERVED",error_class=type(error).__name__,private_failure=save_failure(error,"detail_"+str(index)))
            finally:route.deactivate()
            rows.append(record)
    finally:route.spec["row_microbatch"]=previous
    return rows


def execute(public,cache,asset,plan_sha,batch,spec_path,spec_sha,closed_path,closed_sha,log_path,log_sha,deadline,output):
    started=time.perf_counter();source_sha=sha(__file__)
    state=metadata(public,cache,asset,plan_sha,batch,spec_path,spec_sha,closed_path,closed_sha,log_path,log_sha)
    helper=load_reference();runner=helper.load_runner();plan=state["plan"];post,private=public/NAME,cache/NAME
    if deadline is None or not time.time()<deadline<=min(plan["deadline_unix"],time.time()+180):raise ValueError("Explicit ROOT-guarded maximum180s within campaign deadline required")
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="-1" or any(k.startswith("INFORMS_GPU_") for k in os.environ)
            or any(os.environ.get(k)!="2" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(k)!="1" for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"))
            or safe(os.environ.get("INFORMS_ASSET_ROOT",""))!=asset or safe(os.environ.get("INFORMS_CACHE_ROOT",""))!=private):
        raise ValueError("CPU2 offline separate-cache environment without GPU authorization required")
    output=safe(output)
    if output==post or not output.is_relative_to(post/"diagnostics") or output.exists():raise FileExistsError("Fresh owned public diagnostics required")
    used=sum(p.stat().st_size for root in (public,cache) for p in root.rglob("*") if p.is_file());reserve=256*1024**2
    if used+reserve>plan["storage_limit_bytes"] or min(shutil.disk_usage(root).free for root in (public,cache))<reserve:
        raise ValueError("Original combined30GiB and free256MiB reserve required")
    output.mkdir(parents=True,exist_ok=False)
    new(output/"STARTED.json",dict(run_id=RUN_ID,batch=batch,plan_sha256=plan_sha,helper_sha256=source_sha,actual_pid=os.getpid(),maximum_wall_seconds=180,
        original_parity_calls_planned=1,original_variant_forwards_planned=4,separate_detail_forwards_planned=6,new_T_or_training=False,started_unix=time.time()))
    import psutil
    process=psutil.Process();stop=threading.Event()
    def guard():
        while not stop.wait(.25):
            if process.memory_info().rss>3*1024**3 or time.time()>=deadline or time.perf_counter()-started>=180:
                new(output/"RESOURCE_STOP.json",dict(status="RESOURCE_STOP",rss_bytes=process.memory_info().rss));os._exit(98)
    watch=threading.Thread(target=guard,daemon=True);watch.start()
    def check():
        if process.memory_info().rss>3*1024**3 or time.time()>=deadline or time.perf_counter()-started>=180:raise TimeoutError("CPU diagnostic resource bound reached")
    def failure(error,phase):return runner.save_private_failure(private,cache,"raw_row_diagnostic",RUN_ID,"B"+str(batch)+"_"+phase,error)
    route=runtime=original=details=None;error=post_error=close_error=None;failures=[];preserved=False;original_attempted=False;original_returned=False
    before=runner.rss_record(process);original_counters=None;frozen=False
    try:
        check();runtime=Runtime(public)
        task,physical,observed,origins,task_binding=runtime.histories(runner,plan,cache,batch)
        route=runtime.route(state["spec"],task,cache,asset/plan["model_asset_relative_path"])
        binding=runtime.api.accuracy_binding(state["spec"],route,task,cache,public)
        if (route.family!="RAW" or binding["status"]!="BOUND_EXACT_ROUTE" or binding["artifact_bindings"]!={}
                or binding["foundation_state_sha256"]!=plan["foundation_tensor_sha256"]
                or binding["scaler_sha"]!=plan["task_contract"]["refit_scaler_sha256"]):raise ValueError("Actual original Raw model/data/scaler identity differs")
        input_sha=hashlib.sha256(physical.tobytes()).hexdigest();mask_sha=hashlib.sha256(observed.tobytes()).hexdigest();scale_sha=runtime.api.scaler_fingerprint(route.mean,route.std)
    except Exception as cause:
        error=type(cause).__name__;failures.append(failure(cause,"inputs"))
    try:
        if error is None:
            new(output/("INPUT_BINDING_B"+str(batch)+".json"),dict(plan_sha256=plan_sha,helper_sha256=source_sha,evidence=state["evidence"],original_cost_closure=state["closed"],
                original_private_log_contains_gate_message=state["original_log_contains_gate_message"],task_binding=task_binding,accuracy_binding=binding,
                batch=batch,L=512,H=48,C=21,effective_F_width=21,origins=origins,physical_input_sha256=input_sha,observation_mask_sha256=mask_sha,
                original_worker_default_V_history_selection_unchanged=True,forecast_target_arrays_constructed_or_scored=False,raw_arrays_published=False,
                prior_CPU_checkpoint_scope=state["prior_checkpoint_scope"],safeFIT_asset_may_physically_deserialize_historical_T_bytes=True))
            check();counts_before=route.pool.counters();original_attempted=True
            try:
                original=capture_original(runtime.api.raw_row_parity,route,physical,observed,output/("RAW_PARITY_B"+str(batch)+".json"));original_returned=True
            except Exception as cause:
                failures.append(failure(cause,"original_raw_row_parity"))
                new(output/"ORIGINAL_CALL_FAILURE.json",dict(error_class=type(cause).__name__,structured_return_written=(output/("RAW_PARITY_B"+str(batch)+".json")).is_file()))
            counts_after=route.pool.counters();original_counters={key:counts_after[key]-counts_before[key] for key in counts_after}
            details=detail_pass(route,physical,observed,route.std.tolist(),original,check,failure)
            new(output/("ROW_DETAILS_B"+str(batch)+".json"),dict(rows=details,original_candidates=list(ORIGINAL_ROWS),additional_deployed_row=128,repeat_scope="Second independent row128 forward, same process/model/history",not_cost_timing=True))
            if hashlib.sha256(physical.tobytes()).hexdigest()!=input_sha or hashlib.sha256(observed.tobytes()).hexdigest()!=mask_sha or runtime.api.scaler_fingerprint(route.mean,route.std)!=scale_sha:
                raise ValueError("Inference histories/mask/A60 scaler changed")
            route.pool.foundation.assert_frozen_unchanged();frozen=True
    except Exception as cause:
        error=type(cause).__name__;failures.append(failure(cause,"diagnostic"))
    finally:
        if route is not None:
            try:route.close()
            except Exception as cause:close_error=type(cause).__name__;failures.append(failure(cause,"close"))
        try:
            after=metadata(public,cache,asset,plan_sha,batch,spec_path,spec_sha,closed_path,closed_sha,log_path,log_sha)
            if after!=state or sha(__file__)!=source_sha:raise ValueError("Original inputs/helper changed during diagnostic")
            preserved=True
        except Exception as cause:post_error=type(cause).__name__;failures.append(failure(cause,"postflight"))
        stop.set();watch.join(timeout=1)
    success=error is None and post_error is None and close_error is None and preserved and original_returned and details is not None and all(row["status"]!="ERROR_PRESERVED" for row in details)
    new(output/"RESULT.json",dict(status="DIAGNOSTIC_COMPLETE_ORIGINAL_GATE_PRESERVED" if success else "DIAGNOSTIC_PARTIAL_OR_FAILED_PRESERVED",run_id=RUN_ID,batch=batch,
        original_raw_row_parity_call_attempted=original_attempted,original_structured_return_available=original_returned,
        original_structured_return_file_written=(output/("RAW_PARITY_B"+str(batch)+".json")).is_file(),
        original_successful_variants_parity=original["successful_variants_parity"] if original is not None else None,
        original_four_variant_aggregate_counter_delta=original_counters,separate_detail_forwards_returned=sum(row["status"]!="ERROR_PRESERVED" for row in details or []),
        maximum_forward_calls_planned=10,original_function_or_tolerance_changed=False,atol=ATOL,rtol=RTOL,source39_and_immutable_inputs_preserved=preserved,
        foundation_frozen_hash_checked_before_close=frozen,route_close_passed=route is not None and close_error is None,error_class=error,close_error_class=close_error,postflight_error_class=post_error,
        private_failures=failures,CPU_threads=2,max_RSS_bytes=3*1024**3,actual_pid=process.pid,actual_create_time=process.create_time(),
        resource_before=before,resource_after=runner.rss_record(process),cuda_initialized=runtime.torch.cuda.is_initialized() if runtime is not None else None,
        plan_sha256=plan_sha,helper_sha256=source_sha,elapsed_seconds=time.perf_counter()-started,finished_unix=time.time(),
        cost_protocol_completed=False,old_failed_cost_changed=False,new_training=False,new_T_target_or_scoring=False,raw_predictions_or_targets_published=False,
        ROOT_closed_quiet_external_tree_guard_required=True,forced_resource_exit_may_leave_STARTED_without_RESULT=True,
        interpretation="A returned numerical difference diagnoses this rerun only; original historical causes and low-level kernel cause remain unproven"))
    return 0 if success else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for root in ("public","cache","asset"):parser.add_argument("--"+root+"-root")
    parser.add_argument("--plan-sha");parser.add_argument("--batch",type=int,choices=(1,8));parser.add_argument("--cost-spec");parser.add_argument("--cost-spec-sha")
    parser.add_argument("--closed-result");parser.add_argument("--closed-result-sha");parser.add_argument("--original-log");parser.add_argument("--original-log-sha")
    parser.add_argument("--deadline",type=float);parser.add_argument("--output");parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv);roots=(args.public_root,args.cache_root,args.asset_root)
    bindings=(args.plan_sha,args.batch,args.cost_spec,args.cost_spec_sha,args.closed_result,args.closed_result_sha,args.original_log,args.original_log_sha)
    if args.execute or any(roots) or any(bindings):
        if not all(roots+bindings):parser.error("All original roots, PLAN, B1/B8, exact cost spec/closed result/log SHA required")
        public,cache,asset=map(safe,roots);paths=tuple(inside(cache,name) for name in (args.cost_spec,args.closed_result,args.original_log))
        call=(public,cache,asset,args.plan_sha,args.batch,paths[0],args.cost_spec_sha,paths[1],args.closed_result_sha,paths[2],args.original_log_sha)
        if args.execute:
            if not args.deadline or not args.output:parser.error("Explicit bounded deadline and fresh output required")
            return execute(*call,args.deadline,Path(args.output).absolute())
        metadata(*call)
    print(json.dumps(dict(status="METADATA_ONLY_RAW_DIAGNOSTIC_NOT_EXECUTED",files_written=0,model_array_or_child_operations=0,
        numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,psutil_imported="psutil" in sys.modules,
        helper_sha256=sha(__file__),original_candidates=list(ORIGINAL_ROWS),detail_candidates=list(DETAIL_ROWS),original_cost_gate_changed=False)))
    return 0


if __name__=="__main__":raise SystemExit(main())
