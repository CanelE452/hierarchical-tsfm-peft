"""Jena B8 원 raw_row_parity 반환 보존. 기본 metadata만, 실제 실행은 기존 ROOT owner가 명시 호출."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PLAN_SHA="e7882353006394304161d17d73be4f8429bda1f114208adc2e939be523469dcf"


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def metadata(public,cache,asset):
    if sha(HERE/"PLAN.json")!=PLAN_SHA:raise ValueError("원 진단 계획 봉인 변경")
    plan=read(HERE/"PLAN.json")
    if {name:sha(public/name) for name in plan["source39"]}!=plan["source39"]:raise ValueError("원 source39 변경")
    if sha(cache/plan["spec_cache_relative_path"])!=plan["spec_sha256"]:raise ValueError("원 B8 spec 변경")
    for name,digest in plan["original_failure_execution_receipts"].items():
        if sha(public/name)!=digest:raise ValueError("원 실패 영수증 변경")
    for name,digest in plan["original_failure_private_log_sha256"].items():
        if sha(cache/name)!=digest:raise ValueError("원 실패 log 변경")
    if sha(public/plan["B1_success_public_relative_path"])!=plan["B1_success_sha256"]:raise ValueError("B1 원 근거 변경")
    if sha(asset/plan["external_instance_norm_source_relative_path"])!=plan["external_instance_norm_source_sha256"]:raise ValueError("원 instance norm source 변경")
    return plan


def capture(function,route,physical,observed,lease,path):
    result=function(route,physical,observed,lease=lease)
    new(path,result)
    return result


def execute_child(public,cache,asset,output):
    plan=metadata(public,cache,asset)
    if not output.is_relative_to(public/"diagnostics"):raise ValueError("새 공개 diagnostics 출력 필요")
    output.mkdir(parents=True,exist_ok=False)
    new(output/"RUN_CLAIM.json",dict(plan_sha256=PLAN_SHA,helper_sha256=sha(__file__),original_parity_calls_planned=1,started_unix=time.time()))
    sys.path.insert(0,str(public))
    import measure_costs as original
    import numpy as np
    import torch
    lease=original.worker_lease_check(cache,"cuda")
    if lease is None:raise ValueError("외부 인증 GPU child만 가능")
    torch.set_num_threads(2)
    spec=read(cache/plan["spec_cache_relative_path"])
    task=original.load_task(spec["dataset"],cache_dir=cache)
    candidates=original.origins(task["split_ranges"]["V_select"],task["L"],task["H"],task["H"])
    if not len(candidates) or spec.get("cost_origins") is not None:raise ValueError("원 worker의 기본 V cohort 필요")
    oo=np.resize(candidates[:8],8).tolist()
    if task["L"]!=512 or task["H"]!=48 or len(task["columns"])!=21 or len(oo)!=8:raise ValueError("원 B8 형태 변경")
    physical=np.stack([task["physical"][o-task["L"]:o] for o in oo])
    observed=np.stack([task["observed"][o-task["L"]:o] for o in oo])
    original_B1=read(public/plan["B1_success_public_relative_path"])
    route=None;parity=None;failure=None;close_failure=None;post_failure=None;binding=None;calls=0
    registry_sha=sha(public/"EXPERIMENT_REGISTRY.csv")
    try:
        lease.check()
        route=original.Route(spec,task,cache,asset/plan["model_asset_relative_path"],"cuda")
        binding=original.accuracy_binding(spec,route,task,cache,public)
        if (route.family!="RAW" or binding["status"]!="BOUND_EXACT_ROUTE" or binding["artifact_bindings"]!={}
                or binding["foundation_state_sha256"]!=plan["foundation_tensor_sha256"]
                or binding["data_sha256"]!=original_B1["data_sha256"] or binding["scaler_sha"]!=original_B1["scaler_sha"]):
            raise ValueError("동일 Raw 모델/data/scaler/artifact 계약 변경")
        new(output/"INPUT_BINDING.json",dict(plan_sha256=PLAN_SHA,helper_sha256=sha(__file__),accuracy_binding=binding,
            origins=list(map(int,oo)),batch=8,L=512,H=48,C=21,physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
            observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),target_values_read=False,
            original_worker_cohort_construction_unchanged=True,raw_arrays_published=False))
        lease.check()
        calls=1
        parity=capture(original.raw_row_parity,route,physical,observed,lease,output/"ORIGINAL_PARITY_RETURN.json")
    except Exception as error:
        failure=type(error).__name__
    finally:
        if route is not None:
            try:route.close()
            except Exception as error:close_failure=type(error).__name__
    try:
        lease.check();metadata(public,cache,asset)
        if sha(public/"EXPERIMENT_REGISTRY.csv")!=registry_sha:raise ValueError("registry bytes 변경")
    except Exception as error:post_failure=type(error).__name__
    preserved=(output/"ORIGINAL_PARITY_RETURN.json").is_file()
    complete=preserved and failure is None and close_failure is None and post_failure is None
    new(output/"RESULT.json",dict(status="DIAGNOSTIC_COMPLETE_ORIGINAL_PARITY_PRESERVED" if complete else "DIAGNOSTIC_PARTIAL_OR_FAILED",
        original_parity_return_preserved=preserved,original_parity_return_sha256=sha(output/"ORIGINAL_PARITY_RETURN.json") if preserved else None,
        original_gate_passed=parity.get("successful_variants_parity") if parity else None,
        failure_class=failure,close_failure_class=close_failure,postflight_failure_class=post_failure,
        source39_original_failures_spec_B1_and_registry_preserved=complete,backbone_exact_restore_at_close=close_failure is None and route is not None,
        original_raw_row_parity_calls_actual=calls,parity_defaults_unchanged=True,
        plan_sha256=PLAN_SHA,helper_sha256=sha(__file__),warmups=0,timed_trials=0,stage_trials=0,
        cost_protocol_completed=False,old_failure_receipts_replaced=False,raw_forecast_or_target_publication=False,
        training=0,new_T_or_protected_TEST_scoring=0,finished_unix=time.time()))
    return 0 if complete else 1


def run_with_owner(lease,asset_root,cache_root,public_root):
    if (lease is None or getattr(lease,"purpose",None)!="audit" or not getattr(lease,"_entered",False)
            or getattr(getattr(lease,"lock",None),"file",None) is None or getattr(lease,"pid",None)!=os.getpid()):
        raise ValueError("같은 PID의 기존 ROOT audit owner 필요; 새 owner 생성 금지")
    lease.check()
    public,cache,asset=Path(public_root).resolve(),Path(cache_root).resolve(),Path(asset_root).resolve()
    plan=metadata(public,cache,asset)
    if lease.total_gpu_seconds>plan["cost_cumulative_ceiling_seconds"] or lease.remaining<30:
        raise ValueError("기존39600 누적 상한/최종3600 보존/최소30초 경계 위반")
    stamp=str(time.time_ns());output=public/"diagnostics"/("jena_b8_raw_parity_actual_"+stamp)
    private=cache/"final_execution"/("jena_b8_raw_parity_actual_"+stamp);private.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,**lease.child_environment(),INFORMS_ASSET_ROOT=str(asset),INFORMS_CACHE_ROOT=str(cache),
        CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    command=[sys.executable,"-X","utf8","-u","-B",str(Path(__file__).resolve()),"--public-root",str(public),"--cache-root",str(cache),"--asset-root",str(asset),"--output",str(output),"--execute"]
    with (private/"CHILD.log").open("xb") as stream:
        code=subprocess.Popen(command,cwd=public,env=env,stdout=stream,stderr=subprocess.STDOUT).wait()
    new(private/"CHILD_EXECUTION.json",dict(actual_exitcode=code,public_output_relative_path=output.relative_to(public).as_posix(),
        private_log_sha256=sha(private/"CHILD.log"),new_or_nested_GPU_owner_created=False))
    return dict(actual_exitcode=code,public_output_relative_path=output.relative_to(public).as_posix(),
        private_execution_cache_relative_path=private.relative_to(cache).as_posix(),original_cost_failure_replaced=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root",default=os.environ.get("INFORMS_PUBLIC_ROOT"))
    parser.add_argument("--cache-root",default=os.environ.get("INFORMS_CACHE_ROOT"))
    parser.add_argument("--asset-root",default=os.environ.get("INFORMS_ASSET_ROOT"))
    parser.add_argument("--output");parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not all((args.public_root,args.cache_root,args.asset_root)):parser.error("public/cache/asset root 필요")
    public,cache,asset=Path(args.public_root).resolve(),Path(args.cache_root).resolve(),Path(args.asset_root).resolve()
    if args.execute:
        if not args.output:parser.error("실제 실행에는 새 --output 필요")
        raise SystemExit(execute_child(public,cache,asset,Path(args.output).resolve()))
    metadata(public,cache,asset)
    print(json.dumps(dict(status="PREPARATION_ONLY",plan_sha256=PLAN_SHA,helper_sha256=sha(__file__),numpy_imported="numpy" in sys.modules,
        torch_imported="torch" in sys.modules,actual_GPU_parity_calls=0,model_or_array_reads=0)),flush=True)


if __name__=="__main__":main()
