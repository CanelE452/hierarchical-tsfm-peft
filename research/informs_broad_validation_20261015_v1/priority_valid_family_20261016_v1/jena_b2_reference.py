"""ROOT-only validation/cost entry for the exact existing Jena B2 seed92601.

Default0 I/O/imports. No fit/probe, new weights or whole-Task loader.
The original CPU fullV remains unchanged; only new proof/cost receipts appear.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

NAME="priority_valid_family_20261016_v1"
OLD="postT_V_only_cpu_lag_20261016_v1"
RUN="postT_cpu_jena_E3_B2_k6_s92601_v1"
ARTIFACT="020d7d980e685c38e9b94472676a5d6b7f8b188ab573a16bd367aa20712a5963"
STATE="ce1431156667c35312d7a792779183eb796895199ef65812455badeb21e94dc9"
MANIFEST="6a885f3c8f298009ae93c731409baff116263a41c9c17ecaff713cf5e9499148"
ARCHIVE="578267d5a457165e5a179275802621354bb8fc2190cde1add18769a67365856d"
DATA="3928079b90efb066e02a8a3bef76dfefd92452a2ad0b26c29caf14393a161a58"
SCALER="5f0c7d6aeddcfe9577e381375d36cadc47d2572469118da73e6ad837031fa2b9"
FOUNDATION="c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
FIT_SOURCE="c58ed5ed690a812dd326f9c77a9c941268faeddafd1aea78c55e0036142df5e5"
PREDICTION_MODEL="04486ccee4aab1485744d7fa9b72b622e2f4a38cd5af38f0887dee53ab68f325"
PREDICTION_SOURCE="00f4e618a2842141b95f502ef0bf9c852e15e63c5d8ab529e9d15537c3e96f65"
PREDICTION_CONFIG="6cd7511b1d4bf0151b6ee9c7d257e5cc53bd5de3407ebe35de4477024a31b245"
DEADLINE=1792222550
PRED=OLD+"/predictions/"+RUN+"/V_select"
PRIVATE_V=OLD+"/predictions/"+RUN+"/V_select.npz"
ARTIFACT_REL=OLD+"/runs/"+RUN+"/selected_bundle.pt"
FIT_FOLDER=OLD+"/diagnostics/fit_1792112761493733800_B2"
DEPENDENCIES={
 NAME+"/family_minimum_runner.py":"12462b8f1085e346750a24cd37fd52191f8780799e2f9e5903b4a57bcbc570c1",
 "priority_b2_validation_20261016_v1/priority_runner.py":"f3e4279ddd250a51b09ce88938f8f202dfd02e3b24d1adff3bb36c567dc7ef07",
 "priority_b2_validation_20261016_v1/safe_jena_prefix.py":"6b496dcbbb6cef48acb9010c630ed81359a2d60bff89aaec9e9e77c4b6d0b541",
 "priority_core_followup_20261016_v1/jena_seed_runner.py":"ad4b76acf51fb416dea4f69c36940507db1f92e223c620388ded45381d6e28e7",
 "priority_valid_minima_20261016_v1/beijing_minimum_runner.py":"b84f1473ff4d562890910f7555ef3925823dd921539c8d757118c95504513b9b"}
PUBLIC_PINS={
 OLD+"/PLAN.json":"fedf00ebb1807528514c0685a6b226fd6a466149531adc7a5791ffa07d92af35",
 OLD+"/cpu_lag_runner.py":"36c75b38cd2c072bac4466129249cc5213c2df679ad80232d4ac511ab6d7e7f1",
 OLD+"/runs/"+RUN+"/BASE_FIT_RECEIPT.json":"0100fa0551f4865cfb047716ed5ab12bea3315612f3d11126856bdb6ebed2acf",
 OLD+"/runs/"+RUN+"/SOURCE_CONTRACT.json":"6277e1527cabdb06afba4dc78f36a502e45f18fe83d47c7cd8b2789d0df884b8",
 FIT_FOLDER+"/RESULT.json":"608d6694b2abc1f8bf66cc860e9bc48884f457452af0b70eff60b8beb8d91a2f",
 FIT_FOLDER+"/STARTED.json":"e7da1eeb18638905f616f21de43013b521de777345e73ac1c1d189c11b33321e",
 PRED+"/PREDICTION_MANIFEST.json":MANIFEST,
 PRED+"/REFERENCE_CONTRACT.json":"43b89d5c31195cb46e354f607b465b46cc839a2999f9588cc5996994d4f9cb51",
 PRED+"/SCORES.csv":"edd9eec96146c3d03ea048cb1070c4728cec33402f723b66c602f74afa0badb6",
 PRED+"/PUBLIC_ERROR_STATS.csv.gz":"e61620ee87b54c7670083a848ade0e05362a6b6319c4efdd967b242ac62c2647",
 "postT_V_only_cpu_family_minima_20261016_v1/PLAN.json":"e34033935916dcaded0885ccde8e1aa084f5cc0df994024d0827510d6db53b10"}
CACHE_PINS={ARTIFACT_REL:ARTIFACT,PRIVATE_V:ARCHIVE,
 OLD+"/supervisor_fit_1792112761493733800/RESULT.json":"3252d8d3dac546e30319b1db9c7d64ea7ef8ca271132083b7fdb428805bb053f",
 "postT_cpu_lag_preparation_20261016_v1/fit_1792112761493733800_SPEC.json":"3945a2c9592969e4d59baffcc43327a350332f254426ccc439df33672d9c11bc"}
V_FINGERPRINTS={
 "pred_normalized":dict(shape=[43,48,21],dtype="float32",sha256="4bda04d5109d797d2da95fb501adacfca3d15e73faa70c0ed9dd4cd50ddb0730"),
 "target_normalized":dict(shape=[43,48,21],dtype="float32",sha256="3b16ebbb7722347f5c02af1380e8eef9a7f6524fcbc4f1eec274a50fcb701d12"),
 "observed":dict(shape=[43,48,21],dtype="bool",sha256="004c9cb19d9592eff42a0b7818298ea88f19d515291c2d321c118ff628f37f42"),
 "origins":dict(shape=[43],dtype="int64",sha256="6d6bc4124e0d770ceae0819bbd59b36ca3750ff610aed7d26dbfe5c1a9e775ce"),
 "channel_ids":dict(shape=[21],dtype="<U20",sha256="558b9106a501307d57f131f89e86530120d04cb6da9f0b6525794348a0de3919")}
MODEL_ROOT=".cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8"
MODEL_BINDINGS=[dict(snapshot_relative_path=MODEL_ROOT+"/config.json",resolved_blob_relative_path=".cache/huggingface/models--amazon--chronos-bolt-small/blobs/d8f9b65552d4d211e28d332ad5666484092bbf77",sha256="9ca0ebbe049ccebacb0546361e8aa728f0cf104cd18d1838d78d296aa317bd21",bytes=1121),dict(snapshot_relative_path=MODEL_ROOT+"/model.safetensors",resolved_blob_relative_path=".cache/huggingface/blobs/61/61439ff1becff564d030cdbe1ce9dfce0673fc0ebc4ca0a009d3fe02ce60bba0",sha256="06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd",bytes=190888824)]
JENA_ASSET=dict(relative_path=".cache/tsfm_peft_practical_controls_v7_20260928/data/jena_trainval.npz",sha256="29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933",bytes=1561064,contract_relative_path="research/level_matched_learned_ed_v18_20261006/data_contract_jena.json",contract_sha256="678bd4cbd63265290c0eeed8a78b09545a05219b666fd3b818682ce385906820",source_rows=35135,permitted_rows=20965,prefix_rows=16772)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def inside(root,name):
    base=Path(root).resolve();path=base/name
    if Path(name).is_absolute() or ".." in Path(name).parts or any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in (path,*path.parents)) or not path.resolve().is_relative_to(base):raise ValueError("Root-relative nonsymlink metadata/output required")
    return path


def new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def draft_plan():
    return dict(schema="JENA_EXISTING_B2_REFERENCE_v1",namespace=NAME,status="DRAFT_NOT_EXECUTABLE",runner_sha256=None,ROOT_admitted=False,
        run_id=RUN,stage="POST_T_V_ONLY_DEVELOPMENT",allowed_modes=["rescore","replay","cost"],new_training_conditions=0,
        public_pins=copy.deepcopy(PUBLIC_PINS),cache_pins=copy.deepcopy(CACHE_PINS),dependencies=copy.deepcopy(DEPENDENCIES),
        original_source39_immutable9_plan=OLD+"/PLAN.json",task_contract_source="postT_V_only_cpu_family_minima_20261016_v1/PLAN.json:task_contracts.jena",
        original_CPU_fullV_provenance=FIT_FOLDER+"/RESULT.json",historical_fit_guard_each_RSS_bytes=4*1024**3,historical_fit_guard_group_RSS_bytes=8*1024**3,
        canonical_route=canonical_route(),V_fingerprints=copy.deepcopy(V_FINGERPRINTS),materialized_prefix_rows=16772,model_snapshot_bindings=copy.deepcopy(MODEL_BINDINGS),task_asset_binding=copy.deepcopy(JENA_ASSET),
        original_primary_split_ranges=dict(A_fit=[0,10482],A_stop=[10482,12579],M_cal=[12579,14675],V_select=[14675,16772],T_final=[16772,20965]),
        new_T_target_operations_allowed=False,protected_tail_numeric_parse_allowed=False,deadline_unix=DEADLINE,
        resources=dict(CPU_threads=2,max_worker_RSS_bytes=3*1024**3,maximum_model_workers=1,total_owned_RSS_limit_bytes=3*1024**3,storage_limit_bytes=30*1024**3,admission_reserve_bytes=256*1024**2,GPU_process_seconds=0,ROOT_CPU_supervisor_required=True),
        costs=dict(batches=[1,8],warmups=10,timed_trials=20,stage_trials=20,quiet_max_age_seconds=120,actual_matching_rescore_replay_result_SHA_required=True),
        accuracy_scope="Reuse original seed92601 CPU fullV; new saved-V FP64/CPU6 and fresh matched costs are validation, not new fit/generalization.",
        normalization_ancestry="Inherited legacy TRAIN mean/std used to recover physical prefix; no claim that historical normalization ancestry was removed.")


def validate_plan(plan):
    expected=draft_plan()
    if plan.get("status")!="SEALED_BY_ROOT" or plan.get("ROOT_admitted") is not True or not isinstance(plan.get("runner_sha256"),str) or len(plan["runner_sha256"])!=64 or any(c not in "0123456789abcdef" for c in plan["runner_sha256"]):raise ValueError("Explicit new ROOT seal/admission/current source SHA required")
    if any(plan.get(k)!=v for k,v in expected.items() if k not in {"status","runner_sha256","ROOT_admitted"}):raise ValueError("Only exact existing B2 validation/cost scope and original resources permitted")


def canonical_route():
    return dict(run_id=RUN,dataset="jena",family="B2",fold="primary",latent_k=6,canonical_fit_end=12579,
        artifact_cache_relative_path=ARTIFACT_REL,artifact_sha256=ARTIFACT,foundation_sha=FOUNDATION,expected_prediction_model_sha=PREDICTION_MODEL,row_microbatch=128)


def validate_manifest(value):
    want=dict(run_id=RUN,split="V_select",cohort="jena:primary:V_select:stride48",data_sha=DATA,scaler_sha=SCALER,source_sha=PREDICTION_SOURCE,config_sha=PREDICTION_CONFIG,model_sha=PREDICTION_MODEL,archive_sha256=ARCHIVE,arrays=V_FINGERPRINTS)
    meta=value.get("metadata",{});meta_want=dict(artifact_sha256=ARTIFACT,family="B2",fold="primary",dense=False,role="base_parent",parent_fit_end=12579,canonical_scaler_fit_end=12579,parent_normalization_fit_end=12579)
    if any(value.get(k)!=v for k,v in want.items()) or any(meta.get(k)!=v for k,v in meta_want.items()) or any(value.get("score",{}).get(k)!=v for k,v in dict(n_origins=43,n_observed=43344,horizon=48,num_channels=21).items()):raise ValueError("Exact historical CPU parent manifest/scaler/V fingerprints differ")


def validate_route_identity(route):
    bundle=route.bundle
    if (route.run_id!=RUN or route.family!="B2" or route.artifact_sha256!=ARTIFACT or route.prediction_model_sha!=PREDICTION_MODEL or bundle.get("model_state_sha")!=STATE
        or any(bundle.get("spec",{}).get(k)!=v for k,v in dict(run_id=RUN,family="B2",latent_k=6,seed=92601,device="cpu").items()) or bundle.get("normalization",{}).get("fit_end")!=12579
        or any(bundle.get("contract",{}).get(k)!=v for k,v in dict(source_sha=FIT_SOURCE,data_sha=DATA,model_sha=FOUNDATION,scaler_sha=SCALER).items())):raise ValueError("Original B2 selected tensor/spec/scaler/foundation identity differs")


def expected_counters(batches):
    if not batches or any(isinstance(b,bool) or not isinstance(b,int) or b<1 for b in batches):raise ValueError("Positive physical batches required")
    streams=sum(b*6 for b in batches);return dict(streams=streams,physical_streams=streams,physical_calls=sum(math.ceil(b*6/128) for b in batches))


def check_scientific_modules(public,values):
    for value in values:
        if value.__name__ not in {"data_contract","evaluation","fit_base","deployment","gpu_runtime","measure_costs"} or Path(value.__file__).resolve()!=(Path(public)/(value.__name__+".py")).resolve():raise ValueError("Actual imported scientific module differs from original pinned public source")


def validate_proof(value,mode,plan_sha,entry_sha):
    status={"rescore":"PASS_INDEPENDENT_FULL_V_FP64","replay":"PASS_FRESH_CPU6_REPLAY"}[mode];science=value.get("scientific_result") or {}
    want=dict(run_id=RUN,mode=mode,status=status,plan_sha256=plan_sha,entry_source_sha256=entry_sha,error_class=None,postflight_error_class=None,original_source39_immutable9_B2_reference_pins_preserved=True,new_T_target_operations=0)
    if any(value.get(k)!=v for k,v in want.items()) or science.get("status")!=status or science.get("artifact_sha256")!=ARTIFACT or science.get("manifest_sha256")!=MANIFEST or (mode=="replay" and science.get("executed_now") is not True):raise ValueError("Actual matching new-entry same-plan/artifact validation proof required")


def metadata(public,cache,asset,plan_sha,plan_path):
    plan_path=inside(public,os.path.relpath(Path(plan_path).absolute(),public))
    if plan_path!=public/NAME/"B2_REFERENCE_PLAN.json" or sha(plan_path)!=plan_sha:raise ValueError("Exact new ROOT B2 reference PLAN path/SHA required")
    plan=read(plan_path);validate_plan(plan)
    if sha(inside(public,NAME+"/jena_b2_reference.py"))!=plan["runner_sha256"]:raise ValueError("Current ROOT-pinned reference entry changed")
    for name,digest in {**PUBLIC_PINS,**DEPENDENCIES}.items():
        if sha(inside(public,name))!=digest:raise ValueError("Original fixed reference/dependency changed")
    old=read(inside(public,OLD+"/PLAN.json"));pins={**old["source39"],**old["immutable_public"]}
    if len(old["source39"])!=39 or len(old["immutable_public"])!=9:raise ValueError("Original source39/immutable9 required")
    for root,group in ((public,pins),(cache,CACHE_PINS)):
        for name,digest in group.items():
            if sha(inside(root,name))!=digest:raise ValueError("Original opaque source/reference/artifact bytes changed")
    libs={key:module(inside(public,name),"reference_b2_"+key) for key,name in (("family",NAME+"/family_minimum_runner.py"),("priority","priority_b2_validation_20261016_v1/priority_runner.py"),("primitive","priority_b2_validation_20261016_v1/safe_jena_prefix.py"),("seed","priority_core_followup_20261016_v1/jena_seed_runner.py"),("common","priority_valid_minima_20261016_v1/beijing_minimum_runner.py"))}
    family=read(inside(public,"postT_V_only_cpu_family_minima_20261016_v1/PLAN.json"));original=family["task_contracts"]["jena"]
    for key in ("data_sha256","array_sha256","columns","primary_split_ranges","A_fit_statistics_fingerprints","search_scaler_sha256","refit_scaler_sha256"):
        if original[key]!=old["task_contract"][key]:raise ValueError("Same original Jena task/normalization metadata required")
    binding=libs["family"].draft_plan()
    if binding["model_snapshot_bindings"]!=MODEL_BINDINGS or binding["task_asset_bindings"]["jena"]!=JENA_ASSET:raise ValueError("Reviewed shared prefix/snapshot metadata changed")
    libs["common"].check_model_snapshot(asset,old["model_asset_relative_path"],MODEL_BINDINGS)
    path=inside(asset,JENA_ASSET["relative_path"])
    if path.stat().st_size!=JENA_ASSET["bytes"] or sha(path)!=JENA_ASSET["sha256"] or sha(inside(asset,JENA_ASSET["contract_relative_path"]))!=JENA_ASSET["contract_sha256"]:raise ValueError("Legacy opaque Jena raw asset/contract changed")
    if sha(inside(asset,old["instance_norm_source_relative_path"]))!=old["instance_norm_source_sha256"] or sha(inside(public.parent,"temporal_frontend_campaign_20261015_v1/runtime.py"))!=old["instance_norm_source_sha256"]:raise ValueError("Original actual instance-normalization source changed")
    outer=read(inside(public,FIT_FOLDER+"/RESULT.json"));started=read(inside(public,FIT_FOLDER+"/STARTED.json"));guard=read(inside(cache,OLD+"/supervisor_fit_1792112761493733800/RESULT.json"))
    if outer.get("status")!="COMPLETE_FRESH_CPU_FIT_AND_FULL_V" or outer.get("error_class") is not None or outer.get("postflight_error_class") is not None or outer.get("frozen_foundation_unchanged") is not True or outer.get("cuda_initialized") is not False or outer.get("scientific_result",{}).get("full_V_completed") is not True or outer["scientific_result"].get("V_manifest_sha256")!=MANIFEST:raise ValueError("Actual original frozen CPU fullV provenance required")
    workers=[w for w in guard["processes"] if any(p.get("pid")==started["actual_pid"] and p.get("create_time")==started["actual_create_time"] for p in w.get("owned_processes",[]))]
    if guard.get("status")!="COMPLETE" or guard.get("remaining_owned_processes")!=[] or guard.get("supervisor_failure_class") is not None or guard.get("helper_unchanged") is not True or guard.get("helper_sha256")!="82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b" or guard.get("spec_file_sha256")!=CACHE_PINS["postT_cpu_lag_preparation_20261016_v1/fit_1792112761493733800_SPEC.json"] or len(workers)!=1 or workers[0].get("returncode")!=0 or workers[0].get("stop_reason") is not None or workers[0].get("process_tree_errors")!=[] or workers[0].get("remaining_owned_processes")!=[] or workers[0].get("observed_tree_join_confirmed") is not True:raise ValueError("Recorded original CPU fit worker/source closure differs")
    manifest=read(inside(public,PRED+"/PREDICTION_MANIFEST.json"));validate_manifest(manifest)
    reference=read(inside(public,PRED+"/REFERENCE_CONTRACT.json"));receipt=read(inside(public,OLD+"/runs/"+RUN+"/BASE_FIT_RECEIPT.json"))
    if any(receipt.get(k)!=v for k,v in dict(status="COMPLETE_VALID",run_id=RUN,family="B2",seed=92601,artifact_sha256=ARTIFACT,model_state_sha=STATE,refit_scaler_sha=SCALER).items()) or reference["arrays"]!={k:v for k,v in V_FINGERPRINTS.items() if k!="pred_normalized"}:raise ValueError("Original selected fit/independent V reference differs")
    return plan,old,family,manifest,reference,libs


def load_arrays(cache,manifest,reference,context):
    np=context["np"]
    with np.load(inside(cache,PRIVATE_V),allow_pickle=False) as z:
        if set(z.files)!=set(V_FINGERPRINTS):raise ValueError("Exact original safe-V five members required")
        arrays={name:z[name] for name in z.files}
    if any(context["evaluation"].array_fingerprint(v)!=manifest["arrays"][k] for k,v in arrays.items()):raise ValueError("Original saved-V array fingerprint changed")
    context["libs"]["priority"].validate_cohort(arrays,reference,context["evaluation"]);return arrays


def execute(public,cache,asset,plan_sha,mode,deadline,output,batch=None,plan_path=None,quiet_path=None,quiet_sha=None,proofs=None):
    output=Path(output).absolute()
    if output.exists():raise FileExistsError("Existing output preserved before inputs")
    public,cache,asset=map(lambda p:Path(p).resolve(),(public,cache,asset))
    output=inside(public/NAME,os.path.relpath(output,public/NAME))
    if not output.is_relative_to(public/NAME/"diagnostics"):raise ValueError("Fresh reference diagnostic output required")
    if mode not in ("rescore","replay","cost") or not time.time()<deadline<=DEADLINE or (mode=="cost" and batch not in (1,8)) or (mode!="cost" and batch is not None):raise ValueError("Only explicit original-deadline validation/B1B8 cost modes allowed")
    plan,old,family,manifest,reference,libs=metadata(public,cache,asset,plan_sha,plan_path);libs["priority"].require_environment()
    import psutil
    libs["seed"].resource_preflight(public,cache,plan,psutil)
    output.mkdir(parents=True,exist_ok=False);began=time.time();failure=post_error=None;result=None;pool=None;post_path=None
    new(output/"STARTED.json",dict(run_id=RUN,mode=mode,plan_sha256=plan_sha,entry_source_sha256=plan["runner_sha256"],actual_pid=os.getpid(),actual_create_time=psutil.Process().create_time(),deadline_unix=deadline,new_training=0))
    try:
        sys.path.insert(0,str(public))
        import numpy as np
        import data_contract as data
        import evaluation
        check_scientific_modules(public,[data,evaluation])
        context=dict(np=np,data=data,evaluation=evaluation,libs=libs,reference=reference)
        if mode=="rescore":
            arrays=load_arrays(cache,manifest,reference,context)
            with inside(public,PRED+"/PUBLIC_ERROR_STATS.csv.gz").open("rb") as raw:
                with gzip.open(raw,"rt",encoding="utf-8",newline="") as f:stats=list(csv.DictReader(f))
            with inside(public,PRED+"/SCORES.csv").open(encoding="utf-8",newline="") as f:scores=list(csv.DictReader(f))
            if len(scores)!=1:raise ValueError("One original safe-V score row required")
            result=libs["priority"].independent_rescore(arrays,manifest,stats,scores[0],np);result.update(run_id=RUN,artifact_sha256=ARTIFACT,manifest_sha256=MANIFEST,full_V_origins=43,accuracy_reused=True,new_training=0)
            new(output/"RESCORE_RECEIPT.json",result)
        else:
            if mode=="cost":
                if not quiet_path or not quiet_sha:raise ValueError("Fresh ROOT quiet proof path/SHA required")
                quiet_path=inside(cache,os.path.relpath(Path(quiet_path).absolute(),cache))
                if sha(quiet_path)!=quiet_sha:raise ValueError("Quiet proof bytes differ")
                libs["family"].require_quiet(read(quiet_path),time.time())
                for kind in ("rescore","replay"):
                    path,digest=(proofs or {}).get(kind,(None,None))
                    if not path or not digest:raise ValueError("Actual matching new validation proofs required")
                    path=inside(public/NAME,os.path.relpath(Path(path).absolute(),public/NAME))
                    if sha(path)!=digest:raise ValueError("Validation proof bytes changed")
                    validate_proof(read(path),kind,plan_sha,plan["runner_sha256"]);proofs[kind]=(path,digest)
            import torch
            import fit_base as fit
            import deployment
            import gpu_runtime
            check_scientific_modules(public,[fit,deployment,gpu_runtime])
            torch.set_num_threads(2)
            if torch.cuda.is_initialized():raise RuntimeError("CPU reference initialized CUDA")
            context["fit"]=fit;context["priority"]=libs["priority"]
            task,access=libs["family"].safe_task(asset,family,dict(dataset="jena"),dict(primitive=libs["primitive"],priority=libs["priority"]),context)
            new(output/"PREFIX_ACCESS_RECEIPT.json",access)
            pool=deployment.FoundationPool(inside(asset,old["model_asset_relative_path"]),"cpu");foundation=pool.get(48)
            if gpu_runtime.state_hash(foundation.model)!=FOUNDATION or foundation.row_microbatch!=128 or foundation.checkpoint_inputs is not False or foundation.instance_norm_source_sha256!=old["instance_norm_source_sha256"] or any(p.requires_grad or p.grad is not None for p in foundation.model.parameters()):raise ValueError("Exact frozen CPU/row128 foundation/norm identity differs")
            route=deployment.Route(canonical_route(),task,cache,device="cpu",foundation_pool=pool)
            with libs["family"].route_lifetime(route,foundation):
                validate_route_identity(route);before=pool.counters()
                if mode=="replay":
                    arrays=load_arrays(cache,manifest,reference,context);indices=libs["priority"].replay_indices(43,np);origins=arrays["origins"][indices];pred=[]
                    for start in (0,4):
                        libs["priority"].check_deadline(deadline);oo=origins[start:start+4];physical=np.stack([task["physical"][int(o)-512:int(o)] for o in oo]);observed=np.stack([task["observed"][int(o)-512:int(o)] for o in oo]);raw=route.predict_physical(physical,observed)
                        pred.append(((raw-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32))
                    actual=np.concatenate(pred);delta={k:pool.counters()[k]-before[k] for k in ("streams","physical_streams","physical_calls")};passed=bool(np.allclose(actual,arrays["pred_normalized"][indices],atol=1e-5,rtol=1e-4))
                    if delta!=expected_counters([4,2]):raise ValueError("CPU6 physical K6 counters differ")
                    result=dict(status="PASS_FRESH_CPU6_REPLAY" if passed else "FAIL_REPLAY_PARITY",executed_now=True,run_id=RUN,artifact_sha256=ARTIFACT,manifest_sha256=MANIFEST,origins=origins.tolist(),microbatches=[4,2],F_counters=delta,atol=1e-5,rtol=1e-4,max_abs=float(np.abs(actual.astype(np.float64)-arrays["pred_normalized"][indices].astype(np.float64)).max()),accuracy_reused=True,new_training=0)
                    new(output/"REPLAY_RECEIPT.json",result)
                    if not passed:raise ValueError("Fresh original CPU6 physical route parity failed")
                else:
                    import measure_costs
                    check_scientific_modules(public,[measure_costs])
                    origins=data.origins((14675,16772),512,48,48)[:batch];physical=np.stack([task["physical"][int(o)-512:int(o)] for o in origins]);observed=np.stack([task["observed"][int(o)-512:int(o)] for o in origins])
                    class Lease:
                        def check(self):libs["priority"].check_deadline(deadline)
                    rows=measure_costs.measure_balanced([route],physical,observed,warmups=10,timed_trials=20,stage_trials=20,lease=Lease());delta={k:pool.counters()[k]-before[k] for k in ("streams","physical_streams","physical_calls")};want=expected_counters([batch])
                    if len(rows)!=50 or any((r["F_logical_streams"],r["F_physical_streams"],r["F_calls"])!=(want["streams"],want["physical_streams"],want["physical_calls"]) for r in rows) or delta!={k:v*50 for k,v in want.items()}:raise ValueError("Original actual50 physical whole-route trials/counters differ")
                    if sha(quiet_path)!=quiet_sha or any(sha(p)!=s for p,s in proofs.values()):raise ValueError("Cost quiet/prerequisite proof changed")
                    trials=output/"COST_TRIALS.csv.gz";measure_costs._write_trials(trials,rows)
                    result=dict(status="COMPLETE_PROTOCOL",run_id=RUN,device="cpu",CPU_threads=2,batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,actual_trial_rows=50,summaries=measure_costs.summarize(rows),memory=route.memory_state(),trials_sha256=sha(trials),artifact_sha256=ARTIFACT,accuracy_manifest_sha256=MANIFEST,validation_result_sha256={k:v[1] for k,v in proofs.items()},quiet_proof_sha256=quiet_sha,origins=origins.tolist(),physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),F_counters=delta,F_logical_streams_per_trial=want["streams"],F_calls_per_trial=want["physical_calls"],cached_Z_or_forecast_reads=0,physical_output_and_inverse_dtype="float64",stage_probe_separate_intrusive_scope=True,training_peak_included=False,model_load_seconds=route.load_seconds,model_loading_in_whole_trial_timer=False,accuracy_reused=True,new_training=0)
                    new(output/"COST_RECEIPT.json",result)
            foundation.assert_frozen_unchanged();result["F_final_state_sha256"]=gpu_runtime.state_hash(foundation.model);result["base_restored_after_original_Route_close"]=True
        libs["priority"].check_deadline(deadline)
    except Exception as exc:
        failure=type(exc).__name__;private=cache/NAME/"failures"/("B2_reference_"+mode+"_"+str(time.time_ns())+".json");new(private,dict(error_class=failure,error=str(exc),traceback=traceback.format_exc()));result=dict(status="FAILED_PRESERVED",error_class=failure,private_failure_cache_relative_path=private.relative_to(cache).as_posix())
    finally:
        counters=None if pool is None else pool.counters();instances=0 if pool is None else pool.instances_created
        try:
            if pool is not None:pool.close()
            metadata(public,cache,asset,plan_sha,plan_path);libs["seed"].resource_preflight(public,cache,plan,psutil)
        except Exception as exc:
            post_error=type(exc).__name__;post_path=cache/NAME/"failures"/("B2_reference_post_"+mode+"_"+str(time.time_ns())+".json");new(post_path,dict(error_class=post_error,error=str(exc),traceback=traceback.format_exc()))
        new(output/"RESULT.json",dict(status="FAILED_PRESERVED" if failure or post_error else result["status"],run_id=RUN,reference_role="EXISTING_B2_REFERENCE",mode=mode,plan_sha256=plan_sha,entry_source_sha256=plan["runner_sha256"],scientific_result=result,error_class=failure,postflight_error_class=post_error,postflight_private_failure_cache_relative_path=None if post_path is None else post_path.relative_to(cache).as_posix(),original_source39_immutable9_B2_reference_pins_preserved=post_error is None,new_T_target_operations=0,new_training=0,accuracy_reused=True,actual_F_counters=counters,foundation_instances_created=instances,elapsed_wall_seconds=time.time()-began,actual_CPU_process_seconds=sum(psutil.Process().cpu_times()[:2]),RSS_bytes=psutil.Process().memory_info().rss,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,independent_generalization_claimed=False,ROOT_external_CPU2_RSS3GiB_deadline_quiet_guard_required=True,historical_CPU_fit_guard_4GiB_not_relabelled_as_new_3GiB=True))
    if failure or post_error:raise RuntimeError("Reference proof/cost failed; original assets and new failure receipts preserved")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    for name in ("public-root","cache-root","asset-root","plan","plan-sha","output","quiet-proof","quiet-sha","rescore-proof","rescore-sha","replay-proof","replay-sha"):p.add_argument("--"+name)
    p.add_argument("--mode",choices=("rescore","replay","cost"));p.add_argument("--batch",type=int,choices=(1,8));p.add_argument("--deadline",type=float);a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,writes=0,new_training=0,scientific_or_child_operations=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules)));return 0
    if not a.root_triggered or any(getattr(a,k) is None for k in ("public_root","cache_root","asset_root","plan","plan_sha","mode","deadline","output")):p.error("ROOT trigger/current sealed PLAN SHA/explicit roots/mode/deadline/fresh output required")
    result=execute(a.public_root,a.cache_root,a.asset_root,a.plan_sha,a.mode,a.deadline,a.output,a.batch,a.plan,a.quiet_proof,a.quiet_sha,{k:(getattr(a,k+"_proof"),getattr(a,k+"_sha")) for k in ("rescore","replay")})
    print(json.dumps(dict(status=result["status"],run_id=RUN,new_training=0)));return 0 if result["status"].startswith(("PASS","COMPLETE")) else 1


if __name__=="__main__":raise SystemExit(main())
