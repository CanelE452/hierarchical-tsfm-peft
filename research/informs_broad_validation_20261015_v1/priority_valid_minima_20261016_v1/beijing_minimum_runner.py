"""Three original Beijing minima, fresh prefix-only CPU development.

Default performs no I/O or scientific import. ROOT alone seals the plan,
dispatches one explicit condition and enforces CPU/RSS/quiet-window limits.
Original fit_base._stage supplies all learning/selection; no scientific
source, original registry, old probe claim or protected T is modified.
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

NAME = "priority_valid_minima_20261016_v1"
OLD = "postT_V_only_cpu_required_minima_20261016_v1"
OLD_PLAN_SHA = "04acc4aabffd02a7266f16fee5c6a3884313fac6ca4b5b46ef0db45eb2fe5152"
OLD_REG_SHA = "fc5f4012a6ced7a1849fcd6474f67d81b600dadaa0014f56df4cda4b8a140f29"
UPSTREAM_PLAN_SHA = "475f983058871ff36b4112b6aa56dc252b17aea47eb6ae1744ff119c2ad5138f"
DEADLINE, L, H, C, K, SEED, PREFIX = 1792222550, 512, 48, 11, 6, 92601, 16830
BOUNDS = dict(A_fit=[0,10519], A_stop=[10519,12622], M_cal=[12622,14726], V_select=[14726,16830], T_final=[16830,21038])
DATA_SHA = "b39023bcae04736eff9a0fd9a3d56e10dc7f32bd668a53d681290cb9193d8269"
SCALER_SHA = "3348bcf1fa76d382f23cf20597fa90de6e21dac587d4ca9f6883f235a10f93fb"
DEPENDENCIES = {
    "priority_b2_validation_20261016_v1/priority_runner.py": "f3e4279ddd250a51b09ce88938f8f202dfd02e3b24d1adff3bb36c567dc7ef07",
    "priority_b2_validation_20261016_v1/safe_jena_prefix.py": "6b496dcbbb6cef48acb9010c630ed81359a2d60bff89aaec9e9e77c4b6d0b541",
    "priority_core_followup_20261016_v1/jena_seed_runner.py": "ad4b76acf51fb416dea4f69c36940507db1f92e223c620388ded45381d6e28e7",
    "priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py": "d64d543499ed21c5a294ace19967c555d05bcfde0cd5ea6c98cbe8ddb23a4ae5",
    OLD + "/required_minima_runner.py": "e11c34918862a698e962cf2a3c151ea6f7484032985f52d25cb7220da6b5d37b",
}
PROBE_SHAS = (
    "f40b78124523f6bfae0f00559b87f8a5232923cfe653a61f2183ebf062dbbd76",
    "40879e1c9717dc317ecacf48ec92d11def9620b3fd3677d5b18e13e86c9478c8",
    "51cb0fbda6a1644ea72a7ba60e7dbeb8764050ea9b8df8661e57c9e0f47f7aaf",
)
MODEL_REL = ".cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8"
MODEL_SNAPSHOT_BINDINGS = [
    dict(snapshot_relative_path=MODEL_REL+"/config.json",
         resolved_blob_relative_path=".cache/huggingface/models--amazon--chronos-bolt-small/blobs/d8f9b65552d4d211e28d332ad5666484092bbf77",
         sha256="9ca0ebbe049ccebacb0546361e8aa728f0cf104cd18d1838d78d296aa317bd21",bytes=1121),
    dict(snapshot_relative_path=MODEL_REL+"/model.safetensors",
         resolved_blob_relative_path=".cache/huggingface/blobs/61/61439ff1becff564d030cdbe1ce9dfce0673fc0ebc4ca0a009d3fe02ce60bba0",
         sha256="06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd",bytes=190888824),
]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""): value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write("\n")


def relative(root, name):
    root = Path(root).absolute(); path = root / name
    if Path(name).is_absolute() or any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in (path,*path.parents)):
        raise ValueError("Explicit nonsymlink relative path required")
    root, path = root.resolve(), path.resolve()
    if not path.is_relative_to(root): raise ValueError("Path escapes declared root")
    return path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def conditions():
    rows = []
    for index,family in enumerate(("B2","LAG_CAUSAL","LAG_BIDIR"),1):
        block = "E1" if family == "B2" else "E3"
        upstream = f"primary_beijing_{block}_{family}_k6_s92601"
        identifier = f"postT_cpu_beijing_{block}_{family}_k6_s92601_required_v1"
        spec = dict(run_id=identifier,dataset="beijing",family=family,latent_k=6,seed=92601,fold="primary",epochs=[.25,1.,3.],
            lr=.001,device="cpu",microbatch=4,effective_batch=16,F_row_microbatch=128,checkpoint_inputs=False,lora_rank=8,lora_alpha=16,
            stage_label="postT_cpu_fresh_V_only_required_minima",provenance=dict(upstream_registered_id=upstream))
        rows.append(dict(run_id=identifier,family=family,latent_k=6,upstream_registered_id=upstream,spec=spec,
            probe=dict(relative_path=OLD+f"/diagnostics/probe_root_v1_{index:02d}/RESULT.json",sha256=PROBE_SHAS[index-1])))
    return rows


def draft_plan():
    return dict(schema="PRIORITY_BEIJING_VALID_MINIMA_v1",namespace=NAME,status="DRAFT_NOT_EXECUTABLE",runner_sha256=None,
        stage="POST_T_V_ONLY_DEVELOPMENT",conditions=conditions(),maximum_new_training_conditions=3,
        original_required_plan_sha256=OLD_PLAN_SHA,original_required_registry_sha256=OLD_REG_SHA,
        upstream_priority_plan_sha256=UPSTREAM_PLAN_SHA,dependencies=copy.deepcopy(DEPENDENCIES),
        model_snapshot_bindings=copy.deepcopy(MODEL_SNAPSHOT_BINDINGS),
        original_primary_split_ranges=copy.deepcopy(BOUNDS),materialized_prefix_rows=PREFIX,
        new_T_target_operations_allowed=False,protected_tail_numeric_parse_allowed=False,deadline_unix=DEADLINE,
        resources=dict(CPU_threads=2,max_worker_RSS_bytes=3*1024**3,maximum_model_workers=2,total_owned_RSS_limit_bytes=6*1024**3,
            storage_limit_bytes=30*1024**3,admission_reserve_bytes=256*1024**2,costs_must_not_overlap_training=True,ROOT_CPU_supervisor_required=True),
        V_reference_public_relative_path="predictions/primary_beijing_E1_B2_k6_s92601/V_select/PREDICTION_MANIFEST.json",
        V_reference_sha256=None,
        original_task_identity="INHERITED_NOT_RECOMPUTED; actual prefix/A50/A60/V fingerprints checked separately",
        lag_interpretation="Both lag encoders consume the supplied pre-origin512-step window only; bidirectional offsets are not per-timestamp causal representations or directed causal graphs",
        prior_probe_scope="Exact old per-condition discarded CPU probes reused after source/recipe/Task-prefix/scaler validation; their historical full-loader I/O is not erased",
        interruption="Original _stage deadline path preserves consistent optimizer-boundary checkpoint. Arbitrary signals are not checkpoint-safe here; partial paths/claims remain and are never resumed or renamed",
        actual_training_prediction_replay_cost_or_child_operations=0,
        ROOT_seal_required="Read three source files; replace status/runner_sha256 and bind V_reference_sha256 in a NEW PLAN.json; preserve this draft; admit only after priority/P2/N3 and original resources/T contract review")


def validate_plan(plan):
    required = draft_plan()
    if plan.get("status") != "SEALED_BY_ROOT" or not isinstance(plan.get("runner_sha256"),str) or len(plan["runner_sha256"]) != 64:
        raise ValueError("An explicit ROOT-sealed new PLAN is required; draft/selfsha-null cannot execute")
    fields = ("schema","namespace","stage","conditions","maximum_new_training_conditions","original_required_plan_sha256",
        "original_required_registry_sha256","upstream_priority_plan_sha256","dependencies","original_primary_split_ranges",
        "materialized_prefix_rows","new_T_target_operations_allowed","protected_tail_numeric_parse_allowed","deadline_unix","resources","model_snapshot_bindings")
    if any(plan.get(field) != required[field] for field in fields):
        raise ValueError("Only the exact three original Beijing K6 recipes and unchanged resources are allowed")
    validate_model_bindings(plan["model_snapshot_bindings"],MODEL_REL)


def validate_model_bindings(bindings, model_relative):
    expected = {model_relative+"/"+name for name in ("config.json","model.safetensors")}
    if (not isinstance(bindings,list) or len(bindings) != 2
            or any(not isinstance(row,dict) or set(row) != {"snapshot_relative_path","resolved_blob_relative_path","sha256","bytes"} for row in bindings)
            or {row["snapshot_relative_path"] for row in bindings} != expected):
        raise ValueError("Exactly the original config.json and model.safetensors bindings required")
    for row in bindings:
        if (type(row["bytes"]) is not int or row["bytes"] <= 0 or not isinstance(row["sha256"],str) or len(row["sha256"]) != 64
                or any(value not in "0123456789abcdef" for value in row["sha256"])
                or not isinstance(row["resolved_blob_relative_path"],str) or Path(row["resolved_blob_relative_path"]).is_absolute()):
            raise ValueError("Positive exact blob size/SHA and explicit relative blob required")


def check_model_snapshot(asset, model_relative, bindings):
    validate_model_bindings(bindings,model_relative)
    asset = relative(asset,".")
    for row in bindings:
        snapshot = asset/row["snapshot_relative_path"]
        blob = relative(asset,row["resolved_blob_relative_path"])
        resolved = snapshot.resolve(strict=True)
        if (not resolved.is_relative_to(asset) or resolved != blob or not blob.is_file()
                or blob.stat().st_size != row["bytes"] or sha(blob) != row["sha256"]):
            raise ValueError("Original snapshot target/config/weight blob identity changed")


def existing_state(public, cache, identifier):
    found, completed = [], []
    for root in (Path(public),Path(cache),Path(public)/OLD,Path(cache)/OLD,Path(public)/NAME,Path(cache)/NAME):
        for kind in ("runs","predictions"):
            path = root/kind/identifier
            if path.exists():
                found.append(path)
                receipt = path/"BASE_FIT_RECEIPT.json"
                if receipt.is_file():
                    value = read(receipt)
                    if value.get("status") == "COMPLETE_VALID" and value.get("run_id") == identifier: completed.append(receipt)
        claim = root/"claims"/("fit_"+identifier+".json")
        if claim.exists(): found.append(claim)
    if completed: return dict(status="COMPLETE_REUSE_REQUIRED",matches=completed)
    return dict(status="EXISTING_ATTEMPT_PRESERVED" if found else "UNSTARTED",matches=found)


def require_quiet(proof, now):
    captured = proof.get("captured_unix")
    if (proof.get("status") != "QUIET_CPU_COST_WINDOW" or proof.get("active_model_workers") != [] or proof.get("active_cost_workers") != []
            or isinstance(captured,bool) or not isinstance(captured,(int,float)) or not math.isfinite(captured) or not 0 <= now-captured <= 120):
        raise ValueError("Fresh120s ROOT quiet proof with both worker lists empty required")


def read_prefix_members(archive, primitive, nested, rows=PREFIX, headers=None):
    return nested.read_prefix_members(archive,primitive,rows,nested.HEADERS if headers is None else headers)


def metadata(public, cache, asset, plan_sha, identifier, process_api):
    public,cache,asset = map(lambda p:relative(p,"."),(public,cache,asset))
    path = relative(public,NAME+"/PLAN.json")
    if not plan_sha or sha(path) != plan_sha: raise ValueError("Explicit sealed new PLAN SHA required")
    plan = read(path); validate_plan(plan)
    if sha(relative(public,NAME+"/beijing_minimum_runner.py")) != plan["runner_sha256"]: raise ValueError("New runner source changed")
    matching = [row for row in plan["conditions"] if row["run_id"] == identifier]
    if len(matching) != 1: raise ValueError("One of the exact three explicitly admitted IDs is required")
    for name,digest in DEPENDENCIES.items():
        if sha(relative(public,name)) != digest: raise ValueError("Reviewed dependency source changed")
    libraries = {key:module(relative(public,name),"beijing_"+key) for key,name in (
        ("priority","priority_b2_validation_20261016_v1/priority_runner.py"),
        ("seed","priority_core_followup_20261016_v1/jena_seed_runner.py"),
        ("primitive","priority_b2_validation_20261016_v1/safe_jena_prefix.py"),
        ("nested","priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py"),
        ("required",OLD+"/required_minima_runner.py"))}
    p1 = libraries["priority"].metadata(public,cache,UPSTREAM_PLAN_SHA,process_api)
    old_path = relative(public,OLD+"/PLAN.json")
    if sha(old_path) != OLD_PLAN_SHA or sha(relative(public,OLD+"/REGISTRY.csv")) != OLD_REG_SHA:
        raise ValueError("Original six-condition plan or scientific registry changed")
    old = read(old_path); condition = matching[0]
    old_condition = next(row for row in old["conditions"] if row["run_id"] == identifier)
    if any(old_condition[key] != condition[key] for key in ("run_id","family","latent_k","upstream_registered_id","spec")):
        raise ValueError("Exact old registered condition/recipe differs")
    if old["source39"] != p1["source39"] or old["immutable_public"] != p1["immutable_public"]:
        raise ValueError("Original scientific/locked-T metadata pins differ")
    task = old["task_contract"]
    if task["shape"] != [21038,C] or task["data_sha256"] != DATA_SHA or task["primary_split_ranges"] != BOUNDS or task["refit_scaler_sha256"] != SCALER_SHA:
        raise ValueError("Exact inherited Beijing Task/A60 identity differs")
    source_meta = task["metadata"]
    source_asset = relative(asset,source_meta["source_reference"])
    source_contract = relative(asset,str(Path(source_meta["source_reference"]).with_name("contract.json")).replace("\\","/"))
    if (source_asset.stat().st_size != source_meta["source_bytes"] or sha(source_asset) != source_meta["source_sha256"]
            or sha(source_contract) != source_meta["source_contract_sha256"] or read(source_contract)["bounds"]["fit"] != [0,21038]):
        raise ValueError("Original opaque source asset/contract changed")
    recipe_name = "runs/"+condition["upstream_registered_id"]+"/SOURCE_CONTRACT.json"
    if sha(relative(public,recipe_name)) != old["upstream_recipe_metadata"][recipe_name]: raise ValueError("Original upstream recipe changed")
    libraries["required"].validate_original_recipe(read(relative(public,recipe_name)),old_condition,old["source39"],task,old["foundation_tensor_sha256"])
    probe = condition["probe"]
    if sha(relative(public,probe["relative_path"])) != probe["sha256"]: raise ValueError("Selected old probe bytes changed")
    libraries["required"].validate_probe_proof(read(relative(public,probe["relative_path"])),old_condition,old,OLD_PLAN_SHA)
    model = relative(asset,old["model_asset_relative_path"])
    if not model.is_dir() or model.name != old["model_revision"]: raise FileNotFoundError("Pinned local Small foundation absent")
    check_model_snapshot(asset,old["model_asset_relative_path"],plan["model_snapshot_bindings"])
    for root,name in ((asset,old["instance_norm_source_relative_path"]),(public.parent,"temporal_frontend_campaign_20261015_v1/runtime.py")):
        if sha(relative(root,name)) != old["instance_norm_source_sha256"]: raise ValueError("Actual CPU normalization source changed")
    reference = relative(public,plan["V_reference_public_relative_path"])
    if not plan.get("V_reference_sha256") or sha(reference) != plan["V_reference_sha256"]: raise ValueError("Original full V43 cohort reference must be exact")
    libraries["priority"].check_deadline(plan["deadline_unix"])
    return plan,old,condition,libraries


def safe_task(asset, old, libraries, context):
    task,access = libraries["nested"].safe_task(asset,old,libraries["primitive"],context["np"])
    data,fit,np = context["data"],context["fit"],context["np"]
    if task["physical"].shape != (PREFIX,C) or access["new_T_numeric_values_parsed"] != 0 or task["data_sha256"] != DATA_SHA:
        raise ValueError("Exact actual pre-T prefix and inherited identity required")
    mean,std,count = data.fit_statistics(task,10519)
    if ({key:data.array_sha256(value) for key,value in (("mean",mean),("std",std),("count",count))} != old["task_contract"]["A_fit_statistics_fingerprints"]
            or fit.scaler_fingerprint(mean,std) != old["task_contract"]["search_scaler_sha256"]):
        raise ValueError("Actual A50 prefix statistics differ from original probe/training contract")
    mean,std,_ = data.fit_statistics(task,12622)
    if fit.scaler_fingerprint(mean,std) != SCALER_SHA: raise ValueError("Actual A60 prefix normalization differs")
    arrays = libraries["priority"].cohort_arrays(task,BOUNDS,mean,std,data,np)
    libraries["priority"].validate_cohort(arrays,context["reference"],context["evaluation"])
    access.update(actual_A50_A60_and_V43_original_fingerprints_matched=True,
        old_probe_historical_full_loader_scope_retained=True,old_probe_is_reused_not_reexecuted=True)
    return task,access


def fit_stages(fit, spec, task, foundation, contract, public_run, private_run, deadline):
    search = fit._stage(spec,task,10519,BOUNDS["A_stop"],[.25,1.,3.],private_run/"search",public_run/"search",foundation,deadline,contract)
    if search["summary"]["status"] != "SEGMENT_COMPLETE": return dict(status="BUDGET_LIMIT_PRESERVED",phase="search",summary=search["summary"])
    epoch = search["summary"]["selected_epoch"]
    if epoch not in (0.,.25,1.,3.): raise ValueError("Only original A_stop-selected endpoints allowed")
    refit = fit._stage(spec,task,12622,None,[epoch] if epoch else [],private_run/"refit",public_run/"refit",foundation,deadline,contract)
    if refit["summary"]["status"] != "SEGMENT_COMPLETE": return dict(status="BUDGET_LIMIT_PRESERVED",phase="refit",summary=refit["summary"])
    return dict(status="COMPLETE_STAGES",search=search,refit=refit,selected_epoch=epoch)


def validate_stages(search, refit):
    epoch = search.get("selected_epoch")
    if epoch not in (0.,.25,1.,3.): raise ValueError("Original selected epoch required")
    exposures = math.ceil(epoch*12063)
    if (any(search.get(key) != value for key,value in dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],train_origins=9960,exposures=29880,updates=1868).items())
            or any(refit.get(key) != value for key,value in dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),train_origins=12063,exposures=exposures,updates=math.ceil(exposures/16)).items())):
        raise ValueError("Complete original three-epoch search and fresh selected A60 refit required")


def fit_and_save(condition, task, foundation, context, public_run, private_run, deadline):
    fit,p1 = context["fit"],context["priority"]; spec = condition["spec"]
    source,files = fit._source_contract()
    contract = dict(source_sha=source,config_sha=fingerprint(dict(spec=spec,L=L,H=H,C=C,columns=task["columns"],
        inherited_array_sha256=task["array_sha256"],materialized_prefix_sha256=task["materialized_prefix_sha256"])),data_sha=DATA_SHA,model_sha=fit._model_sha(foundation))
    new(public_run/"SOURCE_CONTRACT.json",dict(**contract,files=files,spec=spec,original_task_identity="INHERITED_NOT_RECOMPUTED",
        actual_prefix_sha256=task["materialized_prefix_sha256"],original_probe=condition["probe"],prefix_only_new_entry=True))
    stage = fit_stages(fit,spec,task,foundation,contract,public_run,private_run,deadline)
    if stage["status"] != "COMPLETE_STAGES": return stage
    search,refit,epoch = stage["search"],stage["refit"],stage["selected_epoch"]
    validate_stages(search["summary"],refit["summary"])
    for label,value in (("search",search),("refit",refit)): p1.write_stage(public_run/label,value)
    bundle = dict(model=refit["model"],normalization=refit["normalization"],spec=spec,contract=refit["summary"]["contract"],
        selected_search_epoch=epoch,search_summary=search["summary"],refit_summary=refit["summary"],
        model_state_sha=p1.state_sha(refit["model"],context["data"]),plan_sha256=context["plan_sha"],actual_prefix_sha256=task["materialized_prefix_sha256"])
    artifact = private_run/"selected_bundle.pt"; p1.save_bundle(artifact,bundle,context["torch"])
    receipt = dict(status="COMPLETE_VALID",run_id=spec["run_id"],family=spec["family"],seed=SEED,epochs=spec["epochs"],selected_epoch=epoch,
        training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch == 3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",
        artifact_sha256=sha(artifact),model_state_sha=bundle["model_state_sha"],refit_scaler_sha=fit.scaler_fingerprint(bundle["normalization"]["mean"],bundle["normalization"]["std"]),
        search_fit_end=10519,refit_fit_end=12622,search_updates=search["summary"]["updates"],refit_updates=refit["summary"]["updates"],
        search_exposures=search["summary"]["exposures"],refit_exposures=refit["summary"]["exposures"],source_sha=source,
        source_stage_API="original_fit_base._stage",original_probe=condition["probe"],probe_weights_optimizer_or_sampler_reused=False,
        original_full_task_identity="INHERITED_NOT_RECOMPUTED",actual_prefix_sha256=task["materialized_prefix_sha256"],
        post_T_development=True,protected_T_target_operations=0,reused_complete_artifact=False)
    new(public_run/"BASE_FIT_RECEIPT.json",receipt)
    model,norm = fit.create_model(spec,task,12622,foundation); model.load_state_dict(bundle["model"],strict=True)
    if any(not context["np"].array_equal(norm[key],bundle["normalization"][key]) for key in ("mean","std")): raise ValueError("Selected deployment normalization differs")
    return dict(status="COMPLETE_VALID",model=model,bundle=bundle,receipt=receipt,artifact=artifact)


def full_prediction(fitted, task, foundation, context, public_folder, private_folder, deadline):
    np,torch,p1 = context["np"],context["torch"],context["priority"]
    norm,spec = fitted["bundle"]["normalization"],fitted["bundle"]["spec"]
    arrays = p1.cohort_arrays(task,BOUNDS,norm["mean"],norm["std"],context["data"],np)
    p1.validate_cohort(arrays,context["reference"],context["evaluation"])
    public_folder.mkdir(parents=True,exist_ok=False); private_folder.mkdir(parents=True,exist_ok=False)
    identity = dict(run_id=spec["run_id"],split="V_select",cohort="beijing:primary:V_select:stride48",data_sha=DATA_SHA,scaler_sha=SCALER_SHA,
        source_sha=fingerprint(context["new_sources"]),config_sha=fingerprint(spec),
        model_sha=fingerprint(dict(artifact_sha=fitted["receipt"]["artifact_sha256"],model_state_sha=fitted["receipt"]["model_state_sha"])))
    new(public_folder/"REFERENCE_CONTRACT.json",dict(**identity,arrays={key:context["evaluation"].array_fingerprint(value) for key,value in arrays.items()}))
    windows = context["training"].FastWindows(task,norm["mean"],norm["std"],"cpu")
    before = {key:int(getattr(foundation,key)) for key in ("streams","physical_streams","physical_calls")}; predictions = []
    fitted["model"].eval()
    with torch.no_grad():
        for start in range(0,43,4):
            p1.check_deadline(deadline); x,_,_ = windows.get(arrays["origins"][start:start+4])
            predictions.append(context["fit"].predict_model(fitted["model"],x,foundation,spec["family"],K).cpu().numpy().astype(np.float32))
    arrays["pred_normalized"] = np.concatenate(predictions)
    delta = {key:int(getattr(foundation,key))-value for key,value in before.items()}
    if arrays["pred_normalized"].shape != (43,H,C) or delta != dict(streams=258,physical_streams=258,physical_calls=11): raise ValueError("Actual full V43 inference geometry or F counters differ")
    archive = private_folder/"V_select.npz"
    with archive.open("xb") as stream: np.savez_compressed(stream,**arrays)
    score,rows = context["evaluation"].error_statistics(arrays["pred_normalized"],arrays["target_normalized"],arrays["observed"],arrays["origins"],spec["run_id"],"V_select",identity["cohort"],arrays["channel_ids"])
    with gzip.open(public_folder/"PUBLIC_ERROR_STATS.csv.gz","xt",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    p1.csv_new(public_folder/"SCORES.csv",[score])
    manifest = dict(**identity,archive_sha256=sha(archive),arrays={key:context["evaluation"].array_fingerprint(value) for key,value in arrays.items()},
        score=score,family=spec["family"],seed=SEED,artifact_sha256=fitted["receipt"]["artifact_sha256"],device="cpu",F_counters=delta,
        lag_scope="Supplied full pre-origin window only; bidirectional is not per-timestamp causal or a causal graph",post_T_development=True,new_T_target_operations=0)
    new(public_folder/"PREDICTION_MANIFEST.json",manifest)
    return manifest


def completed(public_run, private_run, condition):
    receipt = read(public_run/"BASE_FIT_RECEIPT.json"); source = read(public_run/"SOURCE_CONTRACT.json"); artifact = private_run/"selected_bundle.pt"
    if (receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != condition["run_id"] or receipt.get("family") != condition["family"]
            or receipt.get("seed") != SEED or receipt.get("refit_scaler_sha") != SCALER_SHA or source["spec"] != condition["spec"]
            or sha(artifact) != receipt["artifact_sha256"]): raise ValueError("Exact completed new-entry artifact/recipe required")
    return receipt,artifact


def validate_saved(manifest, receipt, condition, context):
    if (manifest.get("run_id") != condition["run_id"] or manifest.get("family") != condition["family"] or manifest.get("seed") != SEED
            or manifest.get("split") != "V_select" or manifest.get("data_sha") != context["task_data_sha"]
            or manifest.get("artifact_sha256") != receipt["artifact_sha256"] or manifest.get("scaler_sha") != receipt["refit_scaler_sha"]
            or manifest.get("source_sha") != fingerprint(context["new_sources"]) or manifest.get("config_sha") != fingerprint(condition["spec"])
            or manifest.get("model_sha") != fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"]))):
        raise ValueError("Accuracy is not the exact selected scientific route")


def saved_arrays(public_folder, private_folder, condition, receipt, context):
    manifest = read(public_folder/"PREDICTION_MANIFEST.json"); reference = read(public_folder/"REFERENCE_CONTRACT.json")
    validate_saved(manifest,receipt,condition,context)
    path = private_folder/"V_select.npz"
    if sha(path) != manifest["archive_sha256"]: raise ValueError("Saved V archive bytes changed")
    with context["np"].load(path,allow_pickle=False) as archive:
        if set(archive.files) != {"pred_normalized","target_normalized","observed","origins","channel_ids"}: raise ValueError("Exact private V members required")
        arrays = {key:archive[key] for key in archive.files}
    for key,value in arrays.items():
        if context["evaluation"].array_fingerprint(value) != manifest["arrays"][key]: raise ValueError("Saved V array fingerprint differs")
    if arrays["pred_normalized"].shape != (43,H,C): raise ValueError("Full original V43/C11/H48 required")
    for original in (reference,context["reference"]): context["priority"].validate_cohort(arrays,original,context["evaluation"])
    if any(reference[key] != manifest[key] for key in ("run_id","split","cohort","data_sha","scaler_sha","source_sha","config_sha","model_sha")): raise ValueError("Saved V reference identity differs")
    return arrays,manifest


def rescore(public_folder, private_folder, condition, receipt, context, output):
    arrays,manifest = saved_arrays(public_folder,private_folder,condition,receipt,context)
    with (public_folder/"SCORES.csv").open(encoding="utf-8",newline="") as stream: scores = list(csv.DictReader(stream))
    if len(scores) != 1 or any(scores[0][key] != manifest[key] for key in ("run_id","split","cohort")): raise ValueError("One exact saved score identity required")
    with gzip.open(public_folder/"PUBLIC_ERROR_STATS.csv.gz","rt",encoding="utf-8",newline="") as stream: rows = list(csv.DictReader(stream))
    result = context["priority"].independent_rescore(arrays,manifest,rows,scores[0],context["np"])
    result.update(run_id=condition["run_id"],artifact_sha256=receipt["artifact_sha256"],manifest_sha256=sha(public_folder/"PREDICTION_MANIFEST.json"),full_V_origins=43)
    new(output/"RESCORE_RECEIPT.json",result)
    return result


def verify_cost_prerequisites(condition, receipt, manifest_sha, proof_paths):
    for mode,status in (("rescore","PASS_INDEPENDENT_FULL_V_FP64"),("replay","PASS_FRESH_CPU6_REPLAY")):
        path,digest = proof_paths.get(mode,(None,None))
        if not path or not digest or sha(path) != digest: raise ValueError("Exact prior per-condition validation RESULT/SHA required before costs")
        value = read(path); science = value.get("scientific_result") or {}
        if (value.get("run_id") != condition["run_id"] or value.get("mode") != mode or value.get("error_class") is not None
                or value.get("postflight_error_class") is not None or value.get("status") != status or science.get("status") != status
                or science.get("artifact_sha256") != receipt["artifact_sha256"] or science.get("manifest_sha256") != manifest_sha):
            raise ValueError("Selected artifact's successful independent rescore and fresh CPU replay are required")
        if mode == "replay" and science.get("executed_now") is not True: raise ValueError("Cached replay cannot count as a fresh CPU checkpoint check")


def cost(fitted, task, pool, context, condition, manifest_sha, output, batch, deadline, quiet_path, quiet_sha, proofs):
    if batch not in (1,8) or not quiet_path or not quiet_sha or sha(quiet_path) != quiet_sha: raise ValueError("B1/B8 with exact ROOT quiet proof required")
    require_quiet(read(quiet_path),time.time()); verify_cost_prerequisites(condition,fitted["receipt"],manifest_sha,proofs)
    import measure_costs
    import psutil
    np,p1 = context["np"],context["priority"]
    origins = context["data"].origins(BOUNDS["V_select"],L,H,H)[:batch]
    physical = np.stack([task["physical"][int(o)-L:int(o)] for o in origins]); observed = np.stack([task["observed"][int(o)-L:int(o)] for o in origins])
    route = context["seed"].actual_route(fitted,task,pool,context)
    class Check:
        def check(self): p1.check_deadline(deadline)
    try:
        rows = measure_costs.measure_balanced([route],physical,observed,warmups=10,timed_trials=20,stage_trials=20,lease=Check())
        if len(rows) != 50 or any((row["F_logical_streams"],row["F_physical_streams"],row["F_calls"]) != (batch*K,batch*K,1) for row in rows): raise ValueError("Original whole-path protocol/F counters differ")
        if sha(quiet_path) != quiet_sha or any(sha(path) != digest for path,digest in proofs.values()):
            raise ValueError("Cost prerequisite evidence changed during measurement")
        memory = route.memory_state(); memory["process_RSS_bytes"] = psutil.Process().memory_info().rss
    finally: route.close()
    trials = output/"COST_TRIALS.csv.gz"; measure_costs._write_trials(trials,rows)
    result = dict(status="COMPLETE_PROTOCOL",run_id=condition["run_id"],device="cpu",CPU_threads=2,batch_size=batch,warmups=10,timed_trials=20,stage_trials=20,
        actual_trial_rows=50,summaries=measure_costs.summarize(rows),memory=memory,trials_sha256=sha(trials),accuracy_manifest_sha256=manifest_sha,
        artifact_sha256=fitted["receipt"]["artifact_sha256"],quiet_proof_sha256=quiet_sha,
        validation_result_sha256={key:value[1] for key,value in proofs.items()},
        physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
        origins=origins.tolist(),F_logical_streams_per_trial=batch*K,F_physical_streams_per_trial=batch*K,F_calls_per_trial=1,
        cached_Z_or_forecast_reads=0,physical_output_and_inverse_dtype="float64",stage_probe_separate_intrusive_scope=True,training_peak_included=False,new_T_target_operations=0)
    new(output/"COST_RECEIPT.json",result)
    return result


def execute(public, cache, asset, plan_sha, identifier, mode, deadline, output, batch=None, quiet_path=None, quiet_sha=None, proofs=None):
    import psutil
    public,cache,asset = map(lambda p:relative(p,"."),(public,cache,asset))
    plan,old,condition,libraries = metadata(public,cache,asset,plan_sha,identifier,psutil)
    p1,seed = libraries["priority"],libraries["seed"]
    if mode not in ("fit","rescore","replay","cost") or not isinstance(deadline,(int,float)) or not time.time() < deadline <= DEADLINE: raise ValueError("Explicit permitted mode/live original deadline required")
    if (mode == "cost" and batch not in (1,8)) or (mode != "cost" and batch is not None): raise ValueError("Only cost takes B1/B8")
    p1.require_environment()
    scope,private = relative(public,NAME),relative(cache,NAME)
    if not private.is_dir(): raise FileNotFoundError("ROOT must prepare the new private namespace")
    output = Path(output).absolute(); output = relative(scope,os.path.relpath(output,scope))
    if output.exists() or not output.is_relative_to(scope/"diagnostics"): raise FileExistsError("New exclusive diagnostic output required")
    if mode == "fit":
        state = existing_state(public,cache,identifier)
        if state["status"] != "UNSTARTED": raise FileExistsError(state["status"]+": preserve/reuse/reconcile exact old paths; no renamed restart")
    seed.resource_preflight(public,cache,plan,psutil)
    output.mkdir(parents=True,exist_ok=False)
    started = time.time(); new(output/"STARTED.json",dict(run_id=identifier,mode=mode,plan_sha256=plan_sha,actual_pid=os.getpid(),actual_create_time=psutil.Process().create_time(),deadline_unix=deadline))
    public_run,private_run = scope/"runs"/identifier,private/"runs"/identifier
    public_prediction,private_prediction = scope/"predictions"/identifier/"V_select",private/"predictions"/identifier
    context,pool,result,failure,post_error = None,None,None,None,None
    try:
        sys.path.insert(0,str(public))
        import numpy as np
        import data_contract as data
        import evaluation
        if Path(data.__file__).resolve() != public/"data_contract.py": raise ValueError("Original scientific import root differs")
        context = dict(np=np,data=data,evaluation=evaluation,priority=p1,seed=seed,task_data_sha=DATA_SHA,plan_sha=plan_sha,
            new_sources=dict(runner=plan["runner_sha256"],**DEPENDENCIES),reference=read(relative(public,plan["V_reference_public_relative_path"])))
        if mode == "rescore":
            receipt,artifact = completed(public_run,private_run,condition)
            result = rescore(public_prediction,private_prediction,condition,receipt,context,output)
        else:
            import torch
            import fit_base as fit
            import training
            import deployment
            import gpu_runtime
            torch.set_num_threads(2)
            if torch.cuda.is_initialized(): raise RuntimeError("Unexpected CUDA initialization")
            context.update(torch=torch,fit=fit,training=training)
            task,access = safe_task(asset,old,libraries,context); new(output/"PREFIX_ACCESS_RECEIPT.json",access)
            pool = deployment.FoundationPool(relative(asset,old["model_asset_relative_path"]),"cpu"); foundation = pool.get(H)
            if (gpu_runtime.state_hash(foundation.model) != old["foundation_tensor_sha256"] or foundation.row_microbatch != 128
                    or foundation.checkpoint_inputs is not False or foundation.instance_norm_source_sha256 != old["instance_norm_source_sha256"]
                    or any(p.requires_grad or p.grad is not None for p in foundation.model.parameters())):
                raise ValueError("Actual frozen CPU foundation identity/runtime differs")
            if mode == "fit":
                new(scope/"claims"/("fit_"+identifier+".json"),dict(run_id=identifier,plan_sha256=plan_sha,created_unix=time.time(),prefix_only_new_entry=True))
                public_run.mkdir(parents=True,exist_ok=False); private_run.mkdir(parents=True,exist_ok=False)
                fitted = fit_and_save(condition,task,foundation,context,public_run,private_run,deadline)
                if "model" not in fitted: result = fitted
                else:
                    manifest = full_prediction(fitted,task,foundation,context,public_prediction,private_prediction,deadline)
                    result = dict(status="COMPLETE_FIT_AND_FULL_V",fit_receipt=fitted["receipt"],V_score=manifest["score"],full_V_completed=True,
                        independent_rescore_pending=True,checkpoint_replay_pending=True)
            else:
                receipt,artifact = completed(public_run,private_run,condition)
                bundle = torch.load(artifact,map_location="cpu",weights_only=False)
                if (bundle["plan_sha256"] != plan_sha or bundle["spec"] != condition["spec"] or p1.state_sha(bundle["model"],data) != receipt["model_state_sha"]
                        or bundle["contract"]["model_sha"] != old["foundation_tensor_sha256"] or bundle["contract"]["source_sha"] != fit._source_contract()[0]):
                    raise ValueError("Actual selected checkpoint/source/foundation identity differs")
                model,norm = fit.create_model(condition["spec"],task,12622,foundation); model.load_state_dict(bundle["model"],strict=True)
                if any(not np.array_equal(norm[key],bundle["normalization"][key]) for key in ("mean","std")): raise ValueError("Selected normalization no longer matches actual prefix")
                fitted = dict(model=model,bundle=bundle,receipt=receipt,artifact=artifact)
                manifest = read(public_prediction/"PREDICTION_MANIFEST.json"); validate_saved(manifest,receipt,condition,context)
                manifest_sha = sha(public_prediction/"PREDICTION_MANIFEST.json")
                if mode == "replay":
                    arrays,_ = saved_arrays(public_prediction,private_prediction,condition,receipt,context)
                    result = seed.replay(fitted,arrays,task,pool,context,output,deadline)
                    result["manifest_sha256"] = manifest_sha
                else: result = cost(fitted,task,pool,context,condition,manifest_sha,output,batch,deadline,quiet_path,quiet_sha,proofs or {})
            foundation.assert_frozen_unchanged(); result["F_final_state_sha256"] = gpu_runtime.state_hash(foundation.model)
        p1.check_deadline(deadline)
    except Exception as error:
        failure = type(error).__name__
        path = private/"failures"/(mode+"_"+identifier+"_"+str(time.time_ns())+".json")
        new(path,dict(error_class=failure,error_message=str(error),traceback=traceback.format_exc()))
        result = dict(status="FAILED_PRESERVED",error_class=failure,private_failure_cache_relative_path=path.relative_to(cache).as_posix())
    finally:
        post_path = None
        try:
            if pool is not None: pool.close()
            metadata(public,cache,asset,plan_sha,identifier,psutil); seed.resource_preflight(public,cache,plan,psutil)
            if context is not None and "torch" in context and context["torch"].cuda.is_initialized(): raise RuntimeError("CPU operation initialized CUDA")
        except Exception as error:
            post_error = type(error).__name__; post_path = private/"failures"/("post_"+mode+"_"+identifier+"_"+str(time.time_ns())+".json")
            new(post_path,dict(error_class=post_error,error_message=str(error),traceback=traceback.format_exc()))
        new(output/"RESULT.json",dict(status="FAILED_PRESERVED" if failure or post_error else (result or {}).get("status","INCOMPLETE"),
            run_id=identifier,mode=mode,plan_sha256=plan_sha,scientific_result=result,error_class=failure,postflight_error_class=post_error,
            postflight_private_failure_cache_relative_path=None if post_path is None else post_path.relative_to(cache).as_posix(),
            elapsed_wall_seconds=time.time()-started,original_source39_immutable9_preserved=post_error is None,new_T_target_operations=0,
            old_full_task_identity_inherited_not_recomputed=True,independent_generalization_claimed=False,
            ROOT_RSS_CPU2_2slot_6GiB_deadline_and_quiet_supervisor_required=True,
            historical_old_probe_T_deserialization_scope_preserved=True,original_GPU_coverage_changed=False))
    if failure or post_error: raise RuntimeError("New Beijing entry failed; original evidence and private traceback preserved")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true"); parser.add_argument("--root-triggered",action="store_true")
    for key in ("public-root","cache-root","asset-root","plan-sha","run-id","output","quiet-proof","quiet-sha","rescore-proof","rescore-sha","replay-proof","replay-sha"):
        parser.add_argument("--"+key)
    parser.add_argument("--mode",choices=("fit","rescore","replay","cost")); parser.add_argument("--deadline",type=float); parser.add_argument("--batch",type=int,choices=(1,8))
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="DRAFT_NOT_EXECUTED",input_reads=0,writes=0,scientific_or_child_operations=0,
            numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,actual_new_training_conditions=0,
            candidate_ids=[row["run_id"] for row in conditions()])))
        return 0
    if not args.root_triggered or any(getattr(args,key) is None for key in ("public_root","cache_root","asset_root","plan_sha","run_id","mode","deadline","output")):
        parser.error("Explicit ROOT trigger, sealed new PLAN SHA, roots, one run ID, mode, deadline and fresh output required")
    result = execute(args.public_root,args.cache_root,args.asset_root,args.plan_sha,args.run_id,args.mode,args.deadline,args.output,args.batch,
        args.quiet_proof,args.quiet_sha,dict(rescore=(args.rescore_proof,args.rescore_sha),replay=(args.replay_proof,args.replay_sha)))
    print(json.dumps(dict(status=result["status"],run_id=args.run_id,mode=args.mode)))
    return 0 if result["status"].startswith(("COMPLETE","PASS")) else 1


if __name__ == "__main__": raise SystemExit(main())
