"""One exact PeMS08 B2 frozen-P/E/D head diagnostic; explicit ROOT execution only."""
import argparse
import ast
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
import zipfile

NAME = "priority_core_followup_20261016_v1"
PRIOR = "priority_b2_validation_20261016_v1"
L, H, C, K, SEED, PREFIX = 512, 12, 170, 43, 92601, 14284
ROLE = "PEMS_FIXED_HEAD"
BOUNDS = dict(A_fit=[0,8928], A_stop=[8928,10713], M_cal=[10713,12499], V_select=[12499,14284], T_final=[14284,17856])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def prefix_payload(stream, member_size, primitive, rows=PREFIX, expected_shape=(17856,170,3)):
    """Decode only a contiguous C-order prefix; never convert protected tail."""
    magic = primitive.read_exact(stream, 8)
    if magic[:6] != b"\x93NUMPY" or magic[6:8] not in (b"\x01\x00",b"\x02\x00",b"\x03\x00"):
        raise ValueError("Supported primitive NPY header required")
    width = 2 if magic[6] == 1 else 4
    count = int.from_bytes(primitive.read_exact(stream, width), "little")
    if not 0 < count <= 65536:
        raise ValueError("Bounded NPY header required")
    header = ast.literal_eval(primitive.read_exact(stream,count).decode("utf-8" if magic[6] == 3 else "latin1").strip())
    if (not isinstance(header,dict) or set(header) != {"descr","fortran_order","shape"}
            or header["shape"] != expected_shape or header["descr"] != "<f8" or header["fortran_order"] is not False
            or type(rows) is not int or not 0 <= rows <= expected_shape[0]):
        raise ValueError("Exact C-order FP64 PeMS source axes and legal prefix required")
    full_bytes = math.prod(expected_shape)*8
    if member_size != 8+width+count+full_bytes:
        raise ValueError("Declared PeMS member size differs")
    selected_bytes = rows*math.prod(expected_shape[1:])*8
    values = primitive.decode_numeric(primitive.read_exact(stream,selected_bytes),"<f8").reshape((rows,)+expected_shape[1:])
    return values, dict(member="data.npy", source_shape=list(expected_shape), fortran_order=False, dtype="float64",
        materialized_source_shape=list(values.shape), numeric_payload_bytes_converted=selected_bytes,
        payload_bytes_left_unread=full_bytes-selected_bytes, opaque_payload_bytes_discarded=0,
        tail_numeric_values_parsed=0, feature_index=0,
        physical_IO_scope="Whole compressed file opaque SHA and ZIP buffering may read tail; tail never decoded numerically")


def load_prefix(cache, old, helper, primitive):
    metadata = old["metadata"]
    source = helper.inside(cache, metadata["source_reference"])
    if (old["shape"] != [17856,C] or old["L"] != L or old["H"] != H
            or old["primary_split_ranges"] != BOUNDS or len(old["columns"]) != C
            or metadata["selected_feature_index"] != 0 or metadata["source_shape"] != [17856,C,3]
            or source.stat().st_size != metadata["source_bytes"] or sha(source) != metadata["source_sha256"]):
        raise ValueError("Original PeMS source/task pins differ")
    import numpy as np
    with zipfile.ZipFile(source) as archive:
        if archive.namelist() != ["data.npy"]:
            raise ValueError("One exact PeMS data member required")
        info = archive.getinfo("data.npy")
        with archive.open(info) as stream:
            raw, member = prefix_payload(stream, info.file_size, primitive)
    physical = raw[:,:,0].astype(np.float32, copy=True)
    observed = np.isfinite(physical)
    physical[~observed] = np.nan
    times = np.arange(PREFIX,dtype=np.int64)*300*1000000000
    hashes = dict(physical=primitive.array_sha(physical), observed=primitive.array_sha(observed), times=primitive.array_sha(times),
        columns=hashlib.sha256(json.dumps(old["columns"],ensure_ascii=False).encode()).hexdigest())
    prefix_sha = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    task = dict(name="pems08", dataset="pems08", physical=physical, observed=observed, times=times,
        columns=copy.deepcopy(old["columns"]),L=L,H=H,interval_seconds=300,period=288,
        data_sha256=old["data_sha256"],array_sha256=copy.deepcopy(old["array_sha256"]),
        data_identity_provenance="INHERITED_NOT_RECOMPUTED",materialized_prefix_sha256=prefix_sha,
        materialized_prefix_array_sha256=hashes, original_total_rows=17856,materialized_rows=PREFIX,
        split_ranges=copy.deepcopy(BOUNDS),secondary_split_ranges=copy.deepcopy(old["secondary_split_ranges"]),
        metadata=copy.deepcopy(metadata))
    if sha(source) != metadata["source_sha256"]:
        raise ValueError("PeMS source changed during prefix materialization")
    return task, dict(status="MATERIALIZED_PRE_T_PREFIX_ONLY", source_cache_relative_path=metadata["source_reference"],
        source_sha256=metadata["source_sha256"],source_bytes=metadata["source_bytes"],source_rows=17856,
        materialized_rows=PREFIX,materialized_bounds=[0,PREFIX],members=[member],new_T_numeric_values_parsed=0,
        inherited_full_task_data_sha256=old["data_sha256"],inherited_full_task_array_sha256=old["array_sha256"],
        inherited_identity_status="INHERITED_NOT_RECOMPUTED",materialized_prefix_sha256=prefix_sha,
        materialized_prefix_array_sha256=hashes,original_primary_split_ranges=BOUNDS,raw_values_published=False)


def fixed_model(adapter, torch):
    class FixedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if (adapter.length,adapter.horizon,adapter.channels,adapter.latent_k) != (L,H,C,K):
                raise ValueError("Exact PeMS parent adapter dimensions required")
            self.adapter = adapter
            for parameter in self.adapter.parameters():
                parameter.requires_grad_(False)
            self.head = torch.nn.Linear(L,H)
            with torch.no_grad():
                self.head.weight.zero_()
                self.head.weight[:,-1] = 1
                self.head.bias.zero_()
            self.adapter.eval()

        def train(self, mode=True):
            super().train(mode)
            self.adapter.eval()
            return self

        def forward(self, x, foundation=None, latent_k=None):
            if foundation is not None or latent_k not in (None,K):
                raise ValueError("Exact K43 and zero foundation required")
            with torch.no_grad():
                z = self.adapter.encode(x,K)
            future = self.head(z.transpose(1,2)).transpose(1,2)
            return self.adapter.decode(x,z,future,K)
    return FixedHead()


def verify_frozen(model, parent, torch):
    actual = model.adapter.state_dict()
    if set(actual) != set(parent["model"]) or any(not torch.equal(actual[key].cpu(),parent["model"][key].cpu()) for key in actual):
        raise ValueError("Frozen exact parent P/E/D changed")
    if any(p.requires_grad or p.grad is not None for p in model.adapter.parameters()):
        raise ValueError("Frozen P/E/D acquired trainability or gradients")
    if tuple(model.head.weight.shape) != (H,L) or model.head.bias.numel() != H:
        raise ValueError("Full-history shared Linear512-to12 head required")


def route_class(deployment):
    class FixedHeadRoute(deployment.Route):
        def __init__(self,model,norm,task,run_id,artifact_sha):
            self.spec = dict(run_id=run_id,family="PRIORITY_NO_BOLT")
            self.task,self.run_id,self.family,self.device = task,run_id,"PRIORITY_NO_BOLT","cpu"
            self.model,self.mean,self.std = model,norm["mean"],norm["std"]
            self.pool,self.owns_pool = deployment.FoundationPool(None,"cpu"),True
            self.g_route = self.b_route = self.bundle = self.ridge_artifact = None
            self._closed,self._stage_mode,self._stages,self.last_receipt = False,False,{},{}
            self.latent_k,self.artifact_sha256 = K,artifact_sha
            self.mean_t = deployment.torch.as_tensor(self.mean,device="cpu")
            self.std_t = deployment.torch.as_tensor(self.std,device="cpu")
            self.model.eval()

        def _predict_normalized(self,x,physical,observed):
            z = self._step("P_E",lambda:self.model.adapter.encode(x,K))
            future = self._step("direct_head",lambda:self.model.head(z.transpose(1,2)).transpose(1,2))
            return self._step("D",lambda:self.model.adapter.decode(x,z,future,K))
    return FixedHeadRoute


def canonical_parent_route(spec,reference,helper):
    parent = spec["parent"]
    return dict(run_id=parent["run_id"],dataset="pems08",family="B2",fold="primary",
        artifact_cache_relative_path=parent["artifact_cache_relative_path"],artifact_sha256=parent["artifact_sha256"],latent_k=K,
        canonical_fit_end=BOUNDS["A_stop"][1],foundation_sha=spec["parent_cpu"]["foundation_tensor_sha256"],
        expected_prediction_model_sha=helper.fingerprint(dict(artifact_sha=parent["artifact_sha256"],model_state_sha=parent["model_state_sha"],normalization=parent["scaler_sha256"])),
        fit_receipt_public_relative_path=parent["fit_receipt_public_relative_path"],fit_receipt_sha256=parent["fit_receipt_sha256"],
        standard_V_binding=dict(model_sha=reference["model_sha"],archive_sha256=reference["archive_sha256"],
            manifest_public_relative_path=parent["V_manifest_public_relative_path"],manifest_sha256=parent["V_manifest_sha256"]),model_size="small")


def operation_receipt_fields(spec,mode,result=None):
    parent_mode = mode in ("parent_replay","parent_cost")
    actual = result or {}
    return dict(role="PARENT_B2" if parent_mode else ROLE,run_id=spec["parent"]["run_id"] if parent_mode else spec["run_id"],
        actual_F_streams=actual.get("actual_F_streams") if parent_mode else 0,
        foundation_instances_created=actual.get("foundation_instances_created",actual.get("foundation",{}).get("foundation_instances_created")) if parent_mode else 0)


def metadata(public,cache,spec_path,spec_sha,process_api):
    spec_path = Path(spec_path).resolve()
    if not spec_sha or not spec_path.is_relative_to(public/NAME) or sha(spec_path) != spec_sha:
        raise ValueError("Explicit ROOT-sealed new SPEC path and SHA required")
    spec = read(spec_path)
    if (spec.get("schema") != "PRIORITY_PEMS_FIXED_HEAD_SPEC_v1" or spec.get("status") != "SEALED_BY_ROOT"
            or spec.get("namespace") != NAME or spec.get("role") != ROLE or spec.get("seed") != SEED
            or spec.get("epochs") != [3.] or spec.get("materialized_prefix_end_exclusive") != PREFIX
            or spec.get("protected_tail_numeric_parse_allowed") is not False
            or spec.get("optimizer") != dict(name="AdamW",lr=.001,weight_decay=0.,effective_batch=16,microbatch=4,gradient_clip=1.)):
        raise ValueError("Exact fixed PeMS head preregistration required")
    if Path(spec["run_id"]).name != spec["run_id"] or spec["run_id"] in (".",".."):
        raise ValueError("Safe one-component run ID required")
    for relative,key in ((NAME+"/pems_fixed_head_runner.py","runner_sha256"),(PRIOR+"/priority_runner.py","helper_runner_sha256"),
                         (PRIOR+"/safe_jena_prefix.py","prefix_primitive_sha256")):
        if sha(public/relative) != spec[key]:
            raise ValueError("Fixed-head source pin changed: "+relative)
    helper = module("pems_priority_helpers",public/PRIOR/"priority_runner.py")
    prior = spec["prior_priority_plan"]
    old_plan = helper.metadata(public,cache,prior["sha256"],process_api,helper.inside(public,prior["public_relative_path"]))
    if spec["deadline_unix"] > old_plan["campaign_deadline_unix"]:
        raise ValueError("Original campaign deadline may not be extended")
    if spec["cost_protocol"] != old_plan["cost_protocol"]:
        raise ValueError("Original cost protocol changed")
    expected_replay = dict(origins=6,microbatch=4,atol=1e-5,rtol=1e-4)
    if spec["replay"] != expected_replay:
        raise ValueError("Preregistered CPU replay changed")
    cpu = spec["parent_cpu"]
    if (cpu["foundation_tensor_sha256"] != old_plan["foundation_tensor_sha256"]
            or cpu["model_asset_relative_path"] != old_plan["model_asset_relative_path"]
            or cpu["model_revision"] != old_plan["model_revision"]
            or cpu["instance_norm_source_sha256"] != old_plan["instance_norm_source_sha256"]
            or cpu["row_microbatch"] != 128 or cpu["checkpoint_inputs"] is not False):
        raise ValueError("Original frozen parent CPU foundation settings changed")
    if sha(public.parent/"temporal_frontend_campaign_20261015_v1/runtime.py") != cpu["instance_norm_source_sha256"]:
        raise ValueError("Actual original worktree SafeInstanceNorm source changed")
    original = read(public/"DATA_CONTRACT.json")["assets"]["pems08"]
    if spec["task_contract"] != original or original["primary_split_ranges"] != BOUNDS:
        raise ValueError("Original immutable PeMS task contract changed")
    parent = spec["parent"]
    for root,pathkey,shakey in ((cache,"artifact_cache_relative_path","artifact_sha256"),
            (public,"fit_receipt_public_relative_path","fit_receipt_sha256"),
            (public,"source_contract_public_relative_path","source_contract_sha256"),
            (public,"V_manifest_public_relative_path","V_manifest_sha256"),
            (public,"old_replay_reference_public_relative_path","old_replay_reference_sha256")):
        if sha(helper.inside(root,parent[pathkey])) != parent[shakey]:
            raise ValueError("Exact conditional parent binding changed")
    fit_receipt = read(helper.inside(public,parent["fit_receipt_public_relative_path"]))
    source = read(helper.inside(public,parent["source_contract_public_relative_path"]))
    manifest = read(helper.inside(public,parent["V_manifest_public_relative_path"]))
    if (fit_receipt["status"] != "COMPLETE_VALID" or fit_receipt["training_status"] != "LIMITED_SCREEN"
            or fit_receipt["selected_epoch"] != 1. or fit_receipt["run_id"] != parent["run_id"]
            or fit_receipt["artifact_sha256"] != parent["artifact_sha256"] or fit_receipt["model_state_sha"] != parent["model_state_sha"]
            or fit_receipt["refit_scaler_sha"] != parent["scaler_sha256"]
            or source["spec"]["family"] != "B2" or source["spec"]["latent_k"] != K or source["spec"]["seed"] != SEED
            or manifest["run_id"] != parent["run_id"] or manifest["split"] != "V_select"
            or manifest["scaler_sha"] != parent["scaler_sha256"] or manifest["data_sha"] != original["data_sha256"]
            or manifest["metadata"]["artifact_sha256"] != parent["artifact_sha256"]
            or manifest["arrays"]["pred_normalized"]["shape"] != [148,H,C]):
        raise ValueError("Parent one-epoch screening/model/normalization/V identity differs")
    old_reference = read(helper.inside(public,parent["old_replay_reference_public_relative_path"]))
    route = canonical_parent_route(spec,manifest,helper)
    if (helper.fingerprint(route) != old_reference["route_sha"] or helper.fingerprint(route) != cpu["canonical_route_sha256"]
            or route["expected_prediction_model_sha"] != manifest["model_sha"]
            or old_reference["standard_manifest_sha256"] != parent["V_manifest_sha256"]
            or old_reference["standard_archive_sha256"] != manifest["archive_sha256"]):
        raise ValueError("Direct canonical parent spec differs from original pre-inference fingerprint")
    return spec,helper,manifest


def context(public,cache,spec,spec_sha,helper,models):
    sys.path.insert(0,str(public))
    import numpy as np
    import evaluation
    import data_contract as data
    if Path(data.__file__).resolve() != public/"data_contract.py":
        raise ValueError("Original modules must come from the calling campaign")
    value = dict(np=np,evaluation=evaluation,data=data,helper=helper,spec_sha=spec_sha,
        sources={"pems_fixed_head_runner.py":spec["runner_sha256"],"priority_runner.py":spec["helper_runner_sha256"],
                 "safe_jena_prefix.py":spec["prefix_primitive_sha256"]})
    if models:
        import torch
        import fit_base as fit
        import training
        import networks
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU route initialized CUDA")
        value.update(torch=torch,fit=fit,training=training,networks=networks,
            primitive=module("pems_safe_prefix_primitives",public/PRIOR/"safe_jena_prefix.py"))
    return value


def load_parent(spec,cache,ctx):
    helper,torch,data,fit = (ctx[key] for key in ("helper","torch","data","fit"))
    parent = torch.load(helper.inside(cache,spec["parent"]["artifact_cache_relative_path"]),map_location="cpu",weights_only=False)
    expected = spec["parent"]
    if (parent["model_state_sha"] != expected["model_state_sha"] or helper.state_sha(parent["model"],data) != expected["model_state_sha"]
            or parent["contract"]["data_sha"] != spec["task_contract"]["data_sha256"]
            or parent["contract"]["source_sha"] != fit._source_contract()[0]
            or fit.scaler_fingerprint(parent["normalization"]["mean"],parent["normalization"]["std"]) != expected["scaler_sha256"]
            or parent["spec"]["family"] != "B2" or parent["spec"]["latent_k"] != K or parent["spec"]["seed"] != SEED
            or parent["normalization"]["train_end"] != BOUNDS["A_stop"][1]):
        raise ValueError("Exact saved PeMS parent tensor/scaler contract differs")
    return parent


def build_model(parent,ctx):
    torch = ctx["torch"]
    torch.manual_seed(SEED)
    norm = parent["normalization"]
    adapter = ctx["networks"].ForecastAdapter("B2",L,H,C,K,norm["pca_center"],norm["basis"],"pointwise",None)
    adapter.load_state_dict(parent["model"],strict=True)
    value = fixed_model(adapter,torch).to("cpu")
    verify_frozen(value,parent,torch)
    return value


def fit_head(spec,task,parent,ctx,public_run,private_run,deadline):
    helper,fit = ctx["helper"],ctx["fit"]
    norm = parent["normalization"]
    model = build_model(parent,ctx)
    train_spec = dict(run_id=spec["run_id"],dataset="pems08",family="PRIORITY_NO_BOLT",latent_k=K,seed=SEED,fold="primary",
        lr=.001,microbatch=4,effective_batch=16,device="cpu",role=ROLE,epochs=[3.],
        head="shared_Linear_512_to_12_persistence_init",followup_spec_sha256=ctx["spec_sha"],new_source_sha256=ctx["sources"])
    source_sha,files = fit._source_contract()
    contract = dict(source_sha=source_sha,config_sha=helper.fingerprint(dict(spec=train_spec,axes=dict(L=L,H=H,C=C,K=K,columns=task["columns"]))),
        data_sha=task["data_sha256"],model_sha="NO_FOUNDATION")
    helper.new(public_run/"SOURCE_CONTRACT.json",dict(**contract,files=files,spec=train_spec))
    stage = fit._stage_body(train_spec,task,BOUNDS["A_stop"][1],None,[3.],private_run/"fixed_refit",public_run/"fixed_refit",
        None,deadline,contract,model,norm)
    helper.write_stage(public_run/"fixed_refit",stage)
    if stage["summary"]["status"] != "SEGMENT_COMPLETE":
        return dict(status="BUDGET_LIMIT_PRESERVED",summary=stage["summary"])
    model.load_state_dict(stage["model"],strict=True)
    verify_frozen(model,parent,ctx["torch"])
    bundle = dict(model=stage["model"],normalization=norm,spec=train_spec,contract=stage["summary"]["contract"],
        model_state_sha=helper.state_sha(stage["model"],ctx["data"]),selected_epoch=3.,refit_summary=stage["summary"],
        parent_artifact_sha256=spec["parent"]["artifact_sha256"],followup_spec_sha256=ctx["spec_sha"])
    artifact = private_run/"selected_bundle.pt"
    helper.save_bundle(artifact,bundle,ctx["torch"])
    receipt = dict(status="COMPLETE_FIXED_PEMS_HEAD_FIT",role=ROLE,run_id=spec["run_id"],artifact_sha256=sha(artifact),
        model_state_sha=bundle["model_state_sha"],scaler_sha=spec["parent"]["scaler_sha256"],selected_epoch=3.,refit_summary=stage["summary"],
        head_parameters=H*(L+1),full_latent_history=L,latent_k=K,exact_parent_PED_frozen=True,
        parent_artifact_sha256=spec["parent"]["artifact_sha256"],parent_training_status="LIMITED_SCREEN_ONE_SELECTED_EPOCH",
        A_stop_unseen_head_selection_claimed=False,new_M_target_training=False,original_B2_retrained=False,
        training_convergence_claimed=False,actual_F_streams=0,foundation_instances_created=0,protected_T_target_operations=0)
    helper.new(public_run/"BASE_FIT_RECEIPT.json",receipt)
    return dict(status="COMPLETE_FIXED_PEMS_HEAD_FIT",model=model,bundle=bundle,artifact=artifact,receipt=receipt)


def cohort(task,norm,reference,ctx):
    np,data = ctx["np"],ctx["data"]
    origins = data.origins(BOUNDS["V_select"],L,H,H)
    if len(origins) != 148 or int(origins[-1])+H > PREFIX:
        raise ValueError("Complete original V148 required")
    target = [data.target_window(task,int(o),norm["mean"],norm["std"]) for o in origins]
    arrays = dict(target_normalized=np.stack([x[0] for x in target]),observed=np.stack([x[1] for x in target]),
        origins=origins,channel_ids=np.asarray(task["columns"],dtype=str))
    ctx["helper"].validate_cohort(arrays,reference,ctx["evaluation"])
    return arrays


def save_prediction(spec,fitted,task,reference,ctx,public_prediction,private_prediction,deadline):
    helper,np,evaluation = ctx["helper"],ctx["np"],ctx["evaluation"]
    norm = fitted["bundle"]["normalization"]
    arrays = cohort(task,norm,reference,ctx)
    identity = dict(run_id=spec["run_id"],split="V_select",cohort="pems08:primary:V_select:stride12",data_sha=task["data_sha256"],
        scaler_sha=spec["parent"]["scaler_sha256"],source_sha=helper.fingerprint(ctx["sources"]),
        config_sha=helper.fingerprint(fitted["bundle"]["spec"]),model_sha=helper.fingerprint(dict(artifact_sha=fitted["receipt"]["artifact_sha256"],model_state_sha=fitted["bundle"]["model_state_sha"])))
    public_prediction.mkdir(parents=True,exist_ok=False)
    private_prediction.mkdir(parents=True,exist_ok=False)
    helper.new(public_prediction/"REFERENCE_CONTRACT.json",dict(**identity,arrays={key:evaluation.array_fingerprint(value) for key,value in arrays.items()},expected_F_streams=0))
    windows = ctx["training"].FastWindows(task,norm["mean"],norm["std"],"cpu")
    output = []
    fitted["model"].eval()
    with ctx["torch"].no_grad():
        for start in range(0,len(arrays["origins"]),4):
            helper.check_deadline(deadline)
            x,_,_ = windows.get(arrays["origins"][start:start+4])
            prediction = fitted["model"](x,None,K)
            if tuple(prediction.shape) != (len(x),H,C) or not ctx["torch"].isfinite(prediction).all():
                raise ValueError("PeMS original-channel output invalid")
            output.append(prediction.cpu().numpy().astype(np.float32))
    arrays["pred_normalized"] = np.concatenate(output)
    path = private_prediction/"V_select.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle,**arrays)
    score,statistics = evaluation.error_statistics(arrays["pred_normalized"],arrays["target_normalized"],arrays["observed"],arrays["origins"],
        spec["run_id"],"V_select",identity["cohort"],arrays["channel_ids"])
    with gzip.open(public_prediction/"PUBLIC_ERROR_STATS.csv.gz","xt",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(statistics[0]));writer.writeheader();writer.writerows(statistics)
    helper.csv_new(public_prediction/"SCORES.csv",[score])
    manifest = dict(**identity,archive_sha256=sha(path),arrays={key:evaluation.array_fingerprint(value) for key,value in arrays.items()},score=score,
        role=ROLE,artifact_sha256=fitted["receipt"]["artifact_sha256"],followup_spec_sha256=ctx["spec_sha"],device="cpu",actual_F_streams=0,
        original_full_L_history_used=True,same_lag_D_current_anchor=True,post_T_development=True,new_T_target_operations=0,
        conditional_on_limited_screen_parent=True,independent_generalization_claimed=False)
    helper.new(public_prediction/"PREDICTION_MANIFEST.json",manifest)
    return manifest


def load_prediction(public_prediction,private_prediction,reference,ctx,spec,receipt):
    if (receipt["status"] != "COMPLETE_FIXED_PEMS_HEAD_FIT" or receipt["run_id"] != spec["run_id"] or receipt["role"] != ROLE
            or receipt["selected_epoch"] != 3. or receipt["scaler_sha"] != spec["parent"]["scaler_sha256"]
            or receipt["parent_artifact_sha256"] != spec["parent"]["artifact_sha256"] or receipt["exact_parent_PED_frozen"] is not True):
        raise ValueError("Completed fixed head/PED/parent binding differs")
    manifest = read(public_prediction/"PREDICTION_MANIFEST.json")
    path = private_prediction/"V_select.npz"
    if sha(path) != manifest["archive_sha256"]:
        raise ValueError("Saved V archive changed")
    with ctx["np"].load(path,allow_pickle=False) as archive:
        if set(archive.files) != {"pred_normalized","target_normalized","observed","origins","channel_ids"}:
            raise ValueError("Saved V archive members differ")
        arrays = {key:archive[key] for key in archive.files}
    for key,value in arrays.items():
        if ctx["evaluation"].array_fingerprint(value) != manifest["arrays"][key]:
            raise ValueError("Saved V fingerprint changed")
    ctx["helper"].validate_cohort(arrays,reference,ctx["evaluation"])
    saved_reference = read(public_prediction/"REFERENCE_CONTRACT.json")
    ctx["helper"].validate_cohort(arrays,saved_reference,ctx["evaluation"])
    source = read(public_prediction.parents[2]/"runs"/spec["run_id"]/"SOURCE_CONTRACT.json")
    expected_model = ctx["helper"].fingerprint(dict(artifact_sha=receipt["artifact_sha256"],model_state_sha=receipt["model_state_sha"]))
    if (arrays["pred_normalized"].shape != (148,H,C) or manifest["run_id"] != spec["run_id"] or manifest["role"] != ROLE
            or manifest["artifact_sha256"] != receipt["artifact_sha256"] or manifest["model_sha"] != expected_model
            or manifest["split"] != "V_select" or manifest["cohort"] != "pems08:primary:V_select:stride12"
            or manifest["config_sha"] != ctx["helper"].fingerprint(source["spec"])
            or manifest["source_sha"] != ctx["helper"].fingerprint(ctx["sources"]) or manifest["followup_spec_sha256"] != ctx["spec_sha"]
            or manifest["data_sha"] != spec["task_contract"]["data_sha256"] or manifest["scaler_sha"] != spec["parent"]["scaler_sha256"]
            or any(saved_reference[key] != manifest[key] for key in ("run_id","split","cohort","data_sha","scaler_sha","source_sha","config_sha","model_sha"))):
        raise ValueError("Saved accuracy artifact/source/spec/normalization binding differs")
    return arrays,manifest


def load_selected(spec,parent,ctx,public_run,private_run):
    receipt = read(public_run/"BASE_FIT_RECEIPT.json")
    artifact = private_run/"selected_bundle.pt"
    if receipt["status"] != "COMPLETE_FIXED_PEMS_HEAD_FIT" or receipt["run_id"] != spec["run_id"] or sha(artifact) != receipt["artifact_sha256"]:
        raise ValueError("Exact completed fixed-head artifact required")
    bundle = ctx["torch"].load(artifact,map_location="cpu",weights_only=False)
    source = read(public_run/"SOURCE_CONTRACT.json")
    if (ctx["helper"].state_sha(bundle["model"],ctx["data"]) != receipt["model_state_sha"]
            or bundle["model_state_sha"] != receipt["model_state_sha"] or bundle["spec"] != source["spec"]
            or bundle["followup_spec_sha256"] != ctx["spec_sha"] or bundle["parent_artifact_sha256"] != spec["parent"]["artifact_sha256"]
            or any(bundle["contract"][key] != source[key] for key in ("source_sha","config_sha","data_sha","model_sha"))
            or ctx["fit"].scaler_fingerprint(bundle["normalization"]["mean"],bundle["normalization"]["std"]) != spec["parent"]["scaler_sha256"]):
        raise ValueError("Saved model/scaler/source/spec binding differs")
    model = build_model(parent,ctx)
    model.load_state_dict(bundle["model"],strict=True)
    model.eval();verify_frozen(model,parent,ctx["torch"])
    return dict(model=model,bundle=bundle,receipt=receipt,artifact=artifact)


def replay(fitted,arrays,manifest,task,ctx,output,deadline):
    import deployment
    helper,np = ctx["helper"],ctx["np"]
    indices = helper.replay_indices(len(arrays["origins"]),np)
    selected = arrays["origins"][indices]
    norm = fitted["bundle"]["normalization"]
    for index,origin in zip(indices,selected):
        target,mask = ctx["data"].target_window(task,int(origin),norm["mean"],norm["std"])
        if not np.array_equal(target,arrays["target_normalized"][index]) or not np.array_equal(mask,arrays["observed"][index]):
            raise ValueError("Replay target/mask differs from saved V")
    route = route_class(deployment)(fitted["model"],norm,task,manifest["run_id"],fitted["receipt"]["artifact_sha256"])
    outputs = []
    try:
        for start in range(0,len(selected),4):
            helper.check_deadline(deadline)
            physical = np.stack([task["physical"][int(o)-L:int(o)] for o in selected[start:start+4]])
            observed = np.stack([task["observed"][int(o)-L:int(o)] for o in selected[start:start+4]])
            raw = route.predict_physical(physical,observed)
            outputs.append(((raw-norm["mean"].astype(np.float64))/norm["std"].astype(np.float64)).astype(np.float32))
        actual = np.concatenate(outputs);expected = arrays["pred_normalized"][indices]
        passed = bool(np.allclose(actual,expected,atol=1e-5,rtol=1e-4))
        if route.pool.instances_created or any(route.pool.counters().values()):
            raise ValueError("No-Bolt replay invoked foundation")
        receipt = dict(status="PASS_FRESH_CPU6_REPLAY" if passed else "FAIL_REPLAY_PARITY",origins=selected.tolist(),actual_origins=len(selected),
            microbatches=[4,2],max_abs=float(np.abs(actual.astype(np.float64)-expected.astype(np.float64)).max()),atol=1e-5,rtol=1e-4,
            artifact_sha256=fitted["receipt"]["artifact_sha256"],full_L_history=L,latent_k=K,same_lag_D_current_anchor=True,
            actual_F_streams=0,foundation_instances_created=0,new_T_target_operations=0,executed_now=True)
    finally:
        route.close()
    helper.new(output/"REPLAY_RECEIPT.json",receipt)
    if not passed:
        raise ValueError("Checkpoint replay differs")
    return receipt


def cost(fitted,task,batch,ctx,output,deadline):
    import deployment
    import measure_costs
    import psutil
    helper,np = ctx["helper"],ctx["np"]
    origins = ctx["data"].origins(BOUNDS["V_select"],L,H,H)[:batch]
    if batch not in (1,8) or len(origins) != batch:
        raise ValueError("Exact first V batch1/8 required")
    physical = np.stack([task["physical"][int(o)-L:int(o)] for o in origins])
    observed = np.stack([task["observed"][int(o)-L:int(o)] for o in origins])
    route = route_class(deployment)(fitted["model"],fitted["bundle"]["normalization"],task,fitted["receipt"]["run_id"],fitted["receipt"]["artifact_sha256"])
    class Deadline:
        def check(self):
            helper.check_deadline(deadline)
    try:
        rows = measure_costs.measure_balanced([route],physical,observed,warmups=10,timed_trials=20,stage_trials=20,lease=Deadline())
        memory = route.memory_state();process = psutil.Process().memory_info()
        memory.update(process_RSS_bytes=process.rss,OS_peak_working_set_bytes=getattr(process,"peak_wset",None))
        if route.pool.instances_created or any(route.pool.counters().values()) or any(row["F_logical_streams"] or row["F_physical_streams"] or row["F_calls"] for row in rows):
            raise ValueError("No-Bolt cost invoked foundation")
    finally:
        route.close()
    path = output/"COST_TRIALS.csv.gz"
    measure_costs._write_trials(path,rows)
    receipt = dict(status="COMPLETE_PROTOCOL",run_id=fitted["receipt"]["run_id"],role=ROLE,device="cpu",CPU_threads=2,batch_size=batch,
        origins=list(map(int,origins)),warmups=10,timed_trials=20,stage_trials=20,physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
        observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),physical_output_dtype="float64",inverse_transform_dtype="float64",
        artifact_sha256=fitted["receipt"]["artifact_sha256"],summaries=measure_costs.summarize(rows),memory=memory,actual_F_streams=0,
        foundation_instances_created=0,cached_Z_or_forecast_reads=0,trials_sha256=sha(path),actual_trial_rows=len(rows),
        stage_probe_is_intrusive_separate_scope=True,training_peak_included=False,protected_T_target_operations=0,
        peak_process_RSS_guard_and_training_quiet_required_from_ROOT=True)
    helper.new(output/"COST_RECEIPT.json",receipt)
    return receipt


def parent_foundation(route,spec,ctx):
    foundation = route.pool.foundation
    cpu = spec["parent_cpu"]
    if (foundation is None or route.pool.instances_created != 1 or foundation.device != "cpu"
            or foundation.horizon != H or foundation.row_microbatch != 128 or foundation.checkpoint_inputs is not False
            or foundation.initial_hash != cpu["foundation_tensor_sha256"]
            or foundation.instance_norm_source_sha256 != cpu["instance_norm_source_sha256"]
            or ctx["fit"]._model_sha(foundation) != cpu["foundation_tensor_sha256"]
            or any(p.requires_grad or p.grad is not None for p in foundation.model.parameters())):
        raise ValueError("Exact frozen CPU small/row128/no-checkpoint parent foundation required")
    foundation.assert_frozen_unchanged()
    return dict(foundation_tensor_sha256=foundation.initial_hash,model_revision=foundation.model_revision,
        instance_norm_source_sha256=foundation.instance_norm_source_sha256,device="cpu",row_microbatch=128,
        checkpoint_inputs=False,foundation_instances_created=1,backbone_parameters_frozen=True,backbone_gradients_absent=True)


def load_parent_arrays(reference,cache,ctx):
    path = ctx["helper"].inside(cache,reference["cache_relative_path"])
    if sha(path) != reference["archive_sha256"]:
        raise ValueError("Original parent V archive bytes changed")
    with ctx["np"].load(path,allow_pickle=False) as archive:
        if set(archive.files) != {"pred_normalized","target_normalized","observed","origins","channel_ids"}:
            raise ValueError("Original parent V members differ")
        arrays = {key:archive[key] for key in archive.files}
    for key,value in arrays.items():
        if ctx["evaluation"].array_fingerprint(value) != reference["arrays"][key]:
            raise ValueError("Original parent V array fingerprint differs")
    if arrays["pred_normalized"].shape != (148,H,C):
        raise ValueError("Original complete parent V148/H12/C170 required")
    return arrays


def parent_replay_gate(path,digest,public,spec,ctx):
    path = Path(path).resolve()
    if not digest or not path.is_relative_to(public/NAME) or sha(path) != digest:
        raise ValueError("Explicit exact new parent CPU replay receipt SHA required before costs")
    receipt = read(path)
    if (receipt["status"] != "PASS_PARENT_FRESH_CPU6_REPLAY" or receipt["device"] != "cpu"
            or receipt["followup_spec_sha256"] != ctx["spec_sha"] or receipt["actual_origins"] != 6
            or receipt["microbatches"] != [4,2] or receipt["atol"] != 1e-5 or receipt["rtol"] != 1e-4
            or receipt["parent_artifact_sha256"] != spec["parent"]["artifact_sha256"]
            or receipt["canonical_route_sha256"] != spec["parent_cpu"]["canonical_route_sha256"]
            or receipt["standard_manifest_sha256"] != spec["parent"]["V_manifest_sha256"]
            or receipt["foundation"]["foundation_tensor_sha256"] != spec["parent_cpu"]["foundation_tensor_sha256"]
            or receipt["new_T_target_operations"] != 0):
        raise ValueError("Exact completed parent CPU replay prerequisite differs")
    return digest


def parent_inference(spec,task,parent,reference,ctx,public,cache,asset,mode,batch,output,deadline,replay_path=None,replay_sha=None):
    import deployment
    import measure_costs
    import psutil
    helper,np = ctx["helper"],ctx["np"]
    if mode == "parent_cost":
        parent_replay_gate(replay_path,replay_sha,public,spec,ctx)
    model_path = helper.inside(asset,spec["parent_cpu"]["model_asset_relative_path"])
    if not model_path.is_dir() or model_path.name != spec["parent_cpu"]["model_revision"]:
        raise ValueError("Exact pinned local parent foundation directory absent")
    canonical = canonical_parent_route(spec,reference,helper)
    route = deployment.Route(canonical,task,cache,model_path,device="cpu")
    try:
        foundation = parent_foundation(route,spec,ctx)
        if (route.prediction_model_sha != reference["model_sha"] or route.artifact_sha256 != spec["parent"]["artifact_sha256"]
                or helper.state_sha(route.model.state_dict(),ctx["data"]) != spec["parent"]["model_state_sha"]
                or ctx["fit"].scaler_fingerprint(route.mean,route.std) != spec["parent"]["scaler_sha256"]
                or any(route.pool.counters().values())):
            raise ValueError("Fresh canonical CPU Route model/scaler/counters differ")
        if mode == "parent_replay":
            arrays = load_parent_arrays(reference,cache,ctx)
            current = cohort(task,parent["normalization"],reference,ctx)
            helper.validate_cohort(current,reference,ctx["evaluation"])
            indices = helper.replay_indices(len(arrays["origins"]),np)
            origins = arrays["origins"][indices]
            old = read(helper.inside(public,spec["parent"]["old_replay_reference_public_relative_path"]))
            if origins.tolist() != old["origins"]:
                raise ValueError("Exact original six replay origins required")
            helper.new(output/"PARENT_PRE_INFERENCE_REFERENCE.json",dict(run_id=spec["parent"]["run_id"],device="cpu",
                canonical_route_sha256=helper.fingerprint(canonical),followup_spec_sha256=ctx["spec_sha"],origins=origins.tolist(),
                arrays={key:ctx["evaluation"].array_fingerprint(value[indices] if key != "channel_ids" else value)
                    for key,value in current.items()},parent_artifact_sha256=spec["parent"]["artifact_sha256"],
                created_before_inference=True,expected_F_streams=6*K))
            predictions = []
            for start in range(0,len(origins),4):
                helper.check_deadline(deadline)
                physical = np.stack([task["physical"][int(o)-L:int(o)] for o in origins[start:start+4]])
                observed = np.stack([task["observed"][int(o)-L:int(o)] for o in origins[start:start+4]])
                raw = route.predict_physical(physical,observed)
                predictions.append(((raw-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32))
            actual = np.concatenate(predictions);expected = arrays["pred_normalized"][indices]
            passed = bool(np.allclose(actual,expected,atol=1e-5,rtol=1e-4))
            counters = route.pool.counters()
            if counters != dict(streams=6*K,row_calls=3,physical_streams=6*K,physical_calls=3):
                raise ValueError("Parent CPU6 logical/physical/call accounting differs")
            parent_foundation(route,spec,ctx)
            if helper.state_sha(route.model.state_dict(),ctx["data"]) != spec["parent"]["model_state_sha"]:
                raise ValueError("Parent adapter weights changed during CPU replay")
            result = dict(status="PASS_PARENT_FRESH_CPU6_REPLAY" if passed else "FAIL_PARENT_CPU_REPLAY_PARITY",device="cpu",origins=origins.tolist(),
                actual_origins=6,microbatches=[4,2],max_abs=float(np.abs(actual.astype(np.float64)-expected.astype(np.float64)).max()),
                atol=1e-5,rtol=1e-4,parent_artifact_sha256=spec["parent"]["artifact_sha256"],canonical_route_sha256=helper.fingerprint(canonical),
                standard_manifest_sha256=spec["parent"]["V_manifest_sha256"],standard_archive_sha256=reference["archive_sha256"],
                followup_spec_sha256=ctx["spec_sha"],foundation=foundation,actual_F_streams=counters["streams"],F_physical_streams=counters["physical_streams"],
                F_calls=counters["physical_calls"],new_T_target_operations=0,parent_retrained=False,
                full_CPU_V_rescoring_performed=False,scope="Fresh CPU6 parity against stored original V; no full CPU148 accuracy claim")
            helper.new(output/"PARENT_REPLAY_RECEIPT.json",result)
            if not passed:
                raise ValueError("Original parent CPU6 parity failed; parent costs remain blocked")
            return result
        origins = ctx["data"].origins(BOUNDS["V_select"],L,H,H)[:batch]
        physical = np.stack([task["physical"][int(o)-L:int(o)] for o in origins])
        observed = np.stack([task["observed"][int(o)-L:int(o)] for o in origins])
        class Deadline:
            def check(self):
                helper.check_deadline(deadline)
        rows = measure_costs.measure_balanced([route],physical,observed,warmups=10,timed_trials=20,stage_trials=20,lease=Deadline())
        expected_calls = math.ceil(batch*K/128)
        if (len(rows) != 50 or any(row["F_logical_streams"] != batch*K or row["F_physical_streams"] != batch*K
                or row["F_calls"] != expected_calls for row in rows)
                or route.pool.counters() != dict(streams=50*batch*K,row_calls=50*expected_calls,physical_streams=50*batch*K,physical_calls=50*expected_calls)):
            raise ValueError("Parent cost50 logical/physical/call accounting differs")
        parent_foundation(route,spec,ctx)
        if helper.state_sha(route.model.state_dict(),ctx["data"]) != spec["parent"]["model_state_sha"]:
            raise ValueError("Parent adapter weights changed during CPU costs")
        memory = route.memory_state();process = psutil.Process().memory_info()
        memory.update(process_RSS_bytes=process.rss,OS_peak_working_set_bytes=getattr(process,"peak_wset",None))
        path = output/"COST_TRIALS.csv.gz"
        measure_costs._write_trials(path,rows)
        result = dict(status="COMPLETE_PARENT_CPU_COST_PROTOCOL",run_id=spec["parent"]["run_id"],role="PARENT_B2",device="cpu",CPU_threads=2,
            batch_size=batch,origins=list(map(int,origins)),warmups=10,timed_trials=20,stage_trials=20,
            physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
            physical_output_dtype="float64",inverse_transform_dtype="float64",parent_artifact_sha256=spec["parent"]["artifact_sha256"],
            canonical_route_sha256=helper.fingerprint(canonical),followup_spec_sha256=ctx["spec_sha"],parent_CPU_replay_receipt_sha256=replay_sha,
            standard_manifest_sha256=spec["parent"]["V_manifest_sha256"],foundation=foundation,summaries=measure_costs.summarize(rows),memory=memory,
            actual_F_streams=50*batch*K,foundation_instances_created=1,cached_Z_or_forecast_reads=0,trials_sha256=sha(path),actual_trial_rows=50,
            stage_probe_is_intrusive_separate_scope=True,training_peak_included=False,protected_T_target_operations=0,parent_retrained=False,
            full_CPU_V_rescoring_performed=False,peak_process_RSS_guard_and_training_quiet_required_from_ROOT=True)
        helper.new(output/"COST_RECEIPT.json",result)
        return result
    finally:
        route.close()


def execute(public,cache,spec_path,spec_sha,mode,output,deadline,batch=None,asset=None,replay_path=None,replay_sha=None):
    import psutil
    public,cache,output = map(lambda path:Path(path).resolve(),(public,cache,output))
    spec,helper,reference = metadata(public,cache,spec_path,spec_sha,psutil)
    if mode not in ("fit","rescore","replay","cost","parent_replay","parent_cost") or deadline is None or not time.time() < deadline <= spec["deadline_unix"]:
        raise ValueError("Explicit supported mode and live sealed deadline required")
    if (mode in ("cost","parent_cost") and batch not in (1,8)) or (mode not in ("cost","parent_cost") and batch is not None):
        raise ValueError("Batch1/8 only for explicit cost mode")
    if mode.startswith("parent_") and not asset:
        raise ValueError("Explicit asset root required for original frozen CPU parent")
    if (mode == "parent_cost" and not all((replay_path,replay_sha))) or (mode != "parent_cost" and any((replay_path,replay_sha))):
        raise ValueError("Parent cost alone requires exact preceding parent CPU replay path/SHA")
    helper.require_environment()
    root = public/NAME
    if output.exists() or output == root or not output.is_relative_to(root):
        raise ValueError("Fresh diagnostic output inside new namespace required")
    public_run,private_run = root/"runs"/spec["run_id"],cache/NAME/"runs"/spec["run_id"]
    public_prediction,private_prediction = root/"predictions"/spec["run_id"]/"V_select",cache/NAME/"predictions"/spec["run_id"]
    if mode == "fit" and (public_run.exists() or private_run.exists()):
        raise FileExistsError("Existing attempt must not be resumed/overwritten")
    output.mkdir(parents=True,exist_ok=False)
    started = time.time();cpu = psutil.Process().cpu_times();result = None;failed = None;ctx = None
    identity = operation_receipt_fields(spec,mode)
    helper.new(output/"STARTED.json",dict(role=identity["role"],mode=mode,run_id=identity["run_id"],spec_sha256=spec_sha,actual_pid=os.getpid(),
        actual_create_time=psutil.Process().create_time(),deadline_unix=deadline,materialized_prefix_end_exclusive=PREFIX,new_T_target_operations=0))
    try:
        ctx = context(public,cache,spec,spec_sha,helper,mode != "rescore")
        if mode == "rescore":
            receipt = read(public_run/"BASE_FIT_RECEIPT.json")
            if sha(private_run/"selected_bundle.pt") != receipt["artifact_sha256"]:
                raise ValueError("Saved checkpoint bytes changed")
            arrays,manifest = load_prediction(public_prediction,private_prediction,reference,ctx,spec,receipt)
            with (public_prediction/"SCORES.csv").open(encoding="utf-8",newline="") as handle:
                scores = list(csv.DictReader(handle))
            with gzip.open(public_prediction/"PUBLIC_ERROR_STATS.csv.gz","rt",encoding="utf-8",newline="") as handle:
                stats = list(csv.DictReader(handle))
            if len(scores) != 1:
                raise ValueError("Exactly one public score required")
            result = helper.independent_rescore(arrays,manifest,stats,scores[0],ctx["np"])
            helper.new(output/"RESCORE_RECEIPT.json",result)
        else:
            task,access = load_prefix(cache,spec["task_contract"],helper,ctx["primitive"])
            helper.new(output/"PREFIX_ACCESS_RECEIPT.json",access)
            mean,std,count = ctx["data"].fit_statistics(task,BOUNDS["A_fit"][1])
            if {key:ctx["data"].array_sha256(value) for key,value in (("mean",mean),("std",std),("count",count))} != spec["task_contract"]["A_fit_statistics_fingerprints"]:
                raise ValueError("Actual A50 prefix statistics differ")
            mean,std,_ = ctx["data"].fit_statistics(task,BOUNDS["A_stop"][1])
            if ctx["fit"].scaler_fingerprint(mean,std) != spec["parent"]["scaler_sha256"]:
                raise ValueError("Actual A60 prefix scaler differs")
            parent = load_parent(spec,cache,ctx)
            cohort(task,parent["normalization"],reference,ctx)
            if mode.startswith("parent_"):
                result = parent_inference(spec,task,parent,reference,ctx,public,cache,Path(asset).resolve(),mode,batch,output,deadline,replay_path,replay_sha)
            elif mode == "fit":
                public_run.mkdir(parents=True,exist_ok=False);private_run.mkdir(parents=True,exist_ok=False)
                fitted = fit_head(spec,task,parent,ctx,public_run,private_run,deadline)
                if "model" not in fitted:
                    result = fitted
                else:
                    manifest = save_prediction(spec,fitted,task,reference,ctx,public_prediction,private_prediction,deadline)
                    result = dict(status="COMPLETE_FIT_AND_FULL_V",fit_receipt=fitted["receipt"],V_score=manifest["score"],full_V_completed=True,
                        independent_rescore_pending=True,checkpoint_replay_pending=True)
            else:
                fitted = load_selected(spec,parent,ctx,public_run,private_run)
                arrays,manifest = load_prediction(public_prediction,private_prediction,reference,ctx,spec,fitted["receipt"])
                if mode == "replay":
                    result = replay(fitted,arrays,manifest,task,ctx,output,deadline)
                else:
                    result = cost(fitted,task,batch,ctx,output,deadline)
                    result["accuracy_manifest_sha256"] = sha(public_prediction/"PREDICTION_MANIFEST.json")
        helper.check_deadline(deadline)
    except BaseException as error:
        failed = type(error).__name__
        raise
    finally:
        metadata(public,cache,spec_path,spec_sha,psutil)
        after = psutil.Process().cpu_times();memory = psutil.Process().memory_info()
        helper.new(output/"RESULT.json",dict(status="FAILED_PRESERVED" if failed else (result or {}).get("status","INCOMPLETE"),mode=mode,
            **operation_receipt_fields(spec,mode,result),error_class=failed,scientific_result=result,elapsed_process_wall_seconds=time.time()-started,
            actual_CPU_process_seconds=after.user+after.system-cpu.user-cpu.system,RSS_bytes=memory.rss,OS_peak_working_set_bytes=getattr(memory,"peak_wset",None),
            new_T_target_operations=0,original_source39_and_immutable9_preserved=True,
            checkpoint_on_exception_supported=False,original_stage_checkpoint_protocol_retained=True,
            old_full_task_fingerprints_inherited_not_recomputed=True,conditional_on_limited_screen_parent=True,independent_generalization_claimed=False))
        if ctx is not None and mode != "rescore" and ctx["torch"].cuda.is_initialized():
            raise RuntimeError("CPU route initialized CUDA")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    for key in ("public-root","cache-root","asset-root","spec","spec-sha","output","parent-replay-receipt","parent-replay-sha"):
        parser.add_argument("--"+key)
    parser.add_argument("--mode",choices=("fit","rescore","replay","cost","parent_replay","parent_cost"))
    parser.add_argument("--deadline",type=float)
    parser.add_argument("--batch",type=int,choices=(1,8))
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="METADATA_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,torch_imported="torch" in sys.modules,
            numpy_imported="numpy" in sys.modules,actual_Task_checkpoint_forward_training_cost_or_child_operations=0)))
        return 0
    if not all((args.public_root,args.cache_root,args.spec,args.spec_sha,args.mode,args.output,args.deadline)):
        parser.error("Explicit ROOT-sealed new SPEC/SHA, roots, mode, fresh output and deadline required")
    result = execute(args.public_root,args.cache_root,args.spec,args.spec_sha,args.mode,args.output,args.deadline,args.batch,
        args.asset_root,args.parent_replay_receipt,args.parent_replay_sha)
    print(json.dumps(dict(status=result["status"],role="PARENT_B2" if args.mode.startswith("parent_") else ROLE,mode=args.mode)))
    return 0 if result["status"].startswith(("COMPLETE","PASS")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
