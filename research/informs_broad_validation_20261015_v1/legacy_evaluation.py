"""Saved-weight E0/B replay and independently implemented FP64 rescoring.

Import and inventory do not load datasets, a foundation model, or CUDA. Actual
evaluation is an explicit fresh-process operation under the central GPU lease.
Legacy B is development data; it is never registered as new primary V or T.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np

import legacy_continuation as lc
from resources import BudgetExceeded, ResourceBusy, validate_gpu_child

KEYS = ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")
SPLIT = "legacy_B_development"


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False).encode()).hexdigest()


def array_fingerprint(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _new_or_same(path, value):
    path = Path(path)
    if path.exists():
        if lc.read_json(path) != value:
            raise ValueError("Existing immutable E0 evaluation metadata differs")
    else:
        lc.write_json(path,value)


def _check(lease, deadline, device):
    if str(device).startswith("cuda") and lease is None:
        raise RuntimeError("E0 GPU evaluation requires the central authenticated live lease")
    if lease is not None:
        lc.check_work_boundary(lease)
    if deadline is not None and time.time() >= deadline:
        raise BudgetExceeded("E0 evaluation deadline reached")


def score_fp64(prediction, target, observed, origins, channel_ids):
    """Independent of the old runtime score and new campaign evaluation module."""
    p, y, m = np.asarray(prediction), np.asarray(target), np.asarray(observed)
    o, c = np.asarray(origins), np.asarray(channel_ids)
    if p.ndim != 3 or p.shape != y.shape or p.shape != m.shape or min(p.shape)<1 or m.dtype != np.bool_:
        raise ValueError("Matching positive [origin,horizon,channel] arrays and original bool mask required")
    if o.shape != (p.shape[0],) or o.dtype.kind not in "iu" or len(set(map(int,o))) != len(o):
        raise ValueError("Unique integer origin order required")
    if c.shape != (p.shape[2],) or len(set(map(str,c))) != len(c) or any(not str(v) for v in c):
        raise ValueError("Unique nonempty channel order required")
    if not np.isfinite(p[m]).all() or not np.isfinite(y[m]).all():
        raise ValueError("Nonfinite observed forecast or target")
    counts, sse, sae, rows = [], [], [], []
    for channel, name in enumerate(c):
        n, sq, ab = 0, 0., 0.
        for index, origin in enumerate(o):
            valid = m[index,:,channel]
            error = p[index,:,channel][valid].astype(np.float64)-y[index,:,channel][valid].astype(np.float64)
            count=int(valid.sum()); square=float(np.sum(error*error,dtype=np.float64)); absolute=float(np.sum(np.abs(error),dtype=np.float64))
            n += count; sq += square; ab += absolute
            rows.append(dict(origin=int(origin),channel=str(name),horizon_start=0,horizon_end=p.shape[1],
                             sse=square,sae=absolute,count=count))
        counts.append(n);sse.append(sq);sae.append(ab)
    active=[i for i,n in enumerate(counts) if n]
    if not active:
        raise ValueError("No original observed B targets")
    channel_mse=[sse[i]/n if n else None for i,n in enumerate(counts)]
    channel_mae=[sae[i]/n if n else None for i,n in enumerate(counts)]
    score=dict(mse=float(np.mean([channel_mse[i] for i in active],dtype=np.float64)),
               mae=float(np.mean([channel_mae[i] for i in active],dtype=np.float64)),
               counts=counts,channel_sse=sse,channel_sae=sae,channel_mse=channel_mse,channel_mae=channel_mae,
               n_observed=sum(counts),n_origins=len(o),num_channels=len(c),active_channels=len(active),horizon=p.shape[1],
               metric_scale="legacy_A_normalized_original_channels",channel_weighting="macro_of_global_observed_channel_means")
    return score,rows


def _compare_curve(score, expected):
    if expected is None:
        return dict(status="NO_RECORDED_B_SCORE",passed=None,atol=1e-7,max_abs_difference=None)
    differences={key:abs(float(score[key])-float(expected[key])) for key in ("mse","mae")}
    maximum=max(differences.values())
    return dict(status="MATCH" if maximum<=1e-7 else "MISMATCH",passed=maximum<=1e-7,
                atol=1e-7,rtol=0.,absolute_differences=differences,max_abs_difference=maximum,
                recorded={key:float(expected[key]) for key in ("mse","mae")})


def _reference(win, columns, horizon):
    origins=np.asarray(win.origins["B"],dtype=np.int64).copy()
    lo,hi=win.fold["bounds"]["B"]
    if not len(origins) or np.any(origins<lo) or np.any(origins+horizon>hi):
        raise ValueError("Only complete legacy B targets may be referenced")
    y,mask=win.truth(origins)  # Exact pinned old API; never truth(A/C).
    result=dict(target_normalized=np.asarray(y).copy(),observed=np.asarray(mask,dtype=bool).copy(),
                origins=origins,channel_ids=np.asarray(columns,dtype=str))
    if result["target_normalized"].shape != (len(origins),horizon,len(columns)):
        raise ValueError("Frozen B reference dimensions differ")
    return result


def rescore_checkpoint(private_dir, public_dir):
    """Rescore only a sealed prediction archive; no model or data-loader access."""
    private,public=Path(private_dir),Path(public_dir)
    manifest=lc.read_json(public/"PREDICTION_MANIFEST.json")
    reference=lc.read_json(public/"REFERENCE_CONTRACT.json")
    archive=private/"predictions.npz"
    if lc.sha(archive)!=manifest["archive_sha256"]:
        raise ValueError("E0 prediction archive hash changed")
    if manifest["reference_sha256"]!=_fingerprint(reference) or not reference.get("created_before_inference"):
        raise ValueError("E0 reference seal changed")
    with np.load(archive,allow_pickle=False) as saved:
        arrays={key:saved[key] for key in saved.files}
    if set(arrays)!=set(KEYS):
        raise ValueError("E0 prediction archive must have exactly five standard keys")
    for key,value in arrays.items():
        actual=array_fingerprint(value)
        if actual!=manifest["arrays"][key] or (key!="pred_normalized" and actual!=reference["arrays"][key]):
            raise ValueError("E0 array or independently frozen target reference changed")
    if manifest["identity"]!=reference["identity"]:
        raise ValueError("E0 reference/model identity differs")
    if manifest["identity"].get("recorded_curve_sha256")!=_fingerprint(manifest.get("expected_B")):
        raise ValueError("E0 recorded B score differs from pre-inference seal")
    if float(reference["created_unix"])>float(manifest["inference_started_unix"]):
        raise ValueError("E0 reference timestamp follows inference")
    score,rows=score_fp64(*(arrays[key] for key in KEYS))
    if _fingerprint(score)!=_fingerprint(manifest["score"]):
        raise ValueError("E0 FP64 rescore differs from saved score")
    comparison=_compare_curve(score,manifest.get("expected_B"))
    if lc.sha(public/"PUBLIC_ERROR_STATS.csv.gz")!=manifest["public_stats_sha256"]:
        raise ValueError("E0 public sufficient statistics changed")
    with gzip.open(public/"PUBLIC_ERROR_STATS.csv.gz","rt",encoding="utf-8",newline="") as stream:
        saved_rows=list(csv.DictReader(stream))
    if len(saved_rows)!=len(rows):
        raise ValueError("E0 public statistics row count differs")
    for actual,expected in zip(saved_rows,rows):
        for key in ("origin","horizon_start","horizon_end","count"):
            if int(actual[key])!=expected[key]:raise ValueError("E0 public statistics cohort/count differs")
        if actual["channel"]!=expected["channel"] or any(float(actual[key])!=expected[key] for key in ("sse","sae")):
            raise ValueError("E0 public SSE/SAE differs from independent archive rescore")
    return dict(status="RESCORE_COMPLETE",score=score,curve_comparison=comparison,
                archive_sha256=manifest["archive_sha256"],reference_sha256=manifest["reference_sha256"],
                independent_FP64_rescore=True,independent_training_rerun=False,
                target_reference_frozen_before_inference=True)


def evaluate_checkpoint(model, foundation, win, columns, identity, private_dir, public_dir, *,
                        expected_B=None, lease=None, deadline=None, device="cuda",horizon=48,batch_size=8,length=512):
    """Root-callable primitive; CPU injection exists solely for synthetic tests."""
    import torch
    _check(lease,deadline,device)
    if batch_size<1:raise ValueError("Positive B inference batch required")
    identity=dict(identity,recorded_curve_sha256=_fingerprint(expected_B))
    private,public=Path(private_dir),Path(public_dir)
    arrays=_reference(win,columns,horizon)
    public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
    reference=dict(identity=identity,split=SPLIT,created_before_inference=True,
                   arrays={key:array_fingerprint(value) for key,value in arrays.items()})
    refpath=public/"REFERENCE_CONTRACT.json";archive=private/"predictions.npz"
    if refpath.exists():
        existing=lc.read_json(refpath)
        if {k:v for k,v in existing.items() if k!="created_unix"}!=reference:
            raise ValueError("Existing E0 pre-inference reference differs")
        reference=existing
    else:
        if archive.exists() or (public/"PREDICTION_MANIFEST.json").exists():
            raise ValueError("Cannot retrospectively create an E0 target reference")
        reference["created_unix"]=time.time();lc.write_json(refpath,reference)
    if archive.exists():
        rescored=rescore_checkpoint(private,public)
        manifest=lc.read_json(public/"PREDICTION_MANIFEST.json")
        if manifest.get("expected_B")!=expected_B:raise ValueError("Existing curve evidence differs")
        return dict(status="CURVE_SCORE_MISMATCH" if rescored["curve_comparison"]["passed"] is False else "REUSED_RESCORED",
                    manifest=manifest,rescore=rescored,cost=manifest["cost"],executed_now=False)
    started=time.time();before_calls=int(foundation.calls);before_streams=int(foundation.streams)
    output=[]
    model.eval()
    with torch.no_grad():
        for lo in range(0,len(arrays["origins"]),int(batch_size)):
            _check(lease,deadline,device)
            batch=arrays["origins"][lo:lo+int(batch_size)]
            x,_,_=win.get(batch,length)
            output.append(model(x,foundation).detach().cpu().numpy())
    arrays["pred_normalized"]=np.concatenate(output)
    score,rows=score_fp64(*(arrays[key] for key in KEYS))
    with archive.open("xb") as stream:np.savez_compressed(stream,**arrays)
    statpath=public/"PUBLIC_ERROR_STATS.csv.gz"
    with statpath.open("xb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as zipped:
            with io.TextIOWrapper(zipped,encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cost=dict(actual_F_calls=int(foundation.calls)-before_calls,actual_F_streams=int(foundation.streams)-before_streams,
              n_origins=len(arrays["origins"]),batch_size=int(batch_size),row_microbatch=getattr(foundation,"row_microbatch",None),
              replay_wall_seconds=time.time()-started,scope="saved_weight_full_legacy_B_reinference_not_training")
    manifest=dict(identity=identity,split=SPLIT,cohort=f"legacy_e0:{identity.get('legacy_id','unknown')}:B:L{length}:H{horizon}",
                  archive_sha256=lc.sha(archive),private_archive_name="predictions.npz",
                  arrays={key:array_fingerprint(value) for key,value in arrays.items()},score=score,expected_B=expected_B,
                  reference_sha256=_fingerprint(reference),public_stats_sha256=lc.sha(statpath),cost=cost,
                  historically_unseen=False,independent_training_rerun=False,raw_target_publication=False,
                  C_scored=False,protected_TEST_accessed=False,reference_created_unix=reference["created_unix"],
                  inference_started_unix=started)
    lc.write_json(public/"PREDICTION_MANIFEST.json",manifest)
    rescore=rescore_checkpoint(private,public)
    lc.write_json(public/"INDEPENDENT_RESCORE.json",rescore)
    return dict(status="CURVE_SCORE_MISMATCH" if rescore["curve_comparison"]["passed"] is False else "REPLAY_RESCORE_COMPLETE",
                manifest=manifest,rescore=rescore,cost=cost,executed_now=True)


def _add_row(rows,row,origin):
    update=int(row["update"])
    if update in rows and rows[update]["row"]!=row:
        raise ValueError(f"Conflicting B checkpoint records at update {update}")
    rows.setdefault(update,dict(row=row,bindings=[]))["bindings"].append(origin)


def _receipt_matches_id(path,rec,name,budget=False):
    if not isinstance(rec,dict) or not isinstance(rec.get("id"),str) or not re.fullmatch(r"[A-Za-z0-9_]+",rec["id"]):
        raise ValueError("Invalid continuation receipt ID")
    endpoint=int(rec["endpoint"])
    if endpoint not in lc.ENDPOINTS:
        raise ValueError("Endpoint outside the frozen continuation schedule")
    stem=re.escape(rec["id"])+"_"+str(endpoint)
    pattern=stem+(r"_[0-9]+" if budget else "")+r"\.json"
    if re.fullmatch(pattern,path.name) is None:
        raise ValueError("Continuation receipt filename/ID/endpoint differs")
    return rec["id"]==name


def inventory_checkpoints(asset_root,cache_root,public_root=None,legacy_ids=None):
    """Verify small checkpoint/receipt evidence only. Never open source arrays."""
    import torch
    asset,cache=Path(asset_root).resolve(),Path(cache_root).resolve()
    public=Path(public_root or lc.HERE).resolve();old_lock,sources=lc.validate_sources(asset)
    fits=public/"legacy_e0/fits";budgets=public/"legacy_e0/budgets"
    if legacy_ids is None:
        ids={lc.read_json(p)["id"] for folder in (fits,budgets) for p in folder.glob("*.json")}
        ids.update(path.name for path in (cache/"legacy_e0/checkpoints").glob("*") if path.is_dir())
    else:ids=set(legacy_ids)
    result=[]
    for name in sorted(ids):
        if not re.fullmatch(r"[A-Za-z0-9_]+",name):raise ValueError("Invalid legacy evaluation ID")
        try:
            old,old_path=lc.receipt_for(asset,name);rows={};bindings=[];selected=set()
            for row in old["checkpoints"]:_add_row(rows,row,dict(scope="old_public",relative_path=old_path.relative_to(asset).as_posix(),sha256=lc.sha(old_path)))
            for path in sorted(fits.glob(f"{name}_*.json")):
                rec=lc.read_json(path)
                if not _receipt_matches_id(path,rec,name):continue
                endpoint=int(rec["endpoint"])
                if (rec["id"]!=name or rec["status"]!="CONTINUED_ENDPOINT_COMPLETE" or
                    rec["source_hashes"]!=sources or rec["statistics_sha256"]!=old_lock["statistics_hash"] or
                    rec["F_model_hash"]!=old_lock["F_model_hash"] or
                    rec["new_runner_source_sha256"]!=lc.sha(Path(lc.__file__)) or
                    rec["legacy_receipt_sha256"]!=lc.sha(old_path) or
                    rec["legacy_last2400_sha256"]!=lc.sha(lc.checkpoint_path(asset,name))):
                    raise ValueError("Endpoint source/statistics/foundation/legacy evidence differs")
                if endpoint not in lc.ENDPOINTS:
                    raise ValueError("Endpoint outside the frozen continuation schedule")
                binding=dict(scope="public",relative_path=path.relative_to(public).as_posix(),sha256=lc.sha(path))
                lastpath=cache/"legacy_e0/training"/name/f"last_{endpoint}.pt"
                selectedpath=cache/"legacy_e0/selected"/f"{name}_{endpoint}.pt"
                if lc.sha(lastpath)!=rec["last_file_sha256"] or lc.sha(selectedpath)!=rec["selected_file_sha256"]:
                    raise ValueError("E0 last/selected saved-weight file hash differs")
                last=torch.load(lastpath,weights_only=True,map_location="cpu");lc.validate_carry(last["carry"],endpoint)
                if last["carry"]["checkpoints"]!=rec["checkpoints"]:
                    raise ValueError("Endpoint full carry and B curve receipt differ")
                for row in rec["checkpoints"]:_add_row(rows,row,binding)
                endpoint_row=rows[endpoint]["row"];chosen=rows[int(rec["selected_update"])]["row"]
                if lc.tensor_state_hash(last["state"])!=endpoint_row["state_sha256"]:
                    raise ValueError("Endpoint last-state tensor hash differs")
                selected_state=torch.load(selectedpath,weights_only=True,map_location="cpu")
                if lc.tensor_state_hash(selected_state)!=rec["selected_state_sha256"] or rec["selected_state_sha256"]!=chosen["state_sha256"] or rec["selected_B"]!=chosen["B"]:
                    raise ValueError("Endpoint selected-state tensor/curve hash differs")
                selected.add(int(rec["selected_update"]))
                bindings.append(dict(**binding,endpoint=endpoint,last_file_sha256=rec["last_file_sha256"],
                    selected_file_sha256=rec["selected_file_sha256"],last_state_verified=True,selected_state_verified=True))
            for path in sorted(budgets.glob(f"{name}_*.json")):
                rec=lc.read_json(path)
                if not _receipt_matches_id(path,rec,name,budget=True):continue
                if not rec.get("initialized"):continue
                relative=Path(rec["budget_checkpoint_cache_relative_path"])
                checkpoint=(cache/relative).resolve()
                if not checkpoint.is_relative_to((cache/"legacy_e0").resolve()) or lc.sha(checkpoint)!=rec["budget_checkpoint_sha256"]:
                    raise ValueError("Budget carry path/hash differs")
                side=lc.read_json(checkpoint.with_suffix(checkpoint.suffix+".json"));contract=side["budget_contract"]
                if (contract["legacy_id"]!=name or contract["endpoint"]!=rec["endpoint"] or
                    contract["source_hashes"]!=sources or contract["statistics_sha256"]!=old_lock["statistics_hash"] or
                    contract["foundation_sha256"]!=old_lock["F_model_hash"] or
                    contract["runner_source_sha256"]!=lc.sha(Path(lc.__file__)) or
                    contract["legacy_receipt_sha256"]!=lc.sha(old_path) or
                    contract["legacy_last2400_sha256"]!=lc.sha(lc.checkpoint_path(asset,name)) or
                    contract["resource_guard_source_sha256"]!=lc.sha(Path(lc.__file__).with_name("resources.py"))):
                    raise ValueError("Budget carry source/statistics/identity differs")
                carry=lc.load_budget_checkpoint(checkpoint,contract)["carry"]
                binding=dict(scope="public",relative_path=path.relative_to(public).as_posix(),sha256=lc.sha(path))
                for row in carry["checkpoints"]:_add_row(rows,row,binding)
                selected.add(int(lc.best_checkpoint(carry["checkpoints"])["update"]))
                bindings.append(dict(**binding,partial_endpoint=True,current_update=carry["updates"],
                                     budget_checkpoint_sha256=lc.sha(checkpoint)))
            candidates=[]
            updates={int(path.stem[2:]) for path in (cache/"legacy_e0/checkpoints"/name).glob("cp*.pt") if re.fullmatch(r"cp\d+",path.stem) and int(path.stem[2:])>=2800}
            updates.update(update for update in selected if update<=2400)
            for update in sorted(updates):
                checkpoint=(asset/".cache"/lc.OLD_NAME/"checkpoints"/name/f"cp{update}.pt" if update<=2400
                            else cache/"legacy_e0/checkpoints"/name/f"cp{update}.pt")
                try:
                    state=torch.load(checkpoint,weights_only=True,map_location="cpu");actual=lc.tensor_state_hash(state)
                    row=rows.get(update,{}).get("row")
                    if row is not None and row["state_sha256"]!=actual:raise ValueError("B checkpoint tensor hash differs")
                    candidates.append(dict(update=update,status="READY",scope="asset" if update<=2400 else "cache",
                        checkpoint_relative_path=checkpoint.relative_to(asset if update<=2400 else cache).as_posix(),
                        file_sha256=lc.sha(checkpoint),state_sha256=actual,expected_B=row["B"] if row else None,
                        curve_evidence_status="BOUND_RECORDED_B_SCORE" if row else "SAVED_STATE_WITHOUT_PERSISTED_B_SCORE",
                        role="selected_old_checkpoint" if update<=2400 else "new_checkpoint",
                        evidence_bindings=rows.get(update,{}).get("bindings",[])))
                except Exception as error:
                    reason=str(error).replace(str(asset),"<asset-root>").replace(str(cache),"<cache-root>")
                    candidates.append(dict(update=update,status="CHECKPOINT_INVENTORY_FAILED",error_type=type(error).__name__,reason=reason))
            result.append(dict(id=name,status="READY" if candidates else "NO_NEW_B_CHECKPOINTS",spec=lc.model_spec(old),
                               legacy_receipt_sha256=lc.sha(old_path),endpoint_bindings=bindings,checkpoints=candidates))
        except (OSError,ValueError,KeyError,RuntimeError,AssertionError) as error:
            result.append(dict(id=name,status="INVENTORY_FAILED",error_type=type(error).__name__,reason=str(error).replace(str(asset),"<asset-root>").replace(str(cache),"<cache-root>").replace(str(public),"<public-root>")))
    return dict(status="INVENTORIED",routes=result,source_hashes=sources,statistics_sha256=old_lock["statistics_hash"],
                foundation_sha256=old_lock["F_model_hash"],dataset_arrays_loaded=False,GPU_used=False)


def load_B_window(modules,asset_root,receipt,lease,deadline=None):
    """Pinned old Windows on A+B only; slice NPZ members before conversion."""
    _check(lease,deadline,"cuda")
    dio,cp=modules["data"],modules["campaign"]
    name=receipt["dataset"];name="beijing" if name=="beijing_aotizhongxin" else name
    spec=dio.ASSETS[name];contract=dio._read_contract(spec);dio.install_data_access_guard()
    source=(Path(asset_root)/spec["path"]).resolve()
    if not source.is_file() or source.stat().st_size!=spec["bytes"] or lc.sha(source)!=spec["sha256"]:
        raise ValueError("Pinned legacy safe source absent or changed")
    lo,hi=spec["permitted_bounds"];fold=dio.make_fold(hi-lo,receipt.get("fold","primary"))
    if spec["kind"]=="physical":
        if tuple(contract["bounds"]["fit"])!=(lo,hi):raise ValueError("Pinned legacy FIT contract boundary differs")
    else:
        original=contract["bounds"]["train"]
        if (original[0],original[0]+int(.8*(original[1]-original[0])))!=(lo,hi):
            raise ValueError("Pinned legacy TRAIN-to-FIT boundary differs")
    stop=fold["bounds"]["B"][1]
    with np.load(source,allow_pickle=False) as archive:
        maskkey="mask" if spec["kind"]=="physical" else "finite"
        observed=archive[maskkey][lo:lo+stop].astype(bool,copy=True)
        if spec["kind"]=="physical":physical=archive["physical"][lo:lo+stop].astype(np.float32,copy=True)
        else:
            x=archive["x"][lo:lo+stop].astype(np.float32,copy=True)
            physical=(x*archive["std"].astype(np.float32)+archive["mean"].astype(np.float32)).astype(np.float32)
        times=archive["times"][lo:lo+stop].copy();columns=archive["columns"].tolist()
    if physical.shape!=(stop,spec["channels"]) or observed.shape!=physical.shape or len(columns)!=spec["channels"]:
        raise ValueError("Legacy B prefix dimensions differ")
    if not np.isfinite(physical[observed]).all():raise ValueError("Nonfinite true observed A/B value")
    expected=contract.get("columns",contract.get("source",{}).get("parse",{}).get("columns"))
    if expected is not None and columns!=expected:raise ValueError("Legacy channel order changed")
    times=times.astype("datetime64[ns]").astype(np.int64) if times.dtype.kind in {"U","S"} else times.astype(np.int64)
    if len(times)!=stop or not np.all(np.diff(times)==spec["interval_seconds"]*1_000_000_000):raise ValueError("Legacy timestamp grid changed")
    physical[~observed]=np.nan
    fold["origins"]["C"]=np.empty(0,np.int64)
    data=dict(physical=physical,observed=observed,columns=columns,times=times,permitted_bounds=(lo,lo+stop))
    _check(lease,deadline,"cuda");win=cp.Windows(data,fold)
    statistics=lc.read_json(Path(asset_root)/"research"/lc.OLD_NAME/"FOLD_STATISTICS.json")[name]["primary"]
    np.testing.assert_array_equal(win.mean,np.asarray(statistics["A_mean"],np.float32))
    np.testing.assert_array_equal(win.std,np.asarray(statistics["A_std"],np.float32))
    if len(win.origins["A"])!=receipt["legal_A_origins"]:raise ValueError("Legacy legal A origin count changed")
    return win,dict(data_sha256=spec["sha256"],contract_sha256=spec["contract_sha256"],
        scaler_sha256=_fingerprint(dict(mean=array_fingerprint(win.mean),std=array_fingerprint(win.std))),
        statistics_scope="old_exact_A_only",array_computation_scope="old_FIT_A_plus_B_prefix_only",
        archive_byte_scope="Whole compressed required NPZ members and hash; not physically row-isolated; no C target extraction or scoring",
        B_bounds=list(fold["bounds"]["B"]),original_permitted_bounds=list(spec["permitted_bounds"])),columns


def _expected_width(spec,channels):
    if spec["type"]=="G":return _expected_width(spec["base"],channels)
    if spec["type"]=="CURRENT":return 0
    return channels if spec["type"]=="RAW" else int(spec["r"])


def evaluate_legacy(asset_root,cache_root,public_root=None,legacy_ids=None,*,lease=None,deadline=None,batch_size=8,row_microbatch=None):
    """Actual E0 B replay. Root calls this in a fresh centrally leased child."""
    import torch
    public=Path(public_root or lc.HERE).resolve();cache=Path(cache_root).resolve();asset=Path(asset_root).resolve()
    lease=lease if lease is not None else validate_gpu_child(cache)
    _check(lease,deadline,"cuda")
    if public!=lc.HERE.resolve():raise ValueError("Actual legacy loader is pinned to this campaign public root")
    if row_microbatch is not None and int(row_microbatch)<1:raise ValueError("Positive foundation row microbatch required")
    inventory=inventory_checkpoints(asset,cache,public,legacy_ids)
    modules,old_lock,sources,_=lc.load_legacy(asset,cache)
    rt=modules["runtime"];results=[];bolt=None;hook=None
    source_sha=lc.sha(Path(__file__));started=time.time();actual_streams=actual_calls=0
    try:
        for route in inventory["routes"]:
            if route["status"]!="READY":
                results.append(dict(id=route["id"],status=route["status"],reason=route.get("reason")));continue
            try:
                _check(lease,deadline,"cuda")
                rec,_=lc.receipt_for(asset,route["id"])
                win,data_identity,columns=load_B_window(modules,asset,rec,lease,deadline)
                if bolt is None:
                    _check(lease,deadline,"cuda");bolt=rt.Bolt()
                    if bolt.initial_hash!=old_lock["F_model_hash"]:raise ValueError("Actual old foundation tensor hash differs")
                    if any(p.requires_grad or p.grad is not None for p in bolt.model.parameters()):raise ValueError("Old foundation must remain frozen")
                    bolt.row_microbatch=row_microbatch
                    hook=bolt.model.register_forward_pre_hook(lambda module,args:_check(lease,deadline,"cuda"))
                for checkpoint in route["checkpoints"]:
                    if checkpoint["status"]!="READY":
                        results.append(dict(id=route["id"],**checkpoint));continue
                    calls_before=int(bolt.calls);streams_before=int(bolt.streams)
                    try:
                        _check(lease,deadline,"cuda")
                        root=asset if checkpoint["scope"]=="asset" else cache
                        path=root/checkpoint["checkpoint_relative_path"]
                        if lc.sha(path)!=checkpoint["file_sha256"]:raise ValueError("Checkpoint file changed after inventory")
                        state=torch.load(path,weights_only=True,map_location="cpu")
                        if lc.tensor_state_hash(state)!=checkpoint["state_sha256"]:raise ValueError("Checkpoint tensor state changed")
                        _check(lease,deadline,"cuda");model=lc.build_legacy_model(route["spec"],state,modules).cuda().eval()
                        if rt.state_hash(model)!=checkpoint["state_sha256"]:raise ValueError("Loaded model tensor hash differs")
                        identity=dict(legacy_id=route["id"],checkpoint_update=checkpoint["update"],spec=route["spec"],
                            model_file_sha256=checkpoint["file_sha256"],model_state_sha256=checkpoint["state_sha256"],
                            checkpoint_relative_path=checkpoint["checkpoint_relative_path"],model_path_scope=checkpoint["scope"],
                            source_hashes=sources,evaluation_source_sha256=source_sha,continuation_source_sha256=lc.sha(Path(lc.__file__)),
                            statistics_sha256=old_lock["statistics_hash"],foundation_sha256=bolt.initial_hash,
                            legacy_receipt_sha256=route["legacy_receipt_sha256"],curve_evidence_status=checkpoint["curve_evidence_status"],
                            recorded_curve_sha256=_fingerprint(checkpoint["expected_B"]),
                            length=512,horizon=48,fold="primary",**data_identity)
                        relative=Path("legacy_e0/evaluation")/route["id"]/f"cp{checkpoint['update']}"
                        result=evaluate_checkpoint(model,bolt,win,columns,identity,cache/relative,public/relative,
                            expected_B=checkpoint["expected_B"],lease=lease,deadline=deadline,batch_size=batch_size)
                        if result["executed_now"] and result["cost"]["actual_F_streams"]!=len(win.origins["B"])*_expected_width(route["spec"],len(columns)):
                            raise ValueError("Actual E0 F streams differ from model input width")
                        if rt.state_hash(model)!=checkpoint["state_sha256"] or rt.state_hash(bolt.model)!=bolt.initial_hash:
                            raise ValueError("Saved checkpoint or frozen foundation changed during replay")
                        results.append(dict(id=route["id"],update=checkpoint["update"],status=result["status"],
                            executed_now=result["executed_now"],score=result["rescore"]["score"],
                            curve_comparison=result["rescore"]["curve_comparison"],manifest_relative_path=(relative/"PREDICTION_MANIFEST.json").as_posix(),
                            actual_F_streams=int(bolt.streams)-streams_before,actual_F_calls=int(bolt.calls)-calls_before))
                        del model,state
                    except (BudgetExceeded,ResourceBusy) as error:
                        results.append(dict(id=route["id"],update=checkpoint["update"],status="BUDGET_PENDING",reason=str(error),
                            actual_F_streams=int(bolt.streams)-streams_before,actual_F_calls=int(bolt.calls)-calls_before))
                    except Exception as error:
                        reason=str(error).replace(str(asset),"<asset-root>").replace(str(cache),"<cache-root>").replace(str(public),"<public-root>")
                        results.append(dict(id=route["id"],update=checkpoint["update"],status="REPLAY_FAILED",error_type=type(error).__name__,reason=reason,
                            actual_F_streams=int(bolt.streams)-streams_before,actual_F_calls=int(bolt.calls)-calls_before))
            except (BudgetExceeded,ResourceBusy) as error:
                results.append(dict(id=route["id"],status="BUDGET_PENDING",reason=str(error)))
            except Exception as error:
                reason=str(error).replace(str(asset),"<asset-root>").replace(str(cache),"<cache-root>").replace(str(public),"<public-root>")
                results.append(dict(id=route["id"],status="ROUTE_FAILED",error_type=type(error).__name__,reason=reason))
        if bolt is not None:
            actual_streams=int(bolt.streams);actual_calls=int(bolt.calls)
            if rt.state_hash(bolt.model)!=old_lock["F_model_hash"]:raise ValueError("Final frozen foundation changed")
        if source_sha!=lc.sha(Path(__file__)):raise ValueError("E0 evaluator source changed during actual execution")
        failed=any(row["status"] in {"BUDGET_PENDING","REPLAY_FAILED","ROUTE_FAILED","INVENTORY_FAILED","CHECKPOINT_INVENTORY_FAILED","CURVE_SCORE_MISMATCH"} for row in results)
        count=sum(len(route.get("checkpoints",[])) for route in inventory["routes"])
        summary=dict(status="EVALUATION_FINISHED_WITH_PENDING_OR_FAILURES" if failed else "EVALUATION_COMPLETE" if count else "NO_ELIGIBLE_CHECKPOINTS",
            available_checkpoint_count=count,
            split=SPLIT,historically_unseen=False,independent_training_rerun=False,C_scored=False,protected_TEST_accessed=False,
            checkpoints_replayed=sum(bool(row.get("executed_now")) for row in results),
            checkpoints_rescored=sum("score" in row for row in results),
            checkpoints_matching_recorded_curve=sum(row.get("curve_comparison",{}).get("passed") is True for row in results),
            actual_F_streams=actual_streams,actual_F_calls=actual_calls,results=results,
            source_hashes=sources,evaluation_source_sha256=source_sha,statistics_sha256=old_lock["statistics_hash"],
            actual_foundation_sha256=old_lock["F_model_hash"] if bolt is not None else None,
            budget=dict(deadline_unix=min(float(deadline),float(lease.deadline)) if deadline is not None else float(lease.deadline),
                        enforcement="central_live_owner_or_authenticated_child; per_origin_batch_and_each_actual_F_forward",row_microbatch=row_microbatch),
            started_unix=started,finished_unix=time.time(),inventory=inventory)
        path=public/"legacy_e0/evaluation"/f"REGISTRY_{time.time_ns()}.json";lc.write_json(path,summary)
        return dict(summary,registry_relative_path=path.relative_to(public).as_posix())
    finally:
        if hook is not None:hook.remove()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("inventory","run"));parser.add_argument("--asset-root",required=True)
    parser.add_argument("--cache-root",required=True);parser.add_argument("--id",action="append",dest="ids")
    parser.add_argument("--batch-size",type=int,default=8);parser.add_argument("--row-microbatch",type=int)
    parser.add_argument("--execute-evaluation",action="store_true");parser.add_argument("--result-json")
    args=parser.parse_args()
    if args.mode=="run":
        if not args.execute_evaluation:parser.error("Actual B replay requires --execute-evaluation and the central child lease")
        result=evaluate_legacy(args.asset_root,args.cache_root,legacy_ids=args.ids,batch_size=args.batch_size,row_microbatch=args.row_microbatch)
    else:result=inventory_checkpoints(args.asset_root,args.cache_root,legacy_ids=args.ids)
    if args.result_json:lc.write_json(args.result_json,result)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":main()
