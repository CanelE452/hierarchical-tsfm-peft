"""Physical-input checkpoint inference with independent pre-inference truth seals."""
import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np

import data_contract as data
import deployment
import evaluation
import fit_base
import route_catalog
from training import fingerprint

KEYS = ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _new_or_same(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _json(path) != value:
            raise ValueError("Existing immutable route reference differs")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def _seal_reference(path,value,artifacts_exist):
    if path.exists():
        existing=_json(path)
        if {key:item for key,item in existing.items() if key!="created_unix"}!=value:
            raise ValueError("Existing pre-inference target reference differs")
        return existing
    if artifacts_exist:
        raise ValueError("Cannot retrospectively create a pre-inference target reference")
    sealed=dict(value,created_unix=time.time())
    _new_or_same(path,sealed)
    return sealed


def _check(lease, deadline, device):
    if str(device).startswith("cuda") and lease is None:
        raise RuntimeError("Physical GPU inference requires the active campaign owner or authenticated child lease")
    if lease is not None:
        lease.check()
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Route inference budget reached")


def _selection_gate(public, run_id, spec):
    path = Path(public)/"SELECTION_LOCK.json"
    if not path.is_file():
        raise ValueError("T_final is locked: selection lock absent")
    lock = _json(path)
    if lock.get("locked") is not True or lock.get("allowed_route_sha", {}).get(run_id) != fingerprint(spec):
        raise ValueError("T_final selection lock does not bind this exact portable route")


def _source_sha():
    names = ("route_catalog.py", "route_predictions.py", "deployment.py", "data_contract.py", "evaluation.py",
             "training.py", "networks.py", "gpu_runtime.py", "ridge_baseline.py", "combination.py", "fit_base.py")
    return fingerprint({name:data.sha256_file(Path(__file__).with_name(name)) for name in names})


def _truth(task, selected_origins, mean, std):
    pairs = [data.target_window(task, int(origin), mean, std) for origin in selected_origins]
    return dict(target_normalized=np.stack([value[0] for value in pairs]), observed=np.stack([value[1] for value in pairs]),
                origins=np.asarray(selected_origins, np.int64), channel_ids=np.asarray(task["columns"],dtype=str))


def _infer(spec, task, selected_origins, cache, model_path, device, deadline, lease, foundation_pool, batch_size):
    _check(lease,deadline,device)
    if foundation_pool is None and spec.get("foundation_asset"):
        if model_path is None:
            raise ValueError("Pinned mini/tiny route needs its local model snapshot")
        for name, file in spec["foundation_asset"]["files"].items():
            path = Path(model_path)/name
            if not path.is_file() or data.sha256_file(path) != file["sha256"]:
                raise ValueError("Caller model snapshot differs from the pinned portable route")
    route = deployment.Route(spec, task, cache, model_path=model_path, device=device, foundation_pool=foundation_pool)
    physical, streams, calls, physical_streams = [], 0, 0, 0
    try:
        def pins(value):
            found = {value["foundation_sha"]} if value.get("foundation_sha") not in {None,"NO_FOUNDATION"} else set()
            for key in ("g_route","b_route"):
                if key in value:found.update(pins(value[key]))
            return found
        expected_pins=pins(spec)
        if expected_pins:
            actual = fit_base._model_sha(route.pool.get(task["H"]))
            if expected_pins != {actual}:
                raise ValueError("Actual foundation differs from a nested route pin")
        for lo in range(0,len(selected_origins),int(batch_size)):
            _check(lease,deadline,device)
            batch = selected_origins[lo:lo+int(batch_size)]
            rows = batch[:,None]+np.arange(-task["L"],0)[None]
            x, mask = task["physical"][rows].copy(),task["observed"][rows].copy()
            physical.append(route.predict_physical(x,mask))
            streams += int(route.last_receipt["F_logical_streams"])
            physical_streams += int(route.last_receipt["F_physical_streams"])
            calls += int(route.last_receipt["F_calls"])
        expected = route_catalog.expected_F_streams(spec,len(selected_origins),len(task["columns"]))
        if streams != expected:
            raise ValueError("Actual route F rows differ from active latent/original width")
        return np.concatenate(physical),dict(actual_F_streams=streams,expected_F_streams=expected,
                F_physical_streams=physical_streams,F_calls=calls,batch=len(selected_origins),
                foundation_instances_created=route.pool.instances_created,scope="actual_physical_input_deployment")
    finally:
        route.close()


def _stats(path, rows):
    fields=("run_id","split","cohort","origin","channel","horizon_start","horizon_end","sse","sae","count")
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as zipped:
            import io
            with io.TextIOWrapper(zipped,encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def predict_route(run_id,task,split,cache_root,public_root,model_path=None,device="cuda",deadline=None,
                  *,lease=None,foundation_pool=None,batch_size=4):
    """Reinfer a full V/T cohort under route_predictions/, preserving old V caches."""
    if split not in {"V_select","T_final"} or int(batch_size)<1:
        raise ValueError("Registered V/T split and positive physical batch required")
    cache,public=Path(cache_root),Path(public_root)
    spec,size=route_catalog.build_route(run_id,task,cache,public)
    if split=="T_final":
        _selection_gate(public,run_id,spec)  # Before target_window, route load or inference.
    bounds=data.split_ranges(task,spec.get("fold","primary"))
    mean,std,_=data.fit_statistics(task,bounds["A_stop"][1])
    selected=data.origins(bounds[split],task["L"],task["H"],task["H"])
    if not len(selected):raise ValueError("No legal route evaluation origins")
    arrays=_truth(task,selected,mean,std)
    identity=dict(run_id=run_id,split=split,cohort=f"{task['name']}:{spec.get('fold','primary')}:{split}:stride{task['H']}",
        source_sha=_source_sha(),config_sha=fingerprint(dict(route_sha=fingerprint(spec),batch_size=int(batch_size))),
        data_sha=task["data_sha256"],model_sha=fingerprint(spec),scaler_sha=fit_base.scaler_fingerprint(mean,std))
    reference=dict(identity,arrays={key:evaluation.array_fingerprint(value) for key,value in arrays.items()},
        expected_F_streams=route_catalog.expected_F_streams(spec,len(selected),len(task["columns"])),
        route_sha=fingerprint(spec),artifact_bindings=route_catalog.artifact_bindings(spec),created_before_inference=True)
    folder=public/"route_predictions"/run_id/split
    private=cache/"route_predictions"/run_id
    folder.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
    path=private/(split+".npz")
    manifest_path=folder/"PREDICTION_MANIFEST.json"
    reference=_seal_reference(folder/"REFERENCE_CONTRACT.json",reference,path.exists() or manifest_path.exists())
    if path.exists():
        manifest=_json(manifest_path)
        if any(manifest.get(key)!=value for key,value in identity.items()) or data.sha256_file(path)!=manifest["archive_sha256"]:
            raise ValueError("Existing route prediction contract or archive changed")
        with np.load(path,allow_pickle=False) as archive:
            saved={key:archive[key] for key in archive.files}
        if set(saved)!=set(KEYS):raise ValueError("Route prediction cache keys changed")
        for key,value in saved.items():
            if evaluation.array_fingerprint(value)!=manifest["arrays"][key]:raise ValueError("Route cached array changed")
            if key!="pred_normalized" and evaluation.array_fingerprint(value)!=reference["arrays"][key]:
                raise ValueError("Route cached target cohort changed")
        return dict(saved,metadata=manifest["metadata"],manifest=manifest,reference_contract=reference,
                    score=manifest["score"],cost_record=manifest["cost_record"],
                    public_stats_relative_path=manifest["public_stats_relative_path"],executed_now=False)
    physical,cost=_infer(spec,task,selected,cache,model_path,device,deadline,lease,foundation_pool,batch_size)
    arrays["pred_normalized"]=((physical.astype(np.float64)-mean.astype(np.float64))/std.astype(np.float64)).astype(np.float32)
    if _source_sha()!=identity["source_sha"]:raise ValueError("Route inference source changed during execution")
    with path.open("xb") as stream:np.savez_compressed(stream,**arrays)
    score,rows=evaluation.error_statistics(arrays["pred_normalized"],arrays["target_normalized"],arrays["observed"],selected,
                                           run_id,split,identity["cohort"],arrays["channel_ids"])
    stat_path=folder/"PUBLIC_ERROR_STATS.csv.gz";_stats(stat_path,rows)
    with (folder/"SCORES.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(score));writer.writeheader();writer.writerow(score)
    metadata=dict(identity,route_sha=fingerprint(spec),route_spec=spec,model_size=size,
                  artifact_bindings=route_catalog.artifact_bindings(spec),T_gate_checked=split=="T_final",
                  independent_training_rerun=False,raw_target_publication=False,
                  execution_scope="actual_physical_input_checkpoint_reinference")
    manifest=dict(identity,cache_relative_path=path.relative_to(cache).as_posix(),archive_sha256=data.sha256_file(path),
                  arrays={key:evaluation.array_fingerprint(value) for key,value in arrays.items()},score=score,
                  cost_record=dict(cost,run_id=run_id),metadata=metadata,
                  public_stats_relative_path=stat_path.relative_to(public).as_posix())
    _new_or_same(manifest_path,manifest)
    return dict(arrays,metadata=metadata,manifest=manifest,reference_contract=reference,score=score,
                cost_record=manifest["cost_record"],public_stats_relative_path=manifest["public_stats_relative_path"],executed_now=True)


def _standard(run_id,cache,public):
    path=public/"predictions"/run_id/"V_select/PREDICTION_MANIFEST.json"
    if not path.exists():path=public/"ordered_predictions"/run_id/"PREDICTION_MANIFEST.json"
    manifest=_json(path)
    archive=route_catalog._inside(cache,manifest["cache_relative_path"])
    if data.sha256_file(archive)!=manifest["archive_sha256"]:raise ValueError("Standard V prediction archive changed")
    with np.load(archive,allow_pickle=False) as value:arrays={key:value[key] for key in value.files}
    if set(arrays)!=set(KEYS):raise ValueError("Standard V cache keys differ")
    for key,value in arrays.items():
        if evaluation.array_fingerprint(value)!=manifest["arrays"][key]:raise ValueError("Standard V array fingerprint changed")
    return arrays,manifest,path


def replay_route(run_id,task,cache_root,public_root,model_path=None,device="cuda",deadline=None,
                 *,lease=None,foundation_pool=None,batch_size=4,atol=1e-5,rtol=1e-4):
    """Replay fixed V first/middle/last plus seeded random origins, never retrain."""
    cache,public=Path(cache_root),Path(public_root)
    spec,size=route_catalog.build_route(run_id,task,cache,public)
    saved,manifest,standard_path=_standard(run_id,cache,public)
    legal=data.origins(data.split_ranges(task,spec.get("fold","primary"))["V_select"],task["L"],task["H"],task["H"])
    if not np.array_equal(saved["origins"],legal):raise ValueError("Standard V origins differ from registered cohort")
    fixed={0,len(legal)//2,len(legal)-1}
    other=np.asarray([index for index in range(len(legal)) if index not in fixed],np.int64)
    random=np.random.default_rng(92601).choice(other,min(3,len(other)),replace=False) if len(other) else []
    indices=np.asarray(sorted(fixed|set(map(int,random))),np.int64)
    selected=legal[indices]
    bounds=data.split_ranges(task,spec.get("fold","primary"));mean,std,_=data.fit_statistics(task,bounds["A_stop"][1])
    truth=_truth(task,selected,mean,std)
    for key,value in truth.items():
        expected=saved[key][indices] if key!="channel_ids" else saved[key]
        if not np.array_equal(value,expected):raise ValueError("Standard V selected target/mask/channel differs from Task truth")
    reference=dict(run_id=run_id,split="V_select",origins=selected.tolist(),route_sha=fingerprint(spec),
        standard_manifest_sha256=data.sha256_file(standard_path),standard_archive_sha256=manifest["archive_sha256"],
        source_sha=_source_sha(),artifact_bindings=route_catalog.artifact_bindings(spec),
        arrays={key:evaluation.array_fingerprint(value) for key,value in truth.items()},
        expected_F_streams=route_catalog.expected_F_streams(spec,len(selected),len(task["columns"])),
        selection_policy="first/middle/last plus 3 remaining origins from seed92601; sorted unique",created_before_inference=True)
    folder=public/"route_replays"/run_id;folder.mkdir(parents=True,exist_ok=True)
    receipt_path=folder/"REPLAY_RECEIPT.json"
    replay_path=cache/"route_replays"/run_id/"V_subset.npz"
    reference=_seal_reference(folder/"PRE_INFERENCE_REFERENCE.json",reference,receipt_path.exists() or replay_path.exists())
    if receipt_path.exists():
        previous=_json(receipt_path)
        archive=route_catalog._inside(cache,previous["cache_relative_path"])
        if previous["reference_sha"]!=fingerprint(reference) or data.sha256_file(archive)!=previous["archive_sha256"]:
            raise ValueError("Existing route replay receipt/cache differs")
        return dict(previous,executed_now=False)
    physical,cost=_infer(spec,task,selected,cache,model_path,device,deadline,lease,foundation_pool,batch_size)
    prediction=((physical.astype(np.float64)-mean.astype(np.float64))/std.astype(np.float64)).astype(np.float32)
    expected=saved["pred_normalized"][indices]
    passed=bool(np.allclose(prediction,expected,atol=atol,rtol=rtol))
    if _source_sha()!=reference["source_sha"]:raise ValueError("Checkpoint replay source changed during execution")
    private=cache/"route_replays"/run_id;private.mkdir(parents=True,exist_ok=True)
    path=private/"V_subset.npz"
    with path.open("xb") as stream:np.savez_compressed(stream,pred_normalized=prediction,**truth)
    receipt=dict(status="PASS" if passed else "FAIL_NUMERICAL_PARITY",run_id=run_id,origins=selected.tolist(),
                 route_sha=fingerprint(spec),model_size=size,reference_sha=fingerprint(reference),
                 selected_artifact_bindings=route_catalog.artifact_bindings(spec),
                 standard_manifest_public_relative_path=standard_path.relative_to(public).as_posix(),
                 standard_manifest_sha256=reference["standard_manifest_sha256"],
                 max_abs=float(np.max(np.abs(prediction.astype(np.float64)-expected.astype(np.float64)))),
                 atol=float(atol),rtol=float(rtol),actual_F_streams=cost["actual_F_streams"],expected_F_streams=cost["expected_F_streams"],
                 F_calls=cost["F_calls"],F_physical_streams=cost["F_physical_streams"],
                 cache_relative_path=path.relative_to(cache).as_posix(),archive_sha256=data.sha256_file(path),
                 independent_training_rerun=False,T_final_accessed=False,scope="actual_physical_input_selected_checkpoint_V_subset_replay")
    _new_or_same(receipt_path,receipt)
    return dict(receipt,executed_now=True)
