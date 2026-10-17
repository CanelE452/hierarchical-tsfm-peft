"""Read-only historical asset inspection; outputs belong to this campaign."""
import importlib
from pathlib import Path
import subprocess
import sys

from .contracts import atomic_json, binding, read, resolve_binding, sha, utc, validate_plan

OLD = "adapter_diagnostics_20261017_v2"


def old_context(ctx):
    return dict(ctx, cache=ctx["previous_cache"], report=Path(ctx["repo"])/"reports"/OLD)


def _bound_old_references(value, ctx):
    found = []
    if isinstance(value, dict):
        if isinstance(value.get("reference"), str) and "sha256" in value:
            old = old_context(ctx)
            target = resolve_binding(value, old)
            found.append(binding(target, ctx))
        for child in value.values():
            found.extend(_bound_old_references(child, ctx))
    elif isinstance(value, list):
        for child in value:
            found.extend(_bound_old_references(child, ctx))
    return found


def verify_old_job(identifier, ctx):
    root = Path(ctx["repo"])/"reports"/OLD
    plan = read(root/"EXPERIMENT_PLAN.json")
    from research.adapter_diagnostics_20261017_v2.core import validate_plan as old_validate
    old_validate(plan, ctx["repo"])
    job = next(job for job in plan["jobs"] if job["job_id"] == identifier)
    receipt = read(root/"receipts"/(identifier+".json"))
    if receipt["status"] != "COMPLETE" or receipt["scientific_identity"] != job["scientific_identity"]:
        raise ValueError("Historical job is not complete under its sealed identity: "+identifier)
    for relative, expected in receipt["artifact_hashes"].items():
        resolve_binding(dict(reference="report://"+relative, sha256=expected), old_context(ctx))
    value = read(root/"jobs"/(identifier+".json"))
    if receipt.get("result") != value:
        raise ValueError("Historical receipt/result binding differs: "+identifier)
    references = _bound_old_references(value, ctx)
    return value, dict(job_id=identifier, status="REUSED_BINDINGS_VERIFIED",
                       scientific_identity=job["scientific_identity"],
                       public_result=binding(root/"jobs"/(identifier+".json"), ctx),
                       public_receipt=binding(root/"receipts"/(identifier+".json"), ctx),
                       private_artifacts=references)


def inspect_assets(ctx):
    import numpy as np
    checked = []
    failures = []
    for seed in (92601, 92602):
        for identifier in (f"B03_s{seed}", f"C02_s{seed}"):
            try:
                _, evidence = verify_old_job(identifier, ctx)
                checked.append(evidence)
            except (FileNotFoundError, ValueError, KeyError, StopIteration) as error:
                failures.append(dict(job_id=identifier, status="BLOCKED_ASSET", reason=str(error)))
    feature_rows = []
    for seed in (92601, 92602):
        for scope, split, expected_n in (("A_fit", "fit", 9923), ("A_fit", "A_stop", 43),
                                          ("A60", "fit", 12020), ("A60", "V_select", 43)):
            matches = list((Path(ctx["previous_cache"])/"decoder_features").glob(f"{seed}_{scope}_{split}_*.json"))
            if len(matches) != 1:
                failures.append(dict(seed=seed, scope=scope, split=split, reason="Expected exactly one bound feature manifest"))
                continue
            manifest = matches[0]
            value = read(manifest)
            path = manifest.with_suffix(".npz")
            if sha(path) != value["sha256"]:
                raise ValueError("Historical feature bytes changed")
            shapes = {}
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"phi", "target", "mask", "anchor"}:
                    raise ValueError("Historical feature schema changed")
                for key in archive.files:
                    with archive.zip.open(key+".npy") as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0):
                            shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise ValueError("Unregistered feature NPY header version")
                    shapes[key] = dict(shape=list(shape), dtype=str(dtype), fortran_order=bool(order))
            if shapes["phi"] != dict(shape=[expected_n,48,42],dtype="float32",fortran_order=False):
                raise ValueError("Feature axes differ from registered Jena scope")
            feature_rows.append(dict(seed=seed,fit_scope=scope,split=split,key=value["key"],
                                     headers=shapes,artifact=binding(path,ctx),manifest=binding(manifest,ctx)))
    assets = []
    relative_paths = [".cache/tsfm_peft_practical_controls_v7_20260928/data/jena_trainval.npz",
                      ".cache/informs_broad_validation_20261015_v1/sources/PEMS08.npz",
                      ".cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8/model.safetensors",
                      ".cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8/config.json"]
    for relative in relative_paths:
        path = Path(ctx["asset"])/relative
        if path.is_file():
            assets.append(binding(path,ctx))
        else:
            failures.append(dict(asset=relative,status="BLOCKED_ASSET",reason="Required local asset missing"))
    return dict(status="PASS" if not failures else "PARTIAL_ASSETS",checked_utc=utc(),
                historical_jobs=checked,feature_caches=feature_rows,assets=assets,failures=failures,
                new_model_loads=0,new_F_calls=0,numeric_source_rows_parsed=0,
                M5="LOCAL_ASSETS_SCHEMA_VERIFIED; numeric task/split/metric NOT_REGISTERED; excluded from new fitting",
                source_audit_scope="Byte hashes and NPZ headers; saved prediction rescoring is a separate execution/verification gate")


def audit(ctx):
    value = inspect_assets(ctx)
    target=Path(ctx["report"])/"REUSE_AUDIT.json"
    if target.exists():
        previous=read(target)
        without_time=lambda item:{k:v for k,v in item.items() if k!="checked_utc"}
        if without_time(previous)!=without_time(value):
            raise ValueError("Historical assets differ from the existing audit; preserve the sealed snapshot and create an explicit new version")
        value=previous
    else:
        atomic_json(target,value)
    start=Path(ctx["report"])/"START_STATE.json"
    if not start.exists():
        def git(*args):
            return subprocess.check_output(["git",*args],cwd=ctx["repo"],text=True).strip()
        atomic_json(start,dict(checked_utc=utc(),parent_commit=git("rev-parse","HEAD"),
            origin_main=git("rev-parse","origin/main"),branch=git("branch","--show-current"),
            origin=git("remote","get-url","origin"),limits=ctx["ledger"]["limits"],
            new_campaign_start_unix=ctx["ledger"]["started_unix"],
            historical_package_tree=git("rev-parse","HEAD:reports/"+OLD)))
    return value


def load_modules(ctx):
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _modules
    return _modules(ctx)


def load_task(ctx, cohort, modules=None):
    modules = load_modules(ctx) if modules is None else modules
    if cohort == "jena":
        from research.adapter_diagnostics_20261017_v2.existing_jobs import _jena
        task = _jena(ctx, modules)
    elif cohort == "pems32":
        resume = importlib.import_module("pems_resume")
        task,access,plan = resume.load_task(modules["campaign"], modules["old_cache"])
        ctx.setdefault("task_access",{})[cohort]=access
        ctx.setdefault("parent_task_plans",{})[cohort]=plan
    else:
        raise ValueError("Unregistered new-training cohort: "+cohort)
    return task


def load_foundation(ctx, modules, horizon, device="cuda"):
    from gpu_runtime import FrozenBolt
    foundation = FrozenBolt(Path(ctx["asset"])/modules["legacy"].MODEL_REL,horizon,
                            row_microbatch=128,checkpoint_inputs=False,device=device)
    if foundation.initial_hash != modules["legacy"].F_SHA:
        raise ValueError("Frozen F tensor-state identity mismatch")
    return foundation
