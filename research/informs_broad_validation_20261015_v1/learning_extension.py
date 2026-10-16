"""Immutable search-state migration into a separately registered longer fit."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import data_contract
import fit_base
import training


def state_fingerprint(value):
    """Hash structure plus exact tensor/array bytes, without pickle dependence."""
    def describe(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return {"tensor": str(tensor.dtype), "shape": list(tensor.shape),
                    "sha256": hashlib.sha256(raw).hexdigest()}
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            return {"array": str(array.dtype), "shape": list(array.shape),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        if isinstance(item, dict):
            return {"dict": [[str(type(key)), str(key), describe(val)]
                             for key, val in sorted(item.items(), key=lambda pair: (str(type(pair[0])), str(pair[0])))]}
        if isinstance(item, (list, tuple)):
            return {"sequence": type(item).__name__, "items": [describe(val) for val in item]}
        if isinstance(item, np.generic):
            return item.item()
        return item
    return training.fingerprint(describe(value))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_id(run_id):
    value = str(run_id)
    if not value or Path(value).name != value or value in {".", ".."} or ":" in value or "\\" in value:
        raise ValueError("run_id must be one safe path component")
    return value


def parent_inventory(run_id, cache, public):
    """Only the immutable fit artifacts; unrelated prediction files may be added."""
    result = {}
    names = {"RUN_CONTRACT.json", "SOURCE_CONTRACT.json", "BASE_FIT_RECEIPT.json", "selected_bundle.pt",
             "LEARNING_EXTENSION_RECEIPT.json", "MIGRATION_PROVENANCE.json"}
    for prefix, root in (("cache", Path(cache)), ("public", Path(public))):
        run = root / "runs" / _safe_id(run_id)
        for path in sorted(run.rglob("*")):
            if path.is_file() and (path.parent.name in {"search", "refit"} or path.name in names):
                result[prefix + "/" + path.relative_to(root).as_posix()] = data_contract.sha256_file(path)
    return result


def _base_contract(spec, task, source_sha, foundation):
    axes = dict(L=int(task["L"]), H=int(task["H"]), C=int(task["physical"].shape[1]),
                columns=list(task["columns"]), fold=spec.get("fold", "primary"),
                source_array_fingerprints=task.get("array_sha256", {}))
    return dict(source_sha=source_sha, config_sha=training.fingerprint(dict(spec=spec, axes=axes)),
                data_sha=task["data_sha256"], model_sha=fit_base._model_sha(foundation))


def _stage_contract(base, normalization, fit_end):
    contract = dict(base, scaler_sha=fit_base.scaler_fingerprint(normalization["mean"], normalization["std"]))
    contract["initialization_sha"] = training.fingerprint(dict(
        scaler_sha=contract["scaler_sha"], fit_end=fit_end,
        basis_sha=None if normalization["basis"] is None else data_contract.array_sha256(normalization["basis"]),
        center_sha=None if normalization["pca_center"] is None else data_contract.array_sha256(normalization["pca_center"])))
    return contract


def _validate_parent(parent_id, task, cache, public, foundation):
    private = Path(cache) / "runs" / parent_id
    published = Path(public) / "runs" / parent_id
    source = _json(published / "SOURCE_CONTRACT.json")
    spec = source["spec"]
    if spec["run_id"] != parent_id:
        raise ValueError("Parent run_id/config identity changed")
    current_sha, files = fit_base._source_contract()
    if source["files"] != files or source["source_sha"] != current_sha:
        raise ValueError("Parent source differs from current six-file source; automatic patch whitelist is forbidden")
    expected = _base_contract(spec, task, current_sha, foundation)
    actual = _json(private / "RUN_CONTRACT.json")
    for key in expected:
        if actual.get(key) != expected[key] or source.get(key) != expected[key]:
            raise ValueError("Parent recipe/config/data/model contract changed: " + key)
    receipt = _json(published / "BASE_FIT_RECEIPT.json")
    if receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != parent_id:
        raise ValueError("Only a completed parent screen may be extended")
    artifact = Path(cache) / receipt["artifact_cache_relative_path"]
    if artifact.resolve() != (private / "selected_bundle.pt").resolve():
        raise ValueError("Parent artifact path differs from registered run")
    if data_contract.sha256_file(artifact) != receipt["artifact_sha256"]:
        raise ValueError("Parent selected artifact bytes changed")
    bundle = torch.load(artifact, map_location="cpu", weights_only=False)
    if bundle["spec"] != spec or bundle["contract"]["config_sha"] != expected["config_sha"]:
        raise ValueError("Parent artifact recipe/config identity changed")
    checkpoints = sorted((private / "search").glob("checkpoint_*.pt"))
    if not checkpoints:
        raise ValueError("Parent has no complete optimizer search checkpoint")
    path = checkpoints[-1]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "sampler", "torch_rng", "cuda_rng", "contract", "updates", "extra"}
    if not required.issubset(checkpoint):
        raise ValueError("Full optimizer/RNG/sampler checkpoint is required; fixed PCA is not a trained continuation")
    bounds = data_contract.split_ranges(task, spec.get("fold", "primary"))
    fit_end = bounds["A_fit"][1]
    model, norm = fit_base.create_model(spec, task, fit_end, foundation)
    try:
        stage = _stage_contract(expected, norm, fit_end)
        if checkpoint["contract"] != stage or bundle["search_summary"]["contract"] != stage:
            raise ValueError("Parent search scaler/initialization/source contract changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=float(spec.get("lr", .001)), weight_decay=0)
        if checkpoint["optimizer"]["param_groups"] != optimizer.state_dict()["param_groups"]:
            raise ValueError("Parent optimizer LR/effective recipe changed")
        optimizer.load_state_dict(checkpoint["optimizer"])
    finally:
        if callable(getattr(model, "close", None)):
            model.close()
        if foundation is not None and callable(getattr(foundation, "assert_frozen_unchanged", None)):
            foundation.assert_frozen_unchanged()
    old_epochs = [float(epoch) for epoch in spec.get("epochs", [.25, 1., 3.])]
    completed = checkpoint["extra"]["completed_epochs"]
    if completed != [0.] + old_epochs:
        raise ValueError("Parent search checkpoint does not end at every completed registered boundary")
    legal = data_contract.origins((0, fit_end), task["L"], task["H"], 1)
    exposures = int(math.ceil(old_epochs[-1] * len(legal)))
    sampler = training.PermutationStream(legal, int(spec.get("seed", 0)))
    sampler.draw(exposures)
    if sampler.state_dict() != checkpoint["sampler"]:
        raise ValueError("Parent sampler/RNG/exposure state differs from exact recipe")
    updates = 0
    previous = 0
    for epoch in old_epochs:
        current = int(math.ceil(epoch * len(legal)))
        updates += int(math.ceil((current - previous) / int(spec.get("effective_batch", 16))))
        previous = current
    if checkpoint["updates"] != updates or receipt["search_updates"] != updates:
        raise ValueError("Parent optimizer update count is inconsistent with segment boundaries")
    summary = _json(published / "search" / "RECEIPT.json")
    if (summary["contract"] != stage or summary["exposures"] != exposures
            or summary["updates"] != updates or summary["completed_epochs"] != completed):
        raise ValueError("Parent published search receipt differs from checkpoint")
    curves = checkpoint["extra"]["curves"]
    if [row["epoch"] for row in curves] != completed or curves[-1]["exposures"] != exposures:
        raise ValueError("Parent search curve/exposure boundary changed")
    if not torch.is_tensor(checkpoint["torch_rng"]) or checkpoint["torch_rng"].dtype != torch.uint8:
        raise ValueError("Parent torch RNG state is missing or invalid")
    return dict(spec=spec, base_contract=expected, stage_contract=stage, normalization=norm,
                checkpoint=checkpoint, path=path, receipt=receipt, inventory=parent_inventory(parent_id, cache, public))


def _payload_fingerprints(checkpoint):
    return {key: state_fingerprint(value) for key, value in checkpoint.items() if key != "contract"}


def extend_fit(parent_run_id, new_run_id, epochs, task, cache, public, foundation, deadline):
    """Continue a completed A50 search under a new identity, then freshly refit A60.

    The caller must register the new ID before calling. This module never edits a
    registry and performs no prediction on M/V/T. A budget interruption can be
    resumed by repeating the exact same call and new ID.
    """
    parent_id, new_id = _safe_id(parent_run_id), _safe_id(new_run_id)
    if parent_id == new_id:
        raise ValueError("Continuation requires a distinct registered ID")
    requested = [float(epoch) for epoch in epochs]
    if not requested or requested != sorted(set(requested)) or any(not math.isfinite(e) or e <= 0 for e in requested):
        raise ValueError("Positive increasing finite epoch schedule required")
    parent = _validate_parent(parent_id, task, cache, public, foundation)
    old_epochs = [float(epoch) for epoch in parent["spec"].get("epochs", [.25, 1., 3.])]
    if len(requested) <= len(old_epochs) or requested[:len(old_epochs)] != old_epochs:
        raise ValueError("Original segment boundaries must be the unchanged prefix of the longer schedule")
    module_sha = data_contract.sha256_file(Path(__file__))
    spec = copy.deepcopy(parent["spec"])
    spec.update(run_id=new_id, epochs=requested, stage_label="SEARCH_CONTINUATION_FRESH_A60_REFIT",
                learning_extension_parent=dict(run_id=parent_id,
                    original_config_sha=parent["base_contract"]["config_sha"],
                    original_search_checkpoint_sha256=data_contract.sha256_file(parent["path"]),
                    extension_source_sha256=module_sha))
    new_contract = _base_contract(spec, task, parent["base_contract"]["source_sha"], foundation)
    stage_contract = _stage_contract(new_contract, parent["normalization"], parent["receipt"]["search_fit_end"])
    private = Path(cache) / "runs" / new_id
    published = Path(public) / "runs" / new_id
    provenance_path = published / "MIGRATION_PROVENANCE.json"
    identity = dict(parent_run_id=parent_id, new_run_id=new_id, old_epochs=old_epochs, new_epochs=requested,
                    original_checkpoint_cache_relative_path=parent["path"].relative_to(Path(cache)).as_posix(),
                    original_checkpoint_sha256=data_contract.sha256_file(parent["path"]),
                    original_contract=parent["stage_contract"], new_contract=stage_contract,
                    parent_inventory=parent["inventory"], original_payload_sha256=_payload_fingerprints(parent["checkpoint"]),
                    extension_source_sha256=module_sha)
    migrated = private / "search" / "checkpoint_000000.pt"
    if provenance_path.exists():
        provenance = _json(provenance_path)
        if provenance["identity"] != identity:
            raise ValueError("Existing migration identity/parent/recipe changed; use a new registered ID")
        if not migrated.is_file() or data_contract.sha256_file(migrated) != provenance["migrated_checkpoint_sha256"]:
            raise ValueError("Immutable migrated checkpoint changed")
        restored = torch.load(migrated, map_location="cpu", weights_only=False)
        if restored["contract"] != stage_contract or _payload_fingerprints(restored) != identity["original_payload_sha256"]:
            raise ValueError("Migrated optimizer/model/RNG/sampler payload changed")
    else:
        if (private.exists() and any(private.iterdir())) or (published.exists() and any(published.iterdir())):
            raise ValueError("New registered ID already contains unrelated or incomplete migration files")
        cloned = copy.deepcopy(parent["checkpoint"])
        cloned["contract"] = stage_contract
        if _payload_fingerprints(cloned) != identity["original_payload_sha256"]:
            raise AssertionError("Model/optimizer/RNG/sampler payload changed during clone")
        fit_base._torch_new(migrated, cloned)
        provenance = dict(identity=identity, created_unix=time.time(), payload_unchanged=True,
                          fresh_A60_refit=True, copied_old_A60_refit=False,
                          source_policy="EXACT_SIX_FILES_NO_AUTOMATIC_PATCH_EXCEPTION",
                          migrated_checkpoint_sha256=data_contract.sha256_file(migrated),
                          migration_only_changes_checkpoint_contract=True,
                          original_payload_authenticity="observed_parent_bytes_plus_contract_checks; no preexisting external last-state signature",
                          interrupted_segment_clipping_diagnostics="Existing engine omits partial segment clipping counts on deadline; optimizer/RNG/sampler remain restartable.")
        fit_base._json_new(provenance_path, provenance)
    try:
        receipt = fit_base.fit_base(spec, task, cache, public, foundation, deadline)
    finally:
        if parent_inventory(parent_id, cache, public) != parent["inventory"]:
            raise RuntimeError("Original parent fit artifacts changed during extension")
    latest = sorted((private / "search").glob("checkpoint_*.pt"))[-1]
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    inherited_updates = int(parent["checkpoint"]["updates"])
    inclusive_search_updates = int(checkpoint["updates"])
    inherited_f = int(parent["checkpoint"]["extra"].get("total_streams", 0))
    inclusive_f = int(checkpoint["extra"].get("total_streams", 0))
    refit_updates = int(receipt.get("refit_updates", receipt.get("updates", 0) if receipt.get("phase") == "refit" else 0))
    refit_f = int(receipt.get("F_logical_streams", 0)) - inclusive_f if receipt["status"] == "COMPLETE_VALID" else int(receipt.get("F_total_logical_streams", 0) if receipt.get("phase") == "refit" else 0)
    interruption_events = []
    for run_id in (parent_id, new_id):
        for stage in ("search", "refit"):
            for path in sorted((Path(public) / "runs" / run_id / stage).glob("event_*.json")):
                if _json(path).get("status") == "BUDGET_LIMIT":
                    interruption_events.append(path.relative_to(Path(public)).as_posix())
    accounting = dict(inherited_search_updates=inherited_updates,
                      inclusive_search_updates=inclusive_search_updates,
                      new_search_updates=inclusive_search_updates - inherited_updates,
                      new_refit_updates=refit_updates,
                      new_total_updates=inclusive_search_updates - inherited_updates + refit_updates,
                      preserved_parent_refit_updates=int(parent["receipt"]["refit_updates"]),
                      inherited_search_F_logical_streams=inherited_f,
                      inclusive_search_F_logical_streams=inclusive_f,
                      new_search_F_logical_streams=inclusive_f - inherited_f,
                      new_refit_F_logical_streams=refit_f,
                      new_total_F_logical_streams=inclusive_f - inherited_f + refit_f,
                      parent_refit_not_reused=True,
                      budget_interruption_events=interruption_events,
                      clipping_diagnostics_complete=not interruption_events)
    result = dict(receipt, extension_accounting=accounting, extension_provenance=provenance,
                  extension_provenance_public_relative_path=provenance_path.relative_to(Path(public)).as_posix())
    if receipt["status"] == "COMPLETE_VALID":
        extension_receipt = published / "LEARNING_EXTENSION_RECEIPT.json"
        if extension_receipt.exists():
            previous = _json(extension_receipt)
            if previous["artifact_sha256"] != result["artifact_sha256"] or previous["extension_accounting"] != accounting:
                raise ValueError("Complete extension receipt identity/accounting changed")
        else:
            fit_base._json_new(extension_receipt, result)
    else:
        fit_base._json_new(published / ("LEARNING_EXTENSION_ATTEMPT_" + str(time.time_ns()) + ".json"), result)
    return result
