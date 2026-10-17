"""Copy-only PeMS C32 continuation for the separately authorized 2026-10-17 run.

This module never calls the historical runner's admission, execute, or queue.
The caller owns process/RSS guards and the new finite campaign deadline.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time


NAME = "priority_pems_subset_20261016_v1"
FAMILIES = ("B0", "B2")
EXPECTED_SHA = {
    "B0": "62f058cd9806335455af140d31bcd2b8aaacf454dae6d10b79740c8876642e6c",
    "B2": "3a9b6bbda6768549ad48ae5011ca3da6d6f14ec76fa75b7c9cec3b9138617c41",
}
RUN_ID = "postT_cpu_pems08_E1_SUBSET_{family}_k8_s92601_C32_priority_v1"
BOUNDS = dict(A_fit=[0, 8928], A_stop=[8928, 10713], M_cal=[10713, 12499],
              V_select=[12499, 14284], T_final=[14284, 17856])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _module(path, label):
    spec = importlib.util.spec_from_file_location("mechanism_pems_" + label, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _family(family):
    if family not in FAMILIES:
        raise ValueError("Only original PeMS C32 B0/B2 conditions are authorized")
    return RUN_ID.format(family=family)


def _original(public_campaign, old_cache_campaign, family):
    identifier = _family(family)
    public = Path(public_campaign).resolve()
    cache = Path(old_cache_campaign).resolve()
    source = read(public / NAME / "runs" / identifier / "SOURCE_CONTRACT.json")
    checkpoint = cache / NAME / "runs" / identifier / "search" / "checkpoint_000003.pt"
    if sha(checkpoint) != EXPECTED_SHA[family]:
        raise ValueError("Original terminal checkpoint SHA changed")
    return identifier, source, checkpoint


def load_task(public_campaign, old_cache_campaign):
    """Read the original pinned pre-T prefix and same hash-ordered C32 subset."""
    public = Path(public_campaign).resolve()
    sys.path.insert(0, str(public))
    import numpy as np
    import data_contract as data
    if Path(data.__file__).resolve() != public / "data_contract.py":
        raise ValueError("Scientific data module import root changed")
    subset = _module(public / NAME / "subset_minimum_runner.py", "subset")
    plan_path = public / NAME / "PLAN.json"
    plan = read(plan_path)
    subset.require_sealed(plan)
    if plan["original_primary_split_ranges"] != BOUNDS or plan["materialized_prefix_rows"] != 14284:
        raise ValueError("Original data boundaries changed")
    for name, digest in plan["dependencies"].items():
        if sha(public / name) != digest:
            raise ValueError("Historical dependency changed: " + name)
    libraries = {key: _module(public / name, key) for key, name in subset.DEPENDENCY_PATHS.items()}
    parent, access = libraries["pems"].load_prefix(Path(old_cache_campaign), plan["parent_task_contract"],
                                                 libraries["priority"], libraries["primitive"])
    task = subset.subset_task(parent, plan, dict(np=np, data=data))
    for family in FAMILIES:
        _, source, _ = _original(public, old_cache_campaign, family)
        if task["data_sha256"] != source["data_sha"] or task["subset_identity_payload"] != source["subset_identity_payload"]:
            raise ValueError("Actual subset data differs from checkpoint source")
    access.update(actual_C=32, K=8, original_plan_sha256=sha(plan_path),
                  derived_subset_data_sha256=task["data_sha256"],
                  actual_subset_prefix_array_sha256=task["array_sha256"],
                  protected_T_numeric_parse_count=0)
    return task, access, plan


def inspect_and_copy(public_campaign, old_cache_campaign, new_cache):
    """Inspect checkpoint payloads on SHA-verified copies; construct no model."""
    import torch
    destination = Path(new_cache).resolve() / "PEMS_checkpoint_copies"
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for family in FAMILIES:
        identifier, source, original = _original(public_campaign, old_cache_campaign, family)
        copied = destination / (family + "_checkpoint_000003.pt")
        shutil.copyfile(original, copied)
        if sha(copied) != EXPECTED_SHA[family]:
            raise ValueError("Checkpoint copy did not preserve bytes")
        bundle = torch.load(copied, map_location="cpu", weights_only=False)
        required = {"model", "optimizer", "sampler", "torch_rng", "cuda_rng", "contract", "updates", "extra"}
        if not required.issubset(bundle) or bundle["cuda_rng"]:
            raise ValueError("Complete CPU restore payload missing")
        sampler = bundle["sampler"]
        if not {"order", "position", "exposures", "epochs_completed", "rng", "origins_sha"}.issubset(sampler):
            raise ValueError("Complete permutation stream state missing")
        if bundle["extra"]["completed_epochs"] != [0., .25, 1.]:
            raise ValueError("Unexpected original completed endpoint history")
        if len(sampler["order"]) != 8405 or bundle["updates"] != {"B0": 847, "B2": 846}[family]:
            raise ValueError("Original training progress differs")
        if any(bundle["contract"][key] != source[key] for key in ("source_sha", "config_sha", "data_sha", "model_sha")):
            raise ValueError("Checkpoint/source contract mismatch")
        row = dict(family=family, run_id=identifier, original_sha256=sha(original), copy_sha256=sha(copied),
                   copy_path=str(copied), original_bytes=original.stat().st_size,
                   payload_keys=sorted(bundle), optimizer_state_entries=len(bundle["optimizer"]["state"]),
                   sampler_keys=sorted(sampler), position=sampler["position"], exposures=sampler["exposures"],
                   updates=bundle["updates"], completed_epochs=bundle["extra"]["completed_epochs"],
                   torch_rng_bytes=bundle["torch_rng"].numel(), cuda_rng_entries=0,
                   independent_numpy_rng_use="sampler owns saved PCG64 state; training has no global numpy RNG draws",
                   source_contract=source, restoration_executed=False)
        rows.append(row)
        if sha(original) != EXPECTED_SHA[family]:
            raise ValueError("Original checkpoint changed during copy/inspection")
    result = dict(status="PAYLOAD_COMPLETE_RESTORE_TEST_PENDING", rows=rows,
                  old_deadline_preserved=True, old_admission_called=False,
                  protected_T_numeric_parse_count=0, model_constructions=0)
    write_new(destination / "INSPECTION.json", result)
    return result


def _scientific(public_campaign):
    public = Path(public_campaign).resolve()
    sys.path.insert(0, str(public))
    import torch
    import fit_base as fit
    import training
    import data_contract as data
    for module in (fit, training, data):
        if Path(module.__file__).resolve().parent != public:
            raise ValueError("Scientific module import root changed")
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU continuation initialized CUDA")
    return torch, fit, training, data


def _contract_check(source, task, foundation, fit):
    if fit._source_contract()[0] != source["source_sha"] or fit._model_sha(foundation) != source["model_sha"]:
        raise ValueError("Historical scientific source/foundation identity changed")
    if task["data_sha256"] != source["data_sha"] or task["physical"].shape != (14284, 32):
        raise ValueError("Exact original pre-T task required")
    spec = source["spec"]
    if (spec["family"] not in FAMILIES or spec["seed"] != 92601 or spec["latent_k"] != 8
            or spec["device"] != "cpu" or spec["microbatch"] != 4 or spec["effective_batch"] != 16
            or spec["epochs"] != [.25, 1., 3.] or task["L"] != 512 or task["H"] != 12):
        raise ValueError("Original PeMS recipe required")
    return copy.deepcopy(spec)


def _tree_equal(a, b, torch):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_tree_equal(a[k], b[k], torch) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_tree_equal(x, y, torch) for x, y in zip(a, b))
    return a == b


def audit_one(family, task, foundation, public_campaign, old_cache_campaign, copied_checkpoint, output_dir):
    """Compare retained in-memory next update against serialized/restored next update.

    Both start at the original partial state. This verifies a fresh save boundary;
    it does not claim access to an uninterrupted historical process.
    """
    torch, fit, training, data = _scientific(public_campaign)
    identifier, source, original = _original(public_campaign, old_cache_campaign, family)
    if sha(copied_checkpoint) != EXPECTED_SHA[family]:
        raise ValueError("Only the exact audited checkpoint copy is permitted")
    spec = _contract_check(source, task, foundation, fit)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(copied_checkpoint, map_location="cpu", weights_only=False)
    legal = data.origins((0, 8928), 512, 12, 1)
    before = time.perf_counter()
    states = []
    batches = []
    normalization = None
    boundary_path = output / "boundary_copy.pt"
    for branch in ("retained_in_memory", "serialized_restore"):
        model, norm = fit.create_model(spec, task, 8928, foundation)
        normalization = norm
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
        sampler = training.PermutationStream(legal, 92601)
        current = checkpoint if branch == "retained_in_memory" else torch.load(boundary_path, map_location="cpu", weights_only=False)
        updates = training.restore_bundle(current, model, optimizer, sampler, checkpoint["contract"])
        if branch == "retained_in_memory":
            boundary = training.checkpoint_bundle(model, optimizer, sampler, checkpoint["contract"], updates, copy.deepcopy(checkpoint["extra"]))
            with boundary_path.open("xb") as handle:
                torch.save(boundary, handle)
        preview = training.PermutationStream(legal, 92601)
        preview.load_state_dict(sampler.state_dict())
        batches.append(preview.draw(16).tolist())
        prefix = dict(task, physical=task["physical"][:10713], observed=task["observed"][:10713])
        windows = training.FastWindows(prefix, norm["mean"], norm["std"], "cpu")
        target = sampler.exposures + 16
        result = training.train_segment(model, windows, legal, sampler, optimizer, target,
                    lambda x, k: fit.predict_model(model, x, foundation, family, k), effective_batch=16,
                    microbatch=4, starting_updates=updates, starting_k_exposures=checkpoint["extra"]["k_exposures"])
        if result["status"] != "SEGMENT_COMPLETE" or result["updates"] != updates + 1:
            raise ValueError("Audit did not perform exactly one optimizer update")
        states.append(copy.deepcopy(training.checkpoint_bundle(model, optimizer, sampler, checkpoint["contract"], result["updates"])))
        del windows, optimizer, sampler, model
    checks = {key: _tree_equal(states[0][key], states[1][key], torch)
              for key in ("model", "optimizer", "sampler", "torch_rng", "cuda_rng", "updates")}
    checks["actual_next_batch_origins"] = batches[0] == batches[1]
    checks["scaler_contract"] = fit.scaler_fingerprint(normalization["mean"], normalization["std"]) == checkpoint["contract"]["scaler_sha"]
    checks["original_checkpoint_unchanged"] = sha(original) == EXPECTED_SHA[family]
    foundation.assert_frozen_unchanged()
    result = dict(status="PASS_EXACT_NEXT_UPDATE" if all(checks.values()) else "FAILED_RESTORE_EQUIVALENCE",
                  family=family, run_id=identifier, checks=checks, original_sha256=sha(original),
                  source_copy_sha256=sha(copied_checkpoint), boundary_copy_sha256=sha(boundary_path),
                  starting_updates=checkpoint["updates"], ending_updates=states[0]["updates"],
                  starting_exposures=checkpoint["sampler"]["exposures"], ending_exposures=states[0]["sampler"]["exposures"],
                  actual_audit_optimizer_updates=2, actual_audit_sample_exposures=32,
                  next_batch_origins=batches[0], model_weights_exact_equal=checks["model"],
                  optimizer_exact_equal=checks["optimizer"], RNG_exact_equal=checks["torch_rng"] and checks["sampler"],
                  sample_position_exact_equal=states[0]["sampler"]["position"] == states[1]["sampler"]["position"],
                  historical_uninterrupted_process_available=False,
                  equivalence_scope="same saved partial state; retained in-memory branch versus serialized/restored branch at a new save boundary",
                  actual_CPU_threads=torch.get_num_threads(), GPU_initialized=torch.cuda.is_initialized(),
                  seconds=time.perf_counter()-before, protected_T_numeric_parse_count=0)
    write_new(output / "RESULT.json", result)
    if not all(checks.values()):
        raise ValueError("Exact next-update restoration equivalence failed")
    return result


def resume_stages(family, task, foundation, public_campaign, old_cache_campaign, new_private_run,
                  new_public_run, deadline, audit_result):
    """Continue one copied search to 3 epochs, then fresh A60 refit selected length.

    Returns selected adapter state/normalization, epoch, histories and lineage.
    No V evaluation or model queue is launched here.
    """
    torch, fit, training, _ = _scientific(public_campaign)
    identifier, source, original = _original(public_campaign, old_cache_campaign, family)
    spec = _contract_check(source, task, foundation, fit)
    if (audit_result.get("status") != "PASS_EXACT_NEXT_UPDATE" or audit_result.get("family") != family
            or audit_result.get("original_sha256") != EXPECTED_SHA[family] or not all(audit_result["checks"].values())):
        raise ValueError("Passing exact-family copy restoration test required")
    if not isinstance(deadline, (float, int)) or not math.isfinite(deadline) or time.time() >= deadline:
        raise ValueError("Caller must supply a live new-campaign deadline")
    private, public = Path(new_private_run).resolve(), Path(new_public_run).resolve()
    old_private = original.parent.parent.resolve()
    old_public = (Path(public_campaign) / NAME / "runs" / identifier).resolve()
    if private == old_private or private.is_relative_to(old_private) or public == old_public or public.is_relative_to(old_public):
        raise ValueError("Continuation output must be outside original run")
    private.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=False)
    (private / "search").mkdir()
    (public / "search").mkdir()
    copies = []
    for path in sorted(original.parent.glob("checkpoint_*.pt")):
        destination = private / "search" / path.name
        shutil.copyfile(path, destination)
        copies.append(dict(name=path.name, original_sha256=sha(path), copy_sha256=sha(destination)))
        if copies[-1]["original_sha256"] != copies[-1]["copy_sha256"]:
            raise ValueError("Search copy bytes differ")
    for path in sorted((old_public / "search").glob("event_*.json")):
        shutil.copyfile(path, public / "search" / path.name)
    contract = {key: source[key] for key in ("source_sha", "config_sha", "data_sha", "model_sha")}
    lineage = dict(schema="PEMS_COPY_CONTINUATION_v1", original_run_id=identifier,
                   original_terminal_sha256=EXPECTED_SHA[family], copied_checkpoints=copies,
                   historical_source_contract_sha256=sha(old_public / "SOURCE_CONTRACT.json"),
                   original_recipe_reused=True, original_admission_or_deadline_modified=False,
                   new_campaign_deadline_unix=deadline, restoration_test=audit_result,
                   audit_updates_excluded_from_fit=True, old_queue_dispatched=False)
    write_new(public / "PEMS_RESUME_LINEAGE.json", lineage)
    search = fit._stage(spec, task, 8928, BOUNDS["A_stop"], [.25, 1., 3.], private / "search",
                        public / "search", foundation, deadline, contract)
    write_new(public / "SEARCH_SUMMARY.json", search["summary"])
    write_new(public / "SEARCH_CURVES.json", search["curves"])
    if search["summary"]["status"] != "SEGMENT_COMPLETE":
        return dict(status="PARTIAL", phase="search", search=search, lineage=lineage)
    epoch = search["summary"]["selected_epoch"]
    if epoch not in (0., .25, 1., 3.) or search["summary"]["completed_epochs"] != [0., .25, 1., 3.]:
        raise ValueError("Original full search endpoints/selection required")
    if search["summary"]["updates"] != 1577 or search["summary"]["exposures"] != 25215:
        raise ValueError("Original segmented search update/exposure count differs")
    refit = fit._stage(spec, task, 10713, None, [epoch] if epoch else [], private / "refit",
                       public / "refit", foundation, deadline, contract)
    write_new(public / "REFIT_SUMMARY.json", refit["summary"])
    write_new(public / "REFIT_CURVES.json", refit["curves"])
    if refit["summary"]["status"] != "SEGMENT_COMPLETE":
        return dict(status="PARTIAL", phase="refit", search=search, refit=refit, lineage=lineage)
    exposures = math.ceil(epoch * 10190)
    if refit["summary"]["exposures"] != exposures or refit["summary"]["updates"] != math.ceil(exposures/16):
        raise ValueError("Original selected-length fresh refit accounting differs")
    if sha(original) != EXPECTED_SHA[family]:
        raise ValueError("Original terminal checkpoint changed during continuation")
    return dict(status="COMPLETE_STAGES", model=refit["model"], normalization=refit["normalization"],
                selected_epoch=epoch, search=search, refit=refit, lineage=lineage,
                spec=spec, contract=refit["summary"]["contract"], original_terminal_unchanged=True)


if __name__ == "__main__":
    print(json.dumps(dict(status="IMPORTABLE_CALLER_CONTROLLED_API", training_launched=False,
                          old_queue_dispatched=False, input_reads=0)))
