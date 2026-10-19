"""A_stop-only search, complete-state continuation and fresh A60 refitting."""
import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import torch

from research.b2_revin_20261018_v1.training_jobs import _state_fingerprint
from . import common, models


def job_spec(dataset, mode, seed, device="cuda"):
    if dataset not in common.EXPECTED or mode not in ("S", "A", "B") or seed not in common.SEEDS:
        raise ValueError("Registered dataset, mode and seed required")
    endpoints = common.EXTENDED_EPOCHS if dataset == "jena" else common.PEMS_EXTENDED_EPOCHS
    return dict(common.RECIPE, run_id=f"{mode}_{dataset}_s{seed}", dataset=dataset,
                family="B2", init_kind="PCA", latent_k=common.EXPECTED[dataset]["K"],
                seed=int(seed), device=device, fold="primary", core_dtype="float32", revin=False,
                scientific_identity=dict(method=common.NAME, raw_context_mode=mode,
                    raw_context_scale=2, raw_context_schedule="native",
                    potential_endpoints=list(endpoints), selection_split="A_stop"))


def _spec(ctx, mode, seed):
    task = ctx["task"]
    spec = job_spec(task["dataset"], mode, seed, ctx.get("device", "cuda"))
    spec["latent_k"] = int(task.get("K", spec["latent_k"]))
    return spec


def _factory_context(ctx):
    sources = dict(ctx.get("scientific_source_hashes", {}))
    for name, value in common.model_sources(ctx["repo"]).items():
        if name in sources and sources[name] != value:
            raise ValueError("Registered scientific source changed: " + name)
        sources[name] = value
    return dict(ctx, scientific_source_hashes=sources)


def _axes(task):
    return dict(L=int(task["L"]), H=int(task["H"]), C=int(task["physical"].shape[1]),
                columns=list(task["columns"]), fold="primary",
                source_array_fingerprints=task.get("array_sha256", {}))


def _contract(ctx, spec):
    fit, training = ctx["modules"]["fit"], ctx["modules"]["training"]
    source_sha, source_files = fit._source_contract()
    return dict(source_sha=source_sha,
                config_sha=training.fingerprint(dict(spec=spec, axes=_axes(ctx["task"]))),
                data_sha=ctx["task"]["data_sha256"], model_sha=fit._model_sha(ctx["foundation"])), source_files


def _stage_contract(ctx, base, norm, fit_end):
    fit, training, data = (ctx["modules"][name] for name in ("fit", "training", "data"))
    active = dict(base, scaler_sha=fit.scaler_fingerprint(norm["mean"], norm["std"]))
    active["initialization_sha"] = training.fingerprint(dict(scaler_sha=active["scaler_sha"], fit_end=fit_end,
        basis_sha=data.array_sha256(norm["basis"]), center_sha=data.array_sha256(norm["pca_center"])))
    return active


def _locations(ctx, spec, cap):
    relative = Path("runs") / spec["run_id"]
    private, public = Path(ctx["cache"]) / relative, Path(ctx["report"]) / relative
    return private, public, private / f"cap{cap}", public / f"cap{cap}"


def _endpoints(dataset, endpoints):
    values = tuple(map(float, endpoints))
    if values and values[0] == 0:
        values = values[1:]
    allowed = (common.INITIAL_EPOCHS, common.EXTENDED_EPOCHS) if dataset == "jena" else (
        common.EXTENDED_EPOCHS, common.PEMS_EXTENDED_EPOCHS)
    if values not in allowed:
        raise ValueError("Only the registered dataset endpoint schedules are allowed")
    return list(values)


def _deadline(ctx):
    return min(ctx["deadline"], ctx.get("job_deadline", ctx["deadline"]))


def _load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _segment_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]


@contextmanager
def _observe_training_segments(ctx, spec, phase, private_dir, public_dir, endpoints):
    """Observe the original training call without changing its arguments/result."""
    module = ctx["modules"]["training"]
    original = module.train_segment
    private_path, public_path = (Path(directory) / "TRAIN_SEGMENTS.jsonl" for directory in (private_dir, public_dir))
    rows, public_rows = _segment_rows(private_path), _segment_rows(public_path)
    if rows[:len(public_rows)] != public_rows:
        raise ValueError("Private/public TRAIN segment history differs")

    def append(path, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    for row in rows[len(public_rows):]:
        append(public_path, row)

    def observe(*args, **kwargs):
        def argument(name, index):
            return kwargs[name] if name in kwargs else args[index]

        sampler = argument("sampler", 3)
        count = len(argument("train_origins", 2))
        target = int(argument("end_exposures", 5))
        start_exposures = int(sampler.exposures)
        start_updates = int(kwargs.get("starting_updates", 0))
        endpoint = next((float(epoch) for epoch in endpoints if math.ceil(float(epoch) * count) == target),
                        target / count)
        identity = dict(run_id=spec["run_id"], phase=phase, start_exposures=start_exposures,
                        target_exposures=target, starting_updates=start_updates)
        started = time.perf_counter()
        result, error = None, None
        try:
            result = original(*args, **kwargs)
            return result
        except BaseException as caught:
            error = caught
            raise
        finally:
            status = result["status"] if result is not None else (
                "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED")
            segment_id = common.fingerprint(identity)
            record_id = common.fingerprint(dict(segment_id=segment_id, status=status))
            if not any(row["record_id"] == record_id for row in rows):
                source = result or {}
                row = dict(identity, segment_id=segment_id, record_id=record_id, status=status,
                    dataset=spec["dataset"], mode=spec["scientific_identity"]["raw_context_mode"], seed=spec["seed"],
                    endpoint_epoch=endpoint, train_origins=count, end_exposures=int(sampler.exposures),
                    updates=source.get("updates"),
                    optimizer_updates_this_segment=source.get("optimizer_updates_this_segment",
                        source["updates"] - start_updates if "updates" in source else None),
                    mean_observed_training_batch_loss=source.get("mean_observed_training_batch_loss"),
                    gradient_norm_min=source.get("gradient_norm_min"), gradient_norm_max=source.get("gradient_norm_max"),
                    segment_seconds=source.get("segment_seconds", time.perf_counter() - started),
                    loss_interpretation="Mean mini-batch macro loss DURING this segment; not full-dataset MSE or a current-checkpoint score",
                    error_type=type(error).__name__ if error is not None else None)
                append(private_path, row)
                rows.append(row)
                append(public_path, row)

    module.train_segment = observe
    try:
        yield rows
    finally:
        module.train_segment = original


def _write_result(ctx, result):
    _, _, private, public = _locations(ctx, result["spec"], result["cap"])
    if result["status"] == "COMPLETE":
        common.sealed_json(private / "SEARCH_RESULT.json", result)
        common.sealed_json(public / "SEARCH_RESULT.json", {key: value for key, value in result.items()
            if key not in ("last_checkpoint", "checkpoint_path", "parent_search_path")})
    else:
        common.atomic_json(private / "PARTIAL_RESULT.json", result)
        common.atomic_json(public / "PARTIAL_RESULT.json", {key: value for key, value in result.items()
            if key not in ("last_checkpoint", "checkpoint_path", "parent_search_path")})
    return result


def search(ctx, mode, seed, endpoints):
    """Extend the same search directory and contract with optimizer/sampler/RNG."""
    task, modules, foundation = ctx["task"], ctx["modules"], ctx["foundation"]
    fit = modules["fit"]
    spec = _spec(ctx, mode, seed)
    epochs = _endpoints(task["dataset"], endpoints)
    cap = int(epochs[-1])
    private, public, cap_private, _ = _locations(ctx, spec, cap)
    with models.model_factory(_factory_context(ctx)) as create:
        base, source_files = _contract(ctx, spec)
        stored = cap_private / "SEARCH_RESULT.json"
        if stored.exists():
            result = common.read(stored)
            if result["spec"] != spec or result["contract"] != base:
                raise ValueError("Completed search scientific contract changed")
            common.resolve_binding(result["checkpoint_binding"], ctx)
            return result
        fit_end = int(task["split_ranges"]["A_fit"][1])
        model, norm = create(spec, task, fit_end, foundation)
        initial_sha = _state_fingerprint(model.state_dict())
        active = _stage_contract(ctx, base, norm, fit_end)
        latest = sorted((private / "search").glob("checkpoint_*.pt"))
        if latest and _load_checkpoint(latest[-1])["contract"] != active:
            raise ValueError("Complete search resume contract, including initialization, changed")
        started = time.perf_counter()
        with _observe_training_segments(ctx, spec, "search", private / "search", public / "search", epochs) as segments:
            stage = fit._stage_body(spec, task, fit_end, tuple(task["split_ranges"]["A_stop"]), epochs,
                private / "search", public / "search", foundation, _deadline(ctx), base, model, norm)
        last = sorted((private / "search").glob("checkpoint_*.pt"))[-1]
        complete = stage["summary"]["status"] == "SEGMENT_COMPLETE"
        result = dict(status="COMPLETE" if complete else "PARTIAL_CHECKPOINTED", spec=spec,
            contract=base, source_files=source_files, summary=stage["summary"], curves=stage["curves"],
            selected_epoch=stage["summary"]["selected_epoch"], cap=cap,
            initial_base_state_sha256=initial_sha, checkpoint_binding=common.binding(last, ctx),
            last_checkpoint=str(last), checkpoint_path=str(last),
            training_segments=segments,
            selection_split="A_stop", V_used=False, session_seconds=time.perf_counter() - started,
            baseline_work="EXTRA_BASELINE_RECOMPUTE" if mode == "S" and not (
                private / "MIGRATION_PROOF.json").exists() else "REGISTERED_SEARCH")
        proof = private / "MIGRATION_PROOF.json"
        if proof.exists():
            result["migration_proof_binding"] = common.binding(proof, ctx)
            result["training_segments_unavailable_reason"] = "Historical S segments before migration were not retained; listed rows cover the new continuation"
        return _write_result(ctx, result)


def strict_last_best(curves, cap):
    """The cap must be strictly below every preceding FP64 A_stop score."""
    values = [(float(row["epoch"]), float(row["astop_mse"])) for row in curves
              if float(row["epoch"]) <= cap and row.get("astop_mse") is not None]
    if (not values or values[-1][0] != float(cap) or len({epoch for epoch, _ in values}) != len(values)
            or any(not math.isfinite(value) for _, value in values)):
        raise ValueError("Completed finite A_stop curve through cap required")
    return len(values) > 1 and values[-1][1] < min(value for _, value in values[:-1])


def choose_transfer(searches):
    """Choose A/B using both registered Jena seeds; an exact mean tie selects A."""
    expected = {(mode, seed) for mode in ("A", "B") for seed in common.SEEDS}
    seen, groups = set(), {"A": [], "B": []}
    for result in searches:
        spec = result["spec"]
        key = (spec["scientific_identity"]["raw_context_mode"], int(spec["seed"]))
        if (result["status"] != "COMPLETE" or result.get("V_used") is not False
                or result.get("selection_split") != "A_stop" or spec["dataset"] != "jena"
                or key not in expected or key in seen):
            raise ValueError("Complete Jena A/B by two-seed A_stop-only group required")
        value = float(result["summary"]["selected_astop_mse"])
        if not math.isfinite(value):
            raise ValueError("Finite selected A_stop MSE required")
        seen.add(key)
        groups[key[0]].append(value)
    if seen != expected:
        raise ValueError("Both registered seeds of A and B are required")
    means = {mode: math.fsum(values) / 2 for mode, values in groups.items()}
    return dict(selected_mode="A" if means["A"] <= means["B"] else "B",
                mean_astop_mse=means, selection_split="A_stop", V_used=False,
                tie_rule="exact_FP64_mean_tie_selects_A")


def _parent_path(ctx, result, parent):
    if isinstance(parent, dict):
        path = Path(parent["path"])
        expected = parent["sha256"]
    else:
        path = Path(parent if parent is not None else result["checkpoint_path"])
        expected = result["checkpoint_binding"]["sha256"]
    if common.sha(path) != expected or expected != result["checkpoint_binding"]["sha256"]:
        raise ValueError("Parent checkpoint bytes changed")
    return path


def _validate_parent(ctx, seed, result, parent_path, model, norm):
    """Validate the old scientific seal and the full continuation state."""
    task, modules = ctx["task"], ctx["modules"]
    fit, training, data = (modules[key] for key in ("fit", "training", "data"))
    old_spec, old_contract = result["spec"], result["contract"]
    expected_epochs = list(common.INITIAL_EPOCHS if task["dataset"] == "jena" else common.EXTENDED_EPOCHS)
    required = dict(family="B2", latent_k=model.kmax, seed=int(seed), dataset=task["dataset"],
                    device=ctx.get("device", "cuda"), revin=False, init_kind="PCA", core_dtype="float32", fold="primary")
    if (result["status"] != "COMPLETE" or result.get("V_used") is not False
            or result.get("selection_split") != "A_stop" or result["cap"] != int(expected_epochs[-1])
            or old_spec.get("epochs") != expected_epochs
            or any(old_spec.get(key) != value for key, value in required.items())
            or any(old_spec.get(key) != value for key, value in common.RECIPE.items())):
        raise ValueError("Parent search scope or recipe is not the registered S baseline")
    files = result["source_files"]
    for name, expected in files.items():
        path = Path(ctx["repo"]) / name if "/" in name else Path(fit.__file__).parent / name
        if common.sha(path) != expected:
            raise ValueError("Parent scientific source bytes changed: " + name)
    if training.fingerprint(files) != old_contract["source_sha"]:
        raise ValueError("Parent source seal mismatch")
    if training.fingerprint(dict(spec=old_spec, axes=_axes(task))) != old_contract["config_sha"]:
        raise ValueError("Parent configuration or axes changed")
    if old_contract["data_sha"] != task["data_sha256"] or old_contract["model_sha"] != fit._model_sha(ctx["foundation"]):
        raise ValueError("Parent data or frozen foundation changed")
    checkpoint = _load_checkpoint(parent_path)
    for key in ("optimizer", "sampler", "torch_rng", "cuda_rng", "updates", "extra", "model"):
        if key not in checkpoint:
            raise ValueError("Parent checkpoint lacks complete " + key)
    if checkpoint["contract"] != result["summary"]["contract"]:
        raise ValueError("Parent checkpoint contract differs from saved search summary")
    if any(checkpoint["contract"][key] != value for key, value in old_contract.items()):
        raise ValueError("Parent checkpoint scientific contract changed")
    active_old = _stage_contract(ctx, old_contract, norm, int(task["split_ranges"]["A_fit"][1]))
    if checkpoint["contract"] != active_old:
        raise ValueError("Parent PCA or scaler initialization scope changed")
    if _state_fingerprint(model.state_dict()) != result["initial_base_state_sha256"]:
        raise ValueError("Parent initial P/E/D state differs from fresh seeded initialization")
    origins = data.origins((0, task["split_ranges"]["A_fit"][1]), task["L"], task["H"], 1)
    sampler = training.PermutationStream(origins, seed)
    exposures = int(math.ceil(expected_epochs[-1] * len(origins)))
    sampler.draw(exposures)
    points = [0] + [math.ceil(epoch * len(origins)) for epoch in expected_epochs]
    updates = sum(math.ceil((right - left) / common.RECIPE["effective_batch"])
                  for left, right in zip(points, points[1:]))
    extra = checkpoint["extra"]
    if (checkpoint["sampler"] != sampler.state_dict() or checkpoint["updates"] != updates
            or result["summary"]["updates"] != updates or result["summary"]["exposures"] != exposures
            or extra.get("completed_epochs") != [0.] + expected_epochs
            or extra.get("curves") != result["curves"]
            or extra.get("best_epoch") != result["selected_epoch"]
            or extra.get("best_mse") != result["summary"]["selected_astop_mse"]):
        raise ValueError("Parent sampler, updates, exposures or checkpoint selection changed")
    optimizer = checkpoint["optimizer"]
    groups = optimizer.get("param_groups", [])
    parameters = list(model.parameters())
    if (len(groups) != 1 or groups[0].get("lr") != .001 or groups[0].get("weight_decay") != 0
            or tuple(groups[0].get("betas", ())) != (.9, .999) or groups[0].get("eps") != 1e-8
            or len(groups[0]["params"]) != len(parameters)):
        raise ValueError("Parent AdamW optimizer recipe changed")
    for index, parameter in zip(groups[0]["params"], parameters):
        state = optimizer["state"].get(index, {})
        if (set(state) != {"step", "exp_avg", "exp_avg_sq"} or int(state["step"].item()) != updates
                or state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape):
            raise ValueError("Parent optimizer state is incomplete")
    if (not isinstance(checkpoint["torch_rng"], torch.Tensor) or checkpoint["torch_rng"].dtype != torch.uint8
            or not isinstance(checkpoint["cuda_rng"], list)):
        raise ValueError("Parent RNG state is incomplete")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.load_state_dict(extra["best_state"], strict=True)
    return checkpoint


def reuse_S_search(ctx, seed, old_SEARCH_RESULT, parent_checkpoint=None):
    """Read and prove an existing S search without fitting or evaluating V."""
    old_path = Path(old_SEARCH_RESULT)
    old = common.read(old_path)
    path = _parent_path(ctx, old, parent_checkpoint)
    spec = _spec(ctx, "S", seed)
    with models.model_factory(_factory_context(ctx)) as create:
        base, source_files = _contract(ctx, spec)
        fit_end = int(ctx["task"]["split_ranges"]["A_fit"][1])
        model, norm = create(spec, ctx["task"], fit_end, ctx["foundation"])
        _validate_parent(ctx, seed, old, path, model, norm)
    result = dict(status="COMPLETE", spec=spec, contract=base, source_files=source_files,
        summary=old["summary"], curves=old["curves"], selected_epoch=old["selected_epoch"], cap=old["cap"],
        initial_base_state_sha256=old["initial_base_state_sha256"], last_checkpoint=str(path),
        checkpoint_path=str(path), checkpoint_binding=common.binding(path, ctx),
        parent_search_path=str(old_path), parent_search_binding=common.binding(old_path, ctx),
        training_segments=[], training_segments_unavailable_reason="Historical producer did not retain TRAIN segment losses",
        reused_parent=True, baseline_work="REUSED_VERIFIED_S", selection_split="A_stop", V_used=False)
    return _write_result(ctx, result)


def import_S_parent(ctx, reused_result):
    """Copy a fully proved S checkpoint, changing only the scientific contract."""
    if not reused_result.get("reused_parent"):
        raise ValueError("A verified reused S result is required")
    spec = reused_result["spec"]
    parent_path = common.resolve_binding(reused_result["checkpoint_binding"], ctx)
    old_path = common.resolve_binding(reused_result["parent_search_binding"], ctx)
    old = common.read(old_path)
    private, _, _, _ = _locations(ctx, spec, reused_result["cap"])
    with models.model_factory(_factory_context(ctx)) as create:
        base, _ = _contract(ctx, spec)
        if base != reused_result["contract"]:
            raise ValueError("New S scientific contract changed since reuse validation")
        fit_end = int(ctx["task"]["split_ranges"]["A_fit"][1])
        model, norm = create(spec, ctx["task"], fit_end, ctx["foundation"])
        checkpoint = _validate_parent(ctx, spec["seed"], old, parent_path, model, norm)
        active = _stage_contract(ctx, base, norm, fit_end)
    proof_path = private / "MIGRATION_PROOF.json"
    if proof_path.exists():
        proof = common.read(proof_path)
        if proof["parent_checkpoint"] != reused_result["checkpoint_binding"] or proof["new_contract"] != active:
            raise ValueError("Existing S migration proof changed")
        common.resolve_binding(proof["original_copy"], ctx)
        common.resolve_binding(proof["imported_checkpoint"], ctx)
        return proof
    stage = private / "search"
    if list(stage.glob("checkpoint_*.pt")):
        raise ValueError("Cannot import a parent into an existing search stage")
    stage.mkdir(parents=True, exist_ok=True)
    original = private / "parent_original.pt"
    if original.exists():
        if common.sha(original) != common.sha(parent_path):
            raise ValueError("Immutable original parent bytes changed")
    else:
        with original.open("xb") as stream:
            stream.write(parent_path.read_bytes())
    migrated = copy.deepcopy(checkpoint)
    migrated["contract"] = active
    preserved = _state_fingerprint({key: value for key, value in checkpoint.items() if key != "contract"})
    if _state_fingerprint({key: value for key, value in migrated.items() if key != "contract"}) != preserved:
        raise ValueError("S migration changed training state")
    destination = stage / "checkpoint_000000.pt"
    ctx["modules"]["fit"]._torch_new(destination, migrated)
    saved = _load_checkpoint(destination)
    if _state_fingerprint({key: value for key, value in saved.items() if key != "contract"}) != preserved:
        raise ValueError("Serialized S migration changed training state")
    proof = dict(status="COMPLETE", parent_checkpoint=reused_result["checkpoint_binding"],
        parent_search=reused_result["parent_search_binding"], original_copy=common.binding(original, ctx),
        imported_checkpoint=common.binding(destination, ctx), old_contract=checkpoint["contract"], new_contract=active,
        changed_fields=["contract"], preserved_training_state_sha256=preserved,
        optimizer_sampler_rng_model_updates_preserved=True, exact_resume=True)
    common.sealed_json(proof_path, proof)
    common.sealed_json(Path(ctx["report"]) / "runs" / spec["run_id"] / "MIGRATION_PROOF.json", proof)
    return proof


def refit(ctx, search_result, selection_binding):
    """Fresh A60 scaler/PCA/P/E/D/optimizer fit at a sealed A_stop epoch."""
    decision = common.read(common.resolve_binding(selection_binding, ctx))
    if (decision.get("selection_split") != "A_stop" or decision.get("V_used") is not False
            or decision.get("status") not in ("SEALED", "COMPLETE")):
        raise ValueError("A sealed A_stop-only group decision is required")
    if "searches" in decision and search_result["checkpoint_binding"] not in decision["searches"]:
        raise ValueError("Search checkpoint is absent from the sealed selection decision")
    if search_result["status"] != "COMPLETE" or search_result.get("V_used") is not False:
        raise ValueError("Complete A_stop-only search required")
    spec, cap = search_result["spec"], search_result["cap"]
    private, public, cap_private, cap_public = _locations(ctx, spec, cap)
    stored = common.read(cap_private / "SEARCH_RESULT.json")
    if stored != search_result:
        raise ValueError("Sealed search result changed before fresh refit")
    common.resolve_binding(search_result["checkpoint_binding"], ctx)
    output = cap_private / "REFIT_RESULT.json"
    with models.model_factory(_factory_context(ctx)):
        base, sources = _contract(ctx, spec)
        if base != search_result["contract"]:
            raise ValueError("Search/refit scientific contract changed")
        if output.exists():
            result = common.read(output)
            if result["selection_binding"] != selection_binding:
                raise ValueError("Fresh refit selection decision changed")
            common.resolve_binding(result["bundle_binding"], ctx)
        selected = float(search_result["selected_epoch"])
        fit_end = int(ctx["task"]["split_ranges"]["A_stop"][1])
        with _observe_training_segments(ctx, spec, f"fresh_A60_refit_cap{cap}",
                private / f"refit_cap{cap}", public / f"refit_cap{cap}", [selected]) as segments:
            stage = ctx["modules"]["fit"]._stage(spec, ctx["task"], fit_end, None,
                [selected] if selected else [], private / f"refit_cap{cap}", public / f"refit_cap{cap}",
                ctx["foundation"], _deadline(ctx), base)
        if stage["summary"]["status"] != "SEGMENT_COMPLETE":
            return dict(status="PARTIAL_CHECKPOINTED", phase="fresh_A60_refit", refit_summary=stage["summary"],
                        training_segments=segments)
        model_sha = ctx["modules"]["training"].fingerprint({key: ctx["modules"]["data"].array_sha256(value.numpy())
                                                           for key, value in stage["model"].items()})
        bundle = dict(model=stage["model"], normalization=stage["normalization"], spec=spec,
            contract=stage["summary"]["contract"], selected_search_epoch=selected,
            search_summary=search_result["summary"], refit_summary=stage["summary"], model_state_sha=model_sha)
        artifact = cap_private / "selected_bundle.pt"
        if artifact.exists():
            saved = _load_checkpoint(artifact)
            deterministic = set(bundle) - {"refit_summary"}
            if (set(saved) != set(bundle) or _state_fingerprint({key: saved[key] for key in deterministic})
                    != _state_fingerprint({key: bundle[key] for key in deterministic})):
                raise ValueError("Existing fresh-A60 bundle differs from the completed deterministic stage")
            session_fields = {"seconds", "F_session_logical_streams", "F_session_physical_streams",
                              "peak_allocated_bytes", "peak_reserved_bytes"}
            old_summary, new_summary = saved["refit_summary"], stage["summary"]
            if (set(old_summary) != set(new_summary)
                    or _state_fingerprint({key: value for key, value in old_summary.items() if key not in session_fields})
                    != _state_fingerprint({key: value for key, value in new_summary.items() if key not in session_fields})
                    or any(not isinstance(old_summary[key], (int, float)) or not math.isfinite(old_summary[key])
                           or old_summary[key] < 0 for key in session_fields)):
                raise ValueError("Existing fresh-A60 bundle summary differs from the completed stage")
            bundle = saved
            stage["summary"] = old_summary
        else:
            ctx["modules"]["fit"]._torch_new(artifact, bundle)
        receipt = dict(status="COMPLETE_VALID", run_id=spec["run_id"], dataset=spec["dataset"],
            mode=spec["scientific_identity"]["raw_context_mode"], seed=spec["seed"], cap=cap,
            selected_epoch=selected, selected_astop_mse=search_result["summary"]["selected_astop_mse"],
            search_fit_end=int(ctx["task"]["split_ranges"]["A_fit"][1]), refit_fit_end=fit_end,
            normalization_fit_end=fit_end, fresh_A60=True, V_used=False,
            selection_split="A_stop", selection_binding=selection_binding,
            refit_updates=stage["summary"]["updates"], refit_exposures=stage["summary"]["exposures"],
            artifact_cache_relative_path=artifact.relative_to(Path(ctx["cache"])).as_posix(),
            artifact_sha256=common.sha(artifact), model_state_sha=model_sha,
            source_sha=base["source_sha"], config_sha=base["config_sha"], data_sha=base["data_sha"],
            refit_scaler_sha=stage["summary"]["contract"]["scaler_sha"])
        result = dict(status="COMPLETE", spec=spec, cap=cap, bundle_binding=common.binding(artifact, ctx),
            fit_receipt=receipt, source_files=sources, selection_binding=selection_binding,
            search_summary=search_result["summary"], refit_summary=stage["summary"], V_used=False,
            training_segments=segments)
        common.sealed_json(output, result)
        common.sealed_json(cap_public / "REFIT_RESULT.json", result)
        common.sealed_json(cap_public / "BASE_FIT_RECEIPT.json", receipt)
        return result
