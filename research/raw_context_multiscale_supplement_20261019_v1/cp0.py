"""Fixed serialized CP0 A_stop diagnostics; no training or checkpoint selection."""
import argparse
from contextlib import nullcontext
import copy
import csv
import io
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from research.b2_revin_20261018_v1.training_jobs import _state_fingerprint
from research.raw_context_multiscale_20261018_v1 import common, data, models, training


NAME = "raw_context_multiscale_supplement_20261019_v1"
F_SHA = "c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"
PLAN_NAME = "SUPPLEMENT_PLAN_001.json"
TERMINAL_NAME = "TERMINAL_CORE_002.json"
CP0_RULE = "Serialized zero-update zero-exposure CP0; follow explicit S migration parents, never infer epoch from filename"
FIELDS = ("dataset", "mode", "seed", "run_id", "epoch", "K_total", "K_raw", "K_ctx",
          "updates", "exposures", "raw_scale", "ctx_scale", "MSE", "MAE", "A_stop_mse", "A_stop_mae",
          "n_origins", "n_observed", "origins_sha256", "fit_end",
          "checkpoint_reference", "checkpoint_sha256", "original_CP0_MSE", "MSE_parity",
          "prediction_reference", "prediction_sha256", "device", "split")


def _deadline(ctx):
    if time.time() >= ctx["deadline"]:
        raise TimeoutError("Original core wall/GPU budget exhausted during CP0 diagnostic")


def expected_keys(selected_mode):
    if selected_mode not in ("A", "B"):
        raise ValueError("Sealed PeMS transfer family A or B required")
    return {(dataset, mode, seed) for dataset, modes in
            (("jena", ("S", "A", "B")), ("pems32", ("S", selected_mode)))
            for mode in modes for seed in common.SEEDS}


def body_finals(ctx):
    transfer = common.read(ctx["report"] / "TRANSFER_SELECTION.json")
    if (transfer.get("status") != "SEALED" or transfer.get("selection_split") != "A_stop"
            or transfer.get("V_used") is not False or transfer.get("Jena_V_opened") is not False):
        raise ValueError("A_stop-only sealed transfer selection required")
    finals = common.read(ctx["cache"] / "FINAL_BODY_MODELS.json")
    keys = [(row["dataset"], row["mode"], int(row["seed"])) for row in finals]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys(transfer["selected_mode"]):
        raise ValueError("Exact ten final body paths required, without a duplicate B(s1) family")
    seal = common.read(ctx["report"] / "MODELS_SEALED_BEFORE_V.json")
    sealed = {row["run_id"]: row for row in seal["finals"]}
    if len(sealed) != 10:
        raise ValueError("All ten body models must appear in the model-before-V seal")
    for row in finals:
        if (row["run_id"] != f'{row["mode"]}_{row["dataset"]}_s{row["seed"]}'
                or row["run_id"] not in sealed
                or sealed[row["run_id"]]["route_spec"] != row["route_spec"]
                or sealed[row["run_id"]]["reused"] != row["reused"]):
            raise ValueError("Final body path differs from the model-before-V seal")
        search = row["search"]
        spec = search.get("spec", {})
        if (search.get("status") != "COMPLETE" or search.get("V_used") is not False
                or search.get("selection_split") != "A_stop"
                or spec.get("dataset") != row["dataset"] or spec.get("seed") != row["seed"]
                or spec.get("run_id") != row["run_id"] or spec.get("family") != "B2"
                or spec.get("latent_k") != common.EXPECTED[row["dataset"]]["K"]
                or spec.get("scientific_identity", {}).get("raw_context_mode") != row["mode"]):
            raise ValueError("Completed A_stop-only body searches required")
    return sorted(finals, key=lambda r: (r["dataset"], r["mode"], r["seed"]))


def validate_plan(ctx, plan_path, device):
    """Check immutable registrations and terminal metadata before numerical reads."""
    from research.raw_context_multiscale_20261018_v1 import run
    plan_path = Path(plan_path).resolve()
    if plan_path != (ctx["report"] / PLAN_NAME).resolve():
        raise ValueError("Use the registered supplement plan in the original report directory")
    plan = common.read(plan_path)
    active = run.registered_plan_path(ctx)
    scope = plan.get("cp0_scope", {})
    if (plan.get("status") != "REGISTERED_BEFORE_NEW_V"
            or plan.get("primary_plan_binding") != common.binding(active, ctx)
            or plan.get("new_V_scores_at_registration") != 0
            or plan.get("scientific_contract_changed") is not False
            or plan.get("selection_changes") is not False or plan.get("new_body_training") is not False
            or scope.get("CP0_rule") != CP0_RULE or scope.get("device") != device
            or scope.get("split") != "A_stop" or scope.get("epoch") != 0
            or scope.get("final_body_paths") != 10 or scope.get("B_s1_alias") != "A"
            or scope.get("atol") != common.ATOL or scope.get("rtol") != common.RTOL
            or scope.get("training_updates") != 0 or scope.get("V_access") is not False
            or scope.get("T_access") is not False):
        raise ValueError("Fixed A_stop-only CP0 supplement registration changed")
    primary = run.check_plan(ctx, active)
    if plan.get("primary_source_files") != primary["source_files"] or len(primary["source_files"]) != 13:
        raise ValueError("The frozen thirteen primary source bindings changed")
    own = "research/" + NAME + "/cp0.py"
    init = "research/" + NAME + "/__init__.py"
    files = plan.get("source_files", {})
    if own not in files or init not in files:
        raise ValueError("Both supplement evaluator source bindings are required")
    for name, digest in files.items():
        path = (ctx["repo"] / name).resolve()
        if not path.is_relative_to((ctx["repo"] / "research" / NAME).resolve()) or common.sha(path) != digest:
            raise ValueError("Supplement source binding changed: " + name)
    terminal = common.read(ctx["cache"] / TERMINAL_NAME)
    if terminal.get("status") != "COMPLETE" or terminal.get("exit_code") != 0:
        raise ValueError("Actual main attempt two must complete before CP0 numerical access")
    for name, status in (("MODELS_SEALED_BEFORE_V.json", "SEALED"), ("SCORES.json", "COMPLETE"),
                         ("COST_INDEX.json", "COMPLETE")):
        path = ctx["report"] / name
        if not path.is_file() or common.read(path).get("status") != status:
            raise ValueError("Complete models, scores and cost index required before CP0 evaluation")
    seal_path = ctx["report"] / "MODELS_SEALED_BEFORE_V.json"
    if (common.read(seal_path).get("V_used") is not False
            or float(plan["registered_unix"]) > seal_path.stat().st_mtime
            or plan_path.stat().st_mtime > seal_path.stat().st_mtime):
        raise ValueError("Supplement must be registered before the model-before-V seal")
    scores = common.read(ctx["report"] / "SCORES.json")
    if scores.get("V_independent_test") is not False or len(scores.get("arms", [])) != 56:
        raise ValueError("All 56 original main paths must complete first")
    facets = common.read(ctx["report"] / "COST_INDEX.json")["facets"]
    expected = {(a["run_id"], schedule, batch) for a in scores["arms"]
                for schedule in (("native", "two_call") if a["mode"] == "A" else ("native",))
                for batch in (1, 8)}
    actual = [(f["run_id"], f["schedule"], int(f["batch"])) for f in facets]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Complete exact whole-path cost coverage required")
    for facet in facets:
        receipt = common.read(common.resolve_binding(facet["output"], ctx))
        if receipt.get("status") != "COMPLETE":
            raise ValueError("Every original cost facet must have a completed bound receipt")
    original = ctx["cache"] / "CORE_EXECUTION_START.json"
    if plan.get("original_execution_start_binding") != common.binding(original, ctx):
        raise ValueError("Original execution start binding changed")
    proof = run.require_preceding_publication(ctx)
    run.start_execution(ctx, proof)
    if plan.get("original_deadline_unix") != ctx["deadline"]:
        raise ValueError("Original wall deadline may not be reset for CP0")
    _deadline(ctx)
    return plan, body_finals(ctx)


def a_stop_task(ctx, dataset):
    """Decode numeric source rows only through A_stop, never through M_cal or V."""
    _deadline(ctx)
    campaign = ctx["modules"]["campaign"]
    primitive = data._module(campaign / data.PRIMITIVE, "cp0_primitive")
    expected = common.read(ctx["repo"] / "reports/b2_revin_20261018_v1/DATA_CONTRACT.json")["tasks"][dataset]
    end = int(expected["split_ranges"]["A_stop"][1])
    if dataset == "jena":
        old = common.read(campaign / "DATA_CONTRACT.json")["assets"]["jena"]
        source = ctx["asset"] / primitive.SOURCE_REL
        if source.stat().st_size != primitive.SOURCE_BYTES or common.sha(ctx["asset"] / primitive.CONTRACT_REL) != primitive.CONTRACT_SHA:
            raise ValueError("Jena ancestry contract or source size changed")
        before = source.stat()
        values, members = {}, []
        with zipfile.ZipFile(source) as archive:
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError("Duplicate Jena source members")
            for key in ("x", "finite", "times", "mean", "std", "columns"):
                _deadline(ctx)
                values[key], evidence = primitive.read_member(
                    archive, key + ".npy", end if key in ("x", "finite", "times") else None)
                members.append(evidence)
        physical = (values["x"] * values["std"] + values["mean"]).astype(np.float32)
        observed = values["finite"].copy()
        physical[~observed] = np.nan
        task = dict(physical=physical, observed=observed, times=values["times"],
                    columns=values["columns"].tolist(), interval_seconds=600, data_sha256=old["data_sha256"])
    elif dataset == "pems32":
        subset = data._module(campaign / data.SUBSET, "cp0_subset")
        pems = data._module(campaign / data.PEMS, "cp0_pems")
        plan = common.read(campaign / "priority_pems_subset_20261016_v1/PLAN.json")
        subset.require_sealed(plan)
        for name, digest in plan["dependencies"].items():
            if common.sha(campaign / name) != digest:
                raise ValueError("PeMS sealed subset dependency changed")
        old = plan["parent_task_contract"]
        meta = old["metadata"]
        source = ctx["modules"]["old_cache"] / meta["source_reference"]
        if source.stat().st_size != meta["source_bytes"]:
            raise ValueError("PeMS source size changed")
        before = source.stat()
        with zipfile.ZipFile(source) as archive:
            if archive.namelist() != ["data.npy"]:
                raise ValueError("Original single PeMS source member required")
            info = archive.getinfo("data.npy")
            with archive.open(info) as stream:
                raw, member = pems.prefix_payload(stream, info.file_size, primitive, rows=end)
        columns = list(old["columns"])
        indices = [columns.index(name) for name in expected["columns"]]
        if (len(indices) != 32 or len(set(indices)) != 32
                or indices != plan["column_selection"]["selected_indices"]
                or expected["columns"] != plan["column_selection"]["selected_ids"]):
            raise ValueError("Exact ordered PeMS32 channel subset required")
        physical = raw[:, indices, 0].astype(np.float32)
        observed = np.isfinite(physical)
        physical[~observed] = np.nan
        task = dict(physical=physical, observed=observed, times=np.arange(end, dtype=np.int64) * 300 * 1000000000,
                    columns=list(expected["columns"]), interval_seconds=300, data_sha256=expected["data_sha256"])
        members = [member]
    else:
        raise ValueError("Only the registered Jena and PeMS32 cohorts are allowed")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while reading A_stop prefix")
    if (task["physical"].shape != (end, expected["C"]) or task["observed"].dtype != np.bool_
            or task["columns"] != expected["columns"] or task["data_sha256"] != expected["data_sha256"]
            or not np.isfinite(task["physical"][task["observed"]]).all()
            or not np.all(np.diff(task["times"]) == task["interval_seconds"] * 1000000000)):
        raise ValueError("Original A_stop prefix, bool mask, ordered channels or time grid changed")
    task.update(dataset=dataset, name=dataset, L=expected["L"], H=expected["H"],
                K=common.EXPECTED[dataset]["K"], split_ranges=copy.deepcopy(expected["split_ranges"]),
                array_sha256=copy.deepcopy(expected["array_sha256"]))
    receipt = dict(dataset=dataset, numeric_prefix_rows=end, M_cal_V_T_numeric_values_parsed=0,
                   full_source_hash_recomputed=False, source_bytes=before.st_size, members=members,
                   prefix_arrays={name: common.array_identity(task[name]) for name in ("physical", "observed", "times")},
                   channel_ids=common.array_identity(np.asarray(task["columns"])),
                   inherited_data_sha256=task["data_sha256"],
                   IO_scope="A_stop numeric prefix only; ZIP may buffer opaque bytes after that prefix")
    _deadline(ctx)
    return task, receipt


def _scoped_ctx(ctx, path):
    path = Path(path).resolve()
    for kind, root_name in (("cache", ".cache"), ("report", "reports")):
        root = (ctx["repo"] / root_name).resolve()
        if path.is_relative_to(root):
            campaign = path.relative_to(root).parts[0]
            return dict(ctx, cache=ctx["repo"] / ".cache" / campaign,
                        report=ctx["repo"] / "reports" / campaign)
    raise ValueError("Historical checkpoint must lie in this repository's campaign roots")


def _cp0_state(checkpoint):
    return (checkpoint.get("updates") == 0 and checkpoint.get("sampler", {}).get("exposures") == 0
            and checkpoint.get("extra", {}).get("completed_epochs") == [0.]
            and not checkpoint.get("optimizer", {}).get("state"))


def resolve_cp0(ctx, final):
    """Follow only explicit S ancestry; checkpoint_000000 may contain trained state."""
    _deadline(ctx)
    search = final["search"]
    chain = []
    if final["mode"] == "S":
        from research.raw_context_multiscale_20261018_v1.audit import old_s_root
        source_result = old_s_root(ctx, final["dataset"], final["seed"]) / "SEARCH_RESULT.json"
        old = common.read(source_result)
        scoped = _scoped_ctx(ctx, source_result)
        terminal = common.resolve_binding(old["checkpoint_binding"], scoped)
        chain.append(dict(role="ORIGINAL_COMPLETED_S_SEARCH", artifact=common.binding(source_result, ctx)))
        candidate = terminal.parent / "checkpoint_000000.pt"
    else:
        candidate = ctx["cache"] / "runs" / final["run_id"] / "search/checkpoint_000000.pt"
    seen = set()
    while True:
        _deadline(ctx)
        candidate = candidate.resolve()
        if candidate in seen:
            raise ValueError("Cyclic CP0 migration ancestry")
        seen.add(candidate)
        checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
        chain.append(dict(role="SERIALIZED_STAGE_CHECKPOINT_ZERO", artifact=common.binding(candidate, ctx),
                          updates=checkpoint.get("updates"), exposures=checkpoint.get("sampler", {}).get("exposures")))
        if _cp0_state(checkpoint):
            return candidate, checkpoint, chain
        if final["mode"] != "S":
            raise ValueError("Serialized new-family CP0 contains training updates or exposures")
        scoped = _scoped_ctx(ctx, candidate)
        relative = candidate.parent.relative_to(scoped["cache"])
        proofs = [scoped["report"] / relative / "MIGRATION_PROOF.json",
                  candidate.parent.parent / "MIGRATION_PROOF.json"]
        available = [p for p in proofs if p.exists()]
        if len(available) != 1:
            raise ValueError("Trained checkpoint zero requires one explicit S migration proof")
        proof_path = available[0]
        proof = common.read(proof_path)
        if (common.resolve_binding(proof["imported_checkpoint"], scoped) != candidate
                or proof.get("changed_fields") != ["contract"]):
            raise ValueError("S migration proof does not bind this trained checkpoint zero")
        parent = common.resolve_binding(proof["parent_checkpoint"], scoped)
        chain.append(dict(role="EXPLICIT_S_MIGRATION", artifact=common.binding(proof_path, ctx),
                          parent_checkpoint=common.binding(parent, ctx)))
        candidate = parent.parent / "checkpoint_000000.pt"


def independent_statistics(prediction, target, observed):
    p, y, mask = np.asarray(prediction), np.asarray(target), np.asarray(observed)
    if p.shape != y.shape or p.ndim != 3 or mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError("Aligned original-channel arrays and bool masks required")
    if not np.isfinite(p).all() or not np.isfinite(y[mask]).all():
        raise ValueError("Finite normalized predictions and observed targets required")
    mse, mae, counts, sse, sae = [], [], [], [], []
    for channel in range(p.shape[2]):
        selected = mask[:, :, channel]
        errors = p[:, :, channel][selected].astype(np.float64) - y[:, :, channel][selected].astype(np.float64)
        count = len(errors)
        square = math.fsum(float(value) * float(value) for value in errors)
        absolute = math.fsum(abs(float(value)) for value in errors)
        counts.append(count)
        sse.append(square)
        sae.append(absolute)
        if count:
            mse.append(square / count)
            mae.append(absolute / count)
    if not mse:
        raise ValueError("No observed A_stop targets")
    return dict(mse=math.fsum(mse) / len(mse), mae=math.fsum(mae) / len(mae),
                n_origins=len(p), n_observed=sum(counts), channel_counts=counts,
                channel_sse=sse, channel_sae=sae, accumulation="FP64 math.fsum",
                aggregation="equal-channel observed macro")


def evaluate_one(ctx, task, final, foundation, device, plan_binding, access):
    _deadline(ctx)
    cp_path, checkpoint, chain = resolve_cp0(ctx, final)
    search = final["search"]
    original = [row for row in search["curves"] if float(row["epoch"]) == 0.]
    if len(original) != 1 or not math.isfinite(float(original[0]["astop_mse"])):
        raise ValueError("Exactly one original finite CP0 A_stop MSE required")
    spec = copy.deepcopy(search["spec"])
    spec["device"] = device
    fit_end = int(task["split_ranges"]["A_fit"][1])
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices), models.model_factory(training._factory_context(ctx)) as create:
        model, norm = create(spec, task, fit_end, foundation)
        _deadline(ctx)
        try:
            initial = _state_fingerprint(model.state_dict())
            if (initial != search["initial_base_state_sha256"]
                    or _state_fingerprint(checkpoint["model"]) != initial or not _cp0_state(checkpoint)):
                raise ValueError("Serialized CP0 weights differ from the original seeded untrained model")
            if foundation.initial_hash != F_SHA:
                raise ValueError("The fixed foundation tensor identity changed")
            active = training._stage_contract(ctx, search["contract"], norm, fit_end)
            for key in ("data_sha", "model_sha", "scaler_sha", "initialization_sha"):
                if checkpoint["contract"].get(key) != active[key]:
                    raise ValueError("CP0 data/F/scaler/PCA initialization contract changed: " + key)
            if norm["fit_end"] != fit_end or norm.get("train_end", fit_end) != fit_end:
                raise ValueError("CP0 normalization must be fitted on A_fit alone")
            model.load_state_dict(checkpoint["model"], strict=True)
            torch.set_rng_state(checkpoint["torch_rng"])
            if device == "cuda" and checkpoint.get("cuda_rng"):
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            model.eval()
            origins = ctx["modules"]["data"].origins(task["split_ranges"]["A_stop"], task["L"], task["H"], task["H"])
            windows = ctx["modules"]["training"].FastWindows(task, norm["mean"], norm["std"], device)
            predictions, targets, masks = [], [], []
            with torch.no_grad():
                for start in range(0, len(origins), int(search["spec"]["microbatch"])):
                    _deadline(ctx)
                    x, target, mask = windows.get(origins[start:start + int(search["spec"]["microbatch"])])
                    prediction = model(x, foundation)
                    predictions.append(prediction.detach().cpu().numpy().astype(np.float32))
                    targets.append(target.detach().cpu().numpy().astype(np.float32))
                    masks.append(mask.detach().cpu().numpy())
            _deadline(ctx)
            prediction, target, mask = map(np.concatenate, (predictions, targets, masks))
            stats = independent_statistics(prediction, target, mask)
            old_mse = float(original[0]["astop_mse"])
            if not math.isclose(stats["mse"], old_mse, abs_tol=common.ATOL, rel_tol=common.RTOL):
                raise ValueError("Serialized CP0 A_stop MSE fails original fixed tolerance parity")
            foundation.assert_frozen_unchanged()
            if _state_fingerprint(model.state_dict()) != initial:
                raise ValueError("CP0 diagnostic changed model state")
            metadata = dict(run_id=final["run_id"], dataset=task["dataset"], mode=final["mode"],
                            seed=final["seed"], split="A_stop", epoch=0, training_updates=0,
                            device=device, foundation_sha256=F_SHA, scaler_sha256=active["scaler_sha"],
                            initialization_sha256=active["initialization_sha"], initial_state_sha256=initial,
                            checkpoint_binding=common.binding(cp_path, ctx), ancestry=chain,
                            supplement_plan_binding=plan_binding, source_files=ctx["plan"]["source_files"],
                            checkpoint_contract=checkpoint["contract"],
                            normalization_arrays={name: common.array_identity(norm[name])
                                                  for name in ("mean", "std", "basis", "pca_center", "observed_counts")},
                            A_fit_end=fit_end, access=access, V_access=False, T_access=False,
                            RNG="serialized CP0 CPU RNG restored; CUDA RNG restored only on CUDA backend")
            path = ctx["cache"] / "supplement/cp0_predictions" / (final["run_id"] + ".npz")
            evidence = common.save_prediction(path, prediction=prediction, target=target, observed=mask,
                                              origins=origins, task=task, metadata=metadata)
            for name in ("mse", "mae"):
                if not math.isclose(stats[name], evidence["score"][name], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Independent FP64 CP0 statistics disagree with saved prediction receipt")
            k = int(spec["latent_k"])
            mode = final["mode"]
            row = dict(dataset=task["dataset"], mode=mode, seed=int(final["seed"]), run_id=final["run_id"],
                       epoch=0, updates=0, exposures=0, K_total=k, K_raw=0 if mode == "S" else k // 2,
                       K_ctx=k if mode == "S" else k // 2, raw_scale=1, ctx_scale=2 if mode == "B" else 1,
                       MSE=stats["mse"], MAE=stats["mae"], A_stop_mse=stats["mse"], A_stop_mae=stats["mae"],
                       n_origins=stats["n_origins"], origins_sha256=common.array_identity(origins)["sha256"],
                       n_observed=stats["n_observed"], fit_end=fit_end, checkpoint_reference=common.binding(cp_path, ctx)["reference"],
                       checkpoint_sha256=common.sha(cp_path), original_CP0_MSE=old_mse, MSE_parity="PASS",
                       prediction_reference=common.binding(path, ctx)["reference"], prediction_sha256=common.sha(path),
                       device=device, split="A_stop")
            return row, dict(run_id=final["run_id"], statistics=stats, arrays=evidence["arrays"],
                             metadata=metadata, prediction_binding=common.binding(path, ctx))
        finally:
            del model


def _sealed_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Preserve existing CP0 CSV; changed diagnostic cannot overwrite it")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)


def evaluate(ctx, plan_path, *, device):
    """Invoke only after actual main completion; retain its wall and GPU budget."""
    from research.adapter_improvements_20261018_v1.audit import load_foundation
    from research.raw_context_multiscale_20261018_v1 import run
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit CPU or CUDA diagnostic backend required")
    started = time.time()
    plan, finals = validate_plan(ctx, plan_path, device)
    plan_binding = common.binding(plan_path, ctx)
    validation_path = ctx["report"] / "CP0_A_STOP_VALIDATION.json"
    if validation_path.exists():
        previous = common.read(validation_path)
        if (previous.get("status") != "PASS" or previous.get("coverage") != 10
                or previous.get("plan_binding") != plan_binding or previous.get("device") != device):
            raise ValueError("Existing CP0 validation identity changed")
        common.resolve_binding(previous["CSV_binding"], ctx)
        for record in previous["evidence"]:
            common.resolve_binding(record["prediction_binding"], ctx)
        return previous
    gpu_before = run.gpu_seconds(ctx)
    rows, evidence = [], []
    manager = run.gpu_session(ctx, "CP0_A_stop_supplement") if device == "cuda" else nullcontext()
    with manager:
        for dataset in ("jena", "pems32"):
            _deadline(ctx)
            task, access = a_stop_task(ctx, dataset)
            foundation = load_foundation(ctx, ctx["modules"], task["H"], device)
            try:
                _deadline(ctx)
                for final in (row for row in finals if row["dataset"] == dataset):
                    print("CP0_A_STOP " + final["run_id"], flush=True)
                    row, proof = evaluate_one(ctx, task, final, foundation, device, plan_binding, access)
                    rows.append(row)
                    evidence.append(proof)
                foundation.assert_frozen_unchanged()
            finally:
                del foundation
                if device == "cuda":
                    torch.cuda.empty_cache()
    _deadline(ctx)
    csv_path = ctx["report"] / "CP0_A_STOP.csv"
    _sealed_csv(csv_path, rows)
    gpu_after = run.gpu_seconds(ctx)
    validation = dict(status="PASS", diagnostic="CP0_A_STOP_MSE_MAE_ONLY", split="A_stop", epoch=0,
                      device=device, training_updates=0, V_access=False, T_access=False,
                      coverage=len(rows), expected_coverage=10, CP0_rule=CP0_RULE,
                      plan_binding=plan_binding, primary_plan_binding=plan["primary_plan_binding"],
                      terminal_binding=common.binding(ctx["cache"] / TERMINAL_NAME, ctx),
                      original_execution_start_binding=plan["original_execution_start_binding"],
                      original_deadline_unix=plan["original_deadline_unix"],
                      CSV_binding=common.binding(csv_path, ctx), evidence=evidence,
                      wall_seconds=time.time() - started,
                      started_unix=started, finished_unix=time.time(),
                      GPU_seconds_before=gpu_before, GPU_seconds_after=gpu_after,
                      GPU_seconds_added=gpu_after-gpu_before,
                      GPU_accounting="GPU_SESSION process wall upper bound" if device == "cuda" else "CPU diagnostic, GPU seconds zero")
    common.sealed_json(validation_path, validation)
    common.event(ctx, "CP0_A_stop_supplement", "COMPLETE", rows=len(rows), device=device,
                 wall_seconds=validation["wall_seconds"], GPU_seconds=0 if device == "cpu" else None)
    return validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    ctx = common.context(args.asset_root, device=args.device)
    evaluate(ctx, args.plan, device=args.device)


if __name__ == "__main__":
    main()
