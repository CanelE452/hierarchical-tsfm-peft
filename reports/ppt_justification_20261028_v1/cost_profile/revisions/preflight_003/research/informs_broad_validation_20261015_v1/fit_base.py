"""Chronological search/refit execution; no data or foundation model loading."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

import data_contract
import evaluation
import networks
import training


SOURCE_NAMES = ("fit_base.py", "training.py", "networks.py", "data_contract.py", "gpu_runtime.py", "evaluation.py")
DIRECT = {"DIRECT_G", "DIRECT_B2", "DIRECT_B4"}


def _source_contract():
    here = Path(__file__).resolve().parent
    files = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
    return training.fingerprint(files), files


def _json_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _torch_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(value, handle)


def _model_sha(foundation):
    if foundation is None:
        return "NO_FOUNDATION"
    if hasattr(foundation, "initial_hash"):
        return str(foundation.initial_hash)
    return networks.module_state_hash(foundation)


def scaler_fingerprint(mean, std):
    return training.fingerprint({"mean": evaluation.array_fingerprint(np.asarray(mean, dtype=np.float32)),
                                 "std": evaluation.array_fingerprint(np.asarray(std, dtype=np.float32))})


class _FixedPCA(nn.Module):
    def __init__(self, center, basis):
        super().__init__()
        self.register_buffer("center", torch.as_tensor(center).float())
        self.register_buffer("E", torch.as_tensor(basis).float())

    def forward(self, x, foundation, latent_k=None):
        k = self.E.shape[1] if latent_k is None else latent_k
        basis = self.E[:, :k]
        return foundation((x - self.center) @ basis) @ basis.T + self.center


def create_model(spec, task, fit_end, foundation):
    """Fresh seeded model/statistics/PCA, usable later for bundle deployment."""
    torch.manual_seed(int(spec.get("seed", 0)))
    device = spec.get("device", "cuda" if foundation is not None else "cpu")
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(spec.get("seed", 0)))
    family = str(spec["family"])
    channels = task["physical"].shape[1]
    length, horizon = int(task["L"]), int(task["H"])
    k = int(spec.get("latent_k", channels))
    grid = tuple(spec["k_grid"]) if spec.get("k_grid") is not None else None
    if grid and (max(grid) != k or min(grid) < 1):
        raise ValueError("Ordered model latent_k must equal maximum registered K")
    mean, std, counts = data_contract.fit_statistics(task, int(fit_end))
    normalization = dict(mean=mean, std=std, observed_counts=counts, fit_end=int(fit_end), train_end=int(fit_end),
                         basis_initialization="not_applicable", pca_center=None, basis=None)
    if family == "DIRECT_G":
        model = networks.DirectG(length, horizon)
    elif family in {"DIRECT_B2", "DIRECT_B4"}:
        model = networks.DirectMixer(family[-2:], length, horizon, channels)
    elif family == "LORA":
        if foundation is None or not hasattr(foundation, "model"):
            raise ValueError("LoRA requires caller-owned foundation.model")
        model = networks.LoRAQvAdapter(foundation.model, rank=int(spec.get("lora_rank", 8)), alpha=float(spec.get("lora_alpha", 16)))
    elif family == "ADAPTS":
        model = networks.AdaPTSLinearPort(length, horizon, channels, k)
    else:
        if family not in {"B0", "B2", "B4", "LAG_CAUSAL", "LAG_BIDIR", "PCA_FIXED", "PCA_FIXED_BOLT"}:
            raise ValueError("Unregistered base family: " + family)
        if not 1 <= k <= channels:
            raise ValueError("K outside original channel count")
        if k == channels and grid is None and family in {"B0", "B2", "B4", "LAG_CAUSAL", "LAG_BIDIR"}:
            basis = np.eye(channels, dtype=np.float32)
            center = np.zeros(channels, np.float32)
            normalization["basis_initialization"] = "identity_at_K_equals_C"
        else:
            pca = data_contract.initialize_pca(task, fit_end, k, mean, std)
            basis, center = pca["E"], pca["pca_center"]
            normalization["basis_initialization"] = "A_fitted_centered_PCA"
            normalization["explained_variance"] = pca["explained_variance"]
        normalization.update(basis=basis, pca_center=center)
        if family.startswith("PCA_FIXED"):
            model = _FixedPCA(center, basis)
        else:
            encoder = {"LAG_CAUSAL": "causal7", "LAG_BIDIR": "bidirectional7"}.get(family, "pointwise")
            front = family if family in {"B0", "B2", "B4"} else "B0"
            model = networks.ForecastAdapter(front, length, horizon, channels, k, center, basis, encoder, grid)
    if family not in DIRECT and foundation is None:
        raise ValueError("This base family requires an explicit foundation callable")
    return model.to(device), normalization


def predict_model(model, x, foundation, family, latent_k=None):
    if family in DIRECT:
        return model(x)
    if family == "LORA":
        return foundation(x)
    if family == "ADAPTS":
        return model(x, foundation)
    return model(x, foundation, latent_k)


def _evaluate(model, windows, origins, predict, grid, microbatch, deadline=None):
    model.eval()
    scores = []
    with torch.no_grad():
        for k in grid or (None,):
            sse = torch.zeros(windows.channels, dtype=torch.float64, device=windows.device)
            counts = torch.zeros(windows.channels, dtype=torch.int64, device=windows.device)
            for start in range(0, len(origins), microbatch):
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("Budget reached during A_stop evaluation")
                x, y, mask = windows.get(origins[start:start + microbatch])
                p = predict(x, k)
                if p.shape != y.shape or not torch.isfinite(p[mask]).all():
                    raise ValueError("Invalid A_stop forecast")
                errors = p.double() - y.double()
                sse += torch.where(mask, errors.square(), 0).sum((0, 1))
                counts += mask.sum((0, 1))
            active = counts > 0
            if not active.any():
                raise ValueError("A_stop contains no observed target")
            scores.append(float((sse[active] / counts[active]).mean().cpu()))
    return float(math.fsum(scores) / len(scores)), scores


def _exposure_summary(task, origins, exposures, seed):
    if exposures == 0:
        return dict(unique_origins=0, unique_target_times=0, observed_target_exposures=0)
    stream = training.PermutationStream(origins, seed)
    sampled = stream.draw(int(exposures))
    unique = np.unique(sampled)
    times = np.zeros(len(task["physical"]), bool)
    for origin in unique:
        times[origin:origin + task["H"]] = True
    observed = np.asarray(task["observed"], bool)
    cumulative = np.concatenate((np.zeros((1, observed.shape[1]), np.int64), observed.cumsum(0)), 0)
    count = (cumulative[sampled + task["H"]] - cumulative[sampled]).sum()
    return dict(unique_origins=len(unique), unique_target_times=int(times.sum()), observed_target_exposures=int(count))


def _f_counters(foundation):
    return {name: int(getattr(foundation, name, 0)) for name in ("streams", "row_calls", "physical_streams", "physical_calls")}


def _stage_body(spec, task, fit_end, stop_bounds, epochs, stage_dir, public_dir, foundation, deadline, base_contract, model, normalization):
    stage_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    family = spec["family"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(spec.get("lr", 1e-3)), weight_decay=0) if params else None
    contract = dict(base_contract, scaler_sha=scaler_fingerprint(normalization["mean"], normalization["std"]))
    contract["initialization_sha"] = training.fingerprint(dict(scaler_sha=contract["scaler_sha"], fit_end=fit_end,
        basis_sha=None if normalization["basis"] is None else data_contract.array_sha256(normalization["basis"]),
        center_sha=None if normalization["pca_center"] is None else data_contract.array_sha256(normalization["pca_center"])))
    legal = data_contract.origins((0, fit_end), task["L"], task["H"], 1)
    if not len(legal):
        raise ValueError("No legal training origins")
    stops = None if stop_bounds is None else data_contract.origins(stop_bounds, task["L"], task["H"], task["H"])
    if stops is not None and not len(stops):
        raise ValueError("No legal A_stop origins")
    # Only this stage's permitted prefix is materialized by FastWindows.
    prefix_end = fit_end if stop_bounds is None else stop_bounds[1]
    prefix = dict(task, physical=task["physical"][:prefix_end], observed=task["observed"][:prefix_end])
    windows = training.FastWindows(prefix, normalization["mean"], normalization["std"], spec.get("device", "cuda" if foundation is not None else "cpu"))
    sampler = training.PermutationStream(legal, int(spec.get("seed", 0)))
    predict = lambda x, k: predict_model(model, x, foundation, family, k)
    grid = spec.get("k_grid")
    curves = []
    completed_epochs = []
    updates = 0
    best_mse = math.inf
    best_epoch = 0.
    best_state = training.cpu_state(model)
    total_clipped = 0
    train_streams = 0
    previous_streams = previous_physical_streams = 0
    k_exposures = {}
    before_all = _f_counters(foundation)
    started = time.perf_counter()
    latest = sorted(stage_dir.glob("checkpoint_*.pt"))
    try:
        if latest:
            checkpoint = torch.load(latest[-1], map_location="cpu", weights_only=False)
            training.verify_contract(contract, checkpoint["contract"])
            extra = checkpoint["extra"]
            if optimizer is not None:
                updates = training.restore_bundle(checkpoint, model, optimizer, sampler, contract)
            else:
                model.load_state_dict(checkpoint["model"])
            curves, completed_epochs = extra["curves"], extra["completed_epochs"]
            best_mse, best_epoch, best_state = extra["best_mse"], extra["best_epoch"], extra["best_state"]
            best_mse = math.inf if best_mse is None else best_mse
            total_clipped, train_streams = extra["total_clipped"], extra["train_streams"]
            k_exposures = extra.get("k_exposures", {})
            previous_streams = extra.get("total_streams", 0)
            previous_physical_streams = extra.get("total_physical_streams", 0)
        status = "SEGMENT_COMPLETE"
        for epoch in [0.] + list(epochs):
            if epoch in completed_epochs:
                continue
            target = int(math.ceil(float(epoch) * len(legal))) if params else 0
            counters = _f_counters(foundation)
            if optimizer is not None and target > sampler.exposures:
                segment = training.train_segment(model, windows, legal, sampler, optimizer, target, predict,
                    effective_batch=int(spec.get("effective_batch", 16)), microbatch=int(spec.get("microbatch", 4)),
                    starting_updates=updates, budget_deadline=deadline, k_grid=grid, starting_k_exposures=k_exposures)
                updates = segment["updates"]
                k_exposures = segment["k_exposures"]
                total_clipped += int(segment.get("clipped_steps", 0))
                status = segment["status"]
            else:
                segment = dict(status="SEGMENT_COMPLETE", segment_seconds=0.)
            after_train = _f_counters(foundation)
            train_streams += after_train["streams"] - counters["streams"]
            if status != "BUDGET_LIMIT":
                try:
                    value, per_k = (None, []) if stops is None else _evaluate(model, windows, stops, predict, grid, int(spec.get("microbatch", 4)), deadline)
                except TimeoutError:
                    status = "BUDGET_LIMIT"
            if status != "BUDGET_LIMIT":
                curves.append(dict(epoch=float(epoch), exposures=sampler.exposures, updates=updates,
                                   astop_mse=value, segment_seconds=segment["segment_seconds"], clipped_steps=segment.get("clipped_steps", 0)))
                if value is not None and value < best_mse:
                    best_mse, best_epoch, best_state = value, float(epoch), training.cpu_state(model)
                elif stops is None:
                    best_epoch, best_state = float(epoch), training.cpu_state(model)
                completed_epochs.append(float(epoch))
            current_source, _ = _source_contract()
            if current_source != base_contract["source_sha"]:
                raise ValueError("Scientific source changed during run; cache may not be continued")
            extra = dict(curves=curves, completed_epochs=completed_epochs, best_mse=best_mse if math.isfinite(best_mse) else None,
                         best_epoch=best_epoch, best_state=best_state, total_clipped=total_clipped,
                         train_streams=train_streams, k_exposures=k_exposures,
                         total_streams=previous_streams + _f_counters(foundation)["streams"] - before_all["streams"],
                         total_physical_streams=previous_physical_streams + _f_counters(foundation)["physical_streams"] - before_all["physical_streams"])
            if optimizer is None:
                checkpoint = dict(model=training.cpu_state(model), contract=contract, updates=0, extra=extra)
            else:
                checkpoint = training.checkpoint_bundle(model, optimizer, sampler, contract, updates, extra)
            checkpoint_name = f"checkpoint_{len(list(stage_dir.glob('checkpoint_*.pt'))):06d}.pt"
            _torch_new(stage_dir / checkpoint_name, checkpoint)
            _json_new(public_dir / f"event_{len(list(public_dir.glob('event_*.json'))):06d}.json",
                      dict(epoch=epoch, status=status, exposures=sampler.exposures, updates=updates, contract=contract))
            if status == "BUDGET_LIMIT":
                break
        if latest and best_mse is None:
            best_mse = math.inf
        after_all = _f_counters(foundation)
        summary = dict(status=status, selected_epoch=best_epoch, selected_astop_mse=best_mse if math.isfinite(best_mse) else None,
                       completed_epochs=completed_epochs, updates=updates, exposures=sampler.exposures,
                       actual_epochs=sampler.exposures / len(legal), train_origins=len(legal), stop_origins=0 if stops is None else len(stops),
                       clipped_steps=total_clipped, clipping_fraction=total_clipped / updates if updates else 0.,
                       F_training_logical_streams=train_streams,
                       F_session_logical_streams=after_all["streams"] - before_all["streams"],
                       F_session_physical_streams=after_all["physical_streams"] - before_all["physical_streams"],
                       F_total_logical_streams=previous_streams + after_all["streams"] - before_all["streams"],
                       F_total_physical_streams=previous_physical_streams + after_all["physical_streams"] - before_all["physical_streams"],
                       seconds=time.perf_counter() - started, contract=contract,
                       trainable_parameters=sum(p.numel() for p in params), k_exposures=k_exposures,
                       optimizer="AdamW" if optimizer is not None else "NONE_FIXED_TRANSFORM", weight_decay=0.,
                       **_exposure_summary(prefix, legal, sampler.exposures, int(spec.get("seed", 0))))
        if str(windows.device).startswith("cuda"):
            summary.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                           peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        else:
            summary.update(peak_allocated_bytes=0, peak_reserved_bytes=0)
        return dict(model=best_state, normalization=normalization, summary=summary, curves=curves)
    finally:
        if callable(getattr(model, "close", None)):
            model.close()
        if foundation is not None and callable(getattr(foundation, "assert_frozen_unchanged", None)):
            foundation.assert_frozen_unchanged()


def _stage(spec, task, fit_end, stop_bounds, epochs, stage_dir, public_dir, foundation, deadline, base_contract):
    model, normalization = create_model(spec, task, fit_end, foundation)
    try:
        return _stage_body(spec, task, fit_end, stop_bounds, epochs, stage_dir, public_dir,
                           foundation, deadline, base_contract, model, normalization)
    except BaseException:
        # Also cover optimizer/window allocation failures before the inner loop.
        if callable(getattr(model, "close", None)):
            model.close()
        if foundation is not None and callable(getattr(foundation, "assert_frozen_unchanged", None)):
            foundation.assert_frozen_unchanged()
        raise


def fit_base(spec, task, cache_root, public_root, foundation, deadline):
    """Search A_fit/A_stop then freshly refit the selected epoch on A prefix60%."""
    spec = copy.deepcopy(dict(spec))
    run_id = str(spec["run_id"])
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one safe path component")
    epochs = list(spec.get("epochs", [.25, 1., 3.]))
    if not epochs or epochs != sorted(set(epochs)) or any(float(value) <= 0 for value in epochs):
        raise ValueError("Positive increasing epoch schedule required")
    source_sha, source_files = _source_contract()
    axis_contract = dict(L=int(task["L"]), H=int(task["H"]), C=int(task["physical"].shape[1]),
                         columns=list(task["columns"]), fold=spec.get("fold", "primary"),
                         source_array_fingerprints=task.get("array_sha256", {}))
    contract = dict(source_sha=source_sha, config_sha=training.fingerprint(dict(spec=spec, axes=axis_contract)),
                    data_sha=task["data_sha256"], model_sha=_model_sha(foundation))
    cache_run, public_run = Path(cache_root) / "runs" / run_id, Path(public_root) / "runs" / run_id
    cache_run.mkdir(parents=True, exist_ok=True)
    public_run.mkdir(parents=True, exist_ok=True)
    configuration = cache_run / "RUN_CONTRACT.json"
    if configuration.exists():
        if json.loads(configuration.read_text(encoding="utf-8")) != contract:
            raise ValueError("Run source/config/data/model contract changed; choose a new run_id")
    else:
        _json_new(configuration, contract)
        _json_new(public_run / "SOURCE_CONTRACT.json", dict(**contract, files=source_files, spec=spec))
    receipt_path = public_run / "BASE_FIT_RECEIPT.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not (Path(cache_root) / receipt["artifact_cache_relative_path"]).is_file():
            raise ValueError("Complete receipt has no selected private artifact")
        if data_contract.sha256_file(Path(cache_root) / receipt["artifact_cache_relative_path"]) != receipt["artifact_sha256"]:
            raise ValueError("Selected artifact bytes changed")
        return dict(receipt, reused_complete_artifact=True)
    bounds = data_contract.split_ranges(task, spec.get("fold", "primary"))
    search = _stage(spec, task, bounds["A_fit"][1], bounds["A_stop"], epochs,
                    cache_run / "search", public_run / "search", foundation, deadline, contract)
    if search["summary"]["status"] == "BUDGET_LIMIT":
        return dict(status="BUDGET_LIMIT", run_id=run_id, phase="search", training_status="UNDERTRAINED_OR_BUDGET_LIMIT", **{k:v for k,v in search["summary"].items() if k != "status"})
    selected_epoch = search["summary"]["selected_epoch"]
    refit = _stage(spec, task, bounds["A_stop"][1], None, [selected_epoch] if selected_epoch else [],
                   cache_run / "refit", public_run / "refit", foundation, deadline, contract)
    if refit["summary"]["status"] == "BUDGET_LIMIT":
        return dict(status="BUDGET_LIMIT", run_id=run_id, phase="refit", training_status="UNDERTRAINED_OR_BUDGET_LIMIT", **{k:v for k,v in refit["summary"].items() if k != "status"})
    artifact = cache_run / "selected_bundle.pt"
    bundle = dict(model=refit["model"], normalization=refit["normalization"], spec=spec,
                  contract=refit["summary"]["contract"], selected_search_epoch=selected_epoch,
                  search_summary=search["summary"], refit_summary=refit["summary"], model_state_sha=training.fingerprint({key:data_contract.array_sha256(value.numpy()) for key,value in refit["model"].items()}))
    _torch_new(artifact, bundle)
    for label, stage in (("search", search), ("refit", refit)):
        _json_new(public_run / label / "RECEIPT.json", stage["summary"])
        with (public_run / label / "CURVE.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "exposures", "updates", "astop_mse", "segment_seconds", "clipped_steps"])
            writer.writeheader(); writer.writerows(stage["curves"])
    training_status = "LIMITED_SCREEN" if max(epochs) < 3 else ("UNDERTRAINED_OR_BUDGET_LIMIT" if selected_epoch == max(epochs) else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED")
    search_values = [row["astop_mse"] for row in search["curves"] if row["astop_mse"] is not None]
    last_cp_best = bool(search_values and search_values[-1] <= min(search_values))
    last_scores_still_falling = bool(len(search_values) >= 2 and search_values[-1] < search_values[-2])
    receipt = dict(status="COMPLETE_VALID", training_status=training_status, run_id=run_id,
                   optimizer="AdamW", weight_decay=0., last_cp_best=last_cp_best,
                   last_scores_still_falling=last_scores_still_falling,
                   family=spec["family"], seed=int(spec.get("seed", 0)), epochs=epochs, selected_epoch=selected_epoch,
                   selected_astop_mse=search["summary"]["selected_astop_mse"],
                   search_fit_end=bounds["A_fit"][1], refit_fit_end=bounds["A_stop"][1],
                   search_train_origins=search["summary"]["train_origins"], search_stop_origins=search["summary"]["stop_origins"],
                   search_exposures=search["summary"]["exposures"], refit_exposures=refit["summary"]["exposures"],
                   search_updates=search["summary"]["updates"], refit_updates=refit["summary"]["updates"],
                   F_logical_streams=search["summary"]["F_total_logical_streams"] + refit["summary"]["F_total_logical_streams"],
                   artifact_cache_relative_path=artifact.relative_to(Path(cache_root)).as_posix(),
                   artifact_sha256=data_contract.sha256_file(artifact), model_state_sha=bundle["model_state_sha"],
                   normalization_fit_end=bounds["A_stop"][1], source_sha=source_sha,
                   refit_scaler_sha=refit["summary"]["contract"]["scaler_sha"], reused_complete_artifact=False)
    _json_new(receipt_path, receipt)
    return receipt
