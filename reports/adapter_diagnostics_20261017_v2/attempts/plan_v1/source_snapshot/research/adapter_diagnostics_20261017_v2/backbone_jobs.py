"""Explicit, budgeted backbone preparation and smoke jobs; import is source only."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import copy
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid
import zipfile

from .core import atomic_json, read, sha


def _resources(ctx):
    path = str(Path(ctx["repo"]) / "research/informs_broad_validation_20261015_v1")
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("resources")


def _download(ctx, url, destination, *, size, expected_sha=None):
    destination = Path(destination)
    if destination.exists() and (expected_sha is None or sha(destination) == expected_sha):
        return destination
    _resources(ctx).record_download(ctx["cache"], url, size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    count = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("xb") as output:
            while True:
                if time.time() >= ctx.get("job_deadline", ctx["deadline"]):
                    raise TimeoutError("download reached campaign deadline")
                block = source.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > size:
                    raise ValueError("download exceeded reserved byte upper bound")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if expected_sha is not None and sha(temporary) != expected_sha:
            raise ValueError("download SHA256 mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_dependencies(ctx):
    """Install only a private wheel overlay; preserve the existing environment.

    Uni2ts Python sources are reconciled byte-for-byte against the pinned
    official Git tree. This avoids silently labeling a PyPI wheel as that commit.
    """
    overlay = Path(ctx["cache"]) / "backbones/python_overlay"
    receipt = overlay / "DEPENDENCY_RECEIPT.json"
    if receipt.exists():
        value = read(receipt)
        for name, expected in value["source_hashes"].items():
            if sha(overlay / name) != expected:
                raise ValueError("private dependency overlay source changed")
        if str(overlay) not in sys.path:
            sys.path.insert(0, str(overlay))
        return value
    wheels = []
    for package, version in (("uni2ts", "2.0.0"), ("lightning", "2.6.6"),
                             ("jaxtyping", "0.2.38"), ("wadler-lindig", "0.1.7")):
        metadata = _download(ctx, f"https://pypi.org/pypi/{package}/{version}/json",
            Path(ctx["cache"]) / "backbones/downloads" / f"{package}-{version}.json", size=512 * 1024)
        item = next(w for w in read(metadata)["urls"] if w["filename"].endswith("py3-none-any.whl"))
        wheel = _download(ctx, item["url"], metadata.parent / item["filename"],
                          size=item["size"], expected_sha=item["digests"]["sha256"])
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                target = (overlay / member.filename).resolve()
                if not target.is_relative_to(overlay.resolve()):
                    raise ValueError("wheel path escapes private overlay")
            archive.extractall(overlay)
        wheels.append(dict(package=package, version=version, filename=item["filename"],
                           sha256=item["digests"]["sha256"], bytes=item["size"]))
    revision = "cfd46d4510ed8896f263116f32928eede05b0a75"
    tree_path = _download(ctx,
        f"https://api.github.com/repos/SalesforceAIResearch/uni2ts/git/trees/{revision}?recursive=1",
        Path(ctx["cache"]) / "backbones/downloads/uni2ts_tree.json", size=2 * 1024 * 1024)
    source_hashes = {}
    for entry in read(tree_path)["tree"]:
        name = entry["path"]
        if entry["type"] != "blob" or not name.startswith("src/uni2ts/") or not name.endswith(".py"):
            continue
        relative = name.removeprefix("src/")
        target = overlay / relative
        actual = target.read_bytes() if target.exists() else b""
        git_sha = hashlib.sha1(b"blob " + str(len(actual)).encode() + b"\0" + actual).hexdigest()
        if git_sha != entry["sha"]:
            # An older wheel is a dependency scaffold, never the pinned model source.
            if target.exists():
                target.unlink()
            _download(ctx, f"https://raw.githubusercontent.com/SalesforceAIResearch/uni2ts/{revision}/{name}",
                      target, size=entry["size"])
        actual = target.read_bytes()
        git_sha = hashlib.sha1(b"blob " + str(len(actual)).encode() + b"\0" + actual).hexdigest()
        if git_sha != entry["sha"]:
            raise ValueError("official uni2ts source Git blob mismatch")
        source_hashes[relative] = hashlib.sha256(actual).hexdigest()
    value = dict(wheels=wheels, uni2ts_source_revision=revision, source_hashes=source_hashes,
                 dependency_resolution="PRIVATE_NO_DEPS_OVERLAY; declared old version constraints not resolved; runtime smoke required")
    atomic_json(receipt, value)
    sys.path.insert(0, str(overlay))
    return value


def prepare_model(ctx, name):
    sources = read(Path(ctx["report"]) / "BACKBONE_SOURCES.json")["models"][name]
    directory = Path(ctx["cache"]) / "backbones/models" / name / sources["revision"]
    for filename in ("config.json", "README.md", "model.safetensors"):
        if filename == "model.safetensors":
            size, expected = sources["weights"]["bytes"], sources["weights"]["sha256"]
        else:
            item = next(s for s in sources["sources"] if s["url"].endswith("/" + filename))
            size, expected = item["bytes"], item["sha256"]
        _download(ctx, f"https://huggingface.co/{sources['repository']}/resolve/{sources['revision']}/{filename}",
                  directory / filename, size=size, expected_sha=expected)
    return directory


def private_bindings(ctx, name, extra_paths=()):
    """Bind every static runtime model/overlay file needed to reuse completion."""
    cache = Path(ctx["cache"]).resolve()
    source = read(Path(ctx["report"]) / "BACKBONE_SOURCES.json")["models"][name]
    model = cache / "backbones/models" / name / source["revision"]
    paths = [model / filename for filename in ("config.json", "README.md", "model.safetensors")]
    if name == "moirai2":
        overlay = cache / "backbones/python_overlay"
        paths.extend(path for path in overlay.rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and path.suffix != ".pyc")
    paths.extend(Path(path) for path in extra_paths)
    bound = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(cache) or not resolved.is_file() or path.is_symlink():
            raise ValueError("private runtime artifact is absent, indirect or outside campaign cache")
        reference = "cache://" + resolved.relative_to(cache).as_posix()
        bound[reference] = dict(reference=reference, sha256=sha(resolved))
    return [bound[key] for key in sorted(bound)]


def _sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@contextmanager
def sealed_split_ranges(data_module, task):
    """Keep original full-task boundaries while materializing only its pre-T prefix."""
    original = data_module.split_ranges
    boundaries = copy.deepcopy(task["split_ranges"])
    if boundaries["V_select"][1] != len(task["physical"]):
        raise ValueError("expected a prefix ending exactly at the sealed V boundary")
    def preserved(value, fold="primary"):
        if isinstance(value, dict):
            if value.get("data_sha256") != task["data_sha256"] or fold != "primary":
                raise ValueError("unregistered dataset or fold in prefix split binding")
            return copy.deepcopy(boundaries)
        return original(value, fold)
    data_module.split_ranges = preserved
    try:
        yield
    finally:
        data_module.split_ranges = original


def _check_deadline(ctx):
    if time.time() >= min(ctx.get("job_deadline", ctx["deadline"]), ctx["deadline"] - 4320):
        raise TimeoutError("backbone job reached finite boundary/reserve")
    if ctx.get("lease"):
        ctx["lease"].check()


def _task(ctx, modules, cohort):
    if cohort == "jena":
        from .existing_jobs import _jena
        return _jena(ctx, modules)
    if cohort == "pems08":
        from . import data_audit
        contract = read(modules["campaign"] / "DATA_CONTRACT.json")["assets"]["pems08"]
        primitive = data_audit.module(modules["campaign"] / "priority_b2_validation_20261016_v1/safe_jena_prefix.py",
                                      "backbone_pems_prefix_primitive")
        task, _ = data_audit._prepared_prefix("pems08", modules["old_cache"], contract, primitive)
        return task
    raise ValueError("unregistered native backbone cohort")


def _evaluate_saved(ctx, modules, task, backbone, *, model, norm, run_id, cohort):
    import numpy as np
    import torch
    from . import existing_jobs
    from .new_jobs import save_npz
    data = modules["data"]
    origins = data.origins(task["split_ranges"]["V_select"], task["L"], task["H"], task["H"])
    windows = modules["training"].FastWindows(task, norm["mean"], norm["std"], str(next(backbone.parameters()).device))
    predictions, targets, masks = [], [], []
    started = time.perf_counter()
    if model is not None:
        model.eval()
    with torch.no_grad():
        for start in range(0, len(origins), 4):
            _check_deadline(ctx)
            x, y, mask = windows.get(origins[start:start + 4])
            p = backbone(x) if model is None else model(x, backbone)
            if p.shape != y.shape or not torch.isfinite(p[mask]).all():
                raise ValueError("forecast shape/nonfinite observed values")
            predictions.append(p.cpu().numpy())
            targets.append(y.cpu().numpy())
            masks.append(mask.cpu().numpy())
    p, y, mask = map(np.concatenate, (predictions, targets, masks))
    path = Path(ctx["cache"]) / "backbone_predictions" / (run_id + ".npz")
    save_npz(path, pred_normalized=p, target_normalized=y, observed=mask,
             origins=origins, channel_ids=np.asarray(task["columns"]))
    # Reload private saved arrays and compare separate FP64 implementations.
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metrics = existing_jobs.score_arrays(arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"])
    independent, _ = modules["evaluation"].error_statistics(
        arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"],
        arrays["origins"], run_id, "V_select", cohort, arrays["channel_ids"])
    if max(abs(independent["mse"] - metrics["MSE"]), abs(independent["mae"] - metrics["MAE"])) > 1e-12:
        raise ValueError("independent saved prediction FP64 rescore mismatch")
    return dict(metrics, prediction_sha256=sha(path), private_prediction_reference="backbone_predictions/" + path.name,
                origins_sha256=data.array_sha256(origins),
                target_sha256=data.array_sha256(y), observed_sha256=data.array_sha256(mask),
                channel_order_sha256=data.array_sha256(np.asarray(task["columns"])),
                scaler_sha256=modules["fit"].scaler_fingerprint(norm["mean"], norm["std"]),
                independent_FP64_saved_rescore="PASS", normalized_V_inference_seconds=time.perf_counter() - started,
                E01_latency_seconds=None)


def _loaded(ctx, name, task, device):
    from .backbones import load_backbone
    if name == "moirai2":
        prepare_dependencies(ctx)
    local = prepare_model(ctx, name)
    return load_backbone(name, device=device, model_path=str(local), local_files_only=True,
                         length=task["L"], horizon=task["H"])


def matched_moirai(job, ctx):
    import numpy as np
    import torch
    from . import existing_jobs
    from .new_jobs import profile_training
    from .backbones import MOIRAI_REVISION
    modules = existing_jobs._modules(ctx)
    task = _task(ctx, modules, "jena")
    seed, device = int(job["seed"]), job.get("device", "cuda")
    backbone = _loaded(ctx, "moirai2", task, device)
    spec_base = dict(dataset="jena", latent_k=6, seed=seed, fold="primary", epochs=[.25, 1., 3.],
        lr=.001, device=device, microbatch=4, effective_batch=16, checkpoint_inputs=False,
        backbone="moirai2", backbone_revision=MOIRAI_REVISION, point_statistic="quantile_0.5",
        routing="independent_univariate", wrapper_source_sha256=sha(Path(__file__).with_name("backbones.py")),
        custom_source_sha256=sha(__file__))
    with sealed_split_ranges(modules["data"], task):
        # One warmup and five disposable real A_fit updates. Restores main RNG.
        profile = profile_training(ctx, modules, task, backbone,
            dict(spec_base, family="B2", run_id="DISPOSABLE_MOIRAI_PROFILE"))
        train = modules["data"].origins(task["split_ranges"]["A_fit"], 512, 48, 1)
        refit = modules["data"].origins((0, task["split_ranges"]["A_stop"][1]), 512, 48, 1)
        evaluation = modules["data"].origins(task["split_ranges"]["A_stop"], 512, 48, 48)
        endpoints = [0., .25, 1., 3.]
        def updates_for(n):
            targets = [int(np.ceil(e * n)) for e in endpoints]
            return sum(int(np.ceil((b - a) / 16)) for a, b in zip(targets, targets[1:]))
        search_updates, refit_updates = updates_for(len(train)), int(np.ceil(3 * len(refit) / 16))
        extra_eval_equivalent_updates = (2 * 4 * len(evaluation) + 43 * (2 + 21 / 6)) / 16
        estimate = (2 * (search_updates + refit_updates) + extra_eval_equivalent_updates) * profile["max_seconds"] * 1.5 + 120
        profile.update(conservative_full_trio_seconds=estimate, full_search_updates_per_arm=search_updates,
                       worst_case_refit_updates_per_arm=refit_updates, extra_load_eval_margin_seconds=120,
                       completed_trio_required_before_new_seed_dispatch=True)
        profile_path = Path(ctx["report"]) / "profiles" / (job["job_id"] + ".json")
        atomic_json(profile_path, profile)
        remaining = min(ctx.get("job_deadline", ctx["deadline"]), ctx["deadline"] - 4320) - time.time()
        if estimate > remaining:
            return dict(status="SKIPPED_BUDGET", reason="Measured complete Moirai Raw/B0/B2 trio exceeds remaining reserve",
                        artifact_paths=[str(profile_path)], result=profile)
        rows, private_paths = [], []
        for family in ("B0", "B2"):
            _check_deadline(ctx)
            run_id = f"moirai_jena_{family}_k6_s{seed}"
            spec = dict(spec_base, family=family, run_id=run_id)
            receipt = modules["fit"].fit_base(spec, task, ctx["cache"], ctx["report"], backbone,
                min(ctx.get("job_deadline", ctx["deadline"]), ctx["deadline"] - 4320))
            if receipt["status"] != "COMPLETE_VALID":
                return dict(status="PARTIAL_CHECKPOINTED", reason="Full matched trio checkpointed at finite boundary",
                            artifact_paths=[str(profile_path)], result=dict(profile=profile, completed_arms=rows))
            artifact = Path(ctx["cache"]) / receipt["artifact_cache_relative_path"]
            private_paths.append(artifact)
            bundle = torch.load(artifact, weights_only=False, map_location="cpu")
            model, norm = modules["fit"].create_model(spec, task, task["split_ranges"]["A_stop"][1], backbone)
            model.load_state_dict(bundle["model"])
            metrics = _evaluate_saved(ctx, modules, task, backbone, model=model, norm=norm, run_id=run_id, cohort="jena")
            private_paths.append(Path(ctx["cache"]) / metrics["private_prediction_reference"])
            sampler_checks = {}
            for stage, origins in (("search", train), ("refit", refit)):
                checkpoint_path = sorted((Path(ctx["cache"]) / "runs" / run_id / stage).glob("checkpoint_*.pt"))[-1]
                checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
                private_paths.append(checkpoint_path)
                stream = modules["training"].PermutationStream(origins, seed)
                order = stream.draw(checkpoint["sampler"]["exposures"])
                if stream.state_dict() != checkpoint["sampler"]:
                    raise ValueError("saved sampler does not match deterministic registered sequence")
                sampler_checks[stage] = dict(consumed_order_sha256=modules["data"].array_sha256(order),
                    checkpoint_sha256=sha(checkpoint_path), consumed_exposures=len(order), state_verified=True)
            rows.append(dict(method="Moirai_" + family, family=family, backbone="moirai2", backbone_revision=MOIRAI_REVISION,
                seed=seed, dataset="jena", cohort="jena", C=21, K=6, L=512, H=48, split="V_select", fit="fresh_A60",
                selected_epoch=receipt["selected_epoch"], trainable_params=sum(p.numel() for p in model.parameters()),
                total_resident_parameters=sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in backbone.parameters()),
                refit_updates=receipt["refit_updates"], refit_exposures=receipt["refit_exposures"],
                search_updates=receipt["search_updates"], search_exposures=receipt["search_exposures"],
                sampler_checks=sampler_checks, device=device, dtype="float32", point_statistic="quantile_0.5",
                F_logical_streams=receipt["F_logical_streams"], training_status=receipt["training_status"], **metrics))
            del model
        mean, std, _ = modules["data"].fit_statistics(task, task["split_ranges"]["A_stop"][1])
        metrics = _evaluate_saved(ctx, modules, task, backbone, model=None, norm=dict(mean=mean, std=std),
                                  run_id=f"moirai_jena_Raw_s{seed}", cohort="jena")
        private_paths.append(Path(ctx["cache"]) / metrics["private_prediction_reference"])
        rows.append(dict(method="Moirai_Raw", family="Raw", backbone="moirai2", backbone_revision=MOIRAI_REVISION,
            seed=seed, dataset="jena", cohort="jena", C=21, K=21, L=512, H=48, split="V_select",
            fit="NO_TARGET_PARAMETER_LEARNING", trainable_params=0, refit_updates=0, refit_exposures=0,
            total_resident_parameters=sum(p.numel() for p in backbone.parameters()), device=device, dtype="float32",
            point_statistic="quantile_0.5", **metrics))
        for identity in ("origins_sha256", "target_sha256", "observed_sha256", "channel_order_sha256", "scaler_sha256"):
            if len({row[identity] for row in rows}) != 1:
                raise ValueError("matched trio evaluation identity differs: " + identity)
        if rows[0]["sampler_checks"]["search"]["consumed_order_sha256"] != rows[1]["sampler_checks"]["search"]["consumed_order_sha256"]:
            raise ValueError("B0/B2 full search input order differs")
    backbone.assert_frozen_unchanged()
    value = dict(status="COMPLETE", rows=rows, profile=profile,
                 private_artifacts=private_bindings(ctx, "moirai2", private_paths),
                 limitation="Repeated development V; Raw has no target fit, adapters are supervised; selected refit updates may differ")
    output = Path(ctx["report"]) / "jobs" / (job["job_id"] + ".json")
    atomic_json(output, value)
    return dict(status="COMPLETE", reason=None, artifact_paths=[str(output), str(profile_path)], result=value)


def native_chronos(job, ctx):
    from . import existing_jobs
    from .backbones import CHRONOS2_REVISION
    modules = existing_jobs._modules(ctx)
    cohort = job.get("cohort", job["dataset_cohort"])
    task = _task(ctx, modules, cohort)
    backbone = _loaded(ctx, "chronos2", task, job.get("device", "cuda"))
    mean, std, _ = modules["data"].fit_statistics(task, task["split_ranges"]["A_stop"][1])
    metrics = _evaluate_saved(ctx, modules, task, backbone, model=None, norm=dict(mean=mean, std=std),
                              run_id="chronos2_native_" + cohort, cohort=cohort)
    channels = len(task["columns"])
    row = dict(method="Chronos2_nativeRaw", family="nativeRaw", backbone="chronos2", backbone_revision=CHRONOS2_REVISION,
        dataset=cohort, cohort=cohort, C=channels, K=channels, L=task["L"], H=task["H"], split="V_select",
        fit="NO_TARGET_PARAMETER_LEARNING", trainable_params=0, refit_updates=0, refit_exposures=0,
        total_resident_parameters=sum(p.numel() for p in backbone.parameters()),
        device=str(next(backbone.parameters()).device), dtype="float32", point_statistic="quantile_0.5",
        routing="native_multivariate_per_origin", target_mask="original observation mask",
        input_mask="All finite causal-filled histories supplied, matching existing Raw information; no future covariates",
        **metrics)
    backbone.assert_frozen_unchanged()
    value = dict(status="COMPLETE", rows=[row],
        private_artifacts=private_bindings(ctx, "chronos2", [Path(ctx["cache"]) / metrics["private_prediction_reference"]]),
        limitation="Practical alternative changes both pretrained backbone and native channel processing; not a single-factor P/E mechanism comparison")
    output = Path(ctx["report"]) / "jobs" / (job["job_id"] + ".json")
    atomic_json(output, value)
    return dict(status="COMPLETE", reason=None, artifact_paths=[str(output)], result=value)


def smoke(ctx, name, device):
    import numpy as np
    import psutil
    import torch

    from .backbones import (Moirai2Independent, assert_optimizer_excludes_backbone,
                            backbone_state_hash, load_backbone, routing_sentinel)

    if name == "moirai2":
        dependencies = prepare_dependencies(ctx)
    else:
        dependencies = {"chronos_forecasting": importlib.metadata.version("chronos-forecasting")}
    local = prepare_model(ctx, name)
    started = time.perf_counter()
    backbone = load_backbone(name, device=device, model_path=str(local), local_files_only=True)
    _sync(device)
    load_seconds = time.perf_counter() - started
    # Fixed FP32 tolerances registered before observing comparisons. No TF32.
    atol, rtol = 2e-5, 2e-5
    generator = torch.Generator(device="cpu").manual_seed(92601)
    x = torch.randn(2, 512, 3, generator=generator).to(device)
    output = backbone(x)
    if not torch.isfinite(output).all():
        raise ValueError("nonfinite backbone prediction")
    if name == "moirai2":
        arrays = list(x.transpose(1, 2).reshape(6, 512, 1).cpu().numpy())
        official = torch.as_tensor(backbone.forecast.predict(arrays), device=device)
        official = official[:, backbone.median_index].reshape(2, 3, 48).transpose(1, 2)
        from uni2ts.transform.imputation import CausalMeanImputation
        raw = arrays[0].copy()
        raw[1:4] = np.nan
        observed = torch.as_tensor(~np.isnan(raw), device=device).reshape(1, 512, 1)
        filled = torch.as_tensor(CausalMeanImputation()(raw), device=device).reshape(1, 512, 1)
        masked = backbone(filled, observed)
        official_masked = torch.as_tensor(backbone.forecast.predict([raw]), device=device)
        official_masked = official_masked[:, backbone.median_index, :, None]
    else:
        from chronos.chronos2.pipeline import Chronos2Pipeline
        pipeline = Chronos2Pipeline(backbone.model)
        official_quantiles = pipeline.predict(list(x.transpose(1, 2)), prediction_length=48,
            context_length=512, batch_size=6, cross_learning=False)
        official = torch.stack([v[:, backbone.median_index] for v in official_quantiles]).transpose(1, 2).to(device)
        observed = torch.ones_like(x, dtype=torch.bool)
        observed[:, 1:4, 0] = False
        raw = x.masked_fill(~observed, float("nan"))
        masked = backbone(x, observed)
        native = pipeline.predict(list(raw.transpose(1, 2)), prediction_length=48,
            context_length=512, batch_size=6, cross_learning=False)
        official_masked = torch.stack([v[:, backbone.median_index] for v in native]).transpose(1, 2).to(device)
    torch.testing.assert_close(output, official, atol=atol, rtol=rtol)
    torch.testing.assert_close(masked, official_masked, atol=atol, rtol=rtol)
    sentinel = routing_sentinel(backbone, x, atol=atol, rtol=rtol)
    before = backbone_state_hash(backbone)
    gradient = {"status": "NOT_APPLICABLE_NATIVE_RAW"}
    fit_seconds = None
    if name == "moirai2":
        from research.informs_broad_validation_20261015_v1.networks import ForecastAdapter
        with torch.random.fork_rng(devices=[torch.device(device)] if str(device).startswith("cuda") else []):
            torch.manual_seed(92601)
            adapter = ForecastAdapter("B2", 512, 48, 3, 2,
                np.zeros(3, np.float32), np.eye(3, dtype=np.float32)).to(device)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=0)
            assert_optimizer_excludes_backbone(optimizer, backbone)
            steps = []
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                _sync(device)
                tick = time.perf_counter()
                z = adapter.encode(x)
                z.retain_grad()
                prediction = adapter.decode(x, z, backbone(z))
                prediction.square().mean().backward()
                gradients = {"Z": z.grad, "E": adapter.E.grad,
                    "P_output": adapter.front.mixer.temporal.output.weight.grad,
                    "P_interior": adapter.front.mixer.temporal.input.weight.grad}
                stats = {key: {"finite": bool(torch.isfinite(value).all()),
                    "l1": float(value.abs().sum().cpu())} for key, value in gradients.items()}
                for key in ("Z", "E", "P_output") + (("P_interior",) if step else ()):
                    if not stats[key]["finite"] or stats[key]["l1"] <= 0:
                        raise ValueError("required input/adapter gradient is zero or nonfinite: " + key)
                optimizer.step()
                _sync(device)
                steps.append(dict(step=step, seconds=time.perf_counter() - tick, gradients=stats))
            fit_seconds = [s["seconds"] for s in steps]
            gradient = {"status": "PASS", "steps": steps,
                        "profile_scope": "synthetic B2 C3K2; not a real Jena completion-time predictor"}
    if before != backbone_state_hash(backbone):
        raise ValueError("frozen backbone weights changed")
    state_path = Path(ctx["cache"]) / "backbones" / (name + "_roundtrip.pt")
    torch.save(backbone.state_dict(), state_path)
    backbone.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    torch.testing.assert_close(output, backbone(x), atol=0, rtol=0)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    measurements = []
    rss = []
    for iteration in range(7):
        _sync(device)
        tick = time.perf_counter()
        backbone(x)
        _sync(device)
        seconds = time.perf_counter() - tick
        rss.append(psutil.Process().memory_info().rss)
        if iteration >= 2:
            measurements.append(seconds)
    return dict(status="PASS", backbone=name, device=device, dtype="float32", point_statistic="quantile_0.5",
        median_index=backbone.median_index, shape=list(output.shape), routing=backbone.routing,
        official_parity_max_abs=float((output - official).abs().max().cpu()),
        mask_parity_max_abs=float((masked - official_masked).abs().max().cpu()),
        tolerance={"atol": atol, "rtol": rtol, "registered_before_measurement": True},
        routing_sentinel=sentinel, adapter_gradient=gradient, backbone_weight_hash=before,
        frozen_weights_unchanged=True, save_load_prediction_parity=True,
        cold_load_seconds=load_seconds, synthetic_fit_step_seconds=fit_seconds,
        inference_seconds=measurements, median_inference_seconds=float(np.median(measurements)),
        inference_peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None,
        inference_peak_reserved_bytes=torch.cuda.max_memory_reserved() if str(device).startswith("cuda") else None,
        rss_sample_max_bytes=max(rss), parameter_count=sum(p.numel() for p in backbone.parameters()),
        trainable_backbone_parameters=sum(p.numel() for p in backbone.parameters() if p.requires_grad),
        dependencies=dependencies, cost_scope="synthetic resident F path B2,L512,C3; not E01 E2E or valid full-training profile")


def execute_job(job, ctx):
    if job["question_id"] == "D02":
        return matched_moirai(job, ctx)
    if job["question_id"] == "D03":
        return native_chronos(job, ctx)
    if job["question_id"] != "D01":
        raise ValueError("unregistered backbone question")
    name = job.get("backbone", job["family"])
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = job.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    result = smoke(ctx, name, device)
    result["private_artifacts"] = private_bindings(ctx, name,
        [Path(ctx["cache"]) / "backbones" / (name + "_roundtrip.pt")])
    path = Path(ctx["report"]) / "jobs" / (job["job_id"] + ".json")
    atomic_json(path, result)
    return dict(status="COMPLETE", reason=None, artifact_paths=[str(path)], result=result)
