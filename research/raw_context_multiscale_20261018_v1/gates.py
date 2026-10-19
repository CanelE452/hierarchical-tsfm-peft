"""TRAIN-only real-foundation parity, gradient and optional resource gates."""
import argparse
import copy
import gc
from pathlib import Path
import re
import time

import numpy as np
import torch

from research.adapter_improvements_20261018_v1.audit import load_foundation
from research.b2_revin_20261018_v1.training_jobs import _state_fingerprint
from . import common, models, training


def _check_deadline(deadline):
    if deadline is not None and time.time() >= float(deadline):
        raise TimeoutError("Registered wall/GPU budget reached during gates")


def _close(left, right):
    torch.testing.assert_close(left, right, atol=common.ATOL, rtol=common.RTOL)
    return dict(max_absolute_difference=float((left.detach() - right.detach()).abs().max().cpu()),
                shape=list(left.shape), atol=common.ATOL, rtol=common.RTOL)


def _finite(value, name, nonzero=False):
    if value is None or not torch.isfinite(value).all() or (nonzero and not bool(value.abs().sum() > 0)):
        raise AssertionError("Missing/nonfinite/zero required gradient: " + name)


def _loss(prediction, target, observed):
    counts = observed.sum((0, 1))
    active = counts > 0
    return (torch.where(observed, (prediction - target).square(), 0).sum((0, 1))[active] / counts[active]).mean()


def _gradient_run(model, foundation, x, y, mask):
    model.zero_grad(set_to_none=True)
    window = x.detach().clone().requires_grad_()
    output = model(window, foundation)
    _loss(output, y, mask).backward()
    _finite(window.grad, "input", nonzero=True)
    gradients = {}
    for name, value in model.named_parameters():
        _finite(value.grad, name)
        gradients[name] = value.grad.detach().clone()
    return output.detach(), window.grad.detach().clone(), gradients


def _initial_hashes(model):
    return dict(P=_state_fingerprint(model.front.state_dict()), E=_state_fingerprint(model.E),
                D=_state_fingerprint(model.D), center=_state_fingerprint(model.center))


def _train_windows(ctx, task, device, norm, count):
    _check_deadline(ctx.get("deadline"))
    end = int(task["split_ranges"]["A_fit"][1])
    origins = np.arange(task["L"], task["L"] + count, dtype=np.int64)
    if origins[-1] + task["H"] > end:
        raise ValueError("Gate windows must lie entirely inside A_fit")
    prefix = dict(task, physical=task["physical"][:end], observed=task["observed"][:end])
    windows = ctx["modules"]["training"].FastWindows(prefix, norm["mean"], norm["std"], device)
    _check_deadline(ctx.get("deadline"))
    return windows.get(origins), origins


def _dataset_gates(ctx, task, foundation, device):
    _check_deadline(ctx.get("deadline"))
    def bounded_foundation(z):
        _check_deadline(ctx.get("deadline"))
        return foundation(z)
    fit_end = int(task["split_ranges"]["A_fit"][1])
    records = dict(dataset=task["dataset"], origins=[task["L"], task["L"] + 1],
                   fitting_scope="A_fit", target_scope="TRAIN_only", CP0=[])
    chosen = None
    with models.model_factory(training._factory_context(ctx)) as create:
        for seed in common.SEEDS:
            _check_deadline(ctx.get("deadline"))
            initialized, norms = {}, {}
            for mode in ("S", "A", "B"):
                _check_deadline(ctx.get("deadline"))
                spec = training.job_spec(task["dataset"], mode, seed, device)
                initialized[mode], norms[mode] = create(spec, task, fit_end, foundation)
                _check_deadline(ctx.get("deadline"))
            hashes = {mode: _initial_hashes(model) for mode, model in initialized.items()}
            if hashes["S"] != hashes["A"] or hashes["S"] != hashes["B"]:
                raise AssertionError("Matched seed P/E/D/PCA initialization changed")
            counts = {mode: sum(value.numel() for value in model.parameters()) for mode, model in initialized.items()}
            expected = 36725 if task["dataset"] == "jena" else 37472
            if set(counts.values()) != {expected}:
                raise AssertionError("Registered trainable parameter count changed")
            (x, y, mask), _ = _train_windows(ctx, task, device, norms["S"], 2)
            with torch.no_grad():
                parity = _close(initialized["S"](x, bounded_foundation), initialized["A"](x, bounded_foundation))
            records["CP0"].append(dict(seed=seed, initial_hashes=hashes, parameter_counts=counts,
                                       S_A_output_parity=parity))
            if seed == common.SEEDS[0]:
                chosen = initialized, norms, x, y, mask
    initialized, norms, x, y, mask = chosen
    b = initialized["B"]
    before = {name: value.detach().clone() for name, value in b.named_parameters()}
    foundation_ids = {id(value) for value in foundation.parameters()}
    optimizer = torch.optim.AdamW(b.parameters(), lr=.001, weight_decay=0)
    if any(id(value) in foundation_ids for group in optimizer.param_groups for value in group["params"]):
        raise AssertionError("Frozen F entered the adapter optimizer")
    calls = []

    def observed_foundation(z):
        z.retain_grad()
        result = bounded_foundation(z)
        calls.append((z, list(result.shape), id(foundation)))
        return result

    for update in range(2):
        _check_deadline(ctx.get("deadline"))
        optimizer.zero_grad(set_to_none=True)
        window = x.detach().clone().requires_grad_()
        prediction = b(window, observed_foundation)
        _loss(prediction, y, mask).backward()
        _finite(window.grad, "B_original_input", nonzero=True)
        for index, (value, _, _) in enumerate(calls[-2:]):
            _finite(value.grad, "B_shared_F_branch_" + str(index), nonzero=True)
        for name, value in b.named_parameters():
            _finite(value.grad, name)
        if update == 0:
            hidden = b.front.mixer.temporal.input.weight.grad
            records["first_hidden_gradient_zero_is_legitimate"] = bool(torch.count_nonzero(hidden) == 0)
        torch.nn.utils.clip_grad_norm_(b.parameters(), 1)
        _check_deadline(ctx.get("deadline"))
        optimizer.step()
    changed = {prefix: any(not torch.equal(before[name], value) for name, value in b.named_parameters()
                           if name.startswith(prefix)) for prefix in ("front.", "E", "D")}
    if not all(changed.values()):
        raise AssertionError("P/E/D did not change after two allowed TRAIN updates")
    records["training_gradient_path"] = dict(two_updates_finite=True, changed=changed,
        foundation_excluded_from_optimizer=True, branch_input_gradients_finite_and_nonzero=True,
        shared_instance_count=len({value[2] for value in calls}),
        calls=[dict(input_shape=list(value.shape), returned_shape=shape) for value, shape, _ in calls[:2]],
        used_fine_horizon=task["H"], used_coarse_horizon=task["H"] // 2,
        original_grid_time_contract=models.time_contract(task["L"], task["H"]))
    del calls, optimizer
    a = models.convert_model(copy.deepcopy(b), "A")
    b1 = models.convert_model(copy.deepcopy(b), "B", scale=1)
    output_a, input_a, parameters_a = _gradient_run(a, bounded_foundation, x, y, mask)
    output_b1, input_b1, parameters_b1 = _gradient_run(b1, bounded_foundation, x, y, mask)
    parameter_differences = {name: _close(parameters_a[name], parameters_b1[name])["max_absolute_difference"]
                             for name in parameters_a}
    records["A_B_scale1_parity"] = dict(output=_close(output_a, output_b1), input_gradient=_close(input_a, input_b1),
        max_parameter_gradient_difference=max(parameter_differences.values()),
        parameter_gradient_differences=parameter_differences, trained_P_state_used=True)
    two_call = models.convert_model(copy.deepcopy(a), "A", schedule="two_call")
    with torch.no_grad():
        records["A_native_two_call_output_parity"] = _close(a(x, bounded_foundation), two_call(x, bounded_foundation))
        records["batch_independence"] = {}
        for mode, model in (("A", a), ("B", b)):
            _check_deadline(ctx.get("deadline"))
            together = model(x, bounded_foundation)
            singles = torch.cat([model(value[None], bounded_foundation) for value in x])
            records["batch_independence"][mode] = _close(together, singles)
        canary_task = dict(task, physical=task["physical"][:fit_end].copy(), observed=task["observed"][:fit_end].copy())
        canary_task["physical"][task["L"]:task["L"] + task["H"]] = 1e9
        (canary_x, canary_y, _), _ = _train_windows(ctx, canary_task, device, norms["S"], 1)
        if not torch.equal(canary_x, x[:1]) or torch.equal(canary_y, y[:1]):
            raise AssertionError("Future canary changed history or failed to change targets")
        records["future_canary"] = {mode: _close(model(canary_x, bounded_foundation), model(x[:1], bounded_foundation))
                                    for mode, model in (("A", a), ("B", b))}
        past, future = b.forecast_latents(x, bounded_foundation)
        anchor_control = copy.deepcopy(b)
        anchor_control.D.zero_()
        records["original_anchor_once"] = _close(anchor_control.decode(x, past, future),
                                                 x[:, -1:].expand(-1, task["H"], -1))
    _check_deadline(ctx.get("deadline"))
    foundation.assert_frozen_unchanged()
    if any(value.requires_grad or value.grad is not None for value in foundation.parameters()):
        raise AssertionError("Frozen F parameters changed gradient policy")
    records["frozen_F_weights_unchanged"] = True
    return records


def _profile_step(model, foundation, x, y, mask, *, deadline=None):
    _check_deadline(deadline)
    optimizer = model._profile_optimizer
    optimizer.zero_grad(set_to_none=True)
    counts = mask.sum((0, 1))
    active = counts > 0
    for start in range(0, len(x), 4):
        _check_deadline(deadline)
        prediction = model(x[start:start + 4], foundation)
        sse = torch.where(mask[start:start + 4], (prediction - y[start:start + 4]).square(), 0).sum((0, 1))
        (sse[active] / counts[active]).mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    _check_deadline(deadline)
    optimizer.step()


def _profile(ctx, task, foundation, device):
    _check_deadline(ctx.get("deadline"))
    def bounded_foundation(z):
        _check_deadline(ctx.get("deadline"))
        return foundation(z)
    rows = []
    with models.model_factory(training._factory_context(ctx)) as create:
        for mode in ("S", "A", "B"):
            _check_deadline(ctx.get("deadline"))
            spec = training.job_spec(task["dataset"], mode, common.SEEDS[0], device)
            model, norm = create(spec, task, task["split_ranges"]["A_fit"][1], foundation)
            _check_deadline(ctx.get("deadline"))
            (x, y, mask), origins = _train_windows(ctx, task, device, norm, 16)
            model._profile_optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
            torch.cuda.synchronize(device)
            warm_started = time.perf_counter()
            _profile_step(model, bounded_foundation, x, y, mask, deadline=ctx.get("deadline"))
            torch.cuda.synchronize(device)
            warm_seconds = time.perf_counter() - warm_started
            torch.cuda.reset_peak_memory_stats(device)
            durations = []
            before = {name: int(getattr(foundation, name)) for name in ("streams", "row_calls", "physical_streams", "physical_calls")}
            for _ in range(5):
                _check_deadline(ctx.get("deadline"))
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                _profile_step(model, bounded_foundation, x, y, mask, deadline=ctx.get("deadline"))
                torch.cuda.synchronize(device)
                durations.append(time.perf_counter() - started)
            after = {name: int(getattr(foundation, name)) for name in before}
            rows.append(dict(mode=mode, dataset=task["dataset"], effective_batch=16, microbatch=4,
                F_row_microbatch=128, origins=origins.tolist(), warmup_updates=1, warmup_seconds=warm_seconds,
                timed_updates=5, update_seconds=durations, median_update_seconds=float(np.median(durations)),
                measured_updates=5, seconds_per_update=float(np.median(durations)),
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                F_counters={name: after[name] - before[name] for name in before},
                target_scope="TRAIN_only", discarded_profile_training=True))
            del model
    return rows


def run_gates(ctx, device="cpu", profile=False, output_id=None):
    """Load one actual frozen foundation per task H; never evaluate A_stop/V/T."""
    device = str(device)
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("Gate device must be cpu or cuda")
    if profile and not device.startswith("cuda"):
        raise ValueError("The registered resource profile requires CUDA")
    if output_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", output_id):
        raise ValueError("Output ID must be one safe component")
    suffix = "GPU" if device.startswith("cuda") else "CPU"
    name = "VALIDATION_" + ("PROFILE_" if profile else "SMOKE_") + suffix
    path = Path(ctx["report"]) / (name + ("_" + output_id if output_id else "") + ".json")
    if path.exists():
        raise FileExistsError("Preserve the sealed gate run; choose a fresh output_id")
    started = time.perf_counter()
    local = dict(ctx, device=device)
    sources = dict(common.model_sources(ctx["repo"]))
    sources[Path(__file__).resolve().relative_to(Path(ctx["repo"]).resolve()).as_posix()] = common.sha(__file__)
    value = dict(status="RUNNING", device=device, profile=bool(profile), tolerances=dict(atol=common.ATOL, rtol=common.RTOL),
        scientific_source_hashes=sources, datasets=[], profiles=[], target_scope="TRAIN_only",
        new_A_stop_V_T_scores=0, original_grid_unit_suite="tests/test_raw_context_multiscale_models.py",
        CUDA_requested=device.startswith("cuda"), GPU_seconds=0., foundation_load_seconds=0.)
    try:
        _check_deadline(local.get("deadline"))
        tasks = local.get("tasks")
        if not tasks:
            tasks = {}
            for name in common.EXPECTED:
                _check_deadline(local.get("deadline"))
                tasks[name] = common.load_task(local, name)
        for dataset in common.EXPECTED:
            _check_deadline(local.get("deadline"))
            task = tasks[dataset]
            load_started = time.perf_counter()
            foundation = load_foundation(local, local["modules"], task["H"], device=device)
            load_seconds = time.perf_counter() - load_started
            value["foundation_load_seconds"] += load_seconds
            _check_deadline(local.get("deadline"))
            requested = torch.device(device)
            if any(parameter.device.type != requested.type or (requested.index is not None
                   and parameter.device.index != requested.index) for parameter in foundation.parameters()):
                raise AssertionError("Foundation parameters are on the wrong gate device")
            result = _dataset_gates(local, task, foundation, device)
            result.update(foundation_class=type(foundation).__name__, foundation_weights_sha256=foundation.initial_hash,
                foundation_load_seconds=load_seconds, wrapper_requested_horizon=int(foundation.horizon),
                actual_fixed_F_head_horizon=int(foundation.model.chronos_config.prediction_length),
                foundation_copies=1, row_microbatch=int(foundation.row_microbatch))
            value["datasets"].append(result)
            if profile:
                profiles = _profile(local, task, foundation, device)
                for row in profiles:
                    row["F_load_seconds"] = load_seconds
                value["profiles"].extend(profiles)
            foundation.assert_frozen_unchanged()
            del foundation
            gc.collect()
        _check_deadline(local.get("deadline"))
        if any(common.sha(Path(ctx["repo"]) / name) != expected for name, expected in sources.items()):
            raise ValueError("Gate scientific source bytes changed during execution")
        value["status"] = "PASS"
    except (AssertionError, ValueError, RuntimeError, FileNotFoundError, TimeoutError) as error:
        value.update(status="FAIL", error_type=type(error).__name__, reason=str(error))
    value["wall_seconds"] = time.perf_counter() - started
    value["GPU_seconds"] = value["wall_seconds"] if device.startswith("cuda") else 0.
    value["discarded_preparation_and_profile_seconds"] = value["wall_seconds"]
    common.sealed_json(path, value)
    common.event(ctx, "gates_" + suffix.lower(), value["status"], wall_seconds=value["wall_seconds"],
                 GPU_seconds=value["GPU_seconds"], output=common.binding(path, ctx), profile=bool(profile))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-id")
    args = parser.parse_args(argv)
    ctx = common.context(args.asset_root, device=args.device)
    value = run_gates(ctx, device=args.device, profile=args.profile, output_id=args.output_id)
    print({key: value[key] for key in ("status", "device", "wall_seconds", "foundation_load_seconds")})
    return 0 if value["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
