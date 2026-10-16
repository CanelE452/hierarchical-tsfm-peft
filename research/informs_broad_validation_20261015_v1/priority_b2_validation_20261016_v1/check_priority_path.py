"""Small explicit P0 check; the default CLI opens no inputs and imports no Torch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

NAME = "priority_b2_validation_20261016_v1"
L, H, C, K = 512, 48, 21, 6
R4_ID = "postT_cpu_jena_E3_B2_k6_s92601_v1"
BINDINGS_REL = "postT_V_only_cpu_lag_20261016_v1/diagnostics/final_cpu_comparison_input_root_20261016_v1/SOURCE_BINDINGS.json"
BINDINGS_SHA = "22f7a756a224da46001a4c42b19ce393611133e79ebcc96180e099dff99e9ff4"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def inside(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
        raise ValueError("Explicit paths must not contain symlinks or junctions")
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Explicit path escapes its root")
    return path


def deadline_check(deadline):
    if time.time() >= deadline:
        raise TimeoutError("P0 deadline reached")


def validate_plan(plan):
    if plan.get("status") != "SEALED_BY_ROOT" or plan.get("namespace") != NAME:
        raise ValueError("An explicit ROOT-sealed priority plan is required")
    task = plan["task_contract"]
    if tuple(task.get(key) for key in ("dataset", "L", "H", "C", "K", "seed")) != ("jena", L, H, C, K, 92601):
        raise ValueError("Only the sealed primary Jena L512/H48/C21/K6/seed92601 contract")
    if task.get("fold") != "primary" or plan.get("materialized_prefix_end_exclusive") != 16772:
        raise ValueError("Only the existing primary pre-T prefix")
    if plan.get("protected_tail_numeric_parse_allowed") is not False or plan.get("new_T_selection_or_target_computation") is not False:
        raise ValueError("Protected T must remain prohibited")
    if plan["resource_limits"]["CPU_threads"] != 2 or plan["R4"]["run_id"] != R4_ID:
        raise ValueError("CPU2 and the exact existing R4 are required")
    for field in ("p0_checker_sha256", "prefix_loader_sha256", "foundation_tensor_sha256", "model_asset_relative_path",
                  "instance_norm_source_relative_path", "instance_norm_source_sha256", "stop_identity_snapshot"):
        if not plan.get(field):
            raise ValueError("Missing sealed field: " + field)
    deadline_check(plan["deadline_unix"])


def stopped_identities(identities, lookup):
    checked = []
    if not identities:
        raise ValueError("No owned process identities to check")
    for item in identities:
        pid, created = int(item["pid"]), float(item["create_time"])
        current = lookup(pid)
        if current is not None and abs(float(current) - created) < .01:
            raise RuntimeError("A stopped campaign process identity is still alive")
        checked.append(dict(pid=pid, create_time=created, old_identity_dead=True, PID_reused=current is not None))
    return checked


def counter_delta(before, after):
    return {name: int(after[name]) - int(before[name]) for name in ("streams", "physical_streams", "physical_calls")}


def validate_gradient_groups(groups):
    for name in ("P", "E", "D"):
        group = groups[name]
        if group["gradient_tensors_present"] < 1 or not group["all_present_gradients_finite"]:
            raise AssertionError("Missing or nonfinite P/E/D gradient connection: " + name)


class InputGradientProbe:
    def __init__(self, module, finite):
        self.contexts, self.backward = [], []
        self.finite = finite
        self.tensor_handles = []
        self.handle = module.register_forward_pre_hook(self.forward_hook, with_kwargs=True)

    def forward_hook(self, module, args, kwargs):
        context = kwargs.get("context", args[0] if args else None)
        if context is None or not context.requires_grad:
            raise AssertionError("Actual F-module context has no gradient connection")
        self.contexts.append(list(context.shape))
        self.tensor_handles.append(context.register_hook(self.backward_hook))

    def backward_hook(self, gradient):
        self.backward.append(dict(shape=list(gradient.shape), finite=bool(self.finite(gradient))))
        return gradient

    def close(self):
        self.handle.remove()
        for handle in self.tensor_handles:
            handle.remove()

    def evidence(self):
        if not self.contexts or len(self.backward) != len(self.contexts) or not all(x["finite"] for x in self.backward):
            raise AssertionError("Actual F input tensor hook did not receive finite backward gradients")
        return dict(module_context_shapes=self.contexts, input_backward_hooks=self.backward)


def transparent_raw_wrapper(physical, observed, mean, std, foundation, data, np, torch):
    physical, observed = np.asarray(physical, np.float32), np.asarray(observed)
    if observed.all():
        filled = physical.copy()
    else:
        filled = np.stack([data.causal_forward_fill(x, mask, mean) for x, mask in zip(physical, observed)])
    normalized = ((filled - mean) / std).astype(np.float32)
    x = torch.as_tensor(normalized, dtype=torch.float32, device="cpu")
    prediction = foundation(x)
    std_t, mean_t = torch.as_tensor(std, device="cpu"), torch.as_tensor(mean, device="cpu")
    return (prediction.double() * std_t.double() + mean_t.double()).cpu().numpy()


def execute(args):
    public, cache, assets = (Path(getattr(args, key)).resolve() for key in ("public_root", "cache_root", "asset_root"))
    plan_path = Path(args.plan).resolve()
    if sha(plan_path) != args.plan_sha:
        raise ValueError("Plan SHA differs")
    plan = read(plan_path)
    validate_plan(plan)
    scope = public / NAME
    output = Path(args.output).resolve()
    if not output.is_relative_to(scope / "diagnostics") or output.exists():
        raise ValueError("Use a fresh output below public priority diagnostics")
    for key, wanted in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2")):
        if os.environ.get(key, wanted) != wanted:
            raise ValueError("CPU2/offline environment differs: " + key)
        os.environ[key] = wanted
    pinned = {str(plan_path): args.plan_sha}
    for field in ("source39", "immutable_public"):
        for relative, expected in plan[field].items():
            pinned[str(inside(public, relative))] = expected
    pinned[str(scope / "check_priority_path.py")] = plan["p0_checker_sha256"]
    pinned[str(scope / "safe_jena_prefix.py")] = plan["prefix_loader_sha256"]
    pinned[str(inside(assets, plan["instance_norm_source_relative_path"]))] = plan["instance_norm_source_sha256"]
    for field in ("stop_proof", "scheduler_blocked", "stop_identity_snapshot"):
        descriptor = plan[field]
        pinned[str(inside(cache, descriptor["cache_relative_path"]))] = descriptor["sha256"]
    for field in ("fit_receipt", "V_manifest"):
        pinned[str(inside(public, plan["R4"][field + "_public_relative_path"]))] = plan["R4"][field + "_sha256"]
    pinned[str(inside(public, BINDINGS_REL))] = BINDINGS_SHA
    for path, expected in pinned.items():
        if sha(path) != expected:
            raise ValueError("A sealed source or metadata input changed")
    stop = read(inside(cache, plan["stop_proof"]["cache_relative_path"]))
    if stop.get("remaining_owned_processes") != [] or stop.get("scheduler_dispatch_disabled") is not True:
        raise RuntimeError("Existing queue closure is not proven")
    import psutil
    def lookup(pid):
        try:
            return psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            return None
    snapshot = read(inside(cache, plan["stop_identity_snapshot"]["cache_relative_path"]))
    closure = stopped_identities(snapshot["owned_tree"], lookup)
    bindings = read(inside(public, BINDINGS_REL))
    reused = next(row for row in bindings["routes"] if row["run_id"] == R4_ID)
    if reused["accuracy"]["manifest"]["sha256"] != plan["R4"]["V_manifest_sha256"]:
        raise ValueError("Existing R4 validation metadata describes a different output")
    for proof in [reused["accuracy"]["independent_FP64"]["result"], *reused["CPU_checkpoint_replay"]["proofs"]]:
        p = inside(public, proof["relative_path"])
        if sha(p) != proof["sha256"]:
            raise ValueError("Existing R4 proof metadata changed")
        pinned[str(p)] = proof["sha256"]
    artifact = inside(cache, plan["R4"]["artifact_cache_relative_path"])
    if sha(artifact) != plan["R4"]["artifact_sha256"]:
        raise ValueError("Exact R4 artifact SHA differs")
    pinned[str(artifact)] = plan["R4"]["artifact_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    pool = route = None
    started = time.time()
    try:
        deadline_check(plan["deadline_unix"])
        sys.path.insert(0, str(public))
        sys.path.insert(0, str(scope))
        import numpy as np
        import torch
        import data_contract as data
        import deployment
        import fit_base as fit
        import gpu_runtime
        import networks
        import safe_jena_prefix
        import training
        torch.set_num_threads(2)
        torch.manual_seed(92601)
        task, access = safe_jena_prefix.load_jena_prefix(assets, read(public / "DATA_CONTRACT.json")["assets"]["jena"])
        if access["new_T_numeric_values_parsed"] != 0 or len(task["physical"]) != 16772:
            raise ValueError("Prefix loader crossed the protected numeric boundary")
        if task["data_sha256"] != plan["task_contract"]["data_sha256"] or task["columns"] != plan["task_contract"]["columns"]:
            raise ValueError("Inherited task identity/channel order differs")
        bundle = torch.load(artifact, map_location="cpu", weights_only=False)
        tensor_sha = training.fingerprint({name: data.array_sha256(value.detach().cpu().numpy()) for name, value in bundle["model"].items()})
        norm, spec = bundle["normalization"], bundle["spec"]
        if tensor_sha != plan["R4"]["model_state_sha"] or tensor_sha != bundle["model_state_sha"]:
            raise ValueError("Actual R4 adapter tensors differ")
        if tuple(spec.get(key) for key in ("family", "latent_k", "seed", "dataset")) != ("B2", K, 92601, "jena"):
            raise ValueError("Actual R4 adapter condition differs")
        mean, std, _ = data.fit_statistics(task, 12579)
        if not np.array_equal(mean, norm["mean"]) or not np.array_equal(std, norm["std"]):
            raise ValueError("Actual A60 scaler differs from exact R4")
        scaler_sha = fit.scaler_fingerprint(mean, std)
        if scaler_sha != plan["R4"]["refit_scaler_sha"] or norm["train_end"] != 12579:
            raise ValueError("R4 scaler fingerprint/fit boundary differs")
        adapter = networks.ForecastAdapter("B2", L, H, C, K, norm["pca_center"], norm["basis"], "pointwise", None)
        adapter.load_state_dict(bundle["model"], strict=True)
        adapter.train()
        model_path = inside(assets, plan["model_asset_relative_path"])
        pool = deployment.FoundationPool(model_path, "cpu")
        foundation = pool.get(H)
        if foundation.row_microbatch != 128 or foundation.checkpoint_inputs is not False or foundation.instance_norm_source_sha256 != plan["instance_norm_source_sha256"]:
            raise ValueError("Actual CPU foundation runtime differs")
        frozen_before = gpu_runtime.state_hash(foundation.model)
        adapter_before = networks.module_state_hash(adapter)
        if frozen_before != plan["foundation_tensor_sha256"] or any(p.requires_grad or p.grad is not None for p in foundation.model.parameters()):
            raise AssertionError("Actual pinned foundation is not frozen and gradient-free")
        origins = np.linspace(L, 12579 - H, 8, dtype=np.int64)
        histories = [data.history_window(task, int(origin), mean, std) for origin in origins]
        targets = [data.target_window(task, int(origin), mean, std) for origin in origins]
        x = torch.as_tensor(np.stack([v[0] for v in histories]), dtype=torch.float32)
        y = torch.as_tensor(np.stack([v[0] for v in targets]), dtype=torch.float32)
        mask = torch.as_tensor(np.stack([v[1] for v in targets]), dtype=torch.bool)
        deadline_check(plan["deadline_unix"])
        probe = InputGradientProbe(foundation.model, lambda g: torch.isfinite(g).all().item())
        before = pool.counters()
        try:
            z = adapter.encode(x, K)
            future = foundation(z)
            prediction = adapter.decode(x, z, future, K)
            if tuple(z.shape) != (8, L, K) or tuple(future.shape) != (8, H, K) or tuple(prediction.shape) != (8, H, C):
                raise AssertionError("Actual P/E/F/D geometry differs")
            loss = training.macro_loss(prediction, y, mask, mask.sum((0, 1)), 8, 8)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite allowed-A60 diagnostic loss")
            loss.backward()
            input_gradient = probe.evidence()
        finally:
            probe.close()
        delta = counter_delta(before, pool.counters())
        if delta != dict(streams=48, physical_streams=48, physical_calls=1):
            raise AssertionError("Expected exactly 8x6 logical/physical streams in one row128 F call")
        groups = {}
        for group, choose in (("P", lambda n: n.startswith("front.")), ("E", lambda n: n == "E"), ("D", lambda n: n == "D")):
            parameters = [p for name, p in adapter.named_parameters() if choose(name)]
            gradients = [p.grad for p in parameters if p.grad is not None]
            groups[group] = dict(parameter_tensors=len(parameters), gradient_tensors_present=len(gradients),
                all_present_gradients_finite=all(bool(torch.isfinite(g).all()) for g in gradients),
                any_nonzero_gradient=any(bool(torch.any(g != 0)) for g in gradients))
        validate_gradient_groups(groups)
        if any(p.requires_grad or p.grad is not None for p in foundation.model.parameters()):
            raise AssertionError("Frozen F acquired parameter trainability/gradient")
        if gpu_runtime.state_hash(foundation.model) != frozen_before or networks.module_state_hash(adapter) != adapter_before:
            raise AssertionError("Weights changed during optimizer-free backward")
        adapter.zero_grad(set_to_none=True)
        adapter.eval()
        deadline_check(plan["deadline_unix"])
        origin = 14675
        altered = dict(task, physical=task["physical"].copy(), observed=task["observed"].copy())
        altered["physical"][origin:] = 12345. + np.arange(C, dtype=np.float32)[None, :]
        altered["observed"][origin:] = ~altered["observed"][origin:]
        first, first_mask = data.history_window(task, origin, mean, std)
        second, second_mask = data.history_window(altered, origin, mean, std)
        if not np.array_equal(first, second) or not np.array_equal(first_mask, second_mask):
            raise AssertionError("Future perturbation changed allowed history")
        with torch.no_grad():
            a = adapter(torch.as_tensor(first[None]), foundation, K)
            b = adapter(torch.as_tensor(second[None]), foundation, K)
        if not torch.equal(a, b):
            raise AssertionError("Future perturbation changed the actual final R4 output")
        deadline_check(plan["deadline_unix"])
        raw_spec = dict(run_id="P0_TRANSPARENT_RAW_PLUMBING_ONLY", family="RAW", canonical_fit_end=12579,
                        foundation_sha=plan["foundation_tensor_sha256"], F_row_microbatch=128)
        route = deployment.Route(raw_spec, task, cache, model_path, "cpu", pool)
        physical = task["physical"][origin - L:origin][None].copy()
        observed = task["observed"][origin - L:origin][None].copy()
        with torch.no_grad():
            raw = route.predict_physical(physical, observed)
            transparent = transparent_raw_wrapper(physical, observed, mean, std, foundation, data, np, torch)
        if raw.dtype != np.float64 or transparent.dtype != np.float64 or not np.allclose(raw, transparent, atol=1e-5, rtol=1e-4):
            raise AssertionError("Transparent Raw physical pipeline differs at original tolerance")
        if pool.instances_created != 1 or foundation.row_microbatch != 128:
            raise AssertionError("P0 must share exactly one CPU row128 foundation")
        foundation.assert_frozen_unchanged()
        final_frozen = gpu_runtime.state_hash(foundation.model)
        final_adapter = networks.module_state_hash(adapter)
        if final_frozen != frozen_before or final_adapter != adapter_before:
            raise AssertionError("F/adapter weights changed during P0")
        deadline_check(plan["deadline_unix"])
        for path, expected in pinned.items():
            if sha(path) != expected:
                raise ValueError("Pinned input changed while P0 ran")
        result = dict(status="PASS_PRIORITY_P0_IMPLEMENTATION_CHECKS", plan_sha256=args.plan_sha, run_id=R4_ID,
            device="cpu", CPU_threads=torch.get_num_threads(), started_unix=started, completed_unix=time.time(),
            prefix_access_receipt=access, closed_prior_identities=closure,
            reused_validation_metadata=dict(SOURCE_BINDINGS_sha256=BINDINGS_SHA,
                full_saved_V_rescore=reused["accuracy"]["independent_FP64"], subset_CPU_replay=reused["CPU_checkpoint_replay"],
                rerun_in_this_command=False),
            geometry=dict(input=[8,L,C], latent_past=[8,L,K], latent_future=[8,H,K], output=[8,H,C]),
            diagnostic_backward=dict(origins=origins.tolist(), target_scope="A60-only", loss=float(loss.detach()),
                F_counters=delta, F_row_microbatch=128, checkpoint_inputs=False,
                input_gradient=input_gradient, PED_gradients=groups, actual_backward=True, optimizer_steps=0,
                new_training_conditions=0, every_gradient_nonzero_required=False),
            frozen=dict(F_before_sha256=frozen_before, F_after_sha256=final_frozen,
                adapter_before_sha256=adapter_before, adapter_after_sha256=final_adapter,
                F_all_requires_grad_false=True, F_all_parameter_grad_none=True, artifact_sha256=plan["R4"]["artifact_sha256"],
                saved_adapter_tensor_sha256=tensor_sha, scaler_sha256=scaler_sha),
            future_boundary=dict(origin=origin, changed_rows_halfopen=[origin,16772], future_physical_and_mask_changed=True,
                history_and_mask_exactly_equal=True, final_R4_output_exactly_equal=True, max_abs=float((a-b).abs().max()),
                causal_within_supplied_window_claimed=False),
            transparent_Raw=dict(batch=1, shape=list(raw.shape), dtype=str(raw.dtype), max_abs=float(np.abs(raw-transparent).max()),
                atol=1e-5, rtol=1e-4, same_foundation_instance=True, foundation_instances_created=pool.instances_created,
                row_microbatch=128, route_receipt=route.last_receipt, learned_D_or_identity_anchor_used=False,
                known_Raw_cost_row_variant_failure_resolved=False),
            actual_model_forward=True, protected_T_numeric_values_parsed=0, new_T_target_operations=0,
            input_SHA_prepost_equal=True, cuda_initialized=torch.cuda.is_initialized(),
            external_deadline_RSS_guard="ROOT supervisor required; helper checkpoints do not bound one blocking operation",
            limitations=["P0 validates computation, not accuracy improvement or independent generalization.",
                "Future mutation tests the already-materialized pre-T prefix and fixed A60 scaler; it is not a historical raw-ingestion rerun.",
                "The inherited legacy source normalization ancestry remains; whole compressed-file opaque hashing/ZIP I/O can include protected bytes without numeric conversion.",
                "One optimizer-free backward does not establish convergence or a successful fitting experiment.",
                "Transparent Raw plumbing is distinct from the previously failed Raw cost row-variant parity."])
        if result["cuda_initialized"]:
            raise AssertionError("CPU P0 unexpectedly initialized CUDA")
    except Exception as error:
        result = dict(status="FAILED_PRIORITY_P0", error_class=type(error).__name__, plan_sha256=args.plan_sha,
                      started_unix=started, completed_unix=time.time(), completed=False)
        with (output / "RESULT.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        raise
    finally:
        if route is not None:
            route.close()
        if pool is not None:
            pool.close()
    with (output / "RESULT.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("plan", "plan-sha", "public-root", "cache-root", "asset-root", "output"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="NOT_EXECUTED", input_reads=0, writes=0,
            torch_imported="torch" in sys.modules, numpy_imported="numpy" in sys.modules,
            model_or_array_loads=0, actual_model_forward=False, actual_backward=False, optimizer_steps=0)))
        return 0
    if any(getattr(args, name) is None for name in ("plan", "plan_sha", "public_root", "cache_root", "asset_root", "output")):
        parser.error("--execute requires explicit plan, plan-sha, public-root, cache-root, asset-root and fresh output")
    result = execute(args)
    print(json.dumps(dict(status=result["status"], output=str(Path(args.output) / "RESULT.json"),
                         optimizer_steps=0, new_training_conditions=0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
