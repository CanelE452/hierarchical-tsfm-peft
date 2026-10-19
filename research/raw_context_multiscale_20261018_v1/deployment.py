"""Physical inference for the registered raw/context models and frozen parents."""
from pathlib import Path

import numpy as np
import torch

from research.b2_revin_20261018_v1.deployment import _deployment as legacy
from research.informs_broad_validation_20261015_v1.networks import ConditionalHead
from . import common, models


def _private_path(root, relative, expected):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not expected or common.sha(path) != expected:
        raise ValueError("Private deployment artifact path or SHA256 changed")
    return path


class Route(legacy.Route):
    """Keep the inherited physical path and dispatch new models through forward."""
    def __init__(self, *args, ctx=None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            mode = getattr(self.model, "raw_context_mode", None)
            if mode is not None and self.bundle is not None:
                recorded = self.bundle["normalization"]["new_scientific_source_hashes"]
                if ctx is not None and recorded != models.scientific_sources(ctx):
                    raise ValueError("Raw/context training source binding changed")
                identity = self.bundle["spec"]["scientific_identity"]
                if identity["raw_context_mode"] != mode:
                    raise ValueError("Serialized raw/context mode differs from loaded model")
            schedule = self.spec.get("schedule", self.spec.get("raw_context_schedule"))
            if schedule is not None:
                if mode != "A" or schedule not in ("native", "two_call"):
                    raise ValueError("Cost-only schedule override requires A native/two_call")
                self.model.raw_context_schedule = schedule
        except BaseException:
            self.close()
            raise

    def _foundation(self, z):
        foundation = self.pool.get(self.task["H"])
        before = self.pool.counters()
        output = super()._foundation(z)
        after = self.pool.counters()
        mode = getattr(self.model, "raw_context_mode", None)
        scale = getattr(self.model, "raw_context_scale", 1)
        used_horizon = self.task["H"] // scale if mode == "B" and z.shape[1] < self.task["L"] else self.task["H"]
        config = getattr(getattr(foundation, "model", None), "chronos_config", None)
        context_limit = getattr(config, "context_length", None)
        trace = getattr(self.pool, "raw_context_trace", None)
        if trace is None:
            self.pool.raw_context_trace = trace = []
        trace.append(dict(route_id=self.run_id, input_shape=list(z.shape),
            requested_context=int(z.shape[1]),
            actual_context=min(int(z.shape[1]), int(context_limit)) if context_limit else int(z.shape[1]),
            wrapper_horizon=int(output.shape[1]), used_horizon=int(used_horizon),
            actual_head_horizon=getattr(config, "prediction_length", None),
            F_logical_streams=after["streams"]-before["streams"],
            F_physical_streams=after["physical_streams"]-before["physical_streams"],
            F_calls=after["physical_calls"]-before["physical_calls"],
            row_microbatch=self.spec.get("row_microbatch", self.spec.get("F_row_microbatch", 128))))
        return output

    def _predict_normalized(self, x, physical, observed):
        if isinstance(self.model, models._RawContextMixin):
            return self._step("P_E_F_repeat_D", lambda: self.model(x, self._foundation, self.latent_k))
        return super()._predict_normalized(x, physical, observed)

    def predict_physical(self, windows_np, observed_np, measure_stages=False):
        self.pool.raw_context_trace = []
        result = super().predict_physical(windows_np, observed_np, measure_stages)
        self.last_receipt["F_invocations"] = list(self.pool.raw_context_trace)
        return result


class CombinedRoute(Route):
    def __init__(self, spec, task, cache_root, *, ctx, device, foundation_pool):
        operation = spec.get("operation", spec.get("variant", "alpha"))
        if operation not in ("alpha", "conditional", "past_control"):
            raise ValueError("Unregistered frozen-parent combination")
        alpha = float(spec.get("alpha", .5))
        if operation == "alpha" and (not np.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError("Global alpha must be finite and in [0,1]")
        super().__init__(dict(spec, family="CURRENT"), task, cache_root,
                         device=device, foundation_pool=foundation_pool, ctx=ctx)
        self.family, self.spec, self.operation = "COMBINED", dict(spec), operation
        if operation == "alpha":
            self.spec["alpha"] = alpha
        try:
            if operation != "alpha" or alpha != 1:
                self.g_route = make_route(ctx, task, spec["g_route"], device=device, foundation_pool=self.pool)
            if operation == "conditional" or (operation == "alpha" and alpha != 0):
                self.b_route = make_route(ctx, task, spec["b_route"], device=device, foundation_pool=self.pool)
            if operation != "alpha":
                path = _private_path(cache_root, spec["head_state_path"], spec["head_state_sha256"])
                saved = torch.load(path, map_location="cpu", weights_only=True)
                config = dict(length=task["L"], horizon=task["H"], hidden=32)
                if "state" in saved:
                    if saved.get("head_config") != config:
                        raise ValueError("Conditional head L/H/hidden32 contract changed")
                    saved = saved["state"]
                if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in saved.values()):
                    raise ValueError("Conditional head requires a CPU state dictionary")
                with torch.random.fork_rng(devices=[]):
                    self.model = ConditionalHead(**config)
                self.model.load_state_dict(saved, strict=True)
                self.model.to(device).eval()
                self.artifact_sha256 = common.sha(path)
            self.latent_k = self.b_route.latent_k if self.b_route is not None else 0
        except BaseException:
            self.close()
            raise

    def _predict_normalized(self, x, physical, observed):
        g = self._parent(self.g_route, x, physical, observed) if self.g_route is not None else None
        b = self._parent(self.b_route, x, physical, observed) if self.b_route is not None else None
        if self.operation == "alpha":
            if g is None:
                return b
            if b is None:
                return g
            return self._step("alpha", lambda: g + float(self.spec["alpha"]) * (b-g))
        return self._step("H", lambda: self.model(x, g, g if b is None else b,
                                                control=self.operation == "past_control"))


def make_route(ctx, task, spec, *, device=None, foundation_pool=None):
    """Load one exact bundle; all nested routes share the same lazy full-H F."""
    device = str(device or ctx.get("device", "cuda"))
    family = spec["family"]
    if family == "RAW" and "foundation_sha" not in spec and "foundation_asset" not in spec:
        spec = dict(spec, foundation_sha=ctx["modules"]["legacy"].F_SHA)
    if family == "RAW_RIDGE" and device != "cpu":
        raise ValueError("Ridge costs require their separate CPU scope")
    if family in ("B2", "DIRECT_G", "LORA", "RAW_RIDGE") and (
            not spec.get("artifact_cache_relative_path") or not spec.get("artifact_sha256")):
        raise ValueError("A trained route requires its exact private artifact path and SHA256")
    own_pool = foundation_pool is None
    if own_pool:
        model_path = Path(ctx["asset"]) / ctx["modules"]["legacy"].MODEL_REL
        loader = ctx.get("foundation_loader")
        factory = (lambda p, h, m, d: loader(ctx, ctx["modules"], h, d)) if loader else None
        foundation_pool = legacy.FoundationPool(model_path, device, factory=factory)
    route = None
    try:
        if family == "COMBINED":
            route = CombinedRoute(spec, task, ctx["cache"], ctx=ctx,
                                  device=device, foundation_pool=foundation_pool)
        elif family == "B2":
            with models.model_factory(ctx):
                route = Route(spec, task, ctx["cache"], ctx=ctx,
                              device=device, foundation_pool=foundation_pool)
        elif family in ("RAW", "DIRECT_G", "LORA", "RAW_RIDGE", "CURRENT", "SEASONAL"):
            route = Route(spec, task, ctx["cache"], ctx=ctx,
                          device=device, foundation_pool=foundation_pool)
        else:
            raise ValueError("Unregistered deployment family: " + str(family))
        route.owns_pool = own_pool
        return route
    except BaseException:
        if route is not None:
            route.close()
        if own_pool:
            foundation_pool.close()
        raise


def resident_state(route):
    """Count distinct resident tensors, including shared F exactly once."""
    modules, arrays = [], []
    def visit(current):
        if current.model is not None:
            modules.append(current.model)
        if current.ridge_artifact is not None:
            arrays.extend(current.ridge_artifact[key] for key in ("coefficients", "mean", "std"))
        for parent in (current.g_route, current.b_route):
            if parent is not None:
                visit(parent)
    visit(route)
    foundation = route.pool.foundation
    if foundation is not None:
        modules.append(getattr(foundation, "model", foundation))
    seen, parameters, trainable, buffers, cpu_bytes, gpu_bytes = set(), 0, 0, 0, 0, 0
    for module in modules:
        for is_parameter, values in ((True, module.parameters()), (False, module.buffers())):
            for value in values:
                key = (str(value.device), value.data_ptr(), value.numel())
                if key in seen:
                    continue
                seen.add(key)
                parameters += value.numel() if is_parameter else 0
                buffers += value.numel() if not is_parameter else 0
                trainable += value.numel() if is_parameter and value.requires_grad else 0
                if value.device.type == "cuda":
                    gpu_bytes += value.numel() * value.element_size()
                else:
                    cpu_bytes += value.numel() * value.element_size()
    cpu_bytes += sum(value.nbytes for value in arrays)
    return dict(resident_parameters=parameters, trainable_parameters=trainable,
                resident_buffer_elements=buffers, resident_weights_cpu_MiB=cpu_bytes/1024**2,
                resident_weights_gpu_MiB=gpu_bytes/1024**2,
                foundation_instances_created=route.pool.instances_created)
