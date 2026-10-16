"""Real physical-input deployment; no saved forecast arrays are consumed."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import time

import numpy as np
import torch

from combination import load_conditional_checkpoint, scaler_fingerprint
from data_contract import array_sha256, causal_forward_fill, fit_statistics, sha256_file
import fit_base
from networks import module_state_hash
from training import fingerprint


ALIASES = {"RAW_BOLT_SMALL": "RAW", "DIRECT_TSMIXER": "DIRECT_B2", "DIRECT_TIMEMIXER": "DIRECT_B4",
           "LORA_R8": "LORA", "ADAPTS_BOLT_PORT": "ADAPTS", "G_ONLY": "DIRECT_G", "SEASONAL_NAIVE": "SEASONAL"}


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class FoundationPool:
    """One lazy F per H/device; direct-only fresh processes instantiate none."""
    def __init__(self, model_path, device="cuda", factory=None):
        self.model_path, self.device, self.factory = model_path, str(device), factory
        self.foundation = None
        self.instances_created = 0
        self.active_lora_owner = None
        self.load_seconds = 0.

    def get(self, horizon):
        if self.foundation is None:
            started = time.perf_counter()
            if self.factory is None:
                from gpu_runtime import FrozenBolt
                self.foundation = FrozenBolt(self.model_path, horizon, 128, False, self.device)
            else:
                self.foundation = self.factory(self.model_path, horizon, 128, self.device)
            self.instances_created += 1
            self.load_seconds += time.perf_counter() - started
        if self.foundation.horizon != horizon:
            raise ValueError("A shared foundation pool must use one H")
        return self.foundation

    def counters(self):
        return {name: int(getattr(self.foundation, name, 0)) for name in ("streams", "row_calls", "physical_streams", "physical_calls")}

    def close(self):
        if self.active_lora_owner is not None:
            raise RuntimeError("Close/deactivate the active LoRA route first")
        if self.foundation is not None and callable(getattr(self.foundation, "assert_frozen_unchanged", None)):
            self.foundation.assert_frozen_unchanged()
        self.foundation = None


class Route:
    def __init__(self, route_spec, task, cache_root, model_path=None, device="cuda", foundation_pool=None):
        started = time.perf_counter()
        self.spec, self.task, self.cache_root = dict(route_spec), task, Path(cache_root)
        self.device, self.run_id = str(device), str(route_spec.get("run_id", route_spec.get("family", "route")))
        self.family = ALIASES.get(str(route_spec["family"]), str(route_spec["family"]))
        self.pool = foundation_pool if foundation_pool is not None else FoundationPool(model_path, self.device)
        self.owns_pool = foundation_pool is None
        self.model = None
        self.g_route = self.b_route = None
        self.bundle = None
        self._closed = False
        self._merged_weights = None
        self._stage_mode = False
        self._stages = {}
        self.last_receipt = {}
        self.artifact_sha256 = None
        self.ridge_artifact = None
        if self.family == "RAW_RIDGE" and self.device != "cpu":
            raise ValueError("Native NumPy FP64 ridge must be measured with device=cpu in its separate hardware scope")
        self.latent_k = int(route_spec.get("latent_k", len(task["columns"])))
        self.period = int(route_spec.get("period", task.get("period", 24)))
        canonical_end = int(route_spec.get("canonical_fit_end", task["split_ranges"]["A_stop"][1]))
        self.mean, self.std, _ = fit_statistics(task, canonical_end)
        if self.family not in {"RAW", "CURRENT", "SEASONAL", "ALPHA"}:
            relative = route_spec.get("artifact_cache_relative_path")
            if not relative:
                raise ValueError("A trained deployment route requires its exact artifact path")
            path = (self.cache_root / relative).resolve()
            if not path.is_relative_to(self.cache_root.resolve()):
                raise ValueError("Artifact must stay in the private campaign cache")
            self.artifact_sha256 = sha256_file(path)
            if route_spec.get("artifact_sha256") and route_spec["artifact_sha256"] != self.artifact_sha256:
                raise ValueError("Deployment artifact fingerprint changed")
            if self.family == "NESTED_PCA":
                with np.load(path, allow_pickle=False) as archive:
                    if str(archive["data_sha"].item()) != task["data_sha256"] or int(archive["fit_end"].item()) != canonical_end:
                        raise ValueError("Nested PCA data/fit boundary changed")
                    if not np.array_equal(archive["mean"], self.mean) or not np.array_equal(archive["std"], self.std):
                        raise ValueError("Nested PCA A60 normalization changed")
                    basis, center = archive["basis"], archive["center"]
                if basis.shape != (len(task["columns"]), len(task["columns"])) or not 1 <= self.latent_k <= basis.shape[1]:
                    raise ValueError("Nested PCA prefix axes differ")
                foundation = self.pool.get(task["H"])
                if not route_spec.get("foundation_sha") or fit_base._model_sha(foundation) != route_spec["foundation_sha"]:
                    raise ValueError("Nested PCA requires its exact original foundation")
                self.model = fit_base._FixedPCA(center, basis).to(self.device)
                self.family = "PCA_FIXED_BOLT"
            elif self.family == "RAW_RIDGE":
                from ridge_baseline import load_artifact
                self.ridge_artifact = load_artifact(path)
                if str(self.ridge_artifact["data_sha"].item()) != task["data_sha256"]:
                    raise ValueError("Ridge deployment data fingerprint differs")
                self.mean, self.std = self.ridge_artifact["mean"], self.ridge_artifact["std"]
                if self.ridge_artifact["coefficients"].shape != (len(task["columns"]), 33, task["H"]):
                    raise ValueError("Ridge coefficient axes differ from task")
            elif self.family == "RAW_CALIBRATED":
                with np.load(path, allow_pickle=False) as archive:
                    bias = archive["bias"]
                    contract = json.loads(str(archive["contract_json"].item()))
                    for key, expected in (("mean", self.mean), ("std", self.std)):
                        if not np.array_equal(archive[key], expected):
                            raise ValueError("Raw calibration normalization changed")
                if contract["data_sha"] != task["data_sha256"] or contract["scaler_sha"] != scaler_fingerprint(self.mean, self.std):
                    raise ValueError("Raw calibration data/scaler contract changed")
                if bias.shape != (len(task["columns"]),) or not np.isfinite(bias).all():
                    raise ValueError("Raw calibration bias axes or values invalid")
                self.model = torch.nn.Module()
                self.model.register_buffer("bias", torch.as_tensor(bias, device=self.device, dtype=torch.float32))
            elif self.family == "COMBINATION":
                self.model = load_conditional_checkpoint(path, self.device)
                self.bundle = torch.load(path, map_location="cpu", weights_only=True)
                if self.bundle["contract"]["data_sha"] != task["data_sha256"]:
                    raise ValueError("Head deployment task differs from its calibration data")
                if self.bundle["contract"]["scaler_sha"] != scaler_fingerprint(self.mean, self.std):
                    raise ValueError("Head deployment canonical scaler differs from its fit")
                variant = str(route_spec.get("variant", self.model.combination_variant))
                if variant != self.model.combination_variant:
                    raise ValueError("Head variant differs from the saved checkpoint")
                self.variant = variant
            else:
                # Only locally generated, hash-bound private bundles use pickle
                # because they contain NumPy normalization arrays.
                self.bundle = torch.load(path, map_location="cpu", weights_only=False)
                if self.bundle["contract"]["data_sha"] != task["data_sha256"]:
                    raise ValueError("Deployment artifact task fingerprint differs")
                actual_state_sha = fingerprint({key: array_sha256(value.numpy()) for key, value in self.bundle["model"].items()})
                if actual_state_sha != self.bundle["model_state_sha"]:
                    raise ValueError("Deployment saved parameter tensor hash changed")
                base_spec = dict(self.bundle["spec"], device=self.device)
                self.family = base_spec["family"]
                self.latent_k = int(route_spec.get("latent_k", base_spec.get("latent_k", len(task["columns"]))))
                if route_spec.get("prefix_variant") == "posthoc":
                    maximum = int(base_spec["latent_k"])
                    if base_spec.get("k_grid") != [maximum] or not 1 <= self.latent_k <= maximum:
                        raise ValueError("Posthoc deployment requires a maximum-only trained artifact")
                    base_spec["k_grid"] = sorted({self.latent_k, maximum})
                foundation = None if self.family in fit_base.DIRECT else self.pool.get(task["H"])
                if fit_base._model_sha(foundation) != self.bundle["contract"]["model_sha"]:
                    raise ValueError("Deployment requires the exact original frozen foundation")
                self.model, regenerated = fit_base.create_model(base_spec, task, self.bundle["normalization"]["fit_end"], foundation)
                self.model.load_state_dict(self.bundle["model"], strict=True)
                for key in ("mean", "std"):
                    if not np.array_equal(regenerated[key], self.bundle["normalization"][key]):
                        raise ValueError("Artifact normalization no longer matches the task")
                self.mean, self.std = self.bundle["normalization"]["mean"], self.bundle["normalization"]["std"]
                if self.family == "LORA":
                    # Inactive adapters must not leave hooks on the shared F.
                    self.model._remove_hooks()
                self.prediction_model_sha = fingerprint(dict(artifact_sha=self.artifact_sha256, model_state_sha=self.bundle["model_state_sha"],
                                                            normalization=self.bundle["contract"]["scaler_sha"]))
                if route_spec.get("expected_prediction_model_sha") and self.prediction_model_sha != route_spec["expected_prediction_model_sha"]:
                    raise ValueError("Deployment parent differs from its calibration forecast model")
        if self.family in {"COMBINATION", "ALPHA"}:
            self.variant = str(route_spec.get("variant", getattr(self, "variant", "alpha")))
            if "g_route" in route_spec and self.variant != "legacy_residual":
                self.g_route = Route(route_spec["g_route"], task, cache_root, model_path, self.device, self.pool)
            if "b_route" in route_spec and self.variant != "past_control":
                self.b_route = Route(route_spec["b_route"], task, cache_root, model_path, self.device, self.pool)
            if self.variant in {"conditional", "past_control"} and self.g_route is None:
                raise ValueError("Conditional/past H deployment requires G")
            if self.variant in {"conditional", "legacy_residual"} and self.b_route is None:
                raise ValueError("Conditional/residual deployment requires b")
            if self.family == "ALPHA" and (self.g_route is None or self.b_route is None):
                raise ValueError("Registered fitted alpha deployment computes both parents")
            self.latent_k = self.b_route.latent_k if self.b_route is not None else 0
        if self.family == "RAW_CALIBRATED":
            if route_spec.get("b_route", {}).get("family") not in {"RAW", "RAW_BOLT_SMALL"}:
                raise ValueError("Raw calibration deployment requires its frozen Raw parent")
            self.b_route = Route(route_spec["b_route"], task, cache_root, model_path, self.device, self.pool)
            self.latent_k = len(task["columns"])
        if self.family in fit_base.DIRECT or self.family in {"CURRENT", "SEASONAL", "RAW_RIDGE"}:
            self.latent_k = 0
        if self.family == "RAW":
            asset = route_spec.get("foundation_asset")
            expected = route_spec.get("foundation_sha")
            if not asset and not expected and self.pool.factory is None:
                raise ValueError("Raw route requires a pinned foundation identity or official asset")
            if asset:
                if self.pool.model_path is None:
                    raise ValueError("Raw official asset requires its pinned local model snapshot")
                model_root = Path(self.pool.model_path).resolve()
                if not asset.get("files"):
                    raise ValueError("Raw official foundation asset has no file bindings")
                for name, binding in asset["files"].items():
                    path = (model_root/name).resolve()
                    if not path.is_relative_to(model_root) or not path.is_file() or sha256_file(path) != binding["sha256"]:
                        raise ValueError("Raw official foundation asset identity changed")
            actual = self.pool.get(task["H"])
            if expected and fit_base._model_sha(actual) != expected:
                raise ValueError("Raw route foundation identity differs from its accuracy artifact")
        if self.model is not None:
            self.model.eval()
        self.mean_t = torch.as_tensor(self.mean, device=self.device)
        self.std_t = torch.as_tensor(self.std, device=self.device)
        self.load_seconds = time.perf_counter() - started

    def _step(self, name, call):
        if not self._stage_mode:
            return call()
        _sync(self.device)
        started = time.perf_counter()
        value = call()
        _sync(self.device)
        self._stages[name] = self._stages.get(name, 0.) + time.perf_counter() - started
        return value

    def activate(self):
        for parent in (self.g_route, self.b_route):
            if parent is not None:
                parent.activate()
        if self.family == "LORA":
            if self.pool.active_lora_owner not in (None, self.run_id):
                raise RuntimeError("Another LoRA route is active on the shared foundation")
            if self.pool.active_lora_owner is None:
                if self._merged_weights is None:
                    self.model.merge()
                    self._merged_weights = [module.weight.detach().cpu().clone() for _, module in self.model._targets]
                    # Exact restore originals and merged snapshots stay on CPU;
                    # steady-state merged inference retains no GPU adapter copy.
                    self.model._originals = [value.cpu() for value in self.model._originals]
                    self.model.adapters.cpu()
                else:
                    with torch.no_grad():
                        for (_, module), weight in zip(self.model._targets, self._merged_weights):
                            module.weight.copy_(weight)
                    self.model._remove_hooks(); self.model._merged = True
                self.pool.active_lora_owner = self.run_id
        elif self.pool.active_lora_owner is not None:
            raise RuntimeError("Unmerged shared foundation required for this route")

    def deactivate(self):
        if self.family == "LORA" and self.pool.active_lora_owner == self.run_id:
            self.model.unmerge()
            self.model._remove_hooks()
            self.pool.active_lora_owner = None
        for parent in (self.g_route, self.b_route):
            if parent is not None:
                parent.deactivate()

    def _preprocess(self, physical, observed):
        def fill():
            if observed.all():
                return physical.copy()
            return np.stack([causal_forward_fill(x, mask, self.mean) for x, mask in zip(physical, observed)])
        filled = self._step("CPU_preprocess", fill)
        normalized = self._step("CPU_preprocess", lambda: ((filled - self.mean) / self.std).astype(np.float32))
        return self._step("host_to_device", lambda: torch.as_tensor(normalized, dtype=torch.float32, device=self.device))

    def _parent(self, parent, x, physical, observed):
        parent._stage_mode, parent._stages = self._stage_mode, self._stages
        same = np.array_equal(parent.mean, self.mean) and np.array_equal(parent.std, self.std)
        px = x if same else parent._preprocess(physical, observed)
        prediction = parent._predict_normalized(px, physical, observed)
        return prediction if same else (prediction * parent.std_t + parent.mean_t - self.mean_t) / self.std_t

    def _foundation(self, z):
        foundation = self.pool.get(self.task["H"])
        previous = foundation.row_microbatch
        row_mb = self.spec.get("row_microbatch", self.spec.get("F_row_microbatch", "full" if self.family == "RAW" else 128))
        foundation.row_microbatch = None if row_mb in (None, "full", 0) else int(row_mb)
        try:
            return self._step("F", lambda: foundation(z))
        finally:
            foundation.row_microbatch = previous

    def _predict_normalized(self, x, physical, observed):
        family = self.family
        if family == "CURRENT":
            return x[:, -1:].expand(-1, self.task["H"], -1)
        if family == "SEASONAL":
            if not 1 <= self.period <= self.task["L"]:
                raise ValueError("Seasonal period outside observed history")
            index = torch.arange(self.task["H"], device=self.device) % self.period
            return x[:, -self.period:][:, index]
        if family in {"RAW", "LORA"}:
            return self._foundation(x)
        if family in fit_base.DIRECT:
            return self._step("G_or_direct", lambda: self.model(x))
        if family == "RAW_RIDGE":
            from ridge_baseline import predict_normalized
            prediction = self._step("ridge_CPU", lambda: predict_normalized(self.ridge_artifact, x.numpy()))
            return torch.as_tensor(prediction, dtype=torch.float64, device="cpu")
        if family == "RAW_CALIBRATED":
            b = self._parent(self.b_route, x, physical, observed)
            return self._step("H_or_alpha", lambda: b + self.model.bias)
        if family in {"COMBINATION", "ALPHA"}:
            g = self._parent(self.g_route, x, physical, observed) if self.g_route is not None else None
            # past_control does not invoke or load an unnecessary b route.
            b = self._parent(self.b_route, x, physical, observed) if self.b_route is not None and self.variant != "past_control" else None
            if family == "ALPHA":
                alpha = float(self.spec["alpha"])
                if not 0 <= alpha <= 1:
                    raise ValueError("Global alpha must be fixed in [0,1]")
                return self._step("H_or_alpha", lambda: g + alpha * (b - g))
            if g is None: g = b
            if b is None: b = g
            return self._step("H_or_alpha", lambda: self.model(x, g, b, control=self.variant == "past_control"))
        if family.startswith("PCA_FIXED"):
            basis = self.model.E[:, :self.latent_k]
            z = self._step("P_E", lambda: (x - self.model.center) @ basis)
            future = self._foundation(z)
            return self._step("D", lambda: future @ basis.T + self.model.center)
        if family == "ADAPTS":
            z = self._step("P_E", lambda: self.model.encoder(x))
            future = self._foundation(z)
            return self._step("D", lambda: self.model.decoder(future))
        z = self._step("P_E", lambda: self.model.encode(x, self.latent_k))
        future = self._foundation(z)
        return self._step("D", lambda: self.model.decode(x, z, future, self.latent_k))

    @torch.no_grad()
    def predict_physical(self, windows_np, observed_np, measure_stages=False):
        if self._closed:
            raise RuntimeError("Deployment route is closed")
        _sync(self.device)
        started = time.perf_counter()
        physical, observed = np.asarray(windows_np, np.float32), np.asarray(observed_np)
        if physical.ndim != 3 or physical.shape[1:] != (self.task["L"], len(self.task["columns"])) or observed.shape != physical.shape or observed.dtype != np.bool_:
            raise ValueError("Expected physical/bool mask [B,L,C]")
        if not np.isfinite(physical[observed]).all():
            raise ValueError("Observed physical input is nonfinite")
        self.activate()
        self._stage_mode, self._stages = bool(measure_stages), {}
        before = self.pool.counters()
        x = self._preprocess(physical, observed)
        prediction = self._predict_normalized(x, physical, observed)
        result = self._step("inverse_and_CPU", lambda: (prediction.double() * self.std_t.double() + self.mean_t.double()).cpu().numpy())
        _sync(self.device)
        after = self.pool.counters()
        if result.shape != (len(physical), self.task["H"], len(self.task["columns"])) or not np.isfinite(result).all():
            raise ValueError("Deployment physical output invalid")
        whole = time.perf_counter() - started
        self.last_receipt = {"whole_seconds": whole, "stage_seconds": dict(self._stages),
                             "stage_mode": bool(measure_stages), "stage_sum_claimed_equal_whole": False,
                             "F_logical_streams": after["streams"] - before["streams"],
                             "F_physical_streams": after["physical_streams"] - before["physical_streams"],
                             "F_calls": after["physical_calls"] - before["physical_calls"],
                             "foundation_instances_created_in_pool": self.pool.instances_created,
                             "active_K": self.latent_k if self.family not in {"RAW", "LORA"} else len(self.task["columns"])}
        return result

    def check_lora_parity(self, physical, observed, atol=1e-5, rtol=1e-4):
        if self.family != "LORA":
            return {"status": "NOT_LORA"}
        self.deactivate()
        self.model.adapters.to(self.device)
        self.model._attach_hooks()
        try:
            with torch.no_grad():
                x = self._preprocess(np.asarray(physical, np.float32), np.asarray(observed, bool))
                before = self.pool.get(self.task["H"])(x).cpu().numpy()
                self.model._remove_hooks()
                self.activate()
                after = self.pool.get(self.task["H"])(x).cpu().numpy()
            passed = bool(np.allclose(before, after, atol=atol, rtol=rtol))
            result = {"status": "PASS" if passed else "FAIL", "max_abs": float(np.abs(before - after).max()), "atol": atol, "rtol": rtol}
            if not passed:
                raise AssertionError("Actual LoRA merged/unmerged prediction parity failed")
            return result
        finally:
            self.model._remove_hooks()

    def memory_state(self):
        modules = [self.model]
        for parent in (self.g_route, self.b_route):
            if parent is not None:
                modules.append(parent.model)
        foundation = self.pool.foundation
        if foundation is not None:
            modules.append(getattr(foundation, "model", foundation))
        seen, gpu_bytes, cpu_bytes, non_backbone_gpu_bytes = set(), 0, 0, 0
        for module_index, module in enumerate(modules):
            if module is None: continue
            for value in list(module.parameters()) + list(module.buffers()):
                key = (value.device.type, value.data_ptr(), value.numel())
                if key in seen: continue
                seen.add(key)
                count = value.numel() * value.element_size()
                if value.device.type == "cuda":
                    gpu_bytes += count
                    if module is not getattr(foundation, "model", foundation): non_backbone_gpu_bytes += count
                else: cpu_bytes += count
        if self.ridge_artifact is not None:
            cpu_bytes += sum(self.ridge_artifact[key].nbytes for key in ("coefficients", "mean", "std"))
        return {"resident_weights_gpu_MiB": gpu_bytes / 1024**2, "resident_weights_cpu_MiB": cpu_bytes / 1024**2,
                "active_adapter_MiB": non_backbone_gpu_bytes / 1024**2,
                "active_adapter_definition": "All active non-foundation model parameter/buffer bytes on GPU; merged LoRA adapter bytes are zero",
                "foundation_instances_created": self.pool.instances_created,
                "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 1024**2 if self.device.startswith("cuda") else 0.,
                "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 1024**2 if self.device.startswith("cuda") else 0.}

    def close(self):
        if self._closed: return
        self.deactivate()
        for parent in (self.g_route, self.b_route):
            if parent is not None: parent.close()
        if self.family == "LORA": self.model.close()
        if self.owns_pool: self.pool.close()
        self._closed = True
