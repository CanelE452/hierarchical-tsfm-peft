"""A pinned native TS2Vec with small-K configuration and causal export."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor/ts2vec"
COMMIT = "b0088e14a99706c05451316dc6db8d3da9351163"
NATIVE = None


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def vendor_provenance() -> dict:
    receipt = json.loads((VENDOR / "provenance.json").read_text(encoding="utf-8"))
    if receipt["commit"] != COMMIT:
        raise RuntimeError("TS2Vec vendor commit mismatch")
    for item in receipt["files"]:
        path = VENDOR / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"TS2Vec vendor hash mismatch: {item['path']}")
    return receipt


def native_module():
    global NATIVE
    if NATIVE is not None:
        return NATIVE
    vendor_provenance()
    alias_names = [name for name in sys.modules if name == "models" or name.startswith("models.") or name == "utils"]
    previous = {name: sys.modules.pop(name) for name in alias_names}
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(VENDOR))
        specification = importlib.util.spec_from_file_location("_level_pinned_native_ts2vec", VENDOR / "ts2vec.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        NATIVE = module
    finally:
        for name in list(sys.modules):
            if name == "models" or name.startswith("models.") or name == "utils":
                sys.modules.pop(name)
        sys.modules.update(previous)
        sys.path[:] = original_path
    return NATIVE


def seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


class NativeTs2VecAdapter(nn.Module):
    family = "TS2VEC_CAUSAL_SMALL"
    encoder_lag = 200

    def __init__(self, channels: int, latent: int, seed: int, device: str = "cpu", configuration: dict | None = None):
        super().__init__()
        defaults = json.loads((HERE / "method_protocol.json").read_text(encoding="utf-8"))["native_configuration"]
        self.configuration = dict(defaults if configuration is None else configuration)
        self.channels = int(channels)
        self.latent = int(latent)
        self.seed = int(seed)
        seed_runtime(self.seed)
        self.native = native_module().TS2Vec(
            input_dims=self.channels,
            output_dims=self.latent,
            hidden_dims=int(self.configuration["hidden_dims"]),
            depth=int(self.configuration["depth"]),
            device=device,
            lr=float(self.configuration["learning_rate"]),
            batch_size=int(self.configuration["batch_size"]),
            max_train_length=int(self.configuration["max_train_length"]),
            temporal_unit=int(self.configuration["temporal_unit"]),
        )
        self.training_net = self.native._net
        self.swa_net = self.native.net
        self.fit_receipt = None
        self.optimizer_state = None

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError("TS2Vec input must be [B,T,C] with configured C")
        if x.shape[0] >= 256:
            raise ValueError("TS2Vec scientific/benchmark origin batch must be below256")
        values = x.detach().cpu().numpy().astype(np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("TS2Vec receives the same finite past-imputed inputs as other encoders")
        encoded = self.native.encode(
            values,
            encoding_window=None,
            causal=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256,
        )
        if encoded.shape != (x.shape[0], x.shape[1], self.latent) or not np.isfinite(encoded).all():
            raise RuntimeError("Native causal TS2Vec returned invalid latent output")
        return torch.as_tensor(encoded, device=x.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def fit(self, windows: np.ndarray, monitor=None) -> dict:
        if windows.ndim != 3 or windows.shape[-1] != self.channels or windows.dtype != np.float32:
            raise ValueError("Native TS2Vec training windows must be FP32[N,T,C]")
        if not np.isfinite(windows).all():
            raise ValueError("Native TS2Vec training uses finite prior-imputed TRAIN windows")
        if len(windows) < 2:
            raise ValueError("Native instance contrast requires multiple TRAIN windows")
        losses = []
        epochs = []
        optimizers = []
        actual_steps = [0]
        original_adamw = torch.optim.AdamW

        def optimizer_factory(*arguments, **keywords):
            optimizer = original_adamw(*arguments, **keywords)
            optimizer.register_step_post_hook(lambda *unused: actual_steps.__setitem__(0, actual_steps[0] + 1))
            optimizers.append(optimizer)
            return optimizer

        def after_iter(model, loss):
            if not np.isfinite(loss):
                raise RuntimeError("Nonfinite native TS2Vec training loss")
            losses.append({"iteration": int(model.n_iters), "loss": float(loss)})
            if monitor is not None:
                monitor.check("ts2vec_fit", dataset=getattr(self, "dataset", None), seed=self.seed, native_iterations=int(model.n_iters))

        def after_epoch(model, loss):
            epochs.append({"epoch": int(model.n_epochs), "loss": float(loss)})

        self.native.after_iter_callback = after_iter
        self.native.after_epoch_callback = after_epoch
        started = time.perf_counter()
        cpu_started = time.process_time()
        with patch.object(torch.optim, "AdamW", optimizer_factory):
            returned_losses = self.native.fit(windows, n_epochs=None, n_iters=int(self.configuration["n_iters"]), verbose=False)
        if len(optimizers) != 1 or actual_steps[0] != int(self.configuration["n_iters"]) or self.native.n_iters != actual_steps[0]:
            raise RuntimeError("Native TS2Vec actual optimizer step count mismatch")
        optimizer = optimizers[0]
        self.optimizer_state = cpu_state(optimizer.state_dict())
        self.fit_receipt = {
            "native_iterations": int(self.native.n_iters),
            "actual_optimizer_steps": int(actual_steps[0]),
            "native_completed_epochs": int(self.native.n_epochs),
            "swa_n_averaged": int(self.swa_net.n_averaged.item()),
            "wall_seconds": time.perf_counter() - started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "optimizer": "AdamW",
            "optimizer_parameters": {key: cpu_state(value) for key, value in optimizer.param_groups[0].items() if key != "params"},
            "iteration_losses": losses,
            "epoch_losses": epochs,
            "native_returned_epoch_losses": [float(value) for value in returned_losses],
            "training_windows_shape": list(windows.shape),
            "training_windows_sha256": array_hash(windows),
            "training_windows_bytes": int(windows.nbytes),
            "selected_encoder": "final SWA",
        }
        self.freeze()
        return self.fit_receipt

    def save_checkpoint(self, path: Path, provenance: dict) -> None:
        torch.save({
            "schema": "native_small_ts2vec_checkpoint_v1",
            "channels": self.channels,
            "latent": self.latent,
            "seed": self.seed,
            "configuration": self.configuration,
            "training_net": cpu_state(self.training_net.state_dict()),
            "swa_net": cpu_state(self.swa_net.state_dict()),
            "optimizer": self.optimizer_state,
            "fit_receipt": self.fit_receipt,
            "native_iterations": int(self.native.n_iters),
            "native_completed_epochs": int(self.native.n_epochs),
            "rng": {"numpy": np.random.get_state(), "python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            "training_provenance": provenance,
            "upstream_commit": COMMIT,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: Path, device: str = "cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["schema"] != "native_small_ts2vec_checkpoint_v1" or payload["upstream_commit"] != COMMIT:
            raise RuntimeError("Native TS2Vec checkpoint contract mismatch")
        adapter = cls(payload["channels"], payload["latent"], payload["seed"], device, payload["configuration"])
        adapter.training_net.load_state_dict(payload["training_net"])
        adapter.swa_net.load_state_dict(payload["swa_net"])
        adapter.native.n_iters = payload["native_iterations"]
        adapter.native.n_epochs = payload["native_completed_epochs"]
        adapter.fit_receipt = payload["fit_receipt"]
        adapter.optimizer_state = payload["optimizer"]
        adapter.freeze()
        return adapter, payload


def training_window_pool(data: dict, canonical_origins: np.ndarray, decoder_origins: np.ndarray, byte_limit: int):
    context = 512
    values = np.asarray(data["X"], dtype=np.float32)
    needed_bytes = int(len(canonical_origins) * context * values.shape[1] * np.dtype(np.float32).itemsize)
    use_all = needed_bytes <= int(byte_limit)
    selected = canonical_origins if use_all else decoder_origins
    selected = np.asarray(selected, dtype=np.int64)
    windows = np.empty((len(selected), context, values.shape[1]), dtype=np.float32)
    for index, origin in enumerate(selected):
        windows[index] = values[int(origin) - context:int(origin)]
    receipt = {
        "pool": "all canonical TRAIN origins" if use_all else "common512 decoder TRAIN origins due to materialization cap",
        "canonical_origin_count": int(len(canonical_origins)),
        "canonical_origin_sha256": array_hash(np.asarray(canonical_origins, dtype=np.int64)),
        "canonical_window_bytes": needed_bytes,
        "origin_count": int(len(selected)),
        "origin_sha256": array_hash(selected),
        "window_bytes": int(windows.nbytes),
        "window_sha256": array_hash(windows),
        "materialized_window_budget_bytes": int(byte_limit),
        "data_order": "official native DataLoader shuffle/drop_last; not MLP origin stream",
        "input_contract": "past-imputed standardized finite TRAIN context512",
    }
    return windows, selected, receipt
