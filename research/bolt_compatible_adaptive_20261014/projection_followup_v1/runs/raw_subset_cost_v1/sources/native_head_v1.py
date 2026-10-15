"""Frozen projection with one affine temporal forecaster per latent channel."""

import hashlib
import time

import numpy as np
import torch
from torch import nn


class NativeHead(nn.Module):
    def __init__(self, K, L=512, H=48):
        super().__init__()
        self.K, self.L, self.H = int(K), int(L), int(H)
        self.weight = nn.Parameter(torch.zeros(self.K, self.H, self.L))
        self.bias = nn.Parameter(torch.zeros(self.K, self.H))

    def forward(self, latent_history):
        if latent_history.shape[1:] != (self.L, self.K):
            raise ValueError("NATIVE_LATENT_HISTORY_SHAPE")
        last = latent_history[:, -1:]
        delta = latent_history - last
        innovation = torch.einsum("nlk,khl->nhk", delta, self.weight)
        innovation = innovation + self.bias.transpose(0, 1).unsqueeze(0)
        return last + innovation


class FrozenProjection(nn.Module):
    def __init__(self, projection):
        super().__init__()
        for key in ("center", "encoder", "decoder", "decoder_bias"):
            value = getattr(projection, key)
            self.register_buffer(key, torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32))
        if self.encoder.shape != (len(self.center), self.decoder.shape[0]):
            raise ValueError("PROJECTION_MATRIX_SHAPE")
        if self.decoder.shape[1] != len(self.center) or self.decoder_bias.shape != self.center.shape:
            raise ValueError("PROJECTION_DECODER_SHAPE")
        if not all(torch.isfinite(value).all() for value in self.buffers()):
            raise ValueError("PROJECTION_NONFINITE")
        self.eval().requires_grad_(False)

    def encode(self, history):
        return (history - self.center) @ self.encoder

    def decode(self, latent):
        return latent @ self.decoder + self.decoder_bias

    def forecast(self, raw_last, latent_future, latent_last):
        if raw_last.ndim == 2:
            raw_last = raw_last[:, None, :]
        if latent_last.ndim == 2:
            latent_last = latent_last[:, None, :]
        return raw_last + (latent_future - latent_last) @ self.decoder


class NativeForecaster(nn.Module):
    def __init__(self, projection, L=512, H=48):
        super().__init__()
        self.projection = FrozenProjection(projection)
        self.head = NativeHead(self.projection.encoder.shape[1], L, H)

    def predict_latents(self, latent_history, raw_last):
        future = self.head(latent_history)
        return self.projection.forecast(raw_last, future, latent_history[:, -1:])

    def forward(self, history, raw_last=None):
        if raw_last is None:
            raw_last = history[:, -1:]
        return self.predict_latents(self.projection.encode(history), raw_last)


def masked_macro_mse(prediction, target, observed):
    counts = observed.sum(dim=(0, 1))
    active = counts > 0
    if not bool(active.any()):
        raise ValueError("NATIVE_BATCH_WITHOUT_OBSERVATIONS")
    difference = torch.where(observed, prediction, 0.0) - torch.where(observed, target, 0.0)
    error = difference.square().sum(dim=(0, 1))
    return (error / counts.clamp_min(1))[active].mean()


def score_predictions(prediction, target, observed):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError("NATIVE_SCORE_NONFINITE_OBSERVED")
    counts = observed.sum(axis=(0, 1))
    active = counts > 0
    if not active.any():
        raise ValueError("NATIVE_SCORE_WITHOUT_OBSERVATIONS")
    difference = np.where(observed, prediction - target, 0.0)
    mse = np.square(difference).sum(axis=(0, 1)) / np.maximum(counts, 1)
    mae = np.abs(difference).sum(axis=(0, 1)) / np.maximum(counts, 1)
    return {"mse": float(mse[active].mean()), "mae": float(mae[active].mean()),
            "observed_counts": counts.tolist(), "active_channels": int(active.sum()),
            "per_channel_mse": [float(value) if valid else None for value, valid in zip(mse, active)],
            "per_channel_mae": [float(value) if valid else None for value, valid in zip(mae, active)]}


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_hash(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def select_checkpoint(checkpoints):
    if not checkpoints:
        raise ValueError("NATIVE_NO_CHECKPOINT")
    return min(checkpoints, key=lambda item: (item["tune"]["mse"], item["update"]))


def predict_cached_latents(model, latent_history, raw_last, batch_size=32):
    device = model.head.weight.device
    latent_history = torch.as_tensor(latent_history, dtype=torch.float32, device=device)
    raw_last = torch.as_tensor(raw_last, dtype=torch.float32, device=device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(latent_history), batch_size):
            outputs.append(model.predict_latents(latent_history[start:start + batch_size],
                                                raw_last[start:start + batch_size]).cpu())
    if not outputs:
        return np.empty((0, model.head.H, len(model.projection.center)), dtype=np.float32)
    return torch.cat(outputs).numpy()


def cache_latents(model, payload, split, progress_hook):
    history = np.asarray(payload["history"], dtype=np.float32)
    if history.ndim != 3 or history.shape[1] != model.head.L:
        raise ValueError("NATIVE_RAW_HISTORY_SHAPE")
    if not np.isfinite(history).all():
        raise ValueError("NATIVE_HISTORY_NONFINITE")
    pieces = []
    with torch.no_grad():
        for start in range(0, len(history), 32):
            history_tensor = torch.as_tensor(history[start:start + 32], dtype=torch.float32)
            pieces.append(model.projection.encode(history_tensor))
            if progress_hook is not None:
                progress_hook({"stage": "cache", "split": split,
                               "completed": min(start + 32, len(history)), "total": len(history)})
    if not pieces:
        raise ValueError("NATIVE_EMPTY_PAYLOAD")
    return torch.cat(pieces)


def train_native(projection, fit_payload, tune_payload, seed, progress_hook=None,
                 checkpoint_callback=None, *, updates=600, batch_size=4):
    if not 0 <= updates <= 600 or batch_size <= 0:
        raise ValueError("NATIVE_TRAINING_LIMIT")
    started = time.perf_counter()
    L = np.asarray(fit_payload["history"]).shape[1]
    H = np.asarray(fit_payload["target"]).shape[1]
    model = NativeForecaster(projection, L, H).cpu()
    frozen_initial = state_hash(clone_state(model.projection))
    initial_model = state_hash(clone_state(model))
    latent = {"fit": cache_latents(model, fit_payload, "fit", progress_hook),
              "tune": cache_latents(model, tune_payload, "tune", progress_hook)}
    raw_last = {}
    for split, payload in (("fit", fit_payload), ("tune", tune_payload)):
        last = np.asarray(payload["raw_last"], dtype=np.float32)
        if last.ndim == 2:
            last = last[:, None, :]
        if last.shape != (len(latent[split]), 1, len(model.projection.center)):
            raise ValueError("NATIVE_RAW_LAST_SHAPE")
        if not np.isfinite(last).all():
            raise ValueError("NATIVE_RAW_LAST_NONFINITE")
        raw_last[split] = torch.as_tensor(last, dtype=torch.float32)
    target = torch.as_tensor(np.asarray(fit_payload["target"]), dtype=torch.float32)
    observed = torch.as_tensor(np.asarray(fit_payload["observed"]), dtype=torch.bool)
    if target.shape != observed.shape or target.shape != (len(latent["fit"]), H, len(model.projection.center)):
        raise ValueError("NATIVE_FIT_TARGET_SHAPE")
    if not torch.isfinite(target[observed]).all():
        raise ValueError("NATIVE_TARGET_NONFINITE_OBSERVED")
    order = np.random.default_rng(seed).integers(0, len(latent["fit"]), size=(updates, batch_size))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001, weight_decay=0.0)
    checkpoint_updates = sorted({0, updates, *(step for step in (100, 200, 400, 600) if step <= updates)})
    curve, checkpoints = [], []
    initial_anchor_difference = None
    for update in range(updates + 1):
        if update in checkpoint_updates:
            prediction = predict_cached_latents(model, latent["tune"], raw_last["tune"])
            score = score_predictions(prediction, tune_payload["target"], tune_payload["observed"])
            state = clone_state(model)
            checkpoints.append({"update": update, "tune": score,
                                "state_sha256": state_hash(state), "state": state})
            if update == 0:
                anchor = raw_last["tune"].numpy()
                initial_anchor_difference = float(np.abs(prediction - anchor).max())
            if checkpoint_callback is not None:
                checkpoint_callback(update, state, score)
            if progress_hook is not None:
                progress_hook({"stage": "checkpoint", "update": update, "tune": score})
        if update == updates:
            break
        indices = torch.as_tensor(order[update], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.predict_latents(latent["fit"][indices], raw_last["fit"][indices])
        loss = masked_macro_mse(prediction, target[indices], observed[indices])
        if not torch.isfinite(loss):
            raise ValueError("NATIVE_NONFINITE_LOSS")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise ValueError("NATIVE_NONFINITE_GRADIENT")
        optimizer.step()
        if not all(torch.isfinite(value).all() for value in model.head.parameters()):
            raise ValueError("NATIVE_NONFINITE_PARAMETER")
        curve.append({"update": update + 1, "loss": float(loss.detach()),
                      "gradient_norm_before_clip": float(gradient_norm.detach())})
        if progress_hook is not None and (update + 1) % 20 == 0:
            progress_hook({"stage": "training", **curve[-1]})
    selected = select_checkpoint(checkpoints)
    model.load_state_dict(selected["state"])
    model.eval().requires_grad_(False)
    frozen_final = state_hash(clone_state(model.projection))
    if frozen_final != frozen_initial:
        raise ValueError("NATIVE_FROZEN_PROJECTION_CHANGED")
    metadata = {
        "seed": int(seed), "updates": int(updates), "batch_size": int(batch_size),
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0, "gradient_clip": 1.0,
        "architecture": "K independent affine L-to-H heads on latent history minus latent last",
        "K": model.head.K, "L": L, "H": H, "parameters": sum(value.numel() for value in model.head.parameters()),
        "encoder_decoder_frozen": True, "E_D_optimized_for_native": False,
        "initial_weights_and_biases_zero": True, "seed_changes_initial_weights": False,
        "initial_model_sha256": initial_model, "frozen_projection_sha256": frozen_final,
        "selected_update": selected["update"], "selected_tune": selected["tune"],
        "selected_state_sha256": selected["state_sha256"],
        "selection_rule": "TUNE observed channel-macro normalized MSE; exact tie earliest update",
        "loss_rule": "Minibatch observed counts per output C; macro over active C",
        "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "checkpoint_updates": checkpoint_updates, "train_wall_seconds": time.perf_counter() - started,
        "CP0_anchor_max_abs_difference_FP32": initial_anchor_difference,
        "FP32_forecast_order": "raw_last + (latent_future - latent_last) @ decoder; decoder_bias cancels",
        "future_encoding_in_forward": False, "new_backbone_training": False,
    }
    return {"model": model, "selected_state": clone_state(model), "metadata": metadata,
            "learning_curve": curve, "checkpoints": checkpoints}
