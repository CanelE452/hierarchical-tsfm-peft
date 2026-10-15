"""Fresh affine latent heads with immutable, past-fitted per-window decoders.

The caller supplies checked cached NativeTorch32 past and selected D32[n]. No
scientific file, model, encoder, shrinkage fit or DEV split is opened here.
"""
import hashlib
import time

import numpy as np
import torch
from torch import nn

from native_head_v1 import NativeHead, clone_state, masked_macro_mse, score_predictions, select_checkpoint, state_hash


class NativeShrunkForecaster(nn.Module):
    def __init__(self, K, L=512, H=48):
        super().__init__()
        self.head = NativeHead(K, L, H)

    def predict_components(self, latent_history, raw_last, decoder):
        n = len(latent_history)
        if (latent_history.dtype != torch.float32 or latent_history.shape != (n, self.head.L, self.head.K)
                or raw_last.dtype != torch.float32 or raw_last.ndim != 2 or len(raw_last) != n
                or decoder.dtype != torch.float32 or decoder.shape != (n, self.head.K, raw_last.shape[1])
                or not all(value.device == self.head.weight.device and torch.isfinite(value).all()
                           for value in (latent_history, raw_last, decoder))):
            raise ValueError("MATCHED_FINITE_FP32_CACHED_PAST_CURRENT_PER_WINDOW_DECODER")
        past, current, fixed_decoder = (value.detach() for value in (latent_history, raw_last, decoder))
        future = self.head(past)
        delta = future - past[:, -1:, :]
        prediction = current[:, None, :] + torch.einsum("nhk,nkc->nhc", delta, fixed_decoder)
        return prediction, future

    def predict_latents(self, latent_history, raw_last, decoder):
        return self.predict_components(latent_history, raw_last, decoder)[0]

    def forward(self, latent_history, raw_last, decoder):
        return self.predict_latents(latent_history, raw_last, decoder)


def _tensor(value, dtype, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def predict_cached_components(model, latent_history, raw_last, decoder, batch_size=32):
    """One canonical batch32 path for checkpoint selection and saved predictions."""
    if batch_size <= 0:
        raise ValueError("POSITIVE_CANONICAL_PREDICTION_BATCH")
    for value in (latent_history, raw_last, decoder):
        dtype = value.dtype if isinstance(value, torch.Tensor) else np.asarray(value).dtype
        if dtype not in (np.dtype("float32"), torch.float32):
            raise ValueError("EXACT_FP32_CACHED_PREDICTION_INPUTS")
    device = model.head.weight.device
    z, current, matrices = (_tensor(value, torch.float32, device) for value in (latent_history, raw_last, decoder))
    if len(z) == 0 or len(z) != len(current) or len(z) != len(matrices):
        raise ValueError("NONEMPTY_MATCHED_PREDICTION_WINDOWS")
    predictions, futures = [], []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            prediction, future = model.predict_components(z[start:start + batch_size], current[start:start + batch_size], matrices[start:start + batch_size])
            predictions.append(prediction.detach().cpu()); futures.append(future.detach().cpu())
    prediction, future = torch.cat(predictions).numpy().copy(), torch.cat(futures).numpy().copy()
    if not np.isfinite(prediction).all() or not np.isfinite(future).all():
        raise ValueError("NONFINITE_NATIVE_SHRUNK_CACHED_FORECAST")
    return prediction, future


def predict_cached_latents(model, latent_history, raw_last, decoder, batch_size=32):
    return predict_cached_components(model, latent_history, raw_last, decoder, batch_size)[0]


def _binding(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def prepare_payload(payload, L, H, K, C):
    required = ("latent_history", "decoder", "raw_last", "target", "observed", "origins")
    arrays = {name: np.array(payload[name], order="C", copy=True) for name in required}
    z, d, current, y, mask, origins = (arrays[name] for name in required)
    n = len(z)
    if (n == 0 or z.dtype != np.float32 or z.shape != (n, L, K) or d.dtype != np.float32 or d.shape != (n, K, C)
            or current.dtype != np.float32 or current.shape != (n, C) or y.dtype != np.float32 or y.shape != (n, H, C)
            or mask.dtype != np.bool_ or mask.shape != y.shape or origins.dtype != np.int64 or origins.shape != (n,)
            or not all(np.isfinite(value).all() for value in (z, d, current)) or not np.isfinite(y[mask]).all()
            or not mask.any()):
        raise ValueError("FINITE_MATCHED_FP32_FROZEN_CACHED_PAYLOAD_AND_OBSERVED_TARGET")
    tensors = {name: torch.from_numpy(value.copy()).detach() for name, value in arrays.items()}
    return arrays, tensors, {name: _binding(value) for name, value in arrays.items()}


def train_native_shrunk(fit_payload, tune_payload, seed, progress_hook=None, checkpoint_callback=None,
                       *, updates=600, batch_size=4):
    """Train only a fresh head; callback receives the scored batch32 prediction.

    checkpoint_callback(update, state, score, prediction) receives detached CPU
    state plus a copy of the exact canonical prediction used to choose the CP.
    """
    if not 0 <= updates <= 600 or batch_size <= 0:
        raise ValueError("NATIVE_SHRUNK_TRAINING_LIMIT")
    started = time.perf_counter()
    torch.set_num_threads(2)
    z = np.asarray(fit_payload["latent_history"]); target = np.asarray(fit_payload["target"])
    if z.ndim != 3 or target.ndim != 3:
        raise ValueError("NATIVE_SHRUNK_TRAINING_DIMENSIONS")
    L, K, H, C = z.shape[1], z.shape[2], target.shape[1], target.shape[2]
    model = NativeShrunkForecaster(K, L, H).cpu()
    inputs = {}
    arrays = {}
    bindings = {}
    for split, payload in (("fit", fit_payload), ("tune", tune_payload)):
        arrays[split], inputs[split], bindings[split] = prepare_payload(payload, L, H, K, C)
    frozen_names = ("latent_history", "decoder", "raw_last")
    frozen_initial = {split: state_hash({name: values[name] for name in frozen_names}) for split, values in inputs.items()}
    initial_model = state_hash(clone_state(model))
    n_fit = len(inputs["fit"]["latent_history"])
    order = np.random.default_rng(seed).integers(0, n_fit, size=(updates, batch_size))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001, weight_decay=0.)
    checkpoints, curve = [], []
    checkpoint_updates = sorted({0, updates, *(value for value in (100, 200, 400, 600) if value <= updates)})
    anchor_difference = None
    for update in range(updates + 1):
        if update in checkpoint_updates:
            tune = inputs["tune"]
            prediction = predict_cached_latents(model, tune["latent_history"], tune["raw_last"], tune["decoder"], batch_size=32)
            score = score_predictions(prediction, arrays["tune"]["target"], arrays["tune"]["observed"])
            state = clone_state(model)
            checkpoints.append({"update": update, "tune": score, "state": state, "state_sha256": state_hash(state)})
            if update == 0: anchor_difference = float(np.abs(prediction - arrays["tune"]["raw_last"][:, None]).max())
            if checkpoint_callback is not None: checkpoint_callback(update, state, score, prediction.copy())
            if progress_hook is not None: progress_hook({"stage": "checkpoint", "update": update, "tune": score})
        if update == updates: break
        indices = torch.as_tensor(order[update], dtype=torch.int64)
        fit = inputs["fit"]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(fit["latent_history"][indices], fit["raw_last"][indices], fit["decoder"][indices])
        loss = masked_macro_mse(prediction, fit["target"][indices], fit["observed"][indices])
        if not torch.isfinite(loss): raise ValueError("NATIVE_SHRUNK_NONFINITE_LOSS")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.)
        if not torch.isfinite(norm): raise ValueError("NATIVE_SHRUNK_NONFINITE_GRADIENT")
        optimizer.step()
        if not all(torch.isfinite(value).all() for value in model.head.parameters()): raise ValueError("NATIVE_SHRUNK_NONFINITE_PARAMETER")
        curve.append({"update": update + 1, "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm.detach())})
        if progress_hook is not None and (update + 1) % 20 == 0: progress_hook({"stage": "training", **curve[-1]})
    selected = select_checkpoint(checkpoints)
    model.load_state_dict(selected["state"]); model.eval().requires_grad_(False)
    frozen_final = {split: state_hash({name: values[name] for name in frozen_names}) for split, values in inputs.items()}
    if frozen_final != frozen_initial or any(inputs[split][name].requires_grad or inputs[split][name].grad is not None
            for split in inputs for name in frozen_names): raise ValueError("NATIVE_SHRUNK_FROZEN_INPUT_CHANGED")
    metadata = {"seed": int(seed), "updates": int(updates), "batch_size": int(batch_size), "K": K, "L": L, "H": H, "C": C,
        "optimizer": "AdamW", "learning_rate": .001, "weight_decay": 0., "gradient_clip": 1.,
        "architecture": "Same K independent affine L-to-H heads on cached latent past minus last",
        "parameters": sum(value.numel() for value in model.head.parameters()), "initial_weights_and_biases_zero": True,
        "seed_changes_initial_weights": False, "initial_model_sha256": initial_model, "selected_update": selected["update"],
        "selected_tune": selected["tune"], "selected_state_sha256": selected["state_sha256"],
        "selection_rule": "Full TUNE observed C-macro normalized MSE; exact tie earliest update",
        "decoder_condition_selected_by_native": False, "rho_selection_inside_trainer": False,
        "loss_rule": "Same per-minibatch observed count per output C, macro over active C",
        "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(), "checkpoint_updates": checkpoint_updates,
        "canonical_checkpoint_prediction_batch_size": 32, "checkpoint_callback_prediction_is_exact_scored_array": True,
        "frozen_payload_before": frozen_initial, "frozen_payload_after": frozen_final, "payload_array_bindings": bindings,
        "encoder_decoder_backbone_optimized": False, "fixed_per_window_D_beta_requires_grad": False,
        "future_encoding_in_forward": False, "new_backbone_training": False, "DEV_loaded_by_trainer": False,
        "old_head_weights_reused_or_warm_started": False, "train_wall_seconds": time.perf_counter() - started,
        "CP0_anchor_max_abs_difference_FP32": anchor_difference, "zero_innovation_anchor_scope": "Numeric FP32 equality; signed-zero bit identity not assumed",
        "forecast_order": "future=head(cached_Z); delta=future-lastZ; currentC+torch.einsum('nhk,nkc->nhc',delta,D32[n])",
        "backend_decoder_scope": "Caller supplies selected NativeTorch32 D[n]; same rho/lambda rule does not imply byte-identical Bolt/native D or lambda"}
    return {"model": model, "selected_state": clone_state(model), "metadata": metadata, "learning_curve": curve, "checkpoints": checkpoints}
