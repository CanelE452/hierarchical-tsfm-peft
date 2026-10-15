"""Fresh actual-C heads with FIT-global fixed D; no data IO or encoder execution."""
import hashlib
import time

import numpy as np
import torch
from torch import nn

from native_head_v1 import NativeHead, clone_state, masked_macro_mse, score_predictions, select_checkpoint, state_hash


class NativeFixedDecoder(nn.Module):
    def __init__(self, decoder, L=512, H=48):
        super().__init__()
        d = np.asarray(decoder)
        if d.dtype != np.float32 or d.ndim != 2 or not d.size or not np.isfinite(d).all():
            raise ValueError("FINITE_FIXED_FIT_DECODER_FP32_KC_REQUIRED")
        self.register_buffer("decoder", torch.from_numpy(d.copy()))
        self.head = NativeHead(d.shape[0], L, H)

    def components(self, latent_history, raw_last):
        n, k, c = len(latent_history), self.head.K, self.decoder.shape[1]
        if (latent_history.dtype != torch.float32 or latent_history.shape != (n, self.head.L, k)
                or raw_last.dtype != torch.float32 or raw_last.shape != (n, c)
                or not torch.isfinite(latent_history).all() or not torch.isfinite(raw_last).all()):
            raise ValueError("FINITE_MATCHED_CACHED_PAST_AND_CURRENT_FP32_REQUIRED")
        z, current = latent_history.detach(), raw_last.detach()
        future = self.head(z)
        prediction = current[:, None] + (future-z[:, -1:]) @ self.decoder.detach()
        return prediction, future

    def forward(self, latent_history, raw_last):
        return self.components(latent_history, raw_last)[0]


def predict_components(model, latent_history, raw_last, batch_size=32):
    z, current = np.asarray(latent_history), np.asarray(raw_last)
    if z.dtype != np.float32 or current.dtype != np.float32 or not len(z) or batch_size != 32:
        raise ValueError("EXACT_FP32_INPUTS_AND_CANONICAL_BATCH32_REQUIRED")
    predictions, futures = [], []
    with torch.no_grad():
        for start in range(0, len(z), 32):
            p, f = model.components(torch.from_numpy(z[start:start+32]), torch.from_numpy(current[start:start+32]))
            predictions.append(p.detach().cpu()); futures.append(f.detach().cpu())
    p, f = (value.numpy().copy() for value in (torch.cat(predictions), torch.cat(futures)))
    if not np.isfinite(p).all() or not np.isfinite(f).all(): raise ValueError("NONFINITE_FIXED_DECODER_NATIVE_FORECAST")
    return p, f


def prepare(payload, L, H, K, C):
    names = ("latent_history", "raw_last", "target", "observed", "origins")
    values = {name: np.array(payload[name], order="C", copy=True) for name in names}
    z, current, target, mask, origins = (values[name] for name in names)
    n = len(z)
    if (n == 0 or z.dtype != np.float32 or z.shape != (n, L, K) or current.dtype != np.float32 or current.shape != (n, C)
            or target.dtype != np.float32 or target.shape != (n, H, C) or mask.dtype != np.bool_ or mask.shape != target.shape
            or origins.dtype != np.int64 or origins.shape != (n,) or not mask.any()
            or not np.isfinite(z).all() or not np.isfinite(current).all() or not np.isfinite(target[mask]).all()):
        raise ValueError("MATCHED_FP32_FIXED_DECODER_NATIVE_TRAINING_PAYLOAD_REQUIRED")
    return values, {name: torch.from_numpy(value.copy()).detach() for name, value in values.items()}


def train_native_fixed(decoder, fit_payload, tune_payload, seed, progress_hook=None, checkpoint_callback=None, *, updates=600):
    if not 0 <= updates <= 600: raise ValueError("FIXED_NATIVE_MAXIMUM_600_UPDATES")
    torch.set_num_threads(2)
    started = time.perf_counter()
    model = NativeFixedDecoder(decoder).cpu()
    K, C = model.decoder.shape
    arrays, tensors = {}, {}
    for split, payload in (("fit", fit_payload), ("tune", tune_payload)):
        arrays[split], tensors[split] = prepare(payload, 512, 48, K, C)
    before = state_hash({"decoder": model.decoder})
    frozen = {split: state_hash({name: values[name] for name in ("latent_history", "raw_last")}) for split, values in tensors.items()}
    initial = state_hash(clone_state(model.head))
    order = np.random.default_rng(seed).integers(0, len(arrays["fit"]["latent_history"]), size=(updates, 4))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001, weight_decay=0.)
    checkpoint_updates = sorted({0, updates, *(value for value in (100, 200, 400, 600) if value <= updates)})
    checkpoints, curve = [], []
    anchor_difference = None
    for update in range(updates+1):
        if update in checkpoint_updates:
            prediction, _ = predict_components(model, arrays["tune"]["latent_history"], arrays["tune"]["raw_last"])
            score = score_predictions(prediction, arrays["tune"]["target"], arrays["tune"]["observed"])
            state = clone_state(model)
            checkpoints.append({"update": update, "tune": score, "state": state, "state_sha256": state_hash(state)})
            if update == 0: anchor_difference = float(np.max(np.abs(prediction-arrays["tune"]["raw_last"][:, None])))
            if checkpoint_callback: checkpoint_callback(update, state, score, prediction.copy())
            if progress_hook: progress_hook({"stage": "checkpoint", "update": update, "tune": score})
        if update == updates: break
        fit, indices = tensors["fit"], torch.from_numpy(order[update])
        optimizer.zero_grad(set_to_none=True)
        loss = masked_macro_mse(model(fit["latent_history"][indices], fit["raw_last"][indices]), fit["target"][indices], fit["observed"][indices])
        if not torch.isfinite(loss): raise ValueError("NONFINITE_FIXED_NATIVE_LOSS")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.)
        if not torch.isfinite(norm): raise ValueError("NONFINITE_FIXED_NATIVE_GRADIENT")
        optimizer.step()
        if not all(torch.isfinite(value).all() for value in model.head.parameters()): raise ValueError("NONFINITE_FIXED_NATIVE_PARAMETERS")
        row = {"update": update+1, "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm.detach())}
        curve.append(row)
        if progress_hook and (update+1) % 20 == 0: progress_hook({"stage": "training", **row})
    selected = select_checkpoint(checkpoints)
    model.load_state_dict(selected["state"]); model.eval().requires_grad_(False)
    after = state_hash({"decoder": model.decoder})
    if after != before or model.decoder.grad is not None or any(tensors[split][name].grad is not None for split in tensors for name in ("latent_history", "raw_last")):
        raise ValueError("FIXED_DECODER_OR_CACHED_INPUTS_CHANGED")
    if frozen != {split: state_hash({name: values[name] for name in ("latent_history", "raw_last")}) for split, values in tensors.items()}:
        raise ValueError("FROZEN_FIXED_NATIVE_PAST_INPUT_CHANGED")
    def binding(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    return {"model": model, "selected_state": clone_state(model), "checkpoints": checkpoints, "learning_curve": curve,
        "metadata": {"seed": int(seed), "updates": updates, "batch_size": 4, "L": 512, "H": 48, "K": K, "C": C,
            "optimizer": "AdamW", "learning_rate": .001, "weight_decay": 0., "gradient_clip": 1.,
            "architecture": "Original K independent affine512-to48 NativeHead; actual-C observed minibatch macro MSE",
            "initial_head_weights_and_bias_zero": True, "seed_changes_initial_weights": False, "initial_head_sha256": initial,
            "selected_update": selected["update"], "selected_tune": selected["tune"], "selected_state_sha256": selected["state_sha256"],
            "checkpoint_updates": checkpoint_updates, "canonical_prediction_batch_size": 32,
            "checkpoint_callback_prediction_is_exact_scored_array": True, "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
            "fixed_decoder_before_sha256": before, "fixed_decoder_after_sha256": after,
            "fixed_global_FIT_decoder": True, "per_origin_decoder_fit_or_rho_selection": False,
            "decoder_backbone_or_encoder_optimized": False, "old_head_warm_start_or_reuse": False,
            "future_encoding_in_forward": False, "DEV_loaded_by_trainer": False,
            "selection": "Entire TUNE observed normalized C macro MSE; exact tie earliest update",
            "CP0_currentC_numeric_max_abs_difference": anchor_difference, "signed_zero_bit_identity_claimed": False,
            "forecast_order": "currentC + (head(Zpast)-lastZ) @ fixed_D32; affine reconstruction bias cancels",
            "payload_array_bindings": {split: {name: binding(value) for name, value in values.items()} for split, values in arrays.items()},
            "train_wall_seconds": time.perf_counter()-started}}
