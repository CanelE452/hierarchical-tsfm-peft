"""SWA-based TS2Vec exports restricted to each supplied past window.

The official pinned source is imported without modification. Tensor exports
retain autograd; NumPy exports are explicitly frozen inference paths.
"""
from pathlib import Path
import copy
import json
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIRECT = HERE.parent / "professor_compressor_direct_20261011"
if str(DIRECT) not in sys.path:
    sys.path.append(str(DIRECT))
from common_direct import digest, state_digest
from ts2_adapter import create_ts2, freeze_encoder, vendor_provenance

PADDING = 200


def canonical_encoder_result(dataset, dim, seed):
    manifest = json.loads((DIRECT / "runs.json").read_text(encoding="utf-8-sig"))
    matches = [item for item in manifest["runs"]
               if item["job"]["family"] == "TS2"
               and item["job"]["dataset"] == dataset
               and item["job"]["dim"] == dim and item["job"]["seed"] == seed]
    if len(matches) != 1:
        raise ValueError("Exactly one canonical normal-size TS2Vec result is required")
    entry = matches[0]
    if entry["status"] != "complete_saved_predictions_verified":
        raise ValueError("Canonical encoder run is not verified complete")
    result_path = ROOT / entry["result_path"]
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    return result, result_path


def load_frozen_encoder(dataset, dim, seed, device="cuda"):
    """Load only the recorded SWA inference state, never an online resume."""
    result, result_path = canonical_encoder_result(dataset, dim, seed)
    record = result["training"]["encoder"]
    config = record["configuration"]
    if (config["output_dims"], config["hidden_dims"], config["depth"]) != (dim, 64, 10):
        raise ValueError("The declared normal-size encoder configuration differs")
    checkpoint = ROOT / record["checkpoint"]
    checkpoint_sha = digest(checkpoint)
    if checkpoint_sha != record["checkpoint_sha256"]:
        raise ValueError("Recorded SWA checkpoint bytes changed")
    source = vendor_provenance()
    model = create_ts2(config["input_dims"], dim, seed, device="cpu",
                       batch_size=config["batch_size"],
                       max_train_length=config["max_train_length"])
    model.net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    if state_digest(model.net) != record["inference_swa_state_sha256"]:
        raise ValueError("Loaded SWA state differs from the actual encoder training record")
    model.net.to(device)
    model.device = device
    freeze_encoder(model)
    return model, {"status": "REUSED_VERIFIED", "dataset": dataset, "dim": dim,
                   "seed": seed, "checkpoint": checkpoint,
                   "checkpoint_sha256": checkpoint_sha, "result": result_path,
                   "result_sha256": digest(result_path),
                   "SWA_state_sha256": state_digest(model.net),
                   "SWA_n_averaged": int(model.net.n_averaged),
                   "configuration": config, "source": source,
                   "new_encoder_optimizer_updates": 0,
                   "loaded_online_weights": False,
                   "restored_optimizer": False, "protected_TEST_new_access": 0}


class DeterministicSWAEncoder(nn.Module):
    """A separate trainable copy of the actual SWA inference tensor network."""
    def __init__(self, network, trainable):
        super().__init__()
        self.network = copy.deepcopy(network)
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(trainable)
        self.input_dims = self.network.input_dims
        self.output_dims = self.network.output_dims
        self.source_module_state_sha256 = state_digest(network)
        if state_digest(self.network) != self.source_module_state_sha256:
            raise RuntimeError("SWA tensor copy changed an encoder weight")
        self.eval()

    def forward(self, past):
        # The official encoder zeroes NaN rows in place, so preserve caller input.
        self.network.eval()
        return self.network(past.clone(), mask="all_true")


def tensor_encoder_from_swa(model, trainable=True):
    return DeterministicSWAEncoder(model.net.module, trainable)


def _mode(mode):
    mode = str(mode).upper()
    if mode in ("CAUSAL", "CAUSAL201"):
        return "CAUSAL201"
    if mode == "OFFLINE":
        return "OFFLINE"
    raise ValueError("Export must be OFFLINE or CAUSAL201")


def export_tensor(encoder, past, mode="OFFLINE", batch_size=128):
    """Differentiate through all independently computed past-ending windows."""
    if past.ndim != 3 or not past.shape[0] or not past.shape[1] or batch_size < 1:
        raise ValueError("Tensor export requires nonempty [N,L,C] and positive chunk size")
    mode = _mode(mode)
    if mode == "OFFLINE":
        return encoder(past)
    instances, length, channels = past.shape
    left = past.new_full((instances, PADDING, channels), float("nan"))
    padded = torch.cat((left, past), dim=1)
    offsets = torch.arange(PADDING + 1, device=past.device)
    chunks = []
    for first in range(0, instances * length, batch_size):
        positions = torch.arange(first, min(first + batch_size, instances * length),
                                 device=past.device)
        rows, times = positions // length, positions % length
        windows = padded[rows[:, None], times[:, None] + offsets[None, :]]
        chunks.append(encoder(windows)[:, -1])
    return torch.cat(chunks, dim=0).reshape(instances, length, -1)


def export_numpy(model, past, mode="OFFLINE", resource=None, batch_size=128,
                 offline_batch=8):
    """Frozen export; the full causal sequence contains L genuine E evaluations."""
    values = np.asarray(past, dtype=np.float32)
    if values.ndim != 3 or not len(values) or not values.shape[1]:
        raise ValueError("Export requires nonempty [N,L,C]")
    if np.isinf(values).any() or batch_size < 1 or offline_batch < 1:
        raise ValueError("Infinite input or invalid batch size")
    mode = _mode(mode)
    n, length, _ = values.shape
    dim = model._net.output_dims
    output = np.empty((n, length, dim), dtype=np.float32)
    previous_mode = model.net.training
    model.net.eval()
    try:
        with torch.no_grad():
            if mode == "OFFLINE":
                for first in range(0, n, offline_batch):
                    last = min(first + offline_batch, n)
                    batch = torch.from_numpy(values[first:last].copy()).to(model.device)
                    output[first:last] = model.net(batch, mask="all_true").cpu().numpy()
                    if resource is not None:
                        resource.check(stage="TS2_export_OFFLINE", completed_origins=last,
                                       total_origins=n, temporal_rows=length)
            else:
                padded = np.pad(values, ((0, 0), (PADDING, 0), (0, 0)),
                                constant_values=np.nan)
                offsets = np.arange(PADDING + 1, dtype=np.int64)
                flat = output.reshape(n * length, dim)
                for first in range(0, n * length, batch_size):
                    last = min(first + batch_size, n * length)
                    positions = np.arange(first, last, dtype=np.int64)
                    rows, times = positions // length, positions % length
                    windows = padded[rows[:, None], times[:, None] + offsets[None, :]]
                    batch = torch.from_numpy(windows).to(model.device)
                    flat[first:last] = model.net(batch, mask="all_true")[:, -1].cpu().numpy()
                    if resource is not None:
                        resource.check(stage="TS2_export_CAUSAL201",
                                       completed_origin_timestamps=last,
                                       total_origin_timestamps=n * length,
                                       outside_supplied_window_history=False)
    finally:
        model.net.train(previous_mode)
    if not np.isfinite(output).all():
        raise RuntimeError("Nonfinite TS2Vec representation")
    return output


def export_last_numpy(model, past, mode="CAUSAL201", batch_size=128):
    values = np.asarray(past, dtype=np.float32)
    if _mode(mode) == "OFFLINE":
        return export_numpy(model, values, mode, batch_size=batch_size)[:, -1]
    tail = values[:, -PADDING - 1:]
    if tail.shape[1] < PADDING + 1:
        tail = np.pad(tail, ((0, 0), (PADDING + 1 - tail.shape[1], 0), (0, 0)),
                      constant_values=np.nan)
    previous_mode = model.net.training
    model.net.eval()
    try:
        with torch.no_grad():
            parts = [model.net(torch.from_numpy(tail[first:first + batch_size].copy())
                               .to(model.device), mask="all_true")[:, -1].cpu().numpy()
                     for first in range(0, len(tail), batch_size)]
        return np.concatenate(parts)
    finally:
        model.net.train(previous_mode)


def verify_export(model, past, mode, sequence=None):
    """Full first-window parity with unmodified official encode, not shape-only."""
    trial = np.asarray(past, dtype=np.float32)[:1].copy()
    actual = export_numpy(model, trial, mode) if sequence is None else sequence[:1]
    mode = _mode(mode)
    reference = model.encode(trial.copy(), causal=mode == "CAUSAL201",
                             sliding_length=1 if mode == "CAUSAL201" else None,
                             sliding_padding=PADDING if mode == "CAUSAL201" else 0,
                             batch_size=128 if mode == "CAUSAL201" else 8)
    maximum = float(np.max(np.abs(actual - reference)))
    if not np.allclose(actual, reference, rtol=2e-5, atol=2e-5):
        raise RuntimeError(f"Official full sequence export differs: {maximum}")
    return {"status": "PASS", "export": mode, "origin_count": 1,
            "timestamp_count": trial.shape[1], "dimension": actual.shape[-1],
            "maximum_abs_difference": maximum, "rtol": 2e-5, "atol": 2e-5,
            "outside_supplied_window_history": False}


def context_dependency_diagnostic(model, past, timestamp=256, early_end=320):
    """Compare one past physical tick under two supplied-window termination rules."""
    values = np.asarray(past, dtype=np.float32)[:1].copy()
    if not 0 <= timestamp < early_end < values.shape[1]:
        raise ValueError("Diagnostic needs timestamp < earlier end < later end")
    first = export_numpy(model, values[:, :early_end], "OFFLINE")[:, timestamp]
    later = export_numpy(model, values, "OFFLINE")[:, timestamp]
    causal_first = export_last_numpy(model, values[:, :timestamp + 1])
    causal_full = export_numpy(model, values, "CAUSAL201")[:, timestamp]
    causal_difference = float(np.max(np.abs(causal_first - causal_full)))
    if not np.allclose(causal_first, causal_full, rtol=2e-5, atol=2e-5):
        raise RuntimeError("Same context/padding causal representation changed")
    return {"status": "PASS", "physical_local_timestamp": timestamp,
            "supplied_history_start": 0, "earlier_window_end": early_end,
            "later_window_end": values.shape[1],
            "offline_maximum_abs_difference": float(np.max(np.abs(first - later))),
            "offline_RMS_difference": float(np.sqrt(np.mean(np.square(first - later)))),
            "causal_same_context_maximum_abs_difference": causal_difference,
            "origin_after_observations_used": False,
            "interpretation": "context dependence diagnostic, not proof of forecasting harm"}
