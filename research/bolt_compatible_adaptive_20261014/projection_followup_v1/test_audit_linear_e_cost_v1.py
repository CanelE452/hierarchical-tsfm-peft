"""Synthetic actual Torch callback/timer records versus independent CPU audit."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import audit_linear_e_cost_v1 as audit
import run_linear_e_cost_v1 as producer
import whole_linear_e_cost_v1 as timer
from linear_e_adaptation_v1 import LinearLagAdapter

audit.initialize_numpy()


def memory():
    return {"RSS_bytes": 4096, "allocated_VRAM_bytes": 100, "reserved_VRAM_bytes": 200, "max_allocated_VRAM_bytes": 300, "max_reserved_VRAM_bytes": 400, "scope": "Shared resident frozen F and all available selected adapters; no exclusive per-method allocation"}


def fixture(folder):
    rng = np.random.default_rng(645)
    x = rng.normal(size=(8, 512, 3)).astype(np.float32); m = np.ones(x.shape, bool); m[:, :4, 0] = False; x[~m] = np.nan
    mean, std = np.array([1., 2., 3.], np.float32), np.array([2., 3., 4.], np.float32)
    history = audit.prior.causal_physical_preprocess(x, m, mean, std); origins = np.arange(8, dtype=np.int64) + 512
    raw = {"history": history, "history_observed": m, "mean": mean, "std": std, "origins": origins, "raw_last": history[:, -1].copy(), "target": np.zeros((8, 48, 3), np.float32), "observed": np.ones((8, 48, 3), bool)}
    physical = {"history_physical": x, "history_observed": m, "origins": origins}
    def toyF(z): return z[:, -1:].expand(-1, 48, -1) + .1
    def tensor(x): return torch.as_tensor(x, dtype=torch.float32)
    callbacks = {"RAW_BOLT": lambda h, mask: toyF(tensor(h)).numpy().copy()}; predictions = {"RAW_BOLT": callbacks["RAW_BOLT"](history, m)}
    metadata = {route: {"synthetic_route": route} for route in audit.ROUTES}; prechecks = []
    for route in audit.ROUTES[1:]:
        lags = audit.LAGS[route[:2]]; state = {"center": np.array([.1, -.2, .3], np.float32), "E": rng.normal(size=(3, 2)).astype(np.float32), "D": rng.normal(scale=.1, size=(len(lags) * 2, 3)).astype(np.float32)}
        model = LinearLagAdapter(state["center"], state["E"], state["D"], lags).eval().requires_grad_(False)
        callbacks[route] = producer.adapter_predictor(model, toyF, tensor)
        with torch.no_grad():
            p, z, f = model(tensor(history), toyF, return_components=True)
            reference = {"prediction": p.numpy().copy(), "latent_history": z.numpy().copy(), "latent_future": f.numpy().copy(), "lag_features": model.lag_features(z, f).numpy().copy(), "origins": origins}
        predictions[route] = reference["prediction"]
        entry, arrays = producer.physical_precheck(physical, raw, reference, model, toyF, tensor)
        prechecks.append((entry, arrays, physical, raw, state, reference, lags))
    result = timer.measure_cost("synthetic", x, m, mean, std, origins, callbacks, predictions, metadata, folder, lambda: None, lambda: None, memory)
    return result, physical, raw, predictions, metadata, prechecks


class LinearECostAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.folder.cleanup)
        cls.values = fixture(Path(cls.folder.name) / "whole")

    def test_full_synthetic_Torch_callbacks_physical_prechecks_and_18_cells_162_saved_timers(self):
        result, physical, raw, predictions, metadata, prechecks = self.values
        checks = [audit.precheck(*args) for args in prechecks]
        verified = audit.check_dataset(result, physical, raw, predictions, metadata, lambda: None, 0.)
        self.assertEqual(verified["complete_cells"], 18); self.assertEqual(verified["saved_samples"], 162)
        self.assertEqual(sum(r["windows_checked"] for r in checks), 64); self.assertFalse(verified["independent_F_or_timer_replayed"])
        json.dumps(verified, allow_nan=False)

    def test_changed_selected_reference_or_ns_or_order_is_rejected(self):
        for field in ("reference", "ns", "order"):
            result, physical, raw, predictions, metadata, _ = copy.deepcopy(self.values)
            if field == "reference": predictions[audit.ROUTES[-1]][0, 0, 0] += .1
            elif field == "ns": result["rows"][0]["samples"][0]["perf_end_ns"] += 1
            else: result["order_seed"] = 24261015
            with self.subTest(field=field), self.assertRaises(ValueError): audit.check_dataset(result, physical, raw, predictions, metadata, lambda: None, 0.)

    def test_exact_state_or_physical_mask_or_origin_tamper_rejected(self):
        for field in ("state", "mask", "origins"):
            args = copy.deepcopy(self.values[-1][0])
            if field == "state": args[4]["E"][0, 0] += .1
            elif field == "mask": args[1]["history_observed"][0, 0, 1] = False
            else: args[1]["origins"][0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError): audit.precheck(*args)

    def test_nine_full_Latin_rotations_and_actual_F_source_boundary(self):
        self.assertEqual(audit.schedule(), timer.balanced_schedule(20261015))
        for route in audit.ROUTES: self.assertEqual(sorted(order.index(route) for order in audit.schedule()), list(range(9)))
        texts = [(audit.HERE / name).read_text(encoding="utf-8") for name in ("whole_linear_e_cost_v1.py", "run_linear_e_cost_v1.py", "whole_causal_lag_cost_v1.py")]
        self.assertTrue(audit.source_contract(*texts)["TUNE_first_B_origins"])
        wrong = list(texts); wrong[1] = wrong[1].replace('raw_input_sources"]["tune"]', 'raw_input_sources"]["eval"]')
        with self.assertRaises(ValueError): audit.source_contract(*wrong)

    def test_memory_is_shared_and_no_cost_access_before_registration(self):
        result, physical, raw, predictions, metadata, _ = copy.deepcopy(self.values)
        result["rows"][0]["memory"][0]["scope"] = "exclusive per method"
        with self.assertRaises(ValueError): audit.check_dataset(result, physical, raw, predictions, metadata, lambda: None, 0.)
        with patch.object(audit, "COST_SHA", None), patch.object(audit.linear, "file_record", side_effect=AssertionError("closed gate read")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.audit_result("missing", None)


if __name__ == "__main__": unittest.main()
