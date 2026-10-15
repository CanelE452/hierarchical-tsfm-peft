"""Synthetic five-route timing contracts; no campaign arrays or actual F."""
from pathlib import Path
import ast
import importlib
import json
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import run_f_input_gradient_cost_v1 as controller
    import whole_f_input_gradient_cost_v1 as timer
except ModuleNotFoundError as error:
    if error.name not in ("run_f_input_gradient_cost_v1", "whole_f_input_gradient_cost_v1"): raise
    controller = timer = None


class FInputGradientCostTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(controller, "new five-route cost controller is missing")
        self.assertIsNotNone(timer, "new five-route timer is missing")

    def test_closed_science_audit_gate_opens_no_asset(self):
        self.api()
        with patch.object(controller, "STOP_AUDIT_SHA", None), patch.object(controller.old, "record", side_effect=AssertionError("closed gate read")), self.assertRaisesRegex(ValueError, "STOP_GRADIENT_PASS_NOT_REGISTERED"):
            controller.load_gates("missing", "missing")

    def test_exactly_two_reused_and_two_new_selected_rows_per_dataset(self):
        self.api(); full = {"datasets": {"robin": {"conditions": {}}}}; stopped = {"datasets": {"robin": {"conditions": {}}}}
        for seed in (92601, 92602):
            full["datasets"]["robin"]["conditions"]["robin_L0_E_LEARN_s" + str(seed)] = {"status": "COMPLETE", "seed": seed, "marker": "reused"}
            stopped["datasets"]["robin"]["conditions"][str(seed)] = {"status": "COMPLETE", "seed": seed, "marker": "new"}
        rows = controller.selected_rows(full, stopped, "robin")
        self.assertEqual([r[0] for r in rows], list(controller.ROUTES[1:])); self.assertEqual([r[1]["marker"] for r in rows], ["reused", "reused", "new", "new"])
        self.assertEqual([r[2] for r in rows], ["FULL_REUSED", "FULL_REUSED", "STOP_NEW", "STOP_NEW"])

    def test_five_full_rotations_and_actual_synthetic_whole_timer_ten_cells_fifty_samples(self):
        self.api(); rng = np.random.default_rng(12)
        physical = rng.normal(size=(8, 512, 2)).astype(np.float32); observed = np.ones(physical.shape, bool)
        observed[:, :3, 0] = False; physical[~observed] = np.nan
        mean, std = np.array([1., 2.], np.float32), np.array([2., 3.], np.float32)
        history = timer.preprocess(physical, observed, mean, std); origins = np.arange(8, dtype=np.int64) + 512
        predictions = {route: np.repeat(history[:, -1:], 48, axis=1) for route in timer.ROUTES}
        callbacks = {route: lambda h, m: np.repeat(h[:, -1:], 48, axis=1) for route in timer.ROUTES}
        metadata = {route: {"synthetic": True} for route in timer.ROUTES}
        memory = lambda: {"RSS_bytes": 1, "allocated_VRAM_bytes": 0, "reserved_VRAM_bytes": 0, "max_allocated_VRAM_bytes": 0, "max_reserved_VRAM_bytes": 0, "scope": "Shared resident frozen F and all available selected adapters; no exclusive per-method allocation"}
        with tempfile.TemporaryDirectory() as folder:
            value = timer.measure_cost("synthetic", physical, observed, mean, std, origins, callbacks, predictions, metadata, Path(folder) / "fresh", lambda: None, lambda: None, memory)
            self.assertEqual(value["status"], "COMPLETE"); self.assertEqual(value["completed_cost_cells"], 10); self.assertEqual(value["recorded_timer_samples"], 50)
            self.assertEqual(len(value["rows"]), 10); self.assertEqual(value["expected_cells_two_datasets"], 20)
            sample_files = [path for path in (Path(folder) / "fresh").rglob("*.npz") if re.search(r"_r\d+\.npz$", path.name)]
            self.assertEqual(len(sample_files), 50)
            json.dumps(value, allow_nan=False)
        for route in timer.ROUTES: self.assertEqual(sorted(order.index(route) for order in timer.balanced_schedule()), list(range(5)))

    def test_preprocessing_tolerances_and_actual_timer_boundary_are_unchanged_copies(self):
        self.api(); import whole_linear_e_cost_v1 as previous
        def functions(path): return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(Path(path).read_text(encoding="utf-8-sig")).body if isinstance(n, ast.FunctionDef)}
        old, new = functions(previous.__file__), functions(timer.__file__)
        for name in ("fill_vectorized", "preprocess", "parity", "whole_predict", "balanced_schedule"): self.assertEqual(old[name], new[name], name)
        self.assertEqual((timer.WARMUP, timer.REPEATS, timer.ORDER_SEED), (2, 5, 20261015))
        self.assertEqual((timer.NORM_ATOL, timer.NORM_RTOL, timer.PHYSICAL_ATOL, timer.PHYSICAL_RTOL), (1e-5, 2e-6, 2e-3, 2e-5))
        self.assertIs(controller.physical_precheck, controller.old.physical_precheck)
        self.assertIs(controller.adapter_predictor, controller.old.adapter_predictor)


if __name__ == "__main__": unittest.main()
