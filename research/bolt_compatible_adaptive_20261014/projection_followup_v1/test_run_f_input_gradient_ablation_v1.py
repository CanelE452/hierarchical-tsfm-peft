"""Generated CPU tensors only: forward parity, gradient identity and four-lock barrier."""
from pathlib import Path
import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from linear_e_adaptation_v1 import LinearLagAdapter

try:
    import run_f_input_gradient_ablation_v1 as subject
except ModuleNotFoundError as error:
    if error.name != "run_f_input_gradient_ablation_v1": raise
    subject = None


class ToyF:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False).eval().requires_grad_(False)
        with torch.no_grad(): self.model.weight.fill_(1.7)
        self.calls = 0
    def __call__(self, z):
        self.calls += 1
        return self.model(.3 * z.mean(1, keepdim=True) + z[:, -1:]).expand(-1, 48, -1)


def models():
    args = (np.zeros(2, np.float32), np.array([[.5], [.2]], np.float32), np.array([[.8, -.3]], np.float32), (0,))
    return LinearLagAdapter(*args), LinearLagAdapter(*args)


class FInputGradientAblationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(subject, "new stop-gradient controller has not been implemented")
        return subject

    def test_detach_wrapper_keeps_actual_forward_and_model_identity_but_removes_F_input_graph(self):
        api = self.api(); f = ToyF(); z = torch.arange(12., requires_grad=True).reshape(2, 6, 1)
        wrapped = api.FDetach(f); full, stopped = f(z), wrapped(z)
        self.assertIs(wrapped.model, f.model); self.assertEqual(f.calls, 2)
        self.assertTrue(torch.equal(full, stopped)); self.assertFalse(stopped.requires_grad); self.assertTrue(full.requires_grad)

    def test_probe_D_equal_and_stop_E_equals_full_minus_F_vjp_no_optimizer_update(self):
        api = self.api(); full, stopped = models(); f = ToyF()
        x = torch.arange(48., dtype=torch.float32).reshape(4, 6, 2) / 10
        y = x[:, -1:].expand(-1, 48, -1).clone() + .2; observed = torch.ones_like(y, dtype=torch.bool)
        observed[0, :, 1] = False; y[~observed] = float("nan")
        state = {k: v.detach().clone() for k, v in full.state_dict().items()}
        info, arrays = api.gradient_probe(full, stopped, f, x, y, observed, observed.sum((0, 1)), 4)
        np.testing.assert_allclose(arrays["stop_E_gradient"], arrays["full_E_gradient"] - arrays["F_path_E_gradient"], atol=1e-5, rtol=2e-6)
        np.testing.assert_allclose(arrays["stop_D_gradient"], arrays["full_D_gradient"], atol=1e-5, rtol=2e-6)
        self.assertGreater(np.linalg.norm(arrays["F_path_E_gradient"]), 0.)
        self.assertGreater(np.linalg.norm(arrays["stop_E_gradient"]), 0.)
        self.assertEqual(info["optimizer_updates"], 0); self.assertTrue(info["forward_byte_exact"])
        self.assertTrue(all(p.grad is None for m in (full, stopped, f.model) for p in m.parameters()))
        self.assertTrue(all(torch.equal(v, full.state_dict()[k]) for k, v in state.items()))
        self.assertTrue(all(torch.equal(v, stopped.state_dict()[k]) for k, v in state.items()))
        json.dumps(info, allow_nan=False)

    def test_probe_rejects_different_initial_state_before_claiming_matched_gradient(self):
        api = self.api(); full, stopped = models()
        with torch.no_grad(): stopped.E.add_(.1)
        x = torch.ones(4, 6, 2); y = torch.ones(4, 48, 2); mask = torch.ones_like(y, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "INITIAL_STATE"): api.gradient_probe(full, stopped, ToyF(), x, y, mask, mask.sum((0, 1)), 4)

    def test_four_success_or_failure_locks_registry_before_any_development(self):
        api = self.api(); events = []; report = {"datasets": {}, "failed_paths": [], "stop_gradient_training_fits": 0, "training_updates_completed": 0, "checkpoint_states_saved": 0}
        def run(key):
            events.append(("run", key))
            if key == api.KEYS[1]: raise RuntimeError("one independent path fails")
            return {"status": "COMPLETE", "training_updates_completed": 600, "checkpoint_count": 5}
        def seal(key, row): events.append(("lock", key)); return {"path": "fixture", "sha256": "fixture"}
        def registry(rows): self.assertEqual(len(rows), 4); events.append(("registry", 4)); return {"path": "registry", "sha256": "fixture"}
        resolved = api.resolve_four(report, run, seal, registry)
        self.assertEqual(len(resolved), 3); self.assertEqual(report["resolved_selection_locks"], 4)
        self.assertEqual(report["training_updates_completed"], 1800); self.assertTrue(report["all_4_locks_before_DEV"])
        self.assertEqual(events[-1][0], "registry"); self.assertEqual(len(report["failed_paths"]), 1)

    def test_None_parent_report_gate_opens_no_asset(self):
        api = self.api()
        with patch.object(api, "REPORT_AUDIT_SHA", None), patch.object(api.old, "record", side_effect=AssertionError("closed gate read")), self.assertRaisesRegex(ValueError, "REPORT_GATE"):
            api.load_gates("missing", "missing", "missing", "missing", "missing")

    def test_reused_training_function_has_same_sample_order_and_updates_are_new_fits_only(self):
        api = self.api(); import run_linear_e_adaptation_v1 as previous
        self.assertIs(api.train_condition, previous.train_condition); self.assertIs(api.predict_components, previous.predict_components)
        full, stopped = models(); f = ToyF(); rng = np.random.default_rng(92)
        history = rng.normal(size=(8, 512, 2)).astype(np.float32); target = np.repeat(history[:, -1:], 48, axis=1) + np.float32(.1)
        payload = {"history": history, "target": target, "observed": np.ones(target.shape, bool)}
        initial = {k: v.detach().clone() for k, v in stopped.state_dict().items()}
        result = api.train_condition(stopped, api.FDetach(f), payload, payload, 92601, lambda *args: None, lambda *args: None, updates=2)
        self.assertEqual(result["training_updates_completed"], 2); self.assertEqual([c["update"] for c in result["checkpoints"]], [0, 2])
        self.assertIsNone(f.model.weight.grad); self.assertFalse(torch.equal(initial["E"], result["checkpoints"][-1]["state"]["E"]))

    def test_failed_numeric_gradient_probe_preserves_vectors_without_any_update(self):
        api = self.api(); full, stopped = models()
        class BadBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, z): return z[:, -1:].expand(-1, 48, -1)
            @staticmethod
            def backward(ctx, grad): return torch.full((len(grad), 6, 1), float("nan"))
        class BadF(ToyF):
            def __call__(self, z): return BadBackward.apply(z)
        x = torch.arange(48., dtype=torch.float32).reshape(4, 6, 2) / 10; y = torch.zeros(4, 48, 2); mask = torch.ones_like(y, dtype=torch.bool)
        info, arrays = api.gradient_probe(full, stopped, BadF(), x, y, mask, mask.sum((0, 1)), 4)
        self.assertEqual(info["status"], "FAILED_PROBE_PRESERVED"); self.assertFalse(info["finite_gradient_vectors"])
        self.assertTrue(np.isnan(arrays["full_E_gradient"]).any()); self.assertEqual(info["optimizer_updates"], 0)
        json.dumps(info, allow_nan=False)


if __name__ == "__main__": unittest.main()
