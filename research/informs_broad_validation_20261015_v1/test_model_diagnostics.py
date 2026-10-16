"""CPU-only dependency/FLOP probes on synthetic histories."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        path = HERE / "model_diagnostics.py"
        if not path.is_file():
            self.fail("CPU encoder diagnostics are not implemented")
        spec = importlib.util.spec_from_file_location("model_diagnostics", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        torch.set_num_threads(2)

    def model(self, encoder="pointwise"):
        import networks
        model = networks.ForecastAdapter("B0", 64, 4, 4, 2, np.zeros(4), np.eye(4, dtype=np.float32), encoder)
        if hasattr(model, "lag_weights"):
            with torch.no_grad():
                model.lag_weights.fill_(.1)
        return model.eval()

    def test_realwindow_probe_is_prefix_causal_only_for_appropriate_architecture(self):
        x = torch.randn(1, 64, 4)
        for encoder in ("pointwise", "causal7"):
            result = self.module.past_perturbation_probe(self.model(encoder), x)
            self.assertEqual(result["prefix_max_abs_change"], 0)
        result = self.module.past_perturbation_probe(self.model("bidirectional7"), x)
        self.assertGreater(result["prefix_max_abs_change"], 0)
        self.assertFalse(result["future_target_was_input"])

    def test_shifted_synthetic_windows_preserve_internal_lag_coordinates(self):
        for encoder in ("pointwise", "causal7", "bidirectional7"):
            result = self.module.synthetic_shift_probe(self.model(encoder), seed=33)
            self.assertEqual(result["overlap_max_abs_change"], 0)
            self.assertEqual(result["excluded_left"], 32)
            self.assertEqual(result["excluded_right"], 16)
            self.assertTrue(result["synthetic_only"])

    def test_decoder_prefix_matches_explicit_lag_sum(self):
        model = self.model()
        with torch.no_grad():
            model.D.copy_(torch.arange(7 * 2 * 4).reshape(7, 2, 4) / 50)
        result = self.module.decoder_prefix_probe(model, torch.randn(1, 64, 4))
        self.assertTrue(result["allclose"])
        self.assertEqual(result["probe_k"], 1)

    def test_cpu_profiler_counts_supported_mm_and_excludes_F(self):
        model = self.model()
        result = self.module.profile_modules(model, torch.randn(1, 64, 4))
        self.assertEqual(result["encoder_mm_flops"], 2 * 64 * 4 * 2)
        self.assertEqual(result["decoder_mm_flops"], 2 * 4 * 7 * 2 * 4)
        self.assertEqual(result["foundation_calls"], 0)
        self.assertEqual(result["profile_device"], "CPU")


if __name__ == "__main__":
    unittest.main()
