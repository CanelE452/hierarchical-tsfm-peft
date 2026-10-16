"""Fresh AplusM controls on CPU synthetic data, never real F/GPU."""
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def toy():
    rng = np.random.default_rng(456)
    physical = np.cumsum(rng.normal(size=(240, 3)), axis=0).astype(np.float32)
    return dict(name="toy", physical=physical, observed=np.ones_like(physical, bool),
                columns=["a", "b", "c"], L=64, H=4, data_sha256="toy-control-data")


class ControlFitTests(unittest.TestCase):
    def setUp(self):
        path = HERE / "fit_control.py"
        if not path.is_file():
            self.fail("Fresh AplusM control fitter is not implemented")
        spec = importlib.util.spec_from_file_location("fit_control", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory(prefix="informs_control_")
        self.root = Path(self.temp.name)
        def cleanup():
            assert self.root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()) and self.root.name.startswith("informs_control_")
            self.temp.cleanup()
        self.addCleanup(cleanup)
        torch.set_num_threads(2)

    def spec(self, family="DIRECT_G"):
        return dict(run_id=family + "_am", family=family, role="AplusM_control", seed=3,
                    epochs=[.25], lr=1e-3, device="cpu", microbatch=4, effective_batch=16)

    def test_fresh_G_AplusM_targets_A60_scaler_V_selection_and_cache_contract(self):
        spec = self.spec()
        receipt = self.module.fit_control(spec, toy(), self.root / "private", self.root / "public", None, None)
        self.assertEqual(receipt["status"], "COMPLETE_VALID")
        self.assertEqual(receipt["training_status"], "LIMITED_SCREEN")
        self.assertEqual(receipt["control_fit_end"], 168)
        self.assertEqual(receipt["normalization_fit_end"], 144)
        self.assertEqual(receipt["checkpoint_selection_split"], "V_select")
        self.assertEqual(receipt["exposures"], 26)
        saved = torch.load(self.root / "private" / receipt["artifact_cache_relative_path"], weights_only=False)
        self.assertEqual(saved["normalization"]["fit_end"], 144)
        self.assertEqual(saved["normalization"]["train_end"], 168)
        np.testing.assert_array_equal(saved["normalization"]["mean"], toy()["physical"][:144].astype(np.float64).mean(0).astype(np.float32))
        self.assertNotIn("search_summary", saved)
        fresh, _ = self.module.fit_base.create_model(spec, toy(), 144, None)
        self.assertEqual(receipt["initial_model_state_sha"], self.module.fit_base.networks.module_state_hash(fresh))
        repeated = self.module.fit_control(spec, toy(), self.root / "private", self.root / "public", None, None)
        self.assertTrue(repeated["reused_complete_artifact"])
        with self.assertRaises(ValueError):
            self.module.fit_control(dict(spec, epochs=[.25, 1]), toy(), self.root / "private", self.root / "public", None, None)

    def test_control_artifact_predicts_V_but_cannot_be_calibration_parent(self):
        import prediction_cache
        spec = self.spec()
        receipt = self.module.fit_control(spec, toy(), self.root / "private", self.root / "public", None, None)
        cache = prediction_cache.predict_parent(spec["run_id"], toy(), "V_select", self.root / "private", self.root / "public", None,
                                                  receipt=receipt, role="AplusM_control")
        self.assertEqual(cache["metadata"]["parent_fit_end"], 168)
        self.assertEqual(cache["metadata"]["parent_normalization_fit_end"], 144)
        with self.assertRaises(ValueError):
            prediction_cache.predict_parent(spec["run_id"], toy(), "M_cal", self.root / "private", self.root / "public", None,
                                              receipt=receipt, role="AplusM_control", dense=True)

    def test_control_budget_resume_matches_fresh_complete_weights(self):
        spec = self.spec()
        stopped = self.module.fit_control(spec, toy(), self.root / "resume", self.root / "resume_public", None, time.time() - 1)
        self.assertEqual(stopped["status"], "BUDGET_LIMIT")
        resumed = self.module.fit_control(spec, toy(), self.root / "resume", self.root / "resume_public", None, None)
        fresh = self.module.fit_control(spec, toy(), self.root / "fresh", self.root / "fresh_public", None, None)
        self.assertEqual(resumed["model_state_sha"], fresh["model_state_sha"])

    def test_fresh_lora_control_runs_updates_then_restores_f_and_hooks(self):
        networks = self.module.fit_base.networks
        class ToyFoundation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.ModuleDict({"q": torch.nn.Linear(64, 4), "v": torch.nn.Linear(64, 4)}).requires_grad_(False)
                self.initial_hash = networks.module_state_hash(self.model)
                self.streams = self.physical_streams = self.row_calls = self.physical_calls = 0
                self.device = "cpu"
            def forward(self, x):
                rows = x.transpose(1, 2).reshape(-1, 64)
                self.streams += len(rows); self.physical_streams += len(rows)
                self.row_calls += 1; self.physical_calls += 1
                y = self.model["q"](rows) + self.model["v"](rows)
                return y.reshape(x.shape[0], x.shape[2], 4).transpose(1, 2)
            def assert_frozen_unchanged(self):
                if networks.module_state_hash(self.model) != self.initial_hash:
                    raise AssertionError("Synthetic F changed")
                if any(parameter.grad is not None for parameter in self.model.parameters()):
                    raise AssertionError("Frozen F received parameter gradients")
        foundation = ToyFoundation()
        receipt = self.module.fit_control(self.spec("LORA"), toy(), self.root / "private", self.root / "public", foundation, None)
        self.assertEqual(receipt["status"], "COMPLETE_VALID")
        self.assertGreater(receipt["updates"], 0)
        self.assertGreater(receipt["F_logical_streams"], 0)
        self.assertFalse(receipt["pretrained_G_or_adapter_parent_reused"])
        foundation.assert_frozen_unchanged()
        self.assertTrue(all(not module._forward_hooks for module in foundation.model.values()))
        import prediction_cache
        predicted = prediction_cache.predict_parent("LORA_am", toy(), "V_select", self.root / "private", self.root / "public", foundation,
                                                    receipt=receipt, role="AplusM_control")
        self.assertEqual(predicted["cost_record"]["actual_F_streams"], len(predicted["origins"]) * 3)
        self.assertTrue(all(not module._forward_hooks for module in foundation.model.values()))


if __name__ == "__main__":
    unittest.main()
