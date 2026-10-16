"""CPU contract/expiry failure injection, using no real data or foundation."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

import combination as c


class CombinationContractTests(unittest.TestCase):
    def fixture(self):
        from test_combination import CombinationTests
        return CombinationTests().toy()

    def test_transient_cache_flag_stable_but_real_model_change_changes_identity(self):
        _, pm, _, _, _ = self.fixture()
        fresh = copy.deepcopy(pm)
        cached = copy.deepcopy(pm)
        fresh["metadata"]["reused_prediction_cache"] = False
        cached["metadata"]["reused_prediction_cache"] = True
        self.assertEqual(c.cache_fingerprint(fresh), c.cache_fingerprint(cached))
        cached["metadata"]["model_sha"] = "different-trained-weights"
        self.assertNotEqual(c.cache_fingerprint(fresh), c.cache_fingerprint(cached))

    def test_completed_receipt_rejects_changed_parent_model_and_artifact_tamper(self):
        task, pm, pv, gm, gv = self.fixture()
        spec = dict(run_id="alpha", variant="alpha_family", seed=9, device="cpu")
        mean, std = np.zeros(3), np.ones(3)
        with tempfile.TemporaryDirectory(prefix="informs_combination_contract_") as temporary:
            root = Path(temporary)
            receipt = c.fit_combination(spec, task, pm, pv, gm, gv, mean, std, root/"cache", root/"public", None)
            c.validate_completed_receipt(receipt, spec, task, pm, pv, gm, gv, mean, std, root/"cache")
            changed_m, changed_v = copy.deepcopy(pm), copy.deepcopy(pv)
            changed_m["metadata"]["model_sha"] = "different-trained-weights"
            changed_v["metadata"]["model_sha"] = "different-trained-weights"
            with self.assertRaisesRegex(ValueError, "[Pp]arent"):
                c.validate_completed_receipt(receipt, spec, task, changed_m, changed_v, gm, gv, mean, std, root/"cache")
            file = root/"cache/combinations/alpha"/receipt["prediction_artifacts"]["fitted"]["V_select"]["file"]
            with file.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                c.validate_completed_receipt(receipt, spec, task, pm, pv, gm, gv, mean, std, root/"cache")

    def test_expired_before_CP0_does_not_infer_or_claim_a_selected_model(self):
        task, pm, pv, gm, gv = self.fixture()
        spec = dict(run_id="expired", variant="conditional", seed=9, device="cpu", epoch_endpoints=[.25, 1])
        with tempfile.TemporaryDirectory(prefix="informs_combination_contract_") as temporary:
            root = Path(temporary)
            with patch.object(c, "predict_head", side_effect=AssertionError("Post-budget inference must not start")):
                receipt = c.fit_combination(spec, task, pm, pv, gm, gv, np.zeros(3), np.ones(3), root/"cache", root/"public", time.time()-1)
            self.assertEqual(receipt["execution_status"], "BUDGET_LIMIT")
            self.assertEqual(receipt["scores"], [])
            self.assertEqual(receipt["prediction_artifacts"], {})
            self.assertIsNone(receipt["selected_V_mse"])
            self.assertIsNone(receipt["selected_state_sha"])
            self.assertFalse((root/"cache/combinations/expired/selected.pt").exists())
            saved = torch.load(root/"cache"/receipt["last_full_checkpoint_reference"], map_location="cpu", weights_only=True)
            self.assertEqual(saved["pending_endpoint"], 0.)
            self.assertEqual(saved["curves"], [])

    def test_budget_during_final_export_retains_resumable_completed_optimizer(self):
        task, pm, pv, gm, gv = self.fixture()
        spec = dict(run_id="export_paused", variant="conditional", seed=17, device="cpu", epoch_endpoints=[.25])
        original, calls = c.predict_head, {"n": 0}
        def interrupted_export(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:  # CP0 and endpoint V completed; final M export cannot complete.
                raise TimeoutError("Injected expiry during final export")
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory(prefix="informs_combination_contract_") as temporary:
            root = Path(temporary)
            with patch.object(c, "predict_head", side_effect=interrupted_export):
                partial = c.fit_combination(spec, task, pm, pv, gm, gv, np.zeros(3), np.ones(3), root/"cache", root/"public", None)
            self.assertEqual(partial["execution_status"], "BUDGET_LIMIT")
            self.assertEqual(partial["scores"], [])
            self.assertEqual(partial["prediction_artifacts"], {})
            checkpoint = root/"cache"/partial["last_full_checkpoint_reference"]
            resumed = c.fit_combination(dict(spec, run_id="export_resumed", resume_checkpoint=str(checkpoint)), task,
                                        pm, pv, gm, gv, np.zeros(3), np.ones(3), root/"cache", root/"public", None)
            self.assertIn("last_full_checkpoint_reference", resumed)
            self.assertTrue((root/"cache"/resumed["last_full_checkpoint_reference"]).is_file())
            whole = c.fit_combination(dict(spec, run_id="export_whole"), task, pm, pv, gm, gv,
                                      np.zeros(3), np.ones(3), root/"cache", root/"public", None)
            self.assertEqual(resumed["last_state_sha"], whole["last_state_sha"])
            self.assertEqual(resumed["selected_state_sha"], whole["selected_state_sha"])

    def test_expiry_inside_CP0_prediction_discards_partial_score_and_can_resume(self):
        task, pm, pv, gm, gv = self.fixture()
        spec = dict(run_id="cp0_partial", variant="conditional", seed=27, device="cpu", epoch_endpoints=[.25])
        original, count = c.ConditionalHead.forward, {"n": 0}
        def counted(head, *args, **kwargs):
            value = original(head, *args, **kwargs)
            count["n"] += 1
            return value
        with tempfile.TemporaryDirectory(prefix="informs_combination_contract_") as temporary:
            root = Path(temporary)
            with patch.object(c.ConditionalHead, "forward", new=counted), patch.object(c.time, "time", side_effect=lambda: 10. if count["n"] >= 2 else 0.):
                partial = c.fit_combination(spec, task, pm, pv, gm, gv, np.zeros(3), np.ones(3), root/"cache", root/"public", 5.)
            self.assertEqual(count["n"], 2)
            self.assertEqual(partial["scores"], [])
            self.assertIsNone(partial["selected_V_mse"])
            checkpoint = root/"cache"/partial["last_full_checkpoint_reference"]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["curves"], [])
            self.assertEqual(saved["pending_endpoint"], 0.)
            resumed = c.fit_combination(dict(spec, run_id="cp0_resumed", resume_checkpoint=str(checkpoint)), task,
                                        pm, pv, gm, gv, np.zeros(3), np.ones(3), root/"cache", root/"public", None)
            whole = c.fit_combination(dict(spec, run_id="cp0_whole"), task, pm, pv, gm, gv,
                                      np.zeros(3), np.ones(3), root/"cache", root/"public", None)
            self.assertEqual(resumed["last_state_sha"], whole["last_state_sha"])
            self.assertEqual(resumed["selected_state_sha"], whole["selected_state_sha"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
