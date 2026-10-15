"""Generated RECON state/probe and new diagnostic fixtures; no campaign assets/F."""
from pathlib import Path
from unittest.mock import patch
import copy
import json
import tempfile
import unittest

import numpy as np

import audit_recon_init_sensitivity_v1 as audit
from test_run_learned_output_space_diagnostic_v1 import fixture

audit.initialize_numpy()


def probe_fixture():
    state, raw, saved = fixture(audit.LAGS)
    reference = {"target": raw["target"], "observed": raw["observed"], "currentC": raw["raw_last"], "origins": raw["origins"], "FIT_mean": raw["mean"], "FIT_std": raw["std"], "ACTUAL_F_prediction_FP32": saved["prediction"], "latent_history": saved["latent_history"], "actual_F_future": saved["latent_future"]}
    arrays = {prefix+key: saved[key].copy() for prefix in ("Torch_", "previous_RECON_") for key in audit.COMPONENTS}; arrays["origins"] = raw["origins"].copy()
    row = {"probe": {"status": "PASS", "optimizer_updates": 0, "initial_state_sha256": audit.base.state_hash(state), "checks": {key: {"pass": True, "atol": 1e-5, "rtol": 2e-6, "max_abs": 0., "rmse": 0.} for key in audit.COMPONENTS}}}
    return row, arrays, reference, state, raw


class ReconInitAuditTests(unittest.TestCase):
    def test_exact_existing_RECON_D_same_PCA_E_center_and_different_FORECAST_D(self):
        _, _, _, state, _ = probe_fixture(); e = {"encoder": state["E"].astype(np.float64), "center": state["center"].astype(np.float64)}
        recon = {"decoder_FP32": state["D"].copy(), "lags": np.asarray(audit.LAGS, np.int64)}; old = {**state, "D": state["D"] + .2}
        actual = audit.initial_state(e, recon, old)
        self.assertEqual(actual["D"].tobytes(), recon["decoder_FP32"].tobytes()); self.assertNotEqual(actual["D"].tobytes(), old["D"].tobytes())
        for field in ("E", "center"):
            bad = copy.deepcopy(old); bad[field].flat[0] += .01
            with self.subTest(field=field), self.assertRaises(ValueError): audit.initial_state(e, recon, bad)
        bad = copy.deepcopy(recon); bad["lags"][-1] = 31
        with self.assertRaises(ValueError): audit.initial_state(e, bad, old)

    def test_zero_optimizer_CP0_original_components_and_all_recorded_errors(self):
        values = probe_fixture(); self.assertEqual(audit.check_probe(*values), 4)
        for key in ("Torch_prediction", "previous_RECON_latent_future", "origins"):
            row, arrays, reference, state, raw = copy.deepcopy(values); arrays[key].flat[0] += 1 if key == "origins" else .05
            with self.subTest(key=key), self.assertRaises(ValueError): audit.check_probe(row, arrays, reference, state, raw)
        row, arrays, reference, state, raw = copy.deepcopy(values); row["probe"]["checks"]["prediction"]["max_abs"] = .01
        with self.assertRaises(ValueError): audit.check_probe(row, arrays, reference, state, raw)
        row, arrays, reference, state, raw = copy.deepcopy(values); row["probe"]["optimizer_updates"] = 1
        with self.assertRaises(ValueError): audit.check_probe(row, arrays, reference, state, raw)

    def test_new_space_calculation_is_separate_and_CP0_supervision_flag_is_false(self):
        state, raw, saved = fixture(audit.LAGS)
        with tempfile.TemporaryDirectory() as folder:
            for update in (0, 400):
                path = Path(folder) / (str(update) + ".npz"); out = audit.new_space_diagnostic(saved, state, raw, update, path)
                self.assertEqual(out["status"], "COMPLETE"); self.assertEqual(out["selected_E_D_previously_supervised_by_FIT_future_targets"], update > 0)
                self.assertEqual(out["calculation_role"], "NEW_OUTPUT_SPACE_DIAGNOSTIC_COMPUTED_DURING_THIS_AUDIT")
                self.assertFalse(out["original_training_verification_replaced_by_this_calculation"]); self.assertFalse(out["independent_external_replay_of_new_diagnostic_claimed"])
                self.assertEqual(out["calculator_source"]["sha256"], audit.SPACE_SOURCE_SHA); self.assertTrue(out["independent_reduced_LS_vs_SVD_verified"])
                self.assertEqual(out["artifact"]["arrays"]["ACTUAL_F_prediction_FP32"]["shape"], list(saved["prediction"].shape)); self.assertTrue(path.exists()); json.dumps(out, allow_nan=False)
                with self.assertRaisesRegex(ValueError, "FRESH_ARRAY_OUTPUT"): audit.new_space_diagnostic(saved, state, raw, update, path)

    def test_modified_selected_prediction_rejected_and_no_universal_bound_claimed(self):
        state, raw, saved = fixture(audit.LAGS)
        with tempfile.TemporaryDirectory() as folder:
            out = audit.new_space_diagnostic(saved, state, raw, 600, Path(folder) / "good.npz")
            self.assertFalse(out["all_real_latent_or_all_FP32_lower_bound_claimed"]); self.assertFalse(out["RAW_bound_applied"])
            saved["prediction"][0, 0, 0] += .1
            with self.assertRaises(ValueError): audit.new_space_diagnostic(saved, state, raw, 600, Path(folder) / "bad.npz")

    def test_same_immutable_trainer_and_two_lock_source_contract(self):
        result = {"controller_source": {"path": str(audit.HERE / "run_recon_init_sensitivity_v1.py")}}
        parent = {"controller_source": {"path": str(audit.HERE / "run_linear_e_adaptation_v1.py")}, "pure_source": {"path": str(audit.HERE / "linear_e_adaptation_v1.py")}}
        audit.source_contract(result, parent)
        text = Path(result["controller_source"]["path"]).read_text()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrong_source.py"; path.write_text(text.replace("train_condition(model,bolt,fit,tune,seed,check,checkpoint)", "train_condition(model,None,fit,tune,seed,check,checkpoint)"))
            with self.assertRaisesRegex(ValueError, "FULL_GRADIENT_TRAINER"): audit.source_contract({"controller_source": {"path": str(path)}}, parent)

    def test_actual_gate_is_closed_before_any_asset_or_metadata_read(self):
        with patch.object(audit, "RESULT_SHA", None), patch.object(audit.base, "file_record", side_effect=AssertionError("closed gate read")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.metadata("absent", lambda: None)


if __name__ == "__main__": unittest.main()
