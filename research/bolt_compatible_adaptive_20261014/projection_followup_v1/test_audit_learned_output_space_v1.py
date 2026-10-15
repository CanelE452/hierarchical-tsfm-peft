"""Generated selected-state tensors only; producer used solely as fixture writer."""
from pathlib import Path
from unittest.mock import patch
import ast
import copy
import json
import unittest

import numpy as np

import audit_learned_output_space_v1 as audit
from test_run_learned_output_space_diagnostic_v1 import fixture
import run_learned_output_space_diagnostic_v1 as fixture_writer

audit.initialize_numpy(); fixture_writer.initialize_numpy()


def cell(lags=(0,)):
    state, raw, saved = fixture(lags); row, arrays = fixture_writer.diagnose_cell(saved, state, raw, lags)
    return row, arrays, saved, state, raw


class LearnedOutputAuditTests(unittest.TestCase):
    def test_L0_L7_new_joint_geometry_all_scores_and_scope_are_independently_checked(self):
        for lags in ((0,), audit.LAGS["L7"]):
            with self.subTest(lags=lags):
                row, arrays, saved, state, raw = cell(lags); out, count = audit.audit_cell(row, arrays, saved, state, raw, lags)
                self.assertEqual(count, 228); self.assertEqual(out["complete_target_positions"], 95)
                self.assertEqual(set(out["methods"]), {"ACTUAL_F", "EY", "FLAT"}); self.assertFalse(row["RAW_bound_applied"])
                self.assertTrue(out["methods"]["ACTUAL_F"]["retained_output_trace_bound_inequality_pass"])
                json.dumps(out, allow_nan=False)

    def test_unobserved_future_horizon_coordinate_remains_for_later_output_and_lag_offset(self):
        lags = (0, 1); state, raw, saved = fixture(lags)
        state["D"] = np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)
        raw["observed"][0, 0, 0] = False; raw["target"][0, 0, 0] = np.nan
        saved.update(observed=raw["observed"].copy(), target=raw["target"].copy(), prediction=audit.math_audit.anchored_forecast(saved["latent_history"], saved["latent_future"], raw["raw_last"], state["D"], lags))
        row, arrays = fixture_writer.diagnose_cell(saved, state, raw, lags); audit.audit_cell(row, arrays, saved, state, raw, lags)
        self.assertEqual(arrays["joint_A_FP64"].shape, (48*3, 48)); self.assertEqual(arrays["joint_A_FP64"][4, 0], 1.)
        bad = copy.deepcopy(arrays); bad["joint_A_FP64"] = bad["joint_A_FP64"][:, 1:]
        with self.assertRaises(ValueError): audit.audit_cell(row, bad, saved, state, raw, lags)

    def test_EY_incomplete_full_C_fill_and_flat_latent_differs_from_C_persistence(self):
        lags = audit.LAGS["L7"]; row, arrays, saved, state, raw = cell(lags)
        out, _ = audit.audit_cell(row, arrays, saved, state, raw, lags)
        np.testing.assert_array_equal(arrays["EY_input_C_FP32"][0, 1], raw["raw_last"][0]); self.assertFalse(out["latent_flat_equals_current_C_persistence_bytes"])
        bad = copy.deepcopy(arrays); bad["EY_input_C_FP32"][0, 1, 0] = raw["target"][0, 1, 0]
        with self.assertRaisesRegex(ValueError, "EY_input"): audit.audit_cell(row, bad, saved, state, raw, lags)

    def test_wrong_new_D_floor_map_offset_saved_future_score_and_retained_nan_rejected(self):
        original = cell()
        for field in ("joint_A_FP64", "joint_known_offset_FP64", "ACTUAL_F_future_FP32", "EY_prediction_FP32", "minimum", "score", "retained_nan"):
            row, arrays, saved, state, raw = copy.deepcopy(original)
            if field == "minimum": row["minimum_normalized_MSE"] += .05
            elif field == "score": row["methods"]["ACTUAL_F"]["scores_complete_mask"]["normalized"]["mse"] += .01
            elif field == "retained_nan": arrays["ACTUAL_F_retained_trace_FP64"][0, 1] = 0.
            else: arrays[field].flat[0] += .05
            with self.subTest(field=field), self.assertRaises(ValueError): audit.audit_cell(row, arrays, saved, state, raw, (0,))

    def test_complete_C_absent_is_explicit_and_does_not_reuse_a_previous_minimum(self):
        state, raw, saved = fixture(); raw["observed"][:, :, 2] = False; raw["target"][~raw["observed"]] = np.nan
        saved.update(observed=raw["observed"].copy(), target=raw["target"].copy())
        row, arrays = fixture_writer.diagnose_cell(saved, state, raw, (0,)); out, count = audit.audit_cell(row, arrays, saved, state, raw, (0,))
        self.assertEqual(row["status"], "NO_COMPLETE_C_TARGET_POSITIONS"); self.assertIsNone(out["minimum_normalized_MSE"]); self.assertEqual(count, 48)
        self.assertNotIn("joint_A_FP64", arrays)
        bad = copy.deepcopy(row); bad["minimum_normalized_MSE"] = .0
        with self.assertRaisesRegex(ValueError, "FABRICATED"): audit.audit_cell(bad, arrays, saved, state, raw, (0,))

    def test_no_universal_FP32_or_physical_minimum_or_new_label_fit_claim(self):
        original = cell()
        for key in ("RAW_bound_applied", "all_real_latent_or_all_FP32_lower_bound_claimed", "old_initial_D_minimum_reused", "diagnostic_future_labels_used_for_new_fit_or_selection"):
            row, arrays, saved, state, raw = copy.deepcopy(original); row[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError): audit.audit_cell(row, arrays, saved, state, raw, (0,))
        row, arrays, saved, state, raw = copy.deepcopy(original); row["joint_metadata"]["MAE_or_physical_MSE_minimum_claimed"] = True
        with self.assertRaisesRegex(ValueError, "NOT_UNIVERSAL"): audit.audit_cell(row, arrays, saved, state, raw, (0,))

    def test_closed_gate_precedes_data_IO_and_auditor_calls_no_producer_function(self):
        with patch.object(audit, "RESULT_SHA", None), patch.object(audit.learned, "file_record", side_effect=AssertionError("closed gate read")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.metadata("missing", lambda: None)
        tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertNotIn("run_learned_output_space_diagnostic_v1", imports); self.assertNotIn("causal_lag_decoder_v1", imports)


if __name__ == "__main__": unittest.main()
