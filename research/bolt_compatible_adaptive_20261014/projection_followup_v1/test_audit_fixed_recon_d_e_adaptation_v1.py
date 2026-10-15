"""Synthetic CPU saved-state/log/geometry evidence; no campaign assets or F."""
from pathlib import Path
from unittest.mock import patch
import copy
import hashlib
import json
import tempfile
import unittest

import numpy as np

import audit_fixed_recon_d_e_adaptation_v1 as audit
from test_run_learned_output_space_diagnostic_v1 import fixture

audit.initialize_numpy()


def probe_fixture():
    state, raw, saved = fixture(audit.LAGS)
    arrays = {prefix+key: saved[key].copy() for prefix in ("E_ONLY_", "JOINT_CP0_") for key in audit.COMPONENTS}; arrays["origins"] = raw["origins"].copy()
    row = {"probe": {"status": "PASS", "optimizer_updates": 0, "initial_state_sha256": audit.base.state_hash(state), "component_byte_exact": {key: True for key in audit.COMPONENTS}, "fixed_decoder_status": copy.deepcopy(audit.FLAGS)}}
    return row, arrays, saved, state, raw


class FixedDecoderAuditTests(unittest.TestCase):
    def test_D_and_center_are_byte_fixed_while_E_may_change_and_one_ULP_is_rejected(self):
        _, _, _, initial, _ = probe_fixture(); state = copy.deepcopy(initial); state["E"] += np.float32(.01); audit.fixed_state(state, initial)
        for key in ("D", "center"):
            bad = copy.deepcopy(state); bad[key].flat[0] = np.nextafter(bad[key].flat[0], np.float32(np.inf))
            with self.subTest(key=key), self.assertRaises(ValueError): audit.fixed_state(bad, initial)

    def test_two_way_CP0_component_bytes_and_zero_optimizer_probe(self):
        values = probe_fixture(); self.assertEqual(audit.check_probe(*values), 8)
        for key in ("E_ONLY_prediction", "JOINT_CP0_latent_future", "origins"):
            row, arrays, original, initial, raw = copy.deepcopy(values); arrays[key].flat[0] += .1 if key != "origins" else 1
            with self.subTest(key=key), self.assertRaises(ValueError): audit.check_probe(row, arrays, original, initial, raw)
        row, arrays, original, initial, raw = copy.deepcopy(values); row["probe"]["optimizer_updates"] = 1
        with self.assertRaisesRegex(ValueError, "ZERO_UPDATE"): audit.check_probe(row, arrays, original, initial, raw)

    def test_600_recorded_step_flags_D_norm_zero_and_scope_remain_distinct_from_replay(self):
        _, _, _, _, fit = probe_fixture(); order = np.random.default_rng(92601).integers(0, len(fit["history"]), size=(600, 4))
        logs = [{"stage": "TRAINING_UPDATE", "update": i+1, "loss": .1, "E_gradient_norm_before_clip": .5, "D_gradient_norm_before_clip": 0., "global_gradient_norm_before_clip": .5, "clipped": False, **audit.FLAGS, "gradient_clipping_parameter_scope": "E_ONLY"} for i in range(600)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"updates.jsonl"
            def write(): path.write_text("\n".join(json.dumps(item) for item in logs), encoding="utf-8")
            write(); row = {"updates_log": audit.base.file_record(path), "training_updates_completed": 600, "seed": 92601, "learn_E": True, "global_clipping_fraction": 0., "FIT_channel_observed_counts": fit["observed"].sum((0, 1)).tolist(), "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(), "fixed_decoder_final_status": copy.deepcopy(audit.FLAGS), "D_frozen_after_optimizer_step_checks": 600, "D_and_center_checkpoint_byte_checks": 5, "D_bytes_checked_at_checkpoints_and_final_only": True, "independent_per_update_D_bytes_replayed": False}
            self.assertEqual(audit.check_logs(row, fit), 600)
            for key, value in (("D_grad_is_None", False), ("D_gradient_norm_before_clip", .01), ("gradient_clipping_parameter_scope", "E_AND_D")):
                before = logs[3][key]; logs[3][key] = value; write(); row["updates_log"] = audit.base.file_record(path)
                with self.subTest(key=key), self.assertRaises(ValueError): audit.check_logs(row, fit)
                logs[3][key] = before
            write(); row["updates_log"] = audit.base.file_record(path); row["independent_per_update_D_bytes_replayed"] = True
            with self.assertRaisesRegex(ValueError, "SCOPE"): audit.check_logs(row, fit)

    def test_new_space_separates_selected_E_supervision_from_always_past_only_D(self):
        state, raw, saved = fixture(audit.LAGS)
        with tempfile.TemporaryDirectory() as folder:
            for update in (0, 400):
                out = audit.new_space(saved, state, raw, update, Path(folder)/(str(update)+".npz"))
                self.assertEqual(out["selected_E_previously_supervised_by_FIT_future_targets"], update > 0)
                self.assertFalse(out["selected_D_previously_supervised_by_FIT_future_targets"]); self.assertTrue(out["E_training_procedure_used_FIT_future_targets"])
                self.assertNotIn("selected_E_D_previously_supervised_by_FIT_future_targets", out)
                self.assertTrue(out["selected_D_held_fixed_at_past_RECON_initialization"]); self.assertFalse(out["independent_external_replay_of_new_diagnostic_claimed"])
                self.assertFalse(out["RAW_bound_applied"]); self.assertFalse(out["all_real_latent_or_all_FP32_lower_bound_claimed"]); json.dumps(out, allow_nan=False)

    def test_optional_bytes_bindings_are_equal_but_different_path_or_SHA_is_rejected(self):
        left = {"path": str(audit.HERE/"synthetic.json"), "sha256": "a"*64}; audit.same_binding(left, {**left, "bytes": 1})
        for key, value in (("path", str(audit.HERE/"other.json")), ("sha256", "b"*64)):
            with self.subTest(key=key), self.assertRaises(ValueError): audit.same_binding(left, {**left, key: value})

    def test_closed_actual_gate_precedes_source_asset_reads(self):
        with patch.object(audit, "RESULT_SHA", None), patch.object(audit.base, "file_record", side_effect=AssertionError("closed read")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.metadata("missing", lambda: None)
        with self.assertRaisesRegex(ValueError, "FLAGS"): audit.flags({**audit.FLAGS, "D_requires_grad": True})


if __name__ == "__main__": unittest.main()
