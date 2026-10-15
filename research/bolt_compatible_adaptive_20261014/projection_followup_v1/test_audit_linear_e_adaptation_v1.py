"""Small generated tensors/JSON logs; no campaign arrays, F or training replay."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import audit_linear_e_adaptation_v1 as audit
from linear_e_adaptation_v1 import LinearLagAdapter

audit.initialize_numpy()


def fixture(lags=(0, 1, 2, 4, 8, 16, 32)):
    rng = np.random.default_rng(666)
    history = rng.normal(size=(4, 512, 3)).astype(np.float32)
    state = {"E": rng.normal(size=(3, 2)).astype(np.float32), "D": rng.normal(size=(len(lags) * 2, 3)).astype(np.float32), "center": np.array([.1, -.2, .3], np.float32)}
    raw = {"history": history, "raw_last": history[:, -1].copy(), "origins": np.arange(4, dtype=np.int64), "mean": np.array([1., 2., 3.], np.float32), "std": np.array([2., 3., 4.], np.float32)}
    model = LinearLagAdapter(state["center"], state["E"], state["D"], lags)
    future = torch.from_numpy(rng.normal(size=(4, 48, 2)).astype(np.float32))
    with torch.no_grad():
        z = model.encode(torch.from_numpy(history)); feature = model.lag_features(z, future); p = model.decode(torch.from_numpy(raw["raw_last"]), z, future)
    raw["target"] = p.numpy().copy() + np.float32(.3); raw["observed"] = np.ones(raw["target"].shape, bool)
    raw["observed"][0, :, 0] = False; raw["observed"][:, :, 2] = False; raw["target"][~raw["observed"]] = np.nan
    saved = {**{k: raw[k].copy() for k in ("target", "observed", "raw_last", "origins", "mean", "std")}, "prediction": p.numpy(), "latent_history": z.numpy(), "latent_future": future.numpy(), "lag_features": feature.numpy()}
    return saved, state, raw


class LinearEIndependentAuditTests(unittest.TestCase):
    def test_independent_numpy_replays_torch_encoding_lag_anchor_and_scores_with_missing_NaN(self):
        saved, state, raw = fixture()
        score = audit.replay(saved, state, raw, audit.LAGS["L7"])
        unit = score["normalized"]
        self.assertEqual(unit["observed_counts"], [144, 192, 0]); self.assertEqual(unit["active_channels"], 2)
        expected = sum(np.mean((saved["prediction"][:, :, c][raw["observed"][:, :, c]].astype(np.float64) - raw["target"][:, :, c][raw["observed"][:, :, c]].astype(np.float64)) ** 2) for c in (0, 1)) / 2
        self.assertEqual(unit["mse"], expected)
        entry = {"scores_all_observed": score, "scores_complete_mask": {"status": "NO_COMPLETE_C"}}
        self.assertEqual(audit.check_score(entry, saved, raw), 12)

    def test_changed_lag_source_current_or_saved_state_rejected(self):
        original = fixture()
        for field in ("lag", "current", "decoder", "encoder"):
            saved, state, raw = copy.deepcopy(original)
            if field == "lag": saved["lag_features"][0, 0, 2] += .1
            elif field == "current": saved["raw_last"][0, 0] += .1
            elif field == "decoder": state["D"][0, 0] += .1
            else: state["E"][0, 0] += .1
            with self.subTest(field=field), self.assertRaises(ValueError): audit.replay(saved, state, raw, audit.LAGS["L7"])

    def test_state_hash_matches_sorted_torch_serialization_and_not_file_SHA(self):
        _, state, _ = fixture((0,))
        from native_head_v1 import state_hash
        self.assertEqual(audit.state_hash(state), state_hash({k: torch.from_numpy(v) for k, v in state.items()}))
        self.assertNotEqual(audit.state_hash(state), hashlib.sha256(state["E"].tobytes()).hexdigest())

    def test_600_log_counts_fixed_FIT_denominators_order_clip_decisions_and_tamper(self):
        _, _, fit = fixture((0,)); order = np.random.default_rng(92601).integers(0, 4, size=(600, 4))
        logs = [{"stage": "TRAINING_UPDATE", "update": i + 1, "loss": .1, "E_gradient_norm_before_clip": 0., "D_gradient_norm_before_clip": 2., "global_gradient_norm_before_clip": 2., "clipped": True} for i in range(600)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "UPDATES.jsonl"; path.write_text("\n".join(json.dumps(r) for r in logs), encoding="utf-8")
            row = {"updates_log": audit.file_record(path), "training_updates_completed": 600, "learn_E": False, "global_clipping_fraction": 1., "FIT_channel_observed_counts": fit["observed"].sum((0, 1)).tolist(), "seed": 92601, "order_sha256": hashlib.sha256(order.tobytes()).hexdigest()}
            self.assertEqual(audit.check_logs(row, fit), 600)
            for key in ("order_sha256", "FIT_channel_observed_counts"):
                bad = copy.deepcopy(row); bad[key] = "wrong" if key.endswith("sha256") else [1, 2, 3]
                with self.subTest(key=key), self.assertRaises(ValueError): audit.check_logs(bad, fit)

    def test_selected_E_geometry_fixed_equal_and_rank_diagnostic_scope(self):
        saved, state, _ = fixture((0,)); initial = copy.deepcopy(state); selected = copy.deepcopy(state)
        first = audit.selected_geometry(initial, selected, saved["latent_history"])
        self.assertTrue(first["E_same_bytes"]); self.assertEqual(first["CP0_to_selected"]["D"]["delta_L2"], 0.)
        selected["E"][:, 1] = selected["E"][:, 0]
        result = audit.selected_geometry(initial, selected, saved["latent_history"])
        self.assertEqual(result["E_numeric_rank"], 1); self.assertFalse(result["E_same_bytes"])
        self.assertEqual(result["E_numeric_rank_relative_cutoff"], 1e-8); json.dumps(result, allow_nan=False)

    def test_actual_source_contract_and_closed_training_gate(self):
        controller = (audit.HERE / "run_linear_e_adaptation_v1.py").read_text(encoding="utf-8")
        pure = (audit.HERE / "linear_e_adaptation_v1.py").read_text(encoding="utf-8")
        self.assertTrue(audit.source_contract(controller, pure)["fixed_FIT_denominators_and_N_over_B"])
        with self.assertRaises(ValueError): audit.source_contract(controller.replace("lr=.001", "lr=.01"), pure)
        with patch.object(audit, "TRAIN_SHA", None), patch.object(audit, "file_record", side_effect=AssertionError("closed gate read")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.metadata("missing", lambda: None)


if __name__ == "__main__": unittest.main()
