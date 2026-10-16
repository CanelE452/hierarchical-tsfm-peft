"""CPU synthetic metric/audit tests; never open research data, models or TEST."""
import copy
import hashlib
import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def load_module(name, test):
    path = HERE / f"{name}.py"
    if not path.exists():
        test.fail(f"Required {name} behavior is not implemented")
    spec = importlib.util.spec_from_file_location(f"informs_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fp(value):
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def fixture():
    p = np.ones((2, 12, 3), dtype=np.float32)
    p[0, 0, 1] = 4
    y = np.zeros_like(p)
    mask = np.zeros_like(p, dtype=bool)
    mask[:, :, 0] = True
    mask[0, 0, 1] = True
    return dict(pred_normalized=p, target_normalized=y, observed=mask,
                origins=np.array([512, 524], np.int64), channel_ids=np.array(["c0", "c1", "empty"]))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.e = load_module("evaluation", self)

    def test_macro_weights_channels_not_observation_counts(self):
        a = fixture()
        score, rows = self.e.error_statistics(a["pred_normalized"], a["target_normalized"], a["observed"], a["origins"], "toy", "V_select", "fixed", a["channel_ids"])
        self.assertEqual(score["mse"], 8.5)
        self.assertEqual(score["mae"], 2.5)
        self.assertEqual(score["n_observed"], 25)
        self.assertEqual(score["active_channels"], 2)
        self.assertEqual(len(rows), 18)

    def test_zero_bands_and_missing_targets_do_not_disappear_or_propagate_nan(self):
        a = fixture()
        a["target_normalized"][~a["observed"]] = np.nan
        score, rows = self.e.error_statistics(a["pred_normalized"], a["target_normalized"], a["observed"], a["origins"], "toy", "V", "fixed", a["channel_ids"])
        self.assertEqual(score["mse"], 8.5)
        zero = [r for r in rows if r["channel"] == "empty"]
        self.assertEqual(len(zero), 6)
        self.assertTrue(all(r["count"] == r["sse"] == r["sae"] == 0 for r in zero))

    def test_horizon_bands_follow_actual_horizon_and_cover_every_pair(self):
        for horizon, expected in ((12, [(0, 3), (3, 6), (6, 12)]), (28, [(0, 7), (7, 14), (14, 28)]), (48, [(0, 12), (12, 24), (24, 48)])):
            p = np.ones((1, horizon, 2))
            _, rows = self.e.error_statistics(p, p, np.ones_like(p, bool), [512], "toy", "T", "c", ["a", "b"])
            for channel in ("a", "b"):
                self.assertEqual([(r["horizon_start"], r["horizon_end"]) for r in rows if r["channel"] == channel], expected)

    def test_nonfinite_observed_target_is_rejected(self):
        a = fixture()
        a["target_normalized"][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.e.error_statistics(a["pred_normalized"], a["target_normalized"], a["observed"], a["origins"], "x", "V", "c", a["channel_ids"])

    def test_alpha_matches_dense_grid_for_masked_channel_macro_objective(self):
        rng = np.random.default_rng(810)
        g = rng.normal(size=(3, 8, 3))
        b = rng.normal(size=g.shape)
        y = 0.3 * g + 0.7 * b + rng.normal(scale=0.1, size=g.shape)
        mask = rng.random(size=g.shape) > 0.4
        mask[:, :, 2] = False
        mask[0, 0, 2] = True
        fit = self.e.fit_global_alpha(g, b, y, mask)
        def objective(alpha):
            error = g + alpha * (b - g) - y
            return np.mean([np.square(error[:, :, c][mask[:, :, c]]).mean() for c in range(3)])
        grid = np.linspace(0, 1, 10001)
        grid_values = [objective(alpha) for alpha in grid]
        self.assertLessEqual(objective(fit["alpha"]), min(grid_values) + 1e-12)
        self.assertLessEqual(abs(fit["alpha"] - grid[int(np.argmin(grid_values))]), 1e-4)

    def test_equal_parent_predictions_choose_alpha_zero(self):
        g = np.ones((2, 4, 2))
        fit = self.e.fit_global_alpha(g, g, np.zeros_like(g), np.ones_like(g, bool))
        self.assertEqual(fit["alpha"], 0)
        self.assertEqual(fit["denominator"], 0)


class PredictionAuditTests(unittest.TestCase):
    def setUp(self):
        self.a = load_module("audit_predictions", self)
        self.e = load_module("evaluation", self)
        self.temp = tempfile.TemporaryDirectory(prefix="informs_evaluation_")
        self.private = pathlib.Path(self.temp.name)
        def cleanup():
            target = self.private.resolve()
            assert target.is_relative_to(pathlib.Path(tempfile.gettempdir()).resolve()) and target.name.startswith("informs_evaluation_")
            self.temp.cleanup()
        self.addCleanup(cleanup)
        self.arrays = fixture()
        self.path = self.private / "toy.npz"
        np.savez(self.path, **self.arrays)
        identity = dict(run_id="toy", split="V_select", cohort="fixed", source_sha="source", config_sha="config", data_sha="data", model_sha="model", scaler_sha="scaler")
        self.reference = dict(identity, arrays={k: fp(v) for k, v in self.arrays.items() if k != "pred_normalized"})
        self.manifest = dict(identity, cache_relative_path="toy.npz", archive_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(), arrays={k: fp(v) for k, v in self.arrays.items()})
        self.score, self.stats = self.e.error_statistics(self.arrays["pred_normalized"], self.arrays["target_normalized"], self.arrays["observed"], self.arrays["origins"], "toy", "V_select", "fixed", self.arrays["channel_ids"])
        self.registry = [{"run_id": "toy", "execution_status": "COMPLETE_VALID"}, {"run_id": "failed", "execution_status": "FAILED", "reason": "synthetic route failure"}]
        self.cost = [{"run_id": "toy", "batch": 2, "latent_k": 2, "actual_F_streams": 4, "expected_F_streams": 4}]

    def audit(self):
        return self.a.audit_prediction_bundle(self.manifest, self.reference, self.score, self.stats, self.registry, ["toy", "failed"], self.cost, self.private)

    def rewrite(self):
        np.savez(self.path, **self.arrays)
        self.manifest["archive_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.manifest["arrays"] = {k: fp(v) for k, v in self.arrays.items()}

    def test_valid_bundle_independently_rescores_and_reaggregates(self):
        result = self.audit()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["mse"], 8.5)
        self.assertTrue(result["prediction_rescoring"])
        self.assertTrue(result["public_stats_reaggregation"])
        self.assertFalse(result["checkpoint_replay"])

    def test_target_one_horizon_shift_is_rejected_even_with_new_archive_hash(self):
        self.arrays["target_normalized"] = np.arange(72, dtype=np.float32).reshape(2, 12, 3)
        self.reference["arrays"]["target_normalized"] = fp(self.arrays["target_normalized"])
        self.score, self.stats = self.e.error_statistics(self.arrays["pred_normalized"], self.arrays["target_normalized"], self.arrays["observed"], self.arrays["origins"], "toy", "V_select", "fixed", self.arrays["channel_ids"])
        self.arrays["target_normalized"] = np.roll(self.arrays["target_normalized"], 1, axis=1)
        self.rewrite()
        with self.assertRaises(ValueError): self.audit()

    def test_channel_swap_is_rejected_even_with_new_archive_hash(self):
        self.arrays["channel_ids"] = self.arrays["channel_ids"][[1, 0, 2]]
        self.rewrite()
        with self.assertRaises(ValueError): self.audit()

    def test_wrong_observation_mask_is_rejected_even_with_new_archive_hash(self):
        self.arrays["observed"][0, 1, 1] = True
        self.rewrite()
        with self.assertRaises(ValueError): self.audit()

    def test_archive_sha_tampering_is_rejected(self):
        self.manifest["archive_sha256"] = "0" * 64
        with self.assertRaises(ValueError): self.audit()

    def test_failed_route_missing_from_registry_is_rejected(self):
        self.registry.pop()
        with self.assertRaises(ValueError): self.audit()

    def test_Kmax_execution_cannot_be_labeled_small_K(self):
        self.cost[0]["actual_F_streams"] = 6
        with self.assertRaises(ValueError): self.audit()

    def test_score_number_tampering_is_rejected(self):
        self.score["mse"] += 0.1
        with self.assertRaises(ValueError): self.audit()

    def test_public_duplicate_band_is_rejected(self):
        self.stats.append(copy.deepcopy(self.stats[0]))
        with self.assertRaises(ValueError): self.audit()

    def test_public_zero_count_coverage_cannot_be_omitted(self):
        self.stats = [row for row in self.stats if row["channel"] != "empty"]
        with self.assertRaises(ValueError): self.audit()

    def test_zero_count_sums_must_be_exactly_zero(self):
        row = next(row for row in self.stats if row["count"] == 0)
        row["sse"] = 1e-10
        with self.assertRaises(ValueError): self.audit()


if __name__ == "__main__":
    unittest.main()
