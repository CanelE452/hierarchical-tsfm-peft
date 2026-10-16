"""A-only pattern and independently weighted group-error CPU checks."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import data_contract as data
import prediction_cache


def task_fixture():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(400, 4))
    x[4:, 1] = x[:-4, 0]
    x[:, 2] = 3.
    mask = np.ones_like(x, bool)
    mask[20:24, 0] = False
    mask[:, 3] = False
    return dict(name="toy", physical=x, observed=mask, columns=["source", "target", "constant", "missing"],
                period=4, L=64, H=4, data_sha256="toy_source", metadata={},
                times=np.arange(400, dtype=np.int64), interval_seconds=1)


class ChannelPatternTests(unittest.TestCase):
    def setUp(self):
        import channel_patterns
        self.module = channel_patterns
        self.tmp = tempfile.TemporaryDirectory(prefix="informs_patterns_")
        self.addCleanup(self.tmp.cleanup)
        self.public, self.cache = Path(self.tmp.name) / "public", Path(self.tmp.name) / "private"
        self.public.mkdir(); self.cache.mkdir()

    def test_known_direction_and_future_suffix_invariance(self):
        task = task_fixture()
        rows, thresholds = self.module.fit_task_patterns(task, 200)
        target = rows[1]
        self.assertEqual(target["delayed_source_channel"], "source")
        self.assertEqual(target["delayed_lag"], 4)
        self.assertAlmostEqual(target["delayed_rho"], 1., places=12)
        changed = copy.deepcopy(task)
        changed["physical"][200:] = np.nan
        changed["observed"][200:] = False
        other, other_thresholds = self.module.fit_task_patterns(changed, 200)
        self.assertEqual(rows, other)
        self.assertEqual(thresholds, other_thresholds)

    def test_constant_missing_exact_zeros_and_gap_difference(self):
        task = task_fixture()
        task["physical"][:200, 0] = np.arange(200)
        task["physical"][4, 0] = 1e-12
        rows, _ = self.module.fit_task_patterns(task, 200)
        self.assertIsNone(rows[2]["variability"])
        self.assertEqual(rows[2]["variability_group"], "NA")
        self.assertIsNone(rows[2]["seasonality"])
        self.assertIsNone(rows[2]["delayed_crosschannel"])
        self.assertIsNone(rows[3]["intermittency"])
        valid = task["observed"][:200, 0]
        pair = valid[1:] & valid[:-1]
        expected = np.std(np.diff(task["physical"][:200, 0])[pair]) / np.std(task["physical"][:200, 0][valid])
        self.assertEqual(rows[0]["variability_pairs"], int(pair.sum()))
        self.assertAlmostEqual(rows[0]["variability"], expected)
        self.assertAlmostEqual(rows[0]["intermittency"], 1 / int(valid.sum()))

    def test_pairwise_mask_means_large_offset_and_minimum_count(self):
        rng = np.random.default_rng(4)
        x = 1e12 + rng.normal(size=(90, 3))
        mask = rng.random(x.shape) > .2
        rho, counts = self.module.delayed_correlation_matrix(x, mask, 4, min_pairs=30)
        for target in range(3):
            for source in range(3):
                good = mask[4:, target] & mask[:-4, source]
                self.assertEqual(counts[target, source], int(good.sum()))
                # The stable reference removes an arbitrary shared offset first.
                a = x[4:, target][good] - 1e12
                b = x[:-4, source][good] - 1e12
                self.assertAlmostEqual(rho[target, source], np.corrcoef(a, b)[0, 1], places=10)
        invalid = x.copy(); invalid[~mask] = np.nan
        again, _ = self.module.delayed_correlation_matrix(invalid, mask, 4, min_pairs=30)
        np.testing.assert_allclose(rho, again, atol=0, rtol=0, equal_nan=True)
        a = rng.normal(size=(31, 2)); m = np.ones_like(a, bool)
        r30, _ = self.module.delayed_correlation_matrix(a, m, 1, min_pairs=30)
        r29, _ = self.module.delayed_correlation_matrix(a[:-1], m[:-1], 1, min_pairs=30)
        self.assertTrue(np.isfinite(r30).all())
        self.assertTrue(np.isnan(r29).all())

    def test_median_boundary_all_equal_and_na_are_explicit(self):
        values = [0., 1., 2., None]
        groups, threshold = self.module.median_groups(values)
        self.assertEqual(threshold, 1.)
        self.assertEqual(groups, ["low", "low", "high", "NA"])
        self.assertEqual(self.module.median_groups([0., 0., None]), (["low", "low", "NA"], 0.))
        self.assertEqual(self.module.median_groups([None]), (["NA"], None))

    def test_pair_subset_constant_and_tie_direction_are_not_spurious(self):
        rng = np.random.default_rng(92)
        source = rng.normal(size=100)
        x = np.column_stack((source, source, np.r_[0., source[:-1]]))
        task = dict(name="toy", physical=x, observed=np.ones_like(x, bool), columns=["first", "second", "target"], period=4)
        rows, _ = self.module.fit_task_patterns(task, 100)
        self.assertEqual(rows[2]["delayed_source_channel"], "first")
        self.assertEqual(rows[2]["delayed_lag"], 1)
        x[:, 0] = 1e12 + .1
        x[0, 0] += 1.
        mask = np.ones_like(x, bool)
        # At lag1 the target side is constant although channel0 is not constant on all A.
        rho, _ = self.module.delayed_correlation_matrix(x, mask, 1)
        self.assertTrue(np.isnan(rho[0]).all())

    def test_group_macro_balances_channels_instead_of_pooling_cells(self):
        p = np.zeros((2, 2, 3), np.float32)
        y = np.zeros_like(p); mask = np.ones_like(p, bool)
        p[:, :, 0] = 1.; p[:, :, 1] = 3.; mask[0, 1:, 1] = False; mask[1, :, 1] = False
        mask[:, :, 2] = False
        totals = self.module.channel_error_totals(p, y, mask)
        score = self.module.aggregate_channel_group(totals, [0, 1, 2], 4)
        self.assertEqual(score["mse"], 5.)
        self.assertEqual(score["mae"], 2.)
        self.assertEqual(score["active_channels"], 2)
        self.assertEqual(score["zero_observed_channels"], 1)
        self.assertEqual(score["observed_target_cells"], 5)
        self.assertNotEqual(score["mse"], (4 + 9) / 5)

    def test_actual_toy_fit_and_v_evaluation_keep_all_groups(self):
        task = task_fixture()
        # Forecast fitting statistics require every channel to be observed on A.
        task["observed"][:, 3] = True
        rows = []
        for run_id, family, baseline in (("current", "CURRENT", "CURRENT"), ("seasonal", "SEASONAL_NAIVE", "SEASONAL")):
            prediction_cache.predict_parent(run_id, task, "V_select", self.cache, self.public,
                                            None, baseline=dict(kind=baseline, period=4))
            rows.append(dict(experiment_id=run_id, dataset="toy", family=family, latent_k=0, seed=1, fold="primary"))
        with (self.public / "EXPERIMENT_REGISTRY.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with patch.object(self.module.data, "load_task", return_value=task):
            snapshot = self.module.fit_patterns(self.public, self.cache, datasets=["toy"])
        result = self.module.evaluate_groups(self.public, self.cache, snapshot)
        with (Path(result) / "GROUP_ERRORS.csv").open() as stream:
            scores = list(csv.DictReader(stream))
        self.assertEqual(len(scores), 2 * 4 * 3)
        self.assertEqual({r["group"] for r in scores}, {"low", "high", "NA"})
        self.assertTrue((Path(result) / "PAIRED_GROUP_DIFFERENCES.csv").is_file())
        manifest = json.loads((Path(result) / "EVALUATION_MANIFEST.json").read_text())
        self.assertEqual(manifest["prediction_count"], 2)
        self.assertFalse(manifest["pattern_fit_used_future_targets"])
        with self.assertRaisesRegex(ValueError, "SELECTION_LOCK"):
            self.module.evaluate_groups(self.public, self.cache, snapshot, split="T_final")

    def test_learning_rate_identity_and_unknown_learning_recipe_are_not_paired(self):
        for name, family, lr in (("left", "B2", .001), ("same", "B0", .001), ("other", "B0", .0001)):
            directory = self.public / "runs" / name; directory.mkdir(parents=True)
            (directory / "SOURCE_CONTRACT.json").write_text(json.dumps({"config_sha": name + "_config", "spec": {"run_id": name, "family": family, "lr": lr}}))
        def identity(name, family):
            return self.module.training_identity(self.public, name, {"metadata": {}}, family)
        left, same, other = identity("left", "B2"), identity("same", "B0"), identity("other", "B0")
        self.assertTrue(self.module.learning_pair_compatible(left, same))
        self.assertFalse(self.module.learning_pair_compatible(left, other))
        unknown = identity("missing", "B0")
        self.assertFalse(self.module.learning_pair_compatible(left, unknown))
        baseline = identity("current", "CURRENT")
        self.assertTrue(self.module.learning_pair_compatible(left, baseline))
        self.assertFalse(self.module.learning_pair_compatible(unknown, baseline))
        self.assertEqual(left["fit_config_sha"], "left_config")

    def test_control_source_identity_preserves_recorded_lr_receipt_and_parent(self):
        recipes = {}
        for name, family, lr in (("g", "DIRECT_G", .001), ("lora", "LORA", .0001)):
            directory = self.public / "runs" / name; directory.mkdir(parents=True)
            source = directory / "CONTROL_SOURCE_CONTRACT.json"
            source.write_text(json.dumps(dict(config_sha=name + "_config", spec=dict(
                run_id=name, family=family, role="AplusM_control", lr=lr, epochs=[.25]))))
            receipt = directory / "CONTROL_FIT_RECEIPT.json"
            receipt.write_text(json.dumps(dict(status="COMPLETE_VALID", run_id=name,
                role="AplusM_control", selected_epoch=.25, epochs=[.25])))
            recipe = self.module.training_identity(self.public, name, {"metadata": {}}, family)
            self.assertEqual(recipe["learning_rate"], lr)
            self.assertTrue(recipe["lr_known"])
            self.assertEqual(recipe["fit_config_sha"], name + "_config")
            self.assertEqual(recipe["fit_epoch_endpoints"], "0.25")
            self.assertEqual(recipe["selected_epoch"], .25)
            self.assertEqual(recipe["input_files"], [
                dict(scope="public", file=source.relative_to(self.public).as_posix(), sha256=data.sha256_file(source)),
                dict(scope="public", file=receipt.relative_to(self.public).as_posix(), sha256=data.sha256_file(receipt))])
            recipes[name] = recipe
        parent = self.module.training_identity(self.public, "g_route", {"metadata": {"parent_run_id": "g"}}, "DIRECT_G")
        self.assertEqual(parent, recipes["g"])
        self.assertTrue(self.module.learning_pair_compatible(parent, recipes["g"]))
        self.assertFalse(self.module.learning_pair_compatible(parent, recipes["lora"]))

    def test_source_priority_and_invalid_control_learning_rate_stay_excluded(self):
        directory = self.public / "runs" / "g"; directory.mkdir(parents=True)
        source, control = directory / "SOURCE_CONTRACT.json", directory / "CONTROL_SOURCE_CONTRACT.json"
        control.write_text(json.dumps(dict(config_sha="control", spec=dict(run_id="g", lr=.0001))))
        source.write_text(json.dumps(dict(config_sha="base", spec=dict(run_id="g", lr=.001))))
        identity = self.module.training_identity(self.public, "g", {}, "DIRECT_G")
        self.assertEqual(identity["learning_rate"], .001)
        self.assertEqual(identity["fit_source_file"], "runs/g/SOURCE_CONTRACT.json")
        source.write_text(json.dumps(dict(config_sha="base", spec=dict(run_id="g", lr=None))))
        self.assertFalse(self.module.training_identity(self.public, "g", {}, "DIRECT_G")["lr_known"])
        source.unlink()
        fixed = self.module.training_identity(self.public, "current", {}, "CURRENT")
        for lr, config_sha in ((None, "c"), (0, "c"), (-.001, "c"), (True, "c"),
                               ("unknown", "c"), ("NaN", "c"), ("inf", "c"), (.001, "")):
            with self.subTest(lr=lr, config_sha=config_sha):
                control.write_text(json.dumps(dict(config_sha=config_sha, spec=dict(run_id="g", lr=lr))))
                unknown = self.module.training_identity(self.public, "g", {}, "DIRECT_G")
                self.assertFalse(unknown["lr_known"])
                self.assertFalse(self.module.learning_pair_compatible(unknown, fixed))


if __name__ == "__main__":
    unittest.main()
