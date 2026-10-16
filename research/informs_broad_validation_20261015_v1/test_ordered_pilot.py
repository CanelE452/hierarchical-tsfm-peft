"""Ordered-prefix protocol tests on tiny CPU arrays and a fake row F."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def toy():
    rng = np.random.default_rng(124)
    x = np.cumsum(rng.normal(size=(240, 4)), axis=0).astype(np.float32)
    return dict(name="toy", physical=x, observed=np.ones_like(x, bool), columns=list("abcd"),
                L=64, H=4, data_sha256="toy-ordered-data")


class FakeF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.streams = self.physical_streams = self.row_calls = self.physical_calls = 0
        self.initial_hash = "toy-row-independent-F"
        self.device = "cpu"
        self.widths = []
    def forward(self, x):
        self.widths.append(x.shape[2])
        self.streams += x.shape[0] * x.shape[2]
        self.physical_streams = self.streams
        self.row_calls += 1; self.physical_calls += 1
        return x[:, -1:].expand(-1, 4, -1) + .1


def candidate_caches():
    import data_contract, fit_base
    task = toy()
    mean, std, _ = data_contract.fit_statistics(task, 144)
    origins = data_contract.origins((168, 192), 64, 4, 4)
    targets, masks = zip(*(data_contract.target_window(task, int(origin), mean, std) for origin in origins))
    target, mask = np.stack(targets), np.stack(masks)
    result = []
    for family in ("B0", "B2"):
        for k in (1, 2, 4):
            result.append(dict(score=dict(mse=1.),
                metadata=dict(run_id=f"primary_toy_E1_{family}_k{k}_s92601", family=family, fold="primary", split="V_select",
                              data_sha256="toy-ordered-data", scaler_sha=fit_base.scaler_fingerprint(mean, std),
                              model_sha=f"{family}{k}", source_sha="toy-candidate-source", cohort="same-V"),
                origins=origins, channel_ids=np.array(list("abcd")),
                observed=mask, target_normalized=target, pred_normalized=target + 1))
    return result


class OrderedTests(unittest.TestCase):
    def setUp(self):
        path = HERE / "ordered_pilot.py"
        if not path.is_file():
            self.fail("Required ordered pilot protocol is not implemented")
        spec = importlib.util.spec_from_file_location("ordered_pilot", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory(prefix="informs_ordered_")
        self.root = Path(self.temp.name)
        def cleanup():
            assert self.root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()) and self.root.name.startswith("informs_ordered_")
            self.temp.cleanup()
        self.addCleanup(cleanup)
        torch.set_num_threads(2)

    def test_V_family_tie_B0_and_lock_is_immutable(self):
        lock = self.module.lock_family(toy(), candidate_caches(), self.root / "public")
        self.assertEqual(lock["chosen_family"], "B0")
        self.assertEqual(lock["k_grid"], [1, 2, 4])
        same = self.module.lock_family(toy(), candidate_caches(), self.root / "public")
        self.assertEqual(lock, same)
        changed = candidate_caches()
        changed[-1]["score"]["mse"] = .5
        with self.assertRaises(ValueError):
            self.module.lock_family(toy(), changed, self.root / "public")
        missing = candidate_caches()[:-1]
        with self.assertRaises(ValueError):
            self.module.lock_family(toy(), missing, self.root / "other")

    def test_lower_mean_selects_B2_and_late_lock_is_rejected(self):
        caches = candidate_caches()
        for cache in caches:
            if cache["metadata"]["family"] == "B2":
                cache["pred_normalized"] = cache["target_normalized"] + .5
                cache["score"]["mse"] = .25
        lock = self.module.lock_family(toy(), caches, self.root / "public")
        self.assertEqual(lock["chosen_family"], "B2")
        late = self.root / "late" / "runs" / "primary_toy_E4_ORDERED_MULTI_K_k4_s92601"
        late.mkdir(parents=True)
        with self.assertRaises(ValueError):
            self.module.lock_family(toy(), caches, self.root / "late")

    def test_shared_and_max_only_start_same_full_PCA_and_equal_total_steps(self):
        lock = self.module.lock_family(toy(), candidate_caches(), self.root / "public")
        specs = self.module.make_specs(toy(), lock, epochs=[.25], device="cpu")
        self.assertEqual(specs["shared"]["k_grid"], [1, 2, 4])
        self.assertEqual(specs["max_only"]["k_grid"], [4])
        a, na = self.module.fit_base.create_model(specs["shared"], toy(), 120, FakeF())
        b, nb = self.module.fit_base.create_model(specs["max_only"], toy(), 120, FakeF())
        self.assertEqual(self.module.fit_base.networks.module_state_hash(a), self.module.fit_base.networks.module_state_hash(b))
        np.testing.assert_array_equal(na["basis"], nb["basis"])
        self.assertEqual(na["basis_initialization"], "A_fitted_centered_PCA")
        f1, f2 = FakeF(), FakeF()
        r1 = self.module.fit_base.fit_base(specs["shared"], toy(), self.root / "private", self.root / "public", f1, None)
        r2 = self.module.fit_base.fit_base(specs["max_only"], toy(), self.root / "private", self.root / "public", f2, None)
        self.assertEqual(r1["search_updates"], r2["search_updates"])
        self.assertEqual(r1["search_exposures"], r2["search_exposures"])
        for receipt in (r1, r2):
            saved = torch.load(self.root / "private" / receipt["artifact_cache_relative_path"], weights_only=False)
            summary = saved["search_summary"]
            expected_rows = sum(int(k) * exposures for k, exposures in summary["k_exposures"].items())
            self.assertEqual(summary["F_training_logical_streams"], expected_rows)

    def test_prefix_D_coordinates_and_actual_F_width(self):
        f = FakeF()
        model, _ = self.module.fit_base.create_model(dict(family="B0", latent_k=4, k_grid=[1, 2, 4], device="cpu"), toy(), 144, f)
        with torch.no_grad():
            model.D.copy_(torch.arange(7 * 4 * 4).reshape(7, 4, 4).float() / 100)
        x = torch.randn(2, 64, 4)
        actual = self.module.prefix_prediction(model, x, f, 2)
        z = model.encode(x, 2)
        future = z[:, -1:].expand(-1, 4, -1) + .1
        joined = torch.cat((z, future), 1)
        expected = x[:, -1:].expand(-1, 4, -1).clone()
        for lag_index, lag in enumerate((0, 1, 2, 4, 8, 16, 32)):
            delta = joined[:, 64-lag:64-lag+4] - z[:, -1-lag, None]
            expected += delta @ model.D[lag_index, :2]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(f.widths, [2])
        self.assertEqual(f.streams, 4)

    def test_orchestration_produces_audit_ready_shared_posthoc_nested_and_actions(self):
        f = FakeF()
        result = self.module.run_ordered_pilot(toy(), candidate_caches(), self.root / "private", self.root / "public", f, None,
                                               epochs=[.25], device="cpu")
        self.assertEqual(result["status"], "COMPLETE_VALID")
        self.assertEqual(len(result["prefix_predictions"]), 10)
        self.assertTrue(result["registry_actions"])
        registry = [dict(run_id=alias, execution_status="COMPLETE_VALID") for alias in result["prefix_predictions"]]
        expected_ids = list(result["prefix_predictions"])
        import audit_predictions, csv, gzip
        for cache in result["prefix_predictions"].values():
            self.assertEqual(cache["cost_record"]["actual_F_streams"], len(cache["origins"]) * cache["metadata"]["inference_latent_k"])
            with gzip.open(self.root / "public" / cache["public_stats_relative_path"], "rt", encoding="utf-8") as handle:
                stats = list(csv.DictReader(handle))
            audit = audit_predictions.audit_prediction_bundle(cache["manifest"], cache["reference_contract"], cache["score"], stats,
                                        registry, expected_ids, [cache["cost_record"]], self.root / "private")
            self.assertEqual(audit["status"], "PASS")
        nested = [cache for cache in result["prefix_predictions"].values() if cache["metadata"]["variant"] == "nested_pca"]
        self.assertEqual(len({cache["metadata"]["model_sha"] for cache in nested}), 1)
        shared = [cache for cache in result["prefix_predictions"].values() if cache["metadata"]["variant"] == "shared"]
        self.assertEqual(len({cache["metadata"]["parent_artifact_sha256"] for cache in shared}), 1)
        self.assertFalse(result["T_final_accessed"])
        before = f.streams
        repeated = self.module.run_ordered_pilot(toy(), candidate_caches(), self.root / "private", self.root / "public", f, None,
                                                  epochs=[.25], device="cpu")
        self.assertEqual(repeated["status"], "COMPLETE_VALID")
        self.assertEqual(f.streams, before)
        existing_nested = nested[0]["metadata"]
        f.initial_hash = "different-foundation"
        with self.assertRaises(ValueError):
            self.module.predict_prefix(existing_nested["run_id"], toy(), existing_nested["inference_latent_k"],
                                       self.root / "private", self.root / "public", f, variant="nested_pca")


if __name__ == "__main__":
    unittest.main()
