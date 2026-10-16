"""Prediction contracts on synthetic CPU arrays; no real data or GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def toy():
    values = np.arange(720, dtype=np.float32).reshape(240, 3) / 100
    observed = np.ones_like(values, bool)
    observed[175, 1] = False
    values[~observed] = np.nan
    return dict(name="toy", physical=values, observed=observed, columns=["a", "b", "c"],
                L=64, H=4, period=12, data_sha256="toy-data")


class FakeFoundation(torch.nn.Module):
    def __init__(self, lie=False):
        super().__init__()
        self.streams = self.physical_streams = 0
        self.initial_hash = "toy-frozen-model"
        self.device = "cpu"
        self.lie = lie
    def forward(self, x):
        self.streams += x.shape[0] * (x.shape[2] + int(self.lie))
        self.physical_streams = self.streams
        return x[:, -1:].expand(-1, 4, -1) + .1


class PredictionCacheTests(unittest.TestCase):
    def setUp(self):
        path = HERE / "prediction_cache.py"
        if not path.is_file():
            self.fail("Required prediction cache is not implemented")
        spec = importlib.util.spec_from_file_location("prediction_cache", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory(prefix="informs_prediction_")
        self.root = Path(self.temp.name)
        def cleanup():
            assert self.root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
            assert self.root.name.startswith("informs_prediction_")
            self.temp.cleanup()
        self.addCleanup(cleanup)
        torch.set_num_threads(2)

    def predict(self, run="current", split="V_select", baseline="CURRENT", foundation=None, dense=False):
        return self.module.predict_parent(run, toy(), split, self.root / "private", self.root / "public", foundation,
                                          baseline=baseline, dense=dense)

    def test_current_reference_full_coverage_and_dense_no_public_statistics(self):
        result = self.predict()
        self.assertEqual(result["cost_record"]["actual_F_streams"], 0)
        self.assertEqual(result["observed"].dtype, np.bool_)
        np.testing.assert_array_equal(result["origins"], np.arange(168, 189, 4))
        self.assertEqual(result["metadata"]["parent_fit_end"], 144)
        self.assertEqual(result["reference_contract"]["arrays"]["observed"], self.module.evaluation.array_fingerprint(result["observed"]))
        dense = self.predict("dense_current", "M_cal", dense=True)
        self.assertIsNone(dense["score"])
        self.assertEqual(len(dense["origins"]), 21)
        self.assertFalse(list((self.root / "public" / "predictions" / "dense_current").rglob("*.gz")))

    def test_raw_counts_actual_rows_and_reuses_frozen_cache_without_new_forward(self):
        foundation = FakeFoundation()
        result = self.predict("raw", baseline="RAW", foundation=foundation)
        self.assertEqual(result["cost_record"]["actual_F_streams"], len(result["origins"]) * 3)
        before = foundation.streams
        reused = self.predict("raw", baseline="RAW", foundation=foundation)
        self.assertEqual(foundation.streams, before)
        self.assertTrue(reused["metadata"]["reused_prediction_cache"])
        np.testing.assert_array_equal(result["pred_normalized"], reused["pred_normalized"])

    def test_stream_count_lie_is_rejected(self):
        with self.assertRaises(ValueError):
            self.predict("lie", baseline="RAW", foundation=FakeFoundation(True))

    def test_T_target_is_not_opened_before_matching_selection_lock(self):
        with patch.object(self.module.data_contract, "target_window", side_effect=AssertionError("T target opened")):
            with self.assertRaises(ValueError):
                self.predict(split="T_final")
        v = self.predict()
        lock = dict(locked=True, allowed_run_ids=["current"], allowed_artifact_sha={"current": v["metadata"]["artifact_sha256"]})
        (self.root / "public" / "SELECTION_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
        t = self.predict(split="T_final")
        self.assertEqual(t["metadata"]["split"], "T_final")

    def test_public_stats_gzip_is_deterministic_header_and_all_bands(self):
        result = self.predict()
        path = self.root / "public" / result["public_stats_relative_path"]
        self.assertEqual(path.read_bytes()[4:8], b"\0\0\0\0")
        import csv, gzip
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(result["origins"]) * 3 * 3)

    def test_wrong_T_artifact_hash_is_rejected(self):
        self.root.joinpath("public").mkdir()
        lock = dict(locked=True, allowed_run_ids=["current"], allowed_artifact_sha={"current": "wrong"})
        (self.root / "public" / "SELECTION_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.predict(split="T_final")

    def save_parent(self, spec, foundation, corrupt=False):
        fit = self.module.fit_base
        model, normalization = fit.create_model(spec, toy(), 120, foundation)
        if spec["family"] == "DIRECT_G":
            with torch.no_grad():
                model.output.bias.fill_(1.)
        state = fit.training.cpu_state(model)
        if corrupt:
            state["unexpected_weight"] = torch.ones(2)
        if callable(getattr(model, "close", None)):
            model.close()
        bundle = dict(model=state, normalization=normalization, spec=spec, model_state_sha="toy-parent-state",
                      contract=dict(data_sha="toy-data", model_sha=fit._model_sha(foundation), source_sha="toy-training-source",
                                    scaler_sha=fit.scaler_fingerprint(normalization["mean"], normalization["std"])))
        path = self.root / "private" / "runs" / spec["run_id"] / "selected_bundle.pt"
        path.parent.mkdir(parents=True)
        torch.save(bundle, path)
        receipt = dict(status="COMPLETE_VALID", run_id=spec["run_id"], artifact_cache_relative_path=path.relative_to(self.root / "private").as_posix(),
                       artifact_sha256=fit.data_contract.sha256_file(path))
        return receipt, normalization

    def test_saved_parent_own_scale_is_converted_to_common_A60_and_MV_same_weights(self):
        spec = dict(run_id="old_scale", family="DIRECT_G", seed=3, device="cpu", microbatch=4)
        receipt, own = self.save_parent(spec, None)
        v = self.module.predict_parent("old_scale", toy(), "V_select", self.root / "private", self.root / "public", None, receipt=receipt)
        mean, std, _ = self.module.data_contract.fit_statistics(toy(), 144)
        task = toy()
        current = np.empty((len(v["origins"]), 3), np.float64)
        for row, origin in enumerate(v["origins"]):
            for channel in range(3):
                seen = np.flatnonzero(task["observed"][origin - 64:origin, channel])
                current[row, channel] = task["physical"][origin - 64 + seen[-1], channel] if len(seen) else own["mean"][channel]
        expected = ((current[:, None] + own["std"].astype(np.float64) - mean) / std).astype(np.float32)
        np.testing.assert_allclose(v["pred_normalized"], np.repeat(expected, 4, axis=1), rtol=1e-6, atol=1e-6)
        m = self.module.predict_parent("old_scale", toy(), "M_cal", self.root / "private", self.root / "public", None, receipt=receipt, dense=True)
        self.assertEqual(m["metadata"]["model_sha"], v["metadata"]["model_sha"])
        self.assertEqual(m["metadata"]["scaler_sha"], v["metadata"]["scaler_sha"])

    def test_generated_raw_cache_passes_independent_auditor(self):
        result = self.predict("raw_audited", baseline="RAW", foundation=FakeFoundation())
        import audit_predictions, csv, gzip
        with gzip.open(self.root / "public" / result["public_stats_relative_path"], "rt", encoding="utf-8") as handle:
            statistics = list(csv.DictReader(handle))
        audit = audit_predictions.audit_prediction_bundle(result["manifest"], result["reference_contract"], result["score"], statistics,
                [dict(run_id="raw_audited", execution_status="COMPLETE_VALID")], ["raw_audited"], [result["cost_record"]], self.root / "private")
        self.assertEqual(audit["status"], "PASS")

    def test_lora_load_error_still_removes_hooks_and_restores_foundation(self):
        networks = self.module.fit_base.networks
        class QVFoundation(FakeFoundation):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.ModuleDict({"q": torch.nn.Linear(64, 4), "v": torch.nn.Linear(64, 4)}).requires_grad_(False)
                self.initial_hash = networks.module_state_hash(self.model)
            def assert_frozen_unchanged(self):
                if networks.module_state_hash(self.model) != self.initial_hash:
                    raise AssertionError("Base changed")
        foundation = QVFoundation()
        spec = dict(run_id="broken_lora", family="LORA", seed=3, device="cpu")
        receipt, _ = self.save_parent(spec, foundation, corrupt=True)
        with self.assertRaises(RuntimeError):
            self.module.predict_parent("broken_lora", toy(), "V_select", self.root / "private", self.root / "public", foundation, receipt=receipt)
        foundation.assert_frozen_unchanged()
        self.assertTrue(all(not layer._forward_hooks for layer in foundation.model.values()))

    def test_AplusM_role_separates_train_end_from_normalization_and_refuses_M(self):
        spec = dict(run_id="am_control", family="DIRECT_G", seed=3, device="cpu", role="AplusM_control")
        receipt, _ = self.save_parent(spec, None)
        path = self.root / "private" / receipt["artifact_cache_relative_path"]
        saved = torch.load(path, weights_only=False)
        saved["normalization"]["train_end"] = 168
        torch.save(saved, path)
        receipt["artifact_sha256"] = self.module.data_contract.sha256_file(path)
        with self.assertRaises(ValueError):
            self.module.predict_parent("am_control", toy(), "M_cal", self.root / "private", self.root / "public", None,
                                       receipt=receipt, role="AplusM_control", dense=True)
        result = self.module.predict_parent("am_control", toy(), "V_select", self.root / "private", self.root / "public", None,
                                            receipt=receipt, role="AplusM_control")
        self.assertEqual(result["metadata"]["parent_fit_end"], 168)
        self.assertEqual(result["metadata"]["parent_normalization_fit_end"], 120)
        self.assertEqual(result["metadata"]["role"], "AplusM_control")
        with self.assertRaises(ValueError):
            self.module.predict_parent("am_control", toy(), "V_select", self.root / "private", self.root / "public", None,
                                       receipt=receipt)

    def save_ridge(self, bad_data=False):
        import ridge_baseline
        mean, std, _ = self.module.data_contract.fit_statistics(toy(), 144)
        coefficients = np.zeros((3, 33, 4), np.float64)
        coefficients[:, -1] = 1.5
        native_scaler = ridge_baseline.scaler_fingerprint(mean, std)
        contract = dict(data_sha="wrong" if bad_data else "toy-data", source_sha="toy-ridge-source", config_sha="toy-ridge-config",
                        model_sha="NO_FOUNDATION", scaler_sha=native_scaler, feature_sha="toy-features", L=64, H=4)
        artifact = dict(coefficients=coefficients, mean=mean, std=std, block_edges=ridge_baseline.block_edges(64),
                        alpha=np.asarray(1.), canonical_A60_end=np.asarray(144), actual_fit_end=np.asarray(144),
                        source_sha=np.asarray(contract["source_sha"]), data_sha=np.asarray(contract["data_sha"]),
                        feature_sha=np.asarray(contract["feature_sha"]), scaler_sha=np.asarray(native_scaler),
                        contract_json=np.asarray(json.dumps(contract)))
        path = self.root / "private" / "ridge" / "ridge_toy" / "artifact.npz"
        path.parent.mkdir(parents=True)
        np.savez_compressed(path, **artifact)
        receipt = dict(status="COMPLETE_VALID", run_id="ridge_toy", family="RAW_RIDGE", fold="primary", role="base_parent",
                       config=dict(family="RAW_RIDGE", L=64, H=4), data_sha=contract["data_sha"], source_sha=contract["source_sha"],
                       scaler_sha=native_scaler, model_sha="NO_FOUNDATION", fit_end=144,
                       artifact_cache_relative_path=path.relative_to(self.root / "private").as_posix(),
                       artifact_sha256=self.module.data_contract.sha256_file(path))
        folder = self.root / "public" / "runs" / "ridge_toy"
        folder.mkdir(parents=True)
        (folder / "RIDGE_FIT_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
        return artifact, receipt

    def test_raw_ridge_npz_parent_no_torch_load_zero_F_and_common_contract(self):
        import ridge_baseline
        artifact, _ = self.save_ridge()
        foundation = FakeFoundation()
        with patch.object(torch, "load", side_effect=AssertionError("Ridge must use NPZ API")):
            result = self.module.predict_parent("ridge_toy", toy(), "V_select", self.root / "private", self.root / "public", foundation)
        self.assertEqual(foundation.streams, 0)
        self.assertEqual(result["cost_record"]["actual_F_streams"], 0)
        self.assertEqual(result["metadata"]["family"], "RAW_RIDGE")
        histories = np.stack([self.module.data_contract.history_window(toy(), int(origin), artifact["mean"], artifact["std"])[0] for origin in result["origins"]])
        np.testing.assert_array_equal(result["pred_normalized"], ridge_baseline.predict_normalized(artifact, histories).astype(np.float32))
        mean, std, _ = self.module.data_contract.fit_statistics(toy(), 144)
        self.assertEqual(result["metadata"]["scaler_sha"], self.module.fit_base.scaler_fingerprint(mean, std))
        m = self.module.predict_parent("ridge_toy", toy(), "M_cal", self.root / "private", self.root / "public", None, dense=True, baseline="RAW_RIDGE")
        self.assertEqual(m["metadata"]["model_sha"], result["metadata"]["model_sha"])
        import audit_predictions, csv, gzip
        with gzip.open(self.root / "public" / result["public_stats_relative_path"], "rt", encoding="utf-8") as handle:
            stats = list(csv.DictReader(handle))
        audited = audit_predictions.audit_prediction_bundle(result["manifest"], result["reference_contract"], result["score"], stats,
                    [dict(run_id="ridge_toy", execution_status="COMPLETE_VALID")], ["ridge_toy"], [result["cost_record"]], self.root / "private")
        self.assertEqual(audited["status"], "PASS")
        with patch.object(self.module.data_contract, "target_window", side_effect=AssertionError("Locked T opened")):
            with self.assertRaises(ValueError):
                self.module.predict_parent("ridge_toy", toy(), "T_final", self.root / "private", self.root / "public", None)
        lock = dict(locked=True, allowed_run_ids=["ridge_toy"], allowed_artifact_sha={"ridge_toy": result["metadata"]["artifact_sha256"]})
        (self.root / "public" / "SELECTION_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
        final = self.module.predict_parent("ridge_toy", toy(), "T_final", self.root / "private", self.root / "public", None)
        self.assertEqual(final["cost_record"]["actual_F_streams"], 0)
        self.assertEqual(final["metadata"]["model_sha"], result["metadata"]["model_sha"])

    def test_raw_ridge_wrong_data_contract_is_rejected(self):
        _, receipt = self.save_ridge(bad_data=True)
        with self.assertRaises(ValueError):
            self.module.predict_parent("ridge_toy", toy(), "V_select", self.root / "private", self.root / "public", None, receipt=receipt)

    def test_forecast_deadline_stops_new_work_but_allows_unchanged_complete_cache(self):
        with self.assertRaises(TimeoutError):
            self.module.predict_parent("expired", toy(), "M_cal", self.root / "private", self.root / "public", FakeFoundation(),
                                       baseline="RAW", dense=True, deadline=time.time() - 1)
        self.assertFalse((self.root / "private" / "predictions" / "expired" / "M_cal_dense.npz").exists())
        first = self.predict()
        reused = self.module.predict_parent("current", toy(), "V_select", self.root / "private", self.root / "public", None,
                                             baseline="CURRENT", deadline=time.time() - 1)
        self.assertTrue(reused["metadata"]["reused_prediction_cache"])
        self.assertEqual(first["manifest"], reused["manifest"])


if __name__ == "__main__":
    unittest.main()
