"""Synthetic variable-axis fixed-PCA verification guards, no real arrays/models."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
HELPER = HERE / "verify_fixed_pca_outputs.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedVerifierTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(HELPER.is_file(), "Fixed-PCA verifier is not implemented")
        self.h = load("fixed_pca_verifier", HELPER)
        fixture_module = load("fixed_pca_test_fixtures", HERE / "test_fixed_pca_runner.py")
        fixture_module.FixedPCATests.setUpClass()
        self.fixtures = fixture_module.FixedPCATests()

    def test_metadata_default_has_no_science_or_writes(self):
        h = self.h
        with tempfile.TemporaryDirectory() as folder:
            public, cache, asset, plan = self.fixtures.fixture(Path(folder))
            before = {str(p.relative_to(folder)): h.sha(p) for p in Path(folder).rglob("*") if p.is_file()}
            result = subprocess.run([sys.executable, "-B", str(HELPER), "--public-root", str(public), "--cache-root", str(cache),
                "--asset-root", str(asset), "--plan-sha", h.sha(public / HERE.name / "PLAN.json")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["registered_conditions"], 6)
            self.assertFalse(data["numpy_imported"])
            self.assertFalse(data["torch_imported"])
            self.assertEqual(data["actual_model_or_array_access"], 0)
            self.assertEqual(before, {str(p.relative_to(folder)): h.sha(p) for p in Path(folder).rglob("*") if p.is_file()})

    def test_all_six_registry_projection_preserves_noncomplete_status(self):
        h = self.h
        with tempfile.TemporaryDirectory() as folder:
            public, _, _, plan = self.fixtures.fixture(Path(folder))
            post = public / HERE.name
            for condition, status in zip(plan["conditions"][:2], ("COMPLETE_VALID", "BUDGET_LIMIT")):
                run = post / "runs" / condition["run_id"]
                run.mkdir(parents=True)
                h.new(run / "BASE_FIT_RECEIPT.json", dict(run_id=condition["run_id"], family="PCA_FIXED_BOLT", status=status))
            rows = h.registry_projection(plan, post)
            self.assertEqual(len(rows), 6)
            self.assertEqual([row["execution_status"] for row in rows[:3]], ["COMPLETE_VALID", "BUDGET_LIMIT", "NO_COMPLETE_BASE_FIT_RECEIPT"])
            self.assertTrue(all(row["reason"] for row in rows))
            incomplete = post / "runs" / plan["conditions"][2]["run_id"]
            incomplete.mkdir(parents=True)
            (incomplete / "BASE_FIT_RECEIPT.json").write_text("{", encoding="utf-8")
            rows = h.registry_projection(plan, post)
            self.assertEqual(rows[0]["execution_status"], "COMPLETE_VALID")
            self.assertEqual(rows[2]["execution_status"], "UNREADABLE_BASE_FIT_RECEIPT")
            self.assertEqual(len(rows), 6)

    def test_exact_six_row_alias_is_additive_and_wrong_existing_refused(self):
        h = self.h
        with tempfile.TemporaryDirectory() as folder:
            public, _, _, plan = self.fixtures.fixture(Path(folder))
            post = public / HERE.name
            data = (post / "REGISTRY.csv").read_bytes()
            self.assertTrue(h.ensure_registry_alias(post, plan)["created_now"])
            self.assertEqual((post / "EXPERIMENT_REGISTRY.csv").read_bytes(), data)
            self.assertFalse(h.ensure_registry_alias(post, plan)["created_now"])
            (post / "EXPERIMENT_REGISTRY.csv").write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                h.ensure_registry_alias(post, plan)
            self.assertEqual((post / "REGISTRY.csv").read_bytes(), data)

    def test_fixed_stage_zero_activity_overrides_base_optimizer_label(self):
        h = self.h
        base = dict(optimizer="AdamW", search_updates=0, refit_updates=0, search_exposures=0, refit_exposures=0)
        stage = dict(optimizer="NONE_FIXED_TRANSFORM", updates=0, exposures=0, actual_epochs=0., trainable_parameters=0, F_training_logical_streams=0)
        h.validate_fixed_receipts(base, {"search": stage, "refit": stage})
        for key in ("updates", "exposures", "trainable_parameters", "F_training_logical_streams"):
            bad = dict(stage, **{key: 1})
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.validate_fixed_receipts(base, {"search": bad, "refit": stage})

    def test_rescore_original_api_receives_all_six_ids_and_same_private_root(self):
        calls = []
        def api(*args):
            calls.append(args)
            return {"status": "PASS"}
        manifest = {"cost_record": {"actual_F_streams": 148*128}}
        ids = [str(i) for i in range(6)]
        self.h.rescore_call(api, manifest, "ref", "score", ["stats"], ["six rows"], ids, "new_private")
        self.assertEqual(calls[0], (manifest, "ref", "score", ["stats"], ["six rows"], ids, [manifest["cost_record"]], "new_private"))

    def test_replay_variable_K_keeps_cpu_fixed_tolerance_and_reuse_identity(self):
        calls = []
        def api(*args, **kwargs):
            calls.append((args, kwargs))
            return dict(status="PASS", run_id="pems", origins=list(range(6)), actual_F_streams=768, expected_F_streams=768,
                        atol=1e-5, rtol=1e-4, executed_now=True, T_final_accessed=False)
        value = self.h.replay_call(api, "pems", {"H": 12}, "newcache", "newpublic", "model", 100, 128)
        self.assertTrue(value["executed_now"])
        self.assertEqual(calls[0][1], dict(device="cpu", deadline=100, lease=None, batch_size=4, atol=1e-5, rtol=1e-4))
        def reused(*args, **kwargs):
            return dict(api(*args, **kwargs), executed_now=False)
        self.assertFalse(self.h.replay_call(reused, "pems", {"H": 12}, "c", "p", "m", 100, 128)["executed_now"])
        with self.assertRaises(ValueError):
            self.h.replay_call(api, "pems", {"H": 12}, "c", "p", "m", 100, 22)

    def test_execute_reports_unattempted_unknown_fresh_and_cached_forward_separately(self):
        h = self.h
        for branch in ("before_call", "forward_then_error", "fresh", "cached"):
            with self.subTest(branch=branch), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                public, cache, asset = root / "public", root / "cache", root / "asset"
                post = public / HERE.name
                post.mkdir(parents=True)
                condition = dict(run_id="tiny", family="PCA_FIXED_BOLT", spec=dict(dataset="pems08", latent_k=22))
                plan = dict(conditions=[condition], deadline_unix=time.time()+20, max_RSS_bytes=3221225472,
                    model_asset_relative_path="tiny_model", postT_registry_sha256="pin",
                    task_contracts=dict(pems08=dict(H=12, legal_origin_counts_primary=dict(V_select=148))))
                records, forwards = {}, []
                runner = types.SimpleNamespace(NAME=HERE.name, metadata=lambda *a: plan,
                    require_environment=lambda *a: None, validate_task=lambda *a: None,
                    rss_record=lambda *a: {}, save_private_failure=lambda *a: dict(phase=a[-2], error_class=type(a[-1]).__name__))
                process = types.SimpleNamespace(pid=123, create_time=lambda: 1., memory_info=lambda: types.SimpleNamespace(rss=1))
                def load_task(*args, **kwargs):
                    if branch == "before_call":
                        raise RuntimeError("synthetic failure before replay")
                    return {"synthetic": True}
                def replay(*args, **kwargs):
                    forwards.append("fake forward entered")
                    if branch == "forward_then_error":
                        raise RuntimeError("synthetic failure after a fake forward")
                    return dict(status="PASS", run_id="tiny", origins=list(range(6)), actual_F_streams=132,
                        expected_F_streams=132, atol=1e-5, rtol=1e-4, executed_now=branch == "fresh", T_final_accessed=False)
                modules = dict(psutil=types.SimpleNamespace(Process=lambda *a: process),
                    torch=types.SimpleNamespace(set_num_threads=lambda *a: None, cuda=types.SimpleNamespace(is_initialized=lambda: False)),
                    data_contract=types.SimpleNamespace(load_task=load_task), fit_base=types.SimpleNamespace(),
                    route_predictions=types.SimpleNamespace(replay_route=replay))
                values = dict(fit_receipt=dict(artifact_sha256="artifact"))
                with mock.patch.dict(sys.modules, modules), mock.patch.object(h, "load_runner", return_value=runner), \
                     mock.patch.object(h, "load_utilities", return_value=runner), mock.patch.object(h, "sha", return_value="pin"), \
                     mock.patch.object(h, "new", side_effect=lambda p, v: records.__setitem__(Path(p).name, v)), \
                     mock.patch.object(h, "prepare_inputs", return_value=(values, {}, {}, {})), \
                     mock.patch.object(h, "registry_projection", return_value=[]), \
                     mock.patch.object(h, "ensure_registry_alias", return_value=dict(sha256="pin")):
                    code = h.execute(public, cache, asset, "pin", "replay", "tiny", time.time()+10, post / "result")
                result = records["RESULT.json"]
                expected = dict(before_call=False, forward_then_error=None, fresh=True, cached=False)[branch]
                self.assertIs(result["actual_checkpoint_forward_in_this_command"], expected)
                self.assertEqual(result["checkpoint_replay_call_attempted"], branch != "before_call")
                self.assertEqual(result["checkpoint_replay_receipt_returned"], branch in {"fresh", "cached"})
                self.assertEqual(len(forwards), int(branch != "before_call"))
                self.assertEqual(code, 0 if branch in {"fresh", "cached"} else 1)

    def test_pems_H12_C170_V148_and_T_or_unknown_ID_rejection(self):
        h = self.h
        condition = dict(run_id="pems", family="PCA_FIXED_BOLT", spec={"dataset": "pems08", "latent_k": 128})
        contract = dict(H=12, shape=[17856, 170], data_sha256="data", refit_scaler_sha256="scaler",
                        legal_origin_counts_primary={"V_select": 148}, primary_split_ranges={"A_stop": [8928, 10713]})
        receipt = dict(artifact_sha256="artifact")
        manifest = dict(run_id="pems", split="V_select", cohort="pems08:primary:V_select:stride12", data_sha="data", scaler_sha="scaler",
            metadata=dict(artifact_sha256="artifact", canonical_scaler_fit_end=10713, family="PCA_FIXED_BOLT", fold="primary"),
            arrays={key: {"shape": shape} for key, shape in dict(pred_normalized=[148,12,170],target_normalized=[148,12,170],observed=[148,12,170],origins=[148],channel_ids=[170]).items()},
            cost_record=dict(actual_F_streams=148*128, expected_F_streams=148*128))
        h.validate_saved_v(manifest, receipt, condition, contract)
        bad = copy.deepcopy(manifest)
        bad["arrays"]["pred_normalized"]["shape"][1] = 48
        with self.assertRaises(ValueError):
            h.validate_saved_v(bad, receipt, condition, contract)
        with self.assertRaises(ValueError):
            h.choose({"conditions": [condition]}, "pems", "T_final")
        with self.assertRaises(ValueError):
            h.choose({"conditions": [condition]}, "missing", "rescore")


if __name__ == "__main__":
    unittest.main(verbosity=2)
