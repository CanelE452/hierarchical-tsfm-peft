"""Explicit synthetic-only tests; never discover the repository test suite."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import native_cpu_depth as native


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class FakeRuntime:
    def __init__(self, public, *, failure=None, bad_f=None, failure_message="synthetic route-local failure"):
        self.public = public
        self.calls = []
        self.failure = failure
        self.bad_f = bad_f
        self.failure_message = failure_message

    def load_task(self, dataset, cache):
        return dict(name=dataset, data_sha256="data_" + dataset, columns=["a", "b"], L=512, H=12 if dataset == "pems08" else 48)

    def fit(self, spec, task, cache, public, foundation, deadline):
        assert foundation is None and spec["device"] == "cpu"
        self.calls.append(spec["run_id"])
        if spec["run_id"] == self.failure:
            raise ValueError(self.failure_message)
        folder = public / "runs" / spec["run_id"]
        f = 1 if spec["run_id"] == self.bad_f else 0
        dump(folder / "SOURCE_CONTRACT.json", dict(spec=spec, model_sha="NO_FOUNDATION"))
        for stage in ("search", "refit"):
            dump(folder / stage / "RECEIPT.json", dict(F_total_logical_streams=f,
                 F_total_physical_streams=f, F_training_logical_streams=f))
        receipt = dict(status="COMPLETE_VALID", run_id=spec["run_id"], family=spec["family"],
                       F_logical_streams=f, epochs=spec["epochs"], selected_epoch=1.0,
                       training_status="SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED", search_updates=7, refit_updates=3)
        dump(folder / "BASE_FIT_RECEIPT.json", receipt)
        return receipt

    def predict(self, identity, task, split, cache, public, foundation, **kwargs):
        assert foundation is None and split == "V_select" and kwargs["fold"] == "primary" and kwargs["role"] == "base_parent"
        manifest = dict(run_id=identity, split=split, metadata=dict(family=identity.split("_E1_")[1].split("_k0")[0], fold="primary"),
                        cost_record=dict(actual_F_streams=0, expected_F_streams=0), score=dict(mse=.25, mae=.3))
        dump(public / "predictions" / identity / split / "PREDICTION_MANIFEST.json", manifest)
        return dict(manifest=manifest, cost_record=manifest["cost_record"])


class NativeCPUFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="native_cpu_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.public = Path(self.temp.name) / "public"
        self.cache = Path(self.temp.name) / "cache"
        self.public.mkdir(); self.cache.mkdir()
        (self.public / "frozen.py").write_text("fixture source\n")
        self.pins = {"frozen.py": native.file_sha(self.public / "frozen.py")}
        self.rows = []
        self.assets = {}
        for dataset in ("jena", "electricity", "pems08"):
            horizon = 12 if dataset == "pems08" else 48
            asset = dict(columns=["a", "b"], L=512, H=horizon, data_sha256="data_" + dataset, array_sha256={"physical": "fixture"})
            self.assets[dataset] = asset
            for family in ("DIRECT_G", "DIRECT_B2", "DIRECT_B4"):
                identity = f"primary_{dataset}_E1_{family}_k0_s92601"
                spec = dict(run_id=identity, dataset=dataset, family=family, latent_k=2, seed=92601,
                            fold="primary", epochs=[.25], lr=.001, device="cuda", microbatch=4,
                            effective_batch=16, F_row_microbatch=128, checkpoint_inputs=False,
                            lora_rank=8, lora_alpha=16, stage_label="registered_initial_breadth")
                axes = dict(L=512, H=horizon, C=2, columns=asset["columns"], fold="primary", source_array_fingerprints=asset["array_sha256"])
                source = dict(spec=spec, files=self.pins, source_sha=fingerprint(self.pins),
                              config_sha=fingerprint(dict(spec=spec, axes=axes)), data_sha=asset["data_sha256"], model_sha="NO_FOUNDATION")
                dump(self.public / "runs" / identity / "SOURCE_CONTRACT.json", source)
                dump(self.public / "runs" / identity / "BASE_FIT_RECEIPT.json", dict(status="COMPLETE_VALID", run_id=identity, artifact_sha256="old_declared_no_read"))
                self.rows.append(dict(experiment_id=identity, dataset=dataset, panel="", fold="primary", block="E1", family=family,
                                      num_channels="2", latent_k="0", seed="92601", length="512", horizon=str(horizon),
                                      requirement="MANDATORY_MINIMUM", execution_status="COMPLETE_VALID", training_status="LIMITED_SCREEN",
                                      validity_status="VALID", scientific_status="UNASSESSED", publication_status="NOT_PUBLISHED",
                                      reason="fixture", planned_epoch_endpoints="0.25", resume_command=""))
        dump(self.public / "DATA_CONTRACT.json", dict(assets=self.assets))
        self.write_registry()

    def write_registry(self):
        with (self.public / "EXPERIMENT_REGISTRY.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0])); writer.writeheader(); writer.writerows(self.rows)

    def plan(self):
        return native.build_plan(self.public, self.cache, expected_sources=self.pins)

    def register(self, plan):
        self.rows.extend(copy.deepcopy(plan["proposed_registry_rows"]))
        self.write_registry()

    def execute(self, plan, runtime):
        return native.execute_plan(plan, self.public, self.cache, max_seconds=30,
                                   expected_sources=self.pins, _runtime=runtime, _skip_resource_monitor=True)

    def test_plan_is_readonly_exact9_cpu_fresh_and_no_framework_import(self):
        before = (self.public / "EXPERIMENT_REGISTRY.csv").read_bytes()
        modules_before = set(sys.modules)
        plan = self.plan()
        self.assertEqual(len(plan["routes"]), 9)
        self.assertEqual(before, (self.public / "EXPERIMENT_REGISTRY.csv").read_bytes())
        self.assertFalse((self.cache / "native_cpu_depth_claims").exists())
        for route in plan["routes"]:
            spec = route["spec"]
            self.assertTrue(spec["run_id"].endswith("__cpu_fresh_e3"))
            self.assertNotIn("beijing", spec["run_id"])
            self.assertEqual(spec["epochs"], [.25, 1., 3.])
            self.assertEqual(spec["device"], "cpu")
            self.assertEqual(spec["latent_k"], 2)
            self.assertEqual(route["foundation_identity"], "NO_FOUNDATION")
        new_framework_modules = {name for name in set(sys.modules) - modules_before
                                 if name.split(".", 1)[0] in {"numpy", "torch"}}
        self.assertEqual(new_framework_modules, set())

    def test_unregistered_routes_never_call_runtime(self):
        runtime = FakeRuntime(self.public)
        result = self.execute(self.plan(), runtime)
        self.assertEqual(runtime.calls, [])
        self.assertEqual(result["completed_routes"], 0)
        self.assertEqual(len(result["records"]), 9)

    def test_no_foundation_f0_and_registry_readonly(self):
        plan = self.plan(); self.register(plan)
        before = (self.public / "EXPERIMENT_REGISTRY.csv").read_bytes()
        result = self.execute(plan, FakeRuntime(self.public))
        self.assertEqual(result["completed_routes"], 9)
        self.assertTrue(all(x["actual_F_streams"] == 0 for x in result["records"]))
        self.assertEqual(before, (self.public / "EXPERIMENT_REGISTRY.csv").read_bytes())

    def test_one_failure_does_not_suppress_other_eight(self):
        plan = self.plan(); self.register(plan)
        runtime = FakeRuntime(self.public, failure=plan["routes"][0]["run_id"])
        result = self.execute(plan, runtime)
        self.assertEqual(len(runtime.calls), 9)
        self.assertEqual(result["completed_routes"], 8)
        self.assertEqual(result["records"][0]["status"], "FAILED_PRESERVED")

    def test_nonzero_f_is_rejected_and_other_routes_continue(self):
        plan = self.plan(); self.register(plan)
        result = self.execute(plan, FakeRuntime(self.public, bad_f=plan["routes"][0]["run_id"]))
        self.assertEqual(result["completed_routes"], 8)
        self.assertEqual(result["records"][0]["status"], "FAILED_PRESERVED")

    def test_selection_lock_rejects_before_fit(self):
        plan = self.plan(); self.register(plan)
        dump(self.public / "SELECTION_LOCK.json", dict(locked=True))
        runtime = FakeRuntime(self.public)
        with self.assertRaises(ValueError): self.execute(plan, runtime)
        self.assertEqual(runtime.calls, [])

    def test_wrong_registered_identity_blocks_only_that_route(self):
        plan = self.plan(); self.register(plan)
        self.rows[9]["family"] = "B0"; self.write_registry()
        result = self.execute(plan, FakeRuntime(self.public))
        self.assertEqual(result["completed_routes"], 8)
        self.assertEqual(result["records"][0]["status"], "NOT_EXECUTED_PRECONDITION")

    def test_existing_directory_preserved_without_resume(self):
        plan = self.plan(); self.register(plan)
        folder = self.public / "runs" / plan["routes"][0]["run_id"]; folder.mkdir()
        (folder / "old.txt").write_text("preserve me")
        result = self.execute(plan, FakeRuntime(self.public))
        self.assertEqual(result["completed_routes"], 8)
        self.assertEqual((folder / "old.txt").read_text(), "preserve me")

    def test_source_change_rejects_before_fit(self):
        plan = self.plan(); self.register(plan)
        (self.public / "frozen.py").write_text("changed\n")
        runtime = FakeRuntime(self.public)
        with self.assertRaises(ValueError): self.execute(plan, runtime)
        self.assertEqual(runtime.calls, [])

    def test_tampered_spec_cannot_change_objective_or_recipe(self):
        plan = self.plan(); self.register(plan)
        plan["routes"][0]["spec"]["lr"] = .5
        with self.assertRaises(ValueError): self.execute(plan, FakeRuntime(self.public))

    def test_repeat_execution_cannot_resume_or_overwrite(self):
        plan = self.plan(); self.register(plan)
        first = self.execute(plan, FakeRuntime(self.public)); self.assertEqual(first["completed_routes"], 9)
        runtime = FakeRuntime(self.public); second = self.execute(plan, runtime)
        self.assertEqual(runtime.calls, []); self.assertEqual(second["completed_routes"], 0)

    def test_duplicate_or_truncated_route_list_is_rejected(self):
        plan = self.plan(); self.register(plan)
        for replacement in (plan["routes"][:-1], plan["routes"] + [plan["routes"][0]]):
            changed = copy.deepcopy(plan); changed["routes"] = replacement
            with self.assertRaises(ValueError): self.execute(changed, FakeRuntime(self.public))

    def test_wall_above_initial_30minutes_is_rejected(self):
        plan = self.plan(); self.register(plan)
        with self.assertRaises(ValueError):
            native.execute_plan(plan, self.public, self.cache, max_seconds=1801,
                                expected_sources=self.pins, _runtime=FakeRuntime(self.public), _skip_resource_monitor=True)

    def test_asset_absolute_error_stays_private(self):
        plan = self.plan(); self.register(plan)
        synthetic_drive = "Q"
        synthetic_root = "synthetic_private_fixture"
        runtime = FakeRuntime(self.public, failure=plan["routes"][0]["run_id"],
                              failure_message=f"Missing {synthetic_drive}:/{synthetic_root}/sensor.npz")
        result = self.execute(plan, runtime)
        self.assertNotIn(f"{synthetic_drive}:/", result["records"][0]["reason"])
        self.assertNotIn(synthetic_root, result["records"][0]["reason"])
        self.assertEqual(result["completed_routes"], 8)


if __name__ == "__main__":
    unittest.main()
