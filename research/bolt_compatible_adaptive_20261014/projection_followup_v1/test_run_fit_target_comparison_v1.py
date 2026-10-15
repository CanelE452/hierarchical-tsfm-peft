"""Controller gates and failed-path isolation on temporary scalar fixtures only."""
from pathlib import Path
import ast
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_fit_target_comparison_v1 as run


HERE = Path(__file__).resolve().parent
EXPECTED_COVERAGE = {"projection_states_checked": 4, "native_fits_checked": 8,
                     "checkpoint_states_checked": 40, "score_field_checks": 1464,
                     "saved_prediction_arrays_checked": 88, "cost_cells_checked": 20,
                     "recorded_timer_samples_checked": 120}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class GateFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.here = self.root / "research/fixture/projection_followup_v1"
        self.source = self.here / "runs/actual_projection_v2/RESULT.json"
        self.audit = self.here / "audits/PROJECTION_RESULTS_AUDIT_v2.json"
        self.source.parent.mkdir(parents=True)
        self.audit.parent.mkdir(parents=True)
        for name in ("PLAN_FIT_TARGET_SUBSPACE_KO_v1.md", "FIT_TARGET_CONTINUATION_SCOPE_KO_v1.md"):
            (self.here / name).write_text("synthetic scope fixture", encoding="utf-8")
        self.source_value = {"status": "COMPLETE", "protected_TEST_accessed": False}
        self.audit_value = {"status": "PASS", "protected_TEST_accessed": False,
                            "totals": dict(EXPECTED_COVERAGE)}
        self.write_source_binding()
        self.patcher = patch.multiple(run, HERE=self.here, ROOT=self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def write_source_binding(self):
        write_json(self.source, self.source_value)
        self.audit_value["producer"] = run.record(self.source)
        write_json(self.audit, self.audit_value)

    def test_exact_complete_source_PASS_gate(self):
        source, binding, audit = run.load_gates()
        self.assertEqual(source, self.source_value)
        self.assertEqual(binding, run.record(self.source))
        self.assertEqual(audit, run.record(self.audit))

    def test_every_accepted_coverage_count_is_required(self):
        for key in EXPECTED_COVERAGE:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.audit_value)
                changed["totals"][key] -= 1
                write_json(self.audit, changed)
                with self.assertRaisesRegex(ValueError, "FULL_ACCEPTED_SOURCE_COVERAGE"):
                    run.load_gates()
        write_json(self.audit, self.audit_value)

    def test_source_sha_mutation_rejected(self):
        self.source.write_text(self.source.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "EXACT_ACCEPTED_SOURCE_AUDIT_BINDING"):
            run.load_gates()

    def test_wrong_source_path_rejected_even_when_bytes_match(self):
        other = self.here / "same_bytes.json"
        other.write_bytes(self.source.read_bytes())
        self.audit_value["producer"] = run.record(other)
        write_json(self.audit, self.audit_value)
        with self.assertRaisesRegex(ValueError, "EXACT_ACCEPTED_SOURCE_AUDIT_BINDING"):
            run.load_gates()

    def test_repo_relative_accepted_binding_resolves_exactly(self):
        self.audit_value["producer"]["path"] = self.source.relative_to(self.root).as_posix()
        write_json(self.audit, self.audit_value)
        self.assertEqual(run.load_gates()[1], run.record(self.source))

    def test_protected_TEST_false_is_required_on_both_records(self):
        for container in ("source", "audit"):
            for value in (True, None):
                with self.subTest(container=container, value=value):
                    changed_source = copy.deepcopy(self.source_value)
                    changed_audit = copy.deepcopy(self.audit_value)
                    if container == "source":
                        changed_source["protected_TEST_accessed"] = value
                    else:
                        changed_audit["protected_TEST_accessed"] = value
                    write_json(self.source, changed_source)
                    changed_audit["producer"] = run.record(self.source)
                    write_json(self.audit, changed_audit)
                    with self.assertRaisesRegex(ValueError, "PROTECTED_TEST_SCOPE"):
                        run.load_gates()
        self.write_source_binding()

    def test_pending_audit_stops_execute_before_scientific_import_or_output(self):
        self.audit_value["status"] = "RUNNING"
        write_json(self.audit, self.audit_value)
        called = []
        real_import = __import__
        def forbid_scientific(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "torch", "psutil", "chronos", "continuation_runtime_v1"):
                called.append(name)
                raise AssertionError("scientific import before accepted gate")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=forbid_scientific):
            with self.assertRaisesRegex(ValueError, "ACTUAL_COMPLETE_SOURCE_AND_PASS_AUDIT"):
                run.execute("synthetic_never_started")
        self.assertEqual(called, [])
        self.assertFalse((self.here / "runs/synthetic_never_started").exists())

    def test_scope_presence_required(self):
        (self.here / "PLAN_FIT_TARGET_SUBSPACE_KO_v1.md").unlink()
        with self.assertRaisesRegex(ValueError, "FRESH_SCOPE_AND_PLAN"):
            run.load_gates()

    def test_prior_recursive_hashes_without_numpy_load(self):
        a, b = self.root / "past_cache.npz", self.root / "sealed_source.py"
        a.write_bytes(b"synthetic non-npz bytes; must not deserialize")
        b.write_text("synthetic source", encoding="utf-8")
        relative = run.record(a)
        relative["path"] = a.relative_to(self.root).as_posix()
        source = {"nested": [{"records": [relative, run.record(b)]}, relative],
                  "array_only_hash": {"shape": [1], "sha256": "not a file"}}
        with patch.object(np, "load", side_effect=AssertionError("actual array load forbidden")):
            actual = run.prior_file_records(source)
        self.assertEqual({item["path"] for item in actual}, {str(a.resolve()), str(b.resolve())})

    def test_prior_cache_mutation_rejected(self):
        cache = self.root / "past_cache.npz"
        cache.write_bytes(b"fixture before")
        source = {"nested": [{"artifact": run.record(cache)}]}
        cache.write_bytes(b"fixture after")
        with self.assertRaisesRegex(ValueError, "PRIOR_FILE_CHANGED"):
            run.prior_file_records(source)

    def test_TEST_reference_rejected_before_file_hash_or_read(self):
        protected = {"path": str(self.root / "test.npz"), "sha256": "irrelevant"}
        with patch.object(run, "record", side_effect=AssertionError("must reject before read")):
            with self.assertRaisesRegex(ValueError, "PROTECTED_TEST_REFERENCE"):
                run.prior_file_records({"nested": [protected]})

    def test_parent_inputs_exact_binding_required(self):
        values = {name: {"path": name + ".npz", "sha256": name + "digest"}
                  for name in ("fit", "tune", "val")}
        parent = {"status": "COMPLETE", "inputs": copy.deepcopy(values)}
        accepted = {"inputs": copy.deepcopy(values)}
        self.assertEqual(run.matching_parent_inputs("robin", parent, accepted), values)
        parent["inputs"]["fit"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "EXACT_ACCEPTED_PARENT_INPUT_BINDING"):
            run.matching_parent_inputs("robin", parent, accepted)

    def test_parent_split_contract_rejected(self):
        parent = {"status": "COMPLETE", "inputs": {"fit": {}, "tune": {}}}
        with self.assertRaisesRegex(ValueError, "COMPLETE_PARENT_INPUT_CONTRACT"):
            run.matching_parent_inputs("robin", parent, parent)

    def test_fresh_tag_rejects_existing_and_path_escape(self):
        expected = self.here / "runs/fresh_fixture"
        self.assertEqual(run.fresh_output_path("fresh_fixture"), expected.resolve())
        expected.mkdir()
        with self.assertRaisesRegex(ValueError, "FRESH_OUTPUT_DIRECTORY"):
            run.fresh_output_path("fresh_fixture")
        for tag in ("../escape", "..", "", str(self.root)):
            with self.subTest(tag=tag):
                with self.assertRaises(ValueError):
                    run.fresh_output_path(tag)


class IsolationAndSourceTests(unittest.TestCase):
    def test_route_failure_does_not_stop_other_route_or_dataset(self):
        failures, completed = [], []
        for dataset in run.DATASETS:
            for route in run.ROUTES:
                def action():
                    if dataset == "robin" and route == "FIT_TARGET_PINV":
                        raise ValueError("restricted past rank insufficient")
                    return (dataset, route)
                value = run.isolated_path(failures, dataset, route, "projection_fit", action)
                if value is not None:
                    completed.append(value)
        self.assertEqual(len(completed), 3)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["route"], "FIT_TARGET_PINV")
        self.assertEqual(failures[0]["failure"]["type"], "ValueError")
        self.assertIn(("beijing_aotizhongxin", "FIT_TARGET_PINV"), completed)

    def test_failed_native_seed_keeps_remaining_seeds_and_routes(self):
        failures, completed = [], []
        for route in run.ROUTES:
            for seed in run.SEEDS:
                def action():
                    if route == "PCA_PINV" and seed == 92601:
                        raise FloatingPointError("fixture nonfinite gradient")
                    return (route, seed)
                result = run.isolated_path(failures, "robin", route, "native_fit", action, seed)
                if result is not None:
                    completed.append(result)
        self.assertEqual(len(completed), 3)
        self.assertEqual(failures[0]["seed"], "92601")
        self.assertIn(("PCA_PINV", 92602), completed)
        self.assertIn(("FIT_TARGET_PINV", 92601), completed)

    def test_interruption_is_not_silently_converted_to_path_completion(self):
        failures = []
        def interrupt():
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run.isolated_path(failures, "robin", "PCA_PINV", "fixture", interrupt)
        self.assertEqual(failures, [])

    def test_controller_import_has_no_scientific_model_imports(self):
        command = """import importlib.util,json,sys
s=importlib.util.spec_from_file_location('synthetic_source_import',sys.argv[1])
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(json.dumps({key:key in sys.modules for key in ('numpy','torch','chronos','prepare_inputs','continuation_runtime_v1')}))
"""
        result = subprocess.run([sys.executable, "-c", command, str(HERE / "run_fit_target_comparison_v1.py")],
                                capture_output=True, text=True, check=True)
        self.assertTrue(all(value is False for value in json.loads(result.stdout).values()))

    def test_cost_function_bodies_match_original_timer(self):
        def functions(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            return {item.name: ast.dump(item, include_attributes=False)
                    for item in tree.body if isinstance(item, ast.FunctionDef)}
        original = functions(HERE / "whole_cost_v1.py")
        matched = functions(HERE / "whole_fit_target_cost_v1.py")
        self.assertEqual(original, matched)

    def test_whole_cost_routes_are_exact_five_and_same_schedule_structure(self):
        specification = importlib.util.spec_from_file_location("synthetic_cost_source", HERE / "whole_fit_target_cost_v1.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        self.assertEqual(module.ROUTES, ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT",
                                         "PCA_PINV_NATIVE", "FIT_TARGET_PINV_NATIVE"))
        schedule = module.balanced_schedule()
        self.assertEqual((module.WARMUP, module.REPEATS), (2, 6))
        for position in range(5):
            self.assertEqual({schedule[repeat][position] for repeat in range(5)}, set(module.ROUTES))
        self.assertEqual(set(schedule[5]), set(module.ROUTES))

    def test_native_cost_seed_is_fixed_92601_and_selection_precedes_DEV(self):
        source = (HERE / "run_fit_target_comparison_v1.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertEqual(run.SEEDS, (92601, 92602))
        seed_conditions = [item for item in ast.walk(tree) if isinstance(item, ast.If)
                           and ast.dump(item.test, include_attributes=False) ==
                           ast.dump(ast.parse("seed == 92601", mode="eval").body, include_attributes=False)]
        self.assertEqual(len(seed_conditions), 1)
        seed_assignment = ast.unparse(seed_conditions[0])
        self.assertIn("native_seed", seed_assignment)
        self.assertIn("92601", seed_assignment)
        self.assertLess(source.index("rt.write_new(output/'SELECTION_LOCK.json'"),
                        source.index("lambda: load_split(dataset, development)"))
        self.assertLess(source.index("rt.write_new(output/'SELECTION_LOCK.json'"),
                        source.index("lambda: oracle_split(dataset, route, condition, state, split)"))


if __name__ == "__main__":
    unittest.main()
