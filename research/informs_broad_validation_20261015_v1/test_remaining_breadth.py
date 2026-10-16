"""Stdlib-only bounded bridge guards; never execute real fits, data, or CUDA."""
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import remaining_breadth as bridge


FIELDS = list(bridge.SCIENTIFIC_FIELDS) + [
    "execution_status", "training_status", "validity_status", "scientific_status",
    "publication_status", "reason", "resume_command",
]


def row(dataset="beijing", family="B0", k=9, fold="primary", **changes):
    width, horizon = bridge.DATASET_AXES[dataset]
    value = dict(experiment_id=f"{fold}_{dataset}_E1_{family}_k{k}_s92601", dataset=dataset, panel="",
                 fold=fold, block="E1", family=family, num_channels=str(width), latent_k=str(k),
                 seed="92601", length="512", horizon=str(horizon),
                 requirement="FULL_REGISTERED_GRID" if fold == "primary" else "TIME_TRANSFER_REQUIRED",
                 planned_epoch_endpoints="0.25|1|3", execution_status="PENDING_QUEUE",
                 training_status="NOT_STARTED", validity_status="NOT_CHECKED",
                 scientific_status="UNASSESSED", publication_status="NOT_PUBLISHED", reason="", resume_command="unchanged")
    value.update(changes)
    return value


def write_registry(root, rows):
    root.mkdir(exist_ok=True)
    with (root / "EXPERIMENT_REGISTRY.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def inventory(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class RemainingBreadthTests(unittest.TestCase):
    def test_explicit_run_ids_preserve_registry_order_and_unselected_rows(self):
        rows = [row(dataset="electricity", family="B0", k=41),
                row(dataset="electricity", family="B2", k=41),
                row(dataset="electricity", family="B0", k=241),
                row(dataset="pems08", family="B2", k=22),
                row(fold="secondary", family="RAW", k=0)]
        before = copy.deepcopy(rows)
        requested = ["primary_pems08_E1_B2_k22_s92601", "primary_electricity_E1_B0_k41_s92601"]
        chosen = bridge.select_pending(rows, scope="primary", run_ids=requested)
        self.assertEqual([r["experiment_id"] for r in chosen],
                         ["primary_electricity_E1_B0_k41_s92601", "primary_pems08_E1_B2_k22_s92601"])
        self.assertEqual(rows, before)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_registry(root, rows)
            files_before = inventory(root)
            default = bridge.build_plan(root, root / "cache")
            self.assertEqual(default, bridge.build_plan(root, root / "cache", run_ids=None))
            self.assertNotIn("requested_run_ids", default)
            plan = bridge.build_plan(root, root / "cache", scope="primary", run_ids=requested)
            self.assertEqual(plan["requested_run_ids"], requested)
            self.assertEqual(plan["exact_run_ids"],
                             ["primary_electricity_E1_B0_k41_s92601", "primary_pems08_E1_B2_k22_s92601"])
            self.assertEqual(plan["registered_pending_routes"], 2)
            self.assertEqual(inventory(root), files_before)

    def test_explicit_ids_reject_unknown_completed_duplicate_and_empty_selection(self):
        pending = row()
        rows = [pending, row(family="B2", execution_status="COMPLETE_VALID")]
        for requested, message in ((["unknown"], "Unknown"),
                                   (["primary_beijing_E1_B2_k9_s92601"], "Unknown"),
                                   ([pending["experiment_id"], pending["experiment_id"]], "Duplicate requested"),
                                   ([], "nonempty"), ([""], "nonempty")):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, message):
                bridge.select_pending(rows, run_ids=requested)

    def test_explicit_ids_cannot_hide_scope_mismatch_or_other_eligible_identity(self):
        primary = row()
        secondary = row(fold="secondary", family="RAW", k=0)
        with self.assertRaisesRegex(ValueError, "scope"):
            bridge.select_pending([primary, secondary], scope="primary", run_ids=[secondary["experiment_id"]])
        malformed = row(fold="secondary", family="B0", k=3, num_channels="12")
        with self.assertRaisesRegex(ValueError, "Registered axes"):
            bridge.select_pending([primary, malformed], scope="primary", run_ids=[primary["experiment_id"]])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_registry(root, [primary, row(execution_status="COMPLETE_VALID")])
            with self.assertRaisesRegex(ValueError, "Duplicate registry"):
                bridge.build_plan(root, root / "cache", scope="primary", run_ids=[primary["experiment_id"]])

    def test_execute_rejects_invalid_id_selection_before_runtime_imports_or_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_registry(root, [row(), row(fold="secondary", family="RAW", k=0)])
            before = inventory(root)
            requests = (("all", ["unknown"], "Unknown"),
                        ("all", ["primary_beijing_E1_B0_k9_s92601"] * 2, "Duplicate requested"),
                        ("primary", ["secondary_beijing_E1_RAW_k0_s92601"], "scope"))
            with patch.dict(os.environ, {key: "" for key in bridge.OWNER_ENVIRONMENT_KEYS}):
                with patch.object(bridge, "verify_sources", return_value=bridge.SOURCE_PINS):
                    with patch.object(bridge, "combined_storage", return_value={}):
                        for scope, requested, message in requests:
                            with self.subTest(scope=scope, requested=requested), self.assertRaisesRegex(ValueError, message):
                                bridge.execute(root, root / "cache", root / "assets", scope=scope, run_ids=requested)
            self.assertEqual(inventory(root), before)

    def test_main_forwards_repeated_run_ids_to_plan_and_execute(self):
        requested = ["primary_electricity_E1_B0_k41_s92601", "primary_pems08_E1_B2_k22_s92601"]
        for execute in (False, True):
            target = "execute" if execute else "build_plan"
            argv = ["--public-root", "public", "--cache-root", "cache", "--asset-root", "assets", "--scope", "primary"]
            for run_id in requested:
                argv.extend(["--run-id", run_id])
            if execute:
                argv.append("--execute")
            with self.subTest(execute=execute), patch.object(bridge, target, return_value={"status": "PLAN_ONLY_NOT_EXECUTED"}) as called:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(bridge.main(argv), 0)
                self.assertEqual(called.call_args.kwargs, {"scope": "primary", "run_ids": requested})

    def test_repeated_cli_run_ids_plan_is_stdlib_only_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cache = Path(temp) / "public", Path(temp) / "cache"
            cache.mkdir()
            write_registry(root, [row(dataset="electricity", family="B0", k=41),
                                  row(dataset="electricity", family="B2", k=41),
                                  row(dataset="pems08", family="B2", k=22),
                                  row(fold="secondary", family="RAW", k=0)])
            before = inventory(Path(temp))
            child = r"""
import builtins, sys
blocked = {"numpy", "torch", "chronos", "psutil", "campaign", "fit_base", "data_contract", "prediction_cache", "resources", "registry_extension"}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("Plan imported execution dependency: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, sys.argv[1])
import remaining_breadth
remaining_breadth.main(sys.argv[2:])
assert not (blocked & set(sys.modules))
"""
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
            done = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", child, str(bridge.HERE),
                                   "--public-root", str(root), "--cache-root", str(cache), "--scope", "primary",
                                   "--run-id", "primary_pems08_E1_B2_k22_s92601",
                                   "--run-id", "primary_electricity_E1_B0_k41_s92601"],
                                  env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(done.returncode, 0, done.stderr)
            plan = json.loads(done.stdout)
            self.assertEqual(plan["exact_run_ids"],
                             ["primary_electricity_E1_B0_k41_s92601", "primary_pems08_E1_B2_k22_s92601"])
            self.assertEqual((plan["registered_pending_routes"], plan["secondary_transfer_routes"]), (2, 0))
            self.assertFalse(plan["registry_written"])
            self.assertFalse(plan["data_or_model_loaded"])
            self.assertEqual(inventory(Path(temp)), before)

    def test_scopes_partition_registered_22_without_mutating_rows_or_registry(self):
        primary = [row(dataset=dataset, family=family, k=k)
                   for dataset, ks in (("beijing", (9,)), ("jena", (16,)),
                                       ("electricity", (41, 241)), ("pems08", (22, 128)))
                   for k in ks for family in ("B0", "B2")]
        secondary = [row(dataset=dataset, family=family, k=0 if family in {"RAW", "LORA"} else k,
                         fold="secondary")
                     for dataset, k in (("beijing", 3), ("jena", 6))
                     for family in ("RAW", "B0", "B2", "B4", "LORA")]
        rows = primary + secondary
        before = copy.deepcopy(rows)
        self.assertEqual(bridge.select_pending(rows), bridge.select_pending(rows, scope="all"))
        self.assertEqual(bridge.select_pending(rows), before)
        self.assertEqual(bridge.select_pending(rows, scope="primary"), primary)
        self.assertEqual(bridge.select_pending(rows, scope="secondary"), secondary)
        self.assertEqual((len(primary), len(secondary)), (12, 10))
        self.assertEqual(rows, before)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_registry(root, rows)
            files_before = inventory(root)
            default = bridge.build_plan(root, root / "cache")
            self.assertEqual(default, bridge.build_plan(root, root / "cache", scope="all"))
            for scope, expected in (("primary", primary), ("secondary", secondary)):
                plan = bridge.build_plan(root, root / "cache", scope=scope)
                self.assertEqual(plan["scope"], scope)
                self.assertEqual(plan["exact_run_ids"], [r["experiment_id"] for r in expected])
                self.assertEqual(plan["registered_pending_routes"], len(expected))
                self.assertTrue(all(r["fold"] == scope for r in plan["routes"]))
            self.assertEqual(inventory(root), files_before)

    def test_scope_does_not_hide_other_scope_invalid_identity_or_global_duplicate(self):
        malformed = row(fold="secondary", family="B0", k=3, num_channels="12")
        with self.assertRaisesRegex(ValueError, "Registered axes"):
            bridge.select_pending([row(), malformed], scope="primary")
        with self.assertRaisesRegex(ValueError, "immutable scientific"):
            bridge.select_pending([row(experiment_id="bad"), row(fold="secondary", family="RAW", k=0)],
                                  scope="secondary")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_registry(root, [row(), row(execution_status="COMPLETE_VALID")])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                bridge.build_plan(root, root / "cache", scope="secondary")

    def test_invalid_scope_fails_closed_before_execution_gates_or_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_registry(root, [row()])
            before = inventory(root)
            for scope in ("", "Secondary", "all,secondary", None):
                with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "scope"):
                    bridge.select_pending([], scope=scope)
                with self.subTest(plan=scope), self.assertRaisesRegex(ValueError, "scope"):
                    bridge.build_plan(root, root / "cache", scope=scope)
                with patch.object(bridge, "require_unlocked", side_effect=AssertionError("gate called")):
                    with self.subTest(execute=scope), self.assertRaisesRegex(ValueError, "scope"):
                        bridge.execute(root, root / "cache", root / "assets", scope=scope)
            self.assertEqual(inventory(root), before)

    def test_main_forwards_scope_to_plan_and_execute(self):
        for execute in (False, True):
            for scope in ("all", "primary", "secondary"):
                with self.subTest(execute=execute, scope=scope):
                    target = "execute" if execute else "build_plan"
                    argv = ["--public-root", "public", "--cache-root", "cache", "--asset-root", "assets",
                            "--reserve-seconds", "123", "--scope", scope] + (["--execute"] if execute else [])
                    with patch.object(bridge, target, return_value={"status": "PLAN_ONLY_NOT_EXECUTED"}) as called:
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(bridge.main(argv), 0)
                    args = (Path("public"), Path("cache"), Path("assets"), 123) if execute else (Path("public"), Path("cache"), 123)
                    called.assert_called_once_with(*args, scope=scope, run_ids=None)

    def test_execute_empty_scope_does_not_import_execution_modules_or_touch_other_rows(self):
        for scope, other in (("secondary", row()),
                             ("primary", row(fold="secondary", family="RAW", k=0))):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_registry(root, [other])
                before = inventory(root)
                with patch.dict(os.environ, {key: "" for key in bridge.OWNER_ENVIRONMENT_KEYS}):
                    with patch.object(bridge, "verify_sources", return_value=bridge.SOURCE_PINS):
                        with patch.object(bridge, "combined_storage", return_value={}):
                            result = bridge.execute(root, root / "cache", root / "assets", scope=scope)
                self.assertEqual(result["status"], "NO_PENDING_ROUTES")
                self.assertEqual(result["plan"]["scope"], scope)
                self.assertEqual(result["plan"]["exact_run_ids"], [])
                self.assertEqual(inventory(root), before)

    def test_exact_pending_scope_excludes_done_direct_and_unrelated(self):
        rows = [row(), row(family="B2"), row(family="RAW", k=0, fold="secondary"),
                row(family="LORA", k=0, fold="secondary"),
                row(family="DIRECT_G", k=0, fold="secondary", execution_status="COMPLETE_VALID"),
                row(family="B0", k=3, execution_status="COMPLETE_VALID"),
                row(dataset="electricity", family="B4", k=81),
                row(requirement="MANDATORY_MINIMUM"),
                row(block="E3")]
        chosen = bridge.select_pending(rows)
        self.assertEqual([r["family"] for r in chosen], ["B0", "B2", "RAW", "LORA"])
        self.assertEqual(rows[0]["latent_k"], "9")

    def test_raw_has_zero_training_and_explicit_baseline(self):
        spec = bridge.route_spec(row(family="RAW", k=0, fold="secondary"))
        self.assertEqual(spec["baseline"], "RAW")
        self.assertIsNone(spec["fit_family"])
        self.assertEqual(spec["epochs"], [])
        self.assertEqual(spec["planned_training_updates"], 0)
        self.assertEqual(spec["action"], "fixed_RAW_V_inference")

    def test_plain_lora_maps_explicitly_without_changing_registry_k(self):
        registered = row(dataset="jena", family="LORA", k=0, fold="secondary")
        before = copy.deepcopy(registered)
        spec = bridge.route_spec(registered)
        self.assertEqual(spec["family"], "LORA")
        self.assertEqual(spec["lr"], .0001)
        self.assertEqual(spec["latent_k"], 21)
        self.assertEqual(spec["epochs"], [.25])
        self.assertEqual(spec["stage_label"], "registered_time_transfer_secondary")
        self.assertEqual((spec["lora_rank"], spec["lora_alpha"]), (8, 16))
        self.assertEqual((spec["microbatch"], spec["effective_batch"], spec["F_row_microbatch"]), (4, 16, 128))
        self.assertFalse(spec["checkpoint_inputs"])
        self.assertEqual(registered, before)

    def test_primary_spec_is_registered_point25_screen(self):
        spec = bridge.route_spec(row(dataset="pems08", family="B2", k=128))
        self.assertEqual(spec["latent_k"], 128)
        self.assertEqual(spec["device"], "cuda")
        self.assertEqual(spec["lr"], .001)
        self.assertEqual(spec["stage_label"], "registered_initial_breadth")

    def test_unsupported_secondary_alias_fails_instead_of_silent_skip(self):
        with self.assertRaisesRegex(ValueError, "Unsupported registered family"):
            bridge.select_pending([row(family="RAW_BOLT_SMALL", k=0, fold="secondary")])
        with self.assertRaisesRegex(ValueError, "Unsupported family"):
            bridge.route_spec(row(family="LORA_R8", k=0, fold="secondary"))

    def test_duplicate_id_rejected_even_outside_the_pending_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_registry(root, [row(), row(execution_status="COMPLETE_VALID")])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                bridge.build_plan(root, root / "cache")

    def test_identity_and_fixed_axes_are_checked(self):
        for change in (dict(experiment_id="../escape"), dict(num_channels="12"),
                       dict(seed="92602"), dict(horizon="12"), dict(latent_k="0")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bridge.select_pending([row(**change)])
        with self.assertRaisesRegex(ValueError, "RAW/LORA registry K"):
            bridge.select_pending([row(family="LORA", k=3, fold="secondary")])

    def test_cpu_cli_plan_imports_no_framework_or_execution_modules_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cache = Path(temp) / "public", Path(temp) / "cache"
            cache.mkdir()
            write_registry(root, [row(), row(family="RAW", k=0, fold="secondary"),
                                  row(dataset="jena", family="LORA", k=0, fold="secondary")])
            before = inventory(Path(temp))
            child = r"""
import builtins, json, sys
from pathlib import Path
blocked = {"numpy", "torch", "chronos", "psutil", "campaign", "fit_base", "data_contract", "prediction_cache", "resources", "registry_extension"}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("Dry-run imported an execution dependency: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, sys.argv[1])
import remaining_breadth
remaining_breadth.main(["--public-root", sys.argv[2], "--cache-root", sys.argv[3]])
assert not (blocked & set(sys.modules))
"""
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
            done = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", child,
                                   str(bridge.HERE), str(root), str(cache)],
                                  env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(done.returncode, 0, done.stderr)
            plan = json.loads(done.stdout)
            self.assertEqual(plan["status"], "PLAN_ONLY_NOT_EXECUTED")
            self.assertEqual(plan["registered_pending_routes"], 3)
            self.assertEqual(plan["reserve_seconds"], 6000)
            self.assertEqual((plan["primary_full_grid_routes"], plan["secondary_transfer_routes"]), (1, 2))
            self.assertTrue(plan["metadata_only"])
            self.assertFalse(plan["registry_written"])
            self.assertFalse(plan["data_or_model_loaded"])
            self.assertTrue(all(r["actual_execution_performed"] is False for r in plan["routes"]))
            self.assertEqual(inventory(Path(temp)), before)

    def test_locked_execute_refuses_before_any_framework_import_or_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_registry(root, [row()])
            (root / "SELECTION_LOCK.json").write_text('{"locked": true}', encoding="utf-8")
            before = inventory(root)
            with self.assertRaisesRegex(ValueError, "selection is locked"):
                bridge.execute(root, root / "cache", root / "assets")
            self.assertEqual(inventory(root), before)

    def test_malformed_lock_is_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "SELECTION_LOCK.json").write_text('{"locked": "false"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Malformed selection"):
                bridge.require_unlocked(root)

    def test_owner_environment_rejects_nested_reservation_before_source_or_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {"INFORMS_GPU_LEASE_ID": "synthetic-parent"}):
                with self.assertRaisesRegex(ValueError, "Standalone GPU ownership"):
                    bridge.execute(root, root / "cache", root / "assets")

    def test_missing_pins_prevent_execution_before_data_or_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_registry(root, [row()])
            env = {key: "" for key in bridge.OWNER_ENVIRONMENT_KEYS}
            with patch.dict(os.environ, env):
                with self.assertRaisesRegex(ValueError, "scientific source pins"):
                    bridge.execute(root, root / "cache", root / "assets")

    def test_partial_fit_and_v_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cache = Path(temp) / "public", Path(temp) / "cache"
            root.mkdir(); cache.mkdir()
            registered = row()
            partial = cache / "runs" / registered["experiment_id"] / "search"
            partial.mkdir(parents=True)
            (partial / "checkpoint_000000.pt").write_bytes(b"synthetic partial; never deserialize")
            before = inventory(Path(temp))
            with self.assertRaisesRegex(ValueError, "partial fit"):
                bridge.preflight_route(registered, root, cache)
            self.assertEqual(inventory(Path(temp)), before)
            raw = row(family="RAW", k=0, fold="secondary")
            public_v = root / "predictions" / raw["experiment_id"] / "V_select"
            public_v.mkdir(parents=True)
            (public_v / "REFERENCE_CONTRACT.json").write_text("{}", encoding="utf-8")
            before = inventory(Path(temp))
            with self.assertRaisesRegex(ValueError, "partial V"):
                bridge.preflight_route(raw, root, cache)
            self.assertEqual(inventory(Path(temp)), before)

    def test_existing_complete_artifact_is_only_reused_with_exact_source_spec_and_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cache = Path(temp) / "public", Path(temp) / "cache"
            registered = row()
            public_run = root / "runs" / registered["experiment_id"]
            private_run = cache / "runs" / registered["experiment_id"]
            public_run.mkdir(parents=True); private_run.mkdir(parents=True)
            artifact = private_run / "selected_bundle.pt"
            artifact.write_bytes(b"synthetic complete artifact; opaque hash only")
            receipt = dict(status="COMPLETE_VALID", run_id=registered["experiment_id"],
                           artifact_cache_relative_path=artifact.relative_to(cache).as_posix(),
                           artifact_sha256=bridge.file_sha(artifact))
            source = dict(spec=bridge.route_spec(registered), files=bridge.SOURCE_PINS)
            (public_run / "BASE_FIT_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
            (public_run / "SOURCE_CONTRACT.json").write_text(json.dumps(source), encoding="utf-8")
            self.assertEqual(bridge.preflight_route(registered, root, cache), receipt)
            artifact.write_bytes(b"changed synthetic artifact")
            before = inventory(Path(temp))
            with self.assertRaisesRegex(ValueError, "artifact is absent or its SHA"):
                bridge.preflight_route(registered, root, cache)
            self.assertEqual(inventory(Path(temp)), before)

    def test_status_writer_preserves_all_scientific_fields_and_other_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = [row(), row(dataset="jena", family="B2", k=16)]
            write_registry(root, original)
            bridge.update_status(root, original[0], execution_status="COMPLETE_VALID",
                                 training_status="LIMITED_SCREEN", validity_status="PENDING_INDEPENDENT_VALIDATION",
                                 reason="Synthetic test; no actual fit")
            _, after = bridge.read_registry(root)
            self.assertEqual(after[1], original[1])
            for name in bridge.SCIENTIFIC_FIELDS:
                self.assertEqual(after[0][name], original[0][name])
            self.assertEqual(after[0]["publication_status"], "NOT_PUBLISHED")
            with self.assertRaisesRegex(ValueError, "Only mutable"):
                bridge.update_status(root, original[0], latent_k="2")

    def test_status_writer_rejects_stale_scientific_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = row()
            write_registry(root, [original])
            stale = dict(original, latent_k="3")
            before = inventory(root)
            with self.assertRaisesRegex(ValueError, "scientific identity differs"):
                bridge.update_status(root, stale, execution_status="EXECUTION_FAILED")
            self.assertEqual(inventory(root), before)

    def test_receipt_summary_preserves_partial_updates_and_resource_status(self):
        summary = bridge._receipt_summary(dict(status="BUDGET_LIMIT", training_status="UNDERTRAINED_OR_BUDGET_LIMIT",
                                               phase="search", updates=17, exposures=272,
                                               private_debug_absolute_path="never publish"))
        self.assertEqual(summary["updates"], 17)
        self.assertEqual(summary["phase"], "search")
        self.assertNotIn("private_debug_absolute_path", summary)

    def test_only_timeout_at_the_actual_deadline_is_a_resource_stop(self):
        self.assertTrue(bridge.deadline_timeout(TimeoutError("prediction deadline"), 100., now=100.))
        self.assertFalse(bridge.deadline_timeout(TimeoutError("unrelated early timeout"), 100., now=99.))
        self.assertFalse(bridge.deadline_timeout(ValueError("other error"), 100., now=101.))

    def test_closed_central_receipt_is_linked_by_existing_file_and_sha_even_after_fatal_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cache = Path(temp) / "public", Path(temp) / "cache"
            root.mkdir(); cache.mkdir()
            central = cache / "gpu_receipts" / "synthetic-closed-lease.json"
            central.parent.mkdir()
            central.write_text('{"status":"CLOSED","charged_process_seconds":2}', encoding="utf-8")
            result = dict(started_unix=0., source_files_before=bridge.SOURCE_PINS,
                          fatal_error_type="SyntheticImportFailure",
                          routes=[dict(run_id="synthetic", status="EXECUTION_FAILED",
                                       fit_call_attempted_now=True, training_executed_now=None,
                                       training_status="PARTIAL_TRAINING_UNRESOLVED")])
            receipt = root / "execution_batches" / "synthetic.json"
            bridge.finish_execution_result(result, root, cache, "synthetic-closed-lease", receipt)
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "EXECUTION_ABORTED_WITH_PRESERVED_STATE")
            self.assertEqual(saved["fit_calls_with_unresolved_execution"], 1)
            self.assertTrue(saved["central_lease_receipt_exists"])
            self.assertEqual(saved["lease_receipt_sha256"], bridge.file_sha(central))
            self.assertEqual(saved["lease_receipt_cache_relative_path"], "gpu_receipts/synthetic-closed-lease.json")
            self.assertFalse(saved["source_six_unchanged"])
            self.assertIsNone(saved["routes"][0]["training_executed_now"])

    def test_final_source_check_failure_preserves_a_receipt_and_does_not_claim_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root, cache = Path(temp) / "public", Path(temp) / "cache"
            root.mkdir(); cache.mkdir()
            result = dict(started_unix=0., source_files_before=bridge.SOURCE_PINS,
                          routes=[dict(run_id="synthetic", status="COMPLETE_VALID")])
            receipt = root / "execution_batches" / "synthetic_source_changed.json"
            bridge.finish_execution_result(result, root, cache, "never-acquired", receipt)
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "EXECUTION_ABORTED_SOURCE_CHANGE")
            self.assertFalse(saved["central_lease_receipt_exists"])
            self.assertNotIn("lease_receipt_sha256", saved)


if __name__ == "__main__":
    unittest.main()

