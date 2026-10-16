"""Explicit stdlib/fake-stage checks only; no real Task/model/child execution."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("tested_jena_seed_runner", Path(__file__).with_name("jena_seed_runner.py"))
api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api)


def plan():
    return dict(schema="PRIORITY_JENA_SECOND_SEED_v1", namespace=api.NAME, stage="POST_T_V_ONLY_DEVELOPMENT",
        status="SEALED_BY_ROOT", seed=92602, role_run_ids=api.IDS,
        conditions=[api.spec_for(family) for family in api.IDS], maximum_new_training_conditions=2,
        original_primary_split_ranges=copy.deepcopy(api.BOUNDS), materialized_prefix_end_exclusive=16772,
        new_T_target_operations_allowed=False, protected_tail_numeric_parse_allowed=False,
        upstream_priority_plan_sha256=api.UPSTREAM_PLAN_SHA, deadline_unix=api.HARD_DEADLINE,
        runner_sha256="toy_runner",
        resources=dict(CPU_threads=2, max_worker_RSS_bytes=3 * 1024**3, maximum_model_workers=2,
            storage_limit_bytes=30 * 1024**3, admission_reserve_bytes=256 * 1024**2,
            costs_must_not_overlap_training=True, ROOT_CPU_supervisor_required=True))


def stage(status="SEGMENT_COMPLETE", selected=1.):
    return dict(summary=dict(status=status, selected_epoch=selected), model="toy_state")


class SeedRunnerTests(unittest.TestCase):
    def test_default_is_input_write_science_and_child_free(self):
        before = {key: key in sys.modules for key in ("numpy", "torch", "data_contract", "psutil")}
        with mock.patch.object(api, "read", side_effect=AssertionError("no input")), mock.patch.object(api, "sha", side_effect=AssertionError("no hashes")), mock.patch.object(api, "execute", side_effect=AssertionError("no execution")), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(api.main([]), 0)
        result = json.loads(out.getvalue())
        self.assertEqual((result["input_reads"], result["writes"], result["actual_new_training_conditions"]), (0, 0, 0))
        self.assertEqual(before, {key: key in sys.modules for key in before})

    def test_exact_two_conditions_and_original_recipe_are_required(self):
        api.validate_proposal(plan())
        for field, value in (("seed", 92601), ("status", "DRAFT_PROPOSAL"), ("maximum_new_training_conditions", 3),
                             ("protected_tail_numeric_parse_allowed", True), ("materialized_prefix_end_exclusive", 20965)):
            changed = copy.deepcopy(plan()); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                api.validate_proposal(changed)
        changed = copy.deepcopy(plan()); changed["conditions"][0]["microbatch"] = 1
        with self.assertRaises(ValueError): api.validate_proposal(changed)
        changed = copy.deepcopy(plan()); changed["conditions"][1]["epochs"] = [.25]
        with self.assertRaises(ValueError): api.validate_proposal(changed)
        with self.assertRaises(ValueError): api.spec_for("LAG_CAUSAL")

    def test_fixed_original_stage_bounds_and_fresh_selected_refit(self):
        fit = types.SimpleNamespace(_stage=mock.Mock(side_effect=[stage(), stage()]),
            data_contract=types.SimpleNamespace(split_ranges=mock.Mock(side_effect=AssertionError("truncated split forbidden"))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = api.fit_stages(fit, api.spec_for("B2"), {"physical": "prefix_only"}, object(), {}, path / "public", path / "private", 900.)
        self.assertEqual(result["status"], "COMPLETE_STAGES")
        first, second = [call.args for call in fit._stage.call_args_list]
        self.assertEqual(first[2:5], (10482, [10482, 12579], [.25, 1., 3.]))
        self.assertEqual(second[2:5], (12579, None, [1.]))
        self.assertNotEqual(first[5], second[5])
        fit.data_contract.split_ranges.assert_not_called()

    def test_zero_selected_epoch_is_a_fresh_zero_update_refit(self):
        fit = types.SimpleNamespace(_stage=mock.Mock(side_effect=[stage(selected=0.), stage(selected=0.)]))
        result = api.fit_stages(fit, api.spec_for("B0"), {}, None, {}, Path("public"), Path("private"), 900.)
        self.assertEqual(result["selected_epoch"], 0.)
        self.assertEqual(fit._stage.call_args_list[1].args[4], [])

    def test_partial_search_or_refit_preserved_without_success_promotion(self):
        fit = types.SimpleNamespace(_stage=mock.Mock(return_value=stage("BUDGET_LIMIT")))
        result = api.fit_stages(fit, api.spec_for("B0"), {}, None, {}, Path("public"), Path("private"), 900.)
        self.assertEqual((result["status"], result["phase"], fit._stage.call_count), ("BUDGET_LIMIT_PRESERVED", "search", 1))
        fit = types.SimpleNamespace(_stage=mock.Mock(side_effect=[stage(), stage("BUDGET_LIMIT")]))
        result = api.fit_stages(fit, api.spec_for("B0"), {}, None, {}, Path("public"), Path("private"), 900.)
        self.assertEqual((result["status"], result["phase"]), ("BUDGET_LIMIT_PRESERVED", "refit"))

    def test_unregistered_selected_epoch_rejected_before_refit(self):
        fit = types.SimpleNamespace(_stage=mock.Mock(return_value=stage(selected=2.)))
        with self.assertRaises(ValueError):
            api.fit_stages(fit, api.spec_for("B0"), {}, None, {}, Path("public"), Path("private"), 900.)
        self.assertEqual(fit._stage.call_count, 1)

    def test_existing_run_or_prediction_is_preserved_and_other_case_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); public, cache = root / "public", root / "private"
            path = cache / "runs" / api.IDS["B0"] / "checkpoint.pt"
            path.parent.mkdir(parents=True); path.write_bytes(b"opaque_toy_checkpoint")
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): api.require_new_fit(public, cache, api.IDS["B0"])
            api.require_new_fit(public, cache, api.IDS["B2"])
            self.assertEqual(path.read_bytes(), before)

    def test_source_pin_mismatch_fails_before_upstream_or_science_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = plan()
            with mock.patch.object(api, "read", return_value=current), mock.patch.object(api, "sha", side_effect=["plan_pin", "changed_runner"]), mock.patch.object(api, "module", side_effect=AssertionError("must fail before import")):
                with self.assertRaises(ValueError): api.metadata(root, root, root, "plan_pin", object())

    def test_path_escape_and_wrong_saved_route_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): api.inside(Path(directory), "../escape")
        spec = api.spec_for("B0")
        receipt = dict(artifact_sha256="artifact", model_state_sha="state", refit_scaler_sha="scaler")
        context = dict(new_sources={"source": "pin"})
        manifest = dict(run_id=spec["run_id"], family="B0", seed=92602, split="V_select", artifact_sha256="artifact",
            config_sha=api.fingerprint(spec), source_sha=api.fingerprint(context["new_sources"]), scaler_sha="scaler",
            model_sha=api.fingerprint(dict(artifact_sha="artifact", model_state_sha="state")))
        api.validate_saved(manifest, receipt, spec, "plan", context)
        for key, value in (("run_id", api.IDS["B2"]), ("seed", 92601), ("artifact_sha256", "foreign"), ("split", "T_final")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                api.validate_saved(dict(manifest, **{key: value}), receipt, spec, "plan", context)

    def test_missing_quiet_cost_proof_fails_before_measurement(self):
        before = "measure_costs" in sys.modules
        with self.assertRaises(ValueError): api.cost(None, None, None, {}, Path("output"), 1, 900., None, None)
        self.assertEqual("measure_costs" in sys.modules, before)

    def test_cli_cannot_execute_without_explicit_root_trigger(self):
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(api, "execute", side_effect=AssertionError("no execution")):
            with self.assertRaises(SystemExit): api.main(["--execute"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
