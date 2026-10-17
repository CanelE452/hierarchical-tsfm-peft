"""Small invented stdlib fixtures only; no actual N6 assets or execution."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("audit_raw_g_v2.py")
spec = importlib.util.spec_from_file_location("test_raw_g_auditor", SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fit_receipt(family, epoch=None):
    value = dict(status="COMPLETE_VALID", run_id=m.RUNS[family], family=family, seed=92601, actual_C=32, L=512, H=12, registered_latent_k=0,
        A50_end=8928, A60_end=10713, full_parent_C=170, full_parent_identity_inherited=True, new_T_numeric_values_parsed=0,
        independent_generalization_claimed=False, old_C170_training_budget_matched=False, PCA_fit_time_separately_recorded=False,
        **{key: "a" * 64 for key in ("artifact_sha256", "model_state_sha", "refit_scaler_sha", "actual_subset_prefix_sha256", "data_sha256")})
    if family == "RAW":
        value.update(selected_epoch=None, search_updates=0, refit_updates=0, search_exposures=0, refit_exposures=0, training_status="NO_TRAINING",
            artifact_kind="RAW_METADATA_IDENTITY_NO_TRAINED_ADAPTER", trained_adapter_checkpoint_exists=False)
    else:
        value.update(selected_epoch=epoch, search_updates=1577, search_exposures=25215, refit_exposures=m.math.ceil(epoch * 10190),
            refit_updates=m.math.ceil(m.math.ceil(epoch * 10190) / 16), source_stage_API="original_fit_base._stage", artifact_kind="TRAINED_SUBSET_BUNDLE",
            training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch == 3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED",
            initialization_by_stage={key: dict(fingerprints=dict(mean="a" * 64, std="b" * 64, observed_counts="c" * 64, basis=None, pca_center=None)) for key in ("A50", "A60")})
    return value


def trial_rows(family, batch):
    stages = ("CPU_preprocess", "host_to_device", "F" if family == "RAW" else "G_or_direct", "inverse_and_CPU")
    values = []; counts = m.counts(family, [batch])
    for phase, n in (("warmup", 10), ("timed", 20), ("stage_probe", 20)):
        for i in range(n):
            values.append(dict(run_id=m.RUNS[family], phase=phase, trial=str(i), order_position="0", batch_size=str(batch),
                whole_seconds=str((i + 1) / 100), F_logical_streams=str(counts["streams"]), F_physical_streams=str(counts["physical_streams"]), F_calls=str(counts["physical_calls"]),
                prediction_sha256="a" * 64, row_microbatch="128", stage_instrumentation=str(phase == "stage_probe"),
                **{"stage_" + key + "_seconds": "0.001" if phase == "stage_probe" else "" for key in stages}))
    return values


class Tests(unittest.TestCase):
    def test_default_zero_io_and_lazy_no_model_imports(self):
        output = io.StringIO()
        with patch("builtins.open", side_effect=AssertionError("no input")), patch.object(m, "load_helpers", side_effect=AssertionError("no helper read")), contextlib.redirect_stdout(output):
            self.assertEqual(m.main([]), 0)
        self.assertEqual(m.json.loads(output.getvalue())["input_reads"], 0)
        tree = m.Path(SOURCE).read_text(encoding="utf-8")
        self.assertNotIn("import numpy", tree); self.assertNotIn("import torch", tree)

    def test_counts_distinguish_parity_full_row128_and_cost(self):
        self.assertEqual(m.counts("RAW", [4] * 37), dict(streams=4736, physical_streams=4736, physical_calls=37))
        self.assertEqual(m.counts("RAW", [4, 2]), dict(streams=192, physical_streams=192, physical_calls=2))
        self.assertEqual(m.counts("RAW", [8], full=True)["physical_calls"], 1)
        self.assertEqual(m.counts("RAW", [8])["physical_calls"], 2)
        self.assertEqual(m.counts("DIRECT_G", [8]), dict(streams=0, physical_streams=0, physical_calls=0))

    def test_bounded_post_soft_deadline_is_not_forced_to_hard_deadline(self):
        m.validate_deadline(["python", "--deadline", str(m.DEADLINE - 100)], float(m.DEADLINE - 100))
        m.validate_deadline(["python", "--deadline", str(m.DEADLINE)], float(m.DEADLINE))
        for command, recorded in ((["--deadline", str(m.DEADLINE + 1)], m.DEADLINE + 1), (["--deadline", str(m.DEADLINE)], m.DEADLINE - 1), (["--deadline", "nan"], 1), (["--deadline", "1"], True)):
            with self.assertRaises(ValueError): m.validate_deadline(command, recorded)

    def test_future_null_sha_descriptor_never_reads_input(self):
        class Inputs:
            def json(self, *args): raise AssertionError("Pending SHA cannot open metadata")
        with self.assertRaises(ValueError): m.bound_json(Inputs(), dict(root="cache", relative_path="future.json", sha256=None), "cache")

    def test_missing_native_receipt_is_honest_absence_not_a_fabricated_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            class Inputs:
                def path(self, category, name): return root / name
                def json(self, *args): raise AssertionError("Missing native file is not parsed or fabricated")
            absent = []
            self.assertIsNone(m.native_json(Inputs(), None, "RESULT.json", absent))
            self.assertEqual(absent, ["RESULT.json"])
            (root / "RESULT.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError): m.native_json(Inputs(), None, "RESULT.json", [])

    def test_band_stats_use_observed_counts_not_full_mask_assumption(self):
        channels = ["sensor" + str(i) for i in range(32)]; rows = []; observed = 0
        for origin in (12499, 12511):
            for channel in channels:
                for a, b in m.BANDS:
                    n = 0 if origin == 12499 and channel == channels[0] and a == 0 else b - a
                    observed += n
                    rows.append(dict(run_id=m.RUNS["RAW"], split="V_select", cohort=m.COHORT, origin=str(origin), channel=channel,
                        horizon_start=str(a), horizon_end=str(b), count=str(n), sse=str(n * 4), sae=str(n * 2)))
        with patch.object(m, "ORIGINS", (12499, 12511)):
            actual = m.aggregate_statistics(rows, m.RUNS["RAW"], channels, observed)
            self.assertEqual((actual["mse"], actual["mae"]), (4, 2))
            self.assertNotEqual(actual["n_observed"], 2 * 32 * 12)
            for bad in (rows[:-1], rows + rows[:1]):
                with self.assertRaises(ValueError): m.aggregate_statistics(bad, m.RUNS["RAW"], channels, observed)
            with self.assertRaises(ValueError): m.aggregate_statistics(rows, m.RUNS["RAW"], channels, observed + 1)

    def test_native_family_four_stages_and_all_fifty_trials(self):
        for family in m.RUNS:
            for batch in (1, 8):
                rows = trial_rows(family, batch); value = m.aggregate_trials(rows, m.RUNS[family], family, batch)
                self.assertEqual(value["trial_rows"], 50); self.assertEqual(len(value["summaries"]), 2)
                self.assertEqual(value["F_calls"], 50 * m.counts(family, [batch])["physical_calls"])
                with self.assertRaises(ValueError): m.aggregate_trials(rows[:-1], m.RUNS[family], family, batch)
                bad = copy.deepcopy(rows); bad[-1]["stage_P_E_seconds"] = "0"
                with self.assertRaises(ValueError): m.aggregate_trials(bad, m.RUNS[family], family, batch)
                bad = copy.deepcopy(rows); bad[0]["F_calls"] = "999"
                with self.assertRaises(ValueError): m.aggregate_trials(bad, m.RUNS[family], family, batch)
                bad = copy.deepcopy(rows); bad[0]["stage_inverse_and_CPU_seconds"] = "0"
                with self.assertRaises(ValueError): m.aggregate_trials(bad, m.RUNS[family], family, batch)

    def test_g_selected_zero_and_segment_rounding(self):
        for epoch in (0., .25, 1., 3.):
            fit = fit_receipt("DIRECT_G", epoch)
            search = dict(status="SEGMENT_COMPLETE", completed_epochs=[0., .25, 1., 3.], train_origins=8405, exposures=25215, updates=1577)
            refit = dict(status="SEGMENT_COMPLETE", completed_epochs=[0.] + ([epoch] if epoch else []), train_origins=10190,
                exposures=fit["refit_exposures"], updates=fit["refit_updates"])
            m.validate_training(fit, search, refit)
            bad = copy.deepcopy(fit); bad["search_updates"] = 1576
            with self.assertRaises(ValueError): m.validate_training(bad, search, refit)
            bad = copy.deepcopy(fit); bad["initialization_by_stage"]["A50"]["fingerprints"]["basis"] = "a" * 64
            with self.assertRaises(ValueError): m.validate_training(bad, search, refit)
        m.validate_training(fit_receipt("RAW"))
        bad = fit_receipt("RAW"); bad["refit_updates"] = 1
        with self.assertRaises(ValueError): m.validate_training(bad)

    def test_frozen_raw_and_absent_g_are_distinct(self):
        m.validate_foundation_state(dict(foundation_instances_created=1, F_final_state_sha256="a" * 64), "RAW", "a" * 64)
        m.validate_foundation_state(dict(foundation_instances_created=0), "DIRECT_G", "a" * 64)
        with self.assertRaises(ValueError): m.validate_foundation_state(dict(foundation_instances_created=0, F_final_state_sha256="a" * 64), "DIRECT_G", "a" * 64)
        with self.assertRaises(ValueError): m.validate_foundation_state(dict(foundation_instances_created=1, F_final_state_sha256="b" * 64), "RAW", "a" * 64)

    def test_raw_parity_cannot_reuse_b1_for_b8(self):
        identity = dict(run_id=m.RUNS["RAW"], manifest_sha256="a" * 64)
        for batch in (1, 8):
            variants = [dict(row_microbatch="full", F_counters=m.counts("RAW", [batch], full=True)), dict(row_microbatch=128, F_counters=m.counts("RAW", [batch]))]
            value = dict(identity, status="PASS_RAW_ROW128_VS_FULL", batch_size=batch, all_passed=True, actual_raw_forward=True, actual_optimizer_updates=0,
                historical_original_Raw_failures_resolved=False, new_T_numeric_values_parsed=0, per_variant_F_counters=variants,
                F_counters={k: sum(v["F_counters"][k] for v in variants) for k in ("streams", "physical_streams", "physical_calls")},
                original_raw_row_parity=dict(all_passed=True, atol=1e-5, rtol=1e-4, variants=[dict(row_microbatch=v["row_microbatch"], status="PASS") for v in variants]))
            m.validate_parity(value, batch, identity)
            with self.assertRaises(ValueError): m.validate_parity(value, 8 if batch == 1 else 1, identity)
            bad = copy.deepcopy(value); bad["per_variant_F_counters"][0]["F_counters"]["physical_calls"] = 2
            with self.assertRaises(ValueError): m.validate_parity(bad, batch, identity)

    def test_rescore_is_saved_full_v_and_replay_is_only_six(self):
        identity = dict(run_id=m.RUNS["DIRECT_G"], artifact_sha256="a" * 64)
        value = dict(identity, status="PASS_INDEPENDENT_FULL_V_FP64", full_V_origins=148, actual_C=32, H=12, n_observed=1234, public_stats_rows=14208,
            saved_predictions_rescored=True, new_model_inference=False, protected_T_target_operations=0, torch_imported=False, new_T_numeric_values_parsed=0)
        m.validate_validation(value, "rescore", "DIRECT_G", identity, 1234)
        bad = dict(value, n_observed=56832)
        with self.assertRaises(ValueError): m.validate_validation(bad, "rescore", "DIRECT_G", identity, 1234)
        replay = dict(identity, status="PASS_FRESH_CPU6_REPLAY", executed_now=True, microbatches=[4, 2], replay_selection_seed=92601,
            F_counters=m.counts("DIRECT_G", [4, 2]), atol=1e-5, rtol=1e-4, full_V_reinference_claimed=False, foundation_instances_created=0,
            origins=[m.ORIGINS[i] for i in (0, 1, 2, 74, 146, 147)], new_T_numeric_values_parsed=0)
        m.validate_validation(replay, "replay", "DIRECT_G", identity, 1234)
        with self.assertRaises(ValueError): m.validate_validation(dict(replay, full_V_reinference_claimed=True), "replay", "DIRECT_G", identity, 1234)

    def test_twelve_expected_native_keys_only(self):
        self.assertEqual(len(m.expected_keys()), 12)
        self.assertEqual(sum(k[1] == "cost" for k in m.expected_keys()), 4)
        self.assertFalse(any("B2" in k[0] for k in m.expected_keys()))

    def test_current_guard_closure_preserves_spec_sha_and_effective_limits(self):
        fixture_root = Path(tempfile.gettempdir()) / "invented_raw_g_guard"
        process = dict(id="invented", command=["python", "invented.py"], cwd=str(fixture_root), max_seconds=10)
        spec = dict(schema="CPU_GROUP_SUPERVISOR_v1", deadline_unix=m.DEADLINE, max_each_RSS=3221225472, max_group_RSS=3221225472,
            poll_seconds=1, terminate_grace_seconds=1, kill_grace_seconds=1, asset_root=str(fixture_root / "assets"), cache_root=str(fixture_root / "cache"), output_root=str(fixture_root / "cache" / "guard"), processes=[process])
        owner = dict(pid=12345, create_time=1.)
        child = dict(id="invented", returncode=0, start_unix=2., end_unix=3., observed_tree_join_confirmed=True, process_tree_errors=[], remaining_owned_processes=[], owned_processes=[owner], root=owner, sampled_peak_RSS_bytes=123)
        guard = dict(schema="CPU_GROUP_SUPERVISOR_RESULT_v1", status="COMPLETE", helper_sha256=m.GUARD_SHA, helper_unchanged=True, spec_file_sha256="a" * 64,
            supervisor_failure_class=None, remaining_owned_processes=[], processes=[child], effective_spec=copy.deepcopy(spec), sampled_group_peak_RSS_bytes=123)
        closure = dict(schema="ROOT_OPAQUE_METADATA_OWNER_ABSENCE_v2", status="VERIFIED_STRONG_OBSERVED_TREE_CLOSURE_AND_CURRENT_ABSENCE", guard_sha256="b" * 64,
            spec_sha256="a" * 64, actual_PID_creation_time_check=True, remaining_exact_owners=[], errors=[], absent_identities=[owner], checked_unix=4.)
        values = dict(spec=spec, guard=guard, closure=closure)
        class Inputs:
            def json(self, root, name, digest):
                return values[name]
        binding = {k: dict(root="cache", relative_path=k, sha256=v * 64) for k, v in (("spec", "a"), ("guard", "b"), ("closure", "c"))}
        binding["workers"] = [dict(id="invented")]
        with patch.object(m.time, "time", return_value=5.):
            m.closed_binding(Inputs(), binding)
            closure["guard_sha256"] = "d" * 64
            with self.assertRaises(ValueError): m.closed_binding(Inputs(), binding)
            closure["guard_sha256"] = "b" * 64
            guard["effective_spec"]["max_each_RSS"] = 6442450944
            with self.assertRaises(ValueError): m.closed_binding(Inputs(), binding)

    def test_native_fit_group_sha_and_lowercase_started_are_bound(self):
        refs = {key: dict(root="cache", relative_path=key, sha256=m.fingerprint(key)) for key in
            ("dispatcher_source", "post_source", "dispatch", "post", "admission", "predecessor", "preparation", "dispatch_started", "post_started")}
        for key, (name, digest) in m.ORCHESTRATION_PINS.items():
            refs[key] = dict(root="cache", relative_path=name, sha256=digest)
        for key, name in (("dispatch", "RESULT.json"), ("post", "POST_RESULT.json"), ("dispatch_started", "STARTED.json"), ("post_started", "POST_STARTED.json"), ("preparation", "PLAN_DRAFT.json")):
            refs[key]["relative_path"] = m.PREPARATION + name
        execution = dict(refs, plan_sha256="1" * 64)
        fit_spec, fit_guard = m.fingerprint("fit-spec"), m.fingerprint("fit-guard")
        common = dict(error=None, plan_sha256="1" * 64, pins_before_after_unchanged=True, changed_or_unreadable_pinned_paths=[], original_worker_deadline=m.DEADLINE, original_campaign_deadline=1792223150,
            dispatcher_source_sha256=refs["dispatcher_source"]["sha256"], source39_immutable9_count=48, actual_C=32, full_parent_C=170, B0_B2_automatically_started=False, whole_MASTER_complete=False)
        dispatch = dict(common, schema="PEMS_RAW_G_TWO_FIT_RESULT_v1", dispatch_spec_sha256=fit_spec, dispatch_guard_sha256=fit_guard,
            admission_sha256=refs["admission"]["sha256"], predecessor_sha256=refs["predecessor"]["sha256"], preparation_sha256=refs["preparation"]["sha256"],
            strong_observed_tree_closure_verified=True, original_four_conditions_preserved=True, conditions=[], skipped=[])
        post = dict(common, schema="PEMS_RAW_G_POST_RESULT_v1", fit_dispatch_result_sha256=refs["dispatch"]["sha256"], fit_dispatch_SPEC_sha256=fit_spec,
            actual_fit_guard_sha256=fit_guard, post_source_sha256=refs["post_source"]["sha256"], new_training=0, steps=[], skipped=[])
        records = {}
        for index, key in enumerate(sorted(m.expected_keys(), key=str)):
            run_id, mode, batch = key; family = next(f for f, r in m.RUNS.items() if r == run_id)
            evidence = dict(id="invented" + str(index), spec_sha256=fit_spec if mode == "fit" else m.fingerprint([index, "spec"]),
                guard_sha256=fit_guard if mode == "fit" else m.fingerprint([index, "guard"]), outer_sha256=m.fingerprint([index, "outer"]), STARTED_sha256=m.fingerprint([index, "started"]))
            records[key] = dict(evidence=evidence)
            row = dict(run_id=run_id, family=family, mode=mode, batch=batch, id=evidence["id"], outer_sha256=evidence["outer_sha256"], started_sha256=evidence["STARTED_sha256"])
            if mode == "fit":
                row["worker_science_complete"] = True; dispatch["conditions"].append(row)
            else:
                row.update(success=True, spec_sha256=evidence["spec_sha256"], guard_sha256=evidence["guard_sha256"], strong_observed_tree_closure_verified=True); post["steps"].append(row)
        post["attempts"] = copy.deepcopy(post["steps"])
        values = dict(dispatch=dispatch, post=post, preparation=dict(status="PREP_ONLY_NOT_ADMITTED", run_ids=[m.RUNS["RAW"], m.RUNS["DIRECT_G"]]),
            admission=dict(schema="PEMS_RAW_G_ADMISSION_v1", status="SEALED_AND_METADATA_VERIFIED_NOT_EXECUTED", plan_sha256="1" * 64,
                dispatcher_sha256=refs["dispatcher_source"]["sha256"], preparation_sha256=refs["preparation"]["sha256"], predecessor_sha256=refs["predecessor"]["sha256"],
                original_four_conditions_preserved=True, B0_B2_automatically_started=False, original_worker_deadline=m.DEADLINE, budgets_reinitialized_or_extended=False,
                new_T_numeric_parsed=0, GPU_added_seconds=0, run_ids=sorted(m.RUNS.values())),
            predecessor=dict(schema="PEMS_RAW_G_N5_PREDECESSOR_v1", status="ROOT_RECONCILED_CLOSED_REPORTED_AND_PUBLISHED", remote_tree_and_HTTP_verified=True, whole_MASTER_complete=False),
            dispatch_started=dict(script_sha256=refs["dispatcher_source"]["sha256"], plan_sha256="1" * 64, admission_sha256=refs["admission"]["sha256"], predecessor_sha256=refs["predecessor"]["sha256"], preparation_sha256=refs["preparation"]["sha256"]),
            post_started=dict(script_sha256=refs["post_source"]["sha256"], dispatcher_source_sha256=refs["dispatcher_source"]["sha256"], plan_sha256="1" * 64,
                fit_dispatch_result_sha256=refs["dispatch"]["sha256"], fit_dispatch_SPEC_sha256=fit_spec, actual_fit_guard_sha256=fit_guard, new_training=0))
        class Inputs:
            def capture(self, root, name, digest): pass
            def json(self, root, name, digest): return values[next(key for key, ref in refs.items() if ref["relative_path"] == name)]
        m.bind_native_execution(Inputs(), execution, records)
        self.assertTrue(all(r["native_wrapper_success"] for r in records.values()))
        saved_post, saved_records = copy.deepcopy(post), copy.deepcopy(records)
        failed = next(r for r in post["steps"] if r["mode"] == "raw_parity" and r["batch"] == 1)
        failed["success"] = False
        missing = (m.RUNS["RAW"], "cost", 1); missing_id = records.pop(missing)["evidence"]["id"]
        post["steps"] = [r for r in post["steps"] if r["id"] != missing_id]
        post["attempts"] = copy.deepcopy(post["steps"])
        post["skipped"] = [dict(run_id=m.RUNS["RAW"], mode="cost", batch=1, actual_guard_created=False, reason="Invented same-batch parity failure")]
        m.bind_native_execution(Inputs(), execution, records)
        self.assertFalse(records[(m.RUNS["RAW"], "raw_parity", 1)]["native_wrapper_success"])
        self.assertTrue(records[(m.RUNS["DIRECT_G"], "cost", 1)]["native_wrapper_success"])
        post.clear(); post.update(saved_post); records.clear(); records.update(saved_records)
        dispatch["dispatch_guard_sha256"] = None
        with self.assertRaises(ValueError): m.bind_native_execution(Inputs(), execution, records)
        dispatch["dispatch_guard_sha256"] = fit_guard
        post["steps"][0]["started_sha256"] = "0" * 64
        with self.assertRaises(ValueError): m.bind_native_execution(Inputs(), execution, records)


if __name__ == "__main__":
    unittest.main()
