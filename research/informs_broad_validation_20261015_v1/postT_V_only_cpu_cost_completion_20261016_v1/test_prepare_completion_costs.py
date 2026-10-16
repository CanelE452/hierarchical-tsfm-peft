"""Tiny opaque metadata fixtures; no actual Task, arrays, F or cost child."""
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
HELPER = HERE / "prepare_completion_costs.py"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class CompletionCostTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(HELPER.is_file(), "Missing per-condition CPU cost preparer")
        self.h = module(HELPER, "tiny_completion_cost_helper")
        self.v = self.h.load_verifier("required", HERE.parent)
        self.r = self.v.load_runner()

    def fixture(self, root, kind="required", index=0):
        test_name = "test_verify_" + ("required_minima" if kind == "required" else "family_minima") + "_outputs.py"
        fixtures = module(HERE.parent / self.h.KINDS[kind]["namespace"] / test_name, "opaque_" + kind + "_fixture")
        case = fixtures.RequiredVerifierTests() if kind == "required" else fixtures.FamilyVerifierTests()
        case.setUp()
        public, cache, plan, condition = case.fixture(root) if kind == "required" else case.fixture(root, index)
        verifier, runner = case.h, case.runner
        asset = root / "asset"
        asset.mkdir()
        plan.update(deadline_unix=1792222550, max_RSS_bytes=3 * 1024**3, storage_limit_bytes=30 * 1024**3,
            admission_reserve_bytes=256 * 1024**2, model_asset_relative_path="model/small")
        for name in self.h.REPLAY_SOURCES:
            plan["source39"].setdefault(name, name + "-sha")
        post, private = public / runner.NAME, cache / runner.NAME
        (post / "EXPERIMENT_REGISTRY.csv").write_bytes((post / "REGISTRY.csv").read_bytes())
        values, _, paths, bindings = verifier.prepare_inputs(runner, plan, public, cache, condition["run_id"], condition)
        binding = self.h.binding_record(condition, plan, values, paths, post, kind)
        channels, streams = len(binding["task_contract"]["columns"]), 6 * binding["width"]
        canonical = self.canonical(binding, plan)
        artifact = dict(route=dict(cache_relative_path=values["fit_receipt"]["artifact_cache_relative_path"],
            sha256=values["fit_receipt"]["artifact_sha256"]))
        run_id = condition["run_id"]
        replay_dir = post / "route_replays" / run_id
        replay_dir.mkdir(parents=True)
        replay_archive = private / "route_replays" / run_id / "V_subset.npz"
        replay_archive.parent.mkdir(parents=True)
        replay_archive.write_bytes(b"opaque replay bytes, never decoded")
        reference = dict(run_id=run_id, split="V_select", origins=[1, 2, 3, 4, 5, 6],
            route_sha=self.r.fingerprint(canonical), standard_manifest_sha256=self.h.sha(paths["manifest"]),
            standard_archive_sha256=values["manifest"]["archive_sha256"],
            artifact_bindings=artifact, expected_F_streams=streams, created_before_inference=True,
            source_sha=self.r.fingerprint({name: plan["source39"][name] for name in self.h.REPLAY_SOURCES}),
            selection_policy="first/middle/last plus 3 remaining origins from seed92601; sorted unique", created_unix=1.)
        replay = dict(status="PASS", run_id=run_id, actual_F_streams=streams, expected_F_streams=streams,
            F_physical_streams=streams, F_calls=math.ceil(streams / 128), atol=1e-5, rtol=1e-4, T_final_accessed=False,
            independent_training_rerun=False, model_size="small", origins=reference["origins"],
            scope="actual_physical_input_selected_checkpoint_V_subset_replay", selected_artifact_bindings=artifact,
            standard_manifest_public_relative_path=paths["manifest"].relative_to(post).as_posix(),
            standard_manifest_sha256=self.h.sha(paths["manifest"]), route_sha=reference["route_sha"],
            reference_sha=self.r.fingerprint(reference), cache_relative_path=replay_archive.relative_to(private).as_posix(),
            archive_sha256=self.h.sha(replay_archive))
        self.h.new(replay_dir / "PRE_INFERENCE_REFERENCE.json", reference)
        self.h.new(replay_dir / "REPLAY_RECEIPT.json", replay)
        proof_paths = {}
        for mode in ("rescore", "replay"):
            proof = post / "diagnostics" / ("tiny_" + mode)
            proof.mkdir(parents=True)
            scientific = (dict(status="PASS", n_origins=43, num_channels=channels, horizon=48,
                checkpoint_replay=False, atol=1e-9, rtol=1e-7) if mode == "rescore" else dict(replay, executed_now=True))
            result = dict(status="PASS", mode=mode, run_id=run_id, scientific_result=scientific,
                full_V_origin_count=43, horizon=48, num_channels=channels, plan_sha256=self.h.KINDS[kind]["plan_sha"],
                verifier_sha256=self.h.KINDS[kind]["verifier_sha"],
                original_source_registry_selection_T_and_captured_inputs_preserved=True,
                requested_device="cpu", CPU_threads=2, actual_Torch_CPU_threads=2, cuda_initialized=False,
                post_T_development=True, new_T_target_or_scoring=False, training_rerun=False,
                actual_fresh_CPU_replay_verified=mode == "replay", actual_checkpoint_forward_in_this_command=mode == "replay",
                replay_batch_size=6 if mode == "replay" else None, F_row_microbatch=128,
                saved_full_V_rescoring=mode == "rescore", replay_origin_count=6 if mode == "replay" else 0)
            self.h.new(proof / "INPUT_BINDINGS.json", dict(files_sha256_by_original_root=bindings,
                selected_artifact_sha256=values["fit_receipt"]["artifact_sha256"], full_three_epoch_search_verified=True))
            self.h.new(proof / "RESULT.json", result)
            if mode == "replay":
                extended = copy.deepcopy(bindings)
                extended["public"].update(verifier.capture_paths(runner, public,
                    [post / "EXPERIMENT_REGISTRY.csv", replay_dir / "REPLAY_RECEIPT.json", replay_dir / "PRE_INFERENCE_REFERENCE.json"]))
                extended["cache"].update(verifier.capture_paths(runner, cache, [replay_archive]))
                self.h.new(proof / "REPLAY_BINDINGS.json", dict(files_sha256_by_original_root=extended,
                    executed_now=True, requested_device="cpu", CPU_threads=2, replay_batch_size=6, original_F_row_microbatch=128))
            proof_paths[mode] = proof / "RESULT.json"
        return public, cache, asset, plan, condition, values, paths, bindings, binding, canonical, proof_paths

    def canonical(self, b, plan):
        base, manifest = b["base"], b["manifest"]
        return dict(run_id=b["run_id"], dataset=b["dataset"], family=b["family"], fold="primary", latent_k=b["width"],
            canonical_fit_end=b["task_contract"]["primary_split_ranges"]["A_stop"][1],
            foundation_sha=plan["foundation_tensor_sha256"], model_size="small",
            artifact_cache_relative_path=base["artifact_cache_relative_path"], artifact_sha256=base["artifact_sha256"],
            expected_prediction_model_sha=manifest["model_sha"], fit_receipt_public_relative_path=b["receipt_relative"],
            fit_receipt_sha256=b["receipt_sha"], standard_V_binding=dict(model_sha=manifest["model_sha"],
                archive_sha256=manifest["archive_sha256"], manifest_public_relative_path=b["manifest_relative"], manifest_sha256=b["manifest_sha"]))

    def test_default_is_no_read_no_write_no_science(self):
        before = {name: name in sys.modules for name in ("numpy", "torch", "psutil")}
        with mock.patch.object(self.h, "metadata", side_effect=AssertionError("no metadata requested")), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.h.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["metadata_files_read"], 0)
        self.assertEqual(result["actual_Task_array_model_or_child_operations"], 0)
        self.assertEqual(before, {name: name in sys.modules for name in before})

    def test_exact_ten_ids_and_lora_registered_zero_actual_twentyone(self):
        self.assertEqual(sum(len(item["ids"]) for item in self.h.KINDS.values()), 10)
        self.assertIn("postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1", self.h.KINDS["family"]["ids"])
        with tempfile.TemporaryDirectory() as td:
            *_, binding, canonical, _ = self.fixture(Path(td))
            result = self.h.cost_spec(canonical, "small", binding, {"foundation_tensor_sha256": "foundation"})
            self.assertEqual(result, dict(canonical, accuracy_run_id=binding["run_id"], row_microbatch=128))
            self.assertNotIn("accuracy_run_id", canonical)

    def test_cost_spec_rejects_changed_scientific_route(self):
        with tempfile.TemporaryDirectory() as td:
            public, cache, asset, plan, condition, values, paths, bindings, binding, canonical, proofs = self.fixture(Path(td))
            for mutation in (dict(latent_k=11), dict(family="LORA"), dict(canonical_fit_end=1), dict(g_route={}), dict(model_size="mini")):
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    self.h.cost_spec(dict(canonical, **mutation), "small", binding, plan)

    def test_actual_fresh_cpu_proofs_match_exact_current_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            p, q, a, plan, condition, values, paths, bindings, b, canonical, proofs = self.fixture(Path(td))
            captured = self.h.validate_proofs(self.v, self.r, plan, "required", condition, values, paths, bindings,
                p, q, proofs["rescore"], self.h.sha(proofs["rescore"]), proofs["replay"], self.h.sha(proofs["replay"]))
            self.assertIn(proofs["replay"].relative_to(p).as_posix(), captured["public"])
            reference = self.h.read(p / self.r.NAME / "route_replays" / condition["run_id"] / "PRE_INFERENCE_REFERENCE.json")
            self.h.validate_route_replay(self.r, plan, canonical, reference)

    def test_cached_cpu_wrong_source_and_changed_actual_binding_reject(self):
        changes = (dict(actual_fresh_CPU_replay_verified=False), dict(actual_checkpoint_forward_in_this_command=False),
            dict(requested_device="cuda"), dict(verifier_sha256="wrong"), dict(replay_batch_size=4), dict(cuda_initialized=True))
        for mutation in changes:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                p, q, a, plan, condition, values, paths, bindings, b, canonical, proofs = self.fixture(Path(td))
                value = self.h.read(proofs["replay"])
                value.update(mutation)
                proofs["replay"].write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.h.validate_proofs(self.v, self.r, plan, "required", condition, values, paths, bindings, p, q,
                        proofs["rescore"], self.h.sha(proofs["rescore"]), proofs["replay"], self.h.sha(proofs["replay"]))

    def test_source_dependency_tamper_is_rejected_before_import(self):
        with mock.patch.object(self.h, "sha", return_value="wrong"), mock.patch.object(self.h.importlib.util, "spec_from_file_location") as importer:
            with self.assertRaises(ValueError): self.h.load_verifier("required", HERE.parent)
            importer.assert_not_called()

    def test_cpu_proof_cannot_claim_fresh_when_scientific_replay_was_cached(self):
        with tempfile.TemporaryDirectory() as td:
            p, q, a, plan, condition, values, paths, bindings, b, canonical, proofs = self.fixture(Path(td))
            result = self.h.read(proofs["replay"])
            result["scientific_result"]["executed_now"] = False
            proofs["replay"].write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.h.validate_proofs(self.v, self.r, plan, "required", condition, values, paths, bindings, p, q,
                    proofs["rescore"], self.h.sha(proofs["rescore"]), proofs["replay"], self.h.sha(proofs["replay"]))

    def test_family_lora_keeps_registered_zero_but_actual_full_width(self):
        with tempfile.TemporaryDirectory() as td:
            p, q, a, plan, condition, values, paths, bindings, b, canonical, proofs = self.fixture(Path(td))
            condition = dict(run_id=self.h.KINDS["family"]["ids"][2], dataset="jena", family="LORA",
                registered_family="LORA_R8", latent_k=21, registered_latent_k=0, effective_F_width=21)
            contract = dict(plan["task_contract"], columns=[str(i) for i in range(21)])
            family_plan = dict(plan, task_contracts={"jena": contract})
            b = self.h.binding_record(condition, family_plan, values, paths, p / self.r.NAME, "family")
            self.assertEqual((b["width"], b["registered_latent_k"], b["family"], b["registered_family"]),
                (21, 0, "LORA", "LORA_R8"))
            canonical = self.canonical(b, family_plan)
            self.assertEqual(self.h.cost_spec(canonical, "small", b, family_plan)["latent_k"], 21)

    def test_completed_and_incomplete_ids_are_not_combined_into_one_gate(self):
        with tempfile.TemporaryDirectory() as td:
            p, q, a, plan, condition, values, paths, bindings, b, canonical, proofs = self.fixture(Path(td))
            other = p / self.r.NAME / "runs" / plan["conditions"][1]["run_id"]
            other.mkdir(parents=True)
            (other / "BASE_FIT_RECEIPT.json").write_text("{", encoding="utf-8")
            self.v.prepare_inputs(self.r, plan, p, q, condition["run_id"], condition)
            self.h.validate_proofs(self.v, self.r, plan, "required", condition, values, paths, bindings, p, q,
                proofs["rescore"], self.h.sha(proofs["rescore"]), proofs["replay"], self.h.sha(proofs["replay"]))

    def test_protocol_argv_is_original_single_worker_not_balanced_or_training(self):
        argv = self.h.worker_arguments("one/spec.json", "costs/token/B1.json", 1, "model/small")
        self.assertEqual(argv[:6], ["{PYTHON}", "-X", "utf8", "-u", "-B", "{ORIGINAL_PUBLIC_ROOT}/measure_costs.py"])
        for flag, value in (("--device", "cpu"), ("--warmups", "10"), ("--timed-trials", "20"), ("--stage-trials", "20")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        with self.assertRaises(ValueError): self.h.worker_arguments("a", "b", 4, "m")

    def test_execute_fake_runtime_writes_two_specs_and_no_cost(self):
        with tempfile.TemporaryDirectory() as td:
            p, q, a, plan, condition, values, paths, bindings, b, canonical, proofs = self.fixture(Path(td))
            state = (self.v, self.r, plan, condition)
            calls = []
            runtime = types.SimpleNamespace(task=lambda *args: (calls.append("task") or object(), {"tiny_metadata": True}),
                route=lambda *args: (calls.append("route") or copy.deepcopy(canonical), "small"))
            output = p / self.h.NAME / "diagnostics" / "one"
            with mock.patch.object(self.h, "metadata", return_value=state), mock.patch.object(self.r, "environment"), \
                    mock.patch.object(self.h, "limits", return_value=contextlib.nullcontext()):
                result = self.h.execute(p, q, a, "required", condition["run_id"], self.h.KINDS["required"]["plan_sha"],
                    proofs["rescore"], self.h.sha(proofs["rescore"]), proofs["replay"], self.h.sha(proofs["replay"]),
                    time.time() + 120, output, _runtime=runtime)
            self.assertEqual(result["status"], "PREPARED_NOT_EXECUTED_CPU_COST_WORKERS")
            self.assertEqual(result["worker_count"], 2)
            self.assertEqual(calls, ["task", "route"])
            self.assertEqual(result["actual_cost_trials"], 0)
            self.assertEqual(result["actual_subprocesses_started"], 0)
            self.assertEqual([job["expected_F_streams_per_trial"] for job in result["jobs"]], [6, 48])
            self.assertTrue((output / "COST_EXECUTION_PLAN.json").is_file())

    def test_all_four_family_fixtures_reuse_actual_validators_and_widths(self):
        for index in range(4):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as td:
                p, q, a, plan, condition, values, paths, bindings, b, canonical, proofs = self.fixture(Path(td), "family", index)
                verifier = self.h.load_verifier("family", HERE.parent)
                runner = verifier.load_runner()
                state = (verifier, runner, plan, condition)
                runtime = types.SimpleNamespace(task=lambda *args: (object(), {"tiny_metadata": True}),
                    route=lambda *args: (copy.deepcopy(canonical), "small"))
                output = p / self.h.NAME / "diagnostics" / "one"
                with mock.patch.object(self.h, "metadata", return_value=state), mock.patch.object(runner, "environment"), \
                        mock.patch.object(self.h, "limits", return_value=contextlib.nullcontext()):
                    result = self.h.execute(p, q, a, "family", condition["run_id"], self.h.KINDS["family"]["plan_sha"],
                        proofs["rescore"], self.h.sha(proofs["rescore"]), proofs["replay"], self.h.sha(proofs["replay"]),
                        time.time() + 120, output, _runtime=runtime)
                self.assertEqual(result["status"], "PREPARED_NOT_EXECUTED_CPU_COST_WORKERS", result)
                self.assertEqual([job["expected_F_streams_per_trial"] for job in result["jobs"]],
                    [b["width"], 8 * b["width"]])
                for job in result["jobs"]:
                    self.assertEqual(job["lora_parity_and_exact_frozen_restore_must_be_confirmed_from_actual_worker"], index == 2)
                    self.assertEqual(job["registered_latent_k"], 0 if index == 2 else b["width"])

    def test_existing_output_and_bad_proof_never_call_science(self):
        with tempfile.TemporaryDirectory() as td:
            p, q, a, plan, condition, values, paths, bindings, b, canonical, proofs = self.fixture(Path(td))
            state = (self.v, self.r, plan, condition)
            runtime = types.SimpleNamespace(task=mock.Mock(), route=mock.Mock())
            output = p / self.h.NAME / "diagnostics" / "existing"
            output.mkdir(parents=True)
            (output / "keep").write_bytes(b"original")
            args = (p, q, a, "required", condition["run_id"], self.h.KINDS["required"]["plan_sha"],
                proofs["rescore"], "0" * 64, proofs["replay"], self.h.sha(proofs["replay"]), time.time() + 120)
            with mock.patch.object(self.h, "metadata", return_value=state), mock.patch.object(self.r, "environment"), \
                    mock.patch.object(self.h, "limits", return_value=contextlib.nullcontext()):
                with self.assertRaises(FileExistsError): self.h.execute(*args, output, _runtime=runtime)
                result = self.h.execute(*args, output.with_name("new_failed"), _runtime=runtime)
            self.assertEqual(result["status"], "FAILED_PRESERVED")
            self.assertEqual(result["worker_count"], 0)
            self.assertEqual((output / "keep").read_bytes(), b"original")
            runtime.task.assert_not_called()
            runtime.route.assert_not_called()

    def test_changed_selected_input_during_catalog_prevents_job_emission(self):
        with tempfile.TemporaryDirectory() as td:
            p, q, a, plan, condition, values, paths, bindings, b, canonical, proofs = self.fixture(Path(td))
            state = (self.v, self.r, plan, condition)
            def route(*args):
                paths["artifact"].write_bytes(b"changed by fake runtime")
                return copy.deepcopy(canonical), "small"
            runtime = types.SimpleNamespace(task=lambda *args: (object(), {"tiny": True}), route=route)
            output = p / self.h.NAME / "diagnostics" / "source_changed"
            with mock.patch.object(self.h, "metadata", return_value=state), mock.patch.object(self.r, "environment"), \
                    mock.patch.object(self.h, "limits", return_value=contextlib.nullcontext()):
                result = self.h.execute(p, q, a, "required", condition["run_id"], self.h.KINDS["required"]["plan_sha"],
                    proofs["rescore"], self.h.sha(proofs["rescore"]), proofs["replay"], self.h.sha(proofs["replay"]),
                    time.time() + 120, output, _runtime=runtime)
            self.assertEqual(result["status"], "FAILED_PRESERVED")
            self.assertEqual(result["worker_count"], 0)
            self.assertFalse(result["original_source39_registry_scores_cost_selection_T_and_captured_inputs_preserved"])
            self.assertEqual(result["actual_cost_trials"], 0)


if __name__ == "__main__":
    unittest.main()
