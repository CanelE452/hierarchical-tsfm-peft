"""Synthetic tests only; no campaign receipts, arrays, models, or child workers."""
import contextlib
import copy
import ast
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
from types import SimpleNamespace
import unittest


PATH = Path(__file__).with_name("audit_new_cpu_costs.py")
spec = importlib.util.spec_from_file_location("new_cpu_cost_audit_under_test", PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def summaries(rows, bindings):
    result = []
    for phase in ("timed", "stage_probe"):
        group = [row for row in rows if row["phase"] == phase]
        if not group:
            continue
        values = sorted(row["whole_seconds"] for row in group)
        position = (len(values) - 1) * .9
        p90 = values[math.floor(position)] + (values[math.ceil(position)] - values[math.floor(position)]) * (position % 1)
        result.append(dict(run_id="condition", condition_id="condition", accuracy_run_id="accuracy",
            phase=phase, batch_size=1, trials=len(group), median_seconds=statistics.median(values),
            mean_seconds=statistics.mean(values), min_seconds=min(values), max_seconds=max(values),
            p90_seconds=p90, stage_instrumentation=phase == "stage_probe", stage_sum_equals_whole_claimed=False))
    return result


def fixture(family="B0", batch=1, k=6, channels=21):
    width = channels if family == "RAW" else 0 if family in {"DIRECT_G", "RAW_RIDGE"} else k
    streams = batch * width
    rows = []
    for phase, count in (("warmup", 10), ("timed", 20), ("stage_probe", 20)):
        for trial in range(count):
            rows.append(dict(run_id="condition", phase=phase, trial=str(trial), order_position="0",
                batch_size=str(batch), whole_seconds=str(.001 + trial / 100000), F_logical_streams=str(streams),
                F_physical_streams=str(streams), F_calls=str(math.ceil(streams / 128)),
                prediction_sha256="a" * 64, row_microbatch="128", stage_instrumentation=str(phase == "stage_probe"),
                stage_preprocess_seconds=".0001" if phase == "stage_probe" else ""))
    route = dict(run_id="condition", family=family, latent_k=k, row_microbatch=128, accuracy_run_id="accuracy")
    receipt = dict(status="COMPLETE_PROTOCOL", run_id="condition", batch_size=batch, C=channels,
        warmups=10, timed_trials=20, stage_trials=20, row_microbatch=128,
        accuracy_bindings={"condition": dict(accuracy_run_id="accuracy", cost_route_spec=route)})
    typed = audit.typed_trials(rows)
    receipt["summaries"] = summaries(typed, receipt["accuracy_bindings"])
    for item in receipt["summaries"]:
        item["batch_size"] = batch
    return receipt, rows, streams


def expected(spec, batch, channels):
    if spec["family"] in {"DIRECT_G", "RAW_RIDGE"}:
        return 0
    return batch * (channels if spec["family"] == "RAW" else spec["latent_k"])


def original_functions(filename, names, environment):
    """Execute only original function definitions on synthetic metadata."""
    source = PATH.parent.parent / filename
    parsed = ast.parse(source.read_text(encoding="utf-8-sig"))
    definitions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != set(names):
        raise AssertionError("Original function signature disappeared")
    scope = dict(environment)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


class AuditNewCPUCostsTests(unittest.TestCase):
    def validate(self, receipt, rows):
        def summarized(typed, bindings):
            result = summaries(typed, bindings)
            for item in result:
                item["batch_size"] = receipt["batch_size"]
            return result
        return audit.validate_trials(receipt, rows, expected, summarized)

    def test_exact_fifty_rows_and_all_supported_counter_widths(self):
        for family, batch, k, c in (("B0", 1, 6, 21), ("LAG_BIDIR", 8, 6, 21),
                ("RAW", 8, 0, 21), ("DIRECT_G", 1, 0, 21), ("RAW_RIDGE", 8, 0, 21),
                ("PCA_FIXED_BOLT", 8, 128, 170), ("NESTED_PCA", 8, 11, 11)):
            with self.subTest(family=family):
                receipt, rows, streams = fixture(family, batch, k, c)
                result = self.validate(receipt, rows)
                self.assertEqual(result["actual_rows"], 50)
                self.assertEqual(result["expected_F_streams_per_trial"], streams)
                self.assertEqual(result["expected_F_calls_per_trial"], math.ceil(streams / 128))
                self.assertEqual(result["phase_counts"], dict(warmup=10, timed=20, stage_probe=20))

    def test_duplicate_missing_trial_and_wrong_physical_calls_are_rejected(self):
        receipt, rows, _ = fixture()
        variants = [rows[:-1], rows + [rows[0]], copy.deepcopy(rows), copy.deepcopy(rows)]
        variants[2][11]["trial"] = variants[2][10]["trial"]
        variants[3][0]["F_calls"] = "2"
        for changed in variants:
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                self.validate(receipt, changed)

    def test_nonfinite_timer_bad_boolean_and_unexpected_stage_are_rejected(self):
        receipt, rows, _ = fixture()
        for column, value, index in (("whole_seconds", "nan", 0), ("whole_seconds", "-1", 0),
                ("stage_preprocess_seconds", "inf", 30), ("stage_instrumentation", "true", 0),
                ("stage_preprocess_seconds", ".1", 0)):
            changed = copy.deepcopy(rows)
            changed[index][column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.validate(receipt, changed)

    def test_saved_summary_mean_p90_or_flags_mismatch_is_rejected(self):
        receipt, rows, _ = fixture()
        for key, value in (("mean_seconds", 9.), ("p90_seconds", 9.), ("trials", 19),
                ("stage_sum_equals_whole_claimed", True)):
            changed = copy.deepcopy(receipt)
            changed["summaries"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(changed, rows)

    def test_failed_or_partial_receipt_can_never_validate_as_complete(self):
        receipt, rows, _ = fixture()
        for status in ("FAILED", "SMOKE_OR_PARTIAL_COST_PROTOCOL"):
            changed = dict(receipt, status=status)
            with self.assertRaisesRegex(ValueError, "NOT_COMPLETE"):
                self.validate(changed, rows)
        with self.assertRaises(ValueError):
            self.validate(dict(receipt, warmups=3), rows)

    def test_descriptor_is_explicit_unique_pinned_and_rejects_transition(self):
        name = audit.NAMES["lag"]
        descriptor = dict(schema=1, namespace="lag", plan_sha256="a" * 64,
            receipts=[dict(receipt_relative_path=name + "/costs/one.json", receipt_sha256="b" * 64,
                           binding_scope="postT_namespace")])
        self.assertEqual(len(audit.validate_descriptor(descriptor, "lag", "a" * 64)), 1)
        variants = [copy.deepcopy(descriptor) for _ in range(4)]
        variants[0]["receipts"] *= 2
        variants[1]["receipts"][0]["receipt_relative_path"] = "../one.json"
        variants[2]["receipts"][0]["kind"] = "k_transition"
        variants[3]["plan_sha256"] = "c" * 64
        for changed in variants:
            with self.assertRaises(ValueError):
                audit.validate_descriptor(changed, "lag", "a" * 64)

    def test_actual_original_binding_api_accepts_nested_execution_family_and_rejects_bad_scaler_or_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public, private = root / "public", root / "private"
            public.mkdir(); private.mkdir()
            run_id = "postT_cpu_beijing_E4_NESTED_PCA_k3_s92601_v1"
            source39 = {name: hashlib.sha256(name.encode()).hexdigest() for name in audit.ORDERED_SOURCES}
            foundation, scaler, data_sha = "c" * 64, "d" * 64, "e" * 64
            task = dict(columns=[str(i) for i in range(11)], L=512, H=48, data_sha256=data_sha,
                refit_scaler_sha256=scaler, primary_split_ranges={"A_stop": [600, 900], "V_select": [1000, 3104]})
            condition = dict(run_id=run_id, kind="nested", latent_k=3, variant="nested_pca", parent_run_id="")
            plan = dict(source39=source39, foundation_tensor_sha256=foundation, task_contract=task, outputs=[condition])
            artifact_relative = "ordered/primary_beijing/nested_pca.npz"
            artifact = private / artifact_relative
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"opaque synthetic artifact, never an array")
            artifact_sha = audit.sha(artifact)
            contract = dict(artifact_sha256=artifact_sha, data_sha=data_sha, fit_end=900, source_sha=audit.fingerprint(source39))
            contract_path = artifact.with_name("nested_pca_contract.json")
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            shapes = dict(pred_normalized=[43, 48, 11], target_normalized=[43, 48, 11], observed=[43, 48, 11], origins=[43], channel_ids=[11])
            arrays = {key: dict(shape=shape, dtype="synthetic", sha256="f" * 64) for key, shape in shapes.items()}
            identity = dict(run_id=run_id, split="V_select", cohort="beijing:primary:V_select:stride48",
                source_sha=audit.fingerprint(dict(sources=source39, training_source_sha=contract["source_sha"])),
                config_sha="a" * 64, data_sha=data_sha, model_sha="b" * 64, scaler_sha=scaler)
            manifest = dict(identity, arrays=arrays, archive_sha256="f" * 64,
                metadata=dict(family="NESTED_PCA", parent_run_id="nested_pca", parent_artifact_sha256=artifact_sha,
                              inference_latent_k=3, variant="nested_pca", T_final_accessed=False))
            manifest_relative = "ordered_predictions/" + run_id + "/PREDICTION_MANIFEST.json"
            manifest_path = public / manifest_relative
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            reference = dict(identity, arrays={key: value for key, value in arrays.items() if key != "pred_normalized"})
            manifest_path.with_name("REFERENCE_CONTRACT.json").write_text(json.dumps(reference), encoding="utf-8")
            canonical = dict(run_id=run_id, dataset="beijing", family="NESTED_PCA", fold="primary", latent_k=3,
                canonical_fit_end=900, foundation_sha=foundation, model_size="small", artifact_cache_relative_path=artifact_relative,
                artifact_sha256=artifact_sha, ordered_manifest_public_relative_path=manifest_relative,
                ordered_manifest_sha256=audit.sha(manifest_path))
            measured = dict(canonical, accuracy_run_id=run_id, row_microbatch=128)
            artifact_bindings = dict(route=dict(cache_relative_path=artifact_relative, sha256=artifact_sha))
            binding = dict(status="BOUND_EXACT_ROUTE", condition_id=run_id, accuracy_run_id=run_id,
                cost_route_spec=measured, cost_route_spec_sha256=audit.fingerprint(measured),
                accuracy_route_spec=canonical, accuracy_route_spec_sha256=audit.fingerprint(canonical),
                accuracy_manifest_public_relative_path=manifest_relative, accuracy_manifest_sha256=audit.sha(manifest_path),
                accuracy_prediction_model_sha=identity["model_sha"], accuracy_prediction_source_sha=identity["source_sha"],
                data_sha256=data_sha, scaler_sha=scaler, artifact_bindings=artifact_bindings, foundation_state_sha256=foundation,
                foundation_bindings=dict(route=dict(foundation_sha=foundation, model_size="small")))
            receipt = dict(run_id=run_id, dataset="beijing", family="PCA_FIXED_BOLT", batch_size=8, C=11, H=48, L=512,
                data_sha256=data_sha, scaler_sha=scaler, artifact_sha256=artifact_sha, accuracy_bindings={run_id: binding},
                dtype="float32", physical_output_dtype="float64", inverse_transform_dtype="float64",
                origins=list(range(1000, 1000 + 8 * 48, 48)), repeated_histories_for_batch_cost_only=False,
                physical_input_sha256="a" * 64, observation_mask_sha256="b" * 64)
            env = dict(Path=Path, fingerprint=audit.fingerprint, _inside=audit.inside, _load=audit.read_json,
                       data=SimpleNamespace(sha256_file=audit.sha))
            functions = original_functions("report_campaign.py", {"_cost_binding", "_input"}, env)
            catalog = original_functions("route_catalog.py", {"artifact_bindings", "expected_F_streams"},
                dict(DIRECT={"DIRECT_G", "DIRECT_B2", "DIRECT_B4", "CURRENT", "SEASONAL", "RAW_RIDGE"}))
            api = SimpleNamespace(cost_binding=functions._cost_binding, artifact_bindings=catalog.artifact_bindings,
                                  expected_F_streams=catalog.expected_F_streams)
            state = dict(plan=plan, post=public, private=private, registry={run_id: {}}, original_registry={})
            item = dict(binding_scope="postT_namespace")
            result = audit.validate_binding(receipt, item, state, "multik", public, private, audit.InputPins(), api)
            self.assertEqual(result["canonical_family"], "NESTED_PCA")
            self.assertEqual(result["cost_execution_family"], "PCA_FIXED_BOLT")
            for changed in (dict(receipt, family="NESTED_PCA"), dict(receipt, scaler_sha="0" * 64)):
                with self.assertRaises(ValueError):
                    audit.validate_binding(changed, item, state, "multik", public, private, audit.InputPins(), api)
            artifact.write_bytes(b"tampered opaque artifact")
            with self.assertRaises(ValueError):
                audit.validate_binding(receipt, item, state, "multik", public, private, audit.InputPins(), api)

    def test_mixed_original_and_namespace_roots_use_actual_original_binding_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_public, old_cache = root / "old_public", root / "old_cache"
            post_public, post_cache = root / "new_public", root / "new_cache"
            for folder in (old_public, old_cache, post_public, post_cache):
                folder.mkdir()
            old_id = "primary_jena_E1_DIRECT_G_k0_s92601__cpu_fresh_e3"
            new_id = "postT_cpu_jena_E3_B0_k6_s92601_v1"
            state = dict(post=post_public, private=post_cache, original_registry={old_id: {}}, registry={new_id: {}})
            env = dict(Path=Path, fingerprint=audit.fingerprint, _inside=audit.inside, _load=audit.read_json,
                       data=SimpleNamespace(sha256_file=audit.sha))
            functions = original_functions("report_campaign.py", {"_cost_binding", "_input"}, env)
            prepared = []
            for run_id, public, private, label in ((old_id, old_public, old_cache, b"old G bytes"),
                                                  (new_id, post_public, post_cache, b"new B0 bytes")):
                relative = "same_relative/selected.bin"
                artifact = private / relative
                artifact.parent.mkdir()
                artifact.write_bytes(label)
                manifest_relative = "same_relative/PREDICTION_MANIFEST.json"
                manifest_path = public / manifest_relative
                manifest_path.parent.mkdir()
                manifest = dict(run_id=run_id, split="V_select", data_sha="a" * 64, scaler_sha="b" * 64,
                                model_sha="c" * 64, source_sha="d" * 64)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                canonical = dict(run_id=run_id, artifact_cache_relative_path=relative, artifact_sha256=audit.sha(artifact))
                measured = dict(canonical, accuracy_run_id=run_id, row_microbatch=128)
                binding = dict(status="BOUND_EXACT_ROUTE", condition_id=run_id, accuracy_run_id=run_id,
                    accuracy_route_spec=canonical, accuracy_route_spec_sha256=audit.fingerprint(canonical),
                    cost_route_spec=measured, cost_route_spec_sha256=audit.fingerprint(measured),
                    accuracy_manifest_public_relative_path=manifest_relative, accuracy_manifest_sha256=audit.sha(manifest_path),
                    data_sha256=manifest["data_sha"], scaler_sha=manifest["scaler_sha"],
                    accuracy_prediction_model_sha=manifest["model_sha"], accuracy_prediction_source_sha=manifest["source_sha"],
                    artifact_bindings=dict(route=dict(cache_relative_path=relative, sha256=audit.sha(artifact))))
                prepared.append(dict(accuracy_bindings={run_id: binding}))
            for run_id, scope, receipt, expected_public, expected_cache in (
                    (old_id, "original", prepared[0], old_public, old_cache),
                    (new_id, "postT_namespace", prepared[1], post_public, post_cache)):
                context = audit.binding_context(scope, run_id, state, "lag", old_public, old_cache)
                self.assertEqual((context["public"], context["cache"]), (expected_public, expected_cache))
                inputs = {}
                result = functions._cost_binding(receipt, run_id, context["public"], context["cache"], context["registry"], inputs)
                self.assertEqual(result, (run_id, "BOUND_EXACT_ROUTE"))
                wrong_public, wrong_cache = (post_public, post_cache) if scope == "original" else (old_public, old_cache)
                with self.assertRaises(ValueError):
                    functions._cost_binding(receipt, run_id, wrong_public, wrong_cache, context["registry"], {})
                with self.assertRaises(ValueError):
                    functions._cost_binding(receipt, run_id, context["public"], wrong_cache, context["registry"], {})
            with self.assertRaises(ValueError):
                audit.binding_context("original", new_id, state, "lag", old_public, old_cache)
            with self.assertRaises(ValueError):
                audit.binding_context("postT_namespace", old_id, state, "lag", old_public, old_cache)

    def test_pin_change_and_output_overwrite_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = root / "selected.bin"
            value.write_bytes(b"trusted synthetic bytes")
            pins = audit.InputPins()
            pins.add(root, "selected.bin", "private")
            self.assertTrue(pins.unchanged())
            value.write_bytes(b"changed")
            self.assertFalse(pins.unchanged())
            target = root / "audit"
            audit.write_result(target, dict(status="SYNTHETIC"))
            original = (target / "AUDIT_RESULT.json").read_bytes()
            with self.assertRaises(FileExistsError):
                audit.write_result(target, dict(status="OVERWRITE"))
            self.assertEqual((target / "AUDIT_RESULT.json").read_bytes(), original)

    def test_default_metadata_does_not_import_science_or_write(self):
        before = {name for name in sys.modules if name.split(".")[0] in {"numpy", "torch", "psutil"}}
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = audit.main([])
        output = json.loads(stream.getvalue())
        after = {name for name in sys.modules if name.split(".")[0] in {"numpy", "torch", "psutil"}}
        self.assertEqual(code, 0)
        self.assertEqual(before, after)
        self.assertEqual(output["status"], "METADATA_ONLY_NO_RECEIPTS_READ")
        self.assertFalse(output["science_api_imported"])
        self.assertFalse(output["files_written"])


if __name__ == "__main__":
    unittest.main()
