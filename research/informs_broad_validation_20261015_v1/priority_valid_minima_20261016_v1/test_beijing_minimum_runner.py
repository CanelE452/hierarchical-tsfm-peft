"""Only synthetic arrays and metadata; no real source values/model/worker."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

SOURCE = Path(__file__).with_name("beijing_minimum_runner.py")


def load():
    spec = importlib.util.spec_from_file_location("tested_beijing_minimum_runner", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BeijingMinimumTests(unittest.TestCase):
    def test_default_is_successful_metadata_only_without_science_import(self):
        api = load()
        before = {name: name in sys.modules for name in ("numpy", "torch", "psutil", "data_contract")}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(api.main([]), 0)
        result = json.loads(out.getvalue())
        self.assertEqual((result["input_reads"], result["writes"], result["scientific_or_child_operations"]), (0, 0, 0))
        self.assertEqual(before, {name: name in sys.modules for name in before})

    def test_only_original_three_specs_and_sealed_plan_are_accepted(self):
        api = load()
        plan = api.draft_plan()
        with self.assertRaises(ValueError): api.validate_plan(plan)
        plan.update(status="SEALED_BY_ROOT", runner_sha256="a" * 64)
        api.validate_plan(plan)
        for field, value in (("materialized_prefix_rows", 21038), ("new_T_target_operations_allowed", True),
                             ("maximum_new_training_conditions", 13), ("original_required_plan_sha256", "foreign")):
            changed = copy.deepcopy(plan); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): api.validate_plan(changed)
        changed = copy.deepcopy(plan); changed["conditions"][1]["spec"]["family"] = "B0"
        with self.assertRaises(ValueError): api.validate_plan(changed)

    def test_fixed_original_bounds_and_selected_zero_fresh_refit(self):
        api = load(); calls = []
        def stage(spec, task, fit_end, stop, epochs, private, public, foundation, deadline, contract):
            calls.append((fit_end, stop, epochs, private))
            return dict(summary=dict(status="SEGMENT_COMPLETE", selected_epoch=0.))
        fit = types.SimpleNamespace(_stage=stage)
        result = api.fit_stages(fit, api.conditions()[0]["spec"], {}, None, {}, Path("public"), Path("private"), 900.)
        self.assertEqual(result["status"], "COMPLETE_STAGES")
        self.assertEqual(calls[0][:3], (10519, [10519, 12622], [.25, 1., 3.]))
        self.assertEqual(calls[1][:3], (12622, None, []))
        self.assertNotEqual(calls[0][3], calls[1][3])

    def test_partial_stage_does_not_start_refit_or_promote_completion(self):
        api = load(); calls = []
        def stage(*args):
            calls.append(args)
            return dict(summary=dict(status="BUDGET_LIMIT", selected_epoch=.25))
        result = api.fit_stages(types.SimpleNamespace(_stage=stage), api.conditions()[0]["spec"], {}, None, {}, Path("public"), Path("private"), 900.)
        self.assertEqual((result["status"], result["phase"], len(calls)), ("BUDGET_LIMIT_PRESERVED", "search", 1))

    def test_existing_completed_partial_and_claim_are_preserved_per_condition(self):
        api = load()
        with tempfile.TemporaryDirectory() as directory:
            public, cache = Path(directory) / "public", Path(directory) / "cache"
            case = api.conditions()[0]; identifier = case["run_id"]
            self.assertEqual(api.existing_state(public, cache, identifier)["status"], "UNSTARTED")
            claim = public / api.OLD / "claims" / ("probe_" + identifier + ".json")
            claim.parent.mkdir(parents=True); claim.write_text("{}", encoding="utf-8")
            self.assertEqual(api.existing_state(public, cache, identifier)["status"], "UNSTARTED")
            path = public / api.OLD / "runs" / identifier / "BASE_FIT_RECEIPT.json"
            path.parent.mkdir(parents=True); path.write_text(json.dumps(dict(status="COMPLETE_VALID", run_id=identifier)), encoding="utf-8")
            original = path.read_bytes()
            self.assertEqual(api.existing_state(public, cache, identifier)["status"], "COMPLETE_REUSE_REQUIRED")
            self.assertEqual(api.existing_state(public, cache, api.conditions()[1]["run_id"])["status"], "UNSTARTED")
            self.assertEqual(path.read_bytes(), original)

    def test_stale_future_and_nonquiet_cost_proofs_fail(self):
        api = load(); proof = dict(status="QUIET_CPU_COST_WINDOW", active_model_workers=[], active_cost_workers=[], captured_unix=100.)
        api.require_quiet(proof, 220.)
        for changed, now in ((proof, 220.01), (proof, 99.99), (dict(proof, active_model_workers=[1]), 101.),
                             (dict(proof, captured_unix=True), 101.)):
            with self.assertRaises(ValueError): api.require_quiet(changed, now)

    def test_public_score_binding_rejects_foreign_run_scaler_and_artifact(self):
        api = load(); condition = api.conditions()[0]
        receipt = dict(run_id=condition["run_id"], family="B2", seed=92601, artifact_sha256="artifact", model_state_sha="model", refit_scaler_sha="scaler")
        context = dict(new_sources={"source": "pin"}, task_data_sha="data")
        manifest = dict(run_id=condition["run_id"], family="B2", seed=92601, split="V_select", data_sha="data", scaler_sha="scaler",
            artifact_sha256="artifact", source_sha=api.fingerprint(context["new_sources"]), config_sha=api.fingerprint(condition["spec"]),
            model_sha=api.fingerprint(dict(artifact_sha="artifact", model_state_sha="model")))
        api.validate_saved(manifest, receipt, condition, context)
        for key, value in (("run_id", api.conditions()[1]["run_id"]), ("scaler_sha", "foreign"), ("artifact_sha256", "foreign"), ("split", "T_final")):
            with self.subTest(key=key), self.assertRaises(ValueError): api.validate_saved(dict(manifest, **{key: value}), receipt, condition, context)

    def test_actual_exposure_contract_rejects_short_search_and_wrong_refit(self):
        api = load()
        search = dict(status="SEGMENT_COMPLETE", completed_epochs=[0., .25, 1., 3.], selected_epoch=1., train_origins=9960, exposures=29880, updates=1868)
        refit = dict(status="SEGMENT_COMPLETE", completed_epochs=[0., 1.], train_origins=12063, exposures=12063, updates=754)
        api.validate_stages(search, refit)
        with self.assertRaises(ValueError): api.validate_stages(dict(search, exposures=9960), refit)
        with self.assertRaises(ValueError): api.validate_stages(search, dict(refit, exposures=12064))

    def test_c_order_prefix_matches_columns_and_never_decodes_tail(self):
        api = load()
        import numpy as np
        source = SOURCE.parents[1] / "priority_b2_validation_20261016_v1" / "safe_jena_prefix.py"
        spec = importlib.util.spec_from_file_location("tiny_primitive", source)
        primitive = importlib.util.module_from_spec(spec); spec.loader.exec_module(primitive)
        source = SOURCE.parents[1] / "priority_nested_cost_20261016_v1" / "prefix_nested_cost_runner.py"
        spec = importlib.util.spec_from_file_location("tiny_nested_source", source)
        nested = importlib.util.module_from_spec(spec); spec.loader.exec_module(nested)
        values = dict(physical=np.asarray([[1,2],[3,4],[900,901],[902,903]],np.float32), mask=np.ones((4,2),bool),
                      times=np.asarray(["2000-01-01T00:00:00","2000-01-01T01:00:00","2000-01-01T02:00:00","2000-01-01T03:00:00"],dtype="<U19"),
                      columns=np.asarray(["a","b"],dtype="<U5"))
        headers = {name:dict(descr=value.dtype.str, shape=list(value.shape), fortran_order=False) for name,value in values.items()}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream,"w") as archive:
            for name,value in values.items():
                payload = io.BytesIO(); np.save(payload,value,allow_pickle=False); archive.writestr(name+".npy",payload.getvalue())
        stream.seek(0); calls = []; original = primitive.decode_numeric
        def decode(blob,descr): calls.append((blob,descr)); return original(blob,descr)
        with zipfile.ZipFile(stream) as archive, mock.patch.object(primitive,"decode_numeric",side_effect=decode):
            prefix, receipts = api.read_prefix_members(archive,primitive,nested,2,headers)
        self.assertTrue(np.array_equal(prefix["physical"],np.asarray([[1,2],[3,4]],np.float32)))
        self.assertEqual(calls[0][0], values["physical"][:2].tobytes())
        self.assertTrue(all(row["tail_numeric_values_parsed"] == 0 for row in receipts))
        bad = copy.deepcopy(headers); bad["physical"]["fortran_order"] = True
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive, self.assertRaises(ValueError): api.read_prefix_members(archive,primitive,nested,2,bad)

    def test_draft_cannot_spawn_or_touch_sources_via_execute_cli(self):
        api = load()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): api.main(["--execute"])

    def test_sealed_plan_without_exact_two_snapshot_bindings_is_rejected(self):
        api = load(); plan = api.draft_plan()
        plan.update(status="SEALED_BY_ROOT",runner_sha256="a"*64,model_snapshot_bindings=[])
        with self.assertRaises(ValueError): api.validate_plan(plan)
        plan["model_snapshot_bindings"] = copy.deepcopy(api.MODEL_SNAPSHOT_BINDINGS)
        api.validate_plan(plan)
        for changed in (plan["model_snapshot_bindings"][:1],plan["model_snapshot_bindings"]+[plan["model_snapshot_bindings"][0]],
                        [dict(plan["model_snapshot_bindings"][0],sha256="0"*64),plan["model_snapshot_bindings"][1]]):
            with self.assertRaises(ValueError): api.validate_plan(dict(plan,model_snapshot_bindings=changed))

    def test_same_weights_config_target_size_hash_and_asset_escape_are_rejected(self):
        api = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); asset = root/"asset"; snapshot = asset/"snapshot"; blobs = asset/"blobs"
            snapshot.mkdir(parents=True); blobs.mkdir()
            original_config = b'{"quantiles":[0.1,0.5,0.9]}'
            original_weights = b"same_tiny_fake_model_weights"
            contents = {"config.json":original_config,"model.safetensors":original_weights}
            targets,bindings = {},[]
            for name,payload in contents.items():
                path = blobs/name; path.write_bytes(payload); (snapshot/name).write_bytes(b"opaque_snapshot_link_fixture")
                targets[snapshot/name] = path
                bindings.append(dict(snapshot_relative_path="snapshot/"+name,resolved_blob_relative_path="blobs/"+name,
                    bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
            resolve = Path.resolve
            # Model only the OS link resolution; size and SHA checks use real tiny files.
            def resolved(path,*args,**kwargs):
                return targets[path] if path in targets else resolve(path,*args,**kwargs)
            with mock.patch.object(Path,"resolve",new=resolved):
                api.check_model_snapshot(asset,"snapshot",bindings)
                for payload in (b'{"quantiles":[0.9,0.5,0.1]}',original_config+b" "):
                    (blobs/"config.json").write_bytes(payload)
                    with self.assertRaises(ValueError): api.check_model_snapshot(asset,"snapshot",bindings)
                (blobs/"config.json").write_bytes(original_config)
                (blobs/"config_copy.json").write_bytes(original_config)
                outside = root/"outside.json"; outside.write_bytes(original_config)
                for target in (blobs/"config_copy.json",outside):
                    targets[snapshot/"config.json"] = target
                    with self.assertRaises(ValueError): api.check_model_snapshot(asset,"snapshot",bindings)
                targets[snapshot/"config.json"] = blobs/"config.json"
                for field,value in (("sha256","0"*64),("bytes",len(original_weights)+1),("resolved_blob_relative_path","../outside.json")):
                    changed = copy.deepcopy(bindings); changed[1][field] = value
                    with self.subTest(field=field),self.assertRaises(ValueError): api.check_model_snapshot(asset,"snapshot",changed)
                api.check_model_snapshot(asset,"snapshot",bindings)
            self.assertEqual((blobs/"model.safetensors").read_bytes(),original_weights)


if __name__ == "__main__":
    unittest.main(verbosity=2)
