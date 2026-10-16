"""Only tiny fake APIs and metadata; no scientific module imports or jobs."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


def load():
    spec = importlib.util.spec_from_file_location("multik_wrapper_tested", Path(__file__).with_name("multik_runner.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Foundation:
    def __init__(self):
        self.streams = self.row_calls = self.physical_streams = self.physical_calls = 0
        self.assertions = 0
    def assert_frozen_unchanged(self):
        self.assertions += 1


class MultiKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = load()

    def lock(self):
        return dict(locked=True, chosen_before_E4_training=True, chosen_family="B0", dataset="beijing", fold="primary",
                    seed=92601, k_grid=[3, 6, 11], data_sha="data", scaler_sha="scaler", source_files={})

    def test_metadata_default_and_twelve_unique_outputs(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(self.r.main([]), 0)
        value = json.loads(stream.getvalue())
        self.assertEqual((value["fit_conditions"], value["V_outputs"]), (5, 12))
        self.assertFalse(value["numpy_imported"] or value["torch_imported"])
        fits, outputs = self.r.recipe_manifest(self.lock())
        self.assertEqual(len({row["run_id"] for row in outputs}), 12)
        self.assertEqual(sum(row["variant"] == "posthoc" for row in outputs), 2)
        self.assertEqual(sum(row["variant"] == "max_only" for row in outputs), 1)
        self.assertEqual({row["latent_k"] for row in outputs if row["kind"] == "independent"}, {3, 6, 11})

    def test_sealed_metadata_guards_original_lock_sources_scalers_and_twelve_rows(self):
        r = self.r
        public, cache, asset = ((Path(tempfile.gettempdir())/"synthetic_multik")/part for part in ("public","cache","asset"))
        names = ["ordered_pilot.py","prediction_cache.py","fit_base.py","training.py","networks.py","data_contract.py","evaluation.py","gpu_runtime.py","campaign.py"]+[f"tiny{i}.py" for i in range(30)]
        source39 = {name:"pin" for name in names}
        contract = dict(shape=[21038,11], L=512, H=48, columns=[f"c{i}" for i in range(11)],
            legal_origin_counts_primary=dict(V_select=43), data_sha256="data", search_scaler_sha256="search", refit_scaler_sha256="scaler")
        original_contract = {k:v for k,v in contract.items() if k not in {"search_scaler_sha256","refit_scaler_sha256"}}
        lock = dict(self.lock(), source_files={name:"pin" for name in names[:8]})
        fits, outputs = r.recipe_manifest(lock,source39)
        immutable = {name:"pin" for name in ("EXPERIMENT_REGISTRY.csv","SELECTION_LOCK.json","DATA_CONTRACT.json","SCORES.csv","COSTS.csv","T_FIXED_ROLE_SCORES.csv")}
        reference_path = "postT_V_only_cpu_fixed_pca_20261016_v1/PLAN.json"
        plan = dict(stage="POST_T_V_ONLY_DEVELOPMENT", output_public_relative_root=r.NAME, output_cache_relative_root=r.NAME,
            CPU_threads=2,max_RSS_bytes=3221225472,maximum_concurrent_workers=2,total_owned_RSS_limit_bytes=6442450944,
            storage_limit_bytes=32212254720,deadline_unix=1792222550,runner_sha256="pin",source39=source39,immutable_public=immutable,
            shared_utility_relative_path=r.UTILITY_PATH,shared_utility_sha256=r.UTILITY_SHA,
            pca_runner_relative_path=r.PCA_RUNNER_PATH,pca_runner_sha256=r.PCA_RUNNER_SHA,
            family_lock=dict(relative_path=r.LOCK_PATH,sha256=r.LOCK_SHA), task_contract=contract,
            task_contract_reference=dict(relative_path=reference_path,sha256="pin"), model_asset_relative_path="models/rev",model_revision="rev",
            instance_norm_source_relative_path="norm.py",instance_norm_source_sha256="pin",fits=fits,outputs=outputs,
            postT_registry_relative_path="REGISTRY.csv",postT_registry_sha256="pin")
        old_rows = [dict(experiment_id=row["upstream_registered_id"],dataset="beijing",family="B0" if row["role"]=="INDEPENDENT_B0" else row["role"],
            latent_k=str(row["spec"]["latent_k"]),seed="92601",fold="primary",length="512",horizon="48") for row in fits]
        registered = [dict(experiment_id=row["run_id"],dataset="beijing",family=row["role"],latent_k=str(row["latent_k"]),seed="92601",fold="primary",
            length="512",horizon="48",device="cpu",stage="POST_T_V_ONLY_DEVELOPMENT",upstream_registered_id=row["upstream_registered_id"]) for row in outputs]
        pca_plan = dict(source39=copy.deepcopy(source39),immutable_public=copy.deepcopy(immutable),task_contracts=dict(beijing=copy.deepcopy(contract)))
        mapping = {str(public/r.NAME/"PLAN.json"):plan,str(public/r.LOCK_PATH):lock,str(public/reference_path):pca_plan,
                   str(public/"SELECTION_LOCK.json"):dict(locked=True),str(public/"DATA_CONTRACT.json"):dict(assets=dict(beijing=original_contract))}
        def fake_sha(path):
            text = Path(path).as_posix()
            for suffix,digest in ((r.LOCK_PATH,r.LOCK_SHA),(r.UTILITY_PATH,r.UTILITY_SHA),(r.PCA_RUNNER_PATH,r.PCA_RUNNER_SHA)):
                if text.endswith(suffix): return digest
            return "pin"
        with mock.patch.object(r,"sha",side_effect=fake_sha),mock.patch.object(r,"read",side_effect=lambda p:mapping[str(p)]), \
             mock.patch.object(r,"rows",side_effect=lambda p:registered if Path(p).name=="REGISTRY.csv" else old_rows), \
             mock.patch.object(Path,"is_dir",return_value=True):
            self.assertIs(r.metadata(public,cache,asset,"pin"),plan)
            for mutation in ("cost_pin","max_grid","scaler"):
                original_plan = copy.deepcopy(plan)
                if mutation == "cost_pin": plan["immutable_public"].pop("COSTS.csv")
                if mutation == "max_grid": plan["fits"][1]["spec"].pop("k_grid")
                if mutation == "scaler": plan["task_contract"]["refit_scaler_sha256"]="changed"
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    r.metadata(public,cache,asset,"pin")
                plan.clear();plan.update(original_plan)

    def test_original_make_specs_keeps_max_grid_and_independent_identity_case(self):
        lock = self.lock()
        fits, _ = self.r.recipe_manifest(lock)
        calls = []
        def original(task, received_lock, **kwargs):
            calls.append((task, received_lock, kwargs))
            return self.r.original_ordered_specs(lock)
        actual = self.r.build_specs(original, {"synthetic": True}, lock)
        self.assertEqual(actual, {row["run_id"]: row["spec"] for row in fits})
        self.assertEqual(calls[0][2], dict(epochs=[.25, 1., 3.], device="cpu"))
        shared = next(row for row in fits if row["role"] == "ORDERED_MULTI_K")
        maximum = next(row for row in fits if row["role"] == "MAXK_PCA_TRAINED")
        self.assertEqual(actual[shared["run_id"]]["k_grid"], [3, 6, 11])
        self.assertEqual(actual[maximum["run_id"]]["k_grid"], [11])
        for row in fits:
            self.assertEqual(row["spec"]["family"], "B0")
            if row["role"] == "INDEPENDENT_B0":
                self.assertNotIn("k_grid", row["spec"])
        def wrong(*a, **kw):
            result = self.r.original_ordered_specs(lock)
            result["max_only"].pop("k_grid")
            return result
        with self.assertRaises(ValueError):
            self.r.build_specs(wrong, {}, lock)

    def test_original_fit_boundary_zero_selected_refit_and_strict_search_depth(self):
        fits, _ = self.r.recipe_manifest(self.lock())
        row = fits[0]
        calls = []
        receipt = dict(status="COMPLETE_VALID", run_id=row["run_id"], family="B0", reused_complete_artifact=False,
                       selected_epoch=0., search_exposures=300, refit_exposures=0, refit_updates=0)
        search = dict(status="SEGMENT_COMPLETE", completed_epochs=[0., .25, 1., 3.], exposures=300, train_origins=100, actual_epochs=3.)
        refit = dict(status="SEGMENT_COMPLETE", selected_epoch=0., exposures=0, updates=0, actual_epochs=0.)
        def api(*args):
            calls.append(args)
            return receipt
        self.assertIs(self.r.fit_call(api, row["spec"], "task", "newcache", "newpublic", "F", 99), receipt)
        self.assertEqual(calls[0], (row["spec"], "task", "newcache", "newpublic", "F", 99))
        self.r.validate_complete_fit(receipt, search, refit, row)
        with self.assertRaises(ValueError):
            self.r.validate_complete_fit(receipt, dict(search, actual_epochs=.25), refit, row)
        with self.assertRaises(ValueError):
            self.r.fit_call(lambda *a: dict(receipt, reused_complete_artifact=True), row["spec"], {}, "c", "p", "F", 99)

    def test_independent_and_ordered_V_use_original_distinct_APIs_and_actual_streams(self):
        fits, outputs = self.r.recipe_manifest(self.lock())
        receipts = {row["run_id"]: dict(status="COMPLETE_VALID", run_id=row["run_id"]) for row in fits}
        for output in outputs:
            foundation, calls = Foundation(), []
            def api(*args, **kwargs):
                calls.append((args, kwargs))
                foundation.streams += 43 * output["latent_k"]
                return dict(score=dict(n_origins=43), metadata=dict(), cost_record=dict(actual_F_streams=43 * output["latent_k"], expected_F_streams=43 * output["latent_k"]))
            self.r.predict_call(api, api, output, {}, "newcache", "newpublic", foundation, receipts.get(output["parent_run_id"]), 99)
            if output["kind"] == "independent":
                self.assertEqual(calls[0][0][2], "V_select")
                self.assertEqual(calls[0][1]["fold"], "primary")
            else:
                self.assertEqual(calls[0][0][2], output["latent_k"])
                self.assertEqual(calls[0][1]["variant"], output["variant"])
            self.assertEqual(foundation.assertions, 2)
        with self.assertRaises(ValueError):
            self.r.predict_call(lambda *a, **kw: {}, lambda *a, **kw: {}, dict(outputs[0], split="T_final"), {}, "c", "p", Foundation(), None, 99)

    def test_three_update_probe_preserves_full_legal_normalization_K_cycle_and_frozen_F(self):
        fits, _ = self.r.recipe_manifest(self.lock())
        for role, streams, exposure in (("ORDERED_MULTI_K", 320, {"3":16,"6":16,"11":16}), ("MAXK_PCA_TRAINED", 528, {"11":48})):
            row = next(row for row in fits if row["role"] == role)
            foundation, calls = Foundation(), []
            def predict(x, k):
                foundation.streams += 4*k
                foundation.physical_streams += 4*k
                foundation.row_calls += 1
                foundation.physical_calls += 1
            def segment(*args, **kwargs):
                calls.append((args, kwargs))
                for k in ([3, 6, 11] if role == "ORDERED_MULTI_K" else [11, 11, 11]):
                    for _ in range(4):
                        args[6]("tiny4", k)
                return dict(status="SEGMENT_COMPLETE", updates=3, exposures=48, optimizer_updates_this_segment=3,
                    k_exposures=exposure, gradient_norm_min=0., gradient_norm_max=1.)
            result = self.r.probe_call(segment, "model", "fullA50windows", range(100), "sampler", "optimizer", predict, foundation, row, 99)
            self.assertEqual(result["F_counter_delta"]["streams"], streams)
            self.assertEqual(calls[0][0][2], range(100))
            self.assertEqual(calls[0][0][5], 48)
            self.assertEqual(calls[0][1]["effective_batch"], 16)
            self.assertEqual(calls[0][1]["microbatch"], 4)
            self.assertTrue(result["probe_state_discarded"])
            self.assertEqual(foundation.assertions, 2)
        with self.assertRaises(ValueError):
            self.r.probe_call(lambda *a, **kw: {}, None, None, range(100), None, None, predict, Foundation(), fits[2], 99)

    def test_explicit_mode_ID_and_no_overwrite(self):
        fits, outputs = self.r.recipe_manifest(self.lock())
        plan = dict(fits=fits, outputs=outputs)
        with self.assertRaises(ValueError):
            self.r.choose(plan, "T_final", outputs[0]["run_id"])
        with self.assertRaises(ValueError):
            self.r.choose(plan, "fit", "unknown")
        with tempfile.TemporaryDirectory() as folder:
            public, cache = Path(folder)/"p", Path(folder)/"c"
            public.mkdir(); cache.mkdir()
            row = fits[0]
            self.r.require_fresh(public, cache, "fit", row["run_id"])
            old = cache / self.r.NAME / "runs" / row["run_id"]
            old.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                self.r.require_fresh(public, cache, "fit", row["run_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
