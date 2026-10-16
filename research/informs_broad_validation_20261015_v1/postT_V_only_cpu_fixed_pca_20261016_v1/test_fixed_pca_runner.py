import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


def load_runner():
    path = Path(__file__).with_name("fixed_pca_runner.py")
    spec = importlib.util.spec_from_file_location("fixed_pca_runner_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FixedPCATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = load_runner()

    def fixture(self, root):
        public, cache, asset = root / "research" / "campaign", root / "cache", root / "asset"
        for path in (public, cache, asset):
            path.mkdir(parents=True)
        (public / self.r.NAME).mkdir()
        names = ["campaign.py", "fit_base.py", "training.py", "networks.py", "gpu_runtime.py", "data_contract.py", "prediction_cache.py"] + [f"tiny{i}.py" for i in range(32)]
        source39 = {}
        for name in names:
            (public / name).write_text("# synthetic source identity\n")
            source39[name] = digest(public / name)
        norm = "norm/runtime.py"
        (asset / "norm").mkdir()
        (asset / norm).write_text("# synthetic norm\n")
        actual_norm = public.parent / "temporal_frontend_campaign_20261015_v1/runtime.py"
        actual_norm.parent.mkdir()
        actual_norm.write_bytes((asset / norm).read_bytes())
        (asset / "models" / "tiny_revision").mkdir(parents=True)
        assets = {}
        for ds, (rows, channels, horizon) in self.r.AXES.items():
            assets[ds] = dict(status="READY_PINNED", shape=[rows, channels], L=512, H=horizon,
                columns=[f"c{i}" for i in range(channels)], array_sha256={"physical": "1" * 64}, data_sha256="2" * 64,
                primary_split_ranges={"A_fit": [0, 10500], "A_stop": [10500, 12600], "M_cal": [12600, 14700], "V_select": [14700, 16800], "T_final": [16800, 21000]},
                secondary_split_ranges={}, A_fit_statistics_fingerprints={"mean": "mean", "std": "std", "count": "count"},
                legal_origin_counts_primary={"V_select": 43 if horizon == 48 else 175}, metadata={"synthetic": True})
        write_json(public / "DATA_CONTRACT.json", {"assets": assets})
        write_json(public / "SELECTION_LOCK.json", {"locked": True})
        old, registered, conditions = [], [], []
        for run_id, (ds, k) in self.r.CONDITIONS.items():
            upstream = self.r.upstream_id(ds, k)
            old.append(dict(experiment_id=upstream, dataset=ds, family="PCA_FIXED_BOLT", fold="primary", latent_k=str(k), seed="92601", length="512", horizon=str(self.r.AXES[ds][2]), execution_status="PENDING_QUEUE"))
            registered.append(dict(experiment_id=run_id, upstream_registered_id=upstream, dataset=ds, family="PCA_FIXED_BOLT", fold="primary", latent_k=str(k), seed="92601", length="512", horizon=str(self.r.AXES[ds][2]), device="cpu", epochs="0.25|1|3", stage="POST_T_V_ONLY_DEVELOPMENT"))
            spec = self.r.expected_spec(run_id, ds, k)
            conditions.append(dict(run_id=run_id, family="PCA_FIXED_BOLT", upstream_registered_id=upstream, spec=spec,
                                   recipe_reference=dict(relative_path="campaign.py", sha256=source39["campaign.py"])))
        write_csv(public / "EXPERIMENT_REGISTRY.csv", old)
        write_csv(public / self.r.NAME / "REGISTRY.csv", registered)
        utility = public / self.r.UTILITY_RELATIVE_PATH
        utility.parent.mkdir()
        utility.write_text("# utility is never imported by these metadata fixtures\n")
        tasks = {ds: dict(value, search_scaler_sha256="3" * 64, refit_scaler_sha256="4" * 64) for ds, value in assets.items()}
        plan = dict(stage="POST_T_V_ONLY_DEVELOPMENT", runner_sha256=digest(self.r.__file__), source39=source39,
            immutable_public={name: digest(public / name) for name in ("EXPERIMENT_REGISTRY.csv", "SELECTION_LOCK.json", "DATA_CONTRACT.json")},
            output_public_relative_root=self.r.NAME, output_cache_relative_root=self.r.NAME,
            postT_registry_relative_path="REGISTRY.csv", postT_registry_sha256=digest(public / self.r.NAME / "REGISTRY.csv"),
            shared_utility_relative_path=self.r.UTILITY_RELATIVE_PATH, shared_utility_sha256=digest(utility),
            task_contracts=tasks, conditions=conditions, CPU_threads=2, max_RSS_bytes=3221225472,
            maximum_concurrent_workers=1, storage_limit_bytes=32212254720, deadline_unix=1792222550,
            instance_norm_source_relative_path=norm, instance_norm_source_sha256=digest(asset / norm),
            model_asset_relative_path="models/tiny_revision", model_revision="tiny_revision", foundation_tensor_sha256="5" * 64)
        write_json(public / self.r.NAME / "PLAN.json", plan)
        return public, cache, asset, plan

    def test_default_metadata_does_not_import_science_or_load_inputs(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(self.r.main([]), 0)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["registered_conditions"], 6)
        self.assertEqual(result["actual_model_array_or_fit_access"], 0)
        self.assertFalse(result["numpy_imported"])
        self.assertFalse(result["torch_imported"])

    def test_six_condition_metadata_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            public, cache, asset, plan = self.fixture(Path(td))
            self.assertEqual(self.r.metadata(public, cache, asset)["conditions"], plan["conditions"])
            bad = copy.deepcopy(plan)
            bad["conditions"][0]["spec"]["latent_k"] += 1
            write_json(public / self.r.NAME / "PLAN.json", bad)
            with self.assertRaises(ValueError):
                self.r.metadata(public, cache, asset)
            write_json(public / self.r.NAME / "PLAN.json", plan)
            (public / "campaign.py").write_text("changed")
            with self.assertRaises(ValueError):
                self.r.metadata(public, cache, asset)

    def test_task_contract_and_pems_horizon_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as td:
            public, cache, asset, plan = self.fixture(Path(td))
            for change in ("horizon", "data", "utility"):
                bad = copy.deepcopy(plan)
                if change == "horizon":
                    bad["task_contracts"]["pems08"]["H"] = 48
                elif change == "data":
                    bad["task_contracts"]["jena"]["data_sha256"] = "9" * 64
                else:
                    bad["shared_utility_sha256"] = "0" * 64
                write_json(public / self.r.NAME / "PLAN.json", bad)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.r.metadata(public, cache, asset)

    def outcome(self, spec, count=43):
        return dict(fit_receipt=dict(status="COMPLETE_VALID", reused_complete_artifact=False, search_updates=0, refit_updates=0, search_exposures=0, refit_exposures=0),
                    prediction=dict(score={"n_origins": count}, metadata=dict(reused_prediction_cache=False),
                                    cost_record=dict(actual_F_streams=count * spec["latent_k"], expected_F_streams=count * spec["latent_k"])))

    def test_original_fit_then_full_V_only_zero_updates_contract(self):
        run_id = next(iter(self.r.CONDITIONS))
        ds, k = self.r.CONDITIONS[run_id]
        spec = self.r.expected_spec(run_id, ds, k)
        result, calls = self.outcome(spec), []
        def fit(*args):
            calls.append(("fit", args))
            return result["fit_receipt"]
        def predict(*args, **kwargs):
            calls.append(("predict", args, kwargs))
            return result["prediction"]
        value = self.r.fit_and_predict(fit, predict, spec, {}, "new_cache", "new_public", "foundation", 100, 43)
        self.assertEqual(value, result)
        self.assertEqual(calls[1][1][2], "V_select")
        self.assertEqual(calls[0][1][2:4], ("new_cache", "new_public"))
        self.assertEqual(calls[1][1][3:5], ("new_cache", "new_public"))
        with self.assertRaises(ValueError):
            self.r.fit_and_predict(fit, predict, spec, {}, "c", "p", "f", 100, 43, split="T_final")

    def test_reuse_training_and_stream_mismatch_are_rejected(self):
        run_id = next(iter(self.r.CONDITIONS))
        ds, k = self.r.CONDITIONS[run_id]
        spec = self.r.expected_spec(run_id, ds, k)
        for mutation in ("reuse", "updates", "exposures", "prediction_reuse", "streams", "cohort"):
            outcome = self.outcome(spec)
            if mutation == "reuse": outcome["fit_receipt"]["reused_complete_artifact"] = True
            if mutation == "updates": outcome["fit_receipt"]["search_updates"] = 1
            if mutation == "exposures": outcome["fit_receipt"]["refit_exposures"] = 16
            if mutation == "prediction_reuse": outcome["prediction"]["metadata"]["reused_prediction_cache"] = True
            if mutation == "streams": outcome["prediction"]["cost_record"]["actual_F_streams"] += 1
            if mutation == "cohort": outcome["prediction"]["score"]["n_origins"] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.r.fit_and_predict(lambda *a: outcome["fit_receipt"], lambda *a, **kw: outcome["prediction"], spec, {}, "c", "p", "f", 100, 43)

    def test_fresh_paths_and_runtime_environment(self):
        with tempfile.TemporaryDirectory() as td:
            public, cache, asset, plan = self.fixture(Path(td))
            run_id = next(iter(self.r.CONDITIONS))
            self.r.require_fresh_paths(public, cache, run_id)
            previous = cache / "runs" / run_id
            previous.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                self.r.require_fresh_paths(public, cache, run_id)
            env = dict(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", INFORMS_ASSET_ROOT=str(asset), INFORMS_CACHE_ROOT=str(cache / self.r.NAME))
            self.r.require_environment(public, cache, asset, env)
            with self.assertRaises(ValueError):
                self.r.require_environment(public, cache, asset, dict(env, INFORMS_GPU_TOKEN="not_allowed"))
            with self.assertRaises(ValueError):
                self.r.require_environment(public, cache, asset, dict(env, INFORMS_CACHE_ROOT=str(cache)))


if __name__ == "__main__":
    unittest.main()
