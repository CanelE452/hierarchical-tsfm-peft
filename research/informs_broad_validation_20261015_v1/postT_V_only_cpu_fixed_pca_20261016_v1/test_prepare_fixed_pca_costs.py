"""Synthetic per-route PCA cost preparation; no numeric arrays or models."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class CostPreparation(unittest.TestCase):
    def setUp(self):
        path = HERE / "prepare_fixed_pca_costs.py"
        self.assertTrue(path.is_file(), "per-route PCA preparation is not implemented")
        self.h = load("tested_fixed_pca_costs", path)
        fixture = load("original_fixed_pca_fixture", HERE / "test_fixed_pca_runner.py")
        fixture.FixedPCATests.setUpClass()
        self.fixture_builder = fixture.FixedPCATests()

    def fixture(self, folder, dataset="pems08", complete=True):
        h = self.h
        public, cache, asset, plan = self.fixture_builder.fixture(Path(folder))
        runner = self.fixture_builder.r
        post, private = public / runner.NAME, cache / runner.NAME
        private.mkdir()
        for index, name in enumerate(("evaluation.py", "measure_costs.py", "route_catalog.py", "deployment.py")):
            old = "tiny%d.py" % index
            plan["source39"].pop(old)
            (public / name).write_text("# synthetic frozen API\n", encoding="utf-8")
            plan["source39"][name] = h.sha(public / name)
        data = runner.read(public / "DATA_CONTRACT.json")
        for ds, count in (("beijing", 43), ("jena", 43), ("electricity", 50), ("pems08", 148)):
            data["assets"][ds]["legal_origin_counts_primary"]["V_select"] = count
            plan["task_contracts"][ds]["legal_origin_counts_primary"]["V_select"] = count
        (public / "DATA_CONTRACT.json").write_text(json.dumps(data), encoding="utf-8")
        plan["immutable_public"]["DATA_CONTRACT.json"] = h.sha(public / "DATA_CONTRACT.json")
        (post / "PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
        self.h.PLAN_SHA = h.sha(post / "PLAN.json")
        condition = next(row for row in plan["conditions"] if row["spec"]["dataset"] == dataset)
        run_id, task = condition["run_id"], plan["task_contracts"][dataset]
        if not complete:
            return public, cache, asset, plan, condition
        (post / "EXPERIMENT_REGISTRY.csv").write_bytes((post / "REGISTRY.csv").read_bytes())
        run = post / "runs" / run_id
        run.mkdir(parents=True)
        artifact = private / "runs" / run_id / "selected_bundle.pt"
        artifact.parent.mkdir(parents=True); artifact.write_bytes(b"synthetic adapter bytes; never loaded")
        archive = private / "predictions" / run_id / "V_select.npz"
        archive.parent.mkdir(parents=True); archive.write_bytes(b"synthetic archive bytes; never loaded")
        receipt = dict(status="COMPLETE_VALID", run_id=run_id, family="PCA_FIXED_BOLT", optimizer="AdamW",
            epochs=[.25,1.,3.], reused_complete_artifact=False, artifact_cache_relative_path=artifact.relative_to(private).as_posix(),
            artifact_sha256=h.sha(artifact), source_sha="synthetic_training_source", search_updates=0, refit_updates=0,
            search_exposures=0, refit_exposures=0, search_fit_end=task["primary_split_ranges"]["A_fit"][1],
            refit_fit_end=task["primary_split_ranges"]["A_stop"][1], refit_scaler_sha=task["refit_scaler_sha256"])
        runner.new(run / "BASE_FIT_RECEIPT.json", receipt)
        runner.new(run / "SOURCE_CONTRACT.json", dict(spec=condition["spec"], data_sha=task["data_sha256"],
            model_sha=plan["foundation_tensor_sha256"], source_sha=receipt["source_sha"],
            files={name: plan["source39"][name] for name in h.FIT_SOURCES}))
        stage = dict(optimizer="NONE_FIXED_TRANSFORM", updates=0, exposures=0, actual_epochs=0., trainable_parameters=0, F_training_logical_streams=0)
        for name in ("search", "refit"):
            (run / name).mkdir(); runner.new(run / name / "RECEIPT.json", stage)
        v = post / "predictions" / run_id / "V_select"; v.mkdir(parents=True)
        count, horizon, channels = task["legal_origin_counts_primary"]["V_select"], task["H"], task["shape"][1]
        arrays = {key:dict(shape=shape, dtype="synthetic", sha256=key) for key,shape in dict(pred_normalized=[count,horizon,channels],
            target_normalized=[count,horizon,channels], observed=[count,horizon,channels], origins=[count], channel_ids=[channels]).items()}
        stats = v / "PUBLIC_SUFFICIENT_STATISTICS.csv.gz"; stats.write_bytes(b"synthetic metadata-only statistic bytes")
        manifest = dict(run_id=run_id, split="V_select", cohort=f"{dataset}:primary:V_select:stride{horizon}",
            data_sha=task["data_sha256"], scaler_sha=task["refit_scaler_sha256"], source_sha="prediction_source", config_sha="config",
            model_sha="prediction_model", arrays=arrays, cache_relative_path=archive.relative_to(private).as_posix(), archive_sha256=h.sha(archive),
            public_stats_relative_path=stats.relative_to(post).as_posix(),
            metadata=dict(family="PCA_FIXED_BOLT", fold="primary", artifact_sha256=receipt["artifact_sha256"],
                canonical_scaler_fit_end=task["primary_split_ranges"]["A_stop"][1], reused_prediction_cache=False),
            score=dict(run_id=run_id, split="V_select", n_origins=count, horizon=horizon, num_channels=channels),
            cost_record=dict(actual_F_streams=count*condition["spec"]["latent_k"], expected_F_streams=count*condition["spec"]["latent_k"]))
        runner.new(v / "PREDICTION_MANIFEST.json", manifest)
        reference = {key:manifest[key] for key in h.IDENTITY}
        reference.update(arrays={k:value for k,value in arrays.items() if k != "pred_normalized"}, expected_F_streams=count*condition["spec"]["latent_k"])
        runner.new(v / "REFERENCE_CONTRACT.json", reference)
        with (v / "SCORES.csv").open("x", encoding="utf-8", newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(manifest["score"]));writer.writeheader();writer.writerow(manifest["score"])
        return public, cache, asset, plan, condition

    def env(self, cache, asset):
        values={key:value for key,value in os.environ.items() if not key.startswith("INFORMS_GPU_")}
        values.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", INFORMS_ASSET_ROOT=str(asset), INFORMS_CACHE_ROOT=str(cache / HERE.name))
        return values

    def test_metadata_missing_one_route_is_no_science_no_write(self):
        before=set(sys.modules)
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,plan,condition=self.fixture(folder,complete=False)
            files={str(p):self.h.sha(p) for p in Path(folder).rglob("*") if p.is_file()}
            state=self.h.metadata(public,cache,asset,condition["run_id"],self.h.PLAN_SHA)
            self.assertFalse(state["ready"]);self.assertTrue(state["missing"])
            self.assertEqual(files,{str(p):self.h.sha(p) for p in Path(folder).rglob("*") if p.is_file()})
        self.assertFalse({"numpy","torch"} & (set(sys.modules)-before))

    def test_single_ready_pems_H12_and_jena_are_independent_of_other_five(self):
        for dataset, horizon, count in (("pems08",12,148),("jena",48,43)):
            with self.subTest(dataset=dataset),tempfile.TemporaryDirectory() as folder:
                public,cache,asset,plan,c=self.fixture(folder,dataset)
                state=self.h.metadata(public,cache,asset,c["run_id"],self.h.PLAN_SHA)
                self.assertTrue(state["ready"])
                self.assertEqual((state["binding"]["H"],state["binding"]["V_origins"]),(horizon,count))
                missing=next(row for row in plan["conditions"] if row["run_id"]!=c["run_id"])
                self.assertFalse(self.h.metadata(public,cache,asset,missing["run_id"],self.h.PLAN_SHA)["ready"])

    def test_wrong_alias_source_and_recipe_or_plan_SHA_are_rejected(self):
        for change in ("alias","source","recipe","plan_sha"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                public,cache,asset,plan,c=self.fixture(folder)
                post=public/HERE.name
                if change=="alias": (post/"EXPERIMENT_REGISTRY.csv").write_text("changed",encoding="utf-8")
                elif change=="source": (public/"campaign.py").write_text("changed",encoding="utf-8")
                elif change=="recipe":
                    path=post/"runs"/c["run_id"]/"SOURCE_CONTRACT.json";value=self.h.read(path)
                    value["spec"]["lr"]*=2;path.write_text(json.dumps(value),encoding="utf-8")
                if change=="recipe":
                    self.assertFalse(self.h.metadata(public,cache,asset,c["run_id"],self.h.PLAN_SHA)["ready"])
                else:
                    with self.assertRaises(ValueError): self.h.metadata(public,cache,asset,c["run_id"],"wrong" if change=="plan_sha" else self.h.PLAN_SHA)

    def canonical(self,state):
        b=state["binding"];r=b["receipt"];m=b["manifest"]
        return dict(run_id=b["run_id"],dataset=b["dataset"],family="PCA_FIXED_BOLT",fold="primary",latent_k=b["K"],
            canonical_fit_end=b["canonical_fit_end"],foundation_sha=state["plan"]["foundation_tensor_sha256"],model_size="small",
            artifact_cache_relative_path=r["artifact_cache_relative_path"],artifact_sha256=r["artifact_sha256"],expected_prediction_model_sha=m["model_sha"],
            fit_receipt_public_relative_path=b["receipt_relative"],fit_receipt_sha256=b["receipt_sha256"],
            standard_V_binding=dict(model_sha=m["model_sha"],archive_sha256=m["archive_sha256"],manifest_public_relative_path=b["manifest_relative"],manifest_sha256=b["manifest_sha256"]))

    def test_canonical_spec_preserved_and_wrong_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,plan,c=self.fixture(folder)
            state=self.h.metadata(public,cache,asset,c["run_id"],self.h.PLAN_SHA);original=self.canonical(state)
            result=self.h.cost_spec(original,state["binding"],plan)
            self.assertEqual({k:v for k,v in result.items() if k not in {"accuracy_run_id","row_microbatch"}},original)
            with self.assertRaises(ValueError): self.h.cost_spec(dict(original,artifact_sha256="changed"),state["binding"],plan)

    def test_fake_runtime_prepares_two_original_workers_and_no_subprocess(self):
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,plan,c=self.fixture(folder)
            state=self.h.metadata(public,cache,asset,c["run_id"],self.h.PLAN_SHA);owner=self
            class Runtime:
                calls=[]
                def task(self,*args): self.calls.append("task");return "fake task"
                def route(self,*args): self.calls.append("route");return owner.canonical(state),"small"
            runtime=Runtime()
            with patch.dict(os.environ,self.env(cache,asset),clear=True),patch("subprocess.Popen",side_effect=AssertionError("No workers may start")):
                result=self.h.execute(public,cache,asset,self.h.PLAN_SHA,c["run_id"],public/HERE.name/"prepared",_runtime=runtime)
            self.assertEqual(runtime.calls,["task","route"])
            self.assertEqual([j["batch"] for j in result["jobs"]],[1,8]);self.assertEqual(result["actual_cost_trials"],0)
            for job in result["jobs"]:
                self.assertEqual((job["H"],job["V_origins"]),(12,148))
                self.assertEqual(job["expected_F_streams_per_trial"],job["batch"]*job["K"])
                self.assertEqual(self.h.sha(cache/HERE.name/job["spec_cache_relative_path"]),job["spec_sha256"])
                for flag,value in (("--device","cpu"),("--warmups","10"),("--timed-trials","20"),("--stage-trials","20")):
                    self.assertEqual(job["arguments"][job["arguments"].index(flag)+1],value)

    def test_existing_output_wrong_environment_and_absent_alias_stop_before_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,plan,c=self.fixture(folder);post=public/HERE.name
            existing=post/"exists";existing.mkdir()
            with patch.dict(os.environ,self.env(cache,asset),clear=True):
                with self.assertRaises(FileExistsError): self.h.execute(public,cache,asset,self.h.PLAN_SHA,c["run_id"],existing,_runtime=object())
            env=self.env(cache,asset);env["CUDA_VISIBLE_DEVICES"]="0"
            with patch.dict(os.environ,env,clear=True):
                with self.assertRaises(ValueError): self.h.execute(public,cache,asset,self.h.PLAN_SHA,c["run_id"],post/"new",_runtime=object())
            (post/"EXPERIMENT_REGISTRY.csv").unlink()
            self.assertFalse(self.h.metadata(public,cache,asset,c["run_id"],self.h.PLAN_SHA)["ready"])
            self.assertFalse((post/"EXPERIMENT_REGISTRY.csv").exists())

    def test_actual_zero_fixed_stage_activity_is_required_despite_BASE_AdamW(self):
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,plan,c=self.fixture(folder)
            self.assertTrue(self.h.metadata(public,cache,asset,c["run_id"],self.h.PLAN_SHA)["ready"])
            path=public/HERE.name/"runs"/c["run_id"]/"refit/RECEIPT.json"
            value=self.h.read(path);value["updates"]=1;path.write_text(json.dumps(value),encoding="utf-8")
            self.assertFalse(self.h.metadata(public,cache,asset,c["run_id"],self.h.PLAN_SHA)["ready"])


if __name__=="__main__":
    unittest.main()
