"""Tiny original-root reference fixtures only; no real model/data/worker calls."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
HELPER = HERE/"prepare_reference_cpu_costs.py"


def load():
    spec=importlib.util.spec_from_file_location("synthetic_reference_cost_helper",HELPER)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class ReferenceCostTests(unittest.TestCase):
    def setUp(self):
        self.h=load()

    def fixture(self,root,family):
        h=self.h
        public,cache,asset=(root/name for name in ("public","cache","asset"))
        public.mkdir();cache.mkdir();asset.mkdir()
        run_id=next(key for key,value in h.REFERENCES.items() if value["family"]==family)
        reference=copy.deepcopy(h.REFERENCES[run_id])
        source_names=("prediction_cache.py","fit_base.py","data_contract.py","training.py","evaluation.py","networks.py","gpu_runtime.py","ridge_baseline.py")
        plan=dict(source39={name:"tiny-source-"+name for name in source_names},immutable_public={"EXPERIMENT_REGISTRY.csv":"tiny","SCORES.csv":"tiny"},
            task_contract=dict(data_sha256="tiny-data",refit_scaler_sha256="tiny-scaler"),foundation_tensor_sha256="tiny-foundation",model_revision="tiny-revision",
            model_asset_relative_path="tiny-model/tiny-revision",deadline_unix=time.time()+30,max_RSS_bytes=4294967296,storage_limit_bytes=30*1024**3)
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("x",encoding="utf-8") as stream:json.dump(value,stream)
        def rows(path):
            with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))
        def csv_write(path,values):
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("x",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values));writer.writeheader();writer.writerow(values)
        def relative(root,name):
            path=(Path(root)/name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(Path(root).resolve()):raise ValueError("escaped")
            return path
        runner=types.SimpleNamespace(NAME=HERE.name,read=lambda path:json.loads(Path(path).read_text()),csv_rows=rows,relative=relative,new=write,
            metadata=lambda *args:plan,validate_task=lambda *a:{"tiny-task-binding":True},save_private_failure=lambda *a:dict(error_class=type(a[-1]).__name__),
            sha=h.sha)
        folder=public/"predictions"/run_id/"V_select"
        folder.mkdir(parents=True)
        sources={key:plan["source39"][key] for key in source_names if key!="ridge_baseline.py" or family=="RAW_RIDGE"}
        receipt=None
        training_source="NO_TRAINING_FIXED_BASELINE" if family=="RAW" else "tiny-training-source"
        if family!="RAW":
            artifact_relative=f"ridge/{run_id}/artifact.npz" if family=="RAW_RIDGE" else f"runs/{run_id}/selected_bundle.pt"
            artifact=cache/artifact_relative;artifact.parent.mkdir(parents=True);artifact.write_bytes(b"tiny-unopened-artifact")
            receipt=dict(status="COMPLETE_VALID",run_id=run_id,family=family,source_sha=training_source,
                artifact_cache_relative_path=artifact_relative,artifact_sha256=h.sha(artifact),model_sha="native-ridge-model",training_status="UNDERTRAINED_OR_BUDGET_LIMIT")
            if family=="RAW_RIDGE":receipt.update(fit_end=12579,scaler_sha="tiny-scaler",source_files={key:plan["source39"][key] for key in ("data_contract.py","ridge_baseline.py")})
            else:receipt.update(refit_fit_end=12579,refit_scaler_sha="tiny-scaler")
            path=public/"runs"/run_id/reference["receipt_file"];write(path,receipt);reference["receipt_sha256"]=h.sha(path)
            if family=="DIRECT_G":
                path=path.with_name("SOURCE_CONTRACT.json")
                write(path,dict(model_sha="NO_FOUNDATION",spec=dict(device="cpu",family="DIRECT_G")))
                reference["source_sha256"]=h.sha(path)
        else:
            write(public/"PROBE_ACTUAL_GPU.json",dict(model_state_sha256="tiny-foundation",model_revision="tiny-revision"))
            h.PROBE_SHA=h.sha(public/"PROBE_ACTUAL_GPU.json")
        shapes=dict(pred_normalized=[43,48,21],target_normalized=[43,48,21],observed=[43,48,21],origins=[43],channel_ids=[21])
        arrays={key:dict(shape=shape,sha256="tiny-"+key,dtype="float32") for key,shape in shapes.items()}
        truth={key:value for key,value in arrays.items() if key!="pred_normalized"}
        h.TRUTH_SHA=h.fingerprint(truth)
        identity=dict(run_id=run_id,split="V_select",cohort="jena:primary:V_select:stride48",source_sha=h.fingerprint(dict(deployment_sources=sources,training_source_sha=training_source)),
            config_sha="tiny-config",data_sha="tiny-data",model_sha="standard-V-model",scaler_sha="tiny-scaler")
        score=dict(run_id=run_id,split="V_select",cohort=identity["cohort"],mse=.4,mae=.3,n_observed=43344,n_origins=43,num_channels=21,horizon=48,active_channels=21,metric_scale="A_normalized_original_channels")
        archive_relative=f"predictions/{run_id}/V_select.npz";archive=cache/archive_relative;archive.parent.mkdir(parents=True);archive.write_bytes(b"tiny-unopened-V-archive")
        manifest=dict(identity,cache_relative_path=archive_relative,archive_sha256=h.sha(archive),arrays=arrays,
            metadata=dict(family=family,fold="primary",dense=False,role="base_parent",parent_fit_end=12579,canonical_scaler_fit_end=12579,
                artifact_sha256=receipt["artifact_sha256"] if receipt else "tiny-pretrained-model",training_source_sha=training_source,source_files=sources),
            score=score,cost_record=dict(run_id=run_id,batch=43,latent_k=reference["F_width"],actual_F_streams=43*reference["F_width"],expected_F_streams=43*reference["F_width"],F_calls=1))
        write(folder/"PREDICTION_MANIFEST.json",manifest);reference["manifest_sha256"]=h.sha(folder/"PREDICTION_MANIFEST.json")
        write(folder/"REFERENCE_CONTRACT.json",dict(identity,arrays=truth,expected_F_streams=43*reference["F_width"]))
        csv_write(folder/"SCORES.csv",score)
        global_score=dict({key:value for key,value in score.items() if key!="metric_scale"},truth_cohort_sha=h.TRUTH_SHA,
            manifest_relative_path=(folder/"PREDICTION_MANIFEST.json").relative_to(public).as_posix(),archive_sha256=manifest["archive_sha256"],
            **{key:identity[key] for key in ("source_sha","config_sha","model_sha","data_sha","scaler_sha")},training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if family=="DIRECT_G" else "CLOSED_FORM_OR_PRETRAINED",checkpoint_replay="NOT_AVAILABLE")
        csv_write(public/"SCORES.csv",global_score)
        csv_write(public/"EXPERIMENT_REGISTRY.csv",dict(experiment_id=run_id,dataset="jena",family=reference["registry_family"],fold="primary",seed=92601,latent_k=0,length=512,horizon=48,execution_status="COMPLETE_VALID"))
        h.REFERENCES=dict(h.REFERENCES,**{run_id:reference})
        return public,cache,asset,runner,plan,run_id

    def canonical(self,binding,plan):
        receipt=binding["receipt"];manifest=binding["manifest"]
        result=dict(run_id=binding["run_id"],dataset="jena",family=binding["family"],fold="primary",latent_k=binding["F_width"],canonical_fit_end=12579,model_size=binding["model_size"],
            standard_V_binding=dict(model_sha=manifest["model_sha"],archive_sha256=manifest["archive_sha256"],manifest_public_relative_path=binding["manifest_relative"],manifest_sha256=binding["metadata_pins"][binding["manifest_relative"]]))
        if receipt is None:result.update(foundation_sha=plan["foundation_tensor_sha256"],foundation_revision=plan["model_revision"],expected_prediction_model_sha=manifest["model_sha"])
        else:result.update(artifact_cache_relative_path=receipt["artifact_cache_relative_path"],artifact_sha256=receipt["artifact_sha256"],
            fit_receipt_public_relative_path=binding["receipt_relative"],fit_receipt_sha256=binding["metadata_pins"][binding["receipt_relative"]],foundation_sha="NO_FOUNDATION",
            expected_prediction_model_sha=receipt["model_sha"] if binding["family"]=="RAW_RIDGE" else manifest["model_sha"])
        return result

    def test_default_metadata_starts_no_science_child_or_file_write(self):
        result=subprocess.run([sys.executable,"-B",str(HELPER)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        value=json.loads(result.stdout)
        self.assertFalse(value["numpy_imported"] or value["torch_imported"] or value["CPU_V_parity_certified"])
        self.assertEqual(value["files_written"],0);self.assertEqual(value["actual_model_array_worker_or_cost_execution"],0)

    def test_each_original_reference_is_independent_of_other_two_missing(self):
        for family in ("RAW","RAW_RIDGE","DIRECT_G"):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as folder:
                public,cache,asset,runner,plan,run_id=self.fixture(Path(folder),family)
                with mock.patch.object(self.h,"load_runner",return_value=runner):
                    state=self.h.metadata(public,cache,asset,run_id,self.h.PLAN_SHA)
                self.assertTrue(state["ready"],state)
                self.assertTrue(state["other_two_references_are_not_a_gate"])
                self.assertEqual(state["binding"]["F_width"],21 if family=="RAW" else 0)
                self.assertEqual(state["binding"]["model_size"],"small" if family=="RAW" else None)
                self.assertIn("NOT_CHECKED",state["binding"]["cpu_vs_original_V_numeric_parity"])

    def test_frozen_source_manifest_and_truth_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,runner,plan,run_id=self.fixture(Path(folder),"DIRECT_G")
            self.h.reference_binding(runner,plan,public,cache,run_id)
            wrong=copy.deepcopy(plan);wrong["source39"]["prediction_cache.py"]="changed"
            with self.assertRaises(ValueError):self.h.reference_binding(runner,wrong,public,cache,run_id)
            path=public/"predictions"/run_id/"V_select/PREDICTION_MANIFEST.json"
            path.write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError):self.h.reference_binding(runner,plan,public,cache,run_id)
            with self.assertRaises(ValueError):self.h.choose("unregistered")

    def test_canonical_specs_keep_raw_noartifact_and_zeroF_native_ridge_model_identity(self):
        for family in ("RAW","RAW_RIDGE","DIRECT_G"):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as folder:
                public,cache,_,runner,plan,run_id=self.fixture(Path(folder),family)
                binding,_=self.h.reference_binding(runner,plan,public,cache,run_id)
                canonical=self.canonical(binding,plan)
                expected=dict(canonical,accuracy_run_id=run_id,row_microbatch=128)
                self.assertEqual(self.h.cost_spec(canonical,binding["model_size"],binding,plan),expected)
                if family=="RAW":self.assertNotIn("artifact_cache_relative_path",expected)
                if family=="RAW_RIDGE":self.assertNotEqual(expected["expected_prediction_model_sha"],expected["standard_V_binding"]["model_sha"])
                wrong=dict(canonical,model_size="small" if family!="RAW" else None)
                with self.assertRaises(ValueError):self.h.cost_spec(wrong,binding["model_size"],binding,plan)
                with self.assertRaises(ValueError):self.h.cost_spec(dict(canonical,accuracy_run_id="invented"),binding["model_size"],binding,plan)

    def test_exact_worker_arguments_use_original_public_cache_and_new_spec_output(self):
        for batch in (1,8):
            argv=self.h.worker_arguments("reference_cost_specs/unique/raw.json","costs/reference_unique/raw.json",batch,"model/revision")
            def arg(name):return argv[argv.index(name)+1]
            self.assertEqual(arg("--public-root"),"{ORIGINAL_PUBLIC_ROOT}")
            self.assertEqual(arg("--cache-root"),"{ORIGINAL_CACHE_ROOT}")
            self.assertEqual(arg("--route-spec"),"{POST_T_CACHE_ROOT}/reference_cost_specs/unique/raw.json")
            self.assertEqual(arg("--output"),"{POST_T_PUBLIC_ROOT}/costs/reference_unique/raw.json")
            self.assertEqual((arg("--device"),arg("--batch"),arg("--warmups"),arg("--timed-trials"),arg("--stage-trials")),("cpu",str(batch),"10","20","20"))
            self.assertIn("--worker",argv)

    def test_actual_only_archive_byte_binding_rejects_changed_original_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            public,cache,_,runner,plan,run_id=self.fixture(Path(folder),"RAW_RIDGE")
            binding,_=self.h.reference_binding(runner,plan,public,cache,run_id)
            result=self.h.bound_archives(runner,cache,binding)
            self.assertEqual(len(result),2)
            path=cache/binding["receipt"]["artifact_cache_relative_path"];path.write_bytes(b"wrong")
            with self.assertRaises(ValueError):self.h.bound_archives(runner,cache,binding)

    def test_execute_fake_runtime_calls_original_roots_no_child_no_foundation_and_preserves_status(self):
        h=self.h
        for family in ("RAW","RAW_RIDGE","DIRECT_G"):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as folder:
                public,cache,asset,runner,plan,run_id=self.fixture(Path(folder),family)
                post,private=public/runner.NAME,cache/runner.NAME;post.mkdir();private.mkdir()
                binding,_=h.reference_binding(runner,plan,public,cache,run_id)
                canonical=self.canonical(binding,plan);calls=[]
                runtime=types.SimpleNamespace(task=lambda *a:(calls.append(("task",a)) or ({"tiny":True},{"tiny-binding":True})),
                    route=lambda *a:(calls.append(("route",a)) or (canonical,binding["model_size"])))
                env=dict(CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                    INFORMS_ASSET_ROOT=str(asset),INFORMS_CACHE_ROOT=str(private))
                process=types.SimpleNamespace(pid=123,create_time=lambda:1.,memory_info=lambda:types.SimpleNamespace(rss=1))
                before={str(path):h.sha(path) for root in (public,cache) for path in root.rglob("*") if path.is_file()}
                with mock.patch.object(h,"load_runner",return_value=runner),mock.patch.dict(os.environ,env,clear=True), \
                     mock.patch.dict(sys.modules,psutil=types.SimpleNamespace(Process=lambda:process)), \
                     mock.patch.object(h,"ScienceRuntime",side_effect=AssertionError("real science runtime forbidden")):
                    result=h.execute(public,cache,asset,h.PLAN_SHA,run_id,post/"newpreparation",deadline=time.time()+10,_runtime=runtime)
                    with self.assertRaises(FileExistsError):h.execute(public,cache,asset,h.PLAN_SHA,run_id,post/"newpreparation",deadline=time.time()+10,_runtime=runtime)
                self.assertEqual(result["status"],"PREPARED_NOT_EXECUTED_CPU_COST_WORKERS",result)
                self.assertEqual([job["batch"] for job in result["jobs"]],[1,8])
                self.assertEqual([job["expected_F_streams_per_trial"] for job in result["jobs"]],[binding["F_width"],8*binding["F_width"]])
                self.assertEqual(calls[0][1][-1],cache)
                self.assertEqual(calls[1][1][-2:],(cache,public))
                self.assertEqual(result["actual_subprocesses_started"],0);self.assertEqual(result["actual_foundation_models_constructed"],0)
                self.assertEqual(result["original_training_status"],binding["original_training_status"])
                self.assertTrue(all(h.sha(Path(name))==digest for name,digest in before.items()))
                self.assertTrue((post/"newpreparation/COST_EXECUTION_PLAN.JSON").is_file())

    def test_route_exception_is_preserved_without_fabricated_jobs_or_stopping_other_routes(self):
        h=self.h
        with tempfile.TemporaryDirectory() as folder:
            public,cache,asset,runner,plan,run_id=self.fixture(Path(folder),"DIRECT_G")
            post,private=public/runner.NAME,cache/runner.NAME;post.mkdir();private.mkdir()
            runtime=types.SimpleNamespace(task=lambda *a:({},{}),route=mock.Mock(side_effect=RuntimeError("synthetic selected artifact error")))
            env=dict(CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",INFORMS_ASSET_ROOT=str(asset),INFORMS_CACHE_ROOT=str(private))
            process=types.SimpleNamespace(pid=123,create_time=lambda:1.,memory_info=lambda:types.SimpleNamespace(rss=1))
            with mock.patch.object(h,"load_runner",return_value=runner),mock.patch.dict(os.environ,env,clear=True), \
                 mock.patch.dict(sys.modules,psutil=types.SimpleNamespace(Process=lambda:process)):
                result=h.execute(public,cache,asset,h.PLAN_SHA,run_id,post/"failed",deadline=time.time()+10,_runtime=runtime)
            self.assertEqual(result["status"],"FAILED_PRESERVED")
            self.assertEqual(result["worker_count"],0);self.assertEqual(result["actual_forecasting"],0)
            self.assertTrue(result["other_two_references_are_not_a_gate"])
            self.assertEqual(result["error_class"],"RuntimeError")


if __name__=="__main__":unittest.main(verbosity=2)
