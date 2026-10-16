"""Family verification fixtures only; ROOT executes after precision-cost quiet."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent


def load():
    spec=importlib.util.spec_from_file_location("tiny_family_output_verifier",HERE/"verify_family_minima_outputs.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class FamilyVerifierTests(unittest.TestCase):
    def setUp(self):
        self.h=load()
        self.runner=self.h.load_runner()
        upstream={}
        for entry in self.runner.TARGETS:
            spec=self.runner.original_spec(*entry);upstream[spec["run_id"]]=dict(spec=spec)
        self.conditions=self.runner.conditions(upstream)

    def fixture(self,root,index=0,epoch=0.):
        h,r=self.h,self.runner
        public,cache=root/"public",root/"cache"
        post,private=public/r.NAME,cache/r.NAME
        condition=self.conditions[index];dataset=condition["dataset"];run_id=condition["run_id"]
        contracts={}
        for ds,c,a50,a60,n in (("jena",21,10482,12579,20965),("beijing",11,10519,12622,21038)):
            contracts[ds]=dict(shape=[n,c],L=512,H=48,columns=["c"+str(i) for i in range(c)],array_sha256=dict(physical="opaque"),
                data_sha256=ds+"-data",search_scaler_sha256=ds+"-A50",refit_scaler_sha256=ds+"-A60",
                legal_origin_counts_primary=dict(V_select=43),primary_split_ranges=dict(A_fit=[0,a50],A_stop=[a50,a60]))
        task=contracts[dataset];c=task["shape"][1];width=condition["effective_F_width"]
        prediction_names=("prediction_cache.py","fit_base.py","data_contract.py","training.py","evaluation.py","networks.py","gpu_runtime.py")
        sources={name:name+"-sha" for name in set(r.SOURCES)|set(prediction_names)}
        plan=dict(conditions=self.conditions,task_contracts=contracts,source39=sources,foundation_tensor_sha256="foundation",
            postT_registry_relative_path="REGISTRY.csv",runner_sha256=h.RUNNER_SHA)
        run,pred=post/"runs"/run_id,post/"predictions"/run_id/"V_select"
        run.mkdir(parents=True);pred.mkdir(parents=True)
        artifact=private/"runs"/run_id/"selected_bundle.pt";artifact.parent.mkdir(parents=True);artifact.write_bytes(b"opaque family adapter")
        archive=private/"predictions"/run_id/"V_select.npz";archive.parent.mkdir(parents=True);archive.write_bytes(b"opaque family V")
        fit_sources={name:sources[name] for name in r.SOURCES}
        spec=condition["spec"]
        axes=dict(L=512,H=48,C=c,columns=task["columns"],fold="primary",source_array_fingerprints=task["array_sha256"])
        source=dict(spec=spec,files=fit_sources,source_sha=r.fingerprint(fit_sources),config_sha=r.fingerprint(dict(spec=spec,axes=axes)),
            data_sha=task["data_sha256"],model_sha="foundation")
        n50,n60=task["primary_split_ranges"]["A_fit"][1]-559,task["primary_split_ranges"]["A_stop"][1]-559
        targets=[0]+[math.ceil(e*n50) for e in (.25,1.,3.)]
        updates=sum(math.ceil((b-a)/16) for a,b in zip(targets,targets[1:]));exposures=math.ceil(epoch*n60)
        base=dict(status="COMPLETE_VALID",run_id=run_id,family=condition["family"],epochs=[.25,1.,3.],seed=92601,selected_epoch=epoch,
            reused_complete_artifact=False,search_exposures=3*n50,search_updates=updates,refit_exposures=exposures,refit_updates=math.ceil(exposures/16),
            search_fit_end=task["primary_split_ranges"]["A_fit"][1],refit_fit_end=task["primary_split_ranges"]["A_stop"][1],
            normalization_fit_end=task["primary_split_ranges"]["A_stop"][1],refit_scaler_sha=task["refit_scaler_sha256"],source_sha=source["source_sha"],
            artifact_cache_relative_path=artifact.relative_to(private).as_posix(),artifact_sha256=h.sha(artifact),model_state_sha="state")
        core={key:source[key] for key in ("source_sha","config_sha","data_sha","model_sha")}
        search=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.,.25,1.,3.],exposures=3*n50,updates=updates,train_origins=n50,
            contract=dict(core,scaler_sha=task["search_scaler_sha256"],initialization_sha="a"*64))
        refit=dict(status="SEGMENT_COMPLETE",completed_epochs=[0.]+([epoch] if epoch else []),exposures=exposures,updates=math.ceil(exposures/16),train_origins=n60,
            contract=dict(core,scaler_sha=task["refit_scaler_sha256"],initialization_sha="b"*64))
        prediction_sources={name:sources[name] for name in prediction_names}
        identity=dict(run_id=run_id,split="V_select",cohort=f"{dataset}:primary:V_select:stride48",
            source_sha=r.fingerprint(dict(deployment_sources=prediction_sources,training_source_sha=source["source_sha"])),
            config_sha=r.fingerprint(dict(parent_spec=spec,dense=False,fold="primary",role="base_parent")),data_sha=task["data_sha256"],
            model_sha=r.fingerprint(dict(artifact_sha=base["artifact_sha256"],model_state_sha="state",normalization=task["refit_scaler_sha256"])),
            scaler_sha=task["refit_scaler_sha256"])
        shapes=dict(pred_normalized=[43,48,c],target_normalized=[43,48,c],observed=[43,48,c],origins=[43],channel_ids=[c])
        arrays={name:dict(shape=shape,dtype="fixture",sha256=name) for name,shape in shapes.items()}
        score=dict(run_id=run_id,split="V_select",cohort=identity["cohort"],mse=.5,mae=.4,n_origins=43,num_channels=c,horizon=48)
        reference=dict(identity,arrays={key:value for key,value in arrays.items() if key!="pred_normalized"},expected_F_streams=43*width)
        manifest=dict(identity,arrays=arrays,score=score,cache_relative_path=archive.relative_to(private).as_posix(),archive_sha256=h.sha(archive),
            metadata=dict(artifact_sha256=base["artifact_sha256"],canonical_scaler_fit_end=task["primary_split_ranges"]["A_stop"][1],
                parent_fit_end=task["primary_split_ranges"]["A_stop"][1],parent_normalization_fit_end=task["primary_split_ranges"]["A_stop"][1],
                family=condition["family"],fold="primary",dense=False,role="base_parent",reused_prediction_cache=False,
                source_files=prediction_sources,training_source_sha=source["source_sha"]),
            cost_record=dict(run_id=run_id,batch=43,latent_k=width,actual_F_streams=43*width,expected_F_streams=43*width,F_calls=1),
            public_stats_relative_path=f"predictions/{run_id}/V_select/PUBLIC_ERROR_STATS.csv.gz")
        for path,value in ((run/"BASE_FIT_RECEIPT.json",base),(run/"SOURCE_CONTRACT.json",source),
                (pred/"PREDICTION_MANIFEST.json",manifest),(pred/"REFERENCE_CONTRACT.json",reference)):
            h.new(path,value)
        for label,value in (("search",search),("refit",refit)):
            (run/label).mkdir();h.new(run/label/"RECEIPT.json",value)
        with (pred/"SCORES.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(score));writer.writeheader();writer.writerow(score)
        (pred/"PUBLIC_ERROR_STATS.csv.gz").write_bytes(b"opaque public statistics")
        (post/"REGISTRY.csv").write_bytes(b"experiment_id\ntiny\n")
        plan["postT_registry_sha256"]=h.sha(post/"REGISTRY.csv")
        return public,cache,plan,condition

    def test_each_of_four_conditions_validates_variable_C_full_V_and_selected_zero(self):
        for index in range(4):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as td:
                public,cache,plan,condition=self.fixture(Path(td),index,0. if index%2==0 else 3.)
                values,_,_,_=self.h.prepare_inputs(self.runner,plan,public,cache,condition["run_id"],condition)
                manifest=values["manifest"];expected=43*condition["effective_F_width"]
                self.assertEqual(manifest["cost_record"]["actual_F_streams"],expected)
                self.assertEqual(manifest["arrays"]["pred_normalized"]["shape"],[43,48,21 if condition["dataset"]=="jena" else 11])
                if condition["family"]=="LORA":
                    self.assertEqual((condition["registered_latent_k"],manifest["cost_record"]["latent_k"]),(0,21))
                    self.assertEqual(expected,903)

    def test_malformed_other_three_do_not_block_one_valid_run_and_selected_invalid_rejects(self):
        h=self.h
        for bad in ([],None,7,"bad",True,"status_list","status_dict"):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as td:
                public,cache,plan,condition=self.fixture(Path(td))
                other=self.conditions[1];folder=public/self.runner.NAME/"runs"/other["run_id"];folder.mkdir(parents=True)
                record=dict(run_id=other["run_id"],family=other["family"],status=[] if bad=="status_list" else {}) if isinstance(bad,str) and bad.startswith("status_") else bad
                (folder/"BASE_FIT_RECEIPT.json").write_text(json.dumps(record),encoding="utf-8")
                h.prepare_inputs(self.runner,plan,public,cache,condition["run_id"],condition)
                projection=h.registry_projection(plan,public/self.runner.NAME)
                self.assertEqual(len(projection),4)
                self.assertEqual(projection[0]["execution_status"],"COMPLETE_VALID")
                self.assertEqual(projection[1]["execution_status"],"UNREADABLE_BASE_FIT_RECEIPT")
                if isinstance(record,dict):record=dict(record,run_id=condition["run_id"],family=condition["family"])
                selected=public/self.runner.NAME/"runs"/condition["run_id"]/"BASE_FIT_RECEIPT.json"
                selected.write_text(json.dumps(record),encoding="utf-8")
                with self.assertRaises(ValueError):h.prepare_inputs(self.runner,plan,public,cache,condition["run_id"],condition)

    def test_source_scaler_stage_model_array_and_artifact_mutations_reject(self):
        for index,name,key,value in ((0,"source_contract","config_sha","changed"),(1,"search","completed_epochs",[0.,.25,1.]),
                (2,"manifest","scaler_sha","A50"),(3,"manifest","model_sha","changed"),(0,"manifest","arrays",{}),
                (1,"refit","contract",{}),(2,"fit_receipt","artifact_sha256","wrong"),(3,"manifest","split","T_final")):
            with self.subTest(index=index,name=name,key=key),tempfile.TemporaryDirectory() as td:
                public,cache,plan,condition=self.fixture(Path(td),index)
                values,_,paths,_=self.h.prepare_inputs(self.runner,plan,public,cache,condition["run_id"],condition)
                record=copy.deepcopy(values[name]);record[key]=value
                paths[name].write_text(json.dumps(record),encoding="utf-8")
                with self.assertRaises((ValueError,KeyError)):
                    self.h.prepare_inputs(self.runner,plan,public,cache,condition["run_id"],condition)

    def test_alias_copies_exact_four_row_bytes_and_preserves_wrong_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            public,_,plan,_=self.fixture(Path(td));post=public/self.runner.NAME
            original=(post/"REGISTRY.csv").read_bytes()
            self.assertTrue(self.h.ensure_registry_alias(self.runner,post,plan)["created_now"])
            self.assertFalse(self.h.ensure_registry_alias(self.runner,post,plan)["created_now"])
            target=post/"EXPERIMENT_REGISTRY.csv";target.write_bytes(b"wrong")
            with self.assertRaises(ValueError):self.h.ensure_registry_alias(self.runner,post,plan)
            self.assertEqual(target.read_bytes(),b"wrong")
            self.assertEqual((post/"REGISTRY.csv").read_bytes(),original)

    def test_rescore_reuses_original_API_scope_and_private_root(self):
        api=mock.Mock(return_value=dict(status="PASS"));manifest=dict(cost_record=dict(actual_F_streams=903))
        self.h.rescore_call(api,manifest,"reference","score",["statistics"],["four statuses"],["four IDs"],"new-private")
        api.assert_called_once_with(manifest,"reference","score",["statistics"],["four statuses"],["four IDs"],[manifest["cost_record"]],"new-private")

    def test_original_CPU_six_origin_single_batch_counts_include_LoRA_C21(self):
        for condition in self.conditions:
            run_id,k=condition["run_id"],condition["effective_F_width"]
            receipt=dict(status="PASS",run_id=run_id,origins=list(range(6)),actual_F_streams=6*k,expected_F_streams=6*k,
                F_physical_streams=6*k,F_calls=1,atol=1e-5,rtol=1e-4,executed_now=True,T_final_accessed=False,
                independent_training_rerun=False,model_size="small",scope="actual_physical_input_selected_checkpoint_V_subset_replay")
            api=mock.Mock(return_value=receipt)
            value=self.h.replay_call(api,run_id,"task","cache","public","model",100.)
            api.assert_called_once_with(run_id,"task","cache","public","model",device="cpu",deadline=100.,lease=None,batch_size=6,atol=1e-5,rtol=1e-4)
            self.h.validate_replay(value,run_id,k)
            self.h.validate_replay(dict(value,executed_now=False),run_id,k)
            for key,wrong in (("F_calls",2),("F_physical_streams",0),("actual_F_streams",0),("atol",1e-3),("T_final_accessed",True)):
                with self.subTest(run=run_id,key=key),self.assertRaises(ValueError):self.h.validate_replay(dict(value,**{key:wrong}),run_id,k)

    def test_forward_receipt_tristate_does_not_promote_cached_or_error_to_fresh(self):
        for attempted,value,expected,returned in ((False,None,False,False),(True,None,None,False),
                (True,dict(executed_now=True),True,True),(True,dict(executed_now=False),False,True),(True,dict(status="bad"),None,True)):
            facts=self.h.forward_facts(attempted,value)
            self.assertIs(facts["actual_checkpoint_forward_in_this_command"],expected)
            self.assertEqual(facts["checkpoint_replay_receipt_returned"],returned)
            self.assertEqual(facts["checkpoint_replay_call_attempted"],attempted)

    def test_pinned_runner_and_required_generic_base_metadata_are_not_interchangeable(self):
        self.assertEqual(self.h.load_runner().NAME,self.runner.NAME)
        base=self.h.load_base()
        self.assertEqual(base.RUNNER_SHA,"e11c34918862a698e962cf2a3c151ea6f7484032985f52d25cb7220da6b5d37b")
        self.assertNotEqual(self.runner.NAME,base.HERE.name)
        with mock.patch.object(self.h,"sha",return_value="changed"):
            with self.assertRaises(ValueError):self.h.load_runner()
            with self.assertRaises(ValueError):self.h.load_base()

    def test_future_sealed_plan_sha_is_explicit_and_keeps_four_family_conditions(self):
        runner=mock.Mock()
        runner.metadata.return_value=dict(runner_sha256=self.h.RUNNER_SHA,conditions=self.conditions)
        with self.assertRaises(ValueError):self.h.metadata(runner,None,None,None,None)
        runner.metadata.assert_not_called()
        with mock.patch.object(self.h,"load_base",return_value=object()):
            result=self.h.metadata(runner,"public","cache","asset","a"*64)
            self.assertEqual(result["conditions"],self.conditions)
            runner.metadata.assert_called_once_with("public","cache","asset","a"*64)
            runner.metadata.return_value=dict(runner_sha256=self.h.RUNNER_SHA,conditions=self.conditions[:3])
            with self.assertRaises(ValueError):self.h.metadata(runner,"p","c","a","a"*64)
            runner.metadata.return_value=dict(runner_sha256="changed",conditions=self.conditions)
            with self.assertRaises(ValueError):self.h.metadata(runner,"p","c","a","a"*64)

    def test_default_is_stdlib_metadata_no_outputs_or_science(self):
        before={name:name in sys.modules for name in ("numpy","torch","psutil")}
        with mock.patch.object(self.h,"metadata",side_effect=AssertionError("no input roots")),contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(self.h.main([]),0)
        result=json.loads(stream.getvalue())
        self.assertEqual(result["registered_conditions"],4)
        self.assertEqual(result["files_written"],0)
        self.assertEqual(before,{name:name in sys.modules for name in before})


if __name__=="__main__":unittest.main(verbosity=2)
