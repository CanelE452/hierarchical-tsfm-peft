"""CPU-only portable route, true toy forward, target seal and replay checks."""
import csv
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import data_contract as data
import deployment
import fit_base
from training import fingerprint

HERE = Path(__file__).resolve().parent


class RoutePredictionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((HERE/"route_catalog.py").is_file(), "The portable route catalog is missing")
        self.assertTrue((HERE/"route_predictions.py").is_file(), "The physical-input route prediction driver is missing")
        self.catalog = importlib.import_module("route_catalog")
        self.predictions = importlib.import_module("route_predictions")
        self.temp = tempfile.TemporaryDirectory(prefix="informs_route_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache, self.public = self.root/"cache", self.root/"public"
        self.public.mkdir()
        rng = np.random.default_rng(14)
        values = rng.normal(size=(240, 3)).astype(np.float32)
        self.task = dict(name="toy", dataset="toy", physical=values, observed=np.ones_like(values, bool),
                         L=64, H=4, period=12, columns=["a", "b", "c"], data_sha256="toy", split_ranges=data.split_ranges(240))
        self.rows = []

    def register(self, run, family, block="E1", k=0):
        self.rows.append(dict(experiment_id=run, dataset="toy", fold="primary", family=family, block=block,
                              latent_k=str(k), seed="92601", execution_status="COMPLETE_VALID"))
        path = self.public/"EXPERIMENT_REGISTRY.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)

    def base_G(self,innovation_bias=0.):
        run = "primary_toy_E1_DIRECT_G_k0_s92601"
        if any(row["experiment_id"]==run for row in self.rows):return run
        self.register(run, "DIRECT_G")
        spec = dict(run_id=run, family="DIRECT_G", seed=92601, device="cpu", fold="primary")
        model, norm = fit_base.create_model(spec, self.task, 144, None)
        with torch.no_grad():model.output.bias.fill_(innovation_bias)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        bundle = dict(spec=spec, normalization=norm, model=state,
                      contract=dict(data_sha="toy", model_sha="NO_FOUNDATION", source_sha="toy-source",
                                    scaler_sha=fit_base.scaler_fingerprint(norm["mean"],norm["std"])),
                      model_state_sha=fingerprint({key:data.array_sha256(value.numpy()) for key,value in state.items()}))
        relative = f"runs/{run}/selected_bundle.pt"
        path = self.cache/relative;path.parent.mkdir(parents=True);torch.save(bundle,path)
        receipt = dict(run_id=run, family="DIRECT_G", status="COMPLETE_VALID", artifact_cache_relative_path=relative,
                       artifact_sha256=data.sha256_file(path))
        folder=self.public/"runs"/run;folder.mkdir(parents=True)
        (folder/"BASE_FIT_RECEIPT.json").write_text(json.dumps(receipt))
        return run

    def head_fixture(self,family,variant):
        from combination import ConditionalHead,LegacyResidualHead,scaler_fingerprint
        gid=self.base_G();bid="primary_toy_E1_RAW_BOLT_SMALL_k0_s92601"
        if not any(row["experiment_id"]==bid for row in self.rows):self.register(bid,"RAW_BOLT_SMALL")
        run="primary_toy_E2_"+family+"_k0_s92601_RAW";self.register(run,family,"E2")
        spec=dict(run_id=run,variant=variant,g_run_id=gid,parent_run_id=bid)
        folder=self.cache/"combinations"/run;folder.mkdir(parents=True)
        (folder/"spec.json").write_text(json.dumps(spec))
        mean,std,_=data.fit_statistics(self.task,144)
        contract=dict(data_sha="toy",scaler_sha=scaler_fingerprint(mean,std),
                      config_sha=fingerprint({key:value for key,value in spec.items() if key!="run_id"}))
        head=(LegacyResidualHead if variant=="legacy_residual" else ConditionalHead)(64,4)
        path=folder/"selected.pt"
        torch.save(dict(model=head.state_dict(),head_config=dict(length=64,horizon=4,hidden=32,kind=variant),
                        contract=contract,control=variant=="past_control"),path)
        public=self.public/"combinations"/run;public.mkdir(parents=True)
        (public/"receipt.json").write_text(json.dumps(dict(run_id=run,variant=variant,execution_status="COMPLETE_VALID",
            contract=contract,selected_checkpoint_sha256=data.sha256_file(path),parent_models={})))
        return run

    def alpha_fixture(self, label):
        gid = self.base_G()
        bid = "primary_toy_E1_RAW_BOLT_SMALL_k0_s92601"
        self.register(bid,"RAW_BOLT_SMALL")
        group = "primary_toy_E2_ALPHA_FIT_k0_s92601_RAW_BOLT_SMALL"
        run = group.replace("_ALPHA_FIT_","_"+label+"_")
        self.register(run,label,"E2")
        private=self.cache/"combinations"/group;private.mkdir(parents=True)
        spec=dict(run_id=group,variant="alpha_family",g_run_id=gid,parent_run_id=bid)
        (private/"spec.json").write_text(json.dumps(spec))
        folder=self.public/"combinations"/group;folder.mkdir(parents=True)
        (folder/"receipt.json").write_text(json.dumps(dict(run_id=group,variant="alpha_family",execution_status="COMPLETE_VALID",
            contract=dict(data_sha="toy",config_sha=fingerprint({key:value for key,value in spec.items() if key!="run_id"})),alpha_fit=dict(alpha_fit=.3),parent_models={})))
        return run,gid,bid

    def test_alpha0_is_actual_G_only_and_alpha1_is_actual_b_only(self):
        run,gid,bid=self.alpha_fixture("ALPHA0")
        route,size=self.catalog.build_route(run,self.task,self.cache,self.public)
        self.assertEqual(route["family"],"DIRECT_G");self.assertIsNone(size)
        self.assertNotIn("b_route",route)
        result=self.predictions.predict_route(run,self.task,"V_select",self.cache,self.public,device="cpu")
        self.assertEqual(result["cost_record"]["actual_F_streams"],0)
        self.assertEqual(result["manifest"]["metadata"]["route_sha"],fingerprint(route))
        from test_costs import ToyFoundation
        alpha1=run.replace("_ALPHA0_","_ALPHA1_");self.register(alpha1,"ALPHA1","E2")
        second,size=self.catalog.build_route(alpha1,self.task,self.cache,self.public)
        self.assertEqual(second["family"],"RAW");self.assertEqual(size,"small")
        self.assertNotIn("g_route",second)
        pool=deployment.FoundationPool(None,"cpu",factory=lambda p,h,r,d:ToyFoundation(h))
        actual=self.predictions.predict_route(alpha1,self.task,"V_select",self.cache,self.public,device="cpu",foundation_pool=pool)
        self.assertEqual(actual["cost_record"]["actual_F_streams"],6*3)
        self.assertEqual(pool.instances_created,1)
        pool.close()

    def test_actual_artifact_tamper_rejected_before_prediction(self):
        run=self.base_G()
        route,_=self.catalog.build_route(run,self.task,self.cache,self.public)
        with (self.cache/route["artifact_cache_relative_path"]).open("ab") as handle:handle.write(b"tamper")
        with self.assertRaises(ValueError):self.catalog.build_route(run,self.task,self.cache,self.public)

    def test_existing_forecast_cannot_gain_a_retrospective_truth_seal(self):
        run="primary_toy_E1_CURRENT_k0_s92601";self.register(run,"CURRENT")
        result=self.predictions.predict_route(run,self.task,"V_select",self.cache,self.public,device="cpu")
        path=self.public/"route_predictions"/run/"V_select/REFERENCE_CONTRACT.json"
        path.unlink()
        with self.assertRaisesRegex(ValueError,"retrospectively"):
            self.predictions.predict_route(run,self.task,"V_select",self.cache,self.public,device="cpu")
        archive=self.cache/result["manifest"]["cache_relative_path"]
        self.assertEqual(data.sha256_file(archive),result["manifest"]["archive_sha256"])

    def test_T_gate_precedes_any_target_materialization(self):
        run="primary_toy_E1_CURRENT_k0_s92601";self.register(run,"CURRENT")
        with patch.object(data,"target_window",side_effect=AssertionError("T target was read")):
            with self.assertRaisesRegex(ValueError,"[Ss]election|locked|lock"):
                self.predictions.predict_route(run,self.task,"T_final",self.cache,self.public,device="cpu")
        route,_=self.catalog.build_route(run,self.task,self.cache,self.public)
        (self.public/"SELECTION_LOCK.json").write_text(json.dumps(dict(locked=True,allowed_route_sha={run:fingerprint(route)})))
        result=self.predictions.predict_route(run,self.task,"T_final",self.cache,self.public,device="cpu")
        self.assertEqual(result["score"]["n_origins"],12)
        self.assertEqual(result["cost_record"]["actual_F_streams"],0)
        self.assertTrue(result["reference_contract"]["created_before_inference"])

    def test_fixed_V_replay_uses_existing_standard_cache_and_no_training(self):
        import prediction_cache
        run="primary_toy_E1_CURRENT_k0_s92601";self.register(run,"CURRENT")
        prediction_cache.predict_parent(run,self.task,"V_select",self.cache,self.public,None,baseline="CURRENT")
        result=self.predictions.replay_route(run,self.task,self.cache,self.public,device="cpu")
        self.assertEqual(result["status"],"PASS")
        self.assertLessEqual(len(result["origins"]),6)
        self.assertEqual(result["actual_F_streams"],0)
        self.assertFalse(result["independent_training_rerun"])

    def test_large_physical_offset_does_not_destroy_normalized_checkpoint_parity(self):
        import prediction_cache
        self.task["physical"]=np.repeat((1e6+np.arange(240,dtype=np.float32)*.125)[:,None],3,axis=1)
        run=self.base_G(innovation_bias=.5)
        prediction_cache.predict_parent(run,self.task,"V_select",self.cache,self.public,None)
        result=self.predictions.replay_route(run,self.task,self.cache,self.public,device="cpu")
        self.assertEqual(result["status"],"PASS",f"Normalized parity lost to physical-output rounding: {result['max_abs']}")

    def test_conditional_past_and_legacy_execute_only_their_actual_parent_routes(self):
        from test_costs import ToyFoundation
        for family,variant,expected in (("CONDITIONAL","conditional",18),("PAST_CONTROL","past_control",0),
                                        ("LEGACY_RESIDUAL","legacy_residual",18)):
            with self.subTest(variant=variant):
                run=self.head_fixture(family,variant)
                spec,size=self.catalog.build_route(run,self.task,self.cache,self.public)
                self.assertEqual("b_route" in spec,variant!="past_control")
                self.assertEqual("g_route" in spec,variant!="legacy_residual")
                pool=deployment.FoundationPool(None,"cpu",factory=lambda p,h,r,d:ToyFoundation(h))
                result=self.predictions.predict_route(run,self.task,"V_select",self.cache,self.public,device="cpu",foundation_pool=pool)
                self.assertEqual(result["cost_record"]["actual_F_streams"],expected)
                self.assertEqual(pool.instances_created,0 if variant=="past_control" else 1)
                pool.close()

    def test_nested_PCA_uses_requested_K_and_exact_foundation_identity(self):
        from test_costs import ToyFoundation
        run="primary_toy_E4_NESTED_PCA_k1_s92601";self.register(run,"NESTED_PCA","E4",1)
        mean,std,_=data.fit_statistics(self.task,144)
        folder=self.cache/"ordered/primary_toy";folder.mkdir(parents=True)
        path=folder/"nested_pca.npz"
        np.savez(path,basis=np.eye(3,dtype=np.float32),center=np.zeros(3,np.float32),mean=mean,std=std,
                 fit_end=np.asarray(144),data_sha=np.asarray("toy"))
        pool=deployment.FoundationPool(None,"cpu",factory=lambda p,h,r,d:ToyFoundation(h))
        expected_sha=pool.get(4).initial_hash
        public=self.public/"ordered_predictions"/run;public.mkdir(parents=True)
        (public/"PREDICTION_MANIFEST.json").write_text(json.dumps(dict(run_id=run,data_sha="toy",metadata=dict(
            inference_latent_k=1,variant="nested_pca",parent_artifact_sha256=data.sha256_file(path),foundation_sha=expected_sha))))
        spec,size=self.catalog.build_route(run,self.task,self.cache,self.public)
        self.assertEqual(spec["latent_k"],1);self.assertEqual(size,"small")
        result=self.predictions.predict_route(run,self.task,"V_select",self.cache,self.public,device="cpu",foundation_pool=pool)
        self.assertEqual(result["cost_record"]["actual_F_streams"],6)
        pool.close()

    def test_raw_bias_is_applied_to_actual_Raw_output(self):
        from test_costs import ToyFoundation
        from combination import scaler_fingerprint
        bid="primary_toy_E1_RAW_BOLT_SMALL_k0_s92601";self.register(bid,"RAW_BOLT_SMALL")
        run="primary_toy_E2_RAW_CALIBRATED_k0_s92601_RAW";self.register(run,"RAW_CALIBRATED","E2")
        mean,std,_=data.fit_statistics(self.task,144)
        private=self.cache/"raw_calibration"/run;private.mkdir(parents=True)
        path=private/"bias.npz";contract=dict(data_sha="toy",scaler_sha=scaler_fingerprint(mean,std))
        np.savez(path,bias=np.full(3,.5),mean=mean,std=std,contract_json=np.asarray(json.dumps(contract)))
        folder=self.public/"raw_calibration"/run;folder.mkdir(parents=True)
        (folder/"receipt.json").write_text(json.dumps(dict(status="COMPLETE_VALID",contract=contract,parent_run_id=bid,
            artifact_cache_relative_path=path.relative_to(self.cache).as_posix(),artifact_sha256=data.sha256_file(path))))
        pool=deployment.FoundationPool(None,"cpu",factory=lambda p,h,r,d:ToyFoundation(h))
        result=self.predictions.predict_route(run,self.task,"V_select",self.cache,self.public,device="cpu",foundation_pool=pool)
        self.assertEqual(result["cost_record"]["actual_F_streams"],18)
        origin=int(result["origins"][0]);x,_=data.history_window(self.task,origin,mean,std)
        np.testing.assert_allclose(result["pred_normalized"][0],np.repeat(x[-1:]+.5,4,axis=0),atol=1e-6)
        pool.close()

    def test_posthoc_full_C_checkpoint_executes_only_requested_untrained_prefix(self):
        from test_costs import ToyFoundation
        parent="primary_toy_E4_MAXK_PCA_TRAINED_k3_s92601";self.register(parent,"MAXK_PCA_TRAINED","E4",3)
        pool=deployment.FoundationPool(None,"cpu",factory=lambda p,h,r,d:ToyFoundation(h))
        foundation=pool.get(4)
        spec=dict(run_id=parent,family="B0",seed=92601,device="cpu",fold="primary",latent_k=3,k_grid=[3])
        model,norm=fit_base.create_model(spec,self.task,144,foundation)
        state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        bundle=dict(spec=spec,normalization=norm,model=state,
            contract=dict(data_sha="toy",model_sha=foundation.initial_hash,source_sha="toy-source",
                          scaler_sha=fit_base.scaler_fingerprint(norm["mean"],norm["std"])),
            model_state_sha=fingerprint({key:data.array_sha256(value.numpy()) for key,value in state.items()}))
        path=self.cache/"runs"/parent/"selected_bundle.pt";path.parent.mkdir(parents=True);torch.save(bundle,path)
        folder=self.public/"runs"/parent;folder.mkdir(parents=True)
        (folder/"BASE_FIT_RECEIPT.json").write_text(json.dumps(dict(run_id=parent,family="B0",status="COMPLETE_VALID",
            artifact_cache_relative_path=path.relative_to(self.cache).as_posix(),artifact_sha256=data.sha256_file(path))))
        run="primary_toy_E4_POSTHOC_PREFIX_k1_s92601";self.register(run,"POSTHOC_PREFIX","E4",1)
        folder=self.public/"ordered_predictions"/run;folder.mkdir(parents=True)
        (folder/"PREDICTION_MANIFEST.json").write_text(json.dumps(dict(run_id=run,data_sha="toy",metadata=dict(
            inference_latent_k=1,variant="posthoc",parent_run_id=parent,parent_artifact_sha256=data.sha256_file(path)))))
        route,size=self.catalog.build_route(run,self.task,self.cache,self.public)
        self.assertEqual(route["prefix_variant"],"posthoc");self.assertEqual(route["latent_k"],1)
        result=self.predictions.predict_route(run,self.task,"V_select",self.cache,self.public,device="cpu",foundation_pool=pool)
        self.assertEqual(result["cost_record"]["actual_F_streams"],6)
        pool.close()


if __name__=="__main__":
    torch.set_num_threads(2)
    unittest.main()
