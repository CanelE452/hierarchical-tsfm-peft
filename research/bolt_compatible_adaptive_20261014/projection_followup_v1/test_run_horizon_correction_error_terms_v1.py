"""Synthetic saved-output contracts; no actual scientific metadata or payload."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import horizon_correction_error_terms_v1 as terms
import run_horizon_correction_error_terms_v1 as run


def independent_score(p,y,mask):
    counts=mask.sum((0,1)).tolist();per=[]
    for j,count in enumerate(counts):
        errors=[(float(p[i,h,j])-float(y[i,h,j]))**2 for i,h in zip(*np.where(mask[:,:,j]))];per.append(sum(errors)/count if count else 0.)
    active=[j for j,count in enumerate(counts) if count]
    return {"mse":sum(per[j] for j in active)/len(active),"per_channel_mse":per,"observed_counts":counts,"active_channels":len(active)} if active else {"status":"NO_OBSERVED_TARGETS"}


def temporal(p,y,mask):
    return {"per_horizon":[independent_score(p[:,h:h+1],y[:,h:h+1],mask[:,h:h+1]) for h in range(48)],"registered_intervals":{key:independent_score(p[:,s:t],y[:,s:t],mask[:,s:t]) for key,s,t in terms.INTERVALS}}


class ErrorTermsControllerTests(unittest.TestCase):
    def test_source_import_and_closed_gate_import_neither_numpy_nor_torch(self):
        script="import sys; from types import SimpleNamespace; import run_horizon_correction_error_terms_v1 as r; r.HORIZON_AUDIT_SHA=None\ntry:r.load_gates(SimpleNamespace(horizon_result='missing',horizon_audit='missing'))\nexcept ValueError:pass\nassert 'numpy' not in sys.modules and 'torch' not in sys.modules"
        completed=subprocess.run([sys.executable,"-c",script],cwd=run.HERE,capture_output=True,text=True);self.assertEqual(completed.returncode,0,completed.stderr)

    def test_actual_gate_is_closed_before_file_io_or_array_import(self):
        with patch.object(run,"HORIZON_AUDIT_SHA",None),patch.object(run.old,"record",side_effect=AssertionError("IO before gate")),self.assertRaisesRegex(ValueError,"GATES_NOT_REGISTERED"):run.load_gates(SimpleNamespace(horizon_result="missing",horizon_audit="missing"))

    def test_score_link_checks_counts_and_ignores_source_zero_for_absent_channel(self):
        b=np.zeros((1,2,2),np.float32);p=b+1;y=b.copy();mask=np.ones_like(b,bool);mask[:,:,1]=False;cell=terms.decompose(b,p,y,mask);source=independent_score(p,y,mask)
        self.assertEqual(run.score_link(cell,source,"selected_MSE")["max_per_channel_MSE_abs_discrepancy"],0.)
        changed=copy.deepcopy(source);changed["observed_counts"][1]=1
        with self.assertRaisesRegex(ValueError,"SOURCE_SCORE_COUNTS"):run.score_link(cell,changed,"selected_MSE")
        changed=copy.deepcopy(source);changed["mse"]+=.01
        with self.assertRaisesRegex(ValueError,"SOURCE_SCORE_MSE"):run.score_link(cell,changed,"selected_MSE")

    def test_registered_metadata_gate_and_audit_parent_binding(self):
        with tempfile.TemporaryDirectory(dir=run.HERE,prefix="synthetic_error_gate_") as directory:
            folder=Path(directory)
            def save(name,value):
                path=folder/name;path.write_text(json.dumps(value),encoding="utf-8");return run.old.record(path)
            component=save("source.json",{});result={"status":"COMPLETE","failed_paths":[],"controller_source":component,"fits_completed":24,"candidate_FIT_TUNE_evaluations_completed":528,"selected_DEV_evaluations_completed":24,"resolved_selection_locks":24,"new_F_forward_calls":0,"all_24_locks_before_DEV":True,"protected_TEST_accessed":False}
            for key in ("module_source","plan","selection_registry","cross_moment_diagnostic","cross_moment_audit","causal_lag_diagnostic"):result[key]=component
            binding=save("result.json",result);audit={"status":"PASS","diagnostic":binding,"partial_paths":[],"audit_source":component,"protected_TEST_accessed":False,"totals":{"decoder_fit_paths_checked":24,"selected_FP32_arithmetic_predictions_checked":72},**{key:result[key] for key in ("controller_source","module_source","plan","selection_registry","cross_moment_diagnostic","cross_moment_audit","causal_lag_diagnostic")}};accepted=save("audit.json",audit);args=SimpleNamespace(horizon_result=binding["path"],horizon_audit=accepted["path"])
            with patch.object(run,"HORIZON_RESULT_SHA",binding["sha256"]),patch.object(run,"HORIZON_AUDIT_SHA",accepted["sha256"]),patch.object(run,"HORIZON_CONTROLLER_SHA",component["sha256"]),patch.object(run,"HORIZON_AUDITOR_SHA",component["sha256"]):
                actual,checked,refs=run.load_gates(args);self.assertEqual(actual,result);self.assertEqual(checked,audit);self.assertEqual(refs["horizon_diagnostic"],binding)
                audit["diagnostic"]={**binding,"sha256":"0"*64};bad=save("audit.json",audit)
                with patch.object(run,"HORIZON_AUDIT_SHA",bad["sha256"]),self.assertRaisesRegex(ValueError,"EXACT_PATH_SHA_BINDING"):run.load_gates(args)

    def test_both_parent_CPU_receipts_require_canonical_fields_and_exact_recorded_contents(self):
        with tempfile.TemporaryDirectory(dir=run.HERE,prefix="synthetic_error_receipt_") as directory:
            folder=Path(directory)
            def save(name,value):
                path=folder/name;path.write_text(json.dumps(value),encoding="utf-8");return run.old.record(path)
            authorization=save("authorization.json",{"status":"APPROVED_HUMAN_UNCAPPED_CONTINUATION"});manifest=save("manifest.json",{"authorization_sha256":authorization["sha256"],"runtime_sha256":run.prior.file_sha(run.HERE/"continuation_runtime_v1.py")})
            for status in ("COMPLETE","PASS"):
                with self.subTest(status=status):
                    value={"status":status,"gpu_process":False,"process_cpu_seconds":.5,"wall_seconds":.75,"authorization":authorization,"continuation_manifest":manifest};binding=save(status+".json",value);receipt={**value,**binding};self.assertEqual(len(run.receipt_records(receipt,status,lambda:None)),3)
                    bad={**receipt,"wall_seconds":.8}
                    with self.assertRaisesRegex(ValueError,"EXACT_PARENT_RECEIPT_FILE"):run.receipt_records(bad,status,lambda:None)
                    bad={key:value for key,value in receipt.items() if key!="process_cpu_seconds"};bad["cpu_seconds"]=.5
                    with self.assertRaisesRegex(ValueError,"CANONICAL_PARENT_CPU_RECEIPT"):run.receipt_records(bad,status,lambda:None)

    def fixture(self):
        rng=np.random.default_rng(152);payloads={};documents={};datasets={"robin":(3,1,"val",2),"beijing_aotizhongxin":(3,1,"eval",2)};result={"datasets":{},"selection_registry":None};registry=[]
        def put(name,value,store):
            record={"path":str((run.HERE/("synthetic_error_terms_"+name)).resolve()),"sha256":("a" if name.endswith(".npz") else "b")*64};store[record["path"]]=value
            if name.endswith(".npz"):record["arrays"]={key:run.array_record(array) for key,array in value.items()}
            return record
        for data,(c,k,dev,dn) in datasets.items():
            owner=result["datasets"][data]={"spaces":{}}
            raws={}
            for split,n in (("fit",3),("tune",2),(dev,dn)):
                target=rng.normal(size=(n,48,c)).astype(np.float32);mask=rng.random(target.shape)>.2;mask[:,:,2]=False;target[~mask]=np.nan;last=rng.normal(size=(n,c)).astype(np.float32)
                raw={"target":target,"observed":mask,"raw_last":last,"raw_forecast":rng.normal(size=target.shape).astype(np.float32),"std":np.ones(c,np.float32),"mean":np.zeros(c,np.float32),"origins":np.arange(n,dtype=np.int64)+100};raws[split]=(raw,put(data+split+"_raw.npz",raw,payloads))
            for space in ("RAW_C","PCA_K1","PCA_K2"):
                space_row=owner["spaces"][space]={"contexts":{},"conditions":{}};streams={}
                for split,(raw,source) in raws.items():
                    cache={"origins":raw["origins"],"prediction":(raw["raw_forecast"]*.7).astype(np.float32),"FLAT_prediction":np.broadcast_to(raw["raw_last"][:,None],raw["target"].shape).copy()};cached=None if space=="RAW_C" else put(data+space+split+"_cache.npz",cache,payloads);metadata={"input_source":source,"cached_latent_source":cached,"frozen_state_source":None if space=="RAW_C" else {"path":data+space+"_state.npz","sha256":"c"*64},"K":c if space=="RAW_C" else int(space[-1]),"array_bindings":{},"baseline_scores":{},"baseline_temporal_diagnostics":{}}
                    for mode in ("F","FLAT"):
                        b=(raw["raw_forecast"] if mode=="F" else np.broadcast_to(raw["raw_last"][:,None],raw["target"].shape).copy()) if space=="RAW_C" else cache["prediction" if mode=="F" else "FLAT_prediction"];streams[split,mode]=b;metadata["array_bindings"][mode]={"baseline":run.array_record(b)};metadata["baseline_scores"][mode]={"scores_all_observed":{"normalized":independent_score(b,raw["target"],raw["observed"])}};metadata["baseline_temporal_diagnostics"][mode]=temporal(b,raw["target"],raw["observed"])
                    space_row["contexts"][split]=metadata
                for mode in ("F","FLAT"):
                    for arm in ("STATIC_DELTA","RAMP_DELTA"):
                        name=mode+"_"+arm;zero=space=="RAW_C" and mode=="FLAT" and arm=="STATIC_DELTA";row={"status":"COMPLETE","space":space,"mode":mode,"arm":arm,"K":space_row["contexts"]["fit"]["K"],"selected_candidate":"ZERO_DELTA" if zero else "ALPHA_10","selected_kind":"ZERO_DELTA" if zero else "ALPHA","selected_alpha":None if zero else 10.,"selected_coefficient_value_sha256":"d"*64,"selected_coefficients":{"path":data+space+name+"_coef.npz","sha256":"e"*64},"FIT_sources":space_row["contexts"]["fit"],"TUNE_sources":space_row["contexts"]["tune"],"selected_temporal_diagnostics":{}}
                        saved={}
                        for label,split in (("FIT","fit"),("TUNE","tune"),("DEV",dev)):
                            raw,_=raws[split];b=streams[split,mode];p=b.copy() if zero else (b+np.float32(.02)*(np.ones(48,np.float32) if arm=="STATIC_DELTA" else np.arange(48,dtype=np.float32)/47)[None,:,None]).astype(np.float32);scores={"status":"COMPLETE","scores_all_observed":{"normalized":independent_score(p,raw["target"],raw["observed"])}};row[label]=scores
                            if label!="DEV":saved[label+"_prediction_FP32"]=p;saved[label+"_origins"]=raw["origins"];row["selected_temporal_diagnostics"][label]=temporal(p,raw["target"],raw["observed"])
                            else:scores.update(artifact=put(data+space+name+"_DEV.npz",{"prediction_FP32":p,"origins":raw["origins"]},payloads),sources=space_row["contexts"][split],selected_coefficients=row["selected_coefficients"],temporal_diagnostics=temporal(p,raw["target"],raw["observed"]))
                        row["selected_TUNE_MSE"]=row["TUNE"]["scores_all_observed"]["normalized"]["mse"];row["selected_FIT_TUNE_prediction_artifact"]=put(data+space+name+"_FT.npz",saved,payloads)
                        lock={key:row[key] for key in ("selected_candidate","selected_kind","selected_alpha","selected_TUNE_MSE","selected_coefficients","selected_coefficient_value_sha256")};lock.update(status="COMPLETE",dataset=data,space=space,mode=mode,arm=arm,DEV_opened_before_lock=False,sealed_unix=1.);row["selection_lock"]=put(data+space+name+"_lock.json",lock,documents);space_row["conditions"][name]=row;registry.append({"dataset":data,"space":space,"mode":mode,"arm":arm,"status":"COMPLETE","selection_lock":row["selection_lock"]})
        result["selection_registry"]=put("registry.json",{"status":"ALL_24_RESOLVED_BEFORE_DEV","conditions":registry,"DEV_opened_before_seal":False,"sealed_unix":2.},documents)
        return result,payloads,documents,datasets

    def test_all_24_conditions_72_full_3456_horizon_216_interval_cells(self):
        result,payloads,documents,datasets=self.fixture()
        with patch.object(run,"DATASETS",datasets),patch.object(run,"SPLIT_N",{"fit":3,"tune":2}):report=run.analyze(result,lambda record:payloads[record["path"]],lambda record:documents[record["path"]])
        for key,value in (("conditions_checked",24),("full_cells_completed",72),("horizon_cells_completed",3456),("interval_cells_completed",216),("source_contexts_checked",18),("selection_locks_preserved",24),("ZERO_split_cells_checked",6)):self.assertEqual(report["totals"][key],value)
        self.assertEqual(report["new_fits"],0);self.assertEqual(report["new_F_forward_calls"],0);self.assertEqual(report["new_selection_decisions"],0);json.dumps(report,allow_nan=False)
        first=report["datasets"]["robin"]["spaces"]["RAW_C"]["conditions"]["F_STATIC_DELTA"];self.assertIn("DEV_minus_TUNE_interaction_term",first["TUNE_DEV_observation"]);self.assertEqual(first["FIT"]["full"]["per_channel"][2]["status"],"NO_OBSERVED_TARGETS")

    def test_origin_baseline_binding_lock_and_source_score_tampering_fail(self):
        for kind in ("origins","baseline","lock","score"):
            with self.subTest(kind=kind):
                result,payloads,documents,datasets=self.fixture();row=result["datasets"]["robin"]["spaces"]["RAW_C"]["conditions"]["F_STATIC_DELTA"]
                if kind=="origins":payloads[row["selected_FIT_TUNE_prediction_artifact"]["path"]]["FIT_origins"]=np.array([-1,-2,-3],np.int64)
                elif kind=="baseline":result["datasets"]["robin"]["spaces"]["RAW_C"]["contexts"]["fit"]["array_bindings"]["F"]["baseline"]["sha256"]="0"*64
                elif kind=="lock":documents[row["selection_lock"]["path"]]["selected_alpha"]=100.
                else:row["FIT"]["scores_all_observed"]["normalized"]["mse"]+=.01
                with patch.object(run,"DATASETS",datasets),patch.object(run,"SPLIT_N",{"fit":3,"tune":2}),self.assertRaises(ValueError):run.analyze(result,lambda record:payloads[record["path"]],lambda record:documents[record["path"]])


if __name__=="__main__":unittest.main()
