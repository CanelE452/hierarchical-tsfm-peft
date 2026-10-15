"""Synthetic metadata and full figure rendering; no experiment arrays or model execution."""
from pathlib import Path
from unittest.mock import patch
import copy
import statistics
import tempfile
import unittest

import make_fixed_l7_latent_calibration_report_v1 as report
from test_audit_f_input_gradient_report_v1 import score


def fixture(folder):
    binding={"path":str(folder/"unopened.npz"),"sha256":"a"*64}
    def entry(value,identity=False):
        return {"status":"COMPLETE","artifact":binding,"scores_all_observed":score(value),"scores_complete_mask":score(value+.1,(70,70,70)),"surrogate_diagnostics":{"max_abs_actual_FP32_minus_linear_surrogate":0. if identity else 2e-6,"RMS_actual_FP32_minus_linear_surrogate":0. if identity else 2e-7,"per_channel_max_abs":[0. if identity else 2e-6]*3,"surrogate_zero_error_required":False},"independent_FP32_deployment_parity":{"latent_future":{"max_abs":1e-7,"pass":True},"C_prediction":{"max_abs":3e-7,"pass":True}},"FP64_surrogate_is_not_deployment_parity_target":True,"identity_checks":{"identity_object_checks":2,"identity_byte_exact":identity,"batch_size":32},"coefficient_value_sha256":"b"*64,"input_source":binding,"cached_Bolt_source":binding}
    values={"train":{"states":{},"matched_l7_native_audit":binding,"failed_paths":[],"failed_candidate_paths":[],"fit_conditions_completed":6,"SVD_fits_completed":6,"candidate_FIT_TUNE_evaluations_completed":96,"selected_DEV_evaluations_completed":6,"resource_terminal":{"wall_seconds":2.}},"audit":{"states":{},"partial_paths":[],"resource_terminal":{"wall_seconds":1.}},"native":{"states":{}},"native_audit":{"new_output_space_diagnostics":{}},"cost":{"cost":{"robin":{"rows":[]}},"failed_paths":[],"resource_terminal":{"wall_seconds":3.}},"cost_audit":{"resource_terminal":{"wall_seconds":.5}},"science":{"datasets":{"robin":{"spaces":{"PCA":{"conditions":{"L7_RECON":{"splits":{s:{"RAW_BOLT":entry(.25)} for s in ("fit","tune","val")}}}}}}}}}
    for si,state in enumerate(report.STATES):
        baseline={"fit":.2+si*.01,"tune":.1+si*.001,"val":.5+si*.01}
        values["native"]["states"][state]={"Bolt":{split:entry(value) for split,value in baseline.items()}}
        owner=values["train"]["states"][state]={"E_previously_supervised_by_FIT_future":si>0,"families":{}}
        values["audit"]["states"][state]={}; values["native_audit"]["new_output_space_diagnostics"][state]={"92601":{}}
        for split in ("tune","val"):
            values["native_audit"]["new_output_space_diagnostics"][state]["92601"][split]={"minimum_normalized_MSE":.06 if split=="tune" else .3,"RAW_BOLT_scores_complete_mask":score(.05 if split=="tune" else .4),"complete_target_positions":70,"target_positions":100}
        for family in report.FAMILIES:
            special=si==2 and family=="SCALE_ONLY"; candidates=[]; vc=[]
            for ci,lam in enumerate((None,*report.LAMBDAS)):
                identity=ci==0; fit_value=baseline["fit"]-(0. if identity else .001+ci*.0001); tune_value=baseline["tune"]+(0. if identity else .002+1e-5*(8-ci))
                if special and ci==7: tune_value=baseline["tune"]-1e-5
                candidate={"id":"IDENTITY" if identity else "RIDGE_"+format(lam,"g"),"kind":"IDENTITY" if identity else "RIDGE","family":family,"lambda":lam,"status":"COMPLETE","metadata":{"state":{"path":"nested label collision sentinel"},"negative_gain_clipped":False},"coefficient_artifact":binding,"coefficient_value_sha256":"b"*64,"FIT":entry(fit_value,identity),"TUNE":entry(tune_value,identity),"TUNE_MSE":tune_value}
                candidates.append(candidate); zeros=[0.]*4; a=zeros if identity else [-.001,.002,-.003,-.006]
                vc.append({**candidate,"a_FP64":a,"b_FP64":zeros,"a_FP32":a,"b_FP32":zeros,"coefficients_FP64":a+(zeros if family=="SCALE_BIAS" else []),"coefficients_FP32":a+(zeros if family=="SCALE_BIAS" else []),"nonfinite_coefficient_indices":[]})
            winner=candidates[-1] if special else candidates[0]
            owner["families"][family]={"status":"COMPLETE","candidates":candidates,"selected_candidate":winner["id"],"selected_kind":winner["kind"],"selected_lambda":winner["lambda"],"selected_TUNE_MSE":winner["TUNE_MSE"],"selected_coefficients":binding,"selected_coefficient_value_sha256":"b"*64,"FIT":winner["FIT"],"TUNE":winner["TUNE"],"DEV":entry(baseline["val"]+(2e-4 if special else 0.),winner["kind"]=="IDENTITY"),"fit_metadata":{"SVD_fits":1,"state":binding},"fit_artifact":binding,"FIT_input_source":binding,"FIT_cached_Bolt_source":binding,"frozen_state_source":binding,"selection_lock":binding,"selection_rule":"actual TUNE; identity exact tie then larger lambda"}
            values["audit"]["states"][state][family]={"candidates":vc,"FIT_solver_rank":4 if family=="SCALE_ONLY" else 8,"independently_recomputed_fit":{"column_scales_FP64":[1.]*4,"active_columns":[True]*4,"singularvalues_FP64":[1.]*4,"zero_lambda_numeric_rank":4,"FIT_channel_observed_counts":[100,90,80],"row_weight_sum":1.}}
    rows=values["cost"]["cost"]["robin"]["rows"]
    for batch in (1,8):
        for route in report.ROUTES:
            duration=(5,1,3,4,2) if route=="RAW_BOLT" else (4,2,6,3,1) if route.endswith("_BOLT") else (2,2,3,4,1)
            sec=[v/1000 for v in duration]; matched="RAW_BOLT" if route=="RAW_BOLT" else route if route.endswith("_BOLT") else next(state+"_BOLT" for state in report.STATES if route.startswith(state+"_"))
            samples=[{"route":route,"batch":batch,"repeat":i,"seconds":s,"perf_start_ns":10**18+i*10**8,"perf_end_ns":10**18+i*10**8+v*10**6} for i,(s,v) in enumerate(zip(sec,duration))]
            rows.append({"route":route,"batch":batch,"status":"COMPLETE","seconds":sec,"samples":samples,"median_seconds":statistics.median(sec),"median_seconds_per_window":statistics.median(sec)/batch,"matched_Bolt_route":matched})
    for row in rows:
        for field,route in (("RAW","RAW_BOLT"),("matched_Bolt",row["matched_Bolt_route"])):
            denominator=report.choose(rows,route=route,batch=row["batch"]); row["paired_"+field+"_median_ratio"]=statistics.median(a/b for a,b in zip(row["seconds"],denominator["seconds"])); row["ratio_of_"+field+"_medians"]=statistics.median(row["seconds"])/statistics.median(denominator["seconds"])
    return values


class FixedLatentCalibrationReportTests(unittest.TestCase):
    def test_all_candidates_masks_scales_coefficients_and_state_labels_survive_nested_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            values=fixture(Path(folder)); tables=report.collect_tables(values)
            self.assertEqual([len(tables[k]) for k in report.TABLES],[384,72,48,72,48,48,6,102,6,6,0,20,100,4])
            for state in report.STATES:
                for family in report.FAMILIES:
                    candidates=[p for p in tables["CANDIDATES"] if p["state_label"]==state and p["family"]==family]
                    self.assertEqual(len(candidates),8); self.assertEqual(sum(p["selected"] for p in candidates),1)
                    self.assertTrue(all(isinstance(p["coefficient_artifact"],dict) and isinstance(p["FIT_prediction_artifact"],dict) for p in candidates))
            row=report.choose(tables["CANDIDATE_SCORES"],state_label="RECON_CP0",family="SCALE_ONLY",candidate_id="IDENTITY",split="fit",mask="complete_C",scale="normalized")
            self.assertEqual(row["observed_counts"],[70]*3); self.assertEqual(row["per_channel_mse"],[.2+.1]*3)
            self.assertEqual(report.choose(tables["RESOURCES"],stage="cost_audit")["wall_seconds"],.5)
            self.assertEqual(sum(p["selected_kind"]=="IDENTITY" for p in tables["SELECTION"]),5)
            self.assertFalse(tables["SPACE_CONTEXT"][0]["new_geometry_computed"]); self.assertFalse(tables["SPACE_CONTEXT"][0]["universal_all_FP32_bound"])

    def test_exact_ties_prefer_identity_then_larger_lambda_and_failures_cannot_win(self):
        with tempfile.TemporaryDirectory() as folder:
            row=fixture(Path(folder))["train"]["states"]["RECON_CP0"]["families"]["SCALE_ONLY"]
            for p in row["candidates"]: p["TUNE_MSE"]=.1
            row["selected_TUNE_MSE"]=.1; self.assertEqual(report.selected(row)["kind"],"IDENTITY")
            row["candidates"][0]["status"]="FAILED_DEPLOYMENT_PARITY_PRESERVED"
            row.update(selected_candidate="RIDGE_10",selected_kind="RIDGE",selected_lambda=10.)
            self.assertEqual(report.selected(row)["lambda"],10.)
            row["selected_lambda"]=1.
            with self.assertRaisesRegex(ValueError,"EXACT_TIE"): report.selected(row)

    def test_two_timing_ratios_remain_distinct_and_large_ns_integer_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            values=fixture(Path(folder)); tables=report.collect_tables(values); row=report.choose(tables["WHOLE_COST"],route="RECON_CP0_SCALE_ONLY",batch=8)
            self.assertEqual(row["paired_RAW_median_ratio"],1.); self.assertEqual(row["ratio_of_RAW_medians"],2/3)
            self.assertEqual(row["paired_matched_Bolt_median_ratio"],1.); self.assertEqual(row["ratio_of_matched_Bolt_medians"],2/3)
            self.assertEqual(tables["WHOLE_COST_SAMPLES"][0]["perf_start_ns"],10**18)
            for change,label in ((lambda v:v["cost"]["cost"]["robin"]["rows"][1].update(paired_RAW_median_ratio=99.),"DISTINCT_PAIRED"),(lambda v:v["cost"]["cost"]["robin"]["rows"][0]["samples"][0].update(perf_end_ns=10**18+5000001),"INTEGER_NS")):
                wrong=copy.deepcopy(values); change(wrong)
                with self.subTest(label=label),self.assertRaisesRegex(ValueError,label): report.collect_tables(wrong)

    def test_report_opposite_tiny_effect_raw_failure_surrogate_scope_and_existing_teacher(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); values=fixture(root); tables=report.collect_tables(values); text=report.make_text(values,tables,{},root)
            self.assertIn("identity 선정은 5/6",text); self.assertIn("TUNE 1/6, 기노출 DEV 0/6",text); self.assertIn("Raw보다 DEV MSE가 낮은 조건은 0/6",text)
            self.assertIn("actual FIT MSE의 후보 최솟값이 identity보다 낮은 조건은 6/6",text); self.assertIn("중복 route의 시간 차이를 서로 다른 학습 방법의 속도 승리로 세지 않는다",text)
            self.assertIn("delta -0.000010000000 (-0.009804%)",text); self.assertIn("delta +0.000200000000 (+0.038462%)",text)
            for token in ("FP64 baseline+Xβ는 이 parity 대상이 아니다","a=-1,b=0","byte-collapse를 보장하지 않는다","별도 byte-exact 경로","같은 세 상태의 두 family를6개 독립 자료로 세지 않는다","Raw-F teacher 증류는 이미 수행","strict K-only도 아니다","5순환은10위치 완전 균형이 아니며","실제 동결 Bolt를 실행","내부 finite/shape 검사","warm2가 전체 비측정 forward2라는 뜻도 아니다"):
                self.assertIn(token,text)
            self.assertIn("2e-06, RMS의 최댓값은 2e-07",text); self.assertIn("FP32 C parity max error의 최댓값은 3e-07",text)
            self.assertEqual(text.count("ms/window; Raw paired"),20)
            for key,rows in tables.items(): self.assertIn(f"{key}.csv](tables/{key}.csv): {len(rows)}행",text)

    def test_all_four_figures_full_render_and_parse_with_identity_and_small_signed_delta(self):
        import xml.etree.ElementTree as ET
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); tables=report.collect_tables(fixture(root)); paths=report.figures(tables,root)
            self.assertEqual(len(paths),8); self.assertEqual(len(set(p.name for p in paths)),8)
            for path in paths:
                self.assertGreater(path.stat().st_size,1000)
                if path.suffix==".png":
                    with Image.open(path) as image: self.assertGreater(image.width,500); self.assertGreater(image.height,500); image.verify()
                else: self.assertTrue(ET.parse(path).getroot().tag.endswith("svg"))

    def test_closed_stage_gate_precedes_any_metadata_or_payload_IO(self):
        with patch.dict(report.FIXED_SHA,{"cost_audit":None}),patch.object(report,"read",side_effect=AssertionError("metadata opened")),patch.object(report,"bound",side_effect=AssertionError("file hashed")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):
            report.load_verified({key:"missing" for key in report.FIXED_SHA})
        with patch.object(report,"FIXED_SHA",{k:"a"*64 for k in report.FIXED_SHA}),patch.object(report,"AUDIT_SOURCES",{"audit":{},"cost_audit":{}}),patch.object(report,"AUDIT_COUNTS",{"fits":6}),patch.object(report,"COST_COUNTS",{"cost_cells_checked":20}),patch.object(report,"read",side_effect=AssertionError("NPZ opened")),patch.object(report,"bound",side_effect=AssertionError("NPZ hashed")),self.assertRaisesRegex(ValueError,"JSON_METADATA_ONLY"):
            report.load_verified({key:"forbidden.npz" for key in report.FIXED_SHA})


if __name__=="__main__": unittest.main()
