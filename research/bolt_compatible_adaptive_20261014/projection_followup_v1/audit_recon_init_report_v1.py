"""Independent metadata-to-published-file checks; no generator or science replay."""
from pathlib import Path
import argparse
import json
import math
import re
from urllib.parse import unquote

import audit_linear_e_report_v1 as common
from audit_linear_e_report_v1 import require,resolve,record,parse,metadata,find,relative_values,check_table


SEEDS=(92601,92602)
TABLES=("SCORES","PAIRED","CHECKPOINTS","UPDATES","CONDITIONS","PROBES","SPACE_CELLS","SPACE_METHODS","SVD_CERTIFICATES","RESOURCE")
REPORT_SHA="cb82dbac77968b510dda060b9534ded20f2386b178f5e9fdc651b473e16e7082"
MANIFEST_SHA="f63836e8060d0b4619e5d9ad1f712aab2884389555b9cac39e8b2c266ba766b7"
REPORT_SOURCE_SHA="fe9a579603f1ef92d298ad76d589dc95410940de94f34d85cab678fd4a5aa685"
INPUT_SHA={"train":"f0fc3f95f1e1a61eff4ffc99b7bb7d62affd34afa064f6e29113bb0c37f52ec8","audit":"b1d28415ade18db8ddc0c987a14e879268413efdd5d1c62ad93e27b24becd52e"}
COMMON_SHA="c87fe5ada4de37e0f46fc61303ea0d18fb0bd9178d4ddc1771e92e6f26acf005"
PARENT_SHA="3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4"
FAILED_AUDIT_SHA="9d5cfdcfd58645669f0a7f1ff56dbceb6e33f99a87784246b79523917ed6693a"


def winner(row):
    require([p["update"] for p in row["checkpoints"]]==[0,100,200,400,600],"FIVE_CHECKPOINTS")
    cp=min(row["checkpoints"],key=lambda p:(p["tune"]["mse"],p["update"]))
    require(cp["update"]==row["selected_update"] and cp["tune"]==row["selected_tune"] and cp["state"]==row["selected_state"],"TUNE_ONLY_SELECTED_CP")
    return cp


def expected_tables(values):
    new,parent,audit=values["train"],values["parent"],values["audit"]; out={key:[] for key in TABLES}
    def append_scores(key,item):
        for mask,field in (("all_observed","scores_all_observed"),("complete_C","scores_complete_mask")):
            for scale in ("normalized","physical_per_channel_diagnostic"):
                out["SCORES"].append({**key,"mask":mask,"scale":scale,**item[field][scale],"artifact":item.get("artifact")})
    for seed in SEEDS:
        rows=(new["datasets"]["robin"]["conditions"][str(seed)],parent["datasets"]["robin"]["conditions"][f"robin_L7_E_LEARN_s{seed}"])
        require(rows[0]["order_sha256"]==rows[1]["order_sha256"],"SAME_PAIRED_MINIBATCH_ORDER")
        for index,row in enumerate(rows):
            method=("RECON_INIT_TRAINED","FORECAST_INIT_TRAINED")[index]; cp=winner(row); base={"seed":seed,"method":method,"new_execution":index==0}
            for split,item in (("fit",row["FIT"]),("tune",cp),("val",row["DEV"])): append_scores({**base,"split":split},item)
            out["CHECKPOINTS"].extend({**base,**item,"selected":item["update"]==cp["update"]} for item in row["checkpoints"])
            updates=[parse(line) for line in metadata(row["updates_log"]).read_text(encoding="utf-8").splitlines()]
            require([r["update"] for r in updates]==list(range(1,601)),"SIX_HUNDRED_UPDATES_PER_CONDITION")
            for item in updates:
                require(item["stage"]=="TRAINING_UPDATE" and all(math.isfinite(item[k]) and item[k]>=0 for k in ("loss","E_gradient_norm_before_clip","D_gradient_norm_before_clip","global_gradient_norm_before_clip")) and item["clipped"]==(item["global_gradient_norm_before_clip"]>1.),"FINITE_LOGGED_LOSS_NORMS_CLIP")
            fraction=sum(x["clipped"] for x in updates)/600; require(fraction==row["global_clipping_fraction"],"RECORDED_CLIP_FRACTION")
            out["UPDATES"].extend({**base,**item} for item in updates)
            out["CONDITIONS"].append({**base,**{k:row[k] for k in ("selected_update","selected_state","selection_lock","order_sha256","global_clipping_fraction","updates_log")},"CP0_parameter_diagnostics":row["checkpoints"][0]["parameter_diagnostics"],"selected_parameter_diagnostics":cp["parameter_diagnostics"],"selected_TUNE_latent_variance":cp["TUNE_latent_variance"],"parameter_change":audit["datasets"]["robin"]["conditions"][str(seed)].get("parameter_change") if index==0 else None})
        probe=rows[0]["probe"]; require(probe["status"]=="PASS" and probe["optimizer_updates"]==0 and all(v["pass"] for v in probe["checks"].values()),"TWO_ZERO_OPTIMIZER_CP0_PROBES")
        out["PROBES"].append({"seed":seed,**probe})
    recon=values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]
    for split in ("fit","tune","val"):
        initial=parent["datasets"]["robin"]["conditions"]["robin_L7_E_LEARN_s92601"]["CP0_TUNE"] if split=="tune" else parent["CP0_LS_and_RAW_baselines"]["robin"]["L7"][split]["CP0_LS"]
        for name,item in (("RECON_CP0",recon[split]),("FORECAST_CP0",initial),("RAW_BOLT",recon[split]["RAW_BOLT"])): append_scores({"seed":0,"method":name,"split":split,"new_execution":False},item)
    for seed in SEEDS:
        for split in ("fit","tune","val"):
            for mask in ("all_observed","complete_C"):
                for scale in ("normalized","physical_per_channel_diagnostic"):
                    key={"seed":seed,"split":split,"mask":mask,"scale":scale}; a,b=(find(out["SCORES"],**key,method=method) for method in ("RECON_INIT_TRAINED","FORECAST_INIT_TRAINED"))
                    out["PAIRED"].append({**key,"RECON_MSE":a["mse"],"FORECAST_MSE":b["mse"],"MSE_RECON_minus_FORECAST":a["mse"]-b["mse"],"RECON_MAE":a["mae"],"FORECAST_MAE":b["mae"],"MAE_RECON_minus_FORECAST":a["mae"]-b["mae"]})
        for init in ("RECON","FORECAST"):
            cells=audit["new_output_space_diagnostics"][str(seed)] if init=="RECON" else values["space"]["datasets"]["robin"]["conditions"][f"FULL_L7_s{seed}"]["splits"]
            for split in ("tune","val"):
                cell=cells[split]; base={"initialization":init,"seed":seed,"split":split,"computed_during_current_audit":init=="RECON"}
                require(cell["status"]=="COMPLETE" and cell["joint_metadata"]["rcond"]==1e-15 and cell["independent_reduced_LS_vs_SVD_verified"] is True and cell["all_real_latent_or_all_FP32_lower_bound_claimed"] is False,"REGISTERED_RETAINED_NUMERIC_SPACE")
                item={**base,"minimum_normalized_MSE":cell["minimum_normalized_MSE"],"current_C_persistence_scores_complete_mask":cell["current_C_persistence_scores_complete_mask"]}
                item.update({label:cell["methods"][name]["scores_complete_mask"]["normalized"]["mse"] for label,name in (("actual_complete_MSE","ACTUAL_F"),("EY_complete_MSE","EY"),("FLAT_complete_MSE","FLAT"))})
                item["RAW_complete_MSE"]=cell["RAW_BOLT_scores_complete_mask"]["normalized"]["mse"]
                item.update({k:cell[k] for k in ("complete_target_positions","target_positions","joint_unknown_future_coordinates","joint_output_coordinates_before_mask","joint_metadata","minimum_solution_scores_complete_C","EY_encoding_backend","EY_incomplete_C_horizons_filled_with_current_C","artifact","input_source","selected_forecast_source")}); out["SPACE_CELLS"].append(item)
                out["SPACE_METHODS"].extend({**base,"method":name,**value} for name,value in cell["methods"].items())
                out["SVD_CERTIFICATES"].append({**base,"certificates":cell["independent_LS_certificates"]})
    out["RESOURCE"]=[{"stage":"new two-fit training","new_deployment_timing":False,**new.get("resource_terminal",{})},{"stage":"saved-training audit plus NEW output-space calculation","new_deployment_timing":False,**audit.get("resource",{})}]
    keys={"SCORES":("seed","method","split","mask","scale"),"PAIRED":("seed","split","mask","scale"),"CHECKPOINTS":("seed","method","update"),"UPDATES":("seed","method","update"),"CONDITIONS":("seed","method"),"PROBES":("seed",),"RESOURCE":("stage",)}
    for name in ("SPACE_CELLS","SPACE_METHODS","SVD_CERTIFICATES"): keys[name]=("initialization","seed","split")+(("method",) if name=="SPACE_METHODS" else ())
    for name,columns in keys.items(): require(len({tuple(r[k] for k in columns) for r in out[name]})==len(out[name]),"UNIQUE_TABLE_KEYS:"+name)
    require([len(out[k]) for k in TABLES]==[84,24,20,2400,4,2,8,24,8,2] and sum(r["new_execution"] for r in out["UPDATES"])==1200,"EXACT_NEW_AND_REUSED_COVERAGE")
    return out


def check_markdown(folder,tables):
    text=(folder/"REPORT_KO.md").read_text(encoding="utf-8"); links=re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text)
    for target in links:
        require(not re.match(r"^[A-Za-z]:|^/|^file:",target),"RELATIVE_MARKDOWN_LINK")
        if not re.match(r"https?://|#",target): require((folder/unquote(target.split("#",1)[0])).resolve().is_file(),"EXISTING_MARKDOWN_LINK")
    def mse(method,seed,split): return find(tables["SCORES"],method=method,seed=seed,split=split,mask="all_observed",scale="normalized")["mse"]
    require(f"TUNE에서 초기 RECON CP0보다 {sum(mse('RECON_INIT_TRAINED',s,'tune')<mse('RECON_CP0',0,'tune') for s in SEEDS)}/2" in text,"MD_TUNE_CP0_COUNT")
    require(f"기노출 DEV에서는 {sum(mse('RECON_INIT_TRAINED',s,'val')>mse('RECON_CP0',0,'val') for s in SEEDS)}/2" in text,"MD_DEV_CP0_COUNT")
    require(f"DEV {sum(mse('RECON_INIT_TRAINED',s,'val')<mse('FORECAST_INIT_TRAINED',s,'val') for s in SEEDS)}/2" in text and f"Raw보다 나쁜 조건은 {sum(mse('RECON_INIT_TRAINED',s,'val')>mse('RAW_BOLT',0,'val') for s in SEEDS)}/2" in text,"MD_PAIRED_AND_RAW_COUNTS")
    require(f"RECON CP0 TUNE {mse('RECON_CP0',0,'tune'):.10f}, DEV {mse('RECON_CP0',0,'val'):.10f}; Raw TUNE {mse('RAW_BOLT',0,'tune'):.10f}, DEV {mse('RAW_BOLT',0,'val'):.10f}." in text,"MD_INITIAL_BASELINE_NUMBERS")
    dev=[r for r in tables["SPACE_CELLS"] if r["initialization"]=="RECON" and r["split"]=="val"]
    require(f"Raw보다 {sum(r['minimum_normalized_MSE']>r['RAW_complete_MSE'] for r in dev)}/2 조건에서 높았다" in text,"MD_CONDITIONAL_MINIMUM_COUNT")
    for seed in SEEDS:
        a,b=(find(tables["CONDITIONS"],method=m,seed=seed) for m in ("RECON_INIT_TRAINED","FORECAST_INIT_TRAINED"))
        line=f"- seed {seed}: 새 RECON-init CP{a['selected_update']}, TUNE {mse('RECON_INIT_TRAINED',seed,'tune'):.10f}, DEV {mse('RECON_INIT_TRAINED',seed,'val'):.10f}; 재사용 FORECAST-init CP{b['selected_update']}, TUNE {mse('FORECAST_INIT_TRAINED',seed,'tune'):.10f}, DEV {mse('FORECAST_INIT_TRAINED',seed,'val'):.10f}."
        require(line in text,"MD_PAIRED_CP_AND_SCORE_ROW")
    for row in tables["SPACE_CELLS"]:
        require(f"- {row['initialization']} / {row['seed']} / {row['split']}: actual {row['actual_complete_MSE']:.10f}, E(Y) {row['EY_complete_MSE']:.10f}, minimum {row['minimum_normalized_MSE']:.10f}, Raw {row['RAW_complete_MSE']:.10f}; complete {row['complete_target_positions']}/{row['target_positions']}." in text,"MD_ALL_EIGHT_SPACE_NUMERIC_ROWS")
    for name,rows in tables.items(): require(f"{name}.csv](tables/{name}.csv): {len(rows)}행" in text,"MD_TABLE_COUNTS")
    for token in ("1,200 update", "재사용", "optimizer update 0", "strict K-only는 아니다", "감사 중 새 계산", "외부 감사자가 다시 재현했다고 부르지 않는다", "모든 FP32/실수 예측의 보편 하한", "E(Y)는 실제 미래", "새 deployment 비용은 계획대로 미측정", "독립 일반화 검증이 아니다", "dict를 통째 비교", "path·SHA", "고정 D 대조"):
        require(token in text,"MD_SCOPE:"+token)
    return len(links)


def verify(folder):
    require(all(isinstance(v,str) and len(v)==64 for v in (REPORT_SHA,MANIFEST_SHA,REPORT_SOURCE_SHA,*INPUT_SHA.values())),"ACTUAL_REPORT_NOT_YET_REGISTERED")
    require(record(common.__file__)["sha256"]==COMMON_SHA,"IMMUTABLE_METADATA_HELPERS")
    folder=resolve(folder); report,manifest_path=folder/"REPORT_KO.md",folder/"MANIFEST.json"
    require(record(report)["sha256"]==REPORT_SHA and record(manifest_path)["sha256"]==MANIFEST_SHA,"EXACT_ACTUAL_REPORT_MANIFEST")
    manifest=parse(manifest_path.read_text(encoding="utf-8")); require(manifest["status"]=="COMPLETE" and Path(manifest["source"]["path"]).name=="make_recon_init_sensitivity_report_v1.py" and manifest["source"]["sha256"]==REPORT_SOURCE_SHA,"EXACT_REPORT_SOURCE")
    metadata(manifest["source"]); values={}
    for binding in manifest["inputs"].values(): metadata(binding)
    for key,digest in INPUT_SHA.items():
        item=manifest["inputs"][key]; require(item["sha256"]==digest,"EXACT_INPUT_SHA")
        values[key]=parse(metadata(item).read_text(encoding="utf-8-sig")); require(values[key]["status"]==("PASS" if key=="audit" else "COMPLETE") and values[key]["protected_TEST_accessed"] is False,"COMPLETE_TRAIN_PASS_NO_TEST")
    train,audit=values["train"],values["audit"]
    require(all(audit["diagnostic"][k]==manifest["inputs"]["train"][k] for k in ("path","sha256")) and audit["training_verification_status"]=="PASS" and audit["new_space_diagnostic_status"]=="COMPLETE" and not audit["new_output_space_diagnostic_failures"],"EXACT_TRAIN_AUDIT_AND_NEW_DIAGNOSTICS")
    for key,source in (("parent","full_gradient_parent"),("science","causal_lag_diagnostic"),("space","learned_output_space_diagnostic")):
        item=train[source]; require(all(audit[source][k]==item[k] for k in ("path","sha256")),"PARENT_METADATA_BINDING"); values[key]=parse(metadata(item).read_text(encoding="utf-8-sig"))
    require(train["full_gradient_parent"]["sha256"]==PARENT_SHA and values["parent"]["status"]=="COMPLETE","EXACT_REUSED_TWO_FORECAST_CONDITIONS")
    registry=parse(metadata(train["selection_registry"]).read_text(encoding="utf-8")); require(registry["status"]=="ALL_TWO_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and {r["seed"] for r in registry["conditions"]}==set(SEEDS) and len(registry["conditions"])==2,"TWO_SELECTION_LOCKS_BEFORE_DEV")
    for item in registry["conditions"]:
        lock=parse(metadata(item["selection_lock"]).read_text(encoding="utf-8")); require(lock["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False,"COMPLETE_ORIGINAL_LOCK")
    old=manifest["inputs"]["preserved_failed_audit"]; require(old["sha256"]==FAILED_AUDIT_SHA and parse(metadata(old).read_text(encoding="utf-8"))["status"]=="FAILED_PRESERVED","FAILED_METADATA_AUDIT_PRESERVED")
    require(manifest["scientific_NPZ_or_model_read"] is False and manifest["new_training_updates_during_generation"]==0 and manifest["new_whole_deployment_cost_measured"] is manifest["protected_TEST_accessed"] is False,"METADATA_ONLY_REPORT_SCOPE")
    tables=expected_tables(values); compared,refs={},0
    for key,rows in tables.items():
        require(manifest["table_row_counts"][key]==len(rows),"MANIFEST_ROW_COUNTS")
        expected=relative_values(rows,folder); compared[key]=check_table(folder/"tables"/(key+".csv"),expected); refs+=common.check_reference_paths(expected,folder)
    paths=common.check_manifest_files(folder,manifest["files"]); png,svg=common.inspect_images(paths)
    require(len(paths)==17 and png==svg==3 and {p.resolve() for p in folder.rglob("*") if p.is_file()}==set(paths)|{manifest_path.resolve()},"TEN_TABLES_SIX_IMAGES_EXACT_FILES")
    return {"status":"PASS","report":record(report),"manifest":record(manifest_path),"source":manifest["source"],"inputs":{k:manifest["inputs"][k] for k in INPUT_SHA},"table_rows_checked":{k:len(v) for k,v in tables.items()},"CSV_primitive_fields_compared":compared,"total_CSV_primitive_fields_checked":sum(compared.values()),"files_checked":len(paths),"PNG_count":png,"SVG_count":svg,"local_links_checked":check_markdown(folder,tables),"relative_artifact_path_existence_checks":refs,"verification_scope":"All ten CSV expected primitive leaves, masks/per-channel scores/paired deltas, 20 CP and TUNE selection, 2400 logged updates with 1200 new, recorded probes, eight numeric spaces and all stored certificate metadata, resources, MD numbers/scope, file hashes and image formats. No original scientific arrays, F/optimizer, diagnostic math, timing or visual-layout replay.","scientific_array_model_F_training_or_new_space_calculation_replayed":False,"generator_imported_or_called":False,"figure_visual_review_performed_by_this_script":False,"protected_TEST_accessed":False,"audit_source":record(__file__)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--report-dir",required=True); parser.add_argument("--output",required=True); args=parser.parse_args()
    try: result=verify(args.report_dir)
    except Exception as error: result={"status":"FAILED_PRESERVED","failure":{"type":type(error).__name__,"message":str(error)},"audit_source":record(__file__)}
    with resolve(args.output).open("x",encoding="utf-8") as handle: json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":result["status"],"output":str(resolve(args.output))}),flush=True); raise SystemExit(0 if result["status"]=="PASS" else 2)
