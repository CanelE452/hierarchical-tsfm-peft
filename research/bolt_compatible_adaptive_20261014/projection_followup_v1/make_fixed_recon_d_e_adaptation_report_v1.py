"""JSON/JSONL-only E-only versus joint E/D report with the same RECON initial state."""
from pathlib import Path
import argparse
import json

from make_linear_e_adaptation_report_v2 import require,resolve,sha,record,read,bound,same_opaque,choose,rel,write_csv
from make_recon_init_sensitivity_report_v1 import selected


HERE=Path(__file__).resolve().parent
SEEDS=(92601,92602)
FIXED_SHA={"train":"4b1559216ea99937ca8e7e0f6cdac9885abc29479a6050d82336dcfa0305bb2b","audit":"f79324dc9452192601ac16b483e4b19c296f3c252f5814062c7c05f7d4f9e85b"}
AUDITOR_SOURCE={"path":str(HERE/"audit_fixed_recon_d_e_adaptation_v1.py"),"sha256":"adaae4d3bc0767192ac8a98a4cacf2f087800558fe72d8b391cafeb72a5c2fba"}
AUDIT_COUNTS={"E_only_fits_checked":2,"checkpoint_states_checked":10,"checkpoint_TUNE_predictions_checked":10,"selected_FIT_DEV_predictions_checked":4,"logged_updates_checked":1200,"recorded_post_optimizer_D_flag_checks":1200,"CP0_probe_cases_checked":2,"CP0_component_byte_comparisons_checked":16,"fixed_D_center_CP_byte_comparisons_checked":20,"reused_joint_conditions_checked":2,"training_score_fields_checked":396,"new_space_diagnostic_cells_computed":4,"new_space_LS_SVD_crosschecks_computed":4,"selection_locks_checked":2,"protected_source_and_input_file_hashes_checked":53}
CONTROLLER_SHA="1c5f5a1e8f878b15705fc8f808af35ab8bc0ea9cdbc21375b7ad789a5dc45ff1"
JOINT_SHA="f0fc3f95f1e1a61eff4ffc99b7bb7d62affd34afa064f6e29113bb0c37f52ec8"
JOINT_AUDIT_SHA="b1d28415ade18db8ddc0c987a14e879268413efdd5d1c62ad93e27b24becd52e"
UTILITIES=({"path":str(HERE/"make_linear_e_adaptation_report_v2.py"),"sha256":"32e2460ea4724144067f264ab1ceb0332cd04c5fce2296838024c119c9572466"},{"path":str(HERE/"make_recon_init_sensitivity_report_v1.py"),"sha256":"fe9a579603f1ef92d298ad76d589dc95410940de94f34d85cab678fd4a5aa685"})
SCALES=("normalized","physical_per_channel_diagnostic")
TABLES=("SCORES","PAIRED","CHECKPOINTS","UPDATES","CONDITIONS","PROBES","SPACE_CELLS","SPACE_METHODS","SVD_CERTIFICATES","RESOURCE")


def load_verified(paths):
    require(set(paths)==set(FIXED_SHA),"TWO_EXACT_STAGE_INPUTS")
    require(all(isinstance(v,str) and len(v)==64 for v in FIXED_SHA.values()) and isinstance(AUDITOR_SOURCE,dict) and bool(AUDIT_COUNTS),"ACTUAL_E_ONLY_TRAIN_AUDIT_NOT_YET_REGISTERED")
    for item in UTILITIES: bound(item)
    values={k:read(p) for k,p in paths.items()}; bindings={k:record(p) for k,p in paths.items()}
    for key in paths: require(bindings[key]["sha256"]==FIXED_SHA[key] and values[key]["status"]==("PASS" if key=="audit" else "COMPLETE") and values[key]["protected_TEST_accessed"] is False,"EXACT_COMPLETE_PASS_NO_TEST")
    train,audit=values["train"],values["audit"]; bound(audit["diagnostic"],bindings["train"]); bound(audit["audit_source"],AUDITOR_SOURCE)
    require(audit["training_verification_status"]=="PASS" and audit["new_space_diagnostic_status"]=="COMPLETE" and not audit["partial_paths"] and not audit["new_output_space_diagnostic_failures"] and all(audit["totals"].get(k)==v for k,v in AUDIT_COUNTS.items()),"TRAIN_VERIFICATION_AND_FOUR_NEW_SPACE_DIAGNOSTICS_COMPLETE")
    require(train["controller_source"]["sha256"]==CONTROLLER_SHA and train["joint_recon_diagnostic"]["sha256"]==JOINT_SHA and train["joint_recon_audit"]["sha256"]==JOINT_AUDIT_SHA,"EXACT_EXECUTED_SOURCE_AND_JOINT_CONTROLS")
    for key in ("joint_recon_diagnostic","joint_recon_audit","full_gradient_parent","full_gradient_parent_audit","causal_lag_diagnostic","causal_lag_audit","selection_registry","joint_selection_registry","plan"):
        bound(train[key]); bound(audit[key],train[key]); bindings[key]=train[key]
    for key,field in (("joint","joint_recon_diagnostic"),("joint_audit","joint_recon_audit"),("science","causal_lag_diagnostic")):
        values[key]=read(train[field]["path"]); require(values[key]["status"]==("PASS" if key=="joint_audit" else "COMPLETE"),"COMPLETE_ACCEPTED_REUSED_METADATA")
    require(train["F_before"]==train["F_after"]==values["joint"]["F_before"]==values["joint"]["F_after"],"SAME_FROZEN_F")
    require(train["fixed_D_E_only_training_fits"]==train["resolved_selection_locks"]==2 and train["training_updates_completed"]==1200 and train["checkpoint_states_saved"]==10 and train["probe_optimizer_updates"]==train["new_joint_RECON_fits"]==0 and train["joint_RECON_conditions_reused"]==2 and train["all_2_locks_before_DEV"] is True and not train["failed_paths"],"TWO_NEW_E_ONLY_FITS_AND_TWO_REUSED_JOINT_CONTROLS")
    for key in ("new_backbone_training","new_D_initialization_fit","D_updated_during_training","D_initialization_supervised_by_FIT_future","independent_holdout","strict_K_only","whole_deployment_cost_measured","new_output_space_diagnostics_computed_by_training_producer"): require(train[key] is False,"FIXED_D_SCOPE:"+key)
    require(train["E_training_supervised_by_FIT_future"] is train["F_input_gradient_retained"] is True and train["gradient_clipping_parameter_scope"]=="E_ONLY","E_ONLY_FORECAST_SUPERVISION_AND_CLIPPING_SCOPE")
    registry=read(train["selection_registry"]["path"]); require(registry["status"]=="ALL_TWO_RESOLVED_BEFORE_DEV" and registry["DEV_opened_before_seal"] is False and {x["seed"] for x in registry["conditions"]}==set(SEEDS) and len(registry["conditions"])==2,"TWO_LOCKS_BEFORE_DEV")
    for item in registry["conditions"]:
        row=train["datasets"]["robin"]["conditions"][str(item["seed"])]; lock=read(bound(item["selection_lock"],row["selection_lock"]))
        require(lock["status"]==row["status"]==item["status"]=="COMPLETE" and lock["DEV_opened_before_lock"] is False and lock["selected_update"]==row["selected_update"],"EXACT_NEW_SELECTED_CP_LOCK"); same_opaque(lock["selected_state"],row["selected_state"])
    for key in ("controller_source","parent_report","parent_report_manifest","parent_report_audit"): bound(train[key]); bindings[key]=train[key]
    bindings.update(audit_source=AUDITOR_SOURCE,metadata_utility=UTILITIES[0],selection_utility=UTILITIES[1]); return values,bindings


def fixed_flags(flags):
    return flags["trainable_parameter_names"]==["E"] and flags["D_requires_grad"] is False and flags["D_grad_is_None"] is True and flags["center_requires_grad"] is False


def collect_tables(values):
    train,joint,audit=values["train"],values["joint"],values["audit"]; out={key:[] for key in TABLES}
    require(set(train["datasets"])=={"robin"} and set(train["datasets"]["robin"]["conditions"])=={str(s) for s in SEEDS},"EXACT_TWO_NEW_CONDITIONS")
    def scores(base,item):
        for mask,key in (("all_observed","scores_all_observed"),("complete_C","scores_complete_mask")):
            for scale in SCALES: out["SCORES"].append({**base,"mask":mask,"scale":scale,**item[key][scale],"artifact":item.get("artifact")})
    for seed in SEEDS:
        new=train["datasets"]["robin"]["conditions"][str(seed)]; old=joint["datasets"]["robin"]["conditions"][str(seed)]
        require(new["order_sha256"]==old["order_sha256"],"SAME_SEED_MINIBATCH_ORDER")
        require(new["D_trainable"] is False and fixed_flags(new["fixed_decoder_final_status"]) and new["D_frozen_after_optimizer_step_checks"]==600 and new["D_and_center_checkpoint_byte_checks"]==5 and new["D_bytes_checked_at_checkpoints_and_final_only"] is True and new["independent_per_update_D_bytes_replayed"] is False and new["gradient_clipping_parameter_scope"]=="E_ONLY","FIXED_D_RECORDED_FLAGS_AND_CHECK_SCOPE")
        for method,row in (("E_ONLY",new),("JOINT_E_D",old)):
            cp=selected(row); is_new=method=="E_ONLY"; base={"method":method,"seed":seed,"new_execution":is_new}
            if is_new: require(all(p["D_and_center_exact_initial_bytes"] is True and fixed_flags(p["fixed_decoder_status"]) for p in row["checkpoints"]),"FIXED_D_AT_ALL_FIVE_CPS")
            for split,item in (("fit",row["FIT"]),("tune",cp),("val",row["DEV"])): scores({**base,"split":split},item)
            out["CHECKPOINTS"] += [{**base,**p,"selected":p["update"]==cp["update"]} for p in row["checkpoints"]]
            path=bound(row["updates_log"]); require(path.suffix==".jsonl","JSONL_ONLY_UPDATE_LOG")
            logs=[json.loads(s,parse_constant=lambda v: (_ for _ in ()).throw(ValueError("NONFINITE_LOG:"+v))) for s in path.read_text(encoding="utf-8").splitlines()]
            require([p["update"] for p in logs]==list(range(1,601)) and sum(p["clipped"] for p in logs)/600==row["global_clipping_fraction"],"SIX_HUNDRED_LOGGED_UPDATES_AND_CLIP_FRACTION")
            if is_new: require(all(p["D_gradient_norm_before_clip"]==0. and p["gradient_clipping_parameter_scope"]=="E_ONLY" and fixed_flags(p) for p in logs),"E_ONLY_LOG_FLAGS_AND_ZERO_D_GRADIENT")
            out["UPDATES"] += [{**base,**p} for p in logs]
            condition={**base,"selected_update":cp["update"],"selected_state":row["selected_state"],"selection_lock":row["selection_lock"],"order_sha256":row["order_sha256"],"updates_log":row["updates_log"],"global_clipping_fraction":row["global_clipping_fraction"],"gradient_clipping_parameter_scope":"E_ONLY" if is_new else "E_AND_D","CP0_parameter_diagnostics":row["checkpoints"][0]["parameter_diagnostics"],"selected_parameter_diagnostics":cp["parameter_diagnostics"],"selected_TUNE_latent_variance":cp["TUNE_latent_variance"],"parameter_change":values["audit" if is_new else "joint_audit"]["datasets"]["robin"]["conditions"][str(seed)].get("parameter_change")}
            if is_new: condition.update({k:row[k] for k in ("D_frozen_after_optimizer_step_checks","D_and_center_checkpoint_byte_checks","fixed_decoder_final_status","initial_D_value_sha256","D_bytes_checked_at_checkpoints_and_final_only","independent_per_update_D_bytes_replayed")})
            out["CONDITIONS"].append(condition)
        probe=new["probe"]; require(probe["status"]=="PASS" and probe["optimizer_updates"]==0 and set(probe["component_byte_exact"])=={"prediction","latent_history","latent_future","lag_features"} and all(probe["component_byte_exact"].values()) and fixed_flags(probe["fixed_decoder_status"]),"CP0_SAME_JOINT_STATE_COMPONENT_BYTES_ZERO_UPDATE")
        out["PROBES"].append({"seed":seed,**probe})
    recon=values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]
    for split in ("fit","tune","val"):
        for method,item in (("RECON_CP0",recon[split]),("RAW_BOLT",recon[split]["RAW_BOLT"])): scores({"method":method,"seed":0,"split":split,"new_execution":False},item)
    for seed in SEEDS:
        for split in ("fit","tune","val"):
            for mask in ("all_observed","complete_C"):
                for scale in SCALES:
                    key={"seed":seed,"split":split,"mask":mask,"scale":scale}; a,b=(choose(out["SCORES"],**key,method=m) for m in ("E_ONLY","JOINT_E_D"))
                    out["PAIRED"].append({**key,"E_ONLY_MSE":a["mse"],"JOINT_MSE":b["mse"],"MSE_E_ONLY_minus_JOINT":a["mse"]-b["mse"],"E_ONLY_MAE":a["mae"],"JOINT_MAE":b["mae"],"MAE_E_ONLY_minus_JOINT":a["mae"]-b["mae"]})
        for method,owner in (("E_ONLY",audit),("JOINT_E_D",values["joint_audit"])):
            for split in ("tune","val"):
                cell=owner["new_output_space_diagnostics"][str(seed)][split]; base={"method":method,"seed":seed,"split":split,"computed_during_current_audit":method=="E_ONLY"}
                require(cell["status"]=="COMPLETE" and cell["independent_reduced_LS_vs_SVD_verified"] is True and cell["all_real_latent_or_all_FP32_lower_bound_claimed"] is False and cell["diagnostic_future_labels_used_for_new_fit_or_selection"] is False,"REGISTERED_NUMERIC_SPACE_SCOPE")
                if method=="E_ONLY": require(cell["selected_D_previously_supervised_by_FIT_future_targets"] is False and cell["selected_E_previously_supervised_by_FIT_future_targets"]==(train["datasets"]["robin"]["conditions"][str(seed)]["selected_update"]>0),"SEPARATE_SELECTED_E_AND_FIXED_D_SUPERVISION")
                row={**base,**{k:cell[k] for k in ("minimum_normalized_MSE","complete_target_positions","target_positions","joint_unknown_future_coordinates","joint_output_coordinates_before_mask","joint_metadata","minimum_solution_scores_complete_C","EY_encoding_backend","EY_incomplete_C_horizons_filled_with_current_C","artifact","input_source","selected_forecast_source")}}
                row.update({label:cell["methods"][name]["scores_complete_mask"]["normalized"]["mse"] for label,name in (("actual_complete_MSE","ACTUAL_F"),("EY_complete_MSE","EY"),("FLAT_complete_MSE","FLAT"))}); row["RAW_complete_MSE"]=cell["RAW_BOLT_scores_complete_mask"]["normalized"]["mse"]
                row.update(selected_E_previously_supervised_by_FIT_future_targets=cell.get("selected_E_previously_supervised_by_FIT_future_targets",cell.get("selected_E_D_previously_supervised_by_FIT_future_targets")),selected_D_previously_supervised_by_FIT_future_targets=cell.get("selected_D_previously_supervised_by_FIT_future_targets",cell.get("selected_E_D_previously_supervised_by_FIT_future_targets")),current_C_persistence_scores_complete_mask=cell["current_C_persistence_scores_complete_mask"]); out["SPACE_CELLS"].append(row)
                out["SPACE_METHODS"] += [{**base,"future_method":name,**value} for name,value in cell["methods"].items()]
                out["SVD_CERTIFICATES"].append({**base,"certificates":cell["independent_LS_certificates"]})
    out["RESOURCE"]=[{"stage":"new E-only two-fit training" if k=="train" else "saved-training audit plus NEW selected-space computation","new_deployment_timing":False,**values[k].get("resource_terminal" if k=="train" else "resource",{})} for k in ("train","audit")]
    require([len(out[k]) for k in TABLES]==[72,24,20,2400,4,2,8,24,8,2],"FULL_TWO_SEED_PROCEDURE_COMPARISON")
    return out


def figures(tables,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder=output/"figures"; folder.mkdir(); files=[]; colors={"E_ONLY":"#0072B2","JOINT_E_D":"#D55E00"}
    def save(fig,name,bottom=.16):
        fig.tight_layout(rect=(0,bottom,1,.90))
        for ext in ("png","svg"):
            p=folder/(name+"."+ext); fig.savefig(p,dpi=150); files.append(p)
        plt.close(fig)
    def mse(method,seed,split): return choose(tables["SCORES"],method=method,seed=seed,split=split,mask="all_observed",scale="normalized")["mse"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,split in zip(axes,("tune","val")):
        for method,color in colors.items(): ax.plot(range(2),[mse(method,s,split) for s in SEEDS],"o-",color=color,label=method)
        for method,color,style in (("RECON_CP0","#009E73",":"),("RAW_BOLT","black","--")): ax.axhline(mse(method,0,split),color=color,linestyle=style,label=method)
        ax.set_xticks(range(2),[str(s) for s in SEEDS]); ax.set_title("Robin / "+("TUNE selection" if split=="tune" else "exposed DEV")); ax.set_ylabel("All-observed normalized MSE"); ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=4); fig.suptitle("Same RECON initialization: fixed-D E adaptation versus reused joint E/D; CP0 and Raw references"); save(fig,"01_E_only_joint_errors")
    fig,ax=plt.subplots(figsize=(11,4.8))
    for method,color in colors.items():
        for seed,style in zip(SEEDS,("-","--")):
            rows=[r for r in tables["CHECKPOINTS"] if r["method"]==method and r["seed"]==seed]; ax.plot([r["update"] for r in rows],[r["tune"]["mse"] for r in rows],marker="o",linestyle=style,color=color,label=method+" / "+str(seed))
            cp=next(r for r in rows if r["selected"]); ax.scatter(cp["update"],cp["tune"]["mse"],s=170,marker="*",color=color,zorder=4)
    ax.set_xlabel("Optimizer update; stars are TUNE-selected CPs"); ax.set_ylabel("All-observed TUNE normalized MSE"); ax.grid(alpha=.2); fig.legend(*ax.get_legend_handles_labels(),loc="lower center",ncol=2)
    fig.suptitle("1,200 new E-only updates versus 1,200 reused joint updates; E-only and E/D clipping scopes differ"); save(fig,"02_TUNE_checkpoints")
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,split in zip(axes,("tune","val")):
        rows=[choose(tables["SPACE_CELLS"],method=m,seed=s,split=split) for m in ("JOINT_E_D","E_ONLY") for s in SEEDS]
        for key,color,marker,label in (("actual_complete_MSE","#0072B2","o","Saved actual F"),("EY_complete_MSE","#D55E00","x","E(Y): future diagnostic"),("minimum_normalized_MSE","#009E73","*","Retained numeric minimum")): ax.plot(range(4),[r[key] for r in rows],color=color,marker=marker,label=label)
        ax.axhline(rows[0]["RAW_complete_MSE"],color="black",linestyle="--",label="Raw same complete-C"); ax.set_xticks(range(4),[r["method"]+"/"+str(r["seed"])[-2:] for r in rows],rotation=15); ax.set_title("Robin / "+split); ax.set_ylabel("Complete-C normalized MSE"); ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2); fig.suptitle("E-only D stays fixed while its known-past offset can change; numeric minima do not prove past predictability"); save(fig,"03_selected_output_spaces")
    return files


def make_text(values,tables,bindings,output):
    def mse(method,seed,split): return choose(tables["SCORES"],method=method,seed=seed,split=split,mask="all_observed",scale="normalized")["mse"]
    initial=mse("RECON_CP0",0,"val"); raw=mse("RAW_BOLT",0,"val")
    better=sum(mse("E_ONLY",s,"val")<mse("JOINT_E_D",s,"val") for s in SEEDS); tune_worse=sum(mse("E_ONLY",s,"tune")>mse("JOINT_E_D",s,"tune") for s in SEEDS)
    initial_worse=sum(mse("E_ONLY",s,"val")>initial for s in SEEDS); raw_worse=sum(mse("E_ONLY",s,"val")>raw for s in SEEDS)
    spaces=[r for r in tables["SPACE_CELLS"] if r["method"]=="E_ONLY"]
    tune_min_above=sum(r["minimum_normalized_MSE"]>r["RAW_complete_MSE"] for r in spaces if r["split"]=="tune")
    dev_min_below=sum(r["minimum_normalized_MSE"]<r["RAW_complete_MSE"] for r in spaces if r["split"]=="val")
    lines=["# 같은 RECON D를 고정한 E 적응과 E/D 공동 적응", "",
        f"[확인] D를 고정한 E_ONLY는 기노출 DEV에서 재사용 JOINT_E_D보다 {better}/2 조건이 좋았지만 TUNE는 {tune_worse}/2 조건에서 나빴다. DEV는 학습 전 RECON CP0보다 {initial_worse}/2, Raw보다 {raw_worse}/2 조건이 나빴다. D 고정이 현재 절차의 DEV 악화를 줄였더라도 해결이나 독립 일반화 성공으로 보지 않는다.","",
        f"All-observed 정규화 MSE 기준 RECON CP0 TUNE {mse('RECON_CP0',0,'tune'):.10f}, DEV {initial:.10f}; Raw TUNE {mse('RAW_BOLT',0,'tune'):.10f}, DEV {raw:.10f}.","",
        f"[확인] 같은 complete-C에서 새 E_ONLY minimum은 TUNE {tune_min_above}/2 조건에서 Raw보다 높고, DEV {dev_min_below}/2 조건에서 Raw보다 낮았다. D가 같아도 E가 만든 known-past offset은 달라진다. 낮은 DEV numeric minimum은 실제 F 출력의 개선이나 과거 예측 가능성의 증명이 아니며 실제 점수·E(Y)와 나눠 봐야 한다.",""]
    for seed in SEEDS:
        a,b=(choose(tables["CONDITIONS"],method=m,seed=seed) for m in ("E_ONLY","JOINT_E_D"))
        lines.append(f"- seed {seed}: 새 E_ONLY CP{a['selected_update']}, TUNE {mse('E_ONLY',seed,'tune'):.10f}, DEV {mse('E_ONLY',seed,'val'):.10f}; 재사용 JOINT_E_D CP{b['selected_update']}, TUNE {mse('JOINT_E_D',seed,'tune'):.10f}, DEV {mse('JOINT_E_D',seed,'val'):.10f}.")
    lines += ["","![실제 점수와 초기값](figures/01_E_only_joint_errors.png)","",
        "**같은 입력과 다른 학습 범위.** Robin C17/K4, L512/H48, FIT256/TUNE64/노출 VAL30, 원 scaler·origin·mask·현재 C anchor와 lag [0,1,2,4,8,16,32]를 유지했다. 새 모델은 같은 RECON 초기 E/D/center와 원 JOINT CP0 상태·4개 forward component를 byte-exact로 대조했고 CP0 probe optimizer update는0이다. 기존 FIT 과거 재구성 D32와 center를 고정하고 E만 FIT 미래 actual-Y 예측 손실로 학습했다. E와 D의 감독 이력을 나눠 기록한다. F 파라미터는 동결하되 실제 F 입력 gradient는 유지했다.","",
        "새 2fit 1,200 update·10CP와 기존 같은 RECON 초기 JOINT_E_D 2fit 1,200 update·10CP 재사용이다. 같은 seed별600×4 minibatch 순서·고정 FIT 관측수 가중 macro MSE·AdamW lr.001/WD0/clip1·CP0/100/200/400/600·전체 TUNE 최소와 이른 tie를 유지했다. 두 성공/실패 lock·registry를 봉인한 뒤 DEV를 추론했다. clipping은 새 E_ONLY에서 E만, JOINT_E_D에서 E와 D를 함께 제한하므로 E의 유효 update scale도 달라질 수 있다. 같은 계산량이나 D 변화의 순수 단일 인과효과로 부르지 않는다. 실제 per-step AdamW update norm은 미측정이다.","",
        "![TUNE 선정 곡선](figures/02_TUNE_checkpoints.png)","",
        "**D 고정 검증의 범위.** 새 각600 optimizer-step 뒤 requires_grad/grad=None flag를 검사하고 로그 D gradient norm=0 및 E_ONLY clipping 범위를 보존했다. 저장된 각5CP와 선정 복원 시점에서 D·center bytes가 초기값과 같음을 검사했다. 이것이 감사자가 실제 optimizer나 매 update bytes를 독립 재생했다는 뜻은 아니다. CP와 최종 bytes 검산, producer의 per-step flag 기록, 저장 score·E/lag/D 산술 검산을 구분한다.","",
        "**선정된 공간.** 새 E-only TUNE/DEV4cell은 학습 감사 중 새 계산이고 JOINT4cell은 이미 PASS한 RECON 감사의 계산 재사용이다. 같은 complete-C 마스크에서 actual/E(Y)/FLAT와 minimum을 보존한다. E-only D와 시간축 공동 행렬 A는 원 RECON 값으로 고정됐지만 학습된 E가 만든 known-past offset은 달라질 수 있다. JOINT 선정 D는 별도 상태이므로 그 A와 같다고 주장하지 않는다. 두 불변 SVD/LS 수학 경로 교차이며 세 번째 외부 재현은 아니다.",""]
    for row in tables["SPACE_CELLS"]: lines.append(f"- {row['method']} / {row['seed']} / {row['split']}: actual {row['actual_complete_MSE']:.10f}, E(Y) {row['EY_complete_MSE']:.10f}, minimum {row['minimum_normalized_MSE']:.10f}, Raw {row['RAW_complete_MSE']:.10f}; complete {row['complete_target_positions']}/{row['target_positions']}.")
    lines += ["","![공간과 잠재 미래 진단](figures/03_selected_output_spaces.png)","",
        "minimum은 D32·known-past offset·같은 관측 출력 행·rcond1e-15 retained numeric space에 조건부인 정규화 MSE 해다. 모든 실수/FP32 미래나 비선형 표현의 보편 하한·과거 예측 가능성은 아니다. HK 미래 미지수를 모두 유지하며 H×7K가 아니다. MAE·물리 단위 값은 이 MSE 해의 점수이지 각각의 minimum이 아니다. 원 real/FP32 false-bound flags와 버린 성분을 그대로 남겼다. E(Y)는 실제 미래를 사용하는 NumPy FP32 진단이며 불완전 C horizon의 현재 C 채움이 L7 뒤 출력에도 전파될 수 있어 배포 불가다. FLAT latent persistence는 알려진 lag prefix 때문에 현재 C persistence와 같다고 볼 수 없다.","",
        "**남은 판단.** D 고정과 E/D 공동 갱신의 TUNE/DEV·공간 차이를 분리해 읽어야 한다. E-only가 공동 적응보다 좋아도 학습 전 RECON/Raw를 이기지 못하면 현재 압축 예측 문제의 해결은 아니다. 큰 모델 학습·무조건 연장·대조학습/그래프/Bolt 필요성으로 확대하지 않는다. 기존 joint-Y/RF3 등 예측 목적 E/D 학습을 이미 수행했다. 기노출 DEV의 문제를 보고 고른 적응적 대조이므로 독립 일반화 검증이 아니다. K streams+K 과거 buffer+현재 C anchor이므로 strict K-only도 아니다.","",
        "새 deployment 비용은 현재 미측정이며 다른 cohort의 기존 timing을 차용하지 않는다. RESOURCE는 새 두 학습 실행과 저장 결과 감사·공간 계산 자원이다. 실제 F forward/backward·optimizer·매 update loss를 감사자가 독립 재실행한 것은 아니다. 원 결과·실패·선정·SOURCE는 보존했다. 보고서 생성은 JSON/JSONL metadata만 읽으며 원 NPZ/model/F/학습/TEST/Git을 실행하지 않는다.",""]
    lines += [f"- [{k}.csv](tables/{k}.csv): {len(v)}행" for k,v in tables.items()]
    lines += [""]+[f"- [{k}]({rel(v['path'],output)}): `{v['sha256']}`" for k,v in bindings.items()]
    return "\n".join(lines)+"\n"


def generate(paths,destination):
    output=resolve(destination); require(output.parent==(HERE/"reports").resolve() and not output.exists(),"FRESH_LOCAL_REPORT_REQUIRED")
    values,bindings=load_verified(paths); tables=collect_tables(values); source=record(__file__); bindings["report_source"]=source
    output.mkdir(parents=True,exist_ok=False); (output/"tables").mkdir()
    for name,rows in tables.items(): write_csv(output/"tables"/(name+".csv"),rows,output)
    images=figures(tables,output); (output/"REPORT_KO.md").write_text(make_text(values,tables,bindings,output),encoding="utf-8",newline="\n")
    require(record(__file__)==source,"REPORT_SOURCE_UNCHANGED")
    for item in bindings.values(): bound(item)
    manifest={"status":"COMPLETE","source":source,"inputs":bindings,"files":[{"path":p.relative_to(output).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()],"table_row_counts":{k:len(v) for k,v in tables.items()},"PNG_count":3,"SVG_count":3,"scientific_NPZ_or_model_read":False,"new_updates_during_generation":0,"new_deployment_cost_measured":False,"new_E_only_training_updates":1200,"reused_joint_training_updates":1200,"independent_per_update_D_bytes_replayed":False,"protected_TEST_accessed":False}
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as handle: json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":"COMPLETE","manifest":str(output/"MANIFEST.json"),"images":len(images)}),flush=True); return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("train","audit","output"): parser.add_argument("--"+key,required=True)
    args=vars(parser.parse_args()); output=args.pop("output"); generate(args,output)
