"""Prepare/render N4 new-three aggregates only after actual audit and report.

Default does no input/output or renderer import. ROOT alone may execute with
the exact plot source, complete report bindings and actual PASS audit SHA.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import types

NAME="priority_valid_minima_20261016_v1"
N3="priority_nested_cost_20261016_v1"
PLAN_SHA="6f7baaad7200c7c6601bce5a0cb6f6fdff7c3497e3a26b36906408c2844ae0f3"
AUDITOR_SHA="933883a052b02e6a41ccdadaa5ec45e80b977dbfaada007607ac1093f518dd04"
BUILDER_SHA="8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
PCA_CSV_SHA="e3d35a0ae6dbac212a4e5b7cfa589f71168ee1b990cbd0916b57e10d8a22a4c9"
PCA_AUDIT_SHA="2d00f17d0520d6cd9482d304f4b403b5cf7da613d5faec9a9cd92483ac090c5b"
POST_REL="priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/BEIJING_MINIMA_POST_ROOT_v1/RESULT.json"
DEADLINE,MEMORY_LIMIT=1792222550,1024**3
ROLES=("B2","LAG_CAUSAL","LAG_BIDIR")
IDS=("postT_cpu_beijing_E1_B2_k6_s92601_required_v1","postT_cpu_beijing_E3_LAG_CAUSAL_k6_s92601_required_v1","postT_cpu_beijing_E3_LAG_BIDIR_k6_s92601_required_v1")
LABELS=("B2","Causal lag","Bidir lag")
COLORS=("#009E73","#0072B2","#CC79A7")
CAPTION=("같은 K6 · frozen F · forecast MSE 학습 · 구조별 parameter capacity와 선택 refit 길이는 다를 수 있음.\n"
    "한 seed · 반복 개발 V · 제한 학습: 수렴·독립 일반화 미확정. Bidir는 과거 창 내부 양방향이며 시점별 causal 검증이 아님.")


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def digest(blob):return hashlib.sha256(blob).hexdigest()


def number(value):
    result=float(value)
    if not math.isfinite(result) or result<0:raise ValueError("Finite nonnegative measured aggregate required")
    return result


def prepare_data(rows,audit,report):
    if (audit.get("schema")!="INDEPENDENT_N4_BEIJING_AUDIT_v1" or audit.get("status")!="PASS" or audit.get("audit_source_sha256")!=AUDITOR_SHA
            or audit.get("plan_sha256")!=PLAN_SHA or audit.get("input_SHA_before_after_equal") is not True
            or audit.get("whole_MASTER_complete") is not False or audit.get("independent_generalization_claimed") is not False
            or report.get("status")!="REPORT_FROM_COMPLETED_THREE_MINIMA" or report.get("new_complete_conditions")!=3
            or report.get("source48_preserved") is not True or report.get("input_SHA_before_after_equal") is not True or report.get("whole_MASTER_complete") is not False):
        raise ValueError("Actual N4 PASS audit and complete source-preserved report required")
    totals=audit.get("totals",{})
    if any(totals.get(key)!=value for key,value in dict(accuracy_conditions=3,public_stat_rows=4257,observed_labels_per_condition=20539,unique_V_observed_labels=20539,
            trial_rows=300,cost_conditions=6,summary_groups=12,summary_numeric_fields_checked=60).items()):raise ValueError("Whole three accuracy/six actual cost audit required")
    selected=[row for row in rows if row.get("role") in ROLES]
    if len(selected)!=3 or tuple(row.get("role") for row in selected)!=ROLES or tuple(row.get("run_id") for row in selected)!=IDS:raise ValueError("Only exact new B2/causal/bidir rows in report order allowed")
    if len(audit.get("accuracy",[]))!=3 or len(audit.get("cost_conditions",[]))!=6:raise ValueError("Exact three/six audit identities required")
    p90={1:True,8:True};count_bindings=[]
    for row,role,identifier in zip(selected,ROLES,IDS):
        expected=dict(K="6",accuracy_device="cpu",comparison_scope="NEW_CPU_V43_FP64_CPU6",training_device="cpu",search_max_epoch="3",
            cost_device="cpu",cost_scope="NEW_CPU2_FULL_PHYSICAL_PROTOCOL",observed_labels="20539",V_origins="43")
        if any(str(row.get(key))!=value for key,value in expected.items()):raise ValueError("Same K6/CPU/fullV/whole-cost report scope required")
        accuracy=[entry for entry in audit["accuracy"] if entry.get("run_id")==identifier and entry.get("family")==role]
        if len(accuracy)!=1:raise ValueError("Actual exact per-method accuracy audit missing or ambiguous")
        accuracy=accuracy[0];count_bindings.append(accuracy["band_count_binding_sha256"])
        if accuracy.get("n_observed")!=20539 or accuracy.get("public_stats_rows")!=1419:raise ValueError("Same full V43/observed cohort required")
        for metric in ("MSE","MAE"):
            if not math.isclose(number(row[metric]),number(accuracy[metric.lower()]),abs_tol=1e-9,rel_tol=1e-7):raise ValueError("CSV accuracy differs from actual public-statistic audit")
        for batch in (1,8):
            matches=[entry for entry in audit["cost_conditions"] if entry.get("run_id")==identifier and entry.get("batch")==batch]
            if len(matches)!=1 or matches[0].get("trial_rows")!=50 or matches[0].get("F_calls")!=50 or matches[0].get("F_logical_streams")!=batch*6*50:raise ValueError("Actual50 full-route K6 costs required")
            timed=[entry for entry in matches[0]["summaries"] if entry.get("phase")=="timed"]
            if len(timed)!=1 or timed[0].get("trials")!=20 or timed[0].get("batch_size")!=batch:raise ValueError("Actual ordinary whole20 timing summary required")
            if number(row[f"B{batch}_median_seconds"])!=number(timed[0]["median_seconds"]):raise ValueError("CSV median differs from independent actual trial audit")
            pvalue=row.get(f"B{batch}_p90_seconds")
            if pvalue in (None,""):p90[batch]=False
            elif number(pvalue)!=number(timed[0]["p90_seconds"]):raise ValueError("Present CSV p90 must match actual independent trial audit")
    if len(set(count_bindings))!=1:raise ValueError("All three observed band-count bindings must be identical")
    for batch in (1,8):
        present=[row.get(f"B{batch}_p90_seconds") not in (None,"") for row in selected]
        if any(present) and not all(present):raise ValueError("Partial per-method p90 disclosure is not a complete panel")
    return dict(rows=selected,p90_confirmed=p90,pca_reference=None)


def pca_reference(rows,audit):
    found=[row for row in rows if row.get("K")=="6"]
    if audit.get("schema")!="INDEPENDENT_NESTED_COST_REVIEW_v1" or audit.get("status")!="PASS" or len(found)!=1:raise ValueError("Exact existing N3 PCA6 audit required")
    row=found[0];expected=dict(dataset="beijing",run_id="postT_cpu_beijing_E4_NESTED_PCA_k6_s92601_v1",L="512",H="48",C="11",V_origins="43",observed_labels="20539",accuracy_device="cpu",CPU_threads="2")
    if any(row.get(key)!=value for key,value in expected.items()):raise ValueError("PCA reference must share original CPU2 V43 cohort")
    data=audit["data_contract"]
    if data.get("A60_fit_end")!=12622 or data.get("A60_scaler_sha256")!="3348bcf1fa76d382f23cf20597fa90de6e21dac587d4ca9f6883f235a10f93fb":raise ValueError("Same original PCA A60 scale required")
    matches=[entry for entry in audit["reused_accuracy"] if entry.get("latent_k")==6 and entry.get("run_id")==row["run_id"]]
    if len(matches)!=1 or any(number(row[key])!=number(matches[0][key.lower()]) for key in ("MSE","MAE")):raise ValueError("Exact reused PCA6 fullV accuracy required")
    return {key:number(row[key]) for key in ("MSE","MAE")}


def draw_panels(axes,data):
    rows=data["rows"];artists=[]
    panels=(("MSE","A   정규화 MSE","MSE",None),("MAE","B   정규화 MAE","MAE",None),
        ("B1_median_seconds","C   CPU2 B1 whole latency","초 / B1 batch",1),("B8_median_seconds","D   CPU2 B8 whole latency","초 / B8 batch",8))
    for ax,(column,title,ylabel,batch) in zip(axes,panels):
        values=[number(row[column]) for row in rows];bars=ax.bar(range(3),values,width=.54,color=COLORS,zorder=3)
        for bar,hatch in zip(bars,("","/","..")):bar.set_hatch(hatch)
        artists.append(dict(column=column,values=[float(bar.get_height()) for bar in bars]))
        top=max(max(values),1e-12)
        if batch and data["p90_confirmed"][batch]:
            key=f"B{batch}_p90_seconds";upper=[number(row[key]) for row in rows]
            if any(p<value for p,value in zip(upper,values)):raise ValueError("p90 cannot be below its median")
            ax.vlines(range(3),values,upper,color="#263238",linewidth=1.1,zorder=4)
            line,=ax.plot(range(3),upper,linestyle="none",marker="_",markersize=12,color="#263238",zorder=5)
            artists.append(dict(column=key,values=[float(value) for value in line.get_ydata()]));top=max(upper)
        if batch is None and data["pca_reference"] is not None:
            reference=data["pca_reference"][column];line=ax.axhline(reference,color="#59666D",linestyle="--",linewidth=1.2,label="PCA6 참고: 다른 D/inverse",zorder=2)
            artists.append(dict(reference=column,values=[float(value) for value in line.get_ydata()]));top=max(top,reference);ax.legend(loc="upper right",frameon=False,fontsize=8)
        for index,value in enumerate(values):ax.text(index,value+top*.035,f"{value:.4g}",ha="center",va="bottom",fontsize=10,fontweight="bold",color="#263238")
        ax.set_title(title,loc="left",fontweight="bold",fontsize=12,pad=10);ax.set_ylabel(ylabel);ax.set_xticks(range(3),LABELS)
        ax.set_ylim(0,top*1.25);ax.grid(axis="y",color="#E4E9ED",linewidth=.7,zorder=0);ax.set_axisbelow(True);ax.tick_params(axis="both",length=0)
    for artist in artists:
        expected=([data["pca_reference"][artist["reference"]]]*2 if "reference" in artist else [number(row[artist["column"]]) for row in rows])
        if artist["values"]!=expected:raise ValueError("Actual artist values differ from exact selected CSV/reference values")
    return artists


class ResourceCheck:
    def __enter__(self):
        import psutil
        self.process=psutil.Process();self.peak=self.os_peak=0;self.stop=threading.Event();self.failed=False;self.check()
        def sample():
            while not self.stop.wait(.05):
                try:self.check()
                except RuntimeError:self.failed=True;break
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def check(self):
        memory=self.process.memory_info();self.peak=max(self.peak,memory.rss);self.os_peak=max(self.os_peak,getattr(memory,"peak_wset",memory.rss))
        if time.time()>=DEADLINE or max(self.peak,self.os_peak)>=MEMORY_LIMIT or self.failed:raise RuntimeError("Original deadline or own1GiB figure limit reached")
    def __exit__(self,*error):self.stop.set();self.thread.join(timeout=1)


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_path=Path(os.environ["WINDIR"])/"Fonts/malgun.ttf"
    if not font_path.is_file():raise FileNotFoundError("Existing Malgun font required; no installation performed")
    font=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family":font,"font.size":10,"axes.unicode_minus":False,"svg.fonttype":"none","svg.hashsalt":"N4_BEIJING_MINIMA_v1",
        "axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})
    fig,axes=plt.subplots(2,2,figsize=(12,8.4));fig.subplots_adjust(left=.08,right=.97,bottom=.23,top=.82,wspace=.27,hspace=.48)
    try:
        fig.suptitle("Beijing K6: 지연 encoder의 추가 가치와 전체 추론 비용",x=.08,y=.965,ha="left",fontsize=16,fontweight="bold",color="#172B3A")
        fig.text(.08,.915,"같은 CPU · C11 / L512 / H48 · 개발 V43 · 관측 labels20,539 · seed92601 · 모든 축0부터",fontsize=9,color="#455A64")
        fig.text(.08,.873,"오차 = 전체 V / 비용 = CPU2 physical→P/E/F/D→FP64 inverse · B8은8개 batch 전체 시간",fontsize=9,color="#455A64")
        artists=draw_panels(list(axes.flat),data)
        p90_text="실제 확정 p90만 위쪽 표시; p90은 신뢰구간이 아님." if any(data["p90_confirmed"].values()) else "p90 미확정: 추가 표시 없음."
        fig.text(.08,.165,"비용 막대 = ordinary whole20 median. "+p90_text,fontsize=8.8)
        fig.text(.08,.126,CAPTION,fontsize=8.4,linespacing=1.65,color="#536471")
        if data["pca_reference"] is not None:fig.text(.08,.052,"PCA6 점선은 기존 정확도 참고만: 고정 basis transpose+center가 B2의 학습 lag D/current anchor와 다름.",fontsize=8.5,color="#536471")
        svg,png=io.BytesIO(),io.BytesIO();fig.savefig(svg,format="svg",metadata={"Date":None,"Creator":"N4 audited aggregate visualization"});fig.savefig(png,format="png",dpi=145,metadata={"Software":"N4 audited aggregate visualization"})
        clean=("\n".join(line.rstrip() for line in svg.getvalue().decode("utf-8").splitlines())+"\n").encode()
        return clean,png.getvalue(),artists,font,sha(font_path)
    finally:plt.close(fig)


def execute(public,cache,plot_sha,report_sha,audit_path,audit_sha,include_pca=False):
    public,cache=Path(public).resolve(),Path(cache).resolve();scope=public/NAME
    svg_path=scope/"figures/BEIJING_MINIMA_v1.svg";binding=scope/"FIGURE_BINDINGS_v1.json";md_path=scope/"VISUALIZATION_KO_v1.md";preview=cache/NAME/"visualization/BEIJING_MINIMA_v1.png"
    if any(path.exists() for path in (svg_path,binding,md_path,preview)):raise FileExistsError("Preserve existing outputs; new SVG/PNG/bindings/MD required")
    if time.time()>=DEADLINE:raise RuntimeError("Original visualization deadline reached")
    if Path(__file__).resolve()!=scope/"plot_beijing.py" or sha(__file__)!=plot_sha:raise ValueError("ROOT-pinned current plot source required")
    if any(os.environ.get(key)!="1" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")):raise ValueError("Explicit CPU/BLAS1 environment required")
    helper_path=scope/"build_beijing_report.py"
    if sha(helper_path)!=BUILDER_SHA:raise ValueError("Pinned stdlib input/cohort helper changed")
    helper=types.ModuleType("n4_plot_evidence_helpers");helper.__file__=str(helper_path);exec(compile(helper_path.read_text(encoding="utf-8"),str(helper_path),"exec"),helper.__dict__)
    inputs=helper.Inputs(public,cache);inputs.capture("public",NAME+"/plot_beijing.py",plot_sha);inputs.capture("public",NAME+"/build_beijing_report.py",BUILDER_SHA)
    inputs.capture("public",NAME+"/independent_beijing_audit.py",AUDITOR_SHA);inputs.capture("public",NAME+"/PLAN.json",PLAN_SHA)
    report=inputs.json("public",NAME+"/REPORT_INPUT_BINDINGS_v1.json",report_sha)
    audit_path=Path(audit_path).resolve()
    if not audit_path.is_relative_to(scope):raise ValueError("Explicit actual audit JSON must be inside N4 namespace")
    audit_relative=audit_path.relative_to(public).as_posix();audit=inputs.json("public",audit_relative,audit_sha)
    for row in report.get("inputs",[]):inputs.capture(row["root"],row["relative_path"],row["sha256"])
    for row in audit.get("inputs",[]):inputs.capture(row["root"],row["relative_path"],row["sha256"])
    post=inputs.json("cache",POST_REL,audit["post_result_sha256"])
    if (post.get("status")!="COMPLETE_THREE_VALIDATIONS_AND_SIX_COSTS" or len(post.get("steps",[]))!=12 or post.get("skipped")!=[]
            or not any(row.get("root")=="cache" and row.get("relative_path")==POST_REL and row.get("sha256")==audit["post_result_sha256"] for row in report.get("inputs",[]))):raise ValueError("Same actual complete POST must bind report and audit")
    csv_relative=NAME+"/RESULTS_WITH_COST_v1.csv";recorded=[row for row in report.get("outputs",[]) if row.get("relative_path")==csv_relative]
    if len(recorded)!=1:raise ValueError("Exactly one report-bound original-precision CSV required")
    csv_path=inputs.capture("public",csv_relative,recorded[0]["sha256"])
    with csv_path.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    data=prepare_data(rows,audit,report)
    for identifier in IDS:
        matches=[row for row in audit["accuracy"] if row["run_id"]==identifier]
        if len(matches)!=1:raise ValueError("Exact audited cohort identity required")
        manifest=inputs.json("public",NAME+"/predictions/"+identifier+"/V_select/PREDICTION_MANIFEST.json",matches[0]["manifest_sha256"]);helper.validate_cohort(manifest)
    if include_pca:
        pca_path=inputs.capture("public",N3+"/RESULTS_WITH_COST_v1.csv",PCA_CSV_SHA)
        with pca_path.open(encoding="utf-8-sig",newline="") as stream:pca_rows=list(csv.DictReader(stream))
        data["pca_reference"]=pca_reference(pca_rows,inputs.json("public",N3+"/INDEPENDENT_COST_REVIEW_ROOT_v1.json",PCA_AUDIT_SHA))
    if not inputs.preserved():raise ValueError("Figure inputs changed before rendering")
    started=time.perf_counter()
    with ResourceCheck() as resources:
        svg,png,artists,font,font_sha=render(data)
        if not inputs.preserved():raise ValueError("Figure inputs changed during rendering")
        resources.check()
    md=("# N4 Beijing K6 시각자료\n\n![B2와 지연 encoder의 정확도 및 실제 전체 비용](figures/BEIJING_MINIMA_v1.svg)\n\n"
        "[원정밀도 CSV](RESULTS_WITH_COST_v1.csv) · [실험 보고](RESULTS_KO_v1.md) · [독립 감사]("+audit_path.relative_to(scope).as_posix()+") · [그림 결속](FIGURE_BINDINGS_v1.json) · [코드](plot_beijing.py)\n\n"
        "새 B2/causal/bidir 세 조건만 같은 순서로 표시했다. 모든 축은0부터이며 표시 숫자만 반올림하고 원 CSV와 실제 artist 값은 바꾸지 않았다. "
        "지연시간은 CPU2 ordinary whole20 median의 **초/batch**이고 B8은8개 batch 전체다. 확정된 p90만 위쪽에 표시하며 신뢰구간이나 seed 분산이 아니다.\n\n"
        +CAPTION.replace("\n"," ")+"\n\n"
        +("PCA6 점선은 exact N3 CPU2 동일 V43/A60의 기존 MSE·MAE 참고다. 고정 basis transpose+center와 학습 lag D/current anchor가 달라 동일 decoder 비교가 아니다. PCA 비용은 이 그림에 섞지 않았다.\n\n" if include_pca else "PCA 참고선은 선택하지 않았다.\n\n")
        +"Raw/G/Ridge의 장치·비용이 다른 기존 행은 위 실험 보고 링크로만 제공한다. 자동 모델 순위, 수렴·독립 일반화 또는 방법론 성공을 주장하지 않는다. "
        "새 통계·trial 재집계, 원자료/Task/NPZ/CP/model 해석, 학습·추론·비용 재측정·프로세스 제어·Git은0이다. SVG는 게시용, PNG는 private preview다.\n")
    svg_path.parent.mkdir(exist_ok=True);preview.parent.mkdir(parents=True,exist_ok=True)
    for path,blob in ((svg_path,svg),(preview,png),(md_path,md.encode())):
        with path.open("xb") as stream:stream.write(blob)
    evidence=dict(schema="N4_BEIJING_FIGURE_BINDINGS_v1",status="PASS_AUDITED_NEW_THREE_AGGREGATE_VISUALIZATION",plot_source_sha256=plot_sha,actual_audit_sha256=audit_sha,report_bindings_sha256=report_sha,
        input_SHA_before_after_equal=True,inputs=[dict(root=root,relative_path=name,sha256=value) for (root,name),value in sorted(inputs.files.items())],
        outputs=[dict(relative_path=path.relative_to(scope).as_posix(),sha256=digest(blob),bytes=len(blob)) for path,blob in ((svg_path,svg),(md_path,md.encode()))],
        private_preview=dict(published=False,cache_relative_path=preview.relative_to(cache).as_posix(),sha256=digest(png),bytes=len(png)),
        artist_values_from_unchanged_CSV=artists,CSV_precision_preserved=True,p90_confirmed=data["p90_confirmed"],PCA6_accuracy_reference_included=include_pca,
        font=font,font_file_sha256=font_sha,CPU_BLAS_threads=1,own_RSS_limit_bytes=MEMORY_LIMIT,sampled_own_peak_RSS_bytes=resources.peak,own_OS_peak_working_set_bytes=resources.os_peak,
        deadline_unix=DEADLINE,elapsed_seconds=time.perf_counter()-started,actual_renderer_executed=True,new_accuracy_cost_or_model_measurements=0,
        Torch_imported="torch" in sys.modules,model_Task_NPZ_raw_T_training_inference_cost_process_control_network_Git_operations=0,whole_MASTER_complete=False,independent_generalization_claimed=False,
        self_hash_policy="Bindings excludes itself; exact source/input/SVG/MD/preview hashes are recorded")
    with binding.open("x",encoding="utf-8",newline="\n") as stream:json.dump(evidence,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    return dict(status=evidence["status"],public_files_written=3,private_previews_written=1,artist_groups=len(artists),font=font)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--root-triggered",action="store_true");parser.add_argument("--include-pca6",action="store_true")
    for key in ("public-root","cache-root","plot-sha","report-binding-sha","audit","audit-sha"):parser.add_argument("--"+key)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,actual_render=0,numpy_imported="numpy" in sys.modules,matplotlib_imported="matplotlib" in sys.modules)));return 0
    if not args.root_triggered or any(getattr(args,key) is None for key in ("public_root","cache_root","plot_sha","report_binding_sha","audit","audit_sha")):
        parser.error("Explicit ROOT trigger, roots/current plot SHA/complete report-binding SHA/actual PASS audit path+SHA required")
    print(json.dumps(execute(args.public_root,args.cache_root,args.plot_sha,args.report_binding_sha,args.audit,args.audit_sha,args.include_pca6),ensure_ascii=False));return 0


if __name__=="__main__":raise SystemExit(main())
