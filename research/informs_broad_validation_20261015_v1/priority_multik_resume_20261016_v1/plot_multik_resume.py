"""Render only a completed multi-K report CSV and its exact bindings.

Preparation defaults to zero I/O. No model, array, statistic or trial loading.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time

NAME="priority_multik_resume_20261016_v1"
COMMON="priority_valid_minima_20261016_v1"
BUILDER_SHA="d616ce65839393e0fdb70bb3a98e557d29870ef62428204436a66429f5516f37"
AUDITOR_SHA="0c22445d86e75610d78caeda0dc645c0fa7c09d3ed46f6a2b9ed44878ea25146"
INPUT_HELPER_SHA="8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
STYLE_SHA="50e25501857a9c3aad0c9f9902397c00075025d1b3556778539b0e400adaf47b"
COLUMNS=("MSE","MAE","B1_median_seconds","B8_median_seconds")
SERIES=(("shared","SHARED_MULTI_K","NEW_TRAINED_PARENT_PREFIX","Shared: 새 부모",(3,6,11),"#0072B2","o","-"),
    ("max_only","MAX_ONLY_POSTHOC","NEW_TRAINED_PARENT_PREFIX","Max-only: 새 부모/prefix",(3,6,11),"#D55E00","s","--"),
    ("pca","NESTED_PCA_REFERENCE","REUSED_FIXED_NESTED_PCA","Nested PCA: 재사용/다른 D",(3,6,11),"#687D8D","^",":"),
    ("independent","INDEPENDENT_K6_REFERENCE","REUSED_INDEPENDENT_K6","Independent K6: 재사용",(6,),"#CC79A7","D","none"))
CAPTION=("한 seed·반복 개발 V·제한 학습: 수렴·독립 일반화 미확정. 선은 관측 K를 연결하며 중간 K를 측정한 것이 아님.\n"
    "새 shared/max는 부모2개/출력6개. PCA는 다른 D, independent는 K6 한 점뿐. 전체 E4/56/MASTER 미완료.")


def sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):result.update(block)
    return result.hexdigest()


def number(value):
    result=float(value)
    if not math.isfinite(result) or result<0:raise ValueError("Finite nonnegative report aggregate required")
    return result


def digest(value):return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def expected_rows():
    rows={}
    for kind,role in (("shared","ORDERED_MULTI_K"),("max_only","MAXK_PCA_TRAINED")):
        for k in (3,6,11):
            tag=role if k==11 else ("SHARED_PREFIX" if kind=="shared" else "POSTHOC_PREFIX")
            rows[f"postT_cpu_beijing_E4_{tag}_k{k}_s92601_v1"]=("NEW_TRAINED_PARENT_PREFIX","SHARED_MULTI_K" if kind=="shared" else "MAX_ONLY_POSTHOC",k)
    for k in (3,6,11):rows[f"postT_cpu_beijing_E4_NESTED_PCA_k{k}_s92601_v1"]=("REUSED_FIXED_NESTED_PCA","NESTED_PCA_REFERENCE",k)
    rows["postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1"]=("REUSED_INDEPENDENT_K6","INDEPENDENT_K6_REFERENCE",6)
    return rows


def prepare_data(rows,report,csv_sha,audit_sha):
    expected=dict(schema="MULTIK_RESUME_REPORT_BINDINGS_v1",status="FROM_ACTUAL_MULTIK_AUDIT_PASS_AND_PINNED_REUSED_REFERENCES",
        audit_sha256=audit_sha,auditor_source_sha256=AUDITOR_SHA,builder_source_sha256=BUILDER_SHA,input_SHA_before_after_equal=True,
        source48_preserved=True,resumed_parent_refits=2,new_registered_conditions=0,new_prefix_accuracy_outputs=6,new_cost_conditions=12,
        reused_fixed_nested_PCA_rows=3,reused_independent_K6_rows=1,unlogged_native_cost_fields_preserved_null=True,
        whole_E4_complete=False,whole_MASTER_complete=False,independent_generalization_claimed=False)
    if not all(digest(v) for v in (csv_sha,audit_sha)) or any(report.get(k)!=v for k,v in expected.items()):raise ValueError("Exact actual complete source-preserved multi-K report bindings required")
    outputs=report.get("outputs",[]);csv_relative=NAME+"/MULTIK_RESUME_RESULTS_v1.csv"
    if (len(outputs)!=2 or {r.get("relative_path") for r in outputs}!={csv_relative,NAME+"/MULTIK_RESUME_KO_v1.md"}
            or any(not digest(r.get("sha256")) for r in outputs)
            or next(r for r in outputs if r["relative_path"]==csv_relative)["sha256"]!=csv_sha):raise ValueError("Completed MD and exact original-precision CSV must be report-bound")
    audit_relative=report.get("audit_relative_path","")
    if not isinstance(audit_relative,str) or not audit_relative.startswith(NAME+"/") or Path(audit_relative).is_absolute() or ".." in Path(audit_relative).parts or Path(audit_relative).suffix!=".json":raise ValueError("Root-relative N7 audit reference required; no audit numeric reread")
    wanted=expected_rows()
    if len(rows)!=10 or {r.get("run_id") for r in rows}!=set(wanted):raise ValueError("Only six current aliases, three reused PCA and one reused independent K6 allowed")
    for row in rows:
        scope,role,k=wanted[row["run_id"]];reused=scope!="NEW_TRAINED_PARENT_PREFIX"
        if (row.get("row_scope"),row.get("role"),str(row.get("K")))!=(scope,role,str(k)) or any(str(row.get(key))!=v for key,v in dict(CPU_threads="2",V_origins="43",observed_labels="20539",accuracy_reused=str(reused),cost_reused=str(reused)).items()):raise ValueError("Exact CPU2/V43/reference/new-method row scope required")
        for column in COLUMNS:number(row[column])
        if row.get("F_resident_bytes") not in (None,"") or row.get("F_resident_bytes_scope")!="NOT_SEPARATELY_RECORDED":raise ValueError("Unmeasured F-only resident bytes must remain NA")
        if not reused:
            for batch in (1,8):
                if (any(row.get(f"B{batch}_"+key) not in (None,"") for key in ("raw_receipt_total_F_counters","receipt_validation_proof_SHA_map","physical_output_dtype_from_receipt"))
                        or row.get(f"B{batch}_raw_receipt_field_scope")!="NOT_RECORDED_BY_ORIGINAL_N7_RECEIPT"):raise ValueError("Native missing receipt fields cannot become inferred zeros or totals")
    series=[]
    for key,role,scope,label,ks,color,marker,linestyle in SERIES:
        selected=sorted((r for r in rows if r["role"]==role),key=lambda r:int(r["K"]))
        if tuple(int(r["K"]) for r in selected)!=ks:raise ValueError("Only registered measured K positions allowed")
        series.append(dict(key=key,role=role,row_scope=scope,label=label,rows=selected,color=color,marker=marker,linestyle=linestyle))
    return dict(series=series,audit_relative_path=audit_relative)


def validate_artists(artists,data):
    expected={(column,s["key"]):(s,[int(r["K"]) for r in s["rows"]],[r[column] for r in s["rows"]]) for column in COLUMNS for s in data["series"]}
    if len(artists)!=16 or {(r.get("column"),r.get("series")) for r in artists}!=set(expected):raise ValueError("Exactly four panels and four measured series per panel required")
    for row in artists:
        series,ks,values=expected[(row["column"],row["series"])]
        if (row.get("K")!=ks or row.get("values")!=[number(v) for v in values] or row.get("source_values")!=values
                or row.get("run_ids")!=[r["run_id"] for r in series["rows"]] or row.get("row_scope")!=series["row_scope"]):raise ValueError("Actual artist positions/values must equal unchanged precise CSV rows")


def draw_panels(axes,data):
    panels=(("MSE","A   정규화 MSE","MSE"),("MAE","B   정규화 MAE","MAE"),
        ("B1_median_seconds","C   CPU2 B1 whole median","초 / B1 batch"),("B8_median_seconds","D   CPU2 B8 whole median","초 / B8 batch"))
    if len(axes)!=4:raise ValueError("Exactly four axes required")
    artists=[]
    for ax,(column,title,ylabel) in zip(axes,panels):
        maximum=1e-12
        for series in data["series"]:
            rows=series["rows"];ks=[int(r["K"]) for r in rows];values=[number(r[column]) for r in rows];maximum=max(maximum,*values)
            line,=ax.plot(ks,values,color=series["color"],marker=series["marker"],linestyle=series["linestyle"],linewidth=1.7,markersize=6,label=series["label"],zorder=4 if series["key"]=="independent" else 3)
            artists.append(dict(column=column,series=series["key"],row_scope=series["row_scope"],run_ids=[r["run_id"] for r in rows],K=[int(v) for v in line.get_xdata()],values=[float(v) for v in line.get_ydata()],source_values=[r[column] for r in rows]))
        ax.set_title(title,loc="left",fontweight="bold",fontsize=12,pad=10);ax.set_xlabel("실제 F 입력 잠재 시계열 K");ax.set_ylabel(ylabel)
        ax.set_xticks([3,6,11]);ax.set_xlim(2,12);ax.set_ylim(0,maximum*1.18);ax.grid(axis="y",color="#E4E9ED",linewidth=.7,zorder=0);ax.set_axisbelow(True);ax.tick_params(axis="both",length=0)
    validate_artists(artists,data);return artists


def module(path,expected,label):
    if sha(path)!=expected:raise ValueError("Pinned current source/helper changed: "+Path(path).name)
    spec=importlib.util.spec_from_file_location(label,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def render(data,csv_sha,audit_sha):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_path=Path(os.environ["WINDIR"])/"Fonts/malgun.ttf"
    if not font_path.is_file():raise FileNotFoundError("Existing Malgun font required; no installation")
    font=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family":font,"font.size":10,"axes.unicode_minus":False,"svg.fonttype":"none","svg.hashsalt":"MULTIK_RESUME_v1","axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})
    fig,axes=plt.subplots(2,2,figsize=(12.7,9.7));fig.subplots_adjust(left=.08,right=.97,bottom=.29,top=.78,wspace=.28,hspace=.50)
    try:
        fig.suptitle("Beijing: shared / max-only 부모의 K별 오차·전체 추론 비용",x=.08,y=.966,ha="left",fontsize=15,fontweight="bold",color="#172B3A")
        fig.text(.08,.927,"C11 / L512 / H48 · CPU2 · 개발 V43 / observed labels20,539 · seed92601 · y축0부터",fontsize=9,color="#455A64")
        fig.text(.08,.891,"실제 새 alias6 + 재사용 fixed nested PCA3 / independent K6 한 점 · 모든 방법에 frozen Bolt 포함",fontsize=8.7,color="#455A64")
        artists=draw_panels(list(axes.flat),data)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper left",bbox_to_anchor=(.075,.866),ncol=2,frameon=False,fontsize=8.6)
        fig.text(.08,.235,"비용 = warm ordinary whole20 median의 초/batch. B8은8개 입력 한 batch 전체이며 cold F/Route load는 제외.",fontsize=8.3)
        fig.text(.08,.200,CAPTION,fontsize=8.1,linespacing=1.7,color="#536471")
        fig.text(.08,.134,"shared/max의 최대 K11 구조·PCA 초기화는 같아도 K별 학습 노출/F 연산·총 compute는 다름. 순수 공유 학습 효과가 아님.",fontsize=8.1,color="#536471")
        fig.text(.08,.104,"원 independent K11은 identity 초기화이며 이번 미실행. K 전환·저장량·메모리 이득은 측정하지 않음; F-only resident=NA.",fontsize=8.1,color="#536471")
        fig.text(.08,.072,"CSV SHA "+csv_sha[:12]+" · 실제 감사 SHA(보고 결속) "+audit_sha[:12]+" · 표시/artist 값은 원 CSV 정밀도 보존",fontsize=8.0,color="#536471")
        svg,png=io.BytesIO(),io.BytesIO();fig.savefig(svg,format="svg",metadata={"Date":None,"Creator":"Audited multi-K report aggregate visualization"});fig.savefig(png,format="png",dpi=145,metadata={"Software":"Audited multi-K report aggregate visualization"})
        clean=("\n".join(line.rstrip() for line in svg.getvalue().decode("utf-8").splitlines())+"\n").encode()
        return clean,png.getvalue(),artists,font,sha(font_path)
    finally:plt.close(fig)


def execute(public,cache,plot_sha,report_sha,csv_sha,audit_sha):
    public,cache=Path(public).resolve(),Path(cache).resolve();scope=public/NAME
    svg_path=scope/"figures/MULTIK_RESUME_v1.svg";preview=cache/NAME/"visualization/MULTIK_RESUME_v1.png"
    md_path=scope/"MULTIK_RESUME_VISUALIZATION_KO_v1.md";binding_path=scope/"MULTIK_RESUME_FIGURE_BINDINGS_v1.json"
    if any(p.exists() for p in (svg_path,preview,md_path,binding_path)):raise FileExistsError("Existing multi-K visualization preserved; CreateNew only")
    if Path(__file__).resolve()!=scope/"plot_multik_resume.py" or sha(__file__)!=plot_sha or not all(digest(v) for v in (plot_sha,report_sha,csv_sha,audit_sha)):raise ValueError("ROOT current source and exact actual report/CSV/audit SHA required")
    if any(os.environ.get(key)!="1" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")):raise ValueError("Explicit CPU/BLAS1 renderer environment required")
    helper=module(public/COMMON/"build_beijing_report.py",INPUT_HELPER_SHA,"multik_plot_inputs")
    style=module(public/COMMON/"plot_beijing.py",STYLE_SHA,"multik_plot_resource_scope")
    if time.time()>=style.DEADLINE:raise RuntimeError("Original worker deadline reached")
    inputs=helper.Inputs(public,cache)
    for name,want in ((NAME+"/plot_multik_resume.py",plot_sha),(NAME+"/build_multik_resume_report.py",BUILDER_SHA),(NAME+"/audit_multik_resume.py",AUDITOR_SHA),(COMMON+"/build_beijing_report.py",INPUT_HELPER_SHA),(COMMON+"/plot_beijing.py",STYLE_SHA)):inputs.capture("public",name,want)
    report=inputs.json("public",NAME+"/MULTIK_RESUME_BINDINGS_v1.json",report_sha)
    inputs.capture("public",NAME+"/MULTIK_RESUME_RESULTS_v1.csv",csv_sha)
    rows=inputs.csv("public",NAME+"/MULTIK_RESUME_RESULTS_v1.csv");data=prepare_data(rows,report,csv_sha,audit_sha)
    if not inputs.preserved():raise ValueError("Exact report inputs changed before renderer")
    started=time.perf_counter()
    with style.ResourceCheck() as resources:
        svg,png,artists,font,font_sha=render(data,csv_sha,audit_sha)
        if not inputs.preserved():raise ValueError("Exact report inputs changed during renderer")
        resources.check()
    audit_link=data["audit_relative_path"][len(NAME)+1:]
    md=("# Beijing multi-K 시각자료\n\n![shared/max-only의 실제 K별 오차 및 전체 추론 비용](figures/MULTIK_RESUME_v1.svg)\n\n"
        "[실험 보고](MULTIK_RESUME_KO_v1.md) · [원정밀도 CSV](MULTIK_RESUME_RESULTS_v1.csv) · [보고 결속](MULTIK_RESUME_BINDINGS_v1.json) · [원 감사]("+audit_link+") · [그림 결속](MULTIK_RESUME_FIGURE_BINDINGS_v1.json)\n\n"
        "같은 Beijing C11/L512/H48·seed92601·CPU2·반복 개발 V43/20539 관측 라벨이다. shared와 max-only는 이번 저장점 재개 부모2개에서 나온6개 출력이며, fixed nested PCA3과 independent K6 한 점은 기존 결과를 재사용했다. 독립 K3/K11은 미실행이다. 선은 실제 K3/6/11 관측값을 연결할 뿐 중간 K를 측정하거나 보간 성능을 입증하지 않는다. 모든 방법은 Bolt를 포함한다.\n\n"
        "네 패널은 원 CSV의 normalized channel-macro MSE·MAE, CPU2 B1/B8 ordinary whole20 median을 그대로 복사한다. y축0부터이고 CI·seed 분산·유의성·p90는 추가하지 않았다. B8은8개 입력 한 batch의 전체 초 단위 시간이며8로 나누지 않는다. warm whole 비용은 physical history/mask부터 원 C 미래·inverse/CPU 반환까지이고 기존 F pool cold load/Route loading은 제외한다.\n\n"
        "고정 PCA는 basis transpose 복원으로 학습 가능한 lag7 D/current anchor와 다르다. shared/max는 최대 K11 부모 구조와 PCA 초기화를 공유하지만 K별 감독·노출·F 연산 배분이 달라 동등 총 compute나 공유 학습만의 순수 효과를 보장하지 않는다. 원 independent K11의 identity 초기화는 shared/max PCA와 다르며 이번 independent K11 결과는 없다. 선택 refit3 endpoint·제한 학습·한 seed와 반복 개발 V이므로 충분 수렴·독립 일반화는 미확정이다.\n\n"
        "search 재구축 호출 시간을 과거 search 전체 wall로 표시하지 않았으며 unknown unsaved progress는 원 보고에 남긴다. 부모 BASE.full_V_completed=false를 변경하지 않는다. K 전환·저장량·메모리 효과는 측정하지 않았고 최대 K11 E/D가 유지된다. F-only resident bytes는 NA이며 전체 Route weights/RSS로 대체하지 않는다. 원 N7 receipt에 미기록인 raw F 합계/proofmap/output dtype도 추정하지 않는다. 전체 E4/56/MASTER 미완료다.\n\n"
        "이 renderer는 완료 보고 CSV와 결속 JSON만 수치 입력으로 사용했다. 감사 JSON·원 통계/trial·원자료·모델/Task/NPZ/CP는 새로 해석하지 않았다. source/helper와 두 보고 입력의 전후 SHA, 실제16 artist/40개 y값을 대사한다. 공개 SVG/MD/결속 JSON과 private PNG를 CreateNew로 만들며 그림 실패는 독립 실험을 중단하는 gate가 아니다. 새 학습·추론·비용 측정·프로세스 제어·Git/network는0이다.\n")
    proof=dict(schema="MULTIK_RESUME_FIGURE_BINDINGS_v1",status="PASS_ACTUAL_REPORT_AGGREGATE_VISUALIZATION",plot_source_sha256=plot_sha,builder_source_sha256=BUILDER_SHA,auditor_source_sha256=AUDITOR_SHA,
        report_bindings_sha256=report_sha,CSV_sha256=csv_sha,audit_sha256_bound_by_report=audit_sha,audit_numeric_reread=False,input_SHA_before_after_equal=True,
        inputs=[dict(root=root,relative_path=name,sha256=want) for (root,name),want in sorted(inputs.files.items())],
        outputs=[dict(relative_path=p.relative_to(scope).as_posix(),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)) for p,blob in ((svg_path,svg),(md_path,md.encode()))],
        private_preview=dict(published=False,cache_relative_path=preview.relative_to(cache).as_posix(),sha256=hashlib.sha256(png).hexdigest(),bytes=len(png)),
        artist_values_from_unchanged_CSV=artists,artist_entries=16,artist_y_scalar_count=40,CSV_precision_preserved=True,actual_renderer_executed=True,
        new_parent_refits=2,new_prefix_outputs=6,reused_fixed_PCA_rows=3,reused_independent_K6_points=1,new_accuracy_or_cost_measurements=0,
        p90_CI_or_seed_variance_added=False,transition_storage_or_memory_benefit_claimed=False,font=font,font_file_sha256=font_sha,CPU_BLAS_threads=1,
        own_RSS_limit_bytes=style.MEMORY_LIMIT,sampled_own_peak_RSS_bytes=resources.peak,own_OS_peak_working_set_bytes=resources.os_peak,deadline_unix=style.DEADLINE,
        elapsed_seconds=time.perf_counter()-started,whole_E4_complete=False,whole_MASTER_complete=False,independent_generalization_claimed=False,
        self_hash_policy="Bindings excludes itself",report_first=True,plot_failure_blocks_other_paths=False)
    svg_path.parent.mkdir(exist_ok=True);preview.parent.mkdir(parents=True,exist_ok=True)
    for path,blob in ((svg_path,svg),(preview,png),(md_path,md.encode())):
        with path.open("xb") as stream:stream.write(blob)
    with binding_path.open("x",encoding="utf-8",newline="\n") as stream:json.dump(proof,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    return dict(status=proof["status"],public_files_written=3,private_previews_written=1,artist_entries=16,artist_y_scalar_count=40)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--root-triggered",action="store_true")
    for name in ("public-root","cache-root","plot-sha","report-binding-sha","csv-sha","audit-sha"):parser.add_argument("--"+name)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,actual_render=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,matplotlib_imported="matplotlib" in sys.modules,future_report_bindings_SHA=None,future_CSV_SHA=None,future_actual_audit_SHA=None)));return 0
    if not args.root_triggered or any(getattr(args,key) is None for key in ("public_root","cache_root","plot_sha","report_binding_sha","csv_sha","audit_sha")):parser.error("ROOT exact current source, completed report bindings/CSV/audit SHAs and CreateNew outputs required")
    print(json.dumps(execute(args.public_root,args.cache_root,args.plot_sha,args.report_binding_sha,args.csv_sha,args.audit_sha)));return 0


if __name__=="__main__":raise SystemExit(main())
