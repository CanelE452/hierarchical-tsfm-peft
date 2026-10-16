"""Prepare four core-contrast panels; default0 I/O and lazy renderer imports.

ROOT alone renders after actual complete audit/builder/source SHA binding.
No statistical gzip, NPZ/model/checkpoint parsing or experiment operations.
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
import time
import types

NAME="priority_valid_minima_20261016_v1"
BUILDER_SHA="051497ad62f53cbba93c4fb2096ae634a0e93018476bf2df673402518db687a8"
AUDITOR_SHA="3c2c7a7863572cab4274f7ca1fbd922a2f4466038b7a04955021f4f550d21e75"
STYLE_SHA="50e25501857a9c3aad0c9f9902397c00075025d1b3556778539b0e400adaf47b"
INPUT_HELPER_SHA="8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"
ROLES=("B0_K6","B2_K6","B2_K11")
IDS=("postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1","postT_cpu_beijing_E1_B2_k6_s92601_required_v1","postT_cpu_beijing_E1_B2_k11_s92601_required_v1")
LABELS=("B0 K6","B2 K6 (재사용)","B2 K11")
COLUMNS=("MSE","MAE","B1_median_seconds","B8_median_seconds")
CAPTION=("P/E/D는 미래 forecast MSE 학습 · K6 PCA 초기화 / K11 identity 초기화 · P/K·capacity·선택 refit 길이가 함께 다름.\n"
    "순수 압축 효과로 해석하지 않음. 단일 seed92601 · 반복 개발 V · 제한 학습: 수렴·독립 일반화 미확정.")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def module(path,expected,label):
    if sha(path)!=expected:raise ValueError("Exact reviewed stdlib source changed")
    value=types.ModuleType(label);value.__file__=str(path)
    exec(compile(path.read_text(encoding="utf-8"),str(path),"exec"),value.__dict__)
    return value


def number(value):
    value=float(value)
    if not math.isfinite(value) or value<0:raise ValueError("Finite nonnegative measured aggregate required")
    return value


def prepare_data(rows,audit,report):
    if (audit.get("schema")!="INDEPENDENT_BEIJING_CORE_CONTRAST_v1" or audit.get("status")!="PASS" or audit.get("error") is not None
            or audit.get("audit_source_sha256")!=AUDITOR_SHA or audit.get("input_SHA_before_after_equal") is not True
            or audit.get("whole_MASTER_complete") is not False or audit.get("independent_generalization_claimed") is not False
            or report.get("schema")!="BEIJING_CORE_CONTRAST_REPORT_BINDINGS_v1" or report.get("status")!="FROM_ACTUAL_COMPLETE_CORE_AUDIT"
            or report.get("builder_source_sha256")!=BUILDER_SHA or report.get("auditor_source_sha256")!=AUDITOR_SHA
            or report.get("input_SHA_before_after_equal") is not True or report.get("whole_MASTER_complete") is not False
            or any(report.get(k)!=v for k,v in dict(new_accuracy_conditions=2,reused_accuracy_conditions=1,new_cost_conditions=4,reused_cost_conditions=2).items())):
        raise ValueError("Actual complete source-bound core audit/report required")
    actual=audit.get("rows",[])
    if len(rows)!=3 or len(actual)!=3 or tuple(r.get("role") for r in rows)!=ROLES or tuple(r.get("role") for r in actual)!=ROLES:
        raise ValueError("Exact three report rows in B0K6/B2K6/B2K11 order required")
    metadata=("K","accuracy_device","training_device","CPU_threads","observed_labels","V_origins","F_used","accuracy_reused","search_max_epoch","selected_refit_epoch","search_updates","refit_updates","search_exposures","refit_exposures","training_status","initialization")
    for index,(row,record,identifier,k) in enumerate(zip(rows,actual,IDS,(6,6,11))):
        expected=dict(run_id=identifier,K=k,accuracy_device="cpu",training_device="cpu",CPU_threads=2,observed_labels=20539,V_origins=43,F_used=True,accuracy_reused=index==1,search_max_epoch=3,
            initialization="identity_at_K_equals_C" if k==11 else "A_fitted_centered_PCA")
        if any(str(row.get(key))!=str(value) or record.get(key)!=value for key,value in expected.items()) or any(str(row.get(key))!=str(record.get(key)) for key in metadata):
            raise ValueError("Same CPU2 cohort/K/initialization/reuse/training metadata required")
        for key in COLUMNS:
            if number(row[key])!=number(record[key]):raise ValueError("Exact source CSV/actual audited values differ")
    return dict(rows=rows,p90_confirmed={1:False,8:False},pca_reference=None)


def render(data,style,csv_sha,audit_sha):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_path=Path(os.environ["WINDIR"])/"Fonts/malgun.ttf"
    if not font_path.is_file():raise FileNotFoundError("Existing Malgun font required; no installation")
    font=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family":font,"font.size":10,"axes.unicode_minus":False,"svg.fonttype":"none","svg.hashsalt":"BEIJING_CORE_CONTRAST_v1","axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})
    style.LABELS=LABELS;style.COLORS=("#687D8D","#009E73","#0072B2")
    fig,axes=plt.subplots(2,2,figsize=(11.8,8.2));fig.subplots_adjust(left=.08,right=.97,bottom=.23,top=.80,wspace=.28,hspace=.48)
    try:
        fig.suptitle("Beijing: B2 앞단의 추가 가치와 K6 / K11 경계",x=.08,y=.965,ha="left",fontsize=16,fontweight="bold",color="#172B3A")
        fig.text(.08,.918,"CPU2 · C11 / L512 / H48 · V43 / observed labels20,539 · seed92601 · 모든 축0부터",fontsize=9,color="#455A64")
        fig.text(.08,.877,"모든 경로 frozen Bolt · 오차=전체 V · 지연시간=ordinary whole20 median / batch",fontsize=9,color="#455A64")
        fig.text(.08,.838,"원 CSV SHA "+csv_sha[:12]+" · 실제 PASS audit SHA "+audit_sha[:12],fontsize=8.5,color="#536471")
        artists=style.draw_panels(list(axes.flat),data)
        if len(artists)!=4 or {r["column"] for r in artists}!=set(COLUMNS):raise ValueError("Only four requested median/error artist groups allowed")
        for artist in artists:
            if artist["values"]!=[number(row[artist["column"]]) for row in data["rows"]]:raise ValueError("Actual artist values must equal unchanged source CSV")
        fig.text(.08,.169,"B8 = 8 origins를 담은 한 batch 전체 시간. B2 K6는 기존 완료 결과를 동일 조건으로 재사용.",fontsize=8.7)
        fig.text(.08,.128,CAPTION,fontsize=8.5,linespacing=1.65,color="#536471")
        fig.text(.08,.054,"표시 숫자만 반올림. 원 CSV 정밀도 보존 · 자동 모델 선택/순위 없음 · 전체 MASTER 미완료",fontsize=8.5,color="#536471")
        svg,png=io.BytesIO(),io.BytesIO();fig.savefig(svg,format="svg",metadata={"Date":None,"Creator":"Audited core contrast visualization"});fig.savefig(png,format="png",dpi=145,metadata={"Software":"Audited core contrast visualization"})
        clean=("\n".join(line.rstrip() for line in svg.getvalue().decode().splitlines())+"\n").encode()
        return clean,png.getvalue(),artists,font,sha(font_path)
    finally:plt.close(fig)


def execute(public,cache,plot_sha,report_sha,audit_path,audit_sha):
    public,cache=Path(public).resolve(),Path(cache).resolve();scope=public/NAME
    svg_path=scope/"figures/CORE_CONTRAST_v1.svg";binding=scope/"CORE_CONTRAST_FIGURE_BINDINGS_v1.json";md_path=scope/"CORE_CONTRAST_VISUALIZATION_KO_v1.md";preview=cache/NAME/"visualization/CORE_CONTRAST_v1.png"
    if any(p.exists() for p in (svg_path,binding,md_path,preview)):raise FileExistsError("Preserve existing outputs; CreateNew only")
    if Path(__file__).resolve()!=scope/"plot_core_contrast.py" or sha(__file__)!=plot_sha:raise ValueError("Current ROOT pinned plot source required")
    if any(os.environ.get(key)!="1" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")):raise ValueError("Explicit CPU/BLAS1 environment required")
    helper=module(scope/"build_beijing_report.py",INPUT_HELPER_SHA,"core_plot_inputs")
    builder=module(scope/"build_core_contrast.py",BUILDER_SHA,"core_plot_builder")
    style=module(scope/"plot_beijing.py",STYLE_SHA,"core_plot_style")
    if time.time()>=style.DEADLINE:raise RuntimeError("Original visualization worker deadline reached")
    inputs=helper.Inputs(public,cache)
    for name,digest in (("plot_core_contrast.py",plot_sha),("build_core_contrast.py",BUILDER_SHA),("audit_core_contrast.py",AUDITOR_SHA),("plot_beijing.py",STYLE_SHA),("build_beijing_report.py",INPUT_HELPER_SHA)):
        inputs.capture("public",NAME+"/"+name,digest)
    report=inputs.json("public",NAME+"/CORE_CONTRAST_BINDINGS_v1.json",report_sha)
    audit_path=Path(audit_path).resolve()
    if not audit_path.is_relative_to(scope) or audit_path.suffix!=".json":raise ValueError("Explicit actual core audit JSON under N4 required")
    relative=audit_path.relative_to(public).as_posix();audit=inputs.json("public",relative,audit_sha);builder.validate_audit(audit)
    if report.get("audit_sha256")!=audit_sha or report.get("audit_relative_path")!=relative:raise ValueError("Report must bind the same actual PASS audit")
    for row in (*report.get("inputs",[]),*audit.get("inputs",[])):inputs.capture(row["root"],row["relative_path"],row["sha256"])
    outputs=report.get("outputs",[]);csv_relative=NAME+"/CORE_CONTRAST_RESULTS_v1.csv"
    if len(outputs)!=2 or {r["relative_path"] for r in outputs}!={csv_relative,NAME+"/CORE_CONTRAST_KO_v1.md"}:raise ValueError("Exact actual builder CSV/MD outputs required")
    for row in outputs:inputs.capture("public",row["relative_path"],row["sha256"])
    with inputs.path("public",csv_relative).open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    data=prepare_data(rows,audit,report);csv_sha=inputs.files[("public",csv_relative)]
    if not inputs.preserved():raise ValueError("Input bytes changed before renderer")
    started=time.perf_counter()
    with style.ResourceCheck() as resources:
        svg,png,artists,font,font_sha=render(data,style,csv_sha,audit_sha)
        if not inputs.preserved():raise ValueError("Input bytes changed during renderer")
        resources.check()
    md=("# Beijing core contrast 시각자료\n\n![B0 K6와 B2 K6/K11의 전체 V 오차 및 실제 CPU2 whole median](figures/CORE_CONTRAST_v1.svg)\n\n"
        "[원정밀도 CSV](CORE_CONTRAST_RESULTS_v1.csv) · [실험 보고](CORE_CONTRAST_KO_v1.md) · [보고 입력 SHA](CORE_CONTRAST_BINDINGS_v1.json) · [실제 감사]("+audit_path.relative_to(scope).as_posix()+") · [그림 결속](CORE_CONTRAST_FIGURE_BINDINGS_v1.json) · [코드](plot_core_contrast.py)\n\n"
        "같은 Beijing C11/L512/H48·CPU2·V43/20539 observed labels·seed92601의 세 경로를 표시한다. B2 K6는 기존 완료 결과를 재사용했고 B0 K6·B2 K11은 새 완료 결과다. "
        "모든 축은0부터이며 표시 숫자만 반올림한다. CSV 원정밀도·12개 실제 artist 값은 변경하지 않았다. 지연시간은 ordinary whole20 median의 초/batch이며 B8은8 origins 한 batch 전체다. p90·seed 분산·신뢰구간 표시는 추가하지 않았다.\n\n"
        +CAPTION.replace("\n"," ")+"\n\n"
        "이 그림은 같은 decoder만의 비교나 같은 가중치 표현 비교가 아니다. B0/B2 모두 frozen Bolt를 사용하며 새 noBolt 실험이나 Bolt의 일반적 필요성을 증명하지 않는다. "
        "오차·전체 비용·학습량/용량 차이를 같이 보며 검사 통과를 성능 개선이나 독립 일반화로 바꾸어 말하지 않는다. 전체 MASTER 미완료다.\n\n"
        "원 CSV/실제 audit/report SHA에 결속하고 입력 전후 SHA·실제 artist 값을 대사했다. 새 통계/trial 재집계·NPZ/CP/model 읽기·학습·추론·비용·process 제어·Git·network는0이다. SVG/이 MD/결속 JSON은 공개용이고 PNG preview는 private다.\n")
    svg_path.parent.mkdir(exist_ok=True);preview.parent.mkdir(parents=True,exist_ok=True)
    for path,blob in ((svg_path,svg),(preview,png),(md_path,md.encode())):
        with path.open("xb") as f:f.write(blob)
    proof=dict(schema="BEIJING_CORE_CONTRAST_FIGURE_BINDINGS_v1",status="PASS_ACTUAL_AUDITED_CORE_AGGREGATE_VISUALIZATION",plot_source_sha256=plot_sha,builder_source_sha256=BUILDER_SHA,auditor_source_sha256=AUDITOR_SHA,reused_style_source_sha256=STYLE_SHA,actual_audit_sha256=audit_sha,report_bindings_sha256=report_sha,CSV_sha256=csv_sha,input_SHA_before_after_equal=True,
        inputs=[dict(root=root,relative_path=name,sha256=digest) for (root,name),digest in sorted(inputs.files.items())],outputs=[dict(relative_path=p.relative_to(scope).as_posix(),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)) for p,blob in ((svg_path,svg),(md_path,md.encode()))],private_preview=dict(published=False,cache_relative_path=preview.relative_to(cache).as_posix(),sha256=hashlib.sha256(png).hexdigest(),bytes=len(png)),artist_values_from_unchanged_CSV=artists,CSV_precision_preserved=True,artist_scalar_count=12,p90_or_CI_added=False,font=font,font_file_sha256=font_sha,CPU_BLAS_threads=1,own_RSS_limit_bytes=style.MEMORY_LIMIT,sampled_own_peak_RSS_bytes=resources.peak,own_OS_peak_working_set_bytes=resources.os_peak,deadline_unix=style.DEADLINE,elapsed_seconds=time.perf_counter()-started,actual_renderer_executed=True,new_accuracy_or_cost_measurements=0,whole_MASTER_complete=False,independent_generalization_claimed=False,self_hash_policy="Bindings excludes itself")
    with binding.open("x",encoding="utf-8",newline="\n") as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    return dict(status=proof["status"],public_files_written=3,private_previews_written=1,artist_scalar_count=12)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true");p.add_argument("--root-triggered",action="store_true")
    for name in ("public-root","cache-root","plot-sha","report-binding-sha","audit","audit-sha"):p.add_argument("--"+name)
    v=p.parse_args(argv)
    if not v.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED",input_reads=0,files_written=0,actual_render=0,matplotlib_imported="matplotlib" in sys.modules)));return 0
    if not v.root_triggered or any(getattr(v,k) is None for k in ("public_root","cache_root","plot_sha","report_binding_sha","audit","audit_sha")):p.error("ROOT/current source/actual complete report+audit SHA and fresh output required")
    print(json.dumps(execute(v.public_root,v.cache_root,v.plot_sha,v.report_binding_sha,v.audit,v.audit_sha)));return 0


if __name__=="__main__":raise SystemExit(main())
