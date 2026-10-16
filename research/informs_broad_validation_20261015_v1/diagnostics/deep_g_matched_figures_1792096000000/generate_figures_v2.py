"""같은 G의 기존 저장 MSE/MAE만 시각화. 기본 metadata 확인, --execute만 새 PNG/SVG 렌더."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PLAN_SHA="c3ae792061c6a77c1c491d581795957f6ef9407c3400605e94e14d8bfa54655f"
METHODS=("G","Hpast","H_B0","H_B2","RawH","ALPHA_FIT")
LABELS=("G","Hpast","H_B0","H_B2","RawH","alphaFIT")
COLORS=("#888888","#56B4E9","#0072B2","#E69F00","#009E73","#CC79A7")
HATCHES=("","//","..","xx","\\\\","++")
TITLES=dict(beijing="Beijing",jena="Jena",electricity="Electricity",pems08="PeMS08")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def metadata(public):
    if sha(HERE/"PLAN.json")!=PLAN_SHA:raise ValueError("그림 준비 계획 봉인 변경")
    plan=read(HERE/"PLAN.json")
    for name,digest in plan["source_pins"].items():
        if sha(public/name)!=digest:raise ValueError("원 metadata/score CSV 변경: "+name)
    for name,digest in plan["source39"].items():
        if sha(public/name)!=digest:raise ValueError("원 과학 source39 변경")
    if sha(HERE/"ORIGINAL_MATCHED_G_COMPARISON.csv")!=plan["source_csv_sha256"] or sha(HERE/"PLOT_SOURCE_VALUES.csv")!=plan["plot_values_csv_sha256"]:
        raise ValueError("보존 CSV 변경")
    original=csv_rows(HERE/"ORIGINAL_MATCHED_G_COMPARISON.csv");rows=csv_rows(HERE/"PLOT_SOURCE_VALUES.csv")
    if len(original)!=40 or len(rows)!=24 or tuple(plan["method_order"])!=METHODS:raise ValueError("원40/그림24/방법 순서 변경")
    source_index={(x["dataset"],x["method"]):x for x in original}
    index={(x["dataset"],x["method"]):x for x in rows}
    if len(index)!=24 or [(x["dataset"],x["method"]) for x in rows]!=[(ds,method) for ds in plan["datasets"] for method in METHODS]:raise ValueError("중복/빠진 조건 또는 데이터 순서 변경")
    for row in rows:
        expected=source_index[(row["dataset"],row["method"])]
        if any(row.get(key)!=value for key,value in expected.items()):raise ValueError("원 CSV 행과 그림 원수치/metadata 불일치")
        if row["seed"]!="92601" or any(not math.isfinite(float(row[k])) or float(row[k])<0 for k in ("mse","mae")):raise ValueError("seed/원수치 오류")
    for ds in plan["datasets"]:
        group=[index[(ds,m)] for m in METHODS]
        for key in ("G_run_id","G_model_sha","G_artifact_sha256","cohort","n_observed","n_origins","horizon","num_channels"):
            if len({x[key] for x in group})!=1:raise ValueError("같은 G/cohort/차원의 비교가 아님")
        if group[0]["cohort"]!=f"{ds}:primary:V_select:stride{group[0]['horizon']}":raise ValueError("실제 자료별 H/stride 계약 변경")
    for metric in ("mse","mae"):
        if len({index[("jena",m)][metric] for m in METHODS[:4]})!=1:raise ValueError("Jena의 실제 exact 동률 변경")
    if any(index[("jena",m)]["selected_updates"]!="0" for m in METHODS[1:4]):raise ValueError("Jena 선택 checkpoint update 변경")
    return plan,rows,index


def panel_budget(plan,ds):
    budget=plan["budget_metadata"][ds];parents=budget["parents"]
    maximum="/".join(f"{max(parents[m]['epochs']):g}" for m in ("G","B0","B2"))
    chosen="/".join(f"{parents[m]['selected_epoch']:g}" for m in ("G","B0","B2"))
    updates="/".join(str(budget["heads"][m]["selected_updates"]) for m in ("Hpast","H_B0","H_B2","RawH"))
    return f"Parent max search epochs G/B0/B2: {maximum}; selected: {chosen}\nH selected updates past/B0/B2/RawH: {updates}"


def render(public,output):
    plan,rows,index=metadata(public)
    if not output.is_relative_to(public/"diagnostics"):raise ValueError("새 diagnostics 하위 출력 폴더 필요")
    output.mkdir(parents=True,exist_ok=False)
    new(output/"RUN_CLAIM.json",dict(generator_sha256=sha(__file__),plan_sha256=PLAN_SHA,source_csv_sha256=plan["source_csv_sha256"],actual_training_inference_or_rescoring=0))
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
        "svg.fonttype":"none","svg.hashsalt":sha(__file__)})
    generated=[]
    footer=("One primary V cohort per dataset; seed 92601; same frozen G within each panel. Lower is better; separate panel scales start at zero.\n"
        "Parents: A50 search / A_stop selection / A60 refit; Raw Bolt frozen. Neural H: dense M fit for 3 epochs / V checkpoint selection; alphaFIT: analytic M fit.\n"
        "Parent budgets differ; limited screens and last-best heads do not establish convergence. One V/seed gives no independent generalization or seed-variance estimate.\n"
        "Only six matched-G conditions are shown, not all strong baselines. Labels: 6 decimals; accompanying CSV preserves full-precision source values.")
    for metric in ("mse","mae"):
        fig,axes=plt.subplots(2,2,figsize=(14,10),squeeze=False)
        fig.subplots_adjust(left=.075,right=.985,top=.91,bottom=.25,hspace=.67,wspace=.25)
        fig.suptitle("Same frozen G: six forecast/readout conditions — "+metric.upper(),fontsize=17,fontweight="bold",y=.97)
        for panel,ds in enumerate(plan["datasets"]):
            ax=axes[panel//2][panel%2];group=[index[(ds,m)] for m in METHODS];values=[float(x[metric]) for x in group]
            bars=ax.bar(range(6),values,color=COLORS,edgecolor="#222222",linewidth=.7,width=.67,zorder=3)
            for bar,hatch in zip(bars,HATCHES):bar.set_hatch(hatch)
            maximum=max(values);ax.set_ylim(0,maximum*(1.48 if ds=="jena" else 1.27))
            for position,value in enumerate(values):ax.text(position,value+maximum*.018,f"{value:.6f}",ha="center",va="bottom",fontsize=9)
            ax.axhline(values[0],color="#555555",linestyle="--",linewidth=.8,zorder=1)
            ax.set_xticks(range(6));ax.set_xticklabels(LABELS,fontsize=10)
            ax.set_ylabel("Normalized macro "+metric.upper())
            ax.set_title(f"{chr(65+panel)}  {TITLES[ds]} | C={group[0]['num_channels']}, H={group[0]['horizon']}, V origins={group[0]['n_origins']}",loc="left",fontsize=11)
            ax.yaxis.grid(True,color="#dddddd",linewidth=.6,zorder=0);ax.set_axisbelow(True)
            ax.text(0,-.24,panel_budget(plan,ds),transform=ax.transAxes,ha="left",va="top",fontsize=8.5,linespacing=1.4)
            if ds=="jena":
                ax.text(.02,.965,"Exact tie: G = Hpast = H_B0 = H_B2\nSelected H checkpoint update = 0",transform=ax.transAxes,ha="left",va="top",fontsize=9,
                    bbox=dict(boxstyle="round,pad=.3",facecolor="white",edgecolor="#777777",alpha=.95))
        fig.text(.075,.065,footer,ha="left",va="bottom",fontsize=9,linespacing=1.55)
        fig.text(.075,.028,"Source CSV SHA256: "+plan["source_csv_sha256"],ha="left",fontsize=8,color="#555555")
        for extension in ("png","svg"):
            path=output/("same_deep_G_"+metric.upper()+"."+extension)
            options=dict(dpi=300) if extension=="png" else dict(metadata={"Date":None})
            fig.savefig(path,**options);generated.append(path)
        plt.close(fig)
    for name in ("ORIGINAL_MATCHED_G_COMPARISON.csv","PLOT_SOURCE_VALUES.csv","PLAN.json"):
        with (output/name).open("xb") as stream:stream.write((HERE/name).read_bytes())
    metadata(public)
    new(output/"FIGURE_MANIFEST.json",dict(status="RENDERED_EXISTING_METADATA_SCORES_ONLY",generator_sha256=sha(__file__),plan_sha256=PLAN_SHA,
        source_pins=plan["source_pins"],source39=plan["source39"],outputs={p.name:sha(p) for p in generated},
        original_csv_sha256=sha(output/"ORIGINAL_MATCHED_G_COMPARISON.csv"),plot_values_csv_sha256=sha(output/"PLOT_SOURCE_VALUES.csv"),
        metrics=["mse","mae"],panels_each=4,rows_plotted=24,seed=92601,split="V_select",Jena_exact_ties_preserved=True,
        original_inputs_unchanged=True,existing_figures_overwritten=False,visual_review_pending=True,
        new_FP64_rescoring=0,model_inference_training_or_GPU=0,independent_generalization_or_convergence_claimed=False))
    print(json.dumps(dict(status="RENDERED_REQUIRES_VISUAL_REVIEW",output_public_relative_path=output.relative_to(public).as_posix(),figures=4,manifest_sha256=sha(output/"FIGURE_MANIFEST.json"))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--public-root",default=os.environ.get("INFORMS_PUBLIC_ROOT"));parser.add_argument("--output");parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not args.public_root:parser.error("public root 필요")
    public=Path(args.public_root).resolve()
    if args.execute:
        if not args.output:parser.error("렌더에는 새 --output 필요")
        render(public,Path(args.output).resolve())
    else:
        metadata(public)
        print(json.dumps(dict(status="METADATA_ONLY_RENDER_NOT_STARTED",generator_sha256=sha(__file__),plan_sha256=PLAN_SHA,rows=24,Jena_exact_tie=True,
            matplotlib_imported="matplotlib" in sys.modules,numpy_imported="numpy" in sys.modules,actual_figures=0,model_array_or_weight_reads=0)),flush=True)


if __name__=="__main__":main()
