"""Preserve v1 and render readability repairs using the same locked results.

Only learning-plot legend placement and cost-plot labeling/layout change.
The other four figure functions are reused directly. No model, F, GPU, fitting,
prediction or scientific-source modification occurs in this renderer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_report as original


HERE = Path(__file__).resolve().parent
DOCS = ("REPORT_KO.md","PROFESSOR_BRIEF.md","DECISION_LOG.md")
SUFFIXES = ("png","svg")


def versioned(name):
    path = Path(name)
    return path.with_name(path.stem+"_v2"+path.suffix)


def original_files():
    return ([HERE/name for name in DOCS]
            + [HERE/"figures"/f"{name}.{suffix}" for name,_ in original.FIGURES for suffix in SUFFIXES]
            + [HERE/"figures"/"DATA_PROVENANCE.json",HERE/"generate_report.py"])


def hashes(paths):
    return {str(path.relative_to(HERE)):original.sha(path) for path in paths}


def figure_learning_v2(bundle):
    fig,axes = plt.subplots(2,2,figsize=(13.6,10.2),constrained_layout=True)
    for index,dataset in enumerate(original.MAIN_DATASETS):
        axis = axes[index,0]
        for kind in ("B0","B2","B4"):
            spec = original.selected_adapter(bundle,dataset,kind,original.SMALL_K[dataset])
            if spec is None:
                continue
            fit = bundle["fits"][spec["id"]]
            checkpoints = fit["checkpoints"]
            updates = [item["update"] for item in checkpoints]
            color = original.COLORS[kind]
            axis.plot(updates,[item["B"]["mse"] for item in checkpoints],"o-",color=color,label=f"{kind}: B")
            axis.plot(updates,[item["A"]["mse"] for item in checkpoints],"--",color=color,alpha=.6,label=f"{kind}: A probe")
            selected = next(item for item in checkpoints if item["update"] == fit["selected_update"])
            axis.scatter(selected["update"],selected["B"]["mse"],s=90,facecolors="none",edgecolors=color,linewidths=1.8)
        axis.set_xlabel("Optimizer updates")
        original.finish_axis(axis,dataset.title()+": checkpoint selection","Normalized macro MSE")
        axis.legend(fontsize=7,ncol=2)
        axis = axes[index,1]
        labels = ["Raw","B0K","B2K","B4K"]
        for fold_index,fold in enumerate(("primary","secondary")):
            values = [1.]
            for kind in ("B0","B2","B4"):
                spec = original.selected_adapter(bundle,dataset,kind,original.SMALL_K[dataset],fold)
                values.append(original.ratio(bundle,original.score_row(bundle,spec["id"])) if spec else np.nan)
            axis.bar(np.arange(4)+(fold_index-.5)*.34,values,width=.32,
                     color=("#0072b2","#e69f00")[fold_index],alpha=.85,
                     label=("A60/B20/C20","A50/B20/C30")[fold_index])
        axis.axhline(1,color="black",linewidth=.8,linestyle="--")
        axis.set_xticks(range(4),labels)
        original.finish_axis(axis,dataset.title()+": full upstream refit","C MSE / same-fold Raw MSE")
        # The legend is outside the axes, below the x labels. The note retains
        # its original top-left role and no longer shares the legend's area.
        axis.legend(loc="upper center",bbox_to_anchor=(.5,-.10),ncol=2,fontsize=8,frameon=False)
        axis.text(.02,.97,"Different C periods; conditional repeats\nFresh A statistics/PCA/P/E/D/G",
                  transform=axis.transAxes,va="top",fontsize=8,
                  bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
    fig.suptitle("R5: Learning diagnostics and a predeclared second time cutoff",weight="bold")
    original.save_figure(fig,original.FIGURES[3][0])


def _point_key(spec):
    order = {"RAW":0,"ADAPTER":1,"G":2,"SWAP":3,"DIRECT":4,"RIDGE":5,"LORA":6}
    return order.get(spec["type"],9),original.label(spec),spec["id"]


def _number_rail(axis,points):
    """Sorted nonoverlapping number labels connect back to unchanged points."""
    ordered = sorted(points,key=lambda item:(item["y"],item["x"],item["number"]))
    positions = np.linspace(.07,.93,len(ordered)) if len(ordered)>1 else [.5]
    for point,position in zip(ordered,positions):
        axis.annotate(f"{point['number']:02d}",xy=(point["x"],point["y"]),xycoords="data",
            xytext=(1.025,float(position)),textcoords="axes fraction",ha="left",va="center",
            fontsize=8.5,color=point["color"],annotation_clip=False,
            bbox=dict(boxstyle="round,pad=.13",facecolor="white",edgecolor=point["color"],linewidth=.55),
            arrowprops=dict(arrowstyle="-",color=point["color"],linewidth=.65,alpha=.75,
                            shrinkA=2,shrinkB=2,relpos=(0,.5)))


def figure_cost_v2(bundle):
    fig,axes = plt.subplots(2,3,figsize=(18.6,10.6),constrained_layout=True,
                           gridspec_kw={"width_ratios":(1,1,.85)})
    figure_points = []
    for index,dataset in enumerate(original.MAIN_DATASETS):
        by_scope = {
            scope:{spec["id"]:(spec,row,cost) for spec,row,cost in original.selected_cost_rows(bundle,dataset,1,scope)}
            for scope in ("general","exclusive")
        }
        specifications = {identifier:value[0] for items in by_scope.values() for identifier,value in items.items()}
        ordered_ids = sorted(specifications,key=lambda identifier:_point_key(specifications[identifier]))
        numbers = {identifier:position+1 for position,identifier in enumerate(ordered_ids)}
        palette = {identifier:plt.get_cmap("tab20")(position%20) for position,identifier in enumerate(ordered_ids)}
        for column,(scope,field) in enumerate((("general","whole_seconds_median"),("exclusive","peak_allocated_bytes_median"))):
            axis = axes[index,column]
            points = []
            for identifier,(spec,row,cost) in by_scope[scope].items():
                value = original.number(cost.get(field))
                if value is None:
                    continue
                x = value*1000 if column == 0 else value/2**20
                y = original.ratio(bundle,row)
                color = palette[identifier]
                axis.scatter(x,y,color=color,s=52,edgecolor="white",linewidth=.55,zorder=4)
                point = {"number":numbers[identifier],"x":x,"y":y,"color":color}
                points.append(point)
                figure_points.append({"dataset":dataset,"id":identifier,"number":numbers[identifier],
                    "scope":scope,"x_field":field,"x_plotted":x,"C_MSE_Raw_ratio":y})
            _number_rail(axis,points)
            axis.axhline(1,color="black",linestyle="--",linewidth=.8)
            axis.set_xlabel("Balanced general median latency (ms), batch 1" if column == 0
                            else "Exclusive-process peak allocated (MiB), batch 1")
            original.finish_axis(axis,dataset.title()+(" / general latency" if column == 0 else " / exclusive memory"),
                                 "C MSE / matched Raw MSE")
            axis.margins(x=.06,y=.08)
        key_axis = axes[index,2]
        key_axis.set_axis_off()
        key_axis.set_title(dataset.title()+": method key",fontsize=11,weight="bold",loc="left")
        spacing = min(.065,.86/max(len(ordered_ids)-1,1))
        for position,identifier in enumerate(ordered_ids):
            key_axis.text(.015,.96-position*spacing,f"{numbers[identifier]:02d}  {original.label(specifications[identifier])}",
                          transform=key_axis.transAxes,color=palette[identifier],fontsize=9,va="top",weight="medium")
        key_axis.text(.015,.015,"Same number = same locked route\nLatency: balanced general only\nMemory: exclusive process only",
                      transform=key_axis.transAxes,fontsize=8,color="#444444",va="bottom")
    fig.suptitle("Accuracy-cost tradeoffs: general latency and exclusive memory are distinct scopes",weight="bold")
    original.save_figure(fig,original.FIGURES[4][0])
    return figure_points


def remap_links(text):
    for name,_ in original.FIGURES:
        for suffix in SUFFIXES:
            text = text.replace(f"figures/{name}.{suffix}",f"figures/{name}_v2.{suffix}")
    for name in DOCS:
        text = text.replace(f"]({name})",f"]({versioned(name).name})")
    return text.replace("figures/DATA_PROVENANCE.json","figures/DATA_PROVENANCE_v2.json")


def oracle_interpretation(bundle):
    receipt = original.read_json(HERE/"ORACLE_DIAGNOSTIC.json")
    values = {(row["dataset"],row["family"]):row for row in receipt["results"] if "fixed_D_FP64_real_oracle" in row}
    beijing2,beijing4,robin4 = (values[key] for key in (("beijing","B2"),("beijing","B4"),("robin","B4")))
    b2,b4,r4 = (row["fixed_D_FP64_real_oracle"]["mse"] for row in (beijing2,beijing4,robin4))
    beijing_raw,robin_raw = beijing2["trigger"]["canonical_Raw_C_mse"],robin4["trigger"]["canonical_Raw_C_mse"]
    if not (b2<beijing_raw and b4<beijing_raw and r4>robin_raw):
        raise ValueError("Dataset-specific oracle interpretation no longer matches locked diagnostic numbers")
    return (
        "**자료별 고정 D 진단의 의미:** Beijing B2/B4의 자유 미래 latent 진단 MSE는 각각 "
        f"{original.fmt(b2)}/{original.fmt(b4)}로 Raw {original.fmt(beijing_raw)}보다 작다. "
        "따라서 이번에 유지한 FP64 수치 공간에서 선정된 고정 D의 출력 공간만으로 Raw 패배가 불가피하다고 할 수는 없다. "
        "실제 과거 입력에서 그 자유 latent를 얻을 수 있는지는 별개다. "
        f"Robin B4의 같은 진단은 {original.fmt(r4)}로 Raw {original.fmt(robin_raw)}보다 커서 "
        "해당 선정 고정 D와 유지된 수치 공간의 제약이 남는다. "
        "이 값도 rcond 절단 없는 모든 실수/모든 FP32 출력의 엄격 하한으로 확대하지 않는다. "
        "둘 다 실제 미래 정답을 사용한 진단이므로 과거 표현의 예측 충분성·F 또는 native head의 학습 가능성을 입증하지 않는다. "
        "다른 decoder나 모든 작은 K 방법에 대한 불가능성 주장도 아니다."
    )


def brief_conclusion(bundle):
    rows = []
    for method in ("Raw","B-selected smallK","G-only","LoRA qv8"):
        values = []
        for dataset in original.MAIN_DATASETS:
            if method == "Raw":
                spec = original.raw_route(bundle,dataset)
            elif method == "B-selected smallK":
                spec = original.chosen_family(bundle,dataset)
            elif method == "G-only":
                spec = original.g_for(bundle,dataset,"G-only")
            else:
                candidates = [item for item in bundle["routes"].values() if item["dataset"] == dataset
                              and item.get("fold","primary") == "primary" and item["type"] == "LORA"]
                if len(candidates) != 1:
                    raise ValueError("Expected the one predeclared LoRA route")
                spec = candidates[0]
            values.append(original.fmt(original.score_row(bundle,spec["id"])["mse"]))
        rows.append([method,*values])
    return (original.code_table(("method","Beijing_C_MSE","Jena_C_MSE"),rows)
        + "\n\n시간 앞단이 r=C에서 도움이 될 수 있지만 작은 K+G의 G-only 대비 추가 가치는 확보하지 못했다. "
          "LoRA도 강한 반례이며 현재 결과를 성공 방법론이나 신규성 확정으로 부르지 않는다. "
          "좁힌 질문은 **원채널 예측의 이득을 소수 F streams에서도 단순 직접예측/LoRA보다 유지할 수 있는가?**다. "
          "현재 답은 미확보다. 위 smallK는 C로 다시 고른 모델이 아니라 기존 locked B 선택이며 두 seed/시간 반복의 한계를 유지한다.")


def main():
    original_paths = original_files()
    before = hashes(original_paths)
    bundle = original.load_bundle()
    oracle = original.read_json(HERE/"ORACLE_DIAGNOSTIC.json")
    if oracle.get("status") != "COMPLETE_VALID":
        raise ValueError("Oracle completion gate is not valid")
    outputs = ([HERE/versioned(name) for name in DOCS]
        + [HERE/"figures"/f"{name}_v2.{suffix}" for name,_ in original.FIGURES for suffix in SUFFIXES]
        + [HERE/"figures"/"DATA_PROVENANCE_v2.json"])
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preserve an existing v2 rendering; do not overwrite")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":9,"legend.fontsize":8,
                         "svg.fonttype":"none","savefig.facecolor":"white"})
    save_original = original.save_figure
    original.save_figure = lambda fig,name:save_original(fig,name+"_v2")
    try:
        original.figure_r1(bundle)
        original.figure_temporal_rank(bundle)
        original.figure_residual(bundle)
        figure_learning_v2(bundle)
        cost_points = figure_cost_v2(bundle)
        original.figure_cases(bundle)
    finally:
        original.save_figure = save_original
    documents = {name:remap_links((HERE/name).read_text(encoding="utf-8")) for name in DOCS}
    oracle_text = oracle_interpretation(bundle)
    documents["REPORT_KO.md"] = documents["REPORT_KO.md"].replace("## R5: 학습·시간 경계·가지 제거·오차 구간",oracle_text+"\n\n## R5: 학습·시간 경계·가지 제거·오차 구간",1)
    documents["PROFESSOR_BRIEF.md"] = documents["PROFESSOR_BRIEF.md"].replace("## 7. 남는 기여와 보류할 주장",brief_conclusion(bundle)+"\n\n## 7. 남는 기여와 보류할 주장",1)
    documents["DECISION_LOG.md"] += ("\n## v2 표현 수선과 자료별 진단 해석\n\n"
        "원 v1 문서·12개 그림·provenance를 보존하고 legend/라벨 가독성만 별도 파일에서 수선했다. "
        "그림1/2/3/6은 원 함수를 재사용했고 그림4의 오른쪽 legend 위치와 그림5의 번호/연결선/방법 key만 바꿨다. "
        "시간축은 balanced general, 메모리축은 exclusive를 유지했으며 수치·축 단위·선정·원 예측·비용은 바꾸지 않았다. "
        "새 fit/F/GPU 실행은 없다. 원본 SHA와 양쪽 source 바인딩은 [v2 provenance](figures/DATA_PROVENANCE_v2.json)에 있다.\n\n"
        + oracle_text+"\n\n"+brief_conclusion(bundle)+"\n")
    for name,text in documents.items():
        with (HERE/versioned(name)).open("x",encoding="utf-8",newline="\n") as handle:
            handle.write(text)
    after = hashes(original_paths)
    if before != after:
        raise RuntimeError("Original v1 artifact or generator changed during readability rendering")
    provenance = {
        "purpose":"readability-only v2; unchanged scientific results and scope",
        "original_v1_sha256_before":before,"original_v1_sha256_after":after,"all_original_v1_hashes_unchanged":True,
        "source_sha256":{"generate_report.py":original.sha(HERE/"generate_report.py"),Path(__file__).name:original.sha(Path(__file__))},
        "scientific_inputs_sha256":{name:original.sha(HERE/name) for name in
            ("SCORES.csv","COST.csv","SELECTION_LOCK.json","COST_LOCK.json","AUDIT_LATEST.json","ANALYSIS_COMPLETE.json","ORACLE_DIAGNOSTIC.json")},
        "audit_file":bundle["audit_path"].name,"audit_sha256":original.sha(bundle["audit_path"]),
        "run_index_file":bundle["run_index_path"].name,"run_index_sha256":original.sha(bundle["run_index_path"]),
        "unchanged_functions":["figure_r1","figure_temporal_rank","figure_residual","figure_cases"],
        "figure04_change":"right-hand legend moved outside plot below axis",
        "figure05_change":"same point values; number rails, connecting lines, separate method key; general latency/exclusive memory",
        "figure05_points":cost_points,"case_receipts":bundle["case_receipts"],
        "new_model_selection":False,"new_fits":0,"F_calls":0,"GPU_calls":0,
        "output_sha256":{str(path.relative_to(HERE)):original.sha(path) for path in outputs if path.name!="DATA_PROVENANCE_v2.json"},
    }
    with (HERE/"figures"/"DATA_PROVENANCE_v2.json").open("x",encoding="utf-8") as handle:
        json.dump(provenance,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status":"READABILITY_V2_GENERATED","reports":3,"PNG":6,"SVG":6,
                      "original_v1_hashes_unchanged":True,"new_fits":0,"F_calls":0,"GPU_calls":0},ensure_ascii=False))


if __name__ == "__main__":
    main()
