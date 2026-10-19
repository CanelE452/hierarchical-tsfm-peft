"""Read-only, registered final report over independently audited public results."""
import argparse
import copy
import json
import math
from pathlib import Path
from time import time

from research.raw_context_multiscale_20261018_v1 import common, report as primary


NAMESPACE = "raw_context_multiscale_supplement_20261019_v1"
EXTRA_TABLES = ("DATA_SCOPES", "CP0_A_STOP", "SUPPLEMENT_FIGURE_POINTS")
TABLES = (*primary.TABLES, *EXTRA_TABLES)
AMBIGUOUS_MEAN_FIELDS = ("selected_A_stop_epoch", "A_stop_mse", "selected_M_select_epoch",
                         "M_select_mse", "M_cal_fit_mse", "alpha", "convergence")
SHARED_FIELDS = {"dataset", "mode", "base_role", "alpha_label", "C", "L", "H", "K", "K_split", "scales",
                 "n_origins", "n_observed", "evaluation", "V_independent_test", "reused", "status"}
OPERATIONS = dict(S_to_A="Keep the first K/2 PCA projections outside P; same total K/P/E/D",
                  A_to_B="Context block mean s=2, F at 256 steps, original-grid block repeat; P remains at 512",
                  S_to_B="Raw/context split plus context block mean/coarse F/original-grid repeat")


def _json(path, value):
    common.atomic_json(path, value)


def _same_or_none(rows, key):
    values = [row.get(key) for row in rows]
    return values[0] if all(value == values[0] for value in values) else None


def canonical_runs(rows):
    """Repair copied auxiliary fields without altering either V metric or model row."""
    result = copy.deepcopy(rows)
    models = [row for row in result if row["row_kind"] == "model"]
    for row in result:
        if row["row_kind"] != "two_seed_mean":
            row["aggregation_scope"] = "one fitted model; no seed aggregation"
            if row.get("mode") in ("conditional", "past_control"):
                row["M_cal_refit_mean_batch_loss_not_full_evaluation"] = row.get("M_cal_fit_mse")
                row["M_cal_fit_mse"] = None
            continue
        pair = [candidate for candidate in models if all(candidate.get(key) == row.get(key)
                for key in ("dataset", "mode", "base_role", "alpha_label"))]
        if len(pair) != 2 or {item["seed"] for item in pair} != set(common.SEEDS):
            raise ValueError("Every mean row requires the two exact registered seed model rows")
        for metric in ("mse", "mae"):
            expected = math.fsum(float(item[metric]) for item in pair)/2
            if row[metric] != expected:
                raise ValueError("Primary V arithmetic mean changed")
        row["by_seed_auxiliary_values"] = {str(item["seed"]): {key:item.get(key) for key in AMBIGUOUS_MEAN_FIELDS}
                                            for item in sorted(pair, key=lambda item:item["seed"])}
        for key in list(row):
            if key in AMBIGUOUS_MEAN_FIELDS:
                row[key] = None
            elif key in SHARED_FIELDS:
                row[key] = _same_or_none(pair, key)
        row["aggregation_scope"] = "arithmetic mean of paired V MSE and MAE; selection/fit auxiliaries are by_seed, never seed1-as-mean"
        row["M_cal_refit_mean_batch_loss_not_full_evaluation"] = None
    return result


def split_scopes(contract):
    rows = []
    for dataset in common.EXPECTED:
        task = contract["tasks"][dataset]
        length, horizon = int(task["L"]), int(task["H"])
        ranges = dict(task["split_ranges"])
        ranges["A60"] = [0, ranges["M_cal"][0]]
        m0, m1 = ranges["M_cal"]
        cut = m0+3*(m1-m0)//5
        ranges["M_fit"], ranges["M_select"] = [m0, cut], [cut, m1]
        for split in ("A_fit", "A_stop", "A60", "M_cal", "M_fit", "M_select", "V_select", "T_final"):
            lo, hi = ranges[split]
            stride = horizon if split in ("A_stop", "V_select") else 1
            first, last = max(lo, length), hi-horizon
            count = max(0, (last-first)//stride+1)
            rows.append(dict(dataset=dataset, split=split, start_row_inclusive=lo, stop_row_exclusive=hi,
                L=length, H=horizon, C=task["C"], K=common.EXPECTED[dataset]["K"],
                origin_first=first if count and split != "T_final" else None,
                origin_last=(first+(count-1)*stride) if count and split != "T_final" else None,
                origin_stride=stride if split != "T_final" else None,
                origin_count=count if split != "T_final" else None,
                origin_count_scope="derived from sealed ranges; checked against actual V identity in T2" if split != "T_final" else "PROTECTED: not read, scored, or materialized by this supplement",
                M_cal_boundary_discarded_origins=horizon-1 if split == "M_cal" else None,
                targets_overlap_between_M_fit_and_M_select=False))
    return rows


def _body_metadata(ctx, arms, contract):
    """Join exact public receipts; never substitute search exposures for fresh refit work."""
    result, bindings = {}, {}
    for arm in arms:
        if arm["mode"] not in ("S", "A", "B"):
            continue
        run_id = arm["run_id"]
        if arm.get("reused"):
            reference = arm["parent"]["search_result"]["reference"]
            prefix = "repo://.cache/"
            if not reference.startswith(prefix) or not reference.endswith("/SEARCH_RESULT.json"):
                raise ValueError("Reused S must identify its original exact search result")
            relative = reference[len(prefix):].replace("/SEARCH_RESULT.json", "/REFIT_RESULT.json")
            path = Path(ctx["repo"])/"reports"/relative
        else:
            candidates = sorted((Path(ctx["report"])/"runs"/run_id).glob("cap*/REFIT_RESULT.json"))
            matching = [path for path in candidates if common.read(path).get("bundle_binding", {}).get("sha256") == arm["route_spec"]["artifact_sha256"]]
            if len(matching) != 1:
                raise ValueError("Fresh body requires one exact public refit receipt: "+run_id)
            path = matching[0]
        refit = common.read(path)
        receipt, summary = refit["fit_receipt"], refit["refit_summary"]
        task = contract["tasks"][arm["dataset"]]
        fit_end, refit_end = task["split_ranges"]["A_fit"][1], task["split_ranges"]["M_cal"][0]
        if (refit.get("status") != "COMPLETE" or receipt.get("artifact_sha256") != arm["route_spec"]["artifact_sha256"]
                or receipt.get("search_fit_end") != fit_end or receipt.get("refit_fit_end") != refit_end
                or receipt.get("fresh_A60") is not True):
            raise ValueError("Public refit receipt is not the exact A60 deployment artifact: "+run_id)
        if arm.get("refit_summary") is not None and arm["refit_summary"] != summary:
            raise ValueError("Public refit summary differs from scored model")
        search = arm["search_summary"]
        for phase, value in (("search", search), ("refit", summary)):
            for key in ("train_origins", "updates", "exposures", "actual_epochs"):
                if key not in value or not math.isfinite(float(value[key])):
                    raise ValueError("Actual "+phase+" learning amount missing: "+key)
        result[run_id] = dict(V_origin_identity=arm["array_identities"]["origins"], V_origin_count=arm["score"]["n_origins"],
            A_fit_end=fit_end, A60_refit_end=refit_end, selected_A_stop_epoch=search["selected_epoch"],
            search_cap=max(float(row["epoch"]) for row in arm["curves"]), reused_final_body=bool(arm.get("reused")),
            **{"search_"+key:search[key] for key in ("train_origins", "updates", "exposures", "actual_epochs")},
            **{"refit_"+key:summary[key] for key in ("train_origins", "updates", "exposures", "actual_epochs")})
        bindings[run_id] = common.binding(path, ctx)
    return result, bindings


def augment_tables(tables, *, body_metadata, contract, cp0):
    tables = copy.deepcopy(tables)
    tables["RUNS"] = canonical_runs(tables["RUNS"])
    models = {row["run_id"]:row for row in tables["RUNS"] if row["row_kind"] == "model"}
    body_ids = {key for key, row in models.items() if row["mode"] in ("S", "A", "B")}
    if set(body_metadata) != body_ids or {row["run_id"] for row in cp0} != body_ids or len(cp0) != len(body_ids):
        raise ValueError("All completed S/A/B bodies need unique audited CP0 and actual learning metadata")
    cp0_by_id = {row["run_id"]:row for row in cp0}
    for key in sorted(body_ids):
        row, diagnostic = models[key], cp0_by_id[key]
        if (diagnostic["dataset"], diagnostic["mode"], diagnostic["seed"]) != (row["dataset"], row["mode"], row["seed"]):
            raise ValueError("CP0 model identity differs")
        for metric in ("mse", "mae"):
            value = float(diagnostic["A_stop_"+metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError("CP0 requires actual finite MSE and MAE")
            row["CP0_A_stop_"+metric] = value
        points = [point for point in tables["TRAINING_CURVES"] if point["run_id"] == key
                  and point["phase"] == "A_fit_search" and float(point["epoch"]) == 0]
        if len(points) != 1 or not math.isclose(points[0]["astop_mse"], row["CP0_A_stop_mse"], abs_tol=common.ATOL, rel_tol=common.RTOL):
            raise ValueError("Audited CP0 does not match the existing epoch-zero A_stop MSE")
        points[0]["astop_mae"] = row["CP0_A_stop_mae"]
        points[0]["astop_mae_scope"] = "evaluation-only exact A_fit CP0; no checkpoint reselection"
    for row in tables["RUNS"]:
        if row["row_kind"] == "two_seed_mean" and row["mode"] in ("S", "A", "B"):
            pair = [item for item in models.values() if item["dataset"] == row["dataset"] and item["mode"] == row["mode"]]
            for metric in ("mse", "mae"):
                row["CP0_A_stop_"+metric] = math.fsum(item["CP0_A_stop_"+metric] for item in pair)/2
            row["CP0_aggregation_scope"] = "arithmetic mean of two independently audited CP0 diagnostics; no model selected"
    for row in tables["T1_ACCURACY_COST"]:
        if row["row_kind"] == "model":
            model = models[row["accuracy_run_id"]]
        else:
            pair = [item for item in models.values() if item["dataset"] == row["dataset"] and item["mode"] == row["mode"]]
            model = {key:_same_or_none(pair, key) for key in ("K", "K_split", "scales")}
        row["K_split"], row["scales"] = model.get("K_split"), model.get("scales")
        k = model.get("K")
        row["K_raw"] = k//2 if k and row["mode"] in ("A", "B") else 0 if row["mode"] == "S" else None
        row["K_context"] = k-row["K_raw"] if k else None
        row["K_split_numeric"] = f"{row['K_raw']}+{row['K_context']}" if k else None
    for row in tables["T2_SINGLE_HYPOTHESIS"]:
        candidate, reference = body_metadata[row["run_id"]], body_metadata[row["reference_run_id"]]
        if candidate["V_origin_identity"] != reference["V_origin_identity"] or candidate["V_origin_count"] != reference["V_origin_count"]:
            raise ValueError("Paired hypothesis requires identical origin arrays, not merely counts")
        row["changed_operation"] = OPERATIONS.get(row["contrast"].replace("_cost", ""), row.get("changed_operation"))
        row["same_actual_V_origins"] = True
        row["learning_amount_scope"] = "actual search and selected fresh A60 refit receipts; reused prior work is not new training"
        row.update({"candidate_"+key:value for key, value in candidate.items()})
        row.update({"reference_"+key:value for key, value in reference.items()})
    tables["DATA_SCOPES"] = split_scopes(contract)
    tables["CP0_A_STOP"] = copy.deepcopy(cp0)
    tables["SUPPLEMENT_FIGURE_POINTS"] = supplement_points(tables)
    return tables


def supplement_points(tables):
    points = []
    curves = [row for row in tables["TRAINING_CURVES"] if row["phase"] == "A_fit_search"]
    for identifier in sorted({row["run_id"] for row in curves}):
        rows = [row for row in curves if row["run_id"] == identifier]
        cap = max(float(row["epoch"]) for row in rows)
        for row in rows:
            points.append(dict(figure="F1_CAP_MARKED", run_id=identifier, dataset=row["dataset"], mode=row["mode"], seed=row["seed"],
                x=row["updates"], y=row["astop_mse"], metric="mse", epoch=row["epoch"], selected=row["selected"], cap=cap,
                is_cap=float(row["epoch"]) == cap))
    for row in tables["CP0_A_STOP"]:
        for metric in ("mse", "mae"):
            points.append(dict(figure="F1_CP0_DIAGNOSTIC", run_id=row["run_id"], dataset=row["dataset"], mode=row["mode"],
                seed=row["seed"], metric=metric, x=row["mode"], y=row["A_stop_"+metric], epoch=0, selected=False,
                interpretation="pre-training evaluation diagnostic only; no new search"))
    thirds = [row for row in tables["PAIRED_DELTAS"] if row["axis"] == "horizon_third"]
    groups = {(row["dataset"], row["contrast"]) for row in thirds}
    for dataset, contrast in sorted(groups):
        for metric in ("mse", "mae"):
            for seed in common.SEEDS:
                selected = [row for row in thirds if (row["dataset"], row["contrast"], row["metric"], row["seed"]) == (dataset, contrast, metric, seed)]
                if len(selected) != 3 or {row["index"] for row in selected} != {"near", "middle", "far"}:
                    raise ValueError("Each paired third plot requires both seeds and all three fixed thirds")
                for row in sorted(selected, key=lambda item:("near", "middle", "far").index(item["index"])):
                    if row["delta"] is None:
                        continue
                    points.append(dict(figure="F2_HORIZON_THIRDS", run_id=row["run_id"], reference_run_id=row["reference_run_id"],
                        dataset=dataset, contrast=contrast, seed=seed, metric=metric, x=row["index"], y=row["delta"], count=row["count"],
                        third_scope="fixed contiguous np.array_split thirds; no post-hoc horizon selection"))
    return points


def _render_extra_figures(tables, output):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    points = tables["SUPPLEMENT_FIGURE_POINTS"]
    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":9, "svg.hashsalt":NAMESPACE,
                         "axes.spines.top":False, "axes.spines.right":False}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
        for ax, dataset in zip(axes, common.EXPECTED):
            values = [point for point in points if point["figure"] == "F1_CAP_MARKED" and point["dataset"] == dataset]
            for identifier in sorted({point["run_id"] for point in values}):
                rows = sorted((row for row in values if row["run_id"] == identifier), key=lambda row:row["x"])
                first = rows[0]
                color = primary.COLORS[first["mode"]]
                ax.plot([row["x"] for row in rows], [row["y"] for row in rows], "-" if first["seed"] == common.SEEDS[0] else "--",
                    color=color, marker="o", markersize=3, label=f"{first['mode']} s{first['seed']}; cap={first['cap']:g}")
                selected = [row for row in rows if row["selected"]]
                cap = [row for row in rows if row["is_cap"]]
                ax.scatter([row["x"] for row in selected], [row["y"] for row in selected], marker="*", s=100, color=color, zorder=4)
                ax.scatter([row["x"] for row in cap], [row["y"] for row in cap], marker="s", s=65, facecolors="none", edgecolors=color, zorder=5)
            ax.set(title=dataset, xlabel="Actual optimizer updates", ylabel="A_stop channel-macro MSE")
            ax.grid(alpha=.18)
            if values: ax.legend(fontsize=7, ncol=2)
        fig.suptitle("Matched body search opportunities: stars = selected, hollow squares = admitted cap\nA best-at-cap checkpoint leaves convergence unresolved; cap is not a convergence claim.", fontsize=10)
        primary._figure_files(fig, output, "F1_CAP_MARKED")
        for name, title, ylabel in (("F1_CP0_DIAGNOSTIC", "Exact A_fit CP0: diagnostic MSE and MAE; no new selection", "A_stop channel-macro error"),
                                     ("F2_HORIZON_THIRDS", "Paired deltas for each seed and all fixed horizon thirds; negative favors candidate", "Candidate minus reference")):
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
            for row_index, dataset in enumerate(common.EXPECTED):
                for col, metric in enumerate(("mse", "mae")):
                    ax = axes[row_index, col]
                    values = [point for point in points if point["figure"] == name and point["dataset"] == dataset and point["metric"] == metric]
                    groups = sorted({point.get("contrast") or "CP0" for point in values})
                    for index, group in enumerate(groups):
                        for seed in common.SEEDS:
                            selected = [point for point in values if (point.get("contrast") or "CP0") == group and point["seed"] == seed]
                            labels = ("S", "A", "B") if name == "F1_CP0_DIAGNOSTIC" else ("near", "middle", "far")
                            selected.sort(key=lambda point:labels.index(point["x"]))
                            if selected:
                                ax.plot([labels.index(point["x"]) for point in selected], [point["y"] for point in selected],
                                    "-" if seed == common.SEEDS[0] else "--", marker="o", color=("#0072B2", "#D55E00", "#009E73")[index % 3],
                                    label=f"{group} s{seed}")
                    ax.set(title=f"{dataset}: {metric.upper()}", ylabel=ylabel)
                    ax.set_xticks(range(3), labels=("S", "A", "B") if name == "F1_CP0_DIAGNOSTIC" else ("near", "middle", "far"))
                    if name == "F2_HORIZON_THIRDS": ax.axhline(0, color="black", linewidth=.8)
                    ax.grid(alpha=.18)
                    if values: ax.legend(fontsize=7, ncol=2)
            fig.suptitle(title, fontsize=11)
            primary._figure_files(fig, output, name)


def usage_text(context):
    plan = context["primary_plan_reference"].removeprefix("repo://")
    terminal = context["terminal_reference"].removeprefix("repo://")
    supplement = context["supplement_plan_reference"].removeprefix("repo://")
    module = "research.raw_context_multiscale_20261018_v1.run"
    addon = "research."+NAMESPACE
    return "\n".join(["# 실제 실행·검산·재생성 명령", "", "Python은 기존 FP32/CUDA 환경에서 실행한다. ASSET_ROOT는 데이터·F snapshot·기존 부모를 보관한 로컬 checkout으로 설정한다. 공개 결과에 private 절대경로를 넣지 않는다.", "", "```powershell", '$env:ASSET_ROOT = "<local-asset-root>"',
        f'python -m {module} audit --asset-root "$env:ASSET_ROOT"',
        f'python -m {module} smoke --asset-root "$env:ASSET_ROOT" --device cpu',
        f'python -m {module} execute --asset-root "$env:ASSET_ROOT" --plan {plan}',
        f'python -m {module} verify --asset-root "$env:ASSET_ROOT"',
        f'python -m {module} report --asset-root "$env:ASSET_ROOT"',
        f'python -m {addon}.cp0 --help',
        *context.get("cp0_commands", []),
        f'python -m {addon}.report generate --asset-root "$env:ASSET_ROOT" --plan {supplement} --terminal {terminal}',
        f'python -m {addon}.report replay --source-dir reports/{common.NAME}/final --output-dir .cache/{common.NAME}/final_replay_001',
        "```", "", "audit는 기존 봉인과 부모를 재검증한다. 최초 smoke 명령은 실제 검증된 인터페이스이나 기존 immutable CPU 파일이 있으면 같은 이름을 덮어쓰지 않고 실패할 수 있다. 반복 TRAIN smoke가 필요하면 gates CLI의 새 --output-id로 별도 attempt를 등록한다.",
        "", "execute 재개는 기존 optimizer·sampler·RNG·checkpoint를 보존하며 같은 immutable CORE_EXECUTION_START와 deadline을 사용한다. 새 실행으로 deadline을 연장하지 않는다. 기본 상한은 core wall 6시간·신규 GPU process wall 상한 4시간이며 마지막 20%를 예측·비용·검산·보고에 남긴다. 미입장 연장을 과거 decision을 덮어써서 수행하지 않는다.",
        "", "generate는 main terminal COMPLETE/exit0, primary 독립 검산 PASS, 전체 SCORES/COST_INDEX, CP0 독립 검산 PASS와 등록 source SHA를 먼저 요구한다. 실제 생성 전에 primary 결과를 수정하지 않는다. 기본 final 폴더는 새 출력이어야 하며 replay는 별도 새 폴더에만 쓴다.",
        "", "PARTIAL branch는 이 wrapper에서 독립 검산이 구현돼 있지 않아 fail closed한다. 그 경우 완료 subset의 prediction·cost 독립 검산과 누락 사유·재개 명령을 갖춘 별도 PARTIAL 보고 경로를 먼저 등록해야 한다. 메타데이터만 보고 검산 완료로 표시하지 않는다.", ""])


def _readme(tables, resources, queue, context):
    excluded = ("발표에서 지지 가능한 문장:", "아직 쓰면 안 되는 문장:", "공개 재생성:", "실제 수치 검산:")
    body = "\n".join(line for line in primary._text_report(tables, resources, queue).splitlines()
                     if not line.startswith(excluded))
    lines = [body, "", "## 최종 보고 보완의 범위", "", "이 final/ 보고서는 primary 공개 CSV와 독립 검산을 입력으로 생성했다. 기존 root 보고·13개 producer 소스·선택·예측·오차 원수치를 바꾸지 않았다. T1은 K 분할과 scales, T2는 동일한 실제 V 원점 identity와 후보·기준선 각각의 search/refit 학습량을 추가했다.",
        "", "RUNS의 two_seed_mean은 V MSE·MAE의 산술평균이다. selected epoch/A_stop/M_select/alpha/수렴 상태는 평균으로 가장하지 않고 빈칸 및 by_seed_auxiliary_values로 남겼다. M_cal mean batch loss는 checkpoint 전체 M_cal MSE가 아니다. CP0 A_stop MSE·MAE는 학습 전 고정 모델의 추가 진단이며 체크포인트·전이 선택을 다시 하지 않았다.",
        "", "정확한 반열린 행 범위와 원점 규칙은 DATA_SCOPES.csv에 있다. 보호 T 행은 범위만 인용하고 새 입력·통계·평가를 하지 않았다.", "", "```text"]
    for row in tables["DATA_SCOPES"]:
        lines.append(f"{row['dataset']:7} {row['split']:9} [{row['start_row_inclusive']},{row['stop_row_exclusive']}) origins={row['origin_count'] if row['origin_count'] is not None else 'PROTECTED'} stride={row['origin_stride']}")
    lines += ["```", "", "F1_CAP_MARKED는 실제 update 축에서 선택 star와 cap square를 분리한다. F1_CP0_DIAGNOSTIC는 모든 완료 S/A/B의 CP0 두 지표를, F2_HORIZON_THIRDS는 각 seed의 near/middle/far ΔMSE·ΔMAE를 생략 없이 보여준다. F3_NATIVE_ALL/FIGURE_POINTS의 모든 32 alpha·8 head·직접 기준선·CPU Ridge·B1/B8 facet은 primary와 동일하다. 장치와 memory layer가 다른 결과의 비율을 만들지 않는다.",
        "", "자세한 실제 사용법·immutable deadline·재개 제한은 CLI_USAGE_KO.md, 미완료 extension은 RESUME_QUEUE.md를 따른다. V는 반복 사용된 개발 평가이며 미사용 독립 테스트가 아니다. 추가 fixed-inference/보고 시간은 FINAL_CONTEXT.json의 실제 resource receipt와 원 primary RESOURCE_LEDGER를 구분해 읽는다.", "", "발표에서 지지 가능한 문장:",
        "- 이 고정 분할·PCA 축 배분·총 K·학습 기회에서 관찰한 seed별 방향과 평균을 함께 제시한다.",
        "- A와 B의 대조는 문맥 평균화·낮은 격자 F 예측·원격자 반복 복원의 조합에 한정한다.",
        "- 정확도와 실제 native/matched 실행비용의 교환관계 및 DirectG와의 열세를 함께 밝힌다.",
        "- cap에서 개선 중이면 등록 예산에서의 결과이며 수렴은 미확정이라고 말한다.", "", "아직 쓰면 안 되는 문장:",
        "- 이 V를 미사용 독립 테스트라고 부르거나 두 seed·겹치는 창·채널로 유의성/신뢰구간을 만든다.",
        "- 같은 K가 동일 FLOPs/메모리를 보장하거나 coarse 출력 일부를 읽으면 F head 연산도 비례 감소한다고 말한다.",
        "- 이 결과만으로 F의 필수성·모든 데이터에서의 우위·TimeMixer/NHITS 전체 재현을 입증했다고 말한다.",
        "- 학습 상한 또는 반복 복원의 한계를 구조적 실패/고해상도 완전 복원으로 단정한다.", ""]
    return "\n".join(lines)


def render_public(tables, resources, queue, context, output_dir):
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Final output must be new or empty; preserve prior generated files")
    primary.render_public(tables, resources, queue, output)
    for name in EXTRA_TABLES:
        primary._csv(output/(name+".csv"), tables[name])
    _render_extra_figures(tables, output)
    manifest = common.read(output/"FIGURE_MANIFEST.json")
    for name in ("F1_CAP_MARKED", "F1_CP0_DIAGNOSTIC", "F2_HORIZON_THIRDS"):
        manifest["figures"].append(dict(name=name, png_sha256=common.sha(output/(name+".png")),
            svg_sha256=common.sha(output/(name+".svg")), source_table="SUPPLEMENT_FIGURE_POINTS.csv",
            point_rows=sum(row["figure"] == name for row in tables["SUPPLEMENT_FIGURE_POINTS"])))
    _json(output/"FIGURE_MANIFEST.json", manifest)
    _write_final_metadata(tables, resources, queue, context, output)
    return dict(status="COMPLETE", files=sorted(path.name for path in output.iterdir() if path.is_file()))


def _write_final_metadata(tables, resources, queue, context, output):
    _json(output/"RESOURCE_LEDGER.json", resources)
    _json(output/"FINAL_CONTEXT.json", context)
    (output/"CLI_USAGE_KO.md").write_text(usage_text(context), encoding="utf8", newline="\n")
    (output/"README_KO.md").write_text(_readme(tables, resources, queue, context), encoding="utf8", newline="\n")
    _json(output/"REPORT_INPUTS.json", dict(status="FULL_AUDITED_PUBLIC_REPLAY", source_scope="Primary independent audit plus registered CP0 evaluation-only audit",
        tables={name:common.sha(output/(name+".csv")) for name in TABLES},
        json_files={name:common.sha(output/name) for name in ("RESOURCE_LEDGER.json", "FINAL_CONTEXT.json")},
        resume_queue=queue, code="research/"+NAMESPACE+"/report.py", producer_sources=context["supplement_source_files"]))


def final_resources(primary_resources, *, core_start, events, started_unix, finished_unix, cp0_timings):
    """Recompute GPU session totals once; never add a primary subtotal twice."""
    result = copy.deepcopy(primary_resources)
    opened, closed = {}, {}
    for row in events:
        if row.get("stage") != "GPU_SESSION":
            continue
        session = row["session_id"]
        destination = opened if row["status"] == "STARTED" else closed if row["status"] in ("COMPLETE", "FAILED") else None
        if destination is None or session in destination:
            raise ValueError("Unknown or duplicate GPU session event")
        destination[session] = row
    if set(opened) != set(closed):
        raise ValueError("Final resource accounting requires every GPU session closed exactly once")
    for session, end in closed.items():
        if end["time_unix"] < opened[session]["time_unix"] or float(end["wall_seconds"]) < 0:
            raise ValueError("GPU session elapsed time is invalid")
    gpu_seconds = math.fsum(float(row["wall_seconds"]) for row in closed.values())
    core_started = core_start["started_unix"]
    deadline = core_started+core_start["limits"]["wall_seconds"]
    result.update(accounted_through_unix=finished_unix, campaign_wall_seconds=finished_unix-result["started_unix"],
        overall_wall_seconds_since_first_CPU_audit=finished_unix-result["started_unix"],
        core_execution_started_unix=core_started, core_execution_wall_seconds=finished_unix-core_started,
        core_wall_budget_remaining_seconds=deadline-finished_unix,
        GPU_reserved_process_wall_upper_bound_seconds=gpu_seconds, unmatched_started_GPU_sessions=0,
        unmatched_started_GPU_accounted_seconds=0., final_GPU_session_count=len(closed),
        final_GPU_budget_remaining_seconds=core_start["limits"]["GPU_reservation_seconds"]-gpu_seconds,
        immutable_core_deadline_unix=deadline, final_report_started_unix=started_unix,
        final_report_CPU_wall_seconds=finished_unix-started_unix,
        final_report_timing_scope="Includes input gates, public metadata joins and all CSV/figure rendering through the frozen end timestamp; final metadata seal I/O and later replay/publication are separately accounted by the root completion receipt",
        CP0_supplement_timings=cp0_timings,
        GPU_accounting="Recomputed once from unique primary+supplement EVENTS GPU sessions, including failed sessions; GPU-owned wall upper bound, not kernel time",
        event_snapshot=[{key:row[key] for key in ("stage", "status", "time_unix", "label", "session_id", "wall_seconds") if key in row} for row in events],
        wall_within_original_limit=finished_unix <= deadline,
        GPU_within_original_limit=gpu_seconds <= core_start["limits"]["GPU_reservation_seconds"])
    return result


def replay(source_dir, output_dir):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("Replay requires a separate output directory outside the source report")
    inputs = common.read(source/"REPORT_INPUTS.json")
    if inputs.get("status") != "FULL_AUDITED_PUBLIC_REPLAY" or set(inputs["tables"]) != set(TABLES):
        raise ValueError("Registered complete final report inputs are required")
    for name, digest in inputs["tables"].items():
        if common.sha(source/(name+".csv")) != digest:
            raise ValueError("Public source CSV changed: "+name)
    for name, digest in inputs["json_files"].items():
        if common.sha(source/name) != digest:
            raise ValueError("Public report JSON changed: "+name)
    tables = {name:primary._read_csv(source/(name+".csv")) for name in TABLES}
    return render_public(tables, common.read(source/"RESOURCE_LEDGER.json"), inputs["resume_queue"],
                         common.read(source/"FINAL_CONTEXT.json"), output)


def _validate_registration(ctx, path):
    plan = common.read(path)
    if (plan.get("status") != "REGISTERED_BEFORE_NEW_V" or plan.get("scientific_contract_changed") is not False
            or plan.get("selection_changes") is not False or plan.get("new_body_training") is not False):
        raise ValueError("Pre-V, evaluation/report-only supplement registration required")
    primary_plan_path = common.resolve_binding(plan["primary_plan_binding"], ctx)
    primary_plan = common.read(primary_plan_path)
    if plan["primary_source_files"] != primary_plan["source_files"] or len(plan["primary_source_files"]) != 13:
        raise ValueError("All thirteen frozen primary sources must match the registered experiment")
    for group in ("primary_source_files", "source_files"):
        if not plan[group]:
            raise ValueError("Supplement sources must be registered")
        for relative, digest in plan[group].items():
            source = (Path(ctx["repo"])/relative).resolve()
            if not source.is_relative_to(Path(ctx["repo"]).resolve()) or common.sha(source) != digest:
                raise ValueError("Registered source changed: "+relative)
    if "research/"+NAMESPACE+"/report.py" not in plan["source_files"]:
        raise ValueError("The actual final report renderer must be registered")
    required = {"research/"+NAMESPACE+"/"+name for name in ("__init__.py", "cp0.py", "report.py")}
    if set(plan["source_files"]) != required:
        raise ValueError("All three supplement sources must be registered together")
    if not isinstance(plan.get("cp0_commands"), list) or not plan["cp0_commands"] or any(
            not isinstance(command, str) or not command.strip() for command in plan["cp0_commands"]):
        raise ValueError("Exact tested CP0 commands must be a nonempty list of complete command strings")
    return plan, primary_plan_path


def generate(ctx, *, plan_path, terminal_path, output_dir=None, mode="full"):
    report_started = time()
    if mode != "full":
        raise ValueError("PARTIAL unsupported: independently audit completed prediction/cost subsets and register a partial report path first; no unverified metrics published")
    plan, primary_plan_path = _validate_registration(ctx, plan_path)
    core_start_path = common.resolve_binding(plan["original_execution_start_binding"], ctx)
    core_start = common.read(core_start_path)
    if core_start["started_unix"]+core_start["limits"]["wall_seconds"] != plan["original_deadline_unix"]:
        raise ValueError("Original core execution deadline changed")
    if report_started >= plan["original_deadline_unix"]:
        raise TimeoutError("Original core deadline exhausted before final report; preserve existing results and register honest partial continuation")
    terminal = common.read(terminal_path)
    if terminal.get("status") != "COMPLETE" or terminal.get("exit_code") != 0:
        raise ValueError("Main terminal must be COMPLETE with exit0 before reading score/report inputs")
    validation_path = Path(ctx["report"])/"VALIDATION_REPORT.json"
    validation = common.read(validation_path)
    if validation.get("status") != "PASS":
        raise ValueError("Primary independent numerical validation must PASS")
    manifest = validation["manifest_bindings"]
    if manifest["registered_plan"] != common.binding(primary_plan_path, ctx):
        raise ValueError("Primary audit binds a different frozen experiment plan")
    for name in ("SCORES.json", "COST_INDEX.json", "MODELS_SEALED_BEFORE_V.json"):
        if common.resolve_binding(manifest[name], ctx) != Path(ctx["report"])/name:
            raise ValueError("Primary validation does not bind exact current manifest")
    cp0_validation_path = Path(ctx["report"])/"CP0_A_STOP_VALIDATION.json"
    cp0_validation = common.read(cp0_validation_path)
    if cp0_validation.get("status") != "PASS":
        raise ValueError("CP0 independent validation must PASS before reading CP0 metrics")
    if (cp0_validation.get("coverage") != 10 or cp0_validation.get("expected_coverage") != 10
            or cp0_validation.get("training_updates") != 0 or cp0_validation.get("V_access") is not False
            or cp0_validation.get("T_access") is not False):
        raise ValueError("All ten evaluation-only CP0 paths must be audited with no training/V/T access")
    scores = common.read(Path(ctx["report"])/"SCORES.json")
    cost_index = common.read(Path(ctx["report"])/"COST_INDEX.json")
    if scores.get("status") != "COMPLETE" or cost_index.get("status") != "COMPLETE":
        raise ValueError("All registered score arms and cost facets must be complete")
    if validation.get("models") != len(scores["arms"]) or validation.get("cost_facets") != len(cost_index["facets"]):
        raise ValueError("Complete primary independent validation coverage differs from the current manifests")
    if len(scores["arms"]) != 56 or len({arm["run_id"] for arm in scores["arms"]}) != 56:
        raise ValueError("FULL report requires all 56 unique registered model arms")
    primary_inputs = common.read(Path(ctx["report"])/"REPORT_INPUTS.json")
    for name in primary.TABLES:
        if common.sha(Path(ctx["report"])/(name+".csv")) != primary_inputs["tables"][name]:
            raise ValueError("Primary public table changed: "+name)
    tables = {name:primary._read_csv(Path(ctx["report"])/(name+".csv")) for name in primary.TABLES}
    if ({row["run_id"] for row in tables["RUNS"] if row["row_kind"] == "model"} != {arm["run_id"] for arm in scores["arms"]}
            or {row["cost_id"] for row in tables["COST"]} != {facet["cost_id"] for facet in cost_index["facets"]}):
        raise ValueError("Public tables must retain every audited model and cost facet")
    contract_path = Path(ctx["repo"])/"reports/b2_revin_20261018_v1/DATA_CONTRACT.json"
    contract = common.read(contract_path)
    cp0_path = Path(ctx["report"])/"CP0_A_STOP.csv"
    cp0_binding = common.binding(cp0_path, ctx)
    if cp0_validation.get("CSV_binding") != cp0_binding:
        raise ValueError("CP0 validation must bind the exact diagnostic CSV")
    if (cp0_validation.get("plan_binding") != common.binding(plan_path, ctx)
            or cp0_validation.get("primary_plan_binding") != common.binding(primary_plan_path, ctx)):
        raise ValueError("CP0 validation must bind this evaluation-only plan and exact primary plan")
    cp0 = primary._read_csv(cp0_path)
    body_metadata, refit_bindings = _body_metadata(ctx, scores["arms"], contract)
    tables = augment_tables(tables, body_metadata=body_metadata, contract=contract, cp0=cp0)
    resources = common.read(Path(ctx["report"])/"RESOURCE_LEDGER.json")
    context = dict(status="FULL", primary_plan_reference="repo://"+primary_plan_path.relative_to(ctx["repo"]).as_posix(),
        supplement_plan_reference="repo://"+Path(plan_path).resolve().relative_to(ctx["repo"]).as_posix(),
        terminal_reference="repo://"+Path(terminal_path).resolve().relative_to(ctx["repo"]).as_posix(),
        primary_source_files=plan["primary_source_files"], supplement_source_files=plan["source_files"],
        primary_input_bindings={name:common.binding(Path(ctx["report"])/name, ctx) for name in
            ("VALIDATION_REPORT.json", "REPORT_INPUTS.json", "RESOURCE_LEDGER.json", "CP0_A_STOP_VALIDATION.json")},
        primary_tables={name:common.binding(Path(ctx["report"])/(name+".csv"), ctx) for name in primary.TABLES},
        supplement_plan_binding=common.binding(plan_path, ctx), data_contract_binding=common.binding(contract_path, ctx),
        terminal_binding=common.binding(terminal_path, ctx), cp0_csv_binding=cp0_binding, body_refit_bindings=refit_bindings,
        cp0_commands=plan.get("cp0_commands", []),
        cp0_resource_receipt={key:cp0_validation[key] for key in ("wall_seconds", "started_unix", "finished_unix",
            "GPU_seconds_before", "GPU_seconds_after", "GPU_seconds_added", "GPU_accounting")},
        primary_resources_scope="Counts/storage/fit receipts preserved; final wall and all GPU session totals recomputed from immutable start and event snapshot without duplicate primary subtotal",
        metric_changes=False, selection_changes=False, new_body_training=False, protected_T_read=False)
    output = Path(ctx["report"])/"final" if output_dir is None else Path(output_dir).resolve()
    if output != Path(ctx["report"])/"final" and not output.is_relative_to(Path(ctx["cache"]).resolve()):
        raise ValueError("Alternative final output must stay in the campaign private cache")
    result = render_public(tables, resources, primary_inputs["resume_queue"], context, output)
    events_path = Path(ctx["cache"])/"EVENTS.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf8").splitlines()]
    context["resource_events_binding"] = common.binding(events_path, ctx)
    context["original_execution_start_binding"] = plan["original_execution_start_binding"]
    resources = final_resources(resources, core_start=core_start, events=events, started_unix=report_started,
        finished_unix=time(), cp0_timings=context["cp0_resource_receipt"])
    context["final_resource_snapshot_unix"] = resources["accounted_through_unix"]
    _write_final_metadata(tables, resources, primary_inputs["resume_queue"], context, output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "replay"))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("full", "partial"), default="full")
    args = parser.parse_args(argv)
    if args.stage == "replay":
        if args.source_dir is None or args.output_dir is None:
            parser.error("replay requires --source-dir and --output-dir")
        result = replay(args.source_dir, args.output_dir)
    else:
        if args.asset_root is None or args.plan is None or args.terminal is None:
            parser.error("generate requires --asset-root, --plan and --terminal")
        ctx = dict(common.paths(), asset=args.asset_root.resolve())
        result = generate(ctx, plan_path=args.plan, terminal_path=args.terminal, output_dir=args.output_dir, mode=args.mode)
    print(json.dumps(dict(status=result["status"], stage=args.stage)))


if __name__ == "__main__":
    main()
