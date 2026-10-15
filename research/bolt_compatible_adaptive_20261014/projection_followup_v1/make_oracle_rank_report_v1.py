"""Addendum for audited future-informed rank-K mathematics; metadata JSON only."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASETS = ("robin", "beijing_aotizhongxin")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    if Path(path).suffix.lower() != ".json":
        raise ValueError("RANK_REPORT_JSON_METADATA_ONLY")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def local_link(report_path, target):
    return Path(os.path.relpath(Path(target).resolve(), Path(report_path).resolve().parent)).as_posix()


def load_verified(diagnostic_path, audit_path):
    diagnostic, audit = read_json(diagnostic_path), read_json(audit_path)
    if diagnostic.get("status") != "COMPLETE" or audit.get("status") != "PASS":
        raise ValueError("RANK_REPORT_REQUIRES_COMPLETE_DIAGNOSTIC_AND_PASS_AUDIT")
    if audit.get("diagnostic", {}).get("sha256") != sha(diagnostic_path):
        raise ValueError("RANK_REPORT_EXACT_DIAGNOSTIC_SHA_REQUIRED")
    bindings = {"producer": "producer", "projection_audit": "projection_audit",
                "fixed_span": "span_diagnostic", "fixed_span_audit": "span_audit"}
    for source_key, audit_key in bindings.items():
        source = diagnostic[source_key]
        path = Path(source["path"])
        if path.suffix.lower() != ".json" or sha(path) != source["sha256"]:
            raise ValueError("RANK_REPORT_LINKED_JSON_CHANGED:" + source_key)
        if audit.get(audit_key, {}).get("sha256") != source["sha256"]:
            raise ValueError("RANK_REPORT_INDEPENDENT_BINDING_MISMATCH:" + source_key)
    if diagnostic.get("protected_TEST_accessed") is not False:
        raise ValueError("RANK_REPORT_PROTECTED_TEST_SCOPE")
    if diagnostic.get("deployment_or_method_checkpoint_branch_selection_used") is not False:
        raise ValueError("RANK_REPORT_ORACLE_SELECTION_USED")
    if diagnostic.get("original_selected_forecasts_and_selection_lock_unchanged") is not True:
        raise ValueError("RANK_REPORT_ORIGINAL_SELECTION_CHANGED")
    if audit.get("totals", {}).get("dataset_split_cells_checked") != 6:
        raise ValueError("RANK_REPORT_SIX_CELLS_REQUIRED")
    return diagnostic, audit


def collect_tables(diagnostic, audit):
    rows, energy, scores = [], [], []
    spectrum = []
    audited_cells = {(cell["dataset"], cell["split"]): cell for cell in audit.get("cells", [])}
    for dataset in DATASETS:
        for split, cell in diagnostic["datasets"][dataset]["splits"].items():
            if cell["directions_fitted_to_known_future_of_this_split"] is not True or cell["one_shared_output_space_across_splits_assumed"] is not False:
                raise ValueError("RANK_REPORT_FUTURE_DIRECTION_SCOPE")
            if cell["deployment_used"] or cell["method_checkpoint_branch_selection_used"] or cell["new_deployable_training"]:
                raise ValueError("RANK_REPORT_ORACLE_IS_NOT_DEPLOYMENT")
            if cell["centered_or_future_intercept_used"]:
                raise ValueError("RANK_REPORT_UNCENTERED_CURRENT_ANCHOR_CONTRACT")
            comparison = cell["fixed_D_and_RAW_comparisons_same_mask"]
            values = cell["energy_certificate"]
            total = values["total_innovation_energy"]
            rows.append({"dataset": dataset, "split": split, "C": cell["channels"], "K": cell["K"],
                         "complete_target_positions": cell["complete_target_positions"],
                         "total_target_positions": cell["total_target_positions"],
                         "known_future_rank_K_normalized_MSE_minimum": cell["minimum_normalized_MSE_from_discarded_energy"],
                         "Raw_normalized_MSE_same_complete_mask": cell["RAW_scores_same_mask"]["normalized"]["mse"],
                         "PCA_fixed_D_real_normalized_MSE_minimum": comparison["PCA"]["fixed_D_floor_MSE_same_mask"],
                         "ForeCA_fixed_D_real_normalized_MSE_minimum": comparison["FORECA"]["fixed_D_floor_MSE_same_mask"],
                         "future_informed_direction_selection": True, "independent_holdout": False,
                         "one_D_across_splits": False, "deployable": False})
            energy.append({"dataset": dataset, "split": split, "K": cell["K"],
                           "future_matrix_shape": cell["future_matrix_shape"],
                           "numerical_rank_future_matrix": cell["numerical_rank_future_matrix"], **values,
                           "retained_energy_fraction": values["retained_energy"] / total,
                           "discarded_energy_fraction": values["discarded_energy"] / total})
            audited = audited_cells.get((dataset, split), {})
            original_values = audited.get("independent_Gram_eigenvalues")
            clipped_values = audited.get("clipped_nonnegative_Gram_eigenvalues")
            if original_values is not None and clipped_values is not None:
                for index, (original, clipped) in enumerate(zip(original_values, clipped_values), 1):
                    spectrum.append({"dataset": dataset, "split": split, "direction_rank": index,
                                     "K": cell["K"], "independent_Gram_eigenvalue": original,
                                     "clipped_nonnegative_Gram_eigenvalue": clipped,
                                     "innovation_energy_fraction": clipped / total,
                                     "retained_in_K": index <= cell["K"],
                                     "negative_roundoff_tolerance": audited.get("Gram_negative_eigenvalue_roundoff_tolerance"),
                                     "source": "Independent audit metadata; not source NPZ read by reporter"})
            for scale, metrics in cell["rank_K_oracle_scores_same_mask"].items():
                if scale not in ("normalized", "physical_per_channel_diagnostic"):
                    continue
                scores.append({"dataset": dataset, "split": split, "metric_scale": scale, **metrics,
                               "MSE_is_minimum": scale == "normalized", "MAE_is_minimum": False,
                               "physical_MSE_is_minimum": False,
                               "scope": "Values of the normalized-MSE-optimal known-future rank-K solution"})
    return {"COMPARISON": rows, "ENERGY_CERTIFICATE": energy, "ORACLE_SCORE_VALUES": scores,
            "INDEPENDENT_RANK_SPECTRUM": spectrum}


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def make_figures(tables, audit, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")
    columns = [("known_future_rank_K_normalized_MSE_minimum", "Future-informed rank-K optimum", "#CC79A7"),
               ("PCA_fixed_D_real_normalized_MSE_minimum", "Existing PCA D: optimum", "#0072B2"),
               ("ForeCA_fixed_D_real_normalized_MSE_minimum", "Existing ForeCA D: optimum", "#D55E00"),
               ("Raw_normalized_MSE_same_complete_mask", "Raw Bolt: actual prediction", "#444444")]
    for ax, dataset in zip(axes, DATASETS):
        cells = [row for row in tables["COMPARISON"] if row["dataset"] == dataset]
        x = np.arange(len(cells))
        for index, (key, label, color) in enumerate(columns):
            values = [row[key] for row in cells]
            ax.bar(x + (index - 1.5)*.20, values, .19, label=label, color=color)
        ax.set_xticks(x, [row["split"].upper() for row in cells]); ax.set_ylabel("Normalized MSE, same complete-C mask")
        ax.set_title(dataset + f" (C={cells[0]['C']}, K={cells[0]['K']})")
        ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Diagnostic comparison: oracle geometry is not a forecasting leaderboard", fontsize=14, fontweight="bold")
    fig.supxlabel("Oracle directions use the known future of each split. Raw is the sole actual predictor shown; no oracle deployment/selection.", fontsize=9)
    save_figure(fig, output, "01_rank_K_and_fixed_D_diagnostics", plt)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for i, dataset in enumerate(DATASETS):
        cells = [row for row in tables["ENERGY_CERTIFICATE"] if row["dataset"] == dataset]
        for j, cell in enumerate(cells):
            ax = axes[i,j]
            terms = [term for term in tables["INDEPENDENT_RANK_SPECTRUM"] if term["dataset"] == dataset and term["split"] == cell["split"]]
            if terms:
                x = [term["direction_rank"] for term in terms]
                values = [term["innovation_energy_fraction"] for term in terms]
                ax.bar(x, values, color=["#0072B2" if term["retained_in_K"] else "#E69F00" for term in terms])
                ax.axvline(cell["K"] + .5, color="#444444", linestyle="--", linewidth=1)
                ax.set_xlabel("Direction rank: independent Gram eigensystem")
            else:
                ax.bar([0,1], [cell["retained_energy_fraction"],cell["discarded_energy_fraction"]], color=["#0072B2", "#E69F00"])
                ax.set_xticks([0,1], ["Retained K", "Discarded tail"])
                ax.set_xlabel("Per-component metadata absent: aggregate energy only")
            ax.set_ylim(0, 1); ax.set_ylabel("Innovation squared-energy fraction")
            ax.set_title(dataset + " / " + cell["split"].upper())
            ax.text(.98, .95, f"K={cell['K']}; retained {100*cell['retained_energy_fraction']:.2f}%\nNumerical rank={cell['numerical_rank_future_matrix']}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Known-future rank spectrum: retained K energy and discarded tail", fontsize=14, fontweight="bold")
    fig.supxlabel("Audit Gram eigenvalues (tiny negative roundoff clipped) or aggregate certificates; not temporal spectral entropy. Future-fitted per split.", fontsize=9)
    save_figure(fig, output, "02_future_rank_energy_certificate", plt)


def save_figure(fig, output, name, pyplot):
    for extension in ("png", "svg"):
        fig.savefig(output / "figures" / (name + "." + extension), dpi=180, bbox_inches="tight", facecolor="white")
    pyplot.close(fig)


def make_text(diagnostic, audit, tables, output, diagnostic_path, audit_path, main_report_path):
    path = output / "REPORT_KO.md"
    rows = tables["COMPARISON"]
    lower_than_raw = sum(row["known_future_rank_K_normalized_MSE_minimum"] < row["Raw_normalized_MSE_same_complete_mask"] for row in rows)
    lines = ["# 알려진 미래로 방향까지 최적화한 rank-K 진단: 고정 D 한계의 범위", "",
             "[확인] 이 보충 보고서는 새 배포 예측 방법의 성능 보고가 아니다. 이미 알려진 각 FIT/TUNE/개발 미래에서 방향까지 별도로 최적화한 강한 rank-K oracle를 검산했다. 실제 작은-K/Bolt·native 결과와 checkpoint 선택은 바꾸지 않았고, 보호 TEST나 새로운 독립 평가 자료도 사용하지 않았다.", "",
             "## 먼저 답할 질문", "",
             f"**고정 D의 실패가 K차원 선형 출력 전체의 불가능성을 의미하는가?** [확인] 같은 complete-target mask의6조건 중 {lower_than_raw}조건에서 미래를 이미 아는 rank-K 최소 normalized MSE가 실제 Raw Bolt 오차보다 낮았다. 반면 원 고정 D의 개발 최소는 Robin PCA/ForeCA와 Beijing ForeCA에서 Raw보다 높았다. 따라서 선택한 D의 제한을 모든 K차원 선형 공간의 불가능성으로 확대할 근거는 없다. 이는 어떤 배포 가능한 압축기가 Raw를 이겼다는 결과가 아니다.", "",
             "**그러면 작은 K 아이디어가 해결되었는가?** [확인] 해결되지 않았다. 이번 oracle는 방향과 각 시점의 좌표를 해당 구간의 실제 미래로 맞췄다. 특히 TUNE·VAL/EVAL에도 각자의 미래를 사용했고, FIT에서 정한 하나의 D를 이후 구간에 고정한 실험이 아니다. 과거만 받는 E가 그 좌표를 만들어 낼 수 있는지, 고정 Bolt가 필요한 미래 좌표를 예측할 수 있는지, 동일 정확도에서 전체 비용을 줄일지는 여전히 미확정이다.", "",
             "**어떤 문제가 분리되었는가?** [확인] 세 층을 구분할 근거가 생겼다. 첫째 실제 선택한 과거 재구성 D의 도달 가능한 부분공간, 둘째 미래를 이미 아는 경우 가능한 최선의 K차원 선형 부분공간, 셋째 실제 과거에서 예측되는 latent와 F의 오차다. 첫째와 둘째의 차이가 있어도 둘째를 셋째가 달성할 수 있다는 뜻은 아니다. [추정] 방향 선택·과거에서의 표현 구성·예측기 호환성·기간별 일반화가 서로 다른 병목일 수 있다.", "",
             "## 같은 mask의 실제 원수치", "",
             "Robin C17/K4·Beijing C11/K2, L512/H48와 원 FIT 통계는 기존 실험과 같다. all-C target이 관측된 위치만 쓰므로 같은 위치에서 모든 방법의 normalized macro MSE를 비교했다. Beijing의 complete-mask Raw는 full-mask Raw와 다르며 서로 바꾸어 쓰지 않는다. 중첩 창과 같은 자료의 FIT/TUNE/개발을 독립 데이터셋이나 반복 실험으로 세지 않는다.", "",
             "```text", "dataset                split complete/total  rank-K oracle   PCA D minimum   ForeCA D minimum  Raw same-mask"]
    for row in rows:
        count = str(row["complete_target_positions"]) + "/" + str(row["total_target_positions"])
        lines.append(f"{row['dataset']:<23}{row['split']:<6}{count:<16}{row['known_future_rank_K_normalized_MSE_minimum']:<16.9f}{row['PCA_fixed_D_real_normalized_MSE_minimum']:<16.9f}{row['ForeCA_fixed_D_real_normalized_MSE_minimum']:<18.9f}{row['Raw_normalized_MSE_same_complete_mask']:.9f}")
    lines.extend(["```", "", "[6조건 비교 CSV](tables/COMPARISON.csv). 이 표는 진단의 제한 범위를 비교하며 oracle를 배포 모델 순위에 넣지 않는다. Raw는 고정 D의 rank-K 부분공간에 속해야 하는 모델이 아니므로 Raw에 하한 부등식을 적용하지 않는다. oracle가 Raw보다 작은 것은 이 자료에서의 실제 관찰이지 Raw가 항상 더 나빠야 한다는 수학적 정리가 아니다.", "",
                  "![rank-K와 고정 D 진단](figures/01_rank_K_and_fixed_D_diagnostics.png)", "",
                  "## 어떤 수학 문제를 풀었는가", "",
                  "완전 관측된 미래 위치들을 M×C 행렬로 모으고, 각 행은 FIT로 정규화한 `실제 미래 C − 그 창의 현재 C` innovation으로 둔다. 미래 평균을 새로 빼거나 미래 intercept를 추가하지 않는다. 이 알려진 미래 행렬의 rank-K 선형 근사를 구하며 방향은 해당 split 안 모든 실제 미래 위치에 공통이다. 다른 split에는 다른 방향을 적합한다.", "",
                  "실수 산술에서 상위 K singular 방향을 남기면 버린 squared singular-value 에너지가 최소 재구성 오차이고, 이를 M×C로 나누면 같은 complete-mask normalized channel-macro MSE다. 이때 시점별 latent 좌표와 출력 방향 모두 이미 알려진 미래를 활용한다. 어느 미래 시점의 좌표가 과거만으로 예측 가능한지 제한하지 않으므로 실제 방법보다 훨씬 강한 진단이다. 이를 causal representation learning, forecasting improvement 또는 future-free inference라고 쓰면 안 된다.", "",
                  "[에너지·직교성·SVD/Gram 인증 CSV](tables/ENERGY_CERTIFICATE.csv)는 total/retained/discarded 에너지와 residual 에너지, Gram trace/tail, M×C 분모, 사전 수치 허용 오차를 남긴다. [독립 Gram 고유값 원수치](tables/INDEPENDENT_RANK_SPECTRUM.csv)가 제공된 경우에만 방향별 에너지 spectrum을 그린다. 그 값은 별도 감사의 JSON metadata이며 보고서 생성기는 NPZ를 열지 않았다. 허용 범위 안 작은 음수 반올림은 원값과0으로 clip한 표시값을 모두 저장한다. 수열이 없으면 retained/discarded aggregate만 표시한다. 시간 주파수의 spectral entropy와 rank singular spectrum도 구분한다.", "",
                  "![알려진 미래 에너지 분해](figures/02_future_rank_energy_certificate.png)", "",
                  "**최소화한 것은 normalized MSE뿐이다.** [oracle 해의 모든 지표](tables/ORACLE_SCORE_VALUES.csv)에 MAE와 원단위 오차도 저장했지만 이는 그 MSE 최소해의 점수다. MAE의 최솟값이나 서로 다른 물리 단위 MSE의 최솟값으로 표현하지 않는다. 또한 이는 현재 C anchor를 허용한 static rank-K 선형 출력의 진단이며 모든 비선형/가변 decoder의 최적 하한이 아니다.", "",
                  "## 고정 D 진단과 결론이 달라진 이유", "",
                  "원 고정-D 진단은 실제 배포 D32를 그대로 놓고 미래 좌표만 자유롭게 정했다. 그래서 그 D를 유지한 F/head의 실수 산술 normalized MSE에는 도달 가능한 부분공간의 제한을 적용할 수 있었다. 이번 진단은 출력 방향 자체까지 실제 미래에서 최적화한다. 따라서 두 숫자의 차이는 다른 질문을 풀어서 생긴 것이며 기존 감사·Raw 예측·선택을 뒤집거나 오류로 바꾸는 결과가 아니다.", "",
                  "[확인] Robin 개발에서 PCA D 최소는 .396541619, ForeCA D 최소는 .461340297이며 Raw는 .340401887이다. 그런데 같은 미래로 방향까지 맞춘 K4 oracle는 .117363274다. Beijing 개발의 PCA D 최소는 .511797660, ForeCA D 최소는 .801825480, 같은 complete-mask Raw는 .598972295이고 K2 oracle는 .349213093이다. 이 결과는 선택된 D의 방향과 자유로운 미래 방향을 구분해야 함을 보여 준다. 미래를 알고 얻은 격차를 실제 가능한 예측 개선량으로 약속할 수 없다.", "",
                  "[추정] 과거 입력 재구성의 중요한 방향과 미래 innovation의 중요한 방향이 다를 수 있다. 하지만 이것이 이전 모든 실패를 재구성 loss 하나로 설명한다는 뜻은 아니다. 기존 ANCHOR_POINTWISE와 MAIN 계열에는 실제-Y 예측 목적 학습·보정도 있었다. 이번 static PCA/ForeCA의 방향 제한과 그 기존 결과는 조건·초기화·head·우회 경로가 달라 구분해서 읽어야 한다.", "",
                  "## 다음 판단과 가능한 기여의 범위", "",
                  "[판단] 다음 최소 질문은 미래에 맞춘 방향이 FIT의 허용된 label과 과거만으로 학습되어 이후 구간에도 유지되는지다. 목표를 Raw보다 낮은 oracle 수치로 바꾸거나 DEV의 방향을 실제 encoder/decoder에 그대로 넣어서는 안 된다. 하나의 학습된 출력 공간을 고정하고 동일 E/D·예측기 대조·실제 전처리와 반환 비용으로 검증해야 한다. 잘 나온 split을 골라 대표 성능으로 발표하는 것도 독립적인 일반화 근거를 만들지 못한다.", "",
                  "[추정·기여 후보] 연구 문제를 ‘작은 K가 존재하는가’와 ‘과거에서 구축한 K표현이 그 미래 정보를 보존하고 고정 Bolt로 예측 가능한가’를 구분하는 진단 구조로 좁힐 수 있다. 고정 D의 실패 경계를 직접 검산하고, 미래-informed 공간의 가능성과 실제 과거-only 예측 사이의 간극을 수치로 남긴 것이 현재의 근거다. SVD 최소화 자체는 표준 선형대수이며 새로운 방법론이나 논문 기여가 확보되었다고 선언하지 않는다. 새로운 압축 방법의 이득·신규성·독립 평가가 추가로 필요하다.", "",
                  "현재 과거-only PCA/ForeCA/Bolt/native의 정확도·비용 판정은 그대로다. 이 추가 진단은 classifier/graph/contrastive/대형 backbone을 새로 학습하지 않았고, 모델·checkpoint·방법·분기 선택·원 예측·비용 순위를 바꾸지 않았다. 실제 미래를 이용한 행렬 분해6개를 ‘배포 학습 성공6회’로 세지 않는다. 자원 사용과 보호 자산 상태는 각 source/receipt에 별도로 남는다.", "",
                  "## 검산과 파일 계보", "",
                  f"[실제 rank-K diagnostic]({local_link(path, diagnostic_path)}) · [독립 Gram/eigensystem 감사]({local_link(path, audit_path)}) · [앞선 실제 예측·고정 D 보고서]({local_link(path, main_report_path)}) · [source와 SHA](PROVENANCE.json) · [그림·CSV SHA](MANIFEST.json).", "",
                  "[확인] 이 보고서는 COMPLETE diagnostic과 독립 PASS audit의 exact SHA, 원 producer·projection audit·fixed-span diagnostic·fixed-span audit의 현재 JSON SHA를 모두 확인했다. 독립 수학 검산의 PASS를 실제 forecaster 재학습이나 새로운 독립 TEST 성능으로 확대하지 않는다. 이 생성기는 terminal JSON metadata만 읽었고 원 NPZ·실제 모델 가중치·checkpoint·TEST를 열지 않았다. 기존 보고서·generator·source·선택·그림을 수정하지 않고 별도 보충 산출물로 저장했다.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build(diagnostic_path, audit_path, output_path, main_report_path):
    diagnostic, audit = load_verified(diagnostic_path, audit_path)
    main_report_path = Path(main_report_path).resolve()
    if not main_report_path.is_file() or main_report_path.suffix.lower() != ".md":
        raise ValueError("EXISTING_MATCHED_MAIN_REPORT_REQUIRED")
    output = Path(output_path).resolve()
    if output.exists(): raise FileExistsError("NEW_RANK_REPORT_OUTPUT_REQUIRED")
    tables = collect_tables(diagnostic, audit)
    output.mkdir(parents=True, exist_ok=False); (output / "tables").mkdir(); (output / "figures").mkdir()
    for name, rows in tables.items(): write_csv(output / "tables" / (name + ".csv"), rows)
    make_figures(tables, audit, output)
    make_text(diagnostic, audit, tables, output, diagnostic_path, audit_path, main_report_path)
    provenance = {"diagnostic": record(diagnostic_path), "independent_audit": record(audit_path),
                  "existing_main_report": record(main_report_path), "generator": record(__file__),
                  "bindings": {key: diagnostic[key] for key in ("producer", "projection_audit", "fixed_span", "fixed_span_audit")},
                  "metadata_only": True, "NPZ_models_checkpoints_TEST_opened": False,
                  "original_forecasts_and_selection_unchanged": True}
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = [{"path": file.relative_to(output).as_posix(), "bytes": file.stat().st_size, "sha256": sha(file)}
             for file in sorted(output.rglob("*")) if file.is_file()]
    manifest = {"status": "GENERATED_FROM_AUDITED_FUTURE_INFORMED_DIAGNOSTIC", "files": files,
                "diagnostic_sha256": sha(diagnostic_path), "audit_sha256": sha(audit_path),
                "csv_rows": {key: len(rows) for key, rows in tables.items()}, "PNG_count": 2, "SVG_count": 2,
                "new_deployment_training_or_inference": False, "independent_holdout": False}
    (output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True); parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--main-report", required=True)
    args = parser.parse_args()
    manifest = build(args.diagnostic, args.audit, args.output, args.main_report)
    print(json.dumps({"status": manifest["status"], "csv_rows": manifest["csv_rows"], "PNG": 2, "SVG": 2}))
