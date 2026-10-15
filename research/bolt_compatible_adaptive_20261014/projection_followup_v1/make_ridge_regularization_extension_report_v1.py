"""Metadata-only report of the adaptive ridge upper-grid follow-up."""
from pathlib import Path
import argparse
import json
import math

from make_linear_e_adaptation_report_v2 import require, resolve, record, read, bound, rel, write_csv, sha

HERE = Path(__file__).resolve().parent
RESULT_SHA = "8fcd2e19fcfdf9f9a0b261553cf8179de129b813a3fef92039c86de415c689b3"
AUDIT_SHA = "80e12795739ff022b9cb02c2f762747d4b208bca3cf0068f38c13acc3afe3c15"
AUDITOR_SOURCE = {"path":str(HERE/"audit_ridge_regularization_extension_v1.py"),"sha256":"145c7e18c7d10a75942a670667f8d9c63d26a5517b4fa964e26678842e9a7bcf"}


def load_verified(result_path, audit_path):
    require(isinstance(AUDIT_SHA, str) and len(AUDIT_SHA) == 64 and isinstance(AUDITOR_SOURCE, dict), "ACTUAL_EXTENSION_AUDIT_NOT_REGISTERED")
    result, audit = read(result_path), read(audit_path)
    references = {"result": record(result_path), "audit": record(audit_path)}
    require(references["result"]["sha256"] == RESULT_SHA and references["audit"]["sha256"] == AUDIT_SHA and result["status"] == "COMPLETE" and audit["status"] == "PASS" and not result["failed_paths"], "EXACT_COMPLETE_EXTENSION_AND_AUDIT")
    bound(audit["diagnostic"], references["result"])
    bound(audit["audit_source"], AUDITOR_SOURCE)
    require(result["extension_fits_completed"] == 10 and result["new_positive_alpha_candidates_completed"] == 30 and result["new_zero_slope_candidates_completed"] == 20 and result["new_FIT_TUNE_prediction_evaluations_completed"] == 100 and result["new_selected_DEV_prediction_evaluations"] + result["reused_original_DEV_prediction_evaluations"] == 10 and result["all_10_locks_before_DEV"] is True, "TEN_FITS_FIFTY_NEW_CANDIDATES_AND_LOCKED_DEV")
    original = read(bound(result["original_ridge_diagnostic"]))
    references["original"] = result["original_ridge_diagnostic"]
    references["original_audit"] = result["original_ridge_audit"]
    for value in (result, audit):
        receipt = value["resource_terminal"]
        require(read(bound(receipt)) == {k: v for k, v in receipt.items() if k not in ("path", "sha256")}, "EXACT_CANONICAL_RESOURCE_RECEIPT")
        require(all(type(receipt[k]) in (int, float) and math.isfinite(receipt[k]) and receipt[k] >= 0 for k in ("process_cpu_seconds", "wall_seconds")), "CANONICAL_CPU_WALL_FIELDS")
    return result, original, references


def tables(result, original):
    candidates, selected = [], []
    for data, group in result["datasets"].items():
        for route, row in group["routes"].items():
            old = original["datasets"][data]["routes"][route]
            for candidate in row["candidates"]:
                candidates.append({"dataset": data, "route": route, "candidate": candidate["candidate_id"], "kind": candidate["kind"], "alpha": candidate["alpha"], "new_in_extension": candidate["new_in_this_extension"], "selected": candidate["candidate_id"] == row["selected_candidate"], "FIT_MSE": candidate["FIT"]["scores_all_observed"]["normalized"]["mse"], "TUNE_MSE": candidate["TUNE_MSE"], "prediction_artifact": candidate["prediction_artifact"], "coefficient_value_sha256": candidate["coefficient_value_sha256"]})
            selected.append({"dataset": data, "route": route, "old_alpha": old["selected_alpha"], "new_candidate": row["selected_candidate"], "new_alpha": row["selected_alpha"], "old_TUNE_MSE": old["selected_TUNE_MSE"], "new_TUNE_MSE": row["selected_TUNE_MSE"], "old_DEV_MSE": old["DEV"]["scores_all_observed"]["normalized"]["mse"], "new_DEV_MSE": row["DEV"]["scores_all_observed"]["normalized"]["mse"], "original_candidate_reselected": row["selected_original_candidate"], "new_DEV_inference": row["DEV"]["new_DEV_inference"], "new_upper_boundary": row["selected_new_alpha_upper_boundary"], "zero_slope_selected": row["zero_slope_selected"], "selected_coefficients": row["selected_coefficients"], "selection_lock": row["selection_lock"], "DEV_artifact": row["DEV"]["artifact"]})
    require(len(candidates) == 120 and len(selected) == 10, "COMPLETE_EXTENSION_TABLE_COUNTS")
    return candidates, selected


def figures(candidates, selected, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = output / "figures"
    folder.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, data in zip(axes, ("robin", "beijing_aotizhongxin")):
        for row in [r for r in selected if r["dataset"] == data]:
            values = [c for c in candidates if c["dataset"] == data and c["route"] == row["route"]]
            ax.plot(range(12), [c["TUNE_MSE"] for c in values], "o-", label=row["route"].replace("_HISTORY512", "").replace("CROSS_MOMENT", "CROSS"))
            index = next(i for i, value in enumerate(values) if value["selected"])
            ax.scatter(index, values[index]["TUNE_MSE"], s=90, facecolors="none", edgecolors="black", zorder=4)
        ax.axvline(6.5, linestyle=":", color="gray")
        ax.set_xticks(range(12), ["0", "1e-4", ".001", ".01", ".1", "1", "10", "100", "1k", "10k", "mean", "anchor"], rotation=40)
        ax.set_yscale("log")
        ax.set_title(data + " / own TUNE")
        ax.set_ylabel("All-observed normalized MSE (log)")
        ax.set_xlabel("Original seven | adaptive three + two zero-slope controls")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle("Ridge boundary follow-up: circle=selected; mean/anchor retain current C without history slopes")
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ("png", "svg"):
        fig.savefig(folder / ("01_extended_TUNE_menu." + extension), dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data in zip(axes, ("robin", "beijing_aotizhongxin")):
        rows = [r for r in selected if r["dataset"] == data]
        positions = np.arange(len(rows))
        ax.bar(positions-.18, [r["old_DEV_MSE"] for r in rows], .36, label="Original alpha<=10")
        ax.bar(positions+.18, [r["new_DEV_MSE"] for r in rows], .36, label="Extended own-TUNE selection")
        ax.set_xticks(positions, [r["route"].replace("_HISTORY512", "").replace("CROSS_MOMENT", "CROSS") for r in rows], rotation=25)
        ax.set_title(data + " / already exposed DEV")
        ax.set_ylabel("All-observed normalized MSE")
        ax.grid(axis="y", alpha=.18)
        ax.legend(fontsize=9)
    fig.suptitle("Different selected models require separate cost; DEV does not select alpha")
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ("png", "svg"):
        fig.savefig(folder / ("02_selected_DEV_before_after." + extension), dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate(result_path, audit_path, destination):
    output = resolve(destination)
    require(output.parent == (HERE / "reports").resolve() and not output.exists(), "FRESH_EXTENSION_REPORT_OUTPUT")
    result, original, references = load_verified(result_path, audit_path)
    candidates, selected = tables(result, original)
    source = record(__file__)
    output.mkdir()
    (output / "tables").mkdir()
    write_csv(output / "tables/CANDIDATES.csv", candidates, output)
    write_csv(output / "tables/SELECTED.csv", selected, output)
    figures(candidates, selected, output)
    changed = sum(not row["original_candidate_reselected"] for row in selected)
    lines = ["# ridge 탐색 경계의 후속 확인", "", f"[확인] 원 history10경로가 모두 alpha10 상단을 골라 alpha100/1000/10000과 mean/현재C 제어를 추가했다. {changed}/10조건이 새 후보를 골랐고 새 alpha10000 상단 선택은 {sum(row['new_upper_boundary'] for row in selected)}/10, zero-slope 선택은 {sum(row['zero_slope_selected'] for row in selected)}/10이다. 정규화를 늘리면 모든 실패가 해결된다는 설명은 지지되지 않는다.", "", "Robin5경로는 원 alpha10을 유지하고 Beijing5경로는100을 골랐다. Bay5의 선정 변화는 alpha≤10의 결론을 더 넓은 규제 범위로 확대할 수 없음을 확인한다. 유한한 메뉴의 내부 선택을 전역 최적 규제로 확대하지 않는다. 같은 노출 DEV를 사용한 적응적 후속이며 독립 일반화 증거가 아니다.", "", "![TUNE 전체 메뉴](figures/01_extended_TUNE_menu.png)", ""]
    for row in selected:
        lines.append(f"- {row['dataset']} {row['route']}: 선정 {row['new_candidate']}, TUNE {row['old_TUNE_MSE']:.12g} → {row['new_TUNE_MSE']:.12g}, DEV {row['old_DEV_MSE']:.12g} → {row['new_DEV_MSE']:.12g}.")
    lines += ["", "![선정 DEV 변화](figures/02_selected_DEV_before_after.png)", "", "새10fit·30양수ridge·20zero-slope·100 FIT/TUNE 평가를 실제 수행했다. 원70후보·140 FIT/TUNE는 immutable binding으로 재사용했고, 모든10잠금 뒤 새 DEV5회와 원 DEV5개 exact 재사용을 구분했다. 원 두 Raw current-C fit은 참고만 재사용했다. 각 경로는 같은 원 과거/E·true-observed mask·전FIT RMS·비벌점 intercept·FP32 feature subtraction·current-C anchor와 CPU canonical32 fused readout을 쓴다.", "", "mean은 각(h,c)의 관측 FIT innovation 평균만 더하고 anchor는 현재C를 반환한다. 이 두 제어는 원 current-C anchor를 유지하지만 전체512과거에 대한 학습 slope와 E/F는 필요 없어 압축 효과로 세지 않는다. 새 F/model/optimizer/GPU 학습0, 보호 TEST 접근0이다. 선정되지 않은 새 계수의 값이 저장되지 않은 경우 모든 후보 fit을 독립 재현했다고 주장하지 않는다.", "", "**비용은 미측정이다.** 기존76셀은 원 v1 선정 계수의 실제 비용으로 유지한다. 변경된 Beijing5계수에 그 비용을 대입하지 않는다. PCA/native/F 대조는 [원 비교 보고](../new_direction_comparison_v1/REPORT_KO.md)에서 확인할 수 있다.", "", "- [후보120행](tables/CANDIDATES.csv)", "- [선정10행](tables/SELECTED.csv)", ""]
    lines += [f"- [{key}]({rel(value['path'], output)}): `{value['sha256']}`" for key, value in references.items()]
    (output / "REPORT_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    require(record(__file__) == source, "REPORT_SOURCE_UNCHANGED")
    for value in references.values():
        bound(value)
    manifest = {"status": "COMPLETE", "source": source, "inputs": references, "files": [{"path": path.relative_to(output).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()], "table_row_counts": {"CANDIDATES": 120, "SELECTED": 10}, "PNG_count": 2, "SVG_count": 2, "scientific_NPZ_or_model_read": False, "protected_TEST_accessed": False, "independent_holdout": False, "whole_extension_cost_measured": False}
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "COMPLETE", "manifest": str(output / "MANIFEST.json")}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("result", "audit", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    generate(args.result, args.audit, args.output)
