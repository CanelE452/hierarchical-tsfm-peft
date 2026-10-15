"""Audited JSON-only report of cached past gates and actual F-only whole costs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
METHODS = ("RAW_BOLT", "PCA_PINV_BOLT", "PCA_PINV_NATIVE_s92601", "PCA_PINV_NATIVE_s92602",
           "FIT_TARGET_PINV_BOLT", "FIT_TARGET_PINV_NATIVE_s92601", "FIT_TARGET_PINV_NATIVE_s92602")
ROUTES = ("RAW_BOLT", "RAW_BOLT_GATED", "PCA_PINV_BOLT", "PCA_PINV_BOLT_GATED", "FIT_TARGET_PINV_BOLT", "FIT_TARGET_PINV_BOLT_GATED")
SCALES = ("normalized", "physical_per_channel_diagnostic")
DISPLAY = {"robin": "Robin", "beijing_aotizhongxin": "Beijing Aotizhongxin"}
LABELS = ("Raw Bolt", "PCA / Bolt", "PCA / native s1", "PCA / native s2", "FIT target / Bolt", "FIT target / native s1", "FIT target / native s2")


def require(value, label):
    if not value: raise ValueError(label)


def sha(path):
    with Path(path).open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read(path):
    require(Path(path).suffix.lower() == ".json", "JSON_METADATA_ONLY")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def bind(item, path, label):
    path = Path(path).resolve()
    require(path.suffix.lower() == ".json" and resolve(item["path"]) == path and item["sha256"] == sha(path), "EXACT_METADATA_BINDING:" + label)


def load_verified(cpu_path, cpu_audit_path, cost_path, cost_audit_path, identity_path):
    cpu, audit, cost, cost_audit, identity = (read(path) for path in (cpu_path, cpu_audit_path, cost_path, cost_audit_path, identity_path))
    require(cpu.get("status") == cost.get("status") == "COMPLETE" and audit.get("status") == cost_audit.get("status") == "PASS", "BOTH_COMPLETE_INDEPENDENT_PASS_REQUIRED")
    bind(audit["diagnostic"], cpu_path, "CPU audit")
    bind(cost_audit["cost"], cost_path, "whole cost audit")
    bind(cost["gate_diagnostic"], cpu_path, "cost CPU source"); bind(cost["gate_audit"], cpu_audit_path, "cost CPU audit")
    for key, cpu_key in (("producer", "producer"), ("fit_target_audit", "independent_FIT_target_audit")):
        require(cost[key] == cost_audit[key] == audit[key] == cpu[cpu_key], "SAME_FIXED_FIT_SOURCE:" + key)
        bind(cost[key], resolve(cpu[cpu_key]["path"]), key)
    require(cost_audit["gate_diagnostic"] == cost["gate_diagnostic"] and cost_audit["gate_audit"] == cost["gate_audit"], "SAME_CPU_GATE_CHAIN")
    require(all(audit["totals"].get(key) == value for key, value in {"dataset_split_cells_checked": 6, "method_pairs_checked": 42,
        "ungated_prediction_arrays_checked": 42, "gated_prediction_arrays_checked": 42, "score_field_checks": 2016}.items()), "FULL_CPU_GATE_AUDIT_COVERAGE")
    require(all(cost_audit["totals"].get(key) == value for key, value in {"cost_cells_checked": 24, "recorded_timer_samples_checked": 144,
        "physical_gate_prechecks_checked": 2, "paired_gate_timed_output_arrays_checked": 72, "original_selected_forecast_hashes_checked": 36}.items()), "FULL_WHOLE_COST_AUDIT_COVERAGE")
    require(cpu["new_training"] is False and cpu["new_encoder_forecaster_or_head_forward"] is False
            and cpu["E_F_native_head_forward_runs"] == 0 and cpu["future_labels_used_for_gate"] is False and cpu["method_selection_used"] is False,
            "PAST_ONLY_CACHED_OUTPUT_INTERVENTION")
    require(cpu["past_C_condition_metadata_added"] is True and cpu["same_decoder_side_information_as_ungated"] is False
            and cpu["static_original_fixed_D_floor_reused_as_gated_lower_bound"] is False, "CHANGED_DECODER_INTERFACE_AND_FLOOR_SCOPE")
    require(cost["new_training_updates"] == cost["projection_fits"] == 0 and cost["new_native_head_forward"] is False
            and cost["native_deployment_cost_measured"] is False and tuple(cost["routes"]) == ROUTES
            and cost["F_before"] == cost["F_after"], "EXACT_SIX_F_ONLY_COST_ROUTES")
    for data in (cpu, audit, cost, cost_audit): require(data["protected_TEST_accessed"] is False, "NO_PROTECTED_TEST")
    require(cpu["source_cache_selection_SHA_unchanged"] is True and cost["source_cache_selection_SHA_unchanged"] is True, "SOURCE_PRESERVATION")
    require(identity["status"] == "CONFIRMED_AUDITOR_REGISTERED_SEED_LITERAL_ERROR_REPAIRED"
            and identity["cost_experiment_reexecuted"] is False and identity["F_or_native_head_called"] is False
            and identity["original_failed_source_audit_and_receipt_preserved"] is True
            and identity["source_difference"]["exactly_one_seed_guard_literal_restoration_verified"] is True
            and identity["source_difference"]["prediction_metric_tolerance_counts_and_cost_measurement_unchanged"] is True, "PRESERVED_AUDITOR_ONLY_REPAIR_RECEIPT")
    bind(identity["fresh_audit"], cost_audit_path, "accepted cost v2 audit")
    bind(identity["cost_producer"], cost_path, "same cost, no rerun")
    bind(identity["failed_audit"], resolve(identity["failed_audit"]["path"]), "preserved failed cost audit")
    for dataset in DATASETS:
        pre = cost["datasets"][dataset]["physical_gate_precheck"]
        require(pre["status"] == "PASS" and pre["all_three_gate_arrays_byte_exact"] is True
                and pre["tolerance_parity_used_to_accept_gate_decision"] is False, "ACTUAL_PHYSICAL_PREFLIGHT_BYTE_EXACT")
    return cpu, audit, cost, cost_audit, identity


def collect_tables(cpu, audit, cost, cost_audit, identity):
    out = {key: [] for key in ("SCORES", "SIGNED_EFFECT", "GATE_CHANNELS", "GATE_CASES", "GATE_METHODS",
        "WHOLE_COST", "WHOLE_COST_SAMPLES", "PHYSICAL_PREFLIGHT", "REGISTERED_ADOPTION", "AUDIT_REPAIR_PROVENANCE")}
    arrays = ("exact_flat_windows_per_channel", "gate_eligible_windows_per_channel", "eligible_windows_with_any_observed_future_per_channel",
        "eligible_windows_with_observed_future_change_per_channel", "eligible_observed_future_positions_per_channel", "eligible_observed_future_changed_positions_per_channel")
    for dataset, (channels, k, dev) in DATASETS.items():
        for split in ("fit", "tune", dev):
            cell = cpu["datasets"][dataset]["splits"][split]
            require(set(cell["methods"]) == set(METHODS), "EXACT_SEVEN_CACHED_METHOD_PAIRS")
            base = {"dataset": dataset, "split": split, "C": channels, "K": k,
                "windows": cell["windows"], "complete_target_positions": cell["complete_target_positions"], "total_target_positions": cell["total_target_positions"]}
            out["GATE_CASES"].append({**base, "eligible_window_channel_pairs": sum(cell["gate_eligible_windows_per_channel"]),
                "eligible_window_channel_pairs_with_observed_future_change": sum(cell["eligible_windows_with_observed_future_change_per_channel"]),
                "input_source": cell["input_source"], "artifact": cell["artifact"], "cached_prediction_sources": cell["cached_prediction_sources"],
                "future_change_diagnostic_scope": cell["future_change_diagnostic_scope"], "independent_sample_count_claimed": False})
            for channel in range(channels):
                out["GATE_CHANNELS"].append({**base, "channel_index_zero_based": channel, **{key: cell[key][channel] for key in arrays},
                    "future_changes_used_for_gate": False, "past_flatness_guarantees_future_flatness": False})
            for method in METHODS:
                values = cell["methods"][method]
                out["GATE_METHODS"].append({**base, "method": method,
                    "unchanged_coordinates_byte_exact": values["unchanged_coordinates_byte_exact"],
                    "gated_coordinates_current_anchor_byte_exact": values["gated_coordinates_current_anchor_byte_exact"],
                    "changed_prediction_coordinates_per_channel": values["changed_prediction_coordinates_per_channel"],
                    "native_head_frozen_intervention": values["native_head_frozen_intervention"], "optimized_gated_native_head_claimed": False})
                for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                    for scale in SCALES:
                        old, new = values["ungated_scores_" + suffix][scale], values["gated_scores_" + suffix][scale]
                        for variant, scores in (("UNGATED", old), ("GATED", new)):
                            out["SCORES"].append({**base, "method": method, "variant": variant, "mask": mask,
                                "metric_scale": scale, **scores, "role": "cached_native_frozen_intervention" if "NATIVE" in method else "cached_F_output_intervention",
                                "new_E_F_head_forward": False, "physical_units_comparable": False})
                        out["SIGNED_EFFECT"].append({**base, "method": method, "mask": mask, "metric_scale": scale,
                            "ungated_MSE": old["mse"], "gated_MSE": new["mse"], "gated_minus_ungated_MSE": new["mse"] - old["mse"],
                            "ungated_MAE": old["mae"], "gated_MAE": new["mae"], "gated_minus_ungated_MAE": new["mae"] - old["mae"],
                            "observed_counts": old["observed_counts"], "statistical_improvement_claimed": False})
        pre = cost["datasets"][dataset]["physical_gate_precheck"]
        independent = cost_audit["datasets"][dataset]["physical_gate_precheck"]
        out["PHYSICAL_PREFLIGHT"].append({"dataset": dataset, "actual_precheck": pre, "independent_precheck": independent,
            "eligible_window_channel_pairs_first8": independent["physical_gate_eligible_window_channel_pairs_first8"],
            "gate_output_change_possible_in_timed_prefix": independent["physical_gate_eligible_window_channel_pairs_first8"] > 0,
            "exclusive_method_memory_measured": False})
        rows = cost["cost"][dataset]["rows"]
        require(len(rows) == 12 and {(row["route"], row["batch"]) for row in rows} == {(route, batch) for route in ROUTES for batch in (1, 8)}, "TWELVE_ACTUAL_COST_CELLS")
        for row in rows:
            raw = next(value for value in rows if value["route"] == "RAW_BOLT" and value["batch"] == row["batch"])
            out["WHOLE_COST"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": row["repeats"],
                "median_seconds": row["median_seconds"], "median_seconds_per_window": row["median_seconds_per_window"],
                "ratio_of_RAW_medians": row["median_seconds"] / raw["median_seconds"], "median_paired_RAW_ratio": row["paired_RAW_median_ratio"],
                "seconds": row["seconds"], "origins": row["origins"], "memory_shared_process": row["memory"],
                "includes": row["whole_includes"], "excludes": row["whole_excludes"], "memory_scope": row["memory_scope"],
                "selected_metadata": row["selected_metadata"], "timed_outputs": row["timed_outputs"], "reference": row["reference"],
                "native_timed": False, "full_six_rotation_Latin": True, "gate_caused_speed_gain_claimed": False})
            for index, sample in enumerate(row["samples"]):
                out["WHOLE_COST_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"],
                    "repeat": sample["repeat"], "position": sample["position"], "seconds": sample["seconds"],
                    "paired_RAW_ratio": row["paired_RAW_ratios"][index], "sample": sample})
        for route in ROUTES[1:]:
            gated = route.endswith("_GATED"); method = route[:-6] if gated else route
            metrics = cpu["datasets"][dataset]["splits"]["tune"]["methods"][method][("gated" if gated else "ungated") + "_scores_all_observed"]["normalized"]
            raw = cpu["datasets"][dataset]["splits"]["tune"]["methods"]["RAW_BOLT"]["ungated_scores_all_observed"]["normalized"]
            timing = next(row for row in out["WHOLE_COST"] if row["dataset"] == dataset and row["route"] == route and row["batch"] == 8)
            mse, mae, latency = metrics["mse"] / raw["mse"], metrics["mae"] / raw["mae"], timing["ratio_of_RAW_medians"]
            out["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "TUNE_MSE_ratio_to_ungated_RAW": mse,
                "TUNE_MAE_ratio_to_ungated_RAW": mae, "B8_ratio_of_ungated_RAW_medians": latency,
                "all_registered_conditions_pass": mse <= 1.01 and mae <= 1.01 and latency <= .9,
                "actual_method_selection_used": False, "gated_RAW_control_included": True})
    out["AUDIT_REPAIR_PROVENANCE"].append({"failed_audit": identity["failed_audit"], "fresh_audit": identity["fresh_audit"],
        "failed_audit_source": identity["failed_audit_source"], "fresh_audit_source": identity["fresh_audit_source"],
        "registered_producer_seed": identity["registered_producer_seed"], "incorrect_v1_auditor_seed_literal": identity["incorrect_v1_auditor_seed_literal"],
        "source_difference": identity["source_difference"], "regression_tests": identity["regression_tests"],
        "cost_experiment_reexecuted": False, "F_or_native_head_called_in_repair": False, "original_failed_artifacts_preserved": True})
    require(len(out["SCORES"]) == 336 and len(out["SIGNED_EFFECT"]) == 168 and len(out["GATE_CHANNELS"]) == 84
            and len(out["WHOLE_COST"]) == 24 and len(out["WHOLE_COST_SAMPLES"]) == 144, "FULL_REPORT_COHORT")
    return out


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def choose(rows, **where):
    return next(row for row in rows if all(row[key] == value for key, value in where.items()))


def rel(path, output):
    return Path(os.path.relpath(Path(path).resolve(), Path(output).resolve())).as_posix()


def figures(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    folder = Path(output) / "figures"; folder.mkdir(exist_ok=False); paths = []
    def save(fig, stem):
        for suffix in ("png", "svg"):
            path = folder / (stem + "." + suffix); fig.savefig(path, dpi=165 if suffix == "png" else None, bbox_inches="tight"); paths.append(path)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("tune", dev)):
            ax = axes[di, si]; maximum = 0
            for variant, offset, color in (("UNGATED", -.18, "#888888"), ("GATED", .18, "#0072B2")):
                values = [choose(tables["SCORES"], dataset=dataset, split=split, method=method, variant=variant, mask="all_original_observed", metric_scale="normalized")["mse"] for method in METHODS]
                ax.barh(np.arange(7) + offset, values, height=.33, color=color, label=variant)
                maximum = max(maximum, max(values))
                for index, value in enumerate(values): ax.text(value, index + offset, f" {value:.4f}", va="center", fontsize=7)
            ax.set_xlim(0, maximum * 1.24); ax.set_yticks(np.arange(7), LABELS); ax.invert_yaxis()
            ax.set_xlabel("Normalized macro MSE: all original observed targets"); ax.set_title(f"{DISPLAY[dataset]} / {split.upper()}"); ax.grid(axis="x", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Same cached forecasts: past-constant coordinate replacement (native heads remain frozen)", fontsize=13)
    save(fig, "01_absolute_scores")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("tune", dev)):
            ax = axes[di, si]
            values = [choose(tables["SIGNED_EFFECT"], dataset=dataset, split=split, method=method, mask="all_original_observed", metric_scale="normalized")["gated_minus_ungated_MSE"] for method in METHODS]
            ax.barh(np.arange(7), values, color=["#0072B2" if value < 0 else "#D55E00" for value in values]); ax.axvline(0, color="#555555", linewidth=.8)
            ax.set_yticks(np.arange(7), LABELS); ax.invert_yaxis(); ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.set_title(f"{DISPLAY[dataset]} / {split.upper()} / min {min(values):.3e}, max {max(values):.3e}")
            ax.set_xlabel("Gated MSE - ungated MSE (negative: smaller error)"); ax.grid(axis="x", alpha=.2)
    fig.suptitle("Signed arithmetic effects, including tiny rounding changes; no statistical significance claim", fontsize=12)
    save(fig, "02_signed_gate_effect")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("tune", dev)):
            ax = axes[di, si]; rows = [row for row in tables["GATE_CHANNELS"] if row["dataset"] == dataset and row["split"] == split]
            x = np.arange(len(rows)); eligible = [row["gate_eligible_windows_per_channel"] for row in rows]
            changed = [row["eligible_windows_with_observed_future_change_per_channel"] for row in rows]
            ax.bar(x - .17, eligible, width=.34, color="#0072B2", label="Past gate eligible")
            ax.bar(x + .17, changed, width=.34, color="#D55E00", label="Eligible AND observed future changed")
            ax.set_xticks(x, [str(index + 1) for index in x]); ax.set_title(f"{DISPLAY[dataset]} / {split.upper()} / {sum(eligible)} eligible C-window pairs")
            ax.set_xlabel("Channel index (one based in figure)"); ax.set_ylabel("Window count per channel, overlapping windows"); ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Future changes are diagnostic only: past flatness does not guarantee a flat future", fontsize=13)
    save(fig, "03_gate_eligibility_and_future_changes")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    labels = ("Raw", "Raw gated", "PCA", "PCA gated", "FIT target", "FIT target gated")
    colors = ("#666666", "#BBBBBB", "#0072B2", "#56B4E9", "#D55E00", "#E69F00")
    for di, dataset in enumerate(DATASETS):
        for bi, batch in enumerate((1, 8)):
            ax = axes[di, bi]; rows = [choose(tables["WHOLE_COST"], dataset=dataset, batch=batch, route=route) for route in ROUTES]
            values = [row["median_seconds_per_window"] * 1000 for row in rows]; ax.bar(np.arange(6), values, color=colors, alpha=.8)
            maximum = max(max(row["seconds"]) * 1000 / batch for row in rows)
            for index, row in enumerate(rows):
                samples = np.asarray(row["seconds"]) * 1000 / batch; ax.scatter(index + np.linspace(-.18, .18, 6), samples, color="black", s=12)
                ax.text(index, max(values[index], samples.max()) + maximum * .025, f"{values[index]:.2f} ms\nratio {row['ratio_of_RAW_medians']:.3f}", ha="center", va="bottom", fontsize=7.5)
            pre = choose(tables["PHYSICAL_PREFLIGHT"], dataset=dataset)
            ax.set_ylim(0, maximum * 1.34); ax.set_xticks(np.arange(6), labels, rotation=17, ha="right"); ax.set_ylabel("Whole latency per window (ms)")
            ax.set_title(f"{DISPLAY[dataset]} / B={batch} / first8 gate-eligible pairs={pre['eligible_window_channel_pairs_first8']}")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Six actual F-only routes; complete Latin6; Beijing timed prefix has ZERO gate-eligible coordinates", fontsize=12)
    save(fig, "04_actual_whole_cost")
    return paths


def text_report(cpu, cost, cost_audit, identity, tables, linked, output):
    lines = ["# 과거 상수 조건으로 출력을 고정하면 무엇이 바뀌는가", "",
        "[확인] 과거만으로 계산한 조건을 기존 예측 출력에 적용하는 대조와 별도의 실제 전체 추론 비용을 완료했다. 이 gate는 표현학습 개선이나 작은 K의 정보 보존 성공이 아니다. 같은 원 과거 입력에서 C별 조건 정보를 새 decoder에 추가하는 출력 개입이다. 성능 변화, 적용된 좌표 수, gate 자체의 비용 효과를 구분해 읽어야 한다.", "",
        "## gate의 입력과 이번 비교의 경계", "",
        "512개의 causal-filled 정규화 FP32 과거값이 현재값과 모두 정확히 같고 원 관측 과거가 두 개 이상인 C 좌표만, 예측 horizon48 전체를 현재 C값으로 바꾼다. 그 밖의 좌표는 저장된 예측의 비트를 그대로 보존한다. gate의 판단에는 실제 미래값·미래 mask·미래 flatness를 쓰지 않는다. threshold는 근사 tolerance가 아니라 exact equality이고 MIN_OBSERVED=2를 고정했다. FIT, TUNE, 개발 결과를 보고 threshold를 fitting하거나 method checkpoint를 새로 선택하지 않았다. 원 정규화 과정이 작은 물리 변화를 반올림할 수 있으므로 normalized FP32에서의 flat은 원 물리 값의 완벽한 상수성과 같다고 단정할 수 없다.", "",
        "raw 과거 입력 전체는 같아도 decoder 정보 interface는 달라졌다. 기존 K streams와 current-C anchor에 과거 C별 flag가 추가된다. 논리적으로 C개의 binary 판단 정보이지만 실제 bit-packed 전송을 측정한 것이 아니다. NumPy bool은 flag 하나당 byte로 저장되고, exact_flat의 bool 배열과 true_observed_counts의 int64 배열도 별도로 존재한다. 네트워크 전송이나 C-bit packet 비용은 측정하지 않았다. 전체 원자료 입력이 같다는 사실을 ‘decoder가 받는 side information도 같다’고 바꾸지 않는다. 이 경로는 엄격한 K-only 압축이 아니며, 그 추가 과거 정보를 원 표현학습의 개선분으로 기록하지 않는다.", "",
        "기존 FIT-target 실험에서 새로 학습한 eight native heads, E/D, Bolt와 checkpoint를 유지했다. CPU probe는 Raw와 두 표현의 Bolt/native 두 seed를 합쳐 7방법 × 두 데이터 × FIT/TUNE/노출 DEV의42쌍을 cached 출력으로 계산했다. 새로운 E/F/head forward는0이고 native는 고정 head의 출력 개입이다. gated native를 다시 학습한 최적화된 benchmark가 아니다. 별도의 GPU 비용은 Raw·gated Raw·PCA Bolt·gated PCA Bolt·FIT-target Bolt·gated FIT-target Bolt의 여섯 실제 forward다. native inference 비용을 이번 새 gate 비용에 포함하지 않는다.", "",
        "## 절대 점수와 signed 효과", "", "```text", "Dataset / Split / Method        ungated MSE       gated MSE          gated-minus-ungated"]
    for row in tables["SIGNED_EFFECT"]:
        if row["split"] != "fit" and row["mask"] == "all_original_observed" and row["metric_scale"] == "normalized":
            lines.append(f"{row['dataset']}/{row['split']}/{row['method']} {row['ungated_MSE']:.12g} {row['gated_MSE']:.12g} {row['gated_minus_ungated_MSE']:+.6e}")
    lines += ["```", "", "![Absolute matched scores](figures/01_absolute_scores.png)", "", "![Signed gate changes](figures/02_signed_gate_effect.png)", "",
        "정규화 MSE·MAE는 원 관측 target의 채널 균등 macro로 계산했고, 같은 complete-C mask 비교와 물리 단위 채널별 진단을 CSV에 따로 남겼다. full observed와 complete-C 표본 집합을 섞지 않는다. 작은 음수 delta도 산술 변화로 표시하지만 통계적 개선이라고 주장하지 않는다. 특히 Raw 출력의 매우 작은 잔차, signed zero·FP32 rounding에 따른 변화는 큰 예측 문제 해결의 증거가 아니다. equality에서는 -0과+0이 같게 판정돼도 np.where는 선택한 anchor의 비트를 보존한다. 성능과 비트 단위 parity는 서로 다른 확인 항목이다.", "",
        "## 과거 상수성과 미래 변화", "", "```text", "Dataset / Split       eligible C-window pairs    eligible with observed future change"]
    for row in tables["GATE_CASES"]:
        lines.append(f"{row['dataset']}/{row['split']} {row['eligible_window_channel_pairs']} {row['eligible_window_channel_pairs_with_observed_future_change']}")
    lines += ["```", "", "![Eligibility and observed future changes](figures/03_gate_eligibility_and_future_changes.png)", "",
        "[확인] 표의 미래 변화는 gate가 적용된 과거 조건에서 나중에 관측된 target이 currentC와 달랐는지 확인하는 진단이다. 미래 변화 여부로 gate를 고르지 않았다. 여러 채널과 겹치는 창의 C-window pair는 독립 사례 수가 아니다. 일부 미래가 결측이면 관측된 부분의 변화만 세며, 변화가 없다는 카운트를 전체 미래가 상수였다는 증명으로 해석하지 않는다. 과거가 flat해도 미래에 움직일 수 있으므로 이 대조가 범용적인 persistence 보장을 주지 않는다.", "",
        "이 gate는 C 조건에 따라 기존 D 출력의 일부 좌표를 유지하거나 anchor로 고정하는 conditional decoder다. 출력이 항상 원 고정 D의 동일 행공간에 속한다는 계약이 사라진다. 따라서 이전 static fixed-D 공간 최소값을 새 gated 예측기의 일반 하한으로 가져오지 않는다. 현재 개입을 현재-K 출력 공간의 수학적 불가능성을 해결한 것으로도 말하지 않는다. 동일한 추가 정보 처리를 Raw에도 적용한 gated Raw 대조가 있기 때문에, 압축기를 쓰는 경로의 이득만 따로 부풀리지 않는다.", "",
        "## 실제 전체 추론 비용과 정확한 gate preflight", "",
        "두 데이터 × B1/B8 × 여섯 F-only 방법의24조건을 각6회 측정해144개 시간 표본을 얻었다. six circular rotations에서 여섯 방법은 각 위치를 정확히 한 번 차지한다. warmup은각2회이며 같은 첫 B개 origins를 반복했다. 숫자144는 새 독립 데이터 표본144개가 아니다. 타이머는 실제 물리 과거 causal fill·FIT normalize·E·고정 Bolt와 전송·D·과거 flag/count 계산·출력 gate·물리 단위 CPU return과 synchronization을 포함한다. 입력·모델·행렬 로딩, 참조·warmup·저장·hash·타이머 밖 검사 및 memory snapshot은 제외했다. 공유 F와 행렬이 있는 프로세스 RSS/VRAM 관측이며 방법별 독점 memory 또는 bit-packed 통신을 측정하지 않았다.", "",
        "[확인] 타이머 전에 실제 물리 과거의 FP32 전처리 first8에서 keep_forecast·exact_flat·true_observed_counts를 다시 계산하고 accepted CPU first8과 byte-exact로 대조했다. 일반 normalized output tolerance parity로 gate의 동등성을 대신 확인하지 않았다. 한 ULP 차이가 출력 tolerance를 통과해도 exact-flat 결정을 바꿀 수 있다는 반례를 포함한다. B1/B8에 쓰는 동일 origins·전처리·원 관측 mask·flag array SHA와 paired 출력 parity가 독립 감사에서 확인됐다.", "", "```text", "Dataset       actual timed first8 gate-eligible C-window pairs"]
    for row in tables["PHYSICAL_PREFLIGHT"]: lines.append(f"{row['dataset']} {row['eligible_window_channel_pairs_first8']}")
    lines += ["```", "", "![Actual full inference cost](figures/04_actual_whole_cost.png)", "",
        "[확인] Beijing의 실제 B1/B8 timed prefix에서는 gate-eligible 좌표가0개다. 따라서 gated와 ungated timing 차이를 변경된 예측 출력에 의한 가속 효과로 돌릴 수 없다. 동일 forward·게이트 연산을 실행하는 새 시행의 짧은 시간 변동을 함께 보아야 한다. Robin에 gate 좌표가 있어도 여섯 반복만으로 gate가 인과적으로 가속했다고 입증한 것은 아니다. 이 시행 내부의 latency 기술통계를 보고하며 이전 시행의 절대 시간과 연결해 추세 개선을 주장하지 않는다.", "", "```text", "Dataset / Route / B      median ms/window    ratio of RAW medians   median paired ratio"]
    for row in tables["WHOLE_COST"]: lines.append(f"{row['dataset']}/{row['route']}/B{row['batch']} {row['median_seconds_per_window']*1000:.6f} {row['ratio_of_RAW_medians']:.6f} {row['median_paired_RAW_ratio']:.6f}")
    passes = sum(row["all_registered_conditions_pass"] for row in tables["REGISTERED_ADOPTION"])
    lines += ["```", "",
        f"사전 비교 기준은 ungated Raw의 같은 TUNE MSE·MAE 각각1.01배 이하와 B8 중앙 latency 비율0.90 이하의 동시 충족이다. gated Raw를 포함한10후보 중 동시 통과는 {passes}/10이다. ratio of medians와 paired ratio의 중앙값을 따로 보고하며 gate에는 전자를 사용했다. 이 기준 확인은 TUNE·DEV를 보고 모델을 새로 선택한 작업이 아니다. 압축 경로의 일부 오차 감소만으로 정확도·비용 목적을 전체 달성했다고 기록하지 않는다.", "",
        "## 감사 실패의 실제 원인과 보존", "",
        f"첫 wholecost 독립 감사 v1은 cardinality 문자열 교체 중 seed guard가 {identity['incorrect_v1_auditor_seed_literal']}로 오염되어 실패했다. 실제 producer의 seed는 {identity['registered_producer_seed']}였고 기록된 Latin6 schedule이 정상인 것이 identity receipt에서 확인됐다. 새 v2 감사는 seed literal 하나만 복원했다. producer의 시간·예측·허용오차는 그대로이고 비용 실험 재실행이나 모델 재학습은0회였다. 실패한 감사 source·JSON·resource receipt를 지우거나 완료 결과로 덮어쓰지 않았다. 보고서의 완료 근거는 v2 PASS이며 v1 실패를 성능 실패로 해석하지 않는다.", "",
        "## 해결한 범위와 남은 판단", "",
        "[확인] 지정한 과거 조건의 출력 개입, 원 출력의 유지·현재 anchor의 byte parity, 실제 whole inference 안에서의 조건 계산 및 비용 측정, Raw에 같은 처리를 적용하는 control을 완료했다. 이는 과거 정보로 실행 가능한 간단한 보호 규칙의 진단이다. 새로운 representation 학습·dynamic graph·contrastive alignment·ExoTimer 연결을 구현했다고 쓰지 않는다. 이 후속은 이미 노출된 DEV·oracle 결과를 본 뒤 정한 탐색적 설계이며 보호된 TEST는 읽지 않았다. 검사 통과를 독립 일반화 또는 성능 개선의 증명으로 바꾸지 않는다.", "",
        "[추정·다음 판단] 특정 과거 상태의 잘못된 잠재 변화가 C로 번지는 문제를 살피는 데는 근거가 되지만, 미래 변화가 있는 gate 사례와 미적용 사례의 오차가 함께 남는다. 조건 정보를 주면 해결되는 부분과 고정 K 표현·미래 예측에서 계속 부족한 부분을 나누어 판단해야 한다. 더 넓은 conditional decoder나 gate를 학습하는 경로는 이번 완료 범위에 포함하지 않았다. 현재 단계에서 전체 작은-K 연구, 고정 Bolt의 필요성 또는 대조학습·동적 graph의 필요를 확정하지 않는다.", "",
        "## 원수치·소스·이전 보고서", ""]
    for name, path in linked.items(): lines.append(f"- [{name}]({rel(path, output)}) — SHA256 `{sha(path)}`")
    for name, rows in tables.items(): lines.append(f"- [{name}: {len(rows)} rows](tables/{name}.csv)")
    lines += ["", "이 보고서 생성기는 JSON metadata를 읽어 표·그림을 만들며 NPZ·모델·checkpoint를 열거나 학습·F forward를 실행하지 않는다. 기존 결과·실행 소스·보고서를 덮어쓰지 않았고, 보호된 TEST·기존 작업 종료·자동 commit/push는 수행하지 않았다.", ""]
    return "\n".join(lines)


def generate(cpu_path, cpu_audit_path, cost_path, cost_audit_path, identity_path, prior_reports, output):
    cpu, audit, cost, cost_audit, identity = load_verified(cpu_path, cpu_audit_path, cost_path, cost_audit_path, identity_path)
    require(len(prior_reports) == 4 and all(Path(path).is_file() and Path(path).suffix == ".md" for path in prior_reports), "FOUR_PRIOR_MD_REPORTS_REQUIRED")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False); (output / "tables").mkdir()
    tables = collect_tables(cpu, audit, cost, cost_audit, identity)
    for name, rows in tables.items(): write_csv(output / "tables" / (name + ".csv"), rows)
    images = figures(tables, output)
    linked = {"CPU producer": cpu_path, "CPU independent audit": cpu_audit_path, "wholecost producer": cost_path,
        "wholecost independent v2 audit": cost_audit_path, "auditor seed-failure identity": identity_path,
        "matched projection report": prior_reports[0], "rank oracle v2 report": prior_reports[1],
        "D metric v2 report": prior_reports[2], "FIT target report": prior_reports[3], "report generator": Path(__file__)}
    (output / "REPORT_KO.md").write_text(text_report(cpu, cost, cost_audit, identity, tables, linked, output), encoding="utf-8")
    manifest = {"status": "COMPLETE", "source": record(Path(__file__)), "CPU_producer": record(cpu_path), "CPU_audit": record(cpu_audit_path),
        "cost": record(cost_path), "cost_audit": record(cost_audit_path), "failure_identity": record(identity_path),
        "prior_reports": [record(path) for path in prior_reports], "source_bindings": {name: record(path) for name, path in linked.items()},
        "metadata_only": True, "NPZ_or_model_opened": False, "training_or_F_forward": False, "protected_TEST_accessed": False,
        "new_native_deployment_cost_measured": False, "strict_K_only_compression": False, "decoder_side_information_changed": True,
        "bit_packed_transmission_cost_measured": False, "static_fixed_D_floor_applied_to_gate": False,
        "independent_generalization_claimed": False, "gate_caused_speed_gain_claimed": False,
        "CSV_rows": {name: len(rows) for name, rows in tables.items()}, "figures": [rel(path, output) for path in images],
        "files": [{"path": rel(path, output), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()]}
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as handle: json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("cpu", "cpu-audit", "cost", "cost-audit", "failure-identity", "main-report", "rank-report", "decoder-report", "fit-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    receipt = generate(args.cpu, args.cpu_audit, args.cost, args.cost_audit, args.failure_identity,
        (args.main_report, args.rank_report, args.decoder_report, args.fit_report), args.output)
    print(json.dumps({"status": receipt["status"], "CSV_rows": receipt["CSV_rows"]}))
