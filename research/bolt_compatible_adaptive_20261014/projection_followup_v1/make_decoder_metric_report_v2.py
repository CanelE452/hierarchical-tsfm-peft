"""Metadata-only audited decoder intervention report; no scientific arrays/models."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
DECODERS = ("ORIGINAL_PAST_LS", "ENCODER_PINV")
DEPLOY = ("BOLT", "NATIVE_s92601", "NATIVE_s92602")
ROUTES = ("RAW_BOLT", "PCA_PAST_LS_BOLT", "FORECA_PAST_LS_BOLT", "PCA_ENCODER_PINV_BOLT", "FORECA_ENCODER_PINV_BOLT")
DISPLAY = {"robin": "Robin", "beijing_aotizhongxin": "Beijing Aotizhongxin"}
LABELS = ("Raw Bolt", "PCA / past LS", "ForeCA / past LS", "PCA / pinv(E)", "ForeCA / pinv(E)")
COLORS = ("#666666", "#0072B2", "#D55E00", "#56B4E9", "#E69F00")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, label):
    if not condition:
        raise ValueError(label)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def bind(reference, path, label):
    actual = Path(path).resolve()
    require(actual.suffix.lower() == ".json", "REPORT_READS_JSON_METADATA_ONLY:" + label)
    require(resolve(reference["path"]) == actual and reference["sha256"] == sha(actual), "EXACT_SHA_BINDING:" + label)


def load_verified(decoder_path, decoder_audit_path, cost_path, cost_audit_path):
    decoder, audit, cost, cost_audit = (read(path) for path in (decoder_path, decoder_audit_path, cost_path, cost_audit_path))
    require(decoder.get("status") == cost.get("status") == "COMPLETE", "BOTH_COMPLETE_REQUIRED")
    require(audit.get("status") == cost_audit.get("status") == "PASS", "BOTH_INDEPENDENT_PASS_REQUIRED")
    bind(audit["diagnostic"], decoder_path, "decoder audit")
    bind(cost_audit["cost"], cost_path, "whole cost audit")
    bind(cost["decoder_diagnostic"], decoder_path, "cost.decoder_diagnostic")
    bind(cost["decoder_audit"], decoder_audit_path, "cost.decoder_audit")
    for key in ("producer", "projection_audit", "span_diagnostic", "span_audit"):
        bind(audit[key], resolve(decoder[key]["path"]), "decoder." + key)
        require(audit[key] == decoder[key], "SAME_DECODER_AUDIT_CHAIN:" + key)
    for key in ("producer", "projection_audit"):
        require(cost[key] == decoder[key], "SAME_FIXED_PROJECTION_CHAIN:" + key)
        bind(cost[key], resolve(decoder[key]["path"]), "cost." + key)
        require(cost_audit[key] == cost[key], "SAME_COST_AUDIT_CHAIN:" + key)
        bind(cost_audit[key], resolve(cost[key]["path"]), "cost_audit." + key)
    for key in ("decoder_diagnostic", "decoder_audit"):
        require(cost_audit[key] == cost[key], "SAME_DECODER_COST_AUDIT_CHAIN:" + key)
    require(all(audit["totals"].get(key) == value for key, value in {
        "base_cells_checked": 12, "decoder_cells_checked": 24,
        "cached_forecast_arrays_checked": 96, "original_predictions_byte_exact_checked": 48}.items()), "FULL_DECODER_AUDIT_REQUIRED")
    require(all(cost_audit["totals"].get(key) == value for key, value in {
        "cost_cells_checked": 20, "recorded_timer_samples_checked": 120}.items()), "FULL_COST_AUDIT_REQUIRED")
    require(decoder.get("completed_cells") == 12 and not decoder.get("failed_paths"), "TWELVE_DECODER_CELLS_REQUIRED")
    require(decoder.get("new_training") is False and decoder.get("new_encoder_F_or_native_head_forward") is False
            and decoder.get("future_labels_used_to_fit_alternative_D") is False
            and decoder.get("method_checkpoint_branch_selection_used") is False, "CACHED_FIXED_MODEL_LABEL_FREE_D_PROBE_REQUIRED")
    require(cost.get("new_training_updates") == cost.get("projection_fits") == cost.get("alternative_decoder_refits") == 0
            and cost.get("method_checkpoint_branch_selection_used") is False, "COST_HAS_NO_TRAINING_OR_SELECTION")
    require(cost.get("F_before") == cost.get("F_after") and cost.get("expected_cost_cells") == 20
            and cost.get("recorded_timer_samples") == 120 and tuple(cost["routes"]) == ROUTES, "ACTUAL_FIXED_F_TWENTY_COST_CELLS")
    for data in (decoder, audit, cost, cost_audit):
        require(data.get("protected_TEST_accessed") is False, "NO_PROTECTED_TEST_REQUIRED")
    require(decoder.get("original_selected_forecasts_and_locks_unchanged") is True
            and cost.get("original_selected_forecasts_and_locks_unchanged") is True, "ORIGINAL_RESULTS_PRESERVED")
    return decoder, audit, cost, cost_audit


def metric_fields(scores):
    normal, physical = scores["normalized"], scores["physical_per_channel_diagnostic"]
    return {"normalized_MSE": normal["mse"], "normalized_MAE": normal["mae"],
            "physical_macro_MSE_diagnostic": physical["mse"], "physical_macro_MAE_diagnostic": physical["mae"],
            "observed_counts": normal["observed_counts"], "active_channels": normal["active_channels"],
            "normalized_per_channel_MSE": normal["per_channel_mse"], "normalized_per_channel_MAE": normal["per_channel_mae"],
            "physical_per_channel_MSE": physical["per_channel_mse"], "physical_per_channel_MAE": physical["per_channel_mae"]}


def collect_tables(decoder, cost):
    tables = {key: [] for key in ("DEPLOYMENT_AND_ORACLE_SCORES", "FIXED_D_DIAGNOSTIC", "GEOMETRY",
                                 "PAST_RECONSTRUCTION", "SOURCE_LINEAGE", "WHOLE_COST", "WHOLE_COST_SAMPLES", "REGISTERED_ADOPTION")}
    for dataset, (channels, k, development) in DATASETS.items():
        dataset_row = decoder["datasets"][dataset]
        for projection in ("PCA", "FORECA"):
            condition = dataset_row["projections"][projection]
            for split in ("fit", "tune", development):
                cell = condition["splits"][split]
                common = {"dataset": dataset, "projection": projection, "split": split, "C": channels, "K": k,
                    "complete_target_positions": cell["complete_target_positions"], "total_target_positions": cell["total_target_positions"]}
                for mask, score_key in (("all_original_observed", "RAW_scores_all_observed"), ("all_C_complete", "RAW_scores_complete_mask")):
                    tables["DEPLOYMENT_AND_ORACLE_SCORES"].append({**common, "decoder": "NA", "predictor": "RAW_BOLT",
                        "mask": mask, "role": "saved_baseline_repeated_per_projection", **metric_fields(cell[score_key])})
                tables["SOURCE_LINEAGE"].append({**common, "source_input": cell["source_input"], "source_state": condition["source_state"],
                    "cached_BOLT": cell["cached_future_sources"]["BOLT"], "decoder_probe_artifact": cell["artifact"],
                    "alternative_construction": cell["alternative_decoder_construction"], "new_E_F_head_forward": False,
                    "D_fitted_to_future": False})
                for variant in DECODERS:
                    value = cell["decoders"][variant]
                    for predictor in DEPLOY:
                        forecast = value["forecasts"][predictor]
                        for mask, score_key in (("all_original_observed", "scores_all_observed"), ("all_C_complete", "scores_complete_mask")):
                            tables["DEPLOYMENT_AND_ORACLE_SCORES"].append({**common, "decoder": variant, "predictor": predictor,
                                "mask": mask, "role": "fixed_cached_forecast_D_intervention", **metric_fields(forecast[score_key])})
                    tables["DEPLOYMENT_AND_ORACLE_SCORES"].append({**common, "decoder": variant, "predictor": "E_Y",
                        "mask": "all_C_complete", "role": "actual_future_diagnostic_only", **metric_fields(value["forecasts"]["E_Y"]["scores_complete_mask"])})
                    diagnostic = value["fixed_D_real_arithmetic_diagnostic_same_complete_mask"]
                    tables["FIXED_D_DIAGNOSTIC"].append({**common, "decoder": variant,
                        "mask": "all_C_complete", "role": "real_arithmetic_fixed_D_span_only", **metric_fields(diagnostic["scores"]),
                        "normalized_MSE_is_minimum": True, "MAE_is_minimum": False, "physical_MSE_is_minimum": False,
                        "Raw_is_constrained_to_this_D": False, "independent_linear_checks": diagnostic["independent_linear_checks"]})
                    tables["GEOMETRY"].append({**common, "decoder": variant, **value["geometry"],
                        "native_heads_reoptimized_for_new_D": False, "causal_obliqueness_only_effect": False})
                    tables["PAST_RECONSTRUCTION"].append({**common, "decoder": variant,
                        "mask": "past_original_observed", **metric_fields(value["past_reconstruction_observed_only"]),
                        "bias_rule": value["past_reconstruction_bias_rule"], "future_prediction_guaranteed": False})
        rows = cost["cost"][dataset]["rows"]
        require(len(rows) == 10 and {(row["route"], row["batch"]) for row in rows} == {(route, batch) for route in ROUTES for batch in (1, 8)}, "EXACT_TEN_COST_CELLS:" + dataset)
        for row in rows:
            raw = next(value for value in rows if value["route"] == "RAW_BOLT" and value["batch"] == row["batch"])
            tables["WHOLE_COST"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": row["repeats"],
                "median_whole_seconds": row["median_seconds"], "median_seconds_per_window": row["median_seconds_per_window"],
                "ratio_of_RAW_medians": row["median_seconds"] / raw["median_seconds"],
                "median_of_paired_RAW_ratios": row["paired_RAW_median_ratio"], "seconds": row["seconds"],
                "paired_RAW_seconds": row["paired_RAW_seconds"], "positions": row["positions"], "origins": row["origins"],
                "warmup_each_route": row["warmup_each_route"], "shared_memory_scope": row["memory_scope"],
                "peak_RSS_bytes_shared_process": max(sample["resident_rss_bytes"] for sample in row["memory"]),
                "peak_reserved_VRAM_bytes_shared_process": max(sample["peak_reserved_vram_bytes"] for sample in row["memory"]),
                "whole_includes": row["whole_includes"], "whole_excludes": row["whole_excludes"],
                "stage_decomposition": False, "native_inference_cost_measured": False,
                "selected_metadata": row["selected_metadata"], "timed_outputs": row["timed_outputs"], "reference": row["reference"]})
            for index, sample in enumerate(row["samples"]):
                tables["WHOLE_COST_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"],
                    "repeat": sample["repeat"], "position": sample["position"], "whole_seconds": sample["seconds"],
                    "RAW_same_repeat_seconds": row["paired_RAW_seconds"][index], "paired_RAW_ratio": row["paired_RAW_ratios"][index],
                    "perf_start_ns": sample["perf_start_ns"], "perf_end_ns": sample["perf_end_ns"],
                    "all_four_parity_checks_pass": all(value["pass"] for value in sample["parity"].values()),
                    "parity": sample["parity"], "shared_process_memory": sample["memory"], "sample_artifact": sample["sample_artifact"]})
        raw_tune = dataset_row["projections"]["PCA"]["splits"]["tune"]["RAW_scores_all_observed"]["normalized"]
        for route in ROUTES[1:]:
            projection = "FORECA" if route.startswith("FORECA") else "PCA"
            variant = "ENCODER_PINV" if "ENCODER_PINV" in route else "ORIGINAL_PAST_LS"
            scores = dataset_row["projections"][projection]["splits"]["tune"]["decoders"][variant]["forecasts"]["BOLT"]["scores_all_observed"]["normalized"]
            row = next(row for row in tables["WHOLE_COST"] if row["dataset"] == dataset and row["route"] == route and row["batch"] == 8)
            mse_ratio, mae_ratio = scores["mse"] / raw_tune["mse"], scores["mae"] / raw_tune["mae"]
            tables["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "registered_B8_ratio_of_medians": row["ratio_of_RAW_medians"],
                "TUNE_MSE_ratio_to_RAW": mse_ratio, "TUNE_MAE_ratio_to_RAW": mae_ratio,
                "latency_le_0_90": row["ratio_of_RAW_medians"] <= .90, "MSE_le_1_01": mse_ratio <= 1.01, "MAE_le_1_01": mae_ratio <= 1.01,
                "all_registered_conditions_pass": row["ratio_of_RAW_medians"] <= .90 and mse_ratio <= 1.01 and mae_ratio <= 1.01,
                "actual_model_or_method_selection_used": False})
    require(len(tables["WHOLE_COST_SAMPLES"]) == 120, "ONE_HUNDRED_TWENTY_ACTUAL_TIMER_SAMPLES")
    return tables


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def rel(path, output):
    return Path(os.path.relpath(Path(path).resolve(), Path(output).resolve())).as_posix()


def score_row(tables, dataset, projection, split, decoder, predictor="BOLT", mask="all_original_observed"):
    return next(row for row in tables["DEPLOYMENT_AND_ORACLE_SCORES"] if all(row[key] == value for key, value in
        {"dataset": dataset, "projection": projection, "split": split, "decoder": decoder, "predictor": predictor, "mask": mask}.items()))


def figure_outputs(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    folder = Path(output) / "figures"
    folder.mkdir(exist_ok=False)
    files = []

    def save(fig, stem):
        for suffix in ("png", "svg"):
            path = folder / (stem + "." + suffix)
            fig.savefig(path, dpi=170 if suffix == "png" else None, bbox_inches="tight")
            files.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8), layout="constrained")
    for di, (dataset, (_, k, development)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("tune", development)):
            ax = axes[di, si]
            raw = score_row(tables, dataset, "PCA", split, "NA", "RAW_BOLT")
            candidates = [raw] + [score_row(tables, dataset, projection, split, decoder) for projection, decoder in
                (("PCA", "ORIGINAL_PAST_LS"), ("FORECA", "ORIGINAL_PAST_LS"), ("PCA", "ENCODER_PINV"), ("FORECA", "ENCODER_PINV"))]
            values = [row["normalized_MSE"] for row in candidates]
            bars = ax.bar(np.arange(5), values, color=COLORS, width=.7)
            ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=9)
            ax.set_ylim(0, max(values) * 1.23)
            ax.set_xticks(np.arange(5), LABELS, rotation=17, ha="right")
            ax.set_title(f"{DISPLAY[dataset]} / {split.upper()} / K={k}")
            ax.set_ylabel("Normalized macro MSE (lower is better)")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("D-only intervention: identical cached E/F; all original observed targets", fontsize=14)
    save(fig, "01_deployed_bolt_scores")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.5), layout="constrained")
    diagnostic_labels = ("Raw Bolt", "Past LS: actual Bolt", "pinv(E): actual Bolt", "Past LS: fixed-D MSE minimum", "pinv(E): fixed-D MSE minimum", "Past LS: E(Y) diagnostic", "pinv(E): E(Y) diagnostic")
    for di, (dataset, (_, _, development)) in enumerate(DATASETS.items()):
        for pi, projection in enumerate(("PCA", "FORECA")):
            ax = axes[di, pi]
            raw = score_row(tables, dataset, projection, development, "NA", "RAW_BOLT", "all_C_complete")
            actual = [score_row(tables, dataset, projection, development, decoder, mask="all_C_complete")["normalized_MSE"] for decoder in DECODERS]
            floors = [next(row["normalized_MSE"] for row in tables["FIXED_D_DIAGNOSTIC"] if row["dataset"] == dataset and row["projection"] == projection and row["split"] == development and row["decoder"] == decoder) for decoder in DECODERS]
            oracle = [score_row(tables, dataset, projection, development, decoder, "E_Y", "all_C_complete")["normalized_MSE"] for decoder in DECODERS]
            values = [raw["normalized_MSE"], *actual, *floors, *oracle]
            ax.barh(np.arange(7), values, color=("#666666", "#0072B2", "#56B4E9", "#CC79A7", "#882255", "#D55E00", "#E69F00"), height=.65)
            for index, value in enumerate(values):
                ax.text(value + max(values) * .018, index, f"{value:.4f}", va="center", fontsize=9)
            ax.set_xlim(0, max(values) * 1.22)
            ax.set_yticks(np.arange(7), diagnostic_labels, fontsize=8.5)
            ax.invert_yaxis()
            ax.set_title(f"{DISPLAY[dataset]} / {projection} / {development.upper()}\ncomplete C-target positions: {raw['complete_target_positions']}/{raw['total_target_positions']}")
            ax.set_xlabel("Normalized macro MSE on the SAME complete-C mask")
            ax.grid(axis="x", alpha=.2)
    fig.suptitle("Future-informed diagnostics are separate from deployed forecasts; no MAE/physical minima claim", fontsize=12)
    save(fig, "02_complete_mask_diagnostics")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8), layout="constrained")
    offsets = np.linspace(-.20, .20, 6)
    for di, dataset in enumerate(DATASETS):
        for bi, batch in enumerate((1, 8)):
            ax = axes[di, bi]
            rows = [next(row for row in tables["WHOLE_COST"] if row["dataset"] == dataset and row["batch"] == batch and row["route"] == route) for route in ROUTES]
            values = [row["median_seconds_per_window"] * 1000 for row in rows]
            bars = ax.bar(np.arange(5), values, color=COLORS, width=.7, alpha=.8)
            for index, row in enumerate(rows):
                ax.scatter(index + offsets, np.asarray(row["seconds"]) * 1000 / batch, color="black", s=14, alpha=.75, zorder=3)
            maximum = max(max(row["seconds"]) * 1000 / batch for row in rows)
            padding = maximum * .025
            for index, (value, row) in enumerate(zip(values, rows)):
                top = max(value, max(row["seconds"]) * 1000 / batch)
                ax.text(index, top + padding, f"{value:.2f} ms\nratio {row['ratio_of_RAW_medians']:.3f}",
                        ha="center", va="bottom", fontsize=8)
            ax.set_ylim(0, maximum * 1.32)
            ax.set_xticks(np.arange(5), LABELS, rotation=17, ha="right")
            ax.set_ylabel("Whole latency per window (ms)")
            ax.set_title(f"{DISPLAY[dataset]} / B={batch} / 6 actual samples per route")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("NEW trial only: causal fill -> FIT normalize -> E -> frozen Bolt -> D -> physical CPU return", fontsize=12)
    save(fig, "03_whole_deployment_cost")
    return files


def make_text(tables, decoder, audit, cost, cost_audit, paths, output, main_report, oracle_report):
    lines = ["# 고정 인코더·예측기에 디코더만 바꾼 실제 진단과 전체 비용", "",
        "[확인] `D = pinv(E_FP32)`로 디코더만 바꾸는 단순한 수정은 이번 두 데이터의 작은 K 예측을 해결하지 못했다. PCA는 원 디코더와 점수가 거의 비슷하고, Robin ForeCA는 TUNE에서 좋아졌어도 DEV에서는 악화됐다. 모든 새 비용 조건을 실제로 측정했으며, B8의 속도 이득이 정확도 보존을 충족시키지는 못했다. 이 결론은 지정한 행렬 교체와 노출된 개발 구간에 관한 것이다. 전체 표현학습 방법, 작은 K, 또는 고정 Bolt의 가능성을 부정하는 결론으로 확대할 수 없다.", "",
        "## 먼저 답해야 할 질문", "",
        "디코더가 잠재 오차를 C 공간에 전달하는 방식만 고치면 되는지 시험했다. 기존 E와 원래 선택된 Bolt·native head의 잠재 미래·마지막 과거 표현을 그대로 두고, 관측된 FIT 과거에 맞춘 affine least-squares 디코더 `ORIGINAL_PAST_LS`를 배포 E의 FP32 값을 FP64로 해석한 의사역행렬 `ENCODER_PINV`로 교체했다. 의사역행렬 cutoff는 고정 `1e-15`이며 seed, TUNE, DEV의 미래값으로 새 D를 선택하거나 맞추지 않았다. 원 D 재현 48건을 독립적으로 byte-exact 확인했다. cached probe의 96회 CPU 복원은 새 E/F/head forward나 모델 학습 96회가 아니다.", "",
        "[확인] 이 개입은 D의 출력 공간·크기·오차 metric을 함께 바꾼다. 비직교성만을 고립한 인과실험이라고 말할 수 없다. native head는 원 D에 맞춘 actual-Y 손실로 이미 학습된 head를 고정했으므로, 새 D를 위해 최적화된 native 모델과 Bolt의 경쟁 결과가 아니다. 별도 비용 실행은 저장한 잠재 미래를 그대로 돌려쓰는 proxy 대신 실제 `E → frozen Bolt → D` forward를 수행했다. 비용표의 다섯 방법에는 native가 없다.", "",
        "## 비교 계약과 실제 실행 범위", "",
        "Robin은 C=17/K=4, Beijing Aotizhongxin은 C=11/K=2를 유지한다. L=512/H=48, 기존 FIT 256창·TUNE 64창, Robin VAL 30창 및 이미 노출된 Beijing EVAL 146창의 같은 origins·FIT 통계·원 관측 mask를 사용했다. 결측 과거는 창 안에서 과거 방향으로 채우고 초기 결측에는 FIT 평균을 쓴다. 배포 입력에 실제 미래값을 넣지 않았다. protected TEST에 접근하지 않았고, 이 개발 결과는 독립 holdout 일반화 검증이 아니다. 이번 행렬 개입을 본 뒤 K, 손실, checkpoint, 분기나 채택 기준을 바꾸지 않았다.", "",
        "예측은 NumPy FP32에서 `raw_last + (latent_future - encoded_history_last) @ D`이다. current-C anchor가 있으므로 엄격한 K-only 통신 압축이 아니다. decoder bias는 과거 재구성에서만 쓰이며 innovation에서 제외된다. 새 D는 encoder center를 과거 재구성 bias로 사용하고 별도로 fit하지 않았다. 주지표는 원 관측 target에 대한 채널 균등 정규화 macro MSE이며 MAE도 함께 보고한다. 단위가 다른 채널의 물리 단위 오차 평균은 보조 진단이고, 이 평균을 일률적인 물리적 효용 점수로 읽으면 안 된다.", "",
        f"독립 검산을 통과한 범위는 base cells {audit['totals']['base_cells_checked']}개, D cells {audit['totals']['decoder_cells_checked']}개, cached CPU forecast {audit['totals']['cached_forecast_arrays_checked']}건, 새 전체 배포 비용 20조건과 시간 표본 120개다. 새 encoder/backbone/head 학습은 0회다. CSV 행 수는 mask·metric·재사용 baseline을 나눈 보고 단위이며 독립 실험 또는 독립 표본 수가 아니다.", "",
        "## 배포 점수: 원 관측 target 모두 사용", "",
        "다음은 Raw와 Bolt 연결의 정규화 MSE/MAE 원수치다. full observed mask에서 채널별 표본 수는 다를 수 있다. matched complete-C 진단값과 혼합하지 않는다.", "", "```text", "Dataset / Split                 Raw MSE   Raw MAE   PCA old/new MSE      ForeCA old/new MSE"]
    for dataset, (_, _, development) in DATASETS.items():
        for split in ("tune", development):
            raw = score_row(tables, dataset, "PCA", split, "NA", "RAW_BOLT")
            scores = [score_row(tables, dataset, p, split, d)["normalized_MSE"] for p, d in (("PCA", DECODERS[0]), ("PCA", DECODERS[1]), ("FORECA", DECODERS[0]), ("FORECA", DECODERS[1]))]
            lines.append(f"{dataset}/{split:<5}  {raw['normalized_MSE']:.9f} {raw['normalized_MAE']:.9f} {scores[0]:.9f}/{scores[1]:.9f} {scores[2]:.9f}/{scores[3]:.9f}")
    lines += ["```", "", "![Actual deployed Bolt scores](figures/01_deployed_bolt_scores.png)", "",
        "Robin ForeCA의 TUNE Bolt MSE는 0.525601060에서 0.484641465로 줄고 MAE도 0.521664313에서 0.375734728로 줄었다. 그러나 DEV MSE는 0.822981362에서 1.210370368로 늘었다. 이 불일치를 TUNE 개선만으로 감추거나 새 성공 gate를 만들지 않는다. PCA는 같은 공간의 가까운 디코더 교체라 작은 수치 차이가 있지만 Robin·Beijing 모두 Raw 정확도에 도달하지 않는다. Beijing ForeCA는 DEV MSE가 0.900557095에서 0.903214674로 변해 역시 해결되지 않았다.", "",
        "native 두 minibatch-order seed도 동일한 잠재 궤적을 새 D로 복원했다. 예를 들어 Robin ForeCA DEV MSE는 원 D에서 0.679014623/0.713484658, pinv(E)에서 1.241763213/1.227917594다. 이 수치는 원 D로 학습한 head에 새 D를 끼워 넣은 결과이며, 새 D로 재학습하면 같은 결과라는 증거는 아니다. 과거 reconstruction의 변화도 별도 CSV에 보존했지만 reconstruction 개선을 미래 예측 개선으로 바꾸어 해석하지 않는다. 이전 POINTWISE/MAIN 경로는 이미 actual-Y 목적을 사용했으므로 전체 실패를 ‘재구성만 학습했기 때문’이라는 하나의 설명으로 정리할 수 없다.", "",
        "## 실제 미래를 사용하는 진단은 배포 점수와 분리", "",
        "E(Y)는 알려진 실제 미래를 고정 E로 인코딩해서 D로 복원하는 진단이다. complete-C target 위치에서만 주요 oracle 비교로 사용한다. 일부 미래 채널이 관측되지 않은 위치에서 E에 넣기 위해 0으로 대체한 성분은 알려진 미래 정보가 아니므로 full observed E(Y) 점수를 주요 비교·그림·CSV에서 제외했다. 배포 모델의 선택이나 forward에 E(Y)를 사용하지 않았다. E(Y)의 오차는 임의의 잠재 미래를 허용한 수학적 최솟값이라고 할 수 없다.", "",
        "고정 D 진단은 같은 complete-C 위치에서 실제 `Y - currentC`를 D의 행공간으로 최소제곱 투영한다. 배포 D32를 실수 FP64 연산으로 해석한 해당 공간의 **정규화 MSE 최솟값**이다. 그 해에서 나온 MAE·물리 단위 MSE는 해당 목적의 최솟값이 아니다. FP32 rounding이나 비선형 decoder, 다른 출력 공간, 무제약 Raw 예측기까지 포함하는 일반 하한도 아니다. 특히 Raw에 고정 D 하한 부등식을 적용하지 않는다.", "", "```text", "Dataset / Projection / DEV     Raw complete MSE   old-D minimum   pinv(E)-D minimum"]
    for dataset, (_, _, development) in DATASETS.items():
        for projection in ("PCA", "FORECA"):
            raw = score_row(tables, dataset, projection, development, "NA", "RAW_BOLT", "all_C_complete")
            floors = [next(row["normalized_MSE"] for row in tables["FIXED_D_DIAGNOSTIC"] if row["dataset"] == dataset and row["projection"] == projection and row["split"] == development and row["decoder"] == variant) for variant in DECODERS]
            lines.append(f"{dataset}/{projection:<6}/{development:<4} {raw['normalized_MSE']:.9f}     {floors[0]:.9f}     {floors[1]:.9f}")
    lines += ["```", "", "![Matched complete-C diagnostic scores](figures/02_complete_mask_diagnostics.png)", "",
        "[확인] Robin ForeCA의 새 D 고정공간 최솟값 자체가 DEV에서 1.179326868로, 같은 mask의 Raw 0.340401887보다 높다. pinv(E)의 E(Y)가 그 최솟값에 가깝다고 해서 예측이 좋아진 것은 아니다. 원 D의 complete-C 최소값 0.461340297와 비교하면 출력 공간 교체가 유리하지 않았다. Beijing complete-C Raw는 0.598972295이며 full-mask Raw 0.589575414와 다르다. 이 두 수치를 같은 비교에 넣지 않는다. 별도 미래-informed rank-K 진단에서 더 좋은 출력 공간이 알려진 미래에 존재했으므로, 특정 D의 실패를 K 차원의 원천적 불가능성으로 바꿀 수도 없다. 과거만으로 그런 공간과 잠재 미래를 안정적으로 알아낼 수 있는지는 여전히 미확정이다.", "",
        "## 새 실제 전체 배포 비용", "",
        "원 D가 나쁘거나 새 D에서 정확도가 개선되지 않았어도 모든 비용 조건을 실행했다. 두 데이터 × B1/B8 × 다섯 Bolt 경로 = 20조건이며, 각 방법 warmup 2회 후 실제 시간 6개를 기록했다. 첫 다섯 회는 회전 순서로 다섯 위치를 균등 배치했고 여섯 번째는 seeded 추가 순서다. 여섯 회 전체를 완전한 balanced 설계라고 말하지 않는다. 각 조건은 같은 첫 B개 개발 origins를 반복하므로 120개의 새 독립 데이터 표본도 아니다.", "",
        "타이머에는 실제 causal fill·FIT normalize·E·고정 Bolt GPU forward와 전송·D·물리 단위 CPU 출력 및 마지막 synchronization을 포함한다. 디스크·모델·행렬 로딩, D 구성, 참조 출력, warmup, 저장·hash, 타이머 밖 guard와 memory snapshot은 제외한다. stage별 비용을 분해하지 않았으므로 특정 단계가 병목이라고 단정하지 않는다. 같은 새 실행 안에서만 Raw와 비교하고 이전 시행의 절대 latency에 연결해 효과를 주장하지 않는다. memory는 F와 행렬들이 공유된 프로세스의 RSS·VRAM 관측값이다. 각 방법의 독점 메모리, isolated peak 또는 K 압축에 따른 VRAM 절감량은 아니다.", "", "```text", "Dataset / B / Route                          median ms/window    ratio of medians    median paired ratio"]
    for row in tables["WHOLE_COST"]:
        lines.append(f"{row['dataset']}/B{row['batch']}/{row['route']:<25} {row['median_seconds_per_window']*1000:9.4f}        {row['ratio_of_RAW_medians']:.6f}            {row['median_of_paired_RAW_ratios']:.6f}")
    adoption_pass = sum(row["all_registered_conditions_pass"] for row in tables["REGISTERED_ADOPTION"])
    speed_pass = sum(row["latency_le_0_90"] for row in tables["REGISTERED_ADOPTION"])
    lines += ["```", "", "![Actual whole deployment cost samples](figures/03_whole_deployment_cost.png)", "",
        f"사전 기준은 B8 **중앙 latency의 비율** ≤0.90 및 같은 TUNE MSE·MAE 각각 Raw의 1.01배 이하를 함께 요구한다. paired 비율의 중앙값은 별도 기술통계이고 채택 gate에 대신 쓰지 않는다. {len(tables['REGISTERED_ADOPTION'])}개 후보 중 속도 조건 통과는 {speed_pass}개, 세 조건 동시 통과는 {adoption_pass}개다. 빠른 경로가 있어도 정확도 보존에는 실패했다. 이 gate 재검산은 모델 선택을 새로 한 것이 아니며, 이후 결과를 보고 임계값을 조정하지 않았다. 여섯 반복과 짧은 실행만으로 deployment 환경 전체의 속도 일반화나 유의성 검증을 주장하지 않는다.", "",
        "## 해결된 부분과 다음에 남은 판단", "",
        "[확인] 원 디코더 replay가 실제 원 예측과 같다는 점, label-free pinv(E) 개입을 실제 inference에 연결할 수 있다는 점, 새 전체 배포 forward와 물리 단위 출력의 parity, 같은 mask의 숫자를 구분한 보고를 해결했다. 따라서 구현이 전혀 되지 않아 방법 가능성을 확인하지 못한 상태는 지났다. 이번 단순 D 교체는 Raw 수준 정확도를 보존하면서 비용을 낮추는 해결책으로 채택하지 않는다.", "",
        "[추정·다음 판단] E가 보존하는 정보와 D의 출력 방향, 그 방향의 미래 변화가 과거로부터 예측 가능한지를 함께 검증하는 비교가 더 직접적이다. 단순 pinv 교체만으로 충분한지에 대한 답은 부정적이지만, 고정된 임의 출력 공간에서 실패했다고 해서 예측 목적 공간 학습, D와 latent predictor의 공동 적합, 과거 조건부 동적 공간까지 실패했다고 볼 근거는 없다. 재학습 비교가 필요하다면 같은 K·raw anchor·FIT/TUNE 분리·actual-Y 정규화 MSE·고정 Bolt·비용 계약을 유지하고 독립 데이터 노출을 구분해야 한다. 이것들은 이번에 완료한 실험으로 쓰지 않는다. 대조학습·학습 그래프·ExoTimer 방식도 이번 D-only 개입으로 실행하거나 입증하지 않았다.", "",
        "## 원수치·코드·검산 연결", "",
        f"- [이전 matched projection 보고서]({rel(main_report, output)})", f"- [미래-informed rank-K 별도 진단]({rel(oracle_report, output)})"]
    for name, path in paths.items():
        lines.append(f"- [{name}]({rel(path, output)}) — SHA256 `{sha(path)}`")
    for name, rows in tables.items():
        lines.append(f"- [{name}: {len(rows)} rows](tables/{name}.csv)")
    lines += ["", "원 JSON·NPZ·모델·과거 보고서는 덮어쓰지 않았다. 이 보고서 생성기는 JSON metadata를 읽어 표·그림을 만들 뿐 학습·F forward·NPZ 또는 checkpoint 로딩을 수행하지 않는다. 완료는 위에서 지정한 실제 개입·비용·검산 범위이고, 전체 연구 목표 달성 또는 독립 일반화 검증의 완료를 의미하지 않는다.", ""]
    return "\n".join(lines)


def generate(decoder_path, decoder_audit_path, cost_path, cost_audit_path, main_report, oracle_report, output):
    decoder, audit, cost, cost_audit = load_verified(decoder_path, decoder_audit_path, cost_path, cost_audit_path)
    for path in (main_report, oracle_report):
        require(Path(path).is_file() and Path(path).suffix.lower() == ".md", "EXISTING_PRIOR_MD_REPORT_REQUIRED")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tables = collect_tables(decoder, cost)
    table_folder = output / "tables"
    table_folder.mkdir()
    for name, rows in tables.items():
        write_csv(table_folder / (name + ".csv"), rows)
    figures = figure_outputs(tables, output)
    paths = {"decoder producer": Path(decoder_path), "decoder independent audit": Path(decoder_audit_path),
             "whole cost producer": Path(cost_path), "whole cost independent audit": Path(cost_audit_path),
             "report generator": Path(__file__), "whole cost runner": HERE / "run_decoder_metric_cost_v1.py",
             "whole timing helper": HERE / "whole_decoder_metric_cost_v1.py"}
    (output / "REPORT_KO.md").write_text(make_text(tables, decoder, audit, cost, cost_audit, paths, output, main_report, oracle_report), encoding="utf-8")
    files = [path for path in output.rglob("*") if path.is_file()]
    provenance = {"status": "COMPLETE", "source": record(Path(__file__)),
        "cosmetic_supersedes": {"generator": str(HERE / "make_decoder_metric_report_v1.py"),
            "generator_sha256": "72a0cec8edac944513a0ae04bcec513da3d8136ae02cfaa516a0c0631d52fa72",
            "change": "Whole-cost figure labels placed above each route's highest sample; extra headroom",
            "scientific_or_numeric_change": False},
        "decoder": record(decoder_path), "decoder_audit": record(decoder_audit_path),
        "cost": record(cost_path), "cost_audit": record(cost_audit_path),
        "prior_reports": [record(main_report), record(oracle_report)],
        "source_bindings": {key: record(path) for key, path in paths.items()},
        "metadata_only": True, "scientific_NPZ_or_model_opened": False, "training_or_F_forward": False,
        "protected_TEST_accessed": False, "independent_generalization_claimed": False,
        "E_Y_full_observed_zero_filled_future_included_in_major_comparison": False,
        "native_head_reoptimized_for_new_D": False, "obliqueness_only_causal_effect_claimed": False,
        "cost_native_inference_measured": False, "counts": {name: len(rows) for name, rows in tables.items()},
        "figures": [rel(path, output) for path in figures],
        "files": [{"path": rel(path, output), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(files)]}
    write_json(output / "MANIFEST.json", provenance)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("decoder", "decoder-audit", "cost", "cost-audit", "main-report", "oracle-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = generate(args.decoder, args.decoder_audit, args.cost, args.cost_audit, args.main_report, args.oracle_report, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output), "counts": result["counts"]}, ensure_ascii=False))
