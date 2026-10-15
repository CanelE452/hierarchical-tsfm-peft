"""JSON-only report of past-fitted local D, conditional-space diagnostics and costs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = {"robin": (17, 4, "val"), "beijing_aotizhongxin": (11, 2, "eval")}
PROJECTIONS = ("PCA_PINV", "FIT_TARGET_PINV")
BACKENDS = {"BOLT_NUMPY": ("BOLT",), "NATIVE_TORCH32": ("NATIVE_s92601", "NATIVE_s92602")}
ROUTES = ("RAW_BOLT", "PCA_PINV_BOLT", "FIT_TARGET_PINV_BOLT", "PCA_LOCAL_BOLT", "FIT_TARGET_LOCAL_BOLT")
SCALES = ("normalized", "physical_per_channel_diagnostic")
DISPLAY = {"robin": "Robin", "beijing_aotizhongxin": "Beijing Aotizhongxin"}


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


def load_verified(cpu_path, cpu_audit_path, cost_path, cost_audit_path):
    cpu, audit, cost, cost_audit = (read(path) for path in (cpu_path, cpu_audit_path, cost_path, cost_audit_path))
    require(cpu.get("status") == cost.get("status") == "COMPLETE" and audit.get("status") == cost_audit.get("status") == "PASS", "BOTH_COMPLETE_INDEPENDENT_PASS_REQUIRED")
    bind(audit["diagnostic"], cpu_path, "LocalD independent audit")
    bind(cost_audit["cost"], cost_path, "actual LocalD whole cost audit")
    bind(cost["local_diagnostic"], cpu_path, "whole cost LocalD source"); bind(cost["local_audit"], cpu_audit_path, "whole cost LocalD audit")
    require(cost_audit["local_diagnostic"] == cost["local_diagnostic"] and cost_audit["local_audit"] == cost["local_audit"], "SAME_LOCAL_AUDIT_CHAIN")
    for key in ("producer", "fit_target_audit", "past_constant_diagnostic", "past_constant_audit", "past_constant_whole_cost", "past_constant_whole_cost_audit"):
        require(audit[key] == cpu[key], "EXACT_PREDECESSOR_CHAIN:" + key)
        bind(cpu[key], resolve(cpu[key]["path"]), key)
    require(cost["producer"] == cost_audit["producer"] == cpu["producer"] and cost["fit_target_audit"] == cost_audit["fit_target_audit"] == cpu["fit_target_audit"], "SAME_FROZEN_FIT_SOURCE")
    expected = {"dataset_split_cells_checked": 6, "backend_states_checked": 24, "completed_backend_batches_checked": 24,
        "method_pairs_checked": 36, "window_matrix_fits_checked": 3264, "channel_slots_checked": 44304,
        "native_seed_D_reuses_checked": 1632, "static_prediction_arrays_byte_preserved": 36,
        "local_prediction_arrays_checked": 36, "static_einsum_control_arrays_checked": 36, "cached_future_arrays_byte_preserved": 36,
        "score_field_checks": 2592, "new_per_origin_output_diagnostic_backend_cells_checked": 24,
        "new_per_origin_output_inequalities_checked": 36, "RAW_bound_inequalities_checked": 0}
    require(all(audit["totals"].get(key) == value for key, value in expected.items()), "FULL_LOCAL_DECODER_AUDIT_COVERAGE")
    require(all(cost_audit["totals"].get(key) == value for key, value in {"cost_cells_checked": 20,
        "recorded_timer_samples_checked": 120, "physical_local_prechecks_checked": 4,
        "physical_local_first8_window_fits_checked": 32, "selected_forecast_file_hashes_checked": 36,
        "canonical_physical_bridges_checked": 2}.items()), "FULL_NEW_LOCAL_WHOLE_COST_AUDIT")
    require(cpu["completed_method_pairs"] == 36 and cpu["completed_backend_batches"] == 24 and not cpu["failed_paths"]
            and cpu["new_E_F_native_head_forward_runs"] == 0 and cpu["new_E_F_or_native_head_training"] is False
            and cpu["future_labels_used_for_local_fit"] is False and cpu["method_checkpoint_threshold_selection_used"] is False,
            "PAST_ONLY_LOCAL_FIT_WITH_IMMUTABLE_CACHED_FORECASTS")
    require(cpu["inference_time_past_decoder_refitting"] is True and cpu["fit_uses_TUNE_DEV_known_past"] is True
            and cpu["same_decoder_side_information_as_static"] is False and cpu["past_C_condition_metadata_added"] is True
            and cpu["static_D_or_global_rankK_oracle_reused_as_local_lower_bound"] is False, "ADAPTIVE_D_INTERFACE_AND_NEW_FLOOR_SCOPE")
    require(cost["new_training_updates"] == 0 and cost["new_backbone_training"] is False
            and cost["F_before"] == cost["F_after"] and tuple(cost["routes"]) == ROUTES
            and cost["completed_cost_cells"] == 20 and cost["recorded_timer_samples"] == 120
            and cost["local_LS_and_implemented_diagnostics_inside_timer"] is True
            and cost["native_heads_loaded_or_timed"] is False and cost["local_coefficients_cached_inside_timed_pipeline"] is False,
            "ACTUAL_FIVE_F_ONLY_COST_ROUTES")
    for data in (cpu, audit, cost, cost_audit): require(data["protected_TEST_accessed"] is False, "NO_PROTECTED_TEST")
    require(cpu["original_source_cache_selection_SHA_unchanged"] is True and cpu["independent_holdout"] is False, "PRIOR_PRESERVATION_EXPOSED_DEV")
    return cpu, audit, cost, cost_audit


def score_rows(base, method, variant, mask, scores, role, extra=None):
    return [{**base, "method": method, "variant": variant, "mask": mask, "metric_scale": scale, **scores[scale], "role": role,
        "this_MSE_is_a_known_per_origin_D_minimum": method == "LOCAL_D_MINIMUM" and scale == "normalized",
        "MAE_is_minimum": False, "physical_MSE_is_minimum": False, "RAW_bound_applied": False, **(extra or {})} for scale in SCALES]


def collect_tables(cpu, audit, cost, cost_audit):
    out = {key: [] for key in ("COMPARISON", "SIGNED_EFFECT", "STATIC_EINSUM_CONTROL", "PAST_RECONSTRUCTION",
        "LOCAL_BACKEND_METADATA", "LOCAL_WINDOW_GEOMETRY", "LOCAL_CHANNEL_SOLVERS", "NEW_PER_ORIGIN_DIAGNOSTIC",
        "NEW_PER_ORIGIN_GEOMETRY", "WHOLE_COST", "WHOLE_COST_SAMPLES", "REGISTERED_ADOPTION")}
    for dataset, (channels, k, dev) in DATASETS.items():
        for split in ("fit", "tune", dev):
            cell = cpu["datasets"][dataset]["splits"][split]
            for mask, key in (("all_original_observed", "RAW_scores_all_observed"), ("all_C_complete", "RAW_scores_complete_mask")):
                out["COMPARISON"] += score_rows({"dataset": dataset, "projection": "NA", "split": split, "backend": "RAW"}, "RAW_BOLT", "STATIC", mask, cell[key], "saved_Raw_baseline")
            for route in PROJECTIONS:
                for backend, methods in BACKENDS.items():
                    branch = cell["routes"][route]["backends"][backend]
                    verified = audit["datasets"][dataset]["splits"][split]["routes"][route]["backends"][backend]
                    metadata = branch["local_fit_metadata"]
                    require(metadata["singular_relative_cut"] == 1e-4 and metadata["future_labels_used_in_local_decoder_fit"] is False
                            and metadata["TUNE_or_DEV_threshold_selection"] is False, "FIXED_OBSERVED_PAST_CENTERED_SLOPE_CUTOFF")
                    base = {"dataset": dataset, "projection": route, "split": split, "backend": backend}
                    diag = verified["new_per_origin_output_space_diagnostic"]
                    require(diag["minimum_only_normalized_MSE_in_each_known_D32_real_rowspace"] is True
                            and diag["old_static_or_global_rank_K_minimum_reused"] is False, "NEW_CONDITIONAL_OUTPUT_DIAGNOSTIC")
                    out["LOCAL_BACKEND_METADATA"].append({**base, "C": channels, "K": k, "local_fit_metadata": metadata,
                        "local_state_artifact": branch["local_state_artifact"], "original_state_source": branch["original_state_source"],
                        "cached_method_sources": branch["cached_method_sources"], "native_cached_past_byte_exact_and_D_shared": branch["native_cached_past_byte_exact_and_D_shared"],
                        "artifact": branch["artifact"], "new_forecast_training": False, "same_side_information_as_static": False})
                    for window in range(metadata["windows"]):
                        out["LOCAL_WINDOW_GEOMETRY"].append({**base, "window_index_zero_based": window,
                            "max_abs_D": metadata["max_abs_decoder_per_window"][window], "D_Frobenius_norm": metadata["decoder_Frobenius_norm_per_window"][window],
                            "D_operator_norm": metadata["decoder_operator_norm_per_window"][window], "new_output_D_rank": diag["rank_D_per_origin"][window],
                            "absolute_origin_value_not_assumed": True, "relative_condition_cap_is_absolute_gain_bound": False})
                        for channel in range(channels):
                            out["LOCAL_CHANNEL_SOLVERS"].append({**base, "window_index_zero_based": window, "channel_index_zero_based": channel,
                                **{key: metadata[key][window][channel] for key in ("true_observed_counts", "centered_slope_ranks", "affine_model_ranks_with_analytic_intercept",
                                    "rank_deficient_slope_channels", "observed_count_below_two", "no_observation_current_C_fallback", "retained_condition_number",
                                    "retained_condition_number_valid", "observed_reconstruction_MSE_real_fit", "observed_reconstruction_MSE_valid")},
                                "singularvalues_observed_centered_Z": metadata["singularvalues_observed_centered_Z"][window][channel]})
                    out["PAST_RECONSTRUCTION"] += score_rows(base, "OBSERVED_PAST_RECONSTRUCTION", "LOCAL", "past_original_observed",
                        branch["past_reconstruction_deployment_scores_observed_only"], "observed_past_fit_only_not_future_objective")
                    out["NEW_PER_ORIGIN_DIAGNOSTIC"] += score_rows(base, "LOCAL_D_MINIMUM", "LOCAL", "all_C_complete", diag["scores"], "known_future_given_known_local_D_real_minimum")
                    out["NEW_PER_ORIGIN_DIAGNOSTIC"] += score_rows(base, "RAW_BOLT", "STATIC", "all_C_complete", diag["RAW_scores_same_complete_mask"], "same_complete_mask_unconstrained_Raw")
                    out["NEW_PER_ORIGIN_GEOMETRY"].append({**base, "rank_D_per_origin": diag["rank_D_per_origin"],
                        "certificate_per_origin": diag["certificate_per_origin"], "minimum_only_normalized_MSE_in_each_known_D32_real_rowspace": True,
                        "old_static_or_global_rank_K_minimum_reused": False})
                    require(set(branch["methods"]) == set(methods), "EXACT_BACKEND_METHODS_AND_SHARED_NATIVE_PAST")
                    for method in methods:
                        values = branch["methods"][method]; checked = verified["methods"][method]
                        for mask, suffix in (("all_original_observed", "all_observed"), ("all_C_complete", "complete_mask")):
                            for variant, prefix in (("STATIC", "static"), ("LOCAL", "local")):
                                out["COMPARISON"] += score_rows(base, method, variant, mask, values[prefix + "_scores_" + suffix],
                                    "frozen_native_head_intervention" if method.startswith("NATIVE") else "cached_Bolt_D_intervention")
                            for scale in SCALES:
                                old, new = values["static_scores_" + suffix][scale], values["local_scores_" + suffix][scale]
                                out["SIGNED_EFFECT"].append({**base, "method": method, "mask": mask, "metric_scale": scale,
                                    "static_MSE": old["mse"], "local_MSE": new["mse"], "local_minus_static_MSE": new["mse"] - old["mse"],
                                    "static_MAE": old["mae"], "local_MAE": new["mae"], "local_minus_static_MAE": new["mae"] - old["mae"],
                                    "statistical_improvement_claimed": False})
                        replay = values["static_D_einsum_arithmetic_replay"]
                        out["STATIC_EINSUM_CONTROL"].append({**base, "method": method, "static_arithmetic_replay": replay, "local_D_effect_includes_arithmetic_change": True})
                        out["NEW_PER_ORIGIN_DIAGNOSTIC"] += score_rows(base, method, "LOCAL", "all_C_complete", checked["local_scores_complete_mask"], "actual_local_prediction_same_complete_mask",
                            {"measured_FP32_rounding_MSE_allowance": checked["measured_FP32_rounding_MSE_allowance"],
                             "local_real_minus_new_origin_specific_minimum_normalized_MSE": checked["local_real_minus_new_origin_specific_minimum_normalized_MSE"],
                             "local_actual_minus_new_origin_specific_minimum_normalized_MSE": checked["local_actual_minus_new_origin_specific_minimum_normalized_MSE"],
                             "new_local_real_arithmetic_scores_complete_mask": checked["new_local_real_arithmetic_scores_complete_mask"]})
        rows = cost["cost"][dataset]["rows"]
        require(len(rows) == 10 and {(row["route"], row["batch"]) for row in rows} == {(route, batch) for route in ROUTES for batch in (1, 8)}, "FIVE_F_ONLY_B1_B8_COST_CELLS")
        for row in rows:
            raw = next(value for value in rows if value["route"] == "RAW_BOLT" and value["batch"] == row["batch"])
            out["WHOLE_COST"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeats": row["repeats"],
                "median_seconds": row["median_seconds"], "median_seconds_per_window": row["median_seconds_per_window"],
                "ratio_of_RAW_medians": row["median_seconds"] / raw["median_seconds"], "median_paired_RAW_ratio": row["paired_RAW_median_ratio"],
                "seconds": row["seconds"], "origins": row["origins"], "memory_shared_process": row["memory"],
                "includes": row["whole_includes"], "excludes": row["whole_excludes"], "memory_scope": row["memory_scope"],
                "selected_metadata": row["selected_metadata"], "reference": row["reference"], "timed_outputs": row["timed_outputs"], "native_timed": False})
            for index, sample in enumerate(row["samples"]): out["WHOLE_COST_SAMPLES"].append({"dataset": dataset, "route": row["route"], "batch": row["batch"], "repeat": sample["repeat"],
                "position": sample["position"], "seconds": sample["seconds"], "paired_RAW_ratio": row["paired_RAW_ratios"][index], "sample": sample})
        for route in ROUTES[1:]:
            projection = "PCA_PINV" if route.startswith("PCA") else "FIT_TARGET_PINV"
            variant = "local" if "LOCAL" in route else "static"
            values = cpu["datasets"][dataset]["splits"]["tune"]["routes"][projection]["backends"]["BOLT_NUMPY"]["methods"]["BOLT"]
            metrics = values[variant + "_scores_all_observed"]["normalized"]
            raw = cpu["datasets"][dataset]["splits"]["tune"]["RAW_scores_all_observed"]["normalized"]
            timing = next(row for row in out["WHOLE_COST"] if row["dataset"] == dataset and row["route"] == route and row["batch"] == 8)
            mse, mae, latency = metrics["mse"] / raw["mse"], metrics["mae"] / raw["mae"], timing["ratio_of_RAW_medians"]
            out["REGISTERED_ADOPTION"].append({"dataset": dataset, "route": route, "TUNE_MSE_ratio_to_RAW": mse,
                "TUNE_MAE_ratio_to_RAW": mae, "B8_ratio_of_RAW_medians": latency,
                "all_registered_conditions_pass": mse <= 1.01 and mae <= 1.01 and latency <= .9, "new_method_selection_used": False})
    require(len(out["COMPARISON"]) == 312 and len(out["SIGNED_EFFECT"]) == 144 and len(out["LOCAL_WINDOW_GEOMETRY"]) == 3264
            and len(out["LOCAL_CHANNEL_SOLVERS"]) == 44304 and len(out["NEW_PER_ORIGIN_DIAGNOSTIC"]) == 168
            and len(out["WHOLE_COST_SAMPLES"]) == 120, "FULL_LOCAL_REPORT_COHORT")
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for si, split in enumerate(("tune", dev)):
            ax = axes[di, si]
            raw = choose(tables["COMPARISON"], dataset=dataset, projection="NA", split=split, method="RAW_BOLT", mask="all_original_observed", metric_scale="normalized")["mse"]
            values = [raw] + [choose(tables["COMPARISON"], dataset=dataset, projection=route, split=split, backend="BOLT_NUMPY", method="BOLT", variant=variant, mask="all_original_observed", metric_scale="normalized")["mse"] for route, variant in (("PCA_PINV", "STATIC"), ("PCA_PINV", "LOCAL"), ("FIT_TARGET_PINV", "STATIC"), ("FIT_TARGET_PINV", "LOCAL"))]
            bars = ax.bar(np.arange(5), values, color=("#666666", "#0072B2", "#56B4E9", "#D55E00", "#E69F00"))
            ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3)
            ax.set_ylim(0, max(values) * 1.22); ax.set_xticks(np.arange(5), ("Raw Bolt", "PCA/static", "PCA/local", "Target/static", "Target/local"), rotation=15, ha="right")
            ax.set_title(f"{DISPLAY[dataset]} / {split.upper()}"); ax.set_ylabel("Normalized macro MSE, all observed target"); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Known-past local D alone: TUNE and exposed DEV must both be inspected", fontsize=13); save(fig, "01_actual_Bolt_comparison")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for pi, route in enumerate(PROJECTIONS):
            ax = axes[di, pi]
            queries = [("BOLT_NUMPY", "LOCAL_D_MINIMUM"), ("NATIVE_TORCH32", "LOCAL_D_MINIMUM"), ("BOLT_NUMPY", "BOLT"), ("NATIVE_TORCH32", "NATIVE_s92601"), ("NATIVE_TORCH32", "NATIVE_s92602")]
            values = [choose(tables["NEW_PER_ORIGIN_DIAGNOSTIC"], dataset=dataset, projection=route, split=dev, backend=backend, method=method, metric_scale="normalized")["mse"] for backend, method in queries]
            raw = choose(tables["NEW_PER_ORIGIN_DIAGNOSTIC"], dataset=dataset, projection=route, split=dev, backend="BOLT_NUMPY", method="RAW_BOLT", metric_scale="normalized")["mse"]
            bars = ax.bar(np.arange(5), values, color=("#CC79A7", "#882255", "#0072B2", "#009E73", "#E69F00"))
            ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=8); ax.axhline(raw, color="#666666", linestyle="--")
            ax.set_ylim(0, max(max(values), raw) * 1.25); ax.set_xticks(np.arange(5), ("D[n] min/Bolt E", "D[n] min/native E", "actual Bolt", "native s1", "native s2"), rotation=17, ha="right")
            ax.set_title(f"{DISPLAY[dataset]} / {route} / {dev.upper()}; dashed Raw={raw:.4f}"); ax.set_ylabel("Normalized MSE on same complete-C positions"); ax.grid(axis="y", alpha=.2)
    fig.suptitle("New known D[n] real minima are diagnostics, and old native heads are NOT reoptimized", fontsize=12); save(fig, "02_new_conditional_space_and_forecast_gap")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8), layout="constrained")
    for di, (dataset, (_, k, dev)) in enumerate(DATASETS.items()):
        groups, labels = [], []
        for split in ("fit", "tune", dev):
            for route in PROJECTIONS:
                rows = [row for row in tables["LOCAL_WINDOW_GEOMETRY"] if row["dataset"] == dataset and row["split"] == split and row["projection"] == route and row["backend"] == "BOLT_NUMPY"]
                groups.append([row["max_abs_D"] for row in rows]); labels.append(split + "/" + ("PCA" if route == "PCA_PINV" else "target"))
        ax = axes[di, 0]; ax.boxplot(groups, tick_labels=labels, showfliers=True)
        ax.set_yscale("symlog", linthresh=1); ax.tick_params(axis="x", labelrotation=20); ax.set_ylabel("Per-window max absolute local D coefficient (symlog)")
        ax.set_title(f"{DISPLAY[dataset]}: relative cutoff does not cap absolute gain"); ax.grid(axis="y", alpha=.2)
        ax = axes[di, 1]
        for rank in range(k + 1):
            counts, bottoms = [], []
            for split in ("fit", "tune", dev):
                for route in PROJECTIONS:
                    rows = [row for row in tables["LOCAL_CHANNEL_SOLVERS"] if row["dataset"] == dataset and row["split"] == split and row["projection"] == route and row["backend"] == "BOLT_NUMPY"]
                    counts.append(sum(row["centered_slope_ranks"] == rank for row in rows) / len(rows))
                    bottoms.append(sum(row["centered_slope_ranks"] < rank for row in rows) / len(rows))
            ax.bar(np.arange(6), counts, bottom=bottoms, label=f"slope rank {rank}")
        ax.set_xticks(np.arange(6), labels, rotation=20); ax.set_ylim(0, 1); ax.set_ylabel("Fraction of window-channel slope fits")
        ax.set_title(f"{DISPLAY[dataset]}: retained centered slope rank; analytic intercept separate")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle("Gain and rank diagnostics; observations/overlapping windows are not independent samples", fontsize=12); save(fig, "03_local_gain_and_rank")
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8), layout="constrained")
    for di, (dataset, (_, _, dev)) in enumerate(DATASETS.items()):
        for pi, route in enumerate(PROJECTIONS):
            ax = axes[di, pi]; past, future = [], []
            for split in ("fit", "tune", dev):
                past.append(choose(tables["PAST_RECONSTRUCTION"], dataset=dataset, projection=route, split=split, backend="BOLT_NUMPY", metric_scale="normalized")["mse"])
                future.append(choose(tables["COMPARISON"], dataset=dataset, projection=route, split=split, backend="BOLT_NUMPY", method="BOLT", variant="LOCAL", mask="all_original_observed", metric_scale="normalized")["mse"])
            x = np.arange(3); ax.bar(x - .18, past, width=.35, color="#009E73", label="Observed past reconstruction")
            ax.bar(x + .18, future, width=.35, color="#D55E00", label="Actual future forecast")
            ax.set_xticks(x, ("FIT", "TUNE", dev.upper())); ax.set_title(f"{DISPLAY[dataset]} / {route}"); ax.set_ylabel("Normalized macro MSE: DIFFERENT target tasks")
            ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Low known-past reconstruction error does not establish future prediction quality", fontsize=13); save(fig, "04_past_reconstruction_vs_future_forecast")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), layout="constrained")
    for di, dataset in enumerate(DATASETS):
        for bi, batch in enumerate((1, 8)):
            ax = axes[di, bi]; rows = [choose(tables["WHOLE_COST"], dataset=dataset, batch=batch, route=route) for route in ROUTES]
            values = [row["median_seconds_per_window"] * 1000 for row in rows]; ax.bar(np.arange(5), values, color=("#666666", "#0072B2", "#D55E00", "#56B4E9", "#E69F00"), alpha=.8)
            maximum = max(max(row["seconds"]) * 1000 / batch for row in rows)
            for index, row in enumerate(rows):
                samples = np.asarray(row["seconds"]) * 1000 / batch; ax.scatter(index + np.linspace(-.18, .18, 6), samples, color="black", s=12)
                ax.text(index, max(values[index], samples.max()) + maximum * .025, f"{values[index]:.2f} ms\nratio {row['ratio_of_RAW_medians']:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_ylim(0, maximum * 1.32); ax.set_xticks(np.arange(5), ("Raw", "PCA/static", "Target/static", "PCA/local", "Target/local"), rotation=17, ha="right")
            ax.set_title(f"{DISPLAY[dataset]} / B={batch} / six actual timer samples"); ax.set_ylabel("Whole latency per window (ms)"); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Actual F-only whole costs include inference-time local decoder fitting; no native cost", fontsize=12); save(fig, "05_actual_local_whole_cost")
    return paths


def text_report(cpu, audit, cost, tables, linked, output):
    lines = ["# 과거만으로 각 창의 디코더를 다시 맞추면 해결되는가", "",
        "이번 대조는 고정된 E와 저장된 잠재 미래를 유지하고, 각 예측 시점의 알려진 과거에서 D[n]을 다시 맞춘 것이다. 일부 개발 구간에서 Bolt 오차가 줄더라도, 같은 TUNE·Raw 비교·전체 비용을 충족해야 작은 K 문제의 해결로 볼 수 있다. 아래는 실제 원수치와 비교 조건의 답부터 제시한다.", ""]
    for dataset, (_, _, dev) in DATASETS.items():
        values = []
        for split in ("tune", dev):
            old = choose(tables["COMPARISON"], dataset=dataset, projection="FIT_TARGET_PINV", split=split, backend="BOLT_NUMPY", method="BOLT", variant="STATIC", mask="all_original_observed", metric_scale="normalized")["mse"]
            new = choose(tables["COMPARISON"], dataset=dataset, projection="FIT_TARGET_PINV", split=split, backend="BOLT_NUMPY", method="BOLT", variant="LOCAL", mask="all_original_observed", metric_scale="normalized")["mse"]
            raw = choose(tables["COMPARISON"], dataset=dataset, projection="NA", split=split, method="RAW_BOLT", mask="all_original_observed", metric_scale="normalized")["mse"]
            values.append(f"{split.upper()}: static {old:.9f} → local {new:.9f}, Raw {raw:.9f}")
        lines.append(f"[확인] {DISPLAY[dataset]} FIT-target/Bolt의 정규화 MSE는 " + "; ".join(values) + ". 개발 오차의 감소와 Raw 정확도 보존 여부, TUNE의 방향은 별개다.")
    lines += ["", "## 과거 입력과 새로운 decoder 정보", "",
        "각 origin에서 512개 정규화 과거의 cached E(X)와 원 관측 C 과거만으로 채널별 affine decoder를 fitting한다. observed subset의 latent·C 평균을 먼저 제거하고 centered slope에 고정 상대 singular cutoff1e-4를 적용한 truncated minimum-norm least squares다. intercept는 관측된 평균으로 계산하고 cutoff의 대상이 아니다. 이는 uncentered [Z,1] 전체 coefficient의 minimum norm이나 unrestricted full-design least squares라는 주장이 아니다. residual의 retained singular 방향 검산도 버린 nonzero 방향의 전체 gradient가0이라는 검산으로 바꾸지 않는다.", "",
        "TUNE/DEV에서 이미 알려진 과거를 사용하는 inference-time adaptation이며 미래 target으로 D를 맞추지 않는다. 원 FIT-target E 자체는 이전 단계에서 FIT 미래 supervision을 받은 E다. 지금의 과거 전용 local fitting을 ‘전체 표현학습 과정에도 미래 label이 없었다’고 부를 수 없다. raw 과거 입력은 같아도 decoder는 K future streams·current C 외에 각 창의 K×C coefficients를 받으므로 정적 decoder와 side information이 같지 않다. 엄격한 K-only 압축이나 표현학습 개선이라고 주장하지 않는다. 계수·bias는 FP64로 fitting한 뒤 FP32 배포하며 전송 bit packing 비용을 측정하지 않았다.", "",
        "예측은 FP32 `currentC + einsum(futureK-lastK,D[n],optimize=False)`이며 bias는 innovation에서 제외한다. old static D를 같은 einsum 순서로 replay한 산술 대조도 따로 기록했다. 원 static 예측 비트를 유지한 데이터와 새 산술 replay 차이를 구분한다. zero latent delta는 currentC의 정확한 FP32 수치값을 내지만 addition이 signed-zero 비트까지 보존한다는 주장은 아니다. no-observation 채널의 명시 규칙은 D column0·reconstruction bias=currentC·forecast=currentC이다. 이것은 전역 static D fallback이나 미래 최적성의 보장이 아니다.", "",
        f"[확인] CPU 완료 범위는 {cpu['completed_method_pairs']} method pairs, {cpu['completed_backend_batches']} backend batches, {cpu['completed_window_matrix_fits']} window matrices, {cpu['completed_channel_solver_calls']} channel solver calls다. no-observation fallback 기록은 {cpu['completed_no_observation_fallbacks']}건이다. 새 E/F/head forward와 backbone/head 학습은0이다. Bolt NumPy와 native Torch32의 cached 과거를 각각 사용했다. native 두 seed의 과거가 byte-identical인 경우 D를 공유하며 같은 local fit을 두 개의 독립 fitting으로 세지 않았다. native head는 원 static D에 맞춘 기존 head이므로 새 D를 위한 최적화된 학습 benchmark가 아니다.", "",
        "## 실제 예측과 비교의 한계", "", "```text", "Dataset / Split / Space / Backend / Method    static MSE       local MSE      local-minus-static"]
    for row in tables["SIGNED_EFFECT"]:
        if row["split"] != "fit" and row["mask"] == "all_original_observed" and row["metric_scale"] == "normalized":
            lines.append(f"{row['dataset']}/{row['split']}/{row['projection']}/{row['backend']}/{row['method']} {row['static_MSE']:.9f} {row['local_MSE']:.9f} {row['local_minus_static_MSE']:+.6e}")
    lines += ["```", "", "![Actual Bolt output intervention](figures/01_actual_Bolt_comparison.png)", "",
        "full observed 정규화 macro MSE·MAE를 주 비교로 보고하고 complete-C 및 물리 단위 채널별 진단은 별도로 남겼다. 서로 다른 단위의 물리 macro 평균은 보조값이며 정규화 목적의 수학적 최소값과 혼합하지 않는다. 모든 native seed와 조건의 나쁜 결과도 표에 포함한다. 개발 구간에서 좋아진 특정 숫자를 TUNE의 불안정성이나 native 폭증을 지운 성공으로 설명하지 않는다. 이전 POINTWISE/MAIN과 native에도 actual-Y 목적 학습이 있었으므로 기존 실패를 reconstruction-only 학습이라는 하나의 원인으로 정리하지 않는다.", "",
        "## 새 per-origin 출력 공간과 forecast gap", "",
        "D가 origin마다 달라져 이전 static fixed-D 최소값이나 per-split global rank-K oracle를 이 예측기의 하한으로 재사용할 수 없다. 독립 감사는 각 이미 fitting된 D32[n]을 실수 연산으로 해석하고, 같은 complete-C 미래 `Y-currentC`를 해당 행공간에 직접 투영한 새 진단을 계산했다. 실제 미래를 쓰는 진단이며 배포·D fitting·threshold/checkpoint 선택에 넣지 않았다. 같은 알려진 D[n]에서의 정규화 MSE 최소값만 의미한다. MAE와 물리 단위 MSE는 이 최소해의 점수이고 각각의 최소값이 아니다. Raw에는 그 공간의 부등식을 적용하지 않는다. FP32 rounding allowance와 actual-vs-real 차이도 따로 기록했다.", "", "```text", "Dataset / Split / Space / Backend     new D[n] minimum MSE    actual local predictor MSE"]
    for row in tables["NEW_PER_ORIGIN_DIAGNOSTIC"]:
        if row["metric_scale"] == "normalized" and row["method"] not in ("LOCAL_D_MINIMUM", "RAW_BOLT"):
            minimum = choose(tables["NEW_PER_ORIGIN_DIAGNOSTIC"], dataset=row["dataset"], projection=row["projection"], split=row["split"], backend=row["backend"], method="LOCAL_D_MINIMUM", metric_scale="normalized")["mse"]
            lines.append(f"{row['dataset']}/{row['split']}/{row['projection']}/{row['backend']}/{row['method']} {minimum:.9f} {row['mse']:.9f}")
    lines += ["```", "", "![New local output space and frozen forecast gap](figures/02_new_conditional_space_and_forecast_gap.png)", "",
        "[추정·다음 판단] 이 최소값이 낮아도 실제 F/native가 크게 나쁘면 해당 과거 표현에서의 잠재 미래 예측·오차 크기·D의 증폭 또는 기존 predictor와의 결합이 남는다. 새 최소값 자체가 높으면 local past로 고른 출력 공간도 그 미래를 잘 담지 못한 것이다. 두 단계를 분리해 판단하며 알려진 미래의 최소값을 이미 얻은 예측 성능이나 과거 정보의 충분성 증명으로 바꾸지 않는다.", "",
        "## gain·rank·과거 재구성은 서로 다른 확인", "",
        "상대 cutoff는 retained latent singular 방향 사이의 condition을 제어하지만, 작은 절대 singular scale이나 C response 크기에 따른 절대 D norm을 cap하지 않는다. 높은 max-abs coefficient, Frobenius/operator norm과 미래 innovation이 만나면 출력 오차를 크게 만들 수 있다. metadata는 centered slope rank, analytic intercept를 포함한 rank, true observed count, no-observation 규칙, per-window D norm을 모두 보존한다. 큰 D와 큰 오차가 같이 나타난다고 하나만의 인과 원인으로 확정하지 않고, metric·rank·reconstruction·future gap을 함께 본다.", "", "![Local gain and retained ranks](figures/03_local_gain_and_rank.png)", "",
        "과거 reconstruction 점수는 D를 fitting한 각 관측 과거에서의 별도 과제다. 미래 forecast 점수와 target 집합·표본 수가 다르다. 과거를 낮은 오차로 복원해도 미래 innovation을 그 D에 통과시킬 때 좋은 예측이 된다는 보장은 없다. 이 관찰을 전체 표현학습이나 예측 목적 학습이 불가능하다는 주장으로 확대하지 않는다. 원수치가 없는 static past reconstruction 개선을 추정해서 덧붙이지도 않는다.", "", "![Past reconstruction and future task](figures/04_past_reconstruction_vs_future_forecast.png)", "",
        "## 실제 whole deployment 비용", "",
        "새 비용은 두 데이터 × B1/B8 × Raw/static PCA/static FIT-target/local PCA/local FIT-target의20조건, 각6회로120시간 표본이다. native 비용은 없다. local route는 실제 물리 과거 causal fill·FIT normalization·E·관측 mask로 local D fitting·고정 Bolt forward·D 복원·물리 CPU return을 실행했다. 매번 D를 다시 계산하며 계수를 cache하지 않는다. `fit_local_past_decoder` 안의 input-array byte hash와 solver·rank·norm·reconstruction 진단 생성도 callback과 타이머 안에 포함하므로 이를 최적화된 최소 production fitting 비용이라고 주장하지 않는다. `LocalPastDecoder.metadata()`의 state-array hash, NPZ/source/artifact 파일 hash·저장, 모델·입력·state loading, warmup·참조 및 타이머 밖 guard와 memory snapshot은 제외했다. saved future만 넣는 decoder proxy 비용이 아니다. 같은 첫 B개 개발 origins의 반복, 공유 프로세스 RSS/VRAM, 짧은6회 기술통계이며 quiet/exclusive benchmark를 확인하지 않았다. worker fixture 실행의 정확한 timer overlap을 확정할 수 없어 timing 차이를 decoder만의 인과 효과나 환경 전체의 유의한 속도 일반화로 주장하지 않는다. 독점 memory 절감도 측정하지 않았다.", "", "```text", "Dataset / Route / B     median ms/window    ratio of Raw medians    median paired ratio"]
    for row in tables["WHOLE_COST"]: lines.append(f"{row['dataset']}/{row['route']}/B{row['batch']} {row['median_seconds_per_window']*1000:.6f} {row['ratio_of_RAW_medians']:.6f} {row['median_paired_RAW_ratio']:.6f}")
    passes = sum(row["all_registered_conditions_pass"] for row in tables["REGISTERED_ADOPTION"])
    lines += ["```", "", "![Actual inference-time local fitting cost](figures/05_actual_local_whole_cost.png)", "",
        f"사전 기준은 같은 ungated Raw TUNE MSE·MAE의1.01배 이하 및 B8 중앙 latency 비율0.90 이하를 함께 요구한다. 해당 기준 동시 충족은 {passes}/{len(tables['REGISTERED_ADOPTION'])}개다. paired ratio 중앙값은 별도 기술통계이고 gate에는 ratio of medians를 쓴다. 이 숫자는 기준 검산이고 결과를 보고 새 모델을 고른 것은 아니다. 이전 시행의 절대 latency와 연결해 local fitting 가속 효과를 주장하지 않는다. 과거 정보를 fitting하는 계산 비용과 extra coefficients 정보를 정확도 개선과 같이 평가해야 한다.", "",
        "## 완료한 질문과 미해결인 질문", "",
        "[확인] 각 origin의 과거 관측만으로 실행 가능한 local affine decoder, backend별 cached E arithmetic 대응, finite solver·retained-direction 검산, 실제 future output과 새 D[n] 공간 진단, 비용을 완료했다. 이 개입은 알려진 C 과거의 조건 정보를 추가한 causal decoding 대조이고 새로운 backbone/표현 학습은 아니다. 이미 노출된 DEV와 이전 oracle를 보고 고른 탐색적 후속이며 독립 holdout 일반화 검증이 아니다. TEST는 읽지 않았고 기존 source·cache·checkpoint·selection lock을 덮어쓰지 않았다.", "",
        "[추정·다음 판단] 유리한 개발 구간 감소가 있어도 TUNE·Raw·비용과 잘못된 native 연결이 함께 남으면 해결이라고 쓰지 않는다. gain을 제한한 대조나 미래 innovation에 맞춘 decoder 목적, 기존 latent predictor와의 공동 적합은 별도 조건에서 시험해야 한다. 이 다음 방향을 이미 실행했다거나, 대조학습·동적 graph·ExoTimer 또는 고정 Bolt의 필요를 입증했다고 말하지 않는다. 원 목적은 작은 K와 고정 Bolt에서 정확도·비용을 유지하는 것이며 이 진단의 수행 자체가 그 목적의 달성은 아니다.", "",
        "## 원수치·소스·이전 보고서", ""]
    for name, path in linked.items(): lines.append(f"- [{name}]({rel(path, output)}) — SHA256 `{sha(path)}`")
    for name, rows in tables.items(): lines.append(f"- [{name}: {len(rows)} rows](tables/{name}.csv)")
    lines += ["", "이 보고서 생성기는 JSON metadata만 읽으며 NPZ·model·checkpoint나 scientific runtime을 호출하지 않는다. 보고서·실험 source를 보존했고 자동 commit/push나 기존 작업 종료를 수행하지 않았다.", ""]
    return "\n".join(lines)


def generate(cpu_path, cpu_audit_path, cost_path, cost_audit_path, prior_reports, output):
    cpu, audit, cost, cost_audit = load_verified(cpu_path, cpu_audit_path, cost_path, cost_audit_path)
    require(len(prior_reports) == 5 and all(Path(path).is_file() and Path(path).suffix == ".md" for path in prior_reports), "FIVE_PRIOR_MD_REPORTS")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False); (output / "tables").mkdir()
    tables = collect_tables(cpu, audit, cost, cost_audit)
    for name, rows in tables.items(): write_csv(output / "tables" / (name + ".csv"), rows)
    images = figures(tables, output)
    linked = {"LocalD CPU producer": cpu_path, "LocalD independent audit": cpu_audit_path, "actual LocalD wholecost": cost_path,
        "LocalD wholecost independent audit": cost_audit_path, "matched report": prior_reports[0], "rank oracle v2": prior_reports[1],
        "D metric v2": prior_reports[2], "FIT target report": prior_reports[3], "past-constant gate report": prior_reports[4], "report generator": Path(__file__)}
    (output / "REPORT_KO.md").write_text(text_report(cpu, audit, cost, tables, linked, output), encoding="utf-8")
    manifest = {"status": "COMPLETE", "source": record(Path(__file__)), "CPU_producer": record(cpu_path), "CPU_audit": record(cpu_audit_path),
        "cost": record(cost_path), "cost_audit": record(cost_audit_path), "prior_reports": [record(path) for path in prior_reports],
        "source_bindings": {name: record(path) for name, path in linked.items()}, "metadata_only": True,
        "NPZ_or_model_opened": False, "training_or_F_forward": False, "protected_TEST_accessed": False,
        "strict_K_only_compression": False, "decoder_side_information_changed": True, "native_reoptimized_for_local_D": False,
        "new_native_deployment_cost_measured": False, "old_static_or_global_K_floor_reused": False,
        "MAE_or_physical_MSE_minimum_claimed": False, "RAW_bound_applied": False, "independent_generalization_claimed": False,
        "CSV_rows": {name: len(rows) for name, rows in tables.items()}, "figures": [rel(path, output) for path in images],
        "files": [{"path": rel(path, output), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()]}
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as handle: json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("cpu", "cpu-audit", "cost", "cost-audit", "main-report", "rank-report", "decoder-report", "fit-report", "gate-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    receipt = generate(args.cpu, args.cpu_audit, args.cost, args.cost_audit,
        (args.main_report, args.rank_report, args.decoder_report, args.fit_report, args.gate_report), args.output)
    print(json.dumps({"status": receipt["status"], "CSV_rows": receipt["CSV_rows"]}))
