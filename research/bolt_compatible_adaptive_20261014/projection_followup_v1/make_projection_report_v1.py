"""Create a report from independently audited terminal JSON; never open experiment arrays."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASETS = ("robin", "beijing_aotizhongxin")
PROJECTIONS = ("PCA", "FORECA")
SEEDS = ("92601", "92602")
SCALES = ("normalized", "physical_per_channel_diagnostic")
LABELS = {"RAW_BOLT": "Raw Bolt", "PCA_BOLT": "PCA / Bolt", "FORECA_BOLT": "ForeCA / Bolt",
          "PCA_NATIVE_92601": "PCA / native s1", "PCA_NATIVE_92602": "PCA / native s2",
          "FORECA_NATIVE_92601": "ForeCA / native s1", "FORECA_NATIVE_92602": "ForeCA / native s2"}
COLORS = {"RAW_BOLT": "#444444", "PCA_BOLT": "#0072B2", "FORECA_BOLT": "#D55E00",
          "PCA_NATIVE_92601": "#56B4E9", "PCA_NATIVE_92602": "#009E73",
          "FORECA_NATIVE_92601": "#E69F00", "FORECA_NATIVE_92602": "#CC79A7"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path).resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("REPORT_METADATA_JSON_ONLY")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def link(source, target):
    return Path(os.path.relpath(Path(target).resolve(), Path(source).resolve().parent)).as_posix()


def load_verified(result_path, audit_path):
    result, audit = read_json(result_path), read_json(audit_path)
    if result.get("status") != "COMPLETE" or audit.get("status") != "PASS":
        raise ValueError("REPORT_REQUIRES_COMPLETE_PRODUCER_AND_PASS_AUDIT")
    if audit.get("producer", {}).get("sha256") != sha(result_path):
        raise ValueError("REPORT_AUDIT_PRODUCER_SHA_MISMATCH")
    if result.get("protected_TEST_accessed") is not False or result.get("new_backbone_training") is not False:
        raise ValueError("REPORT_PROTECTED_SCOPE_MISMATCH")
    totals = audit.get("totals", {})
    expected = {"projection_states_checked": 4, "native_fits_checked": 8,
                "checkpoint_states_checked": 40, "cost_cells_checked": 20,
                "recorded_timer_samples_checked": 120}
    if any(totals.get(key) != value for key, value in expected.items()):
        raise ValueError("REPORT_AUDIT_REGISTERED_COUNTS_MISMATCH")
    return result, audit


def load_diagnostic(result_path, audit_path, diagnostic_path, diagnostic_audit_path):
    if diagnostic_path is None and diagnostic_audit_path is None:
        return None
    if diagnostic_path is None or diagnostic_audit_path is None:
        raise ValueError("FIXED_D_DIAGNOSTIC_REQUIRES_INDEPENDENT_AUDIT")
    diagnostic, audit = read_json(diagnostic_path), read_json(diagnostic_audit_path)
    if diagnostic.get("status") != "COMPLETE" or audit.get("status") != "PASS":
        raise ValueError("FIXED_D_DIAGNOSTIC_REQUIRES_COMPLETE_AND_PASS")
    if audit.get("diagnostic", {}).get("sha256") != sha(diagnostic_path):
        raise ValueError("FIXED_D_DIAGNOSTIC_AUDIT_SHA_MISMATCH")
    if diagnostic.get("producer", {}).get("sha256") != sha(result_path):
        raise ValueError("FIXED_D_DIAGNOSTIC_PRODUCER_SHA_MISMATCH")
    if diagnostic.get("independent_projection_audit", {}).get("sha256") != sha(audit_path):
        raise ValueError("FIXED_D_DIAGNOSTIC_PROJECTION_AUDIT_SHA_MISMATCH")
    if audit.get("producer", {}).get("sha256") != sha(result_path):
        raise ValueError("FIXED_D_INDEPENDENT_AUDIT_PRODUCER_SHA_MISMATCH")
    if audit.get("projection_audit", {}).get("sha256") != sha(audit_path):
        raise ValueError("FIXED_D_INDEPENDENT_PROJECTION_AUDIT_SHA_MISMATCH")
    if diagnostic.get("protected_TEST_accessed") is not False or diagnostic.get("checkpoint_or_method_or_branch_selection_changed") is not False:
        raise ValueError("FIXED_D_DIAGNOSTIC_SCOPE_MISMATCH")
    if diagnostic.get("diagnostic_cells") != 12 or diagnostic.get("original_producer_selected_forecasts_changed") is not False:
        raise ValueError("FIXED_D_DIAGNOSTIC_REGISTERED_COHORT_MISMATCH")
    return diagnostic


def diagnostic_tables(diagnostic):
    rows, geometry = [], []
    for dataset, data in diagnostic["datasets"].items():
        for route, splits in data["projections"].items():
            for split, cell in splits.items():
                metrics = [("FIXED_D_OPTIMAL_REAL_LATENT", cell["fixed_D_bound_scores_same_mask"], None)]
                metrics.extend((method, value["saved_prediction_scores_same_mask"], value)
                               for method, value in cell["comparisons"].items())
                floor = cell["fixed_D_bound_scores_same_mask"]["normalized"]["mse"]
                for method, scores, comparison in metrics:
                    for scale in SCALES:
                        row = score_row(dataset, method, split, scale, scores[scale], projection=route,
                                        role="future_actual_Y_fixed_D_diagnostic_only",
                                        complete=cell["complete_target_positions"], total=cell["total_target_positions"])
                        row.update({"fixed_D_real_normalized_MSE_minimum": floor,
                                    "this_MAE_is_a_minimum": False, "this_physical_MSE_is_a_minimum": False,
                                    "Raw_bound_inequality_applied": False,
                                    "saved_FP32_to_real_MSE_allowance": None if comparison is None else comparison.get("measured_saved_FP32_vs_real_MSE_allowance")})
                        rows.append(row)
                geometry.append({"dataset": dataset, "projection": route, "split": split,
                                 "complete_target_positions": cell["complete_target_positions"],
                                 "total_target_positions": cell["total_target_positions"],
                                 **cell["geometry"], **cell["independent_linear_checks"]})
    return {"FIXED_D_MATCHED_MASK": rows, "FIXED_D_GEOMETRY": geometry}


def score_row(dataset, method, split, scale, score, *, seed=None, projection=None, selected_update=None,
              role="deployment", complete=None, total=None):
    return {"dataset": dataset, "method": method, "seed": seed, "projection": projection,
            "split": split, "split_role": {"fit": "training", "tune": "selection"}.get(split, "exposed_development"),
            "metric_scale": scale, "role": role, "mse": score["mse"], "mae": score["mae"],
            "active_channels": score["active_channels"], "observed_counts": score["observed_counts"],
            "per_channel_mse": score["per_channel_mse"], "per_channel_mae": score["per_channel_mae"],
            "selected_update": selected_update, "complete_target_positions": complete,
            "total_target_positions": total, "independent_holdout": False,
            "physical_channel_units_comparable": False}


def collect_tables(result, audit):
    scores, oracles, checkpoints, curves, spectra, costs, cost_samples = [], [], [], [], [], [], []
    training_cost = []
    for dataset in DATASETS:
        data = result["datasets"][dataset]
        for split, metrics in data["raw_scores"].items():
            for scale in SCALES:
                scores.append(score_row(dataset, "RAW_BOLT", split, scale, metrics[scale]))
        for route in PROJECTIONS:
            condition = data["projections"][route]
            for split, metrics in condition["bolt"]["scores"].items():
                for scale in SCALES:
                    scores.append(score_row(dataset, route + "_BOLT", split, scale, metrics[scale], projection=route))
            for seed in SEEDS:
                native = condition["native"][seed]
                training_cost.append({"dataset": dataset, "projection": route, "seed": int(seed),
                                      "head_parameters": native["metadata"]["parameters"],
                                      "updates": native["metadata"]["updates"],
                                      "selected_update": native["metadata"]["selected_update"],
                                      "train_wall_seconds": native["metadata"]["train_wall_seconds"],
                                      "head_device": "cpu", "E_D_frozen": True,
                                      "scope": "Native fitting including latent cache, checkpoint scoring/callbacks; not isolated optimizer time"})
                for split, metrics in native["scores"].items():
                    for scale in SCALES:
                        scores.append(score_row(dataset, route + "_NATIVE", split, scale, metrics[scale],
                                                seed=int(seed), projection=route,
                                                selected_update=native["metadata"]["selected_update"]))
                for checkpoint in native["checkpoints"]:
                    checkpoints.append({"dataset": dataset, "projection": route, "seed": int(seed),
                                        "update": checkpoint["update"], "tune_mse": checkpoint["tune"]["mse"],
                                        "tune_mae": checkpoint["tune"]["mae"],
                                        "selected": checkpoint["update"] == native["metadata"]["selected_update"],
                                        "state_sha256": checkpoint["state_sha256"]})
                curves.extend({"dataset": dataset, "projection": route, "seed": int(seed), **row}
                              for row in native["learning_curve"])
            for split, oracle in condition["oracle"].items():
                entries = [("EY_ORACLE", None, oracle["scores"]), ("RAW_BOLT", None, oracle["raw_scores_same_mask"]),
                           (route + "_BOLT", None, oracle["bolt_scores_same_mask"])]
                entries.extend((route + "_NATIVE", int(seed), oracle["native_scores_same_mask"][seed]) for seed in SEEDS)
                for method, seed, metrics in entries:
                    for scale in SCALES:
                        oracles.append(score_row(dataset, method, split, scale, metrics[scale], seed=seed, projection=route,
                                                 role="future_EY_diagnostic_matched_complete_mask",
                                                 complete=oracle["complete_target_positions"], total=oracle["total_target_positions"]))
            geometry = audit["datasets"][dataset]["projections"][route]["projection"]
            for component, entropy in enumerate(geometry["FIT_spectral_normalized_entropy"]):
                spectra.append({"dataset": dataset, "projection": route, "component": component + 1,
                                "FIT_normalized_spectral_entropy": entropy,
                                "FIT_spectral_forecastability": geometry["FIT_spectral_forecastability"][component],
                                "FIT_latent_variance": geometry["latent_covariance"][component][component],
                                "effective_rank": geometry["effective_rank"], "K": geometry["components"],
                                "FIT_positions_with_overlap": geometry["FIT_rows_with_overlap"],
                                "decoder_normal_equation_max_abs": geometry["decoder_normal_equation_max_abs"],
                                "independent_samples_claimed": False})
        for row in result["cost"][dataset]["rows"]:
            costs.append({"dataset": dataset, "method": row["route"], "native_seed": row["selected_metadata"].get("native_seed"),
                          "batch": row["batch"], "warmup": row["warmup_each_route"], "repeats": row["repeats"],
                          "median_seconds": row["median_seconds"],
                          "median_ms_per_window": row["median_seconds_per_window"] * 1000,
                          "paired_RAW_median_ratio": row["paired_RAW_median_ratio"],
                          "positions": row["positions"], "seconds": row["seconds"],
                          "resident_rss_max_bytes": max(item["resident_rss_bytes"] for item in row["memory"]),
                          "shared_peak_allocated_vram_max_bytes": max(item["peak_allocated_vram_bytes"] for item in row["memory"]),
                          "shared_peak_reserved_vram_max_bytes": max(item["peak_reserved_vram_bytes"] for item in row["memory"]),
                          "isolated_method_memory": False, "all_six_orders_perfectly_balanced": False})
            cost_samples.extend({"dataset": dataset, "method": row["route"], "batch": row["batch"],
                                 "repeat": sample["repeat"], "position": sample["position"],
                                 "seconds": sample["seconds"], "ms_per_window": sample["seconds"] / row["batch"] * 1000,
                                 **sample["memory"]} for sample in row["samples"])
    tables = {"COMPARISON": scores, "ORACLE_MATCHED_MASK": oracles, "CHECKPOINT_SELECTION": checkpoints,
            "TRAINING_CURVE": curves, "TRAINING_COST": training_cost,
            "SPECTRAL_GEOMETRY": spectra, "COST": costs, "COST_SAMPLES": cost_samples}
    adoption = []
    for dataset in DATASETS:
        tune = selected_rows(tables, dataset, "tune")
        b8 = {row["method"]: row for row in costs if row["dataset"] == dataset and row["batch"] == 8}
        for method in ("PCA_BOLT", "FORECA_BOLT", "PCA_NATIVE", "FORECA_NATIVE"):
            scored = tune[method + ("_92601" if method.endswith("NATIVE") else "")]
            raw = tune["RAW_BOLT"]
            valid = raw["mse"] > 1e-8 and raw["mae"] > 1e-8
            mse_ratio = scored["mse"] / raw["mse"] if valid else None
            mae_ratio = scored["mae"] / raw["mae"] if valid else None
            latency_ratio = b8[method]["median_ms_per_window"] / b8["RAW_BOLT"]["median_ms_per_window"]
            accuracy_ok = valid and mse_ratio <= 1.01 and mae_ratio <= 1.01
            speed_ok = latency_ratio <= .90
            adoption.append({"dataset": dataset, "method": method,
                             "native_seed": 92601 if method.endswith("NATIVE") else None,
                             "TUNE_MSE_ratio_to_Raw": mse_ratio, "TUNE_MAE_ratio_to_Raw": mae_ratio,
                             "B8_ratio_of_median_whole_latency": latency_ratio,
                             "B8_median_of_paired_Raw_ratios": b8[method]["paired_RAW_median_ratio"],
                             "accuracy_within_1pct": accuracy_ok, "median_latency_reduced_10pct": speed_ok,
                             "exploratory_cost_tradeoff_candidate": bool(accuracy_ok and speed_ok),
                             "scope": "Registered exploratory TUNE accuracy + same-trial B8 median latency; not significance or independent validation"})
    tables["COST_ADOPTION"] = adoption
    return tables


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def selected_rows(tables, dataset, split):
    rows = [row for row in tables["COMPARISON"] if row["dataset"] == dataset and row["split"] == split
            and row["metric_scale"] == "normalized"]
    return {row["method"] + ("_" + str(row["seed"]) if row["seed"] is not None else ""): row for row in rows}


def make_figures(result, audit, tables, output, diagnostic=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    figures = []

    def save(fig, name, title, footer):
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.supxlabel(footer, fontsize=9)
        for extension in ("png", "svg"):
            path = output / "figures" / (name + "." + extension)
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append(name)

    methods = list(LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for i, dataset in enumerate(DATASETS):
        split = "val" if dataset == "robin" else "eval"
        rows = selected_rows(tables, dataset, split)
        for j, metric in enumerate(("mse", "mae")):
            ax = axes[i, j]
            values = [rows[method][metric] for method in methods]
            ax.barh(range(7), values, color=[COLORS[method] for method in methods])
            ax.set_yticks(range(7), [LABELS[method] for method in methods])
            ax.invert_yaxis(); ax.set_xlim(0, max(values) * 1.22)
            for y, value in enumerate(values): ax.text(value + max(values) * .02, y, f"{value:.5f}", va="center", fontsize=9)
            ax.set_title(f"{dataset}: {split.upper()} {metric.upper()}")
            ax.set_xlabel("Observed channel-macro normalized " + metric.upper()); ax.grid(axis="x", alpha=.2)
    save(fig, "01_development_scores", "Fixed K + current-C anchor: actual development errors",
         "Exposed development windows; frozen FIT E/D; native seeds change minibatch order, not zero initialization.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for ax, dataset in zip(axes, DATASETS):
        dev = "val" if dataset == "robin" else "eval"
        tune, development = selected_rows(tables, dataset, "tune"), selected_rows(tables, dataset, dev)
        for method in methods:
            ax.plot([0, 1], [tune[method]["mse"], development[method]["mse"]], "o-", color=COLORS[method], label=LABELS[method])
        ax.set_xticks([0, 1], ["TUNE / selection", dev.upper() + " / exposed development"])
        ax.set_yscale("log"); ax.set_ylabel("Normalized macro MSE (log scale)"); ax.set_title(dataset)
        ax.grid(alpha=.2); ax.legend(fontsize=8, loc="best")
    save(fig, "02_selection_development_discrepancy", "Selection-period and development-period errors",
         "Same selected checkpoint in both periods. Different periods and overlapping windows do not form independent replications.")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for i, dataset in enumerate(DATASETS):
        k = result["datasets"][dataset]["K"]
        for route, offset, color in (("PCA", -.17, "#0072B2"), ("FORECA", .17, "#D55E00")):
            geometry = audit["datasets"][dataset]["projections"][route]["projection"]
            axes[i, 0].bar(np.arange(k) + offset, geometry["FIT_spectral_normalized_entropy"], .34, color=color, label=route)
        axes[i, 0].set_xticks(range(k), ["z" + str(index + 1) for index in range(k)])
        axes[i, 0].set_ylim(0, 1); axes[i, 0].set_ylabel("FIT spectral entropy / log(257)")
        axes[i, 0].set_title(dataset + ": spectral criterion"); axes[i, 0].legend(); axes[i, 0].grid(axis="y", alpha=.2)
        covariance = np.vstack([audit["datasets"][dataset]["projections"][route]["projection"]["latent_covariance"] for route in PROJECTIONS])
        axes[i, 1].imshow(covariance, cmap="cividis", vmin=0, vmax=1, aspect="auto")
        axes[i, 1].set_xticks(range(k), ["z" + str(index + 1) for index in range(k)])
        axes[i, 1].set_yticks(range(2*k), [route + " z" + str(index + 1) for route in PROJECTIONS for index in range(k)])
        for y in range(2*k):
            for x in range(k): axes[i, 1].text(x, y, f"{covariance[y,x]:.0f}", ha="center", va="center", color="black" if covariance[y,x] > .5 else "white")
        rank = audit["datasets"][dataset]["projections"]["PCA"]["projection"]["effective_rank"]
        axes[i, 1].set_title(f"FIT covariance (both whitened): K={k}, rank={rank}")
    save(fig, "03_fit_spectrum_and_geometry", "FIT spectral selection and non-collapsed coordinate constraints",
         "Independent audit arithmetic; 256 x 512 overlapping FIT positions. Lower entropy is not a future-H48 accuracy guarantee.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for i, dataset in enumerate(DATASETS):
        for j, route in enumerate(PROJECTIONS):
            ax = axes[i, j]; tune_axis = ax.twinx()
            tune_axis.spines["right"].set_visible(True)
            for seed, color in (("92601", "#0072B2"), ("92602", "#D55E00")):
                native = result["datasets"][dataset]["projections"][route]["native"][seed]
                curve = native["learning_curve"]
                x = np.array([row["update"] for row in curve]); y = np.array([row["loss"] for row in curve])
                ax.plot(x, y, color=color, alpha=.14, linewidth=.7)
                smooth = np.convolve(y, np.ones(20)/20, mode="valid")
                ax.plot(x[19:], smooth, color=color, label=f"s{seed[-1]} batch loss (20-step mean)")
                cps = native["checkpoints"]
                tune_axis.plot([cp["update"] for cp in cps], [cp["tune"]["mse"] for cp in cps], "o--", color=color, alpha=.8)
                chosen = next(cp for cp in cps if cp["update"] == native["metadata"]["selected_update"])
                tune_axis.scatter(chosen["update"], chosen["tune"]["mse"], s=95, marker="*", color=color, zorder=5)
            ax.set_title(f"{dataset} / {route}"); ax.set_xlabel("Optimizer update")
            ax.set_ylabel("Observed macro minibatch loss"); tune_axis.set_ylabel("Full TUNE MSE (dashed)")
            ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.2)
    save(fig, "04_native_learning_and_selection", "Same frozen E/D: native training and exact global TUNE selection",
         "Stars mark selected checkpoints 0/100/200/400/600. Raw minibatch losses and smoothed display are not full-FIT epoch errors.")

    cost_methods = ["RAW_BOLT", "PCA_BOLT", "FORECA_BOLT", "PCA_NATIVE", "FORECA_NATIVE"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for i, dataset in enumerate(DATASETS):
        for j, batch in enumerate((1, 8)):
            ax = axes[i, j]
            rows = {row["method"]: row for row in tables["COST"] if row["dataset"] == dataset and row["batch"] == batch}
            values = [rows[method]["median_ms_per_window"] for method in cost_methods]
            ax.bar(range(5), values, color=["#444444", "#0072B2", "#D55E00", "#56B4E9", "#E69F00"])
            for x, method in enumerate(cost_methods):
                sample_values = [row["ms_per_window"] for row in tables["COST_SAMPLES"] if row["dataset"] == dataset and row["batch"] == batch and row["method"] == method]
                ax.scatter(x + np.linspace(-.12, .12, len(sample_values)), sample_values, s=14, c="black", alpha=.55)
                ax.text(x, values[x]*1.25, f"{values[x]:.3f}", ha="center", fontsize=9)
            ax.set_yscale("log"); ax.set_ylim(min(values)*.55, max(values)*2.1)
            ax.set_xticks(range(5), ["Raw", "PCA/F", "ForeCA/F", "PCA/N", "ForeCA/N"])
            ax.set_ylabel("Whole ms / window (log)"); ax.set_title(f"{dataset}: B{batch}; native seed 92601")
            rss = max(row["resident_rss_max_bytes"] for row in rows.values()) / 2**30
            reserved = max(row["shared_peak_reserved_vram_max_bytes"] for row in rows.values()) / 2**20
            ax.text(.02, .02, f"Shared process RSS <= {rss:.2f} GiB; reserved peak <= {reserved:.0f} MiB", transform=ax.transAxes, fontsize=8)
            ax.grid(axis="y", alpha=.2)
    save(fig, "05_whole_deployment_cost", "Same trial whole deployment latency: measured median and six samples",
         "Warmup 2; first five orders balanced, sixth extra. Shared memory counters are not isolated method allocations; load/save excluded.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for i, dataset in enumerate(DATASETS):
        dev = "val" if dataset == "robin" else "eval"
        for j, route in enumerate(PROJECTIONS):
            ax = axes[i, j]
            oracle = result["datasets"][dataset]["projections"][route]["oracle"][dev]
            values = [oracle[key]["normalized"]["mse"] for key in ("scores", "raw_scores_same_mask", "bolt_scores_same_mask")]
            values.extend(oracle["native_scores_same_mask"][seed]["normalized"]["mse"] for seed in SEEDS)
            labels = ["E(Y) oracle", "Raw", route + "/F", "native s1", "native s2"]
            colors = ["#999999", "#444444", "#0072B2", "#56B4E9", "#009E73"]
            if diagnostic is not None:
                cell = diagnostic["datasets"][dataset]["projections"][route][dev]
                if (cell["complete_target_positions"], cell["total_target_positions"]) != (oracle["complete_target_positions"], oracle["total_target_positions"]):
                    raise ValueError("FIGURE_FIXED_D_ORACLE_MASK_COVERAGE_MISMATCH")
                values.append(cell["fixed_D_bound_scores_same_mask"]["normalized"]["mse"])
                labels.append("D-span optimum"); colors.append("#CC79A7")
            ax.bar(range(len(values)), values, color=colors)
            ax.set_xticks(range(len(values)), labels, rotation=20, ha="right")
            for x, value in enumerate(values): ax.text(x, value + max(values)*.025, f"{value:.4f}", ha="center", fontsize=9)
            ax.set_ylim(0, max(values)*1.2); ax.set_ylabel("Same complete-target mask normalized MSE")
            ax.set_title(f"{dataset}/{route}: coverage {oracle['complete_target_positions']}/{oracle['total_target_positions']}")
            ax.grid(axis="y", alpha=.2)
    save(fig, "06_future_EY_diagnostic", "Actual-future diagnostics on the same complete mask: E(Y) and fixed D",
         "E(Y) is not an error floor. D-span optimum, when present, minimizes normalized MSE only in the fixed-D32 real output span.")
    return figures


def make_text(result, audit, tables, figures, output, result_path, audit_path, diagnostic, prior_attempts):
    report_path = output / "REPORT_KO.md"
    robin_foreca = result["datasets"]["robin"]["projections"]["FORECA"]
    oracle_robin = robin_foreca["oracle"]["val"]["scores"]["normalized"]["mse"]
    native_robin = robin_foreca["native"]["92601"]["scores"]["val"]["normalized"]["mse"]
    lines = ["# 작은 K 투영·고정 Bolt·Native 예측기: 실제 비교", "",
             "[확인] 이 감사에 채택한 producer는 PCA/ForeCA 원리 투영 4개, 고정 복원 D 4개, native 8개의 실제 적합·학습과 20조건·120회 whole 추론을 기록했다. 독립 감사 PASS를 확인한 producer에만 이 보고서를 묶었다. 전체 물리 시도 수나 연구 전체 해결·독립 일반화 검증을 뜻하지 않는다. 앞선 미채택 시도는 각 실행·감사 기록과 구분한다.", "",
             "## 핵심 질문의 답", "",
             "**과거만 사용하는 작은 K 표현이 정확도·비용을 함께 유지했는가?** [확인] 이번 고정 K·현재 C anchor 조건에서는 Raw Bolt보다 작은-K 경로의 TUNE/개발 오차가 컸다. JOINT ridge 실패를 필요조건으로 쓰지 않고 압축 경로를 직접 검사했지만, 이 PCA/ForeCA 묶음에서 정확도 유지 근거를 얻지 못했다. 비용 감소와 정확도 유지 여부는 별개다.", "",
             "**시간 예측성 투영이 PCA보다 좋았는가?** [확인] ForeCA 원리 경로의 FIT spectral entropy와 unit-variance/rank를 확인했다. 아래 원수치에서 ForeCA+Bolt의 개발 MSE는 두 자료 모두 PCA+Bolt보다 컸다. [추정] 낮은 스펙트럼 엔트로피가 H48의 원 C 미래 보존·복원·Bolt 호환성까지 보장하지 않는 경쟁 설명이 남는다. 이를 모든 예측 목적 표현의 실패로 확대하지 않는다.", "",
             "**Bolt를 붙이는 단계가 항상 문제였는가?** [확인] 동일 E/D를 고정한 native로 F만 교체했다. Robin의 PCA에는 Bolt가 두 native보다 개발 MSE가 낮았지만, ForeCA에서는 두 native가 Bolt보다 낮았다. Beijing PCA의 native는 TUNE MSE가 Bolt보다 낮아도 개발 MSE는 높았다. 특정 좌표·시기·head에서의 조건부 비교이며 모든 native보다 Bolt가 우월/불필요하다는 결론은 아니다.", "",
             f"**표현·미래 예측·D 중 무엇이 남았는가?** [확인] 실제 미래 E(Y)는 미래 복원 최적 latent가 아니므로 error floor가 아니다. Robin ForeCA 개발 E(Y) MSE {oracle_robin:.9f}, native seed92601 {native_robin:.9f}를 실제 동일 mask에서 비교했다. E(Y)보다 낮은 실제 예측 오차는 이 진단이 엄밀한 floor라는 주장과 충돌한다. [추정] 현재 E/D와 실제 미래 변화의 불일치 및 forecaster의 조건별 오차가 함께 남는다. 학습 문제 또는 방법의 본질적 한계 하나로 확정할 수 없다.", "",
             "**누출이 있었는가?** [확인] E/D는 FIT 과거로 적합했고 native의 실제 미래 label은 FIT 학습 및 TUNE checkpoint 선택에 사용했다. 배포 입력은 과거와 현재 C anchor뿐이다. 실제 추출된 2-window payload의 target/future-mask 교란 후 고정 모델 출력 불변을 저장했다. 전체 원자료 미래를 교란한 재학습·전처리 재실행이나 모든 가능한 누출 부재의 증명은 아니다. 보호 TEST 접근은 없었다.", "",
             "**대조학습·동적 그래프가 필요한가?** [확인] 이번에는 새 contrastive/graph를 학습하지 않았다. 두 static 선형 투영의 실패로 필요성을 증명하거나 가능성을 기각할 수 없다. 다음 방법은 현재의 정보 보존·복원·예측기 진단을 해결하는 가설로 정해야 한다.", "",
             "## 동일 비교 조건과 실제 원수치", "",
             "Robin C17/K4·Beijing Aotizhongxin C11/K2, L512/H48, FIT256/TUNE64·개발 VAL30/EVAL146를 유지했다. E는 시점별 선형 C→K이고 모든 경로가 current C anchor를 받으므로 strict K-only 압축이 아니다. 같은 전체 C 과거 입력을 읽는 전처리 비용도 포함했다. 개발 구간은 반복 노출되었으며 독립 holdout이 아니다.", "",
             "```mermaid", "flowchart LR", '  X["Past X: 512 x C"] --> E["FIT-only PCA / ForeCA"]',
             '  E --> Z["Past 512 x K"]', '  Z --> F["Frozen Bolt"]', '  Z --> N["Native K temporal heads"]',
             '  F --> Q["Predicted future K minus last K"]', '  N --> Q', '  Q --> D["Fixed D"]',
             '  X --> A["Current C anchor"]', '  D --> Y["Predicted C future"]', '  A --> Y', "```", "",
             "PCA도 whitening하여 두 투영의 FIT 잠재 분산을1로 맞췄다. 같은 중첩 FIT 창256×512 positions의 공분산을 쓰며 이들을 독립 표본으로 세지 않았다. D는 원 관측 FIT 과거만 정답으로 삼은 채널별 masked affine least squares다. 미래 target을 D 적합에 넣지 않았다. 서로 다른 E의 D 클래스·적합 규칙을 맞춘 것이며 D 가중치가 byte 동일하다는 뜻은 아니다.", "",
             "ForeCA는 [원 논문](https://proceedings.mlr.press/v28/goerg13.pdf)의 spectral forecastability 원리를 적용한 고정 투영이다. 이번 Hann·단측257빈·7빈 반사 smoothing·성분별200반복/결정적2초기방향은 사전 고정한 적용 계약이며 공식 R 구현의 정확한 benchmark 재현이라고 부르지 않는다. FIT entropy만으로 방향을 선택했고 TUNE/DEV로 spectral 설정을 탐색하지 않았다.", "",
             "```text", "dataset                split  method / seed                 MSE          MAE          selected CP"]
    for dataset in DATASETS:
        for split in ("fit", "tune", "val" if dataset == "robin" else "eval"):
            for name, row in selected_rows(tables, dataset, split).items():
                lines.append(f"{dataset:<23}{split:<7}{name:<30}{row['mse']:<13.9f}{row['mae']:<13.9f}{str(row['selected_update']):>5}")
    lines.extend(["```", "", "[전체 정규화·채널별 원단위 지표](tables/COMPARISON.csv). 서로 다른 물리 단위의 channel-macro를 하나의 의미 있는 원단위 목표로 해석하지 않고 원단위 지표는 채널별 진단으로 둔다.", ""])
    for name in figures[:3]: lines.extend([f"![{name}](figures/{name}.png)", ""])
    lines.extend(["## 학습과 선택을 분리한 기록", "",
                  "Native는 잠재 K채널 각각의512→48 affine time head이고 입력은 z_past−z_last다. E/D는 동결하고 D 뒤 actual-Y 관측 정규화 macro MSE로 학습했다. AdamW .001·wd0·clip1·batch4·600 update, CP0/100/200/400/600 중 TUNE MSE 최소를 선택하며 exact tie는 가장 이른 CP다. 두 seed는 동일 zero 초기화의 배치 순서를 바꾸며 독립 랜덤 초기화 반복이라고 세지 않았다.", "",
                  "FP32 배포식은 `raw_last + ((predicted_future_K - encoded_history_last_K) @ D)`다. D bias는 과거 재구성에만 쓰고 innovation에서는 상쇄한다. CP0=현재값 유지의 실제 parity를 확인했다. 원 MAIN 연산 순서와 별도 계약이며 그 기존 결과를 덮거나 byte 동일하다고 주장하지 않는다.", "",
                  "[40개 checkpoint 선택](tables/CHECKPOINT_SELECTION.csv) · [모든 실제 minibatch 학습 곡선](tables/TRAINING_CURVE.csv) · [8개 native 학습 시간·용량](tables/TRAINING_COST.csv) · [FIT spectral·rank·분산 검사](tables/SPECTRAL_GEOMETRY.csv). 학습 시간은 latent 캐시·선택·callback을 포함한 각 native fit의 측정 wall이며 optimizer kernel만의 시간이 아니다. 그림의20-step 이동평균은 표시용이며 checkpoint 선택 손실이나 full-FIT 오차가 아니다.", "",
                  f"![학습과 선택](figures/{figures[3]}.png)", "", "## 실제 추론 비용과 불확실성", "",
                  "같은 trial에서 Raw·PCA/F·ForeCA/F·PCA/native·ForeCA/native × 두 자료 × B1/B8의20조건을 측정했다. 각 warmup2·6repeat이며 첫5 circular 순서만 모든 위치에 균형이고6번째는 사전 seeded 추가 순서다. 전체6회를 완전 균형으로 부르지 않는다. 동일 첫 B1/B8 개발 origins를 반복하며 독립 hardware 반복이 아니다. native 비용은 seed92601이다.", "",
                  "타이머는 실제 causal CPU forward-fill·FIT 정규화·E→forecaster→D·필요한 H2D/F/D2H·동일 CPU inverse·원단위 CPU 반환·최종 sync를 포함한다. data/model/state 로드, 참조 추론, warmup, 저장/해시/guard는 밖이다. 단계별 시간 분해를 새로 측정하지 않았다. 공유 상주 F와 모든 state의 RSS/VRAM counter이므로 방법별 단독 메모리 절감으로 해석하지 않는다. 이전 trial의 절대 지연과 직접 순위를 합치지 않는다.", "",
                  "```text", "dataset                batch method          whole ms/window  paired Raw median ratio"])
    for row in tables["COST"]:
        lines.append(f"{row['dataset']:<23} B{row['batch']:<4}{row['method']:<17}{row['median_ms_per_window']:<17.6f}{row['paired_RAW_median_ratio']:.6f}")
    lines.extend(["```", "", "[20조건 비용](tables/COST.csv) · [120 실제 시간 샘플](tables/COST_SAMPLES.csv). 비용의 paired median ratio와 median 시간의 ratio는 서로 다른 요약이며 혼동하지 않는다. 오차 증가1% 이내·MAE 증가1% 이내·B8 median 지연10% 감소는 사전 탐색 기준이고 유의성/독립 확증이 아니다. 정확도 기준을 못 넘은 빠른 경로를 채택 성공으로 쓰지 않는다.", "",
                  "[사전 비용 절충 기준의 8개 실제 판단](tables/COST_ADOPTION.csv)은 TUNE MSE/MAE 각각 Raw의1.01배 이내와 **B8 whole 중앙값 자체의 비율≤.90**을 사용한다. paired ratio의 중앙값을 그 기준에 대입하지 않는다. native seed92602의 별도 비용은 측정하지 않았으므로 seed92601 비용을 다른 seed의 비용 검증으로 바꾸어 말하지 않는다.", "",
                  f"![whole 비용](figures/{figures[4]}.png)", "", "## 실제 미래 E(Y)와 복원 진단의 해석", "",
                  "Oracle는 실제 미래가 모든 C에서 관측된 시점만 사용한다. 방법들도 동일 complete-target mask로 다시 점수화하여 부분 target의 임의 fill에 의한 이득을 피했다. Beijing의 oracle 비교 mask는 원 deployment 관측 mask와 달라 위 full-score 표의 값을 그대로 비교하지 않는다. Oracle는 배포 입력·학습·TUNE/분기 선택·비용·방법 순위에 들어가지 않았다.", "",
                  "```text", "dataset                projection split  complete/total    E(Y) MSE     Raw same-mask MSE"])
    for dataset in DATASETS:
        for route in PROJECTIONS:
            for split, oracle in result["datasets"][dataset]["projections"][route]["oracle"].items():
                lines.append(f"{dataset:<23}{route:<11}{split:<7}{str(oracle['complete_target_positions'])+'/'+str(oracle['total_target_positions']):<18}{oracle['scores']['normalized']['mse']:<13.9f}{oracle['raw_scores_same_mask']['normalized']['mse']:.9f}")
    lines.extend(["```", "", "[동일 mask의 모든 방법·oracle·원단위 진단](tables/ORACLE_MATCHED_MASK.csv). E(Y)는 고정 D를 통한 미래 오차를 최소화하는 latent를 최적화하지 않았다. ForeCA에서는 실제 예측 native가 E(Y)보다 나으므로 이를 lower bound라고 표현하는 것은 실제 반례와 충돌한다. E(Y) 오차는 표현/복원 불일치의 보조 단서다.", "",
                  f"![동일 관측 mask oracle](figures/{figures[5]}.png)", ""])
    if diagnostic is None:
        lines.extend(["고정 D rowspace의 최적 future-latent 진단은 이 보고서 입력에 제공되지 않아 **미포함**으로 표시한다. 입력 부재만으로 실제 실행 여부를 단정하거나 존재하지 않는 floor 값을 가정하지 않았다.", ""])
    else:
        lines.extend(["### 고정 D의 도달 가능한 미래 부분공간", "",
                      "[확인] [추가 진단 원문](ADDITIONAL_DIAGNOSTIC_METADATA.json)은 [추가 독립 감사](ADDITIONAL_DIAGNOSTIC_AUDIT.json) PASS와 diagnostic/producer/projection-audit의 exact SHA를 각각 확인한 별도 입력이다. 실제 미래를 사용했지만 E/D/F·원 checkpoint 선택·분기·예측을 바꾸지 않았다. 배포 방법이나 새로운 미래 예측 실험으로 세지 않는다.", "",
                      "`delta_star = (Y64 - current_C64) @ pinv(D32_as64)`로 구한 실제 미래 latent innovation은 all-C 관측 위치에서 **선택된 고정 D32의 실수 부분공간 안 normalized MSE**를 최소화한다. 이는 고정 D에만 적용되는 기하 진단이고 모든 표현·비선형/가변 D의 최적 하한이나 과거에서 예측 가능한 latent라는 증명이 아니다. Raw는 이 부분공간을 쓰지 않으므로 Raw에 하한 부등식을 적용하지 않는다. FP32 실제 예측과 실수 산술의 차이는 측정된 MSE allowance로 구분한다.", "",
                      "**MAE와 서로 다른 물리 단위의 MSE는 이 normalized-MSE 최소해의 점수일 뿐 각각의 최소가 아니다.** 세 지표의 모두 최적 floor라고 부르지 않는다. [동일 complete-mask 원수치](tables/FIXED_D_MATCHED_MASK.csv) · [rank/특이값/정규방정식/기하](tables/FIXED_D_GEOMETRY.csv).", "",
                      "```text", "dataset                projection split  norm-MSE minimum  Raw same mask   E(Y) same mask"])
        for dataset, data in diagnostic["datasets"].items():
            for route, splits in data["projections"].items():
                for split, cell in splits.items():
                    floor = cell["fixed_D_bound_scores_same_mask"]["normalized"]["mse"]
                    raw = cell["comparisons"]["RAW_BOLT"]["saved_prediction_scores_same_mask"]["normalized"]["mse"]
                    ey = cell["comparisons"]["E_Y"]["saved_prediction_scores_same_mask"]["normalized"]["mse"]
                    lines.append(f"{dataset:<23}{route:<11}{split:<7}{floor:<18.9f}{raw:<16.9f}{ey:.9f}")
        lines.extend(["```", "",
                      "[판단] 같은 mask에서 고정 D의 normalized-MSE 최소가 Raw보다 크면, 그 고정 D를 유지한 채 F/head만 바꾸어 Raw보다 작게 만드는 목표에는 표현 가능한 미래 변화의 제한이 남는다. 최소가 Raw보다 작아도 과거에서 그 latent를 실제로 예측할 수 있다는 뜻은 아니며 예측 단계의 검증이 필요하다. E(Y)와 D-span 최적점의 차이는 기존 E/D의 미래 재구성 좌표 차이도 보여 준다.", ""])
    totals = audit["totals"]
    receipt = result.get("resource_terminal", {})
    lines.extend([f"[확인] 채택 producer의 전체 실행 wall은 {result['wall_seconds']:.6f}초다. terminal receipt의 process CPU는 {receipt.get('process_cpu_seconds')}초, process wall은 {receipt.get('wall_seconds')}초로 별도 기록했다. GPU process 표시는 해당 process의 분류이며 GPU kernel 시간이나 native GPU 학습 시간을 뜻하지 않는다. native는 CPU에서 학습했고 투영 적합·F추론·저장·검산을 각기 완전히 분해한 시간 측정은 이 producer에 없다.", ""])
    physical_native_fits = result["native_training_fits"]
    for attempt in prior_attempts:
        entry = attempt["metadata"]
        if "native_training_fits" in entry:
            physical_native_fits += entry["native_training_fits"]
        lines.extend([f"[보존한 선행 시도]({link(report_path, attempt['record']['path'])}): 상태 `{entry.get('status', 'UNSPECIFIED')}`, source JSON SHA `{attempt['record']['sha256']}`. 실패 정보 `{json.dumps(entry.get('failure'), ensure_ascii=False)}`. 채택 producer의 PASS와 합치지 않는다.", ""])
    if prior_attempts:
        lines.extend([f"[확인·실행 기록] 제공·존재 확인한 producer들의 물리 native fit 합계는 {physical_native_fits}개이며 이번 유효 producer의8개와 구분한다. 두 run이 같은8조건을 반복했다면 이를16개의 다른 방법·독립 초기화로 세지 않는다. 첫 CP 점수 산술/배치 경로의 불일치와 canonical 저장 수선은 숫자 재현의 문제이며 모든 이전 표현 실패의 원인으로 확대하지 않는다.", ""])
    lines.extend(["## 검산·계보·실패 범위", "",
                  f"[확인] 독립 감사 PASS: projection4·native8·CP40·비용20·시간샘플120, 저장 예측 배열 {totals.get('saved_prediction_arrays_checked')}개·지표 필드 {totals.get('score_field_checks')}개를 확인했다. 감사는 저장 native state의 NumPy 예측 산술을 재현했고 새 native 학습을 다시 수행하지 않았다. F 실제 forward/normalizer 동작의 독립 재실행이나 타이머·hardware peak 재측정으로 바꾸어 쓰지 않는다.", "",
                  f"[실제 producer]({link(report_path, result_path)}) · [독립 감사]({link(report_path, audit_path)}) · [보고서 원천/SHA](PROVENANCE.json) · [그림·CSV·MD 파일 SHA](MANIFEST.json). 이 생성기는 terminal JSON과 소스 metadata만 읽으며 NPZ·원자료·checkpoint·모델 가중치·TEST를 열지 않았다. source snapshots의 SHA와 현재 source는 실험 기록과 감사의 명시 범위로 구분한다.", "",
                  "[확인] 새 학습·추론은 이전 SELF/JOINT 중단 gate를 이 고정 묶음에 한해 수정한 사용자 승인 뒤 수행했다. 이후 인간의 상한 해제 지시를 별도 기록하고 기존 승인·시계·결과를 수정하지 않았다. 실제 신경망 backbone 학습·설치/다운로드·기존 job 종료·자동 commit/push는 이 묶음에 포함하지 않는다. 실패/미실행 결과는 완료로 바꾸지 않는다.", "",
                  "## 남은 질문과 다음 판단", "",
                  "[판단] 동일 D에서 미래 변화가 도달 가능한 부분공간에 얼마나 남는지와, 그 latent를 과거만으로 예측하는 난도를 나누어야 한다. E(Y)만으로 이 둘을 분리했다고 선언하지 않는다. 고정 D의 적합한 미래 rowspace 진단, 예측 목적 D/표현의 기존 동일 조건 근거, 현재값·잔차 우회의 동등 비용을 검토한 뒤 빠진 최소 대조를 정한다. 새로운 loss/head/graph를 이 결과만으로 계속 늘리지 않는다.", "",
                  "[판단] 현재 유지되는 기여는 SELF/JOINT의 필요조건 오류를 바로잡고, 실제 작은-K PCA/ForeCA와 같은 E/D의 F/native를 정확도·관측 mask·전체 비용에 연결한 실패 경계다. 아직 주장할 수 없는 것은 작은 K의 원미래 정확도 유지, 예측 목적 표현의 보편적 성공/실패, contrastive/dynamic graph의 필요성, 독립 일반화다. 최상위 목적은 작은 K와 고정 Bolt를 쓸 근거를 찾거나 경계를 판별하는 것으로 유지한다.", ""])
    if diagnostic is not None:
        early = ["**새로 분리된 고정 D의 제한은 무엇인가?** [확인] 다음 수치는 같은 complete-target mask에서 실제 미래를 사용한 기하 진단이다. normalized-MSE 최소가 Raw보다 큰 고정 D는 forecaster만 바꾸는 것으로 그 Raw보다 작은 오차를 얻을 수 없는 제한이 남는다. 선택한 D32의 실수 부분공간/현재 C anchor에 한정하며 다른 D나 학습 목표 전체의 불가능성은 아니다.", ""]
        for dataset, data in diagnostic["datasets"].items():
            dev = data["development_split"]
            for route, splits in data["projections"].items():
                cell = splits[dev]
                floor = cell["fixed_D_bound_scores_same_mask"]["normalized"]["mse"]
                raw = cell["comparisons"]["RAW_BOLT"]["saved_prediction_scores_same_mask"]["normalized"]["mse"]
                interpretation = "선택된 고정 D의 최소가 Raw보다 큼" if floor > raw else "선택된 D의 도달 가능 최소는 Raw보다 작지만 실제 과거 예측 가능성은 미확정"
                early.append(f"- {dataset}/{route}: 고정-D 최소 {floor:.9f}, 같은 mask Raw {raw:.9f}; {interpretation}.")
        early.append("")
        index = lines.index("## 동일 비교 조건과 실제 원수치")
        lines[index:index] = early
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build(result_path, audit_path, output_path, diagnostic_path=None, diagnostic_audit_path=None):
    result, audit = load_verified(result_path, audit_path)
    diagnostic = load_diagnostic(result_path, audit_path, diagnostic_path, diagnostic_audit_path)
    prior_attempts = []
    for path in (HERE / "runs/actual_projection_v1/RESULT.json", HERE / "audits/PROJECTION_RESULTS_AUDIT_v1.json"):
        if path.exists() and path.resolve() not in {Path(result_path).resolve(), Path(audit_path).resolve()}:
            prior_attempts.append({"record": record(path), "metadata": read_json(path)})
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError("REPORT_REQUIRES_NEW_OUTPUT_DIRECTORY")
    tables = collect_tables(result, audit)
    if diagnostic is not None: tables.update(diagnostic_tables(diagnostic))
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir(); (output / "figures").mkdir()
    for name, rows in tables.items(): write_csv(output / "tables" / (name + ".csv"), rows)
    if diagnostic is not None:
        (output / "ADDITIONAL_DIAGNOSTIC_METADATA.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "ADDITIONAL_DIAGNOSTIC_AUDIT.json").write_text(json.dumps(read_json(diagnostic_audit_path), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figures = make_figures(result, audit, tables, output, diagnostic)
    make_text(result, audit, tables, figures, output, result_path, audit_path, diagnostic, prior_attempts)
    provenance = {"producer": record(result_path), "independent_audit": record(audit_path),
                  "generator": record(__file__), "additional_diagnostic": None if diagnostic_path is None else record(diagnostic_path),
                  "additional_diagnostic_audit": None if diagnostic_audit_path is None else record(diagnostic_audit_path),
                  "preserved_prior_attempts": [entry["record"] for entry in prior_attempts],
                  "scientific_source_snapshots": result["source_snapshots"],
                  "authorization": result["authorization"], "continuation_manifest": result["continuation_manifest"],
                  "metadata_only_report_generation": True, "NPZ_model_checkpoint_TEST_opened": False}
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = [{"path": path.relative_to(output).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {"status": "GENERATED_FROM_PASS_AUDIT", "producer_sha256": sha(result_path),
                "audit_sha256": sha(audit_path), "csv_rows": {name: len(rows) for name, rows in tables.items()},
                "PNG_count": len(figures), "SVG_count": len(figures), "files": files,
                "new_scientific_execution": False, "protected_TEST_opened": False}
    (output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True); parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--diagnostic"); parser.add_argument("--diagnostic-audit")
    args = parser.parse_args()
    manifest = build(args.result, args.audit, args.output, args.diagnostic, args.diagnostic_audit)
    print(json.dumps({"status": manifest["status"], "csv_rows": manifest["csv_rows"],
                      "PNG": manifest["PNG_count"], "SVG": manifest["SVG_count"]}, ensure_ascii=False))
