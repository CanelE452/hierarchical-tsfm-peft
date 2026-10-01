"""Assemble v15 publication-facing values, CSVs, and static figures.

The report uses only verified aggregate JSON outputs. It does not load raw
predictions, recompute scores, fit models, or choose new methods.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

from runtime_v15 import DATASETS, HERE, Job, artifact, digest, read_json, require_storage, save_json


ROLES = ("raw_free", "pca_free", "raw_tied", "raw_interp", "pca_tied",
         "direct", "pca_level", "a", "f0", "full_mse", "native")
PERIODS = ("test_a", "test_b", "combined")
METRICS = ("mse", "mae")
ROLE_LABELS = {
    "raw_free": "RAW_FREE",
    "pca_free": "PCA_FREE",
    "raw_tied": "RAW_TIED",
    "raw_interp": "RAW_INTERP",
    "pca_tied": "PCA_TIED",
    "direct": "DIRECT",
    "pca_level": "PCA_LEVEL",
    "a": "A",
    "f0": "F0",
    "full_mse": "FULL_MSE",
    "native": "NATIVE",
}
FIGURES = {
    "accuracy_by_period": HERE / "figure_accuracy_by_period",
    "accuracy_cost": HERE / "figure_accuracy_cost",
}


def read_complete_evaluation():
    value = read_json(HERE / "evaluation01.json")
    if value.get("status") != "complete" or set(value.get("datasets", {})) != set(DATASETS):
        raise ValueError("evaluation01.json must be complete for all four v15 datasets")
    artifact(HERE / "evaluation01.json")
    return value


def read_cost_summary():
    path = HERE / "cost_summary.json"
    if not path.exists():
        return {"status": "missing", "gpu": None, "cpu": None,
                "reason": "cost_summary.json not present at report time"}
    value = read_json(path)
    value.setdefault("status", "partial")
    value.setdefault("gpu", None)
    value.setdefault("cpu", None)
    artifact(path)
    available_path = HERE / "cost_available_summary.json"
    if value["status"] != "complete" and available_path.exists():
        available = read_json(available_path)
        artifact(path, available["original_cost_summary"]["sha256"])
        value["gpu"] = available["gpu"]
    return value


def write_csv(path, rows, fieldnames):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refuse to overwrite report artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    return value


def period_metric(unit, family, period, metric):
    row = unit["role_summary"].get(family)
    if not row or row.get("status") != "available":
        return None
    return row.get("periods", {}).get(period, {}).get(metric)


def accuracy_rows(evaluation):
    rows = []
    for dataset, unit in evaluation["datasets"].items():
        full = unit["role_summary"].get("full_mse", {})
        for family in ROLES:
            role = unit["role_summary"].get(family)
            status = role.get("status") if role else "missing"
            reason = role.get("reason") if role else "role missing from evaluation"
            ids = ";".join(str(item) for item in role.get("ids", [])) if role else ""
            for period in PERIODS:
                full_period = full.get("periods", {}).get(period, {})
                period_row = role.get("periods", {}).get(period, {}) if role else {}
                rows.append({
                    "dataset": dataset,
                    "family": family,
                    "label": ROLE_LABELS.get(family, family),
                    "period": period,
                    "status": status,
                    "ids": ids,
                    "mse": period_row.get("mse"),
                    "mae": period_row.get("mae"),
                    "signed_mean_error": period_row.get("signed_mean_error"),
                    "relative_mse_to_full_mse_percent":
                        100.0 * period_row["mse"] / full_period["mse"]
                        if period_row.get("mse") is not None and full_period.get("mse") else None,
                    "relative_mae_to_full_mse_percent":
                        100.0 * period_row["mae"] / full_period["mae"]
                        if period_row.get("mae") is not None and full_period.get("mae") else None,
                    "unavailable_reason": None if status == "available" else reason,
                })
    return [clean(row) for row in rows]


def paired_difference_rows(evaluation):
    rows = []
    for dataset, unit in evaluation["datasets"].items():
        for key, item in unit.get("comparisons", {}).items():
            if item.get("status") != "available":
                rows.append({
                    "dataset": dataset,
                    "comparison": key,
                    "candidate": item.get("candidate"),
                    "reference": item.get("reference"),
                    "status": item.get("status"),
                    "period": None,
                    "metric": None,
                    "candidate_value": None,
                    "reference_value": None,
                    "absolute_difference": None,
                    "relative_change_percent": None,
                    "conditional_block95_absolute_low": None,
                    "conditional_block95_absolute_high": None,
                    "conditional_block95_relative_percent_low": None,
                    "conditional_block95_relative_percent_high": None,
                    "reason": item.get("reason"),
                })
                continue
            for period, values in item.get("periods", {}).items():
                for metric in METRICS:
                    metric_row = values[metric]
                    absolute = metric_row.get("conditional_block95_absolute")
                    relative = metric_row.get("conditional_block95_relative_percent")
                    rows.append({
                        "dataset": dataset,
                        "comparison": key,
                        "candidate": item.get("candidate"),
                        "reference": item.get("reference"),
                        "status": item.get("status"),
                        "period": period,
                        "metric": metric,
                        "candidate_value": metric_row.get("candidate"),
                        "reference_value": metric_row.get("reference"),
                        "absolute_difference": metric_row.get("absolute_difference"),
                        "relative_change_percent": metric_row.get("relative_change_percent"),
                        "conditional_block95_absolute_low": absolute[0] if absolute else None,
                        "conditional_block95_absolute_high": absolute[1] if absolute else None,
                        "conditional_block95_relative_percent_low": relative[0] if relative else None,
                        "conditional_block95_relative_percent_high": relative[1] if relative else None,
                        "reason": None,
                    })
    return [clean(row) for row in rows]


def cost_rows(cost):
    rows = []
    for device_key, summary in (("gpu", cost.get("gpu")), ("cpu", cost.get("cpu"))):
        if not summary:
            continue
        block_ranges = cost_block_ranges(summary)
        for row in summary.get("groups", []):
            key = (row.get("dataset"), row.get("family"), row.get("batch"),
                   row.get("execution"), row.get("chunk_rows"))
            ranges = block_ranges.get(key, {})
            rows.append({
                "device": summary.get("device", device_key),
                "summary_status": summary.get("status"),
                "dataset": row.get("dataset"),
                "family": row.get("family"),
                "batch": row.get("batch"),
                "execution": row.get("execution"),
                "chunk_rows": row.get("chunk_rows"),
                "ids": ";".join(str(item) for item in row.get("ids", [])),
                "mean_seed_median_ms": row.get("mean_seed_median_ms"),
                "mean_seed_median_origins_s": row.get("mean_seed_median_origins_s"),
                "mean_seed_median_peak_allocated_bytes": row.get("mean_seed_median_peak_allocated_bytes"),
                "mean_seed_median_peak_reserved_bytes": row.get("mean_seed_median_peak_reserved_bytes"),
                "block_origins_s_low": ranges.get("origins_s_low"),
                "block_origins_s_high": ranges.get("origins_s_high"),
                "block_peak_allocated_low_bytes": ranges.get("peak_allocated_low_bytes"),
                "block_peak_allocated_high_bytes": ranges.get("peak_allocated_high_bytes"),
                "allocated_range_low_bytes": row.get("allocated_range_bytes", [None, None])[0]
                    if row.get("allocated_range_bytes") else None,
                "allocated_range_high_bytes": row.get("allocated_range_bytes", [None, None])[1]
                    if row.get("allocated_range_bytes") else None,
                "reserved_range_low_bytes": row.get("reserved_range_bytes", [None, None])[0]
                    if row.get("reserved_range_bytes") else None,
                "reserved_range_high_bytes": row.get("reserved_range_bytes", [None, None])[1]
                    if row.get("reserved_range_bytes") else None,
            })
    return [clean(row) for row in rows]


def cost_block_ranges(summary):
    ranges = {}
    for row in summary.get("instances", []):
        key = (row.get("dataset"), row.get("family"), row.get("batch"),
               row.get("execution"), row.get("chunk_rows"))
        item = ranges.setdefault(key, {"origins_s": [], "allocated": []})
        for ms in row.get("block_ms", []) or []:
            if ms and ms > 0:
                item["origins_s"].append(1000.0 / ms)
        for value in row.get("peak_allocated_bytes", []) or []:
            if value is not None:
                item["allocated"].append(value)
    result = {}
    for key, value in ranges.items():
        result[key] = {
            "origins_s_low": min(value["origins_s"]) if value["origins_s"] else None,
            "origins_s_high": max(value["origins_s"]) if value["origins_s"] else None,
            "peak_allocated_low_bytes": min(value["allocated"]) if value["allocated"] else None,
            "peak_allocated_high_bytes": max(value["allocated"]) if value["allocated"] else None,
        }
    return result


def save_figure(fig, stem):
    outputs = {}
    for suffix in ("png", "svg"):
        path = Path(str(stem) + "." + suffix)
        if path.exists():
            raise FileExistsError("Refuse to overwrite report figure: " + str(path))
        if suffix == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs[suffix] = artifact(path)
    return outputs


def cost_complete(cost):
    return bool(cost.get("status") == "complete" and cost.get("gpu") and cost.get("cpu")
                and cost["gpu"].get("status") == "complete" and cost["cpu"].get("status") == "complete")


def draw_accuracy_figure(evaluation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(12, 13), constrained_layout=True)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#f2f2f2")
    image = None
    for row_index, dataset in enumerate(DATASETS):
        unit = evaluation["datasets"][dataset]
        for col_index, metric in enumerate(METRICS):
            values = np.full((len(ROLES), len(PERIODS)), np.nan, dtype=np.float64)
            for i, family in enumerate(ROLES):
                for j, period in enumerate(PERIODS):
                    numerator = period_metric(unit, family, period, metric)
                    denominator = period_metric(unit, "full_mse", period, metric)
                    if numerator is not None and denominator:
                        values[i, j] = 100.0 * numerator / denominator
            ax = axes[row_index, col_index]
            image = ax.imshow(values, cmap=cmap, aspect="auto",
                              norm=matplotlib.colors.TwoSlopeNorm(vmin=90, vcenter=100, vmax=260))
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    label = "NA" if not np.isfinite(values[i, j]) else f"{values[i, j]:.1f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8,
                            color="black" if not np.isfinite(values[i, j]) or values[i, j] < 180 else "white")
            ax.set_xticks(range(len(PERIODS)), PERIODS, rotation=25, ha="right")
            ax.set_yticks(range(len(ROLES)), [ROLE_LABELS[item] for item in ROLES] if col_index == 0 else [])
            ax.set_title(f"{dataset} {metric.upper()} (% of FULL_MSE)", fontsize=10)
            ax.tick_params(axis="both", labelsize=7)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6,
                     ticks=[90,100,125,150,200,250], label="% of FULL_MSE; lower is better")
    fig.suptitle("V15 exposed development accuracy by period; no cross-dataset raw mean", fontsize=12)
    outputs = save_figure(fig, FIGURES["accuracy_by_period"])
    plt.close(fig)
    return outputs


def draw_cost_figure(evaluation, cost_rows_value, cost_status):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(13, 14), constrained_layout=True, sharey="row")
    markers = {"full": "o", "chunk_k": "s", "chunk_4k": "^", "direct": "D"}
    colors = {"raw_free": "#1f77b4", "pca_free": "#ff7f0e", "direct": "#2ca02c",
              "a": "#d62728", "f0": "#9467bd", "full_mse": "#8c564b", "native": "#e377c2"}
    gpu_b4 = [row for row in cost_rows_value
              if str(row.get("device")).lower() in ("cuda", "gpu") and row.get("batch") == 4]
    for index, dataset in enumerate(DATASETS):
        ax_speed, ax_memory = axes[index, 0], axes[index, 1]
        full = period_metric(evaluation["datasets"][dataset], "full_mse", "combined", "mse")
        rows = [row for row in gpu_b4 if row.get("dataset") == dataset]
        if not rows or full is None:
            for ax in (ax_speed, ax_memory):
                ax.text(0.5, 0.5, f"{dataset}\nCost incomplete", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_axis_off()
            continue
        for row in rows:
            family = row["family"]
            mse = period_metric(evaluation["datasets"][dataset], family, "combined", "mse")
            throughput = row.get("mean_seed_median_origins_s")
            allocated = row.get("mean_seed_median_peak_allocated_bytes")
            if mse is None or throughput is None:
                continue
            relative = 100.0 * mse / full
            execution = row.get("execution")
            color = colors.get(family, "#7f7f7f")
            marker = markers.get(execution, "o")
            label = f"{ROLE_LABELS.get(family, family)}-{execution}"
            low, high = row.get("block_origins_s_low"), row.get("block_origins_s_high")
            xerr = None
            if low is not None and high is not None:
                xerr = [[max(0.0, throughput - low)], [max(0.0, high - throughput)]]
            ax_speed.errorbar(throughput, relative, xerr=xerr, fmt=marker, color=color,
                              ecolor=color, capsize=2, markersize=5, alpha=0.85,
                              markeredgecolor="black", markeredgewidth=0.4)
            if allocated is not None:
                allocated_mib = allocated / (1024 ** 2)
                low_mem = row.get("block_peak_allocated_low_bytes")
                high_mem = row.get("block_peak_allocated_high_bytes")
                xerr_mem = None
                if low_mem is not None and high_mem is not None:
                    low_mib, high_mib = low_mem / (1024 ** 2), high_mem / (1024 ** 2)
                    xerr_mem = [[max(0.0, allocated_mib - low_mib)], [max(0.0, high_mib - allocated_mib)]]
                ax_memory.errorbar(allocated_mib, relative, xerr=xerr_mem, fmt=marker, color=color,
                                   ecolor=color, capsize=2, markersize=5, alpha=0.85,
                                   markeredgecolor="black", markeredgewidth=0.4)
        for ax in (ax_speed, ax_memory):
            ax.axhline(100.0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_ylabel("Combined MSE (% of FULL_MSE)")
            ax.grid(True, alpha=0.25)
        ax_speed.set_xscale("log")
        ax_speed.set_xlabel("B4 throughput (origins/s, log scale)")
        ax_memory.set_xlabel("B4 peak allocated memory (MiB)")
        ax_speed.set_title(f"{dataset}: throughput")
        ax_memory.set_title(f"{dataset}: allocated memory")
    handles = []
    for execution, marker in markers.items():
        handles.append(plt.Line2D([], [], color="black", marker=marker, linestyle="None", label=execution))
    axes[0, 1].legend(handles=handles, loc="best", fontsize=7, title="Execution")
    family_handles = [
        plt.Line2D([], [], color=color, marker="o", linestyle="None", label=ROLE_LABELS.get(family, family))
        for family, color in colors.items()
    ]
    axes[0, 0].legend(handles=family_handles, loc="best", fontsize=6, title="Family color")
    note = f"Cost status: {cost_status}; no frontier claim. " if cost_status != "complete" else ""
    fig.suptitle("Same-session GPU B4 accuracy/cost\n" + note +
                 "Bars span seeds and blocks; not confidence intervals.", fontsize=11)
    outputs = save_figure(fig, FIGURES["accuracy_cost"])
    plt.close(fig)
    return outputs


def build_report(job):
    evaluation = read_complete_evaluation()
    cost = read_cost_summary()
    require_storage(20 * 1024 * 1024)
    accuracy = accuracy_rows(evaluation)
    paired = paired_difference_rows(evaluation)
    costs = cost_rows(cost)
    write_csv(HERE / "comparison.csv", accuracy, [
        "dataset", "family", "label", "period", "status", "ids", "mse", "mae",
        "signed_mean_error", "relative_mse_to_full_mse_percent",
        "relative_mae_to_full_mse_percent", "unavailable_reason"])
    write_csv(HERE / "paired_differences.csv", paired, [
        "dataset", "comparison", "candidate", "reference", "status", "period", "metric",
        "candidate_value", "reference_value", "absolute_difference", "relative_change_percent",
        "conditional_block95_absolute_low", "conditional_block95_absolute_high",
        "conditional_block95_relative_percent_low", "conditional_block95_relative_percent_high",
        "reason"])
    write_csv(HERE / "cost_comparison.csv", costs, [
        "device", "summary_status", "dataset", "family", "batch", "execution", "chunk_rows",
        "ids", "mean_seed_median_ms", "mean_seed_median_origins_s",
        "mean_seed_median_peak_allocated_bytes", "mean_seed_median_peak_reserved_bytes",
        "block_origins_s_low", "block_origins_s_high",
        "block_peak_allocated_low_bytes", "block_peak_allocated_high_bytes",
        "allocated_range_low_bytes", "allocated_range_high_bytes",
        "reserved_range_low_bytes", "reserved_range_high_bytes"])
    job.heartbeat("CSV report tables written")
    figures = {
        "accuracy_by_period": draw_accuracy_figure(evaluation),
        "accuracy_cost": draw_cost_figure(evaluation, costs, "complete" if cost_complete(cost) else cost.get("status")),
    }
    complete_cost = cost_complete(cost)
    values = {
        "schema": "v15_report_values_1",
        "status": "complete" if complete_cost else "partial_cost",
        "scope": "Four exposed development datasets; aggregate report only; no new scores or model runs",
        "evaluation": artifact(HERE / "evaluation01.json"),
        "cost_summary": artifact(HERE / "cost_summary.json") if (HERE / "cost_summary.json").exists() else None,
        "cost_available_summary": artifact(HERE / "cost_available_summary.json") if (HERE / "cost_available_summary.json").exists() else None,
        "selection_seal": artifact(HERE / "selection_seal.json") if (HERE / "selection_seal.json").exists() else None,
        "source_files": {name: digest(HERE / name) for name in
                         ("report_v15.py", "evaluate_v15.py", "cost_v15.py", "runtime_v15.py",
                          "PLAN.md", "protocol.json") if (HERE / name).exists()},
        "tables": {
            "comparison": artifact(HERE / "comparison.csv"),
            "paired_differences": artifact(HERE / "paired_differences.csv"),
            "cost_comparison": artifact(HERE / "cost_comparison.csv"),
        },
        "figures": figures,
        "accuracy_rows": len(accuracy),
        "paired_difference_rows": len(paired),
        "cost_rows": len(costs),
        "roles": list(ROLES),
        "periods": list(PERIODS),
        "cost_status": {
            "top_level": cost.get("status"),
            "gpu": cost.get("gpu", {}).get("status") if cost.get("gpu") else None,
            "cpu": cost.get("cpu", {}).get("status") if cost.get("cpu") else None,
            "complete_for_report": complete_cost,
        },
        "frontier_claim": {
            "made": False,
            "reason": ("cost incomplete; avoid accuracy/cost frontier claims" if not complete_cost
                       else "report lists all rows and draws the scatter only; no frontier is selected here"),
        },
        "aggregation_notes": [
            "Accuracy values are verified mean-seed role summaries from evaluation01.json.",
            "Relative heatmaps use per-dataset/per-period FULL_MSE denominators, not cross-dataset raw means.",
            "Cost values are same-session medians summarized by cost_v15; plotted ranges span seeds and blocks, not statistical confidence intervals.",
            "Unavailable RAW_INTERP is retained explicitly when the selector interpolation is rank deficient.",
        ],
    }
    save_json(HERE / "report_values.json", values)
    job.heartbeat("report_values.json and figures complete")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--reserve-s", required=True, type=float)
    args = parser.parse_args()
    with Job("cpu_analysis", args.job, reserve_s=args.reserve_s,
             metadata={"purpose": "v15 aggregate report values and figures",
                       "fit_count": 0, "model_runs": 0, "new_scores": 0}) as job:
        build_report(job)


if __name__ == "__main__":
    main()
