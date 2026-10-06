"""Render selected stored CSV summaries; do not run models or resample data."""

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "research" / "level_chronos2_controls_v1"
if not SOURCE.is_dir():
    SOURCE = OUT.parent / "public" / "source_snapshots" / "research" / "level_chronos2_controls_v1"
DATASETS = ("robin", "peacock_education", "jena")
DATASET_NAMES = {"robin": "Robin", "peacock_education": "Peacock Education", "jena": "Jena"}
METHODS = ("B", "LEVEL", "DIRECT_NLINEAR", "F0", "MSE_LORA", "C2_MV", "C2_SMALL_MV")
METHOD_NAMES = {
    "B": "b",
    "LEVEL": "LEVEL (original)",
    "DIRECT_NLINEAR": "Direct NLinear",
    "F0": "F0",
    "MSE_LORA": "MSE LoRA",
    "C2_MV": "C2 MV",
    "C2_SMALL_MV": "Small MV",
}
LEVEL_COLOR = "#21618c"
SMALL_COLOR = "#d47a23"
COLORS = {
    "B": "#9da4aa",
    "LEVEL": LEVEL_COLOR,
    "DIRECT_NLINEAR": "#91a1ac",
    "F0": "#77a299",
    "MSE_LORA": "#577d67",
    "C2_MV": "#819cbd",
    "C2_SMALL_MV": SMALL_COLOR,
}
DPI = 180


def read_csv(name):
    path = SOURCE / name
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader, 1):
            row["_data_row_index"] = index
            row["_csv_line"] = reader.line_num
            rows.append(row)
    return rows, {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "data_rows": len(rows),
    }


def select_one(rows, **filters):
    selected = [row for row in rows if all(row.get(key) == value for key, value in filters.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one stored row for {filters}, found {len(selected)}")
    return selected[0]


def row_reference(row, file_name, identity_fields):
    return {
        "input_path": (SOURCE / file_name).relative_to(ROOT).as_posix(),
        "data_row_index_1_based": row["_data_row_index"],
        "csv_line_1_based": row["_csv_line"],
        "row_id": f"{file_name}#L{row['_csv_line']}",
        "identity": {key: row[key] for key in identity_fields},
    }


def tidy_axes(ax, grid_axis="x"):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#a5adb3")
    ax.tick_params(axis="both", length=0, pad=5)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#e4e8eb", linewidth=0.8)


def export(fig, stem):
    paths = []
    for suffix in ("png", "svg"):
        path = OUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=DPI, facecolor="white")
        paths.append(path.relative_to(ROOT).as_posix())
    plt.close(fig)
    return paths


accuracy_rows, accuracy_input = read_csv("accuracy_comparison.csv")
paired_rows, paired_input = read_csv("paired_comparisons.csv")
cost_rows, cost_input = read_csv("cost_comparison.csv")
summary_aggregation = "Mean individual seed losses, not averaged predictions; zero-shot counted once"
accuracy = {}
paired = {}
cost = {}

for dataset in DATASETS:
    accuracy[dataset] = {
        method: select_one(
            accuracy_rows, dataset=dataset, method=method,
            aggregation=summary_aggregation, period="combined",
        )
        for method in METHODS
    }
    paired[dataset] = select_one(
        paired_rows, dataset=dataset, left="LEVEL", right="C2_SMALL_MV",
        period="combined", metric="mse",
    )
    cost[dataset] = {
        method: select_one(
            cost_rows, dataset=dataset, method=method, aggregation="summary",
            device="cuda", batch="4", execution="full" if method == "LEVEL" else "full_group",
        )
        for method in ("LEVEL", "C2_SMALL_MV")
    }
    pair = paired[dataset]
    assert pair["fixed_seeds_conditional"] == "True"
    assert pair["not_equivalence_or_noninferiority"] == "True"
    assert math.isclose(float(pair["denominator"]), float(accuracy[dataset]["C2_SMALL_MV"]["mse"]), rel_tol=1e-12)
    assert math.isclose(float(pair["left_value"]), float(accuracy[dataset]["LEVEL"]["mse"]), rel_tol=1e-12)
    assert math.isclose(float(pair["right_value"]), float(accuracy[dataset]["C2_SMALL_MV"]["mse"]), rel_tol=1e-12)
    assert math.isclose(
        float(pair["relative_percent"]),
        100 * (float(pair["left_value"]) - float(pair["right_value"])) / float(pair["denominator"]),
        rel_tol=1e-12,
    )
    for method in ("LEVEL", "C2_SMALL_MV"):
        selected_cost = cost[dataset][method]
        assert math.isclose(float(selected_cost["combined_mse"]), float(accuracy[dataset][method]["mse"]), rel_tol=1e-12)
        assert json.loads(selected_cost["ids"]) == accuracy[dataset][method]["id"].split(";")

cost_source_hashes = {row["source_sha256"] for dataset in DATASETS for row in cost[dataset].values()}
assert len(cost_source_hashes) == 1
peacock_mae = select_one(
    paired_rows, dataset="peacock_education", left="LEVEL", right="C2_SMALL_MV",
    period="combined", metric="mae",
)
assert peacock_mae["fixed_seeds_conditional"] == "True"
assert peacock_mae["not_equivalence_or_noninferiority"] == "True"
assert float(cost["jena"]["LEVEL"]["mean_seed_median_peak_reserved_bytes"]) < float(cost["jena"]["C2_SMALL_MV"]["mean_seed_median_peak_reserved_bytes"])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.6))
fig.subplots_adjust(left=0.105, right=0.985, top=0.78, bottom=0.18, wspace=0.85)
fig.suptitle("Stored accuracy comparison", x=0.105, y=0.97, ha="left", fontsize=21, fontweight="bold")
fig.text(0.105, 0.89, "Combined test periods | Selected-seed loss means; zero-shot models counted once", fontsize=12, color="#4c5962")

for ax, dataset in zip(axes, DATASETS):
    values = [float(accuracy[dataset][method]["mse"]) for method in METHODS]
    y = list(range(len(METHODS)))
    bars = ax.barh(y, values, height=0.67, color=[COLORS[method] for method in METHODS])
    bars[1].set_hatch("//")
    bars[-1].set_hatch("..");
    ax.set_yticks(y, [METHOD_NAMES[method] for method in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.27)
    ax.set_xlabel("MSE (lower is better)")
    ax.set_title(DATASET_NAMES[dataset], loc="left", fontweight="bold", pad=17)
    tidy_axes(ax)
    for index, value in enumerate(values):
        ax.text(value + max(values) * 0.025, index, f"{value:.4f}", va="center", fontsize=10.5)

fig.text(0.105, 0.055, "Each panel has its own zero-based scale. Original LEVEL summaries only; no cross-dataset MSE average.", fontsize=10.5, color="#4c5962")
accuracy_paths = export(fig, "accuracy_summary")

fig, axes = plt.subplots(3, 3, figsize=(15.8, 10.8), gridspec_kw={"height_ratios": [0.95, 1.05, 1.2]})
fig.subplots_adjust(left=0.10, right=0.98, top=0.79, bottom=0.20, hspace=0.70, wspace=0.48)
fig.suptitle("LEVEL and Small MV: stored practical comparison", x=0.10, y=0.97, ha="left", fontsize=20, fontweight="bold")
fig.text(0.10, 0.91, "Relative MSE = 100 × (LEVEL - Small MV) / Small MV | Positive favors Small MV", fontsize=12)
fig.text(0.10, 0.866, "Accuracy: combined periods, original LEVEL | Cost: same stored CUDA campaign, batch 4", fontsize=11.5, color="#4c5962")

ci_limits = {"robin": (-5, 50), "peacock_education": (-5, 8), "jena": (-5, 36)}
for column, dataset in enumerate(DATASETS):
    pair = paired[dataset]
    point = float(pair["relative_percent"])
    low, high = json.loads(pair["relative_ci95_percent"])
    ax = axes[0, column]
    ax.set_title(DATASET_NAMES[dataset], loc="left", fontweight="bold", pad=18)
    ax.axvline(0, color="#66727a", linewidth=1, linestyle="--")
    ax.errorbar(point, 0, xerr=[[point - low], [high - point]], fmt="o", color=LEVEL_COLOR, ecolor=LEVEL_COLOR, capsize=7, markersize=9, linewidth=2.5)
    ax.set_xlim(*ci_limits[dataset])
    ax.set_ylim(-0.75, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("Relative MSE difference (%)")
    ax.text(0.5, 0.84, f"{point:+.2f}%  [{low:.2f}, {high:.2f}]", transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")
    tidy_axes(ax)
    if column == 0:
        ax.set_ylabel("Stored conditional\n95% CI", labelpad=12)

    ax = axes[1, column]
    throughput = [float(cost[dataset][method]["mean_seed_median_origins_s"]) for method in ("LEVEL", "C2_SMALL_MV")]
    ax.barh([0, 1], throughput, color=[LEVEL_COLOR, SMALL_COLOR], height=0.52)
    ax.set_yticks([0, 1], ["LEVEL", "Small MV"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(throughput) * 1.25)
    ax.set_xlabel("Origins/s (higher is better)")
    for index, value in enumerate(throughput):
        ax.text(value + max(throughput) * 0.025, index, f"{value:.2f}", va="center", fontsize=11)
    tidy_axes(ax)
    if column == 0:
        ax.set_ylabel("B4 throughput", labelpad=12)

    ax = axes[2, column]
    all_memory = []
    for index, method in enumerate(("LEVEL", "C2_SMALL_MV")):
        allocated = float(cost[dataset][method]["mean_seed_median_peak_allocated_bytes"]) / (1024 ** 2)
        reserved = float(cost[dataset][method]["mean_seed_median_peak_reserved_bytes"]) / (1024 ** 2)
        all_memory.extend((allocated, reserved))
        color = LEVEL_COLOR if method == "LEVEL" else SMALL_COLOR
        ax.barh(index - 0.18, allocated, height=0.29, color=color)
        ax.barh(index + 0.18, reserved, height=0.29, facecolor="white", edgecolor=color, hatch="///", linewidth=1.5)
        for y, value in ((index - 0.18, allocated), (index + 0.18, reserved)):
            ax.text(value + 5, y, f"{value:.1f}", va="center", fontsize=10.5)
    ax.set_yticks([0, 1], ["LEVEL", "Small MV"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(all_memory) * 1.27)
    ax.set_xlabel("Peak CUDA memory (MiB)")
    tidy_axes(ax)
    if column == 0:
        ax.set_ylabel("B4 peak memory", labelpad=12)
    if dataset == "jena":
        ax.text(0.02, -0.33, "Reserved: LEVEL 238 < Small MV 252 MiB", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#7d4d17")

fig.legend(handles=[
    Patch(facecolor="#596d7b", label="Allocated"),
    Patch(facecolor="white", edgecolor="#596d7b", hatch="///", label="Reserved"),
], loc="lower left", bbox_to_anchor=(0.095, 0.11), frameon=False, ncol=2, fontsize=11)
mae_point = float(peacock_mae["relative_percent"])
mae_low, mae_high = json.loads(peacock_mae["relative_ci95_percent"])
fig.text(0.10, 0.09, f"Peacock caveat: MAE is adverse for LEVEL: {mae_point:+.2f}% [{mae_low:.2f}, {mae_high:.2f}]. MSE CI crosses zero; no equivalence claim.", fontsize=11, color="#824316")
fig.text(0.10, 0.052, "CIs condition on the selected fixed seeds. Cost summaries: mean of selected-seed medians; no training time shown.", fontsize=10.5, color="#4c5962")
fig.text(0.10, 0.022, "Stored summaries only. Independent panel scales. No new experiment, inference, evaluation, bootstrap, or GPU measurement.", fontsize=10.5, color="#4c5962")
practical_paths = export(fig, "practical_tradeoffs")

accuracy_manifest = []
cost_manifest = []
paired_manifest = []
for dataset in DATASETS:
    for method in METHODS:
        row = accuracy[dataset][method]
        accuracy_manifest.append({
            **row_reference(row, "accuracy_comparison.csv", ("dataset", "method", "id", "seed", "aggregation", "period")),
            "display_method": METHOD_NAMES[method],
            "values": {"mse": float(row["mse"]), "mae": float(row["mae"]), "target_count": int(row["target_count"]), "origins": int(row["origins"])},
            "display_mse": f"{float(row['mse']):.4f}",
        })
    row = paired[dataset]
    paired_manifest.append({
        **row_reference(row, "paired_comparisons.csv", ("dataset", "left", "right", "period", "metric")),
        "values": {
            "left_value": float(row["left_value"]), "right_value": float(row["right_value"]),
            "relative_percent": float(row["relative_percent"]), "denominator": float(row["denominator"]),
            "relative_ci95_percent": json.loads(row["relative_ci95_percent"]),
            "bootstrap_draws_stored": int(row["bootstrap_draws"]), "block_origins_stored": int(row["block_origins"]),
            "fixed_seeds_conditional": row["fixed_seeds_conditional"] == "True",
            "not_equivalence_or_noninferiority": row["not_equivalence_or_noninferiority"] == "True",
        },
    })
    for method in ("LEVEL", "C2_SMALL_MV"):
        row = cost[dataset][method]
        cost_manifest.append({
            **row_reference(row, "cost_comparison.csv", ("aggregation", "dataset", "method", "device", "batch", "execution", "chunk_rows")),
            "ids": json.loads(row["ids"]),
            "stored_summary": {
                "path": "research/level_chronos2_controls_v1/cost_cuda_summary.json",
                "pointer": row["source_pointer"], "sha256_as_recorded_in_csv": row["source_sha256"],
            },
            "cost_aggregation": row["cost_aggregation"],
            "values": {
                "combined_mse": float(row["combined_mse"]),
                "mean_seed_median_origins_s": float(row["mean_seed_median_origins_s"]),
                "mean_seed_median_peak_allocated_bytes": float(row["mean_seed_median_peak_allocated_bytes"]),
                "mean_seed_median_peak_reserved_bytes": float(row["mean_seed_median_peak_reserved_bytes"]),
                "allocated_mib": float(row["mean_seed_median_peak_allocated_bytes"]) / (1024 ** 2),
                "reserved_mib": float(row["mean_seed_median_peak_reserved_bytes"]) / (1024 ** 2),
            },
        })

manifest = {
    "provenance": "Stored CSV summaries only from research/level_chronos2_controls_v1; no new experiment.",
    "operations": {"training": 0, "inference": 0, "reevaluation": 0, "new_bootstrap": 0, "gpu_measurements": 0, "downloads": 0},
    "input_files": [accuracy_input, paired_input, cost_input],
    "source_row_numbering": "Data row indices start at 1 after the header; csv_line_1_based records the source CSV line.",
    "figures": {
        "accuracy_summary": {
            "files": accuracy_paths,
            "description": "Three independent zero-based horizontal MSE panels for Robin, Peacock Education, and Jena; seven stored combined summaries per dataset. Original LEVEL summaries only.",
            "aggregation": summary_aggregation,
            "rows": accuracy_manifest,
            "cross_dataset_mse_average": False,
        },
        "practical_tradeoffs": {
            "files": practical_paths,
            "description": "LEVEL versus Small MV: stored combined relative MSE and conditional 95% CI, same-campaign CUDA B4 throughput, and peak allocated versus reserved memory. Independent dataset panel scales.",
            "relative_definition": "100 * (LEVEL - C2_SMALL_MV) / C2_SMALL_MV; positive favors C2_SMALL_MV",
            "paired_rows": paired_manifest,
            "cost_rows": cost_manifest,
            "peacock_mae_caveat": {
                **row_reference(peacock_mae, "paired_comparisons.csv", ("dataset", "left", "right", "period", "metric")),
                "relative_percent": mae_point,
                "relative_ci95_percent": [mae_low, mae_high],
                "fixed_seeds_conditional": True,
                "not_equivalence_or_noninferiority": True,
                "text": "Stored Peacock MSE CI crosses zero; no equivalence claim. Stored MAE relative difference is adverse for LEVEL.",
            },
            "jena_reserved_caveat": "Stored Jena B4 reserved memory is 238 MiB for LEVEL and 252 MiB for Small MV; allocated memory comparison has the opposite ordering.",
            "memory_conversion": "MiB = bytes / 1024**2; this is a display conversion only.",
        },
    },
    "render": {"png_dpi": DPI, "formats": ["png", "svg"], "english_labels": True},
    "validation": {
        "unique_selected_summary_rows": True,
        "paired_denominator_matches_small_mv_mse": True,
        "paired_values_match_accuracy_summary": True,
        "cost_ids_and_mse_match_accuracy_summary": True,
        "six_cost_rows_share_same_stored_summary_sha256": True,
        "conditional_and_not_equivalence_flags_checked": True,
        "jena_reserved_reversal_checked": True,
        "accuracy_summary_rows": len(accuracy_manifest),
        "mse_pair_rows": len(paired_manifest),
        "cuda_b4_cost_rows": len(cost_manifest),
    },
}
(OUT / "figure_values.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"outputs": accuracy_paths + practical_paths, "values": (OUT / "figure_values.json").relative_to(ROOT).as_posix(), "validation": manifest["validation"]}, indent=2))
