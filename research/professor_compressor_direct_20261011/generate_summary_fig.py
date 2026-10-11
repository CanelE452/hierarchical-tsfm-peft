"""Four-panel canonical future-error comparison from completed actual metric CSV only."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from common_direct import HERE, Resources, digest, read_json, write_json


GROUPS = ("Raw", "Mean-block", "GT full", "GT no spatial", "TS2 d16", "TS2 d320")
METHOD_GROUP = {"GT_full": 2, "GT_no_spatial": 3, "TS2_d16": 4, "TS2_d320": 5}
SEEDS = (92601, 92602)
PATHS = ("TRANSFER_F", "CALIBRATED_F", "TS2_NATIVE_RIDGE")
LABELS = {"RAW_F": "Raw F", "MEAN_BLOCK_F": "Mean-block F", "TRANSFER_F": "A: transfer",
          "CALIBRATED_F": "B: calibrated D", "TS2_NATIVE_RIDGE": "TS2 native ridge"}
COLORS = {"RAW_F": "#0072B2", "MEAN_BLOCK_F": "#777777", "TRANSFER_F": "#E69F00",
          "CALIBRATED_F": "#009E73", "TS2_NATIVE_RIDGE": "#D55E00"}
MARKERS = {"RAW_F": "*", "MEAN_BLOCK_F": "P", "TRANSFER_F": "o",
           "CALIBRATED_F": "s", "TS2_NATIVE_RIDGE": "D"}


def read_canonical_rows(protocol, csv_path):
    planned = {job["id"]: job for job in protocol["jobs"]}
    if len(planned) != 16:
        raise ValueError("Comparison requires all 16 unique explicit canonical protocol IDs")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        source = [row for row in csv.DictReader(handle) if row.get("stage") == "forecast"]
    rows, excluded, keys = [], [], set()
    contracts = read_json(HERE / "DATA_CHECKS.json")["datasets"]
    for row in source:
        baseline = row["method"] == "baseline"
        if not baseline and row["job_id"] not in planned:
            excluded.append({"job_id": row["job_id"], "arm": row["arm"], "reason": "outside explicit canonical protocol IDs"})
            continue
        if row["data_contract_hash"] != contracts[row["dataset"]]["contract_hash"]:
            raise ValueError("Comparison row differs from original TRAIN/VAL data contract")
        if not baseline:
            job = planned[row["job_id"]]
            expected_method = ("GT_no_spatial" if job.get("no_spatial") else "GT_full") if job["family"] == "GT" else f"TS2_d{job['dim']}"
            if row["dataset"] != job["dataset"] or int(row["seed"]) != job["seed"] or row["method"] != expected_method:
                raise ValueError("Metric CSV condition differs from exact canonical protocol ID")
        key = (row["dataset"], row["job_id"], row["arm"])
        if key in keys:
            raise ValueError(f"Duplicate actual metric row: {key}")
        keys.add(key)
        row = {**row, "mse": float(row["mse"]), "mae": float(row["mae"])}
        if not np.isfinite([row["mse"], row["mae"]]).all():
            raise ValueError("Non-finite saved actual future metric")
        rows.append(row)
    for identifier, job in planned.items():
        required = PATHS if job["family"] == "TS2" else PATHS[:2]
        for arm in required:
            if (job["dataset"], identifier, arm) not in keys:
                raise ValueError(f"Comparison remains incomplete: {identifier} / {arm}")
    for dataset in ("robin", "jena"):
        for arm in ("RAW_F", "MEAN_BLOCK_F"):
            if sum(row["dataset"] == dataset and row["method"] == "baseline" and row["arm"] == arm for row in rows) != 1:
                raise ValueError(f"Exactly one actual {dataset}/{arm} baseline is required")
    verification = read_json(HERE / "ANALYSIS_VERIFICATION.json")
    if verification.get("status") != "PASS_SAVED_PREDICTIONS_SCOPE" or verification.get("verified_representation_fits") != 16:
        raise ValueError("Full canonical saved-prediction verification must pass before final comparison")
    return rows, excluded


def panel(fig, grid, dataset, metric, rows):
    sub = grid.subgridspec(2, 1, height_ratios=(3.1, 1.75), hspace=.20)
    ax, table_ax = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    selected = [row for row in rows if row["dataset"] == dataset]
    lookup = {}
    for row in selected:
        if row["method"] == "baseline" and row["arm"] in ("RAW_F", "MEAN_BLOCK_F"):
            group = 0 if row["arm"] == "RAW_F" else 1
            lookup[(group, row["arm"], None)] = row[metric]
        elif row["arm"] in PATHS and row["method"] in METHOD_GROUP:
            lookup[(METHOD_GROUP[row["method"]], row["arm"], int(row["seed"]))] = row[metric]
    raw = lookup[(0, "RAW_F", None)]
    ax.axhline(raw, color=COLORS["RAW_F"], ls="--", lw=1.2, alpha=.8)
    for (group, arm, seed), value in lookup.items():
        path_shift = (-.20, 0., .20)[PATHS.index(arm)] if arm in PATHS else 0.
        seed_shift = -.035 if seed == SEEDS[0] else (.035 if seed == SEEDS[1] else 0.)
        ax.plot(group + path_shift + seed_shift, value, marker=MARKERS[arm], markersize=7.5,
                linestyle="none", markeredgewidth=1.5, color=COLORS[arm],
                markerfacecolor="white" if seed == SEEDS[1] else COLORS[arm])
    ax.set_xticks(range(len(GROUPS)), ("Raw", "Mean\nblock", "GT\nfull", "GT\nno spatial", "TS2\nd16", "TS2\nd320"))
    ax.set_xlim(-.55, 5.55)
    ax.set_ylim(0, max(lookup.values()) * 1.15 if max(lookup.values()) > 0 else 1)
    ax.set_title(f"{dataset.upper()} — observed channel-macro {metric.upper()}", fontsize=14, loc="left")
    ax.set_ylabel(f"TRAIN-standardized {metric.upper()} (lower is better)")
    ax.grid(axis="y", alpha=.20)
    ax.spines[["top", "right"]].set_visible(False)

    table_ax.axis("off")
    table_rows, row_labels = [], []
    for arm, seed in (("baselines", None), *((arm, seed) for arm in PATHS for seed in SEEDS)):
        short_label = {"TRANSFER_F": "A", "CALIBRATED_F": "B", "TS2_NATIVE_RIDGE": "Native"}
        label = "Baseline" if seed is None else f"{short_label[arm]} / {seed}"
        values = []
        for group in range(len(GROUPS)):
            key_arm = ("RAW_F" if group == 0 else "MEAN_BLOCK_F") if seed is None else arm
            value = lookup.get((group, key_arm, seed))
            values.append("—" if value is None else f"{value:.6f}")
        table_rows.append(values)
        row_labels.append(label)
    table = table_ax.table(cellText=[[label, *values] for label, values in zip(row_labels, table_rows)],
                           colLabels=("Path / seed", *GROUPS), colWidths=(.25, *([.125] * 6)),
                           cellLoc="center", loc="center", bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D9DFE6")
        cell.set_linewidth(.45)
        if row == 0:
            cell.set_facecolor("#EEF2F7")
        if column == 0:
            cell.get_text().set_ha("left")
        if row > 1 and column == 0:
            arm = PATHS[(row - 2) // 2]
            cell.get_text().set_color(COLORS[arm])


def main():
    resources = Resources()
    with resources.stage("canonical_method_comparison_4panels_CPU", gpu=False):
        csv_path = HERE / "metrics.csv"
        protocol_path = HERE / "protocol.json"
        rows, excluded = read_canonical_rows(read_json(protocol_path), csv_path)
        figures_dir = HERE / "figures"
        figures_dir.mkdir(exist_ok=True)
        output = figures_dir / "METHOD_COMPARISON.png"
        manifest_path = HERE / "METHOD_COMPARISON_MANIFEST.json"
        hashes = {"metrics_csv_sha256": digest(csv_path), "protocol_sha256": digest(protocol_path),
                  "analysis_verification_sha256": digest(HERE / "ANALYSIS_VERIFICATION.json"),
                  "source_sha256": digest(Path(__file__))}
        if output.exists():
            archive_dir = figures_dir / "summary_archive"
            archive_dir.mkdir(exist_ok=True)
            suffix = str(time.time_ns())
            shutil.copy2(output, archive_dir / f"METHOD_COMPARISON_{suffix}.png")
            if manifest_path.exists():
                shutil.copy2(manifest_path, archive_dir / f"METHOD_COMPARISON_MANIFEST_{suffix}.json")
        fig = plt.figure(figsize=(20, 14), facecolor="white")
        outer = fig.add_gridspec(2, 2, left=.04, right=.98, bottom=.075, top=.89, hspace=.24, wspace=.16)
        for row, dataset in enumerate(("robin", "jena")):
            for column, metric in enumerate(("mse", "mae")):
                panel(fig, outer[row, column], dataset, metric, rows)
        legend = [Line2D([], [], color=COLORS[arm], marker=MARKERS[arm], ls="none", markersize=8,
                         label=LABELS[arm]) for arm in ("RAW_F", "MEAN_BLOCK_F", *PATHS)]
        legend += [Line2D([], [], color="#333333", marker="o", ls="none", label="Seed 92601: filled"),
                   Line2D([], [], color="#333333", marker="o", markerfacecolor="white", ls="none", label="Seed 92602: hollow")]
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .947), ncol=7, frameon=False, fontsize=10)
        fig.suptitle("Actual future errors across native compressor paths", fontsize=20, y=.985)
        fig.text(.04, .915, "Exact values for both seeds appear below each panel; no confidence intervals or significance claims.", fontsize=10)
        fig.text(.04, .048, "Dashed blue line: same-data Raw F. A/B use the same native D; B uses supervised future calibration. "
                 "TS2 d320 is representation expansion. Existing development VAL; no independent TEST claim.", fontsize=9.5)
        fig.text(.04, .028, "Cost/memory are measured separately for the fixed resident prototype process; no isolated deployment-footprint claim. "
                 f"Canonical CSV SHA: {hashes['metrics_csv_sha256'][:16]}", fontsize=9.5)
        fig.savefig(output, dpi=180, facecolor="white")
        plt.close(fig)
        resources.check(phase="method_comparison_figure_saved")
        manifest = {**hashes, "output": output.relative_to(HERE).as_posix(), "output_sha256": digest(output),
                    "canonical_job_count": 16, "displayed_datasets": ["robin", "jena"], "displayed_metrics": ["mse", "mae"],
                    "fixed_group_order": list(GROUPS), "fixed_forecast_path_order": list(PATHS), "fixed_seeds": list(SEEDS),
                    "rule": "Display every canonical seed value, all groups and prescribed A/B/native paths; no score-based selection, CI or significance claim",
                    "excluded_noncanonical_rows": excluded,
                    "persistence_rows": "Preserved in metrics.csv and per-case figures; outside the predeclared main-method overview",
                    "displayed_raw_values": [{key: row[key] for key in ("dataset", "method", "seed", "job_id", "arm", "mse", "mae")}
                                             for row in rows if row["arm"] in ("RAW_F", "MEAN_BLOCK_F", *PATHS)],
                    "cost_scope": "Fixed resident prototype; no isolated deployment footprint inference",
                    "created_UTC": datetime.now(timezone.utc).isoformat()}
    manifest["resources"] = resources.value
    write_json(manifest_path, manifest)
    print(str({"output": str(output), "manifest": str(manifest_path), "metric_rows": len(rows)}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
