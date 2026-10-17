"""Render the stopped campaign's summary CSVs; never run experiments."""

import argparse
import csv
import hashlib
import math
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def only(rows, **conditions):
    selected = [row for row in rows if all(row[key] == str(value) for key, value in conditions.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one summary row for {conditions}; found {len(selected)}")
    return selected[0]


def configure():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "svg.fonttype": "none",
        "svg.hashsalt": "b2-stopped-summary-v1",
        "savefig.facecolor": "white",
    })


def save_figure(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.svg", metadata={"Date": None, "Creator": "Stopped campaign summary CSV renderer"})
    fig.savefig(output_dir / f"{stem}.png", dpi=300, metadata={"Software": "Stopped campaign summary CSV renderer"})


def selected_models(rows, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    families = ("B0", "B2", "B2_CAP_POINTWISE")
    labels = ("B0", "B2", "CAP pointwise")
    colors = ("#6B7280", "#0072B2", "#D55E00")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.1), sharex=True)
    fig.suptitle("Selected models: Jena, C=21 / K=6", x=0.06, y=0.96, ha="left", weight="bold", fontsize=16)
    fig.text(0.06, 0.895, "Fresh A60 refit after the same A-stop selection rule; lower channel-macro MSE is better.", fontsize=10)
    max_score = 0
    for ax, seed in zip(axes, (92601, 92602)):
        values = []
        for index, family in enumerate(families):
            row = only(rows, comparison_id="SELECTED_A60", model=family, seed=seed)
            value = number(row["MSE"])
            if value is None:
                raise ValueError("Selected model summary must have an actual saved score")
            values.append(value)
            max_score = max(max_score, value)
            ax.barh(index, value, height=0.58, color=colors[index], edgecolor="white", linewidth=0.8)
            epoch = number(row["epoch"])
            mae = number(row["MAE"])
            ax.text(value + 0.012, index, f"{value:.4f}\nMAE {mae:.4f}; refit {epoch:g} epoch", va="center", fontsize=9)
        ax.set_yticks(np.arange(3), labels)
        ax.invert_yaxis()
        ax.set_title(f"Seed {seed}", loc="left", pad=14, weight="bold")
        ax.set_xlabel("Channel-macro MSE (A60 normalized units)")
        ax.xaxis.grid(True, color="#E5E7EB", linewidth=0.7)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for ax in axes:
        ax.set_xlim(0, max_score * 1.5)
    fig.text(0.06, 0.107, "Active parameters: B0 1,008 | B2 35,717 | CAP 35,708. Each seed is shown separately; no seed-average or CI.", fontsize=9)
    fig.text(0.06, 0.061, "Repeated development V (43 origins). Selected refit lengths differ: this is not a matched-update comparison.", fontsize=9)
    fig.text(0.06, 0.023, "CAP: saved native score; final independent prediction-array FP64 recheck was not executed before the user stop.", fontsize=9, color="#9A3412")
    fig.subplots_adjust(left=0.125, right=0.97, top=0.79, bottom=0.25, wspace=0.42)
    save_figure(fig, output_dir, "STOPPED_SELECTED_MODELS")
    plt.close(fig)


def matched_updates(rows, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    splits = ("A_fit", "A_stop", "V_select")
    split_labels = ("A-fit evaluation / A50 scale", "A-stop evaluation / A50 scale", "Development V / A60 scale")
    style = {"B0": ("#6B7280", "o", "--"), "B2": ("#0072B2", "s", "-")}
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0), sharex=True, sharey="col")
    fig.suptitle("Matched updates: existing B0 and B2 checkpoints", x=0.065, y=0.965, ha="left", weight="bold", fontsize=16)
    fig.text(0.065, 0.914, "Same sample exposure and optimizer updates at each endpoint; each curve uses saved evaluation summaries.", fontsize=10)
    fig.text(0.065, 0.877, "CAP matched-update evaluation: UNAVAILABLE - interrupted before first saved result; no selected-score substitution.", color="#9A3412", fontsize=10)
    for i, seed in enumerate((92601, 92602)):
        for j, split in enumerate(splits):
            ax = axes[i, j]
            for family, (color, marker, line) in style.items():
                series = [row for row in rows if row["comparison_id"] == "SAME_UPDATE" and row["model"] == family and row["seed"] == str(seed) and row["split"] == split]
                series.sort(key=lambda row: float(row["updates"]))
                if len(series) != 4 or any(number(row["MSE"]) is None for row in series):
                    raise ValueError(f"Incomplete matched-update curve: {family}, {seed}, {split}")
                ax.plot([float(row["updates"]) for row in series], [float(row["MSE"]) for row in series], color=color, marker=marker, linestyle=line, linewidth=1.8, markersize=5)
            ax.grid(True, color="#E5E7EB", linewidth=0.65)
            ax.set_xticks([0, 156, 622, 1863], ["0", "156", "622", "1,863"])
            ax.tick_params(labelsize=9)
            if i == 0:
                ax.set_title(split_labels[j], loc="left", pad=12, fontsize=11, weight="bold")
            if j == 0:
                ax.set_ylabel(f"Seed {seed}\nChannel-macro MSE")
            if i == 1:
                ax.set_xlabel("Optimizer updates")
    handles = [Line2D([0], [0], color=color, marker=marker, linestyle=line, label=family, linewidth=1.8) for family, (color, marker, line) in style.items()]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.965, 0.041), ncol=2, frameon=False)
    fig.text(0.065, 0.078, "Endpoints: 0 / 0.25 / 1 / 3 epochs. Exposures: 0 / 2,481 / 9,923 / 29,769.", fontsize=9)
    fig.text(0.065, 0.040, "A-fit and A-stop use A50 search scaling; V uses canonical A60 scaling. Panels are not directly pooled.", fontsize=9)
    fig.text(0.065, 0.008, "Native summaries only for the new replay; final independent FP64 array recheck was not executed. Convergence is unresolved.", fontsize=9, color="#9A3412")
    fig.subplots_adjust(left=0.085, right=0.965, top=0.79, bottom=0.18, hspace=0.38, wspace=0.22)
    save_figure(fig, output_dir, "STOPPED_MATCHED_UPDATES")
    plt.close(fig)


def work_status(rows, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    labels = {
        "A_saved_error_decomposition": "Existing prediction error decomposition",
        "B_existing_models": "Existing B0 / B2 model conditions",
        "B_CAP_search_refit_selectedV": "CAP search + refit + selected V prediction",
        "B_common_update_saved_scores": "Matched-update saved evaluation scores",
        "C_fixed_linear_probes": "Fixed-representation linear probes",
        "D_PeMS_C32_resume_training": "PeMS C32 / K8 completion",
        "D_new_full_pipeline_cost": "New full-pipeline CPU costs",
        "E_new_independent_FP64_closeout": "Final independent prediction-array check",
        "Independent_future_generalization": "Independent future generalization",
    }
    status_labels = {
        "COMPLETE_AUDITED": "Complete / independently checked",
        "COMPLETE_REUSED": "Complete / reused",
        "COMPLETE_NATIVE_NOT_FINAL_FP64": "Native complete; FP64 pending",
        "PARTIAL_USER_STOP": "Partial / stopped by user",
        "NOT_STARTED_USER_STOP": "Not started / user stop",
        "PENDING_NO_NEW_DATA": "Pending / no new unused data",
    }
    count = len(rows)
    fig, ax = plt.subplots(figsize=(12.2, max(5.6, 0.48 * count + 2.1)))
    fig.suptitle("Work preserved at the user stop", x=0.055, y=0.963, ha="left", weight="bold", fontsize=16)
    fig.text(0.055, 0.912, "Rows have different units. Counts describe execution status, not an overall research completion percentage.", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, count - 0.35)
    for i, row in enumerate(rows):
        y = count - 1 - i
        complete = number(row["completed"])
        registered = number(row["registered"])
        fraction = min(1.0, max(0.0, complete / registered)) if complete is not None and registered and registered > 0 else 0.0
        status = row["status"]
        color = "#0072B2" if status == "COMPLETE_NATIVE_NOT_FINAL_FP64" else "#009E73" if fraction == 1 and fraction > 0 else "#E69F00" if fraction > 0 else "#9CA3AF"
        label = labels.get(row["block"], row["block"].replace("_", " ").capitalize())
        ax.text(0.0, y, label, va="center", fontsize=10)
        ax.add_patch(Rectangle((0.49, y - 0.19), 0.19, 0.38, facecolor="#F0F2F5", edgecolor="#D1D5DB", linewidth=0.6))
        if fraction:
            ax.add_patch(Rectangle((0.49, y - 0.19), 0.19 * fraction, 0.38, facecolor=color, edgecolor="none"))
        numerator = f"{complete:g}" if complete is not None else "NA"
        denominator = f"{registered:g}" if registered is not None else "NA"
        ax.text(0.59, y, f"{numerator}/{denominator}", ha="center", va="center", fontsize=9, color="#111827")
        text = status_labels.get(status, status.replace("_", " "))
        ax.text(0.71, y, text, va="center", fontsize=9, color="#374151")
    ax.axis("off")
    fig.text(0.055, 0.087, "Completed outputs and partial evaluations are preserved; no experiments were restarted for this publication.", fontsize=9)
    fig.text(0.055, 0.045, "Final campaign FP64 verification and independent future generalization remain separate from native execution counts.", fontsize=9, color="#9A3412")
    fig.subplots_adjust(left=0.055, right=0.975, bottom=0.16, top=0.85)
    save_figure(fig, output_dir, "STOPPED_WORK_STATUS")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Render already exported summary CSVs only")
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if not args.execute:
        print("SOURCE_ONLY: no summary reads, rendering, experiments, or model access")
        return
    directory = args.report_dir.resolve()
    output_dir = directory / "figures"
    output_dir.mkdir(exist_ok=True)
    stems = ("STOPPED_SELECTED_MODELS", "STOPPED_MATCHED_UPDATES", "STOPPED_WORK_STATUS")
    outputs = [output_dir / f"{stem}.{extension}" for stem in stems for extension in ("svg", "png")]
    existing = [path.name for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Preserve existing figures; outputs already exist: {existing}")
    comparisons = read_csv(directory / "COMPARISONS_AT_STOP.csv")
    status = read_csv(directory / "STATUS_AT_STOP.csv")
    configure()
    selected_models(comparisons, output_dir)
    matched_updates(comparisons, output_dir)
    work_status(status, output_dir)
    for path in outputs:
        print(f"{path.name} {path.stat().st_size} bytes sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
