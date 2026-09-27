"""Create the fixed v4 comparison figure from completed evaluation/cost JSON files.

This script is read-only with respect to experiment data: it consumes already
computed TEST metrics and cost summaries. It does not load models, predictions,
targets, or raw arrays, and it does not run fitting, forecasting, scoring, or
benchmarking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from runtime_v4 import HERE, ROOT, Job, digest, save_json, source_receipt


FAMILY_LABELS = {
    "f0": "F0",
    "lora": "LoRA",
    "tsfm_res": "TSFM_RES",
    "tsfm_raw": "TSFM_RAW",
    "compress": "COMPRESS",
    "linear_res": "LINEAR_RES",
    "shared": "SHARED",
}
PREFERRED_ROLE_ORDER = ("f0", "lora", "tsfm_res", "tsfm_raw", "compress", "linear_res", "shared")
PERIODS = ("test_a", "test_b", "combined")
PERIOD_LABELS = {"test_a": "TEST-A", "test_b": "TEST-B", "combined": "A+B"}
PERIOD_COLORS = {"test_a": "#5B8FF9", "test_b": "#61DDAA", "combined": "#1F2937"}
FAMILY_COLORS = {
    "f0": "#6B7280",
    "lora": "#B45309",
    "tsfm_res": "#2563EB",
    "tsfm_raw": "#7C3AED",
    "compress": "#059669",
    "linear_res": "#DC2626",
    "shared": "#4B5563",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def receipt(path: Path) -> dict[str, str]:
    actual = resolve(path)
    return {"path": actual.relative_to(ROOT).as_posix(), "sha256": digest(actual)}


def validate_inputs(evaluation: dict[str, Any], gpu_cost: dict[str, Any], cpu_cost: dict[str, Any] | None) -> None:
    if evaluation.get("status") != "complete":
        raise ValueError("Evaluation JSON must be complete")
    if "role_scores" not in evaluation or "models" not in evaluation or "roles" not in evaluation:
        raise ValueError("Evaluation JSON lacks role_scores/models/roles")
    if gpu_cost.get("status") != "complete" or gpu_cost.get("stage") != "gpu":
        raise ValueError("GPU cost JSON must be complete and stage='gpu'")
    if "rows" not in gpu_cost or "summary" not in gpu_cost:
        raise ValueError("GPU cost JSON lacks rows/summary")
    if cpu_cost is not None and (cpu_cost.get("status") != "complete" or cpu_cost.get("stage") != "cpu"):
        raise ValueError("CPU cost JSON, when provided, must be complete and stage='cpu'")


def selected_roles(evaluation: dict[str, Any]) -> list[str]:
    present = set(evaluation["role_scores"])
    ordered = [role for role in PREFERRED_ROLE_ORDER if role in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def role_mse_table(evaluation: dict[str, Any], roles: list[str]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for role in roles:
        table[role] = {}
        for period in PERIODS:
            score = evaluation["role_scores"][role][period]
            table[role][period] = {
                "mse": score["mse"],
                "mae": score.get("mae"),
                "count": score.get("count"),
                "run_ids": score.get("run_ids", evaluation["roles"].get(role, [])),
            }
    return table


def model_period_mse(evaluation: dict[str, Any], identifier: str, period: str) -> float | None:
    return evaluation["models"][identifier]["scores"][period]["mse"]


def role_seed_points(evaluation: dict[str, Any], role: str, period: str) -> list[dict[str, Any]]:
    points = []
    for identifier in evaluation["roles"].get(role, []):
        spec = evaluation["models"][identifier]["spec"]
        points.append({
            "id": identifier,
            "seed": spec.get("seed"),
            "mse": model_period_mse(evaluation, identifier, period),
        })
    return points


def fit_cost_index(gpu_cost: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fits = {}
    for fit in gpu_cost["summary"]["fits"]:
        if fit.get("batch_origins") == 4 and fit.get("scope") == "deployment_cpu_to_cpu":
            fits[fit["id"]] = fit
    return fits


def row_block_ranges(gpu_cost: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ranges: dict[str, dict[str, Any]] = {}
    for row in gpu_cost["rows"]:
        if not row.get("include_in_primary", False):
            continue
        if row.get("sentinel_role") == "post":
            continue
        if row.get("batch_origins") != 4 or row.get("scope") != "deployment_cpu_to_cpu":
            continue
        entry = ranges.setdefault(row["id"], {"throughput": [], "peak_allocated_bytes": []})
        entry["throughput"].append(float(row["origins_per_second"]["median"]))
        if row.get("peak_allocated_bytes") is not None:
            entry["peak_allocated_bytes"].append(int(row["peak_allocated_bytes"]))
    return ranges


def cost_points(evaluation: dict[str, Any], gpu_cost: dict[str, Any]) -> list[dict[str, Any]]:
    fits = fit_cost_index(gpu_cost)
    block_ranges = row_block_ranges(gpu_cost)
    points = []
    for identifier, fit in sorted(fits.items()):
        if identifier not in evaluation["models"]:
            continue
        family = fit["family"]
        if family == "shared":
            continue
        mse = model_period_mse(evaluation, identifier, "combined")
        if mse is None:
            continue
        throughput = fit["origins_per_second"]
        peaks = [value for value in fit.get("peak_allocated_bytes", []) if value is not None]
        if not peaks:
            continue
        blocks = block_ranges.get(identifier, {})
        points.append({
            "id": identifier,
            "family": family,
            "ed_mode": fit.get("ed_mode"),
            "seed": fit.get("seed"),
            "combined_mse": float(mse),
            "throughput_median": float(throughput["median"]),
            "throughput_min": float(throughput["min"]),
            "throughput_max": float(throughput["max"]),
            "throughput_block_medians": blocks.get("throughput", []),
            "peak_allocated_mib_median": float(np.median(peaks) / 1024**2),
            "peak_allocated_mib_min": float(min(peaks) / 1024**2),
            "peak_allocated_mib_max": float(max(peaks) / 1024**2),
            "peak_allocated_block_bytes": peaks,
        })
    return points


def draw_box(ax, xy, text, fc="#F8FAFC", ec="#64748B", width=0.12, height=0.09):
    import matplotlib.patches as patches

    x, y = xy
    box = patches.FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=8, color="#111827")


def draw_arrow(ax, start, end, color="#64748B", rad=0.0):
    import matplotlib.patches as patches

    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def plot_diagram(ax) -> None:
    ax.set_axis_off()
    ax.set_title("A. Evaluated paths", loc="left", fontsize=11, weight="bold")
    rows = (
        ("TSFM_RES", "D(F0(E(X)))", "G(X - D(E(X)))", "#EFF6FF", "#2563EB"),
        ("TSFM_RAW", "D(F0(E(X)))", "G(X)", "#EFF6FF", "#2563EB"),
        ("COMPRESS", "D(F0(E(X)))", None, "#EFF6FF", "#2563EB"),
        ("LINEAR_RES", "D(T̃(E(X)))", "G(X - D(E(X)))", "#FEF2F2", "#DC2626"),
    )
    y_values = (0.82, 0.60, 0.38, 0.16)
    for (label, main_text, aux_text, main_fc, main_ec), y in zip(rows, y_values):
        ax.text(0.08, y, label, ha="right", va="center", fontsize=8.5, weight="bold")
        draw_box(ax, (0.15, y), "X", width=0.07, height=0.07)
        if aux_text is None:
            draw_box(ax, (0.43, y), main_text, fc=main_fc, ec=main_ec, width=0.34, height=0.075)
            draw_box(ax, (0.80, y), "ŷ", width=0.08, height=0.07)
            draw_arrow(ax, (0.19, y), (0.26, y))
            draw_arrow(ax, (0.60, y), (0.75, y))
            continue

        main_y = y + 0.045
        aux_y = y - 0.045
        draw_box(ax, (0.43, main_y), main_text, fc=main_fc, ec=main_ec, width=0.34, height=0.065)
        draw_box(ax, (0.43, aux_y), aux_text, fc="#F0FDF4", ec="#059669", width=0.34, height=0.065)
        draw_box(ax, (0.70, y), "+", width=0.06, height=0.065)
        draw_box(ax, (0.84, y), "ŷ", width=0.08, height=0.07)
        draw_arrow(ax, (0.19, y + 0.018), (0.26, main_y))
        draw_arrow(ax, (0.19, y - 0.018), (0.26, aux_y))
        draw_arrow(ax, (0.60, main_y), (0.66, y + 0.015))
        draw_arrow(ax, (0.60, aux_y), (0.66, y - 0.015))
        draw_arrow(ax, (0.73, y), (0.79, y))
    ax.text(
        0.01,
        0.01,
        "G input: past reconstruction residual X-D(E(X)), or raw X.",
        fontsize=7,
        color="#4B5563",
    )
    ax.set_xlim(-0.12, 0.92)
    ax.set_ylim(0, 1)


def plot_mse_panel(ax, evaluation: dict[str, Any], roles: list[str]) -> dict[str, Any]:
    import numpy as np

    x = np.arange(len(roles), dtype=float)
    offsets = {"test_a": -0.23, "test_b": 0.0, "combined": 0.23}
    numbers = {}
    for period in PERIODS:
        means = []
        for role in roles:
            value = evaluation["role_scores"][role][period]["mse"]
            means.append(np.nan if value is None else float(value))
            numbers.setdefault(role, {})[period] = {"role_mse": None if value is None else float(value)}
        ax.scatter(
            x + offsets[period],
            means,
            marker="D" if period == "combined" else "o",
            s=54 if period == "combined" else 42,
            color=PERIOD_COLORS[period],
            edgecolor="white",
            linewidth=0.7,
            label=f"{PERIOD_LABELS[period]} selected-role mean",
            zorder=3,
        )
        for pos, role in enumerate(roles):
            seed_values = role_seed_points(evaluation, role, period)
            numbers[role][period]["model_points"] = seed_values
            for j, point in enumerate(seed_values):
                if point["mse"] is None:
                    continue
                jitter = (-0.035 + 0.07 * j / max(1, len(seed_values) - 1)) if len(seed_values) > 1 else 0.0
                ax.scatter(
                    pos + offsets[period] + jitter,
                    float(point["mse"]),
                    s=18,
                    facecolor="none",
                    edgecolor=PERIOD_COLORS[period],
                    linewidth=0.9,
                    zorder=4,
                )
    ax.set_title("B. TEST MSE after VAL selection", loc="left", fontsize=11, weight="bold")
    ax.set_ylabel("Masked normalized MSE")
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS.get(role, role) for role in roles], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=1, frameon=False, loc="best")
    return numbers


def plot_scatter(ax, points: list[dict[str, Any]], x_key: str, x_label: str, title: str) -> None:
    labeled = set()
    for point in points:
        family = point["family"]
        color = FAMILY_COLORS.get(family, "#111827")
        x = float(point[x_key])
        y = float(point["combined_mse"])
        if x_key == "throughput_median":
            low, high = point["throughput_min"], point["throughput_max"]
        else:
            low, high = point["peak_allocated_mib_min"], point["peak_allocated_mib_max"]
        xerr = [[max(0.0, x - low)], [max(0.0, high - x)]]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o" if point.get("seed") == 92601 else "^" if point.get("seed") == 92602 else "s",
            markersize=5.2,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            alpha=0.92,
            label=FAMILY_LABELS.get(family, family) if family not in labeled else None,
        )
        labeled.add(family)
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    ax.set_xscale("log")
    ax.set_xlabel(x_label + "; log scale")
    ax.set_ylabel("Combined TEST MSE")
    mse_values = [point["combined_mse"] for point in points]
    ax.set_ylim(min(mse_values) - 0.2, max(mse_values) + 0.7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper right",
              title="o: seed 92601, triangle: 92602, square: fixed", title_fontsize=6.5)


def build_caption(
    args: argparse.Namespace,
    evaluation: dict[str, Any],
    gpu_cost: dict[str, Any],
    cpu_cost: dict[str, Any] | None,
    role_numbers: dict[str, Any],
    cost_numbers: list[dict[str, Any]],
    outputs: dict[str, Path],
) -> dict[str, Any]:
    return {
        "schema": "tsfm_peft_fresh_group_v4_plot_caption_v1",
        "inputs": {
            "evaluation": receipt(args.evaluation),
            "gpu_cost": receipt(args.gpu_cost),
            "cpu_cost": receipt(args.cpu_cost) if args.cpu_cost else None,
        },
        "source": source_receipt(),
        "numbers": {
            "role_mse": role_numbers,
            "gpu_batch4_cost_points": cost_numbers,
            "cpu_cost_excluded_from_gpu_panels": cpu_cost.get("summary") if cpu_cost is not None else None,
            "cost_definition": gpu_cost.get("summary", {}).get("definition"),
            "combined_definition": evaluation.get("combined_definition"),
        },
        "exclusions": {
            "gpu_scatter": "Uses GPU stage, batch4, deployment_cpu_to_cpu, include_in_primary rows only; F0 post sentinels and CPU FP32 SHARED/LINEAR deployment timing are not mixed into GPU panels.",
            "cpu_cost": "If provided, CPU cost is only recorded in this caption for separate reporting.",
            "selection": "Uses all selected roles from evaluation JSON; no channel/period cherry picking.",
        },
        "caption": (
            "V4 Hog comparison. Panel A shows the evaluated adapter/backbone paths. "
            "Panel B reports TEST-A, TEST-B, and combined masked normalized MSE after VAL selection; "
            "open markers are individual seed or deterministic model scores and role means are not ensembles. "
            "Panels C-D relate combined TEST MSE to GPU batch4 throughput and peak allocated memory; horizontal bars preserve the "
            "three order-block range for each fitted model. CPU-only SHARED deployment timing is excluded from GPU panels."
        ),
        "outputs": {
            key: {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}
            for key, path in outputs.items()
        },
    }


def make_figure(args: argparse.Namespace) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluation = read_json(resolve(args.evaluation))
    gpu_cost = read_json(resolve(args.gpu_cost))
    cpu_cost = read_json(resolve(args.cpu_cost)) if args.cpu_cost else None
    validate_inputs(evaluation, gpu_cost, cpu_cost)

    roles = selected_roles(evaluation)
    cost_numbers = cost_points(evaluation, gpu_cost)
    if not cost_numbers:
        raise ValueError("No GPU batch4 primary cost points matched evaluation model IDs")

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })
    fig = plt.figure(figsize=(13.2, 8.2), constrained_layout=True)
    mosaic = [["diagram", "mse"], ["throughput", "memory"]]
    axes = fig.subplot_mosaic(mosaic)

    plot_diagram(axes["diagram"])
    role_numbers = plot_mse_panel(axes["mse"], evaluation, roles)
    plot_scatter(
        axes["throughput"],
        cost_numbers,
        "throughput_median",
        "GPU batch4 throughput (origins/sec, 3-block median with min-max)",
        "C. GPU throughput vs accuracy",
    )
    plot_scatter(
        axes["memory"],
        cost_numbers,
        "peak_allocated_mib_median",
        "Peak allocated GPU memory (MiB, 3-block median with min-max)",
        "D. GPU memory vs accuracy",
    )
    axes["throughput"].text(
        0.01,
        0.02,
        "F0 post sentinels excluded; CPU FP32 SHARED/LINEAR speeds kept out of GPU scatter.",
        transform=axes["throughput"].transAxes,
        fontsize=7,
        color="#4B5563",
        va="bottom",
    )
    axes["memory"].text(
        0.01,
        0.02,
        "Horizontal bars show the three order-block range per seed/deterministic fit.",
        transform=axes["memory"].transAxes,
        fontsize=7,
        color="#4B5563",
        va="bottom",
    )
    fig.suptitle("V4 Hog fresh-group comparison: accuracy and GPU cost after VAL selection", fontsize=13, weight="bold")

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": output_dir / "comparison.png",
        "svg": output_dir / "comparison.svg",
    }
    caption_path = output_dir / "caption.json"
    existing = [path for path in (*outputs.values(), caption_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Preserve existing plot outputs; use a new output directory: {existing}")
    fig.savefig(outputs["png"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    caption = build_caption(args, evaluation, gpu_cost, cpu_cost, role_numbers, cost_numbers, outputs)
    save_json(caption_path, caption)
    outputs["caption"] = caption_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True, help="Completed evaluate_v4 JSON.")
    parser.add_argument("--gpu-cost", type=Path, required=True, help="Completed cost_v4 --stage gpu JSON.")
    parser.add_argument("--cpu-cost", type=Path, help="Optional completed cost_v4 --stage cpu JSON; recorded but not mixed into GPU panels.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures_v4")
    parser.add_argument("--name", default="v4_plot_comparison")
    parser.add_argument("--reserve-seconds", type=float, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Job(args.name, category="cpu_analysis", reserve_seconds=args.reserve_seconds,
             metadata={"plot": "v4_comparison", "no_new_predictions_or_benchmarks": True}) as job:
        job.heartbeat("reading_completed_evaluation_and_cost_json")
        outputs = make_figure(args)
        job.heartbeat("plot_files_written")
    print(json.dumps({key: path.relative_to(ROOT).as_posix() for key, path in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
