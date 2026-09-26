"""Create v2 follow-up result figures from completed evaluation artifacts.

This script is intentionally read-only with respect to model/data artifacts. It
does not train, infer, score new predictions, or inspect raw targets. It only
plots metrics and curve records already saved by evaluate_v2.py and optional
cost measurement reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / "research" / "tsfm_peft_development_20260926"
PLOTS = HERE / "plots"
SAFE_STEM = re.compile(r"[A-Za-z0-9_-]+")

MODE_ORDER = ["current", "fixed_ed", "slow_ed"]
LABEL_ORDER = [
    "f0",
    "seasonal",
    "seasonal24",
    "shared_linear",
    "factor_linear",
    "lora",
    "raw_bypass:current",
    "raw_bypass:fixed_ed",
    "raw_bypass:slow_ed",
    "residual:current",
    "residual:fixed_ed",
    "residual:slow_ed",
]

COLORS = {
    "residual:current": "#4C78A8",
    "residual:fixed_ed": "#F58518",
    "residual:slow_ed": "#54A24B",
    "raw_bypass:current": "#B279A2",
    "raw_bypass:fixed_ed": "#FF9DA6",
    "raw_bypass:slow_ed": "#9D755D",
    "lora": "#E45756",
    "factor_linear": "#72B7B2",
    "shared_linear": "#BAB0AC",
    "seasonal": "#8CD17D",
    "seasonal24": "#8CD17D",
    "f0": "#79706E",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def display_label(label: str) -> str:
    mapping = {
        "f0": "Frozen F0",
        "seasonal": "Seasonal",
        "seasonal24": "Seasonal24",
        "shared_linear": "Shared linear",
        "factor_linear": "Factor linear",
        "lora": "LoRA",
        "compress": "Compress",
        "residual:current": "CURRENT residual",
        "residual:fixed_ed": "Fixed E/D residual",
        "residual:slow_ed": "Slow E/D residual",
        "raw_bypass:current": "CURRENT RAW",
        "raw_bypass:fixed_ed": "Fixed E/D RAW",
        "raw_bypass:slow_ed": "Slow E/D RAW",
    }
    parts = label.split(":")
    if len(parts) >= 3:
        base = ":".join(parts[:2])
        suffix = " ".join(part.upper() if re.fullmatch(r"k\d+", part) else part for part in parts[2:])
        if base in mapping:
            return f"{mapping[base]} {suffix}"
    return mapping.get(label, label.replace("_", " "))


def label_key(label: str) -> tuple[int, str]:
    base = ":".join(label.split(":")[:2]) if ":" in label else label
    return (LABEL_ORDER.index(base) if base in LABEL_ORDER else len(LABEL_ORDER), label)


def color_for(label: str) -> str:
    base = ":".join(label.split(":")[:2]) if ":" in label else label
    return COLORS.get(label, COLORS.get(base, "#333333"))


def safe_output_paths(name: str, include_cost: bool) -> dict[str, Path]:
    if not SAFE_STEM.fullmatch(name):
        raise ValueError("--name must contain only letters, numbers, underscore, and hyphen")
    paths = {
        "method": PLOTS / f"{name}_method.png",
        "learning_curves": PLOTS / f"{name}_learning_curves.png",
        "accuracy": PLOTS / f"{name}_accuracy.png",
        "source_map": PLOTS / f"{name}_figure_sources.json",
    }
    if include_cost:
        paths["accuracy_cost"] = PLOTS / f"{name}_accuracy_cost.png"
        paths["neural_cost_zoom"] = PLOTS / f"{name}_neural_cost_zoom.png"
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing plot artifacts: " + ", ".join(map(str, existing)))
    return paths


def validate_report(report: dict, evaluation_path: Path) -> None:
    required = {"status", "selection", "rows", "comparisons", "aggregation"}
    missing = required - set(report)
    if missing:
        raise ValueError(f"{evaluation_path} is not an evaluate_v2.py development report; missing {sorted(missing)}")
    if report["status"] != "complete":
        raise ValueError("evaluation report must be complete")
    if not isinstance(report["comparisons"], dict) or not report["comparisons"]:
        raise ValueError("evaluation report must contain non-empty period comparisons")
    for period, comparison in report["comparisons"].items():
        mean_scores = comparison.get("mean_scores")
        if not isinstance(mean_scores, dict) or not mean_scores:
            raise ValueError(f"comparison {period} must contain non-empty mean_scores")
        for label, score in mean_scores.items():
            if "seed_loss_mean_mse" not in score or "seed_loss_mean_mae" not in score:
                raise ValueError(f"{period}/{label} must contain seed-loss mean metrics")
            if not isinstance(score.get("per_seed"), list):
                raise ValueError(f"{period}/{label} must contain per_seed records")
    if not isinstance(report.get("aggregation"), str):
        raise ValueError("evaluation aggregation metadata must be a string")


def mode_from_label(label: str) -> str | None:
    if ":" not in label:
        return None
    return label.split(":", 1)[1]


def draw_box(ax, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str, ec: str = "#333333") -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                           fc=fc, ec=ec, lw=1.2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333333"))


def plot_method(path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    draw_box(ax, (0.25, 2.45), (1.05, 0.55), "X\ncontext", "#F2F2F2")
    draw_box(ax, (1.75, 2.45), (1.05, 0.55), "E\nencoder", "#FCE4D6")
    draw_box(ax, (3.25, 2.45), (1.45, 0.55), "Frozen F0\nTSFM", "#D9EAF7")
    draw_box(ax, (5.15, 2.45), (1.05, 0.55), "D\ndecoder", "#FCE4D6")
    draw_box(ax, (7.0, 2.45), (0.7, 0.55), "+", "#F2F2F2")
    draw_box(ax, (8.25, 2.45), (1.1, 0.55), "ŷ", "#E2F0D9")
    for start, end in [((1.3, 2.72), (1.75, 2.72)), ((2.8, 2.72), (3.25, 2.72)),
                       ((4.7, 2.72), (5.15, 2.72)), ((6.2, 2.72), (7.0, 2.72)),
                       ((7.7, 2.72), (8.25, 2.72))]:
        draw_arrow(ax, start, end)

    draw_box(ax, (1.75, 1.05), (1.05, 0.55), "same E", "#FCE4D6")
    draw_box(ax, (3.25, 1.05), (1.05, 0.55), "same D", "#FCE4D6")
    draw_box(ax, (5.15, 1.05), (1.25, 0.55), "residual\nX - D(E(X))", "#FFF2CC")
    draw_box(ax, (7.0, 1.05), (0.9, 0.55), "G", "#EADCF8")
    ax.plot([2.27, 2.27], [2.45, 1.6], color="#B45F06", lw=1.0, ls=":")
    ax.plot([5.67, 3.77], [2.45, 1.6], color="#B45F06", lw=1.0, ls=":")
    ax.text(3.15, 1.83, "shared weights", ha="left", va="center", fontsize=8, color="#7F4B00")
    for start, end in [((1.3, 2.45), (1.75, 1.6)), ((2.8, 1.32), (3.25, 1.32)),
                       ((4.3, 1.32), (5.15, 1.32)), ((6.4, 1.32), (7.0, 1.32)),
                       ((7.9, 1.32), (7.35, 2.45))]:
        draw_arrow(ax, start, end)
    skip_color = "#666666"
    ax.plot([0.78, 0.78, 5.78], [2.18, 2.04, 2.04], color=skip_color, lw=1.1)
    ax.add_patch(FancyArrowPatch((5.78, 2.04), (5.78, 1.60), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color=skip_color))

    ax.text(0.25, 4.45, "Channel-compressed residual PEFT follow-up", fontsize=14, weight="bold", ha="left")
    ax.text(0.25, 4.05, "Prediction path: D(F0(E(X))) + G(X - D(E(X))). F0 is frozen in all modes.",
            fontsize=10, ha="left")

    mode_text = (
        "CURRENT: train E, D, G with the adapter schedule\n"
        "fixed_ed: freeze E and D; train only G\n"
        "slow_ed: train G at normal lr; train E and D at a slower lr\n"
        "RAW bypass comparator: D(F0(E(X))) + G(X), same compressed main path"
    )
    ax.text(0.25, 0.25, mode_text, ha="left", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB"))
    ax.text(9.7, 0.25, "Diagram only; no new scoring", ha="right", va="bottom", fontsize=8, color="#666666")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"kind": "method_diagram", "path": rel(path)}


def curve_path_for(fit: str) -> tuple[Path | None, Path | None]:
    candidates = [HERE / "runs" / fit, V1 / "runs" / fit]
    for directory in candidates:
        curve = directory / "curve.json"
        result = directory / "result.json"
        if curve.exists() and result.exists():
            return curve, result
    return None, None


def collect_curve_records(report: dict) -> tuple[list[dict], list[dict]]:
    records, missing = [], []
    seen = set()
    for row in report["rows"]:
        fit = row.get("fit")
        arm = row.get("arm")
        if not fit or arm not in {"residual", "raw_bypass"}:
            continue
        if fit in seen:
            continue
        seen.add(fit)
        curve_path, result_path = curve_path_for(fit)
        if curve_path is None or result_path is None:
            missing.append({"fit": fit, "reason": "curve.json or result.json not found"})
            continue
        curve = read_json(curve_path)
        result = read_json(result_path)
        if not isinstance(curve, list) or not curve:
            missing.append({"fit": fit, "reason": "curve.json is empty or not a list"})
            continue
        mode = row.get("ed_mode") or result.get("ed_mode", "current")
        records.append({
            "fit": fit,
            "dataset": row.get("dataset") or result.get("dataset"),
            "arm": arm,
            "mode": mode,
            "latent": row.get("latent", result.get("latent")),
            "seed": row.get("seed") if row.get("seed") is not None else result.get("seed"),
            "curve": curve,
            "curve_path": curve_path,
            "result_path": result_path,
            "selected_epoch": result.get("selected_epoch"),
            "best_val_mse": result.get("best_val_mse"),
            "max_epochs": result.get("epochs"),
            "epochs_completed": result.get("epochs_completed"),
            "is_v2": HERE in result_path.resolve().parents,
        })
    return records, missing


def plot_learning_curves(report: dict, path: Path) -> dict:
    records, missing = collect_curve_records(report)
    if not records:
        raise ValueError("no residual/raw_bypass curve records were found for learning-curve figure")
    datasets = sorted({r["dataset"] for r in records})
    arms = sorted({r["arm"] for r in records}, key=lambda x: (x != "residual", x))
    fig, axes = plt.subplots(len(arms), len(datasets), figsize=(5.5 * len(datasets), 3.4 * len(arms)),
                             squeeze=False, constrained_layout=True)
    max_epoch_seen = 0
    for ai, arm in enumerate(arms):
        for di, dataset in enumerate(datasets):
            ax = axes[ai][di]
            subset = [r for r in records if r["arm"] == arm and r["dataset"] == dataset]
            if not subset:
                ax.set_axis_off()
                continue
            for record in sorted(subset, key=lambda r: (MODE_ORDER.index(r["mode"]) if r["mode"] in MODE_ORDER else 99,
                                                        r.get("latent") or -1, r["seed"] or -1, r["fit"])):
                epochs = np.asarray([point["epoch"] for point in record["curve"]], dtype=float)
                vals = np.asarray([point["val_mse"] for point in record["curve"]], dtype=float)
                max_epoch_seen = max(max_epoch_seen, int(np.nanmax(epochs)))
                variant = record["arm"] + ":" + record["mode"]
                if record.get("latent") is not None:
                    variant += f":k{record['latent']}"
                label = f"{display_label(variant)} seed {record['seed']}"
                if not record["is_v2"]:
                    label = "v1 " + label
                style = "-" if record["is_v2"] else "--"
                color = color_for(variant)
                line, = ax.plot(epochs, vals, style, lw=1.5, color=color, alpha=0.9, label=label)
                selected_epoch = record.get("selected_epoch")
                if selected_epoch is not None:
                    selected = [p for p in record["curve"] if p["epoch"] == selected_epoch]
                    if selected:
                        ax.scatter([selected_epoch], [selected[0]["val_mse"]], marker="*", s=75,
                                   color=color, edgecolor="black", linewidth=0.4, zorder=5)
                if record.get("max_epochs"):
                    ax.axvline(record["max_epochs"], color="#DDDDDD", lw=0.8, ls=":")
            planned = [r.get("max_epochs") for r in subset if r.get("max_epochs")]
            if planned:
                max_epoch_seen = max(max_epoch_seen, max(planned))
            ax.set_title(f"{dataset} / {arm}")
            ax.set_xlabel("VAL epoch on exposed development selection axis")
            ax.set_ylabel("VAL macro MSE")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("Learning curves: selected epoch is marked with a star; dotted line marks planned max epochs",
                 fontsize=12)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"kind": "learning_curves", "path": rel(path), "curves": [
        {"fit": r["fit"], "dataset": r["dataset"], "arm": r["arm"], "mode": r["mode"],
         "latent": r.get("latent"), "seed": r["seed"], "curve": rel(r["curve_path"]), "result": rel(r["result_path"])}
        for r in records
    ], "missing_curves": missing}


def period_label(key: str) -> str:
    return key.replace("/", " / ")


def comparison_items(report: dict) -> list[tuple[str, dict]]:
    return sorted(report["comparisons"].items(), key=lambda kv: kv[0])


def plot_accuracy(report: dict, path: Path) -> dict:
    items = comparison_items(report)
    labels = sorted({label for _, comp in items for label in comp["mean_scores"]}, key=label_key)
    if not labels:
        raise ValueError("no mean_scores found in evaluation comparisons")
    fig, axes = plt.subplots(1, len(items), figsize=(5.2 * len(items), 7.2),
                             squeeze=False, constrained_layout=True)
    plotted = []
    for pi, (period, comp) in enumerate(items):
        ax = axes[0][pi]
        period_labels = [label for label in labels if label in comp["mean_scores"]]
        period_labels = sorted(period_labels, key=lambda label: comp["mean_scores"][label]["seed_loss_mean_mse"],
                               reverse=True)
        y = np.arange(len(period_labels), dtype=float)
        x_values = []
        for yi, label in enumerate(period_labels):
            entry = comp["mean_scores"].get(label)
            x_values.append(entry["seed_loss_mean_mse"])
            ax.barh(y[yi], entry["seed_loss_mean_mse"], height=0.68, color=color_for(label), alpha=0.72,
                    edgecolor="white", linewidth=0.5)
            seeds = entry.get("per_seed", [])
            jitters = np.linspace(-0.16, 0.16, max(len(seeds), 1))
            for ji, seed_row in enumerate(seeds):
                marker = "o" if seed_row.get("seed") is not None else "D"
                ax.scatter(seed_row["mse"], y[yi] + jitters[ji], marker=marker, s=22,
                           color=color_for(label), edgecolor="#222222", linewidth=0.35, zorder=4)
            plotted.append(label)
        ax.set_yticks(y)
        ax.set_yticklabels([display_label(label) for label in period_labels], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Macro MSE")
        ax.set_title(f"{period_label(period)}\nindependent x-axis range")
        ax.grid(axis="x", alpha=0.25)
        if x_values:
            xmin, xmax = min(x_values), max(x_values)
            pad = 0.04 * (xmax - xmin) if xmax > xmin else 0.01
            ax.set_xlim(max(0.0, xmin - pad), xmax + pad)
    axes[0][0].set_ylabel("Method")
    fig.suptitle("Accuracy on exposed development periods; each panel has its own x-axis range",
                 fontsize=12)
    fig.text(0.01, -0.01,
             "Bars are mean seed losses from evaluate_v2.py. Dots are individual seed or single deterministic baseline records.",
             ha="left", va="bottom", fontsize=8, color="#555555")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"kind": "accuracy", "path": rel(path), "periods": [key for key, _ in items],
            "labels": sorted(set(plotted), key=label_key), "layout": "per-period horizontal bars; independent x-axis ranges"}


def summarize_cost_rows(identifier: str, batch: int, members: list[dict]) -> dict:
    ms_values = [float(m["milliseconds_per_origin"]["median"]) for m in members]
    if all("origins_per_second" in m for m in members):
        throughput_values = [float(m["origins_per_second"]["median"]) for m in members]
    else:
        throughput_values = [1000.0 / value for value in ms_values if value > 0]
    datasets = sorted({m.get("dataset") for m in members if m.get("dataset")})
    arms = sorted({m.get("arm") for m in members if m.get("arm")})
    return {
        "id": identifier,
        "batch_origins": batch,
        "milliseconds_per_origin_median": float(np.median(ms_values)),
        "milliseconds_per_origin_q25": float(np.quantile(ms_values, 0.25)),
        "milliseconds_per_origin_q75": float(np.quantile(ms_values, 0.75)),
        "origins_per_second_median": float(np.median(throughput_values)) if throughput_values else math.nan,
        "origins_per_second_q25": float(np.quantile(throughput_values, 0.25)) if throughput_values else math.nan,
        "origins_per_second_q75": float(np.quantile(throughput_values, 0.75)) if throughput_values else math.nan,
        "dataset": datasets[0] if len(datasets) == 1 else None,
        "arm": arms[0] if len(arms) == 1 else None,
        "source_files": sorted({m["_source"] for m in members}),
        "rows": len(members),
    }


def load_cost_reports(paths: list[Path]) -> tuple[dict[str, dict], list[dict]]:
    rows = []
    sources = []
    for path in paths:
        report = read_json(path)
        if not isinstance(report, dict) or report.get("status") != "complete" or "rows" not in report:
            raise ValueError(f"cost report must be a complete JSON object with rows: {path}")
        sources.append({"path": rel(path), "sha256": digest(path), "timed_scope": report.get("timed_scope"),
                        "precision": report.get("precision")})
        for row in report["rows"]:
            item = dict(row)
            item["_source"] = rel(path)
            rows.append(item)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        if "id" not in row or "milliseconds_per_origin" not in row:
            continue
        batch = int(row.get("batch_origins", 0))
        if batch <= 0:
            continue
        grouped.setdefault((row["id"], batch), []).append(row)
    by_id: dict[str, dict] = {}
    for (ident, batch), members in grouped.items():
        summary = summarize_cost_rows(ident, batch, members)
        entry = by_id.setdefault(ident, {
            "id": ident,
            "dataset": summary["dataset"],
            "arm": summary["arm"],
            "batches": {},
            "source_files": [],
        })
        if entry["dataset"] != summary["dataset"]:
            entry["dataset"] = None
        if entry["arm"] != summary["arm"]:
            entry["arm"] = None
        entry["batches"][batch] = summary
        entry["source_files"] = sorted(set(entry["source_files"]) | set(summary["source_files"]))
    return by_id, sources


def cost_aliases(cost_by_id: dict[str, dict]) -> dict[str, dict]:
    aliases = dict(cost_by_id)
    for cost in cost_by_id.values():
        dataset, arm = cost.get("dataset"), cost.get("arm")
        if dataset and arm:
            aliases[f"{dataset}_{arm}"] = cost
            aliases[f"{dataset}/{arm}"] = cost
    return aliases


def find_cost(label: str, dataset: str, seed_row: dict, aliases: dict[str, dict]) -> dict | None:
    candidates = []
    fit = seed_row.get("fit")
    if fit:
        candidates.append(fit)
    candidates.extend([f"{dataset}_{label}", f"{dataset}/{label}", label])
    if ":" in label:
        arm, mode = label.split(":", 1)
        candidates.extend([f"{dataset}_{arm}_{mode}", f"{dataset}/{arm}:{mode}"])
    for candidate in candidates:
        if candidate in aliases:
            return aliases[candidate]
    return None


def cost_route(label: str) -> str:
    if label in {"shared_linear", "factor_linear"}:
        return "CPU linear"
    return "GPU neural"


def cost_display_label(label: str) -> str:
    return f"{display_label(label)} [{cost_route(label)}]"


def cost_batch(cost: dict, batch: int) -> dict | None:
    batches = cost.get("batches", {})
    return batches.get(batch) or batches.get(str(batch))


def add_cost_point(points: list[dict], period: str, dataset: str, label: str, seed_row: dict,
                   cost: dict, batch: int, view: str) -> None:
    summary = cost_batch(cost, batch)
    if summary is None:
        return
    if view == "batch1_latency":
        x_value = summary["milliseconds_per_origin_median"]
        x_metric = "milliseconds_per_origin_median"
    elif view == "batch4_throughput":
        x_value = summary["origins_per_second_median"]
        x_metric = "origins_per_second_median"
    else:
        raise ValueError(view)
    if not np.isfinite(x_value) or x_value <= 0:
        return
    points.append({
        "view": view,
        "period": period,
        "dataset": dataset,
        "label": label,
        "seed": seed_row.get("seed"),
        "fit": seed_row.get("fit"),
        "mse": seed_row["mse"],
        "x_value": float(x_value),
        "x_metric": x_metric,
        "batch_origins": batch,
        "cost_id": cost["id"],
        "cost_route": cost_route(label),
    })


def plot_accuracy_cost(report: dict, cost_paths: list[Path], path: Path) -> dict:
    cost_by_id, cost_sources = load_cost_reports(cost_paths)
    aliases = cost_aliases(cost_by_id)
    points = []
    views = [
        {
            "key": "batch1_latency",
            "batch": 1,
            "title": "Batch 1 latency",
            "xlabel": "Median ms/origin, batch 1 (lower is better)",
        },
        {
            "key": "batch4_throughput",
            "batch": 4,
            "title": "Batch 4 throughput",
            "xlabel": "Median origins/second, batch 4 (higher is better)",
        },
    ]
    for key, comp in comparison_items(report):
        dataset, period = key.split("/", 1)
        for label, entry in comp["mean_scores"].items():
            seed_rows = entry.get("per_seed", [])
            matched = []
            for seed_row in seed_rows:
                cost = find_cost(label, dataset, seed_row, aliases)
                if cost is not None:
                    matched.append((seed_row, cost))
                    for view in views:
                        add_cost_point(points, key, dataset, label, seed_row, cost, view["batch"], view["key"])
            if matched:
                for view in views:
                    summaries = [cost_batch(cost, view["batch"]) for _, cost in matched]
                    summaries = [summary for summary in summaries if summary is not None]
                    if not summaries:
                        continue
                    if view["key"] == "batch1_latency":
                        values = [summary["milliseconds_per_origin_median"] for summary in summaries]
                        x_metric = "milliseconds_per_origin_median"
                    else:
                        values = [summary["origins_per_second_median"] for summary in summaries]
                        x_metric = "origins_per_second_median"
                    values = [float(value) for value in values if np.isfinite(value) and value > 0]
                    if not values:
                        continue
                    points.append({
                        "view": view["key"],
                        "period": key,
                        "dataset": dataset,
                        "label": label,
                        "seed": "mean",
                        "fit": None,
                        "mse": entry["seed_loss_mean_mse"],
                        "x_value": float(np.mean(values)),
                        "x_metric": x_metric,
                        "batch_origins": view["batch"],
                        "cost_id": "mean:" + label,
                        "cost_route": cost_route(label),
                        "mean_marker": True,
                    })
    if not points:
        raise ValueError("cost report was supplied but no rows matched evaluation fits or baseline labels")
    periods = sorted({p["period"] for p in points})
    cols = len(periods)
    rows = len(views)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.1 * rows), squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ri, view in enumerate(views):
        for ci, period in enumerate(periods):
            ax = axes[ri][ci]
            ax.set_axis_on()
            subset = [p for p in points if p["period"] == period and p["view"] == view["key"]]
            if not subset:
                ax.text(0.5, 0.5, f"No matched {view['title']} cost", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#666666")
                ax.set_title(f"{view['title']} / {period_label(period)}")
                continue
            for label in sorted({p["label"] for p in subset}, key=label_key):
                lp = [p for p in subset if p["label"] == label]
                seed_points = [p for p in lp if not p.get("mean_marker")]
                mean_points = [p for p in lp if p.get("mean_marker")]
                if seed_points:
                    ax.scatter([p["x_value"] for p in seed_points], [p["mse"] for p in seed_points],
                               s=26, color=color_for(label), edgecolor="#222222", linewidth=0.35, alpha=0.9,
                               label=cost_display_label(label))
                if mean_points:
                    ax.scatter([p["x_value"] for p in mean_points], [p["mse"] for p in mean_points],
                               s=80, marker="D", color=color_for(label), edgecolor="black", linewidth=0.6)
            ax.set_xscale("log")
            ax.set_title(f"{view['title']} / {period_label(period)}")
            ax.set_xlabel(view["xlabel"])
            ax.set_ylabel("Macro MSE (seed loss or mean seed loss)")
            ax.grid(True, alpha=0.25, which="both")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Accuracy-cost view; neural/F0/LoRA/adapters on GPU, linear baselines on CPU",
                 fontsize=12)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"kind": "accuracy_cost", "path": rel(path), "cost_sources": cost_sources,
            "views": views, "points": points}


def plot_neural_cost_zoom(points: list[dict], path: Path) -> dict:
    neural = [p for p in points if p.get("cost_route") == "GPU neural"]
    if not neural:
        return {"kind": "neural_cost_zoom", "path": None, "points": []}
    periods = sorted({p["period"] for p in neural})
    views = [
        {
            "key": "batch1_latency",
            "title": "GPU neural batch 1 latency",
            "xlabel": "Median ms/origin, batch 1 (linear scale; lower is better)",
        },
        {
            "key": "batch4_throughput",
            "title": "GPU neural batch 4 throughput",
            "xlabel": "Median origins/second, batch 4 (linear scale; higher is better)",
        },
    ]
    fig, axes = plt.subplots(len(views), len(periods), figsize=(5.2 * len(periods), 3.8 * len(views)),
                             squeeze=False, constrained_layout=True)
    for ri, view in enumerate(views):
        for ci, period in enumerate(periods):
            ax = axes[ri][ci]
            subset = [p for p in neural if p["period"] == period and p["view"] == view["key"]]
            if not subset:
                ax.set_axis_off()
                continue
            for label in sorted({p["label"] for p in subset}, key=label_key):
                lp = [p for p in subset if p["label"] == label]
                seed_points = [p for p in lp if not p.get("mean_marker")]
                mean_points = [p for p in lp if p.get("mean_marker")]
                if seed_points:
                    ax.scatter([p["x_value"] for p in seed_points], [p["mse"] for p in seed_points],
                               s=24, color=color_for(label), edgecolor="#222222", linewidth=0.35,
                               alpha=0.55, label=display_label(label))
                if mean_points:
                    ax.scatter([p["x_value"] for p in mean_points], [p["mse"] for p in mean_points],
                               s=96, marker="D", color=color_for(label), edgecolor="black", linewidth=0.7)
            xs = np.array([p["x_value"] for p in subset], dtype=float)
            if xs.size and xs.max() > xs.min():
                pad = 0.08 * (xs.max() - xs.min())
                ax.set_xlim(xs.min() - pad, xs.max() + pad)
            ax.set_title(f"{view['title']} / {period_label(period)}")
            ax.set_xlabel(view["xlabel"])
            ax.set_ylabel("Macro MSE (seed loss or mean seed loss)")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Neural-only cost zoom: GPU measurements; diamonds are seed-loss means",
                 fontsize=12)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"kind": "neural_cost_zoom", "path": rel(path), "points": neural}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, help="Completed evaluate_v2.py *_development.json path")
    parser.add_argument("--cost", action="append", default=[], help="Optional completed cost report JSON; repeatable")
    parser.add_argument("--name", required=True, help="Safe artifact stem for new plot files")
    args = parser.parse_args()

    evaluation_path = Path(args.evaluation).resolve()
    cost_paths = [Path(path).resolve() for path in args.cost]
    report = read_json(evaluation_path)
    if not isinstance(report, dict):
        raise ValueError("evaluation report must be a JSON object")
    validate_report(report, evaluation_path)

    PLOTS.mkdir(parents=True, exist_ok=True)
    paths = safe_output_paths(args.name, include_cost=bool(cost_paths))

    sources = {
        "plot_results.py": {"path": rel(Path(__file__)), "sha256": digest(Path(__file__))},
        "evaluation": {"path": rel(evaluation_path), "sha256": digest(evaluation_path)},
        "cost": [{"path": rel(path), "sha256": digest(path)} for path in cost_paths],
    }
    outputs = {}
    outputs["method"] = plot_method(paths["method"])
    outputs["learning_curves"] = plot_learning_curves(report, paths["learning_curves"])
    outputs["accuracy"] = plot_accuracy(report, paths["accuracy"])
    if cost_paths:
        outputs["accuracy_cost"] = plot_accuracy_cost(report, cost_paths, paths["accuracy_cost"])
        outputs["neural_cost_zoom"] = plot_neural_cost_zoom(outputs["accuracy_cost"]["points"],
                                                            paths["neural_cost_zoom"])
    for entry in outputs.values():
        entry["sha256"] = digest(ROOT / entry["path"] if not Path(entry["path"]).is_absolute() else Path(entry["path"]))

    mapping = {
        "status": "complete",
        "name": args.name,
        "sources": sources,
        "outputs": outputs,
        "notes": [
            "Figures are derived only from completed JSON reports and saved learning-curve logs.",
            "Accuracy bars use evaluate_v2.py mean of individual seed losses, not forecast ensembles.",
            "No cost figure is created unless --cost is supplied.",
            "If a raw time-series overlap forecast-mean figure is used alongside these plots, that overlap mean is a visualization-only summary and not a scoring ensemble.",
        ],
        "caption_notes": {
            "raw_overlap_forecast_mean": (
                "Overlapping forecast means in raw time-series diagnostics are for visual inspection only; "
                "reported scores use the evaluator's recorded seed-loss metrics, not an ensemble forecast."
            )
        },
    }
    write_json(paths["source_map"], mapping)
    print(json.dumps({"status": "complete", "outputs": {k: v["path"] for k, v in outputs.items()},
                      "source_map": rel(paths["source_map"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
