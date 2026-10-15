"""Build the one campaign report after completed, pre-locked scoring/cost.

Preparation/import does not read scores or prediction archives. ``main`` first
requires SCORE_COMPLETE, COST_LOCK, FIRST_C_SCORE, and SELECTION_LOCK receipts.
Selection helpers use locked B metrics only; C is used for reporting and for
explicitly retrospective failure-case descriptions, never new fitting.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
MAIN_DATASETS = ("beijing", "jena")
DATASETS = MAIN_DATASETS + ("robin",)
CHANNELS = {"beijing": 11, "jena": 21, "robin": 17}
SMALL_K = {"beijing": 2, "jena": 5, "robin": 4}
COLORS = {"Raw": "#343a40", "B0": "#7f8c8d", "B1": "#e69f00", "B2": "#0072b2",
          "B3": "#cc79a7", "B4": "#009e73", "G-only": "#d55e00"}
FIGURES = (
    ("01_time_transforms", "단순 시간 변환: 같은 미래·cohort의 Raw 대비 오차"),
    ("02_temporal_rank", "시간 앞단과 C/K: 두 seed의 평균·범위"),
    ("03_matched_residual", "같은 잔차 G에서 앞단의 추가 가치"),
    ("04_learning_time_refit", "A/B 곡선과 두 번째 시간 경계의 전체 재적합"),
    ("05_error_time_memory", "실제 전체 경로의 정확도·시간·메모리"),
    ("06_forecast_cases", "사전 고정 origin과 사후 실패 사례"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    csv.field_size_limit(16*1024*1024)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, default=None):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def fmt(value, digits=6):
    value = number(value)
    return "NA" if value is None else f"{value:.{digits}f}"


def scientific(value, digits=3):
    value = number(value)
    return "NA" if value is None else f"{value:.{digits}e}"


def code_table(headers, rows):
    text = [[str(value) for value in headers]] + [[str(value) for value in row] for row in rows]
    widths = [max(len(row[index]) for row in text) for index in range(len(headers))]
    return "```text\n" + "\n".join("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in text) + "\n```"


def load_bundle():
    # These checks precede SCORES.csv, score JSONs, and all prediction NPZ reads.
    completion = read_json(HERE / "SCORE_COMPLETE.json")
    cost_lock = read_json(HERE / "COST_LOCK.json")
    selection = read_json(HERE / "SELECTION_LOCK.json")
    first_c = read_json(HERE / "FIRST_C_SCORE.json")
    if completion.get("status") != "COMPLETE_VALID" or not completion.get("F_unchanged"):
        raise RuntimeError("REPORT_REQUIRES_VALID_SCORE_COMPLETE_AND_F_RECEIPT")
    if cost_lock.get("status") != "COMPLETE_VALID":
        raise RuntimeError("REPORT_REQUIRES_VALID_COST_LOCK")
    if selection.get("status") != "ALL_SELECTIONS_LOCKED_BEFORE_C" or selection.get("C_scored"):
        raise RuntimeError("REPORT_REQUIRES_PRE_C_SELECTION_LOCK")
    if cost_lock.get("selection_lock") != sha(HERE / "SELECTION_LOCK.json"):
        raise RuntimeError("COST_SELECTION_HASH_MISMATCH")
    if first_c.get("selection_lock") != sha(HERE / "SELECTION_LOCK.json") or first_c.get("cost_lock") != sha(HERE / "COST_LOCK.json"):
        raise RuntimeError("FIRST_C_HASH_MISMATCH")
    if not (selection["time_unix"] <= cost_lock["time_unix"] <= first_c["time_unix"] <= completion["time_unix"]):
        raise RuntimeError("SELECTION_COST_C_TIMESTAMP_ORDER_MISMATCH")
    analysis = read_json(HERE / "ANALYSIS_COMPLETE.json")
    if analysis.get("status") != "COMPLETE_VALID" or analysis.get("F_loaded_or_called") is not False:
        raise RuntimeError("REPORT_REQUIRES_VALID_POST_SCORE_ANALYSIS")
    if analysis.get("scores_sha256") != sha(HERE / "SCORES.csv"):
        raise RuntimeError("ANALYSIS_SCORE_HASH_MISMATCH")
    for key, filename in (("score_complete_sha256","SCORE_COMPLETE.json"),
                          ("selection_lock_sha256","SELECTION_LOCK.json"),
                          ("cost_lock_sha256","COST_LOCK.json")):
        if analysis.get(key) != sha(HERE / filename):
            raise RuntimeError("ANALYSIS_RECEIPT_HASH_MISMATCH:" + key)
    for filename, digest in analysis.get("output_sha256", {}).items():
        if (HERE / filename).resolve().parent != HERE or sha(HERE / filename) != digest:
            raise RuntimeError("ANALYSIS_OUTPUT_HASH_MISMATCH:" + filename)
    scores = read_csv(HERE / "SCORES.csv")
    costs = read_csv(HERE / "COST.csv")
    audit_manifest_path = HERE / "AUDIT_LATEST.json"
    audit_manifest = read_json(audit_manifest_path) if audit_manifest_path.is_file() else None
    if audit_manifest is not None:
        if audit_manifest.get("status") != "PASS":
            raise RuntimeError("AUDIT_MANIFEST_NOT_PASS")
        audit_name, run_index_name = audit_manifest["audit_file"],audit_manifest["run_index_file"]
        for filename in (audit_name,run_index_name,*audit_manifest.get("prior_files",{})):
            if Path(filename).name != filename or (HERE/filename).resolve().parent != HERE:
                raise RuntimeError("AUDIT_MANIFEST_PATH_OUTSIDE_CAMPAIGN")
        audit_path, run_index_path = HERE/audit_name,HERE/run_index_name
        if sha(audit_path) != audit_manifest["audit_sha256"] or sha(run_index_path) != audit_manifest["run_index_sha256"]:
            raise RuntimeError("AUDIT_MANIFEST_SELECTED_HASH_MISMATCH")
        for filename,digest in audit_manifest.get("prior_files",{}).items():
            if sha(HERE/filename) != digest:
                raise RuntimeError("AUDIT_MANIFEST_PRIOR_FILE_CHANGED:"+filename)
    else:
        audit_path, run_index_path = HERE/"AUDIT.json",HERE/"RUN_INDEX.csv"
    run_index = read_csv(run_index_path)
    audit = read_json(audit_path)
    if audit_manifest is not None and audit.get("status") != "PASS":
        raise RuntimeError("AUDIT_MANIFEST_BOUND_AUDIT_NOT_PASS")
    if audit.get("run_index_sha256") != sha(run_index_path):
        raise RuntimeError("AUDIT_RUN_INDEX_HASH_MISMATCH")
    for filename, digest in audit.get("input_sha256", {}).items():
        if sha(HERE / filename) != digest:
            raise RuntimeError("AUDIT_INPUT_HASH_MISMATCH:" + filename)
    if len(scores) != completion["rows"]:
        raise RuntimeError("SCORE_ROW_COUNT_MISMATCH")
    for row in scores:
        if row["split"] not in ("B", "C") or number(row["mse"]) is None or number(row["mae"]) is None:
            raise RuntimeError("INVALID_SCORE_ROW")
        if int(row["n_observed"]) <= 0:
            raise RuntimeError("EMPTY_SCORE_TASK")
    routes = {row["id"]: row for row in selection["routes"]}
    fits, ridge_fits, fit_receipts = {}, {}, []
    for path in sorted((HERE / "fits").glob("*.json")):
        fit = read_json(path)
        if "learning_curve" in fit:
            fit_receipts.append(fit)
            old = fits.get(fit["id"])
            if old is None or fit["updates"] > old["updates"]:
                fits[fit["id"]] = fit
        elif "fit_A_origins" in fit and "candidate_diagnostics" in fit:
            ridge_fits[fit["id"]] = fit
    return {"scores": scores, "costs": costs, "run_index": run_index, "routes": routes,
            "fits": fits, "ridge_fits": ridge_fits, "fit_receipts": fit_receipts, "completion": completion, "cost_lock": cost_lock,
            "selection": selection, "first_c": first_c, "analysis": analysis, "audit": audit,
            "audit_path": audit_path,"run_index_path": run_index_path,"audit_manifest": audit_manifest,"case_receipts": []}


def score_row(bundle, identifier, split="C"):
    found = [row for row in bundle["scores"] if row["id"] == identifier and row["split"] == split]
    if len(found) != 1:
        raise ValueError(f"Expected one score for {identifier}/{split}, found {len(found)}")
    return found[0]


def maybe_score(bundle, identifier, split="C"):
    found = [row for row in bundle["scores"] if row["id"] == identifier and row["split"] == split]
    return found[0] if len(found) == 1 else None


def paired(left, right):
    fields = ("dataset", "fold", "split", "cohort", "horizon", "n_observed", "n_origins")
    if any(str(left[key]) != str(right[key]) for key in fields):
        raise ValueError(f"Unmatched comparison: {left['id']} / {right['id']}")


def raw_for(bundle, row):
    # The long-lookback routes retain a freshly recomputed Raw512 on their exact origins.
    identifier = row["id"] + "_matched_RawL512"
    direct = maybe_score(bundle, identifier, row["split"])
    if direct is not None:
        paired(row, direct)
        return direct
    found = [item for item in bundle["scores"] if item["method"] == "RAW" and int(item["length"]) == 512
             and all(item[key] == row[key] for key in ("dataset", "fold", "split", "cohort"))]
    if not found:
        raise ValueError(f"No matched Raw512 for {row['id']}")
    found.sort(key=lambda item: (not item["id"].endswith("_A0"), item["id"]))
    paired(row, found[0])
    return found[0]


def ratio(bundle, row):
    return float(row["mse"]) / float(raw_for(bundle, row)["mse"])


def b_value(bundle, spec):
    if "selected_B" in spec:
        return float(spec["selected_B"]["mse"])
    return float(score_row(bundle, spec["id"], "B")["mse"])


def core_routes(bundle, dataset, kind=None, rank=None, fold="primary"):
    eligible_blocks = ("R2", "R5_STRESS") if fold == "primary" else ("R5_TIME_REFIT",)
    return [spec for spec in bundle["routes"].values()
            if spec["dataset"] == dataset and spec.get("fold", "primary") == fold and spec["type"] == "ADAPTER"
            and spec.get("block", "R2") in eligible_blocks and not spec.get("front_kwargs")
            and (kind is None or spec.get("kind") == kind) and (rank is None or spec.get("r") == rank)]


def selected_adapter(bundle, dataset, kind, rank, fold="primary"):
    candidates = core_routes(bundle, dataset, kind, rank, fold)
    if not candidates:
        return None
    return min(candidates, key=lambda spec: (b_value(bundle, spec), spec.get("seed", 0), spec["id"]))


def chosen_family(bundle, dataset):
    candidates = [selected_adapter(bundle, dataset, kind, SMALL_K[dataset]) for kind in ("B2", "B4")]
    candidates = [spec for spec in candidates if spec is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda spec: (b_value(bundle, spec), spec["id"]))


def label(spec):
    typ = spec["type"]
    if typ == "ADAPTER":
        return f"{spec['kind']}/r{spec['r']}" + (" ablation" if spec.get("front_kwargs") else "")
    if typ == "G":
        base = spec["base"]
        return "G-only" if base["type"] == "CURRENT" else label(base) + "+G"
    if typ == "RAW":
        if spec.get("row_microbatch"):
            return f"Raw micro{spec['row_microbatch']}"
        return f"Raw L{spec.get('length', 512)}"
    if typ == "CURRENT":
        return "Current"
    if typ == "R1":
        return spec["variant"]
    if typ == "DIRECT":
        return f"Direct {spec['kind']}"
    if typ == "SWAP":
        return label(spec["base"]) + " native"
    if typ == "RIDGE":
        return "Self ridge"
    return "LoRA qv8" if typ == "LORA" else typ


def raw_route(bundle, dataset, fold="primary"):
    specs = [spec for spec in bundle["routes"].values() if spec["dataset"] == dataset
             and spec.get("fold", "primary") == fold and spec["type"] == "RAW" and spec.get("length", 512) == 512]
    if not specs:
        return None
    return min(specs, key=lambda spec: (spec.get("variant") != "A0", spec["id"]))


def g_routes(bundle, dataset, fold="primary"):
    return [spec for spec in bundle["routes"].values() if spec["dataset"] == dataset
            and spec.get("fold", "primary") == fold and spec["type"] == "G"]


def g_for(bundle, dataset, kind, fold="primary"):
    candidates = []
    for spec in g_routes(bundle, dataset, fold):
        base = spec["base"]
        matches = (kind == "Raw" and base["type"] == "RAW") or (kind == "G-only" and base["type"] == "CURRENT")
        matches |= base["type"] == "ADAPTER" and base.get("kind") == kind
        if matches:
            candidates.append(spec)
    return min(candidates, key=lambda spec: b_value(bundle, spec)) if candidates else None


def save_figure(fig, basename):
    for suffix in ("png", "svg"):
        path = HERE / "figures" / f"{basename}.{suffix}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def finish_axis(axis, title, ylabel=None):
    axis.set_title(title, fontsize=11, weight="bold")
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.18)


def figure_r1(bundle):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        values = []
        names = []
        for index in range(14):
            identifier = f"primary_{dataset}_A{index}"
            row = maybe_score(bundle, identifier)
            if row is None:
                continue
            names.append(f"A{index}")
            values.append(ratio(bundle, row))
        color = ["#009e73" if value < 1 else "#d55e00" for value in values]
        axis.bar(np.arange(len(values)), values, color=color, alpha=.8, width=.75)
        axis.axhline(1, color=COLORS["Raw"], linewidth=1, linestyle="--")
        axis.set_xticks(np.arange(len(values)), names, rotation=55, fontsize=9)
        finish_axis(axis, dataset.title(), "C MSE / matched Raw512 MSE")
        axis.set_ylim(0, max([1.2]+values)*1.08)
        axis.text(.02, .97, "All predeclared transforms\nL1024/2048: same-origin Raw512", transform=axis.transAxes,
                  ha="left", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=.75, edgecolor="none"))
    fig.suptitle("R1: Original-grid future errors after fixed past transformations", weight="bold")
    save_figure(fig, FIGURES[0][0])


def figure_temporal_rank(bundle):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5), constrained_layout=True)
    for axis, dataset in zip(axes, MAIN_DATASETS):
        for rank_index, rank in enumerate((CHANNELS[dataset], SMALL_K[dataset])):
            x_values, means, minima, maxima = [], [], [], []
            for kind_index, kind in enumerate(("B0", "B1", "B2", "B3", "B4")):
                rows = [score_row(bundle, spec["id"]) for spec in core_routes(bundle, dataset, kind, rank)]
                if not rows:
                    continue
                values = [ratio(bundle, row) for row in rows]
                x_values.append(kind_index + (rank_index-.5)*.34)
                means.append(float(np.mean(values)))
                minima.append(min(values))
                maxima.append(max(values))
            axis.errorbar(x_values, means, yerr=[np.array(means)-minima, np.array(maxima)-means], fmt="o-",
                          capsize=4, color=("#0072b2", "#d55e00")[rank_index],
                          label=f"r={'C' if rank_index==0 else 'K'}={rank}: seed mean/range", linewidth=1.7)
        axis.axhline(1, color=COLORS["Raw"], linestyle="--", linewidth=1)
        axis.set_xticks(range(5), ("B0", "B1", "B2", "B3", "B4"))
        finish_axis(axis, dataset.title(), "C MSE / matched Raw MSE")
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("R2: Temporal front x latent width (ranges are not confidence intervals)", weight="bold")
    save_figure(fig, FIGURES[1][0])


def figure_residual(bundle):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, dataset in zip(axes, MAIN_DATASETS):
        names, values = [], []
        raw = raw_route(bundle, dataset)
        if raw:
            names.append("Raw"); values.append(1.)
        for kind in ("G-only", "Raw", "B0", "B2", "B4"):
            spec = g_for(bundle, dataset, kind)
            if spec:
                names.append("G-only" if kind == "G-only" else kind+"+G")
                values.append(ratio(bundle, score_row(bundle, spec["id"])))
        axis.bar(range(len(values)), values, color=[COLORS["Raw"]]+["#e69f00", "#7f8c8d", "#cc79a7", "#0072b2", "#009e73"][:len(values)-1])
        axis.axhline(1, color="black", linewidth=.9, linestyle="--")
        axis.set_xticks(range(len(values)), names, rotation=35, ha="right")
        finish_axis(axis, dataset.title(), "C MSE / matched Raw MSE")
        axis.text(.02, .97, "Same past-C innovation input / G class\nOne primary G seed; B-selected checkpoints",
                  transform=axis.transAxes, ha="left", va="top", fontsize=8,
                  bbox=dict(facecolor="white", alpha=.75, edgecolor="none"))
    fig.suptitle("R3: Does the frozen forecasting branch add value beyond the same G?", weight="bold")
    save_figure(fig, FIGURES[2][0])


def figure_learning(bundle):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for index, dataset in enumerate(MAIN_DATASETS):
        axis = axes[index, 0]
        for kind in ("B0", "B2", "B4"):
            spec = selected_adapter(bundle, dataset, kind, SMALL_K[dataset])
            if spec is None:
                continue
            fit = bundle["fits"].get(spec["id"])
            if fit is None:
                continue
            checkpoints = fit["checkpoints"]
            updates = [item["update"] for item in checkpoints]
            axis.plot(updates, [item["B"]["mse"] for item in checkpoints], "o-", color=COLORS[kind], label=f"{kind}: B")
            axis.plot(updates, [item["A"]["mse"] for item in checkpoints], "--", color=COLORS[kind], alpha=.6, label=f"{kind}: A probe")
            selected = next(item for item in checkpoints if item["update"] == fit["selected_update"])
            axis.scatter(selected["update"], selected["B"]["mse"], s=90, facecolors="none", edgecolors=COLORS[kind], linewidths=1.8)
        axis.set_xlabel("Optimizer updates")
        finish_axis(axis, dataset.title()+": checkpoint selection", "Normalized macro MSE")
        axis.legend(fontsize=7, ncol=2)
        axis = axes[index, 1]
        for fold_index, fold in enumerate(("primary", "secondary")):
            labels, values = ["Raw", "B0K", "B2K", "B4K"], [1.]
            for kind in ("B0", "B2", "B4"):
                spec = selected_adapter(bundle, dataset, kind, SMALL_K[dataset], fold)
                values.append(ratio(bundle, score_row(bundle, spec["id"])) if spec else np.nan)
            axis.bar(np.arange(4)+(fold_index-.5)*.34, values, width=.32,
                     color=("#0072b2", "#e69f00")[fold_index], alpha=.85,
                     label=("A60/B20/C20", "A50/B20/C30")[fold_index])
        axis.axhline(1, color="black", linewidth=.8, linestyle="--")
        axis.set_xticks(range(4), labels)
        finish_axis(axis, dataset.title()+": full upstream refit", "C MSE / same-fold Raw MSE")
        axis.legend(fontsize=8)
        axis.text(.02, .97, "Different C periods; conditional repeats\nFresh A statistics/PCA/P/E/D/G",
                  transform=axis.transAxes, va="top", fontsize=8,
                  bbox=dict(facecolor="white", alpha=.75, edgecolor="none"))
    fig.suptitle("R5: Learning diagnostics and a predeclared second time cutoff", weight="bold")
    save_figure(fig, FIGURES[3][0])


def selected_cost_rows(bundle, dataset, batch=1, scope="general"):
    candidates = {}
    for cost in bundle["costs"]:
        if cost.get("dataset") != dataset or int(float(cost.get("batch", 0))) != batch or cost.get("scope") != scope:
            continue
        row = maybe_score(bundle, cost["id"])
        if row is None or row["fold"] != "primary" or int(row["length"]) != 512:
            continue
        spec = bundle["routes"].get(cost["id"])
        if spec is None:
            continue
        # Keep a readable set of predeclared families; every full row stays in COST.csv.
        if spec["type"] == "ADAPTER":
            if spec.get("kind") not in ("B0", "B2", "B4") or spec.get("r") != SMALL_K[dataset]:
                continue
        if spec["type"] == "CURRENT" or spec["type"] == "R1":
            continue
        key = label(spec)
        old = candidates.get(key)
        if old is None or b_value(bundle, spec) < b_value(bundle, old[0]):
            candidates[key] = (spec, row, cost)
    return list(candidates.values())


def figure_cost(bundle):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for index, dataset in enumerate(MAIN_DATASETS):
        for column, field in enumerate(("whole_seconds_median", "peak_allocated_bytes_median")):
            axis = axes[index, column]
            scope = "general" if column == 0 else "exclusive"
            triples = selected_cost_rows(bundle, dataset, 1, scope)
            for n, (spec, row, cost) in enumerate(triples):
                x = number(cost.get(field))
                if x is None:
                    continue
                x = x*1000 if column == 0 else x/2**20
                y = ratio(bundle, row)
                color = plt.get_cmap("tab20")(n % 20)
                axis.scatter(x, y, color=color, s=48, edgecolor="white", linewidth=.5)
                axis.annotate(label(spec), (x, y), xytext=(4, 3+(n % 2)*7), textcoords="offset points", fontsize=6.5)
            axis.axhline(1, color="black", linestyle="--", linewidth=.8)
            axis.set_xlabel("Balanced general-session median latency (ms), batch 1" if column == 0 else "Exclusive-process peak GPU allocated (MiB), batch 1")
            finish_axis(axis, dataset.title()+(" / general latency" if column == 0 else " / exclusive memory"), "C MSE / matched Raw MSE")
    fig.suptitle("Accuracy-cost tradeoffs: general latency and exclusive memory are distinct scopes", weight="bold")
    save_figure(fig, FIGURES[4][0])


def prediction_arrays(identifier, split="C"):
    path = CACHE / "predictions" / f"{identifier}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[f"{split}_{name}"].copy() for name in ("pred", "target", "mask", "origins")}


def window_macro(prediction, target, mask):
    count = mask.sum(1)
    sse = np.where(mask, (prediction-target)**2, 0).sum(1)
    per_channel = np.divide(sse, count, out=np.full_like(sse, np.nan, dtype=float), where=count > 0)
    return np.nanmean(per_channel, axis=1)


def figure_cases(bundle):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for dataset_index, dataset in enumerate(MAIN_DATASETS):
        spec = chosen_family(bundle, dataset)
        if spec is None:
            raise RuntimeError(f"No locked small-K B2/B4 case route for {dataset}")
        row = score_row(bundle, spec["id"])
        reference = raw_for(bundle, row)
        predicted, raw = prediction_arrays(spec["id"]), prediction_arrays(reference["id"])
        for key in ("origins", "mask", "target"):
            if not np.array_equal(predicted[key], raw[key], equal_nan=True):
                raise ValueError("Prediction case archives are not exactly paired")
        error = window_macro(predicted["pred"], predicted["target"], predicted["mask"])
        raw_error = window_macro(raw["pred"], raw["target"], raw["mask"])
        finite = np.flatnonzero(np.isfinite(error-raw_error))
        if not len(finite):
            raise ValueError("No observed finite origin for descriptive forecast cases")
        order = finite[np.argsort((error-raw_error)[finite], kind="stable")]
        selections = ((0, 0, "Predeclared first C / channel 0"),
                      (int(order[len(order)//2]), 0, "Post-hoc median error change / channel 0"))
        worst = int(order[-1])
        valid = predicted["mask"][worst]
        counts = valid.sum(0)
        delta = np.where(valid, (predicted["pred"][worst]-predicted["target"][worst])**2
                         - (raw["pred"][worst]-raw["target"][worst])**2, 0).sum(0)/np.maximum(counts, 1)
        worst_channel = int(np.argmax(np.where(counts > 0, delta, -np.inf)))
        worst_reason = "Post-hoc largest worsening / worst channel" if error[worst] > raw_error[worst] else "Post-hoc least improvement / worst channel"
        selections += ((worst, worst_channel, worst_reason),)
        for column, (origin_index, channel, reason) in enumerate(selections):
            axis = axes[dataset_index, column]
            mask = predicted["mask"][origin_index, :, channel]
            target = np.where(mask, predicted["target"][origin_index, :, channel], np.nan)
            axis.plot(np.arange(1, 49), target, color="black", linewidth=1.7, label="Observed target")
            axis.plot(np.arange(1, 49), raw["pred"][origin_index, :, channel], color=COLORS["Raw"], linestyle="--", label="Raw")
            axis.plot(np.arange(1, 49), predicted["pred"][origin_index, :, channel], color=COLORS[spec["kind"]], label=label(spec))
            axis.set_xlabel("Original-grid forecast horizon")
            finish_axis(axis, f"{dataset.title()}: origin {predicted['origins'][origin_index]}, C{channel}", "A-standardized value")
            axis.text(.02, .97, reason, transform=axis.transAxes, fontsize=7, ha="left", va="top",
                      bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
            if dataset_index == 0 and column == 0:
                axis.legend(fontsize=8, loc="best")
            bundle["case_receipts"].append({"dataset": dataset, "id": spec["id"], "raw_id": reference["id"],
                "origin": int(predicted["origins"][origin_index]), "channel": channel, "reason": reason,
                "origin_macro_mse_change": float(error[origin_index]-raw_error[origin_index]),
                "selection": "first origin/channel predeclared; median/worst are descriptive post-hoc diagnostics, no retuning"})
    fig.suptitle("Forecast cases: fixed first origin, representative change, and explicit failure", weight="bold")
    save_figure(fig, FIGURES[5][0])


def effect(left, right):
    paired(left, right)
    delta = (float(left["mse"])/float(right["mse"])-1)*100
    verb = "작다" if delta < 0 else "크다" if delta > 0 else "같다"
    return f"MSE {fmt(left['mse'])} 대 {fmt(right['mse'])}, 상대 차이 {delta:+.2f}%({verb})"


def finding_lines(bundle):
    lines = []
    for dataset in MAIN_DATASETS:
        raw = score_row(bundle, raw_route(bundle, dataset)["id"])
        r1_specs = [spec for spec in bundle["routes"].values() if spec["dataset"] == dataset and spec.get("block") == "R1"
                    and spec.get("length", 512) == 512]
        simple = min(r1_specs, key=lambda spec: (b_value(bundle, spec), spec["id"]))
        simple_row = score_row(bundle, simple["id"])
        lines.append(f"**{dataset.title()} — 단순 시간 변환으로 충분한가:** B에서 선정된 {simple.get('variant','Raw')}는 C에서 {effect(simple_row, raw)}. "
                     "이는 해당 고정 설정의 결과이며 모든 평활·재표본화의 최적성을 증명하지 않는다.")
        candidate = chosen_family(bundle, dataset)
        b0 = selected_adapter(bundle, dataset, "B0", SMALL_K[dataset])
        if candidate and b0:
            row, zero = score_row(bundle, candidate["id"]), score_row(bundle, b0["id"])
            lines.append(f"**학습형 P의 추가 값:** B로 선택한 {label(candidate)}와 같은 K의 B0는 C에서 {effect(row, zero)}. "
                         f"선택 seed는 {candidate.get('seed')}이며 다른 seed도 아래 표·그림에 전부 남긴다. "
                         f"훈련 상태는 `{candidate.get('training_status','NOT_RECORDED')}`다. 수렴 미확정 경로를 구조적 불가능성으로 바꾸지 않는다.")
        for kind in ("B2", "B4"):
            full = selected_adapter(bundle, dataset, kind, CHANNELS[dataset])
            small = selected_adapter(bundle, dataset, kind, SMALL_K[dataset])
            if full and small:
                lines.append(f"**{kind}의 C/K:** K{SMALL_K[dataset]} 대 C{CHANNELS[dataset]}는 {effect(score_row(bundle, small['id']),score_row(bundle, full['id']))}. "
                             f"각각 B로 선택한 seed는 K={small.get('seed')}, C={full.get('seed')}다. "
                             "C/K를 바꾸면 F stream 수뿐 아니라 E/D의 파라미터 수와 복원 공간도 함께 달라진다. "
                             "공간 제약·용량·최적화·예측기와 D의 결합이 함께 달라지는 진단이므로 정보 압축 단독의 인과 효과로 확정하지 않는다.")
        plain_g, raw_g, zero_g = (g_for(bundle, dataset, key) for key in ("G-only", "Raw", "B0"))
        for kind in ("B2", "B4"):
            combined = g_for(bundle, dataset, kind)
            if combined and plain_g and raw_g and zero_g:
                row = score_row(bundle, combined["id"])
                lines.append(f"**{kind}+G의 남는 값:** G-only 대비 {effect(row,score_row(bundle,plain_g['id']))}; "
                             f"Raw+G 대비 {effect(row,score_row(bundle,raw_g['id']))}; "
                             f"B0K+G 대비 {effect(row,score_row(bundle,zero_g['id']))}. "
                             "G는 동일 과거 C innovation·클래스·초기화 원칙이며 주 fold 한 seed의 탐색 비교다.")
        alternatives = [spec for spec in bundle["routes"].values() if spec["dataset"] == dataset
                        and spec.get("fold","primary") == "primary" and spec["type"] in ("SWAP","DIRECT","RIDGE","LORA")]
        if candidate:
            lines.append(f"**Bolt를 유지할 근거:** 고정 {label(candidate)}의 C MSE는 {fmt(score_row(bundle,candidate['id'])['mse'])}다. "
                         + " / ".join(f"{label(spec)} {fmt(score_row(bundle,spec['id'])['mse'])}" for spec in alternatives)
                         + ". 이 비교에서 native는 같은 P/E/D를 고정한 actual-Y 적합이고 direct는 별도 전체 예측기다. "
                         "단, native/ridge는 모든 합법 A origin을 쓰고 신경망은 같은 A 범위의 minibatch prefix이므로 실제 노출·추정법까지 같지는 않다. "
                         "Bolt의 유지 판단은 이 원수치·노출표·전체 비용을 함께 보며, 표현의 분포 부적합이나 디코더 오류 중 하나를 자동 확정하지 않는다.")
        secondary = selected_adapter(bundle, dataset, candidate["kind"], SMALL_K[dataset], "secondary") if candidate else None
        if secondary:
            row = score_row(bundle, secondary["id"])
            lines.append(f"**시간 재적합과 불확실성:** 두 번째 경계에서 전체 upstream을 A부터 다시 학습한 {label(secondary)}는 "
                         f"{effect(row,raw_for(bundle,row))}. 첫 fold와 C 기간이 다르며 같은 시계열의 조건부 반복이다. "
                         "어느 fold도 과거 이력 전체에서 미노출인 독립 자료가 아니다.")
    return lines


def primary_table(bundle, types=None):
    rows = []
    for item in bundle["scores"]:
        if item["split"] != "C" or item["fold"] != "primary" or item["id"] not in bundle["routes"]:
            continue
        spec = bundle["routes"][item["id"]]
        if types is not None and spec["type"] not in types:
            continue
        if spec["type"] == "ADAPTER" and spec.get("block", "R2") not in ("R2", "R5_STRESS"):
            continue
        fit = bundle["fits"].get(spec["id"], {})
        lr = fit.get("learning_rate", spec.get("configured_lr"))
        rows.append([item["dataset"], label(spec), item["r"], item["seed"], fmt(lr,5), item["updates"],
                     item["selected_update"], fmt(item["mse"]), fmt(item["mae"]), fmt(ratio(bundle,item),4), item["status"], item["training_status"]])
    rows.sort(key=lambda row: (row[0], row[1], int(row[3])))
    return code_table(("dataset","method","r","seed","lr","updates","CP","MSE","MAE","RawRatio","score_status","training_status"), rows)


def horizon_table(bundle):
    output = []
    for dataset in MAIN_DATASETS:
        chosen = [raw_route(bundle,dataset)] + [selected_adapter(bundle,dataset,kind,SMALL_K[dataset]) for kind in ("B0","B2","B4")]
        for spec in chosen:
            if spec is None:
                continue
            path = HERE / "scores" / f"{spec['id']}_C.json"
            record = read_json(path)
            for interval, value in record["horizon_metrics"].items():
                output.append([dataset,label(spec),interval,fmt(value["mse"]),fmt(value["mae"]),str(value["n_observed"])])
    return code_table(("dataset","method","horizon","MSE","MAE","observed"), output)


def r4_exposure_table(bundle):
    contract = read_json(HERE / "DATA_CONTRACT.json")
    statistics = read_json(HERE / "FOLD_STATISTICS.json")
    identifiers = set()
    for spec in bundle["routes"].values():
        if spec.get("block") == "R4":
            identifiers.add(spec["id"])
            if spec.get("base"):
                identifiers.add(spec["base"]["id"])
    rows = []
    for identifier in sorted(identifiers):
        spec = bundle["routes"][identifier]
        fit = bundle["fits"].get(identifier)
        if fit is not None:
            rows.append([spec["dataset"],label(spec),"minibatch_prefix",fit["legal_A_origins"],
                fit["distinct_A_origins"],4*fit["updates"],fit["updates"],fit["unique_target_times"],
                fit["unique_true_observed_target_entries"],"NA"])
            continue
        fit = bundle["ridge_fits"][identifier]
        all_origins = contract["folds"][spec["dataset"]][spec["fold"]]["origins"]["A"]
        if fit["fit_A_origins"] != len(all_origins):
            raise ValueError("Ridge all-A exposure differs from contract")
        exposure = statistics[spec["dataset"]][spec["fold"]]["A_exposure"]
        rows.append([spec["dataset"],label(spec),"all_A_sufficient",len(all_origins),fit["fit_A_origins"],
            fit["fit_A_origins"],0,exposure["unique_target_times"],sum(exposure["target_observed_unique"]),
            "ZERO" if fit["selected_alpha"] is None else fit["selected_alpha"]])
    return "\n\n".join([
        "**R4의 학습 정보·노출 차이:** ridge/native는 모든 합법 A origin을 한 번씩 순회해 sufficient statistics를 누적한 닫힌형 해법이다. "
        "신경망은 동일 A 범위의 전체 legal pool에서 사전 고정한 최대2400-update minibatch schedule의 해당 endpoint prefix를 사용한다. "
        "따라서 같은 A 시간 범위를 허용하지만 optimizer update 수나 실제 origin/target 노출이 정확히 같은 비교는 아니다. "
        "origin을 더 촘촘히 쓰는 것과 새로 관측된 시간 범위를 늘리는 것도 다르다.",
        code_table(("dataset","route","estimator","legal_A_pool","distinct_origins","origin_uses","Adam_updates","unique_Y_times","unique_obs_Y","alpha"),rows),
        "표의 neural 수치는 최종 학습 endpoint의 실제 노출이며 B로 채택한 CP는 더 이를 수 있다. "
        "닫힌형 경로는 alpha별로 같은 통계를 재사용하고 ZERO가 선택돼도 이미 실행한 다른 alpha의 적합/비용을 지우지 않는다. "
        "native가 Bolt보다 좋아도 감독 노출·추정법 차이를 제거한 예측기 단독의 인과 효과나 "
        "모든 조건에서 native가 충분하다는 결론으로 확대하지 않는다.",
    ])


def fitting_time_summary(bundle):
    grouped = {}
    seen_segments = set()
    for fit in bundle["fit_receipts"]:
        segment = (fit["id"],fit["updates"])
        if segment in seen_segments:
            raise ValueError("Duplicate neural endpoint receipt would double-count segment wall")
        seen_segments.add(segment)
        key = (fit["block"],fit["dataset"])
        row = grouped.setdefault(key,{"ids":set(),"endpoints":0,"updates":0,"update_seconds":0.,"segment_seconds":0.})
        row["ids"].add(fit["id"])
        row["endpoints"] += 1
        row["segment_seconds"] += float(fit["segment_wall_seconds"])
    for fit in bundle["fits"].values():
        row = grouped[(fit["block"],fit["dataset"])]
        row["updates"] += int(fit["updates"])
        row["update_seconds"] += sum(float(update["seconds"]) for update in fit["learning_curve"])
    audited_updates = bundle["audit"].get("run_index_totals",{}).get("optimizer_updates_excluding_probe")
    if audited_updates is not None and sum(row["updates"] for row in grouped.values()) != int(audited_updates):
        raise ValueError("Fitting-time update sum differs from saved-results audit")
    neural_rows = [[block,dataset,len(row["ids"]),row["endpoints"],row["updates"],
                    fmt(row["update_seconds"],3),fmt(row["segment_seconds"],3)]
                   for (block,dataset),row in sorted(grouped.items())]
    neural_rows.append(["TOTAL","all",len(bundle["fits"]),len(bundle["fit_receipts"]),
        sum(row["updates"] for row in grouped.values()),fmt(sum(row["update_seconds"] for row in grouped.values()),3),
        fmt(sum(row["segment_seconds"] for row in grouped.values()),3)])
    ridge_groups = {}
    for fit in bundle["ridge_fits"].values():
        key = (fit["block"],fit["dataset"])
        row = ridge_groups.setdefault(key,{"logical":0,"alpha_fits":0,"wall":0.})
        row["logical"] += 1
        row["alpha_fits"] += int(fit["fit_candidates"])
        row["wall"] += float(fit["fit_wall_seconds"])
    ridge_rows = [[block,dataset,row["logical"],row["alpha_fits"],fmt(row["wall"],3)]
                  for (block,dataset),row in sorted(ridge_groups.items())]
    ridge_rows.append(["TOTAL","all",len(bundle["ridge_fits"]),sum(row["alpha_fits"] for row in ridge_groups.values()),
                       fmt(sum(row["wall"] for row in ridge_groups.values()),3)])
    phases, pending = [], {}
    for line in (HERE / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        phase, stage = event.get("phase"),event.get("stage")
        if phase not in ("core","followup"):
            continue
        if stage == "PHASE_START":
            if phase in pending:
                phases.append([phase,"OPEN_START_REPLACED","NA"])
            pending[phase] = float(event["time_unix"])
        elif stage in ("PHASE_COMPLETE","PHASE_FAILURE") and phase in pending:
            phases.append([phase,stage,fmt(float(event["time_unix"])-pending.pop(phase),3)])
    phases += [[phase,"NO_END_EVENT","NA"] for phase in pending]
    return "\n\n".join([
        "### 실제 학습·적합에 사용한 시간",
        "신경망 update 시간은 logical condition별 마지막 learning_curve[].seconds를 한 번만 합산했다. "
        "600→1200→2400 receipt마다 반복 저장된 이전 curve prefix를 다시 더하지 않는다. "
        "반면 segment_wall_seconds는 각 endpoint 파일이 새로 실행한 구간만 기록하므로 endpoint 전체의 값을 합한다.",
        code_table(("block","dataset","logical_fits","segment_receipts","Adam_updates","update_seconds_sum","segment_wall_seconds_sum"),neural_rows),
        "update 시간은 해당 forward/backward/optimizer와 동기화를 포함한 host wall이며 GPU kernel 시간이라고 부르지 않는다. "
        "segment wall은 각 새 학습 loop와 CP 평가·저장을 포함하고, 그 이후 최종 선정 checkpoint load와 모델 준비를 포함한 전체 process wall은 아니다. "
        "기록이 없는 실패·중단 구간의 시간을 추정해서 더하지 않는다. 위 합계에서 probe는 별도다.",
        code_table(("block","dataset","ridge_logical_routes","alpha_fits","fit_wall_seconds_sum"),ridge_rows),
        "ridge/native의 fit_wall은 각 logical route의 all-A 통계 누적, 실제 alpha 적합과 B 평가·ZERO 비교 등을 포함한 함수 범위다. "
        "여러 alpha가 같은 통계를 재사용하므로 각 alpha에 전체 함수 wall을 반복 배분하지 않는다. "
        "이 CPU/GPU 혼합 실행 구간을 순수 CPU 시간이나 순수 GPU 시간이라고 부르지 않는다.",
        code_table(("phase","end_event","phase_wall_seconds"),phases),
        "EVENTS의 core/followup start→end 차이는 모델·자료 준비, CP 평가, 학습 외 작업까지 포함하는 보조 phase wall이다. "
        "section 합계와 범위가 다르며 두 수치를 더해 총 소요시간으로 만들지 않는다. [EVENTS 원기록](EVENTS.jsonl).",
    ])


def cost_table(bundle):
    output = []
    for dataset in MAIN_DATASETS:
        for batch in (1,8):
            for scope in ("general","exclusive"):
                for spec,row,cost in selected_cost_rows(bundle,dataset,batch,scope):
                    output.append([dataset,label(spec),batch,cost.get("scope","NA"),fmt(number(cost.get("whole_seconds_median")),6),
                        fmt(number(cost.get("whole_seconds_variance")),9),cost.get("valid_trials","NA"),
                        fmt(number(cost.get("peak_allocated_bytes_median"))/2**20 if number(cost.get("peak_allocated_bytes_median")) is not None else None,2),
                        fmt(number(cost.get("peak_reserved_bytes_median"))/2**20 if number(cost.get("peak_reserved_bytes_median")) is not None else None,2),
                        fmt(number(cost.get("F_resident_bytes"))/2**20 if number(cost.get("F_resident_bytes")) is not None else None,2),
                        fmt(number(cost.get("active_parameter_buffer_bytes"))/2**20 if number(cost.get("active_parameter_buffer_bytes")) is not None else None,2)])
    return code_table(("dataset","route","batch","scope","whole_s_median","whole_s2_var","trials","peak_alloc_MiB","peak_reserv_MiB","F_resid_MiB","active_MiB"),output)


def context_table(bundle):
    rows = []
    for spec in bundle["routes"].values():
        if spec.get("block") != "R1":
            continue
        row = score_row(bundle,spec["id"])
        rows.append([row["dataset"],spec.get("variant","Raw"),row.get("physical_past_length",row["length"]),
            row.get("F_context_length","NA"),row.get("F","NA"),row["r"],row["n_origins"],row["cohort"][:12]])
    rows.sort(key=lambda row:(row[0],int(row[1][1:]) if str(row[1]).startswith("A") else -1))
    return code_table(("dataset","setting","physical_past_L","F_context_L","F","r","C_origins","cohort_prefix"),rows)


def cost_scope_contrast(bundle):
    rows = []
    for dataset in DATASETS:
        for batch in (1,8):
            exclusive = {spec["id"]:cost for spec,_,cost in selected_cost_rows(bundle,dataset,batch,"exclusive")}
            for spec,_,general in selected_cost_rows(bundle,dataset,batch,"general"):
                if spec["type"] != "RAW" or spec["id"] not in exclusive:
                    continue
                separate = exclusive[spec["id"]]
                general_time, exclusive_time = number(general["whole_seconds_median"]),number(separate["whole_seconds_median"])
                rows.append([dataset,label(spec),batch,fmt(general_time*1000 if general_time is not None else None,3),
                    fmt(exclusive_time*1000 if exclusive_time is not None else None,3),
                    fmt(general_time/exclusive_time if general_time is not None and exclusive_time not in (None,0) else None,3)])
    return "\n\n".join([
        "**비용 scope 간 차이는 해결된 원인이 아니다.** 아래 동일 Raw route도 balanced general-session과 fresh exclusive-process의 latency가 다를 수 있다. "
        "그림05의 시간축·경로 간 속도 판단은 사전 계획의 같은 session/balanced general 값만 사용하고, "
        "메모리축은 별도 process의 exclusive peak만 사용한다. 두 scope 중 빠른 값을 골라 속도를 주장하지 않는다.",
        code_table(("dataset","Raw_route","batch","general_median_ms","exclusive_median_ms","general_over_exclusive"),rows),
        "두 scope의 시간 차이 원인은 현재 미확정이다. fresh-process 시간은 보조 진단으로만 남기며 "
        "변동·warmup·장치 부하·환경의 특정 원인을 확인 없이 붙이지 않는다. "
        "이 표·그림의 scope 구분은 기존 COST/선정/예측 원수치를 변경한 재실험이 아니다.",
    ])


def diagnostic_pairs(bundle):
    """C rows for the primary B-chosen K family and its same-family time refit."""
    result = []
    for dataset in DATASETS:
        primary = chosen_family(bundle, dataset)
        if primary is None:
            continue
        for fold in (("primary", "secondary") if dataset in MAIN_DATASETS else ("primary",)):
            spec = primary if fold == "primary" else selected_adapter(bundle,dataset,primary["kind"],SMALL_K[dataset],fold)
            if spec:
                row = score_row(bundle,spec["id"])
                result.append((spec,row,raw_for(bundle,row)))
    return result


def _difference(left, right):
    left, right = number(left), number(right)
    return None if left is None or right is None else left-right


def pattern_summary(bundle):
    groups = read_csv(HERE / "PATTERN_GROUPS.csv")
    index = {(row["id"],row["split"],row["cohort"],row["indicator"],row["group"]): row for row in groups}
    output = []
    for spec, row, raw in diagnostic_pairs(bundle):
        indicators = sorted({item["indicator"] for item in groups if item["id"] == row["id"] and item["split"] == "C" and item["cohort"] == row["cohort"]})
        for indicator in indicators:
            for group in ("low","high","unavailable"):
                chosen = index[(row["id"],"C",row["cohort"],indicator,group)]
                baseline = index[(raw["id"],"C",raw["cohort"],indicator,group)]
                for key in ("dataset","fold","split","cohort","indicator","group","n_origins","n_observed","active_channels","A_median_threshold"):
                    if str(chosen[key]) != str(baseline[key]):
                        raise ValueError("Unmatched pattern group: " + key)
                delta = _difference(chosen["mse"],baseline["mse"])
                output.append([row["dataset"],row["fold"],label(spec),indicator,group,
                    chosen["n_origins"],chosen["n_observed"],chosen["active_channels"],
                    fmt(chosen["A_median_threshold"],4),fmt(baseline["mse"]),fmt(chosen["mse"]),fmt(delta)])
    return "\n\n".join([
        "### 과거 패턴 그룹: 선정 K 가족과 같은 cohort Raw",
        "아래 Δ는 model MSE−Raw MSE이며 음수일 때 오차가 작다. low/high/unavailable를 모두 표시하고 관측이 없는 그룹은 NA로 남긴다. "
        "그룹 경계는 A 전체 창이 아니라 사전 고정한 시간 균일 최대32개 A 과거 창의 finite median이다. "
        "A의 실제 학습 pool·고유 감독 노출·B 첫32 latent 진단 cohort와도 다른 수다. "
        "두 번째 fold에서는 첫 fold B에서 고른 같은 가족을 새 A/B로 재적합한 모델을 비교한다.",
        code_table(("dataset","fold","model","indicator","group","origins","observed","active_C","A32_median","Raw_MSE","model_MSE","delta"),output),
        "과거 변동·계절성·상관과 상대 오차의 관계는 위 전체 행을 함께 읽는다. 그룹별 음수 Δ가 보여도 "
        "해당 패턴이 개선의 원인이라는 증거가 아니며, 그룹을 고르는 추가 selector를 학습하지 않았다. "
        f"전체 {len(groups)}개 그룹 행과 채널별 오차·관측수는 [PATTERN_GROUPS](PATTERN_GROUPS.csv)에 있다."
    ])


def error_summary(bundle):
    errors = read_csv(HERE / "ERROR_DIAGNOSTICS.csv")
    index = {(row["id"],row["split"],row["cohort"]): row for row in errors}
    output, differences = [], []
    for spec,row,raw in diagnostic_pairs(bundle):
        chosen = index[(row["id"],"C",row["cohort"])]
        baseline = index[(raw["id"],"C",raw["cohort"])]
        for key in ("dataset","fold","cohort","n_origins","n_observed","peak_amplitude_valid_origin_channel_pairs",
                    "spike_n_observed","non_spike_n_observed","adjacent_difference_unavailable_n_observed"):
            if str(chosen[key]) != str(baseline[key]):
                raise ValueError("Unmatched error diagnostic: " + key)
        for name,item in (("Raw",baseline),(label(spec),chosen)):
            output.append([row["dataset"],row["fold"],name,item["n_origins"],item["n_observed"],
                fmt(item["peak_time_absolute_error_steps_macro"],3),fmt(item["peak_time_absolute_error_seconds_macro"],1),
                fmt(item["normalized_range_amplitude_absolute_error_macro"]),item["peak_amplitude_valid_origin_channel_pairs"],
                fmt(item["spike_mse"]),item["spike_n_observed"],fmt(item["non_spike_mse"]),item["non_spike_n_observed"],
                fmt(item["adjacent_difference_unavailable_mse"]),item["adjacent_difference_unavailable_n_observed"]])
        differences.append([row["dataset"],row["fold"],label(spec),
            fmt(_difference(chosen["peak_time_absolute_error_steps_macro"],baseline["peak_time_absolute_error_steps_macro"]),3),
            fmt(_difference(chosen["normalized_range_amplitude_absolute_error_macro"],baseline["normalized_range_amplitude_absolute_error_macro"])),
            fmt(_difference(chosen["spike_mse"],baseline["spike_mse"]))])
    return "\n\n".join([
        "### peak 위치·진폭·급변 시점의 오차",
        "peak 시간은 관측된 미래 H48 위치의 argmax 차이이며 물리학적 주기/위상을 추정한 값이 아니다. "
        "동률이면 가장 이른 관측 argmax를 쓴다. 진폭은 같은 관측 격자에서 max−min 범위의 절대 오차다. "
        "두 값 모두 origin×channel에 정답 관측이 최소2개 있을 때만 세고 channel macro로 요약한다.",
        code_table(("dataset","fold","model","origins","observed","peak_steps","peak_seconds","range_amp_err","valid_pairs",
                    "spike_MSE","spike_n","nonspike_MSE","nonspike_n","unavailable_MSE","unavail_n"),output),
        "Δ는 선정 모델−Raw이며 각 열의 단위가 다르다.",
        code_table(("dataset","fold","model","peak_step_delta","amplitude_delta","spike_MSE_delta"),differences),
        "spike는 실제 미래 인접 차이가 A의 관측된 연속 시점 차이 95% 분위수를 넘는지를 사후 분류한다. "
        "첫 horizon은 원래 o−1이 관측됐을 때만 분류하며 미분류 관측도 별도 행수·MSE로 남긴다. "
        "이는 **미래로 정의한 사후 오차 진단**이며 과거 입력·모델 선택·배포 selector의 정보가 아니다. "
        "spike 개선과 일반화·인과성을 동일시하지 않는다. [전체 원수치](ERROR_DIAGNOSTICS.csv)."
    ])


def latent_summary(bundle):
    rows = read_csv(HERE / "LATENT_DIAGNOSTICS.csv")
    index = {row["id"]:row for row in rows}
    selected = {}
    for dataset in DATASETS:
        for fold in (("primary","secondary") if dataset in MAIN_DATASETS else ("primary",)):
            ranks = (CHANNELS[dataset],SMALL_K[dataset]) if fold == "primary" else (SMALL_K[dataset],)
            for rank in ranks:
                for kind in ("B0","B1","B2","B3","B4"):
                    spec = selected_adapter(bundle,dataset,kind,rank,fold)
                    if spec:
                        selected[spec["id"]] = spec
    output = []
    for identifier,spec in selected.items():
        row = index[identifier]
        variance = np.asarray(json.loads(row["latent_variance_per_stream_json"]),dtype=np.float64)
        output.append([row["dataset"],row["fold"],row["kind"],row["r"],row["seed"],row["B_origins"],
            row["E_rank"],row["D_rank"],row["latent_centered_rank"],fmt(variance.mean()),fmt(variance.min(),9),
            int(np.sum(variance < 1e-8)),int(np.sum(variance == 0)),row["training_status"]])
    return "\n\n".join([
        "### 잠재 표현의 수치 상태",
        "각 모델의 선정 CP를 CPU에서 다시 구성하고 B의 첫 최대32개 origin만 사용했다. 아래는 같은 구조/r에서 B로 고른 seed를 보여주며 "
        "전체 seed·singular value·tolerance는 원 CSV에 남긴다. E/D rank는 FP32 상대 SVD tolerance이고 "
        "잠재 rank는 반복 과거를 포함한 centered latent 행렬의 수치 rank다.",
        code_table(("dataset","fold","P","r","seed","B_first_n","E_rank","D_rank","z_rank","z_var_mean","z_var_min","var_lt1e-8","var_eq0","training_status"),output),
        "`var_lt1e-8`은 A-표준화 입력에서 얻은 잠재 stream variance를 탐색적으로 표시하는 고정 수치 기준이다. "
        "스케일 선택에 의존하며 극소 분산이나 rank 저하만으로 표현 붕괴·예측 실패·방법 불가능성을 확정하지 않는다. "
        "반대로 rank가 가득 차거나 gradient가 유한하다는 검사도 예측 성능·수렴의 증거가 아니다. "
        "작은 모델만 CPU로 열었고 이 분석은 F를 로드/호출하지 않았다. [전체 latent 진단](LATENT_DIAGNOSTICS.csv)."
    ])


def optional_analysis(bundle):
    # All three outputs are required by load_bundle's analysis completion gate.
    return "\n\n".join((pattern_summary(bundle),error_summary(bundle),latent_summary(bundle),
        "[ANALYSIS_COMPLETE](ANALYSIS_COMPLETE.json)의 검산은 저장 예측·cohort·정답 mask·원수치 재계산과 고정 source/artifact 확인 범위다. "
        "T+R 항등·좌표 일치·mask 검사 PASS와 위 성능 원수치는 다른 증거다. 이 후처리 분석이 학습·타이머 전체를 독립 재실행했다는 뜻은 아니다."))


def oracle_summary(bundle):
    path = HERE / "ORACLE_DIAGNOSTIC.json"
    if not path.is_file():
        return ("### 고정 D의 미래 잠재값 진단\n\n`NOT_RUN`: 이 보고서 생성 시점에 ORACLE_DIAGNOSTIC.json이 없으므로 "
                "실행했다고 쓰지 않는다. 실제 미래 정답으로 임의 미래 latent를 맞추는 선택적 진단은 "
                "예측 방법·배포 성능표에 포함하지 않으며, 미실행이면 고정 D의 수치 공간에 대한 추가 판단도 보류한다.")
    receipt = read_json(path)
    for key, filename in (("scores_sha256","SCORES.csv"),("score_complete_sha256","SCORE_COMPLETE.json"),
                          ("selection_lock_sha256","SELECTION_LOCK.json"),("cost_lock_sha256","COST_LOCK.json")):
        if receipt.get(key) != sha(HERE / filename):
            raise RuntimeError("ORACLE_DIAGNOSTIC_RECEIPT_MISMATCH:" + key)
    if receipt.get("F_model_loads") != 0 or receipt.get("F_forward_calls") != 0:
        raise RuntimeError("ORACLE_DIAGNOSTIC_UNEXPECTED_F_EXECUTION")
    output, certificates = [], []
    notices = []
    for result in receipt["results"]:
        fixed = result.get("fixed_D_FP64_real_oracle")
        if fixed is None:
            notices.append(f"- {result['dataset']}/{result['family']}: `{result['status']}` — "
                           f"{result.get('reason',result.get('error','진단 수치 없음'))}")
            continue
        trigger, replay = result["trigger"], result["FP32_decoder_replay"]
        actual = score_row(bundle,result["id"])
        if actual["cohort"] != trigger["cohort"] or int(actual["n_observed"]) != int(fixed["n_observed"]):
            raise RuntimeError("ORACLE_REPORT_COHORT_OR_OBSERVATION_MISMATCH")
        if not result.get("future_Y_used_only_in_diagnostic_solver") or result.get("E_of_Y_used"):
            raise RuntimeError("ORACLE_DIAGNOSTIC_ROLE_MISMATCH")
        output.append([result["dataset"],result["family"],result["r"],result["seed"],result["status"],
            fmt(trigger["C_mse"]),fmt(trigger["canonical_Raw_C_mse"]),scientific(fixed["mse"]),
            scientific(replay.get("mse")),scientific(result["free_latent_max_abs"]),fixed["n_observed"]])
        valid_certificates = [item for item in fixed["certificates"] if item.get("svd_executed")]
        ranks = [item["numeric_rank"] for item in valid_certificates]
        conditions = [item["retained_condition_number"] for item in valid_certificates if item.get("retained_condition_number") is not None]
        normal = [item["normal_equation_max_abs"] for item in valid_certificates]
        certificates.append([result["dataset"],result["family"],"x".join(map(str,fixed["A_shape"])),
            fixed["mask_patterns"],min(ranks) if ranks else "NA",max(ranks) if ranks else "NA",
            scientific(max(conditions) if conditions else None),scientific(max(normal) if normal else None),
            fixed["weighted_sse_vs_macro_mse_parity"],fixed["minimum_inequality_vs_feasible_controls"],
            replay["status"],scientific(replay.get("FP32_decode_minus_FP64_real_mse"))])
    return "\n\n".join([
        "### 고정 D의 미래 잠재값 진단 — 실제 예측 성능과 분리",
        f"진단 전체 상태는 `{receipt['status']}`다. primary fold의 B2/B4 작은 K 후보를 locked B MSE로 고른 뒤 "
        "같은 C cohort의 canonical Raw보다 오차가 큰 경로에만 시행한다. C의 비교는 진단 시행 여부를 정할 뿐 "
        "새 checkpoint·손실·표현·배포 selector를 선택하지 않는다. trigger margin0은 유의성 검정이 아니다.",
        "고정된 origin-reference lag7 D와 이미 아는 과거/현재 C anchor를 유지하고 H×K개의 **자유로운 미래 latent를 실제 미래 Y로 직접 적합**한다. "
        "전역 C 관측수의 macro MSE 가중과 원 mask를 쓰며 누락 output 때문에 다른 horizon의 미지 latent 좌표를 삭제하지 않는다. "
        "이는 E(Y)도, 과거 P/E가 예측한 미래도 아니다. F는 로드/호출하지 않았다.",
        code_table(("dataset","family","K","seed","diagnostic_status","actual_C_MSE","Raw_C_MSE","FP64_diag_MSE","FP32_replay_MSE","free_z_maxabs","observed"),output),
        code_table(("dataset","family","map_shape","mask_patterns","rank_min","rank_max","condition_max","normal_resid_max",
                    "weighted_obj_parity","feasible_inequality","replay_status","FP32minusFP64_MSE"),certificates),
        "\n".join(notices) if notices else "모든 해당 진단 경로의 수치·상태를 위 표에 남겼다.",
        "FP64 값은 고정 rcond1e-12로 유지된 가중 SVD 수치 공간에서 얻은 진단치다. "
        "절단하지 않은 모든 실수 해 또는 모든 FP32 출력의 엄격한 하한이라고 부르지 않는다. "
        "FP32 replay는 같은 작은 D의 CPU trace이고 GPU bitwise 동등성 주장이 아니다. "
        "자유 latent의 큰 진폭·conditioning·재생 차이와 미해결 status를 함께 읽는다. "
        "진단 오차가 작아도 과거 표현의 예측 정보가 충분하거나 F가 그 미래 latent에 도달할 수 있다는 뜻은 아니다. "
        "진단 오차가 커도 해당 고정 D와 남긴 수치 공간의 제약일 뿐 다른 decoder·모든 작은 K 방법의 불가능성으로 확대하지 않는다. "
        "[진단 조건·원수치·certificate](ORACLE_DIAGNOSTIC.json).",
    ])


def audit_summary(bundle):
    audit = bundle["audit"]
    findings = audit.get("findings",[])
    text = (f"저장 산출물 감사는 `{audit['status']}`, ERROR {audit.get('error_count',0)}, warning {audit.get('warning_count',0)}다. "
            "선정·data 계약·학습 receipt·저장 예측/mask·비용 원수치를 검산한 범위이며 "
            f"독립 학습 재실행 {audit.get('independent_training_reruns',0)}, 타이머 재실행 {audit.get('independent_timer_reruns',0)}다. "
            "PASS는 성능 개선/충분 수렴/새 독립 일반화의 증거가 아니며 PATH_FINDINGS는 해당 경로의 한계를 그대로 남긴다. "
            f"[현재 전체 감사]({bundle['audit_path'].name}), [현재 실행 목록]({bundle['run_index_path'].name}).")
    manifest = bundle.get("audit_manifest")
    if manifest is not None:
        prior = read_json(HERE/"AUDIT.json") if "AUDIT.json" in manifest.get("prior_files",{}) else None
        prior_text = f"`{prior['status']}`·ERROR {prior.get('error_count',0)}" if prior is not None else "보존된 첫 감사 시도"
        text += ("\n\n첫 감사는 CSV 기본 field 한도131,072자로 큰 비용 메타데이터를 읽지 못해 "
                 + prior_text + "를 기록했다. reader 실패4개와 그에 따른 종속 finding152개를 방법/비용 성능 실패로 해석하지 않는다. "
                 "원 감사·실행 목록·로그를 덮지 않고 16MiB reader 한도와 합성 회귀 후 새 이름으로 저장 산출물을 실제 재검산했다. "
                 "과학 SOURCE·예측·비용 원수치는 바꾸지 않았으며, 현재 보고서는 이름으로 최신 파일을 추측하지 않고 "
                 "[명시적 감사 manifest](AUDIT_LATEST.json)가 지정한 v2 파일과 SHA를 검증해 사용한다. "
                 "이 metadata reader 수선도 학습/타이머의 독립 재실행은 아니다. 보존 기록: "
                 + ", ".join(f"[{filename}]({filename})" for filename in manifest.get("prior_files",{})) + ".")
    if findings:
        text += "\n\n```json\n" + json.dumps(findings,ensure_ascii=False,indent=2) + "\n```"
    return text


def ablation_table(bundle):
    rows = []
    for spec in bundle["routes"].values():
        if spec.get("block") not in ("R5_ABLATION","R5_INTERMEDIATE"):
            continue
        row = score_row(bundle,spec["id"])
        rows.append([spec["dataset"],spec["id"],spec.get("updates",0),fmt(row["mse"]),fmt(ratio(bundle,row),4),spec.get("training_status","NA")])
    text = code_table(("dataset","refit_id","updates","C_MSE","RawRatio","training_status"),rows)
    r5 = read_json(HERE/"R5_SELECTIONS.json")
    reuse = r5.get("reuses",[])
    return text + "\n\n" + "\n".join(f"- `{item['question']}`: `{item['status']}`, `{item['source']}` 대 `{item['comparator']}`. {item['basis']}" for item in reuse)


def numerical_summary(bundle):
    fits = list(bundle["fits"].values())
    under = [fit for fit in fits if fit.get("status") == "UNDERTRAINED_OR_UNRESOLVED"]
    update_total = sum(fit["updates"] for fit in fits)
    lines = [f"Adam 계열 원 condition {len(fits)}개, 최종 endpoint 기준 합계 {update_total:,} update다. 연장분을 별도 새 모델처럼 중복 합산하지 않았다. "
             f"이 중 `{len(under)}`개는 최대 endpoint에서도 마지막 최적/지속 하강 조건이 남아 미수렴·미해결이다."]
    rows = []
    for fit in sorted(fits,key=lambda item:item["id"]):
        if fit.get("block") != "R2":
            continue
        rows.append([fit["dataset"],fit["kind"],fit["r"],fit["seed"],fit["updates"],fit["selected_update"],
            fit.get("distinct_A_origins","NA"),fit.get("legal_A_origins","NA"),fit.get("unique_target_times","NA"),
            fmt(fit.get("clipping_fraction"),3),fit.get("status","NA")])
    lines.append(code_table(("dataset","P","r","seed","updates","CP","seen_origins","legal_A_pool","unique_Y_times","clip_frac","status"),rows))
    if under:
        lines.append("미수렴·미해결 ID: " + ", ".join(f"`{fit['id']}`" for fit in under) + ". 유한 gradient는 최적화 완료 증거가 아니다.")
    return "\n\n".join(lines)


def report_text(bundle):
    findings = finding_lines(bundle)
    r1 = read_json(HERE/"R1_B.json")["records"]
    failed_r1 = [row for row in r1 if row["status"] != "COMPLETE_VALID"]
    train_states = {}
    for spec in bundle["routes"].values():
        state = spec.get("training_status","NOT_APPLICABLE")
        train_states[state] = train_states.get(state,0)+1
    report = ["# 고정 Bolt의 시간 앞단: 실제 R1–R5 결과", "",
        "이 보고서는 새 과거→과거 시간 앞단을 실제 학습한 이번 캠페인의 결과다. 이전 LEVEL/PCA/ForeCA 결과를 새 학습으로 재명명하지 않았다. "
        "주평가는 fold A 관측값으로 표준화한 원채널·원미래 H48의 관측 mask 전용 채널 macro MSE다. 모든 비교는 같은 dataset/fold/target cohort를 검사한다.",
        "", "## 먼저 답해야 할 질문", "", "\n\n".join(findings), "",
        "결과의 방향은 위 원수치에 한정된다. 두 seed의 범위는 신뢰구간이 아니며 겹치는 창·채널·seed를 독립 반복으로 세어 유의성을 주장하지 않는다. "
        "C를 본 뒤 새 구조·checkpoint·alpha·데이터를 다시 선택하지 않았다.",
        "", "## 무엇을 실제로 실행했고 어디까지 완료했는가", "",
        f"[SCORE_COMPLETE](SCORE_COMPLETE.json)는 `{bundle['completion']['status']}`, [COST_LOCK](COST_LOCK.json)는 `{bundle['cost_lock']['status']}`다. "
        f"실제 score행은 {len(bundle['scores'])}개이고 실행 index행은 {len(bundle['run_index'])}개다. 이는 새 학습 모델 수와 같은 숫자가 아니다. "
        f"B/C 재평가·무학습 변환·비용 trial·ridge alpha 적합·학습 update를 구분한 원기록은 [RUN_INDEX]({bundle['run_index_path'].name}), [SCORES](SCORES.csv), [COST](COST.csv)에 있다.",
        "", audit_summary(bundle),
        "", "별도 프로세스 비용의 첫 queue는 Windows venv launcher 부모를 경쟁 작업으로 오탐해 모델 로드·정밀 trial 전에 정지했다. "
        "[실패 attempt·잔여 로그·수선 기록](EXECUTION_REPAIR_KO.md)을 보존했고 이를 성공 trial이나 방법 성능 실패로 세지 않는다. "
        "같은 실제 base Python과 기존 venv 모듈의 경로·버전을 확인해 직접 재실행했으며 동결 과학 SOURCE·선정·통계·guard는 바꾸지 않았다. "
        "환경 경로/버전 확인은 수치의 bitwise 동등성 증명이 아니다. 재시도의 최종 비용 성공 근거는 첫 실패 로그가 아니라 "
        "유효한 COST_LOCK과 그에 연결된 원 trial이다. C 확인이나 모델 재선택을 위한 수선이 아니었다.",
        "", "R1은 GPU queue 운영상 R2 본학습 뒤에 실제 실행했다. 설정과 SOURCE는 R2 동안 준비됐고, R1 결과가 나쁘다는 이유로 R2 가족을 취소하지 않았다. "
        "R4는 사전 고정 ceiling16을 지켜 swap 2alpha(0,.01)×2가족×2자료=8, direct 2가족×2자료×1seed=4, LoRA1구성×2자료=2, "
        "raw ridge alpha .01×2자료=2로 합계16 적합이다. ZERO는 추가 적합이 아니다. 코드의 범용 ALPHAS 상수를 실제 실행 메뉴로 읽지 않는다. "
        "G는 첫 fold10+둘째 fold10=20이며 첫 fold G의 seed는 하나다. LoRA의 configured_lr는 기존 SOURCE에서 사전에 선정한 .0001이고 "
        "나머지 Adam 조건은 .001이다. 실제 학습률은 각 fit receipt와 아래 lr 열에서 확인하며 ridge에는 Adam 학습률을 붙이지 않는다.",
        "", "R1 미실행·무효 경로: " + (" / ".join(f"{row['dataset']}/{row['variant']} `{row['status']}`: {row.get('reason',row.get('error',''))}" for row in failed_r1) if failed_r1 else "없음") + ".",
        "", "## 데이터 계약과 평가 범위", "",
        "Beijing C11/K2와 Jena C21/K5가 주 자료이며 Robin C17/K4는 짧은 기간의 스트레스 진단이다. "
        "기존 허용 FIT 안에서 첫 fold A/B/C=60/20/20, 둘째=50/20/30을 결과 전에 고정했다. "
        "각 fold의 scaler·초기값·PCA·E/D·G는 A에서 새로 적합하고 checkpoint·alpha·가지 선택은 B까지만 사용한다. "
        "A는 모든 합법 origin을 pool로 두고 B/C는 stride48+끝 origin의 고정 cohort다. 마지막 두 창은 겹칠 수 있다. "
        "A와 B, B와 C 사이 target은 겹치지 않고 뒤 구간 history에 앞 구간의 이미 관측된 값이 들어가는 것은 허용한다.",
        "", "Jena/Robin은 옛 전체 TRAIN 정규화 FP32 x를 역변환한 계보가 남으므로 무손실 raw A-only라고 부르지 않는다. "
        "safe NPZ의 member 압축 해제는 옛 TUNE/VAL을 포함할 수 있지만 새 반환·통계·학습에서는 원래 FIT만 사용한다. "
        "보호 TEST 및 원 CSV/ZIP 전체 재파싱은 이번 실행 범위 밖이다. C는 이번 알고리즘의 선택에서 잠갔지만 과거 캠페인에 노출된 시간이며 "
        "`historically_unseen=False`다. 독립 holdout 검증으로 바꾸어 말하지 않는다. [상세 자산·계보](DATA_ASSET_REVIEW.md), [새 계약](DATA_CONTRACT.json).",
        "", "## 방법과 공통 비교 조건", "",
        "기본 경로는 과거 X→P(과거→원격자 과거)→E(C→r)→frozen Bolt→같은 origin-reference lag-linear D→현재 원 C anchor→원미래다. "
        "모든 핵심 D는 lag0/1/2/4/8/16/32를 쓰며 r=C는 항등, r=K는 A-only PCA에서 시작한다. "
        "P는 최종 잔차층만 zero로 초기화해 CP0 no-op을 만들고 나머지 층의 초기값을 유지했다. "
        "B0는 점별 어댑터, B1은 채널 독립 단일 해상도 시간 MLP, B2는 시간/채널 혼합, B3는 분해 없는 다중해상도, "
        "B4는 성분 분해와 반대 방향 다중해상도 혼합을 사용한 Bolt용 변형이다. 내부 coarse/fine를 원격자로 융합한 뒤 r개 stream만 F에 보낸다.",
        "", "B2/B4 앞단은 Google TSMixer/TimeMixer의 전체 논문 성능 재현이 아니다. AdaPTS의 앞뒤 어댑터와 예측 목적 E/D 학습 자체도 신규성으로 주장하지 않는다. "
        "SPRINT-on-Bolt는 공식 기본 T5 wrapper를 Bolt q0.5로 바꾼 이식 대조다. 기간24/144는 기존 metadata에서 고정했으며 "
        "SPRINT 유효 lookback504/432의 Raw도 별도로 재계산했다. [공식 자료·고정 revision 대응](LITERATURE_MAPPING.md).",
        "", "F 가중치는 고정하지만 P/E로의 입력 gradient는 유지한다. 원 InstanceNorm의 양의 분산 순방향/미분을 유지하고 "
        "0-variance sqrt gradient만 유한한0으로 정의했다. 실제 Raw/microbatch/CP0 parity는 [ACTUAL_PARITY](ACTUAL_PARITY.json)의 허용오차와 범위를 확인한다. "
        "q0.5 예측의 MSE 비교이며 조건부 평균이나 확률예측 전체의 우위를 의미하지 않는다.",
        "", "## R1: 단순 시간 조작", "", "![R1](figures/01_time_transforms.png)", "",
        "A0 Raw; A1/2 F(trailing trend)+마지막 residual(폭5/25); A3/4 F(residual)+마지막 trend; "
        "A5/6 F(trend)+F(residual); A7/8 endpoint stride2/4; A9/10 block-mean stride2/4; "
        "A11 SPRINT-on-Bolt; A12/13 Raw L1024/2048이다. A5/6의 두 배 streams를 숨기지 않는다. "
        "R=X−T의 운영적 분해를 실제 계절·잡음 정답으로 해석하지 않는다. endpoint의 물리 미래 좌표와 block 평균의 반복 복원은 별개다. "
        "긴 L은 같은 origin의 Raw512로 나누고, SPRINT의 유효 lookback 대조도 score 원표에 별도로 남는다.",
        "", context_table(bundle), "",
        "physical_past_L은 원격자 과거 관측 범위이고 F_context_L은 시간 변환 이후 F에 전달한 격자의 행 수다. "
        "이는 예측 horizon과 다른 숫자이며 F의 고정 head를 crop했다고 head 전체 계산량이 줄었다고 해석하지 않는다. "
        "cohort prefix는 표시용이며 실제 paired 검사는 full hash·관측수·origin 수로 한다.",
        "", "## R2: 학습형 P와 C/K", "", "![R2](figures/02_temporal_rank.png)", "",
        primary_table(bundle,{"ADAPTER"}), "",
        "위 표의 status는 추론·점수 계산의 유효성과 별개인 실제 훈련 상태를 유지한다. seed 두 개의 평균·범위와 B로 선택한 개별 checkpoint를 구분한다. "
        "C가 좋은 seed만 골라 전반적 성공으로 보고하지 않는다.",
        "", "C/K 대조에서 r을 바꾸면 F stream 수, E(C×r)와 D(7r×C)의 trainable 파라미터 수, "
        "고정 D가 표현할 수 있는 복원 공간이 함께 바뀐다. 따라서 이 차이를 정보 압축만의 순수 인과 효과로 부르지 않는다. "
        "같은 C/r/seed/예산에서 B3/B4는 같은 파라미터 구조·개수를 가진 P-family 대조이므로 위 C/K 용량 변화와 구분한다. "
        "B3는 원신호 반분, B4는 residual/trend 분해를 사용하며 그 대조도 학습 동역학이나 최적화 난도가 같다는 보장은 아니다.",
        "", "## R3: 동일 G에서 남는 값", "", "![R3](figures/03_matched_residual.png)", "",
        primary_table(bundle,{"G"}), "",
        "G 입력은 모든 경로에서 같은 원 C past-current innovation이고 hidden32·zero-output·같은 B 선택 원칙이다. "
        "G-only, Raw+G, B0K+G보다 전체 결합이 실제로 나은지가 판별 대상이다. 압축 단독 실패는 이 결합 실험을 생략하는 gate가 아니었다.",
        "", "## R4: 예측기와 복원을 분리하는 대조", "",
        primary_table(bundle,{"SWAP","DIRECT","RIDGE","LORA"}), "",
        "native는 고정된 같은 P/E/D에서 latent의 마지막8/64/512 평균−현재값 세 특징으로 H48을 예측하는 shared144계수다. "
        "원채널 actual-Y에 선형인 결합 design의 FP64 sufficient statistics로 적합하며 B에서 alpha/ZERO를 선택한다. "
        "direct ridge는 채널별 16행 평균32개+현재값을 현재 anchor에 대해 중심화한33특징이다. 마지막 특징은 항등0이어서 최대 rank32이며 이를 숨기지 않는다. "
        "선형 닫힌형 적합에도 rank/normal-equation 잔차·수치 unresolved를 기록하고, 닫힌형이라는 이유만으로 임의 비선형 모델까지 최적이라고 부르지 않는다.",
        "", r4_exposure_table(bundle),
        "", oracle_summary(bundle),
        "", "## R5: 학습·시간 경계·가지 제거·오차 구간", "", "![Learning](figures/04_learning_time_refit.png)", "",
        numerical_summary(bundle), "",
        "A 학습곡선은 고정된 최대32개 고르게 배치한 A probe의 점수이며 전체 A 점수와 같지 않다. "
        "업데이트 minibatch loss, B checkpoint 점수, 실제 관측 고유 시점과 반복 노출을 각각 구분한다. "
        "두 번째 fold는 scaler/PCA/P/E/D/G의 전체 upstream 재적합이며 서로 다른 C 기간의 조건부 반복이다.",
        "", ablation_table(bundle), "",
        "가지 제거는 원 B4와 같은 endpoint 예산의 재학습 대조다. 재사용 B3의 분해 제거 대응은 위 지정 조건에서만 쓰며 "
        "단일 추론 ablation을 순수 성분 효과나 인과 증거로 바꾸지 않는다.",
        "", horizon_table(bundle), "", optional_analysis(bundle), "",
        "패턴 그룹은 과거 창만으로 계산하고 A의 median 경계로 나눈 설명적 분석이다. "
        "미래 오차로 그룹 기준을 고르거나 그룹 결과를 배포 선택기로 다시 쓰지 않는다. 높은 지연 상관은 인과성이나 추가 예측 정보의 증거가 아니다.",
        "", "## 실제 전체 비용과 메모리", "", "![Cost](figures/05_error_time_memory.png)", "", cost_table(bundle), "",
        cost_scope_contrast(bundle), "",
        "whole은 physical CPU 입력→과거 방향 결측 채움/A 표준화→P/E→실제 F→D/G→역표준화·CPU 반환·동기화를 포함한다. "
        "warmup5와 최소 repeat10(균형 순서를 위한 완전 rotation으로 실제 횟수가 늘 수 있음)의 원 sample/median/variance, batch1/8 및 동일 trial stage는 [COST](COST.csv)와 cost/ 원기록에서 확인한다. "
        "저장 cache 읽기를 F 추론 비용으로 세지 않는다. 실제 Raw 채널 microbatch parity도 검사했다. "
        "R1의 긴 lookback을 포함한 비용 provider는 자료별 공통 legal origin의 최대 필요 과거를 전달하고, 각 Route는 자기 physical_past_L로 먼저 자른 뒤 채움/표준화한다. "
        "따라서 같은 큰 배열을 전달받은 Raw512가 2048행 정보를 쓰는 것은 아니다. "
        "비용 표의 scope 열이 일반 균형 순서/별도 exclusive 프로세스 중 무엇인지 규정한다. "
        "활성 후보는 parameters+buffers의 unique storage 합계이며 중첩 참조를 중복 세지 않는다. 상주 F, 활성 후보 저장량, "
        "peak allocated/reserved는 다른 숫자다. peak−resident 차이를 순수 activation이라고 단정하지 않는다. "
        "정밀 GPU 비용 중 이 캠페인의 학습을 동시에 돌리지 않았지만 외부 장치 부하는 비용 한계로 남는다.",
        "", fitting_time_summary(bundle),
        "", "## 사전 고정 및 실패 사례", "", "![Cases](figures/06_forecast_cases.png)", "",
        "첫 열은 C 첫 origin·채널0을 사전에 고정한 사례다. 가운데는 전체 원채널 macro 오차 변화의 중앙 origin에서 채널0을, "
        "마지막은 오차 변화가 가장 큰 origin에서 채널별 변화가 가장 큰 채널을 사후 선택한 진단이다. "
        "최대 변화도 음수라면 악화 사례로 부르지 않고 least improvement라고 표시한다. "
        "뒤 두 선택을 사전 예측 성공이나 새로운 일반화 검증으로 쓰지 않는다. 관측되지 않은 정답은 선에서 제외한다. "
        "정확한 ID·origin·채널·선택 이유는 [그림 provenance](figures/DATA_PROVENANCE.json)에 남긴다.",
        "", "물리 단위 오차는 scores/의 `physical_per_channel`에서 원 physical target을 직접 비교한 값이다. "
        "정답을 표준화 후 역변환한 값을 원 정답 대신 사용하지 않았다. 예측 NPZ의 B/C_target_physical에도 이 관측 정답을 남겼다. "
        "단위가 다른 채널들의 물리 MSE를 단순 평균해 주성능 수치로 사용하지 않으며 주표·사례 그림은 A-표준화 단위다.",
        "", "## 현재 유지·보류와 다음 판단", "",
        "이번 원수치가 지지하는 범위는 같은 허용 A 시간 범위와 지정한 경로별 적합 예산에서 표준 단순 대안과 시간 앞단을 비교한 조건부 결과다. "
        "학습형 P가 B0를 이겨도 시간 정보의 어느 종류가 유효한지 단독 확정하지 않고, 작은 K가 지면 전체 표현 학습의 불가능성으로 확대하지 않는다. "
        "+G가 좋아도 G-only/Raw+G/B0+G를 이기지 못하면 새 앞단 또는 Bolt 경로의 추가 기여는 보류한다. "
        "native/direct가 같은 정확도·비용에서 유리하면 현재 Bolt 연결의 유지 근거가 약해진다. "
        "반대로 해당 원수치·비용에서 이득이 남아도 새 독립 시간/자료 검증 전에는 확증 기여로 올리지 않는다. "
        "최대 endpoint 미수렴 경로, FP32 물리값 복구 계보, 적은 seed와 짧은 Robin 기간은 미해결로 남긴다.",
        "", "새 contrastive/동적 graph 패키지·대형 모델 재학습·자료 교체를 이번 음성 결과의 자동 후속으로 추가하지 않았다. "
        "논문/PPT 파일 작성이나 자동 commit/push도 이번 실행 범위에 포함하지 않았다. [교수님 설명용 7장안](PROFESSOR_BRIEF.md), "
        "[선택·가지 판단 근거](DECISION_LOG.md).", ""]
    return "\n".join(report)


def professor_text(bundle):
    selected = [chosen_family(bundle,dataset) for dataset in MAIN_DATASETS]
    selected_text = " / ".join(f"{spec['dataset']} {label(spec)} C MSE {fmt(score_row(bundle,spec['id'])['mse'])}" for spec in selected if spec)
    return "\n\n".join([
        "# 교수님 상담 설명안: 고정 Bolt용 시간 앞단의 현재 판단",
        "아래는 7장 설명용 Markdown 구성안이다. 원 논문의 새 성능 재현이나 완성된 논문 기여를 주장하는 PPT 파일이 아니다.",
        "## 1. 문제: 고정 단변량 TSFM에 무엇을 적응시키는가\n\n원채널 미래 정확도가 먼저이며, 작은 K와 실제 전체 비용은 그 이득을 유지할 수 있는지 묻는 조건이다. "
        "입력 복원이 잘된다는 이유로 미래 예측 표현이 충분하다고 가정하지 않는다. 과거만으로 P/E를 만들고 실제 원미래 손실로 학습했다.",
        "## 2. 교수님 제안과 표준 대안\n\nGoogle TSMixer의 시간/채널 혼합, TimeMixer PDM의 분해·다중해상도 혼합을 원격자 과거 앞단으로 변형했다. "
        "AdaPTS식 E/D 예측 학습과 SPRINT식 단순 시간 변환은 선행 대안이다. FMM 대신 Bolt를 붙인 변형이며 원 논문 전체 재현이나 이 원리 자체의 신규성 주장은 없다. "
        "[원문 대응](LITERATURE_MAPPING.md).",
        "## 3. 단순 시간 변환부터\n\n평활·잔여·분리 예측·endpoint/blockmean·SPRINT-on-Bolt·긴 lookback의 원미래 C 오차를 같은 origin의 Raw와 비교했다. "
        "R1 음성 결과로 학습형 경로를 취소하지 않았다. 실제 결과는 [그림1](figures/01_time_transforms.png)과 SCORE 원표에 있다.",
        "## 4. 학습형 P와 C/K\n\n같은 lag7 D와 현재 C anchor에서 B0–B4×C/K×2seed를 실제 학습했다. "
        "B로 선택한 작은 K 후보의 원수치: " + selected_text + ". "
        "[그림2](figures/02_temporal_rank.png)의 두 seed 범위는 신뢰구간이 아니다. "
        "C/K는 stream 수·E/D 파라미터 수·복원 공간을 함께 바꾸므로 압축 단독의 인과 효과가 아니다. "
        "같은 C/r의 B3/B4는 파라미터 구조·개수가 같은 별도 P-family 대조다. 최대 update에서도 하강하면 미수렴·미해결로 남긴다.",
        "## 5. 같은 G와 Bolt/native 대조\n\nRaw+G/B0K+G/B2K+G/B4K+G/G-only의 입력·클래스를 맞췄다. "
        "고정 P/E/D에서 실제 원미래에 맞춘 native144계수와 직접 작은 예측기·self ridge·한 LoRA 구성도 비교했다. "
        "native/ridge는 모든 합법 A origin을 누적하고 신경망은 같은 A 범위의 minibatch prefix여서 실제 표본 노출·optimizer update까지 같은 비교는 아니다. "
        "좋은 full 결합만 보여주지 않고 G-only 및 Raw+G와의 차이를 함께 읽는다. [그림3](figures/03_matched_residual.png).",
        "## 6. 시간 재적합과 전체 비용\n\n기존 FIT 안에서 사전 고정한 두 경계에 모든 upstream을 다시 적합했다. "
        "C는 이번 선택에서 잠갔지만 과거에 미노출인 독립 holdout은 아니다. physical CPU 입력부터 출력 반환까지 batch1/8의 전체 시간·stage·별도 peak 메모리를 측정했다. "
        "[학습/시간 그림](figures/04_learning_time_refit.png), [비용 그림](figures/05_error_time_memory.png).",
        "## 7. 남는 기여와 보류할 주장\n\n이번 답은 표준 대안 대비 추가 가치가 남는 조건을 확인하는 것이다. "
        "원수치가 나쁘거나 미수렴이면 그 상태를 보존한다. G-only나 native/direct가 충분하면 새 앞단·Bolt의 필요성은 보류한다. "
        "재구성/표현/예측기/디코더 중 원인 하나를 자동 확정하지 않고, matched 비교와 시간 refit의 범위 안에서만 좁힌다. "
        "[전체 결과·실패](REPORT_KO.md), [고정/사후 사례](figures/06_forecast_cases.png).",
    ])+"\n"


def decision_text(bundle):
    rows = ["# 이번 캠페인의 판단 근거", "", "이 문서는 비공개 사고일지가 아니라 질문·경쟁 설명·사전 비교·A/B 근거·잠근 선택·C 관찰을 연결한 기록이다.", "",
        "## 실행 전 고정", "", "사용자의 문서 전체 승인 아래 R1–R5를 실제 실행했다. 기존 안전 FIT/로컬 모델만 재사용하고 "
        "기존 과학 결과 cache는 새 입력 적응 결과로 재사용하지 않았다. 첫 fold60/20/20, 둘째50/20/30과 원미래 H48·원 mask·A 통계 MSE를 고정했다. "
        "C로 데이터나 K를 고르지 않았다. R1은 queue 순서상 R2 뒤 실행한 편차를 남긴다.", "",
        "## 무엇을 바꿔 비교했는가", "", "단순 시간 변환 vs Raw; 학습형 P vs B0; C vs K; 같은 G의 다섯 base; "
        "같은 P/E/D의 Bolt vs actual-Y native; direct/LoRA/self ridge; 두 번째 경계 전체 재학습; B4 가지 제거를 각각 구분한다. "
        "R4 ceiling16을 지키기 위한 alpha 축소(swap0/.01, raw ridge .01)는 실제 R4 적합·C 점수 전에 후속 SOURCE에서 고정했다. "
        "G는 primary10/secondary10, 한 primary G seed의 한계를 유지한다. "
        "C/K는 E/D 용량·복원 공간·F stream 수가 동시 변경되는 대조이며, 파라미터 구조가 같은 B3/B4 대조와 구분한다. "
        "C/K의 성능 차이를 정보 압축 하나의 인과 효과로 확정하지 않는다.", "",
        "닫힌형 native/ridge는 all-A sufficient statistics, 신경망은 같은 A 범위에서 최대2400-update schedule의 실제 endpoint prefix다. "
        "허용 기간 일치와 실제 고유 origin/target 노출·optimizer update 일치를 구별하고 R4의 노출 표를 함께 제시한다.", "",
        "## B 선택과 실제 endpoint", ""]
    table = []
    for dataset in MAIN_DATASETS:
        for kind in ("B0","B2","B4"):
            spec = selected_adapter(bundle,dataset,kind,SMALL_K[dataset])
            if spec:
                table.append([dataset,label(spec),spec["id"],fmt(b_value(bundle,spec)),spec.get("updates",0),spec.get("selected_update",0),spec.get("training_status","NA")])
    rows.append(code_table(("dataset","method","B_selected_id","B_MSE","endpoint","selected_CP","training_status"),table))
    rows += ["", "600→1200→2400의 확장은 동일 dataset/r/seed 묶음에서 마지막 CP 최적 또는 최근3점의 엄격 감소에 따른 matched 예산이다. "
             "같은 스케줄 prefix를 쓰며 유한 gradient/검사 통과를 수렴이나 성능 개선으로 바꾸지 않는다.", "",
             "## 선택 잠금 이후 C에서 관찰한 것", "", "\n\n".join(finding_lines(bundle)), "",
             "## 상태와 남는 판단", "", "새 독립 일반화, 충분한 seed의 확증, 모든 방법의 수렴, 표현/예측기/D 중 단일 원인의 확정은 이번 결과로 완료되지 않는다. "
             "미수렴·무효·BLOCKED는 원상태를 유지한다. C의 사후 median/최대악화 origin 선택은 실패 설명용이고 재학습이나 새 선택기로 사용하지 않는다. "
             f"재개가 필요한 항목은 [현재 RUN_INDEX]({bundle['run_index_path'].name})/fit 상태에서 해당 source·checkpoint·허용 범위를 확인해야 하며 기존 결과를 덮지 않는다.", ""]
    return "\n".join(rows)


def main():
    bundle = load_bundle()
    outputs = [HERE/name for name in ("REPORT_KO.md","PROFESSOR_BRIEF.md","DECISION_LOG.md")]
    expected_figures = [HERE/"figures"/f"{name}.{extension}" for name,_ in FIGURES for extension in ("png","svg")]
    if any(path.exists() for path in outputs+expected_figures+[HERE/"figures"/"DATA_PROVENANCE.json"]):
        raise FileExistsError("Report artifacts already exist; preserve them and explicitly version any new rendering")
    (HERE/"figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":9,"legend.fontsize":8,
                         "svg.fonttype":"none","savefig.facecolor":"white"})
    for function in (figure_r1,figure_temporal_rank,figure_residual,figure_learning,figure_cost,figure_cases):
        function(bundle)
    for path,text in zip(outputs,(report_text(bundle),professor_text(bundle),decision_text(bundle))):
        with path.open("x",encoding="utf-8",newline="\n") as handle:
            handle.write(text)
    provenance = {"SCORES_sha256":sha(HERE/"SCORES.csv"),"COST_sha256":sha(HERE/"COST.csv"),
        "SELECTION_LOCK_sha256":sha(HERE/"SELECTION_LOCK.json"),"COST_LOCK_sha256":sha(HERE/"COST_LOCK.json"),
        "generator_sha256":sha(Path(__file__)),"score_rows":len(bundle["scores"]),
        "audit_file":bundle["audit_path"].name,"audit_sha256":sha(bundle["audit_path"]),
        "run_index_file":bundle["run_index_path"].name,"run_index_sha256":sha(bundle["run_index_path"]),
        "audit_manifest_sha256":sha(HERE/"AUDIT_LATEST.json") if bundle["audit_manifest"] is not None else None,
        "selection_rule":"Locked B metrics only; C median/worst cases are explicitly retrospective diagnostics",
        "cases":bundle["case_receipts"],"outputs":[str(path.relative_to(HERE)) for path in outputs+expected_figures]}
    with (HERE/"figures"/"DATA_PROVENANCE.json").open("x",encoding="utf-8") as handle:
        json.dump(provenance,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status":"REPORT_GENERATED","reports":3,"PNG":6,"SVG":6,"C_used_for_new_selection":False},ensure_ascii=False))


if __name__ == "__main__":
    main()
