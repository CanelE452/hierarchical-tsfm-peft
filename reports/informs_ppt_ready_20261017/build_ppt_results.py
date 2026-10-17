"""Build presentation figures from pinned, existing V-only evidence; no models."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


OUTPUT = Path(__file__).resolve().parent
REPO = OUTPUT.parents[1]
COLORS = {"B0": "#667085", "B2": "#0072B2", "Fixed no-Bolt": "#E69F00",
          "Joint no-Bolt": "#009E73", "LoRA": "#CC79A7", "G": "#D55E00", "Raw": "#202020"}
LABELS = {0: "Raw", 1: "Ridge", 2: "Direct G", 3: "PCA K6", 4: "B0", 7: "Fixed no-Bolt",
          8: "Joint no-Bolt", 10: "B2", 11: "B0", 12: "B2", 13: "B0", 14: "B2 (CUDA reference)",
          15: "Raw", 16: "Ridge", 17: "G, 1 epoch", 18: "LoRA", 19: "G, CPU 3 epochs",
          20: "B2", 21: "Fixed no-Bolt", 22: "LoRA", 23: "B4", 24: "AdaPTS LinearPort",
          25: "B2 K11", 26: "B4 K11", 27: "B0 K6", 28: "B2 K6", 38: "Raw", 39: "G"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError("Pinned input changed: " + path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def create_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def scalar(value):
    if value is None or value == "":
        return "NA"
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def cost_value(row, batch):
    cost = row.get("cost_b" + str(batch)) or {}
    return cost.get("seconds") if cost.get("device") == "cpu" and cost.get("threads") == 2 else None


def public_cost(cost, failed_source=None):
    keys = ("status", "seconds", "p90", "device", "threads", "threads_provenance", "warmups", "timed_trials",
            "stage_trials", "physical_input_sha256", "observation_mask_sha256", "scope",
            "physical_output_dtype", "inverse_transform_dtype", "model_loading_in_whole_timer",
            "explicit_same_parent_accuracy_replay_alias", "alias_contract_fields_all_exact", "alias_note")
    result = {key: cost.get(key) for key in keys}
    for key in ("source", "independent_cost_arithmetic_source", "alias_independent_review_source"):
        entry = cost.get(key)
        result[key] = {k: entry[k] for k in ("path", "sha256")} if entry else None
    if failed_source:
        result.update(status="FAILED_RAW_ROW_PARITY_NO_TIMED_COST",
                      failure_source={k: failed_source[k] for k in ("path", "sha256")})
    return result


def public_row(row, table, label):
    result = dict(table_id=table, row_id=row["row_id"], method=label,
                  status="EXISTING_RESULT_REUSED", dataset=row["dataset"], cohort=row["cohort"])
    for key in ("C", "K", "L", "H", "seed", "split", "run_id", "metric_definition", "normalization",
                "accuracy_device", "training_device", "trainable_parameters", "selected_epoch", "search_updates",
                "refit_updates", "training_status", "scaler_sha256", "target_sha256", "mask_sha256",
                "origins_sha256", "channel_sha256", "checkpoint_sha256"):
        result[key] = scalar(row.get(key))
    result.update(MSE=row["mse_text"], MAE=row["mae_text"],
                  source_file=row["score_source"]["path"], source_sha256=row["score_source"]["sha256"])
    index = int(row["row_id"].split("_")[-1])
    result["registered_K"] = scalar(row.get("K"))
    if index in (0, 1, 2, 9, 15, 16, 17, 18, 19, 22, 38, 39):
        result["K"] = "NA"
    result["TSFM_input"] = ("C=" + scalar(row["C"]) if index in (0, 9, 15, 18, 22, 38)
                            else "none" if index in (1, 2, 7, 8, 16, 17, 19, 21, 39)
                            else "K=" + scalar(row.get("K")))
    for batch in (1, 8):
        cost = row.get("cost_b" + str(batch)) or {}
        result["B" + str(batch) + "_seconds"] = scalar(cost_value(row, batch))
        for key in ("status", "device", "threads", "warmups", "timed_trials", "stage_trials"):
            result["B" + str(batch) + "_" + key] = scalar(cost.get(key))
        source = cost.get("source") or {}
        result["B" + str(batch) + "_source_file"] = scalar(source.get("path"))
        result["B" + str(batch) + "_source_sha256"] = scalar(source.get("sha256"))
        for key in ("physical_input_sha256", "observation_mask_sha256", "scope", "threads_provenance",
                    "explicit_same_parent_accuracy_replay_alias", "alias_note"):
            result["B" + str(batch) + "_" + key] = scalar(cost.get(key))
        if index == 0 and row.get("raw_csv_row", {}).get("cost_scope") == "FAILED_RAW_ROW_PARITY_NO_TIMED_COST":
            result["B" + str(batch) + "_status"] = "FAILED_RAW_ROW_PARITY_NO_TIMED_COST"
    result["comparison_note"] = "Repeated development V; limited learning; compare only stated cohort/device/recipe."
    return result


def missing_row(table, dataset, method, C, K, reason):
    return dict(table_id=table, row_id="pending_" + method.replace(" ", "_"), method=method,
                status="PENDING_NOT_A_SCIENTIFIC_FAILURE", dataset=dataset, C=C, K=K, L=512, H=12,
                seed=92601, split="V_select", MSE="NA", MAE="NA", selected_epoch="NA",
                B1_seconds="NA", B8_seconds="NA", reason=reason)


def build_tables(rows):
    r = {int(row["row_id"].split("_")[-1]): row for row in rows}
    tables = []
    groups = {"Temporal_Context_Effect": [4, 10, 11, 12],
              "Frozen_TSFM_Contribution": [10, 7, 8, 20, 21],
              "Overall_Jena": [0, 1, 2, 3, 4, 22, 10],
              "Overall_PeMS_C170": [15, 16, 17, 19, 13, 18, 20],
              "Matched_PeMS_C32": [38, 39],
              "Accuracy_Cost": [4, 10, 7, 8, 22, 13, 20, 21, 19, 38, 39]}
    for table, indices in groups.items():
        tables.extend(public_row(r[i], table, LABELS[i]) for i in indices)
    reason = "Original C32/K8 search stopped before endpoint3; no final refit/full V. Original execution contract expired."
    for method in ("B0", "B2"):
        row = missing_row("Matched_PeMS_C32", "pems08_C32", method, 32, 8, reason)
        row.update(trainable_parameters=2048 if method == "B0" else 37472,
                   search_updates=847 if method == "B0" else 846,
                   training_status="BUDGET_LIMIT_PRESERVED_PARTIAL_SEARCH")
        tables.append(row)
    tables.append(missing_row("Matched_PeMS_C32", "pems08_C32", "Joint no-Bolt", 32, 8,
                              "P1-2 gate not entered: P1-1 benefit is not assessable."))
    tables.append(missing_row("Frozen_TSFM_Contribution", "pems08_C170", "Joint no-Bolt", 170, 43,
                              "No C170 joint no-Bolt result; C32 P1-2 also not entered."))
    tables.append(missing_row("Overall_PeMS_C170", "pems08_C170", "PCA", 170, 43,
                              "No verified existing V-only PCA C170 result in this selected result package."))
    for method in ("B0", "B2"):
        tables.append(missing_row("Temporal_Context_Effect", "pems08_C32", method, 32, 8, reason))
        tables.append(missing_row("Accuracy_Cost", "pems08_C32", method, 32, 8, reason))
    for row in tables:
        if row["table_id"] == "Temporal_Context_Effect" and row["status"] == "EXISTING_RESULT_REUSED":
            seed = int(row["seed"])
            base, proposed = (r[4], r[10]) if seed == 92601 else (r[11], r[12])
            row["B2_minus_B0_MSE_relative_percent"] = format(100 * (proposed["mse"] / base["mse"] - 1), ".17g")
    table_order = {name: i for i, name in enumerate(groups)}
    pems_order = {name: i for i, name in enumerate(["legacy_15", "legacy_16", "legacy_17", "legacy_19", "pending_PCA", "legacy_13", "legacy_18", "legacy_20"])}
    tables.sort(key=lambda row: (table_order[row["table_id"]],
                                 pems_order[row["row_id"]] if row["table_id"] == "Overall_PeMS_C170" else 0))
    appendix_ids = [5, 6, 9, 23, 24, 25, 26, 27, 28, *range(29, 38)]
    appendix = [public_row(r[i], "Appendix", LABELS.get(i, r[i]["method"])) for i in appendix_ids]
    write_csv(OUTPUT / "PPT_MAIN_TABLES.csv", tables)
    write_csv(OUTPUT / "PPT_ABLATION_TABLES.csv", appendix)
    return r, tables, appendix


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 17,
                         "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    return plt


def bars(ax, values, labels, colors, metric):
    for i, (value, label, color) in enumerate(zip(values, labels, colors)):
        if value is None:
            ax.text(i, .13, "Not available", ha="center", va="center", transform=ax.get_xaxis_transform(),
                    fontsize=12, color="#667085")
        else:
            ax.bar(i, value, color=color, width=.58)
            ax.text(i, value, format(value, ".6f"), ha="center", va="bottom", fontsize=12)
    finite = [v for v in values if v is not None]
    ax.set_ylim(0, max(finite) * 1.18)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-.6, len(labels) - .4)
    ax.set_ylabel(metric + " (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)


def save(fig, name, previews):
    target = OUTPUT / "figures" / (name + ".svg")
    if target.exists():
        raise FileExistsError(target.name)
    fig.savefig(target, format="svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(previews / (name + ".png"), dpi=110, bbox_inches="tight")


def temporal_figure(plt, r, previews):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5))
    for ax, metric in zip(axes, ("mse", "mae")):
        for j, (seed, b, p) in enumerate(((92601, 4, 10), (92602, 11, 12))):
            for x, idx, label in ((j - .18, b, "B0"), (j + .18, p, "B2")):
                v = r[idx][metric]
                ax.bar(x, v, width=.31, color=COLORS[label], label=label if j == 0 else None)
                ax.text(x, v, f"{v:.6f}", ha="center", va="bottom", fontsize=12)
        ax.set_xticks([0, 1], ["Seed 92601", "Seed 92602"])
        ax.set_ylabel(metric.upper() + " (A60 standardized channels)")
        ax.set_ylim(0, max(r[i][metric] for i in (4, 10, 11, 12)) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.suptitle("Jena: B0 vs B2 in two optimization seeds", fontsize=21, fontweight="bold")
    fig.text(.5, .03, "C21 / K6 / L512 / H48 / repeated development V43. Selected refit: B0 3 epochs, B2 1 epoch.\n"
             "Capacity and selected training differ. Two seeds are not independent datasets or a significance test.",
             ha="center", fontsize=12)
    fig.tight_layout(rect=(0, .13, 1, .92)); save(fig, "JENA_B0_B2_TWO_SEEDS", previews); plt.close(fig)


def bolt_figure(plt, r, previews):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5))
    labels = ["B2 + Bolt", "Fixed no-Bolt", "Joint no-Bolt"]
    colors = [COLORS["B2"], COLORS["Fixed no-Bolt"], COLORS["Joint no-Bolt"]]
    for ax, title, indices in ((axes[0], "Jena C21 / K6 / V43", [10, 7, 8]),
                               (axes[1], "PeMS08 C170 / K43 / V148", [20, 21, None])):
        bars(ax, [r[i]["mse"] if i is not None else None for i in indices], labels, colors, "MSE")
        ax.set_title(title)
    fig.suptitle("Frozen Bolt contribution: limited full-history linear controls", fontsize=21, fontweight="bold")
    fig.text(.5, .025, "Fixed: same final P/E/D + full 512-step latent history. Joint: fresh no-Bolt objective, same architecture.\n"
             "PeMS has a fixed head only; parent limited to 1 epoch, added head 3 epochs. No universal Bolt-necessity claim.\n"
             "C170 results do not fill the new C32/K8 matched experiment.", ha="center", fontsize=12)
    fig.tight_layout(rect=(0, .17, 1, .92)); save(fig, "BOLT_VS_NO_BOLT_TWO_DATASETS", previews); plt.close(fig)


def accuracy_figure(plt, r, previews):
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8))
    groups = [("Jena C21 / V43", [0, 1, 2, 3, 4, 22, 10]),
              ("PeMS08 C170 / V148", [15, 16, 17, 19, None, 13, 18, 20])]
    for ax, (title, indices) in zip(axes, groups):
        ax.axis("off"); ax.set_title(title, pad=16, fontweight="bold")
        cells = [[LABELS[i], f"{r[i]['mse']:.6f}", f"{r[i]['mae']:.6f}", str(r[i].get("accuracy_device") or "NA")]
                 if i is not None else ["PCA", "NA", "NA", "Not verified"] for i in indices]
        table = ax.table(cellText=cells, colLabels=["Method", "MSE", "MAE", "Eval device"],
                         cellLoc="center", colWidths=[.35, .23, .23, .19], bbox=[0, .06, 1, .87])
        table.auto_set_font_size(False); table.set_fontsize(13)
        for (rr, cc), cell in table.get_celld().items():
            cell.set_edgecolor("#D0D5DD")
            if rr == 0: cell.set_facecolor("#EEF2F6"); cell.set_text_props(weight="bold")
            if rr == len(indices): cell.set_facecolor("#E8F3FB")
    fig.suptitle("Existing forecasting results by dataset and evaluation device", fontsize=21, fontweight="bold")
    fig.text(.5, .06, "All errors are A60-standardized original-channel macro metrics on repeatedly used development V.\n"
             "Jena LoRA = new CPU family run; PeMS rows have different training budgets/devices (see CSV and notes).\n"
             "P1-1 matched PeMS C32/K8: Raw and G available; B0/B2 final scores remain NA. PCA C170 score unavailable.",
             ha="center", fontsize=12)
    fig.tight_layout(rect=(0, .17, 1, .91)); save(fig, "MAIN_ACCURACY_TABLE", previews); plt.close(fig)


def cost_figure(plt, r, previews):
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 7))
    panels = [("Jena C21 / V43", [4, 10, 7, 8, 22]),
              ("PeMS08 C170 / V148", [13, 20, 21, 19]),
              ("PeMS08 C32 / V148", [38, 39])]
    pending = {}
    for ax, (title, ids) in zip(axes, panels):
        omitted = []
        for i in ids:
            cost = cost_value(r[i], 8)
            if cost is None: omitted.append(LABELS[i]); continue
            label = LABELS[i]
            color = COLORS.get(label, COLORS.get("G" if "G," in label else label, "#667085"))
            ax.scatter(cost, r[i]["mse"], s=95, color=color, edgecolor="white", zorder=3)
            ax.annotate(label, (cost, r[i]["mse"]), xytext=(5, 7), textcoords="offset points", fontsize=11)
        ax.set_xscale("log"); ax.set_title(title)
        ax.set_xlabel("CPU2 B8 warm whole (seconds/batch, log)")
        ax.set_ylabel("MSE (lower is better)")
        ax.grid(alpha=.15); ax.spines[["top", "right"]].set_visible(False)
        pending[title] = omitted
        if omitted: ax.text(.02, .02, "Not measured/proven: " + ", ".join(omitted), transform=ax.transAxes, fontsize=10, wrap=True)
    fig.suptitle("Accuracy–cost evidence: separate cohorts and explicit missing cells", fontsize=21, fontweight="bold")
    fig.text(.5, .025, "Batch latency is not per-series latency. Existing CPU2 physical input → preprocessing → model → inverse → CPU return.\n"
             "10 warmup / 20 timed / 20 separate stage trials; model cold load and training cost are separate. No invented zero costs.\n"
             "Different C, recipes and selected epochs prevent a single global Pareto ranking. C32 B0/B2/no-Bolt are still unmeasured.",
             ha="center", fontsize=12)
    fig.tight_layout(rect=(0, .17, 1, .91)); save(fig, "ACCURACY_COST_TRADEOFF", previews); plt.close(fig)
    return pending


def qualitative_figure(plt, curves, metadata, previews):
    import numpy as np
    fig, axes = plt.subplots(3, 6, figsize=(25, 13), squeeze=False)
    methods = {"B0": (COLORS["B0"], "--"), "B2": (COLORS["B2"], "-"),
               "Joint no-Bolt": (COLORS["Joint no-Bolt"], "-."), "Fixed no-Bolt": (COLORS["Fixed no-Bolt"], "-."),
               "LoRA": (COLORS["LoRA"], ":"), "G": (COLORS["G"], ":"), "Truth": ("#111111", "-")}
    legend = {}
    for di, dataset in enumerate(("jena", "pems08")):
        panels = metadata["datasets"][dataset]["panels"]
        for panel in panels:
            ax = axes[panel["origin_slot"], di * 3 + panel["channel_slot"]]
            subset = [r for r in curves if r["dataset"] == dataset and int(r["origin_id"]) == panel["origin_id"] and r["channel_id"] == panel["channel_id"]]
            for method in ("B0", "B2", "Joint no-Bolt", "Fixed no-Bolt", "LoRA", "G", "Truth"):
                points = sorted((r for r in subset if r["method"] == method), key=lambda r: int(r["horizon_step"]))
                if not points: continue
                y = [float(r["value"]) if r["observed"] == "1" else np.nan for r in points]
                color, line = methods[method]
                artist, = ax.plot([int(r["horizon_step"]) for r in points], y, color=color, ls=line,
                                  lw=2.1 if method == "Truth" else 1.5, alpha=.9, label=method)
                legend[method] = artist
            ax.set_title("origin " + str(panel["origin_id"]) + " · " + panel["channel_id"], fontsize=12)
            ax.set_xlabel("Future horizon step"); ax.set_ylabel("A60 z-score")
            ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.12)
    fig.suptitle("Qualitative forecasts: fixed origins and channels, selected before reading errors", fontsize=23, fontweight="bold")
    fig.text(.26, .94, "Jena C21 / H48 / V43", ha="center", fontsize=18, fontweight="bold")
    fig.text(.74, .94, "PeMS08 C170 / H12 / V148", ha="center", fontsize=18, fontweight="bold")
    fig.legend(list(legend.values()), list(legend), loc="lower center", bbox_to_anchor=(.5, .055), ncol=len(legend), frameon=False)
    fig.text(.5, .018, "First / middle / last shared V origin and channel; all H steps. Observed-mask gaps are preserved.\n"
             "Jena: CPU B0/B2 + joint no-Bolt + CPU LoRA. PeMS: CUDA B0, CPU replay of CUDA-trained B2, fixed head, CPU G.\n"
             "Training budgets/devices differ. Normalized curves, not physical units. These C170 panels do not complete P1-1 C32.",
             ha="center", fontsize=12)
    fig.tight_layout(rect=(0, .12, 1, .91)); save(fig, "QUALITATIVE_FORECASTS", previews); plt.close(fig)


def execute(args):
    started = time.time()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "2"
    evidence = read_bound(args.evidence, args.evidence_sha)
    scope = read_bound(args.scope, args.scope_sha)
    qualitative = read_bound(args.qualitative_metadata, args.qualitative_metadata_sha)
    if time.time() >= scope["package_deadline_unix"]: raise TimeoutError("Package wall boundary")
    forbidden = ["PPT_MAIN_TABLES.csv", "PPT_ABLATION_TABLES.csv", "SOURCE_BINDINGS.json", "FIGURE_BUILD_VALIDATION.json"]
    if any((OUTPUT / n).exists() for n in forbidden): raise FileExistsError("Preserve prior generated outputs")
    rows = evidence["rows"]
    for row in rows:
        if row["split"] != "V_select" or not math.isfinite(row["mse"]) or not math.isfinite(row["mae"]):
            raise ValueError("Only explicit existing V scores are allowed")
        if float(row["mse_text"]) != row["mse"] or float(row["mae_text"]) != row["mae"]:
            raise ValueError("Exact original score text must be retained")
        if row["mse"] < 0 or row["mae"] < 0 or row["mae"] ** 2 > row["mse"] + 1e-10:
            raise ValueError("MSE/MAE scale or nonnegative error consistency failed")
        for name in ("score_source", "manifest_source", "training_receipt_source"):
            source = row.get(name)
            if source and sha(REPO / source["path"]) != source["sha256"]: raise ValueError("Source changed")
    r, main, appendix = build_tables(rows)
    plt = style(); previews = Path(args.previews); previews.mkdir(exist_ok=True)
    temporal_figure(plt, r, previews); bolt_figure(plt, r, previews); accuracy_figure(plt, r, previews)
    pending = cost_figure(plt, r, previews)
    with Path(args.qualitative_csv).open(encoding="utf-8", newline="") as stream: curves = list(csv.DictReader(stream))
    if sha(args.qualitative_csv) != qualitative["curves_csv_sha256"]: raise ValueError("Selected curves changed")
    qualitative_figure(plt, curves, qualitative, previews)
    bindings = []
    for row in rows:
        record = dict(row_id=row["row_id"], run_id=row["run_id"], cohort=row["cohort"], source=row["score_source"]["path"],
                      source_sha256=row["score_source"]["sha256"], checkpoint_sha256=row.get("checkpoint_sha256"),
                      scaler_sha256=row.get("scaler_sha256"), target_sha256=row.get("target_sha256"),
                      mask_sha256=row.get("mask_sha256"), origins_sha256=row.get("origins_sha256"),
                      channel_sha256=row.get("channel_sha256"), validation=row.get("csv_vs_V_manifest_score_exact"))
        failed = row["score_source"] if row.get("raw_csv_row", {}).get("cost_scope") == "FAILED_RAW_ROW_PARITY_NO_TIMED_COST" else None
        record.update(cost_b1=public_cost(row.get("cost_b1") or {}, failed), cost_b8=public_cost(row.get("cost_b8") or {}, failed))
        bindings.append(record)
    create_json(OUTPUT / "SOURCE_BINDINGS.json", dict(schema="PPT_SOURCE_BINDINGS_v1", source_rows=bindings,
                evidence_sha256=args.evidence_sha, scope_sha256=args.scope_sha,
                qualitative_metadata_sha256=args.qualitative_metadata_sha, new_training=0, new_model_inference=0,
                new_cost_measurement=0, protected_T_numeric_access=0, whole_MASTER_complete=False))
    import psutil
    memory = psutil.Process().memory_info()
    rss = memory.rss
    peak_rss = getattr(memory, "peak_wset", rss)
    if peak_rss >= scope["maximum_RSS_bytes"] or time.time() >= scope["package_deadline_unix"]:
        raise RuntimeError("Package resource boundary exceeded")
    images = [dict(file=p.relative_to(OUTPUT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted((OUTPUT / "figures").glob("*.svg"))]
    if len(images) != 5: raise ValueError("Exactly five requested figures required")
    value = dict(status="FIGURES_AND_TABLES_BUILT_FROM_PINNED_EXISTING_V_EVIDENCE", main_rows=len(main), appendix_rows=len(appendix),
                 reconciled_sources=len(rows), cost_pending_by_panel=pending, qualitative_panels=18,
                 selected_curve_rows=len(curves), figures=images, current_RSS_bytes=rss, OS_peak_RSS_bytes=peak_rss,
                 wall_seconds=time.time()-started, torch_imported="torch" in sys.modules,
                 new_training=0, new_model_inference=0, protected_T_numeric_access=0, scientific_pipeline_rerun=False,
                 independent_generalization_proven=False, whole_MASTER_complete=False)
    create_json(OUTPUT / "FIGURE_BUILD_VALIDATION.json", value)
    print(json.dumps(value, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("evidence", "evidence-sha", "scope", "scope-sha", "qualitative-metadata", "qualitative-metadata-sha", "qualitative-csv", "previews"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="SOURCE_ONLY_NO_INPUT_READS", model_calls=0, writes=0))); return
    if any(getattr(args, name) is None for name in ("evidence", "evidence_sha", "scope", "scope_sha", "qualitative_metadata", "qualitative_metadata_sha", "qualitative_csv", "previews")):
        parser.error("Pinned source evidence, fixed V selection, scope and private preview directory required")
    execute(args)


if __name__ == "__main__":
    main()
