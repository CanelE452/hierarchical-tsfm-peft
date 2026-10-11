"""Aggregate completed development observations; never run or select a model."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTRACT = "FORECAST_RIDGE_D8"
COLORS = {"RAW_F0": "#66717e", "RAW_C_D8": "#191f28", "PCA": "#286ea5",
          "MLP_POINT": "#cd772b", "TS2VEC_CAUSAL_SMALL": "#278a70"}
NAMES = {"RAW_F0": "Raw F", "RAW_C_D8": "Raw + D8", "PCA": "PCA",
         "MLP_POINT": "Point MLP", "TS2VEC_CAUSAL_SMALL": "TS2Vec small"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def averaged_metrics(metrics):
    return {group: {key: float(np.mean([row[group][key] for row in metrics]))
                    for key in ("mse", "mae")} for group in metrics[0]}


def cost_summary(rows):
    return {"case_count": len(rows),
            "median_ms": float(np.mean([row["median_seconds"] * 1000 for row in rows])),
            "p95_ms": float(np.mean([row["p95_seconds"] * 1000 for row in rows])),
            "allocated_mib": float(np.mean([row["cuda_peak_allocated_bytes"] / 2**20 for row in rows])),
            "reserved_mib": float(np.mean([row["cuda_peak_reserved_bytes"] / 2**20 for row in rows])),
            "sampled_rss_mib": float(np.mean([row["rss"]["peak_rss_bytes"] / 2**20 for row in rows]))}


def pareto_ids(conditions, field):
    ids = []
    for row in conditions:
        x, y = row["cost"]["32"][field], row["metrics"]["all"]["mse"]
        dominated = any(other["cost"]["32"][field] <= x and other["metrics"]["all"]["mse"] <= y
                        and (other["cost"]["32"][field] < x or other["metrics"]["all"]["mse"] < y)
                        for other in conditions)
        if not dominated:
            ids.append(row["id"])
    return ids


def readout_diagnostics(data, dataset):
    import sys
    sys.path.insert(0, str(HERE))
    import sweep
    bundle = sweep.load_original_bundle(dataset)
    name = next(name for name, row in data["decoder_contracts"].items() if row["meta"]["family"] == "RAW_C_D8")
    contract = data["decoder_contracts"][name]["contracts"][CONTRACT]
    path = sweep.resolve_path(contract["prediction_npz"])
    if sweep.sha256_path(path) != contract["prediction_sha256"]:
        raise RuntimeError("Raw readout trace changed")
    with np.load(path, allow_pickle=False) as archive:
        train_origins = archive["train_origins"]
        train_forecast = archive["train_latent_forecast"]
        val_forecast = archive["latent_forecast"]
        train_features = sweep.future_features(archive["train_latent_history"], train_forecast, 8)
        val_features = sweep.future_features(archive["val_latent_history_tail"], val_forecast, 8)
        coefficients = archive["grid_coefficients"]
    pilot = sweep.parent_runner().old_pilot()
    train_target, train_mask = pilot.future_targets(bundle, train_origins)
    val_target, val_mask = pilot.future_targets(bundle, bundle.val_origins)
    metric = sweep.parent_runner().masked_channel_macro_metrics_np
    with np.load(sweep.resolve_path(data["raw_f0"]["prediction_npz"]), allow_pickle=False) as archive:
        expected_raw = archive["prediction_tsfm"]
    identity_parity = bool(np.array_equal(val_forecast, expected_raw))
    if not identity_parity:
        raise RuntimeError("Identity raw readout did not match stored RAW_F0")
    curve = []
    for index, choice in enumerate(contract["choices"]):
        train_metrics = metric(sweep.apply_coeff(train_features, coefficients[index]), train_target, train_mask)
        val_metrics = metric(sweep.apply_coeff(val_features, coefficients[index]), val_target, val_mask)
        curve.append({"lambda": choice["lambda"],
                      "train": {key: train_metrics[key] for key in ("mse", "mae")},
                      "val": {key: val_metrics[key] for key in ("mse", "mae")}})
    train_identity = metric(train_forecast, train_target, train_mask)
    val_identity = metric(val_forecast, val_target, val_mask)
    return {"identity_matches_RAW_F0": identity_parity,
            "identity_train": {key: train_identity[key] for key in ("mse", "mae")},
            "identity_val": {key: val_identity[key] for key in ("mse", "mae")},
            "selected_lambda": contract["selected_lambda"], "selected": curve[contract["selected_index"]],
            "curve": curve,
            "scope": "Saved raw-C forecasts; diagnostic evaluation of existing coefficients only, no new fit/F call; distinguishes TRAIN/VAL readout behavior without identifying its cause"}


def build_summary(result, method, benchmark):
    if result["status"] != "complete" or method["status"] != "complete":
        raise RuntimeError("Both science stages must be complete")
    summary = {"scope": "Repeated exploratory VAL; two-seed mean errors, no prediction ensemble; K/regularization/method choices have no independent TEST confirmation",
               "inputs": {name: sha(HERE / name) for name in ("result.json", "method_result.json", "benchmark.json")},
               "execution": {"baseline": result["execution_counts"], "method": method["execution_counts"]},
               "datasets": {}}
    for dataset, data in result["datasets"].items():
        channels = data["channels"]
        raw = data["raw_f0"]["future_tsfm"]
        grouped = {}
        for source in (data, method["datasets"][dataset]):
            for adapter, entry in source["decoder_contracts"].items():
                meta = entry["meta"]
                key = (meta["family"], int(meta["k"]))
                grouped.setdefault(key, []).append({"adapter": adapter, "meta": meta,
                                                     "contract": entry["contracts"][CONTRACT],
                                                     "fit": source.get("fits", {}).get(adapter)})
        conditions = [{"id": "RAW_F0", "family": "RAW_F0", "k": channels, "k_over_c": 1.0,
                       "adapters": ["RAW_F0"], "metrics": raw, "seed_results": [],
                       "ridge_curve": [], "initial_metrics": raw, "decoder_params": 0,
                       "encoder_weight_count": 0, "encoder_resident_weight_count": 0, "fit_costs": []}]
        for (family, k), entries in grouped.items():
            metrics = averaged_metrics([item["contract"]["future_tsfm"] for item in entries])
            initial = averaged_metrics([item["contract"]["initial_grid_selection"]["selection_metrics"] for item in entries])
            curve = []
            for index, lam in enumerate(result["protocol"]["ridge_lambdas"]):
                values = [item["contract"]["choices"][index] for item in entries]
                if not all(value["lambda"] == lam for value in values):
                    raise RuntimeError("Mismatched common lambda order")
                curve.append({"lambda": lam, "metrics": averaged_metrics([value["metrics"] for value in values])})
            conditions.append({"id": f"{family}_K{k}", "family": family, "k": k, "k_over_c": k/channels,
                               "adapters": [item["adapter"] for item in entries], "metrics": metrics,
                               "initial_metrics": initial, "ridge_curve": curve, "decoder_params": 9*k*channels,
                               "seed_results": [{"seed": item["meta"]["seed"], "adapter": item["adapter"],
                                                 "metrics": item["contract"]["future_tsfm"],
                                                 "selected_lambda": item["contract"]["selected_lambda"],
                                                 "initial_lambda": item["contract"]["initial_grid_selection"]["selected_lambda"]}
                                                for item in entries]})
            fits = [item["fit"] for item in entries if item["fit"] is not None]
            if family == "PCA":
                encoder_weights = channels*k
            elif fits:
                encoder_weights = int(np.mean([fit["encoder_params"] for fit in fits]))
            else:
                encoder_weights = 0
            conditions[-1]["encoder_weight_count"] = encoder_weights
            conditions[-1]["encoder_resident_weight_count"] = int(np.mean([fit.get("resident_params_total", fit["encoder_params"]) for fit in fits])) if fits else encoder_weights
            conditions[-1]["fit_costs"] = [{"adapter": item["adapter"], "seed": item["meta"]["seed"],
                                           "reused_from_parent": bool(item["fit"].get("reused_from_parent", False)),
                                           "fresh_fit_wall_seconds": item["fit"].get("wall_seconds"),
                                           "actual_optimizer_steps": item["fit"].get("optimizer_steps_completed", item["fit"].get("actual_optimizer_steps")),
                                           "selected_epoch": item["fit"].get("best_epoch"),
                                           "wall_scope": item["fit"].get("wall_scope", "native fit wall; window-pool materialization and export excluded")}
                                          for item in entries if item["fit"] is not None]
        family_order = list(NAMES)
        conditions.sort(key=lambda row: (family_order.index(row["family"]), row["k"]))
        raw_d = next(row for row in conditions if row["family"] == "RAW_C_D8")
        for row in conditions:
            row["cost"] = {}
            for batch in (1, 32):
                rows = [case for case in benchmark["rows"] if case["dataset"] == dataset
                        and case["adapter"] in row["adapters"] and case["batch"] == batch]
                if len(rows) != len(row["adapters"]):
                    raise RuntimeError(f"Missing/duplicate cost rows: {dataset} {row['id']} B{batch}")
                row["cost"][str(batch)] = cost_summary(rows)
            for metric in ("mse", "mae"):
                row[f"{metric}_change_vs_raw_pct"] = 100*(row["metrics"]["all"][metric]/raw["all"][metric]-1)
                row[f"{metric}_change_vs_raw_d8_pct"] = 100*(row["metrics"]["all"][metric]/raw_d["metrics"]["all"][metric]-1)
            row["mse_reduction_initial_to_final_pct"] = 100*(1-row["metrics"]["all"]["mse"]/row["initial_metrics"]["all"]["mse"])
        raw_condition = conditions[0]
        for row in conditions:
            for batch, cost in row["cost"].items():
                baseline = raw_condition["cost"][batch]
                cost["speedup_vs_raw"] = baseline["median_ms"]/cost["median_ms"]
                cost["allocated_reduction_vs_raw_pct"] = 100*(1-cost["allocated_mib"]/baseline["allocated_mib"])
        allowance = {}
        for reference in (raw_condition, raw_d):
            selections = {}
            for pct in result["protocol"]["error_allowances_pct"]:
                bound = reference["metrics"]["all"]["mse"]*(1+pct/100)
                feasible = [row for row in conditions if row["k"] < channels and row["metrics"]["all"]["mse"] <= bound]
                selections[str(pct)] = {"mse_bound": bound, "feasible_compressed_ids": [row["id"] for row in feasible],
                                        "fastest_compressed_B32": min(feasible, key=lambda row: row["cost"]["32"]["median_ms"])["id"] if feasible else None}
            allowance[reference["id"]] = selections
        summary["datasets"][dataset] = {"channels": channels, "middle_k": data["middle_k"],
                                         "conditions": conditions, "mse_allowance_sensitivity": allowance,
                                         "B32_mse_latency_pareto": pareto_ids(conditions, "median_ms"),
                                         "B32_mse_allocated_pareto": pareto_ids(conditions, "allocated_mib")}
        summary["datasets"][dataset]["raw_readout_diagnostic"] = readout_diagnostics(data, dataset)
    return summary


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e4e7ea", linewidth=.7)
    ax.set_axisbelow(True)


def save(fig, name):
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot(summary):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), layout="constrained")
    for ax, (dataset, ds) in zip(axes, summary["datasets"].items()):
        for family in NAMES:
            rows = sorted((r for r in ds["conditions"] if r["family"] == family), key=lambda row: row["k"])
            marker = "*" if family == "TS2VEC_CAUSAL_SMALL" else ("s" if family.startswith("RAW") else "o")
            ax.plot([r["cost"]["32"]["median_ms"] for r in rows], [r["metrics"]["all"]["mse"] for r in rows],
                    marker=marker, markersize=9 if marker == "*" else 6, color=COLORS[family], label=NAMES[family])
            for row in rows:
                if row["k"] < ds["channels"]:
                    ax.annotate(f"K{row['k']}", (row["cost"]["32"]["median_ms"], row["metrics"]["all"]["mse"]),
                                xytext=(4, 5), textcoords="offset points", fontsize=8, color=COLORS[family])
        ax.set(xlabel="Full pipeline median latency, B32 (ms)", ylabel="Development VAL MSE",
               title=f"{dataset.title()} · C={ds['channels']}")
        style(ax)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen TSFM MSE–latency tradeoff · lower left is better", fontsize=14)
    save(fig, "accuracy_cost")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), layout="constrained")
    for col, (dataset, ds) in enumerate(summary["datasets"].items()):
        for row in ds["conditions"]:
            family, k = row["family"], row["k"]
            if family not in ("PCA", "MLP_POINT", "TS2VEC_CAUSAL_SMALL"):
                continue
            marker = "*" if family == "TS2VEC_CAUSAL_SMALL" else ("o" if family == "PCA" else "^")
            for index, batch in enumerate(("1", "32")):
                axes[index, col].scatter(k, row["cost"][batch]["median_ms"], color=COLORS[family],
                                         marker=marker, s=90 if marker == "*" else 55, label=NAMES[family])
        for index, batch in enumerate(("1", "32")):
            ax = axes[index, col]
            for family in ("RAW_F0", "RAW_C_D8"):
                row = next(row for row in ds["conditions"] if row["family"] == family)
                ax.axhline(row["cost"][batch]["median_ms"], color=COLORS[family], ls="--", label=NAMES[family])
            ax.set(xlabel="Latent channel count K", ylabel="Full pipeline median latency (ms)", title=f"{dataset.title()} · B{batch}")
            style(ax)
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ordered_labels = [name for name in NAMES.values() if name in unique]
            ax.legend([unique[name] for name in ordered_labels], ordered_labels, fontsize=8)
    save(fig, "batch_cost")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    for ax, (dataset, ds) in zip(axes, summary["datasets"].items()):
        for row in ds["conditions"]:
            if not row["ridge_curve"]:
                continue
            family = row["family"]
            ax.plot([p["lambda"] for p in row["ridge_curve"]], [p["metrics"]["all"]["mse"] for p in row["ridge_curve"]],
                    color=COLORS[family], alpha=.65, marker=".", ls="--" if row["k"] != ds["middle_k"] else "-",
                    label=f"{NAMES[family]} K{row['k']}")
        ax.set(xscale="log", xlabel="Ridge lambda · sum SSE / TRAIN RMS", ylabel="Development VAL MSE", title=dataset.title())
        style(ax)
        ax.legend(fontsize=7, ncol=2)
    save(fig, "ridge_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    for ax, (dataset, ds) in zip(axes, summary["datasets"].items()):
        diag = ds["raw_readout_diagnostic"]
        xs = [row["lambda"] for row in diag["curve"]]
        for split, color in (("train", "#286ea5"), ("val", "#cd772b")):
            ax.plot(xs, [row[split]["mse"] for row in diag["curve"]], color=color, marker=".", label=f"D8 {split.upper()}")
            ax.axhline(diag[f"identity_{split}"]["mse"], color=color, ls="--", label=f"Identity {split.upper()}")
        ax.axvline(diag["selected_lambda"], color="#777777", ls=":", lw=1, label="VAL-selected lambda")
        ax.set(xscale="log", xlabel="Ridge lambda", ylabel="MSE", title=f"{dataset.title()} · raw channels only")
        style(ax)
        ax.legend(fontsize=8)
    fig.suptitle("Readout generalization diagnostic · saved forecasts / no new fit", fontsize=13)
    save(fig, "readout_generalization")


if __name__ == "__main__":
    summary = build_summary(read(HERE / "result.json"), read(HERE / "method_result.json"), read(HERE / "benchmark.json"))
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(summary)
    print(json.dumps({dataset: [{"id": row["id"], "mse": row["metrics"]["all"]["mse"], "mae": row["metrics"]["all"]["mae"],
                                "B1_ms": row["cost"]["1"]["median_ms"], "B32_ms": row["cost"]["32"]["median_ms"],
                                "B32_allocated_mib": row["cost"]["32"]["allocated_mib"]}
                               for row in data["conditions"]] for dataset, data in summary["datasets"].items()}, indent=2))
