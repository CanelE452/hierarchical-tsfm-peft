"""Immutable, CPU-only figures from independently checked prediction artifacts."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import audit_predictions as independent
import data_contract as data
import training


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#333333")
GROUPS = ("accuracy_by_k", "matched_controls", "learning_diagnostics", "cost_accuracy", "lag_prefix_multik", "representative_traces")
REFERENCES = {"CURRENT", "SEASONAL_NAIVE", "RAW_RIDGE", "DIRECT_G", "LORA", "LORA_R8", "RAW", "RAW_BOLT_SMALL", "RAW_BOLT_MINI", "RAW_BOLT_TINY"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "xt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or path == root or not path.is_relative_to(root):
        raise ValueError("Input relative path escapes its declared root")
    return path


def _name(text):
    return "".join(char if char.isascii() and (char.isalnum() or char in "_.-") else "_" for char in str(text))


def curve_identity(row):
    return tuple(row.get(key) for key in ("dataset", "fold", "cohort", "family", "seed", "budget", "cohort_arrays_sha"))


def _input(inputs, path, root, kind):
    relative = path.resolve().relative_to(Path(root).resolve()).as_posix()
    inputs[(kind, relative)] = dict(scope=kind, file=relative, sha256=data.sha256_file(path), bytes=path.stat().st_size)


def _fits(public, registry, inputs):
    result = {}
    for path in sorted((public / "runs").glob("*/*FIT_RECEIPT.json")):
        receipt = _json(path)
        run_id = receipt.get("run_id")
        if run_id not in registry or receipt.get("status") != "COMPLETE_VALID":
            continue
        _input(inputs, path, public, "public")
        result[run_id] = dict(receipt=receipt, folder=path.parent,
            budget=str(float(max(receipt["epochs"]))) if receipt.get("epochs") else "ridge_closed_form",
            selected_epoch=receipt.get("selected_epoch"), training_status=receipt.get("training_status", "CLOSED_FORM"),
            last_cp_best=receipt.get("last_cp_best", False), row=registry[run_id])
    for path in sorted((public / "combinations").glob("*/receipt.json")):
        receipt = _json(path)
        run_id = receipt.get("run_id")
        if run_id not in registry or receipt.get("execution_status") != "COMPLETE_VALID":
            continue
        _input(inputs, path, public, "public")
        epochs = receipt.get("registered_epoch_endpoints", [])
        result[run_id] = dict(receipt=receipt, folder=path.parent,
            budget="Mcal:" + (str(float(max(epochs))) if epochs else "closed_form"),
            selected_epoch=None, training_status=receipt.get("training_status", "M_CAL_CALIBRATION"),
            last_cp_best=receipt.get("selected_updates") == receipt.get("updates"), row=registry[run_id])
    return result


def _fit_for(run_id, manifest, fits):
    meta = manifest.get("metadata", {})
    parent = meta.get("parent_run_id", meta.get("family_run_id", meta.get("route_spec", {}).get("source_parent_run_id")))
    return fits.get(run_id, fits.get(parent))


def _t_allowed(lock, manifest):
    run_id, meta = manifest["run_id"], manifest.get("metadata", {})
    route = meta.get("route_spec")
    if route is not None and lock.get("allowed_route_sha", {}).get(run_id) == training.fingerprint(route):
        return True
    artifact = meta.get("artifact_sha256", meta.get("realized_model_binding", {}).get("selected_artifact_sha256"))
    return artifact is not None and run_id in lock.get("allowed_run_ids", []) and lock.get("allowed_artifact_sha", {}).get(run_id) == artifact


def _predictions(public, private, split, registry, fits, inputs, lock):
    records, excluded = [], []
    seen = set()
    for path in sorted(public.rglob("PREDICTION_MANIFEST.json")):
        if "figures" in path.relative_to(public).parts:
            continue
        try:
            manifest = _json(path)
            if manifest.get("split") != split or manifest.get("score") is None:
                continue
            run_id = manifest["run_id"]
            if run_id not in registry:
                raise ValueError("Prediction has no registered experiment ID")
            if split == "T_final" and not _t_allowed(lock, manifest):
                raise ValueError("T_final run/artifact/route is absent from SELECTION_LOCK")
            reference_path = path.with_name("REFERENCE_CONTRACT.json")
            reference = _json(reference_path)
            for key in independent.IDENTITY_KEYS:
                if manifest.get(key) != reference.get(key) or not str(manifest.get(key, "")):
                    raise ValueError("Independent reference identity differs: " + key)
            arrays = independent._load_arrays(manifest, reference, private)
            expected, rescored = independent._independent_statistics(arrays, manifest)
            statistics_path = _inside(public, manifest["public_stats_relative_path"])
            with gzip.open(statistics_path, "rt", encoding="utf-8", newline="") as stream:
                reaggregated = independent._reaggregate(list(csv.DictReader(stream)), expected, manifest["score"])
            for key, value in rescored.items():
                if key in manifest["score"]:
                    independent._same_number(manifest["score"][key], value, key)
            for key in ("mse", "mae"):
                independent._same_number(reaggregated[key], rescored[key], key)
            independent._check_cost([manifest["cost_record"]], run_id, reference)
            row, fitted = registry[run_id], _fit_for(run_id, manifest, fits)
            anchors = manifest.get("metadata", {}).get("actual_parent_manifests", {})
            record = dict(run_id=run_id, dataset=row["dataset"], fold=row.get("fold", "primary"),
                family=row["family"], k=int(row.get("latent_k") or 0), seed=int(row.get("seed") or 0),
                budget=fitted["budget"] if fitted else "fixed_no_optimizer",
                selected_epoch=fitted["selected_epoch"] if fitted else None,
                training_status=fitted["training_status"] if fitted else "FIXED_BASELINE_NO_OPTIMIZER",
                last_cp_best=fitted["last_cp_best"] if fitted else False,
                cohort=manifest["cohort"], scaler_sha=manifest["scaler_sha"], data_sha=manifest["data_sha"],
                cohort_arrays_sha=training.fingerprint({key: manifest["arrays"][key] for key in sorted(independent.REFERENCE_KEYS)}),
                calibration_parent_signature=training.fingerprint(anchors) if anchors else "",
                parent_b_run_id=anchors.get("b", {}).get("run_id", ""),
                mse=rescored["mse"], mae=rescored["mae"], n_origins=rescored["n_origins"],
                n_observed=rescored["n_observed"], channels=rescored["num_channels"], horizon=rescored["horizon"],
                actual_F_streams=manifest["cost_record"]["actual_F_streams"],
                prediction_execution_scope=manifest.get("metadata", {}).get("execution_scope", "actual_parent_forecast"),
                manifest=manifest, arrays=arrays, fitted=fitted)
            identity = (run_id, manifest["cohort"])
            if identity in seen:
                raise ValueError("Duplicate prediction for identical run/cohort; no automatic score-based choice")
            seen.add(identity)
            for item in (path, reference_path, statistics_path):
                _input(inputs, item, public, "public")
            _input(inputs, _inside(private, manifest["cache_relative_path"]), private, "private")
            records.append(record)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            reason = str(exc)
            for root, label in ((private, "<private cache>"), (public, "<public root>")):
                reason = reason.replace(str(root), label).replace(root.as_posix(), label)
            excluded.append(dict(manifest=path.relative_to(public).as_posix(), reason=reason))
    return records, excluded


def _public_row(row):
    return {key: value for key, value in row.items() if key not in {"manifest", "arrays", "fitted"}}


def _watermark(split, rows):
    if split == "T_final":
        return "T final / frozen selection / locked routes only"
    limited = any(row.get("budget") == "0.25" or row.get("training_status") == "LIMITED_SCREEN" for row in rows)
    return "V selection / preliminary / " + (".25 screen present" if limited else "no independent generalization claim")


def _save(fig, directory, stem, caption, watermark, bottom=.18):
    fig.text(.01, .01, watermark + "\n" + caption, fontsize=8, color="#663300", va="bottom")
    fig.subplots_adjust(bottom=bottom, top=.88, left=.09, right=.97, hspace=.48, wspace=.34)
    files = []
    for suffix in ("png", "svg"):
        path = directory / (stem + "." + suffix)
        fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
        files.append(path.name)
    plt.close(fig)
    return files


def _accuracy(records, directory, split):
    files = []
    for dataset, fold, cohort, arrays_sha in sorted({(r["dataset"], r["fold"], r["cohort"], r["cohort_arrays_sha"]) for r in records}):
        all_rows = [r for r in records if (r["dataset"], r["fold"], r["cohort"], r["cohort_arrays_sha"]) == (dataset, fold, cohort, arrays_sha)]
        budgets = sorted({r["budget"] for r in all_rows if r["budget"] not in {"fixed_no_optimizer", "ridge_closed_form"}}) or ["fixed_no_optimizer"]
        for budget in budgets:
            rows = [r for r in all_rows if r["budget"] == budget or
                    (r["family"] in REFERENCES and r["budget"] in {"fixed_no_optimizer", "ridge_closed_form"})]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
            adapters = [r for r in rows if r["family"] not in REFERENCES and r["k"] > 0]
            for ax, metric in zip(axes, ("mse", "mae")):
                if adapters:
                    groups = sorted({curve_identity(r) for r in adapters})
                    for index, identity in enumerate(groups):
                        points = sorted([r for r in adapters if curve_identity(r) == identity], key=lambda r: (r["k"], r["run_id"]))
                        # Repeated same-K recipes remain individual points; never choose the smallest V/T error.
                        unique = len({r["k"] for r in points}) == len(points)
                        ax.plot([r["k"] for r in points], [r[metric] for r in points],
                            "o-" if unique else "o", color=COLORS[index % len(COLORS)],
                            label=f"{identity[3]} s{identity[4]} maxep={identity[5]}")
                    for index, row in enumerate([r for r in rows if r["family"] in REFERENCES]):
                        ax.axhline(row[metric], color=COLORS[index % len(COLORS)], linestyle="--", alpha=.7,
                                   label=f"{row['family']} s{row['seed']} ep={row['budget']}")
                    ax.set_xlabel("K (actual forecast streams for compressed F routes)")
                else:
                    names = [r["family"] + "\n" + ("closed-form fit" if r["budget"] == "ridge_closed_form" else "no iterative training") for r in rows]
                    ax.bar(np.arange(len(rows)), [r[metric] for r in rows], color=[COLORS[i % len(COLORS)] for i in range(len(rows))])
                    ax.set_xticks(np.arange(len(rows)), names, fontsize=8)
                    ax.set_xlabel("Available references; K sweep pending")
                for row in rows:
                    if row["last_cp_best"] and row["k"] > 0 and adapters:
                        ax.scatter([row["k"]], [row[metric]], marker="x", color="black", s=60)
                ax.set_ylabel("A60-normalized macro " + metric.upper())
                ax.grid(axis="y", alpha=.2)
                if adapters: ax.legend(fontsize=7, loc="best")
            budget_label = "non-iterative references" if budget == "fixed_no_optimizer" else "max training budget " + budget
            subtitle = ""
            if budget == "fixed_no_optimizer":
                notes = []
                if any(r["family"] == "RAW_RIDGE" for r in rows): notes.append("ridge fitted on A")
                if any(r["family"] in {"CURRENT", "SEASONAL_NAIVE"} for r in rows): notes.append("naive untrained")
                if any(r["family"].startswith("RAW_BOLT") or r["family"] == "RAW" for r in rows): notes.append("F frozen pretrained")
                subtitle = "\n" + "; ".join(notes)
            fig.suptitle(f"{dataset} / {fold} / {budget_label}" + subtitle)
            stem = _name(f"accuracy_{dataset}_{fold}_budget_{budget}_{arrays_sha[:8]}")
            _csv(directory / (stem + ".csv"), [_public_row(row) for row in rows])
            caption = "Equal channel weighting; masked targets excluded."
            if adapters and any(r["last_cp_best"] for r in rows): caption += " x: final checkpoint best, convergence unresolved."
            files += _save(fig, directory, stem, caption, _watermark(split, rows), bottom=.25)
    return files


def _matched(records, directory, split):
    rows = []
    for left_family, right_family, label in (("B2", "B0", "B2 - B0"), ("CONDITIONAL", "PAST_CONTROL", "H - past control"),
            ("RAW_CALIBRATED", "RAW_BOLT_SMALL", "Raw calibrated - Raw"), ("CONDITIONAL", "ALPHA_FIT", "H - fitted alpha")):
        for left in [r for r in records if r["family"] == left_family]:
            candidates = [r for r in records if r["family"] == right_family
                and all(r[key] == left[key] for key in ("dataset", "fold", "cohort", "seed", "scaler_sha", "data_sha", "cohort_arrays_sha"))
                and (left["k"] == r["k"] or right_family == "RAW_BOLT_SMALL")
                and (left["budget"] == r["budget"] or right_family in {"RAW_BOLT_SMALL", "ALPHA_FIT"})
                and (left_family != "CONDITIONAL" or (left["calibration_parent_signature"] and left["calibration_parent_signature"] == r["calibration_parent_signature"]))
                and (left_family != "RAW_CALIBRATED" or left["parent_b_run_id"] == r["run_id"])]
            if len(candidates) != 1:
                continue
            right = candidates[0]
            rows.append(dict(comparison=label, dataset=left["dataset"], fold=left["fold"], seed=left["seed"], k=left["k"],
                budget=left["budget"], left_run_id=left["run_id"], right_run_id=right["run_id"],
                mse_difference=left["mse"] - right["mse"], mae_difference=left["mae"] - right["mae"]))
    if not rows:
        return []
    _csv(directory / "matched_controls.csv", rows)
    files = []
    for dataset in sorted({r["dataset"] for r in rows}):
        group = [r for r in rows if r["dataset"] == dataset]
        fig, axes = plt.subplots(1, 2, figsize=(12, max(5, .33 * len(group) + 2)))
        labels = [f"{r['comparison']} | K{r['k']} s{r['seed']} ep={r['budget']}" for r in group]
        for ax, metric in zip(axes, ("mse", "mae")):
            ax.barh(np.arange(len(group)), [r[metric + "_difference"] for r in group], color=COLORS[0])
            ax.set_yticks(np.arange(len(group)), labels, fontsize=7)
            ax.axvline(0, color="black", linewidth=.8); ax.set_xlabel("Delta normalized " + metric.upper() + " (negative favors first)")
        fig.suptitle(dataset + " / matched observed-target cohort; descriptive differences")
        files += _save(fig, directory, _name("matched_" + dataset), "No significance/generalization claim; ambiguous duplicate recipes are not paired automatically.", _watermark(split, records))
    return files


def _learning(fits, directory, split, inputs, public):
    curves, diagnostic = [], []
    for run_id, fitted in sorted(fits.items()):
        receipt, registered = fitted["receipt"], fitted["row"]
        for stage in ("search", "refit"):
            receipt_path = fitted["folder"] / stage / "RECEIPT.json"
            curve_path = fitted["folder"] / stage / "CURVE.csv"
            if not receipt_path.is_file():
                continue
            _input(inputs, receipt_path, public, "public")
            summary = _json(receipt_path)
            diagnostic.append(dict(run_id=run_id, dataset=registered["dataset"], fold=registered.get("fold", "primary"),
                family=registered["family"], k=int(registered.get("latent_k") or 0), seed=registered.get("seed"),
                budget=fitted["budget"], stage=stage, training_status=fitted["training_status"],
                last_cp_best=fitted["last_cp_best"], selected_epoch=fitted["selected_epoch"],
                actual_epochs=summary.get("actual_epochs"), updates=summary.get("updates", 0),
                exposures=summary.get("exposures", 0), unique_origins=summary.get("unique_origins"),
                observed_target_exposures=summary.get("observed_target_exposures"),
                clipped_steps=summary.get("clipped_steps", 0), clipping_fraction=summary.get("clipping_fraction", 0),
                F_streams=summary.get("F_total_logical_streams", 0), k_exposures=json.dumps(summary.get("k_exposures", {}), sort_keys=True)))
            if curve_path.is_file():
                _input(inputs, curve_path, public, "public")
                with curve_path.open(encoding="utf-8", newline="") as stream:
                    for row in csv.DictReader(stream):
                        curves.append(dict(run_id=run_id, dataset=registered["dataset"], family=registered["family"],
                            seed=registered.get("seed"), k=int(registered.get("latent_k") or 0), budget=fitted["budget"], stage=stage, **row))
    if not diagnostic:
        return [], diagnostic
    _csv(directory / "learning_curves.csv", curves); _csv(directory / "training_diagnostics.csv", diagnostic)
    files = []
    for dataset in sorted({r["dataset"] for r in diagnostic}):
        rows = [r for r in diagnostic if r["dataset"] == dataset]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        curves_ds = [r for r in curves if r["dataset"] == dataset and r["stage"] == "search" and r.get("astop_mse") not in {"", None}]
        families = sorted({r["family"] for r in rows})
        for index, family in enumerate(families):
            first = True
            for run_id in sorted({r["run_id"] for r in curves_ds if r["family"] == family}):
                group = [r for r in curves_ds if r["run_id"] == run_id]
                axes[0, 0].plot([float(r["epoch"]) for r in group], [float(r["astop_mse"]) for r in group],
                    marker=".", alpha=.65, color=COLORS[index % len(COLORS)], label=family if first else None)
                first = False
        axes[0, 0].set(xlabel="A50 search epochs; separate run lines", ylabel="A_stop normalized MSE")
        if curves_ds: axes[0, 0].legend(fontsize=7)
        search = [r for r in rows if r["stage"] == "search"]
        axes[0, 1].scatter(np.arange(len(search)), [r["selected_epoch"] or 0 for r in search],
                           c=[COLORS[families.index(r["family"]) % len(COLORS)] for r in search])
        axes[0, 1].set(xlabel="Run index (exact IDs in CSV)", ylabel="Selected A_stop epoch")
        for stage, marker in (("search", "o"), ("refit", "x")):
            group = [r for r in rows if r["stage"] == stage]
            axes[1, 0].scatter([r["exposures"] for r in group], [r["updates"] for r in group], marker=marker, label=stage)
            axes[1, 1].scatter([r["actual_epochs"] or 0 for r in group], [r["clipping_fraction"] for r in group], marker=marker, label=stage)
        axes[1, 0].set(xlabel="Cumulative origin exposures (search includes inherited)", ylabel="Optimizer updates")
        axes[1, 1].set(xlabel="Actual stage epochs", ylabel="Recorded clipped-step fraction")
        for ax in (axes[1, 0], axes[1, 1]): ax.legend(fontsize=8)
        for ax in axes.flat: ax.grid(alpha=.2)
        fig.suptitle(dataset + " / training chronology and diagnostics; budgets are separate run lines")
        files += _save(fig, directory, _name("learning_" + dataset), "Final-best means convergence unresolved. Budget-interrupted clipping diagnostics may be incomplete; see receipts.", _watermark(split, rows))
    return files, diagnostic


def _costs(records, directory, split, public, inputs):
    matched = []
    by_id = {r["run_id"]: r for r in records}
    for path in sorted(public.rglob("*.json")):
        relative = path.relative_to(public)
        if "figures" in relative.parts or not (any("cost" in part.lower() for part in relative.parts)):
            continue
        value = _json(path)
        if value.get("scope") not in {"fresh_process_single_route", "general_shared_F_balanced"}:
            continue
        trials = path.with_name(value["trials_file"])
        if data.sha256_file(trials) != value["trials_sha256"]:
            raise ValueError("Actual cost trials hash changed")
        _input(inputs, path, public, "public"); _input(inputs, trials, public, "public")
        for row in value["summaries"]:
            if row.get("phase") != "timed" or row["run_id"] not in by_id:
                continue
            reference = by_id[row["run_id"]]
            if value.get("data_sha256") != reference["data_sha"]:
                raise ValueError("Deployment cost and prediction data identity differ or are unbound")
            if value.get("scaler_sha") is not None and value["scaler_sha"] != reference["scaler_sha"]:
                raise ValueError("Deployment cost and prediction scaler identity differ")
            metadata = reference["manifest"].get("metadata", {})
            bound_artifacts = {v["sha256"] for v in metadata.get("artifact_bindings", {}).values()}
            for artifact_sha in (metadata.get("artifact_sha256"), metadata.get("realized_model_binding", {}).get("selected_artifact_sha256")):
                if artifact_sha: bound_artifacts.add(artifact_sha)
            cost_artifact = value.get("artifact_sha256")
            if cost_artifact and bound_artifacts and cost_artifact not in bound_artifacts:
                raise ValueError("Deployment cost and prediction artifact SHA differ")
            memory = value["memory"] if value["scope"] == "fresh_process_single_route" else value["memory"].get(row["run_id"], {})
            matched.append(dict(run_id=row["run_id"], dataset=reference["dataset"], family=reference["family"],
                mse=reference["mse"], mae=reference["mae"], batch_size=value["batch_size"], device=value["device"],
                median_whole_ms=float(row["median_seconds"]) * 1000, p90_whole_ms=float(row["p90_seconds"]) * 1000,
                trials=row["trials"], timer_scope=value["scope"], cost_protocol_status=value["status"],
                dtype=value.get("dtype"), fresh_process_memory=value["scope"] == "fresh_process_single_route",
                model_binding_level="exact_artifact_sha" if cost_artifact and bound_artifacts else "registered_run_id_and_data; no per-route artifact SHA in this receipt",
                peak_reserved_MiB=memory.get("peak_reserved_MiB"), resident_weights_gpu_MiB=memory.get("resident_weights_gpu_MiB"),
                resident_weights_cpu_MiB=memory.get("resident_weights_cpu_MiB"), RSS_MiB_after_close=value.get("RSS_MiB_after_close"),
                training_peak_included=value.get("training_peak_included", False), cached_forecast_reads=value.get("cached_forecast_reads", 0)))
    if not matched:
        return []
    _csv(directory / "cost_accuracy.csv", matched)
    files = []
    for dataset, batch, device in sorted({(r["dataset"], r["batch_size"], r["device"]) for r in matched}):
        rows = [r for r in matched if (r["dataset"], r["batch_size"], r["device"]) == (dataset, batch, device)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for index, row in enumerate(rows):
            color = COLORS[index % len(COLORS)]
            axes[0].scatter(row["median_whole_ms"], row["mse"], color=color,
                marker="o" if row["timer_scope"] == "general_shared_F_balanced" else "s", label=row["family"])
            axes[0].annotate(_name(row["run_id"])[-35:], (row["median_whole_ms"], row["mse"]), fontsize=6)
        memory_key = "peak_reserved_MiB" if str(device).startswith("cuda") else "RSS_MiB_after_close"
        fresh = [r for r in rows if r["fresh_process_memory"] and r[memory_key] is not None]
        if fresh:
            axes[1].bar(np.arange(len(fresh)), [r[memory_key] for r in fresh], color=COLORS[0])
            axes[1].set_xticks(np.arange(len(fresh)), [r["family"] for r in fresh], fontsize=7, rotation=35)
            axes[1].set_ylabel("Fresh-worker GPU peak reserved MiB" if str(device).startswith("cuda") else "Fresh-worker RSS after close MiB (not peak)")
        else:
            axes[1].set_visible(False)
        axes[0].set(xlabel="Actual whole physical-input pipeline median ms", ylabel="Normalized macro MSE")
        axes[0].grid(alpha=.2); axes[0].legend(fontsize=6)
        fig.suptitle(f"{dataset} / B={batch} / {device} / measured deployment costs")
        files += _save(fig, directory, _name(f"cost_{dataset}_B{batch}_{device}"), "Square=fresh single route; circle=balanced shared process. Loading excluded; stage probes are separate/intrusive. No stage-sum claim.", _watermark(split, records))
    return files


def _special(records, diagnostic, directory, split):
    special = {"LAG_CAUSAL", "LAG_BIDIR", "POSTHOC_PREFIX", "MAXK_PCA_TRAINED", "NESTED_PCA", "ORDERED_MULTI_K"}
    rows = [r for r in records if r["family"] in special]
    exposure = [r for r in diagnostic if r["family"] in special]
    if not rows and not exposure:
        return []
    _csv(directory / "lag_prefix_multik_scores.csv", [_public_row(r) for r in rows])
    _csv(directory / "lag_prefix_multik_exposure.csv", exposure)
    files = []
    for dataset in sorted({r["dataset"] for r in rows + exposure}):
        score_rows = [r for r in rows if r["dataset"] == dataset]
        exposures = [r for r in exposure if r["dataset"] == dataset]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for index, row in enumerate(score_rows):
            axes[0].scatter(row["k"], row["mse"], color=COLORS[index % len(COLORS)],
                            label=f"{row['family']} s{row['seed']} ep={row['budget']}")
            axes[0].annotate(str(row["actual_F_streams"]) + " F rows", (row["k"], row["mse"]), fontsize=6)
        if score_rows:
            axes[0].set(xlabel="Actual inference K", ylabel="Normalized macro MSE"); axes[0].legend(fontsize=6)
        else:
            axes[0].set_visible(False)
        for index, row in enumerate(exposures):
            axes[1].scatter(row["exposures"], row["F_streams"], color=COLORS[index % len(COLORS)],
                marker="o" if row["stage"] == "search" else "x", label=f"{row['family']} K{row['k']} {row['stage']} ep={row['budget']}")
        axes[1].set(xlabel="Training origin exposures; not unique histories", ylabel="Actual cumulative F rows incl. stage evaluations")
        if exposures: axes[1].legend(fontsize=6)
        fig.suptitle(dataset + " / lag and prefix diagnostics (missing comparisons remain pending)")
        files += _save(fig, directory, _name("lag_prefix_multik_" + dataset), "Multi-K cycles are exposure-sharing; repeated exposures are not independent observations. Exact K exposure ledger is in CSV.", _watermark(split, rows + exposure))
    return files


def _priority(row):
    families = ["RAW_BOLT_SMALL", "RAW", "CURRENT", "RAW_RIDGE", "DIRECT_G", "LORA_R8", "B0", "B2", "CONDITIONAL", "SEASONAL_NAIVE"]
    return (families.index(row["family"]) if row["family"] in families else len(families), row["run_id"])


def _traces(records, public, private, directory, split, inputs):
    files, descriptors = [], []
    for dataset, fold, cohort, arrays_sha in sorted({(r["dataset"], r["fold"], r["cohort"], r["cohort_arrays_sha"]) for r in records}):
        candidates = sorted([r for r in records if (r["dataset"], r["fold"], r["cohort"], r["cohort_arrays_sha"]) == (dataset, fold, cohort, arrays_sha)], key=_priority)
        chosen = candidates[:6]
        anchor = chosen[0]
        arrays = anchor["arrays"]
        task = data.load_task(dataset, cache_dir=private)
        if task["data_sha256"] != anchor["data_sha"]:
            raise ValueError("Trace Task data differs from the independently audited prediction cohort")
        mean, std, _ = data.fit_statistics(task, data.split_ranges(task, fold)["A_stop"][1])
        import fit_base
        if fit_base.scaler_fingerprint(mean, std) != anchor["scaler_sha"]:
            raise ValueError("Trace physical conversion uses a different A60 scaler")
        for row in chosen:
            if any(row[key] != anchor[key] for key in ("scaler_sha", "data_sha", "cohort")):
                raise ValueError("Trace methods do not share the exact target and scaler cohort")
        n, horizon, channels = arrays["pred_normalized"].shape
        channel = 0
        error = np.where(arrays["observed"], (arrays["pred_normalized"].astype(np.float64) - arrays["target_normalized"]) ** 2, 0)
        counts = arrays["observed"].sum((1, 2))
        sample_errors = np.divide(error.sum((1, 2)), counts, out=np.full(n, np.nan), where=counts > 0)
        finite = np.flatnonzero(np.isfinite(sample_errors))
        rng = np.random.default_rng(92601)
        selections = [("fixed first", 0), ("fixed middle", n // 2), ("fixed last", n - 1),
                      ("seeded random (92601)", int(rng.integers(n)))]
        if len(finite) and split == "V_select":
            selections += [("POST-HOC best anchor error", int(finite[np.argmin(sample_errors[finite])])),
                           ("POST-HOC worst anchor error", int(finite[np.argmax(sample_errors[finite])]))]
        nrows = int(math.ceil(len(selections) / 2))
        fig, axes = plt.subplots(nrows, 2, figsize=(13, 3.2 * nrows + .4))
        trace_rows = []
        unit_note = "legacy FP32 inverse-normalized native units" if task.get("metadata", {}).get("source_kind") == "legacy_normalized" else "native physical units"
        for ax, (label, index) in zip(axes.flat, selections):
            origin = int(arrays["origins"][index])
            history = min(48, task["L"])
            past = task["physical"][origin - history:origin, channel]
            past_mask = task["observed"][origin - history:origin, channel]
            target = arrays["target_normalized"][index, :, channel].astype(np.float64) * float(std[channel]) + float(mean[channel])
            mask = arrays["observed"][index, :, channel]
            x = np.arange(horizon)
            ax.plot(np.arange(-history, 0), np.where(past_mask, past, np.nan), color="#777777", linewidth=1, label="observed history")
            ax.plot(x, np.where(mask, target, np.nan), color="black", linewidth=2, label="observed target")
            for method_index, row in enumerate(chosen):
                predicted = row["arrays"]["pred_normalized"][index, :, channel].astype(np.float64) * float(std[channel]) + float(mean[channel])
                width_label = f"K{row['k']}" if row["k"] else (f"C{row['channels']}" if row["actual_F_streams"] else "no F")
                budget_label = {"fixed_no_optimizer": "fixed", "ridge_closed_form": "closed-form"}.get(row["budget"], row["budget"])
                method_label = f"{row['family']} {width_label} s{row['seed']} ep={budget_label}"
                ax.plot(x, predicted, color=COLORS[method_index % len(COLORS)], linewidth=1, label=method_label)
                for step in range(horizon):
                    trace_rows.append(dict(selection=label, origin=origin, channel=str(arrays["channel_ids"][channel]), relative_step=step,
                        observed=bool(mask[step]), truth_native=float(target[step]) if mask[step] else None,
                        run_id=row["run_id"], pred_native=float(predicted[step]), native_unit_note=unit_note))
            for step in range(history):
                trace_rows.append(dict(selection=label, origin=origin, channel=str(arrays["channel_ids"][channel]), relative_step=step - history,
                    observed=bool(past_mask[step]), truth_native=float(past[step]) if past_mask[step] else None,
                    run_id="OBSERVED_HISTORY_ONLY", pred_native=None, native_unit_note=unit_note))
            missing = x[~mask]
            if len(missing):
                ax.scatter(missing, np.full(len(missing), .03), transform=ax.get_xaxis_transform(),
                           marker="x", color="red", s=25, label="target missing (not scored)")
            ax.axvline(-.5, color="black", linewidth=.7, linestyle="--")
            ax.set_title(f"{label} / origin={origin} / observed={int(mask.sum())}/{horizon}", fontsize=9)
            ax.set(xlabel="Native sample steps from forecast origin", ylabel=str(arrays["channel_ids"][channel]))
            ax.grid(alpha=.15)
        for ax in list(axes.flat)[len(selections):]: ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .935), ncol=3, fontsize=7)
        fig.suptitle(f"{dataset} / {fold} / fixed channel 0 / H={horizon} at {task.get('interval_seconds', '?')} s per step", y=.99)
        stem = _name("traces_" + dataset + "_" + fold + "_" + arrays_sha[:8])
        private_directory = private / "figures" / directory.name
        private_directory.mkdir(parents=True, exist_ok=True)
        private_csv = private_directory / (stem + "_exact_plotdata.csv.gz")
        _csv(private_csv, trace_rows)
        _input(inputs, private_csv, private, "private")
        descriptors.append(dict(dataset=dataset, cohort=cohort, anchor_run_id=anchor["run_id"],
            channel_index=0, private_plotdata_relative_path=private_csv.relative_to(private).as_posix(),
            private_plotdata_sha256=data.sha256_file(private_csv), native_unit_note=unit_note,
            data_sha=task["data_sha256"], fixed_selection_seed=92601,
            posthoc_selection="best/worst V origin by predeclared anchor across channels; descriptive only" if split == "V_select" else "disabled: T may not choose figure cases",
            plotted_steps_per_origin=history + horizon))
        caption = unit_note + ". Target gaps stay masked. "
        caption += "Post-hoc extremes are not representative evidence or additional independent samples." if split == "V_select" else "Only predeclared fixed cases; no T-dependent case selection."
        files += _save(fig, directory, stem, caption, _watermark(split, chosen))
    return files, descriptors


def generate(public_root, private_root, split="V_select"):
    """Return a new public snapshot directory. Never fit, infer, select or overwrite."""
    public, private = Path(public_root).resolve(), Path(private_root).resolve()
    if split not in {"V_select", "T_final"}:
        raise ValueError("Only V_select and explicitly locked T_final figures are supported")
    inputs, lock = {}, {}
    if split == "T_final":
        path = public / "SELECTION_LOCK.json"
        if not path.is_file():
            raise ValueError("T figures require SELECTION_LOCK.json")
        lock = _json(path)
        if lock.get("locked") is not True:
            raise ValueError("T figures require a locked SELECTION_LOCK")
        _input(inputs, path, public, "public")
    registry_path = public / "EXPERIMENT_REGISTRY.csv"
    registry_bytes = registry_path.read_bytes()
    registry_rows = list(csv.DictReader(io.StringIO(registry_bytes.decode("utf-8-sig"))))
    registry = {row["experiment_id"]: row for row in registry_rows}
    if len(registry) != len(registry_rows):
        raise ValueError("Duplicate registered experiment IDs")
    inputs[("public", registry_path.relative_to(public).as_posix())] = dict(scope="public", file=registry_path.relative_to(public).as_posix(),
        sha256=hashlib.sha256(registry_bytes).hexdigest(), bytes=len(registry_bytes))
    fits = _fits(public, registry, inputs)
    records, excluded = _predictions(public, private, split, registry, fits, inputs, lock)
    if split == "T_final":
        if not records:
            raise ValueError("No checked locked T_final prediction manifests exist")
        selected_fit_ids = {r["fitted"]["row"]["experiment_id"] for r in records if r["fitted"] is not None}
        fits = {key: value for key, value in fits.items() if key in selected_fit_ids}
    snapshot = "snapshot_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + str(time.time_ns())
    directory = public / "figures" / snapshot
    directory.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False})
    _csv(directory / "verified_scores.csv", [_public_row(row) for row in records])
    groups = []
    def add(group, files, reason):
        groups.append(dict(group=group, status="AVAILABLE_PARTIAL_EVIDENCE" if files else "PENDING", figures=files,
                           reason=reason, complete_campaign_claimed=False))
    add("accuracy_by_k", _accuracy(records, directory, split), "Only actual checked forecasts; fixed references alone do not complete the registered K sweep. Training budgets/seeds remain distinct.")
    add("matched_controls", _matched(records, directory, split), "Only exact cohort/scaler/seed and compatible budget pairs; missing or ambiguous counterparts stay pending.")
    learning, diagnostic = _learning(fits, directory, split, inputs, public)
    add("learning_diagnostics", learning, "Available immutable search/refit receipts, not inferred convergence. Closed-form ridge has no optimizer learning curve.")
    add("cost_accuracy", _costs(records, directory, split, public, inputs), "Requires actual whole-pipeline trial receipts matched to forecast IDs; cached forecast runtime is never a deployment cost.")
    add("lag_prefix_multik", _special(records, diagnostic, directory, split), "Training exposure evidence and forecast evidence are separate; missing lag/prefix/multi-K inference remains pending.")
    traces, trace_descriptors = _traces(records, public, private, directory, split, inputs)
    add("representative_traces", traces, "Fixed first/middle/last/seeded origins plus explicitly post-hoc anchor extremes; exact bounded plot data are private.")
    readme = ["# 실험 시각화 스냅샷", "", "이 디렉터리는 실제 저장 예측의 FP64 재채점·관측 마스크·원점·채널·참조 SHA 검산 후 만든 파생 그림이다.",
        "학습 재실행·모델 선택·독립 일반화 검증을 수행한 결과가 아니다. V는 선택용 예비 결과이며 미실행 그룹은 manifest에 PENDING으로 남긴다.", "",]
    for group in groups:
        readme += [f"- `{group['group']}`: {group['status']} — {group['reason']}"]
        for file in group["figures"]:
            if file.endswith(".png"):
                readme += ["", f"![{file}]({file})", ""]
    readme += ["", "## 집계 원수치와 검산 연결", "", f"검산을 통과한 저장 예측 {len(records)}개, 제외 입력 {len(excluded)}개다. 전체 registry 감사와 학습 재실행을 수행한 수치가 아니다.", "",
               "대표 사례의 정확한 제한된 plot data는 private cache에만 저장하며 공개 CSV에는 전체 원자료나 대표 사례의 시계열 배열을 넣지 않는다.", ""]
    readme += [f"- [{path.name}]({path.name})" for path in sorted(directory.iterdir()) if path.suffix in {".csv", ".svg"}]
    readme += ["- [FIGURE_MANIFEST.json](FIGURE_MANIFEST.json)", ""]
    (directory / "README_KO.md").write_text("\n".join(readme), encoding="utf-8")
    outputs = [dict(file=path.name, bytes=path.stat().st_size, sha256=data.sha256_file(path)) for path in sorted(directory.iterdir()) if path.is_file()]
    manifest = dict(created_unix=time.time(), split=split, snapshot=snapshot, source_sha256=data.sha256_file(Path(__file__)),
        dependency_sha256={name: data.sha256_file(Path(__file__).with_name(name)) for name in ("audit_predictions.py", "data_contract.py", "training.py")},
        font="DejaVu Sans bundled with matplotlib; English labels; SVG glyphs embedded as paths", matplotlib_version=matplotlib.__version__,
        verified_prediction_count=len(records), fitted_receipt_count=len(fits),
        figure_count=sum(file.endswith(".png") for group in groups for file in group["figures"]), groups=groups,
        expected_datasets=sorted({row["dataset"] for row in registry_rows}), observed_datasets=sorted({r["dataset"] for r in records}),
        input_files=list(inputs.values()), outputs=outputs, excluded_prediction_inputs=excluded,
        traces=trace_descriptors, public_exact_trace_arrays=False, full_raw_source_publication=False,
        private_source_data_changed=False, GPU_execution=False, training_rerun=False, prediction_reinference=False,
        independent_prediction_rescoring=True, full_registry_audit_performed_here=False,
        independent_generalization_claimed=False, no_cross_dataset_metric_average=True,
        T_method_selection_performed=False, T_selection_lock_checked=split == "T_final")
    _new_json(directory / "FIGURE_MANIFEST.json", manifest)
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--split", default="V_select", choices=("V_select", "T_final"))
    args = parser.parse_args()
    print(str(generate(args.public_root, args.private_root, args.split)))


if __name__ == "__main__":
    main()
