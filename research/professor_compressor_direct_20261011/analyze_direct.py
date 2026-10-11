"""CPU-only reports of saved actual direct-study predictions, without model/TEST loads."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from common_direct import HERE, ROOT, CACHE, Resources, array_digest, digest, read_json, write_json


ARMS = ("TRANSFER_F", "CALIBRATED_F", "LATENT_PERSIST_TRANSFER",
        "LATENT_PERSIST_CALIBRATED", "TS2_NATIVE_RIDGE")
BASE_ARMS = ("RAW_F", "MEAN_BLOCK_F")
COLORS = {"RAW_F": "#0072B2", "MEAN_BLOCK_F": "#777777", "TRANSFER_F": "#E69F00",
          "CALIBRATED_F": "#009E73", "LATENT_PERSIST_TRANSFER": "#CC79A7",
          "LATENT_PERSIST_CALIBRATED": "#8E44AD", "TS2_NATIVE_RIDGE": "#D55E00"}


def artifact_path(value):
    if isinstance(value, dict):
        value = value["path"]
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(CACHE)):
        raise ValueError(f"Saved prediction must belong to the new direct study: {path}")
    return path


def independent_metric(prediction, target, mask):
    """Channel-macro observed metrics; no call to the runner's metric implementation."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction/target/original mask shape mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Non-finite actual prediction or observed target")
    squared, absolute, counts = [], [], []
    for channel in range(target.shape[-1]):
        selected = mask[..., channel]
        delta = prediction[..., channel][selected] - target[..., channel][selected]
        counts.append(len(delta))
        squared.append(float(np.mean(delta * delta)) if len(delta) else None)
        absolute.append(float(np.mean(np.abs(delta))) if len(delta) else None)
    mse_values = [value for value in squared if value is not None]
    mae_values = [value for value in absolute if value is not None]
    result = {"mse": float(np.mean(mse_values)) if mse_values else None,
              "mae": float(np.mean(mae_values)) if mae_values else None,
              "channel_mse": squared, "channel_mae": absolute, "observed_counts": counts,
              "origin_count": len(target), "observed_count_total": int(mask.sum())}
    if target.ndim == 3:
        horizon = [independent_metric(prediction[:, step], target[:, step], mask[:, step])
                   for step in range(target.shape[1])]
        result["horizon_mse"] = [item["mse"] for item in horizon]
        result["horizon_mae"] = [item["mae"] for item in horizon]
    return result


def planned_jobs(protocol):
    if isinstance(protocol.get("jobs"), list) and len(protocol["jobs"]) == 16:
        return protocol["jobs"], "protocol.json jobs"
    jobs = []
    for dataset in ("robin", "jena"):
        for seed in (92601, 92602):
            for no_spatial in (False, True):
                jobs.append({"dataset": dataset, "seed": seed, "family": "GT",
                             "no_spatial": no_spatial})
            for dim in (16, 320):
                jobs.append({"dataset": dataset, "seed": seed, "family": "TS2", "dim": dim})
    return jobs, "replacement document's 16 planned fits; protocol jobs not yet available"


def job_key(job):
    family = str(job.get("family", "")).upper()
    return (job.get("dataset"), int(job.get("seed", 0)), family,
            bool(job.get("no_spatial", False)) if family == "GT" else int(job.get("dim", 0)))


def method_label(job):
    if str(job.get("family", "")).upper() == "BASELINES":
        return "baseline"
    if str(job.get("family", "")).upper() == "GT":
        return "GT_no_spatial" if job.get("no_spatial", False) else "GT_full"
    return f"TS2_d{job.get('dim')}"


def fixed_indices(count):
    return list(dict.fromkeys([0, count // 2, count - 1])) if count else []


def select_canonical_results(planned, summaries, errors, warnings):
    """Only protocol IDs select scientific runs; retries never add a third seed."""
    jobs_by_id = {job["id"]: job for job in planned if job.get("id")}
    if len(jobs_by_id) != len(planned):
        warnings.append("Canonical protocol IDs are missing or duplicated; actual runs remain outside canonical aggregation")
        jobs_by_id = {}
    jobs_by_condition = {job_key(job): job for job in planned}
    buckets, extras = {}, []

    def extra(summary, reason, planned_job=None):
        result, job = summary["result"], summary["result"].get("job", {})
        extras.append({"id": job.get("id"), "job": job,
                       "status": result.get("status"), "result_path": summary["result_path"],
                       "result_sha256": summary["result_sha256"], "reason": reason,
                       "required_protocol_id": planned_job.get("id") if planned_job else None,
                       "canonical_id": planned_job.get("canonical_id", planned_job.get("id")) if planned_job else None,
                       "excluded_from_canonical_metrics_coverage_and_seed_means": True,
                       "metric_evidence": "preserved original result receipt; not independently recalculated in this canonical analysis",
                       "reported_reconstruction": result.get("reconstruction"),
                       "reported_forecast": result.get("forecast"),
                       "training": result.get("training"), "calibration": result.get("calibration"),
                       "cost": result.get("cost"), "resources": result.get("resources"),
                       "source_pins": result.get("source_pins"),
                       "prediction_path": result.get("prediction_path"),
                       "prediction_sha256": result.get("prediction_sha256")})

    for summary in summaries:
        job = summary["result"].get("job", {})
        identifier = job.get("id")
        planned_job = jobs_by_id.get(identifier)
        if planned_job is None:
            matching = jobs_by_condition.get(job_key(job))
            reason = ("implementation pilot before epoch0 selection fix"
                      if matching and matching.get("canonical_id") == identifier
                      else "actual run excluded by current explicit protocol ID selection")
            extra(summary, reason, matching)
            continue
        if job_key(job) != job_key(planned_job):
            errors.append({"path": summary["result_path"],
                           "error": f"Result job condition differs from explicit protocol ID {identifier}"})
            extra(summary, "protocol ID matches but condition differs", planned_job)
            continue
        buckets.setdefault(identifier, []).append(summary)

    canonical = []
    for identifier, entries in buckets.items():
        if len(entries) == 1:
            canonical.append(entries[0])
        else:
            errors.append({"job_id": identifier, "error": "Multiple result receipts declare the same canonical protocol ID"})
            for summary in entries:
                extra(summary, "ambiguous duplicate result receipts for explicit protocol ID", jobs_by_id[identifier])
    return canonical, extras


def current_live_jobs():
    live = {}
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            args = process.info.get("cmdline") or []
            command = " ".join(args).replace("\\", "/")
            if HERE.name not in command or not any(name in command for name in ("runner_gt.py", "runner_ts2.py", "memory_retry.py")):
                continue
            def argument(flag):
                return args[args.index(flag) + 1] if flag in args else None
            job = {"dataset": argument("--dataset"), "seed": int(argument("--seed")),
                   "family": "GT" if "runner_gt.py" in command else "TS2"}
            if job["family"] == "GT":
                job["no_spatial"] = "--no-spatial" in args
                identifier = f"{job['dataset']}_GT_{'no_spatial' if job['no_spatial'] else 'full'}_s{job['seed']}"
            else:
                job["dim"] = int(argument("--dim"))
                identifier = f"{job['dataset']}_TS2_d{job['dim']}_s{job['seed']}"
                if argument("--retry-tag"):
                    identifier += f"_retry_{argument('--retry-tag')}"
            live[identifier] = {"pid": process.pid, "job_id": identifier,
                                "created_unix": process.info["create_time"]}
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError, IndexError):
            continue
    return live


def collect_attempt_states(planned, live, warnings):
    """Read small failure/protocol/curve receipts, without opening partial weights."""
    planned_by_id = {job.get("id"): job for job in planned}
    planned_by_condition = {job_key(job): job for job in planned}
    identifiers = set()
    for directory in (HERE / "runs", CACHE / "runs"):
        if directory.exists():
            identifiers.update(path.name for path in directory.iterdir() if path.is_dir())
    attempts = {}
    for identifier in sorted(identifiers):
        result_dir, cache_dir = HERE / "runs" / identifier, CACHE / "runs" / identifier
        failure_path = result_dir / "failure.json"
        initial_path = cache_dir / "initial_protocol.json"
        curves = sorted(list(result_dir.glob("*curve.json")) + list(cache_dir.glob("*curve.json")))
        if not failure_path.exists() and ((result_dir / "result.json").exists() or not (initial_path.exists() or curves)):
            continue
        failure, initial, curve_receipts = {}, {}, []
        try:
            failure = read_json(failure_path) if failure_path.exists() else {}
            initial = read_json(initial_path) if initial_path.exists() else {}
            for path in curves:
                entries = read_json(path)
                curve_receipts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                                       "recorded_rows": len(entries),
                                       "last_iteration": max((row.get("iteration", 0) for row in entries), default=0),
                                       "last_epoch": max((row.get("epoch", 0) for row in entries), default=0)})
        except (OSError, ValueError, TypeError) as error:
            warnings.append(f"Partial attempt receipt unavailable for {identifier}: {error!r}")
        job = failure.get("job", initial.get("job", planned_by_id.get(identifier, {"id": identifier})))
        matching = planned_by_id.get(identifier)
        if not matching and job.get("dataset") and job.get("seed") is not None:
            matching = planned_by_condition.get(job_key(job))
        actual_iterations = failure.get("actual_encoder_iterations")
        if actual_iterations is None:
            actual_iterations = max((item["last_iteration"] for item in curve_receipts), default=0) or None
        failed = failure_path.exists()
        attempts[identifier] = {
            "id": identifier, "job": job,
            "status": "FAILED_PRESERVED_PATH" if failed else (
                "RUNNING_CONFIRMED_PROCESS" if identifier in live else "PARTIAL_ARTIFACTS"),
            "original_failure_status": failure.get("status"),
            "reason": failure.get("error", "Saved protocol/curve exists without completed result or live process"),
            "failure_path": failure_path.relative_to(ROOT).as_posix() if failed else None,
            "failure_sha256": digest(failure_path) if failed else None,
            "actual_encoder_iterations": actual_iterations,
            "expected_encoder_iterations": initial.get("expected_official_iterations"),
            "iteration_evidence": "failure receipt actual iterations" if failure.get("actual_encoder_iterations") is not None else "saved curve lower bound only",
            "initial_protocol_path": initial_path.relative_to(ROOT).as_posix() if initial_path.exists() else None,
            "initial_protocol_sha256": digest(initial_path) if initial_path.exists() else None,
            "encoder_batch_size": initial.get("batch_size"),
            "curve_receipts": curve_receipts, "resources": failure.get("resources"),
            "resume": failure.get("resume"), "observed_live_process": live.get(identifier),
            "is_current_protocol_id": identifier in planned_by_id,
            "required_protocol_id": matching.get("id") if matching else None,
            "canonical_id": matching.get("canonical_id", matching.get("id")) if matching else None,
            "included_in_completed_scientific_fit_count": False}
    return attempts


def coverage_item(job, record, observed_live, attempt):
    status = "not_started" if record is None else record["result"].get("status", "unknown")
    if status == "complete":
        status = "complete_saved_predictions_verified" if record.get("valid") else "INVALID_ARTIFACTS"
    elif record is None:
        if attempt and attempt.get("failure_path"):
            status = "FAILED_PRESERVED_PATH"
        elif observed_live:
            status = "RUNNING_CONFIRMED_PROCESS"
        elif attempt:
            status = "PARTIAL_ARTIFACTS"
    return {"job": {**job, **record["result"]["job"]} if record else job, "status": status,
            "execution_kind": "new_training" if record or observed_live or attempt else "not_executed",
            "observed_live_process": observed_live,
            "result_path": record.get("result_path") if record else None,
            "failure_path": attempt.get("failure_path") if attempt else None,
            "reason": attempt.get("reason") if attempt else None,
            "actual_encoder_iterations": attempt.get("actual_encoder_iterations") if attempt else None,
            "expected_encoder_iterations": attempt.get("expected_encoder_iterations") if attempt else None,
            "attempt_artifact_evidence": attempt}


def load_result(path, data_checks, truth, errors, warnings):
    result = read_json(path)
    record = {"result_path": path.relative_to(ROOT).as_posix(), "result_sha256": digest(path),
              "result": result, "valid": False}
    if result.get("status") != "complete":
        return record
    job = result.get("job", {})
    dataset = job.get("dataset", result.get("dataset"))
    record["dataset"] = dataset
    try:
        expected = data_checks.get("datasets", {}).get(dataset, {})
        if expected and result.get("data_contract_hash") != expected.get("contract_hash"):
            raise ValueError("Result data contract differs from checked TRAIN/VAL contract")
        prediction_path = artifact_path(result["prediction_path"])
        actual_sha = digest(prediction_path)
        stored_sha = result.get("prediction_sha256")
        if stored_sha and actual_sha != stored_sha:
            raise ValueError("Saved prediction SHA differs from actual result receipt")
        if not stored_sha:
            warnings.append(f"{path.name}: no prediction SHA declared; current SHA recorded")
        with np.load(prediction_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        target, mask, origins = arrays["targets"], arrays["mask"].astype(bool), arrays["origins"]
        if target.ndim != 3 or target.shape != mask.shape or target.shape[1] != 48:
            raise ValueError("Actual physical target must be [origins,48,C] with original mask")
        if len(origins) != len(target) or not np.all(np.diff(origins) > 0):
            raise ValueError("Saved VAL origin rows do not align or are not increasing")
        expected_origin = expected.get("origin_checks", {}).get("val_origins", {})
        if expected_origin.get("sha256") and array_digest(origins) != expected_origin["sha256"]:
            raise ValueError("Saved origins differ from DATA_CHECKS VAL origin contract")
        signature = {"data_contract_hash": result.get("data_contract_hash"),
                     "origins_sha256": array_digest(origins), "mask_sha256": array_digest(mask),
                     "observed_target_sha256": array_digest(np.where(mask, target, 0)),
                     "shape": list(target.shape)}
        if dataset in truth and signature != truth[dataset]:
            raise ValueError("Methods do not share the same observed targets/mask/origins/normalization contract")
        truth[dataset] = signature
        forecast_metrics = {}
        for arm, reported in result.get("forecast", {}).items():
            if arm not in ARMS + BASE_ARMS:
                warnings.append(f"Unrecognized forecast arm preserved in JSON, excluded from table: {arm}")
                continue
            if arm not in arrays:
                raise ValueError(f"Actual prediction array missing for declared arm {arm}")
            measured = independent_metric(arrays[arm], target, mask)
            for metric in ("mse", "mae"):
                stored = reported.get(metric)
                actual = measured[metric]
                if stored is None or actual is None or not np.isclose(stored, actual, rtol=1e-7, atol=1e-9):
                    raise ValueError(f"{arm} {metric} disagrees with independent observed calculation")
            if "std" in arrays:
                scales = arrays["std"]
                if scales.shape != (target.shape[-1],) or not np.isfinite(scales).all() or not (scales > 0).all():
                    raise ValueError("Saved physical-unit scales are not valid per-channel TRAIN scales")
                expected_std = expected.get("array_hashes", {}).get("std")
                if expected_std and array_digest(scales) != expected_std:
                    raise ValueError("Physical-unit scales differ from DATA_CHECKS TRAIN normalization")
                physical = [value * float(scale) if value is not None else None
                            for value, scale in zip(measured["channel_mae"], scales)]
                if reported.get("physical_channel_mae") is not None:
                    for stored, actual in zip(reported["physical_channel_mae"], physical):
                        if (stored is None) != (actual is None) or (actual is not None and not np.isclose(stored, actual, rtol=1e-7, atol=1e-9)):
                            raise ValueError(f"{arm} physical channel MAE disagrees with saved TRAIN scales")
                measured["physical_channel_mae"] = physical
            forecast_metrics[arm] = measured
        record.update(valid=True, arrays=arrays, forecast_metrics=forecast_metrics,
                      prediction_path=prediction_path, prediction_sha256=actual_sha, truth=signature)
    except (OSError, ValueError, KeyError) as error:
        errors.append({"path": str(path), "error": repr(error)})
    return record


def save_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8-sig")


def fmt(value):
    return "pending" if value is None else f"{float(value):.6f}"


def relative_change(value, reference):
    return None if value is None or reference in (None, 0) else 100 * (value / reference - 1)


def render_table(rows, reconstruction=False):
    lines = ["```text", "dataset  method           seed    path                         MSE          MAE       MSE_vsRaw%  MAE_vsRaw%"]
    for row in rows:
        if reconstruction != (row["stage"] == "reconstruction"):
            continue
        lines.append(f"{row['dataset']:<8} {row['method']:<16} {str(row['seed']):<7} "
                     f"{row['arm']:<28} {fmt(row['mse']):>11} {fmt(row['mae']):>11} "
                     f"{fmt(row.get('mse_vs_raw_percent')):>11} {fmt(row.get('mae_vs_raw_percent')):>11}")
    lines.append("```")
    return "\n".join(lines)


def figure_footer(fig, record):
    fig.text(.02, .012, f"Fixed origin rows 0/middle/last; channels 0/middle/last. "
             f"Contract {str(record['truth']['data_contract_hash'])[:12]} | "
             f"prediction SHA {record['prediction_sha256'][:12]}", fontsize=8)


def plot_job(record, baseline, figures_dir, figure_index, resources, warnings):
    result, arrays = record["result"], record["arrays"]
    job = result["job"]
    identifier = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(job["id"]))
    data = arrays["targets"]
    indices = fixed_indices(len(data))
    channels = fixed_indices(data.shape[-1])
    inputs = {"result_sha256": record["result_sha256"],
              "prediction_sha256": record["prediction_sha256"],
              "baseline_prediction_sha256": baseline["prediction_sha256"] if baseline else None,
              "diagnostic_sha256": None}
    diagnostic = None
    if result.get("diagnostic_path"):
        diagnostic_path = artifact_path(result["diagnostic_path"])
        inputs["diagnostic_sha256"] = digest(diagnostic_path)
        if result.get("diagnostic_sha256") and inputs["diagnostic_sha256"] != result["diagnostic_sha256"]:
            raise ValueError("Diagnostic SHA differs from the actual result")
        with np.load(diagnostic_path, allow_pickle=False) as archive:
            diagnostic = {name: archive[name] for name in archive.files}
    old = figure_index.get(identifier, {})
    if old.get("inputs") == inputs and all((HERE / name).exists() for name in old.get("files", [])):
        return old["files"]
    files = []
    label = f"{job['dataset']} / {method_label(job)} / seed {job['seed']}"

    def save(fig, suffix):
        file = figures_dir / f"{identifier}_{suffix}.png"
        fig.savefig(file, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        files.append(file.relative_to(HERE).as_posix())
        resources.check()

    fig, axes = plt.subplots(len(indices), len(channels), figsize=(16, 10), squeeze=False)
    for row, index in enumerate(indices):
        for column, channel in enumerate(channels):
            ax = axes[row, column]
            target = np.where(arrays["mask"][index, :, channel], data[index, :, channel], np.nan)
            ax.plot(np.arange(1, 49), target, color="black", lw=2, label="Observed target")
            for source in ([baseline] if baseline else []) + [record]:
                for arm in source["forecast_metrics"]:
                    curve = source["arrays"][arm][index, :, channel]
                    linestyle = "--" if "PERSIST" in arm or arm == "MEAN_BLOCK_F" else "-"
                    ax.plot(np.arange(1, 49), curve, color=COLORS[arm], ls=linestyle,
                            lw=1.3, alpha=.85, label=arm)
            ax.set_title(f"origin row {index} (t={arrays['origins'][index]}) / channel {channel}", fontsize=10)
            ax.set_xlabel("Physical forecast step")
            ax.set_ylabel("TRAIN-standardized value")
            ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5, .025), fontsize=8)
    fig.suptitle(f"Actual future predictions — {label}", fontsize=15)
    fig.tight_layout(rect=[0, .09, 1, .955])
    figure_footer(fig, record)
    save(fig, "forecast")

    if diagnostic:
        past, past_mask = diagnostic["past"], diagnostic["past_mask"].astype(bool)
        reconstruction = diagnostic["reconstruction"]
        chosen = np.asarray(diagnostic.get("indices", indices), dtype=int).tolist()
        if chosen != indices or len(past) != len(indices):
            raise ValueError("Diagnostic examples must be preselected full-VAL rows 0/middle/last")
        if past.shape != past_mask.shape or past.shape != reconstruction.shape:
            raise ValueError("Diagnostic reconstruction and original mask shapes differ")
        check_subset = independent_metric(reconstruction, past, past_mask)
        fig, axes = plt.subplots(len(indices), len(channels), figsize=(16, 9), squeeze=False)
        for row, index in enumerate(indices):
            for column, channel in enumerate(channels):
                ax = axes[row, column]
                x_axis = np.arange(-past.shape[1], 0)
                ax.plot(x_axis, np.where(past_mask[row, :, channel], past[row, :, channel], np.nan),
                        color="black", lw=1.4, label="Observed past")
                ax.plot(x_axis, reconstruction[row, :, channel], color="#E69F00", lw=1.1,
                        label="Reconstruction-trained native D")
                ax.set_title(f"origin row {index} / channel {channel}", fontsize=10)
                ax.set_xlabel("Physical step relative to forecast origin")
                ax.set_ylabel("TRAIN-standardized value")
                ax.grid(alpha=.18)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2)
        fig.suptitle(f"Past reconstruction — {label}\nFixed-example subset MSE {fmt(check_subset['mse'])}; "
                     f"MAE {fmt(check_subset['mae'])}; not the full-VAL reconstruction metric", fontsize=13)
        fig.tight_layout(rect=[0, .055, 1, .93])
        save(fig, "reconstruction")
        latent = diagnostic["latent_history"]
        latent = latent.reshape(latent.shape[0], latent.shape[1], -1)
        label_extra = "representation expansion (d320)" if job.get("dim") == 320 else "latent representation"
        fig, axes = plt.subplots(len(indices), 1, figsize=(15, 8), squeeze=False)
        for row, index in enumerate(indices):
            ax = axes[row, 0]
            for coordinate in range(min(8, latent.shape[-1])):
                ax.plot(latent[row, :, coordinate], label=f"coordinate {coordinate}", lw=1)
            ax.set_title(f"origin row {index}; first {min(8, latent.shape[-1])} fixed coordinates", fontsize=10)
            ax.set_xlabel("Latent time tick")
            ax.set_ylabel("Latent coordinate value")
            ax.grid(alpha=.18)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=8, fontsize=8)
        fig.suptitle(f"Actual latent history — {label} / {label_extra}", fontsize=14)
        fig.tight_layout(rect=[0, .05, 1, .96])
        save(fig, "latent_curves")
        fig, axes = plt.subplots(len(indices), 1, figsize=(15, 10), squeeze=False)
        for row, index in enumerate(indices):
            ax = axes[row, 0]
            bound = max(float(np.max(np.abs(latent[row]))), 1e-12)
            image = ax.imshow(latent[row].T, aspect="auto", origin="lower", cmap="coolwarm",
                              vmin=-bound, vmax=bound, interpolation="nearest")
            fig.colorbar(image, ax=ax, label="Latent value (own full-range scale)")
            ax.set_title(f"origin row {index}; all {latent.shape[-1]} coordinates; no clipping", fontsize=10)
            ax.set_xlabel("Latent time tick")
            ax.set_ylabel("Fixed coordinate index")
        fig.suptitle(f"Latent heatmap — {label} / {label_extra}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, .96])
        save(fig, "latent_heatmap")
    else:
        warnings.append(f"No actual diagnostic archive yet for {identifier}; reconstruction/latent figures pending")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for source in ([baseline] if baseline else []) + [record]:
        for arm, metrics in source["forecast_metrics"].items():
            for index, metric in enumerate(("mse", "mae")):
                axes[index].plot(np.arange(1, 49), metrics[f"horizon_{metric}"],
                                 color=COLORS[arm], label=arm,
                                 ls="--" if "PERSIST" in arm or arm == "MEAN_BLOCK_F" else "-")
                axes[index].set_xlabel("Physical forecast step")
                axes[index].set_ylabel(f"Observed channel-macro {metric.upper()}")
                axes[index].grid(alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"All-VAL horizon errors — {label}", fontsize=14)
    fig.tight_layout(rect=[0, .18, 1, .92])
    save(fig, "horizon_errors")
    figure_index[identifier] = {"inputs": inputs, "truth": record["truth"], "files": files,
                                "sha256": {name: digest(HERE / name) for name in files},
                                "fixed_origin_rows": indices, "fixed_channel_indices": channels,
                                "selection_rule": "0/middle/last fixed before looking at error"}
    return files


def build_report(records, baselines, rows, coverage, figures, verification, data_checks, extras, attempts):
    now = datetime.now(timezone.utc).isoformat()
    valid = [record for record in records if record.get("valid")]
    lines = ["# 교수님 제안 직접 실험 — 실제 결과 중간 보고", "",
             f"갱신 UTC: {now}. [확인] 실제 결과와 저장 예측 검산을 통과한 representation fit은 "
             f"**{len(valid)}/16**개다. 데이터 baseline은 **{len(baselines)}/2**개 자료에서 확보했다. "
             "이 문서는 누적 중간 보고이며 전체 목표 완료 선언이 아니다.", "",
             "[확인] canonical 집계는 `protocol.json`에 명시된 run ID와 실제 `result.job.id`가 일치한 결과만 사용한다. "
             f"추가 완료 actual run receipt는 **{verification['extra_completed_actual_runs']}**개이며 "
             "[EXTRA_ACTUAL_RUNS.json](EXTRA_ACTUAL_RUNS.json)에 원래 ID·결과 SHA·수치·비용·제외 이유를 보존한다. "
             "같은 조건의 수정 전 run과 재실행을 두 seed로 세거나 평균에 함께 넣지 않는다. "
             "아직 새 canonical ID의 결과가 없으면 해당 조건은 미완료다.", "",
             f"실패 attempt **{verification['preserved_failed_attempts']}**개와 실행 프로세스 없이 남은 부분 artifact "
             f"**{verification['preserved_partial_artifact_attempts']}**개도 "
             "[EXTRA_FAILED_ATTEMPTS.json](EXTRA_FAILED_ATTEMPTS.json)에 보존한다. "
             "자원 guard 중단이나 partial 학습을 미실행 또는 완료 fit으로 바꾸지 않는다.", "",
             "목표는 GTConvAE/TS2Vec의 표현학습·복원 경로가 frozen TSFM 이후 다변량 미래를 복원할 수 있는지 "
             "확인하는 것이다. 기존 LEVEL, 잔차 G, 마지막 잔차값 유지, PCA/D8로 복귀하지 않는다.", "",
             "## 원형과 이번 적용", "",
             "[구현 설정] GT는 product graph와 polynomial filter를 E/D 양쪽에 사용하고 시간 downsampling/upsampling을 유지한다. "
             "1→8→2 / 2→8→1, order4·5tap, R4다. 원문 Fig1(B)과 Section2.3의 순서 차이는 "
             "zero insertion→filter로 해석했다. 마지막 출력 linear는 signed standardized 자료에 맞춘 변경이다. "
             "isolated reconstruction 앞3시점의 structural zero와 right-edge phase를 명시하며 미래에서는 "
             "observed latent history를 이어 decoder를 호출한다. [GT 원문 대비 변경](GT_METHOD_DELTA.md).", "",
             "[구현 설정] TS2Vec는 공식 pinned encoder hidden64/depth10을 사용한다. d320은 공식 API 기본 출력 차원의 "
             "표현 확장 대조이며 d16은 이번 저차원 적용이다. 별도 d→128→C nonlinear reconstruction D를 학습한다. "
             "원논문의 last representation→ridge forecast는 별도 정상 사용 대조다. E→F→D는 offline window "
             "표현이고 native ridge는 causal sliding last 표현이라는 export 차이도 구분한다. "
             "[TS2 변경](TS2_METHOD_DELTA.md), [소스·재현 안내](SOURCE_README.md).", "",
             "A는 재구성으로 학습한 D를 그대로 쓰는 transfer이며 B는 E/F를 고정한 뒤 actual F TRAIN "
             "latent 예측으로 **동일 native D**를 future-supervised calibration한다. B의 이득은 순수 비지도 "
             "학습의 이득으로 표시하지 않는다. latent 마지막값 유지도 같은 D를 통한 부품 교체 진단이다.", "",
             "## 데이터와 평가", "",
             "TRAIN-only 관측 정규화·상관 graph, causal forward-fill 입력, original observed target mask를 사용한다. "
             "MSE/MAE는 채널별 관측 평균을 낸 뒤 채널 평균으로 계산한다. 재구성과 미래 예측을 별도 구분한다. "
             "동일 자료의 모든 예측은 동일 outer VAL origins/targets/mask/정규화 계약에 맞춘다. "
             "기존 VAL은 반복 사용된 개발 구간이고 두 seed는 초기 재현 확인이며 독립 TEST·유의성 증거가 아니다. "
             "이번 분석은 저장된 신규 예측/진단 NPZ와 JSON만 읽으며 모델·weight·TEST를 열지 않는다. "
             "[데이터 계약](DATA_NOTE_KO.md), [검산](ANALYSIS_VERIFICATION.json).", "", "```text",
             "dataset  TRAIN bounds     VAL bounds       rec TRAIN windows  F TRAIN origins  VAL origins"]
    for dataset, checked in data_checks.get("datasets", {}).items():
        oc = checked.get("origin_checks", {})
        lines.append(f"{dataset:<8} {str(checked.get('bounds',{}).get('train')):<16} "
                     f"{str(checked.get('bounds',{}).get('val')):<16} "
                     f"{oc.get('train_window_starts',{}).get('count','?')!s:<18} "
                     f"{oc.get('train_forecast_origins',{}).get('count','?')!s:<16} "
                     f"{oc.get('val_origins',{}).get('count','?')}")
    lines += ["```", "", "이전 LEVEL 전체 inventory에서 보호된 Electricity tail까지 읽은 "
              "[접근 위반 기록](../level_compressor_fullscreen_20261011/PROTECTED_ACCESS_INCIDENT.json)을 보존한다. "
              "새 direct loader/분석의 TEST 접근0은 그 과거 사건을 되돌리거나 전체 campaign guard를 PASS로 바꾸지 않는다.", "",
              "정확한 달력 날짜·원자료 provenance는 [DATA_PROVENANCE.json](DATA_PROVENANCE.json)에 있다. "
              "이번 입력 context는 L512이고 physical future는 H48이다. GT latent는 T128×C×2로 input scalar의 절반이고 "
              "F 행은 C×2다. TS2Vec의 time length512와 d16/d320은 별도 조건이며 embedding 좌표를 시간축으로 연결하지 않는다.",
              "", "## Q1. 표현과 복원은 실제로 학습되는가?", "",
              render_table(rows, reconstruction=True), "",
              "위 값은 result.json의 실제 재구성 metric이다. 그림의 고정3개 예시 재구성 오차는 "
              "full-VAL metric과 구분한다. CPU tiny overfit/shape 검사는 과학적 fit 수에 포함하지 않는다.", "",
              "## Q2. E→F→D는 같은 원공간 미래를 예측하는가?", "",
              render_table(rows), "",
              "vsRaw%의 음수는 해당 metric의 개선이다. MSE와 MAE가 엇갈리면 교환관계로 보고한다. "
              "임의 5% 합격선을 두거나 확장 표현 d320을 채널 압축이라고 부르지 않는다.", "",
              "## Q3. 재구성 D 재사용과 실제 F 출력에 적응한 D는 어떻게 다른가?", ""]
    if not valid:
        lines.append("실제 run의 transfer/calibrated 미래 예측이 아직 없어서 판단하지 않는다.")
    for record in valid:
        job, metrics = record["result"]["job"], record["forecast_metrics"]
        a, b = metrics.get("TRANSFER_F"), metrics.get("CALIBRATED_F")
        if a and b:
            mse = relative_change(b["mse"], a["mse"])
            mae = relative_change(b["mae"], a["mae"])
            lines.append(f"- {job['dataset']} / {method_label(job)} / seed {job['seed']}: "
                         f"B vs A MSE {fmt(mse)}%, MAE {fmt(mae)}%. "
                         f"[실제 training/calibration/checkpoint 기록]({Path(record['result_path']).relative_to(HERE.relative_to(ROOT)).as_posix()}).")
    lines += ["", "## Q4. GT의 공간 연결은 추가로 도움이 되는가?", ""]
    grouped = {job_key(record["result"]["job"]): record for record in valid}
    spatial_pairs = 0
    for dataset in ("robin", "jena"):
        for seed in (92601, 92602):
            full = grouped.get((dataset, seed, "GT", False))
            no = grouped.get((dataset, seed, "GT", True))
            if full and no:
                spatial_pairs += 1
                for arm in ("TRANSFER_F", "CALIBRATED_F"):
                    if arm in full["forecast_metrics"] and arm in no["forecast_metrics"]:
                        a, b = full["forecast_metrics"][arm], no["forecast_metrics"][arm]
                        lines.append(f"- {dataset} / seed {seed} / {arm}: full spatial vs no-spatial "
                                     f"MSE {fmt(relative_change(a['mse'],b['mse']))}%, "
                                     f"MAE {fmt(relative_change(a['mae'],b['mae']))}%.")
    if not spatial_pairs:
        lines.append("full/no-spatial 양쪽 실제 재학습과 미래 추론이 갖춰진 seed pair가 아직 없어 판단하지 않는다.")
    lines += ["", "## seed 평균과 비용", "", "```text",
              "dataset  method           arm                          n_seed     mean_MSE     mean_MAE"]
    groups = {}
    for row in rows:
        if row["stage"] == "forecast" and row["seed"] != "baseline":
            groups.setdefault((row["dataset"], row["method"], row["arm"]), []).append(row)
    for key, group in sorted(groups.items()):
        lines.append(f"{key[0]:<8} {key[1]:<16} {key[2]:<28} {len(group):<10} "
                     f"{fmt(np.mean([r['mse'] for r in group])):>12} "
                     f"{fmt(np.mean([r['mae'] for r in group])):>12}")
    lines += ["```", "", "n_seed=1은 한 seed 값이고 n_seed=2도 모집단 안정성을 보장하지 않는다. "
              "E/D/F/calibration 시간·메모리는 실제 각 result.json의 training/calibration/cost/resources에 보존한다. "
              "batch1/batch32 경로 비용이 없는 run은 비용 판단 미완료로 남긴다. "
              "GT의 shorter context와 늘어난 F 행, TS2 d320의 representation expansion 비용을 구분한다. "
              "현재 메모리 측정은 각 prototype process에서 동시에 상주하는 모듈을 포함한다. "
              "한 경로만 배포한 isolated deployment의 메모리 절감 증거로 확대하지 않는다.", "",
              "```text", "dataset  method                       batch  p50_ms       p95_ms       CUDA_alloc_MiB"]
    for record in list(baselines.values()) + valid:
        for cost in record["result"].get("cost", []):
            lines.append(f"{record['dataset']:<8} {cost.get('method','?'):<28} "
                         f"{str(cost.get('batch','?')):<6} {fmt(cost.get('p50_ms')):>12} "
                         f"{fmt(cost.get('p95_ms')):>12} "
                         f"{fmt(cost['peak_cuda_allocated_bytes']/1048576 if cost.get('peak_cuda_allocated_bytes') is not None else None):>14}")
    lines += ["```", ""]
    for record in valid:
        job = record["result"]["job"]
        relative = Path(record["result_path"]).relative_to(HERE.relative_to(ROOT)).as_posix()
        training = record["result"].get("training", {})
        stages = [("joint_GT_E_D", training)] if job["family"] == "GT" else [
            ("TS2_encoder", training.get("encoder", {})),
            ("reconstruction_D", training.get("decoder_reconstruction", {}))]
        stages.append(("forecast_calibration_D", record["result"].get("calibration", {})))
        lines += [f"### {job['id']}", "", f"[actual result]({relative})", "",
                  "```text", "stage                      epochs  selected_epoch  max_epochs  updates   seconds"]
        cap_selected = []
        for name, stage in stages:
            epochs = stage.get("epochs")
            selected = stage.get("selected_epoch", stage.get("best_epoch"))
            maximum = stage.get("maximum_epochs", stage.get("max_epochs"))
            updates = stage.get("actual_optimizer_updates", stage.get("optimizer_updates",
                         stage.get("configuration", {}).get("n_iters")))
            seconds = stage.get("seconds")
            if seconds is None:
                matching = [item.get("wall_seconds", 0) for item in record["result"].get("resources", {}).get("stages", [])
                            if (name == "TS2_encoder" and "representation" in item.get("name", ""))
                            or (name == "reconstruction_D" and "reconstruction_decoder" in item.get("name", ""))
                            or (name == "forecast_calibration_D" and "calibration" in item.get("name", ""))]
                seconds = sum(matching) if matching else None
            lines.append(f"{name:<26} {str(epochs):<7} {str(selected):<15} {str(maximum):<11} "
                         f"{str(updates):<9} {fmt(seconds)}")
            if selected is not None and maximum is not None and selected >= maximum:
                cap_selected.append(name)
        lines += ["```", ""]
        if cap_selected:
            lines += ["[확인] selected checkpoint가 학습 상한에 닿은 단계: " + ", ".join(cap_selected)
                      + ". 이 범위에서 실제 학습은 수행했지만 충분한 수렴을 확인했다고 표시하지 않는다.", ""]
        lines += [
                  "실제 shape:", "", "```json", json.dumps(record["result"].get("shape", {}), ensure_ascii=False, indent=2), "```", "",
                  "기록된 실제 비용:", "", "```json", json.dumps(record["result"].get("cost", []), ensure_ascii=False, indent=2), "```", ""]
        for file in figures.get(job["id"], []):
            lines += [f"![{job['id']} {Path(file).stem}]({file})", ""]
    if extras:
        lines += ["## Canonical 집계 밖에 보존한 실제 실행", "",
                  "다음 값은 원래 result receipt의 보존 수치이며 이번 canonical 예측 검산·seed 평균·16fit 완료 수에 포함하지 않는다. "
                  "epoch0 selection 수정 전 TS2 실행은 재학습으로 대체되며 원래 파일과 그림을 보존한다.", "",
                  "```text", "actual_run_id                                                 status       REC_MSE      A_MSE        B_MSE"]
        for extra in extras:
            reconstruction = extra.get("reported_reconstruction") or {}
            forecast = extra.get("reported_forecast") or {}
            lines.append(f"{str(extra['id']):<61} {str(extra['status']):<12} "
                         f"{fmt(reconstruction.get('mse')):>11} "
                         f"{fmt(forecast.get('TRANSFER_F',{}).get('mse')):>11} "
                         f"{fmt(forecast.get('CALIBRATED_F',{}).get('mse')):>11}")
        lines += ["```", ""]
        for extra in extras:
            relative = Path(extra["result_path"]).relative_to(HERE.relative_to(ROOT)).as_posix()
            lines += [f"- [{extra['id']} actual receipt]({relative}): {extra['reason']}; "
                      f"현재 요구 ID `{extra.get('required_protocol_id')}`; result SHA `{extra['result_sha256']}`."]
        lines += [""]
    stopped = [attempt for attempt in attempts.values()
               if attempt["status"] in ("FAILED_PRESERVED_PATH", "PARTIAL_ARTIFACTS")]
    if stopped:
        lines += ["## 실패와 부분 학습을 보존한 attempt", "",
                  "다음 attempt는 실제 완료 미래 성능이 없어 metric을 만들거나 성공 fit으로 세지 않는다. "
                  "protocol이 새 retry ID로 바뀌어도 원래 실패·자원 사용·부분 iteration을 계속 표시한다. "
                  "메모리 원인은 receipt의 RAM/allocated VRAM 값을 각각 읽으며 guard 문구만으로 VRAM 단독 초과를 단정하지 않는다.", "",
                  "```text", "actual_run_id                                                status                 iters/planned   batch   peak_RSS_MiB  peak_alloc_MiB"]
        for attempt in stopped:
            resource = attempt.get("resources") or {}
            iterations = f"{attempt.get('actual_encoder_iterations')}/{attempt.get('expected_encoder_iterations')}"
            lines.append(f"{attempt['id']:<60} {attempt['status']:<22} {iterations:<15} "
                         f"{str(attempt.get('encoder_batch_size')):<7} "
                         f"{fmt(resource['peak_rss_bytes']/1048576 if resource.get('peak_rss_bytes') is not None else None):>13} "
                         f"{fmt(resource['peak_cuda_allocated_bytes']/1048576 if resource.get('peak_cuda_allocated_bytes') is not None else None):>14}")
        lines += ["```", ""]
        for attempt in stopped:
            path = attempt.get("failure_path") or attempt.get("initial_protocol_path")
            relative = Path(path).relative_to(HERE.relative_to(ROOT)).as_posix() if path and Path(path).is_relative_to(HERE.relative_to(ROOT)) else None
            label = f"[{attempt['id']} preserved receipt]({relative})" if relative else f"`{attempt['id']}`"
            lines += [f"- {label}: {attempt['reason']}. Iteration 근거: {attempt['iteration_evidence']}. "
                      f"현재 요구 ID `{attempt.get('required_protocol_id')}`; 재개 안내: {attempt.get('resume') or '새 ID와 원래 protocol을 확인해 실제 재학습; 부분 weight를 optimizer resume로 가정하지 않음'}."]
        lines += [""]
    lines += ["## 실행 상태와 남은 작업", "", "```text", "dataset  method           seed    actual_status"]
    for item in coverage:
        job = item["job"]
        lines.append(f"{job['dataset']:<8} {method_label(job):<16} {str(job['seed']):<7} {item['status']}")
    lines += ["```", "", "[전체 실행 목록](runs.json), [실행 CSV](runs.csv), [seed별 metric CSV](metrics.csv), "
              "[그림의 입력 hash와 고정 선택 규칙](FIGURE_MANIFEST.json), "
              "[집계 밖 실제 실행 기록](EXTRA_ACTUAL_RUNS.json), "
              "[실패·부분 attempt 기록](EXTRA_FAILED_ATTEMPTS.json).", "",
              f"독립 저장 예측 검산: {verification['status']}; errors {len(verification['errors'])}, "
              f"warnings {len(verification['warnings'])}. 미실행·실행중·실패·artifact 불일치는 완료 수에 넣지 않는다. "
              "모든16 representation fit, 두 자료 baseline, A/B/persistence/native head, 비용과 최종 검산을 "
              "모으기 전에는 전체 완료라고 표시하지 않는다. 점수가 나쁘다는 이유로 다른 주 방법을 생략하거나 LEVEL로 복귀하지 않는다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    resources = Resources()
    errors, warnings, truth = [], [], {}
    with resources.stage("saved_actual_predictions_CPU_report", gpu=False):
        data_checks = read_json(HERE / "DATA_CHECKS.json") if (HERE / "DATA_CHECKS.json").exists() else {}
        protocol = read_json(HERE / "protocol.json") if (HERE / "protocol.json").exists() else {}
        planned, plan_source = planned_jobs(protocol)
        baselines = {}
        for path in sorted((HERE / "baselines").glob("*/result.json")):
            record = load_result(path, data_checks, truth, errors, warnings)
            if record.get("valid"):
                baselines[record["dataset"]] = record
        summaries = [{"result_path": path.relative_to(ROOT).as_posix(),
                      "result_sha256": digest(path), "result": read_json(path)}
                     for path in sorted((HERE / "runs").glob("*/result.json"))]
        canonical, extras = select_canonical_results(planned, summaries, errors, warnings)
        records = [load_result(ROOT / summary["result_path"], data_checks, truth, errors, warnings)
                   for summary in canonical]
        actual = {record["result"]["job"]["id"]: record for record in records}
        live = current_live_jobs()
        attempts = collect_attempt_states(planned, live, warnings)
        coverage = [coverage_item(job, actual.get(job.get("id")), live.get(job.get("id")),
                                  attempts.get(job.get("id"))) for job in planned]
        rows = []
        for record in list(baselines.values()) + [record for record in records if record.get("valid")]:
            result = record["result"]
            dataset = record["dataset"]
            job = result.get("job", {})
            baseline = str(job.get("family", "")).upper() not in ("GT", "TS2")
            method = "baseline" if baseline else method_label(job)
            seed = "baseline" if baseline else job["seed"]
            identifier = f"baseline_{dataset}" if baseline else job["id"]
            raw = baselines.get(dataset, {}).get("forecast_metrics", {}).get("RAW_F", {})
            if result.get("reconstruction"):
                rec = result["reconstruction"]
                rows.append({"dataset": dataset, "method": method, "seed": seed, "job_id": identifier,
                             "stage": "reconstruction", "arm": "REC_NATIVE_D", "mse": rec.get("mse"),
                             "mae": rec.get("mae"), "origin_count": rec.get("origin_count"),
                             "metric_evidence": "runner full reconstruction receipt; saved figures are fixed subset",
                             "data_contract_hash": result.get("data_contract_hash"), "result_path": record["result_path"]})
            for arm, metric in record["forecast_metrics"].items():
                rows.append({"dataset": dataset, "method": method, "seed": seed, "job_id": identifier,
                             "stage": "forecast", "arm": arm, "mse": metric["mse"], "mae": metric["mae"],
                             "origin_count": metric["origin_count"], "observed_count_total": metric["observed_count_total"],
                             "mse_vs_raw_percent": relative_change(metric["mse"], raw.get("mse")),
                             "mae_vs_raw_percent": relative_change(metric["mae"], raw.get("mae")),
                             "metric_evidence": "independently recalculated saved actual prediction NPZ",
                             "channel_mse": json.dumps(metric["channel_mse"]),
                             "channel_mae": json.dumps(metric["channel_mae"]),
                             "horizon_mse": json.dumps(metric.get("horizon_mse", [])),
                             "horizon_mae": json.dumps(metric.get("horizon_mae", [])),
                             "physical_channel_mae": json.dumps(metric.get("physical_channel_mae", [])),
                             "data_contract_hash": result.get("data_contract_hash"),
                             "result_path": record["result_path"], "prediction_sha256": record["prediction_sha256"]})
        fields = ["dataset", "method", "seed", "job_id", "stage", "arm", "mse", "mae", "origin_count",
                  "observed_count_total", "mse_vs_raw_percent", "mae_vs_raw_percent", "data_contract_hash",
                  "prediction_sha256", "channel_mse", "channel_mae", "horizon_mse", "horizon_mae",
                  "physical_channel_mae", "metric_evidence", "result_path"]
        save_csv(HERE / "metrics.csv", rows, fields)
        run_rows = [{"dataset": item["job"]["dataset"], "method": method_label(item["job"]),
                     "seed": item["job"]["seed"], "job_id": item["job"].get("id", ""),
                     "status": item["status"], "execution_kind": item["execution_kind"],
                     "result_path": item["result_path"], "failure_path": item["failure_path"],
                     "actual_encoder_iterations": item["actual_encoder_iterations"],
                     "expected_encoder_iterations": item["expected_encoder_iterations"],
                     "reason": item["reason"]} for item in coverage]
        save_csv(HERE / "runs.csv", run_rows, ["dataset", "method", "seed", "job_id", "status", "execution_kind", "result_path",
                                               "failure_path", "actual_encoder_iterations", "expected_encoder_iterations", "reason"])
        figures_dir = HERE / "figures"
        figure_index = read_json(HERE / "FIGURE_MANIFEST.json") if (HERE / "FIGURE_MANIFEST.json").exists() else {}
        figures = {}
        if not args.tables_only:
            figures_dir.mkdir(exist_ok=True)
            for record in records:
                if record.get("valid"):
                    try:
                        files = plot_job(record, baselines.get(record["dataset"]), figures_dir,
                                         figure_index, resources, warnings)
                        figures[record["result"]["job"]["id"]] = files
                    except (OSError, ValueError, KeyError) as error:
                        errors.append({"path": record["result_path"], "figure_error": repr(error)})
        write_json(HERE / "FIGURE_MANIFEST.json", figure_index)
        measured_records = sum(record.get("valid", False) for record in records)
        extra_completed = sum(extra.get("status") == "complete" for extra in extras)
        failed = [attempt for attempt in attempts.values() if attempt.get("failure_path")]
        partial = [attempt for attempt in attempts.values() if attempt["status"] == "PARTIAL_ARTIFACTS"]
        write_json(HERE / "EXTRA_FAILED_ATTEMPTS.json", {
            "scope": "All current and replaced failed attempts, plus non-live partial protocol/curve artifacts; never complete scientific fits",
            "preserved_failed_attempts": len(failed), "preserved_partial_artifact_attempts": len(partial),
            "failed_attempts": failed, "partial_artifact_attempts": partial,
            "protocol_sha256": digest(HERE / "protocol.json") if (HERE / "protocol.json").exists() else None,
            "source_sha256": digest(HERE / "analyze_direct.py"),
            "model_loads": 0, "weight_loads": 0, "protected_TEST_reads": 0,
            "created_UTC": datetime.now(timezone.utc).isoformat()})
        write_json(HERE / "EXTRA_ACTUAL_RUNS.json", {
            "selection_rule": "Only explicit protocol jobs IDs matching result.job.id enter canonical metrics, coverage and seed means",
            "extra_completed_actual_runs": extra_completed, "extra_result_receipts": len(extras),
            "all_completed_actual_result_receipts": sum(summary["result"].get("status") == "complete" for summary in summaries),
            "canonical_protocol_ids": [job.get("id") for job in planned],
            "runs": extras, "protocol_sha256": digest(HERE / "protocol.json") if (HERE / "protocol.json").exists() else None,
            "source_sha256": digest(HERE / "analyze_direct.py"),
            "created_UTC": datetime.now(timezone.utc).isoformat()})
        status = "FAIL" if errors else ("PASS_SAVED_PREDICTIONS_SCOPE" if measured_records or baselines else "NO_ACTUAL_RESULTS_YET")
        verification = {"status": status,
                        "scope": "Saved actual forecast arrays and shared development truth; not full training/inference completion",
                        "errors": errors, "warnings": warnings, "shared_truth": truth,
                        "verified_representation_fits": measured_records,
                        "canonical_selection": "explicit protocol run ID and matching job condition",
                        "extra_completed_actual_runs": extra_completed,
                        "extra_actual_runs_manifest": "EXTRA_ACTUAL_RUNS.json",
                        "preserved_failed_attempts": len(failed),
                        "preserved_partial_artifact_attempts": len(partial),
                        "failed_attempts_manifest": "EXTRA_FAILED_ATTEMPTS.json",
                        "verified_baseline_datasets": len(baselines), "forecast_metric_rows": sum(row["stage"] == "forecast" for row in rows),
                        "model_loads": 0, "weight_loads": 0, "protected_TEST_reads": 0,
                        "source_pins": {name: digest(HERE / name) for name in ("analyze_direct.py", "common_direct.py")},
                        "created_UTC": datetime.now(timezone.utc).isoformat()}
        write_json(HERE / "runs.json", {"status": "PARTIAL_EXECUTION_EVIDENCE",
                                        "plan_source": plan_source, "planned_representation_fits": len(planned),
                                        "runs": coverage, "baseline_datasets": sorted(baselines),
                                        "extra_completed_actual_runs": extra_completed,
                                        "extra_actual_runs_manifest": "EXTRA_ACTUAL_RUNS.json",
                                        "preserved_failed_attempts": len(failed),
                                        "preserved_partial_artifact_attempts": len(partial),
                                        "failed_attempts_manifest": "EXTRA_FAILED_ATTEMPTS.json",
                                        "baseline_comparisons_required": list(BASE_ARMS),
                                        "all_core_implementation_and_budget_audits": "pending"})
        report = build_report(records, baselines, rows, coverage, figures, verification, data_checks, extras, attempts)
        (HERE / "REPORT_KO.md").write_text(report, encoding="utf-8")
        if measured_records:
            (HERE / "INTERIM_RESULTS_KO.md").write_text(report, encoding="utf-8")
        resources.check()
    verification["resources"] = resources.value
    write_json(HERE / "ANALYSIS_VERIFICATION.json", verification)
    print(json.dumps({"status": verification["status"], "verified_representation_fits": verification["verified_representation_fits"],
                      "verified_baseline_datasets": len(baselines), "metric_rows": len(rows),
                      "extra_completed_actual_runs": verification["extra_completed_actual_runs"],
                      "preserved_failed_attempts": verification["preserved_failed_attempts"],
                      "preserved_partial_artifact_attempts": verification["preserved_partial_artifact_attempts"],
                      "figure_files": sum(len(value) for value in figures.values()),
                      "errors": len(errors), "warnings": len(warnings)}, ensure_ascii=False))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
