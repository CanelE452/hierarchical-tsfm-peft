"""A_fit-only descriptive channel groups and independent FP64 group errors."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import audit_predictions as audit
import data_contract as data


PATTERNS = ("variability", "seasonality", "delayed_crosschannel", "intermittency")
LAGS = (1, 2, 4, 8, 16, 32)
MIN_PAIRS = 30
TIE_ATOL = 1e-12
COMPARISONS = (("RAW_RIDGE", "CURRENT"), ("RAW_RIDGE", "SEASONAL_NAIVE"), ("SEASONAL_NAIVE", "CURRENT"),
    ("B0", "RAW_BOLT_SMALL"), ("B2", "B0"), ("B4", "B2"), ("DIRECT_G", "RAW_BOLT_SMALL"),
    ("LORA_R8", "RAW_BOLT_SMALL"), ("CONDITIONAL", "DIRECT_G"), ("CONDITIONAL", "PAST_CONTROL"),
    ("CONDITIONAL", "ALPHA_FIT"), ("RAW_CALIBRATED", "RAW_BOLT_SMALL"), ("LAG_CAUSAL", "B0"), ("LAG_BIDIR", "LAG_CAUSAL"))


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("x", encoding="utf-8", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _snapshot_name(prefix):
    return prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + str(time.time_ns())


def _pearson_pair(a, b, min_pairs=MIN_PAIRS):
    if len(a) < min_pairs:
        return None
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a, b = a - a.mean(), b - b.mean()
    va, vb = float(a @ a), float(b @ b)
    if va == 0 or vb == 0:
        return None
    return float(np.clip(float(a @ b) / math.sqrt(va * vb), -1., 1.))


def delayed_correlation_matrix(values, observed, lag, min_pairs=MIN_PAIRS):
    """rho[target_i,source_j] for source[t-lag] -> target[t]; no fill."""
    x, mask = np.asarray(values, np.float64), np.asarray(observed, bool)
    if x.ndim != 2 or mask.shape != x.shape or lag < 1 or min_pairs < 2:
        raise ValueError("Invalid delayed-correlation axes, lag or pair count")
    if not np.isfinite(x[mask]).all():
        raise ValueError("Observed A values must be finite")
    width = x.shape[1]
    if lag >= len(x):
        return np.full((width, width), np.nan), np.zeros((width, width), np.int64)
    count = mask.sum(0)
    mean = np.divide(np.where(mask, x, 0).sum(0), count, out=np.zeros(width), where=count > 0)
    # Arbitrary A-only centering stabilizes raw moments; pair means below remain exact.
    centered = np.where(mask, x - mean, 0)
    variable = np.asarray([np.ptp(x[:, i][mask[:, i]]) > 0 if count[i] else False for i in range(width)])
    target, source = centered[lag:], centered[:-lag]
    mt, ms = mask[lag:].astype(np.float64), mask[:-lag].astype(np.float64)
    counts = np.rint(mt.T @ ms).astype(np.int64)
    denominator = np.maximum(counts, 1)
    sum_t, sum_s = target.T @ ms, mt.T @ source
    sum_t2, sum_s2 = (target * target).T @ ms, mt.T @ (source * source)
    covariance = target.T @ source - sum_t * sum_s / denominator
    vt, vs = sum_t2 - sum_t * sum_t / denominator, sum_s2 - sum_s * sum_s / denominator
    eligible = (counts >= min_pairs) & variable[:, None] & variable[None, :]
    # Cancellation triggers direct centered recalculation, never removal of a small real variance.
    eps = np.finfo(np.float64).eps * 64
    uncertain = eligible & ((vt <= eps * (np.abs(sum_t2) + sum_t * sum_t / denominator)) |
                            (vs <= eps * (np.abs(sum_s2) + sum_s * sum_s / denominator)))
    rho = np.full((width, width), np.nan)
    stable = eligible & ~uncertain & (vt > 0) & (vs > 0)
    rho[stable] = np.clip(covariance[stable] / np.sqrt(vt[stable] * vs[stable]), -1., 1.)
    for target_id, source_id in np.argwhere(uncertain):
        valid = mask[lag:, target_id] & mask[:-lag, source_id]
        direct = _pearson_pair(target[:, target_id][valid], source[:, source_id][valid], min_pairs)
        if direct is not None:
            rho[target_id, source_id] = direct
    return rho, counts


def median_groups(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    threshold = float(np.median(finite)) if finite else None
    groups = ["NA" if value is None or not math.isfinite(float(value)) else
              ("low" if float(value) <= threshold else "high") for value in values]
    return groups, threshold


def fit_task_patterns(task, fit_end):
    """Return only channel aggregates and A-only median thresholds."""
    if fit_end < 1 or fit_end > len(task["physical"]):
        raise ValueError("Invalid A_fit endpoint")
    x = np.asarray(task["physical"][:fit_end], np.float64)
    mask = np.asarray(task["observed"][:fit_end], bool)
    if x.ndim != 2 or mask.shape != x.shape or not np.isfinite(x[mask]).all():
        raise ValueError("A data/mask contract is invalid")
    period = int(task.get("period") or 0)
    rows = []
    for channel, name in enumerate(task["columns"]):
        values = x[:, channel][mask[:, channel]]
        std = float((values - values[0]).std()) if len(values) else None
        adjacent = mask[1:, channel] & mask[:-1, channel]
        differences = x[1:, channel][adjacent] - x[:-1, channel][adjacent]
        variability = float(differences.std() / std) if std is not None and std > 0 and len(differences) >= 2 else None
        seasonal, seasonal_pairs = None, 0
        if 0 < period < fit_end:
            valid = mask[period:, channel] & mask[:-period, channel]
            seasonal_pairs = int(valid.sum())
            a, b = x[period:, channel][valid], x[:-period, channel][valid]
            # Remove a common large offset before calculating pair-specific means.
            if len(values):
                offset = float(values.mean()); a, b = a - offset, b - offset
            seasonal = _pearson_pair(a, b)
        rows.append(dict(channel_index=channel, channel=str(name), fit_end=int(fit_end),
            observed_A=int(len(values)), std_A=std, variability=variability, variability_pairs=int(adjacent.sum()),
            seasonality=seasonal, seasonal_period=period or None, seasonal_pairs=seasonal_pairs,
            delayed_crosschannel=None, delayed_rho=None, delayed_source_channel=None, delayed_source_index=None,
            delayed_lag=None, delayed_pairs=0, intermittency=float(np.mean(values == 0)) if len(values) else None,
            exact_zero_count=int((values == 0).sum())))
    for lag in LAGS:
        rho, pairs = delayed_correlation_matrix(x, mask, lag)
        np.fill_diagonal(rho, np.nan)
        for target_id, row in enumerate(rows):
            finite = np.flatnonzero(np.isfinite(rho[target_id]))
            if not len(finite):
                continue
            absolute = np.abs(rho[target_id, finite])
            source_id = int(finite[absolute >= float(absolute.max()) - TIE_ATOL][0])
            value = abs(float(rho[target_id, source_id]))
            if row["delayed_crosschannel"] is None or value > row["delayed_crosschannel"] + TIE_ATOL:
                row.update(delayed_crosschannel=value, delayed_rho=float(rho[target_id, source_id]),
                    delayed_source_channel=str(task["columns"][source_id]), delayed_source_index=source_id,
                    delayed_lag=int(lag), delayed_pairs=int(pairs[target_id, source_id]))
    thresholds = {}
    for pattern in PATTERNS:
        groups, threshold = median_groups([row[pattern] for row in rows])
        thresholds[pattern] = dict(median=threshold, finite_channels=sum(group != "NA" for group in groups),
                                   low_channels=groups.count("low"), high_channels=groups.count("high"), NA_channels=groups.count("NA"))
        for row, group in zip(rows, groups):
            row[pattern + "_group"] = group
    return rows, thresholds


def fit_patterns(public_root, cache_root, datasets=("beijing", "jena", "electricity", "pems08"), fold="primary"):
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    public, cache = Path(public_root), Path(cache_root)
    directory = public / "patterns" / _snapshot_name("snapshot_")
    directory.mkdir(parents=True, exist_ok=False)
    all_rows, contracts = [], {}
    for name in datasets:
        task = data.load_task(name, cache_dir=cache)
        fit_end = data.split_ranges(task, fold)["A_fit"][1]
        started = time.perf_counter()
        rows, thresholds = fit_task_patterns(task, fit_end)
        prefix_mask = np.asarray(task["observed"][:fit_end], bool)
        prefix = np.where(prefix_mask, task["physical"][:fit_end], 0)
        all_rows += [dict(dataset=name, fold=fold, **row) for row in rows]
        contracts[name] = dict(data_sha=task["data_sha256"], fit_end=fit_end, fit_scope="A_fit_only_50_percent",
            fit_physical_masked_sha=data.array_sha256(prefix), fit_observation_mask_sha=data.array_sha256(prefix_mask),
            columns=list(task["columns"]), interval_seconds=task.get("interval_seconds"), period=task.get("period"),
            fold=fold, thresholds=thresholds, fit_seconds=time.perf_counter() - started,
            source_kind=task.get("metadata", {}).get("source_kind", "pinned_physical_source"),
            zero_interpretation="observed exact zeros, not proof of intermittent demand",
            exact_zero_ancestry_limit="FP32 inverse normalization may not preserve physical exact zero" if task.get("metadata", {}).get("source_kind") == "legacy_normalized" else None)
        print(json.dumps(dict(event="PATTERN_A_FIT_DONE", dataset=name, channels=len(rows), fit_end=fit_end,
                              seconds=contracts[name]["fit_seconds"])), flush=True)
    path = directory / "CHANNEL_PATTERNS.csv"
    _write_csv(path, all_rows)
    manifest = dict(created_unix=time.time(), source_sha256=data.sha256_file(Path(__file__)),
        data_contract_source_sha256=data.sha256_file(Path(__file__).with_name("data_contract.py")),
        channel_table_file=path.name, channel_table_sha256=data.sha256_file(path), datasets=contracts,
        lag_direction="source_j[t-lag] -> target_i[t]; self excluded", lags=list(LAGS), minimum_correlation_pairs=MIN_PAIRS,
        seasonal_value="signed Pearson at the preexisting fixed task.period; not abs(rho)",
        delayed_value="strongest absolute pairwise Pearson across other channels and registered lags",
        tie_policy="ascending lag then ascending source index for abs-rho ties; lag update requires > best+1e-12",
        variability_definition="population std of genuinely adjacent observed differences / population std of A observations; >=2 pairs",
        threshold_rule="within-dataset finite median; low <= median; high > median; NA kept separately",
        wall_seconds=time.perf_counter() - started_wall, CPU_process_seconds=time.process_time() - started_cpu,
        runtime_environment={key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        fitting_used_M_V_T_targets=False, graph_learning_performed=False, causation_claimed=False, deployment_selector_created=False,
        public_full_raw_arrays=False, public_pairwise_matrices=False, GPU_execution=False)
    _write_json(directory / "PATTERN_MANIFEST.json", manifest)
    (directory / "README_KO.md").write_text("# A_fit 전용 채널 패턴\n\n패턴과 median은 A_fit 50% 관측값만 사용해 고정했다. 미래 성능에 맞춰 경계·채널을 선택하지 않았다. 상관은 인과성·필요 정보·학습 그래프의 증거가 아니다.\n\n- [채널 요약](CHANNEL_PATTERNS.csv)\n- [입력·임계값·구현 증빙](PATTERN_MANIFEST.json)\n", encoding="utf-8")
    return directory


def channel_error_totals(prediction, target, observed):
    p, y, m = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(observed)
    if p.ndim != 3 or p.shape != y.shape or p.shape != m.shape or m.dtype != np.bool_:
        raise ValueError("Matching prediction/target/bool mask [N,H,C] required")
    if not np.isfinite(p[m]).all() or not np.isfinite(y[m]).all():
        raise ValueError("Nonfinite observed prediction/target")
    difference = np.where(m, p - y, 0)
    return dict(sse=(difference * difference).sum((0, 1), dtype=np.float64),
                sae=np.abs(difference).sum((0, 1), dtype=np.float64), count=m.sum((0, 1), dtype=np.int64))


def aggregate_channel_group(totals, indices, cells_per_channel):
    indices = np.asarray(indices, np.int64)
    active = indices[totals["count"][indices] > 0]
    observed = int(totals["count"][indices].sum())
    possible = int(len(indices) * cells_per_channel)
    return dict(mse=math.fsum(float(totals["sse"][i]) / int(totals["count"][i]) for i in active) / len(active) if len(active) else None,
                mae=math.fsum(float(totals["sae"][i]) / int(totals["count"][i]) for i in active) / len(active) if len(active) else None,
                total_group_channels=len(indices), active_channels=len(active), zero_observed_channels=len(indices) - len(active),
                observed_target_cells=observed, possible_target_cells=possible, observed_fraction=observed / possible if possible else None)


def _inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or path == root or not path.is_relative_to(root):
        raise ValueError("Relative cache/stat path escapes declared root")
    return path


def _t_allowed(lock, manifest):
    run_id, meta = manifest["run_id"], manifest.get("metadata", {})
    if meta.get("route_spec") is not None and lock.get("allowed_route_sha", {}).get(run_id) == _fingerprint(meta["route_spec"]):
        return True
    sha = meta.get("artifact_sha256", meta.get("realized_model_binding", {}).get("selected_artifact_sha256"))
    return sha is not None and run_id in lock.get("allowed_run_ids", []) and lock.get("allowed_artifact_sha", {}).get(run_id) == sha


def _budget(public, run_id, manifest):
    meta = manifest.get("metadata", {})
    parent = meta.get("parent_run_id", meta.get("family_run_id", meta.get("route_spec", {}).get("source_parent_run_id", run_id)))
    for candidate in dict.fromkeys((run_id, parent)):
        for name in ("BASE_FIT_RECEIPT.json", "CONTROL_FIT_RECEIPT.json"):
            path = public / "runs" / candidate / name
            if path.is_file():
                receipt = _json(path)
                return "A:" + "|".join(map(str, receipt.get("epochs", [])))
        path = public / "combinations" / candidate / "receipt.json"
        if path.is_file():
            receipt = _json(path)
            return "Mcal:" + "|".join(map(str, receipt.get("registered_epoch_endpoints", [])))
    return "non_iterative_or_calibration_without_epoch_receipt"


def training_identity(public, run_id, manifest, family):
    public = Path(public)
    meta = manifest.get("metadata", {})
    parent = meta.get("parent_run_id", meta.get("family_run_id", meta.get("route_spec", {}).get("source_parent_run_id", run_id)))
    fixed = {"CURRENT", "SEASONAL_NAIVE", "RAW_RIDGE", "RAW_BOLT_SMALL", "RAW_BOLT_MINI", "RAW_BOLT_TINY",
             "PCA_FIXED_BOLT", "PCA_FIXED", "NESTED_PCA", "RAW_CALIBRATED", "ALPHA0", "ALPHA05", "ALPHA1", "ALPHA_FIT"}
    result = dict(optimizer_trained=family not in fixed, learning_rate=None, lr_known=family in fixed,
                  fit_config_sha="", fit_source_file="", fit_epoch_endpoints="", selected_epoch=None, selected_updates=None, input_files=[])
    for candidate in dict.fromkeys((run_id, parent)):
        source_path = public / "runs" / candidate / "SOURCE_CONTRACT.json"
        if not source_path.is_file():
            source_path = source_path.with_name("CONTROL_SOURCE_CONTRACT.json")
        if source_path.is_file():
            source = _json(source_path); spec = source.get("spec", {})
            if spec.get("run_id") != candidate:
                raise ValueError("Learning recipe belongs to a different run")
            lr = spec.get("lr")
            result.update(fit_config_sha=source.get("config_sha", ""), fit_source_file=source_path.relative_to(public).as_posix())
            result["fit_epoch_endpoints"] = "|".join(map(str, spec.get("epochs", [])))
            result["input_files"].append(dict(scope="public", file=result["fit_source_file"], sha256=data.sha256_file(source_path)))
            for receipt_name in ("BASE_FIT_RECEIPT.json", "CONTROL_FIT_RECEIPT.json"):
                own_receipt = source_path.with_name(receipt_name)
                if own_receipt.is_file():
                    result["selected_epoch"] = _json(own_receipt).get("selected_epoch")
                    result["input_files"].append(dict(scope="public", file=own_receipt.relative_to(public).as_posix(), sha256=data.sha256_file(own_receipt)))
                    break
        else:
            receipt_path = public / "combinations" / candidate / "receipt.json"
            if not receipt_path.is_file():
                continue
            receipt = _json(receipt_path)
            if receipt.get("run_id") != candidate:
                raise ValueError("Calibration recipe belongs to a different run")
            lr = receipt.get("learning_rate")
            result.update(fit_config_sha=receipt.get("contract", {}).get("config_sha", ""), fit_source_file=receipt_path.relative_to(public).as_posix())
            result["fit_epoch_endpoints"] = "|".join(map(str, receipt.get("registered_epoch_endpoints", [])))
            result["selected_updates"] = receipt.get("selected_updates")
            result["input_files"].append(dict(scope="public", file=result["fit_source_file"], sha256=data.sha256_file(receipt_path)))
        try:
            numeric_lr = float(lr) if lr is not None and not isinstance(lr, bool) else None
        except (TypeError, ValueError):
            numeric_lr = None
        if numeric_lr is not None and math.isfinite(numeric_lr) and numeric_lr > 0:
            result["learning_rate"] = numeric_lr
            result["lr_known"] = bool(result["fit_config_sha"])
        return result
    return result


def learning_pair_compatible(left, right):
    for row in (left, right):
        if row.get("optimizer_trained", False) and (not row.get("lr_known", False) or not row.get("fit_config_sha")):
            return False
    if left.get("optimizer_trained", False) and right.get("optimizer_trained", False):
        return left.get("learning_rate") == right.get("learning_rate")
    return True


def evaluate_groups(public_root, cache_root, pattern_snapshot, split="V_select"):
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    public, cache, snapshot = Path(public_root).resolve(), Path(cache_root).resolve(), Path(pattern_snapshot).resolve()
    if not snapshot.is_relative_to(public / "patterns") or split not in {"V_select", "T_final"}:
        raise ValueError("Pattern snapshot must be inside public patterns; supported split is V_select/T_final")
    lock = {}
    if split == "T_final":
        path = public / "SELECTION_LOCK.json"
        if not path.is_file() or _json(path).get("locked") is not True:
            raise ValueError("T evaluation requires locked SELECTION_LOCK.json")
        lock = _json(path)
    pattern_manifest = _json(snapshot / "PATTERN_MANIFEST.json")
    table = snapshot / pattern_manifest["channel_table_file"]
    if data.sha256_file(table) != pattern_manifest["channel_table_sha256"]:
        raise ValueError("Frozen A-only channel pattern table changed")
    with table.open(encoding="utf-8", newline="") as stream:
        pattern_rows = list(csv.DictReader(stream))
    raw_registry = (public / "EXPERIMENT_REGISTRY.csv").read_bytes()
    registry_rows = list(csv.DictReader(io.StringIO(raw_registry.decode("utf-8-sig"))))
    registry = {row["experiment_id"]: row for row in registry_rows}
    if len(registry) != len(registry_rows):
        raise ValueError("Duplicate registered experiment IDs")
    directory = snapshot / _snapshot_name("evaluation_"); directory.mkdir(exist_ok=False)
    group_rows, channel_rows, records, excluded, inputs = [], [], [], [], []
    for path in sorted(public.rglob("PREDICTION_MANIFEST.json")):
        try:
            manifest = _json(path)
            if manifest.get("split") != split or manifest.get("score") is None:
                continue
            run_id = manifest["run_id"]
            registered = registry[run_id]
            dataset = registered["dataset"]
            if dataset not in pattern_manifest["datasets"]:
                continue
            contract = pattern_manifest["datasets"][dataset]
            if registered.get("fold", "primary") != contract["fold"]:
                continue
            if split == "T_final" and not _t_allowed(lock, manifest):
                raise ValueError("T run/artifact/route absent from fixed SELECTION_LOCK")
            if manifest["data_sha"] != contract["data_sha"]:
                raise ValueError("Prediction data differs from frozen pattern data")
            reference_path = path.with_name("REFERENCE_CONTRACT.json")
            reference = _json(reference_path)
            if any(manifest.get(key) != reference.get(key) or not str(manifest.get(key, "")) for key in audit.IDENTITY_KEYS):
                raise ValueError("Independent prediction reference identity changed")
            arrays = audit._load_arrays(manifest, reference, cache)
            channels = list(map(str, arrays["channel_ids"]))
            if channels != contract["columns"]:
                raise ValueError("Prediction channel order differs from pattern A-fit channels")
            totals = channel_error_totals(arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"])
            cells = int(np.prod(arrays["observed"].shape[:2]))
            overall = aggregate_channel_group(totals, range(len(channels)), cells)
            for metric in ("mse", "mae"):
                expected = manifest["score"][metric]
                if overall[metric] is None or not math.isclose(overall[metric], float(expected), rel_tol=1e-7, abs_tol=1e-9):
                    raise ValueError("Own independent FP64 overall metric differs: " + metric)
            selected_patterns = sorted([r for r in pattern_rows if r["dataset"] == dataset], key=lambda r: int(r["channel_index"]))
            if [r["channel"] for r in selected_patterns] != channels:
                raise ValueError("Frozen pattern table channel identity differs")
            common = dict(run_id=run_id, dataset=dataset, fold=contract["fold"], split=split, cohort=manifest["cohort"],
                family=registered["family"], seed=int(registered.get("seed") or 0), k=int(registered.get("latent_k") or 0),
                budget=_budget(public, run_id, manifest), data_sha=manifest["data_sha"], scaler_sha=manifest["scaler_sha"],
                metric_scale=manifest["score"].get("metric_scale", "A_normalized_original_channels"))
            recipe = training_identity(public, run_id, manifest, registered["family"])
            common.update({key: value for key, value in recipe.items() if key != "input_files"})
            inputs.extend(recipe["input_files"])
            for index, channel in enumerate(channels):
                count = int(totals["count"][index])
                channel_rows.append(dict(**common, channel=channel, sse=float(totals["sse"][index]), sae=float(totals["sae"][index]),
                    count=count, mse=float(totals["sse"][index]) / count if count else None,
                    mae=float(totals["sae"][index]) / count if count else None,
                    **{p + "_group": selected_patterns[index][p + "_group"] for p in PATTERNS}))
            groups = {}
            for pattern in PATTERNS:
                for group in ("low", "high", "NA"):
                    indices = [i for i, row in enumerate(selected_patterns) if row[pattern + "_group"] == group]
                    score = aggregate_channel_group(totals, indices, cells)
                    group_rows.append(dict(**common, pattern=pattern, group=group, **score))
                    groups[(pattern, group)] = indices
            anchors = manifest.get("metadata", {}).get("actual_parent_manifests", {})
            records.append(dict(common=common, totals=totals, groups=groups,
                cohort_sha=_fingerprint({key: manifest["arrays"][key] for key in sorted(audit.REFERENCE_KEYS)}),
                parent_signature=_fingerprint(anchors) if anchors else "", anchors=anchors))
            for item in (path, reference_path):
                inputs.append(dict(scope="public", file=item.relative_to(public).as_posix(), sha256=data.sha256_file(item)))
            inputs.append(dict(scope="private", file=manifest["cache_relative_path"], sha256=manifest["archive_sha256"]))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            reason = str(exc).replace(str(public), "<public root>").replace(str(cache), "<private cache>")
            excluded.append(dict(manifest=path.relative_to(public).as_posix(), reason=reason))
    paired = []
    same_budget_families = {("B2", "B0"), ("B4", "B2"), ("LAG_CAUSAL", "B0"), ("LAG_BIDIR", "LAG_CAUSAL")}
    for left_family, right_family in COMPARISONS:
        for left in [r for r in records if r["common"]["family"] == left_family]:
            for right in [r for r in records if r["common"]["family"] == right_family]:
                a, b = left["common"], right["common"]
                if any(a[key] != b[key] for key in ("dataset", "fold", "cohort", "data_sha", "scaler_sha")) or left["cohort_sha"] != right["cohort_sha"]:
                    continue
                if not learning_pair_compatible(a, b):
                    continue
                if (left_family, right_family) in same_budget_families and any(a[key] != b[key] for key in ("budget", "seed", "k")):
                    continue
                if left_family == "CONDITIONAL":
                    if right_family in {"PAST_CONTROL", "ALPHA_FIT"} and (not left["parent_signature"] or left["parent_signature"] != right["parent_signature"]):
                        continue
                    if right_family == "PAST_CONTROL" and any(a[key] != b[key] for key in ("budget", "seed")):
                        continue
                    if right_family == "DIRECT_G" and left["anchors"].get("g", {}).get("run_id") != b["run_id"]:
                        continue
                if left_family == "RAW_CALIBRATED" and left["anchors"].get("b", {}).get("run_id") != b["run_id"]:
                    continue
                if not np.array_equal(left["totals"]["count"], right["totals"]["count"]):
                    raise ValueError("Paired methods have different observed channel counts")
                for (pattern, group), indices in left["groups"].items():
                    active = [i for i in indices if left["totals"]["count"][i] > 0]
                    delta = {}
                    for field, numerator in (("delta_mse", "sse"), ("delta_mae", "sae")):
                        delta[field] = math.fsum((float(left["totals"][numerator][i]) - float(right["totals"][numerator][i])) /
                            int(left["totals"]["count"][i]) for i in active) / len(active) if active else None
                    paired.append(dict(dataset=a["dataset"], fold=a["fold"], split=split, cohort=a["cohort"],
                        comparison=left_family + " - " + right_family, left_run_id=a["run_id"], right_run_id=b["run_id"],
                        left_budget=a["budget"], right_budget=b["budget"], left_seed=a["seed"], right_seed=b["seed"], left_k=a["k"], right_k=b["k"],
                        left_lr=a["learning_rate"], right_lr=b["learning_rate"], left_fit_config_sha=a["fit_config_sha"], right_fit_config_sha=b["fit_config_sha"],
                        pattern=pattern, group=group, total_group_channels=len(indices), active_channels=len(active),
                        observed_target_cells=int(left["totals"]["count"][indices].sum()), **delta,
                        comparison_scope="descriptive matched targets/masks; information/training budgets in separate fields; no causal/fairness claim"))
    if split == "T_final" and not records:
        raise ValueError("No allowed, auditable locked T_final prediction manifests exist")
    _write_csv(directory / "CHANNEL_ERRORS.csv", channel_rows)
    _write_csv(directory / "GROUP_ERRORS.csv", group_rows)
    _write_csv(directory / "PAIRED_GROUP_DIFFERENCES.csv", paired)
    outputs = {p.name: data.sha256_file(p) for p in directory.glob("*.csv")}
    _write_json(directory / "EVALUATION_MANIFEST.json", dict(created_unix=time.time(), split=split,
        pattern_manifest_sha256=data.sha256_file(snapshot / "PATTERN_MANIFEST.json"),
        pattern_table_sha256=data.sha256_file(table), source_sha256=data.sha256_file(Path(__file__)),
        audit_loader_source_sha256=data.sha256_file(Path(__file__).with_name("audit_predictions.py")),
        registry_sha256=hashlib.sha256(raw_registry).hexdigest(), prediction_count=len(records), group_rows=len(group_rows), paired_rows=len(paired),
        input_files=inputs, outputs=outputs, excluded_prediction_inputs=excluded, pattern_fit_used_future_targets=False,
        wall_seconds=time.perf_counter() - started_wall, CPU_process_seconds=time.process_time() - started_cpu,
        scoring="Independent own FP64 per-channel SSE/count and SAE/count, then equal active-channel mean within frozen group",
        empty_high_low_NA_groups_preserved=True, test_method_or_case_selection=False, T_selection_lock_checked=split == "T_final",
        full_registry_audit_performed=False, independent_generalization_claimed=False, inference_reexecuted=False, GPU_execution=False))
    (directory / "README_KO.md").write_text("# 고정 A 패턴별 예측 오차\n\n각 채널의 관측 오차를 먼저 계산한 뒤 그룹 내 관측된 채널을 균등 평균했다. low/high/NA와 빈 그룹의 coverage를 모두 남겼다. 인과성·배포 selector·독립 일반화 검증이 아니다.\n\n" +
        "\n".join(f"- [{name}]({name})" for name in list(outputs) + ["EVALUATION_MANIFEST.json"]) + "\n", encoding="utf-8")
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", required=True); parser.add_argument("--cache-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["beijing", "jena", "electricity", "pems08"])
    parser.add_argument("--fold", default="primary", choices=("primary", "secondary"))
    parser.add_argument("--pattern-snapshot"); parser.add_argument("--split", default="V_select", choices=("V_select", "T_final"))
    args = parser.parse_args()
    snapshot = Path(args.pattern_snapshot) if args.pattern_snapshot else fit_patterns(args.public_root, args.cache_root, args.datasets, args.fold)
    print(str(evaluate_groups(args.public_root, args.cache_root, snapshot, args.split)), flush=True)


if __name__ == "__main__":
    main()
