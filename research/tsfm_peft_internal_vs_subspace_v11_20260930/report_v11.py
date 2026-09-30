from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runtime_v11 import HERE, Job, digest, read_json, save_json


DATASETS = ("robin", "jena", "hog")
FAMILIES = (
    "a_p_lora_qfixed",
    "full_lora_mse",
    "b_fixed_u",
    "b_learned_u",
    "b_fixed_gamma",
    "b_learned_gamma",
    "base_level",
    "direct_nlinear",
    "f0",
    "lora_native",
)
REFERENCE_FAMILIES = ("base_level", "f0", "lora_native", "full_lora_mse", "direct_nlinear")
PERIODS = ("combined", "test_a", "test_b")
DATASET_DIMS = {"robin": (17, 5), "jena": (21, 6), "hog": (32, 8)}
UNCOMPRESSED_FAMILIES = {"f0", "full_lora_mse", "lora_native"}
SEEDS = (92601, 92602)
LEVEL_G_PARAMETERS = 512 * 32 + 32 * 48
DIRECT_NLINEAR_PARAMETERS = 512 * 48 + 48
LORA_PARAMETERS = 294_912

DISPLAY = {
    "a_p_lora_qfixed": "A P-LoRA + fixed Q",
    "full_lora_mse": "Full LoRA MSE",
    "b_fixed_u": "B fixed U",
    "b_learned_u": "B learned U",
    "b_fixed_gamma": "B fixed U + gamma",
    "b_learned_gamma": "B learned U + gamma",
    "base_level": "Base LEVEL",
    "direct_nlinear": "Direct NLinear",
    "f0": "F0",
    "lora_native": "Native LoRA",
}

COLORS = {
    "a_p_lora_qfixed": "#0072B2",
    "full_lora_mse": "#56B4E9",
    "b_fixed_u": "#009E73",
    "b_learned_u": "#F0E442",
    "b_fixed_gamma": "#E69F00",
    "b_learned_gamma": "#D55E00",
    "base_level": "#CC79A7",
    "direct_nlinear": "#000000",
    "f0": "#999999",
    "lora_native": "#8C6BB1",
}

MARKERS = {
    "full": "o",
    "chunk_4k": "^",
    "chunk_k": "s",
    "direct": "X",
}

ADAPTATION_COST_FIELDS = (
    "dataset",
    "seed",
    "candidate",
    "component",
    "cost_boundary",
    "current_round_increment",
    "selected",
    "fit_attempt_count",
    "run_id",
    "run_ids",
    "source_family",
    "lr",
    "elapsed_s",
    "elapsed_s_provenance",
    "training_peak_allocated_bytes",
    "training_peak_reserved_bytes",
    "training_peak_provenance",
    "trainable_parameters_recorded",
    "trainable_parameters_provenance",
    "component_source_parameter_count",
    "candidate_system_parameter_count",
    "source_parameter_note",
    "cost_model_info_source_key",
    "cost_current_neural_stage_registered_parameters",
    "cost_reused_parent_fitted_parameters",
    "cost_gamma_coefficients",
    "cost_sum_stage_registered_fit_parameters",
    "cost_parameter_bytes",
    "cost_buffer_bytes",
    "cost_deployment_tensor_bytes",
    "receipt_path",
    "receipt_sha256",
    "status",
    "note",
)


def receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        relative = path.relative_to(HERE)
        name = relative.as_posix()
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": digest(path), "bytes": path.stat().st_size}


def absolute_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report artifact: {path}")


def require_completed_input(path: Path, expected_status: str = "complete") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required completed report input is absent: {path}")
    value = read_json(path)
    status = value.get("status")
    if status != expected_status:
        raise RuntimeError(f"{path.name} status is {status!r}; expected {expected_status!r}")
    return value


def verify_json_receipt(item: dict[str, Any] | None, label: str, required: bool = True) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    if not item or not item.get("path"):
        if required:
            raise FileNotFoundError(f"Missing result receipt for {label}")
        return None, None, "unknown_missing_receipt"
    path = Path(item["path"])
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing result file for {label}: {path}")
        return None, {"path": str(path), "sha256": item.get("sha256"), "bytes": item.get("bytes")}, "unknown_missing_file"
    actual_sha = digest(path)
    expected_sha = item.get("sha256")
    if expected_sha and actual_sha != expected_sha:
        if required:
            raise RuntimeError(f"Hash mismatch for {label}: {path}")
        return None, {"path": str(path), "sha256": actual_sha, "expected_sha256": expected_sha, "bytes": path.stat().st_size}, "unknown_hash_mismatch"
    value = read_json(path)
    if value.get("status") not in (None, "complete"):
        if required:
            raise RuntimeError(f"{label} result is not complete: {path}")
        return value, absolute_receipt(path), "unknown_incomplete_result"
    return value, absolute_receipt(path), "confirmed"


def finite_number(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing numeric value: {label}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite numeric value: {label}={value!r}")
    return number


def relative_gap(candidate: float, reference: float) -> float | None:
    if reference == 0 or not math.isfinite(reference):
        return None
    return 100.0 * (candidate - reference) / reference


def ids_and_seeds(dataset_payload: dict[str, Any], family: str) -> tuple[list[str], list[int | str]]:
    role = dataset_payload["role_summary"][family]
    ids = list(role.get("ids", []))
    models = dataset_payload.get("models", {})
    seeds: list[int | str] = []
    for model_id in ids:
        model = models.get(model_id, {})
        seed = model.get("seed", "na")
        if seed not in seeds:
            seeds.append(seed)
    return ids, seeds


def selected_seed_metric_range(
    dataset_payload: dict[str, Any],
    family: str,
    period: str,
    metric: str,
) -> dict[str, Any]:
    ids = list(dataset_payload["role_summary"][family].get("ids", []))
    models = dataset_payload.get("models", {})
    values: list[float] = []
    seed_values: list[dict[str, Any]] = []
    for model_id in ids:
        if model_id not in models:
            raise ValueError(f"Missing model entry for selected id: {model_id}")
        model = models[model_id]
        value = finite_number(
            model["periods"][period][metric],
            f"{model_id}/{period}/{metric}",
        )
        values.append(value)
        seed_values.append({"id": model_id, "seed": model.get("seed", "na"), metric: value})
    if not values:
        raise ValueError(f"No selected model values for {family}/{period}/{metric}")
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "values": seed_values,
        "range_is_ci": False,
        "note": "Selected-seed min/max range; not a confidence interval. Single-id families are one point.",
        "source_key_template": f"datasets.<dataset>.models.<id>.periods.{period}.{metric}",
    }


def validate_evaluation(evaluation: dict[str, Any]) -> None:
    actual_datasets = set(evaluation.get("datasets", {}))
    expected_datasets = set(DATASETS)
    if actual_datasets != expected_datasets:
        raise ValueError(f"Unexpected evaluation datasets: {sorted(actual_datasets)}")
    for dataset in DATASETS:
        payload = evaluation["datasets"][dataset]
        roles = set(payload.get("role_summary", {}))
        missing = set(FAMILIES) - roles
        if missing:
            raise ValueError(f"{dataset} evaluation is missing role summaries: {sorted(missing)}")
        for family in FAMILIES:
            periods = set(payload["role_summary"][family].get("periods", {}))
            missing_periods = set(PERIODS) - periods
            if missing_periods:
                raise ValueError(f"{dataset}/{family} is missing periods: {sorted(missing_periods)}")


def build_family_rows(evaluation: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    for dataset in DATASETS:
        payload = evaluation["datasets"][dataset]
        nested[dataset] = {}
        refs_by_period: dict[str, dict[str, dict[str, float]]] = {}
        for period in PERIODS:
            refs_by_period[period] = {}
            for reference in REFERENCE_FAMILIES:
                period_metrics = payload["role_summary"][reference]["periods"][period]
                refs_by_period[period][reference] = {
                    "mse": finite_number(period_metrics["mse"], f"{dataset}/{reference}/{period}/mse"),
                    "mae": finite_number(period_metrics["mae"], f"{dataset}/{reference}/{period}/mae"),
                }

        for family in FAMILIES:
            ids, seeds = ids_and_seeds(payload, family)
            nested[dataset][family] = {
                "display": DISPLAY[family],
                "ids": ids,
                "seeds": seeds,
                "periods": {},
            }
            for period in PERIODS:
                metrics = payload["role_summary"][family]["periods"][period]
                mse = finite_number(metrics["mse"], f"{dataset}/{family}/{period}/mse")
                mae = finite_number(metrics["mae"], f"{dataset}/{family}/{period}/mae")
                signed = metrics.get("signed_mean_error")
                signed_value = None if signed is None else finite_number(signed, f"{dataset}/{family}/{period}/signed")
                period_entry: dict[str, Any] = {
                    "mse": mse,
                    "mae": mae,
                    "signed_mean_error": signed_value,
                    "ids": ids,
                    "seeds": seeds,
                    "selected_seed_range": {
                        "mse": selected_seed_metric_range(payload, family, period, "mse"),
                        "mae": selected_seed_metric_range(payload, family, period, "mae"),
                    },
                    "relative_gap_pct": {},
                    "source_key": f"datasets.{dataset}.role_summary.{family}.periods.{period}",
                }
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "family": family,
                    "display": DISPLAY[family],
                    "period": period,
                    "ids": ";".join(ids),
                    "seeds": ";".join(str(seed) for seed in seeds),
                    "mse": mse,
                    "mae": mae,
                    "signed_mean_error": signed_value,
                    "selected_seed_count": period_entry["selected_seed_range"]["mse"]["count"],
                    "selected_seed_mse_min": period_entry["selected_seed_range"]["mse"]["min"],
                    "selected_seed_mse_max": period_entry["selected_seed_range"]["mse"]["max"],
                    "selected_seed_mae_min": period_entry["selected_seed_range"]["mae"]["min"],
                    "selected_seed_mae_max": period_entry["selected_seed_range"]["mae"]["max"],
                    "source_key": period_entry["source_key"],
                }
                for reference in REFERENCE_FAMILIES:
                    mse_gap = relative_gap(mse, refs_by_period[period][reference]["mse"])
                    mae_gap = relative_gap(mae, refs_by_period[period][reference]["mae"])
                    row[f"relative_mse_vs_{reference}_pct"] = mse_gap
                    row[f"relative_mae_vs_{reference}_pct"] = mae_gap
                    period_entry["relative_gap_pct"][reference] = {"mse": mse_gap, "mae": mae_gap}
                rows.append(row)
                nested[dataset][family]["periods"][period] = period_entry
    return rows, nested


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    require_new(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_map_for_combined(family_json: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        result[dataset] = {}
        for family in FAMILIES:
            result[dataset][family] = finite_number(
                family_json[dataset][family]["periods"]["combined"]["mse"],
                f"{dataset}/{family}/combined/mse",
            )
    return result


def collect_batch4_cost_options(cost: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gpu_groups = cost.get("gpu", {}).get("groups", [])
    gpu_instances = cost.get("gpu", {}).get("instances", [])
    cpu_groups = cost.get("cpu", {}).get("groups", [])
    if not isinstance(gpu_groups, list) or not isinstance(gpu_instances, list) or not isinstance(cpu_groups, list):
        raise ValueError("cost_summary.json does not contain expected group arrays")

    gpu_options: list[dict[str, Any]] = []
    for group in gpu_groups:
        if int(group.get("batch", -1)) != 4:
            continue
        family = group.get("family")
        dataset = group.get("dataset")
        if family not in FAMILIES or dataset not in DATASETS:
            continue
        option = {
            "dataset": dataset,
            "family": family,
            "display": DISPLAY[family],
            "execution": group.get("execution"),
            "chunk_rows": group.get("chunk_rows"),
            "device": group.get("device", "cuda"),
            "ids": list(group.get("ids", [])),
            "mean_seed_median_ms": finite_number(
                group.get("mean_seed_median_ms"),
                f"cost/gpu/{dataset}/{family}/{group.get('execution')}/batch4/ms",
            ),
            "mean_seed_median_origins_s": finite_number(
                group.get("mean_seed_median_origins_s"),
                f"cost/gpu/{dataset}/{family}/{group.get('execution')}/batch4/origins_s",
            ),
            "mean_seed_median_peak_allocated_bytes": finite_number(
                group.get("mean_seed_median_peak_allocated_bytes"),
                f"cost/gpu/{dataset}/{family}/{group.get('execution')}/batch4/allocated",
            ),
            "mean_seed_median_peak_reserved_bytes": finite_number(
                group.get("mean_seed_median_peak_reserved_bytes"),
                f"cost/gpu/{dataset}/{family}/{group.get('execution')}/batch4/reserved",
            ),
            "source": "cost_summary.json:gpu.groups",
        }
        option["allocated_mib"] = option["mean_seed_median_peak_allocated_bytes"] / (1024.0 * 1024.0)
        option["reserved_mib"] = option["mean_seed_median_peak_reserved_bytes"] / (1024.0 * 1024.0)
        option.update(cost_range_from_instances(group, gpu_instances))
        gpu_options.append(option)

    expected = expected_batch4_gpu_options()
    observed = {(row["dataset"], row["family"], row["execution"], row["chunk_rows"]) for row in gpu_options}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Missing expected batch4 GPU cost options: {missing[:12]}")

    cpu_options: list[dict[str, Any]] = []
    for group in cpu_groups:
        if int(group.get("batch", -1)) != 4:
            continue
        family = group.get("family")
        dataset = group.get("dataset")
        if family not in FAMILIES or dataset not in DATASETS:
            continue
        cpu_options.append(
            {
                "dataset": dataset,
                "family": family,
                "display": DISPLAY[family],
                "execution": group.get("execution"),
                "chunk_rows": group.get("chunk_rows"),
                "device": group.get("device", "cpu"),
                "ids": list(group.get("ids", [])),
                "mean_seed_median_ms": finite_number(
                    group.get("mean_seed_median_ms"),
                    f"cost/cpu/{dataset}/{family}/{group.get('execution')}/batch4/ms",
                ),
                "mean_seed_median_origins_s": finite_number(
                    group.get("mean_seed_median_origins_s"),
                    f"cost/cpu/{dataset}/{family}/{group.get('execution')}/batch4/origins_s",
                ),
                "source": "cost_summary.json:cpu.groups",
            }
        )
    return gpu_options, cpu_options


def cost_range_from_instances(group: dict[str, Any], instances: list[dict[str, Any]]) -> dict[str, Any]:
    key = (group.get("dataset"), group.get("family"), group.get("batch"), group.get("execution"), group.get("chunk_rows"))
    matched = [
        item
        for item in instances
        if (item.get("dataset"), item.get("family"), item.get("batch"), item.get("execution"), item.get("chunk_rows")) == key
    ]
    expected = 1 if group.get("family") == "f0" else 2
    if len(matched) != expected:
        raise ValueError(f"Cannot derive cost ranges for {key}; expected {expected} instance(s), found {len(matched)}")
    block_ms: list[float] = []
    allocated: list[float] = []
    reserved: list[float] = []
    for item in matched:
        block_ms.extend(
            finite_number(value, f"cost/instance/{key}/block_ms")
            for value in item.get("block_ms", [])
        )
        allocated.extend(
            finite_number(value, f"cost/instance/{key}/peak_allocated")
            for value in item.get("peak_allocated_bytes", [])
            if value is not None
        )
        reserved.extend(
            finite_number(value, f"cost/instance/{key}/peak_reserved")
            for value in item.get("peak_reserved_bytes", [])
            if value is not None
        )
    if len(block_ms) != expected * 3:
        raise ValueError(f"Cost range needs three block medians per selected instance for {key}")
    throughput = [1000.0 / value for value in block_ms]
    result: dict[str, Any] = {
        "range_note": "Min/max over selected instance x three block medians; not a confidence interval.",
        "block_median_ms_range": [min(block_ms), max(block_ms)],
        "origins_s_range_from_block_ms": [min(throughput), max(throughput)],
    }
    if allocated:
        result["peak_allocated_range_bytes"] = [min(allocated), max(allocated)]
        result["allocated_range_mib"] = [min(allocated) / (1024.0 * 1024.0), max(allocated) / (1024.0 * 1024.0)]
    else:
        result["peak_allocated_range_bytes"] = None
        result["allocated_range_mib"] = None
    if reserved:
        result["peak_reserved_range_bytes"] = [min(reserved), max(reserved)]
        result["reserved_range_mib"] = [min(reserved) / (1024.0 * 1024.0), max(reserved) / (1024.0 * 1024.0)]
    else:
        result["peak_reserved_range_bytes"] = None
        result["reserved_range_mib"] = None
    return result


def expected_batch4_gpu_options() -> set[tuple[str, str, str, int | None]]:
    expected: set[tuple[str, str, str, int | None]] = set()
    for dataset in DATASETS:
        channels, latent = DATASET_DIMS[dataset]
        for family in FAMILIES:
            if family == "direct_nlinear":
                expected.add((dataset, family, "direct", None))
            elif family in UNCOMPRESSED_FAMILIES:
                expected.add((dataset, family, "full", 4 * channels))
                expected.add((dataset, family, "chunk_4k", 4 * latent))
                expected.add((dataset, family, "chunk_k", latent))
            else:
                expected.add((dataset, family, "full", 4 * latent))
                expected.add((dataset, family, "chunk_k", latent))
    return expected


def collect_paired_interval_mappings(evaluation: dict[str, Any]) -> dict[str, Any]:
    mappings: dict[str, Any] = {}
    for dataset in DATASETS:
        comparisons = evaluation["datasets"][dataset].get("comparisons", {})
        copied: dict[str, Any] = {}
        for name, value in comparisons.items():
            if not isinstance(value, dict):
                continue
            periods = value.get("periods", {})
            copied[name] = {
                "candidate": value.get("candidate"),
                "reference": value.get("reference"),
                "direction": value.get("direction"),
                "source_key": f"datasets.{dataset}.comparisons.{name}",
                "periods": {},
            }
            for period, metric_map in periods.items():
                if not isinstance(metric_map, dict):
                    continue
                copied[name]["periods"][period] = {}
                for metric in ("mse", "mae"):
                    metric_value = metric_map.get(metric)
                    if not isinstance(metric_value, dict):
                        continue
                    metric_entry: dict[str, Any] = {
                        "source_key": f"datasets.{dataset}.comparisons.{name}.periods.{period}.{metric}",
                    }
                    for field in (
                        "candidate",
                        "reference",
                        "absolute_difference",
                        "relative_change_percent",
                        "conditional_block95_absolute",
                        "conditional_block95_relative_percent",
                    ):
                        raw = metric_value.get(field)
                        if isinstance(raw, list):
                            metric_entry[field] = [finite_number(item, f"{dataset}/{name}/{period}/{metric}/{field}") for item in raw]
                        elif raw is None:
                            metric_entry[field] = None
                        else:
                            metric_entry[field] = finite_number(raw, f"{dataset}/{name}/{period}/{metric}/{field}")
                    copied[name]["periods"][period][metric] = metric_entry
        mappings[dataset] = {
            "source": f"evaluation01.json:datasets.{dataset}.comparisons",
            "bootstrap": evaluation["datasets"][dataset].get("bootstrap"),
            "comparison_keys": sorted(copied),
            "comparisons": copied,
        }
    return mappings


def candidate_system_parameters(candidate: str, dataset: str) -> tuple[int | None, str]:
    channels, latent = DATASET_DIMS[dataset]
    u_raw = channels * latent
    if candidate == "a_p_lora_qfixed":
        return LEVEL_G_PARAMETERS + LORA_PARAMETERS, "A system source count: reused LEVEL G 17,920 + current LoRA 294,912"
    if candidate == "full_lora_mse":
        return LORA_PARAMETERS, "Full MSE LoRA count: LoRA 294,912"
    if candidate == "direct_nlinear":
        return DIRECT_NLINEAR_PARAMETERS, "Direct NLinear count: shared Linear(512,48,bias=True)=24,624"
    if candidate == "lora_native":
        return LORA_PARAMETERS, "Native LoRA count: historical LoRA 294,912 when the result receipt is confirmed"
    if candidate == "b_fixed_u":
        return DIRECT_NLINEAR_PARAMETERS, "B fixed-U system count: selected S* 24,624; no U optimizer fit; Gamma not included"
    if candidate == "b_learned_u":
        return DIRECT_NLINEAR_PARAMETERS + u_raw, f"B learned-U system count: selected S* 24,624 + U raw C*K={u_raw}; Gamma not included"
    if candidate == "b_fixed_gamma":
        return DIRECT_NLINEAR_PARAMETERS + latent, f"B fixed-U+Gamma count: selected S* 24,624 + Gamma K={latent}; no U optimizer fit"
    if candidate == "b_learned_gamma":
        return DIRECT_NLINEAR_PARAMETERS + u_raw + latent, f"B learned-U+Gamma count: selected S* 24,624 + U raw C*K={u_raw} + Gamma K={latent}"
    if candidate == "base_level":
        return LEVEL_G_PARAMETERS, "Base LEVEL source count: reused fitted G 17,920"
    return None, "Parameter count not defined for this report role"


def component_parameter_count(candidate: str, component: str, dataset: str) -> int | None:
    channels, latent = DATASET_DIMS[dataset]
    if component in ("reused_level_parent", "base_level_reference"):
        return LEVEL_G_PARAMETERS
    if component in ("selected_a_fit", "a_2lr_search", "selected_full_lora_mse_fit", "full_lora_mse_2lr_search", "native_lora_historical_fit"):
        return LORA_PARAMETERS
    if component in ("selected_direct_fit", "direct_2lr_search"):
        return DIRECT_NLINEAR_PARAMETERS
    if component in ("selected_u_fit", "u_2lr_search"):
        return channels * latent
    if component in ("fixed_u_no_optimizer_fit",):
        return 0
    if component in ("gamma_fit",):
        return latent
    return None


def cost_model_info_index(cost: dict[str, Any]) -> dict[tuple[str, str, Any], dict[str, Any]]:
    result: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for item in cost.get("gpu", {}).get("instances", []):
        if int(item.get("batch", -1)) != 4 or item.get("execution") != "full":
            continue
        key = (item.get("dataset"), item.get("family"), item.get("seed"))
        result.setdefault(key, item.get("model", {}))
    return result


def training_peak_values(result: dict[str, Any] | None) -> tuple[Any, Any]:
    if not result:
        return None, None
    peak = result.get("training_peak")
    if not isinstance(peak, dict):
        return None, None
    return peak.get("allocated_bytes"), peak.get("reserved_bytes")


def elapsed_from_result(result: dict[str, Any] | None) -> Any:
    if not result or "elapsed_s" not in result:
        return None
    return finite_number(result["elapsed_s"], f"{result.get('id', 'result')}/elapsed_s")


def result_summary_from_receipt(item: dict[str, Any], label: str, required: bool = True) -> dict[str, Any]:
    value, actual, status = verify_json_receipt(item, label, required=required)
    if status != "confirmed" or value is None or actual is None:
        return {"status": status, "result": value, "receipt": actual or item}
    allocated, reserved = training_peak_values(value)
    return {
        "status": "confirmed",
        "result": value,
        "receipt": actual,
        "elapsed_s": elapsed_from_result(value),
        "elapsed_s_provenance": f"{actual['path']}:elapsed_s",
        "training_peak_allocated_bytes": allocated,
        "training_peak_reserved_bytes": reserved,
        "training_peak_provenance": f"{actual['path']}:training_peak",
        "trainable_parameters_recorded": value.get("trainable_parameters"),
        "trainable_parameters_provenance": f"{actual['path']}:trainable_parameters",
    }


def result_summary_from_v11_run_id(run_id: str) -> dict[str, Any]:
    return result_summary_from_receipt({"path": str(HERE / "runs" / run_id / "result.json")}, run_id, required=True)


def path_from_cache_checkpoint_to_result(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    text = str(path)
    text = text.replace("\\.cache\\", "\\research\\").replace("/.cache/", "/research/")
    return Path(text).parent / "result.json"


def full_row(base: dict[str, Any]) -> dict[str, Any]:
    row = {field: None for field in ADAPTATION_COST_FIELDS}
    row.update(base)
    return row


def add_result_cost_row(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    seed: int | str | None,
    candidate: str,
    component: str,
    cost_boundary: str,
    current_round_increment: bool,
    selected: bool,
    result_summary: dict[str, Any],
    source_family: str,
    lr: Any = None,
    fit_attempt_count: int = 1,
    run_id: str | None = None,
    run_ids: list[str] | None = None,
    cost_model_info: dict[str, Any] | None = None,
    note: str = "",
) -> None:
    system_count, system_note = candidate_system_parameters(candidate, dataset)
    component_count = component_parameter_count(candidate, component, dataset)
    receipt_item = result_summary.get("receipt") or {}
    result = result_summary.get("result") or {}
    row = full_row(
        {
            "dataset": dataset,
            "seed": seed,
            "candidate": candidate,
            "component": component,
            "cost_boundary": cost_boundary,
            "current_round_increment": current_round_increment,
            "selected": selected,
            "fit_attempt_count": fit_attempt_count,
            "run_id": run_id or result.get("id"),
            "run_ids": ";".join(run_ids or ([run_id] if run_id else [])),
            "source_family": source_family,
            "lr": lr if lr is not None else result.get("spec", {}).get("lr"),
            "elapsed_s": result_summary.get("elapsed_s"),
            "elapsed_s_provenance": result_summary.get("elapsed_s_provenance"),
            "training_peak_allocated_bytes": result_summary.get("training_peak_allocated_bytes"),
            "training_peak_reserved_bytes": result_summary.get("training_peak_reserved_bytes"),
            "training_peak_provenance": result_summary.get("training_peak_provenance"),
            "trainable_parameters_recorded": result_summary.get("trainable_parameters_recorded"),
            "trainable_parameters_provenance": result_summary.get("trainable_parameters_provenance"),
            "component_source_parameter_count": component_count,
            "candidate_system_parameter_count": system_count,
            "source_parameter_note": system_note,
            "receipt_path": receipt_item.get("path"),
            "receipt_sha256": receipt_item.get("sha256"),
            "status": result_summary.get("status", "unknown"),
            "note": note,
        }
    )
    attach_cost_model_info(row, cost_model_info)
    rows.append(row)


def attach_cost_model_info(row: dict[str, Any], info: dict[str, Any] | None) -> None:
    if not info:
        return
    row["cost_model_info_source_key"] = "cost_summary.json:gpu.instances[].model where batch=4, execution=full"
    row["cost_current_neural_stage_registered_parameters"] = info.get("current_neural_stage_registered_parameters")
    row["cost_reused_parent_fitted_parameters"] = info.get("reused_parent_fitted_parameters")
    row["cost_gamma_coefficients"] = info.get("gamma_coefficients")
    row["cost_sum_stage_registered_fit_parameters"] = info.get("sum_stage_registered_fit_parameters")
    row["cost_parameter_bytes"] = info.get("parameter_bytes")
    row["cost_buffer_bytes"] = info.get("buffer_bytes")
    row["cost_deployment_tensor_bytes"] = info.get("deployment_tensor_bytes")


def aggregate_result_summaries(summaries: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not summaries:
        return {"status": "unknown_no_confirmed_files", "note": label}
    if any(item.get("status") != "confirmed" for item in summaries):
        bad = [item.get("status") for item in summaries if item.get("status") != "confirmed"]
        raise RuntimeError(f"Cannot aggregate unconfirmed results for {label}: {bad}")
    elapsed = [item.get("elapsed_s") for item in summaries]
    if any(value is None for value in elapsed):
        total = None
    else:
        total = sum(float(value) for value in elapsed)
    allocated = [item.get("training_peak_allocated_bytes") for item in summaries if item.get("training_peak_allocated_bytes") is not None]
    reserved = [item.get("training_peak_reserved_bytes") for item in summaries if item.get("training_peak_reserved_bytes") is not None]
    receipt_paths = [item["receipt"]["path"] for item in summaries]
    return {
        "status": "confirmed",
        "result": {"id": label},
        "receipt": {"path": ";".join(receipt_paths), "sha256": None},
        "elapsed_s": total,
        "elapsed_s_provenance": "sum of result.json:elapsed_s across listed child fit attempts; parent GPU job elapsed is not added",
        "training_peak_allocated_bytes": max(allocated) if allocated else None,
        "training_peak_reserved_bytes": max(reserved) if reserved else None,
        "training_peak_provenance": "max of result.json:training_peak across listed child fit attempts",
        "trainable_parameters_recorded": None,
        "trainable_parameters_provenance": "aggregate row; see child fit rows",
    }


def write_adaptation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No adaptation cost rows to write")
    require_new(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ADAPTATION_COST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ADAPTATION_COST_FIELDS})


def selected_run_ids(selection: dict[str, Any], dataset: str, family: str) -> list[str]:
    entry = selection["selected"][dataset][family]
    ids = list(entry.get("run_ids", []))
    if len(ids) != 2:
        raise ValueError(f"Expected two selected seed runs for {dataset}/{family}")
    return ids


def require_completed_selection(
    path: Path,
    expected_main_runs: int,
    *,
    allow_legacy_test_used_false: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required selection input is absent: {path}")
    value = read_json(path)
    if "test_used_for_selection" in value:
        if value.get("test_used_for_selection") is not False:
            raise RuntimeError(f"{path.name} must not use TEST for selection")
    elif allow_legacy_test_used_false and value.get("test_used") is False:
        pass
    else:
        raise RuntimeError(f"{path.name} must explicitly record TEST non-use")
    if len(value.get("all_neural_runs", [])) != expected_main_runs:
        raise ValueError(f"{path.name} expected {expected_main_runs} all_neural_runs entries")
    for dataset in DATASETS:
        if dataset not in value.get("selected", {}):
            raise ValueError(f"{path.name} missing selected dataset: {dataset}")
    return value


def require_selection_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = require_completed_selection(HERE / "selected.json", expected_main_runs=40)
    selected_direct = require_completed_selection(
        HERE / "selected_direct.json",
        expected_main_runs=4,
        allow_legacy_test_used_false=True,
    )
    validate_selection_run_ids(selection, selected_direct)
    gamma = require_completed_input(HERE / "gamma_selection.json")
    if len(gamma.get("rows", [])) != 12:
        raise ValueError("gamma_selection.json must contain all twelve coefficient fits")
    reuse = read_json(HERE / "reuse_manifest.json")
    ledger = read_json(HERE / "ledger.json")
    return selection, selected_direct, gamma, reuse, ledger


def validate_selection_run_ids(selection: dict[str, Any], selected_direct: dict[str, Any]) -> None:
    protocol = read_json(HERE / "protocol.json")
    planned = {}
    for spec in protocol.get("fits", protocol.get("neural_fits", [])):
        planned[spec["id"]] = (spec.get("dataset"), spec.get("family"))
    if not planned:
        raise ValueError("protocol.json does not expose approved fit ids")

    def check_record(record: dict[str, Any], source: str) -> None:
        run_id = record["id"]
        spec = record.get("spec", {})
        if run_id not in planned:
            raise ValueError(f"{source} contains a run id not present in protocol.json: {run_id}")
        if planned[run_id] != (spec.get("dataset"), spec.get("family")):
            raise ValueError(f"{source} dataset/family mismatch for {run_id}")

    for record in selection.get("all_neural_runs", []):
        check_record(record, "selected.json")
    for record in selected_direct.get("all_neural_runs", []):
        check_record(record, "selected_direct.json")
        if record.get("spec", {}).get("family") != "direct_nlinear":
            raise ValueError("selected_direct.json must contain only direct_nlinear run records")


def selected_run_id_map(selection: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    output: dict[str, dict[str, list[str]]] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for family, value in selection.get("selected", {}).get(dataset, {}).items():
            if isinstance(value, dict) and "run_ids" in value:
                output[dataset][family] = list(value["run_ids"])
    return output


def build_v11_run_index(selection: dict[str, Any], selected_direct: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for source_name, source in (("selected.json", selection), ("selected_direct.json", selected_direct)):
        for record in source.get("all_neural_runs", []):
            run_id = record["id"]
            if run_id in index:
                continue
            summary = result_summary_from_v11_run_id(run_id)
            index[run_id] = {"selection_record": record, "source": source_name, **summary}
    return index


def manifest_run_by_role(reuse: dict[str, Any], dataset: str, role: str, seed: int) -> dict[str, Any] | None:
    dataset_block = reuse.get("baselines", {}).get(dataset, {})
    role_ids = dataset_block.get("roles", {}).get(role, [])
    runs = dataset_block.get("runs", {})
    for run_id in role_ids:
        item = runs.get(run_id)
        if not item:
            continue
        if item.get("spec", {}).get("seed") == seed or item.get("seed") == seed:
            return item
    return None


def reuse_parent_summary(reuse: dict[str, Any], dataset: str, seed: int) -> dict[str, Any]:
    parent = reuse.get("parents", {}).get(dataset, {}).get(str(seed))
    if not parent:
        raise ValueError(f"Missing reused LEVEL parent manifest entry for {dataset}/{seed}")
    return result_summary_from_receipt(parent.get("result"), f"{dataset}/{seed}/LEVEL parent", required=True)


def reuse_direct_summary(reuse: dict[str, Any], dataset: str, seed: int) -> dict[str, Any]:
    direct = reuse.get("direct", {}).get(dataset, {}).get(str(seed))
    if not direct:
        raise ValueError(f"Missing reused direct manifest entry for {dataset}/{seed}")
    return result_summary_from_receipt(direct.get("result"), f"{dataset}/{seed}/DIRECT", required=True)


def reuse_native_summary(reuse: dict[str, Any], dataset: str, seed: int) -> dict[str, Any]:
    item = manifest_run_by_role(reuse, dataset, "lora_native", seed)
    if not item:
        return {"status": "unknown_missing_manifest_role", "receipt": None, "result": None}
    return result_summary_from_receipt(item.get("result"), f"{dataset}/{seed}/native LoRA", required=False)


def historical_direct_search_summaries(reuse: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        direct = reuse.get("direct", {}).get(dataset, {}).get(str(seed), {})
        selection_record = direct.get("selection_record")
        value, _, status = verify_json_receipt(selection_record, f"{dataset}/direct selection record", required=False)
        if status != "confirmed" or value is None:
            continue
        for record in value.get("all_neural_runs", []):
            spec = record.get("spec", {})
            if spec.get("family") != "direct_nlinear":
                continue
            run_id = record["id"]
            if run_id in summaries:
                continue
            result_path = path_from_cache_checkpoint_to_result(record["checkpoint"])
            summary = result_summary_from_receipt({"path": str(result_path)}, run_id, required=False)
            if summary.get("status") == "confirmed":
                summaries[run_id] = summary
    return list(summaries.values())


def ledger_job_by_label(ledger: dict[str, Any], label: str, category: str) -> dict[str, Any]:
    matches = [job for job in ledger.get("jobs", []) if job.get("label") == label and job.get("category") == category]
    if len(matches) != 1:
        raise ValueError(f"Expected one ledger job for {label}/{category}, found {len(matches)}")
    job = matches[0]
    if job.get("status") != "complete":
        raise RuntimeError(f"Ledger job is not complete for {label}/{category}: {job.get('status')}")
    return job


def gamma_result_summary(row: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    job = ledger_job_by_label(ledger, row["id"], "coefficient_fit")
    result_receipt = row.get("result") or {"path": str(HERE / "gamma" / f"{row['id']}.json")}
    value, actual, status = verify_json_receipt(result_receipt, row["id"], required=True)
    if status != "confirmed":
        raise RuntimeError(f"Gamma result is not confirmed for {row['id']}")
    return {
        "status": "confirmed",
        "result": value,
        "receipt": actual,
        "elapsed_s": finite_number(job.get("elapsed_s"), f"ledger/{row['id']}/elapsed_s"),
        "elapsed_s_provenance": f"adaptation_ledger_snapshot.json:jobs[label=={row['id']!r}].elapsed_s",
        "training_peak_allocated_bytes": None,
        "training_peak_reserved_bytes": None,
        "training_peak_provenance": None,
        "trainable_parameters_recorded": len(row.get("gamma", [])),
        "trainable_parameters_provenance": f"gamma_selection.json:rows[id=={row['id']!r}].gamma",
    }


def adaptation_ledger_snapshot(gamma: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    labels = sorted(row["id"] for row in gamma.get("rows", []))
    jobs = [ledger_job_by_label(ledger, label, "coefficient_fit") for label in labels]
    return {
        "schema": "tsfm_peft_v11_adaptation_ledger_snapshot_v1",
        "status": "complete",
        "scope": "Only coefficient_fit ledger jobs used for Gamma elapsed fields in adaptation_cost; the running report job is intentionally omitted.",
        "source": "Parsed ledger.json object before report output heartbeats can mutate the live ledger file.",
        "used_job_labels": labels,
        "jobs": jobs,
    }


def add_search_aggregate_row(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    candidate: str,
    component: str,
    cost_boundary: str,
    current_round_increment: bool,
    run_summaries: list[dict[str, Any]],
    source_family: str,
    note: str,
    cost_model_info: dict[str, Any] | None = None,
    expected_attempts: int = 4,
) -> None:
    confirmed = [item for item in run_summaries if item.get("status") == "confirmed"]
    if len(confirmed) != len(run_summaries) or len(confirmed) != expected_attempts:
        system_count, system_note = candidate_system_parameters(candidate, dataset)
        rows.append(
            full_row(
                {
                    "dataset": dataset,
                    "seed": "all",
                    "candidate": candidate,
                    "component": component,
                    "cost_boundary": cost_boundary,
                    "current_round_increment": current_round_increment,
                    "selected": False,
                    "fit_attempt_count": len(run_summaries),
                    "source_family": source_family,
                    "component_source_parameter_count": component_parameter_count(candidate, component, dataset),
                    "candidate_system_parameter_count": system_count,
                    "source_parameter_note": system_note,
                    "status": "unknown_unconfirmed_or_incomplete_search_files",
                    "note": note,
                }
            )
        )
        return
    aggregate = aggregate_result_summaries(confirmed, f"{dataset}/{candidate}/{component}")
    run_ids = [str(item.get("result", {}).get("id") or item.get("receipt", {}).get("path")) for item in confirmed]
    add_result_cost_row(
        rows,
        dataset=dataset,
        seed="all",
        candidate=candidate,
        component=component,
        cost_boundary=cost_boundary,
        current_round_increment=current_round_increment,
        selected=False,
        result_summary=aggregate,
        source_family=source_family,
        run_ids=run_ids,
        fit_attempt_count=len(confirmed),
        cost_model_info=cost_model_info,
        note=note,
    )


def build_adaptation_cost(evaluation: dict[str, Any], cost: dict[str, Any]) -> dict[str, Any]:
    del evaluation
    selection, selected_direct, gamma, reuse, ledger = require_selection_inputs()
    run_index = build_v11_run_index(selection, selected_direct)
    model_info = cost_model_info_index(cost)
    rows: list[dict[str, Any]] = []

    def selected_summary(run_id: str) -> dict[str, Any]:
        if run_id not in run_index:
            raise ValueError(f"Selected run id was not present in v11 all_neural_runs: {run_id}")
        return run_index[run_id]

    for dataset in DATASETS:
        for seed in SEEDS:
            parent_summary = reuse_parent_summary(reuse, dataset, seed)
            add_result_cost_row(
                rows,
                dataset=dataset,
                seed=seed,
                candidate="a_p_lora_qfixed",
                component="reused_level_parent",
                cost_boundary="historical_parent",
                current_round_increment=False,
                selected=True,
                result_summary=parent_summary,
                source_family="base_level",
                cost_model_info=model_info.get((dataset, "base_level", seed)),
                note="Reused parent costs zero GPU fit time in v11 but has nonzero historical training cost.",
            )

        a_run_ids = selected_run_ids(selection, dataset, "a_p_lora_qfixed")
        for run_id in a_run_ids:
            result = selected_summary(run_id)
            seed = result["result"]["spec"]["seed"]
            add_result_cost_row(
                rows,
                dataset=dataset,
                seed=seed,
                candidate="a_p_lora_qfixed",
                component="selected_a_fit",
                cost_boundary="v11_current_selected_fit",
                current_round_increment=True,
                selected=True,
                result_summary=result,
                source_family="a_p_lora_qfixed",
                run_id=run_id,
                cost_model_info=model_info.get((dataset, "a_p_lora_qfixed", seed)),
                note="Selected A LoRA fit; excludes reused LEVEL parent cost.",
            )
        a_search = [selected_summary(record["id"]) for record in selection["all_neural_runs"]
                    if record.get("spec", {}).get("dataset") == dataset and record.get("spec", {}).get("family") == "a_p_lora_qfixed"]
        add_search_aggregate_row(rows, dataset=dataset, candidate="a_p_lora_qfixed", component="a_2lr_search",
                                 cost_boundary="v11_current_lr_search", current_round_increment=True,
                                 run_summaries=a_search, source_family="a_p_lora_qfixed",
                                 note="Current A two-LR x two-seed search cost; selected rows are not added again.")

        full_run_ids = selected_run_ids(selection, dataset, "full_lora_mse")
        for run_id in full_run_ids:
            result = selected_summary(run_id)
            seed = result["result"]["spec"]["seed"]
            add_result_cost_row(
                rows,
                dataset=dataset,
                seed=seed,
                candidate="full_lora_mse",
                component="selected_full_lora_mse_fit",
                cost_boundary="v11_current_selected_fit",
                current_round_increment=True,
                selected=True,
                result_summary=result,
                source_family="full_lora_mse",
                run_id=run_id,
                cost_model_info=model_info.get((dataset, "full_lora_mse", seed)),
                note="Current full-input MSE LoRA reference; separate from native quantile LoRA.",
            )
        full_search = [selected_summary(record["id"]) for record in selection["all_neural_runs"]
                       if record.get("spec", {}).get("dataset") == dataset and record.get("spec", {}).get("family") == "full_lora_mse"]
        add_search_aggregate_row(rows, dataset=dataset, candidate="full_lora_mse", component="full_lora_mse_2lr_search",
                                 cost_boundary="v11_current_lr_search", current_round_increment=True,
                                 run_summaries=full_search, source_family="full_lora_mse",
                                 note="Current full-input MSE LoRA two-LR x two-seed reference search.")

        direct_run_ids = selected_run_ids(selection, dataset, "direct_nlinear")
        for run_id in direct_run_ids:
            seed_text = run_id.rsplit("_", 1)[-1]
            seed = int(seed_text) if seed_text.isdigit() else None
            if run_id.startswith("v11_"):
                result = selected_summary(run_id)
                boundary = "v11_current_selected_fit"
                current = True
            else:
                if seed is None:
                    raise ValueError(f"Cannot infer seed from reused direct run id: {run_id}")
                result = reuse_direct_summary(reuse, dataset, seed)
                boundary = "historical_reused_direct"
                current = False
            add_result_cost_row(
                rows,
                dataset=dataset,
                seed=seed,
                candidate="direct_nlinear",
                component="selected_direct_fit",
                cost_boundary=boundary,
                current_round_increment=current,
                selected=True,
                result_summary=result,
                source_family="direct_nlinear",
                run_id=run_id,
                cost_model_info=model_info.get((dataset, "direct_nlinear", seed)),
                note="S* selected direct NLinear cost; reused rows cost zero fit time in v11.",
            )
        if dataset == "hog":
            direct_search = [run_index[record["id"]] for record in selected_direct["all_neural_runs"]
                             if record.get("spec", {}).get("family") == "direct_nlinear"]
            direct_current = True
            direct_boundary = "v11_current_hog_direct_lr_search"
        else:
            direct_search = historical_direct_search_summaries(reuse, dataset)
            direct_current = False
            direct_boundary = "historical_direct_lr_search_if_confirmed"
        add_search_aggregate_row(rows, dataset=dataset, candidate="direct_nlinear", component="direct_2lr_search",
                                 cost_boundary=direct_boundary, current_round_increment=direct_current,
                                 run_summaries=direct_search, source_family="direct_nlinear",
                                 note="Direct NLinear two-LR search only when actual result files are confirmed.")

        for seed in SEEDS:
            system_count, system_note = candidate_system_parameters("b_fixed_u", dataset)
            row = full_row(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "candidate": "b_fixed_u",
                    "component": "fixed_u_no_optimizer_fit",
                    "cost_boundary": "fit0_fixed_pca_reference",
                    "current_round_increment": False,
                    "selected": True,
                    "fit_attempt_count": 0,
                    "source_family": "b_fixed_u",
                    "component_source_parameter_count": 0,
                    "candidate_system_parameter_count": system_count,
                    "source_parameter_note": system_note,
                    "status": "confirmed_fit0_by_definition",
                    "note": "B_FIXED_U uses fixed PCA U0; it has no U optimizer fit. It still depends on selected S*.",
                }
            )
            attach_cost_model_info(row, model_info.get((dataset, "b_fixed_u", seed)))
            rows.append(row)

        u_run_ids = selected_run_ids(selection, dataset, "b_learned_u")
        for run_id in u_run_ids:
            result = selected_summary(run_id)
            seed = result["result"]["spec"]["seed"]
            add_result_cost_row(
                rows,
                dataset=dataset,
                seed=seed,
                candidate="b_learned_u",
                component="selected_u_fit",
                cost_boundary="v11_current_selected_fit",
                current_round_increment=True,
                selected=True,
                result_summary=result,
                source_family="b_learned_u",
                run_id=run_id,
                cost_model_info=model_info.get((dataset, "b_learned_u", seed)),
                note="Selected U fit; excludes selected S* direct cost and later Gamma fit.",
            )
        u_search = [selected_summary(record["id"]) for record in selection["all_neural_runs"]
                    if record.get("spec", {}).get("dataset") == dataset and record.get("spec", {}).get("family") == "b_learned_u"]
        add_search_aggregate_row(rows, dataset=dataset, candidate="b_learned_u", component="u_2lr_search",
                                 cost_boundary="v11_current_lr_search", current_round_increment=True,
                                 run_summaries=u_search, source_family="b_learned_u",
                                 note="Current learned-U two-LR x two-seed search; selected S* and Gamma are separate rows.")

        for seed in SEEDS:
            native = reuse_native_summary(reuse, dataset, seed)
            add_result_cost_row(
                rows,
                dataset=dataset,
                seed=seed,
                candidate="lora_native",
                component="native_lora_historical_fit",
                cost_boundary="historical_native_lora_if_confirmed",
                current_round_increment=False,
                selected=True,
                result_summary=native,
                source_family="lora_native",
                run_id=(native.get("result") or {}).get("id"),
                cost_model_info=model_info.get((dataset, "lora_native", seed)),
                note="Historical native-quantile LoRA time is reported only when the result file receipt is confirmed; otherwise unknown.",
            )

    for gamma_row in gamma["rows"]:
        dataset = gamma_row["dataset"]
        seed = gamma_row["seed"]
        kind = gamma_row["kind"]
        candidate = f"b_{kind}_gamma"
        summary = gamma_result_summary(gamma_row, ledger)
        add_result_cost_row(
            rows,
            dataset=dataset,
            seed=seed,
            candidate=candidate,
            component="gamma_fit",
            cost_boundary="v11_current_gamma_fit",
            current_round_increment=True,
            selected=True,
            result_summary=summary,
            source_family=candidate,
            run_id=gamma_row["id"],
            cost_model_info=model_info.get((dataset, candidate, seed)),
            note="Bounded Gamma coefficient fit on VAL; elapsed comes from coefficient_fit ledger, not from a parent CPU-analysis job.",
        )

    return {
        "schema": "tsfm_peft_v11_adaptation_cost_v1",
        "status": "complete",
        "scope_notes": [
            "Rows separate historical reused training cost from v11 current incremental cost.",
            "Elapsed fields are heterogeneous wall-clock training/fit records including TRAIN/VAL/probes where the source result did so.",
            "Do not sum parent GPU/cpu_analysis jobs with child fit rows; aggregate rows sum only listed child result elapsed_s fields.",
            "These costs are not controlled training-speed ratios across historical campaigns.",
            "Source parameter counts are stage counts, not function-space dimension, total deployment model size, or scalar update count.",
        ],
        "inputs": {
            "selected": receipt(HERE / "selected.json"),
            "selected_direct": receipt(HERE / "selected_direct.json"),
            "gamma_selection": receipt(HERE / "gamma_selection.json"),
            "reuse_manifest": receipt(HERE / "reuse_manifest.json"),
        },
        "cost_metadata": {
            "selected_run_ids": selected_run_id_map(selection),
            "selected_direct_run_ids": selected_run_id_map(selected_direct),
            "baseline_recipe_boundaries": {
                "base_level": "Historical reused LEVEL parent rows; zero incremental v11 fit time, historical result elapsed retained when confirmed.",
                "direct_nlinear": "Selected S* is separate from B U/Gamma rows; Robin/Jena may be historical reuse, Hog may be current v11 fits.",
                "lora_native": "Historical native-quantile LoRA rows only when reuse_manifest result receipts are confirmed.",
                "full_lora_mse": "Current v11 full-input MSE LoRA reference, separate from native LoRA.",
            },
            "double_count_guard": "Selected child fit rows, LR-search aggregate rows, reused parent/S* rows, and Gamma coefficient rows are separate accounting views and must not be summed as one controlled speed ratio.",
        },
        "_ledger_snapshot": adaptation_ledger_snapshot(gamma, ledger),
        "rows": rows,
    }


def apply_report_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: Any, stem: Path) -> list[Path]:
    require_new(stem.with_suffix(".png"))
    require_new(stem.with_suffix(".svg"))
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [stem.with_suffix(".png"), stem.with_suffix(".svg")]


def plot_accuracy_by_dataset(family_json: dict[str, Any], output_stem: Path) -> list[Path]:
    apply_report_style()
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.0, 4.8), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        values = [family_json[dataset][family]["periods"]["combined"]["mse"] for family in FAMILIES]
        labels = [DISPLAY[family] for family in FAMILIES]
        positions = list(range(len(FAMILIES)))
        ax.barh(positions, values, color=[COLORS[family] for family in FAMILIES], edgecolor="black", linewidth=0.25)
        left_errors: list[float] = []
        right_errors: list[float] = []
        for value, family in zip(values, FAMILIES):
            seed_range = family_json[dataset][family]["periods"]["combined"]["selected_seed_range"]["mse"]
            if seed_range["count"] < 2:
                left_errors.append(0.0)
                right_errors.append(0.0)
            else:
                left_errors.append(max(0.0, value - seed_range["min"]))
                right_errors.append(max(0.0, seed_range["max"] - value))
        ax.errorbar(
            values,
            positions,
            xerr=[left_errors, right_errors],
            fmt="none",
            ecolor="black",
            elinewidth=0.7,
            capsize=2,
            capthick=0.7,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(labels if dataset == DATASETS[0] else [])
        ax.invert_yaxis()
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Combined TEST MSE")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "v11 historical development TEST accuracy; whiskers are 2-seed min/max ranges, not CI; F0 is one point",
        fontsize=10,
    )
    return save_figure(fig, output_stem)


def plot_accuracy_cost(
    family_json: dict[str, Any],
    gpu_options: list[dict[str, Any]],
    output_stem: Path,
) -> list[Path]:
    apply_report_style()
    mse_by_dataset = metric_map_for_combined(family_json)
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(12.0, 7.8))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.90, hspace=0.38, wspace=0.22)
    for column, dataset in enumerate(DATASETS):
        dataset_rows = [row for row in gpu_options if row["dataset"] == dataset]
        for row in dataset_rows:
            family = row["family"]
            execution = row["execution"]
            marker = MARKERS.get(str(execution), "o")
            x = mse_by_dataset[dataset][family]
            throughput = row["mean_seed_median_origins_s"]
            throughput_range = row["origins_s_range_from_block_ms"]
            axes[0, column].errorbar(
                x,
                throughput,
                yerr=[
                    [max(0.0, throughput - throughput_range[0])],
                    [max(0.0, throughput_range[1] - throughput)],
                ],
                fmt=marker,
                color=COLORS[family],
                ecolor=COLORS[family],
                markeredgecolor="black",
                linewidth=0.25,
                elinewidth=0.6,
                capsize=2,
                alpha=0.85,
            )
            allocated = row["allocated_mib"]
            allocated_range = row["allocated_range_mib"]
            yerr = None
            if allocated_range is not None:
                yerr = [
                    [max(0.0, allocated - allocated_range[0])],
                    [max(0.0, allocated_range[1] - allocated)],
                ]
            axes[1, column].errorbar(
                x,
                allocated,
                yerr=yerr,
                fmt=marker,
                color=COLORS[family],
                ecolor=COLORS[family],
                markeredgecolor="black",
                linewidth=0.25,
                elinewidth=0.6,
                capsize=2,
                alpha=0.85,
            )
        axes[0, column].set_title(dataset.capitalize())
        axes[0, column].set_xlabel("Combined TEST MSE")
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylabel("Batch4 origins/s (log scale)" if column == 0 else "")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].set_xlabel("Combined TEST MSE")
        axes[1, column].set_ylabel("Peak allocated MiB" if column == 0 else "")
        axes[1, column].grid(alpha=0.25)

    family_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=COLORS[family], label=DISPLAY[family], markersize=5)
        for family in FAMILIES
    ]
    execution_handles = [
        plt.Line2D([0], [0], marker=marker, linestyle="", color="white", markeredgecolor="black", label=name, markersize=5)
        for name, marker in MARKERS.items()
        if any(row["execution"] == name for row in gpu_options)
    ]
    fig.legend(
        handles=family_handles + execution_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncols=4,
        frameon=False,
        borderaxespad=0.1,
    )
    fig.suptitle(
        "v11 same-session GPU batch4 options; cost bars are seed x block min/max ranges, not CI",
        fontsize=10,
    )
    return save_figure(fig, output_stem)


def build_report(job: Job) -> dict[str, Any]:
    evaluation_path = HERE / "evaluation01.json"
    cost_path = HERE / "cost_summary.json"
    family_csv = HERE / "report_family_comparison.csv"
    family_json_path = HERE / "report_family_comparison.json"
    adaptation_csv = HERE / "adaptation_cost.csv"
    adaptation_json_path = HERE / "adaptation_cost.json"
    adaptation_ledger_snapshot_path = HERE / "adaptation_ledger_snapshot.json"
    report_values_path = HERE / "report_values.json"
    figure_dir = HERE / "figures"
    accuracy_stem = figure_dir / "v11_accuracy_by_dataset"
    cost_stem = figure_dir / "v11_accuracy_cost_batch4"

    for output in (
        family_csv,
        family_json_path,
        adaptation_csv,
        adaptation_json_path,
        adaptation_ledger_snapshot_path,
        report_values_path,
        accuracy_stem.with_suffix(".png"),
        accuracy_stem.with_suffix(".svg"),
        cost_stem.with_suffix(".png"),
        cost_stem.with_suffix(".svg"),
    ):
        require_new(output)

    evaluation = require_completed_input(evaluation_path)
    cost = require_completed_input(cost_path)
    validate_evaluation(evaluation)
    rows, family_nested = build_family_rows(evaluation)
    gpu_options, cpu_options = collect_batch4_cost_options(cost)
    adaptation = build_adaptation_cost(evaluation, cost)
    job.heartbeat("validated completed evaluation/cost/adaptation inputs")

    write_csv(family_csv, rows)
    comparison_json = {
        "schema": "tsfm_peft_v11_report_family_comparison_v1",
        "status": "complete",
        "scope": "exposed development datasets; raw MSE/MAE are reported per dataset and are not pooled across datasets",
        "independent_confirmation": False,
        "families": list(FAMILIES),
        "reference_families": list(REFERENCE_FAMILIES),
        "datasets": family_nested,
        "source": {
            "evaluation01": receipt(evaluation_path),
            "relative_gap_formula": "100 * (candidate - reference) / reference; positive means candidate is larger",
        },
    }
    save_json(family_json_path, comparison_json)
    job.heartbeat("wrote family comparison table")

    ledger_snapshot = adaptation.pop("_ledger_snapshot")
    save_json(adaptation_ledger_snapshot_path, ledger_snapshot)
    adaptation["inputs"]["ledger_snapshot"] = receipt(adaptation_ledger_snapshot_path)
    write_adaptation_csv(adaptation_csv, adaptation["rows"])
    save_json(adaptation_json_path, adaptation)
    job.heartbeat("wrote adaptation cost table")

    accuracy_figures = plot_accuracy_by_dataset(family_nested, accuracy_stem)
    cost_figures = plot_accuracy_cost(family_nested, gpu_options, cost_stem)
    figure_paths = accuracy_figures + cost_figures
    job.heartbeat("wrote report figures")

    generated = {
        "family_comparison_csv": receipt(family_csv),
        "family_comparison_json": receipt(family_json_path),
        "adaptation_cost_csv": receipt(adaptation_csv),
        "adaptation_cost_json": receipt(adaptation_json_path),
        "adaptation_ledger_snapshot_json": receipt(adaptation_ledger_snapshot_path),
        "figures": [receipt(path) for path in figure_paths],
    }
    values = {
        "schema": "tsfm_peft_v11_report_values_v1",
        "status": "complete",
        "inputs": {
            "evaluation01": receipt(evaluation_path),
            "cost_summary": receipt(cost_path),
        },
        "generated": generated,
        "scope_notes": [
            "All Robin/Jena/Hog results are exposed development results, not independent confirmation.",
            "Family accuracy values use the saved selected-seed loss means for each period and overall; they are not prediction ensembles.",
            "Accuracy figure whiskers are selected-seed min/max ranges, not confidence intervals; one-id F0 rows are one point.",
            "Cost figure bars are min/max over selected instance x three block medians, not confidence intervals.",
            "Adaptation cost rows separate reused historical parent/S*/native LoRA costs from this-round incremental fits.",
            "This report does not choose a final method or make a causal novelty claim.",
            "Accuracy values are mapped to their source keys; relative gaps are computed, not separately measured.",
            "Cost values are same-session values from cost_summary.json and are not ratios against historical v4/v6/v7 timings.",
        ],
        "numeric_source_mappings": {
            "mse_mae_signed_bias": {
                "source": "evaluation01.json",
                "key_template": "datasets.{dataset}.role_summary.{family}.periods.{period}.{metric}",
                "datasets": list(DATASETS),
                "families": list(FAMILIES),
                "periods": list(PERIODS),
                "metrics": ["mse", "mae", "signed_mean_error"],
            },
            "selected_seed_ranges": {
                "source": "evaluation01.json",
                "key_template": "datasets.{dataset}.models.{id}.periods.{period}.{metric}",
                "range_definition": "min/max over selected model ids in datasets.{dataset}.role_summary.{family}.ids; not CI",
                "included_in": ["report_family_comparison.csv", "report_family_comparison.json", "accuracy figure"],
            },
            "relative_gaps_pct": {
                "source": "report_family_comparison.csv/json",
                "formula": "100 * (candidate - reference) / reference; positive means larger candidate error",
                "references": list(REFERENCE_FAMILIES),
            },
            "paired_intervals": collect_paired_interval_mappings(evaluation),
            "batch4_gpu_cost_options": {
                "source": "cost_summary.json",
                "key": "gpu.groups rows with batch == 4",
                "count": len(gpu_options),
                "options": gpu_options,
            },
            "batch4_cpu_cost_options": {
                "source": "cost_summary.json",
                "key": "cpu.groups rows with batch == 4",
                "count": len(cpu_options),
                "options": cpu_options,
            },
            "adaptation_cost": {
                "source": "adaptation_cost.json/csv",
                "row_count": len(adaptation["rows"]),
                "elapsed_field_warning": "Elapsed values are copied from the exact source fields named in each row and are not controlled speed ratios.",
                "parameter_warning": "Source parameter counts are stage counts, not scalar update totals or deployment parameter counts.",
            },
        },
    }
    save_json(report_values_path, values)
    job.heartbeat("wrote report_values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v11 compact report tables and figures from completed summaries.")
    parser.add_argument("--job", required=True, help="Unique CPU-analysis ledger job label.")
    parser.add_argument("--reserve-seconds", type=float, required=True, help="CPU-analysis seconds to reserve.")
    args = parser.parse_args()
    with Job(args.job, category="cpu_analysis", reserve_seconds=args.reserve_seconds, metadata={"phase": "report_v11"}) as job:
        build_report(job)


if __name__ == "__main__":
    main()
