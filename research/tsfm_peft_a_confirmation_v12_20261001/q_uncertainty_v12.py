"""Prespecified v11 block resampling applied to the saved v12 Q ablations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np

from runtime_v12 import HERE, CACHE, OLD, Job, artifact, read_json, save_json, import_file
from q_contribution_v12 import array_receipt, load_prediction, verify_file, SCOPE

DATASETS = ("robin", "jena", "hog")
SEEDS = (92601, 92602)
PERIODS = ("test_a", "test_b", "combined")
FAMILIES = ("A_FULL", "A_MAIN", "A_MAIN_LAST")
PAIRS = (("A_FULL", "A_MAIN"), ("A_FULL", "A_MAIN_LAST"), ("A_MAIN_LAST", "A_MAIN"))
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 2000, 9262026


def bootstrap_weights(indices, block, dataset):
    """Exact v11 non-circular, within-period starts and fixed seed schedule."""
    total = sum(len(indices[period]) for period in PERIODS[:2])
    weights = {}
    for period_index, period in enumerate(PERIODS[:2]):
        index = indices[period]
        n = len(index)
        if n < block:
            raise ValueError("Insufficient period for the fixed v11 block length")
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, DATASETS.index(dataset), period_index]))
        matrix = np.zeros((BOOTSTRAP_DRAWS, total), dtype=np.int32)
        for draw in range(BOOTSTRAP_DRAWS):
            starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
            local = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
            matrix[draw] = np.bincount(index[local], minlength=total)
        weights[period] = matrix
    weights["combined"] = weights["test_a"] + weights["test_b"]
    return weights


def error_terms(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction/target/mask shape mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite prediction or observed target")
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    return {"squared": np.sum(error * error, axis=1),
            "absolute": np.sum(np.abs(error), axis=1), "count": np.sum(mask, axis=1)}


def point_scores(terms, index):
    count = terms["count"][index].sum(axis=0)
    if np.any(count == 0):
        raise ValueError("Unobserved fixed channel")
    return {metric: float(np.mean(terms[key][index].sum(axis=0) / count))
            for metric, key in (("mse", "squared"), ("mae", "absolute"))}


def compare(terms, draws, indices, seed):
    rows = []
    for left, reference in PAIRS:
        for period, index in indices.items():
            lhs, rhs = point_scores(terms[left], index), point_scores(terms[reference], index)
            for metric in ("mse", "mae"):
                a, b = draws[period][left][metric], draws[period][reference][metric]
                if rhs[metric] <= 0 or np.any(b <= 0):
                    raise ValueError("Relative intervals require positive reference scores")
                absolute = a - b
                relative = 100.0 * absolute / b
                abs_ci = np.quantile(absolute, [.025, .975]).tolist()
                rows.append({"seed": seed, "candidate": left, "reference": reference,
                             "period": period, "metric": metric,
                             "candidate_value": lhs[metric], "reference_value": rhs[metric],
                             "absolute_difference": lhs[metric] - rhs[metric],
                             "relative_change_percent": 100.0 * (lhs[metric] - rhs[metric]) / rhs[metric],
                             "conditional_block95_absolute": abs_ci,
                             "conditional_block95_relative_percent": np.quantile(relative, [.025, .975]).tolist(),
                             "conditional_interval_excludes_zero": bool(abs_ci[1] < 0 or abs_ci[0] > 0)})
    return rows


def analyze(name, source_name, job):
    started = time.perf_counter()
    source_path = HERE / f"{source_name}.json"
    source = read_json(source_path)
    if source["status"] != "complete":
        raise ValueError("Q contribution source is incomplete")
    for item in source["source_files"].values():
        verify_file(item)
    oldruntime = import_file(OLD / "runtime_v11.py", "v12_q_uncertainty_old_runtime")
    folder = CACHE / name
    if folder.exists() or (HERE / f"{name}.json").exists():
        raise FileExistsError("Preserve the existing uncertainty attempt")
    folder.mkdir(parents=True)
    result = {"schema": "v12_Q_contribution_conditional_uncertainty_v1", "name": name, "status": "running",
              "scope": SCOPE,
              "direction": "candidate minus reference; negative favors candidate (opposite sign to positive Q gains)",
              "interval_scope": "Percentile block95 conditional on fixed selected model/parent seeds and observed development periods. "
                                "No fitting/selection uncertainty, no seed-population uncertainty, no multiplicity adjustment, no independent confirmation.",
              "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                            "period_boundaries_crossed": False, "circular": False,
                            "paired_channels_methods_and_seeds": True,
                            "seed_aggregation": "mean loss of both fixed seeds using identical origin resamples; never prediction ensemble"},
              "source": artifact(source_path), "script": artifact(Path(__file__)),
              "reference_bootstrap_source": artifact(OLD / "evaluate_v11.py"),
              "reference_function_lines": [309, 328, 358, 395],
              "new_fits": 0, "backbone_loads": 0, "backbone_forward_calls": 0, "datasets": {}}
    csv_rows = []
    for dataset in DATASETS:
        job.check_limits()
        prior = source["datasets"][dataset]
        data = oldruntime.load_data(dataset, include_test=True)
        if artifact(data["_path"])["sha256"] != prior["data_arrays"]["sha256"]:
            raise ValueError("Data source differs from Q point estimates")
        origins = np.concatenate([data[period + "_origins"] for period in PERIODS[:2]]).astype(np.int64)
        if array_receipt(origins) != prior["origins"]:
            raise ValueError("Origins differ from Q point estimates")
        n_a = len(data["test_a_origins"])
        indices = {"test_a": np.arange(n_a), "test_b": np.arange(n_a, len(origins)),
                   "combined": np.arange(len(origins))}
        target = np.stack([data["x"][origin:origin + 48] for origin in origins])
        mask = np.stack([data["finite"][origin:origin + 48] for origin in origins]).astype(bool)
        if array_receipt(mask) != prior["target_mask"]:
            raise ValueError("Observed target mask differs from Q point estimates")
        block = 42 if dataset == "jena" else 7
        weights = bootstrap_weights(indices, block, dataset)
        for period, matrix in weights.items():
            if not np.all(matrix.sum(axis=1) == len(indices[period])):
                raise ValueError("Bootstrap origin counts differ from fixed period size")
            excluded = np.setdiff1d(indices["combined"], indices[period])
            if matrix[:, excluded].any():
                raise ValueError("A bootstrap draw crossed a period boundary")
        terms = {}
        prediction_receipts = []
        for seed in SEEDS:
            reconstruction = prior["reconstruction"][str(seed)]
            stored = load_prediction(reconstruction["saved_A_prediction"], origins, target.shape, with_main=True)
            derived_path = verify_file(reconstruction["derived_arrays"])
            with np.load(derived_path, allow_pickle=False) as derived:
                if not np.array_equal(derived["origins"], origins):
                    raise ValueError("Derived MAIN_LAST origins changed")
                main_last = derived["main_last_prediction"].copy()
            predictions = {"A_FULL": stored["prediction"], "A_MAIN": stored["main_only_prediction"],
                           "A_MAIN_LAST": main_last}
            terms[seed] = {family: error_terms(prediction, target, mask) for family, prediction in predictions.items()}
            for family, prediction in predictions.items():
                previous_model = next(m for m in prior["models"].values() if m["family"] == family and m["seed"] == seed)
                if array_receipt(prediction) != previous_model["prediction"]:
                    raise ValueError("Prediction tensor differs from Q point estimates")
                for period, index in indices.items():
                    score = point_scores(terms[seed][family], index)
                    if any(abs(score[metric] - previous_model["periods"][period][metric]) > 1e-10 for metric in score):
                        raise ValueError("Independent origin-term score differs from Q point estimate")
            prediction_receipts.append({"seed": seed, "saved": reconstruction["saved_A_prediction"],
                                        "derived": reconstruction["derived_arrays"]})
        terms["mean_fixed_seeds"] = {}
        for family in FAMILIES:
            if not np.array_equal(terms[SEEDS[0]][family]["count"], terms[SEEDS[1]][family]["count"]):
                raise ValueError("Paired seed target masks differ")
            terms["mean_fixed_seeds"][family] = {key: np.mean([terms[seed][family][key] for seed in SEEDS], axis=0)
                                               for key in ("squared", "absolute")}
            terms["mean_fixed_seeds"][family]["count"] = terms[SEEDS[0]][family]["count"]
        comparisons, exported_draws = [], {}
        for seed, family_terms in terms.items():
            draws = {}
            for period, matrix in weights.items():
                denominator = matrix @ family_terms["A_FULL"]["count"]
                if np.any(denominator == 0):
                    raise ValueError("A bootstrap draw has an unobserved fixed channel")
                draws[period] = {}
                for family, value in family_terms.items():
                    job.check_limits()
                    draws[period][family] = {metric: np.mean((matrix @ value[key]) / denominator, axis=1)
                                             for metric, key in (("mse", "squared"), ("mae", "absolute"))}
                    for metric, vector in draws[period][family].items():
                        exported_draws[f"{seed}__{period}__{family}__{metric}"] = vector
            comparisons.extend(compare(family_terms, draws, indices, seed))
        weights_path, draws_path = folder / f"{dataset}_weights.npz", folder / f"{dataset}_draws.npz"
        np.savez_compressed(weights_path, origins=origins, **weights)
        np.savez_compressed(draws_path, **exported_draws)
        result["datasets"][dataset] = {"block_origins": block, "predictions": prediction_receipts,
            "weights": artifact(weights_path), "draws": artifact(draws_path),
            "weight_array_hashes": {period: array_receipt(matrix) for period, matrix in weights.items()},
            "point_estimate_replay": "All six MSE/MAE period scores per seed/family matched within absolute 1e-10",
            "comparisons": comparisons}
        for item in comparisons:
            row = {"dataset": dataset, **{k: v for k, v in item.items() if not isinstance(v, list)}}
            for key in ("conditional_block95_absolute", "conditional_block95_relative_percent"):
                row[key + "_lower"], row[key + "_upper"] = item[key]
            csv_rows.append(row)
        save_json(HERE / f"{name}_partial.json", result)
        job.heartbeat(f"Q uncertainty {dataset}: 2000 paired fixed-seed, period-stratified block draws complete")
    csv_path = HERE / f"{name}.csv"
    if csv_path.exists():
        raise FileExistsError(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    result["status"] = "complete"
    result["elapsed_wall_s"] = time.perf_counter() - started
    result["csv"] = artifact(csv_path)
    save_json(HERE / f"{name}.json", result)
    lines = ["# Conditional Q contribution intervals", "", SCOPE, "", result["interval_scope"], "",
             result["direction"], "", "The same non-circular block resampling as v11 is used: "
             "2000 draws, fixed seed 9262026, block 7 origins for Robin/Hog and 42 for Jena. "
             "test_a and test_b are independently resampled without crossing boundaries, and "
             "combined pools their per-channel error sums/counts. Both fixed seeds share all resamples.", "",
             "```text", "dataset candidate/reference         metric delta      conditional block95 (combined, mean fixed seeds)"]
    for dataset, data in result["datasets"].items():
        for item in data["comparisons"]:
            if item["seed"] == "mean_fixed_seeds" and item["period"] == "combined":
                interval = item["conditional_block95_absolute"]
                label = item["candidate"] + "/" + item["reference"]
                lines.append(f"{dataset:7s} {label:27s} {item['metric']:4s} {item['absolute_difference']: .8f} [{interval[0]: .8f}, {interval[1]: .8f}]")
    lines.extend(["```", "", "Per-seed and per-period absolute and relative intervals are retained in JSON/CSV. "
                  "An interval excluding zero describes this conditional resampling analysis only; it "
                  "does not establish fresh-unit generalization or account for model/learning-rate selection.", ""])
    (HERE / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="q_uncertainty01")
    parser.add_argument("--source", default="q_contribution02")
    args = parser.parse_args()
    if not re.fullmatch(r"q_uncertainty[0-9]+", args.name) or not re.fullmatch(r"q_contribution[0-9]+", args.source):
        raise ValueError("Use numbered uncertainty/contribution attempt names")
    with Job("cpu_analysis", args.name, reserve_s=900) as job:
        try:
            result = analyze(args.name, args.source, job)
        except Exception:
            save_json(HERE / f"{args.name}_failure_{time.time_ns()}.json",
                      {"status": "failed", "exception": traceback.format_exc(), "source": artifact(Path(__file__)),
                       "name": args.name, "contribution_source": args.source})
            raise
        print(json.dumps({"status": result["status"], "result": str(HERE / f"{args.name}.json"),
                          "elapsed_wall_s": result["elapsed_wall_s"], "new_fits": 0, "backbone_forward_calls": 0}))


if __name__ == "__main__":
    main()
