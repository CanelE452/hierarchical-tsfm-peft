from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import numpy as np

from evaluate_v11 import aggregate_terms, bootstrap_weights, error_terms, space_diagnostic
from gamma_v11 import DATASETS, SEEDS, direct_objective, weighted_problem
from runtime_v11 import HERE, Job, digest, read_json, save_json, sources_for


def require_close(actual, expected, label, atol=1e-12):
    if not np.allclose(actual, expected, atol=atol, rtol=1e-12):
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def check_syntax():
    rows = []
    for path in sorted(HERE.glob("*.py")):
        source = path.read_bytes()
        ast.parse(source.decode("utf-8-sig"), filename=str(path))
        rows.append({"path": str(path), "sha256": hashlib.sha256(source).hexdigest()})
    if not rows:
        raise AssertionError("No v11 sources found")
    return {"status": "passed", "parsed_files": rows, "code_executed": False}


def check_weighted_gamma_objective():
    S = np.asarray([[[0.2, -0.3, 0.5], [0.7, 0.4, -0.6]],
                    [[-0.1, 0.9, 0.3], [0.8, -0.5, 0.0]]], dtype=np.float64)
    z = np.asarray([[[0.4, -0.7], [-0.6, 0.8]], [[0.5, 0.2], [0.1, -0.9]]], dtype=np.float64)
    U = np.asarray([[1 / np.sqrt(2), 0], [1 / np.sqrt(2), 0], [0, 1]], dtype=np.float64)
    target = np.asarray([[[0.0, 1.0, -0.2], [0.5, np.nan, 0.6]],
                         [[-0.4, np.nan, 0.3], [np.nan, -0.8, 0.0]]], dtype=np.float64)
    mask = np.isfinite(target)
    gamma = np.asarray([0.25, 0.75], dtype=np.float64)
    matrix, response, counts = weighted_problem(S, z, U, target, mask)
    prediction = S + (z * gamma) @ U.T
    manual_by_channel = []
    for channel in range(3):
        errors = []
        for origin in range(2):
            for horizon in range(2):
                if mask[origin, horizon, channel]:
                    errors.append((prediction[origin, horizon, channel] - target[origin, horizon, channel]) ** 2)
        manual_by_channel.append(sum(errors) / len(errors))
    manual = sum(manual_by_channel) / 3
    weighted = float(np.sum((matrix @ gamma - response) ** 2))
    direct = direct_objective(prediction, target, mask)
    require_close(counts, [3, 2, 4], "Observed channel counts")
    require_close(weighted, manual, "Weighted least-squares versus independently enumerated macro MSE")
    require_close(direct, manual, "Direct masked objective versus independently enumerated macro MSE")
    if len(response) != 9 or not np.isfinite(matrix).all() or not np.isfinite(response).all():
        raise AssertionError("Missing target rows entered Gamma's observed design")
    changed_missing = target.copy()
    changed_missing[~mask] = 987654.0
    altered_matrix, altered_response, altered_counts = weighted_problem(S, z, U, changed_missing, mask)
    require_close(altered_matrix, matrix, "Unobserved target substitution leaves design unchanged")
    require_close(altered_response, response, "Unobserved target substitution leaves response unchanged")
    require_close(altered_counts, counts, "Unobserved target substitution leaves counts unchanged")
    require_close(direct_objective(prediction, changed_missing, mask), manual,
                  "Masked objective ignores arbitrary unobserved targets")
    return {"status": "passed", "weighted_objective": weighted, "manual_macro_mse": manual,
            "channel_counts": counts.tolist(), "observed_zero_targets": int(np.sum(mask & (target == 0))),
            "solver_calls": 0, "optimization": False, "data": "small synthetic arrays only"}


def check_period_pooling():
    target = np.asarray([[[1.0, 2.0], [1.0, np.nan]], [[3.0, 4.0], [np.nan, np.nan]]])
    mask = np.isfinite(target)
    terms = error_terms(np.zeros_like(target), target, mask)
    first = aggregate_terms(terms, np.asarray([0]))
    second = aggregate_terms(terms, np.asarray([1]))
    combined = aggregate_terms(terms, np.asarray([0, 1]))
    expected = (11 / 3 + 20 / 2) / 2
    require_close(combined["target_counts"], [3, 2], "Pooled observed channel counts")
    require_close(combined["squared_error_sums"], [11, 20], "Pooled channel SSE")
    require_close(combined["mse"], expected, "Pool SSE/count before averaging channels")
    simple_period_average = (first["mse"] + second["mse"]) / 2
    micro_average = 31 / 5
    if abs(combined["mse"] - simple_period_average) < 1e-8 or abs(combined["mse"] - micro_average) < 1e-8:
        raise AssertionError("Counterexample must distinguish pooled channel-macro from both incorrect averages")
    require_close(combined["signed_mean_error"], ((-5 / 3) + (-6 / 2)) / 2,
                  "Signed error uses prediction minus target and channel-macro aggregation")
    return {"status": "passed", "correct_combined_mse": combined["mse"],
            "incorrect_period_mean": simple_period_average, "incorrect_micro_mse": micro_average,
            "combined_channel_counts": combined["target_counts"]}


def check_missing_projection_and_zero():
    target = np.asarray([[[1.0, 2.0], [3.0, np.nan]], [[0.0, 4.0], [5.0, 6.0]]])
    mask = np.isfinite(target)
    prediction = np.zeros_like(target)
    U = np.asarray([[1.0], [0.0]])
    index = np.asarray([0, 1])
    terms = error_terms(prediction, target, mask)
    score = aggregate_terms(terms, index)
    projected = space_diagnostic(prediction, target, mask, U, index)
    require_close(score["target_counts"], [4, 3], "Observed zero contributes one observed target")
    require_close(projected["coverage"], 3 / 4, "Projection complete-vector coverage")
    require_close(projected["p_mse"], 26 / 6, "Complete-vector P energy")
    require_close(projected["q_mse"], 56 / 6, "Complete-vector Q energy")
    require_close(projected["complete_vector_mse"], 82 / 6, "Complete-vector total energy")
    require_close(projected["orthogonal_energy_sum_gap"], 0.0, "P/Q Euclidean identity")
    if projected["complete_rows"] != 3 or not np.isfinite(score["mse"]):
        raise AssertionError("Missing NaN target contaminated masked or complete-vector analysis")
    changed = target.copy()
    changed[~mask] = -123456.0
    check = space_diagnostic(prediction, changed, mask, U, index)
    for key in ("p_mse", "q_mse", "complete_vector_mse", "coverage"):
        require_close(check[key], projected[key], f"Missing target is not projected: {key}")
    return {"status": "passed", "coverage": projected["coverage"],
            "complete_vector_mse": projected["complete_vector_mse"], "masked_macro_mse": score["mse"],
            "observed_counts": score["target_counts"], "missing_target_zero_filled_for_projection": False}


def check_paired_bootstrap():
    indices = {"test_a": np.arange(10), "test_b": np.arange(10, 22), "combined": np.arange(22)}
    weights = bootstrap_weights(indices, block=7, dataset="robin")
    for period, length in (("test_a", 10), ("test_b", 12), ("combined", 22)):
        matrix = weights[period]
        if matrix.shape != (2000, 22) or np.any(matrix < 0) or not np.issubdtype(matrix.dtype, np.integer):
            raise AssertionError("Invalid bootstrap frequency matrix")
        require_close(matrix.sum(axis=1), np.full(2000, length), f"Fixed number of origins in {period}")
    if weights["test_a"][:, 10:].any() or weights["test_b"][:, :10].any():
        raise AssertionError("A bootstrap block crossed the period boundary")
    require_close(weights["combined"], weights["test_a"] + weights["test_b"], "Stratified combined resampling")
    rng = np.random.default_rng(np.random.SeedSequence([9262026, 0, 0]))
    starts = rng.integers(0, 10 - 7 + 1, size=2)
    first_draw = np.concatenate([np.arange(start, start + 7) for start in starts])[:10]
    expected = np.bincount(first_draw, minlength=22)
    require_close(weights["test_a"][0], expected, "Prespecified contiguous-block first draw")
    origin_errors = np.arange(1, 23, dtype=np.float64)
    errors = np.column_stack([origin_errors, 3 * origin_errors])
    counts = np.ones_like(errors)
    matrix = weights["combined"]
    score_channels = (matrix @ errors) / (matrix @ counts)
    require_close(score_channels[:, 1], 3 * score_channels[:, 0], "All channels move under the same origin resampling")
    method_b = (matrix @ (errors + 2.0)) / (matrix @ counts)
    require_close(method_b - score_channels, np.full((2000, 2), 2.0),
                  "Paired methods share identical bootstrap origins")
    return {"status": "passed", "draws": 2000, "block": 7, "seed": 9262026,
            "period_boundaries_crossed": False, "joint_channels_and_methods": True,
            "first_draw_indices": first_draw.tolist()}


def check_json_sources():
    protocol = read_json(HERE / "protocol.json")
    reuse = read_json(HERE / "reuse_manifest.json")
    direct = read_json(HERE / "selected_direct.json")
    if set(protocol["datasets"]) != set(DATASETS) or len(protocol["neural_fits"]) != 40:
        raise AssertionError("Protocol no longer defines the approved three datasets and forty neural fits")
    expected_gamma = {f"v11_{dataset}_{kind}_gamma_{seed}" for dataset in DATASETS
                      for kind in ("fixed", "learned") for seed in SEEDS}
    if {row["id"] for row in protocol["coefficient_fits"]} != expected_gamma:
        raise AssertionError("Gamma ledger IDs do not match all twelve approved coefficient fits")
    if direct.get("test_used", direct.get("test_used_for_selection")) is not False:
        raise AssertionError("Direct selection must exclude TEST")
    rows = []
    for dataset in DATASETS:
        selected = direct["selected"][dataset]["direct_nlinear"]
        if any(len(selected[key]) != 2 for key in ("run_ids", "checkpoints", "checkpoint_sha256")):
            raise AssertionError("Direct selection must provide both paired seed checkpoints")
        for index, seed in enumerate(SEEDS):
            source = sources_for(dataset, seed, require_direct=True)
            parent = reuse["parents"][dataset][str(seed)]
            if parent["spec"]["dataset"] != dataset or parent["spec"].get("ed_mode") != "fixed_ed":
                raise AssertionError("A's parent is not the approved same-dataset fixed PCA LEVEL")
            if Path(source["level_checkpoint"]).resolve() != Path(parent["checkpoint"]["path"]).resolve():
                raise AssertionError("A source is not the learned LEVEL parent's checkpoint")
            if source["level_checkpoint_sha256"] != parent["checkpoint"]["sha256"]:
                raise AssertionError("A parent receipt differs from reuse manifest")
            if Path(source["direct_checkpoint"]).resolve() != Path(selected["checkpoints"][index]).resolve():
                raise AssertionError("B does not point at the selected raw-channel direct NLinear")
            if source["direct_checkpoint_sha256"] != selected["checkpoint_sha256"][index]:
                raise AssertionError("B's S receipt differs from direct selection")
            if not selected["run_ids"][index].endswith(str(seed)):
                raise AssertionError("Selected checkpoint order is not the prespecified seed order")
            rows.append({"dataset": dataset, "seed": seed, "direct_run": selected["run_ids"][index],
                         "parent_run": parent["id"], "sources_match": True})
    selected_path = HERE / "selected.json"
    final_status = "not_yet_available; final selection checks occur before Gamma/TEST"
    if selected_path.exists():
        selected = read_json(selected_path)
        if selected.get("test_used_for_selection") is not False:
            raise AssertionError("Final selection must explicitly exclude TEST")
        for dataset in DATASETS:
            for family in ("a_p_lora_qfixed", "full_lora_mse", "b_learned_u", "direct_nlinear"):
                item = selected["selected"][dataset][family]
                if any(len(item[key]) != 2 for key in ("run_ids", "checkpoints", "checkpoint_sha256")):
                    raise AssertionError("Incomplete final paired selection schema")
        final_status = "complete_schema_confirmed_without_loading_weights_or_arrays"
    return {"status": "passed", "rows": rows, "final_selection_schema": final_status,
            "protocol_sha256": digest(HERE / "protocol.json"),
            "reuse_manifest_sha256": digest(HERE / "reuse_manifest.json"),
            "selected_direct_sha256": digest(HERE / "selected_direct.json"),
            "weights_loaded": False, "dataset_arrays_loaded": False, "new_performance_scores": False}


def run_checks(job):
    checks = {}
    for label, check in (("syntax", check_syntax), ("weighted_gamma_objective", check_weighted_gamma_objective),
                         ("period_pooling", check_period_pooling), ("missing_projection_and_zero", check_missing_projection_and_zero),
                         ("paired_bootstrap", check_paired_bootstrap), ("json_sources", check_json_sources)):
        job.check_limits()
        checks[label] = check()
        job.heartbeat({"phase": "analysis_helper_checks", "completed": len(checks), "last_check": label})
    return {"schema": "v11_analysis_helper_checks_v1", "status": "passed", "checks": checks,
            "real_data_fits": 0, "synthetic_optimizer_updates": 0, "solver_calls": 0,
            "model_forwards": 0, "new_real_data_scores": 0,
            "scope": "Synthetic arithmetic and JSON/source integration only; not evidence of model improvement"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="analysis_helpers01")
    parser.add_argument("--output", type=Path, default=HERE / "analysis_checks01.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the prior check artifact; give a new output/name for a rerun")
    with Job(args.name, category="cpu_check", reserve_seconds=60,
             metadata={"scope": "AST + synthetic arithmetic + JSON schema", "optimizer_updates": 0,
                       "solver_calls": 0, "real_data_scores": 0}) as job:
        result = run_checks(job)
        save_json(args.output, result)
    print(str(args.output))


if __name__ == "__main__":
    main()
