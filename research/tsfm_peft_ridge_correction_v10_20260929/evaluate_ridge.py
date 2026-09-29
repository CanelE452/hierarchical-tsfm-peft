"""Evaluate fixed v10 ridge selections using saved parents and CPU corrections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from runtime_ridge import CACHE, HERE, ROOT, V9, Job, digest, load_data, save_json

PERIODS = ("test_a", "test_b")
SEEDS = (92601, 92602)
FAMILIES = ("ridge_p", "ridge_raw")
PRIOR_ROLES = ("level_res", "staged_p", "staged_raw", "staged_p_alpha", "staged_raw_alpha",
               "pq_split", "raw64", "f0", "lora", "direct_nlinear")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def receipt(path):
    path = resolve(path)
    return {"path": str(path), "sha256": digest(path)}


def checked(item):
    path = resolve(item["path"])
    if digest(path) != item["sha256"]:
        raise ValueError(f"Changed input: {path}")
    return path


def load_prediction(item, origins):
    path = checked(item)
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["origins"], origins):
            raise ValueError(f"Prediction origins differ: {path}")
        prediction = archive["prediction"].astype(np.float32)
    return prediction


def error_terms(prediction, target, observed):
    # The v9 error_terms/scores_from_terms contract is retained, including bias.
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("Prediction/target/mask axes differ")
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError("Nonfinite predictions or observed targets")
    delta = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0.)
    return {"squared": (delta * delta).sum(axis=1), "absolute": np.abs(delta).sum(axis=1),
            "signed": delta.sum(axis=1), "count": observed.sum(axis=1)}


def scores_from_terms(terms, index=None):
    if index is None:
        index = np.arange(len(terms["count"]))
    count = terms["count"][index].sum(axis=0)
    if np.any(count <= 0):
        raise ValueError("A fixed channel has no observed targets; do not drop it")
    squared = terms["squared"][index].sum(axis=0)
    absolute = terms["absolute"][index].sum(axis=0)
    signed = terms["signed"][index].sum(axis=0)
    return {"mse": float(np.mean(squared / count)), "mae": float(np.mean(absolute / count)),
            "mean_signed_error": float(np.mean(signed / count)),
            "channel_mse": (squared / count).tolist(), "channel_mae": (absolute / count).tolist(),
            "channel_mean_signed_error": (signed / count).tolist(),
            "channel_target_count": count.astype(int).tolist(),
            "channel_squared_error_sum": squared.tolist(),
            "channel_absolute_error_sum": absolute.tolist(), "channel_signed_error_sum": signed.tolist()}


def complete_space_scores(prediction, target, observed, basis, index):
    complete = observed[index].all(axis=2)
    coverage = {"complete_target_rows": int(complete.sum()), "total_target_rows": int(complete.size),
                "scope": "Complete target rows only; missing targets are never filled for projection."}
    if not complete.any():
        return dict(coverage, status="unavailable")
    error = prediction[index][complete].astype(np.float64) - target[index][complete].astype(np.float64)
    retained = (error @ basis) @ basis.T
    residual = error - retained
    return dict(coverage, status="available", p_retained_space_mse=float(np.mean(retained ** 2)),
                q_residual_space_mse=float(np.mean(residual ** 2)),
                p_retained_space_mae=float(np.mean(np.abs(retained))),
                q_residual_space_mae=float(np.mean(np.abs(residual))))


def check_seal():
    seal_path = HERE / "selection_seal.json"
    exposure_path = HERE / "test_exposure.json"
    seal = read_json(seal_path)
    selected = read_json(HERE / "selected.json")
    if seal["selected_sha256"] != digest(HERE / "selected.json"):
        raise ValueError("Selected models changed after seal")
    if seal["reuse_manifest_sha256"] != digest(HERE / "reuse_manifest.json"):
        raise ValueError("Reuse manifest changed after seal")
    if set(selected["datasets"]) != {"robin", "jena"}:
        raise ValueError("Both dataset selections must precede either evaluation")
    exposure = read_json(exposure_path)
    if exposure["seal_sha256"] != digest(seal_path) or set(exposure["datasets"]) != {"robin", "jena"}:
        raise ValueError("TEST exposure is not bound to both sealed selections")
    if exposure["first_opened_utc"] < seal["created_utc"]:
        raise ValueError("TEST was opened before global selection seal")
    for dataset, families in selected["datasets"].items():
        if set(families) != set(FAMILIES):
            raise ValueError("Only the approved two ridge families may be selected")
        for family, choice in families.items():
            if float(choice["penalty"]) not in (.001, .1):
                raise ValueError("Unapproved ridge penalty")
            runs = choice["runs"]
            if sorted(int(run["spec"]["seed"]) for run in runs) != list(SEEDS):
                raise ValueError("Each selected arm requires both parent seeds")
            for run in runs:
                spec = run["spec"]
                if spec["dataset"] != dataset or spec["family"] != family or float(spec["penalty"]) != float(choice["penalty"]):
                    raise ValueError("Selected run and family configuration differ")
                checked({"path": run["checkpoint"], "sha256": run["checkpoint_sha256"]})
    return selected, {"seal": receipt(seal_path), "exposure": receipt(exposure_path),
                      "selection": receipt(HERE / "selected.json"),
                      "reuse_manifest": receipt(HERE / "reuse_manifest.json")}


def ridge_prediction(run, parent, parent_record, data, origins, job):
    path = checked({"path": run["checkpoint"], "sha256": run["checkpoint_sha256"]})
    saved = torch.load(path, map_location="cpu", weights_only=False)
    spec = run["spec"]
    for key in ("dataset", "seed", "family", "penalty"):
        if saved[key] != spec[key]:
            raise ValueError(f"Saved ridge {key} differs from selected spec")
    if saved["parent_receipt"] != parent_record:
        raise ValueError("Ridge checkpoint refers to another parent")
    basis = torch.as_tensor(saved["basis"], dtype=torch.float32, device="cpu")
    expected_basis = torch.as_tensor(data["basis"], dtype=torch.float32, device="cpu")
    if not torch.equal(basis, expected_basis):
        raise ValueError("Checkpoint basis differs from fixed TRAIN PCA")
    weight = torch.as_tensor(saved["weight"], dtype=torch.float32, device="cpu")
    if tuple(weight.shape) != (48, 512) or not torch.isfinite(weight).all():
        raise ValueError("Invalid full temporal map")
    prediction = np.empty_like(parent, dtype=np.float32)
    correction_q_max = 0.
    parent_q_max = 0.
    q_violations = 0
    with torch.no_grad():
        for start in range(0, len(origins), 4):
            job.check_limits()
            batch = origins[start:start + 4]
            x = torch.from_numpy(np.stack([data["x"][o - 512:o] for o in batch]).astype(np.float32))
            centered = x - x[:, -1:, :]
            if spec["family"] == "ridge_p":
                centered = torch.nn.functional.linear(torch.nn.functional.linear(centered, basis.T), basis)
            correction = torch.nn.functional.linear(centered.transpose(1, 2), weight).transpose(1, 2)
            base = torch.from_numpy(parent[start:start + len(batch)])
            full = base + correction
            prediction[start:start + len(batch)] = full.numpy()
            if spec["family"] == "ridge_p":
                c_q = correction - torch.nn.functional.linear(torch.nn.functional.linear(correction, basis.T), basis)
                base_q = base - torch.nn.functional.linear(torch.nn.functional.linear(base, basis.T), basis)
                full_q = full - torch.nn.functional.linear(torch.nn.functional.linear(full, basis.T), basis)
                delta = (full_q - base_q).abs()
                correction_q_max = max(correction_q_max, float(c_q.abs().max()))
                parent_q_max = max(parent_q_max, float(delta.max()))
                q_violations += int((delta > 1e-5 + 1e-4 * base_q.abs()).sum())
    q_check = {"expected": spec["family"] == "ridge_p", "atol": 1e-5, "rtol": 1e-4,
               "correction_q_max_abs": correction_q_max if spec["family"] == "ridge_p" else None,
               "prediction_q_difference_max_abs": parent_q_max if spec["family"] == "ridge_p" else None,
               "violating_elements": q_violations if spec["family"] == "ridge_p" else None,
               "pass": q_violations == 0 if spec["family"] == "ridge_p" else None}
    if spec["family"] == "ridge_p" and not q_check["pass"]:
        raise ValueError("Ridge P violated fixed Q-preservation tolerance")
    return prediction, q_check


def paired_ids(ids, models):
    if len(ids) == 1:
        return ids * 2
    by_seed = {int(models[identifier]["spec"]["seed"]): identifier for identifier in ids}
    if set(by_seed) != set(SEEDS):
        raise ValueError("Comparison is neither seed-paired nor a deterministic singleton")
    return [by_seed[seed] for seed in SEEDS]


def bootstrap_weights(indices, dataset, job):
    block = 7 if dataset == "robin" else 42
    output = {}
    for label_index, label in enumerate((*PERIODS, "combined")):
        strata = [indices[label]] if label != "combined" else [indices[p] for p in PERIODS]
        rng = np.random.default_rng(np.random.SeedSequence([9262026, label_index]))
        weights = np.zeros((2000, len(indices["combined"])), dtype=np.float64)
        for draw in range(2000):
            if draw % 250 == 0:
                job.check_limits()
            for stratum in strata:
                n = len(stratum)
                if n < block:
                    raise ValueError("Period shorter than fixed bootstrap block")
                starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
                selected = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
                weights[draw] += np.bincount(stratum[selected], minlength=weights.shape[1])
        output[label] = weights
    return output


def comparison(left_role, right_role, roles, models, terms, indices, weights, dataset):
    left_ids = paired_ids(roles[left_role], models)
    right_ids = paired_ids(roles[right_role], models)
    count = terms[left_ids[0]]["count"]
    if any(not np.array_equal(terms[i]["count"], count) for i in left_ids + right_ids):
        raise ValueError("Paired models do not share target masks")
    left = {key: np.mean([terms[i][key] for i in left_ids], axis=0) for key in ("squared", "absolute")}
    right = {key: np.mean([terms[i][key] for i in right_ids], axis=0) for key in ("squared", "absolute")}
    output = {"name": f"{dataset}_{left_role}_vs_{right_role}", "left_role": left_role,
              "right_role": right_role, "left_seed_ids": left_ids, "right_seed_ids": right_ids,
              "direction": "left minus reference; negative favors left", "periods": {}}
    for label, index in indices.items():
        denominator = count[index].sum(axis=0)
        full = {}
        for metric, key in (("mse", "squared"), ("mae", "absolute")):
            left_channel = left[key][index].sum(axis=0) / denominator
            right_channel = right[key][index].sum(axis=0) / denominator
            lhs, rhs = float(left_channel.mean()), float(right_channel.mean())
            full.update({"left_" + metric: lhs, "reference_" + metric: rhs, "delta_" + metric: lhs - rhs,
                         "channel_delta_" + metric: (left_channel - right_channel).tolist(),
                         "relative_" + metric + "_pct": 100 * (lhs - rhs) / rhs if rhs else None})
        full["relative_to_reference_pct"] = full["relative_mse_pct"]
        denominator_boot = weights[label] @ count
        valid = (denominator_boot > 0).all(axis=1)
        sampled_left = np.mean((weights[label][valid] @ left["squared"]) / denominator_boot[valid], axis=1)
        sampled_right = np.mean((weights[label][valid] @ right["squared"]) / denominator_boot[valid], axis=1)
        valid_rhs = sampled_right > 0
        interval = None
        if valid.all() and valid_rhs.all():
            interval = {"delta_mse": np.quantile(sampled_left - sampled_right, [.025, .975]).tolist(),
                        "relative_to_reference_pct": np.quantile(100 * (sampled_left - sampled_right) / sampled_right, [.025, .975]).tolist()}
        paired = []
        for seed, left_id, right_id in zip(SEEDS, left_ids, right_ids):
            lhs = models[left_id]["scores"][label]
            rhs = models[right_id]["scores"][label]
            paired.append({"seed": seed, "left_id": left_id, "right_id": right_id,
                           "left_mse": lhs["mse"], "reference_mse": rhs["mse"],
                           "delta_mse": lhs["mse"] - rhs["mse"], "delta_mae": lhs["mae"] - rhs["mae"],
                           "relative_to_reference_pct": 100 * (lhs["mse"] - rhs["mse"]) / rhs["mse"] if rhs["mse"] else None})
        output["periods"][label] = {"full": full, "paired_seed_deltas": paired, "conditional_block95": interval,
                                     "draws": 2000, "unavailable_draws": int((~valid).sum() + (~valid_rhs).sum()),
                                     "block_length_origins": 7 if dataset == "robin" else 42,
                                     "strata": list(PERIODS) if label == "combined" else [label]}
    output["interval_scope"] = "Non-circular blocks within periods; all channels/models share resamples; fixed parent seeds. Excludes fitting and selection uncertainty."
    return output


def evaluate(dataset, name, job, reuse_partial=None):
    selected, binding = check_seal()
    reused = read_json(resolve(reuse_partial)) if reuse_partial else None
    if reused and (reused['dataset'] != dataset or reused['binding'] != binding):
        raise ValueError('Partial output belongs to another dataset or selection')
    target_path = HERE / f"{name}.json"
    partial_path = HERE / f"{name}_partial.json"
    prediction_dir = CACHE / f"{name}_predictions"
    if target_path.exists() or partial_path.exists() or prediction_dir.exists():
        raise FileExistsError("Preserve prior evaluation and use a new name")
    data = load_data(dataset, include_test=True)
    origins_by_period = {p: np.asarray(data[p + "_origins"], dtype=np.int64) for p in PERIODS}
    origins = np.concatenate(list(origins_by_period.values()))
    count_a = len(origins_by_period["test_a"])
    indices = {"test_a": np.arange(count_a), "test_b": np.arange(count_a, len(origins)),
               "combined": np.arange(len(origins))}
    target = np.stack([data["x"][o:o + 48] for o in origins]).astype(np.float32)
    mask = np.stack([data["finite"][o:o + 48] for o in origins]).astype(bool)
    basis = np.asarray(data["basis"], dtype=np.float64)
    prior_path = V9 / f"{dataset}_eval01.json"
    prior = read_json(prior_path)
    if prior["status"] != "complete" or prior["dataset"] != dataset:
        raise ValueError("Prior saved evaluation is incomplete or belongs to another dataset")
    for period in PERIODS:
        if not np.array_equal(origins_by_period[period], prior["origins"][period]):
            raise ValueError("v10 origins differ from prior fixed evaluation")
    result = {"schema": "ridge_correction_v10_evaluation_v1", "status": "running", "dataset": dataset,
              "binding": binding, "source": receipt(__file__), "prior_evaluation": receipt(prior_path),
              "online_model_source": receipt(HERE / "model_ridge.py"),
              "score_provenance": {"adapted_from": receipt(V9 / "evaluate_v9.py"),
                                   "changes": "Same masked macro terms; vectorized period-stratified bootstrap with common fixed resamples."},
              "channels": np.asarray(data["columns"]).astype(str).tolist(),
              "origins": {p: origins_by_period[p].tolist() for p in PERIODS},
              "roles": {}, "models": {}, "role_scores": {}, "role_space_scores": {}, "comparisons": [],
              "test_scope": "Previously exposed development evaluation; no independent confirmation.",
              "new_fits": 0, "new_tsfm_test_forward": 0, "ensemble": False}
    prediction_dir.mkdir(parents=True, exist_ok=False)
    save_json(partial_path, result)
    terms, parent_arrays = {}, {}

    def add(identifier, role, spec, prediction, pred_receipt, kind, extra=None):
        terms[identifier] = error_terms(prediction, target, mask)
        scores = {period: scores_from_terms(terms[identifier], index) for period, index in indices.items()}
        spaces = {period: complete_space_scores(prediction, target, mask, basis, index) for period, index in indices.items()}
        result["models"][identifier] = {"spec": spec, "role": role, "kind": kind, "prediction": pred_receipt,
                                       "scores": scores, "space_scores": spaces, **(extra or {})}
        result["roles"].setdefault(role, []).append(identifier)

    for role in PRIOR_ROLES:
        if role not in prior["roles"]:
            raise ValueError(f"Required prior comparison missing: {role}")
        for identifier in prior["roles"][role]:
            job.check_limits()
            row = prior["models"][identifier]
            prediction = load_prediction(row["prediction"], origins)
            add(identifier, role, row["spec"], prediction, row["prediction"], "stored_prediction")
            for period in indices:
                for metric in ("mse", "mae"):
                    if abs(result["models"][identifier]["scores"][period][metric] - row["scores"][period][metric]) > 1e-10:
                        raise ValueError(f"Prior score replay differs: {identifier}/{period}/{metric}")
            if role == "level_res":
                parent_arrays[int(row["spec"]["seed"])] = (identifier, prediction, row["prediction"])
    reuse = read_json(HERE / "reuse_manifest.json")
    for family in FAMILIES:
        for run in sorted(selected["datasets"][dataset][family]["runs"], key=lambda r: r["spec"]["seed"]):
            spec = run["spec"]
            identifier = spec["id"]
            parent_record = reuse["parents"][dataset][str(spec["seed"])]
            parent_id, parent, parent_prediction_receipt = parent_arrays[int(spec["seed"])]
            if parent_record["id"] != parent_id or parent_record["test_prediction"] != parent_prediction_receipt:
                raise ValueError("Saved TEST parent does not match selected checkpoint parent seed")
            if reused:
                old = reused['models'][identifier]
                if old['spec'] != spec or old['checkpoint']['sha256'] != run['checkpoint_sha256'] or old['parent_prediction'] != parent_prediction_receipt:
                    raise ValueError('Partial prediction dependency mismatch')
                prediction = load_prediction(old['prediction'], origins)
                q_check, prediction_receipt = old['q_prediction_parity'], old['prediction']
            else:
                prediction, q_check = ridge_prediction(run, parent, parent_record, data, origins, job)
                prediction_path = prediction_dir / f"{identifier}.npz"
                np.savez_compressed(prediction_path, prediction=prediction, origins=origins)
                prediction_receipt = receipt(prediction_path)
            add(identifier, family, spec, prediction, prediction_receipt, "cpu_ridge_plus_saved_parent",
                {"checkpoint": receipt(run["checkpoint"]), "parent_id": parent_id,
                 "parent_prediction": parent_prediction_receipt, "q_prediction_parity": q_check,
                 "online_gpu_parity": {"status": "requires_separate_verification",
                                       "scope": "Cached-parent CPU evaluation does not itself establish online GPU parity or cost."}})
            save_json(partial_path, result)
            job.heartbeat(f"{dataset}: evaluated {identifier}")
    for role, ids in result["roles"].items():
        result["role_scores"][role] = {}
        result["role_space_scores"][role] = {}
        for period in indices:
            scores = [result["models"][i]["scores"][period] for i in ids]
            averaged = {key: np.mean([s[key] for s in scores], axis=0).tolist()
                        for key in ("mse", "mae", "mean_signed_error", "channel_mse", "channel_mae", "channel_mean_signed_error")}
            result["role_scores"][role][period] = dict(averaged, run_ids=ids, count=len(ids),
                                                       scope="Mean individual parent-seed losses, not forecast ensemble.")
            spaces = [result["models"][i]["space_scores"][period] for i in ids]
            result["role_space_scores"][role][period] = {
                key: float(np.mean([s[key] for s in spaces])) if all(s["status"] == "available" for s in spaces) else None
                for key in ("p_retained_space_mse", "q_residual_space_mse", "p_retained_space_mae", "q_residual_space_mae")}
            result["role_space_scores"][role][period].update(complete_target_rows=spaces[0]["complete_target_rows"],
                                                            total_target_rows=spaces[0]["total_target_rows"])
    weights = bootstrap_weights(indices, dataset, job)
    pairs = [("ridge_p", reference) for reference in ("ridge_raw", "level_res", "staged_p", "staged_p_alpha", "pq_split", "direct_nlinear", "f0", "lora")]
    pairs += [("ridge_raw", reference) for reference in ("level_res", "staged_raw", "staged_raw_alpha", "raw64", "direct_nlinear", "f0", "lora")]
    for left, right in pairs:
        job.check_limits()
        result["comparisons"].append(comparison(left, right, result["roles"], result["models"], terms, indices, weights, dataset))
        job.heartbeat(f"{dataset}: compared {left} vs {right}")
    _, after_binding = check_seal()
    if after_binding != binding:
        raise ValueError("Selection or exposure binding changed during evaluation")
    result.update(status="complete", completed_utc=time.time(), test_reselection=False,
                  new_cpu_correction_predictions=0 if reused else 4, reused_partial=receipt(reuse_partial) if reused else None,
                  bootstrap={"draws": 2000, "seed": 9262026,
                  "block_length_origins": 7 if dataset == "robin" else 42, "boundaries_crossed": False},
                  combined_definition="Within each parent seed/channel, sum period SSE/count before channel mean; then average seed losses.")
    save_json(target_path, result)
    return {"result": receipt(target_path), "models": len(result["models"]), "comparisons": len(result["comparisons"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("robin", "jena"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reserve-seconds", type=float, default=300.)
    parser.add_argument("--reuse-partial")
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        raise ValueError("Use a unique alphanumeric/underscore/hyphen name")
    with Job(args.name, category="cpu_analysis", reserve_seconds=args.reserve_seconds,
             metadata={"dataset": args.dataset, "action": "sealed TEST evaluation", "new_fit": False}) as job:
        global np, torch
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            output = evaluate(args.dataset, args.name, job, args.reuse_partial)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
