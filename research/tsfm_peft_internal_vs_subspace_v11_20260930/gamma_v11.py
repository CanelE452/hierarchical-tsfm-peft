from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np

from runtime_v11 import CACHE, HERE, ROOT, Job, batches, digest, load_data, read_json, save_json, sources_for

DATASETS = ("robin", "jena", "hog")
SEEDS = (92601, 92602)
FIT_FAMILIES = ("a_p_lora_qfixed", "full_lora_mse", "b_learned_u", "direct_nlinear")
ATOL, RTOL = 1e-5, 1e-4


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def receipt(path: str | Path) -> dict[str, Any]:
    path = resolve(path)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def check_receipt(item: dict[str, Any]) -> Path:
    path = resolve(item["path"])
    if digest(path) != item["sha256"]:
        raise ValueError(f"Artifact changed: {path}")
    return path


def load_complete_selection() -> dict[str, Any]:
    selected = read_json(HERE / "selected.json")
    if selected.get("test_used_for_selection") is not False:
        raise ValueError("Selection must explicitly exclude TEST")
    for dataset in DATASETS:
        for family in FIT_FAMILIES:
            row = selected["selected"][dataset][family]
            for key in ("run_ids", "checkpoints", "checkpoint_sha256"):
                if len(row[key]) != 2:
                    raise ValueError(f"Incomplete two-seed selection: {dataset}/{family}/{key}")
            for path, expected in zip(row["checkpoints"], row["checkpoint_sha256"]):
                if digest(resolve(path)) != expected:
                    raise ValueError(f"Selected checkpoint changed: {path}")
    return selected


def selected_checkpoint(selection: dict[str, Any], dataset: str, family: str, seed: int) -> dict[str, Any]:
    row = selection["selected"][dataset][family]
    index = SEEDS.index(seed)
    return {"run_id": row["run_ids"][index], "path": str(resolve(row["checkpoints"][index])),
            "sha256": row["checkpoint_sha256"][index], "lr": row["lr"]}


def model_for_u(selection: dict[str, Any], dataset: str, seed: int, kind: str,
                basis: np.ndarray, device: str):
    from model_v11 import build_model, restore_model

    if kind == "learned":
        item = selected_checkpoint(selection, dataset, "b_learned_u", seed)
        return restore_model(item["path"], device=device).eval(), item
    source = sources_for(dataset, seed)
    spec = {"dataset": dataset, "family": "b_fixed_u", "seed": seed}
    model = build_model(spec, basis, device=device, sources=source).eval()
    return model, {"kind": "fixed_train_pca", "sources": source}


def parity(left: np.ndarray, right: np.ndarray, atol=ATOL, rtol=RTOL) -> dict[str, Any]:
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Parity requires same-shaped finite arrays")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    violation = delta > atol + rtol * np.abs(right.astype(np.float64))
    return {"atol": atol, "rtol": rtol, "max_abs": float(delta.max(initial=0)),
            "max_rel": float((delta / np.maximum(np.abs(right), 1e-12)).max(initial=0)),
            "violations": int(violation.sum()), "elements": int(delta.size),
            "pass": bool(not violation.any())}


def configure() -> None:
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def release_gpu() -> None:
    import torch
    gc.collect()
    torch.cuda.synchronize()
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()


def collect_features(job: Job) -> dict[str, Any]:
    import torch

    selection = load_complete_selection()
    selected_receipt = receipt(HERE / "selected.json")
    destination = HERE / "gamma_features.json"
    if destination.exists():
        raise FileExistsError(f"Do not overwrite an existing feature manifest: {destination}")
    folder = CACHE / "gamma_features"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        data = load_data(dataset, include_test=False)
        origins = np.asarray(data["val_origins"], dtype=np.int64)
        for kind in ("fixed", "learned"):
            for seed in SEEDS:
                identifier = f"v11_{dataset}_{kind}_gamma_{seed}"
                path = folder / f"{identifier}.npz"
                if path.exists():
                    raise FileExistsError(f"Existing partial feature file needs explicit recovery: {path}")
                model, model_source = model_for_u(selection, dataset, seed, kind, data["basis"], "cuda")
                pieces = {key: [] for key in ("S", "latent_delta", "target", "mask")}
                with torch.inference_mode():
                    for batch_origins, x, y, mask in batches(data, origins, batch_size=4, device="cuda"):
                        job.check_limits()
                        features = model.mixture_features(x)
                        pieces["S"].append(features["S"].float().cpu().numpy())
                        pieces["latent_delta"].append(features["latent_delta"].float().cpu().numpy())
                        pieces["target"].append(y.float().cpu().numpy())
                        pieces["mask"].append(mask.bool().cpu().numpy())
                        del features, x, y, mask
                    basis = model.U.detach().float().cpu().numpy().copy()
                arrays = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
                np.savez_compressed(path, **arrays, U=basis, origins=origins)
                row = {"id": identifier, "dataset": dataset, "seed": seed, "kind": kind,
                       "features": receipt(path), "model_source": model_source,
                       "S_checkpoint": selected_checkpoint(selection, dataset, "direct_nlinear", seed),
                       "selected": selected_receipt, "source_split": "VAL", "test_used": False,
                       "shape": list(arrays["S"].shape), "latent": int(basis.shape[1])}
                rows.append(row)
                save_json(HERE / "gamma_features_partial.json", {"status": "collecting", "rows": rows,
                          "selected": selected_receipt, "test_used": False})
                job.heartbeat({"phase": "gamma_val_features", "completed": len(rows), "total": 12,
                               "last_id": identifier})
                del model, arrays, pieces, basis
                release_gpu()
    output = {"schema": "v11_gamma_features_v1", "status": "complete", "selected": selected_receipt,
              "test_used": False, "rows": rows, "feature_count": len(rows)}
    if len(rows) != 12:
        raise ValueError("Gamma requires all twelve preselected feature sets")
    save_json(destination, output)
    return output


def weighted_problem(S: np.ndarray, latent_delta: np.ndarray, U: np.ndarray,
                     target: np.ndarray, mask: np.ndarray):
    mask = mask.astype(bool)
    if S.shape != target.shape or mask.shape != target.shape or latent_delta.shape[:2] != S.shape[:2]:
        raise ValueError("Gamma feature/target/mask axes differ")
    channels, latent = U.shape
    if S.shape[-1] != channels or latent_delta.shape[-1] != latent:
        raise ValueError("Gamma basis shape differs from stored features")
    if not np.isfinite(S).all() or not np.isfinite(latent_delta).all() or not np.isfinite(U).all():
        raise ValueError("Nonfinite frozen Gamma features")
    if not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite observed target")
    counts = mask.sum(axis=(0, 1)).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError("VAL must contain observed targets for every fixed channel")
    origin, horizon, channel = np.nonzero(mask)
    design = latent_delta[origin, horizon].astype(np.float64) * U[channel].astype(np.float64)
    response = target[mask].astype(np.float64) - S[mask].astype(np.float64)
    sqrt_weight = np.sqrt(1.0 / (channels * counts[channel]))
    return design * sqrt_weight[:, None], response * sqrt_weight, counts


def direct_objective(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    count = mask.sum(axis=(0, 1))
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    return float(np.mean(np.sum(error * error, axis=(0, 1)) / count))


def solve_features(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import lsq_linear

    S, z, U = arrays["S"], arrays["latent_delta"], arrays["U"]
    target, mask = arrays["target"], arrays["mask"].astype(bool)
    matrix, response, counts = weighted_problem(S, z, U, target, mask)
    result = lsq_linear(matrix, response, bounds=(0.0, 1.0), method="bvls", lsq_solver="exact",
                        tol=1e-10, max_iter=100)
    gamma = np.asarray(result.x, dtype=np.float64)
    if np.any((gamma > 0.0) & (gamma.astype(np.float32) == 0.0)):
        raise ValueError("A positive solver coefficient underflows to FP32 zero; do not silently prune it")
    residual = matrix @ gamma - response
    gradient = matrix.T @ residual
    projected_gradient = gradient.copy()
    projected_gradient[gamma == 0.0] = np.minimum(projected_gradient[gamma == 0.0], 0.0)
    projected_gradient[gamma == 1.0] = np.maximum(projected_gradient[gamma == 1.0], 0.0)
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank_tol = singular.max(initial=0.0) * max(matrix.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(singular > rank_tol))
    prediction64 = S.astype(np.float64) + (z.astype(np.float64) * gamma) @ U.astype(np.float64).T
    prediction32 = S + (z * gamma.astype(np.float32)) @ U.T
    objective = float(residual @ residual)
    direct = direct_objective(prediction64, target, mask)
    identity = direct_objective(S.astype(np.float64) + z.astype(np.float64) @ U.astype(np.float64).T,
                                target, mask)
    zero = direct_objective(S, target, mask)
    objective_parity = abs(objective - direct) <= 1e-10 * max(1.0, abs(direct))
    stats = {"solver": {"method": "bvls", "lsq_solver": "exact", "bounds": [0, 1], "tol": 1e-10,
                        "max_iter": 100, "success": bool(result.success), "status": int(result.status),
                        "message": str(result.message), "iterations": int(result.nit),
                        "optimality": float(result.optimality), "scipy_half_sse_cost": float(result.cost)},
             "gamma_fp64": gamma.tolist(), "gamma_fp32": gamma.astype(np.float32).tolist(),
             "active_mask_solver": result.active_mask.astype(int).tolist(),
             "exact_zero_indices": np.flatnonzero(gamma == 0.0).tolist(),
             "active_K": int(np.count_nonzero(gamma != 0.0)),
             "rank": rank, "rank_deficient": rank < matrix.shape[1], "singular_values": singular.tolist(),
             "weighted_rows": int(len(response)), "observed_count_by_channel": counts.tolist(),
             "max_projected_gradient": float(np.abs(projected_gradient).max(initial=0.0)),
             "gradient": gradient.tolist(), "weighted_objective": objective,
             "direct_macro_mse_fp64": direct, "direct_macro_mse_fp32": direct_objective(prediction32, target, mask),
             "gamma_zero_mse": zero, "gamma_identity_mse": identity,
             "objective_definition": "sum squared weighted residuals, twice scipy.cost; channel-macro MSE",
             "objective_parity": objective_parity,
             "fp64_vs_fp32_feature_forecast": parity(prediction32, prediction64),
             "in_bounds": bool(np.all((gamma >= 0.0) & (gamma <= 1.0))),
             "finite": bool(np.isfinite(gamma).all()), "source_split": "VAL", "test_used": False}
    stats["valid"] = bool(result.success and stats["in_bounds"] and stats["finite"] and objective_parity
                          and stats["fp64_vs_fp32_feature_forecast"]["pass"]
                          and objective <= min(zero, identity) + 1e-10 * max(1.0, zero, identity))
    return gamma, stats


def fit_gamma(job: Job) -> dict[str, Any]:
    load_complete_selection()
    manifest = read_json(HERE / "gamma_features.json")
    check_receipt(manifest["selected"])
    if manifest["status"] != "complete" or len(manifest["rows"]) != 12:
        raise ValueError("Complete frozen twelve-way VAL features required")
    destination = HERE / "gamma_selection.json"
    if destination.exists():
        raise FileExistsError("Gamma is already selected; do not silently refit")
    folder = HERE / "gamma"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in manifest["rows"]:
        result_path = folder / f"{row['id']}.json"
        if result_path.exists():
            existing = read_json(result_path)
            if existing["status"] != "complete" or existing["feature_receipt"] != row["features"]:
                raise ValueError(f"Failed/changed Gamma attempt requires explicit recovery: {result_path}")
            existing["result"] = receipt(result_path)
            rows.append(existing)
            continue
        path = check_receipt(row["features"])
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        metadata = {"dataset": row["dataset"], "fit_id": row["id"], "seed": row["seed"],
                    "family": f"b_{row['kind']}_gamma", "source_split": "VAL", "feature_receipt": row["features"]}
        with Job(row["id"], category="coefficient_fit", reserve_seconds=60, metadata=metadata) as attempt:
            gamma, diagnostics = solve_features(arrays)
            output = {**row, "status": "complete" if diagnostics["valid"] else "invalid",
                      "feature_receipt": row["features"], "gamma": gamma.tolist(), "diagnostics": diagnostics,
                      "selection_rule": "one bounded VAL fit after frozen S/LR/U checkpoint selection"}
            save_json(result_path, output)
            attempt.heartbeat({"phase": "solver_complete", "valid": diagnostics["valid"]})
            if not diagnostics["valid"]:
                raise RuntimeError(f"Gamma solver/numerical validity failed; evidence saved: {result_path}")
        output["result"] = receipt(result_path)
        rows.append(output)
        save_json(HERE / "gamma_selection_partial.json", {"status": "fitting", "rows": rows})
        job.heartbeat({"phase": "gamma_fits", "completed": len(rows), "total": 12})
        del arrays
    output = {"schema": "v11_gamma_selection_v1", "status": "complete", "rows": rows,
              "selected": receipt(HERE / "selected.json"), "features": receipt(HERE / "gamma_features.json"),
              "test_used_for_selection": False, "coefficient_fit_count": 12,
              "selection_limit": "S/LR/U/Gamma repeatedly use VAL; no independent validation claim"}
    save_json(destination, output)
    return output


def verify_gamma(job: Job) -> dict[str, Any]:
    import torch
    from model_v11 import deployment_copy

    selection = load_complete_selection()
    gamma_selected = read_json(HERE / "gamma_selection.json")
    check_receipt(gamma_selected["selected"])
    destination = HERE / "gamma_verification.json"
    if destination.exists():
        raise FileExistsError("Preserve prior Gamma verification; explicit correction name required")
    rows = []
    for row in gamma_selected["rows"]:
        dataset, seed, kind = row["dataset"], int(row["seed"]), row["kind"]
        data = load_data(dataset, include_test=False)
        model, _ = model_for_u(selection, dataset, seed, kind, data["basis"], "cuda")
        gamma = np.asarray(row["gamma"], dtype=np.float32)
        model.set_gamma(torch.as_tensor(gamma, device="cuda"))
        with np.load(check_receipt(row["features"]), allow_pickle=False) as source:
            origins = source["origins"].copy()
            reference = source["S"].astype(np.float64) + (
                source["latent_delta"].astype(np.float64) * np.asarray(row["gamma"])) @ source["U"].astype(np.float64).T
        outputs = []
        with torch.inference_mode():
            for _, x, y, mask in batches(data, origins, batch_size=4, device="cuda"):
                job.check_limits()
                outputs.append(model(x).float().cpu().numpy())
                del x, y, mask
        actual = np.concatenate(outputs)
        del outputs
        deploy = deployment_copy(model)
        deploy.prune_zero_gamma = True
        pruned = []
        with torch.inference_mode():
            for _, x, y, mask in batches(data, origins, batch_size=4, device="cuda"):
                job.check_limits()
                pruned.append(deploy(x).float().cpu().numpy())
                del x, y, mask
        stats = {"id": row["id"], "feature_fp64_vs_model_fp32": parity(actual, reference),
                 "full_vs_export_pruned": parity(np.concatenate(pruned), actual),
                 "exact_zero_indices": np.flatnonzero(gamma == 0.0).tolist(),
                 "active_K": int(np.count_nonzero(gamma)), "source_split": "VAL"}
        stats["pass"] = stats["feature_fp64_vs_model_fp32"]["pass"] and stats["full_vs_export_pruned"]["pass"]
        rows.append(stats)
        save_json(HERE / "gamma_verification_partial.json", {"rows": rows, "status": "verifying"})
        if not stats["pass"]:
            raise RuntimeError(f"Gamma deployment/precision parity failed: {row['id']}")
        job.heartbeat({"phase": "gamma_verify", "completed": len(rows), "total": 12})
        del model, deploy, data, pruned, actual, reference
        release_gpu()
    output = {"status": "complete", "pass": all(row["pass"] for row in rows), "rows": rows,
              "gamma_selection": receipt(HERE / "gamma_selection.json"), "test_used": False}
    save_json(destination, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("collect", "fit", "verify"))
    parser.add_argument("--reserve-seconds", type=float, required=True)
    parser.add_argument("--name", help="Fresh ledger job label for an explicitly documented retry")
    args = parser.parse_args()
    configure()
    category = "cpu_analysis" if args.action == "fit" else "gpu"
    with Job(args.name or f"gamma_{args.action}", category=category, reserve_seconds=args.reserve_seconds,
             metadata={"action": args.action, "datasets": list(DATASETS), "test_used": False}) as job:
        {"collect": collect_features, "fit": fit_gamma, "verify": verify_gamma}[args.action](job)


if __name__ == "__main__":
    main()
