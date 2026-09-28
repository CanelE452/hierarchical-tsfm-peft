from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

from runtime_v6 import (
    CACHE,
    HERE,
    ROOT,
    Job,
    digest,
    environment_receipt,
    load_data,
    save_json,
    source_receipt,
)


SEEDS = (92601, 92602)
PERIODS = ("test_a", "test_b")
CHANNELS = 17
CONTEXT = 512
HORIZON = 48
NEURAL_FAMILIES = ("level_res", "old_res", "level_raw", "level_linear_res", "lora")
DETERMINISTIC_ROLES = ("f0", "compress", "level_only")
RIDGE_PENALTIES = (0.001, 0.1)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def receipt(path: str | Path) -> dict[str, str]:
    resolved = resolve(path)
    return {"path": str(resolved), "sha256": digest(resolved)}


def check_receipt(item: dict[str, Any]) -> Path:
    path = resolve(item["path"])
    if digest(path) != item["sha256"]:
        raise ValueError(f"Sealed input changed: {path}")
    return path


def load_module(path: Path, name: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def validate_name(name: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not name or any(c not in allowed for c in name):
        raise ValueError("Use a unique alphanumeric/underscore/hyphen run name")


def normal_family(family: str) -> str:
    aliases = {
        "tsfm_res": "old_res",
        "res": "old_res",
        "level_tsfm_res": "level_res",
        "tsfm_level_res": "level_res",
        "level_tsfm_raw": "level_raw",
        "tsfm_level_raw": "level_raw",
        "linear_res": "level_linear_res",
        "linear_level_res": "level_linear_res",
        "shared": "shared",
        "shared_linear": "shared",
    }
    return aliases.get(family, family)


def checkpoint_entry(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["kind"] = "checkpoint"
    result["checkpoint"] = str(resolve(row["checkpoint"]))
    check_receipt({"path": result["checkpoint"], "sha256": row["checkpoint_sha256"]})
    check_receipt(row["best_val_prediction"])
    if not math.isfinite(float(row["best_val_mse"])):
        raise ValueError(f"Nonfinite VAL loss: {row['id']}")
    result_path = check_receipt({"path": row["result_path"], "sha256": row["result_sha256"]})
    completed = read_json(result_path)
    if completed.get("status") != "complete":
        raise ValueError(f"Incomplete run cannot be sealed: {row['id']}")
    for key in ("id", "checkpoint_sha256", "best_val_prediction", "best_val_mse", "selected_epoch"):
        if completed[key] != row[key]:
            raise ValueError(f"Run receipt disagrees with completed result: {row['id']}/{key}")
    if completed["spec"] != row["spec"]:
        raise ValueError(f"Run receipt disagrees with completed result: {row['id']}/spec")
    result["spec"] = dict(result["spec"])
    result["spec"]["family"] = normal_family(result["spec"]["family"])
    return result


def collect_neural(selection: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    runs = [checkpoint_entry(row) for row in selection["all_neural_runs"]]
    by_family = {family: [] for family in NEURAL_FAMILIES}
    for row in runs:
        family = normal_family(row["spec"]["family"])
        if family not in by_family:
            raise ValueError(f"Unexpected v6 neural family in selection: {family}")
        if row["spec"].get("ed_mode") != "fixed_ed":
            raise ValueError(f"v6 neural comparison must be FIXED_ED: {row['id']}")
        by_family[family].append(row)
    for family, members in by_family.items():
        if len(members) != 2 or sorted(row["spec"]["seed"] for row in members) != list(SEEDS):
            raise ValueError(f"{family}: expected exactly paired seeds {SEEDS}")
        members.sort(key=lambda row: row["spec"]["seed"])
    if len(runs) != 10:
        raise ValueError("Exactly ten neural training results are required for v6 seal")
    return by_family


def deterministic_entry(selection: dict[str, Any], key: str) -> dict[str, Any]:
    row = selection["deterministic_models"][key]
    item = checkpoint_entry(row)
    item["kind"] = "checkpoint"
    item["spec"]["family"] = key
    item["spec"]["seed"] = None
    item["deterministic_reference"] = True
    return item


def validate_shared(shared: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    if shared["test_used_for_selection"] is not False:
        raise ValueError("SHARED selection must use VAL only")
    penalties = tuple(float(p) for p in protocol["ridge_penalties"])
    if penalties != RIDGE_PENALTIES:
        raise ValueError("Approved v6 ridge penalty order changed")
    order = {float(p): i for i, p in enumerate(penalties)}
    if len(shared["fits"]) != 2 or {float(row["penalty"]) for row in shared["fits"]} != set(order):
        raise ValueError("Both and only the two approved ridge penalties must complete before TEST")
    for fit in shared["fits"]:
        check_receipt(fit["result_file"])
        check_receipt(fit["weights"])
        check_receipt(fit["validation_predictions"])
        if not math.isfinite(float(fit["val_mse"])):
            raise ValueError(f"Nonfinite SHARED VAL loss: {fit['id']}")
    selected = min(shared["fits"], key=lambda row: (float(row["val_mse"]), order[float(row["penalty"])]))
    if selected["id"] != shared["selected_id"] or float(selected["penalty"]) != float(shared["selected_penalty"]):
        raise ValueError("SHARED selection differs from the approved VAL/tie rule")
    return {
        "id": selected["id"],
        "kind": "shared",
        "spec": {
            "family": "shared",
            "dataset": "robin",
            "seed": None,
            "ed_mode": None,
            "penalty": float(selected["penalty"]),
        },
        "weights": selected["weights"],
        "best_val_mse": float(selected["val_mse"]),
        "best_val_prediction": selected["validation_predictions"],
    }


def seal_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    destination = HERE / "evaluation_seal.json"
    if destination.exists() or (HERE / "test_exposure.json").exists():
        raise FileExistsError("Never overwrite a seal or retroactively reseal exposed TEST")
    protocol = read_json(HERE / "protocol.json")
    selection = read_json(args.selection)
    contract = read_json(args.data_contract)
    if int(protocol["channels"]) != CHANNELS or int(protocol["context"]) != CONTEXT or int(protocol["horizon"]) != HORIZON:
        raise ValueError("Protocol shape contract changed")
    if tuple(float(p) for p in protocol["ridge_penalties"]) != RIDGE_PENALTIES:
        raise ValueError("Protocol ridge penalty contract changed")

    by_family = collect_neural(selection)
    deterministic = {key: deterministic_entry(selection, key) for key in DETERMINISTIC_ROLES}
    shared = validate_shared(read_json(args.shared_selection), protocol)

    models: list[dict[str, Any]] = []
    roles: dict[str, list[str]] = {}
    for family, members in by_family.items():
        models.extend(members)
        roles[family] = [row["id"] for row in members]
    for key, row in deterministic.items():
        models.append(row)
        roles[key] = [row["id"]]
    models.append(shared)
    roles["shared"] = [shared["id"]]
    if len({row["id"] for row in models}) != 14:
        raise ValueError("Expected fourteen unique v6 evaluation model identifiers")

    comparisons = [
        {"name": "level_res_vs_old_res", "primary": True, "left": roles["level_res"], "right": roles["old_res"], "question": "LEVEL modification over the previous residual PEFT"},
        {"name": "level_res_vs_level_raw", "primary": False, "left": roles["level_res"], "right": roles["level_raw"], "question": "residual input contribution under matched level restoration"},
        {"name": "level_res_vs_level_only", "primary": False, "left": roles["level_res"], "right": roles["level_only"], "question": "learned correction beyond level-only reference"},
        {"name": "level_res_vs_compress", "primary": False, "left": roles["level_res"], "right": roles["compress"], "question": "learned residual correction beyond compressed frozen TSFM"},
        {"name": "level_res_vs_f0", "primary": False, "left": roles["level_res"], "right": roles["f0"], "question": "compressed LEVEL path versus uncompressed frozen TSFM"},
        {"name": "level_res_vs_lora", "primary": False, "left": roles["level_res"], "right": roles["lora"], "question": "compressed LEVEL path versus standard PEFT"},
        {"name": "level_res_vs_level_linear_res", "primary": False, "left": roles["level_res"], "right": roles["level_linear_res"], "question": "frozen TSFM main path versus small learned time-linear main path"},
        {"name": "level_res_vs_shared", "primary": False, "left": roles["level_res"], "right": roles["shared"], "question": "LEVEL residual PEFT versus full-TRAIN shared linear ridge"},
    ]

    gpu_ids = list(
        dict.fromkeys(
            identifier
            for role in ("f0", "lora", "old_res", "level_res", "level_raw", "level_linear_res", "level_only")
            for identifier in roles[role]
        )
    )
    cpu_ids = roles["shared"] + roles["level_linear_res"]
    sealed = {
        "schema": "tsfm_peft_level_confirmation_v6_evaluation_seal_v1",
        "status": "sealed",
        "sealed_utc": time.time(),
        "selection": receipt(args.selection),
        "shared_selection": receipt(args.shared_selection),
        "data_contract": receipt(args.data_contract),
        "data_receipts": {k: contract[k] for k in ("trainval", "test")},
        "protocol": receipt(HERE / "protocol.json"),
        "plan": receipt(HERE / "PLAN.md"),
        "approval": receipt(HERE / "APPROVAL.txt"),
        "channels": protocol["selected_columns"],
        "splits": protocol["splits"],
        "origins": contract["origins"]["lists"],
        "origin_counts": protocol["expected_origins"],
        "metric": protocol.get("primary_metric", "TRAIN-std normalized observed-target channel-macro MSE"),
        "bootstrap": protocol.get("bootstrap", {"block_length_origins": 7, "draws": 2000, "seed": 9262026}),
        "models": models,
        "roles": roles,
        "comparisons": comparisons,
        "cost": dict(protocol.get("cost", {}), gpu_model_ids=gpu_ids, cpu_model_ids=cpu_ids),
        "source": source_receipt(),
        "environment": environment_receipt(),
        "test_opened_during_seal": False,
        "scope": "One fixed Robin Office electricity group; global pretraining disjointness is not certified.",
    }
    save_json(destination, sealed)
    return {"seal": str(destination), "sha256": digest(destination), "models": len(models)}


def load_seal() -> dict[str, Any]:
    path = HERE / "evaluation_seal.json"
    sealed = read_json(path)
    if sealed.get("status") != "sealed":
        raise ValueError("Evaluation requires a completed fixed seal")
    for key in ("selection", "shared_selection", "data_contract", "protocol", "plan", "approval"):
        check_receipt(sealed[key])
    for model in sealed["models"]:
        if model["kind"] == "checkpoint":
            check_receipt({"path": model["checkpoint"], "sha256": model["checkpoint_sha256"]})
            check_receipt({"path": model["result_path"], "sha256": model["result_sha256"]})
        else:
            check_receipt(model["weights"])
        check_receipt(model["best_val_prediction"])
    return sealed


def shared_predict(inputs: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    with np.load(check_receipt(row["weights"]), allow_pickle=False) as saved:
        weight, bias = saved["weight"], saved["bias"]
    if weight.shape != (CONTEXT, HORIZON) or bias.shape != (HORIZON,):
        raise ValueError("SHARED saved affine dimensions differ from L512/H48")
    prediction = np.einsum("blc,lh->bhc", inputs.astype(np.float64), weight.astype(np.float64)) + bias[None, :, None]
    return prediction.astype(np.float32)


def split_start_end(splits: dict[str, Any], period: str) -> tuple[str, str]:
    item = splits[period]
    if isinstance(item, dict):
        return item["start"], item["end"]
    if isinstance(item, list | tuple) and len(item) == 2:
        return item[0], item[1]
    raise ValueError(f"Unexpected split format for {period}: {item!r}")


def error_terms(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray) -> dict[str, np.ndarray]:
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("Prediction, target and observation mask shapes differ")
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError("Predictions and observed targets must be finite")
    delta = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0)
    return {
        "squared": (delta * delta).sum(axis=1),
        "absolute": np.abs(delta).sum(axis=1),
        "count": observed.sum(axis=1),
    }


def scores_from_terms(terms: dict[str, np.ndarray], index: np.ndarray | None = None) -> dict[str, Any]:
    if index is None:
        index = np.arange(len(terms["count"]))
    counts = terms["count"][index].sum(axis=0)
    squared = terms["squared"][index].sum(axis=0)
    absolute = terms["absolute"][index].sum(axis=0)
    valid = counts > 0
    channel_mse = [float(squared[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
    channel_mae = [float(absolute[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
    return {
        "status": "available" if valid.all() else "unavailable_missing_channel_targets",
        "mse": float(np.mean(channel_mse)) if valid.all() else None,
        "mae": float(np.mean(channel_mae)) if valid.all() else None,
        "channel_mse": channel_mse,
        "channel_mae": channel_mae,
        "channel_target_count": counts.tolist(),
        "channel_squared_error_sum": squared.tolist(),
        "channel_absolute_error_sum": absolute.tolist(),
        "missing_channel_indices": np.flatnonzero(~valid).tolist(),
    }


def pair_ids(ids: list[str], models: dict[str, dict[str, Any]]) -> list[str]:
    if len(ids) == 1:
        return ids * 2
    by_seed = {models[i]["spec"]["seed"]: i for i in ids}
    if set(by_seed) != set(SEEDS):
        raise ValueError("Comparison must pair both fixed seeds or broadcast one deterministic model")
    return [by_seed[s] for s in SEEDS]


def compare_terms(
    comparison: dict[str, Any],
    terms: dict[str, dict[str, np.ndarray]],
    models: dict[str, dict[str, Any]],
    period_indices: dict[str, np.ndarray],
    job: Job,
) -> dict[str, Any]:
    left_ids, right_ids = (pair_ids(comparison[k], models) for k in ("left", "right"))
    count = terms[left_ids[0]]["count"]
    for identifier in left_ids + right_ids:
        if not np.array_equal(terms[identifier]["count"], count):
            raise ValueError("All comparisons must use the same observed targets")
    left = np.mean([terms[i]["squared"] for i in left_ids], axis=0)
    right = np.mean([terms[i]["squared"] for i in right_ids], axis=0)

    def values(index: np.ndarray) -> dict[str, float] | None:
        denominator = count[index].sum(axis=0)
        if np.any(denominator == 0):
            return None
        a = float(np.mean(left[index].sum(axis=0) / denominator))
        b = float(np.mean(right[index].sum(axis=0) / denominator))
        return {
            "left_mse": a,
            "reference_mse": b,
            "delta_mse": a - b,
            "relative_to_reference_pct": 100 * (a - b) / b if b > 0 else None,
        }

    output = dict(
        comparison,
        direction="LEVEL_RES minus reference; negative favors LEVEL_RES",
        left_seed_ids=left_ids,
        right_seed_ids=right_ids,
        deterministic_reference_broadcast=len(comparison["right"]) == 1,
        periods={},
    )
    for label in (*PERIODS, "combined"):
        strata = [period_indices[label]] if label != "combined" else list(period_indices.values())
        full_index = np.concatenate(strata)
        full = values(full_index)
        seed_values = []
        for seed, left_id, right_id in zip(SEEDS, left_ids, right_ids):
            a = scores_from_terms(terms[left_id], full_index)["mse"]
            b = scores_from_terms(terms[right_id], full_index)["mse"]
            seed_values.append(
                {
                    "seed": seed,
                    "left_id": left_id,
                    "right_id": right_id,
                    "left_mse": a,
                    "reference_mse": b,
                    "delta_mse": a - b if a is not None and b is not None else None,
                    "relative_to_reference_pct": 100 * (a - b) / b if a is not None and b is not None and b > 0 else None,
                }
            )
        rng = np.random.default_rng(9262026)
        draws = []
        unavailable = 0
        for draw in range(2000):
            if draw % 250 == 0:
                job.check_limits()
            selected = []
            for stratum in strata:
                n = len(stratum)
                starts = rng.integers(0, n - 7 + 1, int(np.ceil(n / 7)))
                index = np.concatenate([np.arange(start, start + 7) for start in starts])[:n]
                selected.append(stratum[index])
            value = values(np.concatenate(selected))
            if value is None or value["relative_to_reference_pct"] is None:
                unavailable += 1
            else:
                draws.append(value)
        interval = None
        if full is not None and unavailable == 0:
            interval = {
                key: np.quantile([row[key] for row in draws], [0.025, 0.975]).tolist()
                for key in ("delta_mse", "relative_to_reference_pct")
            }
        output["periods"][label] = {
            "full": full,
            "paired_seed_deltas": seed_values,
            "conditional_block95": interval,
            "draws": 2000,
            "unavailable_draws": unavailable,
            "interval_status": "available" if interval is not None else "unavailable_without_dropping_channels",
            "strata": [label] if label != "combined" else list(PERIODS),
        }
    output["interval_scope"] = "Non-circular 7-origin blocks within each period, common resamples across channels/models; two trained seeds fixed. Does not include training or selection uncertainty."
    return output


def evaluate_sealed(args: argparse.Namespace, job: Job) -> dict[str, Any]:
    sealed = load_seal()
    exposure_path = HERE / "test_exposure.json"
    seal_receipt = receipt(HERE / "evaluation_seal.json")
    previous_result = None
    if exposure_path.exists():
        if not args.correction_reason or args.reuse_evaluation is None:
            raise FileExistsError("TEST exposed: require a documented correction and previous result/partial file to inspect and reuse completed predictions")
        previous = read_json(exposure_path)
        if previous["seal_sha256"] != seal_receipt["sha256"]:
            raise ValueError("Cannot correct against a different TEST selection seal")
        previous_result = read_json(args.reuse_evaluation)
        if previous_result["seal"]["sha256"] != seal_receipt["sha256"]:
            raise ValueError("Previous evaluation does not belong to this frozen seal")
        for entry in previous_result["models"].values():
            check_receipt(entry["prediction"])
    else:
        if args.correction_reason or args.reuse_evaluation or args.recompute_ids:
            raise ValueError("A correction reason is only appropriate after recorded exposure")
        save_json(
            exposure_path,
            {
                "schema": "tsfm_peft_level_confirmation_v6_test_exposure_v1",
                "first_exposed_utc": time.time(),
                "status": "exposed",
                "seal_sha256": seal_receipt["sha256"],
                "seal": seal_receipt,
                "evaluation_name": args.name,
                "command": sys.argv,
                "scope": "Written before the first protected TEST archive is opened; immutable first exposure record.",
            },
        )
    result = {
        "schema": "tsfm_peft_level_confirmation_v6_test_evaluation_v1",
        "status": "running",
        "name": args.name,
        "seal": seal_receipt,
        "first_exposure": receipt(exposure_path),
        "correction_reason": args.correction_reason or None,
        "reuse_evaluation": receipt(args.reuse_evaluation) if args.reuse_evaluation else None,
        "explicit_recompute_ids": args.recompute_ids,
        "source": source_receipt(),
        "environment": environment_receipt(),
        "channels": sealed["channels"],
        "origins": {period: sealed["origins"][period] for period in PERIODS},
        "models": {},
        "roles": sealed["roles"],
        "metric": sealed["metric"],
        "comparisons": [],
    }
    save_json(HERE / f"{args.name}_partial.json", result)
    data = load_data(include_test=True)
    times = np.asarray(data["times"]).astype("datetime64[ns]")
    origins_by_period = {period: np.asarray(data[f"{period}_origins"], dtype=np.int64) for period in PERIODS}
    for period, origins in origins_by_period.items():
        if len(origins) != 40 or len(np.unique(origins)) != 40 or np.any(np.diff(origins) != 24):
            raise ValueError(f"{period}: sealed origin count/stride mismatch")
        if origins.tolist() != sealed["origins"][period]:
            raise ValueError(f"{period}: origins changed from the pre-exposure seal")
        if np.any(origins < CONTEXT) or np.any(origins + HORIZON > len(times)):
            raise ValueError(f"{period}: context/target slice falls outside the archive")
        split_start, split_end = split_start_end(sealed["splits"], period)
        start, end = (np.datetime64(value, "ns") for value in (split_start, split_end))
        if (
            np.any(times[origins] < start)
            or np.any(times[origins + HORIZON - 1] >= end)
            or np.any(times[origins + HORIZON - 1] - times[origins] != np.timedelta64(HORIZON - 1, "h"))
        ):
            raise ValueError(f"{period}: a full 48-hour target crosses its sealed period bounds")
    if origins_by_period["test_a"][-1] + HORIZON > origins_by_period["test_b"][0]:
        raise ValueError("Period target intervals overlap")
    origins = np.concatenate(list(origins_by_period.values()))
    period_indices = {"test_a": np.arange(40), "test_b": np.arange(40, 80)}
    target = np.stack([data["x"][o : o + HORIZON] for o in origins])
    observed = np.stack([data["finite"][o : o + HORIZON] for o in origins]).astype(bool)
    if target.shape != (80, HORIZON, CHANNELS):
        raise ValueError(f"Unexpected sealed TEST shape: {target.shape}")
    if np.asarray(data["columns"]).astype(str).tolist() != sealed["channels"]:
        raise ValueError("TEST channel identities/order differ from the pre-exposure seal")
    prediction_dir = CACHE / f"{args.name}_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    module = load_module(HERE / "model_v6.py", "_v6_evaluation_model")
    models = {row["id"]: row for row in sealed["models"]}
    if not set(args.recompute_ids).issubset(models):
        raise ValueError("Correction requested an unknown model ID")
    terms: dict[str, dict[str, np.ndarray]] = {}
    for identifier, row in models.items():
        job.check_limits()
        reused = previous_result is not None and identifier in previous_result["models"] and identifier not in args.recompute_ids
        if reused:
            prediction_receipt = previous_result["models"][identifier]["prediction"]
            with np.load(check_receipt(prediction_receipt), allow_pickle=False) as archive:
                if not np.array_equal(archive["origins"], origins):
                    raise ValueError("Stored correction predictions have different TEST origins")
                prediction = archive["prediction"].copy()
        elif row["kind"] == "shared":
            inputs = np.stack([data["x"][o - CONTEXT : o] for o in origins])
            prediction = shared_predict(inputs, row)
            del inputs
        else:
            model = module.restore_model(resolve(row["checkpoint"]), device="cuda").eval()
            predictions = []
            for offset in range(0, len(origins), 4):
                job.check_limits()
                inputs = torch.as_tensor(
                    np.stack([data["x"][o - CONTEXT : o] for o in origins[offset : offset + 4]]),
                    dtype=torch.float32,
                    device="cuda",
                )
                predictions.append(model(inputs).float().cpu().numpy())
                del inputs
            prediction = np.concatenate(predictions)
            del predictions, model
            gc.collect()
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated():
                raise RuntimeError("Evaluation retains a prior model or output on CUDA")
        terms[identifier] = error_terms(prediction, target, observed)
        if not reused:
            prediction_path = prediction_dir / f"{identifier}.npz"
            np.savez_compressed(prediction_path, prediction=prediction, origins=origins)
            prediction_receipt = receipt(prediction_path)
        scores = {period: scores_from_terms(terms[identifier], index) for period, index in period_indices.items()}
        scores["combined"] = scores_from_terms(terms[identifier])
        result["models"][identifier] = {
            "spec": row["spec"],
            "kind": row["kind"],
            "prediction": prediction_receipt,
            "reused_prediction_after_documented_correction": reused,
            "scores": scores,
        }
        del prediction
        save_json(HERE / f"{args.name}_partial.json", result)
        job.heartbeat(f'{args.name}: evaluated {identifier}; {len(result["models"])}/{len(models)} models')
    result["role_scores"] = {}
    for role, ids in sealed["roles"].items():
        result["role_scores"][role] = {}
        for period in (*PERIODS, "combined"):
            scores = [result["models"][identifier]["scores"][period] for identifier in ids]
            available = all(score["mse"] is not None for score in scores)
            result["role_scores"][role][period] = {
                "run_ids": ids,
                "count": len(ids),
                "mse": float(np.mean([score["mse"] for score in scores])) if available else None,
                "mae": float(np.mean([score["mae"] for score in scores])) if available else None,
                "scope": "Mean of individual model channel-macro losses; deterministic fit counted once; not an ensemble.",
            }
    for comparison in sealed["comparisons"]:
        result["comparisons"].append(compare_terms(comparison, terms, models, period_indices, job))
        job.heartbeat(f'{args.name}: paired interval {comparison["name"]}')
    result.update(
        status="complete",
        completed_utc=time.time(),
        combined_definition="Within each seed/channel, sum TEST-A/B errors and observed counts before channel mean; then average individual seed losses. Deterministic references broadcast for seed-paired deltas.",
        new_test_model_predictions=sum(not row["reused_prediction_after_documented_correction"] for row in result["models"].values()),
        test_reselection=False,
    )
    save_json(HERE / f"{args.name}.json", result)
    return {"result": str(HERE / f"{args.name}.json"), "models": len(models)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--seal", action="store_true")
    operation.add_argument("--evaluate", action="store_true")
    parser.add_argument("--selection", type=Path, default=HERE / "selected.json")
    parser.add_argument("--shared-selection", type=Path, default=HERE / "shared_ridge_selection.json")
    parser.add_argument("--data-contract", type=Path, default=HERE / "data_contract.json")
    parser.add_argument("--name", required=True)
    parser.add_argument("--reserve-seconds", type=float, required=True)
    parser.add_argument("--correction-reason", default="")
    parser.add_argument("--reuse-evaluation", type=Path)
    parser.add_argument("--recompute-ids", nargs="*", default=[])
    args = parser.parse_args()
    validate_name(args.name)
    paths = [HERE / f"{args.name}{suffix}.json" for suffix in ("", "_partial", "_attempt")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Preserve existing evaluation attempts; use a new name")
    attempt = {
        "status": "running",
        "started_utc": time.time(),
        "command": sys.argv,
        "source": source_receipt(),
        "environment": environment_receipt(),
    }
    save_json(paths[-1], attempt)
    try:
        with Job(args.name, category="cpu_analysis" if args.seal else "gpu", reserve_seconds=args.reserve_seconds) as job:
            if args.seal:
                outcome = seal_evaluation(args)
            else:
                global torch
                import torch
                from threadpoolctl import threadpool_limits

                torch.set_num_threads(4)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                with threadpool_limits(limits=4), torch.inference_mode():
                    outcome = evaluate_sealed(args, job)
            attempt.update(status="complete", outcome=outcome)
            print(json.dumps(outcome), flush=True)
    except BaseException:
        attempt.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        attempt["ended_utc"] = time.time()
        save_json(paths[-1], attempt)


if __name__ == "__main__":
    main()
