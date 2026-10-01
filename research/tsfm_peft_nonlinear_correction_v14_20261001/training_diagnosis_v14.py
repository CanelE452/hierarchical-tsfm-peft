"""Summarize v14 training dynamics from completed JSON receipts only.

This post-hoc diagnostic reads the canonical completed result records, their
VAL curves, and the final joint selection.  It does not open data arrays, load
predictions, score, fit, or rerun models.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

from runtime_v14 import HERE, Job, artifact, protocol, read_json, save_json
from train_v14 import RUNS, result_for


OUTPUT = HERE / "training_diagnosis.json"
SCHEMA = "v14_training_diagnosis_1"


def as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def score(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "mse": as_float(value.get("mse")),
        "mae": as_float(value.get("mae")),
    }


def delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    return {
        "mse": None if after.get("mse") is None or before.get("mse") is None else float(after["mse"] - before["mse"]),
        "mae": None if after.get("mae") is None or before.get("mae") is None else float(after["mae"] - before["mae"]),
        "direction": "negative means lower loss after the later checkpoint",
    }


def curve_path(result: dict[str, Any]) -> Path:
    return RUNS / result["id"] / "curve.json"


def curve_summary(result: dict[str, Any]) -> dict[str, Any]:
    path = curve_path(result)
    curve = read_json(path)
    if not curve:
        raise RuntimeError("Empty training curve: " + str(path))
    epochs = [int(row["epoch"]) for row in curve]
    if epochs != list(range(len(curve))):
        raise RuntimeError("Curve epochs are not contiguous from zero: " + str(path))
    first = curve[0]
    final = curve[-1]
    best = min(curve, key=lambda row: (float(row["val_mse"]), int(row["epoch"])))
    if int(best["epoch"]) != int(result["selected_epoch"]):
        raise RuntimeError("Curve best epoch differs from result selected epoch: " + result["id"])
    if float(best["val_mse"]) != float(result["best_val_mse"]):
        raise RuntimeError("Curve best VAL differs from result best_val_mse: " + result["id"])
    if float(first["val_mse"]) != float(result["initial_val_mse"]):
        raise RuntimeError("Curve epoch0 VAL differs from result initial_val_mse: " + result["id"])
    return {
        "artifact": artifact(path),
        "epoch_count": len(curve),
        "initial": {"epoch": int(first["epoch"]), **score({"mse": first["val_mse"], "mae": first["val_mae"]})},
        "best": {"epoch": int(best["epoch"]), **score({"mse": best["val_mse"], "mae": best["val_mae"]})},
        "final": {"epoch": int(final["epoch"]), "steps": int(final["steps"]),
                  "attempt_updates": int(final["attempt_updates"]),
                  **score({"mse": final["val_mse"], "mae": final["val_mae"]})},
    }


def termination(result: dict[str, Any], curve: dict[str, Any]) -> dict[str, Any]:
    selected_epoch = int(result["selected_epoch"])
    completed = int(result["epochs_completed"])
    cap = bool(result.get("upper_epoch_limit_improving"))
    patience = (not cap) and completed > 0 and completed - selected_epoch >= 6
    if cap:
        reason = "epoch_cap_with_recent_best"
    elif completed >= 120:
        reason = "epoch_cap"
    elif patience:
        reason = "patience_after_selected_best"
    else:
        reason = "completed_without_classified_patience_or_cap"
    return {
        "reason": reason,
        "patience_stop": bool(patience),
        "cap_stop": bool(cap or completed >= 120),
        "upper_epoch_limit_improving": cap,
        "epochs_completed": completed,
        "selected_epoch": selected_epoch,
        "epochs_after_selected": completed - selected_epoch,
        "curve_epoch_count": curve["epoch_count"],
    }


def selected_kind(result: dict[str, Any]) -> str:
    epoch0 = int(result["selected_epoch"]) == 0
    changed = bool(result.get("selected_head_changed"))
    if epoch0 and not changed:
        return "step0_exact_parent_fallback"
    if not epoch0 and changed:
        return "learned_selected"
    if epoch0 and changed:
        return "step0_but_selected_head_changed_inconsistent"
    return "nonstep0_but_selected_head_unchanged_inconsistent"


def gradient_summaries(result: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    gradients = result.get("first_update_gradients") or {}
    for step in sorted(gradients, key=lambda value: int(value))[:2]:
        tensors = gradients[step]
        per_tensor = {}
        max_abs_values = []
        total_numel = total_nonzero = tensors_with_grad = 0
        for name in sorted(tensors):
            item = tensors[name]
            numel = int(item.get("numel", 0))
            nonzero = int(item.get("nonzero", 0))
            has_grad = bool(item.get("has_grad"))
            max_abs = as_float(item.get("max_abs"))
            total_numel += numel
            total_nonzero += nonzero
            tensors_with_grad += int(has_grad)
            if max_abs is not None:
                max_abs_values.append(max_abs)
            per_tensor[name] = {
                "numel": numel,
                "has_grad": has_grad,
                "nonzero": nonzero,
                "max_abs": max_abs,
            }
        summaries.append({
            "step": int(step),
            "tensor_count": len(tensors),
            "tensors_with_grad": tensors_with_grad,
            "total_numel": total_numel,
            "total_nonzero": total_nonzero,
            "max_abs_max": max(max_abs_values) if max_abs_values else None,
            "per_tensor": per_tensor,
        })
    return summaries


def compact_fit(result: dict[str, Any]) -> dict[str, Any]:
    curve = curve_summary(result)
    initial_train = score(result["initial_train_probe"])
    selected_train = score(result["selected_train_probe"])
    final_train = score(result["final_train_probe"])
    initial_val = curve["initial"]
    best_val = curve["best"]
    final_val = curve["final"]
    return {
        "id": result["id"],
        "canonical_id": result["canonical_id"],
        "spec": copy.deepcopy(result["spec"]),
        "dataset": result["spec"]["dataset"],
        "family": result["spec"]["family"],
        "seed": int(result["spec"]["seed"]),
        "lr": float(result["spec"]["lr"]),
        "parent_role": result["parent_role"],
        "selection_state": selected_kind(result),
        "fixed_train_probe": {
            "scope": "fixed TRAIN probe saved in result.json; not full TRAIN scoring",
            "initial": initial_train,
            "selected": selected_train,
            "final": final_train,
            "selected_delta_from_initial": delta(selected_train, initial_train),
            "final_delta_from_initial": delta(final_train, initial_train),
        },
        "val_curve": {
            "initial": initial_val,
            "best": best_val,
            "final": final_val,
            "best_delta_from_initial": delta(best_val, initial_val),
            "final_delta_from_initial": delta(final_val, initial_val),
        },
        "training": {
            "selected_epoch": int(result["selected_epoch"]),
            "epochs_completed": int(result["epochs_completed"]),
            "actual_updates": int(result["actual_updates"]),
            "carried_steps": int(result.get("carried_steps", 0)),
            "cumulative_model_steps": int(result.get("cumulative_model_steps", result["actual_updates"])),
            "termination": termination(result, curve),
            "replay_error": as_float(result.get("replay_error")),
            "restart_roundtrip": copy.deepcopy(result.get("restart_roundtrip")),
        },
        "head_change": {
            "selected_nonstep0": bool(result.get("selected_nonstep0")),
            "selected_head_changed": bool(result.get("selected_head_changed")),
            "changed_parameter_scalars": copy.deepcopy(result.get("changed_parameter_scalars")),
            "parameter_max_updates": copy.deepcopy(result.get("parameter_max_updates")),
            "selected_parameter_max_updates": copy.deepcopy(result.get("selected_parameter_max_updates")),
        },
        "technical_validity": {
            "status": result["status"],
            "ledger_id": result["ledger_id"],
            "data_manifest_sha256": result["data_manifest_sha256"],
            "frozen_parameters_verified": copy.deepcopy(result.get("frozen_parameters_verified")),
            "first2_gradient_summaries": gradient_summaries(result),
        },
        "artifacts": {
            "checkpoint": {"path": result["checkpoint"], "sha256": result["checkpoint_sha256"]},
            "initial_checkpoint": copy.deepcopy(result["initial_checkpoint"]),
            "last_checkpoint": copy.deepcopy(result["last_checkpoint"]),
            "best_val_prediction": copy.deepcopy(result["best_val_prediction"]),
            "curve": curve["artifact"],
        },
    }


def add_group(groups: dict[str, Any], row: dict[str, Any]) -> None:
    dataset = row["dataset"]
    family = row["family"]
    group = groups[dataset][family]
    group["fits"] += 1
    group["step0_exact_parent_fallback"] += int(row["selection_state"] == "step0_exact_parent_fallback")
    group["learned_selected"] += int(row["selection_state"] == "learned_selected")
    group["selected_head_changed"] += int(row["head_change"]["selected_head_changed"])
    group["cap_stop"] += int(row["training"]["termination"]["cap_stop"])
    group["patience_stop"] += int(row["training"]["termination"]["patience_stop"])
    group["replay_zero"] += int(row["training"]["replay_error"] == 0.0)
    train_selected_delta = row["fixed_train_probe"]["selected_delta_from_initial"]["mse"]
    train_final_delta = row["fixed_train_probe"]["final_delta_from_initial"]["mse"]
    val_best_delta = row["val_curve"]["best_delta_from_initial"]["mse"]
    val_final_delta = row["val_curve"]["final_delta_from_initial"]["mse"]
    group["train_probe_selected_improved"] += int(train_selected_delta is not None and train_selected_delta < 0)
    group["train_probe_final_improved"] += int(train_final_delta is not None and train_final_delta < 0)
    group["val_best_improved"] += int(val_best_delta is not None and val_best_delta < 0)
    group["val_final_improved"] += int(val_final_delta is not None and val_final_delta < 0)
    group["train_selected_mse_delta_sum"] += train_selected_delta or 0.0
    group["train_final_mse_delta_sum"] += train_final_delta or 0.0
    group["val_best_mse_delta_sum"] += val_best_delta or 0.0
    group["val_final_mse_delta_sum"] += val_final_delta or 0.0
    group["train_selected_improved_val_best_not_improved"] += int(
        train_selected_delta is not None and train_selected_delta < 0
        and val_best_delta is not None and val_best_delta >= 0
    )
    group["train_final_improved_val_best_not_improved"] += int(
        train_final_delta is not None and train_final_delta < 0
        and val_best_delta is not None and val_best_delta >= 0
    )


def finalized_groups(groups: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for dataset in sorted(groups):
        output[dataset] = {}
        for family in sorted(groups[dataset]):
            group = dict(groups[dataset][family])
            fits = group["fits"]
            if fits:
                for key in ("train_selected_mse_delta", "train_final_mse_delta",
                            "val_best_mse_delta", "val_final_mse_delta"):
                    group[key + "_mean"] = group.pop(key + "_sum") / fits
            output[dataset][family] = group
    return output


def selected_rows(selection: dict[str, Any], all_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset, families in selection["selected"].items():
        for family, item in families.items():
            run_ids = item["run_ids"]
            seeds = item["seeds"]
            if len(run_ids) != len(seeds):
                raise RuntimeError("Selected run_ids/seeds length mismatch")
            for index, run_id in enumerate(run_ids):
                fit = all_by_id[run_id]
                if int(fit["seed"]) != int(seeds[index]):
                    raise RuntimeError("Selected seed differs from fit row: " + run_id)
                rows.append({
                    "dataset": dataset,
                    "family": family,
                    "seed": int(seeds[index]),
                    "run_id": run_id,
                    "lr": float(item["lr"]),
                    "parent_role": item["parent_role"],
                    "selection_state": fit["selection_state"],
                    "selected_epoch": fit["training"]["selected_epoch"],
                    "actual_updates": fit["training"]["actual_updates"],
                    "fixed_train_probe": copy.deepcopy(fit["fixed_train_probe"]),
                    "val_curve": copy.deepcopy(fit["val_curve"]),
                    "head_change": copy.deepcopy(fit["head_change"]),
                    "termination": copy.deepcopy(fit["training"]["termination"]),
                    "replay_error": fit["training"]["replay_error"],
                    "mean_best_val_mse_for_selected_lr": as_float(item.get("mean_best_val_mse")),
                    "family_selection_summary": {
                        "selected_epochs": copy.deepcopy(item.get("selected_epochs")),
                        "nonstep0_seeds": int(item.get("nonstep0_seeds", 0)),
                        "changed_head_seeds": int(item.get("changed_head_seeds", 0)),
                    },
                })
    if len(rows) != 24:
        raise RuntimeError(f"Expected 24 selected seed rows, found {len(rows)}")
    return rows


def build(job: Job) -> dict[str, Any]:
    spec = protocol()
    selection = read_json(HERE / "selected.json")
    allfits = []
    groups = defaultdict(lambda: defaultdict(lambda: {
        "fits": 0,
        "step0_exact_parent_fallback": 0,
        "learned_selected": 0,
        "selected_head_changed": 0,
        "cap_stop": 0,
        "patience_stop": 0,
        "replay_zero": 0,
        "train_probe_selected_improved": 0,
        "train_probe_final_improved": 0,
        "val_best_improved": 0,
        "val_final_improved": 0,
        "train_selected_mse_delta_sum": 0.0,
        "train_final_mse_delta_sum": 0.0,
        "val_best_mse_delta_sum": 0.0,
        "val_final_mse_delta_sum": 0.0,
        "train_selected_improved_val_best_not_improved": 0,
        "train_final_improved_val_best_not_improved": 0,
    }))
    for fit_spec in spec["fits"]:
        result = result_for(fit_spec)
        row = compact_fit(result)
        allfits.append(row)
        add_group(groups, row)
        if len(allfits) % 8 == 0:
            job.heartbeat({"completed_fit_rows_read": len(allfits)})
    if len(allfits) != 48:
        raise RuntimeError(f"Expected 48 fit rows, found {len(allfits)}")
    all_by_id = {row["id"]: row for row in allfits}
    selected24 = selected_rows(selection, all_by_id)
    sources = {
        "protocol": artifact(HERE / "protocol.json"),
        "selected": artifact(HERE / "selected.json"),
        "ledger": artifact(HERE / "ledger.json"),
        "data_manifest": artifact(HERE / "data_manifest.json"),
        "script": artifact(Path(__file__)),
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scope": (
            "Post-hoc metadata diagnosis of completed v14 training only. "
            "No arrays, predictions, scoring, fitting, or causal proof."
        ),
        "sources": sources,
        "counts": {
            "allfits": len(allfits),
            "selected_seed_rows": len(selected24),
            "datasets": len(spec["units"]),
            "families": len(spec["families"]),
            "seeds": len(spec["seeds"]),
        },
        "delta_convention": "negative MSE/MAE deltas mean the later checkpoint has lower saved loss than the initial parent-fallback checkpoint",
        "allfits": allfits,
        "selected24": selected24,
        "group_counts_by_dataset_family": finalized_groups(groups),
        "selection": {
            "criterion": selection.get("criterion"),
            "test_used_for_selection": selection.get("test_used_for_selection"),
            "lr_candidates": copy.deepcopy(selection.get("lr_candidates")),
            "matched_initial_groups": selection.get("matched_initial_groups"),
            "matched_schedule_groups": selection.get("matched_schedule_groups"),
        },
        "claim_limits": (
            "These diagnostics compare saved TRAIN-probe and VAL metadata. They do not establish cause, "
            "do not define a numerical academic threshold, and do not re-evaluate any prediction arrays."
        ),
    }


def write_output(value: dict[str, Any]) -> None:
    if OUTPUT.exists():
        previous = read_json(OUTPUT)
        if previous != value:
            raise RuntimeError("Preserve existing training_diagnosis.json; remove only with explicit correction record")
        return
    save_json(OUTPUT, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--reserve-s", type=float, required=True)
    args = parser.parse_args()
    metadata = {"fit_count": 0, "purpose": "v14 training metadata diagnosis", "reads": "result/curve/selected JSON only"}
    with Job("cpu_analysis", args.job, reserve_s=args.reserve_s, metadata=metadata) as job:
        value = build(job)
        write_output(value)
        job.heartbeat({"training_diagnosis": "complete", "allfits": value["counts"]["allfits"],
                       "selected_seed_rows": value["counts"]["selected_seed_rows"]})


if __name__ == "__main__":
    main()
