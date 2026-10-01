"""Prepare v14 metadata manifests without fitting or prediction generation.

The reserved prepare run writes compact manifests that bind v14 to immutable
v13 arrays and canonical parent/reference rows.  It does not load numpy arrays,
load checkpoints, build models, cache parent forecasts, score, or fit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V13 = ROOT / "research" / "tsfm_peft_group_basis_v13_20261001"
DATASET_ORDER = ("robin", "jena", "hog", "peacock_education")
SEEDS = (92601, 92602)
OLD_REFERENCE_FAMILIES = ("pca_level", "a", "f0", "full_mse", "native", "direct")
REFERENCE_FAMILIES = OLD_REFERENCE_FAMILIES + ("group_res",)
PARENT_ROLES = {"level": "pca_level", "direct": "direct"}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def write_json_new(path: str | Path, value: dict[str, Any]) -> None:
    path = resolve(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with resolve(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = resolve(path)
    sha = sha256_file(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise RuntimeError(f"sha256 mismatch for {path}")
    return {"path": str(path), "sha256": sha, "bytes": path.stat().st_size}


def artifact_from_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return artifact(row["path"], row.get("sha256"))


def require_seed_rows(rows: list[dict[str, Any]], family: str, dataset: str) -> list[dict[str, Any]]:
    matches = sorted((row for row in rows if row["family"] == family), key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in matches] != list(SEEDS):
        raise RuntimeError(f"{dataset} {family} must have exactly the fixed v14 seeds")
    return matches


def compact_source_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keep = (
        "id",
        "dataset",
        "family",
        "seed",
        "checkpoint",
        "sources",
        "prediction_key",
        "prediction_keys",
        "fit0_same_checkpoint_reference",
        "parent",
        "canonical_id",
        "spec",
    )
    return {key: copy.deepcopy(row[key]) for key in keep if key in row}


def compact_old_reference(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    result = {
        "id": row["id"],
        "dataset": dataset,
        "family": row["family"],
        "seed": row.get("seed"),
        "source_generation": row["source_generation"],
        "source_family": row["source_family"],
        "source_model_id": row["source_model_id"],
        "checkpoint": artifact_from_row(row.get("checkpoint")),
        "prediction": artifact_from_row(row.get("prediction")),
        "source_evaluation": artifact_from_row(row.get("source_evaluation")),
        "prediction_source_evaluation": artifact_from_row(row.get("prediction_source_evaluation")),
        "restore": copy.deepcopy(row.get("restore")),
        "source_row_ref": {
            "manifest": str(V13 / "reuse_manifest.json"),
            "dataset": dataset,
            "id": row["id"],
            "family": row["family"],
            "seed": row.get("seed"),
        },
        "source_row": compact_source_row(row.get("source_row")),
    }
    return result


def group_res_references(dataset: str, selected: dict[str, Any], prediction_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    item = selected["selected"][dataset]["group_res"]
    predictions = {
        row["id"]: row
        for row in prediction_manifest["datasets"][dataset]["models"]
        if row["family"] == "group_res"
    }
    rows = []
    for index, seed in enumerate(SEEDS):
        run_id = item["run_ids"][index]
        pred = predictions.get(run_id)
        if pred is None:
            raise RuntimeError(f"v13 group_res prediction missing for {run_id}")
        checkpoint = artifact(item["checkpoints"][index], item["checkpoint_sha256"][index])
        rows.append(
            {
                "id": run_id,
                "dataset": dataset,
                "family": "group_res",
                "seed": seed,
                "source_generation": "v13",
                "source_family": "group_res",
                "source_model_id": run_id,
                "checkpoint": checkpoint,
                "prediction": artifact_from_row(pred["prediction"]),
                "source_evaluation": artifact(V13 / "evaluation01.json"),
                "prediction_source_evaluation": None,
                "restore": {
                    "source_generation": "v13",
                    "source_family": "group_res",
                    "source_model_id": run_id,
                    "checkpoint_available": True,
                    "api": "research/tsfm_peft_group_basis_v13_20261001/model_v13.py::restore_model",
                },
                "source_row_ref": {
                    "manifest": str(V13 / "selected.json"),
                    "dataset": dataset,
                    "id": run_id,
                    "family": "group_res",
                    "seed": seed,
                },
                "source_row": {
                    "id": run_id,
                    "dataset": dataset,
                    "family": "group_res",
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "selected_epoch": item["selected_epochs"][index],
                    "best_val_prediction": artifact_from_row(item["best_val_prediction"][index]),
                },
            }
        )
    return rows


def old_reference_rows(dataset: str, v13_reuse: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    source_rows = v13_reuse["datasets"][dataset]["references"]
    for family in OLD_REFERENCE_FAMILIES:
        if family == "f0":
            matches = [row for row in source_rows if row["family"] == family and row.get("seed") is None]
            if len(matches) != 1:
                raise RuntimeError(f"{dataset} requires exactly one f0 reference")
        else:
            matches = require_seed_rows(source_rows, family, dataset)
        rows.extend(compact_old_reference(row, dataset) for row in matches)
    return rows


def parent_map(references: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for role, family in PARENT_ROLES.items():
        rows = require_seed_rows(references, family, "parent_map")
        result[role] = {}
        for row in rows:
            entry = copy.deepcopy(row)
            entry["parent_role"] = role
            entry["note"] = "Stored prediction is old TEST/development receipt only; do not use for TRAIN correction."
            result[role][str(row["seed"])] = entry
    return result


def initial_g_receipts(init: dict[str, Any], dataset: str) -> dict[str, Any]:
    return {
        str(seed): copy.deepcopy(init["datasets"][dataset][str(seed)]["source_initial_g"])
        for seed in SEEDS
    }


def initial_manifest(v13_data: dict[str, Any]) -> dict[str, Any]:
    datasets = {}
    for dataset in DATASET_ORDER:
        datasets[dataset] = {}
        for seed in SEEDS:
            source = v13_data["datasets"][dataset]["initial_g"][str(seed)]
            datasets[dataset][str(seed)] = {
                "source_initial_g": artifact(source["path"], source["sha256"]),
                "source_schema": source.get("schema"),
                "state_keys": source.get("state_keys"),
                "tensor_hashes": copy.deepcopy(source.get("tensor_hashes")),
                "v14_policy": {
                    "down": "copy original untrained residual.down.weight",
                    "up": "literal zero initialization; never copy fitted or parent up weights",
                    "shared_across_families": True,
                },
            }
    return {
        "schema": "v14_initial_manifest_1",
        "source": artifact(V13 / "data_manifest.json"),
        "scope": "Initial-G recipe only; no tensors are loaded or rewritten during manifest preparation.",
        "datasets": datasets,
    }


def build_manifests() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    v13_reuse = read_json(V13 / "reuse_manifest.json")
    v13_data = read_json(V13 / "data_manifest.json")
    v13_selected = read_json(V13 / "selected.json")
    v13_predictions = read_json(V13 / "prediction_manifest.json")
    init = initial_manifest(v13_data)
    datasets = {}
    data_datasets = {}
    for dataset in DATASET_ORDER:
        old_rows = old_reference_rows(dataset, v13_reuse)
        group_rows = group_res_references(dataset, v13_selected, v13_predictions)
        references = old_rows + group_rows
        family_counts = {
            family: len([row for row in references if row["family"] == family])
            for family in REFERENCE_FAMILIES
        }
        if family_counts != {
            "pca_level": 2,
            "a": 2,
            "f0": 1,
            "full_mse": 2,
            "native": 2,
            "direct": 2,
            "group_res": 2,
        }:
            raise RuntimeError(f"{dataset} reference rows do not match v14 contract: {family_counts}")
        parents = parent_map(references)
        datasets[dataset] = {
            "references": references,
            "parents": parents,
            "reference_family_counts": family_counts,
        }
        v13_entry = v13_data["datasets"][dataset]
        data_datasets[dataset] = {
            "trainval": artifact(v13_entry["trainval"]["path"], v13_entry["trainval"]["sha256"]),
            "test": artifact(v13_entry["test"]["path"], v13_entry["test"]["sha256"]),
            "columns": copy.deepcopy(v13_entry["columns"]),
            "C": v13_entry["C"],
            "K": v13_entry["K"],
            "phase_period": v13_entry["phase_period"],
            "block_origins": v13_entry["block_origins"],
            "initial_g": initial_g_receipts(init, dataset),
            "initial_g_recipe_ref": {
                "manifest": "initial_manifest.json",
                "dataset": dataset,
                "policy": "copy original untrained residual.down.weight; initialize residual.up.weight as literal zero",
            },
            "parents": parents,
            "array_contract": {
                "parent_basis": "use trainval/test pca_basis from v13 archives",
                "prior_group_basis": "basis key is prior v13 GROUP U and is not the v14 parent basis",
                "source_arrays": "exact v13 trainval/test archives; no duplicated raw arrays",
            },
            "parent_cache": {
                "status": "not_prepared",
                "expected_manifest": "parent_cache.json",
                "required_roles": list(PARENT_ROLES),
                "splits": ["train", "val"],
                "prediction_scope": "full parent forecast on TRAIN/VAL origins only; never TEST",
            },
        }
    reuse_manifest = {
        "schema": "v14_reuse_manifest_1",
        "source": {
            "v13_reuse_manifest": artifact(V13 / "reuse_manifest.json"),
            "v13_selected": artifact(V13 / "selected.json"),
            "v13_prediction_manifest": artifact(V13 / "prediction_manifest.json"),
            "v13_evaluation": artifact(V13 / "evaluation01.json"),
        },
        "dataset_order": list(DATASET_ORDER),
        "seeds": list(SEEDS),
        "reference_families": list(REFERENCE_FAMILIES),
        "parent_roles": copy.deepcopy(PARENT_ROLES),
        "datasets": datasets,
        "scope": "Compact restore/evaluation references; source_row excludes periods and repeated diagnostic payloads.",
    }
    data_manifest = {
        "schema": "v14_data_manifest_1",
        "source": {
            "v13_data_manifest": artifact(V13 / "data_manifest.json"),
            "v13_protocol": artifact(V13 / "protocol.json"),
        },
        "dataset_order": list(DATASET_ORDER),
        "datasets": data_datasets,
        "training_parent_cache_required": True,
        "no_parent_trainval_cache_claim": (
            "prepare binds parent checkpoints and v13 arrays only; full TRAIN/VAL parent forecasts "
            "must be generated later under a reserved ledger job."
        ),
    }
    return reuse_manifest, data_manifest, init


def write_manifests() -> dict[str, Any]:
    reuse_manifest, data_manifest, init = build_manifests()
    write_json_new(HERE / "reuse_manifest.json", reuse_manifest)
    write_json_new(HERE / "data_manifest.json", data_manifest)
    write_json_new(HERE / "initial_manifest.json", init)
    return {
        "schema": "v14_prepare_receipt_1",
        "reuse_manifest": artifact(HERE / "reuse_manifest.json"),
        "data_manifest": artifact(HERE / "data_manifest.json"),
        "initial_manifest": artifact(HERE / "initial_manifest.json"),
        "parent_cache_created": False,
        "predictions_created": False,
        "fits_created": False,
    }


def prepare_with_job(job_label: str, reserve_s: float) -> dict[str, Any]:
    import sys

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from runtime_v14 import Job

    metadata = {
        "purpose": "v14 manifest preparation only",
        "fit_count": 0,
        "prediction_count": 0,
        "source": "v13 immutable manifests and archives",
    }
    with Job("cpu_analysis", job_label, reserve_s=reserve_s, metadata=metadata) as job:
        result = write_manifests()
        job.heartbeat("v14 reuse/data/initial manifests prepared; parent TRAIN/VAL cache still absent")
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--reserve-s", type=float, required=True)
    args = parser.parse_args()
    result = prepare_with_job(args.job, args.reserve_s)
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
