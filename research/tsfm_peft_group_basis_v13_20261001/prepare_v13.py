"""Prepare fixed TRAIN-correlation group bases for v13.

This file intentionally does not run model training, scoring, TEST EDA, or any
dataset search.  The default action rewrites the existing v11/v12 arrays with a
new fixed group basis, preserving the original PCA basis as ``pca_basis``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name / "data"
INITIAL_CACHE = ROOT / ".cache" / HERE.name / "initial_g"

REUSE_MANIFEST = HERE / "reuse_manifest.json"
DATA_MANIFEST = HERE / "data_manifest.json"
DATASET_ORDER = ("robin", "jena", "hog", "peacock_education")
SEEDS = (92601, 92602)
INITIAL_G_KEYS = ("residual.down.weight", "residual.up.weight")
REFERENCES = ("pca_level", "a", "f0", "full_mse", "native", "direct")

CONTRACT_PATHS = {
    "robin": "research/tsfm_peft_internal_vs_subspace_v11_20260930/data_contract_robin.json",
    "jena": "research/tsfm_peft_internal_vs_subspace_v11_20260930/data_contract_jena.json",
    "hog": "research/tsfm_peft_internal_vs_subspace_v11_20260930/data_contract_hog.json",
    "peacock_education": (
        "research/tsfm_peft_a_confirmation_v12_20261001/"
        "data_contract_confirmation_peacock_education.json"
    ),
}

EVALUATION_PATHS = {
    "v11": "research/tsfm_peft_internal_vs_subspace_v11_20260930/evaluation01.json",
    "v12": "research/tsfm_peft_a_confirmation_v12_20261001/confirmation_evaluation01.json",
}

INITIAL_RESULT_PATHS = {
    "robin": {
        92601: "research/tsfm_peft_level_confirmation_v6_20260928/runs/"
        "v6_robin_level_linear_res_92601/result.json",
        92602: "research/tsfm_peft_level_confirmation_v6_20260928/runs/"
        "v6_robin_level_linear_res_92602/result.json",
    },
    "jena": {
        92601: "research/tsfm_peft_practical_controls_v7_20260928/runs/"
        "v7_jena_level_res_92601/result.json",
        92602: "research/tsfm_peft_practical_controls_v7_20260928/runs/"
        "v7_jena_level_res_92602/result.json",
    },
    "hog": {
        92601: "research/tsfm_peft_accuracy_recovery_v5_20260927/runs/"
        "v5_hog_level_res_92601/result.json",
        92602: "research/tsfm_peft_accuracy_recovery_v5_20260927/runs/"
        "v5_hog_level_res_92602/result.json",
    },
    "peacock_education": {
        92601: "research/tsfm_peft_a_confirmation_v12_20261001/confirmation_runs/"
        "peacock_education_level_lr1e-3_s92601__retry_heartbeat01/result.json",
        92602: "research/tsfm_peft_a_confirmation_v12_20261001/confirmation_runs/"
        "peacock_education_level_lr1e-3_s92602/result.json",
    },
}

V11_FAMILY_MAP = {
    "pca_level": "base_level",
    "a": "a_p_lora_qfixed",
    "f0": "f0",
    "full_mse": "full_lora_mse",
    "native": "lora_native",
    "direct": "direct_nlinear",
}

V12_FAMILY_MAP = {
    "pca_level": "level",
    "a": "a",
    "f0": "f0",
    "full_mse": "full_mse",
    "native": "native",
    "direct": "direct",
}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = resolve(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def write_json_new(path: str | Path, value: dict[str, Any]) -> None:
    path = resolve(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with resolve(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = resolve(path)
    sha = sha256_file(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise RuntimeError(f"Hash mismatch for {path}: {sha} != {expected_sha256}")
    return {"path": str(path), "sha256": sha, "bytes": path.stat().st_size}


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_protocol() -> dict[str, Any]:
    return read_json(HERE / "protocol.json")


def unit_columns(contract: dict[str, Any], dataset: str) -> list[str]:
    if "unit" in contract:
        return [str(v) for v in contract["unit"]["columns"]]
    if "dataset_info" in contract and contract["dataset_info"].get("columns"):
        return [str(v) for v in contract["dataset_info"]["columns"]]
    dataset_value = contract.get("dataset")
    if isinstance(dataset_value, dict) and dataset_value.get("selected_columns"):
        return [str(v) for v in dataset_value["selected_columns"]]
    raise KeyError(f"Cannot find selected columns in {dataset} contract")


def data_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return receipt(entry["path"], entry.get("sha256"))


def result_checkpoint(result: dict[str, Any]) -> dict[str, Any] | None:
    if "checkpoint" not in result:
        return None
    sha = result.get("checkpoint_sha256")
    item = receipt(result["checkpoint"], sha)
    if "id" in result:
        item["run_id"] = result["id"]
    return item


def result_file_receipt(dataset: str, seed: int) -> dict[str, Any]:
    return receipt(INITIAL_RESULT_PATHS[dataset][seed])


def prediction_source_eval_receipt(row: dict[str, Any]) -> dict[str, Any] | None:
    nested = row.get("prediction", {}).get("source_evaluation")
    if nested:
        return receipt(nested["path"], nested.get("sha256"))
    return None


def selected_checkpoint_from_result_id(identifier: str) -> dict[str, Any] | None:
    candidates = [
        ROOT / "research/tsfm_peft_level_confirmation_v6_20260928/runs" / identifier / "result.json",
        ROOT / "research/tsfm_peft_practical_controls_v7_20260928/runs" / identifier / "result.json",
        ROOT / "research/tsfm_peft_accuracy_recovery_v5_20260927/runs" / identifier / "result.json",
        ROOT / "research/tsfm_peft_fresh_group_v4_20260927/runs" / identifier / "result.json",
    ]
    for path in candidates:
        if path.exists():
            result = read_json(path)
            item = result_checkpoint(result)
            if item is not None:
                item["source_result"] = receipt(path)
            return item
    return None


def reference_rows(dataset: str, eval_path: str | Path, family_map: dict[str, str], source_generation: str) -> list[dict[str, Any]]:
    evaluation = read_json(eval_path)
    models = evaluation["datasets"][dataset]["models"]
    default_eval = receipt(eval_path)
    rows: list[dict[str, Any]] = []
    for canonical in REFERENCES:
        source_family = family_map[canonical]
        matches = [copy.deepcopy(row) for row in models.values() if row.get("family") == source_family]
        if source_family == "f0":
            matches = [row for row in matches if row.get("seed") is None]
            if len(matches) != 1:
                raise RuntimeError(f"{dataset} must have one f0 reference")
        else:
            matches = sorted(matches, key=lambda row: int(row["seed"]))
            if [int(row["seed"]) for row in matches] != list(SEEDS):
                raise RuntimeError(f"{dataset} {canonical} missing the two fixed seeds")
        for row in matches:
            prediction = receipt(row["prediction"]["path"], row["prediction"].get("sha256"))
            source_eval = default_eval
            prediction_source_eval = prediction_source_eval_receipt(row)
            source_model_id = row["id"]
            source_object = read_json(source_eval["path"])
            if source_model_id not in source_object["datasets"][dataset]["models"]:
                raise RuntimeError(f"{source_model_id} not found in its source evaluation")
            checkpoint = row.get("checkpoint")
            if checkpoint is not None:
                checkpoint = receipt(checkpoint["path"], checkpoint.get("sha256"))
            else:
                checkpoint = selected_checkpoint_from_result_id(source_model_id)
            rows.append(
                {
                    "id": source_model_id,
                    "dataset": dataset,
                    "family": canonical,
                    "seed": row.get("seed"),
                    "source_family": source_family,
                    "source_generation": source_generation,
                    "source_model_id": source_model_id,
                    "checkpoint": checkpoint,
                    "prediction": prediction,
                    "prediction_source_evaluation": prediction_source_eval,
                    "source_evaluation": source_eval,
                    "restore": {
                        "source_generation": source_generation,
                        "source_family": source_family,
                        "source_model_id": source_model_id,
                        "source_evaluation": source_eval,
                        "checkpoint_available": checkpoint is not None,
                    },
                    "source_row": row,
                }
            )
    if len(rows) != 11:
        raise RuntimeError(f"{dataset} reference row count changed: {len(rows)}")
    return rows


def build_reuse_manifest() -> dict[str, Any]:
    protocol = load_protocol()
    datasets: dict[str, Any] = {}
    v11_eval = ROOT / EVALUATION_PATHS["v11"]
    v12_eval = ROOT / EVALUATION_PATHS["v12"]
    for dataset in DATASET_ORDER:
        contract_path = ROOT / CONTRACT_PATHS[dataset]
        contract = read_json(contract_path)
        unit = protocol["units"][dataset]
        columns = unit_columns(contract, dataset)
        if len(columns) != unit["C"]:
            raise RuntimeError(f"{dataset} column count differs from protocol")
        if dataset == "peacock_education":
            source_generation = "v12"
            references = reference_rows(dataset, v12_eval, V12_FAMILY_MAP, source_generation)
        else:
            source_generation = "v11"
            references = reference_rows(dataset, v11_eval, V11_FAMILY_MAP, source_generation)
        initial_sources: dict[str, Any] = {}
        for seed in SEEDS:
            result_path = ROOT / INITIAL_RESULT_PATHS[dataset][seed]
            result = read_json(result_path)
            source = copy.deepcopy(result["initial_source"])
            source_receipt = receipt(source["path"], source.get("sha256"))
            tensor_hashes = {
                key: source["tensor_hashes"][key]
                for key in INITIAL_G_KEYS
            }
            initial_sources[str(seed)] = {
                "seed": seed,
                "source_spec": copy.deepcopy(result.get("spec")),
                "source_result": receipt(result_path),
                "source": source_receipt,
                "tensor_hashes": tensor_hashes,
                "initial_parameter_sha256": result.get("initial_parameter_sha256"),
                "selected_checkpoint_excluded": result_checkpoint(result),
                "selected_parameter_max_updates": result.get("selected_parameter_max_updates"),
                "scope": "copy only original untrained LEVEL residual.down/up tensors; do not transfer learned parameters",
            }
        datasets[dataset] = {
            "source_generation": source_generation,
            "source_contract": receipt(contract_path),
            "C": unit["C"],
            "K": unit["K"],
            "columns": columns,
            "phase_period": unit["phase_period"],
            "block_origins": unit["block_origins"],
            "data": {
                "trainval": data_entry(contract["trainval"]),
                "test": data_entry(contract["test"]),
            },
            "initial_g_sources": initial_sources,
            "references": references,
        }
    return {
        "schema": "v13_reuse_manifest_1",
        "scope": "Existing exposed v11/v12 datasets and old reference receipts for v13 group-basis development.",
        "dataset_order": list(DATASET_ORDER),
        "families": list(protocol["families"]),
        "reference_families": list(REFERENCES),
        "fit_plan": {
            "lr": 0.001,
            "seeds": list(SEEDS),
            "families_to_fit": list(protocol["families"]),
            "dataset_count": len(DATASET_ORDER),
            "total_fits": len(DATASET_ORDER) * len(SEEDS) * len(protocol["families"]),
        },
        "basis_policy": protocol["grouping"],
        "datasets": datasets,
    }


def copy_archive_with_group_basis(source_entry: dict[str, Any], group_u: np.ndarray) -> dict[str, np.ndarray]:
    path = receipt(source_entry["path"], source_entry.get("sha256"))["path"]
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if "basis" not in data:
        raise KeyError(f"Source archive lacks basis: {path}")
    if "pca_basis" in data:
        raise KeyError(f"Source archive already has pca_basis: {path}")
    pca_basis = data["basis"].copy()
    data["pca_basis"] = pca_basis
    data["basis"] = group_u.astype(np.float32)
    data["_phase_period"] = np.asarray([source_entry["phase_period"]], dtype=np.int64)
    return data


def train_window(data: dict[str, np.ndarray]) -> tuple[int, int]:
    if "train_start" not in data or "train_end" not in data:
        raise KeyError("train_start/train_end are required for v13 grouping")
    start = int(np.asarray(data["train_start"]).reshape(-1)[0])
    end = int(np.asarray(data["train_end"]).reshape(-1)[0])
    if not (0 <= start < end <= data["x"].shape[0]):
        raise ValueError(f"Invalid TRAIN bounds: {start}, {end}, rows={data['x'].shape[0]}")
    return start, end


def pairwise_distances(x: np.ndarray, finite: np.ndarray, columns: list[str]) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    c = x.shape[1]
    distances = np.zeros((c, c), dtype=np.float64)
    table: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, Any]] = []
    for i in range(c):
        for j in range(i + 1, c):
            mask = finite[:, i].astype(bool) & finite[:, j].astype(bool)
            n = int(mask.sum())
            reason = None
            rho = 0.0
            if n >= 2:
                xi = x[mask, i].astype(np.float64)
                xj = x[mask, j].astype(np.float64)
                xi = xi - xi.mean()
                xj = xj - xj.mean()
                denom = float(np.sqrt(np.sum(xi * xi) * np.sum(xj * xj)))
                if denom > 0.0:
                    rho = float(np.clip(np.sum(xi * xj) / denom, -1.0, 1.0))
                else:
                    reason = "zero_pair_variance"
            else:
                reason = "fewer_than_two_common_train_observations"
            distance = math.sqrt(max(0.0, 2.0 - 2.0 * rho))
            distances[i, j] = distances[j, i] = distance
            row = {
                "left": int(i),
                "right": int(j),
                "left_column": columns[i],
                "right_column": columns[j],
                "common_train_observations": n,
                "rho": rho,
                "distance": distance,
                "status": "fallback" if reason else "observed",
            }
            if reason:
                row["reason"] = reason
                missing_pairs.append(row)
            table.append(row)
    return distances, table, missing_pairs


def canonical_cluster(cluster: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(cluster))


def cluster_distance(a: tuple[int, ...], b: tuple[int, ...], distances: np.ndarray) -> float:
    values = [distances[i, j] for i in a for j in b]
    return float(np.mean(values))


def average_linkage_groups(distances: np.ndarray, k: int) -> list[tuple[int, ...]]:
    clusters: list[tuple[int, ...]] = [(i,) for i in range(distances.shape[0])]
    while len(clusters) > k:
        best_key: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
        best_pair: tuple[int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                left = canonical_cluster(clusters[i])
                right = canonical_cluster(clusters[j])
                key = (cluster_distance(left, right, distances), left, right)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (i, j)
        if best_pair is None:
            raise RuntimeError("No clusters left to merge")
        i, j = best_pair
        merged = canonical_cluster(clusters[i] + clusters[j])
        clusters = [cluster for n, cluster in enumerate(clusters) if n not in best_pair]
        clusters.append(merged)
        clusters = sorted(clusters, key=lambda cluster: canonical_cluster(cluster))
    return sorted((canonical_cluster(cluster) for cluster in clusters), key=lambda cluster: min(cluster))


def group_basis(groups: list[tuple[int, ...]], c: int, k: int) -> np.ndarray:
    if len(groups) != k:
        raise ValueError(f"Expected K={k} groups, got {len(groups)}")
    seen = sorted(i for group in groups for i in group)
    if seen != list(range(c)):
        raise ValueError("Groups must cover every channel exactly once")
    u = np.zeros((c, k), dtype=np.float32)
    for j, group in enumerate(groups):
        value = 1.0 / math.sqrt(len(group))
        for i in group:
            u[i, j] = value
    gram = u.T @ u
    if not np.allclose(gram, np.eye(k, dtype=np.float32), atol=1e-6, rtol=0):
        raise ValueError("Group basis failed orthonormality check")
    if not np.all((u > 0).sum(axis=1) == 1):
        raise ValueError("Each channel must belong to one group")
    return u


def save_npz_new(path: str | Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path = resolve(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return receipt(path)


def extract_initial_g(dataset: str, seed: int, entry: dict[str, Any]) -> dict[str, Any]:
    import torch

    destination = INITIAL_CACHE / f"{dataset}_s{seed}_initial_g.pt"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite initial G copy: {destination}")
    source = entry["source"]
    receipt(source["path"], source["sha256"])
    saved = torch.load(source["path"], map_location="cpu", weights_only=False)
    state = saved.get("state", saved)
    tensors = {key: state[key].detach().cpu().clone() for key in INITIAL_G_KEYS}
    if tensors["residual.down.weight"].shape != (32, 512):
        raise ValueError(f"{dataset} seed {seed} down weight shape changed")
    if tensors["residual.up.weight"].shape != (48, 32):
        raise ValueError(f"{dataset} seed {seed} up weight shape changed")
    if torch.count_nonzero(tensors["residual.up.weight"]):
        raise ValueError(f"{dataset} seed {seed} initial up weight is not zero")
    tensor_hashes = {
        key: hashlib.sha256(tensors[key].contiguous().numpy().view(np.uint8)).hexdigest()
        for key in INITIAL_G_KEYS
    }
    if tensor_hashes != entry["tensor_hashes"]:
        raise RuntimeError(f"{dataset} seed {seed} initial G tensor hashes changed")
    payload = {
        "schema": "v13_initial_g_1",
        "dataset": dataset,
        "seed": int(seed),
        "state": tensors,
        "tensor_hashes": tensor_hashes,
        "source": copy.deepcopy(source),
        "source_result": copy.deepcopy(entry["source_result"]),
        "state_keys": list(INITIAL_G_KEYS),
        "scope": "untrained original LEVEL G only; encoder/decoder are intentionally not reused across the new basis",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    item = receipt(destination)
    item.update(
        {
            "schema": "v13_initial_g_1",
            "state_keys": list(INITIAL_G_KEYS),
            "tensor_hashes": tensor_hashes,
            "source": copy.deepcopy(source),
            "source_result": copy.deepcopy(entry["source_result"]),
        }
    )
    return item


def prepare() -> dict[str, Any]:
    protocol = load_protocol()
    reuse = read_json(REUSE_MANIFEST) if REUSE_MANIFEST.exists() else build_reuse_manifest()
    if reuse["dataset_order"] != list(DATASET_ORDER):
        raise RuntimeError("Dataset order changed")
    datasets: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        unit = protocol["units"][dataset]
        reused = reuse["datasets"][dataset]
        columns = reused["columns"]
        source_trainval = copy.deepcopy(reused["data"]["trainval"])
        source_trainval["phase_period"] = unit["phase_period"]
        source_test = copy.deepcopy(reused["data"]["test"])
        source_test["phase_period"] = unit["phase_period"]

        trainval = copy_archive_with_group_basis(source_trainval, np.empty((unit["C"], unit["K"]), dtype=np.float32))
        if [str(v) for v in trainval["columns"]] != columns:
            raise RuntimeError(f"{dataset} source column order differs from contract")
        start, end = train_window(trainval)
        distances, pair_table, missing_pairs = pairwise_distances(
            trainval["x"][start:end], trainval["finite"][start:end], columns
        )
        groups = average_linkage_groups(distances, unit["K"])
        u = group_basis(groups, unit["C"], unit["K"])

        trainval["basis"] = u
        test = copy_archive_with_group_basis(source_test, u)
        if [str(v) for v in test["columns"]] != columns:
            raise RuntimeError(f"{dataset} test column order differs from contract")
        if not np.array_equal(test["pca_basis"], trainval["pca_basis"]):
            raise RuntimeError(f"{dataset} TRAIN/TEST original PCA bases differ")

        trainval_path = CACHE / dataset / "trainval.npz"
        test_path = CACHE / dataset / "test.npz"
        trainval_receipt = save_npz_new(trainval_path, trainval)
        test_receipt = save_npz_new(test_path, test)
        initial_g = {
            str(seed): extract_initial_g(dataset, seed, reused["initial_g_sources"][str(seed)])
            for seed in SEEDS
        }
        basis_sha = array_sha256(u)
        contract = {
            "schema": "v13_group_basis_data_contract_1",
            "dataset": dataset,
            "C": unit["C"],
            "K": unit["K"],
            "columns": columns,
            "source_contract": reused["source_contract"],
            "source_generation": reused["source_generation"],
            "source_trainval": reused["data"]["trainval"],
            "source_test": reused["data"]["test"],
            "trainval": trainval_receipt,
            "test": test_receipt,
            "initial_g": initial_g,
            "basis": {
                "sha256": basis_sha,
                "groups": [
                    {
                        "group": j,
                        "indices": list(group),
                        "columns": [columns[i] for i in group],
                        "size": len(group),
                        "coefficient": float(1.0 / math.sqrt(len(group))),
                    }
                    for j, group in enumerate(groups)
                ],
                "policy": protocol["grouping"],
                "train_bounds": {"start": start, "end": end},
                "pair_table": pair_table,
                "train_missing_pairs": missing_pairs,
                "gram_max_abs_error": float(np.max(np.abs((u.T @ u) - np.eye(unit["K"])))),
                "original_pca_basis_preserved_as": "pca_basis",
            },
            "selection_and_evaluation_reuse": {
                "references": reused["references"],
                "reuse_manifest": receipt(REUSE_MANIFEST) if REUSE_MANIFEST.exists() else None,
            },
            "test_statistics_reported": False,
        }
        contract_path = HERE / f"data_contract_group_basis_{dataset}.json"
        write_json_new(contract_path, contract)
        datasets[dataset] = {
            "contract": receipt(contract_path),
            "C": unit["C"],
            "K": unit["K"],
            "columns": columns,
            "block_origins": unit["block_origins"],
            "phase_period": unit["phase_period"],
            "trainval": trainval_receipt,
            "test": test_receipt,
            "initial_g": initial_g,
            "basis_sha256": basis_sha,
            "groups": contract["basis"]["groups"],
            "train_missing_pairs": missing_pairs,
            "pca_basis_preserved": True,
        }
    manifest = {
        "schema": "v13_group_basis_data_manifest_1",
        "datasets": datasets,
        "reuse_manifest": receipt(REUSE_MANIFEST) if REUSE_MANIFEST.exists() else None,
        "protocol": receipt(HERE / "protocol.json"),
        "arrays": {
            "basis": "new TRAIN-only disjoint groupU",
            "pca_basis": "original selected PCA basis preserved for restore/evaluation reference checks",
            "required_runtime_keys": [
                "x",
                "finite",
                "basis",
                "columns",
                "train_origins",
                "val_origins",
                "test_a_origins",
                "test_b_origins",
                "_phase_period",
            ],
        },
        "no_test_statistics_or_scores": True,
    }
    write_json_new(DATA_MANIFEST, manifest)
    return manifest


def write_reuse_manifest() -> dict[str, Any]:
    value = build_reuse_manifest()
    write_json_new(REUSE_MANIFEST, value)
    return value


def prepare_with_job(job_label: str, reserve_s: float) -> dict[str, Any]:
    import sys

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from runtime_v13 import Job

    metadata = {
        "purpose": "v13 group-basis data preparation",
        "fit_count": 0,
        "datasets": list(DATASET_ORDER),
        "produces": [
            "data_manifest.json",
            "data_contract_group_basis_{dataset}.json",
            ".cache/tsfm_peft_group_basis_v13_20261001/data/{dataset}/{trainval,test}.npz",
            ".cache/tsfm_peft_group_basis_v13_20261001/initial_g/{dataset}_s{seed}_initial_g.pt",
        ],
    }
    with Job("cpu_analysis", job_label, reserve_s=reserve_s, metadata=metadata) as job:
        value = prepare()
        job.heartbeat("v13 group-basis data manifest prepared")
        return value


def describe() -> dict[str, Any]:
    reuse_exists = REUSE_MANIFEST.exists()
    value = read_json(REUSE_MANIFEST) if reuse_exists else build_reuse_manifest()
    return {
        "schema": "v13_prepare_describe_1",
        "reuse_manifest_exists": reuse_exists,
        "dataset_order": value["dataset_order"],
        "families": value["families"],
        "fit_plan": value["fit_plan"],
        "outputs_if_run": {
            "data_manifest": rel(DATA_MANIFEST),
            "dataset_contracts": [rel(HERE / f"data_contract_group_basis_{dataset}.json") for dataset in DATASET_ORDER],
            "data_archives": [rel(CACHE / dataset / name) for dataset in DATASET_ORDER for name in ("trainval.npz", "test.npz")],
            "initial_g": [rel(INITIAL_CACHE / f"{dataset}_s{seed}_initial_g.pt") for dataset in DATASET_ORDER for seed in SEEDS],
        },
        "execution_note": "describe/write-reuse-manifest do not compute new bases or touch TEST predictions.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "describe", "write-reuse-manifest"),
        default="prepare",
    )
    parser.add_argument("--job", help="Required for prepare: unique runtime_v13 cpu_analysis job label.")
    parser.add_argument("--reserve-s", type=float, help="Required for prepare: cpu_analysis seconds to reserve.")
    args = parser.parse_args()
    if args.action == "describe":
        value = describe()
    elif args.action == "write-reuse-manifest":
        value = write_reuse_manifest()
    else:
        if not args.job or args.reserve_s is None:
            raise SystemExit("prepare requires --job and --reserve-s so it is ledger-reserved")
        value = prepare_with_job(args.job, args.reserve_s)
    print(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
