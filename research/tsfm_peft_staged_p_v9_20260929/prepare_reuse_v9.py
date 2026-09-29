"""Prepare the v9 reuse manifest without loading models or arrays.

This script only copies the already approved v8 data contracts byte-for-byte and
records hashed references to the frozen LEVEL parents, v8 head initializations,
and prior evaluation predictions needed by v9.  It deliberately avoids
torch.load, np.load, inference, scoring, benchmarking, and downloads.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from runtime_v9 import HERE, ROOT, Job, digest, save_json


V6 = ROOT / "research/tsfm_peft_level_confirmation_v6_20260928"
V7 = ROOT / "research/tsfm_peft_practical_controls_v7_20260928"
V8 = ROOT / "research/tsfm_peft_subspace_correction_v8_20260929"

SEEDS = (92601, 92602)

DATASETS = {
    "robin": {
        "selected": V6 / "selected.json",
        "contract": V8 / "data_contract_robin.json",
        "evaluation": V8 / "robin_eval01.json",
        "parent_model_source": V6 / "model_v6.py",
        "parent_initial_manifest": V6 / "initial_manifest.json",
        "parent_initial_record": ("records",),
        "head_initial_manifest": V8 / "initial_manifest_robin.json",
        "scope": "v6/v7/v8 exposed development data; no independent TEST claim",
    },
    "jena": {
        "selected": V7 / "selected_jena.json",
        "contract": V8 / "data_contract_jena.json",
        "evaluation": V8 / "jena_eval01.json",
        "parent_model_source": V7 / "model_v7.py",
        "parent_initial_manifest": V7 / "initial_manifest_jena.json",
        "parent_initial_record": ("records", "edg"),
        "head_initial_manifest": V8 / "initial_manifest_jena.json",
        "scope": "v7/v8 exposed fixed other-domain extension; no independent TEST claim",
    },
}

BASELINE_ROLES = {
    "level_res",
    "pq_split",
    "raw64",
    "f0",
    "lora",
    "direct_nlinear",
}


def read_json(path: Path) -> Any:
    path = require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> Path:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def artifact(path: Path) -> dict[str, Any]:
    path = require_file(path)
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
    }


def copy_byte_identical(src: Path, dst: Path) -> dict[str, Any]:
    src = require_file(src)
    payload = src.read_bytes()
    if dst.exists() and dst.read_bytes() != payload:
        raise RuntimeError(f"Refusing to overwrite non-identical data contract: {dst}")
    dst.write_bytes(payload)
    if dst.read_bytes() != payload:
        raise RuntimeError(f"Byte-identical copy check failed: {dst}")
    return artifact(dst)


def nested_record(data: dict[str, Any], keys: tuple[str, ...], seed: int) -> dict[str, Any]:
    current: Any = data
    for key in keys:
        current = current[key]
    record = copy.deepcopy(current[str(seed)])
    require_file(Path(record["path"]))
    if digest(Path(record["path"])) != record["sha256"]:
        raise RuntimeError(f"Initial artifact hash mismatch: {record['path']}")
    return record


def selected_level_runs(dataset: str, config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    selected = read_json(config["selected"])
    level = selected["selected"]["level_res"]
    all_runs = {run["id"]: run for run in selected["all_neural_runs"]}
    by_seed: dict[int, dict[str, Any]] = {}
    for run_id, checkpoint in zip(level["run_ids"], level["checkpoints"]):
        run = all_runs[run_id]
        spec = run["spec"]
        seed = int(spec["seed"])
        if spec["dataset"] != dataset or spec["family"] != "level_res":
            raise RuntimeError(f"Selected parent is not the approved LEVEL_RES: {run_id}")
        if run["checkpoint"] != checkpoint:
            raise RuntimeError(f"Selected checkpoint mismatch for {run_id}")
        if digest(Path(run["checkpoint"])) != run["checkpoint_sha256"]:
            raise RuntimeError(f"Parent checkpoint hash mismatch for {run_id}")
        if digest(Path(run["result_path"])) != run["result_sha256"]:
            raise RuntimeError(f"Parent result hash mismatch for {run_id}")
        by_seed[seed] = run
    if set(by_seed) != set(SEEDS):
        raise RuntimeError(f"Missing selected LEVEL_RES seeds for {dataset}: {sorted(by_seed)}")
    return by_seed


def evaluation_prediction_index(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for model_id, model in evaluation["models"].items():
        prediction = model.get("prediction")
        if not prediction:
            continue
        prediction_path = Path(prediction["path"])
        require_file(prediction_path)
        if digest(prediction_path) != prediction["sha256"]:
            raise RuntimeError(f"Prediction hash mismatch: {model_id}")
        output[model_id] = {
            "role": model.get("role"),
            "kind": model.get("kind"),
            "prediction": copy.deepcopy(prediction),
        }
    return output


def parent_receipts(dataset: str, config: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    level_runs = selected_level_runs(dataset, config)
    parent_initial = read_json(config["parent_initial_manifest"])
    head_initial = read_json(config["head_initial_manifest"])
    predictions = evaluation_prediction_index(evaluation)
    model_source = artifact(config["parent_model_source"])
    parents: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        run = level_runs[seed]
        parent_prediction = predictions.get(run["id"])
        if parent_prediction is None:
            raise RuntimeError(f"Selected parent TEST prediction missing from v8 evaluation: {run['id']}")
        parents[str(seed)] = {
            "id": run["id"],
            "dataset": dataset,
            "seed": seed,
            "family": "level_res",
            "ed_mode": run["spec"]["ed_mode"],
            "selected_epoch": run.get("selected_epoch"),
            "best_val_mse": run.get("best_val_mse"),
            "checkpoint": {
                "path": run["checkpoint"],
                "sha256": run["checkpoint_sha256"],
                "bytes": Path(run["checkpoint"]).stat().st_size,
            },
            "model_source": model_source,
            "result": {
                "path": run["result_path"],
                "sha256": run["result_sha256"],
                "bytes": Path(run["result_path"]).stat().st_size,
            },
            "best_val_prediction": copy.deepcopy(run.get("best_val_prediction")),
            "parent_initial": nested_record(parent_initial, config["parent_initial_record"], seed),
            "head_initial": nested_record(head_initial, ("records", "edg"), seed),
            "test_prediction": parent_prediction["prediction"],
        }
    return parents


def baseline_references(evaluation: dict[str, Any]) -> dict[str, Any]:
    roles = {role: ids for role, ids in evaluation["roles"].items() if role in BASELINE_ROLES}
    prediction_index = evaluation_prediction_index(evaluation)
    predictions = {
        model_id: prediction_index[model_id]
        for ids in roles.values()
        for model_id in ids
        if model_id in prediction_index
    }
    return {
        "roles": roles,
        "predictions": predictions,
        "note": "Prior predictions are referenced by path and hash only; no arrays are copied.",
    }


def source_files() -> dict[str, str]:
    candidates = [
        HERE / "PLAN.md",
        HERE / "protocol.json",
        HERE / "runtime_v9.py",
        HERE / "model_v9.py",
        HERE / "prepare_reuse_v9.py",
        V6 / "selected.json",
        V6 / "initial_manifest.json",
        V6 / "model_source.json",
        V6 / "model_v6.py",
        V7 / "selected_jena.json",
        V7 / "initial_manifest_jena.json",
        V7 / "model_v7.py",
        V8 / "reuse_manifest.json",
        V8 / "preservation.json",
        V8 / "initial_manifest_robin.json",
        V8 / "initial_manifest_jena.json",
        V8 / "data_contract_robin.json",
        V8 / "data_contract_jena.json",
        V8 / "robin_eval01.json",
        V8 / "jena_eval01.json",
    ]
    return {
        path.relative_to(ROOT).as_posix(): digest(path)
        for path in candidates
        if path.exists()
    }


def build_manifest() -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    parents: dict[str, Any] = {}
    baselines: dict[str, Any] = {}

    for dataset, config in DATASETS.items():
        destination = HERE / f"data_contract_{dataset}.json"
        source_contract = artifact(config["contract"])
        evaluation = read_json(config["evaluation"])
        if evaluation["dataset"] != dataset:
            raise RuntimeError(f"Evaluation dataset mismatch: {config['evaluation']}")
        evaluation_artifact = artifact(config["evaluation"])
        datasets[dataset] = {
            "contract": str(destination.resolve()),
            "contract_sha256": source_contract["sha256"],
            "copied_contract": {
                "path": str(destination.resolve()),
                "sha256": source_contract["sha256"],
                "bytes": source_contract["bytes"],
                "byte_identical_to": source_contract["path"],
            },
            "source_contract": source_contract,
            "evaluation": evaluation_artifact["path"],
            "evaluation_sha256": evaluation_artifact["sha256"],
            "evaluation_artifact": evaluation_artifact,
            "scope": config["scope"],
        }
        parents[dataset] = parent_receipts(dataset, config, evaluation)
        baselines[dataset] = baseline_references(evaluation)

    backbone_source = artifact(V6 / "model_source.json")
    backbone = read_json(V6 / "model_source.json")
    manifest = {
        "schema_version": 1,
        "stage": "v9_staged_p_reuse",
        "created_by": "prepare_reuse_v9.py",
        "job_label": "prepare_reuse01",
        "no_model_or_array_load": True,
        "no_new_download": True,
        "no_backbone_raw_or_array_copy": True,
        "datasets": datasets,
        "parents": parents,
        "baselines": baselines,
        "backbone": {
            "source_receipt": backbone_source,
            "revision": backbone["revision"],
            "snapshot": backbone["snapshot"],
            "files": backbone["files"],
        },
        "source": source_files(),
    }
    return manifest


def main() -> None:
    with Job("prepare_reuse01", category="data_prepare", reserve_seconds=120,
             metadata={"purpose": "copy data contracts and seal v9 reuse manifest"}):
        manifest = build_manifest()
        for dataset, config in DATASETS.items():
            copy_byte_identical(config["contract"], HERE / f"data_contract_{dataset}.json")
        save_json(HERE / "reuse_manifest.json", manifest)


if __name__ == "__main__":
    main()
