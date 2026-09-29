from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from runtime_v9 import (
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

CONTEXT = 512
HORIZON = 48
PERIODS = ("test_a", "test_b")
SEEDS = (92601, 92602)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 9262026
V8 = ROOT / "research/tsfm_peft_subspace_correction_v8_20260929"
PRIOR_EVALUATIONS = {
    "robin": V8 / "robin_eval01.json",
    "jena": V8 / "jena_eval01.json",
}
SPACE_ROLES = {"staged_p", "staged_raw", "staged_p_alpha", "staged_raw_alpha", "level_res", "pq_split", "raw64", "level_raw", "lora", "direct_nlinear", "f0"}
Q_PARITY_ROLES = {"staged_p", "staged_p_alpha", "staged_raw", "staged_raw_alpha", "level_res"}
Q_PARITY_ATOL = 1e-5
Q_PARITY_RTOL = 1e-4


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    return resolve(path).resolve().relative_to(ROOT.resolve()).as_posix()


def receipt(path: str | Path) -> dict[str, str]:
    resolved = resolve(path)
    return {"path": str(resolved), "sha256": digest(resolved)}


def check_receipt(item: dict[str, Any]) -> Path:
    path = resolve(item["path"])
    actual = digest(path)
    if actual != item["sha256"]:
        raise ValueError(f"Sealed input changed: {path} expected {item['sha256']} got {actual}")
    return path


def check_checkpoint_receipt(item: dict[str, Any]) -> Path:
    path = resolve(item["checkpoint"])
    actual = digest(path)
    if actual != item["checkpoint_sha256"]:
        raise ValueError(f"Sealed checkpoint changed: {path} expected {item['checkpoint_sha256']} got {actual}")
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
    if not name or any(char not in allowed for char in name):
        raise ValueError("Use a unique alphanumeric/underscore/hyphen name")


def configure_numeric_runtime() -> None:
    try:
        import torch
    except Exception:
        return
    torch.set_num_threads(4)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def protocol() -> dict[str, Any]:
    return read_json(HERE / "protocol.json")


def dataset_config(dataset: str) -> dict[str, Any]:
    cfg = protocol()["datasets"][dataset]
    if int(cfg["channels"]) not in (17, 21):
        raise ValueError("Only approved v9 datasets are allowed")
    return cfg


def normal_role(value: str) -> str:
    aliases = {
        "pq": "pq_split",
        "pqsplit": "pq_split",
        "raw_64": "raw64",
        "raw": "level_raw",
        "raw32": "level_raw",
        "level_tsfm_raw": "level_raw",
        "res": "old_res",
        "tsfm_res": "old_res",
        "level_tsfm_res": "level_res",
        "tsfm_level_res": "level_res",
        "linear_res": "level_linear_res",
        "direct": "direct_nlinear",
        "nlinear": "direct_nlinear",
        "direct_linear": "direct_nlinear",
    }
    return aliases.get(str(value), str(value))


def validate_selection_test_policy(selection: dict[str, Any]) -> dict[str, Any]:
    if selection.get("test_used_for_selection") is False:
        return {"status": "explicit_false", "test_used_for_selection": False}
    if "test_used_for_selection" in selection:
        raise ValueError("Selection metadata says TEST may have been used for selection")
    blocked: list[str] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_text = str(key).lower()
                if key_text.startswith("test") or key_text in {"test_score", "test_mse", "test_mae", "test_metric", "test_prediction"}:
                    blocked.append(f"{path}.{key}" if path else str(key))
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                walk(value, f"{path}[{idx}]")

    walk({key: value for key, value in selection.items() if key != "test_used_for_selection"}, "")
    if blocked:
        raise ValueError(f"Selection omits test_used_for_selection=false and contains TEST-like keys: {blocked[:10]}")
    if "selected" not in selection or "all_neural_runs" not in selection:
        raise ValueError("v9 selection must be select-only selected/all_neural_runs schema")
    return {
        "status": "missing_but_select_only_schema_without_test_keys",
        "test_used_for_selection": False,
        "note": "Accepted because the selection file has the approved select-only schema and no TEST-like keys.",
    }


def _rows_from_selection(selection: dict[str, Any], dataset: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    run_map = {row["id"]: dict(row) for row in selection.get("all_neural_runs", [])}
    rows: list[dict[str, Any]] = []
    roles: dict[str, list[str]] = {}
    for family, item in selection.get("selected", {}).items():
        role = normal_role(family)
        if role not in {"staged_p", "staged_raw", "staged_p_alpha", "staged_raw_alpha"}:
            raise ValueError(f"v9 selection contains an unapproved new family: {family}")
        ids = list(item.get("run_ids", []))
        if len(ids) != len(SEEDS):
            raise ValueError(f"v9 selected family {family} must contain exactly two seed runs")
        role_ids: list[str] = []
        for identifier in ids:
            if identifier not in run_map:
                raise ValueError(f"selected.{family} references missing all_neural_runs id {identifier}")
            base = copy.deepcopy(run_map[identifier])
            if role.endswith("_alpha"):
                alpha = copy.deepcopy(base.get("alpha_selected") or {})
                if not alpha:
                    raise ValueError(f"Alpha role {family} requires alpha_selected in {identifier}")
                derived_id = f"{identifier}_alpha"
                row = dict(base)
                row["id"] = derived_id
                row["role"] = role
                row["kind"] = "checkpoint"
                row["base_run_id"] = identifier
                row["checkpoint"] = alpha["checkpoint"]
                row["checkpoint_sha256"] = alpha["checkpoint_sha256"]
                row["best_val_prediction"] = alpha["val_prediction"]
                row["best_val_mse"] = alpha["val_mse"]
                row["alpha_selected"] = alpha
                if "result_path" in base:
                    row["result_path"] = base["result_path"]
                if "result_sha256" in base:
                    row["result_sha256"] = base["result_sha256"]
                spec = dict(row.get("spec") or {})
                spec["family"] = role
                spec["alpha"] = alpha["alpha"]
                spec["base_family"] = normal_role(str(base.get("role") or spec.get("family") or family).replace("_alpha", ""))
                row["spec"] = spec
            else:
                row = dict(base)
                row.pop("alpha_selected", None)
                row.setdefault("role", role)
                row.setdefault("kind", "checkpoint")
                spec = dict(row.get("spec") or {})
                spec["family"] = role
                spec.setdefault("alpha", 1.0)
                row["spec"] = spec
            spec = dict(row.get("spec") or {})
            spec.setdefault("dataset", dataset)
            if spec["dataset"] != dataset:
                raise ValueError(f"Selected run {identifier} belongs to {spec['dataset']} not {dataset}")
            row["spec"] = spec
            role_ids.append(row["id"])
            rows.append(row)
        roles[role] = role_ids
    return rows, roles


def load_selection(path: Path, dataset: str) -> dict[str, Any]:
    selection = read_json(path)
    if selection.get("dataset", dataset) != dataset:
        raise ValueError(f"Selection dataset mismatch: {selection.get('dataset')} vs {dataset}")
    selection["_selection_test_policy"] = validate_selection_test_policy(selection)
    rows, roles = _rows_from_selection(selection, dataset)
    selection["models"] = rows
    selection["roles"] = roles
    return selection


def model_entry(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    item = dict(row)
    spec = dict(item.get("spec") or {})
    spec["family"] = normal_role(str(spec.get("family") or item.get("family") or item.get("role")))
    spec.setdefault("dataset", dataset)
    if spec["dataset"] != dataset:
        raise ValueError(f"Model {item.get('id')} belongs to {spec['dataset']} not {dataset}")
    item["spec"] = spec
    item["id"] = item.get("id") or spec.get("id")
    if not item["id"]:
        raise ValueError("Every selected model needs an id")
    item["role"] = normal_role(str(item.get("role") or spec["family"]))
    item["kind"] = item.get("kind") or ("stored_prediction" if "prediction" in item else "checkpoint")
    if item["kind"] == "checkpoint":
        if "checkpoint" not in item or "checkpoint_sha256" not in item:
            raise ValueError(f"Checkpoint model missing receipt: {item['id']}")
        check_receipt({"path": item["checkpoint"], "sha256": item["checkpoint_sha256"]})
        if "result_path" in item and "result_sha256" in item:
            check_receipt({"path": item["result_path"], "sha256": item["result_sha256"]})
        if "best_val_prediction" in item:
            check_receipt(item["best_val_prediction"])
        if "alpha_selected" in item:
            alpha = item["alpha_selected"]
            check_receipt({"path": alpha["checkpoint"], "sha256": alpha["checkpoint_sha256"]})
            check_receipt(alpha["val_prediction"])
            check_receipt({"path": alpha["selection_path"], "sha256": alpha["selection_sha256"]})
            check_checkpoint_receipt(alpha["best_trained_checkpoint"])
        if "best_val_mse" in item and not math.isfinite(float(item["best_val_mse"])):
            raise ValueError(f"Nonfinite VAL loss: {item['id']}")
    elif item["kind"] == "stored_prediction":
        check_receipt(item["prediction"])
    else:
        raise ValueError(f"Unknown model kind {item['kind']} for {item['id']}")
    return item


def collect_selection_models(selection: dict[str, Any], dataset: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    models = [model_entry(row, dataset) for row in selection["models"]]
    if len({row["id"] for row in models}) != len(models):
        raise ValueError("Model ids must be unique")
    roles: dict[str, list[str]] = {}
    for row in models:
        roles.setdefault(row["role"], []).append(row["id"])
    for role in ("staged_p", "staged_raw", "staged_p_alpha", "staged_raw_alpha"):
        if role not in roles or len(roles[role]) != len(SEEDS):
            raise ValueError(f"v9 seal requires exactly two selected {role} seed checkpoints")
    return models, roles


def collect_prior_predictions(dataset: str) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, str]]:
    source_path = PRIOR_EVALUATIONS[dataset]
    evaluation = read_json(source_path)
    if evaluation.get("status") != "complete" or evaluation.get("dataset") != dataset:
        raise ValueError(f"{dataset} v8 evaluation must remain complete for prediction reuse")
    source = {"path": rel(source_path), "sha256": digest(source_path)}
    models: list[dict[str, Any]] = []
    for identifier, row in evaluation["models"].items():
        pred = row["prediction"]
        check_receipt(pred)
        spec = dict(row.get("spec") or {})
        spec["family"] = normal_role(str(row.get("role") or spec.get("family") or identifier))
        spec["dataset"] = dataset
        models.append(
            {
                "id": identifier,
                "role": normal_role(str(row.get("role") or spec["family"])),
                "kind": "stored_prediction",
                "spec": spec,
                "prediction": pred,
                "source_evaluation": source,
                "original_prior_kind": row.get("kind"),
                "scores_from_prior_not_reused_for_v9_scoring": True,
            }
        )
    roles = {normal_role(key): list(ids) for key, ids in evaluation["roles"].items()}
    return models, roles, source


def default_comparisons(roles: dict[str, list[str]], dataset: str) -> list[dict[str, Any]]:
    pairs = [
        ("staged_p_vs_level_res", "staged_p", "level_res", "basic staged P correction beyond the frozen selected LEVEL parent", True),
        ("staged_p_vs_staged_raw", "staged_p", "staged_raw", "P-restricted side correction versus matched RAW side correction", True),
        ("staged_p_vs_pq_split", "staged_p", "pq_split", "staged P procedure versus v8 jointly fitted P/Q split", False),
        ("staged_p_vs_raw64", "staged_p", "raw64", "staged P procedure versus v8 shared RAW64", False),
        ("staged_p_vs_direct_nlinear", "staged_p", "direct_nlinear", "staged P procedure versus direct NLinear baseline", False),
        ("staged_p_vs_f0", "staged_p", "f0", "remaining accuracy gap to uncompressed frozen TSFM", False),
        ("staged_p_vs_lora", "staged_p", "lora", "remaining accuracy gap to uncompressed LoRA", False),
        ("staged_p_alpha_vs_level_res", "staged_p_alpha", "level_res", "VAL-selected alpha staged P versus parent fallback", False),
        ("staged_p_alpha_vs_staged_p", "staged_p_alpha", "staged_p", "VAL-selected alpha staged P versus basic alpha-one staged P", False),
        ("staged_p_alpha_vs_staged_raw_alpha", "staged_p_alpha", "staged_raw_alpha", "VAL-selected alpha P restriction versus matched RAW alpha policy", False),
        ("staged_p_alpha_vs_direct_nlinear", "staged_p_alpha", "direct_nlinear", "VAL-selected alpha staged P versus direct NLinear baseline", False),
        ("staged_p_alpha_vs_f0", "staged_p_alpha", "f0", "VAL-selected alpha staged P remaining gap to F0", False),
        ("staged_p_alpha_vs_lora", "staged_p_alpha", "lora", "VAL-selected alpha staged P remaining gap to LoRA", False),
        ("staged_raw_vs_level_res", "staged_raw", "level_res", "basic matched RAW side correction beyond the frozen selected LEVEL parent", False),
        ("staged_raw_vs_raw64", "staged_raw", "raw64", "staged RAW correction versus v8 RAW64", False),
        ("staged_raw_alpha_vs_level_res", "staged_raw_alpha", "level_res", "VAL-selected alpha staged RAW versus parent fallback", False),
        ("staged_raw_alpha_vs_staged_raw", "staged_raw_alpha", "staged_raw", "VAL-selected alpha staged RAW versus basic alpha-one staged RAW", False),
        ("pq_split_vs_raw64", "pq_split", "raw64", "prior v8 independent P/Q correction versus parameter-matched shared RAW64", False),
    ]
    output = []
    for name, left, right, question, primary in pairs:
        if left in roles and right in roles:
            output.append(
                {
                    "name": f"{dataset}_{name}",
                    "left": roles[left],
                    "right": roles[right],
                    "primary": primary,
                    "question": question,
                }
            )
    return output


def selection_readiness(path: Path, dataset: str) -> dict[str, Any]:
    selection = read_json(path)
    if selection.get("dataset") != dataset:
        raise ValueError(f"{path.name} dataset mismatch: {selection.get('dataset')} vs {dataset}")
    if selection.get("test_used_for_selection") is not False:
        raise ValueError(f"{path.name} must explicitly record test_used_for_selection=false")
    expected_roles = {"staged_p", "staged_raw", "staged_p_alpha", "staged_raw_alpha"}
    selected = selection.get("selected") or {}
    if set(selected) != expected_roles:
        raise ValueError(f"{path.name} must contain exactly the four approved v9 roles: {sorted(selected)}")
    run_map = {run.get("id"): run for run in selection.get("all_neural_runs", [])}
    if len(run_map) != 4:
        raise ValueError(f"{path.name} must contain exactly four real fit receipts")
    complete_runs: dict[str, dict[str, Any]] = {}
    for run_id, run in run_map.items():
        if not run_id:
            raise ValueError(f"{path.name} has a run without id")
        spec = run.get("spec") or {}
        if spec.get("dataset") != dataset or spec.get("family") not in {"staged_p", "staged_raw"}:
            raise ValueError(f"{path.name} has unexpected fit spec: {run_id}")
        result_path = resolve(run.get("result_path"))
        if digest(result_path) != run.get("result_sha256"):
            raise ValueError(f"{path.name} result hash mismatch: {run_id}")
        result = read_json(result_path)
        if result.get("status") != "complete":
            raise ValueError(f"{path.name} fit result is not complete: {run_id}")
        if result.get("id") != run_id or result.get("spec", {}).get("dataset") != dataset:
            raise ValueError(f"{path.name} fit result identity mismatch: {run_id}")
        for record in (run, run.get("best_trained") or {}, run.get("alpha_selected") or {}):
            if "checkpoint" in record and "checkpoint_sha256" in record:
                check_receipt({"path": record["checkpoint"], "sha256": record["checkpoint_sha256"]})
        if "best_val_prediction" in run:
            check_receipt(run["best_val_prediction"])
        alpha = run.get("alpha_selected")
        if not alpha:
            raise ValueError(f"{path.name} fit receipt missing alpha_selected: {run_id}")
        check_receipt(alpha["val_prediction"])
        check_receipt({"path": alpha["selection_path"], "sha256": alpha["selection_sha256"]})
        check_checkpoint_receipt(alpha["best_trained_checkpoint"])
        complete_runs[run_id] = {"family": spec.get("family"), "seed": spec.get("seed")}
    role_summary: dict[str, Any] = {}
    for role, item in selected.items():
        ids = list(item.get("run_ids", []))
        if len(ids) != 2 or set(ids) - set(run_map):
            raise ValueError(f"{path.name} selected.{role} must reference exactly two known fit ids")
        seeds = sorted(int(run_map[run_id]["spec"]["seed"]) for run_id in ids)
        if seeds != list(SEEDS):
            raise ValueError(f"{path.name} selected.{role} must cover seeds {SEEDS}, got {seeds}")
        if role.endswith("_alpha"):
            checkpoints = [run_map[run_id]["alpha_selected"]["checkpoint"] for run_id in ids]
        else:
            checkpoints = [run_map[run_id]["checkpoint"] for run_id in ids]
        if list(item.get("checkpoints", [])) != checkpoints:
            raise ValueError(f"{path.name} selected.{role} checkpoint list differs from fit receipts")
        role_summary[role] = {"run_ids": ids, "seeds": seeds, "policy": item.get("policy")}
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "dataset": dataset,
        "roles": role_summary,
        "real_fit_results": complete_runs,
        "test_used_for_selection": False,
    }


def active_real_fit_guard() -> dict[str, Any]:
    ledger_path = HERE / "ledger.json"
    ledger = read_json(ledger_path)
    active = [
        {key: row.get(key) for key in ("id", "label", "pid", "category", "status", "metadata")}
        for row in ledger.get("jobs", [])
        if row.get("category") == "real_fit" and row.get("status") == "running"
    ]
    if active:
        raise RuntimeError(f"Do not seal TEST while real fits are still active: {active}")
    return {"ledger": receipt(ledger_path), "active_real_fit_jobs": []}


def require_both_dataset_selections() -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    active_real_fit_guard()
    for dataset in ("robin", "jena"):
        path = HERE / f"selected_{dataset}.json"
        if not path.exists():
            raise FileNotFoundError(f"Both selected_robin.json and selected_jena.json are required before sealing TEST evaluation; missing {path.name}")
        receipts[dataset] = selection_readiness(path, dataset)
    return receipts


def seal_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.selection is None:
        raise ValueError("--selection is required for --seal")
    dataset = args.dataset
    cfg = dataset_config(dataset)
    destination = HERE / f"evaluation_seal_{dataset}.json"
    exposure = HERE / f"test_exposure_{dataset}.json"
    if destination.exists() or exposure.exists():
        raise FileExistsError("Never overwrite a dataset selection seal or retroactive exposure record")
    selection_receipts = require_both_dataset_selections()
    selection_readiness = {
        "active_real_fit_guard_at_seal": {"ledger": receipt(HERE / "ledger.json"), "active_real_fit_jobs": []},
        "selections": selection_receipts,
        "scope": "Both dataset selections exist, cover four completed v9 fit receipts and four selected roles, and no real_fit job was active when sealing.",
    }
    data_contract = HERE / f"data_contract_{dataset}.json"
    check_receipt({"path": data_contract, "sha256": digest(data_contract)})
    selection = load_selection(args.selection, dataset)
    models, roles = collect_selection_models(selection, dataset)
    prior_models, prior_roles, prior_source = collect_prior_predictions(dataset)
    known_ids = {row["id"] for row in models}
    models = [row for row in prior_models if row["id"] not in known_ids] + models
    for role, ids in prior_roles.items():
        roles.setdefault(role, [])
        roles[role] = [*ids, *[identifier for identifier in roles[role] if identifier not in ids]]
    required_reuse = {"level_res", "pq_split", "raw64", "direct_nlinear", "lora", "f0"}
    missing = sorted(required_reuse - set(roles))
    if missing:
        raise ValueError(f"{dataset} v9 seal missing required reused roles: {missing}")
    channels = int(cfg["channels"])
    sealed = {
        "schema": "tsfm_peft_staged_p_v9_evaluation_seal_v1",
        "status": "sealed",
        "dataset": dataset,
        "sealed_utc": time.time(),
        "selection": receipt(args.selection),
        "all_selection_files_required_before_test": selection_receipts,
        "selection_readiness": selection_readiness,
        "selection_test_policy": selection.get("_selection_test_policy"),
        "data_contract": receipt(data_contract),
        "reuse_manifest": receipt(HERE / "reuse_manifest.json"),
        "protocol": receipt(HERE / "protocol.json"),
        "plan": receipt(HERE / "PLAN.md"),
        "approval": receipt(HERE / "APPROVAL.txt"),
        "prior_v8_reuse": prior_source,
        "channels": channels,
        "splits": cfg["splits"],
        "expected_origins": cfg["expected_origins"],
        "evaluation_stride": int(cfg["evaluation_stride"]),
        "interval_minutes": int(cfg["interval_minutes"]),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "block_length_origins": int(cfg["bootstrap_block_origins"]),
        },
        "metric": "TRAIN-standardized raw-space observed-channel macro MSE/MAE; seed losses averaged, not ensembled.",
        "space_metric": "P/Q complete-target-row decomposition using TRAIN PCA basis; no target zero filling before projection.",
        "models": models,
        "roles": roles,
        "comparisons": default_comparisons(roles, dataset),
        "test_used_for_selection": False,
        "test_selection_prohibited": True,
    }
    save_json(destination, sealed)
    return {"seal": rel(destination), "sha256": digest(destination), "models": len(models), "roles": sorted(roles)}


def load_seal(dataset: str) -> dict[str, Any]:
    path = HERE / f"evaluation_seal_{dataset}.json"
    sealed = read_json(path)
    if sealed.get("status") != "sealed" or sealed.get("dataset") != dataset:
        raise ValueError("Evaluation requires a completed dataset-specific seal")
    for key in ("selection", "data_contract", "reuse_manifest", "protocol", "plan", "approval", "prior_v8_reuse"):
        check_receipt(sealed[key])
    for dataset_name, selection in sealed.get("all_selection_files_required_before_test", {}).items():
        check_receipt(selection)
        if dataset_name not in ("robin", "jena"):
            raise ValueError("Unexpected selection guard dataset")
    for row in sealed["models"]:
        if row["kind"] == "checkpoint":
            check_receipt({"path": row["checkpoint"], "sha256": row["checkpoint_sha256"]})
            if "alpha_selected" in row and row["alpha_selected"]:
                alpha = row["alpha_selected"]
                check_receipt({"path": alpha["checkpoint"], "sha256": alpha["checkpoint_sha256"]})
                check_receipt(alpha["val_prediction"])
                check_receipt({"path": alpha["selection_path"], "sha256": alpha["selection_sha256"]})
                check_checkpoint_receipt(alpha["best_trained_checkpoint"])
        elif row["kind"] == "stored_prediction":
            check_receipt(row["prediction"])
        else:
            raise ValueError(f"Unknown model kind {row['kind']}")
    return sealed


def split_pair(sealed: dict[str, Any], period: str) -> tuple[np.datetime64, np.datetime64]:
    pair = sealed["splits"][period]
    if isinstance(pair, dict):
        start, end = pair["start"], pair["end"]
    else:
        start, end = pair
    return np.datetime64(start, "ns"), np.datetime64(end, "ns")


def validate_origins(sealed: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
    dataset = sealed["dataset"]
    cfg = dataset_config(dataset)
    times = np.asarray(data["times"]).astype("datetime64[ns]")
    origins_by_period = {period: np.asarray(data[f"{period}_origins"], dtype=np.int64) for period in PERIODS}
    for period, origins in origins_by_period.items():
        expected = int(cfg["expected_origins"][period])
        stride = int(cfg["evaluation_stride"])
        if len(origins) != expected or len(np.unique(origins)) != expected:
            raise ValueError(f"{dataset}/{period}: origin count mismatch")
        if len(origins) > 1 and not np.all(np.diff(origins) == stride):
            raise ValueError(f"{dataset}/{period}: origin stride mismatch")
        if np.any(origins < CONTEXT) or np.any(origins + HORIZON > len(times)):
            raise ValueError(f"{dataset}/{period}: context/target outside archive")
        start, end = split_pair(sealed, period)
        if np.any(times[origins] < start) or np.any(times[origins + HORIZON - 1] >= end):
            raise ValueError(f"{dataset}/{period}: target crosses split bounds")
    return origins_by_period


def load_prediction_receipt(row: dict[str, Any], origins: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    pred_path = check_receipt(row["prediction"])
    with np.load(pred_path, allow_pickle=False) as archive:
        if not np.array_equal(archive["origins"], origins):
            raise ValueError(f"Stored prediction origins differ for {row['id']}")
        return archive["prediction"].astype(np.float32), row["prediction"]


def checkpoint_predict(row: dict[str, Any], inputs_all: np.ndarray, batch_size: int, job: Job) -> np.ndarray:
    import torch

    module = load_module(HERE / "model_v9.py", "_v9_evaluation_model")
    model = module.restore_model(resolve(row["checkpoint"]), device="cuda").eval()
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(inputs_all), batch_size):
            job.check_limits()
            inputs = torch.as_tensor(inputs_all[offset : offset + batch_size], dtype=torch.float32, device="cuda")
            outputs.append(model(inputs).float().cpu().numpy())
            del inputs
    prediction = np.concatenate(outputs, axis=0).astype(np.float32)
    del outputs, model
    gc.collect()
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated():
        raise RuntimeError("Evaluation retained a prior model or output on CUDA")
    return prediction


def error_terms(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray) -> dict[str, np.ndarray]:
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError(f"Prediction/target/mask shapes differ: {prediction.shape}, {target.shape}, {observed.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError("Predictions and observed targets must be finite")
    delta = np.where(observed, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    return {
        "squared": (delta * delta).sum(axis=1),
        "absolute": np.abs(delta).sum(axis=1),
        "signed": delta.sum(axis=1),
        "count": observed.sum(axis=1),
    }


def scores_from_terms(terms: dict[str, np.ndarray], index: np.ndarray | None = None) -> dict[str, Any]:
    if index is None:
        index = np.arange(len(terms["count"]), dtype=np.int64)
    counts = terms["count"][index].sum(axis=0)
    squared = terms["squared"][index].sum(axis=0)
    absolute = terms["absolute"][index].sum(axis=0)
    signed = terms["signed"][index].sum(axis=0)
    valid = counts > 0
    channel_mse = [float(squared[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
    channel_mae = [float(absolute[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
    channel_bias = [float(signed[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
    return {
        "status": "available" if valid.all() else "unavailable_missing_channel_targets",
        "mse": float(np.mean(channel_mse)) if valid.all() else None,
        "mae": float(np.mean(channel_mae)) if valid.all() else None,
        "channel_mse": channel_mse,
        "channel_mae": channel_mae,
        "channel_mean_signed_error": channel_bias,
        "channel_target_count": counts.astype(int).tolist(),
        "channel_squared_error_sum": squared.tolist(),
        "channel_absolute_error_sum": absolute.tolist(),
        "channel_signed_error_sum": signed.tolist(),
        "missing_channel_indices": np.flatnonzero(~valid).tolist(),
    }


def space_terms(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray, basis: np.ndarray) -> dict[str, np.ndarray]:
    channels, latent = basis.shape
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("Prediction/target/mask axes mismatch for P/Q decomposition")
    if prediction.shape[2] != channels:
        raise ValueError("PCA basis channel count differs from prediction shape")
    complete = observed.all(axis=2)
    rows = np.argwhere(complete)
    projector = basis.astype(np.float64) @ basis.astype(np.float64).T
    q_projector = np.eye(channels, dtype=np.float64) - projector
    p_squared = np.zeros((prediction.shape[0], channels), dtype=np.float64)
    q_squared = np.zeros_like(p_squared)
    p_signed = np.zeros_like(p_squared)
    q_signed = np.zeros_like(p_squared)
    p_absolute = np.zeros_like(p_squared)
    q_absolute = np.zeros_like(p_squared)
    if len(rows):
        values = prediction[complete].astype(np.float64) - target[complete].astype(np.float64)
        retained = values @ projector
        residual = values @ q_projector
        origin_rows = rows[:, 0]
        np.add.at(p_squared, origin_rows, retained * retained)
        np.add.at(q_squared, origin_rows, residual * residual)
        np.add.at(p_signed, origin_rows, retained)
        np.add.at(q_signed, origin_rows, residual)
        np.add.at(p_absolute, origin_rows, np.abs(retained))
        np.add.at(q_absolute, origin_rows, np.abs(residual))
    counts = np.repeat(complete.sum(axis=1)[:, None], channels, axis=1)
    return {
        "p_squared": p_squared,
        "q_squared": q_squared,
        "p_absolute": p_absolute,
        "q_absolute": q_absolute,
        "p_signed": p_signed,
        "q_signed": q_signed,
        "count": counts,
        "complete_target_rows": complete.sum(axis=1),
        "basis_rank": np.asarray([latent], dtype=np.int64),
    }


def scores_from_space_terms(terms: dict[str, np.ndarray], index: np.ndarray | None = None) -> dict[str, Any]:
    if index is None:
        index = np.arange(len(terms["count"]), dtype=np.int64)
    counts = terms["count"][index].sum(axis=0)
    valid = counts > 0

    def score(prefix: str) -> tuple[list[float | None], list[float | None], list[float | None], float | None, float | None]:
        squared = terms[f"{prefix}_squared"][index].sum(axis=0)
        absolute = terms[f"{prefix}_absolute"][index].sum(axis=0)
        signed = terms[f"{prefix}_signed"][index].sum(axis=0)
        channel_mse = [float(squared[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
        channel_mae = [float(absolute[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
        channel_bias = [float(signed[c] / counts[c]) if valid[c] else None for c in range(len(counts))]
        return channel_mse, channel_mae, channel_bias, (float(np.mean(channel_mse)) if valid.all() else None), (float(np.mean(channel_mae)) if valid.all() else None)

    p_mse, p_mae, p_bias, p_macro_mse, p_macro_mae = score("p")
    q_mse, q_mae, q_bias, q_macro_mse, q_macro_mae = score("q")
    return {
        "status": "available" if valid.all() else "unavailable_missing_complete_target_rows",
        "scope": "Only origin-horizon rows with complete observed targets across all channels are projected.",
        "complete_target_rows": int(terms["complete_target_rows"][index].sum()),
        "channel_complete_row_count": counts.astype(int).tolist(),
        "p_retained_space_mse": p_macro_mse,
        "p_retained_space_mae": p_macro_mae,
        "q_residual_space_mse": q_macro_mse,
        "q_residual_space_mae": q_macro_mae,
        "channel_p_mse": p_mse,
        "channel_q_mse": q_mse,
        "channel_p_mae": p_mae,
        "channel_q_mae": q_mae,
        "channel_p_mean_signed_error": p_bias,
        "channel_q_mean_signed_error": q_bias,
    }


def q_preservation_summary(role_space_scores: dict[str, Any]) -> dict[str, Any]:
    parent = role_space_scores.get("level_res")
    if not parent:
        return {}
    output: dict[str, Any] = {}
    for role in ("staged_p", "staged_p_alpha", "staged_raw", "staged_raw_alpha"):
        scores = role_space_scores.get(role)
        if not scores:
            continue
        output[role] = {}
        for period in (*PERIODS, "combined"):
            left = scores.get(period, {})
            right = parent.get(period, {})
            left_q = left.get("q_residual_space_mse")
            right_q = right.get("q_residual_space_mse")
            left_q_mae = left.get("q_residual_space_mae")
            right_q_mae = right.get("q_residual_space_mae")
            output[role][period] = {
                "role_q_mse": left_q,
                "parent_q_mse": right_q,
                "delta_q_mse": None if left_q is None or right_q is None else left_q - right_q,
                "role_q_mae": left_q_mae,
                "parent_q_mae": right_q_mae,
                "delta_q_mae": None if left_q_mae is None or right_q_mae is None else left_q_mae - right_q_mae,
                "complete_target_rows": left.get("complete_target_rows"),
                "scope": "Diagnostic complete-target P/Q decomposition; zero delta would support parent Q-error preservation on this subset.",
            }
    return output


def projection_components(prediction: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    channels, latent = basis.shape
    if prediction.ndim != 3 or prediction.shape[2] != channels:
        raise ValueError("Prediction axes differ from TRAIN PCA basis")
    projector = basis.astype(np.float64) @ basis.astype(np.float64).T
    q_projector = np.eye(channels, dtype=np.float64) - projector
    flat = prediction.astype(np.float64).reshape(-1, channels)
    retained = (flat @ projector).reshape(prediction.shape)
    residual = (flat @ q_projector).reshape(prediction.shape)
    reconstruction_error = float(np.max(np.abs((retained + residual) - prediction.astype(np.float64))))
    if reconstruction_error > 1e-5:
        raise ValueError(f"P/Q projection does not reconstruct prediction within tolerance: {reconstruction_error}")
    return retained, residual, reconstruction_error


def parity_stats(left: np.ndarray, right: np.ndarray, atol: float, rtol: float) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError("Parity arrays have different shapes")
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    tolerance = atol + rtol * np.abs(right.astype(np.float64))
    violations = diff > tolerance
    denominator = np.maximum(np.abs(right.astype(np.float64)), 1e-12)
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "max_rel": float((diff / denominator).max()) if diff.size else 0.0,
        "violations": int(violations.sum()),
        "elements": int(diff.size),
        "pass": bool(not violations.any()),
    }


def q_prediction_parity(
    prediction_arrays: dict[str, np.ndarray],
    result_models: dict[str, dict[str, Any]],
    roles: dict[str, list[str]],
    basis: np.ndarray,
) -> dict[str, Any]:
    parent_ids = roles.get("level_res") or []
    parents_by_seed = {
        int(result_models[identifier]["spec"]["seed"]): identifier
        for identifier in parent_ids
        if identifier in result_models and result_models[identifier]["spec"].get("seed") is not None
    }
    if set(parents_by_seed) != set(SEEDS):
        return {"status": "unavailable_missing_seed_paired_parent", "parents_by_seed": parents_by_seed}
    components: dict[str, dict[str, Any]] = {}
    for identifier, prediction in prediction_arrays.items():
        retained, residual, reconstruction_error = projection_components(prediction, basis)
        components[identifier] = {"p": retained, "q": residual, "projection_reconstruction_max_abs": reconstruction_error}
    rows: dict[str, Any] = {}
    expected_passes = []
    for role in ("staged_p", "staged_p_alpha", "staged_raw", "staged_raw_alpha"):
        rows[role] = {}
        for identifier in roles.get(role, []):
            if identifier not in result_models or identifier not in components:
                continue
            spec = result_models[identifier]["spec"]
            seed = int(spec["seed"])
            parent_id = parents_by_seed[seed]
            if parent_id not in components:
                continue
            q_stats = parity_stats(components[identifier]["q"], components[parent_id]["q"], Q_PARITY_ATOL, Q_PARITY_RTOL)
            p_delta = parity_stats(components[identifier]["p"], components[parent_id]["p"], Q_PARITY_ATOL, Q_PARITY_RTOL)
            alpha = spec.get("alpha")
            full_parent_parity = None
            if alpha is not None and float(alpha) == 0.0:
                full_parent_parity = parity_stats(prediction_arrays[identifier], prediction_arrays[parent_id], Q_PARITY_ATOL, Q_PARITY_RTOL)
            expected_q_preserved = role in {"staged_p", "staged_p_alpha"}
            if expected_q_preserved:
                expected_passes.append(q_stats["pass"])
            rows[role][identifier] = {
                "seed": seed,
                "parent_id": parent_id,
                "alpha": alpha,
                "expected_q_preserved": expected_q_preserved,
                "q_prediction_parity": q_stats,
                "p_component_delta_against_parent": {
                    **p_delta,
                    "scope": "P component is allowed to differ; this is reported to confirm the correction acts in retained space.",
                },
                "alpha0_full_parent_parity": full_parent_parity,
                "projection_reconstruction_max_abs": components[identifier]["projection_reconstruction_max_abs"],
                "scope": "Full TEST prediction tensor projected by TRAIN PCA basis; no targets or masks are used.",
            }
    return {
        "status": "available",
        "atol": Q_PARITY_ATOL,
        "rtol": Q_PARITY_RTOL,
        "parents_by_seed": parents_by_seed,
        "all_expected_q_preservation_pass": bool(expected_passes and all(expected_passes)),
        "roles": rows,
    }


def pair_ids(ids: list[str], models: dict[str, dict[str, Any]]) -> list[str]:
    if len(ids) == 1:
        return ids * len(SEEDS)
    by_seed = {models[identifier]["spec"].get("seed"): identifier for identifier in ids}
    if set(by_seed) != set(SEEDS):
        raise ValueError(f"Expected paired seeds {SEEDS} or deterministic singleton; got {ids}")
    return [by_seed[seed] for seed in SEEDS]


def compare_terms(
    comparison: dict[str, Any],
    terms: dict[str, dict[str, np.ndarray]],
    models: dict[str, dict[str, Any]],
    period_indices: dict[str, np.ndarray],
    sealed: dict[str, Any],
    job: Job,
) -> dict[str, Any]:
    left_ids = pair_ids(list(comparison["left"]), models)
    right_ids = pair_ids(list(comparison["right"]), models)
    count = terms[left_ids[0]]["count"]
    for identifier in left_ids + right_ids:
        if not np.array_equal(terms[identifier]["count"], count):
            raise ValueError("All paired comparisons must use the same observed target counts")
    left = np.mean([terms[identifier]["squared"] for identifier in left_ids], axis=0)
    right = np.mean([terms[identifier]["squared"] for identifier in right_ids], axis=0)
    block = int(sealed["bootstrap"]["block_length_origins"])
    draws = int(sealed["bootstrap"]["draws"])
    rng_seed = int(sealed["bootstrap"]["seed"])

    def values(index: np.ndarray) -> dict[str, float] | None:
        denominator = count[index].sum(axis=0)
        if np.any(denominator == 0):
            return None
        left_mse = float(np.mean(left[index].sum(axis=0) / denominator))
        reference_mse = float(np.mean(right[index].sum(axis=0) / denominator))
        return {
            "left_mse": left_mse,
            "reference_mse": reference_mse,
            "delta_mse": left_mse - reference_mse,
            "relative_to_reference_pct": 100 * (left_mse - reference_mse) / reference_mse if reference_mse > 0 else None,
        }

    output = dict(comparison, direction="left minus reference; negative favors left", left_seed_ids=left_ids, right_seed_ids=right_ids, periods={})
    for label in (*PERIODS, "combined"):
        strata = [period_indices[label]] if label != "combined" else [period_indices[period] for period in PERIODS]
        full_index = np.concatenate(strata)
        full = values(full_index)
        seed_values = []
        for seed, left_id, right_id in zip(SEEDS, left_ids, right_ids):
            left_score = scores_from_terms(terms[left_id], full_index)["mse"]
            right_score = scores_from_terms(terms[right_id], full_index)["mse"]
            seed_values.append(
                {
                    "seed": seed,
                    "left_id": left_id,
                    "right_id": right_id,
                    "left_mse": left_score,
                    "reference_mse": right_score,
                    "delta_mse": left_score - right_score if left_score is not None and right_score is not None else None,
                    "relative_to_reference_pct": 100 * (left_score - right_score) / right_score if left_score is not None and right_score is not None and right_score > 0 else None,
                }
            )
        salt = int.from_bytes(hashlib.sha256(f"{comparison['name']}::{label}".encode("utf-8")).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([rng_seed, salt]))
        samples = []
        unavailable = 0
        for draw in range(draws):
            if draw % 250 == 0:
                job.check_limits()
            selected = []
            for stratum in strata:
                n = len(stratum)
                if n < block:
                    unavailable += 1
                    continue
                starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
                local = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
                selected.append(stratum[local])
            if len(selected) != len(strata):
                continue
            value = values(np.concatenate(selected))
            if value is None or value["relative_to_reference_pct"] is None:
                unavailable += 1
            else:
                samples.append(value)
        interval = None
        if full is not None and samples and unavailable == 0:
            interval = {
                key: np.quantile([row[key] for row in samples], [0.025, 0.975]).tolist()
                for key in ("delta_mse", "relative_to_reference_pct")
            }
        output["periods"][label] = {
            "full": full,
            "paired_seed_deltas": seed_values,
            "conditional_block95": interval,
            "draws": draws,
            "unavailable_draws": unavailable,
            "block_length_origins": block,
            "strata": [label] if label != "combined" else list(PERIODS),
            "interval_status": "available" if interval is not None else "unavailable_without_dropping_channels",
        }
    output["interval_scope"] = "Non-circular fixed-length origin blocks within each period, common resamples across channels/models; trained seeds fixed. Does not include training or selection uncertainty."
    return output


def evaluate_sealed(args: argparse.Namespace, job: Job) -> dict[str, Any]:
    dataset = args.dataset
    sealed = load_seal(dataset)
    exposure_path = HERE / f"test_exposure_{dataset}.json"
    seal_receipt = receipt(HERE / f"evaluation_seal_{dataset}.json")
    result_path = HERE / f"{args.name}.json"
    partial_path = HERE / f"{args.name}_partial.json"
    prediction_dir = CACHE / f"{args.name}_predictions"
    if exposure_path.exists():
        raise FileExistsError("This script does not overwrite exposed v9 TEST results; use a documented correction script if needed")
    if result_path.exists() or partial_path.exists() or prediction_dir.exists():
        raise FileExistsError("Preserve existing evaluation attempt; use a new name")
    save_json(
        exposure_path,
        {
            "schema": "tsfm_peft_staged_p_v9_test_exposure_v1",
            "dataset": dataset,
            "status": "exposed",
            "first_exposed_utc": time.time(),
            "seal_sha256": seal_receipt["sha256"],
            "seal": seal_receipt,
            "evaluation_name": args.name,
            "command": sys.argv,
            "scope": "Written after output path collision checks and before the first protected TEST archive is opened for this dataset and seal.",
        },
    )
    data = load_data(dataset, include_test=True)
    origins_by_period = validate_origins(sealed, data)
    origins = np.concatenate([origins_by_period[period] for period in PERIODS])
    period_indices = {
        "test_a": np.arange(len(origins_by_period["test_a"]), dtype=np.int64),
        "test_b": np.arange(len(origins_by_period["test_a"]), len(origins), dtype=np.int64),
    }
    inputs = np.stack([data["x"][origin - CONTEXT : origin] for origin in origins]).astype(np.float32)
    target = np.stack([data["x"][origin : origin + HORIZON] for origin in origins]).astype(np.float32)
    observed = np.stack([data["finite"][origin : origin + HORIZON] for origin in origins]).astype(bool)
    basis = np.asarray(data["basis"], dtype=np.float64)
    if inputs.shape[1:] != (CONTEXT, int(sealed["channels"])) or target.shape[1:] != (HORIZON, int(sealed["channels"])):
        raise ValueError("Sealed TEST tensor shape differs from approved contract")
    prediction_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "tsfm_peft_staged_p_v9_test_evaluation_v1",
        "status": "running",
        "name": args.name,
        "dataset": dataset,
        "seal": seal_receipt,
        "first_exposure": receipt(exposure_path),
        "source": source_receipt(),
        "environment": environment_receipt(),
        "channels": np.asarray(data["columns"]).astype(str).tolist(),
        "origins": {period: origins_by_period[period].astype(int).tolist() for period in PERIODS},
        "roles": sealed["roles"],
        "metric": sealed["metric"],
        "space_metric": sealed["space_metric"],
        "models": {},
        "role_scores": {},
        "role_space_scores": {},
        "q_preservation": {},
        "q_prediction_parity": {},
        "comparisons": [],
    }
    save_json(partial_path, result)
    models = {row["id"]: row for row in sealed["models"]}
    terms: dict[str, dict[str, np.ndarray]] = {}
    spaces: dict[str, dict[str, np.ndarray]] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    for identifier, row in models.items():
        job.check_limits()
        reused = False
        if row["kind"] == "stored_prediction":
            prediction, prediction_receipt = load_prediction_receipt(row, origins)
            reused = True
        elif row["kind"] == "checkpoint":
            prediction = checkpoint_predict(row, inputs, int(protocol()["eval_batch_origins"]), job)
            prediction_path = prediction_dir / f"{identifier}.npz"
            np.savez_compressed(prediction_path, prediction=prediction, origins=origins)
            prediction_receipt = receipt(prediction_path)
        else:
            raise ValueError(f"Unknown model kind {row['kind']} for {identifier}")
        terms[identifier] = error_terms(prediction, target, observed)
        scores = {period: scores_from_terms(terms[identifier], period_indices[period]) for period in PERIODS}
        scores["combined"] = scores_from_terms(terms[identifier])
        model_output = {
            "spec": row["spec"],
            "kind": row["kind"],
            "role": row["role"],
            "base_run_id": row.get("base_run_id"),
            "alpha_selected": row.get("alpha_selected"),
            "prediction": prediction_receipt,
            "reused_prior_prediction": reused,
            "reused_v8_prediction": reused,
            "scores": scores,
        }
        if row["role"] in SPACE_ROLES:
            spaces[identifier] = space_terms(prediction, target, observed, basis)
            space_scores = {period: scores_from_space_terms(spaces[identifier], period_indices[period]) for period in PERIODS}
            space_scores["combined"] = scores_from_space_terms(spaces[identifier])
            model_output["space_scores"] = space_scores
        if row["role"] in Q_PARITY_ROLES:
            prediction_arrays[identifier] = prediction.copy()
        result["models"][identifier] = model_output
        del prediction
        save_json(partial_path, result)
        job.heartbeat(f"{dataset}: evaluated {identifier}; {len(result['models'])}/{len(models)}")
    for role, ids in sealed["roles"].items():
        known = [identifier for identifier in ids if identifier in result["models"]]
        if not known:
            continue
        result["role_scores"][role] = {}
        for period in (*PERIODS, "combined"):
            scores = [result["models"][identifier]["scores"][period] for identifier in known]
            available = all(score["mse"] is not None for score in scores)
            result["role_scores"][role][period] = {
                "run_ids": known,
                "count": len(known),
                "mse": float(np.mean([score["mse"] for score in scores])) if available else None,
                "mae": float(np.mean([score["mae"] for score in scores])) if available else None,
                "scope": "Mean of individual model channel-macro losses; not an ensemble.",
            }
        if all(identifier in spaces for identifier in known):
            result["role_space_scores"][role] = {}
            for period in (*PERIODS, "combined"):
                scores = [result["models"][identifier]["space_scores"][period] for identifier in known]
                available = all(score["p_retained_space_mse"] is not None and score["q_residual_space_mse"] is not None for score in scores)
                result["role_space_scores"][role][period] = {
                    "run_ids": known,
                    "count": len(known),
                    "p_retained_space_mse": float(np.mean([score["p_retained_space_mse"] for score in scores])) if available else None,
                    "q_residual_space_mse": float(np.mean([score["q_residual_space_mse"] for score in scores])) if available else None,
                    "p_retained_space_mae": float(np.mean([score["p_retained_space_mae"] for score in scores])) if available else None,
                    "q_residual_space_mae": float(np.mean([score["q_residual_space_mae"] for score in scores])) if available else None,
                    "complete_target_rows": int(np.mean([score["complete_target_rows"] for score in scores])),
                    "scope": "Mean of individual model complete-row P/Q losses; not an ensemble.",
                }
    result["q_preservation"] = q_preservation_summary(result["role_space_scores"])
    result["q_prediction_parity"] = q_prediction_parity(prediction_arrays, result["models"], result["roles"], basis)
    for comparison in sealed["comparisons"]:
        if all(identifier in terms for identifier in comparison["left"] + comparison["right"]):
            result["comparisons"].append(compare_terms(comparison, terms, models, period_indices, sealed, job))
            job.heartbeat(f"{dataset}: compared {comparison['name']}")
    result.update(
        status="complete",
        completed_utc=time.time(),
        combined_definition="Within each seed/channel, sum TEST-A/B errors and observed counts before channel mean; then average individual seed losses. Deterministic references broadcast for seed-paired deltas.",
        test_reselection=False,
        new_test_model_predictions=sum(not item["reused_prior_prediction"] for item in result["models"].values()),
    )
    save_json(result_path, result)
    return {"result": rel(result_path), "models": len(result["models"]), "comparisons": len(result["comparisons"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal and evaluate approved v9 P/Q correction comparisons.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--seal", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dataset", choices=("robin", "jena"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reserve-seconds", type=float, required=True)
    args = parser.parse_args()
    validate_name(args.name)
    configure_numeric_runtime()
    attempt_suffix = "seal_attempt" if args.seal else "evaluate_attempt"
    attempt_path = HERE / f"{args.name}_{attempt_suffix}.json"
    if attempt_path.exists():
        raise FileExistsError("Preserve existing attempt file; use a new name")
    attempt = {
        "status": "running",
        "started_utc": time.time(),
        "command": sys.argv,
        "dataset": args.dataset,
        "source": source_receipt(),
        "environment": environment_receipt(),
    }
    save_json(attempt_path, attempt)
    try:
        if args.seal:
            with Job(f"v9_seal_{args.dataset}_{args.name}", category="cpu_analysis", reserve_seconds=args.reserve_seconds, metadata={"dataset": args.dataset, "name": args.name, "action": "seal"}):
                result = seal_evaluation(args)
        else:
            with Job(f"v9_evaluate_{args.dataset}_{args.name}", category="gpu", reserve_seconds=args.reserve_seconds, metadata={"dataset": args.dataset, "name": args.name, "action": "evaluate"}) as job:
                result = evaluate_sealed(args, job)
        attempt.update(status="complete", completed_utc=time.time(), result=result)
        save_json(attempt_path, attempt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        attempt.update(status="failed", failed_utc=time.time(), error=repr(exc))
        save_json(attempt_path, attempt)
        raise


if __name__ == "__main__":
    main()
