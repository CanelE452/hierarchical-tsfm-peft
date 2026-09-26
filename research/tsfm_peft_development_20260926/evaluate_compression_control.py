from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_development
from evaluate_development import measured_evaluate, paired_block_effect, validation_scores
from model import make_model
from run import CACHE, HERE, GPUJob, digest, load_data, save_json, score_arrays, source_receipt


NAME = "compression16"
SPECS_PATH = HERE / "compression16_specs.json"
SELECTION_PATH = HERE / f"{NAME}_selection.json"
OUTPUT_PATH = HERE / f"{NAME}_development.json"
LOCAL = CACHE / f"{NAME}_evaluation"
RANK32_NAME = "revision1_rank32"
RANK32_SELECTION_PATH = HERE / f"{RANK32_NAME}_selection.json"
RANK32_DEVELOPMENT_PATH = HERE / f"{RANK32_NAME}_development.json"
RANK32_LOCAL = CACHE / f"{RANK32_NAME}_evaluation"
SCORE_TOL = 1e-10


def save_npz_new(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"Do not overwrite cached prediction: {path}")
    np.savez_compressed(path, **arrays)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_development_report_identity(
    report: dict[str, Any],
    selection: dict[str, Any],
    selection_path: Path,
    data_sha256: str,
    label: str,
) -> None:
    if report.get("data_sha256") != data_sha256:
        raise RuntimeError(f"{label} data_sha256 mismatch: {report.get('data_sha256')} != {data_sha256}")
    actual_selection_sha = digest(selection_path)
    if report.get("selection_sha256") != actual_selection_sha:
        raise RuntimeError(
            f"{label} selection_sha256 mismatch: {report.get('selection_sha256')} != {actual_selection_sha}"
        )
    if report.get("selection") != selection:
        raise RuntimeError(f"{label} embedded selection does not match selection file")


def verify_result_data_sha(result: dict[str, Any], data_sha256: str, label: str) -> None:
    if result.get("data_sha256") != data_sha256:
        raise RuntimeError(f"{label} result data_sha256 mismatch: {result.get('data_sha256')} != {data_sha256}")


def load_specs() -> list[dict[str, Any]]:
    specs = read_json(SPECS_PATH)
    expected = {
        ("electricity", "compress", 92601, 0.0001, 16),
        ("electricity", "compress", 92601, 0.001, 16),
        ("electricity", "compress", 92602, 0.0001, 16),
        ("electricity", "compress", 92602, 0.001, 16),
    }
    seen = {(s["dataset"], s["arm"], s["seed"], s["lr"], s["latent"]) for s in specs}
    if seen != expected or len(specs) != 4:
        raise RuntimeError(f"unexpected compression16 specs: {seen}")
    for spec in specs:
        if spec["id"] != f"compression16_compress_{spec['seed']}_{spec['lr']:g}":
            raise RuntimeError(f"unexpected spec id: {spec}")
        if spec.get("loss") != "mse" or spec.get("epochs") != 120 or spec.get("microbatch_origins") != 4:
            raise RuntimeError(f"spec does not preserve requested recipe: {spec}")
        if spec.get("baseline_capacity_control") is not True:
            raise RuntimeError(f"spec is not marked as capacity control: {spec}")
    return specs


def completed_rows_for_specs(specs: list[dict[str, Any]], data_sha256: str) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        result_path = HERE / "runs" / spec["id"] / "result.json"
        result = read_json(result_path)
        if result.get("status") != "complete":
            raise RuntimeError(f"fit is not complete: {spec['id']}")
        verify_result_data_sha(result, data_sha256, spec["id"])
        for key, value in spec.items():
            if result.get(key) != value:
                raise RuntimeError(f"completed result/spec mismatch for {spec['id']} key {key}: {result.get(key)} != {value}")
        checkpoint = Path(result["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint missing for {spec['id']}: {checkpoint}")
        if digest(checkpoint) != result["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for {spec['id']}")
        val_path = checkpoint.with_name("best_val.npz")
        if not val_path.exists():
            raise FileNotFoundError(f"validation prediction missing for {spec['id']}: {val_path}")
        rows.append(result)
    return rows


def validation_only_selection(specs: list[dict[str, Any]]) -> dict[str, Any]:
    if SELECTION_PATH.exists():
        raise RuntimeError(f"Do not overwrite selection ledger: {SELECTION_PATH}")
    selection = evaluate_development.select_initial(SPECS_PATH)
    if set(selection) != {"compress"}:
        raise RuntimeError(f"compression16 selection should contain only compress, got {selection.keys()}")
    selected = selection["compress"]
    if sorted(selected["fits"]) != sorted([s["id"] for s in specs if s["lr"] == selected["lr"]]):
        raise RuntimeError("selected fit ids do not match the selected LR across both seeds")
    save_json(SELECTION_PATH, selection)
    return selection


def row_by_fit(report: dict[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in report["rows"]:
        if row.get("arm") == arm and row.get("fit"):
            rows[row["fit"]] = row
    return rows


def verify_prediction_row_score(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    row: dict[str, Any],
    fit_id: str,
) -> None:
    if prediction.shape != target.shape:
        raise RuntimeError(f"prediction shape mismatch for {fit_id}: {prediction.shape} != {target.shape}")
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"prediction contains non-finite values for {fit_id}")
    recomputed = score_arrays(prediction, target, mask)
    mse_diff = abs(recomputed["mse"] - row["scores"]["mse"])
    if mse_diff > SCORE_TOL:
        raise RuntimeError(f"score replay MSE mismatch for {fit_id}: {mse_diff} > {SCORE_TOL}")


def load_rank32_residual_predictions(data: dict[str, Any]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    data_sha256 = digest(data["_path"])
    rank32_selection = read_json(RANK32_SELECTION_PATH)
    rank32_report = read_json(RANK32_DEVELOPMENT_PATH)
    if rank32_report.get("status") != "complete":
        raise RuntimeError("revision1_rank32 development report is not complete")
    verify_development_report_identity(
        rank32_report,
        rank32_selection,
        RANK32_SELECTION_PATH,
        data_sha256,
        "revision1_rank32",
    )
    if rank32_report.get("selection") != rank32_selection:
        raise RuntimeError("revision1_rank32 selection JSON does not match development report")
    residual_selection = rank32_selection.get("residual")
    if not residual_selection or len(residual_selection.get("fits", [])) != 2:
        raise RuntimeError("revision1_rank32 residual selection must contain two selected fits")
    residual_rows = row_by_fit(rank32_report, "residual")
    target, mask = target_arrays(data)
    predictions = []
    receipts = []
    for fit_id in sorted(residual_selection["fits"], key=lambda f: read_json(HERE / "runs" / f / "result.json")["seed"]):
        result_path = HERE / "runs" / fit_id / "result.json"
        result = read_json(result_path)
        spec_receipt = {
            key: result.get(key)
            for key in ("id", "dataset", "arm", "seed", "lr", "latent", "loss", "epochs", "microbatch_origins", "residual_rank")
        }
        if result.get("status") != "complete" or result.get("dataset") != "electricity" or result.get("arm") != "residual":
            raise RuntimeError(f"not a complete Electricity residual result: {fit_id}")
        verify_result_data_sha(result, data_sha256, fit_id)
        if result.get("id") != fit_id:
            raise RuntimeError(f"result id mismatch for {fit_id}: {result.get('id')}")
        if result.get("lr") != residual_selection["lr"]:
            raise RuntimeError(f"result LR does not match selected residual LR for {fit_id}")
        if result.get("latent") != 8:
            raise RuntimeError(f"rank32 residual should keep latent8 compression basis, got {spec_receipt}")
        if result.get("residual_rank") != 32:
            raise RuntimeError(f"rank32 residual result expected, got {spec_receipt}")
        checkpoint = Path(result["checkpoint"])
        if not checkpoint.exists() or digest(checkpoint) != result["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint identity check failed for {fit_id}")
        row = residual_rows.get(fit_id)
        if row is None:
            raise RuntimeError(f"revision1_rank32 development row missing for residual fit {fit_id}")
        if row.get("fit") != fit_id or row.get("seed") != result["seed"]:
            raise RuntimeError(f"revision1_rank32 row identity mismatch for {fit_id}")
        if "lr" in row and row["lr"] != result["lr"]:
            raise RuntimeError(f"revision1_rank32 row LR mismatch for {fit_id}")
        pred_path = RANK32_LOCAL / f"{fit_id}.npz"
        if not pred_path.exists():
            raise FileNotFoundError(f"cached rank32 residual DEV prediction missing: {pred_path}")
        pred_hash = digest(pred_path)
        if row.get("prediction_sha256") != pred_hash:
            raise RuntimeError(f"rank32 residual prediction hash mismatch for {fit_id}")
        with np.load(pred_path, allow_pickle=False) as archive:
            origins = archive["origins"]
            prediction = archive["prediction"]
        if not np.array_equal(origins, data["dev_origins"]):
            raise RuntimeError(f"rank32 residual origin mismatch for {fit_id}")
        verify_prediction_row_score(prediction, target, mask, row, fit_id)
        predictions.append(prediction)
        receipts.append(
            {
                "fit": fit_id,
                "seed": result["seed"],
                "spec": spec_receipt,
                "result_file": str(result_path.relative_to(HERE.parents[1])),
                "result_sha256": digest(result_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "prediction_file": str(pred_path.relative_to(HERE.parents[1])),
                "prediction_sha256": pred_hash,
                "row_scores": row["scores"],
            }
        )
    return predictions, receipts


def evaluate_selected_compression16(selection: dict[str, Any], data: dict[str, Any], job: GPUJob) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    selected_fits = sorted(selection["compress"]["fits"], key=lambda f: read_json(HERE / "runs" / f / "result.json")["seed"])
    predictions = []
    rows = []
    LOCAL.mkdir(parents=True, exist_ok=True)
    for fit_id in selected_fits:
        result_path = HERE / "runs" / fit_id / "result.json"
        result = read_json(result_path)
        if result.get("status") != "complete" or result.get("dataset") != "electricity" or result.get("arm") != "compress":
            raise RuntimeError(f"not a complete Electricity compress result: {fit_id}")
        if result.get("latent") != 16 or result.get("baseline_capacity_control") is not True:
            raise RuntimeError(f"not a compression16 capacity-control result: {fit_id}")
        checkpoint = Path(result["checkpoint"])
        if digest(checkpoint) != result["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for {fit_id}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        spec = saved["spec"]
        for key in ("id", "dataset", "arm", "seed", "lr", "latent", "loss", "epochs", "microbatch_origins", "baseline_capacity_control"):
            if spec.get(key) != result.get(key):
                raise RuntimeError(f"checkpoint spec/result mismatch for {fit_id} key {key}")
        model = make_model(spec["arm"], saved["basis"], residual_rank=spec.get("residual_rank", 8))
        model.restore_adapter(saved["state"])
        scores, pred, cost = measured_evaluate(model, data, data["dev_origins"], job, fit_id + " development")
        pred_path = LOCAL / f"{fit_id}.npz"
        save_npz_new(pred_path, prediction=pred, origins=data["dev_origins"])
        predictions.append(pred)
        with np.load(checkpoint.with_name("best_val.npz"), allow_pickle=False) as archive:
            if not np.array_equal(archive["origins"], data["val_origins"]):
                raise RuntimeError(f"validation origins mismatch for {fit_id}")
            validation = validation_scores(archive["prediction"], data)
        rows.append(
            {
                "arm": "compress16",
                "base_arm": "compress",
                "fit": fit_id,
                "seed": spec["seed"],
                "lr": spec["lr"],
                "latent": spec["latent"],
                "scores": scores,
                "validation": validation,
                "evaluation_cost": cost,
                "trainable_parameters": result["trainable_parameters"],
                "selected_epoch": result["selected_epoch"],
                "selected_steps": result["selected_steps"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "result_file": str(result_path.relative_to(HERE.parents[1])),
                "result_sha256": digest(result_path),
                "prediction_file": str(pred_path.relative_to(HERE.parents[1])),
                "prediction_sha256": digest(pred_path),
            }
        )
        del model, saved
        gc.collect()
        torch.cuda.empty_cache()
    return predictions, rows


def target_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    target = np.stack([data["x"][o : o + 48] for o in data["dev_origins"]])
    mask = np.stack([data["finite"][o : o + 48] for o in data["dev_origins"]])
    return target, mask


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Never overwrite completed evaluation: {OUTPUT_PATH}")
    if SELECTION_PATH.exists():
        raise RuntimeError(f"Never overwrite selection ledger: {SELECTION_PATH}")
    if LOCAL.exists() and any(LOCAL.iterdir()):
        raise RuntimeError(f"Never mix cached evaluation outputs: {LOCAL}")

    specs = load_specs()
    data = load_data("electricity")
    data_sha256 = digest(data["_path"])
    completed_rows_for_specs(specs, data_sha256)
    selection = validation_only_selection(specs)
    residual_predictions, residual_receipts = load_rank32_residual_predictions(data)
    compression_seeds = sorted(read_json(HERE / "runs" / f / "result.json")["seed"] for f in selection["compress"]["fits"])
    residual_seeds = sorted(r["seed"] for r in residual_receipts)
    if compression_seeds != residual_seeds:
        raise RuntimeError(f"seed pairing mismatch: compress16 {compression_seeds}, residual {residual_seeds}")

    with GPUJob(NAME + "_development_evaluation") as job:
        compression_predictions, rows = evaluate_selected_compression16(selection, data, job)

    target, mask = target_arrays(data)
    effect = paired_block_effect(residual_predictions, compression_predictions, target, mask)
    report = {
        "status": "complete",
        "scope": "Electricity reused development capacity control; no protected Bull scores; no F0/LoRA/existing baseline re-evaluation.",
        "source": source_receipt() | {"compression16_specs.json": digest(SPECS_PATH)},
        "data_sha256": digest(data["_path"]),
        "specs_file": SPECS_PATH.name,
        "specs_sha256": digest(SPECS_PATH),
        "selection": selection,
        "selection_file": SELECTION_PATH.name,
        "selection_sha256": digest(SELECTION_PATH),
        "selection_policy": "validation-only two-seed mean via evaluate_development.select_initial; DEV not used for LR choice",
        "rank32_residual_reuse": {
            "selection_file": RANK32_SELECTION_PATH.name,
            "selection_sha256": digest(RANK32_SELECTION_PATH),
            "development_file": RANK32_DEVELOPMENT_PATH.name,
            "development_sha256": digest(RANK32_DEVELOPMENT_PATH),
            "prediction_receipts": residual_receipts,
        },
        "rows": rows,
        "residual_vs_compress16": effect,
    }
    save_json(OUTPUT_PATH, report)
    print(
        json.dumps(
            {
                "selected": selection,
                "compress16_dev_mse": [row["scores"]["mse"] for row in rows],
                "residual_vs_compress16": effect,
                "output": str(OUTPUT_PATH),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
