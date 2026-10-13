"""Source/data-contract verifier for forecast_generalization_study_20261013.

This companion audit is narrower than verify_generalization.py.  It checks the
raw TRAIN/VAL source NPZ, the FIT-only normalization contract, target/mask/origin
materialization in every resolved prediction NPZ, post-FIT reconstruction masks,
and source/snapshot hashes.  It does not train models, does not use CUDA, and
does not open TEST data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
H = 48
L_FULL = 512
EXPECTED_ORIGIN_COUNTS = {
    "jena": {"fit": 256, "tune": 64, "val": 371, "channels": 21},
    "robin": {"fit": 256, "tune": 64, "val": 30, "channels": 17},
}
TOL = 1e-6


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ahash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def select_origins(values: np.ndarray, maximum: int | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if maximum is None or len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values) - 1, maximum).astype(int)]


def max_abs(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if mask is not None:
        if not bool(np.any(mask)):
            return 0.0
        aa = aa[mask]
        bb = bb[mask]
    if aa.size == 0:
        return 0.0
    return float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64))))


def preprocess_batch(physical: np.ndarray, observed: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    filled = np.empty_like(physical, dtype=np.float32)
    last = np.broadcast_to(mean, (len(physical), len(mean))).copy()
    for t in range(physical.shape[1]):
        last = np.where(observed[:, t], physical[:, t], last)
        filled[:, t] = last
    return ((filled - mean) / std).astype(np.float32)


def preprocess_single_loop(physical: np.ndarray, observed: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    rows = []
    for i in range(len(physical)):
        last = mean.astype(np.float32).copy()
        one = np.empty_like(physical[i], dtype=np.float32)
        for t in range(physical.shape[1]):
            mask = observed[i, t]
            last[mask] = physical[i, t, mask]
            one[t] = (last - mean) / std
        rows.append(one)
    return np.stack(rows).astype(np.float32)


@dataclass
class DatasetData:
    name: str
    contract: dict[str, Any]
    source_path: Path
    source_sha256: str
    raw_x: np.ndarray
    raw_finite: np.ndarray
    raw_mean: np.ndarray
    raw_std: np.ndarray
    columns: list[str]
    physical: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    filled: np.ndarray
    x: np.ndarray
    origins: dict[str, np.ndarray]


@dataclass
class State:
    output_dir: Path
    opened_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def open_path(self, path: Path) -> None:
        self.opened_paths.append(rel(path))

    def read_json(self, path: Path) -> Any:
        self.open_path(path)
        return read_json(path)


def load_dataset(state: State, dataset: str, full_dir: Path) -> DatasetData:
    contract_path = full_dir / f"{dataset}_contract.json"
    contract = state.read_json(contract_path)
    source_path = resolve_path(contract["source"])
    state.open_path(source_path)
    actual_sha = sha256(source_path)
    if actual_sha != contract.get("source_sha256"):
        state.errors.append(f"{dataset}: source_sha256 mismatch")
    with np.load(source_path, allow_pickle=False) as npz:
        raw_x = np.asarray(npz["x"], dtype=np.float32)
        finite = np.asarray(npz["finite"]).astype(bool)
        raw_mean = np.asarray(npz["mean"], dtype=np.float32)
        raw_std = np.asarray(npz["std"], dtype=np.float32)
        columns = [str(x) for x in npz["columns"].tolist()]
        val_origins = np.asarray(npz["val_origins"], dtype=np.int64)

    physical = (raw_x * raw_std + raw_mean).astype(np.float32)
    physical[~finite] = np.nan
    lo, hi = contract["bounds"]["fit"]
    mean = np.nanmean(physical[lo:hi], axis=0).astype(np.float32)
    std = np.nanstd(physical[lo:hi], axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    if not np.isfinite(mean).all():
        state.errors.append(f"{dataset}: FIT mean contains non-finite value")

    filled = np.empty_like(physical, dtype=np.float32)
    last = mean.copy()
    for t in range(len(physical)):
        observed = finite[t]
        last[observed] = physical[t, observed]
        filled[t] = last
    x = ((filled - mean) / std).astype(np.float32)

    bounds = contract["bounds"]
    origins = {
        "fit": select_origins(np.arange(bounds["fit"][0] + L_FULL, bounds["fit"][1] - H + 1), EXPECTED_ORIGIN_COUNTS[dataset]["fit"]),
        "tune": select_origins(np.arange(bounds["tune"][0], bounds["tune"][1] - H + 1), EXPECTED_ORIGIN_COUNTS[dataset]["tune"]),
        "val": val_origins,
    }
    return DatasetData(
        name=dataset,
        contract=contract,
        source_path=source_path,
        source_sha256=actual_sha,
        raw_x=raw_x,
        raw_finite=finite,
        raw_mean=raw_mean,
        raw_std=raw_std,
        columns=columns,
        physical=physical,
        mean=mean,
        std=std,
        filled=filled,
        x=x,
        origins=origins,
    )


def validate_dataset_contract(state: State, data: DatasetData) -> dict[str, Any]:
    contract = data.contract
    expected = EXPECTED_ORIGIN_COUNTS[data.name]
    checks: dict[str, Any] = {
        "source": rel(data.source_path),
        "source_sha256": data.source_sha256,
        "source_sha256_matches_contract": data.source_sha256 == contract.get("source_sha256"),
        "source_shapes": {"x": list(data.raw_x.shape), "finite": list(data.raw_finite.shape)},
        "channel_count": len(data.columns),
        "expected_channel_count": expected["channels"],
        "normalization_fit_scope": contract.get("normalization_fit_scope"),
        "fit_mean_max_abs_vs_contract": max_abs(data.mean, np.asarray(contract.get("mean"), dtype=np.float32)),
        "fit_std_max_abs_vs_contract": max_abs(data.std, np.asarray(contract.get("std"), dtype=np.float32)),
        "origin_counts": {},
        "origin_hashes_match": {},
        "target_mask_hashes_match": {},
        "bounds_legal": {},
    }
    if checks["channel_count"] != expected["channels"]:
        state.errors.append(f"{data.name}: channel count mismatch")
    if contract.get("normalization_fit_scope") != "FIT_ONLY":
        state.errors.append(f"{data.name}: normalization_fit_scope is not FIT_ONLY")
    if checks["fit_mean_max_abs_vs_contract"] > TOL or checks["fit_std_max_abs_vs_contract"] > TOL:
        state.errors.append(f"{data.name}: FIT mean/std mismatch")
    for split, origins in data.origins.items():
        contract_origins = np.asarray(contract["origins"][split], dtype=np.int64)
        observed = np.stack([data.raw_finite[o : o + H] for o in origins])
        bounds = contract["bounds"][split]
        checks["origin_counts"][split] = int(len(origins))
        checks["origin_hashes_match"][split] = ahash(origins) == contract.get("origin_hashes", {}).get(split)
        checks["target_mask_hashes_match"][split] = ahash(observed) == contract.get("observed_target_masks", {}).get(split)
        checks["bounds_legal"][split] = bool(
            len(origins)
            and np.array_equal(origins, contract_origins)
            and np.all(origins >= bounds[0])
            and np.all(origins + H <= bounds[1])
            and np.all(origins - L_FULL >= 0)
        )
        if len(origins) != expected[split]:
            state.errors.append(f"{data.name}/{split}: origin count mismatch")
        if not checks["origin_hashes_match"][split]:
            state.errors.append(f"{data.name}/{split}: origin hash mismatch")
        if not checks["target_mask_hashes_match"][split]:
            state.errors.append(f"{data.name}/{split}: target mask hash mismatch")
        if not checks["bounds_legal"][split]:
            state.errors.append(f"{data.name}/{split}: origin bounds or contract origins mismatch")
    return checks


def window_and_postfit_checks(state: State, data: DatasetData, resolved: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"ff_windows": {}, "post_fit": {}}
    for length in (128, 512):
        split_checks = {}
        for split, origins in data.origins.items():
            physical = np.stack([data.physical[o - length : o] for o in origins])
            observed = np.stack([data.raw_finite[o - length : o] for o in origins])
            batch = preprocess_batch(physical, observed, data.mean, data.std)
            single = preprocess_single_loop(physical, observed, data.mean, data.std)
            global_slice = np.stack([data.x[o - length : o] for o in origins])
            first_all = observed[:, 0].all(axis=1)
            split_checks[split] = {
                "window_count": int(len(origins)),
                "batch_vs_single_max_abs": max_abs(batch, single),
                "reset_vs_global_max_abs": max_abs(batch, global_slice),
                "first_row_all_observed_count": int(first_all.sum()),
                "first_row_all_observed_vs_global_max_abs": max_abs(batch[first_all], global_slice[first_all]) if bool(first_all.any()) else None,
                "windows_with_reset_difference_count": int(np.any(np.abs(batch - global_slice) > 1e-6, axis=(1, 2)).sum()),
            }
            if split_checks[split]["batch_vs_single_max_abs"] > TOL:
                state.errors.append(f"{data.name}/L{length}/{split}: independent FF loops mismatch")
            if first_all.any() and split_checks[split]["first_row_all_observed_vs_global_max_abs"] > 2e-6:
                state.errors.append(f"{data.name}/L{length}/{split}: reset window differs despite fully observed first row")
        result["ff_windows"][f"L{length}"] = split_checks

    cut = data.contract["bounds"]["fit"][1]
    reconstructed_by_length: dict[int, dict[str, Any]] = {}
    for length in (128, 512):
        tune_origins = data.origins["tune"]
        history_mask = np.stack([data.raw_finite[o - length : o] for o in tune_origins])
        tune_positions = np.stack([np.arange(o - length, o) for o in tune_origins])
        post_fit_mask = history_mask & (tune_positions >= cut)[:, :, None]
        reconstructed_by_length[length] = {
            "cut": cut,
            "mask_sha256": ahash(post_fit_mask),
            "observed_counts": post_fit_mask.sum((0, 1)).astype(int).tolist(),
            "unique_post_fit_timestamps": int(len(np.unique(tune_positions[tune_positions >= cut]))),
        }
    compared = 0
    mismatches = []
    for case in resolved.get("cases", []):
        if case.get("dataset") != data.name or not case.get("training"):
            continue
        length = int(case.get("length", L_FULL))
        reconstructed = reconstructed_by_length.get(length)
        if reconstructed is None:
            mismatches.append(case.get("label"))
            continue
        diagnostic = case.get("training", {}).get("post_fit_reconstruction_diagnostic", {})
        compared += 1
        ok = (
            diagnostic.get("cut") == reconstructed["cut"]
            and diagnostic.get("mask_sha256") == reconstructed["mask_sha256"]
            and diagnostic.get("observed_counts") == reconstructed["observed_counts"]
            and diagnostic.get("unique_post_fit_timestamps") == reconstructed["unique_post_fit_timestamps"]
        )
        if not ok:
            mismatches.append(case.get("label"))
    if mismatches:
        state.errors.append(f"{data.name}: post-FIT reconstruction diagnostic mismatch {len(mismatches)} cases")
    result["post_fit"] = {
        "reconstructed_by_length": {str(k): v for k, v in reconstructed_by_length.items()},
        "compared_training_case_count": compared,
        "mismatch_count": len(mismatches),
        "mismatch_labels": mismatches[:20],
    }
    return result


def collect_prediction_paths(resolved: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    items = []
    seen = set()
    for case in resolved.get("cases", []):
        for method in case.get("methods", []):
            meta = method.get("prediction", {})
            if not meta.get("path"):
                continue
            key = (case.get("label"), method.get("method"), meta["path"])
            if key in seen:
                continue
            seen.add(key)
            items.append((case, method, resolve_path(meta["path"])))
    return items


def validate_prediction_npzs(state: State, datasets: dict[str, DatasetData], resolved: dict[str, Any]) -> dict[str, Any]:
    items = collect_prediction_paths(resolved)
    summary = {
        "prediction_npz_count": len(items),
        "checked_method_count": 0,
        "sha_mismatch_count": 0,
        "mean_std_mismatch_count": 0,
        "origin_mismatch_count": 0,
        "mask_mismatch_count": 0,
        "target_observed_mismatch_count": 0,
        "target_full_mismatch_count": 0,
        "max_target_observed_abs": 0.0,
        "max_target_full_abs": 0.0,
        "examples": [],
    }
    for case, method, path in items:
        dataset = case["dataset"]
        data = datasets[dataset]
        if not path.exists():
            state.errors.append(f"missing prediction NPZ: {rel(path)}")
            continue
        state.open_path(path)
        digest = sha256(path)
        if digest != method.get("prediction", {}).get("sha256"):
            summary["sha_mismatch_count"] += 1
            state.errors.append(f"{case.get('label')}/{method.get('method')}: prediction NPZ sha mismatch")
        with np.load(path, allow_pickle=False) as npz:
            if max_abs(npz["mean"], data.mean) > TOL or max_abs(npz["std"], data.std) > TOL:
                summary["mean_std_mismatch_count"] += 1
            for split, origins in data.origins.items():
                saved_origins = np.asarray(npz[f"{split}_origins"], dtype=np.int64)
                saved_mask = np.asarray(npz[f"{split}_observed"]).astype(bool)
                saved_target = np.asarray(npz[f"{split}_target"], dtype=np.float32)
                ref_mask = np.stack([data.raw_finite[o : o + H] for o in origins])
                ref_target = np.stack([data.x[o : o + H] for o in origins]).astype(np.float32)
                ref_target_observed = ((np.stack([data.physical[o : o + H] for o in origins]) - data.mean) / data.std).astype(np.float32)
                origin_ok = np.array_equal(saved_origins, origins)
                mask_ok = np.array_equal(saved_mask, ref_mask)
                observed_diff = max_abs(saved_target, ref_target_observed, ref_mask)
                full_diff = max_abs(saved_target, ref_target)
                summary["max_target_observed_abs"] = max(summary["max_target_observed_abs"], observed_diff)
                summary["max_target_full_abs"] = max(summary["max_target_full_abs"], full_diff)
                if not origin_ok:
                    summary["origin_mismatch_count"] += 1
                if not mask_ok:
                    summary["mask_mismatch_count"] += 1
                if observed_diff > 2e-6:
                    summary["target_observed_mismatch_count"] += 1
                if full_diff > 2e-6:
                    summary["target_full_mismatch_count"] += 1
                if (
                    len(summary["examples"]) < 10
                    and (not origin_ok or not mask_ok or observed_diff > 2e-6 or full_diff > 2e-6)
                ):
                    summary["examples"].append(
                        {
                            "case": case.get("label"),
                            "method": method.get("method"),
                            "split": split,
                            "origin_ok": origin_ok,
                            "mask_ok": mask_ok,
                            "target_observed_max_abs": observed_diff,
                            "target_full_max_abs": full_diff,
                        }
                    )
        summary["checked_method_count"] += 1

    for key in ("sha_mismatch_count", "mean_std_mismatch_count", "origin_mismatch_count", "mask_mismatch_count", "target_observed_mismatch_count"):
        if summary[key]:
            state.errors.append(f"prediction NPZ {key}={summary[key]}")
    return summary


def validate_source_hashes(state: State, manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks = {}
    for tag, manifest in manifests.items():
        source_hashes = manifest.get("source_hashes", {})
        tag_checks = []
        for path_text, recorded in source_hashes.items():
            path = resolve_path(path_text)
            exists = path.exists()
            actual = sha256(path) if exists else None
            if exists:
                state.open_path(path)
            item = {
                "source": path_text,
                "exists": exists,
                "recorded_sha256": recorded,
                "current_sha256": actual,
                "current_matches_recorded": exists and actual == recorded,
            }
            if not item["current_matches_recorded"]:
                state.errors.append(f"{tag}: source hash mismatch {path_text}")
            tag_checks.append(item)
        snapshot_dir = HERE / f"run_{tag}" / "sources"
        snapshot_checks = []
        if snapshot_dir.exists():
            for item in tag_checks:
                snapshot = snapshot_dir / Path(item["source"]).name
                if not snapshot.exists():
                    snapshot_checks.append({"source": item["source"], "snapshot": rel(snapshot), "exists": False})
                    continue
                state.open_path(snapshot)
                digest = sha256(snapshot)
                snapshot_checks.append(
                    {
                        "source": item["source"],
                        "snapshot": rel(snapshot),
                        "exists": True,
                        "snapshot_sha256": digest,
                        "snapshot_matches_recorded": digest == item["recorded_sha256"],
                    }
                )
                if digest != item["recorded_sha256"]:
                    state.errors.append(f"{tag}: source snapshot hash mismatch {rel(snapshot)}")
        checks[tag] = {
            "source_hash_count": len(source_hashes),
            "source_hash_checks": tag_checks,
            "snapshot_dir_exists": snapshot_dir.exists(),
            "snapshot_checks": snapshot_checks,
        }

    resolved = manifests.get("resolved_v1", {})
    for name, info in (resolved.get("source_runs") or {}).items():
        path = resolve_path(info["path"])
        state.open_path(path)
        actual = sha256(path)
        checks.setdefault("resolved_v1_source_runs", {})[name] = {
            "path": info["path"],
            "recorded_sha256": info.get("sha256"),
            "actual_sha256": actual,
            "matches": actual == info.get("sha256"),
        }
        if actual != info.get("sha256"):
            state.errors.append(f"resolved_v1 source_run hash mismatch: {name}")
    return checks


def protected_access_summary(state: State) -> dict[str, Any]:
    suspicious_data = [
        path
        for path in state.opened_paths
        if path.lower().endswith((".npz", ".npy", ".pt"))
        and "test" in Path(path).name.lower()
    ]
    return {
        "opened_path_count": len(state.opened_paths),
        "opened_trainval_npz_count": len([p for p in state.opened_paths if p.lower().endswith(".npz") and "trainval" in p.lower()]),
        "suspicious_test_data_path_count": len(suspicious_data),
        "suspicious_test_data_paths": suspicious_data,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Source/data contract 검증",
        "",
        f"- 생성 시각: {summary['generated_at']}",
        f"- 상태: `{summary['status']}`",
        f"- 검증 범위: {summary['scope']}",
        "",
        "## 핵심 결과",
        "",
    ]
    for dataset, checks in summary["checks"]["datasets"].items():
        lines.extend(
            [
                f"- {dataset}: source SHA match `{checks['source_sha256_matches_contract']}`, "
                f"FIT mean max abs {checks['fit_mean_max_abs_vs_contract']:.3g}, "
                f"FIT std max abs {checks['fit_std_max_abs_vs_contract']:.3g}, "
                f"origins {checks['origin_counts']}",
            ]
        )
    npz = summary["checks"]["prediction_npzs"]
    lines.extend(
        [
            f"- prediction NPZ 검증: {npz['checked_method_count']}개 / expected 236",
            f"- target observed max abs: {npz['max_target_observed_abs']:.3g}",
            f"- target full max abs: {npz['max_target_full_abs']:.3g}",
            f"- mismatch counts: sha={npz['sha_mismatch_count']}, mean/std={npz['mean_std_mismatch_count']}, "
            f"origin={npz['origin_mismatch_count']}, mask={npz['mask_mismatch_count']}, "
            f"target_observed={npz['target_observed_mismatch_count']}",
            f"- TEST data 의심 접근 수: {summary['checks']['protected_access']['suspicious_test_data_path_count']}",
            "",
            "이 PASS는 위 source/contract/materialization 항목에 한정된다. 분석 조건 전체에서 미래 정보 누출이 전혀 없다는 일반 명제까지 확장하지 않는다.",
            "",
            "## warnings",
            "",
        ]
    )
    lines.extend([f"- {warning}" for warning in summary.get("warnings", [])] or ["- 없음"])
    lines.extend(["", "## errors", ""])
    lines.extend([f"- {error}" for error in summary.get("errors", [])] or ["- 없음"])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = HERE / args.output
    output_dir.mkdir(exist_ok=True)
    state = State(output_dir=output_dir)

    full_manifest_path = HERE / "run_full_v1" / "RESULT.json"
    repair_manifest_path = HERE / "run_repair_v1" / "RESULT.json"
    resolved_manifest_path = HERE / "run_resolved_v1" / "RESULT.json"
    clean_manifest_path = HERE / "run_clean_cost_v1" / "RESULT.json"
    manifests = {
        "full_v1": state.read_json(full_manifest_path),
        "repair_v1": state.read_json(repair_manifest_path),
        "resolved_v1": state.read_json(resolved_manifest_path),
        "clean_cost_v1": state.read_json(clean_manifest_path),
    }

    datasets = {
        dataset: load_dataset(state, dataset, HERE / "run_full_v1")
        for dataset in ("jena", "robin")
    }
    dataset_checks = {dataset: validate_dataset_contract(state, data) for dataset, data in datasets.items()}
    window_checks = {dataset: window_and_postfit_checks(state, data, manifests["resolved_v1"]) for dataset, data in datasets.items()}
    prediction_checks = validate_prediction_npzs(state, datasets, manifests["resolved_v1"])
    source_hash_checks = validate_source_hashes(state, manifests)
    access = protected_access_summary(state)
    if access["suspicious_test_data_path_count"]:
        state.errors.append("TEST data path access detected")

    status = "PASS" if not state.errors else "FAIL"
    summary = {
        "status": status,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "TRAIN/VAL source NPZ, full/repair/resolved/clean terminal manifests, resolved prediction NPZ materialization, source/snapshot hashes. No TEST data, no training.",
        "errors": state.errors,
        "warnings": state.warnings,
        "checks": {
            "datasets": dataset_checks,
            "windows_and_postfit": window_checks,
            "prediction_npzs": prediction_checks,
            "source_hashes": source_hash_checks,
            "protected_access": access,
        },
    }
    write_json(output_dir / "audit.json", summary)
    (output_dir / "audit.md").write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "prediction_npz_count": prediction_checks["prediction_npz_count"],
                "checked_method_count": prediction_checks["checked_method_count"],
                "errors": len(state.errors),
                "warnings": len(state.warnings),
                "output": rel(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="validation_source_v1")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
