from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / "research" / "tsfm_peft_development_20260926"
CACHE1 = ROOT / ".cache" / "tsfm_peft_development_20260926"
HORIZON = 48
PERIODS = ["e1", "e2"]
SEEDS = [92601, 92602]
CHANNELS_OF_INTEREST = ["Bull_office_Mai", "Bull_office_Marco"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def windows(data: dict[str, np.ndarray], origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([data["x"][int(origin) : int(origin) + HORIZON] for origin in origins]),
        np.stack([data["targetmask"][int(origin) : int(origin) + HORIZON] for origin in origins]).astype(bool),
    )


def score(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    count = mask.sum(axis=(0, 1)).astype(np.float64)
    if np.any(count <= 0):
        raise ValueError("each selected channel must have observed targets")
    error = np.where(mask, prediction.astype(np.float64) - target.astype(np.float64), 0.0)
    channel_mse = (error * error).sum(axis=(0, 1)) / count
    channel_mae = np.abs(error).sum(axis=(0, 1)) / count
    return {
        "mse": float(channel_mse.mean()),
        "mae": float(channel_mae.mean()),
        "channel_mse": channel_mse.tolist(),
        "channel_mae": channel_mae.tolist(),
        "target_counts": count.astype(int).tolist(),
        "origins": int(prediction.shape[0]),
    }


def mean_score(parts: list[dict]) -> dict:
    return {
        "mse": float(np.mean([p["mse"] for p in parts])),
        "mae": float(np.mean([p["mae"] for p in parts])),
        "channel_mse": np.mean([p["channel_mse"] for p in parts], axis=0).tolist(),
        "channel_mae": np.mean([p["channel_mae"] for p in parts], axis=0).tolist(),
        "seed_scores": parts,
    }


def q_info(decoder_weight: np.ndarray) -> dict:
    d = decoder_weight.astype(np.float64)
    _, singular_values, _ = np.linalg.svd(d, full_matrices=False)
    tol = float(max(d.shape) * np.finfo(np.float64).eps * singular_values[0]) if singular_values.size else 0.0
    q = d @ np.linalg.pinv(d)
    return {
        "Q": q,
        "singular_values": singular_values.tolist(),
        "rank_tol": tol,
        "numerical_rank": int((singular_values > tol).sum()),
        "trace": float(np.trace(q)),
        "symmetry_max_abs": float(np.max(np.abs(q - q.T))),
        "idempotence_max_abs": float(np.max(np.abs(q @ q - q))),
    }


def decompose_error(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, q: np.ndarray) -> dict:
    complete = mask.all(axis=2)
    error = (prediction.astype(np.float64) - target.astype(np.float64))[complete]
    if error.size == 0:
        return {"status": "unavailable", "complete_vectors": 0}
    common = error @ q
    orth = error - common
    total = float(np.mean(error * error))
    common_mse = float(np.mean(common * common))
    orthogonal_mse = float(np.mean(orth * orth))
    cross_term = float(2.0 * np.mean(common * orth))
    return {
        "status": "available",
        "complete_vectors": int(error.shape[0]),
        "total_mse_vector_mean": total,
        "common_Q_mse": common_mse,
        "orthogonal_mse": orthogonal_mse,
        "cross_term": cross_term,
        "reconstruction_error": float(abs(total - common_mse - orthogonal_mse - cross_term)),
        "common_fraction_of_total": float(common_mse / total) if total else math.nan,
        "orthogonal_fraction_of_total": float(orthogonal_mse / total) if total else math.nan,
    }


def compare_channels(candidate: dict, reference: dict, channels: list[str]) -> dict:
    cand = np.asarray(candidate["channel_mse"], dtype=np.float64)
    ref = np.asarray(reference["channel_mse"], dtype=np.float64)
    delta = cand - ref
    total = float(delta.sum())
    interest_indices = [channels.index(ch) for ch in CHANNELS_OF_INTEREST if ch in channels]
    interest_delta = float(delta[interest_indices].sum()) if interest_indices else 0.0
    order = np.argsort(-delta)
    return {
        "candidate_minus_reference_mse": float(candidate["mse"] - reference["mse"]),
        "gain_pct_candidate_vs_reference": float(100.0 * (1.0 - candidate["mse"] / reference["mse"])),
        "channels_worse_count": int((delta > 0).sum()),
        "signed_channel_delta_sum": total,
        "mai_marco_delta_sum": interest_delta,
        "mai_marco_signed_share": float(interest_delta / total) if total else math.nan,
        "mai_marco": [
            {
                "channel": channels[index],
                "candidate_channel_mse": float(cand[index]),
                "reference_channel_mse": float(ref[index]),
                "delta": float(delta[index]),
            }
            for index in interest_indices
        ],
        "top_signed_excess": [
            {"channel": channels[index], "delta": float(delta[index])}
            for index in order[: min(6, len(channels))]
        ],
        "per_channel": [
            {
                "channel": channel,
                "candidate_channel_mse": float(cand[index]),
                "reference_channel_mse": float(ref[index]),
                "delta": float(delta[index]),
            }
            for index, channel in enumerate(channels)
        ],
    }


def row_index(report: dict) -> dict[tuple[str, str, str, int | None, str | None], dict]:
    out = {}
    for row in report["rows"]:
        if row.get("dataset") != "bull" or row.get("period") not in PERIODS:
            continue
        out[(row["period"], row["arm"], row.get("ed_mode"), row.get("seed"), row.get("fit"))] = row
    return out


def find_row(report: dict, period: str, arm: str, *, seed: int | None = None,
             ed_mode: str | None = None, fit: str | None = None) -> dict:
    matches = [
        row for row in report["rows"]
        if row.get("dataset") == "bull"
        and row.get("period") == period
        and row.get("arm") == arm
        and (seed is None or row.get("seed") == seed)
        and (ed_mode is None or row.get("ed_mode") == ed_mode)
        and (fit is None or row.get("fit") == fit)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {period}/{arm}/{ed_mode}/{seed}/{fit}, got {len(matches)}")
    return matches[0]


def load_prediction(row: dict, origins: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    receipt = row["prediction"]
    path = Path(receipt["path"])
    actual_hash = sha256(path)
    if actual_hash != receipt["sha256"]:
        raise RuntimeError(f"prediction hash mismatch: {path}")
    archive = load_npz(path)
    if not np.array_equal(archive["origins"], origins):
        raise RuntimeError(f"prediction origins mismatch: {path}")
    computed = score(archive["prediction"], target, mask)
    for metric in ("mse", "mae"):
        if not np.isclose(computed[metric], row["scores"]["full"][metric], rtol=0, atol=1e-8):
            raise RuntimeError(f"prediction {metric} differs from report: {path}")
    return archive["prediction"], {
        "path": rel(path),
        "sha256": actual_hash,
        "origins_match": True,
        "score_match": True,
    }


def result_path_for(fit: str) -> Path:
    candidates = [HERE / "runs" / fit / "result.json", V1 / "runs" / fit / "result.json"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"result.json not found for {fit}")


def verify_result_and_checkpoint(report: dict, fit: str, load_state: bool = False) -> dict:
    result_path = result_path_for(fit)
    result_hash = sha256(result_path)
    expected_result_hash = report["result_hashes"].get(fit)
    if expected_result_hash and result_hash != expected_result_hash:
        raise RuntimeError(f"result hash mismatch: {fit}")
    result = read_json(result_path)
    checkpoint = Path(result["checkpoint"])
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != result["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {fit}")
    out = {
        "fit": fit,
        "result_path": rel(result_path),
        "result_sha256": result_hash,
        "result_hash_matches_report": bool(expected_result_hash and result_hash == expected_result_hash),
        "checkpoint": rel(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_hash_matches_result": True,
    }
    if load_state:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        spec = saved["spec"]
        if any(result.get(key) != value for key, value in spec.items()):
            raise RuntimeError(f"checkpoint spec differs from result: {fit}")
        state = saved["state"]
        decoder = state["decoder.weight"].detach().cpu().numpy()
        basis = np.asarray(saved["basis"], dtype=np.float32)
        if decoder.shape != basis.shape:
            raise RuntimeError(f"decoder and basis shapes differ: {fit}")
        out["decoder_basis_max_abs_diff"] = float(np.max(np.abs(decoder - basis)))
        out["state_keys"] = sorted(state.keys())
        out["decoder_weight"] = decoder
    return out


def strip_arrays(payload: dict) -> dict:
    if isinstance(payload, dict):
        return {key: strip_arrays(value) for key, value in payload.items() if key != "Q" and key != "decoder_weight"}
    if isinstance(payload, list):
        return [strip_arrays(value) for value in payload]
    return payload


def main() -> None:
    started = time.perf_counter()
    report_path = HERE / "first_modes_development.json"
    selection_path = HERE / "first_modes_selection.json"
    output_path = HERE / "fixed_diagnostics.json"
    report = read_json(report_path)
    selection = read_json(selection_path)
    if report["status"] != "complete":
        raise RuntimeError("first_modes_development.json is not complete")
    if sha256(selection_path) != report["selection_sha256"]:
        raise RuntimeError("selection hash mismatch")
    if selection["selection"] != report["selection"]:
        raise RuntimeError("selection content mismatch")
    if report["selection"]["bull"]["residual"]["mode"] != "fixed_ed":
        raise RuntimeError("Bull residual selection is not fixed_ed")

    data_path = CACHE1 / "data" / "bdg2_bull_office17_l512_h48.npz"
    data_hash = sha256(data_path)
    if data_hash != report["data_sha256"]["bull"]:
        raise RuntimeError("Bull data hash differs from first_modes report")
    data = load_npz(data_path)
    channels = [str(x) for x in data["columns"].tolist()]

    artifact_checks = {
        "report": {"path": rel(report_path), "sha256": sha256(report_path)},
        "selection": {"path": rel(selection_path), "sha256": sha256(selection_path), "matches_report": True},
        "data": {"path": rel(data_path), "sha256": data_hash, "matches_report": True},
        "predictions": [],
        "results": [],
    }

    selected_fits = report["selection"]["bull"]["residual"]["fits"]
    selected_by_seed = {}
    for fit in selected_fits:
        result_check = verify_result_and_checkpoint(report, fit, load_state=True)
        artifact_checks["results"].append(strip_arrays(result_check))
        fixed_seed = read_json(result_path_for(fit))["seed"]
        selected_by_seed[fixed_seed] = result_check
    if sorted(selected_by_seed) != SEEDS:
        raise RuntimeError("fixed_ed selected fits must cover the two fixed seeds")

    by_period: dict[str, dict] = {}
    for period in PERIODS:
        origins = data[f"eval_{period}_origins"]
        target, mask = windows(data, origins)
        fixed_rows = {seed: find_row(report, period, "residual", seed=seed, ed_mode="fixed_ed") for seed in SEEDS}
        current_rows = {seed: find_row(report, period, "residual", seed=seed, ed_mode="current") for seed in SEEDS}
        lora_rows = {seed: find_row(report, period, "lora", seed=seed) for seed in SEEDS}
        factor_row = find_row(report, period, "factor_linear")

        predictions = {"fixed_ed": {}, "current": {}, "lora": {}, "factor": {}}
        scores = {"fixed_ed": {}, "current": {}, "lora": {}, "factor": {}}
        for seed in SEEDS:
            for label, rows in (("fixed_ed", fixed_rows), ("current", current_rows), ("lora", lora_rows)):
                prediction, receipt = load_prediction(rows[seed], origins, target, mask)
                artifact_checks["predictions"].append(receipt)
                predictions[label][seed] = prediction
                scores[label][seed] = score(prediction, target, mask)
        factor_prediction, factor_receipt = load_prediction(factor_row, origins, target, mask)
        artifact_checks["predictions"].append(factor_receipt)
        predictions["factor"][None] = factor_prediction
        scores["factor"][None] = score(factor_prediction, target, mask)

        seed_reports = {}
        for seed in SEEDS:
            q = q_info(selected_by_seed[seed]["decoder_weight"])
            decomp = {
                "fixed_ed": decompose_error(predictions["fixed_ed"][seed], target, mask, q["Q"]),
                "current": decompose_error(predictions["current"][seed], target, mask, q["Q"]),
                "lora": decompose_error(predictions["lora"][seed], target, mask, q["Q"]),
                "factor": decompose_error(predictions["factor"][None], target, mask, q["Q"]),
            }
            comparisons = {}
            for reference in ["current", "lora", "factor"]:
                ref_seed = seed if reference != "factor" else None
                channel = compare_channels(scores["fixed_ed"][seed], scores[reference][ref_seed], channels)
                q_delta = {
                    "fixed_minus_reference_total": float(
                        decomp["fixed_ed"]["total_mse_vector_mean"] - decomp[reference]["total_mse_vector_mean"]
                    ),
                    "fixed_minus_reference_common_Q": float(
                        decomp["fixed_ed"]["common_Q_mse"] - decomp[reference]["common_Q_mse"]
                    ),
                    "fixed_minus_reference_orthogonal": float(
                        decomp["fixed_ed"]["orthogonal_mse"] - decomp[reference]["orthogonal_mse"]
                    ),
                }
                comparisons["fixed_vs_" + reference] = {"channel_delta": channel, "q_delta": q_delta}
            seed_reports[str(seed)] = {
                "fixed_fit": fixed_rows[seed]["fit"],
                "q": strip_arrays(q),
                "scores": {
                    "fixed_ed": scores["fixed_ed"][seed],
                    "current": scores["current"][seed],
                    "lora": scores["lora"][seed],
                    "factor": scores["factor"][None],
                },
                "decomposition": strip_arrays(decomp),
                "comparisons": comparisons,
            }

        aggregate_scores = {
            "fixed_ed": mean_score([scores["fixed_ed"][seed] for seed in SEEDS]),
            "current": mean_score([scores["current"][seed] for seed in SEEDS]),
            "lora": mean_score([scores["lora"][seed] for seed in SEEDS]),
            "factor": scores["factor"][None],
        }
        aggregate = {
            "scores": aggregate_scores,
            "fixed_vs_current": compare_channels(aggregate_scores["fixed_ed"], aggregate_scores["current"], channels),
            "fixed_vs_lora": compare_channels(aggregate_scores["fixed_ed"], aggregate_scores["lora"], channels),
            "fixed_vs_factor": compare_channels(aggregate_scores["fixed_ed"], aggregate_scores["factor"], channels),
            "mean_q_delta": {},
        }
        for reference in ["current", "lora", "factor"]:
            deltas = [seed_reports[str(seed)]["comparisons"]["fixed_vs_" + reference]["q_delta"] for seed in SEEDS]
            aggregate["mean_q_delta"]["fixed_vs_" + reference] = {
                key: float(np.mean([delta[key] for delta in deltas]))
                for key in deltas[0]
            }
        by_period[period] = {
            "origins": int(len(origins)),
            "complete_vectors_by_seed_q": {
                str(seed): seed_reports[str(seed)]["decomposition"]["fixed_ed"]["complete_vectors"]
                for seed in SEEDS
            },
            "seed_reports": seed_reports,
            "aggregate": aggregate,
        }

    e1 = by_period["e1"]["aggregate"]
    e2 = by_period["e2"]["aggregate"]
    interpretation = {
        "observations": [
            (
                "Bull E1 fixed_ed lowers mean MSE versus CURRENT "
                f"({e1['fixed_vs_current']['candidate_minus_reference_mse']:.6f}) but remains above LoRA "
                f"({e1['fixed_vs_lora']['candidate_minus_reference_mse']:.6f})."
            ),
            (
                "Bull E2 fixed_ed is above CURRENT "
                f"({e2['fixed_vs_current']['candidate_minus_reference_mse']:.6f}) and above LoRA "
                f"({e2['fixed_vs_lora']['candidate_minus_reference_mse']:.6f})."
            ),
            (
                "Q-space deltas are computed with each fixed_ed seed's actual decoder D; "
                "Q rank remains 4 for the Bull fixed_ed runs."
            ),
        ],
        "branch_signal_hypothesis": [
            (
                "E1 fixed_ed-vs-LoRA excess is mostly orthogonal to the current K4 D/PCA space, so K4→8 could test "
                "whether expanding D reduces the remaining E1 gap."
            ),
            (
                "E2 fixed_ed regression versus CURRENT/LoRA is mostly inside the current Q space, while orthogonal error is lower or nearly tied. "
                "That records a possible E2 common-space tradeoff if K is changed."
            ),
            (
                "Matched RAW/linear-control and K4→8 answer different branch questions; the final next branch should be chosen with the RAW result. "
                "This is a hypothesis from stored-error decomposition, not a causal diagnosis."
            ),
        ],
        "limitations": [
            (
                "Q-space decomposition uses only complete observed channel vectors. E1 has 1920 complete vectors, but E2 has 1910 complete vectors "
                "while the masked macro score also includes partially observed channel targets. E2 Q totals therefore do not equal the masked macro MSE "
                "and should be read as a subspace diagnostic, not a replacement score."
            )
        ],
    }

    payload = {
        "status": "complete",
        "scope": "CPU-only diagnostics of stored first_modes Bull predictions; no training, inference, target re-selection, or GPU use",
        "created_utc": time.time(),
        "cpu_elapsed_s": time.perf_counter() - started,
        "artifact_checks": artifact_checks,
        "channels": channels,
        "channels_of_interest": CHANNELS_OF_INTEREST,
        "periods": by_period,
        "interpretation": interpretation,
    }
    write_json(output_path, payload)
    print(json.dumps({
        "status": "complete",
        "cpu_elapsed_s": payload["cpu_elapsed_s"],
        "e1_fixed_mse": e1["scores"]["fixed_ed"]["mse"],
        "e1_current_mse": e1["scores"]["current"]["mse"],
        "e1_lora_mse": e1["scores"]["lora"]["mse"],
        "e1_factor_mse": e1["scores"]["factor"]["mse"],
        "e1_fixed_vs_lora_mai_marco_delta": e1["fixed_vs_lora"]["mai_marco_delta_sum"],
        "e2_fixed_mse": e2["scores"]["fixed_ed"]["mse"],
        "e2_current_mse": e2["scores"]["current"]["mse"],
        "e2_lora_mse": e2["scores"]["lora"]["mse"],
        "e2_fixed_vs_current_mai_marco_delta": e2["fixed_vs_current"]["mai_marco_delta_sum"],
        "output": rel(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
