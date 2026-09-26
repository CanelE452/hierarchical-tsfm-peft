from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V1 = ROOT / "research" / "tsfm_peft_development_20260926"
CACHE = ROOT / ".cache" / "tsfm_peft_development_20260926"
PLOTS = OUT / "plots"
HORIZON = 48


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def windows(data: dict[str, np.ndarray], origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.stack([data["x"][o : o + HORIZON] for o in origins])
    mask = np.stack([data["targetmask"][o : o + HORIZON] for o in origins])
    return target, mask


def score(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, rows: np.ndarray | None = None) -> dict:
    if rows is not None:
        pred, target, mask = pred[rows], target[rows], mask[rows]
    count = mask.sum(axis=(0, 1)).astype(np.float64)
    valid = count > 0
    sq = (pred.astype(np.float64) - target.astype(np.float64)) ** 2 * mask
    ae = np.abs(pred.astype(np.float64) - target.astype(np.float64)) * mask
    channel_mse = np.full(count.shape, np.nan, dtype=np.float64)
    channel_mae = np.full(count.shape, np.nan, dtype=np.float64)
    channel_mse[valid] = sq.sum(axis=(0, 1))[valid] / count[valid]
    channel_mae[valid] = ae.sum(axis=(0, 1))[valid] / count[valid]
    return {
        "mse": float(np.nanmean(channel_mse)),
        "mae": float(np.nanmean(channel_mae)),
        "channel_mse": channel_mse.tolist(),
        "channel_mae": channel_mae.tolist(),
        "target_counts": count.astype(int).tolist(),
        "origins": int(pred.shape[0]),
    }


def seed_mean_scores(preds: list[np.ndarray], target: np.ndarray, mask: np.ndarray, rows: np.ndarray | None = None) -> dict:
    parts = [score(p, target, mask, rows) for p in preds]
    channel_mse = np.mean([p["channel_mse"] for p in parts], axis=0)
    channel_mae = np.mean([p["channel_mae"] for p in parts], axis=0)
    return {
        "mse": float(np.mean([p["mse"] for p in parts])),
        "mae": float(np.mean([p["mae"] for p in parts])),
        "channel_mse": channel_mse.tolist(),
        "channel_mae": channel_mae.tolist(),
        "seed_scores": parts,
    }


def compare(current: dict, reference: dict, channels: list[str]) -> dict:
    cur = np.asarray(current["channel_mse"], dtype=np.float64)
    ref = np.asarray(reference["channel_mse"], dtype=np.float64)
    delta = cur - ref
    denom = float(delta.sum())
    order = np.argsort(-delta)
    return {
        "current_mse": current["mse"],
        "reference_mse": reference["mse"],
        "current_minus_reference": float(current["mse"] - reference["mse"]),
        "gain_pct": float(100.0 * (1.0 - current["mse"] / reference["mse"])),
        "channels_worse_count": int((delta > 0).sum()),
        "signed_delta_sum": denom,
        "per_channel": [
            {
                "channel": channels[i],
                "current_mse": float(cur[i]),
                "reference_mse": float(ref[i]),
                "delta": float(delta[i]),
                "signed_delta_share": float(delta[i] / denom) if denom else math.nan,
            }
            for i in range(len(channels))
        ],
        "top_signed_excess": [
            {
                "channel": channels[i],
                "delta": float(delta[i]),
                "signed_delta_share": float(delta[i] / denom) if denom else math.nan,
            }
            for i in order[: min(6, len(channels))]
        ],
    }


def pred_hash_check(path: Path, expected: str | None, origins: np.ndarray) -> dict:
    actual = sha256(path)
    archive = load_npz(path)
    ok_hash = expected is None or actual == expected
    ok_origins = np.array_equal(archive["origins"], origins)
    if not ok_hash:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    if not ok_origins:
        raise RuntimeError(f"origins mismatch for {path}")
    return {"path": str(path.relative_to(ROOT)), "sha256": actual, "origins_match": ok_origins}


def load_checkpoint_state(run_id: str) -> dict:
    result = read_json(V1 / "runs" / run_id / "result.json")
    path = Path(result["checkpoint"])
    if sha256(path) != result["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {run_id}")
    # Local checkpoint produced by the previous approved run; no model code is executed.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {"checkpoint": str(path.relative_to(ROOT)), "result": result, "state": ckpt["state"], "spec": ckpt["spec"]}


def q_info(decoder_weight: np.ndarray) -> dict:
    d = decoder_weight.astype(np.float64)
    u, s, vt = np.linalg.svd(d, full_matrices=False)
    tol = float(max(d.shape) * np.finfo(np.float64).eps * s[0]) if s.size else 0.0
    rank = int((s > tol).sum())
    q = d @ np.linalg.pinv(d)
    return {
        "Q": q,
        "singular_values": s.tolist(),
        "rank_tol": tol,
        "numerical_rank": rank,
        "symmetry_max_abs": float(np.max(np.abs(q - q.T))),
        "idempotence_max_abs": float(np.max(np.abs(q @ q - q))),
        "trace": float(np.trace(q)),
    }


def decompose_error(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, q: np.ndarray) -> dict:
    complete = mask.all(axis=2)
    e = (pred.astype(np.float64) - target.astype(np.float64))[complete]
    if e.size == 0:
        return {"complete_vectors": 0, "status": "unavailable"}
    common = e @ q
    orth = e - common
    total = float(np.mean(e * e))
    common_mse = float(np.mean(common * common))
    orth_mse = float(np.mean(orth * orth))
    cross = float(2.0 * np.mean(common * orth))
    return {
        "complete_vectors": int(e.shape[0]),
        "total_mse_vector_mean": total,
        "common_Q_mse": common_mse,
        "orthogonal_mse": orth_mse,
        "cross_term": cross,
        "reconstruction_error": float(abs(total - common_mse - orth_mse - cross)),
        "common_fraction_of_total": float(common_mse / total) if total else math.nan,
        "orthogonal_fraction_of_total": float(orth_mse / total) if total else math.nan,
    }


def curve_summary(run_ids: list[str]) -> dict:
    out = {}
    for run_id in run_ids:
        curve = read_json(V1 / "runs" / run_id / "curve.json")
        val = np.asarray([r["val_mse"] for r in curve], dtype=np.float64)
        best_idx = int(np.argmin(val))
        out[run_id] = {
            "records": len(curve),
            "initial_val_mse": float(val[0]),
            "best_epoch": int(curve[best_idx]["epoch"]),
            "best_val_mse": float(val[best_idx]),
            "last_epoch": int(curve[-1]["epoch"]),
            "last_val_mse": float(val[-1]),
            "last_lr": float(curve[-1]["lr"]),
            "last_train_loss": curve[-1]["train_loss"],
            "elapsed_s_last_record": float(curve[-1]["elapsed_s"]),
        }
    return out


def aggregate_raw_curve(data: dict[str, np.ndarray], origins: np.ndarray, preds: dict[str, np.ndarray], channel_idx: int) -> dict:
    start, stop = int(origins.min()), int(origins.max() + HORIZON)
    time_ns = data["times"][start:stop]
    actual = data["raw"][start:stop, channel_idx].astype(np.float64)
    mean = float(data["mean"][channel_idx])
    std = float(data["std"][channel_idx])
    agg = {}
    for name, pred in preds.items():
        sums = np.zeros(stop - start, dtype=np.float64)
        counts = np.zeros(stop - start, dtype=np.float64)
        for oi, origin in enumerate(origins):
            idx = np.arange(int(origin), int(origin) + HORIZON) - start
            vals = pred[oi, :, channel_idx].astype(np.float64) * std + mean
            sums[idx] += vals
            counts[idx] += 1.0
        agg[name] = np.where(counts > 0, sums / counts, np.nan)
    return {"time_ns": time_ns, "actual": actual, "predictions": agg}


def save_plots(results: dict, bull_data: dict[str, np.ndarray], bull_preds: dict, bull_target: np.ndarray) -> list[str]:
    PLOTS.mkdir(parents=True, exist_ok=True)
    paths = []

    e1 = results["comparisons"]["bull_e1"]
    channels = results["datasets"]["bull"]["channels"]
    del_lora = [r["delta"] for r in e1["current_vs_lora"]["per_channel"]]
    del_factor = [r["delta"] for r in e1["current_vs_factor"]["per_channel"]]
    y = np.arange(len(channels))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(y - 0.18, del_lora, height=0.35, label="CURRENT - LoRA")
    ax.barh(y + 0.18, del_factor, height=0.35, label="CURRENT - FACTOR")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("Bull_office_", "") for c in channels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MSE delta on normalized targets (positive = CURRENT worse)")
    ax.set_title("Bull E1 channel losses from stored predictions")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = PLOTS / "diagnostic_bull_e1_channel_excess.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    q_summary = results["qspace_summary_plot"]
    labels = [x["label"] for x in q_summary]
    common = [x["residual_minus_lora_common_Q_mse"] for x in q_summary]
    orth = [x["residual_minus_lora_orthogonal_mse"] for x in q_summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(x - 0.18, common, width=0.35, label="Q-space excess")
    ax.bar(x + 0.18, orth, width=0.35, label="Orthogonal excess")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Vector MSE delta vs LoRA")
    ax.set_title("Selected decoder-space error decomposition")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = PLOTS / "diagnostic_qspace_residual_vs_lora.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    e1_origins = bull_data["eval_e1_origins"]
    seed_mean_residual = np.mean(bull_preds["e1"]["residual"], axis=0)
    seed_mean_lora = np.mean(bull_preds["e1"]["lora"], axis=0)
    factor = bull_preds["e1"]["factor_linear"]
    curve_preds = {"actual": None, "CURRENT": seed_mean_residual, "LoRA": seed_mean_lora, "FACTOR": factor}
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True)
    for ax, cname in zip(axes, ["Bull_office_Mai", "Bull_office_Marco"], strict=True):
        ci = channels.index(cname)
        curve = aggregate_raw_curve(
            bull_data,
            e1_origins,
            {k: v for k, v in curve_preds.items() if v is not None},
            ci,
        )
        dates = np.asarray(curve["time_ns"], dtype="datetime64[ns]").astype("datetime64[ms]").astype(object)
        ax.plot(dates, curve["actual"], color="black", linewidth=1.2, label="actual")
        ax.plot(dates, curve["predictions"]["CURRENT"], linewidth=1.0, label="CURRENT")
        ax.plot(dates, curve["predictions"]["LoRA"], linewidth=1.0, label="LoRA")
        ax.plot(dates, curve["predictions"]["FACTOR"], linewidth=1.0, label="FACTOR")
        ax.set_title(cname.replace("Bull_office_", ""))
        ax.set_ylabel("raw value")
        ax.grid(True, linewidth=0.3, alpha=0.4)
    axes[0].legend(ncol=4, frameon=False, fontsize=8)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.suptitle("Bull E1 actual and overlapping forecast means")
    fig.tight_layout()
    path = PLOTS / "diagnostic_bull_e1_mai_marco_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    return paths


def main() -> None:
    started = time.perf_counter()
    PLOTS.mkdir(parents=True, exist_ok=True)

    protected_seal = read_json(V1 / "protected_data_seal.json")
    revision = read_json(V1 / "revision1_rank32_development.json")
    protected = read_json(V1 / "protected_evaluation.json")
    selection = read_json(V1 / "revision1_rank32_selection.json")

    electricity = load_npz(CACHE / "data" / "electricity_first32_l512_h48.npz")
    bull = load_npz(CACHE / "data" / "bdg2_bull_office17_l512_h48.npz")
    data_checks = {
        "electricity_sha256": sha256(CACHE / "data" / "electricity_first32_l512_h48.npz"),
        "bull_sha256": sha256(CACHE / "data" / "bdg2_bull_office17_l512_h48.npz"),
    }
    if data_checks["electricity_sha256"] != protected_seal["datasets"]["electricity_first32"]["sha256"]:
        raise RuntimeError("electricity data hash mismatch")
    if data_checks["bull_sha256"] != protected_seal["datasets"]["bdg2_bull_office"]["sha256"]:
        raise RuntimeError("bull data hash mismatch")

    row_by_arm_fit = {(r.get("arm"), r.get("fit")): r for r in revision["rows"]}
    elec_origins = electricity["dev_origins"]
    elec_target, elec_mask = windows(electricity, elec_origins)
    elec_paths = {
        "residual": [
            CACHE / "revision1_rank32_evaluation" / f"{fit}.npz"
            for fit in selection["residual"].get("selected_fit_ids", selection["residual"]["fits"])
        ],
        "lora": [
            CACHE / "revision1_rank32_evaluation" / f"{fit}.npz"
            for fit in selection["lora"].get("selected_fit_ids", selection["lora"]["fits"])
        ],
        "factor_linear": [CACHE / "baselines_fulltrain" / "factor_linear_0.001_dev.npz"],
    }
    elec_pred = {}
    artifact_checks = []
    for arm, paths in elec_paths.items():
        elec_pred[arm] = []
        for path in paths:
            fit = path.stem if arm != "factor_linear" else None
            expected = None
            if arm == "factor_linear":
                # This development baseline row records scores in revision1_rank32_development.json
                # and prediction verification in linear_fulltrain_baselines.json, but not a stored
                # prediction hash. Record the actual hash and verify origins/scores instead.
                expected = None
            else:
                expected = row_by_arm_fit[(arm, fit)]["prediction_sha256"]
            artifact_checks.append(pred_hash_check(path, expected, elec_origins))
            elec_pred[arm].append(load_npz(path)["prediction"])

    bull_pred = {period: {"residual": [], "lora": [], "factor_linear": None} for period in ["e1", "e2"]}
    prot_rows = {(r["period"], r["arm"], r.get("fit")): r for r in protected["rows"]}
    prot_rows_no_fit = {(r["period"], r["arm"]): r for r in protected["rows"] if r.get("fit") is None}
    for period in ["e1", "e2"]:
        origins = bull[f"eval_{period}_origins"]
        for arm, fit_ids in {
            "residual": [f"bull_final_residual_{seed}_0.001" for seed in [92601, 92602]],
            "lora": [f"bull_final_lora_{seed}_0.0001" for seed in [92601, 92602]],
        }.items():
            for fit in fit_ids:
                row = prot_rows[(period, arm, fit)]
                path = CACHE / "protected_evaluation" / row["prediction_cache"]
                artifact_checks.append(pred_hash_check(path, row["prediction_sha256"], origins))
                bull_pred[period][arm].append(load_npz(path)["prediction"])
        row = prot_rows_no_fit[(period, "factor_linear")]
        path = CACHE / "protected_evaluation" / row["prediction_cache"]
        artifact_checks.append(pred_hash_check(path, row["prediction_sha256"], origins))
        bull_pred[period]["factor_linear"] = load_npz(path)["prediction"]

    channels_e = [str(x) for x in electricity["columns"].tolist()]
    channels_b = [str(x) for x in bull["columns"].tolist()]
    mid_e = len(elec_origins) // 2
    comparisons = {}
    elec_current = seed_mean_scores(elec_pred["residual"], elec_target, elec_mask)
    elec_lora = seed_mean_scores(elec_pred["lora"], elec_target, elec_mask)
    elec_factor = seed_mean_scores(elec_pred["factor_linear"], elec_target, elec_mask)
    comparisons["electricity_dev"] = {
        "overall": {
            "current": elec_current,
            "lora": elec_lora,
            "factor": elec_factor,
        },
        "current_vs_lora": compare(elec_current, elec_lora, channels_e),
        "current_vs_factor": compare(elec_current, elec_factor, channels_e),
        "front_half": {
            "current_vs_lora": compare(
                seed_mean_scores(elec_pred["residual"], elec_target, elec_mask, np.arange(0, mid_e)),
                seed_mean_scores(elec_pred["lora"], elec_target, elec_mask, np.arange(0, mid_e)),
                channels_e,
            ),
            "current_vs_factor": compare(
                seed_mean_scores(elec_pred["residual"], elec_target, elec_mask, np.arange(0, mid_e)),
                seed_mean_scores(elec_pred["factor_linear"], elec_target, elec_mask, np.arange(0, mid_e)),
                channels_e,
            ),
        },
        "back_half": {
            "current_vs_lora": compare(
                seed_mean_scores(elec_pred["residual"], elec_target, elec_mask, np.arange(mid_e, len(elec_origins))),
                seed_mean_scores(elec_pred["lora"], elec_target, elec_mask, np.arange(mid_e, len(elec_origins))),
                channels_e,
            ),
            "current_vs_factor": compare(
                seed_mean_scores(elec_pred["residual"], elec_target, elec_mask, np.arange(mid_e, len(elec_origins))),
                seed_mean_scores(elec_pred["factor_linear"], elec_target, elec_mask, np.arange(mid_e, len(elec_origins))),
                channels_e,
            ),
        },
    }

    for period in ["e1", "e2"]:
        origins = bull[f"eval_{period}_origins"]
        target, mask = windows(bull, origins)
        mid = len(origins) // 2
        current = seed_mean_scores(bull_pred[period]["residual"], target, mask)
        lora = seed_mean_scores(bull_pred[period]["lora"], target, mask)
        factor = seed_mean_scores([bull_pred[period]["factor_linear"]], target, mask)
        comparisons[f"bull_{period}"] = {
            "overall": {"current": current, "lora": lora, "factor": factor},
            "current_vs_lora": compare(current, lora, channels_b),
            "current_vs_factor": compare(current, factor, channels_b),
            "front_half": {
                "current_vs_lora": compare(
                    seed_mean_scores(bull_pred[period]["residual"], target, mask, np.arange(0, mid)),
                    seed_mean_scores(bull_pred[period]["lora"], target, mask, np.arange(0, mid)),
                    channels_b,
                ),
                "current_vs_factor": compare(
                    seed_mean_scores(bull_pred[period]["residual"], target, mask, np.arange(0, mid)),
                    seed_mean_scores([bull_pred[period]["factor_linear"]], target, mask, np.arange(0, mid)),
                    channels_b,
                ),
            },
            "back_half": {
                "current_vs_lora": compare(
                    seed_mean_scores(bull_pred[period]["residual"], target, mask, np.arange(mid, len(origins))),
                    seed_mean_scores(bull_pred[period]["lora"], target, mask, np.arange(mid, len(origins))),
                    channels_b,
                ),
                "current_vs_factor": compare(
                    seed_mean_scores(bull_pred[period]["residual"], target, mask, np.arange(mid, len(origins))),
                    seed_mean_scores([bull_pred[period]["factor_linear"]], target, mask, np.arange(mid, len(origins))),
                    channels_b,
                ),
            },
        }

    qspace = []
    q_plot_rows = []
    for dataset, data, periods, run_prefix, lora_prefix in [
        (
            "electricity",
            electricity,
            {"dev": (elec_origins, elec_target, elec_mask)},
            "revision1_rank32_residual",
            None,
        ),
        (
            "bull",
            bull,
            {p: windows(bull, bull[f"eval_{p}_origins"]) + (bull[f"eval_{p}_origins"],) for p in ["e1", "e2"]},
            "bull_final_residual",
            None,
        ),
    ]:
        for seed_idx, seed in enumerate([92601, 92602]):
            run_id = f"{run_prefix}_{seed}_0.001"
            ck = load_checkpoint_state(run_id)
            d = ck["state"]["decoder.weight"].numpy()
            qi = q_info(d)
            q = qi.pop("Q")
            for period, arrays in periods.items():
                if dataset == "electricity":
                    origins, target, mask = arrays
                    pred_map = {
                        "current": elec_pred["residual"][seed_idx],
                        "lora": elec_pred["lora"][seed_idx],
                        "factor": elec_pred["factor_linear"][0],
                    }
                else:
                    target, mask, origins = arrays
                    pred_map = {
                        "current": bull_pred[period]["residual"][seed_idx],
                        "lora": bull_pred[period]["lora"][seed_idx],
                        "factor": bull_pred[period]["factor_linear"],
                    }
                decomp = {name: decompose_error(pred, target, mask, q) for name, pred in pred_map.items()}
                row = {
                    "dataset": dataset,
                    "period": period,
                    "seed": seed,
                    "run_id": run_id,
                    "decoder_Q": qi,
                    "decomposition": decomp,
                }
                qspace.append(row)
                if decomp["current"].get("status") != "unavailable":
                    q_plot_rows.append(
                        {
                            "label": f"{dataset}-{period}-s{str(seed)[-2:]}",
                            "residual_minus_lora_common_Q_mse": decomp["current"]["common_Q_mse"] - decomp["lora"]["common_Q_mse"],
                            "residual_minus_lora_orthogonal_mse": decomp["current"]["orthogonal_mse"] - decomp["lora"]["orthogonal_mse"],
                            "residual_minus_factor_common_Q_mse": decomp["current"]["common_Q_mse"] - decomp["factor"]["common_Q_mse"],
                            "residual_minus_factor_orthogonal_mse": decomp["current"]["orthogonal_mse"] - decomp["factor"]["orthogonal_mse"],
                        }
                    )

    mai_idx = channels_b.index("Bull_office_Mai")
    marco_idx = channels_b.index("Bull_office_Marco")
    bull_e1_lora = comparisons["bull_e1"]["current_vs_lora"]
    mai_marco_share = {
        "denominator": "sum over all signed channel deltas: CURRENT channel MSE minus LoRA channel MSE",
        "channels": ["Bull_office_Mai", "Bull_office_Marco"],
        "signed_share_sum": float(
            bull_e1_lora["per_channel"][mai_idx]["signed_delta_share"]
            + bull_e1_lora["per_channel"][marco_idx]["signed_delta_share"]
        ),
        "current_minus_lora_delta_sum": float(
            bull_e1_lora["per_channel"][mai_idx]["delta"] + bull_e1_lora["per_channel"][marco_idx]["delta"]
        ),
        "all_channel_delta_sum": bull_e1_lora["signed_delta_sum"],
    }

    curves = curve_summary(
        [
            "revision1_rank32_residual_92601_0.001",
            "revision1_rank32_residual_92602_0.001",
            "bull_final_residual_92601_0.001",
            "bull_final_residual_92602_0.001",
        ]
    )

    results = {
        "status": "complete",
        "scope": (
            "CPU-only diagnostics from stored v1 predictions/checkpoints; no training, no model inference, "
            "no GPU, no mutation of v1 artifacts."
        ),
        "created_utc": time.time(),
        "datasets": {
            "electricity": {
                "channels": channels_e,
                "origins": int(len(elec_origins)),
                "target_shape": list(elec_target.shape),
            },
            "bull": {
                "channels": channels_b,
                "period_origins": {p: int(len(bull[f"eval_{p}_origins"])) for p in ["e1", "e2"]},
            },
        },
        "artifact_checks": {
            "data": data_checks,
            "prediction_files": artifact_checks,
        },
        "comparisons": comparisons,
        "mai_marco_e1_lora_excess_share": mai_marco_share,
        "qspace": qspace,
        "qspace_summary_plot": q_plot_rows,
        "current_curve_summary": curves,
        "interpretation": {
            "observations": [],
            "hypotheses": [],
            "next_experiment_implication": "",
        },
    }

    # Lightweight interpretation from the computed values; keep causal claims out of observations.
    bull_e1_orth_excess = [
        r["residual_minus_lora_orthogonal_mse"]
        for r in q_plot_rows
        if r["label"].startswith("bull-e1")
    ]
    bull_e1_common_excess = [
        r["residual_minus_lora_common_Q_mse"]
        for r in q_plot_rows
        if r["label"].startswith("bull-e1")
    ]
    results["interpretation"]["observations"] = [
        (
            "Bull E1 CURRENT is much worse than LoRA in seed-mean channel macro MSE, while it is close to "
            "the factor-linear control."
        ),
        (
            "Mai and Marco account for a large signed share of Bull E1 CURRENT-LoRA excess, but most channels "
            "are still worse than LoRA; no channel deletion is justified by this diagnostic."
        ),
        (
            "Decoder-space decomposition is based on selected D only and complete observed channel vectors; "
            "it is not an additive proof for the masked macro metric."
        ),
    ]
    if bull_e1_orth_excess and np.mean(bull_e1_orth_excess) > np.mean(bull_e1_common_excess):
        results["interpretation"]["hypotheses"].append(
            "Bull E1 residual-vs-LoRA excess is larger outside the selected K4 decoder space, which is compatible with testing K4->8 after the first E/D-mode comparison."
        )
        results["interpretation"]["next_experiment_implication"] = (
            "If FIXED_ED/SLOW_ED do not reduce Bull loss, a K4->8 residual/raw paired follow-up has a data-driven rationale."
        )
    else:
        results["interpretation"]["hypotheses"].append(
            "Bull E1 excess is not concentrated outside Q enough to make K4->8 the first follow-up by itself; keep the linear control comparison central."
        )
        results["interpretation"]["next_experiment_implication"] = (
            "Prioritize E/D-mode and matched RAW comparisons; use K4->8 only if those leave a clear decoder-space-capacity pattern."
        )

    plot_paths = save_plots(results, bull, bull_pred, windows(bull, bull["eval_e1_origins"])[0])
    results["plots"] = plot_paths
    results["cpu_elapsed_s"] = time.perf_counter() - started

    tmp = OUT / "existing_diagnostics.json.tmp"
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    tmp.replace(OUT / "existing_diagnostics.json")
    print(
        json.dumps(
            {
                "status": "complete",
                "cpu_elapsed_s": results["cpu_elapsed_s"],
                "bull_e1_current_mse": comparisons["bull_e1"]["overall"]["current"]["mse"],
                "bull_e1_lora_mse": comparisons["bull_e1"]["overall"]["lora"]["mse"],
                "bull_e1_factor_mse": comparisons["bull_e1"]["overall"]["factor"]["mse"],
                "mai_marco_signed_share": mai_marco_share["signed_share_sum"],
                "plots": plot_paths,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
