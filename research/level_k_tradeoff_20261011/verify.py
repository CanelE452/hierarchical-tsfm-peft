"""Read-only verifier for the K tradeoff / professor-method study.

This verifier only inspects saved JSON/NPZ/checkpoint/benchmark receipts.  It
must not train, run TSFM forward, or change science artifacts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
PARENT_RESULT = ROOT / "research" / "level_budget_recheck_20261011" / "result.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sample_epoch_origins_np(train_origins: np.ndarray, seed: int, epoch: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    sampled = None
    for _ in range(int(epoch) + 1):
        sampled = rng.choice(train_origins, size=int(count), replace=True)
    return np.asarray(sampled, dtype=np.int64)


def state_hash_from_state_dict(state: dict[str, Any]) -> str | None:
    try:
        h = hashlib.sha256()
        for name, value in sorted(state.items()):
            if hasattr(value, "detach"):
                arr = value.detach().cpu().contiguous().numpy()
            else:
                arr = np.ascontiguousarray(value)
            h.update(str(name).encode())
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        return h.hexdigest()
    except Exception:
        return None


def torch_load_cpu(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def optimizer_step_values(optimizer_state: dict[str, Any] | None) -> list[int]:
    if not isinstance(optimizer_state, dict):
        return []
    out: list[int] = []
    for state in (optimizer_state.get("state") or {}).values():
        if not isinstance(state, dict) or "step" not in state:
            continue
        value = state["step"]
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        out.append(int(value))
    return out


def hash_windows_incremental(x: np.ndarray, origins: np.ndarray, context: int) -> str:
    h = hashlib.sha256()
    for origin in np.asarray(origins, dtype=np.int64):
        window = np.ascontiguousarray(x[int(origin) - int(context):int(origin)].astype(np.float32, copy=False))
        h.update(window.tobytes())
    return h.hexdigest()


def resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


class Verifier:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, name: str, passed: bool, **details: Any) -> None:
        self.checks.append({"name": name, "passed": bool(passed), **details})
        if not passed:
            suffix = " " + json.dumps(details, ensure_ascii=False, sort_keys=True) if details else ""
            self.failures.append(name + suffix)

    def warn(self, name: str, **details: Any) -> None:
        suffix = " " + json.dumps(details, ensure_ascii=False, sort_keys=True) if details else ""
        self.warnings.append(name + suffix)


def load_required(ver: Verifier, path: Path, label: str) -> Any | None:
    if not path.exists():
        ver.check(f"required file exists:{label}", False, path=str(path.relative_to(ROOT)))
        return None
    ver.check(f"required file exists:{label}", True, path=str(path.relative_to(ROOT)))
    try:
        return read_json(path)
    except Exception as exc:
        ver.check(f"valid json:{label}", False, error=repr(exc))
        return None


def metric(pred: np.ndarray, target: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, op in (("mse", np.square), ("mae", np.abs)):
        vals = []
        for c in range(target.shape[-1]):
            mask = observed[..., c]
            if np.any(mask):
                err = pred[..., c][mask].astype(np.float64) - target[..., c][mask].astype(np.float64)
                vals.append(float(op(err).mean()))
        if not vals:
            raise ValueError("no observed values")
        out[name] = float(np.mean(vals))
    return out


def grouped_metrics(pred: np.ndarray, target: np.ndarray, observed: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        "all": metric(pred[:, :48], target[:, :48], observed[:, :48]),
        "h1_8": metric(pred[:, :8], target[:, :8], observed[:, :8]),
        "h9_48": metric(pred[:, 8:48], target[:, 8:48], observed[:, 8:48]),
    }


def future_features_np(z_history: np.ndarray, z_future: np.ndarray, lag: int) -> np.ndarray:
    prefix = z_history[:, -lag:] if lag else z_history[:, :0]
    if lag and prefix.shape[1] < lag:
        pad = np.zeros((z_history.shape[0], lag - prefix.shape[1], z_history.shape[2]), dtype=z_history.dtype)
        prefix = np.concatenate([pad, prefix], axis=1)
    full = np.concatenate([prefix, z_future], axis=1)
    parts = [full[:, lag - ell : lag - ell + z_future.shape[1], :] for ell in range(lag + 1)]
    return np.concatenate(parts, axis=2).astype(np.float64)


def apply_coeff_np(features: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return np.einsum("btd,dc->btc", features.astype(np.float64), coeff.astype(np.float64)).astype(np.float32)


def coeff_to_taps_np(coeff: np.ndarray, lag: int, latent: int) -> np.ndarray:
    return coeff.reshape(lag + 1, latent, coeff.shape[1]).astype(np.float64)


def deterministic_decoder_origins(archive: np.lib.npyio.NpzFile, protocol: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(protocol["decoder_sampling_seed"]))
    return np.sort(
        rng.choice(np.asarray(archive["train_origins"]), size=int(protocol["decoder_train_origin_count"]), replace=False)
    ).astype(np.int64)


def future_targets_from_archive(archive: np.lib.npyio.NpzFile, origins: np.ndarray, horizon: int = 48) -> tuple[np.ndarray, np.ndarray]:
    x = archive["x"]
    finite = archive["finite"]
    y = np.stack([x[int(o) : int(o) + horizon] for o in origins]).astype(np.float32)
    m = np.stack([finite[int(o) : int(o) + horizon] for o in origins]).astype(bool)
    return y, m


def windows_from_archive(archive: np.lib.npyio.NpzFile, origins: np.ndarray, context: int = 512) -> np.ndarray:
    x = archive["x"]
    return np.stack([x[int(o) - context : int(o)] for o in origins]).astype(np.float32)


def ridge_coeff_from_trace(
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
    train_row_mask: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    rows = train_row_mask.astype(bool).reshape(-1)
    flat = train_features.reshape(-1, train_features.shape[-1])
    rms = np.sqrt(np.mean(np.square(flat[rows]), axis=0))
    scale = np.maximum(rms, 1e-12)
    x_train = train_features / scale
    dim = train_features.shape[-1]
    channels = train_target.shape[-1]
    xtx = np.zeros((channels, dim, dim), dtype=np.float64)
    xty = np.zeros((channels, dim), dtype=np.float64)
    counts = np.zeros(channels, dtype=np.int64)
    for c in range(channels):
        obs = train_mask[:, :, c].astype(bool) & train_row_mask.astype(bool)
        if np.any(obs):
            x = x_train[obs]
            y = train_target[:, :, c][obs].astype(np.float64)
            xtx[c] = x.T @ x
            xty[c] = x.T @ y
            counts[c] = int(obs.sum())
    coeff_scaled = np.zeros((dim, channels), dtype=np.float64)
    for c in range(channels):
        if counts[c]:
            coeff_scaled[:, c] = np.linalg.solve(xtx[c] + float(lam) * np.eye(dim, dtype=np.float64), xty[c])
    return coeff_scaled / scale[:, None], scale.astype(np.float64), int(rows.sum()), sha256_array(train_row_mask.astype(np.uint8))


def pca_basis_from_train_archive(archive: np.lib.npyio.NpzFile, k: int) -> np.ndarray:
    train = np.asarray(archive["x"], dtype=np.float64)[int(archive["train_start"]) : int(archive["train_end"])]
    centered = train - train.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, np.argsort(vals)[::-1][:k]].astype(np.float32)


def projection_close(left: np.ndarray, right: np.ndarray, atol: float = 2e-5) -> tuple[bool, float]:
    if left.shape != right.shape:
        return False, math.inf
    diff = float(np.max(np.abs(left.astype(np.float64) @ left.astype(np.float64).T - right.astype(np.float64) @ right.astype(np.float64).T)))
    return diff <= atol, diff


def iter_named(container: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(container, dict):
        return [(str(k), v) for k, v in container.items() if isinstance(v, dict)]
    if isinstance(container, list):
        return [(str(v.get("id", i)), v) for i, v in enumerate(container) if isinstance(v, dict)]
    return []


def infer_family(item: dict[str, Any], key: str = "") -> str | None:
    text = str(item.get("family") or item.get("method") or item.get("kind") or item.get("meta", {}).get("family") or key)
    up = text.upper()
    if "RAW_C" in up or ("RAW" in up and "D8" in up):
        return "RAW_C_D8"
    if "RAW" in up:
        return "RAW_F0"
    if "PCA" in up:
        return "PCA"
    if "MLP" in up:
        return "MLP_POINT"
    if "TS2" in up or "PROF" in up or "METHOD" in up:
        return "PROFESSOR_METHOD"
    return text if text else None


def infer_k(item: dict[str, Any], key: str = "") -> int | None:
    for source in (item, item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}):
        for name in ("k", "K", "latent", "output_k", "latent_dim"):
            if name in source:
                return int(source[name])
    m = re.search(r"(?:^|[_-])k(\d+)(?:$|[_-])", key, re.IGNORECASE)
    return int(m.group(1)) if m else None


def infer_seed(item: dict[str, Any], key: str = "") -> int | None:
    for source in (item, item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}):
        if "seed" in source:
            return int(source["seed"])
    m = re.search(r"seed(\d+)", key, re.IGNORECASE)
    return int(m.group(1)) if m else None


def collect_fits(result: dict[str, Any], dataset: str) -> list[tuple[str, dict[str, Any]]]:
    data = (result.get("datasets") or {}).get(dataset, {})
    out: list[tuple[str, dict[str, Any]]] = []
    for name in ("fits", "neural", "fit_records", "method_fits"):
        for key, item in iter_named(data.get(name)):
            item.setdefault("dataset", dataset)
            item.setdefault("id", key)
            out.append((key, item))
    return out


def collect_adapters(result: dict[str, Any], dataset: str) -> list[tuple[str, dict[str, Any]]]:
    data = (result.get("datasets") or {}).get(dataset, {})
    out: list[tuple[str, dict[str, Any]]] = []
    for name in ("decoder_contracts", "adapters", "representations", "method_adapters", "controls"):
        for key, item in iter_named(data.get(name)):
            item.setdefault("dataset", dataset)
            item.setdefault("id", key)
            out.append((key, item))
    return out


def get_contract(adapter: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any] | None:
    wanted = protocol["decoder_contract"]
    for name in ("contracts", "decoder_contracts"):
        container = adapter.get(name)
        if isinstance(container, dict):
            if isinstance(container.get(wanted), dict):
                return container[wanted]
            if len(container) == 1 and isinstance(next(iter(container.values())), dict):
                return next(iter(container.values()))
    if adapter.get("contract") == wanted or "prediction_npz" in adapter:
        return adapter
    return None


def find_adapter(adapters: list[tuple[str, dict[str, Any]]], family: str, k: int | None = None, seed: int | None = None) -> dict[str, Any] | None:
    for key, item in adapters:
        fam = infer_family(item, key)
        ok = fam == family if family != "PROFESSOR_METHOD" else fam not in {"PCA", "MLP_POINT", "RAW_F0", "RAW_C_D8"}
        if not ok:
            continue
        if k is not None and infer_k(item, key) != k:
            continue
        if seed is not None and infer_seed(item, key) != seed:
            continue
        return item
    return None


def find_fit(fits: list[tuple[str, dict[str, Any]]], family: str, k: int, seed: int) -> dict[str, Any] | None:
    for key, item in fits:
        fam = infer_family(item, key)
        fam_ok = fam == family if family != "PROFESSOR_METHOD" else fam not in {"PCA", "MLP_POINT", "RAW_F0", "RAW_C_D8"}
        if fam_ok and infer_k(item, key) == k and infer_seed(item, key) == seed:
            return item
    return None


def expected_epoch_from_curve(rows: list[dict[str, Any]], max_epoch: int) -> int:
    def mse(row: dict[str, Any]) -> float:
        if "val_reconstruction" in row:
            return float(row["val_reconstruction"]["mse"])
        return float(row["val_reconstruction_mse"])

    return int(min((r for r in rows if int(r["epoch"]) <= max_epoch), key=lambda r: (mse(r), int(r["epoch"])))["epoch"])


def curve_rows_from_fit(ver: Verifier, label: str, fit: dict[str, Any]) -> list[dict[str, Any]]:
    curve = fit.get("curve") or fit.get("curve_path")
    if not isinstance(curve, str):
        ver.check("fit curve path:" + label, False, curve=curve)
        return []
    path = resolve(curve)
    ver.check("fit curve exists:" + label, path is not None and path.exists(), curve=curve)
    if path is None or not path.exists():
        return []
    if isinstance(fit.get("curve_sha256"), str):
        ver.check("fit curve sha256:" + label, sha256_file(path) == fit["curve_sha256"], expected=fit["curve_sha256"], actual=sha256_file(path))
    obj = read_json(path)
    rows = obj.get("curves", obj if isinstance(obj, list) else [])
    ver.check("fit curve rows present:" + label, isinstance(rows, list) and bool(rows), count=len(rows) if isinstance(rows, list) else None)
    if isinstance(obj, dict) and "optimizer_steps_completed" in obj:
        ver.check("fit curve optimizer total:" + label, int(obj["optimizer_steps_completed"]) == int(fit.get("optimizer_steps_completed", -1)), curve_steps=obj.get("optimizer_steps_completed"), fit_steps=fit.get("optimizer_steps_completed"))
    return rows if isinstance(rows, list) else []


def check_failed_prefix_recovery(ver: Verifier, label: str, fit: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if "robin:MLP:K2:seed92601" not in label:
        return
    archived_curve = HERE / "failed_attempts" / "attempt1" / "cache" / "curves" / "robin_MLP_POINT_K2_seed92601.json"
    if not archived_curve.exists():
        ver.warn("failed prefix recovery archive curve absent", path=str(archived_curve.relative_to(ROOT)))
        return
    try:
        archived = read_json(archived_curve)
        archived_rows = archived.get("curves", [])
        archived_receipts = archived.get("epoch_origin_receipts", [])
        prefix_rows = rows[:len(archived_rows)]
        numeric_pairs = []
        for left, right in zip(archived_rows, prefix_rows):
            if int(left.get("epoch", -1)) != int(right.get("epoch", -2)):
                numeric_pairs.append(float("inf"))
                continue
            for key in ("mse", "mae"):
                if "val_reconstruction" in left and "val_reconstruction" in right and key in left["val_reconstruction"] and key in right["val_reconstruction"]:
                    numeric_pairs.append(abs(float(left["val_reconstruction"][key]) - float(right["val_reconstruction"][key])))
            if left.get("train_loss_mean") is not None and right.get("train_loss_mean") is not None:
                numeric_pairs.append(abs(float(left["train_loss_mean"]) - float(right["train_loss_mean"])))
        max_diff = max(numeric_pairs) if numeric_pairs else float("inf")
        success_curve = read_json(resolve(fit["curve"])) if isinstance(fit.get("curve"), str) and resolve(fit.get("curve")) and resolve(fit.get("curve")).exists() else {}
        success_receipts = success_curve.get("epoch_origin_receipts", [])
        sampling_equal = archived_receipts[:62] == success_receipts[:62]
        if len(prefix_rows) == len(archived_rows) and max_diff <= 5e-5 and sampling_equal:
            ver.check("failed prefix recovery parity:robin_K2_seed92601", True, compared_rows=len(archived_rows), max_numeric_abs=max_diff)
        else:
            ver.warn("failed prefix recovery differs:robin_K2_seed92601", compared_rows=min(len(prefix_rows), len(archived_rows)), archived_rows=len(archived_rows), success_prefix=len(prefix_rows), max_numeric_abs=max_diff, sampling_equal=sampling_equal)
    except Exception as exc:
        ver.warn("failed prefix recovery audit error", error=repr(exc))


def check_fit(ver: Verifier, label: str, fit: dict[str, Any], protocol: dict[str, Any], expect_new: bool, archive: np.lib.npyio.NpzFile | None = None) -> bool:
    if not expect_new:
        reused = fit.get("reused_from_parent") is True or fit.get("reused") is True or fit.get("reuse") is True or fit.get("source") or fit.get("parent_checkpoint_sha256")
        ver.check("middle-K reuse receipt:" + label, bool(reused), fit_keys=sorted(fit))
        ver.check("middle-K parent adapter receipt:" + label, isinstance(fit.get("parent_adapter"), str) and isinstance(fit.get("parent_fit_id"), str), parent_adapter=fit.get("parent_adapter"), parent_fit_id=fit.get("parent_fit_id"))
        return False
    updates = int(fit.get("actual_updates", fit.get("optimizer_steps_completed", fit.get("updates", -1))))
    ver.check("fit updates:" + label, updates == int(protocol["total_updates_per_fit"]), actual=updates)
    rows = curve_rows_from_fit(ver, label, fit)
    if rows:
        epoch_rows = [r for r in rows if int(r.get("epoch", -1)) > 0]
        counts = [int(r.get("optimizer_steps_this_epoch", -1)) for r in epoch_rows]
        ver.check("fit epoch update counts:" + label, len(counts) == int(protocol["max_epochs"]) and all(x == int(protocol["updates_per_epoch"]) for x in counts), count=len(counts), bad=[x for x in counts if x != int(protocol["updates_per_epoch"])][:5])
        selected = fit.get("selected_epoch", fit.get("best_epoch"))
        expected_epoch = expected_epoch_from_curve(rows, int(protocol["max_epochs"]))
        ver.check("fit earliest VAL reconstruction selection:" + label, int(selected) == expected_epoch, expected=expected_epoch, actual=selected)
        if archive is not None and "train_origins" in archive.files:
            receipts = read_json(resolve(fit["curve"])).get("epoch_origin_receipts", []) if isinstance(fit.get("curve"), str) and resolve(fit.get("curve")) and resolve(fit.get("curve")).exists() else []
            recomputed = []
            for epoch in range(1, int(protocol["max_epochs"]) + 1):
                sample = sample_epoch_origins_np(np.asarray(archive["train_origins"]), int(fit["seed"]), epoch - 1, int(protocol["epoch_origins"]))
                recomputed.append({"epoch": epoch, "sha256": sha256_array(sample), "count": int(len(sample))})
            ver.check("fit sampling SHA recompute:" + label, receipts == recomputed, expected_sha=hashlib.sha256(json.dumps(recomputed, sort_keys=True).encode()).hexdigest(), actual_sha=hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest() if isinstance(receipts, list) else None)
    receipt_hash = fit.get("origin_stream_sha256", fit.get("sample_stream_sha256", fit.get("train_sampling_sha256", fit.get("epoch_origin_receipts_sha256"))))
    ver.check("fit stream hash receipt:" + label, isinstance(receipt_hash, str), receipt=receipt_hash)
    ver.check("fit epoch origin receipt count:" + label, int(fit.get("epoch_origin_receipt_count", -1)) == int(protocol["max_epochs"]), actual=fit.get("epoch_origin_receipt_count"))
    check_failed_prefix_recovery(ver, label, fit, rows)
    check_mlp_checkpoints(ver, label, fit, protocol, rows)
    return True

def check_mlp_checkpoints(ver: Verifier, label: str, fit: dict[str, Any], protocol: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    latest_value = fit.get("latest_checkpoint")
    best_value = fit.get("checkpoint")
    for kind, value, expected_sha in (("latest", latest_value, fit.get("latest_checkpoint_sha256")), ("best", best_value, fit.get("checkpoint_sha256"))):
        ver.check(f"MLP {kind} checkpoint path:" + label, isinstance(value, str), path=value)
        if not isinstance(value, str):
            continue
        path = resolve(value)
        ver.check(f"MLP {kind} checkpoint exists:" + label, path is not None and path.exists(), path=value)
        if path is None or not path.exists():
            continue
        if isinstance(expected_sha, str):
            ver.check(f"MLP {kind} checkpoint sha256:" + label, sha256_file(path) == expected_sha, expected=expected_sha, actual=sha256_file(path))
        try:
            payload = torch_load_cpu(path)
        except Exception as exc:
            ver.check(f"MLP {kind} checkpoint load:" + label, False, error=repr(exc))
            continue
        ver.check(f"MLP {kind} checkpoint RNG receipt:" + label, "torch_rng_state_cpu" in payload and "torch_rng_state_cuda" in payload, keys=sorted(payload))
        if kind == "latest":
            expected_steps = int(protocol["total_updates_per_fit"])
        else:
            best_epoch = int(fit.get("best_epoch", -1))
            by_epoch = {int(r.get("epoch", -1)): int(r.get("optimizer_steps_cumulative", -1)) for r in rows if isinstance(r, dict)}
            expected_steps = by_epoch.get(best_epoch, int(payload.get("optimizer_step_count", -2)))
        ver.check(f"MLP {kind} checkpoint optimizer_step_count:" + label, int(payload.get("optimizer_step_count", -1)) == expected_steps, expected=expected_steps, actual=payload.get("optimizer_step_count"))
        steps = optimizer_step_values(payload.get("optimizer"))
        ver.check(f"MLP {kind} optimizer state steps:" + label, bool(steps) and min(steps) == expected_steps and max(steps) == expected_steps, expected=expected_steps, min=min(steps) if steps else None, max=max(steps) if steps else None, count=len(steps))
        if kind == "best" and isinstance(fit.get("encoder_state_sha256"), str) and isinstance(payload.get("encoder"), dict):
            ver.check("MLP best encoder state hash:" + label, state_hash_from_state_dict(payload["encoder"]) == fit["encoder_state_sha256"], expected=fit["encoder_state_sha256"], actual=state_hash_from_state_dict(payload["encoder"]))


def check_method_fit(ver: Verifier, label: str, fit: dict[str, Any], protocol: dict[str, Any], method_protocol: dict[str, Any]) -> None:
    total_updates = int(protocol["total_updates_per_fit"])
    fit_receipt = fit.get("fit_receipt") if isinstance(fit.get("fit_receipt"), dict) else {}
    steps = int(fit.get("actual_optimizer_steps", fit.get("native_iterations", fit_receipt.get("actual_optimizer_steps", fit_receipt.get("native_iterations", fit.get("optimizer_steps_completed", fit.get("actual_updates", -1)))))))
    native_iterations = int(fit.get("native_iterations", fit_receipt.get("native_iterations", steps)))
    ver.check("method native final updates:" + label, steps == total_updates and native_iterations == total_updates, expected=total_updates, actual_steps=steps, native_iterations=native_iterations)
    text = json.dumps(fit, ensure_ascii=False).lower()
    ver.check("method final SWA selected:" + label, "final swa" in text or ("swa" in text and "selected" in text), fit_keys=sorted(fit))
    swa_count = fit.get("swa_n_averaged", fit_receipt.get("swa_n_averaged"))
    if swa_count is not None:
        ver.check("method SWA update count:" + label, int(swa_count) == total_updates + 1, expected=total_updates + 1, actual=swa_count)
    val_selection_keys = {"selected_epoch", "best_epoch", "curve", "curve_path", "val_reconstruction", "validation_curve"}
    ver.check("method no VAL checkpoint selection:" + label, not any(key in fit for key in val_selection_keys), forbidden=sorted(val_selection_keys & set(fit)))
    windows = fit.get("training_windows_shape") or fit_receipt.get("training_windows_shape")
    shape_ok = isinstance(windows, list) and len(windows) == 3 and int(windows[1]) == int(protocol["context"])
    ver.check("method full-context TRAIN window shape:" + label, shape_ok, shape=windows)
    expected_k = int((method_protocol.get("k") or {}).get(str(fit.get("dataset", "")), -1))
    if expected_k < 0 and isinstance(fit.get("dataset"), str):
        expected_k = int((method_protocol.get("k") or {}).get(fit["dataset"], -1))
    if expected_k >= 0 and shape_ok:
        ver.check("method input channels not latent K in training windows:" + label, int(windows[2]) != expected_k, shape=windows, method_k=expected_k)
    provenance = fit.get("training_provenance") or fit.get("training_window_receipt") or fit.get("window_pool") or fit_receipt
    ptext = json.dumps(provenance, ensure_ascii=False).lower()
    ver.check("method TRAIN-only full-window provenance:" + label, "train" in ptext and ("context" in ptext or "512" in ptext) and "window" in ptext and ("eligible" in ptext or "bounds" in ptext), provenance_keys=sorted(provenance) if isinstance(provenance, dict) else None)
    ver.check("method origin exclusion/provenance receipt:" + label, all(word in ptext for word in ("canonical", "eligible", "excluded")), provenance_keys=sorted(provenance) if isinstance(provenance, dict) else None)
    ver.check("method training window hash receipt:" + label, isinstance((provenance or {}).get("window_sha256") if isinstance(provenance, dict) else None, str) or "window_sha256" in text, fit_keys=sorted(fit))
    ver.check("method objective/data order receipt:" + label, ("contrastive" in text or "ts2vec" in text) and ("order" in text or "shuffle" in text or "dataloader" in text), fit_keys=sorted(fit))
    check_ts2vec_checkpoint_and_sampling(ver, label, fit, protocol, method_protocol)
    if "pure temporal" in text or "pure_temporal" in text:
        ver.warn("method pure-temporal claim must be limited:" + label)


def check_ts2vec_checkpoint_and_sampling(ver: Verifier, label: str, fit: dict[str, Any], protocol: dict[str, Any], method_protocol: dict[str, Any]) -> None:
    checkpoint_value = fit.get("checkpoint")
    ver.check("method checkpoint path:" + label, isinstance(checkpoint_value, str), path=checkpoint_value)
    checkpoint_payload = None
    if isinstance(checkpoint_value, str):
        path = resolve(checkpoint_value)
        ver.check("method checkpoint exists:" + label, path is not None and path.exists(), path=checkpoint_value)
        if path is not None and path.exists():
            if isinstance(fit.get("checkpoint_sha256"), str):
                ver.check("method checkpoint sha256:" + label, sha256_file(path) == fit["checkpoint_sha256"], expected=fit["checkpoint_sha256"], actual=sha256_file(path))
            try:
                checkpoint_payload = torch_load_cpu(path)
            except Exception as exc:
                ver.check("method checkpoint load:" + label, False, error=repr(exc))
    if isinstance(checkpoint_payload, dict):
        ver.check("method checkpoint schema:" + label, checkpoint_payload.get("schema") == "native_small_ts2vec_checkpoint_v1", schema=checkpoint_payload.get("schema"))
        ver.check("method checkpoint native iterations:" + label, int(checkpoint_payload.get("native_iterations", -1)) == int(protocol["total_updates_per_fit"]), actual=checkpoint_payload.get("native_iterations"))
        opt = checkpoint_payload.get("optimizer")
        steps = optimizer_step_values(opt)
        ver.check("method checkpoint optimizer state steps:" + label, bool(steps) and min(steps) == int(protocol["total_updates_per_fit"]) and max(steps) == int(protocol["total_updates_per_fit"]), min=min(steps) if steps else None, max=max(steps) if steps else None, count=len(steps))
        groups = (opt or {}).get("param_groups") or []
        weight_decay = groups[0].get("weight_decay") if groups and isinstance(groups[0], dict) else None
        ver.check("method checkpoint AdamW weight_decay:" + label, weight_decay == 0.01, actual=weight_decay)
        receipt = checkpoint_payload.get("fit_receipt") or {}
        ver.check("method checkpoint SWA count:" + label, int(receipt.get("swa_n_averaged", -1)) == int(protocol["total_updates_per_fit"]) + 1, actual=receipt.get("swa_n_averaged"))
    provenance = fit.get("training_provenance") or {}
    origin_npz = resolve(provenance.get("origin_npz")) if isinstance(provenance, dict) else None
    if isinstance(provenance, dict):
        ver.check("method pool origin npz exists:" + label, origin_npz is not None and origin_npz.exists(), path=provenance.get("origin_npz"))
        if origin_npz is not None and origin_npz.exists():
            if isinstance(provenance.get("origin_npz_sha256"), str):
                ver.check("method pool origin npz sha256:" + label, sha256_file(origin_npz) == provenance["origin_npz_sha256"], expected=provenance["origin_npz_sha256"], actual=sha256_file(origin_npz))
            with np.load(origin_npz, allow_pickle=False) as z:
                canonical = z["canonical_origins"].astype(np.int64)
                eligible = z["eligible_origins"].astype(np.int64)
                excluded = z["excluded_origins"].astype(np.int64)
                train_bounds = z["train_bounds"].astype(np.int64)
            ver.check("method pool origin counts/hashes:" + label, int(provenance.get("canonical_origin_count", -1)) == len(canonical) and int(provenance.get("eligible_origin_count", -1)) == len(eligible) and int(provenance.get("excluded_origin_count", -1)) == len(excluded) and provenance.get("canonical_origin_sha256") == sha256_array(canonical) and provenance.get("eligible_origin_sha256") == sha256_array(eligible) and provenance.get("excluded_origin_sha256") == sha256_array(excluded), counts={"canonical": len(canonical), "eligible": len(eligible), "excluded": len(excluded)})
            context = int(protocol["context"])
            ver.check("method pool TRAIN bounds:" + label, np.array_equal(train_bounds, np.asarray(provenance.get("train_bounds"), dtype=np.int64)) and (len(eligible) == 0 or (eligible.min() - context >= train_bounds[0] and eligible.max() <= train_bounds[1])), train_bounds=train_bounds.tolist(), history_bounds=provenance.get("history_timestamp_bounds_inclusive"))
            dataset = str(fit.get("dataset"))
            try:
                parent = read_json(PARENT_RESULT)
                archive, _, _, _ = load_parent_archive(parent, dataset)
                window_hash = hash_windows_incremental(archive["x"], eligible, context)
                ver.check("method pool window hash incremental:" + label, window_hash == provenance.get("window_sha256"), expected=provenance.get("window_sha256"), actual=window_hash)
            except Exception as exc:
                ver.check("method pool window hash incremental:" + label, False, error=repr(exc))
    sampling = fit.get("sampling_receipt") or {}
    sample_npz = resolve(sampling.get("npz")) if isinstance(sampling, dict) else None
    ver.check("method shuffle npz exists:" + label, sample_npz is not None and sample_npz.exists(), path=sampling.get("npz") if isinstance(sampling, dict) else None)
    if sample_npz is not None and sample_npz.exists() and isinstance(sampling, dict):
        if isinstance(sampling.get("npz_sha256"), str):
            ver.check("method shuffle npz sha256:" + label, sha256_file(sample_npz) == sampling["npz_sha256"], expected=sampling["npz_sha256"], actual=sha256_file(sample_npz))
        with np.load(sample_npz, allow_pickle=False) as z:
            opt_idx = z["optimized_pool_indices"].astype(np.int64)
            opt_orig = z["optimized_origins"].astype(np.int64)
            fetched = z["fetched_pool_indices"].astype(np.int64)
            prefetch = z["prefetched_after_budget_indices"].astype(np.int64)
        ver.check("method shuffle optimized shape/count:" + label, opt_idx.shape == (int(protocol["total_updates_per_fit"]), int((method_protocol.get("native_configuration") or {}).get("batch_size", -1))) and int(sampling.get("optimized_updates", -1)) == int(protocol["total_updates_per_fit"]) and int(sampling.get("optimized_windows", -1)) == opt_idx.size, shape=list(opt_idx.shape), optimized_windows=sampling.get("optimized_windows"))
        ver.check("method shuffle hash receipts:" + label, sampling.get("optimized_origin_stream_sha256") == sha256_array(opt_orig) and sampling.get("optimized_pool_index_stream_sha256") == sha256_array(opt_idx), origin_sha=sha256_array(opt_orig), index_sha=sha256_array(opt_idx))
        ver.check("method shuffle prefetch separation:" + label, len(fetched) == opt_idx.size + len(prefetch) and int(sampling.get("fetched_after_budget_windows", -1)) == len(prefetch), fetched=len(fetched), optimized=opt_idx.size, prefetch=len(prefetch))


def check_pca(ver: Verifier, label: str, adapter: dict[str, Any], basis_receipts: dict[str, Any], archive: np.lib.npyio.NpzFile, k: int, middle_k: int) -> None:
    receipt = basis_receipts.get(f"K{k}") if isinstance(basis_receipts, dict) else None
    ver.check("PCA basis receipt exists:" + label, isinstance(receipt, dict), available=sorted(basis_receipts) if isinstance(basis_receipts, dict) else None)
    basis = None
    if isinstance(receipt, dict):
        path = resolve(receipt.get("path"))
        ver.check("PCA basis file exists:" + label, path is not None and path.exists(), path=receipt.get("path"))
        if path is not None and path.exists():
            if isinstance(receipt.get("sha256"), str):
                ver.check("PCA basis file sha256:" + label, sha256_file(path) == receipt["sha256"], expected=receipt["sha256"], actual=sha256_file(path))
            basis = np.load(path, allow_pickle=False)
            if isinstance(receipt.get("array_sha256"), str):
                ver.check("PCA basis array sha256:" + label, sha256_array(basis.astype(np.float32)) == receipt["array_sha256"], expected=receipt["array_sha256"], actual=sha256_array(basis.astype(np.float32)))
        ver.check("PCA basis kind:" + label, receipt.get("kind") in {"new_train_centered_pca", "parent_middle_pca"}, kind=receipt.get("kind"))
    if basis is not None:
        ver.check("PCA basis shape:" + label, basis.shape == (archive["x"].shape[1], k), shape=list(basis.shape))
        if k != middle_k:
            ok, diff = projection_close(basis, pca_basis_from_train_archive(archive, k))
            ver.check("PCA TRAIN-only covariance subspace:" + label, ok, projection_max_abs=diff)
        else:
            ver.check("PCA middle-K parent reuse kind:" + label, isinstance(receipt, dict) and receipt.get("kind") == "parent_middle_pca", kind=receipt.get("kind") if isinstance(receipt, dict) else None)
    ver.check("PCA adapter encoder hash receipt:" + label, isinstance(adapter.get("encoder_state_sha256_before"), str) and adapter.get("encoder_state_sha256_before") == adapter.get("encoder_state_sha256_after"), before=adapter.get("encoder_state_sha256_before"), after=adapter.get("encoder_state_sha256_after"))

def check_contract(ver: Verifier, label: str, adapter: dict[str, Any], contract: dict[str, Any], archive: np.lib.npyio.NpzFile, protocol: dict[str, Any], k: int, val_origins: np.ndarray, target: np.ndarray, mask: np.ndarray, method: bool = False) -> int:
    choices = contract.get("choices") or contract.get("ridge_choices") or []
    lambdas = [float(c.get("lambda")) for c in choices if isinstance(c, dict) and "lambda" in c]
    final_grid = [float(x) for x in protocol["ridge_lambdas"]]
    initial_grid = [float(x) for x in protocol["ridge_initial_lambdas"]]
    ver.check("ridge final grid choices:" + label, lambdas == final_grid, expected=final_grid, actual=lambdas)
    initial = contract.get("initial_choices") or contract.get("diagnostic_initial_choices") or contract.get("initial_grid_choices")
    if initial is not None:
        init_lambdas = [float(c.get("lambda")) for c in initial if isinstance(c, dict) and "lambda" in c]
        ver.check("ridge initial diagnostic preserved:" + label, init_lambdas == initial_grid, actual=init_lambdas)
    else:
        ver.check("ridge initial diagnostic recoverable:" + label, [x for x in lambdas if x in initial_grid] == initial_grid)
    initial_selection = contract.get("initial_grid_selection")
    if isinstance(initial_selection, dict) and choices:
        eligible = [c for c in choices if float(c.get("lambda")) in initial_grid]
        expected_initial = min(eligible, key=lambda row: (float(((row.get("metrics") or {}).get("all") or {}).get("mse", math.inf)), lambdas.index(float(row.get("lambda")))))
        ver.check("ridge initial selection earliest min:" + label, float(initial_selection.get("selected_lambda")) == float(expected_initial.get("lambda")), expected=expected_initial.get("lambda"), actual=initial_selection.get("selected_lambda"))
    if choices:
        selected = min(choices, key=lambda row: (float(((row.get("metrics") or {}).get("all") or {}).get("mse", math.inf)), lambdas.index(float(row.get("lambda")))))
        ver.check("ridge selected lambda earliest min:" + label, float(selected.get("lambda")) == float(contract.get("selected_lambda")), expected=selected.get("lambda"), actual=contract.get("selected_lambda"))
        if float(contract.get("selected_lambda")) in (final_grid[0], final_grid[-1]):
            ver.warn("ridge selected final-grid boundary:" + label, selected_lambda=contract.get("selected_lambda"))
    npz_value = contract.get("prediction_npz") or contract.get("prediction_path")
    ver.check("prediction npz path:" + label, isinstance(npz_value, str), path=npz_value)
    if not isinstance(npz_value, str):
        return len(choices)
    path = resolve(npz_value)
    ver.check("prediction npz exists:" + label, path is not None and path.exists(), path=npz_value)
    if path is None or not path.exists():
        return len(choices)
    if isinstance(contract.get("prediction_sha256"), str):
        ver.check("prediction npz hash:" + label, sha256_file(path) == contract["prediction_sha256"], expected=contract["prediction_sha256"], actual=sha256_file(path))
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    required = ["origins", "train_origins", "val_latent_history_tail", "latent_forecast", "latent_last", "prediction_tsfm", "decoder_coeff", "feature_scale"]
    for key in required:
        ver.check("prediction key:" + label + ":" + key, key in arrays, keys=sorted(arrays))
    if not all(key in arrays for key in required):
        return len(choices)
    lag = int(protocol["decoder_max_lag"])
    ver.check("VAL origins:" + label, np.array_equal(arrays["origins"], val_origins))
    dec_origins = deterministic_decoder_origins(archive, protocol)
    ver.check("TRAIN origins:" + label, np.array_equal(arrays["train_origins"], dec_origins), expected_sha256=sha256_array(dec_origins), actual_sha256=sha256_array(arrays["train_origins"]))
    ver.check("latent forecast shape:" + label, arrays["latent_forecast"].shape == (len(val_origins), int(protocol["horizon"]), k), shape=list(arrays["latent_forecast"].shape))
    ver.check("prediction shape:" + label, arrays["prediction_tsfm"].shape == target.shape, shape=list(arrays["prediction_tsfm"].shape), target=list(target.shape))
    computed = grouped_metrics(arrays["prediction_tsfm"], target, mask)
    reported = contract.get("future_tsfm") or contract.get("metrics") or contract.get("prediction_metrics") or {}
    for group, vals in computed.items():
        for metric_name, actual in vals.items():
            expected = ((reported.get(group) or {}).get(metric_name)) if isinstance(reported, dict) else None
            ver.check("metric recompute:" + label + f":{group}:{metric_name}", expected is not None and abs(float(expected) - actual) < 1e-10, expected=expected, actual=actual)
    features = future_features_np(arrays["val_latent_history_tail"], arrays["latent_forecast"], lag)
    if "grid_coefficients" in arrays:
        if isinstance(contract.get("grid_coefficients_sha256"), str):
            ver.check("grid coefficients sha256:" + label, sha256_array(arrays["grid_coefficients"].astype(np.float64)) == contract["grid_coefficients_sha256"], expected=contract["grid_coefficients_sha256"], actual=sha256_array(arrays["grid_coefficients"].astype(np.float64)))
        ver.check("grid coefficients shape:" + label, arrays["grid_coefficients"].shape[0] == len(choices) and arrays["grid_coefficients"].shape[1:] == arrays["decoder_coeff"].shape, shape=list(arrays["grid_coefficients"].shape))
        for idx, choice in enumerate(choices):
            pred_choice = apply_coeff_np(features, arrays["grid_coefficients"][idx])
            metrics_choice = grouped_metrics(pred_choice, target, mask)
            expected_mse = (((choice.get("metrics") or {}).get("all") or {}).get("mse"))
            ver.check("grid coeff VAL metric:" + label + f":{idx}", expected_mse is not None and abs(float(expected_mse) - metrics_choice["all"]["mse"]) < 1e-10, expected=expected_mse, actual=metrics_choice["all"]["mse"])
    if "prediction_latent_last" in arrays:
        computed_last = grouped_metrics(arrays["prediction_latent_last"], target, mask)
        reported_last = contract.get("future_latent_last") or {}
        for group, vals in computed_last.items():
            for metric_name, actual in vals.items():
                expected = ((reported_last.get(group) or {}).get(metric_name)) if isinstance(reported_last, dict) else None
                ver.check("latent_last metric recompute:" + label + f":{group}:{metric_name}", expected is not None and abs(float(expected) - actual) < 1e-10, expected=expected, actual=actual)
        last_features = future_features_np(arrays["val_latent_history_tail"], arrays["latent_last"], lag)
        rebuilt_last = apply_coeff_np(last_features, arrays["decoder_coeff"])
        ver.check("latent_last prediction from saved traces:" + label, np.allclose(rebuilt_last, arrays["prediction_latent_last"], atol=2e-5, rtol=2e-5), max_abs=float(np.max(np.abs(rebuilt_last - arrays["prediction_latent_last"]))))
    rebuilt = apply_coeff_np(features, arrays["decoder_coeff"])
    ver.check("causal D8 prediction from saved traces:" + label, np.allclose(rebuilt, arrays["prediction_tsfm"], atol=2e-5, rtol=2e-5), max_abs=float(np.max(np.abs(rebuilt - arrays["prediction_tsfm"]))))
    if "decoder_taps" in arrays:
        taps = coeff_to_taps_np(arrays["decoder_coeff"], lag, k)
        ver.check("decoder taps from coeff:" + label, np.allclose(taps, arrays["decoder_taps"], atol=1e-12, rtol=1e-12))
    if {"train_latent_history", "train_latent_forecast"}.issubset(arrays):
        train_target, train_mask = future_targets_from_archive(archive, arrays["train_origins"])
        train_features = future_features_np(arrays["train_latent_history"], arrays["train_latent_forecast"], lag)
        train_row_mask = train_mask.any(axis=2)
        coeff, scale, scale_count, row_mask_sha = ridge_coeff_from_trace(train_features, train_target, train_mask, train_row_mask, float(contract.get("selected_lambda")))
        ver.check("TRAIN-only RMS scale recompute:" + label, np.allclose(scale, arrays["feature_scale"], atol=1e-10, rtol=1e-10) and int(contract.get("scale_count", scale_count)) == scale_count and contract.get("scale_row_mask_sha256", row_mask_sha) == row_mask_sha, max_scale_abs=float(np.max(np.abs(scale - arrays["feature_scale"]))))
        ver.check("selected coeff recompute:" + label, np.allclose(coeff, arrays["decoder_coeff"], atol=1e-6, rtol=1e-6), max_abs=float(np.max(np.abs(coeff - arrays["decoder_coeff"]))))
    else:
        ver.check("TRAIN latent traces saved:" + label, False, keys=sorted(arrays))
    if method:
        ver.check("method adapter family receipt:" + label, infer_family(adapter, str(adapter.get("id", ""))) == "PROFESSOR_METHOD", family=infer_family(adapter, str(adapter.get("id", ""))), adapter_keys=sorted(adapter))
    return len(choices)


def check_trace_reuse(ver: Verifier, label: str, contract: dict[str, Any], expected_parent: str) -> None:
    trace = contract.get("trace_reuse") or {}
    ver.check("middle trace reuse receipt:" + label, isinstance(trace, dict) and trace.get("parent_adapter") == expected_parent and int(trace.get("required_lag", -1)) == 8, trace=trace, expected_parent=expected_parent)
    path = resolve(trace.get("path")) if isinstance(trace, dict) else None
    if path is not None:
        ver.check("middle trace reuse path hash:" + label, path.exists() and sha256_file(path) == trace.get("sha256"), expected=trace.get("sha256"), actual=sha256_file(path) if path.exists() else None)


def load_parent_archive(parent: dict[str, Any], dataset: str) -> tuple[np.lib.npyio.NpzFile, np.ndarray, np.ndarray, np.ndarray]:
    src = parent["datasets"][dataset]["source"]
    path = resolve(src["path"])
    if path is None or not path.exists():
        raise FileNotFoundError(src["path"])
    if src.get("sha256") and sha256_file(path) != src["sha256"]:
        raise RuntimeError(f"parent source hash mismatch: {dataset}")
    archive = np.load(path, allow_pickle=False)
    origins = np.asarray(archive["val_origins"])
    target, mask = future_targets_from_archive(archive, origins)
    return archive, origins, target, mask


def check_raw(ver: Verifier, label: str, entry: dict[str, Any], val_origins: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
    path_value = entry.get("prediction_npz") or entry.get("prediction_path")
    ver.check("RAW prediction path:" + label, isinstance(path_value, str), path=path_value)
    if not isinstance(path_value, str):
        return
    path = resolve(path_value)
    ver.check("RAW prediction exists:" + label, path is not None and path.exists(), path=path_value)
    if path is None or not path.exists():
        return
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    ver.check("RAW origins:" + label, "origins" in arrays and np.array_equal(arrays["origins"], val_origins), keys=sorted(arrays))
    if "prediction_tsfm" in arrays:
        computed = grouped_metrics(arrays["prediction_tsfm"], target, mask)
        reported = entry.get("future_tsfm") or entry.get("metrics") or {}
        for group, vals in computed.items():
            for metric_name, actual in vals.items():
                expected = ((reported.get(group) or {}).get(metric_name)) if isinstance(reported, dict) else None
                ver.check("RAW metric recompute:" + label + f":{group}:{metric_name}", expected is not None and abs(float(expected) - actual) < 1e-10, expected=expected, actual=actual)


def assert_protocol(ver: Verifier, protocol: dict[str, Any]) -> None:
    expected = {
        "schema": "k_tradeoff_and_professor_method_v1",
        "datasets": ["robin", "jena"],
        "seeds": [92601, 92602],
        "context": 512,
        "horizon": 48,
        "max_epochs": 180,
        "updates_per_epoch": 16,
        "total_updates_per_fit": 2880,
        "early_stopping": False,
        "decoder_contract": "FORECAST_RIDGE_D8",
        "decoder_train_origin_count": 512,
        "decoder_sampling_seed": 92603,
        "new_test_calls": 0,
        "frozen_backbone_backward": False,
    }
    for key, value in expected.items():
        ver.check("protocol:" + key, protocol.get(key) == value, expected=value, actual=protocol.get(key))
    ver.check("protocol:final ridge grid", protocol.get("ridge_lambdas") == [0.000001, 0.0001, 0.01, 1, 10, 100, 1000, 10000, 100000, 1000000], actual=protocol.get("ridge_lambdas"))
    ver.check("protocol:initial ridge grid", protocol.get("ridge_initial_lambdas") == [0.0001, 0.01, 1, 10, 100, 1000, 10000], actual=protocol.get("ridge_initial_lambdas"))
    ver.check("protocol:controls include RAW_C_D8", "RAW_C_D8" in protocol.get("controls", []), controls=protocol.get("controls"))
    ver.check("protocol:limits ridge_fits", int((protocol.get("limits") or {}).get("ridge_fits", -1)) == 200, actual=(protocol.get("limits") or {}).get("ridge_fits"))


def check_sources(ver: Verifier, result: dict[str, Any] | None) -> None:
    for filename in ("source_seal.json", "source_seal.k.json", "source_seal.continuation.json", "source_seal.method.json"):
        path = HERE / filename
        if not path.exists():
            if result is not None and filename in {"source_seal.k.json", "source_seal.method.json"}:
                ver.check("required file exists:" + filename[:-5], False, path=str(path.relative_to(ROOT)))
            continue
        seal = read_json(path)
        for row in seal.get("artifacts", []):
            if not isinstance(row, dict):
                continue
            artifact_path = resolve(row.get("path"))
            ver.check("source hash current:" + str(row.get("path")), artifact_path is not None and artifact_path.exists() and sha256_file(artifact_path) == row.get("sha256"), expected=row.get("sha256"), actual=sha256_file(artifact_path) if artifact_path and artifact_path.exists() else None)
    for filename in ("parent_preservation.before.json", "parent_preservation.after.json"):
        path = HERE / filename
        obj = load_required(ver, path, filename[:-5])
        if isinstance(obj, dict) and "after" in filename:
            bad = [row.get("path") for row in obj.get("checks", []) if isinstance(row, dict) and row.get("actual_sha256") != row.get("sha256")]
            ver.check("parent preservation after hashes", not bad and obj.get("status") == "PASS", bad=bad[:10], status=obj.get("status"))
    if isinstance(result, dict):
        source_paths = {
            "PLAN": HERE / "PLAN.md",
            "protocol": HERE / "protocol.json",
            "sweep": HERE / "sweep.py",
            "benchmark": HERE / "benchmark.py",
            "test_sweep": HERE / "test_sweep.py",
            "seal": HERE / "seal.py",
            "method_protocol": HERE / "method_protocol.json",
            "method_run": HERE / "method_run.py",
            "continuation": HERE / "continuation.py",
            "ts2vec_adapter": HERE / "ts2vec_adapter.py",
            "test_ts2vec": HERE / "test_ts2vec.py",
        }
        for name, row in (result.get("source_hashes") or {}).items():
            if isinstance(row, dict):
                path = resolve(row.get("path"))
                expected = row.get("sha256")
            else:
                path = source_paths.get(str(name)) or resolve(str(name))
                expected = row if isinstance(row, str) else None
            ver.check("result source hash:" + str(name), path is not None and path.exists() and expected is not None and sha256_file(path) == expected, expected=expected, actual=sha256_file(path) if path and path.exists() else None)


def check_one_failed_attempt(ver: Verifier, attempt: int) -> dict[str, Any] | None:
    preservation_path = HERE / f"failed_attempt{attempt}_preservation.json"
    ledger_path = HERE / "failed_attempts" / f"attempt{attempt}" / "failure_ledger.json"
    progress_path = HERE / "failed_attempts" / f"attempt{attempt}" / "progress.json"
    if not preservation_path.exists():
        ver.warn(f"failed attempt{attempt} preservation absent")
        return None
    preservation = read_json(preservation_path)
    ver.check(f"failed attempt{attempt} preservation status", preservation.get("status") == "PASS", status=preservation.get("status"))
    artifacts = preservation.get("archived_artifacts") or []
    ver.check(f"failed attempt{attempt} archived artifacts listed", isinstance(artifacts, list) and bool(artifacts), count=len(artifacts) if isinstance(artifacts, list) else None)
    for row in artifacts:
        if not isinstance(row, dict):
            continue
        path = resolve(row.get("path"))
        ver.check(f"failed attempt{attempt} archive hash:" + str(row.get("path")), path is not None and path.exists() and sha256_file(path) == row.get("sha256"), expected=row.get("sha256"), actual=sha256_file(path) if path and path.exists() else None)
    ledger = load_required(ver, ledger_path, f"failed_attempt{attempt}_ledger")
    progress = load_required(ver, progress_path, f"failed_attempt{attempt}_progress")
    if isinstance(ledger, dict):
        ver.check(f"failed attempt{attempt} ledger status", ledger.get("status") == "failed_preserved" and int(ledger.get("attempt", -1)) == attempt, status=ledger.get("status"), attempt=ledger.get("attempt"))
        ver.check(f"failed attempt{attempt} scientific recipe unchanged", ledger.get("scientific_recipe_unchanged") is True, value=ledger.get("scientific_recipe_unchanged"))
        if attempt == 1:
            ver.check("failed attempt1 counts", int(ledger.get("fit_starts", -1)) == 1 and int(ledger.get("completed_fits", -1)) == 0 and int(ledger.get("ridge_fits", -1)) == 0, fit_starts=ledger.get("fit_starts"), completed_fits=ledger.get("completed_fits"), ridge_fits=ledger.get("ridge_fits"))
            ver.check("failed attempt1 WinError5 recorded", "WinError5" in str(ledger.get("reason", "")) or "WinError 5" in json.dumps(ledger, ensure_ascii=False), reason=ledger.get("reason"))
            ver.check("failed attempt1 recorded steps retained", ledger.get("recorded_optimizer_steps") is not None and int(ledger.get("recorded_optimizer_steps")) > 0 and ledger.get("exact_total_optimizer_steps") is None, recorded=ledger.get("recorded_optimizer_steps"), exact=ledger.get("exact_total_optimizer_steps"))
        elif attempt == 2:
            ver.check("failed attempt2 counts", int(ledger.get("fit_starts", -1)) == 6 and int(ledger.get("completed_fits", -1)) == 5 and int(ledger.get("ridge_fits", -1)) == 100, fit_starts=ledger.get("fit_starts"), completed_fits=ledger.get("completed_fits"), ridge_fits=ledger.get("ridge_fits"))
            ver.check("failed attempt2 preservation reuse", int(ledger.get("preserved_completed_fits_reused", -1)) == 5 and int(ledger.get("preserved_completed_ridge_fits_reused", -1)) == 100 and ledger.get("unfinished_trajectory_used_for_selection") is False, reused_fits=ledger.get("preserved_completed_fits_reused"), reused_ridge=ledger.get("preserved_completed_ridge_fits_reused"), used=ledger.get("unfinished_trajectory_used_for_selection"))
            ver.check("failed attempt2 WinError3 recorded", "WinError3" in str(ledger.get("reason", "")) or "WinError 3" in json.dumps(ledger, ensure_ascii=False), reason=ledger.get("reason"))
            ver.check("failed attempt2 unfinished step bounds", int(ledger.get("unfinished_recorded_optimizer_steps", -1)) == 352 and int(ledger.get("unfinished_possible_additional_steps_max", -1)) == 16, recorded=ledger.get("unfinished_recorded_optimizer_steps"), possible=ledger.get("unfinished_possible_additional_steps_max"))
        ver.check(f"failed attempt{attempt} continuation note", isinstance(ledger.get("continuation"), str) and len(ledger.get("continuation", "")) > 10, continuation=ledger.get("continuation"))
    if isinstance(progress, dict) and isinstance(ledger, dict):
        fp = ledger.get("failure_progress") or {}
        ver.check(f"failed attempt{attempt} progress parity", progress.get("pid") == fp.get("pid") and progress.get("stage") == fp.get("stage") and progress.get("error") == fp.get("error"), progress_stage=progress.get("stage"), ledger_stage=fp.get("stage"))
    for seal_name in ("source_seal.original.json", "source_seal.k.json", "source_seal.method.json"):
        old_seal = HERE / "failed_attempts" / f"attempt{attempt}" / seal_name
        if not old_seal.exists():
            continue
        seal = read_json(old_seal)
        for item in seal.get("artifacts", []):
            if not isinstance(item, dict):
                continue
            copied = HERE / "failed_attempts" / f"attempt{attempt}" / "sources" / Path(str(item.get("path"))).name
            if copied.exists():
                ver.check(f"failed attempt{attempt} old source seal copy:" + str(item.get("path")), sha256_file(copied) == item.get("sha256"), expected=item.get("sha256"), actual=sha256_file(copied))
    continuation_path = HERE / "failed_attempts" / f"attempt{attempt}" / "continuation_receipt.json"
    if continuation_path.exists():
        receipt = read_json(continuation_path)
        ver.check(f"failed attempt{attempt} continuation receipt", receipt.get("status") in {"authorized_routine_recovery", "PASS"} and receipt.get("previous_partial_trajectory_used_for_selection") is False, receipt=receipt)
    return ledger if isinstance(ledger, dict) else None


def check_failed_attempt_preservation(ver: Verifier) -> dict[str, dict[str, Any]]:
    ledgers: dict[str, dict[str, Any]] = {}
    for attempt in (1, 2):
        ledger = check_one_failed_attempt(ver, attempt)
        if isinstance(ledger, dict):
            ledgers[str(attempt)] = ledger
    return ledgers


def check_continuation_receipt(ver: Verifier) -> None:
    path = HERE / "continuation_receipt.json"
    receipt = load_required(ver, path, "continuation_receipt")
    if not isinstance(receipt, dict):
        return
    ver.check(
        "continuation receipt status",
        receipt.get("status") == "authorized_routine_recovery"
        and receipt.get("stage") == "K continuation after preserved attempt2"
        and receipt.get("scientific_recipe_unchanged") is True,
        status=receipt.get("status"),
        stage=receipt.get("stage"),
        scientific_recipe_unchanged=receipt.get("scientific_recipe_unchanged"),
    )
    expected_attempts = ["failed_attempts/attempt1/failure_ledger.json", "failed_attempts/attempt2/failure_ledger.json"]
    expected_manifests = ["failed_attempt1_preservation.json", "failed_attempt2_preservation.json"]
    expected_seals = ["source_seal.k.json", "source_seal.continuation.json", "source_seal.method.json"]
    ver.check("continuation receipt previous attempts", receipt.get("previous_attempts") == expected_attempts, actual=receipt.get("previous_attempts"))
    ver.check("continuation receipt preservation manifests", receipt.get("preservation_manifests") == expected_manifests, actual=receipt.get("preservation_manifests"))
    ver.check("continuation receipt new source seals", receipt.get("new_source_seals") == expected_seals, actual=receipt.get("new_source_seals"))
    for listed in expected_attempts + expected_manifests + expected_seals:
        listed_path = resolve(listed)
        if listed_path is None or not listed_path.exists():
            listed_path = HERE / listed
        ver.check("continuation receipt listed file exists:" + listed, listed_path is not None and listed_path.exists(), path=str(listed_path) if listed_path is not None else None)
    remaining = ["jena_MLP_POINT_K3_seed92602", "jena_MLP_POINT_K12_seed92601", "jena_MLP_POINT_K12_seed92602"]
    ver.check(
        "continuation receipt reuse plan",
        int(receipt.get("reuse_completed_encoder_fits", -1)) == 5
        and int(receipt.get("reuse_completed_ridge_fits", -1)) == 100
        and receipt.get("remaining_fit_ids") == remaining
        and receipt.get("failed_partial_trajectories_used_for_selection") is False,
        reuse_completed_encoder_fits=receipt.get("reuse_completed_encoder_fits"),
        reuse_completed_ridge_fits=receipt.get("reuse_completed_ridge_fits"),
        remaining_fit_ids=receipt.get("remaining_fit_ids"),
        failed_partial_trajectories_used_for_selection=receipt.get("failed_partial_trajectories_used_for_selection"),
    )
    limits = receipt.get("new_stage_limits") or {}
    baseline = receipt.get("expected_baseline_counts_if_successful") or {}
    task = receipt.get("expected_task_counts_including_native_if_successful") or {}
    failure = receipt.get("failure_optimizer_steps") or {}
    consumed = receipt.get("expected_all_consumed_optimizer_steps_if_successful") or {}
    ver.check("continuation receipt stage limits", int(limits.get("neural_fit_starts", -1)) == 3 and int(limits.get("ridge_fits", -1)) == 100, limits=limits)
    ver.check("continuation receipt baseline totals", int(baseline.get("physical_fit_starts", -1)) == 10 and int(baseline.get("completed_fits", -1)) == 8 and int(baseline.get("ridge_fits", -1)) == 200, baseline=baseline)
    ver.check("continuation receipt task totals", int(task.get("physical_fit_starts", -1)) == 14 and int(task.get("completed_fits", -1)) == 12 and int(task.get("ridge_fits", -1)) == 240 and int(task.get("successful_optimizer_steps", -1)) == 34560, task=task)
    ver.check("continuation receipt failure bounds", int(failure.get("recorded", -1)) == 1344 and int(failure.get("possible_additional_lower", -1)) == 0 and int(failure.get("possible_additional_upper", -1)) == 31, failure=failure)
    ver.check("continuation receipt consumed bounds", int(consumed.get("lower", -1)) == 35904 and int(consumed.get("upper", -1)) == 35935, consumed=consumed)


def check_continuation_seal(ver: Verifier, result: dict[str, Any] | None) -> None:
    path = HERE / "source_seal.continuation.json"
    if result is None or result.get("status") not in {"complete", "PASS", "passed"}:
        if not path.exists():
            ver.warn("continuation source seal absent before final result")
        return
    obj = load_required(ver, path, "source_seal.continuation")
    if not isinstance(obj, dict):
        return
    ver.check("continuation source seal stage", obj.get("phase") == "before_stage_science" and obj.get("stage") == "continuation", phase=obj.get("phase"), stage=obj.get("stage"))
    artifacts = obj.get("artifacts") or []
    paths = {str(row.get("path")) for row in artifacts if isinstance(row, dict)}
    required_suffixes = {"PLAN.md", "protocol.json", "sweep.py", "benchmark.py", "test_sweep.py", "seal.py", "continuation.py"}
    ver.check("continuation source seal artifacts", all(any(path.endswith(suffix) for path in paths) for suffix in required_suffixes) and len(artifacts) >= 7, required=sorted(required_suffixes), paths=sorted(paths))
    for row in artifacts:
        if not isinstance(row, dict):
            continue
        artifact_path = resolve(row.get("path"))
        ok = artifact_path is not None and artifact_path.exists() and sha256_file(artifact_path) == row.get("sha256")
        if "bytes" in row and artifact_path is not None and artifact_path.exists():
            ok = ok and int(row["bytes"]) == artifact_path.stat().st_size
        ver.check("continuation source seal hash:" + str(row.get("path")), ok, expected=row.get("sha256"), actual=sha256_file(artifact_path) if artifact_path and artifact_path.exists() else None, bytes=row.get("bytes"))


def check_continuation_seal_receipt(ver: Verifier, label: str, receipt: Any) -> None:
    seal_path = HERE / "source_seal.continuation.json"
    seal = load_required(ver, seal_path, "source_seal.continuation_for_" + label)
    if not isinstance(seal, dict):
        return
    ver.check(label + " raw continuation seal stage", seal.get("phase") == "before_stage_science" and seal.get("stage") == "continuation", phase=seal.get("phase"), stage=seal.get("stage"))
    raw_artifacts = seal.get("artifacts") or []
    raw_by_path = {str(row.get("path")): row for row in raw_artifacts if isinstance(row, dict)}
    if not isinstance(receipt, dict):
        ver.check(label + " continuation seal receipt present", False, receipt=receipt)
        return
    receipt_path = resolve(receipt.get("path"))
    actual_seal_sha = sha256_file(seal_path) if seal_path.exists() else None
    ver.check(
        label + " continuation seal receipt target",
        receipt_path is not None
        and receipt_path.resolve() == seal_path.resolve()
        and receipt.get("sha256") == actual_seal_sha,
        path=receipt.get("path"),
        expected_sha=actual_seal_sha,
        actual_sha=receipt.get("sha256"),
    )
    receipt_artifacts = receipt.get("artifacts") or []
    receipt_by_path = {str(row.get("path")): row for row in receipt_artifacts if isinstance(row, dict)}
    ver.check(
        label + " continuation seal receipt artifact set",
        set(receipt_by_path) == set(raw_by_path) and len(receipt_by_path) == 7,
        expected=sorted(raw_by_path),
        actual=sorted(receipt_by_path),
    )
    for artifact_path, raw_row in raw_by_path.items():
        receipt_row = receipt_by_path.get(artifact_path) or {}
        target = resolve(artifact_path)
        current_sha = sha256_file(target) if target is not None and target.exists() else None
        current_bytes = target.stat().st_size if target is not None and target.exists() else None
        expected_bytes = raw_row.get("bytes")
        ver.check(
            label + " continuation seal receipt artifact:" + artifact_path,
            receipt_row.get("sha256") == raw_row.get("sha256")
            and receipt_row.get("bytes") == expected_bytes
            and current_sha == raw_row.get("sha256")
            and current_bytes == expected_bytes,
            receipt_sha=receipt_row.get("sha256"),
            seal_sha=raw_row.get("sha256"),
            current_sha=current_sha,
            receipt_bytes=receipt_row.get("bytes"),
            seal_bytes=expected_bytes,
            current_bytes=current_bytes,
        )


def _static_truthy(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _static_positive_int(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value > 0


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in node.elts:
            names.extend(_target_names(elt))
        return names
    return []


def _static_json_violations(value: Any, path: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = f"{path}.{key}"
            if key == "include_test" and item is True:
                violations.append(f"include_test true at {key_path}")
            if key == "new_test_calls" and isinstance(item, int) and item > 0:
                violations.append(f"new_test_calls positive at {key_path}")
            if key == "test_origins":
                violations.append(f"test_origins key at {key_path}")
            violations.extend(_static_json_violations(item, key_path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            violations.extend(_static_json_violations(item, f"{path}[{idx}]"))
    return violations


def static_no_test_violations(text: str, filename: str = "<memory>") -> list[str]:
    if filename.endswith(".json"):
        try:
            return _static_json_violations(json.loads(text))
        except json.JSONDecodeError as exc:
            return [f"json parse failed:{exc.lineno}:{exc.colno}"]

    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError as exc:
        return [f"python parse failed:{exc.lineno}:{exc.offset}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", "?")
        if isinstance(node, ast.Attribute) and node.attr == "test_origins":
            violations.append(f".test_origins at line {lineno}")
        elif isinstance(node, ast.Name) and node.id == "test_origins":
            violations.append(f"test_origins name at line {lineno}")
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "test_origins":
            violations.append(f"['test_origins'] at line {lineno}")
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "include_test" and _static_truthy(keyword.value):
                    violations.append(f"include_test=True at line {lineno}")
                elif keyword.arg == "new_test_calls" and _static_positive_int(keyword.value):
                    violations.append(f"new_test_calls positive at line {lineno}")
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "include_test" and _static_truthy(value):
                    violations.append(f"include_test dict true at line {lineno}")
                elif isinstance(key, ast.Constant) and key.value == "new_test_calls" and _static_positive_int(value):
                    violations.append(f"new_test_calls dict positive at line {lineno}")
                elif isinstance(key, ast.Constant) and key.value == "test_origins":
                    violations.append(f"test_origins dict key at line {lineno}")
        elif isinstance(node, ast.Assign):
            names = [name for target in node.targets for name in _target_names(target)]
            if "include_test" in names and _static_truthy(node.value):
                violations.append(f"include_test assigned True at line {lineno}")
            if "new_test_calls" in names and _static_positive_int(node.value):
                violations.append(f"new_test_calls assigned positive at line {lineno}")
        elif isinstance(node, ast.AnnAssign):
            names = _target_names(node.target)
            if node.value is not None and "include_test" in names and _static_truthy(node.value):
                violations.append(f"include_test annotated True at line {lineno}")
            if node.value is not None and "new_test_calls" in names and _static_positive_int(node.value):
                violations.append(f"new_test_calls annotated positive at line {lineno}")
    return violations


def check_static_no_test(ver: Verifier) -> None:
    for path in sorted(HERE.glob("*.py")) + [HERE / "protocol.json"]:
        if not path.exists():
            continue
        hits = static_no_test_violations(path.read_text(encoding="utf-8", errors="replace"), path.name)
        ver.check("static no TEST path:" + path.name, path.name in {"verify.py", "test_verify.py"} or not hits, patterns=hits)


def check_science(ver: Verifier, protocol: dict[str, Any], result: dict[str, Any], method_result: dict[str, Any] | None, attempt_ledgers: dict[str, dict[str, Any]] | None = None) -> None:
    parent = read_json(PARENT_RESULT)
    method_protocol = load_required(ver, HERE / "method_protocol.json", "method_protocol") or {}
    if isinstance(method_protocol, dict):
        native_cfg = method_protocol.get("native_configuration") or {}
        export = method_protocol.get("export") or {}
        decoder = method_protocol.get("decoder") or {}
        ver.check("method protocol n_iters", int(native_cfg.get("n_iters", -1)) == int(protocol["total_updates_per_fit"]), expected=protocol["total_updates_per_fit"], actual=native_cfg.get("n_iters"))
        ver.check("method protocol causal rolling export", export.get("causal") is True and int(export.get("sliding_length", -1)) == 1 and int(export.get("scientific_origin_batch", -1)) == int(protocol["forecast_batch"]), export=export)
        ver.check("method protocol final SWA no VAL", "final swa" in json.dumps(method_protocol, ensure_ascii=False).lower() and "no val" in json.dumps(method_protocol, ensure_ascii=False).lower(), encoder_selection=(method_protocol.get("native_algorithm") or {}).get("encoder_selection"))
        ver.check("method protocol decoder grid", decoder.get("ridge_lambdas") == protocol.get("ridge_lambdas") and decoder.get("ridge_initial_lambdas") == protocol.get("ridge_initial_lambdas"), decoder=decoder)
    ver.check("result complete", result.get("status") in {"complete", "PASS", "passed"}, actual=result.get("status"))
    ver.check("method result complete", isinstance(method_result, dict) and method_result.get("status") in {"complete", "PASS", "passed"}, actual=method_result.get("status") if isinstance(method_result, dict) else None)
    expected_counts = {"decoder_adapters": 20, "ridge_fits": 200, "neural_fit_starts": 8, "benchmark_cases": 44}
    effective_counts = result.get("effective_counts") or {}
    for key, expected in expected_counts.items():
        ver.check("effective count:" + key, int(effective_counts.get(key, -1)) == expected, expected=expected, actual=effective_counts.get(key))
    execution_counts = result.get("execution_counts") or {}
    if execution_counts:
        ver.check("successful baseline execution counts", int(execution_counts.get("fit_starts", -1)) == 10 and int(execution_counts.get("completed_fits", -1)) == 8 and int(execution_counts.get("ridge_fits", -1)) == 200, execution_counts=execution_counts)
        if "continuation_counts" in execution_counts:
            cc = execution_counts["continuation_counts"]
            ver.check("baseline continuation counts", int(cc.get("fit_starts", -1)) == 3 and int(cc.get("completed_fits", -1)) == 3 and int(cc.get("ridge_fits", -1)) == 100, continuation_counts=cc)
        attempts = execution_counts.get("attempts") or {}
        if attempts:
            a1 = attempts.get("attempt1") or {}
            a2 = attempts.get("attempt2") or {}
            cont = attempts.get("continuation") or {}
            ver.check("baseline aggregate attempts", int(a1.get("fit_starts", -1)) == 1 and int(a2.get("fit_starts", -1)) == 6 and int(cont.get("fit_starts", -1)) == 3 and int(cont.get("completed_fits", -1)) == 3 and int(cont.get("ridge_fits", -1)) == 100, attempts=attempts)
        ver.check("baseline aggregate update fields", int(execution_counts.get("selected_fit_updates", -1)) == 23040 and int(execution_counts.get("all_baseline_optimizer_steps_lower", -1)) == 24384 and int(execution_counts.get("all_baseline_optimizer_steps_upper", -1)) == 24415 and int(execution_counts.get("native4_expected_final_task_starts", -1)) == 14 and int(execution_counts.get("native4_expected_completed_fits", -1)) == 12 and int(execution_counts.get("native4_successful_updates_total", -1)) == 34560, execution_counts=execution_counts)
        continuation = result.get("continuation") or {}
        if continuation:
            ver.check(
                "continuation metadata status",
                continuation.get("schema") == "k_tradeoff_continuation_v1"
                and isinstance(continuation.get("active_cache_reuse"), dict)
                and isinstance(continuation.get("source_seal_continuation"), dict)
                and isinstance(continuation.get("source_seal_continuation_after"), dict),
                continuation_keys=sorted(continuation),
            )
            seal_meta = continuation.get("source_seal_continuation") or {}
            check_continuation_seal_receipt(ver, "continuation before", seal_meta)
            check_continuation_seal_receipt(ver, "continuation after", continuation.get("source_seal_continuation_after"))
            stage_counts = continuation.get("current_stage_counts") or {}
            if stage_counts:
                ver.check("continuation current stage counts", int(stage_counts.get("fit_starts", -1)) == 3 and int(stage_counts.get("completed_fits", -1)) == 3 and int(stage_counts.get("ridge_fits", -1)) == 100, stage_counts=stage_counts)
    method_counts = (method_result or {}).get("execution_counts") if isinstance(method_result, dict) else {}
    if method_counts:
        ver.check("successful method execution counts", int(method_counts.get("fit_starts", -1)) == 4 and int(method_counts.get("completed_fits", -1)) == 4 and int(method_counts.get("ridge_fits", -1)) == 40, execution_counts=method_counts)
    if isinstance(attempt_ledgers, dict):
        attempt1 = attempt_ledgers.get("1") or {}
        attempt2 = attempt_ledgers.get("2") or {}
        failed1_starts = int(attempt1.get("fit_starts", 0))
        failed2_starts = int(attempt2.get("fit_starts", 0))
        method_starts = 4
        total_starts = int(execution_counts.get("fit_starts", 0)) + method_starts
        ver.check("task total fit starts include continuations", total_starts == 14, failed1=failed1_starts, failed2=failed2_starts, baseline_execution=execution_counts.get("fit_starts"), method=method_starts, total=total_starts)
        successful_updates = (8 + 4) * int(protocol["total_updates_per_fit"])
        failed1_min = int(attempt1.get("recorded_optimizer_steps", 0))
        failed1_extra = int(protocol["updates_per_epoch"]) - 1 if attempt1.get("exact_total_optimizer_steps") is None else 0
        failed2_min = int(attempt2.get("unfinished_recorded_optimizer_steps", 0))
        failed2_extra = int(attempt2.get("unfinished_possible_additional_steps_max", 0))
        lower = successful_updates + failed1_min + failed2_min
        upper = successful_updates + failed1_min + failed1_extra + failed2_min + failed2_extra
        ver.check("task optimizer step bounds include two failures", lower == 35904 and upper == 35935, lower=lower, upper=upper)
        selected_baseline_update = 8 * int(protocol["total_updates_per_fit"])
        ver.check("selected baseline successful updates", selected_baseline_update == 23040, selected_baseline_update=selected_baseline_update)
        baseline_lower = selected_baseline_update + failed1_min + failed2_min
        baseline_upper = selected_baseline_update + failed1_min + failed1_extra + failed2_min + failed2_extra
        ver.check("all baseline optimizer bounds", baseline_lower == 24384 and baseline_upper == 24415, lower=baseline_lower, upper=baseline_upper)
    baseline_choice_total = 0
    method_choice_total = 0
    baseline_adapter_total = 0
    method_adapter_total = 0
    mlp_fresh_fit_total = 0
    mlp_reuse_fit_total = 0
    for dataset in protocol["datasets"]:
        data = (result.get("datasets") or {}).get(dataset, {})
        ver.check("dataset exists:" + dataset, isinstance(data, dict) and bool(data), keys=sorted(data) if isinstance(data, dict) else None)
        if not isinstance(data, dict) or not data:
            continue
        archive, val_origins, target, mask = load_parent_archive(parent, dataset)
        ver.check("dataset source hash:" + dataset, isinstance(data.get("source"), dict) and resolve(data["source"].get("path")) is not None and resolve(data["source"].get("path")).exists() and sha256_file(resolve(data["source"].get("path"))) == data["source"].get("sha256"), source=data.get("source"))
        expected_dec_origins = deterministic_decoder_origins(archive, protocol)
        dec_receipt = data.get("decoder_train_origins") or {}
        ver.check("decoder TRAIN origins receipt:" + dataset, dec_receipt.get("count") == len(expected_dec_origins) and dec_receipt.get("seed") == int(protocol["decoder_sampling_seed"]) and dec_receipt.get("sha256") == sha256_array(expected_dec_origins), receipt=dec_receipt)
        adapters = collect_adapters(result, dataset)
        fits = collect_fits(result, dataset)
        method_adapters = collect_adapters(method_result or {}, dataset)
        method_fits = collect_fits(method_result or {}, dataset)
        middle_k = int(protocol["middle_k"][dataset])
        check_raw(ver, dataset + ":RAW_F0", data.get("raw_f0") or data.get("RAW_F0") or {}, val_origins, target, mask)
        raw_c = find_adapter(adapters, "RAW_C_D8", int(archive["x"].shape[1]))
        ver.check("RAW_C_D8 control exists:" + dataset, raw_c is not None)
        if raw_c:
            raw_basis = (data.get("basis") or {}).get(f"K{int(archive['x'].shape[1])}") if isinstance(data.get("basis"), dict) else None
            if isinstance(raw_basis, dict) and resolve(raw_basis.get("path")) and resolve(raw_basis.get("path")).exists():
                basis_arr = np.load(resolve(raw_basis.get("path")), allow_pickle=False)
                ver.check("RAW_C_D8 identity basis:" + dataset, np.array_equal(basis_arr, np.eye(int(archive["x"].shape[1]), dtype=np.float32)), kind=raw_basis.get("kind"))
            contract = get_contract(raw_c, protocol)
            ver.check("RAW_C_D8 contract exists:" + dataset, contract is not None)
            if contract:
                baseline_adapter_total += 1
                baseline_choice_total += check_contract(ver, dataset + f":RAW_C_D8:K{archive['x'].shape[1]}", raw_c, contract, archive, protocol, int(archive["x"].shape[1]), val_origins, target, mask)
        for k in protocol["k_grid"][dataset]:
            k = int(k)
            pca = find_adapter(adapters, "PCA", k)
            ver.check(f"PCA adapter exists:{dataset}:K{k}", pca is not None)
            if pca:
                baseline_adapter_total += 1
                check_pca(ver, f"{dataset}:PCA:K{k}", pca, data.get("basis") or {}, archive, k, middle_k)
                contract = get_contract(pca, protocol)
                ver.check(f"PCA contract exists:{dataset}:K{k}", contract is not None)
                if contract:
                    baseline_choice_total += check_contract(ver, f"{dataset}:PCA:K{k}", pca, contract, archive, protocol, k, val_origins, target, mask)
                    if k == middle_k:
                        check_trace_reuse(ver, f"{dataset}:PCA:K{k}", contract, expected_parent="PCA_ANCHOR")
            for seed in protocol["seeds"]:
                seed = int(seed)
                mlp = find_adapter(adapters, "MLP_POINT", k, seed)
                ver.check(f"MLP adapter exists:{dataset}:K{k}:seed{seed}", mlp is not None)
                if mlp:
                    baseline_adapter_total += 1
                    contract = get_contract(mlp, protocol)
                    ver.check(f"MLP contract exists:{dataset}:K{k}:seed{seed}", contract is not None)
                    if contract:
                        baseline_choice_total += check_contract(ver, f"{dataset}:MLP:K{k}:seed{seed}", mlp, contract, archive, protocol, k, val_origins, target, mask)
                        if k == middle_k:
                            check_trace_reuse(ver, f"{dataset}:MLP:K{k}:seed{seed}", contract, expected_parent=f"MLP_POINT_seed{seed}_m180")
                fit = find_fit(fits, "MLP_POINT", k, seed)
                ver.check(f"MLP fit receipt exists:{dataset}:K{k}:seed{seed}", fit is not None)
                if fit:
                    if check_fit(ver, f"{dataset}:MLP:K{k}:seed{seed}", fit, protocol, expect_new=(k != middle_k), archive=archive):
                        mlp_fresh_fit_total += 1
                    else:
                        mlp_reuse_fit_total += 1
        backbone = data.get("backbone") or {}
        ver.check("frozen backbone immutable:" + dataset, backbone.get("unchanged") is True and backbone.get("grads_none") is True and backbone.get("state_sha256_before") == backbone.get("state_sha256_after"), backbone=backbone)
        method_items = [(key, item) for key, item in method_adapters if infer_family(item, key) not in {"PCA", "MLP_POINT", "RAW_F0", "RAW_C_D8"} and infer_k(item, key) == middle_k]
        ver.check("method adapter count:" + dataset, len(method_items) == 2, count=len(method_items))
        for key, method in method_items:
            seed = infer_seed(method, key)
            label = f"{dataset}:METHOD:K{middle_k}:seed{seed}"
            method_adapter_total += 1
            contract = get_contract(method, protocol)
            ver.check("method contract exists:" + label, contract is not None)
            if contract:
                method_choice_total += check_contract(ver, label, method, contract, archive, protocol, middle_k, val_origins, target, mask, method=True)
            fit = find_fit(method_fits, "PROFESSOR_METHOD", middle_k, int(seed)) if seed is not None else None
            ver.check("method native fit receipt exists:" + label, fit is not None)
            if fit:
                check_method_fit(ver, label, fit, protocol, method_protocol if isinstance(method_protocol, dict) else {})
    ver.check("baseline decoder adapter count", baseline_adapter_total == 20, actual=baseline_adapter_total)
    ver.check("method adapter count total", method_adapter_total == 4, actual=method_adapter_total)
    ver.check("baseline ridge choices total", baseline_choice_total == 200, actual=baseline_choice_total)
    ver.check("method ridge choices total", method_choice_total == 40, actual=method_choice_total)
    ver.check("MLP fresh fit count", mlp_fresh_fit_total == 8, actual=mlp_fresh_fit_total)
    ver.check("MLP middle-K reuse count", mlp_reuse_fit_total == 4, actual=mlp_reuse_fit_total)


def collect_benchmark_rows(*results: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if (HERE / "benchmark.json").exists():
        obj = read_json(HERE / "benchmark.json")
        vals = obj.get("rows") if isinstance(obj, dict) else obj
        if isinstance(vals, list):
            rows.extend(vals)
    if (CACHE / "benchmarks").exists():
        for path in sorted((CACHE / "benchmarks").glob("*.json")):
            obj = read_json(path)
            rows.extend(obj if isinstance(obj, list) else [obj])
    for result in results:
        if not isinstance(result, dict):
            continue
        for key in ("benchmark_rows", "benchmarks", "cost_rows"):
            if isinstance(result.get(key), list):
                rows.extend(result[key])
    return [row for row in rows if isinstance(row, dict)]


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        seen.setdefault(hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(), row)
    return list(seen.values())


def benchmark_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("adapter_source", row.get("source", "local")),
        row.get("dataset"),
        row.get("adapter", row.get("case", row.get("id"))),
        int(row.get("batch", row.get("batch_size", -1))),
    )


def dedupe_benchmark_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for row in rows:
        key = benchmark_identity(row)
        prior = by_identity.get(key)
        if prior is None:
            by_identity[key] = row
            continue
        comparable = ["median_seconds", "p95_seconds", "cuda_max_allocated_bytes", "cuda_max_reserved_bytes", "rss_peak_bytes"]
        mismatch = [name for name in comparable if name in row and name in prior and row[name] != prior[name]]
        if mismatch:
            conflicts.append({"identity": key, "mismatch": mismatch})
    return list(by_identity.values()), conflicts


def expected_benchmark_identities(protocol: dict[str, Any], method_result: dict[str, Any] | None) -> set[tuple[Any, ...]]:
    identities: set[tuple[Any, ...]] = set()
    for dataset in protocol["datasets"]:
        local_cases = ["RAW_F0"]
        for k in protocol["k_grid"][dataset]:
            local_cases.append(f"{dataset}_PCA_K{int(k)}")
        for k in protocol["k_grid"][dataset]:
            for seed in protocol["seeds"]:
                local_cases.append(f"{dataset}_MLP_POINT_K{int(k)}_seed{int(seed)}")
        local_cases.append(f"{dataset}_RAW_C_D8")
        for adapter in local_cases:
            for batch in protocol["benchmark"]["batches"]:
                identities.add(("local", dataset, adapter, int(batch)))
        for adapter in ((method_result or {}).get("datasets", {}).get(dataset, {}).get("benchmark_cases", [])):
            for batch in protocol["benchmark"]["batches"]:
                identities.add(("method", dataset, adapter, int(batch)))
    return identities


def check_benchmark_input_receipts(ver: Verifier, label: str, row: dict[str, Any], archive: np.lib.npyio.NpzFile, protocol: dict[str, Any]) -> None:
    batch = int(row.get("batch", row.get("batch_size", -1)))
    expected_indices = (np.arange(batch, dtype=np.int64) % len(archive["val_origins"])).astype(np.int64)
    timed_input = row.get("timed_input") or {}
    timed_parity = row.get("timed_input_parity") or {}
    ver.check("benchmark input indices:" + label, timed_input.get("origin_indices") == expected_indices.tolist() and timed_input.get("origin_indices_sha256") == sha256_array(expected_indices) and timed_parity.get("input_indices_sha256") == sha256_array(expected_indices), timed_input=timed_input, timed_parity=timed_parity)
    mean = np.asarray(archive["mean"], dtype=np.float64)
    std = np.asarray(archive["std"], dtype=np.float64)
    val_context = windows_from_archive(archive, np.asarray(archive["val_origins"]), int(protocol["context"]))
    raw_batch = val_context[expected_indices].astype(np.float64) * std[None, None, :] + mean[None, None, :]
    normalized = ((raw_batch.astype(np.float64) - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    norm_receipt = row.get("normalization") or {}
    ver.check("benchmark TRAIN stats SHA:" + label, norm_receipt.get("mean_sha256") == sha256_array(mean) and norm_receipt.get("std_sha256") == sha256_array(std), normalization=norm_receipt)
    ver.check("benchmark raw/normalized input SHA:" + label, timed_input.get("raw_input_sha256") == sha256_array(raw_batch.astype(np.float64)) and timed_parity.get("raw_input_sha256") == sha256_array(raw_batch.astype(np.float64)) and timed_parity.get("normalized_input_sha256") == sha256_array(normalized), timed_input=timed_input, timed_parity=timed_parity)


def check_benchmarks(ver: Verifier, protocol: dict[str, Any], result: dict[str, Any], method_result: dict[str, Any] | None) -> None:
    raw_rows = collect_benchmark_rows(result, method_result)
    rows, conflicts = dedupe_benchmark_rows(raw_rows)
    ver.check("benchmark duplicate identities consistent", not conflicts, conflicts=conflicts[:10])
    ver.check("benchmark rows exist", bool(rows), count=len(rows), raw_count=len(raw_rows))
    expected_rows = 52
    ver.check("benchmark unique row count", len(rows) == expected_rows, expected=expected_rows, actual=len(rows))
    expected_ids = expected_benchmark_identities(protocol, method_result)
    actual_ids = {benchmark_identity(row) for row in rows}
    ver.check("benchmark exact expected identities", actual_ids == expected_ids, missing=sorted(expected_ids - actual_ids)[:10], extra=sorted(actual_ids - expected_ids)[:10], expected=len(expected_ids), actual=len(actual_ids))
    batches = {int(row.get("batch", row.get("batch_size", -1))) for row in rows}
    ver.check("benchmark includes B1/B32", batches >= {1, 32}, batches=sorted(batches))
    for row in rows:
        label = f"{row.get('dataset')}:{row.get('adapter', row.get('case', row.get('id')))}:B{row.get('batch', row.get('batch_size'))}"
        timings = row.get("timings_seconds") or row.get("raw_timings_seconds") or row.get("timings")
        ver.check("benchmark timings:" + label, isinstance(timings, list) and len(timings) == int(protocol["benchmark"]["repetitions"]) and np.all(np.isfinite(np.asarray(timings, dtype=np.float64))) and np.all(np.asarray(timings, dtype=np.float64) > 0), count=len(timings) if isinstance(timings, list) else None)
        if isinstance(timings, list) and timings:
            arr = np.asarray(timings, dtype=np.float64)
            ver.check("benchmark median:" + label, abs(float(row.get("median_seconds", math.nan)) - float(np.median(arr))) <= max(1e-9, float(np.median(arr)) * 1e-6), actual=row.get("median_seconds"))
            ver.check("benchmark p95:" + label, abs(float(row.get("p95_seconds", math.nan)) - float(np.percentile(arr, 95))) <= max(1e-9, float(np.percentile(arr, 95)) * 1e-6), actual=row.get("p95_seconds"))
        source_label = row.get("adapter_source", row.get("source", "local"))
        ver.check("benchmark source label:" + label, source_label in {"local", "baseline", "method"}, source=source_label)
        try:
            parent = read_json(PARENT_RESULT)
            archive, _, _, _ = load_parent_archive(parent, str(row.get("dataset")))
            check_benchmark_input_receipts(ver, label, row, archive, protocol)
        except Exception as exc:
            ver.check("benchmark input receipts:" + label, False, error=repr(exc))
        ver.check("benchmark warmups/repetitions:" + label, int(row.get("warmups", -1)) == int(protocol["benchmark"]["warmups"]) and int(row.get("repetitions", -1)) == int(protocol["benchmark"]["repetitions"]), warmups=row.get("warmups"), repetitions=row.get("repetitions"))
        ver.check("benchmark CUDA/RSS/backbone:" + label, int(row.get("cuda_peak_allocated_bytes", 0)) > 0 and int(row.get("cuda_peak_reserved_bytes", 0)) >= int(row.get("cuda_peak_allocated_bytes", 0)) and isinstance(row.get("rss"), dict) and int(row["rss"].get("sample_count", 0)) > 0 and row.get("backbone_state_sha256_before") == row.get("backbone_state_sha256_after"), cuda_alloc=row.get("cuda_peak_allocated_bytes"), cuda_reserved=row.get("cuda_peak_reserved_bytes"), rss=row.get("rss"))
        for receipt_key in ("parity", "timed_input_parity", "prediction_parity"):
            receipt = row.get(receipt_key)
            if isinstance(receipt, dict):
                ver.check("benchmark parity:" + label + ":" + receipt_key, receipt.get("passed") is True and int(receipt.get("violating_elements", -1)) == 0 and int(receipt.get("elements", 0)) > 0, receipt=receipt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ver = Verifier()
    protocol = load_required(ver, HERE / "protocol.json", "protocol")
    if isinstance(protocol, dict):
        assert_protocol(ver, protocol)
    check_static_no_test(ver)
    attempt_ledgers = check_failed_attempt_preservation(ver)
    result = load_required(ver, HERE / "result.json", "result")
    method_result = load_required(ver, HERE / "method_result.json", "method_result")
    check_sources(ver, result if isinstance(result, dict) else None)
    check_continuation_seal(ver, result if isinstance(result, dict) else None)
    check_continuation_receipt(ver)
    if isinstance(method_result, dict):
        check_sources(ver, method_result)
    if isinstance(protocol, dict) and isinstance(result, dict):
        check_science(ver, protocol, result, method_result if isinstance(method_result, dict) else None, attempt_ledgers)
        check_benchmarks(ver, protocol, result, method_result if isinstance(method_result, dict) else None)
    report = {"status": "PASS" if not ver.failures else "FAIL", "checks": ver.checks, "failure_count": len(ver.failures), "failures": ver.failures, "warnings": ver.warnings}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Verification {report['status']}: {len(ver.checks)} checks, {len(ver.failures)} failures")
        for failure in ver.failures[:80]:
            print("FAIL:", failure)
        if len(ver.failures) > 80:
            print(f"... {len(ver.failures) - 80} more failures")
    return 1 if ver.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
