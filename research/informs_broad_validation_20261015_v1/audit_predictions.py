"""Independent CPU audit of private prediction caches and public statistics.

This module deliberately does not import the training/evaluation scorer. The
reference contract must be frozen independently from the allowed data cohort,
before predictions are generated. Hashes supplied only by the prediction writer
cannot establish target, channel, mask, or origin correctness.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ARRAY_KEYS = {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}
REFERENCE_KEYS = ARRAY_KEYS - {"pred_normalized"}
IDENTITY_KEYS = ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "model_sha", "scaler_sha")
ATOL = 1e-9
RTOL = 1e-7


def _same_number(actual, expected, label):
    if expected is None:
        if actual is not None:
            raise ValueError(f"{label}: expected no observed targets")
        return
    try:
        actual = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: not numeric") from exc
    if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=ATOL, rel_tol=RTOL):
        raise ValueError(f"{label}: {actual} != {expected}")


def _integer(value, label):
    try:
        number = int(value)
        if isinstance(value, (bool, np.bool_)) or float(value) != number:
            raise ValueError()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: expected integer") from exc
    return number


def _fingerprint(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_registry(registry, expected_ids, run_id):
    expected_ids = [str(value) for value in expected_ids]
    actual_ids = [str(row["run_id"]) for row in registry]
    if len(set(expected_ids)) != len(expected_ids) or len(set(actual_ids)) != len(actual_ids):
        raise ValueError("Duplicate expected or actual registry run_id")
    if set(actual_ids) != set(expected_ids):
        raise ValueError("Registry does not cover every locked expected run, including failed routes")
    rows = {str(row["run_id"]): row for row in registry}
    if run_id not in rows:
        raise ValueError("Scored run absent from registry")
    for row in rows.values():
        status = str(row.get("execution_status", ""))
        if not status:
            raise ValueError("Missing registry execution_status")
        if status != "COMPLETE_VALID" and not str(row.get("reason", "")):
            raise ValueError("Noncomplete routes require an explicit reason")
    if rows[run_id]["execution_status"] != "COMPLETE_VALID":
        raise ValueError("Numeric score belongs to a failed/unexecuted route")


def _check_cost(cost_records, run_id, reference):
    rows = [row for row in cost_records if str(row.get("run_id")) == run_id]
    if not rows:
        raise ValueError("Missing F stream execution ledger for scored route")
    for row in rows:
        actual = _integer(row["actual_F_streams"], "actual_F_streams")
        expected = _integer(row["expected_F_streams"], "expected_F_streams")
        if actual < 0 or expected < 0 or actual != expected:
            raise ValueError("Actual F stream count differs from declared route")
        if "batch" in row and "latent_k" in row:
            batch = _integer(row["batch"], "batch")
            k = _integer(row["latent_k"], "latent_k")
            calls = _integer(row.get("F_calls", 1), "F_calls")
            if batch < 1 or k < 0 or calls < 0 or expected != batch * k * calls:
                raise ValueError("F stream count inconsistent with batch, selected K and F calls")
        locked = reference.get("expected_F_streams")
        if locked is not None:
            if isinstance(locked, dict):
                key = str(row.get("batch"))
                if key not in locked or expected != _integer(locked[key], "locked F streams"):
                    raise ValueError("Cost route differs from independent stream budget")
            elif expected != _integer(locked, "locked F streams"):
                raise ValueError("Cost route differs from independent stream budget")
    return len(rows)


def _load_arrays(manifest, reference, private_root):
    relative = Path(str(manifest["cache_relative_path"]))
    root = Path(private_root).resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError("Prediction cache must be a relative path within caller-supplied private root")
    if not path.is_file() or _file_sha(path) != manifest["archive_sha256"]:
        raise ValueError("Prediction archive missing or SHA256 mismatch")
    if set(manifest.get("arrays", {})) != ARRAY_KEYS or set(reference.get("arrays", {})) != REFERENCE_KEYS:
        raise ValueError("Manifest/reference must contain the fixed array fingerprint keys")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != ARRAY_KEYS:
                raise ValueError("Prediction NPZ keys differ from locked cache schema")
            arrays = {key: archive[key] for key in ARRAY_KEYS}
    except (OSError, TypeError) as exc:
        raise ValueError("Unreadable prediction archive") from exc
    for key, value in arrays.items():
        fingerprint = _fingerprint(value)
        if fingerprint != manifest["arrays"][key]:
            raise ValueError(f"Prediction manifest array fingerprint mismatch: {key}")
        if key in REFERENCE_KEYS and fingerprint != reference["arrays"][key]:
            raise ValueError(f"Independent cohort fingerprint mismatch: {key}")
    return arrays


def _independent_statistics(arrays, identity):
    pred, target, mask = (arrays[key] for key in ("pred_normalized", "target_normalized", "observed"))
    origins, channels = arrays["origins"], arrays["channel_ids"]
    if pred.ndim != 3 or pred.shape != target.shape or pred.shape != mask.shape or min(pred.shape) < 1:
        raise ValueError("Prediction/target/mask require matching positive [N,H,C] dimensions")
    n, horizon, width = pred.shape
    if pred.dtype.kind not in "fi" or target.dtype.kind not in "fi" or mask.dtype != np.bool_:
        raise ValueError("Numeric predictions/targets and original boolean observation mask required")
    if origins.shape != (n,) or origins.dtype.kind not in "iu" or len(set(map(int, origins))) != n:
        raise ValueError("Unique ordered integer origins required")
    if channels.shape != (width,) or channels.dtype.kind not in "US" or len(set(map(str, channels))) != width:
        raise ValueError("Unique ordered string channel IDs required")
    if any(not str(channel) for channel in channels):
        raise ValueError("Empty channel ID")
    if not np.isfinite(pred[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite values at observed targets")
    edges = sorted({0, (horizon + 3) // 4, (horizon + 1) // 2, horizon})
    bands = list(zip(edges[:-1], edges[1:]))
    expected = {}
    channel_totals = {str(channel): [[], [], 0] for channel in channels}
    for index, origin in enumerate(origins):
        for col, channel in enumerate(channels):
            channel = str(channel)
            for start, stop in bands:
                observed = mask[index, start:stop, col]
                p = pred[index, start:stop, col][observed].astype(np.float64)
                y = target[index, start:stop, col][observed].astype(np.float64)
                differences = [float(a) - float(b) for a, b in zip(p, y)]
                sse = math.fsum(value * value for value in differences)
                sae = math.fsum(abs(value) for value in differences)
                count = len(differences)
                expected[(int(origin), channel, start, stop)] = (sse, sae, count)
                channel_totals[channel][0].append(sse)
                channel_totals[channel][1].append(sae)
                channel_totals[channel][2] += count
    active = [value for value in channel_totals.values() if value[2]]
    metrics = dict(mse=math.fsum(math.fsum(value[0]) / value[2] for value in active) / len(active) if active else None,
                   mae=math.fsum(math.fsum(value[1]) / value[2] for value in active) / len(active) if active else None,
                   n_observed=sum(value[2] for value in channel_totals.values()),
                   n_origins=n, num_channels=width, horizon=horizon, active_channels=len(active))
    return expected, metrics


def _reaggregate(statistics, expected, score):
    seen = set()
    totals = defaultdict(lambda: [[], [], 0])
    identity = tuple(str(score[key]) for key in ("run_id", "split", "cohort"))
    for row in statistics:
        if tuple(str(row.get(key)) for key in ("run_id", "split", "cohort")) != identity:
            raise ValueError("Public statistics belong to another run/split/cohort")
        key = (_integer(row["origin"], "origin"), str(row["channel"]),
               _integer(row["horizon_start"], "horizon_start"), _integer(row["horizon_end"], "horizon_end"))
        if key in seen or key not in expected:
            raise ValueError("Unexpected, duplicated, shifted, or overlapping public band")
        seen.add(key)
        expected_sse, expected_sae, expected_count = expected[key]
        count = _integer(row["count"], "count")
        if count != expected_count:
            raise ValueError("Public observed count differs from original mask")
        if count == 0 and (float(row["sse"]) != 0 or float(row["sae"]) != 0):
            raise ValueError("Zero-count public bands require exactly zero error sums")
        _same_number(row["sse"], expected_sse, "band SSE")
        _same_number(row["sae"], expected_sae, "band SAE")
        if float(row["sse"]) < 0 or float(row["sae"]) < 0:
            raise ValueError("Negative public error statistic")
        totals[key[1]][0].append(float(row["sse"]))
        totals[key[1]][1].append(float(row["sae"]))
        totals[key[1]][2] += count
    if seen != set(expected):
        raise ValueError("Public statistics omit origin/channel/band coverage, including zero-count rows")
    active = [value for value in totals.values() if value[2]]
    return dict(mse=math.fsum(math.fsum(value[0]) / value[2] for value in active) / len(active) if active else None,
                mae=math.fsum(math.fsum(value[1]) / value[2] for value in active) / len(active) if active else None,
                n_observed=sum(value[2] for value in totals.values()))


def audit_prediction_bundle(manifest, reference_contract, score, statistics, registry,
                            expected_run_ids, cost_records, private_root):
    """Raise ValueError on any contract/metric/coverage/stream-ledger mismatch.

    Each call audits one run/split/cohort. ``statistics`` must already be filtered
    to that identity. The lexical expected-run registry includes failed routes.
    No model is loaded and no inference or timing measurement is replayed.
    """
    for key in IDENTITY_KEYS:
        if not str(manifest.get(key, "")) or manifest.get(key) != reference_contract.get(key):
            raise ValueError(f"Independent identity mismatch: {key}")
    run_id = str(manifest["run_id"])
    for key in ("run_id", "split", "cohort"):
        if str(score.get(key)) != str(manifest[key]):
            raise ValueError(f"Score identity mismatch: {key}")
    _check_registry(registry, expected_run_ids, run_id)
    cost_rows = _check_cost(cost_records, run_id, reference_contract)
    arrays = _load_arrays(manifest, reference_contract, private_root)
    expected, independent = _independent_statistics(arrays, manifest)
    aggregate = _reaggregate(statistics, expected, score)
    for key in ("mse", "mae"):
        _same_number(score.get(key), independent[key], f"score {key}")
        _same_number(aggregate[key], independent[key], f"public reaggregation {key}")
    for key in ("n_observed", "n_origins", "num_channels", "horizon"):
        if _integer(score.get(key), key) != independent[key]:
            raise ValueError(f"Score dimension/count mismatch: {key}")
    if "active_channels" in score and _integer(score["active_channels"], "active_channels") != independent["active_channels"]:
        raise ValueError("Score active channel count mismatch")
    return dict(status="PASS", run_id=run_id, split=str(manifest["split"]), cohort=str(manifest["cohort"]),
                document_audit=True, public_stats_reaggregation=True, prediction_rescoring=True,
                checkpoint_replay=False, training_rerun=False, timer_reexecution=False,
                n_statistics=len(expected), n_cost_records=cost_rows, atol=ATOL, rtol=RTOL, **independent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "reference", "score", "stats", "registry", "expected-ids", "cost", "private-root"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    def read_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    with Path(args.stats).open(encoding="utf-8-sig", newline="") as handle:
        statistics = list(csv.DictReader(handle))
    result = audit_prediction_bundle(read_json(args.manifest), read_json(args.reference), read_json(args.score),
                                     statistics, read_json(args.registry), read_json(args.expected_ids),
                                     read_json(args.cost), args.private_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
