"""Explicit saved-V statistics and small fixed-basis diagnostic; no model API."""
import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import zipfile

FORMAT = "fullC_fixed_basis_saved_V_diagnostic_v1"
FIELDS = ["run_id", "split", "cohort", "origin", "channel", "horizon_start", "horizon_end", "sse", "sae", "count"]
LABELS = ("RAW", "DIRECT_G", "RAW_RIDGE", "PCA_K3", "PCA_K6", "PCA_K11")
BASIS_KEYS = {"basis", "center", "mean", "std", "fit_end", "data_sha", "source_sha"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_bound(item, limit):
    path = Path(item["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Nonregular or oversized explicit input: " + item["label"])
    data = path.read_bytes()
    if len(data) != item["bytes"] or digest(data) != item["sha256"]:
        raise ValueError("Input bytes/SHA changed: " + item["label"])
    return data


def bound_json(item):
    return json.loads(read_bound(item, 1024 * 1024))


def bands(horizon):
    edges = sorted({0, math.ceil(horizon / 4), math.ceil(horizon / 2), horizon})
    return list(zip(edges[:-1], edges[1:]))


def parse_stats(data, manifest):
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        decoded = handle.read(4 * 1024 * 1024 + 1)
    if len(decoded) > 4 * 1024 * 1024:
        raise ValueError("Decompressed public statistics exceed limit")
    reader = csv.DictReader(io.StringIO(decoded.decode("utf-8"), newline=""))
    if reader.fieldnames != FIELDS:
        raise ValueError("Unexpected public statistics header")
    score = manifest["score"]
    horizon, width, n = score["horizon"], score["num_channels"], score["n_origins"]
    allowed_bands = set(bands(horizon))
    result, origins, channels = {}, [], []
    for row in reader:
        if any(row[k] != manifest[k] for k in ("run_id", "split", "cohort")) or row["split"] != "V_select":
            raise ValueError("Statistics identity/split differs")
        if any(not re.fullmatch(r"0|[1-9][0-9]*", row[k]) for k in ("origin", "horizon_start", "horizon_end", "count")):
            raise ValueError("Noncanonical nonnegative integer in statistics")
        origin, start, end, count = (int(row[k]) for k in ("origin", "horizon_start", "horizon_end", "count"))
        if (start, end) not in allowed_bands or count > end - start or not row["channel"]:
            raise ValueError("Invalid horizon/channel/count")
        sse, sae = float(row["sse"]), float(row["sae"])
        if not all(math.isfinite(x) and x >= 0 for x in (sse, sae)) or (count == 0 and (sse != 0 or sae != 0)):
            raise ValueError("Invalid finite nonnegative sufficient statistics")
        key = (origin, row["channel"], start, end)
        if key in result:
            raise ValueError("Duplicate origin/channel/band cell")
        result[key] = (sse, sae, count)
        if origin not in origins:
            origins.append(origin)
        if row["channel"] not in channels:
            channels.append(row["channel"])
    if len(origins) != n or origins != sorted(origins) or len(channels) != width:
        raise ValueError("Incomplete or unordered origins/channels")
    expected = {(o, c, a, b) for o in origins for c in channels for a, b in allowed_bands}
    if set(result) != expected:
        raise ValueError("Public statistics omit/add cohort cells")
    return result, origins, channels


def macro(rows, keys):
    by_channel = {}
    for key in keys:
        by_channel.setdefault(key[1], []).append(rows[key])
    per_channel = []
    for channel, values in by_channel.items():
        count = sum(v[2] for v in values)
        if count:
            per_channel.append((math.fsum(v[0] for v in values) / count, math.fsum(v[1] for v in values) / count))
    return dict(mse=math.fsum(v[0] for v in per_channel) / len(per_channel) if per_channel else None,
                mae=math.fsum(v[1] for v in per_channel) / len(per_channel) if per_channel else None,
                sse=math.fsum(rows[k][0] for k in keys), sae=math.fsum(rows[k][1] for k in keys),
                count=sum(rows[k][2] for k in keys), active_channels=len(per_channel))


def groups(origins, channels, horizon):
    result = [("all", "all", lambda key: True)]
    # Three contiguous groups, sizes differ by at most one; no random resampling.
    q, remainder = divmod(len(origins), 3)
    offset = 0
    for index in range(3):
        size = q + int(index < remainder)
        selected = frozenset(origins[offset:offset + size])
        result.append(("origin_block", str(index + 1), lambda key, values=selected: key[0] in values))
        offset += size
    for start, end in bands(horizon):
        result.append(("horizon_band", f"{start}:{end}", lambda key, a=start, b=end: key[2:] == (a, b)))
    for channel in channels:
        result.append(("channel", channel, lambda key, value=channel: key[1] == value))
    return result


def comparisons(all_rows, origins, channels, horizon):
    output = []
    for left, right in itertools.combinations(all_rows, 2):
        a, b = all_rows[left], all_rows[right]
        if set(a) != set(b) or any(a[k][2] != b[k][2] for k in a):
            raise ValueError("Paired cohorts/cell observation counts differ")
        for kind, name, predicate in groups(origins, channels, horizon):
            keys = [key for key in a if predicate(key)]
            observed = [key for key in keys if a[key][2] > 0]
            am, bm = macro(a, keys), macro(b, keys)
            row = dict(left=left, right=right, group_kind=kind, group=name, count=am["count"],
                       active_channels=am["active_channels"], paired_cells=len(observed))
            for metric in ("sse", "sae", "mse", "mae"):
                row["left_" + metric], row["right_" + metric] = am[metric], bm[metric]
                row["delta_" + metric] = am[metric] - bm[metric] if am[metric] is not None else None
            for index, metric in enumerate(("sse", "sae")):
                row[metric + "_left_wins"] = sum(a[k][index] < b[k][index] for k in observed)
                row[metric + "_right_wins"] = sum(a[k][index] > b[k][index] for k in observed)
                row[metric + "_ties"] = sum(a[k][index] == b[k][index] for k in observed)
            output.append(row)
    return output


def array_fingerprint(value, np):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=digest(array.tobytes()))


def basis_diagnostic(arrays, np):
    e, center = np.asarray(arrays["basis"], dtype=np.float64), np.asarray(arrays["center"], dtype=np.float64)
    c = len(center)
    if e.shape != (c, c) or center.shape != (c,) or c != 11:
        raise ValueError("Expected Beijing full-C 11x11 basis and center")
    for name in ("basis", "center", "mean", "std"):
        if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError("Expected finite original FP32 basis/scaler parameters")
    if any(arrays[name].shape != (c,) for name in ("mean", "std")) or not np.all(arrays["std"] > 0):
        raise ValueError("Invalid scaler dimensions or standard deviation")
    singular = np.linalg.svd(e, compute_uv=False)
    rank_tolerance = float(singular.max() * max(e.shape) * np.finfo(np.float64).eps)
    prefix_rows = []
    for k in (3, 6, 11):
        ek = e[:, :k]
        projection = ek @ ek.T
        residual = projection - np.eye(c)
        # Deterministic coordinate probes diagnose the affine operator, not real history reconstruction.
        probes = np.vstack([np.zeros((1, c)), np.eye(c), -np.eye(c)])
        reconstructed = (probes - center) @ ek @ ek.T + center
        prefix_rows.append(dict(k=k, gram_max_abs=float(np.max(np.abs(ek.T @ ek - np.eye(k)))),
            projection_rank=int(np.linalg.matrix_rank(projection)),
            projection_identity_max_abs=float(np.max(np.abs(residual))),
            projection_identity_frobenius=float(np.linalg.norm(residual, "fro")),
            affine_zero_offset_max_abs=float(np.max(np.abs(center - center @ projection))),
            deterministic_coordinate_probe_max_abs=float(np.max(np.abs(reconstructed - probes)))))
    return dict(stored_dtype=str(arrays["basis"].dtype), arithmetic_dtype="float64",
                singular_values=[float(v) for v in singular], full_basis_rank=int(np.sum(singular > rank_tolerance)),
                rank_tolerance_numeric_only=rank_tolerance,
                condition_number_2=float(singular.max() / singular.min()) if singular.min() > 0 else None,
                prefix_diagnostics=prefix_rows, performance_pass_threshold=None,
                actual_history_reconstruction_tested=False, target_or_prediction_arrays_read=False)


def load_basis(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if set(names) != {key + ".npy" for key in BASIS_KEYS} or len(names) != len(BASIS_KEYS):
            raise ValueError("Small basis NPZ has unexpected members")
        if sum(item.file_size for item in archive.infolist()) > 65536:
            raise ValueError("Basis NPZ uncompressed parameters exceed 64KiB")
    import numpy as np
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, np


def verify_metadata(manifests, references, score_rows, origins, channels, np):
    first = next(iter(manifests.values()))
    shared_names = ("data_sha", "scaler_sha", "split", "cohort")
    array_names = ("target_normalized", "observed", "origins", "channel_ids")
    if first["split"] != "V_select" or first["score"]["metric_scale"] != "A_normalized_original_channels":
        raise ValueError("Only saved developmental normalized V is permitted")
    for label, manifest in manifests.items():
        reference, score = references[label], score_rows[label]
        for key in shared_names:
            if manifest[key] != first[key] or reference[key] != manifest[key]:
                raise ValueError("Data/scaler/cohort metadata mismatch: " + key)
        for key in ("run_id", "source_sha", "model_sha", "config_sha"):
            if reference[key] != manifest[key]:
                raise ValueError("Reference identity differs: " + key)
        for key in array_names:
            if manifest["arrays"][key] != first["arrays"][key] or reference["arrays"][key] != manifest["arrays"][key]:
                raise ValueError("Saved original cohort array fingerprint mismatch: " + key)
        for key in ("run_id", "split", "cohort"):
            if score[key] != manifest[key]:
                raise ValueError("Score identity differs")
        for key in ("mse", "mae"):
            if float(score[key]) != manifest["score"][key]:
                raise ValueError("Score CSV and manifest disagree")
        for name, values in (("origins", origins), ("channel_ids", channels)):
            dtype = manifest["arrays"][name]["dtype"]
            if array_fingerprint(np.asarray(values, dtype=dtype), np) != manifest["arrays"][name]:
                raise ValueError("Statistics coordinates differ from manifest fingerprint")
    return {key: first[key] for key in shared_names} | {"arrays": {key: first["arrays"][key] for key in array_names}}


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(spec_item, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Fresh output directory required")
    spec_bytes = read_bound(spec_item, 1024 * 1024)
    spec = json.loads(spec_bytes)
    if spec.get("format") != FORMAT or tuple(spec["routes"]) != LABELS:
        raise ValueError("Expected explicit six-route Beijing diagnostic specification")
    if spec.get("expected_dimensions") != {"origins": 43, "horizon": 48, "channels": 11}:
        raise ValueError("This diagnostic is limited to Beijing V43/H48/C11")
    all_items = [spec["basis"], spec["basis_contract"]] + list(spec["sources"].values())
    for route in spec["routes"].values():
        if set(route) != {"stats", "manifest", "reference", "scores"}:
            raise ValueError("Unexpected route input")
        all_items.extend(route.values())
    paths = [str(Path(item["path"]).resolve()) for item in all_items]
    if len(paths) != len(set(paths)) or any(output.resolve() == Path(path) or output.resolve() in Path(path).parents for path in paths):
        raise ValueError("Input duplication or output/input overlap")
    source_bytes = {name: read_bound(item, 1024 * 1024) for name, item in spec["sources"].items()}
    basis_bytes = read_bound(spec["basis"], 1024 * 1024)
    contract = bound_json(spec["basis_contract"])
    if contract["artifact_sha256"] != digest(basis_bytes):
        raise ValueError("Immutable basis contract differs")
    arrays, np = load_basis(basis_bytes)
    diagnostics = basis_diagnostic(arrays, np)
    if int(arrays["fit_end"].item()) != contract["fit_end"] or str(arrays["data_sha"].item()) != contract["data_sha"] or str(arrays["source_sha"].item()) != contract["source_sha"]:
        raise ValueError("Basis embedded data/time/source contract mismatch")
    ordered_sources = ("ordered_pilot.py", "prediction_cache.py", "fit_base.py", "training.py", "networks.py", "data_contract.py", "evaluation.py", "gpu_runtime.py")
    if set(source_bytes) != set(ordered_sources) | {"ridge_baseline.py"}:
        raise ValueError("Explicit nine traced source files required")
    if fingerprint({name: digest(source_bytes[name]) for name in ordered_sources}) != contract["source_sha"]:
        raise ValueError("Actual current ordered sources differ from basis contract")
    manifests, references, score_rows, rows = {}, {}, {}, {}
    origins = channels = None
    for label, inputs in spec["routes"].items():
        manifests[label], references[label] = bound_json(inputs["manifest"]), bound_json(inputs["reference"])
        with io.StringIO(read_bound(inputs["scores"], 65536).decode("utf-8"), newline="") as stream:
            scores = list(csv.DictReader(stream))
        if len(scores) != 1:
            raise ValueError("Expected one saved score row")
        score_rows[label] = scores[0]
        rows[label], own_origins, own_channels = parse_stats(read_bound(inputs["stats"], 1024 * 1024), manifests[label])
        if origins is not None and (own_origins != origins or own_channels != channels):
            raise ValueError("Statistics original coordinates/order differ")
        origins, channels = own_origins, own_channels
    if (len(origins), len(channels)) != (43, 11):
        raise ValueError("Unexpected actual cohort dimensions")
    cohort = verify_metadata(manifests, references, score_rows, origins, channels, np)
    scaler = fingerprint({name: array_fingerprint(arrays[name], np) for name in ("mean", "std")})
    if scaler != cohort["scaler_sha"] or cohort["data_sha"] != contract["data_sha"] or contract["fit_end"] != 12622:
        raise ValueError("Basis/scaler/time differ from saved V")
    for label, k in (("PCA_K3", 3), ("PCA_K6", 6), ("PCA_K11", 11)):
        meta = manifests[label]["metadata"]
        if meta["parent_artifact_sha256"] != digest(basis_bytes) or meta["inference_latent_k"] != k or meta["stored_latent_k"] != 11 or meta["variant"] != "nested_pca":
            raise ValueError("Saved prefix is not the exact fixed common-basis route")
    foundation_sha = manifests["PCA_K11"]["metadata"]["foundation_sha"]
    if any(manifests[label]["metadata"]["foundation_sha"] != foundation_sha for label in ("PCA_K3", "PCA_K6")):
        raise ValueError("Nested prefixes have different foundation identities")
    raw_identity = fingerprint(dict(baseline=dict(kind="RAW", fold="primary", L=512, H=48),
                                   foundation_sha=foundation_sha, scaler_sha=scaler, data_sha=cohort["data_sha"]))
    if manifests["RAW"]["model_sha"] != raw_identity:
        raise ValueError("Raw saved model fingerprint does not bind the same foundation/scaler/data")
    scores_out = []
    for label, cells in rows.items():
        score = macro(cells, list(cells))
        for metric in ("mse", "mae"):
            if not math.isclose(score[metric], manifests[label]["score"][metric], rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError("Full-V sufficient-statistics rescore mismatch")
        if score["count"] != manifests[label]["score"]["n_observed"]:
            raise ValueError("Observed target count differs")
        scores_out.append(dict(label=label, run_id=manifests[label]["run_id"], **score,
                               n_origins=len(origins), horizon=48, original_channels=len(channels)))
    paired = comparisons(rows, origins, channels, 48)
    # Recheck all inputs after computation; never mutate original scientific files.
    for item in all_items:
        read_bound(item, 1024 * 1024)
    read_bound(spec_item, 1024 * 1024)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "RESCORED_V.csv", scores_out)
    write_csv(output / "PAIRED_GROUPS.csv", paired)
    binding = {"spec_sha256": digest(spec_bytes), "inputs": [{key: item[key] for key in ("label", "bytes", "sha256")} for item in all_items],
               "source_sha256": {name: digest(value) for name, value in source_bytes.items()}}
    (output / "INPUT_BINDINGS.json").write_text(json.dumps(binding, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    result = dict(format=FORMAT, status="SAVED_V_STATISTICS_AND_FIXED_BASIS_DIAGNOSTIC_COMPLETE",
        science_completion_claimed=False, performance_success_claimed=False, cohort=cohort,
        basis=diagnostics, scores=scores_out, paired_group_rows=len(paired),
        raw_and_nested_foundation_identity_formula_matched=True, foundation_sha=foundation_sha,
        cell_wins_are_strict_sufficient_statistic_comparisons=True,
        cell_definition="one origin x original channel x registered horizon band, observed count > 0",
        uncertainty=["single seed and repeated post-T developmental V; no independent generalization",
                     "metadata target/mask fingerprints matched, but target arrays/mask positions not independently reconstructed",
                     "no model/weights/raw/Task/target/prediction arrays or protected T opened; no forward, training, cost or process operation",
                     "FP64 rescore covers saved SSE/SAE statistics, not a new prediction-array audit",
                     "paired origin/channel/band cells and temporal blocks are descriptive, not independent significance samples",
                     "basis invertibility does not imply row forecaster rotation equivariance or future predictability",
                     "normalization/centering/decoder/anchor route differences prevent TSFM-only causal attribution"])
    (output / "RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--spec")
    parser.add_argument("--spec-sha256")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(format=FORMAT, mode="metadata_only", input_reads=0, numpy_imported=False, model_or_data_loading=False)))
        return 0
    if not all((args.spec, args.spec_sha256, args.output)) or not re.fullmatch("[0-9a-f]{64}", args.spec_sha256):
        parser.error("--execute requires explicit spec, SHA256 and fresh output")
    path = Path(args.spec)
    result = execute(dict(path=str(path), label="explicit_spec", bytes=path.stat().st_size, sha256=args.spec_sha256), args.output)
    print(json.dumps(dict(status=result["status"], routes=len(result["scores"]), paired_group_rows=result["paired_group_rows"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
