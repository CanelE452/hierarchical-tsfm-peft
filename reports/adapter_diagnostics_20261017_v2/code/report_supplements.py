"""Saved-result-only paired-origin diagnostics and masked horizon-band summaries."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from research.adapter_diagnostics_20261017_v2.core import atomic_json, read, sha


CONTRACT_FIELDS = ("scaler_sha256", "target_sha256", "mask_sha256", "channel_sha256")
BANDS = ((1, 16), (17, 32), (33, 48))
ORIGIN_SCOPE = ("Each origin: mean over observed channels of SSE_h/count_h or SAE_h/count_h. "
                "The unweighted mean of these origin macro differences generally differs from "
                "the global channel macro difference when observation counts vary. No CI or p-value.")


def _csv(path, values, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for value in values:
        writer.writerow({key: json.dumps(item, sort_keys=True) if isinstance(item, (list, dict)) else item
                         for key, item in value.items()})
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(buffer.getvalue()); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _encoded_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _fingerprint(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=_hash(array))


def _reference(path, ctx):
    path = Path(path).resolve()
    for scheme in ("report", "cache", "repo", "asset"):
        if scheme in ctx and path.is_relative_to(Path(ctx[scheme]).resolve()):
            return scheme+"://"+path.relative_to(Path(ctx[scheme]).resolve()).as_posix()
    raise ValueError("Supplement source outside registered roots")


def _resolved(reference, ctx):
    scheme, relative = reference.split("://", 1)
    if scheme not in ctx:
        raise FileNotFoundError("Source root unavailable: "+scheme)
    root = Path(ctx[scheme]).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Supplement source escapes registered root")
    return path


def _source_record(run, ctx):
    path = (Path(ctx["repo"])/run["source_file"]).resolve()
    if not path.is_relative_to(Path(ctx["repo"]).resolve()) or not path.is_file():
        raise FileNotFoundError("Bound public producer absent")
    if sha(path) != run["source_sha256"]:
        raise ValueError("Bound public producer SHA mismatch")
    if path.suffix != ".json":
        return {}, [dict(reference=_reference(path, ctx), sha256=sha(path))]
    value = read(path)
    value = value.get("result", value)
    if isinstance(value.get("rows"), list):
        candidates = [item for item in value["rows"] if item.get("method") == run["method"]
                      and item.get("evaluation", item.get("split", "V_select")) == run["split"]
                      and all(abs(float(item[key])-float(run[key])) <= 1e-10 for key in ("MSE", "MAE"))]
        if len(candidates) != 1:
            return {}, [dict(reference=_reference(path, ctx), sha256=sha(path))]
        value = candidates[0]
    return value, [dict(reference=_reference(path, ctx), sha256=sha(path))]


def _registered_origins(run, ctx):
    contract_path = Path(ctx["repo"])/"research/informs_broad_validation_20261015_v1/DATA_CONTRACT.json"
    contract = read(contract_path)["assets"]["pems08" if run["cohort"] == "pems32" else run["cohort"]]
    start, stop = contract["primary_split_ranges"][run["split"]]
    length, horizon = int(float(run["L"])), int(float(run["H"]))
    origins = np.arange(max(start, length), stop-horizon+1, horizon, dtype=np.int64)
    if _hash(origins) != run.get("origins_sha256"):
        raise ValueError("Registered evaluation origins do not match saved fingerprint")
    return origins, dict(reference=_reference(contract_path, ctx), sha256=sha(contract_path))


def _validate_statistics(run, origins, sse, sae, count, sources):
    origins, sse, sae, count = map(np.asarray, (origins, sse, sae, count))
    if (origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer)
            or len(set(origins.tolist())) != len(origins) or sse.ndim != 2
            or sse.shape != sae.shape or sse.shape != count.shape or len(origins) != len(sse)
            or sse.shape[1] != int(float(run["C"])) or not np.isfinite(sse).all()
            or not np.isfinite(sae).all() or not np.isfinite(count).all()
            or np.any(sse < 0) or np.any(sae < 0) or np.any(count < 0)
            or np.any(count != np.floor(count)) or np.any(count > int(float(run["H"])))
            or np.any((count == 0) & ((sse != 0) | (sae != 0)))):
        raise ValueError("Invalid saved per-origin sufficient statistics")
    totals = count.sum(0); active = totals > 0
    if not active.any():
        raise ValueError("No observed channels")
    actual = (np.mean(sse.sum(0)[active]/totals[active]), np.mean(sae.sum(0)[active]/totals[active]))
    if max(abs(float(run[key])-value) for key, value in zip(("MSE", "MAE"), actual)) > 1e-10:
        raise ValueError("Saved origin statistics disagree with global FP64 score")
    order = np.argsort(origins)
    return dict(origins=origins[order], sse=sse[order], sae=sae[order], count=count[order], sources=sources)


def _registered_channels(run, ctx, data, sources, b02):
    if not str(run.get("channel_order_validation", "")).startswith("DERIVED_"):
        raise ValueError("Saved prediction channel order is unverified")
    if b02 and run.get("channel_order_validation") != "DERIVED_REGISTERED_ORDER_FROM_BOUND_B02_WRITER":
        raise ValueError("B02 saved prediction channel order lacks explicit writer derivation")
    repo = Path(ctx["repo"])
    contract = "research/informs_broad_validation_20261015_v1/DATA_CONTRACT.json"
    paths = [contract]
    if b02:
        paths.append("research/adapter_diagnostics_20261017_v2/ridge_jobs.py")
    subset = "research/informs_broad_validation_20261015_v1/priority_pems_subset_20261016_v1/PLAN.json"
    if run["cohort"] == "pems32":
        paths.append(subset)
    for relative in paths:
        path = repo/relative
        if not path.is_file():
            raise ValueError("Derived channel order source is absent: "+relative)
        digest = sha(path)
        if b02 and data.get("bindings", {}).get(relative) != digest:
            raise ValueError("B02 derived channel order source binding mismatch: "+relative)
        sources.append(dict(reference=_reference(path, ctx), sha256=digest))
    if run["cohort"] == "pems32":
        columns = read(repo/subset)["column_selection"]["selected_ids"]
    else:
        columns = read(repo/contract)["assets"][run["cohort"]]["columns"]
    return np.asarray(columns, dtype=str)


def _producer_fingerprints(record, actual, canonical):
    checks = []
    fields = {"target_sha256": "target_normalized", "mask_sha256": "observed", "observed_sha256": "observed",
              "origins_sha256": "origins", "channel_sha256": "channel_ids", "channel_order_sha256": "channel_ids"}

    def check(field, key, expected, metadata=None):
        candidates = [actual[key]]
        if _fingerprint(canonical[key]) != _fingerprint(actual[key]):
            candidates.append(canonical[key])
        for value in candidates:
            if metadata is not None and (metadata.get("shape", list(value.shape)) != list(value.shape)
                                         or metadata.get("dtype", str(value.dtype)) != str(value.dtype)):
                continue
            for scheme, digest in (("raw_array_bytes", _hash(value)), ("dtype_shape_and_bytes", _encoded_hash(value))):
                if expected == digest:
                    checks.append(dict(field=field, array=key, scheme=scheme, sha256=expected,
                                       shape=list(value.shape), dtype=str(value.dtype)))
                    return
        raise ValueError("Producer array fingerprint mismatch: "+field)

    for field, key in fields.items():
        if record.get(field) is not None:
            check(field, key, record[field])
    for container in ("array_fingerprints", "arrays"):
        for key, metadata in record.get(container, {}).items():
            if key in actual and isinstance(metadata, dict) and metadata.get("sha256"):
                check(container+"."+key, key, metadata["sha256"], metadata)
    return checks


def _archive_stats(run, record, ctx, data, sources):
    expected = run.get("independent_private_score_sha256") or record.get("private_prediction_archive_sha256")
    candidates = []
    if run.get("_archive_path") and expected:
        candidates.append(dict(reference=_reference(run["_archive_path"], ctx), sha256=expected))
    if record.get("private_prediction_ref") and record.get("private_prediction_archive_sha256"):
        candidates.append(dict(reference=record["private_prediction_ref"], sha256=record["private_prediction_archive_sha256"]))
    for item in data.get("new_stats_replay", []):
        if item.get("run_id") == run["run_id"]:
            candidates += [binding for binding in item.get("private_artifacts", [])
                           if binding.get("sha256") == expected]
    strong = bool(run.get("_archive_path") and run.get("independent_private_score_sha256"))
    for binding in candidates[:1] if strong else candidates:
        path = _resolved(binding["reference"], ctx)
        if not path.is_file():
            if strong:
                raise FileNotFoundError("Independently bound saved prediction is absent")
            continue
        if sha(path) != binding["sha256"]:
            raise ValueError("Saved prediction SHA mismatch")
        for field in ("prediction_sha256", "private_prediction_archive_sha256"):
            if record.get(field) is not None and record[field] != binding["sha256"]:
                raise ValueError("Producer prediction archive SHA mismatch: "+field)
        for item in record.get("private_artifacts", []):
            if item.get("reference") == binding["reference"] and item.get("sha256") != binding["sha256"]:
                raise ValueError("Producer prediction archive SHA mismatch")
        if record.get("scaler_sha256") is not None and record["scaler_sha256"] != run.get("scaler_sha256"):
            raise ValueError("Producer/run contract mismatch: scaler_sha256")
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            b02 = False
            if {"pred_normalized", "target_normalized", "observed", "origins"}.issubset(keys):
                p, y, mask, origins = [archive[key] for key in ("pred_normalized", "target_normalized", "observed", "origins")]
            elif {"prediction", "y", "mask", "origins"}.issubset(keys):
                p, y, mask, origins = [archive[key] for key in ("prediction", "y", "mask", "origins")]
            elif {"own_prediction", "cross_prediction", "target", "observed", "origins"}.issubset(keys):
                method_keys = {"OWN_CHANNEL_32BLOCK_RIDGE": "own_prediction", "ALL_CHANNEL_32BLOCK_RIDGE": "cross_prediction"}
                if run["method"] not in method_keys:
                    raise ValueError("B02 saved prediction method is unregistered")
                p, y, mask, origins = [archive[key] for key in (method_keys[run["method"]], "target", "observed", "origins")]
                b02 = True
            else:
                if strong:
                    raise ValueError("Independently bound saved prediction schema is unsupported")
                continue
            if (p.shape != y.shape or p.shape != mask.shape or p.ndim != 3 or mask.dtype != np.bool_
                    or y.shape[1:] != (int(float(run["H"])), int(float(run["C"])))):
                raise ValueError("Saved prediction axes/mask mismatch")
            canonical_target = y
            target_hash = _hash(y)
            if target_hash != run["target_sha256"]:
                converted = y.astype(np.float32)
                if (not run.get("evaluation_contract_derivation") or _hash(converted) != run["target_sha256"]
                        or not np.array_equal(y, converted.astype(y.dtype), equal_nan=True)):
                    raise ValueError("Saved prediction target fingerprint mismatch")
                canonical_target = converted
            if _hash(mask) != run["mask_sha256"] or _hash(origins) != run["origins_sha256"]:
                raise ValueError("Saved prediction mask/origin fingerprint mismatch")
            if "channel_ids" in keys:
                channels = archive["channel_ids"]
            else:
                channels = _registered_channels(run, ctx, data, sources, b02)
            if (channels.shape != (int(float(run["C"])),) or len(set(channels.tolist())) != len(channels)
                    or _hash(channels) != run["channel_sha256"]):
                raise ValueError("Saved prediction channel fingerprint mismatch")
            actual = dict(pred_normalized=p, target_normalized=y, observed=mask, origins=origins, channel_ids=channels)
            canonical = dict(actual, target_normalized=canonical_target)
            fingerprints = {key: _fingerprint(value) for key, value in canonical.items()}
            for item in data.get("new_stats_replay", []):
                if item.get("run_id") != run["run_id"]:
                    continue
                recorded = item.get("arrays", {})
                if item.get("private_archive_sha256") not in (None, binding["sha256"]):
                    raise ValueError("Independent replay archive fingerprint mismatch")
                for key in fingerprints.keys() & recorded.keys():
                    if fingerprints[key] != recorded[key]:
                        raise ValueError("Independent replay array fingerprint mismatch: "+key)
            producer_checks = _producer_fingerprints(record, actual, canonical)
            error = np.where(mask, p.astype(np.float64)-y.astype(np.float64), 0.)
            stats = _validate_statistics(run, origins, np.square(error).sum(1), np.abs(error).sum(1), mask.sum(1), sources+[binding])
            stats.update(array_fingerprints=fingerprints, producer_fingerprint_validation=producer_checks)
            return stats
    raise FileNotFoundError("No bound prediction or saved origin sufficient statistics")


def _origin_stats(run, ctx, data):
    record, sources = _source_record(run, ctx)
    if run.get("_archive_path") and run.get("independent_private_score_sha256"):
        return _archive_stats(run, record, ctx, data, sources)
    arrays = record.get("array_fingerprints", record.get("arrays", {}))
    fields = {"target_sha256": "target_normalized", "mask_sha256": "observed",
              "origins_sha256": "origins", "channel_sha256": "channel_ids"}
    aliases = {"mask_sha256": "observed_sha256", "channel_sha256": "channel_order_sha256"}
    for field in (*CONTRACT_FIELDS, "origins_sha256"):
        value = record.get(field, record.get(aliases.get(field, "")))
        if field in fields:
            value = arrays.get(fields[field], {}).get("sha256", value)
        if value is not None and value != run.get(field):
            raise ValueError("Producer/run contract mismatch: "+field)
    if all(key in record for key in ("origin_sse", "origin_sae", "origin_counts")):
        if "origins" in record:
            origins = np.asarray(record["origins"], dtype=np.int64)
            if _hash(origins) != run.get("origins_sha256"):
                raise ValueError("Saved origin IDs disagree with run fingerprint")
        else:
            origins, binding = _registered_origins(run, ctx)
            sources.append(binding)
        return _validate_statistics(run, origins, record["origin_sse"], record["origin_sae"], record["origin_counts"], sources)
    return _archive_stats(run, record, ctx, data, sources)


def paired_origin_rows(reference, method, left, right):
    for field in (*CONTRACT_FIELDS, "cohort", "split", "metric_scale", "C", "H"):
        a, b = reference.get(field), method.get(field)
        if a in (None, "", "NA") or a != b:
            raise ValueError("Origin pairing contract mismatch: "+field)
    if not np.array_equal(left["origins"], right["origins"]):
        raise ValueError("Origin pairing ID mismatch")
    if not np.array_equal(left["count"], right["count"]):
        raise ValueError("Origin pairing observed counts mismatch")
    count = left["count"]; active = count > 0
    usable = active.sum(1)
    if np.any(usable == 0):
        raise ValueError("Origin without observed channels")
    rows = []
    for index, origin in enumerate(left["origins"]):
        keep = active[index]
        values = {prefix+"_"+metric: float(np.mean(stats[key][index, keep]/count[index, keep]))
                  for prefix, stats in (("reference", left), ("method", right))
                  for metric, key in (("MSE", "sse"), ("MAE", "sae"))}
        rows.append(dict(origin_id=int(origin), observed_channels=int(usable[index]), observed_count=int(count[index].sum()),
                         delta_MSE=values["method_MSE"]-values["reference_MSE"],
                         delta_MAE=values["method_MAE"]-values["reference_MAE"], **values))
    return rows


def horizon_band_rows(data, pairs):
    by_run = defaultdict(dict)
    for item in data["horizon"]:
        key = (int(item["horizon"]), str(item["channel_id"]))
        if key in by_run[item["run_id"]]:
            raise ValueError("Duplicate horizon/channel sufficient statistic")
        by_run[item["run_id"]][key] = item
    runs = {run["run_id"]: run for run in data["runs"]+data.get("endpoints", [])}
    output = []
    for pair in pairs:
        a, b = runs.get(pair["reference_run_id"]), runs.get(pair["method_run_id"])
        if a is None or b is None or a["cohort"] != "jena" or b["cohort"] != "jena":
            continue
        for lo, hi in BANDS:
            row = dict(reference_run_id=a["run_id"], method_run_id=b["run_id"], reference=a["method"], method=b["method"],
                       cohort="jena", seed=pair.get("seed"), split=a["split"], metric_scale=a["metric_scale"],
                       horizon_start=lo, horizon_end=hi, match_status=pair.get("match_status"),
                       contract_hashes={field: [a.get(field), b.get(field)] for field in (*CONTRACT_FIELDS, "origins_sha256")},
                       reference_source_file=a["source_file"], reference_source_sha256=a["source_sha256"],
                       method_source_file=b["source_file"], method_source_sha256=b["source_sha256"])
            summaries = []
            try:
                if any(a.get(field) in (None, "", "NA") or a.get(field) != b.get(field)
                       for field in (*CONTRACT_FIELDS, "origins_sha256", "split", "metric_scale")):
                    raise ValueError("Different or missing canonical evaluation contract")
                for run in (a, b):
                    entries = by_run.get(run["run_id"], {})
                    channels = sorted({channel for _, channel in entries if channel != "ALL_CHANNELS_EQUAL_COUNTS"})
                    if len(channels) != int(float(run["C"])) or any((h, c) not in entries for c in channels for h in range(lo, hi+1)):
                        raise FileNotFoundError("Missing channel-resolved band sufficient statistics")
                    cells = []
                    for channel in channels:
                        selected = [entries[(h, channel)] for h in range(lo, hi+1)]
                        if any(float(v["count"]) != int(v["count"]) for v in selected):
                            raise ValueError("Noninteger band observation count")
                        cell = dict(channel_id=channel, SSE=math.fsum(float(v["SSE"]) for v in selected),
                                    SAE=math.fsum(float(v["SAE"]) for v in selected), count=sum(int(v["count"]) for v in selected))
                        if (cell["count"] < 0 or cell["SSE"] < 0 or cell["SAE"] < 0
                                or not all(math.isfinite(cell[k]) for k in ("SSE", "SAE"))
                                or (cell["count"] == 0 and (cell["SSE"] != 0 or cell["SAE"] != 0))):
                            raise ValueError("Invalid band sufficient statistics")
                        cells.append(cell)
                    observed = [cell for cell in cells if cell["count"] > 0]
                    if not observed:
                        raise ValueError("Band has no observed channels")
                    summaries.append(dict(MSE=math.fsum(v["SSE"]/v["count"] for v in observed)/len(observed),
                                          MAE=math.fsum(v["SAE"]/v["count"] for v in observed)/len(observed), cells=cells))
                if [v["count"] for v in summaries[0]["cells"]] != [v["count"] for v in summaries[1]["cells"]]:
                    raise ValueError("Band paired observation counts differ")
                row.update(status="COMPLETE", reason=None)
                for prefix, summary in zip(("reference", "method"), summaries):
                    row.update({prefix+"_"+key: summary[key] for key in ("MSE", "MAE")})
                    row[prefix+"_channel_statistics"] = summary["cells"]
                for metric in ("MSE", "MAE"):
                    row["delta_"+metric] = row["method_"+metric]-row["reference_"+metric]
                    row["delta_"+metric+"_pct"] = 100*row["delta_"+metric]/row["reference_"+metric] if row["reference_"+metric] else None
            except (ValueError, FileNotFoundError) as error:
                row.update(status="CONTRACT_MISMATCH" if isinstance(error, ValueError) else "MISSING_SUFFICIENT_STATISTICS", reason=str(error))
            output.append(row)
    return output


def build_supplements(ctx, data, pairs):
    runs = {run["run_id"]: run for run in data["runs"]+data.get("endpoints", [])}
    unique = {(pair["reference_run_id"], pair["method_run_id"]): pair for pair in pairs}
    pairs = list(unique.values())
    memo, coverage, private_rows = {}, [], []
    for pair in pairs:
        row = {key: pair.get(key) for key in ("reference_run_id", "method_run_id", "cohort", "seed", "comparison_scope", "match_status")}
        try:
            selected = [runs[identifier] for identifier in (pair["reference_run_id"], pair["method_run_id"])]
            for run in selected:
                identifier = run["run_id"]
                if identifier not in memo:
                    try:
                        memo[identifier] = _origin_stats(run, ctx, data)
                    except (ValueError, FileNotFoundError, KeyError) as error:
                        memo[identifier] = error
                if isinstance(memo[identifier], Exception):
                    raise type(memo[identifier])(identifier+": "+str(memo[identifier]))
            left, right = [memo[run["run_id"]] for run in selected]
            values = paired_origin_rows(*selected, left, right)
            private_rows.extend(dict(reference_run_id=selected[0]["run_id"], method_run_id=selected[1]["run_id"], **value) for value in values)
            row.update(status="COMPLETE", reason=None, origin_count=len(values),
                       mean_origin_delta_MSE=math.fsum(v["delta_MSE"] for v in values)/len(values),
                       mean_origin_delta_MAE=math.fsum(v["delta_MAE"] for v in values)/len(values),
                       min_origin_delta_MSE=min(v["delta_MSE"] for v in values), max_origin_delta_MSE=max(v["delta_MSE"] for v in values),
                       min_origin_delta_MAE=min(v["delta_MAE"] for v in values), max_origin_delta_MAE=max(v["delta_MAE"] for v in values),
                       global_delta_MSE=float(selected[1]["MSE"])-float(selected[0]["MSE"]),
                       global_delta_MAE=float(selected[1]["MAE"])-float(selected[0]["MAE"]),
                       origin_ids_sha256=_hash(left["origins"]), observed_counts_sha256=_hash(left["count"]),
                       contract_hashes={field: selected[0][field] for field in CONTRACT_FIELDS},
                       evaluation_fingerprint_validation={prefix: dict(canonical_arrays=stats.get("array_fingerprints"),
                           producer_checks=stats.get("producer_fingerprint_validation"))
                           for prefix, stats in (("reference", left), ("method", right))},
                       source_bindings=left["sources"]+right["sources"])
        except (ValueError, FileNotFoundError, KeyError) as error:
            row.update(status="CONTRACT_MISMATCH" if isinstance(error, ValueError) else "MISSING_BOUND_STATISTICS", reason=str(error), origin_count=None)
        row["statistic_scope"] = ORIGIN_SCOPE
        coverage.append(row)
    band_rows = horizon_band_rows(data, pairs)
    private_path = Path(ctx["cache"])/"report_supplements/PAIRED_ORIGIN_DELTAS.csv"
    _csv(private_path, private_rows, ["reference_run_id", "method_run_id", "origin_id", "observed_channels", "observed_count",
                                    "reference_MSE", "method_MSE", "delta_MSE", "reference_MAE", "method_MAE", "delta_MAE"])
    report = Path(ctx["report"])
    paired_path, band_path = report/"PAIRED_ORIGIN_COVERAGE.csv", report/"HORIZON_BAND_SUMMARY.csv"
    _csv(paired_path, coverage, list(dict.fromkeys(key for row in coverage for key in row)) or ["status"])
    _csv(band_path, band_rows, list(dict.fromkeys(key for row in band_rows for key in row)) or ["status"])
    binding = dict(schema="REPORT_SUPPLEMENTS_v1", helper_sha256=sha(__file__), registered_pairs=len(pairs),
                   paired_coverage_counts=dict(Counter(row["status"] for row in coverage)),
                   band_coverage_counts=dict(Counter(row["status"] for row in band_rows)),
                   private_artifacts=[dict(reference=_reference(private_path, ctx), sha256=sha(private_path))],
                   public_outputs=[dict(reference=_reference(path, ctx), sha256=sha(path)) for path in (paired_path, band_path)],
                   statistic_scope=ORIGIN_SCOPE,
                   publication_policy="Origin-wise numeric CSV private: policy does not explicitly authorize this release. Public coverage, summaries and bindings only; no raw predictions/targets.",
                   uncertainty="No new p-values, CIs, bootstrap or independent-generalization claim")
    binding_path = report/"PAIRED_ORIGIN_BINDINGS.json"
    atomic_json(binding_path, binding)
    return dict(public_files=[paired_path, band_path, binding_path], private_files=[private_path], band_rows=band_rows,
                paired_coverage=coverage, coverage_counts=binding["paired_coverage_counts"], binding=binding)
