"""Independent read-only verification of public supplement coverage and private statistics."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from research.adapter_diagnostics_20261017_v2.core import bound_references_valid, read, sha


FIELDS = ("scaler_sha256", "target_sha256", "mask_sha256", "channel_sha256", "origins_sha256")
BANDS = ((1, 16), (17, 32), (33, 48))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def pair(row):
    return row["reference_run_id"], row["method_run_id"]


def near(actual, expected, label):
    if actual in (None, "", "NA") or not math.isfinite(float(actual)) or abs(float(actual)-expected) > 1e-10:
        raise ValueError(label+" arithmetic mismatch")


def decoded(value, default):
    return json.loads(value) if isinstance(value, str) and value else value if value is not None else default


def resolve(reference, ctx):
    scheme, relative = reference.split("://", 1)
    if scheme not in ("repo", "report", "cache", "asset") or scheme not in ctx:
        raise ValueError("Supplement binding has no registered root")
    root = Path(ctx[scheme]).resolve()
    target = (root/relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Supplement binding escapes registered root")
    return target


def _band_score(statistics, run, lo, hi):
    channels = {channel for h, channel in statistics if channel != "ALL_CHANNELS_EQUAL_COUNTS"}
    if len(channels) != int(float(run["C"])):
        raise FileNotFoundError("Missing channel-resolved horizon coverage")
    cells, active = [], []
    for channel in sorted(channels):
        selected = []
        for horizon in range(lo, hi+1):
            if (horizon, channel) not in statistics:
                raise FileNotFoundError("Missing horizon position")
            row = statistics[(horizon, channel)]
            count = float(row["count"])
            if not count.is_integer() or count < 0:
                raise ValueError("Invalid horizon count")
            sse, sae = float(row["SSE"]), float(row["SAE"])
            if not all(math.isfinite(value) and value >= 0 for value in (sse, sae)) or (count == 0 and (sse or sae)):
                raise ValueError("Invalid horizon sufficient statistics")
            selected.append((sse, sae, int(count)))
        cell = dict(channel_id=channel, SSE=math.fsum(x[0] for x in selected),
                    SAE=math.fsum(x[1] for x in selected), count=sum(x[2] for x in selected))
        cells.append(cell)
        if cell["count"]:
            active.append(cell)
    if not active:
        raise ValueError("Horizon band has no observed channel")
    return dict(MSE=math.fsum(x["SSE"]/x["count"] for x in active)/len(active),
                MAE=math.fsum(x["SAE"]/x["count"] for x in active)/len(active), cells=cells)


def _trajectory(ctx, run_map, bind):
    report = Path(ctx["report"])
    path = report/"TRAJECTORY_DIAGNOSTICS.json"
    if not path.is_file():
        return dict(status="NOT_PRESENT", reason="Trajectory output is not present in this report")
    value = read(path)
    plan = read(report/"EXPERIMENT_PLAN.json")
    a03 = {job["job_id"] for job in plan["jobs"] if job["question_id"] == "A03" and job["split"] == "V_select"}
    b03 = {job["job_id"] for job in plan["jobs"] if job["question_id"] == "B03"}
    records = [record for group in value["a03_sampler_pairs"] for record in group["records"]]
    identifiers = [record["job_id"] for record in records]
    if (len(identifiers) != len(set(identifiers)) or not set(identifiers).issubset(a03)
            or value["expected_a03_snapshots"] != len(a03) or value["checked_a03_snapshots"] != len(records)
            or value["expected_b03_jobs"] != len(b03) or value["checked_b03_jobs"] > len(b03)):
        raise ValueError("Trajectory expected/checked coverage differs")
    checked_b03 = {record["run_id"].rsplit("_", 1)[0] for record in value["b03_fit_rows"]}
    if len(checked_b03) != value["checked_b03_jobs"] or not checked_b03.issubset(b03):
        raise ValueError("Trajectory B03 checked job coverage differs")
    if value["status"].startswith("PASS") and (set(identifiers) != a03 or checked_b03 != b03):
        raise ValueError("Trajectory PASS conceals missing registered jobs")
    if value.get("model_calls") != 0 or value.get("new_training") is not False:
        raise ValueError("Trajectory scope includes unregistered model/training work")
    for relative, expected in value["public_sources"].items():
        bind(dict(reference="repo://"+relative, sha256=expected))
    for item in value["private_artifacts"]:
        bind(item)
    for record in records:
        bind(record["checkpoint"])
        for item in record["public_references"]:
            bind(item)
    joins = 0
    for record in value["b03_fit_rows"]:
        for item in record.get("public_references", [])+record.get("private_artifacts", []):
            bind(item)
        if record["run_id"] not in run_map:
            continue
        run = run_map[record["run_id"]]
        for field in ("MSE", "MAE", "fit_MSE", "fit_MAE"):
            if record.get(field) is not None:
                near(run.get(field), float(record[field]), "Trajectory fit/run join "+field)
        if (run.get("method") != record["method"] or run.get("split") != record["evaluation"]
                or run.get("fit_score_source") != "TRAJECTORY_DIAGNOSTICS.json"
                or not run.get("fit_score_scope")):
            raise ValueError("Trajectory fit/run join scope differs")
        joins += 1
    return dict(status=value["status"], checked_a03_snapshots=len(records), expected_a03_snapshots=len(a03),
                checked_b03_jobs=len(checked_b03), expected_b03_jobs=len(b03), checked_fit_run_joins=joins,
                uncovered_jobs=sorted((a03-set(identifiers)) | (b03-checked_b03)))


def verify_supplements(ctx, data=None):
    report = Path(ctx["report"])
    registered = {}
    for filename in ("MATCHED_COMPARISONS.csv", "CONTEXTUAL_COMPARISONS.csv", "SAME_UPDATE_COMPARISONS.csv"):
        for row in rows(report/filename):
            registered[pair(row)] = row
    run_map = {}
    for filename in ("RUNS_FULL_PRECISION.csv", "SAME_UPDATE_RESULTS.csv"):
        for row in rows(report/filename):
            if row["run_id"] in run_map:
                raise ValueError("Duplicate public run identity")
            run_map[row["run_id"]] = row
    coverage = rows(report/"PAIRED_ORIGIN_COVERAGE.csv")
    if len(coverage) != len(registered) or {pair(row) for row in coverage} != set(registered):
        raise ValueError("Registered unique pair coverage differs")
    binding = read(report/"PAIRED_ORIGIN_BINDINGS.json")
    bindings = {}
    def bind(item):
        reference, expected = item["reference"], item["sha256"]
        resolve(reference, ctx)
        if reference in bindings and bindings[reference]["sha256"] != expected:
            raise ValueError("Supplement source binding changes within report")
        bindings[reference] = dict(reference=reference, sha256=expected)
    for item in binding["public_outputs"]+binding["private_artifacts"]:
        bind(item)
    if not bound_references_valid(list(bindings.values()), ctx):
        raise ValueError("Supplement public/private binding rejected")
    if binding["registered_pairs"] != len(registered):
        raise ValueError("Registered unique pair count differs")
    private_rows = defaultdict(list)
    if len(binding["private_artifacts"]) != 1:
        raise ValueError("Paired origin private CSV binding is ambiguous")
    for row in rows(resolve(binding["private_artifacts"][0]["reference"], ctx)):
        private_rows[pair(row)].append(row)
    complete = 0
    uncovered = []
    for row in coverage:
        key = pair(row)
        if any(identifier not in run_map for identifier in key):
            raise ValueError("Pair coverage refers to absent public run")
        left, right = [run_map[identifier] for identifier in key]
        if row["status"] != "COMPLETE":
            if row["status"] not in ("CONTRACT_MISMATCH", "MISSING_BOUND_STATISTICS") or not row.get("reason") or private_rows.get(key):
                raise ValueError("Incomplete pair coverage lacks honest status/reason")
            uncovered.append(dict(reference_run_id=key[0], method_run_id=key[1], status=row["status"], reason=row["reason"]))
            continue
        complete += 1
        for field in FIELDS[:-1]:
            if not left.get(field) or left[field] != right.get(field):
                raise ValueError("Completed pair has different canonical contracts")
        source_bindings = decoded(row.get("source_bindings"), [])
        for item in source_bindings:
            bind(item)
        for run in (left, right):
            expected = dict(reference="repo://"+run["source_file"], sha256=run["source_sha256"])
            if not any(resolve(item["reference"], ctx) == resolve(expected["reference"], ctx)
                       and item["sha256"] == expected["sha256"] for item in source_bindings):
                raise ValueError("Completed pair producer source binding missing")
            bind(expected)
        origins = private_rows.get(key, [])
        if not origins or len(origins) != int(row["origin_count"]) or len({int(value["origin_id"]) for value in origins}) != len(origins):
            raise ValueError("Completed pair origin coverage differs")
        for value in origins:
            if int(value["observed_channels"]) < 1 or int(value["observed_count"]) < 1:
                raise ValueError("Invalid paired origin observed count")
            for metric in ("MSE", "MAE"):
                near(value["delta_"+metric], float(value["method_"+metric])-float(value["reference_"+metric]), "Private origin delta")
        for metric in ("MSE", "MAE"):
            deltas = [float(value["delta_"+metric]) for value in origins]
            for prefix, expected in (("mean", math.fsum(deltas)/len(deltas)), ("min", min(deltas)), ("max", max(deltas))):
                near(row[prefix+"_origin_delta_"+metric], expected, "Public paired origin "+prefix)
            near(row["global_delta_"+metric], float(right[metric])-float(left[metric]), "Global pair delta")
    if set(private_rows)-{pair(row) for row in coverage if row["status"] == "COMPLETE"}:
        raise ValueError("Private origin statistics have unregistered pair coverage")
    if dict(Counter(row["status"] for row in coverage)) != binding["paired_coverage_counts"]:
        raise ValueError("Paired coverage status counts differ")
    horizon = defaultdict(dict)
    horizon_binding = read(report/"HORIZON_STATS_BINDINGS.json")
    if horizon_binding["statistic_sha256"] != sha(report/"ERROR_BY_HORIZON.csv"):
        raise ValueError("Horizon statistics binding rejected")
    for row in rows(report/"ERROR_BY_HORIZON.csv"):
        identity = (int(row["horizon"]), row["channel_id"])
        if identity in horizon[row["run_id"]]:
            raise ValueError("Duplicate horizon sufficient statistics")
        horizon[row["run_id"]][identity] = row
    if set(horizon) != set(horizon_binding["runs"]):
        raise ValueError("Horizon source binding run coverage differs")
    for descriptor in horizon_binding["runs"].values():
        if descriptor.get("source_file") and descriptor.get("source_sha256"):
            bind(dict(reference="repo://"+descriptor["source_file"], sha256=descriptor["source_sha256"]))
    bands = rows(report/"HORIZON_BAND_SUMMARY.csv")
    expected_bands = {(*key, lo, hi) for key in registered
                      if all(identifier in run_map and run_map[identifier]["cohort"] == "jena" for identifier in key)
                      for lo, hi in BANDS}
    identities = [(*pair(row), int(row["horizon_start"]), int(row["horizon_end"])) for row in bands]
    if len(identities) != len(set(identities)) or set(identities) != expected_bands:
        raise ValueError("Registered Jena horizon band coverage differs")
    checked_bands = 0
    for row in bands:
        left, right = [run_map[identifier] for identifier in pair(row)]
        lo, hi = int(row["horizon_start"]), int(row["horizon_end"])
        problem = None
        try:
            if any(not left.get(field) or left[field] != right.get(field) for field in (*FIELDS, "split", "metric_scale")):
                raise ValueError("Different canonical band evaluation contracts")
            summaries = [_band_score(horizon[run["run_id"]], run, lo, hi) for run in (left, right)]
            if [cell["count"] for cell in summaries[0]["cells"]] != [cell["count"] for cell in summaries[1]["cells"]]:
                raise ValueError("Paired band observation counts differ")
        except (ValueError, FileNotFoundError) as error:
            problem = error
        if problem is not None:
            expected_status = "CONTRACT_MISMATCH" if isinstance(problem, ValueError) else "MISSING_SUFFICIENT_STATISTICS"
            if row["status"] != expected_status or not row.get("reason"):
                raise ValueError("Incomplete band lacks honest coverage status/reason")
            continue
        if row["status"] != "COMPLETE":
            raise ValueError("Computable band is silently marked incomplete")
        checked_bands += 1
        for metric in ("MSE", "MAE"):
            for prefix, summary in zip(("reference", "method"), summaries):
                near(row[prefix+"_"+metric], summary[metric], "Independent band macro")
            delta = summaries[1][metric]-summaries[0][metric]
            near(row["delta_"+metric], delta, "Independent band delta")
            if summaries[0][metric]:
                near(row["delta_"+metric+"_pct"], 100*delta/summaries[0][metric], "Independent band percent")
    if dict(Counter(row["status"] for row in bands)) != binding["band_coverage_counts"]:
        raise ValueError("Band coverage status counts differ")
    trajectory = _trajectory(ctx, run_map, bind)
    if not bound_references_valid(list(bindings.values()), ctx):
        raise ValueError("Supplement source/private binding rejected")
    return dict(gate="G8_INDEPENDENT_REPORT_SUPPLEMENTS", status="PASS_RECORDED_SCOPE", registered_unique_pairs=len(registered),
                checked_complete_pairs=complete, paired_status_counts=binding["paired_coverage_counts"],
                expected_bands=len(expected_bands), checked_complete_bands=checked_bands,
                band_status_counts=binding["band_coverage_counts"], uncovered=uncovered, trajectory=trajectory,
                checked_unique_bindings=len(bindings), model_calls=0, new_training=False)
