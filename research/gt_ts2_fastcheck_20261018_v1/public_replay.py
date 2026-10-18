"""Verify public sufficient statistics and regenerate figures without NPZ data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import common as c
from . import reporting as r


INPUTS = ("RUNS.csv", "COMPARISONS.csv", "CHANNEL_HORIZON_STATS.csv", "COST.csv", "COST_TRIALS.csv")
IMAGES = tuple(f"FIG{index}_{name}.{suffix}" for index, name in ((1, "Lookback"), (2, "Adaptation"), (3, "Cost"))
               for suffix in ("png", "svg"))


def _cell(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def read_typed_csv(path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV columns are empty or duplicated")
        rows = [{key: _cell(value) for key, value in row.items()} for row in reader]
    r._public(rows)
    return rows


def _close(actual, expected, label):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if (actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, rtol=1e-12, atol=1e-12)):
        raise ValueError(label + " disagrees with independently aggregated public statistics")


def _integer(value, label, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(label + " must be an integer >= " + str(minimum))
    return value


def _runs(rows):
    index = {}
    for row in rows:
        run_id = row["run_id"]
        if run_id in index:
            raise ValueError("Duplicate public RUNS identity")
        index[run_id] = row
    expected = {row["run_id"] for row in r._expected()}
    if set(index) != expected:
        raise ValueError("Public RUNS must contain the 33 registered A0/A/CP0/B metrics")
    return index


def _metrics(run_index, channel_rows):
    grouped = {}
    for row in channel_rows:
        key = row["run_id"], row["channel"], row["horizon_band"]
        if key in grouped or row["run_id"] not in run_index:
            raise ValueError("Duplicate or unregistered public channel/horizon statistic")
        grouped[key] = row
    verified_channels = 0
    for run_id, run in run_index.items():
        channels = _integer(run["C"], "C", 1)
        expected = {(run_id, channel, band) for channel in range(channels)
                    for band in ("TOTAL", "1-16", "17-32", "33-48")}
        actual = {key for key in grouped if key[0] == run_id}
        if actual != expected:
            raise ValueError("Channel/horizon sufficient statistics are incomplete")
        sse, sae, counts = [], [], []
        for channel in range(channels):
            total = grouped[(run_id, channel, "TOTAL")]
            bands = [grouped[(run_id, channel, band)] for band in ("1-16", "17-32", "33-48")]
            for row in (total, *bands):
                count = _integer(row["count"], "Observed channel count", 1)
                values = np.asarray([row["sse"], row["sae"]], np.float64)
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError("Public SSE/SAE must be finite and nonnegative")
                _close(row["mse"], row["sse"] / count, "Channel MSE")
                _close(row["mae"], row["sae"] / count, "Channel MAE")
            if sum(row["count"] for row in bands) != total["count"]:
                raise ValueError("Disjoint horizon-band counts do not reconstruct TOTAL")
            for key in ("sse", "sae"):
                _close(sum(row[key] for row in bands), total[key], "Horizon-band " + key)
            sse.append(total["sse"])
            sae.append(total["sae"])
            counts.append(total["count"])
        sse, sae, counts = map(lambda value: np.asarray(value, np.float64), (sse, sae, counts))
        _close(float(np.mean(sse / counts)), run["MSE"], run_id + " MSE")
        _close(float(np.mean(sae / counts)), run["MAE"], run_id + " MAE")
        for key, value in (("channel_sse", sse), ("channel_sae", sae), ("channel_count", counts),
                           ("channel_mse", sse / counts), ("channel_mae", sae / counts)):
            _close(value, run[key], run_id + " " + key)
        verified_channels += channels
    return dict(runs_verified=len(run_index), channels_verified=verified_channels,
                channel_horizon_rows_verified=len(channel_rows), reduction="FP64 SSE/count per fixed channel, then equal-channel mean")


def _comparisons(run_index, rows):
    expected = set(r._comparisons())
    actual = {(row["comparison"], row["seed"], row["candidate"], row["reference"]) for row in rows}
    if actual != expected or len(actual) != len(rows):
        raise ValueError("Public paired comparison set is incomplete or duplicated")
    for row in rows:
        candidate, reference = run_index[row["candidate"]], run_index[row["reference"]]
        for key in ("target", "mask", "origins", "mean", "std"):
            field = key + "_sha256"
            if candidate[field] != reference[field] or row["cohort_hashes"][key] != candidate[field]:
                raise ValueError("Published paired cohort fingerprints differ")
        for metric in ("mse", "mae"):
            cand, ref = candidate[metric.upper()], reference[metric.upper()]
            if ref <= 0:
                raise ValueError("Published paired reference error must be positive")
            _close(row["candidate_" + metric], cand, "Paired candidate " + metric)
            _close(row["reference_" + metric], ref, "Paired reference " + metric)
            _close(row[metric + "_delta"], cand - ref, "Paired " + metric + " delta")
            _close(row[metric + "_relative_percent"], (cand - ref) / ref * 100., "Paired " + metric + " relative percent")
    return dict(pairs_verified=len(rows), cohort_scope="Equality of published fingerprints; no private target/mask arrays opened")


def _costs(run_index, summaries, trial_rows):
    expected = {(run_id, batch) for run_id, row in run_index.items() if row["phase"] in ("A", "B") for batch in (1, 8)}
    indexed = {}
    for row in summaries:
        key = row["run_id"], row["batch"]
        if key in indexed:
            raise ValueError("Duplicate public run/batch cost summary")
        indexed[key] = row
    if set(indexed) != expected or len(expected) != 42:
        raise ValueError("Public cost summaries must contain 21 routes x batch1/8")
    trials = {key: [] for key in indexed}
    for row in trial_rows:
        key = row["run_id"], row["batch"]
        if key not in trials:
            raise ValueError("Unregistered public cost trial")
        trials[key].append(row)
    for (run_id, batch), summary in indexed.items():
        rows = trials[(run_id, batch)]
        timed = [row for row in rows if row["stage"] == "timed"]
        warmup = [row for row in rows if row["stage"] == "warmup"]
        if len(rows) != 30 or len(timed) != 20 or len(warmup) != 10:
            raise ValueError("Cost requires exactly 10 warmup and 20 timed public trials")
        groups = {}
        for stage, selected in (("warmup", warmup), ("timed", timed)):
            if sorted(row["iteration"] for row in selected) != list(range(len(selected))):
                raise ValueError("Cost trial iterations are incomplete or duplicated")
            for row in selected:
                group = row["iteration"] % 3
                if row["group_index"] != group:
                    raise ValueError("Cost repetitions must cycle the three fixed origin groups")
                start = _integer(row["groupstart"], "Cost group start")
                if start + batch > run_index[run_id]["origin_count"]:
                    raise ValueError("Cost origin group falls outside the public forecast cohort")
                identity = start, row["origins_sha256"]
                if group in groups and groups[group] != identity:
                    raise ValueError("A fixed cost origin group changed between repetitions")
                groups[group] = identity
                if row.get("TS2cache_used") is not False:
                    raise ValueError("Whole-route timing must use online TS2 rather than feature cache")
        if len(groups) != 3 or len(set(groups.values())) != 3:
            raise ValueError("Cost requires three distinct fixed origin groups")
        values = np.asarray([row["whole_batch_ms"] for row in timed], np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Timed whole-route values must be finite and positive")
        q25, median, q75, p95 = np.quantile(values, [.25, .5, .75, .95], method="linear")
        aggregate = dict(p50_ms=median, q25_ms=q25, q75_ms=q75, iqr_ms=q75 - q25, p95_ms=p95,
                         repetitions=20, origins_per_second_at_median=1000 * batch / median)
        for key in ("peak_allocated_MiB", "peak_reserved_MiB", "RSS_MiB"):
            numbers = np.asarray([row[key] for row in timed], np.float64)
            if not np.isfinite(numbers).all() or np.any(numbers < 0):
                raise ValueError("Cost memory values must be finite and nonnegative")
            aggregate[key] = numbers.max()
        for key, value in aggregate.items():
            _close(summary[key], value, run_id + " cost " + key)
        _close(summary["MSE"], run_index[run_id]["MSE"], "Cost-bound MSE")
        _close(summary["MAE"], run_index[run_id]["MAE"], "Cost-bound MAE")
    return dict(run_batch_pairs_verified=len(summaries), timed_trials_verified=20 * len(summaries),
                warmup_trials_verified=10 * len(summaries), fixed_groups_per_pair=3,
                quantile_scope="Empirical fixed-group repetitions; not service-tail guarantees")


def replay(source_report, output_dir):
    source, output = Path(source_report).resolve(), Path(output_dir).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Replay output must be separate from the public source directory")
    if output.exists():
        raise FileExistsError("Public replay requires a fresh output directory")
    tables = {name: read_typed_csv(source / name) for name in INPUTS}
    check = dict(status="PASS", private_arrays_read=False,
                 scope="Public CSV sufficient statistics and fresh PNG/SVG regeneration only; no NPZ/cache access",
                 input_csv_sha256={name: c.file_sha(source / name) for name in INPUTS},
                 metrics={}, comparisons={}, costs={}, images={}, issues=[])
    try:
        run_index = _runs(tables["RUNS.csv"])
    except (ValueError, KeyError, TypeError) as error:
        run_index = {}
        check["issues"].append("RUNS: " + str(error))
    if run_index:
        for name, function, args in (
                ("metrics", _metrics, (run_index, tables["CHANNEL_HORIZON_STATS.csv"])),
                ("comparisons", _comparisons, (run_index, tables["COMPARISONS.csv"])),
                ("costs", _costs, (run_index, tables["COST.csv"], tables["COST_TRIALS.csv"]))):
            try:
                check[name] = function(*args)
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                check["issues"].append(name + ": " + str(error))
    output.mkdir(parents=True, exist_ok=False)
    try:
        figures = r._figure_bytes(tables["RUNS.csv"], tables["COMPARISONS.csv"], tables["COST.csv"])
        for name in IMAGES:
            with (output / name).open("xb") as stream:
                stream.write(figures[name])
            actual = c.file_sha(output / name)
            expected = c.file_sha(source / name)
            check["images"][name] = dict(actual_sha256=actual, expected_sha256=expected, matches=actual == expected)
            if actual != expected:
                check["issues"].append("image hash differs: " + name)
    except (ValueError, KeyError, TypeError, OSError) as error:
        check["issues"].append("figure regeneration: " + str(error))
    if check["issues"]:
        check["status"] = "FAIL"
    with (output / "REPLAY_CHECK.json").open("xb") as stream:
        stream.write(c.json_bytes(check))
    return check


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = replay(args.source_report, args.output_dir)
    print(c.json_bytes(result).decode(), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
