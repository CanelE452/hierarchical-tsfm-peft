import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1]/"reports/adapter_diagnostics_20261017_v2/code/verify_report_supplements.py"


def load():
    assert PATH.is_file(), "Independent supplement gate missing"
    spec = importlib.util.spec_from_file_location("supplement_validation_fixture", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader(); writer.writerows(rows)


def fixture(tmp_path):
    report, cache = tmp_path/"report", tmp_path/"cache"
    report.mkdir(); cache.mkdir()
    contract = {key: "a"*64 for key in ("scaler_sha256", "target_sha256", "mask_sha256", "channel_sha256", "origins_sha256")}
    sources, runs = [], []
    for name, mse, mae in (("ref", 1., 1.), ("method", 2., 1.5)):
        source = report/(name+".json"); source.write_text("{}")
        sources.append(dict(reference="report://"+source.name, sha256=digest(source)))
        runs.append(dict(run_id=name, method=name, cohort="jena", C=1, H=48, split="V_select", metric_scale="canonical_A60",
                         MSE=mse, MAE=mae, source_file="report/"+source.name, source_sha256=digest(source), **contract))
    write_csv(report/"RUNS_FULL_PRECISION.csv", runs)
    write_csv(report/"SAME_UPDATE_RESULTS.csv", [])
    pair = dict(reference_run_id="ref", method_run_id="method", cohort="jena")
    for name, rows in (("MATCHED_COMPARISONS.csv", [pair]), ("CONTEXTUAL_COMPARISONS.csv", []), ("SAME_UPDATE_COMPARISONS.csv", [])):
        write_csv(report/name, rows)
    origins = [dict(pair, origin_id=origin, reference_MSE=1., method_MSE=2., delta_MSE=1.,
                    reference_MAE=1., method_MAE=1.5, delta_MAE=.5, observed_channels=1, observed_count=48) for origin in (10, 20)]
    private = cache/"origin.csv"; write_csv(private, origins)
    coverage = dict(pair, status="COMPLETE", reason="", origin_count=2, mean_origin_delta_MSE=1.,
        mean_origin_delta_MAE=.5, min_origin_delta_MSE=1., max_origin_delta_MSE=1., min_origin_delta_MAE=.5,
        max_origin_delta_MAE=.5, global_delta_MSE=1., global_delta_MAE=.5,
        source_bindings=json.dumps(sources), contract_hashes=json.dumps({key:contract[key] for key in contract if key!="origins_sha256"}))
    write_csv(report/"PAIRED_ORIGIN_COVERAGE.csv", [coverage])
    horizon = [dict(run_id=run["run_id"], horizon=h, channel_id="0", SSE=2*run["MSE"], SAE=2*run["MAE"], count=2)
               for run in runs for h in range(1, 49)]
    write_csv(report/"ERROR_BY_HORIZON.csv", horizon)
    (report/"HORIZON_STATS_BINDINGS.json").write_text(json.dumps(dict(statistic_sha256=digest(report/"ERROR_BY_HORIZON.csv"),
        runs={run["run_id"]:dict(origins_sha256=contract["origins_sha256"], source_file=run["source_file"], source_sha256=run["source_sha256"]) for run in runs})))
    bands = [dict(pair, horizon_start=lo, horizon_end=hi, status="COMPLETE", reason="", reference_MSE=1.,method_MSE=2.,
        reference_MAE=1., method_MAE=1.5, delta_MSE=1.,delta_MAE=.5,delta_MSE_pct=100.,delta_MAE_pct=50.) for lo,hi in ((1,16),(17,32),(33,48))]
    write_csv(report/"HORIZON_BAND_SUMMARY.csv", bands)
    binding = dict(registered_pairs=1, paired_coverage_counts=dict(COMPLETE=1), band_coverage_counts=dict(COMPLETE=3),
        private_artifacts=[dict(reference="cache://origin.csv", sha256=digest(private))],
        public_outputs=[dict(reference="report://"+name, sha256=digest(report/name)) for name in ("PAIRED_ORIGIN_COVERAGE.csv","HORIZON_BAND_SUMMARY.csv")])
    (report/"PAIRED_ORIGIN_BINDINGS.json").write_text(json.dumps(binding))
    return dict(repo=tmp_path, report=report, cache=cache), binding, origins, bands


def rebind(ctx, binding):
    for row in binding["public_outputs"]:
        row["sha256"] = digest(ctx["report"]/row["reference"].split("://")[1])
    (ctx["report"]/"PAIRED_ORIGIN_BINDINGS.json").write_text(json.dumps(binding))


def test_independent_gate_accepts_exact_coverage_and_arithmetic(tmp_path):
    ctx, _, _, _ = fixture(tmp_path)
    value = load().verify_supplements(ctx)
    assert value["registered_unique_pairs"] == 1
    assert value["checked_complete_pairs"] == 1
    assert value["checked_complete_bands"] == 3


def test_missing_pair_cannot_be_hidden_by_refreshing_output_hash(tmp_path):
    ctx, binding, _, _ = fixture(tmp_path)
    write_csv(ctx["report"]/"PAIRED_ORIGIN_COVERAGE.csv", [])
    rebind(ctx,binding)
    with pytest.raises(ValueError, match="coverage"):
        load().verify_supplements(ctx)


def test_private_delta_arithmetic_is_checked_even_after_hash_refresh(tmp_path):
    ctx, binding, origins, _ = fixture(tmp_path)
    origins[0]["delta_MSE"] = 9.
    write_csv(ctx["cache"]/"origin.csv", origins)
    binding["private_artifacts"][0]["sha256"] = digest(ctx["cache"]/"origin.csv")
    rebind(ctx,binding)
    with pytest.raises(ValueError, match="arithmetic"):
        load().verify_supplements(ctx)


def test_band_delta_is_independently_recomputed_after_hash_refresh(tmp_path):
    ctx, binding, _, bands = fixture(tmp_path)
    bands[0]["delta_MSE"] = 3.
    write_csv(ctx["report"]/"HORIZON_BAND_SUMMARY.csv", bands)
    rebind(ctx,binding)
    with pytest.raises(ValueError, match="band"):
        load().verify_supplements(ctx)


def test_private_sufficient_statistic_mutation_rejects_old_binding(tmp_path):
    ctx, _, _, _ = fixture(tmp_path)
    with (ctx["cache"]/"origin.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="binding"):
        load().verify_supplements(ctx)
