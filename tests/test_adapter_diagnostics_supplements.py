"""Small CPU fixtures for saved-statistic-only report supplements."""
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.adapter_diagnostics_20261017_v2.core import atomic_json, read, sha


PATH = Path(__file__).resolve().parents[1]/"reports/adapter_diagnostics_20261017_v2/code/report_supplements.py"
SPEC = importlib.util.spec_from_file_location("diagnostic_report_supplements", PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def contract():
    return dict(cohort="jena", split="V_select", metric_scale="canonical_A60", C=2, H=48, L=512,
                **{field: "a"*64 for field in helper.CONTRACT_FIELDS})


def fixture(tmp_path):
    report, cache = tmp_path/"report", tmp_path/"cache"
    report.mkdir(); cache.mkdir()
    origins = np.array([20, 10], np.int64)
    count = np.array([[1, 2], [3, 1]])
    runs = []
    for name, sse, sae in (("ref", np.array([[1., 8.], [9., 1.]]), np.array([[1., 4.], [3., 1.]])),
                           ("method", np.array([[4., 2.], [3., 4.]]), np.array([[2., 2.], [3., 2.]]))):
        path = report/(name+".json")
        atomic_json(path, dict(origin_sse=sse.tolist(), origin_sae=sae.tolist(), origin_counts=count.tolist(), origins=origins.tolist()))
        runs.append(dict(contract(), run_id=name, method=name, origins_sha256=helper._hash(origins),
                         MSE=float(np.mean(sse.sum(0)/count.sum(0))), MAE=float(np.mean(sae.sum(0)/count.sum(0))),
                         source_file=path.relative_to(tmp_path).as_posix(), source_sha256=sha(path)))
    pair = dict(reference_run_id="ref", method_run_id="method", cohort="jena", seed=1, match_status="EXACT_MATCHED")
    horizon = []
    for run in runs:
        for h in range(1, 49):
            for channel, n, error in ((0, 1, 1.), (1, 3, 3.)):
                horizon.append(dict(run_id=run["run_id"], horizon=h, channel_id=str(channel),
                                    SSE=n*error*(2 if run["run_id"] == "method" else 1), SAE=n, count=n))
    return dict(repo=tmp_path, report=report, cache=cache), dict(runs=runs, endpoints=[], horizon=horizon), pair


def test_supplements_origin_ids_sorted_macro_is_not_global_and_private_csv_is_bound(tmp_path):
    ctx, data, pair = fixture(tmp_path)
    result = helper.build_supplements(ctx, data, [pair, pair])
    assert result["coverage_counts"] == {"COMPLETE": 1}
    coverage = result["paired_coverage"][0]
    assert coverage["mean_origin_delta_MSE"] != pytest.approx(coverage["global_delta_MSE"])
    assert [row["origin_id"] for row in helper.csv.DictReader(result["private_files"][0].open())] == ["10", "20"]
    binding = read(ctx["report"]/"PAIRED_ORIGIN_BINDINGS.json")
    assert binding["private_artifacts"][0]["sha256"] == sha(result["private_files"][0])
    assert "generally differs" in binding["statistic_scope"]
    assert all(path.is_relative_to(ctx["report"]) for path in result["public_files"])


def test_bands_sum_counts_per_channel_before_macro_instead_of_micro(tmp_path):
    _, data, pair = fixture(tmp_path)
    rows = helper.horizon_band_rows(data, [pair])
    assert len(rows) == 3
    assert all(row["status"] == "COMPLETE" for row in rows)
    assert rows[0]["reference_MSE"] == 2.
    assert rows[0]["reference_MSE"] != 2.5
    assert rows[0]["method_MSE"] == 4.
    assert rows[0]["delta_MSE_pct"] == 100.
    assert rows[0]["reference_channel_statistics"][1]["count"] == 48


@pytest.mark.parametrize("field", helper.CONTRACT_FIELDS)
def test_pair_contract_difference_is_covered_without_values(tmp_path, field):
    ctx, data, pair = fixture(tmp_path)
    data["runs"][1][field] = "b"*64
    result = helper.build_supplements(ctx, data, [pair])
    assert result["paired_coverage"][0]["status"] == "CONTRACT_MISMATCH"
    assert list(helper.csv.DictReader(result["private_files"][0].open())) == []


def test_bound_public_statistic_mutation_and_missing_source_are_not_complete(tmp_path):
    ctx, data, pair = fixture(tmp_path)
    (ctx["report"]/"method.json").write_text("{}")
    result = helper.build_supplements(ctx, data, [pair])
    assert result["paired_coverage"][0]["status"] == "CONTRACT_MISMATCH"
    (ctx["report"]/"method.json").unlink()
    result = helper.build_supplements(ctx, data, [pair])
    assert result["paired_coverage"][0]["status"] == "MISSING_BOUND_STATISTICS"


def test_pair_observed_counts_and_ids_must_match(tmp_path):
    ctx, data, _ = fixture(tmp_path)
    a, b = data["runs"]
    left, right = helper._origin_stats(a, ctx, data), helper._origin_stats(b, ctx, data)
    right["count"][0, 0] += 1
    with pytest.raises(ValueError, match="observed counts"):
        helper.paired_origin_rows(a, b, left, right)
    right["count"] = left["count"].copy(); right["origins"][0] += 1
    with pytest.raises(ValueError, match="ID mismatch"):
        helper.paired_origin_rows(a, b, left, right)


def test_missing_horizon_and_different_counts_are_explicit(tmp_path):
    _, data, pair = fixture(tmp_path)
    data["horizon"].pop(0)
    rows = helper.horizon_band_rows(data, [pair])
    assert rows[0]["status"] == "MISSING_SUFFICIENT_STATISTICS"
    assert rows[1]["status"] == "COMPLETE"
    next(row for row in data["horizon"] if row["run_id"] == "method" and row["horizon"] == 17)["count"] += 1
    assert helper.horizon_band_rows(data, [pair])[1]["status"] == "CONTRACT_MISMATCH"


def test_producer_contract_is_checked_against_run_metadata(tmp_path):
    ctx, data, pair = fixture(tmp_path)
    source = ctx["report"]/"method.json"
    value = read(source); value["scaler_sha256"] = "b"*64
    atomic_json(source, value)
    data["runs"][1]["source_sha256"] = sha(source)
    result = helper.build_supplements(ctx, data, [pair])
    assert result["paired_coverage"][0]["status"] == "CONTRACT_MISMATCH"
    assert "Producer/run contract" in result["paired_coverage"][0]["reason"]


def test_archive_fallback_checks_bytes_and_evaluation_fingerprints(tmp_path):
    ctx, data, _ = fixture(tmp_path)
    run = data["runs"][0]
    prediction = np.ones((2, 48, 2), np.float32)
    target = np.zeros_like(prediction)
    mask = np.ones_like(prediction, dtype=bool); mask[0, 1:, 0] = False
    origins = np.array([20, 10], np.int64)
    channels = np.array(["a", "b"])
    archive = ctx["cache"]/"prediction.npz"
    np.savez(archive, pred_normalized=prediction, target_normalized=target, observed=mask, origins=origins, channel_ids=channels)
    source = ctx["report"]/"ref.json"
    atomic_json(source, {})
    run.update(source_sha256=sha(source), MSE=1., MAE=1., _archive_path=archive,
               independent_private_score_sha256=sha(archive), target_sha256=helper._hash(target),
               mask_sha256=helper._hash(mask), channel_sha256=helper._hash(channels))
    value = helper._origin_stats(run, ctx, data)
    assert value["origins"].tolist() == [10, 20]
    assert value["count"].tolist() == [[48, 48], [1, 48]]
    run["mask_sha256"] = "b"*64
    with pytest.raises(ValueError, match="fingerprint"):
        helper._origin_stats(run, ctx, data)
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA mismatch"):
        helper._origin_stats(run, ctx, data)


def test_origin_zero_observation_channel_is_excluded_from_that_origins_macro(tmp_path):
    _, data, _ = fixture(tmp_path)
    a, b = data["runs"]
    common = dict(origins=np.array([10]), count=np.array([[0, 2]]), sources=[])
    left = dict(common, sse=np.array([[0., 6.]]), sae=np.array([[0., 4.]]))
    right = dict(common, sse=np.array([[0., 2.]]), sae=np.array([[0., 2.]]))
    row = helper.paired_origin_rows(a, b, left, right)[0]
    assert row["observed_channels"] == 1
    assert row["delta_MSE"] == -2.
    assert row["delta_MAE"] == -1.


def producer_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def archive_fixture(tmp_path, b02=False, method="OWN_CHANNEL_32BLOCK_RIDGE"):
    ctx, data, _ = fixture(tmp_path)
    run = data["runs"][0]
    target = np.zeros((2, 48, 2), np.float64 if b02 else np.float32)
    mask = np.ones_like(target, dtype=bool); mask[0, 1:, 0] = False
    origins = np.array([20, 10], np.int64)
    channels = np.array(["a", "b"])
    prediction = np.ones_like(target)*(1 if method.startswith("OWN") else 2)
    values = dict(pred_normalized=prediction, target_normalized=target, observed=mask,
                  origins=origins, channel_ids=channels)
    archive = ctx["cache"]/"prediction.npz"
    if b02:
        np.savez(archive, own_prediction=np.ones_like(target), cross_prediction=np.ones_like(target)*2,
                 target=target, observed=mask, origins=origins)
        contract_path = tmp_path/"research/informs_broad_validation_20261015_v1/DATA_CONTRACT.json"
        atomic_json(contract_path, dict(assets=dict(jena=dict(columns=channels.tolist()))))
        writer = tmp_path/"research/adapter_diagnostics_20261017_v2/ridge_jobs.py"
        writer.parent.mkdir(parents=True, exist_ok=True); writer.write_text("# bound B02 writer fixture\n")
        data["bindings"] = {path.relative_to(tmp_path).as_posix(): sha(path) for path in (contract_path, writer)}
        run["channel_order_validation"] = "DERIVED_REGISTERED_ORDER_FROM_BOUND_B02_WRITER"
        run["method"] = method
    else:
        np.savez(archive, **values)
    source = ctx["report"]/"ref.json"
    record = dict(scaler_sha256=run["scaler_sha256"], target_sha256=producer_hash(target),
                  observed_sha256=producer_hash(mask), origins_sha256=producer_hash(origins),
                  channel_order_sha256=producer_hash(channels))
    if b02:
        record.pop("target_sha256"); record.pop("observed_sha256")
        record["private_artifacts"] = [dict(reference="cache://prediction.npz", sha256=sha(archive))]
    else:
        record["prediction_sha256"] = sha(archive)
    atomic_json(source, record)
    run.update(source_sha256=sha(source), MSE=float(prediction[0, 0, 0]**2), MAE=float(prediction[0, 0, 0]),
               _archive_path=archive, independent_private_score_sha256=sha(archive),
               target_sha256=helper._hash(target), mask_sha256=helper._hash(mask),
               origins_sha256=helper._hash(origins), channel_sha256=helper._hash(channels))
    arrays = {key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=helper._hash(value))
              for key, value in values.items()}
    data["new_stats_replay"] = [dict(run_id=run["run_id"], private_archive_sha256=sha(archive), arrays=arrays)]
    return ctx, data, run, source, archive, values


def test_bound_archive_precedes_encoded_producer_hashes_and_preserves_both_hash_schemes(tmp_path):
    ctx, data, run, source, _, values = archive_fixture(tmp_path)
    record = read(source)
    record.update(origin_sse=[[0., 0.]], origin_sae=[[0., 0.]], origin_counts=[[0, 0]])
    atomic_json(source, record); run["source_sha256"] = sha(source)
    stats = helper._origin_stats(run, ctx, data)
    assert stats["origins"].tolist() == [10, 20]
    assert stats["count"].tolist() == [[48, 48], [1, 48]]
    assert stats["array_fingerprints"]["target_normalized"]["sha256"] == helper._hash(values["target_normalized"])
    checked = stats["producer_fingerprint_validation"]
    assert {item["field"] for item in checked} >= {"target_sha256", "observed_sha256", "origins_sha256", "channel_order_sha256"}
    assert all(item["scheme"] == "dtype_shape_and_bytes" for item in checked)


@pytest.mark.parametrize("method,expected", [("OWN_CHANNEL_32BLOCK_RIDGE", 1.), ("ALL_CHANNEL_32BLOCK_RIDGE", 4.)])
def test_b02_saved_schema_selects_own_or_cross_and_verifies_derived_registered_channels(tmp_path, method, expected):
    ctx, data, run, _, _, _ = archive_fixture(tmp_path, b02=True, method=method)
    stats = helper._origin_stats(run, ctx, data)
    assert np.mean(stats["sse"].sum(0)/stats["count"].sum(0)) == expected
    assert len(stats["sources"]) == 4
    assert {item["field"] for item in stats["producer_fingerprint_validation"]} == {"origins_sha256", "channel_order_sha256"}


@pytest.mark.parametrize("field", ["target_sha256", "observed_sha256", "origins_sha256", "channel_order_sha256"])
def test_encoded_producer_hash_cannot_be_ignored_by_independent_archive_path(tmp_path, field):
    ctx, data, run, source, _, _ = archive_fixture(tmp_path)
    record = read(source); record[field] = "b"*64
    atomic_json(source, record); run["source_sha256"] = sha(source)
    with pytest.raises(ValueError, match="Producer.*fingerprint"):
        helper._origin_stats(run, ctx, data)


def test_bound_archive_replay_prediction_hash_rejects_same_score_mutation(tmp_path):
    ctx, data, run, source, archive, values = archive_fixture(tmp_path)
    values["pred_normalized"] *= -1
    np.savez(archive, **values)
    run["independent_private_score_sha256"] = sha(archive)
    data["new_stats_replay"][0]["private_archive_sha256"] = sha(archive)
    record = read(source); record["prediction_sha256"] = sha(archive)
    atomic_json(source, record); run["source_sha256"] = sha(source)
    with pytest.raises(ValueError, match="replay.*fingerprint"):
        helper._origin_stats(run, ctx, data)


def test_strong_archive_statistics_must_reproduce_published_global_fp64_score(tmp_path):
    ctx, data, run, _, _, _ = archive_fixture(tmp_path)
    run["MSE"] += .01
    with pytest.raises(ValueError, match="global FP64 score"):
        helper._origin_stats(run, ctx, data)


@pytest.mark.parametrize("mutation", ["archive_bytes", "canonical_target", "producer_shape", "scaler"])
def test_strong_archive_path_rejects_changed_binding_arrays_and_producer_contract(tmp_path, mutation):
    ctx, data, run, source, archive, values = archive_fixture(tmp_path)
    if mutation == "archive_bytes":
        archive.write_bytes(b"modified")
    elif mutation == "canonical_target":
        values["target_normalized"][0, 0, 0] = 2
        np.savez(archive, **values)
        run["independent_private_score_sha256"] = sha(archive)
    else:
        record = read(source)
        if mutation == "producer_shape":
            record["array_fingerprints"] = {"target_normalized": dict(shape=[1, 96, 2], dtype="float32", sha256=producer_hash(values["target_normalized"]))}
        else:
            record["scaler_sha256"] = "b"*64
        atomic_json(source, record); run["source_sha256"] = sha(source)
    with pytest.raises(ValueError):
        helper._origin_stats(run, ctx, data)


@pytest.mark.parametrize("missing", ["explicit_derivation", "writer_binding", "contract_binding"])
def test_b02_no_channel_member_requires_explicit_bound_order_evidence(tmp_path, missing):
    ctx, data, run, _, _, _ = archive_fixture(tmp_path, b02=True)
    if missing == "explicit_derivation":
        run["channel_order_validation"] = "EXACT_REGISTERED_ORDER"
    else:
        suffix = "ridge_jobs.py" if missing == "writer_binding" else "DATA_CONTRACT.json"
        data["bindings"] = {key: value for key, value in data["bindings"].items() if not key.endswith(suffix)}
    with pytest.raises(ValueError, match="channel|order"):
        helper._origin_stats(run, ctx, data)
