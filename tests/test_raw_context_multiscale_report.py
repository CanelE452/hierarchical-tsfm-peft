"""Independent audit and deterministic public replay on synthetic CPU fixtures."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.raw_context_multiscale_20261018_v1 import common, report


def test_independent_score_weights_observed_channels_equally_and_ignores_missing_targets(monkeypatch):
    monkeypatch.setattr(common, "statistics", lambda *a, **k: (_ for _ in ()).throw(AssertionError("producer called")))
    prediction = np.ones((2, 2, 2), np.float32)
    prediction[:, :, 1] = 10
    mask = np.ones(prediction.shape, bool)
    mask[:, :, 1] = False
    mask[0, 0, 1] = True
    target = np.zeros_like(prediction)
    target[~mask] = np.nan
    actual = report.independent_statistics(prediction, target, mask)
    assert actual["mse"] == 50.5
    assert actual["mae"] == 5.5
    assert actual["n_observed"] == 5
    assert actual["channel_counts"] == [4, 1]
    assert sum(map(sum, actual["horizon_channel_sse"])) == 104
    assert actual["mse"] != 104/5
    with pytest.raises(ValueError, match="bool"):
        report.independent_statistics(prediction, target, mask.astype(np.uint8))


def test_earliest_tie_includes_cp0_and_rejects_missing_or_duplicate_cp0():
    curve = [dict(epoch=0, astop_mse=1), dict(epoch=.25, astop_mse=1), dict(epoch=1, astop_mse=1.1)]
    assert report.earliest_selection(curve) == (0., 1.)
    with pytest.raises(ValueError, match="CP0"):
        report.earliest_selection(curve[1:])
    with pytest.raises(ValueError, match="unique"):
        report.earliest_selection(curve+[curve[0]])


def test_independent_alpha_uses_macro_channel_weights_and_zero_denominator_rule():
    g = np.zeros((2, 2, 2), np.float32)
    b = np.ones_like(g)
    target = np.zeros_like(g)
    target[:, :, 0] = 1
    mask = np.ones(g.shape, bool)
    mask[:, :, 1] = False
    mask[0, 0, 1] = True
    target[~mask] = np.nan
    assert report.independent_alpha(g, b, target, mask) == .5
    assert report.independent_alpha(g, g, target, mask) == 0


def test_canonical_past_fills_each_window_from_its_own_left_edge():
    task = dict(L=4, H=2, columns=["x"], physical=np.arange(8, dtype=np.float32)[:, None],
                observed=np.ones((8, 1), bool))
    task["observed"][1:3] = False
    task["physical"][1:3] = np.nan
    past = report.canonical_past(task, np.array([4, 5]), np.array([10.], np.float32), np.ones(1, np.float32))
    np.testing.assert_array_equal(past[:, :, 0], [[-10., -10., -10., -7.], [0., 0., -7., -6.]])


def test_saved_head_checkpoints_are_independently_rescored_on_split_only_batches():
    from research.raw_context_multiscale_20261018_v1 import calibration
    origins = np.arange(20, 69, dtype=np.int64)
    rng = np.random.default_rng(419)
    past = rng.normal(size=(len(origins), 8, 2)).astype(np.float32)
    g = np.repeat(past[:, -1:], 4, axis=1)
    b, target = g+.3, g+.2
    mask = np.ones(g.shape, bool)
    mask[::3, :, 1] = False
    task = dict(L=8, H=4, columns=["x", "y"], split_ranges=dict(M_cal=(20, 72)))
    checkpoints = []
    search = calibration.head_search(past, g, b, target, mask, origins, task, seed=92601,
        checkpoint_sink=lambda state, curve:checkpoints.append(dict(state=state, curve=curve)))
    saved = dict(search=search, contract=dict(control=False), search_checkpoints=checkpoints)
    report.verify_head_checkpoints(saved, past, g, b, target, mask, origins, task)
    bad = copy.deepcopy(saved)
    bad["search"]["curves"][1]["M_select_mse"] += .01
    bad["search_checkpoints"][1]["curve"]["M_select_mse"] += .01
    with pytest.raises(ValueError, match="independent M_select"):
        report.verify_head_checkpoints(bad, past, g, b, target, mask, origins, task)


def test_missing_seals_never_open_task_or_predictions(tmp_path, monkeypatch):
    ctx = dict(report=tmp_path/"report", cache=tmp_path/"cache")
    ctx["report"].mkdir()
    monkeypatch.setattr(common, "load_task", lambda *a: (_ for _ in ()).throw(AssertionError("task opened")))
    with pytest.raises(ValueError, match="before prediction access"):
        report.verify(ctx)
    assert common.read(ctx["report"]/"VALIDATION_REPORT.json")["status"] == "FAIL"


def fixture_cost():
    origins = np.array([100, 148], np.int64)
    arm = dict(run_id="B_jena_s92601", dataset="jena", mode="B",
        score=dict(mse=1., mae=.5, n_origins=2), array_identities=dict(origins=common.array_identity(origins)))
    invocations = []
    for length, used in ((512, 48), (256, 24)):
        invocations.append(dict(input_shape=[1, length, 6], requested_context=length, actual_context=length,
            wrapper_horizon=48, used_horizon=used, actual_head_horizon=64,
            F_calls=1, F_physical_streams=6, F_logical_streams=6))
    trials = [dict(phase=phase, trial=i, whole_seconds=(i+1)/1000, F_invocations=copy.deepcopy(invocations),
                   F_calls=2, F_physical_streams=12, F_logical_streams=12,
                   peak_allocated_MiB=200. if phase == "timed" else None,
                   peak_reserved_MiB=250. if phase == "timed" else None)
              for phase, count in (("warmup", 10), ("timed", 20)) for i in range(count)]
    row = dict(cost_valid=True, fresh_process_cli=True, process_pid=9, process_create_time=100.,
        accuracy_run_id=arm["run_id"], dataset="jena", batch_size=1, H=48, schedule="native",
        p50_batch_ms=10.5, median_batch_ms=10.5, IQR_batch_ms=9.5, p95_batch_ms=19.05,
        median_ms_per_origin=10.5, p95_ms_per_origin=19.05, MSE=1., MAE=.5,
        peak_allocated_MiB=200., peak_reserved_MiB=250., process_GPU_sampling_status="UNAVAILABLE",
        process_GPU_sampled_peak_MiB=None, device="cuda", comparison_scope="GPU")
    parity = dict(status="PASS", saved_reference="PASS", n_origins=2,
        origins=arm["array_identities"]["origins"], atol=common.ATOL, rtol=common.RTOL)
    return dict(status="COMPLETE", row=row, trials=trials, parity=parity), arm


def test_independent_cost_statistics_and_f_lengths_tampering():
    receipt, arm = fixture_cost()
    assert report.verify_cost(receipt, arm)["median_batch_ms"] == 10.5
    for mutation in ("median", "p95", "calls", "length", "pid_memory", "saved_reference"):
        bad = copy.deepcopy(receipt)
        if mutation == "median": bad["row"]["median_batch_ms"] = 11
        elif mutation == "p95": bad["row"]["p95_batch_ms"] = 20
        elif mutation == "calls": bad["trials"][0]["F_calls"] = 1
        elif mutation == "length": bad["trials"][0]["F_invocations"][1]["requested_context"] = 512
        elif mutation == "pid_memory": bad["row"]["process_GPU_sampled_peak_MiB"] = 1000
        else: bad["parity"]["saved_reference"] = "NOT_PROVIDED"
        with pytest.raises(ValueError):
            report.verify_cost(bad, arm)


def test_fresh_process_identity_allows_pid_reuse_only_at_different_creation_time():
    receipt, arm = fixture_cost()
    identities = set()
    report.verify_cost(receipt, arm, pid_identities=identities)
    with pytest.raises(ValueError, match="distinct PID"):
        report.verify_cost(receipt, arm, pid_identities=identities)
    receipt["row"]["process_create_time"] += 1
    report.verify_cost(receipt, arm, pid_identities=identities)


def test_raw_cost_csv_sha_and_all_values_are_bound(tmp_path):
    receipt, arm = fixture_cost()
    path = tmp_path/"trials.csv"
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, list(receipt["trials"][0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in receipt["trials"])
    receipt["trials_sha256"] = common.sha(path)
    report.verify_cost(receipt, arm, trial_csv=path)
    path.write_text(path.read_text().replace("0.001", "0.002", 1), encoding="utf8")
    with pytest.raises(ValueError, match="CSV SHA"):
        report.verify_cost(receipt, arm, trial_csv=path)


def test_canonical_prediction_audit_rejects_rebound_target_mask_and_origin_tampering(tmp_path):
    physical = np.arange(40, dtype=np.float32).reshape(20, 2)/10
    mask = np.ones((20, 2), bool)
    mask[15, 1] = False
    physical[15, 1] = np.nan
    task = dict(L=4, H=2, columns=["x", "y"], physical=physical, observed=mask,
                data_sha256="d"*64, split_ranges=dict(M_cal=(10, 12), V_select=(12, 18)))
    origins = np.arange(12, 17, 2, dtype=np.int64)
    rows = origins[:, None]+np.arange(2)[None]
    target = np.where(mask[rows], physical[rows], 0).astype(np.float32)
    prediction = np.zeros_like(target)
    route = dict(family="RAW", canonical_fit_end=10)
    metadata = dict(route_spec=route, data_sha256=task["data_sha256"], split="V_select",
        scaler_sha256="s"*64, parent_fit_end=10, T_access=False,
        forecast_source_sha256=common.sha(Path(report.__file__).with_name("pipeline.py")))
    ctx = dict(cache=tmp_path, report=tmp_path, repo=tmp_path, asset=tmp_path,
        modules=dict(data=SimpleNamespace(fit_statistics=lambda *a:(np.zeros(2, np.float32), np.ones(2, np.float32), None)),
                     fit=SimpleNamespace(scaler_fingerprint=lambda *a:"s"*64)))
    path = tmp_path/"prediction.npz"
    evidence = common.save_prediction(path, prediction=prediction, target=target, observed=mask[rows], origins=origins,
                                      task=task, metadata=metadata)
    arm = dict(route_spec=route, prediction_binding=common.binding(path, ctx), score=evidence["score"], array_identities=evidence["arrays"])
    assert report._verify_prediction(ctx, task, arm)["n_observed"] == 11
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key:archive[key] for key in archive.files}
    for key in ("target_normalized", "observed", "origins"):
        changed = copy.deepcopy(arrays)
        if key == "target_normalized": changed[key][0, 0, 0] += 1
        elif key == "observed": changed[key] = changed[key].astype(np.uint8)
        else: changed[key] = changed[key][::-1]
        tampered = tmp_path/(key+".npz")
        np.savez_compressed(tampered, **changed)
        rebound = dict(arm, prediction_binding=common.binding(tampered, ctx))
        with pytest.raises(ValueError):
            report._verify_prediction(ctx, task, rebound)


def synthetic_data():
    arms, stats, costs = [], {}, []
    rng = np.random.default_rng(41)
    for dataset in ("jena", "pems32"):
        modes = ("S", "A", "B", "Raw", "G", "Ridge") if dataset == "jena" else ("S", "A", "Raw", "G")
        for mode in modes:
            for seed in (common.SEEDS if mode in ("S", "A", "B") else (None,)):
                identifier = f"{mode}_{dataset}_{seed}"
                prediction = rng.normal(size=(3, 6, 3)).astype(np.float32)
                target = np.zeros_like(prediction)
                mask = np.ones(prediction.shape, bool)
                mask[0, 0, 1] = False
                stats[identifier] = report.independent_statistics(prediction, target, mask)
                curve = [dict(epoch=0., astop_mse=1.3, updates=0, exposures=0),
                         dict(epoch=1., astop_mse=1.2, updates=10, exposures=160)]
                arm = dict(run_id=identifier, dataset=dataset, mode=mode, seed=seed, score=stats[identifier], reused=mode == "S",
                    curves=curve if mode in ("S", "A", "B") else [],
                    search_summary=dict(selected_epoch=1., selected_astop_mse=1.2) if mode in ("S", "A", "B") else {})
                arms.append(arm)
                for batch in (1, 8):
                    for schedule in (("native", "two_call") if mode == "A" else ("native",)):
                        value = .5 if mode == "G" else 2.
                        costs.append(dict(cost_id=identifier+schedule+str(batch), accuracy_run_id=identifier, dataset=dataset,
                            device="cpu" if mode == "Ridge" else "cuda", memory_layer="cpu_RSS" if mode == "Ridge" else "pytorch_allocated_tensor",
                            batch_size=batch, schedule=schedule, MSE=stats[identifier]["mse"], MAE=stats[identifier]["mae"],
                            median_batch_ms=value, peak_allocated_MiB=None if mode == "Ridge" else 200.,
                            peak_reserved_MiB=None if mode == "Ridge" else 250., resident_parameters=10,
                            trainable_parameters=5, F_calls=0 if mode in ("G", "Ridge") else 2 if mode == "B" or schedule == "two_call" else 1))
    return dict(arms=arms, stats=stats, costs=costs, trials=[], selection=dict(resume_queue=[]))


def full_coverage_tables():
    tables = report._public_tables(synthetic_data())
    model_template = next(row for row in tables["RUNS"] if row["mode"] == "Raw" and row["dataset"] == "jena")
    cost_template = next(row for row in tables["COST"] if row["accuracy_run_id"] == model_template["run_id"])
    def add(dataset, mode, seed, *, role=None, label=None, alpha=None):
        identifier = f"{mode}_{dataset}_{role}_{seed}_{label}"
        row = dict(model_template, run_id=identifier, dataset=dataset, mode=mode, seed=seed,
                   base_role=role, alpha_label=label, alpha=alpha)
        tables["RUNS"].append(row)
        for batch in (1, 8):
            tables["COST"].append(dict(cost_template, cost_id=identifier+"_native_B"+str(batch),
                accuracy_run_id=identifier, dataset=dataset, batch_size=batch, schedule="native",
                status="COMPLETE", cost_valid=True))
    add("jena", "LoRA", 92601)
    for dataset in common.EXPECTED:
        for seed in common.SEEDS:
            for role in ("Raw", "G"):
                for label, alpha in (("alpha0", 0.), ("alpha05", .5), ("alpha1", 1.), ("fitted", .37)):
                    add(dataset, "alpha", seed, role=role, label=label, alpha=alpha)
            for mode in ("conditional", "past_control"):
                add(dataset, mode, seed)
    for cost in tables["COST"]:
        if cost["device"] == "cpu":
            cost["CPU_RSS_sampled_peak_MiB"] = 52.
    tables["FIGURE_POINTS"] = report.figure_points(tables)
    return tables


def test_all_f3_native_routes_and_matched_batches_have_exact_explicit_coverage():
    tables = full_coverage_tables()
    models = {row["run_id"]:row for row in tables["RUNS"] if row["row_kind"] == "model"}
    assert len(models) == 56
    points = tables["FIGURE_POINTS"]
    identity = lambda row:(row["run_id"], row["device"], row["batch_size"], row["schedule"])
    expected = {(cost["accuracy_run_id"], cost["device"], cost["batch_size"], "native")
                for cost in tables["COST"] if cost["schedule"] == "native"}
    actual = [point for point in points if point["view"] == "all_native" and point["metric"] == "latency"]
    assert len(actual) == 112
    assert {identity(point) for point in actual} == expected
    assert {point["mode"] for point in actual} >= {"alpha", "conditional", "past_control", "Ridge"}
    for name in {point["figure"] for point in actual}:
        facet = [point for point in actual if point["figure"] == name]
        assert len({(point["dataset"], point["device"], point["batch_size"]) for point in facet}) == 1
    for point in actual:
        if point["seed"] is not None:
            assert "s"+str(point["seed"]) in point["label"]
    cpu = [point for point in points if point["device"] == "cpu"]
    assert {point["metric"] for point in cpu} == {"latency", "RSS"}
    assert all(point["mode"] == "Ridge" for point in cpu)
    matched = [point for point in points if point["view"] == "matched_two_call"]
    expected_matched = {(cost["accuracy_run_id"], cost["device"], cost["batch_size"], cost["schedule"])
        for cost in tables["COST"] if models[cost["accuracy_run_id"]]["mode"] in ("A", "B") and cost["device"].startswith("cuda")}
    assert {identity(point) for point in matched} == expected_matched
    assert {point["batch_size"] for point in matched} == {1, 8}
    assert {point["schedule"] for point in matched} == {"native", "two_call"}
    assert len({(point["mode"], point["schedule"], point["marker"]) for point in matched}) == 3


def test_f3_keeps_coincident_endpoint_identities_and_excludes_invalid_facets():
    tables = full_coverage_tables()
    points = [point for point in tables["FIGURE_POINTS"] if point["view"] == "all_native" and point["metric"] == "latency"
              and point["dataset"] == "jena" and point["batch_size"] == 1 and point["group"] == "alpha_Raw" and point["label"].startswith("alpha0 ")]
    assert len(points) == 2 and points[0]["run_id"] != points[1]["run_id"]
    assert (points[0]["x"], points[0]["y"]) == (points[1]["x"], points[1]["y"])
    invalid = copy.deepcopy(tables["COST"][0])
    invalid.update(cost_id="invalid_facet", cost_valid=False)
    tables["COST"].append(invalid)
    assert all(point["cost_id"] != "invalid_facet" for point in report.figure_points(tables))


def test_audit_uses_dynamic_registered_plan_reader_before_loading_arrays(tmp_path, monkeypatch):
    from research.raw_context_multiscale_20261018_v1 import run
    ctx = dict(report=tmp_path)
    monkeypatch.setattr(report, "_required", lambda context:({}, [], []))
    monkeypatch.setattr(common, "load_task", lambda *a:(_ for _ in ()).throw(AssertionError("array read before plan")))
    seen = []
    def current_plan(context):
        seen.append(context)
        raise RuntimeError("dynamic registered repair selected")
    monkeypatch.setattr(run, "load_registered_plan", current_plan)
    with pytest.raises(RuntimeError, match="dynamic registered repair"):
        report._audit(ctx)
    assert seen == [ctx]


def test_resource_ledger_counts_unique_cache_and_head_storage_and_fit_costs(tmp_path, monkeypatch):
    from research.raw_context_multiscale_20261018_v1 import run
    ctx = dict(common.paths(tmp_path), asset=tmp_path)
    ctx["report"].mkdir(parents=True)
    ctx["cache"].mkdir(parents=True)
    original = ctx["report"]/"EXPERIMENT_PLAN.json"
    repair = ctx["report"]/"EXPERIMENT_PLAN_REPAIR_001.json"
    common.atomic_json(original, dict(preserved="original"))
    original_bytes = original.read_bytes()
    common.atomic_json(repair, dict(synthetic="registered repair"))
    plan = dict(started_unix=1000., limits=dict(wall_seconds=21600), supersedes=common.binding(original, ctx))
    monkeypatch.setattr(run, "load_registered_plan", lambda context:plan)
    monkeypatch.setattr(run, "registered_plan_path", lambda context:repair)
    parent, first, second = (ctx["cache"]/name for name in ("parent.npz", "head1.pt", "head2.pt"))
    parent.write_bytes(b"synthetic-parent-cache")
    first.write_bytes(b"head1")
    second.write_bytes(b"head-two")
    parent_binding = common.binding(parent, ctx)
    events = [dict(stage="parent_cache", status="COMPLETE", time_unix=1001., run_id="fixed_parent", wall_seconds=3.5,
                   prediction_binding=parent_binding)]
    (ctx["cache"]/"EVENTS.jsonl").write_text("\n".join(json.dumps(row) for row in events), encoding="utf8")
    arms = [dict(mode=mode, dataset="jena", run_id=mode, parent_g_prediction_binding=parent_binding,
                 parent_b_prediction_binding=parent_binding if mode == "conditional" else None,
                 head_binding=common.binding(path, ctx), head_search_seconds=.25, head_refit_seconds=.5)
            for mode, path in (("conditional", first), ("past_control", second))]
    result = report._resource_ledger(ctx, dict(arms=arms, costs=[]))
    assert result["registered_plan_revision"] == repair.name
    assert result["registered_plan_binding"] == common.binding(repair, ctx)
    assert result["fixed_parent_M_cal_cache_count"] == 1
    assert result["fixed_parent_M_cal_cache_bytes"] == parent.stat().st_size
    assert result["fixed_parent_M_cal_cache_generation_seconds"] == 3.5
    assert result["fixed_parent_M_cal_cache_events"][0]["prediction_binding"] == parent_binding
    assert result["head_artifact_count"] == 2
    assert result["head_artifact_bytes"] == first.stat().st_size+second.stat().st_size
    assert result["head_search_fit_seconds"] == .5
    assert result["head_fresh_refit_fit_seconds"] == 1.
    assert original.read_bytes() == original_bytes


def test_public_tables_keep_negative_results_seeds_means_and_matching_cost_controls():
    tables = report._public_tables(synthetic_data())
    means = [r for r in tables["RUNS"] if r["row_kind"] == "two_seed_mean"]
    assert len(means) == 5
    assert {r["schedule"] for r in tables["T2_SINGLE_HYPOTHESIS"] if r["contrast"] == "A_to_B_cost"} == {"native_deployment", "matched_two_call"}
    assert {r["metric"] for r in tables["PAIRED_DELTAS"]} == {"mse", "mae"}
    assert {r["index"] for r in tables["PAIRED_DELTAS"] if r["axis"] == "horizon_third"} == {"near", "middle", "far"}
    assert any(r["mode"] == "Ridge" and r["device"] == "cpu" for r in tables["T1_ACCURACY_COST"])


def test_synthetic_public_csv_render_and_replay_are_byte_exact(tmp_path):
    tables = full_coverage_tables()
    resources = dict(campaign_wall_seconds=100., GPU_reserved_process_wall_upper_bound_seconds=50.,
        new_body_configurations=6, body_fresh_refits=6, reused_bodies=4, head_searches=8, head_fresh_refits=8)
    source, replay = tmp_path/"source", tmp_path/"replay"
    report.render_public(tables, resources, [], source)
    report.replay(source, replay)
    actual = sorted(p.name for p in source.iterdir())
    assert len(actual) >= len(report.TABLES)+13
    for name in actual:
        assert (source/name).read_bytes() == (replay/name).read_bytes(), name
    assert "V는 반복 사용된 개발 평가" in (source/"README_KO.md").read_text(encoding="utf8")
    assert "F가 반드시 필요하다" in (source/"README_KO.md").read_text(encoding="utf8")
    assert all((source/name).stat().st_size > 5000 for name in ("F1_TRAINING_CURVES.png", "F2_PAIRED_DELTAS.png", "F3_ACCURACY_COST.png"))
    manifest = common.read(source/"FIGURE_MANIFEST.json")
    assert len(manifest["all_native_facets"]) == 112
    assert {item["run_id"] for item in manifest["all_native_facets"]} == {row["run_id"] for row in tables["RUNS"] if row["row_kind"] == "model"}
    assert len({item["figure"] for item in manifest["all_native_facets"]}) == 6
    assert all(item["png_sha256"] == common.sha(source/(item["name"]+".png")) for item in manifest["figures"])
    (source/"RUNS.csv").write_text("tamper", encoding="utf8")
    with pytest.raises(ValueError, match="Public source CSV"):
        report.replay(source, tmp_path/"tampered")
