"""Final report tests use public synthetic CPU fixtures only; no job or data access."""
import copy
import importlib.util
from pathlib import Path

import pytest

from research.raw_context_multiscale_20261018_v1 import common, report as primary
from research.raw_context_multiscale_supplement_20261019_v1 import report


def fixture_tables():
    path = Path(__file__).with_name("test_raw_context_multiscale_report.py")
    spec = importlib.util.spec_from_file_location("_synthetic_primary_report_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.full_coverage_tables()


def fixture_contract():
    ranges = dict(jena=dict(A_fit=[0,10482], A_stop=[10482,12579], M_cal=[12579,14675],
                           V_select=[14675,16772], T_final=[16772,20965]),
                  pems32=dict(A_fit=[0,8928], A_stop=[8928,10713], M_cal=[10713,12499],
                              V_select=[12499,14284], T_final=[14284,17856]))
    return dict(tasks={dataset:dict(C=value["C"], L=512, H=value["H"], split_ranges=ranges[dataset])
                       for dataset, value in common.EXPECTED.items()})


def fixture_metadata(tables):
    metadata, cp0 = {}, []
    for row in tables["RUNS"]:
        if row["row_kind"] != "model" or row["mode"] not in ("S", "A", "B"):
            continue
        identifier = row["run_id"]
        metadata[identifier] = dict(V_origin_identity=dict(shape=[3], dtype="int64", sha256=row["dataset"]+"-cohort"),
            V_origin_count=3, A_fit_end=10482, A60_refit_end=12579, search_train_origins=9923,
            search_updates=1863, search_exposures=29769, search_actual_epochs=3., search_cap=3.,
            selected_A_stop_epoch=1., refit_train_origins=12020, refit_updates=752, refit_exposures=12020,
            refit_actual_epochs=1., reused_final_body=row["mode"] == "S")
        cp0.append(dict(run_id=identifier, dataset=row["dataset"], mode=row["mode"], seed=row["seed"],
            epoch=0, updates=0, exposures=0, A_stop_mse=1.3, A_stop_mae=.7, n_origins=3,
            origins_sha256=row["dataset"]+"-astop", MSE_parity="PASS", split="A_stop"))
    return metadata, cp0


def fixture_context(tmp_path):
    ctx = dict(common.paths(tmp_path), asset=tmp_path/"asset")
    ctx["report"].mkdir(parents=True)
    ctx["cache"].mkdir(parents=True)
    sources = {}
    for index in range(13):
        relative = f"research/frozen_primary/module_{index}.py"
        source = tmp_path/relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"FIXTURE = {index}\n", encoding="utf8")
        sources[relative] = common.sha(source)
    primary_path = ctx["report"]/"EXPERIMENT_PLAN_REPAIR_002.json"
    common.atomic_json(primary_path, dict(source_files=sources))
    relative = "research/"+report.NAMESPACE+"/report.py"
    source = tmp_path/relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("SYNTHETIC_SOURCE_ONLY = True\n", encoding="utf8")
    supplement_sources = {relative:common.sha(source)}
    for filename in ("__init__.py", "cp0.py"):
        extra = source.with_name(filename)
        extra.write_text("SYNTHETIC_SOURCE_ONLY = True\n", encoding="utf8")
        supplement_sources[extra.relative_to(tmp_path).as_posix()] = common.sha(extra)
    start = dict(started_unix=1000., limits=dict(wall_seconds=21600., GPU_reservation_seconds=14400.))
    common.atomic_json(ctx["cache"]/"CORE_EXECUTION_START.json", start)
    plan = dict(status="REGISTERED_BEFORE_NEW_V", registered_unix=1050., primary_source_files=sources,
        source_files=supplement_sources, primary_plan_binding=common.binding(primary_path, ctx),
        scientific_contract_changed=False, selection_changes=False, new_body_training=False,
        original_execution_start_binding=common.binding(ctx["cache"]/"CORE_EXECUTION_START.json", ctx),
        original_deadline_unix=22600., cp0_commands=["python -m research."+report.NAMESPACE+".cp0 --asset-root <local-asset-root> --plan reports/"+common.NAME+"/SUPPLEMENT_PLAN_001.json --device cuda"])
    plan_path = ctx["report"]/"SUPPLEMENT_PLAN_001.json"
    common.atomic_json(plan_path, plan)
    terminal_path = ctx["cache"]/"TERMINAL_CORE_002.json"
    common.atomic_json(terminal_path, dict(status="COMPLETE", exit_code=0))
    return ctx, plan_path, terminal_path


def write_full_synthetic_inputs(ctx, plan_path):
    tables = fixture_tables()
    metadata, cp0 = fixture_metadata(tables)
    contract = fixture_contract()
    common.atomic_json(ctx["repo"]/"reports/b2_revin_20261018_v1/DATA_CONTRACT.json", contract)
    for name in primary.TABLES:
        primary._csv(ctx["report"]/(name+".csv"), tables[name])
    common.atomic_json(ctx["report"]/"REPORT_INPUTS.json", dict(resume_queue=[],
        tables={name:common.sha(ctx["report"]/(name+".csv")) for name in primary.TABLES}))
    common.atomic_json(ctx["report"]/"RESOURCE_LEDGER.json", dict(started_unix=900., accounted_through_unix=1080.,
        campaign_wall_seconds=180., GPU_reserved_process_wall_upper_bound_seconds=20.,
        new_body_configurations=6, body_fresh_refits=6, reused_bodies=4, head_searches=8, head_fresh_refits=8))
    arms = []
    for row in tables["RUNS"]:
        if row["row_kind"] != "model":
            continue
        arm = dict(run_id=row["run_id"], dataset=row["dataset"], mode=row["mode"], seed=row["seed"],
            score=dict(n_origins=row["n_origins"]), array_identities=dict(origins=dict(shape=[3], dtype="int64", sha256=row["dataset"]+"-cohort")))
        if row["mode"] in ("S", "A", "B"):
            task = contract["tasks"][row["dataset"]]
            arm.update(route_spec=dict(artifact_sha256="frozen-artifact-"+row["run_id"]), reused=row["mode"] == "S",
                curves=[dict(epoch=0.), dict(epoch=1.)], search_summary=dict(selected_epoch=1., train_origins=9923,
                    updates=1863, exposures=29769, actual_epochs=3.))
            summary = dict(train_origins=12020, updates=752, exposures=12020, actual_epochs=1.)
            refit = dict(status="COMPLETE", bundle_binding=dict(sha256=arm["route_spec"]["artifact_sha256"]),
                fit_receipt=dict(artifact_sha256=arm["route_spec"]["artifact_sha256"], search_fit_end=task["split_ranges"]["A_fit"][1],
                    refit_fit_end=task["split_ranges"]["M_cal"][0], fresh_A60=True), refit_summary=summary)
            if arm["reused"]:
                reference = f"historical/runs/{row['run_id']}/cap3/SEARCH_RESULT.json"
                arm["parent"] = dict(search_result=dict(reference="repo://.cache/"+reference))
                path = ctx["repo"]/"reports"/reference.replace("SEARCH_RESULT.json", "REFIT_RESULT.json")
            else:
                arm["refit_summary"] = summary
                path = ctx["report"]/"runs"/row["run_id"]/"cap3/REFIT_RESULT.json"
            common.atomic_json(path, refit)
        arms.append(arm)
    common.atomic_json(ctx["report"]/"SCORES.json", dict(status="COMPLETE", arms=arms))
    common.atomic_json(ctx["report"]/"COST_INDEX.json", dict(status="COMPLETE", facets=[dict(cost_id=row["cost_id"]) for row in tables["COST"]]))
    common.atomic_json(ctx["report"]/"MODELS_SEALED_BEFORE_V.json", dict(status="SEALED"))
    plan = common.read(plan_path)
    common.atomic_json(ctx["report"]/"VALIDATION_REPORT.json", dict(status="PASS", models=len(arms), cost_facets=len(tables["COST"]),
        manifest_bindings={**{name:common.binding(ctx["report"]/name, ctx) for name in ("SCORES.json", "COST_INDEX.json", "MODELS_SEALED_BEFORE_V.json")},
                           "registered_plan":plan["primary_plan_binding"]}))
    primary._csv(ctx["report"]/"CP0_A_STOP.csv", cp0)
    common.atomic_json(ctx["report"]/"CP0_A_STOP_VALIDATION.json", dict(status="PASS", plan_binding=common.binding(plan_path, ctx),
        primary_plan_binding=plan["primary_plan_binding"], CSV_binding=common.binding(ctx["report"]/"CP0_A_STOP.csv", ctx),
        coverage=10, expected_coverage=10, training_updates=0, V_access=False, T_access=False,
        wall_seconds=3., started_unix=1081., finished_unix=1084., GPU_seconds_before=20.,
        GPU_seconds_after=22., GPU_seconds_added=2., GPU_accounting="GPU-owned wall upper bound"))
    events = [dict(stage="GPU_SESSION", status="STARTED", session_id="body", time_unix=1001.),
              dict(stage="GPU_SESSION", status="COMPLETE", session_id="body", time_unix=1021., wall_seconds=20.),
              dict(stage="GPU_SESSION", status="STARTED", session_id="cp0", time_unix=1081.),
              dict(stage="GPU_SESSION", status="COMPLETE", session_id="cp0", time_unix=1083., wall_seconds=2.)]
    import json
    (ctx["cache"]/"EVENTS.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf8")
    return tables


def test_mean_selection_auxiliaries_are_never_seed1_disguised_as_mean():
    rows = fixture_tables()["RUNS"]
    model = next(row for row in rows if row["mode"] == "S" and row["dataset"] == "jena" and row["seed"] == 92602)
    model.update(selected_A_stop_epoch=3., A_stop_mse=.9, convergence="selected_within_registered_budget")
    output = report.canonical_runs(rows)
    mean = next(row for row in output if row["mode"] == "S" and row["dataset"] == "jena" and row["row_kind"] == "two_seed_mean")
    assert mean["selected_A_stop_epoch"] is None and mean["A_stop_mse"] is None and mean["convergence"] is None
    assert mean["by_seed_auxiliary_values"]["92602"]["selected_A_stop_epoch"] == 3.
    assert mean["by_seed_auxiliary_values"]["92601"]["selected_A_stop_epoch"] == 1.
    assert mean["n_origins"] == 3
    original = next(row for row in rows if row["run_id"] == mean["run_id"])
    assert mean["mse"] == original["mse"] and mean["mae"] == original["mae"]
    assert "arithmetic mean" in mean["aggregation_scope"]


def test_all_t1_k_splits_t2_learning_and_fixed_cp0_are_joined_without_metric_changes():
    source = fixture_tables()
    metadata, cp0 = fixture_metadata(source)
    output = report.augment_tables(source, body_metadata=metadata, contract=fixture_contract(), cp0=cp0)
    for row in output["T1_ACCURACY_COST"]:
        if row["mode"] in ("A", "B"):
            assert row["K_split_numeric"] == ("6+6" if row["dataset"] == "jena" else "4+4")
            assert row["scales"] == ("1+2" if row["mode"] == "B" else "1+1")
        elif row["mode"] == "S":
            assert row["K_raw"] == 0
    for row in output["T2_SINGLE_HYPOTHESIS"]:
        assert row["same_actual_V_origins"] is True
        assert row["candidate_refit_exposures"] == 12020
        assert row["reference_search_exposures"] == 29769
        assert row["candidate_V_origin_identity"] == row["reference_V_origin_identity"]
    assert len(output["CP0_A_STOP"]) == 10
    assert len([point for point in output["SUPPLEMENT_FIGURE_POINTS"] if point["figure"] == "F1_CP0_DIAGNOSTIC"]) == 20
    assert output["COST"] == source["COST"] and output["CHANNEL_HORIZON_STATS"] == source["CHANNEL_HORIZON_STATS"]
    assert output["FIGURE_POINTS"] == source["FIGURE_POINTS"]
    assert len([point for point in output["FIGURE_POINTS"] if point["view"] == "all_native" and point["metric"] == "latency"]) == 112
    assert source["RUNS"] != output["RUNS"]


def test_equal_counts_with_different_origin_hashes_fail_instead_of_pairing():
    tables = fixture_tables()
    metadata, cp0 = fixture_metadata(tables)
    metadata[next(identifier for identifier in metadata if identifier.startswith("A_jena"))]["V_origin_identity"]["sha256"] = "different"
    with pytest.raises(ValueError, match="identical origin arrays"):
        report.augment_tables(tables, body_metadata=metadata, contract=fixture_contract(), cp0=cp0)


def test_cp0_parity_and_complete_seed_horizon_thirds_fail_closed():
    tables = fixture_tables()
    metadata, cp0 = fixture_metadata(tables)
    bad = copy.deepcopy(cp0)
    bad[0]["A_stop_mse"] += .01
    with pytest.raises(ValueError, match="existing epoch-zero"):
        report.augment_tables(tables, body_metadata=metadata, contract=fixture_contract(), cp0=bad)
    tables["PAIRED_DELTAS"].pop(next(index for index, row in enumerate(tables["PAIRED_DELTAS"]) if row["axis"] == "horizon_third"))
    with pytest.raises(ValueError, match="both seeds and all three"):
        report.augment_tables(tables, body_metadata=metadata, contract=fixture_contract(), cp0=cp0)


def test_split_ranges_and_origin_counts_include_unread_protected_scope():
    rows = report.split_scopes(fixture_contract())
    lookup = {(row["dataset"], row["split"]):row for row in rows}
    assert lookup["jena", "V_select"]["origin_count"] == 43
    assert lookup["pems32", "V_select"]["origin_count"] == 148
    assert lookup["jena", "M_fit"]["origin_count"] == 1210
    assert lookup["jena", "M_select"]["origin_count"] == 792
    assert lookup["pems32", "M_fit"]["origin_count"] == 1060
    assert lookup["pems32", "M_select"]["origin_count"] == 704
    assert all(row["origin_count"] is None and "PROTECTED" in row["origin_count_scope"] for row in rows if row["split"] == "T_final")


def test_partial_is_explicitly_unsupported_before_any_score_access(tmp_path):
    with pytest.raises(ValueError, match="PARTIAL unsupported"):
        report.generate({}, plan_path=tmp_path/"missing", terminal_path=tmp_path/"missing", mode="partial")


def test_nonterminal_and_source_changes_block_before_score_access(tmp_path, monkeypatch):
    ctx, plan_path, terminal_path = fixture_context(tmp_path)
    common.atomic_json(terminal_path, dict(status="FAILED", exit_code=1))
    monkeypatch.setattr(report, "time", lambda:1100.)
    monkeypatch.setattr(primary, "_read_csv", lambda *args:(_ for _ in ()).throw(AssertionError("score accessed")))
    with pytest.raises(ValueError, match="Main terminal"):
        report.generate(ctx, plan_path=plan_path, terminal_path=terminal_path)
    source = next(iter(common.read(plan_path)["primary_source_files"]))
    (ctx["repo"]/source).write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="Registered source changed"):
        report.generate(ctx, plan_path=plan_path, terminal_path=terminal_path)


def test_gpu_resource_sessions_are_counted_once_with_failed_attempts_and_report_wall():
    events = [dict(stage="GPU_SESSION", status="STARTED", session_id="failed", time_unix=10),
              dict(stage="GPU_SESSION", status="FAILED", session_id="failed", time_unix=15, wall_seconds=5),
              dict(stage="GPU_SESSION", status="STARTED", session_id="cp0", time_unix=16),
              dict(stage="GPU_SESSION", status="COMPLETE", session_id="cp0", time_unix=18, wall_seconds=2)]
    original = dict(started_unix=0, GPU_reserved_process_wall_upper_bound_seconds=5)
    kwargs = dict(core_start=dict(started_unix=10, limits=dict(wall_seconds=100, GPU_reservation_seconds=50)),
                  events=events, started_unix=20, finished_unix=24, cp0_timings=dict(wall_seconds=2))
    actual = report.final_resources(original, **kwargs)
    assert actual["GPU_reserved_process_wall_upper_bound_seconds"] == 7
    assert actual["final_report_CPU_wall_seconds"] == 4
    assert actual["campaign_wall_seconds"] == 24
    assert actual["core_execution_wall_seconds"] == 14
    assert original["GPU_reserved_process_wall_upper_bound_seconds"] == 5
    with pytest.raises(ValueError, match="duplicate GPU"):
        report.final_resources(original, **dict(kwargs, events=events+[events[-1]]))
    with pytest.raises(ValueError, match="closed exactly once"):
        report.final_resources(original, **dict(kwargs, events=events[:-1]))


def test_public_refit_join_never_substitutes_search_counts_for_missing_refit(tmp_path):
    ctx, plan_path, _ = fixture_context(tmp_path)
    write_full_synthetic_inputs(ctx, plan_path)
    arms = common.read(ctx["report"]/"SCORES.json")["arms"]
    metadata, bindings = report._body_metadata(ctx, arms, fixture_contract())
    assert len(metadata) == len(bindings) == 10
    assert all(value["refit_exposures"] == 12020 and value["search_exposures"] == 29769 for value in metadata.values())
    arm = next(arm for arm in arms if arm["mode"] == "A")
    del arm["refit_summary"]["train_origins"]
    with pytest.raises(ValueError, match="differs from scored"):
        report._body_metadata(ctx, arms, fixture_contract())


def test_full_generate_public_replay_are_byte_exact_and_preserve_primary(tmp_path, monkeypatch):
    ctx, plan_path, terminal_path = fixture_context(tmp_path)
    write_full_synthetic_inputs(ctx, plan_path)
    before = {path.relative_to(ctx["report"]).as_posix():common.sha(path) for path in ctx["report"].rglob("*") if path.is_file()}
    ticks = iter((1100., 1110.))
    monkeypatch.setattr(report, "time", lambda:next(ticks))
    result = report.generate(ctx, plan_path=plan_path, terminal_path=terminal_path)
    assert result["status"] == "COMPLETE"
    output = ctx["report"]/"final"
    resources = common.read(output/"RESOURCE_LEDGER.json")
    assert resources["GPU_reserved_process_wall_upper_bound_seconds"] == 22.
    assert resources["final_report_CPU_wall_seconds"] == 10.
    assert resources["accounted_through_unix"] == 1110.
    target = ctx["cache"]/"replay"
    report.replay(output, target)
    names = sorted(path.name for path in output.iterdir() if path.is_file())
    assert len(names) >= 50
    for name in names:
        assert (output/name).read_bytes() == (target/name).read_bytes(), name
    assert all(common.sha(ctx["report"]/relative) == digest for relative, digest in before.items())
    assert "REGISTERED" not in (output/"README_KO.md").read_text(encoding="utf8")
    commands = (output/"CLI_USAGE_KO.md").read_text(encoding="utf8")
    assert all(".run "+stage in commands for stage in ("audit", "smoke", "execute", "verify", "report"))
    assert "SUPPLEMENT_PLAN_001.json" in commands
    assert all((output/(name+".png")).stat().st_size > 5000 for name in ("F1_CAP_MARKED", "F1_CP0_DIAGNOSTIC", "F2_HORIZON_THIRDS"))
    manifest = common.read(output/"FIGURE_MANIFEST.json")
    assert len(manifest["all_native_facets"]) == 112
    assert len(manifest["figures"]) == 14
    with pytest.raises(ValueError, match="new or empty"):
        report.replay(output, target)
    (output/"CP0_A_STOP.csv").write_text("tampered", encoding="utf8")
    with pytest.raises(ValueError, match="Public source CSV"):
        report.replay(output, ctx["cache"]/"tampered_replay")
