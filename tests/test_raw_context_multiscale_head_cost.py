"""Synthetic public receipts only: no campaign results, model state, or GPU access."""
import builtins
import copy
import csv
import json
from types import ModuleType, SimpleNamespace

import pytest

from research.raw_context_multiscale_head_cost_20261019_v1 import run


def synthetic_lineage(receipt, redirected=False):
    row = receipt["row"]
    worker = dict(pid=row["process_pid"], create_time=row["process_create_time"])
    collector = dict(pid=900, create_time=90.)
    launcher = dict(pid=row["process_pid"]+20000, create_time=row["process_create_time"]-.001) if redirected else worker
    receipt["process_lineage"] = dict(worker=worker, ancestors=([launcher] if redirected else [])+[collector])
    receipt["launch_proof"] = dict(collector=collector, launcher=launcher)
    receipt["request_collector_identity"] = collector
    return receipt


def synthetic_trials(batch=1, origins=None):
    origins = list(range(13000, 13008)) if origins is None else origins
    output = []
    for phase, count in (("warmup", 10), ("timed", 20)):
        for index in range(count):
            output.append(dict(phase=phase, trial=index,
                whole_seconds=1. if phase == "warmup" else (index+1)/1000.,
                origins=origins if batch == 8 else [origins[index % 8]], F_calls=0,
                peak_allocated_MiB=1000. if phase == "warmup" else 32.+index,
                peak_reserved_MiB=2000. if phase == "warmup" else 64.+index))
    return output


def synthetic_receipts():
    output = []
    for index, (dataset, mode, seed, batch) in enumerate(sorted(run.EXPECTED)):
        identifier = f"{mode}_{dataset}_s{seed}"
        origins = list(range(13000, 13008)) if dataset == "jena" else list(range(11000, 11008))
        row = dict(run_id=identifier, dataset=dataset, mode=mode, seed=seed, batch_size=batch,
            component_cost_id=identifier+f"_B{batch}_HEAD_COMPONENT", whole_cost_id=identifier+f"_B{batch}_native",
            device="cuda", process_pid=10000+index, process_create_time=100.+index,
            fixed_first8_origins=origins, F_calls=0, forward_scope=run.SCOPE,
            p50_batch_ms=10.5, IQR_batch_ms=9.5, p95_batch_ms=19.05,
            p50_ms_per_origin=10.5/batch, p95_ms_per_origin=19.05/batch,
            peak_allocated_MiB=51., peak_reserved_MiB=83., resident_parameters=1337,
            trainable_parameters=0, process_GPU_sampled_peak_MiB=None,
            process_GPU_sampling_status="NOT_MEASURED")
        output.append(synthetic_lineage(dict(status="COMPLETE", row=row, trials=synthetic_trials(batch, origins),
            parity=dict(status="PASS", atol=1e-5, rtol=1e-4),
            request_binding=dict(reference=f"cache://synthetic/{identifier}_B{batch}.json", sha256="a"*64),
            input_binding=dict(reference=f"cache://synthetic/{identifier}_input.npz", sha256="b"*64),
            head_binding=dict(reference=f"cache://synthetic/{identifier}_head.pt", sha256="c"*64))))
    return output


def synthetic_resources():
    events = []
    for index, receipt in enumerate(synthetic_receipts()):
        events.extend((dict(status="STARTED", session_id=f"component-{index}", time_unix=1.+index*2,
            request_binding=receipt["request_binding"]),
            dict(status="COMPLETE", session_id=f"component-{index}", time_unix=2.+index*2, wall_seconds=1.,
                request_binding=receipt["request_binding"])))
    primary_events = dict(reference="cache://synthetic_primary_EVENTS.jsonl", sha256="e"*64)
    return dict(original_deadline_unix=21700., GPU_base_seconds=50.,
        component_owned_worker_wall_seconds=16., GPU_total_process_wall_upper_bound_seconds=66.,
        GPU_limit_seconds=14400, owned_sessions=dict(seconds=16., open_sessions=0, closed_sessions=16),
        owned_events=events, finished_unix=100., primary_EVENTS_before=primary_events, primary_EVENTS_after=primary_events,
        primary_EVENTS_appended=False,
        prerequisite_bindings=dict(scope=dict(reference="report://synthetic_scope.json", sha256="d"*64), primary_EVENTS=primary_events),
        new_training_updates=0, original_budget_reset=False)


def synthetic_gate_context(tmp_path):
    ctx = run.context(tmp_path/"asset", repo=tmp_path)
    ctx["cache"].mkdir(parents=True)
    ctx["report"].mkdir(parents=True)
    sources = {}
    for index in range(20):
        relative = f"research/synthetic_frozen/module_{index}.py"
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"SYNTHETIC_SOURCE = {index}\n", encoding="utf8")
        sources[relative] = run.sha(path)
    new_sources = {}
    for name in ("__init__.py", "run.py"):
        relative = "research/"+run.NAME+"/"+name
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SYNTHETIC_COMPONENT_SOURCE = True\n", encoding="utf8")
        new_sources[relative] = run.sha(path)
    test_relative = "tests/test_raw_context_multiscale_head_cost.py"
    test_path = tmp_path/test_relative
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("SYNTHETIC_TEST_SOURCE = True\n", encoding="utf8")
    start_path = ctx["cache"]/"CORE_EXECUTION_START.json"
    run.write_new(start_path, dict(started_unix=100., limits=dict(wall_seconds=21600, GPU_reservation_seconds=14400)))
    plan_path = ctx["report"]/"SYNTHETIC_PRIMARY_PLAN.json"
    seal_path = ctx["report"]/"MODELS_SEALED_BEFORE_V.json"
    run.write_new(plan_path, dict(status="REGISTERED_SYNTHETIC"))
    run.write_new(seal_path, dict(status="SEALED_SYNTHETIC"))
    (ctx["cache"]/"EVENTS.jsonl").write_text('{"synthetic":true}\n', encoding="utf8")
    for name in ("SCORES.json", "COST_INDEX.json"):
        run.write_new(ctx["report"]/name, dict(status="SYNTHETIC_PUBLIC_METADATA_ONLY"))
    primary = dict(status="PASS", models=56, cost_facets=120,
        manifest_bindings=dict(registered_plan=run.binding(plan_path, ctx),
            **{name:run.binding(ctx["report"]/name, ctx) for name in ("SCORES.json", "COST_INDEX.json", "MODELS_SEALED_BEFORE_V.json")}))
    run.write_new(ctx["report"]/"VALIDATION_REPORT.json", primary)
    run.write_new(ctx["report"]/"INDEPENDENT_FINAL_AUDIT_001.json", dict(status="PASS", coverage=dict(
        V_models=56, whole_path_cost_facets=120, CP0_models=10, expected_CP0_models=10)))
    (ctx["report"]/"M_CAL_DIAGNOSTICS.csv").write_text("synthetic_metadata_only\n", encoding="utf8")
    run.write_new(ctx["report"]/"M_CAL_VALIDATION.json", dict(status="PASS", models=40, head_models=8,
        CSV_binding=run.binding(ctx["report"]/"M_CAL_DIAGNOSTICS.csv", ctx)))
    run.write_new(ctx["report"]/"FINAL_CONTEXT.json", dict(resource_events_binding=run.binding(ctx["cache"]/"EVENTS.jsonl", ctx)))
    run.write_new(ctx["report"]/"RESOURCE_LEDGER.json", dict(immutable_core_deadline_unix=21700.,
        core_execution_started_unix=100., unmatched_started_GPU_sessions=0,
        GPU_reserved_process_wall_upper_bound_seconds=50.))
    run.write_new(ctx["cache"]/"TERMINAL_CORE_002.json", dict(status="COMPLETE", exit_code=0))
    run.write_new(ctx["cache"]/"TERMINAL_SUPPLEMENT_CHAIN_001.json", dict(
        status="SUPPLEMENT_STAGES_COMPLETE_PENDING_ROOT_REVIEW_AND_PUBLICATION", exit_code=0))
    public_names = dict(primary_validation="VALIDATION_REPORT.json", independent_validation="INDEPENDENT_FINAL_AUDIT_001.json",
        Mcal_validation="M_CAL_VALIDATION.json", Mcal_diagnostics="M_CAL_DIAGNOSTICS.csv", final_context="FINAL_CONTEXT.json")
    protocol = dict(heads=8, facets=16, batches=[1, 8], warmup=10, timed=20, device="cuda", dtype="FP32",
        inference_mode=True, origins="first8_dense_Mcal", split="M_cal", atol=1e-5, rtol=1e-4, Fcalls=0,
        new_training_updates=0, new_model_selection=False, new_V_forecasts=False, T_access=False)
    scope = dict(status="REGISTERED_BEFORE_COMPONENT_MEASUREMENT", scientific_contract_changed=False,
        selection_changes=False, source_files=sources, component_protocol=protocol,
        original_execution_start_binding=run.binding(start_path, ctx), original_deadline_unix=21700.,
        limits=dict(wall_seconds=21600, GPU_reservation_seconds=14400),
        primary_plan_binding=run.binding(plan_path, ctx), model_seal_binding=run.binding(seal_path, ctx),
        required_terminal_references=dict(core="cache://TERMINAL_CORE_002.json", chain="cache://TERMINAL_SUPPLEMENT_CHAIN_001.json"),
        required_public_references={name:"report://"+filename for name, filename in public_names.items()},
        GPU_base_ledger_reference="report://RESOURCE_LEDGER.json")
    scope_path = ctx["report"]/"SYNTHETIC_COMPONENT_SCOPE.json"
    run.write_new(scope_path, scope)
    preflight_path = ctx["report"]/"SYNTHETIC_COMPONENT_PREFLIGHT.json"
    run.write_new(preflight_path, dict(status="PASS_SYNTHETIC_AND_FROZEN_CODE", scope_binding=run.binding(scope_path, ctx),
        actual_evaluation_performed=False, source_files=new_sources, test_source_files={test_relative:run.sha(test_path)}))
    return ctx, scope_path, preflight_path


def rewrite_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf8")


@pytest.mark.parametrize("batch", [1, 8])
def test_linear_quantiles_exclude_warmup_and_normalize_per_origin(batch):
    trials = list(reversed(synthetic_trials(batch)))
    stats = run.latency_statistics(trials, batch)
    assert stats == pytest.approx(dict(p50_batch_ms=10.5, IQR_batch_ms=9.5, p95_batch_ms=19.05,
        p50_ms_per_origin=10.5/batch, p95_ms_per_origin=19.05/batch), abs=1e-12)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "phase", "timed_index", "warmup_index"])
def test_statistics_require_registered_trial_coverage(mutation):
    trials = synthetic_trials()
    if mutation == "missing":
        trials.pop()
    elif mutation == "duplicate":
        trials[-1] = copy.deepcopy(trials[-2])
    elif mutation == "phase":
        trials[-1]["phase"] = "measure"
    elif mutation == "timed_index":
        trials[-1]["trial"] = 20
    else:
        trials[0]["trial"] = 10
    with pytest.raises(ValueError, match="ten warmup and twenty"):
        run.latency_statistics(trials, 1)


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
@pytest.mark.parametrize("index", [0, 10])
def test_warmup_and_timed_trials_must_both_be_finite_positive(value, index):
    trials = synthetic_trials()
    trials[index]["whole_seconds"] = value
    with pytest.raises(ValueError, match="Finite positive"):
        run.latency_statistics(trials, 1)


def test_unregistered_batch_is_rejected():
    with pytest.raises(ValueError, match="B1/B8"):
        run.latency_statistics(synthetic_trials(), 4)


def test_owned_wall_counts_closed_once_failed_once_and_open_elapsed():
    events = [dict(status="STARTED", session_id="done", time_unix=100.),
        dict(status="COMPLETE", session_id="done", time_unix=110., wall_seconds=10.),
        dict(status="STARTED", session_id="failed", time_unix=110.),
        dict(status="FAILED", session_id="failed", time_unix=114., wall_seconds=4.),
        dict(status="STARTED", session_id="open", time_unix=120.)]
    assert run.owned_wall(events, 127.) == dict(seconds=21., open_sessions=1, closed_sessions=2)
    assert run.owned_wall([], 127.) == dict(seconds=0., open_sessions=0, closed_sessions=0)


@pytest.mark.parametrize("events", [
    [dict(status="STARTED", session_id="a", time_unix=1.)]*2,
    [dict(status="COMPLETE", session_id="a", time_unix=2., wall_seconds=1.)],
    [dict(status="STARTED", session_id="a", time_unix=1.), dict(status="COMPLETE", session_id="a", time_unix=2., wall_seconds=1.)]*2,
    [dict(status="UNKNOWN", session_id="a", time_unix=1.)],
    [dict(status="STARTED", session_id="a", time_unix=20.)],
    [dict(status="STARTED", session_id="a", time_unix=1.), dict(status="FAILED", session_id="a", time_unix=2., wall_seconds=-1.)],
    [dict(status="STARTED", session_id="a", time_unix=1.), dict(status="FAILED", session_id="a", time_unix=2., wall_seconds=float("nan"))],
    [dict(status="STARTED", session_id="a", time_unix=5.), dict(status="COMPLETE", session_id="a", time_unix=2., wall_seconds=1.)],
    [dict(status="STARTED", session_id="a", time_unix=1.), dict(status="COMPLETE", session_id="a", time_unix=11., wall_seconds=10.)],
    [dict(status="STARTED", session_id="a", time_unix=1.), dict(status="COMPLETE", session_id="a", time_unix=2., wall_seconds=2.)],
])
def test_invalid_owned_worker_journals_fail_closed(events):
    with pytest.raises(ValueError):
        run.owned_wall(events, 10.)


def test_receipts_cover_exact_eight_heads_sixteen_facets_and_480_trials():
    receipts = list(reversed(synthetic_receipts()))
    before = copy.deepcopy(receipts)
    rows, trials = run.audit_receipts(receipts)
    assert len(rows) == 16 and len(trials) == 480
    assert {(row["dataset"], row["mode"], row["seed"], row["batch_size"]) for row in rows} == run.EXPECTED
    assert len({row["run_id"] for row in rows}) == 8
    assert len({(row["process_pid"], row["process_create_time"]) for row in rows}) == 16
    assert sum(trial["phase"] == "timed" for trial in trials) == 320
    assert all(trial["F_calls"] == 0 for trial in trials)
    assert receipts == before


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "dataset", "mode", "seed", "batch", "incomplete"])
def test_missing_duplicate_unregistered_or_incomplete_facets_are_rejected(mutation):
    receipts = synthetic_receipts()
    if mutation == "missing":
        receipts.pop()
    elif mutation == "duplicate":
        receipts[-1] = copy.deepcopy(receipts[0])
    elif mutation == "incomplete":
        receipts[0]["status"] = "PARTIAL"
    else:
        field = "batch_size" if mutation == "batch" else mutation
        receipts[0]["row"][field] = {"dataset":"other", "mode":"other", "seed":123, "batch":4}[mutation]
    with pytest.raises(ValueError):
        run.audit_receipts(receipts)


def test_process_identity_requires_a_fresh_pid_creation_pair_per_facet():
    receipts = synthetic_receipts()
    receipts[1]["row"]["process_pid"] = receipts[0]["row"]["process_pid"]
    synthetic_lineage(receipts[1])
    run.audit_receipts(receipts)
    receipts[1]["row"]["process_create_time"] = receipts[0]["row"]["process_create_time"]
    with pytest.raises(ValueError, match="fresh PID"):
        run.audit_receipts(receipts)


@pytest.mark.parametrize("field,value", [("status", "FAIL"), ("atol", 1e-4), ("rtol", 1e-3)])
def test_parity_must_use_registered_tolerances_and_pass(field, value):
    receipts = synthetic_receipts()
    receipts[0]["parity"][field] = value
    with pytest.raises(ValueError, match="parity/component-only"):
        run.audit_receipts(receipts)


@pytest.mark.parametrize("field,value", [("F_calls", 1), ("forward_scope", "features cached outside timer"), ("device", "cpu")])
def test_head_component_contract_cannot_hide_F_or_skip_features(field, value):
    receipts = synthetic_receipts()
    receipts[0]["row"][field] = value
    with pytest.raises(ValueError, match="parity/component-only"):
        run.audit_receipts(receipts)


@pytest.mark.parametrize("name", ["request_binding", "input_binding", "head_binding"])
@pytest.mark.parametrize("field", ["reference", "sha256"])
def test_all_request_input_and_final_head_bindings_are_required(name, field):
    receipts = synthetic_receipts()
    del receipts[0][name][field]
    with pytest.raises(ValueError, match="provenance"):
        run.audit_receipts(receipts)


@pytest.mark.parametrize("name", ["input_binding", "head_binding"])
def test_batch_facets_must_bind_the_same_input_bundle_and_head_state(name):
    receipts = synthetic_receipts()
    receipts[1][name]["sha256"] = "e"*64
    with pytest.raises(ValueError, match="same head state"):
        run.audit_receipts(receipts)


@pytest.mark.parametrize("mutation", ["short", "duplicate", "non_dense", "reversed", "other_head", "other_batch"])
def test_every_head_in_dataset_uses_exact_same_eight_dense_Mcal_origins(mutation):
    receipts = synthetic_receipts()
    target = receipts[0]
    origins = target["row"]["fixed_first8_origins"]
    if mutation == "short":
        replacement = origins[:-1]
    elif mutation == "duplicate":
        replacement = origins[:-1]+[origins[0]]
    elif mutation == "non_dense":
        replacement = [value*2 for value in origins]
    elif mutation == "reversed":
        replacement = list(reversed(origins))
    else:
        target = receipts[2] if mutation == "other_head" else receipts[1]
        replacement = [value+20 for value in target["row"]["fixed_first8_origins"]]
    target["row"]["fixed_first8_origins"] = replacement
    if len(replacement) == 8:
        target["trials"] = synthetic_trials(target["row"]["batch_size"], replacement)
    with pytest.raises(ValueError, match="origins|origin"):
        run.audit_receipts(receipts)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("mutation", ["origin", "F_calls"])
def test_each_trial_keeps_the_registered_batch_trace_and_zero_F(batch, mutation):
    receipts = synthetic_receipts()
    target = next(receipt for receipt in receipts if receipt["row"]["batch_size"] == batch)
    trial = target["trials"][10]
    if mutation == "origin":
        trial["origins"] = [0]*batch
    else:
        trial["F_calls"] = 1
    with pytest.raises(ValueError, match="trial origin trace"):
        run.audit_receipts(receipts)


@pytest.mark.parametrize("field", ["p50_batch_ms", "IQR_batch_ms", "p95_batch_ms", "p50_ms_per_origin", "p95_ms_per_origin"])
def test_summary_quantiles_are_independently_recomputed_from_raw_trials(field):
    receipts = synthetic_receipts()
    receipts[0]["row"][field] += .01
    with pytest.raises(ValueError, match="trial quantile differs"):
        run.audit_receipts(receipts)


@pytest.mark.parametrize("field", ["peak_allocated_MiB", "peak_reserved_MiB"])
def test_allocator_peaks_use_timed_trials_without_warmup_contamination(field):
    receipts = synthetic_receipts()
    rows, _ = run.audit_receipts(receipts)
    assert rows[0][field] in (51., 83.)
    receipts[0]["row"][field] = 1000.
    with pytest.raises(ValueError, match="allocator peak"):
        run.audit_receipts(receipts)


@pytest.mark.parametrize("field,value", [("process_GPU_sampled_peak_MiB", 0.), ("process_GPU_sampling_status", "DEVICE_WIDE")])
def test_unmeasured_pid_memory_is_NA_instead_of_zero_or_device_total(field, value):
    receipts = synthetic_receipts()
    receipts[0]["row"][field] = value
    with pytest.raises(ValueError, match="per-PID memory"):
        run.audit_receipts(receipts)


def test_public_outputs_include_component_scope_and_replay_byte_for_byte(tmp_path):
    source, replay = tmp_path/"source", tmp_path/"replay"
    resources = synthetic_resources()
    validation = run.render_public(synthetic_receipts(), resources, source)
    assert validation["component_facets"] == 16 and validation["trials"] == 480
    assert validation["heads"] == 8 and validation["warmup_trials"] == 160 and validation["timed_trials"] == 320
    assert validation["whole_path_replacement"] is False
    assert validation["parent_latency_subtracted"] is False and validation["add_to_parent_latency_valid"] is False
    assert validation["new_training_updates"] == validation["F_calls"] == 0
    assert validation["new_V_forecasts"] is False and validation["T_access"] is False
    assert validation["resources"] == resources
    assert run.replay(source, replay) == validation
    names = {run.CSV_NAME, run.TRIALS_NAME, run.VALIDATION_NAME, run.INPUTS_NAME}
    assert {path.name for path in source.iterdir()} == names
    for name in names:
        assert (source/name).read_bytes() == (replay/name).read_bytes()
    with (source/run.CSV_NAME).open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16 and json.loads(rows[0]["fixed_first8_origins"]) == list(range(13000, 13008))
    assert rows[0]["process_GPU_sampled_peak_MiB"] == ""


@pytest.mark.parametrize("target", ["same", "descendant", "occupied"])
def test_replay_cannot_overwrite_source_or_write_into_occupied_output(tmp_path, target):
    source = tmp_path/"source"
    run.render_public(synthetic_receipts(), synthetic_resources(), source)
    output = source if target == "same" else source/"child" if target == "descendant" else tmp_path/"occupied"
    if target == "occupied":
        output.mkdir()
        (output/"preserve.txt").write_text("original", encoding="utf8")
    before = {path.name:path.read_bytes() for path in source.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="separate output|new or empty"):
        run.replay(source, output)
    assert {path.name:path.read_bytes() for path in source.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("name", [run.CSV_NAME, run.TRIALS_NAME, run.VALIDATION_NAME])
def test_replay_rejects_changed_public_table_or_validation_bytes(tmp_path, name):
    source = tmp_path/"source"
    run.render_public(synthetic_receipts(), synthetic_resources(), source)
    with (source/name).open("a", encoding="utf8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="table changed|validation changed"):
        run.replay(source, tmp_path/"replay")


def test_replay_reaudits_public_receipts_rather_than_trusting_input_json(tmp_path):
    source = tmp_path/"source"
    run.render_public(synthetic_receipts(), synthetic_resources(), source)
    path = source/run.INPUTS_NAME
    inputs = run.read(path)
    inputs["receipts"][0]["row"]["p50_batch_ms"] += .1
    rewrite_json(path, inputs)
    with pytest.raises(ValueError, match="trial quantile differs"):
        run.replay(source, tmp_path/"replay")


def test_full_prerequisite_gate_accepts_special_completed_chain_without_budget_reset(tmp_path, monkeypatch):
    ctx, scope, preflight = synthetic_gate_context(tmp_path)
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    result = run.gates(ctx, scope, preflight)
    assert result["GPU_base_seconds"] == 50.
    assert result["original_deadline_unix"] == 21700.
    assert result["prerequisite_bindings"]["primary_EVENTS"] == run.binding(ctx["cache"]/"EVENTS.jsonl", ctx)


@pytest.mark.parametrize("filename,status", [("TERMINAL_CORE_002.json", "RUNNING"), ("TERMINAL_SUPPLEMENT_CHAIN_001.json", "FAILED")])
def test_nonterminal_preconditions_block_before_component_journal_access(tmp_path, monkeypatch, filename, status):
    ctx, scope, preflight = synthetic_gate_context(tmp_path)
    rewrite_json(ctx["cache"]/filename, dict(status=status, exit_code=0))
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    monkeypatch.setattr(run, "_events", lambda _:(_ for _ in ()).throw(AssertionError("component private journal accessed")))
    with pytest.raises(ValueError, match="Core and supplement"):
        run.gates(ctx, scope, preflight)


@pytest.mark.parametrize("status", ["PASS_MAIN_CP0_NOT_YET_PRESENT", "PARTIAL", "FAIL"])
def test_incomplete_independent_validation_never_admits_component_measurement(tmp_path, monkeypatch, status):
    ctx, scope, preflight = synthetic_gate_context(tmp_path)
    path = ctx["report"]/"INDEPENDENT_FINAL_AUDIT_001.json"
    value = run.read(path)
    value["status"] = status
    rewrite_json(path, value)
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    with pytest.raises(ValueError, match="independent\\+CP0"):
        run.gates(ctx, scope, preflight)


def test_source_or_primary_events_changes_fail_before_measurement(tmp_path, monkeypatch):
    ctx, scope, preflight = synthetic_gate_context(tmp_path)
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    frozen = next(iter(run.read(scope)["source_files"]))
    path = ctx["repo"]/frozen
    original = path.read_bytes()
    path.write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="Frozen source changed"):
        run.gates(ctx, scope, preflight)
    path.write_bytes(original)
    with (ctx["cache"]/"EVENTS.jsonl").open("a", encoding="utf8") as stream:
        stream.write('{"changed":true}\n')
    with pytest.raises(ValueError, match="Primary EVENTS"):
        run.gates(ctx, scope, preflight)


def test_original_wall_deadline_is_enforced_without_extension(tmp_path, monkeypatch):
    ctx, scope, preflight = synthetic_gate_context(tmp_path)
    monkeypatch.setattr(run.time, "time", lambda:21700.)
    with pytest.raises(TimeoutError, match="Original core deadline exhausted"):
        run.gates(ctx, scope, preflight)


def test_original_gpu_reservation_includes_owned_worker_time(tmp_path, monkeypatch):
    ctx, scope, preflight = synthetic_gate_context(tmp_path)
    journal = run._journal_path(ctx)
    journal.parent.mkdir(parents=True)
    events = [dict(status="STARTED", session_id="old-component", time_unix=100.),
        dict(status="COMPLETE", session_id="old-component", time_unix=14450., wall_seconds=14350.)]
    journal.write_text("\n".join(json.dumps(event) for event in events), encoding="utf8")
    monkeypatch.setattr(run.time, "time", lambda:15000.)
    with pytest.raises(TimeoutError, match="Original GPU process-wall reservation exhausted"):
        run.gates(ctx, scope, preflight)


def test_binding_resolver_rejects_escape_and_changed_bytes(tmp_path):
    ctx = run.context(tmp_path/"asset", repo=tmp_path/"repo")
    ctx["report"].mkdir(parents=True)
    path = ctx["report"]/"synthetic.json"
    run.write_new(path, dict(synthetic=True))
    value = run.binding(path, ctx)
    assert run.resolve(value, ctx) == path
    rewrite_json(path, dict(synthetic=False))
    with pytest.raises(ValueError, match="SHA256 changed"):
        run.resolve(value, ctx)
    with pytest.raises(ValueError, match="escapes registered root"):
        run.reference_path("report://../outside.json", ctx)


def test_collect_blocks_before_input_preparation_when_main_is_not_terminal(tmp_path, monkeypatch):
    ctx, scope, preflight = synthetic_gate_context(tmp_path)
    rewrite_json(ctx["cache"]/"TERMINAL_CORE_002.json", dict(status="RUNNING", exit_code=0))
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    monkeypatch.setattr(run, "prepare_inputs", lambda *args:(_ for _ in ()).throw(AssertionError("private preparation started")))
    with pytest.raises(ValueError, match="Core and supplement"):
        run.collect(ctx, scope, preflight)


def test_unclosed_owned_session_blocks_new_collection_before_input_preparation(tmp_path, monkeypatch):
    ctx, scope, preflight = synthetic_gate_context(tmp_path)
    journal = run._journal_path(ctx)
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps(dict(status="STARTED", session_id="still-owned", time_unix=900.))+"\n", encoding="utf8")
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    monkeypatch.setattr(run, "prepare_inputs", lambda *args:(_ for _ in ()).throw(AssertionError("private preparation started")))
    with pytest.raises(ValueError, match="Unclosed owned-worker"):
        run.collect(ctx, scope, preflight)


@pytest.mark.parametrize("blocked_by", ["terminal", "prerequisites"])
def test_worker_checks_admission_before_numerical_imports_or_private_arrays(tmp_path, monkeypatch, blocked_by):
    ctx, scope, preflight = synthetic_gate_context(tmp_path)
    request = ctx["cache"]/"synthetic_worker_request.json"
    run.write_new(request, dict(asset_root=str(ctx["asset"]), repo_root=str(ctx["repo"]),
        scope_binding=run.binding(scope, ctx), preflight_binding=run.binding(preflight, ctx), prerequisite_bindings={}))
    if blocked_by == "terminal":
        rewrite_json(ctx["cache"]/"TERMINAL_CORE_002.json", dict(status="FAILED", exit_code=1))
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("numpy", "torch", "psutil"):
            raise AssertionError("numerical runtime imported before admission rejection")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    expected = "Core and supplement" if blocked_by == "terminal" else "prerequisites changed"
    with pytest.raises(ValueError, match=expected):
        run.worker(request, ctx["cache"]/"never_created.json")
    assert not (ctx["cache"]/"never_created.json").exists()


@pytest.mark.parametrize("mutation", ["dataset", "mode", "seed", "batch", "control", "scope", "head", "core_deadline", "worker_deadline"])
def test_worker_rejects_unregistered_metadata_or_extended_deadlines_before_numerical_imports(tmp_path, monkeypatch, mutation):
    ctx, scope, preflight = synthetic_gate_context(tmp_path)
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    proof = run.gates(ctx, scope, preflight)
    scope_binding = run.binding(scope, ctx)
    head_binding = dict(reference="cache://synthetic_head.pt", sha256="a"*64)
    request = dict(asset_root=str(ctx["asset"]), repo_root=str(ctx["repo"]),
        scope_binding=scope_binding, preflight_binding=run.binding(preflight, ctx),
        prerequisite_bindings=proof["prerequisite_bindings"], head_binding=head_binding,
        metadata=dict(dataset="jena", mode="conditional", seed=92601, control=False,
            scope_binding=scope_binding, head_binding=head_binding),
        batch_size=1, original_deadline_unix=21700., worker_deadline_unix=2000.)
    if mutation in ("dataset", "mode", "seed"):
        request["metadata"][mutation] = dict(dataset="alien", mode="alien", seed=1)[mutation]
    elif mutation == "batch":
        request["batch_size"] = 4
    elif mutation == "control":
        request["metadata"]["control"] = True
    elif mutation in ("scope", "head"):
        request["metadata"][mutation+"_binding"] = dict(reference="cache://alien", sha256="f"*64)
    elif mutation == "core_deadline":
        request["original_deadline_unix"] += 1.
    else:
        request["worker_deadline_unix"] = 21701.
    request_path = ctx["cache"]/"synthetic_worker_request.json"
    run.write_new(request_path, request)
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("numpy", "torch", "psutil"):
            raise AssertionError("numerical runtime imported before metadata admission rejection")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="admitted frozen head"):
        run.worker(request_path, ctx["cache"]/"never_created.json")


@pytest.mark.parametrize("worker_deadline,core_deadline", [(1000., 2000.), (2000., 1000.)])
def test_owned_worker_deadline_enforces_both_immutable_core_and_residual_gpu_limits(monkeypatch, worker_deadline, core_deadline):
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    with pytest.raises(TimeoutError, match="Original wall/GPU reservation"):
        run._worker_deadline(dict(worker_deadline_unix=worker_deadline, original_deadline_unix=core_deadline))
    run._worker_deadline(dict(worker_deadline_unix=1001., original_deadline_unix=1002.))


@pytest.mark.parametrize("exit_code", [0, 1])
def test_exited_wrapper_closes_ledger_only_after_actual_worker_confirmation_or_cleanup(tmp_path, monkeypatch, exit_code):
    ctx = run.context(tmp_path/"asset", repo=tmp_path)
    attempt = ctx["cache"]/"synthetic_attempt"
    attempt.mkdir(parents=True)
    request_path, output_path = attempt/"request.json", attempt/"output.json"
    request = dict(worker_deadline_unix=1000., collector_identity=dict(pid=900, create_time=90.))
    run.write_new(request_path, request)
    launches = []
    class SyntheticProcess:
        pid = 4242
        def wait(self, timeout):
            assert timeout > 0
            return exit_code
        def poll(self):
            return exit_code
    def fake_popen(command, **kwargs):
        launches.append((command, kwargs))
        return SyntheticProcess()
    ticks = iter((100., 101., 110.))
    monkeypatch.setattr(run.time, "time", lambda:next(ticks))
    monkeypatch.setattr(run.subprocess, "Popen", fake_popen)
    monkeypatch.setitem(run.sys.modules, "psutil", SimpleNamespace(Process=lambda _:SimpleNamespace(create_time=lambda:42.)))
    confirmed, stopped = [], []
    monkeypatch.setattr(run, "_confirm_owned_exit", lambda current, *args:confirmed.append((current, args)))
    monkeypatch.setattr(run, "_stop_owned_tree", lambda current, *args:stopped.append((current, args)))
    if exit_code:
        with pytest.raises(RuntimeError, match="Owned component worker failed"):
            run._spawn_owned(ctx, {}, request, request_path, output_path)
    else:
        run._spawn_owned(ctx, {}, request, request_path, output_path)
    assert len(launches) == 1
    assert len(confirmed) == (0 if exit_code else 1)
    assert len(stopped) == (1 if exit_code else 0)
    checked = stopped[0] if exit_code else confirmed[0]
    assert checked[0].poll() == exit_code
    assert checked[1][-2] == dict(pid=4242, create_time=42.)
    assert launches[0][0][:4] == [run.sys.executable, "-u", "-m", "research."+run.NAME+".run"]
    events = run._events(ctx)
    assert [event["status"] for event in events] == ["STARTED", "FAILED" if exit_code else "COMPLETE"]
    assert events[0]["session_id"] == events[1]["session_id"]
    assert events[1]["wall_seconds"] == 10. and events[1]["worker_pid"] == 4242
    assert run.owned_wall(events, 120.) == dict(seconds=10., closed_sessions=1, open_sessions=0)


def test_mock_owned_process_timeout_terminates_and_accounts_failed_attempt(tmp_path, monkeypatch):
    ctx = run.context(tmp_path/"asset", repo=tmp_path)
    attempt = ctx["cache"]/"synthetic_attempt"
    attempt.mkdir(parents=True)
    request_path, output_path = attempt/"request.json", attempt/"output.json"
    request = dict(worker_deadline_unix=1000., collector_identity=dict(pid=900, create_time=90.))
    run.write_new(request_path, request)
    class SyntheticProcess:
        pid, terminated, waits = 4242, False, 0
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise run.subprocess.TimeoutExpired("synthetic", timeout)
            return -15
        def terminate(self):
            self.terminated = True
        def poll(self):
            return -15 if self.terminated else None
    process = SyntheticProcess()
    ticks = iter((100., 101., 110.))
    monkeypatch.setattr(run.time, "time", lambda:next(ticks))
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs:process)
    monkeypatch.setitem(run.sys.modules, "psutil", SimpleNamespace(Process=lambda _:SimpleNamespace(create_time=lambda:42.)))
    stopped = []
    def fake_stop(current, *args):
        stopped.append(current)
        current.terminate()
        current.wait(timeout=5)
    monkeypatch.setattr(run, "_stop_owned_tree", fake_stop)
    with pytest.raises(TimeoutError, match="exhausted original reservation"):
        run._spawn_owned(ctx, {}, request, request_path, output_path)
    assert process.terminated is True and stopped and all(current is process for current in stopped)
    assert process.waits == 1+len(stopped)
    events = run._events(ctx)
    assert events[-1]["status"] == "FAILED"
    assert run.owned_wall(events, 120.) == dict(seconds=10., closed_sessions=1, open_sessions=0)


@pytest.mark.parametrize("redirected", [False, True])
def test_owned_lineage_accepts_direct_worker_or_windows_venv_redirector(redirected):
    receipts = [synthetic_lineage(receipt, redirected) for receipt in synthetic_receipts()]
    rows, _ = run.audit_receipts(receipts)
    assert len(rows) == 16
    for receipt in receipts:
        assert receipt["launch_proof"]["collector"] in receipt["process_lineage"]["ancestors"]
        assert (receipt["launch_proof"]["launcher"] == receipt["process_lineage"]["worker"]) is (not redirected)


@pytest.mark.parametrize("mutation", ["worker_pid", "worker_creation", "alien_launcher", "missing_collector", "request_collector", "self_collector", "reused_launcher_pid"])
def test_alien_reused_or_mismatched_process_lineage_is_rejected(mutation):
    receipt = synthetic_lineage(synthetic_receipts()[0], redirected=True)
    if mutation == "worker_pid":
        receipt["process_lineage"]["worker"]["pid"] += 1
    elif mutation == "worker_creation":
        receipt["process_lineage"]["worker"]["create_time"] += 1
    elif mutation == "alien_launcher":
        receipt["launch_proof"]["launcher"] = dict(pid=99999, create_time=1.)
    elif mutation == "missing_collector":
        receipt["process_lineage"]["ancestors"] = [receipt["launch_proof"]["launcher"]]
    elif mutation == "request_collector":
        receipt["request_collector_identity"] = dict(pid=901, create_time=90.)
    elif mutation == "self_collector":
        receipt["launch_proof"]["collector"] = receipt["process_lineage"]["worker"]
        receipt["request_collector_identity"] = receipt["process_lineage"]["worker"]
        receipt["process_lineage"]["ancestors"].append(receipt["process_lineage"]["worker"])
    else:
        receipt["launch_proof"]["launcher"] = dict(receipt["launch_proof"]["launcher"], create_time=1.)
    with pytest.raises(ValueError, match="process lineage|descend"):
        run.validate_lineage(receipt)


@pytest.mark.parametrize("mutation", ["total_drift", "owned_drift", "open_session", "deadline_overrun", "gpu_overrun", "primary_events_changed"])
def test_public_resource_ledger_is_recomputed_and_must_preserve_original_limits(tmp_path, mutation):
    resources = synthetic_resources()
    if mutation == "total_drift":
        resources["GPU_total_process_wall_upper_bound_seconds"] += 1.
    elif mutation == "owned_drift":
        resources["component_owned_worker_wall_seconds"] += 1.
    elif mutation == "open_session":
        resources["owned_events"].pop()
        resources["owned_sessions"] = dict(seconds=84., open_sessions=1, closed_sessions=15)
        resources["component_owned_worker_wall_seconds"] = 84.
        resources["GPU_total_process_wall_upper_bound_seconds"] = 134.
    elif mutation == "deadline_overrun":
        resources["finished_unix"] = resources["original_deadline_unix"]+1.
    elif mutation == "gpu_overrun":
        resources["GPU_base_seconds"] = 14400.
        resources["GPU_total_process_wall_upper_bound_seconds"] = 14416.
    else:
        resources["primary_EVENTS_after"] = dict(reference="cache://changed_EVENTS.jsonl", sha256="f"*64)
    with pytest.raises(ValueError):
        run.render_public(synthetic_receipts(), resources, tmp_path/"public")


def synthetic_started_attempt(tmp_path):
    ctx = run.context(tmp_path/"asset", repo=tmp_path)
    attempt = ctx["cache"]/"synthetic_attempt"
    attempt.mkdir(parents=True)
    request_path, output_path = attempt/"request.json", attempt/"output.json"
    collector = dict(pid=900, create_time=90.)
    launcher, worker = dict(pid=4242, create_time=42.), dict(pid=4343, create_time=43.)
    request = dict(worker_deadline_unix=1000., collector_identity=collector)
    run.write_new(request_path, request)
    run.write_new(output_path.with_suffix(".started.json"), dict(request_binding=run.binding(request_path, ctx),
        process_lineage=dict(worker=worker, ancestors=[launcher, collector])))
    return ctx, request, request_path, output_path, launcher


@pytest.mark.parametrize("exit_code", [0, 1])
def test_unverified_actual_worker_start_proof_preserves_open_reservation(tmp_path, monkeypatch, exit_code):
    ctx, request, request_path, output_path, _ = synthetic_started_attempt(tmp_path)
    startup_path = output_path.with_suffix(".started.json")
    startup = run.read(startup_path)
    startup["request_binding"]["sha256"] = "f"*64
    rewrite_json(startup_path, startup)
    class SyntheticProcess:
        pid = 4242
        def wait(self, timeout):
            return exit_code
        def poll(self):
            return exit_code
    class SyntheticParent:
        pid = 4242
        def create_time(self):
            return 42.
        def children(self, recursive):
            return []
        def terminate(self):
            raise AssertionError("unverified worker proof must not authorize termination")
    class SyntheticNoSuchProcess(Exception):
        pass
    monkeypatch.setattr(run.time, "time", lambda:100.)
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs:SyntheticProcess())
    monkeypatch.setitem(run.sys.modules, "psutil", SimpleNamespace(Process=lambda _:SyntheticParent(), NoSuchProcess=SyntheticNoSuchProcess))
    with pytest.raises(ValueError, match="Unverified started worker lineage"):
        run._spawn_owned(ctx, {}, request, request_path, output_path)
    events = run._events(ctx)
    assert [event["status"] for event in events] == ["STARTED"]
    assert run.owned_wall(events, 120.) == dict(seconds=20., closed_sessions=0, open_sessions=1)
    assert not output_path.exists()


@pytest.mark.parametrize("worker_state", ["missing", "reused_pid", "exits_after_wrapper"])
def test_successful_wrapper_confirmation_uses_actual_worker_pid_creation_and_exit(tmp_path, monkeypatch, worker_state):
    ctx, request, request_path, output_path, launcher = synthetic_started_attempt(tmp_path)
    class SyntheticNoSuchProcess(Exception):
        pass
    child = SimpleNamespace(pid=4343, create_time=lambda:44. if worker_state == "reused_pid" else 43.)
    queried, waited = [], []
    def fake_process(pid):
        queried.append(pid)
        if worker_state == "missing":
            raise SyntheticNoSuchProcess()
        return child
    def fake_wait(children, timeout):
        waited.append((children, timeout))
        return children, []
    monkeypatch.setitem(run.sys.modules, "psutil", SimpleNamespace(Process=fake_process, NoSuchProcess=SyntheticNoSuchProcess, wait_procs=fake_wait))
    monkeypatch.setattr(run.time, "time", lambda:110.)
    monkeypatch.setattr(run, "_stop_owned_tree", lambda *args:(_ for _ in ()).throw(AssertionError("exited or reused worker must not be terminated")))
    assert run._confirm_owned_exit(object(), request, request_path, output_path, launcher, ctx) is True
    assert queried == [4343]
    assert len(waited) == (1 if worker_state == "exits_after_wrapper" else 0)
    if waited:
        assert waited == [([child], 890.)]


def test_live_actual_worker_after_wrapper_waits_only_to_deadline_then_requests_verified_cleanup(tmp_path, monkeypatch):
    ctx, request, request_path, output_path, launcher = synthetic_started_attempt(tmp_path)
    child = SimpleNamespace(pid=4343, create_time=lambda:43.)
    stopped, waited = [], []
    def fake_wait(children, timeout):
        waited.append((children, timeout))
        return [], children
    monkeypatch.setitem(run.sys.modules, "psutil", SimpleNamespace(Process=lambda _:child, NoSuchProcess=RuntimeError, wait_procs=fake_wait))
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    monkeypatch.setattr(run, "_stop_owned_tree", lambda *args:stopped.append(args))
    process = object()
    assert run._confirm_owned_exit(process, request, request_path, output_path, launcher, ctx) is False
    assert waited == [([child], 0.)]
    assert stopped == [(process, request, request_path, output_path, launcher, ctx)]


@pytest.mark.parametrize("actual_pid_reused", [False, True])
def test_failed_exited_wrapper_cleanup_reaches_proved_actual_child_and_preserves_other_processes(tmp_path, monkeypatch, actual_pid_reused):
    ctx, request, request_path, output_path, launcher = synthetic_started_attempt(tmp_path)
    class SyntheticNoSuchProcess(Exception):
        pass
    terminated, killed, queried, waits = [], [], [], []
    class SyntheticChild:
        def __init__(self, pid, created):
            self.pid, self.created = pid, created
        def create_time(self):
            return self.created
        def children(self, recursive):
            assert recursive is True
            return [descendant] if self.pid == 4343 else []
        def terminate(self):
            terminated.append(self.pid)
        def kill(self):
            killed.append(self.pid)
    child = SyntheticChild(4343, 44. if actual_pid_reused else 43.)
    descendant, unrelated = SyntheticChild(4344, 43.1), SyntheticChild(9999, 99.)
    def fake_process(pid):
        queried.append(pid)
        if pid == 4242:
            raise SyntheticNoSuchProcess()
        return {4343:child, 4344:descendant, 9999:unrelated}[pid]
    def fake_wait(children, timeout):
        waits.append(([item.pid for item in children], timeout))
        return children, []
    process_waits = []
    process = SimpleNamespace(pid=4242, poll=lambda:1, wait=lambda timeout:process_waits.append(timeout))
    monkeypatch.setitem(run.sys.modules, "psutil", SimpleNamespace(Process=fake_process, NoSuchProcess=SyntheticNoSuchProcess, wait_procs=fake_wait))
    run._stop_owned_tree(process, request, request_path, output_path, launcher, ctx)
    assert queried == [4242, 4343]
    assert terminated == ([] if actual_pid_reused else [4343, 4344])
    assert killed == [] and 9999 not in terminated
    assert waits[-1] == ([], 3) and process_waits == [3]


def test_verified_actual_worker_stopped_after_successful_wrapper_closes_failed_ledger_once(tmp_path, monkeypatch):
    ctx, request, request_path, output_path, _ = synthetic_started_attempt(tmp_path)
    process = SimpleNamespace(pid=4242, poll=lambda:0, wait=lambda timeout:0)
    ticks = iter((100., 101., 110.))
    monkeypatch.setattr(run.time, "time", lambda:next(ticks))
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs:process)
    monkeypatch.setitem(run.sys.modules, "psutil", SimpleNamespace(Process=lambda _:SimpleNamespace(create_time=lambda:42.)))
    monkeypatch.setattr(run, "_confirm_owned_exit", lambda *args:False)
    with pytest.raises(TimeoutError, match="outlived its successful wrapper"):
        run._spawn_owned(ctx, {}, request, request_path, output_path)
    events = run._events(ctx)
    assert [event["status"] for event in events] == ["STARTED", "FAILED"]
    assert run.owned_wall(events, 120.) == dict(seconds=10., closed_sessions=1, open_sessions=0)


def test_missing_start_proof_and_already_dead_wrapper_leave_owned_reservation_open(tmp_path, monkeypatch):
    ctx, request, request_path, output_path, _ = synthetic_started_attempt(tmp_path)
    output_path.with_suffix(".started.json").unlink()
    class SyntheticNoSuchProcess(Exception):
        pass
    queried, waited = [], []
    def fake_process(pid):
        queried.append(pid)
        if len(queried) == 1:
            return SimpleNamespace(pid=pid, create_time=lambda:42.)
        raise SyntheticNoSuchProcess()
    process = SimpleNamespace(pid=4242, poll=lambda:1, wait=lambda timeout:waited.append(timeout) or 1)
    monkeypatch.setattr(run.time, "time", lambda:100.)
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs:process)
    monkeypatch.setitem(run.sys.modules, "psutil", SimpleNamespace(Process=fake_process, NoSuchProcess=SyntheticNoSuchProcess,
        wait_procs=lambda *args, **kwargs:(_ for _ in ()).throw(AssertionError("unproved tree must not be waited on or closed"))))
    with pytest.raises(RuntimeError, match="No started-worker proof or live owned tree"):
        run._spawn_owned(ctx, {}, request, request_path, output_path)
    assert queried == [4242, 4242] and waited == [900.]
    events = run._events(ctx)
    assert [event["status"] for event in events] == ["STARTED"]
    assert run.owned_wall(events, 120.) == dict(seconds=20., closed_sessions=0, open_sessions=1)
    assert not output_path.exists()


@pytest.mark.parametrize("mutation", ["missing_binding", "failed_only", "duplicate_completion"])
def test_every_measured_receipt_is_charged_to_exactly_one_completed_request(tmp_path, mutation):
    resources = synthetic_resources()
    if mutation == "missing_binding":
        del resources["owned_events"][1]["request_binding"]
    elif mutation == "failed_only":
        resources["owned_events"][1]["status"] = "FAILED"
    else:
        resources["owned_events"][2]["request_binding"] = resources["owned_events"][1]["request_binding"]
        resources["owned_events"][3]["request_binding"] = resources["owned_events"][1]["request_binding"]
    assert run.audit_resources(resources) == dict(seconds=16., open_sessions=0, closed_sessions=16)
    with pytest.raises(ValueError, match="exactly one completed owned-worker reservation"):
        run.render_public(synthetic_receipts(), resources, tmp_path/"public")


def synthetic_recovery_context(tmp_path):
    ctx, legacy_scope_path, legacy_preflight_path = synthetic_gate_context(tmp_path)
    original = run.read(legacy_scope_path)
    rewrite_json(ctx["cache"]/"TERMINAL_CORE_002.json", dict(status="FAILED", exit_code=1))
    rewrite_json(ctx["cache"]/"TERMINAL_SUPPLEMENT_CHAIN_001.json", dict(status="FAILED", exit_code=1))
    original_path = ctx["report"]/"HEAD_COMPONENT_COST_SCOPE_001.json"
    run.write_new(original_path, original)
    recovered = copy.deepcopy(original)
    recovered["replaces_scope_binding"] = run.binding(original_path, ctx)
    recovered["required_terminal_references"] = dict(core="cache://TERMINAL_COMPLETION_RECOVERY_002.json",
        chain="cache://TERMINAL_SUPPLEMENT_CHAIN_RECOVERY_002.json")
    recovered["required_public_references"]["primary_validation"] = "report://RECOVERY_VALIDATION_REPORT_002.json"
    recovered_path = ctx["report"]/"HEAD_COMPONENT_COST_SCOPE_RECOVERY_002.json"
    run.write_new(recovered_path, recovered)
    preflight = run.read(legacy_preflight_path)
    preflight["scope_binding"] = run.binding(recovered_path, ctx)
    preflight_path = ctx["report"]/"SYNTHETIC_RECOVERY_PREFLIGHT.json"
    run.write_new(preflight_path, preflight)
    run.write_new(ctx["cache"]/"TERMINAL_COMPLETION_RECOVERY_002.json", dict(status="COMPLETE", exit_code=0))
    run.write_new(ctx["cache"]/"TERMINAL_SUPPLEMENT_CHAIN_RECOVERY_002.json", dict(
        status="SUPPLEMENT_STAGES_COMPLETE_PENDING_ROOT_REVIEW_AND_PUBLICATION", exit_code=0))
    run.write_new(ctx["report"]/"RECOVERY_VALIDATION_REPORT_002.json", run.read(ctx["report"]/"VALIDATION_REPORT.json"))
    return ctx, recovered_path, preflight_path, original_path


def mock_completion_lineage(monkeypatch, ctx, *, terminal_path=None, validation_path=None, failure=False):
    package_name = "research.raw_context_multiscale_completion_recovery_20261019_v1"
    package = ModuleType(package_name)
    package.__path__ = []
    module = ModuleType(package_name+".revision2")
    calls = []
    def validate_completion(active):
        calls.append(active)
        if failure:
            raise ValueError("synthetic transitive completion lineage failed")
        return dict(terminal_path=ctx["cache"]/"TERMINAL_COMPLETION_RECOVERY_002.json" if terminal_path is None else terminal_path,
            validation_path=ctx["report"]/"RECOVERY_VALIDATION_REPORT_002.json" if validation_path is None else validation_path)
    module.validate_completion = validate_completion
    package.revision2 = module
    monkeypatch.setitem(run.sys.modules, package_name, package)
    monkeypatch.setitem(run.sys.modules, package_name+".revision2", module)
    return calls


def test_legacy_completion_scope_does_not_invoke_recovery_or_accept_failed_001(tmp_path, monkeypatch):
    ctx, _, _, _ = synthetic_recovery_context(tmp_path)
    scope_path = ctx["report"]/"SYNTHETIC_COMPONENT_SCOPE.json"
    scope = run.read(scope_path)
    calls = mock_completion_lineage(monkeypatch, ctx, failure=True)
    prerequisites = dict(synthetic="unchanged")
    run._completion_references(ctx, scope_path, scope, prerequisites)
    assert prerequisites == dict(synthetic="unchanged") and calls == []
    rewrite_json(ctx["cache"]/"TERMINAL_CORE_002.json", dict(status="COMPLETE", exit_code=0))
    assert run.read(ctx["cache"]/"TERMINAL_SUPPLEMENT_CHAIN_001.json")["status"] == "FAILED"
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    with pytest.raises(ValueError, match="Core and supplement"):
        run.gates(ctx, scope_path, ctx["report"]/"SYNTHETIC_COMPONENT_PREFLIGHT.json")
    assert calls == []


def test_explicit_recovery_scope_preserves_contract_and_binds_exact_replaced_scope(tmp_path, monkeypatch):
    ctx, scope_path, preflight, original_path = synthetic_recovery_context(tmp_path)
    calls = mock_completion_lineage(monkeypatch, ctx)
    preserved = {path:path.read_bytes() for path in (original_path, ctx["cache"]/"TERMINAL_CORE_002.json",
        ctx["cache"]/"TERMINAL_SUPPLEMENT_CHAIN_001.json")}
    monkeypatch.setattr(run.time, "time", lambda:1000.)
    proof = run.gates(ctx, scope_path, preflight)
    assert calls == [ctx]
    assert proof["prerequisite_bindings"]["replaced_head_scope"] == run.binding(original_path, ctx)
    assert proof["prerequisite_bindings"]["terminal_core"]["reference"] == "cache://TERMINAL_COMPLETION_RECOVERY_002.json"
    assert proof["prerequisite_bindings"]["terminal_chain"]["reference"] == "cache://TERMINAL_SUPPLEMENT_CHAIN_RECOVERY_002.json"
    assert proof["prerequisite_bindings"]["primary_validation"]["reference"] == "report://RECOVERY_VALIDATION_REPORT_002.json"
    assert proof["original_deadline_unix"] == 21700. and proof["GPU_base_seconds"] == 50.
    assert {path:path.read_bytes() for path in preserved} == preserved


@pytest.mark.parametrize("field", ["status", "source_files", "primary_plan_binding", "model_seal_binding",
    "original_execution_start_binding", "original_deadline_unix", "limits", "GPU_base_ledger_reference",
    "component_protocol", "scientific_contract_changed", "selection_changes"])
def test_recovery_cannot_change_scientific_source_protocol_or_original_budget_contracts(tmp_path, monkeypatch, field):
    ctx, scope_path, _, _ = synthetic_recovery_context(tmp_path)
    scope = run.read(scope_path)
    if field == "source_files":
        scope[field][next(iter(scope[field]))] = "f"*64
    elif field.endswith("_binding"):
        scope[field] = dict(reference="cache://changed_contract.json", sha256="f"*64)
    elif field == "original_deadline_unix":
        scope[field] += 1.
    elif field == "limits":
        scope[field]["GPU_reservation_seconds"] += 1
    elif field == "component_protocol":
        scope[field]["warmup"] = 11
    elif field in ("scientific_contract_changed", "selection_changes"):
        scope[field] = True
    else:
        scope[field] = "CHANGED"
    calls = mock_completion_lineage(monkeypatch, ctx)
    prerequisites = {}
    with pytest.raises(ValueError, match="cannot change scientific/protocol/source/original budget"):
        run._completion_references(ctx, scope_path, scope, prerequisites)
    assert calls == [] and prerequisites == {}


@pytest.mark.parametrize("mutation", ["original_sha", "original_alias", "new_alias"])
def test_recovery_requires_canonical_scope_names_and_exact_original_sha(tmp_path, monkeypatch, mutation):
    ctx, scope_path, _, original_path = synthetic_recovery_context(tmp_path)
    scope = run.read(scope_path)
    if mutation == "original_sha":
        scope["replaces_scope_binding"]["sha256"] = "f"*64
    elif mutation == "original_alias":
        alias = ctx["report"]/"SYNTHETIC_OLD_ALIAS.json"
        run.write_new(alias, run.read(original_path))
        scope["replaces_scope_binding"] = run.binding(alias, ctx)
    else:
        scope_path = ctx["report"]/"SYNTHETIC_NEW_ALIAS.json"
    calls = mock_completion_lineage(monkeypatch, ctx)
    with pytest.raises(ValueError, match="SHA256 changed|exact original head scope"):
        run._completion_references(ctx, scope_path, scope, {})
    assert calls == []


@pytest.mark.parametrize("field", ["Mcal_validation", "final_context", "primary_validation"])
def test_recovery_cannot_replace_other_public_evidence_or_choose_another_primary_pass(tmp_path, monkeypatch, field):
    ctx, scope_path, _, _ = synthetic_recovery_context(tmp_path)
    scope = run.read(scope_path)
    scope["required_public_references"][field] = "report://UNREGISTERED_PASS.json"
    calls = mock_completion_lineage(monkeypatch, ctx)
    with pytest.raises(ValueError, match="explicit recovered primary PASS and terminal lineage"):
        run._completion_references(ctx, scope_path, scope, {})
    assert calls == []


@pytest.mark.parametrize("mutation", ["legacy_failed_core", "failed_recovery_v1", "wrong_chain", "extra", "missing"])
def test_recovery_terminal_replacements_are_exact_instead_of_accepting_legacy_failure(tmp_path, monkeypatch, mutation):
    ctx, scope_path, _, _ = synthetic_recovery_context(tmp_path)
    scope = run.read(scope_path)
    if mutation == "legacy_failed_core":
        scope["required_terminal_references"]["core"] = "cache://TERMINAL_CORE_001.json"
    elif mutation == "failed_recovery_v1":
        scope["required_terminal_references"]["core"] = "cache://TERMINAL_COMPLETION_RECOVERY_001.json"
    elif mutation == "wrong_chain":
        scope["required_terminal_references"]["chain"] = "cache://TERMINAL_SUPPLEMENT_CHAIN_001.json"
    elif mutation == "extra":
        scope["required_terminal_references"]["other"] = "cache://OTHER_COMPLETE.json"
    else:
        del scope["required_terminal_references"]["chain"]
    calls = mock_completion_lineage(monkeypatch, ctx)
    with pytest.raises(ValueError, match="explicit recovered primary PASS and terminal lineage"):
        run._completion_references(ctx, scope_path, scope, {})
    assert calls == []


@pytest.mark.parametrize("field", ["terminal_path", "validation_path"])
def test_transitive_recovery_audit_must_return_exact_registered_completion_paths(tmp_path, monkeypatch, field):
    ctx, scope_path, _, _ = synthetic_recovery_context(tmp_path)
    calls = mock_completion_lineage(monkeypatch, ctx, **{field:ctx["cache"]/"UNREGISTERED_COMPLETE.json"})
    prerequisites = {}
    with pytest.raises(ValueError, match="outside the registered replacement lineage"):
        run._completion_references(ctx, scope_path, run.read(scope_path), prerequisites)
    assert calls == [ctx] and prerequisites == {}


def test_failed_transitive_recovery_audit_does_not_add_replacement_approval(tmp_path, monkeypatch):
    ctx, scope_path, _, _ = synthetic_recovery_context(tmp_path)
    calls = mock_completion_lineage(monkeypatch, ctx, failure=True)
    prerequisites = {}
    with pytest.raises(ValueError, match="synthetic transitive completion lineage failed"):
        run._completion_references(ctx, scope_path, run.read(scope_path), prerequisites)
    assert calls == [ctx] and prerequisites == {}


@pytest.mark.parametrize("mutation", ["recovery_filename", "core", "chain", "primary_validation", "all_replacements"])
def test_replacement_paths_cannot_bypass_recovery_by_omitting_replaces_scope_binding(tmp_path, monkeypatch, mutation):
    ctx, recovered_path, _, original_path = synthetic_recovery_context(tmp_path)
    scope = run.read(original_path)
    scope_path = ctx["report"]/"SYNTHETIC_COMPONENT_SCOPE.json"
    if mutation == "recovery_filename":
        scope_path = recovered_path
    elif mutation in ("core", "chain"):
        scope["required_terminal_references"][mutation] = run.read(recovered_path)["required_terminal_references"][mutation]
    elif mutation == "primary_validation":
        scope["required_public_references"][mutation] = "report://RECOVERY_VALIDATION_REPORT_002.json"
    else:
        scope = run.read(recovered_path)
        del scope["replaces_scope_binding"]
    calls = mock_completion_lineage(monkeypatch, ctx)
    prerequisites = {}
    with pytest.raises(ValueError, match="require explicit registered completion recovery"):
        run._completion_references(ctx, scope_path, scope, prerequisites)
    assert calls == [] and prerequisites == {}
