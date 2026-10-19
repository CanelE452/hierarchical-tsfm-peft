"""Synthetic CPU checks for the separately registered zero-update diagnostic."""
import copy
import importlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch
from torch import nn

from research.b2_revin_20261018_v1.training_jobs import _state_fingerprint
from research.raw_context_multiscale_20261018_v1 import common, models, training, run
from research.raw_context_multiscale_supplement_20261019_v1 import cp0


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")
    return path


def final_rows(mode="B"):
    return [dict(dataset=d, mode=m, seed=s, run_id=f"{m}_{d}_s{s}", reused=m == "S",
                 route_spec=dict(family="B2", run_id=f"{m}_{d}_s{s}"),
                 search=dict(status="COMPLETE", V_used=False, selection_split="A_stop",
                             spec=dict(dataset=d, seed=s, run_id=f"{m}_{d}_s{s}", family="B2",
                                       latent_k=common.EXPECTED[d]["K"],
                                       scientific_identity=dict(raw_context_mode=m))))
            for d, m, s in sorted(cp0.expected_keys(mode))]


@pytest.fixture
def registered(tmp_path, monkeypatch):
    ctx = dict(repo=tmp_path, report=tmp_path / "reports" / common.NAME,
               cache=tmp_path / ".cache" / common.NAME, asset=tmp_path / "asset")
    primary = {}
    for number in range(13):
        name = f"research/{common.NAME}/p{number}.py"
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen" + str(number), encoding="utf8")
        primary[name] = common.sha(path)
    primary_path = dump(ctx["report"] / "EXPERIMENT_PLAN_REPAIR_002.json", dict(source_files=primary))
    own = {}
    for name in ("cp0.py", "__init__.py"):
        relative = f"research/{cp0.NAME}/{name}"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("registered supplement " + name, encoding="utf8")
        own[relative] = common.sha(path)
    started = time.time() - 100
    start = dump(ctx["cache"] / "CORE_EXECUTION_START.json", dict(started_unix=started))
    plan = dict(status="REGISTERED_BEFORE_NEW_V", primary_plan_binding=common.binding(primary_path, ctx),
                primary_source_files=primary, source_files=own, registered_unix=started + 1,
                new_V_scores_at_registration=0, scientific_contract_changed=False,
                selection_changes=False, new_body_training=False,
                cp0_scope=dict(split="A_stop", epoch=0, final_body_paths=10, B_s1_alias="A",
                               CP0_rule=cp0.CP0_RULE, device="cpu", atol=common.ATOL, rtol=common.RTOL,
                               training_updates=0, V_access=False, T_access=False),
                original_execution_start_binding=common.binding(start, ctx),
                original_deadline_unix=started + 21600)
    plan_path = dump(ctx["report"] / cp0.PLAN_NAME, plan)
    finals = final_rows()
    dump(ctx["cache"] / "FINAL_BODY_MODELS.json", finals)
    dump(ctx["report"] / "TRANSFER_SELECTION.json", dict(status="SEALED", selected_mode="B",
         selection_split="A_stop", V_used=False, Jena_V_opened=False))
    dump(ctx["report"] / "MODELS_SEALED_BEFORE_V.json", dict(status="SEALED", V_used=False,
         finals=[{k: r[k] for k in ("run_id", "route_spec", "reused")} for r in finals]))
    dump(ctx["cache"] / cp0.TERMINAL_NAME, dict(status="COMPLETE", exit_code=0))
    arms = [{k: r[k] for k in ("run_id", "mode")} for r in finals]
    arms.extend(dict(run_id="parent" + str(i), mode="ALPHA") for i in range(46))
    dump(ctx["report"] / "SCORES.json", dict(status="COMPLETE", V_independent_test=False, arms=arms))
    facets = []
    for arm in arms:
        for schedule in (("native", "two_call") if arm["mode"] == "A" else ("native",)):
            for batch in (1, 8):
                receipt = dump(ctx["report"] / "costs" / f'{arm["run_id"]}_{schedule}_{batch}.json',
                               dict(status="COMPLETE"))
                facets.append(dict(run_id=arm["run_id"], schedule=schedule, batch=batch,
                                   output=common.binding(receipt, ctx)))
    dump(ctx["report"] / "COST_INDEX.json", dict(status="COMPLETE", facets=facets))

    def check_plan(active, path):
        current = common.read(path)
        for name, digest in current["source_files"].items():
            if common.sha(active["repo"] / name) != digest:
                raise ValueError("Frozen primary source changed")
        active["plan"] = current
        active["deadline"] = 1e12
        return current

    def start_execution(active, proof):
        active["deadline"] = started + 21600
        if time.time() >= active["deadline"]:
            raise TimeoutError("Original core wall budget exhausted")
    monkeypatch.setattr(run, "check_plan", check_plan)
    monkeypatch.setattr(run, "require_preceding_publication", lambda c: dict(commit="a" * 40))
    monkeypatch.setattr(run, "start_execution", start_execution)
    return ctx, plan_path, plan


def test_fixed_plan_requires_terminal_and_all_original_paths_before_numeric_access(registered):
    ctx, path, plan = registered
    actual, finals = cp0.validate_plan(ctx, path, "cpu")
    assert actual == plan and len(finals) == 10
    assert ctx["deadline"] == plan["original_deadline_unix"]
    assert not list(ctx["cache"].rglob("*.npz"))


@pytest.mark.parametrize("status,code", [("FAILED", 1), ("RUNNING", None), ("COMPLETE", 1)])
def test_actual_terminal_is_mandatory(registered, status, code):
    ctx, path, _ = registered
    dump(ctx["cache"] / cp0.TERMINAL_NAME, dict(status=status, exit_code=code))
    with pytest.raises(ValueError, match="main attempt two"):
        cp0.validate_plan(ctx, path, "cpu")


@pytest.mark.parametrize("which", ["supplement", "primary", "start", "tolerance", "device", "new_scores"])
def test_bound_sources_original_budget_and_fixed_protocol_fail_closed(registered, which):
    ctx, path, plan = registered
    if which == "supplement":
        (ctx["repo"] / next(iter(plan["source_files"]))).write_text("tampered")
    elif which == "primary":
        (ctx["repo"] / next(iter(plan["primary_source_files"]))).write_text("tampered")
    elif which == "start":
        dump(ctx["cache"] / "CORE_EXECUTION_START.json", dict(started_unix=time.time()))
    else:
        if which == "tolerance":
            plan["cp0_scope"]["atol"] = 1.
        elif which == "device":
            plan["cp0_scope"]["device"] = "cuda"
        else:
            plan["new_V_scores_at_registration"] = 1
        dump(path, plan)
    with pytest.raises(ValueError):
        cp0.validate_plan(ctx, path, "cpu")


def test_changed_cost_binding_or_incomplete_exact_coverage_is_rejected(registered):
    ctx, path, _ = registered
    index = common.read(ctx["report"] / "COST_INDEX.json")
    index["facets"] = index["facets"][:-1]
    dump(ctx["report"] / "COST_INDEX.json", index)
    with pytest.raises(ValueError, match="cost coverage"):
        cp0.validate_plan(ctx, path, "cpu")


def test_transfer_coverage_rejects_extra_s1_model_and_wrong_transferred_family(registered):
    ctx, _, _ = registered
    assert cp0.body_finals(ctx) == sorted(final_rows(), key=lambda r: (r["dataset"], r["mode"], r["seed"]))
    rows = final_rows() + [copy.deepcopy(final_rows()[0])]
    rows[-1]["mode"] = "B_s1"
    dump(ctx["cache"] / "FINAL_BODY_MODELS.json", rows)
    with pytest.raises(ValueError, match="Exact ten"):
        cp0.body_finals(ctx)
    assert cp0.expected_keys("A") != cp0.expected_keys("B")


def test_original_wall_budget_is_not_reset(registered, monkeypatch):
    ctx, path, plan = registered
    monkeypatch.setattr(cp0.time, "time", lambda: plan["original_deadline_unix"] + 1)
    with pytest.raises(TimeoutError, match="Original core"):
        cp0.validate_plan(ctx, path, "cpu")


def test_independent_fsum_statistics_keep_equal_channel_weights_and_original_missing_mask(monkeypatch):
    monkeypatch.setattr(common, "statistics", lambda *a: (_ for _ in ()).throw(AssertionError("producer called")))
    p = np.ones((2, 2, 3), np.float32)
    p[:, :, 1] = 10
    p[:, :, 2] = 100
    y = np.zeros_like(p)
    mask = np.zeros_like(p, bool)
    mask[:, :, 0] = True
    mask[0, 0, 1] = True
    y[~mask] = np.nan
    value = cp0.independent_statistics(p, y, mask)
    assert value["mse"] == 50.5 and value["mae"] == 5.5
    assert value["n_observed"] == 5 and value["channel_counts"] == [4, 1, 0]
    assert value["mse"] != 104 / 5
    with pytest.raises(ValueError, match="bool"):
        cp0.independent_statistics(p, y, mask.astype(np.uint8))
    with pytest.raises(ValueError, match="No observed"):
        cp0.independent_statistics(p, y, np.zeros_like(mask))


def zero_checkpoint():
    return dict(updates=0, sampler=dict(exposures=0), optimizer=dict(state={}),
                extra=dict(completed_epochs=[0.]), model=dict(x=torch.ones(1)))


def save_pt(path, checkpoint):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def test_s_resolves_true_original_cp0_through_migrated_nonzero_checkpoint_zero(tmp_path, monkeypatch):
    from research.raw_context_multiscale_20261018_v1 import audit
    ctx = dict(repo=tmp_path, cache=tmp_path / ".cache/new", report=tmp_path / "reports/new",
               asset=tmp_path, deadline=time.time() + 100)
    genuine = save_pt(tmp_path / ".cache/older/runs/old/search/checkpoint_000000.pt", zero_checkpoint())
    parent = save_pt(genuine.with_name("checkpoint_000003.pt"), dict(updates=9))
    old_root = tmp_path / ".cache/parent/runs/S/cap6"
    trained = zero_checkpoint()
    trained["updates"] = 9
    trained["sampler"]["exposures"] = 90
    imported = save_pt(old_root / "search/checkpoint_000000.pt", trained)
    terminal = save_pt(imported.with_name("checkpoint_000003.pt"), trained)
    scoped = dict(ctx, cache=tmp_path / ".cache/parent", report=tmp_path / "reports/parent")
    dump(scoped["report"] / "runs/S/cap6/search/MIGRATION_PROOF.json",
         dict(imported_checkpoint=common.binding(imported, scoped), parent_checkpoint=common.binding(parent, scoped),
              changed_fields=["contract"]))
    dump(old_root / "SEARCH_RESULT.json", dict(checkpoint_binding=common.binding(terminal, scoped)))
    monkeypatch.setattr(audit, "old_s_root", lambda *args: old_root)
    final = dict(mode="S", dataset="pems32", seed=92601, run_id="S_pems32_s92601", search={})
    path, checkpoint, ancestry = cp0.resolve_cp0(ctx, final)
    assert path == genuine
    assert checkpoint["updates"] == 0 and checkpoint["sampler"]["exposures"] == 0
    assert len(ancestry) == 4 and ancestry[2]["role"] == "EXPLICIT_S_MIGRATION"


def test_new_family_checkpoint_zero_cannot_hide_trained_state(tmp_path):
    ctx = dict(repo=tmp_path, cache=tmp_path / ".cache/new", report=tmp_path / "reports/new",
               asset=tmp_path, deadline=time.time() + 100)
    final = dict(mode="B", dataset="jena", seed=92601, run_id="B_jena_s92601", search={})
    trained = zero_checkpoint()
    trained["sampler"]["exposures"] = 1
    save_pt(ctx["cache"] / "runs/B_jena_s92601/search/checkpoint_000000.pt", trained)
    with pytest.raises(ValueError, match="training updates or exposures"):
        cp0.resolve_cp0(ctx, final)


@pytest.fixture
def synthetic_body(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / "research/informs_broad_validation_20261015_v1"))
    modules = {k: importlib.import_module(n) for k, n in
               (("fit", "fit_base"), ("training", "training"), ("data", "data_contract"))}
    torch.set_num_threads(2)
    physical = np.random.default_rng(41).normal(size=(96, 4)).astype(np.float32)
    mask = np.ones_like(physical, bool)
    mask[::7, 1] = False
    physical[~mask] = np.nan
    task = dict(dataset="jena", K=2, L=64, H=4, physical=physical, observed=mask,
                columns=["a", "b", "c", "d"], data_sha256=modules["data"].array_sha256(physical),
                split_ranges=dict(A_fit=(0, 80), A_stop=(80, 96)), array_sha256={})

    class Frozen(nn.Module):
        initial_hash = cp0.F_SHA

        def forward(self, x):
            return (x[:, -1:] * .7 + x.mean(dim=1, keepdim=True) * .1).expand(-1, 4, -1)

        def assert_frozen_unchanged(self):
            pass

    foundation = Frozen()
    ctx = dict(repo=repo, cache=tmp_path / "cache", report=tmp_path / "report", asset=tmp_path,
               modules=modules, task=task, foundation=foundation, device="cpu", deadline=time.time() + 120,
               plan=dict(source_files={"frozen.py": "hash"}))
    return ctx, task, foundation


def make_body_cp0(ctx, task, foundation, mode):
    spec = training._spec(ctx, mode, 92601)
    with models.model_factory(training._factory_context(ctx)) as create:
        model, norm = create(spec, task, 80, foundation)
        base, _ = training._contract(ctx, spec)
        active = training._stage_contract(ctx, base, norm, 80)
        origins = ctx["modules"]["data"].origins(task["split_ranges"]["A_stop"], task["L"], task["H"], task["H"])
        windows = ctx["modules"]["training"].FastWindows(task, norm["mean"], norm["std"], "cpu")
        with torch.no_grad():
            x, y, mask = windows.get(origins)
            value = cp0.independent_statistics(model(x, foundation).numpy(), y.numpy(), mask.numpy())["mse"]
        state = ctx["modules"]["training"].cpu_state(model)
        checkpoint = dict(model=state, contract=active, updates=0, optimizer=dict(state={}),
                          sampler=dict(exposures=0), torch_rng=torch.get_rng_state(), cuda_rng=[],
                          extra=dict(completed_epochs=[0.]))
        path = save_pt(ctx["cache"] / "runs" / spec["run_id"] / "search/checkpoint_000000.pt", checkpoint)
        search = dict(spec=spec, contract=base, initial_base_state_sha256=_state_fingerprint(state),
                      curves=[dict(epoch=0., astop_mse=value)])
        return dict(dataset="jena", mode=mode, seed=92601, run_id=spec["run_id"], search=search), path


@pytest.mark.parametrize("mode", ["A", "B"])
def test_actual_adapter_cp0_is_reloaded_without_updates_and_saved_as_fp32_with_rng_preserved(synthetic_body, mode):
    ctx, task, foundation = synthetic_body
    final, path = make_body_cp0(ctx, task, foundation, mode)
    before = torch.get_rng_state().clone()
    row, proof = cp0.evaluate_one(ctx, task, final, foundation, "cpu", dict(reference="plan", sha256="pin"), {})
    assert torch.equal(before, torch.get_rng_state())
    assert row["epoch"] == row["updates"] == row["exposures"] == 0
    assert row["MSE_parity"] == "PASS" and row["A_stop_mse"] == row["MSE"]
    assert row["ctx_scale"] == (2 if mode == "B" else 1)
    assert row["checkpoint_sha256"] == common.sha(path)
    with np.load(common.resolve_binding(proof["prediction_binding"], ctx), allow_pickle=False) as saved:
        assert saved["pred_normalized"].dtype == saved["target_normalized"].dtype == np.float32
        assert saved["observed"].dtype == np.bool_
        assert saved["origins"].tolist() == [80, 84, 88, 92]
        assert saved["channel_ids"].tolist() == task["columns"]
    assert not list(ctx["cache"].rglob("*V_select*"))
    assert proof["metadata"]["A_fit_end"] == 80 and proof["metadata"]["V_access"] is False


def test_initial_mse_parity_is_fail_closed_without_writing_predictions(synthetic_body):
    ctx, task, foundation = synthetic_body
    final, _ = make_body_cp0(ctx, task, foundation, "B")
    final["search"]["curves"][0]["astop_mse"] += .5
    with pytest.raises(ValueError, match="fixed tolerance parity"):
        cp0.evaluate_one(ctx, task, final, foundation, "cpu", {}, {})
    assert not list(ctx["cache"].rglob("*.npz"))


def test_initial_weights_and_scaler_contract_are_validated_before_evaluation(synthetic_body):
    ctx, task, foundation = synthetic_body
    final, path = make_body_cp0(ctx, task, foundation, "A")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["contract"]["scaler_sha"] = "changed"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="scaler/PCA"):
        cp0.evaluate_one(ctx, task, final, foundation, "cpu", {}, {})
    assert not list(ctx["cache"].rglob("*.npz"))


def test_sealed_csv_never_overwrites_different_cp0_rows(tmp_path):
    path = tmp_path / "CP0.csv"
    row = dict.fromkeys(cp0.FIELDS, 0)
    cp0._sealed_csv(path, [row])
    first = path.read_bytes()
    cp0._sealed_csv(path, [row])
    assert path.read_bytes() == first
    row["MAE"] = 1
    with pytest.raises(ValueError, match="cannot overwrite"):
        cp0._sealed_csv(path, [row])
    assert path.read_bytes() == first


@pytest.mark.parametrize("dataset", ["jena", "pems32"])
def test_numeric_loader_requests_only_a_stop_prefix_and_keeps_exact_order_mask_and_time(tmp_path, monkeypatch, dataset):
    campaign = tmp_path / "campaign"
    asset = tmp_path / "asset"
    asset.mkdir()
    ctx = dict(repo=tmp_path, cache=tmp_path / "cache", report=tmp_path / "report", asset=asset,
               modules=dict(campaign=campaign, old_cache=asset), deadline=time.time() + 30)
    end = 10
    count = 2 if dataset == "jena" else 32
    columns = ["c" + str(i) for i in range(count)]
    expected = dict(C=count, L=4, H=2, data_sha256="original-cohort", columns=columns,
                    split_ranges=dict(A_fit=[0, 6], A_stop=[6, end]), array_sha256={})
    dump(tmp_path / "reports/b2_revin_20261018_v1/DATA_CONTRACT.json", dict(tasks={dataset: expected}))
    requested = []
    primitive = SimpleNamespace()
    if dataset == "jena":
        source = asset / "jena.npz"
        payload = dict(x=np.arange(30, dtype=np.float32).reshape(15, 2),
                       finite=np.ones((15, 2), bool), times=np.arange(15, dtype=np.int64) * 600 * 1000000000,
                       mean=np.array([2., 3.], np.float32), std=np.array([2., 4.], np.float32),
                       columns=np.asarray(columns))
        payload["finite"][7, 1] = False
        np.savez(source, **payload)
        metadata = asset / "ancestry.json"
        metadata.write_text("ancestry", encoding="utf8")
        primitive.SOURCE_REL = "jena.npz"
        primitive.SOURCE_BYTES = source.stat().st_size
        primitive.CONTRACT_REL = "ancestry.json"
        primitive.CONTRACT_SHA = common.sha(metadata)

        def read_member(archive, name, rows):
            requested.append((name, rows))
            with archive.open(name) as stream:
                values = np.load(stream, allow_pickle=False)
            return (values if rows is None else values[:rows]), dict(name=name, rows=rows)
        primitive.read_member = read_member
        dump(campaign / "DATA_CONTRACT.json", dict(assets=dict(jena=dict(data_sha256="original-cohort"))))
        monkeypatch.setattr(cp0.data, "_module", lambda *args: primitive)
    else:
        source = asset / "pems.npz"
        payload = np.arange(15 * 170 * 3, dtype=np.float32).reshape(15, 170, 3)
        payload[7, 1, 0] = np.nan
        np.savez(source, data=payload)
        all_columns = ["c" + str(i) for i in range(170)]
        plan = dict(dependencies={}, parent_task_contract=dict(columns=all_columns,
                    metadata=dict(source_reference="pems.npz", source_bytes=source.stat().st_size)),
                    column_selection=dict(selected_indices=list(range(32)), selected_ids=columns))
        dump(campaign / "priority_pems_subset_20261016_v1/PLAN.json", plan)

        def prefix_payload(stream, file_size, primitive_arg, rows):
            requested.append(("data.npy", rows))
            return np.load(stream, allow_pickle=False)[:rows], dict(rows=rows)
        pems = SimpleNamespace(prefix_payload=prefix_payload)
        subset = SimpleNamespace(require_sealed=lambda p: None)
        def module(path, label):
            return subset if path.name == "subset_minimum_runner.py" else pems if path.name == "pems_fixed_head_runner.py" else primitive
        monkeypatch.setattr(cp0.data, "_module", module)
    task, access = cp0.a_stop_task(ctx, dataset)
    assert task["physical"].shape == (end, count) and task["columns"] == columns
    assert task["observed"].dtype == np.bool_ and not task["observed"][7, 1]
    assert np.isnan(task["physical"][7, 1])
    assert access["M_cal_V_T_numeric_values_parsed"] == 0
    assert access["numeric_prefix_rows"] == end
    assert all(rows == end for name, rows in requested if name in ("data.npy", "x.npy", "finite.npy", "times.npy"))
    if dataset == "jena":
        assert task["physical"][0].tolist() == [2., 7.]
    else:
        np.testing.assert_array_equal(task["physical"][0], payload[0, :32, 0])


def test_cp0_history_fill_never_uses_values_before_window_or_future_targets(synthetic_body):
    ctx, task, _ = synthetic_body
    task = dict(task, L=4, H=2, physical=np.arange(16, dtype=np.float32)[:, None],
                observed=np.ones((16, 1), bool), columns=["x"])
    task["observed"][1:3] = False
    task["physical"][1:3] = np.nan
    windows = ctx["modules"]["training"].FastWindows(task, np.array([10.], np.float32),
                                                  np.ones(1, np.float32), "cpu")
    x, _, _ = windows.get(np.array([4, 5]))
    np.testing.assert_array_equal(x.numpy()[:, :, 0], [[-10., -10., -10., -7.], [0., 0., -7., -6.]])
    task["physical"][5:] = 1e6
    mutated = ctx["modules"]["training"].FastWindows(task, np.array([10.], np.float32),
                                                  np.ones(1, np.float32), "cpu")
    np.testing.assert_array_equal(mutated.get(np.array([5]))[0].numpy(), x.numpy()[1:])
