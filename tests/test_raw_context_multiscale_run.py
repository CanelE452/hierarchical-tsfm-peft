"""Synthetic orchestration/restart checks; no actual source, V or CUDA work."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.raw_context_multiscale_20261018_v1 import audit, calibration, common, gates, pipeline, run, training


@contextmanager
def no_gpu_session(*args, **kwargs):
    yield


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def roots(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = dict(repo=root, cache=tmp_path / "cache", report=tmp_path / "report", asset=tmp_path,
                  device="cpu", deadline=1e12, modules={})
    result["cache"].mkdir()
    result["report"].mkdir()
    return result


def _search(dataset, mode, seed, cap, reused=False):
    return dict(status="COMPLETE", spec=dict(run_id=f"{mode}_{dataset}_{seed}", dataset=dataset,
                    seed=seed, scientific_identity=dict(raw_context_mode=mode)),
                cap=cap, selected_epoch=1., summary=dict(selected_astop_mse=.1 if mode == "A" else .2),
                curves=[dict(epoch=0., astop_mse=1.), dict(epoch=float(cap), astop_mse=.1 if mode == "A" else .2)],
                checkpoint_binding=dict(reference=f"cache://{dataset}_{mode}_{seed}_{cap}.pt", sha256="synthetic"),
                selection_split="A_stop", V_used=False, reused_parent=reused)


@pytest.fixture
def execute_fixture(roots, monkeypatch):
    tasks = {dataset: dict(L=2, H=1, K=1, columns=["c0"], dataset=dataset,
                          split_ranges=dict(A_fit=(0, 8), A_stop=(8, 12), M_cal=(12, 20)))
             for dataset in ("jena", "pems32")}
    parents = [dict(role="S", dataset=dataset, seed=seed,
                    route_spec=dict(family="RAW", run_id=f"parent_{dataset}_{seed}"))
               for dataset in tasks for seed in common.SEEDS]
    profiles = [dict(dataset=d, mode=m, seconds_per_update=.001) for d in tasks for m in ("S", "A", "B")]
    plan = dict(limits=dict(wall_seconds=21600, GPU_reservation_seconds=14400, report_fraction=.2))
    monkeypatch.setattr(run, "check_plan", lambda ctx, path: ctx.update(plan=plan, deadline=1e12))
    monkeypatch.setattr(run, "require_preceding_publication", lambda ctx: {})
    monkeypatch.setattr(run, "start_execution", lambda ctx, proof: None)
    monkeypatch.setattr(audit, "prepare", lambda ctx: (dict(parents=parents), tasks))
    monkeypatch.setattr(run, "gpu_session", no_gpu_session)
    monkeypatch.setattr(run, "_release", lambda ctx: ctx.pop("foundation", None))
    import research.adapter_improvements_20261018_v1.audit as old_audit
    monkeypatch.setattr(old_audit, "load_foundation", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "calibrate", lambda *args: [])
    monkeypatch.setattr(pipeline, "complete", lambda *args: None)
    monkeypatch.setattr(training, "reuse_S_search", lambda ctx, seed, path: _search(
        ctx["task"]["dataset"], "S", seed, 3 if ctx["task"]["dataset"] == "jena" else 6, True))
    monkeypatch.setattr(training, "search", lambda ctx, mode, seed, endpoints: _search(
        ctx["task"]["dataset"], mode, seed, int(endpoints[-1])))

    def fake_refit(ctx, result, selection):
        artifact = ctx["cache"] / (result["spec"]["run_id"] + ".pt")
        if not artifact.exists():
            artifact.write_bytes(b"synthetic-body")
        return dict(status="COMPLETE", bundle_binding=common.binding(artifact, ctx), refit_summary={})

    monkeypatch.setattr(training, "refit", fake_refit)
    monkeypatch.setattr(run, "estimate", lambda ctx, tasks, profiles, requests:
                        200. if any(mode == "S" for _, mode, _, _ in requests) else 100.)
    ticks = [0]

    def budget(ctx, seconds, reserve=True):
        ticks[0] += 1
        return dict(admitted=seconds in (0., 1000.), estimated_GPU_seconds=seconds,
                    wall_remaining_after_reserve=10000. - ticks[0], GPU_remaining=9000. - ticks[0],
                    report_reserve_seconds=4320. if reserve else 0.)

    monkeypatch.setattr(run, "_budget", budget)
    return roots, profiles


def test_execute_reuses_the_actual_profile_gate_filename(execute_fixture, monkeypatch):
    ctx, profiles = execute_fixture
    gate_path = ctx["report"] / "VALIDATION_PROFILE_GPU_GPU.json"
    common.sealed_json(gate_path, dict(status="PASS", profiles=profiles))
    monkeypatch.setattr(gates, "run_gates", lambda *args, **kwargs: pytest.fail("Existing profiled gate must be reused"))
    assert run.execute(ctx, "unused-by-fixture")["status"] == "COMPLETE"


def _use_second_repair(monkeypatch):
    original=run.check_plan
    def check(ctx,path):
        original(ctx,path)
        ctx['plan']['repair']=dict(GPU_profile_output_id='GPU_REPAIR_002',
                                   correction='LEGACY_PARENT_LOADING_SCHEMA_ONLY')
    monkeypatch.setattr(run,'check_plan',check)


def test_second_repair_runs_fresh_profile_and_separate_initial_admission(execute_fixture,monkeypatch):
    ctx,profiles=execute_fixture
    _use_second_repair(monkeypatch)
    old_profile=ctx['report']/'VALIDATION_PROFILE_GPU_GPU.json'
    common.sealed_json(old_profile,dict(status='PASS',profiles=[]))
    old_admission=ctx['report']/'INITIAL_ADMISSION.json'
    common.sealed_json(old_admission,dict(admitted=False,historical=True))
    preserved=[path.read_bytes() for path in (old_profile,old_admission)]
    calls=[]
    def gates_run(active,**kwargs):
        calls.append(kwargs)
        common.sealed_json(active['report']/'VALIDATION_PROFILE_GPU_GPU_REPAIR_002.json',
                           dict(status='PASS',profiles=profiles))
    monkeypatch.setattr(gates,'run_gates',gates_run)
    assert run.execute(ctx,'unused-by-fixture')['status']=='COMPLETE'
    assert calls==[dict(device='cuda',profile=True,output_id='GPU_REPAIR_002')]
    new_admission=ctx['report']/'INITIAL_ADMISSION_REPAIR_002.json'
    assert common.read(new_admission)['admitted'] is True
    assert [path.read_bytes() for path in (old_profile,old_admission)]==preserved
    receipt=new_admission.read_bytes()
    monkeypatch.setattr(gates,'run_gates',lambda *a,**k:pytest.fail('Corrected profile must be reused'))
    assert run.execute(ctx,'unused-by-fixture')['status']=='COMPLETE'
    assert new_admission.read_bytes()==receipt


def test_second_repair_declines_full_group_if_corrected_profile_exceeds_budget(execute_fixture,monkeypatch):
    ctx,profiles=execute_fixture
    _use_second_repair(monkeypatch)
    common.sealed_json(ctx['report']/'VALIDATION_PROFILE_GPU_GPU_REPAIR_002.json',dict(status='PASS',profiles=profiles))
    historical=ctx['report']/'INITIAL_ADMISSION.json'
    common.sealed_json(historical,dict(admitted=True,historical=True))
    original=historical.read_bytes()
    monkeypatch.setattr(gates,'run_gates',lambda *a,**k:pytest.fail('Corrected synthetic profile must be reused'))
    monkeypatch.setattr(run,'_budget',lambda *a,**k:dict(admitted=False,estimated_GPU_seconds=1000.))
    monkeypatch.setattr(pipeline,'calibrate',lambda *a,**k:pytest.fail('No M_cal fit after failed group admission'))
    with pytest.raises(TimeoutError,match='complete matched initial group'):
        run.execute(ctx,'unused-by-fixture')
    assert historical.read_bytes()==original
    assert common.read(ctx['report']/'INITIAL_ADMISSION_REPAIR_002.json')['admitted'] is False


def test_cuda_smoke_selects_second_repair_profile_output_id(execute_fixture,monkeypatch):
    ctx,_=execute_fixture
    _use_second_repair(monkeypatch)
    monkeypatch.setattr(common,'context',lambda *a,**k:ctx)
    calls=[]
    monkeypatch.setattr(gates,'run_gates',lambda *a,**k:calls.append(k) or dict(status='PASS'))
    run.main(['smoke','--asset-root',str(ctx['asset']),'--device','cuda'])
    assert calls==[dict(device='cuda',profile=True,output_id='GPU_REPAIR_002')]


def test_restart_preserves_historical_admission_decisions_and_queue(execute_fixture, monkeypatch):
    ctx, profiles = execute_fixture
    common.sealed_json(ctx["report"] / "VALIDATION_PROFILE_GPU_GPU.json", dict(status="PASS", profiles=profiles))
    monkeypatch.setattr(gates, "run_gates", lambda *args, **kwargs: pytest.fail("No GPU work is allowed"))
    first = run.execute(ctx, "unused-by-fixture")
    initial = (ctx["report"] / "INITIAL_ADMISSION.json").read_bytes()
    selection = (ctx["report"] / "SELECTIONS_SEALED.json").read_bytes()
    for row in first["resume_queue"]:
        cap = 3 if row["dataset"] == "jena" else 6
        decision = common.read(ctx["report"] / "decisions" / f"{row['dataset']}_cap{cap}.json")
        assert row["budget"] == decision["budget"]
    second = run.execute(ctx, "unused-by-fixture")
    assert second == first
    assert (ctx["report"] / "INITIAL_ADMISSION.json").read_bytes() == initial
    assert (ctx["report"] / "SELECTIONS_SEALED.json").read_bytes() == selection


def test_admission_uses_worst_actual_pems_profile_and_pending_jena_refits(execute_fixture, monkeypatch):
    ctx, profiles = execute_fixture
    for row in profiles:
        if row["dataset"] == "pems32" and row["mode"] == "A":
            row["seconds_per_update"] = .009
    common.sealed_json(ctx["report"] / "VALIDATION_PROFILE_GPU_GPU.json", dict(status="PASS", profiles=profiles))
    requests_seen = []
    def estimate(ctx, tasks, profiles, requests):
        requests_seen.append(list(requests))
        return 200. if any(mode == "S" for _, mode, _, _ in requests) else 100.
    monkeypatch.setattr(run, "estimate", estimate)
    run.execute(ctx, "unused-by-fixture")
    assert {mode for dataset, mode, _, _ in requests_seen[0] if dataset == "pems32"} == {"A"}
    pending = [(mode, epoch, end) for dataset, mode, epoch, end in requests_seen[-1] if dataset == "jena"]
    assert sorted(pending) == [("A", 1., 12579), ("A", 1., 12579), ("B", 1., 12579), ("B", 1., 12579)]


@pytest.fixture
def calibration_fixture(roots, monkeypatch):
    rng = np.random.default_rng(879)
    origins = np.arange(20, 69, dtype=np.int64)
    past = rng.normal(size=(len(origins), 8, 2)).astype(np.float32)
    values = rng.normal(size=(80, 2)).astype(np.float32)
    task = dict(dataset="jena", data_sha256="synthetic-data", L=8, H=4, columns=["c0", "c1"],
                physical=values, observed=np.ones(values.shape, bool),
                split_ranges=dict(A_fit=(0, 16), A_stop=(16, 20), M_cal=(20, 72)))
    parents = [dict(dataset="jena", role=role, seed=seed, parent_fit_end=20,
                    route_spec=dict(family="RAW", run_id=f"parent_{role}_{seed}", role=role, seed=seed))
               for role, seed in (("S", 92601), ("S", 92602), ("G", 92601), ("Raw", None))]
    roots["modules"] = dict(legacy=SimpleNamespace(MODEL_REL="not-opened"),
        fit=SimpleNamespace(scaler_fingerprint=lambda mean, std: common.fingerprint(
            dict(mean=mean.tolist(), std=std.tolist()))))

    class NoFoundationPool:
        def __init__(self, *args, **kwargs):
            pass
        def close(self):
            pass

    monkeypatch.setattr(pipeline.deployment.legacy, "FoundationPool", NoFoundationPool)
    monkeypatch.setattr(run, "gpu_session", no_gpu_session)
    monkeypatch.setattr(pipeline.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(pipeline, "_past", lambda *args: past.copy())

    def forecast(ctx, task, spec, run_id, split, **kwargs):
        assert split == "M_cal", "Synthetic calibration must not request V"
        g = np.repeat(past[:, -1:], task["H"], axis=1)
        prediction = g + (.3 if spec["role"] == "S" else 0.)
        target = g + .2
        mask = np.ones(target.shape, bool)
        path = ctx["cache"] / "synthetic_parents" / (run_id + ".npz")
        evidence = common.save_prediction(path, prediction=prediction, target=target, observed=mask,
            origins=origins, task=task, metadata=dict(run_id=run_id, split="M_cal"))
        evidence["prediction_binding"] = common.binding(path, ctx)
        return dict(prediction=prediction, target=target, mask=mask, origins=origins,
                    mean=np.zeros(2, np.float32), std=np.ones(2, np.float32), evidence=evidence)

    monkeypatch.setattr(pipeline, "_forecast", forecast)
    return roots, dict(jena=task), dict(parents=parents)


def test_calibration_public_receipt_precedes_completed_index(calibration_fixture, monkeypatch):
    ctx, tasks, inventory = calibration_fixture
    original = common.sealed_json

    def verify_order(path, value):
        if Path(path).name == "CALIBRATION_MODELS.json":
            assert (ctx["report"] / "COMPLEMENTARITY_FITS.json").exists()
        return original(path, value)

    monkeypatch.setattr(common, "sealed_json", verify_order)
    result = pipeline.calibrate(ctx, tasks, inventory)
    assert len(result) == 20


def _interrupt_index(ctx, monkeypatch):
    original = common.sealed_json
    def interrupted(path, value):
        if Path(path).name == "CALIBRATION_MODELS.json":
            raise KeyboardInterrupt("Synthetic interruption before completion index")
        return original(path, value)
    monkeypatch.setattr(common, "sealed_json", interrupted)
    return original


def test_partial_calibration_reuses_heads_and_preserves_elapsed_receipt(calibration_fixture, monkeypatch):
    ctx, tasks, inventory = calibration_fixture
    search, refit = calibration.head_search, calibration.head_refit
    counter = [0]
    def timed_search(*args, **kwargs):
        counter[0] += 1
        return dict(search(*args, **kwargs), elapsed_seconds=counter[0] / 10.)
    def timed_refit(*args, **kwargs):
        state, receipt = refit(*args, **kwargs)
        return state, dict(receipt, wall_seconds=counter[0] / 7.)
    monkeypatch.setattr(calibration, "head_search", timed_search)
    monkeypatch.setattr(calibration, "head_refit", timed_refit)
    original = _interrupt_index(ctx, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        pipeline.calibrate(ctx, tasks, inventory)
    fits = ctx["report"] / "COMPLEMENTARITY_FITS.json"
    assert fits.exists(), "The fit receipt must survive an interruption before the index"
    preserved = fits.read_bytes()
    monkeypatch.setattr(common, "sealed_json", original)
    monkeypatch.setattr(calibration, "head_search", lambda *args, **kwargs: pytest.fail("Valid head search must be reused"))
    monkeypatch.setattr(calibration, "head_refit", lambda *args, **kwargs: pytest.fail("Valid fresh head must be reused"))
    assert len(pipeline.calibrate(ctx, tasks, inventory)) == 20
    assert fits.read_bytes() == preserved


def test_partial_calibration_rejects_head_state_that_disagrees_with_refit_receipt(calibration_fixture, monkeypatch):
    ctx, tasks, inventory = calibration_fixture
    original = _interrupt_index(ctx, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        pipeline.calibrate(ctx, tasks, inventory)
    target = next((ctx["cache"] / "calibration_heads").glob("*.pt"))
    saved = torch.load(target, map_location="cpu", weights_only=False)
    saved["state"]["output.bias"].add_(1.)
    torch.save(saved, target)
    monkeypatch.setattr(common, "sealed_json", original)
    with pytest.raises(ValueError, match="state|SHA|identity|changed"):
        pipeline.calibrate(ctx, tasks, inventory)


def test_partial_calibration_rejects_changed_data_identity(calibration_fixture, monkeypatch):
    ctx, tasks, inventory = calibration_fixture
    original = _interrupt_index(ctx, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        pipeline.calibrate(ctx, tasks, inventory)
    tasks["jena"]["data_sha256"] = "changed-synthetic-data"
    monkeypatch.setattr(common, "sealed_json", original)
    with pytest.raises(ValueError, match="contract|identity|changed|source|binding"):
        pipeline.calibrate(ctx, tasks, inventory)


@pytest.mark.parametrize("has_binding,tampered", [(True, True), (False, True), (False, False)])
def test_forecast_requires_proof_before_reusing_private_cache(roots, monkeypatch, has_binding, tampered):
    rng = np.random.default_rng(901)
    values = rng.normal(size=(80, 2)).astype(np.float32)
    task = dict(dataset="jena", data_sha256="synthetic-data", L=8, H=4, columns=["c0", "c1"],
                physical=values, observed=np.ones(values.shape, bool),
                split_ranges=dict(A_stop=(16, 20), M_cal=(20, 72)))
    roots["modules"] = dict(fit=SimpleNamespace(scaler_fingerprint=lambda mean, std: "synthetic-scaler"))
    calls = [0]
    class SyntheticRoute:
        mean = np.zeros(2, np.float32)
        std = np.ones(2, np.float32)
        def predict_physical(self, physical, observed):
            calls[0] += 1
            return np.repeat(physical[:, -1:], 4, axis=1)
        def close(self):
            pass
    monkeypatch.setattr(pipeline.deployment, "make_route", lambda *args, **kwargs: SyntheticRoute())
    spec = dict(family="RAW", run_id="synthetic_raw")
    pipeline._forecast(roots, task, spec, "synthetic_raw", "M_cal", device="cpu")
    path = roots["cache"] / "predictions" / "synthetic_raw" / "M_cal.npz"
    if not has_binding:
        path.with_suffix(".binding.json").unlink()
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if tampered:
        arrays["pred_normalized"][0, 0, 0] += 1.
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        with pytest.raises(ValueError, match="(?i)changed|SHA|identity|binding|fingerprint|unsealed|unbound|missing|parity"):
            pipeline._forecast(roots, task, spec, "synthetic_raw", "M_cal", device="cpu")
    else:
        before, count = path.read_bytes(), calls[0]
        pipeline._forecast(roots, task, spec, "synthetic_raw", "M_cal", device="cpu")
        assert calls[0] > count, "Unbound cached output requires actual regeneration"
        assert path.read_bytes() == before
        assert path.with_suffix(".binding.json").exists()


def test_gpu_budget_charges_interrupted_sessions_without_using_gpu(roots, monkeypatch):
    rows = [dict(stage="GPU_SESSION", status="STARTED", session_id="done", time_unix=10., budget_deadline_unix=60.),
            dict(stage="GPU_SESSION", status="COMPLETE", session_id="done", time_unix=22., wall_seconds=12.),
            dict(stage="GPU_SESSION", status="STARTED", session_id="interrupted", time_unix=40., budget_deadline_unix=70.)]
    (roots["cache"] / "EVENTS.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf8")
    monkeypatch.setattr(run.time, "time", lambda: 100.)
    assert run.gpu_seconds(roots) == 42.


def test_gpu_session_propagates_remaining_budget_and_restores_deadline(roots, monkeypatch):
    rows = [dict(stage="GPU_SESSION", status="STARTED", session_id="done", time_unix=10., budget_deadline_unix=60.),
            dict(stage="GPU_SESSION", status="COMPLETE", session_id="done", time_unix=20., wall_seconds=10.)]
    journal = roots["cache"] / "EVENTS.jsonl"
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8")
    roots.update(deadline=1000., plan=dict(limits=dict(GPU_reservation_seconds=100.)))
    monkeypatch.setattr(run.time, "time", lambda: 100.)
    with run.gpu_session(roots, "synthetic_session"):
        assert roots["deadline"] == 190.
    assert roots["deadline"] == 1000.
    saved = [json.loads(line) for line in journal.read_text(encoding="utf8").splitlines()]
    assert saved[-2]["session_id"] == saved[-1]["session_id"]
    assert saved[-1]["status"] == "COMPLETE"


def _execution_start_fixture(ctx):
    ctx['plan'] = dict(started_unix=1.,limits=dict(wall_seconds=21600,GPU_reservation_seconds=14400,report_fraction=.2))
    proof = dict(status='PUSHED_AND_REMOTE_VERIFIED',independent_positive_verified=True,commit='a'*40)
    common.sealed_json(ctx['report']/'EXPERIMENT_PLAN.json',ctx['plan'])
    common.sealed_json(ctx['cache']/'PRECEDING_DATASET_PUBLICATION.json',proof)
    return proof


def test_core_budget_starts_after_publication_and_never_resets_on_resume(roots,monkeypatch):
    proof = _execution_start_fixture(roots)
    monkeypatch.setattr(run.time,'time',lambda:100000.)
    first = run.start_execution(roots,proof)
    assert first['started_unix']==100000.
    assert roots['deadline']==121600.
    preserved = (roots['report']/'EXECUTION_START.json').read_bytes()
    monkeypatch.setattr(run.time,'time',lambda:100500.)
    assert run.start_execution(roots,proof)==first
    assert roots['deadline']==121600.
    assert (roots['report']/'EXECUTION_START.json').read_bytes()==preserved
    monkeypatch.setattr(run.time,'time',lambda:121601.)
    with pytest.raises(TimeoutError,match='original core execution'):
        run.start_execution(roots,proof)


def test_core_start_rejects_changed_preceding_publication(roots,monkeypatch):
    proof = _execution_start_fixture(roots)
    monkeypatch.setattr(run.time,'time',lambda:100000.)
    run.start_execution(roots,proof)
    path = roots['cache']/'PRECEDING_DATASET_PUBLICATION.json'
    common.atomic_json(path,dict(proof,commit='b'*40))
    with pytest.raises(ValueError,match='publication identity'):
        run.start_execution(roots,dict(proof,commit='b'*40))
