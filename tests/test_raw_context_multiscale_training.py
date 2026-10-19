import copy
import importlib
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from research.b2_revin_20261018_v1 import revin
from research.raw_context_multiscale_20261018_v1 import common, models, training


@pytest.fixture
def ctx(monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / "research/informs_broad_validation_20261015_v1"))
    modules = {key: importlib.import_module(name) for key, name in
               (("fit", "fit_base"), ("training", "training"), ("data", "data_contract"))}
    torch.set_num_threads(2)
    values = np.random.default_rng(42).normal(size=(96, 4)).astype(np.float32)
    observed = np.ones_like(values, bool)
    observed[::7, 1] = False
    task = dict(dataset="jena", K=2, L=64, H=4, physical=values, observed=observed,
                columns=["a", "b", "c", "d"], data_sha256=modules["data"].array_sha256(values),
                split_ranges=dict(A_fit=(0, 80), A_stop=(80, 96), M_cal=(96, 112),
                                  V_select=(112, 128), T_final=(128, 144)))

    class Frozen(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(.7), requires_grad=False)
            self.initial_hash = "synthetic-frozen-F"
            self.streams = self.row_calls = self.physical_streams = self.physical_calls = 0

        def forward(self, x):
            self.streams += len(x) * x.shape[-1]
            self.physical_streams += len(x) * x.shape[-1]
            self.row_calls += 1
            self.physical_calls += 1
            return (x[:, -1:] * self.weight + x.mean(dim=1, keepdim=True) * .1).expand(-1, 4, -1)

        def assert_frozen_unchanged(self):
            assert self.weight.grad is None and self.weight.item() == torch.tensor(.7).item()

    return dict(repo=repo, cache=tmp_path / "cache", report=tmp_path / "report", asset=tmp_path,
                modules=modules, task=task, foundation=Frozen(), device="cpu", deadline=time.time() + 120)


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_state(a, b)
    else:
        assert left == right


def alternative_context(ctx, name):
    return dict(ctx, cache=Path(ctx["asset"]) / name / "cache", report=Path(ctx["asset"]) / name / "report")


def selection(ctx, searches):
    path = Path(ctx["report"]) / "SELECTION.json"
    common.sealed_json(path, dict(status="SEALED", selection_split="A_stop", V_used=False,
                                 searches=[value["checkpoint_binding"] for value in searches]))
    return common.binding(path, ctx)


@pytest.mark.parametrize("mode", ["A", "B"])
def test_search_is_a_stop_only_with_cp0_and_complete_cache_reuse(ctx, mode):
    result = training.search(ctx, mode, 92601, common.INITIAL_EPOCHS)
    assert result["status"] == "COMPLETE"
    assert [row["epoch"] for row in result["curves"]] == [0., .25, 1., 3.]
    assert result["V_used"] is False and result["selection_split"] == "A_stop"
    assert result["summary"]["exposures"] == 39
    assert result["summary"]["updates"] == 4
    assert result["summary"]["stop_origins"] == 4
    assert "epochs" not in result["spec"]
    assert result["spec"]["scientific_identity"]["potential_endpoints"] == list(common.EXTENDED_EPOCHS)
    assert not list(Path(ctx["cache"]).rglob("selected_bundle.pt"))
    assert not list(Path(ctx["cache"]).rglob("*.npz"))
    streams = ctx["foundation"].streams
    assert training.search(ctx, mode, 92601, common.INITIAL_EPOCHS) == result
    assert ctx["foundation"].streams == streams


@pytest.mark.parametrize("mode", ["A", "B"])
def test_three_to_six_keeps_exact_optimizer_sampler_rng_and_model(ctx, mode):
    first = training.search(ctx, mode, 92601, common.INITIAL_EPOCHS)
    continued = training.search(ctx, mode, 92601, common.EXTENDED_EPOCHS)
    reference = training.search(alternative_context(ctx, "direct"), mode, 92601, common.EXTENDED_EPOCHS)
    resumed = torch.load(continued["last_checkpoint"], map_location="cpu", weights_only=False)
    uninterrupted = torch.load(reference["last_checkpoint"], map_location="cpu", weights_only=False)
    assert first["spec"] == continued["spec"]
    assert first["contract"] == continued["contract"]
    assert continued["curves"][:4] == first["curves"]
    for key in ("model", "optimizer", "sampler", "torch_rng", "cuda_rng", "updates"):
        equal_state(resumed[key], uninterrupted[key])
    assert continued["summary"]["exposures"] == 78
    assert continued["summary"]["updates"] == 7
    assert common.resolve_binding(first["checkpoint_binding"], ctx).exists()
    segments = continued["training_segments"]
    assert segments[:3] == first["training_segments"]
    assert [row["endpoint_epoch"] for row in segments] == [.25, 1., 3., 4., 5., 6.]
    assert [row["end_exposures"] for row in segments] == [4, 13, 39, 52, 65, 78]
    assert len({row["segment_id"] for row in segments}) == 6
    assert len({row["record_id"] for row in segments}) == 6
    for row in segments:
        assert np.isfinite(row["mean_observed_training_batch_loss"])
        assert np.isfinite(row["gradient_norm_min"]) and np.isfinite(row["gradient_norm_max"])
        assert row["optimizer_updates_this_segment"] > 0
        assert row["segment_seconds"] >= 0
        assert "DURING" in row["loss_interpretation"]
    private = Path(continued["last_checkpoint"]).parent / "TRAIN_SEGMENTS.jsonl"
    public = Path(ctx["report"]) / "runs" / continued["spec"]["run_id"] / "search/TRAIN_SEGMENTS.jsonl"
    assert private.read_bytes() == public.read_bytes()
    assert len(training._segment_rows(private)) == 6


def test_changed_initialization_contract_cannot_resume(ctx):
    first = training.search(ctx, "A", 92601, common.INITIAL_EPOCHS)
    checkpoint = torch.load(first["last_checkpoint"], map_location="cpu", weights_only=False)
    checkpoint["contract"]["initialization_sha"] = "altered"
    torch.save(checkpoint, first["last_checkpoint"])
    with pytest.raises(ValueError, match="including initialization"):
        training.search(ctx, "A", 92601, common.EXTENDED_EPOCHS)


def test_fresh_a60_refit_initializes_new_scope_and_keeps_v_unmaterialized(ctx):
    search = training.search(ctx, "B", 92601, common.INITIAL_EPOCHS)
    result = training.refit(ctx, search, selection(ctx, [search]))
    assert result["status"] == "COMPLETE" and result["V_used"] is False
    assert result["fit_receipt"]["fresh_A60"] is True
    assert result["fit_receipt"]["search_fit_end"] == 80
    assert result["fit_receipt"]["refit_fit_end"] == 96
    bundle = torch.load(common.resolve_binding(result["bundle_binding"], ctx), map_location="cpu", weights_only=False)
    assert bundle["normalization"]["fit_end"] == 96
    assert bundle["selected_search_epoch"] == search["selected_epoch"]
    assert result["refit_summary"]["stop_origins"] == 0
    assert result["refit_summary"]["train_origins"] == 29
    with models.model_factory(training._factory_context(ctx)) as create:
        fresh, fresh_norm = create(search["spec"], ctx["task"], 96, ctx["foundation"])
        old, _ = create(search["spec"], ctx["task"], 80, ctx["foundation"])
    initial = Path(ctx["cache"]) / "runs" / search["spec"]["run_id"] / "refit_cap3/checkpoint_000000.pt"
    cp0 = torch.load(initial, map_location="cpu", weights_only=False)
    equal_state(cp0["model"], fresh.state_dict())
    assert not torch.equal(fresh.E, old.E)
    assert not torch.equal(fresh.center, old.center)
    assert cp0["updates"] == 0 and not cp0["optimizer"]["state"]
    assert cp0["sampler"]["exposures"] == 0
    assert not list(Path(ctx["cache"]).rglob("*.npz"))
    segments = result["training_segments"]
    if search["selected_epoch"]:
        assert len(segments) == 1 and segments[0]["endpoint_epoch"] == search["selected_epoch"]
        assert segments[0]["phase"] == "fresh_A60_refit_cap3"
        assert np.isfinite(segments[0]["mean_observed_training_batch_loss"])
    else:
        assert segments == []
    assert training.refit(ctx, search, selection(ctx, [search])) == result


def test_refit_requires_actual_hash_bound_group_selection(ctx):
    with pytest.raises((KeyError, ValueError, FileNotFoundError)):
        training.refit(ctx, {}, dict(reference="report://absent.json", sha256="0" * 64))
    search = training.search(ctx, "A", 92601, common.INITIAL_EPOCHS)
    path = Path(ctx["report"]) / "BAD_SELECTION.json"
    common.sealed_json(path, dict(status="SEALED", selection_split="V_select", V_used=True))
    with pytest.raises(ValueError, match="A_stop-only"):
        training.refit(ctx, search, common.binding(path, ctx))
    assert not list(Path(ctx["cache"]).rglob("selected_bundle.pt"))


def test_cap_rule_strict_best_and_transfer_exact_tie_selects_a():
    curve = [dict(epoch=e, astop_mse=v) for e, v in [(0, 3.), (.25, 2.), (1, 1.), (3, .9)]]
    assert training.strict_last_best(curve, 3)
    assert not training.strict_last_best(curve[:-1] + [dict(epoch=3, astop_mse=1.)], 3)
    with pytest.raises(ValueError, match="through cap"):
        training.strict_last_best(curve[:-1], 3)
    searches = []
    for mode in ("A", "B"):
        for seed in common.SEEDS:
            searches.append(dict(status="COMPLETE", spec=training.job_spec("jena", mode, seed),
                summary=dict(selected_astop_mse=.5 if seed == 92601 else 1.5), V_used=False,
                selection_split="A_stop"))
    assert training.choose_transfer(searches)["selected_mode"] == "A"
    searches[-1]["summary"]["selected_astop_mse"] = 1.4
    assert training.choose_transfer(searches)["selected_mode"] == "B"
    with pytest.raises(ValueError, match="Both registered seeds"):
        training.choose_transfer(searches[:-1])
    searches[-1]["V_used"] = True
    with pytest.raises(ValueError, match="A_stop-only"):
        training.choose_transfer(searches)


def legacy_parent(ctx):
    fit, data = ctx["modules"]["fit"], ctx["modules"]["data"]
    spec = dict(common.RECIPE, run_id="legacy_S", family="B2", dataset="jena", latent_k=2,
        seed=92601, device="cpu", revin=False, init_kind="PCA", core_dtype="float32", fold="primary",
        scientific_identity="legacy-all-context", epochs=list(common.INITIAL_EPOCHS))
    directory = Path(ctx["cache"]) / "legacy_S"
    with revin.model_factory(ctx) as create:
        contract, sources = training._contract(ctx, spec)
        base, norm = create(spec, ctx["task"], 80, ctx["foundation"])
        initial = training._state_fingerprint(base.state_dict())
        stage = fit._stage_body(spec, ctx["task"], 80, (80, 96), list(common.INITIAL_EPOCHS),
            directory / "search", directory / "public", ctx["foundation"], ctx["deadline"], contract, base, norm)
    last = sorted((directory / "search").glob("checkpoint_*.pt"))[-1]
    result = dict(status="COMPLETE", spec=spec, contract=contract, source_files=sources,
        summary=stage["summary"], curves=stage["curves"], selected_epoch=stage["summary"]["selected_epoch"],
        initial_base_state_sha256=initial, cap=3, V_used=False, selection_split="A_stop",
        checkpoint_path=str(last), checkpoint_binding=common.binding(last, ctx))
    path = directory / "SEARCH_RESULT.json"
    common.sealed_json(path, result)
    return path, last


def test_verified_s_reuse_import_preserves_state_and_extension_is_exact(ctx):
    old_path, old_checkpoint = legacy_parent(ctx)
    before = torch.load(old_checkpoint, map_location="cpu", weights_only=False)
    streams = ctx["foundation"].streams
    reused = training.reuse_S_search(ctx, 92601, old_path, old_checkpoint)
    assert reused["baseline_work"] == "REUSED_VERIFIED_S"
    assert reused["training_segments"] == []
    assert "did not retain" in reused["training_segments_unavailable_reason"]
    assert ctx["foundation"].streams == streams
    proof = training.import_S_parent(ctx, reused)
    assert proof["changed_fields"] == ["contract"] and proof["exact_resume"]
    imported = torch.load(common.resolve_binding(proof["imported_checkpoint"], ctx), map_location="cpu", weights_only=False)
    for key in before:
        if key != "contract":
            equal_state(before[key], imported[key])
    assert common.sha(old_checkpoint) == common.sha(common.resolve_binding(proof["original_copy"], ctx))
    extended = training.search(ctx, "S", 92601, common.EXTENDED_EPOCHS)
    direct = training.search(alternative_context(ctx, "direct_s"), "S", 92601, common.EXTENDED_EPOCHS)
    a = torch.load(extended["last_checkpoint"], map_location="cpu", weights_only=False)
    b = torch.load(direct["last_checkpoint"], map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "sampler", "torch_rng", "cuda_rng", "updates"):
        equal_state(a[key], b[key])
    assert training.import_S_parent(ctx, reused) == proof
    assert [row["endpoint_epoch"] for row in extended["training_segments"]] == [4., 5., 6.]
    assert "before migration" in extended["training_segments_unavailable_reason"]


@pytest.mark.parametrize("alteration", ["optimizer", "source", "initialization", "sampler"])
def test_s_reuse_rejects_incomplete_or_changed_parent(ctx, alteration):
    path, checkpoint_path = legacy_parent(ctx)
    result = common.read(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if alteration == "optimizer":
        del checkpoint["optimizer"]
    elif alteration == "source":
        result["source_files"]["networks.py"] = "0" * 64
    elif alteration == "initialization":
        result["initial_base_state_sha256"] = "0" * 64
    else:
        checkpoint["sampler"]["position"] += 1
    torch.save(checkpoint, checkpoint_path)
    result["checkpoint_binding"] = common.binding(checkpoint_path, ctx)
    common.atomic_json(path, result)
    with pytest.raises(ValueError, match="optimizer|source|initial|sampler"):
        training.reuse_S_search(ctx, 92601, path, checkpoint_path)


def test_registered_specs_and_schedules_are_not_silent_overrides(ctx):
    with pytest.raises(ValueError, match="Registered"):
        training.job_spec("other", "A", 92601)
    with pytest.raises(ValueError, match="endpoint"):
        training.search(ctx, "A", 92601, [1, 3, 12])
    spec = training.job_spec("pems32", "B", 92602)
    assert spec["latent_k"] == 8 and spec["revin"] is False
    assert spec["scientific_identity"]["potential_endpoints"][-2:] == [9., 12.]


def test_train_observer_forwards_exact_arguments_result_and_rng_and_deduplicates(ctx, monkeypatch):
    module = ctx["modules"]["training"]
    sampler = SimpleNamespace(exposures=0)
    arguments = (object(), object(), np.arange(10), sampler, object(), 10, object())
    keywords = dict(effective_batch=16, microbatch=4, starting_updates=3)
    result = dict(status="SEGMENT_COMPLETE", updates=4, optimizer_updates_this_segment=1,
        segment_seconds=.125, mean_observed_training_batch_loss=1.25, gradient_norm_min=.2, gradient_norm_max=.4)
    expected_result = copy.deepcopy(result)
    seen = []

    def original(*args, **kwargs):
        seen.append((args, kwargs))
        sampler.exposures = 10
        return result

    monkeypatch.setattr(module, "train_segment", original)
    private, public = Path(ctx["cache"]) / "observer", Path(ctx["report"]) / "observer"
    spec = training._spec(ctx, "A", 92601)
    rng = torch.get_rng_state().clone()
    with training._observe_training_segments(ctx, spec, "search", private, public, [1.]) as rows:
        returned = module.train_segment(*arguments, **keywords)
        assert returned is result and result == expected_result
    assert module.train_segment is original
    assert all(a is b for a, b in zip(seen[0][0], arguments))
    assert seen[0][1] == keywords and torch.equal(rng, torch.get_rng_state())
    assert rows[0]["mean_observed_training_batch_loss"] == 1.25
    assert rows[0]["updates"] == 4 and rows[0]["starting_updates"] == 3
    sampler.exposures = 0
    with training._observe_training_segments(ctx, spec, "search", private, public, [1.]) as repeated:
        assert module.train_segment(*arguments, **keywords) is result
    assert len(repeated) == 1
    assert training._segment_rows(private / "TRAIN_SEGMENTS.jsonl") == repeated
    assert (private / "TRAIN_SEGMENTS.jsonl").read_bytes() == (public / "TRAIN_SEGMENTS.jsonl").read_bytes()


@pytest.mark.parametrize("error,status", [(KeyboardInterrupt, "INTERRUPTED"), (RuntimeError, "FAILED")])
def test_train_observer_preserves_failure_marker_and_restores_original(ctx, monkeypatch, error, status):
    module = ctx["modules"]["training"]
    sampler = SimpleNamespace(exposures=0)

    def original(*args, **kwargs):
        sampler.exposures = 1
        raise error("training interrupted")

    monkeypatch.setattr(module, "train_segment", original)
    private, public = Path(ctx["cache"]) / "failed", Path(ctx["report"]) / "failed"
    with pytest.raises(error):
        with training._observe_training_segments(ctx, training._spec(ctx, "B", 92601), "search", private, public, [1.]):
            module.train_segment(None, None, np.arange(10), sampler, None, 10, None)
    assert module.train_segment is original
    rows = training._segment_rows(private / "TRAIN_SEGMENTS.jsonl")
    assert len(rows) == 1 and rows[0]["status"] == status
    assert rows[0]["end_exposures"] == 1 and rows[0]["target_exposures"] == 10
    assert rows[0]["mean_observed_training_batch_loss"] is None
    assert rows[0]["error_type"] == error.__name__


def interrupt_after_bundle_write(ctx, monkeypatch):
    search = training.search(ctx, "B", 92601, common.INITIAL_EPOCHS)
    decision = selection(ctx, [search])
    fit = ctx["modules"]["fit"]
    original = fit._torch_new

    def interrupted(path, value):
        original(path, value)
        if Path(path).name == "selected_bundle.pt":
            raise RuntimeError("interrupt after selected bundle write")

    with monkeypatch.context() as patch:
        patch.setattr(fit, "_torch_new", interrupted)
        with pytest.raises(RuntimeError, match="after selected bundle"):
            training.refit(ctx, search, decision)
    artifact = Path(ctx["cache"]) / "runs" / search["spec"]["run_id"] / "cap3/selected_bundle.pt"
    assert artifact.exists() and not (artifact.parent / "REFIT_RESULT.json").exists()
    return search, decision, artifact


def test_refit_resumes_after_orphan_bundle_write_without_overwrite_or_training(ctx, monkeypatch):
    search, decision, artifact = interrupt_after_bundle_write(ctx, monkeypatch)
    before_hash = common.sha(artifact)
    before = torch.load(artifact, map_location="cpu", weights_only=False)
    streams = ctx["foundation"].streams
    result = training.refit(ctx, search, decision)
    assert result["status"] == "COMPLETE"
    assert common.sha(artifact) == before_hash
    assert ctx["foundation"].streams == streams
    assert result["refit_summary"] == before["refit_summary"]
    public = Path(ctx["report"]) / "runs" / search["spec"]["run_id"] / "cap3"
    assert common.read(public / "REFIT_RESULT.json") == result
    assert common.read(public / "BASE_FIT_RECEIPT.json") == result["fit_receipt"]


@pytest.mark.parametrize("window", ["private_result", "public_result"])
def test_refit_cached_private_result_restores_missing_public_evidence(ctx, monkeypatch, window):
    search = training.search(ctx, "A", 92601, common.INITIAL_EPOCHS)
    decision = selection(ctx, [search])
    private = Path(ctx["cache"]) / "runs" / search["spec"]["run_id"] / "cap3"
    public = Path(ctx["report"]) / "runs" / search["spec"]["run_id"] / "cap3"
    target = (private if window == "private_result" else public) / "REFIT_RESULT.json"
    original = common.sealed_json

    def interrupted(path, value):
        original(path, value)
        if Path(path) == target:
            raise RuntimeError("interrupt after result seal")

    with monkeypatch.context() as patch:
        patch.setattr(common, "sealed_json", interrupted)
        with pytest.raises(RuntimeError, match="after result seal"):
            training.refit(ctx, search, decision)
    cached = common.read(private / "REFIT_RESULT.json")
    assert not (public / "BASE_FIT_RECEIPT.json").exists()
    streams = ctx["foundation"].streams
    result = training.refit(ctx, search, decision)
    assert result == cached and ctx["foundation"].streams == streams
    assert common.read(public / "REFIT_RESULT.json") == cached
    assert common.read(public / "BASE_FIT_RECEIPT.json") == cached["fit_receipt"]


@pytest.mark.parametrize("tamper", ["weights", "normalization", "selected_epoch", "summary"])
def test_refit_rejects_tampered_orphan_bundle_against_completed_stage(ctx, monkeypatch, tamper):
    search, decision, artifact = interrupt_after_bundle_write(ctx, monkeypatch)
    bundle = torch.load(artifact, map_location="cpu", weights_only=False)
    if tamper == "weights":
        bundle["model"]["E"][0, 0] += 1
        bundle["model_state_sha"] = ctx["modules"]["training"].fingerprint({
            key: ctx["modules"]["data"].array_sha256(value.numpy()) for key, value in bundle["model"].items()})
    elif tamper == "normalization":
        bundle["normalization"]["mean"][0] += 1
    elif tamper == "selected_epoch":
        bundle["selected_search_epoch"] += 1
    else:
        bundle["refit_summary"]["updates"] += 1
    torch.save(bundle, artifact)
    tampered_hash = common.sha(artifact)
    with pytest.raises(ValueError, match="Existing fresh-A60 bundle"):
        training.refit(ctx, search, decision)
    assert common.sha(artifact) == tampered_hash
    assert not (artifact.parent / "REFIT_RESULT.json").exists()
