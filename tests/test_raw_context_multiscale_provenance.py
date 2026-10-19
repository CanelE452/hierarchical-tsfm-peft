"""Exact cost provenance checks using synthetic receipts and source bytes only."""
import copy
import json
from pathlib import Path

import pytest

from research.raw_context_multiscale_20261018_v1 import common, costs


@pytest.fixture
def bound_receipt(tmp_path):
    ctx = dict(repo=common.REPO, cache=tmp_path / "cache", report=tmp_path / "report", asset=tmp_path)
    request = dict(asset_root=str(tmp_path), dataset="jena", accuracy_run_id="synthetic_A_jena",
                   cost_id="synthetic_A_jena_native_B8", batch_size=8, device="cpu",
                   route_spec=dict(family="B2", run_id="synthetic_A_jena", latent_k=12,
                       artifact_cache_relative_path="synthetic/checkpoint_a.pt", artifact_sha256="a" * 64),
                   prediction_path="synthetic/prediction.npz", prediction_sha256="b" * 64)
    path = ctx["cache"] / "cost_requests" / "synthetic.json"
    common.sealed_json(path, request)
    receipt = dict(status="COMPLETE", request_contract_sha256=costs.request_identity(request),
                   route_spec=copy.deepcopy(request["route_spec"]), producer_sources=costs.producer_sources(ctx),
                   prediction_reference_sha256=request["prediction_sha256"], request_binding=common.binding(path, ctx),
                   summary=dict(MSE=.25, MAE=.3, p50_ms=1.))
    return ctx, request, path, receipt


def test_complete_receipt_requires_exact_request_file_and_actual_source_graph(bound_receipt):
    ctx, request, path, receipt = bound_receipt
    assert costs.validate_receipt(receipt, request, ctx, path) is receipt
    sources = receipt["producer_sources"]
    assert "research/raw_context_multiscale_20261018_v1/costs.py" in sources
    assert "research/raw_context_multiscale_20261018_v1/deployment.py" in sources
    assert "research/informs_broad_validation_20261015_v1/gpu_runtime.py" in sources
    assert all((Path(ctx["repo"]) / relative).is_file() and len(digest) == 64
               for relative, digest in sources.items())
    with pytest.raises(ValueError, match="binding changed"):
        costs.validate_receipt(dict(status="COMPLETE"), request, ctx, path)


def test_same_accuracy_id_cannot_substitute_a_different_checkpoint(bound_receipt):
    ctx, request, _, receipt = bound_receipt
    changed = copy.deepcopy(request)
    changed["route_spec"]["artifact_cache_relative_path"] = "synthetic/checkpoint_b.pt"
    changed["route_spec"]["artifact_sha256"] = "c" * 64
    assert changed["accuracy_run_id"] == request["accuracy_run_id"]
    receipt["request_contract_sha256"] = costs.request_identity(changed)
    with pytest.raises(ValueError, match="route/source/prediction"):
        costs.validate_receipt(receipt, changed, ctx)


def test_changed_producer_bytes_invalidate_previously_complete_timing(bound_receipt, monkeypatch):
    ctx, request, path, receipt = bound_receipt
    original = common.sha
    changed_path = Path(costs.__file__).resolve()
    def changed_source(path):
        return "f" * 64 if Path(path).resolve() == changed_path else original(path)
    monkeypatch.setattr(common, "sha", changed_source)
    with pytest.raises(ValueError, match="source/prediction"):
        costs.validate_receipt(receipt, request, ctx, path)


def test_json_equivalent_request_with_changed_bytes_is_not_the_bound_file(bound_receipt):
    ctx, request, path, receipt = bound_receipt
    before = path.read_bytes()
    path.write_text(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf8")
    assert common.read(path) == request
    assert path.read_bytes() != before
    assert costs.request_identity(common.read(path)) == receipt["request_contract_sha256"]
    with pytest.raises(ValueError, match="exact sealed request file"):
        costs.validate_receipt(receipt, request, ctx, path)


@pytest.mark.parametrize("field,value", [
    ("request_contract_sha256", "0" * 64),
    ("prediction_reference_sha256", "0" * 64),
    ("producer_sources", {}),
])
def test_individual_scientific_bindings_are_required(bound_receipt, field, value):
    ctx, request, path, receipt = bound_receipt
    receipt[field] = value
    with pytest.raises(ValueError, match="binding changed"):
        costs.validate_receipt(receipt, request, ctx, path)


def test_ephemeral_flags_and_deadline_preserve_scientific_request_identity(bound_receipt):
    ctx, request, path, receipt = bound_receipt
    runtime_request = dict(request, _fresh_process_cli=True, _deadline_unix=1234., _runtime_attempt="retry-2")
    assert costs.request_identity(runtime_request) == costs.request_identity(request)
    assert costs.validate_receipt(receipt, runtime_request, ctx, path) is receipt
    changed = dict(runtime_request, batch_size=1)
    assert costs.request_identity(changed) != costs.request_identity(request)


def test_cost_deadline_is_checked_without_loading_any_resources(monkeypatch):
    monkeypatch.setattr(costs.time, "time", lambda: 100.)
    with pytest.raises(TimeoutError, match="budget reached"):
        costs._deadline(dict(_deadline_unix=100.))
    costs._deadline(dict(_deadline_unix=101.))
