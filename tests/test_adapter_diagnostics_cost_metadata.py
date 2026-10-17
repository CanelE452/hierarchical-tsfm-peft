"""Source-derived cost metadata uses existing evidence and rejects contradictory identity/dtype."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.adapter_diagnostics_20261017_v2.core import atomic_json, seal_plan, sha


PATH = Path(__file__).resolve().parents[1]/"reports/adapter_diagnostics_20261017_v2/code/pems32_cost_metadata.py"
SPEC = importlib.util.spec_from_file_location("diagnostic_pems32_cost_metadata", PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def fixture(tmp_path, monkeypatch):
    repo, report, cache, asset = tmp_path, tmp_path/helper.REPORT, tmp_path/"cache", tmp_path/"asset"
    report.mkdir(parents=True); cache.mkdir(); asset.mkdir()
    source_hashes = {}
    for name in helper.SOURCE_PINS:
        path = repo/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("sealed "+name)
        source_hashes[name] = sha(path)
    monkeypatch.setattr(helper, "SOURCE_PINS", source_hashes)
    reader = report/"code/trajectory_diagnostics.py"; reader.parent.mkdir(); reader.write_text("metadata-reader-fixture")
    weight = asset/"weights"; weight.write_bytes(b"opaque weight fixture")
    subset = dict(column_selection=dict(selected_ids=[str(i) for i in range(32)]), foundation_tensor_sha256="f"*64,
                  model_snapshot_bindings=[dict(resolved_blob_relative_path="weights", sha256=sha(weight))])
    atomic_json(repo/helper.BROAD/"priority_pems_subset_20261016_v1/PLAN.json", subset)
    monkeypatch.setattr(helper, "SUBSET_PLAN_SHA256", sha(repo/helper.BROAD/"priority_pems_subset_20261016_v1/PLAN.json"))
    jobs = [dict(job_id="E01_B0_b1", question_id="E01", scientific_identity="identity-cpu", seed=1),
            dict(job_id="E03_jena_gpu", question_id="E03", dataset_cohort="jena", scientific_identity="identity-gpu")]
    atomic_json(report/"EXPERIMENT_PLAN.json", seal_plan(dict(jobs=jobs, source_hashes=source_hashes)))
    bundle = cache/"runs/pems_B0/selected_bundle.pt"; bundle.parent.mkdir(parents=True); bundle.write_bytes(b"opaque checkpoint fixture")
    contract = dict(model_sha="f"*64, data_sha="d"*64)
    payload = dict(spec=dict(family="B0", dataset="pems08", latent_k=8, seed=1, device="cuda"), contract=contract,
                   refit_summary=dict(status="SEGMENT_COMPLETE", contract=contract, trainable_parameters=2048),
                   model={key: SimpleNamespace(shape=shape, storage=dict(dtype="<f4")) for key, shape in
                          (("E", (32, 8)), ("D", (7, 8, 32)), ("center", (32,)))})
    monkeypatch.setattr(helper, "_checkpoint_metadata", lambda ctx, path: copy.deepcopy(payload))
    e01 = dict(status="COMPLETE_PROTOCOL", family="B0", batch_size=1, device="cpu", CPU_threads=2, C=32, K=8,
               physical_output_and_inverse_dtype="float64", forward_parity=dict(status="PASS"), artifact_sha256=sha(bundle),
               source_protocol=dict(original_scientific_sources={Path(name).name: digest for name, digest in source_hashes.items() if name.startswith(helper.BROAD+"/")},
                                    original_receipts=[dict(data_sha256="d"*64)]),
               memory=dict(resident_weights_cpu_MiB=(10+2080)*4/1024**2))
    e03 = dict(rows=[dict(method="Bolt_Raw", B=batch, status="COMPLETE", parity="PASS", core_dtype="float32", trainable_params=0,
                         total_resident_parameters=10) for batch in (1, 8)], private_artifacts=[dict(reference="asset://weights", sha256=sha(weight))])
    results = {}
    for job, result, relative in ((jobs[0], e01, "costs/pems_B0_b1/RESULT.json"), (jobs[1], e03, "costs/E03_JENA_BACKBONES.json")):
        output = report/relative; atomic_json(output, result)
        receipt = report/"receipts"/(job["job_id"]+".json")
        atomic_json(receipt, dict(status="COMPLETE", scientific_identity=job["scientific_identity"], result=result, artifact_hashes={relative: sha(output)}))
        results[job["job_id"]] = (job, result, receipt)
    data = dict(costs=[dict(job_id=jobs[0]["job_id"], cohort="pems32", method="B0", B=1, dtype="source_bound")], bindings={})
    return dict(repo=repo, report=report, cache=cache, asset=asset), results, data, payload


def test_cost_metadata_uses_dynamic_e03_count_and_existing_measured_buffers(tmp_path, monkeypatch):
    ctx, results, data, _ = fixture(tmp_path, monkeypatch)
    proof = helper.complete_cost_metadata(ctx, results, data)
    row = data["costs"][0]
    assert row["core_dtype"] == "float32" and row["device"] == "cpu"
    assert row["backbone_resident_parameters"] == 10
    assert row["total_resident_model_params"] == 2058
    assert row["resident_weights_and_buffers_cpu_bytes"] == 8360
    assert row["resident_buffer_bytes_above_parameters"] == 128
    assert proof["checked_rows"] == 1 and proof["tensor_storage_loaded"] is False
    assert len(proof["private_artifacts"]) == 2 and data["bindings"]
    assert "no new parameter measurement" in row["resident_parameter_scope"]


def test_adapter_dtype_mismatch_is_rejected_before_cost_mutation(tmp_path, monkeypatch):
    ctx, results, data, payload = fixture(tmp_path, monkeypatch)
    payload["model"]["E"].storage["dtype"] = "<f8"
    with pytest.raises(ValueError, match="dtype/shape"):
        helper.complete_cost_metadata(ctx, results, data)
    assert data["costs"][0]["dtype"] == "source_bound"


@pytest.mark.parametrize("kind", ("source", "weights", "spec", "foundation"))
def test_cost_metadata_identity_mismatches_are_rejected(tmp_path, monkeypatch, kind):
    ctx, results, data, payload = fixture(tmp_path, monkeypatch)
    if kind == "source":
        (ctx["repo"]/next(iter(helper.SOURCE_PINS))).write_text("changed")
    elif kind == "weights":
        (ctx["asset"]/"weights").write_bytes(b"changed")
    elif kind == "spec":
        payload["spec"]["family"] = "B2"
    else:
        payload["contract"]["model_sha"] = "a"*64
    with pytest.raises(ValueError):
        helper.complete_cost_metadata(ctx, results, data)
    assert data["costs"][0]["dtype"] == "source_bound"
