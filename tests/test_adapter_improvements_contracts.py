import copy
import pytest

from research.adapter_improvements_20261018_v1 import contracts as c


def test_resumed_campaign_keeps_start_usage_and_limits(tmp_path):
    path = tmp_path / "campaign.json"
    first = c.campaign_ledger(path, started_unix=100, lineage=[{"campaign": "old", "gpu_seconds": 7}])
    first["gpu_seconds"] = 123
    c.atomic_json(path, first)
    resumed = c.campaign_ledger(path, started_unix=999)
    assert resumed["started_unix"] == 100
    assert resumed["gpu_seconds"] == 123
    assert resumed["lineage"] == first["lineage"]


def test_complete_pair_admission_preserves_reserves_and_80percent_stop():
    ledger = dict(started_unix=100, limits=c.DEFAULT_LIMITS, gpu_seconds=0)
    assert c.admit_bundle(ledger, 1000, 1000, now=100)
    assert not c.admit_bundle(ledger, 18000, 1000, now=100)
    assert not c.admit_bundle(ledger, 100, 100, now=100 + .8 * 21600)
    ledger["gpu_seconds"] = .8 * 14400
    assert not c.admit_bundle(ledger, 0, 0, now=100)
    assert c.admit_bundle(ledger, 10, 10, now=100, final=True)


def test_private_binding_rejects_traversal_and_modified_bytes(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    p = cache / "feature.bin"
    p.write_bytes(b"bound")
    ctx = dict(cache=cache)
    bound = c.binding(p, ctx)
    assert c.resolve_binding(bound, ctx) == p
    bad = dict(bound, reference="cache://../escape")
    with pytest.raises(ValueError):
        c.resolve_binding(bad, ctx)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        c.resolve_binding(bound, ctx)


def test_plan_rejects_forward_dependencies_and_tampered_scientific_source(tmp_path):
    p = tmp_path / "science.py"
    p.write_text("a=1\n")
    job = c.seal_job(dict(job_id="candidate", dependency_ids=[], family="P_SPLIT"), {"science.py": c.sha(p)})
    plan = c.seal_plan(dict(source_hashes={"science.py": c.sha(p)}, jobs=[job]))
    c.validate_plan(plan, tmp_path)
    p.write_text("a=2\n")
    with pytest.raises(ValueError, match="source"):
        c.validate_plan(plan, tmp_path)
    bad = copy.deepcopy(plan)
    bad["jobs"][0]["dependency_ids"] = ["future"]
    bad = c.seal_plan(bad)
    with pytest.raises(ValueError, match="identity|Dependency"):
        c.validate_plan(bad)


def test_only_selects_registered_exact_job_ids():
    plan = dict(jobs=[dict(job_id="p_s1"), dict(job_id="p_s2")])
    assert [j["job_id"] for j in c.select_jobs(plan, ["p_s2"])] == ["p_s2"]
    with pytest.raises(ValueError, match="registered"):
        c.select_jobs(plan, ["p"])
