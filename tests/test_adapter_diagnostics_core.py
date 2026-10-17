import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def core():
    try:
        return importlib.import_module('research.adapter_diagnostics_20261017_v2.core')
    except ModuleNotFoundError:
        pytest.fail('Campaign core is not implemented')


def test_sealed_identity_rejects_changed_recipe():
    c = core()
    plan = c.seal_plan({'jobs': [{'job_id': 'x', 'seed': 1}], 'source_hashes': {}})
    c.validate_plan(plan)
    plan['jobs'][0]['seed'] = 2
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_completion_requires_artifacts_and_reuse_verifies_bytes(tmp_path):
    c = core()
    receipt = tmp_path / 'receipt.json'
    with pytest.raises(ValueError):
        c.finish_job(receipt, 'identity', {'status': 'COMPLETE', 'artifact_paths': []}, tmp_path)
    output = tmp_path / 'score.json'
    output.write_text('{"mse": 1.5}')
    c.finish_job(receipt, 'identity', {'status': 'COMPLETE', 'artifact_paths': [str(output)]}, tmp_path)
    assert c.reusable(receipt, 'identity', tmp_path)
    assert not c.reusable(receipt, 'different', tmp_path)
    output.write_text('{"mse": 9}')
    assert not c.reusable(receipt, 'identity', tmp_path)


def test_budget_preserves_reserve_and_campaign_start(tmp_path):
    c = core()
    p = tmp_path / 'campaign.json'
    first = c.campaign_ledger(p, now=1000)
    again = c.campaign_ledger(p, now=2000)
    assert first['started_unix'] == again['started_unix'] == 1000
    assert c.dispatch_allowed(first, 100, 0, now=1100)
    assert not c.dispatch_allowed(first, 18000, 0, now=1100)
    assert not c.dispatch_allowed(first, 100, 12000, now=1100)
    assert c.dispatch_allowed(first, 100, 100, now=22000, final=True)


def test_atomic_json_is_finite_and_no_partial_replace(tmp_path):
    c = core()
    p = tmp_path / 'out.json'
    c.atomic_json(p, {'good': True})
    with pytest.raises(ValueError):
        c.atomic_json(p, {'bad': float('nan')})
    assert json.loads(p.read_text()) == {'good': True}


def test_reuse_rejects_private_artifact_mutation(tmp_path):
    c = core()
    public, private = tmp_path/'report', tmp_path/'cache'
    public.mkdir(); private.mkdir()
    prediction = private/'prediction.npz'
    prediction.write_bytes(b'prediction')
    output = public/'result.json'
    c.atomic_json(output, {'private_artifacts': [{'reference':'cache://prediction.npz','sha256':c.sha(prediction)}]})
    receipt = public/'receipt.json'
    c.finish_job(receipt, 'x', {'status':'COMPLETE','artifact_paths':[str(output)]}, public)
    assert c.reusable(receipt, 'x', public, private_roots={'cache':private})
    prediction.write_bytes(b'changed')
    assert not c.reusable(receipt, 'x', public, private_roots={'cache':private})


def test_default_invocation_is_source_only(tmp_path):
    c = core()
    root = Path(c.__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-m', 'research.adapter_diagnostics_20261017_v2.run'],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'SOURCE_ONLY' in result.stdout


def test_amendment_preserves_unchanged_completed_jobs(tmp_path, monkeypatch):
    from research.adapter_diagnostics_20261017_v2 import run
    c=core()
    report=tmp_path/'report'; report.mkdir()
    (report/'receipts').mkdir(); (report/'jobs').mkdir()
    identities={'stable':'same','changed':'old'}
    monkeypatch.setattr(run,'source_hashes',lambda _: {})
    monkeypatch.setattr(run,'planned_jobs',lambda _: [dict(job_id=k,scientific_identity=v) for k,v in identities.items()])
    ctx=dict(repo=tmp_path,report=report,ledger=dict(limits={}))
    run.make_plan(ctx)
    for name in identities:
        c.atomic_json(report/'receipts'/(name+'.json'),dict(status='COMPLETE'))
        c.atomic_json(report/'jobs'/(name+'_handler.json'),dict(status='COMPLETE'))
    identities['changed']='new'
    run.make_plan(ctx,'Fix changed implementation only')
    assert (report/'receipts/stable.json').exists()
    assert (report/'jobs/stable_handler.json').exists()
    assert not (report/'receipts/changed.json').exists()
