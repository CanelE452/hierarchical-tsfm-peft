"""Interrupted fresh-process facets retain prior attempts on restart."""
from pathlib import Path

import pytest

from research.raw_context_multiscale_20261018_v1 import common,costs,pipeline


@pytest.fixture
def context(tmp_path,monkeypatch):
    ctx = dict(repo=tmp_path,asset=tmp_path/'asset',cache=tmp_path/'cache',report=tmp_path/'report',deadline=1e12)
    for key in ('asset','cache','report'):
        ctx[key].mkdir()
    prediction = ctx['cache']/'prediction.npz'
    prediction.write_bytes(b'synthetic-prediction')
    arm = dict(run_id='fixture',dataset='jena',mode='S',route_spec=dict(family='RAW'),
               prediction_binding=common.binding(prediction,ctx))
    monkeypatch.setattr(costs,'validate_receipt',lambda value,*args: value['status']=='COMPLETE')
    calls = []
    def subprocess_run(command,**kwargs):
        output = Path(command[command.index('--output')+1])
        calls.append(output)
        common.sealed_json(output,dict(status='COMPLETE'))
    monkeypatch.setattr(pipeline.subprocess,'run',subprocess_run)
    return ctx,arm,calls


@pytest.mark.parametrize('interruption',['invalid_receipt','orphan_trials'])
def test_interrupted_cost_uses_new_process_attempt_and_reuses_success(context,interruption):
    ctx,arm,calls = context
    output = ctx['report']/'costs/fixture_native_B1.json'
    output.parent.mkdir()
    trials = output.with_suffix('.trials.csv')
    trials.write_bytes(b'partial-prior-measurements')
    if interruption=='invalid_receipt':
        common.sealed_json(output,dict(status='INVALID',reason='synthetic interrupted measurement'))
    preserved = trials.read_bytes()
    result = pipeline._costs(ctx,[arm])
    assert calls[0].name=='fixture_native_B1.attempt001.json'
    assert trials.read_bytes()==preserved
    assert len(result)==2
    first = (ctx['report']/'COST_INDEX.json').read_bytes()
    assert pipeline._costs(ctx,[arm])==result
    assert len(calls)==2
    assert (ctx['report']/'COST_INDEX.json').read_bytes()==first


def test_complete_cost_provenance_failure_is_not_hidden_by_retry(context,monkeypatch):
    ctx,arm,calls = context
    output = ctx['report']/'costs/fixture_native_B1.json'
    common.sealed_json(output,dict(status='COMPLETE'))
    def changed(*args):
        raise ValueError('Changed completed source identity')
    monkeypatch.setattr(costs,'validate_receipt',changed)
    with pytest.raises(ValueError,match='Changed completed source identity'):
        pipeline._costs(ctx,[arm])
    assert not calls
