import json

import pytest

from . import execute


def fixture_plan(tmp_path, monkeypatch, *, failed=(), new_k=False, spent=100):
    monkeypatch.setattr(execute, 'REPORT', tmp_path)
    monkeypatch.setattr(execute, 'elapsed', lambda: spent)
    monkeypatch.setattr(execute, 'clock', lambda: {'limit_seconds': 24300})
    decisions = {d: {'reused_historical_results': d not in failed} for d in execute.DATASETS}
    (tmp_path/'G4_RECHECK_DECISIONS.json').write_text(json.dumps(decisions))
    (tmp_path/'PREREGISTRATION.json').write_text('{}')
    (tmp_path/'RUN_PLAN_PREFLIGHT.json').write_text('{}')
    ks = dict(jena=5 if new_k else 6, ETTh1=4, ETTh2_safe17400=4, appliances=14)
    execute.csv_write(tmp_path/'K_TAU.csv', [dict(dataset=d,tau=.9,K_A_fit=k) for d,k in ks.items()])
    monkeypatch.setattr(execute, 'baseline', lambda d,k: dict(
        seed_MSE_json='{"92601":0.3,"92602":0.4}',
        seed_MAE_json='{"92601":0.2,"92602":0.3}', source_paths_json='["old.json"]'))


def test_matching_registered_width_reuses_only_fixed_first_two_seeds(tmp_path, monkeypatch):
    fixture_plan(tmp_path, monkeypatch)
    plan=execute.make_plan()
    assert len(plan['jobs'])==32 and len(plan['reused'])==8
    assert {e['job']['seed'] for e in plan['jobs']}==set(range(92603,92611))
    assert plan['status']=='ADMITTED'


def test_new_rule_width_trains_all_ten_without_reselecting(tmp_path, monkeypatch):
    fixture_plan(tmp_path, monkeypatch,new_k=True)
    plan=execute.make_plan()
    jena=[e['job'] for e in plan['jobs'] if e['job']['cohort']=='jena']
    assert {j['seed'] for j in jena}==set(execute.SEEDS)
    assert {j['K'] for j in jena}=={5}
    assert len(plan['reused'])==6


def test_failed_fingerprint_registers_six_matched_comparisons(tmp_path, monkeypatch):
    fixture_plan(tmp_path, monkeypatch,failed=('jena',))
    plan=execute.make_plan()
    rows=[e for e in plan['jobs'] if e['job']['cohort']=='jena']
    assert len([e for e in rows if e['role']=='main'])==10
    assert {(e['job']['K'],e['job']['seed']) for e in rows if e['role']=='comparison'}=={
        (k,s) for k in (3,6,12) for s in (92601,92602)}


def test_fixed_budget_refuses_a_plan_before_training(tmp_path, monkeypatch):
    fixture_plan(tmp_path, monkeypatch,spent=24000)
    assert execute.make_plan()['status']=='STOP_BUDGET'


@pytest.mark.parametrize('seed',[92600,92611])
def test_out_of_range_seed_is_not_an_experiment(seed):
    with pytest.raises(ValueError):execute.job('jena',6,seed)
