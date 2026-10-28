"""CPU-only fixed-group scheduling and result-integrity regressions."""
import copy
import json

import pytest

from research.crossdata_component_screen_20261028_v1 import controller, run_one, runtime


def job(method, dataset, seed):
    item = runtime.DATASETS[dataset]
    return dict(job_id=f'XC_{method}_{dataset}_K{item["K"]}_s{seed}',
                method_id=method, dataset=dataset, K=item['K'], seed=seed, training_cap=item['cap'])


@pytest.fixture
def screen(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'REPORT', tmp_path/'reports')
    monkeypatch.setattr(runtime, 'CACHE', tmp_path/'cache')
    monkeypatch.setattr(runtime, 'SHARED_CACHE', tmp_path/'shared')
    monkeypatch.setattr(run_one, 'validate_registration', lambda item: {'seal_sha256': 'synthetic'})
    monkeypatch.setattr(runtime, 'guard_preserved', lambda: {'checked': 0, 'changed': []})
    plan = dict(jobs=[job(method, dataset, seed) for dataset in ('pems32','beijing')
                     for seed in (92601,92602) for method in controller.METHODS],
                optional_jobs=[job(method, dataset, 92603) for dataset in ('pems32','beijing')
                               for method in controller.METHODS],
                group_order=list(controller.GROUP_SPECS), extension_jobs=[])
    protocol = dict(training_deadline='2030-01-01T00:00:00+00:00', controller_policy=dict(
        dataset_job_seconds={'pems32':1020,'beijing':650}, reestimate_floor_seconds={'pems32':850,'beijing':500},
        reestimate_multiplier=1.35, admission_margin_seconds=120, retries=1))
    runtime.atomic(runtime.REPORT/'RUN_PLAN.json',plan)
    runtime.atomic(runtime.REPORT/'PREREGISTRATION.json',protocol)
    return controller.Controller()


def finish(screen, item, epoch=1., wall=400.):
    screen.state['run_statuses'][item['job_id']] = 'COMPLETE'
    screen.state['attempts'][item['job_id']] = [dict(attempt=1,exit_code=0,wall_seconds=wall)]
    runtime.atomic(runtime.REPORT/'jobs'/item['job_id']/'RESULT.json',
                   dict(row=dict(selected_epoch=epoch,MSE='never used for admission')))


@pytest.mark.parametrize('mutation',['duplicate','missing_arm','wrong_group_order','extension'])
def test_plan_requires_exact_fixed_groups(screen,mutation):
    if mutation == 'duplicate': screen.mandatory[1] = copy.deepcopy(screen.mandatory[0])
    elif mutation == 'missing_arm': screen.mandatory.pop()
    elif mutation == 'wrong_group_order': screen.plan['group_order'].reverse()
    else: screen.plan['extension_jobs'] = [copy.deepcopy(screen.mandatory[0])]
    screen.jobs = screen.mandatory+screen.optional
    with pytest.raises(ValueError): screen.validate_plan()


def test_first_full_wall_time_updates_estimate_without_result_scores(screen,monkeypatch):
    first = job('OURS','pems32',92601)
    assert screen.job_estimate(first) == 1020
    finish(screen,first,wall=400)
    original_read = runtime.read
    def forbid_score(path):
        if path.name == 'RESULT.json': raise AssertionError('Admission read a performance result')
        return original_read(path)
    monkeypatch.setattr(runtime,'read',forbid_score)
    assert screen.job_estimate(first) == 850
    second = job('TIME_ONLY','pems32',92601)
    screen.state['run_statuses'][second['job_id']] = 'COMPLETE'
    screen.state['attempts'][second['job_id']] = [dict(exit_code=0,wall_seconds=900)]
    assert screen.job_estimate(first) == 1215
    screen.state['attempts'][second['job_id']].append(dict(exit_code=1,wall_seconds=9999))
    assert screen.job_estimate(first) == 1215
    assert screen.job_estimate(job('OURS','beijing',92601)) == 650


def test_whole_base_group_budget_excludes_every_unstarted_arm(screen,monkeypatch):
    calls=[]
    monkeypatch.setattr(screen,'run',lambda item:calls.append(item))
    monkeypatch.setattr(controller.time,'time',lambda:100.)
    screen.deadline=100+8*1020+119
    assert screen.run_group('pems32_base') is False
    assert not calls
    assert set(screen.state['run_statuses'].values()) == {'NOT_RUN_BUDGET'}
    assert len(screen.state['run_statuses']) == 8


def test_seed3_requires_entire_base_group(screen,monkeypatch):
    for item in screen.group_jobs('pems32_base')[:-1]: finish(screen,item)
    calls=[]
    monkeypatch.setattr(screen,'run',lambda item:calls.append(item))
    assert screen.run_group('pems32_seed3') is False
    assert not calls
    assert all(screen.state['run_statuses'][item['job_id']] == 'BASE_INCOMPLETE'
               for item in screen.group_jobs('pems32_seed3'))


def test_seed3_budget_admits_all_four_or_none(screen,monkeypatch):
    for item in screen.group_jobs('beijing_base'): finish(screen,item)
    monkeypatch.setattr(controller.time,'time',lambda:100.)
    screen.deadline=100+4*540+119
    calls=[]
    monkeypatch.setattr(screen,'run',lambda item:calls.append(item))
    assert screen.run_group('beijing_seed3') is False
    assert not calls


def test_group_does_not_repeat_existing_success(screen,monkeypatch):
    first=screen.group_jobs('pems32_base')[0];finish(screen,first)
    monkeypatch.setattr(controller,'verify_result',lambda item: {'status':'COMPLETE'})
    called=[]
    def run(item):
        if screen.state['run_statuses'].get(item['job_id'])!='COMPLETE': called.append(item['job_id'])
        finish(screen,item)
    monkeypatch.setattr(screen,'run',run)
    assert screen.run_group('pems32_base')
    assert first['job_id'] not in called and len(called)==7


@pytest.mark.parametrize('optional_completed',[0,1,3,4])
def test_partial_third_seed_never_changes_primary_group(screen,optional_completed):
    for item in screen.group_jobs('pems32_base'): finish(screen,item,epoch=6)
    for item in screen.group_jobs('pems32_seed3')[:optional_completed]: finish(screen,item,epoch=1)
    screen.summarize_groups()
    expected=3 if optional_completed==4 else 2
    ds=screen.state['dataset_status']['pems32']
    assert ds['primary_n']==expected and len(ds['primary_job_ids'])==4*expected
    assert ds['pilot']==(expected==2)
    assert ds['partial_seed3_count']==optional_completed
    for method in controller.METHODS:
        status=screen.state['method_status'][method+':pems32']
        assert status['primary_n']==expected and status['status']=='UNDERTRAINED_PROVISIONAL'


def test_two_seed_noncap_is_pilot_and_three_seed_noncap_complete(screen):
    for item in screen.group_jobs('beijing_base'): finish(screen,item,epoch=1)
    screen.summarize_groups()
    assert screen.state['method_status']['OURS:beijing']['status']=='PILOT'
    for item in screen.group_jobs('beijing_seed3'): finish(screen,item,epoch=1)
    screen.summarize_groups()
    assert screen.state['method_status']['OURS:beijing']['status']=='COMPLETE'
    assert not screen.state['method_status']['OURS:beijing']['provisional']


def test_incomplete_base_has_no_primary_even_with_completed_method(screen):
    for item in screen.group_jobs('pems32_base')[:5]: finish(screen,item)
    screen.summarize_groups()
    assert screen.state['dataset_status']['pems32']['primary_n']==0
    assert all(screen.state['method_status'][method+':pems32']['primary_n']==0 for method in controller.METHODS)


def test_scientific_failure_receipt_is_not_automatically_retried(screen):
    item=screen.jobs[0]
    runtime.atomic(runtime.REPORT/'jobs'/item['job_id']/'WORKER_RECEIPT_attempt1.json',dict(exit_code=20))
    with pytest.raises(ValueError,match='Scientific failure'):screen.next_attempt(item)


def test_technical_retry_and_exhaustion_are_bounded(screen):
    item=screen.jobs[0]
    runtime.atomic(runtime.REPORT/'jobs'/item['job_id']/'WORKER_RECEIPT_attempt1.json',dict(exit_code=1))
    assert screen.next_attempt(item)==2
    runtime.atomic(runtime.REPORT/'jobs'/item['job_id']/'WORKER_RECEIPT_attempt2.json',dict(exit_code=1))
    assert screen.next_attempt(item)==3
