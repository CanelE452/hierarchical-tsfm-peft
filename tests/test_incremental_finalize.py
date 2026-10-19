import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from research.incremental_latent_adaptation_20261019_v1 import common


NAME='research.incremental_latent_adaptation_20261019_v1.finalize'


def module():
    assert importlib.util.find_spec(NAME) is not None, 'CPU packaging helper is missing'
    return importlib.import_module(NAME)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def stage(phase,zero=False):
    return dict(status='COMPLETE',phase=phase,selected_epoch=0 if zero else 1,
                curves=[{'epoch':0},{'epoch':3 if phase=='search' else 1}],
                updates=0 if zero else 3,exposures=0 if zero else 33,
                actual_epochs=0 if zero else 1,training_seconds=1.,evaluation_seconds=.1,
                checkpoint_seconds=.01,training_F_forward_rows=33,
                training_F_backward_rows=33,training_F_physical_rows=33)


@pytest.fixture
def campaign(tmp_path,monkeypatch):
    report,cache=tmp_path/'reports',tmp_path/'cache'
    report.mkdir();cache.mkdir()
    monkeypatch.setattr(common,'REPO',tmp_path)
    monkeypatch.setattr(common,'REPORT',report)
    monkeypatch.setattr(common,'CACHE',cache)
    monkeypatch.setattr(common,'ledger',lambda:dict(start_unix=10,elapsed_seconds=20,
        GPU_owned_process_seconds=5,sessions=[]))
    put(report/'EXECUTION_START.json',{'python':'python','start_unix':10})
    put(report/'PARENT_AND_DATA_CONTRACT.json',dict(status='PASS',parents=[],parent_cache_count=8))
    put(report/'C_ADMISSION.json',{'datasets':{'jena':{'status':'ADMITTED'},
        'pems32':{'status':'NOT_ADMITTED_NO_INCREMENTAL_EVIDENCE'}}})
    for dataset in common.DATASETS:
        for seed in common.SEEDS:
            for method in common.METHODS:
                directory=cache/'runs'/dataset/f'{method}_s{seed}'
                zero=method=='RESIDUAL_RAW_MLP' and seed==92602
                put(directory/'search/RESULT.json',stage('search',zero))
                put(directory/'refit/RESULT.json',stage('refit',zero))
            put(cache/'probes'/dataset/f'alpha_s{seed}.json',{'fit_alpha':{},'refit_alpha':{},'rows':[{}]*4})
        selected={kind:{'fit_refit_seconds':.1} for kind in ('ZERO','MEAN','OWN','ALL')}
        put(cache/'probes'/dataset/'RESULT.json',{'selected':selected})
        for kind in selected:
            (cache/'probes'/dataset/(kind+'.npz')).write_bytes(b'synthetic probe artifact; never deserialize')
    for method in ('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'):
        for seed in common.SEEDS:
            count=2 if method=='SEQ_BLOCKS' else 1
            value=dict(status='COMPLETE',dataset='jena',method=method,seed=seed,cap=3,
                selected_epoch=[1,1] if count==2 else 1,selected_prefix=6,
                searches=[stage('search') for _ in range(count)],
                refits=[stage('refit') for _ in range(count)],
                foundation_counts_all_work={'streams':33,'physical_streams':33,'calls':0},
                stage1_caches=[{'phase':'search','seconds':.2,
                    'foundation_counts':{'streams':3,'physical_streams':3,'calls':0}}] if count==2 else [])
            directory=cache/'C/jena'/f'{method}_s{seed}'
            put(directory/'RESULT.json',value)
            put(directory/'SELECTION_SEAL.json',{'selected_prefix':6})
            (directory/'model.pt').write_bytes(b'do not deserialize or modify')
            put(report/'C/jena'/f'{method}_s{seed}.json',value)
    return report,cache


def test_counts_epoch_zero_and_honest_pending_workflow(campaign):
    finalize=module()
    report,cache=campaign
    before={p:p.read_bytes() for p in (cache/'C').rglob('*') if p.is_file()}
    result=finalize.build_status()
    assert result['new_neural_configurations']==22
    assert result['configured_training_stages']==48
    assert result['completed_training_stages']==48
    assert result['epoch_zero_refit_stages']==2
    assert result['overall_status']=='PENDING'
    assert '미완료 없음' not in (report/'RESUME_QUEUE.md').read_text(encoding='utf8')
    assert all(p.read_bytes()==value for p,value in before.items())
    public=common.read(report/'C/jena/SEQ_BLOCKS_s92601.json')
    assert public['foundation_counts_all_work']['calls'] is None
    assert public['foundation_counts_all_work']['calls_raw_value']==0
    assert public['stage1_caches'][0]['foundation_counts']['calls'] is None
    original=common.read(report/'attempts/counter_metadata_001/C/jena/SEQ_BLOCKS_s92601.json')
    assert original['foundation_counts_all_work']['calls']==0


def test_counter_rewrite_and_backup_are_idempotent(campaign):
    finalize=module()
    report,cache=campaign
    finalize.build_status()
    original=(report/'attempts/counter_metadata_001/C/jena/SEQ_BLOCKS_s92601.json').read_bytes()
    public=(report/'C/jena/SEQ_BLOCKS_s92601.json').read_bytes()
    result=finalize.build_status(refresh_ledger=False)
    assert (report/'C/jena/SEQ_BLOCKS_s92601.json').read_bytes()==public
    assert (report/'attempts/counter_metadata_001/C/jena/SEQ_BLOCKS_s92601.json').read_bytes()==original
    assert result['counter_interpretation']['status']=='COMPLETE'


def test_missing_refit_remains_pending_and_is_not_counted_complete(campaign):
    finalize=module()
    report,cache=campaign
    path=cache/'runs/jena/RESIDUAL_F_s92601/refit/RESULT.json'
    path.unlink()
    result=finalize.build_status()
    assert result['completed_training_stages']==47
    assert result['overall_status']=='PENDING'
    text=(report/'RESUME_QUEUE.md').read_text(encoding='utf8')
    assert 'finish-training' in text and 'report --verify-replay' in text


def test_ledger_preserves_upper_bound_scope_and_no_f_training_memory(campaign,monkeypatch):
    finalize=module()
    report,cache=campaign
    monkeypatch.setattr(common,'ledger',lambda:dict(GPU_owned_process_seconds=70,sessions=[dict(
        session_id='smoke_recovery',start_unix=1,end_unix=63,gpu_owned=True,
        time_accounting='Conservative upper bound: actual exit was not observed')]))
    finalize.build_status()
    ledger=common.read(report/'RESOURCE_LEDGER.json')
    assert ledger['GPU_conservative_upper_bound_component_seconds']==62
    assert 'not entirely exact' in ledger['GPU_time_precision']
    text=(report/'TRAINING_COST.csv').read_text(encoding='utf8')
    assert 'Shared foundation remains resident' in text
    assert 'Standalone no-F training peak is NA' in text


def test_complete_requires_actual_report_cost_replay_and_validator_evidence(campaign):
    finalize=module()
    report,cache=campaign
    put(report/'EXPERIMENT_PLAN.json',{})
    for dataset,count in (('jena',41),('pems32',23)):
        for index in range(count):
            directory=cache/'final'/f'{dataset}_{index}'
            directory.mkdir(parents=True)
            (directory/'V.npz').write_bytes(b'never opened by helper')
            put(directory/'V.binding.json',{})
        put(cache/f'FINAL_MANIFEST_{dataset}.json',[])
    put(cache/'FINAL_MANIFEST.json',[])
    (report/'RUNS.csv').write_text('run_id\n',encoding='utf8')
    put(report/'COST_REQUESTS.json',dict(expected_count=1,requests=[dict(cost_id='cost1')]))
    (report/'COST.csv').write_text('cost_id,parity_status,fresh_process\ncost1,PASS,True\n',encoding='utf8')
    (report/'COST_TRIALS.csv').write_text('cost_id,phase\n'+''.join('cost1,warmup\n' for _ in range(10))+''.join('cost1,timed\n' for _ in range(20)),encoding='utf8')
    for name in ('TRAINABLE_STATE_VALIDATION','CUDA_RESUME_VALIDATION','NUMERICAL_VALIDATION','REPLAY_VALIDATION'):
        put(report/(name+'.json'),dict(status='PASS'))
    for name in ('README_KO.md','RESULT_ANALYSIS_KO.md','CLAIM_STATUS.md','REPRODUCE.md','FIGURE_POINTS.csv','figure.png'):
        (report/name).write_bytes(b'synthetic')
    put(report/'FIGURE_MANIFEST.json',dict(status='COMPLETE',figures=[dict(file='figure.png')]))
    assert finalize.build_status()['overall_status']=='PENDING'
    assert '미완료 없음' not in (report/'RESUME_QUEUE.md').read_text(encoding='utf8')
    put(report/'VALIDATION_REPORT.json',dict(status='PASS'))
    assert finalize.build_status(refresh_ledger=False)['overall_status']=='COMPLETE'
    assert '미완료 없음' in (report/'RESUME_QUEUE.md').read_text(encoding='utf8')
