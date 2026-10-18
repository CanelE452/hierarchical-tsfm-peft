import hashlib
import json
from pathlib import Path

import pytest

from research.b2_dataset_search_20261019_v1 import run


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _wave(repo):
    config, prefix = run.paths(repo, 'fixture')
    _write(config, dict(status='READY', dataset='fixture', C=7, prefix_only_approved=True))
    prefix.write_bytes(b'isolated pre-T fixture')
    entry = dict(dataset='fixture', K_grid=[1, 2, 4],
                 config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                 prefix_sha256=hashlib.sha256(prefix.read_bytes()).hexdigest())
    value = dict(status='REGISTERED_BEFORE_TRAINING', datasets=[entry])
    _write(repo / 'reports' / run.NAME / 'waves' / 'WAVE_001.json', value)
    return value, config, prefix


def test_wave_rejects_prefix_changed_after_registration(tmp_path):
    _value, _config, prefix = _wave(tmp_path)
    run.load_wave(tmp_path, 1)
    prefix.write_bytes(b'different prefix')
    with pytest.raises(ValueError, match='byte identity'):
        run.load_wave(tmp_path, 1)


def test_unready_or_duplicate_dataset_cannot_enter_wave(tmp_path):
    value, config, _prefix = _wave(tmp_path)
    _write(config, dict(status='NOT_READY', dataset='fixture', C=7, prefix_only_approved=True))
    with pytest.raises(ValueError, match='readiness'):
        run.load_wave(tmp_path, 1)
    value['datasets'].append(value['datasets'][0])
    _write(tmp_path / 'reports' / run.NAME / 'waves' / 'WAVE_001.json', value)
    with pytest.raises(ValueError, match='repeat'):
        run.load_wave(tmp_path, 1)


def test_cost_requests_cover_all_widths_seeds_and_both_policies(tmp_path):
    wave, _config, _prefix = _wave(tmp_path)
    rows = [dict(K=k, seed=seed, run_id=f'B2_fixture_K{k}_OFF_s{seed}')
            for k in (1, 2, 4) for seed in (92601, 92602)]
    target = tmp_path / '.cache' / run.NAME / 'datasets' / 'fixture' / 'VERIFIED_ROWS.json'
    _write(target, rows)
    requests = run.cost_requests(tmp_path, Path('model'), wave)
    assert len(requests) == 18
    assert len({(r['run_id'], r['row_policy']) for r in requests}) == 18
    assert {r['row_policy'] for r in requests} == set(run.POLICIES)
    assert sum(r['family'] == 'RAW' for r in requests) == 2
    assert {r['K'] for r in requests if r['family'] == 'RAW'} == {2}
    _write(target, rows[:-1])
    with pytest.raises(ValueError, match='Every registered K'):
        run.cost_requests(tmp_path, Path('model'), wave)


@pytest.mark.parametrize('name', ('../outside', 'a/b', 'a\\b', ''))
def test_dataset_name_cannot_escape_output_namespace(name):
    with pytest.raises(ValueError, match='Unsafe'):
        run.paths(Path.cwd(), name)


def _completed_direct(tmp_path):
    import numpy as np
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _modules
    from research.b2_dataset_extension_20261018_v1 import baselines
    from research.b2_revin_20261018_v1.contracts import binding
    task = dict(dataset='fixture', L=512, H=48, columns=[f'c{i}' for i in range(7)],
        physical=np.zeros((3200,7),np.float32), observed=np.ones((3200,7),bool),
        data_sha256='d'*64, array_sha256={}, split_ranges=dict(A_fit=(0,2000),
        A_stop=(2000,2400),M_cal=(2400,2800),V_select=(2800,3200),T_final=(3200,4000)))
    ctx = dict(repo=run.REPO, asset=tmp_path/'model', deadline=1e12,
        cache=tmp_path/'.cache'/run.NAME/'datasets'/'fixture',
        report=tmp_path/'reports'/run.NAME/'datasets'/'fixture')
    ctx['modules'] = _modules(ctx)
    fit, training = ctx['modules']['fit'], ctx['modules']['training']
    job = 'DirectMLP_fixture_s92601'
    spec = dict(run_id=job,family='DIRECT_G',dataset='fixture',seed=92601,device='cuda',
        fold='primary',epochs=[.25,1.,3.],lr=.001,effective_batch=16,microbatch=4,
        F_row_microbatch=128,checkpoint_inputs=False,core_dtype='float32')
    with baselines.direct_scope(ctx['modules'],task):
        source_sha, files = fit._source_contract()
    axes = dict(L=512,H=48,C=7,columns=task['columns'],fold='primary',source_array_fingerprints={})
    contract = dict(source_sha=source_sha,config_sha=training.fingerprint(dict(spec=spec,axes=axes)),
                    data_sha=task['data_sha256'],model_sha='NO_FOUNDATION')
    bundle = ctx['cache']/'runs'/job/'selected_bundle.pt'
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b'synthetic complete bundle guarded by SHA')
    receipt = dict(status='COMPLETE_VALID',run_id=job,family='DIRECT_G',seed=92601,
        epochs=[.25,1.,3.],selected_epoch=1.,source_sha=source_sha,
        artifact_cache_relative_path=bundle.relative_to(ctx['cache']).as_posix(),
        artifact_sha256=run.sha(bundle),reused_complete_artifact=False)
    _write(ctx['cache']/'runs'/job/'RUN_CONTRACT.json',contract)
    public = ctx['report']/'runs'/job
    _write(public/'SOURCE_CONTRACT.json',dict(**contract,files=files,spec=spec))
    _write(public/'BASE_FIT_RECEIPT.json',receipt)
    evidence = dict(receipt,dataset='fixture',C=7,L=512,H=48,bundle_binding=binding(bundle,ctx),
                    T_access=False,V_independent_test=False,F_instances=0)
    path=ctx['report']/(job+'_FIT.json')
    _write(path,evidence)
    return task,ctx,path,bundle


def test_completed_direct_reuse_preserves_original_evidence_bytes(tmp_path, monkeypatch):
    task,ctx,path,_bundle = _completed_direct(tmp_path)
    before=path.read_bytes()
    fit=ctx['modules']['fit']
    monkeypatch.setattr(fit,'_stage',lambda *args,**kwargs:pytest.fail('Completed guard must not train'))
    result=run.fit_direct_or_reuse(task,ctx,92601)
    assert result['reused_complete_artifact'] is True
    assert path.read_bytes()==before
    assert json.loads(before)['reused_complete_artifact'] is False


@pytest.mark.parametrize('changed', ('artifact','data','source','metadata'))
def test_completed_direct_reuse_rejects_changed_scientific_identity(tmp_path,monkeypatch,changed):
    task,ctx,path,bundle = _completed_direct(tmp_path)
    monkeypatch.setattr(ctx['modules']['fit'],'_stage',lambda *args,**kwargs:pytest.fail('Invalid resume must not train'))
    if changed=='artifact':bundle.write_bytes(b'changed bundle')
    elif changed=='data':task['data_sha256']='e'*64
    elif changed=='source':
        target=ctx['report']/'runs'/'DirectMLP_fixture_s92601'/'SOURCE_CONTRACT.json'
        value=json.loads(target.read_text());value['files']['fit_base.py']='0'*64;_write(target,value)
    else:
        value=json.loads(path.read_text());value['selected_epoch']=3.;_write(path,value)
    before=path.read_bytes()
    with pytest.raises(ValueError):run.fit_direct_or_reuse(task,ctx,92601)
    assert path.read_bytes()==before


def test_missing_direct_evidence_uses_original_fitter(tmp_path,monkeypatch):
    from research.b2_dataset_extension_20261018_v1 import baselines
    task=dict(dataset='fixture',columns=list(range(7)),L=512,H=48)
    ctx=dict(report=tmp_path/'reports',cache=tmp_path/'cache')
    expected=dict(status='COMPLETE_VALID')
    calls=[]
    monkeypatch.setattr(baselines,'fit_direct',lambda *args:calls.append(args) or expected)
    assert run.fit_direct_or_reuse(task,ctx,92601) is expected
    assert len(calls)==1
