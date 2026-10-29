"""CPU synthetic invariants for the three-dataset branch extension."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from research.p_branch_appl_ett_20261029_v1 import models, runtime, run_one

BROAD = runtime.REPO / 'research/informs_broad_validation_20261015_v1'
sys.path.insert(0, str(BROAD))
import networks

torch.set_num_threads(2)


def job(dataset='ETTh1', method='P_TIME_ONLY', seed=92601):
    k = runtime.KS[dataset]
    return dict(job_id=f'PBA_{method}_{dataset}_K{k}_s{seed}', method_id=method,
                dataset=dataset, K=k, seed=seed, extended=False)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def test_original_models_bytes_and_numerical_functions_unchanged():
    old = runtime.REPO/'research/jena_p_branch_20261027_v1'
    new = Path(run_one.__file__).parent
    assert (old/'models.py').read_bytes() == (new/'models.py').read_bytes()
    def functions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf8')).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name in ['build_spec', 'choose_epoch', 'check_selection', 'predict_saved', 'run_stages']:
        assert functions(old/'run_one.py')[name] == functions(new/'run_one.py')[name]


@pytest.mark.parametrize('dataset', runtime.DATASETS)
@pytest.mark.parametrize('method', models.METHODS)
def test_branch_equation_initialization_parameters_and_input_gradient(dataset, method):
    channels, k = runtime.V_SHAPES[dataset][2], runtime.KS[dataset]
    torch.manual_seed(92601)
    full = networks.ForecastAdapter('B2', 512, 48, channels, k,
                                    np.zeros(channels, np.float32), np.eye(channels, k, dtype=np.float32))
    assert sum(p.numel() for p in full.parameters() if p.requires_grad) == runtime.EXPECTED_PARAMETERS[dataset]['OURS']
    e, d = full.E.detach().clone(), full.D.detach().clone()
    old = full.front.mixer
    rng = torch.get_rng_state().clone()
    retained = models.SingleBranchMixer(old, method)
    assert torch.equal(torch.get_rng_state(), rng)
    full.front.mixer = retained
    assert sum(p.numel() for p in full.parameters() if p.requires_grad) == runtime.EXPECTED_PARAMETERS[dataset][method]
    assert torch.equal(full.E, e) and torch.equal(full.D, d)
    x = torch.randn(2, 512, channels, requires_grad=True)
    assert torch.equal(full.front(x), x)
    branch = retained.temporal if method == 'P_TIME_ONLY' else retained.feature
    with torch.no_grad():
        branch.output.weight.fill_(.003)
        branch.output.bias.fill_(.01)
    if method == 'P_TIME_ONLY':
        expected = x + old.temporal(old.temporal_norm(x).transpose(1, 2)).transpose(1, 2)
        assert retained.temporal is old.temporal
    else:
        expected = x + old.feature(old.feature_norm(x))
        assert retained.feature is old.feature
    assert torch.equal(retained(x), expected) and not torch.equal(expected, x)
    class FrozenProxy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.1), requires_grad=False)
        def forward(self, z):
            return self.scale * z[:, -48:]
    foundation = FrozenProxy()
    full(x, foundation).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert foundation.scale.grad is None


@pytest.mark.parametrize('dataset', runtime.DATASETS)
@pytest.mark.parametrize('method', models.METHODS)
def test_branch_factory_reuses_full_seed_initialization(dataset, method):
    channels, k = runtime.V_SHAPES[dataset][2], runtime.KS[dataset]
    created=[]
    def factory(spec, task, endpoint, foundation):
        torch.manual_seed(int(spec['seed']))
        model=networks.ForecastAdapter('B2', 512, 48, channels, k,
            np.zeros(channels, np.float32), np.eye(channels, k, dtype=np.float32))
        created.append(dict(model=model,mixer=model.front.mixer,rng=torch.get_rng_state().clone()))
        return model, {'fit_end': endpoint}
    spec = dict(method_id=method, seed=92601, family='B2')
    reference, _ = factory(spec, None, 123, None)
    reference.front.mixer = models.SingleBranchMixer(reference.front.mixer, method)
    fit = SimpleNamespace(create_model=factory, predict_model=object())
    with models.patch_core(fit):
        actual, norm = fit.create_model(spec, None, 123, None)
        assert norm['fit_end'] == 123
        assert actual is created[-1]['model']
        retained=('temporal_norm','temporal') if method=='P_TIME_ONLY' else ('feature_norm','feature')
        assert all(getattr(actual.front.mixer,key) is getattr(created[-1]['mixer'],key) for key in retained)
        assert torch.equal(torch.get_rng_state(),created[-1]['rng'])
        assert actual.state_dict().keys()==reference.state_dict().keys()
        assert all(torch.equal(value, reference.state_dict()[name]) for name, value in actual.state_dict().items())
    assert fit.create_model is factory


@pytest.mark.parametrize('dataset',runtime.DATASETS)
@pytest.mark.parametrize('method',models.METHODS)
def test_production_factory_preserves_full_retained_initial_tensors(dataset,method):
    import fit_base
    channels=runtime.V_SHAPES[dataset][2]
    physical=np.random.default_rng(123).normal(size=(64,channels)).astype(np.float32)
    task=dict(physical=physical,observed=np.ones_like(physical,bool),L=512,H=48)
    spec=dict(run_one.build_spec(job(dataset,method)),device='cpu')
    foundation=object()
    full,reference_norm=fit_base.create_model(spec,task,64,foundation)
    full_state={name:value.detach().clone() for name,value in full.state_dict().items()}
    expected_rng=torch.get_rng_state().clone()
    removed=('feature_norm','feature') if method=='P_TIME_ONLY' else ('temporal_norm','temporal')
    expected={name:value for name,value in full_state.items()
              if not any(name.startswith('front.mixer.'+prefix+'.') for prefix in removed)}
    original_factory=fit_base.create_model
    with models.patch_core(fit_base):
        branch,actual_norm=fit_base.create_model(spec,task,64,foundation)
        assert branch.state_dict().keys()==expected.keys()
        assert all(torch.equal(value,expected[name]) for name,value in branch.state_dict().items())
        assert torch.equal(torch.get_rng_state(),expected_rng)
        for key in ['mean','std','observed_counts','basis','pca_center']:
            assert np.array_equal(actual_norm[key],reference_norm[key])
    assert fit_base.create_model is original_factory


def test_exact_sixty_axes_and_recipe():
    values = [job(ds, method, seed) for seed in range(92601,92611) for ds in runtime.DATASETS for method in models.METHODS]
    assert len(values) == len({r['job_id'] for r in values}) == 60
    for value in values:
        assert run_one.validate_job(value) is value
        spec = run_one.build_spec(value)
        assert spec['epochs'] == [.25,1.,3.] and spec['lr'] == .001
        assert spec['effective_batch'] == 16 and spec['microbatch'] == 4 and spec['F_row_microbatch'] == 128
        assert spec['core_dtype'] == 'float32' and spec['device'] == 'cuda' and spec['revin'] is False


@pytest.mark.parametrize('changes', [dict(dataset='jena'),dict(K=3),dict(seed=92600),dict(seed=True),
    dict(method_id='A1'),dict(extended=True),dict(extended=0),dict(job_id='bad')])
def test_wrong_axes_and_extensions_rejected(changes):
    with pytest.raises(ValueError):run_one.validate_job(dict(job(), **changes))


def test_complete_astop_curve_and_earliest_tie():
    search=dict(curves=[dict(epoch=e,astop_mse=m) for e,m in [(0,2),(.25,1),(1,1),(3,3)]],
                summary=dict(status='SEGMENT_COMPLETE',selected_epoch=.25,selected_astop_mse=1))
    assert run_one.choose_epoch(search,[.25,1.,3.]) == (.25,1.)
    search['summary']['selected_epoch']=1.
    with pytest.raises(ValueError):run_one.choose_epoch(search,[.25,1.,3.])
    search['summary']['status']='BUDGET_LIMIT'
    with pytest.raises(RuntimeError):run_one.choose_epoch(search,[.25,1.,3.])


@pytest.mark.parametrize('dataset', runtime.DATASETS)
def test_prefix_header_and_protected_T_proof(dataset, tmp_path, monkeypatch):
    monkeypatch.setitem(runtime.T_STARTS,dataset,4)
    monkeypatch.setitem(runtime.V_SHAPES,dataset,(1,48,3))
    prefix=tmp_path/'prefix.npz';config=tmp_path/'config.json';config.write_text('{}')
    np.savez(prefix,physical=np.zeros((4,3),np.float32),observed=np.ones((4,3),bool))
    proof=runtime.bounded_prefix_headers(prefix,4,3)
    access=dict(dataset=dataset,prefix_path=str(prefix),config_path=str(config),prefix_sha256=runtime.sha(prefix),
                config_sha256=runtime.sha(config),prefix_rows=4,T_start=4,prefix_header_proof=proof,T_numeric_values_parsed=0)
    runtime.assert_protected_T(dict(job=job(dataset),task_access=access))
    for key,value in [('T_numeric_values_parsed',1),('prefix_rows',5),('T_start',5),('prefix_sha256','wrong')]:
        with pytest.raises(ValueError):runtime.assert_protected_T(dict(job=job(dataset),task_access=dict(access,**{key:value})))
    np.savez(prefix,physical=np.zeros((5,3),np.float32),observed=np.ones((5,3),bool))
    with pytest.raises(ValueError):runtime.bounded_prefix_headers(prefix,4,3)


def test_original_jena_T_proof_is_retained():
    proof=dict(new_T_numeric_values_parsed=0,members=[dict(member=k,tail_numeric_values_parsed=0)
                for k in ['x.npy','finite.npy','times.npy','mean.npy','std.npy','columns.npy']])
    runtime.assert_protected_T(dict(task_access=proof))
    proof['members'][0]['tail_numeric_values_parsed']=1
    with pytest.raises(ValueError):runtime.assert_protected_T(dict(task_access=proof))


def registration_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime,'REPORT',tmp_path)
    sources={'research/p_branch_appl_ett_20261029_v1/models.py':runtime.sha(Path(models.__file__))}
    monkeypatch.setattr(run_one,'source_hashes',lambda:sources)
    names=['PREREGISTRATION.json','PREREGISTRATION.md','RUN_PLAN.json','PRESERVATION_BASELINE.json','REFERENCE_RUNS.json','DATA_REGISTRY.json']
    for n in names:write(tmp_path/n, {})
    write(tmp_path/'PREREGISTRATION.json',dict(registered_sources=sources,registration_documents=names))
    write(tmp_path/'RUN_PLAN.json',dict(jobs=[job()],extension_jobs=[]))
    write(tmp_path/'PRESERVATION_BASELINE.json',dict(files=[]))
    write(tmp_path/'REGISTRATION_SEAL.json',dict(created_utc='2026-01-01T00:00:00+00:00',files={n:runtime.sha(tmp_path/n) for n in names}))
    return sources


def test_actual_registration_roster_and_hashes(tmp_path, monkeypatch):
    registration_fixture(tmp_path,monkeypatch)
    assert run_one.validate_registration(job())['files'].keys() == runtime.read(tmp_path/'REGISTRATION_SEAL.json')['files'].keys()
    (tmp_path/'DATA_REGISTRY.json').write_text('{"tampered":true}')
    with pytest.raises(ValueError):run_one.validate_registration(job())


def test_recovery_attempts_require_exact_prior_sealed_authorization(tmp_path, monkeypatch):
    sources=registration_fixture(tmp_path,monkeypatch)
    assert run_one.allowed_attempts(job()) == [1,2]
    assert run_one.validate_attempt(job(),1) == {'kind':'BASE','attempt':1}
    with pytest.raises(ValueError):run_one.validate_attempt(job(),3)
    auth=dict(status='APPROVED_TECHNICAL_RECOVERY',job_id=job()['job_id'],allowed_attempts=[3,4],
              registration_seal_sha256=runtime.sha(tmp_path/'REGISTRATION_SEAL.json'),scientific_source_hashes=sources,
              created_utc='2026-01-02T00:00:00+00:00')
    path=tmp_path/'RECOVERY_AUTHORIZATION_001.json';write(path,auth)
    assert run_one.allowed_attempts(job()) == [1,2,3,4]
    binding=run_one.validate_attempt(job(),3,started_unix=1767398400.)
    assert binding['authorization_sha256'] == runtime.sha(path) and binding['kind']=='TECHNICAL_RECOVERY'
    with pytest.raises(ValueError):run_one.validate_attempt(job(),3,started_unix=1767225600.)
    for changes in [dict(scientific_source_hashes={}),dict(allowed_attempts=[3,5]),dict(status='unchecked')]:
        write(path,dict(auth,**changes))
        with pytest.raises(ValueError):run_one.validate_attempt(job(),3)


@pytest.mark.parametrize('exc,phase,kind',[
    (ValueError('Historical pre-T data, registered bounds or L/H changed'),'DATA','SCIENTIFIC'),
    (AssertionError('Frozen backbone weights changed'),'FINAL_VALIDATION','SCIENTIFIC'),
    (PermissionError('Cannot change protected existing file: x'),'TRAINING','SCIENTIFIC'),
    (ValueError('Independent saved-array FP64 score mismatch'),'FINAL_VALIDATION','SCIENTIFIC'),
    (ValueError('unknown parser bug'),'IMPORTS','UNKNOWN'),
    (RuntimeError('CUDA out of memory'),'TRAINING','TECHNICAL'),
    (TimeoutError('watchdog'),'TRAINING','TECHNICAL'),
    (ModuleNotFoundError('missing module'),'IMPORTS','TECHNICAL'),
    (ModuleNotFoundError('missing module'),'DATA','UNKNOWN'),
    (RuntimeError('device-side assert triggered'),'TRAINING','UNKNOWN'),
])
def test_failure_classification_uses_evidence_not_exception_type(exc,phase,kind):
    assert run_one.classify_failure(exc,phase)==kind


def test_worker_reservations_retained_until_process_exit(monkeypatch):
    events=[]
    class Lock:
        def __init__(self,path):self.path=path
        def __enter__(self):events.append(('enter',self.path))
        def __exit__(self,*args):events.append(('exit',self.path))
    monkeypatch.setattr(runtime,'CampaignLock',Lock)
    monkeypatch.setattr(run_one,'WORKER_RESERVATIONS',[])
    run_one.retain_worker_reservations()
    assert events == [('enter',runtime.CACHE/'worker.lock'),('enter',runtime.SHARED_CACHE/'worker.lock')]
    assert len(run_one.WORKER_RESERVATIONS)==2


def test_preservation_hash_cache_rechecks_metadata_and_explicit_full(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime,'REPORT',tmp_path/'reports')
    monkeypatch.setattr(runtime,'_PRESERVATION_HASH_CACHE',{})
    old=tmp_path/'protected.txt';old.write_bytes(b'old')
    digest=runtime.sha(old)
    write(runtime.REPORT/'PRESERVATION_BASELINE.json',dict(files=[dict(path=str(old),sha256=digest)]))
    real_sha=runtime.sha;reads=[]
    def counting_sha(path):
        reads.append(Path(path))
        return real_sha(path)
    monkeypatch.setattr(runtime,'sha',counting_sha)
    first=runtime.guard_preserved()
    second=runtime.guard_preserved()
    assert first['scan_mode']=='FULL_SHA256' and first['full_hash_checked']==1
    assert second['cached_checked']==1 and second['full_hash_checked']==0
    assert reads==[old]
    forced=runtime.guard_preserved(full=True)
    assert forced['requested_full'] and forced['full_hash_checked']==1 and forced['cached_checked']==0
    old.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Existing protected file changed'):
        runtime.guard_preserved()
    assert reads==[old,old,old]


def test_preservation_critical_inputs_always_freshly_hashed(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime,'REPORT',tmp_path/'reports')
    monkeypatch.setattr(runtime,'_PRESERVATION_HASH_CACHE',{})
    critical=tmp_path/'reference.npz';critical.write_bytes(b'reference')
    write(runtime.REPORT/'PRESERVATION_BASELINE.json',dict(files=[dict(path=str(critical),sha256=runtime.sha(critical))]))
    write(runtime.REPORT/'REFERENCE_RUNS.json',dict(rows=[dict(prediction_path=str(critical))]))
    runtime.guard_preserved()
    second=runtime.guard_preserved()
    assert second['full_hash_checked']==1 and second['cached_checked']==0


@pytest.mark.parametrize('dataset',runtime.DATASETS)
def test_selection_is_sealed_before_fresh_refit_and_V(dataset,tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'REPORT',tmp_path/'reports')
    monkeypatch.setattr(runtime,'CACHE',tmp_path/'cache')
    public,cache=runtime.REPORT/'attempt',runtime.CACHE/'attempt'
    public.mkdir(parents=True);cache.mkdir(parents=True)
    events=[];selected_job=job(dataset);spec=run_one.build_spec(selected_job)
    toy=torch.nn.Linear(2,2)
    norm=dict(mean=np.zeros(2,np.float32),std=np.ones(2,np.float32))
    fit_end,refit_end=runtime.FIT_ENDS[dataset]
    v_start,v_end=runtime.V_RANGES[dataset]
    bounds=dict(A_fit=(0,fit_end),A_stop=(fit_end,refit_end),M_cal=(refit_end,v_start),
                V_select=(v_start,v_end),T_final=(v_end,v_end+100))
    task=dict(physical=np.zeros((v_end,2),np.float32),split_ranges=bounds,L=512,H=48,
              columns=['a','b'],array_sha256={},data_sha256='data',dataset=dataset,_campaign_access={})
    def stage(specification,data,endpoint,stop,epochs,*unused):
        if stop is not None:
            events.append(('search',endpoint,stop))
            result=dict(curves=[dict(epoch=e,astop_mse=m) for e,m in [(0.,2.),(.25,1.),(1.,1.),(3.,3.)]],
                        summary=dict(status='SEGMENT_COMPLETE',selected_epoch=.25,selected_astop_mse=1.))
        else:
            assert runtime.read(public/'SELECTION.json')['V_used'] is False and epochs==[.25]
            events.append(('refit',endpoint,stop))
            result=dict(summary=dict(status='SEGMENT_COMPLETE'))
        result['summary'].update(seconds=1.,exposures=1,updates=1)
        result.update(model=toy.state_dict(),normalization=norm)
        return result
    def predict(model,normalization,data,foundation,specification,path,modules,**kwargs):
        seal=runtime.read(public/'SELECTION.json')
        assert seal['V_used'] is False and seal['selected_epoch']==.25
        events.append(('V',bounds['V_select']))
        path.write_bytes(b'fake V arrays')
        return dict(MSE=1.,MAE=1.,prediction_path=str(path),prediction_sha256=runtime.sha(path))
    fake_fit=SimpleNamespace(_source_contract=lambda:('core',{'fit.py':'hash'}),_stage=stage,
        create_model=lambda *a:(toy,norm),_torch_new=lambda path,value:path.write_bytes(b'fake bundle'))
    fake_training=SimpleNamespace(fingerprint=lambda value:'fingerprint',cpu_state=lambda model:model.state_dict())
    monkeypatch.setattr(run_one,'predict_saved',predict)
    result=run_one.run_stages(selected_job,task,SimpleNamespace(initial_hash='f'),
        dict(fit=fake_fit,training=fake_training),spec,cache,public,1000.)
    assert events==[('search',fit_end,bounds['A_stop']),('refit',refit_end,None),('V',bounds['V_select'])]
    assert result['selection_sealed_before_V']


def test_no_cuda_initialized_by_CPU_suite():
    assert torch.cuda.is_initialized() is False
