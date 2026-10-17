from contextlib import nullcontext
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


def impl():
    return importlib.import_module('research.adapter_improvements_20261018_v1.decoder_transfer')


def synthetic(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.E = nn.Parameter(torch.eye(3)[:, :2])
            self.length, self.horizon, self.latent_k = 40, 4, 2
        def encode(self, x):
            return x@self.E
    class Frozen:
        initial_hash = 'synthetic-F'
        streams = row_calls = 0
        def __call__(self, z):
            assert not torch.is_grad_enabled()
            self.streams += z.shape[0]*z.shape[-1]
            self.row_calls += 1
            return z[:, -4:] + 1
        def assert_frozen_unchanged(self):
            pass
    class Windows:
        def __init__(self, task, mean, std, device):
            self.values = torch.tensor(task['physical'], dtype=torch.float32)
        def get(self, origins):
            return (torch.stack([self.values[o-40:o] for o in origins]),
                    torch.stack([self.values[o:o+4] for o in origins]),
                    torch.ones((len(origins), 4, 3), dtype=torch.bool))
    task = dict(L=40, H=4, columns=['a', 'b', 'c'], physical=np.arange(70*3).reshape(70, 3).astype(np.float32),
                observed=np.ones((70, 3), bool), data_sha256='synthetic-data')
    modules = dict(training=SimpleNamespace(FastWindows=Windows),
                   fit=SimpleNamespace(scaler_fingerprint=lambda a,b:'synthetic-scaler'),
                   campaign=tmp_path/'science')
    ctx = dict(cache=tmp_path/'cache', repo=tmp_path, deadline=10**12)
    return ctx, modules, task, Model(), dict(mean=np.zeros(3), std=np.ones(3)), Frozen()


def test_generic_extract_exact_lag_major_single_F_pass_and_byte_verified_reuse(tmp_path):
    m = impl()
    ctx, modules, task, model, norm, f = synthetic(tmp_path)
    checkpoint = tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'synthetic checkpoint')
    from research.adapter_improvements_20261018_v1.contracts import binding
    origins = np.array([40, 44, 48], np.int64)
    first, receipt = m.extract_cache(ctx, modules, task, model, norm, origins, f,
                                     binding(checkpoint, ctx), 'synthetic', lambda: None, batch_size=2)
    assert receipt['new_F_rows'] == len(origins)*2
    assert receipt['new_F_calls'] == 2 and f.row_calls == 2
    x, _, _ = modules['training'].FastWindows(task, norm['mean'], norm['std'], 'cpu').get(origins)
    z, u = model.encode(x), model.encode(x)[:, -4:]+1
    joined = torch.cat((z, u), 1)
    expected = torch.cat([joined[:,40-lag:44-lag]-z[:,-1-lag,None] for lag in (0,1,2,4,8,16,32)], -1)
    np.testing.assert_array_equal(first['phi'], expected.detach().numpy())
    again, reused = m.extract_cache(ctx, modules, task, model, norm, origins, f,
                                     binding(checkpoint, ctx), 'synthetic', lambda: None, batch_size=2)
    assert reused['new_F_calls'] == reused['new_F_rows'] == 0
    assert f.row_calls == 2
    np.testing.assert_array_equal(again['phi'], first['phi'])
    cache_path = Path(ctx['cache'])/receipt['artifact']['reference'].split('://', 1)[1]
    cache_path.write_bytes(cache_path.read_bytes()+b'tamper')
    with pytest.raises(ValueError, match='bytes'):
        m.extract_cache(ctx, modules, task, model, norm, origins, f,
                        binding(checkpoint, ctx), 'synthetic', lambda: None, batch_size=2)


def test_profile_admission_includes_all_stages_grid_and_verification_reserve():
    m = impl()
    result = m.admission_estimate([dict(n_origins=8, seconds=1.), dict(n_origins=8, seconds=2.)],
                                   dict(A_fit_fit=100, A_fit_A_stop=10, A60_fit=120, A60_V_select=10), 1000.)
    assert result['estimated_extraction_seconds'] >= 240*2/8
    assert result['estimated_fitting_seconds'] > result['estimated_extraction_seconds']
    assert result['estimated_complete_seconds'] >= result['estimated_fitting_seconds']/0.8
    assert result['admitted']
    assert not m.admission_estimate([dict(n_origins=1, seconds=10.)], dict(A_fit_fit=100), 60.)['admitted']
    assert not m.admission_estimate([dict(n_origins=8,seconds=1.)],dict(A_fit_fit=100),400.)['admitted']


def test_new_complete_parent_is_verified_and_selected_A_fit_checkpoint_is_bound(tmp_path):
    m = impl()
    from research.adapter_improvements_20261018_v1.contracts import atomic_json, binding, sha
    ctx = dict(repo=tmp_path, report=tmp_path/'report', cache=tmp_path/'cache')
    parent_id = 'B2_PCA_PEMS32_s92601'
    checkpoint, bundle = tmp_path/'A_fit.pt', tmp_path/'A60.pt'
    checkpoint.write_bytes(b'fit'); bundle.write_bytes(b'refit')
    parent = dict(status='COMPLETE', bundle_binding=binding(bundle, ctx),
                  fit_receipt=dict(selected_epoch=1.), common_endpoint_rows=[dict(epoch=1., evaluation='A_stop',
                  checkpoint_binding=binding(checkpoint, ctx))])
    path = ctx['report']/'jobs'/(parent_id+'.json')
    atomic_json(path, parent)
    parent_job = dict(job_id=parent_id, scientific_identity='synthetic-parent')
    ctx['plan'] = dict(jobs=[parent_job])
    atomic_json(ctx['report']/'receipts'/(parent_id+'.json'), dict(status='COMPLETE',
                scientific_identity='synthetic-parent', artifact_hashes={'jobs/'+parent_id+'.json':sha(path)}, result=parent))
    selected = m.select_parent({}, ctx)
    assert selected['kind'] == 'FRESH_CUDA_PCA'
    assert selected['A_fit_checkpoint'] == checkpoint
    assert selected['A60_bundle'] == bundle
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        m.select_parent({}, ctx)


@pytest.mark.parametrize('job', [{}, {'fallback_A_fit_checkpoint_binding': None}])
def test_fallback_requires_registered_A_fit_byte_binding(tmp_path, job):
    with pytest.raises(FileNotFoundError, match='fallback_A_fit_checkpoint_binding'):
        impl().select_parent(job, dict(repo=tmp_path, report=tmp_path/'report', cache=tmp_path/'cache'))


def test_complete_synthetic_transfer_seals_before_V_and_saves_all_four_heads(tmp_path, monkeypatch):
    import sys
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research/informs_broad_validation_20261015_v1'))
    from networks import ForecastAdapter
    from training import FastWindows
    from data_contract import origins
    from research.adapter_improvements_20261018_v1.contracts import binding, read, resolve_binding
    m = impl()
    rng = np.random.default_rng(23)
    task = dict(L=512,H=12,physical=rng.normal(size=(613,32)).astype(np.float32),
                observed=np.ones((613,32),bool),columns=[str(i) for i in range(32)],data_sha256='synthetic',
                split_ranges=dict(A_fit=(0,529),A_stop=(529,553),M_cal=(553,565),V_select=(565,589),T_final=(589,613)))
    ctx = dict(repo=tmp_path,cache=tmp_path/'cache',report=tmp_path/'report',deadline=time.time()+600)
    checkpoint, bundle = tmp_path/'fit.pt', tmp_path/'refit.pt'
    checkpoint.write_bytes(b'synthetic-fit');bundle.write_bytes(b'synthetic-refit')
    parent = dict(kind='FRESH_CUDA_PCA', parent_job_id='synthetic_parent',selected_epoch=1.,
                  A_fit_checkpoint_binding=binding(checkpoint,ctx),A60_bundle_binding=binding(bundle,ctx))
    modules = dict(data=SimpleNamespace(origins=origins),training=SimpleNamespace(FastWindows=FastWindows),
                   fit=SimpleNamespace(scaler_fingerprint=lambda a,b:'synthetic-scaler'),campaign=tmp_path/'science')
    class Frozen:
        initial_hash='synthetic-F'
        streams=row_calls=0
        def __call__(self,z):
            assert not torch.is_grad_enabled()
            self.streams += len(z)*z.shape[-1];self.row_calls+=1
            return z[:,-12:]+.2
        def assert_frozen_unchanged(self):
            pass
    f = Frozen()
    def model():
        return ForecastAdapter('B0',512,12,32,8,np.zeros(32),np.eye(32)[:,:8])
    norm = dict(mean=np.zeros(32,np.float32),std=np.ones(32,np.float32))
    monkeypatch.setattr(m,'_resources',lambda job,context:(modules,task,parent,f))
    monkeypatch.setattr(m,'_load_models',lambda *args:(model(),norm,model(),norm))
    original_extract = m.extract_cache
    def guarded_extract(*args,**kwargs):
        label=args[8]
        if label.endswith('_A60_V_select'):
            assert (ctx['cache']/'decoder_transfer'/'synthetic_transfer'/'SELECTION_SEAL.json').exists()
        return original_extract(*args,**kwargs)
    monkeypatch.setattr(m,'extract_cache',guarded_extract)
    result=m.execute_job(dict(job_id='synthetic_transfer',seed=92601,K=8,device='cpu',scientific_identity='synthetic'),ctx)
    assert result['status']=='COMPLETE'
    public=read(result['artifact_paths'][0])
    assert len(public['rows'])==4 and len(public['selection_trials'])==40
    assert public['new_F_rows']==8*(6+2+30+2)
    assert public['all_four_heads_transferred'] and public['choice_independent_of_Jena_V']
    assert len(public['rows'][0]['bands'])==3
    assert public['rows'][0]['bands'][2]['horizon_start']==9
    for row in public['rows']:
        assert resolve_binding(row['state_binding'],ctx).exists()
        with np.load(resolve_binding(row['prediction_binding'],ctx)) as saved:
            assert saved['pred_normalized'].dtype==np.float64
            assert saved['pred_normalized'].shape==(2,12,32)
    seal_path=ctx['cache']/'decoder_transfer'/'synthetic_transfer'/'SELECTION_SEAL.json'
    original_seal=seal_path.read_bytes()
    original_family=m.decoder_jobs.run_family
    def different_timing_family(fit,stop,refit,load_validation,seal,check):
        import copy
        def timed_seal(selection):
            revised=copy.deepcopy(selection)
            for trial in revised['selection_trials']:
                trial['seconds'] += 1000.
            seal(revised)
        replay=original_family(fit,stop,refit,load_validation,timed_seal,check)
        for trial in replay['selection_trials']:
            trial['seconds'] += 1000.
        return replay
    monkeypatch.setattr(m.decoder_jobs,'run_family',different_timing_family)
    repeated=m.execute_job(dict(job_id='synthetic_transfer',seed=92601,K=8,device='cpu',scientific_identity='synthetic'),ctx)
    replay_public=read(repeated['artifact_paths'][0])
    assert repeated['status']=='COMPLETE'
    assert seal_path.read_bytes()==original_seal
    assert replay_public['selections']==public['selections']
    assert replay_public['selection_trials'][0]['seconds'] > 1000.
    assert replay_public['new_F_calls']==replay_public['new_F_rows']==0
