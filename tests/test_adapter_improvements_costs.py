import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def impl():
    try:
        return importlib.import_module('research.adapter_improvements_20261018_v1.costs')
    except ModuleNotFoundError:
        pytest.fail('Whole-path cost implementation is missing')


def test_latency_uses_timed_samples_batch_denominator_and_separate_stage_scope():
    rows = [dict(phase='timed', whole_seconds=x, batch_size=8) for x in range(1, 21)]
    rows += [dict(phase='warmup', whole_seconds=1000, batch_size=8),
             dict(phase='stage_probe', whole_seconds=2000, batch_size=8)]
    value = impl().summarize_trials(rows)
    assert value['median_batch_ms'] == 10500
    assert value['median_ms_per_origin'] == 1312.5
    assert value['IQR_batch_ms'] == 9500
    assert value['p95_batch_ms'] == pytest.approx(19050)
    assert value['timed_trials'] == 20
    assert value['stage_sum_equals_whole_claimed'] is False


class Route:
    def __init__(self, task, drift=0.):
        self.task, self.device, self.family, self.run_id = task, 'cpu', 'P_SPLIT', 'synthetic'
        self.mean, self.std = np.array([2., 3.]), np.array([2., 4.])
        self.drift, self.closed, self.calls = drift, False, []
        self.spec = {}
        self.model = None
        self.pool = type('Pool', (), dict(foundation=None, instances_created=0, load_seconds=0.))()

    def predict_physical(self, physical, observed, measure_stages=False):
        self.calls.append(len(physical))
        filled = physical.copy()
        for b in range(len(filled)):
            previous = self.mean.copy()
            for t in range(filled.shape[1]):
                filled[b, t] = np.where(observed[b, t], filled[b, t], previous)
                previous = filled[b, t]
        y = np.repeat(filled[:, -1:], self.task['H'], axis=1).astype(np.float64) + self.drift
        self.last_receipt = dict(whole_seconds=.001, F_logical_streams=len(physical)*2,
                                F_physical_streams=len(physical)*2, F_calls=1, stage_seconds={})
        return y

    def activate(self):
        pass

    def deactivate(self):
        pass

    def memory_state(self):
        return dict(resident_weights_cpu_MiB=0.)

    def close(self):
        self.closed = True


def fixture():
    campaign = Path(__file__).resolve().parents[1]/'research/informs_broad_validation_20261015_v1'
    if str(campaign) not in sys.path:
        sys.path.insert(0, str(campaign))
    x = np.arange(80, dtype=np.float32).reshape(40, 2)
    mask = np.ones_like(x, dtype=bool)
    mask[18, 0] = False
    task = dict(physical=x, observed=mask, L=4, H=2, columns=['a', 'b'])
    origins = np.array([16, 18, 20, 22, 24, 26, 28, 30, 32])
    route = Route(task)
    physical = np.stack([x[o-4:o] for o in origins])
    observed = np.stack([mask[o-4:o] for o in origins])
    expected = (route.predict_physical(physical, observed)-route.mean)/route.std
    saved = dict(pred_normalized=expected, origins=origins, channel_ids=np.array(task['columns']))
    return task, saved


def test_full_saved_parity_checks_remainder_and_channel_order():
    task, saved = fixture()
    route = Route(task)
    check = impl().parity_saved(route, task, saved, batch=8)
    assert check['status'] == 'PASS'
    assert check['n_origins'] == 9 and check['last_batch_size'] == 1
    assert route.calls == [8, 1]
    corrupt = dict(saved, channel_ids=np.array(['b', 'a']))
    with pytest.raises(ValueError, match='channel'):
        impl().parity_saved(route, task, corrupt, batch=8)


def test_parity_mismatch_invalidates_route_and_closes_both_loads():
    task, saved = fixture()
    routes = []
    def loader():
        route = Route(task, drift=0 if not routes else 1)
        routes.append(route)
        return route
    candidate = dict(run_id='synthetic', method='P_SPLIT', prediction_binding={'sha256':'a'*64})
    result = impl().measure_candidate(candidate, task, saved, loader, batch=8, device='cpu')
    assert result['row']['cost_valid'] is False
    assert result['row']['status'] == 'FAILED_PARITY'
    assert len(routes) == 2 and all(route.closed for route in routes)
    assert result['trials'] == []


def test_candidate_uses_reloaded_route_for_timing_and_full_protocol():
    task, saved = fixture()
    routes = []
    def loader():
        route = Route(task)
        routes.append(route)
        return route
    candidate = dict(run_id='synthetic', method='P_SPLIT', prediction_binding={'sha256':'a'*64})
    result = impl().measure_candidate(candidate, task, saved, loader, batch=8, device='cpu')
    row = result['row']
    assert row['cost_valid'] and row['status'] == 'COMPLETE'
    assert row['warmups'] == 10 and row['timed_trials'] == 20
    assert row['precision_comparison'] == 'METHOD_LEVEL'
    assert row['memory_scope'] == 'cost phase; training excluded; process RSS sampled at 5ms'
    assert len([t for t in result['trials'] if t['phase']=='timed']) == 20
    assert len(routes[0].calls) == 3
    assert len(routes[1].calls) == 34
    assert all(route.closed for route in routes)


def test_invalid_and_incomplete_accuracy_candidates_are_not_measured():
    rows = [dict(run_id='good', status='COMPLETE', evaluation='V_select', prediction_binding={'sha256':'x'}),
            dict(run_id='partial', status='PARTIAL_CHECKPOINTED', evaluation='V_select'),
            dict(run_id='invalid', status='COMPLETE', evaluation='V_select', independent_FP64='FAIL')]
    accepted, skipped = impl().eligible_candidates(rows)
    assert [r['run_id'] for r in accepted] == ['good']
    assert len(skipped) == 2 and all(not r['cost_valid'] for r in skipped)


def test_actual_f_rows_reject_computing_maximum_k_then_slicing():
    task, saved = fixture()
    route = Route(task)
    route.latent_k = 1
    with pytest.raises(ValueError, match='F.*rows'):
        impl().parity_saved(route, task, saved, batch=8)


def test_frozen_backbone_cannot_silently_cast_to_fp64():
    import torch
    task, _ = fixture()
    route = Route(task)
    route.pool.foundation = torch.nn.Linear(2,2).double()
    with pytest.raises(ValueError, match='FP32'):
        impl()._dtypes(route, {'head_dtype':'float64'})


def test_real_physical_route_preserves_fill_normalize_f_inverse_and_cpu_return(tmp_path):
    import torch
    from torch import nn
    fixture()  # Put the original campaign modules on their documented import path.
    deployment = importlib.import_module('deployment')
    data = importlib.import_module('data_contract')
    task, _ = fixture()
    task['split_ranges'] = {'A_stop':(10,14),'V_select':(16,36)}
    origins = np.arange(16,34,2,dtype=np.int64)
    mean,std,_ = data.fit_statistics(task,14)
    physical,observed = impl()._windows(task,origins)
    filled = np.stack([data.causal_forward_fill(x,m,mean) for x,m in zip(physical,observed)])
    normalized = ((filled-mean)/std).astype(np.float32)
    expected = np.repeat(normalized[:,-1:]+np.float32(.125),2,axis=1)
    class Foundation(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(.125),requires_grad=False)
            self.initial_hash,self.horizon,self.row_microbatch = 'synthetic-exact',2,128
            self.streams = self.physical_streams = self.physical_calls = self.row_calls = 0
        def forward(self,x):
            self.streams += x.shape[0]*x.shape[-1]
            self.physical_streams += x.shape[0]*x.shape[-1]
            self.physical_calls += 1
            return (x[:,-1:]+self.weight).repeat(1,2,1)
    def loader():
        pool = deployment.FoundationPool(None,'cpu',factory=lambda p,h,m,d: Foundation())
        route = deployment.Route(dict(family='RAW',foundation_sha='synthetic-exact',run_id='raw'),
                                 task,tmp_path,device='cpu',foundation_pool=pool)
        route.owns_pool = True
        return route
    candidate = dict(run_id='raw',method='RAW',prediction_dtype='float32',prediction_binding={'sha256':'x'})
    saved = dict(pred_normalized=expected,origins=origins,channel_ids=np.array(task['columns']))
    result = impl().measure_candidate(candidate,task,saved,loader,batch=8,device='cpu')
    assert result['row']['cost_valid']
    assert result['row']['backbone_dtype']=='float32'
    assert result['row']['F_rows_per_timed_batch']==[16]
    assert result['row']['serialized_reload_saved_V_parity']['n_origins']==9


def test_parent_owner_lease_is_accepted_without_gpu_child_environment():
    task,saved = fixture()
    class Owner:
        checks = 0
        def check(self):
            self.checks += 1
    owner = Owner()
    candidate = dict(run_id='synthetic',method='P_SPLIT',prediction_binding={'sha256':'x'})
    result = impl().measure_candidate(candidate,task,saved,lambda:Route(task),batch=1,device='cuda',lease=owner)
    assert result['row']['cost_valid']
    assert owner.checks == 31


def _public_accuracy(tmp_path, row):
    from research.adapter_improvements_20261018_v1.contracts import atomic_json,binding,sha
    root = tmp_path/'report'
    path = root/'jobs/conditional.json'
    value = dict(status='COMPLETE',rows=[row])
    atomic_json(path,value)
    atomic_json(root/'receipts/conditional.json',dict(status='COMPLETE',result=value,
        artifact_hashes={'jobs/conditional.json':sha(path)}))
    return dict(report=root,cache=tmp_path,previous_cache=tmp_path,asset=tmp_path,repo=tmp_path)


def test_conditional_pems_heads_use_explicit_parent_and_never_jena_fallback(tmp_path):
    parent = tmp_path/'pems_parent.pt'
    parent.write_bytes(b'exact-parent')
    from research.adapter_improvements_20261018_v1.contracts import binding
    ctx = dict(cache=tmp_path)
    expected = binding(parent,ctx)
    row = dict(method='D_LINEAR',arm='D_LINEAR',dataset='pems32',seed=92601,
               state_binding={'reference':'cache://head.json','sha256':'a'*64},
               prediction_binding={'reference':'cache://p.npz','sha256':'b'*64},parent_bundle_binding=expected)
    ctx = _public_accuracy(tmp_path,row)
    accepted,skipped = impl().collect_candidates(dict(cohort='pems32',seed=92601),ctx)
    assert not skipped and len(accepted)==1
    assert accepted[0]['bundle_binding']==expected


def test_calibrator_candidates_keep_separate_bindings_and_plain_array_hashes(tmp_path):
    row = dict(method='B2_LIFT',dataset='jena',seed=92601,
               prediction_binding={'reference':'cache://p.npz','sha256':'a'*64},
               base_checkpoint={'reference':'cache://base.pt','sha256':'b'*64},
               refiner_bundle={'reference':'cache://cal.pt','sha256':'c'*64},
               array_fingerprints={'origins':'d'*64})
    ctx = _public_accuracy(tmp_path,row)
    accepted,skipped = impl().collect_candidates(dict(cohort='jena',seed=92601),ctx)
    assert not skipped and len(accepted)==1
    assert 'bundle_binding' not in accepted[0]
    assert accepted[0]['base_checkpoint']==row['base_checkpoint']
    assert accepted[0]['refiner_bundle']==row['refiner_bundle']
    assert accepted[0]['origins_sha256']=='d'*64


def test_post_decoder_calibrator_sees_normalized_full_context_and_one_f_pass(tmp_path):
    import torch
    from torch import nn
    fixture()
    deployment = importlib.import_module('deployment')
    data = importlib.import_module('data_contract')
    task,_ = fixture()
    task['split_ranges']={'A_stop':(10,14),'V_select':(16,36)}
    mean,std,_ = data.fit_statistics(task,14)
    origins = np.arange(16,34,2,dtype=np.int64)
    physical,observed = impl()._windows(task,origins)
    filled=np.stack([data.causal_forward_fill(x,m,mean) for x,m in zip(physical,observed)])
    x=((filled-mean)/std).astype(np.float32)
    base=np.repeat(x[:,-1:]+np.float32(.125),2,axis=1)
    expected=(base+.25*x.mean(axis=1,keepdims=True)).astype(np.float32)
    class Foundation(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=nn.Parameter(torch.tensor(.125),requires_grad=False)
            self.horizon,self.initial_hash,self.row_microbatch=2,'synthetic-exact',128
            self.streams=self.physical_streams=self.physical_calls=self.row_calls=0
        def forward(self,x):
            self.streams+=x.shape[0]*x.shape[-1]
            self.physical_streams+=x.shape[0]*x.shape[-1]
            self.physical_calls+=1
            return (x[:,-1:]+self.weight).repeat(1,2,1)
    class Calibrator(nn.Module):
        def forward(self,context,final_prediction):
            assert context.shape[-1]==2 and context.dtype==torch.float32
            return final_prediction+.25*context.mean(1,keepdim=True)
    def loader():
        pool=deployment.FoundationPool(None,'cpu',factory=lambda p,h,m,d:Foundation())
        route=deployment.Route(dict(family='RAW',foundation_sha='synthetic-exact',run_id='cal'),task,tmp_path,
                               device='cpu',foundation_pool=pool)
        route.owns_pool=True
        route.model=nn.Module()
        return impl().install_calibrator(route,Calibrator())
    saved=dict(pred_normalized=expected,origins=origins,channel_ids=np.array(task['columns']))
    result=impl().measure_candidate(dict(run_id='cal',method='B2_LIFT',prediction_binding={'sha256':'x'}),
                                    task,saved,loader,batch=8,device='cpu')
    assert result['row']['cost_valid']
    assert result['row']['F_rows_per_timed_batch']==[16]
    assert any('stage_refiner_seconds' in row for row in result['trials'])


def test_complex_calibrator_parameter_inventory_keeps_elements_dof_and_bytes_separate():
    import torch
    from torch import nn
    task,_=fixture()
    route=Route(task)
    route.model=nn.Module()
    route.model.register_parameter('base',nn.Parameter(torch.ones(3)))
    calibrator=nn.Module()
    calibrator.register_parameter('weights',nn.Parameter(torch.ones(2,dtype=torch.complex64)))
    route.model.add_module('post_decoder_calibrator',calibrator)
    value=impl()._dtypes(route,{'method':'B2_LIFT','core_dtype':'float32'})
    assert value['adapter_params']==5
    assert value['adapter_real_degrees_of_freedom']==7
    assert value['refiner_parameter_elements']==2
    assert value['refiner_real_degrees_of_freedom']==4
    assert value['refiner_parameter_bytes']==16


def test_exact_gain_loader_uses_original_b2_bundle_and_accepts_declared_archive_extras(tmp_path):
    import torch
    from torch import nn
    fixture()
    fit=importlib.import_module('fit_base')
    data=importlib.import_module('data_contract')
    training=importlib.import_module('training')
    deployment=importlib.import_module('deployment')
    from research.adapter_improvements_20261018_v1.contracts import atomic_json,binding
    x=np.random.default_rng(12).normal(size=(80,2)).astype(np.float32)
    task=dict(physical=x,observed=np.ones_like(x,dtype=bool),L=40,H=2,columns=['a','b'],
              data_sha256=data.array_sha256(x),split_ranges={'A_stop':(45,50),'V_select':(60,78)})
    class Foundation(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=nn.Parameter(torch.tensor(.125),requires_grad=False)
            self.horizon,self.initial_hash,self.row_microbatch=2,'synthetic-exact',128
            self.streams=self.physical_streams=self.physical_calls=self.row_calls=0
        def forward(self,x):
            self.streams+=x.shape[0]*x.shape[-1]
            self.physical_streams+=x.shape[0]*x.shape[-1]
            self.physical_calls+=1
            return (x[:,-1:]+self.weight).repeat(1,2,1)
    f=Foundation()
    spec=dict(family='B2',latent_k=1,seed=92601,device='cpu',run_id='original_B2')
    model,norm=fit.create_model(spec,task,50,f)
    state=model.state_dict()
    bundle=dict(spec=spec,model=state,normalization=norm,
                model_state_sha=training.fingerprint({k:data.array_sha256(v.numpy()) for k,v in state.items()}),
                contract=dict(data_sha=task['data_sha256'],model_sha=f.initial_hash,
                              scaler_sha=fit.scaler_fingerprint(norm['mean'],norm['std'])))
    base_path=tmp_path/'base.pt'
    torch.save(bundle,base_path)
    gain_path=tmp_path/'gain.json'
    atomic_json(gain_path,dict(H=2,C=2,rms=[1.,1.],theta=[[2.,0.],[2.,0.]]))
    ctx=dict(repo=Path(__file__).resolve().parents[1],cache=tmp_path,previous_cache=tmp_path,asset=tmp_path,
             foundation_loader=lambda c,m,h,d:Foundation())
    from research.adapter_improvements_20261018_v1 import lift_jobs
    atomic_json(gain_path,dict(H=2,C=2,rms=[1.,1.],theta=[[2.,0.],[2.,0.]],
                              scientific_source_hashes=lift_jobs.scientific_sources(ctx)))
    candidate=dict(run_id='gain',method='B2_OUTPUT_GAIN',base_checkpoint=binding(base_path,ctx),
                   refiner_bundle=binding(gain_path,ctx),head_dtype='float64')
    modules=dict(deployment=deployment,fit=fit)
    route=impl()._route_loader(candidate,{'device':'cpu'},ctx,modules,task)()
    try:
        origins=np.array([60,62],dtype=np.int64)
        physical,observed=impl()._windows(task,origins)
        expected_x=((physical-norm['mean'])/norm['std']).astype(np.float32)
        with torch.no_grad():
            base=model(torch.as_tensor(expected_x),f).numpy()
        expected=expected_x[:,-1:].astype(np.float64)+2*(base.astype(np.float64)-expected_x[:,-1:].astype(np.float64))
        predicted=(route.predict_physical(physical,observed)-norm['mean'])/norm['std']
        np.testing.assert_allclose(predicted,expected,rtol=1e-6,atol=1e-6)
        assert route.last_receipt['F_logical_streams']==2
        assert route.model.D.dtype==torch.float32
        assert route.model.post_decoder_calibrator.theta.dtype==torch.float64
        archive=tmp_path/'predictions.npz'
        np.savez(archive,pred_normalized=expected,origins=origins,channel_ids=np.array(['a','b']),
                 base_pred_normalized=base,anchor=expected_x[:,-1:])
        saved=impl()._load_prediction({'prediction_binding':binding(archive,ctx)},ctx)
        assert set(saved)=={'pred_normalized','origins','channel_ids'}
    finally:
        route.close()


def test_unexecuted_conditional_methods_remain_explicit_unmeasured_rows(tmp_path):
    ctx=dict(report=tmp_path,plan={'jobs':[dict(job_id='optional_lift',kind='lift_jobs',cohort='jena',seed=92601)]})
    accepted,skipped=impl().collect_candidates(dict(cohort='jena',seed=92601),ctx)
    assert not accepted
    assert {r['method'] for r in skipped}=={'B2_LIFT','B2_OUTPUT_GAIN'}
    assert all(r['cost_valid'] is False and 'no completed scored prediction' in r['reason'] for r in skipped)
