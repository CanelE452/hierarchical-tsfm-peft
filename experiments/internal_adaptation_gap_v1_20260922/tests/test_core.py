from __future__ import annotations
import copy,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from gap_screen.metrics import *
from gap_screen.io import *
from gap_screen.model import *
from gap_screen.data import prepare_frame,save_panel,read_panel,Panel


def example():
    rng=np.random.default_rng(4);return rng.normal(size=(3,2,3,4)),rng.normal(size=(3,2,4)),np.array([2.,5.]),np.array([.1,.5,.9])

def test_metric_scalar_agreement():
    q,y,s,t=example();assert np.isclose(score(q,y,s,t)[0]['primary'],scalar_primary(q,y,s,t),rtol=1e-13)

def test_metric_scale_equivariance():
    q,y,s,t=example();assert np.isclose(score(q*7,y*7,s*7,t)[0]['primary'],score(q,y,s,t)[0]['primary'])

def test_crossing_raw_but_scoring_sorted():
    q,y,s,t=example();a=score(q,y,s,t);b=score(np.sort(q,axis=2),y,s,t)
    assert a[0]['crossing']>0 and b[0]['crossing']==0 and a[0]['primary']==b[0]['primary']

def test_bad_metric_shape():
    q,y,s,t=example()
    with pytest.raises(ValueError):score(q,y[:,:,:2],s,t)

def test_nan_metric_refused():
    q,y,s,t=example();q[0,0,0,0]=np.nan
    with pytest.raises(ValueError):score(q,y,s,t)

def test_native_loss_gradient_and_mask():
    q=torch.zeros(2,3,4,requires_grad=True);y=torch.tensor([[1.,float('nan'),2.,3.],[0.,0.,1.,1.]])
    loss=normalized_loss(q,y,torch.zeros(2,1),torch.ones(2,1),torch.tensor([.1,.5,.9]),True)
    loss.backward();assert torch.isfinite(loss) and torch.isfinite(q.grad).all() and (q.grad[0,:,1]==0).all()

def test_inverse_transform_round_trip():
    x=torch.randn(2,3,4);loc=torch.randn(2,1);sc=torch.rand(2,1)+.1
    raw=raw_inverse(x,loc,sc,True);torch.testing.assert_close(((raw-loc[:,None])/sc[:,None]).asinh(),x)

def test_checkpoint_ties_earlier():
    rows=[dict(step=2,val_primary=1.),dict(step=0,val_primary=1.)];assert choose_checkpoint(rows)['step']==0

def test_budget_cannot_extrapolate():
    rows=[dict(step=0,val_primary=2,cold_ready_s=3),dict(step=2,val_primary=1,cold_ready_s=10)]
    assert best_under_budget(rows,2) is None and best_under_budget(rows,9)['step']==0

def test_time_target_is_interval_not_exact_event():
    rows=[dict(step=0,val_primary=3,cold_ready_s=2),dict(step=8,val_primary=1,cold_ready_s=12)]
    z=first_target(rows,2);assert z['step']==8 and z['lower_step']==0 and not z['interpolated']

def test_time_target_no_headroom():
    assert first_target([dict(step=0,val_primary=1,cold_ready_s=2)],2)['status']=='NO_ADAPTATION_HEADROOM'

def test_time_target_censored():
    assert first_target([dict(step=0,val_primary=3,cold_ready_s=2)],2)['status']=='CENSORED'

def test_bootstrap_pairing_and_sign():
    b=np.ones((2,20,3));z=paired_bootstrap(b*.9,b,100,7,42)
    assert abs(z['gain_pct']-10)<1e-12 and np.allclose(z['ci95'],10)

def test_calendar_blocks_with_missing_dates():
    b=np.ones((2,6,3));r=paired_bootstrap(b,b,20,7,1,labels=np.array([0,0,2,2,4,4]));assert r['blocks']==3

def test_optimizer_ratio(fake_peft,tiny_base,cfg):
    m=Predictor(tiny_base,'LORAPLUS',cfg,8);opt=build_optimizer(m,'LORAPLUS',cfg)
    assert opt.param_groups[1]['lr']/opt.param_groups[0]['lr']==16
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==156

def test_lora_plus_same_initial_function(fake_peft,cfg):
    from conftest import TinyBase
    seed_all(10);b=TinyBase();a=Predictor(copy.deepcopy(b),'LORA',cfg,22);z=Predictor(copy.deepcopy(b),'LORAPLUS',cfg,22)
    assert all(torch.equal(trainable_state(a)[k],v) for k,v in trainable_state(z).items())
    x=torch.randn(2,16);g=torch.zeros(2,dtype=torch.long)
    torch.testing.assert_close(a(x,g,4)[1],z(x,g,4)[1],rtol=0,atol=0)

def test_head_zero_output_parity(tiny_base,cfg):
    m=Predictor(tiny_base,'F0',cfg,1);x=torch.randn(2,16);g=torch.zeros(2,dtype=torch.long);f=m.encode(x,g,4)
    for kind in ['mlp','full']:
        h=CachedHead(tiny_base,kind,8,4);torch.testing.assert_close(h(*f,4)[1],m(x,g,4)[1],rtol=cfg['prediction_rtol'],atol=cfg['prediction_atol'])

def test_head_backprop_without_backbone(tiny_base,cfg):
    m=Predictor(tiny_base,'F0',cfg,1);x=torch.randn(2,16);g=torch.zeros(2,dtype=torch.long)
    with torch.no_grad():f=m.encode(x,g,4)
    h=CachedHead(tiny_base,'mlp',8,4);h(*f,4)[1].sum().backward()
    assert h.head[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in tiny_base.parameters())

def test_group_isolation(tiny_base,cfg):
    m=Predictor(tiny_base,'F0',cfg,1);x=torch.randn(4,16);g=torch.tensor([0,0,1,1]);a=m(x,g,4)[1];x[2:]+=99;b=m(x,g,4)[1]
    torch.testing.assert_close(a[:2],b[:2],rtol=0,atol=0)

def test_nonreentrant_checkpoint_preserves_gradient(fake_peft,tiny_base,cfg):
    a=Predictor(copy.deepcopy(tiny_base),'LORA',cfg,3);cc=copy.deepcopy(cfg);cc['gradient_checkpointing']=True
    b=Predictor(copy.deepcopy(tiny_base),'LORA',cc,3);x=torch.randn(2,16);g=torch.zeros(2,dtype=torch.long)
    for m in [a,b]:m(x,g,4)[1].sum().backward()
    for (n,p),(n2,q) in zip(a.named_parameters(),b.named_parameters()):
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad)

def test_ledger_rejects_duplicate(tmp_path):
    l=Ledger(tmp_path/'l.jsonl',2);l.intent('a',1);l.commit('a',1)
    with pytest.raises(Blocked):l.intent('a',1)

def test_ledger_detects_interrupted_update(tmp_path):
    l=Ledger(tmp_path/'l.jsonl',2);l.intent('a',1)
    with pytest.raises(Blocked):Ledger(tmp_path/'l.jsonl',2)

def test_ledger_cap(tmp_path):
    l=Ledger(tmp_path/'l.jsonl',1);l.intent('a',1);l.commit('a',1)
    with pytest.raises(Blocked):l.intent('a',2)

def test_json_numpy_nonfinite(tmp_path):
    p=tmp_path/'r.json';write_json(p,{'x':np.bool_(True),'y':np.array([np.nan,1.])});assert read_json(p)=={'x':True,'y':[None,1.0]}

def frame_fixture():
    n=100*96;t=pd.date_range('2016-01-01',periods=n,freq='15min');d={'date':t.strftime('%Y-%m-%d %H:%M:%S')}
    for i in range(7):d[str(i)]=np.sin(np.arange(n)/30+i)+i
    return pd.DataFrame(d)

def test_temporal_split_and_scale_train_only(cfg):
    f=frame_fixture();a=prepare_frame(f,'ettm2',cfg,{});f.iloc[-2000:,1:]+=1e5;b=prepare_frame(f,'ettm2',cfg,{})
    np.testing.assert_array_equal(a.sigma,b.sigma)
    for role,oo in a.origins.items():
        low,high=a.metadata['bounds'][role];assert (oo>=low).all() and (oo+a.horizon<=high).all()

def test_frequency_is_not_resampled(cfg):
    f=frame_fixture().drop(index=100)
    with pytest.raises(Blocked,match='IRREGULAR'):prepare_frame(f,'ettm2',cfg,{})

def test_duplicates_not_silently_deduped(cfg):
    f=frame_fixture();f=pd.concat([f,f.iloc[[30]]])
    with pytest.raises(Blocked,match='DUPLICATE'):prepare_frame(f,'ettm2',cfg,{})

def test_data_save_restore(tmp_path,cfg):
    a=prepare_frame(frame_fixture(),'ettm2',cfg,{});save_panel(a,tmp_path/'p.npz');b=read_panel(tmp_path/'p.npz')
    np.testing.assert_array_equal(a.values,b.values);assert a.columns==b.columns

def test_source_seal_detects_edit(tmp_path):
    p=tmp_path/'a.py';p.write_text('a=1');seal={'code':code_manifest(tmp_path)};verify_code(tmp_path,seal)
    p.write_text('a=2')
    with pytest.raises(Blocked):verify_code(tmp_path,seal)

def test_secret_detector():
    import importlib.util
    spec=importlib.util.spec_from_file_location('publisher',Path(__file__).parents[1]/'publish.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.SECRET.search('ghp_'+'a'*30)
    assert not module.SECRET.search('No tokens or credentials are published.')
