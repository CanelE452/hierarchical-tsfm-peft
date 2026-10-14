"""SOURCE-only CPU fixtures; root GO required before actual test execution."""
from copy import deepcopy
import inspect
import json

import numpy as np
import pytest

from research.smallk_failure_followup_20261014 import profile_stage_native_cost_v1 as profile


def producers():
    out=[]
    for d in profile.DATASETS:
        for s in profile.SEEDS:
            families={}
            for family in profile.FAMILIES:
                grid=[{'alpha':a,'tune_actual_Y_mse':1. if i==2 else 2.,'model_sha256':family+str(i),
                    'coefficient_artifact':{'path':family+str(i),'sha256':family+str(i)}} for i,a in enumerate(profile.ALPHAS)]
                families[family]={'status':'COMPLETE','diagnostic_only':False,'alpha_grid':grid,'selected_index':2,
                    'selected_alpha':profile.ALPHAS[2],'selected_model_sha256':family+'2',
                    'selected_coefficient_artifact':grid[2]['coefficient_artifact'],
                    'selected_model_metadata':{'fixed_n1_model_sha256':'N12' if family=='N3' else None}}
            out.append({'dataset':d,'seed':s,'status':'COMPLETE','old_E_D_F_updated':False,'F_loaded':False,
                'E_forward_replayed':False,'codec_before':{'old':'fixed'},'codec_after':{'old':'fixed'},
                'codec_terminal_hash':{'old':'fixed'},'families':families,
                'N3_fixed_N1_contract':{'N1_coefficients_reselected_for_N3':False}})
    return out


def test_schedule_has9_routes_at_every_position_once_and_excludes_oracles():
    schedule=profile.balanced_schedule()
    assert len(schedule)==9 and len(profile.ROUTES)==9 and profile.ORDER_SEED==92725
    assert schedule==profile.balanced_schedule()
    for position in range(9):assert {order[position] for order in schedule}==set(profile.ROUTES)
    assert all('O0' not in route and 'O1' not in route for route in profile.ROUTES)


def test_complete4_gate_rejects_reselected_N1_failed_family_and_fake_selection():
    values=producers();assert len(profile.validate_producers(values))==4
    values[0]['families']['N2']['status']='FAILED_PRESERVED'
    with pytest.raises(ValueError,match='DEPLOYMENT_FAMILY'):profile.validate_producers(values)
    values=producers();values[0]['N3_fixed_N1_contract']['N1_coefficients_reselected_for_N3']=True
    with pytest.raises(ValueError,match='SAME_SELECTED_N1'):profile.validate_producers(values)
    values=producers();values[0]['families']['N1']['selected_index']=6
    with pytest.raises(ValueError,match='DEPLOYMENT_FAMILY'):profile.validate_producers(values)


def test_audit_requires_pass_and_all4_actual_producer_sha_bindings():
    values=producers();records=[{'sha256':str(i)} for i in range(4)]
    audit={'status':'PASS_WITHOUT_F_REPLAY','failed_count':0,'conditions':[
        {'dataset':p['dataset'],'seed':p['seed'],'producer_sha256':r['sha256']} for p,r in zip(values,records)]}
    profile.audit_binding(audit,values,records)
    audit['conditions'][0]['producer_sha256']='wrong'
    with pytest.raises(ValueError,match='SHA_BINDING'):profile.audit_binding(audit,values,records)


class Tensor:
    def __init__(self,value):self.value=np.asarray(value,np.float32)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value
    def __getitem__(self,index):return Tensor(self.value[index])


def test_native_hybrid_uses_E_CPU_recipe_and_ignores_F_argument():
    calls=[]
    class Codec:
        def encode(self,history):calls.append('E');return history
    def predict(family,model,past,last,decoder,n1_model=None):
        calls.append('CPU_'+family);assert n1_model is models['N1']
        return np.repeat((last+np.float32(1.))[:,None],48,axis=1)
    def to_device(value):calls.append('normalized_H2D');return Tensor(value)
    class PoisonF:
        def __call__(self,*args):raise AssertionError('native must not call F')
    models={'N1':object(),'N2':object(),'N3':object(),'decoder':object()}
    route=profile.HybridNativeRoute(Codec(),models,'N2',predict,to_device)
    history=Tensor(np.zeros((2,512,3),np.float32));result=route.forward_path(history,PoisonF(),return_delta=False)
    assert result.value.shape==(2,48,3) and calls==['E','CPU_N2','normalized_H2D']
    with pytest.raises(ValueError,match='DELTA_PATH_UNSUPPORTED'):route.forward_path(history,PoisonF(),return_delta=True)
    with pytest.raises(ValueError,match='ORACLE_EXCLUDED'):profile.HybridNativeRoute(Codec(),models,'O1',predict,to_device)


class Backend:
    def synchronize(self):pass
    def reset_memory(self):pass
    def to_numpy(self,value):return np.array(value,copy=True)
    def memory_snapshot(self):return {'fixture_only':True}


class Pipeline:
    def __init__(self,backend):self.backend=backend
    def whole(self,physical,observed,preprocess):
        return np.ones((len(physical),48,2),np.float32),np.ones((len(physical),48,2),np.float32)*3


class Parity:
    @staticmethod
    def parity(actual,reference,atol,rtol):return {'pass':bool(np.allclose(actual,reference,atol=atol,rtol=rtol))}


def fixture_group(tmp_path,fast=lambda:None):
    backend=Backend();pipelines={r:Pipeline(backend) for r in profile.ROUTES}
    physical=np.zeros((1,512,2),np.float32);mask=np.ones_like(physical,bool)
    refs={r:{'normalized':np.ones((1,48,2),np.float32),'physical':np.ones((1,48,2),np.float32)*3} for r in profile.ROUTES}
    bulk={r:{'normalized':refs[r]['normalized']+1,'physical':refs[r]['physical']+1} for r in profile.ROUTES}
    saved={}
    def save(path,arrays):saved[str(path)]=deepcopy(arrays);return {'path':str(path),'sha256':'synthetic_only'}
    rows=profile.measure_group(pipelines,physical,mask,refs,bulk,tmp_path,fast,lambda:None,save,Parity)
    return rows,saved


def test_measurement_saves_all9_outputs_positions_and_keeps_bulk_failures_separate(tmp_path):
    rows,saved=fixture_group(tmp_path)
    assert len(rows)==9 and all(r['status']=='COMPLETE' and r['repeats']==9 for r in rows)
    for row in rows:
        arrays=saved[row['timed_outputs']['path']]
        assert arrays['normalized'].shape==(9,1,48,2) and set(arrays['position'])==set(range(9))
        assert all(c['matched_batch_normalized']['pass'] and not c['original_bulk_normalized']['pass'] for c in row['parity'])
        assert row['normalized_D2H_copy_outside_timer'] is True
    assert all(r['paired_RAW_median_ratio'] is not None for r in rows)


def test_interrupted_measurement_preserves_partial_rows_and_never_marks_complete(tmp_path):
    calls=0
    def budget():
        nonlocal calls
        calls+=1
        if calls>45:raise RuntimeError('synthetic budget')
    with pytest.raises(RuntimeError,match='synthetic budget'):fixture_group(tmp_path,budget)
    recorded=json.loads((tmp_path/'ROWS.json').read_text(encoding='utf-8'))
    assert recorded['rows'] and all(r['status']!='COMPLETE' for r in recorded['rows'])


def test_source_default_no_actual_assets_and_fresh_stage_runtime_only():
    source=inspect.getsource(profile)
    assert 'import stage_separation_runtime_v1 as runtime' in source
    assert 'import continuation_runtime' not in source and 'Limit(' not in source.replace('ContinuationLimit(','')
    assert profile.LOCAL_SECONDS==180 and profile.REPEATS==9
    assert '--describe' in source and '--no-contention-attested' in source
