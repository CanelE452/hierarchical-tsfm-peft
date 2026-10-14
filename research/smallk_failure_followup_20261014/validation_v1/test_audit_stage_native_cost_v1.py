"""SOURCE-only saved arithmetic fixtures; no real data/models or timing replay."""
from copy import deepcopy
import inspect

import numpy as np
import pytest

from research.smallk_failure_followup_20261014.validation_v1 import audit_stage_native_cost_v1 as audit


def test_float64_tolerance_matches_fixed_absolute_plus_relative_rule():
    reference=np.array([0.,1000.],np.float32)
    assert audit.close_arrays(reference+np.array([1e-5,1e-3],np.float32),reference,'normalized')
    assert not audit.close_arrays(reference+np.array([2e-5,0.],np.float32),reference,'normalized')
    assert not audit.close_arrays(np.array([np.nan,1.]),reference,'normalized')


def test_channel_macro_metrics_handle_unequal_counts_missing_nan_and_empty_channel():
    target=np.zeros((1,2,3),np.float32)
    prediction=np.array([[[2.,3.,np.nan],[2.,np.nan,np.nan]]],np.float32)
    observed=np.array([[[True,True,False],[True,False,False]]])
    score=audit.masked_score(prediction,target,observed)
    assert score['mse']==pytest.approx(6.5) and score['mae']==pytest.approx(2.5)
    assert score['active_channels']==2 and score['per_channel_counts']==[2,1,0]
    physical=audit.masked_score(prediction,target,observed,np.array([2.,1.,5.]))
    assert physical['mse']==pytest.approx(12.5) and physical['mae']==pytest.approx(3.5)
    empty=audit.masked_score(prediction,target,np.zeros_like(observed))
    assert empty['mse'] is None and empty['active_channels']==0


def test_independent_schedule_and_paired_ratio_detect_wrong_block_pairing():
    schedule=audit.expected_schedule();route='N2_s92601'
    raw={'block':list(range(9)),'seconds':np.arange(1,10).tolist()}
    times=np.arange(1,10,dtype=float)*2
    row={'status':'COMPLETE','warmup_each_route':2,'repeats':9,'route':route,'batch':8,
        'seconds':times.tolist(),'block':list(range(9)),'position':[order.index(route) for order in schedule],
        'median_seconds':10.,'median_seconds_per_window':1.25,'paired_RAW_seconds':raw['seconds'],
        'paired_RAW_ratios':[2.]*9,'paired_RAW_median_ratio':2.}
    failures=[]
    audit.timing_checks(row,raw,lambda name,passed:failures.append(name) if not passed else None,'fixture')
    assert not failures
    row['paired_RAW_seconds']=raw['seconds'][::-1]
    audit.timing_checks(row,raw,lambda name,passed:failures.append(name) if not passed else None,'fixture')
    assert any('paired_raw' in name for name in failures)


def test_coefficient_hashes_change_when_actual_weight_changes():
    feature={'center':np.zeros(33,np.float64),'scale':np.ones(33,np.float64)}
    coefficient={'coefficients':np.zeros((33,48,1),np.float64),'intercept':np.zeros((48,1),np.float64),
        'observed_counts':np.ones((48,1),np.int64),'channel_counts':np.array([48],np.int64)}
    solution={'kind':'N1_LATENT','alpha':.1,'feature_state_sha256':'fit-feature'}
    producer={'feature_state':{'latent_channels':1,'fit_rows':4,'fit_raw_features_sha256':'raw-sha'},
        'families':{'N1':{'selected_model_metadata':{'solution':solution}}}}
    initial=audit.coefficient_hashes(producer,'N1',feature,coefficient)
    changed=deepcopy(coefficient);changed['coefficients'][0,0,0]=.001
    after=audit.coefficient_hashes(producer,'N1',feature,changed)
    assert initial[0]==after[0] and initial[1]!=after[1] and initial[2]!=after[2]


class Tensor:
    def __init__(self,value):self.value=np.asarray(value,np.float32)
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return self.value


def test_original_and_Bi_CP_hash_algorithms_keep_their_distinct_order_contracts():
    first={'a':Tensor([1.]),'b':Tensor([2.])};reverse={'b':first['b'],'a':first['a']}
    one,two=audit.tensor_state_hashes(first),audit.tensor_state_hashes(reverse)
    assert one['old_algorithm']!=two['old_algorithm'] and one['bi_algorithm']==two['bi_algorithm']


def test_audit_default_source_requires_execute_and_declares_no_forward_or_timing_replay():
    source=inspect.getsource(audit)
    assert '--execute' in source and 'stage_separation_runtime_v1' in source
    assert 'import engine' not in source and 'FrozenBolt(' not in source
    assert len(audit.ROUTES)==9 and audit.REPEATS==9
