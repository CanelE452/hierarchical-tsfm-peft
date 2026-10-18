import importlib

import numpy as np
import pytest


def observation():
    try:
        return importlib.import_module('research.gt_ts2_fastcheck_20261018_v1.observation')
    except ModuleNotFoundError:
        pytest.fail('mask-only matched history diagnostic is missing')


def test_complete_history_subset_uses_both_windows_and_no_future():
    o=observation()
    mask=np.ones((24,2),bool)
    mask[3,0]=False
    selected=o.complete_history_mask(mask,np.array([8,12]),4,8)
    np.testing.assert_array_equal(selected,[False,True])
    mask[12:]=False
    np.testing.assert_array_equal(o.complete_history_mask(mask,np.array([8,12]),4,8),selected)


def test_subset_rejects_illegal_history_boundaries():
    o=observation()
    with pytest.raises(ValueError):
        o.complete_history_mask(np.ones((8,2),bool),np.array([3]),4,8)


def test_no_complete_context_is_recorded_without_replacing_primary():
    o=observation()
    p=np.ones((2,48,2),np.float32)
    bundle=dict(prediction=p,target=p*2,mask=np.ones_like(p,bool),origins=np.array([8,12]),
                mean=np.zeros(2,np.float32),std=np.ones(2,np.float32))
    result=o.subset_scores(bundle,bundle,np.zeros(2,bool))
    assert result['status']=='NO_FULLY_OBSERVED_HISTORY_CASES'
    assert result['primary_origin_count']==2
    assert result['diagnostic_origin_count']==0
    assert result['primary_cohort_replaced'] is False
