import importlib

import pytest


def admission():
    try:
        return importlib.import_module('research.gt_ts2_fastcheck_20261018_v1.admission')
    except ModuleNotFoundError:
        pytest.fail('current eight-arm resource admission is missing')


def test_phase_a_consumption_can_deny_previously_fitting_training_set():
    a=admission()
    decision={'updates':300,'eight_arm_estimate_seconds':2500.}
    before={'GPU_used_seconds':1000.,'elapsed_seconds':2000.}
    after={'GPU_used_seconds':2000.,'elapsed_seconds':3000.}
    assert a.check(decision,before)['status']=='ADMITTED'
    denied=a.check(decision,after)
    assert denied['status']=='NOT_RUN_BUDGET'
    assert denied['registered_updates']==300
    assert decision['updates']==300


def test_admission_keeps_report_reserve_and_checks_both_clocks():
    a=admission()
    decision={'updates':100,'eight_arm_estimate_seconds':800.}
    assert a.check(decision,{'GPU_used_seconds':3520.,'elapsed_seconds':5000.})['status']=='ADMITTED'
    assert a.check(decision,{'GPU_used_seconds':3521.,'elapsed_seconds':5000.})['status']=='NOT_RUN_BUDGET'
    assert a.check(decision,{'GPU_used_seconds':0.,'elapsed_seconds':7841.})['status']=='NOT_RUN_BUDGET'
    assert a.check({'updates':None,'eight_arm_estimate_seconds':None},{'GPU_used_seconds':0.,'elapsed_seconds':0.})['status']=='NOT_RUN_BUDGET'
