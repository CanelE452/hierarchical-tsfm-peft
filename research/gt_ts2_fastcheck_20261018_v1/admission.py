"""Recheck the sealed full-set estimate after Phase A, without changing updates."""
import math


def check(decision,budget):
    updates=decision['updates']
    available=min(5400*.8-budget['GPU_used_seconds'],10800*.8-budget['elapsed_seconds'])
    estimate=decision['eight_arm_estimate_seconds']
    fits=updates in (100,300) and isinstance(estimate,(int,float)) and math.isfinite(estimate) and 0<estimate<=available
    return dict(status='ADMITTED' if fits else 'NOT_RUN_BUDGET',registered_updates=updates,
        full_eight_arm_estimate_seconds=estimate,current_training_available_seconds=available,
        budget=budget,reason='FULL_EIGHT_ARM_SET_PLUS_REPORT_RESERVE_FITS' if fits else 'FULL_SET_NO_LONGER_FITS_WITH_RESERVE',
        updates_changed=False,selection_uses_VAL_scores=False)
