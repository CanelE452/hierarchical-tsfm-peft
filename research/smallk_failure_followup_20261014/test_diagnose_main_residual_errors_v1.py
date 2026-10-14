"""Tiny CPU-only arithmetic fixtures; never opens scientific assets."""
import numpy as np
import pytest
import diagnose_main_residual_errors_v1 as diag


def test_unequal_channel_counts_conserve_true_channel_macro():
    y = np.zeros((2, 2, 2))
    mask = np.ones(y.shape, bool); mask[0, :, 1] = False; mask[1, 1, 1] = False
    p = np.ones_like(y); p[:, :, 1] = 2
    r = np.zeros_like(y)
    value = diag.decompose(p, r, y, mask, r)
    assert value['prediction_scores'] == {'mse': 2.5, 'mae': 1.5}
    assert value['channel'][0]['per_observed_cell_weight'] == .125
    assert value['channel'][1]['per_observed_cell_weight'] == .5
    for field in ('channel', 'block', 'error_bins'):
        assert sum(v['signed_mse_contribution'] for v in value[field]) == 2.5
        assert sum(v['signed_mae_contribution'] for v in value[field]) == 1.5


def test_outlier_bias_can_improve_mse_and_worsen_mae():
    y = np.zeros((4, 1, 1)); main = np.array([10., 0., 0., 0.]).reshape(4, 1, 1)
    p = np.array([9., 1., 1., 1.]).reshape(4, 1, 1)
    value = diag.decompose(p, main, y, np.ones(y.shape, bool), main)
    assert value['delta'] == {'mse': -4., 'mae': .5}
    assert value['error_bins'][4]['signed_mse_contribution'] == -4.75
    assert value['error_bins'][0]['signed_mae_contribution'] == .75
    assert max(abs(v) for v in value['conservation_residuals'].values()) == 0


def test_fixed_boundary_ties_and_selected_tune_tie_are_preserved():
    main = np.array([0., .25, .5, 1., 2.]).reshape(5, 1, 1)
    value = diag.decompose(main, main, np.zeros_like(main), np.ones(main.shape, bool), main)
    assert [r['observed_count'] for r in value['error_bins']] == [1] * 5
    assert [r['window_count'] for r in value['block']] == [2, 1, 1, 1]
    grid = [{'candidate_id': name, 'status': 'COMPLETE', 'tune_actual_Y_mse': score, 'alpha': None}
            for name, score in (('ZERO_CORRECTION', 1.), ('MEAN_CORRECTION', 1.), ('RIDGE_ALPHA_0', 2.))]
    rows = diag.tune_margins({'candidate_grid': grid, 'selected_index': 0, 'selected_candidate_id': 'ZERO_CORRECTION'})
    assert rows[0]['selected_margin_to_best_other'] == 0
    assert rows[0]['selected_improvement_vs_MEAN'] == 0
    assert rows[0]['is_selected'] is True and rows[1]['is_selected'] is False
    assert all(r['selection_changed'] is False for r in rows)


def test_masked_nan_and_no_observed_channel_are_safe_without_fake_scores():
    y = np.full((2, 1, 2), np.nan); p = y.copy(); r = y.copy()
    mask = np.zeros(y.shape, bool)
    y[:, :, 0] = 0.; p[:, :, 0] = 2.; r[:, :, 0] = 1.; mask[:, :, 0] = True
    value = diag.decompose(p, r, y, mask, r)
    assert value['delta'] == {'mse': 3., 'mae': 1.}
    assert value['channel'][1]['global_channel_observed_count'] == 0
    assert value['channel'][1]['per_observed_cell_weight'] == 0
    empty = diag.decompose(p, r, y, np.zeros(y.shape, bool), r)
    assert empty['prediction_scores'] == {'mse': None, 'mae': None}
    assert empty['delta'] == {'mse': None, 'mae': None}
    assert all(v['signed_mse_contribution'] == 0 for v in empty['channel'])
    bad = p.copy(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='FINITE_OBSERVED'):
        diag.decompose(bad, r, y, mask, r)
