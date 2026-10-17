import importlib
import numpy as np
import pytest
import torch


def impl():
    try:
        return importlib.import_module('research.adapter_diagnostics_20261017_v2.new_jobs')
    except ModuleNotFoundError:
        pytest.fail('New diagnostic solver not implemented')


def test_horizon_solver_uses_global_channel_count():
    m = impl()
    rng = np.random.default_rng(92)
    x = rng.normal(size=(5, 3, 2))
    y = rng.normal(size=(5, 3, 2))
    mask = rng.random(y.shape) > .3
    rms = np.sqrt((x*x).mean((0, 1)))
    fitted = m.fit_decoder(x, y, mask, rms, .1, horizon_specific=True)
    for h in range(3):
        for c in range(2):
            active = mask[:, h, c]
            f = x[active, h] / rms
            target = y[active, h, c]
            expected = np.linalg.solve(f.T @ f + .1 * mask[:, :, c].sum()/3*np.eye(2), f.T @ target)
            np.testing.assert_allclose(fitted[h, :, c], expected, rtol=1e-12, atol=1e-12)


def test_qr_rng_namespace_keeps_training_rng_and_d_inverse():
    m = impl()
    torch.manual_seed(92601)
    before = torch.get_rng_state().clone()
    q = m.qr_basis(21, 6, 10092601)
    assert torch.equal(before, torch.get_rng_state())
    np.testing.assert_allclose(q.T @ q, np.eye(6), atol=2e-7)
    np.testing.assert_allclose(q @ np.linalg.pinv(q) @ q, q, atol=2e-7)
    np.testing.assert_array_equal(q, m.qr_basis(21, 6, 10092601))


def test_score_reaggregates_masked_macro_and_keeps_zero_targets():
    m = impl()
    y = np.zeros((2, 3, 2), np.float32)
    p = np.ones_like(y)
    p[:, :, 1] = 2
    mask = np.ones_like(y, dtype=bool)
    mask[1, :, 0] = False
    result = m.array_metrics(p, y, mask)
    assert result['MSE'] == 2.5
    assert result['MAE'] == 1.5
    assert result['counts'] == [3, 6]


def test_prefix_keeps_original_registered_split_boundaries():
    m = impl()
    task = {'physical':np.zeros((16772,21)),
            'split_ranges':dict(A_fit=[0,10482], A_stop=[10482,12579], M_cal=[12579,14675],
                                V_select=[14675,16772], T_final=[16772,20965])}
    actual = m.registered_jena_splits(task)
    assert actual['A_fit'] == (0,10482)
    assert actual['A_stop'] == (10482,12579)
    assert actual['T_final'] == (16772,20965)


def endpoint_fixtures():
    epochs = (0., .25, 1., 3.)
    exposures = (0, 5, 17, 51)
    updates = (0, 2, 5, 14)
    return [(str(i), dict(updates=updates[i], sampler={'exposures': exposures[i]}, extra={
        'completed_epochs': list(epochs[:i+1]),
        'curves': [dict(epoch=epochs[i], exposures=exposures[i], updates=updates[i], astop_mse=.1)]}))
        for i in range(4)]


def test_common_endpoint_selection_skips_retained_mid_segment_resume_checkpoint():
    m = impl()
    checkpoints = endpoint_fixtures()
    partial = ('partial', dict(updates=9, sampler={'exposures': 30}, extra={
        'completed_epochs': [0., .25, 1.],
        'curves': [dict(epoch=1., exposures=17, updates=5, astop_mse=.1)]}))
    checkpoints.insert(3, partial)
    selected = m.select_common_endpoints(checkpoints, origin_count=17, effective_batch=4)
    assert [path for path, checkpoint in selected] == ['0', '1', '2', '3']
    assert [checkpoint['updates'] for path, checkpoint in selected] == [0, 2, 5, 14]


def test_common_endpoint_selection_rejects_missing_endpoint_and_stale_curve():
    m = impl()
    checkpoints = endpoint_fixtures()
    checkpoints[2][1]['extra']['curves'][-1]['exposures'] = 16
    with pytest.raises(ValueError, match='Missing registered common endpoint'):
        m.select_common_endpoints(checkpoints, origin_count=17, effective_batch=4)
