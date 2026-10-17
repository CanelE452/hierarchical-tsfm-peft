import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def impl():
    try:
        return importlib.import_module('research.adapter_improvements_20261018_v1.initialization')
    except ModuleNotFoundError:
        pytest.fail('DCA initialization module is not implemented')


@pytest.fixture
def official():
    path = ROOT / '.cache/adapter_improvements_20261018_v1/official'
    if not (path / 'DCA/src/dca/dca.py').is_file():
        pytest.skip('Pinned official DCA source is an explicit private replay dependency')
    return path


def task_fixture():
    rng = np.random.default_rng(812)
    x = rng.normal(size=(180, 3)).astype(np.float32)
    x[1:] += .6 * x[:-1]
    observed = np.ones_like(x, bool)
    observed[11:14, 1] = False
    x[~observed] = np.nan
    return dict(name='synthetic', physical=x, observed=observed, columns=['a', 'b', 'c'],
                times=np.arange(180), split_ranges={'A_fit': (0, 110), 'A_stop': (110, 140)})


def normalization(task, fit_end=110, k=2):
    dc = importlib.import_module('research.informs_broad_validation_20261015_v1.data_contract')
    return dc.initialize_pca(task, fit_end, k)


def test_gaussian_pi_analytic_t1_and_invalid_sign():
    m = impl()
    rho = np.array([.2, .7, -.4])
    cross = np.stack([np.eye(3), np.diag(rho)])
    assert m.gaussian_pi(cross, np.eye(3)) == pytest.approx(-.5 * np.log1p(-rho**2).sum(), abs=1e-13)
    with pytest.raises(ValueError, match='positive definite'):
        m.gaussian_pi(np.stack([np.eye(3), np.diag([2., 2., .2])]), np.eye(3))


def test_fresh_gradient_repeats_after_value_and_matches_finite_difference():
    m = impl()
    objective = m.FreshObjective(lambda z: (z.sin() + .3 * z.square()).sum())
    x = np.array([.2, -.7, 1.3], dtype=np.float64)
    expected = np.cos(x) + .6 * x
    objective.func(x)
    g1, g2 = objective.grad(x), objective.grad(x)
    np.testing.assert_allclose(g1, expected, rtol=1e-13)
    np.testing.assert_array_equal(g1, g2)
    h = 1e-6
    fd = np.array([(objective.func(x + np.eye(3)[j] * h) - objective.func(x - np.eye(3)[j] * h)) / (2*h) for j in range(3)])
    np.testing.assert_allclose(g1, fd, rtol=1e-9, atol=1e-9)


def test_official_objective_parity_and_upstream_repeated_gradient_bug(official):
    m = impl()
    api = m.official_api(official)
    cross = torch.tensor(np.stack([np.eye(3), np.diag([.7, .2, .4])]), dtype=torch.float64)
    loss = api['dca'].build_loss(cross, 2, 10., False)
    x = np.array([[1., .1], [.2, 1.], [.1, -.3]]).ravel()
    safe = m.FreshObjective(loss)
    leaf = torch.tensor(x, requires_grad=True)
    value = loss(leaf)
    grad, = torch.autograd.grad(value, leaf)
    assert safe.func(x) == pytest.approx(float(value.detach()), abs=1e-13)
    np.testing.assert_allclose(safe.grad(x), grad.numpy(), atol=1e-12)
    upstream = api['base'].ObjectiveWrapper(lambda z: _leaf_loss(loss, z))
    first = upstream.grad(x).copy()
    np.testing.assert_allclose(upstream.grad(x), 2*first, atol=1e-12)
    np.testing.assert_array_equal(safe.grad(x), safe.grad(x))
    h = 1e-6
    fd = np.array([(safe.func(x+np.eye(len(x))[j]*h)-safe.func(x-np.eye(len(x))[j]*h))/(2*h) for j in range(len(x))])
    np.testing.assert_allclose(safe.grad(x), fd, rtol=2e-7, atol=2e-7)


def _leaf_loss(loss, z):
    leaf = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    return loss(leaf), leaf


def test_covariance_exact_official_path_and_rectification(official):
    m = impl()
    x = np.random.default_rng(4).normal(size=(130, 3))
    cross, diag = m.official_statistics(x, official, T=2)
    cov = m.official_api(official)['cov']
    expected = cov.calc_cross_cov_mats_from_data(x, 4, stride=1, chunks=None,
                                                regularization=None, method='toeplitzify')
    np.testing.assert_array_equal(cross, expected)
    assert diag['lag_rows'] == len(x)-4+1
    assert diag['rectification_shift'] == 0
    singular = np.zeros((40, 3))
    _, diag = m.official_statistics(singular, official, T=2)
    assert diag['rectification_shift'] == pytest.approx(1e-6)
    assert diag['covariance']['min_eigenvalue'] > 0


def test_stream_matches_pca_normalization_and_cannot_read_future():
    m = impl()
    task = task_fixture()
    pca = normalization(task)
    stream, meta = m.prepare_stream(task, 110, 2, pca)
    changed = dict(task, physical=task['physical'].copy(), observed=task['observed'].copy())
    changed['physical'][110:] = 1e20
    changed['observed'][110:] = False
    again, meta2 = m.prepare_stream(changed, 110, 2, pca)
    np.testing.assert_array_equal(stream, again)
    assert meta == meta2
    assert meta['fit_scope']['fit_end'] == 110
    assert meta['imputed_count'] == 3
    assert meta['center_residual_max'] < 1e-7
    with pytest.raises(ValueError, match='scope'):
        m.prepare_stream(task, 111, 2, pca)


def test_procrustes_keeps_subspace_and_pi_and_rejects_rank_loss():
    m = impl()
    rng = np.random.default_rng(512)
    pca, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    v = rng.normal(size=(5, 2))
    e, diag = m.align_to_pca(v, pca)
    q, _ = np.linalg.qr(v)
    np.testing.assert_allclose(e@e.T, q@q.T, atol=1e-13)
    np.testing.assert_allclose(e.T@e, np.eye(2), atol=1e-13)
    cross = np.stack([np.eye(5), np.diag([.1, .2, .3, .4, .5])])
    assert m.gaussian_pi(cross, v) == pytest.approx(m.gaussian_pi(cross, e), abs=1e-12)
    assert diag['rank'] == 2
    with pytest.raises(ValueError, match='rank'):
        m.align_to_pca(np.ones((5, 2)), pca)


def test_install_uses_actual_e32_pinv_and_zero_other_lags():
    m = impl()
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.float32))
            self.D = torch.nn.Parameter(torch.ones(7, 2, 3, dtype=torch.float32))
    model = Adapter()
    e64 = np.array([[1., .2], [.3, .9], [.1, .4]], np.float64)
    m.install_initialization(model, e64.astype(np.float32))
    assert torch.equal(model.D[0], torch.linalg.pinv(model.E))
    assert torch.count_nonzero(model.D[1:]) == 0
    with pytest.raises(ValueError, match='FP32'):
        m.install_initialization(model, e64)


def test_bounded_solver_returns_explicit_timeout_and_nonconvergence(official):
    m = impl()
    x = np.random.default_rng(22).normal(size=(160, 3))
    init = np.eye(3)[:, :2]
    timed = m.fit_official_dca(x, init, official, max_seconds=.001)
    assert timed['status'] == 'TIMEOUT'
    assert not timed['converged']
    limited = m.fit_official_dca(x, init, official, maxiter=0, max_seconds=30)
    assert limited['status'] == 'NOT_CONVERGED'
    assert not limited['converged']


def test_fit_cache_is_split_bound_read_only_and_sha_checked(tmp_path, official, monkeypatch):
    m = impl()
    task = task_fixture()
    pca = normalization(task)
    ctx = dict(cache_root=tmp_path, official_source_root=official)
    result = m.fit_dca_initialization(ctx, task, 110, 2, pca)
    assert result['status'] == 'COMPLETE', result
    assert result['converged']
    assert len(result['cache_sha256']) == 64
    np.testing.assert_equal(result['fit_scope']['fit_end'], 110)
    monkeypatch.setattr(m, 'fit_official_dca', lambda *a, **kw: pytest.fail('Loader optimized DCA'))
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}
    e32, meta = m.load_initialization(ctx, task, 110, 2, pca)
    assert e32.dtype == np.float32 and e32.shape == (3, 2)
    assert meta['cache_sha256'] == result['cache_sha256']
    assert before == {p: p.stat().st_mtime_ns for p in before}
    with pytest.raises(FileNotFoundError):
        m.load_initialization(ctx, task, 140, 2, normalization(task, 140))
    data_path = tmp_path / result['cache_binding']
    data_path.write_bytes(data_path.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='hash'):
        m.load_initialization(ctx, task, 110, 2, pca)


def test_scope_cache_identity_changes_for_mask_scaler_and_source(tmp_path):
    m = impl()
    task = task_fixture()
    pca = normalization(task)
    a = m.initialization_binding(task, 110, 2, pca)
    changed = dict(task, observed=task['observed'].copy())
    changed['observed'][0, 0] = False
    assert m.initialization_binding(changed, 110, 2, pca)['key'] != a['key']
    bad_norm = dict(pca, mean=pca['mean']+.1)
    assert m.initialization_binding(task, 110, 2, bad_norm)['key'] != a['key']
    evaluation = importlib.import_module('research.informs_broad_validation_20261015_v1.evaluation')
    training = importlib.import_module('research.informs_broad_validation_20261015_v1.training')
    expected = training.fingerprint({'mean': evaluation.array_fingerprint(pca['mean']), 'std': evaluation.array_fingerprint(pca['std'])})
    assert a['binding']['scaler_sha256'] == expected
