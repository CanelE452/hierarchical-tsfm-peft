import numpy as np
import pytest

import masked_linear as ml


def dense_temporal_fit(x, y, penalty):
    examples = x.transpose(0, 2, 1).reshape(-1, x.shape[1]).astype(np.float64)
    target = y.transpose(0, 2, 1).reshape(-1, y.shape[1]).astype(np.float64)
    xmean = examples.mean(axis=0)
    ymean = target.mean(axis=0)
    xc = examples - xmean
    yc = target - ymean
    covariance = xc.T @ xc / len(examples)
    cross = xc.T @ yc / len(examples)
    weight = np.linalg.solve(covariance + penalty * np.eye(covariance.shape[0]), cross)
    return {"weight": weight, "bias": ymean - xmean @ weight}


def masked_horizon_fit(x, y_h, mask_h, penalty):
    examples = x.transpose(0, 2, 1).reshape(-1, x.shape[1])
    target = y_h.reshape(-1)
    observed = mask_h.reshape(-1)
    return dense_temporal_fit(
        examples[observed, :, None].transpose(0, 1, 2),
        target[observed, None, None],
        penalty,
    )


def orthonormal_basis(channels=4, rank=2):
    rng = np.random.default_rng(57)
    q, _ = np.linalg.qr(rng.normal(size=(channels, channels)))
    return q[:, :rank]


def assert_fit_close(actual, expected, atol=1e-10):
    np.testing.assert_allclose(actual["weight"], expected["weight"], atol=atol, rtol=1e-10)
    np.testing.assert_allclose(actual["bias"], expected["bias"], atol=atol, rtol=1e-10)


def test_shared_dense_mask_matches_dense_ridge():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 4, 3))
    true_w = rng.normal(size=(4, 2))
    y = np.einsum("blc,lh->bhc", x, true_w) + rng.normal(size=(5, 2, 3)) * 0.01
    mask = np.ones_like(y, dtype=bool)
    fit = ml.fit_masked_shared(x, y, mask, penalty=0.05)
    expected = dense_temporal_fit(x, y, penalty=0.05)
    assert_fit_close(fit, expected)
    np.testing.assert_allclose(ml.predict_shared(x, fit), ml.predict_shared(x, expected), atol=1e-10, rtol=1e-10)
    assert fit["target_counts"].tolist() == [15, 15]


def test_shared_masked_horizon_matches_direct_solution():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 5, 2))
    y = rng.normal(size=(4, 3, 2))
    mask = np.ones_like(y, dtype=bool)
    mask[0, 0, 1] = False
    mask[1, 1, :] = False
    mask[2:, 2, 0] = False
    fit = ml.fit_masked_shared(x, y, mask, penalty=0.2)
    assert fit["target_counts"].tolist() == [7, 6, 6]
    for h in range(y.shape[1]):
        expected = masked_horizon_fit(x, y[:, h, :], mask[:, h, :], penalty=0.2)
        np.testing.assert_allclose(fit["weight"][:, h], expected["weight"][:, 0], atol=1e-10, rtol=1e-10)
        assert fit["bias"][h] == pytest.approx(expected["bias"][0], abs=1e-10)


def test_shared_missing_target_values_are_ignored():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 4, 3))
    y = rng.normal(size=(4, 2, 3))
    mask = np.ones_like(y, dtype=bool)
    mask[0, 0, 1] = False
    mask[3, 1, 2] = False
    fit = ml.fit_masked_shared(x, y, mask, penalty=0.1)
    changed = y.copy()
    changed[~mask] = np.nan
    changed[0, 0, 1] = 1e9
    fit_changed = ml.fit_masked_shared(x, changed, mask, penalty=0.1)
    assert_fit_close(fit_changed, fit)


def test_factor_dense_mask_matches_dense_factor_ridge():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, 4, 4))
    y = rng.normal(size=(5, 3, 4))
    basis = orthonormal_basis(channels=4, rank=2)
    mask = np.ones_like(y, dtype=bool)
    fit = ml.fit_masked_factor(x, y, mask, basis, penalty=0.05)

    z = x @ basis
    zy = y @ basis
    residual_x = x - z @ basis.T
    residual_y = y - zy @ basis.T
    expected_common = dense_temporal_fit(z, zy, penalty=0.05)
    expected_residual = dense_temporal_fit(residual_x, residual_y, penalty=0.05)
    assert_fit_close(fit["common"], expected_common)
    assert_fit_close(fit["residual"], expected_residual)
    np.testing.assert_allclose(
        ml.predict_factor(x, fit),
        ml.predict_shared(z, expected_common) @ basis.T + ml.predict_shared(residual_x, expected_residual),
        atol=1e-10,
        rtol=1e-10,
    )
    assert fit["full_vector_origin_counts"].tolist() == [5, 5, 5]


def test_factor_excludes_partially_observed_future_vectors():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 3, 3))
    y = rng.normal(size=(4, 2, 3))
    basis = orthonormal_basis(channels=3, rank=2)
    mask = np.ones_like(y, dtype=bool)
    mask[1, 0, 2] = False
    mask[2, 1, 0] = False
    fit = ml.fit_masked_factor(x, y, mask, basis, penalty=0.3)
    assert fit["full_vector_origin_counts"].tolist() == [3, 3]
    assert fit["common"]["target_counts"].tolist() == [6, 6]
    assert fit["residual"]["target_counts"].tolist() == [9, 9]

    changed = y.copy()
    changed[1, 0, :] = [1000.0, -2000.0, np.nan]
    changed[2, 1, :] = [np.nan, 1234.0, -999.0]
    changed_fit = ml.fit_masked_factor(x, changed, mask, basis, penalty=0.3)
    assert_fit_close(changed_fit["common"], fit["common"])
    assert_fit_close(changed_fit["residual"], fit["residual"])


def test_no_target_errors_are_explicit():
    x = np.ones((2, 3, 2))
    y = np.ones((2, 2, 2))
    mask = np.ones_like(y, dtype=bool)
    mask[:, 1, :] = False
    with pytest.raises(ValueError, match="no observed target examples"):
        ml.fit_masked_shared(x, y, mask, penalty=0.1)

    factor_mask = np.ones_like(y, dtype=bool)
    factor_mask[:, 0, 0] = False
    with pytest.raises(ValueError, match="factor unavailable"):
        ml.fit_masked_factor(x, y, factor_mask, np.eye(2, 1), penalty=0.1)
