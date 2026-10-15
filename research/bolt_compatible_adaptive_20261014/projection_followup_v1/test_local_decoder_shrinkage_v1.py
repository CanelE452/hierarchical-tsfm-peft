import inspect

import numpy as np
import pytest

from local_decoder_shrinkage_v1 import fit_shrunk_local_decoder


def fixture():
    rng = np.random.default_rng(144)
    z = rng.normal(size=(2, 512, 2)).astype(np.float32)
    prior = np.array([[.2, 1., -.3], [.1, -.5, .7]], np.float32)
    truth = np.array([[1., .8, 2.], [-.5, 1., .1]], np.float32)
    x = (z @ truth + np.array([2., -1., 3.], np.float32)).astype(np.float32)
    observed = np.ones_like(x, bool)
    observed[:, ::4, 1] = False
    x[:, ::4, 1] = 1000.
    return x, observed, z, prior


def test_coefficients_match_independent_augmented_least_squares():
    x, observed, z, prior = fixture(); rho = .1
    local = fit_shrunk_local_decoder(x, observed, z, prior, rho)
    for n in range(2):
        for c in range(3):
            valid = observed[n, :, c]
            features, target = z[n, valid].astype(np.float64), x[n, valid, c].astype(np.float64)
            mean_z, mean_y = features.mean(axis=0), target.mean()
            centered = features - mean_z
            penalty = rho * np.sum(centered * centered) / len(features) / 2
            design = np.vstack([centered / np.sqrt(len(features)), np.sqrt(penalty) * np.eye(2)])
            response = np.concatenate([(target - mean_y) / np.sqrt(len(features)), np.sqrt(penalty) * prior[:, c]])
            beta = np.linalg.lstsq(design, response, rcond=None)[0]
            np.testing.assert_allclose(local.decoder_fit64[n, :, c], beta, atol=1e-12)
            assert abs(local.bias_fit64[n, c] - (mean_y - mean_z @ beta)) < 1e-12
    assert np.max(local.diagnostics["real_stationarity_residual_norm"]) < 1e-12


def test_filled_response_is_excluded_from_fit():
    x, observed, z, prior = fixture()
    first = fit_shrunk_local_decoder(x, observed, z, prior, 1.)
    x[~observed] = -1e10
    second = fit_shrunk_local_decoder(x, observed, z, prior, 1.)
    np.testing.assert_array_equal(first.decoder_fit64, second.decoder_fit64)
    np.testing.assert_array_equal(first.bias_fit64, second.bias_fit64)


def test_zero_trace_and_single_observation_keep_prior_slope():
    x, observed, z, prior = fixture()
    z[:] = 2.; observed[0] = False; observed[0, 7] = True
    local = fit_shrunk_local_decoder(x, observed, z, prior, .001)
    np.testing.assert_array_equal(local.decoder32, np.broadcast_to(prior[None], (2, 2, 3)))
    assert np.array(local.diagnostics["zero_trace_prior_slope_retained"]).all()


def test_no_observation_fallback_uses_prior_and_current_intercept():
    x, observed, z, prior = fixture(); observed[:] = False
    local = fit_shrunk_local_decoder(x, observed, z, prior, 1.)
    np.testing.assert_array_equal(local.decoder32, np.broadcast_to(prior[None], (2, 2, 3)))
    np.testing.assert_allclose(local.bias_fit64, x[:, -1].astype(np.float64) - z[:, -1].astype(np.float64) @ prior.astype(np.float64), atol=1e-12)
    assert np.array(local.diagnostics["no_observation_prior_with_current_intercept"]).all()


def test_zero_innovation_preserves_numeric_current_anchor():
    x, observed, z, prior = fixture()
    local = fit_shrunk_local_decoder(x, observed, z, prior, .01)
    future = np.repeat(z[:, -1:], 48, axis=1)
    np.testing.assert_array_equal(local.forecast(x[:, -1], future, z[:, -1]), np.repeat(x[:, -1:], 48, axis=1))


def test_relative_regularization_does_not_bound_absolute_gain():
    s = np.tile(np.array([-1., 1.], np.float32), 256)
    z = (s * 1e-7)[None, :, None]
    x = np.stack([s, 2 * s], axis=1)[None]
    local = fit_shrunk_local_decoder(x, np.ones_like(x, bool), z, np.zeros((1, 2), np.float32), .001)
    assert local.decoder_fit64.max() > 1e7
    assert max(local.diagnostics["ridge_matrix_condition"][0]) == 1.
    assert local.diagnostics["absolute_decoder_gain_bound_claimed"] is False


def test_rank_deficient_fit_retains_prior_unidentified_coordinates():
    s = np.tile(np.array([-1., 1.], np.float32), 256)
    z = np.stack([s, s], axis=1)[None]
    x = np.stack([s, 2 * s], axis=1)[None]
    prior = np.array([[2., 0.], [-2., 0.]], np.float32)
    local = fit_shrunk_local_decoder(x, np.ones_like(x, bool), z, prior, .1)
    np.testing.assert_allclose(local.decoder_fit64[0, 0] - local.decoder_fit64[0, 1], [4., 0.], atol=1e-12)


@pytest.mark.parametrize("rho", [0, -1., float("nan"), float("inf"), True])
def test_invalid_rho_is_rejected(rho):
    with pytest.raises(ValueError, match="POSITIVE_RHO"): fit_shrunk_local_decoder(*fixture(), rho)


def test_fit_api_has_no_future_or_selection_argument():
    assert tuple(inspect.signature(fit_shrunk_local_decoder).parameters) == ("history", "observed", "latent_history", "prior_decoder", "rho")
