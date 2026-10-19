import numpy as np

from research.incremental_latent_adaptation_20261019_v1.probes import ridge_fit, ridge_predict


def test_masked_penalty_uses_total_channel_count_and_unpenalized_intercept():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(12, 3, 2))
    y = rng.normal(size=(12, 4, 2)) + 10
    mask = rng.random(y.shape) > .3
    mask[:5, 0, 0] = False
    lam = .1
    for kind in ('OWN', 'ALL'):
        model = ridge_fit(x, y, mask, lam, kind)
        prediction = ridge_predict(model, x)
        for c in range(2):
            source = x[:, :, c] if kind == 'OWN' else x.transpose(0, 2, 1).reshape(12, -1)
            j = c if kind == 'OWN' else 0
            source = (source-model['means'][j])/model['scales'][j]
            for h in range(4):
                active = mask[:, h, c]
                error = prediction[active, h, c]-y[active, h, c]
                np.testing.assert_allclose(error.sum(), 0, atol=1e-10)
                derivative = source[active].T @ error + lam*mask[:, :, c].sum()*model['weights'][h,c]
                np.testing.assert_allclose(derivative, 0, atol=1e-10)


def test_identical_masks_share_factorization_and_linear_predictions():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8, 3, 2))
    y = rng.normal(size=(8, 4, 2))
    model = ridge_fit(x, y, np.ones_like(y, dtype=bool), .01, 'ALL')
    assert model['factorizations'] == 1
    assert ridge_predict(model, x).shape == y.shape
