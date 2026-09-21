import numpy as np
import pytest
from hierarchicalforecast.methods import MinTrace
from hier_peft.reconcile import Reconciliation


def test_official_mint_parity_and_coherence_with_permuted_nodes():
    rng = np.random.default_rng(123)
    S = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
    y = np.einsum("nb,obh->onh", S, rng.normal(10, 3, (13, 2, 12)))
    forecast = y + rng.normal(1, 2, y.shape)
    future_pred = rng.normal(10, 3, (2, 3, 12))
    official = MinTrace(method="mint_shrink").fit(S=S, y_hat=future_pred[0],
        y_insample=y.transpose(1, 0, 2).reshape(3, -1),
        y_hat_insample=forecast.transpose(1, 0, 2).reshape(3, -1))
    perm = [2, 0, 1]
    rec = Reconciliation(S[perm], [2, 0]).fit(y[:, perm], forecast[:, perm], role="CALIBRATION")
    actual = rec.transform(future_pred[:, perm], "mint_shrink")[:, np.argsort(perm)]
    expected = np.stack([official.predict(S=S, y_hat=p)["mean"] for p in future_pred])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual[:, 0], actual[:, 1] + actual[:, 2], atol=1e-10)
    assert rec.audit["residual_observations"] == 156


def test_refuse_test_labels_for_covariance():
    S = np.array([[1, 1], [1, 0], [0, 1]])
    x = np.ones((13, 3, 12))
    with pytest.raises(ValueError, match="CALIBRATION-only"):
        Reconciliation(S, [1, 2]).fit(x, x, role="TEST")
