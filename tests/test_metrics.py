import numpy as np
from hier_peft.metrics import score, seasonal_scales, level_weights


def test_hierarchy_balancing_does_not_count_large_level_more():
    tags = {"total": np.array([0]), "bottom": np.array([1, 2])}
    y = np.zeros((2, 3, 12))
    prediction = np.ones_like(y)
    prediction[:, 0] = 3
    result, levels = score(y, prediction, np.ones(3), tags, np.array([1, 2]),
                           np.array([[1, 1], [1, 0], [0, 1]]))
    assert levels == {"total": 3.0, "bottom": 1.0}
    assert result["primary"] == 2.0
    np.testing.assert_allclose(level_weights(3, tags), [0.5, 0.25, 0.25])


def test_scale_is_seasonal_and_floors_constant_train_series():
    train = np.stack([np.arange(36), np.ones(36)]).astype(float)
    scales, audit = seasonal_scales(train)
    assert scales[0] == 144
    assert scales[1] == audit["floor"] > 0
    assert audit["floored_series"] == 1


def test_zero_error_and_bottom_up_coherence():
    S = np.array([[1, 1], [1, 0], [0, 1]])
    y = np.einsum("nb,obh->onh", S, np.arange(48).reshape(2, 2, 12))
    result, _ = score(y, y.copy(), np.ones(3), {"total": [0], "bottom": [1, 2]}, [1, 2], S)
    assert all(v == 0 for v in result.values())
