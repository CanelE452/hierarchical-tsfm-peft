import numpy as np


def seasonal_scales(train_values, seasonality=12):
    values = np.asarray(train_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] <= seasonality:
        raise ValueError("Need nodes x TRAIN times with more than one season")
    raw = np.mean(np.square(values[:, seasonality:] - values[:, :-seasonality]), axis=1)
    positive = raw[raw > 0]
    floor = max(1e-8, (float(np.mean(positive)) if len(positive) else 1.0) * 1e-8)
    return np.maximum(raw, floor), {"floor": floor, "floored_series": int(np.sum(raw < floor)), "role": "TRAIN"}


def level_weights(n_nodes, tags):
    weights = np.zeros(n_nodes, dtype=np.float64)
    for indices in tags.values():
        weights[np.asarray(indices)] += 1.0 / (len(tags) * len(indices))
    if not np.isclose(weights.sum(), 1) or np.any(weights <= 0):
        raise ValueError("Invalid official level coverage")
    return weights


def score(y, prediction, scales, tags, bottom_indices, S):
    # Inputs are origins x nodes x horizon; aggregate each series over origins and horizons.
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or y.ndim != 3:
        raise ValueError("Expected matching origins x nodes x horizon arrays")
    error = prediction - y
    node_rmsse = np.sqrt(np.mean(error ** 2, axis=(0, 2)) / scales)
    levels = {name: float(np.mean(node_rmsse[indices])) for name, indices in tags.items()}
    bottom_sum = np.einsum("nb,obh->onh", S, prediction[:, bottom_indices, :])
    result = {"primary": float(np.mean(list(levels.values()))),
              "bottom_rmsse": float(np.mean(node_rmsse[bottom_indices])),
              "raw_mae": float(np.mean(np.abs(error))),
              "coherence_mae": float(np.mean(np.abs(prediction - bottom_sum))),
              "coherence_max": float(np.max(np.abs(prediction - bottom_sum)))}
    if not all(np.isfinite(list(result.values()))):
        raise ValueError("Nonfinite metric")
    return result, levels
