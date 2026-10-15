"""Past-window descriptions and A-only median groups for R5.

Inputs are already past-only filled/A-standardized windows [B,L,C]. No targets,
future masks, dataset loaders or model APIs are accepted here. STL can use the
whole observed past at an origin; historical-row prefix invariance is not claimed.

Primary definitions (checked 2026-10-15):
FPP3 section 4.3, https://otexts.com/fpp3/features.html#stl-features
    F_T=max(0,1-Var(R)/Var(T+R)); F_S=max(0,1-Var(R)/Var(S+R)).
statsmodels STL: https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html
    period is supplied by the caller, seasonal=7, robust=False; other defaults.
statsmodels ACF: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html
    unadjusted (denominator n), full-window mean, lag1 and metadata period.
SciPy Pearson: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    overlapping shifted vectors are separately centered for cross-channel lags.

Every channel summary averages finite channel values equally. Volatility is the
mean channel population std. Cross-channel description is the maximum absolute
Pearson correlation over ordered i!=j pairs and predeclared lags 1..min(24,L-2).
This simple maximum is not a p-value, causality test or incremental-predictive-
information estimate; common trends/autocorrelation can explain high values.
Its number of candidate pairs changes with C, so cross-dataset maxima are not
directly comparable. All indicators are descriptive, not learned selectors.

Undefined constant-series correlations and numerically degenerate STL ratios stay
NaN. A-only finite medians define low<=median/high>median; unavailable windows are
kept in their own group. All three groups must be reported, including empty ones.
"""

import importlib.metadata

import numpy as np

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:
    STL = None


MAX_CROSS_LAG = 24
SOURCES = {
    "stl_strength": "https://otexts.com/fpp3/features.html#stl-features",
    "stl_implementation": "https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html",
    "acf": "https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html",
    "pearson": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html",
}


def _channel_mean(values):
    finite = np.isfinite(values)
    count = finite.sum(axis=-1)
    return np.divide(np.where(finite, values, 0).sum(axis=-1), count,
                     out=np.full(values.shape[:-1], np.nan), where=count > 0)


def _acf(x, lag):
    if lag is None or lag < 1 or lag >= x.shape[1]:
        return np.full(x.shape[0], np.nan)
    centered = x - x.mean(axis=1, keepdims=True)
    denominator = np.square(centered).sum(axis=1)
    numerator = (centered[:, lag:] * centered[:, :-lag]).sum(axis=1)
    per_channel = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return _channel_mean(per_channel)


def _cross_lag_max(x):
    batch, length, channels = x.shape
    if channels < 2 or length < 3:
        return np.full(batch, np.nan)
    off_diagonal = ~np.eye(channels, dtype=bool)
    maxima = np.full(batch, -np.inf)
    for lag in range(1, min(MAX_CROSS_LAG, length - 2) + 1):
        late = x[:, lag:]
        early = x[:, :-lag]
        late = late - late.mean(axis=1, keepdims=True)
        early = early - early.mean(axis=1, keepdims=True)
        numerator = np.einsum("btc,btd->bcd", late, early)
        denominator = np.sqrt(np.square(late).sum(axis=1)[:, :, None] * np.square(early).sum(axis=1)[:, None, :])
        correlation = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        selected = np.abs(correlation[:, off_diagonal])
        local = np.where(np.isfinite(selected), selected, -np.inf).max(axis=1)
        maxima = np.maximum(maxima, local)
    return np.where(np.isfinite(maxima), np.clip(maxima, 0, 1), np.nan)


def _linear_fallback(x):
    time = np.arange(x.shape[1], dtype=np.float64)
    time = time - time.mean()
    centered = x - x.mean(axis=1, keepdims=True)
    slope = np.einsum("t,btc->bc", time, centered) / np.square(time).sum()
    total = np.square(centered).sum(axis=1)
    explained = np.square(slope) * np.square(time).sum()
    r2 = np.divide(explained, total, out=np.full_like(total, np.nan), where=total > 0)
    return {"linear_trend_r2": _channel_mean(np.clip(r2, 0, 1)), "linear_slope_abs": _channel_mean(np.abs(slope))}


def indicators(x, period):
    """Compute origin-level descriptions from historical windows only.

    period must be fixed using A or existing sampling metadata before evaluation.
    STL requires at least two full periods in this diagnostic contract. If the
    installed environment lacks STL, explicitly named linear R2/absolute slope
    replace it rather than being mislabeled as STL trend strength.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 3 or x.shape[2] < 1:
        raise ValueError("x must be past windows [B,L>=3,C>=1]")
    if not np.isfinite(x).all():
        raise ValueError("ROOT must provide finite past-only filled/A-standardized input")
    if period is not None:
        if int(period) != period or period < 2:
            raise ValueError("period must be a predeclared integer >=2 or None")
        period = int(period)
    result = {
        "volatility_std": np.std(x, axis=1, ddof=0).mean(axis=1),
        "acf_lag1": _acf(x, 1),
        "acf_period": _acf(x, period),
        "cross_channel_lag_max_abs_corr": _cross_lag_max(x),
    }
    if STL is None:
        result.update(_linear_fallback(x))
        return result
    trend_strength = np.full((x.shape[0], x.shape[2]), np.nan)
    seasonal_strength = np.full_like(trend_strength, np.nan)
    if period is not None and 2 * period <= x.shape[1]:
        for origin in range(x.shape[0]):
            for channel in range(x.shape[2]):
                series = x[origin, :, channel]
                numerical_variance_floor = np.finfo(np.float64).eps * max(1.0, np.var(series))
                if np.var(series) <= numerical_variance_floor:
                    continue
                fitted = STL(series, period=period, seasonal=7, robust=False).fit()
                remainder_variance = np.var(fitted.resid, ddof=0)
                for destination, component in ((trend_strength, fitted.trend), (seasonal_strength, fitted.seasonal)):
                    denominator = np.var(component + fitted.resid, ddof=0)
                    if denominator > numerical_variance_floor:
                        destination[origin, channel] = np.clip(1 - remainder_variance / denominator, 0, 1)
    result.update(trend_strength_stl=_channel_mean(trend_strength), seasonal_strength_stl=_channel_mean(seasonal_strength))
    return result


def fit_groups(A_indicators):
    """Fit one finite median/indicator on A only; None means no A boundary exists."""
    thresholds = {}
    for name, values in A_indicators.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Each indicator must have shape [origins]")
        finite = values[np.isfinite(values)]
        thresholds[name] = float(np.median(finite)) if len(finite) else None
    return thresholds


def assign_groups(values, thresholds):
    """Use frozen A medians; missing/undefined values have an explicit third group."""
    groups = {}
    for name, threshold in thresholds.items():
        indicator = np.asarray(values[name], dtype=np.float64)
        if indicator.ndim != 1:
            raise ValueError("Each indicator must have shape [origins]")
        finite = np.isfinite(indicator)
        if threshold is None:
            groups[f"{name}__low"] = np.zeros(indicator.shape, dtype=bool)
            groups[f"{name}__high"] = np.zeros(indicator.shape, dtype=bool)
            groups[f"{name}__unavailable"] = np.ones(indicator.shape, dtype=bool)
        else:
            if not np.isfinite(threshold):
                raise ValueError("Thresholds must be finite or None")
            groups[f"{name}__low"] = finite & (indicator <= threshold)
            groups[f"{name}__high"] = finite & (indicator > threshold)
            groups[f"{name}__unavailable"] = ~finite
    return groups


def indicator_definitions(period, length=512):
    """Serializable mapping for ROOT's single report/contract, without data access."""
    try:
        version = importlib.metadata.version("statsmodels")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "input": "ROOT supplied finite past-only filled/A-standardized [B,L,C]; no future/targets",
        "period": period, "length": length, "period_selection": "A-only or pre-existing sampling metadata, ROOT must record provenance",
        "statsmodels_version": version, "stl_available": STL is not None,
        "trend": "max(0,1-var(R)/var(T+R)), macro finite channel mean" if STL is not None else "explicit fallback linear_trend_r2 and linear_slope_abs; not STL strength",
        "seasonality": "max(0,1-var(R)/var(S+R)), macro finite channel mean" if STL is not None else "unavailable without STL",
        "stl_settings": {"seasonal": 7, "robust": False, "other_settings": "installed defaults, supplied period", "minimum_history_periods": 2,
                         "numerical_variance_floor": "float64 epsilon * max(1, variance(series)); degenerate ratios stay unavailable"},
        "volatility_std": "mean across channels of population std along observed historical grid",
        "acf": "sum[(x[t]-full_mean)*(x[t-lag]-full_mean)]/sum[(x-full_mean)^2], unadjusted, macro finite channel mean",
        "cross_channel_lag_max_abs_corr": "max |Pearson(x_i[lag:],x_j[:-lag])| over ordered i!=j and lag1..min(24,L-2), overlap separately centered",
        "cross_channel_lags": list(range(1, min(MAX_CROSS_LAG, length - 2) + 1)),
        "groups": "A finite median; low<=median/high>median/unavailable; report all counts and errors including empty groups",
        "interpretation": "past descriptions only; no causality, significance, independent-generalization or predictive-sufficiency claim",
        "sources": SOURCES,
    }
