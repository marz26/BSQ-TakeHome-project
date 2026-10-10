from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class FeatureCalculationError(ValueError):
    """Raised when input data cannot produce a valid trading feature.

    Parameters
    ----------
    message : str
        Description of the feature calculation failure.

    Returns
    -------
    None
        Exception constructors do not return a value.

    Raises
    ------
    None
        This exception class does not raise errors during construction.

    Notes
    -----
    This exception distinguishes invalid or insufficient input data from
    implementation errors in downstream feature calculations.
    """


def _validate_series(series: pd.Series, name: str) -> None:
    """Validate a numeric, non-empty series with finite values.

    Parameters
    ----------
    series : pandas.Series
        Input values to validate.
    name : str
        Logical name used in an error message.

    Returns
    -------
    None
        Validation completes without returning a value.

    Raises
    ------
    FeatureCalculationError
        If the series is empty, non-numeric, or contains missing/non-finite values.

    Notes
    -----
    The feature functions use complete paired observations rather than silently
    dropping rows, because dropping values independently can misalign the
    underlying price relationship.
    """
    if series.empty:
        raise FeatureCalculationError(f"{name} must not be empty")
    try:
        values = series.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise FeatureCalculationError(f"{name} must contain numeric values") from error
    if not np.isfinite(values).all():
        raise FeatureCalculationError(f"{name} must not contain missing or non-finite values")


def compute_hedge_ratio(
    price_a: pd.Series,
    price_b: pd.Series,
) -> tuple[float, float, pd.Series]:
    """Estimate an OLS hedge ratio and calculate the corresponding spread.

    Parameters
    ----------
    price_a : pandas.Series
        Price history of the dependent asset, indexed by observation date.
    price_b : pandas.Series
        Price history of the independent asset, indexed by observation date.

    Returns
    -------
    alpha : float
        Estimated intercept in ``Price_A = alpha + beta * Price_B + residual``.
    beta : float
        Estimated OLS slope used as the hedge ratio.
    spread : pandas.Series
        Residual spread ``Price_A - (alpha + beta * Price_B)`` with the input
        index preserved.

    Raises
    ------
    FeatureCalculationError
        If either input is empty or invalid, the series are not aligned, fewer
        than two observations are available, or OLS cannot be estimated.

    Notes
    -----
    Ordinary least squares with an explicit constant estimates alpha and beta
    by minimizing squared residuals in price space. The spread is computed
    from those same fitted coefficients and preserves the paired input index;
    data is rejected rather than implicitly reindexed or dropped.
    """
    _validate_series(price_a, "price_a")
    _validate_series(price_b, "price_b")
    if not price_a.index.equals(price_b.index):
        raise FeatureCalculationError("price_a and price_b must have identical indexes")
    if len(price_a) < 2:
        raise FeatureCalculationError("at least two aligned price observations are required")

    assert len(price_a) == len(price_b)
    assert price_a.index.equals(price_b.index)

    dependent = price_a.astype(float)
    independent = price_b.astype(float)
    if independent.nunique() < 2:
        raise FeatureCalculationError("price_b must contain at least two distinct values")

    try:
        design_matrix = sm.add_constant(independent, has_constant="add")
        model = sm.OLS(dependent, design_matrix, missing="raise").fit()
        alpha = float(model.params.iloc[0])
        beta = float(model.params.iloc[1])
    except (ValueError, np.linalg.LinAlgError, IndexError) as error:
        raise FeatureCalculationError("OLS hedge-ratio estimation failed") from error

    spread = (dependent - (alpha + beta * independent)).rename("spread")
    assert len(spread) == len(price_a)
    assert spread.index.equals(price_a.index)
    assert np.isfinite(alpha) and np.isfinite(beta)

    logger.info("Estimated OLS hedge ratio: alpha=%.6f, beta=%.6f", alpha, beta)
    return alpha, beta, spread


def compute_z_score(spread: pd.Series, window: int) -> pd.Series:
    """Standardize a spread using its trailing rolling mean and deviation.

    Parameters
    ----------
    spread : pandas.Series
        Spread history indexed by observation date.
    window : int
        Number of trailing observations used for each rolling statistic.
        Must be at least two and no larger than the input history.

    Returns
    -------
    pandas.Series
        Rolling z-score ``(spread - rolling_mean) / rolling_std``. Values
        before a complete window, and values with zero rolling deviation, are
        represented as ``NaN``.

    Raises
    ------
    FeatureCalculationError
        If the spread is empty or invalid, or ``window`` is not a valid rolling
        window for the supplied history.

    Notes
    -----
    Each observation is standardized against its current trailing window,
    including the observation itself. The sample standard deviation
    (``ddof=1``, pandas' rolling default) is used. A zero deviation makes the
    standardized value undefined, so those results remain ``NaN`` and a
    warning is logged rather than manufacturing a finite signal.
    """
    _validate_series(spread, "spread")
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise FeatureCalculationError("window must be an integer greater than or equal to 2")
    if window > len(spread):
        raise FeatureCalculationError("window must not exceed the number of spread observations")

    values = spread.astype(float)
    rolling = values.rolling(window=window, min_periods=window)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std(ddof=1)
    zero_deviation = rolling_std.eq(0)
    if zero_deviation.any():
        logger.warning(
            "Returning NaN z-scores for %d observations with zero rolling deviation",
            int(zero_deviation.sum()),
        )

    z_score = ((values - rolling_mean) / rolling_std).rename("z_score")
    assert len(z_score) == len(spread)
    assert z_score.index.equals(spread.index)
    return z_score