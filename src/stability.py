from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class StabilityAnalysisError(ValueError):
    """Raised when stability statistics cannot be computed from the inputs.

    Parameters
    ----------
    message : str
        Description of the invalid data or unsupported calculation.

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
    Malformed or misaligned inputs are rejected explicitly rather than
    silently dropped, preserving the date correspondence needed for
    retrospective performance comparisons.
    """


@dataclass(frozen=True)
class StabilityPerformanceComparison:
    """Compare daily strategy PnL on stable and non-stationary observations.

    Parameters
    ----------
    significance_level : float
        ADF p-value cutoff used to label observations.
    stable_observations : int
        Matched observations with p-value at or below the cutoff.
    unstable_observations : int
        Matched observations with p-value above the cutoff.
    stable_pnl : float or None
        Sum of daily PnL on stable observations, or ``None`` if absent.
    unstable_pnl : float or None
        Sum of daily PnL on unstable observations, or ``None`` if absent.
    stable_mean_daily_pnl : float or None
        Mean daily PnL on stable observations, or ``None`` if absent.
    unstable_mean_daily_pnl : float or None
        Mean daily PnL on unstable observations, or ``None`` if absent.
    stable_max_drawdown : float or None
        Worst drawdown observed on stable dates, or ``None`` if absent.
    unstable_max_drawdown : float or None
        Worst drawdown observed on unstable dates, or ``None`` if absent.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Instances are created by :func:`compare_stability_with_performance`.

    Notes
    -----
    Stable versus unstable daily PnL is a descriptive split, not evidence that
    non-stationarity caused a loss. Drawdowns are measured on the cumulative
    PnL curve over the full matched timeline, then attributed to the ADF label
    at the date each drawdown observation occurred.
    """

    significance_level: float
    stable_observations: int
    unstable_observations: int
    stable_pnl: float | None
    unstable_pnl: float | None
    stable_mean_daily_pnl: float | None
    unstable_mean_daily_pnl: float | None
    stable_max_drawdown: float | None
    unstable_max_drawdown: float | None


def _as_date_index(
    series: pd.Series,
    name: str,
    allow_nan: bool = False,
) -> pd.Series:
    """Validate a time series and normalize its index to calendar dates.

    Parameters
    ----------
    series : pandas.Series
        Numeric series to validate.
    name : str
        Logical input name used in error messages.
    allow_nan : bool, default=False
        Whether missing numeric values are accepted. Used for incomplete ADF
        windows, whose leading p-values are intentionally undefined.

    Returns
    -------
    pandas.Series
        Copy of the series with a unique, increasing Python-date index.

    Raises
    ------
    StabilityAnalysisError
        If the series is empty, non-numeric, contains disallowed non-finite
        values, or has invalid, duplicate, or unsorted date labels.

    Notes
    -----
    Requiring strictly increasing unique dates ensures rolling windows are
    chronological and that joins with daily strategy results are unambiguous.
    """
    if series.empty:
        raise StabilityAnalysisError(f"{name} must not be empty")
    try:
        date_index = pd.Index(
            [pd.Timestamp(index_value).date() for index_value in series.index],
            name=series.index.name or "Date",
        )
        values = series.to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise StabilityAnalysisError(
            f"{name} must have valid date labels and numeric values"
        ) from error
    if date_index.has_duplicates:
        raise StabilityAnalysisError(f"{name} must not contain duplicate dates")
    if not date_index.is_monotonic_increasing:
        raise StabilityAnalysisError(f"{name} dates must be in increasing order")
    invalid_values = np.isinf(values).any() or (
        not allow_nan and np.isnan(values).any()
    )
    if invalid_values:
        raise StabilityAnalysisError(f"{name} must not contain missing or non-finite values")

    normalized = pd.Series(values, index=date_index, name=series.name, dtype=float)
    assert len(normalized) == len(series)
    assert normalized.index.is_monotonic_increasing
    return normalized


def compute_rolling_adf(
    spread: pd.Series,
    window: int = 90,
    as_of: date | None = None,
) -> pd.Series:
    """Calculate trailing-window Augmented Dickey-Fuller p-values.

    Parameters
    ----------
    spread : pandas.Series
        Fixed-parameter residual spread, indexed by observation date.
    window : int, default=90
        Number of consecutive observations in each ADF test window. Must be
        between 20 and the number of available observations.
    as_of : datetime.date or None, default=None
        Optional inclusive date cutoff. When supplied, later spread
        observations are rejected; otherwise the final input date is the cutoff.

    Returns
    -------
    pandas.Series
        ADF p-values indexed by the end date of each trailing window. The first
        ``window - 1`` observations are ``NaN`` because no complete test window
        is available; a failed individual ADF fit is also ``NaN``.

    Raises
    ------
    StabilityAnalysisError
        If the spread or window is invalid, the cutoff is exceeded, or all
        complete windows fail to produce a finite ADF p-value.

    Notes
    -----
    The ADF null hypothesis is a unit root (non-stationarity); p-values above
    the chosen significance level are therefore weak evidence against a
    stationary residual, not proof of a structural break. Each p-value uses
    observations through its own index date only, with a constant term and
    AIC-selected lag order. Constant or otherwise degenerate windows yield
    ``NaN`` instead of an invented test result.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 20:
        raise StabilityAnalysisError("window must be an integer greater than or equal to 20")

    normalized = _as_date_index(spread, "spread")
    cutoff = as_of if as_of is not None else normalized.index[-1]
    if any(observation_date > cutoff for observation_date in normalized.index):
        raise StabilityAnalysisError("spread contains observations after as_of")
    if window > len(normalized):
        raise StabilityAnalysisError("window must not exceed the number of spread observations")

    p_values = pd.Series(np.nan, index=normalized.index, name="adf_pvalue", dtype=float)
    failed_windows = 0
    for end_position in range(window - 1, len(normalized)):
        window_values = normalized.iloc[end_position - window + 1 : end_position + 1]
        try:
            result = adfuller(
                window_values.to_numpy(dtype=float),
                regression="c",
                autolag="AIC",
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            failed_windows += 1
            logger.debug("ADF test failed at %s: %s", normalized.index[end_position], error)
            continue

        p_value = float(result[1])
        if np.isfinite(p_value):
            p_values.iloc[end_position] = p_value
        else:
            failed_windows += 1

    valid_windows = int(p_values.notna().sum())
    if valid_windows == 0:
        raise StabilityAnalysisError("no rolling ADF window produced a valid p-value")

    assert len(p_values) == len(normalized)
    assert p_values.index.equals(normalized.index)
    assert all(observation_date <= cutoff for observation_date in p_values.index)
    assert p_values.dropna().between(0.0, 1.0).all()
    logger.info(
        "Rolling ADF complete: %d valid windows, %d failed windows, window=%d",
        valid_windows,
        failed_windows,
        window,
    )
    return p_values


def compare_stability_with_performance(
    adf_p_values: pd.Series,
    daily_pnl: pd.Series,
    significance_level: float = 0.05,
    as_of: date | None = None,
) -> StabilityPerformanceComparison:
    """Compare ADF stability labels with date-aligned strategy PnL and drawdown.

    Parameters
    ----------
    adf_p_values : pandas.Series
        Rolling ADF p-values, indexed by each test window's end date.
    daily_pnl : pandas.Series
        Realized daily strategy PnL indexed by its earning date.
    significance_level : float, default=0.05
        ADF cutoff: observations with p-value above it are labeled unstable.
        Must be strictly between zero and one.
    as_of : datetime.date or None, default=None
        Optional inclusive cutoff applied to both input series.

    Returns
    -------
    StabilityPerformanceComparison
        Counts, summed and average daily PnL, and worst observed drawdown for
        stable and unstable matched dates.

    Raises
    ------
    StabilityAnalysisError
        If inputs are invalid, dates exceed ``as_of``, or no finite ADF values
        overlap with daily PnL.

    Notes
    -----
    The calculation is retrospective: it aligns the ADF p-value timestamp with
    PnL earned on that date and never uses the labels to alter trades. Dates
    without a valid complete-window ADF test are excluded. Drawdown is computed
    from cumulative PnL over the entire matched timeline, with initial equity
    set to zero, and its troughs are grouped by their contemporaneous label.
    """
    if not np.isfinite(significance_level) or not 0.0 < significance_level < 1.0:
        raise StabilityAnalysisError("significance_level must be strictly between 0 and 1")

    p_values = _as_date_index(adf_p_values, "adf_p_values", allow_nan=True)
    pnl = _as_date_index(daily_pnl, "daily_pnl")
    cutoff = as_of if as_of is not None else max(p_values.index[-1], pnl.index[-1])
    if any(observation_date > cutoff for observation_date in p_values.index):
        raise StabilityAnalysisError("adf_p_values contains observations after as_of")
    if any(observation_date > cutoff for observation_date in pnl.index):
        raise StabilityAnalysisError("daily_pnl contains observations after as_of")
    if (p_values.dropna() < 0.0).any() or (p_values.dropna() > 1.0).any():
        raise StabilityAnalysisError("finite ADF p-values must be between 0 and 1")

    aligned = pd.concat(
        [p_values.rename("p_value"), pnl.rename("daily_pnl")],
        axis=1,
        join="inner",
    ).dropna(subset=["p_value"])
    if aligned.empty:
        raise StabilityAnalysisError("ADF p-values and daily PnL have no valid overlapping dates")

    unstable_mask = aligned["p_value"] > significance_level
    stable_mask = ~unstable_mask
    equity = aligned["daily_pnl"].cumsum()
    running_peak = equity.cummax().clip(lower=0.0)
    drawdown = equity - running_peak

    def group_metrics(mask: pd.Series) -> tuple[int, float | None, float | None, float | None]:
        """Calculate PnL and observed drawdown for one stability label."""
        observations = int(mask.sum())
        if observations == 0:
            return 0, None, None, None
        group_pnl = aligned.loc[mask, "daily_pnl"]
        group_drawdown = drawdown.loc[mask]
        return (
            observations,
            float(group_pnl.sum()),
            float(group_pnl.mean()),
            float(group_drawdown.min()),
        )

    stable = group_metrics(stable_mask)
    unstable = group_metrics(unstable_mask)
    assert stable[0] + unstable[0] == len(aligned)
    assert aligned.index.is_monotonic_increasing
    assert all(observation_date <= cutoff for observation_date in aligned.index)
    comparison = StabilityPerformanceComparison(
        significance_level=float(significance_level),
        stable_observations=stable[0],
        unstable_observations=unstable[0],
        stable_pnl=stable[1],
        unstable_pnl=unstable[1],
        stable_mean_daily_pnl=stable[2],
        unstable_mean_daily_pnl=unstable[2],
        stable_max_drawdown=stable[3],
        unstable_max_drawdown=unstable[3],
    )
    logger.info(
        "Stability/performance comparison at alpha=%.3f: stable[n=%d pnl=%s max_drawdown=%s] "
        "unstable[n=%d pnl=%s max_drawdown=%s]",
        significance_level,
        comparison.stable_observations,
        comparison.stable_pnl,
        comparison.stable_max_drawdown,
        comparison.unstable_observations,
        comparison.unstable_pnl,
        comparison.unstable_max_drawdown,
    )
    return comparison